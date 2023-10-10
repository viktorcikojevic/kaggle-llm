from kaggle_llm.adapted_models import *
from kaggle_llm.core import (
    DataCollatorForMultipleChoice,
    compute_map3_hf,
    load_train_and_val_df,
    get_tokenize_dataset_from_df,
    WrappedPeftModel,
    train_and_save_best_model_on_error,
)
from transformers import (
    AutoModelForMultipleChoice,
    TrainingArguments,
    Trainer,
    AutoTokenizer,
    EarlyStoppingCallback,
)
from peft import get_peft_model, LoraConfig, prepare_model_for_kbit_training, AdaLoraConfig, PeftModel
from loguru import logger
from datetime import datetime
import argparse
import json
import yaml
import sys
import os
import pandas as pd


logger.add(sys.stdout, format="{time} {level} {message}", level="INFO")


def print_trainable_parameters(model):
    trainable_params = 0
    all_param = 0
    for _, param in model.named_parameters():
        num_params = param.numel()
        # if using DS Zero 3 and the weights are initialized empty
        if num_params == 0 and hasattr(param, "ds_numel"):
            num_params = param.ds_numel

        all_param += num_params
        if param.requires_grad:
            trainable_params += num_params
    print(
        f"trainable params: {trainable_params:,d} || all params: {all_param:,d} || trainable%: {100 * trainable_params / all_param}"
    )


def count_conversion_ratio(model, print_convert_names: bool = True):
    f16_count = 0
    all_count = 0
    for n, p in model.named_parameters():
        all_count += 1
        if p.dtype == torch.float16:
            if print_convert_names:
                print(n)
            f16_count += 1
    conversion_ratio = f16_count / all_count
    print(f"{conversion_ratio = }")


def main(config_path: str):
    
    
    
    
    with open(config_path, "rb") as f:
        config = yaml.load(f, yaml.FullLoader)

    load_from = config["load_from"]
    input_paths = config["inputs"]
    logger.info(json.dumps(config, indent=4))
    logger.info("loading data")
    
    if "eval_on" in config.keys():
        
        
        
        # Load training data 
        train_df_1, val_df_1 = load_train_and_val_df(
            input_paths=config["inputs"],
            i_fold=config["fold"]["num"] if "fold" in config else 0,
            total_fold=config["fold"]["of"] if "fold" in config else 10,
        )
        train_df = pd.concat([train_df_1, val_df_1])
        
        
        # Load validation data 
        train_df_2, val_df_2 = load_train_and_val_df(
            input_paths=config["eval_on"],
            i_fold=config["fold"]["num"] if "fold" in config else 0,
            total_fold=config["fold"]["of"] if "fold" in config else 10,
        )
        if "eval_all_folds" in config and config["eval_all_folds"]:
            val_df = pd.concat([train_df_2, val_df_2])
        else:
            val_df = val_df_2
    else:
        train_df, val_df = load_train_and_val_df(
            input_paths=input_paths,
            i_fold=config["fold"]["num"],
            total_fold=config["fold"]["of"],
        )  
        
        
    print("train_df.dtypes", train_df.dtypes)
    
    print("train_df:")
    print(train_df.iloc[0])
    print("val_df:")
    print(val_df.iloc[0])
    if "add_context" in config and config["add_context"]:
        train_df = add_context(train_df)
        val_df = add_context(val_df)

        print(f"New train_df size: {len(train_df)}")
        print(f"New val_df size: {len(val_df)}")
    
        print(train_df.sample(1)['new_prompt'].values[0])
        
    print(f"[INFO] train df size is {len(train_df)}")
    print(f"[INFO] val df size is {len(val_df)}")
    
    if "train_size" in config:
        train_df = train_df.sample(config["train_size"], replace=False).reset_index(drop=True)
        print(f"[INFO] Resampled df. New train df size is {len(train_df)}")

    
    
    model_name = load_from.split("/")[-1]
    model_output_dir = f"peft-{model_name}-{datetime.now().strftime('%Y-%m-%d-%H-%M-%S')}"
    if os.path.exists(model_output_dir) == False:
        os.makedirs(model_output_dir)
    
    # train_df.to_csv(model_output_dir / "train_df.csv")
    # val_df.to_csv(model_output_dir / "val_df.csv")
    logger.info(f"splitted dataset of size {len(train_df) + len(val_df)} -> {len(train_df)} & {len(val_df)}")
    logger.info("loaded data")

    logger.info("initting models")
    tokenizer = AutoTokenizer.from_pretrained(load_from)
    if config["use_8bit"]:
        model = AutoModelForMultipleChoice.from_pretrained(load_from, load_in_8bit=True)
        count_conversion_ratio(model, True)
        model = prepare_model_for_kbit_training(model)
    else:
        model = AutoModelForMultipleChoice.from_pretrained(load_from)
    logger.info(f"{model.__class__.__name__ = }")
    # https://github.com/huggingface/peft/blob/main/examples/int8_training/peft_adalora_whisper_large_training.py
    # peft_config = AdaLoraConfig(
    #     init_r=12,
    #     target_r=4,
    #     beta1=0.85,
    #     beta2=0.85,
    #     tinit=200,
    #     tfinal=1000,
    #     deltaT=10,
    #     lora_alpha=lora_alpha,
    #     lora_dropout=0.1,
    #     target_modules=["q_proj", "v_proj"],
    #     orth_reg_weight=0.5,
    # )
    logger.info(json.dumps(config["peft_kwargs"]))
    peft_config = LoraConfig(
        inference_mode=False,
        **config["peft_kwargs"],
    )
    model = WrappedPeftModel(model, peft_config)
    # model = get_peft_model(model, peft_config)
    logger.info(print_trainable_parameters(model))
    logger.info("initted models")

    logger.info("initting dataset")
    
    if 'separate_prompt_and_context' in config and config['separate_prompt_and_context']:
        def get_context(text):
            x = text.split(" ### ")
            # remove empty strings
            x = [x for x in x if len(x) > 0]
            
            assert len(x) == 2, f"Unsuccesful prompt splitting . len(x) = {len(x)}, x={x}"
            return x[0]
        
        
        train_df['context'] = train_df['prompt'].apply(lambda x: get_context(x))
        val_df['context'] = val_df['prompt'].apply(lambda x: get_context(x))
        
        def get_prompt(text):
            x = text.split(" ### ")
            # remove empty strings
            x = [x for x in x if len(x) > 0]
            assert len(x) == 2, f"Unsuccesful prompt splitting . len(x) = {len(x)}"
            return x[1]
        
        train_df['prompt'] = train_df['prompt'].apply(lambda x: get_prompt(x))
        val_df['prompt'] = val_df['prompt'].apply(lambda x: get_prompt(x))
        
    
    if "max_context_size" in config:
        max_context_size = config["max_context_size"]
        def limit_context(x, max_context_size):
            x = x[:max_context_size]
            return x
        train_df['context'] = train_df['context'].apply(lambda x: limit_context(x, max_context_size))
        val_df['context'] = val_df['context'].apply(lambda x: limit_context(x, max_context_size))
    
    preprocess_type = 'sumo' if 'preprocess_type' not in config else config['preprocess_type']
    print(f"preprocess_type: {preprocess_type}")
    max_input = 512 if 'max_input' not in config else config['max_input']
    
    # Trim context length to reasonable size
    train_df['context'] = train_df['context'].apply(lambda x: x[:12000])
    val_df['context'] = val_df['context'].apply(lambda x: x[:12000])
    
    train_df['context_len'] = train_df['context'].apply(lambda x: len(x))
    train_df['prompt_len'] = train_df['prompt'].apply(lambda x: len(x))
    
    val_df['context_len'] = val_df['context'].apply(lambda x: len(x))
    val_df['prompt_len'] = val_df['prompt'].apply(lambda x: len(x))
    
    
    
    print("train_df['context_len'].min()", train_df['context_len'].min())
    print("train_df['context_len'].max()", train_df['context_len'].max())
    print("val_df['context_len'].min()", val_df['context_len'].min())
    print("val_df['context_len'].max()", val_df['context_len'].max())
    print("train_df['prompt_len'].min()", train_df['prompt_len'].min())
    print("train_df['prompt_len'].max()", train_df['prompt_len'].max())
    print("val_df['prompt_len'].min()", val_df['prompt_len'].min())
    print("val_df['prompt_len'].max()", val_df['prompt_len'].max())
    
    # print unique answers
    print("train_df['answer'].unique()", train_df['answer'].unique())
    print("val_df['answer'].unique()", val_df['answer'].unique())
    
    # take only prompt, context, A, B, C, D, E and answer (if there's an answer)
    train_df = train_df[['prompt', 'context', 'A', 'B', 'C', 'D', 'E', 'answer']]
    val_df = val_df[['prompt', 'context', 'A', 'B', 'C', 'D', 'E', 'answer']]
    
    train_df['prompt'] = train_df['prompt'].astype(str)
    val_df['prompt'] = val_df['prompt'].astype(str)
    train_df['context'] = train_df['context'].astype(str)
    val_df['context'] = val_df['context'].astype(str)
    train_df['answer'] = train_df['answer'].astype(str)
    val_df['answer'] = val_df['answer'].astype(str)

    options = 'ABCDE'
    for option in options:
        train_df[option] = train_df[option].astype(str)
        val_df[option] = val_df[option].astype(str)
    
    
            
    # train_df = train_df.sample(n=20, random_state=42).reset_index(drop=True)        
    # val_df = train_df.copy()
    
    train_tokenized_dataset = get_tokenize_dataset_from_df(train_df, tokenizer, preprocess_type, max_input)
    val_tokenized_dataset = get_tokenize_dataset_from_df(val_df, tokenizer, preprocess_type, max_input)
    logger.info("initted dataset")

    logger.info("initting trainer")
    warmup_epochs = config["warmup_epochs"]
    total_epochs = config["total_epochs"]
    warmup_ratio = warmup_epochs / total_epochs
    training_args = TrainingArguments(
        metric_for_best_model="map3",
        greater_is_better=True,
        warmup_ratio=warmup_ratio,
        learning_rate=float(config["lr"]),
        per_device_train_batch_size=1,
        load_best_model_at_end=True,
        evaluation_strategy="steps",
        save_strategy="steps",
        eval_steps=50,
        save_steps=50,
        logging_steps=50,
        per_device_eval_batch_size=2,
        num_train_epochs=total_epochs,
        save_total_limit=config["save_total_limit"] if "save_total_limit" in config else 10,
        report_to=config["report_to"],
        output_dir=str(model_output_dir),
        remove_unused_columns=False,  # HF infers the cols based on model's forward signature, and peft corrupts it
        label_names=["labels"],  # for peft
        # deepspeed=str((ROOT_PATH / "configs" / "deepspeed.json").resolve().absolute()),
        fp16=True,
        ddp_find_unused_parameters=False,
        # optim=transformers.training_args.OptimizerNames.ADAMW_BNB,
        # gradient_checkpointing=True,
        gradient_accumulation_steps=config["gradient_accumulation_steps"]
    )
    trainer = Trainer(
        model=model,
        args=training_args,
        tokenizer=tokenizer,
        data_collator=DataCollatorForMultipleChoice(tokenizer=tokenizer),
        train_dataset=train_tokenized_dataset,
        eval_dataset=val_tokenized_dataset,
        compute_metrics=compute_map3_hf,
        # callbacks=[
        #     EarlyStoppingCallback(early_stopping_patience=4),
        # ],
    )
    logger.info("initting trainer")
    # trainer.train()

    train_and_save_best_model_on_error(trainer, model_output_dir, "best_map3_peft")


if __name__ == "__main__":
    parser = argparse.ArgumentParser()
    parser.add_argument("config")
    _args, _ = parser.parse_known_args()
    main(_args.config)
