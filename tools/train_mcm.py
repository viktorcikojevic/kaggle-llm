from kaggle_llm.adapted_models import (
    LlamaModelForMultipleChoice, 
    DebertaV2ForMultipleChoice2
)
from kaggle_llm.core import (
    DataCollatorForMultipleChoice,
    DataCollatorForMultipleChoicePrompting,
    WORK_DIRS_PATH,
    ROOT_PATH,
    compute_map3_hf,
    build_peft_model,
    load_train_and_val_df,
    get_tokenize_dataset_from_df,
    get_mcp_tokenize_dataset_from_df,
    train_and_save_best_model_on_error,
    add_context
)
from transformers import TrainingArguments, Trainer, EarlyStoppingCallback
from loguru import logger
from datetime import datetime
import argparse
import json
import yaml
import sys
import os
from pathlib import Path
import pandas as pd

logger.add(sys.stdout, format="{time} {level} {message}", level="INFO")
os.environ["TOKENIZERS_PARALLELISM"] = "false"

def main(config_path: str,
         work_dir_path: str = None,
         ):
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
    print("model_name:", model_name)
    if work_dir_path is None:
        work_dir_path = WORK_DIRS_PATH
    model_output_dir = os.path.join(work_dir_path, f"{model_name}-{datetime.now().strftime('%Y-%m-%d-%H-%M-%S')}")
    model_output_dir = Path(model_output_dir)
    model_output_dir.mkdir(exist_ok=True, parents=True)
    logger.info(f"splitted dataset of size {len(train_df) + len(val_df)} -> {len(train_df)} & {len(val_df)}")
    logger.info("loaded data")

    logger.info("initting models")
    
    
    model, tokenizer = build_peft_model(
        config["load_from"],
        use_peft=config["use_peft"],
        peft_class=config["peft_class"],
        transformer_class="AutoModelForMultipleChoice",
        use_8bit=config["use_8bit"],
        **config["peft_kwargs"] if "peft_kwargs" in config else {},
    )

    for param, param_name in zip(model.parameters(), model.state_dict().keys()):
        if param.requires_grad == True:
            print(f"param_name: {param_name}, requires_grad: {param.requires_grad}")
        param.requires_grad = True
    return
    
    
    if 'freeze_embeddings' in config and config['freeze_embeddings'] and 'deberta' in config['load_from']:
        print('Freezing embeddings.')
        for param in model.deberta.embeddings.parameters():
            param.requires_grad = False
    if 'freeze_layers' in config and 'deberta' in config['load_from']:
        freeze_layers = config['freeze_layers']
        print(f'Freezing first {freeze_layers} layers.')
        for layer in model.deberta.encoder.layer[:freeze_layers]:
            for param in layer.parameters():
                param.requires_grad = False
    
    logger.info(f"model.num_parameters() = {model.num_parameters() * 1e-6} Million")
    logger.info(f"model.num_parameters() = {model.num_parameters() * 1e-9} Billion")
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
    
    
    train_df.to_csv(model_output_dir / "train_df.csv")
    val_df.to_csv(model_output_dir / "val_df.csv")
            
    train_df = train_df.sample(n=100).reset_index(drop=True)        
    
    train_tokenized_dataset = get_tokenize_dataset_from_df(train_df, tokenizer, preprocess_type, max_input)
    val_tokenized_dataset = get_tokenize_dataset_from_df(val_df, tokenizer, preprocess_type, max_input)
    # train_tokenized_dataset = get_mcp_tokenize_dataset_from_df(train_df, tokenizer)
    # val_tokenized_dataset = get_mcp_tokenize_dataset_from_df(val_df, tokenizer)
    logger.info("initted dataset")
    

    logger.info("initting trainer")
    warmup_epochs = 1
    total_epochs = config["total_epochs"]
    warmup_ratio = warmup_epochs / total_epochs
    training_args = TrainingArguments(
        metric_for_best_model="map3",
        lr_scheduler_type="cosine",
        greater_is_better=True,
        warmup_ratio=warmup_ratio,
        learning_rate=float(config["lr"]),
        per_device_train_batch_size=1,
        load_best_model_at_end=False,
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
        # label_names=["label"],  # for peft
        fp16=False if 'use_8bit' in config and config['use_8bit'] else True,
        # gradient_checkpointing=True,
        gradient_accumulation_steps=config["gradient_accumulation_steps"],
        # deepspeed=str((ROOT_PATH / "configs" / "deepspeed.json").resolve().absolute()),
    )
    
    
    trainer = Trainer(
        model=model,
        args=training_args,
        tokenizer=tokenizer,
        data_collator=DataCollatorForMultipleChoice(tokenizer=tokenizer),
        # data_collator=DataCollatorForMultipleChoicePrompting(tokenizer=tokenizer),
        train_dataset=train_tokenized_dataset,
        eval_dataset=val_tokenized_dataset,
        compute_metrics=compute_map3_hf,
        # callbacks=[
        #     EarlyStoppingCallback(early_stopping_patience=config["early_stopping_patience"]),
        # ],
    )
    logger.info("initting trainer")

    trainer.train()
    # if "use_peft" in config and config["use_peft"]:
        # train_and_save_best_model_on_error(
        #     trainer,
        #     model_output_dir,
        #     "best_map3_peft" if config["use_peft"] else "best_map3",
        # )
    
    if config["report_to"] == "wandb":
        wandb.finish()
    


if __name__ == "__main__":
    parser = argparse.ArgumentParser()
    parser.add_argument("config")
    parser.add_argument("--work-dir-path", type=str, help="path to work dir", required=False)
    _args, _ = parser.parse_known_args()
    print(f"Using args: {_args}")
    print("--"*64)
    main(_args.config, _args.work_dir_path)
