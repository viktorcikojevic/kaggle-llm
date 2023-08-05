from kaggle_llm.adapted_models import *
from kaggle_llm.core import (
    DataCollatorForMultipleChoice,
    WORK_DIRS_PATH,
    ROOT_PATH,
    compute_metrics,
    load_train_and_val_df,
    get_tokenize_dataset_from_df,
)
from transformers import (
    AutoModelForMultipleChoice,
    TrainingArguments,
    Trainer,
    AutoTokenizer,
    EarlyStoppingCallback,
)
from peft import get_peft_model, LoraConfig, prepare_model_for_kbit_training, AdaLoraConfig
from loguru import logger
from datetime import datetime
import argparse
import json
import yaml
import sys


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
    train_df, val_df = load_train_and_val_df(
        input_paths=input_paths,
        i_fold=config["fold"]["num"],
        total_fold=config["fold"]["of"],
    )
    model_name = load_from.split("/")[-1]
    model_output_dir = WORK_DIRS_PATH / f"peft-{model_name}-{datetime.now().strftime('%Y-%m-%d-%H-%M-%S')}"
    model_output_dir.mkdir(exist_ok=True, parents=True)
    # train_df.to_csv(model_output_dir / "train_df.csv")
    # val_df.to_csv(model_output_dir / "val_df.csv")
    logger.info(f"splitted dataset of size {len(train_df) + len(val_df)} -> {len(train_df)} & {len(val_df)}")
    logger.info("loaded data")

    logger.info("initting models")
    tokenizer = AutoTokenizer.from_pretrained(load_from)
    model = AutoModelForMultipleChoice.from_pretrained(load_from, load_in_8bit=True, device_map="auto")
    count_conversion_ratio(model, True)
    model = prepare_model_for_kbit_training(model)
    # https://github.com/huggingface/peft/blob/main/examples/int8_training/peft_adalora_whisper_large_training.py
    r = 8
    lora_alpha = r * 2
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
    #     # target_modules=["k_proj", "q_proj", "v_proj", "out_proj", "fc1", "fc2"],
    #     orth_reg_weight=0.5,
    # )
    peft_config = LoraConfig(
        inference_mode=False,
        r=r,
        # target_modules=["q_proj", "v_proj", "classifier", "pooler", "dropout"],
        # target_modules=["q_proj", "v_proj", "classifier"],
        # target_modules=["q_proj", "v_proj"],
        lora_alpha=lora_alpha,
        lora_dropout=0.1,
    )
    model = get_peft_model(model, peft_config)
    logger.info(print_trainable_parameters(model))
    logger.info("initted models")

    logger.info("initting dataset")
    train_tokenized_dataset = get_tokenize_dataset_from_df(train_df, tokenizer)
    val_tokenized_dataset = get_tokenize_dataset_from_df(val_df, tokenizer)
    logger.info("initted dataset")

    logger.info("initting trainer")
    warmup_epochs = 1
    total_epochs = 20
    warmup_ratio = warmup_epochs / total_epochs
    training_args = TrainingArguments(
        metric_for_best_model="map3",
        greater_is_better=True,
        warmup_ratio=warmup_ratio,
        learning_rate=float(config["peft_lr"]),
        per_device_train_batch_size=1,
        load_best_model_at_end=True,
        evaluation_strategy="epoch",
        save_strategy="epoch",
        per_device_eval_batch_size=2,
        num_train_epochs=total_epochs,
        save_total_limit=2,
        report_to=config["report_to"],
        output_dir=str(model_output_dir),
        remove_unused_columns=False,  # HF infers the cols based on model's forward signature, and peft corrupts it
        label_names=["labels"],  # for peft
        # deepspeed=str((ROOT_PATH / "configs" / "deepspeed.json").resolve().absolute()),
        fp16=True,
        ddp_find_unused_parameters=False,
        # optim=transformers.training_args.OptimizerNames.ADAMW_BNB,
        # gradient_checkpointing=True,
        # gradient_accumulation_steps=4,
    )
    trainer = Trainer(
        model=model,
        args=training_args,
        tokenizer=tokenizer,
        data_collator=DataCollatorForMultipleChoice(tokenizer=tokenizer),
        train_dataset=train_tokenized_dataset,
        eval_dataset=val_tokenized_dataset,
        compute_metrics=compute_metrics,
        callbacks=[
            EarlyStoppingCallback(early_stopping_patience=4),
        ],
    )
    logger.info("initting trainer")

    try:
        trainer.train()
    except KeyboardInterrupt:
        print(f"training interrupted, moving on to save the model")
    finally:
        print(f"saving the model")
        checkpoint_output_dir = str(model_output_dir / "best_map3_peft")
        trainer.save_model(checkpoint_output_dir)
        if hasattr(model.base_model, "save_extra_modules"):
            model.base_model.save_extra_modules(checkpoint_output_dir)
        print(f"model saved")


if __name__ == "__main__":
    parser = argparse.ArgumentParser()
    parser.add_argument("config")
    _args, _ = parser.parse_known_args()
    main(_args.config)
