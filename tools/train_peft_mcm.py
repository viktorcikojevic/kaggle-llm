from kaggle_llm.adapted_models import *
from kaggle_llm.core import (
    DataCollatorForMultipleChoice,
    WORK_DIRS_PATH,
    ROOT_PATH,
    compute_metrics,
    load_train_and_val_df,
    get_tokenize_dataset_from_df,
)
from transformers import AutoModelForMultipleChoice, TrainingArguments, Trainer, AutoTokenizer, EarlyStoppingCallback
from peft import get_peft_model, LoraConfig, prepare_model_for_int8_training
from loguru import logger
from datetime import datetime
import argparse
import json
import yaml
import sys
import os


os.environ["MASTER_ADDR"] = "localhost"
os.environ["MASTER_PORT"] = "9994"  # modify if RuntimeError: Address already in use
os.environ["RANK"] = "0"
os.environ["LOCAL_RANK"] = "0"
os.environ["WORLD_SIZE"] = "1"


logger.add(sys.stdout, format="{time} {level} {message}", level="INFO")


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
    train_df.to_csv(model_output_dir / "train_df.csv")
    # val_df.to_csv(model_output_dir / "val_df.csv")
    logger.info(f"splitted dataset of size {len(train_df) + len(val_df)} -> {len(train_df)} & {len(val_df)}")
    logger.info("loaded data")

    logger.info("initting models")
    tokenizer = AutoTokenizer.from_pretrained(load_from)
    # model = AutoModelForMultipleChoice.from_pretrained(load_from, load_in_8bit=True)
    model = AutoModelForMultipleChoice.from_pretrained(load_from)
    r = 8
    lora_alpha = r * 2
    peft_config = LoraConfig(
        inference_mode=False,
        r=r,
        lora_alpha=lora_alpha,
        lora_dropout=0.1,
    )
    # model = prepare_model_for_int8_training(model)
    # model.enable_input_require_grads()
    model = get_peft_model(model, peft_config)
    # model = prepare_model_for_int8_training(model)
    logger.info(model.print_trainable_parameters())
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
        trainer.save_model(str(model_output_dir / "best_map3_peft"))
        trainer.model = trainer.model.merge_and_unload()
        trainer.save_model(str(model_output_dir / "best_map3"))
        print(f"model saved")


if __name__ == "__main__":
    parser = argparse.ArgumentParser()
    parser.add_argument("config")
    _args, _ = parser.parse_known_args()
    main(_args.config)
