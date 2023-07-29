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
    model_output_dir = WORK_DIRS_PATH / f"{model_name}-{datetime.now().strftime('%Y-%m-%d-%H-%M-%S')}"
    model_output_dir.mkdir(exist_ok=True, parents=True)
    val_df.to_csv(model_output_dir / "val_df.csv")
    logger.info(f"splitted dataset of size {len(train_df) + len(val_df)} -> {len(train_df)} & {len(val_df)}")
    logger.info("loaded data")

    logger.info("initting models")
    tokenizer = AutoTokenizer.from_pretrained(load_from)
    # model = AutoModelForMultipleChoice.from_pretrained(load_from, load_in_8bit=True)
    model = AutoModelForMultipleChoice.from_pretrained(load_from)
    logger.info(f"model.num_parameters() = {model.num_parameters() * 1e-6} Million")
    logger.info(f"model.num_parameters() = {model.num_parameters() * 1e-9} Billion")
    logger.info("initted models")

    logger.info("initting dataset")
    train_tokenized_dataset = get_tokenize_dataset_from_df(train_df, tokenizer)
    val_tokenized_dataset = get_tokenize_dataset_from_df(val_df, tokenizer)
    logger.info("initted dataset")

    logger.info("initting trainer")
    training_args = TrainingArguments(
        metric_for_best_model="map3",
        greater_is_better=True,
        warmup_ratio=0.8,
        learning_rate=float(config["lr"]),
        per_device_train_batch_size=1,
        load_best_model_at_end=True,
        evaluation_strategy="epoch",
        save_strategy="epoch",
        per_device_eval_batch_size=2,
        num_train_epochs=20,
        save_total_limit=2,
        report_to=["wandb"],
        output_dir=str(model_output_dir),
        # fp16=True,
        # gradient_checkpointing=True,
        # gradient_accumulation_steps=4,
        # deepspeed=str((ROOT_PATH / "configs" / "deepspeed.json").resolve().absolute()),
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

    trainer.train()
    trainer.save_model(str(model_output_dir / "best_map3"))


if __name__ == "__main__":
    parser = argparse.ArgumentParser()
    parser.add_argument("config")
    _args, _ = parser.parse_known_args()
    main(_args.config)
