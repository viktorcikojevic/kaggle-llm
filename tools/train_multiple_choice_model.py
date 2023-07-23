from src.kaggle_llm.core import multiple_choice_preprocess, DataCollatorForMultipleChoice, WORK_DIRS_PATH
from transformers import AutoModelForMultipleChoice, TrainingArguments, Trainer, AutoTokenizer
from datasets import Dataset
from loguru import logger
from datetime import datetime
import pandas as pd
import argparse
import json
import yaml
import sys


logger.add(sys.stdout, format="{time} {level} {message}", level="INFO")


def main(config_path: str):
    with open(config_path, "rb") as f:
        config = yaml.load(f, yaml.FullLoader)

    load_from = config["load_from"]
    input_paths = config["inputs"]
    logger.info(json.dumps(config, indent=4))

    logger.info("loading data")
    dfs = [pd.read_csv(ip) for ip in input_paths]
    df = pd.concat(dfs, axis=0)
    logger.info("loaded data")

    logger.info("initting models")
    tokenizer = AutoTokenizer.from_pretrained(load_from)
    model = AutoModelForMultipleChoice.from_pretrained(load_from)
    logger.info("initted models")

    logger.info("initting dataset")
    dataset = Dataset.from_pandas(df)
    tokenized_dataset = dataset.map(
        lambda example: multiple_choice_preprocess(tokenizer, example),
        remove_columns=["prompt", "A", "B", "C", "D", "E", "answer"]
    )
    logger.info("initted dataset")

    logger.info("initting trainer")
    model_name = load_from.split("/")[-1]
    training_args = TrainingArguments(
        warmup_ratio=0.8,
        learning_rate=5e-6,
        per_device_train_batch_size=1,
        per_device_eval_batch_size=2,
        num_train_epochs=3,
        report_to="none",
        output_dir=str(WORK_DIRS_PATH / f"{model_name}-{datetime.now().strftime('%Y-%m-%d-%H-%M-%S')}"),
    )
    trainer = Trainer(
        model=model,
        args=training_args,
        tokenizer=tokenizer,
        data_collator=DataCollatorForMultipleChoice(tokenizer=tokenizer),
        train_dataset=tokenized_dataset,
    )
    logger.info("initting trainer")

    trainer.train()


if __name__ == "__main__":
    parser = argparse.ArgumentParser()
    parser.add_argument("config")
    _args, _ = parser.parse_known_args()
    main(_args.config)
