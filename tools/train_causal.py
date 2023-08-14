from kaggle_llm.adapted_models import *
from kaggle_llm.core import (
    DataCollatorForMultipleChoice,
    DataCollatorForMultipleChoicePrompting,
    WORK_DIRS_PATH,
    load_causal_lm_dataset,
    get_tokenize_dataset_from_df,
    get_mcp_tokenize_dataset_from_df,
    train_and_save_best_model_on_error,
    build_trainer,
    build_peft_model,
)
from transformers import DataCollatorForLanguageModeling
from loguru import logger
from datetime import datetime
from datasets import Dataset
import pandas as pd
import argparse
import json
import yaml
import sys


def causal_lm_preprocess(examples: Dict[str, Any], tokenizer):
    tokens = tokenizer(
        examples["sentences"],
        # padding=True,
        # truncation=True,
    )
    return tokens


def tokens_not_too_long(example: Dict[str, Any], cutoff_len: int = 150):
    return len(example["input_ids"]) < cutoff_len


def get_causal_lm_dataset(df: pd.DataFrame, tokenizer, cutoff_len: int = 150):
    dataset = Dataset.from_pandas(df)
    dataset = dataset.map(
        lambda examples: causal_lm_preprocess(examples, tokenizer),
        remove_columns=dataset.column_names,
        # num_proc=4,
        # batched=True,
    )
    before_filter_len = len(dataset)
    dataset = dataset.filter(
        lambda example: tokens_not_too_long(example, cutoff_len=cutoff_len)
    )
    print(f"filtered dataset: {len(dataset)/before_filter_len*100:.2f}% remaining")
    return dataset


logger.add(sys.stdout, format="{time} {level} {message}", level="INFO")


def main(config_path: str):
    with open(config_path, "rb") as f:
        config = yaml.load(f, yaml.FullLoader)
    logger.info(json.dumps(config, indent=4))

    logger.info("loading data")
    df = load_causal_lm_dataset(
        input_paths=config["inputs"],
    )
    model_name = config["load_from"].split("/")[-1]
    model_output_dir = WORK_DIRS_PATH / f"clm-peft-{model_name}-{datetime.now().strftime('%Y-%m-%d-%H-%M-%S')}"
    model_output_dir.mkdir(exist_ok=True, parents=True)
    # logger.info(f"splitted dataset of size {len(train_df) + len(val_df)} -> {len(train_df)} & {len(val_df)}")
    logger.info("loaded data")

    logger.info("initting models")
    model, tokenizer = build_peft_model(
        config["load_from"],
        use_peft=config["use_peft"],
        peft_class=config["peft_class"],
        transformer_class=config["transformer_class"],
        use_8bit=config["use_8bit"],
        **config["peft_kwargs"]
    )
    logger.info("initted models")

    logger.info("initting dataset")
    tokenized_dataset = get_causal_lm_dataset(df, tokenizer=tokenizer, cutoff_len=config["cutoff_len"])
    tokenized_dataset = tokenized_dataset.train_test_split(test_size=0.05)
    train_tokenized_dataset = tokenized_dataset["train"]
    val_tokenized_dataset = tokenized_dataset["test"]
    logger.info("initted dataset")

    logger.info("initting trainer")
    trainer = build_trainer(
        model=model,
        tokenizer=tokenizer,
        learning_rate=float(config["peft_lr"]),
        report_to=config["report_to"],
        output_dir=str(model_output_dir),
        train_dataset=train_tokenized_dataset,
        eval_dataset=val_tokenized_dataset,
        data_collator=(
            DataCollatorForLanguageModeling(tokenizer, mlm=True, mlm_probability=0.15)
            if "mask" in config["transformer_class"].lower() else
            DataCollatorForLanguageModeling(tokenizer, mlm=False)
        ),
        metric_for_best_model="eval_loss",
        greater_is_better=False,
        per_device_train_batch_size=config["batch_size"],
        per_device_eval_batch_size=2,
        compute_metrics=None,
    )
    logger.info("initting trainer")

    trainer.train()
    # train_and_save_best_model_on_error(trainer, model_output_dir, "best_eval_loss_peft")


if __name__ == "__main__":
    parser = argparse.ArgumentParser()
    parser.add_argument("config")
    _args, _ = parser.parse_known_args()
    main(_args.config)
