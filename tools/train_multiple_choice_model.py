from kaggle_llm.core import multiple_choice_preprocess, DataCollatorForMultipleChoice, WORK_DIRS_PATH, get_map3, infer_pred_from_scores
from transformers import AutoModelForMultipleChoice, TrainingArguments, Trainer, AutoTokenizer, EvalPrediction, EarlyStoppingCallback
from datasets import Dataset
from loguru import logger
from datetime import datetime
from sklearn.model_selection import KFold
from typing import *
import pandas as pd
import argparse
import json
import yaml
import sys


logger.add(sys.stdout, format="{time} {level} {message}", level="INFO")


def compute_metrics(preds: EvalPrediction) -> Dict:
    ids = list(range(len(preds.label_ids)))
    preds_df = pd.DataFrame({
        "id": ids,
        **{f"score{k}": preds.predictions[:, k] for k in range(5)}
    }).set_index("id")
    preds_df = infer_pred_from_scores(preds_df)
    label_df = pd.DataFrame({
        "id": ids,
        "answer": ["ABCDE"[label] for label in preds.label_ids]
    })
    map3 = get_map3(label_df=label_df, pred_df=preds_df)
    return {"map3": map3}


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
    kf = KFold(n_splits=config["fold"]["of"], shuffle=True, random_state=42)
    train_idx, val_idx = list(kf.split(df))[config["fold"]["num"]]
    train_dataset = Dataset.from_pandas(df.loc[train_idx, :])
    val_dataset = Dataset.from_pandas(df.loc[val_idx, :])
    train_tokenized_dataset = train_dataset.map(
        lambda example: multiple_choice_preprocess(tokenizer, example),
        remove_columns=["prompt", "A", "B", "C", "D", "E", "answer"]
    )
    val_tokenized_dataset = val_dataset.map(
        lambda example: multiple_choice_preprocess(tokenizer, example),
        remove_columns=["prompt", "A", "B", "C", "D", "E", "answer"]
    )
    logger.info(f"splitted dataset of size {len(df)} -> {len(train_idx)} & {len(val_idx)}")
    logger.info("initted dataset")

    logger.info("initting trainer")
    model_name = load_from.split("/")[-1]
    training_args = TrainingArguments(
        metric_for_best_model="map3",
        greater_is_better=True,
        warmup_ratio=0.8,
        learning_rate=5e-6,
        per_device_train_batch_size=1,
        load_best_model_at_end=True,
        evaluation_strategy="epoch",
        save_strategy="epoch",
        per_device_eval_batch_size=2,
        num_train_epochs=3,
        save_total_limit=3,
        report_to="none",
        output_dir=str(WORK_DIRS_PATH / f"{model_name}-{datetime.now().strftime('%Y-%m-%d-%H-%M-%S')}"),
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
            EarlyStoppingCallback(early_stopping_patience=2),
        ]
    )
    logger.info("initting trainer")

    trainer.train()


if __name__ == "__main__":
    parser = argparse.ArgumentParser()
    parser.add_argument("config")
    _args, _ = parser.parse_known_args()
    main(_args.config)
