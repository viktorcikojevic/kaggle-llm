from kaggle_llm.adapted_models import *
from kaggle_llm.core import (
    DataCollatorForMultipleChoice,
    DataCollatorForMultipleChoicePrompting,
    WORK_DIRS_PATH,
    load_train_and_val_df,
    get_tokenize_dataset_from_df,
    get_mcp_tokenize_dataset_from_df,
    train_and_save_best_model_on_error,
    build_trainer,
    build_peft_model,
)
from loguru import logger
from datetime import datetime
import argparse
import json
import yaml
import sys


logger.add(sys.stdout, format="{time} {level} {message}", level="INFO")


def main(config_path: str):
    with open(config_path, "rb") as f:
        config = yaml.load(f, yaml.FullLoader)
    logger.info(json.dumps(config, indent=4))

    logger.info("loading data")
    train_df, val_df = load_train_and_val_df(
        input_paths=config["inputs"],
        i_fold=config["fold"]["num"],
        total_fold=config["fold"]["of"],
    )
    model_name = config["load_from"].split("/")[-1]
    model_output_dir = WORK_DIRS_PATH / f"peft-{model_name}-{datetime.now().strftime('%Y-%m-%d-%H-%M-%S')}"
    model_output_dir.mkdir(exist_ok=True, parents=True)
    # train_df.to_csv(model_output_dir / "train_df.csv")
    # val_df.to_csv(model_output_dir / "val_df.csv")
    logger.info(f"splitted dataset of size {len(train_df) + len(val_df)} -> {len(train_df)} & {len(val_df)}")
    logger.info("loaded data")

    logger.info("initting models")
    model, tokenizer = build_peft_model(
        config["load_from"],
        config["peft_class"],
        transformer_class="AutoModelForMultipleChoice",
        use_8bit=config["use_8bit"],
        **config["peft_kwargs"]
    )
    logger.info("initted models")

    logger.info("initting dataset")
    # train_tokenized_dataset = get_tokenize_dataset_from_df(train_df, tokenizer)
    # val_tokenized_dataset = get_tokenize_dataset_from_df(val_df, tokenizer)
    train_tokenized_dataset = get_mcp_tokenize_dataset_from_df(train_df, tokenizer)
    val_tokenized_dataset = get_mcp_tokenize_dataset_from_df(val_df, tokenizer)
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
        # data_collator=DataCollatorForMultipleChoice(tokenizer=tokenizer),
        data_collator=DataCollatorForMultipleChoicePrompting(tokenizer=tokenizer),
    )
    logger.info("initting trainer")

    # trainer.train()
    train_and_save_best_model_on_error(trainer, model_output_dir, "best_map3_peft")


if __name__ == "__main__":
    parser = argparse.ArgumentParser()
    parser.add_argument("config")
    _args, _ = parser.parse_known_args()
    main(_args.config)
