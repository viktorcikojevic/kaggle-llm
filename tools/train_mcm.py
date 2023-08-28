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
    train_df.to_csv(model_output_dir / "train_df.csv")
    val_df.to_csv(model_output_dir / "val_df.csv")
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
    logger.info(f"model.num_parameters() = {model.num_parameters() * 1e-6} Million")
    logger.info(f"model.num_parameters() = {model.num_parameters() * 1e-9} Billion")
    logger.info("initted models")
 
    logger.info("initting dataset")
    train_tokenized_dataset = get_tokenize_dataset_from_df(train_df, tokenizer)
    val_tokenized_dataset = get_tokenize_dataset_from_df(val_df, tokenizer)
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
        load_best_model_at_end=True,
        evaluation_strategy="epoch",
        save_strategy="epoch",
        per_device_eval_batch_size=2,
        num_train_epochs=total_epochs,
        save_total_limit=config["save_total_limit"] if "save_total_limit" in config else 10,
        report_to=config["report_to"],
        output_dir=str(model_output_dir),
        # fp16=True,
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
        callbacks=[
            EarlyStoppingCallback(early_stopping_patience=config["early_stopping_patience"]),
        ],
    )
    logger.info("initting trainer")

    trainer.train()
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
