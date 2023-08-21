import pandas as pd

from kaggle_llm.adapted_models import *
from kaggle_llm.core import (
    DataCollatorForMultipleChoice,
    WORK_DIRS_PATH,
    ROOT_PATH,
    compute_map3_hf,
    load_train_and_val_df,
    get_tokenize_dataset_from_df,
    get_mcp_tokenize_dataset_from_df,
    WrappedPeftModel,
    train_and_save_best_model_on_error,
    multiple_choice_prompting_preprocess,
    DataCollatorForMultipleChoicePrompting,
)
from transformers import (
    AutoModelForMultipleChoice,
    AutoModelForCausalLM,
    TrainingArguments,
    Trainer,
    AutoTokenizer,
    DataCollatorForLanguageModeling,
    EarlyStoppingCallback,
    PreTrainedTokenizerBase,
)
from transformers.tokenization_utils_base import PaddingStrategy, PreTrainedTokenizerBase
from datasets import Dataset, load_dataset
from dataclasses import dataclass


def group_texts(examples: Dict[str, Any], block_size: int = 128):
    concatenated_examples = {k: sum(examples[k], []) for k in examples.keys()}
    total_length = len(concatenated_examples[list(examples.keys())[0]])
    if total_length >= block_size:
        total_length = (total_length // block_size) * block_size
    result = {
        k: [t[i: i + block_size] for i in range(0, total_length, block_size)]
        for k, t in concatenated_examples.items()
    }
    result["labels"] = result["input_ids"].copy()
    return result


def causal_lm_preprocess(examples: Dict[str, Any], tokenizer):
    return tokenizer(
        [" ".join(x) for x in examples["answers.text"]],
        padding=True,
        truncation=True,
    )


def main():
    input_paths = ["/home/clay/research/kaggle/kaggle_llm/data/kaggle-llm-science-exam/train.csv"]
    tokenizer = AutoTokenizer.from_pretrained("distilgpt2")
    tokenizer.pad_token = tokenizer.eos_token
    model = AutoModelForCausalLM.from_pretrained("distilgpt2")

    eli5 = load_dataset("eli5", split="train_asks[:5000]")
    eli5 = eli5.train_test_split(test_size=0.2)
    train_dataset = eli5["train"].flatten()
    val_dataset = eli5["test"].flatten()
    train_dataset = train_dataset.map(
        lambda examples: causal_lm_preprocess(examples, tokenizer),
        remove_columns=train_dataset.column_names,
        batched=True,
    )
    train_dataset = train_dataset.map(
        lambda examples: group_texts(examples, 128),
        batched=True,
    )
    val_dataset = val_dataset.map(
        lambda examples: causal_lm_preprocess(examples, tokenizer),
        remove_columns=val_dataset.column_names,
        batched=True,
    )
    val_dataset = val_dataset.map(
        lambda examples: group_texts(examples, 128),
        batched=True,
    )

    data_collator = DataCollatorForLanguageModeling(tokenizer, mlm=False)

    # train_batch_before_collation = [train_dataset[i] for i in range(5)]
    # train_batch = data_collator(train_batch_before_collation)
    print("hehe")

    training_args = TrainingArguments(
        output_dir="my_awesome_eli5_clm-model",
        evaluation_strategy="epoch",
        learning_rate=2e-5,
        weight_decay=0.01,
        # per_device_train_batch_size=1,
        # per_device_eval_batch_size=1,
        report_to=[],
    )

    trainer = Trainer(
        model=model,
        args=training_args,
        train_dataset=train_dataset,
        eval_dataset=val_dataset,
        data_collator=data_collator,
    )

    trainer.train()


if __name__ == "__main__":
    main()
