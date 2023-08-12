import pandas as pd

from kaggle_llm.adapted_models import *
from kaggle_llm.core import (
    DataCollatorForMultipleChoice,
    WORK_DIRS_PATH,
    ROOT_PATH,
    compute_metrics,
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
    TrainingArguments,
    Trainer,
    AutoTokenizer,
    EarlyStoppingCallback,
)
from datasets import Dataset


def main():
    input_paths = ["/home/clay/research/kaggle/kaggle_llm/data/kaggle-llm-science-exam/train.csv"]
    tokenizer = AutoTokenizer.from_pretrained("microsoft/deberta-v3-large")
    train_df, val_df = load_train_and_val_df(
        input_paths=input_paths,
        i_fold=0,
        total_fold=10,
    )
    # train_dataset = get_tokenize_dataset_from_df(train_df, tokenizer)
    train_dataset = get_mcp_tokenize_dataset_from_df(train_df, tokenizer)

    # collator = DataCollatorForMultipleChoice(tokenizer)
    collator = DataCollatorForMultipleChoicePrompting(tokenizer)

    train_batch_before_collation = [train_dataset[i] for i in range(5)]
    train_batch = collator(train_batch_before_collation)
    print("hehe")


if __name__ == "__main__":
    main()
