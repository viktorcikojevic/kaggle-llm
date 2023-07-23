from typing import *
import numpy as np
import torch
from datasets import Dataset
from transformers.tokenization_utils_base import PaddingStrategy, PreTrainedTokenizerBase
from transformers import AutoModelForMultipleChoice, TrainingArguments, Trainer
from transformers import AutoTokenizer
from dataclasses import dataclass
import pandas as pd

options = ["A", "B", "C", "D", "E"]
option_to_index = {v: i for i, v in enumerate(options)}


def multiple_choice_preprocess(tokenizer: PreTrainedTokenizerBase, example: Dict[str, Any]):
    """
    The example is expected to be a dictionary with keys "prompt", "A", "B", "C", "D", "E", and "answer".
    """
    # The AutoModelForMultipleChoice class expects a set of question/answer pairs,
    # so we"ll copy our question 5 times before tokenizing
    first_sentence = [example["prompt"]] * 5
    second_sentence = [example[option] for option in options]
    # Our tokenizer will turn our text into token IDs BERT can understand
    tokenized_example = tokenizer(first_sentence, second_sentence, truncation=True)
    tokenized_example["label"] = option_to_index[example["answer"]]
    return tokenized_example


@dataclass
class DataCollatorForMultipleChoice:
    tokenizer: PreTrainedTokenizerBase
    padding: Union[bool, str, PaddingStrategy] = True
    max_length: Optional[int] = None
    pad_to_multiple_of: Optional[int] = None

    def __call__(self, features):
        label_name = "label" if "label" in features[0].keys() else "labels"
        labels = [feature.pop(label_name) for feature in features]
        batch_size = len(features)
        num_choices = len(features[0]["input_ids"])
        flattened_features = [
            [
                {k: v[i] for k, v in feature.items()}
                for i in range(num_choices)
            ] for feature in features
        ]
        flattened_features = sum(flattened_features, [])

        batch = self.tokenizer.pad(
            flattened_features,
            padding=self.padding,
            max_length=self.max_length,
            pad_to_multiple_of=self.pad_to_multiple_of,
            return_tensors="pt",
        )
        batch = {k: v.view(batch_size, num_choices, -1) for k, v in batch.items()}
        batch["labels"] = torch.tensor(labels, dtype=torch.int64)
        return batch


def get_map3(label_df: pd.DataFrame, pred_df: pd.DataFrame):
    pred_df = pred_df["prediction"].str.split(" ", expand=True).rename({0: "pred0", 1: "pred1", 2: "pred2"}, axis=1)
    joined_df = label_df.join(pred_df, how="left")
    assert not joined_df["pred0"].isna().any(), f"{joined_df['pred0'].isna().sum() = }"
    assert not joined_df["pred1"].isna().any(), f"{joined_df['pred1'].isna().sum() = }"
    assert not joined_df["pred2"].isna().any(), f"{joined_df['pred2'].isna().sum() = }"

    map3 = 0
    ranks_to_scores = [1.0, 1 / 2, 1 / 3]
    for k in range(3):
        map3 += ranks_to_scores[k] * (joined_df[f"pred{k}"] == joined_df[f"answer"]).sum() / len(joined_df)
    return map3
