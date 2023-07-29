from typing import *
from transformers.tokenization_utils_base import PaddingStrategy, PreTrainedTokenizerBase
from transformers import EvalPrediction
from sklearn.preprocessing import normalize
from sklearn.model_selection import KFold
from dataclasses import dataclass
from datasets import Dataset
from pathlib import Path
import pandas as pd
import numpy as np
import torch


ROOT_PATH = Path(__file__).resolve().absolute().parent.parent.parent
WORK_DIRS_PATH = ROOT_PATH / "work_dirs"


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
    if "answer" in example:
        tokenized_example["label"] = option_to_index[example["answer"]]
    return tokenized_example


@dataclass
class DataCollatorForMultipleChoice:
    tokenizer: PreTrainedTokenizerBase
    padding: Union[bool, str, PaddingStrategy] = True
    max_length: Optional[int] = None
    pad_to_multiple_of: Optional[int] = None

    def __call__(self, features):
        batch_size = len(features)
        num_choices = len(features[0]["input_ids"])
        flattened_features = [
            [
                {k: v[i] for k, v in feature.items() if k not in ("label", "labels")}
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
        if "label" in features[0].keys() or "labels" in features[0].keys():
            label_name = "label" if "label" in features[0].keys() else "labels"
            labels = [feature.pop(label_name) for feature in features]
            batch["labels"] = torch.tensor(labels, dtype=torch.int64)
        return batch


def infer_pred_from_scores(pred_df: pd.DataFrame):
    CHOICES_ARR = np.array(list("ABCDE"))
    pred_df = pred_df.copy()

    def get_pred(row):
        scores = row[[f"score{i}" for i in range(5)]]
        pred_letters = CHOICES_ARR[np.argsort(scores)][:3].tolist()
        return " ".join(pred_letters)

    pred_df["prediction"] = pred_df[[f"score{i}" for i in range(5)]].apply(get_pred, axis=1)
    return pred_df


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


def compute_metrics(preds: EvalPrediction) -> Dict:
    ids = list(range(len(preds.label_ids)))
    normalised_preds = normalize(-preds.predictions)
    preds_df = pd.DataFrame({
        "id": ids,
        **{f"score{k}": normalised_preds[:, k] for k in range(5)}
    }).set_index("id")
    preds_df = infer_pred_from_scores(preds_df)
    label_df = pd.DataFrame({
        "id": ids,
        "answer": ["ABCDE"[label] for label in preds.label_ids]
    })
    map3 = get_map3(label_df=label_df, pred_df=preds_df)
    return {"map3": map3}


def load_train_and_val_df(
        input_paths: List[Union[str, Path]],
        i_fold: int,
        total_fold: int,
) -> Tuple[pd.DataFrame, pd.DataFrame]:
    kf = KFold(n_splits=total_fold, shuffle=True, random_state=42)
    train_dfs = []
    val_dfs = []
    for ip in input_paths:
        df = pd.read_csv(ip)
        if "id" in df:
            df = df.drop("id", axis=1)
        if "index" in df:
            df = df.drop("index", axis=1)
        train_idx, val_idx = list(kf.split(df))[i_fold]
        train_df = df.loc[train_idx, :]
        val_df = df.loc[val_idx, :]
        train_dfs.append(train_df)
        val_dfs.append(val_df)

    train_df = pd.concat(train_dfs, axis=0).reset_index()
    val_df = pd.concat(val_dfs, axis=0).reset_index()
    return train_df, val_df


def get_tokenize_dataset_from_df(df: pd.DataFrame, tokenizer: PreTrainedTokenizerBase):
    dataset = Dataset.from_pandas(df)
    return dataset.map(
        lambda example: multiple_choice_preprocess(tokenizer, example),
        remove_columns=["prompt", "A", "B", "C", "D", "E", "answer"] + (["topic"] if "topic" in df else [])
    )
