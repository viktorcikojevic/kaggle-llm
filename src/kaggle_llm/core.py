from typing import *
from transformers.tokenization_utils_base import PaddingStrategy, PreTrainedTokenizerBase
from transformers import EvalPrediction, PreTrainedModel
from sklearn.preprocessing import normalize
from sklearn.model_selection import KFold
from dataclasses import dataclass
from datasets import Dataset
from pathlib import Path
from peft import PeftModel, PeftConfig
import pandas as pd
import numpy as np
import torch
import os


ROOT_PATH = Path(__file__).resolve().absolute().parent.parent.parent
WORK_DIRS_PATH = ROOT_PATH / "work_dirs"


options = ["A", "B", "C", "D", "E"]
option_to_index = {v: i for i, v in enumerate(options)}


def count_words(text):
    return sum([1 for i in text.split() if len(i) > 0])


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

    # _max_length = max([len(x) for x in tokenized_example["input_ids"]])
    # print(f"{_max_length = }")
    # if _max_length > 350:
    #     print(example)
    #     assert False, ":D"

    if "answer" in example:
        tokenized_example["label"] = option_to_index[example["answer"]]
    return tokenized_example


@dataclass
class DataCollatorForMultipleChoice:
    tokenizer: PreTrainedTokenizerBase
    padding: Union[bool, str, PaddingStrategy] = True
    max_length: Optional[int] = None
    # max_length: Optional[int] = 512  # sometimes the max_length is 980+ which breaks the gpu
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

        # _max_length = max([len(x["input_ids"]) for x in flattened_features])
        # print(f"{_max_length = }")

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


def get_map3_df(label_df: pd.DataFrame, pred_df: pd.DataFrame) -> pd.DataFrame:
    pred_df = pred_df["prediction"].str.split(" ", expand=True).rename({0: "pred0", 1: "pred1", 2: "pred2"}, axis=1)
    joined_df = label_df.join(pred_df, how="inner")
    assert not joined_df["pred0"].isna().any(), f"{joined_df['pred0'].isna().sum() = }"
    assert not joined_df["pred1"].isna().any(), f"{joined_df['pred1'].isna().sum() = }"
    assert not joined_df["pred2"].isna().any(), f"{joined_df['pred2'].isna().sum() = }"

    ranks_to_scores = [1.0, 1 / 2, 1 / 3]
    joined_df["scores"] = 0.0
    for k in range(3):
        joined_df["scores"] = joined_df["scores"] + ranks_to_scores[k] * (joined_df[f"pred{k}"] == joined_df[f"answer"])
    return joined_df


def get_map3(label_df: pd.DataFrame, pred_df: pd.DataFrame):
    return get_map3_df(label_df, pred_df)["scores"].mean()


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


def drop_df_cols_for_dataset(df: pd.DataFrame) -> pd.DataFrame:
    if "Unnamed: 0" in df:
        df = df.drop("Unnamed: 0", axis=1)
    if "id" in df:
        df = df.drop("id", axis=1)
    if "index" in df:
        df = df.drop("index", axis=1)
    return df


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
        df = drop_df_cols_for_dataset(df)
        train_idx, val_idx = list(kf.split(df))[i_fold]
        train_df = df.loc[train_idx, :]
        val_df = df.loc[val_idx, :]
        train_dfs.append(train_df)
        val_dfs.append(val_df)

    train_df = pd.concat(train_dfs, axis=0).reset_index()
    val_df = pd.concat(val_dfs, axis=0).reset_index()

    print("train:")
    choices = ["A", "B", "C", "D", "E"]
    dbg_train_df = train_df.copy()
    dbg_train_df["prompt_wc"] = dbg_train_df["prompt"].apply(count_words)
    for c in choices:
        dbg_train_df[f"{c}_wc"] = dbg_train_df[c].apply(count_words)
    dbg_train_df["choice_wc"] = dbg_train_df[[f"{c}_wc" for c in choices]].max(axis=1)
    dbg_train_df["all_wc"] = dbg_train_df["prompt_wc"] + dbg_train_df["choice_wc"]
    print(dbg_train_df["all_wc"].describe(np.linspace(0, 1, 11)))
    print("val:")
    dbg_val_df = val_df.copy()
    dbg_val_df["prompt_wc"] = dbg_val_df["prompt"].apply(count_words)
    for c in choices:
        dbg_val_df[f"{c}_wc"] = dbg_val_df[c].apply(count_words)
    dbg_val_df["choice_wc"] = dbg_val_df[[f"{c}_wc" for c in choices]].max(axis=1)
    dbg_val_df["all_wc"] = dbg_val_df["prompt_wc"] + dbg_val_df["choice_wc"]
    print(dbg_val_df["all_wc"].describe(np.linspace(0, 1, 11)))

    return train_df, val_df


def get_tokenize_dataset_from_df(df: pd.DataFrame, tokenizer: PreTrainedTokenizerBase):
    dataset = Dataset.from_pandas(df)
    topic_cols = [c for c in df.columns if "topic" in c]
    return dataset.map(
        lambda example: multiple_choice_preprocess(tokenizer, example),
        remove_columns=(
            ["prompt", "A", "B", "C", "D", "E"]
            + (["answer"] if "answer" in df else [])
            + topic_cols
            + (["index"] if "index" in df else [])
        )
    )


class WrappedPeftModel(PeftModel):
    """ peft model with some custom layers that's not handled by peft, e.g. classifier layers """
    def __init__(self, model: PreTrainedModel, peft_config: PeftConfig, adapter_name: str = "default"):
        PeftModel.__init__(self, model, peft_config, adapter_name)

    def save_pretrained(
        self,
        save_directory: str,
        safe_serialization: bool = False,
        selected_adapters: Optional[List[str]] = None,
        **kwargs: Any,
    ):
        PeftModel.save_pretrained(self, save_directory, safe_serialization, selected_adapters, **kwargs)
        if hasattr(self.base_model, "save_extra_modules"):
            self.base_model.save_extra_modules(save_directory)
            print(f"saved extra modules for class {self.base_model.__class__.__name__}")

    @classmethod
    def from_pretrained(
        cls,
        model: PreTrainedModel,
        model_id: Union[str, os.PathLike],
        adapter_name: str = "default",
        is_trainable: bool = False,
        config: Optional[PeftConfig] = None,
        **kwargs: Any,
    ):
        out_model = PeftModel.from_pretrained(model, model_id, adapter_name, is_trainable, config, **kwargs)
        if hasattr(out_model.base_model, "load_extra_modules"):
            out_model.base_model.load_extra_modules(model_id)
            print(f"loaded extra modules for class {out_model.base_model.__class__.__name__}")
        return out_model
