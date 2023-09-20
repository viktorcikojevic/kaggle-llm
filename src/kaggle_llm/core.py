from typing import *
from kaggle_llm.causal_lm_filters import *
from transformers.tokenization_utils_base import PaddingStrategy, PreTrainedTokenizerBase
from transformers import EvalPrediction, PreTrainedModel, Trainer, TrainingArguments, EarlyStoppingCallback, AutoTokenizer, LlamaTokenizer
import transformers
from sklearn.preprocessing import normalize
from sklearn.model_selection import KFold
from dataclasses import dataclass
from datasets import Dataset
from pathlib import Path
from peft import PeftModel, PeftConfig, prepare_model_for_kbit_training
import peft
import pandas as pd
import numpy as np
import torch
import json
import os
from tqdm import tqdm


ROOT_PATH = Path(__file__).resolve().absolute().parent.parent.parent
WORK_DIRS_PATH = ROOT_PATH / "work_dirs"


options = ["A", "B", "C", "D", "E"]
option_to_index = {v: i for i, v in enumerate(options)}


def count_words(text):
    return sum([1 for i in text.split() if len(i) > 0])


def multiple_choice_prompting_preprocess(
        tokenizer: PreTrainedTokenizerBase,
        example: Dict[str, Any],
):
    text = (
        f"Select the most correct answer between A, B, C, D, E to the given Question.{tokenizer.sep_token}"
        f"Question: {example['prompt']}{tokenizer.sep_token}"
        f"A. {example['A']}{tokenizer.sep_token}"
        f"B. {example['B']}{tokenizer.sep_token}"
        f"C. {example['C']}{tokenizer.sep_token}"
        f"D. {example['D']}{tokenizer.sep_token}"
        f"E. {example['E']}{tokenizer.sep_token}"
        f"Answer: "
    )
    tokenized_example = tokenizer(
        text,
        truncation=True,
    )
    all_stop_token_indices = [i for i, x in enumerate(tokenized_example.data["input_ids"]) if x == tokenizer.sep_token_id]
    first_stop_token_idx = len(all_stop_token_indices) - 1 - 5
    tokenized_example["stop_token_indices"] = all_stop_token_indices[first_stop_token_idx: first_stop_token_idx+5]
    if "answer" in example:
        tokenized_example["label"] = option_to_index[example["answer"]]
    return tokenized_example


@dataclass
class DataCollatorForMultipleChoicePrompting:
    tokenizer: PreTrainedTokenizerBase
    padding: Union[bool, str, PaddingStrategy] = True
    max_length: Optional[int] = None
    pad_to_multiple_of: Optional[int] = None

    def __call__(self, features: List[Dict[str, Any]]):
        flattened_features = [
            {
                k: v
                for k, v in feature.items()
                if k not in ("label", "labels")
            }
            for feature in features
        ]

        # _max_length = max([len(x["input_ids"]) for x in flattened_features])
        # print(f"{_max_length = }")

        batch = self.tokenizer.pad(
            flattened_features,
            padding=self.padding,
            max_length=self.max_length,
            pad_to_multiple_of=self.pad_to_multiple_of,
            return_tensors="pt",
        )
        if "label" in features[0].keys() or "labels" in features[0].keys():
            label_name = "label" if "label" in features[0].keys() else "labels"
            labels = [feature.pop(label_name) for feature in features]
            batch["labels"] = torch.tensor(labels, dtype=torch.int64)
        return batch


def multiple_choice_preprocess(tokenizer: PreTrainedTokenizerBase, example: Dict[str, Any], preprocess_type: str, max_input: int):
    """
    The example is expected to be a dictionary with keys "prompt", "A", "B", "C", "D", "E", and "answer".
    """
    assert preprocess_type in ["sumo", "deotte"], "preprocess_type must be either sumo or deotte"
    if preprocess_type == "sumo":
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

    if preprocess_type == "deotte":
        first_sentence = [ "[CLS] " + example['context'] ] * 5
        second_sentences = [" #### " + example['prompt'] + " [SEP] " + example[option] + " [SEP]" for option in 'ABCDE']
        tokenized_example = tokenizer(first_sentence, second_sentences, truncation='only_first', 
                                    max_length=max_input, add_special_tokens=False)        
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
                {k: v[i] for k, v in feature.items() if k not in ("context", "__index_level_0__", "label", "labels")}
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


def compute_map3_hf(preds: EvalPrediction) -> Dict:
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
        print(f"Loading {ip} ...")
        df = pd.read_csv(ip)
        df = df.fillna("None") # Weird bug: when loading "None" string, it becomes NaN in the pandas df
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
        dbg_train_df[c] = dbg_train_df[c].astype(str)
        dbg_train_df[f"{c}_wc"] = dbg_train_df[c].apply(count_words)
    dbg_train_df["choice_wc"] = dbg_train_df[[f"{c}_wc" for c in choices]].max(axis=1)
    dbg_train_df["all_wc"] = dbg_train_df["prompt_wc"] + dbg_train_df["choice_wc"]
    print(dbg_train_df["all_wc"].describe(np.linspace(0, 1, 11)))
    print("val:")
    dbg_val_df = val_df.copy()
    dbg_val_df["prompt_wc"] = dbg_val_df["prompt"].apply(count_words)
    for c in choices:
        # convert to str
        dbg_val_df[c] = dbg_val_df[c].astype(str)
        dbg_val_df[f"{c}_wc"] = dbg_val_df[c].apply(count_words)
    dbg_val_df["choice_wc"] = dbg_val_df[[f"{c}_wc" for c in choices]].max(axis=1)
    dbg_val_df["all_wc"] = dbg_val_df["prompt_wc"] + dbg_val_df["choice_wc"]
    print(dbg_val_df["all_wc"].describe(np.linspace(0, 1, 11)))

    return train_df, val_df


def get_tokenize_dataset_from_df(df: pd.DataFrame, tokenizer: PreTrainedTokenizerBase, preprocess_type: str, max_input: int):
    dataset = Dataset.from_pandas(df)
    topic_cols = [c for c in df.columns if "topic" in c]
    return dataset.map(
        lambda example: multiple_choice_preprocess(tokenizer, example, preprocess_type, max_input),
        remove_columns=(
            ["prompt", "A", "B", "C", "D", "E"]
            + (["answer"] if "answer" in df else [])
            + topic_cols
            + (["index"] if "index" in df else [])
        )
    )


def get_mcp_tokenize_dataset_from_df(df: pd.DataFrame, tokenizer: PreTrainedTokenizerBase):
    dataset = Dataset.from_pandas(df)
    topic_cols = [c for c in df.columns if "topic" in c]
    return dataset.map(
        lambda example: multiple_choice_prompting_preprocess(tokenizer, example),
        remove_columns=(
            ["prompt", "A", "B", "C", "D", "E"]
            + (["answer"] if "answer" in df else [])
            + topic_cols
            + (["index"] if "index" in df else [])
        ),
        nproc=4,
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


def train_and_save_best_model_on_error(
        trainer: Trainer,
        model_output_dir: Path,
        best_model_dir_name: str = "best_map3"
):
    try:
        trainer.train()
    except KeyboardInterrupt:
        print(f"training interrupted, moving on to save the model")
    except Exception as e:
        print(f"training FAILED: {repr(e)}")
    finally:
        if trainer.state.best_model_checkpoint is not None:
            print(f"trainer has some best models stored: {trainer.state.best_model_checkpoint}, setting it as the best checkpoint")
            os.symlink(trainer.state.best_model_checkpoint, str(model_output_dir / best_model_dir_name))
        else:
            print(f"trainer has NO best models stored, returning")
            return


def print_trainable_parameters(model):
    trainable_params = 0
    all_param = 0
    for _, param in model.named_parameters():
        num_params = param.numel()
        # if using DS Zero 3 and the weights are initialized empty
        if num_params == 0 and hasattr(param, "ds_numel"):
            num_params = param.ds_numel

        all_param += num_params
        if param.requires_grad:
            trainable_params += num_params
    print(
        f"trainable params: {trainable_params:,d} || all params: {all_param:,d} || trainable%: {100 * trainable_params / all_param}"
    )


def count_conversion_ratio(model, print_convert_names: bool = True):
    f16_count = 0
    all_count = 0
    for n, p in model.named_parameters():
        all_count += 1
        if p.dtype == torch.float16:
            if print_convert_names:
                print(n)
            f16_count += 1
    conversion_ratio = f16_count / all_count
    print(f"{conversion_ratio = }")


def build_peft_model(
        load_from: str,
        use_peft: bool = False,
        peft_class: str = "AdaLoraConfig",
        transformer_class: str = "AutoModelForMultipleChoice",
        use_8bit: bool = False,
        **peft_kwargs
) -> Tuple[WrappedPeftModel, PreTrainedTokenizerBase]:
    tokenizer = AutoTokenizer.from_pretrained(load_from)
    transformer_constructor = getattr(transformers, transformer_class)
    if use_8bit:
        model = transformer_constructor.from_pretrained(load_from, load_in_8bit=True)
        count_conversion_ratio(model, True)
        model = prepare_model_for_kbit_training(model)
    else:
        model = transformer_constructor.from_pretrained(load_from)
    print(f"{model.__class__.__name__ = }")
    if not use_peft:
        return model, tokenizer
    print(json.dumps(peft_kwargs))
    config_class = getattr(peft, peft_class)
    print(f"{config_class = }")
    peft_config = config_class(
        inference_mode=False,
        **peft_kwargs,
    )
    model = WrappedPeftModel(model, peft_config)
    print(print_trainable_parameters(model))
    return model, tokenizer


def build_trainer(
        model,
        tokenizer: PreTrainedTokenizerBase,
        learning_rate: float,
        report_to: List[str],
        output_dir: str,
        train_dataset,
        eval_dataset,
        data_collator,
        metric_for_best_model: str = "map3",
        greater_is_better: bool = True,
        per_device_train_batch_size: int = 1,
        per_device_eval_batch_size: int = 2,
        early_stopping_patience: int = 4,
        warmup_epochs: int = 1,
        total_epochs: int = 20,
        compute_metrics: Optional[Callable] = compute_map3_hf,
):
    warmup_ratio = warmup_epochs / total_epochs
    training_args = TrainingArguments(
        lr_scheduler_type="cosine",
        metric_for_best_model=metric_for_best_model,
        greater_is_better=greater_is_better,
        warmup_ratio=warmup_ratio,
        learning_rate=learning_rate,
        per_device_train_batch_size=per_device_train_batch_size,
        per_device_eval_batch_size=per_device_eval_batch_size,
        load_best_model_at_end=True,
        evaluation_strategy="epoch",
        save_strategy="epoch",
        num_train_epochs=total_epochs,
        save_total_limit=2,
        report_to=report_to,
        output_dir=output_dir,
        remove_unused_columns=False,  # HF infers the cols based on model's forward signature, and peft corrupts it
        label_names=["labels"],  # for peft
        # deepspeed=str((ROOT_PATH / "configs" / "deepspeed.json").resolve().absolute()),
        fp16=True,
        ddp_find_unused_parameters=False,
        # optim=transformers.training_args.OptimizerNames.ADAMW_BNB,
        # gradient_checkpointing=True,
        # gradient_accumulation_steps=4,
    )
    return Trainer(
        model=model,
        args=training_args,
        tokenizer=tokenizer,
        data_collator=data_collator,
        train_dataset=train_dataset,
        eval_dataset=eval_dataset,
        compute_metrics=compute_metrics,
        callbacks=[
            EarlyStoppingCallback(early_stopping_patience=early_stopping_patience),
        ],
    )


def load_causal_lm_dataset(
        input_paths: List[Union[str, Path]],
) -> pd.DataFrame:
    dfs = []
    for ip in input_paths:
        df = pd.read_csv(ip)
        df = drop_df_cols_for_dataset(df)
        dfs.append(df)
    catted_df = pd.concat(dfs, axis=0).reset_index()
    return catted_df



def generate_new_prompt(prompt, analyze_option, options):
    
    new_prompt = f'''You are a university professor, renowned as an expert in your field. Your teaching style is known for providing comprehensive and lengthy explanations, ensuring that your students grasp the depths of the topic at hand.
        
        Context: A student of yours approaches you after class, presenting a query that's been puzzling her. She's received varied opinions from her peers and is seeking clarity.
        
        Student's Question: {prompt}
        
        She think that the answer is: {analyze_option}.
        
        She's been told by her colleagues that potential answers could be:
        Option 1: {options[0]}.
        Option 2: {options[1]}.
        Option 3: {options[2]}.
        Option 4: {options[3]}.

        Kindly delve into the problem, elucidating the validity of her answer. Your aim is to tell her whether the answer is correct or wrong at the end. Critically examine each option, helping her discern how "wrong" or "right" her answer is. Provide a step-by-step analysis.
    '''
    return new_prompt.replace("..", ".")

def add_context(df):
    
    new_df = df.copy()

    # append new_csv 4 times
    for i in range(4):
        new_df = pd.concat([new_df, df])
        
    print(f"New df length: {len(new_df)}")
    new_df['analyze_answer'] = ["A", "B", "C", "D", "E"] * (len(new_df) // 5)
    
    print(f"Adding context ... ")
    for i, row in tqdm(new_df.iterrows(), total=len(new_df)):
        possible_options = ["A", "B", "C", "D", "E"]
        
        # remove the correct answer from the possible options
        possible_options.remove(row['answer'])
        
        options = [row[option] for option in possible_options]
        
        analyze_option = row[row['answer']]
        
        new_df.at[i, 'new_prompt'] = generate_new_prompt(row['prompt'], analyze_option, options)    
        
    return new_df