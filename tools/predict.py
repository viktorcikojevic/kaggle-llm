from kaggle_llm.adapted_models import *
from kaggle_llm.core import (
    ROOT_PATH,
    DataCollatorForMultipleChoice,
    WORK_DIRS_PATH,
    infer_pred_from_scores,
    get_tokenize_dataset_from_df,
    drop_df_cols_for_dataset,
)
from transformers import AutoModelForMultipleChoice, Trainer, AutoTokenizer, TrainingArguments
from peft import PeftModel
from sklearn.preprocessing import normalize
from pathlib import Path
import pandas as pd
import argparse
import yaml
import json


def main(
        input_df_path: str,
        output_dir: str,
        base_models_dir: str = "",
):
    input_df_path = Path(input_df_path)
    assert input_df_path.is_file(), f"{input_df_path} not found"
    output_dir = Path(output_dir)
    output_dir.mkdir(exist_ok=True, parents=True)

    with open(ROOT_PATH / "configs" / "submission.yaml", "rb") as f:
        submission_config = yaml.load(f, yaml.FullLoader)

    print(json.dumps(submission_config, indent=4))
    models = submission_config["models"]
    df = pd.read_csv(input_df_path)
    df = drop_df_cols_for_dataset(df)

    for i, load_from in enumerate(models):
        print(f"initting models [{i}]")
        abs_load_from = WORK_DIRS_PATH / load_from
        is_peft = "peft" in abs_load_from.parent.name
        tokenizer = AutoTokenizer.from_pretrained(abs_load_from)
        if is_peft:
            config = json.loads(Path(abs_load_from / "adapter_config.json").read_text())
            base_model_path = Path(config["base_model_name_or_path"])
            print(f"{base_models_dir = }")
            if base_models_dir != "":
                base_model_path = Path(base_models_dir) / base_model_path.name
                print(f"base_model_dir given, overriding peft base_model_path to: {base_model_path}")
            model = AutoModelForMultipleChoice.from_pretrained(base_model_path, load_in_8bit=True, device_map="auto")
            model = PeftModel.from_pretrained(model, abs_load_from)
            if hasattr(model.base_model, "load_extra_modules"):
                model.base_model.load_extra_modules(abs_load_from)
            kwargs = dict(
                remove_unused_columns=False,
                label_names=["labels"],
            )
        else:
            model = AutoModelForMultipleChoice.from_pretrained(abs_load_from)
            kwargs = {}
        print(f"initted models [{i}]")

        print(f"initting tokenizer and trainer [{i}]")
        tokenized_dataset = get_tokenize_dataset_from_df(df, tokenizer)
        training_args = TrainingArguments(
            per_device_eval_batch_size=1,
            output_dir="/tmp/kaggle_llm_pred",
            **kwargs,
            fp16=True,
        )
        trainer = Trainer(
            model=model.eval(),
            args=training_args,
            tokenizer=tokenizer,
            data_collator=DataCollatorForMultipleChoice(tokenizer=tokenizer),
            train_dataset=None,
        )
        print(f"initted tokenizer and trainer [{i}]")

        print(f"predicting [{i}]")
        with torch.no_grad():
            preds = trainer.predict(tokenized_dataset)
        print(f"predicted [{i}]")

        model_output_path = output_dir / (load_from.replace("/", "-") + ".csv")
        normalised_preds = normalize(-preds.predictions)
        preds_df = pd.DataFrame({
            "id": df.index,
            **{f"score{k}": normalised_preds[:, k] for k in range(5)}
        }).set_index("id")
        preds_df = infer_pred_from_scores(preds_df)
        preds_df.to_csv(model_output_path, header=True)


if __name__ == "__main__":
    parser = argparse.ArgumentParser()
    parser.add_argument("df_path")
    parser.add_argument("--output-dir", default=f"{str(ROOT_PATH)}/preds")
    parser.add_argument("--base-models-dir", default="")
    args, _ = parser.parse_known_args()
    main(args.df_path, args.output_dir, args.base_models_dir)
