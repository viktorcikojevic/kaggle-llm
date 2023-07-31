from kaggle_llm.adapted_models import *
from kaggle_llm.core import (
    ROOT_PATH,
    multiple_choice_preprocess,
    DataCollatorForMultipleChoice,
    WORK_DIRS_PATH,
    infer_pred_from_scores,
)
from transformers import AutoModelForMultipleChoice, Trainer, AutoTokenizer
from sklearn.preprocessing import normalize
from datasets import Dataset
from pathlib import Path
import pandas as pd
import argparse
import yaml
import json


def main(
        input_df_path: str,
        output_dir: str,
):
    input_df_path = Path(input_df_path)
    assert input_df_path.is_file(), f"{input_df_path} not found"
    output_dir = Path(output_dir)
    output_dir.mkdir(exist_ok=True, parents=True)

    with open(ROOT_PATH / "configs" / "submission.yaml", "rb") as f:
        submission_config = yaml.load(f, yaml.FullLoader)

    print(json.dumps(submission_config, indent=4))
    models = submission_config["models"]

    print("initting dataset")
    df = pd.read_csv(input_df_path)
    dataset = Dataset.from_pandas(df)
    print("initted dataset")

    for i, load_from in enumerate(models):
        print(f"initting models [{i}]")
        abs_load_from = WORK_DIRS_PATH / load_from
        tokenizer = AutoTokenizer.from_pretrained(abs_load_from)
        model = AutoModelForMultipleChoice.from_pretrained(abs_load_from)
        # model = AutoModelForMultipleChoice.from_pretrained(abs_load_from)
        print(f"initted models [{i}]")

        print(f"initting tokenizer and trainer [{i}]")
        tokenized_dataset = dataset.map(
            lambda example: multiple_choice_preprocess(tokenizer, example),
            remove_columns=["prompt", "A", "B", "C", "D", "E"] + (["answer"] if "answer" in df.columns else [])
        )
        trainer = Trainer(
            model=model,
            args=None,
            tokenizer=tokenizer,
            data_collator=DataCollatorForMultipleChoice(tokenizer=tokenizer),
            train_dataset=None,
        )
        print(f"initted tokenizer and trainer [{i}]")

        print(f"predicting [{i}]")
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
    args, _ = parser.parse_known_args()
    main(args.df_path, args.output_dir)
