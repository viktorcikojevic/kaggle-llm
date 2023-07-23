from kaggle_llm.core import infer_pred_from_scores
from pathlib import Path
import pandas as pd
import numpy as np
import argparse
import json


CHOICES_ARR = np.array(list("ABCDE"))


def main(
        pred_dir: str,
        output_path: str = "",
):
    pred_dir = Path(pred_dir)
    assert pred_dir.is_dir(), f"{pred_dir} not found"

    csv_paths = [p for p in pred_dir.glob("*.csv") if p.name != "ensembled.csv"]
    csv_paths_str = json.dumps([str(x) for x in csv_paths], indent=4)
    print(f"found the following files under {pred_dir}\n{csv_paths_str}")

    if not output_path:
        output_path = pred_dir / "ensembled.csv"
    output_path = Path(output_path).resolve().absolute()

    sum_df = None
    for csv_path in csv_paths:
        csv = pd.read_csv(csv_path, index_col=0, header=0)
        scores_df = csv[[f"score{i}" for i in range(5)]]
        if sum_df is None:
            sum_df = scores_df.copy()
        else:
            sum_df += scores_df
    for i in range(5):
        sum_df[f"score{i}"] /= len(csv_paths)

    sum_df = infer_pred_from_scores(sum_df)
    sum_df.to_csv(output_path)
    print(f"wrote ensembled results to {output_path}")


if __name__ == "__main__":
    parser = argparse.ArgumentParser()
    parser.add_argument("pred_dir")
    parser.add_argument("--output-path", default="")
    args, _ = parser.parse_known_args()
    main(args.pred_dir, args.output_path)
