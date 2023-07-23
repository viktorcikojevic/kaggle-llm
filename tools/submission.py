from pathlib import Path
import pandas as pd
import numpy as np
import argparse
import sys


CHOICES_ARR = np.array(list("ABCDE"))


def main(
        pred_path: str,
        output_path: str = "",
):
    pred_path = Path(pred_path)
    assert pred_path.is_file(), f"{pred_path} not found"

    if not output_path:
        output_path = "submission.csv"
    output_path = Path(output_path).resolve().absolute()

    df = pd.read_csv(pred_path, index_col=0)
    # in case there are many columns lying around from ensembling
    df["prediction"].to_csv(output_path)
    print(f"wrote submission to {output_path}")


if __name__ == "__main__":
    parser = argparse.ArgumentParser()
    parser.add_argument("pred_path")
    parser.add_argument("--output-path", default="")
    args, _ = parser.parse_known_args()
    main(args.pred_path, args.output_path)
