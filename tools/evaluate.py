from src.kaggle_llm.core import get_map3
from pathlib import Path
import pandas as pd
import argparse


def main(pred_path: str):
    label_df = pd.read_csv("/home/clay/research/kaggle/kaggle_llm/data/kaggle-llm-science-exam-splitted/test.csv")
    pred_path = Path(pred_path)
    if pred_path.is_dir():
        pred_paths = sorted(list(pred_path.glob("*.csv")))
    elif pred_path.is_file():
        pred_paths = [pred_path]
    else:
        raise RuntimeError(f"{pred_path} does not exist")
    for p in pred_paths:
        pred_df = pd.read_csv(p, index_col=0)
        print(f"{p.name}: {get_map3(label_df=label_df, pred_df=pred_df)}")


if __name__ == "__main__":
    parser = argparse.ArgumentParser()
    parser.add_argument("pred")
    args, _ = parser.parse_known_args()
    main(args.pred)
