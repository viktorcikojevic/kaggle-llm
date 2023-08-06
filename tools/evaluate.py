from kaggle_llm.core import get_map3, get_map3_df, ROOT_PATH
from pathlib import Path
import pandas as pd
import argparse


def main(
        pred_path: str,
        label_path: str,
):
    out_path = ROOT_PATH / "eval"
    out_path.mkdir(parents=True, exist_ok=True)
    label_df = pd.read_csv(label_path)
    pred_path = Path(pred_path)
    if pred_path.is_dir():
        pred_paths = sorted(list(pred_path.glob("*.csv")))
    elif pred_path.is_file():
        pred_paths = [pred_path]
    else:
        raise RuntimeError(f"{pred_path} does not exist")
    for p in pred_paths:
        pred_df = pd.read_csv(p, index_col=0)
        map3_df = get_map3_df(label_df, pred_df)
        mean_scores = map3_df.groupby("context_topic_0")["scores"].mean().sort_values(ascending=False)
        print("----")
        print(f"{p.name}: {get_map3(label_df=label_df, pred_df=pred_df)}")
        print(mean_scores)
        mean_scores.to_csv(out_path / f"{p.stem}_res.csv")
        print("----")


if __name__ == "__main__":
    parser = argparse.ArgumentParser()
    parser.add_argument("pred")
    parser.add_argument("--label-path", type=str, required=False, default="/home/clay/research/kaggle/kaggle_llm/data/kaggle-llm-science-exam-splitted/test.csv")
    args, _ = parser.parse_known_args()
    main(args.pred, args.label_path)
