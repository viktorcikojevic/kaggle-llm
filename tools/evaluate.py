from src.kaggle_llm.core import get_map3
import pandas as pd
import argparse


def main(pred_path: str):
    label_df = pd.read_csv("/home/clay/research/kaggle/kaggle_llm/data/kaggle-llm-science-exam-splitted/test.csv")
    pred_df = pd.read_csv(pred_path, index_col=0)
    print(get_map3(label_df=label_df, pred_df=pred_df))


if __name__ == "__main__":
    parser = argparse.ArgumentParser()
    parser.add_argument("pred")
    args, _ = parser.parse_known_args()
    main(args.pred_path)
