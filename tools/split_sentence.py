from kaggle_llm.causal_lm_filters import *
from pathlib import Path
import blingfire as bf
from tqdm import tqdm
from typing import *
import pandas as pd
import argparse
import yaml
import json


def split_text_into_chunks(text: str, chunk_size: int) -> List[str]:
    tokens = [i for i in text.split() if len(i) > 0]
    start = 0
    end = chunk_size
    chunks = []
    while end < len(tokens):
        chunks.append(" ".join(tokens[start: end]))
        start += chunk_size
        end += chunk_size
    return chunks


def split_text_to_sentences(text: str, block_size: int) -> List[str]:
    sentences = []
    _, sentence_offsets = bf.text_to_sentences_and_offsets(text)
    for start_idx, end_idx in sentence_offsets:
        is_long_enough = (end_idx - start_idx) > 3
        sentence = text[start_idx: end_idx]
        is_math = "\\" in sentence  # leads to excessive tokens
        has_too_much_css = "style=" in sentence
        if is_long_enough and (not is_math) and (not has_too_much_css):
            sentences += split_text_into_chunks(sentence, block_size)
    return sentences


def main(
        config_path: str,
        output_dir: str,
        block_size: int,
):
    with open(config_path, "rb") as f:
        config = yaml.load(f, yaml.FullLoader)
    print(json.dumps(config, indent=4))

    output_dir = Path(output_dir)
    output_dir.mkdir(exist_ok=True, parents=True)

    for df_path in tqdm(config["raw_inputs"]):
        df_path = Path(df_path)
        df = pd.read_parquet(df_path)
        blacklist = df[
            (
                    df["title"].apply(begins_with_year)
                    | df["title"].apply(is_year_in_something)
                    | df["title"].apply(is_list_of)
                    | df["title"].apply(is_glossary)
                    | df["title"].apply(is_fiction)
                    | df["title"].apply(is_timeline_of)
                    | df["title"].apply(is_weather_of)
                    | df["title"].apply(is_data_page)
            )
        ]["title"].to_list()
        filtered_df = df[~df["title"].isin(blacklist)]
        sentences_for_df = []
        for _, row in tqdm(filtered_df.iterrows(), total=filtered_df.shape[0]):
            sentences = split_text_to_sentences(row["text"], block_size)
            sentences_for_df += sentences
        out_df = pd.DataFrame({"sentences": sentences_for_df})
        out_df.to_csv(output_dir / f"{df_path.stem}.csv")


if __name__ == "__main__":
    parser = argparse.ArgumentParser()
    parser.add_argument("config")
    parser.add_argument("output_dir")
    parser.add_argument("--block-size", required=False, type=int, default=128)
    _args, _ = parser.parse_known_args()
    main(
        _args.config,
        _args.output_dir,
        _args.block_size
    )
