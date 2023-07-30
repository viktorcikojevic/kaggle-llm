from kaggle_llm.core import (
    ROOT_PATH,
    count_words,
)
import pandas as pd
import numpy as np
from tqdm import tqdm
from typing import *
from sentence_transformers import SentenceTransformer
from pathlib import Path
import blingfire as bf
import argparse
import torch
import faiss
import json
import yaml


too_long_prompt_wc = 80
context_cutoff_len = 70


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


@torch.no_grad()
def get_sentence_embeddings(
        wiki_df_path: Union[str, Path],
        model: SentenceTransformer
):
    wiki_df = pd.read_csv(wiki_df_path, index_col=0)

    wc_per_page = wiki_df.groupby("page")[["word_count"]].sum().sort_values("word_count", ascending=False)
    black_list = list(wc_per_page.loc[
        (wc_per_page["word_count"] > 10000)
        | (wc_per_page.index.map(lambda x: "list of equations" in x.lower()))
    ].index)
    print(json.dumps(black_list, indent=4))

    filtered_wiki_df = wiki_df.loc[~wiki_df["page"].isin(black_list), :].copy()
    print(len(wiki_df), len(filtered_wiki_df))

    batch_size = 16
    sentences_df = []

    for _, row in tqdm(filtered_wiki_df.iterrows(), total=len(filtered_wiki_df)):
        _, sentence_offsets = bf.text_to_sentences_and_offsets(row["text"])
        for start_idx, end_idx in sentence_offsets:
            if (end_idx - start_idx) > 3:
                sentences_df.append({
                    "page": row["page"],
                    "i_sentence": len(sentences_df),
                    "text": row["text"][start_idx: end_idx],
                })

    sentences_df = pd.DataFrame.from_records(sentences_df)
    print(f"extracted: {len(sentences_df)} sentences")

    print(f"dropping too long sentences")
    pass_indices = sentences_df.loc[sentences_df["text"].apply(count_words) < context_cutoff_len, "text"].index
    print(f"keeping {len(pass_indices) / len(sentences_df) * 100} % at cutoff {context_cutoff_len}")
    sentences_df = sentences_df.loc[pass_indices, :].reset_index().copy()

    sentence_embeddings = model.encode(
        sentences_df["text"].values,
        batch_size=batch_size,
        device="cuda",
        show_progress_bar=True,
        convert_to_tensor=True,
        normalize_embeddings=True,
    ).half()
    sentence_embeddings = sentence_embeddings.detach().cpu().numpy()

    sentence_index = faiss.IndexFlatL2(sentence_embeddings.shape[1])
    sentence_index.add(sentence_embeddings)
    print(f"{sentence_index.ntotal = }")
    return sentences_df, sentence_index


@torch.no_grad()
def main(
        wiki_df_path: Union[str, Path],
        out_dir: Union[str, Path],
        sentence_model: Union[str, Path] = "/home/clay/research/kaggle/kaggle_llm/data/sentence_transformer_model",
        k: int = 3,
):
    config_path = ROOT_PATH / "configs" / "multiple_choice.yaml"
    with open(config_path, "rb") as f:
        config = yaml.load(f, yaml.FullLoader)

    model = SentenceTransformer(sentence_model, device="cuda")
    model.max_seq_length = 384
    model = model.half()

    wiki_df_path = Path(wiki_df_path)
    print(f"computing wiki embeddings")
    sentences_df, sentence_index = get_sentence_embeddings(wiki_df_path, model)
    out_dir = Path(out_dir)
    out_dir.mkdir(parents=True, exist_ok=True)
    batch_size = 16
    print(f"computed wiki embeddings")

    for train_df_path in config["raw_inputs"]:
        train_df_path = Path(train_df_path).resolve().absolute()
        print(f"computing contexts for: {train_df_path}")

        train_df = pd.read_csv(train_df_path)
        if "id" in train_df.columns:
            train_df = train_df.drop("id", axis=1)

        train_df["prompt_and_answer"] = (
                train_df["prompt"]
                + " " + train_df["A"]
                + " " + train_df["B"]
                + " " + train_df["C"]
                + " " + train_df["D"]
                + " " + train_df["E"]
        )
        question_embeddings = model.encode(
            train_df["prompt_and_answer"].values,
            batch_size=batch_size,
            device="cuda",
            show_progress_bar=True,
            convert_to_tensor=True,
            normalize_embeddings=True,
        ).half()
        question_embeddings = question_embeddings.detach().cpu().numpy()

        distance, indices = sentence_index.search(question_embeddings, k)

        for i in range(k):
            train_df[f"context_{i}_idx"] = indices[:, i]

        for i in range(k):
            train_df[f"context_{i}"] = train_df.join(
                sentences_df["text"],
                on=f"context_{i}_idx",
                how="left",
            )["text"]

        assert not train_df["prompt"].isna().any(), f"{train_df_path} contains {train_df['prompt'].isna().sum()} dumbass prompts"

        def join_prompt_with_context(_row):
            joined = _row["prompt"]
            current_len = count_words(joined)
            already_added_context = False
            _i_context = 0
            for _i in range(k):
                context = _row[f"context_{_i}"]
                context_len = count_words(context)
                if already_added_context and (current_len + context_len > too_long_prompt_wc):
                    continue
                current_len += context_len
                already_added_context = True
                joined += f"\ncontext {_i_context}: " + context
                _i_context += 1
            return joined

        train_df["prompt"] = train_df.apply(join_prompt_with_context, axis=1)
        assert not train_df["prompt"].isna().any(), f"{train_df_path} contains {train_df['prompt'].isna().sum()} dumbass prompts after proc"
        train_df = train_df.drop(
            (
                ["prompt_and_answer"]
                + [f"context_{i}" for i in range(k)]
                + [f"context_{i}_idx" for i in range(k)]
            ),
            axis=1,
        )

        out_path = (out_dir / train_df_path.name).resolve().absolute()
        print(train_df["prompt"].apply(count_words).describe(np.linspace(0, 1, 11)))
        train_df.to_csv(out_path)
        print(f"saved: {out_path}")


if __name__ == "__main__":
    parser = argparse.ArgumentParser()
    parser.add_argument("wiki_df_path")
    parser.add_argument("out_dir")
    parser.add_argument("sentence_model")
    parser.add_argument("-k", type=int, default=3)
    args, _ = parser.parse_known_args()
    main(
        wiki_df_path=args.wiki_df_path,
        out_dir=args.out_dir,
        sentence_model=args.sentence_model,
        k=args.k
    )
