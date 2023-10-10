#!/usr/bin/env python
# coding: utf-8

# # OpenBook DeBERTaV3-Large with an updated model
# 
# This work is based on the great [work](https://www.kaggle.com/code/nlztrk/openbook-debertav3-large-baseline-single-model) of [nlztrk](https://www.kaggle.com/nlztrk).
# 
# I trained a model offline using the dataset I shared [here](https://www.kaggle.com/datasets/mgoksu/llm-science-exam-dataset-w-context). I just added my model to the original notebook. The model is available [here](https://www.kaggle.com/datasets/mgoksu/llm-science-run-context-2).
# 
# I also addressed the problem of [CSV Not Found at submission](https://www.kaggle.com/competitions/kaggle-llm-science-exam/discussion/434228) with this notebook by clipping the context like so:
# 
# `test_df["prompt"] = test_df["context"].apply(lambda x: x[:1500]) + " #### " +  test_df["prompt"]`
# 
# You can probably get more than 1500 without getting an OOM.




# # installing offline dependencies
# !pip install -U /kaggle/input/faiss-gpu-173-python310/faiss_gpu-1.7.2-cp310-cp310-manylinux_2_17_x86_64.manylinux2014_x86_64.whl
# !cp -rf /kaggle/input/sentence-transformers-222/sentence-transformers /kaggle/working/sentence-transformers
# !pip install -U /kaggle/working/sentence-transformers
# !pip install -U /kaggle/input/blingfire-018/blingfire-0.1.8-py3-none-any.whl

# !pip install --no-index --no-deps /kaggle/input/llm-whls/transformers-4.31.0-py3-none-any.whl
# !pip install --no-index --no-deps /kaggle/input/llm-whls/peft-0.4.0-py3-none-any.whl
# !pip install --no-index --no-deps /kaggle/input/llm-whls/datasets-2.14.3-py3-none-any.whl
# !pip install --no-index --no-deps /kaggle/input/llm-whls/trl-0.5.0-py3-none-any.whl



import sys

N_TOP_DOCS = 5
## Parameter to determine how many relevant sentences to include
NUM_SENTENCES_INCLUDE = 20





import os
import gc
import pandas as pd
import numpy as np
import re
from tqdm.auto import tqdm
import blingfire as bf
# from __future__ import annotations

from collections.abc import Iterable

import faiss
from faiss import write_index, read_index

from sentence_transformers import SentenceTransformer

import torch
import ctypes
libc = ctypes.CDLL("libc.so.6")

from dataclasses import dataclass
from typing import Optional, Union

import torch
import numpy as np
import pandas as pd
from datasets import Dataset
from transformers import AutoTokenizer
from transformers import AutoModelForMultipleChoice, TrainingArguments, Trainer
from transformers.tokenization_utils_base import PreTrainedTokenizerBase, PaddingStrategy
from torch.utils.data import DataLoader





def process_documents(documents: Iterable[str],
                      document_ids: Iterable,
                      split_sentences: bool = True,
                      filter_len: int = 3,
                      disable_progress_bar: bool = False) -> pd.DataFrame:
    """
    Main helper function to process documents from the EMR.

    :param documents: Iterable containing documents which are strings
    :param document_ids: Iterable containing document unique identifiers
    :param document_type: String denoting the document type to be processed
    :param document_sections: List of sections for a given document type to process
    :param split_sentences: Flag to determine whether to further split sections into sentences
    :param filter_len: Minimum character length of a sentence (otherwise filter out)
    :param disable_progress_bar: Flag to disable tqdm progress bar
    :return: Pandas DataFrame containing the columns `document_id`, `text`, `section`, `offset`
    """
    
    df = sectionize_documents(documents, document_ids, disable_progress_bar)

    if split_sentences:
        df = sentencize(df.text.values, 
                        df.document_id.values,
                        df.offset.values, 
                        filter_len, 
                        disable_progress_bar)
    return df


def sectionize_documents(documents: Iterable[str],
                         document_ids: Iterable,
                         disable_progress_bar: bool = False) -> pd.DataFrame:
    """
    Obtains the sections of the imaging reports and returns only the 
    selected sections (defaults to FINDINGS, IMPRESSION, and ADDENDUM).

    :param documents: Iterable containing documents which are strings
    :param document_ids: Iterable containing document unique identifiers
    :param disable_progress_bar: Flag to disable tqdm progress bar
    :return: Pandas DataFrame containing the columns `document_id`, `text`, `offset`
    """
    processed_documents = []
    for document_id, document in tqdm(zip(document_ids, documents), total=len(documents), disable=disable_progress_bar):
        row ={}
        text, start, end = (document, 0, len(document))
        row['document_id'] = document_id
        row['text'] = text
        row['offset'] = (start, end)

        processed_documents.append(row)

    _df = pd.DataFrame(processed_documents)
    if _df.shape[0] > 0:
        return _df.sort_values(['document_id', 'offset']).reset_index(drop=True)
    else:
        return _df


def sentencize(documents: Iterable[str],
               document_ids: Iterable,
               offsets: Iterable[tuple[int, int]],
               filter_len: int = 3,
               disable_progress_bar: bool = False) -> pd.DataFrame:
    """
    Split a document into sentences. Can be used with `sectionize_documents`
    to further split documents into more manageable pieces. Takes in offsets
    to ensure that after splitting, the sentences can be matched to the
    location in the original documents.

    :param documents: Iterable containing documents which are strings
    :param document_ids: Iterable containing document unique identifiers
    :param offsets: Iterable tuple of the start and end indices
    :param filter_len: Minimum character length of a sentence (otherwise filter out)
    :return: Pandas DataFrame containing the columns `document_id`, `text`, `section`, `offset`
    """

    document_sentences = []
    for document, document_id, offset in tqdm(zip(documents, document_ids, offsets), total=len(documents), disable=disable_progress_bar):
        try:
            _, sentence_offsets = bf.text_to_sentences_and_offsets(document)
            for o in sentence_offsets:
                if o[1]-o[0] > filter_len:
                    sentence = document[o[0]:o[1]]
                    abs_offsets = (o[0]+offset[0], o[1]+offset[0])
                    row ={}
                    row['document_id'] = document_id
                    row['text'] = sentence
                    row['offset'] = abs_offsets
                    document_sentences.append(row)
        except:
            continue
    return pd.DataFrame(document_sentences)



def process(slice_num):


    # SIM_MODEL = '/kaggle/input/sentencetransformers-allminilml6v2/sentence-transformers_all-MiniLM-L6-v2'
    SIM_MODEL = 'sentence-transformers/all-MiniLM-L6-v2'
    DEVICE = 0
    MAX_LENGTH = 512
    BATCH_SIZE = 16

    WIKI_PATH = "/kaggle/input/wikipedia-20230701"
    WIKI_PATH = "/home/viktor/Documents/kaggle/kaggle_llm/data/kaggle-datasets/wikipedia-2023-07-faiss-index"
    wiki_files = os.listdir(WIKI_PATH)


    # # Relevant Title Retrieval




    # trn = pd.read_csv("/home/viktor/Documents/kaggle/kaggle_llm/data/kaggle-llm-science-exam/train.csv")
    
    trn = pd.read_csv("/home/viktor/Documents/kaggle/kaggle_llm/data/data_dumps/more_questions/more_questions_raw_questions_wiki_sci_1_and_2.csv")
    trn = trn[slice_num*1000: slice_num*1000+1000].reset_index(drop=True)
    if len(trn) == 0:
        return None
    
    trn.fillna("", inplace=True)
    trn





    # # trn = pd.read_csv("/kaggle/input/kaggle-llm-science-exam/train.csv").drop("id", 1)
    # trn = pd.read_csv("/home/viktor/Documents/kaggle/kaggle_llm/data/kaggle-llm-science-exam/train.csv")


    # trn.head()





    model = SentenceTransformer(SIM_MODEL, device='cuda')
    model.max_seq_length = MAX_LENGTH
    model = model.half()





    # sentence_index = read_index("/kaggle/input/wikipedia-2023-07-faiss-index/wikipedia_202307.index")
    sentence_index = read_index("/home/viktor/Documents/kaggle/kaggle_llm/data/kaggle-datasets/wikipedia-2023-07-faiss-index/wikipedia_202307.index")





    prompt_embeddings = model.encode(trn.prompt.values, batch_size=BATCH_SIZE, device=DEVICE, show_progress_bar=True, convert_to_tensor=True, normalize_embeddings=True)
    prompt_embeddings = prompt_embeddings.detach().cpu().numpy()
    _ = gc.collect()





    ## Get the top N_TOP_DOCS pages that are likely to contain the topic of interest
    search_score, search_index = sentence_index.search(prompt_embeddings, N_TOP_DOCS)





    search_score.shape





    ## Save memory - delete sentence_index since it is no longer necessary
    del sentence_index
    del prompt_embeddings
    _ = gc.collect()
    libc.malloc_trim(0)


    # # Getting Sentences from the Relevant Titles




    # df = pd.read_parquet("/kaggle/input/wikipedia-20230701/wiki_2023_index.parquet", columns=['id', 'file'])
    df = pd.read_parquet("/home/viktor/Documents/kaggle/kaggle_llm/data/kaggle-datasets/wikipedia-2023-07-faiss-index/wiki_2023_index.parquet", columns=['id', 'file'])





    ## Get the article and associated file location using the index
    wikipedia_file_data = []

    for i, (scr, idx) in tqdm(enumerate(zip(search_score, search_index)), total=len(search_score)):
        scr_idx = idx
        _df = df.loc[scr_idx].copy()
        _df['prompt_id'] = i
        wikipedia_file_data.append(_df)
    wikipedia_file_data = pd.concat(wikipedia_file_data).reset_index(drop=True)
    wikipedia_file_data = wikipedia_file_data[['id', 'prompt_id', 'file']].drop_duplicates().sort_values(['file', 'id']).reset_index(drop=True)

    ## Save memory - delete df since it is no longer necessary
    del df
    _ = gc.collect()
    libc.malloc_trim(0)





    wikipedia_file_data





    wikipedia_file_data.prompt_id.unique().shape





    ## Get the full text data
    wiki_text_data = []

    for file in tqdm(wikipedia_file_data.file.unique(), total=len(wikipedia_file_data.file.unique())):
        _id = [str(i) for i in wikipedia_file_data[wikipedia_file_data['file']==file]['id'].tolist()]
        _df = pd.read_parquet(f"{WIKI_PATH}/{file}", columns=['id', 'text'])

        _df_temp = _df[_df['id'].isin(_id)].copy()
        del _df
        _ = gc.collect()
        libc.malloc_trim(0)
        wiki_text_data.append(_df_temp)
    wiki_text_data = pd.concat(wiki_text_data).drop_duplicates().reset_index(drop=True)
    _ = gc.collect()





    wiki_text_data





    # # merge wikipedia_file_data with wiki_text_data in id
    # merged_df = pd.merge(wikipedia_file_data, wiki_text_data, on='id', how='inner')

    # # take only prompt_id and text
    # merged_df = merged_df[['prompt_id', 'text']]

    # # group by prompt_id, concatenate all the text
    # merged_df = merged_df.groupby('prompt_id')['text'].apply(lambda x: ' '.join(x)).reset_index()

    # # merge trn with merged_df
    # merged_df = pd.merge(trn, merged_df, left_index=True, right_on='prompt_id', how='inner')

    # merged_df['context'] = merged_df['text']

    # # save to train_with_dense_context.csv
    # merged_df.to_csv('train_with_dense_context.csv', index=False)


    # merged_df





    # def split_text(text, word_limit=1000):

        
    #     chunks = [text[i:i + word_limit] for i in range(0, len(text), word_limit//2)]
    #     return chunks


    #     words = text.split()
    #     chunks = [text[i:i + word_limit] for i in range(0, len(words), word_limit//2)]
    #     return chunks
    #     return [' '.join(chunk) for chunk in chunks]

    # merged_df['context_splitted'] = merged_df['context'].apply(split_text)
    # merged_df = merged_df.explode('context_splitted')
    # merged_df['context'] = merged_df['context_splitted']#.apply(lambda x: x.strip())

    # merged_df.to_csv("train_with_dense_context_exploded.csv")


    # merged_df





    trn





    # wiki_text_data['text_len'] = wiki_text_data['text'].str.len()
    # wiki_text_data





    ## Parse documents into sentences
    processed_wiki_text_data = process_documents(wiki_text_data.text.values, wiki_text_data.id.values)





    processed_wiki_text_data





    ## Get embeddings of the wiki text data
    wiki_data_embeddings = model.encode(processed_wiki_text_data.text,
                                        batch_size=BATCH_SIZE,
                                        device=DEVICE,
                                        show_progress_bar=True,
                                        convert_to_tensor=True,
                                        normalize_embeddings=True)#.half()
    wiki_data_embeddings = wiki_data_embeddings.detach().cpu().numpy()





    _ = gc.collect()





    ## Combine all answers
    trn['answer_all'] = trn.apply(lambda x: " ".join([x['A'], x['B'], x['C'], x['D'], x['E']]), axis=1)

    ## Search using the prompt and answers to guide the search
    trn['prompt_answer_stem'] = trn['prompt'] + " " + trn['answer_all']





    question_embeddings = model.encode(trn.prompt_answer_stem.values, batch_size=BATCH_SIZE, device=DEVICE, show_progress_bar=True, convert_to_tensor=True, normalize_embeddings=True)
    question_embeddings = question_embeddings.detach().cpu().numpy()


    # # Extracting Matching Prompt-Sentence Pairs



    ## List containing just Context

    contexts = []

    for r in tqdm(trn.itertuples(), total=len(trn)):

        prompt_id = r.Index

        prompt_indices = processed_wiki_text_data[processed_wiki_text_data['document_id'].isin(wikipedia_file_data[wikipedia_file_data['prompt_id']==prompt_id]['id'].values)].index.values

        context = ""
        
        if prompt_indices.shape[0] > 0:
            prompt_index = faiss.index_factory(wiki_data_embeddings.shape[1], "Flat")
            prompt_index.add(wiki_data_embeddings[prompt_indices])

            
            ## Get the top matches
            ss, ii = prompt_index.search(question_embeddings, NUM_SENTENCES_INCLUDE)
            for _s, _i in zip(ss[prompt_id], ii[prompt_id]):
                context += processed_wiki_text_data.loc[prompt_indices]['text'].iloc[_i] + " "
            
        contexts.append(context)
    
    
    trn['context'] = contexts
    
    # if "answer" in trn.columns:
    #     trn[["prompt", "context", "A", "B", "C", "D", "E", "answer"]].to_csv(f"./train_data/train_context_{slice_num}.csv", index=False)
    # else:
    #     trn[["prompt", "context", "A", "B", "C", "D", "E"]].to_csv(f"./train_data/train_context_{slice_num}.csv", index=False)
    
    
    if "answer" in trn.columns:
        trn[["prompt", "context", "A", "B", "C", "D", "E", "answer"]].to_csv(f"./train_data_1_and_2/train_context_{slice_num}.csv", index=False)
    else:
        trn[["prompt", "context", "A", "B", "C", "D", "E"]].to_csv(f"./train_data_1_and_2/train_context_{slice_num}.csv", index=False)



















if __name__ == "__main__":
    
    # get slice_num from command line
    slice_num = int(sys.argv[1])
    process(slice_num)
    