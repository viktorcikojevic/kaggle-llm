#!/usr/bin/env python
# coding: utf-8

# # get context
# 
# - context is obtained from the [https://www.kaggle.com/code/mbanaei/86-2-with-only-270k-articles](https://www.kaggle.com/code/mbanaei/86-2-with-only-270k-articles) notebook

# In[1]:


get_ipython().run_cell_magic('writefile', 'get_context.py', '\nRUN_ON_KAGGLE = False\nDEBUG = True\n\nimport numpy as np\nimport pandas as pd \nfrom datasets import load_dataset, load_from_disk\nfrom sklearn.feature_extraction.text import TfidfVectorizer\nimport torch\nfrom transformers import LongformerTokenizer, LongformerForMultipleChoice\nimport transformers\nimport pandas as pd\nimport pickle\nimport numpy as np\nimport matplotlib.pyplot as plt\nfrom tqdm import tqdm\nimport unicodedata\nimport gc\nimport os\n\nstop_words = [\'each\', \'you\', \'the\', \'use\', \'used\',\n                  \'where\', \'themselves\', \'nor\', "it\'s", \'how\', "don\'t", \'just\', \'your\',\n                  \'about\', \'himself\', \'with\', "weren\'t", \'hers\', "wouldn\'t", \'more\', \'its\', \'were\',\n                  \'his\', \'their\', \'then\', \'been\', \'myself\', \'re\', \'not\',\n                  \'ours\', \'will\', \'needn\', \'which\', \'here\', \'hadn\', \'it\', \'our\', \'there\', \'than\',\n                  \'most\', "couldn\'t", \'both\', \'some\', \'for\', \'up\', \'couldn\', "that\'ll",\n                  "she\'s", \'over\', \'this\', \'now\', \'until\', \'these\', \'few\', \'haven\',\n                  \'of\', \'wouldn\', \'into\', \'too\', \'to\', \'very\', \'shan\', \'before\', \'the\', \'they\',\n                  \'between\', "doesn\'t", \'are\', \'was\', \'out\', \'we\', \'me\',\n                  \'after\', \'has\', "isn\'t", \'have\', \'such\', \'should\', \'yourselves\', \'or\', \'during\', \'herself\',\n                  \'doing\', \'in\', "shouldn\'t", "won\'t", \'when\', \'do\', \'through\', \'she\',\n                  \'having\', \'him\', "haven\'t", \'against\', \'itself\', \'that\',\n                  \'did\', \'theirs\', \'can\', \'those\',\n                  \'own\', \'so\', \'and\', \'who\', "you\'ve", \'yourself\', \'her\', \'he\', \'only\',\n                  \'what\', \'ourselves\', \'again\', \'had\', "you\'d", \'is\', \'other\',\n                  \'why\', \'while\', \'from\', \'them\', \'if\', \'above\', \'does\', \'whom\',\n                  \'yours\', \'but\', \'being\', "wasn\'t", \'be\']\n\n\nfrom sklearn.feature_extraction.text import TfidfVectorizer\nimport unicodedata\n\n\ndef SplitList(mylist, chunk_size):\n    return [mylist[offs:offs+chunk_size] for offs in range(0, len(mylist), chunk_size)]\n\n\ndef get_relevant_documents(df_valid):\n    df_chunk_size=800\n    if RUN_ON_KAGGLE:\n        cohere_dataset_filtered = load_from_disk("/home/viktor/Documents/kaggle/kaggle_llm/data/kaggle-datasets/wiki-stem-cohere")\n    else:\n        cohere_dataset_filtered = load_from_disk("/home/viktor/Documents/kaggle/kaggle_llm/data/kaggle-datasets/wiki-stem-cohere")\n    modified_texts = cohere_dataset_filtered.map(lambda example:\n                                             {\'temp_text\':\n                                              unicodedata.normalize("NFKD", f"{example[\'title\']} {example[\'text\']}").replace(\'"\',"")},\n                                             num_proc=2)["temp_text"]\n    \n    all_articles_indices = []\n    all_articles_values = []\n    for idx in tqdm(range(0, df_valid.shape[0], df_chunk_size)):\n        df_valid_ = df_valid.iloc[idx: idx+df_chunk_size]\n    \n        articles_indices, merged_top_scores = retrieval(df_valid_, modified_texts)\n        all_articles_indices.append(articles_indices)\n        all_articles_values.append(merged_top_scores)\n        \n    article_indices_array =  np.concatenate(all_articles_indices, axis=0)\n    articles_values_array = np.concatenate(all_articles_values, axis=0).reshape(-1)\n    \n    top_per_query = article_indices_array.shape[1]\n    articles_flatten = [(\n                         articles_values_array[index],\n                         cohere_dataset_filtered[idx.item()]["title"],\n                         unicodedata.normalize("NFKD", cohere_dataset_filtered[idx.item()]["text"]),\n                        )\n                        for index,idx in enumerate(article_indices_array.reshape(-1))]\n    retrieved_articles = SplitList(articles_flatten, top_per_query)\n    return retrieved_articles\n\n\n\ndef retrieval(df_valid, modified_texts):\n    \n    corpus_df_valid = df_valid.apply(lambda row:\n                                     f\'{row["prompt"]}\\n{row["prompt"]}\\n{row["prompt"]}\\n{row["A"]}\\n{row["B"]}\\n{row["C"]}\\n{row["D"]}\\n{row["E"]}\',\n                                     axis=1).values\n    vectorizer1 = TfidfVectorizer(ngram_range=(1,2),\n                                 token_pattern=r"(?u)\\b[\\w/.-]+\\b|!|/|\\?|\\"|\\\'",\n                                 stop_words=stop_words)\n    vectorizer1.fit(corpus_df_valid)\n    vocab_df_valid = vectorizer1.get_feature_names_out()\n    vectorizer = TfidfVectorizer(ngram_range=(1,2),\n                                 token_pattern=r"(?u)\\b[\\w/.-]+\\b|!|/|\\?|\\"|\\\'",\n                                 stop_words=stop_words,\n                                 vocabulary=vocab_df_valid)\n    vectorizer.fit(modified_texts[:500000])\n    corpus_tf_idf = vectorizer.transform(corpus_df_valid)\n    \n    print(f"length of vectorizer vocab is {len(vectorizer.get_feature_names_out())}")\n\n    chunk_size = 100000\n    top_per_chunk = 30\n    top_per_query = 30\n\n    all_chunk_top_indices = []\n    all_chunk_top_values = []\n\n    for idx in tqdm(range(0, len(modified_texts), chunk_size)):\n        wiki_vectors = vectorizer.transform(modified_texts[idx: idx+chunk_size])\n        temp_scores = (corpus_tf_idf * wiki_vectors.T).toarray()\n        chunk_top_indices = temp_scores.argpartition(-top_per_chunk, axis=1)[:, -top_per_chunk:]\n        chunk_top_values = temp_scores[np.arange(temp_scores.shape[0])[:, np.newaxis], chunk_top_indices]\n\n        all_chunk_top_indices.append(chunk_top_indices + idx)\n        all_chunk_top_values.append(chunk_top_values)\n\n    top_indices_array = np.concatenate(all_chunk_top_indices, axis=1)\n    top_values_array = np.concatenate(all_chunk_top_values, axis=1)\n    \n    merged_top_scores = np.sort(top_values_array, axis=1)[:,-top_per_query:]\n    merged_top_indices = top_values_array.argsort(axis=1)[:,-top_per_query:]\n    articles_indices = top_indices_array[np.arange(top_indices_array.shape[0])[:, np.newaxis], merged_top_indices]\n    \n    return articles_indices, merged_top_scores\n\nif RUN_ON_KAGGLE:\n    if DEBUG:\n        df = pd.read_csv("/kaggle/input/kaggle-llm-science-exam/train.csv", index_col="id")\n    else:\n        df = pd.read_csv("/kaggle/input/kaggle-llm-science-exam/test.csv", index_col="id")\nelse:\n    # df = pd.read_csv("/home/viktor/Documents/kaggle/kaggle_llm/data/kaggle-llm-science-exam/train.csv", index_col="id")\n    df = pd.read_csv("/home/viktor/Documents/kaggle/kaggle_llm/data/data_dumps/more_questions/more_questions_raw_questions_wiki_sci_3.csv", index_col="id").sample(n=2048).reset_index(drop=True)\n\n\nretrieved_articles = get_relevant_documents(df)\ngc.collect()\n\n\ncontexts = []\n\nfor index in tqdm(range(df.shape[0])):\n    row = df.iloc[index]\n    # question is \'prompt\'\n    question = row[\'prompt\']\n    options = [row[\'A\'], row[\'B\'], row[\'C\'], row[\'D\'], row[\'E\']]\n    context = f"{retrieved_articles[index][-4][2]}\\n{retrieved_articles[index][-3][2]}\\n{retrieved_articles[index][-2][2]}\\n{retrieved_articles[index][-1][2]}"\n    contexts.append(context)\n    \ndf[\'context\'] = contexts\ndf.to_parquet("test_with_context.parquet")\n')


# In[2]:


get_ipython().system('python get_context.py')


# In[3]:


import pandas as pd
df = pd.read_parquet("test_with_context.parquet")
# remove rows for which answer is not either A, B, C, D or E. Make direct comparison
df = df[df['answer'].isin(['A', 'B', 'C', 'D', 'E'])]
print(df['answer'].value_counts())

df['context_len'] = df['context'].apply(lambda x: len(x))
import matplotlib.pyplot as plt

plt.hist(df['context_len'], bins=10);


# # llm-science-run-context-2

# In[4]:


import os, time
import gc
import pandas as pd
import numpy as np
import re
from tqdm.auto import tqdm
import blingfire as bf
from __future__ import annotations

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

from scipy.special import softmax


# In[5]:


DEVICE = 0
MAX_LENGTH = 384
BATCH_SIZE = 32

DEBUG = True
# DEBUG = False if len(trn)!=200 else True # If you want to save GPU Quota, check off this comment-out. But cannot get accurate weight on saving notebook
FILTER_LEN = 1 if DEBUG else 9
IND_SEARCH = 1 if DEBUG else 7
NUM_SENTENCES_INCLUDE = 1 if DEBUG else 25
CONTEXT_LEN = 1000 if DEBUG else 2305
VAL_SIZE = 200 if DEBUG else 1500


# In[6]:


test_df = pd.read_parquet("test_with_context.parquet")
test_df = test_df[test_df['answer'].isin(['A', 'B', 'C', 'D', 'E'])]
print(test_df['answer'].value_counts())

test_df.index = list(range(len(test_df)))
test_df['id'] = list(range(len(test_df)))
if DEBUG:
    
    def split_prompt(prompt, max_size=400): 
        """
        Splits a given prompt into chunks of size max_size.
        """
        return [prompt[i:i+max_size] for i in range(0, len(prompt), max_size)]

    # Apply the split_prompt function to each row in the "prompt" column
    test_df["context"] = test_df["context"].apply(lambda x: split_prompt(x))

    # Explode the "prompt" column
    test_df = test_df.explode("context", ignore_index=True)
    
    
    test_df["prompt_and_context"] = test_df["context"].apply(lambda x: x[:CONTEXT_LEN]) + " #### " +  test_df["prompt"]
    
else:
    test_df["prompt_and_context"] = test_df["context"].apply(lambda x: x[:CONTEXT_LEN]) + " #### " +  test_df["prompt"]
    
    
if "answer" not in test_df.columns:
    test_df['answer'] = 'A'


# In[7]:


options = 'ABCDE'
indices = list(range(5))

option_to_index = {option: index for option, index in zip(options, indices)}
index_to_option = {index: option for option, index in zip(options, indices)}

def preprocess(example):
  
    first_sentence = [example['prompt_and_context']] * 5
    second_sentence = []
    for option in options:
        second_sentence.append(example[option])
    
    tokenized_example = tokenizer(first_sentence, second_sentence, truncation='only_first')
    tokenized_example['label'] = option_to_index[example['answer']]
    return tokenized_example


# In[8]:


test_df


# In[9]:


@dataclass
class DataCollatorForMultipleChoice:
    tokenizer: PreTrainedTokenizerBase
    padding: Union[bool, str, PaddingStrategy] = True
    max_length: Optional[int] = None
    pad_to_multiple_of: Optional[int] = None
    
    def __call__(self, features):
        label_name = "label" if 'label' in features[0].keys() else 'labels'
        labels = [feature.pop(label_name) for feature in features]
        batch_size = len(features)
        num_choices = len(features[0]['input_ids'])
        flattened_features = [
            [{k: v[i] for k, v in feature.items()} for i in range(num_choices)] for feature in features
        ]
        flattened_features = sum(flattened_features, [])
        
        batch = self.tokenizer.pad(
            flattened_features,
            padding=self.padding,
            max_length=self.max_length,
            pad_to_multiple_of=self.pad_to_multiple_of,
            return_tensors='pt',
        )
        batch = {k: v.view(batch_size, num_choices, -1) for k, v in batch.items()}
        batch['labels'] = torch.tensor(labels, dtype=torch.int64)
        return batch


# In[10]:


# model_dir = "/kaggle/input/how-to-train-open-book-model-part-1/model_v2"
# model_dir = "/kaggle/input/llm-submissions-viktor/work_dirs/deberta-v3-data-wiki_sci-with-wiki-sentence-context-eval-kaggle-all-folds-grad-accum-128-60k/deberta-v3-large-2023-09-05-07-35-55/checkpoint-3281"
model_dir ="/home/viktor/Documents/kaggle/kaggle_llm/work_dirs/160k-viktor-and-deotte-dataset-deotte-preproc-deberta/deberta-v3-large-2023-09-17-10-00-20/checkpoint-14400"
tokenizer = AutoTokenizer.from_pretrained(model_dir)
model = AutoModelForMultipleChoice.from_pretrained(model_dir).cuda()
model.eval()


# In[11]:


tokenized_test_dataset = Dataset.from_pandas(test_df[['id', 'prompt_and_context', 'A', 'B', 'C', 'D', 'E', 'answer']].drop(columns=['id'])).map(preprocess, remove_columns=['prompt_and_context', 'A', 'B', 'C', 'D', 'E', 'answer'])
# tokenized_test_dataset = tokenized_test_dataset.remove_columns(["__index_level_0__"])
data_collator = DataCollatorForMultipleChoice(tokenizer=tokenizer)
test_dataloader = DataLoader(tokenized_test_dataset, batch_size=1, shuffle=False, collate_fn=data_collator)


# # viktor

# In[12]:


test_predictions_viktor = []


for batch in tqdm(test_dataloader):
    for k in batch.keys():
        batch[k] = batch[k].cuda()
    with torch.no_grad():
        outputs = model(**batch)
    test_predictions_viktor.append(outputs.logits.cpu().detach())
    
test_predictions_viktor = torch.cat(test_predictions_viktor)


# In[13]:


def sigmoid(x):
    return 1 / (1 + np.exp(-x))
    
test_predictions_viktor = sigmoid(test_predictions_viktor).numpy()


# In[14]:


test_predictions_viktor.shape


# In[15]:


test_df['predictions'] = test_predictions_viktor.tolist()
test_df


# In[ ]:


# save test_df as rag_2.parquet
test_df.to_parquet("rag_2.parquet")


# In[ ]:


import pandas as pd
test_df = pd.read_parquet("rag_2.parquet")


# In[16]:


ids = sorted(list(set(test_df['id'].values)))

avgs = []
maxes  = []
answers = []
diffs_maxes = []

for id in ids:
    df_id = test_df[test_df['id']==id].reset_index(drop=True)
    answer = df_id['answer'].values[0]
    answers.append(answer)
    
    predictions = np.vstack(df_id['predictions'].values)
    
    predictions_avg = np.mean(predictions, axis=0)
    predictions_max = np.max(predictions, axis=0)
    
    predictions_diff = predictions - predictions_avg
    predictions_diff_max = np.max(predictions_diff, axis=0)
    
    
    avgs.append(predictions_avg)
    maxes.append(predictions_max)
    diffs_maxes.append(predictions_diff_max)
    
    diffs_maxes_argmax = np.argmax(diffs_maxes, axis=1)
    
    


# In[17]:


df_agg = pd.DataFrame({'id': ids, 'answer': answers, 'avg': avgs, 'max': maxes, 'diff_max': diffs_maxes, 'answers': answers})
df_agg


# In[18]:


df_agg['pred_avg'] = df_agg['avg'].apply(lambda x: index_to_option[np.argmax(x)])
df_agg['pred_max'] = df_agg['max'].apply(lambda x: index_to_option[np.argmax(x)])
df_agg['pred_diff_max'] = df_agg['diff_max'].apply(lambda x: index_to_option[np.argmax(x)])

# 2nd to argmax
df_agg['pred_avg_2'] = df_agg['avg'].apply(lambda x: index_to_option[np.argsort(x)[-2]])
df_agg['pred_max_2'] = df_agg['max'].apply(lambda x: index_to_option[np.argsort(x)[-2]])
df_agg['pred_diff_max_2'] = df_agg['diff_max'].apply(lambda x: index_to_option[np.argsort(x)[-2]])

# 3nd to argmax
df_agg['pred_avg_3'] = df_agg['avg'].apply(lambda x: index_to_option[np.argsort(x)[-3]])
df_agg['pred_max_3'] = df_agg['max'].apply(lambda x: index_to_option[np.argsort(x)[-3]])
df_agg['pred_diff_max_3'] = df_agg['diff_max'].apply(lambda x: index_to_option[np.argsort(x)[-3]])

df_agg  


# In[19]:


np.average(df_agg['pred_avg'] == df_agg['answer']), np.average(df_agg['pred_max'] == df_agg['answer']), np.average(df_agg['pred_diff_max'] == df_agg['answer'])


# In[20]:


np.average(df_agg['pred_avg_2'] == df_agg['answer']), np.average(df_agg['pred_max_2'] == df_agg['answer']), np.average(df_agg['pred_diff_max_2'] == df_agg['answer'])


# In[21]:


np.average(df_agg['pred_avg_3'] == df_agg['answer']), np.average(df_agg['pred_max_3'] == df_agg['answer']), np.average(df_agg['pred_diff_max_3'] == df_agg['answer'])


# In[ ]:




