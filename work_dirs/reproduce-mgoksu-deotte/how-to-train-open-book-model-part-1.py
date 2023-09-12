#!/usr/bin/env python
# coding: utf-8

# # How To Train Model for Open Book Q&A Technique
# In this notebook we demonstrate how to train a model to be used with top scoring Open Book Q&A method. The Open Book method was first presented by JJ (@jjinho) [here][1], then Quangteo (@quangbk) improved RAM usage [here][2], and Anil (@nlztrk) combined with Q&A [here][3]. Radek (@radek1) demonstrated the strength of Q&A [here][5]. Next Mgoksu (@mgoksu) demonstrated how to achieve top public LB=0.807 using this method [here][4] by finetuning DeBerta large on this method.
# 
# In order to train a model for use with Open Book Q&A, we need a CSV that contains; `prompt` (i.e. question), `A, B, C, D, E` (i.e. answer choices), and we need a column of `context` extracted from wikipedia pages for each question. To generate the `context` column, we run Mgoksu's notebook [here][4]. In code cell #5, we load our CSV without `context` column with code `trn = pd.read_csv(OUR_DATASET.CSV)`. Then in code cell #21 our dataset is saved to disk as `test_context.csv` with the column `context` added.
# 
# I have searched and concatenated all publicly shared datasets into one 60k CSV and then ran Mgoksu's notebook with `NUM_TITLES_INCLUDE = 5` and `NUM_SENTENCES_INCLUDE = 20`. This added an additional `context` column. I uploaded the resultant CSV file to a Kaggle dataset [here][6]. If you enjoy the notebook you are reading, please upvote the dataset too. Thanks! 
# 
# ![](https://miro.medium.com/v2/resize:fit:800/format:webp/1*bTGY3fKIgNefQxNsOYpnBw.png)
#  
# (image source [here][7])
# 
# [1]: https://www.kaggle.com/code/jjinho/open-book-llm-science-exam
# [2]: https://www.kaggle.com/code/quangbk/open-book-llm-science-exam-reduced-ram-usage
# [3]: https://www.kaggle.com/code/nlztrk/openbook-debertav3-large-baseline-single-model
# [4]: https://www.kaggle.com/code/mgoksu/0-807-sharing-my-trained-with-context-model
# [5]: https://www.kaggle.com/code/radek1/new-dataset-deberta-v3-large-training
# [6]: https://www.kaggle.com/datasets/cdeotte/60k-data-with-context-v2
# [7]: https://blog.gopenai.com/enrich-llms-with-retrieval-augmented-generation-rag-17b82a96b6f0

# # Load CSV
# We will load 60k CSV of `prompts`, `A,B,C,D,E`, and `context` from my Kaggle dataset [here][1]. This dataset is all publicly shared datasets concatenated then processed with Mgoksu's notebook [here][2] to create a `context` column. (To learn more about the datasets within read my discussion post). This Kaggle dataset also contains competition `train.csv` with added `context` column (to be used as a validation dataset).
# 
# In this train notebook, we have internet turned on and can choose whatever model we wish to download and train. After we finetune this model, we will create a second notebook with the Open Book Q&A technique and load the finetuned model from the output of this notebook. The second notebook will have internet turned off so that it can be submitted to Kaggle's competition.
# 
# [1]: https://www.kaggle.com/datasets/cdeotte/60k-data-with-context-v2
# [2]: https://www.kaggle.com/code/mgoksu/0-807-sharing-my-trained-with-context-model

# In[ ]:


import os
# os.environ["CUDA_VISIBLE_DEVICES"]="0,1"

from typing import Optional, Union
import pandas as pd, numpy as np, torch
from datasets import Dataset
from dataclasses import dataclass
from transformers import AutoTokenizer
from transformers import EarlyStoppingCallback
from transformers.tokenization_utils_base import PreTrainedTokenizerBase, PaddingStrategy
from transformers import AutoModelForMultipleChoice, TrainingArguments, Trainer

VER=2
# TRAIN WITH SUBSET OF 60K
NUM_TRAIN_SAMPLES = 1_024
# PARAMETER EFFICIENT FINE TUNING
# PEFT REQUIRES 1XP100 GPU NOT 2XT4
USE_PEFT = False
# NUMBER OF LAYERS TO FREEZE 
# DEBERTA LARGE HAS TOTAL OF 24 LAYERS
FREEZE_LAYERS = 0
# BOOLEAN TO FREEZE EMBEDDINGS
FREEZE_EMBEDDINGS = True
# LENGTH OF CONTEXT PLUS QUESTION ANSWER
MAX_INPUT = 512
# HUGGING FACE MODEL
MODEL = 'microsoft/deberta-v3-large'


# In[ ]:


df_train = pd.read_csv("/home/viktor/Documents/kaggle/kaggle_llm/data/wiki-sci-2-w-sentence-context/train_1_60k.csv")

def get_context(text):
    x = text.split(" ### ")
    # remove empty strings
    x = [x for x in x if len(x) > 0]

    assert len(x) == 2, f"Unsuccesful prompt splitting . len(x) = {len(x)}, x={x}"
    return x[0]


df_train['context'] = df_train['prompt'].apply(lambda x: get_context(x))

def get_prompt(text):
    x = text.split(" ### ")
    # remove empty strings
    x = [x for x in x if len(x) > 0]
    assert len(x) == 2, f"Unsuccesful prompt splitting . len(x) = {len(x)}"
    return x[1]

df_train['prompt'] = df_train['prompt'].apply(lambda x: get_prompt(x))

df_train = df_train.fillna('')

# remove Unnamed: 0
df_train = df_train.drop(['Unnamed: 0'], axis=1)


# df_train = df_train[:1000]

df_train


# In[ ]:


# df_valid = pd.read_csv('/kaggle/input/60k-data-with-context-v2/train_with_context2.csv')
# print('Validation data size:', df_valid.shape )
# df_valid.head()


# In[ ]:


df_valid_1 = pd.read_csv("/home/viktor/Documents/kaggle/kaggle_llm/data/wiki-sci-2-w-sentence-context/test_1.csv")
df_valid_2 = pd.read_csv("/home/viktor/Documents/kaggle/kaggle_llm/data/wiki-sci-2-w-sentence-context/test_2.csv")

df_valid = pd.concat([df_valid_1, df_valid_2]).reset_index(drop=True)

def get_context(text):
    x = text.split(" ### ")
    # remove empty strings
    x = [x for x in x if len(x) > 0]

    assert len(x) == 2, f"Unsuccesful prompt splitting . len(x) = {len(x)}, x={x}"
    return x[0]


df_valid['context'] = df_valid['prompt'].apply(lambda x: get_context(x))

def get_prompt(text):
    x = text.split(" ### ")
    # remove empty strings
    x = [x for x in x if len(x) > 0]
    assert len(x) == 2, f"Unsuccesful prompt splitting . len(x) = {len(x)}"
    return x[1]

df_valid['prompt'] = df_valid['prompt'].apply(lambda x: get_prompt(x))

df_valid = df_valid.fillna('')


df_valid



# In[ ]:


# df_train = pd.read_csv('/kaggle/input/60k-data-with-context-v2/all_12_with_context2.csv')
# df_train = df_train.drop(columns="source")
# df_train = df_train.fillna('').sample(NUM_TRAIN_SAMPLES)
# print('Train data size:', df_train.shape )
# df_train.head()


# # Data Loader
# Code is from Radek's notebook [here][1] with modifications to the tokenization process.
# 
# [1]: https://www.kaggle.com/code/radek1/new-dataset-deberta-v3-large-training

# In[ ]:


option_to_index = {option: idx for idx, option in enumerate('ABCDE')}
index_to_option = {v: k for k,v in option_to_index.items()}

def preprocess(example):
    first_sentence = [ "[CLS] " + example['context'] ] * 5
    second_sentences = [" #### " + example['prompt'] + " [SEP] " + example[option] + " [SEP]" for option in 'ABCDE']
    tokenized_example = tokenizer(first_sentence, second_sentences, truncation='only_first', 
                                  max_length=MAX_INPUT, add_special_tokens=False)
    tokenized_example['label'] = option_to_index[example['answer']]
    
    return tokenized_example

@dataclass
class DataCollatorForMultipleChoice:
    tokenizer: PreTrainedTokenizerBase
    padding: Union[bool, str, PaddingStrategy] = True
    max_length: Optional[int] = None
    pad_to_multiple_of: Optional[int] = None
    
    def __call__(self, features):
        label_name = 'label' if 'label' in features[0].keys() else 'labels'
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


# In[ ]:


tokenizer = AutoTokenizer.from_pretrained(MODEL)
dataset_valid = Dataset.from_pandas(df_valid)
dataset = Dataset.from_pandas(df_train)
# dataset = dataset.remove_columns(["__index_level_0__"])
dataset


# In[ ]:


tokenized_dataset_valid = dataset_valid.map(preprocess, remove_columns=['prompt', 'context', 'A', 'B', 'C', 'D', 'E', 'answer'])
tokenized_dataset = dataset.map(preprocess, remove_columns=['prompt', 'context', 'A', 'B', 'C', 'D', 'E', 'answer'], num_proc=4)
tokenized_dataset


# # Build Model
# We will use a Hugging Face AutoModelForMultipleChoice. For the list of possible models, see Hugging Face's repository [here][1]. We can optionally use PEFT to accelerate training and use less memory. However i have noticed that validation accuracy is less. (Note that PEFT requires us to use 1xP100 not 2xT4 GPU. I'm not sure why). We can also optionally freeze layers. This also accelerates training and uses less memory. However validation accuracy may become less.
# 
# [1]: https://huggingface.co/models

# In[ ]:


model = AutoModelForMultipleChoice.from_pretrained(MODEL)


# In[ ]:


# # NOTE PEFT REQUIRES US TO USE 1XP100 NOT 2XT4. I'M NOT SURE WHY.
# if USE_PEFT:
#     !pip install --no-index --no-deps /kaggle/input/llm-whls/peft-0.4.0-py3-none-any.whl


# In[ ]:


if USE_PEFT:
    print('We are using PEFT.')
    from peft import LoraConfig, get_peft_model, TaskType
    peft_config = LoraConfig(
        r=8, lora_alpha=4, task_type=TaskType.SEQ_CLS, lora_dropout=0.1, 
        bias="none", inference_mode=False, 
        target_modules=["query_proj", "value_proj"],
        modules_to_save=['classifier','pooler'],
    )
    model = get_peft_model(model, peft_config)
    model.print_trainable_parameters()


# In[ ]:


if FREEZE_EMBEDDINGS:
    print('Freezing embeddings.')
    for param in model.deberta.embeddings.parameters():
        param.requires_grad = False
if FREEZE_LAYERS>0:
    print(f'Freezing {FREEZE_LAYERS} layers.')
    for layer in model.deberta.encoder.layer[:FREEZE_LAYERS]:
        for param in layer.parameters():
            param.requires_grad = False


# # MAP@3 Metric
# The competition metric is MAP@3 therefore we will make a custom code to add to Hugging Face's trainer. Discussion [here][1]
# 
# [1]: https://www.kaggle.com/competitions/kaggle-llm-science-exam/discussion/435602

# In[ ]:


def map_at_3(predictions, labels):
    map_sum = 0
    pred = np.argsort(-1*np.array(predictions),axis=1)[:,:3]
    for x,y in zip(pred,labels):
        z = [1/i if y==j else 0 for i,j in zip([1,2,3],x)]
        map_sum += np.sum(z)
    return map_sum / len(predictions)

def compute_metrics(p):
    predictions = p.predictions.tolist()
    labels = p.label_ids.tolist()
    return {"map@3": map_at_3(predictions, labels)}


# # Train and Save 
# We will now train and save our model using Hugging Face's easy to use trainer. By adjusting the parameters in this notebook, we can achieve `CV MAP@3 = 0.915+` and corresponding single model `LB MAP@3 = 0.830+` wow!
# 
# In we run this notebook outside of Kaggle then we can train longer and with more RAM. If we run this notebook on Kaggle, then we need to use tricks to train models efficiently. Here are some ideas:
# * use fp16 (this speeds up T4 not P100)
# * use gradient_accumlation_steps (this simulates larger batch sizes)
# * use gradient_checkpointing (this uses disk to save RAM)
# * use 2xT4 instead of 1xP100 (this doubles GPUs)
# * freeze model embeddings (this reduces weights to train)
# * freeze some model layers (this reduces weights to train)
# * use PEFT (this reduces weights to train)
# * increase LR and decrease epochs (this reduces work)
# * use smaller models (this reduces weights to train)

# In[ ]:


training_args = TrainingArguments(
    warmup_ratio=0.04, 
    learning_rate=2e-6,
    per_device_train_batch_size=1,
    per_device_eval_batch_size=2,
    num_train_epochs=30,
    report_to='wandb',
    output_dir = f'./checkpoints_{VER}',
    overwrite_output_dir=True,
    fp16=True,
    gradient_accumulation_steps=16,
    logging_steps=200,
    evaluation_strategy='steps',
    eval_steps=200,
    save_strategy="steps",
    save_steps=200,
    load_best_model_at_end=False,
    metric_for_best_model='map@3',
    lr_scheduler_type='cosine',
    weight_decay=0.01,
    save_total_limit=5,
)


# In[ ]:


trainer = Trainer(
    model=model,
    args=training_args,
    tokenizer=tokenizer,
    data_collator=DataCollatorForMultipleChoice(tokenizer=tokenizer),
    train_dataset=tokenized_dataset,
    eval_dataset=tokenized_dataset_valid,
    compute_metrics = compute_metrics,
    #callbacks=[EarlyStoppingCallback(early_stopping_patience=5)],
)

trainer.train()
trainer.save_model(f'model_v{VER}')


# # Verify Saved Model
# During training, we see the MAP@3 validation score above. Let's load the saved model and compute it again here to verify that our model is saved correctly.

# In[ ]:


del model, trainer
if USE_PEFT:
    model = AutoModelForMultipleChoice.from_pretrained(MODEL)
    model = get_peft_model(model, peft_config)
    checkpoint = torch.load(f'model_v{VER}/pytorch_model.bin')
    model.load_state_dict(checkpoint)
else:
    model = AutoModelForMultipleChoice.from_pretrained(f'model_v{VER}')
trainer = Trainer(model=model)


# In[ ]:


test_df = pd.read_csv('/kaggle/input/60k-data-with-context-v2/train_with_context2.csv')
tokenized_test_dataset = Dataset.from_pandas(test_df).map(
        preprocess, remove_columns=['prompt', 'context', 'A', 'B', 'C', 'D', 'E'])

test_predictions = trainer.predict(tokenized_test_dataset).predictions
predictions_as_ids = np.argsort(-test_predictions, 1)
predictions_as_answer_letters = np.array(list('ABCDE'))[predictions_as_ids]
predictions_as_string = test_df['prediction'] = [
    ' '.join(row) for row in predictions_as_answer_letters[:, :3]
]


# # Compute Validation Score

# In[ ]:


# https://www.kaggle.com/code/philippsinger/h2ogpt-perplexity-ranking
import numpy as np
def precision_at_k(r, k):
    """Precision at k"""
    assert k <= len(r)
    assert k != 0
    return sum(int(x) for x in r[:k]) / k

def MAP_at_3(predictions, true_items):
    """Score is mean average precision at 3"""
    U = len(predictions)
    map_at_3 = 0.0
    for u in range(U):
        user_preds = predictions[u].split()
        user_true = true_items[u]
        user_results = [1 if item == user_true else 0 for item in user_preds]
        for k in range(min(len(user_preds), 3)):
            map_at_3 += precision_at_k(user_results, k+1) * user_results[k]
    return map_at_3 / U


# In[ ]:


m = MAP_at_3(test_df.prediction.values, test_df.answer.values)
print( 'CV MAP@3 =',m )

