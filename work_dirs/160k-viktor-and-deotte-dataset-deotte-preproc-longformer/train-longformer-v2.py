#!/usr/bin/env python
# coding: utf-8



import pandas as pd
import wandb

df_1 = pd.read_csv("/home/viktor/Documents/kaggle/kaggle_llm/work_dirs/reproduce-mgoksu-deotte/train_data_1_and_2_final/train_data_final.csv")
df_2 = pd.read_csv("/home/viktor/Documents/kaggle/kaggle_llm/data/cdeotte-60k-data-with-context-v2/all_12_with_context2.csv")

df = pd.concat([df_1, df_2], axis=0, ignore_index=True)

# REPLACE nan with empty string
df = df.fillna("")

df




df_valid = pd.read_csv("/home/viktor/Documents/kaggle/kaggle_llm/work_dirs/reproduce-mgoksu-deotte/test_data/train_context_0.csv")
df_valid




df['context_len'] = df['context'].apply(lambda x: len(x))

import matplotlib.pyplot as plt

plt.hist(df['context_len'], bins=100, range=(0, 16000));




from transformers import LongformerTokenizer, LongformerForMultipleChoice

tokenizer = LongformerTokenizer.from_pretrained("/home/viktor/Documents/kaggle/kaggle_llm/data/kaggle-datasets/longformer-race-model/longformer_qa_model")
model = LongformerForMultipleChoice.from_pretrained("/home/viktor/Documents/kaggle/kaggle_llm/data/kaggle-datasets/longformer-race-model/longformer_qa_model").cuda()




import torch 
torch.cuda.empty_cache()




def prepare_answering_input(
        tokenizer, 
        question,  
        options,   
        context,   
        max_seq_length=2048,
    ):
    c_plus_q   = context + ' ' + tokenizer.bos_token + ' ' + question
    c_plus_q_4 = [c_plus_q] * len(options)
    tokenized_examples = tokenizer(
        c_plus_q_4, options,
        max_length=max_seq_length,
        padding="longest",
        truncation=False,
        return_tensors="pt",
    )
    input_ids = tokenized_examples['input_ids'].unsqueeze(0)
    attention_mask = tokenized_examples['attention_mask'].unsqueeze(0)
    example_encoded = {
        "input_ids": input_ids.to(model.device.index),
        "attention_mask": attention_mask.to(model.device.index),
    }
    return example_encoded




import numpy as np
import torch
from tqdm import tqdm

contexts = []
predictions = []
submit_ids = []

loss_fct = torch.nn.CrossEntropyLoss()
grad_accum = 256
optimizer = torch.optim.Adam(model.parameters(), lr=2e-6 )

indices = np.arange(df.shape[0])
loss_train = 0

n_epochs = 3


total_step = 0  # Initialize total_step before entering your loops
wandb.init(project="huggingface")

map_3_best = 0

for epoch in range(n_epochs):
    for index in tqdm(indices, total=len(indices)):
        
        
        row = df.iloc[index]
        # question is 'prompt'
        question = row['prompt']
        options = [row['A'], row['B'], row['C'], row['D'], row['E']]
        context = row['context']
        answer = row['answer']

        answer_onehot = np.zeros(5)
        answer_onehot[ord(answer) - ord('A')] = 1

        inputs = prepare_answering_input(
            tokenizer=tokenizer, question=question,
            options=options, context=context[:512],
        )

        logits = model(**inputs).logits
        probability = torch.softmax(logits, dim=-1)

        # apply categorical cross entropy loss
        answer_index = torch.argmax(torch.tensor(answer_onehot)).cuda().long()
        answer_index = answer_index.unsqueeze(0)
        loss = loss_fct(logits, answer_index)

        # backward pass
        loss.backward()
        
        loss_train += loss.item()

        # update each grad_accum steps
        if index % grad_accum == 0 and index > 0:
            total_step += 1
            
            optimizer.step()
            optimizer.zero_grad()
            average_loss_train = loss_train / grad_accum
            loss_train = 0
            
            model.eval()
            torch.cuda.empty_cache()
            # evaluate 
            indices_val = np.arange(df_valid.shape[0])
            map3 = 0
            for index_val in tqdm(indices_val, total=len(indices_val)):
                row = df_valid.iloc[index_val]
                # question is 'prompt'
                question = row['prompt']
                options = [row['A'], row['B'], row['C'], row['D'], row['E']]
                context = row['context']
                answer = row['answer']
                answer_index = ord(answer) - ord('A')
                
                inputs = prepare_answering_input(
                    tokenizer=tokenizer, question=question,
                    options=options, context=context[:4096],
                )
                with torch.no_grad():
                    logits = model(**inputs).logits
                    probability = torch.softmax(logits, dim=-1)
                    # get indices of top 3 predictions
                    top3 = torch.topk(probability, 3, dim=-1).indices.squeeze(0).tolist()
                    
                    # if best answer is answer_index, then map3 += 1
                    # if second best answer is answer_index, then map3 += 1/2
                    # if third best answer is answer_index, then map3 += 1/3
                    for i, ans in enumerate(top3):
                        if ans == answer_index:
                            map3 += 1 / (i + 1)
                
            map3 /= df_valid.shape[0]
            
            
            if map3 > map_3_best:
                map_3_best = map3
                torch.save(model.state_dict(), f"models/longformer-model-best-map3-{map3}.pt")
                
            
            print(f"map3={map3}")
            # log total_step, average_loss_train, map3
            wandb.log({
                "train/global_step": total_step,
                "train/loss": average_loss_train,
                "eval/map3": map3,
            })
            
            
            model.train()


