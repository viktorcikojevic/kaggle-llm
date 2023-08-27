

# Experiment Log



- 2023-08-27, Viktor: performed Deberta experiments using "teacher" schema, where the model sees all possible answers. Check the report [on wanbd here](https://api.wandb.ai/links/viktor-cikojevic/6rax1t92). My conclusions:
    - Local CV scores the same as before, meaning that this training scheme doesn't really help.

- 2023-08-24, Viktor: performed bunch of Deberta experiments [on wanbd here](https://wandb.ai/viktor-cikojevic/huggingface/reports/Untitled-Report--Vmlldzo1MjIyNzc3). My conclusions:

    - gradient accumulation works pretty well in these experiments!
    - loss function is aligned with the metric only at the beginning of the training. After the model starts to overfit, you can see that the metric stays the same while the eval loss increases.
    - When evaluating of fold 0 (out of 10) of Kaggle dataset, it is the same whether we train on Kaggle data or wiki_sci `data/kaggle-llm-science-exam-splitted/more_questions_raw_questions_wiki_sci_1_test.csv` dataset. This gives me hope that we can continue training on wiki_sci and evaluate on Kaggle data.
    - When evaluating on all Kaggle data, the score is ~0.78, while when evaluating on the first fold of Kaggle data, the score is ~0.9. 



## Typical workflow


### Data generation

- [data_llm_generate_question_wiki_sci.ipynb](../notebooks/data_llm_generate_question_wiki_sci.ipynb): generates questions based on all sci wiki texts gpt3.5

### Data splitting

- [notebooks/data_splitter.ipynb](notebooks/data_splitter.ipynb): splits data into train and test. Shuffles the possible answers and saves the data in a csv.



