

# Experiment Log

## 2023-08-28



- [deberta-v3-large-on-raw_questions_wiki_sci_1-eval-kaggle-all-folds-grad-accum-16](https://www.kaggle.com/code/viktorcikojevic/kaggle-llm?scriptVersionId=141219820) submission, publice score: 0.713. Local CV was about 0.77
- [wiki sci 1 with context](https://wandb.ai/viktor-cikojevic/huggingface/reports/Untitled-Report--Vmlldzo1MjUzODM3/edit?firstReport&runsetFilter)  reaches local CV about 0.68. Maybe this is because too much context is given. 
- I've analyzed the dependence of local CV vs the number of training data, using deberta and wiki-sci-1 (a dataset I created with gpt-3.5 and scientific wiki). I get the practically same results for 1k and the whole dataset.
![Alt text](assets/image-evolution.png)
- I've analyzed the dependence of local CV vs the number of training data, using deberta and wiki-sci-1 (a dataset I created with gpt-3.5 and scientific wiki). I get the practically same results for 1k and the whole dataset.


## 2023-08-27


- important insight: On the left image is the distribution of max dot products between encoded Kaggle prompts and wiki_sci texts. On the right image is the same, but I combine our generated dataset from wiki_sci, mixed with the "15k" dataset. You can see that "15k" dataset contains a lof of non-sci texts, which is why the distribution is so different.
![Alt text](assets/image-1.png)

- Viktor: performed Deberta experiments using "teacher" schema, where the model sees all possible answers. Check the report [on wanbd here](https://api.wandb.ai/links/viktor-cikojevic/6rax1t92). My conclusion is that since the local CV scores the same as before, this training scheme doesn't really help.





## 2023-08-24


-    Viktor: performed bunch of Deberta experiments [on wanbd here](https://wandb.ai/viktor-cikojevic/huggingface/reports/Untitled-Report--Vmlldzo1MjIyNzc3). My conclusions:

- gradient accumulation works pretty well in these experiments!
- loss function is aligned with the metric only at the beginning of the training. After the model starts to overfit, you can see that the metric stays the same while the eval loss increases.
- When evaluating of fold 0 (out of 10) of Kaggle dataset, it is the same whether we train on Kaggle data or wiki_sci `data/kaggle-llm-science-exam-splitted/more_questions_raw_questions_wiki_sci_1_test.csv` dataset. This gives me hope that we can continue training on wiki_sci and evaluate on Kaggle data.
- When evaluating on all Kaggle data, the score is ~0.78, while when evaluating on the first fold of Kaggle data, the score is ~0.9. 


## Typical workflow


### Data generation

- [data_llm_generate_question_wiki_sci.ipynb](../notebooks/data_llm_generate_question_wiki_sci.ipynb): generates questions based on all sci wiki texts gpt3.5

### Data splitting

- [notebooks/data_splitter.ipynb](notebooks/data_splitter.ipynb): splits data into train and test. Shuffles the possible answers and saves the data in a csv.



