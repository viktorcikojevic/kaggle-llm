#!/bin/bash


source source.bash
rm -rf /home/clay/research/kaggle/kaggle_llm/preds
python3 /home/clay/research/kaggle/kaggle_llm/tools/predict.py /home/clay/research/kaggle/kaggle_llm/data/kaggle-llm-science-exam-splitted/test.csv
python3 /home/clay/research/kaggle/kaggle_llm/tools/ensemble.py /home/clay/research/kaggle/kaggle_llm/preds
python3 /home/clay/research/kaggle/kaggle_llm/tools/evaluate.py /home/clay/research/kaggle/kaggle_llm/preds
