#!/bin/bash


# https://www.kaggle.com/code/jjinho/open-book-llm-science-exam/input?scriptVersionId=137071419
source source.bash
rm -rf /home/clay/research/kaggle/kaggle_llm/data/data_dumps/context_df
set -e

python3 /home/clay/research/kaggle/kaggle_llm/tools/generate_context.py \
  /home/clay/research/kaggle/kaggle_llm/data/physics_pages_list/physics_pages_formatted.csv \
  /home/clay/research/kaggle/kaggle_llm/data/data_dumps/context_df/ \
  /home/clay/research/kaggle/kaggle_llm/data/sentence_transformer_model \
  -k 3
