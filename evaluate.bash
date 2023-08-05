#!/bin/bash


source source.bash
rm -rf /home/clay/research/kaggle/kaggle_llm/preds
set -e

root_dir=$( cd -- "$( dirname -- "${BASH_SOURCE[0]}" )" &> /dev/null && pwd )
test_file_path="/home/clay/research/kaggle/kaggle_llm/data/kaggle-llm-science-exam-splitted/test.csv"
processed_test_file_dir="/home/clay/research/kaggle/kaggle_llm/data/data_dumps/context_df_test/"

python3 $root_dir/tools/generate_context.py \
  /home/clay/research/kaggle/kaggle_llm/data/physics_pages_list/physics_pages_formatted.csv \
  $processed_test_file_dir \
  /home/clay/research/kaggle/kaggle_llm/data/sentence_transformer_model \
  --input-path $test_file_path \
  -k 3
test_file_path="$processed_test_file_dir/test.csv"
#test_file_path="/home/clay/research/kaggle/kaggle_llm/data/data_dumps/context_df/train.csv"

python3 $root_dir/tools/predict.py $test_file_path
python3 $root_dir/tools/ensemble.py /home/clay/research/kaggle/kaggle_llm/preds
python3 $root_dir/tools/submission.py /home/clay/research/kaggle/kaggle_llm/tools/ensembled.csv

#python3 $root_dir/tools/evaluate.py \
#  /home/clay/research/kaggle/kaggle_llm/preds \
#  --label-path $test_file_path
#python3 $root_dir/tools/predict.py /home/clay/research/kaggle/kaggle_llm/data/kaggle-llm-science-exam/test.csv
