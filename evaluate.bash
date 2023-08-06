#!/bin/bash


source source.bash
set -e

root_dir=$( cd -- "$( dirname -- "${BASH_SOURCE[0]}" )" &> /dev/null && pwd )
rm -rf $root_dir/preds
rm -rf $root_dir/eval
test_file_path="/home/clay/research/kaggle/kaggle_llm/data/kaggle-llm-science-exam-splitted/test.csv"
processed_test_file_dir="/home/clay/research/kaggle/kaggle_llm/data/data_dumps/context_df_test/"

python3 $root_dir/tools/generate_context.py \
  $root_dir/data/physics_pages_list/physics_pages_formatted.csv \
  $processed_test_file_dir \
  $root_dir/data/sentence_transformer_model \
  --input-path $test_file_path \
  -k 3 \
  --dont-add-context
test_file_path="$processed_test_file_dir/test.csv"

python3 $root_dir/tools/predict.py $test_file_path
python3 $root_dir/tools/ensemble.py $root_dir/preds
python3 $root_dir/tools/submission.py $root_dir/tools/ensembled.csv

python3 $root_dir/tools/evaluate.py \
  $root_dir/preds \
  --label-path $test_file_path
