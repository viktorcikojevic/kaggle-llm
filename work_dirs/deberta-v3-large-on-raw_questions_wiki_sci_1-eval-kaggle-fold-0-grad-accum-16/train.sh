#!/bin/bash

# Full path of the repo
repo_dir="/home/viktor/Documents/kaggle/kaggle_llm"

# Path of the test file
test_file_path=$repo_dir/data/raw_questions_wiki_sci_1-splitted/more_questions_raw_questions_wiki_sci_1_test.csv

# Output directory for the processed test file
processed_test_file_dir=$repo_dir/data/raw_questions_wiki_sci_1-context

# # Generate context for the test file
# python $repo_dir/tools/generate_context.py \
#   $repo_dir/data/kaggle-llm-science-exam-splitted/more_questions_raw_questions_wiki_sci_1_test.csv \
#   $processed_test_file_dir \
#   "BAAI/bge-large-en" \
#   --input-path $test_file_path \
#   -k 3 \
#   --dont-add-context

test_file_path="$processed_test_file_dir/test.csv"

python $repo_dir/tools/train_mcm.py configs/multiple_choice.yaml --work-dir-path ./

