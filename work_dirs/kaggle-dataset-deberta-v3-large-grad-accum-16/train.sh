#!/bin/bash


# Full path of the repo
repo_dir="/home/viktor/Documents/kaggle/kaggle_llm"

# Path of the test file
test_file_path=$repo_dir/data/kaggle-llm-science-exam-splitted/test.csv

# Output directory for the processed test file
processed_test_file_dir=$repo_dir/data/kaggle-llm-science-exam-test-context

# # Generate context for the test file
# python $repo_dir/tools/generate_context.py \
#   $repo_dir/data/physics_pages_list/physics_pages_formatted.csv \
#   $processed_test_file_dir \
#   "sentence-transformers/all-MiniLM-L6-v2" \
#   --input-path $test_file_path \
#   -k 3 \
#   --dont-add-context

test_file_path="$processed_test_file_dir/test.csv"

# current directory, where the script is located, full path
work_dir_path=$repo_dir/work_dirs/kaggle-dataset-deberta-v3-large-grad-accum-16

echo "Work dir:"  $work_dir_path

python $repo_dir/tools/train_mcm.py $repo_dir/work_dirs/kaggle-dataset-deberta-v3-large-grad-accum-16/configs/multiple_choice.yaml --work-dir-path $work_dir_path

