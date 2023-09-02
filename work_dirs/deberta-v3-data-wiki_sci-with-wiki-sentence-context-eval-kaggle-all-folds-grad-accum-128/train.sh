#!/bin/bash

# Full path of the repo
repo_dir="/home/viktor/Documents/kaggle/kaggle_llm"


dir_to_create=$repo_dir/data/wiki-sci-1-w-sentence-context
mkdir -p $dir_to_create


# input  for train
train_file_1=$repo_dir/data/raw_questions_wiki_sci_1-splitted/more_questions_raw_questions_wiki_sci_1_train.csv
train_file_2=$repo_dir/data/raw_questions_wiki_sci_1-splitted/more_questions_raw_questions_wiki_sci_1_test.csv


echo "train_file_1: $train_file_1"
echo "train_file_2: $train_file_2"

# input for test 

test_file_1=$repo_dir/data/kaggle-llm-science-exam-splitted/train.csv
test_file_2=$repo_dir/data/kaggle-llm-science-exam-splitted/test.csv


echo "test_file_1: $test_file_1"
echo "test_file_2: $test_file_2"




# # # Generate context for the test file
# python $repo_dir/tools/generate_context_wiki_sci.py \
#       --wiki-sci-parquets $repo_dir/data/wikipedia_pages2_w_embd_sentences \
#     --input-csv $test_file_1  \
#     --model-dir $repo_dir/data/huggingface_hub_models/bge-large-en \
#     --out-dir $repo_dir/data/wiki-sci-1-w-sentence-context \
#     --out-name "test_1" \
#     -k 10 \
#     --max-context-len 1000


# # # Generate context for the test file
# python $repo_dir/tools/generate_context_wiki_sci.py \
#       --wiki-sci-parquets $repo_dir/data/wikipedia_pages2_w_embd_sentences \
#     --input-csv $test_file_2  \
#     --model-dir $repo_dir/data/huggingface_hub_models/bge-large-en \
#     --out-dir $repo_dir/data/wiki-sci-1-w-sentence-context \
#     --out-name "test_2" \
#     -k 10 \
#     --max-context-len 1000



# # # Generate context for the test file
# python $repo_dir/tools/generate_context_wiki_sci.py \
#       --wiki-sci-parquets $repo_dir/data/wikipedia_pages2_w_embd_sentences \
#     --input-csv $train_file_1  \
#     --model-dir $repo_dir/data/huggingface_hub_models/bge-large-en \
#     --out-dir $repo_dir/data/wiki-sci-1-w-sentence-context \
#     --out-name "train_1" \
#     -k 10 \
#     --max-context-len 1000

# # # Generate context for the test file
# python $repo_dir/tools/generate_context_wiki_sci.py \
#       --wiki-sci-parquets $repo_dir/data/wikipedia_pages2_w_embd_sentences \
#     --input-csv $train_file_2  \
#     --model-dir $repo_dir/data/huggingface_hub_models/bge-large-en \
#     --out-dir $repo_dir/data/wiki-sci-1-w-sentence-context \
#     --out-name "train_2" \
#     -k 10 \
#     --max-context-len 1000
  



python $repo_dir/tools/train_mcm.py configs/multiple_choice.yaml --work-dir-path ./

