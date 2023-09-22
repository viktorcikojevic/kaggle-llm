#!/bin/bash

# Full path of the repo
repo_dir="/home/viktor/Documents/kaggle/kaggle_llm"


python $repo_dir/tools/train_peft_mcm.py --config configs/multiple_choice.yaml --work-dir-path ./
# python $repo_dir/tools/train_mcm.py configs/multiple_choice.yaml --work-dir-path ./

