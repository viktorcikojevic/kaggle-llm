import pandas as pd
import os
from tqdm import tqdm

root_path = "/home/viktor/Documents/kaggle/kaggle_llm/work_dirs/reproduce-mgoksu-deotte/train_data_1_and_2"

files = os.listdir(root_path)
files = [file for file in files if file.endswith(".csv")]
files = [f"{root_path}/{file}" for file in files]

df_final = pd.read_csv(files[0])

for file in tqdm(files[1:], total=len(files[1:])):

    df = pd.read_csv(file)

    # remove rows where answer is not either A, B, C, D or E

    df_final = pd.concat([df_final, df])
    
df_final = df_final.reset_index(drop=True)
df_final = df_final[df_final['answer'].isin(['A', 'B', 'C', 'D', 'E'])]
    
df_final.to_csv(f"train_data_1_and_2_final/train_data_final.csv", index=False)

