import os
import sys
import pandas as pd
import numpy as np
from tqdm import tqdm
from sentence_transformers import SentenceTransformer
import faiss
import argparse
import gc

def main(wiki_sci_parquets, model_dir, input_csv, out_dir, out_name, k, max_context_len):
    
    
    # you save a model like this
    # models_folder = "/home/viktor/Documents/kaggle/kaggle_llm/data/huggingface_hub_models/bge-large-en"
    # model = SentenceTransformer("BAAI/bge-large-en")
    # model.save(models_folder)
    
    # and you load it like this
    
    # models_folder = "/home/viktor/Documents/kaggle/kaggle_llm/data/huggingface_hub_models/bge-large-en"
    # model = SentenceTransformer(models_folder)
    
    model = SentenceTransformer(model_dir)

    # load csv from args.input_csv
    csv = pd.read_csv(input_csv)
    csv = csv.fillna("None") # Weird bug: when loading "None" string, it becomes NaN in the pandas df
    print(csv)

    print("Computing embeddings for prompts ... ")
    csv['prompt_joined'] = csv[['prompt', 'A', 'B', 'C', 'D', 'E']].apply(lambda x: x['prompt'] + " " + x['A'] + " " + x['B'] + " " + x['C'] + " " + x['D'] + " " + x['E'], axis=1)
    tqdm.pandas()
    csv['embd_prompt'] = csv['prompt_joined'].progress_apply(lambda x: model.encode(x, normalize_embeddings=True))

    # build index from --wiki-sci-parquets directory
    parquet_files = [os.path.join(wiki_sci_parquets, f) for f in os.listdir(wiki_sci_parquets) if f.endswith('.parquet')]
    wiki_sci_df = pd.read_parquet(parquet_files[0])

    ##
    
    
    sentence_embeddings = np.stack(wiki_sci_df['sentences_embd'].values).astype('float32')
    sentence_index = faiss.IndexFlatIP(sentence_embeddings.shape[1])
    sentence_index.add(sentence_embeddings)

    sentences = list(wiki_sci_df['sentences'].values)
    
    del wiki_sci_df, sentence_embeddings

    print("Creating faiss embedding index for wiki sentences ... ")
    for file_path in tqdm(parquet_files[1:]):
        df_tmp = pd.read_parquet(file_path)

        sentence_embeddings = np.stack(df_tmp['sentences_embd'].values).astype('float32')
        sentence_index.add(sentence_embeddings)

        # sentences.extend(list(df_tmp['sentences'].values))
        
        del sentence_embeddings, df_tmp
        gc.collect()
        
        
    # # get the number of clusters
    # nlist = sentence_index.ntotal
    # print(f"Number of clusters: {nlist}")
    # # get the number of subquantizers   
    # m = sentence_index.nprobe
    # print(f"Number of subquantizers: {m}")
    # return
        
        
    # d = 1024  # dimension
    # nlist = 1000  # number of clusters
    # m = 16  # number of subquantizers

    # quantizer = faiss.IndexFlatL2(d)
    # sentence_index = faiss.IndexIVFPQ(quantizer, d, nlist, m, 8)
    # # 8 specifies that each sub-vector is encoded as 8 bits

    # sentence_embeddings = np.stack(wiki_sci_df['sentences_embd'].values).astype('float32')
    # sentence_index.train(sentence_embeddings)
    # sentence_index.add(sentence_embeddings)

    # sentences = list(wiki_sci_df['sentences'].values)

    # del wiki_sci_df, sentence_embeddings

    # print("Creating faiss embedding index for wiki sentences ... ")
    # for file_path in tqdm(parquet_files[1:]):
    #     df_tmp = pd.read_parquet(file_path)

    #     sentence_embeddings = np.stack(df_tmp['sentences_embd'].values).astype('float32')
    #     sentence_index.train(sentence_embeddings)
    #     sentence_index.add(sentence_embeddings)

    #     sentences.extend(list(df_tmp['sentences'].values))

    #     del sentence_embeddings, df_tmp
    #     gc.collect()



    # iterate over csv and find top k closest sentences
    # for loop, connect the sentences into one big string, put into csv['context']
    contexts = []
    for embd_prompt in tqdm(csv['embd_prompt']):
        distances, indices = sentence_index.search(embd_prompt.reshape(1, -1), k)
        best_sentences = [sentences[i] for i in indices[0]]
        context = '. '.join(best_sentences)
        context = context.replace("..", ".").replace("\n", " ")
        context = context[:max_context_len]  # truncate context to max_context_len
        contexts.append(context)

    csv['context'] = contexts

    # replace csv['prompt'] with
    csv['prompt'] = csv[['context', 'prompt']].apply(lambda x: 'Context: '+ x['context'] + " ###  " + x['prompt'], axis=1)

    # save csv to out-dir
    csv.to_csv(os.path.join(out_dir, f"{out_name}.csv"), index=False)


if __name__ == "__main__":
    parser = argparse.ArgumentParser()
    parser.add_argument("--wiki-sci-parquets", type=str, required=True, default="")
    parser.add_argument("--input-csv", type=str, required=True, default="")
    parser.add_argument("--model-dir", type=str, required=True, default="")
    parser.add_argument("--out-dir", type=str, required=True, default="")
    parser.add_argument("--out-name", type=str, required=True, default="")
    parser.add_argument("-k", type=int, required=True, default=10)
    parser.add_argument("--max-context-len", type=int, required=True, default=1000)  # max length of context
    args = parser.parse_args()
    main(**vars(args))
