import os
import sys
import pandas as pd
import numpy as np
from tqdm import tqdm
from sentence_transformers import SentenceTransformer
import faiss
import argparse
import gc
from sklearn.metrics.pairwise import cosine_similarity
from memory_profiler import profile
import math
from joblib import Parallel, delayed
import threading
from queue import Queue

# @profile
def main(wiki_sci_parquets, model_dir, input_csv, out_dir, out_name, k, max_context_len, njobs, faiss_index=None):
    
    
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
    tqdm.pandas()
    csv['prompt_joined'] = csv[['prompt', 'A', 'B', 'C', 'D', 'E']].progress_apply(lambda x: x['prompt'] + " " + x['A'] + " " + x['B'] + " " + x['C'] + " " + x['D'] + " " + x['E'], axis=1)
    tqdm.pandas()
    csv['embd_prompt'] = csv['prompt_joined'].progress_apply(lambda x: model.encode(x, normalize_embeddings=True))
    print("Prompt embeddings computed")

    del model
    gc.collect()


    # build index from --wiki-sci-parquets directory
    parquet_files = [os.path.join(wiki_sci_parquets, f) for f in os.listdir(wiki_sci_parquets) if f.endswith('.parquet')]
    
    
    # Number of csv files
    num_files = math.ceil(len(csv) / 100)

    # Chunk the csv
    csv_chunks = np.array_split(csv, num_files)
    
        
    def process_chunk(chunk_index):
        
        # copy it and delete it after the loop
        csv_chunk = csv_chunks[chunk_index].copy().reset_index(drop=True)
       
        for pfile_indx, parquet_file in enumerate(parquet_files):
            
            # Create dataframe for a given parquet file
            wiki_sci_df = pd.read_parquet(parquet_file)
            sentence_embeddings = np.stack(wiki_sci_df['sentences_embd'].values).astype('float32')
            
            # Create a faiss index for the sentence embeddings
            d = sentence_embeddings.shape[1]  # dimensionality of the data
            index = faiss.IndexFlatIP(d)
            index.add(sentence_embeddings)
                        
            final_sentences = []
            final_distances = []
            
            for indx, embd_prompt in enumerate(csv_chunk['embd_prompt']):
                
                distances, indices = index.search(embd_prompt.reshape(1, -1),  1)
                sentences = [wiki_sci_df['sentences'].values[i] for i in indices[0]]
                
                final_sentences.append(sentences)
                final_distances.append(distances[0])

            column_name_distances = f"parquet_{pfile_indx}_distances"
            column_name_sentences = f"parquet_{pfile_indx}_sentences"
            
            csv_chunk[column_name_sentences] = final_sentences
            csv_chunk[column_name_distances] = final_distances
            
            
            del wiki_sci_df, sentence_embeddings, index, final_sentences, final_distances
            gc.collect()
            
            
        # Now that we have a full csv_chunk with all the sentences, we can create the context
        columns_distances = [c for c in csv_chunk.columns if c.endswith('_distances')]
        columns_distances_n = [int(c.split('_')[1]) for c in columns_distances]
        sorted_indices = np.argsort(columns_distances_n)
        columns_distances = [columns_distances[i] for i in sorted_indices]

        columns_sentences = [c for c in csv_chunk.columns if c.endswith('_sentences')]
        columns_sentences_n = [int(c.split('_')[1]) for c in columns_sentences]
        sorted_indices = np.argsort(columns_sentences_n)
        columns_sentences = [columns_sentences[i] for i in sorted_indices]

        distances_all = np.stack([np.stack(x) for x in csv_chunk[columns_distances].values])
        sentences_all = np.stack([np.stack(x) for x in csv_chunk[columns_sentences].values])

        n_rows, n_parquets, n_sentences = distances_all.shape
        distances_all = np.reshape(distances_all, (n_rows, n_sentences * n_parquets))
        sentences_all = np.reshape(sentences_all, (n_rows, n_sentences * n_parquets))


        csv_chunk = csv_chunk[['id', 'prompt', 'A', 'B', 'C', 'D', 'E', 'answer'] if 'answer' in csv_chunk.columns else ['id', 'prompt', 'A', 'B', 'C', 'D', 'E']]
        gc.collect()
        
        # take the top k sentences
        context_sentences = []
        for i in range(len(csv_chunk)):
            
            
            distances = distances_all[i]
            sentences = sentences_all[i]
                    
            best_sentences = [sentences[j] for j in np.argsort(distances)[-k:]]
            # flip the order of sentences
            best_sentences = best_sentences[::-1]        
            context = '. '.join(best_sentences)
            context = context.replace("..", ".").replace(". .", ". ")
            context = context[:max_context_len]  # truncate context to max_context_len
            context_sentences.append(context)
        
        csv_chunk['context'] = context_sentences
        csv_chunk['prompt_len'] = csv_chunk['prompt'].apply(lambda x: len(x))
        three_d = " ###  "
        csv_chunk['prompt'] = csv_chunk[['context', 'prompt', 'prompt_len']].apply(lambda x: 'Context: '+ x['context'][:max_context_len-x['prompt_len']-len(three_d)] + three_d + x['prompt'], axis=1)
        
        # remove context	prompt_len from csv_chunk
        csv_chunk = csv_chunk.drop(['context', 'prompt_len'], axis=1)
        
        csv_chunk.to_parquet(f"/tmp/chunk_tmp_{chunk_index}.parquet")
            
        
        del csv_chunk
        gc.collect()
    
    def worker():
        while True:
            chunk_index = q.get()
            if chunk_index is None:
                break
            process_chunk(chunk_index)
            q.task_done()    
    
    print(f"[INFO] Running for {len(csv_chunks)} chunks")
    
    # Create the queue and thread pool.
    q = Queue()

    threads = []
    for i in range(njobs):
        t = threading.Thread(target=worker)
        t.start()
        threads.append(t)
    
    for chunk_index in tqdm(range(len(csv_chunks))):
        q.put(chunk_index)
    
    
    # Block until all tasks are done
    q.join()

    # Stop the threads
    for i in range(njobs):
        q.put(None)
    for t in threads:
        t.join()
    
    # Parallel(n_jobs=njobs, verbose=10)(delayed(process_chunk)(chunk_index) for chunk_index, _ in tqdm(enumerate(csv_chunks), total=len(csv_chunks)))    
    
    del csv
    gc.collect()
        
    output_csv = pd.concat([pd.read_parquet(f'/tmp/chunk_tmp_{i}.parquet') for i in range(num_files)])

    
    # save csv to out-dir
    out_name = out_name.replace(".csv", "")
    output_csv.to_csv(os.path.join(out_dir, f"{out_name}.csv"), index=False)


if __name__ == "__main__":
    parser = argparse.ArgumentParser()
    parser.add_argument("--wiki-sci-parquets", type=str, required=False, default=None)
    parser.add_argument("--input-csv", type=str, required=True, default="")
    parser.add_argument("--model-dir", type=str, required=False, default="")
    parser.add_argument("--out-dir", type=str, required=True, default="")
    parser.add_argument("--out-name", type=str, required=True, default="")
    parser.add_argument("-k", type=int, required=True, default=10)
    parser.add_argument("--max-context-len", type=int, required=True, default=1000)  # max length of context
    parser.add_argument("--njobs", type=int, required=False, default=1)  # njobs
    args = parser.parse_args()
    print(args)
    main(**vars(args))
