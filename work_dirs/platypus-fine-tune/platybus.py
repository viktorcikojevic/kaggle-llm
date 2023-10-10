
#!/usr/bin/env python
# coding: utf-8

# # Platypus2-70B + Wikipedia RAG
# 




import gc
import logging
from time import time
from pathlib import Path
from concurrent.futures import ThreadPoolExecutor
import ctypes
from functools import partial


import torch
import numpy as np
import pandas as pd
from tqdm.auto import tqdm

# For RAG
import faiss
import torch.nn.functional as F
from torch.utils.data import DataLoader
from datasets import load_from_disk, Dataset

NUM_TITLES = 5
MAX_SEQ_LEN = 512
# MODEL_PATH = "/kaggle/input/bge-small-faiss/"
MODEL_PATH = "/home/viktor/Documents/kaggle/kaggle_llm/data/kaggle-datasets/bge-small-faiss"

# For LLM
from transformers import AutoConfig, AutoModelForCausalLM, AutoTokenizer, AutoModel
from accelerate import init_empty_weights
from accelerate.utils.modeling import set_module_tensor_to_device
from safetensors.torch import load_file

N_BATCHES = 1 
MAX_CONTEXT = 2300
MAX_LENGTH = 4096




# Function to clean RAM & vRAM
def clean_memory():
    gc.collect()
    ctypes.CDLL("libc.so.6").malloc_trim(0)
    torch.cuda.empty_cache()

# Load data
# df = pd.read_csv("/kaggle/input/kaggle-llm-science-exam/test.csv", index_col="id")
# df = pd.read_csv("/home/viktor/Documents/kaggle/kaggle_llm/work_dirs/160k-viktor-and-deotte-dataset-deotte-preproc-deberta-window-inference/test_with_context.csv", index_col="id")[:30]
df = pd.read_csv("/home/viktor/Documents/kaggle/kaggle_llm/notebooks/generate-v5-dataset/train_with_context.csv")[:10000]

# make all rows string
df = df.astype(str)



# Variable used to avoid running the notebook for 3 hours when submitting. Credit : CPMP
IS_TEST_SET = len(df) != 200

# Uncomment this to see results on the train set
# df = pd.read_csv("/kaggle/input/kaggle-llm-science-exam/train.csv", index_col="id")
# IS_TEST_SET = True
# N_BATCHES = 1


# ## 1. Wikipedia Retrieval Augmented Generation (RAG)
# 
# The following code is adapted from [the notebook of @MGöksu](https://www.kaggle.com/code/mgoksu/0-807-sharing-my-trained-with-context-model) and [the notebook of @MB](https://www.kaggle.com/code/mbanaei/86-2-with-only-270k-articles/notebook). We use the [bge-small-en-v1.5](https://huggingface.co/BAAI/bge-small-en-v1.5) to embed the Wikipedia dataset.



# New SentenceTransformer class similar to the one used in @Mgöksu notebook but relying on the transformers library only

class SentenceTransformer:
    def __init__(self, checkpoint, device="cuda:0"):
        self.device = device
        self.checkpoint = checkpoint
        self.model = AutoModel.from_pretrained(checkpoint).to(self.device).half()
        self.tokenizer = AutoTokenizer.from_pretrained(checkpoint)

    def transform(self, batch):
        tokens = self.tokenizer(batch["text"], truncation=True, padding=True, return_tensors="pt", max_length=MAX_SEQ_LEN)
        return tokens.to(self.device)  

    def get_dataloader(self, sentences, batch_size=32):
        sentences = ["Represent this sentence for searching relevant passages: " + x for x in sentences]
        dataset = Dataset.from_dict({"text": sentences})
        dataset.set_transform(self.transform)
        dataloader = DataLoader(dataset, batch_size=batch_size, shuffle=False)
        return dataloader

    def encode(self, sentences, show_progress_bar=False, batch_size=32):
        dataloader = self.get_dataloader(sentences, batch_size=batch_size)
        pbar = tqdm(dataloader) if show_progress_bar else dataloader

        embeddings = []
        for batch in pbar:
            with torch.no_grad():
                e = self.model(**batch).pooler_output
                e = F.normalize(e, p=2, dim=1)
                embeddings.append(e.detach().cpu().numpy())
        embeddings = np.concatenate(embeddings, axis=0)
        return embeddings




if IS_TEST_SET:
    # Load embedding model
    start = time()
    print(f"Starting prompt embedding, t={time() - start :.1f}s")
    model = SentenceTransformer(MODEL_PATH, device="cuda:0")

    # Get embeddings of prompts
    f = lambda row : " ".join([row["prompt"], row["A"], row["B"], row["C"], row["D"], row["E"]])
    inputs = df.apply(f, axis=1).values # better results than prompt only
    prompt_embeddings = model.encode(inputs, show_progress_bar=False)

    # Search closest sentences in the wikipedia index 
    print(f"Loading faiss index, t={time() - start :.1f}s")
    faiss_index = faiss.read_index(MODEL_PATH + '/faiss.index')
    # faiss_index = faiss.index_cpu_to_all_gpus(faiss_index) # causes OOM, and not that long on CPU

    print(f"Starting text search, t={time() - start :.1f}s")
    search_index = faiss_index.search(np.float32(prompt_embeddings), NUM_TITLES)[1]

    print(f"Starting context extraction, t={time() - start :.1f}s")
    # dataset = load_from_disk("/kaggle/input/all-paraphs-parsed-expanded")
    dataset = load_from_disk("/home/viktor/Documents/kaggle/kaggle_llm/data/kaggle-datasets/all-paraphs-parsed-expanded")
    for i in range(len(df)):
        df.loc[i, "context"] = "-" + "\n-".join([dataset[int(j)]["text"] for j in search_index[i]])

    # Free memory
    faiss_index.reset()
    del faiss_index, prompt_embeddings, model, dataset
    clean_memory()
    print(f"Context added, t={time() - start :.1f}s")


# ## 2: Run Platypus2-70B
# 
# To run such a large model on a single T4 GPU, we run it layer by layer and sample by sample



# Create symlinks from kaggle datasets to fake cached model

checkpoint_path = Path("./cache")
checkpoint_path.mkdir(exist_ok=True, parents=True)

for part in [1]:
    # source_dir = Path(f"/kaggle/input/platypus2-70b-instruct-part{part}")
    source_dir = Path(f"/home/viktor/Documents/kaggle/kaggle_llm/data/kaggle-datasets/platybus")
    for path in source_dir.glob("*"):
        try:
            (checkpoint_path / path.name).symlink_to(path)
        except:
            pass




# Class for sharded llama

class ShardedLlama:
    def __init__(self, checkpoint_path, device="cuda:0", dtype=torch.float16):
        """
        Sharded version of LlamaForCausalLM : the model is splitted into layer shards to reduce GPU memory usage.
        During the forward pass, the inputs are processed layer by layer, and the GPU memory is freed after each layer.
        To avoid loading the layers multiple times, we could save all the intermediate activations in RAM, but
        as Kaggle accelerators have more GPU memory than CPU, we simply batch the inputs and keep them on the GPU.

        Parameters
        ----------
        checkpoint_path : str or Path
            path to the checkpoint
        device : str, optional
            device, by default "cuda:0"
        dtype : torch.dtype, optional
            dtype, by default torch.float16
        """
        
        # Save parameters
        self.checkpoint_path = Path(checkpoint_path)
        self.device = device 
        self.dtype = dtype

        # Create model
        self.config = AutoConfig.from_pretrained(self.checkpoint_path)
        
        # For flash attention when Turing architecture will be supported : https://github.com/Dao-AILab/flash-attention/issues/542
        # self.config.auto_map = {"AutoModelForCausalLM" : "togethercomputer/LLaMA-2-7B-32K--modeling_flash_llama.LlamaForCausalLM"} 
        
        self.tokenizer = AutoTokenizer.from_pretrained(checkpoint_path)
        self.tokenizer.pad_token = self.tokenizer.eos_token
        self.tokenizer.padding_side = "right"
        self.init_model()
        # self.layer_names = ["model.embed_tokens"] + [f"model.layers.{i}" for i in range(len(self.model.model.layers))] + ["model.norm", "lm_head"]
        self.layer_names = ["model.embed_tokens"] + [f"model.layers.{i}" for i in range(len(self.model.model.layers))] + ["model.norm", "lm_head"]

    def init_model(self):
    
        # Load meta model (no memory used)
        with init_empty_weights():
            self.model = AutoModelForCausalLM.from_config(self.config, trust_remote_code=True)
            self.model.tie_weights()
            
        # self.layers = [self.model.model.embed_tokens] + list(self.model.model.layers) + [self.model.model.norm, self.model.lm_head]
        self.layers = [self.model.model.embed_tokens] + list(self.model.model.layers) + [self.model.model.norm, self.model.lm_head]
            
        # Move buffers to device (not that much GPU memory used)
        for buffer_name, buffer in self.model.named_buffers():
            set_module_tensor_to_device(self.model, buffer_name, self.device, value=buffer, dtype=self.dtype)

    def load_layer(self, layer_name):
        state_dict = load_file(self.checkpoint_path / (layer_name + ".safetensors"), device=self.device)
        for param_name, param in state_dict.items():
            assert param.dtype != torch.int8, "int8 not supported (need to add fp16_statistics)"
            set_module_tensor_to_device(self.model, param_name, self.device, value=param, dtype=self.dtype)
            
            
    def __call__(self, inputs, answers, output_token):
        # inputs = [(prefix, suffix), ...] with prefix.shape[0] = 1 and suffix.shape[0] = 5
        
        # Reboot the model to make sure buffers are loaded and memory is clean
        del self.model
        clean_memory()
        self.init_model()
        
       # Send batch to device
        batch = [(prefix.to(self.device), suffix.to(self.device)) for prefix, suffix in inputs]
        n_suffixes = len(batch[0][1])
        suffix_eos = [(suffix != self.tokenizer.pad_token_id).sum(1) - 1 for _, suffix in inputs]

        # Create attention mask for the largest input, and position ids to use KV cache
        attention_mask = torch.finfo(self.dtype).min * torch.ones(MAX_LENGTH, MAX_LENGTH)
        attention_mask = attention_mask.triu(diagonal=1)[None, None, ...]
        attention_mask = attention_mask.to(self.device)
        position_ids = torch.arange(MAX_LENGTH, dtype=torch.long, device=self.device)[None, :]

        # get device num
        device_num = int(self.device.split(":")[-1])

        
        with ThreadPoolExecutor() as executor, torch.inference_mode():

            # Load first layer
            #future = executor.submit(self.load_layer, "model.embed_tokens")
            # self.load_layer("model.embed_tokens")

            for i, (layer_name, layer) in tqdm(enumerate(zip(self.layer_names, self.layers)), desc=self.device, total=len(self.layers)):
                
                # print(f"Turns out {layer_name} is index {i}")

                # Wait for previous layer to be loaded and load next layer
                #future.result()
                # if (i + 1) < len(self.layer_names):
                    #future = executor.submit(self.load_layer, self.layer_names[i + 1])
                    # self.load_layer(self.layer_names[i + 1])
                    
                self.load_layer(self.layer_names[i])

                # Run layer
                for j, (prefix, suffix) in enumerate(batch):
                    
                    
                    if "model.layers." in layer_name:
                        # Process with current transformer layer
                        len_p, len_s = prefix.shape[1], suffix.shape[1]
                        new_prefix, (k_cache, v_cache) = layer(prefix, use_cache=True, attention_mask=attention_mask[:, :, -len_p:, -len_p:])
                        pos = position_ids[:, len_p:len_p + len_s].repeat(n_suffixes, 1)
                        attn = attention_mask[:, :, -len_s:, -len_p - len_s:].repeat(n_suffixes, 1, 1, 1)
                        kv_cache = (k_cache.repeat(n_suffixes, 1, 1, 1), v_cache.repeat(n_suffixes, 1, 1, 1))
                        new_suffix = layer(suffix, past_key_value=kv_cache, position_ids=pos, attention_mask=attn)[0]
                        batch[j] = (new_prefix, new_suffix)
                        
                        # layer_number = int(layer_name.split(".")[2])
                        # save new_suffix to disk if layer_number == 
                    
                    else:
                        if layer_name == "model.embed_tokens":
                            batch[j] = (layer(prefix), layer(suffix))
                        elif layer_name == "model.norm":
                            # Only keep the last token at this point
                            input_to_layer = suffix[torch.arange(n_suffixes), suffix_eos[j]][:, None]
                            batch[j] = (None, layer(input_to_layer))
                            
                            suffix = batch[j][1]
                            print(f"input_to_layer.shape = {input_to_layer.shape}")
                            print(f"suffix.min.item = {suffix.min().item()}")
                            print(f"suffix.max.item = {suffix.max().item()}")
                            
                        elif layer_name == "lm_head":
                            # batch[j] = layer(suffix)[:, 0, output_token].detach().cpu().numpy()
                            
                            
                            batch[j] = self.model.lm_head(suffix)[:, 0, output_token].detach().cpu().numpy()
                            
                            # To fine-tune
                            suffix_np = suffix.detach().cpu().numpy()
                            randint = np.random.randint(0, 1000000)
                            answer = answers[j]
                            options = 'ABCDE'
                            options_to_int = {letter: i for i, letter in enumerate(options)}
                            answer = options_to_int[answer]
                            np.save(f"suffixes/suffix_np_{randint}_answer_{answer}", suffix_np)
                            
                            
                        
                # clean current layer
                layer.to("meta")
                clean_memory() # proposed by CPMP
                
        
    
        batch = np.vstack(batch)
        
        return batch
        
        # Get scores
        # return batch


# Run model on the 2 GPUs

def get_tokens(row, tokenizer):
        system_prefix = "Below is an instruction that describes a task, paired with an input that provides further context. Write a response that appropriately completes the request.\n\n### Instruction:\n{instruction}\n\n### Input:\n{input_prefix}"
        instruction = "Your task is to analyze the question and answer below. If the answer is correct, respond yes, if it is not correct respond no. As a potential aid to your answer, background context from Wikipedia articles is at your disposal, even if they might not always be relevant."
        input_prefix = f"Context: {row['context'][:MAX_CONTEXT]}\nQuestion: {row['prompt']}\nProposed answer: "
        prompt_prefix = system_prefix.format(instruction=instruction, input_prefix=input_prefix)
        prefix = tokenizer(prompt_prefix, return_tensors="pt", return_attention_mask=False, truncation=True, max_length=MAX_LENGTH)["input_ids"]
        prompt_suffix = [f"{row[letter]}\n\n### Response:\n" for letter in "ABCDE"]
        suffix = tokenizer(prompt_suffix, return_tensors="pt", return_attention_mask=False, truncation=True, max_length=MAX_LENGTH, padding=True)["input_ids"][:, 1:]
        return prefix, suffix

def run_model(device, df):
    model = ShardedLlama(checkpoint_path, device=f"cuda:{device}")
    f = partial(get_tokens, tokenizer=model.tokenizer)
    inputs = df.apply(f, axis=1).values
    answers = df["answer"].values
    # print(inputs[:2])
    batches = np.array_split(inputs, N_BATCHES)
    answers_batches = np.array_split(answers, N_BATCHES)
    outputs = []
    for i, (batch, answer) in enumerate(zip(batches, answers_batches)):
        model(batch, answer, output_token=4874)

# Run model
if IS_TEST_SET: 
    
    # take only first 1000 samples, boost only part of the test
    n_limit = 0
    n_zeros = len(df) - n_limit
    df = df[:n_zeros]
    
    
    with ThreadPoolExecutor() as executor:
        outputs = run_model(0, df)
        # outputs = list(executor.map(run_model, [0, 1], np.array_split(df, 2))) 
        # outputs = sum(outputs, [])
    
    print(f"final outputs: {outputs}")
    
    outputs = np.vstack(outputs).astype(np.float32)
    
    # save to disk
    np.save("platybus_scores.npy", outputs)
    
    # load from disk
    # outputs = np.load("platybus_scores.npy")
    
