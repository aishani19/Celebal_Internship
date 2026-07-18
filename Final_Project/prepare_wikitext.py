import os
import numpy as np
import tiktoken
from datasets import load_dataset
from tqdm import tqdm

def process():
    dataset = load_dataset("wikitext", "wikitext-103-raw-v1")
    split_dataset = dataset["train"].train_test_split(test_size=0.0005, seed=2357, shuffle=True)
    split_dataset['val'] = split_dataset.pop('test') # rename test to val
    
    enc = tiktoken.get_encoding("gpt2")
    
    out_dir = os.path.join(os.path.dirname(__file__), 'data', 'wikitext103')
    os.makedirs(out_dir, exist_ok=True)
    
    for split, dset in split_dataset.items():
        arr_len = np.sum([len(enc.encode(example['text'])) for example in dset])
        filename = os.path.join(out_dir, f'{split}.bin')
        dtype = np.uint16 # gpt2 vocab is 50257, so uint16 is sufficient
        arr = np.memmap(filename, dtype=dtype, mode='w+', shape=(arr_len,))
        
        print(f"Writing {split} split to {filename} ({arr_len} tokens)")
        idx = 0
        for example in tqdm(dset):
            tokens = enc.encode(example['text'])
            arr[idx:idx+len(tokens)] = tokens
            idx += len(tokens)
        arr.flush()

if __name__ == '__main__':
    process()
