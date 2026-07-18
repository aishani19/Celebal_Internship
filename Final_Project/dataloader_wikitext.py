import os
import torch
import numpy as np

class DataLoaderWikitext:
    def __init__(self, B, T, split="train", data_root="data/wikitext103", process_rank=0, num_processes=1):
        self.B = B
        self.T = T
        self.process_rank = process_rank
        self.num_processes = num_processes
        
        filename = os.path.join(data_root, f"{split}.bin")
        if not os.path.exists(filename):
            raise FileNotFoundError(f"File {filename} not found. Please run prepare_wikitext.py first.")
            
        self.data = np.memmap(filename, dtype=np.uint16, mode='r')
        self.current_position = self.B * self.T * self.process_rank
        
        print(f"Loaded {len(self.data)} tokens from {filename}")

    def next_batch(self):
        B, T = self.B, self.T
        
        if self.current_position + B * T + 1 > len(self.data):
            self.current_position = self.B * self.T * self.process_rank
            
        # Extract chunk and convert to int64 for torch
        buf = self.data[self.current_position:self.current_position + B * T + 1]
        
        # if buf doesn't have enough tokens (e.g. at the very end), wrap around
        if len(buf) < B * T + 1:
            self.current_position = self.B * self.T * self.process_rank
            buf = self.data[self.current_position:self.current_position + B * T + 1]
            
        self.current_position += B * T * self.num_processes
        
        x = torch.from_numpy(buf[:-1].astype(np.int64)).view(B, T)
        y = torch.from_numpy(buf[1:].astype(np.int64)).view(B, T)
        
        return x, y
        
    def reset(self):
        self.current_position = self.B * self.T * self.process_rank
