from dataclasses import dataclass
from torch.utils.data import Dataset
import fsspec
import torch

@dataclass
class DataConfig:
    path: str = None
    block_size: int = None
    train_split: float = None
    truncate: float = 1.0

class CharDataset(Dataset):
    def __init__(self, data_config: DataConfig):
        data = fsspec.open(data_config.path).open().read().decode("utf-8")
        data = data[:int(len(data) * data_config.truncate)]

        chars = sorted(list(set(data)))
        data_size, vocab_size = len(data), len(chars)
        print(f"data has {data_size} characters, {vocab_size} unique characters")

        self.stoi = { ch:i for i,ch in enumerate(chars) }
        self.itos = { i:ch for i,ch in enumerate(chars) }

        self.block_size = data_config.block_size
        self.vocab_size = vocab_size

        self.data = data


    def __len__(self):
        return len(self.data) - self.block_size
    
    def __getitem__(self, idx):

        chunk = self.data[idx:idx+self.block_size+1]
        dix = [self.stoi[s] for s in chunk]
        x = torch.tensor(dix[:-1], dtype=torch.long)
        y = torch.tensor(dix[1:], dtype=torch.long)
        return x, y
        
        