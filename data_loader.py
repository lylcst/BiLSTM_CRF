#-*-coding:utf-8-*- 
# author lyl
import numpy as np
import torch
import torch.nn as nn
from torch.utils.data import Dataset


def prepare_data(samples, vocab, pad, unk, device=None):
    samples = list(map(lambda s: s.strip().split(" "), samples))
    batch_size = len(samples)
    sizes = [len(s) for s in samples]
    max_size = max(sizes)
    x_np = np.full((batch_size, max_size), fill_value=vocab[pad], dtype='int64')
    for i in range(batch_size):
        x_np[i, :sizes[i]] = [vocab[token] if token in vocab else vocab[unk] for token in samples[i]]
    return torch.LongTensor(x_np.T).to(device)

class SequenceLabelingDataset(Dataset):
    def __init__(self, filename):
        self.datas = []
        with open(filename, "rt", encoding="utf-8") as f:
            sent = []
            tags = []
            for line in f.readlines():
                if not line.strip():
                    continue
                token, tag = line.strip().split()
                if token == "。":
                    assert len(sent) == len(tags)
                    if len(sent):
                        self.datas.append((" ".join(sent), " ".join(tags)))
                    sent, tags = [], []
                else:
                    sent.append(token)
                    tags.append(tag)

    def __getitem__(self, idx):
        return self.datas[idx]

    def __len__(self):
        return len(self.datas)