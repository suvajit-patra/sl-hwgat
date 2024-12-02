"""
Custom dataset class to load data samples during training and evaluaating the model
"""

import torch
from torch.utils.data import Dataset
import pickle
from decord import VideoReader
from decord import cpu
import numpy as np

class CustomDataset(Dataset):
    def __init__(self, vid_splits, vid_data, vid_tar, max_len, transformation):
        super().__init__()
        self.vid_splits = vid_splits
        self.vid_data = vid_data
        self.vid_tar = vid_tar
        self.max_len = max_len
        self.transformation = transformation

    def load_data(self, data):
        if isinstance(data, np.ndarray):
            return data 
        if data[-3:] == 'pkl':
            return pickle.load(open(data, "rb"))
        if data[-3:] == "mp4":
            return VideoReader(data, ctx=cpu(0)).asnumpy()
        return None
    
    def __len__(self):
        return len(self.vid_splits)

    def __getitem__(self, index):
        vid_name = self.vid_splits[index]

        vid_feat_temp = self.load_data(self.vid_data[vid_name])

        if self.transformation is not None:
            vid_feat = self.transformation(vid_feat_temp)

        return torch.tensor(vid_feat, dtype=torch.float32), torch.tensor(self.vid_tar[vid_name], dtype=torch.long)
