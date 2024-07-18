"""
Custom dataset class to load data samples during training and evaluaating the model
"""

import torch
from torch.utils.data import Dataset
import pickle
from decord import VideoReader
from decord import cpu

class CustomDataset(Dataset):
    def __init__(self, vid_splits, vid_data, vid_tar, max_len, transformation):
        super().__init__()
        self.vid_splits = vid_splits
        self.vid_data = vid_data
        self.vid_tar = vid_tar
        self.max_len = max_len
        self.transformation = transformation

    def load_data(self, path):
        if path[-3:] == 'pkl':
            data = pickle.load(open(path, "rb"))
        elif path[-3:] == "mp4":
            data = VideoReader(path, ctx=cpu(0)).asnumpy()
        else:
            data = None
        return data
    
    def __len__(self):
        return len(self.vid_splits)

    def __getitem__(self, index):
        vid_name = self.vid_splits[index]

        vid_feat_temp = self.load_data(self.vid_data[vid_name])

        if self.transformation is not None:
            vid_feat = self.transformation(vid_feat_temp)

        return torch.tensor(vid_feat, dtype=torch.float32), torch.tensor(self.vid_tar[vid_name], dtype=torch.long)
