from utils import *
import torch
from torch.utils.data import Dataset
import pickle
from decord import VideoReader
from decord import cpu
from tqdm import tqdm


class CustomDataset_(Dataset):
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
        
        # print(self.vid_tar[vid_name])

        return torch.tensor(vid_feat, dtype=torch.float32), self.vid_tar[vid_name]
    
def get_dataloader_(cfg):

    with open(cfg.vid_split_path, 'rb') as f:
        # {'train':[vid1, vid2, ...], 'val':[], 'test:[]}
        vid_splits = pickle.load(f)
    with open(cfg.vid_class_path, 'rb') as f:
        # {vid1:0, vid2:1, ...}
        vid_cls = pickle.load(f)
    with open(cfg.data_map_path, 'rb') as f:
        # {vid1:path, ...}
        vid_feat = pickle.load(f)

    if cfg.mode == 'test':
        train_dataset = CustomDataset_(
            vid_splits['train'], vid_feat, vid_cls, cfg.src_len, cfg.test_transform)
        val_dataset = CustomDataset_(
            vid_splits['val'], vid_feat, vid_cls, cfg.src_len, cfg.test_transform)
        test_dataset = CustomDataset_(
            vid_splits['test'], vid_feat, vid_cls, cfg.src_len, cfg.test_transform)
    
    else:
        train_dataset = CustomDataset_(
            vid_splits['train'], vid_feat, vid_cls, cfg.src_len, cfg.train_transform)
        val_dataset = CustomDataset_(
            vid_splits['val'], vid_feat, vid_cls, cfg.src_len, cfg.val_transform)
        test_dataset = CustomDataset_(
            vid_splits['test'], vid_feat, vid_cls, cfg.src_len, cfg.test_transform)

    return train_dataset, val_dataset, test_dataset

def decode_classes(target):
    tr_str = str(target)
    num_cls = int(tr_str[0])
    tr_str = tr_str[1:]
    all_cls = []
    for i in range(num_cls):
        all_cls.append(int(tr_str[i*4:i*4+4]))
    return num_cls, all_cls


def evaluate_(model, eval_loader, device, criterion) -> float:
    model.eval()  # turn on evaluation mode
    eval_acc = 0
    length = len(eval_loader)

    with torch.no_grad():
        for idx in tqdm(range(len(eval_loader))):
            data, target = eval_loader[idx]
            if target == 0:
                length -= 1
                continue
            data = data.to(device)
            output = model(data.unsqueeze(dim=0))

            prediction = torch.argsort(output, dim=-1, descending=True).cpu().tolist()[0]
            num_cls, all_cls = decode_classes(target)
            cls_count = 0
            for cls in all_cls:
                if cls in prediction[0:5]:
                    cls_count += 1
            if cls_count == num_cls:
                eval_acc += 1

            # total_correct_k = (prediction[:, 0:k] == target.unsqueeze(dim=-1)).any(dim=-1).float()

            # eval_acc += total_correct_k

    return (eval_acc / length)

def show_final_result_(model, train_loader, val_loader, test_loader, criterion, cfg, k=1):
    model, _, _, _ = load_checkpoint(f"{cfg.save_model_path}_{cfg.postfix}.pt", cfg)

    train_acc = evaluate_(model, train_loader, cfg.device, criterion)
    val_acc = evaluate_(model, val_loader, cfg.device, criterion)
    test_acc = evaluate_(model, test_loader, cfg.device, criterion)

    print('=' * 89)
    print(
        f'train acc {train_acc:5.4f} | val acc {val_acc:5.4f} | test acc {test_acc:5.4f}')
    # print(
    #     f'train loss {train_loss:2.4f} | val loss {val_loss:2.4f} | test loss {test_loss:2.4f}')
    print('=' * 89)