import csv
import importlib
import os
import numpy as np
import torch
from torch import nn
import pickle
from torch.utils.data import DataLoader
from dataset import CustomDataset
from losses.SmoothCrossEntropy import SmoothedCrossEntropyLoss
from tqdm import tqdm
import matplotlib.pyplot as plt
import shutil
from sklearn.metrics import confusion_matrix, ConfusionMatrixDisplay
            

def get_dataloader(cfg):

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
        train_dataset = CustomDataset(
            vid_splits['train'], vid_feat, vid_cls, cfg.src_len, cfg.test_transform)
        val_dataset = CustomDataset(
            vid_splits['val'], vid_feat, vid_cls, cfg.src_len, cfg.test_transform)
        test_dataset = CustomDataset(
            vid_splits['test'], vid_feat, vid_cls, cfg.src_len, cfg.test_transform)
    
    else:
        train_dataset = CustomDataset(
            vid_splits['train'], vid_feat, vid_cls, cfg.src_len, cfg.train_transform)
        val_dataset = CustomDataset(
            vid_splits['val'], vid_feat, vid_cls, cfg.src_len, cfg.val_transform)
        test_dataset = CustomDataset(
            vid_splits['test'], vid_feat, vid_cls, cfg.src_len, cfg.test_transform)
    
    train_loader = DataLoader(train_dataset, batch_size=cfg.batch_size, num_workers=cfg.n_workers,
                              shuffle=cfg.train_shuffle, pin_memory=cfg.train_pin_memory)
    val_loader = DataLoader(val_dataset, batch_size=cfg.batch_size, num_workers=cfg.n_workers,
                            shuffle=cfg.val_shuffle, pin_memory=cfg.val_pin_memory)
    test_loader = DataLoader(test_dataset, batch_size=cfg.batch_size, num_workers=cfg.n_workers,
                             shuffle=cfg.test_shuffle, pin_memory=cfg.test_pin_memory)


    return train_loader, val_loader, test_loader


def load_model(cfg):
    module = importlib.import_module('models.'+cfg.model_type)
    model = getattr(module, 'Model')(*cfg.model_params.get_model_params())
    model.to(cfg.device)
    return model


def get_criterion(cfg):
    if cfg.criterion_type == "cross_entropy":
        criterion = nn.CrossEntropyLoss()
    elif cfg.criterion_type == "smooth_cross_entropy":
        criterion = SmoothedCrossEntropyLoss()
    elif cfg.criterion_type == "mse":
        criterion = nn.MSELoss()

    return criterion


def get_optimizer(model, cfg):

    if cfg.optimizer_type == "adamw":
        optimizer = torch.optim.AdamW(model.parameters(), lr=cfg.lr)
    elif cfg.optimizer_type == "adam":
        optimizer = torch.optim.Adam(model.parameters(), lr=cfg.lr)
    elif cfg.optimizer_type == "nadam":
        optimizer = torch.optim.NAdam(model.parameters(), lr=cfg.lr)
    elif cfg.optimizer_type == "sgd":
        optimizer = torch.optim.SGD(model.parameters(), lr=cfg.lr)

    return optimizer

def get_scheduler(optimizer, cfg):
    if cfg.scheduler == "CosineAnnealingLR":
        scheduler = torch.optim.lr_scheduler.CosineAnnealingLR(optimizer, T_max=20, last_epoch=-1)
    else:
        scheduler = None
    return scheduler

def train(model, train_loader, device, criterion, optimizer):
    model.train()  # turn on train mode
    total_loss = 0
    train_acc = []
    num_batches = len(train_loader)

    for data, target in train_loader:
        data = data.to(device)
        target = target.to(device)
        output = model(data)
        # print(output)
        loss = criterion(output, target)
        optimizer.zero_grad()
        loss.backward()
        optimizer.step()

        total_loss += loss.item()

        _, y_pred_tags = torch.max(output, dim=-1)  # batch_size, nclass
        correct_pred = (y_pred_tags == target).float()

        train_acc += correct_pred.cpu().tolist()

    return total_loss / num_batches, (sum(train_acc) / len(train_acc))

def evaluate(model, eval_loader, device, criterion, k=1) -> float:
    model.eval()  # turn on evaluation mode
    total_loss = 0
    eval_acc = []
    num_batches = len(eval_loader)

    with torch.no_grad():
        for data, target in eval_loader:
            data = data.to(device)
            target = target.to(device)
            output = model(data)
            loss = criterion(output, target)

            total_loss += loss.item()

            prediction = torch.argsort(output, dim=-1, descending=True)
            total_correct_k = (prediction[:, 0:k] == target.unsqueeze(dim=-1)).any(dim=-1).float()


            # _, y_pred_tags = torch.max(output, dim=-1)  # batch_size, nclass
            # correct_pred = (y_pred_tags == target).float()

            eval_acc += total_correct_k.cpu().tolist()

    return total_loss / num_batches, (sum(eval_acc) / len(eval_acc))

def predictions_plus_true(model, eval_loader, cfg):
    model, _, _, _ = load_checkpoint(f"{cfg.save_model_path}_{cfg.postfix}.pt", cfg)
    model.eval()  # turn on evaluation mode
    y_true, y_pred = [], []

    with torch.no_grad():
        for data, target in eval_loader:
            data = data.to(cfg.device)
            target = target.to(cfg.device)

            output = model(data)

            _, y_pred_tags = torch.max(output, dim=-1)

            y_true += target.tolist()
            y_pred += y_pred_tags.tolist()

    return y_pred, y_true


def save_checkpoint(path, model, optimizer, scheduler, train_acc_list, train_loss_list, val_acc_list, val_loss_list, epoch, lr):

    torch.save({
        'model_state_dict': model.state_dict(),
        'optimizer_state_dict': optimizer.state_dict(),
        'train_loss_list': train_loss_list,
        'val_loss_list': val_loss_list,
        'train_acc_list': train_acc_list,
        'val_acc_list': val_acc_list,
        'epoch': epoch,
        'learning_rate': lr,
        'scheduler': scheduler.state_dict(),
    }, path)

def save_config_model_transform_params(path, cfg):
    pickle.dump(cfg, open(path, 'wb'))
    shutil.copy('configs.py', os.path.join(cfg.out_folder, 'configs.py'))
    shutil.copy(os.path.join('models', cfg.model_type+'.py'), os.path.join(cfg.out_folder, cfg.model_type+'.py'))
    shutil.copy(os.path.join('models', 'model_params.py'), os.path.join(cfg.out_folder, 'model_params.py'))
    shutil.copy('dataTransform.py', os.path.join(cfg.out_folder, 'dataTransform.py'))

def load_weights_from_pretrained(model, pretrained_model_path, cfg):
    ckpt = torch.load(pretrained_model_path, map_location=cfg.device)['model_state_dict']
    ckpt_dict = ckpt.items()
    pretrained_dict = {k.replace("model.", ""): v for k, v in ckpt_dict}

    model_dict = model.state_dict()
    tmp = {}
    print("\n=======Check Weights Loading======")
    print("Weights not used from pretrained file:")
    for k, v in pretrained_dict.items():
        if k in model_dict:
            if pretrained_dict[k].shape == model_dict[k].shape:
                tmp[k] = v
            else:
                tmp[k] = model_dict[k]
                print(k)
        else:
            print(k)
    print("---------------------------")
    print("Weights not loaded into new model:")
    for k, v in model_dict.items():
        if k not in pretrained_dict:
            print(k)
    print("===================================\n")
    del pretrained_dict
    model_dict.update(tmp)
    del tmp
    model.load_state_dict(model_dict)
    model.to(dtype=torch.float)
    return model

def load_checkpoint(path, cfg):
    model = load_model(cfg)
    optimizer = get_optimizer(model, cfg)
    scheduler = get_scheduler(optimizer, cfg)

    if cfg.model_weights is not None:
        path = cfg.model_weights
        model = load_weights_from_pretrained(model, path, cfg)
        return model, optimizer, scheduler, [[], [], [], []]

    checkpoint = torch.load(path, map_location=cfg.device)

    model = load_weights_from_pretrained(model, path, cfg)
    optimizer.load_state_dict(checkpoint['optimizer_state_dict'])
    scheduler.load_state_dict(checkpoint['scheduler'])
    cfg.start_epoch = checkpoint['epoch'] + 1
    train_loss_list = checkpoint['train_loss_list']
    val_loss_list = checkpoint['val_loss_list']
    train_acc_list = checkpoint['train_acc_list']
    val_acc_list = checkpoint['val_acc_list']

    return model, optimizer, scheduler, [train_loss_list, val_loss_list, train_acc_list, val_acc_list]


def run_epochs(model, train_loader, val_loader, criterion, optimizer, scheduler, cfg, early_stopper = None):

    train_loss_list, val_loss_list, train_acc_list, val_acc_list = [], [], [], []
    best_val_loss = 9999
    best_val_acc = 0.0

    if cfg.mode == 'load':  # load previous half trained model
        model, optimizer, scheduler, [train_loss_list, val_loss_list, train_acc_list, val_acc_list]  = load_checkpoint(
            f"{cfg.save_model_path}_{cfg.postfix}.pt", cfg)
        best_val_acc = 0.0 if len(val_acc_list) < 1 else max(val_acc_list)
        best_val_loss = 99999 if len(val_acc_list) < 1 else min(val_loss_list)

    loop = tqdm(range(cfg.start_epoch, cfg.epochs + 1),
                total=(cfg.epochs-cfg.start_epoch))
    for epoch in loop:

        train_loss, train_acc = train(model, train_loader, cfg.device, criterion, optimizer)

        scheduler.step()

        val_loss, val_acc = evaluate(model, val_loader, cfg.device, criterion)

        train_loss_list.append(train_loss)
        train_acc_list.append(train_acc)
        val_loss_list.append(val_loss)
        val_acc_list.append(val_acc)

        if val_loss < best_val_loss:
            best_val_loss = val_loss
            save_checkpoint(f"{cfg.save_model_path}_best_loss.pt", model, optimizer, scheduler,
                            train_acc_list, train_loss_list, val_acc_list, val_loss_list, epoch, cfg.lr)
        if val_acc > best_val_acc:
            best_val_acc = val_acc
            save_checkpoint(f"{cfg.save_model_path}_best_acc.pt", model, optimizer, scheduler,
                            train_acc_list, train_loss_list, val_acc_list, val_loss_list, epoch, cfg.lr)
            
        if epoch > 0 and epoch % cfg.save_interval == 0:
            save_checkpoint(f"{cfg.save_model_path}_{epoch}.pt", model, optimizer, scheduler,
                            train_acc_list, train_loss_list, val_acc_list, val_loss_list, epoch, cfg.lr)

        loop.set_description(f'Epochs')
        loop.set_postfix(tr_ac=f'{train_acc:3.4f}', v_ac=f'{val_acc:3.4f}',
                         tr_ls=f'{train_loss:5.2f}', v_ls=f'{val_loss:5.2f}',
                         min_v_ls=f'{best_val_loss:3.4f}')

        
        plot_results(train_loss_list, val_loss_list, 'loss', cfg.criterion_type, cfg.save_loss_curve_path)
        plot_results(train_acc_list, val_acc_list, 'acc', cfg.criterion_type, cfg.save_acc_curve_path)

        if cfg.early_stopping and early_stopper.early_stop(val_acc):             
            break
    
def plot_results(train_list, val_list, option, criterion_type, save_path):
    x = list(range(0, len(train_list)))
    plt.grid()
    # Plotting all lines with specifying labels
    plt.plot(x, train_list, label=f'train {option}')
    plt.plot(x, val_list, label=f'validation {option}')
    # Adding legend, x and y labels, and titles for the lines
    plt.legend()
    plt.xlabel('epochs')
    plt.ylabel(criterion_type)
    plt.title(f'{option} curve')
    # Displaying the plot
    plt.savefig(save_path)

    plt.cla()
    plt.close()

def show_final_result(model, train_loader, val_loader, test_loader, criterion, cfg, k=1):
    model, _, _, _ = load_checkpoint(f"{cfg.save_model_path}_{cfg.postfix}.pt", cfg)

    train_loss, train_acc = evaluate(
        model, train_loader, cfg.device, criterion, k)
    val_loss, val_acc = evaluate(model, val_loader, cfg.device, criterion, k)
    test_loss, test_acc = evaluate(model, test_loader, cfg.device, criterion, k)

    print('=' * 89)
    print(
        f'train acc {train_acc:5.4f} | val acc {val_acc:5.4f} | test acc {test_acc:5.4f}')
    print(
        f'train loss {train_loss:2.4f} | val loss {val_loss:2.4f} | test loss {test_loss:2.4f}')
    print('=' * 89)

def gen_cm_w(model, test_loader, cfg):
    y_pred, y_true = predictions_plus_true(model, test_loader, cfg)

    class_map = {}
    reader = csv.reader(open(cfg.class_map_path))
    header = next(reader)
    for row in reader:
        class_map[int(row[0])] = row[1]

    # cm = confusion_matrix(y_true, y_pred)
    cm = np.zeros((len(class_map), len(class_map)))
    for tr, pr in zip(y_true, y_pred):
        cm[tr, pr] += 1
    
    with open(cfg.save_cm_path, 'w') as file:
        writer_object = csv.writer(file)
        header = ['word', 'total', 'predicted']
        writer_object.writerow(header)
        for i, row in enumerate(cm):
            total = 0
            predicted = ''
            for idx, col in enumerate(row):
                total += col
                if col > 0:
                    predicted += ('word-'+str(class_map[idx])+'('+str(col)+') ')
            row = ['Word-'+str(class_map[i]), str(total), predicted]
            writer_object.writerow(row)


class EarlyStopper:
    def __init__(self, patience=1, min_delta=0):
        self.patience = patience
        self.min_delta = min_delta
        self.counter = 0
        self.max_validation_acc = 0.0

    def early_stop(self, validation_acc):
        if validation_acc > self.max_validation_acc:
            self.max_validation_acc = validation_acc
            self.counter = 0
        elif validation_acc < (self.max_validation_acc + self.min_delta):
            self.counter += 1
            if self.counter >= self.patience:
                return True
        return False
    