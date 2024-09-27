import numpy as np
import torch.nn as nn
import torch

class HWGATParams():
    def __init__(self, dataset_params, input_dim, device=None) -> None:
        self.kp_dim=input_dim
        self.num_kps=64
        self.temporal_dim=dataset_params['src_len']
        self.num_classes=dataset_params['num_class']
        self.embed_dim=128
        self.embed_dim_inc_rate=2
        self.temporal_patch_size=2
        self.ape=True
        self.depths=[4, 4, 8]
        self.num_heads=[2, 4, 8]
        self.window_size=16
        self.drop_rate=0.1
        self.attn_drop_rate=0.0
        self.ff_ratio=2.
        self.norm_layer=nn.LayerNorm
        self.kp_norm=True
        self.device=device

        self.edges = [[
                    [0, 1],
                    [0, 2],
                    [0, 3],
                    [3, 4],
                    [4, 5],
                    [5, 6],
                    [6, 7],
                    [6, 8],
                    [8, 9],
                    [8, 10],
                    [6, 10],
                    [10, 11],
                    [10, 12],
                    [6, 12],
                    [12, 13],
                    [12, 14],
                    [14, 15],
                    [6, 14],
                    [7, 9],
                    [9, 11],
                    [11, 13],
                    [13, 15],
                    [7, 15],
                    [7, 11],
                    [7, 13]

                ],
                [
                    [0, 1],
                    [0, 2],
                    [0, 3],
                    [3, 4],
                    [4, 5],
                    [5, 6],
                    [6, 7],
                    [6, 8],
                    [8, 9],
                    [8, 10],
                    [6, 10],
                    [10, 11],
                    [10, 12],
                    [6, 12],
                    [12, 13],
                    [12, 14],
                    [14, 15],
                    [6, 14],
                    [7, 9],
                    [9, 11],
                    [11, 13],
                    [13, 15],
                    [7, 15],
                    [7, 11],
                    [7, 13]
                ],
                [
                    [0, 1],
                    [0, 2],
                    [0, 3],
                    [3, 4],
                    [4, 5],
                    [5, 6],
                    [6, 7],
                    [6, 8],
                    [8, 9],
                    [8, 10],
                    [6, 10],
                    [10, 11],
                    [10, 12],
                    [6, 12],
                    [12, 13],
                    [12, 14],
                    [14, 15],
                    [6, 14],
                    [7, 9],
                    [9, 11],
                    [11, 13],
                    [13, 15],
                    [7, 15],
                    [7, 11],
                    [7, 13]
                ],
                [
                    [0, 1],
                    [0, 2],
                    [0, 3],
                    [3, 4],
                    [4, 5],
                    [5, 6],
                    [6, 7],
                    [6, 8],
                    [8, 9],
                    [8, 10],
                    [6, 10],
                    [10, 11],
                    [10, 12],
                    [6, 12],
                    [12, 13],
                    [12, 14],
                    [14, 15],
                    [6, 14],
                    [7, 9],
                    [9, 11],
                    [11, 13],
                    [13, 15],
                    [7, 15],
                    [7, 11],
                    [7, 13]
                ],]

        self.edge_bias = torch.tensor(self.get_edge_bias(), dtype=torch.float32)
    
    def get_edge_bias(self):
        TP, W, K = self.temporal_patch_size, self.window_size, self.num_kps
        edge_bias_adj = [self.get_adj(i) for i in range(len(self.edges))]
        edge_bias = []
        for w in range(K//W):
            edge_bias_w = []
            for i in range(TP):
                edge_bias_r = []
                for j in range(TP):
                    if i==j:
                        edge_bias_r.append(edge_bias_adj[w])
                    elif abs(i-j) == 1.0:
                        edge_bias_r.append(np.eye(W))
                    else:
                        edge_bias_r.append(np.zeros((W, W)))
                edge_bias_w.append(np.concatenate(edge_bias_r, axis=1))
            edge_bias.append(np.concatenate(edge_bias_w))
        edge_bias = np.array(edge_bias)

        return edge_bias

    def get_adj(self, index):
        temp = np.eye(self.window_size)

        for i in self.edges[index]:
            temp[tuple(i)] = 1
            temp[tuple(i)[::-1]] = 1
        return temp
    
    def get_model_params(self):
        return self.kp_dim, self.num_kps, self.temporal_dim,self.num_classes,self.embed_dim,self.embed_dim_inc_rate,self.temporal_patch_size,self.ape,self.depths,self.num_heads,self.window_size,self.edge_bias,self.drop_rate,self.attn_drop_rate,self.ff_ratio,self.norm_layer,self.kp_norm,self.device

class HWGAT_no_windowParams():
    def __init__(self, dataset_params, input_dim, device=None) -> None:
        self.kp_dim=input_dim
        self.num_kps=29
        self.temporal_dim=dataset_params['src_len']
        self.num_classes=dataset_params['num_class']
        self.embed_dim=64
        self.embed_dim_inc_rate=2
        self.temporal_patch_size=2
        self.ape=True
        self.depths=[12, 4, 2]
        self.num_heads=[2, 4, 8]
        self.drop_rate=0.1
        self.attn_drop_rate=0.1
        self.ff_ratio=2.
        self.norm_layer=nn.LayerNorm
        self.kp_norm=True
        self.device=device

        self.edges = [[
                    [2, 0],
                    [1, 0],
                    [0, 3],
                    [0, 4],
                    [3, 5],
                    [4, 6],
                    [5, 7],
                    [6, 8],
                    [7, 9],
                    [9, 10],
                    [9, 11],
                    [11, 12],
                    [11, 13],
                    [13, 14],
                    [9, 13],
                    [13, 15],
                    [9, 15],
                    [15, 16],
                    [15, 17],
                    [9, 17],
                    [17, 18],
                    [8, 19],
                    [9+10, 10+17],
                    [9+10, 10+10],
                    [9+10, 10+11],
                    [11+10, 10+12],
                    [11+10, 10+13],
                    [13+10, 10+14],
                    [9+10, 10+13],
                    [13+10, 10+15],
                    [9+10, 10+15],
                    [15+10, 10+16],
                    [9+10, 10+17],
                    [15+10, 10+17],
                    [17+10, 10+18],
                    [9+10, 10+17],   
                ]]

        self.edge_bias = torch.tensor(self.get_edge_bias(), dtype=torch.float32)
    
    def get_edge_bias(self):
        TP, K = self.temporal_patch_size, self.num_kps
        edge_bias = []
        for i in range(TP):
            edge_bias_r = []
            for j in range(TP):
                if i==j:
                    edge_bias_r.append([self.get_adj()][0])
                elif abs(i-j) == 1:
                    edge_bias_r.append(np.eye(K))
                else:
                    edge_bias_r.append(np.zeros((K, K)))
            edge_bias.append((np.concatenate(edge_bias_r, axis=1)))
        edge_bias = np.array(np.concatenate(edge_bias, axis=0))
        edge_bias = torch.tensor(edge_bias, dtype=torch.float32)

        return edge_bias

    def get_adj(self):
        temp = np.eye(self.num_kps)
        for i in self.edges[0]:
            temp[tuple(i)] = 1
            temp[tuple(i)[::-1]] = 1  
        return temp
    
    def get_model_params(self):
        return self.kp_dim, self.num_kps, self.temporal_dim,self.num_classes,self.embed_dim,self.embed_dim_inc_rate,self.temporal_patch_size,self.ape,self.depths,self.num_heads,self.edge_bias,self.drop_rate,self.attn_drop_rate,self.ff_ratio,self.norm_layer,self.kp_norm,self.device


class STGCNParams():
    def __init__(self, dataset_params, input_dim, device=None) -> None:
        self.kp_dim=input_dim
        self.num_kps=29
        self.num_classes=dataset_params['num_class']
        self.center = 0
        self.edge_importance_weighting = True
        self.n_out_features=256
        self.dropout_ratio=0.05
        self.batch_norm=False
        self.edges_ = [
                    [2, 0],
                    [1, 0],
                    [0, 3],
                    [0, 4],
                    [3, 5],
                    [4, 6],
                    [5, 7],
                    [6, 8],
                    [7, 9],
                    [9, 10],
                    [9, 11],
                    [11, 12],
                    [11, 13],
                    [13, 14],
                    [9, 13],
                    [13, 15],
                    [9, 15],
                    [15, 16],
                    [15, 17],
                    [9, 17],
                    [17, 18],
                    [8, 19],
                    [9+10, 10+17],
                    [9+10, 10+10],
                    [9+10, 10+11],
                    [11+10, 10+12],
                    [11+10, 10+13],
                    [13+10, 10+14],
                    [9+10, 10+13],
                    [13+10, 10+15],
                    [9+10, 10+15],
                    [15+10, 10+16],
                    [9+10, 10+17],
                    [15+10, 10+17],
                    [17+10, 10+18],
                    [9+10, 10+17],   
                ]
 
    def get_model_params(self):
        return self.kp_dim, self.num_kps, self.center, self.edges_, self.edge_importance_weighting, self.n_out_features, self.num_classes, self.dropout_ratio, self.batch_norm

class DecoupledGCNParams():
    def __init__(self, dataset_params, input_dim, device=None) -> None:
        self.kp_dim=input_dim
        self.num_kps=29
        self.num_classes=dataset_params['num_class']
        self.groups=8
        self.block_size=41
        self.n_out_features=256
        self.dropout_ratio=0
        self.batch_norm=False
        self.edges_ = [
                    [2, 0],
                    [1, 0],
                    [0, 3],
                    [0, 4],
                    [3, 5],
                    [4, 6],
                    [5, 7],
                    [6, 8],
                    [7, 9],
                    [9, 10],
                    [9, 11],
                    [11, 12],
                    [11, 13],
                    [13, 14],
                    [9, 13],
                    [13, 15],
                    [9, 15],
                    [15, 16],
                    [15, 17],
                    [9, 17],
                    [17, 18],
                    [8, 19],
                    [9+10, 10+17],
                    [9+10, 10+10],
                    [9+10, 10+11],
                    [11+10, 10+12],
                    [11+10, 10+13],
                    [13+10, 10+14],
                    [9+10, 10+13],
                    [13+10, 10+15],
                    [9+10, 10+15],
                    [15+10, 10+16],
                    [9+10, 10+17],
                    [15+10, 10+17],
                    [17+10, 10+18],
                    [9+10, 10+17],   
                ]
 
    def get_model_params(self):
        return self.kp_dim, self.num_kps, self.edges_, self.groups, self.block_size, self.n_out_features, self.num_classes, self.dropout_ratio, self.batch_norm
