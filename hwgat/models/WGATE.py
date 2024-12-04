import torch
import torch.nn as nn
import numpy as np
from timm.models.layers import trunc_normal_
from torch.autograd import Variable
import math

class PositionalEncoding(nn.Module):
    "Implement the PE function."

    def __init__(self, d_model, dropout, max_len=5000):
        super(PositionalEncoding, self).__init__()
        self.dropout = nn.Dropout(p=dropout)

        # Compute the positional encodings once in log space.
        pe = torch.zeros(max_len, d_model)
        position = torch.arange(0, max_len).unsqueeze(1)
        div_term = torch.exp(torch.arange(0, d_model, 2) *
                             -(math.log(10000.0) / d_model))
        pe[:, 0::2] = torch.sin(position * div_term)
        pe[:, 1::2] = torch.cos(position * div_term)
        pe = pe.unsqueeze(0).unsqueeze(2)
        self.register_buffer('pe', pe)

    def forward(self, x):
        x = x + Variable(self.pe[:, :x.size(1)],
                         requires_grad=False)
        return self.dropout(x)
    


def window_partition(x, window_size=16,):
    """
    Args:
        x: (B, H, W, C)
        window_size (int): window size

    Returns:
        windows: (num_windows*B, window_size, window_size, C)
    """
    W = window_size
    B, F, K, ED = x.shape
    nW = K//W
    x = x.reshape(B, F, nW, W, ED).transpose(1, 2).contiguous()  # B, K//W, F, TP, W, ED
    x = x.view(B*nW, F*W, ED)  # B_nW, W_F, ED
    return x



def window_reverse(x, window_size=16, temporal_dim=128, num_kp=64):
    """
    Args:
        x: (B, H, W, C)
        window_size (int): window size

    Returns:
        windows: (num_windows*B, window_size, window_size, C)
    """
    W = window_size
    F, K = temporal_dim, num_kp
    B_nW, F_W, ED = x.shape
    nW = K//W
    B = B_nW // nW
    x = x.reshape(B, nW, F, W, ED).transpose(1, 2).contiguous()  # B, F, K//W, W, ED
    x = x.view(B, F, K, ED)
    return x
    
class MSA(nn.Module):
    def __init__(self, num_heads, dim, adj_mask=None,
            attn_drop=0., proj_drop=0.) -> None:
        super().__init__()
        self.dim = dim
        self.num_heads = num_heads

        assert dim%num_heads == 0, 'dim and number of heads are incompatible'

        head_dim = dim // num_heads
        self.scale = head_dim ** -0.5
        self.adj_mask = adj_mask
        self.qkv = nn.Linear(dim, dim * 3)
        self.attn_drop = nn.Dropout(attn_drop)
        self.proj = nn.Linear(dim, dim)
        self.proj_drop = nn.Dropout(proj_drop)

        self.softmax = nn.Softmax(dim=-1)
    
    def forward(self, x, B, nW, parent):
        B_nW, F_W, ED = x.shape
        qkv = self.qkv(x).reshape(B_nW, F_W, 3, self.num_heads, ED // self.num_heads).permute(2, 0, 3, 1, 4)
        q, k, v = qkv[0], qkv[1], qkv[2]  # make torchscript happy (cannot use tensor as tuple)

        q = q * self.scale

        attn = (q @ k.transpose(-2, -1))
        # print(attn.size(), self.adj_mask.size())
        
        if self.adj_mask is not None:
            adj_mask = getattr(parent, self.adj_mask)
            attn = attn.view(B, nW, *attn.shape[1:]) + adj_mask.unsqueeze(1)
            attn = attn.view(B*nW, *attn.shape[2:])
        
        attn = self.softmax(attn)
        attn = self.attn_drop(attn)

        x = (attn @ v).transpose(1, 2).reshape(B_nW, F_W, ED)
        x = self.proj(x)
        x = self.proj_drop(x)
        return x
    
class FeedForward(nn.Module):
    def __init__(self, in_features, hidden_features=None, out_features=None, act_layer=nn.GELU, drop=0.):
        super().__init__()
        out_features = out_features or in_features
        hidden_features = hidden_features or in_features
        self.fc1 = nn.Linear(in_features, hidden_features)
        self.act = act_layer()
        self.fc2 = nn.Linear(hidden_features, out_features)
        self.drop = nn.Dropout(drop)

    def forward(self, x):
        x = self.fc1(x)
        x = self.act(x)
        x = self.drop(x)
        x = self.fc2(x)
        x = self.drop(x)
        return x
    
class PartAttentionBlock(nn.Module):
    def __init__(self, dim, num_kps=64,
                num_heads=4, window_size=16,
                ff_ratio=4.,
                temporal_dim=128,
                adj_mask=None,
                drop=0., attn_drop=0.,
                act_layer=nn.GELU,
                norm_layer=nn.LayerNorm):
        super().__init__()
        self.dim = dim
        self.num_kps = num_kps
        self.num_heads = num_heads
        self.window_size = window_size
        self.temporal_dim = temporal_dim
        self.drop = drop
        self.ff_dim = self.dim * ff_ratio
        self.attn_drop = attn_drop

        self.norm1 = norm_layer(dim)
        self.attn = MSA(num_heads, dim, adj_mask=adj_mask, attn_drop=attn_drop, proj_drop=drop)
        self.norm2 = norm_layer(dim)
        self.ff = FeedForward(in_features=dim, hidden_features=int(self.ff_dim), act_layer=act_layer, drop=drop)
    
    def forward(self, x, parent):
        W = self.window_size
        B, F, K, ED = x.shape
        nW = K//W
        shortcut = x
        x = window_partition(x, W)
        x = self.attn(self.norm1(x), B, nW, parent)  # B_nW, W_F, ED
        x = window_reverse(x, W, F, K)
        x = shortcut + x
        x = x + self.ff(self.norm2(x))
        return x
    
class Model(nn.Module):
    def __init__(self,
                 kp_dim=26,
                 num_kps=64,
                 temporal_dim=256,
                 num_classes=1000,
                 embed_dim=64,
                 pe=False,
                 depths=16,
                 num_heads=8,
                 window_size=16,
                 ff_ratio=4.,
                 adj_mat=None,
                 drop_rate=0., attn_drop_rate=0.,
                 norm_layer=nn.LayerNorm,
                 device=None,
                 ) -> None:
        super().__init__()
        self.kp_dim = kp_dim
        self.num_kps = num_kps
        self.temporal_dim = temporal_dim
        self.window_size = window_size
        self.num_classes = num_classes
        self.pe = pe
        self.num_heads = num_heads
        self.ff_ratio = ff_ratio
        adj_mask_ = adj_mat.masked_fill(adj_mat == 0, float(-10000)).masked_fill(adj_mat == 1, float(0))
        adj_mask_new = nn.Parameter(adj_mask_, requires_grad=False).to(device)
        self.adj_mask_name = 'adj_mask'
        self.embed_dim = embed_dim
        self.drop_rate = drop_rate
        self.attn_drop_rate = attn_drop_rate
        self.norm_layer = norm_layer
        if self.adj_mask_name in self._buffers:
            del self._buffers[self.adj_mask_name]
        self.register_buffer(self.adj_mask_name, adj_mask_new)
        
        assert self.num_kps%window_size == 0, "window size and number of kps are incompatible"
        
        mapping_size = embed_dim//2
        scale = 10
        
        B_gauss = torch.normal(0.0, 1.0, (mapping_size, self.kp_dim))
        B = B_gauss * scale
        self.B = nn.Parameter(B, requires_grad=False)

        # position encoding
        if self.pe:
            self.pos_encoder = PositionalEncoding(self.embed_dim, self.drop_rate, self.temporal_dim)
        
        # self.pos_drop = nn.Dropout(p=drop_rate)

        # build layers
        self.layers = nn.ModuleList()
        for _ in range(depths):
            layer = PartAttentionBlock(dim=self.embed_dim,
                               num_kps=self.num_kps,
                               num_heads=self.num_heads,
                               window_size=self.window_size,
                               ff_ratio=self.ff_ratio,
                               adj_mask=self.adj_mask_name,
                               drop=self.drop_rate, 
                               attn_drop=self.attn_drop_rate,
                               norm_layer=self.norm_layer)
            self.layers.append(layer)

        self.norm = norm_layer(self.embed_dim)
        self.avgpool = nn.AvgPool1d(self.temporal_dim*self.num_kps)
        # self.weightedAvg = nn.Linear(self.temporal_dim*self.num_kps, 1)
        self.head = nn.Linear(self.embed_dim, num_classes) if num_classes > 0 else nn.Identity()

        self.apply(self._init_weights)

    def _init_weights(self, m):
        if isinstance(m, nn.Linear):
            trunc_normal_(m.weight, std=.02)
            if isinstance(m, nn.Linear) and m.bias is not None:
                nn.init.constant_(m.bias, 0)
        elif isinstance(m, nn.LayerNorm):
            nn.init.constant_(m.bias, 0)
            nn.init.constant_(m.weight, 1.0)

    def forward_features(self, x):
        x_proj = (2.*torch.pi*x) @ self.B.transpose(1,0)
        x_ = torch.cat([torch.sin(x_proj), torch.cos(x_proj)], axis=-1)
        x = x_
        if self.pe:
            x = self.pos_encoder(x)

        for layer in self.layers:
            x = layer(x, self)
        B, F, K, d = x.shape
        x = self.norm(x)  # B F K ED
        x = self.avgpool(x.transpose(1, 3).reshape(B, d, -1)).squeeze(-1)  # B ED
        return x

    def forward(self, x):
        x = self.forward_features(x)
        x = self.head(x)
        return x
