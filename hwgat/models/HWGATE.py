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

def window_partition(x, window_size=16, temporal_patch_size=4):
    W, TP = window_size, temporal_patch_size
    B, F, K, ED = x.shape
    f, nW = F//TP, K//W
    x = x.reshape(B, f, TP, nW, W, ED).transpose(2, 3).contiguous()  # B, F//TP, K//W, TP, W, ED
    x = x.view(B*f*nW, TP*W, ED)  # B_f_nW, W_TP, ED
    return x


def window_reverse(x, window_size=16, temporal_patch_size=4, temporal_dim=128, num_kp=64):
    W, TP = window_size, temporal_patch_size
    F, K = temporal_dim, num_kp
    B_f_nW, W_TP, ED = x.shape
    f, nW = F//TP, K//W
    B = int(B_f_nW / (f * nW))
    x = x.reshape(B, f, nW, TP, W, ED).transpose(2, 3).contiguous()  # B, F//TP, TP, K//W, W, ED
    x = x.view(B, F, K, ED)
    return x

class TemporalMerging(nn.Module):
    def __init__(self, dim, temporal_patch_size):
        super().__init__()
        self.dim = dim
        self.temporal_patch_size = temporal_patch_size
    
    def forward(self, x):
        TP = self.temporal_patch_size
        B, F, K, ED = x.shape
        f = F//TP

        x = x.reshape(B, f, TP, K, ED).transpose(2, 3).contiguous()
        x = x.reshape(B, f, K, -1).contiguous() # B F//TP ED*TP

        return x
    
class MSA(nn.Module):
    def __init__(self, dim, num_heads, adj_mat=None,
            attn_drop=0., proj_drop=0.) -> None:
        super().__init__()
        self.dim = dim
        self.num_heads = num_heads

        assert dim%num_heads == 0, 'dim and number of heads are incompatible'

        head_dim = dim // num_heads
        self.scale = head_dim ** -0.5
        self.adj_mat = adj_mat
        self.qkv = nn.Linear(dim, dim * 3)
        self.attn_drop = nn.Dropout(attn_drop)
        self.proj = nn.Linear(dim, dim)
        self.proj_drop = nn.Dropout(proj_drop)

        self.softmax = nn.Softmax(dim=-1)
    
    def forward(self, x, B, f, nW, mask=None):
        B_f_nW, W_TP, ED = x.shape
        qkv = self.qkv(x).reshape(B_f_nW, W_TP, 3, self.num_heads, ED // self.num_heads).permute(2, 0, 3, 1, 4)
        q, k, v = qkv[0], qkv[1], qkv[2]

        q = q * self.scale

        attn = (q @ k.transpose(-2, -1))

        # making some attn 0 to prevent overfitting
        if self.training:
            attn_copy = attn.detach().clone()
            seed_ = torch.rand(1).item()
            attn_copy = self.softmax(attn_copy)
            index_array = attn_copy > seed_
            index_array = torch.where(index_array == True, 0, 1)
            attn = attn * index_array

        if mask is not None:
            attn = attn.view(B, f*nW, *attn.shape[1:]) * mask.unsqueeze(1).unsqueeze(0)
            attn = attn.view(B*f*nW, *attn.shape[2:])
        
        if self.adj_mat is not None:
            attn = attn.view(B, f*nW, *attn.shape[1:]) * self.adj_mat.unsqueeze(1)
            attn = attn.view(B*f*nW, *attn.shape[2:])
        
        attn = attn.masked_fill(attn == 0, float(-10000))
        attn = self.softmax(attn)
        attn = self.attn_drop(attn)

        x = (attn @ v).transpose(1, 2).reshape(B_f_nW, W_TP, ED)
        x = self.proj(x)
        x = self.proj_drop(x)
        # print('MSA out', x.shape)
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
                temporal_patch_size=4,
                temporal_dim=128,
                shift_size=0,
                adj_mat=None,
                drop=0., attn_drop=0.,
                ff_ratio=4.,
                act_layer=nn.GELU,
                norm_layer=nn.LayerNorm):
        super().__init__()
        self.dim = dim
        self.num_kps = num_kps
        self.num_heads = num_heads
        self.window_size = window_size
        self.temporal_patch_size = temporal_patch_size
        self.temporal_dim = temporal_dim
        self.shift_size = shift_size
        self.drop = drop
        self.attn_drop = attn_drop
        self.ff_dim = dim * ff_ratio
        self.act_layer = act_layer

        self.norm1 = norm_layer(dim)
        self.attn = MSA(dim, num_heads=num_heads, adj_mat=adj_mat,
            attn_drop=attn_drop, proj_drop=drop)

        self.norm2 = norm_layer(dim)
        self.ff = FeedForward(in_features=dim, hidden_features=int(self.ff_dim), act_layer=act_layer, drop=drop)

        if self.shift_size > 0:
            F, K = temporal_dim, num_kps
            frame_mask = torch.zeros((1, F, K, 1))  # 1 H W 1
            t_slices = (slice(0, -temporal_patch_size),
                        slice(-temporal_patch_size, -shift_size),
                        slice(-shift_size, None))

            cnt = 0
            for t in t_slices:
                frame_mask[:, t, :] = cnt
                cnt += 1

            mask_windows = window_partition(frame_mask, window_size, temporal_patch_size).squeeze(2)
            attn_mask = mask_windows.unsqueeze(1) - mask_windows.unsqueeze(2)
            attn_mask = attn_mask.masked_fill(attn_mask != 0, float(0.0)).masked_fill(attn_mask == 0, float(1))
        else:
            attn_mask = None

        self.register_buffer("attn_mask", attn_mask)
    
    def forward(self, x):
        W, TP = self.window_size, self.temporal_patch_size
        B, F, K, ED = x.shape
        f, nW = F//TP, K//W

        shortcut = x

        # cyclic shift
        if self.shift_size > 0:
            shifted_x = torch.roll(x, shifts=-self.shift_size, dims=1)
        else:
            shifted_x = x
        x = window_partition(shifted_x, W, TP)

        x = self.norm1(x)
        # MSA/TS-MSA
        x = self.attn(x, B, f, nW, mask=self.attn_mask)  # B_f_nW, W_TP, ED

        x = window_reverse(x, W, TP, F, K)

        # cyclic shift
        if self.shift_size > 0:
            shifted_x = torch.roll(x, shifts=self.shift_size, dims=1)
            # partition windows
        else:
            shifted_x = x
            # partition windows

        x = shortcut + shifted_x
        # FFN
        x = x + self.ff(self.norm2(x))

        return x

class PartAttentionLayer(nn.Module):
    def __init__(self, dim, temporal_patch_size, temporal_dim, num_kps, depth, num_heads, window_size, adj_mat,
                 drop=0., attn_drop=0., ff_ratio=4., norm_layer=nn.LayerNorm, downsample=None, i_layer=0, device=None):
        super().__init__()
        self.dim = dim
        self.depth = depth
        self.num_heads = num_heads
        self.window_size = window_size
        self.adj_mat = adj_mat.to(device)
        self.i_layer = i_layer

        # build blocks
        self.blocks = nn.ModuleList([
            PartAttentionBlock(dim=dim, num_kps=num_kps,
                                 num_heads=num_heads, window_size=window_size,
                                 temporal_patch_size=temporal_patch_size,
                                 temporal_dim = temporal_dim,
                                 shift_size=0 if (i % 2 == 0) else temporal_patch_size // 2,
                                 adj_mat=self.adj_mat,
                                 drop=drop, attn_drop=attn_drop,
                                 ff_ratio=ff_ratio,
                                 norm_layer=norm_layer)
            for i in range(depth)])

        # temporal merging layer
        if downsample is not None:
            self.downsample = downsample(dim, temporal_patch_size)
        else:
            self.downsample = None

    def forward(self, x):
        for blk in self.blocks:
            x = blk(x)
        if self.downsample is not None:
            x = self.downsample(x)
        return x
    
class Model(nn.Module):
    def __init__(self,
                 kp_dim=26,
                 num_kps=64,
                 temporal_dim=256,
                 num_classes=1000,
                 embed_dim=64,
                 temporal_patch_size=4,
                 pe=False,
                 depths=[2, 2, 6, 2],
                 num_heads=[2, 4, 8, 16],
                 window_size=16,
                 adj_mat=None,
                 drop_rate=0., attn_drop_rate=0., ff_ratio=4.,
                 norm_layer=nn.LayerNorm,
                 device=None,
                 ) -> None:
        super().__init__()
        self.kp_dim = kp_dim
        self.num_kps = num_kps
        self.temporal_dim = temporal_dim
        self.window_size = window_size
        self.num_classes = num_classes
        self.num_layers = len(depths)
        self.pe = pe
        self.adj_mat = adj_mat
        self.embed_dim = embed_dim
        self.num_features = int(embed_dim * 2 ** (self.num_layers - 1))
        self.temporal_out_dim = temporal_dim // temporal_patch_size ** (self.num_layers - 1)

        assert self.num_kps%window_size == 0, "window size and number of kps are incompatible"
        assert self.temporal_dim%temporal_patch_size == 0, "temporal dimension and temporal patch size are incompatible"

        mapping_size = embed_dim//2
        scale = 10
        
        # fourier mapping
        B_gauss = torch.normal(0.0, 1.0, (mapping_size, self.kp_dim))
        B = B_gauss * scale
        self.B = nn.Parameter(B, requires_grad=False)

        # position encoding
        if self.pe:
            self.pos_encoder = PositionalEncoding(embed_dim, drop_rate, temporal_dim)

        # build layers
        self.layers = nn.ModuleList()
        for i_layer in range(self.num_layers):
            if adj_mat is not None:
                adj_mat_t = torch.concatenate([adj_mat for _ in range(temporal_dim // temporal_patch_size ** (i_layer+1))])
            else:
                adj_mat_t = None
            layer = PartAttentionLayer(dim=int(embed_dim * 2 ** i_layer),
                               temporal_patch_size=temporal_patch_size,
                               temporal_dim=temporal_dim//(temporal_patch_size ** i_layer),
                               num_kps=num_kps,
                               depth=depths[i_layer],
                               num_heads=num_heads[i_layer],
                               window_size=window_size,
                               adj_mat=adj_mat_t,
                               drop=drop_rate, attn_drop=attn_drop_rate, ff_ratio=ff_ratio,
                               norm_layer=norm_layer,
                               downsample=TemporalMerging if (i_layer < self.num_layers - 1) else None,
                               i_layer=i_layer,
                               device=device)
            self.layers.append(layer)

        self.norm = norm_layer(self.num_features)
        self.avgpool = nn.AvgPool1d(self.temporal_out_dim * self.num_kps)
        self.head = nn.Linear(self.num_features, num_classes) if num_classes > 0 else nn.Identity()

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
            x = layer(x)

        B, f, K, d = x.shape
        x = self.norm(x)
        x = self.avgpool(x.transpose(1, 3).reshape(B, d, -1)).squeeze(-1)
        return x

    def forward(self, x):
        x = self.forward_features(x)
        x = self.head(x)
        return x
