import torch
import torch.nn as nn
import numpy as np
from timm.models.layers import trunc_normal_
from torch.autograd import Variable
import math
import random

class DropPart(nn.Module):
    "regularisation"

    def __init__(self, drop_prob=0.2, num_kps=64, device=torch.device("cuda:0")):
        super(DropPart, self).__init__()
        self.drop_prob = drop_prob
        self.parts_idx = {
            "head": [0, 1, 2],
            "l_arm": [3, 5, 7],
            "l_hand": list(range(9, 19)),
            "r_arm": [4, 6, 8],
            "r_hand": list(range(19, 29)),
        }

        self.window_idx = self.parts_idx["head"] + self.parts_idx["l_arm"] + self.parts_idx["l_hand"] + \
            self.parts_idx["head"] + self.parts_idx["r_arm"] + self.parts_idx["r_hand"] + \
            self.parts_idx["head"] + self.parts_idx["l_arm"] + self.parts_idx["r_hand"] + \
            self.parts_idx["head"] + self.parts_idx["r_arm"] + self.parts_idx["l_hand"]
        
        self.parts_map = {"head":[], "l_arm": [], "l_hand": [], "r_arm": [], "r_hand": []}


        for idx, val in enumerate(self.window_idx):
            for key in self.parts_idx.keys():
                if val in self.parts_idx[key]:
                    self.parts_map[key].append(idx)

        self.list_ = []
        for key in list(self.parts_map.keys()):
            temp = torch.ones((num_kps, 1), device = device)
            indices = torch.tensor(self.parts_map[key], device=device)
            temp.index_fill_(0, indices, 0.0)
            self.list_.append(temp.unsqueeze(0))

    def forward(self, x):
        if self.training:
            B, F, K, D = x.shape           
            num_frames_drop = int(self.drop_prob*F)
            temp = random.choice(self.list_)
            start_frame = random.choice(list(range(F - num_frames_drop)))
            x[:, start_frame:start_frame+num_frames_drop, :, :] *= temp
        return x
        

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
    """
    Args:
        x: (B, H, W, C)
        window_size (int): window size

    Returns:
        windows: (num_windows*B, window_size, window_size, C)
    """
    W, TP = window_size, temporal_patch_size
    B, F, K, ED = x.shape
    f, nW = F//TP, K//W
    x = x.reshape(B, f, TP, nW, W, ED).transpose(2, 3).contiguous()  # B, F//TP, K//W, TP, W, ED
    x = x.view(B*f*nW, TP*W, ED)  # B_f_nW, W_TP, ED
    return x



def window_reverse(x, window_size=16, temporal_patch_size=4, temporal_dim=128, num_kp=64):
    """
    Args:
        x: (B, H, W, C)
        window_size (int): window size

    Returns:
        windows: (num_windows*B, window_size, window_size, C)
    """
    W, TP = window_size, temporal_patch_size
    F, K = temporal_dim, num_kp
    B_f_nW, W_TP, ED = x.shape
    f, nW = F//TP, K//W
    B = int(B_f_nW / (f * nW))
    x = x.reshape(B, f, nW, TP, W, ED).transpose(2, 3).contiguous()  # B, F//TP, TP, K//W, W, ED
    x = x.view(B, F, K, ED)
    return x

class KpEmbed(nn.Module):
    def __init__(self, kp_dim=26, embed_dim=64, norm_layer=None) -> None:
        super().__init__()
        self.kp_dim = kp_dim
        self.embed_dim = embed_dim

        self.proj = nn.Linear(in_features=self.kp_dim, out_features=embed_dim)
        if norm_layer is not None:
            self.norm = norm_layer(embed_dim)
        else:
            self.norm = None

    def forward(self, x):
        B, F, K, D = x.shape
        x = self.proj(x)  # B, F, K, ed
        if self.norm is not None:
            x = self.norm(x)
        return x

class TemporalMerging(nn.Module):
    def __init__(self, dim, embed_dim_inc_rate, temporal_patch_size, norm_layer=nn.LayerNorm, device = "cpu") -> None:
        super().__init__()
        self.dim = dim
        self.embed_dim_inc_rate = embed_dim_inc_rate
        self.temporal_patch_size = temporal_patch_size
        # self.reduction = nn.Linear(temporal_patch_size * dim, embed_dim_inc_rate * dim, bias=False)
        # self.norm = norm_layer(temporal_patch_size * dim)
        self.device = device
    
    def forward(self, x):
        B, F, K, ED = x.shape

        x_list = []
        for i in range(self.temporal_patch_size):
            x_list.append(torch.index_select(x, 1, torch.arange(i, F, self.temporal_patch_size, device = self.device)))

        x = torch.cat(x_list, -1)  # B F/temporal_patch_size ED
        # print('after concate', x.shape)

        # x = self.norm(x)
        # x = self.reduction(x)

        return x
    
class MSA(nn.Module):
    def __init__(self, dim, num_heads, edge_bias=None,
            attn_drop=0., proj_drop=0.) -> None:
        super().__init__()
        self.dim = dim
        self.num_heads = num_heads

        assert dim%num_heads == 0, 'dim and number of heads are incompatible'

        head_dim = dim // num_heads
        self.scale = head_dim ** -0.5
        self.edge_bias = edge_bias
        # print(edge_bias.shape)
        
        self.qkv = nn.Linear(dim, dim * 3)
        self.attn_drop = nn.Dropout(attn_drop)
        self.proj = nn.Linear(dim, dim)
        self.proj_drop = nn.Dropout(proj_drop)

        self.softmax = nn.Softmax(dim=-1)
        self.attn_act = nn.ReLU()
    
    def forward(self, x, B, f, w, mask=None):
        B_f_nW, W_TP, ED = x.shape
        qkv = self.qkv(x).reshape(B_f_nW, W_TP, 3, self.num_heads, ED // self.num_heads).permute(2, 0, 3, 1, 4)
        q, k, v = qkv[0], qkv[1], qkv[2]  # make torchscript happy (cannot use tensor as tuple)

        q = q * self.scale

        attn = (q @ k.transpose(-2, -1))
        # print('QKt shape', attn.shape)

        # making some attn 0 to prevent overfitting
        if self.training:
            attn_copy = attn.detach().clone()
            seed_ = torch.rand(1).item()
            attn_copy = self.softmax(attn_copy)
            index_array = attn_copy > seed_
            index_array = torch.where(index_array == True, 0, 1)
            attn = attn * index_array

        if mask is not None:
            # print('mask and attention shape ',mask.shape, attn.view(B, f*w, *attn.shape[1:]).shape)
            attn = attn.view(B, f*w, *attn.shape[1:]) * mask.unsqueeze(1).unsqueeze(0)
            attn = attn.view(B*f*w, *attn.shape[2:])
        
        if self.edge_bias is not None:
            attn = attn.view(B, f*w, *attn.shape[1:]) * self.edge_bias.unsqueeze(1)
            attn = attn.view(B*f*w, *attn.shape[2:])

        # if self.training:
        #     attn_copy = attn.detach().clone()
        #     attn_copy = self.softmax(attn_copy)
        #     sample_array = 1 - torch.bernoulli(attn_copy)
        #     attn = attn * sample_array
        
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
                edge_bias=None,
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
        self.attn = MSA(dim, num_heads=num_heads, edge_bias=edge_bias,
            attn_drop=attn_drop, proj_drop=drop)

        self.norm2 = norm_layer(dim)
        self.ff = FeedForward(in_features=dim, hidden_features=int(self.ff_dim), act_layer=act_layer, drop=drop)

        if self.shift_size > 0:
            # calculate attention mask for SW-MSA
            # print(temporal_dim)
            F, K = temporal_dim, num_kps
            frame_mask = torch.zeros((1, F, K, 1))  # 1 H W 1
            t_slices = (slice(0, -temporal_patch_size),
                        slice(-temporal_patch_size, -shift_size),
                        slice(-shift_size, None))

            cnt = 0
            for t in t_slices:
                frame_mask[:, t, :] = cnt
                cnt += 1

            # print('frame mask', frame_mask.shape)
            mask_windows = window_partition(frame_mask, window_size, temporal_patch_size).squeeze(2)
            # print('mask_windows', mask_windows.shape)
            attn_mask = mask_windows.unsqueeze(1) - mask_windows.unsqueeze(2)
            attn_mask = attn_mask.masked_fill(attn_mask != 0, float(0.0)).masked_fill(attn_mask == 0, float(1))
            # print('attn_mask', attn_mask.shape)
        else:
            attn_mask = None

        self.register_buffer("attn_mask", attn_mask)
    
    def forward(self, x):
        # print('in p_att_blk', x.shape)
        W, TP = self.window_size, self.temporal_patch_size
        B, F, K, ED = x.shape
        f, nW = F//TP, K//W

        shortcut = x

        # cyclic shift
        if self.shift_size > 0:
            shifted_x = torch.roll(x, shifts=-self.shift_size, dims=1)
            # partition windows
        else:
            shifted_x = x
            # partition windows
        x = window_partition(shifted_x, W, TP)

        x = self.norm1(x)
        # W-MSA/TS-MSA
        x = self.attn(x, B, f, nW, mask=self.attn_mask)  # B_f_nW, W_TP, ED

        # cyclic shift
        if self.shift_size > 0:
            shifted_x = torch.roll(x, shifts=self.shift_size, dims=1)
            # partition windows
        else:
            shifted_x = x
            # partition windows
        x = window_reverse(shifted_x, W, TP, F, K)

        x = shortcut + x
        # FFN
        x = x + self.ff(self.norm2(x))

        return x

class PartAttentionLayer(nn.Module):
    def __init__(self, dim, temporal_patch_size, temporal_dim, embed_dim_inc_rate, num_kps, depth, num_heads, window_size, edge_bias,
                 drop=0., attn_drop=0., ff_ratio=4., norm_layer=nn.LayerNorm, downsample=None, i_layer=0, device=None):
        super().__init__()
        self.dim = dim
        self.depth = depth
        self.num_heads = num_heads
        self.window_size = window_size
        self.edge_bias = edge_bias.to(device)
        self.drop_part = DropPart(0.2, num_kps=num_kps, device=device)
        self.i_layer = i_layer
        # self.register_buffer('edge_bias', self.edge_bias)

        # build blocks
        self.blocks = nn.ModuleList([
            PartAttentionBlock(dim=dim, num_kps=num_kps,
                                 num_heads=num_heads, window_size=window_size,
                                 temporal_patch_size=temporal_patch_size,
                                 temporal_dim = temporal_dim,
                                 shift_size=0 if (i % 2 == 0) else temporal_patch_size // 2,
                                 edge_bias=self.edge_bias,
                                 drop=drop, attn_drop=attn_drop,
                                 ff_ratio=ff_ratio,
                                 norm_layer=norm_layer)
            for i in range(depth)])

        # temporal merging layer
        if downsample is not None:
            self.downsample = downsample(dim, embed_dim_inc_rate, temporal_patch_size, norm_layer=norm_layer, device=device)
        else:
            self.downsample = None

    def forward(self, x):
        input = x
        for idx, blk in enumerate(self.blocks):
            x = blk(x) # + input
            # if self.i_layer == 0 and idx > len(self.blocks) - 4:
            #     x = self.drop_part(x)
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
                 embed_dim_inc_rate=1,
                 temporal_patch_size=4,
                 ape=False,
                 depths=[2, 2, 6, 2],
                 num_heads=[2, 4, 8, 16],
                 window_size=16,
                 edge_bias=None,
                 drop_rate=0., attn_drop_rate=0., ff_ratio=4.,
                 norm_layer=nn.LayerNorm,
                 kp_norm=True,
                 device=None,
                 ) -> None:
        super().__init__()
        self.kp_dim = kp_dim
        self.num_kps = num_kps
        self.temporal_dim = temporal_dim
        self.window_size = window_size
        self.num_classes = num_classes
        self.num_layers = len(depths)
        self.ape = ape
        self.edge_bias = edge_bias
        self.embed_dim = embed_dim
        self.embed_dim_inc_rate = embed_dim_inc_rate
        self.num_features = int(embed_dim * embed_dim_inc_rate ** (self.num_layers - 1))
        self.temporal_out_dim = temporal_dim // temporal_patch_size ** (self.num_layers - 1)

        assert self.num_kps%window_size == 0, "window size and number of kps are incompatible"
        assert self.temporal_dim%temporal_patch_size == 0, "temporal dimension and temporal patch size are incompatible"

        mapping_size = embed_dim//2
        scale = 10
        
        B_gauss = torch.normal(0.0, 1.0, (mapping_size, self.kp_dim))
        B = B_gauss * scale
        self.B = nn.Parameter(B, requires_grad=False)

        # self.patch_embed = KpEmbed(kp_dim=self.kp_dim, embed_dim=self.embed_dim,
        #     norm_layer=norm_layer if kp_norm else None)

        # absolute position embedding
        if self.ape:
            self.pos_encoder = PositionalEncoding(embed_dim, drop_rate, temporal_dim)

        # build layers
        self.layers = nn.ModuleList()
        for i_layer in range(self.num_layers):
            if edge_bias is not None:
                edge_bias_t = torch.concatenate([edge_bias for _ in range(temporal_dim // temporal_patch_size ** (i_layer+1))])
            else:
                edge_bias_t = None
            layer = PartAttentionLayer(dim=int(embed_dim * embed_dim_inc_rate ** i_layer),
                               temporal_patch_size=temporal_patch_size,
                               temporal_dim=temporal_dim//(temporal_patch_size ** i_layer),
                               embed_dim_inc_rate=embed_dim_inc_rate,
                               num_kps=num_kps,
                               depth=depths[i_layer],
                               num_heads=num_heads[i_layer],
                               window_size=window_size,
                               edge_bias=edge_bias_t,
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
        # print('input shape', x.shape)
        x_proj = (2.*torch.pi*x) @ self.B.transpose(1,0)
        x_ = torch.cat([torch.sin(x_proj), torch.cos(x_proj)], axis=-1)
        x = x_
        # x = self.patch_embed(x) + x_
        # x = self.patch_embed(x)
        # print('embed input shape', x.shape)
        if self.ape:
            x = self.pos_encoder(x)
        # print('after ape shape', x.shape)

        for layer in self.layers:
            x = layer(x)

        B, f, K, d = x.shape
        x = self.norm(x)
        x = self.avgpool(x.transpose(1, 3).reshape(B, d, -1))
        x = x.reshape(B, d)
        return x

    def forward(self, x):
        x = self.forward_features(x)
        # print('feature shape', x.shape)
        x = self.head(x)
        return x
