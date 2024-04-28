import torch
import torch.nn as nn
import math


class MultiheadAttention(nn.Module):
    def __init__(self, dim, n_heads, dropout=0.1):
        super().__init__()

        self.n_heads = n_heads
        self.dim = dim
        self.dropout = nn.Dropout(p=dropout)

        assert self.dim % self.n_heads == 0

        self.q_lin = nn.Linear(in_features=self.dim, out_features=self.dim)
        self.k_lin = nn.Linear(in_features=self.dim, out_features=self.dim)
        self.v_lin = nn.Linear(in_features=self.dim, out_features=self.dim)
        self.out_lin = nn.Linear(in_features=self.dim, out_features=self.dim)

    def forward(self, query: torch.Tensor, key: torch.Tensor, value: torch.Tensor, key_padding_mask: torch.Tensor = None, attn_mask: torch.Tensor = None):
        bs, q_length, dim = query.size()
        k_length = key.size(1)
        dim_per_head = self.dim // self.n_heads
        mask_reshp = (bs, 1, 1, k_length)

        def shape(x):
            """ separate heads """
            return x.view(bs, -1, self.n_heads, dim_per_head).transpose(1, 2)

        def unshape(x):
            """ group heads """
            return (x.transpose(1, 2).contiguous().view(bs, -1, self.n_heads * dim_per_head))

        q = shape(self.q_lin(query))  # (bs, n_heads, q_length, dim_per_head)
        k = shape(self.k_lin(key))    # (bs, n_heads, k_length, dim_per_head)
        v = shape(self.v_lin(value))  # (bs, n_heads, k_length, dim_per_head)

        q = q / math.sqrt(dim_per_head)              # (bs, n_heads, q_length, dim_per_head)
        scores = torch.matmul(q, k.transpose(2, 3))  # (bs, n_heads, q_length, k_length)
        if key_padding_mask is not None:
            key_padding_mask = ((~key_padding_mask).view(mask_reshp).expand_as(scores))  # (bs, n_heads, q_length, k_length)
            scores.masked_fill_(key_padding_mask, -float("inf"))                          # (bs, n_heads, q_length, k_length)
        
        weights = nn.Softmax(dim=-1)(scores)  # (bs, n_heads, q_length, k_length)
        weights = self.dropout(weights)       # (bs, n_heads, q_length, k_length)

        if attn_mask is not None:
            attn_mask = attn_mask.unsqueeze(1).repeat(1, self.n_heads, 1).unsqueeze(-1)
            weights = weights * attn_mask.float()

        context = torch.matmul(weights, v)  # (bs, n_heads, q_length, dim_per_head)
        context = unshape(context)          # (bs, q_length, dim)
        context = self.out_lin(context)     # (bs, q_length, dim)

        return context
    
    
class Uni_CAST(nn.Module):
    def __init__(self, updim: int = 1024, downdim: int = 256, n_heads: int = 8, dropout: float = 0.1):
        super().__init__()
        
        self.frame_projection = nn.Sequential(
            nn.Linear(updim, downdim),
            nn.LayerNorm(downdim),
            nn.ReLU(),
            nn.Linear(downdim, downdim),
        )
        
        self.frame_cast_pc = MultiheadAttention(dim=downdim, n_heads=n_heads, dropout=dropout)  # point cloud: query || frame: key, value
        
    def forward(self, pc_token: torch.Tensor, frame_token: torch.Tensor, frame_token_mask: torch.Tensor):
        frame_token = self.frame_projection(frame_token)  # [bs, frame_len, d_frame] -> [bs, frame_len, d_pc]
        pc_token = self.frame_cast_pc(query=pc_token, key=frame_token, value=frame_token, key_padding_mask=frame_token_mask)
        
        return pc_token


class Adapter(nn.Module):
    def __init__(self, dim: int = 256, mlp_ratio: float = 0.25, act_layer = nn.GELU, skip_connect = True):
        super().__init__()
        self.skip_connect = skip_connect
        down_dim = int(dim * mlp_ratio)
        self.act = act_layer()
        self.D_fc1 = nn.Linear(dim, down_dim)
        self.D_fc2 = nn.Linear(down_dim, dim)
        
    def forward(self, x):
        # x: [bs, len, d]
        xs = self.D_fc1(x)
        xs = self.act(xs)
        xs = self.D_fc2(xs)
        if self.skip_connect:
            x = x + xs
        else:
            x = xs
        return x
    

if __name__ == "__main__":
    query = torch.rand(2, 3, 4)
    query_mask = torch.Tensor([
        [1, 1, 1],
        [1, 1, 0]
    ]).bool()
    
    key = torch.rand(2, 4, 4)
    key_mask = torch.Tensor([
        [1, 1, 1, 1],
        [1, 1, 0, 0]
    ]).bool()
    value = key
    
    attn_layer = MultiheadAttention(dim=4, n_heads=1)
    output = attn_layer(query=query, key=key, value=value, key_padding_mask=key_mask, attn_mask=query_mask)