#  -*- coding: utf-8 -*-
'''
author: xuechao.wang@ugent.be
'''
import torch
from torch import nn, einsum
import numpy as np
from torchsummary import summary
from einops import rearrange, repeat


def pair(t):
    return t if isinstance(t, tuple) else (t, t)


class PreNorm(nn.Module):
    # layernorm层
    def __init__(self, dim, fn):
        super().__init__()
        self.norm = nn.LayerNorm(dim)
        self.fn = fn
    def forward(self, x, **kwargs):
        return self.fn(self.norm(x), **kwargs)


def get_positional_embeddings(sequence_length, d):
    # sequence_length: 序列长度（注意，这里第一个位置是cls，从第二个位置开始才是真正的序列）
    # d: 代表序列中每一时刻的特征维度（也就是词嵌入长度）
    result = torch.ones(sequence_length, d)
    for i in range(sequence_length):
        for j in range(d):
            result[i][j] = np.sin(i / (10000 ** (j / d))) if j % 2 == 0 else np.cos(i / (10000 ** ((j - 1) / d)))
    return result


class FeedForward(nn.Module):
    #MLP层
    def __init__(self, dim, hidden_dim, dropout=0.):
        super().__init__()
        self.net = nn.Sequential(
            nn.Linear(dim, hidden_dim),
            nn.GELU(),
            nn.Dropout(dropout),
            nn.Linear(hidden_dim, dim),
            nn.Dropout(dropout)
        )
    def forward(self, x):
        return self.net(x)


class Attention(nn.Module):
    def __init__(self, dim, heads=8, dim_head=64, dropout=0.):
        super().__init__()

        assert dim % heads == 0, f"Can't divide dimension {dim} into {heads} heads"

        inner_dim = dim_head * heads
        project_out = not (heads == 1 and dim_head == dim) # 如果不是多头 或者 输入和输出维度相等 ，则进行空操作

        self.heads = heads
        self.scale = dim_head ** -0.5 # 缩放因子

        self.attend = nn.Softmax(dim=-1)
        self.to_qkv = nn.Linear(dim, inner_dim * 3, bias=False)

        self.to_out = nn.Sequential(
            nn.Linear(inner_dim, dim),
            nn.Dropout(dropout),
        ) if project_out else nn.Identity()

    def forward(self, x):
        b, n, _, h = *x.shape, self.heads

        qkv = self.to_qkv(x).chunk(3, dim=-1)  # (b, n(65), dim*3) ---> 3 * (b, n, dim)
        q, k, v = map(lambda t: rearrange(t, 'b n (h d) -> b h n d', h=h), qkv)  # q, k, v   (b, h, n, dim_head(64))

        dots = einsum('b h i d, b h j d -> b h i j', q, k) * self.scale
        attn = self.attend(dots)
        out = einsum('b h i j, b h j d -> b h i d', attn, v)

        out = rearrange(out, 'b h n d -> b n (h d)')
        return self.to_out(out)


class Transformer(nn.Module):
    def __init__(self, dim, depth, heads, dim_head, mlp_dim, dropout=0.):
        super().__init__()
        self.layers = nn.ModuleList([])
        for _ in range(depth):
            self.layers.append(nn.ModuleList([
                PreNorm(dim, Attention(dim, heads=heads, dim_head=dim_head, dropout=dropout)),
                PreNorm(dim, FeedForward(dim, mlp_dim, dropout=dropout))
            ]))

    def forward(self, x):
        for attn, ff in self.layers:
            x = attn(x) + x
            x = ff(x) + x
        return x


class SiT(nn.Module):
    def __init__(self, *,
                 seq_size,
                 dim,
                 num_classes,
                 depth,
                 heads,
                 mlp_dim,
                 pool='cls',
                 dim_head=64,
                 dropout=0.,
                 emb_dropout=0.
                 ):
        super().__init__()

        assert pool in {'cls', 'mean'}

        self.pos_embedding = nn.Parameter(torch.randn(1, seq_size+1, dim)) # 这个位置信息是通过学习得到的

        self.cls_token = nn.Parameter(torch.randn(1, 1, dim))	# nn.Parameter()定义可学习参数
        self.dropout = nn.Dropout(emb_dropout)

        self.transformer = Transformer(dim, depth, heads, dim_head, mlp_dim, dropout)

        self.pool = pool
        self.to_latent = nn.Identity()

        self.mlp_head = nn.Sequential(
            nn.LayerNorm(dim),
            nn.Linear(dim, 128),
            nn.GELU(),
            nn.Linear(128, 32),
            nn.GELU(),
            nn.Linear(32, num_classes)
        )

    def forward(self, x):
        x = torch.squeeze(x)
        b, n, m = x.shape           # b表示batch size, n表示映射完的序列长度, m表示序列每时刻对应的维度数

        cls_tokens = repeat(self.cls_token, '() n d -> b n d', b=b)  # self.cls_token: (1, 1, dim) -> cls_tokens: (batchSize, 1, dim)
        x = torch.cat((cls_tokens, x), dim=1)   # 将cls_token拼接到patch token中去 -> (b, n+1, dim)

        # x += self.pos_embedding[:, :(n+1)]   # 加位置嵌入,采用可学习的形式（直接加） -> (b, n+1, dim)

        x = self.dropout(x)

        x = self.transformer(x)                                                 # (b, 65, dim)

        x = x.mean(dim=1) if self.pool == 'mean' else x[:, 0]                   # (b, dim)

        x = self.to_latent(x)                                                   # Identity (b, dim)
        # print(x.shape)

        return self.mlp_head(x)                                                 #  (b, num_classes)


if __name__=='__main__':

    model_sit = SiT(
        seq_size=512,
        dim=5,  # 序列中每一时刻dim维度（类似词嵌入的维度）
        num_classes=2,
        depth=4, # encoder层数
        heads=5, # encoder中多头注意力机制中头的个数
        mlp_dim=512, # encoder的后半部分MLP中 将dim维度数映射到mpl_dim维度数
        dropout=0., # encoder的后半部分MLP中 对input的随机省略概率
        emb_dropout=0. # 在送入transformer之前对input的随机省略概率
    )

    seq = torch.randn(32, 512, 5)

    preds = model_sit(seq)

    print(preds.shape)  # (16, 1000)

    summary(SiT(
        seq_size=512,
        dim=6,  # 将patch映射成dim维度序列（类似词嵌入的维度）
        num_classes=2,
        depth=4, # encoder层数
        heads=3, # encoder中多头注意力机制中头的个数
        mlp_dim=512, # encoder的后半部分MLP中 将dim维度数映射到mpl_dim维度数
        dropout=0., # encoder的后半部分MLP中 对input的随机省略概率
        emb_dropout=0. # 在送入transformer之前对input的随机省略概率
    ), (512, 6), device='cpu')


