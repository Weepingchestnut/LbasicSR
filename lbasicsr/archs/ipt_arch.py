import copy
import math
import numpy as np
import torch
from torch import nn as nn
from torch.nn import functional as F

from lbasicsr.utils.registry import ARCH_REGISTRY


def default_conv(in_channels, out_channels, kernel_size, bias=True):
    return nn.Conv2d(in_channels, out_channels, kernel_size, padding=(kernel_size//2), bias=bias)


class ResBlock(nn.Module):
    def __init__(self, conv, n_feats, kernel_size, bias=True, bn=False, act=nn.ReLU(True), res_scale=1):
        super(ResBlock, self).__init__()
        m = []
        for i in range(2):
            m.append(conv(n_feats, n_feats, kernel_size, bias=bias))
            if bn:
                m.append(nn.BatchNorm2d(n_feats))
            if i == 0:
                m.append(act)

        self.body = nn.Sequential(*m)
        self.res_scale = res_scale

    def forward(self, x):
        res = self.body(x).mul(self.res_scale)
        res += x

        return res


class Upsampler(nn.Sequential):
    def __init__(self, conv, scale, n_feats, bn=False, act=False, bias=True):

        m = []
        if (scale & (scale - 1)) == 0:    # Is scale = 2^n?
            for _ in range(int(math.log(scale, 2))):
                m.append(conv(n_feats, 4 * n_feats, 3, bias))
                m.append(nn.PixelShuffle(2))
                if bn:
                    m.append(nn.BatchNorm2d(n_feats))
                if act == 'relu':
                    m.append(nn.ReLU(True))
                elif act == 'prelu':
                    m.append(nn.PReLU(n_feats))

        elif scale == 3:
            m.append(conv(n_feats, 9 * n_feats, 3, bias))
            m.append(nn.PixelShuffle(3))
            if bn:
                m.append(nn.BatchNorm2d(n_feats))
            if act == 'relu':
                m.append(nn.ReLU(True))
            elif act == 'prelu':
                m.append(nn.PReLU(n_feats))
        else:
            raise NotImplementedError

        super(Upsampler, self).__init__(*m)


def _get_clones(module, N):
    return nn.ModuleList([copy.deepcopy(module) for i in range(N)])


def _get_activation_fn(activation):
    """Return an activation function given a string"""
    if activation == "relu":
        return F.relu
    if activation == "gelu":
        return F.gelu
    if activation == "glu":
        return F.glu
    raise RuntimeError(F"activation should be relu/gelu, not {activation}.")


class TransformerEncoder(nn.Module):

    def __init__(self, encoder_layer, num_layers):
        super().__init__()
        self.layers = _get_clones(encoder_layer, num_layers)
        self.num_layers = num_layers

    def forward(self, src, pos = None):
        output = src

        for layer in self.layers:
            output = layer(output, pos=pos)

        return output


class TransformerEncoderLayer(nn.Module):

    def __init__(self, d_model, nhead, dim_feedforward=2048, dropout=0.1, no_norm = False,
                 activation="relu"):
        super().__init__()
        self.self_attn = nn.MultiheadAttention(d_model, nhead, dropout=dropout, bias=False)
        # Implementation of Feedforward model
        self.linear1 = nn.Linear(d_model, dim_feedforward)
        self.dropout = nn.Dropout(dropout)
        self.linear2 = nn.Linear(dim_feedforward, d_model)
        
        self.norm1 = nn.LayerNorm(d_model) if not no_norm else nn.Identity()
        self.norm2 = nn.LayerNorm(d_model) if not no_norm else nn.Identity()
        self.dropout1 = nn.Dropout(dropout)
        self.dropout2 = nn.Dropout(dropout)

        self.activation = _get_activation_fn(activation)
        
        nn.init.kaiming_uniform_(self.self_attn.in_proj_weight, a=math.sqrt(5))

    def with_pos_embed(self, tensor, pos):
        return tensor if pos is None else tensor + pos
    
    def forward(self, src, pos = None):
        src2 = self.norm1(src)
        q = k = self.with_pos_embed(src2, pos)
        src2 = self.self_attn(q, k, src2)
        src = src + self.dropout1(src2[0])
        src2 = self.norm2(src)
        src2 = self.linear2(self.dropout(self.activation(self.linear1(src2))))
        src = src + self.dropout2(src2)
        return src


class TransformerDecoder(nn.Module):

    def __init__(self, decoder_layer, num_layers):
        super().__init__()
        self.layers = _get_clones(decoder_layer, num_layers)
        self.num_layers = num_layers

    def forward(self, tgt, memory, pos = None, query_pos = None):
        output = tgt
        
        for layer in self.layers:
            output = layer(output, memory, pos=pos, query_pos=query_pos)

        return output


class TransformerDecoderLayer(nn.Module):

    def __init__(self, d_model, nhead, dim_feedforward=2048, dropout=0.1, no_norm = False,
                 activation="relu"):
        super().__init__()
        self.self_attn = nn.MultiheadAttention(d_model, nhead, dropout=dropout, bias=False)
        self.multihead_attn = nn.MultiheadAttention(d_model, nhead, dropout=dropout, bias=False)
        # Implementation of Feedforward model
        self.linear1 = nn.Linear(d_model, dim_feedforward)
        self.dropout = nn.Dropout(dropout)
        self.linear2 = nn.Linear(dim_feedforward, d_model)

        self.norm1 = nn.LayerNorm(d_model) if not no_norm else nn.Identity()
        self.norm2 = nn.LayerNorm(d_model) if not no_norm else nn.Identity()
        self.norm3 = nn.LayerNorm(d_model) if not no_norm else nn.Identity()
        self.dropout1 = nn.Dropout(dropout)
        self.dropout2 = nn.Dropout(dropout)
        self.dropout3 = nn.Dropout(dropout)

        self.activation = _get_activation_fn(activation)

    def with_pos_embed(self, tensor, pos):
        return tensor if pos is None else tensor + pos

    def forward(self, tgt, memory, pos = None, query_pos = None):
        tgt2 = self.norm1(tgt)
        q = k = self.with_pos_embed(tgt2, query_pos)
        tgt2 = self.self_attn(q, k, value=tgt2)[0]
        tgt = tgt + self.dropout1(tgt2)
        tgt2 = self.norm2(tgt)
        tgt2 = self.multihead_attn(query=self.with_pos_embed(tgt2, query_pos),
                                   key=self.with_pos_embed(memory, pos),
                                   value=memory)[0]
        tgt = tgt + self.dropout2(tgt2)
        tgt2 = self.norm3(tgt)
        tgt2 = self.linear2(self.dropout(self.activation(self.linear1(tgt2))))
        tgt = tgt + self.dropout3(tgt2)
        return tgt


class LearnedPositionalEncoding(nn.Module):
    def __init__(self, max_position_embeddings, embedding_dim, seq_length):
        super(LearnedPositionalEncoding, self).__init__()
        self.pe = nn.Embedding(max_position_embeddings, embedding_dim)
        self.seq_length = seq_length

        self.register_buffer(
            "position_ids", torch.arange(self.seq_length).expand((1, -1))
        )

    def forward(self, x, position_ids=None):
        if position_ids is None:
            position_ids = self.position_ids[:, : self.seq_length]

        position_embeddings = self.pe(position_ids)
        return position_embeddings


class VisionTransformer(nn.Module):
    def __init__(self,
                 img_dim: int = 48,
                 patch_dim: int = 3,
                 num_channels: int = 64,
                 embedding_dim: int = 576,      # num_feat * patch_dim^2
                 num_heads: int = 12,
                 num_layers: int = 12,
                 hidden_dim: int = 2304,        # num_feat * patch_dim^2 * 4
                 num_queries: int = 1,
                 positional_encoding_type="learned",
                 dropout_rate=0,
                 no_norm=False,
                 mlp=False,
                 pos_every=False,
                 no_pos=False) -> None:
        super(VisionTransformer, self).__init__()
        
        assert embedding_dim % num_heads == 0
        assert img_dim % patch_dim == 0
        self.no_norm = no_norm
        self.mlp = mlp
        self.embedding_dim = embedding_dim
        self.num_heads = num_heads
        self.patch_dim = patch_dim
        self.num_channels = num_channels
        
        self.img_dim = img_dim
        self.pos_every = pos_every
        self.num_patches = int((img_dim // patch_dim) ** 2)
        self.seq_length = self.num_patches
        self.flatten_dim = patch_dim * patch_dim * num_channels
        
        self.out_dim = patch_dim * patch_dim * num_channels
        
        self.no_pos = no_pos
        
        if self.mlp == False:
            self.linear_encoding = nn.Linear(self.flatten_dim, embedding_dim)
            self.mlp_head = nn.Sequential(
                nn.Linear(embedding_dim, hidden_dim),
                nn.Dropout(dropout_rate),
                nn.ReLU(),
                nn.Linear(hidden_dim, self.out_dim),
                nn.Dropout(dropout_rate)
            )
            
            self.query_embed = nn.Embedding(num_queries, embedding_dim * self.seq_length)
        
        encoder_layer = TransformerEncoderLayer(embedding_dim, num_heads, hidden_dim, dropout_rate, self.no_norm)
        self.encoder = TransformerEncoder(encoder_layer, num_layers)
        
        decoder_layer = TransformerDecoderLayer(embedding_dim, num_heads, hidden_dim, dropout_rate, self.no_norm)
        self.decoder = TransformerDecoder(decoder_layer, num_layers)
        
        if not self.no_pos:
            self.position_encoding = LearnedPositionalEncoding(
                    self.seq_length, self.embedding_dim, self.seq_length
                )
            
        self.dropout_layer1 = nn.Dropout(dropout_rate)
        
        if no_norm:
            for m in self.modules():
                if isinstance(m, nn.Linear):
                    nn.init.normal_(m.weight, std = 1/m.weight.size(1))
    
    def forward(self, x, query_idx, con=False):     # x: torch.Size([B, C(64), 48, 48])

        x = F.unfold(x, self.patch_dim, stride=self.patch_dim).transpose(1,2).transpose(0,1).contiguous()   # torch.Size([B, C*patch_dim^2(576), 256]) --> torch.Size([256, 1, 576])
               
        if self.mlp == False:
            x = self.dropout_layer1(self.linear_encoding(x)) + x    # torch.Size([256, 1, 576])

            query_embed = self.query_embed.weight[query_idx].view(-1, 1, self.embedding_dim).repeat(1, x.size(1), 1)   # torch.Size([256, 1, 576])
        else:
            query_embed = None

        
        if not self.no_pos:
            pos = self.position_encoding(x).transpose(0, 1)      # torch.Size([1, 256, 576]) --> torch.Size([256, 1, 576])

        if self.pos_every:
            x = self.encoder(x, pos=pos)
            x = self.decoder(x, x, pos=pos, query_pos=query_embed)
        elif self.no_pos:
            x = self.encoder(x)
            x = self.decoder(x, x, query_pos=query_embed)
        else:
            x = self.encoder(x + pos)
            x = self.decoder(x, x, query_pos=query_embed)
        
        
        if self.mlp == False:
            x = self.mlp_head(x) + x
        
        x = x.transpose(0,1).contiguous().view(x.size(1), -1, self.flatten_dim)
        
        if con:
            con_x = x
            x = torch.nn.functional.fold(x.transpose(1,2).contiguous(),int(self.img_dim),self.patch_dim,stride=self.patch_dim)
            return x, con_x
        
        x = torch.nn.functional.fold(x.transpose(1,2).contiguous(),int(self.img_dim),self.patch_dim,stride=self.patch_dim)
        
        return x
        

# @ARCH_REGISTRY.register()
class IPT(nn.Module):
    def __init__(self,
                 num_in_ch: int = 3,
                 num_feat: int = 64,
                 conv = default_conv,
                 scale_list: list = [4],
                 patch_size: int = 48,
                 patch_dim: int = 3,
                 num_heads: int = 12,
                 num_layers: int = 12,
                 num_queries: int = 1,
                 dropout_rate: float = 0.,
                 no_mlp: bool = False,
                 pos_every: bool = False,
                 no_pos: bool = False,
                 no_norm: bool = False) -> None:
        super(IPT, self).__init__()
        
        self.scale_idx = 0
        self.num_feat = num_feat
        
        kernel_size = 3
        
        self.act = nn.ReLU(True)
        
        self.head = nn.ModuleList([
            nn.Sequential(
                conv(num_in_ch, num_feat, kernel_size),
                ResBlock(conv, num_feat, kernel_size=5, act=self.act),
                ResBlock(conv, num_feat, kernel_size=5, act=self.act),
            ) for _ in scale_list
        ])
        
        self.body = VisionTransformer(
            img_dim=patch_size, 
            patch_dim=patch_dim, 
            num_channels=num_feat, 
            embedding_dim=num_feat*patch_dim*patch_dim, 
            num_heads=num_heads, 
            num_layers=num_layers, 
            hidden_dim=num_feat*patch_dim*patch_dim*4, 
            num_queries = num_queries, 
            dropout_rate=dropout_rate, 
            mlp=no_mlp, 
            pos_every=pos_every, 
            no_pos=no_pos, 
            no_norm=no_norm)
        
        self.tail = nn.ModuleList([
            nn.Sequential(
                Upsampler(conv, s, num_feat, act=False),
                conv(num_feat, num_in_ch, kernel_size)
            ) for s in scale_list
        ])
    
    def forward(self, x):
        # x = self.sub_mean(x)
        x = self.head[self.scale_idx](x)

        res = self.body(x, self.scale_idx)
        res += x

        x = self.tail[self.scale_idx](res)
        # x = self.add_mean(x)

        return x 

    def set_scale(self, scale_idx):
        self.scale_idx = scale_idx
        
        
if __name__ == '__main__':
    from fvcore.nn import flop_count_table, FlopCountAnalysis, ActivationCountAnalysis

    # device = 'cuda' if torch.cuda.is_available() else 'cpu'
    device = 'cpu'
    
    model = IPT(
        num_in_ch=3,
        num_feat=64,
    ).to(device)
    model.eval()

    print(
        "Model have {:.3f}M parameters in total".format(sum(x.numel() for x in model.parameters()) / 1000000.0))

    input = torch.rand(1, 3, 48, 48).to(device)

    with torch.no_grad():
        print(flop_count_table(FlopCountAnalysis(model, input), activations=ActivationCountAnalysis(model, input)))
        out = model(input)

    print(out.shape)


"""
Model have 115.608M parameters in total
| module                              | #parameters or shape   | #flops    | #activations   |
|:------------------------------------|:-----------------------|:----------|:---------------|
| model                               | 0.116G                 | 34.775G   | 77.23M         |
|  head.0                             |  0.412M                |  0.948G   |  0.737M        |
|   head.0.0                          |   1.792K               |   3.981M  |   0.147M       |
|    head.0.0.weight                  |    (64, 3, 3, 3)       |           |                |
|    head.0.0.bias                    |    (64,)               |           |                |
|   head.0.1.body                     |   0.205M               |   0.472G  |   0.295M       |
|    head.0.1.body.0                  |    0.102M              |    0.236G |    0.147M      |
|    head.0.1.body.2                  |    0.102M              |    0.236G |    0.147M      |
|   head.0.2.body                     |   0.205M               |   0.472G  |   0.295M       |
|    head.0.2.body.0                  |    0.102M              |    0.236G |    0.147M      |
|    head.0.2.body.2                  |    0.102M              |    0.236G |    0.147M      |
|  body                               |  0.115G                |  32.065G  |  73.433M       |
|   body.linear_encoding              |   0.332M               |   84.935M |   0.147M       |
|    body.linear_encoding.weight      |    (576, 576)          |           |                |
|    body.linear_encoding.bias        |    (576,)              |           |                |
|   body.mlp_head                     |   2.657M               |   0.679G  |   0.737M       |
|    body.mlp_head.0                  |    1.329M              |    0.34G  |    0.59M       |
|    body.mlp_head.3                  |    1.328M              |    0.34G  |    0.147M      |
|   body.query_embed                  |   0.147M               |           |                |
|    body.query_embed.weight          |    (1, 147456)         |           |                |
|   body.encoder.layers               |   47.838M              |   13.154G |   27.132M      |
|    body.encoder.layers.0            |    3.986M              |    1.096G |    2.261M      |
|    body.encoder.layers.1            |    3.986M              |    1.096G |    2.261M      |
|    body.encoder.layers.2            |    3.986M              |    1.096G |    2.261M      |
|    body.encoder.layers.3            |    3.986M              |    1.096G |    2.261M      |
|    body.encoder.layers.4            |    3.986M              |    1.096G |    2.261M      |
|    body.encoder.layers.5            |    3.986M              |    1.096G |    2.261M      |
|    body.encoder.layers.6            |    3.986M              |    1.096G |    2.261M      |
|    body.encoder.layers.7            |    3.986M              |    1.096G |    2.261M      |
|    body.encoder.layers.8            |    3.986M              |    1.096G |    2.261M      |
|    body.encoder.layers.9            |    3.986M              |    1.096G |    2.261M      |
|    body.encoder.layers.10           |    3.986M              |    1.096G |    2.261M      |
|    body.encoder.layers.11           |    3.986M              |    1.096G |    2.261M      |
|   body.decoder.layers               |   63.777M              |   18.146G |   45.416M      |
|    body.decoder.layers.0            |    5.315M              |    1.512G |    3.785M      |
|    body.decoder.layers.1            |    5.315M              |    1.512G |    3.785M      |
|    body.decoder.layers.2            |    5.315M              |    1.512G |    3.785M      |
|    body.decoder.layers.3            |    5.315M              |    1.512G |    3.785M      |
|    body.decoder.layers.4            |    5.315M              |    1.512G |    3.785M      |
|    body.decoder.layers.5            |    5.315M              |    1.512G |    3.785M      |
|    body.decoder.layers.6            |    5.315M              |    1.512G |    3.785M      |
|    body.decoder.layers.7            |    5.315M              |    1.512G |    3.785M      |
|    body.decoder.layers.8            |    5.315M              |    1.512G |    3.785M      |
|    body.decoder.layers.9            |    5.315M              |    1.512G |    3.785M      |
|    body.decoder.layers.10           |    5.315M              |    1.512G |    3.785M      |
|    body.decoder.layers.11           |    5.315M              |    1.512G |    3.785M      |
|   body.position_encoding.pe         |   0.147M               |   0       |   0            |
|    body.position_encoding.pe.weight |    (256, 576)          |           |                |
|  tail.0                             |  0.297M                |  1.762G   |  3.06M         |
|   tail.0.0                          |   0.295M               |   1.699G  |   2.949M       |
|    tail.0.0.0                       |    0.148M              |    0.34G  |    0.59M       |
|    tail.0.0.2                       |    0.148M              |    1.359G |    2.359M      |
|   tail.0.1                          |   1.731K               |   63.701M |   0.111M       |
|    tail.0.1.weight                  |    (3, 64, 3, 3)       |           |                |
|    tail.0.1.bias                    |    (3,)                |           |                |
torch.Size([1, 3, 192, 192])
"""
        
