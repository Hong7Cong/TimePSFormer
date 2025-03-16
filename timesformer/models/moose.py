import sys
# sys.path.append('/data2/hongn/RAFT')

import argparse
import os
import cv2
import glob
import numpy as np
import torch
from PIL import Image
import torch.nn as nn
from raft.raft import RAFT
from raft.utils import flow_viz
from raft.utils.utils import InputPadder
from PIL import Image
from einops import rearrange
from .build import MODEL_REGISTRY
import matplotlib.pyplot as plt
from matplotlib.patches import Polygon
import torch
import torch.nn as nn
import torchvision
from torchvision import transforms as pth_transforms
import numpy as np
from PIL import Image
# sys.path.append('/data2/hongn/dino')
# import utils
import vision_transformer as vits
import math
import torch.nn.functional as F

def cross_attention(Q, K, V, mask=None):
    # Compute the dot products between Q and K, then scale
    d_k = Q.size(-1)
    scores = torch.matmul(Q, K.transpose(-2, -1)) / torch.sqrt(torch.tensor(d_k, dtype=torch.float32))
    
    # Apply mask if provided
    if mask is not None:
        scores = scores.masked_fill(mask == 0, float('-inf'))
    
    # Softmax to normalize scores and get attention weights
    attention_weights = F.softmax(scores, dim=-1)
    
    # Weighted sum of values
    output = torch.matmul(attention_weights, V)
    return output, attention_weights

class CustomAttention(nn.Module):
    def __init__(self, embed_size, num_heads=1, attention_type="cross"):
        super(CustomAttention, self).__init__()
        self.embed_size = embed_size
        self.num_heads = num_heads
        self.attention_type = attention_type

        # Linear layers for Q, K, V
        self.query = nn.Linear(embed_size, embed_size)
        self.key = nn.Linear(embed_size, embed_size)
        self.value = nn.Linear(embed_size, embed_size)
        
        # Final linear layer after concatenating heads
        self.fc_out = nn.Linear(embed_size, embed_size)

    def forward(self, target, source=None, mask=None):
        if self.attention_type == "self":
            Q = self.query(target)
            K = self.key(target)
            V = self.value(target)
        elif self.attention_type == "cross":
            assert source is not None, "Source input required for cross-attention"
            Q = self.query(target)
            K = self.key(source)
            V = self.value(source)

        # Perform attention calculation (self or cross)
        out, _ = cross_attention(Q, K, V, mask)
        return self.fc_out(out)

class CustomAttentionWithResidual(nn.Module):
    def __init__(self, embed_size, num_heads=1, attention_type="self"):
        super(CustomAttentionWithResidual, self).__init__()
        self.attention = CustomAttention(embed_size, num_heads, attention_type)
        self.norm = nn.LayerNorm(embed_size)
        self.dropout = nn.Dropout(0.1)

    def forward(self, target, source=None, mask=None):
        attention_out = self.attention(target, source, mask)
        # Add residual connection and layer normalization
        out = self.norm(target + self.dropout(attention_out))
        return out

# class PatchEmbed(nn.Module):
#     """ Image to Patch Embedding
#     """
#     def __init__(self, img_size=224, patch_size=16, in_chans=3, embed_dim=768):
#         super().__init__()
#         num_patches = (img_size // patch_size) * (img_size // patch_size)
#         self.img_size = img_size
#         self.patch_size = patch_size
#         self.num_patches = num_patches

#         self.proj = nn.Conv2d(in_chans, embed_dim, kernel_size=patch_size, stride=patch_size)

#     def forward(self, x):
#         B, C, H, W = x.shape
#         x = self.proj(x).flatten(2).transpose(1, 2)
#         return x

def load_image(imfile):
    img = np.array(Image.open(imfile)).astype(np.uint8)
    img = torch.from_numpy(img).permute(2, 0, 1).float()
    return img[None]


def load_image_from_uint8array(img):
    img = torch.from_numpy(img).permute(0, 3, 1, 2).float()
    return img[None]

def denomalizing_img(tensor_img, mean = [0.45, 0.45, 0.45], std = [0.225, 0.225, 0.225]): # tensor_img (c t w h)
    device = tensor_img.device
    if(len(tensor_img.shape) == 4):
        ret = []
        for i in range(tensor_img.shape[0]):
            unnormalize_img = tensor_img[i].permute(1, 2, 0) * torch.tensor(std).to(device) + torch.tensor(mean).to(device)
            unnormalize_img = unnormalize_img * 225
            # unnormalize_img = unnormalize_img.astype(np.uint8)
            ret.append(unnormalize_img.permute(2, 0, 1).float())
        ret = torch.stack(ret)

    else:
        assert False, "len(tensor_img.shape) must equal 4 [b, t, w, h]"
    return ret

# class MOOSE_Encoder(nn.Module):
#     """ Vision Transformer """
#     def __init__(self, raft_args, num_classes=400, embed_dim=768):
#         super().__init__()

#         # self.device = torch.device("cuda") if torch.cuda.is_available() else torch.device("cpu")
#         self.motion_model = self._init_motion_model(raft_args)
#         # self.patch_embed = PatchEmbed(img_size=28, patch_size=2, in_chans=2, embed_dim=768) # Flow patches embedding
#         self.visual_model = self._init_visual_model(raft_args)
#         self.motion_feature_extractor = VisionTransformer(img_size=img_size, num_classes=num_classes, patch_size=patch_size, embed_dim=1024, depth=12, num_heads=16, mlp_ratio=4, qkv_bias=True, norm_layer=partial(nn.LayerNorm, eps=1e-6), drop_rate=0., attn_drop_rate=0., drop_path_rate=0.1, num_frames=num_frames, attention_type=attention_type, **kwargs)

#         # self.crossatt = CustomAttentionWithResidual(embed_size = embed_dim)
#         # self.head = nn.Linear(embed_dim, num_classes) if num_classes > 0 else nn.Identity()

#     def _init_motion_model(self, args):
#         with torch.no_grad():
#             motion_model = torch.nn.DataParallel(RAFT(args))
#             for p in motion_model.parameters():
#                 p.requires_grad = False
#         # motion_model.eval()
#         # motion_model.to(device)    
#         motion_model.load_state_dict(torch.load(args.model))
#         motion_model = motion_model.module
#         return motion_model

#     def _init_visual_model(self, args):
#         # device = torch.device("cuda") if torch.cuda.is_available() else torch.device("cpu")
#         # build model
#         with torch.no_grad():
#             model = vits.__dict__['vit_base'](patch_size=16, num_classes=0)
#             for p in model.parameters():
#                 p.requires_grad = False
#         # model.eval()
#         # model.to(device)
#         print("Since no pretrained weights have been provided, we load the reference pretrained DINO weights.")
#         url = "dino_vitbase16_pretrain/dino_vitbase16_pretrain.pth"
#         state_dict = torch.hub.load_state_dict_from_url(url="https://dl.fbaipublicfiles.com/dino/" + url)
#         model.load_state_dict(state_dict, strict=True)
#         return model

#     def motion_forward(self, x):
#         '''
#             Inputs should have shape of (b, c, t, w, h) Example: (4, 3, 8, 224, 224)
#         '''
#         with torch.no_grad():
#             x = rearrange(x, 'b c t w h -> b t c w h')
#             flow_embs_batches = []
#             for inputs in x: # loop over batches
#                 # denomalized_imgs = denomalizing_img(inputs)
#                 # images = load_image_from_uint8array(denomalized_imgs)
#                 images = denomalizing_img(inputs)
#                 flow_embs = []
#                 for index in range(images.shape[0]-1): #loop over time
#                     image1 = images[index].unsqueeze(0)
#                     image2 = images[index+1].unsqueeze(0)
#                     # print(image1.shape)
#                     padder = InputPadder(image1.shape)
#                     image1, image2 = padder.pad(image1, image2)

#                     flow_low, flow_up = self.motion_model(image1, image2, iters=1, test_mode=True)
#                     # flow_embedding = self.patch_embed(flow_low)
#                     flow_embs.append(flow_low)
                    
#                     # viz(image1, flow_up, index)
#                 flow_embs = torch.stack(flow_embs)
#                 flow_embs_batches.append(flow_embs)
#             ret = torch.stack(flow_embs_batches)
#         return ret

#     def visual_forward(self, inputs):
#         '''
#             Inputs should have shape of (b, c, t, w, h) Example: (4, 3, 8, 224, 224)
#         '''
#         with torch.no_grad():
#             index_of_time_to_get_principle_embeddings = 0
#             img = inputs[:,:,index_of_time_to_get_principle_embeddings,:]
#             features = self.visual_model.get_intermediate_layers(img, len(self.visual_model.blocks))
#             features = features[-1]  # residual stream @ final block
#             # features.requires_grad = False
#         return features
#         # print([features.shape])

#     def forward(self, inputs):
#         with torch.no_grad():
#             visual_embeddings = self.visual_forward(inputs) # b x 1 x 197 x 768
#             motion_embeddings = self.motion_forward(inputs) # b x t x 196 x 768
#             # visual_embeddings.requires_grad = False
#             # motion_embeddings.requires_grad = False
#         # video_embeddings = visual_embeddings
#         # for i in range(motion_embeddings.shape[1]):
#         #     video_embeddings = self.crossatt(video_embeddings, motion_embeddings[:,i,:][1])
#         # x = video_embeddings[:, 0]
#         # x = self.head(x)
#         return visual_embeddings, motion_embeddings

# import torch
# from torch import nn
from einops import rearrange
from torch import einsum

def exists(val):
    return val is not None

def default(val, d):
    return val if exists(val) else d

def ndiag_matrix(dim = 5, n = 2):
    ret = torch.diag(torch.ones(dim), 0)
    for i in range(dim-n, dim):
        ret = ret + torch.diag(torch.ones(i), dim-i) + torch.diag(torch.ones(i), i-dim)
    return ret

def arrow_matrix(size = 8, n = 2):
    dim = size - 1
    ret = torch.diag(torch.ones(dim), 0)
    for j in range(n):
        for i in range(dim-n-j, dim-j):
            ret = ret + torch.diag(torch.ones(i), dim-i) + torch.diag(torch.ones(i), i-dim)
    row = torch.ones(dim, 1)
    col = torch.ones(1, size)

    return torch.concat((col,torch.concat((row, ret), axis=1)), axis=0)

def triage_matrix(n):
    """
    Generates an n x n boolean triage matrix.
    
    Args:
        n (int): Size of the matrix.
    
    Returns:
        np.ndarray: Boolean triage matrix.
    """
    matrix = np.zeros((n, n), dtype=bool)
    
    # Example pattern: Upper triangular matrix
    for i in range(n):
        for j in range(i, n):
            matrix[i, j] = True  # Fill upper triangle with True
    
    return matrix
# bidirectional cross attention - have two sequences attend to each other with 1 attention step

class BidirectionalCrossAttention(nn.Module):
    def __init__(
        self,
        *,
        dim,
        heads = 8,
        dim_head = 64,
        context_dim = None,
        dropout = 0.,
        talking_heads = False,
        prenorm = False
    ):
        super().__init__()
        context_dim = default(context_dim, dim)

        self.norm = nn.LayerNorm(dim) if prenorm else nn.Identity()
        self.context_norm = nn.LayerNorm(context_dim) if prenorm else nn.Identity()

        self.heads = heads
        self.scale = dim_head ** -0.5
        inner_dim = dim_head * heads

        self.dropout = nn.Dropout(dropout)
        self.context_dropout = nn.Dropout(dropout)

        self.to_qk = nn.Linear(dim, inner_dim, bias = False)
        self.context_to_qk = nn.Linear(context_dim, inner_dim, bias = False)

        self.to_v = nn.Linear(dim, inner_dim, bias = False)
        self.context_to_v = nn.Linear(context_dim, inner_dim, bias = False)

        self.to_out = nn.Linear(inner_dim, dim)
        self.context_to_out = nn.Linear(inner_dim, context_dim)

        self.talking_heads = nn.Conv2d(heads, heads, 1, bias = False) if talking_heads else nn.Identity()
        self.context_talking_heads = nn.Conv2d(heads, heads, 1, bias = False) if talking_heads else nn.Identity()

    def forward(
        self,
        x,
        context,
        mask = None,
        context_mask = None,
        return_attn = False,
        rel_pos_bias = None,
        matrix_mask = None
    ):
        b, i, j, h, device = x.shape[0], x.shape[-2], context.shape[-2], self.heads, x.device

        x = self.norm(x)
        context = self.context_norm(context)

        # get shared query/keys and values for sequence and context

        qk, v = self.to_qk(x), self.to_v(x)
        context_qk, context_v = self.context_to_qk(context), self.context_to_v(context)

        # split out head

        qk, context_qk, v, context_v = map(lambda t: rearrange(t, 'b n (h d) -> b h n d', h = h), (qk, context_qk, v, context_v))

        # get similarities

        sim = einsum('b h i d, b h j d -> b h i j', qk, context_qk) * self.scale

        # relative positional bias, if supplied

        if exists(rel_pos_bias):
            sim = sim + rel_pos_bias

        # mask

        if exists(mask) or exists(context_mask):
            mask = default(mask, torch.ones((b, i), device = device, dtype = torch.bool))
            context_mask = default(context_mask, torch.ones((b, j), device = device, dtype = torch.bool))
            if(matrix_mask == None):
                attn_mask = rearrange(mask, 'b i -> b 1 i 1') * rearrange(context_mask, 'b j -> b 1 1 j')
            elif(matrix_mask == 'arrow'):
                # print(i, j)
                attn_mask = arrow_matrix(i, 1).unsqueeze(0).repeat(b, 1, 1).unsqueeze(1).bool().to(device)#rearrange(mask, 'b i -> b 1 i 1') * rearrange(context_mask, 'b j -> b 1 1 j')
                # assert False, "Get Here"
            elif(matrix_mask == 'triag'):
                attn_mask = triage_matrix(i).unsqueeze(0).repeat(b, 1, 1).unsqueeze(1).to(device)#rearrange(mask, 'b i -> b 1 i 1') * rearrange(context_mask, 'b j -> b 1 1 j')
            else:
                assert False, f"No matrix_mask {matrix_mask} found"
            sim = sim.masked_fill(~attn_mask, -torch.finfo(sim.dtype).max)
        elif(matrix_mask == 'triag'):
            attn_mask = triage_matrix(i).unsqueeze(0).repeat(b, 1, 1).unsqueeze(1).to(device)#rearrange(mask, 'b i -> b 1 i 1') * rearrange(context_mask, 'b j -> b 1 1 j')
            sim = sim.masked_fill(~attn_mask, -torch.finfo(sim.dtype).max)
            assert False, "code get here!!"
        # get attention along both sequence length and context length dimensions
        # shared similarity matrix

        attn = sim.softmax(dim = -1)
        context_attn = sim.softmax(dim = -2)

        # dropouts

        attn = self.dropout(attn)
        context_attn = self.context_dropout(context_attn)

        # talking heads

        attn = self.talking_heads(attn)
        context_attn = self.context_talking_heads(context_attn)

        # src sequence aggregates values from context, context aggregates values from src sequence

        out = einsum('b h i j, b h j d -> b h i d', attn, context_v)
        context_out = einsum('b h j i, b h j d -> b h i d', context_attn, v)

        # merge heads and combine out

        out, context_out = map(lambda t: rearrange(t, 'b h n d -> b n (h d)'), (out, context_out))

        out = self.to_out(out)
        context_out = self.context_to_out(context_out)

        if return_attn:
            return out, context_out, attn, context_attn

        return out, context_out
    

class CausalSelfAttention(nn.Module):

    def __init__(
        self,
        d,
        H,
        T,
        bias=False,
        dropout=0.2,
    ):
        """
        Arguments:
        d: size of embedding dimension
        H: number of attention heads
        T: maximum length of input sequences (in tokens)
        bias: whether or not to use bias in linear layers
        dropout: probability of dropout
        """
        super().__init__()
        assert d % H == 0

        # key, query, value projections for all heads, but in a batch
        # output is 3X the dimension because it includes key, query and value
        self.c_attn = nn.Linear(d, 3*d, bias=bias)

        # projection of concatenated attention head outputs
        self.c_proj = nn.Linear(d, d, bias=bias)

        # dropout modules
        self.attn_dropout = nn.Dropout(dropout)
        self.resid_dropout = nn.Dropout(dropout)
        self.H = H
        self.d = d

        # causal mask to ensure that attention is only applied to
        # the left in the input sequence
        self.register_buffer("mask", torch.tril(torch.ones(T, T))
                                    .view(1, 1, T, T))

    def forward(self, x):
        B, T, _ = x.size() # batch size, sequence length, embedding dimensionality

        # compute query, key, and value vectors for all heads in batch
        # split the output into separate query, key, and value tensors
        q, k, v  = self.c_attn(x).split(self.d, dim=2) # [B, T, d]

        # reshape tensor into sequences of smaller token vectors for each head
        k = k.view(B, T, self.H, self.d // self.H).transpose(1, 2) # [B, H, T, d // H]
        q = q.view(B, T, self.H, self.d // self.H).transpose(1, 2)
        v = v.view(B, T, self.H, self.d // self.H).transpose(1, 2)

        # compute the attention matrix, perform masking, and apply dropout
        att = (q @ k.transpose(-2, -1)) * (1.0 / math.sqrt(k.size(-1))) # [B, H, T, T]
        att = att.masked_fill(self.mask[:,:,:T,:T] == 0, float('-inf'))
        att = F.softmax(att, dim=-1)
        att = self.attn_dropout(att)

        # compute output vectors for each token
        y = att @ v # [B, H, T, d // H]

        # concatenate outputs from each attention head and linearly project
        y = y.transpose(1, 2).contiguous().view(B, T, self.d)
        y = self.resid_dropout(self.c_proj(y))
        return y