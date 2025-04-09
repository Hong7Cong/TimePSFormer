# Copyright (c) Facebook, Inc. and its affiliates. All Rights Reserved.
# Copyright 2020 Ross Wightman
# Modified Model definition

import torch
import torch.nn as nn
from functools import partial
import math
import warnings
import torch.nn.functional as F
import numpy as np
from torch import inf
from timesformer.models.vit_utils import IMAGENET_DEFAULT_MEAN, IMAGENET_DEFAULT_STD
from timesformer.models.helpers import load_pretrained
from timesformer.models.vit_utils import DropPath, to_2tuple, trunc_normal_
# import logging
from .build import MODEL_REGISTRY
from torch import einsum
from einops import rearrange, reduce, repeat
import sys
from mmpretrain import FeatureExtractor, get_model
# import torch_xla.core.xla_model as xm
# import torch_xla.core.xla_model as xm
# device = xm.xla_device()
# sys.path.append('/home/hongn/sapiens/pretrain')
# sys.path.append('/mnt/c/Users/PCM/Documents/GitHub/VideoUnderstanding/sapiens/pretrain/demo')
# from extract_feature import *

def _cfg(url='', **kwargs):
    return {
        'url': url,
        'num_classes': 1000, 'input_size': (3, 224, 224), 'pool_size': None,
        'crop_pct': .9, 'interpolation': 'bicubic',
        'mean': IMAGENET_DEFAULT_MEAN, 'std': IMAGENET_DEFAULT_STD,
        'first_conv': 'patch_embed.proj', 'classifier': 'head',
        **kwargs
    }


default_cfgs = {
    'vit_base_patch16_224': _cfg(
        url='https://github.com/rwightman/pytorch-image-models/releases/download/v0.1-vitjx/jx_vit_base_p16_224-80ecf9dd.pth',
        mean=(0.5, 0.5, 0.5), std=(0.5, 0.5, 0.5),
    ),
}


class Mlp(nn.Module):
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

class Attention(nn.Module):
    def __init__(self, dim, num_heads=8, qkv_bias=False, qk_scale=None, attn_drop=0., proj_drop=0., with_qkv=True):
        super().__init__()
        self.num_heads = num_heads
        head_dim = dim // num_heads
        self.scale = qk_scale or head_dim ** -0.5
        self.with_qkv = with_qkv
        if self.with_qkv:
           self.qkv = nn.Linear(dim, dim * 3, bias=qkv_bias)
           self.proj = nn.Linear(dim, dim)
           self.proj_drop = nn.Dropout(proj_drop)
        self.attn_drop = nn.Dropout(attn_drop)

    def forward(self, x):
        B, N, C = x.shape
        if self.with_qkv:
        #    print('[The qkc dimentions]', B, N, 3, self.num_heads, C // self.num_heads, C)
           qkv = self.qkv(x).reshape(B, N, 3, self.num_heads, C // self.num_heads).permute(2, 0, 3, 1, 4)
           q, k, v = qkv[0], qkv[1], qkv[2]
        else:
           qkv = x.reshape(B, N, self.num_heads, C // self.num_heads).permute(0, 2, 1, 3)
           q, k, v  = qkv, qkv, qkv

        attn = (q @ k.transpose(-2, -1)) * self.scale
        attn = attn.softmax(dim=-1)
        attn = self.attn_drop(attn)

        x = (attn @ v).transpose(1, 2).reshape(B, N, C)
        if self.with_qkv:
           x = self.proj(x)
           x = self.proj_drop(x)
        return x

class Block(nn.Module):

    def __init__(self, dim, num_heads, mlp_ratio=4., qkv_bias=False, qk_scale=None, drop=0., attn_drop=0.,
                 drop_path=0.1, act_layer=nn.GELU, norm_layer=nn.LayerNorm, attention_type='divided_space_time', pretrain_attn = None, pretrain_spatial_embs = None):
        super().__init__()
        self.attention_type = attention_type
        assert(attention_type in ['divided_space_time', 'space_only','joint_space_time', 'time_only'])
        self.norm1 = norm_layer(dim)
        self.attn = Attention(
           dim, num_heads=num_heads, qkv_bias=qkv_bias, qk_scale=qk_scale, attn_drop=attn_drop, proj_drop=drop)
        
        ## Pretrain LVm modules
        self.pretrain_attn = pretrain_attn
        self.pretrain_spatial_embs = pretrain_spatial_embs
        # stride = 64//28
        # kernelsize = 64-(28-1)*stride
        # m = nn.MaxPool2d(kernelsize, stride=stride)
        # m = nn.MaxPool2d((kernelsize, kernelsize), stride=(stride, stride))
        # self.x_spatial = m(pretrain_spatial_embs)

        ## Temporal Attention Parameters
        if self.attention_type in ['divided_space_time', 'time_only']:
            self.temporal_norm1 = norm_layer(dim)
            self.temporal_attn = Attention(
              dim, num_heads=num_heads, qkv_bias=qkv_bias, qk_scale=qk_scale, attn_drop=attn_drop, proj_drop=drop)
            self.temporal_fc = nn.Linear(dim, dim)

        ## drop path
        self.drop_path = DropPath(drop_path) if drop_path > 0. else nn.Identity()
        self.norm2 = norm_layer(dim)
        mlp_hidden_dim = int(dim * mlp_ratio)
        self.mlp = Mlp(in_features=dim, hidden_features=mlp_hidden_dim, act_layer=act_layer, drop=drop)


    def forward(self, x, B, T, W, x_spatial = None):
        num_spatial_tokens = (x.size(1) - 1) // T
        H = num_spatial_tokens // W

        if self.attention_type in ['space_only', 'joint_space_time']:
            x = x + self.drop_path(self.attn(self.norm1(x)))
            x = x + self.drop_path(self.mlp(self.norm2(x)))
            return x
        elif self.attention_type == 'divided_space_time':
            ## Temporal
            xt = x[:,1:,:]
            xt = rearrange(xt, 'b (h w t) m -> (b h w) t m',b=B,h=H,w=W,t=T)
            res_temporal = self.drop_path(self.temporal_attn(self.temporal_norm1(xt)))
            res_temporal = rearrange(res_temporal, '(b h w) t m -> b (h w t) m',b=B,h=H,w=W,t=T)
            res_temporal = self.temporal_fc(res_temporal)
            xt = x[:,1:,:] + res_temporal

            ## Spatial
            init_cls_token = x[:,0,:].unsqueeze(1)
            cls_token = init_cls_token.repeat(1, T, 1)
            cls_token = rearrange(cls_token, 'b t m -> (b t) m',b=B,t=T).unsqueeze(1)
            xs = xt
            xs = rearrange(xs, 'b (h w t) m -> (b t) (h w) m',b=B,h=H,w=W,t=T)
            xs = torch.cat((cls_token, xs), 1)
            res_spatial = self.drop_path(self.attn(self.norm1(xs)))

            ### Taking care of CLS token
            cls_token = res_spatial[:,0,:]
            cls_token = rearrange(cls_token, '(b t) m -> b t m',b=B,t=T)
            cls_token = torch.mean(cls_token,1,True) ## averaging for every frame
            res_spatial = res_spatial[:,1:,:]
            res_spatial = rearrange(res_spatial, '(b t) (h w) m -> b (h w t) m',b=B,h=H,w=W,t=T)
            res = res_spatial
            x = xt

            ## Mlp
            x = torch.cat((init_cls_token, x), 1) + torch.cat((cls_token, res), 1)
            x = x + self.drop_path(self.mlp(self.norm2(x)))
            return x

        elif self.attention_type == 'time_only':
            ## Temporal
            xt = x[:,1:,:]
            xt = rearrange(xt, 'b (h w t) m -> (b h w) t m',b=B,h=H,w=W,t=T)
            res_temporal = self.drop_path(self.temporal_attn(self.temporal_norm1(xt)))
            res_temporal = rearrange(res_temporal, '(b h w) t m -> b (h w t) m',b=B,h=H,w=W,t=T)
            res_temporal = self.temporal_fc(res_temporal)
            # xt = x[:,1:,:] + res_temporal

            ## Spatial
            init_cls_token = x[:,0,:].unsqueeze(1)
            cls_token = x_spatial[:,0,:]
            cls_token = rearrange(cls_token, '(b t) m -> b t m',b=B,t=T)
            cls_token = torch.mean(cls_token,1,True) ## averaging for every frame
            xs = x_spatial[:, 1:, :]
            xs = rearrange(xs, '(b t) (h w) m -> b (h w t) m',b=B,h=H,w=W,t=T)

            ## Mlp
            # print(res_temporal.device.type, xs.device.type)
            # print(xs.shape, res_temporal.shape, init_cls_token.shape, cls_token.shape)
            x = torch.cat((init_cls_token, res_temporal), 1) + torch.cat((cls_token, xs), 1)
            x = x + self.drop_path(self.mlp(self.norm2(x)))
            return x


class PatchEmbed(nn.Module):
    """ Image to Patch Embedding
    """
    def __init__(self, img_size=224, patch_size=16, in_chans=3, embed_dim=1024):
        super().__init__()
        # print('-----------', img_size, patch_size)
        img_size =  to_2tuple(img_size)
        patch_size =  to_2tuple(patch_size)
        # print('-----------', img_size[0], img_size[1], patch_size[0], patch_size[1])
        num_patches = (img_size[1] // patch_size[1]) * (img_size[0] // patch_size[0])
        self.img_size = img_size
        self.patch_size = patch_size
        self.num_patches = num_patches

        self.proj = nn.Conv2d(in_chans, embed_dim, kernel_size=patch_size, stride=patch_size)

    def forward(self, x):
        B, C, T, H, W = x.shape
        x = rearrange(x, 'b c t h w -> (b t) c h w')
        x = self.proj(x)
        W = x.size(-1)
        x = x.flatten(2).transpose(1, 2)
        return x, T, W

# class PreVisual_PatchEmbed(nn.Module):
#     """ Image to Patch Embedding
#     """
#     def __init__(self, inferencer, img_size=1024, patch_size=16, in_chans=3, embed_dim=1024):
#         super().__init__()
#         img_size = to_2tuple(img_size)
#         patch_size = to_2tuple(patch_size)
#         num_patches = (img_size[1] // patch_size[1]) * (img_size[0] // patch_size[0])
#         self.img_size = img_size
#         self.patch_size = patch_size
#         self.num_patches = num_patches

#         self.proj = inferencer.model.backbone.patch_embed.projection.to('cpu')

#     def forward(self, x):
#         B, C, T, H, W = x.shape
#         x = rearrange(x, 'b c t h w -> (b t) c h w')
#         x = self.proj(x)
#         # print(x.shape)
#         # # from 64x64 to 14x14 by pooling
#         # stride = 64//14
#         # kernelsize = 64-(14-1)*stride
#         # m = nn.MaxPool2d(kernelsize, stride=stride)
#         # m = nn.MaxPool2d((kernelsize, kernelsize), stride=(stride, stride))
#         # x = m(x)
#         # print(x.shape)
#         W = x.size(-1)
#         x = x.flatten(2).transpose(1, 2)
#         return x, T, W

class VisionTransformer(nn.Module):
    """ Vision Transformere
    """
    # def __init__(self, img_size=224, patch_size=16, in_chans=3, num_classes=1000, embed_dim=768, depth=12, pretrain_space_embs = None,
    #              num_heads=12, mlp_ratio=4., qkv_bias=False, qk_scale=None, drop_rate=0., attn_drop_rate=0.,
    #              drop_path_rate=0.1, hybrid_backbone=None, norm_layer=nn.LayerNorm, num_frames=8, attention_type='divided_space_time', dropout=0.):
    def __init__(self, img_size=1024, patch_size=16, in_chans=3, num_classes=1000, embed_dim=1024, depth=12, pretrain_space_embs_path = "/data2/hongn/sapiens/pretrain/checkpoints/sapiens_0.3b/sapiens_0.3b_epoch_1600_clean.pth",
                num_heads=12, mlp_ratio=4., qkv_bias=False, qk_scale=None, drop_rate=0., attn_drop_rate=0.,
                drop_path_rate=0.1, hybrid_backbone=None, norm_layer=nn.LayerNorm, num_frames=8, attention_type='space_only', dropout=0.):
        super().__init__()
        self.attention_type = attention_type
        self.depth = depth
        self.pretrain_space_embs_path = pretrain_space_embs_path
        self.dropout = nn.Dropout(dropout)
        self.num_classes = num_classes
        self.num_features = self.embed_dim = embed_dim  # num_features for consistency with other models

        ## Load Sapiens   
        # import os
        # import time
        # from argparse import ArgumentParser

        if(self.attention_type == 'time_only'):
            assert self.pretrain_space_embs_path != None, "Please input pretrain_space_embs=path/to/pretrain_sapiens_models"
            model_config = "/data2/hongn/sapiens/pretrain/configs/sapiens_mae/humans_300m_test/mae_sapiens_0.3b-p16_8xb512-coslr-1600e_humans_300m_test.py"
            # checkpoint = "/mnt/c/Users/PCM/Documents/GitHub/VideoUnderstanding/sapiens/pretrain/checkpoints/sapiens_0.3b/sapiens_0.3b_epoch_1600_clean.pth"
            with torch.no_grad():
                self.pretrain_space_embs = get_model(model=model_config, pretrained=self.pretrain_space_embs_path, device='cpu', backbone=dict(out_indices=(0, 2, 4, 6, 8, 10, 12, 14, 16, 18, 20, 22, 24)))
                self.pretrain_space_embs.requires_grad = False
            # self.pretrain_space_embs.model.backbone.out_type = 'featmap'
            # results, inputs, outputs = self.pretrain_space_embs(image_path)
        ## Patch Embeddings

        self.patch_embed = PatchEmbed(
            img_size=img_size, patch_size=patch_size, in_chans=in_chans, embed_dim=self.embed_dim) #if(self.attention_type != 'time_only') else PreVisual_PatchEmbed(self.pretrain_space_embs, img_size=img_size, patch_size=patch_size, in_chans=in_chans, embed_dim=embed_dim)
        num_patches = self.patch_embed.num_patches

        ## Positional Embeddings
        self.cls_token = nn.Parameter(torch.zeros(1, 1, embed_dim))
        self.pos_embed = nn.Parameter(torch.zeros(1, num_patches+1, embed_dim))
        self.pos_drop = nn.Dropout(p=drop_rate)
        if self.attention_type != 'space_only':
            self.time_embed = nn.Parameter(torch.zeros(1, num_frames, embed_dim))
            self.time_drop = nn.Dropout(p=drop_rate)

        ## Attention Blocks
        dpr = [x.item() for x in torch.linspace(0, drop_path_rate, self.depth)]  # stochastic depth decay rule
        self.blocks = nn.ModuleList([
            Block(
                dim=embed_dim, num_heads=num_heads, mlp_ratio=mlp_ratio, qkv_bias=qkv_bias, qk_scale=qk_scale,
                drop=drop_rate, attn_drop=attn_drop_rate, drop_path=dpr[i], norm_layer=norm_layer, attention_type=self.attention_type)
            for i in range(self.depth)])
        self.norm = norm_layer(embed_dim)

        # Classifier head
        self.head = nn.Linear(embed_dim, num_classes) if num_classes > 0 else nn.Identity()

        trunc_normal_(self.pos_embed, std=.02)
        trunc_normal_(self.cls_token, std=.02)
        self.apply(self._init_weights)

        ## initialization of temporal attention weights
        if self.attention_type in ['divided_space_time', 'time_only']:
            i = 0
            for m in self.blocks.modules():
                m_str = str(m)
                if 'Block' in m_str:
                    if i > 0:
                      nn.init.constant_(m.temporal_fc.weight, 0)
                      nn.init.constant_(m.temporal_fc.bias, 0)
                    i += 1

    def _init_weights(self, m):
        if isinstance(m, nn.Linear):
            trunc_normal_(m.weight, std=.02)
            if isinstance(m, nn.Linear) and m.bias is not None:
                nn.init.constant_(m.bias, 0)
        elif isinstance(m, nn.LayerNorm):
            nn.init.constant_(m.bias, 0)
            nn.init.constant_(m.weight, 1.0)

    @torch.jit.ignore
    def no_weight_decay(self):
        return {'pos_embed', 'cls_token', 'time_embed'}

    def get_classifier(self):
        return self.head

    def reset_classifier(self, num_classes, global_pool=''):
        self.num_classes = num_classes
        self.head = nn.Linear(self.embed_dim, num_classes) if num_classes > 0 else nn.Identity()

    def forward_features(self, x):
        ## pre-extract [2, 4, 6, ..., 24] embeddings from pretrained Sapiens as frozen space embs
        if self.attention_type == 'time_only':
            with torch.no_grad():
                x_temp = rearrange(x, 'b c t h w -> (b t) c h w').detach()
                x_spatial = self.pretrain_space_embs(x_temp)
            # x_spatial.requires_grad = False
        # print('x_spatial len', len(x_spatial))

        ## Start regular TimeSFormer
        B = x.shape[0]
        x, T, W = self.patch_embed(x)
        cls_tokens = self.cls_token.expand(x.size(0), -1, -1)
        # print(cls_tokens.shape, x.shape, T, W)
        x = torch.cat((cls_tokens, x), dim=1)

        ## resizing the positional embeddings in case they don't match the input at inference
        if x.size(1) != self.pos_embed.size(1):
            pos_embed = self.pos_embed
            cls_pos_embed = pos_embed[0,0,:].unsqueeze(0).unsqueeze(1)
            other_pos_embed = pos_embed[0,1:,:].unsqueeze(0).transpose(1, 2)
            P = int(other_pos_embed.size(2) ** 0.5)
            H = x.size(1) // W
            other_pos_embed = other_pos_embed.reshape(1, x.size(2), P, P)
            new_pos_embed = F.interpolate(other_pos_embed, size=(H, W), mode='nearest')
            new_pos_embed = new_pos_embed.flatten(2)
            new_pos_embed = new_pos_embed.transpose(1, 2)
            new_pos_embed = torch.cat((cls_pos_embed, new_pos_embed), 1)
            x = x + new_pos_embed
        else:
            x = x + self.pos_embed
        x = self.pos_drop(x)


        ## Time Embeddings
        if self.attention_type != 'space_only':
            cls_tokens = x[:B, 0, :].unsqueeze(1)
            x = x[:,1:]
            x = rearrange(x, '(b t) n m -> (b n) t m',b=B,t=T)
            ## Resizing time embeddings in case they don't match
            if T != self.time_embed.size(1):
                time_embed = self.time_embed.transpose(1, 2)
                new_time_embed = F.interpolate(time_embed, size=(T), mode='nearest')
                new_time_embed = new_time_embed.transpose(1, 2)
                x = x + new_time_embed
            else:
                x = x + self.time_embed
            x = self.time_drop(x)
            x = rearrange(x, '(b n) t m -> b (n t) m',b=B,t=T)
            x = torch.cat((cls_tokens, x), dim=1)

        ## Attention blocks
        if(self.attention_type == 'time_only'):
            for idx, blk in enumerate(self.blocks):
                # print(idx)
                x = blk(x, B, T, W, x_spatial[idx].detach())
        else:
            for idx, blk in enumerate(self.blocks):
                # print(idx)
                x = blk(x, B, T, W)

        ## Free redundant space in TPU
        # del x_spatial

        ### Predictions for space-only baseline
        if self.attention_type == 'space_only':
            x = rearrange(x, '(b t) n m -> b t n m',b=B,t=T)
            x = torch.mean(x, 1) # averaging predictions for every frame

        x = self.norm(x)
        return x#[:, 0]

    def forward(self, x):
        x = self.forward_features(x)
        x = self.head(x[:, 0])
        return x


def _conv_filter(state_dict, patch_size=16):
    """ convert patch embedding weight from manual patchify + linear proj to conv"""
    out_dict = {}
    for k, v in state_dict.items():
        if 'patch_embed.proj.weight' in k:
            if v.shape[-1] != patch_size:
                patch_size = v.shape[-1]
            v = v.reshape((v.shape[0], 3, patch_size, patch_size))
        out_dict[k] = v
    return out_dict

@MODEL_REGISTRY.register()
class vit_base_patch16_224(nn.Module):
    def __init__(self, cfg, **kwargs):
        super(vit_base_patch16_224, self).__init__()
        self.pretrained=True
        patch_size = 16
        self.model = VisionTransformer(img_size=cfg.DATA.TRAIN_CROP_SIZE, num_classes=cfg.MODEL.NUM_CLASSES, patch_size=patch_size, embed_dim=768, depth=12, num_heads=12, mlp_ratio=4, qkv_bias=True, norm_layer=partial(nn.LayerNorm, eps=1e-6), drop_rate=0., attn_drop_rate=0., drop_path_rate=0.1, num_frames=cfg.DATA.NUM_FRAMES, attention_type=cfg.TIMESFORMER.ATTENTION_TYPE, **kwargs)

        self.attention_type = cfg.TIMESFORMER.ATTENTION_TYPE
        self.model.default_cfg = default_cfgs['vit_base_patch16_224']
        self.num_patches = (cfg.DATA.TRAIN_CROP_SIZE // patch_size) * (cfg.DATA.TRAIN_CROP_SIZE // patch_size)
        pretrained_model=cfg.TIMESFORMER.PRETRAINED_MODEL
        if self.pretrained:
            load_pretrained(self.model, num_classes=self.model.num_classes, in_chans=kwargs.get('in_chans', 3), filter_fn=_conv_filter, img_size=cfg.DATA.TRAIN_CROP_SIZE, num_patches=self.num_patches, attention_type=self.attention_type, pretrained_model=pretrained_model)

    def forward(self, x):
        x = self.model(x)
        return x
    
@MODEL_REGISTRY.register()
class vit_base_PS_224(nn.Module):
    def __init__(self, cfg, **kwargs):
        super(vit_base_PS_224, self).__init__()
        self.pretrained=False
        patch_size = 16
        self.model = VisionTransformer(img_size=cfg.DATA.TRAIN_CROP_SIZE, num_classes=cfg.MODEL.NUM_CLASSES, patch_size=patch_size, embed_dim=1024, depth=12, num_heads=16, mlp_ratio=4, qkv_bias=True, norm_layer=partial(nn.LayerNorm, eps=1e-6), drop_rate=0., attn_drop_rate=0., drop_path_rate=0.1, num_frames=cfg.DATA.NUM_FRAMES, attention_type=cfg.TIMESFORMER.ATTENTION_TYPE, **kwargs)

        self.attention_type = cfg.TIMESFORMER.ATTENTION_TYPE
        self.model.default_cfg = default_cfgs['vit_base_patch16_224']
        self.num_patches = (cfg.DATA.TRAIN_CROP_SIZE // patch_size) * (cfg.DATA.TRAIN_CROP_SIZE // patch_size)
        pretrained_model=cfg.TIMESFORMER.PRETRAINED_MODEL
        if self.pretrained:
            load_pretrained(self.model, num_classes=self.model.num_classes, in_chans=kwargs.get('in_chans', 3), filter_fn=_conv_filter, img_size=cfg.DATA.TRAIN_CROP_SIZE, num_patches=self.num_patches, attention_type=self.attention_type, pretrained_model=pretrained_model)

    def forward(self, x):
        x = self.model(x)
        return x

@MODEL_REGISTRY.register()
class TimeSformer(nn.Module):
    def __init__(self, img_size=224, patch_size=16, num_classes=400, num_frames=8, attention_type='divided_space_time',  pretrained_model='', **kwargs):
        super(TimeSformer, self).__init__()
        self.pretrained=True
        self.model = VisionTransformer(img_size=img_size, num_classes=num_classes, patch_size=patch_size, embed_dim=768, depth=12, num_heads=12, mlp_ratio=4, qkv_bias=True, norm_layer=partial(nn.LayerNorm, eps=1e-6), drop_rate=0., attn_drop_rate=0., drop_path_rate=0.1, num_frames=num_frames, attention_type=attention_type, **kwargs)

        self.attention_type = attention_type
        self.model.default_cfg = default_cfgs['vit_base_patch'+str(patch_size)+'_224']
        self.num_patches = (img_size // patch_size) * (img_size // patch_size)
        if self.pretrained:
            load_pretrained(self.model, num_classes=self.model.num_classes, in_chans=kwargs.get('in_chans', 3), filter_fn=_conv_filter, img_size=img_size, num_frames=num_frames, num_patches=self.num_patches, attention_type=self.attention_type, pretrained_model=pretrained_model)
    def forward(self, x):
        x = self.model(x)
        return x
    
@MODEL_REGISTRY.register()
class TimePSformer(nn.Module):
    def __init__(self, img_size=224, patch_size=16, num_classes=400, num_frames=8, attention_type='divided_space_time',  pretrained_model='', **kwargs):
        super(TimePSformer, self).__init__()
        self.pretrained=False
        self.model = VisionTransformer(img_size=img_size, num_classes=num_classes, patch_size=patch_size, embed_dim=1024, depth=12, num_heads=16, mlp_ratio=4, qkv_bias=True, norm_layer=partial(nn.LayerNorm, eps=1e-6), drop_rate=0., attn_drop_rate=0., drop_path_rate=0.1, num_frames=num_frames, attention_type=attention_type, **kwargs)

        self.attention_type = attention_type
        # self.model.default_cfg = default_cfgs['vit_base_patch'+str(patch_size)+'_224']
        self.num_patches = (img_size // patch_size) * (img_size // patch_size)
        if self.pretrained:
            load_pretrained(self.model, num_classes=self.model.num_classes, in_chans=kwargs.get('in_chans', 3), filter_fn=_conv_filter, img_size=img_size, num_frames=num_frames, num_patches=self.num_patches, attention_type=self.attention_type, pretrained_model=pretrained_model)
    def forward(self, x):
        x = self.model(x)
        return x
    

class MotionPatchEmbed(nn.Module):
    """ Image to Patch Embedding
    """
    def __init__(self, img_size=224, patch_size=16, in_chans=3, embed_dim=768):
        super().__init__()
        num_patches = (img_size // patch_size) * (img_size // patch_size)
        self.img_size = img_size
        self.patch_size = patch_size
        self.num_patches = num_patches

        self.proj = nn.Conv2d(in_chans, embed_dim, kernel_size=patch_size, stride=patch_size)

    def forward(self, x):
        B, C, H, W = x.shape
        x = self.proj(x).flatten(2).transpose(1, 2)
        return x
    
from transformers import AutoImageProcessor, AutoModel
from timesformer.models.moose import *
from torchvision.transforms import Resize

class MOOSE_Encoder(nn.Module):
    """ Vision Transformer """
    def __init__(self, raft_args, cfg):
        super().__init__()
        self.cfg = cfg
        # self.device = torch.device("cuda") if torch.cuda.is_available() else torch.device("cpu")
        self.motion_model = self._init_motion_model(raft_args)
        # self.patch_embed = PatchEmbed(img_size=28, patch_size=2, in_chans=2, embed_dim=768) # Flow patches embedding
        self.visual_model = self._init_visual_model()
        # self.motion_feature_extractor = VisionTransformer(img_size=img_size, num_classes=num_classes, patch_size=motion_patch_size, embed_dim=embed_dim, depth=3, num_heads=12, mlp_ratio=1, qkv_bias=True, norm_layer=partial(nn.LayerNorm, eps=1e-6), drop_rate=0., attn_drop_rate=0., drop_path_rate=0.1, num_frames=num_frames)

        # self.crossatt = CustomAttentionWithResidual(embed_size = embed_dim)
        # self.head = nn.Linear(embed_dim, num_classes) if num_classes > 0 else nn.Identity()

    def _init_motion_model(self, args):
        with torch.no_grad():
            motion_model = torch.nn.DataParallel(RAFT(args))
            for p in motion_model.parameters():
                p.requires_grad = False   
        motion_model.load_state_dict(torch.load(args.model))
        motion_model = motion_model.module
        return motion_model

    def _init_visual_model(self):
        if(self.cfg.MODEL.VISUAL_MODEL == "dino"):
            with torch.no_grad():
                vi_model = vits.__dict__['vit_base'](patch_size=16, num_classes=0)
                for p in model.parameters():
                    p.requires_grad = False
            print("Since no pretrained weights have been provided, we load the reference pretrained DINO weights.")
            url = "dino_vitbase16_pretrain/dino_vitbase16_pretrain.pth"
            state_dict = torch.hub.load_state_dict_from_url(url="https://dl.fbaipublicfiles.com/dino/" + url)
            model.load_state_dict(state_dict, strict=True)
        elif(self.cfg.MODEL.VISUAL_MODEL == "dinov2"):
            with torch.no_grad():
                model = torch.hub.load('facebookresearch/dinov2', 'dinov2_vitb14_reg')
        elif(self.cfg.MODEL.VISUAL_MODEL == "sapiens"):
            with torch.no_grad():
                model = get_sapiens()
        else:
            assert False, f"no cfg.MODEL.VISUAL_MODEL == {self.cfg.MODEL.VISUAL_MODEL} found!"
        return model

    def motion_forward(self, x):
        '''
            Inputs should have shape of (b, c, t, w, h) Example: (4, 3, 8, 224, 224)
        '''
        if isinstance(x, (list,)):
            x = x[0]

        assert len(x.shape) == 5, "Input should have len = 5, represent (b, c, t, w, h)"
        b = x.shape[0]
        t = x.shape[2]
        # with torch.no_grad():
        #     x = rearrange(x, 'b c t w h -> b t c w h')
        #     flow_embs_batches = []
        #     for inputs in x: # loop over batches
        #         # denomalized_imgs = denomalizing_img(inputs)
        #         # images = load_image_from_uint8array(denomalized_imgs)
        #         images = denomalizing_img(inputs)
        #         flow_embs = []
        #         for index in range(images.shape[0]-1): #loop over time
        #             image1 = images[index].unsqueeze(0) #[1,1,c,w,h]
        #             image2 = images[index+1].unsqueeze(0)
        #             # print(image1.shape)
        #             padder = InputPadder(image1.shape)
        #             image1, image2 = padder.pad(image1, image2)

        #             flow_low, flow_up = self.motion_model(image1, image2, iters=1, test_mode=True)
        #             # flow_embedding = self.patch_embed(flow_low)
        #             flow_embs.append(flow_up)
                    
        #             # viz(image1, flow_up, index)
        #         flow_embs = torch.stack(flow_embs)
        #         flow_embs_batches.append(flow_embs)
        #     ret = torch.stack(flow_embs_batches)
        # return ret
        with torch.no_grad():
            x = rearrange(x, 'b c t w h -> (b t) c w h', b=b, t=t)
            x = denomalizing_img(x)
            if(self.cfg.MODEL.VISUAL_MODEL == "sapiens"):
                x = Resize(224)(x)
            x = rearrange(x, '(b t) c w h -> b t c w h', b=b, t=t)
            # print(f'b = {b}, t = {t}')
            # flow_embs_batches = []
            flow_embs = []
            for index in range(t-1): #loop over time
                image1 = x[:,index,:]
                image2 = x[:,index+1,:]
                # print(image1.shape)
                padder = InputPadder(image1.shape)
                image1, image2 = padder.pad(image1, image2)

                flow_low, flow_up = self.motion_model(image1, image2, iters=1, test_mode=True)
                if(self.cfg.MODEL.VISUAL_MODEL == "sapiens"):
                    flow_up = Resize(1024)(flow_up)
                flow_embs.append(flow_up)
            flow_embs = torch.stack(flow_embs)

            ret = rearrange(flow_embs, 't b c w h -> b t c w h') #torch.stack(flow_embs)
        return ret
    
    def visual_forward(self, inputs):
        '''
            Inputs should have shape of (b, c, t, w, h) Example: (4, 3, 8, 224, 224)
        '''
        if isinstance(inputs, (list,)):
            inputs = inputs[0]

        # print(inputs)
        assert len(inputs.shape) == 5, f"Input should have len = 5, represent (b, c, t, w, h) but {inputs.shape}"
        b = inputs.shape[0]
        t = inputs.shape[2]
        if(self.cfg.MODEL.VISUAL_MODEL == "dino"):
            with torch.no_grad():
                index_of_time_to_get_principle_embeddings = 0 # only get the first frame for now
                img = rearrange(inputs, 'b c t w h -> (b t) c w h', b=b, t=t)#inputs[:,:,index_of_time_to_get_principle_embeddings,:]
                features = self.visual_model.get_intermediate_layers(img, len(self.visual_model.blocks))
                features = features[-1]  # residual stream @ final block
        elif(self.cfg.MODEL.VISUAL_MODEL == "dinov2"):
            with torch.no_grad():
                index_of_time_to_get_principle_embeddings = 0 # only get the first frame for now
                img = rearrange(inputs, 'b c t w h -> (b t) c w h', b=b, t=t) #inputs[:,:,index_of_time_to_get_principle_embeddings,:]
                output = self.visual_model.forward_features(img)
                features = torch.concat((output['x_norm_clstoken'].unsqueeze(1), output['x_norm_patchtokens']), dim=1)
        elif(self.cfg.MODEL.VISUAL_MODEL == "sapiens"):
            img = rearrange(inputs, 'b c t w h -> (b t) c w h', b=b, t=t) #inputs[:,:,index_of_time_to_get_principle_embeddings,:]
            features = self.visual_model(img)[-1] # [b, 4097, 1024]
        else:
            assert False, f"no VISUAL_MODEL name {self.cfg.MODEL.VISUAL_MODEL} found"
            # features.requires_grad = False
        features = rearrange(features, '(b t) p d -> b t p d', b=b, t=t)    
        return features
        # print([features.shape])

    def forward(self, inputs):
        with torch.no_grad():
            visual_embeddings = self.visual_forward(inputs) # b x 1 x 197 x 768
            motion_embeddings = self.motion_forward(inputs) # [b, t, 1, 2, 28, 28]
        return visual_embeddings, motion_embeddings

import argparse
parser = argparse.ArgumentParser()
parser.add_argument('--model', help="restore checkpoint")
parser.add_argument('--path', help="dataset for evaluation")
parser.add_argument('--small', action='store_true', help='use small model')
parser.add_argument('--mixed_precision', action='store_true', help='use mixed precision')
parser.add_argument('--alternate_corr', action='store_true', help='use efficent correlation implementation')

raft_args = parser.parse_args(['--model', '/data2/hongn/RAFT/models/raft-things.pth', 
                        '--path', '/data2/hongn/RAFT/demo-frames/care'])
# from timesformer.models.moose import BidirectionalCrossAttention
from mamba_ssm import Mamba

@MODEL_REGISTRY.register()
class MOOSE(nn.Module):
    def __init__(self, cfg, raft_args = raft_args, norm_layer=partial(nn.LayerNorm, eps=1e-6), mlp_ratio=1., act_layer=nn.GELU, drop=0., **kwargs):
        super(MOOSE, self).__init__()
        self.pretrained=False
        patch_size = 14 if(cfg.MODEL.VISUAL_MODEL == 'dinov2') else 16
        self.fusion_mode = cfg.MODEL.FUSION_MODE #"concat" # can be [concat, ofattention, biattention]
        self.st_masking = cfg.MODEL.ST_MASKING
        self.time_aggregation = cfg.MODEL.TIME_AGGREGATION
        # self.model = MOOSE(raft_args, raft_args, num_classes=cfg.MODEL.NUM_CLASSES)
        # self.crossatt = CustomAttentionWithResidual(embed_size = 768)
        with torch.no_grad():
            self.moose_encoder = MOOSE_Encoder(raft_args, cfg)
        # self.patch_embed = MotionPatchEmbed(img_size=224, patch_size=14, in_chans=2, embed_dim=768) # Flow patches embedding
        num_classes = cfg.MODEL.NUM_CLASSES
        visual_dim = 768 if(cfg.MODEL.VISUAL_MODEL == 'dinov2') else 1024
        motion_dim = 192
        # self.drop_path = DropPath(drop_path) if drop_path > 0. else nn.Identity()
        # self.motion_feature_extractor = VisionTransformer(img_size=28, num_classes=400, patch_size=2, embed_dim=128, depth=3, num_heads=2, in_chans=2, mlp_ratio=1, qkv_bias=True, norm_layer=partial(nn.LayerNorm, eps=1e-6), drop_rate=0., attn_drop_rate=0., drop_path_rate=0.1, num_frames=7, attention_type='space_only')
        self.motion_feature_extractor  = vits.__dict__['vit_tiny'](patch_size=patch_size, num_classes=0, in_chans=2, img_size=(cfg.DATA.TRAIN_CROP_SIZE,cfg.DATA.TRAIN_CROP_SIZE))
        if(self.fusion_mode == "ofattention"):
            fc_dim = visual_dim
        elif(self.fusion_mode == "concat" or self.fusion_mode == "biconcat"):
            fc_dim = visual_dim + motion_dim
        elif(self.fusion_mode == "viattention"):
            fc_dim = motion_dim
        else:
            fc_dim = visual_dim

        self.norm2 = norm_layer(fc_dim)
        mlp_hidden_dim = int(fc_dim * mlp_ratio)
        self.mlp = Mlp(in_features=fc_dim, hidden_features=mlp_hidden_dim, act_layer=act_layer, drop=drop)
        self.head = nn.Linear(fc_dim, num_classes) if num_classes > 0 else nn.Identity()
        # print("self.fusion_mode = onevisualmotion")
        
        self.visual_mask = torch.ones((1, 257)).bool() if(cfg.MODEL.VISUAL_MODEL != 'sapiens') else torch.ones((1, 4097)).bool()
        self.motion_mask = torch.ones((1, 257)).bool() if(cfg.MODEL.VISUAL_MODEL != 'sapiens') else torch.ones((1, 4097)).bool()

        if(self.fusion_mode == "viattention"):
            self.joint_cross_attn = BidirectionalCrossAttention(
                dim = motion_dim,
                heads = 8,
                dim_head = 64,
                context_dim = visual_dim
            )
        else:
            self.joint_cross_attn = BidirectionalCrossAttention(
                dim = visual_dim,
                heads = 8,
                dim_head = 64,
                context_dim = motion_dim
            )

        if(self.time_aggregation == 'mamba'):
            self.mamba = Mamba(
                # This module uses roughly 3 * expand * d_model^2 parameters
                d_model=fc_dim, # Model dimension d_model
                d_state=16,  # SSM state expansion factor
                d_conv=4,    # Local convolution width
                expand=2,    # Block expansion factor
                device=torch.cuda.current_device()
            )
            for p in self.mamba.parameters():
                p.requires_grad = True
        elif(self.time_aggregation == 'causal') :
            self.causal = CausalSelfAttention(fc_dim, 3, 9*224)


    def forward(self, x):
        # assert False, "self.fusion_mode = onevisualmotion"
        with torch.no_grad():
            visual_embeddings = self.moose_encoder.visual_forward(x)[:, :-1, :] # [b, t, p+1, d] Discard the last frame
            b = visual_embeddings.shape[0]
            t = visual_embeddings.shape[1]
            p = visual_embeddings.shape[2]
            visual_embeddings = rearrange(visual_embeddings, 'b t p d -> (b t) p d' ,b=b, t=t) 
            if(self.fusion_mode != 'space_only'):
                flow_low = self.moose_encoder.motion_forward(x) # [b, t, c, w, h]
                # visual_embeddings, flow_low = x[0], x[1]
                # print(visual_embeddings.shape, flow_low.shape)
                # assert False
                # flow_low = rearrange(flow_low, 'b t a c w h -> b (c t a) w h')
                flow_low = rearrange(flow_low, 'b t c w h -> (b t) c w h' ,b=b, t=t) 
                # print(flow_low.shape, '----------------------')
                # video_embeddings = visual_embeddings
                # motion_embeddings = self.motion_feature_extractor.forward_features(flow_low) #[2, 197, 768]
                motion_embeddings = self.motion_feature_extractor.get_intermediate_layers(flow_low, n=3)[-1] #[2, 197, 192] = [c, patchs+1, emb_dim]

        # print(visual_embeddings.shape, motion_embeddings.shape)
        # assert False
        ## Visual-Motion Fussion
        
        if(self.fusion_mode == "ofattention"):
            # video_embeddings = self.crossatt(visual_embeddings, motion_embeddings)
            # video_embeddings = self.mlp(self.norm2(video_embeddings))
            visual_embeddings, motion_embeddings = self.joint_cross_attn(
                                    visual_embeddings,
                                    motion_embeddings,
                                    mask = self.visual_mask.cuda(),
                                    context_mask = self.motion_mask.cuda(),
                                    matrix_mask = self.st_masking
                                )
            video_embeddings = self.mlp(self.norm2(visual_embeddings))
        elif(self.fusion_mode == "viattention"):
            motion_embeddings, visual_embeddings = self.joint_cross_attn(
                                    motion_embeddings,
                                    visual_embeddings,
                                    mask = self.visual_mask.cuda(),
                                    context_mask = self.motion_mask.cuda(),
                                    matrix_mask = self.st_masking
                                )
            video_embeddings = self.mlp(self.norm2(motion_embeddings))
        elif(self.fusion_mode == "biconcat"):
            visual_embeddings, motion_embeddings = self.joint_cross_attn(
                                    visual_embeddings,
                                    motion_embeddings,
                                    mask = self.visual_mask.cuda(),
                                    context_mask = self.motion_mask.cuda(),
                                    matrix_mask = self.st_masking
                                )
            video_embeddings = torch.concat((visual_embeddings, motion_embeddings),dim=2)
            video_embeddings = self.mlp(self.norm2(video_embeddings))
            # print(video_embeddings.shape, visual_embeddings.shape, motion_embeddings.shape)
            # assert False
        elif(self.fusion_mode == "concat"):
            video_embeddings = torch.concat((visual_embeddings, motion_embeddings),dim=2)
            video_embeddings = self.mlp(self.norm2(video_embeddings))
        elif(self.fusion_mode == "space_only"):
            video_embeddings = visual_embeddings
        elif(self.fusion_mode == "ofseq"):
            visual_embeddings = rearrange(visual_embeddings, '(b t) p d -> b t p d' ,b=b, t=t) 
            motion_embeddings = rearrange(motion_embeddings, '(b t) p d -> b t p d' ,b=b, t=t)
            visual_temp = visual_embeddings[:,0,:]
            motion_temp = motion_embeddings[:,0,:]
            for i in range(t): 
                visual_temp, motion_temp = self.joint_cross_attn(
                            visual_temp,
                            motion_temp,
                            mask = self.visual_mask.cuda(),
                            context_mask = self.motion_mask.cuda(),
                            matrix_mask = self.st_masking
                        )
        else:
            assert False, f"No fusion_mode {self.fusion_mode} found!"

        video_embeddings = rearrange(video_embeddings, '(b t) p d -> b t p d' ,b=b, t=t) 
        if(self.time_aggregation == 'mean'):
            video_embeddings = torch.mean(video_embeddings, dim=1)
            x = video_embeddings[:, 0, :] # Get the cls_token
        elif(self.time_aggregation == 'mamba'):
            out = self.mamba(video_embeddings[:, :, 0, :])
            x = out[:, -1, :]
            # x = torch.mean(out, dim=1)
            # print(x.shape)
            # assert False, "get to MAMBA"
        elif(self.time_aggregation == 'causal'):
            video_embeddings = rearrange(video_embeddings, 'b t p d -> (b p) t d' ,b=b, t=t)
            res = self.causal(video_embeddings)
            video_embeddings = video_embeddings + res
            video_embeddings = rearrange(video_embeddings, '(b p) t d -> b t p d' ,b=b, t=t, p=p)
            # print(video_embeddings.shape)
            x = video_embeddings[:, -1, 0, :] # Get the cls_token
            # print(x.shape)
            # assert False, "get to CAUSAL"
        else:
            assert False, f"No time_aggregation {self.time_aggregation} found!"
        # print(video_embeddings.shape)
        # assert False

        x = self.head(x)
        return x
    


def get_sapiens():
    from mmpretrain import get_model
    import os
    import mmengine

    config = "/data2/hongn/sapiens/pretrain/configs/sapiens_mae/humans_300m_test/mae_sapiens_0.3b-p16_8xb512-coslr-1600e_humans_300m_test.py"
    checkpoint = "/data2/hongn/sapiens/pretrain/checkpoints/sapiens_0.3b/sapiens_0.3b_epoch_1600_clean.pth"
    input = "/data2/hongn/sapiens/pose/demo/data/itw_videos"
    output_root = "/data2/hongn/sapiens/outputs/ados2_pretrain"

    if output_root:
        mmengine.mkdir_or_exist(output_root)
        output_file = os.path.join(output_root, os.path.basename(input))

    # inferencer = FeatureExtractor(model=config, pretrained=checkpoint, device='cuda', backbone=dict(out_indices=(0, 1, 2, 3, 4)))
    inferencer = get_model(model=config, pretrained=True)
    return inferencer
