from __future__ import absolute_import, division, print_function
import copy, logging, math
from os.path import join as pjoin
from torch.nn import CrossEntropyLoss, Dropout, Softmax, Linear, Conv2d, LayerNorm
from torch.nn.modules.utils import _pair
from scipy import ndimage

import math

from os.path import join as pjoin
from collections import OrderedDict

import os
import time
import numpy as np
import matplotlib.pyplot as plt
import torch
import torch.nn as nn
import torch.optim as optim
import timm

import torch.nn.functional as F

######################## UNET ########################

class DoubleConv(nn.Module):
  def __init__(self, in_channels, out_channels):
    super().__init__()
    self.conv = nn.Sequential(
        nn.Conv2d(in_channels, out_channels, 3, 1, 1, bias = False),
        nn.BatchNorm2d(out_channels),
        nn.ReLU(True),
        nn.Conv2d(out_channels, out_channels, 3, 1, 1, bias = False),
        nn.BatchNorm2d(out_channels),
        nn.ReLU(True)
    )

  def forward(self, x):
    return self.conv(x)

class UNET(nn.Module):
  def __init__(self, in_channels = 1, out_channels = 4):
    super().__init__()

    # Down

    self.d1 = DoubleConv(in_channels, 64)
    self.d2 = DoubleConv(64, 128)
    self.d3 = DoubleConv(128, 256)
    self.d4 = DoubleConv(256, 512)

    # Bottleneck

    self.d5 = DoubleConv(512, 512 * 2)

    # Up

    self.u1 = nn.ConvTranspose2d(512 * 2, 512, 2, 2)
    self.du1 = DoubleConv(512 * 2, 512)

    self.u2 = nn.ConvTranspose2d(256 * 2, 256, 2, 2)
    self.du2 = DoubleConv(256 * 2, 256)

    self.u3 = nn.ConvTranspose2d(128 * 2, 128, 2, 2)
    self.du3 = DoubleConv(128 * 2, 128)

    self.u4 = nn.ConvTranspose2d(64 * 2, 64, 2, 2)
    self.du4 = DoubleConv(64 * 2, 64)

    # Output Layer

    self.out = nn.Conv2d(64, out_channels, kernel_size = 1)

    # Other Layers

    self.pool = nn.MaxPool2d(2, 2)

  def forward(self, x):

    d1 = self.d1(x) # 256 x 256 x 1 => 256 x 256 x 64
    d1_ = self.pool(d1) # 256 x 256 x 64 => 128 x 128 x 64
    d2 = self.d2(d1_) # 128 x 128 x 64 => 128 x 128 x 128
    d2_ = self.pool(d2) # 128 x 128 x 128 => 64 x 64 x 128
    d3 = self.d3(d2_) # 64 x 64 x 128 => 64 x 64 x 256
    d3_ = self.pool(d3) # 64 x 64 x 256 => 32 x 32 x 256
    d4 = self.d4(d3_) # 32 x 32 x 256 => 32 x 32 x 512
    d4_ = self.pool(d4) # 32 x 32 x 512 => 16 x 16 x 512

    d5 = self.d5(d4_) # 16 x 16 x 512 => 16 x 16 x 1024

    u1 = self.u1(d5) # 16 x 16 x 1024 => 32 x 32 x 512
    t1 = torch.cat((d4, u1), 1) # 32 x 32 x 512 => 32 x 32 x 1024
    t1 = self.du1(t1) # 32 x 32 x 1024 => 32 x 32 x 512

    u2 = self.u2(t1) # 32 x 32 x 512 => 64 x 64 x 256
    t2 = torch.cat((d3, u2), 1) # 64 x 64 x 256 => 64 x 64 x 512
    t2 = self.du2(t2) # 64 x 64 x 512 => 64 x 64 x 256

    u3 = self.u3(t2) # 64 x 64 x 256 => 128 x 128 x 128
    t3 = torch.cat((d2, u3), 1) # 128 x 128 x 128 => 128 x 128 x 256
    t3 = self.du3(t3) # 128 x 128 x 256 => 128 x 128 x 128

    u4 = self.u4(t3) # 128 x 128 x 128 => 256 x 256 x 64
    t4 = torch.cat((d1, u4), 1) # 256 x 256 x 64 => 256 x 256 x 128
    t4 = self.du4(t4) # 256 x 256 x 64 => 256 x 256 x 64

    out = self.out(t4)

    return out
    
######################## UNET Compressed ########################    
    
class DoubleConv(nn.Module):
  def __init__(self, in_channels, out_channels):
    super().__init__()
    self.conv = nn.Sequential(
        nn.Conv2d(in_channels, out_channels, 3, 1, 1, bias = False),
        nn.BatchNorm2d(out_channels),
        nn.ReLU(True),
        nn.Conv2d(out_channels, out_channels, 3, 1, 1, bias = False),
        nn.BatchNorm2d(out_channels),
        nn.ReLU(True)
    )

  def forward(self, x):
    return self.conv(x)

class UNET_Compressed(nn.Module):
  def __init__(self, in_channels = 1, out_channels = 4):
    super().__init__()

    # Down

    self.d1 = DoubleConv(in_channels, 64)
    self.d2 = DoubleConv(64, 128)

    # Up

    self.du3 = DoubleConv(128, 128)

    self.u4 = nn.ConvTranspose2d(64 * 2, 64, 2, 2)
    self.du4 = DoubleConv(64 * 2, 64)

    # Output Layer

    self.out = nn.Conv2d(64, out_channels, kernel_size = 1)

    # Other Layers

    self.pool = nn.MaxPool2d(2, 2)

  def forward(self, x):

    d1 = self.d1(x) # 256 x 256 x 1 => 256 x 256 x 64
    d1_ = self.pool(d1) # 256 x 256 x 64 => 128 x 128 x 64
    d2 = self.d2(d1_) # 128 x 128 x 64 => 128 x 128 x 128

    t3 = self.du3(d2) # 128 x 128 x 128 => 128 x 128 x 128

    u4 = self.u4(t3) # 128 x 128 x 128 => 256 x 256 x 64
    t4 = torch.cat((d1, u4), 1) # 256 x 256 x 64 => 256 x 256 x 128
    t4 = self.du4(t4) # 256 x 256 x 128 => 256 x 256 x 64
    out = self.out(t4)

    return out 
    
######################## TransUNet ########################    

ATTENTION_Q = "MultiHeadDotProductAttention_1/query"
ATTENTION_K = "MultiHeadDotProductAttention_1/key"
ATTENTION_V = "MultiHeadDotProductAttention_1/value"
ATTENTION_OUT = "MultiHeadDotProductAttention_1/out"
FC_0 = "MlpBlock_3/Dense_0"
FC_1 = "MlpBlock_3/Dense_1"
ATTENTION_NORM = "LayerNorm_0"
MLP_NORM = "LayerNorm_2"


def np2th(weights, conv=False):
    """Possibly convert HWIO to OIHW."""
    if conv:
        weights = weights.transpose([3, 2, 0, 1])
    return torch.from_numpy(weights)

class StdConv2d(nn.Conv2d):

    def forward(self, x):
        w = self.weight
        v, m = torch.var_mean(w, dim=[1, 2, 3], keepdim=True, unbiased=False)
        w = (w - m) / torch.sqrt(v + 1e-5)
        return F.conv2d(x, w, self.bias, self.stride, self.padding,
                        self.dilation, self.groups)


def conv3x3(cin, cout, stride=1, groups=1, bias=False):
    return StdConv2d(cin, cout, kernel_size=3, stride=stride,
                     padding=1, bias=bias, groups=groups)


def conv1x1(cin, cout, stride=1, bias=False):
    return StdConv2d(cin, cout, kernel_size=1, stride=stride,
                     padding=0, bias=bias)


class PreActBottleneck(nn.Module):
    """Pre-activation (v2) bottleneck block.
    """

    def __init__(self, cin, cout=None, cmid=None, stride=1):
        super().__init__()
        cout = cout or cin
        cmid = cmid or cout//4

        self.gn1 = nn.GroupNorm(32, cmid, eps=1e-6)
        self.conv1 = conv1x1(cin, cmid, bias=False)
        self.gn2 = nn.GroupNorm(32, cmid, eps=1e-6)
        self.conv2 = conv3x3(cmid, cmid, stride, bias=False)  # Original code has it on conv1!!
        self.gn3 = nn.GroupNorm(32, cout, eps=1e-6)
        self.conv3 = conv1x1(cmid, cout, bias=False)
        self.relu = nn.ReLU(inplace=True)

        if (stride != 1 or cin != cout):
            # Projection also with pre-activation according to paper.
            self.downsample = conv1x1(cin, cout, stride, bias=False)
            self.gn_proj = nn.GroupNorm(cout, cout)

    def forward(self, x):

        # Residual branch
        residual = x
        if hasattr(self, 'downsample'):
            residual = self.downsample(x)
            residual = self.gn_proj(residual)

        # Unit's branch
        y = self.relu(self.gn1(self.conv1(x)))
        y = self.relu(self.gn2(self.conv2(y)))
        y = self.gn3(self.conv3(y))

        y = self.relu(residual + y)
        return y

    def load_from(self, weights, n_block, n_unit):
        conv1_weight = np2th(weights[pjoin(n_block, n_unit, "conv1/kernel")], conv=True)
        conv2_weight = np2th(weights[pjoin(n_block, n_unit, "conv2/kernel")], conv=True)
        conv3_weight = np2th(weights[pjoin(n_block, n_unit, "conv3/kernel")], conv=True)

        gn1_weight = np2th(weights[pjoin(n_block, n_unit, "gn1/scale")])
        gn1_bias = np2th(weights[pjoin(n_block, n_unit, "gn1/bias")])

        gn2_weight = np2th(weights[pjoin(n_block, n_unit, "gn2/scale")])
        gn2_bias = np2th(weights[pjoin(n_block, n_unit, "gn2/bias")])

        gn3_weight = np2th(weights[pjoin(n_block, n_unit, "gn3/scale")])
        gn3_bias = np2th(weights[pjoin(n_block, n_unit, "gn3/bias")])

        self.conv1.weight.copy_(conv1_weight)
        self.conv2.weight.copy_(conv2_weight)
        self.conv3.weight.copy_(conv3_weight)

        self.gn1.weight.copy_(gn1_weight.view(-1))
        self.gn1.bias.copy_(gn1_bias.view(-1))

        self.gn2.weight.copy_(gn2_weight.view(-1))
        self.gn2.bias.copy_(gn2_bias.view(-1))

        self.gn3.weight.copy_(gn3_weight.view(-1))
        self.gn3.bias.copy_(gn3_bias.view(-1))

        if hasattr(self, 'downsample'):
            proj_conv_weight = np2th(weights[pjoin(n_block, n_unit, "conv_proj/kernel")], conv=True)
            proj_gn_weight = np2th(weights[pjoin(n_block, n_unit, "gn_proj/scale")])
            proj_gn_bias = np2th(weights[pjoin(n_block, n_unit, "gn_proj/bias")])

            self.downsample.weight.copy_(proj_conv_weight)
            self.gn_proj.weight.copy_(proj_gn_weight.view(-1))
            self.gn_proj.bias.copy_(proj_gn_bias.view(-1))

class ResNetV2(nn.Module):
    """Implementation of Pre-activation (v2) ResNet mode."""

    def __init__(self, block_units, width_factor):
        super().__init__()
        width = int(64 * width_factor)
        self.width = width

        self.root = nn.Sequential(OrderedDict([
            ('conv', StdConv2d(3, width, kernel_size=7, stride=2, bias=False, padding=3)),
            ('gn', nn.GroupNorm(32, width, eps=1e-6)),
            ('relu', nn.ReLU(inplace=True)),
            # ('pool', nn.MaxPool2d(kernel_size=3, stride=2, padding=0))
        ]))

        self.body = nn.Sequential(OrderedDict([
            ('block1', nn.Sequential(OrderedDict(
                [('unit1', PreActBottleneck(cin=width, cout=width*4, cmid=width))] +
                [(f'unit{i:d}', PreActBottleneck(cin=width*4, cout=width*4, cmid=width)) for i in range(2, block_units[0] + 1)],
                ))),
            ('block2', nn.Sequential(OrderedDict(
                [('unit1', PreActBottleneck(cin=width*4, cout=width*8, cmid=width*2, stride=2))] +
                [(f'unit{i:d}', PreActBottleneck(cin=width*8, cout=width*8, cmid=width*2)) for i in range(2, block_units[1] + 1)],
                ))),
            ('block3', nn.Sequential(OrderedDict(
                [('unit1', PreActBottleneck(cin=width*8, cout=width*16, cmid=width*4, stride=2))] +
                [(f'unit{i:d}', PreActBottleneck(cin=width*16, cout=width*16, cmid=width*4)) for i in range(2, block_units[2] + 1)],
                ))),
        ]))

    def forward(self, x):
        features = []
        b, c, in_size, _ = x.size()
        x = self.root(x)
        features.append(x)
        x = nn.MaxPool2d(kernel_size=3, stride=2, padding=0)(x)
        for i in range(len(self.body)-1):
            x = self.body[i](x)
            right_size = int(in_size / 4 / (i+1))
            if x.size()[2] != right_size:
                pad = right_size - x.size()[2]
                assert pad < 3 and pad > 0, "x {} should {}".format(x.size(), right_size)
                feat = torch.zeros((b, x.size()[1], right_size, right_size), device=x.device)
                feat[:, :, 0:x.size()[2], 0:x.size()[3]] = x[:]
            else:
                feat = x
            features.append(feat)
        x = self.body[-1](x)
        return x, features[::-1]
    
logger = logging.getLogger(__name__)

def swish(x):
  return x * torch.sigmoid(x)

ACT2FN = {"gelu": torch.nn.functional.gelu, "relu": torch.nn.functional.relu, "swish": swish}

class Attention(nn.Module):
  def __init__(self, vis):
    super(Attention, self).__init__()
    self.vis = vis
    self.num_attention_heads = 12
    self.attention_head_size = int(768 / self.num_attention_heads)
    self.all_head_size = self.num_attention_heads * self.attention_head_size

    self.query = Linear(768, self.all_head_size)
    self.key = Linear(768, self.all_head_size)
    self.value = Linear(768, self.all_head_size)

    self.out = Linear(768, 768)
    self.attn_dropout = Dropout(0.0)
    self.proj_dropout = Dropout(0.0)

    self.softmax = Softmax(dim = -1)

  def transpose_for_scores(self, x):
    # perhaps -> batch_size x  patches x embeddings => batch_size x heads x patches x new_embedding_size
    new_x_shape = x.size()[:-1] + (self.num_attention_heads, self.attention_head_size)
    x = x.view(*new_x_shape)
    return x.permute(0, 2, 1, 3)

  def forward(self, hidden_states):
    mixed_query_layer = self.query(hidden_states)
    mixed_key_layer = self.key(hidden_states)
    mixed_value_layer = self.value(hidden_states)

    query_layer = self.transpose_for_scores(mixed_query_layer)
    key_layer = self.transpose_for_scores(mixed_key_layer)
    value_layer = self.transpose_for_scores(mixed_value_layer)

    attention_scores = torch.matmul(query_layer, key_layer.transpose(-1, -2))
    attention_scores = attention_scores / math.sqrt(self.attention_head_size)
    attention_probs = self.softmax(attention_scores)
    weights = attention_probs if self.vis else None
    attention_probs = self.attn_dropout(attention_probs)

    context_layer = torch.matmul(attention_probs, value_layer)

    # perhaps -> batch_size x heads x patches x new_embedding_size => batch_size x  patches x heads x new_embedding_size
    context_layer = context_layer.permute(0, 2, 1, 3).contiguous()

    new_context_layer_shape = context_layer.size()[:-2] + (self.all_head_size,)
    context_layer = context_layer.view(*new_context_layer_shape)
    attention_output = self.out(context_layer)
    attention_output = self.proj_dropout(attention_output)

    if self.vis == True:
      return attention_output, weights
    else:
      return attention_output

class Mlp(nn.Module):
  def __init__(self):
    super(Mlp, self).__init__()
    self.fc1 = Linear(768, 3072)
    self.fc2 = Linear(3072, 768)
    self.act_fc = ACT2FN["gelu"]
    self.dropout = Dropout(0.1)

    self._init_weights()

  def _init_weights(self):
    nn.init.xavier_uniform_(self.fc1.weight)
    nn.init.xavier_uniform_(self.fc2.weight)
    nn.init.normal_(self.fc1.bias, std = 1e-6)
    nn.init.normal_(self.fc2.bias, std = 1e-6)

  def forward(self, x):
    x = self.fc1(x)
    x = self.act_fc(x)
    x = self.dropout(x)
    x = self.fc2(x)
    x = self.dropout(x)
    return x

class Embeddings(nn.Module):
  def __init__(self, img_size, in_channels = 3):
    super(Embeddings, self).__init__()
    self.hybrid = None
    img_size = _pair(img_size)

    # 1 x 1    
    patch_size = (img_size[0] // 16 // (img_size[0] // 16), img_size[1] // 16 // (img_size[1] // 16))
    patch_size_real = (patch_size[0] * 16, patch_size[1] * 16)
    n_patches = (img_size[0] // patch_size_real[0]) * (img_size[1] // patch_size_real[1])
    self.hybrid = True

    self.hybrid_model = ResNetV2(block_units = (3, 4, 9), width_factor = 1)
    in_channels = self.hybrid_model.width * 16


    self.patch_embeddings = Conv2d(in_channels = in_channels, out_channels = 768, kernel_size = patch_size, stride = patch_size)
    self.position_embeddings = nn.Parameter(torch.zeros(1, n_patches, 768))

    self.dropout = Dropout(0.1)

  def forward(self, x):
    x, features = self.hybrid_model(x)
    x = self.patch_embeddings(x)
    x = x.flatten(2)
    x = x.transpose(-1, -2)
    embeddings = x + self.position_embeddings
    embeddings = self.dropout(embeddings)

    return embeddings, features

class Block(nn.Module):
  def __init__(self, vis):
    super(Block, self).__init__()
    self.hidden_size = 768
    self.attention_norm = LayerNorm(768, eps = 1e-6)
    self.ffn_norm = LayerNorm(768, eps = 1e-6)
    self.ffn = Mlp()
    self.attn = Attention(vis = vis)
    self.vis = vis

  def forward(self, x):
    h = x
    x = self.attention_norm(x)
    if self.vis == True:
      x, weights = self.attn(x)
    else:
      x = self.attn(x)
    x = x + h
    h = x
    x = self.ffn_norm(x)
    x = self.ffn(x)
    x = x + h
    if self.vis == True:
      return x, weights
    else:
      return x

  def load_from(self, weights, n_block):
        ROOT = f"Transformer/encoderblock_{n_block}"
        with torch.no_grad():
            query_weight = np2th(weights[pjoin(ROOT, ATTENTION_Q, "kernel")]).view(self.hidden_size, self.hidden_size).t()
            key_weight = np2th(weights[pjoin(ROOT, ATTENTION_K, "kernel")]).view(self.hidden_size, self.hidden_size).t()
            value_weight = np2th(weights[pjoin(ROOT, ATTENTION_V, "kernel")]).view(self.hidden_size, self.hidden_size).t()
            out_weight = np2th(weights[pjoin(ROOT, ATTENTION_OUT, "kernel")]).view(self.hidden_size, self.hidden_size).t()

            query_bias = np2th(weights[pjoin(ROOT, ATTENTION_Q, "bias")]).view(-1)
            key_bias = np2th(weights[pjoin(ROOT, ATTENTION_K, "bias")]).view(-1)
            value_bias = np2th(weights[pjoin(ROOT, ATTENTION_V, "bias")]).view(-1)
            out_bias = np2th(weights[pjoin(ROOT, ATTENTION_OUT, "bias")]).view(-1)

            self.attn.query.weight.copy_(query_weight)
            self.attn.key.weight.copy_(key_weight)
            self.attn.value.weight.copy_(value_weight)
            self.attn.out.weight.copy_(out_weight)
            self.attn.query.bias.copy_(query_bias)
            self.attn.key.bias.copy_(key_bias)
            self.attn.value.bias.copy_(value_bias)
            self.attn.out.bias.copy_(out_bias)

            mlp_weight_0 = np2th(weights[pjoin(ROOT, FC_0, "kernel")]).t()
            mlp_weight_1 = np2th(weights[pjoin(ROOT, FC_1, "kernel")]).t()
            mlp_bias_0 = np2th(weights[pjoin(ROOT, FC_0, "bias")]).t()
            mlp_bias_1 = np2th(weights[pjoin(ROOT, FC_1, "bias")]).t()

            self.ffn.fc1.weight.copy_(mlp_weight_0)
            self.ffn.fc2.weight.copy_(mlp_weight_1)
            self.ffn.fc1.bias.copy_(mlp_bias_0)
            self.ffn.fc2.bias.copy_(mlp_bias_1)

            self.attention_norm.weight.copy_(np2th(weights[pjoin(ROOT, ATTENTION_NORM, "scale")]))
            self.attention_norm.bias.copy_(np2th(weights[pjoin(ROOT, ATTENTION_NORM, "bias")]))
            self.ffn_norm.weight.copy_(np2th(weights[pjoin(ROOT, MLP_NORM, "scale")]))
            self.ffn_norm.bias.copy_(np2th(weights[pjoin(ROOT, MLP_NORM, "bias")]))
        
class Encoder(nn.Module):
  def __init__(self, vis, num_layers = 12):
    super(Encoder, self).__init__()
    self.vis = vis
    self.layer = nn.ModuleList()
    self.encoder_norm = LayerNorm(768, eps = 1e-6)
    for _ in range(num_layers):
      layer = Block(vis)
      self.layer.append(copy.deepcopy(layer))
    self.vis = vis

  def forward(self, hidden_states):
    attn_weights = []
    count = 1
    for layer_block in self.layer:
      count += 1
      if self.vis == True:
        hidden_states, weights = layer_block(hidden_states)
        attn_weights.append(weights)
      else:
        hidden_states = layer_block(hidden_states)
    encoded = self.encoder_norm(hidden_states)
    if self.vis == True:
      return encoded, attn_weights
    else:
      return encoded

class Transformer(nn.Module):
  def __init__(self, img_size, vis, num_layers = 12):
    super(Transformer, self).__init__()
    self.embeddings = Embeddings(img_size = img_size)
    self.encoder = Encoder(vis, num_layers)
    self.vis = vis

  def forward(self, input_ids):
    embedding_output, features = self.embeddings(input_ids)
    
    if self.vis == True:
      encoded, attn_weights = self.encoder(embedding_output)
      return encoded, attn_weights, features
    else:
      encoded = self.encoder(embedding_output)
      return encoded, features

class Conv2dReLU(nn.Sequential):
  def __init__(self, in_channels, out_channels, kernel_size, padding = 0, stride = 1, use_batchnorm = True):
    conv = nn.Conv2d(in_channels, out_channels, kernel_size, stride = stride, padding = padding, bias = not (use_batchnorm))
    relu = nn.ReLU(inplace = True)
    bn = nn.BatchNorm2d(out_channels)
    super(Conv2dReLU, self).__init__(conv, bn, relu) 
    
class DecoderBlock(nn.Module):
  def __init__(self, in_channels, out_channels, skip_channels = 0, use_batchnorm = True):
    super().__init__()
    self.conv1 = Conv2dReLU(
        in_channels + skip_channels, out_channels, kernel_size = 3, padding = 1, use_batchnorm = use_batchnorm
    )
    self.conv2 = Conv2dReLU(out_channels, out_channels, kernel_size = 3, padding = 1, use_batchnorm = use_batchnorm)
    self.up = nn.UpsamplingBilinear2d(scale_factor = 2)

  def forward(self, x, skip = None):
    x = self.up(x)
    if skip is not None:
      x = torch.cat([x, skip], dim = 1)
    x = self.conv1(x)
    x = self.conv2(x)
    return x

class SegmentationHead(nn.Sequential):
  def __init__(self, in_channels, out_channels, kernel_size = 3, upsampling = 1):
    conv2d = nn.Conv2d(in_channels, out_channels, kernel_size = kernel_size, padding = kernel_size // 2)
    upsampling = nn.UpsamplingBilinear2d(scale_factor = upsampling) if upsampling > 1 else nn.Identity()
    super().__init__(conv2d, upsampling)
    
class DecoderCup(nn.Module):
  def __init__(self):
    super().__init__()
    head_channels = 512
    self.conv_more = Conv2dReLU(768, head_channels, kernel_size = 3, padding = 1, use_batchnorm = True)
    decoder_channels = (256, 128, 64, 16)
    in_channels = [head_channels] + list(decoder_channels[:-1])
    out_channels = decoder_channels

    self.n_skip = 3

    if self.n_skip != 0:
      skip_channels = [512, 256, 64, 16]
      for i in range(4 - self.n_skip):
        skip_channels[3 - i] = 0
    else:
      skip_channels = [0, 0, 0, 0]

    blocks = [
              DecoderBlock(in_ch, out_ch, sk_ch) for in_ch, out_ch, sk_ch in zip(in_channels, out_channels, skip_channels)
    ]

    self.blocks = nn.ModuleList(blocks)

  def forward(self, hidden_states, features = None):
    B, n_patch, hidden = hidden_states.size()
    h, w = int(np.sqrt(n_patch)), int(np.sqrt(n_patch))
    x = hidden_states.permute(0, 2, 1)
    x = x.contiguous().view(B, hidden, h, w)
    x = self.conv_more(x)
    for i, decoder_block in enumerate(self.blocks):
      if features is not None:
        skip = features[i] if (i < self.n_skip) else None    
      else:
        skip = None
      x = decoder_block(x, skip = skip)
    return x

class VisionTransformer(nn.Module):
  def __init__(self, img_size = 224, num_classes = 21843, zero_head = False, vis = False, num_layers = 12):
    super(VisionTransformer, self).__init__()
    self.num_classes = num_classes
    self.zero_head = zero_head
    self.classifier = "seg"
    self.transformer = Transformer(img_size, vis, num_layers)
    self.decoder = DecoderCup()
    self.segmentation_head = SegmentationHead(
        in_channels = (256, 128, 64, 16)[-1], out_channels = num_classes, kernel_size = 3
    )
    self.vis = vis

  def forward(self, x):
    if x.size()[1] == 1:
      x = x.repeat(1, 3, 1, 1)
    if self.vis == True:
      x, attn_weights, features = self.transformer(x)
    else:
      x, features = self.transformer(x)
    x = self.decoder(x, features)
    logits = self.segmentation_head(x)
    if self.vis == True:
      return logits, attn_weights, features
    else:
      return logits

  def load_from(self, weights):
    with torch.no_grad():
      res_weight = weights
      self.transformer.embeddings.patch_embeddings.weight.copy_(np2th(weights["embedding/kernel"], conv=True))
      self.transformer.embeddings.patch_embeddings.bias.copy_(np2th(weights["embedding/bias"]))

      self.transformer.encoder.encoder_norm.weight.copy_(np2th(weights["Transformer/encoder_norm/scale"]))
      self.transformer.encoder.encoder_norm.bias.copy_(np2th(weights["Transformer/encoder_norm/bias"]))

      posemb = np2th(weights["Transformer/posembed_input/pos_embedding"])

      posemb_new = self.transformer.embeddings.position_embeddings
      if posemb.size() == posemb_new.size():
          self.transformer.embeddings.position_embeddings.copy_(posemb)
      elif posemb.size()[1]-1 == posemb_new.size()[1]:
          posemb = posemb[:, 1:]
          self.transformer.embeddings.position_embeddings.copy_(posemb)
      else:
          logger.info("load_pretrained: resized variant: %s to %s" % (posemb.size(), posemb_new.size()))
          ntok_new = posemb_new.size(1)
          if self.classifier == "seg":
              _, posemb_grid = posemb[:, :1], posemb[0, 1:]
          gs_old = int(np.sqrt(len(posemb_grid)))
          gs_new = int(np.sqrt(ntok_new))
          print('load_pretrained: grid-size from %s to %s' % (gs_old, gs_new))
          posemb_grid = posemb_grid.reshape(gs_old, gs_old, -1)
          zoom = (gs_new / gs_old, gs_new / gs_old, 1)
          posemb_grid = ndimage.zoom(posemb_grid, zoom, order=1)  # th2np
          posemb_grid = posemb_grid.reshape(1, gs_new * gs_new, -1)
          posemb = posemb_grid
          self.transformer.embeddings.position_embeddings.copy_(np2th(posemb))

      # Encoder whole
      for bname, block in self.transformer.encoder.named_children():
          for uname, unit in block.named_children():
              unit.load_from(weights, n_block=uname)

      if self.transformer.embeddings.hybrid:
          self.transformer.embeddings.hybrid_model.root.conv.weight.copy_(np2th(res_weight["conv_root/kernel"], conv=True))
          gn_weight = np2th(res_weight["gn_root/scale"]).view(-1)
          gn_bias = np2th(res_weight["gn_root/bias"]).view(-1)
          self.transformer.embeddings.hybrid_model.root.gn.weight.copy_(gn_weight)
          self.transformer.embeddings.hybrid_model.root.gn.bias.copy_(gn_bias)

          for bname, block in self.transformer.embeddings.hybrid_model.body.named_children():
              for uname, unit in block.named_children():
                  unit.load_from(res_weight, n_block=bname, n_unit=uname)
                  
def TransUNet(num_classes = 4, load_pretrained = True, num_layers = 12, vis = True):
    if load_pretrained == True:
        try:
            if not os.path.isdir("imagenet21k"): 
                os.mkdir("imagenet21k")
            files = os.listdir("imagenet21k")
            if "R50+ViT-B_16.npz" not in files:
              os.system("wget https://storage.googleapis.com/vit_models/imagenet21k/R50+ViT-B_16.npz >/dev/null 2>&1")
              os.system("mv R50+ViT-B_16.npz imagenet21k/R50+ViT-B_16.npz >/dev/null 2>&1")
        except Exception as e:
            print(e)
    net = VisionTransformer(num_classes = num_classes, num_layers = num_layers, vis = vis)
    if load_pretrained == True:
        net.load_from(weights=np.load("imagenet21k/R50+ViT-B_16.npz"))
    return net                        

######################## ViT ########################

ATTENTION_Q = "MultiHeadDotProductAttention_1/query"
ATTENTION_K = "MultiHeadDotProductAttention_1/key"
ATTENTION_V = "MultiHeadDotProductAttention_1/value"
ATTENTION_OUT = "MultiHeadDotProductAttention_1/out"
FC_0 = "MlpBlock_3/Dense_0"
FC_1 = "MlpBlock_3/Dense_1"
ATTENTION_NORM = "LayerNorm_0"
MLP_NORM = "LayerNorm_2"


def np2th(weights, conv=False):
    """Possibly convert HWIO to OIHW."""
    if conv:
        weights = weights.transpose([3, 2, 0, 1])
    return torch.from_numpy(weights)
    
logger = logging.getLogger(__name__)

def swish(x):
  return x * torch.sigmoid(x)

ACT2FN = {"gelu": torch.nn.functional.gelu, "relu": torch.nn.functional.relu, "swish": swish}

class Attention(nn.Module):
  def __init__(self, vis):
    super(Attention, self).__init__()
    self.vis = vis
    self.num_attention_heads = 12
    self.attention_head_size = int(768 / self.num_attention_heads)
    self.all_head_size = self.num_attention_heads * self.attention_head_size

    self.query = Linear(768, self.all_head_size)
    self.key = Linear(768, self.all_head_size)
    self.value = Linear(768, self.all_head_size)

    self.out = Linear(768, 768)
    self.attn_dropout = Dropout(0.0)
    self.proj_dropout = Dropout(0.0)

    self.softmax = Softmax(dim = -1)

  def transpose_for_scores(self, x):
    # perhaps -> batch_size x  patches x embeddings => batch_size x heads x patches x new_embedding_size
    new_x_shape = x.size()[:-1] + (self.num_attention_heads, self.attention_head_size)
    x = x.view(*new_x_shape)
    return x.permute(0, 2, 1, 3)

  def forward(self, hidden_states):
    mixed_query_layer = self.query(hidden_states)
    mixed_key_layer = self.key(hidden_states)
    mixed_value_layer = self.value(hidden_states)

    query_layer = self.transpose_for_scores(mixed_query_layer)
    key_layer = self.transpose_for_scores(mixed_key_layer)
    value_layer = self.transpose_for_scores(mixed_value_layer)

    attention_scores = torch.matmul(query_layer, key_layer.transpose(-1, -2))
    attention_scores = attention_scores / math.sqrt(self.attention_head_size)
    attention_probs = self.softmax(attention_scores)
    weights = attention_probs if self.vis else None
    attention_probs = self.attn_dropout(attention_probs)

    context_layer = torch.matmul(attention_probs, value_layer)

    # perhaps -> batch_size x heads x patches x new_embedding_size => batch_size x  patches x heads x new_embedding_size
    context_layer = context_layer.permute(0, 2, 1, 3).contiguous()

    new_context_layer_shape = context_layer.size()[:-2] + (self.all_head_size,)
    context_layer = context_layer.view(*new_context_layer_shape)
    attention_output = self.out(context_layer)
    attention_output = self.proj_dropout(attention_output)

    if self.vis == True:
      return attention_output, weights
    else:
      return attention_output

class Mlp_ViT(nn.Module):
  def __init__(self):
    super(Mlp_ViT, self).__init__()
    self.fc1 = Linear(768, 3072)
    self.fc2 = Linear(3072, 768)
    self.act_fc = ACT2FN["gelu"]
    self.dropout = Dropout(0.1)

    self._init_weights()

  def _init_weights(self):
    nn.init.xavier_uniform_(self.fc1.weight)
    nn.init.xavier_uniform_(self.fc2.weight)
    nn.init.normal_(self.fc1.bias, std = 1e-6)
    nn.init.normal_(self.fc2.bias, std = 1e-6)

  def forward(self, x):
    x = self.fc1(x)
    x = self.act_fc(x)
    x = self.dropout(x)
    x = self.fc2(x)
    x = self.dropout(x)
    return x

class Embeddings_ViT(nn.Module):
  def __init__(self, img_size, in_channels = 3):
    super(Embeddings_ViT, self).__init__()
    self.hybrid = None
    img_size = _pair(img_size)

    # 1 x 1    
    patch_size = (16, 16)
    n_patches = (img_size[0] // patch_size[0]) * (img_size[1] // patch_size[1])
    self.hybrid = False

    self.patch_embeddings = Conv2d(in_channels = in_channels, out_channels = 768, kernel_size = patch_size, stride = patch_size)
    self.position_embeddings = nn.Parameter(torch.zeros(1, n_patches, 768))

    self.dropout = Dropout(0.1)

  def forward(self, x):
    features = None
    x = self.patch_embeddings(x)
    x = x.flatten(2)
    x = x.transpose(-1, -2)
    embeddings = x + self.position_embeddings
    embeddings = self.dropout(embeddings)

    return embeddings, features

class Block_ViT(nn.Module):
  def __init__(self, vis):
    super(Block_ViT, self).__init__()
    self.hidden_size = 768
    self.attention_norm = LayerNorm(768, eps = 1e-6)
    self.ffn_norm = LayerNorm(768, eps = 1e-6)
    self.ffn = Mlp_ViT()
    self.vis = vis
    self.attn = Attention(vis = self.vis)

  def forward(self, x):
    h = x
    x = self.attention_norm(x)
    
    if self.vis == True:
      x, weights = self.attn(x)
    else:
      x = self.attn(x)
    x = x + h
    h = x
    x = self.ffn_norm(x)
    x = self.ffn(x)
    x = x + h
    if self.vis == True:
      return x, weights
    else:
      return x

  def load_from(self, weights, n_block):
        ROOT = f"Transformer/encoderblock_{n_block}"
        with torch.no_grad():
            query_weight = np2th(weights[pjoin(ROOT, ATTENTION_Q, "kernel")]).view(self.hidden_size, self.hidden_size).t()
            key_weight = np2th(weights[pjoin(ROOT, ATTENTION_K, "kernel")]).view(self.hidden_size, self.hidden_size).t()
            value_weight = np2th(weights[pjoin(ROOT, ATTENTION_V, "kernel")]).view(self.hidden_size, self.hidden_size).t()
            out_weight = np2th(weights[pjoin(ROOT, ATTENTION_OUT, "kernel")]).view(self.hidden_size, self.hidden_size).t()

            query_bias = np2th(weights[pjoin(ROOT, ATTENTION_Q, "bias")]).view(-1)
            key_bias = np2th(weights[pjoin(ROOT, ATTENTION_K, "bias")]).view(-1)
            value_bias = np2th(weights[pjoin(ROOT, ATTENTION_V, "bias")]).view(-1)
            out_bias = np2th(weights[pjoin(ROOT, ATTENTION_OUT, "bias")]).view(-1)

            self.attn.query.weight.copy_(query_weight)
            self.attn.key.weight.copy_(key_weight)
            self.attn.value.weight.copy_(value_weight)
            self.attn.out.weight.copy_(out_weight)
            self.attn.query.bias.copy_(query_bias)
            self.attn.key.bias.copy_(key_bias)
            self.attn.value.bias.copy_(value_bias)
            self.attn.out.bias.copy_(out_bias)

            mlp_weight_0 = np2th(weights[pjoin(ROOT, FC_0, "kernel")]).t()
            mlp_weight_1 = np2th(weights[pjoin(ROOT, FC_1, "kernel")]).t()
            mlp_bias_0 = np2th(weights[pjoin(ROOT, FC_0, "bias")]).t()
            mlp_bias_1 = np2th(weights[pjoin(ROOT, FC_1, "bias")]).t()

            self.ffn.fc1.weight.copy_(mlp_weight_0)
            self.ffn.fc2.weight.copy_(mlp_weight_1)
            self.ffn.fc1.bias.copy_(mlp_bias_0)
            self.ffn.fc2.bias.copy_(mlp_bias_1)

            self.attention_norm.weight.copy_(np2th(weights[pjoin(ROOT, ATTENTION_NORM, "scale")]))
            self.attention_norm.bias.copy_(np2th(weights[pjoin(ROOT, ATTENTION_NORM, "bias")]))
            self.ffn_norm.weight.copy_(np2th(weights[pjoin(ROOT, MLP_NORM, "scale")]))
            self.ffn_norm.bias.copy_(np2th(weights[pjoin(ROOT, MLP_NORM, "bias")]))
        
class Encoder_ViT(nn.Module):
  def __init__(self, vis, num_layers = 12):
    super(Encoder_ViT, self).__init__()
    self.vis = vis
    self.layer = nn.ModuleList()
    self.encoder_norm = LayerNorm(768, eps = 1e-6)
    for _ in range(num_layers):
      layer = Block_ViT(self.vis)
      self.layer.append(copy.deepcopy(layer))

  def forward(self, hidden_states):
    attn_weights = []
    
    features = []
    for layer_block in self.layer:
      if self.vis == True:
        hidden_states, weights = layer_block(hidden_states)
        attn_weights.append(weights)
      else:
        hidden_states = layer_block(hidden_states)
      features.append(hidden_states)        

    features = features[:-1]

    encoded = self.encoder_norm(hidden_states)
    
    features.append(encoded)
    
    if self.vis == True:
      return features, attn_weights
    else:
      return features

class Transformer_ViT(nn.Module):
  def __init__(self, img_size, vis, num_layers = 12):
    super(Transformer_ViT, self).__init__()
    self.embeddings = Embeddings_ViT(img_size = img_size)
    self.encoder = Encoder_ViT(vis, num_layers)
    self.vis = vis

  def forward(self, input_ids):
    embedding_output, features = self.embeddings(input_ids)
    if self.vis == True:
      encoded, attn_weights = self.encoder(embedding_output)
      return encoded, attn_weights, features
    else:
      encoded = self.encoder(embedding_output)
      return encoded

def load_from(model, weights):
  with torch.no_grad():
    res_weight = weights
    model.embeddings.patch_embeddings.weight.copy_(np2th(weights["embedding/kernel"], conv=True))
    model.embeddings.patch_embeddings.bias.copy_(np2th(weights["embedding/bias"]))

    model.encoder.encoder_norm.weight.copy_(np2th(weights["Transformer/encoder_norm/scale"]))
    model.encoder.encoder_norm.bias.copy_(np2th(weights["Transformer/encoder_norm/bias"]))

    posemb = np2th(weights["Transformer/posembed_input/pos_embedding"])

    posemb_new = model.embeddings.position_embeddings
    if posemb.size() == posemb_new.size():
        model.embeddings.position_embeddings.copy_(posemb)
    elif posemb.size()[1]-1 == posemb_new.size()[1]:
        posemb = posemb[:, 1:]
        model.embeddings.position_embeddings.copy_(posemb)
    else:
        logger.info("load_pretrained: resized variant: %s to %s" % (posemb.size(), posemb_new.size()))
        ntok_new = posemb_new.size(1)
        if model.classifier == "seg":
            _, posemb_grid = posemb[:, :1], posemb[0, 1:]
        gs_old = int(np.sqrt(len(posemb_grid)))
        gs_new = int(np.sqrt(ntok_new))
        print('load_pretrained: grid-size from %s to %s' % (gs_old, gs_new))
        posemb_grid = posemb_grid.reshape(gs_old, gs_old, -1)
        zoom = (gs_new / gs_old, gs_new / gs_old, 1)
        posemb_grid = ndimage.zoom(posemb_grid, zoom, order=1)  # th2np
        posemb_grid = posemb_grid.reshape(1, gs_new * gs_new, -1)
        posemb = posemb_grid
        model.embeddings.position_embeddings.copy_(np2th(posemb))

    # Encoder whole
    for bname, block in model.encoder.named_children():
        for uname, unit in block.named_children():
            unit.load_from(weights, n_block=uname)
                  
def ViT(load_pretrained = True, num_layers = 12, vis = True):
    if load_pretrained == True:
        try:
            try:
                os.mkdir("imagenet21k")
            except Exception as e:
                print(e)
            files = os.listdir("imagenet21k")
            if "ViT-B_16.npz" not in files:            
              os.system("wget https://storage.googleapis.com/vit_models/imagenet21k/ViT-B_16.npz >/dev/null 2>&1")
              os.system("mv ViT-B_16.npz imagenet21k/ViT-B_16.npz >/dev/null 2>&1")
        except Exception as e:
            print(e)
    net = Transformer_ViT(224, vis = vis, num_layers = num_layers)
    if load_pretrained == True:
        load_from(net, weights=np.load("imagenet21k/ViT-B_16.npz"))
    return net                   
    
######################## UNETR ########################    
    
class Deconv(nn.Module):
  def __init__(self, in_channels, out_channels):
    super().__init__()
    self.conv = nn.Sequential(
        nn.ConvTranspose2d(in_channels, out_channels, 2, 2),
        nn.Conv2d(out_channels, out_channels, 3, 1, 1, bias = False),
        nn.BatchNorm2d(out_channels),
        nn.ReLU(True)
    )

  def forward(self, x):
    return self.conv(x)


class DoubleConv(nn.Module):
  def __init__(self, in_channels, out_channels):
    super().__init__()
    self.conv = nn.Sequential(
        nn.Conv2d(in_channels, out_channels, 3, 1, 1, bias = False),
        nn.BatchNorm2d(out_channels),
        nn.ReLU(True),
        nn.Conv2d(out_channels, out_channels, 3, 1, 1, bias = False),
        nn.BatchNorm2d(out_channels),
        nn.ReLU(True)
    )

  def forward(self, x):
    return self.conv(x)

class UNETR(nn.Module):
  def __init__(self, num_classes = 4, all_trans = False, pretrained = True, layers_to_block = [], vis = True):
    super().__init__()

    self.layers_to_block = layers_to_block

    self.trans = ViT(load_pretrained = pretrained, vis = vis)

    # image

    self.all_trans = all_trans
    
    if self.all_trans == True:    
        self.u01 = Deconv(768, 512)
        self.u02 = Deconv(512, 256)
        self.u03 = Deconv(256, 128)
        self.u04 = Deconv(128, 64)
    else:
        self.img_conv = DoubleConv(3, 64)
        
    # z3

    # n x 384 x 14 x 14
    self.u11 = Deconv(768, 512)
    self.u12 = Deconv(512, 256)
    self.u13 = Deconv(256, 128)

    # z6

    # n x 384 x 14 x 14
    self.u21 = Deconv(768, 512)
    self.u22 = Deconv(512, 256)

    # z9

    # n x 384 x 14 x 14
    self.u31 = Deconv(768, 512)

    # z12

    # n x 384 x 14 x 14
    self.u41 = nn.ConvTranspose2d(768, 512, 2, 2)

    # dc1

    self.dc1 = DoubleConv(1024, 512)

    # du1

    self.du1 = nn.ConvTranspose2d(512, 256, 2, 2)

    # dc2

    self.dc2 = DoubleConv(512, 256)

    # du2

    self.du2 = nn.ConvTranspose2d(256, 128, 2, 2)

    # dc3

    self.dc3 = DoubleConv(256, 128)

    # du3

    self.du3 = nn.ConvTranspose2d(128, 64, 2, 2)

    # dc4

    self.dc4 = DoubleConv(128, 64)

    # dc5

    self.segmentation_head = nn.Conv2d(64, num_classes, 3, 1, 1)
    
    self.vis = vis

  def forward(self, x):

    if self.vis == True:
      features, attn_weights, _ = self.trans(x)
    else:
      features = self.trans(x)
    n, _, c = features[0].shape
    features = [i.permute(0, 2, 1).reshape(n, c, 14, 14) for i in features]

    if len(self.layers_to_block) != 0:
      features = [features[i] * 0 if i in self.layers_to_block else features[i] for i in range(len(features))]
    
    if self.all_trans == True:
        img = self.u01(features[0])
        img = self.u02(img)
        img = self.u03(img)
        img = self.u04(img)
    else:
        img = self.img_conv(x)
    
    x1 = self.u11(features[2])
    x1 = self.u12(x1)
    x1 = self.u13(x1)

    
    x2 = self.u21(features[5])
    x2 = self.u22(x2)

    x3 = self.u31(features[8])

    x4 = self.u41(features[11])

    x5 = torch.cat((x3, x4), 1)
    
    x6 = self.dc1(x5)
    x7 = self.du1(x6)
    
    x8 = torch.cat((x2, x7), 1)

    x9 = self.dc2(x8)
    
    x10 = self.du2(x9)
    
    x11 = torch.cat((x1, x10), 1)
    
    x12 = self.dc3(x11)
    x13 = self.du3(x12)

    x14 = torch.cat((img, x13), 1)

    x15 = self.dc4(x14)
    x16 = self.segmentation_head(x15)

    if self.vis == True:
      return x16, attn_weights    
    else:
      return x16
    
######################## UNETR Compressed ########################    
    
class Deconv(nn.Module):
  def __init__(self, in_channels, out_channels):
    super().__init__()
    self.conv = nn.Sequential(
        nn.ConvTranspose2d(in_channels, out_channels, 2, 2),
        nn.Conv2d(out_channels, out_channels, 3, 1, 1, bias = False),
        nn.BatchNorm2d(out_channels),
        nn.ReLU(True)
    )

  def forward(self, x):
    return self.conv(x)

class DoubleConv(nn.Module):
  def __init__(self, in_channels, out_channels):
    super().__init__()
    self.conv = nn.Sequential(
        nn.Conv2d(in_channels, out_channels, 3, 1, 1, bias = False),
        nn.BatchNorm2d(out_channels),
        nn.ReLU(True),
        nn.Conv2d(out_channels, out_channels, 3, 1, 1, bias = False),
        nn.BatchNorm2d(out_channels),
        nn.ReLU(True)
    )

  def forward(self, x):
    return self.conv(x)

class UNETR_Compressed(nn.Module):
  def __init__(self, num_classes = 4, all_trans = False, pretrained = True, layers_to_block = [], vis = True):
    super().__init__()

    self.layers_to_block = layers_to_block

    self.trans = ViT(load_pretrained = pretrained, num_layers = 3, vis = vis)

    # image

    self.all_trans = all_trans
    
    if self.all_trans == True:    
        self.u01 = Deconv(768, 512)
        self.u02 = Deconv(512, 256)
        self.u03 = Deconv(256, 128)
        self.u04 = Deconv(128, 64)
    else:
        self.img_conv = DoubleConv(3, 64)
        
    # z3

    # n x 384 x 14 x 14
    self.u11 = Deconv(768, 512)
    self.u12 = Deconv(512, 256)
    self.u13 = Deconv(256, 128)


    # dc3

    self.dc3 = DoubleConv(128, 128)

    # du3

    self.du3 = nn.ConvTranspose2d(128, 64, 2, 2)

    # dc4

    self.dc4 = DoubleConv(128, 64)

    # dc5

    self.segmentation_head = nn.Conv2d(64, num_classes, 3, 1, 1)
    
    self.vis = vis

  def forward(self, x):
    if self.vis == True:
      features, attn_weights, _ = self.trans(x)
    else:
      features = self.trans(x)
    n, _, c = features[0].shape
    features = [i.permute(0, 2, 1).reshape(n, c, 14, 14) for i in features]

    if len(self.layers_to_block) != 0:
      features = [features[i] * 0 for i in range(len(features)) if i in self.layers_to_block]
    
    if self.all_trans == True:
        img = self.u01(features[0])
        img = self.u02(img)
        img = self.u03(img)
        img = self.u04(img)
    else:
        img = self.img_conv(x)
    
    x1 = self.u11(features[2])
    x1 = self.u12(x1)
    x1 = self.u13(x1)

    
    x12 = self.dc3(x1)
    x13 = self.du3(x12)
   
    x14 = torch.cat((img, x13), 1)
   
    x15 = self.dc4(x14)
    x16 = self.segmentation_head(x15)

    if self.vis == True:
      return x16, attn_weights
    else:
      return x16
    
######################## UNETR Compressed Single Layer ########################    

class UNETR_Compressed_Single_Layer(nn.Module):
  def __init__(self, num_classes = 4, all_trans = False, pretrained = True, layers_to_block = [], vis = True):
    super().__init__()

    self.layers_to_block = layers_to_block

    self.trans = ViT(load_pretrained = pretrained, num_layers = 1, vis = vis)

    # image

    self.all_trans = all_trans
    
    if self.all_trans == True:    
        self.u01 = Deconv(768, 512)
        self.u02 = Deconv(512, 256)
        self.u03 = Deconv(256, 128)
        self.u04 = Deconv(128, 64)
    else:
        self.img_conv = DoubleConv(3, 64)
        
    # z3

    # n x 384 x 14 x 14
    self.u11 = Deconv(768, 512)
    self.u12 = Deconv(512, 256)
    self.u13 = Deconv(256, 128)


    # dc3

    self.dc3 = DoubleConv(128, 128)

    # du3

    self.du3 = nn.ConvTranspose2d(128, 64, 2, 2)

    # dc4

    self.dc4 = DoubleConv(128, 64)

    # dc5

    self.segmentation_head = nn.Conv2d(64, num_classes, 3, 1, 1)
    
    self.vis = vis

  def forward(self, x):
    if self.vis == True:
      features, attn_weights, _ = self.trans(x)
    else:
      features = self.trans(x)
    n, _, c = features[0].shape
    features = [i.permute(0, 2, 1).reshape(n, c, 14, 14) for i in features]

    if len(self.layers_to_block) != 0:
      features = [features[i] * 0 for i in range(len(features)) if i in self.layers_to_block]
    
    if self.all_trans == True:
        img = self.u01(features[0])
        img = self.u02(img)
        img = self.u03(img)
        img = self.u04(img)
    else:
        img = self.img_conv(x)
    
    x1 = self.u11(features[0])
    x1 = self.u12(x1)
    x1 = self.u13(x1)

    
    x12 = self.dc3(x1)
    x13 = self.du3(x12)
   
    x14 = torch.cat((img, x13), 1)
   
    x15 = self.dc4(x14)
    x16 = self.segmentation_head(x15)

    if self.vis == True:
      return x16, attn_weights
    else:
      return x16
    
######################## CATS ########################    
    
class Deconv(nn.Module):
  def __init__(self, in_channels, out_channels):
    super().__init__()
    self.conv = nn.Sequential(
        nn.ConvTranspose2d(in_channels, out_channels, 2, 2),
        nn.Conv2d(out_channels, out_channels, 3, 1, 1, bias = False),
        nn.BatchNorm2d(out_channels),
        nn.ReLU(True)
    )

  def forward(self, x):
    return self.conv(x)


class DoubleConv(nn.Module):
  def __init__(self, in_channels, out_channels):
    super().__init__()
    self.conv = nn.Sequential(
        nn.Conv2d(in_channels, out_channels, 3, 1, 1, bias = False),
        nn.BatchNorm2d(out_channels),
        nn.ReLU(True),
        nn.Conv2d(out_channels, out_channels, 3, 1, 1, bias = False),
        nn.BatchNorm2d(out_channels),
        nn.ReLU(True)
    )

  def forward(self, x):
    return self.conv(x)

class CATS(nn.Module):
  def __init__(self, num_classes = 4, pretrained = True, backbone = "resnet50", pretrained_backbone = True, vis = True):
    super().__init__()
    self.trans = ViT(load_pretrained = pretrained, vis = vis)

    # image
    
    self.img_conv = DoubleConv(3, 64)
    self.backbone = timm.create_model(backbone, features_only = True, pretrained = pretrained_backbone, out_indices = (0, 1, 2, 3))
    self.channels = self.backbone.feature_info.channels()
        
    # z3

    # n x 384 x 14 x 14
    self.u11 = Deconv(768, 512)
    self.u12 = Deconv(512, 256)
    self.u13 = Deconv(256, self.channels[0])

    # z6

    # n x 384 x 14 x 14
    self.u21 = Deconv(768, 512)
    self.u22 = Deconv(512, self.channels[1])

    # z9

    # n x 384 x 14 x 14
    self.u31 = Deconv(768, self.channels[2])

    # z12

    # n x 384 x 14 x 14
    self.u40 = DoubleConv(768, self.channels[3])
    self.u41 = nn.ConvTranspose2d(self.channels[3], self.channels[2], 2, 2)

    # dc1
    

    self.dc1 = DoubleConv(self.channels[2] + self.channels[2], self.channels[2])

    # du1

    self.du1 = nn.ConvTranspose2d(self.channels[2], self.channels[1], 2, 2)

    # dc2

    self.dc2 = DoubleConv(self.channels[1] + self.channels[1], self.channels[1])

    # du2

    self.du2 = nn.ConvTranspose2d(self.channels[1], self.channels[0], 2, 2)

    # dc3

    self.dc3 = DoubleConv(self.channels[0] + self.channels[0], self.channels[0])

    # du3

    self.du3 = nn.ConvTranspose2d(self.channels[0], 64, 2, 2)

    # dc4

    self.dc4 = DoubleConv(128, 64)

    # dc5

    self.segmentation_head = nn.Conv2d(64, num_classes, 3, 1, 1)
    
    self.vis = vis
    

  def forward(self, x):
    if self.vis == True:
      features, attn_weights, _ = self.trans(x)
    else:
      features = self.trans(x)
    n, _, c = features[0].shape
    features = [i.permute(0, 2, 1).reshape(n, c, 14, 14) for i in features]
    
    img = self.img_conv(x)
    backbone_features = self.backbone(x)
    
    x1 = self.u11(features[2])
    x1 = self.u12(x1)
    x1 = self.u13(x1)
    x1 = x1 + backbone_features[0]

    x2 = self.u21(features[5])
    x2 = self.u22(x2)
    x2 = x2 + backbone_features[1]

    x3 = self.u31(features[8])
    x3 = x3 + backbone_features[2] 

    x4 = self.u40(features[11])
    x4 = x4 + backbone_features[3]
    x4 = self.u41(x4) 

    x5 = torch.cat((x3, x4), 1)
    
    x6 = self.dc1(x5)
    x7 = self.du1(x6)
    
    x8 = torch.cat((x2, x7), 1)

    x9 = self.dc2(x8)
    x10 = self.du2(x9)
    
    x11 = torch.cat((x1, x10), 1)
    
    x12 = self.dc3(x11)
    x13 = self.du3(x12)

    x14 = torch.cat((img, x13), 1)

    x15 = self.dc4(x14)
    x16 = self.segmentation_head(x15)

    if self.vis == True:
      return x16, attn_weights
    else:
      return x16
    
######################## CATS Compressed ########################    
    
class Deconv(nn.Module):
  def __init__(self, in_channels, out_channels):
    super().__init__()
    self.conv = nn.Sequential(
        nn.ConvTranspose2d(in_channels, out_channels, 2, 2),
        nn.Conv2d(out_channels, out_channels, 3, 1, 1, bias = False),
        nn.BatchNorm2d(out_channels),
        nn.ReLU(True)
    )

  def forward(self, x):
    return self.conv(x)


class DoubleConv(nn.Module):
  def __init__(self, in_channels, out_channels):
    super().__init__()
    self.conv = nn.Sequential(
        nn.Conv2d(in_channels, out_channels, 3, 1, 1, bias = False),
        nn.BatchNorm2d(out_channels),
        nn.ReLU(True),
        nn.Conv2d(out_channels, out_channels, 3, 1, 1, bias = False),
        nn.BatchNorm2d(out_channels),
        nn.ReLU(True)
    )

  def forward(self, x):
    return self.conv(x)

class CATS_Compressed(nn.Module):
  def __init__(self, num_classes = 4, pretrained = True, backbone = "resnet50", pretrained_backbone = True, vis = True):
    super().__init__()
    self.trans = ViT(load_pretrained = pretrained, num_layers = 3, vis = vis)

    # image

    self.img_conv = DoubleConv(3, 64)
    self.backbone = timm.create_model(backbone, features_only = True, pretrained = pretrained_backbone, out_indices = (0, 1, 2, 3))
    self.channels = self.backbone.feature_info.channels()
        
    # z3

    # n x 384 x 14 x 14
    self.u10 = DoubleConv(768, self.channels[3])
    self.u11 = Deconv(self.channels[3], 512)
    self.u12 = Deconv(512, 256)
    self.u13 = Deconv(256, 64)

    # dc3

    self.dc3 = DoubleConv(64, 128)

    # du3

    self.du3 = nn.ConvTranspose2d(128, 64, 2, 2)

    # dc4

    self.dc4 = DoubleConv(128, 64)

    # dc5

    self.segmentation_head = nn.Conv2d(64, num_classes, 3, 1, 1)
    
    self.vis = vis

  def forward(self, x):

    if self.vis == True:
      features, attn_weights, _ = self.trans(x)
    else:
      features = self.trans(x)
    n, _, c = features[0].shape
    features = [i.permute(0, 2, 1).reshape(n, c, 14, 14) for i in features]
    
    img = self.img_conv(x)
    backbone_features = self.backbone(x)
    
    x1 = self.u10(features[2]) + backbone_features[3]
    x1 = self.u11(x1)
    x1 = self.u12(x1)
    x1 = self.u13(x1)

    x12 = self.dc3(x1)

    x13 = self.du3(x12)


    x14 = torch.cat((img, x13), 1)

    x15 = self.dc4(x14)
    x16 = self.segmentation_head(x15)

    if self.vis == True:
      return x16, attn_weights
    else:
      return x16
    
######################## Wrappers ########################

class TransUNet_wrapper(nn.Module):
  def __init__(self, model):
    super(TransUNet_wrapper, self).__init__()
    self.model = model

  def forward(self, x):
    return self.model(x)[0]
    
class ViT_wrapper(nn.Module):
  def __init__(self, model):
    super(ViT_wrapper, self).__init__()
    self.model = model

  def forward(self, x):
    return self.model(x)[0]
    
class UNETR_wrapper(nn.Module):
  def __init__(self, model):
    super(UNETR_wrapper, self).__init__()
    self.model = model

  def forward(self, x):
    return self.model(x)[0]
    
class CATS_wrapper(nn.Module):
  def __init__(self, model):
    super(CATS_wrapper, self).__init__()
    self.model = model

  def forward(self, x):
    return self.model(x)[0]
