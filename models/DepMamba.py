"""DepMamba implementation.
Authors
-------
* Jiaxin Ye 2024
"""

import warnings
from dataclasses import dataclass
from typing import List, Optional

import abc
import torch
import torch.nn as nn
import torch.nn.functional as F

import speechbrain as sb
from speechbrain.nnet.activations import Swish
from speechbrain.nnet.attention import (
    MultiheadAttention,
    PositionalwiseFeedForward,
    RelPosMHAXL,
)
from speechbrain.nnet.hypermixing import HyperMixing
from speechbrain.nnet.normalization import LayerNorm
from speechbrain.utils.dynamic_chunk_training import DynChunkTrainConfig

# Mamba
from mamba_ssm import Mamba
from .mamba.bimamba import Mamba as BiMamba 
from .mamba.mm_bimamba import Mamba as MMBiMamba 
from .base import BaseNet



class MMMambaEncoderLayer(nn.Module):
    def __init__(
        self,
        d_model,
        d_ffn,
        activation='Swish',
        dropout=0.0,
        causal=False,
        mamba_config=None
    ):
        super().__init__()
        assert mamba_config != None

        if activation == 'Swish':
            activation = Swish
        elif activation == "GELU":
            activation = torch.nn.GELU
        else:
            activation = Swish

        bidirectional = mamba_config.pop('bidirectional')
        if causal or (not bidirectional):
            self.mamba = Mamba(
                d_model=d_model,
                **mamba_config
            )
        else:
            self.mamba = MMBiMamba(
                d_model=d_model,
                bimamba_type='v2',
                **mamba_config
            )
        mamba_config['bidirectional'] = bidirectional

        self.norm1 = LayerNorm(d_model, eps=1e-6)
        self.norm2 = LayerNorm(d_model, eps=1e-6)
        self.drop = nn.Dropout(dropout)

        self.a_downsample = nn.Sequential(
            nn.Conv1d(d_model, d_model, kernel_size=16, stride=2, padding=8),
            nn.BatchNorm1d(d_model),
        )

    def forward(
        self,
        a_x, v_x, 
        a_inference_params = None,
        v_inference_params = None
    ):
        
        a_out1, v_out1 = self.mamba(a_x, v_x,a_inference_params,v_inference_params)
        a_out = a_x + self.norm1(a_out1)
        v_out = v_x + self.norm2(v_out1)

        return a_out, v_out

class MMCNNEncoderLayer(nn.Module):
    def __init__(
        self,
        input_size,
        output_size,
        dropout=0.0,
        causal=False,
        dilation=1,
    ):
        super().__init__()

        self.a_conv = nn.Conv1d(input_size, output_size, 3, padding=1, dilation=dilation, bias=False)
        self.a_bn = nn.BatchNorm1d(output_size)

        self.v_conv = nn.Conv1d(input_size, output_size, 3, padding=1, dilation=dilation, bias=False)
        self.v_bn = nn.BatchNorm1d(output_size)

        self.relu = nn.ReLU()

        self.a_drop = nn.Dropout(dropout)
        self.v_drop = nn.Dropout(dropout)

        self.a_net = nn.Sequential(self.a_conv, self.a_bn, self.relu, self.a_drop)
        self.v_net = nn.Sequential(self.v_conv, self.v_bn, self.relu, self.v_drop)

        if input_size != output_size:
            self.a_skipconv = nn.Conv1d(input_size, output_size, 1, padding=0, dilation=dilation, bias=False)
            self.v_skipconv = nn.Conv1d(input_size, output_size, 1, padding=0, dilation=dilation, bias=False)
        else:
            self.a_skipconv = None
            self.v_skipconv = None

        self.init_weights()

    def init_weights(self):
        nn.init.xavier_uniform_(self.a_conv.weight.data)
        nn.init.xavier_uniform_(self.v_conv.weight.data)
        # nn.init.xavier_uniform_(self.conv2.weight.data)

    def forward(self, xa, xv):
        a_out = self.a_net(xa)
        v_out = self.v_net(xv)
        if self.a_skipconv is not None:
            xa = self.a_skipconv(xa)
        if self.v_skipconv is not None:
            xv = self.v_skipconv(xv)
        a_out = a_out+xa
        v_out = v_out+xv
        return a_out, v_out

class MambaEncoderLayer(nn.Module):
    def __init__(
        self,
        d_model,
        d_ffn,
        activation='Swish',
        dropout=0.0,
        causal=False,
        mamba_config=None
    ):
        super().__init__()
        assert mamba_config != None

        if activation == 'Swish':
            activation = Swish
        elif activation == "GELU":
            activation = torch.nn.GELU
        else:
            activation = Swish

        bidirectional = mamba_config.pop('bidirectional')
        if causal or (not bidirectional):
            self.mamba = Mamba(
                d_model=d_model,
                **mamba_config
            )
        else:
            self.mamba = BiMamba(
                d_model=d_model,
                bimamba_type='v2',
                **mamba_config
            )
        mamba_config['bidirectional'] = bidirectional

        self.norm1 = LayerNorm(d_model, eps=1e-6)
        self.drop = nn.Dropout(dropout)


    def forward(
        self,
        x, inference_params = None
    ):
        out = x + self.norm1(self.mamba(x, inference_params))
        return out

class CNNEncoderLayer(nn.Module):
    def __init__(
        self,
        input_size,
        output_size,
        dropout=0.0,
        causal=False,
        dilation=1,
    ):
        super().__init__()

        self.conv1 = nn.Conv1d(input_size, output_size, 3, padding=1, dilation=dilation, bias=False)
        self.bn1 = nn.BatchNorm1d(output_size)
        self.relu1 = nn.ReLU()
        # self.conv2 = nn.Conv1d(output_size, output_size, 5, padding=2, dilation=dilation, bias=False)
        # self.bn2 = nn.BatchNorm1d(output_size)
        # self.relu2 = nn.ReLU()

        self.drop = nn.Dropout(dropout)

        self.net = nn.Sequential(self.conv1, self.bn1, self.relu1, self.drop)

        if input_size != output_size:
            self.conv = nn.Conv1d(input_size, output_size, 1, padding=0, dilation=dilation, bias=False)
        else:
            self.conv = None
        self.init_weights()

    def init_weights(self):
        nn.init.xavier_uniform_(self.conv1.weight.data)
        # nn.init.xavier_uniform_(self.conv2.weight.data)

    def forward(self, x):
        out = self.net(x)
        if self.conv is not None:
            x = self.conv(x)
        out = out+x
        return out

class CoSSM(nn.Module):
    """This class implements the CoSSM encoder.
    """
    def __init__(
        self,
        num_layers,
        input_size,
        output_sizes=[256,512,512],
        d_ffn=1024,
        activation='Swish',
        dropout=0.0,
        kernel_size = 3,
        causal=False,
        mamba_config=None
    ):
        super().__init__()
        prev_input_size = input_size

        cnn_list = []
        mamba_list = []
        # print(output_sizes)
        for i in range(len(output_sizes)):
            cnn_list.append(MMCNNEncoderLayer(
                    input_size = input_size if i<1 else output_sizes[i-1],
                    output_size = output_sizes[i],
                    dropout=dropout
                ))
            mamba_list.append(MMMambaEncoderLayer(
                    d_model=output_sizes[i],
                    d_ffn=d_ffn,
                    dropout=dropout,
                    activation=activation,
                    causal=causal,
                    mamba_config=mamba_config,
                ))

        self.mamba_layers = torch.nn.ModuleList(mamba_list)
        self.cnn_layers = torch.nn.ModuleList(cnn_list)


    def forward(
        self,
        a_x, v_x, 
        a_inference_params = None,
        v_inference_params = None
    ):
        a_out = a_x
        v_out = v_x

        for cnn_layer, mamba_layer in zip(self.cnn_layers, self.mamba_layers):
            a_out, v_out  = cnn_layer(a_out.permute(0,2,1), v_out.permute(0,2,1))
            a_out = a_out.permute(0,2,1)
            v_out = v_out.permute(0,2,1)
            a_out, v_out = mamba_layer(
                a_out, v_out,
                a_inference_params = a_inference_params,
                v_inference_params = v_inference_params
            )
            
        return a_out, v_out

class EnSSM(nn.Module):
    """This class implements the EnSSM encoder.
    """
    def __init__(
        self,
        num_layers,
        input_size,
        output_sizes=[256,512,512],
        d_ffn=1024,
        activation='Swish',
        dropout=0.0,
        causal=False,
        mamba_config=None
    ):
        super().__init__()
        prev_input_size = input_size

        cnn_list = []
        mamba_list = []
        # print(output_sizes)
        for i in range(len(output_sizes)):
            cnn_list.append(CNNEncoderLayer(
                    input_size = input_size if i<1 else output_sizes[i-1],
                    output_size = output_sizes[i],
                    dropout=dropout
                ))
            mamba_list.append(MambaEncoderLayer(
                    d_model=output_sizes[i],
                    d_ffn=d_ffn,
                    dropout=dropout,
                    activation=activation,
                    causal=causal,
                    mamba_config=mamba_config,
                ))

        self.mamba_layers = torch.nn.ModuleList(mamba_list)
        self.cnn_layers = torch.nn.ModuleList(cnn_list)


    def forward(
        self,
        x,
        inference_params = None,
    ):
        out = x

        for cnn_layer, mamba_layer in zip(self.cnn_layers, self.mamba_layers):
            out  = cnn_layer(out.permute(0,2,1))
            out = out.permute(0,2,1)
            out = mamba_layer(
                out,
                inference_params = inference_params,
            )

        return out


def masked_temporal_mean(x, padding_mask=None, eps=1e-8):
    if padding_mask is None:
        return x.mean(dim=1)
    mask = padding_mask.unsqueeze(-1).to(dtype=x.dtype, device=x.device)
    denom = mask.sum(dim=1).clamp_min(eps)
    return (x * mask).sum(dim=1) / denom


def local_cosine_alignment_loss(xa, xv, padding_mask=None, eps=1e-8):
    xa = F.normalize(xa, p=2, dim=-1, eps=eps)
    xv = F.normalize(xv, p=2, dim=-1, eps=eps)
    frame_loss = 1.0 - (xa * xv).sum(dim=-1)
    if padding_mask is None:
        return frame_loss.mean()
    mask = padding_mask.to(dtype=frame_loss.dtype, device=frame_loss.device)
    return (frame_loss * mask).sum() / mask.sum().clamp_min(eps)


def _directional_window_soft_alignment_loss(
    query,
    key,
    query_mask=None,
    key_mask=None,
    window_size=4,
    temperature=0.1,
    eps=1e-8,
):
    batch_size, query_steps, _ = query.shape
    key_steps = key.size(1)
    query_n = F.normalize(query, p=2, dim=-1, eps=eps)
    key_n = F.normalize(key, p=2, dim=-1, eps=eps)

    if query_mask is None:
        query_mask = torch.ones(
            batch_size, query_steps, dtype=torch.bool, device=query.device
        )
    else:
        query_mask = query_mask.to(dtype=torch.bool, device=query.device)

    if key_mask is None:
        key_mask = torch.ones(
            batch_size, key_steps, dtype=torch.bool, device=key.device
        )
    else:
        key_mask = key_mask.to(dtype=torch.bool, device=key.device)

    frame_losses = []
    valid_queries = []
    temperature = max(float(temperature), eps)
    window_size = max(int(window_size), 0)

    for t in range(query_steps):
        left = max(0, t - window_size)
        right = min(key_steps, t + window_size + 1)

        q = query_n[:, t:t + 1, :]
        k = key_n[:, left:right, :]
        key_window_mask = key_mask[:, left:right]

        sim = torch.sum(q * k, dim=-1)
        sim = sim.masked_fill(~key_window_mask, -1e9)
        weights = torch.softmax(sim / temperature, dim=-1)

        aligned_key = torch.sum(weights.unsqueeze(-1) * k, dim=1)
        aligned_key = F.normalize(aligned_key, p=2, dim=-1, eps=eps)
        loss_t = 1.0 - torch.sum(query_n[:, t, :] * aligned_key, dim=-1)

        valid_t = query_mask[:, t] & key_window_mask.any(dim=1)
        frame_losses.append(loss_t)
        valid_queries.append(valid_t)

    frame_loss = torch.stack(frame_losses, dim=1)
    valid_mask = torch.stack(valid_queries, dim=1).to(
        dtype=frame_loss.dtype, device=frame_loss.device
    )
    return (frame_loss * valid_mask).sum() / valid_mask.sum().clamp_min(eps)


def window_soft_alignment_loss(
    xa,
    xv,
    padding_mask=None,
    window_size=4,
    temperature=0.1,
    eps=1e-8,
):
    a2v_loss = _directional_window_soft_alignment_loss(
        xa,
        xv,
        query_mask=padding_mask,
        key_mask=padding_mask,
        window_size=window_size,
        temperature=temperature,
        eps=eps,
    )
    v2a_loss = _directional_window_soft_alignment_loss(
        xv,
        xa,
        query_mask=padding_mask,
        key_mask=padding_mask,
        window_size=window_size,
        temperature=temperature,
        eps=eps,
    )
    return 0.5 * (a2v_loss + v2a_loss)


def rbf_mmd_loss(x, y, kernel_mul=2.0, kernel_num=5, fixed_sigma=None, eps=1e-8):
    batch_size = x.size(0)
    if batch_size == 0:
        return x.new_tensor(0.0)

    total = torch.cat([x, y], dim=0)
    sq_dist = torch.cdist(total, total, p=2).pow(2)

    if fixed_sigma is None:
        denom = max(total.size(0) * total.size(0) - total.size(0), 1)
        bandwidth = sq_dist.detach().sum() / denom
    else:
        bandwidth = x.new_tensor(float(fixed_sigma))
    bandwidth = bandwidth.clamp_min(eps)
    bandwidth = bandwidth / (kernel_mul ** (kernel_num // 2))

    kernels = 0.0
    for i in range(kernel_num):
        kernels = kernels + torch.exp(
            -sq_dist / ((bandwidth * (kernel_mul ** i)).clamp_min(eps))
        )

    xx = kernels[:batch_size, :batch_size]
    yy = kernels[batch_size:, batch_size:]
    xy = kernels[:batch_size, batch_size:]
    yx = kernels[batch_size:, :batch_size]
    return xx.mean() + yy.mean() - xy.mean() - yx.mean()

class DepMamba(BaseNet):

    def __init__(self, audio_input_size=161, video_input_size=161, mm_input_size=128, mm_output_sizes=[256,64], d_ffn=1024, num_layers=8, dropout=0.1, activation='Swish', causal=False, mamba_config=None, use_local_alignment=False, use_global_alignment=False, local_alignment_mode="window_soft", local_alignment_window=4, local_alignment_temperature=0.1, mmd_kernel_mul=2.0, mmd_kernel_num=5, mmd_fixed_sigma=None):
        super().__init__()
        self.use_local_alignment = use_local_alignment
        self.use_global_alignment = use_global_alignment
        self.local_alignment_mode = str(local_alignment_mode).lower()
        self.local_alignment_window = local_alignment_window
        self.local_alignment_temperature = local_alignment_temperature
        self.mmd_kernel_mul = mmd_kernel_mul
        self.mmd_kernel_num = mmd_kernel_num
        self.mmd_fixed_sigma = mmd_fixed_sigma
        self.aux_losses = {}

        if self.local_alignment_mode not in ("hard", "window_soft"):
            raise ValueError(
                "local_alignment_mode must be either 'hard' or 'window_soft'"
            )

        self.cossm_encoder = CoSSM(num_layers,
                                         mm_input_size,
                                    mm_output_sizes,
                                    d_ffn,
                                    activation=activation,
                                    dropout=dropout,
                                    causal=causal,
                                    mamba_config=mamba_config)

        self.conv_audio = nn.Conv1d(audio_input_size, mm_input_size, 1, padding=0, dilation=1, bias=False)
        self.conv_video = nn.Conv1d(video_input_size, mm_input_size, 1, padding=0, dilation=1, bias=False)
        
        self.enssm_encoder = EnSSM(num_layers,
                                    mm_output_sizes[-1]*2,
                                    [mm_output_sizes[-1]*2],
                                    d_ffn,
                                    activation=activation,
                                    dropout=dropout,
                                    causal=causal,
                                    mamba_config=mamba_config)
        
        self.pool = nn.AdaptiveMaxPool1d(1)

        self.output = nn.Linear(mm_output_sizes[-1]*2, 1)
        self.m = nn.Sigmoid()

        nn.init.xavier_uniform_(self.conv_audio.weight.data)
        nn.init.xavier_uniform_(self.conv_video.weight.data)
        

    def feature_extractor(self, x, padding_mask=None, a_inference_params = None, v_inference_params = None):
        xa = x[:, :, 136:]
        xv = x[:, :, :136]
        xa = self.conv_audio(xa.permute(0,2,1)).permute(0,2,1)
        xv = self.conv_video(xv.permute(0,2,1)).permute(0,2,1)

        zero = x.new_tensor(0.0)
        self.aux_losses = {
            "local_align_loss": zero,
            "global_align_loss": zero,
        }
        if self.use_local_alignment:
            if self.local_alignment_mode == "hard":
                self.aux_losses["local_align_loss"] = local_cosine_alignment_loss(
                    xa, xv, padding_mask
                )
            elif self.local_alignment_mode == "window_soft":
                self.aux_losses["local_align_loss"] = window_soft_alignment_loss(
                    xa,
                    xv,
                    padding_mask,
                    window_size=self.local_alignment_window,
                    temperature=self.local_alignment_temperature,
                )
        if self.use_global_alignment:
            za = masked_temporal_mean(xa, padding_mask)
            zv = masked_temporal_mean(xv, padding_mask)
            self.aux_losses["global_align_loss"] = rbf_mmd_loss(
                za,
                zv,
                kernel_mul=self.mmd_kernel_mul,
                kernel_num=self.mmd_kernel_num,
                fixed_sigma=self.mmd_fixed_sigma,
            )

        xa, xv = self.cossm_encoder(xa, xv, a_inference_params, v_inference_params)

        x = torch.cat([xa,xv],dim=-1)
        x = self.enssm_encoder(x)
        
        if padding_mask is not None:
            x = x * (padding_mask.unsqueeze(-1).float())
            x = x.sum(dim=1) / (padding_mask.unsqueeze(-1).float()
                                ).sum(dim=1, keepdim=False)  # Compute average
        else:
            x = self.pool(x.permute(0,2,1)).squeeze(-1)
        return x

    def classifier(self, x):
        return self.output(x)
