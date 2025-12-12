import torch
import torch.nn.functional as F
from torch import nn

import matplotlib.pyplot as plt
import math
import os

from functools import partial
from torchmetrics.classification import BinaryJaccardIndex, F1Score, BinaryPrecisionRecallCurve

import pytorch_lightning as pl

from sam.segment_anything.modeling.image_encoder import ImageEncoderViT
from sam.segment_anything.modeling.mask_decoder import MaskDecoder
from sam.segment_anything.modeling.prompt_encoder import PromptEncoder
from sam.segment_anything.modeling.transformer import TwoWayTransformer
from sam.segment_anything.modeling.common import LayerNorm2d

import pprint
import torchvision
import numpy as np

import vitdet
from typing import Optional, List


class BilinearSampler(nn.Module):
    def __init__(self, image_size):
        super(BilinearSampler, self).__init__()
        self.image_size = image_size

    def forward(self, feature_maps, sample_points):
        """
        Args:
            feature_maps (Tensor): The input feature tensor of shape [B, D, H, W].
            sample_points (Tensor): The 2D sample points of shape [B, N_points, 2],
                                    each point in the range [-1, 1], format (x, y).
        Returns:
            Tensor: Sampled feature vectors of shape [B, N_points, D].
        """
        B, D, H, W = feature_maps.shape
        _, N_points, _ = sample_points.shape

        # normalize cooridinates to (-1, 1) for grid_sample
        sample_points = (sample_points / self.image_size) * 2.0 - 1.0
        
        # sample_points from [B, N_points, 2] to [B, N_points, 1, 2] for grid_sample
        sample_points = sample_points.unsqueeze(2)
        
        # Use grid_sample for bilinear sampling. Align_corners set to False to use -1 to 1 grid space.
        # [B, D, N_points, 1]
        sampled_features = F.grid_sample(feature_maps, sample_points, mode='bilinear', align_corners=False)
        
        # sampled_features is [B, N_points, D]
        sampled_features = sampled_features.squeeze(dim=-1).permute(0, 2, 1)
        return sampled_features
    

class TopoNet(nn.Module):
    def __init__(self, config, feature_dim):
        super(TopoNet, self).__init__()
        self.config = config

        self.hidden_dim = 128
        self.heads = 4
        self.num_attn_layers = 3

        self.feature_proj = nn.Linear(feature_dim, self.hidden_dim) # (256, 128)
        self.pair_proj = nn.Linear(2 * self.hidden_dim + 2, self.hidden_dim) # (256 + 2, 128)

        # Create Transformer Encoder Layer
        encoder_layer = nn.TransformerEncoderLayer(
            d_model=self.hidden_dim,
            nhead=self.heads,
            dim_feedforward=self.hidden_dim,
            dropout=0.1,
            activation='relu',
            batch_first=True  # Input format is [batch size, sequence length, features]
        )
        
        # Stack the Transformer Encoder Layers
        if self.config.TOPONET_VERSION != 'no_transformer':
            self.transformer_encoder = nn.TransformerEncoder(encoder_layer, num_layers=self.num_attn_layers)
        self.output_proj = nn.Linear(self.hidden_dim, 1)

    def forward(self, points, point_features, pairs, pairs_valid, mask_logits=None):
        # points: [B, N_points, 2]
        # point_features: [B, N_points, D]
        # pairs: [B, N_samples, N_pairs, 2]
        # pairs_valid: [B, N_samples, N_pairs]
        # [bs, N_points, 256] -> [bs, N_points, 128]
        point_features = F.relu(self.feature_proj(point_features))
        # gathers pairs
        batch_size, n_samples, n_pairs, _ = pairs.shape # bs, 512, 16, 2
        pairs = pairs.view(batch_size, -1, 2) # [B, N_samples * N_pairs, 2]
        # [B, N_samples * N_pairs]
        batch_indices = torch.arange(batch_size).view(-1, 1).expand(-1, n_samples * n_pairs)
        # Use advanced indexing to fetch the corresponding feature vectors
        # [B, N_samples * N_pairs, D]
        src_features = point_features[batch_indices, pairs[:, :, 0]] # [B, 255, 128] index:[B, 512*16], [B, 512*16] ->result:[16, 512*16, 128]
        tgt_features = point_features[batch_indices, pairs[:, :, 1]]
        # [B, N_samples * N_pairs, 2]
        src_points = points[batch_indices, pairs[:, :, 0]]
        tgt_points = points[batch_indices, pairs[:, :, 1]]
        offset = tgt_points - src_points

        ## ablation study
        # [B, N_samples * N_pairs, 2D + 2]
        if self.config.TOPONET_VERSION == 'no_tgt_features':
            pair_features = torch.concat([src_features, torch.zeros_like(tgt_features), offset], dim=2)
        if self.config.TOPONET_VERSION == 'no_offset':
            pair_features = torch.concat([src_features, tgt_features, torch.zeros_like(offset)], dim=2)
        else:
            pair_features = torch.concat([src_features, tgt_features, offset], dim=2) # [16, 8192, 256 + 2]
        
        
        # [B, N_samples * N_pairs, D]
        pair_features = F.relu(self.pair_proj(pair_features)) # [16, 8192, 256 + 2] -> [16, 8192, 128]
        
        # attn applies within each local graph sample
        pair_features = pair_features.view(batch_size * n_samples, n_pairs, -1)
        # valid->not a padding
        pairs_valid = pairs_valid.view(batch_size * n_samples, n_pairs)

        # [B * N_samples, 1]
        #### flips mask for all-invalid pairs to prevent NaN
        all_invalid_pair_mask = torch.eq(torch.sum(pairs_valid, dim=-1), 0).unsqueeze(-1) # True means all pairs are invalid for this sample
        pairs_valid = torch.logical_or(pairs_valid, all_invalid_pair_mask)

        padding_mask = ~pairs_valid
        
        ## ablation study
        if self.config.TOPONET_VERSION != 'no_transformer':
            pair_features = self.transformer_encoder(pair_features, src_key_padding_mask=padding_mask) # input shape [S, B, D]
        
        ## Seems like at inference time, the returned n_pairs heres might be less - it's the
        # max num of valid pairs across all samples in the batch
        _, n_pairs, _ = pair_features.shape
        pair_features = pair_features.view(batch_size, n_samples, n_pairs, -1)

        # [B, N_samples, N_pairs, 1]
        logits = self.output_proj(pair_features)

        scores = torch.sigmoid(logits)

        return logits, scores


def fourier_encode_angle(angle: torch.Tensor, num_bases: int = 4) -> torch.Tensor:
    """
    angle: [*, 1] or [*], radians
    return: [*, 2 * num_bases] with [sin(mθ), cos(mθ)]_{m=1..num_bases}
    """
    if angle.dim() == 1:
        angle = angle.unsqueeze(-1)
    outs = []
    for m in range(1, num_bases + 1):
        outs.append(torch.sin(m * angle))
        outs.append(torch.cos(m * angle))
    return torch.cat(outs, dim=-1)


def softmin(t: torch.Tensor, dim: int = -1, tau: float = 10.0) -> torch.Tensor:
    # softmin(x) = -1/tau * log sum exp(-tau x)
    return -torch.logsumexp(-tau * t, dim=dim) / tau


def normalize_grid(points_xy: torch.Tensor, image_size: int) -> torch.Tensor:
    """
    points_xy: [B, E, S, 2] in [0, image_size]
    return: normalized to [-1, 1] for grid_sample
    """
    return (points_xy / image_size) * 2.0 - 1.0


def sample_line_points(src: torch.Tensor, dst: torch.Tensor, num_samples: int) -> torch.Tensor:
    """
    src: [B, E, 2], dst: [B, E, 2]
    return: [B, E, S, 2] where S=num_samples, linear interpolation points
    """
    B, E, _ = src.shape
    t = torch.linspace(0, 1, num_samples, device=src.device).view(1, 1, num_samples, 1)
    src_exp = src.unsqueeze(2)  # [B,E,1,2]
    dst_exp = dst.unsqueeze(2)  # [B,E,1,2]
    pts = src_exp + t * (dst_exp - src_exp)
    return pts  # [B,E,S,2]


class GeodesicPathExtractor(nn.Module):
    """
    Lightweight, differentiable path-aware feature extractor:
    - Sample equidistant points on the line segments from the road probability map in mask_logits
    - Sample multiple times after multi-scale smoothing (avg pooling)
    - Statistical features: mean / std / softmin (softmin of (1-p), approximate worst road)
    Return [B, E, F_path] vector (E is number of edges, S*K)
    """
    def __init__(self, image_size: int, num_samples: int = 32,
                 pool_kernel_sizes: Optional[List[int]] = None,
                 tau_softmin: float = 10.0):
        super().__init__()
        self.image_size = image_size
        self.num_samples = num_samples
        self.pool_kernel_sizes = pool_kernel_sizes or [1, 5, 11]  # 1=original image, then two smoothing
        self.tau = tau_softmin

    def forward(self, mask_logits: torch.Tensor,
                src_points: torch.Tensor,  # [B, E, 2]
                dst_points: torch.Tensor   # [B, E, 2]
                ) -> torch.Tensor:
        """
        mask_logits: [B, 2, H, W], road probability in channel 1
        src_points/dst_points: [B, E, 2] in [0, image_size]
        return: [B, E, F_path]
        """
        B, _, H, W = mask_logits.shape
        assert H == self.image_size and W == self.image_size, "mask size mismatch"
        road_prob = torch.sigmoid(mask_logits[:, 1:2])  # [B,1,H,W]

        # prepare sampling points
        pts = sample_line_points(src_points, dst_points, self.num_samples)  # [B, E, S, 2]
        pts_norm = normalize_grid(pts, self.image_size)  # [-1,1]
        grid = pts_norm.view(B, -1, 1, 2)  # [B, E*S, 1, 2]

        feats = []
        for ksz in self.pool_kernel_sizes:
            if ksz == 1:
                smoothed = road_prob
            else:
                pad = ksz // 2
                smoothed = F.avg_pool2d(road_prob, kernel_size=ksz, stride=1, padding=pad)

            # sampling: grid_sample input is [B,C,H,W] and [B, N, 1, 2]
            sampled = F.grid_sample(smoothed, grid, mode='bilinear', align_corners=False)  # [B,1,E*S,1]
            sampled = sampled.view(B, -1, self.num_samples)  # [B, E, S]

            # statistics
            mean_v = sampled.mean(dim=-1, keepdim=True)                         # [B,E,1]
            std_v  = sampled.std(dim=-1, unbiased=False, keepdim=True)          # [B,E,1]
            softmin_v = softmin(1.0 - sampled, dim=-1, tau=self.tau).unsqueeze(-1)  # [B,E,1]

            feats.extend([mean_v, std_v, softmin_v])

        path_feat = torch.cat(feats, dim=-1)  # [B, E, 3 * len(scales)]
        return path_feat


class BiasedSelfAttentionLayer(nn.Module):
    """
    Customized multi-head self-attention layer with "additive bias" + FFN
    Support batch-wise [B, L, L] shape bias matrix bias
    """
    def __init__(self, d_model: int, nhead: int, dim_ff: int, dropout: float = 0.1):
        super().__init__()
        self.d_model = d_model
        self.nhead = nhead
        self.dk = d_model // nhead

        self.q_proj = nn.Linear(d_model, d_model)
        self.k_proj = nn.Linear(d_model, d_model)
        self.v_proj = nn.Linear(d_model, d_model)
        self.out_proj = nn.Linear(d_model, d_model)

        self.dropout = nn.Dropout(dropout)
        self.norm1 = nn.LayerNorm(d_model)
        self.ffn = nn.Sequential(
            nn.Linear(d_model, dim_ff),
            nn.ReLU(),
            nn.Linear(dim_ff, d_model),
        )
        self.norm2 = nn.LayerNorm(d_model)

    def forward(self, x: torch.Tensor,
                key_padding_mask: Optional[torch.Tensor] = None,  # [B,L], True means to mask the position
                bias: Optional[torch.Tensor] = None               # [B,L,L]
                ) -> torch.Tensor:
        """
        x: [B, L, H]
        """
        B, L, H = x.shape
        residual = x

        # Q,K,V
        q = self.q_proj(x).view(B, L, self.nhead, self.dk).transpose(1, 2)  # [B,h,L,dk]
        k = self.k_proj(x).view(B, L, self.nhead, self.dk).transpose(1, 2)  # [B,h,L,dk]
        v = self.v_proj(x).view(B, L, self.nhead, self.dk).transpose(1, 2)  # [B,h,L,dk]

        attn_scores = torch.matmul(q, k.transpose(-2, -1)) / math.sqrt(self.dk)  # [B,h,L,L]

        if bias is not None:
            # expand to each head
            attn_scores = attn_scores + bias.unsqueeze(1)  # [B,1,L,L] -> [B,h,L,L]

        if key_padding_mask is not None:
            # mask invalid keys: set the corresponding column to -inf
            # key_padding_mask: [B, L], True=pad
            mask = key_padding_mask.unsqueeze(1).unsqueeze(2)  # [B,1,1,L]
            attn_scores = attn_scores.masked_fill(mask, float('-inf'))

        attn = F.softmax(attn_scores, dim=-1)
        attn = self.dropout(attn)
        out = torch.matmul(attn, v)  # [B,h,L,dk]
        out = out.transpose(1, 2).contiguous().view(B, L, H)
        out = self.out_proj(out)
        x = self.norm1(residual + out)

        # FFN
        residual = x
        x = self.ffn(x)
        x = self.norm2(residual + x)
        return x


def build_edge_bias(src_xy: torch.Tensor,
                    dst_xy: torch.Tensor,
                    valid: torch.Tensor,
                    angle_sigma: float = math.pi / 4,
                    lambda_turn: float = 0.6,
                    lambda_compete: float = 0.2) -> torch.Tensor:
    """
    Construct edge-edge attention bias matrix within "a group of candidate edges from the same source point":
    - turn compatibility: smaller Δθ is more compatible (Gaussian kernel)
    - competitive prior: negative bias is applied to off-diagonal uniformly, encouraging sparse selection
    src_xy/dst_xy: [B, L, 2] (B is actually B*S, L is N_pairs)
    valid: [B, L], True means valid
    return: bias [B, L, L]
    """
    B, L, _ = src_xy.shape
    offset = dst_xy - src_xy  # [B,L,2]
    angle = torch.atan2(offset[..., 1], offset[..., 0])  # [B,L]

    # pairwise Δθ, mapped to [-pi, pi]
    theta_i = angle.unsqueeze(-1)           # [B,L,1]
    theta_j = angle.unsqueeze(-2)           # [B,1,L]
    delta = theta_i - theta_j               # [B,L,L]
    delta = (delta + math.pi) % (2 * math.pi) - math.pi

    k_turn = torch.exp(- (delta ** 2) / (2 * (angle_sigma ** 2)))  # [B,L,L]
    bias_turn = lambda_turn * (k_turn - 0.5)                        # centered

    # competitive prior: negative bias is applied to off-diagonal uniformly, encouraging sparse selection
    eye = torch.eye(L, device=src_xy.device).unsqueeze(0)          # [1,L,L]
    off_diag = 1.0 - eye
    bias_comp = -lambda_compete * off_diag                          # [1,L,L] -> broadcast

    bias = bias_turn + bias_comp

    # for invalid edges, avoid positive bias with anyone: set to 0 (actually masked by key_padding_mask)
    v = valid.float()
    bias = bias * v.unsqueeze(-1) * v.unsqueeze(-2)

    # set main diagonal to 0
    bias = bias * off_diag + 0.0 * eye
    return bias  # [B,L,L]


class MaGTopoNet(nn.Module):
    """
    Mask-aware Geodesic Line-Graph Transformer
    - Optional use point_features (image features)
    - Use geometric encoding (offset/dist/angle-Fourier)
    - Use path features (sampling statistics along the line on road prob)
    - Do self-attention with biased (multi-layer) on "each sample's candidate edge set"
    Interface kept consistent with TopoNet
    """
    def __init__(self, config, feature_dim: int,
                 use_point_features: bool = True,
                 use_path_features: bool = True,
                 use_edge_bias: bool = True):
        super().__init__()
        self.config = config
        self.hidden_dim = 256
        self.heads = 8
        self.num_layers = 4

        self.use_point_features = use_point_features
        self.use_path_features = use_path_features
        self.use_edge_bias = use_edge_bias

        # node feature projection
        if self.use_point_features:
            self.node_proj = nn.Linear(feature_dim, self.hidden_dim)

        # geometric feature encoding: dx,dy + dist + angle fourier(8) -> H/2
        geo_in_dim = 2 + 1 + 8
        self.geo_proj = nn.Linear(geo_in_dim, self.hidden_dim // 2)

        # path feature
        if self.use_path_features:
            # path_feat: 3 * len(scales)，默认=3*3=9
            self.path_extractor = GeodesicPathExtractor(
                image_size=config.PATCH_SIZE,
                num_samples=config.NUM_INTERPOLATIONS,
                pool_kernel_sizes=config.POOL_KERNEL_SIZES,
                tau_softmin=5.0,
            )
            self.path_proj = nn.Linear(9, self.hidden_dim // 2)

        # fused projection to H
        fused_in = 0
        if self.use_point_features:
            fused_in += self.hidden_dim * 2
        fused_in += self.hidden_dim // 2  # geo
        if self.use_path_features:
            fused_in += self.hidden_dim // 2

        self.edge_proj = nn.Linear(fused_in, self.hidden_dim)

        # Transformer encoder with biased
        self.layers = nn.ModuleList([
            BiasedSelfAttentionLayer(self.hidden_dim, self.heads, dim_ff=self.hidden_dim, dropout=0.10)
            for _ in range(self.num_layers)
        ])

        self.out = nn.Linear(self.hidden_dim, 1)

    def forward(self, points: torch.Tensor,              # [B, N_points, 2]
                point_features: torch.Tensor,            # [B, N_points, D]
                pairs: torch.Tensor,                     # [B, N_samples, N_pairs, 2]
                pairs_valid: torch.Tensor,               # [B, N_samples, N_pairs]
                mask_logits: Optional[torch.Tensor] = None  # [B, 2, H, W]
                ):
        B, S, K, _ = pairs.shape
        dev = points.device

        # 1) prepare indices
        pairs_flat = pairs.view(B, -1, 2)  # [B, S*K, 2]
        BE = S * K
        batch_idx = torch.arange(B, device=dev).view(-1, 1).expand(-1, BE)  # [B, S*K]

        # 2) node features
        if self.use_point_features:
            node = F.gelu(self.node_proj(point_features))  # [B, N, H]
            src_idx = pairs_flat[:, :, 0]  # [B,BE]
            dst_idx = pairs_flat[:, :, 1]  # [B,BE]
            src_feat = node[batch_idx, src_idx]  # [B, BE, H]
            dst_feat = node[batch_idx, dst_idx]  # [B, BE, H]

        # 3) geometric features
        src_xy = points[batch_idx, pairs_flat[:, :, 0]].float()  # [B,BE,2]
        dst_xy = points[batch_idx, pairs_flat[:, :, 1]].float()  # [B,BE,2]
        offset = dst_xy - src_xy  # [B,BE,2]
        dist = torch.norm(offset, dim=-1, keepdim=True)  # [B,BE,1]
        angle = torch.atan2(offset[..., 1], offset[..., 0]).unsqueeze(-1)  # [B,BE,1]
        angle_enc = fourier_encode_angle(angle, num_bases=4)  # [B,BE,8]
        # normalize offset and dist to align with angle_enc
        offset_norm = offset / mask_logits.shape[-1]
        dist_norm = dist / (math.sqrt(2) * mask_logits.shape[-1])
        geo_in = torch.cat([offset_norm, dist_norm, angle_enc], dim=-1)
        geo_feat = F.gelu(self.geo_proj(geo_in))  # [B,BE,H/2]

        # 4) path features (differentiable sampling from mask_logits)
        if self.use_path_features:
            assert mask_logits is not None, "mask_logits is required when use_path_features=True"
            path_feat_raw = self.path_extractor(mask_logits, src_xy, dst_xy)  # [B,BE,9]
            path_feat = F.gelu(self.path_proj(path_feat_raw))                 # [B,BE,H/2]

        # 5) edge token fusion
        feats = [geo_feat]
        if self.use_point_features:
            feats = [src_feat, dst_feat] + feats
        if self.use_path_features:
            feats = feats + [path_feat]

        edge_tok = torch.cat(feats, dim=-1)     # [B,BE, fused_in]
        edge_tok = F.gelu(self.edge_proj(edge_tok))  # [B,BE,H]

        # 6) within-group (each sample) encoding: reshape to [B*S, K, H]
        x = edge_tok.view(B, S, K, -1).view(B * S, K, -1)        # [B*S, K, H]
        valid = pairs_valid.view(B * S, K)                       # [B*S, K]
        # ensure at least one True in each group, prevent NaN due to all invalid
        all_invalid = (valid.sum(dim=-1, keepdim=True) == 0)
        valid = torch.logical_or(valid, all_invalid)

        # 7) construct bias
        if self.use_edge_bias:
            src_xy_g = src_xy.view(B, S, K, 2).view(B * S, K, 2)
            dst_xy_g = dst_xy.view(B, S, K, 2).view(B * S, K, 2)
            bias = build_edge_bias(src_xy_g, dst_xy_g, valid,
                                   angle_sigma=self.config.ANGLE_SIGMA,
                                   lambda_turn=self.config.LAMBDA_TURN,
                                   lambda_compete=self.config.LAMBDA_COMPETE)
        else:
            bias = None

        # 8) self-attention encoder with biased (multi-layer)
        key_padding_mask = ~valid  # True=pad
        for layer in self.layers:
            x = layer(x, key_padding_mask=key_padding_mask, bias=bias)  # [B*S,K,H]

        # 9) output
        x = x.view(B, S, K, -1)
        logits = self.out(x)                     # [B,S,K,1]
        scores = torch.sigmoid(logits)
        return logits, scores
    

class _LoRA_qkv(nn.Module):
    """In Sam it is implemented as
    self.qkv = nn.Linear(dim, dim * 3, bias=qkv_bias)
    B, N, C = x.shape
    qkv = self.qkv(x).reshape(B, N, 3, self.num_heads, self.head_dim).permute(2, 0, 3, 1, 4)
    q, k, v = qkv.unbind(0)
    """

    def __init__(
            self,
            qkv: nn.Module,
            linear_a_q: nn.Module,
            linear_b_q: nn.Module,
            linear_a_v: nn.Module,
            linear_b_v: nn.Module,
    ):
        super().__init__()
        # self.qkv = qkv
        self.weight = qkv.weight
        self.bias = qkv.bias
        self.linear_a_q = linear_a_q
        self.linear_b_q = linear_b_q
        self.linear_a_v = linear_a_v
        self.linear_b_v = linear_b_v
        self.dim = qkv.in_features
        self.w_identity = torch.eye(qkv.in_features)

    def forward(self, x):
        # qkv = self.qkv(x)  # B,N,N,3*org_C
        qkv = F.linear(x, self.weight, self.bias)
        new_q = self.linear_b_q(self.linear_a_q(x))
        new_v = self.linear_b_v(self.linear_a_v(x))
        qkv[:, :, :, : self.dim] += new_q
        qkv[:, :, :, -self.dim:] += new_v
        return qkv



class MaGRoad(pl.LightningModule):
    """This is the RelationFormer module that performs object detection"""

    def __init__(self, config):
        super().__init__()
        self.config = config

        assert config.SAM_VERSION in {'vit_b', 'vit_l', 'vit_h'}
        if config.SAM_VERSION == 'vit_b':
            ### SAM config (B)
            encoder_embed_dim=768
            encoder_depth=12
            encoder_num_heads=12
            encoder_global_attn_indexes=[2, 5, 8, 11]
            ###
        elif config.SAM_VERSION == 'vit_l':
            ### SAM config (L)
            encoder_embed_dim=1024
            encoder_depth=24
            encoder_num_heads=16
            encoder_global_attn_indexes=[5, 11, 17, 23]
            ###
        elif config.SAM_VERSION == 'vit_h':
            ### SAM config (H)
            encoder_embed_dim=1280
            encoder_depth=32
            encoder_num_heads=16
            encoder_global_attn_indexes=[7, 15, 23, 31]
            ###
            
        prompt_embed_dim = 256
        # SAM default is 1024
        image_size = config.PATCH_SIZE
        self.image_size = image_size
        vit_patch_size = 16
        image_embedding_size = image_size // vit_patch_size

        encoder_output_dim = prompt_embed_dim

        self.register_buffer("pixel_mean", torch.Tensor([123.675, 116.28, 103.53]).view(-1, 1, 1), False)
        self.register_buffer("pixel_std", torch.Tensor([58.395, 57.12, 57.375]).view(-1, 1, 1), False)

        if self.config.NO_SAM:
            ### im1k + mae pre-trained vitb
            self.image_encoder = vitdet.VITBEncoder(image_size=image_size, output_feature_dim=prompt_embed_dim)
            self.matched_param_names = self.image_encoder.matched_param_names
        else:
            ### SAM vitb
            self.image_encoder = ImageEncoderViT(
                depth=encoder_depth,
                embed_dim=encoder_embed_dim,
                img_size=image_size,
                mlp_ratio=4,
                norm_layer=partial(torch.nn.LayerNorm, eps=1e-6),
                num_heads=encoder_num_heads,
                patch_size=vit_patch_size,
                qkv_bias=True,
                use_rel_pos=True,
                global_attn_indexes=encoder_global_attn_indexes,
                window_size=14,
                out_chans=prompt_embed_dim
            )

        if self.config.USE_SAM_DECODER:
            # SAM DECODER
            # Not used, just produce null embeddings
            self.prompt_encoder=PromptEncoder(
                embed_dim=prompt_embed_dim,
                image_embedding_size=(image_embedding_size, image_embedding_size),
                input_image_size=(image_size, image_size),
                mask_in_chans=16,
            )
            for param in self.prompt_encoder.parameters():
                param.requires_grad = False
            self.mask_decoder=MaskDecoder(
                num_multimask_outputs=2, # keypoint, road
                transformer=TwoWayTransformer(
                    depth=2,
                    embedding_dim=prompt_embed_dim,
                    mlp_dim=2048,
                    num_heads=8,
                ),
                transformer_dim=prompt_embed_dim,
                iou_head_depth=3,
                iou_head_hidden_dim=256,
            )
        else:
            #### Naive decoder
            activation = nn.GELU
            self.map_decoder = nn.Sequential(
                nn.ConvTranspose2d(encoder_output_dim, 128, kernel_size=2, stride=2),
                LayerNorm2d(128),
                activation(),
                nn.ConvTranspose2d(128, 64, kernel_size=2, stride=2),
                activation(),
                nn.ConvTranspose2d(64, 32, kernel_size=2, stride=2),
                activation(),
                nn.ConvTranspose2d(32, 2, kernel_size=2, stride=2),
            )

        
        #### TOPONet
        self.bilinear_sampler = BilinearSampler(image_size=self.image_size)
        if config.TOPONET == 'transformer': # default
            self.topo_net = TopoNet(config, encoder_output_dim)
        elif config.TOPONET == 'maGTopoNet': # new
            self.topo_net = MaGTopoNet(config, encoder_output_dim,
                                       use_point_features=config.USE_POINT_FEATURES,
                                       use_path_features=config.USE_PATH_FEATURES,
                                       use_edge_bias=config.USE_EDGE_BIAS)


        #### LORA
        if config.ENCODER_LORA:
            r = self.config.LORA_RANK
            lora_layer_selection = None
            assert r > 0
            if lora_layer_selection:
                self.lora_layer_selection = lora_layer_selection
            else:
                self.lora_layer_selection = list(
                    range(len(self.image_encoder.blocks)))  # Only apply lora to the image encoder by default
            # create for storage, then we can init them or load weights
            self.w_As = []  # These are linear layers
            self.w_Bs = []

            # lets freeze first
            for param in self.image_encoder.parameters():
                param.requires_grad = False

            # Here, we do the surgery
            for t_layer_i, blk in enumerate(self.image_encoder.blocks):
                # If we only want few lora layer instead of all
                if t_layer_i not in self.lora_layer_selection:
                    continue
                w_qkv_linear = blk.attn.qkv
                dim = w_qkv_linear.in_features
                w_a_linear_q = nn.Linear(dim, r, bias=False)
                w_b_linear_q = nn.Linear(r, dim, bias=False)
                w_a_linear_v = nn.Linear(dim, r, bias=False)
                w_b_linear_v = nn.Linear(r, dim, bias=False)
                self.w_As.append(w_a_linear_q)
                self.w_Bs.append(w_b_linear_q)
                self.w_As.append(w_a_linear_v)
                self.w_Bs.append(w_b_linear_v)
                blk.attn.qkv = _LoRA_qkv(
                    w_qkv_linear,
                    w_a_linear_q,
                    w_b_linear_q,
                    w_a_linear_v,
                    w_b_linear_v,
                )
            # Init LoRA params
            for w_A in self.w_As:
                nn.init.kaiming_uniform_(w_A.weight, a=math.sqrt(5))
            for w_B in self.w_Bs:
                nn.init.zeros_(w_B.weight)

        #### Losses
        if self.config.FOCAL_LOSS: # None
            self.mask_criterion = partial(torchvision.ops.sigmoid_focal_loss, reduction='mean')
        else:
            # self.mask_criterion = torch.nn.BCEWithLogitsLoss(pos_weight=torch.tensor([10.0]))
            self.mask_criterion = torch.nn.BCEWithLogitsLoss()
        self.topo_criterion = torch.nn.BCEWithLogitsLoss(reduction='none')

        #### Metrics
        self.train_keypoint_iou = BinaryJaccardIndex(threshold=0.5)
        self.train_road_iou = BinaryJaccardIndex(threshold=0.5)
        self.train_topo_f1 = F1Score(task='binary', threshold=0.5, ignore_index=-1)
        self.val_keypoint_iou = BinaryJaccardIndex(threshold=0.5)
        self.val_road_iou = BinaryJaccardIndex(threshold=0.5)
        self.val_topo_f1 = F1Score(task='binary', threshold=0.5, ignore_index=-1)
        # testing only, not used in training
        self.keypoint_pr_curve = BinaryPrecisionRecallCurve(ignore_index=-1)
        self.road_pr_curve = BinaryPrecisionRecallCurve(ignore_index=-1)
        self.topo_pr_curve = BinaryPrecisionRecallCurve(ignore_index=-1)

        if self.config.NO_SAM:
            return
        with open(config.SAM_CKPT_PATH, "rb") as f:
            ckpt_state_dict = torch.load(f)

            ## Resize pos embeddings, if needed
            if image_size != 1024:
                new_state_dict = self.resize_sam_pos_embed(ckpt_state_dict, image_size, vit_patch_size, encoder_global_attn_indexes)
                ckpt_state_dict = new_state_dict
            
            matched_names = []
            mismatch_names = []
            state_dict_to_load = {}
            for k, v in self.named_parameters():
                if k in ckpt_state_dict and v.shape == ckpt_state_dict[k].shape:
                    matched_names.append(k)
                    state_dict_to_load[k] = ckpt_state_dict[k]
                else:
                    mismatch_names.append(k)
            print("###### Matched params ######")
            pprint.pprint(matched_names)
            print("###### Mismatched params ######")
            pprint.pprint(mismatch_names)

            self.matched_param_names = set(matched_names)
            self.load_state_dict(state_dict_to_load, strict=False)

    def transfer_batch_to_device(self, batch, device, dataloader_idx: int):
        # use non_blocking=True to transfer to GPU
        return {
            k: v.to(device, non_blocking=True) for k, v in batch.items()
        }

    def resize_sam_pos_embed(self, state_dict, image_size, vit_patch_size, encoder_global_attn_indexes):
        new_state_dict = {k : v for k, v in state_dict.items()}
        pos_embed = new_state_dict['image_encoder.pos_embed']
        token_size = int(image_size // vit_patch_size) # 512 // 16 = 32
        if pos_embed.shape[1] != token_size:
            # Copied from SAMed
            # resize pos embedding, which may sacrifice the performance, but I have no better idea
            pos_embed = pos_embed.permute(0, 3, 1, 2)  # [b, c, h, w] # [1, 768, 64, 64]
            pos_embed = F.interpolate(pos_embed, (token_size, token_size), mode='bilinear', align_corners=False) # [1, 768, 64, 64]
            pos_embed = pos_embed.permute(0, 2, 3, 1)  # [b, h, w, c]
            new_state_dict['image_encoder.pos_embed'] = pos_embed
            rel_pos_keys = [k for k in state_dict.keys() if 'rel_pos' in k]
            global_rel_pos_keys = [k for k in rel_pos_keys if any([str(i) in k for i in encoder_global_attn_indexes])]
            for k in global_rel_pos_keys:
                rel_pos_params = new_state_dict[k]
                h, w = rel_pos_params.shape
                rel_pos_params = rel_pos_params.unsqueeze(0).unsqueeze(0)
                rel_pos_params = F.interpolate(rel_pos_params, (token_size * 2 - 1, w), mode='bilinear', align_corners=False)
                new_state_dict[k] = rel_pos_params[0, 0, ...] # [2 * 64 - 1, 64] -> [2 * 32 - 1, 64]
        return new_state_dict

    
    def forward(self, rgb, graph_points, pairs, valid):
        # rgb: [B, H, W, C]
        # graph_points: [B, N_points, 2]
        # pairs: [B, N_samples, N_pairs, 2]
        # valid: [B, N_samples, N_pairs]

        x = rgb.permute(0, 3, 1, 2)
        # [B, C, H, W]
        x = (x - self.pixel_mean) / self.pixel_std
        # [B, D, h, w]
        image_embeddings = self.image_encoder(x)
        # mask_logits, mask_scores: [B, 2, H, W]
        if self.config.USE_SAM_DECODER:
            sparse_embeddings, dense_embeddings = self.prompt_encoder(
                points=None, boxes=None, masks=None
            )
            low_res_logits, iou_predictions = self.mask_decoder(
                image_embeddings=image_embeddings,
                image_pe=self.prompt_encoder.get_dense_pe(),
                sparse_prompt_embeddings=sparse_embeddings,
                dense_prompt_embeddings=dense_embeddings,
                multimask_output=True
            )
            mask_logits = F.interpolate(
                low_res_logits,
                (self.image_encoder.img_size, self.image_encoder.img_size),
                mode="bilinear",
                align_corners=False,
            )
            mask_scores = torch.sigmoid(mask_logits)
        else:
            mask_logits = self.map_decoder(image_embeddings)
            mask_scores = torch.sigmoid(mask_logits)
        
        ## Predicts local topology
        point_features = self.bilinear_sampler(image_embeddings, graph_points)
        # [B, N_sample, N_pair, 1]
        topo_logits, topo_scores = self.topo_net(graph_points, point_features, pairs, valid, mask_logits)
        
        
        # [B, H, W, 2]
        mask_logits = mask_logits.permute(0, 2, 3, 1)
        mask_scores = mask_scores.permute(0, 2, 3, 1)
        return mask_logits, mask_scores, topo_logits, topo_scores
    
    def infer_masks_and_img_features(self, rgb):
        # rgb: [B, H, W, C]
        # graph_points: [B, N_points, 2]
        # pairs: [B, N_samples, N_pairs, 2]
        # valid: [B, N_samples, N_pairs]

        x = rgb.permute(0, 3, 1, 2)
        # [B, C, H, W]
        x = (x - self.pixel_mean) / self.pixel_std
        # [B, D, h, w]
        image_embeddings = self.image_encoder(x)
        # mask_logits, mask_scores: [B, 2, H, W]
        if self.config.USE_SAM_DECODER:
            sparse_embeddings, dense_embeddings = self.prompt_encoder(
                points=None, boxes=None, masks=None
            )
            low_res_logits, iou_predictions = self.mask_decoder(
                image_embeddings=image_embeddings,
                image_pe=self.prompt_encoder.get_dense_pe(),
                sparse_prompt_embeddings=sparse_embeddings,
                dense_prompt_embeddings=dense_embeddings,
                multimask_output=True
            )
            mask_logits = F.interpolate(
                low_res_logits,
                (self.image_encoder.img_size, self.image_encoder.img_size),
                mode="bilinear",
                align_corners=False,
            )
            mask_scores = torch.sigmoid(mask_logits)
        else:
            mask_logits = self.map_decoder(image_embeddings)
            mask_scores = torch.sigmoid(mask_logits)
        
        # [B, H, W, 2]
        mask_scores = mask_scores.permute(0, 2, 3, 1)
        return image_embeddings, mask_logits, mask_scores
    

    def infer_toponet(self, image_embeddings, graph_points, pairs, valid, mask_logits):
        # image_embeddings: [B, D, h, w]
        # graph_points: [B, N_points, 2]
        # pairs: [B, N_samples, N_pairs, 2]
        # valid: [B, N_samples, N_pairs]

        ## Predicts local topology
        point_features = self.bilinear_sampler(image_embeddings, graph_points)
        # [B, N_sample, N_pair, 1]
        topo_logits, topo_scores = self.topo_net(graph_points, point_features, pairs, valid, mask_logits)
        return topo_scores


    def training_step(self, batch, batch_idx):
        # masks: [B, H, W]
        rgb, keypoint_mask, road_mask = batch['rgb'], batch['keypoint_mask'], batch['road_mask']
        graph_points, pairs, valid = batch['graph_points'], batch['pairs'], batch['valid']

        # [B, H, W, 2]
        mask_logits, mask_scores, topo_logits, topo_scores = self(rgb, graph_points, pairs, valid)

        gt_masks = torch.stack([keypoint_mask, road_mask], dim=3)
        # mask_loss = self.mask_criterion(mask_logits, gt_masks)
        bce_loss = self.mask_criterion(mask_logits, gt_masks)
        mask_loss = bce_loss

        topo_gt, topo_loss_mask = batch['connected'].to(torch.int32), valid.to(torch.float32)
        # [B, N_samples, N_pairs, 1]
        topo_loss = self.topo_criterion(topo_logits, topo_gt.unsqueeze(-1).to(torch.float32))

        #### DEBUG NAN
        for nan_index in torch.nonzero(torch.isnan(topo_loss[:, :, :, 0])):
            print('nan index: B, Sample, Pair')
            print(nan_index)
            import pdb
            pdb.set_trace()

        #### DEBUG NAN


        topo_loss *= topo_loss_mask.unsqueeze(-1)
        # topo_loss = torch.nansum(torch.nansum(topo_loss) / topo_loss_mask.sum())
        topo_loss = topo_loss.sum() / topo_loss_mask.sum()

        loss = mask_loss + topo_loss
        self.log('train_bce_loss', bce_loss, on_step=True, on_epoch=False, prog_bar=True, sync_dist=True)
        self.log('train_mask_loss', mask_loss, on_step=True, on_epoch=False, prog_bar=True, sync_dist=True)
        self.log('train_topo_loss', topo_loss, on_step=True, on_epoch=False, prog_bar=True, sync_dist=True)
        self.log('train_loss', loss, on_step=True, on_epoch=False, prog_bar=True, sync_dist=True)

        # Track metrics during training similar to validation
        self.train_keypoint_iou.update(mask_scores[..., 0], keypoint_mask)
        self.train_road_iou.update(mask_scores[..., 1], road_mask)
        
        valid = valid.to(torch.int32)
        topo_gt = (1 - valid) * -1 + valid * topo_gt
        self.train_topo_f1.update(topo_scores, topo_gt.unsqueeze(-1))

        # Log images during training
        if (batch_idx == 0 or batch_idx == 10) and self.current_epoch % 5 == 0:  # Visualize less frequently than validation
            max_viz_num = 4
            viz_rgb = rgb[:max_viz_num, :, :]
            viz_pred_keypoint = mask_scores[:max_viz_num, :, :, 0]
            viz_pred_road = mask_scores[:max_viz_num, :, :, 1]
            viz_gt_keypoint = keypoint_mask[:max_viz_num, ...]
            viz_gt_road = road_mask[:max_viz_num, ...]

            # Create image grids for TensorBoard
            for i in range(min(max_viz_num, rgb.size(0))):
                rgb_img = viz_rgb[i].cpu().numpy()  # [H, W, C] [0-255]
                gt_keypoint_img = viz_gt_keypoint[i].cpu().numpy()
                gt_road_img = viz_gt_road[i].cpu().numpy()
                pred_keypoint_img = viz_pred_keypoint[i].detach().cpu().numpy()
                pred_road_img = viz_pred_road[i].detach().cpu().numpy()
                
                # Create a figure with subplots
                fig, axs = plt.subplots(1, 5, figsize=(15, 3))
                
                # Plot the images
                axs[0].imshow(rgb_img.astype(np.uint8))
                axs[0].set_title('RGB')
                axs[0].axis('off')
                
                axs[1].imshow(gt_keypoint_img, cmap='gray')
                axs[1].set_title('GT Keypoint')
                axs[1].axis('off')
                
                axs[2].imshow(gt_road_img, cmap='gray')
                axs[2].set_title('GT Road')
                axs[2].axis('off')
                
                axs[3].imshow(pred_keypoint_img, cmap='gray')
                axs[3].set_title('Pred Keypoint')
                axs[3].axis('off')
                
                axs[4].imshow(pred_road_img, cmap='gray')
                axs[4].set_title('Pred Road')
                axs[4].axis('off')
                
                # Log the figure to TensorBoard
                if batch_idx == 10:
                    i += 4
                self.logger.experiment.add_figure(f'training_sample_{i}', fig, self.current_epoch)
                plt.close(fig)
                
        return loss

    def on_train_epoch_end(self): # not on_training_epoch_end
        keypoint_iou = self.train_keypoint_iou.compute()
        road_iou = self.train_road_iou.compute()
        topo_f1 = self.train_topo_f1.compute()
        self.log("train_keypoint_iou", keypoint_iou, sync_dist=True)
        self.log("train_road_iou", road_iou, sync_dist=True)
        self.log("train_topo_f1", topo_f1, sync_dist=True)
        self.train_keypoint_iou.reset()
        self.train_road_iou.reset()
        self.train_topo_f1.reset()  

    def validation_step(self, batch, batch_idx):
        # masks: [B, H, W]
        rgb, keypoint_mask, road_mask = batch['rgb'], batch['keypoint_mask'], batch['road_mask']
        graph_points, pairs, valid = batch['graph_points'], batch['pairs'], batch['valid']

        # masks: [B, H, W, 2] topo: [B, N_samples, N_pairs, 1]
        mask_logits, mask_scores, topo_logits, topo_scores = self(rgb, graph_points, pairs, valid)

        gt_masks = torch.stack([keypoint_mask, road_mask], dim=3)

        # mask_loss = self.mask_criterion(mask_logits, gt_masks)
        bce_loss = self.mask_criterion(mask_logits, gt_masks)
        mask_loss = bce_loss

        topo_gt, topo_loss_mask = batch['connected'].to(torch.int32), valid.to(torch.float32)
        # [B, N_samples, N_pairs, 1]
        topo_loss = self.topo_criterion(topo_logits, topo_gt.unsqueeze(-1).to(torch.float32))
        topo_loss *= topo_loss_mask.unsqueeze(-1)
        topo_loss = topo_loss.sum() / topo_loss_mask.sum()
        loss = mask_loss + topo_loss
        self.log('val_bce_loss', bce_loss, on_step=False, on_epoch=True, prog_bar=True, sync_dist=True)
        self.log('val_mask_loss', mask_loss, on_step=False, on_epoch=True, prog_bar=True, sync_dist=True) # about 0.6
        self.log('val_topo_loss', topo_loss, on_step=False, on_epoch=True, prog_bar=True, sync_dist=True) # about 0.8
        self.log('val_loss', loss, on_step=False, on_epoch=True, prog_bar=True, sync_dist=True)

        # Log images
        if batch_idx == 0:
            max_viz_num = 4
            viz_rgb = rgb[:max_viz_num, :, :]
            viz_pred_keypoint = mask_scores[:max_viz_num, :, :, 0]
            viz_pred_road = mask_scores[:max_viz_num, :, :, 1]
            viz_gt_keypoint = keypoint_mask[:max_viz_num, ...]
            viz_gt_road = road_mask[:max_viz_num, ...]

            # Create image grids for TensorBoard
            for i in range(min(max_viz_num, rgb.size(0))):
                rgb_img = viz_rgb[i].cpu().numpy() # [H, W, C] [0-255]
                gt_keypoint_img = viz_gt_keypoint[i].cpu().numpy()
                gt_road_img = viz_gt_road[i].cpu().numpy()
                pred_keypoint_img = viz_pred_keypoint[i].cpu().numpy()
                pred_road_img = viz_pred_road[i].cpu().numpy()
                
                # Create a figure with subplots
                fig, axs = plt.subplots(1, 5, figsize=(15, 3))
                
                # Plot the images
                axs[0].imshow(rgb_img.astype(np.uint8))
                axs[0].set_title('RGB')
                axs[0].axis('off')
                
                axs[1].imshow(gt_keypoint_img, cmap='gray')
                axs[1].set_title('GT Keypoint')
                axs[1].axis('off')
                
                axs[2].imshow(gt_road_img, cmap='gray')
                axs[2].set_title('GT Road')
                axs[2].axis('off')
                
                axs[3].imshow(pred_keypoint_img, cmap='gray')
                axs[3].set_title('Pred Keypoint')
                axs[3].axis('off')
                
                axs[4].imshow(pred_road_img, cmap='gray')
                axs[4].set_title('Pred Road')
                axs[4].axis('off')
                
                # Log the figure to TensorBoard
                self.logger.experiment.add_figure(f'validation_sample_{i}', fig, self.current_epoch)
                plt.close(fig)

        self.val_keypoint_iou.update(mask_scores[..., 0], keypoint_mask)
        self.val_road_iou.update(mask_scores[..., 1], road_mask)
        
        valid = valid.to(torch.int32)
        topo_gt = (1 - valid) * -1 + valid * topo_gt
        self.val_topo_f1.update(topo_scores, topo_gt.unsqueeze(-1))

    def on_validation_epoch_end(self):
        keypoint_iou = self.val_keypoint_iou.compute()
        road_iou = self.val_road_iou.compute()
        topo_f1 = self.val_topo_f1.compute()
        self.log("val_keypoint_iou", keypoint_iou, sync_dist=True)
        self.log("val_road_iou", road_iou, sync_dist=True)
        self.log("val_topo_f1", topo_f1, sync_dist=True)
        self.val_keypoint_iou.reset()
        self.val_road_iou.reset()
        self.val_topo_f1.reset()

    def test_step(self, batch, batch_idx):
        # masks: [B, H, W]
        rgb, keypoint_mask, road_mask = batch['rgb'], batch['keypoint_mask'], batch['road_mask']
        graph_points, pairs, valid = batch['graph_points'], batch['pairs'], batch['valid']

        # masks: [B, H, W, 2] topo: [B, N_samples, N_pairs, 1]
        mask_logits, mask_scores, topo_logits, topo_scores = self(rgb, graph_points, pairs, valid)

        topo_gt, topo_loss_mask = batch['connected'].to(torch.int32), valid.to(torch.float32)
        self.keypoint_pr_curve.update(mask_scores[..., 0], keypoint_mask.to(torch.int32))
        self.road_pr_curve.update(mask_scores[..., 1], road_mask.to(torch.int32))

        # Save predictions and GT to disk for offline threshold computation
        try:
            save_dir = os.path.join(os.path.dirname(os.path.abspath(__file__)), 'compute_threshold', 'dump', f'batch_{batch_idx:06d}')
            os.makedirs(save_dir, exist_ok=True)
            # predictions to uint8 [0,255]
            kp_pred_u8 = torch.round(mask_scores[..., 0].detach().clamp(0, 1) * 255).to(torch.uint8).cpu().numpy()
            road_pred_u8 = torch.round(mask_scores[..., 1].detach().clamp(0, 1) * 255).to(torch.uint8).cpu().numpy()
            # ground truth to uint8 {0,1}
            kp_gt_u8 = (keypoint_mask > 0.5).to(torch.uint8).cpu().numpy()
            road_gt_u8 = (road_mask > 0.5).to(torch.uint8).cpu().numpy()
            # save arrays
            np.save(os.path.join(save_dir, 'pred_kp_mask.npy'), kp_pred_u8)
            np.save(os.path.join(save_dir, 'gt_kp_mask.npy'), kp_gt_u8)
            np.save(os.path.join(save_dir, 'pred_road_mask.npy'), road_pred_u8)
            np.save(os.path.join(save_dir, 'gt_road_mask.npy'), road_gt_u8)
            del kp_pred_u8, road_pred_u8, kp_gt_u8, road_gt_u8
        except Exception as e:
            print(f'[test_step] Error saving masks for batch {batch_idx}: {e}')
        
        valid = valid.to(torch.int32)
        topo_gt = (1 - valid) * -1 + valid * topo_gt
        self.topo_pr_curve.update(topo_scores, topo_gt.unsqueeze(-1).to(torch.int32))

    def on_test_end(self):
        def find_best_threshold(pr_curve_metric, category):
            print(f'======= {category} ======')   
            try:
                precision, recall, thresholds = pr_curve_metric.compute()

                # Align lengths if needed (torchmetrics often returns len(thresholds) == len(pr) - 1)
                if thresholds is not None and thresholds.numel() == precision.numel() - 1:
                    precision = precision[:-1]
                    recall = recall[:-1]

                # Compute F1 robustly: handle 0/0 and NaNs
                denom = precision + recall
                valid = torch.isfinite(precision) & torch.isfinite(recall) & (denom > 0)
                if valid.sum() == 0:
                    print(f'No valid precision/recall points for {category}. Skipping.')
                    return

                f1_scores = torch.zeros_like(precision)
                f1_scores[valid] = 2 * (precision[valid] * recall[valid]) / (denom[valid] + 1e-6)

                best_threshold_index = torch.argmax(f1_scores)
                best_threshold = thresholds[best_threshold_index] if thresholds is not None else torch.tensor(float('nan'))
                best_precision = precision[best_threshold_index]
                best_recall = recall[best_threshold_index]
                best_f1 = f1_scores[best_threshold_index]

                print(f'Best threshold {best_threshold:.6f}, P={best_precision:.6f} R={best_recall:.6f} F1={best_f1:.6f}')
                if self.global_rank == 0 and hasattr(self.logger, 'experiment'):
                    # Log as text for visibility
                    self.logger.experiment.add_text(
                        f'{category}_best_threshold',
                        f'Best threshold {float(best_threshold):.6f}, P={float(best_precision):.6f} R={float(best_recall):.6f} F1={float(best_f1):.6f}'
                    )

                # Plot Precision/Recall/F1 vs Threshold and save under TensorBoard version directory
                try:
                    # Determine output directory alongside TensorBoard version folder
                    out_dir = getattr(self.logger, 'log_dir', None)
                    if out_dir is None:
                        # Fallback to current working directory if logger doesn't expose log_dir
                        out_dir = os.path.join(os.getcwd(), 'lightning_logs/test_curves')
                    os.makedirs(out_dir, exist_ok=True)

                    # Convert to CPU numpy for plotting
                    th_np = thresholds.detach().cpu().numpy() if thresholds is not None else None
                    p_np = precision.detach().cpu().numpy()
                    r_np = recall.detach().cpu().numpy()
                    f1_np = f1_scores.detach().cpu().numpy()

                    # Filter thresholds between 0.1 and 0.95
                    if th_np is not None:
                        valid_indices = np.where((th_np >= 0.1) & (th_np <= 0.95))[0]
                        th_np = th_np[valid_indices]
                        p_np = p_np[valid_indices]
                        r_np = r_np[valid_indices]
                        f1_np = f1_np[valid_indices]

                    if th_np is not None and th_np.size > 0 and p_np.size == th_np.size:
                        fig, ax = plt.subplots(1, 1, figsize=(7, 4.5))
                        ax.plot(th_np, p_np, label='Precision')
                        ax.plot(th_np, r_np, label='Recall')
                        ax.plot(th_np, f1_np, label='F1')
                        ax.set_xlabel('Threshold')
                        ax.set_ylabel('Score')
                        ax.set_title(f'{category} P/R/F vs Threshold')
                        ax.grid(True, linestyle='--', alpha=0.4)
                        ax.legend()
                        save_path = os.path.join(out_dir, f'{category}_PRF_curve.png')
                        fig.savefig(save_path, dpi=150, bbox_inches='tight')
                        plt.close(fig)
                    else:
                        print(f'Skip plotting {category}: thresholds not available or mismatched shapes.')
                except Exception as plot_e:
                    print(f'Error while plotting PRF curve for {category}: {plot_e}')
            except Exception as e:
                print(f'Error while computing best threshold for {category}: {e}')
            finally:
                # Ensure metric state is cleared regardless of outcome
                pr_curve_metric.reset()
        
        print('======= Finding best thresholds ======')
        find_best_threshold(self.keypoint_pr_curve, 'keypoint')
        find_best_threshold(self.road_pr_curve, 'road')
        find_best_threshold(self.topo_pr_curve, 'topo')


    def configure_optimizers(self):
        param_dicts = []

        if not self.config.FREEZE_ENCODER and not self.config.ENCODER_LORA:
            encoder_params = {
                'params': [p for k, p in self.image_encoder.named_parameters() if 'image_encoder.'+k in self.matched_param_names],
                'lr': self.config.BASE_LR * self.config.ENCODER_LR_FACTOR, # 0.001 * 0.1
            }
            param_dicts.append(encoder_params)
        if self.config.ENCODER_LORA:
            # LoRA params only
            encoder_params = {
                'params': [p for k, p in self.image_encoder.named_parameters() if 'qkv.linear_' in k],
                'lr': self.config.BASE_LR,
            }
            param_dicts.append(encoder_params)
        
        if self.config.USE_SAM_DECODER:
            matched_decoder_params = {
                'params': [p for k, p in self.mask_decoder.named_parameters() if 'mask_decoder.'+k in self.matched_param_names],
                'lr': self.config.BASE_LR * 0.1
            }
            fresh_decoder_params = {
                'params': [p for k, p in self.mask_decoder.named_parameters() if 'mask_decoder.'+k not in self.matched_param_names],
                'lr': self.config.BASE_LR
            }
            decoder_params = [matched_decoder_params, fresh_decoder_params]
        else:
            decoder_params = [{
                'params': [p for p in self.map_decoder.parameters()],
                'lr': self.config.BASE_LR # 0.001
            }]
        param_dicts += decoder_params

        topo_net_params = [{
            'params': [p for p in self.topo_net.parameters()],
            'lr': self.config.BASE_LR # 0.001
        }]
        param_dicts += topo_net_params
        for i, param_dict in enumerate(param_dicts):
            param_num = sum([int(p.numel()) for p in param_dict['params']])
            print(f'optim param dict {i} params num: {param_num}')

        base_lr = self.config.BASE_LR
        optimizer_name = self.config.get("OPTIMIZER", "AdamW").lower()
        weight_decay = self.config.get("WEIGHT_DECAY", 0.01)

        if optimizer_name == "adamw":
             print(f"Using AdamW optimizer with weight_decay={weight_decay}")
             optimizer = torch.optim.AdamW(param_dicts, lr=base_lr, weight_decay=weight_decay)
        elif optimizer_name == "adam":
             print("Using Adam optimizer")
             optimizer = torch.optim.Adam(param_dicts, lr=base_lr) # Adam usually doesn't use weight_decay arg directly
        else:
             raise ValueError(f"Unsupported optimizer: {optimizer_name}")
        # --- Scheduler ---
        max_epochs = self.trainer.max_epochs
        step_lr = torch.optim.lr_scheduler.MultiStepLR(optimizer, milestones=[int(max_epochs * 0.8), ], gamma=0.1)
        return {'optimizer': optimizer, 'lr_scheduler': step_lr}

