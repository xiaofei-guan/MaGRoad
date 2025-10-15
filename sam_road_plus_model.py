import torch
import torch.nn.functional as F
from torch import nn
import matplotlib.pyplot as plt
import math
import copy
import os
from functools import partial
from torchmetrics.classification import BinaryJaccardIndex, F1Score, BinaryPrecisionRecallCurve
# import lightning.pytorch as pl
import pytorch_lightning as pl
from sam.segment_anything.modeling.image_encoder import ImageEncoderViT
from sam.segment_anything.modeling.mask_decoder import MaskDecoder
from sam.segment_anything.modeling.prompt_encoder import PromptEncoder
from sam.segment_anything.modeling.transformer import TwoWayTransformer
from sam.segment_anything.modeling.common import LayerNorm2d
# import wandb
import pprint
import torchvision
import numpy as np
import vitdet
from typing import Optional, List


def find_highest_mask_point(x, y, mask, device='cuda'):
    H, W, D = mask.shape
    x = torch.clamp(x, 0, W)
    y = torch.clamp(y, 0, D)
    x = int(x)
    y = int(y)
    radius = torch.tensor(2)
    # limit coordinate range
    x_min = max(0, x - radius)
    x_max = min(W, x + radius)
    y_min = max(0, y - radius)
    y_max = min(D, y + radius)

    mask_region = mask[:, x_min:x_max, y_min:y_max].to(device)
    
    x_coords = torch.arange(x_min, x_max, device=device).view(-1, 1).expand(x_max - x_min, y_max - y_min)
    y_coords = torch.arange(y_min, y_max, device=device).view(1, -1).expand(x_max - x_min, y_max - y_min)
    
    distances = torch.sqrt((x_coords - x) ** 2 + (y_coords - y) ** 2)

    within_radius = (distances <= radius).to(device)
    mask_scores = mask_region[1] * within_radius + mask_region[0] * within_radius
    
    if mask_scores.numel() > 0:
        mask_max = torch.max(mask_scores)
        max_pos = torch.nonzero(mask_scores == mask_max)
        if len(max_pos) > 0:
            x_final = max_pos[0][0] + x_min
            y_final = max_pos[0][1] + y_min
        else:
            x_final, y_final = x, y
    else:
        x_final, y_final = x, y
        
    return x_final, y_final

def extract_point(x1,y1,x2,y2,image,num_points):
    H, W = image.shape[-2:] 
    x_values = torch.linspace(0, 1, steps=num_points).unsqueeze(0).unsqueeze(0).to(image.device)
    y_values = torch.linspace(0, 1, steps=num_points).unsqueeze(0).unsqueeze(0).to(image.device)#uniform sampling

    x_interp = x1.unsqueeze(-1) + (x2 - x1).unsqueeze(-1) * x_values
    y_interp = y1.unsqueeze(-1) + (y2 - y1).unsqueeze(-1) * y_values
    
    x_interp = torch.clamp(x_interp.long(), min=0, max=W-1)
    y_interp = torch.clamp(y_interp.long(), min=0, max=H-1)
    
    x_plus_1 = torch.clamp(x_interp + 1, max=W-1)
    y_plus_1 = torch.clamp(y_interp + 1, max=H-1)
    
    x_final = torch.cat([x_interp,x_interp,x_plus_1], dim=-1)
    y_final = torch.cat([y_interp,y_plus_1,y_interp], dim=-1)
    
    return(x_final,y_final)# sample points

def extendline(points1, points2, image):
    """Using linear interpolation to get pixels between two points in batch."""
    B, N, _ = points1.shape  # B: batch size, N: number of point pairs
    H, W = image.shape[-2:]  
    height, width = H,W
    extend_length=8 #extend length
    batch_A = points1
    batch_B = points2
    # direction vector
    directions = batch_B - batch_A  # (N, 2)
    
    lengths = torch.norm(directions, dim=2, keepdim=True)  # (N, 1)
    lengths = lengths.masked_fill(lengths == 0, 1e-8)
    directions_norm = directions / lengths  # (N, 2)
    
    extended_A = batch_A - directions_norm * extend_length  # (N, 2)
    extended_B = batch_B + directions_norm * extend_length  # (N, 2)
      # Round the coordinates to integers
    extended_A = torch.round(extended_A).long()
    extended_B = torch.round(extended_B).long()
    
    extended_A[:, 0] = extended_A[:, 0].clamp(0, width - 1)
    extended_A[:, 1] = extended_A[:, 1].clamp(0, height - 1)
    extended_B[:, 0] = extended_B[:, 0].clamp(0, width - 1)
    extended_B[:, 1] = extended_B[:, 1].clamp(0, height - 1)

    extend_x1,extend_y1 = extended_A[...,0],extended_A[...,1]
    extend_x2,extend_y2 = extended_B[...,0],extended_B[...,1]

    x1, y1 = points1[..., 0], points1[..., 1]
    x2, y2 = points2[..., 0], points2[..., 1]
    
    x_final_1,y_final_1  = extract_point(extend_x1,extend_y1,x1,y1,image,num_points =15 )#sample extend point one side
    x_final,y_final  = extract_point(x1,y1,x2,y2,image,num_points =20 ) #sample between point
    x_final_2,y_final_2  = extract_point(extend_x2,extend_y2,x2,y2,image,num_points=15 )#sample extend point other side

    features1 = image[np.arange(B)[:, None, None], x_final_1,  y_final_1]
    features = image[np.arange(B)[:, None, None], x_final,  y_final]
    features2 = image[np.arange(B)[:, None, None], x_final_2,  y_final_2]#extract mask feature
    features = torch.concat([features1,features,features2], dim=2)

    return features


class BilinearSamplerInfer(nn.Module):
    def __init__(self, config):
        super(BilinearSamplerInfer, self).__init__()
        self.config = config

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
        sample_points = (sample_points / self.config.PATCH_SIZE) * 2.0 - 1.0
        
        # sample_points from [B, N_points, 2] to [B, N_points, 1, 2] for grid_sample
        sample_points = sample_points.unsqueeze(2)
        # Use grid_sample for bilinear sampling. Align_corners set to False to use -1 to 1 grid space.
        # [B, D, N_points, 1]
        sampled_features = F.grid_sample(feature_maps, sample_points, mode='bilinear', align_corners=False)
        
        # sampled_features is [B, N_points, D]
        sampled_features = sampled_features.squeeze(dim=-1).permute(0, 2, 1)
        return sampled_features


class BilinearSampler(nn.Module):
    def __init__(self, config):
        super(BilinearSampler, self).__init__()
        self.config = config

    def forward(self, feature_maps, sample_points,mask_scores):
        """
        Args:
            feature_maps (Tensor): The input feature tensor of shape [B, D, H, W].
            sample_points (Tensor): The 2D sample points of shape [B, N_points, 2],
                                    each point in the range [-1, 1], format (x, y).
        Returns:
            Tensor: Sampled feature vectors of shape [B, N_points, D].
        """
        B, D, H, W = feature_maps.shape#[16, 256, 32, 32]
        batch_size, N_points, _ = sample_points.shape

        target_new_points = torch.zeros_like(sample_points).cuda()
        for batch_index in range(batch_size):
            for point_index in range(N_points):
                x, y = sample_points[batch_index, point_index]
                if (x.item(), y.item()) == (0, 0):
                    target_new_points[batch_index, point_index] = torch.tensor([x, y])
                else:
                    current_mask = mask_scores[batch_index]#torch.Size([16, 3, 512, 512]
                    x_new,y_new = find_highest_mask_point(x, y, current_mask)
                    target_new_points[batch_index, point_index] = torch.tensor([x_new, y_new], dtype=torch.float32)
        point = target_new_points 
        
        target_new_points = (target_new_points / self.config.PATCH_SIZE) * 2.0 - 1.0    
        target_new_points = target_new_points.unsqueeze(2)  
        sampled_features = F.grid_sample(feature_maps, target_new_points, mode='bilinear', align_corners=False)
        sampled_features_target = sampled_features.squeeze(dim=-1).permute(0, 2, 1)#get target_features   
        
        sample_points = (sample_points / self.config.PATCH_SIZE) * 2.0 - 1.0   
        sample_points = sample_points.unsqueeze(2)
        sampled_features_o = F.grid_sample(feature_maps, sample_points, mode='bilinear', align_corners=False)       
        sampled_features_source = sampled_features_o.squeeze(dim=-1).permute(0, 2, 1)#get source_features

        return sampled_features_target,point,sampled_features_source


class DiceLoss(nn.Module):
    """Dice Loss for segmentation tasks."""
    def __init__(self, smooth=0.6):
        super(DiceLoss, self).__init__()
        self.smooth = smooth
        
    def forward(self, pred, target):
        """
        Args:
            pred: logits tensor of shape [B, H, W, C] or [B, C, H, W]
            target: ground truth tensor of shape [B, H, W, C] or [B, C, H, W]
        Returns:
            Dice loss (scalar)
        """
        pred = torch.sigmoid(pred)
        intersection = (pred * target).sum(dim=(1, 2, 3))
        union = pred.sum(dim=(1, 2, 3)) + target.sum(dim=(1, 2, 3))
        dice = (2. * intersection + self.smooth) / (union + self.smooth)
        return 1 - dice.mean()


class TopoNet(nn.Module):
    def __init__(self, config, feature_dim):
        super(TopoNet, self).__init__()
        self.config = config
        self.hidden_dim = 128
        self.heads = 4
        self.num_attn_layers = 3
        self.feature_proj = nn.Linear(feature_dim, self.hidden_dim)
        self.pair_proj = nn.Linear(2*self.hidden_dim+ 152 , self.hidden_dim)
        # Create Transformer Encoder Layer
        encoder_layer = nn.TransformerEncoderLayer(
            d_model=self.hidden_dim ,
            nhead=self.heads,
            dim_feedforward=self.hidden_dim ,
            dropout=0.1,
            activation='relu',
            batch_first=True  # Input format is [batch size, sequence length, features]
        )
        
        # Stack the Transformer Encoder Layers
        if self.config.TOPONET_VERSION != 'no_transformer':
            self.transformer_encoder = nn.TransformerEncoder(encoder_layer, num_layers=self.num_attn_layers)
        self.output_proj = nn.Linear(self.hidden_dim, 1)

    def forward(self, points, point_features,graph_points,point_features_o, pairs, pairs_valid,mask_scores):
        # points: [B, N_points, 2]
        # point_features: [B, N_points, D]
        # pairs: [B, N_samples, N_pairs, 2]
        # pairs_valid: [B, N_samples, N_pairs]
        # mask scores:[B,3,512,512]
        B,_,H,W = mask_scores.shape
        point_features = F.relu(self.feature_proj(point_features))
        point_features_o = F.relu(self.feature_proj(point_features_o))
        # gathers pairs
        batch_size, n_samples, n_pairs, _ = pairs.shape
        pairs = pairs.view(batch_size, -1, 2)
        batch_indices = torch.arange(batch_size).view(-1, 1).expand(-1, n_samples * n_pairs)
        # Use advanced indexing to fetch the corresponding feature vectors
        # [B, N_samples * N_pairs, D]
        src_features = point_features_o[batch_indices, pairs[:, :, 0]]
        #src_features = point_features[batch_indices, pairs[:, :, 0]]
        tgt_features = point_features[batch_indices, pairs[:, :, 1]]
        # [B, N_samples * N_pairs, 2]
        src_points = graph_points[batch_indices, pairs[:, :, 0]].float()
        #src_points = points[batch_indices, pairs[:, :, 0]]
        tgt_points = points[batch_indices, pairs[:, :, 1]].float()
        _,N,_ = tgt_points.shape
        mask_road_dim = mask_scores[:, 1, :, :] 
        line_features = extendline(src_points, tgt_points, mask_road_dim)#][B,N,64]
        #line_features = F.relu(self.point_proj(line_features))
        offset_x = tgt_points - src_points
        ##ablation study
        # [B, N_samples * N_pairs, 2D + 2]
        if self.config.TOPONET_VERSION == 'no_tgt_features':
            pair_features = torch.concat([src_features, torch.zeros_like(tgt_features), offset_x], dim=2)
        if self.config.TOPONET_VERSION == 'no_offset':
            pair_features = torch.concat([src_features, tgt_features, torch.zeros_like(offset_x)], dim=2)
        else:
            #pair_features = torch.concat([line_features,offset_x], dim=2)
            pair_features=torch.concat([src_features, tgt_features,line_features,offset_x], dim=2)
        # [B, N_samples * N_pairs, D]256+122
        pair_features = F.relu(self.pair_proj(pair_features))
        #pair_features = torch.concat([pair_features,line_features,offset_x], dim=2)     
        # attn applies within each local graph sample
        pair_features = pair_features.view(batch_size * n_samples, n_pairs, -1)
        # valid->not a padding
        pairs_valid = pairs_valid.view(batch_size * n_samples, n_pairs)
        # [B * N_samples, 1]
        #### flips mask for all-invalid pairs to prevent NaN
        all_invalid_pair_mask = torch.eq(torch.sum(pairs_valid, dim=-1), 0).unsqueeze(-1)
        pairs_valid = torch.logical_or(pairs_valid, all_invalid_pair_mask)
        padding_mask = ~pairs_valid  
        ## ablation study
        if self.config.TOPONET_VERSION != 'no_transformer':
            pair_features = self.transformer_encoder(pair_features, src_key_padding_mask=padding_mask)
        ## Seems like at inference time, the returned n_pairs heres might be less - it's the
        # max num of valid pairs across all samples in the batch
        _, n_pairs, _ = pair_features.shape
        pair_features = pair_features.view(batch_size, n_samples, n_pairs, -1)
        # [B, N_samples, N_pairs, 1]
        logits = self.output_proj(pair_features)
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

class SAMRoadplus(pl.LightningModule):
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
                num_multimask_outputs=3, # keypoint, road
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
                nn.ConvTranspose2d(32,  2, kernel_size=2, stride=2),
            )
        #### TOPONet
        if config.INFER == 'infer':
            self.bilinear_sampler = BilinearSamplerInfer(config)
        else:
            self.bilinear_sampler = BilinearSampler(config)
        self.topo_net = TopoNet(config, 256)
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
        if self.config.FOCAL_LOSS:
            self.mask_criterion = partial(torchvision.ops.sigmoid_focal_loss, reduction='mean')
        else:
            # Support configurable BCE pos_weight
            pos_weight_val = self.config.get('BCE_POS_WEIGHT', 1.0)
            if pos_weight_val != 1.0:
                print(f"Using BCEWithLogitsLoss with pos_weight={pos_weight_val}")
                self.mask_criterion = torch.nn.BCEWithLogitsLoss(pos_weight=torch.tensor([pos_weight_val]))
            else:
                self.mask_criterion = torch.nn.BCEWithLogitsLoss()
        
        # Add Dice Loss
        dice_smooth = self.config.get('DICE_SMOOTH', 0.6)
        print(f"Using Dice Loss with smooth={dice_smooth}")
        self.mask_dice_criterion = DiceLoss(smooth=dice_smooth)
        
        self.topo_criterion = torch.nn.BCEWithLogitsLoss(reduction='none')
        
        #### Metrics
        # Training metrics
        self.train_keypoint_iou = BinaryJaccardIndex(threshold=0.5)
        self.train_road_iou = BinaryJaccardIndex(threshold=0.5)
        self.train_topo_f1 = F1Score(task='binary', threshold=0.5, ignore_index=-1)
        # Validation metrics
        self.val_keypoint_iou = BinaryJaccardIndex(threshold=0.5)
        self.val_road_iou = BinaryJaccardIndex(threshold=0.5)
        self.val_topo_f1 = F1Score(task='binary', threshold=0.5, ignore_index=-1)
        # Testing only, not used in training
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
    def resize_sam_pos_embed(self, state_dict, image_size, vit_patch_size, encoder_global_attn_indexes):
        new_state_dict = {k : v for k, v in state_dict.items()}
        pos_embed = new_state_dict['image_encoder.pos_embed']
        token_size = int(image_size // vit_patch_size)
        if pos_embed.shape[1] != token_size:
            # Copied from SAMed
            # resize pos embedding, which may sacrifice the performance, but I have no better idea
            pos_embed = pos_embed.permute(0, 3, 1, 2)  # [b, c, h, w]
            pos_embed = F.interpolate(pos_embed, (token_size, token_size), mode='bilinear', align_corners=False)
            pos_embed = pos_embed.permute(0, 2, 3, 1)  # [b, h, w, c]
            new_state_dict['image_encoder.pos_embed'] = pos_embed
            rel_pos_keys = [k for k in state_dict.keys() if 'rel_pos' in k]
            global_rel_pos_keys = [k for k in rel_pos_keys if any([str(i) in k for i in encoder_global_attn_indexes])]
            for k in global_rel_pos_keys:
                rel_pos_params = new_state_dict[k]
                h, w = rel_pos_params.shape
                rel_pos_params = rel_pos_params.unsqueeze(0).unsqueeze(0)
                rel_pos_params = F.interpolate(rel_pos_params, (token_size * 2 - 1, w), mode='bilinear', align_corners=False)
                new_state_dict[k] = rel_pos_params[0, 0, ...]
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
        image_embeddings = self.image_encoder(x)#[16, 256, 32, 32]
        #print(image_embeddings.shape)
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
            mask_scores = torch.sigmoid(mask_logits)#torch.Size([16, 3, 512, 512])
        point_features,newpoint,point_features_o = self.bilinear_sampler(image_embeddings, graph_points,mask_scores)
       # point_features = self.bilinear_sampler(image_embeddings, graph_points,mask_logits)
        # [B, N_sample, N_pair, 1]
       # topo_logits, topo_scores = self.topo_net(graph_points, point_features, pairs, valid) 
        topo_logits, topo_scores = self.topo_net(newpoint, point_features,graph_points,point_features_o, pairs, valid,mask_scores)
        # [B, H, W, 3]
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
        mask_scores = torch.sigmoid(mask_logits)
        point_features = self.bilinear_sampler(image_embeddings, graph_points)
        # [B, N_sample, N_pair, 1]
        topo_logits, topo_scores = self.topo_net(graph_points, point_features, graph_points, point_features, pairs, valid,mask_scores)
        return topo_scores

    def training_step(self, batch, batch_idx):
        # masks: [B, H, W]
        rgb, keypoint_mask, road_mask  = batch['rgb'], batch['keypoint_mask'], batch['road_mask']
        graph_points, pairs, valid = batch['graph_points'], batch['pairs'], batch['valid']
        
        # [B, H, W, 2]
        mask_logits, mask_scores, topo_logits, topo_scores = self(rgb, graph_points, pairs, valid)
        
        gt_masks = torch.stack([keypoint_mask, road_mask], dim=3)
        # Compute BCE loss and Dice loss
        bce_loss = self.mask_criterion(mask_logits, gt_masks)
        dice_loss = self.mask_dice_criterion(mask_logits, gt_masks)
        mask_loss = bce_loss + dice_loss
        
        topo_gt, topo_loss_mask = batch['connected'].to(torch.int32), valid.to(torch.float32)
        # [B, N_samples, N_pairs, 1]
        topo_loss = self.topo_criterion(topo_logits, topo_gt.unsqueeze(-1).to(torch.float32))
        topo_loss *= topo_loss_mask.unsqueeze(-1)
        topo_loss = torch.nansum(torch.nansum(topo_loss) / topo_loss_mask.sum())

        loss = mask_loss + topo_loss

        if torch.any(torch.isnan(loss)):
            print("NaN detected in loss. Using default loss value.")
            loss = torch.tensor(0.0, device=loss.device)
        
        # Logging
        self.log('train_loss', loss, on_step=True, on_epoch=True, prog_bar=True, sync_dist=True)
        self.log('train_mask_loss', mask_loss, on_step=True, on_epoch=True, prog_bar=False, sync_dist=True)
        self.log('train_bce_loss', bce_loss, on_step=True, on_epoch=True, prog_bar=False, sync_dist=True)
        self.log('train_dice_loss', dice_loss, on_step=True, on_epoch=True, prog_bar=False, sync_dist=True)
        self.log('train_topo_loss', topo_loss, on_step=True, on_epoch=True, prog_bar=True, sync_dist=True)
        
        # Update metrics
        self.train_keypoint_iou.update(mask_scores[..., 0], keypoint_mask)
        self.train_road_iou.update(mask_scores[..., 1], road_mask)
        valid = valid.to(torch.int32)
        topo_gt = (1 - valid) * -1 + valid * topo_gt
        self.train_topo_f1.update(topo_scores, topo_gt.unsqueeze(-1))
        
        # Log images periodically
        if batch_idx == 0 and self.current_epoch % 5 == 0:
            self._log_images(batch, mask_scores, prefix="train")
        
        return loss

    def validation_step(self, batch, batch_idx):
        # masks: [B, H, W]
        rgb, keypoint_mask, road_mask = batch['rgb'], batch['keypoint_mask'], batch['road_mask']
        graph_points, pairs, valid = batch['graph_points'], batch['pairs'], batch['valid']
        
        # masks: [B, H, W, 2] topo: [B, N_samples, N_pairs, 1]
        mask_logits, mask_scores, topo_logits, topo_scores = self(rgb, graph_points, pairs, valid)
        
        gt_masks = torch.stack([keypoint_mask, road_mask], dim=3)
        # Compute BCE loss and Dice loss
        bce_loss = self.mask_criterion(mask_logits, gt_masks)
        dice_loss = self.mask_dice_criterion(mask_logits, gt_masks)
        mask_loss = bce_loss + dice_loss
        
        topo_gt, topo_loss_mask = batch['connected'].to(torch.int32), valid.to(torch.float32)
        # [B, N_samples, N_pairs, 1]
        topo_loss = self.topo_criterion(topo_logits, topo_gt.unsqueeze(-1).to(torch.float32))
        topo_loss *= topo_loss_mask.unsqueeze(-1)
        topo_loss = topo_loss.sum() / topo_loss_mask.sum()
        loss = mask_loss + topo_loss
        
        # Logging
        self.log('val_loss', loss, on_step=False, on_epoch=True, prog_bar=True, sync_dist=True)
        self.log('val_mask_loss', mask_loss, on_step=False, on_epoch=True, prog_bar=False, sync_dist=True)
        self.log('val_bce_loss', bce_loss, on_step=False, on_epoch=True, prog_bar=False, sync_dist=True)
        self.log('val_dice_loss', dice_loss, on_step=False, on_epoch=True, prog_bar=False, sync_dist=True)
        self.log('val_topo_loss', topo_loss, on_step=False, on_epoch=True, prog_bar=True, sync_dist=True)
        
        # Update metrics
        self.val_keypoint_iou.update(mask_scores[..., 0], keypoint_mask)
        self.val_road_iou.update(mask_scores[..., 1], road_mask)
        valid = valid.to(torch.int32)
        topo_gt = (1 - valid) * -1 + valid * topo_gt
        self.val_topo_f1.update(topo_scores, topo_gt.unsqueeze(-1))
        
        # Log first batch of validation images
        if batch_idx == 0:
            self._log_images(batch, mask_scores, prefix="val")
    
    def on_train_epoch_end(self):
        """Compute and log training metrics at the end of each epoch."""
        keypoint_iou = self.train_keypoint_iou.compute()
        road_iou = self.train_road_iou.compute()
        topo_f1 = self.train_topo_f1.compute()
        self.log("train_keypoint_iou", keypoint_iou, sync_dist=True)
        self.log("train_road_iou", road_iou, sync_dist=True)
        self.log("train_topo_f1", topo_f1, sync_dist=True)
        self.train_keypoint_iou.reset()
        self.train_road_iou.reset()
        self.train_topo_f1.reset()
        
    def on_validation_epoch_end(self):
        """Compute and log validation metrics at the end of each epoch."""
        keypoint_iou = self.val_keypoint_iou.compute()
        road_iou = self.val_road_iou.compute()
        topo_f1 = self.val_topo_f1.compute()
        self.log("val_keypoint_iou", keypoint_iou, sync_dist=True)
        self.log("val_road_iou", road_iou, sync_dist=True)
        self.log("val_topo_f1", topo_f1, sync_dist=True)
        self.val_keypoint_iou.reset()
        self.val_road_iou.reset()
        self.val_topo_f1.reset()
    
    def _log_images(self, batch, mask_scores, prefix="train", max_viz=4):
        """
        Log mask prediction visualizations to TensorBoard.
        """
        # Visualize only occasionally for training, always first batch for validation
        should_log = prefix == "val" or (prefix == "train" and self.global_rank == 0)
        
        if not should_log:
            return
        
        # Log images to TensorBoard
        rgb = batch['rgb'][:max_viz]
        keypoint_mask = batch['keypoint_mask'][:max_viz]
        road_mask = batch['road_mask'][:max_viz]
        pred_keypoint = mask_scores[:max_viz, :, :, 0]
        pred_road = mask_scores[:max_viz, :, :, 1]
        
        num_viz = min(max_viz, rgb.size(0))
        for i in range(num_viz):
            fig, axs = plt.subplots(1, 5, figsize=(15, 3))
            
            # Ensure correct types and ranges for plotting
            rgb_img = rgb[i].cpu().numpy().astype(np.uint8)
            gt_kp_img = keypoint_mask[i].cpu().numpy().astype(float)
            gt_rd_img = road_mask[i].cpu().numpy().astype(float)
            pred_kp_img = pred_keypoint[i].detach().cpu().numpy()
            pred_rd_img = pred_road[i].detach().cpu().numpy()
            
            axs[0].imshow(rgb_img)
            axs[0].set_title('RGB')
            axs[0].axis('off')
            
            axs[1].imshow(gt_kp_img, cmap='gray', vmin=0, vmax=1)
            axs[1].set_title('GT Keypoint')
            axs[1].axis('off')
            
            axs[2].imshow(gt_rd_img, cmap='gray', vmin=0, vmax=1)
            axs[2].set_title('GT Road')
            axs[2].axis('off')
            
            axs[3].imshow(pred_kp_img, cmap='gray', vmin=0, vmax=1)
            axs[3].set_title('Pred Keypoint')
            axs[3].axis('off')
            
            axs[4].imshow(pred_rd_img, cmap='gray', vmin=0, vmax=1)
            axs[4].set_title('Pred Road')
            axs[4].axis('off')
            
            plt.tight_layout()
            self.logger.experiment.add_figure(f'{prefix}_sample_{i}', fig, self.global_step)
            plt.close(fig)

    def test_step(self, batch, batch_idx):
        # masks: [B, H, W]
        rgb, keypoint_mask, road_mask = batch['rgb'], batch['keypoint_mask'], batch['road_mask']
        graph_points, pairs, valid = batch['graph_points'], batch['pairs'], batch['valid']
        # masks: [B, H, W, 2] topo: [B, N_samples, N_pairs, 1]
        mask_logits, mask_scores, topo_logits, topo_scores = self(rgb, graph_points, pairs, valid)
        topo_gt, topo_loss_mask = batch['connected'].to(torch.int32), valid.to(torch.float32)
        
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
                
                # Align lengths if needed
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
                    
                    # Plot PR curve
                    try:
                        fig, ax = plt.subplots(1, 1, figsize=(8, 6))
                        ax.plot(recall.cpu(), precision.cpu(), label=f'{category} PR Curve')
                        ax.scatter(best_recall.cpu(), best_precision.cpu(), color='red', 
                                 label=f'Best F1 (Thresh={best_threshold:.3f})', zorder=5)
                        ax.set_xlabel("Recall")
                        ax.set_ylabel("Precision")
                        ax.set_title(f'{category} Precision-Recall Curve')
                        ax.legend()
                        ax.grid(True)
                        self.logger.experiment.add_figure(f'test_{category}_PR_Curve', fig, self.global_step)
                        plt.close(fig)
                    except Exception as plot_e:
                        print(f'Error plotting PR curve for {category}: {plot_e}')
            except Exception as e:
                print(f'Error processing PR curve for {category}: {e}')
            finally:
                pr_curve_metric.reset()
        
        print('======= Finding best thresholds ======')
        # find_best_threshold(self.keypoint_pr_curve, 'keypoint')
        # find_best_threshold(self.road_pr_curve, 'road')
        find_best_threshold(self.topo_pr_curve, 'topo')

    def configure_optimizers(self):
        param_dicts = []
        
        if not self.config.FREEZE_ENCODER and not self.config.ENCODER_LORA:
            encoder_params = {
                'params': [p for k, p in self.image_encoder.named_parameters() if 'image_encoder.'+k in self.matched_param_names],
                'lr': self.config.BASE_LR * self.config.ENCODER_LR_FACTOR,
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
                'lr': self.config.BASE_LR
            }]
        param_dicts += decoder_params

        topo_net_params = [{
            'params': [p for p in self.topo_net.parameters()],
            'lr': self.config.BASE_LR
        }]
        param_dicts += topo_net_params
        
        # Print parameter counts
        for i, param_dict in enumerate(param_dicts):
            param_num = sum([int(p.numel()) for p in param_dict['params']])
            print(f'optim param dict {i} params num: {param_num}')

        # Optimizer configuration
        base_lr = self.config.BASE_LR
        optimizer_name = self.config.get("OPTIMIZER", "AdamW").lower()
        weight_decay = self.config.get("WEIGHT_DECAY", 0.01)
        
        if optimizer_name == "adamw":
            print(f"Using AdamW optimizer with weight_decay={weight_decay}")
            optimizer = torch.optim.AdamW(param_dicts, lr=base_lr, weight_decay=weight_decay)
        elif optimizer_name == "adam":
            print("Using Adam optimizer")
            optimizer = torch.optim.Adam(param_dicts, lr=base_lr)
        else:
            raise ValueError(f"Unsupported optimizer: {optimizer_name}")
        
        # Scheduler - adaptive to max_epochs
        max_epochs = self.trainer.max_epochs
        step_lr = torch.optim.lr_scheduler.MultiStepLR(optimizer, milestones=[int(max_epochs * 0.8)], gamma=0.1)
        return {'optimizer': optimizer, 'lr_scheduler': step_lr}

