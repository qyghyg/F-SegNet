"""
Author: YAG
"""

import torch
import torch.nn as nn
import torch.nn.functional as F
import math
from typing import List, Tuple, Optional
import warnings

class DropPath(nn.Module):
    
    def __init__(self, drop_prob: float = 0.0):
        super(DropPath, self).__init__()
        self.drop_prob = drop_prob

    def forward(self, x):
        if self.drop_prob == 0.0 or not self.training:
            return x
        keep_prob = 1 - self.drop_prob
        shape = (x.shape[0],) + (1,) * (x.ndim - 1)  # work with diff dim tensors, not just 2D ConvNets
        random_tensor = keep_prob + torch.rand(shape, dtype=x.dtype, device=x.device)
        random_tensor.floor_()  # binarize
        output = x.div(keep_prob) * random_tensor
        return output

class MLP(nn.Module):
    
    def __init__(self, in_features: int, hidden_features: int = None, out_features: int = None, 
                 act_layer: nn.Module = nn.GELU, drop: float = 0.0):
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
    
    def __init__(self, dim: int, num_heads: int = 8, qkv_bias: bool = False, 
                 attn_drop: float = 0.0, proj_drop: float = 0.0, sr_ratio: int = 1):
        super().__init__()
        assert dim % num_heads == 0, f"dim {dim} should be divided by num_heads {num_heads}."

        self.dim = dim
        self.num_heads = num_heads
        head_dim = dim // num_heads
        self.scale = head_dim ** -0.5

        self.q = nn.Linear(dim, dim, bias=qkv_bias)
        self.kv = nn.Linear(dim, dim * 2, bias=qkv_bias)
        self.attn_drop = nn.Dropout(attn_drop)
        self.proj = nn.Linear(dim, dim)
        self.proj_drop = nn.Dropout(proj_drop)

        self.sr_ratio = sr_ratio
        if sr_ratio > 1:
            self.sr = nn.Conv2d(dim, dim, kernel_size=sr_ratio, stride=sr_ratio)
            self.norm = nn.LayerNorm(dim)

    def forward(self, x, H: int, W: int):
        B, N, C = x.shape
        q = self.q(x).reshape(B, N, self.num_heads, C // self.num_heads).permute(0, 2, 1, 3)

        if self.sr_ratio > 1:
            x_ = x.permute(0, 2, 1).reshape(B, C, H, W)
            x_ = self.sr(x_).reshape(B, C, -1).permute(0, 2, 1)
            x_ = self.norm(x_)
            kv = self.kv(x_).reshape(B, -1, 2, self.num_heads, C // self.num_heads).permute(2, 0, 3, 1, 4)
        else:
            kv = self.kv(x).reshape(B, -1, 2, self.num_heads, C // self.num_heads).permute(2, 0, 3, 1, 4)
        k, v = kv[0], kv[1]

        attn = (q @ k.transpose(-2, -1)) * self.scale
        attn = attn.softmax(dim=-1)
        attn = self.attn_drop(attn)

        x = (attn @ v).transpose(1, 2).reshape(B, N, C)
        x = self.proj(x)
        x = self.proj_drop(x)

        return x

class Block(nn.Module):
    
    def __init__(self, dim: int, num_heads: int, mlp_ratio: float = 4.0, qkv_bias: bool = False,
                 drop: float = 0.0, attn_drop: float = 0.0, drop_path: float = 0.0,
                 act_layer: nn.Module = nn.GELU, norm_layer: nn.Module = nn.LayerNorm, sr_ratio: int = 1):
        super().__init__()
        self.norm1 = norm_layer(dim)
        self.attn = Attention(
            dim, num_heads=num_heads, qkv_bias=qkv_bias,
            attn_drop=attn_drop, proj_drop=drop, sr_ratio=sr_ratio)
        
        self.drop_path = DropPath(drop_path) if drop_path > 0.0 else nn.Identity()
        self.norm2 = norm_layer(dim)
        mlp_hidden_dim = int(dim * mlp_ratio)
        self.mlp = MLP(in_features=dim, hidden_features=mlp_hidden_dim, act_layer=act_layer, drop=drop)

    def forward(self, x, H: int, W: int):
        x = x + self.drop_path(self.attn(self.norm1(x), H, W))
        x = x + self.drop_path(self.mlp(self.norm2(x)))
        return x

class OverlapPatchEmbed(nn.Module):
    
    def __init__(self, img_size: int = 1000, patch_size: int = 7, stride: int = 4, 
                 in_chans: int = 3, embed_dim: int = 768):
        super().__init__()
        img_size = (img_size, img_size)
        patch_size = (patch_size, patch_size)

        self.img_size = img_size
        self.patch_size = patch_size
        self.H, self.W = img_size[0] // stride, img_size[1] // stride
        self.num_patches = self.H * self.W
        
        self.proj = nn.Conv2d(in_chans, embed_dim, kernel_size=patch_size, stride=stride,
                              padding=(patch_size[0] // 2, patch_size[1] // 2))
        self.norm = nn.LayerNorm(embed_dim)

    def forward(self, x):
        x = self.proj(x)
        _, _, H, W = x.shape
        x = x.flatten(2).transpose(1, 2)
        x = self.norm(x)
        return x, H, W

class MixVisionTransformer(nn.Module):
    
    def __init__(self, img_size: int = 1000, patch_size: int = 16, in_chans: int = 3, 
                 embed_dims: List[int] = [64, 128, 256, 512], num_heads: List[int] = [1, 2, 4, 8],
                 mlp_ratios: List[int] = [4, 4, 4, 4], qkv_bias: bool = True, drop_rate: float = 0.0,
                 attn_drop_rate: float = 0.0, drop_path_rate: float = 0.0, norm_layer: nn.Module = nn.LayerNorm,
                 depths: List[int] = [3, 4, 6, 3], sr_ratios: List[int] = [8, 4, 2, 1],
                 num_stages: int = 4, pretrained: Optional[str] = None):
        super().__init__()
        
        self.depths = depths
        self.num_stages = num_stages

        dpr = [x.item() for x in torch.linspace(0, drop_path_rate, sum(depths))]  # stochastic depth decay rule
        cur = 0

        for i in range(num_stages):
            if i == 0:
                patch_embed = OverlapPatchEmbed(img_size=img_size, patch_size=7, stride=4, 
                                              in_chans=in_chans, embed_dim=embed_dims[i])
            else:
                patch_embed = OverlapPatchEmbed(img_size=img_size // (2 ** (i + 1)), patch_size=3, 
                                              stride=2, in_chans=embed_dims[i - 1], embed_dim=embed_dims[i])

            block = nn.ModuleList([Block(
                dim=embed_dims[i], num_heads=num_heads[i], mlp_ratio=mlp_ratios[i], qkv_bias=qkv_bias,
                drop=drop_rate, attn_drop=attn_drop_rate, drop_path=dpr[cur + j], norm_layer=norm_layer,
                sr_ratio=sr_ratios[i])
                for j in range(depths[i])])
            norm = norm_layer(embed_dims[i])
            cur += depths[i]

            setattr(self, f"patch_embed{i + 1}", patch_embed)
            setattr(self, f"block{i + 1}", block)
            setattr(self, f"norm{i + 1}", norm)

    def forward(self, x):
        B = x.shape[0]
        outs = []

        for i in range(self.num_stages):
            patch_embed = getattr(self, f"patch_embed{i + 1}")
            block = getattr(self, f"block{i + 1}")
            norm = getattr(self, f"norm{i + 1}")
            
            x, H, W = patch_embed(x)
            for blk in block:
                x = blk(x, H, W)
            x = norm(x)
            x = x.reshape(B, H, W, -1).permute(0, 3, 1, 2).contiguous()
            outs.append(x)

        return outs

class SegformerHead(nn.Module):
    
    def __init__(self, in_channels: List[int], channels: int = 256, 
                 dropout_ratio: float = 0.1, align_corners: bool = False):
        super().__init__()
        
        self.in_channels = in_channels
        self.channels = channels
        self.align_corners = align_corners
        
        self.convs = nn.ModuleList()
        for in_channel in in_channels:
            self.convs.append(
                nn.Sequential(
                    nn.Conv2d(in_channel, channels, 1, bias=False),
                    nn.BatchNorm2d(channels),
                    nn.ReLU(inplace=True)
                )
            )
        
        self.fusion_conv = nn.Sequential(
            nn.Conv2d(channels * len(in_channels), channels, 1, bias=False),
            nn.BatchNorm2d(channels),
            nn.ReLU(inplace=True)
        )

        self.classifier = nn.Sequential(
            nn.Dropout2d(dropout_ratio),
            nn.Conv2d(channels, 1, 1),  
            nn.Sigmoid()  
        )

    def forward(self, inputs: List[torch.Tensor]) -> torch.Tensor:
        outs = []
        for idx, x in enumerate(inputs):
            conv = self.convs[idx]
            x = conv(x)
            if idx != 0:
                x = F.interpolate(x, size=inputs[0].shape[2:], 
                                mode='bilinear', align_corners=self.align_corners)
            outs.append(x)
        
        out = self.fusion_conv(torch.cat(outs, dim=1))
        out = self.classifier(out)
        
        return out

class SegFormer(nn.Module):
    def __init__(self, img_size: int = 1000, 
                 embed_dims: List[int] = [64, 128, 320, 512],
                 num_heads: List[int] = [1, 2, 5, 8],
                 depths: List[int] = [3, 6, 40, 3],
                 sr_ratios: List[int] = [8, 4, 2, 1],
                 mlp_ratios: List[int] = [4, 4, 4, 4],
                 drop_path_rate: float = 0.1,
                 decoder_channels: int = 256):
        super().__init__()
        
        self.backbone = MixVisionTransformer(
            img_size=img_size,
            embed_dims=embed_dims,
            num_heads=num_heads,
            depths=depths,
            sr_ratios=sr_ratios,
            mlp_ratios=mlp_ratios,
            drop_path_rate=drop_path_rate
        )
        
        self.decode_head = SegformerHead(
            in_channels=embed_dims,
            channels=decoder_channels
        )
        
    def forward(self, x):
        features = self.backbone(x)
        seg_probs = self.decode_head(features)
        
        seg_probs = F.interpolate(seg_probs, size=x.shape[2:], 
                                mode='bilinear', align_corners=False)
        
        seg_probs = seg_probs.squeeze(1)
        
        return seg_probs

def create_segformer_b5(img_size: int = 1000) -> SegFormer:
    return SegFormer(
        img_size=img_size,
        embed_dims=[64, 128, 320, 512],
        num_heads=[1, 2, 5, 8], 
        depths=[3, 6, 40, 3],
        sr_ratios=[8, 4, 2, 1],
        drop_path_rate=0.1,
        decoder_channels=256
    )


if __name__ == "__main__":
    
    output = model(x)
    print(f"Input shape: {x.shape}")
    
    total_params = sum(p.numel() for p in model.parameters())
    trainable_params = sum(p.numel() for p in model.parameters() if p.requires_grad)
    
    print(f"Total parameters: {total_params:,}")
    print(f"Trainable parameters: {trainable_params:,}")
    print(f"- Recommended batch size: 4 (as per paper, but may need GPU with >16GB memory)")
    print(f"- Consider gradient accumulation for smaller GPUs")
    
    try:
        x_batch = torch.randn(4, 3, 1000, 1000)
        output_batch = model(x_batch)
        print(f"\nBatch test successful:")
        print(f"Batch input shape: {x_batch.shape}")
        print(f"Batch output shape: {output_batch.shape}")
    except RuntimeError as e:
        print(f"\nBatch size 4 failed (likely due to memory): {e}")
        print("Consider reducing batch size or using gradient accumulation")
