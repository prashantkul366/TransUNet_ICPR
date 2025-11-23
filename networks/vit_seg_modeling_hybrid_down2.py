# coding=utf-8
from __future__ import absolute_import
from __future__ import division
from __future__ import print_function

import copy
import logging
import math

from os.path import join as pjoin

import torch
import torch.nn as nn
import numpy as np

from torch.nn import CrossEntropyLoss, Dropout, Softmax, Linear, Conv2d, LayerNorm
from torch.nn.modules.utils import _pair
from scipy import ndimage
from . import vit_seg_configs as configs
from .vit_seg_modeling_resnet_skip import ResNetV2
# from .kan import KAN 
from .kan_fJNB import KAN
from einops import rearrange, repeat
from timm.models.layers import trunc_normal_, DropPath, LayerNorm2d
from mamba_ssm.ops.selective_scan_interface import selective_scan_fn
logger = logging.getLogger(__name__)
import torch.nn.functional as F

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


def swish(x):
    return x * torch.sigmoid(x)


ACT2FN = {"gelu": torch.nn.functional.gelu, "relu": torch.nn.functional.relu, "swish": swish}



##########################################################################
## Multi-DConv Head Transposed Self-Attention (MDTA)
class Attention(nn.Module):
    def __init__(self, dim, num_heads, bias):
        super(Attention, self).__init__()
        self.num_heads = num_heads
        self.temperature = nn.Parameter(torch.ones(num_heads, 1, 1))

        self.qkv = nn.Conv2d(dim, dim*3, kernel_size=1, bias=bias)
        self.qkv_dwconv = nn.Conv2d(dim*3, dim*3, kernel_size=3, stride=1, padding=1, groups=dim*3, bias=bias)
        self.project_out = nn.Conv2d(dim, dim, kernel_size=1, bias=bias)
        


    def forward(self, x):
        b,c,h,w = x.shape

        qkv = self.qkv_dwconv(self.qkv(x))
        q,k,v = qkv.chunk(3, dim=1)   
        
        q = rearrange(q, 'b (head c) h w -> b head c (h w)', head=self.num_heads)
        k = rearrange(k, 'b (head c) h w -> b head c (h w)', head=self.num_heads)
        v = rearrange(v, 'b (head c) h w -> b head c (h w)', head=self.num_heads)

        q = torch.nn.functional.normalize(q, dim=-1)
        k = torch.nn.functional.normalize(k, dim=-1)

        attn = (q @ k.transpose(-2, -1)) * self.temperature
        attn = attn.softmax(dim=-1)

        out = (attn @ v)
        
        out = rearrange(out, 'b head c (h w) -> b (head c) h w', head=self.num_heads, h=h, w=w)

        out = self.project_out(out)
        return out




class Mlp(nn.Module):
    def __init__(self, config):
        super(Mlp, self).__init__()
        self.fc1 = Linear(config.hidden_size, config.transformer["mlp_dim"])
        self.fc2 = Linear(config.transformer["mlp_dim"], config.hidden_size)
        self.act_fn = ACT2FN["gelu"]
        self.dropout = Dropout(config.transformer["dropout_rate"])

        self._init_weights()

    def _init_weights(self):
        nn.init.xavier_uniform_(self.fc1.weight)
        nn.init.xavier_uniform_(self.fc2.weight)
        nn.init.normal_(self.fc1.bias, std=1e-6)
        nn.init.normal_(self.fc2.bias, std=1e-6)

    def forward(self, x):
        x = self.fc1(x)
        x = self.act_fn(x)
        x = self.dropout(x)
        x = self.fc2(x)
        x = self.dropout(x)
        return x

class KANMLP(nn.Module):
    """
    Drop-in replacement for Mlp that uses KAN.
    Expects (B, N, D) in and out. Internally flattens tokens to (B*N, D).
    """
    def __init__(self, config):
        super().__init__()
        hidden = config.hidden_size
        mlp_dim = config.transformer["mlp_dim"]

        # Read KAN hyperparams from config with sensible defaults
        self.grid_size    = getattr(config, "kan_grid_size", 5)
        self.spline_order = getattr(config, "kan_spline_order", 3)
        self.scale_noise  = getattr(config, "kan_scale_noise", 0.1)
        self.scale_base   = getattr(config, "kan_scale_base", 1.0)
        self.scale_spline = getattr(config, "kan_scale_spline", 1.0)
        self.grid_eps     = getattr(config, "kan_grid_eps", 0.02)
        self.grid_range   = getattr(config, "kan_grid_range", [-1, 1])

        # Two-layer MLP path: hidden -> mlp_dim -> hidden, but using KAN
        self.kan = KAN(
            layers_hidden=[hidden, mlp_dim, hidden],
            grid_size=self.grid_size,
            spline_order=self.spline_order,
            scale_noise=self.scale_noise,
            scale_base=self.scale_base,
            scale_spline=self.scale_spline,
            grid_eps=self.grid_eps,
            grid_range=self.grid_range,
        )

        # Keep the same dropout semantics as the original Mlp
        self.dropout = Dropout(config.transformer["dropout_rate"])

        # Optional: tiny LayerNorm to stabilize ranges before KAN (KAN grid default is [-1,1])
        # LayerNorm keeps zero mean/unit variance, which is friendly to KAN splines.
        self.pre_norm = LayerNorm(hidden, eps=1e-6)

    def forward(self, x):
        # x: (B, N, D)
        B, N, D = x.shape
        x = self.pre_norm(x)

        x_flat = x.reshape(B * N, D)   # (BN, D)
        y_flat = self.kan(x_flat)      # (BN, D)
        y = y_flat.reshape(B, N, D)    # (B, N, D)

        # Match original MLP’s dropout-after-each-linear feel
        y = self.dropout(y)
        return y

    def regularization_loss(self, reg_act=1.0, reg_ent=1.0):
        # Expose KAN’s regularizer so your training loop can add it
        return self.kan.regularization_loss(reg_act, reg_ent)

class Embeddings(nn.Module):
    """Pure ViT-style patch + position embeddings (no conv backbone)."""
    def __init__(self, config, img_size, in_channels=3):
        super().__init__()
        self.hybrid = False
        self.config = config
        img_size = _pair(img_size)

        # IMPORTANT: set config.patches["size"] (e.g. 4 or 8) in your config
        patch_size = _pair(config.patches["size"])
        grid_h = img_size[0] // patch_size[0]
        grid_w = img_size[1] // patch_size[1]
        n_patches = grid_h * grid_w

        self.grid_size = (grid_h, grid_w)

        self.patch_embeddings = Conv2d(
            in_channels=in_channels,
            out_channels=config.hidden_size,
            kernel_size=patch_size,
            stride=patch_size,
        )

        self.position_embeddings = nn.Parameter(
            torch.zeros(1, n_patches, config.hidden_size)
        )
        self.dropout = Dropout(config.transformer["dropout_rate"])

    def forward(self, x):
        # x: (B, 3, H, W)
        print("\n[Embeddings] input:", x.shape)
        x = self.patch_embeddings(x)         # (B, D, H', W')
        B, D, H, W = x.shape
        print(f"[Embeddings] after patch_embeddings: {x.shape} (H'={H}, W'={W})")

        x = x.flatten(2).transpose(1, 2)     # (B, N, D), N = H'*W'
        print("[Embeddings] tokens:", x.shape)
        # PE was created for N = grid_h*grid_w, which matches H', W'
        embeddings = x + self.position_embeddings
        embeddings = self.dropout(embeddings)

        # No conv features, we are pure now
        features = None
        return embeddings, features


class TokenMDTA(nn.Module):
    def __init__(self, dim, num_heads, bias):
        super().__init__()
        self.dim = dim
        self.num_heads = num_heads
        self.inner = Attention(dim, num_heads, bias)

    def forward(self, x):
        # x: (B, N, D)
        B, N, D = x.shape
        H = W = int(math.sqrt(N))
        assert H * W == N, "Token count N must be a perfect square"

        x_2d = x.permute(0, 2, 1).reshape(B, D, H, W)   # (B, D, H, W)
        out_2d = self.inner(x_2d)                       # (B, D, H, W)
        out = out_2d.reshape(B, D, N).permute(0, 2, 1)  # (B, N, D)

        # no explicit attention weights here
        weights = None
        return out, weights

class T_Block(nn.Module):
    def __init__(self, config, vis):
        super(T_Block, self).__init__()
        self.hidden_size = config.hidden_size
        self.attention_norm = LayerNorm(config.hidden_size, eps=1e-6)
        self.ffn_norm = LayerNorm(config.hidden_size, eps=1e-6)

        print("Transformer block with MDTA + KAN MLP initiated")
        # ############################################################
        # # self.ffn = Mlp(config)
        # self.ffn = KANMLP(config)
        # ############################################################
        
        num_heads = config.transformer.get("num_heads", 4)
        self.attn = TokenMDTA(dim=self.hidden_size, num_heads=num_heads, bias=True)
        self.ffn = KANMLP(config)
        

    def forward(self, x):
        h = x
        x = self.attention_norm(x)
        x, weights = self.attn(x)
        x = x + h

        h = x
        x = self.ffn_norm(x)
        x = self.ffn(x)
        x = x + h
        return x, weights
    

class Transformer_Encoder(nn.Module):
    def __init__(self, config, vis):
        super(Transformer_Encoder, self).__init__()
        self.vis = vis
        self.layer = nn.ModuleList()
        self.encoder_norm = LayerNorm(config.hidden_size, eps=1e-6)
        for _ in range(config.transformer["num_layers"]):
            layer = T_Block(config, vis)
            self.layer.append(copy.deepcopy(layer))

    def forward(self, hidden_states):
        attn_weights = []
        for layer_block in self.layer:
            hidden_states, weights = layer_block(hidden_states)
            if self.vis:
                attn_weights.append(weights)
        encoded = self.encoder_norm(hidden_states)
        return encoded, attn_weights


class Mamba_Encoder(nn.Module):
    def __init__(self, config, vis):
        super(Mamba_Encoder, self).__init__()
        self.vis = vis
        self.layer = nn.ModuleList()
        self.encoder_norm = LayerNorm(config.hidden_size, eps=1e-6)
        for _ in range(config.transformer["num_layers"]):
            layer = M_Block(config, vis)
            self.layer.append(copy.deepcopy(layer))

    def forward(self, hidden_states):
        attn_weights = []
        for layer_block in self.layer:
            hidden_states, weights = layer_block(hidden_states)
            if self.vis:
                attn_weights.append(weights)
        encoded = self.encoder_norm(hidden_states)
        return encoded, attn_weights


class M_Block(nn.Module):
    def __init__(self, config, vis):
        super(M_Block, self).__init__()
        self.hidden_size = config.hidden_size
        self.attention_norm = LayerNorm(config.hidden_size, eps=1e-6)
        self.ffn_norm = LayerNorm(config.hidden_size, eps=1e-6)
        self.vis = vis

        print("Block with MambaVisionMixer + fJNB KAN instead of MLP initiated")
        
        self.attn = MambaVisionMixer(
            d_model=config.hidden_size,
            d_state=8,
            d_conv=3,
            expand=1,
        )
        self.ffn = KANMLP(config)


    def forward(self, x):
        h = x
        x = self.attention_norm(x)
        x = self.attn(x)
        x = x + h

        h = x
        x = self.ffn_norm(x)
        x = self.ffn(x)
        x = x + h

        weights = None
        return x, weights

    
class MambaVisionMixer(nn.Module):
    def __init__(
        self,
        d_model,
        d_state=16,
        d_conv=4,
        expand=2,
        dt_rank="auto",
        dt_min=0.001,
        dt_max=0.1,
        dt_init="random",
        dt_scale=1.0,
        dt_init_floor=1e-4,
        conv_bias=True,
        bias=False,
        use_fast_path=True, 
        layer_idx=None,
        device=None,
        dtype=None,
    ):
        factory_kwargs = {"device": device, "dtype": dtype}
        super().__init__()
        self.d_model = d_model
        self.d_state = d_state
        self.d_conv = d_conv
        self.expand = expand
        self.d_inner = int(self.expand * self.d_model)
        self.dt_rank = math.ceil(self.d_model / 16) if dt_rank == "auto" else dt_rank
        self.use_fast_path = use_fast_path
        self.layer_idx = layer_idx
        self.in_proj = nn.Linear(self.d_model, self.d_inner, bias=bias, **factory_kwargs)    
        self.x_proj = nn.Linear(
            self.d_inner//2, self.dt_rank + self.d_state * 2, bias=False, **factory_kwargs
        )
        self.dt_proj = nn.Linear(self.dt_rank, self.d_inner//2, bias=True, **factory_kwargs)
        dt_init_std = self.dt_rank**-0.5 * dt_scale
        if dt_init == "constant":
            nn.init.constant_(self.dt_proj.weight, dt_init_std)
        elif dt_init == "random":
            nn.init.uniform_(self.dt_proj.weight, -dt_init_std, dt_init_std)
        else:
            raise NotImplementedError
        dt = torch.exp(
            torch.rand(self.d_inner//2, **factory_kwargs) * (math.log(dt_max) - math.log(dt_min))
            + math.log(dt_min)
        ).clamp(min=dt_init_floor)
        inv_dt = dt + torch.log(-torch.expm1(-dt))
        with torch.no_grad():
            self.dt_proj.bias.copy_(inv_dt)
        self.dt_proj.bias._no_reinit = True
        A = repeat(
            torch.arange(1, self.d_state + 1, dtype=torch.float32, device=device),
            "n -> d n",
            d=self.d_inner//2,
        ).contiguous()
        A_log = torch.log(A)
        self.A_log = nn.Parameter(A_log)
        self.A_log._no_weight_decay = True
        self.D = nn.Parameter(torch.ones(self.d_inner//2, device=device))
        self.D._no_weight_decay = True
        self.out_proj = nn.Linear(self.d_inner, self.d_model, bias=bias, **factory_kwargs)
        self.conv1d_x = nn.Conv1d(
            in_channels=self.d_inner//2,
            out_channels=self.d_inner//2,
            bias=conv_bias//2,
            kernel_size=d_conv,
            groups=self.d_inner//2,
            **factory_kwargs,
        )
        self.conv1d_z = nn.Conv1d(
            in_channels=self.d_inner//2,
            out_channels=self.d_inner//2,
            bias=conv_bias//2,
            kernel_size=d_conv,
            groups=self.d_inner//2,
            **factory_kwargs,
        )

    def forward(self, hidden_states):
        """
        hidden_states: (B, L, D)
        Returns: same shape as hidden_states
        """
        _, seqlen, _ = hidden_states.shape
        xz = self.in_proj(hidden_states)
        xz = rearrange(xz, "b l d -> b d l")
        x, z = xz.chunk(2, dim=1)
        A = -torch.exp(self.A_log.float())
        x = F.silu(F.conv1d(input=x, weight=self.conv1d_x.weight, bias=self.conv1d_x.bias, padding='same', groups=self.d_inner//2))
        z = F.silu(F.conv1d(input=z, weight=self.conv1d_z.weight, bias=self.conv1d_z.bias, padding='same', groups=self.d_inner//2))
        x_dbl = self.x_proj(rearrange(x, "b d l -> (b l) d"))
        dt, B, C = torch.split(x_dbl, [self.dt_rank, self.d_state, self.d_state], dim=-1)
        dt = rearrange(self.dt_proj(dt), "(b l) d -> b d l", l=seqlen)
        B = rearrange(B, "(b l) dstate -> b dstate l", l=seqlen).contiguous()
        C = rearrange(C, "(b l) dstate -> b dstate l", l=seqlen).contiguous()
        y = selective_scan_fn(x, 
                              dt, 
                              A, 
                              B, 
                              C, 
                              self.D.float(), 
                              z=None, 
                              delta_bias=self.dt_proj.bias.float(), 
                              delta_softplus=True, 
                              return_last_state=None)
        
        y = torch.cat([y, z], dim=1)
        y = rearrange(y, "b d l -> b l d")
        out = self.out_proj(y)
        return out
    
class TokenDownsample2x(nn.Module):
    """
    Downsample tokens by 2x in H and W in a pure way:
    - Reshape (B, N, D) -> (B, D, H, W)
    - AvgPool2d by 2
    - Optional 1x1 conv on channels
    - Back to tokens (B, N', D_out)
    """
    def __init__(self, dim_in, dim_out=None):
        super().__init__()
        dim_out = dim_out or dim_in
        self.proj = nn.Conv2d(dim_in, dim_out, kernel_size=1)

    def forward(self, x, H, W):
        B, N, D = x.shape
        assert N == H * W, "Token count must match H*W"
        x_2d = x.permute(0, 2, 1).reshape(B, D, H, W)  # (B, D, H, W)

        # spatial downsample by 2
        x_2d = F.avg_pool2d(x_2d, kernel_size=2, stride=2)  # (B, D, H/2, W/2)
        x_2d = self.proj(x_2d)                              # (B, D_out, H/2, W/2)

        H2, W2 = H // 2, W // 2
        x_ds = x_2d.flatten(2).permute(0, 2, 1)             # (B, H2*W2, D_out)
        return x_ds, H2, W2, x_2d


class Transformer(nn.Module):
    def __init__(self, config, img_size, vis):
        super(Transformer, self).__init__()
        self.embeddings = Embeddings(config, img_size=img_size)
        self.vis = vis
        self.hidden_size = config.hidden_size

        # 4 stages: e.g. 56x56 -> 28x28 -> 14x14 -> 7x7 (if patch_size=4)
        self.num_stages = 4

        # Each stage: Transformer block (MDTA + KAN) + Mamba block (Mamba + KAN)
        self.stages = nn.ModuleList([
            nn.ModuleList([
                T_Block(config, vis),
                M_Block(config, vis),
            ])
            for _ in range(self.num_stages)
        ])

        # Downsampling between stages (3 downs for 4 stages)
        self.downsamples = nn.ModuleList([
            TokenDownsample2x(self.hidden_size, self.hidden_size)
            for _ in range(self.num_stages - 1)
        ])

        self.encoder_norm = LayerNorm(config.hidden_size, eps=1e-6)

    def forward(self, input_ids):
        # input_ids: (B, 3, 224, 224)
        x, _ = self.embeddings(input_ids)   # tokens only, (B, N, D)
        B, N, D = x.shape
        H = W = int(math.sqrt(N))
        print(f"\n[Transformer] tokens in: {x.shape}, H={H}, W={W}")
        assert H * W == N, "Token count must be a perfect square"

        all_attn_weights = []
        multi_scale_feats = []  # will store [56x56, 28x28, 14x14, 7x7] as 2D maps

        for stage_idx in range(self.num_stages):
            t_block, m_block = self.stages[stage_idx]
            print(f"[Transformer] Stage {stage_idx} - in: {x.shape} (H={H}, W={W})")

            # Transformer-style block
            x, w_t = t_block(x)
            print(f"[Transformer] Stage {stage_idx} - after T_Block: {x.shape}")
            if self.vis:
                all_attn_weights.append(w_t)

            # Mamba-style block
            x, w_m = m_block(x)
            print(f"[Transformer] Stage {stage_idx} - after M_Block: {x.shape}")
            if self.vis:
                all_attn_weights.append(w_m)

            # Save current scale as 2D feature map
            feat_2d = x.permute(0, 2, 1).reshape(B, D, H, W)  # (B, D, H, W)
            print(f"[Transformer] Stage {stage_idx} - feat_2d: {feat_2d.shape}")
            multi_scale_feats.append(feat_2d)

            # Downsample tokens for next stage (except after last stage)
            if stage_idx < self.num_stages - 1:
                x, H, W, _ = self.downsamples[stage_idx](x, H, W)
                print(f"[Transformer] Stage {stage_idx} - downsampled tokens: {x.shape}, new H={H}, W={W}")

        # Normalize final tokens (bottleneck, e.g. 7x7)
        x = self.encoder_norm(x)
        print("[Transformer] final tokens:", x.shape)

        for i, f in enumerate(multi_scale_feats):
            print(f"[Transformer] multi_scale_feats[{i}] = {f.shape}")

        # multi_scale_feats order currently: [56x56, 28x28, 14x14, 7x7]
        # Decoder wants deep -> shallow: first skip used at smallest spatial size.
        feat0, feat1, feat2, feat3 = multi_scale_feats  # 56, 28, 14, 7
        skip_features = [feat3, feat2, feat1, feat0]    # [7, 14, 28, 56]

        return x, all_attn_weights, skip_features

class Conv2dReLU(nn.Sequential):
    def __init__(
            self,
            in_channels,
            out_channels,
            kernel_size,
            padding=0,
            stride=1,
            use_batchnorm=True,
    ):
        conv = nn.Conv2d(
            in_channels,
            out_channels,
            kernel_size,
            stride=stride,
            padding=padding,
            bias=not (use_batchnorm),
        )
        relu = nn.ReLU(inplace=True)

        bn = nn.BatchNorm2d(out_channels)

        super(Conv2dReLU, self).__init__(conv, bn, relu)


class DecoderBlock(nn.Module):
    def __init__(
            self,
            in_channels,
            out_channels,
            skip_channels=0,
            use_batchnorm=True,
    ):
        super().__init__()
        self.conv1 = Conv2dReLU(
            in_channels + skip_channels,
            out_channels,
            kernel_size=3,
            padding=1,
            use_batchnorm=use_batchnorm,
        )
        self.conv2 = Conv2dReLU(
            out_channels,
            out_channels,
            kernel_size=3,
            padding=1,
            use_batchnorm=use_batchnorm,
        )
        self.up = nn.UpsamplingBilinear2d(scale_factor=2)

    def forward(self, x, skip=None):

        # print("\nDecoderBlock - input x:", x.shape)
        x = self.up(x)
        # print("After upsample:", x.shape)

        if skip is not None:
            if skip.shape[2:] != x.shape[2:]:
                skip = F.interpolate(
                    skip,
                    size=x.shape[2:],
                    mode="bilinear",
                    align_corners=False,
                )
                # print("Resized skip:", skip.shape)

            x = torch.cat([x, skip], dim=1)

        # if skip is not None:
        #     x = torch.cat([x, skip], dim=1)


        x = self.conv1(x)
        # print("After conv1:", x.shape)
        x = self.conv2(x)
        # print("After conv2:", x.shape)
        return x


class SegmentationHead(nn.Sequential):

    def __init__(self, in_channels, out_channels, kernel_size=3, upsampling=1):
        conv2d = nn.Conv2d(in_channels, out_channels, kernel_size=kernel_size, padding=kernel_size // 2)
        upsampling = nn.UpsamplingBilinear2d(scale_factor=upsampling) if upsampling > 1 else nn.Identity()
        super().__init__(conv2d, upsampling)


class DecoderCup(nn.Module):
    def __init__(self, config):
        super().__init__()
        self.config = config
        head_channels = 512
        self.conv_more = Conv2dReLU(
            config.hidden_size,
            head_channels,
            kernel_size=3,
            padding=1,
            use_batchnorm=True,
        )
        decoder_channels = config.decoder_channels
        in_channels = [head_channels] + list(decoder_channels[:-1])
        out_channels = decoder_channels

        if self.config.n_skip != 0:
            skip_channels = self.config.skip_channels
            for i in range(4-self.config.n_skip):  # re-select the skip channels according to n_skip
                skip_channels[3-i]=0

        else:
            skip_channels=[0,0,0,0]

        blocks = [
            DecoderBlock(in_ch, out_ch, sk_ch) for in_ch, out_ch, sk_ch in zip(in_channels, out_channels, skip_channels)
        ]
        self.blocks = nn.ModuleList(blocks)

    def forward(self, hidden_states, features=None):
        B, n_patch, hidden = hidden_states.size()  # reshape from (B, n_patch, hidden) to (B, h, w, hidden)
        h, w = int(np.sqrt(n_patch)), int(np.sqrt(n_patch))

        print("\n[DecoderCup] hidden_states:", hidden_states.shape)
        x = hidden_states.permute(0, 2, 1)
        x = x.contiguous().view(B, hidden, h, w)
        print("[DecoderCup] reshaped to feature map:", x.shape)
        x = self.conv_more(x)
        print("[DecoderCup] after conv_more:", x.shape)

        for i, decoder_block in enumerate(self.blocks):
            # print(f"\n[DecoderCup] -- Decoder block {i} --")
            if features is not None:
                skip = features[i] if (i < self.config.n_skip) else None
            else:
                skip = None

            if skip is not None:
                print(f"[DecoderCup] skip[{i}] before resize:", skip.shape)
            else:
                print(f"[DecoderCup] skip[{i}] is None")  

            x = decoder_block(x, skip=skip)
            print("[DecoderCup] Output of decoder_block:", x.shape)

        return x


class VisionTransformer(nn.Module):
    # def __init__(self, config, img_size=224, num_classes=21843, zero_head=False, vis=False):
    def __init__(self, config, img_size=224, num_classes=2, zero_head=False, vis=False):
        super(VisionTransformer, self).__init__()
        self.num_classes = num_classes
        self.zero_head = zero_head
        self.classifier = config.classifier
        self.transformer = Transformer(config, img_size, vis)
        self.decoder = DecoderCup(config)
        # self.segmentation_head = SegmentationHead(
        #     in_channels=config['decoder_channels'][-1],
        #     out_channels=config['n_classes'],
        #     kernel_size=3,
        # )
        self.segmentation_head = SegmentationHead(
            in_channels=config.decoder_channels[-1],
            out_channels=config.n_classes,
            kernel_size=3,
        )
        self.config = config
        print("Proposed Hybrid Model with pure patches Created")

    def forward(self, x):
        print("\n=== VisionTransformer.forward ===")
        print("Input to ViT (before channel repeat):", x.shape)

        if x.size()[1] == 1:
            x = x.repeat(1,3,1,1)
        
        print("Input to transformer (after repeat if any):", x.shape)
        x, attn_weights, features = self.transformer(x)  # (B, n_patch, hidden)
        
        print("Output tokens from transformer:", x.shape)
        print("Num skip features from transformer:", len(features))

        for i, f in enumerate(features):
            print(f"  Skip {i} shape:", f.shape)

        x = self.decoder(x, features)
        print("Decoder output feature map:", x.shape)

        logits = self.segmentation_head(x)
        print("Final logits:", logits.shape)
        return logits

   
CONFIGS = {
    'ViT-B_16': configs.get_b16_config(),
    'ViT-B_32': configs.get_b32_config(),
    'ViT-L_16': configs.get_l16_config(),
    'ViT-L_32': configs.get_l32_config(),
    'ViT-H_14': configs.get_h14_config(),
    'R50-ViT-B_16': configs.get_r50_b16_config(),
    'R50-ViT-L_16': configs.get_r50_l16_config(),
    'testing': configs.get_testing(),
}


