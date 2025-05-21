"""
reference
- https://github.com/PaddlePaddle/PaddleDetection/blob/develop/ppdet/modeling/backbones/hgnet_v2.py

Copyright (c) 2024 The D-FINE Authors. All Rights Reserved.
"""

import torch
import torch.nn as nn
import torch.nn.functional as F
import os
from .common import FrozenBatchNorm2d
from ..core import register
import logging

# Constants for initialization
kaiming_normal_ = nn.init.kaiming_normal_
zeros_ = nn.init.zeros_
ones_ = nn.init.ones_

__all__ = ['HGNetv2']

class FusionModule(nn.Module):
    """
    Fusion module for RGB and IR features in HGNetv2
    """

    def __init__(self, channels, fusion_type='concat'):
        super().__init__()
        self.fusion_type = fusion_type

        if fusion_type == 'concat':
            # Simple concatenation followed by a 1x1 conv to maintain channel count
            self.fusion_conv = ConvBNAct(
                channels * 2,  # Combined channels from RGB and IR
                channels,  # Output same number of channels
                kernel_size=1,
                stride=1,
                use_act=True
            )
        elif fusion_type == 'add':
            # Element-wise addition (no parameters needed)
            self.fusion_conv = nn.Identity()
        elif fusion_type == 'attention':
            # Channel attention-based fusion
            self.rgb_attention = EseModule(channels)
            self.ir_attention = EseModule(channels)
            self.fusion_conv = ConvBNAct(
                channels * 2,
                channels,
                kernel_size=1,
                stride=1,
                use_act=True
            )
        else:
            raise ValueError(f"Unsupported fusion type: {fusion_type}")

    def forward(self, rgb_feat, ir_feat):
        if self.fusion_type == 'concat':
            # Simple concatenation followed by projection
            fused = torch.cat([rgb_feat, ir_feat], dim=1)
            return self.fusion_conv(fused)
        elif self.fusion_type == 'add':
            # Element-wise addition
            return rgb_feat + ir_feat
        elif self.fusion_type == 'attention':
            # Apply channel attention to each modality
            rgb_feat = self.rgb_attention(rgb_feat)
            ir_feat = self.ir_attention(ir_feat)
            # Concatenate and project
            fused = torch.cat([rgb_feat, ir_feat], dim=1)
            return self.fusion_conv(fused)

class LearnableAffineBlock(nn.Module):
    def __init__(
            self,
            scale_value=1.0,
            bias_value=0.0
    ):
        super().__init__()
        self.scale = nn.Parameter(torch.tensor([scale_value]), requires_grad=True)
        self.bias = nn.Parameter(torch.tensor([bias_value]), requires_grad=True)

    def forward(self, x):
        return self.scale * x + self.bias


class ConvBNAct(nn.Module):
    def __init__(
            self,
            in_chs,
            out_chs,
            kernel_size,
            stride=1,
            groups=1,
            padding='',
            use_act=True,
            use_lab=False
    ):
        super().__init__()
        self.use_act = use_act
        self.use_lab = use_lab
        if padding == 'same':
            self.conv = nn.Sequential(
                nn.ZeroPad2d([0, 1, 0, 1]),
                nn.Conv2d(
                    in_chs,
                    out_chs,
                    kernel_size,
                    stride,
                    groups=groups,
                    bias=False
                )
            )
        else:
            self.conv = nn.Conv2d(
                in_chs,
                out_chs,
                kernel_size,
                stride,
                padding=(kernel_size - 1) // 2,
                groups=groups,
                bias=False
            )
        self.bn = nn.BatchNorm2d(out_chs)
        if self.use_act:
            self.act = nn.ReLU()
        else:
            self.act = nn.Identity()
        if self.use_act and self.use_lab:
            self.lab = LearnableAffineBlock()
        else:
            self.lab = nn.Identity()

    def forward(self, x):
        x = self.conv(x)
        x = self.bn(x)
        x = self.act(x)
        x = self.lab(x)
        return x


class LightConvBNAct(nn.Module):
    def __init__(
            self,
            in_chs,
            out_chs,
            kernel_size,
            groups=1,
            use_lab=False,
    ):
        super().__init__()
        self.conv1 = ConvBNAct(
            in_chs,
            out_chs,
            kernel_size=1,
            use_act=False,
            use_lab=use_lab,
        )
        self.conv2 = ConvBNAct(
            out_chs,
            out_chs,
            kernel_size=kernel_size,
            groups=out_chs,
            use_act=True,
            use_lab=use_lab,
        )

    def forward(self, x):
        x = self.conv1(x)
        x = self.conv2(x)
        return x


class StemBlock(nn.Module):
    # for HGNetv2
    def __init__(self, in_chs, mid_chs, out_chs, use_lab=False):
        super().__init__()
        self.stem1 = ConvBNAct(
            in_chs,
            mid_chs,
            kernel_size=3,
            stride=2,
            use_lab=use_lab,
        )
        self.stem2a = ConvBNAct(
            mid_chs,
            mid_chs // 2,
            kernel_size=2,
            stride=1,
            use_lab=use_lab,
        )
        self.stem2b = ConvBNAct(
            mid_chs // 2,
            mid_chs,
            kernel_size=2,
            stride=1,
            use_lab=use_lab,
        )
        self.stem3 = ConvBNAct(
            mid_chs * 2,
            mid_chs,
            kernel_size=3,
            stride=2,
            use_lab=use_lab,
        )
        self.stem4 = ConvBNAct(
            mid_chs,
            out_chs,
            kernel_size=1,
            stride=1,
            use_lab=use_lab,
        )
        self.pool = nn.MaxPool2d(kernel_size=2, stride=1, ceil_mode=True)

    def forward(self, x):
        x = self.stem1(x)
        x = F.pad(x, (0, 1, 0, 1))
        x2 = self.stem2a(x)
        x2 = F.pad(x2, (0, 1, 0, 1))
        x2 = self.stem2b(x2)
        x1 = self.pool(x)
        x = torch.cat([x1, x2], dim=1)
        x = self.stem3(x)
        x = self.stem4(x)
        return x


class EseModule(nn.Module):
    def __init__(self, chs):
        super().__init__()
        self.conv = nn.Conv2d(
            chs,
            chs,
            kernel_size=1,
            stride=1,
            padding=0,
        )
        self.sigmoid = nn.Sigmoid()

    def forward(self, x):
        identity = x
        x = x.mean((2, 3), keepdim=True)
        x = self.conv(x)
        x = self.sigmoid(x)
        return torch.mul(identity, x)


class HG_Block(nn.Module):
    def __init__(
            self,
            in_chs,
            mid_chs,
            out_chs,
            layer_num,
            kernel_size=3,
            residual=False,
            light_block=False,
            use_lab=False,
            agg='ese',
            drop_path=0.,
    ):
        super().__init__()
        self.residual = residual

        self.layers = nn.ModuleList()
        for i in range(layer_num):
            if light_block:
                self.layers.append(
                    LightConvBNAct(
                        in_chs if i == 0 else mid_chs,
                        mid_chs,
                        kernel_size=kernel_size,
                        use_lab=use_lab,
                    )
                )
            else:
                self.layers.append(
                    ConvBNAct(
                        in_chs if i == 0 else mid_chs,
                        mid_chs,
                        kernel_size=kernel_size,
                        stride=1,
                        use_lab=use_lab,
                    )
                )

        # feature aggregation
        total_chs = in_chs + layer_num * mid_chs
        if agg == 'se':
            aggregation_squeeze_conv = ConvBNAct(
                total_chs,
                out_chs // 2,
                kernel_size=1,
                stride=1,
                use_lab=use_lab,
            )
            aggregation_excitation_conv = ConvBNAct(
                out_chs // 2,
                out_chs,
                kernel_size=1,
                stride=1,
                use_lab=use_lab,
            )
            self.aggregation = nn.Sequential(
                aggregation_squeeze_conv,
                aggregation_excitation_conv,
            )
        else:
            aggregation_conv = ConvBNAct(
                total_chs,
                out_chs,
                kernel_size=1,
                stride=1,
                use_lab=use_lab,
            )
            att = EseModule(out_chs)
            self.aggregation = nn.Sequential(
                aggregation_conv,
                att,
            )

        self.drop_path = nn.Dropout(drop_path) if drop_path else nn.Identity()

    def forward(self, x):
        identity = x
        output = [x]
        for layer in self.layers:
            x = layer(x)
            output.append(x)
        x = torch.cat(output, dim=1)
        x = self.aggregation(x)
        if self.residual:
            x = self.drop_path(x) + identity
        return x


class HG_Stage(nn.Module):
    def __init__(
            self,
            in_chs,
            mid_chs,
            out_chs,
            block_num,
            layer_num,
            downsample=True,
            light_block=False,
            kernel_size=3,
            use_lab=False,
            agg='se',
            drop_path=0.,
    ):
        super().__init__()
        self.downsample = downsample
        if downsample:
            self.downsample = ConvBNAct(
                in_chs,
                in_chs,
                kernel_size=3,
                stride=2,
                groups=in_chs,
                use_act=False,
                use_lab=use_lab,
            )
        else:
            self.downsample = nn.Identity()

        blocks_list = []
        for i in range(block_num):
            blocks_list.append(
                HG_Block(
                    in_chs if i == 0 else out_chs,
                    mid_chs,
                    out_chs,
                    layer_num,
                    residual=False if i == 0 else True,
                    kernel_size=kernel_size,
                    light_block=light_block,
                    use_lab=use_lab,
                    agg=agg,
                    drop_path=drop_path[i] if isinstance(drop_path, (list, tuple)) else drop_path,
                )
            )
        self.blocks = nn.Sequential(*blocks_list)

    def forward(self, x):
        x = self.downsample(x)
        x = self.blocks(x)
        return x



@register()
class HGNetv2(nn.Module):
    """
    HGNetV2 with RGB-IR fusion capability
    Args:
        name: str. Model size ('B0', 'B1', etc.)
        extra_channels: int. Number of IR channels (default: 1)
        use_lab: boolean. Whether to use LearnableAffineBlock in network.
        return_idx: list. Indices of stages to return outputs from.
        freeze_stem_only: boolean. Whether to only freeze the stem.
        freeze_at: int. Freeze stages up to this index.
        freeze_norm: boolean. Whether to freeze normalization layers.
        pretrained: boolean. Whether to load pretrained weights.
        local_model_dir: str. Directory to store/load pretrained models.
        fusion_type: str. Type of fusion to use ('concat', 'add', or 'attention').
    """

    arch_configs = {
        'B0': {
            'stem_channels': [3, 16, 16],
            'stage_config': {
                # in_channels, mid_channels, out_channels, num_blocks, downsample, light_block, kernel_size, layer_num
                "stage1": [16, 16, 64, 1, False, False, 3, 3],
                "stage2": [64, 32, 256, 1, True, False, 3, 3],
                "stage3": [256, 64, 512, 2, True, True, 5, 3],
                "stage4": [512, 128, 1024, 1, True, True, 5, 3],
            },
            'url': 'https://github.com/Peterande/storage/releases/download/dfinev1.0/PPHGNetV2_B0_stage1.pth'
        },
        # Other configurations remain the same...
    }

    def __init__(self,
                 name,
                 extra_channels=1,
                 use_lab=False,
                 return_idx=[1, 2, 3],
                 freeze_stem_only=True,
                 freeze_at=0,
                 freeze_norm=True,
                 pretrained=True,
                 local_model_dir='weight/hgnetv2/',
                 fusion_type='attention'):
        super().__init__()
        self.use_lab = use_lab
        self.return_idx = return_idx

        stem_channels = self.arch_configs[name]['stem_channels']
        stage_config = self.arch_configs[name]['stage_config']
        download_url = self.arch_configs[name]['url']

        self._out_strides = [4, 8, 16, 32]
        self._out_channels = [stage_config[k][2] for k in stage_config]

        # stem
        self.half_mid_channel = int(stem_channels[1] / 2)
        self.half_out_channels = int(stem_channels[2] / 2)

        # Create separate stem blocks for RGB and IR
        self.stem_rgb = StemBlock(
            in_chs=stem_channels[0],
            mid_chs=self.half_mid_channel,
            out_chs=self.half_out_channels,
            use_lab=use_lab)

        self.stem_ir = StemBlock(
            in_chs=extra_channels,
            mid_chs=self.half_mid_channel,
            out_chs=self.half_out_channels,
            use_lab=use_lab)

        # Create fusion module for RGB and IR features
        self.fusion_module = FusionModule(
            channels=stem_channels[2],
            fusion_type=fusion_type
        )

        # stages
        self.stages = nn.ModuleList()
        for i, k in enumerate(stage_config):
            in_channels, mid_channels, out_channels, block_num, downsample, light_block, kernel_size, layer_num = \
                stage_config[k]
            self.stages.append(
                HG_Stage(
                    in_channels,
                    mid_channels,
                    out_channels,
                    block_num,
                    layer_num,
                    downsample,
                    light_block,
                    kernel_size,
                    use_lab))

        if freeze_at >= 0:
            self._freeze_parameters(self.stem_rgb)
            self._freeze_parameters(self.stem_ir)
            if not freeze_stem_only:
                for i in range(min(freeze_at + 1, len(self.stages))):
                    self._freeze_parameters(self.stages[i])

        if freeze_norm:
            self._freeze_norm(self)

        if pretrained:
            self._load_pretrained(name, download_url, local_model_dir)

    def _load_pretrained(self, name, download_url, local_model_dir):
        """Helper method to load pretrained weights"""
        RED, GREEN, RESET = "\033[91m", "\033[92m", "\033[0m"
        try:
            # Ensure the directory exists
            os.makedirs(local_model_dir, exist_ok=True)

            # Construct full model path
            model_filename = f'PPHGNetV2_{name}_stage1.pth'
            model_path = os.path.join(local_model_dir, model_filename)

            # Rank 0 handles the download
            if torch.distributed.get_rank() == 0:
                if not os.path.exists(model_path):
                    print(
                        GREEN + "If the pretrained HGNetV2 can't be downloaded automatically. Please check your network connection." + RESET)
                    print(
                        GREEN + "Please check your network connection. Or download the model manually from " + RESET + f"{download_url}" + GREEN + " to " + RESET + f"{local_model_dir}." + RESET)

                    # Download and explicitly save the file
                    state = torch.hub.load_state_dict_from_url(
                        download_url,
                        map_location='cpu',
                        model_dir=local_model_dir,
                        check_hash=True
                    )
                    torch.save(state, model_path)
                    print(f"Saved pretrained weights to {model_path}")

            # All ranks wait here
            torch.distributed.barrier()

            # All ranks load the file
            if not os.path.exists(model_path):
                raise FileNotFoundError(f"Pretrained weights not found at {model_path}")

            state = torch.load(model_path, map_location='cpu', weights_only=True)

            # Filter out only the stem_rgb parameters (we can't use IR weights from the pretrained model)
            model_dict = self.state_dict()
            pretrained_dict = {k: v for k, v in state.items() if k in model_dict and 'stem_ir' not in k}

            # Update stem_ir with corresponding weights from stem_rgb
            for k in state:
                if 'stem.' in k:
                    ir_key = k.replace('stem.', 'stem_ir.')
                    if ir_key in model_dict:
                        # Skip copying weights for the first convolution which has different input channels
                        if 'stem1.conv' not in ir_key:
                            pretrained_dict[ir_key] = state[k]

            model_dict.update(pretrained_dict)
            self.load_state_dict(model_dict, strict=False)
            print(
                f"Loaded stage1 {name} HGNetV2 from {'local file' if torch.distributed.get_rank() != 0 else 'URL'}")

        except (Exception, KeyboardInterrupt) as e:
            if torch.distributed.get_rank() == 0:
                print(f"{str(e)}")
                logging.error(RED + "CRITICAL WARNING: Failed to load pretrained HGNetV2 model" + RESET)
                logging.error(GREEN + "Please check your network connection. Or download the model manually from " \
                              + RESET + f"{download_url}" + GREEN + " to " + RESET + f"{local_model_dir}." + RESET)
            torch.distributed.barrier()  # Ensure all processes exit
            raise RuntimeError("Failed to load pretrained weights") from e

    def _freeze_norm(self, m: nn.Module):
        if isinstance(m, nn.BatchNorm2d):
            m = FrozenBatchNorm2d(m.num_features)
        else:
            for name, child in m.named_children():
                _child = self._freeze_norm(child)
                if _child is not child:
                    setattr(m, name, _child)
        return m

    def _freeze_parameters(self, m: nn.Module):
        for p in m.parameters():
            p.requires_grad = False

    def forward(self, x):
        """
        Forward pass with RGB-IR fusion
        x: Input tensor with RGB+IR channels concatenated along channel dimension
        """
        # Split input into RGB and IR channels
        rgb_input = x[:, :3]
        ir_input = x[:, 3:] if x.size(1) > 3 else None

        # Process RGB through its stem block
        rgb_feat = self.stem_rgb(rgb_input)

        # Process IR through its stem block if IR channels exist
        if ir_input is not None and ir_input.size(1) > 0:
            ir_feat = self.stem_ir(ir_input)
        else:
            # Create a tensor of zeros matching the shape of rgb_feat
            ir_feat = torch.zeros_like(rgb_feat)

        # Fuse RGB and IR features
        x = self.fusion_module(rgb_feat, ir_feat)

        # Continue with the rest of the network
        outs = []
        for idx, stage in enumerate(self.stages):
            x = stage(x)
            if idx in self.return_idx:
                outs.append(x)
        return outs
