import torch
import torch.nn as nn
from typing import Union, Tuple, Optional, List
from timm.models.layers import trunc_normal_, DropPath

from nndepth.blocks.attn_block import LinearSelfAttention
from nndepth.blocks.conv import MobileOneBlock, RepLargeKernelConv


class ConvFFN(nn.Module):
    """Convolutional FFN Module."""

    def __init__(
        self,
        in_channels: int,
        hidden_channels: Optional[int] = None,
        out_channels: Optional[int] = None,
        act_layer: nn.Module = nn.GELU,
        drop: float = 0.0,
    ) -> None:
        """Build convolutional FFN module.

        Args:
            in_channels: Number of input channels.
            hidden_channels: Number of channels after expansion. Default: None
            out_channels: Number of output channels. Default: None
            act_layer: Activation layer. Default: ``GELU``
            drop: Dropout rate. Default: ``0.0``.
        """
        super().__init__()
        out_channels = out_channels or in_channels
        hidden_channels = hidden_channels or in_channels
        self.conv = nn.Sequential()
        self.conv.add_module(
            "conv",
            nn.Conv2d(
                in_channels=in_channels,
                out_channels=out_channels,
                kernel_size=7,
                padding=3,
                groups=in_channels,
                bias=False,
            ),
        )
        self.conv.add_module("bn", nn.BatchNorm2d(num_features=out_channels))
        self.fc1 = nn.Conv2d(in_channels, hidden_channels, kernel_size=1)
        self.act = act_layer()
        self.fc2 = nn.Conv2d(hidden_channels, out_channels, kernel_size=1)
        self.drop = nn.Dropout(drop)
        self.apply(self._init_weights)

    def _init_weights(self, m: nn.Module) -> None:
        if isinstance(m, nn.Conv2d):
            trunc_normal_(m.weight, std=0.02)
            if m.bias is not None:
                nn.init.constant_(m.bias, 0)

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        x = self.conv(x)
        x = self.fc1(x)
        x = self.act(x)
        x = self.drop(x)
        x = self.fc2(x)
        x = self.drop(x)
        return x


class ChannelMixer(nn.Module):
    """Convolutional FFN Module."""

    def __init__(
        self,
        in_channels: int,
        hidden_channels: Optional[int] = None,
        out_channels: Optional[int] = None,
        act_layer: nn.Module = nn.GELU,
        drop: float = 0.0,
    ) -> None:
        """Build convolutional FFN module.

        Args:
            in_channels: Number of input channels.
            hidden_channels: Number of channels after expansion. Default: None
            out_channels: Number of output channels. Default: None
            act_layer: Activation layer. Default: ``GELU``
            drop: Dropout rate. Default: ``0.0``.
        """
        super().__init__()
        out_channels = out_channels or in_channels
        hidden_channels = hidden_channels or in_channels

        self.fc1 = nn.Conv2d(in_channels, hidden_channels, kernel_size=1)
        self.act = act_layer()
        self.fc2 = nn.Conv2d(hidden_channels, out_channels, kernel_size=1)
        self.drop = nn.Dropout(drop)
        self.apply(self._init_weights)

    def _init_weights(self, m: nn.Module) -> None:
        if isinstance(m, nn.Conv2d):
            trunc_normal_(m.weight, std=0.02)
            if m.bias is not None:
                nn.init.constant_(m.bias, 0)

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        x = self.fc1(x)
        x = self.act(x)
        x = self.drop(x)
        x = self.fc2(x)
        x = self.drop(x)
        return x


class RepTokenMixer(nn.Module):
    """Reparameterizable token mixer.

    For more details, please refer to our paper:
    `FastViT: A Fast Hybrid Vision Transformer using Structural Reparameterization
    <https://arxiv.org/pdf/2303.14189.pdf>`_
    """

    def __init__(
        self,
        dim,
        kernel_size=3,
        use_layer_scale=True,
        layer_scale_init_value=1e-5,
        inference_mode: bool = False,
    ):
        """Build RepMixer Module.

        Args:
            dim: Input feature map dimension. :math:`C_{in}` from an expected input of size :math:`(B, C_{in}, H, W)`.
            kernel_size: Kernel size for spatial mixing. Default: 3
            use_layer_scale: If True, learnable layer scale is used. Default: ``True``
            layer_scale_init_value: Initial value for layer scale. Default: 1e-5
            inference_mode: If True, instantiates model in inference mode. Default: ``False``
        """
        super().__init__()
        self.dim = dim
        self.kernel_size = kernel_size
        self.inference_mode = inference_mode

        if inference_mode:
            self.reparam_conv = nn.Conv2d(
                in_channels=self.dim,
                out_channels=self.dim,
                kernel_size=self.kernel_size,
                stride=1,
                padding=self.kernel_size // 2,
                groups=self.dim,
                bias=True,
            )
        else:
            self.norm = MobileOneBlock(
                dim,
                dim,
                kernel_size,
                padding=kernel_size // 2,
                groups=dim,
                use_act=False,
                use_scale_branch=False,
                num_conv_branches=0,
            )
            self.mixer = MobileOneBlock(
                dim,
                dim,
                kernel_size,
                padding=kernel_size // 2,
                groups=dim,
                use_act=False,
            )
            self.use_layer_scale = use_layer_scale
            if use_layer_scale:
                self.layer_scale = nn.Parameter(layer_scale_init_value * torch.ones((dim, 1, 1)), requires_grad=True)

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        if hasattr(self, "reparam_conv"):
            x = self.reparam_conv(x)
            return x
        else:
            if self.use_layer_scale:
                x = x + self.layer_scale * (self.mixer(x) - self.norm(x))
            else:
                x = x + self.mixer(x) - self.norm(x)
            return x

    def reparameterize(self) -> None:
        """Reparameterize mixer and norm into a single
        convolutional layer for efficient inference.
        """
        if self.inference_mode:
            return

        self.mixer.reparameterize()
        self.norm.reparameterize()

        if self.use_layer_scale:
            w = self.mixer.id_tensor + self.layer_scale.unsqueeze(-1) * (
                self.mixer.reparam_conv.weight - self.norm.reparam_conv.weight
            )
            b = torch.squeeze(self.layer_scale) * (self.mixer.reparam_conv.bias - self.norm.reparam_conv.bias)
        else:
            w = self.mixer.id_tensor + self.mixer.reparam_conv.weight - self.norm.reparam_conv.weight
            b = self.mixer.reparam_conv.bias - self.norm.reparam_conv.bias

        self.reparam_conv = nn.Conv2d(
            in_channels=self.dim,
            out_channels=self.dim,
            kernel_size=self.kernel_size,
            stride=1,
            padding=self.kernel_size // 2,
            groups=self.dim,
            bias=True,
        )
        self.reparam_conv.weight.data = w
        self.reparam_conv.bias.data = b

        for para in self.parameters():
            para.detach_()
        self.__delattr__("mixer")
        self.__delattr__("norm")
        if self.use_layer_scale:
            self.__delattr__("layer_scale")


class RepFormerBlock(nn.Module):
    """Implementation of Metaformer block with RepMixer as token mixer.

    For more details on Metaformer structure, please refer to:
    `MetaFormer Is Actually What You Need for Vision <https://arxiv.org/pdf/2111.11418.pdf>`_
    """

    def __init__(
        self,
        dim: int,
        kernel_size: int = 3,
        use_ffn: bool = True,
        ffn_exp_ratio: float = 4.0,
        act_layer: nn.Module = nn.GELU,
        drop: float = 0.0,
        drop_path: float = 0.0,
        use_layer_scale: bool = True,
        layer_scale_init_value: float = 1e-5,
        inference_mode: bool = False,
    ):
        """Build RepTransformer Block.

        Args:
            dim: Number of embedding dimensions.
            kernel_size: Kernel size for repmixer. Default: 3
            ffn_exp_ratio: FFN expansion ratio. Default: 4.0
            act_layer: Activation layer. Default: ``nn.GELU``
            drop: Dropout rate. Default: 0.0
            drop_path: Drop path rate. Default: 0.0
            use_layer_scale: Flag to turn on layer scale. Default: ``True``
            layer_scale_init_value: Layer scale value at initialization. Default: 1e-5
            inference_mode: Flag to instantiate block in inference mode. Default: ``False``
        """

        super().__init__()

        self.token_mixer = RepTokenMixer(
            dim,
            kernel_size=kernel_size,
            use_layer_scale=use_layer_scale,
            layer_scale_init_value=layer_scale_init_value,
            inference_mode=inference_mode,
        )

        self.use_ffn = use_ffn
        assert ffn_exp_ratio > 0, "MLP ratio should be greater than 0, found: {}".format(ffn_exp_ratio)
        mlp_hidden_dim = int(dim * ffn_exp_ratio)

        if use_ffn:
            self.convffn = ChannelMixer(
                in_channels=dim,
                hidden_channels=mlp_hidden_dim,
                act_layer=act_layer,
                drop=drop,
            )

            # Drop Path
            self.drop_path = DropPath(drop_path) if drop_path > 0.0 else nn.Identity()

            # Layer Scale
            self.use_layer_scale = use_layer_scale
            if use_layer_scale:
                self.layer_scale = nn.Parameter(layer_scale_init_value * torch.ones((dim, 1, 1)), requires_grad=True)

    def forward(self, x):
        if self.use_ffn:
            if self.use_layer_scale:
                x = self.token_mixer(x)
                x = x + self.drop_path(self.layer_scale * self.convffn(x))
            else:
                x = self.token_mixer(x)
                x = x + self.drop_path(self.convffn(x))
        else:
            x = self.token_mixer(x)

        return x


class ConvPatchEmbed(nn.Module):
    """Convolutional patch embedding layer."""

    def __init__(
        self,
        patch_size: int,
        stride: int,
        in_channels: int,
        embed_dim: int,
        act_layer: nn.Module = nn.GELU,
        inference_mode: bool = False,
    ) -> None:
        """Build patch embedding layer.

        Args:
            patch_size: Patch size for embedding computation.
            stride: Stride for convolutional embedding layer.
            in_channels: Number of channels of input tensor.
            embed_dim: Number of embedding dimensions.
            inference_mode: Flag to instantiate model in inference mode. Default: ``False``
        """
        super().__init__()
        block = list()
        block.append(
            RepLargeKernelConv(
                in_channels=in_channels,
                out_channels=in_channels,
                kernel_size=patch_size,
                stride=stride,
                groups=in_channels,
                small_kernel=3,
                inference_mode=inference_mode,
                activation=act_layer(),
            )
        )
        block.append(
            MobileOneBlock(
                in_channels=in_channels,
                out_channels=embed_dim,
                kernel_size=1,
                stride=1,
                padding=0,
                groups=1,
                activation=act_layer(),
                inference_mode=inference_mode,
                use_se=False,
                num_conv_branches=1,
            )
        )
        self.proj = nn.Sequential(*block)

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        x = self.proj(x)
        return x


class AttentionBlock(nn.Module):
    """Implementation of metaformer block with MHSA as token mixer.

    For more details on Metaformer structure, please refer to:
    `MetaFormer Is Actually What You Need for Vision <https://arxiv.org/pdf/2111.11418.pdf>`_
    """

    def __init__(
        self,
        dim: int,
        ffn_exp_ratio: float = 4.0,
        act_layer: nn.Module = nn.GELU,
        norm_layer: nn.Module = nn.BatchNorm2d,
        drop: float = 0.0,
        drop_path: float = 0.0,
        use_layer_scale: bool = True,
        layer_scale_init_value: float = 1e-5,
    ):
        """Build Attention Block.

        Args:
            dim: Number of embedding dimensions.
            ffn_exp_ratio: FFN expansion ratio. Default: 4.0
            act_layer: Activation layer. Default: ``nn.GELU``
            norm_layer: Normalization layer. Default: ``nn.BatchNorm2d``
            drop: Dropout rate. Default: 0.0
            drop_path: Drop path rate. Default: 0.0
            use_layer_scale: Flag to turn on layer scale. Default: ``True``
            layer_scale_init_value: Layer scale value at initialization. Default: 1e-5
        """

        super().__init__()

        self.norm = norm_layer(dim)
        self.token_mixer = LinearSelfAttention(dim=dim)

        assert ffn_exp_ratio > 0, "MLP ratio should be greater than 0, found: {}".format(ffn_exp_ratio)
        mlp_hidden_dim = int(dim * ffn_exp_ratio)

        self.convffn = ChannelMixer(
            in_channels=dim,
            hidden_channels=mlp_hidden_dim,
            act_layer=act_layer,
            drop=drop,
        )

        # Drop path
        self.drop_path = DropPath(drop_path) if drop_path > 0.0 else nn.Identity()

        # Layer Scale
        self.use_layer_scale = use_layer_scale
        if use_layer_scale:
            self.layer_scale_1 = nn.Parameter(layer_scale_init_value * torch.ones((dim, 1, 1)), requires_grad=True)
            self.layer_scale_2 = nn.Parameter(layer_scale_init_value * torch.ones((dim, 1, 1)), requires_grad=True)

    def forward(self, x):
        if self.use_layer_scale:
            x = x + self.drop_path(self.layer_scale_1 * self.token_mixer(self.norm(x)))
            x = x + self.drop_path(self.layer_scale_2 * self.convffn(x))
        else:
            x = x + self.drop_path(self.token_mixer(self.norm(x)))
            x = x + self.drop_path(self.convffn(x))
        return x


def convolutional_stem(
    in_channels: int,
    out_channels: int,
    inference_mode: bool = False,
    act_layer: nn.Module = nn.GELU,
    strides: Union[List[int], List[Tuple[int, int]]] = [2, 2, 1],
) -> nn.Sequential:
    """Build convolutional stem with MobileOne blocks.

    Args:
        in_channels: Number of input channels.
        out_channels: Number of output channels.
        inference_mode: Flag to instantiate model in inference mode. Default: ``False``

    Returns:
        nn.Sequential object with stem elements.
    """
    return nn.Sequential(
        MobileOneBlock(
            in_channels=in_channels,
            out_channels=out_channels,
            kernel_size=3,
            stride=strides[0],
            padding=1,
            groups=1,
            activation=act_layer(),
            inference_mode=inference_mode,
            use_se=False,
            num_conv_branches=1,
        ),
        MobileOneBlock(
            in_channels=out_channels,
            out_channels=out_channels,
            kernel_size=3,
            stride=strides[1],
            padding=1,
            groups=out_channels,
            activation=act_layer(),
            inference_mode=inference_mode,
            use_se=False,
            num_conv_branches=1,
        ),
        MobileOneBlock(
            in_channels=out_channels,
            out_channels=out_channels,
            kernel_size=1,
            stride=strides[2],
            padding=0,
            groups=1,
            activation=act_layer(),
            inference_mode=inference_mode,
            use_se=False,
            num_conv_branches=1,
        ),
    )


class RepViT(nn.Module):
    BASE_NUM_CHANNELS = [32, 64, 128, 256]

    def __init__(
        self,
        in_channels: int = 3,
        patch_size: int = 7,
        stem_strides: Union[List[int], List[Tuple[int, int]]] = [(2, 2), (2, 2), (1, 1)],
        num_blocks_per_stage: List[int] = [4, 4, 6, 2],
        width_multipliers: List[int] = [1, 1, 1, 1],
        use_ffn_per_stage: List[bool] = [False, True, True, True],
        ffn_exp_ratios: List[float] = [1.0, 3.0, 3.0, 4.0],
        downsample_ratios: Union[List[Tuple[int, int]], List[List[int]]] = [(2, 2), (2, 2), (2, 2), (2, 2)],
        token_mixer_types: List[str] = ["repmixer", "repmixer", "repmixer", "attention"],
        drop_rate: float = 0.0,
        drop_path_rate: float = 0.0,
        use_layer_scale: bool = True,
        layer_scale_init_value: float = 1e-5,
        act_layer: torch.nn.Module = nn.GELU,
        inference_mode: bool = False,
        **kwargs,
    ):
        super().__init__()
        assert len(num_blocks_per_stage) == len(width_multipliers)
        self.num_blocks_per_stage = num_blocks_per_stage
        self.width_multipliers = width_multipliers

        self.stem = convolutional_stem(
            in_channels, 16, inference_mode=inference_mode, act_layer=act_layer, strides=stem_strides
        )

        self.stage_0 = nn.Sequential(
            ConvPatchEmbed(
                patch_size=patch_size,
                stride=downsample_ratios[0],
                in_channels=16,
                embed_dim=int(self.BASE_NUM_CHANNELS[0] * width_multipliers[0]),
                inference_mode=inference_mode,
                act_layer=act_layer,
            ),
            self._make_stage(
                channels=int(self.BASE_NUM_CHANNELS[0] * width_multipliers[0]),
                block_index=0,
                num_blocks=num_blocks_per_stage,
                token_mixer_type=token_mixer_types[0],
                kernel_size=3,
                use_ffn=use_ffn_per_stage[0],
                ffn_exp_ratio=ffn_exp_ratios[0],
                drop_rate=drop_rate,
                drop_path_rate=drop_path_rate,
                use_layer_scale=use_layer_scale,
                layer_scale_init_value=layer_scale_init_value,
                inference_mode=inference_mode,
                act_layer=act_layer,
            ),
        )

        self.stage_1 = nn.Sequential(
            ConvPatchEmbed(
                patch_size=patch_size,
                stride=downsample_ratios[1],
                in_channels=self.num_channels,
                embed_dim=int(self.BASE_NUM_CHANNELS[1] * width_multipliers[1]),
                inference_mode=inference_mode,
                act_layer=act_layer,
            ),
            self._make_stage(
                channels=int(self.BASE_NUM_CHANNELS[1] * width_multipliers[1]),
                block_index=1,
                num_blocks=num_blocks_per_stage,
                token_mixer_type=token_mixer_types[1],
                use_ffn=use_ffn_per_stage[1],
                kernel_size=3,
                ffn_exp_ratio=ffn_exp_ratios[1],
                drop_rate=drop_rate,
                drop_path_rate=drop_path_rate,
                use_layer_scale=use_layer_scale,
                layer_scale_init_value=layer_scale_init_value,
                inference_mode=inference_mode,
                act_layer=act_layer,
            ),
        )

        self.stage_2 = nn.Sequential(
            ConvPatchEmbed(
                patch_size=patch_size,
                stride=downsample_ratios[2],
                in_channels=self.num_channels,
                embed_dim=int(self.BASE_NUM_CHANNELS[2] * width_multipliers[2]),
                act_layer=act_layer,
                inference_mode=inference_mode,
            ),
            self._make_stage(
                channels=int(self.BASE_NUM_CHANNELS[2] * width_multipliers[2]),
                block_index=2,
                num_blocks=num_blocks_per_stage,
                token_mixer_type=token_mixer_types[2],
                kernel_size=3,
                use_ffn=use_ffn_per_stage[2],
                ffn_exp_ratio=ffn_exp_ratios[2],
                drop_rate=drop_rate,
                drop_path_rate=drop_path_rate,
                use_layer_scale=use_layer_scale,
                layer_scale_init_value=layer_scale_init_value,
                act_layer=act_layer,
                inference_mode=inference_mode,
            ),
        )

        self.stage_3 = nn.Sequential(
            ConvPatchEmbed(
                patch_size=patch_size,
                stride=downsample_ratios[3],
                in_channels=self.num_channels,
                embed_dim=int(self.BASE_NUM_CHANNELS[3] * width_multipliers[3]),
                act_layer=act_layer,
                inference_mode=inference_mode,
            ),
            self._make_stage(
                channels=int(self.BASE_NUM_CHANNELS[3] * width_multipliers[3]),
                block_index=3,
                num_blocks=num_blocks_per_stage,
                token_mixer_type=token_mixer_types[3],
                kernel_size=3,
                use_ffn=use_ffn_per_stage[3],
                ffn_exp_ratio=ffn_exp_ratios[3],
                drop_rate=drop_rate,
                drop_path_rate=drop_path_rate,
                use_layer_scale=use_layer_scale,
                layer_scale_init_value=layer_scale_init_value,
                act_layer=act_layer,
                inference_mode=inference_mode,
            ),
        )

    def _make_stage(
        self,
        channels: int,
        block_index: int,
        num_blocks: List[int],
        token_mixer_type: str,
        kernel_size: int = 3,
        use_ffn: bool = True,
        ffn_exp_ratio: float = 4.0,
        act_layer: nn.Module = nn.GELU,
        norm_layer: nn.Module = nn.BatchNorm2d,
        drop_rate: float = 0.0,
        drop_path_rate: float = 0.0,
        use_layer_scale: bool = True,
        layer_scale_init_value: float = 1e-5,
        inference_mode=False,
    ) -> nn.Sequential:
        """Build FastViT blocks within a stage.

        Args:
            channels: Number of embedding dimensions.
            block_index: block index.
            num_blocks: List containing number of blocks per stage.
            token_mixer_type: Token mixer type.
            kernel_size: Kernel size for repmixer.
            mlp_ratio: MLP expansion ratio.
            act_layer: Activation layer.
            norm_layer: Normalization layer.
            drop_rate: Dropout rate.
            drop_path_rate: Drop path rate.
            use_layer_scale: Flag to turn on layer scale regularization.
            layer_scale_init_value: Layer scale value at initialization.
            inference_mode: Flag to instantiate block in inference mode.

        Returns:
            nn.Sequential object of all the blocks within the stage.
        """
        blocks = []
        for block_idx in range(num_blocks[block_index]):
            block_dpr = drop_path_rate * (block_idx + sum(num_blocks[:block_index])) / (sum(num_blocks) - 1)
            if token_mixer_type == "repmixer":
                blocks.append(
                    RepFormerBlock(
                        channels,
                        kernel_size=kernel_size,
                        ffn_exp_ratio=ffn_exp_ratio,
                        use_ffn=use_ffn,
                        act_layer=act_layer,
                        drop=drop_rate,
                        drop_path=block_dpr,
                        use_layer_scale=use_layer_scale,
                        layer_scale_init_value=layer_scale_init_value,
                        inference_mode=inference_mode,
                    )
                )
            elif token_mixer_type == "attention":
                blocks.append(
                    AttentionBlock(
                        channels,
                        ffn_exp_ratio=ffn_exp_ratio,
                        act_layer=act_layer,
                        norm_layer=norm_layer,
                        drop=drop_rate,
                        drop_path=block_dpr,
                        use_layer_scale=use_layer_scale,
                        layer_scale_init_value=layer_scale_init_value,
                    )
                )
            else:
                raise ValueError("Token mixer type: {} not supported".format(token_mixer_type))
            self.num_channels = channels
        blocks = nn.Sequential(*blocks)

        return blocks

    def reparameterize_model(self):
        """Method returns a model where a multi-branched structure
            used in training is re-parameterized into a single branch
            for inference.

        :param model: MobileOne model in train mode.
        :return: MobileOne model in inference mode.
        """
        for module in self.modules():
            if hasattr(module, "reparameterize"):
                module.reparameterize()

    def forward(self, x: torch.Tensor):
        features = []

        for layer in [self.stem, self.stage_0, self.stage_1, self.stage_2, self.stage_3]:
            x = layer(x)
            features.append(x)

        return features


if __name__ == "__main__":
    model = RepViT()
    x = torch.randn(1, 3, 480, 640)
    y = model(x)
    print([t.shape for t in y[1:]])
