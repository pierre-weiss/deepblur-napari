r"""
Identification and noise networks of deepblur, shipped with the plugin until
they are available in deepinv.

The ``resnet`` models (z3_z36) are the :class:`DeepBlur` network of deepinv PR
#1494 (see ``_deepblur_net.py``). The other architectures are copied from the
deepblur package (``deepblur.model``), keeping only the code paths used by the
published weights. The parameter names are unchanged, so that the published
checkpoints load as they are.
"""

import re

import torch
import torch.nn as nn
import torch.nn.functional as F
import torchvision
import yaml

from ._deepblur_net import DeepBlur
from ._weights import get_weights_files

ID_MODELS = [
    "z3_z6",
    "z3_z10",
    "z3_z15",
    "z3_z21",
    "z3_z28",
    "z3_z36",
    "chromatic_z3_z36",
    "motion",
    "motion_x_z3_z16",
]


class ModelProperties(dict):
    r"""Training configuration of an identification model, with attribute access."""

    def __getattr__(self, name):
        try:
            return self[name]
        except KeyError:
            raise AttributeError(name) from None


def load_id_model(
    name: str, device: torch.device | str = "cpu"
) -> tuple[nn.Module, ModelProperties]:
    r"""
    Loads an identification model from the MAMBO weights server, or from the
    Hugging Face hub if the server can not provide it.

    :param str name: name of the model, one of :data:`ID_MODELS`.
    :param device: device to load the model onto.
    :return: the model, and its training configuration (``image_size`` is the
        size of the patches it was trained on, ``psf_size`` the size of the PSFs
        it estimates), with the source of the weights in ``weights_source``
        (see :func:`get_weights_files`).
    """
    paths, source = get_weights_files(
        [
            f"{name}/checkpoint.pth",
            f"{name}/model_config.yaml",
            f"{name}/training_config.yaml",
        ]
    )
    ckpt_path, model_config_path, training_config_path = paths
    with open(model_config_path) as f:
        model = build_id_model(yaml.safe_load(f))
    state_dict = torch.load(ckpt_path, map_location="cpu", weights_only=True)
    if isinstance(model, DeepBlur):
        state_dict = _to_deepblur_names(state_dict)
    model.load_state_dict(state_dict)

    with open(training_config_path) as f:
        properties = ModelProperties(yaml.safe_load(f))
    properties["weights_source"] = source
    return model.to(device).eval(), properties


def build_id_model(config: dict) -> nn.Module:
    r"""
    Builds an identification network from its ``model_config.yaml``, as
    ``deepblur.model.get_model_from_config``.
    """
    psf_size = config.get("psf_size", (51, 51))[0]
    norm_kwargs = config.get("norm_kwargs", {})
    model = config["model"].lower()

    if model == "resnet":
        return DeepBlur(psf_size=psf_size, pretrained=None)
    if model == "rescondnet":
        if config["inform_fc"] or config["inform_amp"]:
            raise ValueError("Conditional rescondnet models are not supported")
        return ResUnCondNet(
            size=config["model_size"], kernel_size=psf_size, **norm_kwargs
        )
    if model == "resnetsachromatic":
        return ResNetAttnIdNetChromatic(
            size=config["model_size"],
            psf_size=psf_size,
            activation=config["activation"],
            input_channels=config.get("input_channels", 1),
            stem_channels=config.get("stem_channels"),
            color_mode=config.get("color_mode", "grayscale"),
            psf_channels=config.get("psf_channels"),
            **norm_kwargs,
        )
    raise ValueError(f"Unsupported model type: {config['model']}")


def _to_deepblur_names(state_dict: dict) -> dict:
    r"""
    Renames the parameters of a deepblur ``ResNet`` checkpoint to those of
    :class:`DeepBlur`, which only differ by the containers of the residual
    blocks and of the mixing layer.
    """
    renamed = {}
    for name, value in state_dict.items():
        name = re.sub(r"^rb(\d+)\.", r"res_blocks.\1.", name)
        name = re.sub(r"^mixing\.1\.", "mixing.", name)
        renamed[name] = value
    return renamed


def softmax2d(x: torch.Tensor) -> torch.Tensor:
    r"""Softmax over the spatial dimensions."""
    shape = x.size()
    return F.softmax(x.view(*shape[:2], -1), dim=-1).view(shape)


def _norm_layer(
    channels: int,
    affine: bool,
    momentum: float = 0.1,
    track_running_stats: bool = True,
    use_bn: bool = True,
    use_gn: bool = False,
    use_in: bool = False,
    num_groups: int = 32,
) -> nn.Module:
    r"""Normalization layer selected by the ``norm_kwargs`` of the model configs."""
    if use_bn + use_gn + use_in > 1:
        raise ValueError("At most one normalization can be used")
    if use_bn:
        return nn.BatchNorm2d(
            channels,
            momentum=momentum,
            track_running_stats=track_running_stats,
        )
    if use_gn:
        return nn.GroupNorm(num_groups, channels)
    if use_in:
        return nn.InstanceNorm2d(
            channels,
            momentum=momentum,
            track_running_stats=track_running_stats,
            affine=affine,
        )
    return nn.Identity()


######################################################################################
#############          rescondnet (deepblur.model.identification)       ################
######################################################################################


class _UnCondResidualBlock(nn.Module):
    def __init__(self, in_channels, out_channels, stride=1, **norm_kwargs):
        super().__init__()
        self.conv1 = nn.Conv2d(in_channels, out_channels, 3, stride, padding=1)
        self.bn1 = _norm_layer(out_channels, affine=False, **norm_kwargs)
        self.conv2 = nn.Conv2d(out_channels, out_channels, 3, 1, padding=1)
        self.bn2 = _norm_layer(out_channels, affine=False, **norm_kwargs)
        self.shortcut = (
            nn.Conv2d(in_channels, out_channels, 1, stride)
            if stride > 1 or in_channels != out_channels
            else nn.Identity()
        )
        self.relu = nn.ReLU()

    def forward(self, x):
        identity = self.shortcut(x)
        out = self.relu(self.bn1(self.conv1(x)))
        out = self.bn2(self.conv2(out))
        out += identity
        return self.relu(out)


class _CandidatesToChannels(nn.Module):
    r"""``(B, h*w, 8, 8)`` -> ``(B, 64, h, w)``: one channel per candidate PSF."""

    def __init__(self, kernel_size):
        super().__init__()
        self.kernel_size = kernel_size

    def forward(self, x):
        b, _, c, d = x.shape
        k = self.kernel_size
        return x.reshape(b, k, k, c * d).permute(0, 3, 1, 2)


class ResUnCondNet(nn.Module):
    r"""
    Residual identification network, without conditioning (``rescondnet``
    models with ``inform_fc = inform_amp = 0``).

    :param int size: width multiplier of the network.
    :param int kernel_size: size of the estimated PSF.
    """

    def __init__(self, size: int = 1, kernel_size: int = 51, **norm_kwargs):
        super().__init__()

        # Encoder
        self.initial = nn.Conv2d(1, 32 * size, 7, 2, 3)
        self.early_fusion = nn.Conv2d(32 * size, 32 * size, 3, 1, 1)
        self.rb1 = _UnCondResidualBlock(32 * size, 64 * size, 2, **norm_kwargs)
        self.rb2 = _UnCondResidualBlock(
            64 * size, 128 * size, 2, **norm_kwargs
        )
        self.rb3 = _UnCondResidualBlock(
            128 * size, 256 * size, 2, **norm_kwargs
        )
        self.rb4 = _UnCondResidualBlock(
            256 * size, 256 * size, 2, **norm_kwargs
        )
        self.trans = nn.Conv2d(256 * size, 512 * size, 3, 1, 1)

        # Feature fusion
        self.fusion = nn.Conv2d(512 * size, 512 * size, 3, 1, 1)

        # Decoder
        self.up = nn.Upsample(scale_factor=2, mode="bilinear")
        self.dec1 = nn.Conv2d(512 * size + 256 * size, 384 * size, 3, 1, 1)
        self.dec2 = nn.Conv2d(384 * size, 256 * size, 3, 1, 1)
        self.final = nn.Conv2d(256 * size, kernel_size * kernel_size, 3, 2, 1)

        self.mixing = nn.Sequential(
            nn.AdaptiveAvgPool2d((8, 8)),
            _CandidatesToChannels(kernel_size),
            nn.Conv2d(64, 1, kernel_size=1),
        )

    def forward(self, x):
        x = self.initial(x)
        x = self.early_fusion(x)

        # Encoder
        r1 = self.rb1(x)
        r2 = self.rb2(r1)
        r3 = self.rb3(r2)
        r4 = self.rb4(r3)
        x = self.trans(r4)
        x = self.fusion(x)

        # Decoder
        x = self.up(x)
        x = torch.cat([x, r3], dim=1)
        x = self.dec1(x)
        x = self.dec2(x)
        x = self.final(x)

        x = self.mixing(x)
        return softmax2d(x)


######################################################################################
#############        resnetsachromatic (deepblur.model._resnet_id)      ################
######################################################################################


class _ConvBlock(nn.Module):
    r"""Conv -> Norm -> GELU"""

    def __init__(
        self,
        in_channels,
        out_channels,
        kernel_size,
        stride=1,
        padding=0,
        dilation=1,
        **norm_kwargs,
    ):
        super().__init__()
        norm = _norm_layer(out_channels, affine=True, **norm_kwargs)
        # no biases since there is a norm layer
        self.conv = nn.Conv2d(
            in_channels,
            out_channels,
            kernel_size,
            stride,
            padding,
            dilation,
            bias=isinstance(norm, nn.Identity),
        )
        self.norm = norm
        self.act = nn.GELU()

    def forward(self, x):
        return self.act(self.norm(self.conv(x)))


class _ResidualBlock(nn.Module):
    def __init__(self, in_channels, out_channels, stride=1, **norm_kwargs):
        super().__init__()
        self.conv1 = nn.Conv2d(
            in_channels, out_channels, 3, stride, padding=1, bias=False
        )
        self.norm1 = _norm_layer(out_channels, affine=True, **norm_kwargs)
        self.conv2 = nn.Conv2d(
            out_channels, out_channels, 3, 1, padding=1, bias=False
        )
        self.norm2 = _norm_layer(out_channels, affine=True, **norm_kwargs)
        self.shortcut = (
            nn.Conv2d(in_channels, out_channels, 1, stride, bias=False)
            if stride > 1 or in_channels != out_channels
            else nn.Identity()
        )
        self.act = nn.GELU()

    def forward(self, x):
        identity = self.shortcut(x)
        out = self.act(self.norm1(self.conv1(x)))
        out = self.norm2(self.conv2(out))
        out += identity
        return self.act(out)


def _sincos_pos_2d(h: int, w: int, dim: int, device, dtype) -> torch.Tensor:
    r"""2D sin-cos positional embedding ``(1, h*w, dim)``, parameter-free."""
    quarter = dim // 4
    omega = torch.arange(quarter, device=device, dtype=torch.float32) / quarter
    omega = 1.0 / (10000.0**omega)

    y = torch.arange(h, device=device, dtype=torch.float32)
    x = torch.arange(w, device=device, dtype=torch.float32)
    out_y = torch.einsum("i,j->ij", y, omega)
    out_x = torch.einsum("i,j->ij", x, omega)

    pos_y = torch.cat([out_y.sin(), out_y.cos()], dim=1)  # (h, dim/2)
    pos_x = torch.cat([out_x.sin(), out_x.cos()], dim=1)  # (w, dim/2)
    pos_y = pos_y[:, None, :].expand(h, w, dim // 2)
    pos_x = pos_x[None, :, :].expand(h, w, dim // 2)
    return torch.cat([pos_y, pos_x], dim=2).reshape(1, h * w, dim).to(dtype)


class _SelfAttentionBottleneck(nn.Module):
    r"""
    Dense self-attention bottleneck: a pre-norm transformer layer on the
    tokens of the feature map, added to the input as an outer residual.
    """

    def __init__(self, channels, num_heads=8, mlp_ratio=4, reduce_dim=256):
        super().__init__()
        dim = reduce_dim if reduce_dim is not None else channels
        self.proj_in = (
            nn.Conv2d(channels, dim, 1) if dim != channels else nn.Identity()
        )
        self.norm1 = nn.LayerNorm(dim)
        self.attn = nn.MultiheadAttention(dim, num_heads, batch_first=True)
        self.norm2 = nn.LayerNorm(dim)
        self.ffn = nn.Sequential(
            nn.Linear(dim, dim * mlp_ratio),
            nn.GELU(),
            nn.Linear(dim * mlp_ratio, dim),
        )
        self.proj_out = (
            nn.Conv2d(dim, channels, 1) if dim != channels else nn.Identity()
        )

    def forward(self, x):
        identity = x
        h = self.proj_in(x)
        b, c, hh, ww = h.shape
        tokens = h.flatten(2).permute(0, 2, 1)  # (B, H*W, dim)
        tokens = tokens + _sincos_pos_2d(hh, ww, c, x.device, tokens.dtype)

        a = self.norm1(tokens)
        a, _ = self.attn(a, a, a, need_weights=False)
        tokens = tokens + a
        tokens = tokens + self.ffn(self.norm2(tokens))

        out = tokens.permute(0, 2, 1).reshape(b, c, hh, ww)
        return identity + self.proj_out(out)


class _AttnPSFHead(nn.Module):
    r"""
    Attention PSF head: ``psf_h * psf_w`` learnable position queries
    cross-attend to the feature tokens, and a linear layer gives
    ``out_channels`` PSF values per query.
    """

    def __init__(
        self,
        in_channels,
        psf_size,
        embed_dim=256,
        num_heads=4,
        activation="softmax",
        out_channels=1,
    ):
        super().__init__()
        if activation != "softmax":
            raise ValueError(f"Unsupported activation: {activation!r}")
        self.psf_size = psf_size
        self.out_channels = out_channels
        num_queries = psf_size[0] * psf_size[1]

        self.kv_proj = nn.Conv2d(in_channels, embed_dim, kernel_size=1)
        self.queries = nn.Parameter(torch.randn(num_queries, embed_dim) * 0.02)
        self.mha = nn.MultiheadAttention(
            embed_dim=embed_dim, num_heads=num_heads, batch_first=True
        )
        self.norm1 = nn.LayerNorm(embed_dim)
        self.norm2 = nn.LayerNorm(embed_dim)
        self.ffn = nn.Sequential(
            nn.Linear(embed_dim, embed_dim * 2),
            nn.GELU(),
            nn.Linear(embed_dim * 2, embed_dim),
        )
        self.to_value = nn.Linear(embed_dim, out_channels)

    def forward(self, x):
        b = x.size(0)
        kv = self.kv_proj(x).flatten(2).permute(0, 2, 1)  # (B, H*W, embed_dim)
        q = self.queries.unsqueeze(0).expand(b, -1, -1)

        attn_out, _ = self.mha(query=q, key=kv, value=kv, need_weights=False)
        out = self.norm1(q + attn_out)
        out = self.norm2(out + self.ffn(out))

        out = self.to_value(out)  # (B, num_queries, out_channels)
        out = out.permute(0, 2, 1).reshape(
            b, self.out_channels, *self.psf_size
        )
        return softmax2d(out)


class ResNetAttnIdNetChromatic(nn.Module):
    r"""
    Residual encoder, self-attention bottleneck and attention PSF head, for
    inputs of 1 to 3 channels (one input adapter per number of channels).
    With ``color_mode='chromatic'``, it estimates one PSF per input channel.
    """

    def __init__(
        self,
        size: int = 2,
        psf_size: int = 71,
        activation: str = "softmax",
        input_channels: int | list[int] = 1,
        stem_channels: int | None = None,
        color_mode: str = "grayscale",
        psf_channels: int | None = None,
        **norm_kwargs,
    ):
        super().__init__()
        psf_size = (psf_size, psf_size)
        if isinstance(input_channels, int):
            input_channels = (input_channels,)
        input_channel_options = tuple(sorted(set(input_channels)))

        if color_mode not in ("grayscale", "chromatic"):
            raise ValueError(
                f"color_mode must be 'grayscale' or 'chromatic', got {color_mode!r}"
            )
        self.color_mode = color_mode
        if color_mode == "grayscale":
            self.out_psf_channels = 1
        else:
            self.out_psf_channels = (
                psf_channels
                if psf_channels is not None
                else input_channel_options[-1]
            )

        stem_channels = (
            stem_channels
            if stem_channels is not None
            else max(4, input_channel_options[-1])
        )
        self.input_adapter = nn.ModuleDict(
            {
                str(c): _ConvBlock(c, stem_channels, 3, 1, 1, **norm_kwargs)
                for c in input_channel_options
            }
        )
        self.initial = _ConvBlock(
            stem_channels, 32 * size, 3, 1, 1, **norm_kwargs
        )
        self.rb0 = _ResidualBlock(32 * size, 64 * size, 2, **norm_kwargs)
        self.rb1 = _ResidualBlock(64 * size, 128 * size, 2, **norm_kwargs)
        self.rb2 = _ResidualBlock(128 * size, 256 * size, 2, **norm_kwargs)
        self.rb3 = _ResidualBlock(256 * size, 384 * size, 2, **norm_kwargs)
        self.rb4 = _ResidualBlock(384 * size, 512 * size, 1, **norm_kwargs)

        self.bridge = _SelfAttentionBottleneck(512 * size)
        self.head = _AttnPSFHead(
            512 * size,
            psf_size,
            activation=activation,
            out_channels=self.out_psf_channels,
        )

    def forward(self, x):
        n_channels = x.size(1)
        key = str(n_channels)
        if key not in self.input_adapter:
            raise ValueError(
                f"Got {n_channels}-channel input but this model only supports "
                f"{list(self.input_adapter)} channels."
            )
        x = self.initial(self.input_adapter[key](x))
        x = self.rb0(x)
        x = self.rb1(x)
        x = self.rb2(x)
        x = self.rb3(x)
        x = self.rb4(x)
        x = self.bridge(x)
        out = self.head(x)

        if (
            self.color_mode == "chromatic"
            and n_channels < self.out_psf_channels
        ):
            out = out[:, :n_channels]
        return out


######################################################################################
#############      Noise level (deepblur.model.noise_level_identification) ##########
######################################################################################


class NoiseLevelIdNet(nn.Module):
    r"""
    Estimates the noise of a grayscale image in [0, 1]: Gaussian std ``sigma``
    and Poisson gain, in photon counts (see ``_identify_noise`` in the widget).
    """

    def __init__(self, in_channels=1, out_channels=2) -> None:
        super().__init__()
        # efficientnet as the feature extractor, with in_channels inputs
        # instead of 3
        self.enet = torchvision.models.efficientnet_b0(weights=None)
        self.enet.features[0][0] = nn.Conv2d(
            in_channels,
            32,
            kernel_size=(3, 3),
            stride=(2, 2),
            padding=(1, 1),
            bias=False,
        )

        self.head = nn.Sequential(
            nn.Linear(1000, 512),
            nn.ReLU(),
            nn.Linear(512, 32),
            nn.ReLU(),
            nn.Linear(32, out_channels),
            nn.ReLU(),
        )

        _any_norm_to_instance_norm(self)

    def forward(self, x):
        return self.head(self.enet(x))


def _any_norm_to_instance_norm(model: nn.Module):
    for name, child in model.named_children():
        if isinstance(child, (nn.BatchNorm2d, nn.GroupNorm)):
            num_features = (
                child.num_features
                if isinstance(child, nn.BatchNorm2d)
                else child.num_channels
            )
            setattr(model, name, nn.InstanceNorm2d(num_features=num_features))
        else:
            _any_norm_to_instance_norm(child)
