import ml_collections

def get_b16_config():
    """Returns the ViT-B/16 configuration."""
    config = ml_collections.ConfigDict()
    config.patches = ml_collections.ConfigDict({'size': (16, 16)})
    config.hidden_size = 768
    config.transformer = ml_collections.ConfigDict()
    config.transformer.mlp_dim = 3072
    config.transformer.num_heads = 12
    config.transformer.num_layers = 12
    config.transformer.attention_dropout_rate = 0.0
    config.transformer.dropout_rate = 0.1

    config.classifier = 'seg'
    config.representation_size = None
    config.resnet_pretrained_path = None
    # config.pretrained_path = '../model/vit_checkpoint/imagenet21k/ViT-B_16.npz'
    config.pretrained_path = '/content/drive/MyDrive/Prashant/model/vit_checkpoint/imagenet21k/R50-ViT-B_16.npz'
    config.patch_size = 16

    config.decoder_channels = (256, 128, 64, 16)
    config.n_classes = 2
    config.activation = 'softmax'
    return config


def get_testing():
    """Returns a minimal configuration for testing."""
    config = ml_collections.ConfigDict()
    config.patches = ml_collections.ConfigDict({'size': (16, 16)})
    config.hidden_size = 1
    config.transformer = ml_collections.ConfigDict()
    config.transformer.mlp_dim = 1
    config.transformer.num_heads = 1
    config.transformer.num_layers = 1
    config.transformer.attention_dropout_rate = 0.0
    config.transformer.dropout_rate = 0.1
    config.classifier = 'token'
    config.representation_size = None
    return config

def get_r50_b16_config():
    """Returns the Resnet50 + ViT-B/16 configuration."""
    config = get_b16_config()
    config.patches.grid = (16, 16)
    config.resnet = ml_collections.ConfigDict()
    config.resnet.num_layers = (3, 4, 9)
    config.resnet.width_factor = 1

    config.classifier = 'seg'
    # config.pretrained_path = '../model/vit_checkpoint/imagenet21k/R50+ViT-B_16.npz'
    config.pretrained_path = '/content/drive/MyDrive/Prashant/model/vit_checkpoint/imagenet21k/R50-ViT-B_16.npz'
    # config.decoder_channels = (256, 128, 64, 16)
    config.decoder_channels = (512, 256, 128, 64)
    # config.skip_channels = [512, 256, 64, 16]
    config.skip_channels = [768, 768, 768, 768]
    config.n_classes = 2
    config.n_skip = 3
    config.activation = 'softmax'

    config.use_kan_ffn = True
    config.wavelet_type = "mexican_hat"
    config.kan_grid_size = 5
    config.kan_spline_order = 3
    config.kan_scale_noise = 0.1
    config.kan_scale_base = 1.0
    config.kan_scale_spline = 1.0
    config.kan_grid_eps = 0.02
    config.kan_grid_range = [-1, 1]


    return config


def get_b32_config():
    """Returns the ViT-B/32 configuration."""
    config = get_b16_config()
    config.patches.size = (32, 32)
    config.pretrained_path = '../model/vit_checkpoint/imagenet21k/ViT-B_32.npz'
    return config


def get_l16_config():
    """Returns the ViT-L/16 configuration."""
    config = ml_collections.ConfigDict()
    config.patches = ml_collections.ConfigDict({'size': (16, 16)})
    config.hidden_size = 1024
    config.transformer = ml_collections.ConfigDict()
    config.transformer.mlp_dim = 4096
    config.transformer.num_heads = 16
    config.transformer.num_layers = 24
    config.transformer.attention_dropout_rate = 0.0
    config.transformer.dropout_rate = 0.1
    config.representation_size = None

    # custom
    config.classifier = 'seg'
    config.resnet_pretrained_path = None
    config.pretrained_path = '../model/vit_checkpoint/imagenet21k/ViT-L_16.npz'
    config.decoder_channels = (256, 128, 64, 16)
    config.n_classes = 2
    config.activation = 'softmax'
    return config


def get_r50_l16_config():
    """Returns the Resnet50 + ViT-L/16 configuration. customized """
    config = get_l16_config()
    config.patches.grid = (16, 16)
    config.resnet = ml_collections.ConfigDict()
    config.resnet.num_layers = (3, 4, 9)
    config.resnet.width_factor = 1

    config.classifier = 'seg'
    config.resnet_pretrained_path = '../model/vit_checkpoint/imagenet21k/R50+ViT-B_16.npz'
    config.decoder_channels = (256, 128, 64, 16)
    config.skip_channels = [512, 256, 64, 16]
    config.n_classes = 2
    config.activation = 'softmax'
    return config


def get_l32_config():
    """Returns the ViT-L/32 configuration."""
    config = get_l16_config()
    config.patches.size = (32, 32)
    return config


def get_h14_config():
    """Returns the ViT-L/16 configuration."""
    config = ml_collections.ConfigDict()
    config.patches = ml_collections.ConfigDict({'size': (14, 14)})
    config.hidden_size = 1280
    config.transformer = ml_collections.ConfigDict()
    config.transformer.mlp_dim = 5120
    config.transformer.num_heads = 16
    config.transformer.num_layers = 32
    config.transformer.attention_dropout_rate = 0.0
    config.transformer.dropout_rate = 0.1
    config.classifier = 'token'
    config.representation_size = None

    return config


def get_mamba_vss_config():
    config = ml_collections.ConfigDict()

    config.backbone = 'mamba_vss'
    config.mamba_dims   = [96, 192, 384, 768]
    config.mamba_depths = [2, 2, 9, 2]
    config.mamba_patch  = 4
    config.mamba_d_state = 16
    config.mamba_drop_rate = 0.0
    config.mamba_drop_path_rate = 0.1
    config.mamba_use_checkpoint = False

    config.hidden_size = config.mamba_dims[-1]      # 768
    config.decoder_channels = (256, 128, 64, 16)

    config.n_classes = 2
    config.activation = 'softmax'

    # skips correspond to features returned by the adapter: [H/16, H/8, H/4, None]
    config.n_skip = 3
    config.skip_channels = [config.mamba_dims[2],  # 384
                            config.mamba_dims[1],  # 192
                            config.mamba_dims[0],  # 96
                            0]

    # keep keys that other code may access
    config.patches = ml_collections.ConfigDict({'size': (4, 4)})
    config.classifier = 'seg'
    config.representation_size = None
    config.resnet_pretrained_path = None
    config.pretrained_path = None  # don't load ViT weights

    return config

def _mobilemamba_variant_specs(variant: str):
    """
    Returns (img_size, embed_dim) for a given MobileMamba variant name.
    Keep this in sync with mobilemamba.py CFG_*.
    """
    v = variant.upper()
    table = {
        "T2": (192, [144, 272, 368]),
        "T4": (192, [176, 368, 448]),
        "S6": (224, [192, 384, 448]),
        "B1": (256, [200, 376, 448]),
        "B2": (384, [200, 376, 448]),
        "B4": (512, [200, 376, 448]),
    }
    if v not in table:
        v = "S6"  # sensible default
    return table[v]


def get_mobilemamba_config(
    n_classes: int = 2,
    decoder_channels=(256, 128, 64, 16),
    mobilemamba_variant: str = "S6",
    n_skip: int = 3,
):
    """
    Config for using MobileMamba as the encoder/backbone in TransUNet-style model.
    Matches the expectations of vit_seg_modelling_mobile_mamba.py after you swap in MobileMambaFeatures.
    """
    cfg = ml_collections.ConfigDict()
    cfg.backbone = "mobilemamba"

    # Variant-specific widths (embed_dim) and a nominal input size (not strictly required here)
    img_size, embed_dim = _mobilemamba_variant_specs(mobilemamba_variant)
    C1, C2, C3 = embed_dim  # shallow -> deep per your MobileMamba

    # The decoder consumes the encoder bottleneck as 'hidden_size'
    cfg.hidden_size = C3

    # TransUNet head settings
    cfg.decoder_channels = tuple(decoder_channels)
    cfg.n_classes = int(n_classes)
    cfg.activation = "softmax"  # use "sigmoid" for binary w/o one-hot targets

    # Skip connections: provide deepest->shallowest (DecoderCup expects features[i])
    # We’ll pass features as [f3, f2, f1], so skip_channels should be [C3, C2, C1, 0]
    cfg.n_skip = int(n_skip)
    if cfg.n_skip > 0:
        cfg.skip_channels = [C3, C2, C1, 0]
    else:
        cfg.skip_channels = [0, 0, 0, 0]

    # Keep keys some code paths expect even if not used by MobileMamba
    cfg.patches = ml_collections.ConfigDict({"size": (4, 4)})  # dummy; ViT-only code ignores for MobileMamba
    cfg.classifier = "seg"
    cfg.representation_size = None
    cfg.resnet_pretrained_path = None
    cfg.pretrained_path = None

    # Let the model wrapper know which MobileMamba variant to instantiate
    cfg.mobilemamba_variant = mobilemamba_variant

    # (Optional) expose img_size so your training script can set datasets/transforms accordingly
    cfg.img_size = img_size

    return cfg

def get_kat_b16_config():
    c = get_b16_config()                 # start from your ViT-B/16 base
    c.backbone = "kat"

    # KAT pretrained variants are patch16, and this field is typed as a tuple:
    c.patches['size'] = (16, 16)

    # KAT-B uses embed_dim=768 (tiny=192, small=384)
    c.hidden_size = 768

    # No ResNet hybrid skips when using KAT
    c.n_skip = 0

    # Tell the adapter which KAT checkpoint to use
    c.kat_variant = "kat_base_patch16_224.vitft"   # or: kat_small_patch16_224(.vitft), kat_tiny_patch16_224(.vitft)
    c.kat_pretrained = True

    # Usual TransUNet fields (keep whatever you already have)
    c.classifier = "seg"
    # c.decoder_channels, c.skip_channels, etc. should already be in the base; n_skip=0 disables skips anyway
    return c


def get_tsmamba_config(
    n_classes: int = 2,
    decoder_channels=(256, 128, 64, 16),
    n_skip: int = 3,
):
    c = ml_collections.ConfigDict()
    c.backbone = "tsmamba"

    c.mamba_dims   = [48, 96, 192, 384]
    c.mamba_depths = [2, 2, 2, 2]
    c.mamba_drop_rate = 0.0

    c.hidden_size = c.mamba_dims[-1]  # 384

    c.decoder_channels = tuple(decoder_channels)
    c.n_classes = int(n_classes)
    c.activation = "softmax"

    c.n_skip = int(n_skip)
    if c.n_skip > 0:
        c.skip_channels = [c.mamba_dims[2],  # 192 (28×28)
                           c.mamba_dims[1],  #  96 (56×56)
                           c.mamba_dims[0],  #  48 (112×112)
                           0]
    else:
        c.skip_channels = [0, 0, 0, 0]

    # dummy fields to keep ViT code happy
    c.patches = ml_collections.ConfigDict({"size": (4, 4)})
    c.classifier = "seg"
    c.representation_size = None
    c.resnet_pretrained_path = None
    c.pretrained_path = None

    return c


