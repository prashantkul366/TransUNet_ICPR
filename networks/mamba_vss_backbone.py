# backbones/mamba_vss_backbone.py
import torch
import torch.nn as nn

from .mamba_sys import VSSM  # adjust relative path if needed

class MambaVSSBackbone(nn.Module):
    """
    Wrap VSSM to look like your old ViT encoder:
      forward(x) -> (tokens[B,N,C], None, features[list of BCHW for skips])
    We only use VSSM's *encoder*; TransUNet's DecoderCup does the upsampling.
    """
    def __init__(self, img_size=224, in_chans=3, dims=(96,192,384,768), depths=(2,2,9,2),
                 d_state=16, patch_size=4, drop_rate=0., drop_path_rate=0., 
                 patch_norm=True, use_checkpoint=False):
        super().__init__()
        print("---init mamba vss backbone---")
        self.vssm = VSSM(
            patch_size=patch_size,
            in_chans=in_chans,
            num_classes=1,              # unused (we won’t call VSSM.forward)
            depths=list(depths),
            dims=list(dims),
            d_state=d_state,
            drop_rate=drop_rate,
            drop_path_rate=drop_path_rate,
            patch_norm=patch_norm,
            use_checkpoint=use_checkpoint,
        )
        self.hidden_size = dims[-1]

    @staticmethod
    def _bhwc_to_bchw(x):
        # VSSM emits BHWC inside; TransUNet decoder expects BCHW skips
        return x.permute(0, 3, 1, 2).contiguous()

    def forward(self, x):
        """
        Returns:
          tokens:   (B, N, C) where C = hidden_size
          weights:  None (no MHSA maps)
          features: [B, Cx, H/16, W/16], [B, Cx, H/8, W/8], [B, Cx, H/4, W/4], None
        """
        # Encoder only
        x_bottleneck, x_down = self.vssm.forward_features(x)   # x_bottleneck: BHWC, x_down: list[4x BHWC]
        B, Hb, Wb, Cb = x_bottleneck.shape
        assert Cb == self.hidden_size

        # tokens for DecoderCup entry (TransUNet expects (B,N,C) then reshapes back)
        tokens = x_bottleneck.view(B, Hb * Wb, Cb)

        # Prepare 3 skips (order to match DecoderCup’s up blocks)
        feats_bchw = [self._bhwc_to_bchw(t) for t in x_down]  # index: 0=H/4, 1=H/8, 2=H/16, 3=H/32
        features = [feats_bchw[2], feats_bchw[1], feats_bchw[0], None]  # H/16, H/8, H/4, None

        return tokens, None, features

    # --- Optional: load encoder-only weights from a VSSM ckpt like your MambaUnet ---
    def load_encoder_from_vssm_ckpt(self, ckpt_path):
        """
        Loads only encoder (layers.*) into self.vssm. Ignores decoder/up layers.
        Compatible with your 'MambaUnet.load_from' style checkpoints.
        """
        if ckpt_path is None:
            print("No VSSM ckpt given; training encoder from scratch.")
            return

        device = torch.device('cuda' if torch.cuda.is_available() else 'cpu')
        state = torch.load(ckpt_path, map_location=device)
        if "model" in state:
            state = state["model"]

        vssm_model = self.vssm.state_dict()
        filtered = {}
        for k, v in state.items():
            # keep encoder blocks only
            if k.startswith("layers.") or k.startswith("patch_embed.") or k.startswith("norm"):
                if k in vssm_model and vssm_model[k].shape == v.shape:
                    filtered[k] = v

        missing, unexpected = self.vssm.load_state_dict(filtered, strict=False)
        print(f"[MambaVSSBackbone] Loaded encoder weights: "
              f"{len(filtered)} tensors | missing: {len(missing)} | unexpected: {len(unexpected)}")
