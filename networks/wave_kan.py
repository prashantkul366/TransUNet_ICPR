# wave_kan.py
import math
import torch
import torch.nn as nn
import torch.nn.functional as F

__all__ = ["WaveKAN", "WaveKANLinear"]

class WaveKANLinear(nn.Module):
    def __init__(self, in_features, out_features, wavelet_type='mexican_hat', eps=1e-4):
        super().__init__()
        self.in_features = in_features
        self.out_features = out_features
        self.wavelet_type = wavelet_type
        self.eps = eps

        # Learnable wavelet params per (out, in)
        self._scale_raw = nn.Parameter(torch.zeros(out_features, in_features))  # will softplus
        self.translation = nn.Parameter(torch.zeros(out_features, in_features))

        # weights over wavelet responses (out, in)
        self.wavelet_weights = nn.Parameter(torch.empty(out_features, in_features))
        nn.init.kaiming_uniform_(self.wavelet_weights, a=math.sqrt(5))

        # optional linear "base" branch (like Spl-KAN); kept but default disabled
        self.base_weight = nn.Parameter(torch.empty(out_features, in_features))
        nn.init.kaiming_uniform_(self.base_weight, a=math.sqrt(5))
        self.use_base = False

        # BN over features (expects (N, C))
        self.bn = nn.BatchNorm1d(out_features)

    def _scale(self):
        # strictly positive
        return torch.nn.functional.softplus(self._scale_raw) + self.eps

    def _apply_wavelet(self, x_scaled):
        # x_scaled: (BN, out, in)
        if self.wavelet_type == 'mexican_hat':
            term1 = (x_scaled ** 2) - 1.0
            term2 = torch.exp(-0.5 * x_scaled ** 2)
            wavelet = (2.0 / (math.sqrt(3) * math.pi ** 0.25)) * term1 * term2
        elif self.wavelet_type == 'morlet':
            omega0 = 5.0
            wavelet = torch.exp(-0.5 * x_scaled ** 2) * torch.cos(omega0 * x_scaled)
        elif self.wavelet_type == 'dog':
            wavelet = -x_scaled * torch.exp(-0.5 * x_scaled ** 2)
        else:
            raise ValueError(f"Unsupported wavelet type: {self.wavelet_type}")

        # weight per input dim, then sum over input dimension -> (BN, out)
        wavelet_weighted = wavelet * self.wavelet_weights.unsqueeze(0)  # (1,out,in)
        return wavelet_weighted.sum(dim=2)

    def forward(self, x):
        # x: (BN, in_features)
        BN, Din = x.shape
        # broadcast (out,in) across batch: -> (BN, out, in)
        trans = self.translation.unsqueeze(0).expand(BN, -1, -1)
        scale = self._scale().unsqueeze(0).expand(BN, -1, -1)

        x_exp = x.unsqueeze(1).expand(-1, self.out_features, -1)  # (BN,out,in)
        x_scaled = (x_exp - trans) / scale

        wav_out = self._apply_wavelet(x_scaled)  # (BN, out)

        if self.use_base:
            base_out = F.linear(x, self.base_weight)  # (BN, out)
            out = wav_out + base_out
        else:
            out = wav_out

        return self.bn(out)


class WaveKAN(nn.Module):
    """
    Wave-KAN MLP-style stack, same signature as your KAN:
      WaveKAN(layers_hidden=[D, H, D], wavelet_type='mexican_hat')
    """
    def __init__(self, layers_hidden, wavelet_type='mexican_hat'):
        super().__init__()
        self.layers = nn.ModuleList()
        for in_f, out_f in zip(layers_hidden[:-1], layers_hidden[1:]):
            self.layers.append(WaveKANLinear(in_f, out_f, wavelet_type=wavelet_type))

    def forward(self, x):
        # x: (BN, D_in)
        for layer in self.layers:
            x = layer(x)
        return x

# networks/wave_kan.py
import math
import torch
import torch.nn as nn
import torch.nn.functional as F

__all__ = ["WaveKANLite", "WaveKANLinearLite"]

class WaveKANLinearLite(nn.Module):
    """
    Memory-safe: wavelet per INPUT channel, then a linear map to outputs.
    x: (BN, Din) -> wave(x): (BN, Din) -> Linear(Din->Dout) -> (BN, Dout)
    """
    def __init__(self, in_features, out_features, wavelet_type='mexican_hat', eps=1e-4):
        super().__init__()
        self.in_features = in_features
        self.out_features = out_features
        self.wavelet_type = wavelet_type
        self.eps = eps

        # Per-INPUT parameters (no (out,in) grid!)
        self._scale_raw = nn.Parameter(torch.zeros(in_features))   # -> softplus to keep positive
        self.translation = nn.Parameter(torch.zeros(in_features))

        # Linear projection Din -> Dout
        self.weight = nn.Parameter(torch.empty(out_features, in_features))
        self.bias   = nn.Parameter(torch.zeros(out_features))
        nn.init.kaiming_uniform_(self.weight, a=math.sqrt(5))

        # BN over features (expects (N, C))
        self.bn = nn.BatchNorm1d(out_features)

    def _scale(self):
        return F.softplus(self._scale_raw) + self.eps  # positive

    def _wavelet_1d(self, z):
        # z: (BN, Din)
        if self.wavelet_type == 'mexican_hat':
            term1 = (z ** 2) - 1.0
            term2 = torch.exp(-0.5 * z ** 2)
            return (2.0 / (math.sqrt(3) * math.pi ** 0.25)) * term1 * term2
        elif self.wavelet_type == 'morlet':
            omega0 = 5.0
            return torch.exp(-0.5 * z ** 2) * torch.cos(omega0 * z)
        elif self.wavelet_type == 'dog':
            return -z * torch.exp(-0.5 * z ** 2)
        else:
            raise ValueError(f"Unsupported wavelet type: {self.wavelet_type}")

    def forward(self, x):
        # x: (BN, Din)
        scale = self._scale()            # (Din,)
        trans = self.translation         # (Din,)
        z = (x - trans) / scale          # (BN, Din)
        w = self._wavelet_1d(z)          # (BN, Din)
        out = F.linear(w, self.weight, self.bias)  # (BN, Dout)
        return self.bn(out)


class WaveKANLite(nn.Module):
    """
    Two-layer stack like MLP: [D, mlp_dim, D] using WaveKANLinearLite.
    """
    def __init__(self, layers_hidden, wavelet_type='mexican_hat'):
        super().__init__()
        layers = []
        for Din, Dout in zip(layers_hidden[:-1], layers_hidden[1:]):
            layers.append(WaveKANLinearLite(Din, Dout, wavelet_type=wavelet_type))
        self.layers = nn.ModuleList(layers)

    def forward(self, x):
        for layer in self.layers:
            x = layer(x)
        return x
