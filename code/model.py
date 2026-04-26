"""
Sensor-agnostic speech enhancement model.

Processes arbitrary subsets of heterogeneous body-conducted sensors through a
shared Conv1d frontend with FiLM (Feature-wise Linear Modulation) conditioning
on learned sensor embeddings. Sensor dropout during training forces robustness
to any sensor subset, enabling zero-shot generalization to unseen sensor types.

Architecture:
    For each sensor:
        waveform -> Shared Conv1d Frontend -> FiLM(sensor_embedding)
    Optional cross-attention across sensor tokens
    -> Mean pooling -> Mamba backbone -> Conv1d head + skip -> output

Extensions:
    - sensor_cross_attention: Multi-head cross-attention across sensor tokens
    - attention_pooling: Content-dependent weighted average (replaces mean pool)
    - deep_film: FiLM conditioning at every Mamba block
    - use_stft_branch: Parallel STFT branch with gated spectral injection (V12)
"""

import torch
import torch.nn as nn
import torch.nn.functional as F

try:
    from mamba_ssm import Mamba
except ImportError:
    Mamba = None


# ---------------------------------------------------------------------------
# Building blocks
# ---------------------------------------------------------------------------

class MambaBlock(nn.Module):
    """Single Mamba layer with pre-norm residual: x + dropout(Mamba(LN(x)))."""

    def __init__(self, d_model, d_state=16, d_conv=4, expand=2, dropout=0.0, layer_idx=0):
        super().__init__()
        self.norm = nn.LayerNorm(d_model)
        self.mamba = Mamba(
            d_model=d_model, d_state=d_state, d_conv=d_conv,
            expand=expand, layer_idx=layer_idx,
        )
        self.dropout = nn.Dropout(dropout) if dropout > 0 else nn.Identity()

    def forward(self, x):
        return x + self.dropout(self.mamba(self.norm(x)))


class FiLMLayer(nn.Module):
    """Feature-wise Linear Modulation: output = gamma(cond) * x + beta(cond)."""

    def __init__(self, d_model):
        super().__init__()
        self.gamma = nn.Linear(d_model, d_model)
        self.beta = nn.Linear(d_model, d_model)

    def forward(self, x, conditioning):
        g = self.gamma(conditioning).unsqueeze(1)
        b = self.beta(conditioning).unsqueeze(1)
        return g * x + b


class SensorCrossAttention(nn.Module):
    """Multi-head cross-attention across sensor tokens at each timestep."""

    def __init__(self, d_model=256, n_heads=4, dropout=0.1):
        super().__init__()
        self.attn = nn.MultiheadAttention(d_model, n_heads, dropout=dropout, batch_first=True)
        self.norm = nn.LayerNorm(d_model)
        self.dropout = nn.Dropout(dropout)

    def forward(self, x):
        B, K, T, D = x.shape
        x_flat = x.permute(0, 2, 1, 3).reshape(B * T, K, D)
        attn_out, attn_w = self.attn(x_flat, x_flat, x_flat)
        x_flat = self.norm(x_flat + self.dropout(attn_out))
        return x_flat.reshape(B, T, K, D).permute(0, 2, 1, 3), attn_w


class AttentionPooling(nn.Module):
    """Content-dependent weighted average over sensor tokens."""

    def __init__(self, d_model=256):
        super().__init__()
        self.query = nn.Linear(d_model, 1)

    def forward(self, x, mask=None):
        scores = self.query(x).squeeze(-1)
        if mask is not None:
            scores = scores.masked_fill(~mask.unsqueeze(-1), float("-inf"))
        weights = F.softmax(scores, dim=1)
        pooled = (x * weights.unsqueeze(-1)).sum(dim=1)
        return pooled, weights


class STFTBranch(nn.Module):
    """Parallel frequency-domain feature extractor for multi-sensor input."""

    def __init__(self, num_sensors=5, d_model=256, n_fft=1024,
                 hop_length=120, win_length=600, branch_channels=128):
        super().__init__()
        self.num_sensors = num_sensors
        self.n_fft = n_fft
        self.hop_length = hop_length
        self.win_length = win_length
        self.register_buffer("window", torch.hann_window(win_length))
        c = branch_channels
        self.conv1 = nn.Conv2d(num_sensors, c // 2, kernel_size=(5, 3), padding=(2, 1))
        self.conv2 = nn.Conv2d(c // 2, c, kernel_size=(5, 3), padding=(2, 1))
        self.conv3 = nn.Conv2d(c, c, kernel_size=(5, 3), padding=(2, 1))
        self.act = nn.GELU()
        self.freq_pool = nn.AdaptiveAvgPool2d((1, None))
        self.proj = nn.Conv1d(c, d_model, kernel_size=1)
        self.norm = nn.LayerNorm(d_model)

    def forward(self, x):
        B, S, T = x.shape
        x_flat = x.reshape(B * S, T).float()
        stft = torch.stft(
            x_flat, n_fft=self.n_fft, hop_length=self.hop_length,
            win_length=self.win_length, window=self.window.to(x_flat.device),
            return_complex=True, center=True,
        )
        log_mag = torch.log1p(stft.abs())
        F_bins, Frames = log_mag.shape[-2], log_mag.shape[-1]
        feats = log_mag.view(B, S, F_bins, Frames)
        h = self.act(self.conv1(feats))
        h = self.act(self.conv2(h))
        h = self.act(self.conv3(h))
        h = self.freq_pool(h).squeeze(2)
        h = self.proj(h).transpose(1, 2)
        return self.norm(h)


class SpectralInjection(nn.Module):
    """Gated cross-attention injection of STFT features into time-domain stream."""

    def __init__(self, d_model=256, n_heads=8, hop=120, gate_init=0.0):
        super().__init__()
        self.hop = hop
        self.downsample = nn.AvgPool1d(kernel_size=hop, stride=hop)
        self.cross_attn = nn.MultiheadAttention(d_model, n_heads, batch_first=True)
        self.norm = nn.LayerNorm(d_model)
        self.gate = nn.Parameter(torch.full((1,), float(gate_init)))

    def forward(self, mamba_features, stft_features):
        B, T, D = mamba_features.shape
        x_ds = self.downsample(mamba_features.transpose(1, 2)).transpose(1, 2)
        attended, _ = self.cross_attn(x_ds, stft_features, stft_features)
        up = F.interpolate(
            attended.transpose(1, 2), size=T, mode="linear", align_corners=False,
        ).transpose(1, 2)
        return mamba_features + torch.tanh(self.gate) * self.norm(up)


# ---------------------------------------------------------------------------
# Main model
# ---------------------------------------------------------------------------

class SensorAgnosticEnhancer(nn.Module):
    """
    Sensor-agnostic speech enhancement with FiLM conditioning and sensor dropout.

    Args:
        d_model: Hidden dimension (default 256).
        d_state: SSM state expansion (default 16).
        d_conv: Local conv width in Mamba (default 4).
        n_layers: Number of Mamba blocks (default 8).
        expand: Inner dimension multiplier (default 2).
        dropout: Dropout rate (default 0.1).
        num_sensors: Number of known sensor types (default 5).
        sensor_dropout: Probability of dropping each sensor during training (default 0.5).
        min_keep: Minimum sensors retained during dropout (default 1).
        skip_connection: Additive skip from mean input (default True).
        frontend_kernel_size: Frontend conv kernel size (default 7).
        sensor_cross_attention: Enable cross-attention across sensors (default True).
        cross_attention_heads: Number of attention heads (default 4).
        deep_film: FiLM at every Mamba block (default False).
        use_stft_branch: Parallel STFT branch with gated injection (default False).
    """

    def __init__(
        self,
        d_model=256, d_state=16, d_conv=4, n_layers=8, expand=2, dropout=0.1,
        num_sensors=5, sensor_dropout=0.5, min_keep=1, skip_connection=True,
        frontend_kernel_size=7,
        sensor_cross_attention=True, cross_attention_heads=4, cross_attention_dropout=0.1,
        attention_pooling=False, deep_film=False,
        use_stft_branch=False, stft_n_fft=1024, stft_hop_length=120,
        stft_win_length=600, stft_branch_channels=128,
        spectral_injection_layers=(1, 3, 5), spectral_injection_heads=8,
    ):
        super().__init__()
        self.num_sensors = num_sensors
        self.sensor_dropout = sensor_dropout
        self.min_keep = min_keep
        self.use_skip = skip_connection
        self.use_cross_attention = sensor_cross_attention
        self.use_attention_pooling = attention_pooling
        self.use_deep_film = deep_film
        self.use_stft_branch = use_stft_branch
        self._stft_hop = stft_hop_length
        self._injection_layers = tuple(spectral_injection_layers)

        self.frontend = nn.Sequential(
            nn.Conv1d(1, d_model, kernel_size=frontend_kernel_size,
                      padding=frontend_kernel_size // 2),
            nn.GELU(),
        )
        self.sensor_embedding = nn.Embedding(num_sensors + 1, d_model)
        self.film = FiLMLayer(d_model)

        if self.use_cross_attention:
            self.sensor_cross_attn = SensorCrossAttention(
                d_model, cross_attention_heads, cross_attention_dropout)

        if self.use_attention_pooling:
            self.attn_pool = AttentionPooling(d_model)

        self.layers = nn.ModuleList([
            MambaBlock(d_model, d_state, d_conv, expand, dropout, i)
            for i in range(n_layers)
        ])

        if self.use_deep_film:
            self.deep_film_generators = nn.ModuleList([
                nn.Linear(d_model, 2 * d_model) for _ in range(n_layers)
            ])

        self.final_norm = nn.LayerNorm(d_model)
        self.head = nn.Conv1d(d_model, 1, kernel_size=1)
        self.register_buffer("_default_sensor_ids", torch.arange(num_sensors))
        self._eval_sensor_ids = None

        if self.use_stft_branch:
            self.stft_branch = STFTBranch(
                num_sensors, d_model, stft_n_fft, stft_hop_length,
                stft_win_length, stft_branch_channels)
            self.spectral_injections = nn.ModuleList([
                SpectralInjection(d_model, spectral_injection_heads, stft_hop_length)
                for _ in self._injection_layers
            ])

    def set_eval_sensor_ids(self, sensor_ids):
        """Override sensor IDs for zero-shot eval on unseen sensors."""
        self._eval_sensor_ids = sensor_ids

    def forward(self, x):
        """
        Args:
            x: (B, S, T) -- S sensors, each a mono waveform.
        Returns:
            (B, 1, T) enhanced waveform.
        """
        B, S, T = x.shape

        if self.use_skip:
            skip = x.mean(dim=1, keepdim=True)

        stft_features = self.stft_branch(x) if self.use_stft_branch else None

        sensor_ids = (self._eval_sensor_ids.to(x.device)
                      if self._eval_sensor_ids is not None
                      else self._default_sensor_ids[:S].to(x.device))

        features, sensor_embs = [], []
        for s in range(S):
            h_s = self.frontend(x[:, s:s+1, :]).transpose(1, 2)
            emb = self.sensor_embedding(sensor_ids[s]).unsqueeze(0).expand(B, -1)
            sensor_embs.append(emb)
            features.append(self.film(h_s, emb))

        # Sensor dropout (training only)
        device = features[0].device
        if self.training and self.sensor_dropout > 0 and S > self.min_keep:
            mask = torch.rand(S, device=device) > self.sensor_dropout
            if mask.sum() < self.min_keep:
                keep = torch.randperm(S, device=device)[:self.min_keep]
                mask = torch.zeros(S, dtype=torch.bool, device=device)
                mask[keep] = True
            n_active = int(mask.sum().item())
            for s in range(S):
                if not mask[s]:
                    features[s] = torch.zeros_like(features[s])
        else:
            mask = torch.ones(S, dtype=torch.bool, device=device)
            n_active = S

        stacked = torch.stack(features, dim=1)

        if self.use_cross_attention:
            stacked, _ = self.sensor_cross_attn(stacked)
            if n_active < S:
                for s in range(S):
                    if not mask[s]:
                        stacked[:, s] = 0.0

        if self.use_attention_pooling:
            fused, _ = self.attn_pool(stacked, mask=mask.unsqueeze(0).expand(B, -1))
        else:
            fused = stacked.sum(dim=1) / max(n_active, 1)

        # Mamba backbone
        h = fused
        for i, layer in enumerate(self.layers):
            h = layer(h)
            if self.use_deep_film:
                emb_stack = torch.stack(sensor_embs, dim=1)
                active_w = mask.float() / max(n_active, 1)
                desc = (emb_stack * active_w.unsqueeze(0).unsqueeze(-1)).sum(dim=1)
                gamma, beta = self.deep_film_generators[i](desc).chunk(2, dim=-1)
                h = gamma.clamp(-2, 2).unsqueeze(1) * h + beta.unsqueeze(1)
            if self.use_stft_branch and i in self._injection_layers:
                idx = self._injection_layers.index(i)
                h = self.spectral_injections[idx](h, stft_features)

        out = self.head(self.final_norm(h).transpose(1, 2))
        if self.use_skip:
            out = out + skip
        return out
