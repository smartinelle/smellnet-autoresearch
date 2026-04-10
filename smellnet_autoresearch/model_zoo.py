from __future__ import annotations

import math
from typing import Optional

import torch
import torch.nn as nn
import torch.nn.functional as F


class SinusoidalPositionalEncoding(nn.Module):
    def __init__(self, d_model: int, max_len: int = 10000):
        super().__init__()
        pe = torch.zeros(max_len, d_model)
        position = torch.arange(0, max_len).unsqueeze(1)
        div_term = torch.exp(torch.arange(0, d_model, 2) * (-math.log(10000.0) / d_model))
        pe[:, 0::2] = torch.sin(position * div_term)
        pe[:, 1::2] = torch.cos(position * div_term)
        self.register_buffer("pe", pe, persistent=False)

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        return x + self.pe[: x.size(1)].unsqueeze(0)


class Transformer(nn.Module):
    def __init__(
        self,
        input_dim: int,
        model_dim: int,
        num_classes: int,
        num_heads: int = 8,
        num_layers: int = 4,
        dropout: float = 0.1,
        activation: str = "gelu",
        use_positional_encoding: bool = True,
        use_cls_token: bool = False,
        pool: str = "mean",
    ):
        super().__init__()
        assert pool in ("mean", "cls")
        self.input_proj = nn.Sequential(
            nn.Linear(input_dim, model_dim),
            nn.LayerNorm(model_dim),
        )
        self.pos = SinusoidalPositionalEncoding(model_dim) if use_positional_encoding else nn.Identity()
        self.use_cls_token = use_cls_token
        self.pool = pool

        enc_layer = nn.TransformerEncoderLayer(
            d_model=model_dim,
            nhead=num_heads,
            dim_feedforward=4 * model_dim,
            dropout=dropout,
            batch_first=True,
            activation=activation,
            norm_first=True,
        )
        self.transformer = nn.TransformerEncoder(enc_layer, num_layers=num_layers)

        if use_cls_token:
            self.cls_token = nn.Parameter(torch.zeros(1, 1, model_dim))
            nn.init.normal_(self.cls_token, std=0.02)

        self.dropout = nn.Dropout(dropout)
        self.classifier = nn.Sequential(
            nn.Linear(model_dim, model_dim // 2),
            nn.GELU(),
            nn.Dropout(dropout),
            nn.Linear(model_dim // 2, num_classes),
        )

    def _key_padding_mask(self, lengths: torch.Tensor, t: int) -> torch.Tensor:
        rng = torch.arange(t, device=lengths.device).unsqueeze(0)
        return rng >= lengths.unsqueeze(1)

    def forward_features(self, x: torch.Tensor, lengths: Optional[torch.Tensor] = None) -> torch.Tensor:
        x = self.input_proj(x)
        x = self.pos(x)

        key_padding_mask = None
        if lengths is not None:
            key_padding_mask = self._key_padding_mask(lengths, x.size(1))

        if self.use_cls_token:
            cls = self.cls_token.expand(x.size(0), -1, -1)
            x = torch.cat([cls, x], dim=1)
            if key_padding_mask is not None:
                pad0 = torch.zeros((key_padding_mask.size(0), 1), dtype=torch.bool, device=x.device)
                key_padding_mask = torch.cat([pad0, key_padding_mask], dim=1)

        h = self.transformer(x, src_key_padding_mask=key_padding_mask)

        if self.use_cls_token and self.pool == "cls":
            feat = h[:, 0]
        else:
            tokens = h[:, 1:] if self.use_cls_token else h
            if key_padding_mask is None:
                feat = tokens.mean(dim=1)
            else:
                mask = (~key_padding_mask).float()
                if self.use_cls_token:
                    mask = mask[:, 1:]
                denom = mask.sum(dim=1, keepdim=True).clamp_min(1e-6)
                feat = (tokens * mask.unsqueeze(-1)).sum(dim=1) / denom
        return feat

    def forward(self, x: torch.Tensor, lengths: Optional[torch.Tensor] = None) -> torch.Tensor:
        feat = self.forward_features(x, lengths)
        return self.classifier(self.dropout(feat))


class GCMSMLPEncoder(nn.Module):
    def __init__(
        self,
        in_features: int,
        embedding_dim: int = 256,
        hidden: tuple[int, ...] = (512, 256),
        dropout: float = 0.1,
        use_layernorm: bool = True,
        use_batchnorm: bool = False,
        l2_normalize: bool = False,
    ):
        super().__init__()
        layers: list[nn.Module] = []
        if use_layernorm:
            layers.append(nn.LayerNorm(in_features))
        last = in_features
        for h in hidden:
            layers.append(nn.Linear(last, h))
            if use_batchnorm:
                layers.append(nn.BatchNorm1d(h))
            layers.append(nn.ReLU(inplace=True))
            if dropout > 0:
                layers.append(nn.Dropout(dropout))
            last = h
        layers.append(nn.Linear(last, embedding_dim))
        self.net = nn.Sequential(*layers)
        self.l2_normalize = l2_normalize

    def forward_features(self, x: torch.Tensor, lengths: Optional[torch.Tensor] = None) -> torch.Tensor:
        if x.dim() != 2:
            x = x.view(x.size(0), -1)
        z = self.net(x)
        if self.l2_normalize:
            z = F.normalize(z, dim=-1)
        return z

    def forward(self, x: torch.Tensor, lengths: Optional[torch.Tensor] = None) -> torch.Tensor:
        return self.forward_features(x, lengths)
