"""Bloques propios: ProbSparse encoder, distilling, decoder causal y LSTM."""
from dataclasses import dataclass
import math

import torch
from torch import nn


@dataclass
class ModelConfig:
    name: str = "dmel"
    history: int = 336
    horizon: int = 48
    d_model: int = 256
    heads: int = 8
    encoder_layers: int = 2
    ff: int = 1024
    lstm_hidden: int = 256
    lstm_layers: int = 2
    label_len: int = 48
    factor: int = 5
    dropout: float = 0.1


class ProbSparse(nn.Module):
    """Solo las queries seleccionadas calculan atención contra todas las keys.

    Muestreo aleatorio en train; muestreo fijo con seed 0 en eval para que la
    predicción no dependa del orden de batches. No hay máscara: todo es historia.
    """
    def __init__(self, dim, heads, factor, dropout):
        super().__init__()
        if dim % heads or factor < 1:
            raise ValueError("d_model debe ser divisible por heads; factor>=1")
        self.heads, self.factor, self.depth = heads, factor, dim // heads
        self.qkv = nn.Linear(dim, 3 * dim)
        self.projection = nn.Linear(dim, dim)
        self.dropout = nn.Dropout(dropout)

    def forward(self, x):
        b, length, dim = x.shape
        q, k, v = self.qkv(x).reshape(b, length, 3, self.heads, self.depth).permute(2, 0, 3, 1, 4)
        count = min(length, max(1, self.factor * math.ceil(math.log(length + 1))))
        if count == length:
            out = self.dropout((q @ k.transpose(-1, -2) / math.sqrt(self.depth)).softmax(-1)) @ v
        else:
            if self.training:
                sample = torch.randint(length, (length, count), device=x.device)
            else:
                sample = torch.randint(length, (length, count), generator=torch.Generator().manual_seed(0)).to(x.device)
            # La selección discreta no tiene gradiente. Bloques pequeños evitan
            # materializar [B,heads,L,samples,depth] para toda la secuencia.
            with torch.no_grad():
                estimates = []
                for start in range(0, length, 32):
                    stop = min(length, start + 32)
                    sampled_scores = torch.einsum("bhqd,bhqkd->bhqk", q[:, :, start:stop], k[:, :, sample[start:stop], :])
                    estimates.append(sampled_scores.amax(-1) - sampled_scores.mean(-1))
                importance = torch.cat(estimates, dim=-1)
                selected = importance.topk(count, dim=-1, sorted=False).indices
            queries = q.gather(2, selected.unsqueeze(-1).expand(-1, -1, -1, self.depth))
            scores = queries @ k.transpose(-1, -2) / math.sqrt(self.depth)
            attended = self.dropout(scores.softmax(-1)) @ v
            out = v.mean(2, keepdim=True).expand(-1, -1, length, -1).clone()
            out.scatter_(2, selected.unsqueeze(-1).expand_as(attended), attended)
        return self.projection(out.transpose(1, 2).reshape(b, length, dim))


def position(length, dim, device, dtype, offset=0):
    p = torch.arange(offset, offset + length, device=device, dtype=torch.float32).unsqueeze(1)
    frequencies = torch.exp(torch.arange(0, dim, 2, device=device) * (-math.log(10000.0) / dim))
    values = torch.zeros(length, dim, device=device)
    values[:, 0::2] = torch.sin(p * frequencies)
    values[:, 1::2] = torch.cos(p * frequencies[:dim // 2])
    return values.to(dtype)


class EncoderBlock(nn.Module):
    def __init__(self, cfg):
        super().__init__()
        self.attention = ProbSparse(cfg.d_model, cfg.heads, cfg.factor, cfg.dropout)
        self.norm1, self.norm2 = nn.LayerNorm(cfg.d_model), nn.LayerNorm(cfg.d_model)
        self.ff = nn.Sequential(nn.Linear(cfg.d_model, cfg.ff), nn.GELU(), nn.Dropout(cfg.dropout), nn.Linear(cfg.ff, cfg.d_model))
        self.dropout = nn.Dropout(cfg.dropout)

    def forward(self, x):
        x = self.norm1(x + self.dropout(self.attention(x)))
        return self.norm2(x + self.dropout(self.ff(x)))


class Informer(nn.Module):
    def __init__(self, cfg, inputs=12):
        super().__init__()
        self.cfg = cfg
        self.embedding = nn.Linear(inputs, cfg.d_model)
        self.encoder = nn.ModuleList([EncoderBlock(cfg) for _ in range(cfg.encoder_layers)])
        self.distill = nn.ModuleList([
            nn.Sequential(nn.Conv1d(cfg.d_model, cfg.d_model, 3, padding=1), nn.ELU(), nn.MaxPool1d(3, stride=2, padding=1))
            for _ in range(cfg.encoder_layers - 1)])
        # Una capa decoder: self-attention causal y cross-attention completa.
        self.decoder = nn.TransformerDecoderLayer(cfg.d_model, cfg.heads, cfg.ff, cfg.dropout, activation="gelu", batch_first=True)
        self.head = nn.Linear(cfg.d_model, 1)

    def forward(self, x):
        cfg = self.cfg
        memory = self.embedding(x) + position(x.shape[1], cfg.d_model, x.device, x.dtype)
        for i, encoder in enumerate(self.encoder):
            memory = encoder(memory)
            if i < len(self.distill):
                memory = self.distill[i](memory.transpose(1, 2)).transpose(1, 2)
        # Las variables futuras son ceros: no entran y ni y_aux.
        prefix = x[:, -cfg.label_len:]
        decoder_input = torch.cat((prefix, x.new_zeros(x.shape[0], cfg.horizon, x.shape[2])), dim=1)
        length = decoder_input.shape[1]
        query = self.embedding(decoder_input) + position(length, cfg.d_model, x.device, x.dtype, x.shape[1] - cfg.label_len)
        mask = torch.ones(length, length, device=x.device, dtype=torch.bool).triu(1)
        decoded = self.decoder(query, memory, tgt_mask=mask)
        return self.head(decoded[:, -cfg.horizon:]).squeeze(-1) + x[:, -1, -1:]


class LSTM(nn.Module):
    def __init__(self, cfg, inputs=12):
        super().__init__()
        self.rnn = nn.LSTM(inputs, cfg.lstm_hidden, cfg.lstm_layers, batch_first=True,
                           dropout=cfg.dropout if cfg.lstm_layers > 1 else 0)
        self.head = nn.Linear(cfg.lstm_hidden, cfg.horizon)

    def forward(self, x):
        _, (hidden, _) = self.rnn(x)
        return self.head(hidden[-1]) + x[:, -1, -1:]


class ForecastModel(nn.Module):
    def __init__(self, cfg):
        super().__init__()
        if cfg.history != 336 or cfg.horizon != 48 or not 1 <= cfg.label_len <= cfg.history:
            raise ValueError("Contrato Lab2: history=336, horizon=48; label_len válido")
        if cfg.encoder_layers < 1 or cfg.lstm_layers < 1 or not 0 <= cfg.dropout < 1:
            raise ValueError("Configuración de capas/dropout inválida")
        self.cfg = cfg
        if cfg.name == "lstm":
            self.single = LSTM(cfg)
        elif cfg.name == "informer":
            self.single = Informer(cfg)
        elif cfg.name in ("dmel", "mrs_informer", "mrs_lstm"):
            self.high = LSTM(cfg) if cfg.name == "mrs_lstm" else Informer(cfg)
            self.low = Informer(cfg) if cfg.name == "mrs_informer" else LSTM(cfg)
        else:
            raise ValueError(f"Modelo desconocido: {cfg.name}")

    def forward(self, x, components=None):
        if x.ndim != 3 or x.shape[1:] != (336, 12):
            raise ValueError("X debe ser [B,336,12]")
        if hasattr(self, "single"):
            return self.single(x)
        if components is None or components.shape != (*x.shape[:2], 2):
            raise ValueError("DMEL/ablación necesita components [B,336,2]")
        weather = x[:, :, :11]
        return self.high(torch.cat((weather, components[:, :, :1]), -1)) + self.low(torch.cat((weather, components[:, :, 1:]), -1))
