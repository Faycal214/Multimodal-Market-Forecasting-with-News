import torch
import torch.nn as nn
from transformers import AutoModel


class MultimodalForecaster(nn.Module):
    def __init__(
        self,
        text_model_name="distilbert-base-uncased",
        price_dim=3,
        extra_dim=20,
        dropout=0.2,
        freeze_text_encoder=True,
    ):
        super().__init__()

        self.text_encoder = AutoModel.from_pretrained(text_model_name)
        hidden_size = self.text_encoder.config.hidden_size

        if freeze_text_encoder:
            for p in self.text_encoder.parameters():
                p.requires_grad = False
            # unfreeze the last two transformer blocks when available
            if hasattr(self.text_encoder, "transformer") and hasattr(self.text_encoder.transformer, "layer"):
                for layer in self.text_encoder.transformer.layer[-2:]:
                    for p in layer.parameters():
                        p.requires_grad = True
            elif hasattr(self.text_encoder, "encoder") and hasattr(self.text_encoder.encoder, "layer"):
                for layer in self.text_encoder.encoder.layer[-2:]:
                    for p in layer.parameters():
                        p.requires_grad = True

        self.ts_proj = nn.Sequential(
            nn.Linear(price_dim, hidden_size),
            nn.ReLU(),
            nn.Dropout(dropout),
            nn.Linear(hidden_size, hidden_size),
            nn.LayerNorm(hidden_size),
            nn.ReLU(),
        )

        encoder_layer = nn.TransformerEncoderLayer(
            d_model=hidden_size,
            nhead=8,
            dim_feedforward=hidden_size * 2,
            dropout=dropout,
            batch_first=True,
        )
        self.ts_encoder = nn.TransformerEncoder(encoder_layer, num_layers=1)

        self.extra_mlp = nn.Sequential(
            nn.Linear(extra_dim, hidden_size // 2),
            nn.ReLU(),
            nn.Dropout(dropout),
            nn.Linear(hidden_size // 2, hidden_size),
            nn.LayerNorm(hidden_size),
            nn.ReLU(),
        )

        self.head = nn.Sequential(
            nn.Linear(hidden_size * 3, hidden_size),
            nn.ReLU(),
            nn.Dropout(dropout),
            nn.Linear(hidden_size, hidden_size // 2),
            nn.ReLU(),
            nn.Dropout(dropout),
            nn.Linear(hidden_size // 2, price_dim),
        )

    def forward(self, lag_seq, extra_feats, input_ids, attention_mask):
        # lag_seq: [B, lag_days, 3]
        ts = self.ts_proj(lag_seq)
        ts = self.ts_encoder(ts).mean(dim=1)

        text_out = self.text_encoder(input_ids=input_ids, attention_mask=attention_mask)
        if hasattr(text_out, "last_hidden_state"):
            text_repr = text_out.last_hidden_state[:, 0, :]
        else:
            text_repr = text_out[0][:, 0, :]

        extra_repr = self.extra_mlp(extra_feats)

        fused = torch.cat([ts, text_repr, extra_repr], dim=-1)
        return self.head(fused)