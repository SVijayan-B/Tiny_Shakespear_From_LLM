import torch
import torch.nn as nn

from model.layers import TransformerBlock

class MiniGPT(nn.Module):
    def __init__(
        self,
        vocab_size,
        context_length,
        d_model,
        n_heads,
        n_layers,
        dropout,
    ):
        super().__init__()

        self.context_length = context_length

        # Token + positional embeddings
        self.token_embedding = nn.Embedding(vocab_size, d_model)
        self.position_embedding = nn.Embedding(context_length, d_model)

        self.dropout = nn.Dropout(dropout)

        # Transformer blocks
        self.blocks = nn.ModuleList([
            TransformerBlock(
                d_model=d_model,
                n_heads=n_heads,
                context_length=context_length,
                dropout=dropout,
            )
            for _ in range(n_layers)
        ])

        # Final layer norm
        self.ln_f = nn.LayerNorm(d_model)

        # Language modeling head
        self.lm_head = nn.Linear(d_model, vocab_size, bias=False)

        # Weight tying (optional but GPT-style)
        self.lm_head.weight = self.token_embedding.weight


    def forward(self, idx):
        """
        idx: (B, T) token indices
        """
        B, T = idx.size()
        assert T <= self.context_length, "Sequence too long!"

        # Token + position embeddings
        token_emb = self.token_embedding(idx)            # (B, T, C)
        pos = torch.arange(0, T, device=idx.device)
        pos_emb = self.position_embedding(pos)            # (T, C)

        x = token_emb + pos_emb
        x = self.dropout(x)

        # Transformer blocks
        for block in self.blocks:
            x = block(x)

        # Final norm + head
        x = self.ln_f(x)
        logits = self.lm_head(x)                           # (B, T, vocab)

        return logits
