import torch
import torch.nn as nn
import math

class CausalSelfAttention(nn.Module):
    def __init__(self, d_model, n_heads, context_length, dropout):
        super().__init__()

        assert d_model % n_heads == 0

        self.n_heads = n_heads
        self.head_dim = d_model // n_heads

        self.key   = nn.Linear(d_model, d_model)
        self.query = nn.Linear(d_model, d_model)
        self.value = nn.Linear(d_model, d_model)

        self.proj = nn.Linear(d_model, d_model)
        self.dropout = nn.Dropout(dropout)

        # Causal mask (registered as buffer, not parameter)
        self.register_buffer(
            "mask",
            torch.tril(torch.ones(context_length, context_length))
        )

    def forward(self, x):
        """
        x: (B, T, C)
        """
        B, T, C = x.size()

        # Compute Q, K, V
        q = self.query(x)
        k = self.key(x)
        v = self.value(x)

        # Split heads
        q = q.view(B, T, self.n_heads, self.head_dim).transpose(1, 2)
        k = k.view(B, T, self.n_heads, self.head_dim).transpose(1, 2)
        v = v.view(B, T, self.n_heads, self.head_dim).transpose(1, 2)

        # Attention scores
        att = (q @ k.transpose(-2, -1)) / math.sqrt(self.head_dim)

        # Apply causal mask
        att = att.masked_fill(
            self.mask[:T, :T] == 0,
            float("-inf")
        )

        att = torch.softmax(att, dim=-1)
        att = self.dropout(att)

        # Weighted sum
        out = att @ v

        # Recombine heads
        out = out.transpose(1, 2).contiguous().view(B, T, C)

        return self.proj(out)

class MLP(nn.Module):
    def __init__(self, d_model, dropout):
        super().__init__()

        self.net = nn.Sequential(
            nn.Linear(d_model, 4 * d_model),
            nn.GELU(),
            nn.Linear(4 * d_model, d_model),
            nn.Dropout(dropout),
        )

    def forward(self, x):
        return self.net(x)


class TransformerBlock(nn.Module):
    def __init__(self, d_model, n_heads, context_length, dropout):
        super().__init__()

        self.ln1 = nn.LayerNorm(d_model)
        self.attn = CausalSelfAttention(
            d_model=d_model,
            n_heads=n_heads,
            context_length=context_length,
            dropout=dropout,
        )

        self.ln2 = nn.LayerNorm(d_model)
        self.mlp = MLP(d_model, dropout)

    def forward(self, x):
        # Attention with residual
        x = x + self.attn(self.ln1(x))

        # MLP with residual
        x = x + self.mlp(self.ln2(x))

        return x
