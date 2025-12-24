import torch
from model.gpt import MiniGPT

model = MiniGPT(
    vocab_size=65,
    context_length=128,
    d_model=256,
    n_heads=4,
    n_layers=4,
    dropout=0.1,
)

x = torch.randint(0, 65, (2, 16))  # (B=2, T=16)
logits = model(x)

print("Input shape:", x.shape)
print("Logits shape:", logits.shape)
