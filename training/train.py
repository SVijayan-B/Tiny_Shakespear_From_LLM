import torch
import torch.nn.functional as F
from torch.utils.data import DataLoader
from torch.optim import AdamW
from torch.cuda.amp import autocast, GradScaler
from tqdm import tqdm
from pathlib import Path

import config
from model.gpt import MiniGPT
from training.dataset import ChatGPTDataset


# =========================
# Setup
# =========================
DEVICE = "cuda" if torch.cuda.is_available() else "cpu"
CHECKPOINT_PATH = Path("checkpoints/mini_gpt_chat.pt")
CHECKPOINT_PATH.parent.mkdir(exist_ok=True)


# =========================
# Dataset & Loader
# =========================
dataset = ChatGPTDataset(
    text_path="data/processed/chat_data.txt",
    tokenizer_path=config.TOKENIZER_MODEL,
    context_length=config.CONTEXT_LENGTH,
)

loader = DataLoader(
    dataset,
    batch_size=config.BATCH_SIZE,
    shuffle=True,
    pin_memory=True,
    drop_last=True,
)


# =========================
# Model
# =========================
model = MiniGPT(
    vocab_size=config.VOCAB_SIZE,
    context_length=config.CONTEXT_LENGTH,
    d_model=config.D_MODEL,
    n_heads=config.N_HEADS,
    n_layers=config.N_LAYERS,
    dropout=config.DROPOUT,
).to(DEVICE)


# =========================
# Optimizer & AMP
# =========================
optimizer = AdamW(model.parameters(), lr=config.LEARNING_RATE)
scaler = GradScaler()


# =========================
# Training Loop
# =========================
model.train()

for epoch in range(config.EPOCHS):
    pbar = tqdm(loader, desc=f"Epoch {epoch+1}/{config.EPOCHS}")

    for step, (x, y) in enumerate(pbar):
        x = x.to(DEVICE)
        y = y.to(DEVICE)

        optimizer.zero_grad()

        with autocast():
            logits = model(x)
            loss = F.cross_entropy(
                logits.view(-1, logits.size(-1)),
                y.view(-1),
            )

        scaler.scale(loss).backward()
        scaler.step(optimizer)
        scaler.update()

        pbar.set_postfix(loss=f"{loss.item():.4f}")

    # Save checkpoint after each epoch
    torch.save(model.state_dict(), CHECKPOINT_PATH)
    print(f"Saved checkpoint to {CHECKPOINT_PATH}")
