import torch
import torch.nn.functional as F
import sentencepiece as spm
from collections import deque

import config
from model.gpt import MiniGPT


# =========================
# Device
# =========================
DEVICE = "cuda" if torch.cuda.is_available() else "cpu"


# =========================
# Special tokens
# =========================
USER_TOKEN = "<|user|>"
ASSISTANT_TOKEN = "<|assistant|>"
END_TOKEN = "<|end|>"


# =========================
# Load tokenizer
# =========================
sp = spm.SentencePieceProcessor()
sp.load(config.TOKENIZER_MODEL)

end_token_id = sp.piece_to_id(END_TOKEN)


# =========================
# Load model
# =========================
model = MiniGPT(
    vocab_size=config.VOCAB_SIZE,
    context_length=config.CONTEXT_LENGTH,
    d_model=config.D_MODEL,
    n_heads=config.N_HEADS,
    n_layers=config.N_LAYERS,
    dropout=config.DROPOUT,
).to(DEVICE)

model.load_state_dict(
    torch.load("checkpoints/mini_gpt_chat.pt", map_location=DEVICE)
)
model.eval()


# =========================
# Conversation memory
# =========================
MAX_TURNS = 4  # last N user-assistant exchanges
conversation = deque(maxlen=MAX_TURNS * 2)


# =========================
# Sampling helpers
# =========================
def top_p_sampling(logits, top_p=0.9):
    """
    Nucleus (top-p) sampling.
    logits: (1, vocab_size)
    """
    probs = F.softmax(logits, dim=-1)

    # Sort probabilities
    sorted_probs, sorted_indices = torch.sort(probs, descending=True)

    cumulative_probs = torch.cumsum(sorted_probs, dim=-1)

    # Mask tokens with cumulative prob above top_p
    sorted_indices_to_remove = cumulative_probs > top_p

    # Ensure at least one token is kept
    sorted_indices_to_remove[..., 0] = False

    # Map back to original indices
    indices_to_remove = sorted_indices[sorted_indices_to_remove]

    logits[:, indices_to_remove] = -float("inf")
    return logits


# =========================
# Generation function
# =========================
@torch.no_grad()
def generate_reply(user_input, max_new_tokens=200, top_p=0.9, repetition_penalty=1.2):
    # Build prompt from conversation memory
    prompt = ""
    for turn in conversation:
        prompt += turn + "\n"

    prompt += f"{USER_TOKEN} {user_input}\n{ASSISTANT_TOKEN}"

    input_ids = sp.encode(prompt)
    idx = torch.tensor([input_ids], dtype=torch.long).to(DEVICE)

    generated_tokens = set()

    for _ in range(max_new_tokens):
        idx_cond = idx[:, -config.CONTEXT_LENGTH:]
        logits = model(idx_cond)
        logits = logits[:, -1, :] / config.TEMPERATURE

        # Repetition penalty
        for token_id in generated_tokens:
            logits[0, token_id] /= repetition_penalty

        # Top-k
        if config.TOP_K is not None:
            v, _ = torch.topk(logits, config.TOP_K)
            logits[logits < v[:, [-1]]] = -float("inf")

        # Top-p
        logits = top_p_sampling(logits, top_p)

        probs = F.softmax(logits, dim=-1)
        next_token = torch.multinomial(probs, num_samples=1)

        if next_token.item() == end_token_id:
            break

        generated_tokens.add(next_token.item())
        idx = torch.cat([idx, next_token], dim=1)

    decoded = sp.decode(idx[0].tolist())
    reply = decoded.split(ASSISTANT_TOKEN)[-1]
    reply = reply.split(END_TOKEN)[0].strip()

    # Update memory
    conversation.append(f"{USER_TOKEN} {user_input}")
    conversation.append(f"{ASSISTANT_TOKEN} {reply} {END_TOKEN}")

    return reply


# =========================
# Interactive loop
# =========================
print("\n🧙 ShakespeareGPT (memory + quality) is ready. Type 'exit' to quit.\n")

while True:
    user_input = input("You: ").strip()

    if user_input.lower() in {"exit", "quit"}:
        print("Bot: Fare thee well, good soul.")
        break

    response = generate_reply(user_input)
    print(f"\nBot: {response}\n")
