# =========================
# Tokenizer / Data
# =========================
TOKENIZER_MODEL = "tokenizer/spm.model"
VOCAB_SIZE = 4000          # must match SentencePiece vocab
CONTEXT_LENGTH = 256       # chat needs more context


# =========================
# Model Architecture
# =========================
D_MODEL = 256
N_HEADS = 4
N_LAYERS = 4
DROPOUT = 0.1


# =========================
# Training
# =========================
BATCH_SIZE = 16            # BPE is heavier than char-level
LEARNING_RATE = 2e-4      # slightly lower for stability
EPOCHS = 3                # chat needs more than 1 epoch


# =========================
# Generation
# =========================
TEMPERATURE = 0.7
TOP_K = 40
