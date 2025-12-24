import sentencepiece as spm
from pathlib import Path


CHAT_DATA_PATH = Path("data/processed/chat_data.txt")
TOKENIZER_DIR = Path("tokenizer")

VOCAB_SIZE = 4000
MODEL_PREFIX = TOKENIZER_DIR / "spm"

SPECIAL_TOKENS = [
    "<|user|>",
    "<|assistant|>",
    "<|end|>",
]


def train_tokenizer():
    TOKENIZER_DIR.mkdir(exist_ok=True)

    spm.SentencePieceTrainer.train(
        input=str(CHAT_DATA_PATH),
        model_prefix=str(MODEL_PREFIX),
        vocab_size=VOCAB_SIZE,
        model_type="bpe",
        user_defined_symbols=SPECIAL_TOKENS,
        character_coverage=1.0,
        pad_id=0,
        unk_id=1,
        bos_id=-1,
        eos_id=-1,
    )

    print("SentencePiece tokenizer trained.")
    print("Saved files:")
    print(" - tokenizer/spm.model")
    print(" - tokenizer/spm.vocab")


if __name__ == "__main__":
    train_tokenizer()
