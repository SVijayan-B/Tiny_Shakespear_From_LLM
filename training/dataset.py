import torch
from torch.utils.data import Dataset
from pathlib import Path
import sentencepiece as spm


class ChatGPTDataset(Dataset):
    def __init__(self, text_path, tokenizer_path, context_length):
        """
        text_path: path to chat_data.txt
        tokenizer_path: path to spm.model
        context_length: max sequence length
        """
        self.context_length = context_length

        # Load SentencePiece tokenizer
        self.sp = spm.SentencePieceProcessor()
        self.sp.load(str(tokenizer_path))

        # Load chat text
        with open(text_path, "r", encoding="utf-8") as f:
            text = f.read()

        # Encode entire dataset once (IMPORTANT)
        ids = self.sp.encode(text)
        self.tokens = torch.tensor(ids, dtype=torch.long)

        print(f"Total tokens: {len(self.tokens)}")

    def __len__(self):
        return len(self.tokens) - self.context_length

    def __getitem__(self, idx):
        x = self.tokens[idx : idx + self.context_length]
        y = self.tokens[idx + 1 : idx + self.context_length + 1]
        return x, y
