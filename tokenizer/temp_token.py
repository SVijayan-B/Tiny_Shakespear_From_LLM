import sentencepiece as spm

sp = spm.SentencePieceProcessor()
sp.load("tokenizer/spm.model")

text = "<|user|> What is love?\n<|assistant|>"
ids = sp.encode(text)

print("Tokens:", ids)
print("Decoded:", sp.decode(ids))
