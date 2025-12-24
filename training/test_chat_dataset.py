from training.dataset import ChatGPTDataset

dataset = ChatGPTDataset(
    text_path="data/processed/chat_data.txt",
    tokenizer_path="tokenizer/spm.model",
    context_length=64,
)

print("Dataset size:", len(dataset))

x, y = dataset[0]

print("Input tokens:", x[:20])
print("Target tokens:", y[:20])
