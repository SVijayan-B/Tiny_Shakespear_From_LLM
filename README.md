# ShakespeareGPT — Mini ChatGPT Built From Scratch (PyTorch)

ShakespeareGPT is a fully custom **GPT-style chatbot LLM** implemented **from scratch** using PyTorch.  
The model is trained on Shakespeare text converted into **chat-format data** and is capable of **multi-turn conversational responses** in a Shakespearean style.

This project demonstrates **end-to-end LLM engineering**, covering:
- Tokenization
- Transformer architecture
- Training
- Inference
- Chat behavior
- Generation quality control

The entire system runs **locally** on a consumer GPU (RTX 3050, 6GB VRAM).

---

## Key Features

- Decoder-only Transformer (GPT-style) implemented from scratch
- SentencePiece **BPE tokenizer**
- Chat-formatted training data (`<|user|>`, `<|assistant|>`, `<|end|>`)
- Mixed-precision (FP16) training
- Multi-turn conversational memory
- Top-k + Top-p (nucleus) sampling
- Repetition penalty for improved text quality
- Fully local training and inference (no external APIs)

---

## Model Architecture

- Decoder-only Transformer (GPT)
- Token embedding + positional embedding
- Masked multi-head self-attention (causal)
- Feed-forward MLP
- Pre-LayerNorm with residual connections
- Weight tying between embedding and output layer

Approximate model size: **~15M parameters**

---

## Tokenization

This project uses **SentencePiece BPE tokenization**.

### Why BPE?

- Shorter sequences than character-level tokenization
- Better semantic understanding
- Improved coherence and generation quality

### Special Tokens

  <|user|>
  <|assistant|>
  <|end|>

These tokens define chat behavior and response boundaries.

---

## Chat-Formatted Training Data

Raw Shakespeare text is converted into chat-style samples:

  <|user|> What is honor?
  <|assistant|> Honor is the crown worn by a virtuous soul.
  <|end|>

The model learns:
- Role separation
- Question → answer behavior
- When to stop generating

This is the key difference between **GPT** and **ChatGPT-style models**.

---

## Training Configuration

- GPU: NVIDIA RTX 3050 (6GB VRAM)
- Context length: 256
- Batch size: 16
- Optimizer: AdamW
- Precision: FP16 (AMP)
- Epochs: 3

### Training Loss

  Epoch 1 ≈ 2.65
  Epoch 2 ≈ 1.51
  Epoch 3 ≈ 0.95


A final loss below **1.0** indicates strong learning of structure, intent, and style.

---

## Inference & Generation

### Prompt Template

<|user|> {user input}
<|assistant|>


### Generation Controls
- Temperature sampling
- Top-k sampling
- Top-p (nucleus) sampling
- Repetition penalty
- Stop generation at `<|end|>`

These controls significantly improve output quality without retraining.

---

## Multi-Turn Conversational Memory

Conversation memory is implemented via **prompt accumulation**, not model state.

- Maintains last N conversation turns
- Automatically trimmed to fit context length
- GPU-safe and scalable

---

## Example Outputs

### Example 1

  You: What is love?
  Bot: Love is a tender madness, born of sighs and fed by longing hearts.


### Example 2 (Multi-turn)

  You: What is honor?
  Bot: Honor is the jewel set upon the brow of virtue.

  You: And why do men betray it?
  Bot: Because desire oft whispers louder than conscience, and frailty bends the will.


### Example 3

  You: Give me advice on life.
  Bot: Walk humbly, speak honestly, and let thy deeds outshine thy words.


---

## How to Run

### Train the model

  python -m training.train

### Start the chatbot
python -m inference.chat

---

## Why This Project Matters

This project avoids black-box abstractions and demonstrates:
- Deep understanding of Transformer internals
- Practical LLM training on limited hardware
- Chat behavior driven by data design
- Real-world inference techniques
- End-to-end LLM system engineering

---

## Skills Demonstrated

- PyTorch
- Transformer architectures
- Tokenization (BPE)
- Autoregressive language modeling
- Mixed-precision training
- GPU optimization
- Prompt engineering
- LLM inference optimization

---

## Future Improvements

- RoPE positional encoding
- KV-cache for faster inference
- FastAPI-based serving
- Larger datasets
- Evaluation metrics (perplexity, BLEU-style)
- Model scaling

---

## Final Note

This project demonstrates that modern LLM systems are not magic — they are the result of:
- Careful architecture design
- Correct data formatting
- Efficient training
- Thoughtful inference control

Building this system from scratch represents a strong foundation in **LLM engineering**.


