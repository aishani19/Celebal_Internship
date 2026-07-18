# GPT-2 From Scratch — 50M Parameter WikiText-103 Edition

A clean PyTorch implementation of GPT-2 trained from scratch on the **WikiText-103** dataset. The model uses 50 million parameters and is served via an interactive Streamlit web UI for next-token prediction.

> Built as part of the Celebal Technologies internship project.

---

## Project Structure

```
gpt2-from-scratch/
├── model_50M.py            # GPT-2 model definition (50M params)
├── train_50M.py            # Training script
├── dataloader_wikitext.py  # Data loader for WikiText .bin files
├── prepare_wikitext.py     # Downloads & tokenizes WikiText-103
├── app.py                  # Streamlit UI
├── requirements.txt        # Python dependencies
├── data/
│   └── wikitext103/        # Tokenized dataset (train.bin, val.bin)
└── logs/
    └── model_50M_wikitext.pt  # Trained model checkpoint
```

---

## Model Architecture

| Parameter       | Value       |
|-----------------|-------------|
| Layers          | 8           |
| Attention Heads | 8           |
| Embedding Size  | 512         |
| Context Length  | 512 tokens  |
| Vocab Size      | 50,257      |
| **Total Params**| **~50M**    |

Architecture: Token embedding + positional embedding → 8× Transformer blocks (LayerNorm → CausalSelfAttention → LayerNorm → MLP with residual connections) → LM head. Uses Flash Attention (`scaled_dot_product_attention`) and weight tying between the embedding and LM head.

---

## Getting Started

### 1. Install dependencies

```bash
pip install -r requirements.txt
```

### 2. Prepare the dataset

Downloads and tokenizes WikiText-103 into binary shards:

```bash
python prepare_wikitext.py
```

This creates `data/wikitext103/train.bin` and `data/wikitext103/val.bin`.

### 3. Train the model

```bash
python train_50M.py
```

Training runs for **1,000 steps** with a cosine learning rate schedule. The final checkpoint is saved to `logs/model_50M_wikitext.pt`.

**Estimated training time:**
- CPU: ~2–3 hours
- GPU: ~10–15 minutes

### 4. Launch the Streamlit UI

```bash
streamlit run app.py
```

Open your browser at `http://localhost:8501`. Type a prompt and click **Generate** to see the top-k next token predictions with probabilities.

---

## Training Configuration

| Hyperparameter  | Value   |
|-----------------|---------|
| Total Steps     | 1,000   |
| Batch Size      | 4       |
| Context Length  | 512     |
| Max LR          | 3e-4    |
| Min LR          | 3e-5    |
| Warmup Steps    | 50      |
| Weight Decay    | 0.1     |
| Optimizer       | AdamW   |
| LR Schedule     | Cosine  |

---

## References

- [Language Models are Unsupervised Multitask Learners (GPT-2 Paper)](https://cdn.openai.com/better-language-models/language_models_are_unsupervised_multitask_learners.pdf)
- [Attention is All You Need](https://arxiv.org/abs/1706.03762)
- [FlashAttention: Fast and Memory-Efficient Exact Attention](https://arxiv.org/abs/2205.14135)
- [WikiText-103 Dataset](https://huggingface.co/datasets/wikitext)
- Andrej Karpathy's Video Tutorial on GPT

## Acknowledgments

This implementation is inspired by Andrej Karpathy's tutorial on building GPT from scratch.