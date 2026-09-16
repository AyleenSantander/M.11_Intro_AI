# Generative AI Essentials — GPT Architecture and Text Generation

Exploring generative pre-trained transformers through architectural inspection, empirical measurement, and implementation from scratch, using public-domain text from Project Gutenberg.

**Ayleen Santander** · Assignment 13, Module 11 · Business Intelligence and Data Analytics Diploma, Willis College Online

---

## Overview

This project examines how GPT models work from three angles rather than treating them as a black box:

1. **Inspection** — reading the architecture directly out of a loaded GPT-2 model and extracting its attention weights
2. **Measurement** — using perplexity to quantify what pretraining actually encoded, and where it fails
3. **Implementation** — building two text generation models from scratch to isolate architecture from scale

## Dataset

Text is drawn from [Project Gutenberg](https://www.gutenberg.org/), a public-domain digital library. Two acquisition methods are used:

| Source | Method | Size |
|---|---|---|
| *Pride and Prejudice* (ID 1342) | HTTP request, boilerplate stripped with regex | 744,279 chars |
| *Emma* | NLTK Gutenberg corpus (offline) | 887,071 chars |

The offline route avoids the rate limiting Project Gutenberg applies to repeated requests from a single address. Downloaded text is cached to `pride.txt` for reuse.

## Tech stack

- **PyTorch** 2.11 — model implementation and training
- **Hugging Face Transformers** 5.16 — pre-trained GPT-2 and tokenizer
- **NLTK** — offline Gutenberg corpus
- **matplotlib / seaborn** — attention heatmaps and loss curves
- **Google Colab** — GPU runtime

## Structure

```
.
├── M11.A.13_Generative_AI_Essentials.ipynb    # main notebook
├── Generative_AI_Essentials_Report.docx        # full report (12 pages)
├── Generative_AI_Essentials_Report_Short.docx  # condensed report (4 pages)
└── README.md
```

## What the notebook does

### Part 1 — Dataset preparation
Loads text from Project Gutenberg, strips licence boilerplate, and chunks it into 128-token training blocks (1,615 chunks from *Pride and Prejudice*).

### Part 2 — Architecture exploration
Reads the configuration out of GPT-2 small and extracts attention weights for inspection.

| Component | Value |
|---|---|
| Decoder blocks / attention heads | 12 / 12 |
| Embedding dimension (per head) | 768 (64) |
| Context length / vocabulary | 1,024 / 50,257 |
| Total parameters | 124,439,808 |
| Feed-forward / attention / embedding | 56.7M / 28.3M / 39.4M |

Attention weights require loading with `attn_implementation="eager"` — the default fused kernel never materializes the attention matrix:

```python
model = AutoModelForCausalLM.from_pretrained("gpt2", attn_implementation="eager")
attn_out = model(**inp, output_attentions=True)
# 12 tensors of shape (batch, heads, seq, seq)
```

Verified: every row sums to 1.0 after softmax, and the entire upper triangle is zero, confirming the causal mask. The mechanism is also reproduced manually from the raw Q/K/V matrices.

### Part 3 — Training
Two models trained from scratch on 200,000 characters (89-character vocabulary), under identical conditions:

| | Character LSTM | Mini-GPT |
|---|---|---|
| Architecture | 2-layer LSTM, 256 hidden | 4 blocks, 6 heads, 192-dim |
| Parameters | 955,865 | 1,836,377 |
| Steps | 500 | 3,000 |
| Initial loss | 1.989 | 4.715 |
| Final loss (train / val) | 1.265 / 1.326 | 1.209 / 1.370 |

The mini-GPT implements every transformer component explicitly — scaled dot-product attention, causal masking via `masked_fill`, residual connections, layer normalization, and the 4× feed-forward expansion.

## Key findings

**Perplexity reveals what pretraining encoded.** Measured across four text types with GPT-2:

| Text | Perplexity |
|---|---|
| *Emma* (Austen prose) | 36.18 |
| *Pride and Prejudice* excerpt | 257.74 |
| Shuffled nonsense | 899.80 |
| Contemporary slang | 1,332.91 |

The 37-fold spread shows competence is bounded by the training distribution. GPT-2's corpus was scraped in 2019, so vocabulary coined later falls outside what it learned.

**The model learns distributions, not documents.** Prompted with the opening line of *Pride and Prejudice*, GPT-2 produced text about monetary policy and academic citations rather than continuing the novel — despite the book being present in its training corpus.

**Architecture beats recurrence at equal scale.** Both models saw identical data with an identical objective. The mini-GPT reached lower loss from a much worse starting point, because self-attention gives every position direct access to all previous positions while the LSTM must compress context into a fixed-size hidden state.

**Fluency and meaning are separate properties.** At one to two million parameters, both models produce correct spelling, paired quotation marks, and period-appropriate rhythm — with no semantic coherence. Style transfer is achievable at small scale; meaningful content is not.

## Ethical considerations

| Concern | Evidence from this project |
|---|---|
| Factual unreliability | GPT-2 generated fabricated citations with fictional authors, journals, and page numbers |
| Training data bias | Perplexity far worse on text outside the 2019 web corpus; same applies to dialects and non-dominant registers |
| Copyright and consent | This project uses public-domain text; most commercial models were trained on scraped material |
| Misinformation | The same fluency enables propaganda and phishing at negligible cost |
| Environmental and labour cost | Training energy and annotator working conditions are rarely discussed alongside capability |

## Running it

Open the notebook in Google Colab with a GPU runtime (Runtime → Change runtime type → T4 GPU), then **Runtime → Restart and run all**.

```bash
pip install transformers torch nltk
```

Expect roughly 3–4 minutes for the LSTM and mini-GPT training loops on GPU. The GPT-2 inspection cells run on CPU without issue.

> **Note:** run cells in order. Several cells reassign `model`, `text`, and `out`, so running them out of sequence produces stale-variable errors.

## License

Code is available for educational use. Source texts are public domain via Project Gutenberg.
