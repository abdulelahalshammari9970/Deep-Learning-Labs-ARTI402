# ARTI 402 — Deep Learning · Lab 10: Attention (Scaled Dot-Product, Self-Attention, Causal Mask)

This lab builds attention from scratch in NumPy, with no training at all. It starts from the fixed-size bottleneck that limited the RNN/LSTM in Lab 4, implements the attention formula, shows why it is scaled, and ends with self-attention and the causal mask that separates a translation encoder from a GPT-style decoder.

## Files

| File | Description |
|---|---|
| `arti402_Lab10_2240005261.ipynb` | Completed lab notebook with all outputs visible |
| `arti402_figures.py` | Helper file that draws the lab diagrams (provided with the course, not modified) |

## Dataset

No external dataset is used and nothing is trained. All data is generated in the notebook with fixed random seeds:

- Exercise 2: four one-hot keys with four stored values, so the result can be checked by hand.
- Stage 4: random key-value pairs with 16 features, 400 trials for each sequence length.
- Stage 5: the sentence "the cat sat on the mat" with random 8-number embeddings (seed 3).
- Assessment: a random 5 × 6 sequence (seed 1).

## What was implemented

| Part | Function / Class | Idea |
|---|---|---|
| Exercise 1 | `attention(Q, K, V, mask)` | Scores = `QKᵀ / √d_k`, then softmax along the last axis, then weights times `V`. Optional boolean mask |
| Exercise 2 | one-hot lookup | Exact, 5x sharper and split queries on four memory slots |
| Exercise 3 | scaling test | Raw and scaled scores compared for d_k = 4, 64 and 512 |
| Exercise 4 | `retrieval_trial` | Attention retrieval compared with one averaged summary vector, for sequence lengths 2 to 64 |
| Exercise 5 | self-attention | `attention(X, X, X)` on a six-word sentence, shown as a heatmap |
| Q1 | causal mask | Lower-triangular mask with `np.tril`, passed to `attention` |
| Q2 | mask comparison | Weight each position puts on itself, with and without the mask |
| Checkpoints | written answers | Why the first row of the causal weights is `[1, 0, 0, 0, 0]`, and what the mask costs |

## Results

**One-hot lookup (4 slots, scores scaled by √4):**

| Query | Weights | Output |
|---|---|---|
| Exact, `[0, 1, 0, 0]` | `[0.215 0.355 0.215 0.215]` | `[24.302 1.43]` |
| 5x sharper | `[0.066 0.802 0.066 0.066]` | `[21.317 1.132]` |
| Torn between slots 1 and 2 | `[0.409 0.409 0.091 0.091]` | `[18.649 0.865]` |

**Why divide by √d_k (largest softmax weight over 6 keys):**

| d_k | Raw score range | Max weight (raw) | Max weight (scaled) |
|---|---|---|---|
| 4 | -5.06 to 3.49 | 0.332 | 0.272 |
| 64 | -11.37 to 4.99 | 0.969 | 0.320 |
| 512 | -24.36 to 11.33 | 1.000 | 0.300 |

**Retrieval accuracy (400 trials per length):**

| Sequence length | 2 | 4 | 8 | 16 | 32 | 64 |
|---|---|---|---|---|---|---|
| Attention | 1.000 | 0.993 | 0.990 | 0.990 | 0.980 | 0.958 |
| One summary vector | 0.430 | 0.225 | 0.135 | 0.050 | 0.035 | 0.015 |
| Chance | 0.500 | 0.250 | 0.125 | 0.062 | 0.031 | 0.016 |

**Self-attention:** the weight matrix is 6 × 6 and every row sums to 1. The two rows for "the" (positions 0 and 4) are identical (`[0.431 0.011 0.075 0.012 0.431 0.038]`), because attention has no notion of word order.

**Causal mask (T = 5):** every weight above the diagonal is exactly 0, and position 0 can only see itself (`W_c[0] = [1, 0, 0, 0, 0]`).

| Position | Keys visible | Self-weight, unmasked | Self-weight, masked |
|---|---|---|---|
| 0 | 1 | 0.585 | 1.000 |
| 1 | 2 | 0.556 | 0.577 |
| 2 | 3 | 0.412 | 0.480 |
| 3 | 4 | 0.695 | 0.758 |
| 4 | 5 | 0.358 | 0.358 |

## Key takeaways

- Attention is a soft, differentiable lookup: it compares a query with every key, turns the scores into weights with softmax, and returns a weighted blend of the values.
- The output is a blend, not a choice. Even an exact query leaves some weight on the other slots, and a query torn between two slots returns a mix of both values.
- The size of the query controls how decisive the lookup is: multiplying it by 5 raised the winning weight from 0.355 to 0.802.
- Dividing by √d_k keeps the softmax from saturating. At d_k = 512 the unscaled softmax puts a weight of 1.000 on one key, so every other gradient is zero.
- Attention does not ration information. Its accuracy stays near 0.96 at length 64, while one averaged summary vector falls to chance level.
- Self-attention uses the same sequence as query, key and value, so every position can read every other position directly. It contains no information about order, which is why transformers add positional encoding.
- The causal mask sets future scores to -1e9 before softmax, so they become exactly zero after it. This one change turns an encoder into a GPT-style decoder.
- Masking redistributes the softmax over fewer keys, so the self-weight can only go up or stay the same. Early positions change the most, and the last position does not change.

## How to run

1. Put `arti402_Lab10_2240005261.ipynb` and `arti402_figures.py` in the same folder.
2. Install the requirements: `pip install numpy matplotlib jupyter`
3. Open the notebook and run all cells from top to bottom (Kernel → Restart & Run All). The full run takes a few seconds on CPU.

## Notes

- This lab is one attention head. Multi-head attention and positional encoding are only described in the notebook and were not implemented.
- Nothing is trained, so the results come from the architecture itself, not from an optimizer.
- Results depend on the fixed random seeds used in the notebook.
