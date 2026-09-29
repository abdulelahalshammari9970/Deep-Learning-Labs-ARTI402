# ARTI 402 — Deep Learning · Lab 4: Recurrent Networks (RNN, LSTM, GRU)

This lab builds recurrent networks from scratch in NumPy and shows experimentally why a plain RNN forgets while an LSTM remembers. The key idea is weight sharing across time: one cell, applied at every step of a sequence.

## Files

| File | Description |
|---|---|
| `arti402_Lab4_2240005261.ipynb` | Completed lab notebook with all outputs visible |
| `arti402_figures.py` | Helper file that draws the lab diagrams (provided with the course, not modified) |

## Dataset

No external dataset is used. The data is generated synthetically in the notebook by `make_sequences`:

- Each sequence has `T` time steps with 4 features per step.
- Step 1 carries the answer: a spike of 3.0 in feature 0, 1 or 2 gives the class (3 classes).
- All later steps are random noise.
- To classify correctly, a model must remember step 1 through `T − 1` steps of noise.

## What was implemented

| Part | Function | Idea |
|---|---|---|
| Exercise 1 | `rnn_step` | `h_new = tanh(x·Wx + h_prev·Wh + b)` |
| Exercise 2 | `rnn_forward` | Unroll one RNN cell over a sequence of any length |
| Exercise 3 | `rnn_params`, `gru_params`, `lstm_params` | Parameter counts (1, 3 and 4 blocks), independent of sequence length |
| Exercise 4 | `lstm_step` | Forget gate, input gate, candidate memory, output gate, `c = f*c_prev + i*g`, `h = o*tanh(c)` |
| Q1 | `final_hidden_states` | Frozen RNN/LSTM as a feature extractor, trained dense head on the final hidden state |
| Q2 | sequence-length experiment | Test accuracy of RNN vs LSTM for T = 2 to 80 |
| Q3 | forget-gate experiment | Effect of the forget-gate bias on memory at T = 20 |
| Q4 + Checkpoints | written answers | Explanations of vanishing memory, gate compounding and LSTM limits |

Parameter counts for 4 inputs and 12 hidden units: RNN = 204, GRU = 612, LSTM = 816.

## Results

**RNN vs LSTM across sequence lengths (test accuracy, chance = 0.33):**

| T | RNN | LSTM |
|---|---|---|
| 2 | 1.000 | 1.000 |
| 5 | 0.608 | 0.992 |
| 10 | 0.292 | 1.000 |
| 20 | 0.317 | 0.983 |
| 40 | 0.275 | 0.942 |
| 80 | 0.333 | 0.792 |

**Forget-gate bias at T = 20:**

| Forget bias | % kept per step | Test accuracy |
|---|---|---|
| −2.0 | 0.12 | 0.283 |
| 0.0 | 0.50 | 0.292 |
| +2.0 | 0.88 | 0.600 |
| +4.0 | 0.98 | 0.983 |

## Key takeaways

- A plain RNN passes its memory through the same weights and `tanh` at every step, so the signal from step 1 shrinks repeatedly (vanishing memory) and the RNN drops to chance level by T = 10.
- The LSTM's long-term memory update `c = f*c_prev + i*g` has no weight matrix or `tanh` on its path, so with a forget gate close to 1 the memory is kept for many steps.
- The fraction kept per step compounds: `0.88^20 ≈ 0.078` versus `0.98^20 ≈ 0.668`.
- An LSTM does not have perfect memory; its accuracy drops at T = 80.

## How to run

1. Put `arti402_Lab4_2240005261.ipynb` and `arti402_figures.py` in the same folder.
2. Install the requirements: `pip install numpy matplotlib jupyter`
3. Open the notebook and run all cells from top to bottom (Kernel → Restart & Run All).

## Notes

- The recurrent weights are frozen (no backpropagation through time in this lab); only the dense head is trained.
- The LSTM gate biases are set by hand: forget gate +4 and input gate −2.
- The GRU is only described and counted, not implemented.
