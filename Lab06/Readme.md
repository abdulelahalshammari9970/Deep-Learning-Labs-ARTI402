# ARTI 402 — Deep Learning · Lab 6: Regularization (L1, L2, Dropout, Weight Initialisation, Batch Normalisation)

This lab tackles overfitting. A large network (128 hidden units) is trained on very little data (40 samples per class), and regularization tools are built and compared to close the gap between training and validation performance.

## Files

| File | Description |
|---|---|
| `arti402_Lab6_2240005261.ipynb` | Completed lab notebook with all outputs visible |
| `arti402_figures.py` | Helper file that draws the lab diagrams (provided with the course, not modified) |

## Dataset

No external dataset is used. The data is the NNFS spiral dataset, generated in the notebook by `spiral_data`:

- 3 interleaved spiral classes with 2 features per sample.
- Training set: 120 samples, 40 per class (seed 0). This set is small on purpose, to make overfitting likely.
- Validation set: 300 samples (seed 2), used to compare settings.
- Test set: 300 samples (seed 1), kept untouched for the final, honest estimate.

## What was implemented

| Part | Function / Class | Idea |
|---|---|---|
| Exercise 1 | `evaluate` | Returns loss and accuracy with dropout switched off |
| Exercise 2 | `regularization_loss` / `add_regularizer_gradients` | L1 and L2 penalties, in the loss and in the gradient |
| Exercise 3 | `Layer_Dropout` | Inverted dropout: random mask divided by the keep rate, off at test time |
| Exercise 4 | `activation_sizes` | He initialisation (`sqrt(2 / n)`) compared with too-small and too-large scales |
| Exercise 5 | `batchnorm_forward` | Batch normalisation forward pass (normalise per feature, then scale by gamma and shift by beta) |
| Q1 | regularization showdown | Baseline vs L2 vs dropout vs L2 + dropout, 4,000 epochs each |
| Q2 | decision boundaries | Boundaries of baseline, L2 and dropout, plus L2 loss curves |
| Checkpoints | written answers | Dropout keep rate and scaling, He vs Xavier, the role of gamma and beta |

## Results

**Baseline (no regularization):** training accuracy is 0.983 but validation accuracy is only 0.683. Validation loss reaches its lowest point early and then rises for the rest of training, which is a clear sign of overfitting.

**Effect of the penalties on first-layer weights (4,000 epochs):**

| Model | Largest weight | Effect |
|---|---|---|
| No penalty | about 19.5 | very large weights |
| L1 (1e-3) | about 4.8 | most weights pushed to almost exactly zero (sparse) |
| L2 (1e-3) | about 1.7 | every weight shrunk, few exactly zero |

**Weight initialisation (10 layers, width 128, ReLU):**

| Scale | Activation std at layer 10 |
|---|---|
| 0.01 (too small) | 1.92e-11 (vanishes) |
| He, `sqrt(2/128)` | 1.78 (stable) |
| 1.0 (too large) | 1.92e+09 (explodes) |

**Batch normalisation:** input features with mean about 7.04 and std about 2.97 became mean 0.00 and std 1.00 after normalisation. With gamma = 2 and beta = 5, they became mean 5.00 and std 2.00.

**Regularization showdown (4,000 epochs, Adam):**

| Model | Train acc | Val acc | Val loss | Gap |
|---|---|---|---|---|
| Baseline | 0.983 | 0.683 | 1.897 | +0.300 |
| L2 (5e-4) | 0.950 | 0.750 | 1.075 | +0.200 |
| Dropout (0.2) | 0.867 | 0.667 | 1.447 | +0.200 |
| L2 + dropout | 0.850 | 0.633 | 1.152 | +0.217 |

## Key takeaways

- Overfitting shows clearly in the loss curve: training loss keeps falling while validation loss rises.
- Use three sets with different jobs. Training adjusts the weights, validation chooses settings, and test is used once, at the end.
- L2 shrinks every weight. L1 drives many weights to exactly zero.
- Dropout must be on during training and off at test time. Inverted dropout divides by the keep rate so the expected output stays the same.
- He initialisation keeps the signal stable through deep ReLU networks. Batch normalisation keeps activations well scaled during training.
- All regularized models shrank the gap and lowered validation loss. L2 gave the best validation accuracy (0.750).
- A regularized model fits the training data worse on purpose; training accuracy is not the goal.

## How to run

1. Put `arti402_Lab6_2240005261.ipynb` and `arti402_figures.py` in the same folder.
2. Install the requirements: `pip install numpy matplotlib jupyter`
3. Open the notebook and run all cells from top to bottom (Kernel → Restart & Run All). The full run takes less than a minute.

## Notes

- Only the forward pass of batch normalisation is implemented; its backward pass is left to frameworks (Labs 7–8).
- Early stopping is described in the lab but not implemented.
- Results depend on the fixed random seeds used in the notebook.
