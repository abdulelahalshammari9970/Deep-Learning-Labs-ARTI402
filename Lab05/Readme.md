# ARTI 402 — Deep Learning · Lab 5: Optimizers (SGD, Mini-batches, Momentum, RMSProp, Adam)

This lab replaces numerical gradients with real backpropagation, then builds and compares the main optimizers on the same network and data. Only the way the weights are updated changes between runs.

## Files

| File | Description |
|---|---|
| `arti402_Lab5_2240005261.ipynb` | Completed lab notebook with all outputs visible |
| `arti402_figures.py` | Helper file that draws the lab diagrams (provided with the course, not modified) |

## Dataset

No external dataset is used. The data is the NNFS spiral dataset, generated in the notebook by `spiral_data`:

- 3 interleaved spiral classes with 2 features per sample.
- Training set: 300 samples (seed 0). Test set: 300 new samples (seed 1), never used for training.
- A straight line cannot separate the classes, so the network must learn curved boundaries.

## What was implemented

| Part | Function / Class | Idea |
|---|---|---|
| Exercise 1 | `SpiralNet.forward` / `backward` | Full forward and backward pass (dense → ReLU → dense → softmax + loss), checked against numerical gradients |
| Exercise 2 | `Optimizer_SGD` | Step against the gradient |
| Exercise 3 | `get_batches` | Shuffle and cut the data into mini-batches |
| Exercise 4 | `Optimizer_SGD_Momentum` | Remembers direction, damps zig-zagging |
| Exercise 5 | `Optimizer_RMSprop` | A separate step size for every parameter (remembers size) |
| Exercise 6 | `Optimizer_Adam` | Momentum + RMSProp with start-up correction |
| Q1 | optimizer race | 5 optimizers, 10,000 epochs, full batch |
| Q2 | Adam learning-rate test | 5 learning rates, 2,000 epochs each |
| Checkpoints | written answers | Batch size trade-off, momentum vs RMSProp, Adam's correction |

## Results

**Backpropagation vs numerical gradient:** the largest difference is below 1e-9, and backprop is about 342x faster (0.42 ms vs 144.9 ms per step for 387 parameters).

**Batch size (plain SGD, learning rate 1.0, 1,000 epochs):**

| Batch size | Steps | Accuracy |
|---|---|---|
| 300 (full) | 1,001 | 0.427 |
| 100 | 3,003 | 0.583 |
| 32 | 10,010 | 0.797 |
| 8 | 38,038 | 0.570 |

**Optimizer race (10,000 epochs, full batch):**

| Optimizer | Train acc | Train loss | Test acc |
|---|---|---|---|
| SGD | 0.703 | 0.639 | 0.637 |
| SGD + decay | 0.717 | 0.729 | 0.567 |
| SGD + momentum | 0.970 | 0.082 | 0.777 |
| RMSProp | 0.890 | 0.269 | 0.807 |
| Adam | 0.967 | 0.091 | 0.820 |

**Adam at different learning rates (2,000 epochs):**

| Learning rate | Train acc | Worst loss seen |
|---|---|---|
| 0.0005 | 0.670 | 1.10 |
| 0.005 | 0.927 | 1.10 |
| 0.05 | 0.950 | 1.10 |
| 0.5 | 0.503 | 2.73 |
| 2.0 | 0.340 | 10.45 |

## Key takeaways

- Backpropagation gives the same gradients as the numerical method but is hundreds of times faster.
- Mini-batches give more updates per epoch, but a batch size that is too small makes the gradient too noisy.
- Momentum remembers direction and RMSProp remembers size; Adam combines both and uses a start-up correction (`iterations + 1`) to avoid biased and zero-division steps.
- Adaptive optimizers still need a sensible learning rate: Adam failed at 0.5 and 2.0.
- Higher training accuracy does not mean higher test accuracy: Adam reached 0.967 on the training data but 0.820 on new data.

## How to run

1. Put `arti402_Lab5_2240005261.ipynb` and `arti402_figures.py` in the same folder.
2. Install the requirements: `pip install numpy matplotlib jupyter`
3. Open the notebook and run all cells from top to bottom (Kernel → Restart & Run All). The full run takes about 1 minute.

## Notes

- Gradients come from backpropagation; `numerical_gradient` is used only to check it.
- Results depend on the fixed random seeds used in the notebook.
