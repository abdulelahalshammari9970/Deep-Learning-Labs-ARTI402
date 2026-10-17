# ARTI 402 — Deep Learning · Lab 7: TensorFlow and Keras (Sequential Models, Callbacks, tf.data, Saving)

This lab rebuilds the work of Labs 1–6 with TensorFlow and Keras. The same spiral network from Lab 6 is defined, trained, regularized and saved using the framework, and the Lab 6 regularization experiment is repeated to check that it leads to the same conclusions.

## Files

| File | Description |
|---|---|
| `arti402_Lab7_2240005261.ipynb` | Completed lab notebook with all outputs visible |
| `arti402_figures_Lab7.py` | Helper file that draws the lab diagrams (provided with the course, not modified) |

## Dataset

No external dataset is used. The data is the NNFS spiral dataset, generated in the notebook by `spiral_data`, with exactly the same setup as Lab 6:

- 3 interleaved spiral classes with 2 features per sample.
- Training set: 120 samples, 40 per class (seed 0). This set is small on purpose, to make overfitting likely.
- Validation set: 300 samples (seed 2), used to compare settings.
- Test set: 300 samples (seed 1), kept untouched.

## What was implemented

| Part | Function / Tool | Idea |
|---|---|---|
| Idea 1 | `keras.Sequential` | The Lab 6 network (2 → 128 ReLU → 3 softmax) in a few lines; Keras counts 771 parameters, the same as `n_in * n_out + n_out` |
| Exercise 1 | `build_model` | Builds the network with optional hidden size, L2 and dropout |
| Exercise 2 | `compile` / `fit` / `evaluate` | Adam (0.02), sparse categorical cross-entropy, 100 epochs, batch size 32 |
| Exercise 3 | `kernel_regularizer=regularizers.l2(5e-4)` | L2 regularization as a layer argument |
| Exercise 4 | `keras.callbacks.EarlyStopping` | Stops when validation loss stops improving and restores the best weights |
| Idea 5 | `tf.data.Dataset` + `steps_per_epoch` | 2,000 gradient steps reported as 100 epochs, to cut Keras's per-epoch overhead |
| Idea 6 | `model.save` / `keras.models.load_model` | Saves and reloads the model with identical predictions |
| Q1 | regularization showdown | Baseline vs L2 vs dropout vs early stopping, 2,000 updates each |
| Q2 | decision boundaries | Boundaries of baseline, L2 and dropout, plus the baseline's training curves |

## Results

**Exercise 2 — baseline (100 epochs, batch size 32):** training accuracy 0.817 and validation accuracy 0.593 (gap +0.223). The curves show the same overfitting pattern as Lab 6.

**Exercise 3 — L2 (5e-4):** training accuracy 0.808 and validation accuracy 0.593. The L2 penalty adds 0.0324 to the loss, and the parameter count stays at 771.

**Exercise 4 — early stopping (patience 25):** training was set to 200 epochs but stopped after 160. The best epoch was 135, with validation loss 0.931 and validation accuracy 0.643.

**Saving:** the reloaded model gives identical predictions. The file is about 30 KB because it also stores the optimizer state.

**Regularization showdown (2,000 updates, Adam):**

| Model | Train acc | Val acc | Val loss | Gap |
|---|---|---|---|---|
| Baseline | 0.983 | 0.720 | 2.301 | +0.263 |
| L2 (5e-4) | 0.967 | 0.733 | 1.115 | +0.233 |
| Dropout (0.2) | 0.908 | 0.677 | 1.852 | +0.232 |
| Early stopping | 0.917 | 0.670 | 0.946 | +0.247 |

## Key takeaways

- Keras contains everything built by hand in Labs 1–6: dense layers, activations, losses, optimizers, regularization and batch normalisation.
- Automatic differentiation replaces every hand-written `backward()` method.
- Regularization is just a layer argument (`kernel_regularizer`, `layers.Dropout`).
- The accuracy printed by `fit` is measured during training, with dropout on. Use `evaluate` afterwards for the real number.
- Early stopping with `restore_best_weights=True` gives the lowest validation loss and finishes sooner.
- `tf.data` with `steps_per_epoch` keeps the number of updates the same while cutting the per-epoch bookkeeping.
- The Keras results reach the same conclusion as Lab 6: every regularized model lowers validation loss, and L2 gives the best validation accuracy (0.733).

## How to run

1. Put `arti402_Lab7_2240005261.ipynb` and `arti402_figures_Lab7.py` in the same folder.
2. Install the requirements: `pip install tensorflow numpy matplotlib jupyter`
3. Open the notebook and run all cells from top to bottom (Kernel → Restart & Run All). The full run takes less than a minute.

## Notes

- The notebook saves the trained model as `spiral_model.keras` in the same folder when it runs.
- Results depend on the fixed random seeds used in the notebook (`keras.utils.set_random_seed`).
