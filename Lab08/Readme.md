# ARTI 402 — Deep Learning · Lab 8: PyTorch (Autograd, nn.Module, Training Loop, DataLoader)

This lab rebuilds the Lab 6 network in PyTorch. PyTorch gives the layers and automatic gradients, but not a training loop, so the loop is written by hand again: zero the gradients, forward pass, backward pass, optimizer step.

## Files

| File | Description |
|---|---|
| `arti402_Lab8_2240005261.ipynb` | Completed lab notebook with all outputs visible |
| `arti402_figures.py` | Helper file that draws the lab diagrams (provided with the course, not modified) |

## Dataset

No external dataset is used. The data is the NNFS spiral dataset, generated in the notebook by `spiral_data`, with the same setup as Labs 6 and 7:

- 3 interleaved spiral classes with 2 features per sample.
- Training set: 120 samples, 40 per class (seed 0).
- Validation set: 300 samples (seed 2).

## What was implemented

| Part | Function / Class | Idea |
|---|---|---|
| Exercise 1 | gradient check | Autograd gradient of an `nn.Linear` layer compared with the numerical central difference |
| Exercise 2 | `Net(nn.Module)` | The Lab 6 network: Linear(2, 128), then ReLU, then optional Dropout, then Linear(128, 3), returning logits (no softmax) |
| Exercise 3 | `train_one_epoch` | Manual mini-batch loop: `zero_grad`, forward + `F.cross_entropy`, `backward`, `step` |
| Exercise 4 | `evaluate` | Loss and accuracy with `model.eval()` and `torch.no_grad()` |
| Exercise 5 | `train_one_epoch_loader` | The same loop, with batching and shuffling done by a `DataLoader` |
| Exercise 6 | `run` | Full training run: Adam with `weight_decay` (L2), optional dropout, and a recorded history |
| Checkpoints | written answers | Chain rule and `zero_grad`, step order, `eval()` vs `no_grad()`, why `argmax` needs no softmax |

## Results

**Autograd basics:** for `loss = (w * x - 5)^2` with w = 2 and x = 3, autograd gives `w.grad = 6.0`, which matches the hand calculation. Calling `backward()` three times without clearing gives 6, 12, 18, which shows that gradients accumulate.

**Gradient check:** the largest difference between autograd and the numerical gradient is 5.93e-11.

**Model size:** 771 parameters, the same as the Lab 6 NumPy network and the Lab 7 Keras model.

| Layer | Shape | Parameters |
|---|---|---|
| `fc1.weight` | (128, 2) | 256 |
| `fc1.bias` | (128,) | 128 |
| `fc2.weight` | (3, 128) | 384 |
| `fc2.bias` | (3,) | 3 |

**Training loop (Adam, lr 0.02, batch size 32):**

| Check | Result |
|---|---|
| Manual loop, epoch 1 loss | 1.201 |
| Manual loop, epoch 10 loss | 1.003 |
| `DataLoader` loop, loss after 10 epochs | 0.996 |
| Batches per epoch (120 samples / 32) | 4 |
| Short `run` (100 epochs) | 0.5 s, 21 history rows, training accuracy improved |

**Dropout (`nn.Dropout(0.5)`):** in train mode, about half of the values are zeroed and the rest are doubled. In eval mode, the input passes through unchanged.

## Key takeaways

- PyTorch builds the computational graph during the forward pass, and `loss.backward()` applies the chain rule through it.
- Gradients accumulate in `.grad`, so every training step must start with `optimizer.zero_grad()`.
- `nn.Linear` stores weights as `(out, in)`, which is the transpose of the Lab 5 `Layer_Dense`.
- The model returns logits because `F.cross_entropy` applies softmax internally. Adding a softmax would apply it twice.
- `model.eval()` changes layer behaviour (dropout, batch norm), and `torch.no_grad()` stops gradient tracking.
- In PyTorch, L2 is the optimizer's `weight_decay` (it also applies to biases), and dropout is a layer controlled by `train()` / `eval()`.

## How to run

1. Put `arti402_Lab8_2240005261.ipynb` and `arti402_figures.py` in the same folder.
2. Install the requirements: `pip install torch numpy matplotlib jupyter`
3. Open the notebook and run all cells from top to bottom (Kernel → Restart & Run All). The full run takes a few seconds on CPU.

## Notes

- The training loop is written by hand. There is no `fit` function in PyTorch.
- The labs are cumulative: Lab 5's NumPy loop, Lab 7's Keras `fit`, and Lab 8's PyTorch loop all train the same 771-parameter network.
- Results depend on the fixed random seeds used in the notebook.
- 
