# ARTI 402 — Deep Learning · Lab 9: Autoencoders (Bottleneck, Representation Learning, Denoising)

This lab builds the first network in the course whose goal is not classification. An autoencoder squeezes each image through a small bottleneck and rebuilds it, learning a compact representation without using any labels. The PyTorch training loop from Lab 8 is reused, with MSE as the loss and the input itself as the target.

## Files

| File | Description |
|---|---|
| `arti402_Lab9_2240005261.ipynb` | Completed lab notebook with all outputs visible |
| `arti402_figures.py` | Helper file that draws the lab diagrams (provided with the course, not modified) |

## Dataset

The scikit-learn `load_digits` dataset, bundled with the library (nothing is downloaded):

- 1,797 hand-written digits (0–9), each 8 × 8 = 64 pixels, flattened into a row.
- Pixel values are scaled from 0–16 to 0–1.
- Split 75 / 25 with stratification (`random_state=0`): 1,347 training images and 450 test images.
- The labels are not used to train the autoencoders. They are used only to colour the plots and to test the learned codes.

## What was implemented

| Part | Function / Class | Idea |
|---|---|---|
| Exercise 1 | `Autoencoder` | Encoder: Linear(64, 32), ReLU, Linear(32, latent). Decoder: Linear(latent, 32), ReLU, Linear(32, 64), Sigmoid |
| Exercise 2 | `train_autoencoder` | The Lab 8 loop with `F.mse_loss(model(xb), xb)`, so the target is the input |
| Exercise 3 | bottleneck sweep | One autoencoder for each code size (2, 4, 8, 16, 32), with train and test reconstruction error |
| Exercise 4 | 2-D latent space | The encoder alone maps test images to 2 numbers, plotted and coloured by digit |
| Exercise 5 | representation test | Logistic regression trained on the codes and compared with the raw 64 pixels |
| Exercise 6 | `train_denoising` | Noisy input (`noise = 0.5`, clamped to 0–1), with the loss measured against the clean image |
| Checkpoints | written answers | Why the bottleneck matters, why it is unsupervised, how to use the encoder with few labels, which digits overlap |

## Results

**Basic autoencoder (8-number code, 60 epochs, Adam lr 1e-3, batch size 64):** 4,744 parameters. Training loss fell from 0.1782 to 0.0216, and the test reconstruction error is 0.0219. The rebuilt digits are blurrier than the originals, but all of them are still readable.

**Bottleneck sweep (60 epochs each):**

| Code size | Compression | Train MSE | Test MSE |
|---|---|---|---|
| 2 | 32× | 0.0467 | 0.0479 |
| 4 | 16× | 0.0346 | 0.0368 |
| 8 | 8× | 0.0213 | 0.0219 |
| 16 | 4× | 0.0193 | 0.0203 |
| 32 | 2× | 0.0179 | 0.0190 |
| Predicting the average image | - | - | 0.0740 |

**How much class information is in the codes (logistic regression, test accuracy):**

| Input to the classifier | Test accuracy |
|---|---|
| 2-number code | 0.622 |
| 8-number code | 0.924 |
| All 64 pixels | 0.969 |

**Denoising (8-number code, test images with noise 0.5):**

| Model | Clean input MSE | Noisy input MSE |
|---|---|---|
| Plain autoencoder | 0.0219 | 0.0534 |
| Denoising autoencoder | 0.0315 | 0.0436 |
| Do nothing (noisy image as output) | - | 0.1157 |

## Key takeaways

- An autoencoder is trained with no labels. The target is the input itself, and the loss is `MSE(output, input)`.
- The bottleneck is what makes it useful. With a code as wide as the input, the network could just copy the pixels and learn nothing.
- Going from 2 to 8 numbers improves the reconstruction a lot, and going from 8 to 32 improves it very little.
- Train and test errors stay almost equal, so there is no overfitting here (1,347 images, under 5,000 parameters).
- The encoder is the useful part. Eight numbers learned without labels reach 0.924 accuracy, close to the 0.969 from all 64 pixels.
- The denoising autoencoder handles noisy input better but is slightly worse on clean input. Like every regularizer, it is a trade-off.
- The training loop is the same as Lab 8: `zero_grad`, forward, `backward`, `step`. Only the loss and the target changed.

## How to run

1. Put `arti402_Lab9_2240005261.ipynb` and `arti402_figures.py` in the same folder.
2. Install the requirements: `pip install torch scikit-learn numpy matplotlib jupyter`
3. Open the notebook and run all cells from top to bottom (Kernel → Restart & Run All). The full run takes about 1–2 minutes on CPU.

## Notes

- The labels (`y_train`, `y_test`) are never used to train the autoencoders. They are used only for plots and for the logistic regression test.
- The comparison with PCA in the notebook is given by the lab and was not re-run.
- Results depend on the fixed random seeds used in the notebook.
