# Training Iris

The whole recipe, and the two measurements that shaped it.

## Train wide, expose narrow

The species the network *learns* and the species it is allowed to *answer with*
are two different decisions. They were the same decision here for eight
versions, because the training script made a class out of every folder it
found.

Both halves are measured:

- feeding the backbone every species that could be collected — 991,000 images
  over 5,259 species instead of 290,000 over 1,457 — made it **+6.8 points**
  better on the species it already knew;
- exposing all 5,259 of them at the output cost **−10.2 points** of top-1 on
  those same species, on the same weights. Every exposed class competes for the
  answer with all the others.

So: collect everything, train once on everything, then cut the last layer down
to what the product actually serves. Re-slicing takes minutes
(`training/reslice.py`), which means the riskiest decision — how many classes
to expose — is no longer chained to the most expensive step.

Iris 9 was trained on roughly 5,300 species and ships 1,569.

## The recipe

```bash
python3 train.py --dataset ../collection/dataset --out ./out \
  --backbone large --input-size 320 --dropout 0.5 --unfreeze 100 \
  --batch 64 --mixed-precision \
  --head-epochs 40 --fine-epochs 30 \
  --feature-cache .cache/features --checkpoint .cache/ckpt --version 9
```

| Option | Why |
|---|---|
| `--backbone large` | MobileNetV3-Large: three times the computation of Small, and it sees the details that separate close species |
| `--input-size 320` | +2 to 3 points over 224, at roughly twice the inference cost — about a second on a recent phone |
| `--dropout 0.5` | dropped mean confidence by 5.7 points without costing accuracy: a better-calibrated model, which matters when a threshold decides whether to answer |
| `--unfreeze 100` | the top 100 backbone layers are fine-tuned; freezing everything below keeps the ImageNet features that a plant set of this size cannot relearn |
| `--batch 64` | at 320 px a single intermediate tensor is 98 MB; 128 runs out of memory on an 8 GB card. This is the card's decision, and it is written down rather than discovered |
| `--mixed-precision` | roughly doubles throughput on RTX-class tensor cores. The softmax stays float32 — in float16 it overflows once logits pass ~11, and those probabilities are what the acceptance threshold reads |

Training runs in two stages: the frozen backbone encodes every image once into
a feature cache, the head trains on those vectors in minutes, and only then is
the top of the backbone unfrozen for fine-tuning. The cache is what makes
trying one recipe change at a time affordable.

## What the last run cost

Thirty fine-tuning epochs, uninterrupted, 12,421 steps of 64 per epoch, 953
seconds per epoch: **eight hours on one RTX 2070 Super**. Validation accuracy
was still rising at the last epoch (0.5473, val_loss 2.2515) and early stopping
— patience 4 on val_accuracy — never fired. The shipped model is not a
converged one.

Two things learned the expensive way:

- **resuming is not extending.** Restoring weights rebuilds Adam from zero, and
  a converged network resents it: eight extra epochs from a checkpoint landed
  2.3 points *below* their starting point, with a clearly worse validation
  loss. Resuming is for a machine that fell over, not for adding epochs to a
  finished run;
- **a run that does not apply the recipe does not measure the recipe.** One
  pass silently used the defaults — 224 px, dropout 0.3, 60 layers unfrozen —
  because three flags were missing from the command line. It plateaued twelve
  points below. Read the command line before spending eight hours on it.

## Hardware, and the bottleneck

On four CPU cores the data pipeline delivers about 630 images/s while the
network eats 93: decoding is not the constraint. On a GPU doing 1,000 images/s
it is, and the fix is to store the image set already reduced to the load size
instead of decoding full-size JPEGs every epoch. The set must also be on an
SSD: a million files read in a different order every epoch is the worst case
for a spinning disk.

Before moving 50 GB to a rented machine, measure it on a few hundred megabytes:
`collection/sample.py` takes a stratified slice — same images, same pipeline,
same recipe, fewer classes — and the seconds-per-step of the *second* epoch
compares two machines to within a few percent. The first epoch pays for graph
compilation.

## Exporting

`train.py` converts to float16 TFLite and writes three files: `plants.tflite`,
`labels.txt`, and `model.json` — the version, the input recipe, the SHA-256 of
the weights, the metrics, the threshold curve, the species names and the
context masks. The conversion always runs in float32: the converter cannot
convert a float16 graph, so the network is rebuilt before evaluation and
export, and the numbers in `model.json` describe the file that ships rather
than something that resembles it.

`reslice.py` does the same from a checkpoint, keeping only a subset of the
output classes:

```bash
python3 reslice.py --weights .cache/ckpt/fine.weights.h5 \
  --labels ./out/labels.txt --dataset ../collection/dataset \
  --out ./iris9 --version 9 \
  --mask indoor=../collection/mask_indoor.txt \
  --mask outdoor=../collection/mask_outdoor.txt
```

With two masks it exports their union as one file and records both lists in
`model.json`. The application renormalises over the mask of the place the
plant lives, which is exactly equal to a model trained on those classes alone
— and four points better than no mask at all on photographs of cultivated
plants.

## Evaluating

| Script | What it answers |
|---|---|
| `compare_models.py` | is the new model better than the shipped one, on the same images, at the app's own thresholds |
| `confusions.py` | which species are mistaken for which, and whether the error crosses the genus or the family |
| `curve.py` | autonomy and precision at every (threshold, margin) pair — the curve that goes into `model.json` |
| `indoor.py` | what the model does on cultivated plants specifically |
| `off_topic.py` | what it claims when shown a plant it does not know, or something that is not a plant |
| `multi_photo.py` | what a second and a third photograph of the same plant are worth |
| `genus.py` | how much is gained by answering at genus level instead |
| `benchmark.py` | inference time per photograph |

A version is not better because the top-1 went up. It is better when it is
better on *the same test images*, at the threshold the product ships, on the
species the product exposes — which is what `compare_models.py` measures and
why it exists.
