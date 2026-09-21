# Training the species classifier

Trains the classifier that ships inside the app and exports it to TensorFlow
Lite. The overview is in [`../docs/training.md`](../docs/training.md); the
image set is built by [`../collection`](../collection/README.md).

## Install

```bash
cd training
python3 -m pip install -r requirements.txt   # tensorflow-cpu, numpy, Pillow
```

## Train

```bash
python3 train.py --dataset ../collection/dataset --out ./out
```

On a machine without a GPU a full pass takes hours and can be interrupted. The
resumable form:

```bash
python3 train.py --dataset ../collection/dataset --out ./out \
  --backbone large --head-epochs 40 --fine-epochs 12 \
  --feature-cache .cache/features --checkpoint .cache/ckpt --version 9
```

Running the same line again resumes from the last checkpoint. With
`--fine-epochs 0`, training is skipped: the model is evaluated and exported
from the checkpoint as it stands.

**Resuming is not extending**: only the weights are restored, Adam starts from
zero, and adding epochs to an already converged network costs it points. The
checkpoint is also rewritten at every epoch, so a resume erases what it
started from.

Outputs:

| File | Contents |
|---|---|
| `plants.tflite` | the weights, float16 |
| `labels.txt` | one internal id per line, in output order |
| `model.json` | version, input size, SHA-256, metrics, threshold curve, species names, masks |

## Train on a GPU (Windows + RTX, through WSL2)

A full pass takes about ten hours on four CPU cores and around one hour on an
RTX 2070 Super. Everything else depends on it: it is the first thing to set up.

**TensorFlow has had no native GPU support on Windows since 2.10.** The route
that works is WSL2, where the Windows driver is seen by Linux without
installing a driver on the WSL side:

```bash
# On Windows, in PowerShell
wsl --install -d Ubuntu

# Then, in Ubuntu
sudo apt update && sudo apt install -y python3-pip python3-venv
python3 -m venv ~/venv && source ~/venv/bin/activate
pip install -r requirements-gpu.txt         # tensorflow[and-cuda]: CUDA and cuDNN included
python3 -c "import tensorflow as tf; print(tf.config.list_physical_devices('GPU'))"
```

That last line must print a card. `train.py` says so too at startup: with no
card visible it announces the fact instead of silently running ten times
slower.

```bash
python3 train.py --dataset ../collection/dataset --out ./out \
  --backbone large --batch 64 --mixed-precision \
  --head-epochs 40 --fine-epochs 30 \
  --feature-cache .cache/features --checkpoint .cache/ckpt --version 9
```

| Option | Why, on a GPU |
|---|---|
| `--mixed-precision` | float16 computation: the tensor cores of RTX 20xx and later roughly double throughput, and the freed memory allows bigger batches. Useless, even slow, on a CPU. |
| `--batch 64` | at 320 px a single intermediate tensor is 98 MB and an 8 GB card does not follow at 128. Raise it as long as memory allows. |
| `--ram-budget` | no effect on the full set: 232,000 images at 256 px are 45 GB, beyond any reasonable amount of RAM, and preloading is all or nothing. Leave it alone. |

The TFLite export always runs in float32: the converter cannot convert a
float16 graph, so the network is rebuilt before evaluation and export. The
numbers in `model.json` are those of the shipped file, not those of a model
resembling it.

**The bottleneck moves.** Measured here on four cores, the data pipeline
delivers 630 images/s from a cold cache while the network eats 93: JPEG
decoding is eight times too fast to get in the way. On a card doing 1,000
images/s that ratio inverts and the CPU becomes the one keeping the GPU
waiting. An 8-core i7 holds up easily; below that, watch it. **And the set
must be on an SSD**: 290,000 files read in a different order at every epoch
are the worst case for a spinning disk.

**Measure the machine before moving the set onto it.** Copying tens of
gigabytes only to discover a card brings nothing is a day lost.
[`../collection/sample.py`](../collection/sample.py) takes a few hundred
megabytes — same images, same pipeline, same recipe, only fewer classes — and
that is enough to compare two machines: at 320 px the backbone costs about
0.45 GFLOP per image against 0.01 for the head, so a throughput read on 120
classes describes the machine to within a few percent. Then run
`train.py --steps-per-epoch 60 --fine-epochs 2 --ram-budget 0` on the sample
and read the s/step of the **second** epoch: the first one pays for graph
compilation.

**The image set is not in Git** (about 50 GB). It rebuilds with
[`../collection`](../collection/README.md), in parallel shards: count three
hours on four cores, less on eight. Copy `../collection/cache/*.json` into
`dataset/` first — those are hours of name resolution already done.

## Options

| Option | Default | Meaning |
|---|---|---|
| `--batch` | 32 | |
| `--backbone` | `small` | `small` (MobileNetV3-Small) or `large` (three times the computation, better on close species) |
| `--input-size` | 224 | side of the network input; the loading size follows at the same crop margin. 320 doubles the computation on the phone without changing the size of the `.tflite` — the weights do not depend on the resolution |
| `--head-epochs` | 4 | epochs with the network frozen |
| `--fine-epochs` | 12 | fine-tuning epochs, early stopping on validation |
| `--min-train` | 25 | a class below this threshold is left out of the model |
| `--min-val` | 3 | a class with no validation images cannot be measured |
| `--unfreeze` | 60 | layers unfrozen at the top of the network |
| `--dropout` | 0.3 | |
| `--version` | `1` | version written into `model.json` |
| `--checkpoint DIR` | | weights saved every 200 batches and at every epoch; rerunning with the same folder resumes there |
| `--feature-cache DIR` | | enables the frozen-network vectors for the head phase (see below) |
| `--mixed-precision` | off | float16 computation; doubles throughput on a card with tensor cores, useless on a CPU |
| `--steps-per-epoch N` | | batches per epoch: short epochs, hence frequent checkpoints |
| `--ram-budget` | 5 | GB of preloading at most; beyond that, images are read from the files |

## Shipping a model: reslicing, and the context masks

`reslice.py` re-exports a trained model keeping only some of its classes. It
is not retraining: the columns of the last layer that are not kept are
removed, and the shipped file shrinks accordingly — the head is
`960 x classes x 2 bytes`.

```bash
python3 reslice.py --weights .cache/ckpt/fine.weights.h5 \
  --labels ./out/labels.txt --keep ../collection/mask_indoor.txt \
  --dataset ../collection/dataset --out ./indoor --version Indoor
```

**A union model** keeps the classes of several places and records which belong
to which. `--keep` becomes unnecessary: the union of the masks makes the list.

```bash
python3 reslice.py --weights … --labels … --dataset … --out … --version 9 \
  --mask indoor=../collection/mask_indoor.txt \
  --mask outdoor=../collection/mask_outdoor.txt
```

`model.json` then carries a `masks` object, and the application renormalises
its outputs over the classes of the place at inference time:
`exp(zi) / sum_kept exp(zj)`, that is **exactly what this model resliced on
that mask would return**. Two files would cost two copies of the same backbone
for two last layers.

A model without a `masks` object behaves as before: no mask, and the context
has no effect.

## What the recipe does

1. **Loading into memory**: every image is decoded once, as uint8 at the load
   size — as long as the set fits in `--ram-budget`. Beyond that, the images
   are re-read at every epoch, and that is fine: measured on four cores, the
   data pipeline delivers 630 images/s cold while the network eats 70. The
   decoder is not the bottleneck at this size, the network is. Preloading only
   serves small sets, where it saves a few minutes.
2. **Augmentation**: random crop, horizontal mirror, slight variation of light
   and saturation, and resolution jitter. No strong rotation: in a photograph,
   a pot stands upright.
3. **Transfer**: an ImageNet-pretrained MobileNetV3, head replaced, trained
   alone first, then the top layers unfrozen at a low learning rate. With
   `--feature-cache`, the head phase does not push the images through the
   network at every epoch: the network is frozen, its outputs do not change,
   so they are computed once (one vector of 960 numbers per image) and the
   head trains on them in minutes. That replaces four thirty-minute epochs
   with one twenty-minute pass — and the head can run to convergence, which
   gives fine-tuning a better start. The vectors are those of the centred
   square, without augmentation; fine-tuning keeps all of its own.
4. **Imbalance**: per-class weights inversely proportional to the number of
   images. Without it the model learns to answer the most frequent species.
5. **Evaluation** on the test set, never seen: top-1, top-3, macro-F1, and the
   **threshold / fallback curve** used to tune the acceptance rule on the
   application side.

The splits come from `splits.csv`: the photographs of one observation are all
on the same side, or the measured accuracy would be a lie.

## What the model gets wrong

```bash
python3 confusions.py --dataset ../collection/dataset --model ./out
```

Separates the errors that stay within the genus — expected, two close species,
and the screen offers five candidates — from those that leave it, which are
real collection defects. Returns the genus pairs responsible and the species
missed most often, with what is answered in their place.

## What the model returns on houseplants

```bash
python3 indoor.py --coverage          # without TensorFlow or the image set
python3 indoor.py --dataset ../collection/dataset --model ./out
```

The published top-1 is an average over every exposed class — their number is
in `model.json` — and most of them are wild. The application serves the names
of `phase1_species.txt` first. The tool returns two things that do not replace
each other: the **coverage** — how many of these plants the model can so much
as name, an absent class being a certain failure and invisible in any accuracy
measurement — then the **top-1 on the test images of those species**, with the
whole catalogue and then masked down to the indoor plants. The gap between the
two is what the breadth of the catalogue costs the person photographing their
living room.

## Choosing the fallback thresholds

`model.json` holds, for every (threshold, margin) pair, the rate of accepted
answers and the precision on those answers. The chosen pair is carried into
the application's acceptance rule.

A rule of "precision above 97 %" is reachable: at threshold 0.95 the model
returns 98.2 % precision. But it then accepts only 32.6 % of the answers — two
photographs out of three would leave for a remote service, at the expense of
the monthly quota. The shipped threshold is 0.70 with a margin of 0.25:
**60.8 % autonomy for 90.5 % precision** on the current version.

Those four numbers are read from the `threshold_curve` of the shipped
`model.json` rather than from here — it is what counts, and it changes at
every version.

So it is a deliberate trade, seven points of correctness against twenty-two of
autonomy, and not the application of the rule above. The real rule bears on
versions rather than on thresholds: **at equal autonomy, take the more correct
version.**
