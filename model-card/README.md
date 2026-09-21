---
license: cc-by-4.0
pipeline_tag: image-classification
tags:
  - image-classification
  - plant-identification
  - biodiversity
  - tflite
  - on-device
  - mobile
  - mobilenetv3
metrics:
  - accuracy
model-index:
  - name: Iris 9
    results:
      - task:
          type: image-classification
          name: Plant species classification
        dataset:
          name: Iris held-out test split (2026-09)
          type: custom
        metrics:
          - type: accuracy
            name: Top-1
            value: 0.6876
          - type: accuracy
            name: Top-3
            value: 0.8227
          - type: f1
            name: Macro F1
            value: 0.6702
---

# Iris 9

An offline plant species classifier for phones. One 9 MB TensorFlow Lite file,
1,569 species, about one second per photo on a recent handset, no network call.

Iris is the on-device model of [Auxine](https://vergasta.ch/auxine/en/),
a plant care app. It is published here on its own so that it can be used,
measured and criticised outside that app. Training and evaluation code:
[github.com/brunopaiva15/iris](https://github.com/brunopaiva15/iris).

## The model in one table

Every number below is read from `model.json`, which is produced by the export
itself. Nothing here is typed by hand.

| | |
|---|---|
| Version | 9 |
| Architecture | MobileNetV3-Large, ImageNet-pretrained backbone, softmax head |
| Exposed classes | 1,569 species |
| Input | 320 × 320 × 3, float32, **values in 0–255** |
| Normalisation | inside the graph (`included_in_graph_uint8_0_255`) |
| Output | 1,569 softmax probabilities |
| Weights | float16 TFLite, 9,005,584 bytes |
| SHA-256 | `a919ae6a24f7e7717a36d5213a800a3410f7cee0f694f5014ba8eb07f0a6caa3` |
| Top-1 / Top-3 | 0.6876 / 0.8227 on 30,954 held-out test images |
| Macro F1 | 0.6702 |
| Mean confidence | 0.7299 |

On cultivated plants — the photographs the app actually sees — top-1 is 0.6728
and top-3 0.8101 over 4,649 images, rising to **0.7650** when the indoor
context mask is applied (see *Context masks*).

## Files

| File | What it is |
|---|---|
| `plants.tflite` | the weights |
| `labels.txt` | one internal id per line, in output order (`monstera-deliciosa`) |
| `model.json` | version, input recipe, SHA-256, metrics, threshold curve, `species` map (id → scientific name), `masks` |

`labels.txt` and the `species` map in `model.json` are the only two things you
need to turn an output index into a name. `model.json` is the model's own fact
sheet: read it rather than hardcoding any value from this page.

## Quick start

```bash
pip install huggingface_hub ai-edge-litert pillow numpy
```

```python
import json
import numpy as np
from PIL import Image, ImageOps
from huggingface_hub import hf_hub_download
from ai_edge_litert.interpreter import Interpreter

REPO = "brunopaiva15/iris"
weights = hf_hub_download(REPO, "plants.tflite")
meta = json.load(open(hf_hub_download(REPO, "model.json")))
labels = open(hf_hub_download(REPO, "labels.txt")).read().split()

def prepare(path, size, load_size, source_size):
    """Centre square, area-resize to source_size, bilinear to load_size,
    centre crop to size. The two-step resize is not optional — see below."""
    im = ImageOps.exif_transpose(Image.open(path)).convert("RGB")
    side = min(im.size)
    left, top = (im.width - side) // 2, (im.height - side) // 2
    im = im.crop((left, top, left + side, top + side))
    if im.width > source_size > load_size:
        im = im.resize((source_size, source_size), Image.BOX)
    im = im.resize((load_size, load_size), Image.BILINEAR)
    off = (load_size - size) // 2
    im = im.crop((off, off, off + size, off + size))
    return np.asarray(im, dtype=np.float32)[None]  # 0–255, NOT 0–1

interp = Interpreter(model_path=weights)
interp.allocate_tensors()
inp, out = interp.get_input_details()[0], interp.get_output_details()[0]

x = prepare("photo.jpg", meta["input_size"], meta["load_size"], meta["source_size"])
interp.set_tensor(inp["index"], x)
interp.invoke()
scores = interp.get_tensor(out["index"])[0]

for i in scores.argsort()[::-1][:5]:
    print(f"{meta['species'][labels[i]]:40s} {scores[i]:.4f}")
```

On a photograph of a *Monstera deliciosa* this prints `Monstera deliciosa
0.9119`, the next candidate two orders of magnitude behind.

A fuller script — context masks, the acceptance rule, multiple photos — is in
[`examples/identify.py`](https://github.com/brunopaiva15/iris/blob/main/examples/identify.py).

## Preprocessing, exactly

The graph normalises its own input, so the caller only decodes and frames. But
the framing is part of the model: the training images went through a two-step
reduction, and skipping it costs several points of top-1, measured.

1. honour the EXIF orientation;
2. crop the **largest centred square**;
3. if the square is wider than `source_size` (448), area-average it down to
   448 — this is the state the training images were stored in;
4. bilinear resize to `load_size` (366);
5. centre crop to `input_size` (320);
6. hand over float32 values in **0–255**. Do not divide by 255, do not apply
   ImageNet mean/std. The rescaling layer is in the graph.

Resizing a 4,000 px phone photo straight to 366 px produces aliasing the model
has never seen in training. Step 3 is what avoids it.

## Output, and the two context masks

The output is a softmax over all 1,569 classes. `model.json` also carries a
`masks` object:

```json
"masks": { "indoor": ["monstera-deliciosa", …], "outdoor": ["abies-alba", …] }
```

336 classes indoor, 1,424 outdoor. When you know where the plant lives, keep
only that mask's classes and renormalise by their sum. This is not an
approximation of a smaller model — it is exactly equal to it: dropping columns
from the last layer and re-applying softmax gives `exp(zᵢ) / Σ_kept exp(zⱼ)`,
which is what dividing the kept probabilities by their mass computes.

Measured on cultivated plants:

| | top-1 | accepted at 0.70 | precision when accepted |
|---|---|---|---|
| indoor mask applied | **0.7650** | 65.7 % | 0.9284 |
| no mask | 0.7230 | 67.5 % | 0.9030 |

And on the full test split:

| | classes | top-1 | top-3 |
|---|---|---|---|
| outdoor mask | 1,424 | 0.6898 | 0.8246 |
| no mask | 1,569 | 0.6876 | 0.8227 |

The value of a mask is in what it removes, not in what it keeps: the indoor
mask drops 1,233 classes and buys 4.2 points; the outdoor mask drops 145 and
buys two tenths.

**Do not use a mask as a hard filter.** A class outside the context should be
demoted, never deleted — a model that cannot say the right answer will say a
wrong one with confidence.

## When to trust an answer

A classifier always answers something, including in front of a cat. The app
accepts the model's answer alone only when

```
best score ≥ 0.70  and  best − second ≥ 0.25
```

and treats anything below 0.10 as no answer at all. Otherwise it asks a remote
service. `model.json` carries the full `threshold_curve` — autonomy and
precision for every (threshold, margin) pair — so you can pick the trade-off
your product needs rather than inheriting this one:

| threshold | accepted | precision when accepted |
|---|---|---|
| 0.50 | 74.3 % | 0.8371 |
| 0.60 | 67.2 % | 0.8736 |
| 0.70 | 60.8 % | 0.9051 |
| 0.80 | 53.9 % | 0.9361 |

A threshold does not travel from one version to the next. Re-read it from the
curve of the model you actually ship.

## More than one photo

Averaging the probability vectors of two photos of the same plant was worth
**+13.7 points of top-1** and three photos **+22.4**, measured on Iris 7 with
the app's exact cascade (not re-measured on Iris 9). It costs no extra model
and no extra training. If your interface can ask for a second photo when the
first answer is uncertain, do it before anything else.

## Training data

About 992,000 photographs of roughly 5,300 species, collected from public
biodiversity sources, with a hard licence filter applied twice — once in the
query, once on each individual image:

| Licence | Kept |
|---|---|
| CC0 1.0, Public Domain Mark | yes |
| CC BY 2.0 → 4.0 | yes |
| CC BY-SA | yes |
| CC BY-NC, ND, NC-SA, NC-ND | no |
| unknown, absent, proprietary, scraped | no |

Sources: **GBIF** occurrence media, **iNaturalist** (the only source that
photographs plants as people actually grow them, indoors and in pots),
**Wikimedia Commons**. Each kept image carries its source, source id, author,
licence and checksum in the collection manifest.

Cleaning: readability, EXIF orientation, shortest side ≥ 320 px, aspect ratio
≤ 1:12, reduction to 1024 px then to 448 px, JPEG. Exact duplicates by SHA-256
and near-duplicates by 64-bit DCT perceptual hash (Hamming ≤ 6) within a
species. Splits are 80/10/10 **by observation group**, so a photograph and its
near-duplicates never land on both sides of the split.

### Per-image attribution is not shipped with this model

The collection pipeline writes an `ATTRIBUTIONS.md` and an `attributions.csv`
— author, licence and observation link for every kept image. **Those files
were lost with the training machine and are not published here.** The
provenance that remains is structural rather than per-image: the sources, the
licence filter, and the collection code, which rebuilds an equivalent
manifest from scratch (`collection/build_dataset.py` in the GitHub repo).

This is stated plainly rather than quietly: if you need per-image credit —
for a CC BY compliance review, for instance — it is not available for the
exact set this model saw, and must be regenerated by re-running the
collection.

## Training procedure

A single head is trained wide and exported narrow. The backbone sees every
species collected; the shipped head exposes only the 1,569 the product serves,
because every exposed class competes for the answer with all the others — and
1,444 → 5,259 outputs cost 10.2 points of top-1 on the same weights.

| | |
|---|---|
| Backbone | MobileNetV3-Large, ImageNet weights, top 100 layers unfrozen |
| Head | global average pooling → dropout 0.5 → dense softmax |
| Input | 320 px, batch 64, mixed precision (softmax kept in float32) |
| Fine-tuning | 30 epochs, ~12,400 steps of 64 per epoch, 8 h on one RTX 2070 Super |
| Early stopping | on val_accuracy, patience 4 — it never fired |
| Export | float16 TFLite, then the head re-sliced to the union of the two masks |

Validation accuracy was still climbing at epoch 30 (0.5473, val_loss 2.2515).
This model is not a converged one; it is the one that was shipped.

## Limitations

- **It cannot say "not a plant."** All 1,569 outputs are species, so the mass
  is always spread over species. On 40 out-of-catalogue plants measured on the
  previous versions, 27.5 % came back **above the 0.70 acceptance threshold** —
  confidently wrong, no fallback triggered. A margin rule and a floor are the
  only guards here; there is no "other" class. Treat high confidence on an
  unknown species as a known failure mode, not an edge case.
- **Wild-plant photographs dominate.** The data are citizen-science
  observations, overwhelmingly from Europe and North America, mostly of plants
  in the ground and in flower. Potted indoor plants are the weakest-covered
  domain and the one the app cares most about — hence the indoor mask.
- **Errors cross the botanical family in about 73 % of cases** (measured on an
  earlier version). These are not near-misses between look-alike species; they
  are photographs outside the learned domain.
- **Six species appear under two accepted names** among the exposed classes
  (*Cupressus macrocarpa* / *Hesperocyparis macrocarpa*, *Dracaena* /
  *Sansevieria trifasciata*, *Echinocactus* / *Kroenleinia grusonii*,
  *Coleus* / *Plectranthus scutellarioides*, and two *Citrus* pairs). Each
  splits its own images and each halves its own score.
- **Cultivars are not distinguished.** A variegated *Monstera* is a
  *Monstera deliciosa*.
- **It is not a diagnostic tool.** Species identification is not edibility,
  toxicity, medicinal or legal advice, and a 0.99 score is not a
  determination. Anything that could harm a person or an animal needs a
  human expert.

## Intended use

Suggesting candidate species to a person who then chooses, on-device, offline —
in a plant app, a garden notebook, a museum kiosk, a field survey. The output
is a ranked list with scores, meant to be shown as such.

Out of scope: anything that acts on an identification without a human, and any
foraging, medical, veterinary or regulatory decision.

## Licence and attribution

The weights, `labels.txt` and `model.json` are released under
[CC BY 4.0](https://creativecommons.org/licenses/by/4.0/). Commercial use and
derivatives are allowed, with attribution. The training and evaluation code in
the GitHub repository is Apache-2.0.

Suggested credit: *Iris 9 — brunopaiva15, CC BY 4.0*.

The training images themselves are not redistributed here, in any form a
person can look at. They belong to the thousands of naturalists who published
them under CC0, CC BY and CC BY-SA on GBIF, iNaturalist and Wikimedia Commons.
This model exists because of them. The position taken when CC BY-SA images
were let into the training set is written down: a trained network is not an
adaptation of the photographs — it reproduces none of them, and never
redistributes them. That position has not been tested in court.

## Citation

```bibtex
@software{iris9_2026,
  title  = {Iris 9: an offline plant species classifier},
  author = {brunopaiva15},
  year   = {2026},
  url    = {https://huggingface.co/brunopaiva15/iris},
  note   = {MobileNetV3-Large, 1569 species, TensorFlow Lite}
}
```

## Version history

| Version | What changed |
|---|---|
| **9** | one head, two context masks (indoor + outdoor) in a single file; 1,569 exposed classes; first full uninterrupted fine-tuning pass |
| Indoor | 336 indoor classes cut from an earlier head; the app's previous model |
| 8 | 5,259 classes trained, 1,444 exposed — the version that proved width is paid for at the output, not in training |
| 7 | 320 px input, dropout 0.5, acceptance threshold back to 0.70 |
| 1–6 | MobileNetV3-Small, 78 → 1,445 classes |
