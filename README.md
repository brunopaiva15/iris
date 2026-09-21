# Iris

An offline plant species classifier for phones, and everything needed to
rebuild it: the image collection pipeline, the training recipe, the evaluation
tools, and the model card.

The shipped model is **Iris 9** — MobileNetV3-Large, 1,569 species, 9 MB of
TensorFlow Lite, about a second per photograph on a recent handset, no network
call. It is published on the Hugging Face Hub:

> **[huggingface.co/brunopaiva15/iris](https://huggingface.co/brunopaiva15/iris)**

Iris is the on-device model of [Auxine](https://vergasta.ch/auxine/en/),
a plant care app. It is published on its own so it can be used, measured and
criticised outside that app.

| | |
|---|---|
| Exposed classes | 1,569 species |
| Top-1 / Top-3 | 0.6876 / 0.8227 on 30,954 held-out images |
| On cultivated plants | 0.6728, and **0.7650** with the indoor context mask |
| Input | 320 × 320 float32 in 0–255, normalisation inside the graph |
| Weights | float16 TFLite, 9,005,584 bytes |
| Trained on | ~992,000 CC0 / CC BY / CC BY-SA photographs, ~5,300 species taught |

The full fact sheet, the preprocessing recipe, the masks, the thresholds and
the limitations are in the model card: [`model-card/README.md`](model-card/README.md).
That file is what gets published as the Hub page — there is no second copy to
keep in sync.

## Identify a plant

```bash
pip install -r examples/requirements.txt
python3 examples/identify.py photo.jpg --context indoor
```

```
Iris 9 — 1569 species, 1 photograph(s), context: indoor

1. Monstera deliciosa                       0.9215
2. Syngonium angustatum                     0.0428
3. Selenicereus undatus                     0.0101
4. Zamia furfuracea                         0.0074
5. Monstera adansonii                       0.0032

accepted — good enough to show as the answer
```

The model downloads itself from the Hub on first use and everything after that
runs offline. See [`examples/`](examples/) for what the script demonstrates and
for the three details that are easy to get wrong: the two-step resize, the
0–255 input range, and the context mask.

## What is in here

| Directory | What it holds |
|---|---|
| [`model-card/`](model-card/) | the model card, published verbatim as the Hub page |
| [`examples/`](examples/) | inference: preprocessing, context masks, acceptance rule, several photographs |
| [`training/`](training/) | training, evaluation, head re-slicing, benchmarks, confusion analysis |
| [`collection/`](collection/) | building the image set: sources, licence filter, cleaning, de-duplication, splits |
| [`scripts/`](scripts/) | publishing a new version to the Hub |
| [`docs/`](docs/) | [training](docs/training.md), [data provenance](docs/data-provenance.md), [publishing](docs/publishing-to-the-hub.md) |

## Rebuild it

Nothing here is a black box, but the two halves cost very different amounts.

```bash
# 1. collect the images — hours, and roughly 50 GB
cd collection && pip install -r requirements.txt
python3 build_dataset.py --plants plants.csv --out dataset

# 2. train — one GPU, hours
cd ../training && pip install -r requirements-gpu.txt
python3 train.py --dataset ../collection/dataset --out ./out \
  --backbone large --input-size 320 --dropout 0.5 --unfreeze 100 \
  --batch 64 --mixed-precision --head-epochs 40 --fine-epochs 30

# 3. expose a narrower set of classes than the network learned — minutes
python3 reslice.py --weights .cache/ckpt/fine.weights.h5 \
  --labels ./out/labels.txt --dataset ../collection/dataset --out ./iris9 --version 9 \
  --mask indoor=../collection/mask_indoor.txt \
  --mask outdoor=../collection/mask_outdoor.txt

# 4. publish
python3 ../scripts/push_to_hub.py --model ./iris9 --tag v9
```

The image set is not in Git — it is 50 GB — and it does not rebuild
identically: observations get withdrawn and licences change. See
[`docs/data-provenance.md`](docs/data-provenance.md) for what that means for
attribution.

## The one idea worth stealing

Train wide, expose narrow. The classes the network learns and the classes it
answers with are two separate decisions, and they were confused here for eight
versions. Feeding the backbone every species that could be collected made it
better at the ones people actually photograph; exposing all of them at the
output cost 10.2 points of top-1 on the same weights. Re-slicing the last layer
takes minutes and can be redone as often as you like, so the riskiest decision
is no longer attached to the most expensive step.

## Licence

Code: Apache-2.0 ([`LICENSE`](LICENSE)).
Model files: CC BY 4.0 ([`MODEL-LICENSE.md`](MODEL-LICENSE.md)).

The training photographs belong to the naturalists who published them under
CC0, CC BY and CC BY-SA on GBIF, iNaturalist and Wikimedia Commons. None of
them is redistributed here. Per-image attribution for the exact training set
was lost with the training machine — stated plainly, with the consequences, in
[`docs/data-provenance.md`](docs/data-provenance.md).
