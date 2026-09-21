# Inference examples

```bash
pip install -r requirements.txt
python3 identify.py photo.jpg
python3 identify.py leaf.jpg whole-plant.jpg --context indoor
python3 identify.py photo.jpg --model /path/to/local/model   # instead of the Hub
```

`identify.py` is deliberately a single readable file. It shows the four things
that separate a working integration from a disappointing one.

## 1. The preprocessing is part of the model

Centre square, area-average down to 448, bilinear to 366, centre crop to 320.
The training images were stored at 448 px and reduced in two steps; resizing a
4,000 px phone photograph straight to 366 produces aliasing the model never saw
in training, and it costs several points of top-1.

## 2. The input is 0–255, not 0–1

The rescaling layer is inside the graph (`"preprocessing":
"included_in_graph_uint8_0_255"` in `model.json`). Dividing by 255 out of habit
does not raise an error — it just makes the model much worse, quietly.

## 3. The context mask

`model.json` carries two lists of species: 336 that live indoors, 1,424 that
live outdoors. Keeping one list and renormalising by its mass is exactly
equivalent to a model trained on those classes alone, and on photographs of
cultivated plants it is worth 4.2 points of top-1.

Demote the classes outside the context; never delete them. A model that cannot
say the right answer says a wrong one with confidence.

## 4. Knowing when not to answer

A softmax over 1,569 species always answers something, including in front of a
cat. The rule here — best ≥ 0.70 and a margin of 0.25 over the runner-up,
nothing below 0.10 — is the app's, tuned to send uncertain photographs to a
remote service. `model.json` carries the whole threshold curve so you can pick
your own trade-off. A threshold does not travel from one model to the next.

## And if you can ask for a second photograph

Averaging the probability vectors of two photographs of the same plant was
worth +13.7 points of top-1 on an earlier version, and three were worth +22.4.
No extra model, no extra training, no measurable computation. `identify.py`
takes several files for exactly this reason.
