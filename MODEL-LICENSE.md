# Licence of the model files

This repository holds code under the Apache License 2.0 (see [`LICENSE`](LICENSE)).

The **model files** — `plants.tflite`, `labels.txt` and `model.json`, published
at [huggingface.co/brunopaiva15/iris](https://huggingface.co/brunopaiva15/iris)
— are released separately, under the
**[Creative Commons Attribution 4.0 International licence](https://creativecommons.org/licenses/by/4.0/)**
(CC BY 4.0). You may use them commercially, run them, fine-tune them and
redistribute them, including inside closed-source products, as long as you give
credit.

Suggested credit:

> Iris 9 — brunopaiva15, CC BY 4.0 — https://huggingface.co/brunopaiva15/iris

## Where the training images come from, and what is owed to them

The weights were trained on photographs published by naturalists under CC0,
CC BY and CC BY-SA on GBIF, iNaturalist and Wikimedia Commons. Images under
any non-commercial or no-derivatives licence were refused, twice: in the query
and again on each individual file.

No training image is redistributed here, in any form a person can look at.
The position taken when CC BY-SA images were allowed into the training set is
that a trained network is not an adaptation of the photographs — it reproduces
none of them and never redistributes them — and therefore carries no
share-alike obligation. That reading is defensible, widely shared, and has not
been tested in court. Anyone building on these weights should form their own
view.

Per-image attribution for the exact training set is **not available**: the file
that recorded it was lost with the training machine. What remains is the
collection code, which rebuilds an equivalent manifest from scratch. This is
set out in [`docs/data-provenance.md`](docs/data-provenance.md).
