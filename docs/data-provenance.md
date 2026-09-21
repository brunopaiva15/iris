# Where the training images come from

## The rule

Only images a commercial product may use, with attribution at most:

| Licence | Kept |
|---|---|
| CC0 1.0, Public Domain Mark | yes |
| CC BY 2.0 → 4.0 | yes, attribution owed |
| CC BY-SA | yes, attribution owed |
| CC BY-NC, BY-ND, BY-NC-SA, BY-NC-ND | no |
| unknown, absent, proprietary, scraped from a search engine | no |

The filter runs twice: once in the query, where the source supports it, and
again on **each individual file**, because sources routinely serve a media item
under a licence different from the record that points at it
(`collection/plant_dataset/licenses.py`). An image whose licence cannot be read
is refused; an unreadable licence is not a permissive one.

CC BY-SA was let in on 6 September 2026, on the reading that a trained network
is not an adaptation of the photographs: it reproduces none of them and never
redistributes them. That reading is defensible and untested in court. It is
also the reason the model files carry a CC BY 4.0 licence rather than a
share-alike one.

## The sources

| Source | Why it is in |
|---|---|
| **GBIF** | occurrence media at scale, every family, worldwide; about 18 % of the media carry a usable licence |
| **iNaturalist** | the only source that photographs plants **as people grow them** — in pots, indoors, on balconies — and it marks them `captive`. Roughly 80 % of the media carry a usable licence |
| **Wikimedia Commons** | cultivated and ornamental plants, 97 % usable licences, no API key |

Sources that were measured and refused — Kew, Smithsonian Gardens, Trefle,
Openverse, Pexels, Roboflow, Kaggle plant sets — failed on one of three counts:
no verified species identification, licences declared by an uploader rather
than by the photographer, or a licence field that is free text rather than a
licence. PlantNet-300K was measured too: 99.6 % CC BY-SA, but overwhelmingly
close-ups of organs rather than whole plants.

## What is recorded, when the collection runs

Every kept image gets a line in `dataset/manifest.jsonl`: species, internal id,
source, source id, observation id, original page, image url, **author**,
**licence**, licence url, download time, SHA-256, dimensions, perceptual hash,
status. The manifest is the only truth — the folders rebuild from it.

From it, `write_attributions` produces `dataset/ATTRIBUTIONS.md` and
`dataset/attributions.csv`: author, licence and observation link for every
single image.

## What is missing, and why it matters

**Those attribution files do not exist for the model published here.** The
training set lived on a rented machine that has since been returned, and the
files were not copied off it. About 992,000 images, and the per-image record of
who took them is gone.

This is a real gap, not a formality:

- CC BY and CC BY-SA both require attribution. The obligation attaches to the
  images, and the argument that the weights carry no obligation is an argument
  about the weights, not about the collection that was made and kept;
- the set does not rebuild identically. Observations get withdrawn, licences
  get changed, accounts get deleted. Re-running the collection produces an
  equivalent set, not the same one, so the record of what this model actually
  saw cannot be reconstructed after the fact.

What is published instead is structural provenance: the sources, the filter,
the cleaning rules, and the code that rebuilds an equivalent manifest —
`collection/build_dataset.py`. Anyone needing per-image credit for a compliance
review has to re-run the collection and work from its manifest.

The practical lesson, written here so the next version does not repeat it:
**`attributions.csv` leaves the training machine before the training machine is
returned.** Compressed it is a few tens of megabytes. Nothing else about a
training run is unrecoverable.

## Cleaning and splitting, in one paragraph

Readability, EXIF orientation applied, shortest side ≥ 320 px, aspect ratio
≤ 1:12, stored reduced to a fixed long side (448 px for the set Iris 9 saw —
that is the `source_size` in `model.json`; the current collector stores 384,
`MAX_SIDE` in `collection/plant_dataset/images.py`), JPEG. Exact duplicates by
SHA-256 and near-duplicates by 64-bit DCT perceptual hash (Hamming distance
≤ 6) within a species; the same visual under two species goes to review rather
than to training. Splits are 80/10/10 **by observation group**, deterministic
by hash, so a photograph and its near-duplicates never end up on both sides —
without that, the test score measures memory rather than recognition.
