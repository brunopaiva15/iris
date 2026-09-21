# Building the image set

The Python collector that builds the training set for the on-device model. It
never runs on a phone: it is a development tool, run on a workstation, whose
result (`dataset/`) stays out of Git.

The overview — where the images come from, under what licences, and what that
means for attribution — is in
[`../docs/data-provenance.md`](../docs/data-provenance.md). This file only says
how to run the tool.

## Install

```bash
cd collection
python3 -m pip install -r requirements.txt   # requests, Pillow, numpy, pytest
python3 -m pytest -q                          # no network
```

## Choosing the species of the next version

```bash
python3 candidates.py --how-many 3543 --out candidates-v8.txt
python3 availability.py --species-file candidates-v8.txt --csv availability.csv
```

A random draw from the extended catalogue returns only 40 % usable classes, and
grows the population where the model is already weakest — trees and wild
plants. So the candidates must be ranked by what people **grow**.

**GBIF cannot say it**: its `degreeOfEstablishment=cultivated` field counts 288
occurrences out of 76 million. iNaturalist can — its "captive" flag is set by
the observers, and the API returns 58,000 cultivated species sorted by
observations. Selected that way, **83 to 86 %** of the candidates reach the 25
images of `--min-train`, and the rate does not fall between the 1,543rd and the
3,543rd rank.

## The classes that are the same plant

```bash
python3 duplicates.py --labels ../training/out/labels.txt
```

*Sansevieria trifasciata* and *Dracaena trifasciata* are one plant — the snake
plant changed genus in 2017 — and both names are in the catalogue. The model
therefore has two classes for it: the images are shared, each trains on half of
what it should, and the resulting confusion is **unwinnable** since there is
nothing to settle. Yet it shows in `confusions.py` as a defect of the model.

The tool resolves every catalogue name to its accepted taxon at GBIF and groups
those landing on the same key. No heuristic: looking for shared epithets within
one family returns 34 groups for 4 real duplicates — *Populus alba* and *Salix
alba* are not the same plant. The keys are cached, a resume asks for nothing
again.

## Files

| File | Role |
|---|---|
| `plants.csv` | The reference list: one row per plant, with its canonical name, internal identifier, common names, GBIF key and Wikidata identifier. Versioned. |
| `enrich_plants.py` | Fills in `gbif_key` and `wikidata_id` (network). |
| `build_dataset.py` | Collects the images, checks, deduplicates, splits, attributes. |
| `merge_shards.py` | Puts back together collections run in parallel over disjoint shards of the catalogue. |
| `availability.py` | How many images a species has, before deciding to collect it. Sorts the candidates into solid, thin, never photographed and unresolved names. |
| `candidates.py` | The species to add, ranked by what people grow (iNaturalist `captive`). |
| `commons_gain.py` | Measures what Wikimedia Commons would add, species by species, without downloading an image. |
| `plantnet300k.py` | Measures what PlantNet-300K would bring, from its metadata alone (66 MB): species overlap, licences, framing. |
| `cultivars.py` | The known cultivars of each species, from Wikidata — a second axis, not extra classes. |
| `duplicates.py` | The classes that are one plant under two names. |
| `failed_species.py` | Reads the collection logs back and returns the species to retry. |
| `sample.py` | A few hundred megabytes of the set, to measure a machine before moving 50 GB onto it. |
| `plant_dataset/` | The package: `taxonomy` (names), `licenses`, `manifest`, `images`, `dedup`, `splits`, `fetchers/`. |
| `tests/` | Unit tests, with real GBIF answers recorded in `tests/fixtures/`. |
| `dataset/` | Output. **Ignored by Git.** |

## The smoke test (5 plants, 100 images)

This is the test to redo after any change to the pipeline. It takes under a
minute and downloads about 30 MB.

```bash
cd collection
rm -rf dataset
python3 build_dataset.py \
  --plants plants.csv --out dataset \
  --only "Monstera deliciosa,Epipremnum aureum,Ficus elastica,Chlorophytum comosum,Spathiphyllum wallisii" \
  --target-per-species 20 --max-candidates 200
```

Expected result:

```
100 images kept out of 100; exact duplicates 0, near 0; 0 in review; 100 attributions
licences: {'CC BY 4.0': 30, 'CC0 1.0': 70}
```

Then check by hand:

- `dataset/stats.json`: counts per species, licence, source;
- `dataset/ATTRIBUTIONS.md`: every image has an author, an accepted licence and
  a link to the observation;
- `dataset/manifest.jsonl`: one line per image, with `checksum`, `phash`,
  `status`, `reason`;
- `dataset/splits.csv`: `train` / `val` / `test`, with the `group` column (one
  observation = one group);
- open five or six images at random: they really are plants, of the right
  species.

## Options of `build_dataset.py`

| Option | Default | Meaning |
|---|---|---|
| `--plants` | `plants.csv` | reference list |
| `--out` | `dataset` | output folder |
| `--target-per-species N` | 200 | kept images targeted per species |
| `--only "A,B"` | | process only these scientific names |
| `--limit-species N` | | process only the first N plants of the CSV |
| `--max-candidates N` | 1500 | GBIF occurrences walked through at most, per licence |
| `--allow-sa` | off | also accept CC BY-SA (see licences below) |
| `--captive-file F` | | species (one per line) for which to reserve a share of cultivated plants (iNaturalist `captive=true`) |
| `--captive-share` | 0.5 | share of the target reserved for cultivated plants |
| `--captive-place ID` | | iNaturalist place (97391 = Europe): the cultivated plants of that region are collected **first**, on top of the target |
| `--place-share` | 0.25 | share of the target added as cultivated plants from the region |
| `--skip-fetch` | | download nothing: deduplicate, split, count what is already there |
| `--repair-splits` / `--no-repair-splits` | on | give a validation group, then a test one, to the species that have none; `--no-repair-splits` returns the earlier split, to reproduce a previous model |
| `--workers N` | 6 | parallel downloads |
| `--gbif-pause` / `--inat-pause` | 0.25 / 1 | request pace, in seconds; raise them when several collections run |
| `--wikimedia` | off | complement with Wikimedia Commons (see below) |
| `--commons-pause` | 1 | Commons pace; below one second the API answers 429 |

The tool is re-runnable: what already appears in `manifest.jsonl` is not
downloaded again, and the source identifiers (`gbif`, `<occurrence key>#<n>`)
prevent any duplicate at the origin.

## Collecting the whole catalogue, in parallel

A single-run collection occupies one core and leaves the network waiting: 1,558
species take nearly seven hours. Cut into disjoint shards, it fits on the
machine's four cores and drops under three hours. The shards do not overlap, so
merging is a move of folders.

```bash
python3 - <<'PY'
from pathlib import Path
names = [l.strip() for l in Path('all_species.txt').read_text().splitlines() if l.strip()]
indoor = {l.strip() for l in Path('phase1_species.txt').read_text().splitlines() if l.strip()}
# Indoor plants cost twice as much (the "potted" passes): they go first so that
# the three shards take the same time.
order = [n for n in names if n in indoor] + [n for n in names if n not in indoor]
for i in range(3):
    Path(f'shard{i}.txt').write_text('\n'.join(order[i::3]) + '\n')
PY

for i in 0 1 2; do
  mkdir -p shard$i && cp cache/species.json cache/species_inat.json shard$i/
  nohup python3 build_dataset.py --plants plants.csv --out shard$i --only-file shard$i.txt \
    --target-per-species 200 --allow-sa \
    --captive-file phase1_species.txt --captive-share 0.5 \
    --captive-place 97391 --place-share 0.25 \
    --workers 8 --gbif-pause 0.6 --inat-pause 2.0 > shard$i.log 2>&1 &
done
wait

python3 merge_shards.py --out dataset shard0 shard1 shard2
python3 build_dataset.py --out dataset --plants plants.csv --skip-fetch
```

The last line is not optional: cross-species deduplication, splitting and
statistics bear on the whole set, and no shard can do them alone.

A second pass over `phase1_species.txt` at `--target-per-species 300` then adds
the potted-plant photographs on top of the target: those are the ones that
describe the application's real use.

Shards fail. `failed_species.py` reads the logs back and writes the retry
lists, one per shard, so that a species comes back into **its own** shard.

## Wikimedia Commons, as a complement

GBIF and iNaturalist describe field observations. Commons is a media library:
people photograph their monstera in the living room, a ficus at a florist. That
is the distribution the model lacks — the yucca taken for maize and the
unreadable ginseng ficus come from that gap.

Two figures measured before writing the connector, on our own species: **97 %
of the files carry a usable licence**, against 18 % on GBIF where
non-commercial licences swamp everything; and a species category holds on the
order of a hundred files. It is a complement, not a replacement.

```bash
python3 build_dataset.py --plants plants.csv --out dataset \
  --target-per-species 200 --allow-sa --wikimedia
```

Commons comes last, after GBIF and iNaturalist, and only if the target is not
reached. It has no notion of an observation: every file is its own split group.

Two filters are specific to it. Files that are not photographs — botanical
plates, herbarium scans, maps, diagrams — are discarded on their title; that is
crude and owned as such. And the licence is read from the URL rather than from
the label: "CC BY 3.0 us" or "CC-BY 4.0 Int" read badly, the URL never does.

**The API rate-limits.** Below one second between requests it answers 429. The
connector waits and retries, then **lets the error surface**: a source that
falls over must be written `FAILED` in the log, not disguise itself as "this
species has no images".

## What is not published here

Two scripts of the private tooling are deliberately left out: the one that
regenerates `plants.csv` from the application's own source, and the one that
pulls the photographs users labelled in the app. The first needs the app's
repository to run; the second reads a private backend and personal data. The
collection described here uses public sources only.

## Accepted licences

Images under **CC0 1.0**, **Public Domain Mark** or **CC BY** (every version)
are kept. **CC BY-SA** is too, with `--allow-sa`, and that is how the set is
collected: the decision was taken on the grounds that a trained model is not an
adaptation of the photographs, it reproduces none of them, and the photographs
are never redistributed. Each one stays attributed with its licence in
`ATTRIBUTIONS.md`. **NC**, **ND**, unknown or absent licences, anything
proprietary or coming from an image search engine: refused, without exception.

The filter runs twice: in the GBIF query (`license=CC0_1_0` then
`license=CC_BY_4_0`), then on **each medium's own licence**, because a CC BY
occurrence can carry a CC BY-NC photograph (which is the case in the
`tests/fixtures/gbif_occurrence_page.json` fixture, and it is duly refused).

Every kept image is traced in the manifest with: source, observation
identifier, origin URL, image URL, author, licence, licence URL, download date,
SHA-256. `ATTRIBUTIONS.md` and `attributions.csv` are generated from that and
must be shipped with any trained model — see
[`../docs/data-provenance.md`](../docs/data-provenance.md), which explains what
it costs when they are not.

## What the pipeline does, image by image

1. **Name** — `plants.csv` → GBIF `species/match` (kingdom Plantae). Only
   `EXACT` matches, or `FUZZY` ones with confidence ≥ 95, at species rank or
   below, are accepted; the rest is recorded in `species.json` and skipped.
2. **Collection** — `occurrence/search` with `mediaType=StillImage`,
   `basisOfRecord=HUMAN_OBSERVATION`, per licence, pages of 100, 0.25 s between
   requests and an identifiable User-Agent.
3. **Download** — 25 MB at most; JPEG, PNG, WEBP formats.
4. **Checking** — readable image, EXIF orientation applied, shortest side
   320 px, aspect ratio ≤ 1:12, no animation; reduced to `MAX_SIDE` on the long
   side and re-encoded as JPEG (quality 92) when needed.
5. **Duplicates** — exact ones by SHA-256, near-duplicates by perceptual hash
   (64-bit DCT, Hamming distance ≤ 6) within a species. The same visual under
   two different species goes to `_review/`: that is a doubtful label, to be
   settled by hand.
6. **Splitting** — 80 / 10 / 10 by group (all the photographs of one
   observation, and their near-duplicates, together), deterministic by a hash
   of the group: adding images does not move the old ones. On 20 images per
   species the split is necessarily coarse; it becomes proportional with a few
   hundred images.

## Regenerating or enriching the plant list

```bash
python3 enrich_plants.py            # GBIF then Wikidata
```

Some names have no GBIF key at all: horticultural taxa with no nomenclatural
existence (`Rosa × hybrida`, `Cymbidium hybridum`). They have to be renamed or
treated as genus-level classes before entering the model. Others resolve
through their accepted synonyms (`Prunus dulcis` → `Prunus amygdalus`,
`Allium porrum` → `Allium ampeloprasum`).
