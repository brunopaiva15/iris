"""Train / validation / test splitting.

The unit is not the image but the group: all the photographs of one
observation, and everything judged a near-duplicate of one of them, go
together. Otherwise the test set would see images it has already seen in
training, and the measured accuracy would be a lie.

The assignment is deterministic (a hash of the group): re-running the split
after adding images does not move the old ones.

It does not look at the species, and that is its flaw: a species whose
photographs come from few observations can have all its draws fall on the same
side, or receive only a handful of images, and be left out of the model for
lack of validation. `repair_species_coverage` repairs that one case, without
touching the rest — aiming at the thresholds `train.py` really applies, rather
than at the mere presence of a group.
"""
from __future__ import annotations

import csv
import hashlib
from collections import defaultdict
from pathlib import Path

from .dedup import NEAR_THRESHOLD, UnionFind, near_duplicate_groups
from .manifest import STATUS_KEPT, ImageRecord

RATIOS = {'train': 0.8, 'val': 0.1, 'test': 0.1}

# `train.py` only admits a class if it has `--min-train` training images *and*
# `--min-val` validation images. The repair must aim at those numbers: filling
# the "val" column with a single image satisfies the split and still leaves the
# class out.
MIN_TRAIN = 25
MIN_VAL = 3


def group_key(r: ImageRecord) -> str:
    return f'{r.source}:{r.observation_id}' if r.observation_id else f'{r.source}:{r.source_id}'


def assign_groups(records: list[ImageRecord], threshold: int = NEAR_THRESHOLD) -> dict[str, str]:
    """Checksum → group identifier, after merging observations linked by a
    near-duplicate."""
    uf = UnionFind()
    kept = [r for r in records if r.status == STATUS_KEPT]
    for r in kept:
        uf.union(group_key(r), r.checksum)
    # Two observations whose photographs resemble each other too much end up
    # together.
    for group in near_duplicate_groups(kept, threshold).values():
        for r in group[1:]:
            uf.union(group[0].checksum, r.checksum)
    return {r.checksum: uf.find(r.checksum) for r in kept}


def split_for(group_id: str, ratios: dict[str, float] = RATIOS, salt: str = 'flora-v1') -> str:
    h = int(hashlib.sha256(f'{salt}:{group_id}'.encode()).hexdigest()[:8], 16) / 0xFFFFFFFF
    acc = 0.0
    for name, ratio in ratios.items():
        acc += ratio
        if h < acc:
            return name
    return list(ratios)[-1]


def _groups_to_move(sizes: dict[str, list[str]], side: dict[str, str],
                    missing: int, min_train: int) -> list[str]:
    """The training groups to move to the other side in order to fill
    `missing` images — or nothing, if it cannot be done without damaging the
    species.

    Two plans are considered: the smallest group that fills the gap on its own,
    and the smallest ones accumulated. The one taking the fewest images out of
    training wins. Both are needed: on groups of 2, 2 and 100 images, the first
    plan would move a hundred to fill three; on groups of 1 and 4, the second
    would move five where four are enough.

    Two safeguards, and a plan violating either is abandoned whole rather than
    played by halves:

    - **at least one group stays in training** — a species with only one group
      has nothing to give;
    - **at least `min_train` images remain** — dropping below the `train.py`
      threshold would push the class out of the model, which is precisely what
      the repair is meant to avoid. A species already below the threshold is
      not touched either: taking images from it would not save it.

    The group identifier breaks ties at equal size, so that two runs give the
    same split.
    """
    candidates = sorted((g for g, s in side.items() if s == 'train'),
                        key=lambda g: (len(sizes[g]), g))
    in_train = sum(len(sizes[g]) for g in candidates)

    plans: list[list[str]] = []
    single = next((g for g in candidates if len(sizes[g]) >= missing), None)
    if single is not None:
        plans.append([single])
    running, left = [], missing
    for g in candidates:
        running.append(g)
        left -= len(sizes[g])
        if left <= 0:
            plans.append(list(running))
            break

    def size_of(plan: list[str]) -> int:
        return sum(len(sizes[g]) for g in plan)

    valid = [plan for plan in plans
             if len(plan) < len(candidates) and in_train - size_of(plan) >= min_train]
    return min(valid, key=lambda plan: (size_of(plan), len(plan), plan)) if valid else []


def repair_species_coverage(records: list[ImageRecord], splits: dict[str, str],
                            groups: dict[str, str], min_val: int = MIN_VAL,
                            min_train: int = MIN_TRAIN) -> dict[str, list[str]]:
    """Give something to validate on, then something to test on, to the species
    that lack it.

    `split_for` hashes the group and does not look at the species. On a species
    whose photographs come from few observations, the draws can all fall on the
    same side, or leave a single image in validation: `train.py` then drops the
    class (`--min-val`), and these are species that had plenty to learn from.
    *Howea forsteriana* was absent from one version with 95 training images and
    none in validation; the kentia palm did not lack photographs, it lacked a
    draw.

    **The threshold matters as much as the presence.** Aiming at "at least one
    group" filled the column without letting the class in: in one version, 112
    species with a median of 47 training images — up to 193 — stayed out with
    one or two validation images, because the repair had given them their
    smallest group. So `min_val` images are aimed at, and a validation set that
    exists but stays below the threshold is repaired too.

    **The repair is minimal, and that is essential.** Nothing is
    redistributed: training groups are moved for the species missing a side
    only, and every other assignment stays exactly what it was. A general
    re-split would push into the test set images the previous model saw in
    training — it would look better than it is, and the comparison between two
    versions would mean nothing. A species already satisfying the thresholds is
    never touched: this fix therefore cannot move an image of a class already
    present in the previous model.

    Validation comes before test, because validation is what decides that a
    class exists. The test split has no threshold in `train.py`: one image is
    enough.

    Returns the repaired species and what they received. `splits` is modified
    in place.
    """
    per_species: dict[str, dict[str, list[str]]] = defaultdict(lambda: defaultdict(list))
    for r in records:
        if r.status == STATUS_KEPT:
            per_species[r.species][groups[r.checksum]].append(r.checksum)

    repaired: dict[str, list[str]] = {}
    for species in sorted(per_species):
        sizes = per_species[species]
        side = {g: splits[cks[0]] for g, cks in sizes.items()}
        for missing_side, target in (('val', min_val), ('test', 1)):
            present = sum(len(sizes[g]) for g, s in side.items() if s == missing_side)
            if present >= target:
                continue
            to_move = _groups_to_move(sizes, side, target - present, min_train)
            for g in to_move:
                side[g] = missing_side
                for ck in sizes[g]:
                    splits[ck] = missing_side
            if to_move:
                repaired.setdefault(species, []).append(missing_side)
    return repaired


def make_splits(records: list[ImageRecord], threshold: int = NEAR_THRESHOLD,
                groups: dict[str, str] | None = None, repair: bool = True,
                min_val: int = MIN_VAL, min_train: int = MIN_TRAIN) -> dict[str, str]:
    """Checksum → train | val | test, for the kept images.

    `repair=False` returns the split as it was before `repair_species_coverage`
    existed: that is what reproducing a model trained before it requires.
    """
    if groups is None:
        groups = assign_groups(records, threshold)
    splits = {ck: split_for(g) for ck, g in groups.items()}
    if repair:
        repair_species_coverage(records, splits, groups, min_val, min_train)
    return splits


def write_splits(records: list[ImageRecord], path: str | Path, threshold: int = NEAR_THRESHOLD,
                 repair: bool = True, min_val: int = MIN_VAL,
                 min_train: int = MIN_TRAIN) -> dict[str, dict[str, int]]:
    # The groups are computed once: `assign_groups` compares the hashes of
    # every kept image, and that is the expensive part.
    groups = assign_groups(records, threshold)
    splits = make_splits(records, threshold, groups=groups, repair=repair,
                         min_val=min_val, min_train=min_train)
    counts: dict[str, dict[str, int]] = defaultdict(lambda: defaultdict(int))
    with open(path, 'w', newline='', encoding='utf-8') as f:
        w = csv.writer(f)
        # `captive`: the photograph is of a cultivated plant (iNaturalist).
        # Accuracy measured on those images is the only figure describing what
        # the application will do on its users' photographs.
        w.writerow(['path', 'species', 'internal_plant_id', 'split', 'group', 'captive'])
        for r in sorted((r for r in records if r.status == STATUS_KEPT), key=lambda r: (r.species, r.path)):
            s = splits[r.checksum]
            w.writerow([r.path, r.species, r.internal_plant_id, s, groups[r.checksum], '1' if (r.extra or {}).get('captive') else '0'])
            counts[r.species][s] += 1
    return {k: dict(v) for k, v in counts.items()}
