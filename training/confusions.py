#!/usr/bin/env python3
"""What the model gets wrong — not only how much.

    python3 confusions.py --dataset ../collection/dataset \\
        --model ./out --sample 6000

`train.py` returns top-1, top-3, macro-F1 and the threshold curve: we know
*how much* the model gets wrong, never *what*. Without a confusion breakdown,
the defects are found one at a time, by hand, on real photographs — that is
how a yucca taken for maize was discovered, and it had had time to cross three
versions.

Three drawers, because two of them lied:

- **within the genus**: two maples, two firs, two peperomias. The screen
  offers five candidates and the right answer is nearly always among them;
- **within the family**: *Picea* → *Abies*, two Pinaceae. Botanically close,
  visually close, and hard for a human too;
- **beyond**: *Parthenocissus* → *Petroselinum*, a Virginia creeper taken for
  parsley. That is where the real defects are, and those pairs are fixed with
  images.

**And each share is read against chance, or it says nothing.** 38 % of the
classes are alone in their genus: their errors *cannot* stay within the genus.
An error drawn at random would stay there 0.17 % of the time; announcing 13 %
is therefore not "a small expected share" but seventy times chance. The first
version of this file called the remaining 87 % "real defects", which amounted
to counting the structure of the catalogue as a defect.

The report therefore reads top to bottom: the three shares and their chance
baseline, then the pairs responsible, and finally the species that fail most
often, with what is answered in their place.
"""
from __future__ import annotations

import argparse
import csv
import random
import sys
from collections import Counter, defaultdict
from pathlib import Path

sys.path.insert(0, str(Path(__file__).resolve().parent))


def genus(internal_id: str, names: dict[str, str] | None = None) -> str:
    """The genus of a species, from its internal id.

    Ids are `genus-epithet` (`monstera-deliciosa`), so the prefix is enough —
    and stays right for hybrids, whose id carries the `x` in second position
    (`abelia-x-grandiflora`). `plants.csv` is used when present, but the tool
    must run without it.
    """
    if names and internal_id in names:
        return names[internal_id].split()[0].lower()
    return internal_id.split('-')[0]


def family(internal_id: str, families: dict[str, str] | None = None) -> str:
    """The family of a species, when `plants.csv` gives it.

    It gave it for the classes of the shipped model, but the tool must run
    without it: an unknown family is worth its own identifier, so it is never
    confused with another.
    """
    if families and internal_id in families:
        return families[internal_id].strip().lower()
    return f'?{internal_id}'


def by_chance(classes: list[str], families: dict[str, str] | None = None,
              names: dict[str, str] | None = None) -> dict:
    """Where an error drawn at random would fall — the only baseline worth
    anything.

    Without it, "13 % of the errors stay within the genus" reads as a small
    share, when it is seventy times what chance would give. The computation
    assumes the classes are equiprobable: false in detail, but the order of
    magnitude is enough to prevent the misreading.
    """
    n = len(classes)
    if n < 2:
        return {'genus': 0.0, 'family': 0.0, 'alone_in_their_genus': 0}
    per_genus: Counter = Counter(genus(c, names) for c in classes)
    per_family: Counter = Counter(family(c, families) for c in classes)
    pg = sum(per_genus[genus(c, names)] - 1 for c in classes) / (n * (n - 1))
    pf = sum(per_family[family(c, families)] - per_genus[genus(c, names)] for c in classes) / (n * (n - 1))
    return {
        'genus': pg,
        'family': pf,
        # The third drawer deserves its baseline like the other two. Without
        # it, "73 % beyond" reads as a disaster, when chance would put 97.9 %
        # there: the model is better than chance, simply not by much. That is
        # a workload, not a scandal — and the opposite would be a real
        # scandal, a model unable to recognise even a family.
        # The floating-point residue of 1 - pg - pf is 1e-16 when all the
        # classes share one family; left as is, it would print a fifteen-digit
        # ratio in the report.
        'beyond': rest if (rest := 1.0 - pg - pf) > 1e-12 else 0.0,
        'alone_in_their_genus': sum(1 for c in classes if per_genus[genus(c, names)] == 1),
    }


def cross(pairs: list[tuple[str, str]], names: dict[str, str] | None = None,
          families: dict[str, str] | None = None) -> dict:
    """Sort the errors by taxonomic distance between the two species.

    `pairs` is a list of (truth, prediction) in internal ids.

    Without `families` only two drawers remain and `same_family` is zero: the
    report then says so rather than announcing a null share.
    """
    right = within_genus = same_family = outside_family = 0
    between_genera: Counter = Counter()
    between_families: Counter = Counter()
    per_species: dict[str, Counter] = defaultdict(Counter)
    seen: Counter = Counter()
    for truth, predicted in pairs:
        seen[truth] += 1
        if truth == predicted:
            right += 1
            continue
        per_species[truth][predicted] += 1
        gt, gp = genus(truth, names), genus(predicted, names)
        if gt == gp:
            within_genus += 1
            continue
        between_genera[(gt, gp)] += 1
        ft, fp = family(truth, families), family(predicted, families)
        if families and ft == fp:
            same_family += 1
        else:
            outside_family += 1
            if families:
                between_families[(ft, fp)] += 1
    return {
        'images': len(pairs),
        'right': right,
        'within_genus': within_genus,
        'same_family': same_family,
        'outside_family': outside_family,
        'outside_genus': same_family + outside_family,
        'between_genera': between_genera,
        'between_families': between_families,
        'per_species': per_species,
        'seen': seen,
    }


def struggling_species(stats: dict, minimum: int = 5) -> list[tuple[str, int, int, str, int]]:
    """The species missed most often, with what is answered in their place.

    Below `minimum` test images a failure rate means nothing — one does not
    rank a species on three photographs.
    """
    out = []
    for species, seen in stats['seen'].items():
        if seen < minimum:
            continue
        errors = stats['per_species'].get(species)
        if not errors:
            continue
        culprit, how_many = errors.most_common(1)[0]
        out.append((species, sum(errors.values()), seen, culprit, how_many))
    out.sort(key=lambda r: (-r[1] / r[2], -r[2]))
    return out


def rate_per_species(stats: dict, minimum: int = 5) -> dict[str, float]:
    """The top-1 of every species seen often enough for it to mean something.
    One does not rank a species on three photographs."""
    out = {}
    for species, seen in stats['seen'].items():
        if seen < minimum:
            continue
        # The ratio of the right answers, not the complement of the missed
        # ones: `1 - 4/5` returns 0.199999… and two identical species would
        # stop being equal.
        out[species] = (seen - sum(stats['per_species'].get(species, {}).values())) / seen
    return out


def weakness(stats: dict, minimum: int = 5) -> dict:
    """The distribution of species by top-1 — the real collection list.

    **Counting the species with zero correct answers does not work**, and the
    measurement showed it by contradicting itself: on a sample of 6,000
    images, 11 species out of 575 had nothing right; on the 29,000 of the full
    test, 5 out of 1,422. The second figure is not an improvement, it is the
    same model. A species at 15 % top-1 easily misses its five images; it
    almost never misses thirty. **The zero measures the number of test images,
    not the weakness of the class.**

    A threshold, on the other hand, does not move with the sample size. Hence
    bands, and a "never recognised" kept for the record but announced for what
    it is.
    """
    rates = rate_per_species(stats, minimum)
    return {
        'measurable': len(rates),
        'zero': sorted(s for s, t in rates.items() if t == 0.0),
        'under_25': sorted(s for s, t in rates.items() if t < 0.25),
        'under_50': sorted(s for s, t in rates.items() if t < 0.50),
        'rates': rates,
    }


def _report(stats: dict, names: dict, families: dict, chance: dict, top: int) -> None:
    n, right = stats['images'], stats['right']
    errors = n - right
    print(f'{n} images, {right} right ({right / n:.1%})\n')
    if not errors:
        print('no error: nothing to say.')
        return

    def share(how_many, baseline=None):
        line = f"  {how_many:6d}  ({how_many / errors:5.1%})"
        if baseline is not None:
            line += f"   by chance: {baseline:5.2%}"
            if baseline > 0:
                ratio = how_many / errors / baseline
                line += f"  → x{ratio:.0f}" if ratio >= 2 else f"  → x{ratio:.2f}"
        # Fixed width: the three shares must read as a column, including the
        # one with no baseline to show.
        return line.ljust(48)

    print(f'{errors} errors, sorted by taxonomic distance:')
    print(f"{share(stats['within_genus'], chance.get('genus'))}   within the same genus")
    if families:
        print(f"{share(stats['same_family'], chance.get('family'))}   within the same family")
        print(f"{share(stats['outside_family'], chance.get('beyond'))}   beyond  ← the real defects")
    else:
        print(f"{share(stats['outside_genus'])}   outside the genus (without `plants.csv`, the family is unknown)")
    alone = chance.get('alone_in_their_genus')
    if alone:
        print(f"\n{alone} classes are alone in their genus: their errors *cannot* stay")
        print('there. That is why the "by chance" column exists — without it, the first')
        print('line would read as a small share instead of its opposite.')

    def pretty(i):
        return names.get(i, i)

    print(f'\nthe {top} most frequent genus confusions:')
    for (gt, gp), how_many in stats['between_genera'].most_common(top):
        print(f'  {gt:22s} → {gp:22s} {how_many:5d}')

    if stats['between_families']:
        print(f'\nthe {top} most frequent family confusions — the ones that cost images:')
        for (ft, fp), how_many in stats['between_families'].most_common(top):
            print(f'  {ft:22s} → {fp:22s} {how_many:5d}')

    w = weakness(stats)
    if w['measurable']:
        n = w['measurable']
        print(f"\nthe {n} species seen at least 5 times, by top-1:")
        print(f"  {len(w['under_25']):5d}  ({len(w['under_25']) / n:5.1%})   under 25 %  ← the collection list")
        print(f"  {len(w['under_50']):5d}  ({len(w['under_50']) / n:5.1%})   under 50 %")
        print(f"  {len(w['zero']):5d}  ({len(w['zero']) / n:5.1%})   not a single right answer")
        print('\nThe last line mostly measures the number of test images: a species at')
        print('15 % easily misses its five photographs, almost never its thirty. The')
        print('first two make the work list.')

    print(f'\nthe {top} most often missed species (at least 5 test images):')
    for species, missed, seen, culprit, how_many in struggling_species(stats)[:top]:
        if genus(species, names) == genus(culprit, names):
            word = 'same genus'
        elif families and family(species, families) == family(culprit, families):
            word = 'same family'
        else:
            word = 'BEYOND'
        print(f'  {pretty(species):34s} {missed:3d}/{seen:3d} missed → {pretty(culprit):32s} x{how_many} ({word})')


def main(argv: list[str] | None = None) -> int:
    ap = argparse.ArgumentParser(description=__doc__, formatter_class=argparse.RawDescriptionHelpFormatter)
    ap.add_argument('--dataset', default='../collection/dataset')
    ap.add_argument('--model', default='./out')
    ap.add_argument('--sample', type=int, default=6000, help='test images drawn at random, 0 = all of them')
    ap.add_argument('--seed', type=int, default=20260905)
    ap.add_argument('--captive', action='store_true', help='keep only the photographs of cultivated plants')
    ap.add_argument('--top', type=int, default=20)
    ap.add_argument('--csv', help='write every (truth, prediction) pair, to dig elsewhere')
    ap.add_argument('--pairs', help='read back a CSV of pairs instead of running the inferences again')
    ap.add_argument('--plants', default='../collection/plants.csv')
    args = ap.parse_args(argv)

    names, families = {}, {}
    plants = Path(args.plants)
    if plants.exists():
        for r in csv.DictReader(plants.open(encoding='utf-8')):
            names[r['internal_id']] = r['scientific_name']
            if r.get('family', '').strip():
                families[r['internal_id']] = r['family']

    classes = [l.strip() for l in (Path(args.model) / 'labels.txt').read_text(encoding='utf-8').splitlines() if l.strip()]

    if args.pairs:
        # The pairs are enough for the report: reading back a file already
        # written costs a second where redoing 6,000 inferences costs twenty
        # minutes. That is what made it possible to fix the reading of this
        # report without calling up the training machine again.
        with open(args.pairs, newline='', encoding='utf-8') as f:
            pairs = [(r['truth'], r['prediction']) for r in csv.DictReader(f)]
        print(f'{len(pairs)} pairs read back from {args.pairs}\n', flush=True)
        _report(cross(pairs, names, families), names, families,
                by_chance(classes, families, names), args.top)
        return 0

    # Imported here: they pull in TensorFlow, which the pure functions of this
    # file do not need in order to be tested.
    import numpy as np
    from compare_models import load_model, predict, read_test

    model = load_model(Path(args.model))
    rows = [(p, t) for p, t, cap in read_test(Path(args.dataset)) if not args.captive or cap]
    rows = [(p, t) for p, t in rows if t in model['index']]
    if args.sample and len(rows) > args.sample:
        rows = random.Random(args.seed).sample(rows, args.sample)
    print(f"model v{model['version']} — {len(rows)} test images"
          f"{' (cultivated plants)' if args.captive else ''}\n", flush=True)

    pairs = []
    for i, (path, truth) in enumerate(rows, 1):
        pairs.append((truth, model['labels'][int(np.argmax(predict(model, path)))]))
        if i % 500 == 0:
            print(f'  {i}/{len(rows)}', flush=True)
    print()

    if args.csv:
        with open(args.csv, 'w', newline='', encoding='utf-8') as f:
            w = csv.writer(f)
            w.writerow(['truth', 'prediction'])
            w.writerows(pairs)
        print(f'{len(pairs)} pairs written to {args.csv}\n')

    _report(cross(pairs, names, families), names, families,
            by_chance(classes, families, names), args.top)
    return 0


if __name__ == '__main__':
    raise SystemExit(main())
