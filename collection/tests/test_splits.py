import csv
import hashlib
from collections import Counter

from plant_dataset.manifest import STATUS_KEPT, STATUS_REJECTED, ImageRecord, now_iso
from plant_dataset.splits import (MIN_TRAIN, MIN_VAL, RATIOS, assign_groups, group_key,
                                  make_splits, repair_species_coverage, split_for, write_splits)


def rec(i, species='Monstera deliciosa', obs=None, phash=None, status=STATUS_KEPT):
    return ImageRecord(species=species, internal_plant_id='x', source='gbif', source_id=f'{i}#0', original_url='', image_url='',
                       author='a', license='CC BY 4.0', license_url='', downloaded_at=now_iso(), checksum=f'{i:064x}',
                       path=f'{species}/{i}.jpg', observation_id=obs or str(i), phash=phash or hashlib.sha256(str(i).encode()).hexdigest()[:16], status=status)


def test_split_is_deterministic_and_roughly_proportional():
    counts = {'train': 0, 'val': 0, 'test': 0}
    for i in range(5000):
        counts[split_for(f'gbif:{i}')] += 1
    assert split_for('gbif:1') == split_for('gbif:1')
    for name, ratio in RATIOS.items():
        assert abs(counts[name] / 5000 - ratio) < 0.03, counts


def test_same_observation_stays_together():
    records = [rec(1, obs='obs'), rec(2, obs='obs'), rec(3)]
    groups = assign_groups(records)
    assert groups[records[0].checksum] == groups[records[1].checksum]
    assert groups[records[0].checksum] != groups[records[2].checksum]
    assert group_key(records[0]) == 'gbif:obs'
    splits = make_splits(records)
    assert splits[records[0].checksum] == splits[records[1].checksum]


def test_near_duplicates_across_observations_merge_groups():
    records = [rec(1, phash='0' * 16), rec(2, phash='0' * 15 + '1'), rec(3, phash='f' * 16)]
    groups = assign_groups(records)
    assert groups[records[0].checksum] == groups[records[1].checksum]
    assert groups[records[2].checksum] != groups[records[0].checksum]


def test_rejected_records_have_no_split(tmp_path):
    records = [rec(1), rec(2, status=STATUS_REJECTED), rec(3, species='Ficus elastica')]
    counts = write_splits(records, tmp_path / 'splits.csv')
    rows = list(csv.DictReader(open(tmp_path / 'splits.csv')))
    assert [r['path'] for r in rows] == ['Ficus elastica/3.jpg', 'Monstera deliciosa/1.jpg']
    assert set(rows[0]) == {'path', 'species', 'internal_plant_id', 'split', 'group', 'captive'}
    assert sum(sum(v.values()) for v in counts.values()) == 2
    assert 'Ficus elastica' in counts and 'Monstera deliciosa' in counts


def test_captive_column_marks_cultivated_photos(tmp_path):
    potted = rec(7, species='Yucca gigantea')
    potted.extra = {'captive': True}
    wild = rec(8, species='Yucca gigantea')
    write_splits([potted, wild], tmp_path / 'splits.csv')
    rows = {r['path']: r['captive'] for r in csv.DictReader(open(tmp_path / 'splits.csv'))}
    assert rows[potted.path] == '1'
    assert rows[wild.path] == '0'


def espece(nom: str, groupes: list[int]) -> list[ImageRecord]:
    """A species whose photographs split into observations of given sizes —
    that imbalance is what decides everything here."""
    # A range of identifiers specific to the species, and stable from one run
    # to the next: `hash()` is salted per process, sha256 is not.
    records, i = [], int(hashlib.sha256(nom.encode()).hexdigest()[:6], 16) * 1000
    for n, taille in enumerate(groupes):
        for _ in range(taille):
            i += 1
            records.append(rec(i, species=nom, obs=f'{nom}-o{n}'))
    return records


def par_taille(groups: dict[str, str]) -> list[str]:
    """Identifiants de groupe, du plus petit au plus grand."""
    tailles = Counter(groups.values())
    return sorted(tailles, key=lambda g: (tailles[g], g))


def tout_en_train(groups: dict[str, str]) -> dict[str, str]:
    return {ck: 'train' for ck in groups}


def test_repair_gives_val_then_test_to_a_species_that_had_neither():
    records = espece('Howea forsteriana', [40, 5, 3, 1])
    groups = assign_groups(records)
    splits = tout_en_train(groups)
    assert repair_species_coverage(records, splits, groups) == {'Howea forsteriana': ['val', 'test']}
    cotes = Counter(splits.values())
    assert cotes['val'] >= MIN_VAL and cotes['test'] >= 1 and cotes['train'] >= MIN_TRAIN
    # The moved group stays whole: an observation is not cut in two.
    assert len({splits[r.checksum] for r in records if r.observation_id.endswith('-o2')}) == 1


def test_repair_completes_a_validation_that_exists_but_is_below_the_threshold():
    """The case seen in one version: 112 species dropped with one or two
    validation images. The old repair held them to be served — the column was
    filled, the `train.py` threshold was not."""
    records = espece('Sinapis alba', [60, 6, 2])
    groups = assign_groups(records)
    petit = par_taille(groups)[0]
    splits = {ck: ('val' if g == petit else 'train') for ck, g in groups.items()}
    assert Counter(splits.values())['val'] == 2

    # Le test repart les mains vides, et c'est voulu : il ne restait qu'un
    # group in training, and `train.py` imposes no threshold on the test.
    assert repair_species_coverage(records, splits, groups) == {'Sinapis alba': ['val']}
    cotes = Counter(splits.values())
    assert cotes['val'] >= MIN_VAL
    assert cotes['train'] >= MIN_TRAIN


def test_repair_prefers_small_groups_to_one_big_one():
    """Filling three images must not move thirty: the smallest groups
    accumulated beat a single one that would do."""
    records = espece('Ficus elastica', [30, 30, 2, 2])
    groups = assign_groups(records)
    splits = tout_en_train(groups)
    repair_species_coverage(records, splits, groups)
    assert Counter(splits.values())['val'] == 4


def test_repair_prefers_one_group_when_accumulating_would_cost_more():
    """L'inverse est vrai aussi : accumuler 1 + 4 pour combler 3 sort une
    image de plus que de prendre le groupe de 4 seul."""
    records = espece('Aloe vera', [40, 4, 1])
    groups = assign_groups(records)
    splits = tout_en_train(groups)
    repair_species_coverage(records, splits, groups)
    assert Counter(splits.values())['val'] == 4


def test_repair_refuses_to_drop_the_training_set_below_the_threshold():
    """A species with 30 images in two groups cannot give its validation
    without falling under `--min-train`: it would be dropped anyway, and ten
    training images would have been lost for nothing. Nothing is touched."""
    records = espece('Crassula ovata', [20, 10])
    groups = assign_groups(records)
    splits = tout_en_train(groups)
    assert repair_species_coverage(records, splits, groups) == {}
    assert set(splits.values()) == {'train'}


def test_repair_leaves_alone_a_species_that_already_meets_the_thresholds():
    """The property that makes the fix safe: a class already admitted by
    `train.py` is never touched, so no image changes side in the previous model
    and the comparison between versions stays honest."""
    records = espece('Monstera deliciosa', [30, 4, 4])
    groups = assign_groups(records)
    val_g, test_g = par_taille(groups)[:2]
    splits = {ck: ('val' if g == val_g else 'test' if g == test_g else 'train')
              for ck, g in groups.items()}
    avant = dict(splits)
    assert repair_species_coverage(records, splits, groups) == {}
    assert splits == avant


def test_repair_never_empties_the_training_set():
    """A species whose photographs all come from one observation has nothing
    to give: emptying it to measure it would advance nothing."""
    records = espece('Dracaena trifasciata', [40])
    groups = assign_groups(records)
    splits = tout_en_train(groups)
    assert repair_species_coverage(records, splits, groups) == {}
    assert set(splits.values()) == {'train'}


def test_repair_only_moves_what_was_in_train():
    """The repair is additive: what was already in validation or test does not
    move, or the comparison with the previous model would lie."""
    records = [rec(i, species='Hoya kerrii', obs=f'o{i}') for i in range(1, 60)]
    brut = make_splits(records, repair=False)
    repare = make_splits(records, repair=True)
    for ck, cote in brut.items():
        if cote != 'train':
            assert repare[ck] == cote
    assert {'train', 'val'} <= set(repare.values())


def test_repair_is_deterministic():
    records = espece('Pilea peperomioides', [30, 6, 3, 2, 1])
    groups = assign_groups(records)
    a, b = tout_en_train(groups), tout_en_train(groups)
    repair_species_coverage(records, a, groups)
    repair_species_coverage(records, b, groups)
    assert a == b
    assert Counter(a.values())['val'] >= MIN_VAL


def test_write_splits_repairs_by_default(tmp_path):
    records = espece('Howea forsteriana', [60, 20, 8, 4, 3])
    counts = write_splits(records, tmp_path / 'splits.csv')
    assert counts['Howea forsteriana'].get('val', 0) >= MIN_VAL
    assert counts['Howea forsteriana'].get('train', 0) >= MIN_TRAIN
