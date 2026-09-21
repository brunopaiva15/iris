"""Coverage before accuracy.

The top-1 on houseplants means nothing until it is said **how many** of them
the model can so much as name: a species absent from the catalogue is never
got wrong, it is simply never offered. Counting as absent a plant the model
knows under another name would therefore falsify both numbers at once, and
that is what these tests watch.
"""
import sys
from pathlib import Path

sys.path.insert(0, str(Path(__file__).resolve().parents[1]))

from indoor import alias_from, cost_of_breadth, resolve  # noqa: E402

LABELS = {'monstera-deliciosa', 'saintpaulia-ionantha', 'citrus-x-limon', 'goeppertia-makoyana'}


def test_a_name_that_is_already_a_class():
    r = resolve(['Monstera deliciosa'], LABELS)
    assert r['classes'] == ['monstera-deliciosa']
    assert r['missing'] == [] and r['coverage'] == 1.0


def test_the_synonym_attaches_the_plant_instead_of_losing_it():
    alias = alias_from([{'internal_id': 'saintpaulia-ionantha',
                         'scientific_name': 'Saintpaulia ionantha',
                         'synonyms': 'Streptocarpus ionanthus'}])
    r = resolve(['Streptocarpus ionanthus'], LABELS, alias)
    assert r['classes'] == ['saintpaulia-ionantha']
    assert r['synonyms'] == {'Streptocarpus ionanthus': 'saintpaulia-ionantha'}


def test_the_hybrid_cross_does_not_make_a_missing_species():
    # The catalogue writes "Citrus × limon", the list writes "Citrus limon".
    r = resolve(['Citrus limon'], LABELS)
    assert r['classes'] == ['citrus-x-limon']
    assert r['hybrids'] == {'Citrus limon': 'citrus-x-limon'} and r['missing'] == []


def test_two_names_for_one_plant_count_once():
    alias = alias_from([{'internal_id': 'saintpaulia-ionantha',
                         'scientific_name': 'Saintpaulia ionantha',
                         'synonyms': 'Streptocarpus ionanthus'}])
    r = resolve(['Saintpaulia ionantha', 'Streptocarpus ionanthus'], LABELS, alias)
    assert r['requested'] == 2 and r['distinct'] == 1
    assert r['duplicates'] == ['saintpaulia-ionantha'] and r['coverage'] == 1.0


def test_the_duplicate_also_counts_when_both_names_are_missing():
    # Calathea and Goeppertia orbifolia: the same plant, neither in the
    # catalogue. Counting them twice would inflate the denominator and lower
    # the coverage for nothing.
    alias = alias_from([{'internal_id': 'calathea-orbifolia',
                         'scientific_name': 'Calathea orbifolia',
                         'synonyms': 'Goeppertia orbifolia'}])
    r = resolve(['Monstera deliciosa', 'Calathea orbifolia', 'Goeppertia orbifolia'], LABELS, alias)
    assert r['distinct'] == 2, 'two plants, not three'
    assert r['coverage'] == 0.5
    assert len(r['missing']) == 2, 'both names are still reported; it is the plants that are deduplicated'


def test_a_missing_plant_is_reported_missing_even_when_its_genus_is_known():
    # `goeppertia-makoyana` is in the catalogue; orbifolia is not, and "the
    # genus is there" does not make the species nameable.
    r = resolve(['Goeppertia orbifolia'], LABELS)
    assert r['classes'] == [] and r['missing'] == ['Goeppertia orbifolia']


def test_synonyms_are_read_with_either_separator():
    alias = alias_from([{'internal_id': 'a-b', 'scientific_name': 'A b', 'synonyms': 'C d; E f, G h'}])
    assert alias['c-d'] == 'a-b' and alias['e-f'] == 'a-b' and alias['g-h'] == 'a-b'


def test_a_row_without_synonyms_breaks_nothing():
    alias = alias_from([{'internal_id': 'a-b', 'scientific_name': 'A b', 'synonyms': ''},
                        {'internal_id': 'c-d', 'scientific_name': 'C d'}])
    assert alias['a-b'] == 'a-b' and alias['c-d'] == 'c-d'


def test_the_price_of_breadth_is_positive_when_restricting_helps():
    gap = cost_of_breadth({'top1': 0.60, 'top3': 0.75, 'accepted_rate': 0.47},
                          {'top1': 0.68, 'top3': 0.82, 'accepted_rate': 0.55})
    assert gap['top1'] == 0.08 and gap['accepted'] == 0.08


def test_no_price_is_announced_without_a_measurement():
    assert cost_of_breadth({'top1': None}, {'top1': 0.68}) is None
