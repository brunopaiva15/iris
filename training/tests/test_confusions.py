"""Sorting the errors between "within the genus" and "between genera".

That is the whole value of the breakdown: two peperomias confused ask for
nothing — the screen offers five candidates and the right answer is among
them. A yucca taken for maize asks for images, and that is the kind of pair
that must surface.
"""
import sys
from pathlib import Path

sys.path.insert(0, str(Path(__file__).resolve().parents[1]))

from confusions import (by_chance, cross, family, genus,  # noqa: E402
                        rate_per_species, struggling_species, weakness)


def test_the_genus_comes_from_the_identifier():
    assert genus('monstera-deliciosa') == 'monstera'
    assert genus('abelia-x-grandiflora') == 'abelia', 'hybrids keep their genus'


def test_plants_csv_has_the_last_word_when_it_is_there():
    assert genus('goeppertia-orbifolia', {'goeppertia-orbifolia': 'Calathea orbifolia'}) == 'calathea'


def test_a_confusion_within_the_genus_is_not_counted_as_a_defect():
    stats = cross([('peperomia-caperata', 'peperomia-obtusifolia')])
    assert stats['within_genus'] == 1 and stats['outside_genus'] == 0
    assert stats['between_genera'] == {}


def test_a_confusion_between_genera_is_kept_with_its_pair():
    stats = cross([('yucca-gigantea', 'zea-mays')])
    assert stats['outside_genus'] == 1
    assert stats['between_genera'][('yucca', 'zea')] == 1


def test_right_answers_do_not_count_as_errors():
    stats = cross([('monstera-deliciosa', 'monstera-deliciosa')] * 3)
    assert stats['right'] == 3 and stats['outside_genus'] == 0 and stats['within_genus'] == 0
    assert stats['per_species'] == {}


def test_the_direction_of_the_confusion_is_preserved():
    """A yucca taken for maize is not maize taken for a yucca: two different
    defects, and two different collections."""
    stats = cross([('yucca-gigantea', 'zea-mays'), ('zea-mays', 'yucca-gigantea')])
    assert stats['between_genera'][('yucca', 'zea')] == 1
    assert stats['between_genera'][('zea', 'yucca')] == 1


def test_species_seen_too_rarely_are_not_ranked():
    """A failure rate over three photographs means nothing."""
    pairs = [('rara-avis', 'monstera-deliciosa')] * 3
    pairs += [('yucca-gigantea', 'zea-mays')] * 6
    names = [s for s, *_ in struggling_species(cross(pairs), minimum=5)]
    assert names == ['yucca-gigantea']


def test_species_are_ranked_by_failure_rate():
    pairs = [('a-one', 'b-two')] * 5 + [('a-one', 'a-one')] * 5          # 50 % missed
    pairs += [('c-three', 'd-four')] * 9 + [('c-three', 'c-three')]      # 90 % missed
    ranking = struggling_species(cross(pairs))
    assert [s for s, *_ in ranking] == ['c-three', 'a-one']
    species, missed, seen, culprit, how_many = ranking[0]
    assert (missed, seen, culprit, how_many) == (9, 10, 'd-four', 9)


# --- The family, and the baseline without which the shares say nothing ----
#
# The first version sorted the errors into two piles and called everything
# outside the genus a "real defect". On an earlier model that was 87 % — a
# number that counts the structure of the catalogue as a defect, since 549
# classes out of 1,457 are alone in their genus and *cannot* err there.

FAMILIES = {
    'picea-abies': 'Pinaceae', 'abies-alba': 'Pinaceae',
    'parthenocissus-inserta': 'Vitaceae', 'petroselinum-crispum': 'Apiaceae',
}


def test_two_genera_of_one_family_are_not_a_defect():
    stats = cross([('picea-abies', 'abies-alba')], None, FAMILIES)
    assert stats['within_genus'] == 0
    assert stats['same_family'] == 1 and stats['outside_family'] == 0
    assert stats['outside_genus'] == 1, 'the outside-genus total stays readable as before'


def test_a_creeper_taken_for_parsley_is_a_real_defect():
    stats = cross([('parthenocissus-inserta', 'petroselinum-crispum')], None, FAMILIES)
    assert stats['outside_family'] == 1 and stats['same_family'] == 0
    assert stats['between_families'][('vitaceae', 'apiaceae')] == 1


def test_without_families_the_third_drawer_stays_empty_rather_than_lying():
    stats = cross([('picea-abies', 'abies-alba')])
    assert stats['same_family'] == 0 and stats['outside_family'] == 1
    assert stats['between_families'] == {}


def test_an_unknown_family_is_never_confused_with_another():
    # Two species with no recorded family must not be declared "same family"
    # on the strength of two empty values.
    stats = cross([('aaa-one', 'bbb-one')], None, {'ccc-one': 'Rosaceae'})
    assert stats['same_family'] == 0 and stats['outside_family'] == 1


def test_chance_is_computed_on_the_catalogue_not_on_the_errors():
    c = by_chance(['a-one', 'a-two', 'b-one'], {'a-one': 'X', 'a-two': 'X', 'b-one': 'X'})
    assert round(c['genus'], 4) == 0.3333, 'a-one and a-two answer each other, b-one never'
    assert round(c['family'], 4) == 0.6667
    assert c['alone_in_their_genus'] == 1


def test_a_catalogue_of_one_class_has_no_chance_baseline():
    assert by_chance(['a-one'])['genus'] == 0.0


def test_the_family_comes_from_plants_csv_and_reads_in_lower_case():
    assert family('picea-abies', FAMILIES) == 'pinaceae'
    assert family('unknown-x', FAMILIES).startswith('?'), 'a missing family stays distinct'


def test_the_third_drawer_has_its_baseline_like_the_other_two():
    # Without it, "73 % beyond" reads as a disaster when chance would put
    # 98 % there: the model is better, not worse.
    c = by_chance(['a-one', 'a-two', 'b-one'], {'a-one': 'X', 'a-two': 'X', 'b-one': 'X'})
    assert round(c['genus'] + c['family'] + c['beyond'], 6) == 1.0
    assert c['beyond'] == 0.0, 'here every class belongs to a single family'


def test_the_rate_per_species_only_counts_species_seen_often_enough():
    stats = cross([('a-one', 'b-one')] * 4 + [('a-one', 'a-one')] + [('c-one', 'd-one')] * 3)
    rates = rate_per_species(stats)
    assert rates == {'a-one': 0.2}, 'c-one has only 3 images: it is not ranked'


def test_the_bands_do_not_depend_on_the_sample_size():
    # The same model, the same species at 20 %, seen 5 times then 30 times:
    # the "zero right answer" flips, the band does not. That is what replaced
    # the measurement — 11 species out of 575 in the sample, 5 out of 1,422 in
    # the full test, for an unchanged model.
    small = cross([('a-one', 'b-one')] * 5)
    large = cross([('a-one', 'b-one')] * 24 + [('a-one', 'a-one')] * 6)
    assert weakness(small)['zero'] == ['a-one']
    assert weakness(large)['zero'] == [], 'the same weakness no longer counts as zero'
    assert weakness(small)['under_25'] == weakness(large)['under_25'] == ['a-one']


def test_a_solid_species_appears_in_no_band():
    stats = cross([('a-one', 'a-one')] * 9 + [('a-one', 'b-one')])
    w = weakness(stats)
    assert w['measurable'] == 1 and w['under_50'] == [] and w['zero'] == []
