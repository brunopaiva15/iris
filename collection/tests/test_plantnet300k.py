"""The measurement that decides whether PlantNet-300K is worth a connector.

The set had been ruled out on an intuition — European wild flora against
houseplants. An intuition does not check itself: these tests hold the
machinery of the measurement, so that the number can be redone and disputed.
"""
import sys
from pathlib import Path

sys.path.insert(0, str(Path(__file__).resolve().parents[1]))

from plantnet300k import WHOLE_PLANT, count, mapping  # noqa: E402


def test_the_names_meet_despite_the_authors():
    """"Lactuca virosa L." and "lactuca-virosa" are the same plant."""
    mapped = mapping({'1': 'Lactuca virosa L.', '2': 'Cirsium vulgare (Savi) Ten.'},
                     {'lactuca-virosa', 'cirsium-vulgare', 'monstera-deliciosa'})
    assert mapped == {'1': 'lactuca-virosa', '2': 'cirsium-vulgare'}


def test_a_species_we_do_not_know_is_ignored():
    assert mapping({'9': 'Quercus robur L.'}, {'monstera-deliciosa'}) == {}


def test_two_identifiers_can_land_on_a_single_species():
    """Synonyms and subspecies: their images would rejoin on our side."""
    mapped = mapping({'1': 'Lactuca virosa L.', '2': 'Lactuca virosa'}, {'lactuca-virosa'})
    assert set(mapped) == {'1', '2'} and set(mapped.values()) == {'lactuca-virosa'}


def _img(sid, licence='cc-by-sa', organ='flower'):
    return {'species_id': sid, 'license': licence, 'organ': organ}


def test_refused_licences_are_counted_but_not_kept():
    """Knowing whether a set is 100 % or 18 % usable changes the decision."""
    stats = count({'a': _img('1'), 'b': _img('1', 'cc-by-nc'), 'c': _img('1', 'cc0')},
                  {'1': 'lactuca-virosa'})
    assert stats['images'] == 2, 'only cc-by-sa and cc0 are kept'
    assert stats['images_all_licences'] == 3
    assert stats['per_licence']['cc-by-nc'] == 1


def test_images_outside_the_overlap_do_not_count():
    stats = count({'a': _img('1'), 'b': _img('unknown')}, {'1': 'lactuca-virosa'})
    assert stats['images'] == 1 and stats['species'] == 1


def test_the_whole_plant_is_counted_apart():
    """It is the only framing that resembles what a user photographs: ignoring
    it has already cost dearly once."""
    stats = count({'a': _img('1', organ=WHOLE_PLANT), 'b': _img('1', organ='leaf'),
                   'c': _img('1', organ='flower')}, {'1': 'lactuca-virosa'})
    assert stats['whole']['lactuca-virosa'] == 1
    assert stats['per_organ'] == {WHOLE_PLANT: 1, 'leaf': 1, 'flower': 1}
    assert stats['images'] == 3


def test_a_set_without_overlap_returns_nothing():
    stats = count({'a': _img('1')}, {})
    assert stats['images'] == 0 and stats['species'] == 0
