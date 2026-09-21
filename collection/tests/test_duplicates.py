"""Two names for one plant make two classes, and that is a catalogue defect no
photograph can repair.

The images are shared between the two, each trains on half of what it should,
and the resulting confusion shows in the breakdown as an error of the model
when there was nothing to settle.
"""
import sys
from pathlib import Path

sys.path.insert(0, str(Path(__file__).resolve().parents[1]))

from duplicates import group, images_per_class  # noqa: E402


def test_two_names_on_one_taxon_make_a_duplicate():
    g = group({'sansevieria-trifasciata': 11041822, 'dracaena-trifasciata': 11041822,
               'monstera-deliciosa': 2868536})
    assert g == [['dracaena-trifasciata', 'sansevieria-trifasciata']]


def test_two_distinct_plants_do_not_group():
    # Populus alba and Salix alba share their epithet and their family;
    # they are not the same plant.
    assert group({'populus-alba': 3040233, 'salix-alba': 5372513}) == []


def test_unresolved_names_do_not_make_one_big_duplicate():
    # Without this, every unknown would land on `None` and be announced as one
    # and the same plant.
    assert group({'a-one': None, 'b-one': None, 'c-one': 42}) == []


def test_a_taxon_with_three_names_returns_a_single_group():
    g = group({'a-one': 7, 'b-one': 7, 'c-one': 7})
    assert g == [['a-one', 'b-one', 'c-one']], 'one group of three, not three pairs'


def test_the_groups_come_out_in_a_stable_order():
    keys = {'zeta-one': 1, 'alpha-one': 1, 'nu-two': 2, 'beta-two': 2}
    assert group(keys) == [['alpha-one', 'zeta-one'], ['beta-two', 'nu-two']]


def test_without_an_image_set_the_tool_stays_usable(tmp_path):
    assert images_per_class(tmp_path) == {}


def test_the_image_count_says_what_the_split_separates(tmp_path):
    (tmp_path / 'splits.csv').write_text(
        'internal_plant_id,split\n'
        'sansevieria-trifasciata,train\n'
        'sansevieria-trifasciata,train\n'
        'dracaena-trifasciata,test\n', encoding='utf-8')
    assert images_per_class(tmp_path) == {'sansevieria-trifasciata': 2, 'dracaena-trifasciata': 1}
