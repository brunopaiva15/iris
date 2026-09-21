"""The cultivar is a second axis, not one more class.

A photograph of "Thai Constellation" is a photograph of *Monstera deliciosa*:
iNaturalist records it that way and Commons has one of it. Making classes of
them would recreate the duplicate-class defect — two labels for one plant, the
images shared, an unwinnable confusion. The model answers the species, the
application offers the known cultivars: this file is that list.
"""
import sys
from pathlib import Path

sys.path.insert(0, str(Path(__file__).resolve().parents[1]))

from cultivars import epithet, group  # noqa: E402


def test_the_epithet_comes_out_of_straight_quotes():
    assert epithet("Epipremnum aureum 'Neon'", 'Epipremnum aureum') == 'Neon'


def test_wikidata_writes_with_every_apostrophe_in_unicode():
    # "Hosta ʽFortunei’", "Ficus elastica ʽRobusta’": the horticultural rule
    # knows only one, and two spellings would make two cultivars.
    assert epithet('Ficus elastica ʽRobusta’', 'Ficus elastica') == 'Robusta'
    assert epithet('Hosta ʽGold Standard’', 'Hosta') == 'Gold Standard'


def test_a_taxon_that_is_not_a_cultivar_is_discarded():
    # Wikidata files things under Q4886 that are not cultivars. The
    # horticultural code settles it: a cultivar name takes a capital, a species
    # or variety epithet never does.
    assert epithet('Hosta decorata', 'Hosta') is None, 'decorata is a species'
    assert epithet('Agave americana var. medio-picta alba', 'Agave americana') is None
    assert epithet('Hosta', 'Hosta') is None


def test_a_cultivar_without_quotes_passes_if_it_has_its_capital():
    assert epithet('Ficus elastica Robusta', 'Ficus elastica') == 'Robusta'


def test_an_empty_name_does_not_bring_the_list_down():
    assert epithet('', 'Hosta') is None
    assert epithet(None, 'Hosta') is None


def test_cultivars_are_grouped_by_species_and_deduplicated():
    rows = [('Ficus elastica', "Ficus elastica 'Robusta'"),
            ('Ficus elastica', 'Ficus elastica ʽRobusta’'),
            ('Ficus elastica', "Ficus elastica 'Tineke'"),
            ('Epipremnum aureum', "Epipremnum aureum 'Neon'")]
    assert group(rows) == {
        'Epipremnum aureum': ['Neon'],
        'Ficus elastica': ['Robusta', 'Tineke'],
    }


def test_a_species_without_a_cultivar_does_not_enter_the_list():
    assert group([('Monstera deliciosa', 'Monstera deliciosa')]) == {}
