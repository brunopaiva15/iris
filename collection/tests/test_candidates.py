"""Choosing the next species by what people actually grow.

The confusion breakdown showed that 62 of the model's 63 weakest species are
trees and wild plants, and the indoor measurement that the breadth of the
catalogue costs ten points to whoever photographs their living room. Drawing
1,500 more names from the available flora would grow the wrong population:
hence a ranking, and hence these tests on what it lets through.
"""
import sys
from pathlib import Path

sys.path.insert(0, str(Path(__file__).resolve().parents[1]))

from candidates import already_in_catalogue, keep  # noqa: E402


def taxon(name, count, rank='species'):
    return {'count': count, 'taxon': {'name': name, 'rank': rank}}


def test_the_inaturalist_order_is_preserved():
    # That is the whole point: the rank says how many people grow the plant.
    r = keep([taxon('Nerium oleander', 59037), taxon('Acer palmatum', 52626)], set())
    assert r == [('Nerium oleander', 59037), ('Acer palmatum', 52626)]


def test_a_species_already_in_the_catalogue_does_not_count_as_a_novelty():
    r = keep([taxon('Monstera deliciosa', 999), taxon('Nerium oleander', 10)],
             {'monstera-deliciosa'})
    assert r == [('Nerium oleander', 10)]


def test_a_genus_or_a_family_does_not_make_a_class():
    r = keep([taxon('Rosa', 9000, 'genus'), taxon('Rosaceae', 8000, 'family'),
              taxon('Rosa canina', 700)], set())
    assert r == [('Rosa canina', 700)]


def test_hybrids_are_kept():
    # "Hibiscus × rosa-sinensis" is the most cultivated plant in the world;
    # discarding it because its rank is "hybrid" would be absurd.
    r = keep([taxon('Hibiscus × rosa-sinensis', 62498, 'hybrid')], set())
    assert r == [('Hibiscus × rosa-sinensis', 62498)]


def test_the_hybrid_and_its_signless_form_are_the_same_plant():
    # iNaturalist writes sometimes "Citrus × limon", sometimes "Citrus limon".
    # Without the shared internal key, the same plant would be collected twice.
    r = keep([taxon('Citrus × limon', 500, 'hybrid'), taxon('Citrus limon', 400)], set())
    assert len(r) == 1


def test_a_duplicate_within_the_page_passes_only_once():
    r = keep([taxon('Nerium oleander', 100), taxon('nerium oleander', 90)], set())
    assert r == [('Nerium oleander', 100)]


def test_a_taxon_without_a_name_does_not_bring_the_list_down():
    assert keep([{'count': 5, 'taxon': {}}, taxon('Aloe vera', 3)], set()) == [('Aloe vera', 3)]


def test_what_is_known_joins_the_catalogue_and_the_shipped_classes(tmp_path):
    # The two do not coincide: a species can be in the catalogue and have been
    # dropped from the model for lack of images.
    plants = tmp_path / 'plants.csv'
    plants.write_text('internal_id,scientific_name\nmonstera-deliciosa,Monstera deliciosa\n', encoding='utf-8')
    labels = tmp_path / 'labels.txt'
    labels.write_text('ficus-lyrata\n\n', encoding='utf-8')
    assert already_in_catalogue(plants, labels) == {'monstera-deliciosa', 'ficus-lyrata'}


def test_without_files_the_known_catalogue_is_empty(tmp_path):
    assert already_in_catalogue(tmp_path / 'absent.csv', tmp_path / 'absent.txt') == set()


def test_register_returns_the_rows_missing_from_the_catalogue():
    # `build_dataset.py --only-file` only *filters* plants.csv: a candidate
    # absent from the catalogue is not collected, it is ignored in silence.
    # This is the piece that was missing between the two.
    import sys
    from pathlib import Path as P
    sys.path.insert(0, str(P(__file__).resolve().parents[1]))
    from candidates import register
    from plant_dataset.taxonomy import PlantEntry

    existing = [PlantEntry.from_name('Monstera deliciosa')]
    additions = register(existing, [('Monstera deliciosa', 9), ('Ixora coccinea', 28355)])
    assert [e.internal_id for e in additions] == ['ixora-coccinea']
    assert additions[0].genus == 'Ixora' and additions[0].epithet == 'coccinea'


def test_register_does_not_double_a_repeated_candidate():
    import sys
    from pathlib import Path as P
    sys.path.insert(0, str(P(__file__).resolve().parents[1]))
    from candidates import register
    additions = register([], [('Aloe vera', 5), ('aloe vera', 4)])
    assert len(additions) == 1
