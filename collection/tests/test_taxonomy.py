from plant_dataset.taxonomy import (PlantEntry, epithet_of, genus_of, internal_id, load_plants, match_to_catalog,
                                    normalize_scientific_name, save_plants, species_slug)


def test_strips_authorship():
    assert normalize_scientific_name('Monstera deliciosa Liebm.') == 'Monstera deliciosa'
    assert normalize_scientific_name('Citrus × limon (L.) Osbeck') == 'Citrus × limon'
    assert normalize_scientific_name('Ficus benjamina var. nuda (Miq.) Barrett') == 'Ficus benjamina var. nuda'


def test_hybrid_sign_and_case():
    assert normalize_scientific_name('citrus x aurantium L.') == 'Citrus × aurantium'
    assert normalize_scientific_name('MONSTERA DELICIOSA') == 'Monstera deliciosa'
    assert normalize_scientific_name('  Ficus   elastica ') == 'Ficus elastica'


def test_hybrid_sign_glued_to_the_epithet_is_separated():
    """"Citrus ×sinensis" and "Citrus × sinensis" are the same plant; letting
    them differ would make two classes of the model."""
    assert normalize_scientific_name('Citrus ×sinensis') == 'Citrus × sinensis'
    assert normalize_scientific_name('Citrus ×sinensis') == normalize_scientific_name('Citrus × sinensis')
    assert normalize_scientific_name('Mentha ×piperita L.') == 'Mentha × piperita'
    assert internal_id('Citrus ×sinensis') == internal_id('Citrus × sinensis')


def test_cultivar_keeps_its_capitals():
    assert normalize_scientific_name("Rosa 'Peace'") == "Rosa 'Peace'"


def test_genus_only_and_empty():
    assert normalize_scientific_name('Monstera') == 'Monstera'
    assert normalize_scientific_name('') == ''
    assert normalize_scientific_name('Citrus ×') == 'Citrus'


def test_slug_and_id():
    assert species_slug('Citrus × limon') == 'Citrus_x_limon'
    assert species_slug('Ficus benjamina var. nuda') == 'Ficus_benjamina_var_nuda'
    assert internal_id('Monstera deliciosa') == 'monstera-deliciosa'
    assert genus_of('Citrus × limon') == 'Citrus'
    assert epithet_of('Citrus × limon') == 'limon'
    assert epithet_of('Monstera') == ''


def test_csv_roundtrip(tmp_path):
    a = PlantEntry.from_name('Monstera deliciosa Liebm.', 'Araceae', fr='Monstera', en='Swiss cheese plant')
    a.synonyms = ['Philodendron pertusum']
    a.gbif_key = 2868241
    b = PlantEntry.from_name('Epipremnum aureum', 'Araceae', fr='Pothos')
    path = tmp_path / 'plants.csv'
    save_plants(path, [a, b])
    back = load_plants(path)
    assert [e.scientific_name for e in back] == ['Monstera deliciosa', 'Epipremnum aureum']
    assert back[0].internal_id == 'monstera-deliciosa'
    assert back[0].common_names == {'fr': 'Monstera', 'en': 'Swiss cheese plant'}
    assert back[0].synonyms == ['Philodendron pertusum']
    assert back[0].gbif_key == 2868241
    assert back[1].gbif_key is None


def test_match_to_catalog_by_name_and_synonym():
    entries = [PlantEntry.from_name('Monstera deliciosa', 'Araceae'), PlantEntry.from_name('Dracaena trifasciata', 'Asparagaceae')]
    entries[1].synonyms = ['Sansevieria trifasciata Prain']
    assert match_to_catalog('Monstera deliciosa Liebm.', entries) is entries[0]
    assert match_to_catalog('Sansevieria trifasciata', entries) is entries[1]
    assert match_to_catalog('Plantus imaginarius', entries) is None
    assert match_to_catalog('', entries) is None


def test_negative_resolutions_are_not_trusted_from_the_cache():
    """A memoised failure would hide every improvement to the search: that is
    what had masked the support for synonyms."""
    import build_dataset

    calls = []

    class Client:
        def match(self, name):
            calls.append(name)
            return None

    cache = {'Deja vue': None}
    entry = PlantEntry.from_name('Deja vue')
    assert build_dataset.resolve(Client(), entry, cache) is None
    assert calls == ['Deja vue'], 'the name must be asked again despite the cached null'

    cache_ok = {'Deja vue': {'key': 1, 'usable': True}}
    assert build_dataset.resolve(Client(), entry, cache_ok) == {'key': 1, 'usable': True}
    assert calls == ['Deja vue'], 'a successful resolution stays memoised'


def test_first_usable_falls_back_on_the_catalog_synonyms():
    from build_dataset import _first_usable
    from plant_dataset.taxonomy import PlantEntry

    class M:
        def __init__(self, usable):
            self.usable = usable

    answers = {'Sorbus aria': M(False), 'Aria edulis': M(True)}
    plant = PlantEntry(internal_id='sorbus-aria', scientific_name='Sorbus aria', genus='Sorbus', epithet='aria', family='Rosaceae', synonyms=['Aria edulis'])
    assert _first_usable(lambda n: answers.get(n), plant) is answers['Aria edulis']
    assert _first_usable(lambda n: None, plant) is None


def test_le_catalogue_de_collecte_n_a_pas_de_doublon():
    """A duplicated name sends the same species to two collection shards,
    which download twice and fight over the same folder at merge time.
    `Citrus × sinensis` y figurait deux fois."""
    from pathlib import Path
    lignes = [l.strip() for l in (Path(__file__).resolve().parents[1] / 'all_species.txt').read_text().splitlines() if l.strip()]
    doublons = {n for n in lignes if lignes.count(n) > 1}
    assert not doublons, f'noms en double dans all_species.txt : {sorted(doublons)}'


def test_deux_noms_du_catalogue_ne_partagent_pas_un_dossier():
    """A species folder comes from its name: two names giving the same folder
    would mix their images without anyone seeing it."""
    from collections import defaultdict
    from pathlib import Path
    from plant_dataset.taxonomy import species_slug
    lignes = [l.strip() for l in (Path(__file__).resolve().parents[1] / 'all_species.txt').read_text().splitlines() if l.strip()]
    par_slug = defaultdict(list)
    for nom in lignes:
        par_slug[species_slug(nom)].append(nom)
    partages = {k: v for k, v in par_slug.items() if len(v) > 1}
    assert not partages, f'shared folders: {partages}'
