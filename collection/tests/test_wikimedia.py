"""The Wikimedia Commons connector.

The responses are those of the real API, cut down: one species category and
the metadata of its files. What matters here is not the number of candidates
returned but what is **discarded** — an audio file, a botanical drawing, a
non-commercial licence — and the fact that a network failure never disguises
itself as "this species has no images".
"""
import sys
from pathlib import Path

import pytest
import requests

sys.path.insert(0, str(Path(__file__).resolve().parents[1]))

from plant_dataset.fetchers.wikimedia import (  # noqa: E402
    CommonsClient, category_names, _plain)


def members(*titles):
    return {'batchcomplete': True,
            'query': {'categorymembers': [{'pageid': 1000 + i, 'ns': 6, 'title': t}
                                          for i, t in enumerate(titles)]}}


def one_file(pageid, title, mime='image/jpeg', licence='https://creativecommons.org/licenses/by/4.0',
             short='CC BY 4.0', author='<a href="//commons.wikimedia.org/wiki/User:X">Amanda Grobe</a>'):
    return {
        'pageid': pageid, 'title': title,
        'imageinfo': [{
            'mime': mime,
            'url': f'https://upload.wikimedia.org/{title}',
            'thumburl': f'https://upload.wikimedia.org/thumb/{title}/1280px.jpg',
            'descriptionurl': f'https://commons.wikimedia.org/wiki/{title}',
            'extmetadata': {
                'LicenseUrl': {'value': licence} if licence else {},
                'LicenseShortName': {'value': short},
                'Artist': {'value': author},
            },
        }],
    }


class FakeCommons(CommonsClient):
    """A client answering from a script, without a network."""

    def __init__(self, categories=None, files=None, error=None):
        super().__init__(session=requests.Session(), pause=0)
        self.categories = categories or {}
        self.files = files or {}
        self.error = error
        self.calls = 0

    def _get(self, **params):
        self.calls += 1
        if self.error:
            raise self.error
        if params.get('list') == 'categorymembers':
            cat = params['cmtitle'].split(':', 1)[1]
            return self.categories.get((cat, params['cmtype']), members())
        pages = [self.files[t] for t in params['titles'].split('|') if t in self.files]
        return {'query': {'pages': pages}}


def test_the_hybrid_sign_variants_are_tried():
    """Commons writes "Citrus × limon", "Citrus x limon" or "Citrus limon"
    depending on the category. The hybrid sign had already cost every
    lemon-tree photograph of one version, on the iNaturalist side."""
    assert category_names('Citrus × limon') == ['Citrus × limon', 'Citrus x limon', 'Citrus limon']
    assert category_names('Monstera deliciosa') == ['Monstera deliciosa']
    assert category_names('  ') == []


def test_the_author_is_a_name_not_html():
    assert _plain('<a href="/wiki/User:X">Amanda Grobe</a>') == 'Amanda Grobe'
    assert _plain('Forest &amp; Kim Starr') == 'Forest & Kim Starr'
    assert _plain(None) == ''


def test_only_photographs_under_an_accepted_licence_come_out():
    c = FakeCommons(
        categories={('Pilea peperomioides', 'file'): members(
            'File:Pilea in a pot.jpg',                # kept
            'File:Pilea pronunciation.ogg',           # audio: wrong type
            'File:Pilea botanical illustration.jpg',  # drawing: discarded on the title
            'File:Pilea leaf NC.jpg',                 # non-commercial licence
        ), ('Pilea peperomioides', 'subcat'): members()},
        files={
            'File:Pilea in a pot.jpg': one_file(1, 'File:Pilea in a pot.jpg'),
            'File:Pilea pronunciation.ogg': one_file(2, 'File:Pilea pronunciation.ogg', mime='audio/ogg'),
            'File:Pilea leaf NC.jpg': one_file(4, 'File:Pilea leaf NC.jpg',
                                               licence='https://creativecommons.org/licenses/by-nc/4.0',
                                               short='CC BY-NC 4.0'),
        })
    out = list(c.image_candidates('Pilea peperomioides'))
    assert [x.source_id for x in out] == ['commons:1']
    only = out[0]
    assert only.source == 'wikimedia'
    assert only.author == 'Amanda Grobe'
    assert only.image_url.startswith('https://upload.wikimedia.org/thumb/')
    assert only.license_raw == 'https://creativecommons.org/licenses/by/4.0'


def test_share_alike_follows_the_flag():
    script = dict(
        categories={('Sedum', 'file'): members('File:Sedum SA.jpg'), ('Sedum', 'subcat'): members()},
        files={'File:Sedum SA.jpg': one_file(9, 'File:Sedum SA.jpg',
                                             licence='https://creativecommons.org/licenses/by-sa/4.0',
                                             short='CC BY-SA 4.0')})
    assert list(FakeCommons(**script).image_candidates('Sedum')) == []
    with_sa = list(FakeCommons(**script).image_candidates('Sedum', allow_share_alike=True))
    assert len(with_sa) == 1


def test_subcategories_are_visited():
    """"Monstera deliciosa (cultivars)", "... in Brazil": that is often where
    the photographs of cultivated plants are."""
    c = FakeCommons(
        categories={
            ('Monstera deliciosa', 'file'): members('File:A.jpg'),
            ('Monstera deliciosa', 'subcat'): members('Category:Monstera deliciosa (cultivars)'),
            ('Monstera deliciosa (cultivars)', 'file'): members('File:B.jpg'),
        },
        files={'File:A.jpg': one_file(1, 'File:A.jpg'), 'File:B.jpg': one_file(2, 'File:B.jpg')})
    assert sorted(x.source_id for x in c.image_candidates('Monstera deliciosa')) == ['commons:1', 'commons:2']


def test_a_network_failure_does_not_disguise_itself_as_a_species_without_images():
    """The bug found on the first real run: Commons answered 429, the connector
    swallowed the error, and the species was announced at zero images when it
    had seventy. A source that falls over must be visible — `build_dataset`
    writes it as `FAILED` and the resume catches it."""
    c = FakeCommons(error=requests.ConnectionError('outage'))
    with pytest.raises(requests.RequestException):
        list(c.image_candidates('Pilea peperomioides'))


def test_a_species_without_a_category_simply_returns_nothing():
    """To be distinguished from the previous case: here Commons answers, there
    simply is no category. That is not an error."""
    c = FakeCommons(categories={})
    assert list(c.image_candidates('Cymbidium hybridum')) == []


# --- The subcategory order decides the visual domain of the photographs ----
#
# Commons sorts the photographs of a species by context: `(potted)` holds
# potted plants, `(products)` jars of jam, `- botanical illustrations`
# nineteenth-century engravings. Taking the six first returned by the API
# amounted to drawing lots — and 73 % of the model's errors come from never
# having seen the plant as it is grown.

from plant_dataset.fetchers.wikimedia import rank_subcategories  # noqa: E402


def test_potted_plants_come_first():
    names = ['Monstera deliciosa (flowers)', 'Monstera deliciosa (potted)',
             'Monstera deliciosa (fruit)']
    assert rank_subcategories(names, 3)[0] == 'Monstera deliciosa (potted)'


def test_plates_and_herbaria_are_never_visited():
    names = ['Ficus elastica - botanical illustrations', 'Ficus elastica (herbarium specimens)',
             'Ficus elastica (potted)']
    assert rank_subcategories(names, 6) == ['Ficus elastica (potted)']


def test_derived_products_neither():
    # "Monstera deliciosa (products)": jams, not plants.
    assert rank_subcategories(['Monstera deliciosa (products)'], 6) == []


def test_the_priority_order_is_respected():
    names = ['X (garden)', 'X (cultivars)', 'X (in pots)']
    assert rank_subcategories(names, 3) == ['X (in pots)', 'X (cultivars)', 'X (garden)']


def test_neutral_categories_follow_without_being_discarded():
    names = ['X (leaves)', 'X (potted)']
    assert rank_subcategories(names, 2) == ['X (potted)', 'X (leaves)']


def test_the_cap_is_respected():
    names = [f'X (potted {i})' for i in range(10)]
    assert len(rank_subcategories(names, 4)) == 4
