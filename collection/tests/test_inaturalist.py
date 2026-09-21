import json
from pathlib import Path

from plant_dataset.fetchers.inaturalist import InatClient, InatTaxon, candidates_from_observation, photo_id_of

FIXTURES = Path(__file__).parent / 'fixtures'


def load(name):
    return json.loads((FIXTURES / name).read_text())


class FakeSession:
    def __init__(self, responses):
        self.responses = responses
        self.headers = {}
        self.calls = []

    def get(self, url, params=None, timeout=None):
        self.calls.append((url, params))
        return FakeResponse(self.responses[url.rsplit('/v1', 1)[1]])


class FakeResponse:
    status_code = 200

    def __init__(self, payload):
        self.payload = payload

    def raise_for_status(self):
        pass

    def json(self):
        return self.payload


def client(responses):
    return InatClient(session=FakeSession(responses), pause=0)


def test_match_requires_the_exact_name():
    c = client({'/taxa': load('inat_taxa_pilea.json')})
    t = c.match('Pilea peperomioides')
    assert isinstance(t, InatTaxon)
    assert t.id == 125439 and t.rank == 'species' and t.observations > 0
    assert client({'/taxa': load('inat_taxa_pilea.json')}).match('Pilea cadierei') is None


def test_candidates_from_recorded_observations():
    page = load('inat_observations_pilea.json')
    cands = [c for obs in page['results'] for c in candidates_from_observation(obs)]
    # 4 observations, 7 photographs, all CC0 or CC BY.
    assert len(cands) == 7
    assert all(c.source == 'inaturalist' for c in cands)
    assert all(c.author for c in cands), 'every photograph must have an author'
    assert all(c.license_raw in ('cc0', 'cc-by') for c in cands)
    assert all(c.original_url.startswith('https://www.inaturalist.org/observations/') for c in cands)
    # The square thumbnail is replaced by the large version.
    assert all('/large.' in c.image_url and '/square.' not in c.image_url for c in cands)
    assert all(c.extra['photo_id'] for c in cands)
    first = cands[0]
    assert first.observation_id == '13754959'
    assert first.source_id == '13754959#20276282'


def test_captive_observations_are_kept():
    """That is the whole point: GBIF does not receive potted plants."""
    page = load('inat_observations_pilea.json')
    cands = [c for obs in page['results'] for c in candidates_from_observation(obs)]
    assert any(c.extra['quality_grade'] == 'casual' for c in cands)


def test_photo_licence_filter_and_hidden_photos():
    obs = {'id': 7, 'uri': 'https://www.inaturalist.org/observations/7', 'user': {'login': 'ada'},
           'photos': [{'id': 1, 'license_code': 'cc-by-nc', 'url': 'https://x/photos/1/square.jpg'},
                      {'id': 2, 'license_code': 'cc0', 'url': 'https://x/photos/2/square.jpg', 'hidden': True},
                      {'id': 3, 'license_code': None, 'url': 'https://x/photos/3/square.jpg'},
                      {'id': 4, 'license_code': 'cc-by', 'url': 'https://x/photos/4/square.jpg'}]}
    cands = list(candidates_from_observation(obs))
    assert [c.extra['photo_id'] for c in cands] == ['4']
    assert cands[0].author == 'ada'


def test_share_alike_only_when_asked():
    obs = {'id': 8, 'user': {}, 'photos': [{'id': 9, 'license_code': 'cc-by-sa', 'url': 'https://x/photos/9/square.jpg'}]}
    assert list(candidates_from_observation(obs)) == []
    assert len(list(candidates_from_observation(obs, allow_share_alike=True))) == 1


def test_photo_id_is_the_cross_source_key():
    # GBIF relays the same iNaturalist URLs: the same photo identifier on
    # both sides, hence no duplicate between the two fetchers.
    assert photo_id_of('https://inaturalist-open-data.s3.amazonaws.com/photos/726492519/original.jpg') == '726492519'
    assert photo_id_of('https://inaturalist-open-data.s3.amazonaws.com/photos/726492519/large.jpeg') == '726492519'
    assert photo_id_of('https://example.org/image.jpg') is None
    assert photo_id_of('') is None


def test_image_candidates_asks_only_for_allowed_licences():
    page = load('inat_observations_pilea.json')
    c = client({'/observations': page})
    cands = list(c.image_candidates(125439))
    assert len(cands) == 7
    (_, params), = c.session.calls
    assert params['photo_license'] == 'cc0,cc-by'
    assert params['quality_grade'] == 'any', 'potted plants are "casual"'
    assert params['taxon_id'] == 125439


def test_synonym_is_accepted_when_inaturalist_matched_that_exact_name():
    """"Schefflera arboricola" is sold everywhere under that name;
    iNaturalist knows it as a synonym of "Heptapleurum arboricola"."""
    c = client({'/taxa': load('inat_taxa_schefflera.json')})
    t = c.match('Schefflera arboricola')
    assert t is not None
    assert t.name == 'Heptapleurum arboricola'
    assert t.rank == 'species'
    assert t.is_synonym and t.matched_as == 'Schefflera arboricola'
    assert t.observations > 1000


def test_a_genus_is_not_an_answer():
    """"Alocasia amazonica" brings back only the genus: that is not a species,
    and no image beats images of any old Alocasia."""
    c = client({'/taxa': load('inat_taxa_alocasia.json')})
    assert c.match('Alocasia amazonica') is None


def test_hybrid_sign_does_not_block_the_match():
    payload = {'results': [
        {'id': 331124, 'name': 'Citrus × limon', 'rank': 'hybrid', 'is_active': True, 'matched_term': 'Citrus × limon', 'observations_count': 17479},
        {'id': 1275720, 'name': 'Citrus × limonia', 'rank': 'hybrid', 'is_active': True, 'matched_term': 'Citrus × limonia', 'observations_count': 436},
    ]}
    c = client({'/taxa': payload})
    t = c.match('Citrus limon')
    assert t is not None and t.id == 331124 and t.rank == 'hybrid'
    assert c.match('Citrus x limon').id == 331124


def test_exact_name_wins_over_a_synonym_match():
    payload = {'results': [
        {'id': 1, 'name': 'Other species', 'rank': 'species', 'is_active': True, 'matched_term': 'Pilea peperomioides', 'observations_count': 9},
        {'id': 2, 'name': 'Pilea peperomioides', 'rank': 'species', 'is_active': True, 'matched_term': 'Pilea peperomioides', 'observations_count': 3},
    ]}
    t = client({'/taxa': payload}).match('Pilea peperomioides')
    assert t.id == 2 and not t.is_synonym


def test_inactive_taxa_are_ignored():
    payload = {'results': [{'id': 3, 'name': 'Zombie plantus', 'rank': 'species', 'is_active': False, 'matched_term': 'Zombie plantus'}]}
    assert client({'/taxa': payload}).match('Zombie plantus') is None


def test_captive_filter_is_only_sent_when_asked():
    """Potted plants are "captive" on iNaturalist; that filter is what brings
    the training set closer to users' photographs."""
    page = load('inat_observations_pilea.json')
    c = client({'/observations': page})
    list(c.image_candidates(125439, captive=True))
    (_, params), = c.session.calls
    assert params['captive'] == 'true'

    c = client({'/observations': page})
    list(c.image_candidates(125439))
    (_, params), = c.session.calls
    assert 'captive' not in params, 'unasked, every observation'


def test_place_filter_is_sent_and_recorded():
    """In Europe, an observed yucca is a houseplant: the place filter is
    passed to the API, and every image keeps a trace of it so that a resume
    knows what it already has."""
    page = load('inat_observations_pilea.json')
    c = client({'/observations': page})
    found = list(c.image_candidates(125439, captive=True, place_id=97391))
    (_, params), = c.session.calls
    assert params['captive'] == 'true'
    assert params['place_id'] == 97391
    assert found and all(f.extra['place_id'] == 97391 for f in found)

    c = client({'/observations': page})
    found = list(c.image_candidates(125439))
    assert found and all('place_id' not in f.extra for f in found)
