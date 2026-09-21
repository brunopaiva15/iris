import json
from pathlib import Path

from plant_dataset.fetchers.gbif import INATURALIST_DATASET, GbifClient, TaxonMatch, candidates_from_occurrence

FIXTURES = Path(__file__).parent / 'fixtures'


def load(name):
    return json.loads((FIXTURES / name).read_text())


class FakeSession:
    """Replay the recorded answers: no network call in the tests."""

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
    return GbifClient(session=FakeSession(responses), pause=0)


def test_match_exact_species_is_usable():
    c = client({'/species/match': load('gbif_match_monstera.json')})
    m = c.match('Monstera deliciosa')
    assert isinstance(m, TaxonMatch)
    assert m.key == 2868241
    assert m.canonical_name == 'Monstera deliciosa'
    assert m.family == 'Araceae'
    assert m.usable
    assert c.session.headers['User-Agent'].startswith('IrisPlantDataset')


def test_match_to_higher_rank_is_not_usable():
    c = client({'/species/match': load('gbif_match_none.json')})
    m = c.match('Plantus imaginarius')
    # GBIF answers "Plantae" (kingdom): that is not a species, we do not want it.
    assert m is not None and m.rank == 'KINGDOM'
    assert not m.usable


def test_match_none_returns_none():
    c = client({'/species/match': {'matchType': 'NONE', 'confidence': 100, 'synonym': False}})
    assert c.match('zzzz') is None


def test_fuzzy_match_needs_high_confidence():
    base = dict(key=1, canonical_name='X y', scientific_name='X y', rank='SPECIES', status='ACCEPTED', family='', genus='X', accepted_key=None)
    assert TaxonMatch(match_type='FUZZY', confidence=97, **base).usable
    assert not TaxonMatch(match_type='FUZZY', confidence=80, **base).usable
    assert not TaxonMatch(match_type='HIGHERRANK', confidence=100, **base).usable


def test_candidates_keep_only_allowed_media_licences():
    page = load('gbif_occurrence_page.json')
    cands = [c for occ in page['results'] for c in candidates_from_occurrence(occ)]
    # 3 CC BY occurrences; 3 + 2 CC BY media, 1 CC BY-NC medium refused.
    assert len(cands) == 5
    assert {c.observation_id for c in cands} == {'6130686462', '6130914712'}
    assert all(c.source == 'gbif' for c in cands)
    assert all(c.image_url.startswith('https://') for c in cands)
    assert all(c.author for c in cands), 'chaque image doit avoir un auteur pour l\'attribution'
    assert all('creativecommons.org/licenses/by/4.0' in c.license_raw for c in cands)
    assert all(c.extra['inaturalist'] for c in cands)
    assert all(c.dataset_key == INATURALIST_DATASET for c in cands)
    first = cands[0]
    assert first.source_id == '6130686462#0'
    assert first.original_url.startswith('https://www.inaturalist.org/observations/')


def test_media_without_licence_falls_back_to_occurrence_licence():
    occ = {'key': 42, 'license': 'CC0_1_0', 'recordedBy': 'Ada',
           'media': [{'type': 'StillImage', 'identifier': 'https://x/1.jpg'},
                     {'type': 'Sound', 'identifier': 'https://x/1.mp3'},
                     {'type': 'StillImage', 'identifier': 'ftp://x/2.jpg'}]}
    cands = list(candidates_from_occurrence(occ))
    assert len(cands) == 1
    assert cands[0].author == 'Ada' and cands[0].license_raw == 'CC0_1_0'
    assert cands[0].original_url == 'https://www.gbif.org/occurrence/42'


def test_no_licence_anywhere_yields_nothing():
    occ = {'key': 43, 'media': [{'type': 'StillImage', 'identifier': 'https://x/1.jpg'}]}
    assert list(candidates_from_occurrence(occ)) == []


def test_share_alike_media_only_when_allowed():
    occ = {'key': 44, 'license': 'CC_BY_SA_4_0', 'media': [{'type': 'StillImage', 'identifier': 'https://x/1.jpg'}]}
    assert list(candidates_from_occurrence(occ)) == []
    assert len(list(candidates_from_occurrence(occ, allow_share_alike=True))) == 1


def test_image_candidates_pages_and_filters_by_licence_code():
    page = load('gbif_occurrence_page.json')
    page = dict(page, endOfRecords=True)
    c = client({'/occurrence/search': page})
    cands = list(c.image_candidates(2868241, license_codes=['CC_BY_4_0']))
    assert len(cands) == 5
    (_, params), = c.session.calls
    assert params['license'] == 'CC_BY_4_0' and params['mediaType'] == 'StillImage' and params['taxonKey'] == 2868241


def test_transient_errors_are_retried_then_given_up_on():
    """A network outage is crossed; a lasting failure eventually surfaces, and
    the caller is the one deciding to skip the species."""
    import requests as rq

    class Flaky:
        def __init__(self, failures):
            self.failures = failures
            self.headers = {}
            self.calls = 0

        def get(self, url, params=None, timeout=None):
            self.calls += 1
            if self.calls <= self.failures:
                raise rq.ConnectionError('Connection reset by peer')
            return FakeResponse(load('gbif_match_monstera.json'))

    import plant_dataset.fetchers.gbif as g
    pause = g.time.sleep
    g.time.sleep = lambda _: None            # no real waiting in a test
    try:
        c = GbifClient(session=Flaky(3), pause=0)
        assert c.match('Monstera deliciosa').key == 2868241
        assert c.session.calls == 4, 'three failures crossed, the fourth succeeds'

        dead = GbifClient(session=Flaky(99), pause=0)
        try:
            dead.match('Monstera deliciosa')
            assert False, 'une panne durable doit remonter'
        except rq.ConnectionError:
            pass
        assert dead.session.calls == g.RETRIES
    finally:
        g.time.sleep = pause
