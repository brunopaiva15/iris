"""iNaturalist directly: the "casual" observations too.

GBIF only receives iNaturalist's "research" grade observations, that is wild
plants identified by several people. An indoor plant in a pot is marked
"captive/cultivated", stays "casual", and never reaches GBIF. Yet for the
application's use — pots on a windowsill — those are the best photographs.

The API asks for a User-Agent, at most about one request per second, and
returns the licence of every photograph. The `photo_license` filter on the
server side, then each photograph's licence on the client side, as for GBIF.
"""
from __future__ import annotations

import re
import time
from dataclasses import dataclass
from typing import Iterator

import requests

from ..licenses import is_allowed, parse_license
from . import ImageCandidate

API = 'https://api.inaturalist.org/v1'
DEFAULT_UA = 'IrisPlantDataset/0.1 (github.com/brunopaiva15/iris; dataset builder)'
PAGE = 200
RETRIES = 6
PHOTO_LICENSES = 'cc0,cc-by'
PHOTO_LICENSES_WITH_SA = 'cc0,cc-by,cc-by-sa'
_PHOTO_ID = re.compile(r'/photos/(\d+)/')


SPECIES_RANKS = {'species', 'subspecies', 'variety', 'form', 'hybrid'}


@dataclass
class InatTaxon:
    id: int
    name: str
    rank: str
    observations: int

    #: The requested name, when it differs from the accepted one (a synonym).
    matched_as: str = ''

    @property
    def is_synonym(self) -> bool:
        return bool(self.matched_as) and self.matched_as.lower() != self.name.lower()


_HYBRID = re.compile(r'\s*(?:×|\bx\b)\s*')


def _comparable(name: str) -> str:
    """The name as it is compared: lower case, whitespace squeezed, and
    without the hybrid sign. The lemon tree is "Citrus limon" in the catalogue
    and "Citrus × limon" on iNaturalist: the same plant, and one version
    shipped without a single lemon-tree photograph because of that sign."""
    return ' '.join(_HYBRID.sub(' ', (name or '').strip().lower()).split())


def photo_id_of(url: str) -> str | None:
    """The iNaturalist identifier of a photograph, from any of its URLs
    (`.../photos/726492519/square.jpg`). The same one for GBIF, which relays
    these URLs as they are: it is the deduplication key across sources."""
    m = _PHOTO_ID.search(url or '')
    return m.group(1) if m else None


class InatClient:
    def __init__(self, session: requests.Session | None = None, user_agent: str = DEFAULT_UA, pause: float = 1.0, timeout: float = 60.0):
        self.session = session or requests.Session()
        self.session.headers['User-Agent'] = user_agent
        self.pause = pause
        self.timeout = timeout

    def _get(self, path: str, **params) -> dict:
        for attempt in range(RETRIES):
            try:
                r = self.session.get(f'{API}{path}', params=params, timeout=self.timeout)
                if r.status_code == 429 or r.status_code >= 500:
                    raise requests.HTTPError(f'{r.status_code}', response=r)
                r.raise_for_status()
                time.sleep(self.pause)
                return r.json()
            except (requests.ConnectionError, requests.Timeout, requests.HTTPError):
                if attempt == RETRIES - 1:
                    raise
                time.sleep(min(3 * 2 ** attempt, 45))
        raise RuntimeError('unreachable')

    def match(self, name: str) -> InatTaxon | None:
        """The taxon matching the requested name, at species rank or below.

        Many houseplants are sold under a name taxonomy has since abandoned:
        "Schefflera arboricola" became "Heptapleurum arboricola", "Saintpaulia
        ionantha" became "Streptocarpus ionanthus". iNaturalist knows these
        synonyms and answers with the accepted name, stating in `matched_term`
        the name that found it. Refusing those answers would mean going without
        the most common species of the catalogue.

        A synonym is accepted only if `matched_term` is exactly the requested
        name — never on a mere resemblance — and the rank is a species one.
        "Alocasia amazonica" brings back the *genus* Alocasia: that is not an
        answer, it is an admission of ignorance.
        """
        d = self._get('/taxa', q=name, per_page=10)
        wanted = _comparable(name)
        fallback = None
        for r in d.get('results', []):
            if not r.get('is_active', True) or (r.get('rank') or '') not in SPECIES_RANKS:
                continue
            taxon = InatTaxon(id=r['id'], name=r.get('name', ''), rank=r.get('rank', ''),
                              observations=int(r.get('observations_count') or 0), matched_as=name.strip())
            if _comparable(taxon.name) == wanted:
                return taxon
            if fallback is None and _comparable(r.get('matched_term') or '') == wanted:
                fallback = taxon
        return fallback

    def image_candidates(self, taxon_id: int, max_observations: int = 2000, allow_share_alike: bool = False,
                         captive: bool | None = None, place_id: int | None = None) -> Iterator[ImageCandidate]:
        """The photographs of a taxon, filtered on their licence.

        `captive=True` returns only the observations of cultivated plants. But
        "cultivated" on iNaturalist means planted by people, not "in a pot in a
        living room": cultivated Yucca gigantea are mostly garden trees in
        California against a blue sky, and a model trained on those still did
        not recognise the cane yucca of a living room. `place_id` restricts to
        a region: in Europe (97391), an observed yucca, monstera or ficus is
        nearly always a houseplant on a windowsill — what users photograph.
        """
        licenses = PHOTO_LICENSES_WITH_SA if allow_share_alike else PHOTO_LICENSES
        extra = {'captive': 'true'} if captive else {}
        if place_id is not None:
            extra['place_id'] = place_id
        page = 1
        seen = 0
        while seen < max_observations:
            d = self._get('/observations', taxon_id=taxon_id, photo_license=licenses, quality_grade='any', photos='true',
                          per_page=PAGE, page=page, order_by='id', order='asc', **extra)
            results = d.get('results', [])
            if not results:
                break
            for obs in results:
                seen += 1
                for c in candidates_from_observation(obs, allow_share_alike=allow_share_alike):
                    if place_id is not None:
                        c.extra['place_id'] = place_id
                    yield c
            if len(results) < PAGE:
                break
            page += 1


def candidates_from_observation(obs: dict, allow_share_alike: bool = False) -> Iterator[ImageCandidate]:
    """The photographs of one observation, filtered on their own licence. A
    photograph hidden by moderation is ignored."""
    key = str(obs.get('id', ''))
    user = obs.get('user') or {}
    author = user.get('name') or user.get('login') or ''
    page = obs.get('uri') or (f'https://www.inaturalist.org/observations/{key}' if key else '')
    for p in obs.get('photos') or []:
        if p.get('hidden'):
            continue
        raw = p.get('license_code') or ''
        if not is_allowed(parse_license(raw), allow_share_alike=allow_share_alike):
            continue
        url = p.get('url') or ''
        if not url.startswith('http'):
            continue
        # `square` is a 75 px thumbnail; `large` is 1024 px on the long side,
        # which is what the pipeline keeps anyway.
        url = url.replace('/square.', '/large.')
        dims = p.get('original_dimensions') or {}
        yield ImageCandidate(
            source='inaturalist',
            source_id=f'{key}#{p.get("id")}',
            observation_id=key,
            original_url=page,
            image_url=url,
            author=author,
            license_raw=raw,
            publisher='iNaturalist',
            dataset_key='',
            extra={'photo_id': str(p.get('id') or photo_id_of(url) or ''), 'quality_grade': obs.get('quality_grade', ''),
                   'captive': bool(obs.get('captive')), 'original_width': dims.get('width'), 'original_height': dims.get('height')},
        )
