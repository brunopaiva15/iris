"""Wikimedia Commons: plants as people actually grow them.

GBIF and iNaturalist describe field observations. Commons is a media library:
people photograph their monstera in the living room, a pelargonium on a
balcony, a ficus at a florist. That is exactly the distribution the model
lacks — the living-room yucca taken for maize and the unreadable ginseng ficus
come from that gap.

Two measurements made before writing this file, on our own species:

- **97 % of the files carry a usable licence**, against 18 % on GBIF, where
  non-commercial licences swamp everything. On Commons they are CC BY-SA 4.0,
  CC BY 4.0, CC BY-SA 3.0, CC0 and public domain;
- a species category holds on the order of a hundred files, more with its
  subcategories. It is a **complement** to GBIF, not a replacement: the
  project's target is 200 to 300 images per species.

Commons has no notion of an observation: every file is independent. The page
identifier therefore serves as the source identifier, and the split group falls
back to the file itself. Two photographs of the same plant taken by the same
person will not be recognised as a group — an accepted limit; deduplication by
perceptual hash catches the near-duplicates.

Taxonomy does not move: Commons is an image source, GBIF stays the reference
for names. Every image keeps its provenance and its licence, and attaches to
the species through `species` and `internal_plant_id`, like the others.
"""
from __future__ import annotations

import re
import time
from typing import Iterator

import requests

from ..licenses import is_allowed, parse_license
from . import ImageCandidate

API = 'https://commons.wikimedia.org/w/api.php'
DEFAULT_UA = 'IrisPlantDataset/0.1 (github.com/brunopaiva15/iris; dataset builder)'
RETRIES = 5
PAGE = 100

#: Width requested from Commons. The originals reach tens of megabytes; the
#: pipeline reduces them anyway. Asking for a large thumbnail spares bandwidth
#: on both sides.
THUMB_WIDTH = 1280

#: Accepted formats. Commons also hosts SVG, PDF and TIFF files, which the
#: pipeline would not know how to read.
PHOTO_TYPES = {'image/jpeg', 'image/png', 'image/webp'}

#: A species category does not hold photographs alone: nineteenth-century
#: botanical plates, herbarium scans, distribution maps, anatomical diagrams.
#: The model must recognise a living plant; these images are noise, and the
#: pipeline cannot tell them from a photograph. A crude filter on the title,
#: owned as such: it lets unnamed drawings through and may discard a real
#: photograph with a bad title.
NOT_A_PHOTO = re.compile(
    r'\b(illustration|drawing|dessin|zeichnung|engraving|gravure|lithograph|'
    r'botanical\s+plate|planche|k[öo]hler|flora\s+von|flora\s+of|'
    r'herbarium|herbier|specimen|holotype|isotype|lectotype|type\s+sheet|'
    r'distribution\s+map|carte|diagram|schema|logo|icon|stamp|timbre|'
    r'coat\s+of\s+arms|chromolith)\b', re.I)

_HYBRID = re.compile(r'\s*×\s*')

#: Subcategories to visit first, in this order. Commons sorts the photographs
#: of a species by context, and not every context is worth the same to us:
#: `Monstera deliciosa (potted)` holds 41 files of potted plants, that is
#: **exactly** what the application receives and what the model lacks (73 % of
#: the errors cross the family, for want of having seen the plant as it is
#: grown).
#:
#: Without this order, the six subcategories kept were the six first returned
#: by the API — `(potted)` passed or did not at random, and `(products)` took
#: its place.
PREFERRED_SUBCATS = ('potted', 'in pots', 'indoor', 'houseplant', 'cultivars',
                     'cultivated', 'in gardens', 'garden')

#: Subcategories not to visit at all: they are not photographs of the living
#: plant. `NOT_A_PHOTO` already catches them by file name, but discarding them
#: at category level avoids spending slots there.
REFUSED_SUBCATS = ('illustration', 'herbarium', 'specimen', 'products', 'stamps',
                   'coins', 'maps', 'diagram', 'seeds', 'wood', 'timber')


def rank_subcategories(names: list[str], how_many: int) -> list[str]:
    """The subcategories worth the detour, the best ones first.

    Returns at most `how_many` names: the preferred ones in the order of
    `PREFERRED_SUBCATS`, then the others as they come, never the refused ones.
    """
    kept = [n for n in names if not any(r in n.lower() for r in REFUSED_SUBCATS)]

    def rank(name: str) -> int:
        low = name.lower()
        for i, word in enumerate(PREFERRED_SUBCATS):
            if word in low:
                return i
        return len(PREFERRED_SUBCATS)

    return sorted(kept, key=rank)[:how_many]
_TAGS = re.compile(r'<[^>]+>')


def _plain(html: str) -> str:
    """The author arrives as HTML (a link to the user page, most of the time).
    Attribution wants a name, not a fragment of a page."""
    text = _TAGS.sub(' ', html or '')
    text = (text.replace('&amp;', '&').replace('&lt;', '<').replace('&gt;', '>')
                .replace('&quot;', '"').replace('&#039;', "'").replace('&nbsp;', ' '))
    return ' '.join(text.split())[:200]


def category_names(scientific_name: str) -> list[str]:
    """The category titles to try for a species, from most to least likely.

    Commons names its categories after the scientific name, but the hybrid sign
    is written there sometimes as "×", sometimes as "x", sometimes not at all.
    The lemon tree had already cost a whole version over that detail on the
    iNaturalist side.
    """
    name = ' '.join((scientific_name or '').split())
    if not name:
        return []
    variants = [name, _HYBRID.sub(' x ', name), _HYBRID.sub(' ', name)]
    seen, out = set(), []
    for v in variants:
        v = ' '.join(v.split())
        if v and v.lower() not in seen:
            seen.add(v.lower())
            out.append(v)
    return out


class CommonsClient:
    def __init__(self, session: requests.Session | None = None, user_agent: str = DEFAULT_UA,
                 pause: float = 1.0, timeout: float = 60.0):
        self.session = session or requests.Session()
        self.session.headers['User-Agent'] = user_agent
        self.pause = pause
        self.timeout = timeout

    def _get(self, **params) -> dict:
        params.update(action='query', format='json', formatversion='2')
        for attempt in range(RETRIES):
            try:
                r = self.session.get(API, params=params, timeout=self.timeout)
                if r.status_code == 429:
                    # Commons does not always say how long to wait; failing
                    # that, back off firmly rather than insist.
                    time.sleep(float(r.headers.get('Retry-After') or min(15 * (attempt + 1), 60)))
                    raise requests.HTTPError('429', response=r)
                if r.status_code >= 500:
                    raise requests.HTTPError(str(r.status_code), response=r)
                r.raise_for_status()
                data = r.json()
                time.sleep(self.pause)
                return data
            except (requests.ConnectionError, requests.Timeout, requests.HTTPError,
                    requests.exceptions.JSONDecodeError):
                if attempt == RETRIES - 1:
                    raise
                time.sleep(min(2 ** attempt, 30))
        raise RuntimeError('unreachable')

    def _members(self, category: str, kind: str, limit: int) -> list[dict]:
        """The members of a category, files (`file`) or subcategories
        (`subcat`), on a single page — no point paginating a thousand of them
        for a complement."""
        data = self._get(list='categorymembers', cmtitle=f'Category:{category}',
                         cmtype=kind, cmlimit=str(min(limit, 500)))
        return (data.get('query') or {}).get('categorymembers', []) or []

    def _files_info(self, titles: list[str]) -> list[dict]:
        """Metadata of several files at once: thumbnail URL, licence, author,
        type."""
        out = []
        for i in range(0, len(titles), 50):   # the API accepts 50 per call
            data = self._get(titles='|'.join(titles[i:i + 50]), prop='imageinfo',
                             iiprop='url|extmetadata|mime', iiurlwidth=str(THUMB_WIDTH))
            out.extend((data.get('query') or {}).get('pages', []) or [])
        return out

    def category_for(self, scientific_name: str) -> str | None:
        """The first category that exists and holds files.

        Network errors are **not** swallowed: a rate limit would make the
        category unfindable, hence the species empty, and the log would calmly
        announce "0 images" for a plant that has a hundred. `build_dataset`
        already catches exceptions species by species and writes them as
        `FAILED` — that is where it shows, and it can be recovered.
        """
        for name in category_names(scientific_name):
            if self._members(name, 'file', 1):
                return name
        return None

    def image_candidates(self, scientific_name: str, max_files: int = 300,
                         allow_share_alike: bool = False, subcategories: int = 6) -> Iterator[ImageCandidate]:
        """The photographs of a species, acceptable licences only.

        One level of subcategories is visited: Commons sorts "Monstera
        deliciosa in <place>", "... flowers", "... leaves" there, and that is
        often where the photographs of cultivated plants are.
        """
        root = self.category_for(scientific_name)
        if root is None:
            return
        categories = [root]
        if subcategories:
            # Ask broadly and sort: potted plants first, botanical plates
            # never. Taking the six first returned by the API amounted to
            # drawing the context of the photographs at random.
            everything = [c['title'].split(':', 1)[-1] for c in self._members(root, 'subcat', 50)]
            categories += rank_subcategories(everything, subcategories)

        titles, seen = [], set()
        for cat in categories:
            if len(titles) >= max_files:
                break
            members = self._members(cat, 'file', max_files - len(titles))
            for m in members:
                title = m.get('title', '')
                if title and title not in seen and not NOT_A_PHOTO.search(title):
                    seen.add(title)
                    titles.append(title)

        for page in self._files_info(titles):
            info = (page.get('imageinfo') or [{}])[0]
            if info.get('mime') not in PHOTO_TYPES:
                continue
            meta = info.get('extmetadata') or {}
            # The licence URL is safer than the label: "CC BY 3.0 us" or
            # "CC-BY 4.0 Int" read badly, the URL never does.
            raw = ((meta.get('LicenseUrl') or {}).get('value')
                   or (meta.get('LicenseShortName') or {}).get('value') or '')
            if not is_allowed(parse_license(raw), allow_share_alike=allow_share_alike):
                continue
            url = info.get('thumburl') or info.get('url')
            if not url:
                continue
            yield ImageCandidate(
                source='wikimedia',
                source_id=f'commons:{page.get("pageid")}',
                # Commons has no observation: the file is its own group. Two
                # photographs of the same plant will therefore not be kept
                # together at split time; deduplication by perceptual hash
                # stays fully effective.
                observation_id=f'commons:{page.get("pageid")}',
                original_url=info.get('descriptionurl') or f'https://commons.wikimedia.org/wiki/{page.get("title", "")}',
                image_url=url,
                author=_plain((meta.get('Artist') or {}).get('value', '')),
                license_raw=raw,
                publisher='Wikimedia Commons',
                extra={'title': page.get('title', ''), 'category': root},
            )
