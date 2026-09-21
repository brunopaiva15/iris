"""Scientific names: normalisation, and the reference plant list.

The canonical name is "Genus epithet", without author or year, with the hybrid
sign kept ("Citrus × aurantium") and infraspecific ranks kept abbreviated
("Ficus benjamina var. nuda"). It is the key joining the app's catalogue, the
image sources and the model's classes.
"""
from __future__ import annotations

import csv
import re
import unicodedata
from dataclasses import dataclass, field
from pathlib import Path

_RANKS = {'subsp', 'ssp', 'var', 'f', 'forma', 'cv', 'subvar'}
_HYBRID = '×'


def _fold(s: str) -> str:
    s = unicodedata.normalize('NFKD', s)
    return ''.join(c for c in s if not unicodedata.combining(c))


def normalize_scientific_name(raw: str) -> str:
    """Reduce a name to its canonical form.

    >>> normalize_scientific_name('Monstera deliciosa Liebm.')
    'Monstera deliciosa'
    >>> normalize_scientific_name('citrus x aurantium L.')
    'Citrus × aurantium'
    >>> normalize_scientific_name('Ficus benjamina var. nuda (Miq.) Barrett')
    'Ficus benjamina var. nuda'
    >>> normalize_scientific_name('× Fatshedera lizei')
    '× Fatshedera lizei'
    """
    s = _fold(raw or '').replace('_', ' ').strip()
    # The hybrid sign is sometimes stuck to the epithet ("Citrus ×sinensis",
    # in GBIF as in the floras). Without separating it, the name gives a class
    # distinct from "Citrus × sinensis": the same plant learned twice, under
    # two labels.
    s = re.sub(r'×(?=\S)', '× ', s)
    s = re.sub(r'\s+', ' ', s)
    if not s:
        return ''
    words = s.split(' ')
    out: list[str] = []
    expecting_epithet = False
    genus_taken = False
    for i, w in enumerate(words):
        token = w.strip(',;')
        if not token:
            continue
        # An intergeneric hybrid — "× Fatshedera lizei" — opens on the sign:
        # that is not the genus, the next word is. Without this distinction,
        # the sign passed for the genus and the capital of "Fatshedera" read as
        # an author name, which cut the name there and left it empty.
        if not genus_taken:
            if not out and token in ('x', 'X', _HYBRID):
                out.append(_HYBRID)
                continue
            out.append(token[:1].upper() + token[1:].lower())
            genus_taken = True
            expecting_epithet = True
            continue
        low = token.lower().rstrip('.')
        if token in ('x', 'X', _HYBRID) and expecting_epithet:
            out.append(_HYBRID)
            continue
        if low in _RANKS:
            out.append(low + '.')
            expecting_epithet = True
            continue
        if token[0] in ("'", '"', '‘', '“'):
            # A cultivar keeps its capitals: Rosa 'Peace'.
            out.append("'" + token.strip('\'"‘’“”') + "'")
            expecting_epithet = False
            continue
        if expecting_epithet:
            # The epithet is lower case; anything starting with a capital or
            # a parenthesis, or containing a digit, is an author.
            shouting = token.isalpha() and token.isupper() and len(token) >= 3  # "DELICIOSA": a shouted name, not an author
            if (token[0].isupper() and not shouting) or token[0] == '(' or any(ch.isdigit() for ch in token):
                break
            out.append(low)
            expecting_epithet = False
            continue
        # After the epithet: either a rank (handled above) or the author.
        break
    # A hybrid with no epithet after the sign makes no sense: drop it.
    if out and out[-1] == _HYBRID:
        out.pop()
    return ' '.join(out)


def species_slug(canonical: str) -> str:
    """"Monstera deliciosa" → "Monstera_deliciosa", for a folder name."""
    s = _fold(canonical).replace(_HYBRID, 'x')
    s = re.sub(r'[^A-Za-z0-9]+', '_', s).strip('_')
    return s


def internal_id(name: str) -> str:
    """Stable internal identifier: the canonical name, lower case, hyphenated.

    The name is normalised first, as the application's own implementation does:
    both must return the same key for the same plant, including when handed a
    raw name.
    """
    return species_slug(normalize_scientific_name(name)).lower().replace('_', '-')


def genus_of(canonical: str) -> str:
    return canonical.split(' ')[0] if canonical else ''


def epithet_of(canonical: str) -> str:
    parts = canonical.split(' ')
    if len(parts) < 2:
        return ''
    return parts[2] if parts[1] == _HYBRID and len(parts) > 2 else parts[1]


@dataclass
class PlantEntry:
    """One plant of the reference catalogue."""

    internal_id: str
    scientific_name: str
    genus: str
    epithet: str
    family: str
    common_names: dict[str, str] = field(default_factory=dict)
    synonyms: list[str] = field(default_factory=list)
    gbif_key: int | None = None
    wikidata_id: str | None = None
    plantnet_id: str | None = None
    image: str | None = None

    @classmethod
    def from_name(cls, name: str, family: str = '', **common) -> 'PlantEntry':
        canonical = normalize_scientific_name(name)
        return cls(
            internal_id=internal_id(canonical),
            scientific_name=canonical,
            genus=genus_of(canonical),
            epithet=epithet_of(canonical),
            family=family,
            common_names={k: v for k, v in common.items() if v},
        )


COLUMNS = ['internal_id', 'scientific_name', 'genus', 'epithet', 'family', 'common_fr', 'common_en', 'common_de', 'common_it',
           'synonyms', 'gbif_key', 'wikidata_id', 'plantnet_id', 'image']


def load_plants(path: str | Path) -> list[PlantEntry]:
    """Read `plants.csv`. The names are renormalised as they are read: the
    list stays correct even if someone pastes a name with its author into
    it."""
    entries: list[PlantEntry] = []
    with open(path, newline='', encoding='utf-8') as f:
        for row in csv.DictReader(f):
            canonical = normalize_scientific_name(row['scientific_name'])
            if not canonical:
                continue
            common = {lang: row.get(f'common_{lang}', '') for lang in ('fr', 'en', 'de', 'it')}
            entries.append(PlantEntry(
                internal_id=row.get('internal_id') or internal_id(canonical),
                scientific_name=canonical,
                genus=genus_of(canonical),
                epithet=epithet_of(canonical),
                family=row.get('family', ''),
                common_names={k: v for k, v in common.items() if v},
                synonyms=[s.strip() for s in (row.get('synonyms') or '').split('|') if s.strip()],
                gbif_key=int(row['gbif_key']) if row.get('gbif_key') else None,
                wikidata_id=row.get('wikidata_id') or None,
                plantnet_id=row.get('plantnet_id') or None,
                image=row.get('image') or None,
            ))
    return entries


def save_plants(path: str | Path, entries: list[PlantEntry]) -> None:
    with open(path, 'w', newline='', encoding='utf-8') as f:
        w = csv.DictWriter(f, fieldnames=COLUMNS)
        w.writeheader()
        for e in entries:
            w.writerow({
                'internal_id': e.internal_id,
                'scientific_name': e.scientific_name,
                'genus': e.genus,
                'epithet': e.epithet,
                'family': e.family,
                'common_fr': e.common_names.get('fr', ''),
                'common_en': e.common_names.get('en', ''),
                'common_de': e.common_names.get('de', ''),
                'common_it': e.common_names.get('it', ''),
                'synonyms': '|'.join(e.synonyms),
                'gbif_key': e.gbif_key or '',
                'wikidata_id': e.wikidata_id or '',
                'plantnet_id': e.plantnet_id or '',
                'image': e.image or '',
            })


def match_to_catalog(name: str, entries: list[PlantEntry]) -> PlantEntry | None:
    """Find a catalogue plant from a name coming from elsewhere (a remote
    identification service, GBIF, PlantNet-300K): canonical name first,
    synonyms next."""
    canonical = normalize_scientific_name(name)
    if not canonical:
        return None
    by_name = {e.scientific_name: e for e in entries}
    if canonical in by_name:
        return by_name[canonical]
    for e in entries:
        if canonical in (normalize_scientific_name(s) for s in e.synonyms):
            return e
    return None
