"""Licences: what is accepted, and nothing else.

By default, only licences allowing commercial use without a share-alike
condition enter the set: CC0, public domain, CC BY (every version). CC BY-SA
is added on request (`--allow-sa`): a decision taken on the grounds that a
trained model is not an adaptation of the photographs — it reproduces none of
them. The photographs themselves are never redistributed, and each one stays
attributed with its licence. Everything else is refused: NC, ND, unknown
licence, all rights reserved.

A licence that cannot be read is a licence refused.
"""
from __future__ import annotations

import re
from dataclasses import dataclass

_CC_URL = re.compile(r'creativecommons\.org/(licenses|publicdomain)/([a-z\-]+)/(\d\.\d)', re.I)
_CC_CODE = re.compile(r'^cc[\s_\-]*(0|by(?:[\s_\-]*(?:nc|sa|nd))*)(?:[\s_\-]*(\d)[\s_\.\-]*(\d))?$', re.I)


@dataclass(frozen=True)
class License:
    code: str          # "CC BY 4.0", "CC0 1.0", "Public Domain Mark 1.0"
    url: str
    commercial: bool   # commercial use allowed
    share_alike: bool  # share-alike obligation
    derivatives: bool  # derivative works allowed


def _build(kind: str, version: str) -> License:
    kind = kind.lower()
    if kind in ('zero', '0'):
        return License(f'CC0 {version}', f'https://creativecommons.org/publicdomain/zero/{version}/', True, False, True)
    if kind == 'mark':
        return License(f'Public Domain Mark {version}', f'https://creativecommons.org/publicdomain/mark/{version}/', True, False, True)
    parts = kind.split('-')
    if parts[0] != 'by':
        raise ValueError(kind)
    flags = set(parts[1:])
    code = 'CC ' + '-'.join(p.upper() for p in parts) + f' {version}'
    return License(code, f'https://creativecommons.org/licenses/{kind}/{version}/', 'nc' not in flags, 'sa' in flags, 'nd' not in flags)


def parse_license(value: str | None) -> License | None:
    """Recognise a licence in its common forms; `None` when unknown.

    Forms read: Creative Commons URLs, GBIF codes (`CC_BY_4_0`, `CC0_1_0`),
    labels (`CC BY 4.0`, `cc-by-sa-4.0`, `CC0`).
    """
    if not value:
        return None
    v = value.strip()
    m = _CC_URL.search(v)
    if m:
        kind, version = m.group(2), m.group(3)
        try:
            return _build(kind, version)
        except ValueError:
            return None
    if re.search(r'creativecommons\.org/publicdomain/zero', v, re.I):
        return _build('zero', '1.0')
    m = _CC_CODE.match(v)
    if m:
        body = m.group(1).lower()
        version = f'{m.group(2)}.{m.group(3)}' if m.group(2) else ('1.0' if body == '0' else '4.0')
        kind = 'zero' if body == '0' else re.sub(r'[\s_]+', '-', body)
        try:
            return _build(kind, version)
        except ValueError:
            return None
    if v.lower() in ('public domain', 'pd', 'publicdomain', 'pdm'):
        return License('Public Domain Mark 1.0', 'https://creativecommons.org/publicdomain/mark/1.0/', True, False, True)
    return None


def is_allowed(lic: License | None, *, allow_share_alike: bool = False) -> bool:
    """The project's rule: commercial, derivatives allowed, no SA unless asked."""
    if lic is None:
        return False
    if not lic.commercial or not lic.derivatives:
        return False
    if lic.share_alike and not allow_share_alike:
        return False
    return True


# Codes of the `license=` filter of the GBIF occurrence API, in the order they
# are requested: public domain first, attribution next. GBIF only knows CC0,
# CC BY and CC BY-NC: asking it for "CC_BY_SA_4_0" makes it answer 400, and the
# whole species was skipped — which happened to 89 species during one CC BY-SA
# pass. The BY-SA share comes from iNaturalist directly; on the GBIF side, each
# medium's licence is checked one by one anyway.
GBIF_LICENSE_CODES = ['CC0_1_0', 'CC_BY_4_0']
GBIF_LICENSE_CODES_WITH_SA = GBIF_LICENSE_CODES
