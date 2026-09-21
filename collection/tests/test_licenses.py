import pytest

from plant_dataset.licenses import is_allowed, parse_license


@pytest.mark.parametrize('value, code', [
    ('http://creativecommons.org/licenses/by/4.0/legalcode', 'CC BY 4.0'),
    ('https://creativecommons.org/licenses/by/4.0/', 'CC BY 4.0'),
    ('http://creativecommons.org/publicdomain/zero/1.0/', 'CC0 1.0'),
    ('https://creativecommons.org/publicdomain/mark/1.0/', 'Public Domain Mark 1.0'),
    ('CC_BY_4_0', 'CC BY 4.0'),
    ('CC0_1_0', 'CC0 1.0'),
    ('CC BY 3.0', 'CC BY 3.0'),
    ('cc-by-sa-4.0', 'CC BY-SA 4.0'),
    ('http://creativecommons.org/licenses/by-nc/4.0/', 'CC BY-NC 4.0'),
    ('CC0', 'CC0 1.0'),
    ('Public Domain', 'Public Domain Mark 1.0'),
])
def test_parse_known_forms(value, code):
    assert parse_license(value).code == code


@pytest.mark.parametrize('value', ['', None, 'All rights reserved', 'http://example.com/terms', 'CC BY-WTF 4.0', 'GPL-3.0'])
def test_unknown_is_none(value):
    assert parse_license(value) is None


def test_default_rule_accepts_only_cc0_pd_and_by():
    assert is_allowed(parse_license('CC0_1_0'))
    assert is_allowed(parse_license('CC_BY_4_0'))
    assert is_allowed(parse_license('http://creativecommons.org/licenses/by/2.0/'))
    assert is_allowed(parse_license('Public Domain'))
    assert not is_allowed(parse_license('http://creativecommons.org/licenses/by-nc/4.0/'))
    assert not is_allowed(parse_license('http://creativecommons.org/licenses/by-nc-sa/4.0/'))
    assert not is_allowed(parse_license('http://creativecommons.org/licenses/by-nd/4.0/'))
    assert not is_allowed(parse_license('cc-by-sa-4.0'))
    assert not is_allowed(None)


def test_share_alike_only_when_asked():
    sa = parse_license('http://creativecommons.org/licenses/by-sa/4.0/')
    assert not is_allowed(sa)
    assert is_allowed(sa, allow_share_alike=True)
    # NC stays refused even with share-alike accepted.
    assert not is_allowed(parse_license('cc-by-nc-sa-4.0'), allow_share_alike=True)
