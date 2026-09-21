"""Retrying after collection failures.

The logs below are those of a real collection in four shards, where 216
species out of 1,558 hit rate limits. What matters: finding the names exactly,
not confusing a `:` of the error message with the one following the name, and
returning the reason — because "429" and "an unreadable image" do not call for
the same response.
"""
import pytest
import sys
from pathlib import Path

sys.path.insert(0, str(Path(__file__).resolve().parents[1]))

from failed_species import failures, main  # noqa: E402

LOG = """[1/390] Monstera deliciosa: 42 kept (gbif 30, inat 12)
[2/390] Ficus elastica: FAILED (HTTPError: 429 Client Error: Too Many Requests for url: https://api.inaturalist.org/v1/observations), species skipped
[3/390] Citrus × limon: FAILED (ConnectionError: ('Connection aborted.', RemoteDisconnected)), species skipped
[4/390] Pilea peperomioides: 200 kept (gbif 140, inat 60)
"""


def test_the_name_stops_before_the_reason():
    """The error message itself contains ":" — it is `FAILED` that bounds the
    name, not the first colon that comes along."""
    assert failures(LOG) == [('Ficus elastica', 'HTTPError'),
                             ('Citrus × limon', 'ConnectionError')]


def test_a_species_that_fell_twice_is_retried_once():
    twice = LOG + '[9/390] Ficus elastica: FAILED (Timeout: read), species skipped\n'
    assert [n for n, _ in failures(twice)] == ['Ficus elastica', 'Citrus × limon']


def test_a_log_written_before_the_translation_is_still_read():
    """The French marker of the older logs stays recognised: a collection run
    from before the tools were translated can still be replayed."""
    old = '[2/390] Ficus elastica: ÉCHEC (HTTPError: 429), espèce sautée\n'
    assert failures(old) == [('Ficus elastica', 'HTTPError')]


def test_the_retry_lists_follow_the_order_of_the_logs(tmp_path, monkeypatch, capsys):
    """A species must come back into **its** shard: split between two folders,
    it would end up counted twice at merge time."""
    (tmp_path / 'shard0.log').write_text(LOG, encoding='utf-8')
    (tmp_path / 'shard1.log').write_text(
        '[7/390] Sedum morganianum: FAILED (HTTPError: 429), species skipped\n', encoding='utf-8')
    monkeypatch.chdir(tmp_path)
    assert main(['--write', 'retry', 'shard0.log', 'shard1.log']) == 0
    assert (tmp_path / 'retry0.txt').read_text(encoding='utf-8').splitlines() == [
        'Ficus elastica', 'Citrus × limon']
    assert (tmp_path / 'retry1.txt').read_text(encoding='utf-8').splitlines() == ['Sedum morganianum']
    assert '2  HTTPError' in capsys.readouterr().err


def test_a_log_without_failures_gives_an_empty_list(tmp_path, monkeypatch):
    (tmp_path / 'ok.log').write_text('[1/2] Monstera deliciosa: 42 kept\n', encoding='utf-8')
    monkeypatch.chdir(tmp_path)
    assert main(['--write', 'retry', 'ok.log']) == 0
    assert (tmp_path / 'retry0.txt').read_text(encoding='utf-8') == ''


def test_a_species_never_processed_is_retried_too(tmp_path, monkeypatch, capsys):
    """The "389/390" case: the shard was killed before its last species, which
    therefore has no line — not even a FAILED. Nothing distinguishes it from a
    species that does not exist, except the list handed to the shard."""
    (tmp_path / 'shard0.log').write_text(LOG, encoding='utf-8')
    (tmp_path / 'shard0.txt').write_text(
        'Monstera deliciosa\nFicus elastica\nCitrus × limon\nPilea peperomioides\nSedum morganianum\n',
        encoding='utf-8')
    monkeypatch.chdir(tmp_path)
    assert main(['--write', 'retry', '--against', 'shard0.txt', 'shard0.log']) == 0
    assert (tmp_path / 'retry0.txt').read_text(encoding='utf-8').splitlines() == [
        'Ficus elastica', 'Citrus × limon', 'Sedum morganianum']
    assert 'never processed' in capsys.readouterr().err


def test_an_unresolved_name_is_not_to_be_retried(tmp_path, monkeypatch):
    """"name unresolved at GBIF" is not a breakdown: rerunning will change
    nothing, `synonyms.txt` answers that. The species has a line, so
    `--against` does not pick it up either."""
    (tmp_path / 'shard0.log').write_text(
        '[1/2] Sorbus aria: name unresolved at GBIF (FAMILY), to review\n'
        '[2/2] Monstera deliciosa: 42 kept\n', encoding='utf-8')
    (tmp_path / 'shard0.txt').write_text('Sorbus aria\nMonstera deliciosa\n', encoding='utf-8')
    monkeypatch.chdir(tmp_path)
    assert main(['--write', 'retry', '--against', 'shard0.txt', 'shard0.log']) == 0
    assert (tmp_path / 'retry0.txt').read_text(encoding='utf-8') == ''


def test_as_many_lists_as_logs(tmp_path, monkeypatch):
    (tmp_path / 'a.log').write_text(LOG, encoding='utf-8')
    (tmp_path / 'b.log').write_text(LOG, encoding='utf-8')
    (tmp_path / 'a.txt').write_text('Ficus elastica\n', encoding='utf-8')
    monkeypatch.chdir(tmp_path)
    with pytest.raises(SystemExit):
        main(['--against', 'a.txt', 'a.log', 'b.log'])
