"""The network's input size, and the loading size that must follow it.

`IMAGE_SIZE` and `LOAD_SIZE` used to be constants one had to edit by hand
before training at 320 px. A hand edit just before a half-hour pass is the
kind of thing that goes wrong and only shows in the final number: changing the
input without the loading size frames differently from what `model.json`
announces to the application, and costs points nobody can account for.
"""
import sys
from pathlib import Path

import pytest

sys.path.insert(0, str(Path(__file__).resolve().parents[1]))

import train  # noqa: E402


@pytest.fixture(autouse=True)
def restore():
    """The constants are global: every test puts the originals back."""
    before = (train.IMAGE_SIZE, train.LOAD_SIZE)
    yield
    train.IMAGE_SIZE, train.LOAD_SIZE = before


def test_load_size_follows_the_input():
    train.set_input_size(320)
    assert train.IMAGE_SIZE == 320
    assert train.LOAD_SIZE == 366          # 320 x 256/224, the same one-eighth margin


def test_the_crop_margin_is_preserved():
    ratio = train.LOAD_SIZE / train.IMAGE_SIZE
    for px in (256, 288, 320, 384):
        train.set_input_size(px)
        assert train.LOAD_SIZE / train.IMAGE_SIZE == pytest.approx(ratio, abs=0.005)
        train.IMAGE_SIZE, train.LOAD_SIZE = 224, 256


def test_beyond_the_stored_size_is_refused():
    """The set is stored at 448 px: beyond that, we would upscale without
    creating any detail."""
    with pytest.raises(SystemExit, match='448'):
        train.set_input_size(448)


def test_a_size_that_fits_exactly_is_accepted():
    train.set_input_size(392)              # 392 x 256/224 = 448, the limit
    assert train.LOAD_SIZE == train.SOURCE_SIZE


def test_too_small_is_refused():
    with pytest.raises(SystemExit, match='too small'):
        train.set_input_size(16)


def test_the_exported_metadata_follows():
    """`model.json` carries the preprocessing recipe the application reads back."""
    train.set_input_size(320)
    assert (train.IMAGE_SIZE, train.LOAD_SIZE) == (320, 366)
