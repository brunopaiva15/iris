import io

import numpy as np
import pytest
from PIL import Image

from plant_dataset.dedup import flag_cross_species, mark_duplicates, near_duplicate_groups
from plant_dataset.images import ImageRejected, hamming, phash64, prepare
from plant_dataset.manifest import STATUS_DUPLICATE, STATUS_KEPT, STATUS_REVIEW, ImageRecord, now_iso


def picture(seed: int, size=(380, 340), fmt='JPEG', **save) -> bytes:
    """A test image: a smooth random field, unique to each seed.

    It is not a plant, it is a file whose truth is known. The field is drawn at
    low resolution then enlarged: two seeds give clearly different images —
    which sine waves of neighbouring frequencies did not guarantee — while one
    seed re-encoded stays the same image.
    """
    rng = np.random.default_rng(seed)
    coarse = rng.uniform(0, 255, size=(6, 7, 3))
    img = Image.fromarray(coarse.astype(np.uint8), 'RGB').resize(size, Image.BICUBIC)
    arr = np.clip(np.asarray(img, dtype=np.float64) + rng.normal(0, 6, size=(size[1], size[0], 3)), 0, 255)
    out = io.BytesIO()
    Image.fromarray(arr.astype(np.uint8), 'RGB').save(out, fmt, **save)
    return out.getvalue()


def test_prepare_accepts_a_sound_jpeg():
    # Below MAX_SIDE: the image goes through the preparer untouched.
    p = prepare(picture(1))
    assert p.width == 380 and p.height == 340
    assert len(p.sha256) == 64 and len(p.phash) == 16
    assert p.source_format == 'JPEG'


def test_prepare_rejects_corrupt_small_and_odd_formats():
    with pytest.raises(ImageRejected, match='corrupt'):
        prepare(b'not an image at all')
    with pytest.raises(ImageRejected, match='too small'):
        prepare(picture(2, size=(200, 150)))
    with pytest.raises(ImageRejected, match='format'):
        prepare(picture(3, size=(400, 400), fmt='GIF'))
    with pytest.raises(ImageRejected, match='aspect ratio'):
        prepare(picture(4, size=(4000, 320)))


def test_prepare_applies_exif_orientation():
    data = picture(5, size=(380, 340))
    img = Image.open(io.BytesIO(data))
    exif = img.getexif()
    exif[0x0112] = 6  # rotate by 90 degrees
    out = io.BytesIO()
    img.save(out, 'JPEG', exif=exif.tobytes())
    p = prepare(out.getvalue())
    assert (p.width, p.height) == (340, 380)
    assert Image.open(io.BytesIO(p.data)).getexif().get(0x0112, 1) == 1


def test_large_image_is_downscaled_to_max_side():
    p = prepare(picture(12, size=(3000, 2000)))
    assert (p.width, p.height) == (384, 256)
    assert Image.open(io.BytesIO(p.data)).size == (384, 256)
    # An image already below the limit is stored as it is, byte for byte: no
    # re-encoding, hence no generation loss.
    small = picture(13, size=(380, 340))
    assert prepare(small).data == small


def test_png_is_stored_as_jpeg():
    p = prepare(picture(6, fmt='PNG'))
    assert p.source_format == 'PNG'
    assert Image.open(io.BytesIO(p.data)).format == 'JPEG'


def test_phash_close_for_recompressed_far_for_different():
    a = prepare(picture(7))
    a_again = prepare(picture(7, quality=55))
    b = prepare(picture(8))
    assert hamming(int(a.phash, 16), int(a_again.phash, 16)) <= 6
    assert hamming(int(a.phash, 16), int(b.phash, 16)) > 12


def rec(species, checksum, phash, obs, when='2026-01-01T00:00:00Z'):
    return ImageRecord(species=species, internal_plant_id=species.lower().replace(' ', '-'), source='gbif', source_id=f'{obs}#0',
                       original_url='', image_url='', author='x', license='CC BY 4.0', license_url='', downloaded_at=when,
                       checksum=checksum, path=f'{species}/{checksum[:4]}.jpg', observation_id=obs, phash=phash)


def test_mark_duplicates_exact_and_near():
    a = prepare(picture(9)); a2 = prepare(picture(9, quality=50)); b = prepare(picture(10))
    records = [
        rec('Monstera deliciosa', a.sha256, a.phash, '1'),
        rec('Monstera deliciosa', a.sha256, a.phash, '2'),               # exact
        rec('Monstera deliciosa', a2.sha256, a2.phash, '3', when='2026-01-02T00:00:00Z'),  # quasi
        rec('Monstera deliciosa', b.sha256, b.phash, '4'),
    ]
    counts = mark_duplicates(records)
    assert counts == {'exact': 1, 'near': 1}
    assert records[1].status == STATUS_DUPLICATE and records[1].duplicate_of == a.sha256
    assert records[2].status == STATUS_DUPLICATE and records[2].duplicate_of == a.sha256
    assert records[0].status == STATUS_KEPT and records[3].status == STATUS_KEPT


def test_cross_species_near_duplicate_goes_to_review():
    a = prepare(picture(11)); a2 = prepare(picture(11, quality=60))
    records = [rec('Monstera deliciosa', a.sha256, a.phash, '1'), rec('Ficus elastica', a2.sha256, a2.phash, '2')]
    assert not near_duplicate_groups(records)   # not a *within-species* duplicate
    assert flag_cross_species(records) == 2
    assert all(r.status == STATUS_REVIEW for r in records)
    assert 'Ficus elastica' in records[0].reason


def test_band_bucketing_finds_exactly_the_same_pairs_as_brute_force():
    """The optimisation must lose nothing: on hashes drawn at random, the
    close pairs found by banding are exactly those an exhaustive comparison
    would have found."""
    from plant_dataset.dedup import candidate_pairs
    from plant_dataset.images import hamming

    rng = np.random.default_rng(1234)
    values = [int(rng.integers(0, 2 ** 63)) for _ in range(400)]
    # A few deliberate neighbours, 1 to 6 bits from the first one.
    for bits in range(1, 7):
        flip = 0
        for b in rng.choice(64, size=bits, replace=False):
            flip |= 1 << int(b)
        values.append(values[0] ^ flip)
    # And a neighbour at 7 bits, which must not be counted.
    flip = 0
    for b in rng.choice(64, size=7, replace=False):
        flip |= 1 << int(b)
    values.append(values[1] ^ flip)

    items = [(None, v) for v in values]
    threshold = 6
    brute = {(i, j) for i in range(len(values)) for j in range(i + 1, len(values))
             if hamming(values[i], values[j]) <= threshold}
    fast = {tuple(sorted(p)) for p in candidate_pairs(items, threshold)
            if hamming(values[p[0]], values[p[1]]) <= threshold}
    assert fast == brute
    assert len(brute) >= 6, 'the planted neighbours must indeed be found'


def test_band_bucketing_finds_planted_near_pairs_at_scale():
    """At scale, the search returns only the close pairs, finds them all, and
    does not keep the millions of pairs of one bucket in memory: that is what
    killed the end of an earlier collection run."""
    from plant_dataset.dedup import candidate_pairs

    rng = np.random.default_rng(7)
    values = [int(v) for v in rng.integers(0, 2 ** 63, size=8000)]
    planted = set()
    for k in range(0, 200, 2):
        # a twin within 3 bits, for a hundred pairs
        values[k + 1] = values[k] ^ 0b101 ^ (1 << 40)
        planted.add((k, k + 1))
    items = [(None, v) for v in values]
    found = set(candidate_pairs(items, 6))
    assert planted <= found
    # random 63-bit hashes are never close by chance
    assert len(found - planted) < 5, f'{len(found - planted)} spurious pairs'
