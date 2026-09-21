"""Downloading and checking the images.

An image enters the set only if it opens, has a useful size, is in an expected
format, and once its EXIF orientation has been applied. Its background is not
judged: a photograph taken in a cluttered living room is exactly what the model
will see on a phone.
"""
from __future__ import annotations

import hashlib
import io
from dataclasses import dataclass
from pathlib import Path

import numpy as np
import requests
from PIL import Image, ImageOps, UnidentifiedImageError

MIN_SIDE = 320             # below that, it is a thumbnail
# Above it, the image is reduced. Mobile models train at 224; 384 px leaves
# room to crop and resize, for seven times less disk than 1024 px — which
# decides how many species can be held, and the number of species matters more
# than the sharpness of images that will be reduced anyway.
#
# The set is entirely re-collected at this size: unlike the previous
# generations (1024, 640, 448), the resolution is therefore uniform and can no
# longer serve as a shortcut for telling batches of species apart.
MAX_SIDE = 384
MAX_BYTES = 25 * 1024 * 1024
ALLOWED_FORMATS = {'JPEG', 'PNG', 'WEBP'}
JPEG_QUALITY = 92


@dataclass
class Prepared:
    data: bytes           # the stored bytes (JPEG)
    sha256: str
    phash: str
    width: int
    height: int
    source_format: str


class ImageRejected(Exception):
    """The image does not fit; the message says why."""


def download(url: str, session: requests.Session, timeout: float = 30.0, max_bytes: int = MAX_BYTES) -> bytes:
    with session.get(url, stream=True, timeout=timeout) as r:
        r.raise_for_status()
        length = r.headers.get('content-length')
        if length and int(length) > max_bytes:
            raise ImageRejected(f'file too large ({length} bytes)')
        buf = io.BytesIO()
        for chunk in r.iter_content(64 * 1024):
            buf.write(chunk)
            if buf.tell() > max_bytes:
                raise ImageRejected('file too large')
        return buf.getvalue()


def phash64(img: Image.Image) -> int:
    """A 64-bit perceptual hash: DCT of a 32 x 32 greyscale thumbnail, sign of
    the 8 x 8 low frequencies against their median. Two photographs differing
    only by compression, a slight crop or a size land within a few bits of each
    other."""
    g = np.asarray(img.convert('L').resize((32, 32), Image.LANCZOS), dtype=np.float64)
    n = 32
    k = np.arange(n)
    # Orthonormal DCT-II matrix
    c = np.sqrt(2.0 / n) * np.cos(np.pi * (2 * k[None, :] + 1) * k[:, None] / (2 * n))
    c[0, :] /= np.sqrt(2.0)
    dct = c @ g @ c.T
    low = dct[:8, :8].flatten()
    med = np.median(low[1:])
    bits = 0
    for i, v in enumerate(low):
        if v > med:
            bits |= 1 << (63 - i)
    return bits


def hamming(a: int, b: int) -> int:
    return bin(a ^ b).count('1')


def prepare(data: bytes, min_side: int = MIN_SIDE, max_side: int = MAX_SIDE) -> Prepared:
    """Check the image and reduce it to an upright JPEG, ready to store."""
    try:
        probe = Image.open(io.BytesIO(data))
        probe.verify()
    except (UnidentifiedImageError, OSError, SyntaxError) as e:
        raise ImageRejected(f'corrupt or unreadable file: {e}') from e
    fmt = probe.format or ''
    if fmt not in ALLOWED_FORMATS:
        raise ImageRejected(f'format refused: {fmt or "unknown"}')
    img = Image.open(io.BytesIO(data))
    rotated = _needs_rewrite(img)
    try:
        img = ImageOps.exif_transpose(img)
    except Exception:  # damaged EXIF: keep the image as it is
        pass
    if getattr(img, 'n_frames', 1) > 1:
        raise ImageRejected('animated image')
    w, h = img.size
    if min(w, h) < min_side:
        raise ImageRejected(f'too small ({w}x{h})')
    if max(w, h) > 12 * min(w, h):
        raise ImageRejected(f'aberrant aspect ratio ({w}x{h})')
    rgb = img.convert('RGB')
    resized = max(w, h) > max_side
    if resized:
        scale = max_side / max(w, h)
        rgb = rgb.resize((max(1, round(w * scale)), max(1, round(h * scale))), Image.LANCZOS)
        w, h = rgb.size
    if fmt == 'JPEG' and img.mode == 'RGB' and not rotated and not resized:
        stored = data
    else:
        out = io.BytesIO()
        rgb.save(out, 'JPEG', quality=JPEG_QUALITY, optimize=True)
        stored = out.getvalue()
    return Prepared(
        data=stored,
        sha256=hashlib.sha256(stored).hexdigest(),
        phash=f'{phash64(rgb):016x}',
        width=w,
        height=h,
        source_format=fmt,
    )


def _needs_rewrite(img: Image.Image) -> bool:
    """A JPEG whose EXIF orientation is not 1 must be rewritten upright."""
    try:
        return img.getexif().get(0x0112, 1) != 1
    except Exception:
        return False


def store(prepared: Prepared, path: Path) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    path.write_bytes(prepared.data)
