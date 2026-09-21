"""Deduplication: exact duplicates by sha256, near-duplicates by perceptual hash.

Near-duplicates are grouped by union-find; the first arrival of each group
stays, the others move to `duplicate` with the original's checksum. A
near-duplicate across two different species is not a duplicate but a signal of
a doubtful label: it goes to `review`.
"""
from __future__ import annotations

from collections import defaultdict

from .images import hamming
from .manifest import STATUS_DUPLICATE, STATUS_KEPT, STATUS_REVIEW, ImageRecord

NEAR_THRESHOLD = 6   # bits of difference, out of 64
HASH_BITS = 64


def candidate_pairs(items: list[tuple[ImageRecord, int]], threshold: int):
    """The pairs of hashes within `threshold` bits of each other.

    Comparing every pair costs n^2: on 45,000 images that is a billion
    comparisons, hours of work — the collection of the full catalogue stopped
    there.

    Pigeonhole principle: if two hashes differ by at most `threshold` bits and
    the 64 bits are cut into `threshold + 1` bands, then at least one band is
    identical on both sides — otherwise there would have to be at least one
    difference per band, hence more than `threshold` in total. So it is enough
    to group by band and to compare only inside the groups. No true pair is
    lost; only the comparisons that never had a chance are saved.

    Each bucket is compared in one block with numpy, and only the genuinely
    close pairs are returned. The previous version returned every pair of a
    bucket and remembered those already seen so as not to repeat them: at
    236,000 images that made hundreds of millions of tuples in memory, and the
    process died at the end of a collection run.
    """
    import numpy as np

    bands = threshold + 1
    # The bits are shared out evenly: cutting into fixed-width slices would
    # leave a last band of a few bits only, hence few possible values, hence
    # enormous buckets — and the optimisation would vanish in that last one. 64
    # bits in 7 bands give 9,9,9,9,9,9,10.
    offsets = []
    start = 0
    for band in range(bands):
        width = (HASH_BITS - start) // (bands - band)
        offsets.append((start, width))
        start += width
    values = np.array([v for _, v in items], dtype=np.uint64)
    buckets: dict[tuple[int, int], list[int]] = defaultdict(list)
    for index, value in enumerate(values.tolist()):
        for band, (start, width) in enumerate(offsets):
            buckets[(band, (value >> start) & ((1 << width) - 1))].append(index)
    seen: set[tuple[int, int]] = set()
    for group in buckets.values():
        if len(group) < 2:
            continue
        idx = np.array(group)
        h = values[idx]
        # Hamming distance of every pair in the bucket, in one go.
        x = np.bitwise_xor.outer(h, h)
        d = np.zeros(x.shape, dtype=np.uint8)
        for shift in range(0, 64, 8):
            d += _POPCOUNT[((x >> np.uint64(shift)) & np.uint64(0xFF)).astype(np.uint8)]
        ai, bi = np.nonzero(np.triu(d <= threshold, k=1))
        for a, b in zip(idx[ai].tolist(), idx[bi].tolist()):
            pair = (a, b) if a < b else (b, a)
            if pair not in seen:
                seen.add(pair)
                yield pair


_POPCOUNT = __import__('numpy').array([bin(i).count('1') for i in range(256)], dtype='uint8')


class UnionFind:
    def __init__(self):
        self.parent: dict[str, str] = {}

    def find(self, x: str) -> str:
        self.parent.setdefault(x, x)
        while self.parent[x] != x:
            self.parent[x] = self.parent[self.parent[x]]
            x = self.parent[x]
        return x

    def union(self, a: str, b: str) -> None:
        ra, rb = self.find(a), self.find(b)
        if ra != rb:
            self.parent[rb] = ra


def near_duplicate_groups(records: list[ImageRecord], threshold: int = NEAR_THRESHOLD) -> dict[str, list[ImageRecord]]:
    """Groups of near-duplicates among the kept images, keyed by the
    representative's checksum. The comparison runs species by species, and by
    hash bands inside each one."""
    uf = UnionFind()
    by_species: dict[str, list[ImageRecord]] = defaultdict(list)
    for r in records:
        if r.status == STATUS_KEPT and r.phash:
            by_species[r.species].append(r)
    for rows in by_species.values():
        hashes = [(r, int(r.phash, 16)) for r in rows]
        for i, j in candidate_pairs(hashes, threshold):
            if hamming(hashes[i][1], hashes[j][1]) <= threshold:
                uf.union(hashes[i][0].checksum, hashes[j][0].checksum)
    groups: dict[str, list[ImageRecord]] = defaultdict(list)
    for rows in by_species.values():
        for r in rows:
            groups[uf.find(r.checksum)].append(r)
    return {k: v for k, v in groups.items() if len(v) > 1}


def mark_duplicates(records: list[ImageRecord], threshold: int = NEAR_THRESHOLD) -> dict[str, int]:
    """Full pass: exact ones first, near ones next. Changes the statuses in
    place and returns the tally."""
    exact = 0
    seen: dict[str, ImageRecord] = {}
    for r in records:
        if r.status != STATUS_KEPT:
            continue
        first = seen.get(r.checksum)
        if first is None:
            seen[r.checksum] = r
        else:
            r.status = STATUS_DUPLICATE
            r.duplicate_of = first.checksum
            r.reason = 'exact duplicate'
            exact += 1
    near = 0
    for group in near_duplicate_groups(records, threshold).values():
        group.sort(key=lambda r: r.downloaded_at)
        keeper = group[0]
        for r in group[1:]:
            r.status = STATUS_DUPLICATE
            r.duplicate_of = keeper.checksum
            r.reason = 'near-duplicate (perceptual hash)'
            near += 1
    return {'exact': exact, 'near': near}


def flag_cross_species(records: list[ImageRecord], threshold: int = NEAR_THRESHOLD) -> int:
    """Two nearly identical images under two species: one of the two labels is
    wrong. Both are sent to manual review."""
    kept = [r for r in records if r.status == STATUS_KEPT and r.phash]
    flagged = 0
    hashes = [(r, int(r.phash, 16)) for r in kept]
    for i, j in candidate_pairs(hashes, threshold):
        a, b = hashes[i][0], hashes[j][0]
        if a.species != b.species and hamming(hashes[i][1], hashes[j][1]) <= threshold:
            for r in (a, b):
                if r.status == STATUS_KEPT:
                    r.status = STATUS_REVIEW
                    r.reason = f'nearly identical to an image of {b.species if r is a else a.species}'
                    flagged += 1
    return flagged
