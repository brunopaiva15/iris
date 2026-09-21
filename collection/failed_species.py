#!/usr/bin/env python3
"""Read the collection logs back and return the species to retry.

    python3 failed_species.py shard*.log            # the list, one per line
    python3 failed_species.py --write retry shard*.log   # retry0.txt, retry1.txt…
    python3 failed_species.py --why shard*.log      # how many failures, and of what
    python3 failed_species.py --write retry \
        --against shard0.txt --against shard1.txt shard0.log shard1.log

A species whose source falls over is written `FAILED` and skipped: that is its
problem, not the other three hundred's (`build_dataset.py`). The price to pay
is having to catch up on them afterwards, and over 1,558 species spread across
four shards they are not going to be copied out by hand.

`--against` adds to the list the species that have **no** line in the log at
all: a shard killed on the way stops without writing anything, and "389/390"
is otherwise indistinguishable from "390/390" by eye. The files given are
paired with the logs in order.

Retrying is cheap and nearly always succeeds: collection failures are
overwhelmingly rate limits, and two hundred species relaunched on their own
weigh nothing on the servers. It is the same command as the collection, with
`--only-file` on the list returned here and the **same** output shard — so
that a species does not end up split between two folders.
"""
from __future__ import annotations

import argparse
import collections
import re
import sys
from pathlib import Path

#: `[12/390] Monstera deliciosa: FAILED (HTTPError: 429), species skipped`
#: The French marker of the older logs is still recognised, so that a log
#: written before the tools were translated can still be replayed.
LINE = re.compile(r'^\[\d+/\d+\]\s+(?P<name>.+?):\s+(?:FAILED|ÉCHEC)\s+\((?P<type>[A-Za-z_][\w.]*)[:)]')

#: Every species the log mentions, whatever its fate: kept, FAILED, or "name
#: unresolved at GBIF". What does not appear there was not processed at all.
SEEN = re.compile(r'^\[\d+/\d+\]\s+(?P<name>.+?):\s')

#: The reason carried by a species the log never mentioned.
NEVER = 'never processed'


def failures(text: str) -> list[tuple[str, str]]:
    """The (species, error type) pairs of a log, without duplicates.

    One species can fail twice if the log holds two passes; it is retried
    once, under its first reason.
    """
    out: dict[str, str] = {}
    for line in text.splitlines():
        m = LINE.match(line.strip())
        if m:
            out.setdefault(m.group('name').strip(), m.group('type'))
    return list(out.items())


def never_reached(text: str, expected: list[str]) -> list[str]:
    """The expected species the log does not mention at all.

    An interrupted shard — a machine going to sleep, a `nohup` falling over —
    leaves its last species without a single line. They did not FAIL: they were
    simply never tried, and nothing in the log distinguishes them from species
    that did not exist.
    """
    seen = {m.group('name').strip() for m in map(SEEN.match, (l.strip() for l in text.splitlines())) if m}
    return [n for n in expected if n not in seen]


def main(argv: list[str] | None = None) -> int:
    ap = argparse.ArgumentParser(description=__doc__, formatter_class=argparse.RawDescriptionHelpFormatter)
    ap.add_argument('logs', nargs='+', help='collection logs, e.g. shard*.log')
    ap.add_argument('--write', metavar='PREFIX',
                    help='write one list per log: PREFIX0.txt, PREFIX1.txt…, '
                         'in the order of the logs given')
    ap.add_argument('--against', action='append', metavar='FILE', default=[],
                    help='the species list handed to the corresponding log; whatever has no '
                         'line there is retried too. Repeatable, paired with the logs in order.')
    ap.add_argument('--why', action='store_true', help='the count per error type')
    args = ap.parse_args(argv)
    if args.against and len(args.against) != len(args.logs):
        ap.error(f'{len(args.against)} --against list(s) for {len(args.logs)} log(s): '
                 'they pair up in order, so there must be as many.')

    reasons: collections.Counter = collections.Counter()
    total = 0
    for i, path in enumerate(args.logs):
        text = Path(path).read_text(encoding='utf-8', errors='replace')
        missed = failures(text)
        if args.against:
            expected = [l.strip() for l in Path(args.against[i]).read_text(encoding='utf-8').splitlines() if l.strip()]
            known = {n for n, _ in missed}
            missed += [(n, NEVER) for n in never_reached(text, expected) if n not in known]
        reasons.update(t for _, t in missed)
        total += len(missed)
        names = [n for n, _ in missed]
        if args.write:
            target = Path(f'{args.write}{i}.txt')
            target.write_text('\n'.join(names) + ('\n' if names else ''), encoding='utf-8')
            print(f'{target}: {len(names)} species  ← {path}', file=sys.stderr)
        elif not args.why:
            print('\n'.join(names))
    if args.why or args.write:
        print(f'\n{total} failure(s) in total:', file=sys.stderr)
        for t, n in reasons.most_common():
            print(f'   {n:5d}  {t}', file=sys.stderr)
    return 0


if __name__ == '__main__':
    raise SystemExit(main())
