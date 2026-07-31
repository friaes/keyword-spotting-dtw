#!/usr/bin/env python3
import csv
import random
import statistics
from pathlib import Path
import pickle
import math

from utils import dtw_distance

ROOT = Path(__file__).resolve().parent


def load_cache(p: Path):
    if not p.exists():
        print('No cache', p)
        return {}
    return pickle.loads(p.read_bytes())


def load_submission(p: Path):
    rows = list(csv.reader(p.open(encoding='utf8')))
    header = rows[0]
    test_ids = header[1:]
    data = {}
    for r in rows[1:]:
        kw = r[0]
        vals = [float(x) for x in r[1:]]
        data[kw] = dict(test_ids=test_ids, vals=vals)
    return data


def per_key_stats(sub):
    stats = {}
    for kw, info in sub.items():
        vals = [v for v in info['vals'] if math.isfinite(v)]
        if not vals:
            stats[kw] = {'count': 0}
            continue
        stats[kw] = {
            'count': len(vals),
            'min': min(vals),
            'max': max(vals),
            'mean': statistics.mean(vals),
            'median': statistics.median(vals),
            'stdev': statistics.pstdev(vals) if len(vals) > 1 else 0.0,
        }
    return stats


def sample_symmetry_checks(cache, sub, n=10, window=20):
    # pick random pairs (from cache) and compare dtw(a,b) vs dtw(b,a)
    ids = list(cache.keys())
    if not ids:
        return []
    checks = []
    for _ in range(n):
        a, b = random.sample(ids, 2)
        fa = cache[a]['features']
        fb = cache[b]['features']
        dab = dtw_distance(fa, fb, window=window)
        dba = dtw_distance(fb, fa, window=window)
        checks.append((a, b, dab, dba, dab - dba))
    return checks


def self_distance_checks(cache, n=20, window=20):
    ids = list(cache.keys())
    checks = []
    if not ids:
        return checks
    for _ in range(min(n, len(ids))):
        a = random.choice(ids)
        fa = cache[a]['features']
        d = dtw_distance(fa, fa, window=window)
        checks.append((a, d))
    return checks


def inspect_keyword_top(sub, cache, kw, topk=5):
    info = sub.get(kw)
    if not info:
        print('No keyword', kw)
        return
    vals = info['vals']
    ids = info['test_ids']
    pairs = sorted([(v, ids[i]) for i, v in enumerate(vals)], key=lambda x: x[0])[:topk]
    print(f"Top {topk} for {kw}:")
    for v, wid in pairs:
        entry = cache.get(wid)
        doc = entry['doc'] if entry else ''
        print(f"  {wid} (doc={doc}) score={v:.6f}")


def main():
    import argparse
    ap = argparse.ArgumentParser()
    ap.add_argument('--submission', type=Path, default=ROOT / 'submission_train_val.csv')
    ap.add_argument('--cache', type=Path, default=ROOT / 'features_cache.pkl')
    ap.add_argument('--topk', type=int, default=5)
    ap.add_argument('--sym-checks', type=int, default=10)
    ap.add_argument('--window', type=int, default=20)
    args = ap.parse_args()

    print('Loading cache', args.cache)
    cache = load_cache(args.cache)
    print('Loading submission', args.submission)
    sub = load_submission(args.submission)

    stats = per_key_stats(sub)
    print('\nPer-key summary:')
    for kw, s in stats.items():
        if s.get('count', 0) == 0:
            print(f"  {kw}: no finite observations")
        else:
            print(f"  {kw}: count={s['count']} min={s['min']:.4f} max={s['max']:.4f} mean={s['mean']:.4f} stdev={s['stdev']:.4f}")

    # show keywords with suspicious ranges
    print('\nKeywords with very small variance or odd ranges:')
    for kw, s in stats.items():
        if s.get('count', 0) > 0:
            if s['stdev'] < 1e-6:
                print(f"  {kw}: almost constant values (stdev={s['stdev']:.6f})")
            if s['min'] > 1e3:
                print(f"  {kw}: all values large (min={s['min']:.1f})")

    # symmetry checks
    print('\nDTW symmetry checks (dab vs dba):')
    sym = sample_symmetry_checks(cache, sub, n=args.sym_checks, window=args.window)
    for a, b, dab, dba, diff in sym:
        print(f"  {a} <-> {b}: dab={dab:.6f} dba={dba:.6f} diff={diff:.6e}")

    # self distances
    print('\nSelf-distance checks (should be near 0):')
    selfc = self_distance_checks(cache, n=20, window=args.window)
    for wid, d in selfc:
        print(f"  {wid}: self-dtwd={d:.6e}")

    # inspect a few keywords top matches
    print('\nTop matches for first few keywords:')
    for i, kw in enumerate(list(sub.keys())[:5]):
        inspect_keyword_top(sub, cache, kw, topk=args.topk)


if __name__ == '__main__':
    main()
