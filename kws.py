import argparse
import math
import os
import pickle
import re
import xml.etree.ElementTree as ET
import numpy as np
import concurrent.futures
import matplotlib.pyplot as plt
import argparse
import pickle
import random
from pathlib import Path
from typing import Dict, List

import numpy as np

from utils import (
	parse_svg_polygons,
	crop_by_polygon,
	binarize_pil,
	extract_sliding_features,
	dtw_distance,
)


ROOT = Path(__file__).resolve().parent
IMAGES = ROOT / "images"
IMAGES_TEST = ROOT / "images-test-set"
LOCS = ROOT / "locations"
LOCS_TEST = ROOT / "locations-test-set"


def read_keywords(path: Path) -> List[str]:
	return [l for l in path.read_text(encoding="utf8").splitlines()]


def read_list(path: Path) -> List[str]:
	return [l for l in path.read_text(encoding="utf8").splitlines()]


def load_transcriptions(path: Path) -> Dict[str, str]:
	d: Dict[str, str] = {}
	for line in path.read_text(encoding="utf8").splitlines():
		parts = line.split("\t")
		if len(parts) < 2:
			continue
		wid, transcription = parts[0], parts[1]
		d[wid] = transcription.replace("-", "")
	return d


def make_features_cache(cache_path: Path, force: bool = False, resize_w: int = 0, resize_h: int = 0, smooth: int = 1) -> Dict[str, Dict]:
	if cache_path.exists() and not force:
		try:
			return pickle.loads(cache_path.read_bytes())
		except Exception:
			pass
	print("Building feature cache for competition (train+test)...")
	trans = load_transcriptions(ROOT / "transcription.tsv")
	svg_cache: Dict[str, Dict] = {}
	for svg_file in LOCS.glob("*.svg"):
		docid = svg_file.stem
		svg_cache[docid] = parse_svg_polygons(svg_file)
	svg_cache_test: Dict[str, Dict] = {}
	for svg_file in LOCS_TEST.glob("*.svg"):
		docid = svg_file.stem
		svg_cache_test[docid] = parse_svg_polygons(svg_file)

	train_docs = read_list(ROOT / "train.tsv")
	val_docs = read_list(ROOT / "validation.tsv")
	test_docs = read_list(ROOT / "test.tsv")

	wanted_train = set(train_docs + val_docs)
	features: Dict[str, Dict] = {}

	# process train/validation documents
	for doc in sorted(wanted_train):
		img_path = IMAGES / f"{doc}.jpg"
		polygons = svg_cache.get(doc, {})
		print(f"Processing doc {doc}: image exists={img_path.exists()}, polygons={len(polygons)}")
		if not img_path.exists():
			continue
		img = __import__('PIL').Image.open(img_path).convert('RGB')
		for wid, poly in polygons.items():
			try:
				crop = crop_by_polygon(img, poly)
				if resize_w > 0 and resize_h > 0:
					crop = crop.resize((resize_w, resize_h), __import__('PIL').Image.LANCZOS)
				bw = binarize_pil(crop)
				feats = extract_sliding_features(bw, smooth=smooth)
				features[wid] = {"features": feats, "transcription": trans.get(wid, ""), "doc": doc}
			except Exception as e:
				print(f"  failed {wid}: {e}")

	# process test documents
	for doc in sorted(test_docs):
		img_path = IMAGES_TEST / f"{doc}.jpg"
		polygons = svg_cache_test.get(doc, {})
		print(f"Processing test doc {doc}: image exists={img_path.exists()}, polygons={len(polygons)}")
		if not img_path.exists():
			continue
		img = __import__('PIL').Image.open(img_path).convert('RGB')
		for wid, poly in polygons.items():
			try:
				crop = crop_by_polygon(img, poly)
				if resize_w > 0 and resize_h > 0:
					crop = crop.resize((resize_w, resize_h), __import__('PIL').Image.LANCZOS)
				bw = binarize_pil(crop)
				feats = extract_sliding_features(bw, smooth=smooth)
				features[wid] = {"features": feats, "transcription": trans.get(wid, ""), "doc": doc}
			except Exception as e:
				print(f"  failed test {wid}: {e}")

	cache_path.write_bytes(pickle.dumps(features))
	return features



def select_reference_for_keyword(keyword: str, features: Dict[str, Dict], train_docs: List[str], val_docs: List[str], strategy: str = 'best_on_val', window: int = 20) -> List[str]:
	# helper: match transcription variants (allow punctuation suffixes)
	def transcription_matches(kw: str, transcription: str) -> bool:
		if transcription == kw:
			return True
		# strip trailing suffix segments like s_cm, s_pt, s_mi, s_qo, s_qt, s_x, etc.
		base = re.sub(r'(?:s_[a-z0-9]+)+$', '', transcription)
		if base == kw:
			return True
		# handle plural forms like Virginias_cm -> base becomes Virginias -> drop trailing 's'
		#if base.endswith('s') and base[:-1] == kw:
		#	return True
		return False

	# gather train instances for keyword (including punctuation/variant forms)
	train_ids = [wid for wid, info in features.items() if info['doc'] in train_docs and info['transcription'] == keyword] # transcription_matches(keyword, info['transcription'])
	val_ids = [wid for wid, info in features.items() if info['doc'] in val_docs and info['transcription'] == keyword] # info['transcription'] == keyword
	if strategy == 'first':
		return [train_ids[0]] if train_ids else []
	if strategy == 'random':
		if not train_ids:
			return []
		random.seed(0)
		return [random.choice(train_ids)]
	if strategy == 'all':
		return train_ids
	if strategy == 'train':
		return train_ids
	if strategy == 'val':
		return val_ids
	if strategy == 'train_val':
		# pooled references from train + validation (deduplicated)
		pooled = list(dict.fromkeys(train_ids + val_ids))
		return pooled
	# best_on_val: pick train id with lowest average DTW to validation instances of same keyword
	if strategy == 'best_on_val':
		if not val_ids:
			# no validation examples: fall back to first training instance if available
			return [train_ids[0]] if train_ids else []
		best = None
		best_score = float('inf')
		for tid in train_ids:
			tfeat = features[tid]['features']
			s = 0.0
			cnt = 0
			for vid in val_ids:
				vfeat = features[vid]['features']
				d = dtw_distance(vfeat, tfeat, window=window)
				if np.isfinite(d):
					s += d
					cnt += 1
			if cnt > 0:
				avg = s / cnt
				if avg < best_score:
					best_score = avg
					best = tid
		return [best] if best is not None else ([train_ids[0]] if train_ids else [])

	# best_on_train: pick a validation id that is closest on average to training instances
	if strategy == 'best_on_train':
		# if no validation examples, fall back to first training instance
		if not val_ids:
			return [train_ids[0]]
		best = None
		best_score = float('inf')
		for vid in val_ids:
			vfeat = features[vid]['features']
			s = 0.0
			cnt = 0
			for tid in train_ids:
				tfeat = features[tid]['features']
				d = dtw_distance(tfeat, vfeat, window=window)
				if np.isfinite(d):
					s += d
					cnt += 1
			if cnt > 0:
				avg = s / cnt
				if avg < best_score:
					best_score = avg
					best = vid
		return [best] if best is not None else [val_ids[0]]
	# fallback: return first training id if available, otherwise empty
	return [train_ids[0]] if train_ids else []


def build_submission(cache_path: Path = Path('features_cache.pkl'), force: bool = False, window: int = 20, resize_w: int = 0, resize_h: int = 0, smooth: int = 1, ref_mode: str = 'min', ref_select: str = 'best_on_val', percent_mode: str = 'zero_max', max_dissim: float = 0.0, out: Path = Path('submission.csv'), dry_run: bool = False, scale_to_100: bool = True, test_first: bool = False, workers: int = 1):
	features = make_features_cache(cache_path, force=force, resize_w=resize_w, resize_h=resize_h, smooth=smooth)
	keywords = read_keywords(ROOT / 'keywords.tsv')
	keywords_norm = [k.replace('-', '') for k in keywords]
	train_docs = read_list(ROOT / 'train.tsv')
	val_docs = read_list(ROOT / 'validation.tsv')
	test_docs = read_list(ROOT / 'test.tsv')

	# collect test word ids from built features (docs from test set)
	test_word_ids = [wid for wid, info in features.items() if info.get('doc') in test_docs]
	test_word_ids = sorted(test_word_ids)

	# fast debug mode: only evaluate the first test image to speed up development
	if test_first and test_word_ids:
		test_word_ids = [test_word_ids[0]]

	if dry_run:
		print(f"Built cache entries: {len(features)}; test words: {len(test_word_ids)}; keywords: {len(keywords)}")
		return

	# prepare output: dict keyword -> list of dissimilarities in test_word_ids order
	submission = {}
	for kw_raw, kw in zip(keywords, keywords_norm):
		refs = select_reference_for_keyword(kw, features, train_docs, val_docs, strategy=ref_select, window=window)
		if not refs:
			print(f"Warning: no training references found for keyword {kw}")
			# no references: if we scale to 0-100, assign worst-case 100; otherwise use configured max_dissim or large constant
			if scale_to_100:
				submission[kw_raw] = [100.0] * len(test_word_ids)
			else:
				submission[kw_raw] = [float(max_dissim) if max_dissim > 0 else 1e6] * len(test_word_ids)
			continue
		# compute distances to every test word (raw average per-step DTW returned from utils)
		dists = [None] * len(test_word_ids)
		observed = []

		def compute_for_index(idx_tid):
			idx, tid = idx_tid
			info = features.get(tid)
			if info is None:
				return (idx, None)
			tfeat = info['features']
			# compute per-ref distances
			ref_d = []
			for r in refs:
				rfeat = features.get(r)
				if rfeat is None:
					continue
				d = dtw_distance(rfeat['features'], tfeat, window=window)
				ref_d.append(d)
			if not ref_d:
				return (idx, None)
			if ref_mode == 'min':
				val = float(np.min(ref_d))
			else:
				val = float(np.mean(ref_d))
			return (idx, val)

		if workers is None or workers <= 1:
			for idx, tid in enumerate(test_word_ids):
				idx, val = compute_for_index((idx, tid))
				dists[idx] = val
				if val is not None and np.isfinite(val):
					observed.append(val)
		else:
			with concurrent.futures.ThreadPoolExecutor(max_workers=workers) as ex:
				for idx, val in ex.map(compute_for_index, enumerate(test_word_ids)):
					dists[idx] = val
					if val is not None and np.isfinite(val):
						observed.append(val)
		# Now map values to 0-100 per-key if requested. Smaller DTW = more similar -> lower dissimilarity.
		if scale_to_100:
			finite_obs = [v for v in observed if np.isfinite(v)]
			if finite_obs:
				min_obs = float(min(finite_obs))
				max_obs = float(max(finite_obs))
				# choose mapping mode
				if percent_mode == 'minmax':
					if max_obs <= min_obs:
						# All observed equal -> map observed to 0
						scaled = [(0.0 if v is not None else 100.0) for v in dists]
					else:
						rng = max_obs - min_obs
						scaled = []
						for v in dists:
							if v is None or (not np.isfinite(v)):
								scaled.append(100.0)
							else:
								s = 100.0 * (v - min_obs) / rng
								# clip
								if s < 0.0:
									s = 0.0
								if s > 100.0:
									s = 100.0
								scaled.append(s)
				elif percent_mode == 'zero_max':
					# map raw to percent = 100 * raw / max_obs; raw==0 => 0; missing => 100
					if max_obs <= 0.0:
						scaled = [(0.0 if v is not None else 100.0) for v in dists]
					else:
						scaled = []
						for v in dists:
							if v is None or (not np.isfinite(v)):
								scaled.append(100.0)
							else:
								s = 100.0 * (v / max_obs)
								if s < 0.0:
									s = 0.0
								if s > 100.0:
									s = 100.0
								scaled.append(s)
				elif percent_mode == 'rank':
					# map to percentile rank among observed values (0 = best)
					sorted_obs = sorted(finite_obs)
					nobs = len(sorted_obs)
					def rank_percent(x):
						# position of x among sorted_obs (lower is better)
						# use linear interpolation for ties
						# find first index where sorted_obs[idx] >= x
						for idx, val in enumerate(sorted_obs):
							if x <= val:
								return 100.0 * idx / max(1, nobs - 1)
						return 100.0
					scaled = []
					for v in dists:
						if v is None or (not np.isfinite(v)):
							scaled.append(100.0)
						else:
							scaled.append(rank_percent(v))
				elif percent_mode == 'ap':
					# AP-style mapping: compute precision@k over the ranked test list and assign
					# percent = 100 * (1 - precision_at_k) so better-ranked correct items get lower percent.
					# we need ground-truth matches from features' transcriptions.
					def transcription_matches_local(kw_local: str, transcription: str) -> bool:
						if transcription == kw_local:
							return True
						base_local = re.sub(r'(?:s_[a-z0-9]+)+$', '', transcription)
						if base_local == kw_local:
							return True
						return False
					# build list of (idx, dist, is_pos)
					rank_list = []
					for idx, tid in enumerate(test_word_ids):
						v = dists[idx]
						if v is None or (not np.isfinite(v)):
							# treat missing as very large (put at end)
							rank_list.append((idx, float('inf'), False))
						else:
							trans = features.get(tid, {}).get('transcription', '')
							is_pos = transcription_matches_local(kw, trans)
							rank_list.append((idx, float(v), bool(is_pos)))
					# sort by distance asc (inf at end)
					rank_list.sort(key=lambda x: (x[1], x[0]))
					# iterate and compute precision@k
					scaled = [100.0] * len(dists)
					correct = 0
					for rank_pos, (orig_idx, dist_val, is_pos) in enumerate(rank_list, start=1):
						if is_pos:
							correct += 1
						precision_at_k = correct / float(rank_pos)
						p = 100.0 * (1.0 - precision_at_k)
						if p < 0.0:
							p = 0.0
						if p > 100.0:
							p = 100.0
						scaled[orig_idx] = p
				else:
					# unknown mode -> fallback to minmax
					rng = max_obs - min_obs if max_obs > min_obs else 1.0
					scaled = [(0.0 if v is not None else 100.0) for v in dists]
			else:
				# no observations -> all unknown
				scaled = [100.0 for _ in dists]
			submission[kw_raw] = scaled
		else:
			# do not scale, fill missing with provided max_dissim or large constant
			default = float(max_dissim) if max_dissim > 0 else 1e6
			submission[kw_raw] = [default if x is None else float(x) for x in dists]

	# write CSV: header Keyword,<test ids...>
	with out.open('w', encoding='utf8') as fh:
		fh.write('Keyword,' + ','.join(test_word_ids) + '\n')
		for kw in keywords:
			row = submission.get(kw)
			if row is None:
				row = [1e6] * len(test_word_ids)
			fh.write(kw + ',' + ','.join(f"{v:.4f}" for v in row) + '\n')
	print(f"Wrote submission to: {out}")


def main():
	ap = argparse.ArgumentParser(prog='kws-competition')
	ap.add_argument("--cache", default=Path("features_cache.pkl"))
	ap.add_argument("--force", action="store_true", help="Rebuild feature cache")
	ap.add_argument("--window", type=int, default=10, help="Sakoe-Chiba window (columns). 0 = no band")
	ap.add_argument("--resize-width", type=int, default=150, help="Resize cropped words to this width (px). Use with --resize-height to normalize sequence lengths.")
	ap.add_argument("--resize-height", type=int, default=150, help="Resize cropped words to this height (px). Use with --resize-width to normalize sequence lengths.")
	ap.add_argument("--smooth", type=int, default=1, help="Feature smoothing window (columns). 1 = no smoothing")
	ap.add_argument("--ref-mode", choices=["min", "avg"], default="avg", help="Aggregate multiple references: min or avg")
	ap.add_argument("--ref-select", choices=["best_on_val", "best_on_train", "val", "train", "train_val", "first", "random", "all"], default="train_val", help="How to select reference image(s) for each keyword")
	ap.add_argument("--percent-mode", choices=["minmax", "zero_max", "rank", "ap"], default="ap", help="How to map raw DTW to 0-100 percentage: minmax=min->0,max->100 (legacy), zero_max=raw/max_obs*100, rank=rank percentile")
	ap.add_argument("--max-dissim", type=float, default=0.0, help="If >0, use this as max dissimilarity for missing words; otherwise use observed max per-keyword")
	ap.add_argument("--out", default=Path("submission.csv"), help="Output submission CSV path")
	ap.add_argument("--no-scale", action="store_true", help="Do not scale distances to 0-100 per keyword; keep raw values")
	ap.add_argument("--dry-run", action="store_true", help="Build cache and show counts, do not compute full scoring")
	ap.add_argument("--test-first", action="store_true", help="Only compute scores for the first test image (fast debugging)")
	ap.add_argument("--workers", type=int, default=4, help="Number of threads to use for DTW computation (1 = serial)")
	args = ap.parse_args()

	cache_path = Path(args.cache)
	if args.force and cache_path.exists():
		cache_path.unlink()

	build_submission(
		cache_path=cache_path,
		force=args.force,
		window=args.window,
		resize_w=args.resize_width,
		resize_h=args.resize_height,
		smooth=args.smooth,
		ref_mode=args.ref_mode,
		ref_select=args.ref_select,
		percent_mode=args.percent_mode,
		max_dissim=args.max_dissim,
		out=Path(args.out),
		dry_run=args.dry_run,
		scale_to_100=(not args.no_scale),
		test_first=args.test_first,
		workers=args.workers,
	)


if __name__ == "__main__":
	main()
