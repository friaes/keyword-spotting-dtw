#!/usr/bin/env python3
import argparse
import csv
import pickle
from pathlib import Path
from typing import Dict

from utils import parse_svg_polygons, crop_by_polygon


ROOT = Path(__file__).resolve().parent
IMAGES_TEST = ROOT / "images-test-set"
LOCS_TEST = ROOT / "locations-test-set"


def load_cache(cache_path: Path) -> Dict[str, Dict]:
    if not cache_path.exists():
        return {}
    return pickle.loads(cache_path.read_bytes())


def analyze_submission(sub_path: Path, cache_path: Path, topk: int, out_csv: Path, copy_images: Path = None):
    cache = load_cache(cache_path)

    # pre-load svg caches on demand
    svg_cache = {}

    rows = list(csv.reader(sub_path.open(encoding='utf8')))
    if not rows:
        print("Empty submission file")
        return
    header = rows[0]
    test_ids = header[1:]

    out_rows = []
    if copy_images:
        copy_images.mkdir(parents=True, exist_ok=True)

    for row in rows[1:]:
        kw = row[0]
        vals = []
        for v in row[1:]:
            try:
                vals.append(float(v))
            except Exception:
                vals.append(float('inf'))
        # indices of smallest values
        import numpy as np

        arr = np.array(vals)
        idxs = np.argsort(arr)[:topk]
        for rank, i in enumerate(idxs, start=1):
            wid = test_ids[int(i)]
            score = float(arr[int(i)])
            info = cache.get(wid, {})
            doc = info.get('doc', '')
            img_path = ''
            crop_path = ''
            if doc:
                img_path = str(IMAGES_TEST / f"{doc}.jpg")
                if copy_images:
                    # load svg for doc if needed
                    if doc not in svg_cache:
                        svg_path = LOCS_TEST / f"{doc}.svg"
                        if svg_path.exists():
                            svg_cache[doc] = parse_svg_polygons(svg_path)
                        else:
                            svg_cache[doc] = {}
                    poly = svg_cache[doc].get(wid)
                    if poly is not None:
                        try:
                            img = __import__('PIL').Image.open(IMAGES_TEST / f"{doc}.jpg").convert('RGB')
                            crop = crop_by_polygon(img, poly)
                            out_subdir = copy_images / kw
                            out_subdir.mkdir(parents=True, exist_ok=True)
                            fname = f"{rank:02d}_{wid}_{score:.4f}.png"
                            crop_path = str((out_subdir / fname).resolve())
                            crop.save(crop_path)
                        except Exception as e:
                            crop_path = ''
            out_rows.append({'keyword': kw, 'rank': rank, 'test_id': wid, 'score': score, 'doc': doc, 'image_path': img_path, 'crop_path': crop_path})

    # write CSV
    with out_csv.open('w', encoding='utf8', newline='') as fh:
        writer = csv.writer(fh)
        writer.writerow(['Keyword', 'Rank', 'TestWordID', 'Score', 'Doc', 'ImagePath', 'CropPath'])
        for r in out_rows:
            writer.writerow([r['keyword'], r['rank'], r['test_id'], f"{r['score']:.6f}", r['doc'], r['image_path'], r['crop_path']])

    print(f"Wrote analysis to: {out_csv}")
    if copy_images:
        print(f"Saved cropped images under: {copy_images}")


def main():
    ap = argparse.ArgumentParser(prog='analyze-submission')
    ap.add_argument('--submission', type=Path, required=True)
    ap.add_argument('--cache', type=Path, default=ROOT / 'features_cache.pkl')
    ap.add_argument('--topk', type=int, default=5)
    ap.add_argument('--out', type=Path, default=ROOT / 'submission_analysis.csv')
    ap.add_argument('--copy-images', type=Path, default=ROOT / 'submission_analysis_images')
    args = ap.parse_args()

    analyze_submission(args.submission, args.cache, args.topk, args.out, copy_images=args.copy_images)


if __name__ == '__main__':
    main()
