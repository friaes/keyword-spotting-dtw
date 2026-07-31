import math
import re
from typing import Dict, List, Tuple

import xml.etree.ElementTree as ET
import numpy as np
try:
    from numba import njit
    _have_numba = True
except Exception:
    _have_numba = False
from PIL import Image, ImageDraw


# Numeric regex for SVG path coords
_CMD_NUM = re.compile(r"-?\d+\.?\d*")


def parse_svg_polygons(svg_path) -> Dict[str, List[Tuple[float, float]]]:
    out = {}
    tree = ET.parse(str(svg_path))
    root = tree.getroot()
    for elem in root.iter():
        if not isinstance(elem.tag, str):
            continue
        if not elem.tag.lower().endswith('path'):
            continue
        pathid = elem.get('id')
        d = elem.get('d')
        if not pathid or not d:
            continue
        nums = [float(x) for x in _CMD_NUM.findall(d)]
        out[pathid] = [(nums[i], nums[i + 1]) for i in range(0, len(nums), 2)]
    return out


def polygon_bbox(pts: List[Tuple[float, float]]):
    xs = [p[0] for p in pts]
    ys = [p[1] for p in pts]
    minx, maxx = int(math.floor(min(xs))), int(math.ceil(max(xs)))
    miny, maxy = int(math.floor(min(ys))), int(math.ceil(max(ys)))
    return minx, miny, maxx, maxy


def crop_by_polygon(img: Image.Image, polygon: List[Tuple[float, float]]):
    if not polygon:
        return img.copy()
    minx, miny, maxx, maxy = polygon_bbox(polygon)
    crop = img.crop((minx, miny, maxx, maxy)).convert('L')
    local = [(x - minx, y - miny) for x, y in polygon]
    mask = Image.new('L', crop.size, 0)
    ImageDraw.Draw(mask).polygon(local, fill=255)
    bg = Image.new('L', crop.size, 255)
    bg.paste(crop, mask=mask)
    return bg


def otsu_threshold(arr: np.ndarray) -> int:
    hist, _ = np.histogram(arr.ravel(), bins=256, range=(0, 255))
    total = arr.size
    sum_all = np.dot(np.arange(256), hist)
    sumB = 0.0
    wB = 0.0
    maxvar = 0.0
    thresh = 0
    for i in range(256):
        wB += hist[i]
        if wB == 0:
            continue
        wF = total - wB
        if wF == 0:
            break
        sumB += i * hist[i]
        mB = sumB / wB
        mF = (sum_all - sumB) / wF
        varBetween = wB * wF * (mB - mF) ** 2
        if varBetween > maxvar:
            maxvar = varBetween
            thresh = i
    return thresh


def binarize_pil(pilimg: Image.Image) -> np.ndarray:
    g = pilimg.convert('L')
    arr = np.array(g)
    t = otsu_threshold(arr)
    bw = (arr <= t).astype(np.uint8)
    return bw


def smooth_features(feats: np.ndarray, window: int) -> np.ndarray:
    # feats shape (M, W). Apply 1D uniform smoothing along W axis.
    if window <= 1:
        return feats
    M, W = feats.shape
    kernel = np.ones(window, dtype=float) / float(window)
    out = np.zeros_like(feats)
    for i in range(M):
        out[i] = np.convolve(feats[i], kernel, mode='same')
    return out


def extract_sliding_features(bw: np.ndarray, smooth: int = 1) -> np.ndarray:
    H, W = bw.shape
    if W == 0:
        return np.zeros((7, 0), dtype=float)
    # vectorized computation across columns
    bw_bool = (bw > 0).astype(np.uint8)
    any_black = bw_bool.any(axis=0)
    # upper contour: first index of True in column (argmax works because False=0, True=1)
    uc = np.argmax(bw_bool, axis=0).astype(float)
    # for columns with no black, set uc to H
    uc[~any_black] = float(H)
    # lower contour: find first True from bottom
    rev_idx = np.argmax(bw_bool[::-1, :], axis=0)
    lc = (H - 1 - rev_idx).astype(float)
    lc[~any_black] = 0.0
    # transitions: sum of abs diff along rows per column
    transitions = np.sum(np.abs(np.diff(bw_bool.astype(np.int8), axis=0)), axis=0).astype(float)
    # black fraction per column
    black_frac = bw_bool.sum(axis=0).astype(float) / float(H)
    # black between lc and uc inclusive: use column-wise cumulative sums
    cs = bw_bool.cumsum(axis=0)
    idx_uc = uc.astype(int)
    idx_lc = lc.astype(int)
    segment_sum = np.empty(W, dtype=float)
    seg_size = (idx_lc - idx_uc + 1).astype(float)
    for i in range(W):
        if not any_black[i]:
            segment_sum[i] = 0.0
            seg_size[i] = 1.0
            continue
        u = idx_uc[i]
        l = idx_lc[i]
        if u <= 0:
            s = cs[l, i]
        else:
            s = cs[l, i] - cs[u - 1, i]
        segment_sum[i] = float(s)
    black_between = segment_sum / np.maximum(1.0, seg_size)
    lc_diff = np.zeros(W, dtype=float)
    uc_diff = np.zeros(W, dtype=float)
    lc_diff[1:] = lc[1:] - lc[:-1]
    uc_diff[1:] = uc[1:] - uc[:-1]
    # HOG features per column (vectorized)
    num_hog_bins = 8
    hog_half_w = 1
    # compute gradients for whole image
    imgf = bw.astype(float)
    gy, gx = np.gradient(imgf)
    mag = np.hypot(gx, gy)
    ang = np.arctan2(gy, gx)
    ang = np.mod(np.abs(ang), np.pi)
    edges = np.linspace(0.0, np.pi, num_hog_bins + 1)
    # per-pixel bin index
    bin_idx = np.digitize(ang.ravel(), edges) - 1
    bin_idx = np.clip(bin_idx, 0, num_hog_bins - 1)
    mag_r = mag.ravel()
    # accumulate per-column per-bin magnitudes
    hog_feats = np.zeros((num_hog_bins, W), dtype=float)
    # compute per-column sums for each bin
    for b in range(num_hog_bins):
        mask = (bin_idx == b)
        if not mask.any():
            continue
        mag_bin = np.zeros(W, dtype=float)
        mags_b = mag_r[mask]
        cols = np.repeat(np.arange(W), imgf.shape[0])[mask]
        np.add.at(mag_bin, cols, mags_b)
        if hog_half_w > 0:
            kernel = np.ones(2 * hog_half_w + 1, dtype=float)
            mag_bin = np.convolve(mag_bin, kernel, mode='same')
        hog_feats[b, :] = mag_bin
    # normalize per-column histograms
    norms = np.linalg.norm(hog_feats, axis=0, keepdims=True)
    norms[norms == 0] = 1.0
    hog_feats = hog_feats / norms
    feats = np.vstack([uc, lc, transitions, black_frac, black_between, lc_diff, uc_diff, hog_feats])
    if smooth and smooth > 1:
        feats = smooth_features(feats, smooth)
    mu = feats.mean(axis=1, keepdims=True)
    s = feats.std(axis=1, keepdims=True)
    s[s == 0] = 1.0
    feats = (feats - mu) / s
    return feats


def dtw_distance(a: np.ndarray, b: np.ndarray, window: int = 0, max_cost: float | None = None) -> float:
    if a.size == 0 or b.size == 0:
        return float('inf')
    _, N = a.shape
    _, K = b.shape
    if window and window < abs(N - K):
        window = abs(N - K)
    cost = np.full((N + 1, K + 1), np.inf, dtype=float)
    cost[0, 0] = 0.0
    for i in range(1, N + 1):
        jstart = max(1, i - window)
        jend = min(K, i + window)
        row_min = np.inf
        for j in range(jstart, jend + 1):
            d = np.linalg.norm(a[:, i - 1] - b[:, j - 1])
            c1 = cost[i - 1, j]
            c2 = cost[i, j - 1]
            c3 = cost[i - 1, j - 1]
            cost[i, j] = d + min(c1, c2, c3)
            if cost[i, j] < row_min:
                row_min = cost[i, j]
        if max_cost is not None and row_min > max_cost:
            return float('inf')
    # Normalize by approximate warping path length to get an average per-step cost.
    # Exact path length requires backtracking; approximate with (N+K)/2.
    path_len = max(1.0, (N + K) / 2.0)
    return float(cost[N, K] / path_len)
