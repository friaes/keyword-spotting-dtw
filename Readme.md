# Keyword Spotting in Historical Documents (DTW)

A **learning-free keyword-spotting (KWS)** system that retrieves handwritten words from historical document images by visual similarity — without transcribing them. Built for the **Pattern Recognition** course at the **University of Fribourg** (BeNeFri joint MSc programme) 2025, on the **George Washington** dataset.

The idea: given a query keyword, find the word images in the collection that *look* most like it. This matters for historical manuscripts, where OCR and manual transcription are unreliable — appearance-based retrieval sidesteps the need for a transcription.

## Approach

A classical pattern-recognition pipeline, with no trained model:

1. **Word extraction** — crop each word from the page using the provided polygon annotations (SVG bounding boxes in `locations/`).
2. **Preprocessing** — greyscale, **Otsu binarisation**, and height normalisation to a fixed size.
3. **Feature extraction** — a **sliding-window** column descriptor turns each word image into a sequence of feature vectors (so a word becomes a variable-length time series).
4. **Sequence comparison** — **Dynamic Time Warping (DTW)** measures the dissimilarity between a keyword template and each candidate word, with a **Sakoe–Chiba band** to constrain the warping path and speed up the computation.
5. **Ranking** — for each keyword, candidate words are ranked by DTW dissimilarity and mapped to a 0–100 score.
6. **Evaluation** — standard KWS metrics: **mean Average Precision (mAP)** and precision/recall at top-k.

## Results

On the test set the system reaches **AP ≈ 0.27**, with precision@1 ≈ 0.64 (of the top-ranked retrievals, ~64% are correct), decreasing as recall increases — a reasonable result for a learning-free DTW approach on difficult historical handwriting. Full analysis is in `KWS-Exercise3-Report.pdf`.

## Code

| File | Purpose |
|------|---------|
| `kws.py` | Main pipeline: feature extraction + caching, reference selection per keyword, DTW scoring, and submission generation. Configurable via a rich CLI. |
| `utils.py` | Shared helpers (I/O, image handling, metrics). |
| `analyze_submission.py` | Computes evaluation metrics and per-keyword analysis from a submission. |
| `diagnose_dissimilarities.py` | Debugging tool that renders the top matches per keyword for visual inspection. |

`kws.py` exposes many knobs — DTW band width, image resize dimensions, feature smoothing, reference-selection strategy (`best_on_val`, `train_val`, …), score-normalisation mode, and multi-threaded DTW — used to explore what improves retrieval.

## Data

The **George Washington dataset** is provided by the course, not authored here:

- `images/`, `images-test-set/` — page images
- `locations/`, `locations-test-set/` — per-word polygon annotations (SVG)
- `transcription.tsv`, `keywords.tsv` — word transcriptions and the query keywords
- `train.tsv`, `validation.tsv`, `test.tsv` — document splits
- `sample-submission.csv` — the submission format template

## Requirements

Python 3 with `numpy`, `pillow`, and `matplotlib` (plus `tqdm`). DTW and the sliding-window features are implemented directly.

## Usage

```bash
# Build the feature cache (first run) and produce a ranked submission
python3 kws.py --out submission.csv

# Then evaluate it
python3 analyze_submission.py
```

Common options:

```bash
python3 kws.py --window 20 --resize-width 150 --resize-height 150 \
               --ref-select train_val --percent-mode ap --workers 4 --out submission.csv
```

The first run extracts features from every word image and writes a cache
(`features_cache.pkl`); later runs reuse it (pass `--force` to rebuild).

## Repository layout

```
.
├── kws.py                     # main KWS pipeline (CLI)
├── utils.py                   # helpers
├── analyze_submission.py      # evaluation / metrics
├── diagnose_dissimilarities.py# debug visualisation
├── images/  images-test-set/  # George Washington page images (provided)
├── locations/  locations-test-set/  # per-word polygon annotations (provided)
├── *.tsv                      # transcriptions, keywords, train/val/test splits (provided)
├── sample-submission.csv      # submission format (provided)
├── KWS-Exercise3-Report.pdf   # written report and analysis
└── Exercise-3.pdf             # assignment brief
```

Generated artefacts — the feature cache, produced submissions, evaluation CSVs and diagnostic images — are not tracked (see `.gitignore`); they are reproduced by running the scripts.

## Notes

This was a group project (Group 10). The George Washington data and assignment were provided by the course. Master's-level coursework at the University of Fribourg.
