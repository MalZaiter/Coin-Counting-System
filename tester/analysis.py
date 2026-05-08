"""
analysis.py — Coin Detection Accuracy Analysis

Runs the detection pipeline against all 150 images in data/raw/
and produces a full accuracy report including:
  - Exact match rate
  - Within ±1 rate
  - Per-image breakdown (correct / over-detected / under-detected)
  - Summary statistics (MAE, worst offenders)

Usage:
    python analysis.py
    python analysis.py --images_dir /path/to/other/folder
"""

import argparse
import os
import sys
import cv2
import numpy as np

sys.path.insert(0, os.path.join(os.path.dirname(__file__), "scripts", "coin_eval"))
from src.detect import detect_coins

GROUND_TRUTH = {
    "001.jpg": 3,  "002.jpg": 1,  "003.jpg": 1,  "004.jpg": 1,  "005.jpg": 1,
    "006.jpg": 1,  "007.jpg": 3,  "008.jpg": 1,  "009.jpg": 1,  "010.jpg": 1,
    "011.jpg": 2,  "012.jpg": 2,  "013.jpg": 1,  "014.jpg": 1,  "015.jpg": 1,
    "016.jpg": 2,  "017.jpg": 3,  "018.jpg": 3,  "019.jpg": 1,  "020.jpg": 3,
    "021.jpg": 1,  "022.jpg": 1,  "023.jpg": 2,  "024.jpg": 3,  "025.jpg": 3,
    "026.jpg": 2,  "027.jpg": 5,  "028.jpg": 2,  "029.jpg": 1,  "030.jpg": 3,
    "031.jpg": 2,  "032.jpg": 3,  "033.jpg": 1,  "034.jpg": 1,  "035.jpg": 1,
    "036.jpg": 2,  "037.jpg": 3,  "038.jpg": 7,  "039.jpg": 2,  "040.jpg": 5,
    "041.jpg": 1,  "042.jpg": 1,  "043.jpg": 2,  "044.jpg": 2,  "045.jpg": 3,
    "046.jpg": 3,  "047.jpg": 3,  "048.jpg": 1,  "049.jpg": 3,  "050.jpg": 2,
    "051.jpg": 4,  "052.jpg": 6,  "053.jpg": 1,  "054.jpg": 2,  "055.jpg": 2,
    "056.jpg": 1,  "057.jpg": 3,  "058.jpg": 3,  "059.jpg": 2,  "060.jpg": 1,
    "061.jpg": 1,  "062.jpg": 2,  "063.jpg": 3,  "064.jpg": 1,  "065.jpg": 1,
    "066.jpg": 3,  "067.jpg": 2,  "068.jpg": 1,  "069.jpg": 1,  "070.jpg": 1,
    "071.jpg": 1,  "072.jpg": 3,  "073.jpg": 3,  "074.jpg": 2,  "075.jpg": 2,
    "076.jpg": 4,  "077.jpg": 3,  "078.jpg": 3,  "079.jpg": 5,  "080.jpg": 2,
    "081.jpg": 1,  "082.jpg": 1,  "083.jpg": 1,  "084.jpg": 4,  "085.jpg": 5,
    "086.jpg": 3,  "087.jpg": 1,  "088.jpg": 6,  "089.jpg": 2,  "090.jpg": 2,
    "091.jpg": 1,  "092.jpg": 3,  "093.jpg": 1,  "094.jpg": 3,  "095.jpg": 3,
    "096.jpg": 8,  "097.jpg": 4,  "098.jpg": 4,  "099.jpg": 1,  "100.jpg": 1,
    "101.jpg": 3,  "102.jpg": 3,  "103.jpg": 3,  "104.jpg": 1,  "105.jpg": 1,
    "106.jpg": 3,  "107.jpg": 1,  "108.jpg": 1,  "109.jpg": 2,  "110.jpg": 2,
    "111.jpg": 3,  "112.jpg": 3,  "113.jpg": 1,  "114.jpg": 1,  "115.jpg": 4,
    "116.jpg": 1,  "117.jpg": 1,  "118.jpg": 8,  "119.jpg": 1,  "120.jpg": 2,
    "121.jpg": 4,  "122.jpg": 3,  "123.jpg": 1,  "124.jpg": 3,  "125.jpg": 2,
    "126.jpg": 3,  "127.jpg": 2,  "128.jpg": 3,  "129.jpg": 1,  "130.jpg": 2,
    "131.jpg": 1,  "132.jpg": 4,  "133.jpg": 3,  "134.jpg": 2,  "135.jpg": 1,
    "136.jpg": 2,  "137.jpg": 3,  "138.jpg": 3,  "139.jpg": 2,  "140.jpg": 2,
    "141.jpg": 1,  "142.jpg": 1,  "143.jpg": 2,  "144.jpg": 1,  "145.jpg": 1,
    "146.jpg": 6,  "147.jpg": 1,  "148.jpg": 1,  "149.jpg": 1,  "150.jpg": 1,
}


def run_analysis(images_dir: str) -> None:
    results = []
    missing = []
    errors = []

    files = sorted(GROUND_TRUTH.keys())
    total_expected = len(files)

    print(f"\nScanning {images_dir} for {total_expected} images ...\n")

    for filename in files:
        path = os.path.join(images_dir, filename)
        truth = GROUND_TRUTH[filename]

        if not os.path.exists(path):
            missing.append(filename)
            continue

        try:
            image = cv2.imread(path)
            if image is None:
                errors.append((filename, "Could not read file"))
                continue

            detected = len(detect_coins(image))
            diff = detected - truth

            results.append({
                "file": filename,
                "truth": truth,
                "detected": detected,
                "diff": diff,
                "abs_diff": abs(diff),
                "correct": diff == 0,
                "over": diff > 0,
                "under": diff < 0,
            })

            status = "OK" if diff == 0 else (f"OVER +{diff}" if diff > 0 else f"UNDER {diff}")
            print(f"  {filename}  truth={truth}  detected={detected}  [{status}]")

        except Exception as exc:
            errors.append((filename, str(exc)))
            print(f"  {filename}  ERROR: {exc}")

    processed = len(results)
    if processed == 0:
        print("\nNo images processed. Check that images exist in the folder.")
        return

    # ── Aggregate stats ──────────────────────────────────────────────────────
    exact       = sum(1 for r in results if r["correct"])
    within_one  = sum(1 for r in results if r["abs_diff"] <= 1)
    over_count  = sum(1 for r in results if r["over"])
    under_count = sum(1 for r in results if r["under"])
    abs_diffs   = [r["abs_diff"] for r in results]
    raw_diffs   = [r["diff"]     for r in results]
    mae         = float(np.mean(abs_diffs))
    max_err     = max(abs_diffs)

    print(f"\n{'='*60}")
    print(f"  RESULTS SUMMARY  ({processed} / {total_expected} images processed)")
    print(f"{'='*60}")
    print(f"  Exact match   (detected == truth)    : {exact:3d} / {processed}  ({100*exact/processed:.1f}%)")
    print(f"  Within ±1     (|detected - truth| ≤ 1): {within_one:3d} / {processed}  ({100*within_one/processed:.1f}%)")
    print(f"  Over-detected (detected > truth)     : {over_count:3d} images")
    print(f"  Under-detected(detected < truth)     : {under_count:3d} images")
    print(f"  Mean Absolute Error (MAE)            : {mae:.3f} coins")
    print(f"  Worst single error                   : {max_err} coins")
    print(f"{'='*60}")

    # ── Breakdown by truth count ──────────────────────────────────────────────
    by_truth = {}
    for r in results:
        t = r["truth"]
        by_truth.setdefault(t, []).append(r["correct"])

    print("\n  ACCURACY BY COIN COUNT IN IMAGE:")
    print(f"  {'Coins':>6}  {'Images':>7}  {'Correct':>8}  {'Rate':>7}")
    print(f"  {'-'*35}")
    for t in sorted(by_truth.keys()):
        group = by_truth[t]
        n = len(group)
        c = sum(group)
        print(f"  {t:>6}  {n:>7}  {c:>8}  {100*c/n:>6.1f}%")

    # ── Worst offenders ───────────────────────────────────────────────────────
    worst = sorted(results, key=lambda r: r["abs_diff"], reverse=True)[:10]
    print(f"\n  TOP 10 WORST PREDICTIONS:")
    print(f"  {'File':<12} {'Truth':>6} {'Got':>6} {'Error':>7}")
    print(f"  {'-'*38}")
    for r in worst:
        sign = f"+{r['diff']}" if r["diff"] > 0 else str(r["diff"])
        print(f"  {r['file']:<12} {r['truth']:>6} {r['detected']:>6} {sign:>7}")

    # ── Missing / error report ────────────────────────────────────────────────
    if missing:
        print(f"\n  MISSING FILES ({len(missing)}) — not found in {images_dir}:")
        for f in missing:
            print(f"    {f}")

    if errors:
        print(f"\n  PROCESSING ERRORS ({len(errors)}):")
        for f, msg in errors:
            print(f"    {f}: {msg}")

    print()


if __name__ == "__main__":
    parser = argparse.ArgumentParser(
        description="Analyse coin detection accuracy against ground truth."
    )
    parser.add_argument(
        "--images_dir",
        default="data/raw",
        help="Folder containing 001.jpg … 150.jpg  (default: data/raw)",
    )
    args = parser.parse_args()
    run_analysis(args.images_dir)
