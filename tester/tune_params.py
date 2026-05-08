"""
test_phase6.py — Evaluate the Phase 6 detect.py against ground truth
=====================================================================

Copy BOTH files before running:
    copy attached_assets\\detect_phase6.py  src\\detect.py
    copy attached_assets\\test_phase6.py    tester\\test_phase6.py

Run:
    $env:PYTHONPATH="."; python tester/test_phase6.py

Compared to Phase 3 best: error=28, exact=137/150 (91%)

Phase 6 changes:
  1. RADIUS BUG FIX  max_r = int(0.40 * max(h,w)) after resize
                     (was max(10, int(220 * sc)) — too small for close-ups)
  2. CLAHE 2.0 → 4.0 + unsharp mask before Canny
  3. MIN_DIST bug fix: min_d = int(0.05 * max(h,w)) after resize
"""

import sys
import time
import cv2
import numpy as np
from pathlib import Path

# Use the src/detect.py that the user copied from detect_phase6.py
from src.detect import count_coins, build_preprocessed
from src.detect import (
    PARAM1, PARAM2, NMS_THRESH, RING_WIDTH,
    R_MIN_FRAC, R_MAX_FRAC, MD_FRAC,
    ARC_MIN, ED_MIN, CU_MIN, BR_MIN, SH_MIN,
    MAX_IMAGE_DIM,
)

IMAGES_DIR = Path("data/raw")

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

# Phase 3 reference for comparison
PHASE3 = {
    "011.jpg":(6,2),"036.jpg":(6,2),"044.jpg":(5,2),"134.jpg":(3,2),
    "085.jpg":(1,5),"040.jpg":(2,5),"043.jpg":(0,2),"136.jpg":(0,2),
    "024.jpg":(2,3),"027.jpg":(4,5),"047.jpg":(2,3),
    "081.jpg":(0,1),"132.jpg":(3,4),
}


def main():
    print("=" * 65)
    print("Phase 6 — Radius fix + CLAHE 4.0 + Unsharp mask")
    print("=" * 65)
    print(f"  R_MIN_FRAC={R_MIN_FRAC}  R_MAX_FRAC={R_MAX_FRAC}  "
          f"MD_FRAC={MD_FRAC}")
    print(f"  param2={PARAM2}  sh_min={SH_MIN}  arc_min={ARC_MIN}")
    print()

    if not IMAGES_DIR.exists():
        print(f"[ERROR] {IMAGES_DIR.resolve()}"); sys.exit(1)

    results = {}
    total_err = total_over = 0
    t0 = time.time()

    for fname, truth in GROUND_TRUTH.items():
        path = IMAGES_DIR / fname
        if not path.exists():
            print(f"  [WARN] {fname} not found"); continue
        try:
            circles = count_coins(path)
            det = len(circles)
        except Exception as e:
            print(f"  [ERR] {fname}: {e}"); det = 0
        results[fname] = det
        total_err  += abs(det - truth)
        total_over += max(0, det - truth)
        elapsed = time.time() - t0
        print(f"  {fname}  det={det:2d}  truth={truth}  "
              f"{'✓' if det==truth else ('+' if det>truth else '-')}"
              f"{abs(det-truth) if det!=truth else ''}", flush=True)

    total_under = total_err - total_over
    n_imgs  = len(results)
    n_exact = sum(1 for f, t in GROUND_TRUTH.items()
                  if f in results and results[f] == t)
    print(f"\n{'='*65}")
    print(f"SUMMARY")
    print(f"{'='*65}")
    print(f"  Total error : {total_err}  "
          f"(over={total_over}  under={total_under})")
    print(f"  Exact count : {n_exact}/{n_imgs}  "
          f"({100*n_exact/n_imgs:.1f}%)")
    print(f"  Runtime     : {time.time()-t0:.1f}s")
    print()
    print(f"  Phase 3 ref : error=28  exact=137/150 (91%)")
    delta_err = total_err - 28
    delta_ex  = n_exact - 137
    print(f"  Delta       : error{delta_err:+d}  exact{delta_ex:+d}")

    # Remaining misses
    misses = [(f, results[f], GROUND_TRUTH[f], results[f]-GROUND_TRUTH[f])
              for f in GROUND_TRUTH if f in results and results[f] != GROUND_TRUTH[f]]
    if misses:
        print(f"\n  Remaining misses ({len(misses)} images):")
        for f, det, truth, diff in sorted(misses, key=lambda x: -abs(x[3])):
            print(f"    {f}  det={det}  truth={truth}  {diff:+d}")

    # Focus table (Phase 3 problem images)
    print(f"\n{'='*65}")
    print("FOCUS — Phase 3 problem images")
    print(f"{'='*65}")
    print(f"  {'img':12s}  {'truth':5s}  {'P3 det':7s}  "
          f"{'P6 det':7s}  {'change':10s}")
    print(f"  {'-'*55}")
    for fname, (p3_det, truth) in sorted(PHASE3.items()):
        p3_diff = p3_det - truth
        p6_det  = results.get(fname, -1)
        p6_diff = (p6_det - truth) if p6_det >= 0 else 999
        if p6_diff == 0:
            change = "FIXED  ✓"
        elif abs(p6_diff) < abs(p3_diff):
            change = f"better  ({p3_diff:+d}→{p6_diff:+d})"
        elif abs(p6_diff) > abs(p3_diff):
            change = f"WORSE   ({p3_diff:+d}→{p6_diff:+d})"
        else:
            change = f"same    ({p6_diff:+d})"
        print(f"  {fname:<12}  {truth:5d}  {p3_det:4d}({p3_diff:+d})  "
              f"{p6_det:4d}({p6_diff:+d})  {change}")

    # NEW regressions: images that were ✓ in Phase 3 but not now
    print(f"\n{'='*65}")
    print("REGRESSIONS vs Phase 3  (images that WERE exact but now aren't)")
    print(f"{'='*65}")
    p3_exact = set(GROUND_TRUTH.keys()) - set(PHASE3.keys())
    regressions = []
    for fname in p3_exact:
        if fname not in results: continue
        det   = results[fname]
        truth = GROUND_TRUTH[fname]
        if det != truth:
            regressions.append((fname, det, truth, det - truth))
    if regressions:
        for f, det, truth, diff in sorted(regressions, key=lambda x: -abs(x[3])):
            print(f"  {f:<12}  det={det}  truth={truth}  {diff:+d}")
    else:
        print("  None — no regressions!")

    print(f"\n{'='*65}")
    # Radius diagnostic: show what radius range was used per image
    print("RADIUS DIAGNOSTIC (sample of 5 images)")
    print(f"{'='*65}")
    sample = ["081.jpg", "043.jpg", "040.jpg", "052.jpg", "096.jpg"]
    for fname in sample:
        path = IMAGES_DIR / fname
        if not path.exists(): continue
        img = cv2.imread(str(path))
        h0, w0 = img.shape[:2]
        sc = min(1.0, MAX_IMAGE_DIM / max(h0, w0))
        h = int(h0*sc); w = int(w0*sc)
        max_dim = max(h, w)
        min_r = max(5, int(R_MIN_FRAC * max_dim))
        max_r = min(max_dim//2, int(R_MAX_FRAC * max_dim))
        old_max_r = max(10, int(220 * sc))
        print(f"  {fname}  orig={w0}x{h0}  sc={sc:.3f}  "
              f"resized={w}x{h}  "
              f"min_r={min_r}  max_r={max_r}  "
              f"(old max_r was {old_max_r})")


if __name__ == "__main__":
    main()
