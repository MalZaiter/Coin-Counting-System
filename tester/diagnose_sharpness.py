"""
diagnose_sharpness.py — Ring Sharpness Diagnostic for False Positives
======================================================================

Runs the full pipeline with the Phase 1 best params (param2=50, arc_min=0.35)
and NO sharpness filter, then for every false-positive circle prints its
ring_edge_sharpness score alongside the other filter scores.

This tells you exactly what sharpness_min to use before running the full sweep.

Outputs:
  • Per-image table for the 15 worst overcounting images
  • Score distribution histogram (text) across all false-positive circles
  • Score distribution for true-positive circles (for comparison)
  • Recommended sharpness_min threshold based on the score gap

Usage:
    $env:PYTHONPATH="."; python tester/diagnose_sharpness.py
"""

import sys
import time
import cv2
import numpy as np
from pathlib import Path
from typing import Dict

# ===========================================================================
# CONFIGURATION — match Phase 3 script exactly
# ===========================================================================

IMAGES_DIR = Path("data/raw")

GROUND_TRUTH: Dict[str, int] = {
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

MAX_IMAGE_DIM = 960

PARAM2     = 50
PARAM1     = 40
MIN_DIST   = 80
ARC_MIN    = 0.35
ED_MIN     = 0.08
CU_MIN     = 0.25
BR_MIN     = 30.0
NMS_THRESH = 0.5
RING_WIDTH = 8

# How many worst-overcounting images to show in detail
TOP_WORST = 15

# ===========================================================================
# Sampling arrays
# ===========================================================================

_N_SEC  = 36
_A36    = np.linspace(0, 2 * np.pi, _N_SEC, endpoint=False)
_C36    = np.cos(_A36)
_S36    = np.sin(_A36)
_SEC_I  = (_A36 / (2 * np.pi) * _N_SEC).astype(int) % _N_SEC

_N_RING = 72
_A72    = np.linspace(0, 2 * np.pi, _N_RING, endpoint=False)
_C72    = np.cos(_A72)
_S72    = np.sin(_A72)

_FRACS  = np.linspace(0.15, 0.90, 8)
_N_DISC = 20
_Ad     = np.linspace(0, 2 * np.pi, _N_DISC, endpoint=False)
_Cd     = np.cos(_Ad)
_Sd     = np.sin(_Ad)


# ===========================================================================
# Filter helpers
# ===========================================================================

def _pts(cx, cy, radius, C, S, H, W):
    xs = np.clip(np.round(cx + radius * C).astype(np.int32), 0, W - 1)
    ys = np.clip(np.round(cy + radius * S).astype(np.int32), 0, H - 1)
    return xs, ys


def _nms(circles, thresh):
    circles = sorted(circles, key=lambda c: c[2], reverse=True)
    keep = []
    for x1, y1, r1 in circles:
        if all(np.hypot(x1 - x2, y1 - y2) >= (r1 + r2) * (1 - thresh)
               for x2, y2, r2 in keep):
            keep.append((x1, y1, r1))
    return keep


def _arc(edges, cx, cy, r):
    H, W = edges.shape
    w    = RING_WIDTH
    hit  = np.zeros(_N_SEC, dtype=bool)
    for frac in (1 - w / max(r, 1), 1.0, 1 + w / max(r, 1)):
        xs, ys = _pts(cx, cy, r * frac, _C36, _S36, H, W)
        hit[_SEC_I[edges[ys, xs] > 0]] = True
    return hit.sum() / _N_SEC


def _edge_density(edges, cx, cy, r):
    H, W   = edges.shape
    ri, ro = max(r - RING_WIDTH, 1), r + RING_WIDTH
    hits = tot = 0
    for frac in np.linspace(ri / max(r, 1), ro / max(r, 1), 5):
        xs, ys = _pts(cx, cy, r * frac, _C72, _S72, H, W)
        hits  += int((edges[ys, xs] > 0).sum())
        tot   += _N_RING
    return hits / tot if tot else 0.0


def _cu(hsv, cx, cy, r):
    H, W    = hsv.shape[:2]
    inner_r = max(int(r * 0.80), 1)
    hues, sats, vals = [], [], []
    for frac in _FRACS:
        xs, ys = _pts(cx, cy, inner_r * frac, _Cd, _Sd, H, W)
        p = hsv[ys, xs]
        hues.append(p[:, 0])
        sats.append(p[:, 1])
        vals.append(p[:, 2])
    hue = np.concatenate(hues).astype(np.float32)
    sat = np.concatenate(sats)
    val = np.concatenate(vals).astype(np.float32)
    col = sat > 12
    if col.sum() >= 10:
        return float(1 / (1 + np.std(hue[col]) / 30))
    return float(1 / (1 + np.std(val) / 40)) if len(val) else 0.0


def _brightness(hsv, cx, cy, r):
    H, W    = hsv.shape[:2]
    inner_r = max(int(r * 0.75), 1)
    vals = []
    for frac in _FRACS:
        xs, ys = _pts(cx, cy, inner_r * frac, _Cd, _Sd, H, W)
        vals.append(hsv[ys, xs, 2])
    v = np.concatenate(vals).astype(np.float32)
    return float(np.mean(v)) if len(v) else 0.0


def _edge_density_at_radius(edges, cx, cy, sample_r):
    H, W   = edges.shape
    sample_r = max(sample_r, 1.0)
    xs, ys = _pts(cx, cy, sample_r, _C72, _S72, H, W)
    return float((edges[ys, xs] > 0).mean())


def _ring_edge_sharpness(edges, cx, cy, r):
    """
    edge_density_at(r) / mean(edge_density_at(0.55r), edge_density_at(1.55r))

    Returns (sharpness_score, ring, inner, outer) for diagnostics.
    """
    ring  = _edge_density_at_radius(edges, cx, cy, r)
    inner = _edge_density_at_radius(edges, cx, cy, max(r * 0.55, 1))
    outer = _edge_density_at_radius(edges, cx, cy, r * 1.55)
    bg    = (inner + outer) * 0.5
    if bg < 1e-4:
        return ring * 100.0, ring, inner, outer
    return ring / bg, ring, inner, outer


# ===========================================================================
# Preprocessing + cache
# ===========================================================================

def _prep(img):
    h, s, v = cv2.split(cv2.cvtColor(img, cv2.COLOR_BGR2HSV))
    v    = cv2.createCLAHE(2.0, (8, 8)).apply(v)
    img  = cv2.cvtColor(cv2.merge((h, s, v)), cv2.COLOR_HSV2BGR)
    gray = cv2.cvtColor(img, cv2.COLOR_BGR2GRAY)
    sm   = cv2.GaussianBlur(cv2.bilateralFilter(gray, 9, 75, 75), (7, 7), 2.0)
    bl   = cv2.GaussianBlur(cv2.bilateralFilter(gray, 13, 90, 90), (9, 9), 2.5)
    e    = cv2.Canny(bl, 15, 50)
    k1   = cv2.getStructuringElement(cv2.MORPH_ELLIPSE, (3, 3))
    k2   = cv2.getStructuringElement(cv2.MORPH_ELLIPSE, (7, 7))
    edges = cv2.morphologyEx(cv2.dilate(e, k1, iterations=2),
                             cv2.MORPH_CLOSE, k2, iterations=2)
    hsv = cv2.cvtColor(img, cv2.COLOR_BGR2HSV)
    return sm, edges, hsv


def build_cache(images_dir, gt):
    cache = []
    for fname, truth in gt.items():
        path = images_dir / fname
        if not path.exists():
            print(f"  [WARN] {fname} not found")
            continue
        img = cv2.imread(str(path))
        if img is None:
            print(f"  [WARN] cannot read {fname}")
            continue
        h, w = img.shape[:2]
        sc = min(1.0, MAX_IMAGE_DIM / max(h, w))
        if sc < 1.0:
            img = cv2.resize(img, (int(w * sc), int(h * sc)),
                             interpolation=cv2.INTER_AREA)
        sm, edges, hsv = _prep(img)
        cache.append((fname, sm, edges, hsv, sc, truth))
    return cache


# ===========================================================================
# Score histogram (text-based)
# ===========================================================================

def _histogram(scores, bins=20, width=50, label=""):
    if not scores:
        print(f"  (no {label} scores)")
        return
    lo, hi = min(scores), max(scores)
    if lo == hi:
        hi = lo + 0.001
    step  = (hi - lo) / bins
    counts = [0] * bins
    for s in scores:
        idx = min(int((s - lo) / step), bins - 1)
        counts[idx] += 1
    mx = max(counts)
    print(f"  {'score range':>20}  {'count':>5}  {'bar'}")
    for i, c in enumerate(counts):
        lo_b = lo + i * step
        hi_b = lo_b + step
        bar  = "█" * int(c / mx * width) if mx else ""
        print(f"  {lo_b:9.3f}–{hi_b:<9.3f}  {c:5d}  {bar}")
    pct = [np.percentile(scores, p) for p in (10, 25, 50, 75, 90)]
    print(f"\n  p10={pct[0]:.3f}  p25={pct[1]:.3f}  "
          f"median={pct[2]:.3f}  p75={pct[3]:.3f}  p90={pct[4]:.3f}")


# ===========================================================================
# Main
# ===========================================================================

def main():
    print("=" * 70)
    print("Ring Edge Sharpness Diagnostic — false-positive score analysis")
    print("=" * 70)
    print(f"Pipeline: param2={PARAM2}, arc_min={ARC_MIN}, ed_min={ED_MIN}, "
          f"cu_min={CU_MIN}, br_min={BR_MIN}")
    print(f"(Sharpness filter is OFF — collecting scores for ALL passing circles)\n")

    if not IMAGES_DIR.exists():
        print(f"[ERROR] IMAGES_DIR not found: {IMAGES_DIR.resolve()}")
        sys.exit(1)

    print(f"Caching {len(GROUND_TRUTH)} images…")
    t0    = time.time()
    cache = build_cache(IMAGES_DIR, GROUND_TRUTH)
    print(f"  {len(cache)} images ready in {time.time() - t0:.1f}s\n")

    # Run pipeline — collect scores per image, labelled as TP / FP
    # "extra" circles (det > truth) → FP-labelled (in score terms)
    # We assign FP to the lowest-sharpness circles first (worst candidates)

    per_image_results = []  # (fname, truth, det, circles_with_scores)
    all_fp_scores  = []
    all_tp_scores  = []

    t0 = time.time()
    for fname, sm, edges, hsv, sc, truth in cache:
        min_r = max(5,  int(30  * sc))
        max_r = max(10, int(220 * sc))
        min_d = max(10, int(MIN_DIST * sc))

        raw = cv2.HoughCircles(sm, cv2.HOUGH_GRADIENT,
                               dp=1, minDist=min_d,
                               param1=PARAM1, param2=PARAM2,
                               minRadius=min_r, maxRadius=max_r)

        circles = ([] if raw is None
                   else np.uint16(np.around(raw[0])).tolist())
        circles = _nms(circles, NMS_THRESH)

        scored = []
        for x, y, r in circles:
            x, y, r = int(x), int(y), int(r)
            arc  = _arc(edges, x, y, r)
            ed   = _edge_density(edges, x, y, r)
            cu   = _cu(hsv, x, y, r)
            br   = _brightness(hsv, x, y, r)

            if arc < ARC_MIN: continue
            if ed  < ED_MIN:  continue
            if cu  < CU_MIN:  continue
            if br  < BR_MIN:  continue

            sh, sh_ring, sh_inner, sh_outer = _ring_edge_sharpness(edges, x, y, r)
            scored.append({
                "x": x, "y": y, "r": r,
                "arc": arc, "ed": ed, "cu": cu, "br": br,
                "sh": sh, "sh_ring": sh_ring,
                "sh_inner": sh_inner, "sh_outer": sh_outer,
            })

        det = len(scored)

        # Label: sort by sharpness ascending — the lowest-sharpness ones are
        # the most likely false positives when det > truth
        scored_sorted = sorted(scored, key=lambda c: c["sh"])
        n_fp = max(0, det - truth)
        n_tp = det - n_fp

        fp_circles = scored_sorted[:n_fp]
        tp_circles = scored_sorted[n_fp:]

        for c in fp_circles:
            all_fp_scores.append(c["sh"])
        for c in tp_circles:
            all_tp_scores.append(c["sh"])

        per_image_results.append((fname, truth, det, n_fp, scored_sorted))

    print(f"Processing done in {time.time() - t0:.1f}s\n")

    # -----------------------------------------------------------------------
    # Per-image detail for worst overcounting images
    # -----------------------------------------------------------------------
    overcounts = [(fname, truth, det, n_fp, scored)
                  for fname, truth, det, n_fp, scored in per_image_results
                  if det > truth]
    overcounts.sort(key=lambda x: -(x[2] - x[1]))

    print("=" * 70)
    print(f"TOP {TOP_WORST} WORST OVERCOUNTING IMAGES — ring sharpness scores")
    print("=" * 70)

    for fname, truth, det, n_fp, scored_sorted in overcounts[:TOP_WORST]:
        diff = det - truth
        print(f"\n  {fname}  det={det}  truth={truth}  +{diff}")
        print(f"  {'status':>6}  {'x':>5}  {'y':>5}  {'r':>4}  "
              f"{'arc':>5}  {'ed':>5}  {'cu':>5}  {'br':>5}  {'sharpness':>9}  "
              f"{'ring':>6}  {'inner':>6}  {'outer':>6}")
        print(f"  {'-'*80}")
        for i, c in enumerate(scored_sorted):
            status = "  FP?" if i < n_fp else "  ok "
            print(f"  {status}  {c['x']:5d}  {c['y']:5d}  {c['r']:4d}  "
                  f"{c['arc']:5.2f}  {c['ed']:5.2f}  {c['cu']:5.2f}  "
                  f"{c['br']:5.0f}  {c['sh']:9.3f}  "
                  f"{c['sh_ring']:6.3f}  {c['sh_inner']:6.3f}  {c['sh_outer']:6.3f}")

    # -----------------------------------------------------------------------
    # Score distributions
    # -----------------------------------------------------------------------
    print(f"\n{'=' * 70}")
    print("FALSE-POSITIVE sharpness score distribution")
    print(f"  (n={len(all_fp_scores)} circles from overcounting images)")
    print("=" * 70)
    _histogram(all_fp_scores, label="FP")

    print(f"\n{'=' * 70}")
    print("TRUE-POSITIVE sharpness score distribution (for comparison)")
    print(f"  (n={len(all_tp_scores)} circles)")
    print("=" * 70)
    _histogram(all_tp_scores, label="TP")

    # -----------------------------------------------------------------------
    # Threshold recommendation
    # -----------------------------------------------------------------------
    print(f"\n{'=' * 70}")
    print("THRESHOLD RECOMMENDATION")
    print("=" * 70)

    if all_fp_scores and all_tp_scores:
        fp_p75 = np.percentile(all_fp_scores, 75)
        fp_p90 = np.percentile(all_fp_scores, 90)
        tp_p10 = np.percentile(all_tp_scores, 10)
        tp_p25 = np.percentile(all_tp_scores, 25)

        print(f"\n  FP scores:  p75={fp_p75:.3f}  p90={fp_p90:.3f}")
        print(f"  TP scores:  p10={tp_p10:.3f}  p25={tp_p25:.3f}")

        gap = tp_p10 - fp_p90
        print(f"\n  Separation gap (TP_p10 - FP_p90) = {gap:.3f}")

        if gap > 0:
            midpoint = (fp_p90 + tp_p10) / 2
            print(f"\n  ✓ Clear separation exists.")
            print(f"  Recommended sharpness_min = {midpoint:.2f}  "
                  f"(midpoint of gap: {fp_p90:.2f} → {tp_p10:.2f})")
            print(f"  Conservative (fewer FP, risk more FN): {fp_p90:.2f}")
            print(f"  Aggressive   (fewer FN, risk more FP): {tp_p10:.2f}")
        elif gap > -0.3:
            print(f"\n  ⚠ Partial overlap — some FP and TP scores intermix.")
            print(f"  Try sharpness_min = {fp_p75:.2f} (FP p75) in Phase 3 sweep.")
            print(f"  Expect some collateral TP loss at this threshold.")
        else:
            print(f"\n  ✗ Heavy overlap — sharpness alone cannot cleanly separate")
            print(f"  FP from TP at the current ring radii (0.55r / 1.55r).")
            print(f"  Consider adjusting probe radii or combining with param2 increase.")
    else:
        print("  Insufficient data for recommendation.")

    print(f"\n{'=' * 70}")
    print("Run tune_params_phase3.py for the full sweep with these values.")
    print("=" * 70)


if __name__ == "__main__":
    main()
