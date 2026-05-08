"""
scripts/generate_step_images.py

Runs the coin-detection pipeline on each image and saves a single grid PNG
(all steps side-by-side) to tester/step_by_step_output/<image_stem>_steps.png

Usage:
    python scripts/generate_step_images.py                      # all jpgs in data/complex_tests/
    python scripts/generate_step_images.py path/to/folder/      # all jpgs in that folder
    python scripts/generate_step_images.py path/to/image.jpg    # single image
"""

import sys
import pathlib
import cv2
import numpy as np

ROOT = pathlib.Path(__file__).resolve().parent.parent
sys.path.insert(0, str(ROOT))

from src.preprocess import (
    _scale_for_processing,
    enhance_contrast,
    preprocess,
    build_enhanced_edges,
    compute_background_edge_density,
)
from src.detect import (
    _hough_candidates,
    non_maximum_suppression,
    _arc_coverage,
    _edge_density_on_ring,
    _color_uniformity_inside,
    _ring_edge_sharpness,
    _edge_density_at_radius,
    filter_false_positives,
    _merge_by_arc_coverage,
    _remove_cluster_wrappers,
    _remove_nested_circles,
)

CELL_W = 480
CELL_H = 360
COLS   = 4
LABEL_H = 28
FONT    = cv2.FONT_HERSHEY_SIMPLEX


def _cell(img, label):
    """Resize img to CELL_W x CELL_H, convert to BGR, add a label bar."""
    if img.ndim == 2:
        img = cv2.cvtColor(img, cv2.COLOR_GRAY2BGR)
    frame = cv2.resize(img, (CELL_W, CELL_H), interpolation=cv2.INTER_AREA)
    bar = np.zeros((LABEL_H, CELL_W, 3), np.uint8)
    cv2.putText(bar, label, (6, LABEL_H - 7), FONT, 0.52, (220, 220, 220), 1, cv2.LINE_AA)
    return np.vstack([bar, frame])


def _build_grid(cells):
    """Arrange cells into a COLS-wide grid."""
    rows = []
    for i in range(0, len(cells), COLS):
        row_cells = cells[i:i + COLS]
        # pad last row if needed
        while len(row_cells) < COLS:
            row_cells.append(np.zeros((LABEL_H + CELL_H, CELL_W, 3), np.uint8))
        rows.append(np.hstack(row_cells))
    return np.vstack(rows)


def run(img_path: str, out_path: pathlib.Path):
    """Process one image, write one grid PNG to out_path. Returns coin count."""

    def draw_circles(base, circles, color, thickness=2):
        out = base.copy()
        for (x, y, r) in circles:
            cv2.circle(out, (x, y), r, color, thickness)
            cv2.circle(out, (x, y), 3, color, -1)
        return out

    def highlight(base, mask, color, alpha=0.45):
        ov = base.copy()
        ov[mask > 0] = color
        return cv2.addWeighted(ov, alpha, base, 1 - alpha, 0)

    image = cv2.imread(img_path)
    if image is None:
        print(f"  SKIP (cannot read): {img_path}")
        return 0

    cells = []

    # ── step 00  original ────────────────────────────────────────────────────
    cells.append(_cell(image, "00 – original"))

    # ── step 01  downscaled ──────────────────────────────────────────────────
    small, scale = _scale_for_processing(image)
    cells.append(_cell(small, "01 – downscaled"))

    # ── step 02  CLAHE ───────────────────────────────────────────────────────
    clahe_bgr = enhance_contrast(small, method="clahe")
    side = np.hstack([small, clahe_bgr])
    cv2.putText(side, "before", (6, 22), FONT, 0.55, (0,255,255), 1)
    cv2.putText(side, "after CLAHE", (small.shape[1]+6, 22), FONT, 0.55, (0,255,255), 1)
    cells.append(_cell(side, "02 – CLAHE"))

    # ── step 03  bilateral + gaussian ────────────────────────────────────────
    gray_smooth = preprocess(small)
    cells.append(_cell(gray_smooth, "03 – bilateral + gaussian"))

    # ── step 04  enhanced edges ──────────────────────────────────────────────
    edges = build_enhanced_edges(image)
    cells.append(_cell(edges, "04 – enhanced edges"))

    # ── step 05  background edge density ─────────────────────────────────────
    bg_density = compute_background_edge_density(image, border_px=60)
    h, w = image.shape[:2]
    bp = min(60, h // 3, w // 3)
    dv = image.copy()
    ov = dv.copy()
    ov[:bp, :]          = (0, 0, 200)
    ov[h-bp:, :]        = (0, 0, 200)
    ov[bp:h-bp, :bp]    = (0, 0, 200)
    ov[bp:h-bp, w-bp:]  = (0, 0, 200)
    dv = cv2.addWeighted(ov, 0.35, dv, 0.65, 0)
    cv2.putText(dv, f"bg_density={bg_density:.4f}", (10, h-14),
                FONT, 0.7, (0,255,255), 2)
    cells.append(_cell(dv, "05 – bg edge density"))

    # ── step 06  Hough candidates ─────────────────────────────────────────────
    hough_cands = _hough_candidates(image, bg_edge_density=bg_density)
    pv = draw_circles(image, hough_cands, (255, 80, 0))
    cv2.putText(pv, f"Hough: {len(hough_cands)} cands", (10, 30),
                FONT, 0.8, (255, 80, 0), 2)
    cells.append(_cell(pv, f"06 – Hough ({len(hough_cands)} candidates)"))

    fp_edges = build_enhanced_edges(image)
    demo = sorted(hough_cands, key=lambda c: c[2])[len(hough_cands)//2] if hough_cands else None

    if demo:
        x, y, r = int(demo[0]), int(demo[1]), int(demo[2])
        ring_w = 8

        # ── step 07  arc coverage ─────────────────────────────────────────────
        av = cv2.cvtColor(fp_edges, cv2.COLOR_GRAY2BGR)
        for s in range(36):
            angle = -np.pi + (s + 0.5) * 2 * np.pi / 36
            a0    = -np.pi + s * 2 * np.pi / 36
            a1    = a0 + 2 * np.pi / 36
            hit   = False
            for a in np.linspace(a0, a1, 12):
                for rr in range(max(r - ring_w, 1), r + ring_w + 1):
                    sx, sy = int(round(x + rr*np.cos(a))), int(round(y + rr*np.sin(a)))
                    if 0 <= sy < fp_edges.shape[0] and 0 <= sx < fp_edges.shape[1] and fp_edges[sy, sx] > 0:
                        hit = True; break
                if hit: break
            col = (0, 200, 0) if hit else (0, 0, 200)
            cv2.line(av, (x, y),
                     (int(x + (r+ring_w+6)*np.cos(angle)),
                      int(y + (r+ring_w+6)*np.sin(angle))), col, 2)
        cv2.circle(av, (x, y), r, (0, 255, 255), 2)
        arc_cov = _arc_coverage(fp_edges, x, y, r)
        cv2.putText(av, f"arc_cov={arc_cov:.2f} (min 0.35)", (10, 30),
                    FONT, 0.7, (0, 255, 255), 2)
        cells.append(_cell(av, f"07 – arc coverage  [{arc_cov:.2f}]"))

        # ── step 08  edge density on ring ─────────────────────────────────────
        rv  = cv2.cvtColor(fp_edges, cv2.COLOR_GRAY2BGR)
        mo  = np.zeros(fp_edges.shape, np.uint8)
        mi  = np.zeros(fp_edges.shape, np.uint8)
        cv2.circle(mo, (x, y), r + ring_w, 255, -1)
        cv2.circle(mi, (x, y), max(r - ring_w, 0), 255, -1)
        rv[cv2.subtract(mo, mi) > 0] = (0, 180, 255)
        cv2.circle(rv, (x, y), r, (0, 255, 255), 1)
        ed = _edge_density_on_ring(fp_edges, x, y, r)
        cv2.putText(rv, f"edge_density={ed:.3f} (min 0.08)", (10, 30),
                    FONT, 0.7, (0, 255, 255), 2)
        cells.append(_cell(rv, f"08 – edge density on ring  [{ed:.3f}]"))

        # ── step 09  colour uniformity ────────────────────────────────────────
        im = np.zeros(image.shape[:2], np.uint8)
        cv2.circle(im, (x, y), max(int(r * 0.80), 1), 255, -1)
        cu_v = highlight(image, im, (255, 180, 0))
        cv2.circle(cu_v, (x, y), r, (0, 255, 255), 2)
        cu = _color_uniformity_inside(image, x, y, r)
        cv2.putText(cu_v, f"color_unif={cu:.3f} (min 0.28)", (10, 30),
                    FONT, 0.7, (0, 255, 255), 2)
        cells.append(_cell(cu_v, f"09 – colour uniformity  [{cu:.3f}]"))

        # ── step 10  ring sharpness ───────────────────────────────────────────
        sv   = cv2.cvtColor(fp_edges, cv2.COLOR_GRAY2BGR)
        ri_s = max(int(r * 0.55), 1)
        ro_s = int(r * 1.55)
        cv2.circle(sv, (x, y), ri_s, (255, 100, 0), 2)
        cv2.circle(sv, (x, y), r,    (0, 255, 0),   2)
        cv2.circle(sv, (x, y), ro_s, (0, 80, 255),  2)
        sharpness = _ring_edge_sharpness(fp_edges, x, y, r)
        ring_d    = _edge_density_at_radius(fp_edges, x, y, r)
        inner_d   = _edge_density_at_radius(fp_edges, x, y, ri_s)
        outer_d   = _edge_density_at_radius(fp_edges, x, y, ro_s)
        cv2.putText(sv, f"ring={ring_d:.3f} in={inner_d:.3f} out={outer_d:.3f}",
                    (10, 30), FONT, 0.62, (0, 255, 255), 2)
        cv2.putText(sv, f"sharpness={sharpness:.2f}", (10, 58),
                    FONT, 0.7, (0, 255, 0), 2)
        cells.append(_cell(sv, f"10 – ring sharpness  [{sharpness:.2f}]"))

    else:
        for label in ("07 – arc coverage", "08 – edge density on ring",
                      "09 – colour uniformity", "10 – ring sharpness"):
            cells.append(_cell(np.zeros((CELL_H, CELL_W, 3), np.uint8), label + "  [no demo coin]"))

    # ── step 11  final detections ─────────────────────────────────────────────
    filtered, fp_edges2 = filter_false_positives(image, hough_cands, bg_edge_density=bg_density)
    deduped  = non_maximum_suppression(filtered, overlap_thresh=0.4)
    merged   = _merge_by_arc_coverage(deduped, fp_edges2)
    merged   = _remove_cluster_wrappers(merged, fp_edges2)
    final    = _remove_nested_circles(merged, proximity_ratio=0.60)
    fv = image.copy()
    for (fx, fy, fr) in final:
        cv2.circle(fv, (fx, fy), fr, (0, 220, 0), 2)
        cv2.circle(fv, (fx, fy), 4,  (0, 220, 0), -1)
        cv2.putText(fv, f"r={fr}", (fx-18, fy-fr-6), FONT, 0.5, (0, 255, 255), 1)
    cv2.putText(fv, f"Final: {len(final)} coins", (10, 30),
                FONT, 0.9, (0, 220, 0), 2)
    cells.append(_cell(fv, f"11 – final  ({len(final)} coins)"))

    # ── assemble and save grid ────────────────────────────────────────────────
    grid = _build_grid(cells)
    out_path.parent.mkdir(parents=True, exist_ok=True)
    cv2.imwrite(str(out_path), grid)
    return len(final)


def main():
    base_out = ROOT / "tester" / "step_by_step_output"

    if len(sys.argv) == 1:
        search_dir = ROOT / "data" / "complex_tests"
        paths = sorted(search_dir.glob("*.jpg"))
        if not paths:
            sys.exit(f"No jpg files found in {search_dir}")
        print(f"Found {len(paths)} images in {search_dir}\n")
    else:
        arg = pathlib.Path(sys.argv[1])
        if arg.is_dir():
            paths = sorted(arg.glob("*.jpg"))
            print(f"Found {len(paths)} images in {arg}\n")
        else:
            paths = [pathlib.Path(p) for p in sys.argv[1:]]

    summary = []
    for p in paths:
        out_path = base_out / f"{p.stem}_steps.png"
        print(f"── {p.name} ──")
        try:
            n = run(str(p), out_path)
            summary.append((p.name, n))
            print(f"   {n} coins  →  {out_path}")
        except Exception as e:
            print(f"   ERROR: {e}")
            summary.append((p.name, "ERROR"))

    print(f"\n{'─'*40}")
    print(f"{'IMAGE':<25}  COINS")
    for name, n in summary:
        print(f"  {name:<23}  {n}")
    print(f"\nOutputs saved to: {base_out}")


if __name__ == "__main__":
    main()
