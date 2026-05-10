"""
scripts/generate_step_images.py

Instruments the coin-detection pipeline on one test image and saves one PNG
per step to tester/step_by_step_output/<image_stem>/.

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


def run(img_path: str, out_dir: pathlib.Path):
    out_dir.mkdir(parents=True, exist_ok=True)

    def save(name, img):
        cv2.imwrite(str(out_dir / name), img)

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

    # step_00 original
    save("step_00_original.png", image)

    # step_01 downscaled
    small, scale = _scale_for_processing(image)
    save("step_01_downscaled.png", small)

    # step_02 CLAHE
    clahe_bgr = enhance_contrast(small, method="clahe")
    side = np.hstack([small, clahe_bgr])
    cv2.putText(side, "Before CLAHE", (10, 24), cv2.FONT_HERSHEY_SIMPLEX, 0.7, (0,255,255), 2)
    cv2.putText(side, "After CLAHE", (small.shape[1]+10, 24), cv2.FONT_HERSHEY_SIMPLEX, 0.7, (0,255,255), 2)
    save("step_02_clahe.png", side)

    # step_03 bilateral + gaussian
    gray_smooth = preprocess(small)
    save("step_03_bilateral_gaussian.png", gray_smooth)

    # step_04 enhanced edges
    edges = build_enhanced_edges(image)
    save("step_04_enhanced_edges.png", edges)

    # step_05 background edge density
    bg_density = compute_background_edge_density(image, border_px=60)
    h, w = image.shape[:2]
    bp = min(60, h // 3, w // 3)
    dv = image.copy()
    ov = dv.copy()
    ov[:bp, :] = ov[h-bp:, :] = ov[bp:h-bp, :bp] = ov[bp:h-bp, w-bp:] = (0, 0, 200)
    dv = cv2.addWeighted(ov, 0.35, dv, 0.65, 0)
    cv2.putText(dv, f"bg_edge_density = {bg_density:.4f}", (10, h-14),
                cv2.FONT_HERSHEY_SIMPLEX, 0.75, (0,255,255), 2)
    save("step_05_bg_density.png", dv)

    # step_06 Hough candidates
    hough_cands = _hough_candidates(image, bg_edge_density=bg_density)
    pv = draw_circles(image, hough_cands, (255, 80, 0))
    cv2.putText(pv, f"Hough: {len(hough_cands)} candidates", (10, 30),
                cv2.FONT_HERSHEY_SIMPLEX, 0.8, (255,80,0), 2)
    save("step_06_hough_circles.png", pv)

    fp_edges = build_enhanced_edges(image)
    demo = sorted(hough_cands, key=lambda c: c[2])[len(hough_cands)//2] if hough_cands else None

    if demo:
        x, y, r = int(demo[0]), int(demo[1]), int(demo[2])
        ring_w = 8

        # step_07 arc coverage
        av = cv2.cvtColor(fp_edges, cv2.COLOR_GRAY2BGR)
        for s in range(36):
            angle = -np.pi + (s + 0.5) * 2 * np.pi / 36
            a0 = -np.pi + s * 2 * np.pi / 36
            a1 = a0 + 2 * np.pi / 36
            hit = False
            for a in np.linspace(a0, a1, 12):
                for rr in range(max(r-ring_w, 1), r+ring_w+1):
                    sx, sy = int(round(x+rr*np.cos(a))), int(round(y+rr*np.sin(a)))
                    if 0<=sy<fp_edges.shape[0] and 0<=sx<fp_edges.shape[1] and fp_edges[sy,sx]>0:
                        hit = True; break
                if hit: break
            col = (0,200,0) if hit else (0,0,200)
            cv2.line(av, (x,y),
                     (int(x+(r+ring_w+6)*np.cos(angle)), int(y+(r+ring_w+6)*np.sin(angle))),
                     col, 2)
        cv2.circle(av, (x,y), r, (0,255,255), 2)
        arc_cov = _arc_coverage(fp_edges, x, y, r)
        cv2.putText(av, f"arc_coverage={arc_cov:.2f} (min 0.35)", (10,30),
                    cv2.FONT_HERSHEY_SIMPLEX, 0.7, (0,255,255), 2)
        save("step_07_arc_coverage.png", av)

        # step_08 edge density on ring
        rv = cv2.cvtColor(fp_edges, cv2.COLOR_GRAY2BGR)
        mo, mi = np.zeros(fp_edges.shape, np.uint8), np.zeros(fp_edges.shape, np.uint8)
        cv2.circle(mo, (x,y), r+ring_w, 255, -1)
        cv2.circle(mi, (x,y), max(r-ring_w,0), 255, -1)
        ring_mask = cv2.subtract(mo, mi)
        rv[ring_mask > 0] = (0, 180, 255)
        cv2.circle(rv, (x,y), r, (0,255,255), 1)
        ed = _edge_density_on_ring(fp_edges, x, y, r)
        cv2.putText(rv, f"edge_density={ed:.3f} (min 0.08)", (10,30),
                    cv2.FONT_HERSHEY_SIMPLEX, 0.7, (0,255,255), 2)
        save("step_08_edge_density.png", rv)

        # step_09 color uniformity
        im = np.zeros(image.shape[:2], np.uint8)
        cv2.circle(im, (x,y), max(int(r*0.80),1), 255, -1)
        cu_v = highlight(image, im, (255,180,0))
        cv2.circle(cu_v, (x,y), r, (0,255,255), 2)
        cu = _color_uniformity_inside(image, x, y, r)
        cv2.putText(cu_v, f"color_uniformity={cu:.3f} (min 0.28)", (10,30),
                    cv2.FONT_HERSHEY_SIMPLEX, 0.7, (0,255,255), 2)
        save("step_09_color_uniformity.png", cu_v)

        # step_10 ring sharpness
        sv = cv2.cvtColor(fp_edges, cv2.COLOR_GRAY2BGR)
        ri_s, ro_s = max(int(r*0.55),1), int(r*1.55)
        cv2.circle(sv, (x,y), ri_s, (255,100,0), 2)
        cv2.circle(sv, (x,y), r,    (0,255,0),   2)
        cv2.circle(sv, (x,y), ro_s, (0,80,255),  2)
        ring_d  = _edge_density_at_radius(fp_edges, x, y, r)
        inner_d = _edge_density_at_radius(fp_edges, x, y, ri_s)
        outer_d = _edge_density_at_radius(fp_edges, x, y, ro_s)
        sharpness = _ring_edge_sharpness(fp_edges, x, y, r)
        cv2.putText(sv, f"ring={ring_d:.3f}  inner={inner_d:.3f}  outer={outer_d:.3f}",
                    (10,30), cv2.FONT_HERSHEY_SIMPLEX, 0.65, (0,255,255), 2)
        cv2.putText(sv, f"sharpness={sharpness:.2f}", (10,58),
                    cv2.FONT_HERSHEY_SIMPLEX, 0.7, (0,255,0), 2)
        save("step_10_sharpness.png", sv)

    # step_11 final
    filtered, fp_edges2 = filter_false_positives(image, hough_cands, bg_edge_density=bg_density)
    deduped = non_maximum_suppression(filtered, overlap_thresh=0.4)
    merged  = _merge_by_arc_coverage(deduped, fp_edges2)
    merged  = _remove_cluster_wrappers(merged)
    final   = _remove_nested_circles(merged, proximity_ratio=0.60)
    fv = image.copy()
    for (fx, fy, fr) in final:
        cv2.circle(fv, (fx,fy), fr, (0,220,0), 2)
        cv2.circle(fv, (fx,fy), 4,  (0,220,0), -1)
        cv2.putText(fv, f"r={fr}", (fx-18, fy-fr-6), cv2.FONT_HERSHEY_SIMPLEX, 0.5, (0,255,255), 1)
    cv2.putText(fv, f"Final detections: {len(final)}", (10,30),
                cv2.FONT_HERSHEY_SIMPLEX, 0.9, (0,220,0), 2)
    save("step_11_final.png", fv)
    return len(final)


def main():
    base_out = ROOT / "tester" / "step_by_step_output"

    if len(sys.argv) == 1:
        # No args — process every jpg in data/complex_tests/
        search_dir = ROOT / "data" / "complex_tests"
        paths = sorted(search_dir.glob("*.jpg"))
        if not paths:
            sys.exit(f"No jpg files found in {search_dir}")
        print(f"Found {len(paths)} images in {search_dir}\n")
    else:
        arg = pathlib.Path(sys.argv[1])
        if arg.is_dir():
            # Directory passed — process all jpgs inside
            paths = sorted(arg.glob("*.jpg"))
            print(f"Found {len(paths)} images in {arg}\n")
        else:
            # One or more file paths passed
            paths = [pathlib.Path(p) for p in sys.argv[1:]]

    summary = []
    for p in paths:
        out = base_out / p.stem          # one subfolder per image
        print(f"── {p.name} ──")
        try:
            n = run(str(p), out)
            summary.append((p.name, n))
            print(f"   {n} coins  →  {out}")
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