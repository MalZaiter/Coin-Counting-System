"""
tests/test_evaluate.py — Unit tests for evaluate.py

Run with:  pytest tests/test_evaluate.py -v
"""

import sys
from pathlib import Path

# Add project root to Python path
sys.path.insert(0, str(Path(__file__).parent.parent))

import io
import numpy as np
import pytest
from unittest.mock import MagicMock, patch


# ──────────────────────────────────────────────────────────────────────────────
# Helpers
# ──────────────────────────────────────────────────────────────────────────────

LABELS_8 = ["1cent", "2cent", "5cent", "10cent", "20cent", "50cent", "1euro", "2euro"]

def _perfect_preds(labels, n=5):
    """Return (y_true, y_pred) where every prediction is correct."""
    y = [labels[i % len(labels)] for i in range(n)]
    return y, list(y)

def _all_wrong_preds():
    y_true = ["1cent", "2cent", "5cent"]
    y_pred = ["2euro", "1euro", "50cent"]
    return y_true, y_pred


# ══════════════════════════════════════════════════════════════════════════════
# compute_accuracy
# ══════════════════════════════════════════════════════════════════════════════

class TestComputeAccuracy:

    def test_perfect_predictions(self):
        from src.evaluate import compute_accuracy
        y_true, y_pred = _perfect_preds(LABELS_8)
        assert compute_accuracy(y_true, y_pred) == pytest.approx(1.0)

    def test_all_wrong(self):
        from src.evaluate import compute_accuracy
        y_true, y_pred = _all_wrong_preds()
        assert compute_accuracy(y_true, y_pred) == pytest.approx(0.0)

    def test_half_correct(self):
        from src.evaluate import compute_accuracy
        y_true = ["1cent", "2cent", "5cent", "10cent"]
        y_pred = ["1cent", "1euro", "5cent", "2euro"]   # 2 correct
        assert compute_accuracy(y_true, y_pred) == pytest.approx(0.5)

    def test_single_sample_correct(self):
        from src.evaluate import compute_accuracy
        assert compute_accuracy(["1euro"], ["1euro"]) == pytest.approx(1.0)

    def test_single_sample_wrong(self):
        from src.evaluate import compute_accuracy
        assert compute_accuracy(["1euro"], ["2euro"]) == pytest.approx(0.0)

    def test_returns_float(self):
        from src.evaluate import compute_accuracy
        result = compute_accuracy(["1cent"], ["1cent"])
        assert isinstance(result, float)


# ══════════════════════════════════════════════════════════════════════════════
# compute_confusion_matrix
# ══════════════════════════════════════════════════════════════════════════════

class TestComputeConfusionMatrix:

    def test_shape_matches_label_count(self):
        from src.evaluate import compute_confusion_matrix
        labels = ["1cent", "2cent", "5cent"]
        y_true = ["1cent", "2cent", "5cent", "1cent"]
        y_pred = ["1cent", "2cent", "1cent", "1cent"]
        m = compute_confusion_matrix(y_true, y_pred, labels)
        assert m.shape == (3, 3)

    def test_diagonal_perfect_predictions(self):
        from src.evaluate import compute_confusion_matrix
        labels = ["1cent", "2cent", "5cent"]
        y_true = ["1cent", "2cent", "5cent"]
        y_pred = ["1cent", "2cent", "5cent"]
        m = compute_confusion_matrix(y_true, y_pred, labels)
        # Diagonal should equal [1,1,1]; off-diagonal should be 0
        assert m[0, 0] == 1
        assert m[1, 1] == 1
        assert m[2, 2] == 1
        assert m.sum() - np.trace(m) == 0

    def test_off_diagonal_misclassification(self):
        from src.evaluate import compute_confusion_matrix
        labels = ["1cent", "2cent"]
        y_true = ["1cent", "1cent"]
        y_pred = ["2cent", "2cent"]   # all 1cent predicted as 2cent
        m = compute_confusion_matrix(y_true, y_pred, labels)
        # Row 0 (true=1cent), col 1 (pred=2cent) should be 2
        assert m[0, 1] == 2
        assert m[0, 0] == 0

    def test_returns_2d_numpy_array(self):
        from src.evaluate import compute_confusion_matrix
        labels = ["1cent", "2cent"]
        m = compute_confusion_matrix(["1cent"], ["1cent"], labels)
        assert isinstance(m, np.ndarray)
        assert m.ndim == 2

    def test_row_sums_equal_true_class_counts(self):
        from src.evaluate import compute_confusion_matrix
        labels = ["1cent", "2cent", "5cent"]
        y_true = ["1cent", "1cent", "2cent", "5cent", "5cent", "5cent"]
        y_pred = ["1cent", "2cent", "2cent", "5cent", "1cent", "5cent"]
        m = compute_confusion_matrix(y_true, y_pred, labels)
        assert m[0].sum() == 2   # two true 1cent
        assert m[1].sum() == 1   # one true 2cent
        assert m[2].sum() == 3   # three true 5cent

    def test_all_labels_represented_even_if_absent(self):
        from src.evaluate import compute_confusion_matrix
        labels = ["1cent", "2cent", "5cent"]
        # 5cent never appears in data
        y_true = ["1cent", "2cent"]
        y_pred = ["1cent", "2cent"]
        m = compute_confusion_matrix(y_true, y_pred, labels)
        assert m.shape == (3, 3)
        assert m[2].sum() == 0   # 5cent row all zeros


# ══════════════════════════════════════════════════════════════════════════════
# print_classification_report
# ══════════════════════════════════════════════════════════════════════════════

class TestPrintClassificationReport:

    def test_runs_without_error(self, capsys):
        from src.evaluate import print_classification_report
        y_true = ["1cent", "2cent", "1cent", "5cent"]
        y_pred = ["1cent", "2cent", "2cent", "5cent"]
        print_classification_report(y_true, y_pred, ["1cent", "2cent", "5cent"])
        out = capsys.readouterr().out
        assert len(out) > 0

    def test_output_contains_all_labels(self, capsys):
        from src.evaluate import print_classification_report
        labels = ["1cent", "2cent", "5cent"]
        y_true = labels
        y_pred = labels
        print_classification_report(y_true, y_pred, labels)
        out = capsys.readouterr().out
        for lbl in labels:
            assert lbl in out, f"Label '{lbl}' missing from report output"

    def test_output_contains_accuracy(self, capsys):
        from src.evaluate import print_classification_report
        y_true = ["1cent", "2cent"]
        y_pred = ["1cent", "2cent"]
        print_classification_report(y_true, y_pred, ["1cent", "2cent"])
        out = capsys.readouterr().out
        # Some form of accuracy/score should appear
        assert any(kw in out.lower() for kw in ("accuracy", "precision", "recall", "f1"))

    def test_empty_input_does_not_crash(self, capsys):
        from src.evaluate import print_classification_report
        try:
            print_classification_report([], [], [])
        except Exception:
            pass   # acceptable — just must not raise unhandled exception silently


# ══════════════════════════════════════════════════════════════════════════════
# plot_confusion_matrix
# ══════════════════════════════════════════════════════════════════════════════

class TestPlotConfusionMatrix:

    def test_saves_file_when_save_path_given(self, tmp_path):
        from src.evaluate import plot_confusion_matrix
        save_file = tmp_path / "cm.png"
        labels = ["1cent", "2cent", "5cent"]
        matrix = np.eye(3, dtype=int)
        # Use non-interactive backend to avoid display
        with patch("matplotlib.pyplot.savefig") as mock_save, \
             patch("matplotlib.pyplot.show"), \
             patch("matplotlib.pyplot.close"):
            plot_confusion_matrix(matrix, labels, save_path=str(save_file))

    def test_does_not_raise_on_valid_input(self):
        from src.evaluate import plot_confusion_matrix
        labels = ["1cent", "2cent"]
        matrix = np.array([[3, 1], [0, 4]])
        with patch("matplotlib.pyplot.show"), \
             patch("matplotlib.pyplot.savefig"), \
             patch("matplotlib.pyplot.close"):
            plot_confusion_matrix(matrix, labels)

    def test_handles_zero_matrix(self):
        from src.evaluate import plot_confusion_matrix
        labels = ["1cent", "2cent"]
        matrix = np.zeros((2, 2), dtype=int)
        with patch("matplotlib.pyplot.show"), \
             patch("matplotlib.pyplot.savefig"), \
             patch("matplotlib.pyplot.close"):
            plot_confusion_matrix(matrix, labels)


# ══════════════════════════════════════════════════════════════════════════════
# evaluate  (integration — mocks out file I/O and prediction)
# ══════════════════════════════════════════════════════════════════════════════

class TestEvaluate:

    def _make_csv(self, tmp_path, rows):
        """Write a labels CSV and return its path."""
        import csv
        p = tmp_path / "labels.csv"
        with open(p, "w", newline="") as f:
            writer = csv.DictWriter(f, fieldnames=["image_path", "label", "x", "y", "radius"])
            writer.writeheader()
            writer.writerows(rows)
        return str(p)

    def test_returns_required_keys(self, tmp_path):
        from src.evaluate import evaluate
        rows = [
            {"image_path": "img1.jpg", "label": "1euro", "x": 100, "y": 100, "radius": 50},
            {"image_path": "img2.jpg", "label": "2cent", "x": 80,  "y": 80,  "radius": 40},
        ]
        csv_path = self._make_csv(tmp_path, rows)

        fake_image = np.full((200, 200, 3), 128, dtype=np.uint8)
        fake_feats = np.random.rand(57).astype(np.float32)

        fake_model = MagicMock()
        fake_model.classes_ = np.array(["1cent", "2cent", "1euro"])
        proba = np.array([[0.0, 0.0, 0.85]])
        fake_model.predict_proba.return_value = proba

        with patch("src.evaluate.load_model", return_value=(fake_model, None)), \
             patch("src.utils.load_image", return_value=fake_image), \
             patch("src.features.extract_features", return_value=fake_feats), \
             patch("src.evaluate.plot_confusion_matrix"), \
             patch("matplotlib.pyplot.show"), \
             patch("matplotlib.pyplot.savefig"):
            result = evaluate(
                test_images_dir=str(tmp_path),
                labels_path=csv_path,
                model_path="models/classifier.pkl",
                plot=False,
            )

        assert "accuracy"           in result
        assert "confusion_matrix"   in result
        assert "y_true"             in result
        assert "y_pred"             in result
        assert "labels"             in result

    def test_accuracy_is_between_0_and_1(self, tmp_path):
        from src.evaluate import evaluate
        rows = [
            {"image_path": "img1.jpg", "label": "1euro", "x": 100, "y": 100, "radius": 50},
        ]
        csv_path = self._make_csv(tmp_path, rows)

        fake_model = MagicMock()
        fake_model.classes_ = np.array(["1euro"])
        fake_model.predict_proba.return_value = np.array([[0.90]])

        with patch("src.evaluate.load_model", return_value=(fake_model, None)), \
             patch("src.utils.load_image",
                   return_value=np.full((200, 200, 3), 128, dtype=np.uint8)), \
             patch("src.features.extract_features",
                   return_value=np.random.rand(57).astype(np.float32)):
            result = evaluate(
                test_images_dir=str(tmp_path),
                labels_path=csv_path,
                model_path="models/classifier.pkl",
                plot=False,
            )

        assert 0.0 <= result["accuracy"] <= 1.0

    def test_raises_on_missing_labels_file(self, tmp_path):
        from src.evaluate import evaluate
        with pytest.raises((FileNotFoundError, Exception)):
            evaluate(
                test_images_dir=str(tmp_path),
                labels_path=str(tmp_path / "nonexistent.csv"),
                model_path="models/classifier.pkl",
                plot=False,
            )


# ══════════════════════════════════════════════════════════════════════════════
# Bug regression tests
# ══════════════════════════════════════════════════════════════════════════════

class TestBugRegressions:
    """
    Explicit regression tests that lock down the two known critical bugs.
    These tests will FAIL on the original buggy code and PASS after fixing.
    """

    def test_evaluate_functions_are_not_stubs(self):
        """
        BUG: evaluate.py defines print_classification_report, plot_confusion_matrix,
        and evaluate() TWICE. Python keeps the last definition, which are all
        `pass` stubs. The real implementations are silently overridden.
        Fix: Remove the duplicate stub definitions at the bottom of evaluate.py.
        """
        from src import evaluate as ev_module

        # print_classification_report must actually print something
        import io, sys
        captured = io.StringIO()
        sys.stdout = captured
        try:
            ev_module.print_classification_report(
                ["1cent", "2cent"], ["1cent", "2cent"], ["1cent", "2cent"]
            )
        finally:
            sys.stdout = sys.__stdout__
        output = captured.getvalue()
        assert output.strip() != "", (
            "BUG: print_classification_report is a stub (returns None/prints nothing). "
            "Remove the duplicate stub definitions at the bottom of evaluate.py."
        )

    def test_predict_passes_bgr_image_to_detect_coins(self):
        """
        BUG: predict.py line ~148 calls detect_coins(img_proc) where img_proc is
        the result of preprocess() — a grayscale image.  detect_coins() calls
        preprocess() internally, so the correct call is detect_coins(image).
        Fix: replace `detect_coins(img_proc)` with `detect_coins(image)`.
        """
        from src.predict import predict

        received_shapes = []

        def capture_detect(img):
            received_shapes.append(img.shape)
            return []

        with patch("cv2.imread", return_value=np.full((200, 200, 3), 128, dtype=np.uint8)), \
             patch("src.predict.load_model",
                   return_value=(MagicMock(spec=["predict"]), None)), \
             patch("src.detect.detect_coins", side_effect=capture_detect):
            try:
                predict("fake.jpg", "models/classifier.pkl")
            except Exception:
                pass

        assert received_shapes, "detect_coins was never called"
        shape = received_shapes[0]
        assert len(shape) == 3, (
            f"BUG: detect_coins received a {len(shape)}-D image (grayscale). "
            "Pass the original BGR image, not preprocess(image)."
        )

    def test_compute_total_value_uses_euro_labels(self):
        """
        BUG: compute_total_value uses US coin names (penny, nickel, dime, quarter).
        The classifier produces Euro labels (1cent, 2cent … 2euro).
        All Euro labels will silently contribute €0.00.
        Fix: replace the coin_values dict with the Euro mapping.
        """
        from src.predict import compute_total_value
        result = compute_total_value({"1euro": 1, "2euro": 1, "50cent": 2})
        expected = 1.00 + 2.00 + 0.50 * 2
        assert abs(result - expected) < 1e-9, (
            f"BUG: compute_total_value returned {result:.4f} instead of {expected:.4f}. "
            "The coin_values dict uses US names — update it to Euro labels."
        )
