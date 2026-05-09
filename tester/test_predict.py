"""
tests/test_predict.py — Unit tests for predict.py

Run with:  pytest tests/test_predict.py -v
"""

import sys
from pathlib import Path

# Add project root to Python path
sys.path.insert(0, str(Path(__file__).parent.parent))

import numpy as np
import pytest
from unittest.mock import MagicMock, patch, call


# ──────────────────────────────────────────────────────────────────────────────
# Helpers
# ──────────────────────────────────────────────────────────────────────────────

def _blank_bgr(h=200, w=200):
    """Return a plain grey BGR image."""
    return np.full((h, w, 3), 128, dtype=np.uint8)


def _fake_model(label="1euro"):
    """Mock sklearn model that always predicts `label`."""
    m = MagicMock()
    m.predict.return_value = np.array([label])
    m.classes_ = np.array(["1cent", "2cent", "5cent", "10cent",
                            "20cent", "50cent", "1euro", "2euro"])
    proba = np.zeros(8)
    proba[6] = 0.85          # 1euro with 85% confidence
    m.predict_proba.return_value = proba.reshape(1, -1)
    return m


def _fake_scaler():
    """Mock StandardScaler that returns its input unchanged."""
    s = MagicMock()
    s.transform.side_effect = lambda x: x
    return s


# ══════════════════════════════════════════════════════════════════════════════
# load_model
# ══════════════════════════════════════════════════════════════════════════════

class TestLoadModel:

    @patch("joblib.load")
    def test_returns_model_and_none_scaler_when_no_scaler_path(self, mock_load):
        from src.predict import load_model
        mock_load.return_value = _fake_model()
        model, scaler = load_model("models/classifier.pkl")
        assert model is not None
        assert scaler is None
        mock_load.assert_called_once_with("models/classifier.pkl")

    @patch("joblib.load")
    def test_returns_both_model_and_scaler(self, mock_load):
        from src.predict import load_model
        fake_model = _fake_model()
        fake_scaler = _fake_scaler()
        mock_load.side_effect = [fake_model, fake_scaler]
        model, scaler = load_model("models/classifier.pkl", "models/scaler.pkl")
        assert model is fake_model
        assert scaler is fake_scaler

    @patch("joblib.load", side_effect=FileNotFoundError("no file"))
    def test_raises_when_model_missing(self, mock_load):
        from src.predict import load_model
        with pytest.raises((FileNotFoundError, Exception)):
            load_model("nonexistent.pkl")


# ══════════════════════════════════════════════════════════════════════════════
# predict_coin
# ══════════════════════════════════════════════════════════════════════════════

class TestPredictCoin:

    def test_returns_string_label(self):
        from src.predict import predict_coin
        model = _fake_model("2euro")
        fv = np.random.rand(57).astype(np.float32)
        label, conf = predict_coin(fv, model)
        assert isinstance(label, str)
        assert label == "2euro"

    def test_confidence_above_threshold_accepted(self):
        from src.predict import predict_coin
        model = _fake_model("1euro")
        fv = np.random.rand(57).astype(np.float32)
        label, conf = predict_coin(fv, model, confidence_threshold=0.55)
        assert label == "1euro"
        assert conf >= 0.55

    def test_low_confidence_returns_not_a_coin(self):
        from src.predict import predict_coin, NOT_A_COIN_LABEL
        model = MagicMock()
        model.classes_ = np.array(["1cent", "2cent", "5cent"])
        # Spread probability equally — no winner
        model.predict_proba.return_value = np.array([[0.34, 0.33, 0.33]])
        fv = np.random.rand(57).astype(np.float32)
        label, conf = predict_coin(fv, model, confidence_threshold=0.55)
        assert label == NOT_A_COIN_LABEL
        assert conf < 0.55

    def test_applies_scaler_before_predicting(self):
        from src.predict import predict_coin
        model = _fake_model("5cent")
        scaler = _fake_scaler()
        fv = np.random.rand(57).astype(np.float32)
        predict_coin(fv, model, scaler=scaler)
        scaler.transform.assert_called_once()

    def test_1d_feature_vector_is_reshaped(self):
        from src.predict import predict_coin
        model = _fake_model("10cent")
        fv = np.random.rand(57).astype(np.float32)   # 1-D
        label, _ = predict_coin(fv, model)
        # model.predict_proba should receive shape (1, 57)
        call_arg = model.predict_proba.call_args[0][0]
        assert call_arg.shape == (1, 57)

    def test_model_without_predict_proba_falls_back(self):
        from src.predict import predict_coin
        model = MagicMock(spec=["predict"])          # no predict_proba
        model.predict.return_value = np.array(["2cent"])
        fv = np.random.rand(57).astype(np.float32)
        label, conf = predict_coin(fv, model)
        assert label == "2cent"
        assert conf == 1.0


# ══════════════════════════════════════════════════════════════════════════════
# draw_results
# ══════════════════════════════════════════════════════════════════════════════

class TestDrawResults:

    def test_returns_same_shape_as_input(self):
        from src.predict import draw_results
        img = _blank_bgr()
        coins = [(100, 100, 40)]
        labels = ["1euro"]
        result = draw_results(img, coins, labels)
        assert result.shape == img.shape

    def test_does_not_modify_original_image(self):
        from src.predict import draw_results
        img = _blank_bgr()
        original = img.copy()
        draw_results(img, [(100, 100, 40)], ["2euro"])
        np.testing.assert_array_equal(img, original)

    def test_handles_empty_coins(self):
        from src.predict import draw_results
        img = _blank_bgr()
        result = draw_results(img, [], [])
        assert result.shape == img.shape

    def test_handles_more_labels_than_coins(self):
        from src.predict import draw_results
        img = _blank_bgr()
        result = draw_results(img, [(50, 50, 20)], ["1cent", "2cent", "5cent"])
        assert result.shape == img.shape

    def test_coin_label_changes_pixel_values(self):
        """Drawing on an image should change at least some pixels."""
        from src.predict import draw_results
        img = _blank_bgr()
        result = draw_results(img, [(100, 100, 40)], ["1euro"])
        assert not np.array_equal(img, result)


# ══════════════════════════════════════════════════════════════════════════════
# count_coins
# ══════════════════════════════════════════════════════════════════════════════

class TestCountCoins:

    def test_empty_labels(self):
        from src.predict import count_coins
        assert count_coins([]) == {}

    def test_single_label(self):
        from src.predict import count_coins
        assert count_coins(["1euro"]) == {"1euro": 1}

    def test_multiple_same_labels(self):
        from src.predict import count_coins
        assert count_coins(["2cent", "2cent", "2cent"]) == {"2cent": 3}

    def test_mixed_labels(self):
        from src.predict import count_coins
        result = count_coins(["1euro", "2euro", "1euro", "50cent"])
        assert result["1euro"] == 2
        assert result["2euro"] == 1
        assert result["50cent"] == 1

    def test_returns_dict(self):
        from src.predict import count_coins
        assert isinstance(count_coins(["1cent"]), dict)


# ══════════════════════════════════════════════════════════════════════════════
# compute_total_value
# ══════════════════════════════════════════════════════════════════════════════

class TestComputeTotalValue:

    def test_empty_returns_zero(self):
        from src.predict import compute_total_value
        assert compute_total_value({}) == 0.0

    def test_single_1euro(self):
        from src.predict import compute_total_value
        assert abs(compute_total_value({"1euro": 1}) - 1.00) < 1e-9

    def test_single_2euro(self):
        from src.predict import compute_total_value
        assert abs(compute_total_value({"2euro": 1}) - 2.00) < 1e-9

    def test_all_euro_coins(self):
        from src.predict import compute_total_value
        counts = {
            "1cent": 1,   # 0.01
            "2cent": 1,   # 0.02
            "5cent": 1,   # 0.05
            "10cent": 1,  # 0.10
            "20cent": 1,  # 0.20
            "50cent": 1,  # 0.50
            "1euro": 1,   # 1.00
            "2euro": 1,   # 2.00
        }
        expected = 0.01 + 0.02 + 0.05 + 0.10 + 0.20 + 0.50 + 1.00 + 2.00
        assert abs(compute_total_value(counts) - expected) < 1e-9

    def test_non_coin_labels_contribute_zero(self):
        from src.predict import compute_total_value
        assert compute_total_value({"bottle_cap": 5, "not_a_coin": 3}) == 0.0

    def test_multiple_of_same_coin(self):
        from src.predict import compute_total_value
        assert abs(compute_total_value({"2euro": 3}) - 6.00) < 1e-9

    def test_returns_float(self):
        from src.predict import compute_total_value
        assert isinstance(compute_total_value({"1euro": 1}), float)


# ══════════════════════════════════════════════════════════════════════════════
# predict  (full pipeline — integration-style with mocks)
# ══════════════════════════════════════════════════════════════════════════════

class TestPredict:

    def _make_patches(self):
        """Return a dict of patches needed for the full predict() pipeline."""
        return {
            "cv2_imread":      patch("cv2.imread", return_value=_blank_bgr()),
            "load_model":      patch("src.predict.load_model",
                                     return_value=(_fake_model(), _fake_scaler())),
            "detect_coins":    patch("src.detect.detect_coins",
                                     return_value=[(100, 100, 40)]),
            "extract_features":patch("src.features.extract_features",
                                     return_value=np.random.rand(57).astype(np.float32)),
        }

    def test_returns_required_keys(self):
        from src.predict import predict
        patches = self._make_patches()
        with patch("cv2.imread", return_value=_blank_bgr()), \
             patch("src.predict.load_model", return_value=(_fake_model(), None)), \
             patch("src.detect.detect_coins", return_value=[(100, 100, 40)]), \
             patch("src.features.extract_features",
                   return_value=np.random.rand(57).astype(np.float32)):
            result = predict("fake.jpg", "models/classifier.pkl")
        assert "annotated_image" in result
        assert "label_counts"    in result
        assert "total_value"     in result
        assert "coins"           in result
        assert "labels"          in result

    def test_no_coins_detected_returns_empty(self):
        from src.predict import predict
        with patch("cv2.imread", return_value=_blank_bgr()), \
             patch("src.predict.load_model", return_value=(_fake_model(), None)), \
             patch("src.detect.detect_coins", return_value=[]):
            result = predict("fake.jpg", "models/classifier.pkl")
        assert result["coins"] == []
        assert result["labels"] == []
        assert result["total_value"] == 0.0

    def test_raises_when_image_not_found(self):
        from src.predict import predict
        with patch("cv2.imread", return_value=None), \
             patch("src.predict.load_model", return_value=(_fake_model(), None)):
            with pytest.raises((FileNotFoundError, ValueError)):
                predict("nonexistent.jpg", "models/classifier.pkl")

    def test_passes_original_image_to_detect_coins(self):
        """
        BUG CHECK: detect_coins must receive the original BGR image,
        not the preprocessed grayscale (preprocess() returns grayscale).
        """
        from src.predict import predict
        original_bgr = _blank_bgr()
        captured = {}

        def fake_detect(img):
            captured["ndim"] = img.ndim
            captured["shape"] = img.shape
            return []

        with patch("cv2.imread", return_value=original_bgr), \
             patch("src.predict.load_model", return_value=(_fake_model(), None)), \
             patch("src.detect.detect_coins", side_effect=fake_detect):
            predict("fake.jpg", "models/classifier.pkl")

        # detect_coins must receive a 3-channel BGR image, not a grayscale one
        assert captured.get("ndim") == 3, (
            "BUG: detect_coins received a grayscale image. "
            "Pass the original BGR image, not preprocess(image)."
        )
