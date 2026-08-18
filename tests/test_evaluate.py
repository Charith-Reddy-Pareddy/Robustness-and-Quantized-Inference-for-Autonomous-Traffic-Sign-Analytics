import numpy as np
import pytest

from src.models.evaluate import summarize


def test_macro_f1_differs_from_accuracy_under_class_imbalance():
    """Regression test for a bug where average='micro' was used instead of 'macro':
    micro-F1 is mathematically identical to accuracy for single-label multiclass
    predictions, so it can never detect that a model is failing on a minority class.
    """
    labels = np.array([0] * 90 + [1] * 10)
    preds = np.array([0] * 90 + [0] * 10)  # model never predicts the minority class

    result = summarize(preds, labels)

    assert result["accuracy"] == pytest.approx(0.9)
    assert result["macro_f1"] == pytest.approx(0.4737, abs=1e-3)
    assert result["macro_f1"] < result["accuracy"] - 0.3


def test_macro_f1_equals_accuracy_when_all_correct():
    labels = np.array([0, 1, 2, 0, 1, 2])
    preds = np.array([0, 1, 2, 0, 1, 2])
    result = summarize(preds, labels)
    assert result["accuracy"] == pytest.approx(1.0)
    assert result["macro_f1"] == pytest.approx(1.0)


def test_summarize_returns_expected_keys():
    labels = np.array([0, 1, 1, 0])
    preds = np.array([0, 1, 0, 0])
    result = summarize(preds, labels)
    assert set(result.keys()) == {"accuracy", "macro_f1", "report", "confusion_matrix"}
    assert result["confusion_matrix"].shape == (2, 2)
