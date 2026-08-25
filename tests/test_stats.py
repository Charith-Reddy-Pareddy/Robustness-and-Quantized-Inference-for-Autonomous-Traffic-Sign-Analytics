import numpy as np
import pytest

from src.analysis.stats import bootstrap_paired_delta_ci, mean_std_ci


def test_mean_std_ci_matches_known_values():
    result = mean_std_ci([2.0, 4.0, 4.0, 4.0, 5.0])
    assert result["n"] == 5
    assert result["mean"] == 3.8
    assert result["ci_low"] < result["mean"] < result["ci_high"]


def test_mean_std_ci_single_value_has_zero_width():
    result = mean_std_ci([1.0])
    assert result["std"] == 0.0
    assert result["ci_low"] == result["ci_high"] == 1.0


def test_bootstrap_ci_contains_true_mean_delta_for_constant_shift():
    rng = np.random.default_rng(0)
    baseline = rng.normal(loc=50.0, scale=1.0, size=5)
    comparison = baseline - 9.0  # exact constant delta, no noise on the shift itself

    result = bootstrap_paired_delta_ci(baseline, comparison, n_resamples=2000)

    assert result["mean_delta"] == pytest.approx(-9.0)
    assert result["bootstrap_ci_low"] <= result["mean_delta"] <= result["bootstrap_ci_high"]


def test_bootstrap_ci_no_effect_gives_zero_effect_size():
    values = [90.0, 91.0, 89.5, 90.5, 90.2]
    result = bootstrap_paired_delta_ci(values, values, n_resamples=500)

    assert result["mean_delta"] == 0.0
    assert result["cohens_dz"] == 0.0
    assert result["paired_t_pvalue"] == 1.0


def test_bootstrap_ci_raises_on_mismatched_lengths():
    with pytest.raises(ValueError):
        bootstrap_paired_delta_ci([1.0, 2.0], [1.0, 2.0, 3.0])
