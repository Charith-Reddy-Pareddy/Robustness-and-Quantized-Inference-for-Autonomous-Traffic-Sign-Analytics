"""Small-sample (n=5 seeds) statistics for robustness comparisons: mean/std/CI for a
single condition, and bootstrap CI + paired significance test + effect size for a
FP32-vs-INT8 (or any paired) delta.
"""

import numpy as np
from scipy import stats as scipy_stats


def mean_std_ci(values, confidence: float = 0.95) -> dict:
    """Mean, sample std, and a t-distribution confidence interval for a small sample."""
    values = np.asarray(values, dtype=float)
    n = len(values)
    mean = float(values.mean())
    std = float(values.std(ddof=1)) if n > 1 else 0.0
    if n > 1 and std > 0:
        se = std / np.sqrt(n)
        t_crit = scipy_stats.t.ppf((1 + confidence) / 2, df=n - 1)
        half_width = float(t_crit * se)
    else:
        half_width = 0.0
    return {
        "n": n,
        "mean": mean,
        "std": std,
        "ci_low": mean - half_width,
        "ci_high": mean + half_width,
    }


def bootstrap_paired_delta_ci(
    baseline_values, comparison_values, confidence: float = 0.95, n_resamples: int = 10000, seed: int = 42
) -> dict:
    """Bootstrap CI on the mean paired delta (comparison - baseline), resampling seeds
    with replacement. Also returns a paired t-test p-value and Cohen's dz effect size on
    the same paired differences (parametric, not bootstrap, but standard alongside it).
    """
    baseline_values = np.asarray(baseline_values, dtype=float)
    comparison_values = np.asarray(comparison_values, dtype=float)
    if len(baseline_values) != len(comparison_values):
        raise ValueError("baseline and comparison must have the same number of paired seeds")

    diffs = comparison_values - baseline_values
    n = len(diffs)
    mean_diff = float(diffs.mean())

    rng = np.random.default_rng(seed)
    resampled_means = np.empty(n_resamples)
    for i in range(n_resamples):
        idx = rng.integers(0, n, size=n)
        resampled_means[i] = diffs[idx].mean()
    alpha = 1 - confidence
    ci_low, ci_high = np.percentile(resampled_means, [100 * alpha / 2, 100 * (1 - alpha / 2)])

    if n > 1 and diffs.std(ddof=1) > 0:
        t_result = scipy_stats.ttest_rel(comparison_values, baseline_values)
        p_value = float(t_result.pvalue)
        cohens_dz = float(mean_diff / diffs.std(ddof=1))
    else:
        p_value = 1.0
        cohens_dz = 0.0

    return {
        "n": n,
        "mean_delta": mean_diff,
        "bootstrap_ci_low": float(ci_low),
        "bootstrap_ci_high": float(ci_high),
        "paired_t_pvalue": p_value,
        "cohens_dz": cohens_dz,
    }
