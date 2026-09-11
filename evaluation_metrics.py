from __future__ import annotations

from collections.abc import Sequence

import numpy as np


def intraclass_correlation_2_1(
    observed: Sequence[float] | np.ndarray,
    predicted: Sequence[float] | np.ndarray,
) -> float:
    """Return ICC(2,1): two-way random, absolute agreement, single measure.

    Recordings are the targets and the observed and predicted scores are the
    two raters. Negative values are retained because they indicate agreement
    worse than expected from the within-target variability.
    """
    observed_array = np.asarray(observed, dtype=np.float64)
    predicted_array = np.asarray(predicted, dtype=np.float64)
    if observed_array.ndim != 1 or predicted_array.ndim != 1:
        raise ValueError("ICC inputs must be one-dimensional")
    if observed_array.shape != predicted_array.shape:
        raise ValueError("Observed and predicted ICC inputs must have equal length")
    if len(observed_array) < 2:
        return float("nan")
    if not np.all(np.isfinite(observed_array)) or not np.all(
        np.isfinite(predicted_array)
    ):
        raise ValueError("ICC inputs must contain only finite values")

    ratings = np.column_stack((observed_array, predicted_array))
    target_count, rater_count = ratings.shape
    grand_mean = float(ratings.mean())
    target_means = ratings.mean(axis=1)
    rater_means = ratings.mean(axis=0)

    mean_square_targets = float(
        rater_count
        * np.square(target_means - grand_mean).sum()
        / (target_count - 1)
    )
    mean_square_raters = float(
        target_count
        * np.square(rater_means - grand_mean).sum()
        / (rater_count - 1)
    )
    residuals = (
        ratings
        - target_means[:, np.newaxis]
        - rater_means[np.newaxis, :]
        + grand_mean
    )
    mean_square_error = float(
        np.square(residuals).sum()
        / ((target_count - 1) * (rater_count - 1))
    )
    denominator = (
        mean_square_targets
        + (rater_count - 1) * mean_square_error
        + rater_count
        * (mean_square_raters - mean_square_error)
        / target_count
    )
    if np.isclose(denominator, 0.0):
        return float("nan")
    return float((mean_square_targets - mean_square_error) / denominator)
