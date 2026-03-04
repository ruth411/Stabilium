from __future__ import annotations

from dataclasses import dataclass
from math import sqrt
from statistics import NormalDist, stdev


@dataclass(frozen=True)
class MeanConfidence:
    sample_size: int
    mean: float
    std_dev: float
    std_error: float
    confidence_level: float
    ci_low: float
    ci_high: float
    method: str = "normal_approx"

    def to_dict(self) -> dict[str, object]:
        return {
            "sample_size": self.sample_size,
            "mean": self.mean,
            "std_dev": self.std_dev,
            "std_error": self.std_error,
            "confidence_level": self.confidence_level,
            "ci_low": self.ci_low,
            "ci_high": self.ci_high,
            "method": self.method,
        }


def summarize_mean_confidence(
    values: list[float],
    confidence_level: float = 0.95,
) -> MeanConfidence:
    if not values:
        msg = "values must not be empty"
        raise ValueError(msg)
    if confidence_level <= 0.0 or confidence_level >= 1.0:
        msg = "confidence_level must be in (0, 1)"
        raise ValueError(msg)

    sample_size = len(values)
    mean = sum(values) / sample_size
    std_dev = stdev(values) if sample_size > 1 else 0.0
    std_error = std_dev / sqrt(sample_size) if sample_size > 1 else 0.0
    z_value = _z_value(confidence_level)
    margin = z_value * std_error
    return MeanConfidence(
        sample_size=sample_size,
        mean=mean,
        std_dev=std_dev,
        std_error=std_error,
        confidence_level=confidence_level,
        ci_low=mean - margin,
        ci_high=mean + margin,
    )


def one_sample_threshold_significance(
    summary: MeanConfidence,
    threshold: float,
    *,
    alpha: float = 0.05,
) -> dict[str, object]:
    if alpha <= 0.0 or alpha >= 1.0:
        msg = "alpha must be in (0, 1)"
        raise ValueError(msg)

    if summary.sample_size < 2 or summary.std_error == 0.0:
        if summary.mean > threshold:
            p_value = 0.0
        elif summary.mean < threshold:
            p_value = 1.0
        else:
            p_value = 0.5
    else:
        z_score = (summary.mean - threshold) / summary.std_error
        p_value = 1.0 - NormalDist().cdf(z_score)
        p_value = min(max(p_value, 0.0), 1.0)

    significant_pass = p_value < alpha and summary.mean > threshold
    return {
        "method": "one_sided_z_approx",
        "alpha": alpha,
        "threshold": threshold,
        "p_value": p_value,
        "significant_pass": significant_pass,
    }


def compare_sample_means(
    left_values: list[float],
    right_values: list[float],
    *,
    alpha: float = 0.05,
    confidence_level: float = 0.95,
) -> dict[str, object]:
    if not left_values or not right_values:
        msg = "both left_values and right_values must be non-empty"
        raise ValueError(msg)
    if alpha <= 0.0 or alpha >= 1.0:
        msg = "alpha must be in (0, 1)"
        raise ValueError(msg)
    if confidence_level <= 0.0 or confidence_level >= 1.0:
        msg = "confidence_level must be in (0, 1)"
        raise ValueError(msg)

    left = summarize_mean_confidence(left_values, confidence_level=confidence_level)
    right = summarize_mean_confidence(right_values, confidence_level=confidence_level)

    left_term = (left.std_dev**2) / left.sample_size if left.sample_size > 1 else 0.0
    right_term = (right.std_dev**2) / right.sample_size if right.sample_size > 1 else 0.0
    delta_mean = left.mean - right.mean
    delta_std_error = sqrt(left_term + right_term)

    if delta_std_error == 0.0:
        if delta_mean == 0.0:
            p_value = 1.0
        else:
            p_value = 0.0
    else:
        z_score = delta_mean / delta_std_error
        p_value = 2.0 * (1.0 - NormalDist().cdf(abs(z_score)))
        p_value = min(max(p_value, 0.0), 1.0)

    z_value = _z_value(confidence_level)
    margin = z_value * delta_std_error
    delta_ci_low = delta_mean - margin
    delta_ci_high = delta_mean + margin

    if delta_mean > 0:
        better = "left"
    elif delta_mean < 0:
        better = "right"
    else:
        better = "tie"

    return {
        "method": "two_sided_welch_z_approx",
        "alpha": alpha,
        "confidence_level": confidence_level,
        "left": left.to_dict(),
        "right": right.to_dict(),
        "delta_mean": delta_mean,
        "delta_std_error": delta_std_error,
        "delta_ci_low": delta_ci_low,
        "delta_ci_high": delta_ci_high,
        "p_value": p_value,
        "significant_difference": p_value < alpha and better != "tie",
        "better_sample": better,
    }


def _z_value(confidence_level: float) -> float:
    tail_probability = (1.0 + confidence_level) / 2.0
    return NormalDist().inv_cdf(tail_probability)
