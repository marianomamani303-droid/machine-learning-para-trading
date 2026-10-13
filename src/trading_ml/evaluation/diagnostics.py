from __future__ import annotations

from typing import Any

import numpy as np

from trading_ml.evaluation.metrics import classification_metrics


def probability_diagnostics(
    target: np.ndarray, probability: np.ndarray, threshold: float = 0.55
) -> dict[str, Any]:
    metrics = classification_metrics(target, probability)
    p = np.asarray(probability, dtype=float)
    y = np.asarray(target)
    prevalence = float(y.mean())
    reliability = sum(
        b["count"] / len(p) * (b["mean_probability"] - b["observed_up_fraction"]) ** 2
        for b in metrics["reliability_bins"]
    )
    resolution = sum(
        b["count"] / len(p) * (b["observed_up_fraction"] - prevalence) ** 2
        for b in metrics["reliability_bins"]
    )
    uncertainty = prevalence * (1 - prevalence)
    return {
        "rows": len(p),
        "minimum": float(p.min()),
        "maximum": float(p.max()),
        "span": float(np.ptp(p)),
        "std": float(p.std()),
        "unique_scores": len(np.unique(p)),
        "quantiles": dict(
            zip(
                ["p01", "p10", "p50", "p90", "p99"],
                np.quantile(p, [0.01, 0.1, 0.5, 0.9, 0.99]).tolist(),
                strict=True,
            )
        ),
        "near_constant_span_below_001": bool(np.ptp(p) < 0.01),
        "bullish_count": int((p >= threshold).sum()),
        "bearish_count": int((p <= np.nextafter(1 - threshold, np.inf)).sum()),
        "calibration_bias": float(p.mean() - prevalence),
        "brier_grouped_decomposition": {
            "uncertainty": uncertainty,
            "reliability": reliability,
            "resolution": resolution,
            "binning_residual": metrics["brier_score"] - (uncertainty - resolution + reliability),
            "note": "Grouped decomposition, not exact individual-score decomposition; residual is reported.",
        },
        "metrics": metrics,
    }
