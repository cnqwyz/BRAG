"""
Extended evaluation metrics including High-Tg bias analysis.
"""

import torch
import numpy as np
from typing import Dict, Tuple, Optional


def compute_metrics_with_bias(
    pred: torch.Tensor,
    target: torch.Tensor,
    high_tg_threshold: Optional[float] = None
) -> Dict[str, float]:
    """
    Compute regression metrics with High-Tg bias analysis.

    All metrics are computed on test set only (not validation or training).

    Metrics:
        - MAE, RMSE, R²: Standard regression metrics
        - High-Tg MAE: Mean absolute error on high-Tg region (>= threshold)
        - Low-Tg MAE: Mean absolute error on low-Tg region (< threshold)
        - Bias Δ: Absolute bias gap = High-Tg MAE - Low-Tg MAE (in Kelvin)
        - Normalized Bias: Bias normalized by region variance (accounts for heteroscedastic noise)
        - High-Tg Ratio: Percentage of samples in high-Tg region

    Args:
        pred: Predictions [N]
        target: Ground truth [N]
        high_tg_threshold: Threshold for High-Tg region. If None, use Q75 of target.

    Returns:
        dict: Metrics including bias analysis
    """
    pred = pred.detach().cpu().numpy()
    target = target.detach().cpu().numpy()

    # Standard metrics
    mse = np.mean((pred - target) ** 2)
    mae = np.mean(np.abs(pred - target))
    rmse = np.sqrt(mse)
    ss_res = np.sum((target - pred) ** 2)
    ss_tot = np.sum((target - np.mean(target)) ** 2)
    r2 = 1 - (ss_res / ss_tot) if ss_tot > 0 else 0

    metrics = {
        "mse": float(mse),
        "mae": float(mae),
        "rmse": float(rmse),
        "r2": float(r2)
    }

    # High-Tg bias analysis
    if high_tg_threshold is None:
        high_tg_threshold = np.percentile(target, 75)

    # Full test metrics
    full_mae = float(np.mean(np.abs(pred - target)))

    # High-Tg region metrics
    high_mask = target >= high_tg_threshold
    if high_mask.sum() > 0:
        high_pred = pred[high_mask]
        high_target = target[high_mask]
        high_mae = float(np.mean(np.abs(high_pred - high_target)))
        high_r2 = float(1 - (np.sum((high_target - high_pred) ** 2) /
                            (np.sum((high_target - np.mean(high_target)) ** 2))))
        high_std = float(np.std(high_target))
    else:
        high_mae = 0.0
        high_r2 = 0.0
        high_std = 1.0  # Avoid division by zero

    # Low-Tg region metrics
    low_mask = target < high_tg_threshold
    if low_mask.sum() > 0:
        low_pred = pred[low_mask]
        low_target = target[low_mask]
        low_mae = float(np.mean(np.abs(low_pred - low_target)))
        low_std = float(np.std(low_target))
    else:
        low_mae = 0.0
        low_std = 1.0  # Avoid division by zero

    # Bias delta (absolute)
    bias_delta = high_mae - low_mae

    # Normalized bias delta (accounts for heteroscedastic noise)
    # This is the more rigorous metric that accounts for variance differences
    normalized_bias = (high_mae / high_std) - (low_mae / low_std)

    metrics.update({
        "high_tg_threshold": float(high_tg_threshold),
        "high_tg_mae": high_mae,
        "high_tg_r2": high_r2,
        "low_tg_mae": low_mae,
        "bias_delta": bias_delta,
        "normalized_bias": normalized_bias,  # NEW: variance-normalized bias
        "high_tg_ratio": float(high_mask.sum() / len(target))
    })

    return metrics


def format_bias_metrics(metrics: Dict[str, float]) -> str:
    """
    Format bias metrics for logging.

    Args:
        metrics: Metrics dictionary from compute_metrics_with_bias

    Returns:
        str: Formatted string
    """
    parts = [
        f"MAE: {metrics['mae']:.2f}",
        f"RMSE: {metrics['rmse']:.2f}",
        f"R²: {metrics['r2']:.3f}",
        f"High-Tg MAE: {metrics['high_tg_mae']:.2f} (threshold={metrics['high_tg_threshold']:.0f})",
        f"Bias Δ: {metrics['bias_delta']:+.2f}",
        f"Norm. Bias: {metrics['normalized_bias']:+.2f}"
    ]
    return " | ".join(parts)
