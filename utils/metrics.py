"""
Utility functions for computing evaluation metrics.
"""

import torch
import numpy as np


def compute_metrics(pred, target):
    """
    Compute regression metrics.
    
    Args:
        pred: Predictions [N] or [batch_size]
        target: Ground truth [N] or [batch_size]
        
    Returns:
        dict: Dictionary containing various metrics
    """
    pred = pred.detach().cpu().numpy()
    target = target.detach().cpu().numpy()
    
    # MSE
    mse = np.mean((pred - target) ** 2)
    
    # MAE
    mae = np.mean(np.abs(pred - target))
    
    # RMSE
    rmse = np.sqrt(mse)
    
    # R^2
    ss_res = np.sum((target - pred) ** 2)
    ss_tot = np.sum((target - np.mean(target)) ** 2)
    r2 = 1 - (ss_res / ss_tot) if ss_tot > 0 else 0
    
    return {
        "mse": float(mse),
        "mae": float(mae),
        "rmse": float(rmse),
        "r2": float(r2)
    }


def format_metrics(metrics: dict, prefix: str = ""):
    """
    Format metrics for logging.
    
    Args:
        metrics: Metrics dictionary
        prefix: Prefix for metric names
        
    Returns:
        str: Formatted string
    """
    parts = []
    for k, v in metrics.items():
        parts.append(f"{prefix}{k}: {v:.4f}")
    return " | ".join(parts)
