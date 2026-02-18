# ─── Standard library ───────────────────────────────────────────────
import json
import os
import pickle
import platform
import sys
import time
import tracemalloc
from typing import (
    Any,
    Callable,
    Dict,
    Optional,
    Protocol,
    Tuple,
    Union,
    runtime_checkable,
)

# ─── Third-party libraries ─────────────────────────────────────────
import pkg_resources
import matplotlib.pyplot as plt
import numpy as np
import optuna
import pandas as pd
import psutil
from scipy.stats import kurtosis, skew
from skimage import color, feature, filters
from skimage.feature import canny
from sklearn.base import BaseEstimator
from sklearn.ensemble import (
    GradientBoostingClassifier,
    RandomForestClassifier,
)
from sklearn.metrics import (
    accuracy_score,
    f1_score,
    precision_score,
    recall_score,
)

from sklearn.metrics import mean_squared_error

from sklearn.model_selection import train_test_split
from torch.utils.data import random_split
from torchvision.datasets import EuroSAT

from optuna.importance import get_param_importances as _get_param_importances
from optuna.samplers import BaseSampler, TPESampler
from optuna.study import Study
from optuna.trial import FrozenTrial

def timed_augmentation(
    X_data: Any,
    M_data: Any, 
    y_data: Any,
    augmentation_fn: Callable[
        [Any, Any, Any], Tuple[Any, Any, Any]
    ],
    augmentation_params: Optional[Dict[str, Any]] = None
) -> Tuple[Any, Any, Any, float, int]:
    """
    Measure execution time of an augmentation function.

    Args:
        X_data: Input data for augmentation.
        M_data: Input masks for augmentation.
        y_data: Input labels for augmentation.
        augmentation_fn: Function returning (X_aug, M_aug, y_aug).
        augmentation_params: Optional keyword arguments.

    Returns:
        X_aug, M_aug, y_aug, elapsed_time (s), number_of_augmented_samples

    Raises:
        ValueError: If augmented data is empty.
    """
    start = time.perf_counter()
    X_aug, M_aug, y_aug = augmentation_fn(X_data, M_data, y_data, **(augmentation_params or {}))
    elapsed = time.perf_counter() - start
    
    if len(X_aug) == 0:
        raise ValueError("Augmentation function returned empty data")
    
    return X_aug, M_aug, y_aug, elapsed, len(X_aug)


def preprocessing_data(
    X_train: Any,
    M_train: Any,
    y_train: Any,
    augmentation_fn: Callable[[Any, Any, Any], Tuple[Any, Any, Any]],
    augmentation_params_optional: Optional[Dict] = None
) -> Tuple[Any, Any, Any, Dict[str, Any]]:
    """
    Apply augmentation to training data, timing the process and measuring memory usage,
    following the same metrics structure as extract_features_split.
    
    Args:
        X_train: Training data to augment
        M_train: Training masks to augment
        y_train: Training labels to augment
        augmentation_fn: Function that applies augmentation to the data
        augmentation_params_optional: Optional parameters for the augmentation function
    
    Returns:
        X_aug_train: Augmented training data
        M_aug_train: Augmented training masks
        y_aug_train: Augmented training labels
        augmentation_metrics: Dictionary containing metrics about augmentation process
    
    Raises:
        Exception: If there's an error measuring memory usage
    """
    # PSUtil RSS memory measurement (in MB) - Snapshot before the augmentation
    try:
        mem_rss_mb_before = psutil.Process().memory_info().rss / (1024 ** 2)
    except Exception:
        print("ERROR: Failed to measure initial memory usage")
        mem_rss_mb_before = 0.0

    # Start memory tracking
    tracemalloc.start()
    
    # Apply augmentation to training data
    X_aug_train, M_aug_train, y_aug_train, augmentation_time, n_aug_samples = timed_augmentation(
        X_train, M_train, y_train, augmentation_fn, augmentation_params_optional
    )

    # Measure memory after augmentation
    try:
        mem_rss_mb_after = psutil.Process().memory_info().rss / (1024 ** 2)
        rss_delta_augmentation_mb = mem_rss_mb_after - mem_rss_mb_before
    except Exception:
        print("ERROR: Failed to measure final memory usage")
        mem_rss_mb_after = 0.0
        rss_delta_augmentation_mb = 0.0

    # Get peak memory usage from tracemalloc
    _, peak_trace_augmentation_byte = tracemalloc.get_traced_memory()
    tracemalloc.stop()

    # Assemble augmentation metrics dictionary
    augmentation_metrics = {
        "original_train_samples": len(X_train),
        "augmented_samples": n_aug_samples,
        "augmentation_ratio": n_aug_samples / len(X_train) if len(X_train) > 0 else 0,
        "augmentation_time_s": augmentation_time,
        "PSUtil_RSS_Before_MB": mem_rss_mb_before,
        "PSUtil_RSS_After_MB": mem_rss_mb_after,
        "PSUtil_RSS_Delta_MB": rss_delta_augmentation_mb,
        "Tracemalloc_Peak_Bytes": peak_trace_augmentation_byte 
    }

    return X_aug_train, M_aug_train, y_aug_train, augmentation_metrics


def timed_feature_extraction(
    dataset: Any,
    extractor_fn: Callable[..., Tuple[np.ndarray, np.ndarray]],
    extractor_params: Optional[Dict[str, Any]] = None,
    return_field_sizes: bool = False
) -> Union[
      Tuple[np.ndarray, np.ndarray, float, int],
      Tuple[np.ndarray, np.ndarray, np.ndarray, float, int]
]:
    start = time.perf_counter()
    out = extractor_fn(dataset, **(extractor_params or {}))
    elapsed = time.perf_counter() - start

    # out can be (X, y) or (X, y, field_sizes)
    if len(out) == 2:
        X, y = out
        field_sizes = None
    elif len(out) == 3:
        X, y, field_sizes = out
    else:
        raise ValueError(f"Extractor returned {len(out)} items, expected 2 or 3")

    if X.ndim != 2:
        raise ValueError(f"Expected 2D array, got ndim={X.ndim}")

    n_feat = X.shape[1]
    if return_field_sizes:
        return X, y, field_sizes, elapsed, n_feat
    else:
        return X, y, elapsed, n_feat

def extract_features_split(
    ds_train: Any,
    ds_test: Any,
    ds_train_aug: Any,
    extractor_fn: Callable[..., Tuple[np.ndarray, np.ndarray]],
    extractor_params_optional: Optional[Dict] = None,
    return_field_sizes: bool = False
) -> Union[
    # old signature
    Tuple[np.ndarray, np.ndarray, np.ndarray, np.ndarray, np.ndarray, np.ndarray, Dict[str, Any]],
    # extended signature
    Tuple[np.ndarray, np.ndarray, np.ndarray, np.ndarray, np.ndarray, np.ndarray,
          np.ndarray, np.ndarray, np.ndarray, Dict[str, Any]]
]:
    """
    Compute features on train/test (and optional aug) splits, collect timing/memory,
    and optionally return field-size arrays.
    """
    # initialize
    X_train_aug = np.array([]); y_train_aug = np.array([])
    train_aug_time = 0.0; n_feat_train_aug = 0
    avg_train = avg_test = avg_train_aug = None

    # memory before
    try:
        mem_before = psutil.Process().memory_info().rss / (1024 ** 2)
    except:
        mem_before = 0.0

    tracemalloc.start()

    # extract with or without field sizes
    if return_field_sizes:
        X_train, y_train, avg_train, t_train, n_feat_train = timed_feature_extraction(
            ds_train, extractor_fn, extractor_params_optional, return_field_sizes=True
        )
        X_test,  y_test,  avg_test,  t_test,  n_feat_test  = timed_feature_extraction(
            ds_test,  extractor_fn, extractor_params_optional, return_field_sizes=True
        )
        if ds_train_aug is not None:
            X_train_aug, y_train_aug, avg_train_aug, t_aug, n_feat_train_aug = timed_feature_extraction(
                ds_train_aug, extractor_fn, extractor_params_optional, return_field_sizes=True
            )
            train_aug_time = t_aug
        else:
            train_aug_time = 0.0
    else:
        X_train, y_train, t_train, n_feat_train = timed_feature_extraction(
            ds_train, extractor_fn, extractor_params_optional
        )
        X_test,  y_test,  t_test,  n_feat_test  = timed_feature_extraction(
            ds_test,  extractor_fn, extractor_params_optional
        )
        if ds_train_aug is not None:
            X_train_aug, y_train_aug, train_aug_time, n_feat_train_aug = timed_feature_extraction(
                ds_train_aug, extractor_fn, extractor_params_optional
            )

    # memory after
    try:
        mem_after = psutil.Process().memory_info().rss / (1024 ** 2)
        rss_delta = mem_after - mem_before
    except:
        mem_after = rss_delta = 0.0

    peak_mem = tracemalloc.get_traced_memory()[1]
    tracemalloc.stop()

    # assemble metrics
    features_metrics = {
        "train_samples": X_train.shape[0],
        "test_samples":  X_test.shape[0],
        "train_aug_samples": X_train_aug.shape[0],
        "train_num_features": n_feat_train,
        "test_num_features":  n_feat_test,
        "train_aug_num_features": n_feat_train_aug,
        "train_feature_extraction_time_s": t_train,
        "test_feature_extraction_time_s":  t_test,
        "train_aug_feature_extraction_time_s": train_aug_time,
        "PSUtil_RSS_Before_MB": mem_before,
        "PSUtil_RSS_After_MB": mem_after,
        "PSUtil_RSS_Delta_MB": rss_delta,
        "Tracemalloc_Peak_Bytes": peak_mem
    }

    if return_field_sizes:
        return (
            X_train, y_train,
            X_test,  y_test,
            X_train_aug, y_train_aug,
            avg_train, avg_test, avg_train_aug,
            features_metrics
        )
    else:
        return (
            X_train, y_train,
            X_test,  y_test,
            X_train_aug, y_train_aug,
            features_metrics
        )

def evaluation_score(args, y_v, y_hat, y_b, cons):
    score = 0
    for i in range(len(args.col_ix)):
        print(f'Soil idx {i} / {len(args.col_ix) - 1}')
        mse_rf = mean_squared_error(y_v[:, i] * cons[i], y_hat[:, i] * cons[i])
        mse_bl = mean_squared_error(y_v[:, i] * cons[i], y_b[:, i] * cons[i])

        score += mse_rf / mse_bl

        print(f'Baseline MSE:      {mse_bl:.2f}')
        print(f'Random Forest MSE: {mse_rf:.2f} ({1e2 * (mse_rf - mse_bl) / mse_bl:+.2f} %)')
        print(f'Evaluation score: {score / len(args.col_ix)}')

    return score / 4


class BaselineRegressor:
    """
    Baseline regressor, which calculates the mean value of the target from the training
    data and returns it for each testing sample.
    """

    def __init__(self):
        self.mean = 0

    def fit(self, X_train: np.ndarray, y_train: np.ndarray):
        self.mean = np.mean(y_train, axis=0)
        self.classes_count = y_train.shape[1]
        return self

    def predict(self, X_test: np.ndarray):
        return np.full((len(X_test), self.classes_count), self.mean)