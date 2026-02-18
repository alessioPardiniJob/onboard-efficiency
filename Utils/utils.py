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
    List,
)

import joblib


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
from sklearn.model_selection import train_test_split
from torch.utils.data import random_split
from torchvision.datasets import EuroSAT

from optuna.importance import get_param_importances as _get_param_importances
from optuna.samplers import BaseSampler, TPESampler
from optuna.study import Study
import torch

from optuna.trial import FrozenTrial, Trial # Importa Trial direttamente

from sklearn.ensemble import RandomForestClassifier, GradientBoostingClassifier
from sklearn.ensemble import RandomForestRegressor, GradientBoostingRegressor
from Utils.p_utils import evaluation_score, BaselineRegressor



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







@runtime_checkable
class ModelProtocol(Protocol):
    """
    Protocol for any predictive model: must implement fit() and predict().
    """
    def fit(self, X: Any, y: Any) -> Any: ...
    def predict(self, X: Any) -> Any: ...



def load_json_config(path: str) -> Dict[str, Any]:
    """
    Load a JSON file and return its contents as a dictionary.

    Args:
        path: Path to the JSON configuration file.

    Returns:
        Parsed JSON as a dictionary.
    """
    with open(path, 'r') as f:
        return json.load(f)



def create_study_from_config(config: Dict[str, Any]) -> Study:
    """
    Create and configure an Optuna Study based on a configuration dictionary.

    Args: 
        config : Dict[str, Any]
            A dictionary with the following keys:
                - "study": {
                    "study_name": str,
                    "direction": str,
                    "storage": str ("null" means in-memory),
                    "load_if_exists": bool
                }
                - "execution": {
                    "n_jobs": int,
                    "timeout_s": int
                }
                - "sampler": {
                    "type": str,
                    "params": Dict[str, Any]
                }

    Returns:
        A configured optuna.study.Study object.

    Raises:
        ValueError: If sampler type is unsupported.
    """
    # ── Study parameters ─────────────────────────────────────────────
    study_cfg: Dict[str, Any]  = config["study"]
    study_name: str            = study_cfg["study_name"]
    direction: str             = study_cfg["direction"]
    storage_str: Optional[str] = study_cfg["storage"]
    load_if_exists: bool       = study_cfg["load_if_exists"]

    storage: Optional[str] = None if storage_str == "null" else storage_str

    # ── Execution parameters ────────────────────────────────────────
    exec_cfg: Dict[str, Any] = config["execution"]
    n_jobs: int              = exec_cfg["n_jobs"]
    #timeout_sec: int         = exec_cfg["timeout_s"]
    n_trials:int             = exec_cfg["n_trials"]

    # ── Sampler parameters ─────────────────────────────────────────
    sampler_cfg: Dict[str, Any]    = config["sampler"]
    sampler_type: str              = sampler_cfg["type"]
    sampler_params: Dict[str, Any] = sampler_cfg.get("params", {})

    if sampler_type == "TPESampler":
        sampler: TPESampler = TPESampler(**sampler_params)
    else:
        raise ValueError(f"Unsupported sampler type: {sampler_type!r}")

    # ── Create the study object ─────────────────────────────────────
    study: Study = optuna.create_study(
        study_name=study_name,
        direction=direction,
        storage=storage,
        load_if_exists=load_if_exists,
        sampler=sampler
    )

    # ── Attach execution metadata ───────────────────────────────────
    study._n_jobs  = n_jobs      # type: ignore[attr-defined]
    #study._timeout = timeout_sec # type: ignore[attr-defined]
    study._n_trials = n_trials    # type: ignore[attr-defined]


    return study

def record_metrics_callback(study: optuna.Study, trial: FrozenTrial) -> None:
    """
    Callback to record timing, best score, trial duration, and memory usage.
    
    Args:
        study: Optuna study object
        trial: Current trial object
    
    Returns:
        None
    
    Note: 
        tracemalloc peak here is cumulative since tracemalloc.start() was called.
    """
    # Record elapsed time since start
    elapsed = time.time() - record_metrics_callback.start_time
    record_metrics_callback.timestamps.append(elapsed)
    
    # Record best score if available
    record_metrics_callback.best_scores.append(study.best_value if study.best_value is not None else None)

    # Calculate trial duration
    try:
        if trial.datetime_start and trial.datetime_complete:
            duration = (trial.datetime_complete - trial.datetime_start).total_seconds()
        else:
            duration = 0.0
    except Exception:
        duration = 0.0

    record_metrics_callback.trial_times[trial.number] = duration

    # Measure memory usage with PSUtil
    try:
        mem_rss_mb = psutil.Process().memory_info().rss / (1024 ** 2)
        record_metrics_callback.trial_memory.append(mem_rss_mb)
    except Exception:
        record_metrics_callback.trial_memory.append(0.0)

    # Measure peak memory usage with tracemalloc
    try:
        _, peak = tracemalloc.get_traced_memory()
        record_metrics_callback.trial_tracemalloc_peaks[trial.number] = peak
    except Exception:
        record_metrics_callback.trial_tracemalloc_peaks[trial.number] = 0.0


# Initialize static variables for the callback
record_metrics_callback.timestamps = []
record_metrics_callback.best_scores = []
record_metrics_callback.trial_times = {}
record_metrics_callback.trial_memory = []
record_metrics_callback.trial_tracemalloc_peaks = {}
record_metrics_callback.start_time = 0.0


def run_optimization(
    objective_fn: Callable[[Trial, Any, Optional[Any], Dict[str, Any]], float], # Usa Trial importato direttamente
    data: Any,
    augmented_data: Any,
    project_cfg: Dict[str, Any],
    optuna_cfg: Dict[str, Any],
) -> Dict[str, Any]:
    """
    Runs Optuna optimization, recording memory usage and performance metrics.
    
    Args:
        objective_fn: The objective function to optimize
        X_full: The feature data
        y_full: The target data
        project_cfg: Configuration for the project
        optuna_cfg: Configuration for Optuna
    
    Returns:
        Dictionary containing detailed optimization results and metrics
    
    Raises:
        Exception: If study creation fails or optimization encounters errors
    """
    # Initialize study based on configuration
    study = create_study_from_config(optuna_cfg)
    
    # Reset metrics for the callback
    record_metrics_callback.start_time = time.time()
    record_metrics_callback.timestamps.clear()
    record_metrics_callback.best_scores.clear()
    record_metrics_callback.trial_times.clear()
    record_metrics_callback.trial_memory.clear()
    record_metrics_callback.trial_tracemalloc_peaks.clear()
    
    # Get initial memory usage
    psutil_rss_initial_mb = _get_memory_usage("initial")
    
    # Start memory tracking with tracemalloc
    tracemalloc.start()
    
    # Wrap the objective function to track memory usage per trial
    def _wrapped_objective(trial: optuna.trial.Trial) -> float:
        # Measure memory before trial
        before_rss_mb = _get_memory_usage("before trial")
        trial.set_user_attr("psutil_rss_before_mb", before_rss_mb)
        
        # Execute the trial
        value = objective_fn(trial, data, augmented_data, project_cfg)
        
        # Measure memory after trial
        after_rss_mb = _get_memory_usage("after trial")
        trial.set_user_attr("psutil_rss_after_mb", after_rss_mb)
        trial.set_user_attr("psutil_rss_delta_mb", after_rss_mb - before_rss_mb)
        
        return value
    
    # Run the optimization
    study.optimize(
        _wrapped_objective,
        n_jobs=study._n_jobs,
        n_trials=study._n_trials,
        callbacks=[record_metrics_callback]
    )
    
    # Get final memory usage
    psutil_rss_final_mb = _get_memory_usage("final")
    
    # Stop tracemalloc and get peak memory usage
    _, tracemalloc_peak_overall_bytes = tracemalloc.get_traced_memory()
    tracemalloc.stop()
    
    # Calculate total optimization time
    total_optimization_time_s = time.time() - record_metrics_callback.start_time
    
    # Generate results dictionary
    results = {
        "optimization_performance": _get_optimization_performance(
            study, 
            total_optimization_time_s,
            psutil_rss_initial_mb,
            psutil_rss_final_mb,
            tracemalloc_peak_overall_bytes
        ),
        "best_result": _get_best_result(study),
        "trial_info": {      
            "memory_metrics": _get_memory_metrics(study, record_metrics_callback),
            "timing_metrics": _get_timing_metrics(study, record_metrics_callback),
            "trial_params": _get_trial_params(study),
        },
        "visualization": _generate_visualizations(
            study, 
            project_cfg, 
            record_metrics_callback
        )
    }
    
    return results


def _get_memory_usage(stage: str) -> float:
    """
    Get current memory usage with error handling
    
    Args:
        stage: Description of the measurement stage for error reporting
    
    Returns:
        Current memory usage in MB
    """
    try:
        memory_mb = psutil.Process().memory_info().rss / (1024 ** 2)
        return memory_mb
    except Exception as e:
        print(f"[WARNING] Could not get {stage} RSS: {e}")
        return 0.0


def _get_optimization_performance(
    study: optuna.study.Study,
    total_time: float,
    initial_memory_mb: float,
    final_memory_mb: float,
    peak_memory_bytes: int
) -> Dict[str, Any]:
    """
    Extract general study performance metrics
    
    Args:
        study: The completed Optuna study
        total_time: Total time spent on optimization in seconds
        initial_memory_mb: Memory usage at start in MB
        final_memory_mb: Memory usage at end in MB
        peak_memory_bytes: Peak memory allocation in bytes
    
    Returns:
        Dictionary with optimization performance metrics
    """
    return {
        "study_name": study.study_name,
        "direction": str(study.direction),
        "total_optimization_time_s": total_time,
        "total_trials": len(study.trials),
        "memory_usage": {
            "initial_mb": initial_memory_mb,
            "final_mb": final_memory_mb,
            "delta_mb": final_memory_mb - initial_memory_mb,
            "tracemalloc_peak_bytes": peak_memory_bytes
        }
    }


def _get_memory_metrics(
    study: optuna.study.Study,
    callback_data
) -> Dict[str, Any]:
    """
    Extract memory metrics per trial and overall
    
    Args:
        study: The completed Optuna study
        callback_data: Data collected by the callback function
    
    Returns:
        Dictionary with memory metrics
    """
    per_trial_memory = [
        {
            "trial_number": t.number,
            "psutil_before_mb": t.user_attrs.get("psutil_rss_before_mb", 0.0),
            "psutil_after_mb": t.user_attrs.get("psutil_rss_after_mb", 0.0),
            "psutil_delta_mb": t.user_attrs.get("psutil_rss_delta_mb", 0.0),
            "peak_allocated_memory_bytes": callback_data.trial_tracemalloc_peaks.get(t.number)
        }
        for t in study.trials
    ]
    
    avg_delta_mb = (
        sum(t.user_attrs.get("psutil_rss_delta_mb", 0.0) for t in study.trials) / len(study.trials)
        if study.trials else 0.0
    )
    
    avg_peak_bytes = (
        sum(v for v in callback_data.trial_tracemalloc_peaks.values()) / len(callback_data.trial_tracemalloc_peaks)
        if callback_data.trial_tracemalloc_peaks else 0.0
    )
    
    return {
        "per_trial": per_trial_memory,
        "average_delta_mb": avg_delta_mb,
        "average_peak_bytes": avg_peak_bytes
    }


def _get_timing_metrics(
    study: optuna.study.Study,
    callback_data
) -> Dict[str, Any]:
    """
    Extract timing metrics per trial and overall
    
    Args:
        study: The completed Optuna study
        callback_data: Data collected by the callback function
    
    Returns:
        Dictionary with timing metrics
    """
    per_trial_time = [
        {
            "trial_number": t.number,
            "trial_time_s": callback_data.trial_times.get(t.number)
        }
        for t in study.trials
    ]
    
    avg_trial_time_s = (
        sum(v for v in callback_data.trial_times.values()) / len(callback_data.trial_times)
        if callback_data.trial_times else 0.0
    )
    
    return {
        "per_trial": per_trial_time,
        "average_trial_time_s": avg_trial_time_s
    }


def _get_trial_params(study: optuna.study.Study) -> Dict[str, Any]:
    """
    Extract parameters used in each trial
    
    Args:
        study: The completed Optuna study
    
    Returns:
        Dictionary with parameters for each trial
    """
    per_trial_params = [
        {
            "trial_number": t.number,
            "trial_params": t.params
        }
        for t in study.trials
    ]

    return per_trial_params


def _get_best_result(study: optuna.study.Study) -> Dict[str, Any]:
    """
    Extract information about the best trial
    
    Args:
        study: The completed Optuna study
    
    Returns:
        Dictionary with best trial information
    """
    return {
        "trial_number": study.best_trial.number,
        "score": study.best_value,
        "hyperparameters": study.best_trial.params
    }


def _generate_visualizations(
    study: optuna.study.Study,
    project_cfg: Dict[str, Any],
    callback_data
) -> Dict[str, Any]:
    """
    Generate and save visualization artifacts
    
    Args:
        study: The completed Optuna study
        project_cfg: Project configuration dictionary
        callback_data: Data collected by the callback function
    
    Returns:
        Dictionary with visualization paths and data
    
    Raises:
        Exception: If directory creation or file saving fails
    """
    base_dir = project_cfg["output_paths"]["output_result_path"]
    os.makedirs(base_dir, exist_ok=True)
    
    # Convergence curve
    timestamps = callback_data.timestamps
    best_scores = callback_data.best_scores
    conv_path = os.path.join(base_dir, project_cfg["output_paths"]["convergence"])
    
    plt.figure()
    plt.plot(timestamps, best_scores)
    plt.xlabel("Time (s)")
    plt.ylabel("Best Objective Value")
    plt.title("Convergence Curve (Best Value Over Time)")
    plt.savefig(conv_path)
    plt.close()
    
    # Optimization history
    hist_fig = optuna.visualization.plot_optimization_history(study)
    hist_path = os.path.join(base_dir, project_cfg["output_paths"]["history_html"])
    hist_fig.write_html(hist_path)
    
    # Parameter importances
    importances = get_param_importances(study)
    imp_fig = optuna.visualization.plot_param_importances(study)
    imp_path = os.path.join(base_dir, project_cfg["output_paths"]["importances_html"])
    imp_fig.write_html(imp_path)
    
    return {
        "convergence_curve": {
            "timestamps_s": timestamps,
            "best_scores": best_scores,
            "plot_path": conv_path,
        },
        "optimization_history": {
            "path": hist_path
        },
        "parameter_importances": {
            "importances": importances,
            "figure_path": imp_path,
        }
    }



# --- Modified save_run_results to support regression tasks ---
def save_run_results(
    project_cfg: Dict[str, Any],
    cfg_file_path: str,
    feature_metrics: Dict[str, Any],
    optuna_run_metrics: Dict[str, Any],
    evaluation_metrics: Dict[str, Any],
    extra: Optional[Dict] = None,
) -> None:
    import os, json, platform, psutil
    # Determine task type from evaluation_metrics keys
    task_type = 'regression' if 'evaluation_score' in evaluation_metrics else 'classification'

    base_dir = project_cfg['output_paths']['output_result_path']
    os.makedirs(base_dir, exist_ok=True)

    requirements_txt = get_requirements_versions(project_cfg)

    output = {
        'Config_filepath': cfg_file_path,
        'Experiment_Metadata': {
            'optimizer': 'Optuna',
            'task_type': task_type,
            'random_seed': project_cfg['seed'],
            'hardware': {
                'cpu': platform.processor(),
                'gpu': None,
                'ram_gb': psutil.virtual_memory().total / (1024 ** 3)
            },
            'requirements_txt': requirements_txt
        },
        'Feature_Extraction_Metrics': feature_metrics,
        'Optuna': optuna_run_metrics,
        'Evaluation_Metrics': evaluation_metrics
    }

    if extra is not None:
        output['Extra'] = extra

    json_path = os.path.join(base_dir, 'output_result.json')
    with open(json_path, 'w') as f:
        json.dump(output, f, indent=2)

    print(f"All results saved to {base_dir}")


def save_json(path: str, data: dict) -> None:
    """
    Save dictionary data as a JSON file with indent=2.
    
    Args:
        path: Path where the JSON file will be saved
        data: Dictionary data to save
    
    Returns:
        None
    
    Raises:
        IOError: If file cannot be written
    """
    with open(path, "w") as f:
        json.dump(data, f, indent=2)

def evaluate_tuning_results(
    X_train: np.ndarray,
    y_train: np.ndarray,
    X_test: np.ndarray,
    y_test: np.ndarray,
    build_model: Callable[[Dict[str, Any], Dict[str, Any]], ModelProtocol],
    build_model_params: Dict[str, Any],
    generate_submission: bool = False,
    submission_filename: str = "submission.csv",
    label_maxs: Optional[np.ndarray] = None,
    prediction_columns: Optional[List[str]] = None,
    model_save_path: Optional[str] = None,  # <-- nuovo parametro opzionale
) -> Dict[str, Any]:
    """
    Generate a comprehensive report following hyperparameter optimization,
    measuring memory, timing, model size, and either classification or regression performance.

    Backward-compatible: if 'evaluation_score_args' is provided in build_model_params,
    uses custom regression evaluation; otherwise defaults to classification accuracy.

    New parameters:
    - generate_submission: If True, generates submission file with predictions
    - submission_filename: Name of the submission CSV file
    - label_maxs: Array to scale predictions back from [0,1] range
    - prediction_columns: Column names for the submission file
    - model_save_path: Optional path to save the trained model (if None, model is not saved)
    """
    import psutil, tracemalloc, pickle, sys, time, os, joblib
    from sklearn.metrics import accuracy_score, mean_squared_error, r2_score
    import pandas as pd

    # Extract config and best_params
    config = build_model_params.get("Config")
    best_params = build_model_params.get("best_params")

    # Instantiate model
    model = build_model(best_params, config)
    proc = psutil.Process()

    # --- TRAINING PHASE ---
    mem_before_train = proc.memory_info().rss / (1024 ** 2)
    tracemalloc.start()
    t0 = time.perf_counter()
    model.fit(X_train, y_train)
    train_duration = time.perf_counter() - t0
    _, peak_train = tracemalloc.get_traced_memory()
    tracemalloc.stop()
    mem_after_train = proc.memory_info().rss / (1024 ** 2)

    # --- INFERENCE PHASE ---
    mem_before_infer = proc.memory_info().rss / (1024 ** 2)
    tracemalloc.start()
    t1 = time.perf_counter()
    y_pred = model.predict(X_test)
    infer_duration = time.perf_counter() - t1
    _, peak_infer = tracemalloc.get_traced_memory()
    tracemalloc.stop()
    mem_after_infer = proc.memory_info().rss / (1024 ** 2)

    # --- MODEL SIZE ---
    model_bytes = pickle.dumps(model)
    model_size = sys.getsizeof(model_bytes)

    # --- SAVE MODEL (optional) ---
    if model_save_path is not None:
        os.makedirs(os.path.dirname(model_save_path) or ".", exist_ok=True)
        joblib.dump(model, model_save_path)
        print(f"[INFO] Model saved successfully at: {model_save_path}")
        print(f"[INFO] Model size (pickle serialized): {model_size / 1024:.2f} KB")
    else:
        print("[INFO] Model not saved (no path provided).")

    # --- PERFORMANCE METRIC ---
    results = {
        'train_duration_s': train_duration,
        'inference_duration_s': infer_duration,
        'model_size_bytes': model_size,
        'psutil_rss_before_train_mb': mem_before_train,
        'psutil_rss_after_train_mb': mem_after_train,
        'psutil_rss_delta_train_mb': mem_after_train - mem_before_train,
        'psutil_rss_before_inference_mb': mem_before_infer,
        'psutil_rss_after_inference_mb': mem_after_infer,
        'psutil_rss_delta_inference_mb': mem_after_infer - mem_before_infer,
        'tracemalloc_peak_train_bytes': peak_train,
        'tracemalloc_peak_inference_bytes': peak_infer,
    }

    # Determine if using custom regression evaluation
    eval_args = build_model_params.get('evaluation_score_args', None)
    if eval_args is not None:
        results['evaluation_score'] = None  # Placeholder for custom metric
    else:
        # Classification: fallback to accuracy
        results['accuracy'] = accuracy_score(y_test, y_pred)

    # --- SUBMISSION GENERATION (NEW FEATURE) ---
    if generate_submission:
        predictions_for_submission = y_pred.copy()

        # Scale predictions back if label_maxs is provided
        if label_maxs is not None:
            predictions_for_submission = predictions_for_submission * label_maxs

        # Set default column names if not provided
        if prediction_columns is None:
            if predictions_for_submission.ndim == 1:
                prediction_columns = ["prediction"]
            else:
                prediction_columns = [f"target_{i}" for i in range(predictions_for_submission.shape[1])]

        # Create submission DataFrame
        submission_df = pd.DataFrame(data=predictions_for_submission, columns=prediction_columns)
        submission_df.to_csv(submission_filename, index_label="sample_index")

        results['submission_file'] = submission_filename
        results['submission_shape'] = predictions_for_submission.shape
        print(f"[INFO] Submission file saved as: {submission_filename}")
        print(f"[INFO] Submission shape: {predictions_for_submission.shape}")

    return results

def get_requirements_versions(project_cfg: Dict[str, Any]) -> Dict[str, str]:
    """
    Reads requirements.txt file and returns a dictionary with library names and installed versions.
    Also adds the dictionary to project_cfg["requirements_versions"].
    
    Args:
        project_cfg: Project configuration dictionary, should contain "requirements_txt_path"
    
    Returns:
        Dictionary mapping library names to installed versions (or None if not installed)
    
    Raises:
        Exception: If file reading fails (prints warning instead of raising)
    """
    requirements_txt_path = project_cfg.get("requirements_txt_path", "requirements.txt")
    requirements_txt = None
    installed_versions = {}

    if os.path.exists(requirements_txt_path):
        with open(requirements_txt_path, "r") as file:
            requirements_txt = file.read().strip()
        
        libraries = [line.strip() for line in requirements_txt.splitlines() if line.strip()]
        
        for library in libraries:
            try:
                version = pkg_resources.get_distribution(library).version
                installed_versions[library] = version
            except pkg_resources.DistributionNotFound:
                print(f"Warning: '{library}' is listed in requirements.txt but not installed.")
                installed_versions[library] = None
    else:
        print(f"Warning: '{requirements_txt_path}' not found. Proceeding without it.")

    project_cfg["requirements_versions"] = installed_versions
    return installed_versions


def get_param_importances(study: optuna.study.Study) -> Dict[str, float]:
    """
    Get parameter importances from a study.
    
    Args:
        study: The completed Optuna study
    
    Returns:
        Dictionary mapping parameter names to importance scores
    """
    importances = optuna.importance.get_param_importances(study)
    return importances
