# main.py
import argparse
import os
import json
import time
import platform
import psutil
import random
import numpy as np
import torch
from datetime import datetime
from sklearn.model_selection import train_test_split
from torchvision.datasets import EuroSAT
from torch.utils.data import random_split
import matplotlib.pyplot as plt
import sys
import copy


# Add missing imports
from sklearn.ensemble import RandomForestClassifier, GradientBoostingClassifier
from sklearn.metrics import accuracy_score
from skimage import feature, color  # type: ignore
from skimage.feature import local_binary_pattern
from skimage.feature import graycomatrix, graycoprops
from skimage.feature import canny

from scipy.stats import skew, kurtosis


from typing import Any, Callable, Dict, Optional, Union

from Utils import utils, p_utils

import ssl             
import urllib.request  

# --- SSL FIX PER REVIEWERS ---
# Forza Python a ignorare la verifica del certificato SSL se fallisce.
# Questo risolve l'errore "certificate verify failed" durante il download del dataset.
try:
    _create_unverified_https_context = ssl._create_unverified_context
except AttributeError:
    # Versioni datate di Python che non hanno HTTPS verification
    pass
else:
    ssl._create_default_https_context = _create_unverified_https_context



debug_mode: bool = True  # Set to True to load only a subset of the dataset for faster iterations during development.
# -----------------------------------------------------------------------------
# Modular pipeline for dataset loading, feature extraction, optimization, and
# result saving. Each function encapsulates a distinct responsibility for
# clarity, testability, and reuse.
# -----------------------------------------------------------------------------

def set_random_seed(seed: int = 42) -> None:
    """Set seeds for reproducibility across libraries.

    Currently sets the seed for: 
    - Python's built-in `random`
    - NumPy
    - PyTorch

    Note:
        If additional libraries are used in the pipeline (e.g., TensorFlow, DGL, etc.),
        their random seeds should also be set here to ensure full reproducibility.
    """
    random.seed(seed)
    np.random.seed(seed)
    torch.manual_seed(seed)


def initialize_environment(cache_dir: str) -> None:
    """Configure environment variables (e.g., cache paths)."""
    os.environ['TORCH_HOME'] = cache_dir


def load_configurations(optuna_cfg_path: str, project_cfg_path: str):
    # Verifica esistenza file prima di caricare
    if not os.path.exists(optuna_cfg_path):
        raise FileNotFoundError(f"Optuna config not found at: {optuna_cfg_path}")
    if not os.path.exists(project_cfg_path):
        raise FileNotFoundError(f"Project config not found at: {project_cfg_path}")

    optuna_cfg = utils.load_json_config(optuna_cfg_path)
    project_cfg = utils.load_json_config(project_cfg_path)

    return optuna_cfg, project_cfg

def prepare_dataset(root_path: str, split_ratio: float):
    """
    Download the EuroSAT dataset and perform a train/test split.

    Args:
        root_path (str): The directory where the dataset will be stored.
        split_ratio (float): The proportion of the dataset to allocate to the test set.

    Returns:
        tuple: A tuple containing the training and testing datasets.
    """
    print(f"[INFO] Checking dataset directory at: '{root_path}'")

    if not os.path.exists(root_path):
        os.makedirs(root_path)
        print(f"[INFO] Directory '{root_path}' created.")

    print("[INFO] Downloading the EuroSAT dataset...")
    dataset = EuroSAT(root=root_path, transform=None, download=True)
    print(f"[INFO] Dataset downloaded successfully. Total samples: {len(dataset)}")


    print("Tipo di dataset:", type(dataset))  # Stampa il tipo


    # If debug_mode is True, load only 20% of the dataset
    # If debug_mode is True, load only 20% of the dataset
    if debug_mode:
        debug_size = int(len(dataset) * 0.10)  # 20% of the dataset
        dataset, _ = random_split(dataset, [debug_size, len(dataset) - debug_size])  # Split dataset to 20% for debug
        print(f"[INFO] Debug mode enabled. Loaded 20% of the dataset.")

    # Perform train/test split
    num_train_samples = int(len(dataset) * (1 - split_ratio))
    num_test_samples = len(dataset) - num_train_samples
    train_dataset, test_dataset = random_split(dataset, [num_train_samples, num_test_samples])

    print(f"[INFO] Dataset split completed.")
    print(f"[INFO] Training samples: {len(train_dataset)}")
    print(f"[INFO] Testing samples: {len(test_dataset)}")

    return train_dataset, test_dataset

def objective(trial, data, augmented_data, config) -> float:

# Recuperiamo le scelte da riga di comando salvate nella config
    model_choice = config.get("cli_model")  # 'rf' o 'xg'
    model_size = config.get("cli_size")   # 'small' o 'big'

    X_train_full, y_train_full = data

    # --- Data Splitting for Optuna ---
    # Split the full training set (80% of data) into training (80%) and validation (20%) subsets.
    X_train, X_valid, y_train, y_valid = train_test_split(
        X_train_full, y_train_full, test_size=config['tr_val_split'], random_state=config['seed']
    )

    
    # --- Feature Category Selection via Optuna Flags ---
    use_color_stats_feature       = trial.suggest_categorical("use_color_stats_feature", [True, False])
    use_color_hist_feature        = trial.suggest_categorical("use_color_hist_feature", [True, False])
    use_glcm_feature              = trial.suggest_categorical("use_glcm_feature", [True, False])
    use_lbp_feature               = trial.suggest_categorical("use_lbp_feature", [True, False])
    use_edge_density_feature      = trial.suggest_categorical("use_edge_density_feature", [True, False])
    use_higher_order_feature      = trial.suggest_categorical("use_higher_order_feature", [True, False])
    
    # Define feature indices for each category based on the new feature vector order:
    # Color Statistics: indices [0:6] (6 features)
    # Color Histograms: indices [6:54] (48 features)
    # GLCM Texture Features: indices [54:58] (4 features)
    # LBP Histogram: indices [58:68] (10 features)
    # Edge Density: index [68] (1 feature)
    # Higher-Order Statistics: indices [69:75] (6 features)
    idx_color_stats   = list(range(0, 6))    if use_color_stats_feature   else []
    idx_color_hist    = list(range(6, 54))   if use_color_hist_feature    else []
    idx_glcm          = list(range(54, 58))  if use_glcm_feature          else []
    idx_lbp           = list(range(58, 68))  if use_lbp_feature           else []
    idx_edge          = [68]                 if use_edge_density_feature  else []
    idx_higher_order  = list(range(69, 75))  if use_higher_order_feature  else []
    selected_indices  = idx_color_stats + idx_color_hist + idx_glcm + idx_lbp + idx_edge + idx_higher_order
    
    # Ensure a minimum of 5 features to proceed.
    if len(selected_indices) < 5:
        return 1.0  # Return a high error to discourage this setting.
    
    X_train_custom: np.ndarray = X_train[:, selected_indices]
    X_valid_custom: np.ndarray = X_valid[:, selected_indices]
    
    # --- Model Selection and Hyperparameter Tuning ---
    #model_choice: str = trial.suggest_categorical("model", ["RandomForest", "GradientBoosting"])

    # --- Model Selection & Hyperparameters Dinamici ---
    
    clf = None

    if model_choice == "rf":
        if model_size == "big":
            n_est = trial.suggest_int("rf_n_estimators", 100, 300)
            max_d = trial.suggest_int("rf_max_depth", 5, 50)
            clf = RandomForestClassifier(
                n_estimators=n_est,
                max_depth=max_d,
                random_state=config['seed']
            )
        elif model_size == "small":
            n_est = trial.suggest_int("rf_n_estimators", 10, 50)
            max_d = trial.suggest_int("rf_max_depth", 3, 10)
            min_leaf = trial.suggest_int("rf_min_samples_leaf", 5, 20)
            max_feat = trial.suggest_float("rf_max_features", 0.3, 0.8)
            clf = RandomForestClassifier(
                n_estimators=n_est,
                max_depth=max_d,
                min_samples_leaf=min_leaf,
                max_features=max_feat,
                random_state=config['seed']
            )
        else:
            print(f"[ERROR] Invalid model_size '{model_size}'. Expected 'big' or 'small'.")
            sys.exit(1)
    elif model_choice == "xg":
        #old configuration 
        if model_size == "big":
            n_est = trial.suggest_int("gb_n_estimators", 100, 300)
            lr    = trial.suggest_float("gb_learning_rate", 0.01, 0.2)
            md    = trial.suggest_int("gb_max_depth", 3, 10)
            clf = GradientBoostingClassifier(
                n_estimators=n_est,
                learning_rate=lr,
                max_depth=md,
                random_state=config['seed']
            )
        elif model_size == "small":
            # Range ristretto per favorire modelli più piccoli e veloci
            n_est = trial.suggest_int("gb_n_estimators", 20, 80)
            lr    = trial.suggest_float("gb_learning_rate", 0.1, 0.3)
            md    = trial.suggest_int("gb_max_depth", 2, 6)
            subs  = trial.suggest_float("gb_subsample", 0.6, 0.9)
            colsb = trial.suggest_float("gb_colsample_bytree", 0.5, 0.9)

            clf = GradientBoostingClassifier(
                n_estimators=n_est,
                learning_rate=lr,
                max_depth=md,
                subsample=subs,
                max_features=colsb,
                random_state=config['seed']
            )
        else:
            print(f"[ERROR] Invalid model_size '{model_size}'. Expected 'big' or 'small'.")
            sys.exit(1)
    else:
        print(f"[ERROR] Invalid model_choice '{model_choice}'. Expected 'rf' or 'xg'.")
        sys.exit(1)

    
    # --- Model Training ---
    clf.fit(X_train_custom, y_train)
    y_pred: np.ndarray = clf.predict(X_valid_custom)
    accuracy: float = accuracy_score(y_valid, y_pred)
    
    
    # Save the selected feature category flags.
    trial.set_user_attr("selected_feature_categories", {
        "use_color_stats_feature": use_color_stats_feature,
        "use_color_hist_feature": use_color_hist_feature,
        "use_glcm_feature": use_glcm_feature,
        "use_lbp_feature": use_lbp_feature,
        "use_edge_density_feature": use_edge_density_feature,
        "use_higher_order_feature": use_higher_order_feature
    })

    # Save chosen model info for reference
    trial.set_user_attr("model_type", model_choice)
    trial.set_user_attr("model_size", model_size)
    
    return 1.0 - accuracy

def extract_classic_features(dataset_subset, feature_flags: dict) -> (np.ndarray, np.ndarray):
    features_list: list = []
    labels_list: list = []
    for img, label in dataset_subset:
        feat: np.ndarray = compute_features(img, feature_flags)
        features_list.append(feat)
        labels_list.append(label)
    return np.array(features_list), np.array(labels_list)

def compute_features(pil_img, feature_flags: dict) -> np.ndarray:
    """
    Compute classical features for a given PIL image based on feature flags.
    """
    img_np: np.ndarray = np.array(pil_img)

    # If the image is grayscale, stack it to simulate 3 channels.
    if img_np.ndim == 2:
        img_np = np.stack((img_np,) * 3, axis=-1)

    feature_vector_parts = []

    # Prepare variables that might be reused
    gray_img = None
    gray_img_uint8 = None

    # === Color Statistics ===
    if feature_flags.get('use_color_stats_feature', False):
        mean_channels: np.ndarray = np.mean(img_np, axis=(0, 1))
        std_channels: np.ndarray = np.std(img_np, axis=(0, 1))
        feature_vector_parts.append(mean_channels)
        feature_vector_parts.append(std_channels)

    # === Color Histograms ===
    if feature_flags.get('use_color_hist_feature', False):
        hist_bins: int = 16
        hist_features_list: list = []
        for c in range(3):
            hist, _ = np.histogram(img_np[..., c], bins=hist_bins, range=(0, 255), density=True)
            hist_features_list.extend(hist)
        hist_features: np.ndarray = np.array(hist_features_list)
        feature_vector_parts.append(hist_features)

    # === GLCM Texture Features ===
    if feature_flags.get('use_glcm_feature', False):
        if gray_img is None:
            gray_img = color.rgb2gray(img_np)
        quantized_img: np.ndarray = np.floor(gray_img * 8).astype(np.uint8)
        glcm: np.ndarray = feature.graycomatrix(quantized_img, distances=[1], angles=[0], symmetric=True, normed=True)
        contrast_val: float = feature.graycoprops(glcm, prop='contrast')[0, 0]
        correlation_val: float = feature.graycoprops(glcm, prop='correlation')[0, 0]
        energy_val: float = feature.graycoprops(glcm, prop='energy')[0, 0]
        homogeneity_val: float = feature.graycoprops(glcm, prop='homogeneity')[0, 0]
        glcm_texture_features: np.ndarray = np.array([contrast_val, correlation_val, energy_val, homogeneity_val])
        feature_vector_parts.append(glcm_texture_features)

    # === LBP Texture Features ===
    if feature_flags.get('use_lbp_feature', False):
        if gray_img is None:
            gray_img = color.rgb2gray(img_np)
        if gray_img_uint8 is None:
            gray_img_uint8 = (gray_img * 255).astype(np.uint8)
        lbp: np.ndarray = feature.local_binary_pattern(gray_img_uint8, P=8, R=1, method='uniform')
        lbp_bins: int = 10
        lbp_hist: np.ndarray = np.histogram(lbp, bins=lbp_bins, range=(0, lbp.max() + 1), density=True)[0]
        feature_vector_parts.append(lbp_hist)

    # === Edge Density ===
    if feature_flags.get('use_edge_density_feature', False):
        if gray_img is None:
            gray_img = color.rgb2gray(img_np)
        edges: np.ndarray = canny(gray_img)
        edge_density: float = float(np.sum(edges) / edges.size)
        feature_vector_parts.append(np.array([edge_density]))

    # === Higher-Order Statistics ===
    if feature_flags.get('use_higher_order_feature', False):
        num_channels: int = img_np.shape[2]
        channel_skews_list: list = []
        channel_kurtoses_list: list = []
        for c in range(num_channels):
            channel_data: np.ndarray = img_np[:, :, c].ravel()
            skew_val: float = float(skew(channel_data))
            kurtosis_val: float = float(kurtosis(channel_data))
            channel_skews_list.append(skew_val)
            channel_kurtoses_list.append(kurtosis_val)
        skew_features: np.ndarray = np.array(channel_skews_list)
        kurtosis_features: np.ndarray = np.array(channel_kurtoses_list)
        feature_vector_parts.append(skew_features)
        feature_vector_parts.append(kurtosis_features)

    # --- Concatenate all parts ---
    feature_vector: np.ndarray = np.concatenate(feature_vector_parts)

    return feature_vector

def my_model_builder(
    params: Dict[str, Any],
    config: Dict[str, Any]
) -> utils.ModelProtocol:
    """
    Instantiate a classifier based on the provided parameters.

    Args:
        params (dict): Hyperparameters, must include 'model' key specifying type.
        config (dict): Configuration dictionary, should contain 'seed'.

    Returns:
        BaseEstimator: An unfitted scikit-learn classifier.

    Raises:
        ValueError: If 'model' type is unsupported.
    """

    #model_type = params.get("model")
    model_type = config.get("cli_model", "rf")
    seed = config.get('seed', None)

    if model_type == 'rf':
        return RandomForestClassifier(
            n_estimators=params.get('rf_n_estimators', 100),
            max_depth=params.get('rf_max_depth', None),
            min_samples_leaf=params.get('rf_min_samples_leaf', 1),
            max_features=params.get('rf_max_features', 1.0), 
            random_state=seed,
            n_jobs=-1
        )
    elif model_type == 'xg':
        return GradientBoostingClassifier(
            n_estimators=params.get('gb_n_estimators', 100),
            learning_rate=params.get('gb_learning_rate', 0.1),
            max_depth=params.get('gb_max_depth', 3),
            subsample=params.get('gb_subsample', 1.0),
            max_features=params.get('gb_colsample_bytree', None),
            random_state=seed
        )
    else:
        raise ValueError(f"Unsupported model type: {model_type}")

def _get_selected_features(optuna_run_metrics: Dict[str, Any]) -> Dict[str, Any]:
    """
    Extract parameters ending with '_feature' from best_result['hyperparameters'].

    Args:
        optuna_run_metrics (Dict[str, Any]): Dictionary containing Optuna run results.

    Returns:
        Dict[str, Any]: Dictionary of selected feature parameters.
    """
    l_dct_selected_features: Dict[str, Any] = {}

    l_dct_best_result: Dict[str, Any] = optuna_run_metrics.get("best_result", {})
    l_dct_params: Dict[str, Any] = l_dct_best_result.get("hyperparameters", {})

    for l_s_param_name, l_x_param_value in l_dct_params.items():
        if l_s_param_name.endswith("_feature"):
            l_dct_selected_features[l_s_param_name] = l_x_param_value

    return l_dct_selected_features  
if __name__ == "__main__":
    parser = argparse.ArgumentParser(description="EuroSAT Classification Pipeline")
    parser.add_argument("--model", type=str, default="rf", choices=["rf", "xg"], dest="cli_model",
                        help="Model type: 'rf' or 'xg'")
    parser.add_argument("--size", type=str, default="small", choices=["small", "big"], dest="cli_size",
                        help="Search space size")
    args = parser.parse_args()

    # --- Calcolo Percorsi (Ancoraggio alla root del progetto) ---
    current_script_path = os.path.abspath(__file__)
    src_dir = os.path.dirname(current_script_path)          # .../src/pipelineMl
    src_root = os.path.dirname(src_dir)                     # .../src
    project_root = os.path.dirname(src_root)                # .../EuroSAT
    
    # Percorsi file config
    config_dir = os.path.join(project_root, "configuration", "jsonConfigurations")
    optuna_cfg_file = os.path.join(config_dir, "optuna_config_ml.json")
    project_cfg_file = os.path.join(config_dir, "project_config.json")

    # Load Configuration
    try:
        optuna_cfg_initial, project_cfg_initial = load_configurations(optuna_cfg_file, project_cfg_file)
    except FileNotFoundError as e:
        print(f"[CRITICAL ERROR] {e}")
        sys.exit(1)

    # Iniezione parametri CLI
    project_cfg_initial["cli_model"] = args.cli_model
    project_cfg_initial["cli_size"] = args.cli_size

    # --- FIX PATH RELATIVI (Fondamentale per portabilità) ---
    # 1. Dataset
    raw_ds_path = project_cfg_initial['dataset_root_path']
    if not os.path.isabs(raw_ds_path):
        project_cfg_initial['dataset_root_path'] = os.path.join(project_root, raw_ds_path)
    
    # 2. Cache
    raw_cache_path = project_cfg_initial['cache_directory']
    if not os.path.isabs(raw_cache_path):
        project_cfg_initial['cache_directory'] = os.path.join(project_root, raw_cache_path)

    # 3. Output Base
    raw_output_path = project_cfg_initial["output_paths"]["output_result_path"]
    if not os.path.isabs(raw_output_path):
        raw_output_path = os.path.join(project_root, raw_output_path)

    # Setup Ambiente
    set_random_seed(int(project_cfg_initial['seed']))
    initialize_environment(project_cfg_initial["cache_directory"])

    # Caricamento Dataset (Ora usa il path assoluto corretto)
    ds_train, ds_test = prepare_dataset(project_cfg_initial['dataset_root_path'], 
                                        project_cfg_initial['train_test_split'])
    

    print(f" Feature extraction...")
    all_flags = {
        "use_color_stats_feature": True, "use_color_hist_feature": True,
        "use_glcm_feature": True, "use_lbp_feature": True,
        "use_edge_density_feature": True, "use_higher_order_feature": True
    }

    X_train, y_train, X_test, y_test, _, _, all_feat_metrics = p_utils.extract_features_split(
        ds_train, ds_test, None, extract_classic_features, 
        extractor_params_optional={"feature_flags": all_flags}
    )
    print(f"Feature extraction completed.")
    
    experiment_folder_name = f"{args.cli_model}_{args.cli_size}"
    NUM_RUNS = 5

    # =========================================================================
    #                               LOOP 5 RUN
    # =========================================================================
    for run_idx in range(1, NUM_RUNS + 1):
        print(f"\n{'#'*60}")
        print(f" STARTING RUN {run_idx}/{NUM_RUNS} | Model: {args.cli_model} | Size: {args.cli_size}")
        print(f"{'#'*60}\n")

        # [FIX] Deepcopy per isolare le configurazioni di ogni run
        project_cfg = copy.deepcopy(project_cfg_initial)
        optuna_cfg = copy.deepcopy(optuna_cfg_initial)

        # Costruzione Path specifico: .../Result/rf_small/test_1
        run_output_path = os.path.join(raw_output_path, experiment_folder_name, f"test_{run_idx}")
        project_cfg["output_paths"]["output_result_path"] = run_output_path


        print(f"[INFO-RUN-{run_idx}] Starting Optuna optimization...")
        optuna_run_metrics = utils.run_optimization(
            objective, (X_train, y_train), None, project_cfg, optuna_cfg
        )
        
        print(f"[INFO-RUN-{run_idx}] Evaluation & Saving...")
        best_feat_flags = _get_selected_features(optuna_run_metrics)
        X_train_sel, y_train_sel, X_test_sel, y_test_sel, _, _, sel_feat_metrics = p_utils.extract_features_split(
            ds_train, ds_test, None, extract_classic_features, 
            extractor_params_optional={"feature_flags": best_feat_flags}
        )

        evaluation_metrics = utils.evaluate_tuning_results(
            X_train_sel, y_train_sel, X_test_sel, y_test_sel,
            build_model=my_model_builder,
            build_model_params={
                "Config": project_cfg,
                "best_params": optuna_run_metrics.get("best_result", {}).get("hyperparameters", {})
            }
        )
    
        utils.save_run_results(
            project_cfg=project_cfg,
            cfg_file_path={'optuna': optuna_cfg_file, 'project': project_cfg_file},
            feature_metrics=all_feat_metrics,
            optuna_run_metrics=optuna_run_metrics,
            evaluation_metrics=evaluation_metrics,
            extra=sel_feat_metrics
        )
        
        print(f"[INFO] RUN {run_idx} COMPLETED SUCCESSFULLY.\n")


