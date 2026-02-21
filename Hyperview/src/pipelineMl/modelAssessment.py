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
from glob import glob
from tqdm import tqdm
import pandas as pd
import sys
import copy
import pywt
import glob as python_glob  # renamed to avoid collision with custom glob usage

# Add missing imports
from sklearn.ensemble import RandomForestRegressor, GradientBoostingRegressor
from sklearn.multioutput import MultiOutputRegressor
from sklearn.metrics import mean_squared_error
from sklearn.model_selection import KFold, StratifiedKFold
from sklearn.neighbors import KNeighborsRegressor
from sklearn import preprocessing
from scipy.stats import skew, kurtosis
from typing import Any, Callable, Dict, Optional, Union
from Utils import utils, p_utils
from argparse import Namespace

# --- CONFIGURAZIONE COSTANTI HYPERVIEW ---
LABEL_NAMES = ["P2O5", "K", "Mg", "pH"]
LABEL_MAXS = np.array([325.0, 625.0, 400.0, 7.8])
COL_IX = [0, 1, 2, 3]
args_eval = Namespace(col_ix=COL_IX, cons=LABEL_MAXS)

debug_mode: bool = False

# -----------------------------------------------------------------------------
# Modular pipeline for dataset loading, feature extraction, optimization, and
# result saving. Each function encapsulates a distinct responsibility for
# clarity, testability, and reuse.
# -----------------------------------------------------------------------------

def set_random_seed(seed: int = 42) -> None:
    random.seed(seed)
    np.random.seed(seed)
    torch.manual_seed(seed)

def initialize_environment(cache_dir: str) -> None:
    os.environ['TORCH_HOME'] = cache_dir

def load_configurations(optuna_cfg_path: str, project_cfg_path: str):
    if not os.path.exists(optuna_cfg_path):
        raise FileNotFoundError(f"Optuna config not found at: {optuna_cfg_path}")
    if not os.path.exists(project_cfg_path):
        raise FileNotFoundError(f"Project config not found at: {project_cfg_path}")
    optuna_cfg = utils.load_json_config(optuna_cfg_path)
    project_cfg = utils.load_json_config(project_cfg_path)
    return optuna_cfg, project_cfg

def load_gt(file_path: str):
    """Load labels for train set from the ground truth file."""
    gt_file = pd.read_csv(file_path)
    labels = gt_file[["P", "K", "Mg", "pH"]].values / LABEL_MAXS
    return labels

def load_data(directory: str, gt_file_path: str = None, is_train: bool = True):
    datalist = []
    masklist = []

    if is_train and gt_file_path:
        labels = load_gt(gt_file_path)

    all_files = np.array(
        sorted(
            glob(os.path.join(directory, "*.npz")),
            key=lambda x: int(os.path.basename(x).replace(".npz", "")),
        )
    )

    if debug_mode:
        debug_size = int(len(all_files) * 0.10)
        all_files = all_files[:debug_size]
        if is_train and gt_file_path:
            labels = labels[:debug_size]
        print(f"[INFO] Debug mode enabled. Loading {debug_size} samples.")

    for idx, file_name in tqdm(enumerate(all_files), total=len(all_files), desc=f"Loading {'training' if is_train else 'test'} data"):
        try:
            with np.load(file_name) as npz:
                mask = npz["mask"]
                data = npz["data"]
                datalist.append(data)
                masklist.append(mask)
        except Exception as e:
            print(f"[WARNING] Error loading {file_name}: {e}")
            continue

    if is_train and gt_file_path:
        return datalist, masklist, labels
    else:
        return datalist, masklist

def apply_hyperview_augmentation(X_train, M_train, y_train, augment_constant: int = 1):
    print(f"[INFO] Applying Hyperview augmentation with constant: {augment_constant}")
    
    aug_datalist = []
    aug_masklist = []
    aug_labellist = []
    
    for i in range(augment_constant):
        print(f"[INFO] Augmentation iteration {i + 1}/{augment_constant}")
        for idx, (data, mask, label) in tqdm(enumerate(zip(X_train, M_train, y_train)), total=len(X_train), desc=f"Augmenting data - iteration {i + 1}"):
            aug_data, aug_mask, aug_label = apply_hyperview_single_augmentation(data, mask, label)
            aug_datalist.append(aug_data)
            aug_masklist.append(aug_mask)
            aug_labellist.append(aug_label)
            
    print(f"[INFO] Augmentation completed. Generated {len(aug_datalist)} augmented samples.")
    return aug_datalist, aug_masklist, np.array(aug_labellist)

def apply_hyperview_single_augmentation(data, mask, label):
    flag = True
    ma = np.max(data, keepdims=True)
    sh = data.shape[1:]
    
    for attempt in range(10): 
        edge = 11  
        x = np.random.randint(sh[0] + 1 - edge)
        y = np.random.randint(sh[1] + 1 - edge)
        
        if np.sum(mask[0, x : (x + edge), y : (y + edge)]) > 120: 
            aug_data = (data[:, x : (x + edge), y : (y + edge)]
                        + np.random.uniform(-0.01, 0.01, (150, edge, edge)) * ma)
            aug_mask = mask[:, x : (x + edge), y : (y + edge)] | np.random.randint(0, 1, (150, edge, edge))
            flag = False  
            break

    if flag: 
        max_edge = np.max(sh)
        min_edge = np.min(sh) 
        edge = min_edge
        x = np.random.randint(sh[0] + 1 - edge)
        y = np.random.randint(sh[1] + 1 - edge)
        aug_data = (data[:, x : (x + edge), y : (y + edge)]
                    + np.random.uniform(-0.001, 0.001, (150, edge, edge)) * ma)
        aug_mask = mask[:, x : (x + edge), y : (y + edge)] | np.random.randint(0, 1, (150, edge, edge))

    aug_label = label + label * np.random.uniform(-0.001, 0.001, 4)
    return aug_data, aug_mask, aug_label

class SpectralCurveFiltering:
    def __init__(self, merge_function=np.mean):
        self.merge_function = merge_function

    def __call__(self, sample: np.ndarray):
        return self.merge_function(sample, axis=(1, 2))

def extract_features_with_flags(data_list, mask_list, feature_flags=None):
    default_flags = {
        'arr': True, 'dXdl': True, 'd2Xdl2': True, 'd3Xdl3': True, 'dXds1': True,
        's0': True, 's1': True, 's2': True, 's3': True, 's4': True,
        'real': True, 'imag': True, 'reals': True, 'imags': True,
        'cAw1': True, 'cAw2': True
    }
    if feature_flags is None:
        feature_flags = default_flags
    else:
        feature_flags = {**default_flags, **feature_flags}

    def _shape_pad(data):
        max_edge = np.max(data.shape[1:])
        shape = (max_edge, max_edge)
        return np.pad(data, ((0, 0), (0, (shape[0] - data.shape[1])), (0, (shape[1] - data.shape[2]))), "wrap")

    filtering = SpectralCurveFiltering()
    w1, w2 = pywt.Wavelet("sym3"), pywt.Wavelet("dmey")

    processed_data, average_edge = [], []

    for idx, (data, mask) in enumerate(
        tqdm(zip(data_list, mask_list), total=len(data_list), desc="INFO: Preprocessing data with feature flags...")
    ):
        data = data / 2210
        m = 1 - mask.astype(int)
        image = data * m
        average_edge.append((image.shape[1] + image.shape[2]) / 2)
        image = _shape_pad(image)

        s = np.linalg.svd(image, full_matrices=False, compute_uv=False)
        s0, s1, s2, s3, s4 = s[:, 0], s[:, 1], s[:, 2], s[:, 3], s[:, 4]
        dXds1 = s0 / (s1 + np.finfo(float).eps)

        arr = filtering(np.ma.MaskedArray(data, mask))

        cAw1, cAw2 = None, None
        if feature_flags['cAw2']:
            cA0, _ = pywt.dwt(arr, wavelet=w2, mode="constant")
            cAx, _ = pywt.dwt(cA0[12:92], wavelet=w2, mode="constant")
            cAy, _ = pywt.dwt(cAx[15:55], wavelet=w2, mode="constant")
            cAz, _ = pywt.dwt(cAy[15:35], wavelet=w2, mode="constant")
            cAw2 = np.concatenate((cA0[12:92], cAx[15:55], cAy[15:35], cAz[15:25]), -1)

        if feature_flags['cAw1']:
            cA0, _ = pywt.dwt(arr, wavelet=w1, mode="constant")
            cAx, _ = pywt.dwt(cA0[1:-1], wavelet=w1, mode="constant")
            cAy, _ = pywt.dwt(cAx[1:-1], wavelet=w1, mode="constant")
            cAz, _ = pywt.dwt(cAy[1:-1], wavelet=w1, mode="constant")
            cAw1 = np.concatenate((cA0, cAx, cAy, cAz), -1)

        dXdl, d2Xdl2, d3Xdl3 = np.gradient(arr, axis=0), np.gradient(np.gradient(arr, axis=0), axis=0), np.gradient(np.gradient(np.gradient(arr, axis=0), axis=0), axis=0)

        fft = np.fft.fft(arr)
        real, imag = np.real(fft), np.imag(fft)
        ffts = np.fft.fft(s0)
        reals, imags = np.real(ffts), np.imag(ffts)

        features_to_concat = []
        if feature_flags['arr']: features_to_concat.append(arr)
        if feature_flags['dXdl']: features_to_concat.append(dXdl)
        if feature_flags['d2Xdl2']: features_to_concat.append(d2Xdl2)
        if feature_flags['d3Xdl3']: features_to_concat.append(d3Xdl3)
        if feature_flags['dXds1']: features_to_concat.append(dXds1)
        if feature_flags['s0']: features_to_concat.append(s0)
        if feature_flags['s1']: features_to_concat.append(s1)
        if feature_flags['s2']: features_to_concat.append(s2)
        if feature_flags['s3']: features_to_concat.append(s3)
        if feature_flags['s4']: features_to_concat.append(s4)
        if feature_flags['real']: features_to_concat.append(real)
        if feature_flags['imag']: features_to_concat.append(imag)
        if feature_flags['reals']: features_to_concat.append(reals)
        if feature_flags['imags']: features_to_concat.append(imags)
        if feature_flags['cAw1'] and cAw1 is not None: features_to_concat.append(cAw1)
        if feature_flags['cAw2'] and cAw2 is not None: features_to_concat.append(cAw2)

        out_features = np.concatenate(features_to_concat, -1) if features_to_concat else arr
        processed_data.append(out_features)

    return np.array(processed_data), np.array(average_edge)

def extract_hyperview_features_wrapper(dataset_info, feature_flags=None):
    data_list = dataset_info['data_list']
    mask_list = dataset_info['mask_list']
    labels = dataset_info.get('labels', None)
    
    features_array, _ = extract_features_with_flags(data_list, mask_list, feature_flags)
    
    if labels is not None:
        labels_array = np.array(labels)
    else:
        labels_array = np.zeros(len(features_array))

    return features_array, labels_array
AI4EO_DATASET_PAGE = "https://platform.ai4eo.eu/seeing-beyond-the-visible-permanent"
EOTDL_DATASET_PAGE = "https://www.eotdl.com/datasets/SeeingBeyondTheVisible"

def prepare_dataset(root_path: str, train_test_split: float = None):
    """
    Load Hyperview dataset from disk.

    Note: The HYPERVIEW dataset is not distributed with this repository and is
    not downloaded automatically.
    """

    print(f"[INFO] Preparing dataset at: '{root_path}'")

    train_data_dir = os.path.join(root_path, "train_data")
    test_data_dir = os.path.join(root_path, "test_data")
    gt_data_path = os.path.join(root_path, "train_gt.csv")

    dataset_ready = (
        os.path.isdir(train_data_dir)
        and os.path.isdir(test_data_dir)
        and os.path.isfile(gt_data_path)
    )

    if not dataset_ready:
        abs_root = os.path.abspath(root_path)
        raise FileNotFoundError(
            "HYPERVIEW dataset not found locally.\n\n"
            f"Expected structure under: {abs_root}\n"
            "  train_data/   (NPZ files)\n"
            "  test_data/    (NPZ files)\n"
            "  train_gt.csv\n\n"
            "Please obtain the dataset from the official source (registration and acceptance of terms may be required):\n"
            f"  {AI4EO_DATASET_PAGE}\n\n"
            "Download instructions (EOTDL):\n"
            f"  {EOTDL_DATASET_PAGE}\n\n"
            "After downloading, set `dataset_root_path` to the folder that contains the files above "
            "(default config points to `Hyperview/data`)."
        )

    print(f"[INFO] Loading training data from: {train_data_dir}")
    print(f"[INFO] Loading test data from: {test_data_dir}")
    print(f"[INFO] Loading ground truth from: {gt_data_path}")

    X_train, M_train, y_train = load_data(train_data_dir, gt_data_path, is_train=True)
    X_test, M_test = load_data(test_data_dir, None, is_train=False)

    print("[INFO] Dataset loading completed.")
    print(f"[INFO] Training samples: {len(X_train)}")
    print(f"[INFO] Testing samples: {len(X_test)}")

    return X_train, M_train, y_train, X_test, M_test

def my_model_builder(params: Dict[str, Any], config: Dict[str, Any]) -> utils.ModelProtocol:
    """
    Instantiate a multi-output regressor based on CLI / Optuna parameters.
    """
    model_type = config.get("cli_model", "rf")
    seed = config.get("seed", None)

    if model_type == "rf":
        return RandomForestRegressor(
            n_estimators=params.get("rf_n_estimators", 100),
            max_depth=params.get("rf_max_depth", None),
            min_samples_leaf=params.get("rf_min_samples_leaf", 1),
            max_features=params.get("rf_max_features", 1.0),
            random_state=seed,
            n_jobs=-1
        )

    elif model_type == "xg":
        base = GradientBoostingRegressor(
            n_estimators=params.get("gb_n_estimators", 100),
            learning_rate=params.get("gb_learning_rate", 0.1),
            max_depth=params.get("gb_max_depth", 3),
            subsample=params.get("gb_subsample", 1.0),
            max_features=params.get("gb_colsample_bytree", None),
            random_state=seed
        )
        return MultiOutputRegressor(base)

    else:
        raise ValueError(f"Unsupported model type: {model_type}")

# -----------------------------------------------------------------------------
# Function to Find Best Params Automatically
# -----------------------------------------------------------------------------
def find_best_params_from_history(base_results_path: str, model_type: str, model_size: str) -> Dict[str, Any]:
    """
    Scans the results directory, inspects 'Optuna' -> 'best_result' -> 'score',
    finds the run with the LOWEST score (minimize error), and returns its hyperparameters.
    """
    search_pattern = os.path.join(base_results_path, "ModelSelection", f"{model_type}_{model_size}", "*", "output_result.json")
    files = python_glob.glob(search_pattern)
    
    if not files:
        print(f"[WARNING] No previous results found in {search_pattern}. Cannot auto-load params.")
        return {}

    best_score = float('inf') # Regression minimize error (lower is better)
    best_params = {}
    best_file = ""
    found_any = False

    for f_path in files:
        try:
            with open(f_path, 'r') as f:
                data = json.load(f)
                
                optuna_section = data.get("Optuna", {})
                best_result = optuna_section.get("best_result", {})
                
                current_score = best_result.get("score")
                current_params = best_result.get("hyperparameters")

                if current_score is not None and current_params:
                    # In Hyperview score is an error metric (evaluation_score), lower is better
                    if current_score < best_score:
                        best_score = current_score
                        best_params = current_params
                        best_file = f_path
                        found_any = True
                        
        except Exception as e:
            print(f"[WARN] Error reading {f_path}: {e}")

    if found_any:
        print(f"[INFO] Auto-selected best params from: {best_file}")
        print(f"[INFO] Best Optuna Score (Error Metric): {best_score:.4f}")
    else:
        print("[WARN] Could not find valid Optuna best_result data in scanned files.")

    return best_params


# -----------------------------------------------------------------------------
# Main Execution
# -----------------------------------------------------------------------------
if __name__ == "__main__":
    parser = argparse.ArgumentParser(description="Hyperview Model Assessment (Retraining)")
    parser.add_argument("--model", type=str, default="rf", choices=["rf", "xg"], dest="cli_model", help="Model type")
    parser.add_argument("--size", type=str, default="small", choices=["small", "big"], dest="cli_size", help="Config size")
    parser.add_argument("--best-params-mode", type=str, default="manual", choices=["manual", "auto"], 
                        help="How to get hyperparameters: 'manual' (hardcoded) or 'auto' (from best previous result)")
    parser.add_argument("--debug", action="store_true", help="Enable debug/subsampling mode (loads a small subset).")
    args = parser.parse_args()

    debug_mode = args.debug

    # --- Path Resolution ---
    current_script_path = os.path.abspath(__file__)
    src_dir = os.path.dirname(current_script_path)
    src_root = os.path.dirname(src_dir)
    project_root = os.path.dirname(src_root)
    
    config_dir = os.path.join(project_root, "configuration", "jsonConfigurations")
    optuna_cfg_file = os.path.join(config_dir, "optuna_config_ml.json")
    project_cfg_file = os.path.join(config_dir, "project_config.json")

    try:
        optuna_cfg, project_cfg = load_configurations(optuna_cfg_file, project_cfg_file)
    except FileNotFoundError as e:
        print(f"[CRITICAL ERROR] {e}")
        sys.exit(1)

    # Iniezione CLI
    project_cfg["cli_model"] = args.cli_model
    project_cfg["cli_size"] = args.cli_size

    # Fix Paths
    if not os.path.isabs(project_cfg['dataset_root_path']):
        project_cfg['dataset_root_path'] = os.path.join(project_root, project_cfg['dataset_root_path'])
    if not os.path.isabs(project_cfg['cache_directory']):
        project_cfg['cache_directory'] = os.path.join(project_root, project_cfg['cache_directory'])
    
    raw_output_path = project_cfg["output_paths"]["output_result_path"]
    if not os.path.isabs(raw_output_path):
        raw_output_path = os.path.join(project_root, raw_output_path)

    raw_req_path = project_cfg.get("requirements_txt_path", "requirements.txt")
    if not os.path.isabs(raw_req_path):
        project_cfg["requirements_txt_path"] = os.path.join(project_root, raw_req_path) 

    # Setup Ambiente
    global_seed = int(project_cfg['seed'])
    print(f"[INFO] Setting GLOBAL seed to {global_seed} for Dataset Split consistency.")
    set_random_seed(global_seed)    
    initialize_environment(project_cfg["cache_directory"])

    # Load Hyperview dataset (train_test_split is ignored since dataset is pre-divided)
    X_train, M_train, y_train, X_test, M_test = prepare_dataset(
        project_cfg['dataset_root_path'], 
        project_cfg['train_test_split']
    )
    
    # Define augmentation parameters
    augmentation_params = {
        "augment_constant": 1 
    }
    
    # Apply augmentation
    X_aug_train, M_aug_train, y_aug_train, aug_metrics = utils.preprocessing_data(
        X_train=X_train, M_train=M_train, y_train=y_train,
        augmentation_fn=apply_hyperview_augmentation,
        augmentation_params_optional=augmentation_params
    )

    # =========================================================================
    # 1. RETRIEVE BEST PARAMETERS
    # =========================================================================
    best_params = {}

    if args.best_params_mode == "auto":
        print(f"[INFO] Searching for best parameters automatically...")
        base_selection_path = os.path.join(raw_output_path, "..", "ModelSelection") 
        if not os.path.exists(base_selection_path):
             base_selection_path = raw_output_path
        
        best_params = find_best_params_from_history(base_selection_path, args.cli_model, args.cli_size)
        if not best_params:
            print("[ERROR] Auto-mode failed. Falling back to manual mode defaults.")
            args.best_params_mode = "manual"

    if args.best_params_mode == "manual":
        print(f"[INFO] Using manual best parameters.")
        if args.cli_model == 'xg' and args.cli_size == 'small':
            best_params = {
                "use_spectral": False,
                "use_grad": True,
                "use_svd": True,
                "use_fft": True,
                "use_wv1": True,
                "use_wv2": False,
                "gb_n_estimators": 70,
                "gb_learning_rate": 0.11117122948337661,
                "gb_max_depth": 6,
                "gb_subsample": 0.8431831524232583,
                "gb_colsample_bytree": 0.5181412045603372
            }
        elif args.cli_model == 'rf' and args.cli_size == 'small':
            best_params = {
                "use_spectral": False,
                "use_grad": False,
                "use_svd": True,
                "use_fft": True,
                "use_wv1": False,
                "use_wv2": False,
                "rf_n_estimators": 24,
                "rf_max_depth": 10,
                "rf_min_samples_leaf": 7,
                "rf_max_features": 0.5214543139188026
            }
        elif args.cli_model == 'rf' and args.cli_size == 'big':
            best_params = {
                "use_spectral": False,
                "use_grad": True,
                "use_svd": False,
                "use_fft": True,
                "use_wv1": False,
                "use_wv2": False,
                "rf_n_estimators": 192,
                "rf_max_depth": 44
            }
        elif args.cli_model == 'xg' and args.cli_size == 'big':
            best_params = {
                "use_spectral": False,
                "use_grad": True,
                "use_svd": True,
                "use_fft": True,
                "use_wv1": False,
                "use_wv2": True,
                "gb_n_estimators": 204,
                "gb_learning_rate": 0.022359159775903782,
                "gb_max_depth": 9
            }
        # Aggiungere eventuali fallback hardcoded qui per altre varianti come rf_small, rf_big, xg_big
        else:
            print(f"[WARNING] No manual params defined for model {args.cli_model} and size {args.cli_size}. Using empty params.")
            best_params = {}

    print(f"[INFO] Best Params Loaded: {best_params}")

    # =========================================================================
    # 2. FEATURE EXTRACTION (ONE TIME) BASED ON BEST PARAMS
    # =========================================================================
    print(f"\n{'='*60}")
    print(f" [RETRAIN] Extracting SPECIFIC features for best configuration...")
    print(f"{'='*60}\n")
    
    # Map Optuna variables to feature extraction flags
    use_spectral = best_params.get("use_spectral", True)
    use_grad = best_params.get("use_grad", True)
    use_svd = best_params.get("use_svd", True)
    use_fft = best_params.get("use_fft", True)
    use_wv1 = best_params.get("use_wv1", True)
    use_wv2 = best_params.get("use_wv2", True)

    feature_flags = {
        'arr': use_spectral,
        'dXdl': use_grad, 'd2Xdl2': use_grad, 'd3Xdl3': use_grad,
        'dXds1': use_svd, 's0': use_svd, 's1': use_svd, 's2': use_svd, 's3': use_svd, 's4': use_svd,
        'real': use_fft, 'imag': use_fft, 'reals': use_fft, 'imags': use_fft,
        'cAw1': use_wv1,
        'cAw2': use_wv2
    }

    train_dataset_info = {'data_list': X_train, 'mask_list': M_train, 'labels': y_train}
    test_dataset_info = {'data_list': X_test, 'mask_list': M_test, 'labels': None}
    train_aug_dataset_info = {'data_list': X_aug_train, 'mask_list': M_aug_train, 'labels': y_aug_train}

    X_train_ex, y_train_ex, X_test_final, y_test_final, X_train_aug_ex, y_train_aug_ex, \
    _, _, _, feat_metrics = p_utils.extract_features_split(
        train_dataset_info,
        test_dataset_info,
        train_aug_dataset_info,
        extract_hyperview_features_wrapper,
        extractor_params_optional={"feature_flags": feature_flags},
        return_field_sizes=True
    )
    
    # Concatena il training originale e gli augmentati (come fatto nel tuning, per preservare le dimensioni dei dati d'addestramento)
    X_train_final = np.vstack([X_train_ex, X_train_aug_ex])  
    y_train_final = np.vstack([y_train_ex, y_train_aug_ex]) 

    print(f" [RETRAIN] Features ready. Full Train Shape (Orig+Aug): {X_train_final.shape}")

    # =========================================================================
    # 3. ASSESSMENT LOOP (5 SEEDS)
    # =========================================================================
    assessment_folder_name = f"ModelAssessment/{args.cli_model}_{args.cli_size}"
    SEEDS = [1, 2, 3, 4, 5]

    all_scores = []
    all_inference_times = []
    all_model_sizes = []

    for seed_val in SEEDS:
        print(f"\n{'#'*60}")
        print(f" STARTING ASSESSMENT SEED {seed_val} | Model: {args.cli_model}")
        print(f"{'#'*60}\n")
        
        current_project_cfg = copy.deepcopy(project_cfg)
        current_project_cfg['seed'] = seed_val
        set_random_seed(seed_val)

        seed_output_path = os.path.join(raw_output_path, assessment_folder_name, f"test_{seed_val}")    
        current_project_cfg["output_paths"]["output_result_path"] = seed_output_path
        
        model_filename = f"model_seed_{seed_val}.pkl"
        model_full_path = os.path.join(seed_output_path, model_filename)
        submission_path = os.path.join(seed_output_path, "submission_full_dataset.csv")

        # Train & Evaluate
        evaluation_metrics = utils.evaluate_tuning_results(
            X_train_final, y_train_final, X_test_final, y_test_final,
            build_model=my_model_builder,
            build_model_params={
                "Config": {**current_project_cfg, "task": "regression"},
                "best_params": best_params,
                "evaluation_score_args": args_eval
            },
            generate_submission=True,
            submission_filename=submission_path,
            label_maxs=LABEL_MAXS,
            prediction_columns=["P", "K", "Mg", "pH"],
            model_save_path=model_full_path
        )
        
        # Hyperview usa 'score' invece di 'accuracy'
        inf_time = evaluation_metrics.get('inference_duration_s', 0)
        size_mib = evaluation_metrics.get('model_size_bytes', 0) / (1024 ** 2)

        all_inference_times.append(inf_time)
        all_model_sizes.append(size_mib)

        print(
            f"[INFO-SEED-{seed_val}] Inference Time: {inf_time:.6f}s\n"
            f"Model Size: {size_mib:.2f} MiB\n"
            f"Note: To record the score, submit your results on the Hyperview challenge website."
        )


        # Save Results
        utils.save_run_results(
            project_cfg=current_project_cfg,
            cfg_file_path={'optuna_config_filepath': optuna_cfg_file, 'project_config_filepath': project_cfg_file},
            feature_metrics=feat_metrics,
            optuna_run_metrics={}, 
            evaluation_metrics=evaluation_metrics,
            extra={
                "best_params_source": args.best_params_mode,
                "preprocessing_metrics": aug_metrics
            }
        )
        
    print(f"\n[DONE] All 5 seeds assessed successfully.")

    # =========================================================================
    # 4. FINAL AVERAGE AND STANDARD DEVIATION (AFTER ALL SUBMISSIONS)
    # =========================================================================
    # Note: To get the official final score, all 5 submissions must be made on the Hyperview challenge website.
    #       The following statistics are computed locally for reference only.
    print(f"\n{'='*60}")
    print(f" FINAL STATISTICS FOR {args.cli_model} {args.cli_size} ({len(SEEDS)} SEEDS)")
    print(f"{'='*60}")

    metrics_to_compute = {
        "Inference Time (s)": all_inference_times,
        "Model Size (MiB)": all_model_sizes
    }

    # Compute and display mean and standard deviation
    for name, data in metrics_to_compute.items():
        arr = np.array(data)
        mean_val = np.mean(arr)
        std_val = np.std(arr, ddof=1)
        print(f"{name}: {mean_val:.4f} ± {std_val:.4f}")

    # Optional: remind that the official evaluation score requires Hyperview submission
    print("\nNote: The official Evaluation Score (Error) will be recorded only after submitting all 5 runs on Hyperview.")
