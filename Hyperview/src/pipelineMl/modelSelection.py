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

# Add missing imports
from sklearn.ensemble import RandomForestRegressor, GradientBoostingRegressor
from sklearn.multioutput import MultiOutputRegressor
from sklearn.metrics import mean_squared_error
from sklearn.model_selection import train_test_split
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

# -----------------------------------------------------------------------------
# Helper Functions
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
    """
    Load Hyperview hyperspectral data from NPZ files.
    Adapted from CODICE1's load_data function, but without augmentation.

    Args:
        directory (str): Directory to either train or test set
        gt_file_path (str): File path for the ground truth labels (expected CSV file)
        is_train (bool): Binary flag for setting loader for Train (TRUE) or Test (FALSE)

    Returns:
        tuple: For training: (datalist, masklist, labels)
               For testing: (datalist, masklist)
    """
    datalist = []
    masklist = []

    if is_train and gt_file_path:
        labels = load_gt(gt_file_path)

    # Get all NPZ files sorted by numerical order 
    all_files = np.array(
        sorted(
            glob(os.path.join(directory, "*.npz")),
            key=lambda x: int(os.path.basename(x).replace(".npz", "")),
        )
    )

    # Debug mode: limit number of files
    if debug_mode:
        debug_size = int(len(all_files) * 0.01)  # Load only 1% of the data for debugging
        all_files = all_files[:debug_size]
        if is_train and gt_file_path:
            labels = labels[:debug_size]
        print(f"[INFO] Debug mode enabled. Loading {debug_size} samples.")

    # Load each NPZ file
    for idx, file_name in tqdm(enumerate(all_files), 
                              total=len(all_files), 
                              desc=f"Loading {'training' if is_train else 'test'} data"):
        try:
            # Load NPZ file with mask and data 
            with np.load(file_name) as npz:
                mask = npz["mask"]
                data = npz["data"]
                datalist.append(data)
                masklist.append(mask)
        except Exception as e:
            print(f"[WARNING] Error loading {file_name}: {e}")
            continue

    # Return appropriate data based on train/test
    if is_train and gt_file_path:
        return datalist, masklist, labels
    else:
        return datalist, masklist

def apply_hyperview_augmentation(X_train, M_train, y_train, augment_constant: int = 1):
    """
    Apply augmentation to Hyperview training data as a separate process.
    Directly adapted from CODICE1's augmentation logic.

    Args:
        X_train: Training data list
        M_train: Training mask list  
        y_train: Training labels array
        augment_constant (int): Number of augmentation iterations

    Returns:
        tuple: (X_aug_train, M_aug_train, y_aug_train) - augmented data, masks, and labels
    """
    print(f"[INFO] Applying Hyperview augmentation with constant: {augment_constant}")
    
    aug_datalist = []
    aug_masklist = []
    aug_labellist = []
    
    # Apply augmentation for each iteration (same logic as CODICE1)
    for i in range(augment_constant):
        print(f"[INFO] Augmentation iteration {i + 1}/{augment_constant}")
        
        for idx, (data, mask, label) in tqdm(enumerate(zip(X_train, M_train, y_train)), 
                                            total=len(X_train),
                                            desc=f"Augmenting data - iteration {i + 1}"):
            
            # Apply the same augmentation logic as CODICE1
            aug_data, aug_mask, aug_label = apply_hyperview_single_augmentation(data, mask, label)
            
            aug_datalist.append(aug_data)
            aug_masklist.append(aug_mask)
            aug_labellist.append(aug_label)
    
    print(f"[INFO] Augmentation completed. Generated {len(aug_datalist)} augmented samples.")
    return aug_datalist, aug_masklist, np.array(aug_labellist)

def apply_hyperview_single_augmentation(data, mask, label):
    """
    Apply augmentation to a single Hyperview sample.
    Exact same logic as CODICE1's augmentation within the loop.
    """
    flag = True
    ma = np.max(data, keepdims=True)
    sh = data.shape[1:]
    
    # Try 11x11 cropping first (same as CODICE1)
    for attempt in range(10): 
        edge = 11  
        x = np.random.randint(sh[0] + 1 - edge)
        y = np.random.randint(sh[1] + 1 - edge)
        
        # Get crops having meaningful pixels, not zeros
        if np.sum(mask[0, x : (x + edge), y : (y + edge)]) > 120: 
            aug_data = (data[:, x : (x + edge), y : (y + edge)]
                        + np.random.uniform(-0.01, 0.01, (150, edge, edge)) * ma)
            aug_mask = mask[:, x : (x + edge), y : (y + edge)] | np.random.randint(0, 1, (150, edge, edge))
            
            flag = False  # Break the loop when you have a meaningful crop
            break

    # If 11x11 cropping failed, use minimum edge approach (same as CODICE1)
    if flag: 
        max_edge = np.max(sh)
        min_edge = np.min(sh)  # AUGMENT BY SHAPE
        edge = min_edge
        x = np.random.randint(sh[0] + 1 - edge)
        y = np.random.randint(sh[1] + 1 - edge)
        aug_data = (data[:, x : (x + edge), y : (y + edge)]
                    + np.random.uniform(-0.001, 0.001, (150, edge, edge)) * ma)
        aug_mask = mask[:, x : (x + edge), y : (y + edge)] | np.random.randint(0, 1, (150, edge, edge))

    # Apply label augmentation (same as CODICE1)
    aug_label = label + label * np.random.uniform(-0.001, 0.001, 4)
    
    return aug_data, aug_mask, aug_label

def preprocess(data_list, mask_list, is_for_KNN=False): 
    """Extract high-level features from the raw field data.

    Args:
        data_list: Directory to either train or test set
        mask_list: File path for the ground truth labels (expected CVS file)
        is_for_KNN: Binary flag for determining if the features are generated for KNN (TRUE) or Random Forest (FALSE)
    Returns:
        [type]: Tuple of lists composed of (features , field size) pairs for each field, 
                where field size will be used performance analysis.
    """
        
    def _shape_pad(data):
        # This sub-function makes padding to have square fields sizes.
        # Not mandatory but eliminates the risk of calculation error in singular value decomposition,
        # padding by warping also improves the performance slightly.
        max_edge = np.max(image.shape[1:])
        shape = (max_edge, max_edge)
        padded = np.pad(data,((0, 0), (0, (shape[0] - data.shape[1])), (0, (shape[1] - data.shape[2]))),"wrap")
        return padded
    
    filtering = SpectralCurveFiltering()
    w1 = pywt.Wavelet("sym3")
    w2 = pywt.Wavelet("dmey")

    processed_data = []
    average_edge = []

    for idx, (data, mask) in enumerate(
        tqdm(
            zip(data_list, mask_list),
            total=len(data_list),
            position=0,
            leave=True,
            desc="INFO: Preprocessing data ...",
        )
    ):
        data = data / 2210   # max-max=5419 mean-max=2210
        m = 1 - mask.astype(int)
        image = data * m

        average_edge.append((image.shape[1] + image.shape[2]) / 2)
        image = _shape_pad(image)

        s = np.linalg.svd(image, full_matrices=False, compute_uv=False)
        s0 = s[:, 0]  
        s1 = s[:, 1]  
        s2 = s[:, 2] 
        s3 = s[:, 3]  
        s4 = s[:, 4]   
        dXds1 = s0 / (s1 + np.finfo(float).eps)


        data = np.ma.MaskedArray(data, mask)
        arr = filtering(data)

        cA0, cD0 = pywt.dwt(arr, wavelet=w2, mode="constant")
        cAx, cDx = pywt.dwt(cA0[12:92], wavelet=w2, mode="constant")
        cAy, cDy = pywt.dwt(cAx[15:55], wavelet=w2, mode="constant")
        cAz, cDz = pywt.dwt(cAy[15:35], wavelet=w2, mode="constant")
        cAw2 = np.concatenate((cA0[12:92], cAx[15:55], cAy[15:35], cAz[15:25]), -1)
        cDw2 = np.concatenate((cD0[12:92], cDx[15:55], cDy[15:35], cDz[15:25]), -1)

        cA0, cD0 = pywt.dwt(arr, wavelet=w1, mode="constant")
        cAx, cDx = pywt.dwt(cA0[1:-1], wavelet=w1, mode="constant")
        cAy, cDy = pywt.dwt(cAx[1:-1], wavelet=w1, mode="constant")
        cAz, cDz = pywt.dwt(cAy[1:-1], wavelet=w1, mode="constant")
        cAw1 = np.concatenate((cA0, cAx, cAy, cAz), -1)
        cDw1 = np.concatenate((cD0, cDx, cDy, cDz), -1)

        dXdl = np.gradient(arr, axis=0)
        d2Xdl2 = np.gradient(dXdl, axis=0)
        d3Xdl3 = np.gradient(d2Xdl2, axis=0)


        fft = np.fft.fft(arr)
        real = np.real(fft)
        imag = np.imag(fft)
        ffts = np.fft.fft(s0)
        reals = np.real(ffts)
        imags = np.imag(ffts)

        # The best Feature combination for Random Forest based regression
        out_rf = np.concatenate(
            [
                arr,
                dXdl,
                d2Xdl2,
                d3Xdl3,
                dXds1,
                s0,
                s1,
                s2,
                s3,
                s4,
                real,
                imag,
                reals,
                imags,
                cAw1,
                cAw2,
            ],
            -1,
        )
        
        # The best Feature combination for KNN based regression
        out_knn = np.concatenate(
            [
                arr,
                dXdl,
                d2Xdl2,
                d3Xdl3,
                s0,
                s1,
                s2,
                s3,
                s4,
                real,
                imag,

            ],
            -1,
        )


        
      
        if is_for_KNN:
            processed_data.append(out_knn)
        else:
            processed_data.append(out_rf)

    return np.array(processed_data), np.array(average_edge)

class SpectralCurveFiltering: # Default class provided by the challenge organizers
    """
    Create a histogram (a spectral curve) of a 3D cube, using the merge_function
    to aggregate all pixels within one band. The return array will have
    the shape of [CHANNELS_COUNT]
    """

    def __init__(self, merge_function=np.mean):
        self.merge_function = merge_function

    def __call__(self, sample: np.ndarray):
        return self.merge_function(sample, axis=(1, 2))

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

def objective_wrapper(trial, data, augmented_data, project_cfg):
    X_orig, y_orig, args = data
    X_aug, y_aug, aug_source_idx = augmented_data
    return objective(
        trial,
        X_orig, y_orig,
        X_aug, y_aug,
        aug_source_idx,
        project_cfg,
        args
    )

import numpy as np

def train_test_split(indices, test_size=0.2, random_state=None):
    """
    Simple hold-out split function mimicking sklearn.model_selection.train_test_split,
    returning train and validation indices arrays.

    Parameters:
    - indices (array-like): Array of indices to split.
    - test_size (float): Fraction of data to reserve for validation (0 < test_size < 1).
    - random_state (int or None): Seed for reproducibility.

    Returns:
    - train_idx (np.ndarray): Training indices.
    - valid_idx (np.ndarray): Validation indices.
    """
    if not 0 < test_size < 1:
        raise ValueError("test_size should be between 0 and 1.")

    rng = np.random.default_rng(random_state)
    indices = np.array(indices)
    rng.shuffle(indices)
    split_point = int(len(indices) * (1 - test_size))
    train_idx = indices[:split_point]
    valid_idx = indices[split_point:]
    return train_idx, valid_idx

def objective(trial,
              X_orig, y_orig,
              X_aug, y_aug,
              aug_source_idx,
              config, args) -> float:
    
    n_orig = len(X_orig)
    orig_indices = np.arange(n_orig)

    # Hold-out split
    train_idx, valid_idx = train_test_split(
        orig_indices,
        test_size=config['tr_val_split'],
        random_state=config['seed']
    )

    # Build training set: original train + augmentations derived from train_idx
    aug_idx_train = [i for i, src in enumerate(aug_source_idx) if src in train_idx]

    X_train_arr = np.vstack([X_orig[train_idx], X_aug[aug_idx_train]])
    y_train_arr = np.vstack([y_orig[train_idx], y_aug[aug_idx_train]])

    X_valid_arr = X_orig[valid_idx]
    y_valid_arr = y_orig[valid_idx]

    # --- Feature selection ---
    use_spectral = trial.suggest_categorical("use_spectral", [True, False])
    use_grad     = trial.suggest_categorical("use_grad", [True, False])
    use_svd      = trial.suggest_categorical("use_svd", [True, False])
    use_fft      = trial.suggest_categorical("use_fft", [True, False])
    use_wv1      = trial.suggest_categorical("use_wv1", [True, False])
    use_wv2      = trial.suggest_categorical("use_wv2", [True, False])

    idx = 0
    idx_map = {}
    def take(n, name):
        nonlocal idx
        start = idx
        idx += n
        idx_map[name] = list(range(start, idx))

    take(150, 'spectral')
    take(150, 'grad1')
    take(150, 'grad2')
    take(150, 'grad3')
    take(150, 'svd_ratio')
    take(150, 's0')
    take(150, 's1')
    take(150, 's2')
    take(150, 's3')
    take(150, 's4')
    take(150, 'fft_real')
    take(150, 'fft_imag')
    take(150, 'fft_s0_real')
    take(150, 'fft_s0_imag')
    take(150, 'wv1')
    take(150, 'wv2')

    sel = []
    if use_spectral: sel += idx_map['spectral']
    if use_grad:     sel += idx_map['grad1'] + idx_map['grad2'] + idx_map['grad3']
    if use_svd:      sel += idx_map['svd_ratio'] + idx_map['s0'] + idx_map['s1'] + idx_map['s2'] + idx_map['s3'] + idx_map['s4']
    if use_fft:      sel += idx_map['fft_real'] + idx_map['fft_imag'] + idx_map['fft_s0_real'] + idx_map['fft_s0_imag']
    if use_wv1:      sel += idx_map['wv1']
    if use_wv2:      sel += idx_map['wv2']
    if len(sel) < 10:
        return 999.0  # penalize too few features

    X_tr = X_train_arr[:, sel]
    X_va = X_valid_arr[:, sel]

    # --- Model selection ---
    #model_name = trial.suggest_categorical("model", ["RF", "GB"])
    # hardcoded
    # --- Model Definition ---
    model_choice = config.get("cli_model")
    model_size = config.get("cli_size")
    model = None
    if model_choice == "rf":
        if model_size == "big":
            n_est = trial.suggest_int("rf_n_estimators", 100, 300)
            max_d = trial.suggest_int("rf_max_depth", 5, 50)
            model = RandomForestRegressor(n_estimators=n_est, max_depth=max_d, random_state=config['seed'])
        elif model_size == "small":
            n_est = trial.suggest_int("rf_n_estimators", 10, 50)
            max_d = trial.suggest_int("rf_max_depth", 3, 10)
            min_leaf = trial.suggest_int("rf_min_samples_leaf", 5, 20)
            max_feat = trial.suggest_float("rf_max_features", 0.3, 0.8)
            model = RandomForestRegressor(n_estimators=n_est, max_depth=max_d, min_samples_leaf=min_leaf, max_features=max_feat, random_state=config['seed'])
    
    elif model_choice == "xg":
        if model_size == "big":
            n_est = trial.suggest_int("gb_n_estimators", 100, 300)
            lr = trial.suggest_float("gb_learning_rate", 0.01, 0.2)
            md = trial.suggest_int("gb_max_depth", 3, 10)
            base = GradientBoostingRegressor(n_estimators=n_est, learning_rate=lr, max_depth=md, random_state=config['seed'])
        elif model_size == "small":
            n_est = trial.suggest_int("gb_n_estimators", 20, 80)
            lr = trial.suggest_float("gb_learning_rate", 0.1, 0.3)
            md = trial.suggest_int("gb_max_depth", 2, 6)
            subs = trial.suggest_float("gb_subsample", 0.6, 0.9)
            colsb = trial.suggest_float("gb_colsample_bytree", 0.5, 0.9)
            base = GradientBoostingRegressor(n_estimators=n_est, learning_rate=lr, max_depth=md, subsample=subs, max_features=colsb, random_state=config['seed'])
        
        model = MultiOutputRegressor(base)
    
    else:
        sys.exit(1)

    # --- Training & Prediction ---
    model.fit(X_tr, y_train_arr)
    y_pred = model.predict(X_va)

    # Baseline
    baseline = p_utils.BaselineRegressor().fit(X_tr, y_train_arr)
    y_base = baseline.predict(X_va)

    # Scoring
    score = p_utils.evaluation_score(args, y_valid_arr, y_pred, y_base, args.cons)

    # --- Store results in trial ---
    trial.set_user_attr("feat_flags", dict(
        spectral=use_spectral, grad=use_grad, svd=use_svd,
        fft=use_fft, wv1=use_wv1, wv2=use_wv2
    ))
    trial.set_user_attr("model_type", model_choice)
    trial.set_user_attr("model_size", model_size)

    return score  # lower is better

def my_model_builder(
    params: Dict[str, Any],
    config: Dict[str, Any]
) -> utils.ModelProtocol:
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

def extract_features_with_flags(data_list, mask_list, feature_flags=None):
    """Extract high-level features from the raw field data with selectable features.

    Args:
        data_list: Directory to either train or test set
        mask_list: File path for the ground truth labels (expected CVS file)
        feature_flags: Dict with boolean flags for each feature type
                      Available flags: 'arr', 'dXdl', 'd2Xdl2', 'd3Xdl3', 'dXds1', 
                      's0', 's1', 's2', 's3', 's4', 'real', 'imag', 'reals', 
                      'imags', 'cAw1', 'cAw2'
    Returns:
        Tuple of (features, field_sizes) where features are concatenated based on flags
    """
    
    # Default feature flags (all Random Forest features enabled)
    default_flags = {
        'arr': True,
        'dXdl': True,
        'd2Xdl2': True,
        'd3Xdl3': True,
        'dXds1': True,
        's0': True,
        's1': True,
        's2': True,
        's3': True,
        's4': True,
        'real': True,
        'imag': True,
        'reals': True,
        'imags': True,
        'cAw1': True,
        'cAw2': True
    }
    
    if feature_flags is None:
        feature_flags = default_flags
    else:
        # Merge with defaults to ensure all keys exist
        feature_flags = {**default_flags, **feature_flags}
        
    def _shape_pad(data):
        # This sub-function makes padding to have square fields sizes.
        # Not mandatory but eliminates the risk of calculation error in singular value decomposition,
        # padding by warping also improves the performance slightly.
        max_edge = np.max(data.shape[1:])
        shape = (max_edge, max_edge)
        padded = np.pad(data, ((0, 0), (0, (shape[0] - data.shape[1])), (0, (shape[1] - data.shape[2]))), "wrap")
        return padded
    
    filtering = SpectralCurveFiltering()
    w1 = pywt.Wavelet("sym3")
    w2 = pywt.Wavelet("dmey")

    processed_data = []
    average_edge = []

    for idx, (data, mask) in enumerate(
        tqdm(
            zip(data_list, mask_list),
            total=len(data_list),
            position=0,
            leave=True,
            desc="INFO: Preprocessing data with feature flags...",
        )
    ):
        data = data / 2210   # max-max=5419 mean-max=2210
        m = 1 - mask.astype(int)
        image = data * m

        average_edge.append((image.shape[1] + image.shape[2]) / 2)
        image = _shape_pad(image)

        # SVD features
        s = np.linalg.svd(image, full_matrices=False, compute_uv=False)
        s0 = s[:, 0]  
        s1 = s[:, 1]  
        s2 = s[:, 2] 
        s3 = s[:, 3]  
        s4 = s[:, 4]   
        dXds1 = s0 / (s1 + np.finfo(float).eps)

        # Spectral curve filtering
        data = np.ma.MaskedArray(data, mask)
        arr = filtering(data)

        # Wavelet features with dmey
        if feature_flags['cAw2']:
            cA0, cD0 = pywt.dwt(arr, wavelet=w2, mode="constant")
            cAx, cDx = pywt.dwt(cA0[12:92], wavelet=w2, mode="constant")
            cAy, cDy = pywt.dwt(cAx[15:55], wavelet=w2, mode="constant")
            cAz, cDz = pywt.dwt(cAy[15:35], wavelet=w2, mode="constant")
            cAw2 = np.concatenate((cA0[12:92], cAx[15:55], cAy[15:35], cAz[15:25]), -1)

        # Wavelet features with sym3
        if feature_flags['cAw1']:
            cA0, cD0 = pywt.dwt(arr, wavelet=w1, mode="constant")
            cAx, cDx = pywt.dwt(cA0[1:-1], wavelet=w1, mode="constant")
            cAy, cDy = pywt.dwt(cAx[1:-1], wavelet=w1, mode="constant")
            cAz, cDz = pywt.dwt(cAy[1:-1], wavelet=w1, mode="constant")
            cAw1 = np.concatenate((cA0, cAx, cAy, cAz), -1)

        # Gradient features
        dXdl = np.gradient(arr, axis=0)
        d2Xdl2 = np.gradient(dXdl, axis=0)
        d3Xdl3 = np.gradient(d2Xdl2, axis=0)

        # FFT features
        fft = np.fft.fft(arr)
        real = np.real(fft)
        imag = np.imag(fft)
        ffts = np.fft.fft(s0)
        reals = np.real(ffts)
        imags = np.imag(ffts)

        # Build feature vector based on flags
        features_to_concat = []
        
        if feature_flags['arr']:
            features_to_concat.append(arr)
        if feature_flags['dXdl']:
            features_to_concat.append(dXdl)
        if feature_flags['d2Xdl2']:
            features_to_concat.append(d2Xdl2)
        if feature_flags['d3Xdl3']:
            features_to_concat.append(d3Xdl3)
        if feature_flags['dXds1']:
            features_to_concat.append(dXds1)
        if feature_flags['s0']:
            features_to_concat.append(s0)
        if feature_flags['s1']:
            features_to_concat.append(s1)
        if feature_flags['s2']:
            features_to_concat.append(s2)
        if feature_flags['s3']:
            features_to_concat.append(s3)
        if feature_flags['s4']:
            features_to_concat.append(s4)
        if feature_flags['real']:
            features_to_concat.append(real)
        if feature_flags['imag']:
            features_to_concat.append(imag)
        if feature_flags['reals']:
            features_to_concat.append(reals)
        if feature_flags['imags']:
            features_to_concat.append(imags)
        if feature_flags['cAw1']:
            features_to_concat.append(cAw1)
        if feature_flags['cAw2']:
            features_to_concat.append(cAw2)

        # Concatenate selected features
        if features_to_concat:
            out_features = np.concatenate(features_to_concat, -1)
        else:
            # If no features selected, use original arr as fallback
            out_features = arr

            
        #print(f"Sample {idx}:")
        #print(f"  arr shape: {arr.shape}")
        #print(f"  dXdl shape: {dXdl.shape}")
        #print(f"  d2Xdl2 shape: {d2Xdl2.shape}")
        #print(f"  d3Xdl3 shape: {d3Xdl3.shape}")
        #print(f"  dXds1 shape: {dXds1.shape}")
        #print(f"  s0 shape: {s0.shape}")
        #print(f"  s1 shape: {s1.shape}")
        #print(f"  s2 shape: {s2.shape}")
        #print(f"  s3 shape: {s3.shape}")
        #print(f"  s4 shape: {s4.shape}")
        #print(f"  real shape: {real.shape}")
        #print(f"  imag shape: {imag.shape}")
        #print(f"  reals shape: {reals.shape}")
        #print(f"  imags shape: {imags.shape}")
        #print(f"  cAw1 shape: {cAw1.shape}")
        #print(f"  cAw2 shape: {cAw2.shape}")

            
        processed_data.append(out_features)

    return np.array(processed_data), np.array(average_edge)

def extract_hyperview_features_wrapper(dataset_info, feature_flags=None):
    """
    Wrapper function to make Hyperview feature extraction compatible with p_utils.extract_features_split
    
    Args:
        dataset_info: Dictionary containing 'data_list', 'mask_list', and optionally 'labels'
        feature_flags: Feature flags dictionary for feature selection
        
    Returns:
        Tuple of (features_array, labels_array)
    """
    data_list = dataset_info['data_list']
    mask_list = dataset_info['mask_list']
    labels = dataset_info.get('labels', None)
    
    # Extract features using the existing function
    features_array, _ = extract_features_with_flags(data_list, mask_list, feature_flags)
    
    # If labels are provided, return them; otherwise create dummy labels
    if labels is not None:
        labels_array = np.array(labels)
    else:
        # For test data without labels, create dummy labels (they won't be used)
        labels_array = np.zeros(len(features_array))
    
    return features_array, labels_array


if __name__ == "__main__":
    parser = argparse.ArgumentParser(description="Hyperview Model Selection Pipeline")
    parser.add_argument("--model", type=str, default="rf", choices=["rf", "xg"], dest="cli_model", help="Model type")
    parser.add_argument("--size", type=str, default="small", choices=["small", "big"], dest="cli_size", help="Config size")
    parser.add_argument("--debug", action="store_true", help="Enable debug/subsampling mode (loads a small subset).")
    args = parser.parse_args()

    debug_mode = args.debug

    # Path Resolution
    current_script_path = os.path.abspath(__file__)
    src_dir = os.path.dirname(current_script_path)
    src_root = os.path.dirname(src_dir)
    project_root = os.path.dirname(src_root)

    config_dir = os.path.join(project_root, "configuration", "jsonConfigurations")
    optuna_cfg_file = os.path.join(config_dir, "optuna_config_ml.json")
    project_cfg_file = os.path.join(config_dir, "project_config.json")

    try:
        optuna_cfg_initial, project_cfg_initial = load_configurations(optuna_cfg_file, project_cfg_file)
    except FileNotFoundError as e:
        print(f"[CRITICAL ERROR] {e}")
        sys.exit(1)

    project_cfg_initial["cli_model"] = args.cli_model
    project_cfg_initial["cli_size"] = args.cli_size

    # Fix Paths
    raw_ds_path = project_cfg_initial['dataset_root_path']
    if not os.path.isabs(raw_ds_path):
        project_cfg_initial['dataset_root_path'] = os.path.join(project_root, raw_ds_path)
    
    raw_cache_path = project_cfg_initial['cache_directory']
    if not os.path.isabs(raw_cache_path):
        project_cfg_initial['cache_directory'] = os.path.join(project_root, raw_cache_path)

    raw_output_path = project_cfg_initial["output_paths"]["output_result_path"]
    if not os.path.isabs(raw_output_path):
        raw_output_path = os.path.join(project_root, raw_output_path)
    
    base_results_dir = os.path.join(project_root, "result", "ModelSelection")
    if not os.path.exists(base_results_dir):
        os.makedirs(base_results_dir, exist_ok=True)

    # Initial Global Seed
    set_random_seed(int(project_cfg_initial['seed']))
    initialize_environment(project_cfg_initial["cache_directory"])

    # Load Data (Once)
    # Load Hyperview dataset (train_test_split is ignored since dataset is pre-divided)
    X_train, M_train, y_train, X_test, M_test = prepare_dataset(
        project_cfg_initial['dataset_root_path'], 
        project_cfg_initial['train_test_split']  # This will be ignored
    )
    
    # Define augmentation parameters
    augmentation_params = {
        "augment_constant": 1  # AUGMENT_CONSTANT_RF
    }
    
    # Apply augmentation with metrics
    X_aug_train, M_aug_train, y_aug_train, aug_metrics = utils.preprocessing_data(
        X_train=X_train,
        M_train=M_train, 
        y_train=y_train,
        augmentation_fn=apply_hyperview_augmentation,
        augmentation_params_optional=augmentation_params
    )

    # Print metrics
    print("Augmentation Metrics:")
    for key, value in aug_metrics.items():
        if isinstance(value, float):
            print(f"  {key}: {value:.4f}")
        else:
            print(f"  {key}: {value}")
    
    print(f"\nOriginal train data size: {aug_metrics['original_train_samples']}")
    print(f"Augmented data size: {aug_metrics['augmented_samples']}")
    print(f"Augmentation ratio: {aug_metrics['augmentation_ratio']:.2f}x")
    print(f"Augmentation time: {aug_metrics['augmentation_time_s']:.4f} seconds")

    print(f"Train data size: {len(X_train)}")
    print(f"Train aug data size: {len(X_aug_train)}")
    print(f"Test data size: {len(X_test)}")

    # Start feature extraction
    print("Starting feature extraction...")

    feature_flags = {
        'arr': True,
        'dXdl': True,
        'd2Xdl2': True,
        'd3Xdl3': True,
        'dXds1': True,
        's0': True,
        's1': True,
        's2': True,
        's3': True,
        's4': True,
        'real': True,
        'imag': True,
        'reals': True,
        'imags': True,
        'cAw1': True,
        'cAw2': True
    }

     # Prepare dataset info for the wrapper
    train_dataset_info = {
        'data_list': X_train,
        'mask_list': M_train,
        'labels': y_train
    }
    
    test_dataset_info = {
        'data_list': X_test,
        'mask_list': M_test,
        'labels': None  # Test set doesn't have labels
    }

    # Prepare augmented dataset info
    train_aug_dataset_info = {
        'data_list': X_aug_train,
        'mask_list': M_aug_train,
        'labels': y_aug_train
    }

    ds_train_aug = (X_aug_train, M_aug_train, y_aug_train)

    # Use p_utils.extract_features_split with the wrapper
    result = p_utils.extract_features_split(
        train_dataset_info,
        test_dataset_info,
        train_aug_dataset_info,
        extract_hyperview_features_wrapper,
        extractor_params_optional={"feature_flags": feature_flags},
        return_field_sizes=True
    )

    # unpack exactly as returned:
    X_train, y_train, X_test, y_test, X_train_aug, y_train_aug, \
    avg_field_size_train, avg_field_size_test, avg_field_size_train_aug, \
    all_features_metrics = result
   
    print(f"[INFO] Feature extraction completed.")
    print(f"[INFO] Training features shape: {X_train.shape}")
    print(f"[INFO] Test features shape: {X_test.shape}")
    print(f"[INFO] Augmented training features shape: {X_train_aug.shape}")
    print(f"[INFO] Feature extraction metrics: {all_features_metrics}")

    # --- Augmentation source idx mapping ---
    augment_constant = augmentation_params["augment_constant"]
    n_train_samples = len(X_train)

    # I cicli sono invertiti per corrispondere alla logica di augmentation
    aug_source_idx = [
        idx for _ in range(augment_constant) for idx in range(n_train_samples)
    ]

    # --- Build tuples for optimization ---
    data = (X_train, y_train, args_eval)
    augmented_data = (X_train_aug, y_train_aug, aug_source_idx)


    experiment_folder_name = f"{args.cli_model}_{args.cli_size}"
    NUM_RUNS = 5

    # =========================================================================
    # LOOP 5 RUN
    # =========================================================================
    for run_idx in range(1, NUM_RUNS + 1):
        print(f"\n{'#'*60}")
        print(f" STARTING RUN {run_idx}/{NUM_RUNS} | Model: {args.cli_model} | Size: {args.cli_size}")
        print(f"{'#'*60}\n")

        project_cfg = copy.deepcopy(project_cfg_initial)
        optuna_cfg = copy.deepcopy(optuna_cfg_initial)

        current_run_dir_name = f"{args.cli_model}_{args.cli_size}/test{run_idx}" 
        current_run_path = os.path.join(base_results_dir, current_run_dir_name)

        if not os.path.exists(current_run_path):
            os.makedirs(current_run_path, exist_ok=True)

        optuna_run_metrics = utils.run_optimization(
            objective_wrapper,
            data,
            augmented_data,
            project_cfg,
            optuna_cfg
        )
        print("[INFO] Optuna optimization process completed.")

        # Extract only the selected features
        # These metrics correspond to the selected feature categories.
        feature_flags = _get_selected_features(optuna_run_metrics)

        result = p_utils.extract_features_split(
            train_dataset_info,
            test_dataset_info,
            train_aug_dataset_info,
            extract_hyperview_features_wrapper,
            extractor_params_optional={"feature_flags": feature_flags},
            return_field_sizes=True
        )

        X_train, y_train, X_test, y_test, X_train_aug, y_train_aug, \
        avg_field_size_train, avg_field_size_test, avg_field_size_train_aug, \
        feature_metrics = result

        features_selected_metrics = feature_metrics

        # Concatena il training originale e gli augmentati (come fatto nel tuning)
        X_train_full = np.vstack([X_train, X_train_aug])  # shape: (n_train_total, n_features)
        y_train_full = np.vstack([y_train, y_train_aug])  # shape: (n_train_total, n_targets)

        # Seleziona le colonne (features)
        X_train_selected = X_train_full
        y_train_selected = y_train_full
        X_test_selected = X_test
        y_test_selected = y_test

        best_params = optuna_run_metrics["best_result"]["hyperparameters"]

        base_dir = project_cfg['output_paths']['output_result_path']
        submission_filename="submission_full_dataset.csv"
        submission_path = os.path.join(base_dir, submission_filename)

        evaluation_metrics = utils.evaluate_tuning_results(
            X_train_selected,
            y_train_selected,
            X_test_selected,
            y_test_selected,
            build_model=my_model_builder,
            build_model_params = {
                "Config": {**project_cfg, "task": "regression"},
                "best_params": best_params,
                "evaluation_score_args": args
            },

            generate_submission=False,
            submission_filename=submission_path,
            label_maxs=LABEL_MAXS,
            prediction_columns=["P", "K", "Mg", "pH"]  # Adjust column names as needed
        )

        print(f"[INFO] Saving results to: {current_run_path}")

        # ### NEW: Aggiorna il path nel config object ###
        # Questo è fondamentale perché le funzioni utils usano project_cfg per sapere dove salvare
        project_cfg['output_paths']['output_result_path'] = current_run_path

        # Save all relevant results in a single call and generate the final JSON output.
        utils.save_run_results(
            project_cfg=project_cfg,
            cfg_file_path={'optuna_config_filepath': optuna_cfg_file, 'project_config_filepath': project_cfg_file},
            feature_metrics=all_features_metrics,
            optuna_run_metrics=optuna_run_metrics,
            evaluation_metrics=evaluation_metrics,
            extra={
                "preprocessing_metrics": aug_metrics,
                "features_selected_metrics": features_selected_metrics
            }
        )










