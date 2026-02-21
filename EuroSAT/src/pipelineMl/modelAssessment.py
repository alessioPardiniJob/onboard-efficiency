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
import ssl
import urllib.request
import glob

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

# --- SSL FIX PER REVIEWERS ---
try:
    _create_unverified_https_context = ssl._create_unverified_context
except AttributeError:
    pass
else:
    ssl._create_default_https_context = _create_unverified_https_context

debug_mode: bool = False

# -----------------------------------------------------------------------------
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

def prepare_dataset(root_path: str, split_ratio: float):
    print(f"[INFO] Checking dataset directory at: '{root_path}'")
    if not os.path.exists(root_path):
        os.makedirs(root_path)
    dataset = EuroSAT(root=root_path, transform=None, download=True)
    
    if debug_mode:
        debug_size = int(len(dataset) * 0.10)
        dataset, _ = random_split(dataset, [debug_size, len(dataset) - debug_size])
        print(f"[INFO] Debug mode enabled. Loaded 10% of the dataset.")

    num_train_samples = int(len(dataset) * (1 - split_ratio))
    num_test_samples = len(dataset) - num_train_samples
    train_dataset, test_dataset = random_split(dataset, [num_train_samples, num_test_samples])
    return train_dataset, test_dataset

def get_indices_from_flags(feature_flags: dict) -> list:
    idx_color_stats   = list(range(0, 6))    if feature_flags.get("use_color_stats_feature")   else []
    idx_color_hist    = list(range(6, 54))   if feature_flags.get("use_color_hist_feature")    else []
    idx_glcm          = list(range(54, 58))  if feature_flags.get("use_glcm_feature")          else []
    idx_lbp           = list(range(58, 68))  if feature_flags.get("use_lbp_feature")           else []
    idx_edge          = [68]                 if feature_flags.get("use_edge_density_feature")  else []
    idx_higher_order  = list(range(69, 75))  if feature_flags.get("use_higher_order_feature")  else []
    return idx_color_stats + idx_color_hist + idx_glcm + idx_lbp + idx_edge + idx_higher_order

def extract_classic_features(dataset_subset, feature_flags: dict) -> (np.ndarray, np.ndarray):
    features_list: list = []
    labels_list: list = []
    for img, label in dataset_subset:
        feat: np.ndarray = compute_features(img, feature_flags)
        features_list.append(feat)
        labels_list.append(label)
    return np.array(features_list), np.array(labels_list)

def compute_features(pil_img, feature_flags: dict) -> np.ndarray:
    img_np: np.ndarray = np.array(pil_img)
    if img_np.ndim == 2:
        img_np = np.stack((img_np,) * 3, axis=-1)

    feature_vector_parts = []
    gray_img = None
    gray_img_uint8 = None

    if feature_flags.get('use_color_stats_feature', False):
        feature_vector_parts.extend([np.mean(img_np, axis=(0, 1)), np.std(img_np, axis=(0, 1))])

    if feature_flags.get('use_color_hist_feature', False):
        hist_features_list = []
        for c in range(3):
            hist, _ = np.histogram(img_np[..., c], bins=16, range=(0, 255), density=True)
            hist_features_list.extend(hist)
        feature_vector_parts.append(np.array(hist_features_list))

    if feature_flags.get('use_glcm_feature', False):
        if gray_img is None: gray_img = color.rgb2gray(img_np)
        quantized_img = np.floor(gray_img * 8).astype(np.uint8)
        glcm = feature.graycomatrix(quantized_img, distances=[1], angles=[0], symmetric=True, normed=True)
        props = [feature.graycoprops(glcm, prop=p)[0, 0] for p in ['contrast', 'correlation', 'energy', 'homogeneity']]
        feature_vector_parts.append(np.array(props))

    if feature_flags.get('use_lbp_feature', False):
        if gray_img is None: gray_img = color.rgb2gray(img_np)
        if gray_img_uint8 is None: gray_img_uint8 = (gray_img * 255).astype(np.uint8)
        lbp = feature.local_binary_pattern(gray_img_uint8, P=8, R=1, method='uniform')
        lbp_hist = np.histogram(lbp, bins=10, range=(0, lbp.max() + 1), density=True)[0]
        feature_vector_parts.append(lbp_hist)

    if feature_flags.get('use_edge_density_feature', False):
        if gray_img is None: gray_img = color.rgb2gray(img_np)
        edges = canny(gray_img)
        feature_vector_parts.append(np.array([float(np.sum(edges) / edges.size)]))

    if feature_flags.get('use_higher_order_feature', False):
        skews, kurtoses = [], []
        for c in range(img_np.shape[2]):
            d = img_np[:, :, c].ravel()
            skews.append(float(skew(d)))
            kurtoses.append(float(kurtosis(d)))
        feature_vector_parts.extend([np.array(skews), np.array(kurtoses)])

    return np.concatenate(feature_vector_parts)

def my_model_builder(params: Dict[str, Any], config: Dict[str, Any]) -> utils.ModelProtocol:
    """
    Builds the model using specific parameters found during selection.
    Uses 'cli_model' from config to decide class, but parameters come from 'params' dict.
    """
    model_type = config.get("cli_model", "rf")
    # Nota: Il seed qui verrà passato dinamicamente nel loop principale, sovrascrivendo config['seed'] se necessario
    seed = config.get('seed', 42)

    if model_type == 'rf':
        return RandomForestClassifier(
            n_estimators=params.get('rf_n_estimators', 100),
            max_depth=params.get('rf_max_depth', None),
            min_samples_leaf=params.get('rf_min_samples_leaf', 1),
            max_features=params.get('rf_max_features', "sqrt"),
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

# -----------------------------------------------------------------------------
# Function to Find Best Params Automatically
# -----------------------------------------------------------------------------
def find_best_params_from_history(base_results_path: str, model_type: str, model_size: str) -> Dict[str, Any]:
    """
    Scans the results directory, inspects 'Optuna' -> 'best_result' -> 'score',
    finds the run with the LOWEST score (minimize error), and returns its hyperparameters.
    """
    # Costruisce il path di ricerca: es. .../rf_small/*/output_result.json
    search_pattern = os.path.join(base_results_path, "ModelSelection", f"{model_type}_{model_size}", "*", "output_result.json")
    files = glob.glob(search_pattern)
    
    if not files:
        print(f"[WARNING] No previous results found in {search_pattern}. Cannot auto-load params.")
        return {}

    # Inizializziamo a infinito perché stiamo cercando il MINIMO score (errore)
    best_score = float('inf') 
    best_params = {}
    best_file = ""
    found_any = False

    for f_path in files:
        try:
            with open(f_path, 'r') as f:
                data = json.load(f)
                
                # Navigazione sicura nel JSON secondo la struttura fornita
                optuna_section = data.get("Optuna", {})
                best_result = optuna_section.get("best_result", {})
                
                # Estrazione dati
                current_score = best_result.get("score")
                current_params = best_result.get("hyperparameters")

                # Verifica validità dati
                if current_score is not None and current_params:
                    # LOGICA: Poiché lo score è (1 - accuracy), cerchiamo il valore PIÙ BASSO.
                    if current_score < best_score:
                        best_score = current_score
                        best_params = current_params
                        best_file = f_path
                        found_any = True
                        
        except Exception as e:
            print(f"[WARN] Error reading {f_path}: {e}")

    if found_any:
        print(f"[INFO] Auto-selected best params from: {best_file}")
        print(f"[INFO] Best Optuna Score (Error Rate): {best_score:.4f}")
    else:
        print("[WARN] Could not find valid Optuna best_result data in scanned files.")

    return best_params
# -----------------------------------------------------------------------------
# Main Execution
# -----------------------------------------------------------------------------
if __name__ == "__main__":
    parser = argparse.ArgumentParser(description="EuroSAT Model Assessment (Retraining)")
    parser.add_argument("--model", type=str, default="rf", choices=["rf", "xg"], dest="cli_model", help="Model type")
    parser.add_argument("--size", type=str, default="small", choices=["small", "big"], dest="cli_size", help="Config size")
    parser.add_argument("--best-params-mode", type=str, default="manual", choices=["manual", "auto"], 
                        help="How to get hyperparameters: 'manual' (hardcoded) or 'auto' (from best previous result)")
    parser.add_argument("--debug", action="store_true", help="Enable debug/subsampling mode (loads a subset).")
    args = parser.parse_args()

    debug_mode = args.debug

    # --- Path Resolution ---
    current_script_path = os.path.abspath(__file__)
    src_dir = os.path.dirname(current_script_path)    # .../src/pipelineMl
    src_root = os.path.dirname(src_dir)              # .../src
    project_root = os.path.dirname(src_root)         # .../EuroSAT (ROOT DEL PROGETTO)
    
    config_dir = os.path.join(project_root, "configuration", "jsonConfigurationsRetrain")
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

    # Setup
    global_seed = int(project_cfg['seed'])
    print(f"[INFO] Setting GLOBAL seed to {global_seed} for Dataset Split consistency.")
    set_random_seed(global_seed)    
    initialize_environment(project_cfg["cache_directory"])

    # Load Dataset
    ds_train, ds_test = prepare_dataset(project_cfg['dataset_root_path'], project_cfg['train_test_split'])

    # =========================================================================
    # 1. RETRIEVE BEST PARAMETERS
    # =========================================================================
    best_params = {}

    if args.best_params_mode == "auto":
        print(f"[INFO] Searching for best parameters automatically...")
        # Usa la cartella 'ModelSelection' (definita nel main precedente) come base di ricerca
        base_selection_path = os.path.join(raw_output_path, "..", "ModelSelection") # Adjust based on your folder structure if needed
        # Fallback: cerca nella root output se ModelSelection non esiste relativo
        if not os.path.exists(base_selection_path):
             base_selection_path = raw_output_path
        
        best_params = find_best_params_from_history(base_selection_path, args.cli_model, args.cli_size)
        if not best_params:
            print("[ERROR] Auto-mode failed. Falling back to manual mode defaults.")
            args.best_params_mode = "manual"

    if args.best_params_mode == "manual":

        print(f"[INFO] Using manual best parameters.")
        if args.cli_model == 'rf' and args.cli_size == 'small':
            best_params = {
                "use_color_stats_feature": False,
                "use_color_hist_feature": True,
                "use_glcm_feature": False,
                "use_lbp_feature": True,
                "use_edge_density_feature": True,
                "use_higher_order_feature": True,
                "rf_n_estimators": 17,
                "rf_max_depth": 10,
                "rf_min_samples_leaf": 5,
                "rf_max_features": 0.3362682243263383

            }
        elif args.cli_model == 'xg' and args.cli_size == 'small':
            best_params = {
                "use_color_stats_feature": True,
                "use_color_hist_feature": True,
                "use_glcm_feature": False,
                "use_lbp_feature": True,
                "use_edge_density_feature": True,
                "use_higher_order_feature": False,
                "gb_n_estimators": 75,
                "gb_learning_rate": 0.26663782445491496,
                "gb_max_depth": 6,
                "gb_subsample": 0.8138347307375998,
                "gb_colsample_bytree": 0.6141181332252498
            }
        elif args.cli_model == 'rf' and args.cli_size == 'big':
            best_params = {
                "use_color_stats_feature": True,
                "use_color_hist_feature": True,
                "use_glcm_feature": False,
                "use_lbp_feature": True,
                "use_edge_density_feature": True,
                "use_higher_order_feature": False,
                "rf_n_estimators": 266,
                "rf_max_depth": 38
            }
        elif args.cli_model == 'xg' and args.cli_size == 'big':
            best_params = {
                "use_color_stats_feature": True,
                "use_color_hist_feature": True,
                "use_glcm_feature": True,
                "use_lbp_feature": True,
                "use_edge_density_feature": True,
                "use_higher_order_feature": False,
                "gb_n_estimators": 270,
                "gb_learning_rate": 0.1328241674287609,
                "gb_max_depth": 7
            }
        else:
            print(f"[WARNING] No manual params defined for model {args.cli_model} and size {args.cli_size}. Using empty params.")
            best_params = {}
            system.exit(1)

    print(f"[INFO] Best Params Loaded: {best_params}")

    # =========================================================================
    # 2. FEATURE EXTRACTION (ONE TIME) BASED ON BEST PARAMS
    # =========================================================================
    print(f"\n{'='*60}")
    print(f" [RETRAIN] Extracting SPECIFIC features for best configuration...")
    print(f"{'='*60}\n")
    
    # Estraiamo solo le feature richieste dai best_params
    # Creiamo un dizionario di flag basato sui parametri
    feature_flags = {k: v for k, v in best_params.items() if k.endswith("_feature")}

    X_train_final, y_train_final, X_test_final, y_test_final, _, _, feat_metrics = p_utils.extract_features_split(
        ds_train, ds_test, None, extract_classic_features, 
        extractor_params_optional={"feature_flags": feature_flags}
    )
    print(f" [RETRAIN] Features ready. Shape: {X_train_final.shape}")

    # =========================================================================
    # 3. ASSESSMENT LOOP (5 SEEDS)
    # =========================================================================
    assessment_folder_name = f"ModelAssessment/{args.cli_model}_{args.cli_size}"
    SEEDS = [1, 2, 3, 4, 5]

    all_accuracies = []
    all_inference_times = []
    all_model_sizes = []

    for seed_val in SEEDS:
        print(f"\n{'#'*60}")
        print(f" STARTING ASSESSMENT SEED {seed_val} | Model: {args.cli_model}")
        print(f"{'#'*60}\n")
        
        # Aggiorna il seed nel config per il model builder
        current_project_cfg = copy.deepcopy(project_cfg)
        current_project_cfg['seed'] = seed_val
        set_random_seed(seed_val)

        # Output path: .../result/ModelAssessment/rf_small/model_seed_1.pkl
        seed_output_path = os.path.join(raw_output_path, assessment_folder_name, f"test_{seed_val}")    
        current_project_cfg["output_paths"]["output_result_path"] = seed_output_path
        
        # Model path
        model_filename = f"model_seed_{seed_val}.pkl"
        model_full_path = os.path.join(seed_output_path, model_filename)

        # Train & Evaluate
        evaluation_metrics = utils.evaluate_tuning_results(
            X_train_final, y_train_final, X_test_final, y_test_final,
            build_model=my_model_builder,
            build_model_params={
                "Config": current_project_cfg,
                "best_params": best_params
            },
            model_save_path=model_full_path
        )
        
        acc = evaluation_metrics.get('accuracy', 0)
        inf_time = evaluation_metrics.get('inference_duration_s', 0)
        size_mib = evaluation_metrics.get('model_size_bytes', 0) / (1024 ** 2)

        all_accuracies.append(acc)
        all_inference_times.append(inf_time)
        all_model_sizes.append(size_mib)

        print(f"[INFO-SEED-{seed_val}] Accuracy: {acc:.4f} \n"
              f"---------------------- Inference Time: {inf_time:.6f}s \n"
              f"---------------------- Model Size: {size_mib:.2f} MiB")

        # Save Results
        utils.save_run_results(
            project_cfg=current_project_cfg,
            cfg_file_path={'optuna': optuna_cfg_file, 'project': project_cfg_file},
            feature_metrics=feat_metrics,
            optuna_run_metrics={}, # Empty because no optimization here
            evaluation_metrics=evaluation_metrics,
            extra={"best_params_source": args.best_params_mode}
        )
        
    print(f"\n[DONE] All 5 seeds assessed successfully.")

# =========================================================================
    # 4. CALCOLO MEDIA E STD FINALE (FUORI DAL LOOP)
    # =============================================================
    print(f"\n{'='*60}")
    print(f" FINAL STATISTICS FOR {args.cli_model} {args.cli_size} ({len(SEEDS)} SEEDS)")
    print(f"{'='*60}")
    
    metrics_to_compute = {
        "Accuracy": all_accuracies,
        "Inference Time (s)": all_inference_times,
        "Model Size (MiB)": all_model_sizes
    }

    for name, data in metrics_to_compute.items():
        arr = np.array(data)
        mean_val = np.mean(arr)
        std_val = np.std(arr, ddof=1) # ddof=1 per la deviazione standard campionaria
        print(f"{name}: {mean_val:.4f} ± {std_val:.4f}")

    print(f"{'='*60}\n")
