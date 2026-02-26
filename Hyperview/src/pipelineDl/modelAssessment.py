# modelAssessment.py — HyperView Deep Learning Model Assessment (Retrain + Evaluate)
import os
import sys
import argparse
import json
import glob
import time
import platform
import psutil
import random
import copy
import numpy as np
import torch
from datetime import datetime
from torch.utils.data import random_split

from torchvision import transforms, models
from torch import nn

from typing import Any, Dict, Optional

import Utils.utils as utils
import Utils.dp_utils as dp_utils
import Utils.dataloader as dl


debug_mode: bool = False

device = "cuda" if torch.cuda.is_available() else "cpu"
DEVICE = torch.device(device)

# -----------------------------------------------------------------------------
# Utility Functions
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


def prepare_hyperview(root_path: str):
    """
    Load the HyperView dataset (train + test).

    Returns:
        tuple: (train_data, X_test)
            train_data = (X_train, y_train) tensors
            X_test = tensor
    """
    X_train_base, sizes = dl.load_data(root_path + "train_data", tr=dl.resize_and_reduce_transform)
    y_train_base = dl.load_gt(root_path + "train_gt.csv")
    X_test_base, _ = dl.load_data(root_path + "test_data", tr=dl.resize_and_reduce_transform)

    X_train = torch.stack([x.detach() for x in X_train_base])
    y_train = torch.tensor(y_train_base, dtype=torch.float32)

    if debug_mode:
        debug_size = int(len(X_train) * 0.1)
        X_train = X_train[:debug_size]
        y_train = y_train[:debug_size]
        print(f"[INFO] Debug mode enabled. Loaded 10% of the dataset.")

    train_data = (X_train, y_train)

    X_test = torch.stack([x.detach() for x in X_test_base])

    print(f"[INFO] Dataset loaded.")
    print(f"[INFO] Training samples: {len(train_data[0])}")
    print(f"[INFO] Testing samples: {len(X_test)}")

    return train_data, X_test


# -----------------------------------------------------------------------------
# Model Builders
# -----------------------------------------------------------------------------

def get_model(model_name, pretrained, ds):
    if ds == "eurosat":
        num_classes = 10
    elif ds == "hyperview":
        num_classes = 4

    if model_name == "mobilenetv3":
        model = models.mobilenet_v3_large(weights=models.MobileNet_V3_Large_Weights.DEFAULT if pretrained else None)
        in_features = model.classifier[3].in_features
        model.classifier[3] = nn.Linear(in_features, num_classes)
    elif model_name == "resnet":
        model = models.resnet50(weights=models.ResNet50_Weights.DEFAULT if pretrained else None)
        model.fc = nn.Linear(model.fc.in_features, num_classes)
    elif model_name == "shufflenet":
        model = models.shufflenet_v2_x1_0(weights=models.ShuffleNet_V2_X1_0_Weights.DEFAULT if pretrained else None)
        model.fc = nn.Linear(model.fc.in_features, num_classes)

    model.to(DEVICE)
    return model


def get_student_model(model_name, ds):
    if ds == "eurosat":
        num_classes = 10
    elif ds == "hyperview":
        num_classes = 4

    if model_name == "mobilenetv3":
        model = models.mobilenet_v3_small(weights=None)
        in_features = model.classifier[3].in_features
        model.classifier[3] = nn.Linear(in_features, num_classes)
    elif model_name == "resnet":
        model = models.resnet18(weights=None)
        model.fc = nn.Linear(model.fc.in_features, num_classes)
    elif model_name == "shufflenet":
        model = models.shufflenet_v2_x0_5(weights=None)
        model.fc = nn.Linear(model.fc.in_features, num_classes)

    model.to(DEVICE)
    return model


def get_quantization_model(model_name, ds):
    if ds == "eurosat":
        num_classes = 10
    elif ds == "hyperview":
        num_classes = 4

    if model_name == "mobilenetv3":
        model = models.quantization.mobilenet_v3_large()
        in_features = model.classifier[3].in_features
        model.classifier[3] = nn.Linear(in_features, num_classes)
    elif model_name == "resnet":
        model = models.quantization.resnet50()
        model.fc = nn.Linear(model.fc.in_features, num_classes)
    elif model_name == "shufflenet":
        model = models.quantization.shufflenet_v2_x1_0()
        model.fc = nn.Linear(model.fc.in_features, num_classes)

    model.to("cpu")
    return model


def best_model_builder(
    params: Dict[str, Any],
    config: Dict[str, Any],
    opt: str,
    ds: str
):
    """Build the model + optimizer + scheduler from best hyperparameters."""
    batch_size = config["training_params"]["batch_size_optuna"]

    if config["training_params"]["model_name"]:
        model_name = config["training_params"]["model_name"]
    else:
        model_name = params.get("model")

    if config["training_params"]["pretrained"]:
        pretrained = config["training_params"]["pretrained"]
    else:
        pretrained = params.get("pretrained")

    if config["training_params"]["optimizer"]:
        optimizer_name = config["training_params"]["optimizer"]
    else:
        optimizer_name = params.get("optimizer")

    if config["training_params"]["scheduler"]:
        scheduler_name = config["training_params"]["scheduler"]
    else:
        scheduler_name = params.get("scheduler")

    learning_rate = params.get("learning_rate")
    weight_decay = params.get("weight_decay")

    if opt == "quant":
        model = get_quantization_model(model_name, ds)
    elif opt == "dist":
        model = get_student_model(model_name, ds)
    else:
        model = get_model(model_name, pretrained, ds)

    if optimizer_name == "Adam":
        optimizer = torch.optim.Adam(model.parameters(), lr=learning_rate, weight_decay=weight_decay)
    elif optimizer_name == "AdamW":
        optimizer = torch.optim.AdamW(model.parameters(), lr=learning_rate, weight_decay=weight_decay)
    elif optimizer_name == "SGD":
        momentum = params.get("momentum")
        optimizer = torch.optim.SGD(model.parameters(), lr=learning_rate, weight_decay=weight_decay, momentum=momentum)

    scheduler = None
    if scheduler_name == "StepLR":
        step_size = params.get("step_size")
        gamma = params.get("gamma")
        scheduler = torch.optim.lr_scheduler.StepLR(optimizer, step_size=step_size, gamma=gamma)
    elif scheduler_name == "CosineAnnealingLR":
        scheduler = torch.optim.lr_scheduler.CosineAnnealingLR(optimizer, T_max=config["training_params"]["max_epochs_training"])

    if ds == "eurosat":
        criterion = nn.CrossEntropyLoss()
    elif ds == "hyperview":
        criterion = nn.MSELoss()

    return model, optimizer, scheduler, batch_size, criterion


# -----------------------------------------------------------------------------
# Auto-load best params from previous Selection results
# -----------------------------------------------------------------------------

def find_best_params_from_history(base_results_path: str) -> Dict[str, Any]:
    """
    Scans ModelSelection results, finds the run with the LOWEST Optuna best score,
    and returns its hyperparameters.
    """
    search_pattern = os.path.join(base_results_path, "ModelSelection", "*", "output_result.json")
    files = glob.glob(search_pattern)

    if not files:
        print(f"[WARNING] No previous results found in {search_pattern}. Cannot auto-load params.")
        return {}

    best_score = float('inf')
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
                    if current_score < best_score:
                        best_score = current_score
                        best_params = current_params
                        best_file = f_path
                        found_any = True
        except Exception as e:
            print(f"[WARN] Error reading {f_path}: {e}")

    if found_any:
        print(f"[INFO] Auto-selected best params from: {best_file}")
        print(f"[INFO] Best Optuna Score (val loss): {best_score:.6f}")
    else:
        print("[WARN] Could not find valid Optuna best_result data in scanned files.")

    return best_params


# =============================================================================
# Main — Model Assessment
# =============================================================================

if __name__ == "__main__":
    parser = argparse.ArgumentParser(description="HyperView DL Model Assessment (Retrain + Evaluate)")
    parser.add_argument("--best-params-mode", type=str, default="manual", choices=["manual", "auto"],
                        help="How to get hyperparameters: 'manual' (hardcoded) or 'auto' (from best previous result)")
    args = parser.parse_args()

    # --- Path Resolution ---
    current_script_path = os.path.abspath(__file__)
    src_dir = os.path.dirname(current_script_path)       # .../src/pipelineDl
    src_root = os.path.dirname(src_dir)                   # .../src
    project_root = os.path.dirname(src_root)              # .../Hyperview

    config_dir = os.path.join(project_root, "configuration", "jsonConfigurationsRetrain")
    optuna_cfg_file = os.path.join(config_dir, "optuna_config_ml.json")
    project_cfg_file = os.path.join(config_dir, "project_config.json")

    dataset_name = "hyperview"

    # Load configs
    try:
        optuna_cfg, project_cfg = load_configurations(optuna_cfg_file, project_cfg_file)
    except FileNotFoundError as e:
        print(f"[CRITICAL ERROR] {e}")
        sys.exit(1)

    project_cfg["dataset_name"] = dataset_name

    # --- Fix Relative Paths ---
    if not os.path.isabs(project_cfg.get('cache_directory', 'data/cache')):
        project_cfg['cache_directory'] = os.path.join(project_root, project_cfg.get('cache_directory', 'data/cache'))

    raw_output_path = project_cfg["output_paths"]["output_result_path"]
    if not os.path.isabs(raw_output_path):
        raw_output_path = os.path.join(project_root, raw_output_path)

    raw_req_path = project_cfg.get("requirements_txt_path", "requirements.txt")
    if not os.path.isabs(raw_req_path):
        project_cfg["requirements_txt_path"] = os.path.join(project_root, raw_req_path)

    # --- Setup ---
    global_seed = int(project_cfg['seed'])
    print(f"[INFO] Setting GLOBAL seed to {global_seed} for Dataset Split consistency.")
    set_random_seed(global_seed)
    initialize_environment(project_cfg["cache_directory"])

    # --- Dataset ---
    ds_train, ds_test = prepare_hyperview("/root/HyperView/data/")

    # =========================================================================
    # 1. RETRIEVE BEST PARAMETERS
    # =========================================================================
    best_params: Dict[str, Any] = {}

    if args.best_params_mode == "auto":
        print(f"[INFO] Searching for best parameters automatically...")
        base_selection_path = raw_output_path
        best_params = find_best_params_from_history(base_selection_path)
        if not best_params:
            print("[ERROR] Auto-mode failed. Falling back to manual mode defaults.")
            args.best_params_mode = "manual"

    if args.best_params_mode == "manual":
        print(f"[INFO] Using manual best parameters.")
        # -----------------------------------------------------------------
        # Inserire qui i migliori iperparametri trovati durante la selezione.
        # Lasciati liberi: l'utente deve compilarli con i valori desiderati.
        # -----------------------------------------------------------------
        best_params = {
            # "model": "mobilenetv3",        # oppure "resnet", "shufflenet"
            # "pretrained": True,
            # "optimizer": "Adam",            # oppure "AdamW", "SGD"
            # "scheduler": "CosineAnnealingLR",
            # "learning_rate": 1e-3,
            # "weight_decay": 1e-5,
            # "momentum": 0.9,               # solo per SGD
            # "step_size": 5,                 # solo per StepLR
            # "gamma": 0.5,                   # solo per StepLR
        }

    print(f"[INFO] Best Params Loaded: {best_params}")

    # =========================================================================
    # 2. ASSESSMENT LOOP (5 SEEDS)
    # =========================================================================
    assessment_folder_name = "ModelAssessment"

    evaluation_metrics = []
    extra = []

    for i in range(5):
        seed = int.from_bytes(os.urandom(4), "little")
        set_random_seed(seed)

        print(f"\n{'#'*60}")
        print(f" STARTING ASSESSMENT SEED {seed} — Trial ({i+1}/5)")
        print(f"{'#'*60}\n")

        current_project_cfg = copy.deepcopy(project_cfg)
        seed_output_path = os.path.join(raw_output_path, assessment_folder_name, f"test_{i+1}")
        current_project_cfg["output_paths"]["output_result_path"] = seed_output_path
        os.makedirs(seed_output_path, exist_ok=True)

        build_model_params = {
            "Seed": seed,
            "Run": i + 1,
            "Config": current_project_cfg,
            "best_params": best_params
        }

        # --- Evaluate (train + inference) ---
        evaluation_metrics.append(
            dp_utils.deep_learning_evaluate_tuning_results(
                ds_train,
                ds_test,
                build_model=best_model_builder,
                build_model_params=build_model_params,
                ds=dataset_name
            )
        )
        print(f"[INFO] Evaluating Tuning Results completed. Trial ({i+1}/5)")

        # --- Extra optimizations ---
        extra.append({})

        extra[i]["Pruning"] = dp_utils.prune_optimization(
            ds_train,
            ds_test,
            build_model=best_model_builder,
            build_model_params=build_model_params,
            ds=dataset_name
        )
        print(f"[INFO] Pruning Optimization completed. Trial ({i+1}/5)")

        extra[i]["Quantization"] = dp_utils.quantize_optimization(
            ds_train,
            ds_test,
            build_model=best_model_builder,
            build_model_params=build_model_params,
            ds=dataset_name
        )
        print(f"[INFO] Quantization Optimization completed. Trial ({i+1}/5)")

        extra[i]["Distillation"] = dp_utils.distill_optimization(
            ds_train,
            ds_test,
            build_model=best_model_builder,
            build_model_params=build_model_params,
            ds=dataset_name
        )
        print(f"[INFO] Distillation Optimization completed. Trial ({i+1}/5)")

    # --- Save Final Results ---
    utils.save_run_results(
        project_cfg=project_cfg,
        cfg_file_path={'optuna_config_filepath': optuna_cfg_file, 'project_config_filepath': project_cfg_file},
        feature_metrics=None,
        optuna_run_metrics={},
        evaluation_metrics=evaluation_metrics,
        extra=extra
    )

    print(f"\n[DONE] All 5 seeds assessed successfully.")
