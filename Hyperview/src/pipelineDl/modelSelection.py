# modelSelection.py — HyperView Deep Learning Model Selection (Optuna)
import os
import sys
import argparse
import json
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

    Args:
        root_path (str): The base directory containing train_data/, test_data/, train_gt.csv.

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


# -----------------------------------------------------------------------------
# Optuna Objective
# -----------------------------------------------------------------------------

def objective(trial, data, augmented_data, config) -> float:
    """Optuna objective for DL model selection.

    Signature matches utils.run_optimization:
        objective_fn(trial, data, augmented_data, project_cfg)
    """
    train_dataset = data
    ds = config.get("dataset_name", "hyperview")

    batch_size = config["training_params"]["batch_size_optuna"]

    if config["training_params"]["model_name"]:
        model_name = config["training_params"]["model_name"]
    else:
        model_name = trial.suggest_categorical("model", ["mobilenetv3", "resnet", "shufflenet"])

    if config["training_params"]["pretrained"]:
        pretrained = config["training_params"]["pretrained"]
    else:
        pretrained = trial.suggest_categorical("pretrained", [True, False])

    if config["training_params"]["optimizer"]:
        optimizer_name = config["training_params"]["optimizer"]
    else:
        optimizer_name = trial.suggest_categorical("optimizer", ["Adam", "AdamW", "SGD"])

    if config["training_params"]["scheduler"]:
        scheduler_name = config["training_params"]["scheduler"]
    else:
        scheduler_name = trial.suggest_categorical("scheduler", ["CosineAnnealingLR", "OneCycleLR", "StepLR", "None"])

    learning_rate = trial.suggest_float("learning_rate", 1e-05, 5e-01, log=True)
    weight_decay = trial.suggest_float("weight_decay", 1e-05, 1e-04, log=True)

    train_dataloader, val_dataloader = dl.get_dataloader(train_dataset, ds, config, batch_size)

    model = get_model(model_name, pretrained, ds)

    if optimizer_name == "Adam":
        optimizer = torch.optim.Adam(model.parameters(), lr=learning_rate, weight_decay=weight_decay)
    elif optimizer_name == "AdamW":
        optimizer = torch.optim.AdamW(model.parameters(), lr=learning_rate, weight_decay=weight_decay)
    elif optimizer_name == "SGD":
        momentum = trial.suggest_float("momentum", 0.5, 0.99)
        optimizer = torch.optim.SGD(model.parameters(), lr=learning_rate, weight_decay=weight_decay, momentum=momentum)

    scheduler = None
    if scheduler_name == "StepLR":
        step_size = trial.suggest_int("step_size", 3, 10)
        gamma = trial.suggest_float("gamma", 0.3, 0.7)
        scheduler = torch.optim.lr_scheduler.StepLR(optimizer, step_size=step_size, gamma=gamma)
    elif scheduler_name == "CosineAnnealingLR":
        scheduler = torch.optim.lr_scheduler.CosineAnnealingLR(optimizer, T_max=config["training_params"]["max_epochs_optuna"])

    if ds == "eurosat":
        criterion = nn.CrossEntropyLoss()
    elif ds == "hyperview":
        criterion = nn.MSELoss()

    best_vloss = dp_utils.train(model, criterion, optimizer, scheduler, train_dataloader, val_dataloader, True, config, ds)

    return best_vloss.item() if isinstance(best_vloss, torch.Tensor) else best_vloss


# =============================================================================
# Main — Model Selection
# =============================================================================

if __name__ == "__main__":
    parser = argparse.ArgumentParser(description="HyperView DL Model Selection (Optuna)")
    args = parser.parse_args()

    # --- Path Resolution ---
    current_script_path = os.path.abspath(__file__)
    src_dir = os.path.dirname(current_script_path)       # .../src/pipelineDl
    src_root = os.path.dirname(src_dir)                   # .../src
    project_root = os.path.dirname(src_root)              # .../Hyperview

    config_dir = os.path.join(project_root, "configuration", "jsonConfigurations")
    optuna_cfg_file = os.path.join(config_dir, "optuna_config_ml.json")
    project_cfg_file = os.path.join(config_dir, "project_config.json")

    dataset_name = "hyperview"

    # Load configs
    try:
        optuna_cfg_initial, project_cfg_initial = load_configurations(optuna_cfg_file, project_cfg_file)
    except FileNotFoundError as e:
        print(f"[CRITICAL ERROR] {e}")
        sys.exit(1)

    project_cfg_initial["dataset_name"] = dataset_name

    # --- Fix Relative Paths ---
    if not os.path.isabs(project_cfg_initial.get('cache_directory', 'data/cache')):
        project_cfg_initial['cache_directory'] = os.path.join(project_root, project_cfg_initial.get('cache_directory', 'data/cache'))

    raw_output_path = project_cfg_initial["output_paths"]["output_result_path"]
    if not os.path.isabs(raw_output_path):
        raw_output_path = os.path.join(project_root, raw_output_path)

    # --- Setup ---
    set_random_seed(int(project_cfg_initial['seed']))
    initialize_environment(project_cfg_initial["cache_directory"])

    # --- Dataset ---
    ds_train, ds_test = prepare_hyperview("/root/HyperView/data/")

    # =========================================================================
    #  LOOP — 5 Independent Optuna Runs
    # =========================================================================
    NUM_RUNS = 5

    for run_idx in range(1, NUM_RUNS + 1):
        print(f"\n{'#'*60}")
        print(f" STARTING RUN {run_idx}/{NUM_RUNS}")
        print(f"{'#'*60}\n")

        project_cfg = copy.deepcopy(project_cfg_initial)
        optuna_cfg = copy.deepcopy(optuna_cfg_initial)

        run_output_path = os.path.join(raw_output_path, "ModelSelection", f"test_{run_idx}")
        project_cfg["output_paths"]["output_result_path"] = run_output_path

        # --- Optuna Optimization ---
        print(f"[INFO-RUN-{run_idx}] Starting Optuna optimization...")
        optuna_run_metrics = utils.run_optimization(
            objective, ds_train, None, project_cfg, optuna_cfg
        )
        print(f"[INFO-RUN-{run_idx}] Optuna optimization completed.")

        # --- Save Results ---
        utils.save_run_results(
            project_cfg=project_cfg,
            cfg_file_path={'optuna': optuna_cfg_file, 'project': project_cfg_file},
            feature_metrics=None,
            optuna_run_metrics=optuna_run_metrics,
            evaluation_metrics={},
            extra={}
        )

        print(f"[INFO] RUN {run_idx} COMPLETED SUCCESSFULLY.\n")
