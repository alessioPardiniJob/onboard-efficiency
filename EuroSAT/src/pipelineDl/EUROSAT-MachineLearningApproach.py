# main.py
import os
import json
import time
import platform
import psutil
import random
import numpy as np
import torch
from datetime import datetime
from torchvision.datasets import EuroSAT
from torch.utils.data import random_split
import matplotlib.pyplot as plt

# Add missing imports
from sklearn.ensemble import RandomForestClassifier, GradientBoostingClassifier
from skimage import feature, color  # type: ignore
from skimage.feature import canny
from torchvision import transforms, models
from torch import nn

from scipy.stats import skew, kurtosis
from typing import Any, Callable, Dict, Optional, Union

import Utils.utils as utls
import Utils.dp_utils as dp_utls

import Utils.dataloader as dl

debug_mode: bool = False
optimize: bool = False  # Set to False to skip optimization and use default parameters
# -----------------------------------------------------------------------------
# Modular pipeline for dataset loading, feature extraction, optimization, and
# result saving. Each function encapsulates a distinct responsibility for
# clarity, testability, and reuse.
# -----------------------------------------------------------------------------

device = "cuda" if torch.cuda.is_available() else "cpu"
DEVICE = torch.device(device)

# print(DEVICE)

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
    """Load Optuna and project configurations using utils."""
    optuna_cfg = utls.load_json_config(optuna_cfg_path)
    project_cfg = utls.load_json_config(project_cfg_path)
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

    transform = transforms.Compose([transforms.ToTensor(), transforms.Normalize([0.485, 0.456, 0.406],[0.229, 0.224, 0.225])])

    print("[INFO] Downloading the EuroSAT dataset...")
    dataset = EuroSAT(root=root_path, transform=transform, download=True)
    print(f"[INFO] Dataset downloaded successfully. Total samples: {len(dataset)}")


    print("Tipo di dataset:", type(dataset))  # Stampa il tipo


    # If debug_mode is True, load only 10% of the dataset
    if debug_mode:
        debug_size = int(len(dataset) * 0.1)  # 10% of the dataset
        dataset, _ = random_split(dataset, [debug_size, len(dataset) - debug_size])  # Split dataset to 10% for debug
        print(f"[INFO] Debug mode enabled. Loaded 10% of the dataset.")

    # Perform train/test split
    num_train_samples = int(len(dataset) * (1 - split_ratio))
    num_test_samples = len(dataset) - num_train_samples
    train_dataset, test_dataset = random_split(dataset, [num_train_samples, num_test_samples])
    optuna_size = int(len(train_dataset) * 0.3) # 30% of the training dataset for Optuna optimization
    optuna_dataset, _ = random_split(train_dataset, [optuna_size, len(train_dataset) - optuna_size])  

    print(f"[INFO] Dataset split completed.")
    print(f"[INFO] Optuna samples: {len(optuna_dataset)}")
    print(f"[INFO] Training samples: {len(train_dataset)}")
    print(f"[INFO] Testing samples: {len(test_dataset)}")

    return train_dataset, test_dataset, optuna_dataset

def get_model(model_name, pretrained, ds):
    if ds == "eurosat":
        num_classess = 10
    elif ds == "hyperview":
        num_classess = 4
    
    if model_name == "mobilenetv3":
        model = models.mobilenet_v3_large(weights=models.MobileNet_V3_Large_Weights.DEFAULT if pretrained else None)
        in_features = model.classifier[3].in_features
        model.classifier[3] = nn.Linear(in_features, num_classess)
        
    elif model_name == "resnet":
        model = models.resnet50(weights=models.ResNet50_Weights.DEFAULT if pretrained else None)
        model.fc = nn.Linear(model.fc.in_features, num_classess)
        
    elif model_name == "shufflenet":
        model = models.shufflenet_v2_x1_0(weights=models.ShuffleNet_V2_X1_0_Weights.DEFAULT if pretrained else None)
        model.fc = nn.Linear(model.fc.in_features, num_classess)
               
    model.to(DEVICE)
    return model

def get_student_model(model_name, ds):
    if ds == "eurosat":
        num_classess = 10
    elif ds == "hyperview":
        num_classess = 4
    
    if model_name == "mobilenetv3":
        model = models.mobilenet_v3_small(weights=None)
        in_features = model.classifier[3].in_features
        model.classifier[3] = nn.Linear(in_features, num_classess)
        
    elif model_name == "resnet":
        model = models.resnet18(weights=None)
        model.fc = nn.Linear(model.fc.in_features, num_classess)
        
    elif model_name == "shufflenet":
        model = models.shufflenet_v2_x0_5(weights=None)
        model.fc = nn.Linear(model.fc.in_features, num_classess)
               
    model.to(DEVICE)
    return model

def get_quantization_model(model_name, ds):
    if ds == "eurosat":
        num_classess = 10
    elif ds == "hyperview":
        num_classess = 4
    
    if model_name == "mobilenetv3":
        model = models.quantization.mobilenet_v3_large()
        in_features = model.classifier[3].in_features
        model.classifier[3] = nn.Linear(in_features, num_classess)
        
    elif model_name == "resnet":
        model = models.quantization.resnet50()
        model.fc = nn.Linear(model.fc.in_features, num_classess)
        
    elif model_name == "shufflenet":
        model = models.quantization.shufflenet_v2_x1_0()
        model.fc = nn.Linear(model.fc.in_features, num_classess)
               
    model.to("cpu")
    return model

def objective(trial, train_dataset, config, ds) -> float:
    batch_size = config["training_params"]["batch_size_optuna"]    
    if config["training_params"]["model_name"]:
        model_name = config["training_params"]["model_name"]
    else:
        model_name = trial.suggest_categorical("model", ["mobilenetv3", "resnet", "shufflenet"])

    if config["training_params"]["pretrained"]:
        pretrained = config["training_params"]["pretrained"]
    else:
        pretrained = trial.suggest_categorical("model", [True, False])

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
    
    best_vloss = dp_utls.train(model, criterion, optimizer, scheduler, train_dataloader, val_dataloader, True, config, ds)
    
    return best_vloss.item() if isinstance(best_vloss, torch.Tensor) else best_vloss

def best_model_builder(
    params: Dict[str, Any],
    config: Dict[str, Any],
    opt: str,
    ds: str
):
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

if __name__ == "__main__":
    optuna_cfg_file = "configuration/jsonConfigurations/optuna_config.json"
    project_cfg_file = "configuration/jsonConfigurations/project_config.json"
    dataset_name = "eurosat"  # Change to "hyperview" if needed

    optuna_cfg, project_cfg = load_configurations(optuna_cfg_file, project_cfg_file)

    set_random_seed(int(project_cfg['seed']))

    initialize_environment(project_cfg["cache_directory"])

    ds_train, ds_test, ds_optuna = prepare_dataset(project_cfg['dataset_root_path'], 
                                        project_cfg['train_test_split'])

    filename = os.path.join(project_cfg["output_paths"]["output_result_path"], project_cfg["output_paths"]["best_params_json"])
    
    if optimize == True:
        optuna_run_metrics = utls.run_optimization(
            objective, 
            ds_optuna,
            project_cfg,
            optuna_cfg,
            dataset_name
        )
        print("[INFO] Optuna optimization process completed.")
        with open(filename, "w") as f:
            json.dump(optuna_run_metrics, f, indent=4)
    else:
        optuna_run_metrics = {}
        with open(filename, "r") as f:
            optuna_run_metrics = json.load(f)
        print("[INFO] Optuna optimization skipped. Using default parameters.")

    ############## End of personal section – Alessio Pardini ##############

    evaluation_metrics = []
    extra = []
    
    
    seeds = [2842370751, 3432591263, 2754121295, 4271628354, 358403260]
    
    for i in range(5):
        seed = seeds[i]
        # seed = int.from_bytes(os.urandom(4), "little")
        set_random_seed(seed)
        # evaluation_metrics.append(
        #     dp_utls.deep_learning_evaluate_tuning_results(
        #         ds_train,
        #         ds_test,
        #         build_model=best_model_builder,
        #         build_model_params={
        #             "Seed": seed,
        #             "Run": i+1,
        #             "Config": project_cfg,
        #             "best_params": optuna_run_metrics.get("best_result", {}).get("hyperparameters", {}) 
        #         },
        #         ds=dataset_name
        #     )
        # )
        # print(f"[INFO] Evaluating Tuning Results process completed. Trial({i+1}/5)")

        extra.append({})        
        # extra[i]["Pruning"] = dp_utls.prune_optimization(
        #     ds_train,
        #     ds_test,
        #     build_model=best_model_builder,
        #     build_model_params={
        #         "Seed": seed,
        #         "Run": i+1,
        #         "Config": project_cfg,
        #         "best_params": optuna_run_metrics.get("best_result", {}).get("hyperparameters", {}) 
        #     },
        #     ds=dataset_name
        # )
        # print(f"[INFO] Pruning Optimization process completed. Trial({i+1}/5)")
        
        extra[i]["Quantization"] = dp_utls.quantize_optimization(
            ds_train,
            ds_test,
            build_model=best_model_builder,
            build_model_params={
                "Seed": seed,
                "Run": i+1,
                "Config": project_cfg,
                "best_params": optuna_run_metrics.get("best_result", {}).get("hyperparameters", {}) 
            },
            ds=dataset_name
        )
        print(f"[INFO] Quantization Optimization process completed. Trial({i+1}/5)")
        
        # extra[i]["Distillation"] = dp_utls.distill_optimization(
        #     ds_train,
        #     ds_test,
        #     build_model=best_model_builder,
        #     build_model_params={
        #         "Seed": seed,
        #         "Run": i+1,
        #         "Config": project_cfg,
        #         "best_params": optuna_run_metrics.get("best_result", {}).get("hyperparameters", {}) 
        #     },
        #     ds=dataset_name
        # )
        # print(f"[INFO] Distillation Optimization process completed. Trial({i+1}/5)")
     
    utls.save_run_results(
        project_cfg=project_cfg,
        cfg_file_path={'optuna_config_filepath': optuna_cfg_file, 'project_config_filepath': project_cfg_file},
        feature_metrics=None,
        optuna_run_metrics=optuna_run_metrics,
        evaluation_metrics=evaluation_metrics,
        extra=extra
    )