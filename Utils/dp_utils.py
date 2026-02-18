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
import torch.nn.utils.prune as prune
import torch.nn as nn

from optuna.importance import get_param_importances as _get_param_importances
from optuna.samplers import BaseSampler, TPESampler
from optuna.study import Study
from optuna.trial import FrozenTrial

import random
import torch
from torch.utils.data import DataLoader, TensorDataset
from Utils.dataloader import CustomDataset, train_transform, get_dataloader

import Utils.dataloader as dl

import math
import copy

device = "cuda" if torch.cuda.is_available() else "cpu"
DEVICE = torch.device(device)


 
def train_one_epoch(model, optimizer, criterion, train_dataloader, ds):
    running_loss = 0.
    train_loss = 0.
    train_acc = 0.
    num_correct = 0
    num_samples = 0

    for _, data in enumerate(train_dataloader):
        inputs, labels = data
        optimizer.zero_grad()
        outputs = model(inputs.to(device, non_blocking=True))
        loss = criterion(outputs, labels.to(device, non_blocking=True))
        loss.backward()
        optimizer.step()
        running_loss += loss.item()
        
        if ds == "eurosat":
            _, predictions = outputs.max(dim=-1)
            num_correct += (predictions == labels.to(device, non_blocking=True)).sum()
            num_samples += predictions.size(0)
            
    return running_loss / len(train_dataloader)


def train(model, criterion, optimizer, scheduler, train_dataloader, val_dataloader, optuna, project_cfg, ds):
    best_tloss = 1_000_000.
    best_vloss = 1_000_000.
    patience = 0
    max_patience = project_cfg["training_params"]["max_patience"]
    best_wts = None

    if optuna:
        max_epochs = project_cfg["training_params"]["max_epochs_optuna"]
    else:
        max_epochs = project_cfg["training_params"]["max_epochs_training"]
        
    for epoch in range(max_epochs):
        model.train(True)
        avg_tloss = train_one_epoch(model, optimizer, criterion, train_dataloader, ds)
        
        if not val_dataloader:
            if math.isnan(avg_tloss) or math.isinf(avg_tloss):
                break
            if avg_tloss < best_tloss:
                best_tloss = avg_tloss
                best_wts = copy.deepcopy(model.state_dict())
                # torch.save(model.state_dict(), project_cfg["output_paths"]["output_result_path"] + "/best_final_model")
            if scheduler:
                scheduler.step()
            continue
        
        running_vloss = 0.0
        val_acc = 0.0
        num_vcorrect = 0
        num_vsamples = 0

        model.eval()
        with torch.no_grad():
            for i, vdata in enumerate(val_dataloader):
                vinputs, vlabels = vdata
                voutputs = model(vinputs.to(device, non_blocking=True))
                vloss = criterion(voutputs, vlabels.to(device, non_blocking=True))
                running_vloss += vloss
                if ds == "eurosat":
                    _, vpredictions = voutputs.max(dim=-1)
                    num_vcorrect += (vpredictions == vlabels.to(device, non_blocking=True)).sum()
                    num_vsamples += vpredictions.size(0)

        avg_vloss = running_vloss / len(val_dataloader)
        
        if math.isnan(avg_vloss) or math.isinf(avg_vloss):
            break

        if avg_vloss < best_vloss:
            best_vloss = avg_vloss
            patience = 0
        else:
            if patience >= max_patience and not optuna:
                break
            patience+=1

        if scheduler:
            scheduler.step()

    if not optuna:
        return best_wts
    return best_vloss

def evaluate(model, test_dataloader, device, get_output):
    preds, labels = [], []
    with torch.no_grad():
        for _, (tinputs, tlabels) in enumerate(test_dataloader):
            toutputs = model(tinputs.to(device))
            if get_output:
                preds.append(toutputs.cpu())
                labels.append(tlabels.cpu())
    
    if not get_output:
        return
    
    y_pred = torch.cat(preds)
    y_test = torch.cat(labels)

    y_pred= torch.argmax(y_pred, dim=1)

    return y_pred.numpy(), y_test.numpy()

def predict(model, test_dataloader, std, mean, path):
    model.eval()
    predictions = []
    with torch.no_grad():
        for _, data in enumerate(test_dataloader):
            inputs = data[0]
            # Effettua le previsioni utilizzando il modello
            outputs = model(inputs.to(device))
            outputs = outputs.cpu() * std + mean
            # Aggiungi le predizioni alla lista delle predizioni
            predictions.append(outputs.cpu().numpy())

    predictions_array = np.concatenate(predictions)
    submission_df = pd.DataFrame(data=predictions_array, columns=["P", "K", "Mg", "pH"])
    submission_df.to_csv(path, index_label="sample_index")
    

def deep_learning_evaluate_tuning_results(
    ds_train,
    ds_test,
    build_model,
    build_model_params: Dict[str, Any],
    ds: str,
) -> Dict[str, Any]:
    """
    Generate a comprehensive report following hyperparameter optimization,
    measuring both RSS via psutil and peak allocations via tracemalloc.
    For Deep Learning models
    """
    
    # Extract config and best_params
    config = build_model_params.get("Config")
    best_params = build_model_params.get("best_params")
    run = build_model_params.get("Run")
    seed = build_model_params.get("Seed")

    std, mean = (0,0)    
    if ds == "hyperview":
        X_train, y_train =  ds_train
        mean = y_train.mean(axis=0)
        std = y_train.std(axis=0)
        ds_train = CustomDataset(X_train, y_train, train_transform, mean, std)
        ds_test = TensorDataset(ds_test)
        
    # Instantiate model
    model, optimizer, scheduler, batch_size, criterion = build_model(best_params, config, None, ds)
    best_model, _, _, _, _ = build_model(best_params, config, None, ds)
    proc = psutil.Process()
    
    # --- TRAINING PHASE ---
    mem_rss_before_train = proc.memory_info().rss / (1024 ** 2)
    tracemalloc.start()    
    train_dataloader = DataLoader(ds_train, batch_size=batch_size, shuffle=True)
    t0 = time.perf_counter()
    best_wts = train(model, criterion, optimizer, scheduler, train_dataloader, None, False, config, ds)
    train_duration = time.perf_counter() - t0
    best_model.load_state_dict(best_wts)
    torch.save(best_model.state_dict(), config["output_paths"]["output_result_path"] + f"/best_final_model_{run}")
    _, peak_trace_train = tracemalloc.get_traced_memory()
    tracemalloc.reset_peak()
    mem_rss_after_train = proc.memory_info().rss / (1024 ** 2)
    rss_delta_train_mb = mem_rss_after_train - mem_rss_before_train
    
     # --- INFERENCE PHASE ---
    mem_rss_before_infer = proc.memory_info().rss / (1024 ** 2)
    tracemalloc.start()
    test_dataloader = DataLoader(ds_test, batch_size=batch_size, shuffle=False)
    t1 = time.perf_counter()
    if ds == "eurosat":
        y_pred, y_test = evaluate(best_model, test_dataloader, DEVICE, True)
    elif ds == "hyperview":
        predict(best_model, test_dataloader, std, mean, config["output_paths"]["output_result_path"] + f"/submission_{run}.csv")
    inference_duration = time.perf_counter() - t1
    _, peak_trace_infer = tracemalloc.get_traced_memory()
    tracemalloc.stop()
    mem_rss_after_infer = proc.memory_info().rss / (1024 ** 2)
    rss_delta_infer_mb = mem_rss_after_infer - mem_rss_before_infer

    # --- MODEL SIZE ---
    model_bytes = pickle.dumps(best_model)
    model_size_bytes = sys.getsizeof(model_bytes)

    # --- COMPILE RESULTS ---
    results: Dict[str, Any] = {
        'seed': seed,
        'train_duration_s': train_duration,
        'inference_duration_s': inference_duration,
        'model_size_bytes': model_size_bytes,
        'psutil_rss_before_train_mb': mem_rss_before_train,
        'psutil_rss_after_train_mb': mem_rss_after_train,
        'psutil_rss_delta_train_mb': rss_delta_train_mb,
        'psutil_rss_before_inference_mb': mem_rss_before_infer,
        'psutil_rss_after_inference_mb': mem_rss_after_infer,
        'psutil_rss_delta_inference_mb': rss_delta_infer_mb,
        'tracemalloc_peak_train_bytes': peak_trace_train,
        'tracemalloc_peak_inference_bytes': peak_trace_infer
    }
    
    if ds == "eurosat":
        results['accuracy'] = accuracy_score(y_test, y_pred),

    return results

def prune_optimization(
    ds_train,
    ds_test,
    build_model,
    build_model_params: Dict[str, Any],
    ds: str,
):
    """
    Generate a comprehensive report following pruning optimization,
    measuring both RSS via psutil and peak allocations via tracemalloc.
    For Deep Learning models
    """
    config = build_model_params.get("Config")
    best_params = build_model_params.get("best_params")
    run = build_model_params.get("Run")
    seed = build_model_params.get("Seed")

    std, mean = (0,0)    
    if ds == "hyperview":
        X_train, y_train = ds_train
        mean = y_train.mean(axis=0)
        std = y_train.std(axis=0)
        ds_train = CustomDataset(X_train, y_train, train_transform, mean, std)
        ds_test = TensorDataset(ds_test)
        
    # Instantiate model
    pruned_model, optimizer, scheduler, batch_size, criterion = build_model(best_params, config, None, ds)
    pruned_model.load_state_dict(torch.load(config["output_paths"]["output_result_path"] + f"/best_final_model_{run}" ))    
    pruned_model.eval()

    proc = psutil.Process()

    # --- PRUNING PHASE ---
    mem_rss_before_prune = proc.memory_info().rss / (1024 ** 2)
    tracemalloc.start()
    t0 = time.perf_counter()
    parameters_to_prune = []
    for module_name, module in pruned_model.named_modules():
        if isinstance(module, torch.nn.Conv2d):
            parameters_to_prune.append((module, "weight"))

    # Gestione del layer finale in base alla struttura
    if hasattr(pruned_model, "fc"):
        parameters_to_prune.append((pruned_model.fc, "weight"))
    elif hasattr(pruned_model, "classifier") and isinstance(pruned_model.classifier, torch.nn.Sequential):
        # Prune solo l'ultimo Linear della Sequential
        for layer in reversed(pruned_model.classifier):
            if isinstance(layer, torch.nn.Linear):
                parameters_to_prune.append((layer, "weight"))
                break
    prune.global_unstructured(
        parameters_to_prune,
        pruning_method=prune.L1Unstructured,
        amount=0.4
    )
    for module,name in parameters_to_prune:
        prune.remove(module,name)
        
    prune_duration = time.perf_counter() - t0
    _, peak_trace_prune = tracemalloc.get_traced_memory()
    tracemalloc.reset_peak()
    mem_rss_after_prune = proc.memory_info().rss / (1024 ** 2)
    rss_delta_prune_mb = mem_rss_after_prune - mem_rss_before_prune
    torch.save(pruned_model.state_dict(), config["output_paths"]["output_result_path"] + f"/pruned_model_{run}")
        
     # --- INFERENCE PHASE ---
    mem_rss_before_infer = proc.memory_info().rss / (1024 ** 2)
    tracemalloc.start()
    test_dataloader = DataLoader(ds_test, batch_size=batch_size, shuffle=False)
    t1 = time.perf_counter()
    if ds == "eurosat":
        y_pred, y_test = evaluate(pruned_model, test_dataloader, DEVICE, True)
    elif ds == "hyperview":
        predict(pruned_model, test_dataloader, std, mean, config["output_paths"]["output_result_path"] + f"/submission_{run}.csv")
    inference_duration = time.perf_counter() - t1
    _, peak_trace_infer = tracemalloc.get_traced_memory()
    tracemalloc.stop()
    mem_rss_after_infer = proc.memory_info().rss / (1024 ** 2)
    rss_delta_infer_mb = mem_rss_after_infer - mem_rss_before_infer

    # --- MODEL SIZE ---
    model_bytes = pickle.dumps(pruned_model)
    model_size_bytes = sys.getsizeof(model_bytes)

    # --- COMPILE RESULTS ---
    results: Dict[str, Any] = {
        'seed': seed,
        'prune_duration_s': prune_duration,
        'inference_duration_s': inference_duration,
        'model_size_bytes': model_size_bytes,
        'psutil_rss_before_prune_mb': mem_rss_before_prune,
        'psutil_rss_after_prune_mb': mem_rss_after_prune,
        'psutil_rss_delta_prune_mb': rss_delta_prune_mb,
        'psutil_rss_before_inference_mb': mem_rss_before_infer,
        'psutil_rss_after_inference_mb': mem_rss_after_infer,
        'psutil_rss_delta_inference_mb': rss_delta_infer_mb,
        'tracemalloc_peak_prune_bytes': peak_trace_prune,
        'tracemalloc_peak_inference_bytes': peak_trace_infer
    }
    
    if ds == "eurosat":
        results['accuracy'] = accuracy_score(y_test, y_pred),

    return results

def quantize_optimization(
    ds_train,
    ds_test,
    build_model,
    build_model_params: Dict[str, Any],
    ds: str,
):
    """
    Generate a comprehensive report following pruning optimization,
    measuring both RSS via psutil and peak allocations via tracemalloc.
    For Deep Learning models
    """
    config = build_model_params.get("Config")
    best_params = build_model_params.get("best_params")
    run = build_model_params.get("Run")
    seed = build_model_params.get("Seed")

    std, mean = (0,0)    
    if ds == "hyperview":
        X_train, y_train = ds_train
        mean = y_train.mean(axis=0)
        std = y_train.std(axis=0)
        ds_train = CustomDataset(X_train, y_train, train_transform, mean, std)
        ds_test = TensorDataset(ds_test)
        
    
    # Instantiate model
    q_model, optimizer, scheduler, batch_size, criterion = build_model(best_params, config, "quant", ds)
    q_model.load_state_dict(torch.load(config["output_paths"]["output_result_path"] + f"/best_final_model_{run}"))    
    q_model.eval()

    proc = psutil.Process()

    # --- QUANTIZATION PHASE ---
    mem_rss_before_quant = proc.memory_info().rss / (1024 ** 2)
    tracemalloc.start()
    test_dataloader = DataLoader(ds_test, batch_size=batch_size, shuffle=False)
    t0 = time.perf_counter()
    q_model.qconfig = torch.quantization.QConfig(
        activation=torch.quantization.MinMaxObserver.with_args(
            dtype=torch.quint8, quant_min=0, quant_max=255
        ),
        weight=torch.quantization.PerChannelMinMaxObserver.with_args(
            dtype=torch.qint8, quant_min=-128, quant_max=127,
            qscheme=torch.per_channel_symmetric
        )
    )
    # q_model.qconfig = torch.quantization.get_default_qconfig("fbgemm")
    q_model.fuse_model(is_qat=False)
    prepared_model = torch.quantization.prepare(q_model)
    evaluate(prepared_model, test_dataloader, "cpu", False)
    prepared_model.eval()
    quantized_model = torch.quantization.convert(prepared_model)
    quant_duration = time.perf_counter() - t0
    _, peak_trace_quant = tracemalloc.get_traced_memory()
    tracemalloc.reset_peak()
    mem_rss_after_quant = proc.memory_info().rss / (1024 ** 2)
    rss_delta_quant_mb = mem_rss_after_quant - mem_rss_before_quant
    torch.save(quantized_model.state_dict(), config["output_paths"]["output_result_path"] + f"/quantized_model_{run}")
        
     # --- INFERENCE PHASE ---
    mem_rss_before_infer = proc.memory_info().rss / (1024 ** 2)
    tracemalloc.start()
    test_dataloader = DataLoader(ds_test, batch_size=batch_size, shuffle=False)
    t1 = time.perf_counter()
    if ds == "eurosat":
        y_pred, y_test = evaluate(quantized_model, test_dataloader, "cpu", True)
    elif ds == "hyperview":
        predict(quantized_model, test_dataloader, std, mean, config["output_paths"]["output_result_path"] + f"/submission_{run}.csv")
    inference_duration = time.perf_counter() - t1
    _, peak_trace_infer = tracemalloc.get_traced_memory()
    tracemalloc.stop()
    mem_rss_after_infer = proc.memory_info().rss / (1024 ** 2)
    rss_delta_infer_mb = mem_rss_after_infer - mem_rss_before_infer

    # --- MODEL SIZE ---
    model_bytes = pickle.dumps(quantized_model)
    model_size_bytes = sys.getsizeof(model_bytes)

    # --- COMPILE RESULTS ---
    results: Dict[str, Any] = {
        'seed': seed,
        'quant_duration_s': quant_duration,
        'inference_duration_s': inference_duration,
        'model_size_bytes': model_size_bytes,
        'psutil_rss_before_quant_mb': mem_rss_before_quant,
        'psutil_rss_after_quant_mb': mem_rss_after_quant,
        'psutil_rss_delta_quant_mb': rss_delta_quant_mb,
        'psutil_rss_before_inference_mb': mem_rss_before_infer,
        'psutil_rss_after_inference_mb': mem_rss_after_infer,
        'psutil_rss_delta_inference_mb': rss_delta_infer_mb,
        'tracemalloc_peak_quant_bytes': peak_trace_quant,
        'tracemalloc_peak_inference_bytes': peak_trace_infer
    }
    
    if ds == "eurosat":
        results['accuracy'] = accuracy_score(y_test, y_pred),
    return results

def train_knowledge_distillation(teacher, student, dist_params, criterion, optimizer, scheduler, train_dataloader, val_dataloader, project_cfg, run, ds):
    ce_loss = nn.CrossEntropyLoss()
    teacher.eval()

    epoch_number = 0
    best_vloss = 1_000_000.
    patience = 0
    max_patience = project_cfg["training_params"]["max_patience"]
    max_epochs = project_cfg["training_params"]["max_epochs_training"]
    
    T = dist_params["temperature"]

    for epoch in range(max_epochs):
        student.train() # Student to train mode

        running_loss = 0.
        num_correct = 0
        num_samples = 0
        running_vloss = 0.0
        num_vcorrect = 0
        num_vsamples = 0

        for _, data in enumerate(train_dataloader):
            inputs, labels = data
            optimizer.zero_grad()
            with torch.no_grad():
                teacher_logits = teacher(inputs.to(device, non_blocking=True))
            student_logits = student(inputs.to(device, non_blocking=True))

            # Calculate the soft targets loss. Scaled by T**2 as suggested by the authors of the paper "Distilling the knowledge in a neural network"
            log_probs = nn.functional.log_softmax(student_logits / T, dim=-1)
            target_probs = nn.functional.softmax(teacher_logits / T, dim=-1)
            soft_targets_loss = nn.functional.kl_div(log_probs, target_probs, reduction='batchmean')

            soft_targets_loss.backward()
            optimizer.step()
            running_loss += soft_targets_loss.item()
            
            if ds == "eurosat":
                _, predictions = student_logits.max(dim=-1)
                num_correct += (predictions == labels.to(device, non_blocking=True)).sum()
                num_samples += predictions.size(0)

        student.eval()
        with torch.no_grad():
            for i, vdata in enumerate(val_dataloader):
                vinputs, vlabels = vdata
                voutputs = student(vinputs.to(device, non_blocking=True))
                vloss = criterion(voutputs, vlabels.to(device, non_blocking=True))
                running_vloss += vloss
                if ds == "eurosat":
                    _, vpredictions = voutputs.max(dim=-1)
                    num_vcorrect += (vpredictions == vlabels.to(device, non_blocking=True)).sum()
                    num_vsamples += vpredictions.size(0)

        avg_vloss = running_vloss / len(val_dataloader)

        if math.isnan(avg_vloss) or math.isinf(avg_vloss):
            break

        if avg_vloss < best_vloss:
            best_vloss = avg_vloss
            patience = 0
        else:
            if patience >= max_patience:
                break
            patience+=1

        if scheduler:
            scheduler.step()
        epoch_number += 1
        
    return

def distill_optimization(
    ds_train,
    ds_test,
    build_model,
    build_model_params: Dict[str, Any],
    ds: str,
):
    """
    Generate a comprehensive report following pruning optimization,
    measuring both RSS via psutil and peak allocations via tracemalloc.
    For Deep Learning models
    """
    config = build_model_params.get("Config")
    best_params = build_model_params.get("best_params")
    run = build_model_params.get("Run")
    seed = build_model_params.get("Seed")

    std, mean = (0,0)    
    if ds == "hyperview":
        X_train, y_train = ds_train
        mean = y_train.mean(axis=0)
        std = y_train.std(axis=0)
        ds_train = CustomDataset(X_train, y_train, train_transform, mean, std)
        ds_test = TensorDataset(ds_test)  
        
    # Instantiate model
    
    teacher, _, _, _, _ = build_model(best_params, config, None, ds)
    teacher.load_state_dict(torch.load(config["output_paths"]["output_result_path"] + f"/best_final_model_{run}"))    
    teacher.eval()   
    
    student, optimizer, scheduler, batch_size, criterion = build_model(best_params, config, "dist", ds)
    student.eval()   
    
    student_baseline, optimizer_baseline, scheduler_baseline, batch_size, criterion_baseline = build_model(best_params, config, "dist", ds)
    student_baseline.eval()
    best_model_baseline, _, _, _, _ = build_model(best_params, config, "dist", ds)
    
    proc = psutil.Process() 
    
    # --- BASELINE TRAINING PHASE ---
    mem_rss_before_train = proc.memory_info().rss / (1024 ** 2)
    tracemalloc.start()    
    train_dataloader = DataLoader(ds_train, batch_size=batch_size, shuffle=True)
    t0 = time.perf_counter()
    best_wts = train(student_baseline, criterion_baseline, optimizer_baseline, scheduler_baseline, train_dataloader, None, False, config, ds)
    train_duration = time.perf_counter() - t0
    best_model_baseline.load_state_dict(best_wts)
    torch.save(best_model_baseline.state_dict(), config["output_paths"]["output_result_path"] + f"/best_student_baseline_model_{run}")
    _, peak_trace_train = tracemalloc.get_traced_memory()
    tracemalloc.reset_peak()
    mem_rss_after_train = proc.memory_info().rss / (1024 ** 2)
    rss_delta_train_mb = mem_rss_after_train - mem_rss_before_train
       
    # --- DISTILLATION PHASE ---
    distillation_params = {}
    distillation_params["temperature"] = 2
    distillation_params["soft_target_loss_weight"] = 0.9
    distillation_params["ce_loss_weight"] = 0.1
    
    mem_rss_before_dist = proc.memory_info().rss / (1024 ** 2)
    tracemalloc.start()
    train_dataloader, val_dataloader = dl.get_dataloader(ds_train, ds, config, batch_size)
    t0 = time.perf_counter()
    train_knowledge_distillation(
        teacher,
        student,
        distillation_params,
        criterion,
        optimizer,
        scheduler,
        train_dataloader,
        val_dataloader,
        config,
        ds,
        run
    )    
    dist_duration = time.perf_counter() - t0
    _, peak_trace_dist = tracemalloc.get_traced_memory()
    tracemalloc.reset_peak()
    mem_rss_after_dist = proc.memory_info().rss / (1024 ** 2)
    rss_delta_dist_mb = mem_rss_after_dist - mem_rss_before_dist
    torch.save(student.state_dict(), config["output_paths"]["output_result_path"] + f"/student_model_{run}")
      
     # --- INFERENCE PHASE ---
    # Baseline Inference
    mem_rss_before_infer_baseline = proc.memory_info().rss / (1024 ** 2)
    tracemalloc.start()
    test_dataloader = DataLoader(ds_test, batch_size=batch_size, shuffle=False)
    t1 = time.perf_counter()
    if ds == "eurosat":
        y_pred_baseline, y_test_baseline = evaluate(best_model_baseline, test_dataloader, DEVICE, True)
    elif ds == "hyperview":
        predict(best_model_baseline, test_dataloader, std, mean, config["output_paths"]["output_result_path"] + f"/submission_baseline_{run}.csv")
    inference_duration_baseline = time.perf_counter() - t1
    _, peak_trace_infer_baseline = tracemalloc.get_traced_memory()
    tracemalloc.stop()
    mem_rss_after_infer_baseline = proc.memory_info().rss / (1024 ** 2)
    rss_delta_infer_baseline_mb = mem_rss_after_infer_baseline - mem_rss_before_infer_baseline 
     
    # Student Inference
    mem_rss_before_infer = proc.memory_info().rss / (1024 ** 2)
    tracemalloc.start()
    test_dataloader = DataLoader(ds_test, batch_size=batch_size, shuffle=False)
    t1 = time.perf_counter()
    if ds == "eurosat":
        y_pred, y_test = evaluate(student, test_dataloader, DEVICE, True)
    elif ds == "hyperview":
        predict(student, test_dataloader, std, mean, config["output_paths"]["output_result_path"] + f"/submission_{run}.csv")
    inference_duration = time.perf_counter() - t1
    _, peak_trace_infer = tracemalloc.get_traced_memory()
    tracemalloc.stop()
    mem_rss_after_infer = proc.memory_info().rss / (1024 ** 2)
    rss_delta_infer_mb = mem_rss_after_infer - mem_rss_before_infer

    # --- MODEL SIZE ---
    baseline_model_bytes = pickle.dumps(best_model_baseline)
    baseline_model_size_bytes = sys.getsizeof(baseline_model_bytes)
    model_bytes = pickle.dumps(student)
    model_size_bytes = sys.getsizeof(model_bytes)

    # --- COMPILE RESULTS ---
    results: Dict[str, Any] = {
        'seed': seed,
        'baseline_train_duration_s': train_duration,
        'baseline_inference_duration_s': inference_duration_baseline,
        'dist_duration_s': dist_duration,
        'inference_duration_s': inference_duration,
        'model_size_bytes': model_size_bytes,
        'baseline_model_size_bytes': baseline_model_size_bytes,
        'psutil_rss_before_train_mb': mem_rss_before_train,
        'psutil_rss_after_train_mb': mem_rss_after_train,
        'psutil_rss_delta_train_mb': rss_delta_train_mb,
        'psutil_rss_before_inference_baseline_mb': mem_rss_before_infer_baseline,
        'psutil_rss_after_inference_baseline_mb': mem_rss_after_infer_baseline,
        'psutil_rss_delta_inference_baseline_mb': rss_delta_infer_baseline_mb,
        'psutil_rss_before_dist_mb': mem_rss_before_dist,
        'psutil_rss_after_dist_mb': mem_rss_after_dist,
        'psutil_rss_delta_dist_mb': rss_delta_dist_mb,
        'psutil_rss_before_inference_mb': mem_rss_before_infer,
        'psutil_rss_after_inference_mb': mem_rss_after_infer,
        'psutil_rss_delta_inference_mb': rss_delta_infer_mb,
        'tracemalloc_peak_train_bytes': peak_trace_train,
        'tracemalloc_peak_inference_bytes_baseline': peak_trace_infer_baseline,
        'tracemalloc_peak_dist_bytes': peak_trace_dist,
        'tracemalloc_peak_inference_bytes': peak_trace_infer
    }
    
    if ds == "eurosat":
        results['accuracy'] = accuracy_score(y_test, y_pred)
        results['baseline_accuracy'] = accuracy_score(y_test_baseline, y_pred_baseline)

    return results