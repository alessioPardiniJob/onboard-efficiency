## Prerequisites

Ensure the following requirements are satisfied before running the project:

- **Operating System:** Linux (Ubuntu/Debian recommended) or Windows with WSL2  
- **Python:** ≥ 3.8 (including `venv`)  
- **System Tools:** `make`, `build-essential`, `git`  
- **Network:** Internet access for dependency and dataset download  

> ⚠️ **WSL2 DNS Issue**  
> If you encounter errors such as `Temporary failure in name resolution` or `[Errno -3]`, execute:
> ```powershell
> wsl --shutdown
> ```
> Then reopen the Linux terminal and retry.

If your environment already meets these prerequisites, proceed directly to the **Execution Workflow**.  
For improved reproducibility and isolation, the Pyenv-based setup below is recommended.

---

## Recommended Setup (Pyenv)

This procedure installs a local Python version without modifying the system interpreter.

### 1. Install System Dependencies (Linux)

```bash
sudo apt update
sudo apt install -y \
  build-essential make git cmake \
  libbz2-dev libssl-dev libffi-dev \
  libncurses5-dev libncursesw5-dev \
  libreadline-dev libsqlite3-dev \
  liblzma-dev zlib1g-dev tk-dev \
  xz-utils wget curl llvm
```
### 2. Install Pyenv
```bash
curl https://pyenv.run | bash

echo 'export PATH="$HOME/.pyenv/bin:$PATH"' >> ~/.bashrc
echo 'eval "$(pyenv init --path)"' >> ~/.bashrc
echo 'eval "$(pyenv virtualenv-init -)"' >> ~/.bashrc

source ~/.bashrc
```
### 3. Configure Project Python Version
```bash
cd path/to/onboard-efficiency

pyenv install 3.10.13
pyenv local 3.10.13
```
### 4. Verify Installation
```bash
python --version   # Expected: Python 3.10.13
make --version
```
## Execution Workflow

All experiments are managed through the centralized `Makefile`.

Replace `[DATASET]` with one of:

- `EuroSAT`
- `Hyperview`

---

## Datasets

### EuroSAT

EuroSAT is downloaded automatically via `torchvision.datasets.EuroSAT` when you run the EuroSAT pipelines.

### HYPERVIEW (Seeing Beyond the Visible)

The HYPERVIEW Challenge dataset is **not distributed with this repository** and is **not downloaded automatically** by the code. Please obtain it from the official source (registration and acceptance of terms may be required):

- ESA Φ-lab AI4EO platform: https://platform.ai4eo.eu/seeing-beyond-the-visible-permanent

Download instructions (including the EOTDL tool) are available here:

- https://www.eotdl.com/datasets/SeeingBeyondTheVisible

Place the dataset under the path configured by `dataset_root_path` (default: `Hyperview/data`) with this structure:

```
train_data/   (NPZ files)
test_data/    (NPZ files)
train_gt.csv
```

### Environment Setup

Creates the dataset-specific virtual environment and installs dependencies.

```bash
make setup-[DATASET]
```
**Example**

```bash
make setup-EuroSAT
```

## Machine Learning Pipeline

### 1. Model Selection (Hyperparameter Optimization)

Runs Optuna-based hyperparameter tuning.

```bash
make ml-select-[DATASET] MODEL=[rf|xg] SIZE=[small|big]
```
**Example**
```bash
make ml-select-EuroSAT MODEL=xg SIZE=small
```
### 2. Model Assessment (Evaluation)

Retrains the model using the best configuration and evaluates on the test set.

```bash
make ml-assess-[DATASET] MODEL=[rf|xg] SIZE=[small|big] MODE=[manual|auto]
```
**Example**
```bash
make ml-assess-EuroSAT MODEL=xg SIZE=small MODE=manual
```
## Notes on Assessment

### MODE Parameter

- **manual**: Uses the best hyperparameters obtained from our own model selection process for each model and size configuration.  

- **auto**: Automatically retrieves the best hyperparameters from previous Optuna-based model selection results stored in the project’s output directories.

### Hyperview-specific Assessment

For `Hyperview`, assessment results are not directly computed locally. After retraining, model outputs must be submitted on the official Hyperview platform. Submission results are stored in the project under:
```bash
Hyperview/result/modelAssessment/<model_name>/test<n>/
```
Each test seed produces a separate folder (`test1`, `test2`, etc.) containing the assessment metrics. This ensures that evaluation follows Hyperview's official protocol.


## Deep Learning Pipeline

Deep learning experiments follow the same structure:

```bash
make dl-select-[DATASET]
make dl-assess-[DATASET]
```
(Refer to dataset-specific configuration files for architecture and training parameters.)

## Maintenance & Cleanup
Removes virtual environments and cached artifacts.
```bash
make clean-[DATASET]
```
**Example**
```bash
make clean-Hyperview
```
## Notes for Reviewers

- Each dataset uses an isolated virtual environment: `[DATASET]/venv`
- Hyperparameter search results are persisted automatically
- Experiments are deterministic where seeds are defined
- Debug/subsampling is disabled by default; run the Python entrypoints directly with `--debug` for quick development runs.
- Re-running targets overwrites previous outputs unless explicitly cached
