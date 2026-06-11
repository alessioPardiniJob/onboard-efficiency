# Makefile: Setup and execution workflow for ML/DL pipelines on EuroSAT & Hyperview.

SHELL := /bin/bash
PROJECT_ROOT := $(shell pwd)

# --------------------------------------------------------
# CONFIGURATION VARIABLES
# --------------------------------------------------------

# Python interpreter detection
PYTHON ?= python3

# Default parameters for ML pipelines (overridable via CLI)
# Example usage: make ml-select-EuroSAT MODEL=gb SIZE=big
MODEL ?= rf
SIZE ?= small
MODE ?= manual

# List of supported datasets (for reference only)
DATASETS := EuroSAT Hyperview

.PHONY: setup-% clean-% help
.PHONY: ml-select-% ml-assess-% dl-select-% dl-assess-%

# --------------------------------------------------------
# MACROS / HELPERS
# --------------------------------------------------------

define VENV_PYTHON_PATH
$*/venv/bin/python
endef

# --------------------------------------------------------
# HELP / GUIDE
# --------------------------------------------------------
help:
	@echo "---- Project Makefile Guide ----"
	@echo "Usage: make [target]"
	@echo ""
	@echo "Environment Setup:"
	@echo "  make setup-EuroSAT          : Creates virtual environment and installs dependencies."
	@echo "  make setup-Hyperview        : Same as above, for Hyperview dataset."
	@echo ""
	@echo "Machine Learning Pipeline (Module: Pardini):"
	@echo "  make ml-select-EuroSAT      : Executes Model Selection (Optuna optimization)."
	@echo "                                Options: MODEL=rf|gb SIZE=small|big"
	@echo "  make ml-assess-EuroSAT      : Executes Model Assessment (Retraining best model)."
	@echo "                                Options: MODEL=rf|gb SIZE=small|big"
	@echo ""
	@echo "Deep Learning Pipeline (Module: DiPalma):"
	@echo "  make dl-select-EuroSAT      : Executes DL Model Selection (Optuna optimization)."
	@echo "  make dl-assess-EuroSAT      : Executes DL Model Assessment (Retrain + Evaluate)."
	@echo "                                Options: MODE=manual|auto"
	@echo ""
	@echo "modelSelectiontenance:"
	@echo "  make clean-EuroSAT          : Removes virtual environment and temporary files."

# --------------------------------------------------------
# SETUP TARGETS
# --------------------------------------------------------

setup-%:
	@echo
	@echo "---- 🛠️  Setting up dataset: $* ----"
	@if [ ! -d "Utils" ]; then echo "⚠️  Warning: 'Utils/' directory not found. Please verify project structure."; fi
	# Create venv
	@if [ ! -d "$*/venv" ]; then \
		echo "📦 Creating virtual environment for $*..."; \
		$(PYTHON) -m venv "$*/venv" || { echo "❌ Virtual environment creation failed."; exit 1; }; \
	else \
		echo "✅ Virtual environment already exists for $*."; \
	fi
	# Install dependencies
	@echo "📥 Installing/Upgrading dependencies..."
	@TMPDIR="$(TMPDIR)" bash scripts/setup_env.sh "$*/venv" "$*/requirements.txt" || { echo "❌ Dependency installation failed."; exit 1; }
	@echo "✅ Setup completed successfully for $*."
	@echo

# --------------------------------------------------------
# MACHINE LEARNING PIPELINE (code-Pardini)
# --------------------------------------------------------

# 1. Model Selection (Runs modelSelection.py with CLI arguments)
ml-select-%:
	@echo
	@echo "---- 🧠 ML Model SELECTION for $* [Model: $(MODEL) | Size: $(SIZE)] ----"
	@if [ ! -d "$*/venv" ]; then echo "❌ Error: Environment not found. Please run 'make setup-$*' first."; exit 1; fi
	
	@# Execute modelSelection.py passing the defined variables
	bash -c 'set -e; \
		$(call VENV_PYTHON_PATH) "$*/src/pipelineMl/modelSelection.py" --model $(MODEL) --size $(SIZE)' \
		|| { echo "❌ ML Selection failed for $*. Check logs for details."; exit 1; }
	
	@echo "✅ ML Selection completed successfully for $*."
	@echo

# 2. Model Assessment (Runs retrain_best.py with CLI arguments)
# 2. Model Assessment (Runs retrain_best.py with CLI arguments)
ml-assess-%:
	@echo
	@echo "---- 📊 ML Model ASSESSMENT for $* [Model: $(MODEL) | Size: $(SIZE) | Mode: $(MODE)] ----"
	@if [ ! -d "$*/venv" ]; then echo "❌ Error: Environment not found. Please run 'make setup-$*' first."; exit 1; fi
	
	@bash -c 'set -e; \
		$(call VENV_PYTHON_PATH) "$*/src/pipelineMl/modelAssessment.py" \
		--model $(MODEL) \
		--size $(SIZE) \
		--best-params-mode $(MODE)' \
		|| { echo "❌ ML Assessment failed for $*. Check logs for details."; exit 1; }
	
	@echo "✅ ML Assessment completed successfully for $*."
	@echo

# --------------------------------------------------------
# DEEP LEARNING PIPELINE (code-DiPalma)
# --------------------------------------------------------

# 1. Model Selection (DL)
dl-select-%:
	@echo
	@echo "---- 🕸️  DL Model SELECTION for $* ----"
	@if [ ! -d "$*/venv" ]; then echo "❌ Error: Environment not found. Please run 'make setup-$*' first."; exit 1; fi
	
	@bash -c 'set -e; \
		$(call VENV_PYTHON_PATH) "$*/src/pipelineDl/modelSelection.py"' \
		|| { echo "❌ DL Selection failed for $*."; exit 1; }
	
	@echo "✅ DL Selection completed successfully for $*."
	@echo

# 2. Model Assessment (DL)
dl-assess-%:
	@echo
	@echo "---- 📉 DL Model ASSESSMENT for $* [Mode: $(MODE)] ----"
	@if [ ! -d "$*/venv" ]; then echo "❌ Error: Environment not found. Please run 'make setup-$*' first."; exit 1; fi
	
	@bash -c 'set -e; \
		$(call VENV_PYTHON_PATH) "$*/src/pipelineDl/modelAssessment.py" \
		--best-params-mode $(MODE)' \
		|| { echo "❌ DL Assessment failed for $*."; exit 1; }
	
	@echo "✅ DL Assessment completed successfully for $*."
	@echo

# --------------------------------------------------------
# CLEANUP TARGETS
# --------------------------------------------------------

clean-%:
	@echo "---- 🧹 Cleaning environment for $* ----"
	@rm -rf "$*/venv" && echo "🗑️  Removed virtual environment ($*/venv)."
	@if [ -d "$(PROJECT_ROOT)/utils.egg-info" ]; then \
		rm -rf "$(PROJECT_ROOT)/utils.egg-info" && echo "🗑️  Removed root utils.egg-info."; \
	fi
	@echo "✅ Cleanup completed."
