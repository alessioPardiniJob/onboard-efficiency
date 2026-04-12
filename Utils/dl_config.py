DEFAULT_TRAINING_PARAMS = {
    "batch_size_optuna": 32,
    "max_epochs_optuna": 5,
    "max_epochs_training": 10,
    "max_patience": 3,
    "model_name": None,
    "pretrained": None,
    "optimizer": None,
    "scheduler": None,
}


def ensure_training_params(config):
    training_params = dict(DEFAULT_TRAINING_PARAMS)
    training_params.update(config.get("training_params", {}))
    config["training_params"] = training_params
    return config


def normalize_directory_dataset_root(path):
    if path.endswith("/"):
        return path
    return path + "/"
