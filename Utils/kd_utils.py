from dataclasses import dataclass

import torch
import torch.nn.functional as F


@dataclass(frozen=True)
class KDLossConfig:
    task_type: str
    alpha: float
    temperature: float = 1.0

    def __call__(self, student_outputs, teacher_outputs, labels):
        if self.task_type == "classification":
            return classification_kd_loss(
                student_outputs,
                teacher_outputs,
                labels,
                alpha=self.alpha,
                temperature=self.temperature,
            )
        if self.task_type == "regression":
            return regression_kd_loss(
                student_outputs,
                teacher_outputs,
                labels,
                alpha=self.alpha,
            )
        raise ValueError(f"Unsupported KD task type: {self.task_type}")


def classification_kd_loss(student_logits, teacher_logits, labels, alpha, temperature):
    kl_term = F.kl_div(
        F.log_softmax(student_logits / temperature, dim=-1),
        F.softmax(teacher_logits / temperature, dim=-1),
        reduction="batchmean",
    ) * (temperature ** 2)
    ce_term = F.cross_entropy(student_logits, labels)
    return alpha * kl_term + (1.0 - alpha) * ce_term


def regression_kd_loss(student_outputs, teacher_outputs, labels, alpha):
    teacher_term = F.mse_loss(student_outputs, teacher_outputs)
    label_term = F.mse_loss(student_outputs, labels)
    return alpha * teacher_term + (1.0 - alpha) * label_term


def build_kd_loss(dataset_name, alpha, temperature=1.0):
    if dataset_name == "eurosat":
        return KDLossConfig(task_type="classification", alpha=alpha, temperature=temperature)
    if dataset_name == "hyperview":
        return KDLossConfig(task_type="regression", alpha=alpha, temperature=temperature)
    raise ValueError(f"Unsupported dataset for KD: {dataset_name}")
