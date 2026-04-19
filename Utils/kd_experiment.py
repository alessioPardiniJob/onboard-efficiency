import os


def _parse_seed_list(seed_text):
    if not seed_text:
        return None
    return [int(item.strip()) for item in seed_text.split(",") if item.strip()]


def resolve_student_variant_specs():
    return {
        "resnet": {
            "small": "resnet18",
            "medium": "resnet34",
        },
        "mobilenetv3": {
            "small": "mobilenet_v3_small",
        },
        "shufflenet": {
            "small": "shufflenet_v2_x0_5",
        },
    }


def resolve_kd_study_config(args, dataset_name):
    return {
        "distillation_params": {
            "alpha": 0.9 if args.kd_alpha is None else args.kd_alpha,
            "temperature": 2.0 if args.kd_temperature is None else args.kd_temperature,
        },
        "student_variant": args.kd_student_variant or "small",
        "teacher_checkpoint_root": getattr(args, "teacher_checkpoint_root", None),
        "baseline_only": bool(getattr(args, "baseline_only", False)),
        "run_flags": {
            "distillation": bool(args.kd_enabled),
            "pruning": bool(args.prune_enabled),
            "quantization": bool(args.quant_enabled),
        },
        "output_tag": args.output_tag,
        "assessment_seeds": _parse_seed_list(args.assessment_seeds),
        "dataset_name": dataset_name,
    }


def resolve_teacher_checkpoint_path(config, run, seed=None):
    shared_root = config.get("kd_study", {}).get("teacher_checkpoint_root")
    if shared_root:
        if seed is not None:
            teacher_root = os.path.join(shared_root, "ModelAssessment")
            if os.path.isdir(teacher_root):
                for entry in sorted(os.listdir(teacher_root)):
                    seed_checkpoint = os.path.join(teacher_root, entry, f"best_final_model_seed_{seed}")
                    if os.path.exists(seed_checkpoint):
                        return seed_checkpoint
        return os.path.join(shared_root, "ModelAssessment", f"test_{run}", f"best_final_model_{run}")
    return os.path.join(config["output_paths"]["output_result_path"], f"best_final_model_{run}")
