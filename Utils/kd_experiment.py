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
        "run_flags": {
            "distillation": bool(args.kd_enabled),
            "pruning": bool(args.prune_enabled),
            "quantization": bool(args.quant_enabled),
        },
        "output_tag": args.output_tag,
        "assessment_seeds": _parse_seed_list(args.assessment_seeds),
        "dataset_name": dataset_name,
    }
