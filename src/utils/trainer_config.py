def resolve_trainer_strategy(num_gpus: int) -> str:
    if num_gpus > 1:
        return "ddp_find_unused_parameters_false"
    return "auto"
