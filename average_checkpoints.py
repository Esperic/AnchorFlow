import argparse
from pathlib import Path

import torch


def average_checkpoints(paths):
    checkpoints = [torch.load(path, map_location="cpu") for path in paths]
    state_dicts = [checkpoint["state_dict"] for checkpoint in checkpoints]
    if any(state.keys() != state_dicts[0].keys() for state in state_dicts[1:]):
        raise ValueError("checkpoints have different state_dict keys")

    averaged = {}
    for key in state_dicts[0]:
        tensors = [state[key] for state in state_dicts]
        if any(tensor.shape != tensors[0].shape for tensor in tensors[1:]):
            raise ValueError(f"checkpoint tensor shape mismatch: {key}")
        if tensors[0].is_floating_point():
            value = sum(tensor.double() for tensor in tensors) / len(tensors)
            averaged[key] = value.to(tensors[0].dtype)
        else:
            averaged[key] = tensors[-1].clone()

    checkpoint = checkpoints[-1]
    checkpoint["state_dict"] = averaged
    return checkpoint


def main():
    parser = argparse.ArgumentParser(
        description="Average model weights from checkpoints of the same training run."
    )
    parser.add_argument("checkpoints", nargs="+", type=Path)
    parser.add_argument("--output", required=True, type=Path)
    args = parser.parse_args()
    if len(args.checkpoints) < 2:
        parser.error("provide at least two checkpoints")

    args.output.parent.mkdir(parents=True, exist_ok=True)
    torch.save(average_checkpoints(args.checkpoints), args.output)
    print(f"saved averaged checkpoint to {args.output}")


if __name__ == "__main__":
    main()
