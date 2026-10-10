import json
from pathlib import Path

import torch
from safetensors.torch import save_file, load_file


def save(path: Path, tensors: dict[str, torch.Tensor], meta: dict) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    save_file({k: v.contiguous().to(torch.float16) for k, v in tensors.items()}, str(path.with_suffix(".safetensors")))
    path.with_suffix(".json").write_text(json.dumps(meta, indent=1))


def load(path: Path) -> tuple[dict[str, torch.Tensor], dict]:
    return load_file(str(path.with_suffix(".safetensors"))), json.loads(path.with_suffix(".json").read_text())
