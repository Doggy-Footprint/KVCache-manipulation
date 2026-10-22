import copy

import torch


def _lengths(cache) -> list[int | None]:
    return [None if hasattr(l, "recurrent_states") else l.keys.shape[2] for l in cache.layers]


@torch.no_grad()
def stack_caches(caches: list):
    if not caches:
        raise ValueError("no caches to stack")
    ref = _lengths(caches[0])
    lengths = [c.get_seq_length() for c in caches]
    if any(_lengths(c) != ref for c in caches) or len(set(lengths)) > 1:
        raise ValueError(f"batch requires equal-length caches, got {lengths}")
    out = copy.deepcopy(caches[0])
    for i, layer in enumerate(out.layers):
        src = [c.layers[i] for c in caches]
        if hasattr(layer, "recurrent_states"):
            for s in layer.conv_states:
                if layer.conv_states[s] is not None:
                    layer.conv_states[s] = torch.cat([l.conv_states[s] for l in src])
                if layer.recurrent_states[s] is not None:
                    layer.recurrent_states[s] = torch.cat([l.recurrent_states[s] for l in src])
        else:
            layer.keys = torch.cat([l.keys for l in src])
            layer.values = torch.cat([l.values for l in src])
    return out


def split_logits(logits: torch.Tensor, n: int | None = None) -> list[torch.Tensor]:
    if n is not None and logits.shape[0] != n:
        raise ValueError(f"expected batch {n}, got {logits.shape[0]}")
    return list(logits.unbind(0))
