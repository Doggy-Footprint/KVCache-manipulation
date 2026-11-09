import copy
import sys
from collections.abc import Iterable

import torch

from . import model as M
from .batch import stack_caches


def _apply_rope(model):
    attn = M.text_model(model).layers[M.kv_layers(model)[0]].self_attn
    return sys.modules[type(attn).__module__].apply_rotary_pos_emb


def drop_span(cache, span: tuple[int, int]):
    a, b = span
    out = copy.deepcopy(cache)
    for layer in out.layers:
        layer.keys = torch.cat([layer.keys[:, :, :a], layer.keys[:, :, b:]], dim=2)
        layer.values = torch.cat([layer.values[:, :, :a], layer.values[:, :, b:]], dim=2)
    return out


@torch.no_grad()
def rerope(model, cache, start: int, shift: int):
    """Moves cached post-RoPE keys from position p to p - shift for every token at index >= start; RoPE rotations
    compose additively, so one rotation by -shift relocates them without the pre-RoPE keys."""
    rope = _apply_rope(model)
    rotary = M.text_model(model).rotary_emb
    out = copy.deepcopy(cache)
    for layer in out.layers:
        k = layer.keys[:, :, start:].float()
        if k.shape[2] == 0:
            continue
        pos = torch.full((k.shape[0], k.shape[2]), -shift, device=k.device)
        cos, sin = rotary(k, pos)
        _, k_new = rope(k, k, cos.float(), sin.float())
        layer.keys = torch.cat([layer.keys[:, :, :start], k_new.to(layer.keys.dtype)], dim=2)
    return out


def close_gap(model, cache, span: tuple[int, int]):
    a, b = span
    return rerope(model, drop_span(cache, span), a, b - a)


def splice(base, donor, rows: Iterable[int] | tuple[int, int], layers: Iterable[int] | None = None):
    rows = list(range(*rows)) if isinstance(rows, tuple) else list(rows)
    out = copy.deepcopy(base)
    layers = range(len(out.layers)) if layers is None else layers
    idx = torch.tensor(rows, device=out.layers[0].keys.device)
    for l in layers:
        out.layers[l].keys[:, :, idx] = donor.layers[l].keys[:, :, idx].to(out.layers[l].keys.device)
        out.layers[l].values[:, :, idx] = donor.layers[l].values[:, :, idx].to(out.layers[l].values.device)
    return out


@torch.no_grad()
def readout_batch(model, caches: list, suffix_ids: list[int], masked: list[list[tuple[int, int]]] | None = None,
                  start_pos: int | None = None) -> torch.Tensor:
    """Last-position logits [B, vocab] for one shared suffix over equal-length caches; `masked` spans are hidden
    from the suffix through the padding mask, so cached positions and RoPE stay untouched."""
    c = stack_caches(caches) if len(caches) > 1 else copy.deepcopy(caches[0])
    B, T, S = len(caches), caches[0].get_seq_length(), len(suffix_ids)
    dev = c.layers[0].keys.device
    am = torch.ones(B, T + S, dtype=torch.long, device=dev)
    for i, spans in enumerate(masked or [[]] * B):
        for a, b in spans:
            am[i, a:b] = 0
    kw = {}
    if start_pos is not None:
        kw["position_ids"] = torch.arange(start_pos, start_pos + S, device=dev)[None].expand(B, S)
    ids = torch.tensor([suffix_ids] * B, device=dev)
    return model(input_ids=ids, past_key_values=c, attention_mask=am, use_cache=True, **kw).logits[:, -1].float()


@torch.no_grad()
def greedy(model, cache, suffix_ids: list[int], max_new: int, masked: list[tuple[int, int]] = (),
           start_pos: int | None = None, stop_ids: Iterable[int] = ()) -> list[int]:
    c = copy.deepcopy(cache)
    T = c.get_seq_length()
    dev = c.layers[0].keys.device
    am = torch.ones(1, T, dtype=torch.long, device=dev)
    for a, b in masked:
        am[0, a:b] = 0
    pos = T if start_pos is None else start_pos
    stop, ids, out = set(stop_ids), list(suffix_ids), []
    for _ in range(max_new):
        S = len(ids)
        am = torch.cat([am, torch.ones(1, S, dtype=torch.long, device=dev)], dim=1)
        logits = model(input_ids=torch.tensor([ids], device=dev), past_key_values=c, attention_mask=am,
                       position_ids=torch.arange(pos, pos + S, device=dev)[None], use_cache=True).logits[0, -1]
        pos += S
        nxt = int(logits.argmax())
        if nxt in stop:
            break
        out.append(nxt)
        ids = [nxt]
    return out


@torch.no_grad()
def recompute_tail(model, cache, ids: list[int], m: int):
    """Drops the last m cached rows and re-runs those m tokens on top, so they see the current (edited) cache."""
    T = cache.get_seq_length()
    c = drop_span(cache, (T - m, T))
    model(input_ids=torch.tensor([ids[T - m:T]], device=c.layers[0].keys.device), past_key_values=c, use_cache=True)
    return c


@torch.no_grad()
def greedy_batch(model, caches: list, suffix_ids: list[int], max_new: int,
                 masked: list[list[tuple[int, int]]] | None = None, start_pos: int | None = None,
                 stop_ids: Iterable[int] = ()) -> list[list[int]]:
    """Rows share one suffix and one position sequence; finished rows keep decoding until all stop and their
    extra tokens are discarded."""
    c = stack_caches(caches) if len(caches) > 1 else copy.deepcopy(caches[0])
    B, T = len(caches), caches[0].get_seq_length()
    dev = c.layers[0].keys.device
    am = torch.ones(B, T, dtype=torch.long, device=dev)
    for i, spans in enumerate(masked or [[]] * B):
        for a, b in spans:
            am[i, a:b] = 0
    pos = T if start_pos is None else start_pos
    stop = set(stop_ids)
    ids = torch.tensor([suffix_ids] * B, device=dev)
    out, alive = [[] for _ in range(B)], [True] * B
    for _ in range(max_new):
        S = ids.shape[1]
        am = torch.cat([am, torch.ones(B, S, dtype=torch.long, device=dev)], dim=1)
        logits = model(input_ids=ids, past_key_values=c, attention_mask=am,
                       position_ids=torch.arange(pos, pos + S, device=dev)[None].expand(B, S), use_cache=True).logits[:, -1]
        pos += S
        nxt = logits.argmax(-1)
        for i, t in enumerate(nxt.tolist()):
            if alive[i]:
                if t in stop:
                    alive[i] = False
                else:
                    out[i].append(t)
        if not any(alive):
            break
        ids = nxt[:, None]
    return out
