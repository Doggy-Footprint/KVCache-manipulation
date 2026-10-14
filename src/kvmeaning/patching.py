import argparse
import copy
import json
import time
from collections.abc import Iterable
from pathlib import Path

import torch
from transformers import DynamicCache

from . import model as M
from .spans import assert_aligned, exp1_encode, exp1_prefix_ids

DATA = Path(__file__).resolve().parents[2] / "data"
READOUT_SUFFIX = "<|im_end|>\n<|im_start|>assistant\n<think>\n\n</think>\n\n"
DONORS = ("H+", "H-", "P1", "P2", "F2")
CONT_LEN = 16
_KINDS = ("k", "v")
# "prefix" patches [0, body start): the template head is identical across conditions, the delimiter is prefix-dependent.
REGIONS = ("body", "prefix")


def _kv(cache, i: int):
    if hasattr(cache, "layers"):
        return cache.layers[i].keys, cache.layers[i].values
    return cache.key_cache[i], cache.value_cache[i]


def _num_layers(cache) -> int:
    return len(cache.layers) if hasattr(cache, "layers") else len(cache.key_cache)


@torch.no_grad()
def prefill(model, ids: list[int]) -> DynamicCache:
    out = model(input_ids=torch.tensor([ids], device=model.device), use_cache=True)
    return out.past_key_values


@torch.no_grad()
def mix_cache(recipient, donor, span: tuple[int, int], layers: Iterable[int], heads: Iterable[int] | None = None,
              kinds: Iterable[str] = ("k", "v")) -> DynamicCache:
    layers, kinds = list(layers), list(kinds)
    n_layers = _num_layers(recipient)
    k0 = _kv(recipient, 0)[0]
    T, n_heads = k0.shape[2], k0.shape[1]
    heads = list(range(n_heads)) if heads is None else list(heads)
    a, b = span
    if _num_layers(donor) != n_layers or _kv(donor, 0)[0].shape[2] != T or _kv(donor, 0)[0].shape[1] != n_heads:
        raise ValueError("recipient and donor caches differ in layers, length, or KV heads")
    if not 0 <= a < b <= T:
        raise ValueError(f"span {span} invalid for cache length {T}")
    if any(not 0 <= l < n_layers for l in layers):
        raise ValueError(f"layer outside [0, {n_layers})")
    if any(not 0 <= h < n_heads for h in heads):
        raise ValueError(f"head outside [0, {n_heads})")
    if any(k not in _KINDS for k in kinds):
        raise ValueError(f"kinds must be subset of {_KINDS}")

    out = copy.deepcopy(recipient)
    for l in layers:
        ok, ov = _kv(out, l)
        dk, dv = _kv(donor, l)
        if "k" in kinds:
            ok[:, heads, a:b] = dk[:, heads, a:b].to(ok.device)
        if "v" in kinds:
            ov[:, heads, a:b] = dv[:, heads, a:b].to(ov.device)
    return out


@torch.no_grad()
def _suffix_logits(model, cache, ids: list[int]) -> torch.Tensor:
    c = copy.deepcopy(cache)
    return model(input_ids=torch.tensor([ids], device=model.device), past_key_values=c, use_cache=True).logits[0]


@torch.no_grad()
def readout(model, cache, suffix_ids: list[int]) -> torch.Tensor:
    return _suffix_logits(model, cache, suffix_ids)[-1].float().cpu()


@torch.no_grad()
def continuation_logprob(model, cache, suffix_ids: list[int], cont_ids: list[int]) -> float:
    logits = _suffix_logits(model, cache, suffix_ids + cont_ids[:-1])[len(suffix_ids) - 1:].float().cpu()
    lp = torch.log_softmax(logits, dim=-1)
    return lp[torch.arange(len(cont_ids)), torch.tensor(cont_ids)].sum().item()


@torch.no_grad()
def greedy_continuation(model, cache, suffix_ids: list[int], n: int = CONT_LEN) -> list[int]:
    c = copy.deepcopy(cache)
    ids, out = suffix_ids, []
    for _ in range(n):
        logits = model(input_ids=torch.tensor([ids], device=model.device), past_key_values=c, use_cache=True).logits[0, -1]
        nxt = int(logits.argmax())
        out.append(nxt)
        ids = [nxt]
    return out


def recovery(donor: torch.Tensor, recipient: torch.Tensor, patched: torch.Tensor) -> float:
    d, r, p = (torch.log_softmax(x.double(), dim=-1) for x in (donor, recipient, patched))
    kl = lambda q: (d.exp() * (d - q)).sum().item()
    den = kl(r)
    if den == 0:
        return float("nan")
    return 1 - kl(p) / den


def _configs(n_layers: int, n_heads: int, heads: bool = True):
    allL = range(n_layers)
    yield "all_kv", allL, None, ("k", "v")
    yield "all_k", allL, None, ("k",)
    yield "all_v", allL, None, ("v",)
    for l in allL:
        yield f"layer_{l}", [l], None, ("k", "v")
    for l in allL:
        yield f"cum_fwd_{l}", range(l + 1), None, ("k", "v")
    for l in allL:
        yield f"cum_bwd_{l}", range(l, n_layers), None, ("k", "v")
    for l in allL if heads else ():
        for h in range(n_heads):
            yield f"head_{l}_{h}", [l], [h], ("k", "v")


def run(model, tok, out_dir: Path, limit: int | None, region: str = "body", heads: bool = True):
    cfg = json.loads((DATA / "exp1_prefixes.json").read_text())
    every = [json.loads(l) for l in (DATA / "exp1_passages.jsonl").read_text().splitlines() if l.strip()]
    passages = every[:limit]
    suffix = tok(READOUT_SUFFIX, add_special_tokens=False)["input_ids"]
    if region not in REGIONS:
        raise ValueError(f"region must be one of {REGIONS}")
    prompts, spans = {}, {}
    for p in every:
        prefixes = {c: s.format(**p) for c, s in cfg["prefixes"].items()}
        pids = exp1_prefix_ids(tok, prefixes, cfg["fillers"], cfg["pad_text"])
        encs = {c: exp1_encode(tok, pids[c], p["text"]) for c in (*DONORS, "F1")}
        assert_aligned(encs)
        spans[p["id"]] = encs["F1"].spans["body"]
        assert all(e.spans["body"] == spans[p["id"]] for e in encs.values())
        prompts[p["id"]] = {c: e.ids[:spans[p["id"]][1]] for c, e in encs.items()}

    n_layers, n_heads = model.config.num_hidden_layers, model.config.num_key_value_heads
    out_dir.mkdir(parents=True, exist_ok=True)
    with open(out_dir / "results.jsonl", "w") as f:
        for pi, p in enumerate(passages):
            pid, body = p["id"], spans[p["id"]]
            span = body if region == "body" else (0, body[0])
            caches = {c: prefill(model, ids) for c, ids in prompts[pid].items()}
            logits = {c: readout(model, k, suffix) for c, k in caches.items()}
            conts = {c: greedy_continuation(model, k, suffix) for c, k in caches.items()}

            def emit(donor, recipient, config, target, mixed):
                rec_lp = continuation_logprob(model, caches[recipient], suffix, conts[target])
                row = dict(
                    passage=pid, donor=donor, recipient=recipient, config=config,
                    recovery=recovery(logits[target], logits[recipient], readout(model, mixed, suffix)),
                    cont_logprob_delta=continuation_logprob(model, mixed, suffix, conts[target]) - rec_lp,
                )
                f.write(json.dumps(row) + "\n")
                f.flush()

            for donor in DONORS:
                for name, ls, hs, ks in _configs(n_layers, n_heads, heads):
                    emit(donor, "F1", name, donor, mix_cache(caches["F1"], caches[donor], span, ls, hs, ks))
                print(f"patch {pid} donor={donor} t={time.time() - t0:.0f}s", flush=True)

            emit("F1", "P1", "reverse_all_kv", "F1", mix_cache(caches["P1"], caches["F1"], span, range(n_layers)))

            if region == "prefix":
                for donor in DONORS:
                    emit(donor, "F1", "full_all_kv", donor, mix_cache(caches["F1"], caches[donor], (0, body[1]), range(n_layers)))
            elif len(every) > 1:
                other = every[(pi + 1) % len(every)]["id"]
                cut = min(span[1], spans[other][1])
                donor_ids = prompts[other]["P1"][:cut] + prompts[pid]["P1"][cut:]
                mixed = mix_cache(caches["F1"], prefill(model, donor_ids), (span[0], cut), range(n_layers))
                emit("P1_shuffled", "F1", "shuffled_all_kv", "P1", mixed)


t0 = time.time()


def main():
    global t0
    ap = argparse.ArgumentParser()
    ap.add_argument("--model", default="0.6B", choices=list(M.MODELS))
    ap.add_argument("--limit", type=int, default=None)
    ap.add_argument("--out", type=Path, default=Path("runs"))
    ap.add_argument("--region", default="body", choices=REGIONS)
    ap.add_argument("--no-heads", action="store_true")
    a = ap.parse_args()

    t0 = time.time()
    model, tok = M.load(a.model)
    run_dir = a.out / f"{time.strftime('%Y%m%d-%H%M%S')}-{a.model}"
    run(model, tok, run_dir / ("exp1_patch" if a.region == "body" else f"exp1_patch_{a.region}"), a.limit, a.region, not a.no_heads)
    print(f"elapsed {(time.time() - t0) / 60:.1f} min")
    print(run_dir)


if __name__ == "__main__":
    main()
