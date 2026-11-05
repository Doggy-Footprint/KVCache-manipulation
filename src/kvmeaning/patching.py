import argparse
import copy
import json
import random
import time
from collections.abc import Iterable
from pathlib import Path

import torch
from transformers import DynamicCache

from . import model as M
from .batch import stack_caches
from .progress import Progress
from .spans import assert_aligned, exp1_encode, exp1_prefix_ids

DATA = Path(__file__).resolve().parents[2] / "data"
DESIGN_FILES = {"v1": "exp1_prefixes.json", "v2": "exp1_prefixes_v2.json"}
PATCH_MODES = ("kv", "state", "both")
BATCH_SIZE = 16
READOUT_SUFFIX = "<|im_end|>\n<|im_start|>assistant\n<think>\n\n</think>\n\n"
DONORS = ("H+", "H-", "P1", "P2", "F2")
CONT_LEN = 16
_KINDS = ("k", "v")
# "prefix" patches [0, body start): the template head is identical across conditions, the delimiter is prefix-dependent.
REGIONS = ("body", "prefix")
SHUFFLE_K = 3
SHUFFLE_SEED = 0


def shuffle_sources(pid: str, pool: list[str], k: int = SHUFFLE_K, seed: int = SHUFFLE_SEED) -> list[str]:
    others = [q for q in pool if q != pid]
    return random.Random(f"{seed}:{pid}").sample(others, min(k, len(others))) if len(pool) >= 2 else []


def _kv(cache, i: int):
    return cache.layers[i].keys, cache.layers[i].values


def _num_layers(cache) -> int:
    return len(cache.layers)


def _is_linear(cache, i: int) -> bool:
    return hasattr(cache.layers[i], "recurrent_states")


def _state_tensors(layer):
    return [(d, s) for d in (layer.conv_states, layer.recurrent_states) for s in d if d[s] is not None]


def _to_device(rows, device) -> torch.Tensor:
    # A pageable host-to-device copy waits for queued GPU work; pinned + non_blocking lets the host run ahead.
    t = torch.tensor(rows)
    if torch.device(device).type != "cuda":
        return t.to(device)
    return t.pin_memory().to(device, non_blocking=True)


def _head_index(heads: list[int]):
    # A list index is copied to the GPU synchronously on every use; a slice needs no copy.
    if heads == list(range(heads[0], heads[-1] + 1)):
        return slice(heads[0], heads[-1] + 1)
    return heads


@torch.no_grad()
def prefill(model, ids: list[int]) -> DynamicCache:
    out = model(input_ids=_to_device([ids], model.device), use_cache=True)
    return out.past_key_values


@torch.no_grad()
def mix_cache(recipient, donor, span: tuple[int, int], layers: Iterable[int], heads: Iterable[int] | None = None,
              kinds: Iterable[str] = ("k", "v"), state: bool = False) -> DynamicCache:
    """K/V rows are swapped on full-attention layers only; state=True additionally swaps every linear layer's
    whole conv+recurrent state (cumulative, so no span applies)."""
    layers, kinds = list(layers), list(kinds)
    n_layers = _num_layers(recipient)
    linear = [i for i in range(n_layers) if _is_linear(recipient, i)]
    kv_i = next((i for i in range(n_layers) if i not in linear), None)
    if _num_layers(donor) != n_layers or [i for i in range(n_layers) if _is_linear(donor, i)] != linear:
        raise ValueError("recipient and donor caches differ in layers or layer types")
    if state and not linear:
        raise ValueError("state patching requires a hybrid model with linear-attention layers")
    T = n_heads = None
    if kv_i is not None:
        k0 = _kv(recipient, kv_i)[0]
        T, n_heads = k0.shape[2], k0.shape[1]
        if _kv(donor, kv_i)[0].shape[2] != T or _kv(donor, kv_i)[0].shape[1] != n_heads:
            raise ValueError("recipient and donor caches differ in length or KV heads")
    for i in linear:
        r, d = _state_tensors(recipient.layers[i]), _state_tensors(donor.layers[i])
        if [x[0][x[1]].shape for x in r] != [x[0][x[1]].shape for x in d]:
            raise ValueError(f"layer {i}: recipient and donor linear states differ in shape")
    if any(not 0 <= l < n_layers for l in layers):
        raise ValueError(f"layer outside [0, {n_layers})")
    if any(k not in _KINDS for k in kinds):
        raise ValueError(f"kinds must be subset of {_KINDS}")
    kv_layers = [l for l in layers if l not in linear]
    if kv_layers or not state:
        heads = list(range(n_heads)) if heads is None else list(heads)
        a, b = span
        if not 0 <= a < b <= T:
            raise ValueError(f"span {span} invalid for cache length {T}")
        if any(not 0 <= h < n_heads for h in heads):
            raise ValueError(f"head outside [0, {n_heads})")

    out = copy.deepcopy(recipient)
    if kv_layers:
        heads = _head_index(heads)
    for l in kv_layers:
        ok, ov = _kv(out, l)
        dk, dv = _kv(donor, l)
        if "k" in kinds:
            ok[:, heads, a:b] = dk[:, heads, a:b].to(ok.device)
        if "v" in kinds:
            ov[:, heads, a:b] = dv[:, heads, a:b].to(ov.device)
    if state:
        for i in linear:
            for (od, s), (dd, _) in zip(_state_tensors(out.layers[i]), _state_tensors(donor.layers[i])):
                od[s] = dd[s].detach().clone().to(od[s].device)
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


@torch.no_grad()
def eval_batch(model, caches: list, suffix_ids: list[int], conts: list[list[int]]) -> tuple[torch.Tensor, torch.Tensor]:
    stacked = stack_caches(caches)
    ids = _to_device([suffix_ids + c[:-1] for c in conts], model.device)
    logits = model(input_ids=ids, past_key_values=stacked, use_cache=True).logits[:, len(suffix_ids) - 1:].float()
    lp = torch.log_softmax(logits, dim=-1)
    gathered = lp.gather(-1, _to_device(conts, lp.device)[..., None])[..., 0].sum(-1)
    return logits[:, 0], gathered


@torch.no_grad()
def greedy_batch(model, caches: list, suffix_ids: list[int], n: int = CONT_LEN) -> tuple[torch.Tensor, list[list[int]]]:
    c = stack_caches(caches)
    ids = _to_device([suffix_ids] * len(caches), model.device)
    outs, first = [], None
    for _ in range(n):
        logits = model(input_ids=ids, past_key_values=c, use_cache=True).logits[:, -1]
        if first is None:
            first = logits.float()
        ids = logits.argmax(-1, keepdim=True)
        outs.append(ids[:, 0])
    return first, torch.stack(outs, 1).tolist()


def recovery_batch(donor: torch.Tensor, recipient: torch.Tensor, patched: torch.Tensor) -> torch.Tensor:
    d, r, p = (torch.log_softmax(x.double(), dim=-1) for x in (donor, recipient, patched))
    pd = d.exp()
    den, num = (pd * (d - r)).sum(-1), (pd * (d - p)).sum(-1)
    return torch.where(den == 0, torch.full_like(den, float("nan")), 1 - num / den)


def recovery(donor: torch.Tensor, recipient: torch.Tensor, patched: torch.Tensor) -> float:
    return recovery_batch(donor[None], recipient[None], patched[None])[0].item()


def _configs(n_layers: int, n_heads: int, heads: bool = True, layers: list[int] | None = None):
    allL = list(range(n_layers)) if layers is None else list(layers)
    yield "all_kv", allL, None, ("k", "v")
    yield "all_k", allL, None, ("k",)
    yield "all_v", allL, None, ("v",)
    for l in allL:
        yield f"layer_{l}", [l], None, ("k", "v")
    for l in allL:
        yield f"cum_fwd_{l}", [x for x in allL if x <= l], None, ("k", "v")
    for l in allL:
        yield f"cum_bwd_{l}", [x for x in allL if x >= l], None, ("k", "v")
    for l in allL if heads else ():
        for h in range(n_heads):
            yield f"head_{l}_{h}", [l], [h], ("k", "v")


def _variants(name, ls, hs, ks, hybrid: bool, modes, singleton: bool):
    """Yield (config name, layers, heads, kinds, state). State swaps are span/layer independent, so only singleton
    configs are expanded over the state and both modes; sweeps run in kv mode."""
    if not hybrid:
        yield name, ls, hs, ks, False
        return
    for m in modes:
        if m == "kv":
            yield f"{name}@kv", ls, hs, ks, False
        elif singleton:
            yield f"{name}@{m}", (ls if m == "both" else []), hs, ks, True


def _chunks(xs: list, n: int):
    for i in range(0, len(xs), n):
        yield xs[i:i + n]


def run(model, tok, out_dir: Path, limit: int | None, region: str = "body", heads: bool = True,
        design: str = "v1", batch_size: int = BATCH_SIZE, modes: Iterable[str] | None = None, on_passage=None):
    cfg = json.loads((DATA / DESIGN_FILES[design]).read_text())
    every = [json.loads(l) for l in (DATA / "exp1_passages.jsonl").read_text().splitlines() if l.strip()]
    passages = every[:limit]
    suffix = tok(READOUT_SUFFIX, add_special_tokens=False)["input_ids"]
    if region not in REGIONS:
        raise ValueError(f"region must be one of {REGIONS}")
    kvl = M.kv_layers(model)
    tcfg = model.config.get_text_config()
    n_layers, n_heads = tcfg.num_hidden_layers, tcfg.num_key_value_heads
    hybrid = len(kvl) < n_layers
    modes = tuple(modes) if modes is not None else (PATCH_MODES if hybrid else ("kv",))
    if any(m not in PATCH_MODES for m in modes):
        raise ValueError(f"modes must be subset of {PATCH_MODES}")
    if not hybrid and set(modes) - {"kv"}:
        raise ValueError("state/both patching requires a hybrid model")
    prompts, spans = {}, {}
    for p in every:
        prefixes = {c: s.format(**p) for c, s in cfg["prefixes"].items()}
        pids = exp1_prefix_ids(tok, prefixes, cfg["fillers"], cfg["pad_text"], design)
        encs = {c: exp1_encode(tok, pids[c], p["text"]) for c in (*DONORS, "F1")}
        assert_aligned(encs)
        spans[p["id"]] = encs["F1"].spans["body"]
        assert all(e.spans["body"] == spans[p["id"]] for e in encs.values())
        prompts[p["id"]] = {c: e.ids[:spans[p["id"]][1]] for c, e in encs.items()}

    out_dir.mkdir(parents=True, exist_ok=True)
    results, done_file = out_dir / "results.jsonl", out_dir / "done.txt"
    done = set(done_file.read_text().split()) if done_file.exists() else set()
    if results.exists():
        keep = [l for l in results.read_text().splitlines() if l.strip() and json.loads(l)["passage"] in done]
        results.write_text("".join(l + "\n" for l in keep))
    prog = Progress(f"patch {design}/{region} B={batch_size}", len(passages))
    with open(results, "a") as f:
        for pi, p in enumerate(passages):
            pid, body = p["id"], spans[p["id"]]
            if pid in done:
                prog.step(pid, skipped=True)
                continue
            span = body if region == "body" else (0, body[0])
            names = list(prompts[pid])
            caches = {c: prefill(model, prompts[pid][c]) for c in names}
            first, cont_list = greedy_batch(model, [caches[c] for c in names], suffix)
            logits, conts = dict(zip(names, first)), dict(zip(names, cont_list))

            shuffled = {}
            tasks = []  # (donor, recipient, config, target, make_mixed, extra_fields)

            def add(donor, recipient, target, rspan, name, ls, hs=None, ks=("k", "v"), singleton=True, donor_cache=None, only=None, extra=None):
                use = [m for m in modes if only is None or m in only]
                for cname, l_, h_, k_, st in _variants(name, ls, hs, ks, hybrid, use, singleton):
                    make = lambda l_=l_, h_=h_, k_=k_, st=st: mix_cache(
                        caches[recipient], donor_cache() if donor_cache else caches[donor], rspan, l_, h_, k_, state=st)
                    tasks.append((donor, recipient, cname, target, make, extra or {}))

            for donor in DONORS:
                for name, ls, hs, ks in _configs(n_layers, n_heads, heads, kvl):
                    add(donor, "F1", donor, span, name, ls, hs, ks, singleton=name == "all_kv")
            add("F1", "P1", "F1", span, "reverse_all_kv", kvl)

            if region == "prefix":
                for donor in DONORS:
                    add(donor, "F1", donor, (0, body[1]), "full_all_kv", kvl)
            else:
                if hybrid:
                    for donor in DONORS:
                        add(donor, "F1", donor, (0, body[1]), "full_all_kv", kvl, only=("both",))
                # Pool is the whole passage file, not `passages`, so sources do not change with `limit`.
                for q in shuffle_sources(pid, [e["id"] for e in every], SHUFFLE_K, SHUFFLE_SEED):
                    cut = min(span[1], spans[q][1])
                    donor_ids = prompts[q]["P1"][:cut] + prompts[pid]["P1"][cut:]
                    add("P1_shuffled", "F1", "P1", (span[0], cut), "shuffled_all_kv", kvl,
                        donor_cache=lambda q=q, ids=donor_ids: shuffled[q] if q in shuffled else shuffled.setdefault(q, prefill(model, ids)),
                        extra={"shuffle_source": q})

            pairs = sorted({(t[1], t[3]) for t in tasks})
            base_parts = [eval_batch(model, [caches[r] for r, _ in chunk], suffix, [conts[t] for _, t in chunk])[1]
                          for chunk in _chunks(pairs, batch_size)]
            rec_parts, lp_parts = [], []
            for chunk in _chunks(tasks, batch_size):
                read, lps = eval_batch(model, [t[4]() for t in chunk], suffix, [conts[t[3]] for t in chunk])
                rec_parts.append(recovery_batch(torch.stack([logits[t[3]] for t in chunk]),
                                                torch.stack([logits[t[1]] for t in chunk]), read))
                lp_parts.append(lps)
            # Results are read back once per passage so the host keeps queueing chunks while the GPU computes.
            base_lp = dict(zip(pairs, torch.cat(base_parts).tolist()))
            recs, lps = torch.cat(rec_parts).tolist(), torch.cat(lp_parts).tolist()

            rows = []
            for (donor, recipient, cname, target, _, extra), rec, lp in zip(tasks, recs, lps):
                rows.append(dict(
                    passage=pid, donor=donor, recipient=recipient, config=cname,
                    recovery=rec,
                    cont_logprob_delta=lp - base_lp[(recipient, target)], **extra,
                ))
            f.write("".join(json.dumps(r) + "\n" for r in rows))
            f.flush()
            with open(done_file, "a") as df:
                df.write(pid + "\n")
            prog.step(pid, f"rows={len(rows)}")
            if on_passage:
                on_passage(pid)


t0 = time.time()


def main():
    global t0
    ap = argparse.ArgumentParser()
    ap.add_argument("--model", default="0.6B", choices=list(M.MODELS))
    ap.add_argument("--limit", type=int, default=None)
    ap.add_argument("--out", type=Path, default=Path("runs"))
    ap.add_argument("--region", default="body", choices=REGIONS)
    ap.add_argument("--no-heads", action="store_true")
    ap.add_argument("--design", default="v1", choices=list(DESIGN_FILES))
    ap.add_argument("--device", default=None)
    ap.add_argument("--batch-size", type=int, default=BATCH_SIZE)
    a = ap.parse_args()

    t0 = time.time()
    model, tok = M.load(a.model, device=a.device)
    run_dir = a.out / f"{time.strftime('%Y%m%d-%H%M%S')}-{a.model}{'-v2' if a.design == 'v2' else ''}"
    run(model, tok, run_dir / ("exp1_patch" if a.region == "body" else f"exp1_patch_{a.region}"), a.limit, a.region,
        not a.no_heads, a.design, a.batch_size)
    print(f"elapsed {(time.time() - t0) / 60:.1f} min")
    print(run_dir)


if __name__ == "__main__":
    main()
