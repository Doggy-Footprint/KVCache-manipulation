import argparse
import gc
import json
import sys
import time
from pathlib import Path

import torch

sys.path.insert(0, str(Path(__file__).resolve().parents[1] / "src"))
from kvmeaning import inject as I  # noqa: E402
from kvmeaning import kv_surgery as S  # noqa: E402
from kvmeaning import model as M  # noqa: E402
from kvmeaning import patching as P  # noqa: E402

MAIN = Path(__file__).resolve().parents[1] / "data/inject/main.json"
TAIL_M = (5, 20)


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--model", default="Q3-4B")
    ap.add_argument("--out", type=Path, default=Path("runs/inject/surgery.jsonl"))
    ap.add_argument("--max-new", type=int, default=160)
    ap.add_argument("--limit", type=int, default=None)
    a = ap.parse_args()
    a.out.parent.mkdir(parents=True, exist_ok=True)
    model, tok = M.load(a.model)
    data = I.load(MAIN)
    stop = [tok.convert_tokens_to_ids("<|im_end|>"), tok.eos_token_id]
    jobs = [(t, kn, j, inj) for kn, k in data["kinds"].items() for j, inj in enumerate(k["inj"]) for t in data["tasks"]]
    jobs = jobs[: a.limit] if a.limit else jobs
    done = {tuple(json.loads(l)["key"]) for l in open(a.out)} if a.out.exists() else set()
    dev = next(model.parameters()).device
    t0 = time.time()
    print(f"{len(jobs)} prompts, {len(done)} done", flush=True)
    with open(a.out, "a") as fo:
        for n, (task, kn, j, inj) in enumerate(jobs):
            key = (task["name"], kn, j)
            if key in done:
                continue
            base = data["kinds"][kn]["base"]
            C = I.encode(tok, task, kn, {"base": base, "inj": inj}, True)
            K = I.encode(tok, task, kn, {"base": base, "inj": inj}, False)
            N = I.encode(tok, task, kn, {"base": base, "inj": data["neutral"]}, True)
            assert C.ids[:C.inj[0]] + C.ids[C.inj[1]:] == K.ids == N.ids[:N.inj[0]] + N.ids[N.inj[1]:]
            cC, cK, cN = (P.prefill(model, p.ids[:-1]) for p in (C, K, N))
            closed = S.close_gap(model, cC, C.inj)
            rows_a = ["contaminated", "masked"]
            out_a = S.greedy_batch(model, [cC, cC], C.ids[-1:], a.max_new, masked=[[], [C.inj]], stop_ids=stop)
            rows_b = ["clean_prefill", "cut_rerope"] + [f"cut_rerope_recompute{m}" for m in TAIL_M] + ["neutral_cut_rerope"]
            caches_b = [cK, closed] + [S.recompute_tail(model, closed, K.ids, m) for m in TAIL_M] + [S.close_gap(model, cN, N.inj)]
            out_b = S.greedy_batch(model, caches_b, K.ids[-1:], a.max_new, stop_ids=stop)
            out_n = S.greedy_batch(model, [cN], N.ids[-1:], a.max_new, stop_ids=stop)
            texts = dict(zip(rows_a + rows_b + ["neutral_present"], [tok.decode(x) for x in out_a + out_b + out_n]))
            fo.write(json.dumps(dict(key=list(key), kind=kn, inj=inj, n_tokens=len(C.ids), inj_span=C.inj,
                                     tail_tokens=len(C.ids) - C.inj[1],
                                     score={k: I.measure(kn, v) for k, v in texts.items()}, text=texts)) + "\n")
            fo.flush()
            del cC, cK, cN, closed, caches_b
            gc.collect()
            if dev.type == "mps":
                torch.mps.empty_cache()
            print(f"[{n + 1}/{len(jobs)}] {key} {time.time() - t0:.0f}s "
                  + " ".join(f"{k}={I.measure(kn, v)}" for k, v in texts.items()), flush=True)
    print(a.out)


if __name__ == "__main__":
    main()
