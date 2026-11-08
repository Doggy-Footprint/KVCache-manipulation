import argparse
import collections
import json
import sys
import time
from pathlib import Path

import torch

sys.path.insert(0, str(Path(__file__).resolve().parents[1] / "src"))
from kvmeaning import inject as I  # noqa: E402
from kvmeaning import model as M  # noqa: E402


@torch.no_grad()
def generate(model, tok, prompts, max_new):
    L = max(len(p.ids) for p in prompts)
    pad = tok.pad_token_id if tok.pad_token_id is not None else tok.eos_token_id
    ids = torch.tensor([[pad] * (L - len(p.ids)) + p.ids for p in prompts], device=model.device)
    am = torch.tensor([[0] * (L - len(p.ids)) + [1] * len(p.ids) for p in prompts], device=model.device)
    out = model.generate(input_ids=ids, attention_mask=am, max_new_tokens=max_new, do_sample=False,
                         temperature=None, top_p=None, top_k=None, pad_token_id=pad)
    return [tok.decode(o[L:], skip_special_tokens=True) for o in out]


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--model", default="Q3-4B")
    ap.add_argument("--out", type=Path, default=Path("runs/inject/screen.jsonl"))
    ap.add_argument("--max-new", type=int, default=200)
    ap.add_argument("--kinds", nargs="*", default=None)
    a = ap.parse_args()
    a.out.parent.mkdir(parents=True, exist_ok=True)
    model, tok = M.load(a.model)
    data = I.load()
    done = {(json.loads(l)["kind"]) for l in open(a.out)} if a.out.exists() else set()
    t0 = time.time()
    with open(a.out, "a") as fo:
        for kname, kind in data["kinds"].items():
            if (a.kinds and kname not in a.kinds) or kname in done:
                continue
            ps = [I.encode(tok, t, kname, kind, c) for c in (False, True) for t in data["tasks"]]
            texts = generate(model, tok, ps, a.max_new)
            for p, txt in zip(ps, texts):
                fo.write(json.dumps(dict(kind=kname, task=p.task, contaminated=p.contaminated, n_tokens=len(p.ids),
                                         inj=p.inj, score=I.measure(kname, txt), text=txt)) + "\n")
            fo.flush()
            print(f"{kname} {time.time() - t0:.0f}s", flush=True)
    rows = [json.loads(l) for l in open(a.out)]
    agg = collections.defaultdict(lambda: collections.defaultdict(list))
    for r in rows:
        agg[r["kind"]][r["contaminated"]].append(r["score"])
    for k, v in agg.items():
        f = lambda xs: f"{sum(x > 0 for x in xs)}/{len(xs)} (mean {sum(xs) / len(xs):.1f})"
        print(f"  {k:11s} clean {f(v[False])}  contaminated {f(v[True])}")


if __name__ == "__main__":
    main()
