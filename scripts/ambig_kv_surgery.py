import argparse
import gc
import json
import sys
import time
from pathlib import Path

import torch

sys.path.insert(0, str(Path(__file__).resolve().parents[1] / "src"))
from kvmeaning import ambig as A  # noqa: E402
from kvmeaning import kv_surgery as S  # noqa: E402
from kvmeaning import model as M  # noqa: E402
from kvmeaning import patching as P  # noqa: E402

SRC = ("A", "B", "U")
BANDS = {"L0-11": range(0, 12), "L12-23": range(12, 24), "L24-35": range(24, 36)}
TOPK = (1, 2, 4, 8)


def cells(c, e, hdr, body, block, n_layers, dev_rank, rand4):
    out = []
    for E in SRC + ("mask",):
        for D in SRC:
            cache = c[D] if E in (D, "mask") else S.splice(c[D], c[E], e)
            out.append((f"F:{E}|{D}", cache, [e] if E == "mask" else []))
    for E in ("mask", "U"):
        for D in ("A", "B"):
            for part, rows in (("hdr", hdr), ("body", body)):
                out.append((f"H:{E}|{D}:{part}", S.splice(c["U"], c[D], rows), [e] if E == "mask" else []))
    for D in ("A", "B"):
        for band, layers in BANDS.items():
            out.append((f"L:mask|{D}:{band}", S.splice(c["U"], c[D], block, layers), [e]))
    for D in ("A", "B"):
        for k in TOPK:
            out.append((f"R:mask|{D}:top{k}", S.splice(c[D], c["U"], dev_rank[D][:k]), [e]))
        out.append((f"R:mask|{D}:rand4", S.splice(c[D], c["U"], rand4[D]), [e]))
    return out


def deviation_rank(c, D, block):
    a, b = block
    d = torch.stack([(l.values[0, :, a:b] - u.values[0, :, a:b]).float().norm(dim=-1)
                     / u.values[0, :, a:b].float().norm(dim=-1).clamp_min(1e-6)
                     for l, u in zip(c[D].layers, c["U"].layers)]).mean((0, 1))
    return [a + int(i) for i in d.argsort(descending=True)], d


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--model", default="Q3-4B")
    ap.add_argument("--out", type=Path, default=Path("runs/ambig/surgery/cells.jsonl"))
    ap.add_argument("--only", nargs="*", default=None)
    ap.add_argument("--batch", type=int, default=12)
    a = ap.parse_args()
    a.out.parent.mkdir(parents=True, exist_ok=True)
    model, tok = M.load(a.model)
    ids_ab = A.answer_ids(tok)
    items = [it for it in A.load_items() if not a.only or it["id"] in a.only]
    prompts = A.build(tok, items, para=False, closed=False, neutral=False, content=False)
    groups = {}
    for p in prompts:
        groups.setdefault((p.item, p.cue or "N", p.order), {})[p.cond] = p
    done = {tuple(json.loads(l)["key"]) for l in open(a.out)} if a.out.exists() else set()
    dev = next(model.parameters()).device
    n_layers = model.config.num_hidden_layers
    t0 = time.time()
    print(f"{len(groups)} groups, {len(done)} done", flush=True)
    with open(a.out, "a") as fo:
        for n, (key, g) in enumerate(groups.items()):
            if key in done:
                continue
            pa = g["A_pre"]
            sp = pa.enc.spans
            e, q0 = tuple(sp["expl"]), sp["doc"][1]
            hdr, body, block = (e[1], sp["doc"][0]), (sp["doc"][0], q0), (e[1], q0)
            for c_ in ("B_pre", "U_pre"):
                o = g[c_].enc.ids
                assert len(o) == len(pa.enc.ids) and o[:e[0]] == pa.enc.ids[:e[0]] and o[e[1]:] == pa.enc.ids[e[1]:]
            inv = {v: k for k, v in pa.letters.items()}
            lp = lambda logits: {r: float(logits.log_softmax(-1)[ids_ab[inv[r]]]) for r in "AB"}
            c = {s: P.prefill(model, g[f"{s}_pre"].enc.ids[:q0]) for s in SRC}
            suffix = pa.enc.ids[q0:]
            rank, devs = {}, {}
            gen = torch.Generator().manual_seed(hash(key) % (2 ** 31))
            rand4 = {}
            for D in ("A", "B"):
                rank[D], devs[D] = deviation_rank(c, D, block)
                rand4[D] = [block[0] + int(i) for i in torch.randperm(block[1] - block[0], generator=gen)[:4]]
            cl = cells(c, e, hdr, body, block, n_layers, rank, rand4)
            res = {}
            for j in range(0, len(cl), a.batch):
                chunk = cl[j:j + a.batch]
                logits = S.readout_batch(model, [x[1] for x in chunk], suffix, masked=[x[2] for x in chunk])
                res.update({name: lp(logits[i]) for i, (name, _, _) in enumerate(chunk)})
            gap = [S.close_gap(model, c[D], e) for D in SRC]
            gl = S.readout_batch(model, gap, suffix)
            res.update({f"G:closed|{D}": lp(gl[i]) for i, D in enumerate(SRC)})
            with torch.no_grad():
                res["none"] = lp(model(torch.tensor([g["none"].enc.ids], device=dev)).logits[0, -1].float())
            fo.write(json.dumps(dict(
                key=list(key), type=pa.type, gold=pa.gold, letters=pa.letters, cells=res,
                top_tokens={D: [tok.decode([pa.enc.ids[i]]) for i in rank[D][:8]] for D in ("A", "B")},
                top_dev={D: [round(float(devs[D][i - block[0]]), 4) for i in rank[D][:8]] for D in ("A", "B")},
                rand4={D: [tok.decode([pa.enc.ids[i]]) for i in rand4[D]] for D in ("A", "B")})) + "\n")
            fo.flush()
            del c, cl, gap, logits, gl
            gc.collect()
            if dev.type == "mps":
                # The MPS allocator keeps freed cache blocks; without releasing them each group got slower (16 s to 150 s).
                torch.mps.empty_cache()
            if n % 20 == 0 or a.only:
                print(f"[{n + 1}/{len(groups)}] {key} {time.time() - t0:.0f}s", flush=True)
    print(a.out)


if __name__ == "__main__":
    main()
