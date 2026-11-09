import argparse
import json
import re
import sys
import time
from pathlib import Path

import torch

sys.path.insert(0, str(Path(__file__).resolve().parents[1] / "src"))
from kvmeaning import ambig as A  # noqa: E402
from kvmeaning import model as M  # noqa: E402

# Generations often say "cannot choose between A or B" before committing, so the last letter-shaped match wins.
LETTER_PATTERNS = (
    re.compile(r"answer\s*(?:is)?\s*[:：]?\s*\**\s*\(?([AB])\b", re.I),
    re.compile(r"(?:^|\n|:)\s*\**([AB])(?:\.|\*\*|\s*$)"),
)


def parse(text: str) -> str | None:
    hits = [(m.start(1), m.group(1)) for pat in LETTER_PATTERNS for m in pat.finditer(text)]
    return max(hits)[1] if hits else None


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--run-dir", type=Path, required=True)
    ap.add_argument("--model", default="Q3-4B")
    ap.add_argument("--max-new-tokens", type=int, default=300)
    ap.add_argument("--out", default="gen_nonletter.jsonl")
    a = ap.parse_args()
    rows = [r for f in sorted((a.run_dir / "items").glob("*.json")) for r in json.loads(f.read_text())["rows"]]
    todo = {(r["item"], r["cond"], r["cue"], r["order"]): r for r in rows if r["top_text"] not in ("A", "B")}
    out = a.run_dir / a.out
    done = {tuple(json.loads(l)["key"]) for l in open(out)} if out.exists() else set()
    model, tok = M.load(a.model)
    items = {it["id"]: it for it in A.load_items()}
    prompts = {(p.item, p.cond, p.cue, p.order): p for p in A.build(tok, list(items.values()))}
    dev = next(model.parameters()).device
    print(f"{len(todo)} prompts, {len(done)} done", flush=True)
    with open(out, "a") as fo:
        for n, (key, r) in enumerate(todo.items()):
            if key in done:
                continue
            p = prompts[key]
            t0 = time.time()
            with torch.no_grad():
                g = model.generate(torch.tensor([p.enc.ids], device=dev), max_new_tokens=a.max_new_tokens,
                                   do_sample=False, temperature=None, top_p=None, top_k=None)
            text = tok.decode(g[0, len(p.enc.ids):], skip_special_tokens=True)
            letter = parse(text)
            forced = max(r["logp"], key=r["logp"].get)
            fo.write(json.dumps(dict(key=list(key), type=r["type"], relation=r["relation"], gold=r["gold"],
                                     top_text=r["top_text"], forced=forced, generated=letter,
                                     text=text)) + "\n")
            fo.flush()
            print(f"[{n + 1}/{len(todo)}] {key} forced={forced} gen={letter} {time.time() - t0:.1f}s "
                  f"{text[:70]!r}", flush=True)
    print(out)


if __name__ == "__main__":
    main()
