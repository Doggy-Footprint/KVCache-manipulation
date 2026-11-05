"""Build data/exp1_passages_bbc.jsonl from the BBC News classification set (Greene & Cunningham 2006).

Articles are BBC copyright with no explicit license; the derived file stays in this local-only repo.
"""

import argparse
import json
import random
import re
import urllib.request
import zipfile
from pathlib import Path

from transformers import AutoTokenizer

from kvmeaning.model import MODELS

ROOT = Path(__file__).resolve().parents[1]
URL = "http://mlg.ucd.ie/files/datasets/bbc-fulltext.zip"
CATS = ["business", "entertainment", "politics", "sport", "tech"]
LABELS = {"business": "business and the economy", "entertainment": "entertainment", "politics": "politics",
          "sport": "sport", "tech": "technology"}
MIN_TOK, MAX_TOK = 225, 264
SENT = re.compile(r'(?<=[.!?])["\'”’]?\s+(?=["\'“‘]?[A-Z0-9])')


def articles(zip_path: Path):
    with zipfile.ZipFile(zip_path) as z:
        for name in sorted(z.namelist()):
            m = re.fullmatch(r"bbc/(\w+)/(\d+)\.txt", name)
            if m and m.group(1) in CATS:
                title, _, body = z.read(name).decode("latin-1").partition("\n")
                yield m.group(1), m.group(2), title.strip(), re.sub(r"\s+", " ", body).strip()


def truncate(n_tok, body: str) -> str:
    acc = ""
    for s in SENT.split(body):
        c = f"{acc} {s}".strip()
        if n_tok(c) > MAX_TOK:
            break
        acc = c
    return acc


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--zip", type=Path, default=ROOT / "data/raw/bbc-fulltext.zip")
    ap.add_argument("--per-category", type=int, default=20)
    ap.add_argument("--seed", type=int, default=0)
    ap.add_argument("--out", type=Path, default=ROOT / "data/exp1_passages_bbc.jsonl")
    a = ap.parse_args()

    if not a.zip.exists():
        a.zip.parent.mkdir(parents=True, exist_ok=True)
        urllib.request.urlretrieve(URL, a.zip)
    spec = MODELS["Q3-4B"]
    tok = AutoTokenizer.from_pretrained(spec.repo, revision=spec.revision, local_files_only=True)
    n_tok = lambda s: len(tok(s, add_special_tokens=False)["input_ids"])

    seen: set[str] = set()
    pool = {c: [] for c in CATS}
    for cat, num, title, body in articles(a.zip):
        # The set contains ~100 duplicated articles and reused headlines; a reused headline would make H- equal another item's H+.
        if body in seen or title in seen:
            continue
        seen |= {body, title}
        text = truncate(n_tok, body)
        k = n_tok(text) if text else 0
        if MIN_TOK <= k <= MAX_TOK:
            pool[cat].append(dict(id=f"{cat[:3]}{num}", category=cat, headline=title, text=text, n_tokens=k))

    rng = random.Random(a.seed)
    picked = {}
    for c in CATS:
        if len(pool[c]) < a.per_category:
            raise SystemExit(f"{c}: only {len(pool[c])} fitting articles")
        picked[c] = rng.sample(pool[c], a.per_category)

    rows = []
    for i, c in enumerate(CATS):
        # H- pairs item k with item k of the next category, so the wrong headline/label is human-written and off-category.
        w = CATS[(i + 1) % len(CATS)]
        for it, wrong in zip(picked[c], picked[w]):
            rows.append(dict(id=it["id"], category=c, label=LABELS[c], headline=it["headline"], wrong_id=wrong["id"],
                             wrong_category=w, wrong_label=LABELS[w], wrong_headline=wrong["headline"],
                             n_tokens=it["n_tokens"], text=it["text"]))
    a.out.write_text("".join(json.dumps(r, ensure_ascii=False) + "\n" for r in rows))
    print(f"{len(rows)} passages -> {a.out}; pool sizes {({c: len(v) for c, v in pool.items()})}")


if __name__ == "__main__":
    main()
