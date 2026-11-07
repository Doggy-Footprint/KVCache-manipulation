import argparse
import json
from collections import Counter
from dataclasses import dataclass
from pathlib import Path

from kvmeaning.spans import DELIM, Encoded, _ids

ROOT = Path(__file__).resolve().parents[2]
ITEMS = ROOT / "data/ambig_rag"
SYSTEM = "You are a careful reader. Answer the question about the retrieved document."
SYSTEM_CLOSED = "You are a careful reader. Answer the question."
PRE = "Note on the retrieved document: "
DOC_HEAD = "Retrieved document:\n"
ASK = "Answer with A or B."
ORDERS = ("AB", "BA")
CUES = ("A", "B", None)


@dataclass(frozen=True)
class Cond:
    expl: str | None
    pos: str | None
    para: bool = False


CONDS = {
    "none": Cond(None, None),
    "A_pre": Cond("A", "pre"), "A_post": Cond("A", "post"),
    "B_pre": Cond("B", "pre"), "B_post": Cond("B", "post"),
    "U_pre": Cond("U", "pre"),
}
PARA_CONDS = {
    "pA_pre": Cond("A", "pre", True), "pA_post": Cond("A", "post", True),
    "pB_pre": Cond("B", "pre", True), "pB_post": Cond("B", "post", True),
}


@dataclass
class Prompt:
    item: str
    type: str
    cond: str
    cue: str | None
    order: str
    relation: str
    letters: dict[str, str]
    gold: str | None
    enc: Encoded


def load_items(root: Path = ITEMS) -> list[dict]:
    return [json.loads(l) for f in sorted(root.glob("*.jsonl")) for l in open(f) if l.strip()]


def letters(order: str) -> dict[str, str]:
    return {"A": order[0], "B": order[1]}


def relation(cond: Cond, cue: str | None) -> str:
    if cond.expl is None:
        return "none"
    if cond.expl == "U":
        return "unrelated"
    if cue is None:
        return "cueless"
    return "correct" if cond.expl == cue else "wrong"


def _expl_text(item: dict, cond: Cond) -> str:
    if cond.expl == "U":
        return PRE + item["unrelated"]
    return PRE + item["expl_para" if cond.para else "expl"][cond.expl]


def _qa(tok, item: dict, order: str) -> list[int]:
    opts = "".join(f"{d}. {item['options'][r]}\n" for d, r in letters(order).items())
    return _ids(tok, "Question: " + item["question"] + "\n" + opts + ASK) + \
        _ids(tok, "<|im_end|>\n<|im_start|>assistant\n")


def encode(tok, item: dict, cond: Cond, cue: str | None, order: str) -> Encoded:
    ids = _ids(tok, f"<|im_start|>system\n{SYSTEM}<|im_end|>\n<|im_start|>user\n")
    spans = {}

    def add(name, seg):
        nonlocal ids
        spans[name] = (len(ids), len(ids) + len(seg))
        ids = ids + seg

    if cond.pos == "pre":
        add("expl", _ids(tok, _expl_text(item, cond)))
        ids = ids + _ids(tok, DELIM)
    ids = ids + _ids(tok, DOC_HEAD)
    d0 = len(ids)
    if item["lead"]:
        add("lead", _ids(tok, item["lead"]))
    add("span", _ids(tok, (" " if item["lead"] else "") + item["span"]))
    if cue is not None:
        add("cue", _ids(tok, " " + item["cues"][cue]))
    spans["doc"] = (d0, len(ids))
    if cond.pos == "post":
        ids = ids + _ids(tok, DELIM)
        add("expl", _ids(tok, _expl_text(item, cond)))
    ids = ids + _ids(tok, DELIM)
    q0 = len(ids)
    ids = ids + _qa(tok, item, order)
    spans["question"] = (q0, len(ids))
    spans["answer"] = (len(ids) - 1, len(ids))
    return Encoded(ids, spans)


def encode_closed(tok, item: dict, order: str) -> Encoded:
    ids = _ids(tok, f"<|im_start|>system\n{SYSTEM_CLOSED}<|im_end|>\n<|im_start|>user\n")
    q0 = len(ids)
    ids = ids + _qa(tok, item, order)
    return Encoded(ids, {"question": (q0, len(ids)), "answer": (len(ids) - 1, len(ids))})


def _gold(cue: str | None, order: str) -> str | None:
    if cue is None:
        return None
    return next(d for d, r in letters(order).items() if r == cue)


def build(tok, items: list[dict], para: bool = True, closed: bool = True) -> list[Prompt]:
    out = []
    for it in items:
        conds = dict(CONDS)
        if para and "expl_para" in it:
            conds.update(PARA_CONDS)
        for cue in CUES:
            for order in ORDERS:
                for name, c in conds.items():
                    out.append(Prompt(it["id"], it["type"], name, cue, order, relation(c, cue), letters(order),
                                      _gold(cue, order), encode(tok, it, c, cue, order)))
        if closed:
            for order in ORDERS:
                out.append(Prompt(it["id"], it["type"], "closed", None, order, "closed", letters(order), None,
                                  encode_closed(tok, it, order)))
    return out


def answer_ids(tok) -> dict[str, int]:
    out = {}
    for d in "AB":
        i = _ids(tok, d)
        assert len(i) == 1, f"answer letter {d!r} is {len(i)} tokens"
        out[d] = i[0]
    return out


def check(prompts: list[Prompt]) -> list[str]:
    """Return violations of the alignment the KV comparisons rely on."""
    errs = []
    by = {}
    for p in prompts:
        if p.cond != "closed":
            by.setdefault((p.item, p.cue, p.order), {})[p.cond] = p.enc
    for key, encs in by.items():
        doc = {n: e.ids[slice(*e.spans["doc"])] for n, e in encs.items()}
        if len({tuple(v) for v in doc.values()}) != 1:
            errs.append(f"{key}: document tokens differ across conditions")
        for group in ([n for n in encs if CONDS.get(n, PARA_CONDS.get(n)).pos == "pre"],
                      [n for n in encs if CONDS.get(n, PARA_CONDS.get(n)).pos != "pre"]):
            if len({encs[n].spans["doc"] for n in group}) > 1:
                errs.append(f"{key}: document positions differ within {group}")
        post = [n for n in encs if CONDS.get(n, PARA_CONDS.get(n)).pos != "pre"]
        end = encs["none"].spans["doc"][1]
        if any(encs[n].ids[:end] != encs["none"].ids[:end] for n in post):
            errs.append(f"{key}: prompts without a preceding explanation differ before the document ends")
    by_cue = {}
    for p in prompts:
        if p.cond != "closed":
            e = p.enc
            by_cue.setdefault((p.item, p.cond, p.order), []).append(e.ids[e.spans["doc"][0]:e.spans["span"][1]])
    for key, segs in by_cue.items():
        if len({tuple(s) for s in segs}) != 1:
            errs.append(f"{key}: lead+span tokens differ across cues")
    return errs


def canonical_mismatches(tok, items: list[dict]) -> list[str]:
    """Items whose segment-wise document ids differ from tokenizing the joined document text."""
    bad = []
    for it in items:
        for cue in CUES:
            seg = encode(tok, it, CONDS["none"], cue, "AB")
            got = seg.ids[slice(*seg.spans["doc"])]
            text = " ".join(s for s in (it["lead"], it["span"], it["cues"][cue] if cue else "") if s)
            if got != _ids(tok, text):
                bad.append(f"{it['id']} cue={cue}")
    return bad


def main():
    from transformers import AutoTokenizer

    from kvmeaning.model import MODELS

    ap = argparse.ArgumentParser()
    ap.add_argument("--size", default="Q3-4B")
    ap.add_argument("--show", default="lex-01")
    a = ap.parse_args()
    spec = MODELS[a.size]
    tok = AutoTokenizer.from_pretrained(spec.repo, revision=spec.revision)
    items = load_items()
    prompts = build(tok, items)
    answer_ids(tok)
    errs = check(prompts)
    bad = canonical_mismatches(tok, items)
    lens = [len(p.enc.ids) for p in prompts]
    print(f"{len(items)} items, {len(prompts)} prompts, tokens min {min(lens)} max {max(lens)} total {sum(lens)}")
    print(Counter(p.relation for p in prompts))
    print(f"alignment violations: {len(errs)}; non-canonical document tokenizations: {len(bad)} {bad[:10]}")
    for e in errs[:20]:
        print("  ", e)
    for p in prompts:
        if p.item == a.show and p.cue == "A" and p.order == "AB" and p.cond == "B_pre":
            print(f"--- {p.item} cue={p.cue} cond={p.cond} relation={p.relation} gold={p.gold}")
            print(tok.decode(p.enc.ids))
    raise SystemExit(1 if errs else 0)


if __name__ == "__main__":
    main()
