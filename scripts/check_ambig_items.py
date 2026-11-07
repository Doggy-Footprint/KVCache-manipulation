"""Validate hand-authored ambiguity-RAG items in data/ambig_rag/*.jsonl.

Errors (exit 1) are constraints the KV comparison relies on; warnings need a human look.
Derived from AmbiEnt (CC BY 4.0), Circa (CC BY 4.0 / BY-SA 4.0), and AmbigDocs (card metadata apache-2.0);
the files stay in this local-only repo.
"""

import argparse
import json
import re
import sys
from pathlib import Path

from transformers import AutoTokenizer

from kvmeaning.model import MODELS

ROOT = Path(__file__).resolve().parents[1]
RAW = ROOT / "data/raw/ambig"
PRE = "Note on the retrieved document: "
TYPES = {"lexical": "lex", "structural": "str", "pragmatic": "prag", "entity": "ent"}
PER_TYPE = 15
MIN_PARA = 5
STOP = set("""a an the of to in on at for by with and or but not no is are was were be been it its this that there
these those he she they them his her their you your we our i me my as from into about than then so if do does did
has have had will would can could should may might must just only very more most such what which who whom whose
note retrieved document here means mean refers refer""".split())


def words(s):
    return {w for w in re.findall(r"[a-z0-9]+(?:-[a-z0-9]+)*", s.lower()) if len(w) >= 4 and w not in STOP}


def doc(it, cue):
    return " ".join(s for s in (it["lead"], it["span"], it["cues"][cue] if cue else "") if s)


def sources():
    # The same AmbiEnt id can name different sentences in different files, so keep every candidate.
    amb = {}
    for f in ("test.jsonl", "dev.jsonl", "analysis/category_annotations.jsonl"):
        for line in open(RAW / "ambient-github/AmbiEnt" / f):
            r = json.loads(line)
            amb.setdefault(str(r["id"]), []).append(r)
    ad = {str(x["qid"]): x for x in json.load(open(RAW / "AmbigDocs/test.json"))}
    return amb, ad


def check(items, tok, amb, ad):
    n = lambda s: len(tok(s, add_special_tokens=False)["input_ids"])
    ids = lambda s: tok(s, add_special_tokens=False)["input_ids"]
    errs, warns, seen = [], [], set()
    for it in items:
        i = it.get("id", "?")
        E = lambda m: errs.append(f"{i}: {m}")
        W = lambda m: warns.append(f"{i}: {m}")
        need = {"id", "type", "source", "lead", "span", "cues", "question", "options", "expl", "unrelated"}
        if missing := need - it.keys():
            E(f"missing {sorted(missing)}")
            continue
        if i in seen:
            E("duplicate id")
        seen.add(i)
        if it["type"] not in TYPES or not i.startswith(TYPES[it["type"]] + "-"):
            E(f"type/id prefix mismatch ({it['type']})")
        for k in ("cues", "options", "expl") + (("expl_para",) if "expl_para" in it else ()):
            if set(it[k]) != {"A", "B"} or not all(isinstance(v, str) and v.strip() for v in it[k].values()):
                E(f"{k} must have non-empty A and B")
        if errs and errs[-1].startswith(f"{i}:"):
            continue

        le = {k: n(PRE + it["expl"][k]) for k in "AB"}
        lu = n(PRE + it["unrelated"])
        if not (le["A"] == le["B"] == lu):
            E(f"explanation lengths A={le['A']} B={le['B']} unrelated={lu} must be equal")
        if "unrelated2" in it and n(PRE + it["unrelated2"]) != lu:
            E(f"unrelated2 length {n(PRE + it['unrelated2'])} must equal {lu}")
        if "expl_para" in it:
            lp = {k: n(PRE + it["expl_para"][k]) for k in "AB"}
            if not (lp["A"] == lp["B"] == le["A"]):
                E(f"paraphrase lengths A={lp['A']} B={lp['B']} must equal {le['A']}")
        dA, dB, d0 = ids(doc(it, "A")), ids(doc(it, "B")), ids(doc(it, None))
        if len(dA) != len(dB):
            E(f"document lengths with cue A={len(dA)} B={len(dB)} must be equal")
        if dA[:len(d0)] != d0 or dB[:len(d0)] != d0:
            E("lead+span tokens change when a cue is appended (tokenizer merge across the boundary)")

        expls = [it["expl"][k] for k in "AB"] + [it.get("expl_para", {}).get(k, "") for k in "AB"]
        for k in "AB":
            opt = re.compile(r"\b" + re.escape(it["options"][k].lower()) + r"\b")
            for e in expls:
                if e and opt.search(e.lower()):
                    E(f"option {k} text appears in an explanation: {e!r}")
            for e in expls:
                if e and it["cues"][k].lower().rstrip(".") in e.lower():
                    E(f"cue {k} text appears in an explanation")
            shared = words(it["options"][k]) & words(" ".join(expls)) - words(it["span"] + " " + it["question"])
            if shared:
                E(f"explanations share content words with option {k}: {sorted(shared)}")
        if it["options"]["A"].strip().lower() == it["options"]["B"].strip().lower():
            E("options A and B are identical")
        bad = words(it["unrelated"]) & words(doc(it, "A") + " " + doc(it, "B") + " " + it["question"]
                                              + " " + " ".join(it["options"].values()))
        if bad:
            W(f"unrelated sentence shares words with the item: {sorted(bad)}")

        src = it["source"]
        if m := re.match(r"AmbiEnt (\S+)", src):
            rs = amb.get(m.group(1))
            if rs is None:
                E(f"AmbiEnt id {m.group(1)} not found")
            elif not any(it["span"] in (r["premise"], r["hypothesis"]) for r in rs):
                W("span differs from the AmbiEnt premise/hypothesis (edited)")
        elif m := re.match(r"AmbigDocs test qid (\d+)", src):
            x = ad.get(m.group(1))
            if x is None:
                E(f"AmbigDocs qid {m.group(1)} not found")
            else:
                ans = " ".join(d["answer"] + " " + d["title"] for d in x["documents"]).lower()
                for k in "AB":
                    if it["options"][k].lower() not in ans:
                        W(f"option {k} is not a substring of the AmbigDocs answers/titles")
        elif src.startswith("Circa"):
            pass
        elif src != "hand-made":
            E(f"unknown source {src!r}")
    by = {t: sum(it.get("type") == t for it in items) for t in TYPES}
    for t, c in by.items():
        if c != PER_TYPE:
            errs.append(f"type {t}: {c} items, expected {PER_TYPE}")
        p = sum(it.get("type") == t and "expl_para" in it for it in items)
        if c and p < MIN_PARA:
            errs.append(f"type {t}: {p} items with expl_para, expected >= {MIN_PARA}")
    return errs, warns


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("files", nargs="*", type=Path)
    ap.add_argument("--partial", action="store_true", help="skip per-type count checks")
    a = ap.parse_args()
    files = a.files or sorted((ROOT / "data/ambig_rag").glob("*.jsonl"))
    items = [json.loads(l) for f in files for l in open(f) if l.strip()]
    spec = MODELS["Q3-4B"]
    tok = AutoTokenizer.from_pretrained(spec.repo, revision=spec.revision)
    errs, warns = check(items, tok, *sources())
    if a.partial:
        errs = [e for e in errs if not e.startswith("type ")]
    for w in warns:
        print("WARN ", w)
    for e in errs:
        print("ERROR", e)
    print(f"{len(items)} items, {len(errs)} errors, {len(warns)} warnings")
    sys.exit(1 if errs else 0)


if __name__ == "__main__":
    main()
