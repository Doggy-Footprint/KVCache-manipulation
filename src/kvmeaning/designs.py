import json
from dataclasses import dataclass
from pathlib import Path

DONORS = ("H+", "H-", "P1", "P2", "F2")
BBC_DONORS = ("H+", "H-", "C+", "C-", "P1", "P2", "F2")


@dataclass(frozen=True)
class Design:
    prefixes: str
    passages: str
    layout: str
    donors: tuple[str, ...]


DESIGNS = {
    "v1": Design("exp1_prefixes.json", "exp1_passages.jsonl", "v1", DONORS),
    "v2": Design("exp1_prefixes_v2.json", "exp1_passages.jsonl", "v2", DONORS),
    "bbc-v1": Design("exp1_prefixes_bbc.json", "exp1_passages_bbc.jsonl", "v1", BBC_DONORS),
    "bbc-v2": Design("exp1_prefixes_bbc_v2.json", "exp1_passages_bbc.jsonl", "v2", BBC_DONORS),
}


def get(name: str) -> Design:
    if name not in DESIGNS:
        raise ValueError(f"design must be one of {list(DESIGNS)}, got {name!r}")
    return DESIGNS[name]


def load(data: Path, name: str) -> tuple[dict, list[dict]]:
    d = get(name)
    cfg = json.loads((data / d.prefixes).read_text())
    passages = [json.loads(l) for l in (data / d.passages).read_text().splitlines() if l.strip()]
    return cfg, passages
