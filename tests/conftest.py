import json
from pathlib import Path

import pytest

from kvmeaning import model as M
from kvmeaning.spans import exp1_encode, exp1_prefix_ids

DATA = Path(__file__).resolve().parents[1] / "data"


@pytest.fixture(scope="session")
def model_tok():
    return M.load("0.6B")


def encode_passage(tok, idx):
    cfg = json.loads((DATA / "exp1_prefixes.json").read_text())
    p = json.loads((DATA / "exp1_passages.jsonl").read_text().splitlines()[idx])
    prefixes = {c: s.format(**p) for c, s in cfg["prefixes"].items()}
    pids = exp1_prefix_ids(tok, prefixes, cfg["fillers"], cfg["pad_text"])
    return {c: exp1_encode(tok, ids, p["text"]) for c, ids in pids.items()}


@pytest.fixture(scope="session")
def p01_encs(model_tok):
    return encode_passage(model_tok[1], 0)


@pytest.fixture(scope="session")
def p02_encs(model_tok):
    return encode_passage(model_tok[1], 1)


def pytest_configure(config):
    config.addinivalue_line("markers", "slow: end-to-end CLI sweep")
