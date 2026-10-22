"""Segments are tokenized independently and concatenated as ids instead of tokenizing the
rendered string, so the body's token ids cannot change with the preceding context
(BPE merges across the prefix/body boundary would otherwise break cross-condition alignment).
ChatML is rendered by hand because the Qwen3-0.6B template injects an empty <think> block
into the final assistant turn, which makes per-turn spans non prefix-stable.
"""

from dataclasses import dataclass, field

DELIM = "\n\n"


@dataclass
class Encoded:
    ids: list[int]
    spans: dict[str, tuple[int, int]] = field(default_factory=dict)


def _ids(tok, text: str) -> list[int]:
    return tok(text, add_special_tokens=False)["input_ids"]


def _fit(tok, pad_text: str, n: int) -> list[int]:
    ids = _ids(tok, pad_text)
    assert len(ids) >= n, f"pad text too short: need {n} tokens, have {len(ids)}"
    return ids[:n]


def exp1_prefix_ids(tok, prefixes: dict[str, str], fillers: dict[str, str], pad_text: str, design: str = "v1") -> dict[str, list[int]]:
    if design == "v2":
        # Every condition is shared left pad + one complete core sentence, so format and length are identical across conditions.
        cores = {**prefixes, **fillers}
        ids = {c: _ids(tok, t) for c, t in cores.items()}
        n = max(len(v) for v in ids.values())
        return {c: _fit(tok, pad_text, n - len(v)) + v for c, v in ids.items()}
    if design != "v1":
        raise ValueError(f"design must be 'v1' or 'v2', got {design!r}")
    # Left-padding keeps the instruction adjacent to the body.
    instr = {c: _ids(tok, p) for c, p in prefixes.items()}
    n = max(len(v) for v in instr.values())
    out = {c: _fit(tok, pad_text, n - len(v)) + v for c, v in instr.items()}
    out.update({c: _fit(tok, f, n) for c, f in fillers.items()})
    out["N"] = []
    return out


def exp1_encode(tok, prefix_ids: list[int], body: str) -> Encoded:
    head = _ids(tok, "<|im_start|>user\n")
    delim = _ids(tok, DELIM) if prefix_ids else []
    body_ids = _ids(tok, body)
    tail = _ids(tok, "<|im_end|>\n<|im_start|>assistant\n")
    p0 = len(head)
    b0 = p0 + len(prefix_ids) + len(delim)
    b1 = b0 + len(body_ids)
    ids = head + prefix_ids + delim + body_ids + tail
    return Encoded(ids, {"prefix": (p0, p0 + len(prefix_ids)), "body": (b0, b1)})


def exp2_encode(tok, turns: list[dict]) -> Encoded:
    ids: list[int] = []
    spans = {}
    for i, t in enumerate(turns):
        ids += _ids(tok, f"<|im_start|>{t['role']}\n")
        c = _ids(tok, t["content"])
        spans[f"turn_{i}"] = (len(ids), len(ids) + len(c))
        ids += c + _ids(tok, "<|im_end|>\n")
    return Encoded(ids, spans)


def assert_aligned(encs: dict[str, Encoded], span: str = "body") -> None:
    ref_name, ref = next(iter(encs.items()))
    a, b = ref.spans[span]
    for name, e in encs.items():
        x, y = e.spans[span]
        assert e.ids[x:y] == ref.ids[a:b], f"{span} tokens differ: {name} vs {ref_name}"
