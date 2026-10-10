import torch


def attention_probs(q: torch.Tensor, k: torch.Tensor, q_rows: slice) -> torch.Tensor:
    h_q, t, d = q.shape
    k = k.repeat_interleave(h_q // k.shape[0], dim=0)
    rows = torch.arange(t)[q_rows]
    scores = q[:, q_rows] @ k.transpose(1, 2) / d**0.5
    scores = scores.masked_fill(torch.arange(t)[None, :] > rows[:, None], float("-inf"))
    return scores.softmax(-1)


def span_mass(
    q: torch.Tensor,
    k: torch.Tensor,
    spans: dict[str, tuple[int, int]],
    sink: int = 1,
    block: int = 256,
) -> dict[str, torch.Tensor]:
    t = q.shape[1]
    out = {n: torch.zeros(q.shape[0], t) for n in spans}
    for s in range(0, t, block):
        p = attention_probs(q, k, slice(s, min(s + block, t)))
        p[..., :sink] = 0
        p = p / p.sum(-1, keepdim=True).clamp_min(1e-12)
        for n, (a, b) in spans.items():
            out[n][:, s : s + p.shape[1]] = p[..., a:b].sum(-1)
    return out
