import sys
from dataclasses import dataclass, field

import torch

from . import model as M


@dataclass
class Capture:
    v: torch.Tensor
    k_pre: torch.Tensor
    k_post: torch.Tensor
    q_post: torch.Tensor
    resid: torch.Tensor
    logits_last: torch.Tensor
    logits: dict[str, torch.Tensor] = field(default_factory=dict)
    kv_layers: list[int] = field(default_factory=list)


@torch.no_grad()
def capture(model, input_ids: torch.Tensor, logit_positions: dict[str, int] | None = None) -> Capture | list[Capture]:
    if isinstance(input_ids, (list, tuple)):
        lengths = [len(x) for x in input_ids]
        if len(set(lengths)) > 1:
            raise ValueError(f"batch requires equal lengths, got {lengths}")
        input_ids = torch.tensor([list(x) for x in input_ids])
    if input_ids.ndim == 1:
        input_ids = input_ids[None]
    B, T = input_ids.shape
    logit_positions = logit_positions or {}
    for name, pos in logit_positions.items():
        if not -T <= pos < T:
            raise IndexError(f"logit position {name}={pos} outside [{-T}, {T})")
    layers = M.text_model(model).layers
    kv = M.kv_layers(model)
    store: dict[str, list] = {"v": [], "k_pre": [], "q_pre": [], "cos_sin": []}
    handles = []

    for i in kv:
        attn = layers[i].self_attn
        handles.append(attn.v_proj.register_forward_hook(
            lambda m, i, o, a=attn: store["v"].append(o.view(*o.shape[:-1], -1, a.head_dim).transpose(1, 2))))
        handles.append(attn.k_norm.register_forward_hook(lambda m, i, o: store["k_pre"].append(o.transpose(1, 2))))
        handles.append(attn.q_norm.register_forward_hook(lambda m, i, o: store["q_pre"].append(o.transpose(1, 2))))
        # RoPE is applied inline with no submodule to hook, so post-RoPE Q is rebuilt
        # from q_norm output and the position_embeddings kwarg passed to the attention module.
        handles.append(attn.register_forward_pre_hook(
            lambda m, a, kw: store["cos_sin"].append(kw["position_embeddings"]), with_kwargs=True))

    try:
        out = model(input_ids=input_ids.to(model.device), use_cache=True, output_hidden_states=True)
    finally:
        for h in handles:
            h.remove()

    cache = out.past_key_values
    k_post, q_post = [], []
    for j, i in enumerate(kv):
        cos, sin = store["cos_sin"][j]
        # Partial-RoPE (hybrid) and dense variants define different apply_rotary_pos_emb; use the layer's own module.
        rope = sys.modules[type(layers[i].self_attn).__module__].apply_rotary_pos_emb
        q, _ = rope(store["q_pre"][j], store["k_pre"][j], cos, sin)
        q_post.append(q)
        k_post.append(cache.layers[i].keys)

    f = lambda xs, b: torch.stack([x[b] for x in xs]).float().cpu()
    caps = [
        Capture(
            v=f(store["v"], b),
            k_pre=f(store["k_pre"], b),
            k_post=f(k_post, b),
            q_post=f(q_post, b),
            resid=f(out.hidden_states, b),
            logits_last=out.logits[b, -1].float().cpu(),
            logits={n: out.logits[b, p].float().cpu() for n, p in logit_positions.items()},
            kv_layers=list(kv),
        )
        for b in range(B)
    ]
    return caps[0] if B == 1 else caps
