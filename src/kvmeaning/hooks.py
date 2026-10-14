from dataclasses import dataclass, field

import torch
from transformers.models.qwen3.modeling_qwen3 import apply_rotary_pos_emb


@dataclass
class Capture:
    v: torch.Tensor
    k_pre: torch.Tensor
    k_post: torch.Tensor
    q_post: torch.Tensor
    resid: torch.Tensor
    logits_last: torch.Tensor
    logits: dict[str, torch.Tensor] = field(default_factory=dict)


@torch.no_grad()
def capture(model, input_ids: torch.Tensor, logit_positions: dict[str, int] | None = None) -> Capture:
    T = input_ids.shape[-1]
    logit_positions = logit_positions or {}
    for name, pos in logit_positions.items():
        if not -T <= pos < T:
            raise IndexError(f"logit position {name}={pos} outside [{-T}, {T})")
    layers = model.model.layers
    store: dict[str, list] = {"v": [], "k_pre": [], "q_pre": [], "cos_sin": []}
    handles = []

    for layer in layers:
        attn = layer.self_attn
        handles.append(attn.v_proj.register_forward_hook(
            lambda m, i, o, a=attn: store["v"].append(o.view(*o.shape[:-1], -1, a.head_dim).transpose(1, 2))))
        handles.append(attn.k_norm.register_forward_hook(lambda m, i, o: store["k_pre"].append(o.transpose(1, 2))))
        handles.append(attn.q_norm.register_forward_hook(lambda m, i, o: store["q_pre"].append(o.transpose(1, 2))))
        # Qwen3Attention applies RoPE inline with no submodule to hook, so post-RoPE Q is rebuilt
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
    for i in range(len(layers)):
        cos, sin = store["cos_sin"][i]
        q, _ = apply_rotary_pos_emb(store["q_pre"][i], store["k_pre"][i], cos, sin)
        q_post.append(q)
        k_post.append(cache.layers[i].keys if hasattr(cache, "layers") else cache.key_cache[i])

    f = lambda xs: torch.stack([x[0] for x in xs]).float().cpu()
    return Capture(
        v=f(store["v"]),
        k_pre=f(store["k_pre"]),
        k_post=f(k_post),
        q_post=f(q_post),
        resid=f(out.hidden_states),
        logits_last=out.logits[0, -1].float().cpu(),
        logits={n: out.logits[0, p].float().cpu() for n, p in logit_positions.items()},
    )
