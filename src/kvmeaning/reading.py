import sys

import torch

from . import model as M
from .attention_mass import attention_probs


def token_index(tok, ids: list[int], word: str, occurrence: int = -1) -> int:
    hits = [i for i, t in enumerate(ids) if tok.decode([t]).strip() == word]
    if not hits:
        raise ValueError(f"{word!r} is not a single token in {tok.decode(ids)!r}")
    return hits[occurrence]


def attention_maps(cap) -> torch.Tensor:
    """[layer, q_head, T, T] attention probabilities recomputed from the captured post-RoPE Q/K."""
    t = cap.q_post.shape[2]
    return torch.stack([attention_probs(q, k, slice(0, t)) for q, k in zip(cap.q_post, cap.k_post)])


def _rope(model):
    attn = M.text_model(model).layers[M.kv_layers(model)[0]].self_attn
    return sys.modules[type(attn).__module__].apply_rotary_pos_emb


@torch.no_grad()
def rope_at(model, k_pre: torch.Tensor, pos: int) -> torch.Tensor:
    """Rotate pre-RoPE keys [layer, kv_head, head_dim] as if their token sat at `pos`."""
    k = k_pre[:, :, None].to(model.device, torch.float32)
    cos, sin = M.text_model(model).rotary_emb(k, torch.tensor([[pos]], device=model.device))
    _, k_post = _rope(model)(k, k, cos.float(), sin.float())
    return k_post[:, :, 0].cpu()


@torch.no_grad()
def v_writes(model, v: torch.Tensor) -> torch.Tensor:
    """[layer, kv_head, hidden]: what each KV head's value at one token adds to the residual when every query head
    sharing it attends to that token alone (sum over the GQA group of V @ W_O slice)."""
    layers = M.text_model(model).layers
    out = []
    for j, i in enumerate(M.kv_layers(model)):
        attn = layers[i].self_attn
        w = attn.o_proj.weight.float().cpu()
        hq, hkv, hd = w.shape[1] // attn.head_dim, v.shape[1], attn.head_dim
        w = w.view(w.shape[0], hkv, hq // hkv, hd).sum(2)
        out.append(torch.einsum("khd,kd->kh", w.permute(1, 0, 2), v[j].float()))
    return torch.stack(out)


_UNEMBED: dict[int, torch.Tensor] = {}


@torch.no_grad()
def logit_lens(model, x: torch.Tensor, top: int = 5) -> tuple[torch.Tensor, torch.Tensor]:
    """Top tokens of final-norm + unembedding applied to residual-space vectors x [..., hidden]."""
    if id(model) not in _UNEMBED:
        # RMS scaling does not change the ranking, so only the norm's elementwise weight is folded in.
        w = model.get_output_embeddings().weight.float() * M.text_model(model).norm.weight.float()
        _UNEMBED[id(model)] = w
    w = _UNEMBED[id(model)]
    v, i = (x.to(w.device, torch.float32) @ w.T).topk(top, dim=-1)
    return v.cpu(), i.cpu()


@torch.no_grad()
def transplant(model, ids: list[int], slot: int, k_pre=None, v=None, layers=None):
    """Forward `ids` after overwriting the KV at position `slot` with a donor token's KV; returns the model output.

    k_pre/v are [layer, kv_head, head_dim]; the donor key is re-rotated to `slot`. None leaves the slot unpatched."""
    if not 0 <= slot < len(ids) - 1:
        raise ValueError("slot must be followed by at least one token")
    x = torch.tensor([ids], device=model.device)
    cache = model(input_ids=x[:, :slot + 1], use_cache=True).past_key_values
    if k_pre is not None:
        k_new = rope_at(model, k_pre, slot)
        for j, i in enumerate(M.kv_layers(model)):
            if layers is not None and i not in layers:
                continue
            lay = cache.layers[i]
            lay.keys[0, :, slot] = k_new[j].to(lay.keys)
            lay.values[0, :, slot] = v[j].to(lay.values)
    return model(input_ids=x[:, slot + 1:], past_key_values=cache, use_cache=True)


@torch.no_grad()
def transplant_generate(model, tok, ids: list[int], slot: int, k_pre=None, v=None, layers=None,
                        max_new: int = 24) -> str:
    out = transplant(model, ids, slot, k_pre, v, layers)
    stop = {tok.eos_token_id, tok.convert_tokens_to_ids("<|im_end|>")}
    gen = []
    for _ in range(max_new):
        nxt = out.logits[0, -1].argmax()
        if nxt.item() in stop:
            break
        gen.append(nxt.item())
        out = model(input_ids=nxt.view(1, 1), past_key_values=out.past_key_values, use_cache=True)
    return tok.decode(gen)
