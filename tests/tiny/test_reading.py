import torch

import tiny_helpers as H
from kvmeaning import model as M
from kvmeaning import reading as R
from kvmeaning.hooks import capture


def _ids(tok, s):
    return tok(s, add_special_tokens=False)["input_ids"]


def test_rope_at_reproduces_captured_post_rope_keys(tok):
    model = H.get_model("dense", 1.0)
    cap = capture(model, torch.tensor([_ids(tok, "The bank was steep and muddy after the rain.")]))
    for t in (0, 3, 7):
        assert torch.allclose(R.rope_at(model, cap.k_pre[:, :, t], t), cap.k_post[:, :, t], atol=1e-5)


def test_v_writes_match_o_proj_on_one_hot_attention(tok):
    model = H.get_model("dense", 1.0)
    cap = capture(model, torch.tensor([_ids(tok, "The team released a new version.")]))
    t, w = 2, R.v_writes(model, cap.v[:, :, 2])
    for j, i in enumerate(M.kv_layers(model)):
        attn = M.text_model(model).layers[i].self_attn
        hq, hkv = attn.o_proj.weight.shape[1] // attn.head_dim, cap.v.shape[1]
        for k in range(hkv):
            heads = torch.zeros(hq, attn.head_dim)
            heads[k * (hq // hkv):(k + 1) * (hq // hkv)] = cap.v[j, k, t]
            assert torch.allclose(w[j, k], attn.o_proj(heads.reshape(1, -1))[0], atol=1e-4)


def test_attention_maps_are_causal_distributions(tok):
    model = H.get_model("dense", 1.0)
    cap = capture(model, torch.tensor([_ids(tok, "One two three four five.")]))
    a = R.attention_maps(cap)
    assert torch.allclose(a.sum(-1), torch.ones(a.shape[:-1]), atol=1e-5)
    assert (a.triu(1) == 0).all()


def test_self_transplant_is_identity_and_foreign_kv_changes_output(tok):
    model = H.get_model("dense", 1.0)
    ids = _ids(tok, "Word: x. It means")
    slot = R.token_index(tok, ids, "x")
    cap = capture(model, torch.tensor([ids]))
    plain = R.transplant_generate(model, tok, ids, slot, max_new=8)
    same = R.transplant_generate(model, tok, ids, slot, cap.k_pre[:, :, slot], cap.v[:, :, slot], max_new=8)
    assert same == plain
    src = _ids(tok, "The quarterly profits of the bank fell sharply")
    scap = capture(model, torch.tensor([src]))
    t = R.token_index(tok, src, "bank")
    other = R.transplant_generate(model, tok, ids, slot, scap.k_pre[:, :, t], scap.v[:, :, t], max_new=8)
    assert other != plain
