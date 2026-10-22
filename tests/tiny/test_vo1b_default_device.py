"""torch.nn.Module.to is recorded and replaced by a no-op, so nothing is moved onto MPS."""
import pytest
import torch
import transformers

import tiny_helpers as H
from kvmeaning import model as M


def _resolve(cfg, name):
    cfg = getattr(cfg, "text_config", cfg)
    val = getattr(cfg, name, None)
    if val is None and name == "layer_types":
        val = ["full_attention"] * cfg.num_hidden_layers
    return val


def _device_args(args, kwargs):
    out = []
    for a in list(args) + [kwargs.get("device")]:
        if isinstance(a, torch.device):
            out.append(a)
        elif isinstance(a, str):
            out.append(torch.device(a))
        elif isinstance(a, int) and not isinstance(a, bool):
            out.append(torch.device("cuda", a))
    return out


@pytest.mark.parametrize("mps", [True, False], ids=["mps", "nomps"])
@pytest.mark.parametrize("key", ["0.6B", "4B"])
def test_default_device_target_recorded(monkeypatch, guard, tok, key, mps):
    monkeypatch.setattr(transformers.AutoTokenizer, "from_pretrained", classmethod(lambda c, t=None, *a, **kw: tok))
    model = H.build_dense(layers=3, seed=51)
    model.config._attn_implementation = "sdpa"
    guard.fake = lambda: model
    exp = {k: _resolve(model.config, k) for k in M.MODELS[key].expected}
    monkeypatch.setitem(M.MODELS, key, M.ModelSpec(repo="tiny/local-stand-in", revision=M.MODELS[key].revision, expected=exp))

    monkeypatch.setattr(torch.backends.mps, "is_available", lambda: mps)
    recorded = []

    def fake_to(self, *args, **kwargs):
        recorded.append((args, kwargs))
        return self

    monkeypatch.setattr(torch.nn.Module, "to", fake_to)
    got, _ = M.load(key, dtype=torch.float32)
    assert got is model

    expected = torch.device("mps" if mps else "cpu")
    targets = [d for a, kw in recorded for d in _device_args(a, kw)]
    assert expected in targets, (recorded, expected)
    assert all(d == expected for d in targets), targets
    (call,) = [c for c in guard.calls if c[0] == "weights"]
    dm = call[2].get("device_map")
    assert dm is None or "cuda" not in str(dm), call[2]
