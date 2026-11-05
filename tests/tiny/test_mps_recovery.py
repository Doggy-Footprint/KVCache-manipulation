import pytest
import torch

from kvmeaning import patching as P


def _logits(seed):
    g = torch.Generator().manual_seed(seed)
    return [torch.randn(4, 50, generator=g) for _ in range(3)]


def test_host_branch_matches_direct_double(monkeypatch):
    d, r, p = _logits(0)
    want = P.recovery_batch(d, r, p)
    calls = []
    orig = torch.Tensor.cpu
    monkeypatch.setattr(torch.Tensor, "cpu", lambda self, *a, **k: calls.append(1) or orig(self, *a, **k))
    monkeypatch.setattr(P, "NO_F64", {"cpu"})
    got = P.recovery_batch(d, r, p)
    assert len(calls) == 3
    assert got.dtype == torch.float64 and got.device.type == "cpu"
    assert torch.equal(got, want)


def test_cuda_and_cpu_keep_device_double():
    assert P.NO_F64 == {"mps"}
    x = torch.randn(3)
    assert P._f64(x).dtype == torch.float64


@pytest.mark.skipif(not torch.backends.mps.is_available(), reason="no MPS device")
def test_real_mps_tensors():
    d, r, p = _logits(1)
    want = P.recovery_batch(d, r, p)
    got = P.recovery_batch(d.to("mps"), r.to("mps"), p.to("mps"))
    assert got.device.type == "cpu"
    assert torch.equal(got, want)
