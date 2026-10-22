import os

os.environ["HF_HUB_OFFLINE"] = "1"
os.environ["TRANSFORMERS_OFFLINE"] = "1"

from pathlib import Path  # noqa: E402

import pytest  # noqa: E402
import transformers  # noqa: E402
from transformers import modeling_utils  # noqa: E402
from transformers.models.auto import auto_factory  # noqa: E402

import tiny_helpers as H  # noqa: E402

WEIGHT_LOADERS = [
    modeling_utils.PreTrainedModel,
    auto_factory._BaseAutoModelClass,
    transformers.AutoModel,
    transformers.AutoModelForCausalLM,
    transformers.Qwen3ForCausalLM,
    transformers.Qwen3_5ForCausalLM,
    transformers.Qwen3_5ForConditionalGeneration,
]
CONFIG_LOADERS = [transformers.AutoConfig]


class Guard:
    def __init__(self):
        self.calls = []
        self.fake = None

    def _check(self, kind, target, kwargs):
        local = isinstance(target, (str, Path)) and Path(str(target)).exists()
        self.calls.append((kind, str(target), dict(kwargs)))
        if not local:
            raise AssertionError(f"VO10 guard: {kind} from_pretrained({target!r}) is not a local path")

    def weights(self, cls, target=None, *a, **kw):
        if self.fake is not None:
            self.calls.append(("weights", str(target), dict(kw)))
            return self.fake()
        self._check("weights", target, kw)

    def config(self, cls, target=None, *a, **kw):
        if self.fake is not None:  # load() fetches the config before weights; serve the tiny model's config
            self.calls.append(("config", str(target), dict(kw)))
            cfg = self.fake().config
            return (cfg, {}) if kw.get("return_unused_kwargs") else cfg
        self._check("config", target, kw)


@pytest.fixture(autouse=True)
def guard(monkeypatch, tok):
    g = Guard()
    for cls in WEIGHT_LOADERS:
        monkeypatch.setattr(cls, "from_pretrained", classmethod(lambda c, t=None, *a, **kw: g.weights(c, t, *a, **kw)))
    for cls in CONFIG_LOADERS:
        monkeypatch.setattr(cls, "from_pretrained", classmethod(lambda c, t=None, *a, **kw: g.config(c, t, *a, **kw)))
    return g


@pytest.fixture(scope="session")
def tok():
    return H.load_tokenizer()


@pytest.fixture(scope="session")
def dense_model():
    return H.build_dense(partial=1.0, layers=3, seed=1)


@pytest.fixture(scope="session")
def hybrid_model():
    return H.build_hybrid(partial=0.25, seed=2)


@pytest.fixture(scope="session")
def hybrid_model_full_rope():
    return H.build_hybrid(partial=1.0, seed=3)


def pytest_configure(config):
    config.addinivalue_line("markers", "slow: end-to-end CLI sweep")
