"""Direct unit tests for EMA and frozen_params."""

import pytest
import torch
from torch import nn

from ebm.utils import EMA, frozen_params


def _tiny_module():
    m = nn.Linear(3, 2)
    m.register_buffer("count", torch.zeros(1))
    return m


def test_ema_update_is_exact_lerp():
    torch.manual_seed(0)
    src = _tiny_module()
    ema = EMA(src, decay=0.9)

    old = [p.clone() for p in ema.module.parameters()]
    with torch.no_grad():  # move the source away from the EMA copy
        for p in src.parameters():
            p.add_(1.0)
        src.count.add_(5.0)
    ema.update()

    # ema <- decay * old + (1 - decay) * new, per parameter.
    for ema_p, old_p, new_p in zip(ema.module.parameters(), old, src.parameters(), strict=True):
        expected = 0.9 * old_p + 0.1 * new_p.detach()
        assert torch.allclose(ema_p, expected)
    # Buffers are copied verbatim, not averaged.
    assert torch.equal(ema.module.count, src.count)


def test_ema_decay_validation():
    for bad in (0.0, 1.0, -0.1, 1.5):
        with pytest.raises(ValueError):
            EMA(_tiny_module(), decay=bad)


def test_ema_state_dict_roundtrip():
    src = _tiny_module()
    ema = EMA(src, decay=0.99)
    with torch.no_grad():
        for p in src.parameters():
            p.add_(0.3)
    ema.update()

    state = ema.state_dict()
    restored = EMA(_tiny_module(), decay=0.5)  # different decay on purpose
    restored.load_state_dict(state)
    assert restored.decay == 0.99
    for a, b in zip(restored.module.parameters(), ema.module.parameters(), strict=True):
        assert torch.equal(a, b)


def test_frozen_params_restores_original_states():
    m = nn.Linear(2, 2)
    m.bias.requires_grad_(False)  # mixed: weight trainable, bias frozen
    before = {n: p.requires_grad for n, p in m.named_parameters()}

    with frozen_params(m):
        assert not any(p.requires_grad for p in m.parameters())

    after = {n: p.requires_grad for n, p in m.named_parameters()}
    assert after == before  # per-parameter states restored, including the frozen bias


def test_frozen_params_accepts_none():
    with frozen_params(None):  # plain-callable energies pass None — must be a no-op
        pass
