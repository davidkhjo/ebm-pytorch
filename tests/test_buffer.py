import torch

import ebm


def test_buffer_shapes_and_len():
    buf = ebm.ReplayBuffer(capacity=100, shape=(2,))
    assert len(buf) == 100
    x = buf.sample(16)
    assert x.shape == (16, 2)


def test_push_writes_back_to_sampled_slots():
    buf = ebm.ReplayBuffer(capacity=50, shape=(3,), reinit_prob=0.0)
    x = buf.sample(8)
    new = torch.full_like(x, 7.0)
    buf.push(new)
    assert torch.equal(buf.data[buf._last_idx], new)


def test_reinit_prob_one_always_reinitializes():
    init_fn = lambda shape: torch.full(shape, 42.0)
    buf = ebm.ReplayBuffer(capacity=20, shape=(2,), reinit_prob=1.0, init_fn=init_fn)
    buf.data.zero_()
    x = buf.sample(10)
    assert torch.equal(x, torch.full((10, 2), 42.0))


def test_push_size_mismatch_raises():
    buf = ebm.ReplayBuffer(capacity=20, shape=(2,))
    buf.sample(4)
    try:
        buf.push(torch.zeros(3, 2))
    except ValueError:
        return
    raise AssertionError("expected ValueError")


def test_state_dict_roundtrip():
    buf = ebm.ReplayBuffer(capacity=10, shape=(2,))
    state = buf.state_dict()
    buf2 = ebm.ReplayBuffer(capacity=10, shape=(2,))
    buf2.load_state_dict(state)
    assert torch.equal(buf.data, buf2.data)
