"""CPU numerical/gradient checks of an affine scan; no GPU speed claims."""
import pytest
import torch

from controlled_lm.invariant_ssm import InvariantSelectiveSSM
from controlled_lm.parallel_ssm import ParallelSelectiveSSM
from controlled_lm.structured_decision import canonical_state_dict_digest


def matched(hidden=7, inputs=5, dtype=torch.float64, rho=.99):
    torch.manual_seed(15)
    sequential = InvariantSelectiveSSM(hidden, input_size=inputs, rho=rho).to(dtype)
    torch.manual_seed(15)
    parallel = ParallelSelectiveSSM(hidden, input_size=inputs, rho=rho).to(dtype)
    assert canonical_state_dict_digest(sequential) == canonical_state_dict_digest(parallel)
    return sequential, parallel


@pytest.mark.parametrize("dtype", [torch.float32, torch.float64])
@pytest.mark.parametrize("time", [1, 7, 128, 257])
def test_forward_and_final_state_numerically_match_with_masks(dtype, time):
    sequential, parallel = matched(dtype=dtype)
    generator = torch.Generator().manual_seed(44)
    inputs = torch.randn(3, time, 5, dtype=dtype, generator=generator)
    hx = torch.rand(1, 3, 7, dtype=dtype, generator=generator) * 2 - 1
    mask = torch.rand(3, time, generator=generator) > .25
    mask[0] = False
    for valid in [None, mask]:
        expected = sequential(inputs, hx, mask=valid)
        actual = parallel(inputs, hx, mask=valid)
        for left, right in zip(expected, actual):
            torch.testing.assert_close(left, right, rtol=2e-6 if dtype == torch.float32 else 1e-12,
                                       atol=3e-7 if dtype == torch.float32 else 1e-14)
        assert actual[0].abs().max() <= 1 + (1e-6 if dtype == torch.float32 else 1e-14)


@pytest.mark.parametrize("dtype", [torch.float32, torch.float64])
def test_input_state_and_parameter_gradients_match(dtype):
    sequential, parallel = matched(dtype=dtype)
    x = torch.randn(2, 33, 5, dtype=dtype, requires_grad=True)
    hx = torch.zeros(1, 2, 7, dtype=dtype, requires_grad=True)
    mask = torch.rand(2, 33) > .2
    mask[0] = False
    weight = torch.randn(2, 33, 7, dtype=dtype)
    gradients = []
    for core in [sequential, parallel]:
        output, final = core(x, hx, mask=mask)
        loss = (output * weight).sum() + final.square().sum()
        gradients.append(torch.autograd.grad(loss, [x, hx, *core.parameters()]))
    for expected, actual in zip(*gradients):
        assert torch.isfinite(actual).all()
        torch.testing.assert_close(expected, actual, rtol=2e-5 if dtype == torch.float32 else 1e-11,
                                   atol=3e-6 if dtype == torch.float32 else 1e-13)
    assert torch.count_nonzero(gradients[1][0][~mask]) == 0


def test_chunked_streaming_keeps_autograd_and_prefix_causality():
    _, core = matched()
    inputs = torch.randn(2, 71, 5, dtype=torch.float64, requires_grad=True)
    hx = torch.rand(1, 2, 7, dtype=torch.float64) * 2 - 1
    mask = torch.rand(2, 71) > .3
    full, full_state = core(inputs, hx, mask=mask)
    chunks, state = [], hx
    for start, end in [(0, 3), (3, 20), (20, 20), (20, 71)]:
        output, state = core(inputs[:, start:end], state, mask=mask[:, start:end])
        chunks.append(output)
    stream = torch.cat(chunks, dim=1)
    torch.testing.assert_close(stream, full, atol=1e-14, rtol=1e-12)
    torch.testing.assert_close(state, full_state, atol=1e-14, rtol=1e-12)
    full_grad = torch.autograd.grad(full.square().sum(), inputs, retain_graph=True)[0]
    stream_grad = torch.autograd.grad(stream.square().sum(), inputs)[0]
    torch.testing.assert_close(full_grad, stream_grad, atol=1e-13, rtol=1e-11)
    changed = inputs.detach().clone()
    changed[:, 20:] *= 100
    torch.testing.assert_close(core(changed, hx, mask=mask)[0][:, :20], full[:, :20], atol=1e-14, rtol=1e-12)


@pytest.mark.parametrize("batch,time", [(0, 0), (2, 0), (0, 5)])
def test_empty_shapes_and_lengths(batch, time):
    sequential, parallel = matched()
    inputs = torch.zeros(batch, time, 5, dtype=torch.float64)
    hx = torch.full((1, batch, 7), .2, dtype=torch.float64)
    lengths = torch.zeros(batch, dtype=torch.int64)
    expected = sequential(inputs, hx, lengths=lengths)
    actual = parallel(inputs, hx, lengths=lengths)
    for a, b in zip(actual, expected):
        torch.testing.assert_close(a, b, rtol=0, atol=0)


def test_lengths_padding_and_noncontiguous_inputs():
    sequential, parallel = matched()
    inputs = torch.randn(19, 3, 5, dtype=torch.float64).transpose(0, 1)
    hx = torch.rand(1, 3, 7, dtype=torch.float64)
    lengths = torch.tensor([19, 5, 0])
    expected = sequential(inputs, hx, lengths=lengths)
    actual = parallel(inputs, hx, lengths=lengths)
    assert not inputs.is_contiguous()
    for a, b in zip(actual, expected):
        torch.testing.assert_close(a, b, atol=1e-14, rtol=1e-12)
    torch.testing.assert_close(actual[0][2], hx[0, 2].expand(19, -1), atol=0, rtol=0)


@pytest.mark.parametrize("dtype", [torch.bfloat16, torch.float16])
def test_low_precision_projection_retains_float32_recurrence(dtype):
    sequential, parallel = matched(dtype=dtype)
    inputs = torch.randn(2, 17, 5).to(dtype)
    expected = sequential(inputs)
    actual = parallel(inputs)
    assert actual[0].dtype == actual[1].dtype == torch.float32
    torch.testing.assert_close(actual[0], expected[0], atol=3e-7, rtol=2e-6)


@pytest.mark.parametrize("rho", [.99, .99999])
def test_saturated_gates_and_long_prefix_bounds_are_measured_numerically(rho):
    sequential, parallel = matched(hidden=4, inputs=4, dtype=torch.float32, rho=rho)
    with torch.no_grad():
        for core in [sequential, parallel]:
            core.input_projection.weight.zero_()
            core.input_projection.bias.fill_(30)
    inputs = torch.randn(2, 1025, 4)
    hx = torch.ones(1, 2, 4)
    actual, _ = parallel(inputs, hx)
    expected, _ = sequential(inputs, hx)
    assert torch.isfinite(actual).all()
    assert actual.abs().max().item() <= 1 + 5e-6
    torch.testing.assert_close(actual, expected, atol=5e-6, rtol=5e-6)


def test_checkpoint_keys_and_rho_compatible_backend_identity_is_external():
    sequential, parallel = matched()
    parallel.load_state_dict(sequential.state_dict())
    sequential.load_state_dict(parallel.state_dict())
    assert list(parallel.state_dict()) == list(sequential.state_dict())
    assert "backend_name" not in parallel.state_dict()
    assert parallel.backend_name == "hillis_steele_affine_prefix_v1"
    with pytest.raises(ValueError, match="rho differs"):
        ParallelSelectiveSSM(7, input_size=5, rho=.8).load_state_dict(parallel.state_dict())


def test_invalid_arguments_match_sequential_validation():
    core = ParallelSelectiveSSM(3)
    inputs = torch.randn(2, 4, 3)
    for kwargs in [{"lengths": torch.tensor([5, 0])}, {"lengths": torch.tensor([1., 2.])},
                   {"mask": torch.ones(2, 4)}, {"hx": torch.zeros(2, 3)},
                   {"mask": torch.ones(2, 4, dtype=torch.bool), "lengths": torch.tensor([2, 3])}]:
        with pytest.raises(ValueError):
            core(inputs, **kwargs)
    with pytest.raises(ValueError, match="inputs"):
        core(inputs[:, :, :2])
    with pytest.raises(TypeError, match="floating"):
        core(torch.ones(2, 4, 3, dtype=torch.int64))
