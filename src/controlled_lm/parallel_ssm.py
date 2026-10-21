"""Vectorized affine prefix scan for the bounded diagonal selective SSM.

Each step is (a,b): h' = a*h+b, with b=(1-a)*tanh(W_b*x).
Composition is (a2,b2) o (a1,b1) = (a2*a1, b2+a2*b1). The
Hillis-Steele scan doubles the represented prefix length each round; after
ceil(log2(T)) rounds every position has its complete affine prefix. Incoming
state contributes prefix_a*hx. Padding is the exact-real identity pair (1,0).
No cumulative-product division, clipping, custom kernel or compilation is used.

The real-arithmetic cube/fixed-input contraction theorem is unchanged. This
changes association of floating-point sums/products, so numerical results,
gradients, padding and chunk boundaries may differ by rounding. It is not a
proof of PyTorch/CUDA refinement or semantic safety. Finite affine arithmetic
is still a precondition; source tests measure rather than certify FP bounds.

Work is O(B*T*H*log(T)), with similar worst-case saved autograd storage, versus
the sequential core's linear work. Only O(log(T)) rounds are sequential. Extra
temporary tensors may cost more than the saved launches; benchmark actual
training shapes/device before selecting this backend. CPU timings are not
evidence of GPU speedup. Checkpoints contain exactly the sequential core's
projection tensors and rho extra state; record backend_name plus source hash
separately in experiment provenance because those state_dicts are compatible.
"""
from __future__ import annotations

import torch
from torch import Tensor

from .invariant_ssm import InvariantSelectiveSSM


class ParallelSelectiveSSM(InvariantSelectiveSSM):
    """Same constructor/state_dict/I/O as InvariantSelectiveSSM; tree scan."""

    backend_name = "hillis_steele_affine_prefix_v1"

    def forward(self, inputs: Tensor, hx: Tensor | None = None, *,
                mask: Tensor | None = None,
                lengths: Tensor | None = None) -> tuple[Tensor, Tensor]:
        if inputs.ndim != 3 or inputs.shape[-1] != self.input_size:
            raise ValueError("inputs must have shape [batch, time, input_size]")
        if not inputs.is_floating_point():
            raise TypeError("inputs must have floating point dtype")
        batch, time, _ = inputs.shape
        if mask is not None and lengths is not None:
            raise ValueError("supply mask or lengths, not both")
        if lengths is not None:
            lengths = torch.as_tensor(lengths, device=inputs.device)
            if lengths.shape != (batch,) or lengths.dtype not in (
                    torch.int8, torch.int16, torch.int32, torch.int64, torch.uint8):
                raise ValueError("lengths must be an integer vector [batch]")
            if bool(((lengths < 0) | (lengths > time)).any()):
                raise ValueError("lengths must lie between 0 and time")
            mask = torch.arange(time, device=inputs.device)[None, :] < lengths[:, None]
        if mask is not None:
            if mask.dtype != torch.bool or mask.shape != (batch, time):
                raise ValueError("mask must be boolean with shape [batch, time]")
            mask = mask.to(device=inputs.device)

        projected = self.input_projection(inputs)
        dtype = torch.float64 if projected.dtype == torch.float64 else torch.float32
        gate, candidate = projected.to(dtype).chunk(2, dim=-1)
        a = self.rho * gate.sigmoid()
        b = (1 - a) * candidate.tanh()
        if hx is None:
            initial = torch.zeros(batch, self.hidden_size, device=inputs.device, dtype=dtype)
        else:
            if hx.shape != (1, batch, self.hidden_size):
                raise ValueError("hx must have shape [1, batch, hidden_size]")
            if hx.device != inputs.device or not hx.is_floating_point():
                raise ValueError("hx must be floating point on the input device")
            initial = hx[0].to(dtype)
        if time == 0:
            return initial.new_empty(batch, 0, self.hidden_size), initial.unsqueeze(0)
        if mask is not None:
            a = torch.where(mask[:, :, None], a, 1)
            b = torch.where(mask[:, :, None], b, 0)
        offset = 1
        while offset < time:
            # Both new tensors read the PREVIOUS round, avoiding autograd
            # mutation/version errors and keeping composition order explicit.
            next_b = torch.cat((b[:, :offset], b[:, offset:] + a[:, offset:] * b[:, :-offset]), dim=1)
            a = torch.cat((a[:, :offset], a[:, offset:] * a[:, :-offset]), dim=1)
            b = next_b
            offset *= 2
        output = b + a * initial[:, None]
        return output, output[:, -1].unsqueeze(0)
