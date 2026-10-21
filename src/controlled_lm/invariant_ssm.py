"""A from-scratch diagonal selective state-space recurrence with bounded states.

The structural guarantees concern recurrent state dynamics, not semantic safety.
See ``docs/invariant_ssm.md`` for the proof and numerical assumptions.
"""

from __future__ import annotations

import math

import torch
from torch import Tensor, nn


class InvariantSelectiveSSM(nn.Module):
    """Input-selective affine recurrence with a fixed contraction ceiling.

    For each token, ``a = rho * sigmoid(W_a x + b_a)``,
    ``b = tanh(W_b x + b_b)``, and ``h = a*h + (1-a)*b``.
    The input projection is initialized randomly by PyTorch; no weights are
    inherited. This is one recurrent layer with batch-first, GRU-shaped I/O.

    ``hx`` is optional and has shape ``[1, batch, hidden_size]``. The cube
    invariance guarantee requires an initial state in [-1, 1]. An omitted
    initial state is zero. Padding holds the previous state, including in the
    returned sequence. Low precision projections use float32 recurrence state;
    float64 projections retain float64 state. No state or gradient clipping is
    applied. Input/parameter finiteness is a caller precondition, not checked
    with per-token device synchronizations.
    """

    def __init__(self, hidden_size: int, *, input_size: int | None = None,
                 rho: float = 0.99) -> None:
        super().__init__()
        input_size = hidden_size if input_size is None else input_size
        if hidden_size <= 0 or input_size <= 0:
            raise ValueError("input_size and hidden_size must be positive")
        # The minimum recurrence precision is fp32. Reject a ceiling that
        # rounds to 1 there, since that would silently remove contraction.
        if not math.isfinite(rho) or not 0 < rho < 1:
            raise ValueError("rho must be finite and strictly between 0 and 1")
        if torch.tensor(rho, dtype=torch.float32).item() >= 1:
            raise ValueError("rho must remain below 1 in float32")
        self.input_size = input_size
        self.hidden_size = hidden_size
        self.rho = float(rho)
        self.input_projection = nn.Linear(input_size, 2 * hidden_size)

    def get_extra_state(self) -> dict[str, float]:
        # rho is architectural configuration, not an optimizable parameter.
        return {"rho": self.rho}

    def set_extra_state(self, state: dict[str, float]) -> None:
        # A checkpoint with a different ceiling is a different architecture;
        # do not silently change its recurrence contract during loading.
        if state != self.get_extra_state():
            raise ValueError("checkpoint rho differs from this SSM's rho")

    def forward(self, inputs: Tensor, hx: Tensor | None = None, *,
                mask: Tensor | None = None,
                lengths: Tensor | None = None) -> tuple[Tensor, Tensor]:
        """Return sequence states and final state, preserving chunk autograd.

        ``mask`` is a boolean [B,T] validity mask (holes are supported).
        ``lengths`` is an integer [B] vector of valid prefix lengths. Supply
        at most one. For streaming, slice the mask with the input or convert
        global lengths to the valid lengths within each chunk.
        """
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

        # Project all tokens together. The only sequential operation is the
        # affine diagonal state update; no hidden-state-dependent gate exists.
        projected = self.input_projection(inputs)
        state_dtype = torch.float64 if projected.dtype == torch.float64 else torch.float32
        gate_logits, candidate_logits = projected.to(state_dtype).chunk(2, dim=-1)
        coefficients = self.rho * torch.sigmoid(gate_logits)
        candidates = torch.tanh(candidate_logits)
        if hx is None:
            state = torch.zeros(batch, self.hidden_size, dtype=state_dtype,
                                device=inputs.device)
        else:
            if hx.shape != (1, batch, self.hidden_size):
                raise ValueError("hx must have shape [1, batch, hidden_size]")
            if hx.device != inputs.device or not hx.is_floating_point():
                raise ValueError("hx must be floating point on the input device")
            state = hx[0].to(state_dtype)

        sequence = []
        for index in range(time):
            coefficient = coefficients[:, index]
            next_state = (coefficient * state
                          + (1 - coefficient) * candidates[:, index])
            state = (next_state if mask is None else
                     torch.where(mask[:, index, None], next_state, state))
            sequence.append(state)
        output = (torch.stack(sequence, dim=1) if sequence else
                  state.new_empty(batch, 0, self.hidden_size))
        return output, state.unsqueeze(0)
