"""Exact observed-event workspace using a parallel last-reset prefix scan.

This is an alternative implementation of FixedSemanticTransitions.forward,
with the same input validation and immutable KEEP/WRITE Boolean bank. It
records observed tokens, not authenticated truth or permission. No generated
text feedback, learned state writes, parameters, or checkpoint-loaded buffers
are accepted. Callers must supply original observed discrete token history.

Proof (each bit independently): the fixed bank has WRITE => not KEEP. An
event thus either preserves the previous bit (KEEP=1, WRITE=0), or resets it
to WRITE (KEEP=0). Masked tokens are identities. Starting at zero, the bit at
position t is zero if no valid reset has occurred, otherwise it is WRITE at
the latest valid reset position <= t. Prefix cummax of reset positions finds
exactly that position; gathering WRITE from a prepended zero sentinel gives
exactly the sequential recurrence. Independence lifts this bit argument to
all 32 bits, all batches, and any finite token/mask history.

The argument reuses the unchanged Boolean transition bank. Correspondence
between this source and a particular compiled PyTorch/device implementation
is tested rather than formally verified. Integer/Boolean operations avoid
floating accumulation and return exact 0/1 float32 features. There is no
claim of GPU speedup: prefix work, integer index memory, and gather cost must
be measured on the actual training workload. This materializes O(B*T*32)
int64 indices, trading memory and parallel work for fewer sequential launches.
"""

from __future__ import annotations

import torch
from torch import Tensor, nn

from .semantic_transitions import (
    KEYS, VALUES, KEEP_WEIGHTS, WRITE_WEIGHTS, _mask, _tokens,
)


class ParallelSemanticTransitions(nn.Module):
    """Return exact post-token prefix facts [B,T,4,8], starting empty.

    No mutable bank is stored on the module. Calling `.to()`, optimization,
    or loading a state_dict cannot change the immutable transition weights.
    Masked positions hold the preceding state, including interior holes.
    """

    def forward(self, input_ids: Tensor,
                attention_mask: Tensor | None = None) -> Tensor:
        _tokens(input_ids, 2)
        valid = _mask(attention_mask, input_ids)
        batch, time = input_ids.shape
        if time == 0 or batch == 0:
            return torch.empty((batch, time, KEYS, VALUES), dtype=torch.float32,
                               device=input_ids.device)
        keep_bank = torch.tensor(KEEP_WEIGHTS, dtype=torch.bool, device=input_ids.device)
        write_bank = torch.tensor(WRITE_WEIGHTS, dtype=torch.bool, device=input_ids.device)
        indices = input_ids.long()
        reset = (~keep_bank[indices]).reshape(batch, time, KEYS, VALUES)
        writes = write_bank[indices].reshape(batch, time, KEYS, VALUES)
        if valid is not None:
            reset = reset & valid[:, :, None, None]
        # Reserve index zero for the empty initial workspace. Every reset
        # position is positive; cummax therefore selects the latest reset.
        positions = torch.arange(1, time + 1, dtype=torch.int64,
                                 device=input_ids.device).reshape(1, time, 1, 1)
        last_reset = torch.where(reset, positions, 0).cummax(dim=1).values
        zero = torch.zeros((batch, 1, KEYS, VALUES), dtype=torch.bool,
                           device=input_ids.device)
        source_writes = torch.cat((zero, writes), dim=1)
        return source_writes.gather(dim=1, index=last_reset).to(torch.float32)
