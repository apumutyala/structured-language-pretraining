"""Equal-encoding typed workspace placement, action-learning fit diagnostic.

No added parameters. Internal injects the full typed workspace before the SSM;
typed_readout adds the same projection immediately before the common final
RMS/SwiGLU/readout. Query bytes, workspace encoding, labels and allocated/active
parameters match. This isolates placement better than the raw JSON arm, but it
is not a general tool-language external baseline or joint language model.
"""
from __future__ import annotations

import torch
from torch.nn import functional as F

from .normalized_typed_language_model import parameter_free_rms
from .record_learning_comparison import RecordProposer, typed_workspace


class RecordPlacementProposer(RecordProposer):
    """Same inherited fresh initialization; only workspace routing differs."""

    def forward(self, observations, arm, hx=None):
        if arm not in ("internal", "typed_readout"):
            raise ValueError("placement must be internal or typed_readout")
        encoded = [[b + 1 for b in obs["question"].encode("ascii")] for obs in observations]
        device = self.embedding.weight.device
        lengths = torch.tensor([len(row) for row in encoded], device=device)
        ids = torch.zeros(len(encoded), int(lengths.max()), dtype=torch.long, device=device)
        for i, row in enumerate(encoded):
            ids[i, :len(row)] = torch.tensor(row, device=device)
        valid = torch.arange(ids.shape[1], device=device)[None] < lengths[:, None]
        workspace = torch.tensor([typed_workspace(obs) for obs in observations],
                                 dtype=self.embedding.weight.dtype, device=device)
        semantic = self.semantic_projection(workspace)
        embeddings = self.embedding(ids)
        inputs = embeddings + semantic[:, None] * (arm == "internal")
        states, final = self.core(inputs, hx, mask=valid)
        rows = torch.arange(ids.shape[0], device=device)
        residual = states[rows, lengths - 1] + embeddings[rows, lengths - 1]
        residual = residual + semantic * (arm == "typed_readout")
        normalized = parameter_free_rms(residual)
        stream = residual + self.down(F.silu(self.gate(normalized)) * self.value(normalized))
        outputs = tuple(head(parameter_free_rms(stream)) for head in self.heads)
        return outputs, final, lengths.tolist()
