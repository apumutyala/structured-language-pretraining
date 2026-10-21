"""Public digest-helper excerpt; legacy model implementations are omitted."""
from collections.abc import Mapping
import hashlib
import json
import torch
from torch import Tensor, nn

def _extra_state_tree(value):
    """Type-tag supported extra state; never hash object repr or pickle bytes."""
    if value is None:
        return ["null"]
    if type(value) in (str, bool, int, float):
        return [type(value).__name__, value]
    if type(value) in (list, tuple):
        return [type(value).__name__, [_extra_state_tree(item) for item in value]]
    if type(value) is dict and all(type(key) is str for key in value):
        return ["dict", [[key, _extra_state_tree(value[key])] for key in sorted(value)]]
    raise TypeError("extra_state must contain finite JSON scalars, lists, tuples, or string-key dicts")


def canonical_state_dict_digest(model_or_state: nn.Module | Mapping) -> str:
    """Hash names, tensor dtype/shape/raw bytes, and canonical non-tensor state.

    Mapping iteration order and tensor device do not affect the result. BF16
    bytes are hashed through uint8, avoiding unsupported NumPy BF16 conversion.
    Architecture choices absent from state_dict (e.g. semantic mode) must be
    recorded separately. This compares initialization, not forward semantics.
    """
    state = (model_or_state.state_dict() if isinstance(model_or_state, nn.Module)
             else model_or_state)
    digest = hashlib.sha256()

    def framed(blob: bytes) -> None:
        digest.update(len(blob).to_bytes(8, "big"))
        digest.update(blob)

    for name in sorted(state):
        framed(name.encode("utf-8"))
        value = state[name]
        if isinstance(value, Tensor):
            if value.layout != torch.strided:
                raise TypeError("digest supports dense strided state tensors")
            value = value.detach().cpu().contiguous()
            framed(json.dumps(["tensor", str(value.dtype), list(value.shape)],
                              separators=(",", ":")).encode())
            framed(value.reshape(-1).view(torch.uint8).numpy().tobytes())
        else:
            framed(json.dumps(["extra_state", _extra_state_tree(value)],
                              separators=(",", ":"), allow_nan=False).encode("utf-8"))
    return digest.hexdigest()

