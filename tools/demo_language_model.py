"""Tiny fresh CPU forward only; no corpus, optimizer or generated-prose claim."""
import json
import torch
from controlled_lm.normalized_typed_language_model import NormalizedTypedLanguageModel
from controlled_lm.structured_decision import canonical_state_dict_digest

torch.set_num_threads(1)
models = []
for mode in ("none", "internal", "output_only"):
    torch.manual_seed(23)
    model = NormalizedTypedLanguageModel(8, 16, num_layers=2, semantic_mode=mode,
        projection_layout="shared", recurrent_residual=True).cpu().eval()
    with torch.no_grad():
        output = model(torch.tensor([[1, 68, 69, 2]]))
    assert torch.isfinite(output.raw_logits).all()
    assert output.final_evidence.count_nonzero() == 0
    models.append(canonical_state_dict_digest(model))
assert len(set(models)) == 1
print(json.dumps({"device":"cpu", "optimizer_updates":0,
    "matched_random_initialization":True, "raw_byte_outputs":"uncertified",
    "record_kernel_integration":"proposed; separate interface demo only"}, indent=2))
