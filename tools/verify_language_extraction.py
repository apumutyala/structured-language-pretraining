"""Verify regenerated TRAIN/dev bytes against the pinned public split specification."""
import hashlib
import json
from pathlib import Path
ROOT = Path(__file__).resolve().parents[1]
spec = json.loads((ROOT / "data/source-spec.json").read_text())
manifest = json.loads((ROOT / "results/language_pilot_extraction.json").read_text())
for split, expected in spec["expected_splits"].items():
    row = manifest["splits"][split]
    for key, value in expected.items():
        assert row[key] == value, (split, key)
    path = ROOT / row["path"]
    with path.open("rb") as stream:
        assert hashlib.file_digest(stream,"sha256").hexdigest() == expected["sha256"]
assert manifest["source_acquisition"]["revision"] == spec["revision"]
print(json.dumps({"train_dev_sha256_verified":True,"official_test_model_evaluated":False}))
