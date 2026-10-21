"""Explicit bounded download of pinned public data; does not train or rent compute."""
import argparse
import hashlib
import json
from pathlib import Path
import urllib.request

ROOT = Path(__file__).resolve().parents[1]
SPEC = ROOT / "data/source-spec.json"
CAP = 256 * 1024 * 1024

def digest(path):
    with path.open("rb") as stream:
        return hashlib.file_digest(stream, "sha256").hexdigest()

def fetch(url, target, expected_hash, expected_bytes=None):
    if target.exists():
        if digest(target) != expected_hash or (expected_bytes is not None and target.stat().st_size != expected_bytes):
            raise ValueError("existing source does not match pinned artifact: " + target.name)
        return
    temporary = target.with_suffix(target.suffix + ".partial")
    if temporary.exists():
        raise FileExistsError("preserve incomplete download; inspect/remove it explicitly")
    limit = expected_bytes if expected_bytes is not None else 1024 * 1024
    total = 0
    try:
        with urllib.request.urlopen(url, timeout=60) as response, temporary.open("xb") as output:
            while True:
                block = response.read(min(1024 * 1024, limit + 1 - total))
                if not block:
                    break
                total += len(block)
                if total > limit:
                    raise ValueError("download exceeded declared size cap")
                output.write(block)
        if ((expected_bytes is not None and total != expected_bytes) or digest(temporary) != expected_hash):
            raise ValueError("pinned source size or digest mismatch")
        temporary.replace(target)
    except Exception:
        # Preserve partial bytes for inspection; never treat them as a verified source.
        raise

def main():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--download", action="store_true", help="explicitly perform the bounded network acquisition")
    args = parser.parse_args()
    spec = json.loads(SPEC.read_text(encoding="utf-8")); acquisition = spec["source_acquisition"]
    total = sum(item["bytes"] for item in acquisition["files"])
    if total > CAP:
        raise ValueError("source plan exceeds 256 MiB cap")
    if not args.download:
        print(json.dumps({"dataset":spec["dataset"],"revision":spec["revision"],
            "declared_file_bytes":total,"action":"inspection only; add --download to acquire"},indent=2))
        return
    raw = ROOT / "data/language/simplestories/raw"; raw.mkdir(parents=True, exist_ok=True)
    card = acquisition["card"]
    fetch(card["url"], raw / "README.md", card["sha256"])
    for item in acquisition["files"]:
        fetch(item["url"], raw / item["name"], item["sha256"], item["bytes"])
    receipt = ROOT / "results/language_source_acquisition.json"
    text = json.dumps(acquisition, indent=2) + "\n"
    if receipt.exists() and json.loads(receipt.read_text()) != acquisition:
        raise FileExistsError("preserve different existing acquisition receipt")
    receipt.write_text(text, encoding="utf-8")
    print(json.dumps({"verified_files":len(acquisition["files"]),"file_bytes":total}))

if __name__ == "__main__":
    main()
