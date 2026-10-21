"""Pinned SimpleStories pilot: completion groups, exact decontamination, no model eval."""
from collections import Counter, defaultdict
import hashlib
import json
from pathlib import Path
import re
import unicodedata

import pyarrow
import pyarrow.parquet as pq

ROOT = Path(__file__).resolve().parents[1]
RAW = ROOT / "data/language/simplestories/raw"
OUT = RAW.parent / "pilot-v1"
PATTERN = re.compile(r"^([0-9a-f]{32})-([0-9]+)$")
CAPS = {"train": 20 * 1024 ** 2, "dev": 1024 ** 2}


def normalized_hash(text):
    normalized = " ".join(unicodedata.normalize("NFC", text).split()).casefold()
    return hashlib.sha256(normalized.encode("utf-8")).hexdigest()


def main():
    if OUT.exists():
        raise FileExistsError("preserve pilot artifact; choose an explicit new version")
    acquisition = json.loads((ROOT / "results/language_source_acquisition.json").read_text())
    for entry in acquisition["files"]:
        path = RAW / entry["name"]
        digest = hashlib.file_digest(path.open("rb"), "sha256").hexdigest()
        if path.stat().st_size != entry["bytes"] or digest != entry["sha256"]:
            raise ValueError("pinned source changed after acquisition")
    test_groups, test_hashes = set(), set()
    test_rows = 0
    # Official TEST is read only for exclusion hashes/IDs, never displayed or
    # scored. All output text comes from the pinned TRAIN source.
    for batch in pq.ParquetFile(RAW / "test-00000-of-00001.parquet").iter_batches(columns=["story", "generation_id"]):
        for row in batch.to_pylist():
            test_rows += 1
            match = PATTERN.fullmatch(row["generation_id"] or "")
            if match:
                test_groups.add(match[1])
            if isinstance(row["story"], str):
                test_hashes.add(normalized_hash(row["story"]))
    parents, by_hash, rows, exclusions = {}, {}, [], set()
    counters = Counter()

    def find(group):
        parents.setdefault(group, group)
        while parents[group] != group:
            parents[group] = parents[parents[group]]
            group = parents[group]
        return group

    def union(left, right):
        left, right = find(left), find(right)
        parents[max(left, right)] = min(left, right)

    source = pq.ParquetFile(RAW / "train-00000-of-00007.parquet")
    columns = ["story", "generation_id", "num_stories_in_completion"]
    for batch in source.iter_batches(columns=columns):
        for row in batch.to_pylist():
            index = counters["source_rows"]
            counters["source_rows"] += 1
            match = PATTERN.fullmatch(row["generation_id"] or "")
            text = row["story"]
            if not match or not isinstance(text, str) or not text.strip():
                counters["quarantined_invalid"] += 1
                continue
            group = match[1]
            find(group)
            digest = normalized_hash(text)
            if digest in by_hash:
                union(group, by_hash[digest])
            else:
                by_hash[digest] = group
            if group in test_groups or digest in test_hashes:
                exclusions.add(group)
            rows.append({"source_row": index, "group": group, "generation_id": row["generation_id"],
                         "story": text, "normalized_sha256": digest,
                         "raw_sha256": hashlib.sha256(text.encode("utf-8")).hexdigest(),
                         "reported_completion_stories": row["num_stories_in_completion"]})
    excluded_roots = {find(group) for group in exclusions}
    groups = defaultdict(list)
    seen = set()
    for row in rows:
        root = find(row["group"])
        if root in excluded_roots:
            counters["excluded_test_related_rows"] += 1
        elif row["normalized_sha256"] in seen:
            counters["duplicate_rows"] += 1
        else:
            seen.add(row["normalized_sha256"])
            groups[root].append(row)
    order = lambda group: hashlib.sha256(("controlled-alignment-language-v1|" + group).encode()).hexdigest()
    chosen = {split: [] for split in CAPS}
    text_bytes = Counter()
    for group in sorted(groups, key=order):
        split = "dev" if int(order(group), 16) % 100 < 5 else "train"
        members = sorted(groups[group], key=lambda row: row["source_row"])
        size = sum(len(row["story"].encode("utf-8")) for row in members)
        if text_bytes[split] + size <= CAPS[split]:
            chosen[split].extend(members)
            text_bytes[split] += size
    # No text, completion sibling or normalized duplicate may cross folds.
    for field in ("group", "normalized_sha256"):
        if {r[field] for r in chosen["train"]} & {r[field] for r in chosen["dev"]}:
            raise AssertionError("pilot train/dev overlap")
    OUT.mkdir(parents=True)
    manifests = {}
    for split, members in chosen.items():
        path = OUT / (split + ".jsonl")
        with path.open("w", encoding="utf-8", newline="\n") as stream:
            for row in members:
                stream.write(json.dumps(row, ensure_ascii=False) + "\n")
        manifests[split] = {"path": str(path.relative_to(ROOT)), "documents": len(members),
                            "text_bytes": text_bytes[split], "file_bytes": path.stat().st_size,
                            "groups": len({r["group"] for r in members}),
                            "sha256": hashlib.file_digest(path.open("rb"), "sha256").hexdigest()}
    manifest = {"status": "development_pilot_extracted_not_trained", "pyarrow": pyarrow.__version__,
                "source_acquisition": acquisition, "counts": dict(counters), "splits": manifests,
                "official_test": {"rows_hashed": test_rows, "completion_groups": len(test_groups),
                                  "normalized_text_hashes": len(test_hashes), "model_evaluated": False},
                "selection": "fixed hash fold, retain all observed deduplicated source-group members within text-byte caps",
                "sibling_scope": "only first TRAIN shard; source groups can lack siblings in other unused shards",
                "dedup": "NFC/whitespace/case-folded exact duplicates union completion groups before split",
                "near_duplicate_or_template_decontamination": "not performed; no clean broad language benchmark claim",
                "input_policy": "story text only, no generation metadata inputs; stories never count as observed evidence",
                "license_declared_by_card": "MIT", "paid_compute": False,
                "source_sha256": hashlib.sha256(Path(__file__).read_bytes()).hexdigest()}
    (OUT / "manifest.json").write_text(json.dumps(manifest, indent=2) + "\n")
    (ROOT / "results/language_pilot_extraction.json").write_text(json.dumps(manifest, indent=2) + "\n")
    print(json.dumps({"status": manifest["status"], "counts": dict(counters), "splits": manifests}), flush=True)


if __name__ == "__main__":
    main()
