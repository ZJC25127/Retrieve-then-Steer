"""Verify source snapshots or Git checkouts against the pinned included files."""

import argparse
import hashlib
import json
from pathlib import Path

ROOT = Path(__file__).resolve().parents[1]


def verify(root):
    manifest = json.loads((root / "provenance/upstream_source_manifest.json").read_text())
    count = 0
    for repo in manifest["repositories"]:
        tree = root / repo["destination"]
        for record in repo["files"]:
            p = tree / record["path"]
            if not p.is_file() or hashlib.sha256(p.read_bytes()).hexdigest() != record["sha256"]:
                raise RuntimeError("Upstream content mismatch: " + str(p))
            count += 1
    return dict(
        repositories=len(manifest["repositories"]),
        upstream_files=count,
        unchanged_source_bytes=True,
    )


if __name__ == "__main__":
    p = argparse.ArgumentParser(description=__doc__)
    p.add_argument("--root", type=Path, default=ROOT)
    print(json.dumps(verify(p.parse_args().root), indent=2))
