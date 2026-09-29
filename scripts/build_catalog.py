#!/usr/bin/env python3
"""Regenerate src/hibiki_asr/models/catalog.json from Hugging Face.

Every file is pinned to a commit revision with its size and sha256, so what a user
downloads is exactly what was reviewed. Run it when a model gets a new version:

    python scripts/build_catalog.py            # rewrites the catalog
    python scripts/build_catalog.py --check    # exits 1 if the committed catalog is stale

Add a new version by appending to ENTRIES (keep the old ones; users may have them installed).
"""

from __future__ import annotations

import argparse
import hashlib
import json
import sys
from pathlib import Path

import httpx

HF = "https://huggingface.co"
CATALOG_PATH = Path(__file__).resolve().parents[1] / "src" / "hibiki_asr" / "models" / "catalog.json"

# kind "model": listed to users, versioned, can be selected for a job.
# kind "component": a shared dependency (VAD, feature extractor) pulled in through `requires`.
ENTRIES: list[dict] = [
    {
        "id": "chickenrice",
        "kind": "model",
        "display_name": "ChickenRice (海南鸡) ja→zh",
        "task": "translate",
        "source_languages": ["ja"],
        "output_languages": ["zh"],
        "license_note": "Model card declares Apache-2.0 (base: openai/whisper-large-v2); training data provenance is not disclosed.",
        "requires": ["vad-asr@1", "whisper-base-fe@1"],
        "versions": [
            {
                "version": "v2",
                "notes": "Whisper large-v2 fine-tuned on 5000 h of Japanese audio with Chinese subtitles; translates straight to Chinese.",
                "repo": "chickenrice0721/whisper-large-v2-translate-zh-v0.2-st-ct2",
                "revision": "2a896581429aa8100eaa5586bc36a5b3011871bc",
                "files": ["config.json", "model.bin", "preprocessor_config.json", "tokenizer.json", "vocabulary.json"],
            }
        ],
    },
    {
        "id": "whisper-ja",
        "kind": "model",
        "display_name": "Whisper Japanese 1.5B (transcription)",
        "task": "transcribe",
        "source_languages": ["ja"],
        "output_languages": ["ja"],
        "license_note": "No license is declared for these weights (nor for the original efwkjn/whisper-ja-1.5B). Check with the authors before redistributing.",
        "requires": ["vad-asr@1", "whisper-base-fe@1"],
        "versions": [
            {
                "version": "1.5b",
                "notes": "Whisper large-v3 fine-tune for Japanese, CTranslate2 bfloat16.",
                "repo": "TransWithAI/whisper-ja-1.5B-ct2",
                "revision": "1527314b14da5bdf0d14e7328649d4fa26840188",
                "files": ["config.json", "model.bin", "preprocessor_config.json", "tokenizer.json", "vocabulary.json"],
            }
        ],
    },
    {
        "id": "vad-asr",
        "kind": "component",
        "display_name": "ASMR voice activity detection",
        "license_note": "MIT",
        "versions": [
            {
                "version": "1",
                "notes": "Whisper-base encoder with a small decoder, trained on about 500 h of Japanese ASMR.",
                "repo": "TransWithAI/Whisper-Vad-EncDec-ASMR-onnx",
                "revision": "6ac29e2cbf2f4f8e9b639861766a8639dd666e9c",
                "files": ["model.onnx", "model_metadata.json"],
            }
        ],
    },
    {
        "id": "whisper-base-fe",
        "kind": "component",
        "display_name": "Whisper-base feature extractor config",
        "license_note": "Apache-2.0",
        "versions": [
            {
                "version": "1",
                "notes": "preprocessor_config.json only; the VAD was trained on these log-mel features.",
                "repo": "openai/whisper-base",
                "revision": "e37978b90ca9030d5170a5c07aadb050351a65bb",
                "files": ["preprocessor_config.json"],
            }
        ],
    },
]


def file_info(client: httpx.Client, repo: str, revision: str, tree: dict[str, dict], path: str) -> dict:
    node = tree.get(path)
    if node is None:
        raise SystemExit(f"{repo}@{revision[:8]} has no file {path}")
    lfs = node.get("lfs")
    if lfs and lfs.get("sha256"):
        return {"path": path, "size": node["size"], "sha256": lfs["sha256"]}

    # Small files are stored in git, not LFS, so the API has no sha256 for them: hash the bytes.
    digest = hashlib.sha256()
    size = 0
    with client.stream("GET", f"{HF}/{repo}/resolve/{revision}/{path}", follow_redirects=True) as response:
        response.raise_for_status()
        for chunk in response.iter_bytes():
            digest.update(chunk)
            size += len(chunk)
    if size != node["size"]:
        raise SystemExit(f"{repo}/{path}: downloaded {size} bytes, API says {node['size']}")
    return {"path": path, "size": size, "sha256": digest.hexdigest()}


def build() -> dict:
    entries: list[dict] = []
    with httpx.Client(timeout=120) as client:
        for spec in ENTRIES:
            entry = {k: v for k, v in spec.items() if k != "versions"}
            entry["versions"] = []
            for version in spec["versions"]:
                repo, revision = version["repo"], version["revision"]
                response = client.get(f"{HF}/api/models/{repo}/revision/{revision}", params={"blobs": "true"})
                response.raise_for_status()
                tree = {s["rfilename"]: s for s in response.json()["siblings"]}
                entry["versions"].append(
                    {
                        **{k: v for k, v in version.items() if k != "files"},
                        "files": [file_info(client, repo, revision, tree, p) for p in version["files"]],
                    }
                )
            entries.append(entry)
    return {"schema": 1, "entries": entries}


def main() -> int:
    parser = argparse.ArgumentParser(description=__doc__, formatter_class=argparse.RawDescriptionHelpFormatter)
    parser.add_argument("--check", action="store_true", help="fail if the committed catalog differs")
    args = parser.parse_args()

    text = json.dumps(build(), indent=2, ensure_ascii=False) + "\n"
    if args.check:
        if CATALOG_PATH.read_text(encoding="utf-8") != text:
            print("catalog.json is stale; run scripts/build_catalog.py", file=sys.stderr)
            return 1
        return 0
    CATALOG_PATH.write_text(text, encoding="utf-8")
    print(f"wrote {CATALOG_PATH}")
    return 0


if __name__ == "__main__":
    sys.exit(main())
