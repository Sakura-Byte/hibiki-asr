#!/usr/bin/env python3
"""Write (or verify) openapi.json, the contract clients build against.

    python scripts/export_openapi.py            # regenerate openapi.json
    python scripts/export_openapi.py --check    # exit 1 if the committed file is stale (used by CI)

Change the API, run this, and commit openapi.json together with the code. If the change is not
backwards compatible, bump API_VERSION in src/hibiki_asr/__init__.py as well.
"""

from __future__ import annotations

import argparse
import sys
from pathlib import Path

from hibiki_asr.api.openapi import render_openapi

TARGET = Path(__file__).resolve().parents[1] / "openapi.json"


def main() -> int:
    parser = argparse.ArgumentParser(
        description=__doc__, formatter_class=argparse.RawDescriptionHelpFormatter
    )
    parser.add_argument("--check", action="store_true", help="fail if openapi.json differs from the code")
    args = parser.parse_args()

    text = render_openapi()
    if args.check:
        if not TARGET.exists() or TARGET.read_text(encoding="utf-8") != text:
            print(
                "openapi.json is stale; run `python scripts/export_openapi.py` and commit it", file=sys.stderr
            )
            return 1
        return 0
    TARGET.write_text(text, encoding="utf-8")
    print(f"wrote {TARGET}")
    return 0


if __name__ == "__main__":
    sys.exit(main())
