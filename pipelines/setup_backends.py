#!/usr/bin/env python3
from __future__ import annotations

import argparse
import subprocess
from pathlib import Path

PROJECTS = {
    "m3-agent": (
        "https://github.com/ByteDance-Seed/m3-agent.git",
        "0e3e41939bd8a0b66d756e7b7eb8d5fe9992da5c",
    ),
    "silvr": (
        "https://github.com/CeeZh/SILVR.git",
        "56c0afe10cb12cbcc7ee7bc321a1a43d7bae737c",
    ),
    "adaptive-k": (
        "https://github.com/megagonlabs/adaptive-k-retrieval.git",
        "fc86f12d7786cc68e7bd21885457c06e07c7a38c",
    ),
    "adaptive-rag": (
        "https://github.com/starsuzi/Adaptive-RAG.git",
        "0c88670af8707667eb5c1163151bb5ce61b14acb",
    ),
}


def main() -> None:
    parser = argparse.ArgumentParser(description="Clone optional host-agent backends")
    parser.add_argument("--output", type=Path, default=Path("third_party"))
    parser.add_argument("--project", choices=sorted(PROJECTS), action="append")
    args = parser.parse_args()
    selected = args.project or list(PROJECTS)
    args.output.mkdir(parents=True, exist_ok=True)
    for name in selected:
        url, commit = PROJECTS[name]
        destination = args.output / name
        if not destination.exists():
            subprocess.run(["git", "clone", url, str(destination)], check=True)
        if commit:
            subprocess.run(["git", "-C", str(destination), "checkout", commit], check=True)


if __name__ == "__main__":
    main()
