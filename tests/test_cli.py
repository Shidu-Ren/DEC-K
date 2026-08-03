from __future__ import annotations

import json
from pathlib import Path

from deck.cli import main


def test_select_cli(tmp_path: Path) -> None:
    root = Path(__file__).resolve().parents[1]
    output = tmp_path / "selected.jsonl"
    main(
        [
            "select",
            "--config",
            str(root / "configs" / "silvr" / "deck.yaml"),
            "--input",
            str(root / "examples" / "prepared_candidates.jsonl"),
            "--output",
            str(output),
        ]
    )
    value = json.loads(output.read_text(encoding="utf-8").strip())
    assert 2 <= value["selected_k"] <= 4
    assert value["trace"]["config"]["lambda"] == 0.85
