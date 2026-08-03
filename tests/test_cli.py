from __future__ import annotations

import json
from pathlib import Path

from deck.cli import main


def test_select_cli(tmp_path: Path) -> None:
    root = Path(__file__).resolve().parents[1]
    input_path = tmp_path / "candidates.jsonl"
    output = tmp_path / "selected.jsonl"
    input_path.write_text(
        json.dumps(
            {
                "query_id": "test",
                "question": "Where was the mug placed?",
                "candidates": [
                    {"clip_id": "1", "relevance": 0.93, "vector": [1.0, 0.0]},
                    {"clip_id": "2", "relevance": 0.91, "vector": [0.99, 0.01]},
                    {"clip_id": "3", "relevance": 0.82, "vector": [0.0, 1.0]},
                    {"clip_id": "4", "relevance": 0.40, "vector": [0.1, 0.9]},
                ],
            }
        )
        + "\n",
        encoding="utf-8",
    )
    main(
        [
            "select",
            "--config",
            str(root / "configs" / "silvr" / "deck.yaml"),
            "--input",
            str(input_path),
            "--output",
            str(output),
        ]
    )
    value = json.loads(output.read_text(encoding="utf-8").strip())
    assert 2 <= value["selected_k"] <= 4
    assert value["trace"]["config"]["lambda"] == 0.85
