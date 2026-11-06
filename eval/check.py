"""Fail when evaluation scores drop below eval/thresholds.json.

    python -m eval.check retrieval
    python -m eval.check answers
"""

from __future__ import annotations

import argparse
import json
import pathlib
from typing import Any

EVAL_DIR = pathlib.Path(__file__).parent
THRESHOLDS = EVAL_DIR / "thresholds.json"
RESULTS = {
    "retrieval": EVAL_DIR / "results" / "retrieval.json",
    "answers": EVAL_DIR / "results" / "answers.json",
}


def scores(kind: str, report: dict[str, Any], configuration: str | None) -> dict[str, Any]:
    if kind == "answers":
        return report["summary"]
    for row in report["results"]:
        if row["configuration"] == configuration:
            return row["metrics"]
    raise SystemExit(f"Configuration {configuration!r} not found in the retrieval report")


def failures(actual: dict[str, Any], minimums: dict[str, float]) -> list[str]:
    return [
        f"{metric} {actual.get(metric)} < {minimum}"
        for metric, minimum in minimums.items()
        if actual.get(metric) is None or actual[metric] < minimum
    ]


def main() -> None:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("kind", choices=sorted(RESULTS))
    parser.add_argument("--results", type=pathlib.Path)
    args = parser.parse_args()

    threshold = json.loads(THRESHOLDS.read_text())[args.kind]
    report = json.loads((args.results or RESULTS[args.kind]).read_text())
    actual = scores(args.kind, report, threshold.get("configuration"))

    for metric, minimum in threshold["min"].items():
        print(f"{metric:<18} {actual.get(metric)!s:>8}  (min {minimum})")

    if problems := failures(actual, threshold["min"]):
        raise SystemExit("Below threshold: " + "; ".join(problems))
    print("All thresholds met.")


if __name__ == "__main__":
    main()
