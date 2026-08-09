#!/usr/bin/env python3
"""Inspect public NHANES tables needed for the mercury–psoriasis replication."""

from __future__ import annotations

import hashlib
import json
from pathlib import Path
from typing import Any

import pandas as pd
import requests

BASE_URL = "https://nhanes.kylegrealis.com"
TARGET_YEARS = {2005, 2013}
DATASETS = [
    "demo",
    "mcq",
    "pbcd",
    "bmx",
    "alq",
    "smq",
    "diq",
    "bpq",
    "tchol",
    "trigly",
    "hdl",
    "glu",
    "biopro",
    "cbc",
]
CANDIDATE_VARIABLES = {
    "demo": [
        "RIDAGEYR", "RIAGENDR", "RIDRETH1", "RIDRETH3", "DMDEDUC2",
        "INDFMPIR", "WTMEC2YR", "SDMVPSU", "SDMVSTRA",
    ],
    "mcq": ["MCQ160M"],
    "pbcd": ["LBXTHG", "LBXBPB", "LBXBCD"],
    "bmx": ["BMXBMI", "BMXWAIST"],
    "alq": ["ALQ101", "ALQ110", "ALQ120Q", "ALQ120U", "ALQ130"],
    "smq": ["SMQ020", "SMQ040", "SMD030"],
    "diq": ["DIQ010"],
    "bpq": ["BPQ020"],
    "tchol": ["LBXTC"],
    "trigly": ["LBXTR", "LBDLDL", "LBDLDLSI"],
    "hdl": ["LBDHDD", "LBDHDDSI"],
    "glu": ["LBXGLU", "LBDGLUSI"],
    "biopro": ["LBXSTB", "LBDS TBSI", "LBDSTBSI"],
    "cbc": ["LBXWBCSI"],
}


def sha256(path: Path) -> str:
    digest = hashlib.sha256()
    with path.open("rb") as stream:
        for block in iter(lambda: stream.read(1024 * 1024), b""):
            digest.update(block)
    return digest.hexdigest()


def jsonable(value: Any) -> Any:
    if pd.isna(value):
        return None
    if hasattr(value, "item"):
        return value.item()
    return value


def value_counts(series: pd.Series, limit: int = 20) -> list[dict[str, Any]]:
    counts = series.value_counts(dropna=False).head(limit)
    return [
        {"value": jsonable(index), "count": int(count)}
        for index, count in counts.items()
    ]


def main() -> None:
    output_dir = Path("artifacts/inspection")
    raw_dir = output_dir / "raw"
    raw_dir.mkdir(parents=True, exist_ok=True)

    report: dict[str, Any] = {
        "base_url": BASE_URL,
        "target_years": sorted(TARGET_YEARS),
        "tables": {},
    }

    session = requests.Session()
    session.headers.update({"User-Agent": "NHANES-mercury-psoriasis-replication/0.1"})

    for dataset in DATASETS:
        url = f"{BASE_URL}/{dataset}.parquet"
        path = raw_dir / f"{dataset}.parquet"
        response = session.get(url, timeout=180)
        response.raise_for_status()
        path.write_bytes(response.content)

        frame = pd.read_parquet(path)
        frame.columns = [str(column) for column in frame.columns]
        year_column = "year" if "year" in frame.columns else "YEAR"
        seqn_column = "seqn" if "seqn" in frame.columns else "SEQN"
        target = frame.loc[frame[year_column].isin(TARGET_YEARS)].copy()

        columns_upper = {column.upper(): column for column in frame.columns}
        candidates: dict[str, Any] = {}
        for requested in CANDIDATE_VARIABLES.get(dataset, []):
            actual = columns_upper.get(requested.upper())
            if actual is not None:
                candidates[requested] = {
                    "actual_column": actual,
                    "dtype": str(target[actual].dtype),
                    "nonmissing": int(target[actual].notna().sum()),
                    "value_counts": value_counts(target[actual]),
                }

        keyword_columns = [
            column
            for column in frame.columns
            if any(term in column.lower() for term in ("psor", "mercur", "cadmi", "lead"))
        ]

        report["tables"][dataset] = {
            "url": url,
            "path": str(path),
            "bytes": path.stat().st_size,
            "sha256": sha256(path),
            "rows_total": int(len(frame)),
            "rows_target": int(len(target)),
            "rows_by_year": {
                str(jsonable(year)): int(count)
                for year, count in target[year_column].value_counts(dropna=False).sort_index().items()
            },
            "year_column": year_column,
            "seqn_column": seqn_column,
            "columns": frame.columns.tolist(),
            "dtypes": {column: str(dtype) for column, dtype in frame.dtypes.items()},
            "candidate_variables": candidates,
            "keyword_columns": keyword_columns,
        }

        target.head(25).to_csv(output_dir / f"{dataset}_target_head.csv", index=False)
        print(f"{dataset}: {len(frame):,} rows; {len(target):,} in target cycles; {len(frame.columns)} columns")

    (output_dir / "inspection.json").write_text(
        json.dumps(report, indent=2, sort_keys=True), encoding="utf-8"
    )
    print(f"Wrote {output_dir / 'inspection.json'}")


if __name__ == "__main__":
    main()
