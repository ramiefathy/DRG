#!/usr/bin/env python3
"""Download and inventory public 2021-2023 MEPS files for a dermatology-access study."""

from __future__ import annotations

import hashlib
import json
import zipfile
from pathlib import Path
from typing import Any

import pandas as pd
import requests

YEARS = {
    2021: {"person": "h233", "conditions": "h231", "office": "h229g", "outpatient": "h229f", "link": "h229i"},
    2022: {"person": "h243", "conditions": "h241", "office": "h239g", "outpatient": "h239f", "link": "h239i"},
    2023: {"person": "h251", "conditions": "h249", "office": "h248g", "outpatient": "h248f", "link": "h248i"},
}
BASE = "https://meps.ahrq.gov/mepsweb/data_files/pufs"
KEYWORDS = (
    "dupersid", "evntidx", "condidx", "icd10", "ccsr", "drsplty", "seedoc",
    "age", "sex", "race", "hisp", "educ", "povcat", "inscov", "region",
    "perwt", "varstr", "varpsu", "eventype", "obcond", "opcond",
)


def sha256(path: Path) -> str:
    digest = hashlib.sha256()
    with path.open("rb") as stream:
        for block in iter(lambda: stream.read(1024 * 1024), b""):
            digest.update(block)
    return digest.hexdigest()


def serializable(value: Any) -> Any:
    if pd.isna(value):
        return None
    if hasattr(value, "item"):
        return value.item()
    return value


def find_dta(directory: Path) -> Path:
    candidates = list(directory.rglob("*.dta"))
    if len(candidates) != 1:
        raise RuntimeError(f"Expected one Stata file in {directory}; found {candidates}")
    return candidates[0]


def main() -> None:
    out = Path("artifacts/meps_inspection")
    raw = out / "raw"
    raw.mkdir(parents=True, exist_ok=True)
    session = requests.Session()
    session.headers.update({"User-Agent": "MEPS-dermatology-access-study/0.1"})
    inventory: dict[str, Any] = {"source": BASE, "files": {}}

    for year, files in YEARS.items():
        for role, stem in files.items():
            url = f"{BASE}/{stem}/{stem}dta.zip"
            zip_path = raw / f"{stem}dta.zip"
            response = session.get(url, timeout=180)
            response.raise_for_status()
            zip_path.write_bytes(response.content)
            extract_dir = raw / stem
            extract_dir.mkdir(exist_ok=True)
            with zipfile.ZipFile(zip_path) as archive:
                archive.extractall(extract_dir)
            dta_path = find_dta(extract_dir)
            frame = pd.read_stata(dta_path, convert_categoricals=False)
            columns = [str(column) for column in frame.columns]
            matched = [
                column for column in columns
                if any(keyword in column.lower() for keyword in KEYWORDS)
            ]
            value_summaries: dict[str, Any] = {}
            for column in matched:
                series = frame[column]
                if series.nunique(dropna=False) <= 50:
                    counts = series.value_counts(dropna=False).head(50)
                    value_summaries[column] = [
                        {"value": serializable(index), "count": int(count)}
                        for index, count in counts.items()
                    ]
            key = f"{year}_{role}"
            inventory["files"][key] = {
                "year": year,
                "role": role,
                "stem": stem,
                "url": url,
                "zip_bytes": zip_path.stat().st_size,
                "zip_sha256": sha256(zip_path),
                "dta_path": str(dta_path),
                "dta_bytes": dta_path.stat().st_size,
                "dta_sha256": sha256(dta_path),
                "rows": int(len(frame)),
                "columns": columns,
                "matched_columns": matched,
                "dtypes": {column: str(frame[column].dtype) for column in matched},
                "value_summaries": value_summaries,
            }
            frame.head(20).to_csv(out / f"{key}_head.csv", index=False)
            print(f"{key}: {len(frame):,} rows, {len(columns)} columns; matched {matched}")

    (out / "inventory.json").write_text(
        json.dumps(inventory, indent=2, sort_keys=True), encoding="utf-8"
    )


if __name__ == "__main__":
    main()
