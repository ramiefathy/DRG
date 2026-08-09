#!/usr/bin/env python3
"""Download and inventory the PLOS ONE supplementary files for the target study."""

from __future__ import annotations

import hashlib
import json
from pathlib import Path
from typing import Any

import pandas as pd
import requests

FILES = {
    "s1_description": "https://journals.plos.org/plosone/article/file?id=10.1371/journal.pone.0309147.s001&type=supplementary",
    "s2_data": "https://journals.plos.org/plosone/article/file?id=10.1371/journal.pone.0309147.s002&type=supplementary",
    "s3_strobe": "https://journals.plos.org/plosone/article/file?id=10.1371/journal.pone.0309147.s003&type=supplementary",
}


def sha256(data: bytes) -> str:
    return hashlib.sha256(data).hexdigest()


def scalar(value: Any) -> Any:
    if pd.isna(value):
        return None
    if hasattr(value, "item"):
        return value.item()
    return value


def main() -> None:
    out = Path("artifacts/supplement")
    out.mkdir(parents=True, exist_ok=True)
    session = requests.Session()
    session.headers.update({"User-Agent": "NHANES-mercury-psoriasis-replication/0.1"})

    inventory: dict[str, Any] = {}
    for name, url in FILES.items():
        response = session.get(url, timeout=180)
        response.raise_for_status()
        data = response.content
        disposition = response.headers.get("content-disposition", "")
        content_type = response.headers.get("content-type", "")
        suffix = ".bin"
        if "spreadsheetml" in content_type or data[:2] == b"PK":
            suffix = ".xlsx"
        elif data[:8] == bytes.fromhex("D0CF11E0A1B11AE1"):
            suffix = ".xls"
        elif data[:4] == b"PK\x03\x04":
            suffix = ".docx"
        path = out / f"{name}{suffix}"
        path.write_bytes(data)
        record: dict[str, Any] = {
            "url": url,
            "path": str(path),
            "bytes": len(data),
            "sha256": sha256(data),
            "content_type": content_type,
            "content_disposition": disposition,
            "magic_hex": data[:16].hex(),
        }
        if name == "s2_data":
            try:
                book = pd.ExcelFile(path)
                record["sheets"] = book.sheet_names
                sheet_records: dict[str, Any] = {}
                for sheet in book.sheet_names:
                    frame = pd.read_excel(path, sheet_name=sheet)
                    frame.columns = [str(column) for column in frame.columns]
                    frame.head(50).to_csv(out / f"s2_{sheet}_head.csv", index=False)
                    sheet_records[sheet] = {
                        "rows": int(len(frame)),
                        "columns": frame.columns.tolist(),
                        "dtypes": {column: str(dtype) for column, dtype in frame.dtypes.items()},
                        "nonmissing": {column: int(frame[column].notna().sum()) for column in frame.columns},
                        "first_row": {column: scalar(frame.iloc[0][column]) for column in frame.columns} if len(frame) else {},
                    }
                record["sheet_inventory"] = sheet_records
            except Exception as error:  # preserve the raw file even if parsing fails
                record["excel_error"] = repr(error)
        inventory[name] = record
        print(name, content_type, len(data), path)

    (out / "inventory.json").write_text(json.dumps(inventory, indent=2, sort_keys=True), encoding="utf-8")


if __name__ == "__main__":
    main()
