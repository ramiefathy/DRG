#!/usr/bin/env python3
"""Enrich the published analysis file with NHANES cycle, strata, PSU, and correct metal weights."""

from __future__ import annotations

import hashlib
import json
from pathlib import Path

import numpy as np
import pandas as pd
import requests

BASE_URL = "https://nhanes.kylegrealis.com"
SUPPLEMENT_URL = (
    "https://journals.plos.org/plosone/article/file?"
    "id=10.1371/journal.pone.0309147.s002&type=supplementary"
)
TARGET_YEARS = {2005, 2013}


def sha256(path: Path) -> str:
    digest = hashlib.sha256()
    with path.open("rb") as stream:
        for block in iter(lambda: stream.read(1024 * 1024), b""):
            digest.update(block)
    return digest.hexdigest()


def read_remote_parquet(name: str, raw_dir: Path) -> pd.DataFrame:
    path = raw_dir / f"{name}.parquet"
    if not path.exists():
        response = requests.get(f"{BASE_URL}/{name}.parquet", timeout=180)
        response.raise_for_status()
        path.write_bytes(response.content)
    return pd.read_parquet(path)


def main() -> None:
    out = Path("artifacts/prepared")
    raw_dir = out / "raw"
    raw_dir.mkdir(parents=True, exist_ok=True)

    supplement_path = raw_dir / "published_s2_data.tsv"
    response = requests.get(SUPPLEMENT_URL, timeout=180)
    response.raise_for_status()
    supplement_path.write_bytes(response.content)
    published = pd.read_csv(supplement_path, sep="\t")

    demo = read_remote_parquet("demo", raw_dir)
    pbcd = read_remote_parquet("pbcd", raw_dir)
    demo = demo.loc[demo["year"].isin(TARGET_YEARS)].copy()
    pbcd = pbcd.loc[pbcd["year"].isin(TARGET_YEARS)].copy()

    design = demo[
        ["year", "seqn", "wtmec2yr", "sdmvstra", "sdmvpsu", "ridageyr"]
    ].merge(
        pbcd[["year", "seqn", "lbxthg", "wtsh2yr", "wtph2yr"]],
        on=["year", "seqn"],
        how="left",
        validate="one_to_one",
    )

    enriched = published.merge(
        design,
        left_on="SEQN",
        right_on="seqn",
        how="left",
        validate="one_to_one",
        indicator=True,
    )

    if not (enriched["_merge"] == "both").all():
        raise RuntimeError("Not every published record matched the public NHANES source tables")
    if not np.allclose(
        enriched["BLOOD.MERCURY..TOTAL..UG.L."], enriched["lbxthg"], rtol=0, atol=1e-12
    ):
        raise RuntimeError("Published mercury values differ from public NHANES source values")
    if not np.allclose(enriched["AGE"], enriched["ridageyr"], rtol=0, atol=0):
        raise RuntimeError("Published age values differ from public NHANES source values")

    enriched["cycle"] = enriched["year"].map({2005: "2005-2006", 2013: "2013-2014"})
    enriched["published_weight_wtmec2yr"] = enriched["WTMEC2YR"]
    enriched["correct_component_weight_2yr"] = np.where(
        enriched["year"].eq(2005), enriched["wtmec2yr"], enriched["wtsh2yr"]
    )
    enriched["correct_component_weight_4yr"] = enriched["correct_component_weight_2yr"] / 2.0
    enriched["combined_stratum"] = enriched["year"].astype(str) + "_" + enriched["sdmvstra"].astype("Int64").astype(str)
    enriched["combined_psu"] = enriched["combined_stratum"] + "_" + enriched["sdmvpsu"].astype("Int64").astype(str)
    enriched = enriched.drop(columns=["_merge", "seqn"])

    if enriched["correct_component_weight_4yr"].isna().any():
        raise RuntimeError("Correct component weights are missing in the analytic sample")
    if (enriched["correct_component_weight_4yr"] <= 0).any():
        raise RuntimeError("Correct component weights must be positive in the analytic sample")

    output_path = out / "enriched_analysis_data.tsv"
    enriched.to_csv(output_path, sep="\t", index=False)

    checks = {
        "published_rows": int(len(published)),
        "enriched_rows": int(len(enriched)),
        "psoriasis_cases": int(enriched["PSORIASIS"].sum()),
        "rows_by_cycle": enriched["cycle"].value_counts().sort_index().astype(int).to_dict(),
        "correct_weight_by_cycle": {
            cycle: {
                "n": int(len(group)),
                "min": float(group["correct_component_weight_4yr"].min()),
                "median": float(group["correct_component_weight_4yr"].median()),
                "max": float(group["correct_component_weight_4yr"].max()),
            }
            for cycle, group in enriched.groupby("cycle")
        },
        "published_weight_matches_demo_wtmec": bool(
            np.allclose(enriched["published_weight_wtmec2yr"], enriched["wtmec2yr"])
        ),
        "published_vs_correct_weight_correlation": float(
            enriched[["published_weight_wtmec2yr", "correct_component_weight_2yr"]]
            .corr()
            .iloc[0, 1]
        ),
        "source_checksums": {
            "published_s2_data": sha256(supplement_path),
            "demo_parquet": sha256(raw_dir / "demo.parquet"),
            "pbcd_parquet": sha256(raw_dir / "pbcd.parquet"),
            "enriched_analysis_data": sha256(output_path),
        },
    }
    (out / "preparation_checks.json").write_text(
        json.dumps(checks, indent=2, sort_keys=True), encoding="utf-8"
    )
    print(json.dumps(checks, indent=2, sort_keys=True))


if __name__ == "__main__":
    main()
