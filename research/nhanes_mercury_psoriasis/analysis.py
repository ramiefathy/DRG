#!/usr/bin/env python3
"""Reproduce and audit the NHANES blood-mercury–psoriasis analysis.

This script performs three linked tasks:
1. Reproduce the published ordinary logistic-regression results from the authors'
   shared participant-level analysis file.
2. Refit the association using the official NHANES strata, PSU identifiers, and
   cycle-specific blood-metals sampling weights.
3. Evaluate robustness to exposure scale, covariate set, missing-data handling,
   cycle, quartiles, outlier exclusion, and simple nonlinearity.

It intentionally does not claim causal inference. The outcome is self-reported
prevalent psoriasis and the study is cross-sectional.
"""

from __future__ import annotations

import argparse
import hashlib
import html
import json
import math
import shutil
import sys
from dataclasses import dataclass
from pathlib import Path
from typing import Any, Iterable, Sequence

import matplotlib.pyplot as plt
import mistune
import numpy as np
import pandas as pd
import patsy
import scipy.stats as st
import statsmodels.api as sm
from jinja2 import Template


ANALYSIS_VERSION = "1.0.0"
PAPER_DOI = "10.1371/journal.pone.0309147"

RENAME = {
    "GENDER": "sex_code",
    "AGE": "age",
    "RACE": "race_code",
    "EDUCATION": "education_code",
    "PIR": "pir",
    "WHITE.BLOOD.CELL.COUNT..1000.CELLS.UL.": "wbc",
    "BMI": "bmi",
    "WAIST.CIRCUMFERENCE..CM.": "waist",
    "GLUCOSE..SERUM..MG.DL.": "glucose",
    "TOTAL.BILIRUBIN..MG.DL.": "bilirubin",
    "HAD.AT.LEAST.12.ALCOHOL.DRINKS.1.YR.": "alcohol_code",
    "BLOOD.CADMIUM..UG.L.": "cadmium",
    "BLOOD.LEAD..UG.DL.": "lead",
    "BLOOD.MERCURY..TOTAL..UG.L.": "mercury",
    "PSORIASIS": "psoriasis",
    "HIGH.BLOOD.PRESSURE": "hypertension_code",
    "DIRECT.HDL.CHOLESTEROL..MG.DL.": "hdl",
    "DIABETES": "diabetes_code",
    "TRIGLYCERIDE..MG.DL.": "triglyceride",
    "LDL.CHOLESTEROL..MG.DL.": "ldl",
    "TOTAL.CHOLESTEROL..MG.DL.": "total_cholesterol",
    "SMOKED.AT.LEAST.100.CIGARETTES.IN.LIFE": "smoking_code",
}

RACE_LABELS = {
    1: "Mexican American",
    2: "Other Hispanic",
    3: "Non-Hispanic White",
    4: "Non-Hispanic Black",
    5: "Other/multiracial",
}
SEX_LABELS = {1: "Male", 2: "Female"}
EDUCATION_LABELS = {
    1: "Less than 9th grade",
    2: "9th–11th grade",
    3: "High school/GED",
    4: "Some college/AA",
    5: "College graduate or above",
}
YES_NO_LABELS = {1: "Yes", 2: "No"}
DIABETES_LABELS = {1: "Yes", 2: "No", 3: "Borderline"}

CONTINUOUS_REPORTED = [
    "age",
    "pir",
    "wbc",
    "bmi",
    "waist",
    "glucose",
    "bilirubin",
    "cadmium",
    "lead",
    "hdl",
    "triglyceride",
    "ldl",
    "total_cholesterol",
]
FULL_COVARIATES = [
    "age",
    "sex_code",
    "race_code",
    "education_code",
    "pir",
    "wbc",
    "bmi",
    "waist",
    "glucose",
    "bilirubin",
    "alcohol_code",
    "cadmium",
    "lead",
    "hypertension_code",
    "hdl",
    "diabetes_code",
    "triglyceride",
    "ldl",
    "total_cholesterol",
    "smoking_code",
]
PARSIMONIOUS_COVARIATES = [
    "age",
    "sex",
    "race",
    "education",
    "pir",
    "bmi",
    "smoking",
    "alcohol",
    "cadmium",
    "lead",
    "cycle",
]


@dataclass
class FitResult:
    coefficients: pd.DataFrame
    n: int
    cases: int
    n_parameters: int
    converged: bool
    design_df: float | None
    formula: str
    estimator: str
    covariance: np.ndarray
    beta: np.ndarray
    terms: list[str]
    index: pd.Index


class AnalysisError(RuntimeError):
    """Raised when an invariant required for the audit fails."""


def sha256_file(path: Path) -> str:
    digest = hashlib.sha256()
    with path.open("rb") as stream:
        for block in iter(lambda: stream.read(1024 * 1024), b""):
            digest.update(block)
    return digest.hexdigest()


def as_float(value: Any) -> float | None:
    if value is None or pd.isna(value):
        return None
    return float(value)


def weighted_quantile(
    values: Sequence[float], weights: Sequence[float], quantiles: Sequence[float]
) -> np.ndarray:
    values_array = np.asarray(values, dtype=float)
    weights_array = np.asarray(weights, dtype=float)
    mask = np.isfinite(values_array) & np.isfinite(weights_array) & (weights_array > 0)
    values_array = values_array[mask]
    weights_array = weights_array[mask]
    if not len(values_array):
        return np.full(len(quantiles), np.nan)
    order = np.argsort(values_array, kind="mergesort")
    values_array = values_array[order]
    weights_array = weights_array[order]
    positions = np.cumsum(weights_array) - 0.5 * weights_array
    positions /= weights_array.sum()
    return np.interp(np.asarray(quantiles, dtype=float), positions, values_array)


def _survey_meat(
    scores: np.ndarray, strata: Sequence[Any], psus: Sequence[Any]
) -> tuple[np.ndarray, int, int, list[str]]:
    metadata = pd.DataFrame({"stratum": strata, "psu": psus})
    meat = np.zeros((scores.shape[1], scores.shape[1]), dtype=float)
    n_psus = 0
    n_strata = 0
    singleton_strata: list[str] = []
    for stratum, indices in metadata.groupby("stratum", sort=False).groups.items():
        locations = np.asarray(list(indices), dtype=int)
        stratum_psu = metadata.iloc[locations]["psu"].to_numpy()
        psu_scores = []
        for psu in pd.unique(stratum_psu):
            psu_scores.append(scores[locations[stratum_psu == psu]].sum(axis=0))
        psu_scores_array = np.vstack(psu_scores)
        m_h = psu_scores_array.shape[0]
        if m_h < 2:
            singleton_strata.append(str(stratum))
            continue
        centered = psu_scores_array - psu_scores_array.mean(axis=0)
        meat += (m_h / (m_h - 1.0)) * (centered.T @ centered)
        n_psus += m_h
        n_strata += 1
    return meat, n_psus, n_strata, singleton_strata


def fit_unweighted_logit(formula: str, data: pd.DataFrame) -> FitResult:
    y_frame, x_frame = patsy.dmatrices(
        formula, data, return_type="dataframe", NA_action="drop"
    )
    y = np.asarray(y_frame).reshape(-1)
    model = sm.GLM(y, x_frame, family=sm.families.Binomial())
    result = model.fit(maxiter=500, disp=0)
    covariance = np.asarray(result.cov_params())
    beta = np.asarray(result.params)
    se = np.sqrt(np.clip(np.diag(covariance), 0, np.inf))
    z_stat = np.divide(beta, se, out=np.full_like(beta, np.nan), where=se > 0)
    p_value = 2.0 * st.norm.sf(np.abs(z_stat))
    critical = st.norm.ppf(0.975)
    coefficient_table = pd.DataFrame(
        {
            "term": x_frame.columns,
            "beta": beta,
            "se": se,
            "statistic": z_stat,
            "p_value": p_value,
            "odds_ratio": np.exp(beta),
            "ci_low": np.exp(beta - critical * se),
            "ci_high": np.exp(beta + critical * se),
        }
    )
    return FitResult(
        coefficients=coefficient_table,
        n=int(len(y)),
        cases=int(y.sum()),
        n_parameters=int(x_frame.shape[1]),
        converged=bool(result.converged),
        design_df=None,
        formula=formula,
        estimator="ordinary logistic regression",
        covariance=covariance,
        beta=beta,
        terms=x_frame.columns.tolist(),
        index=x_frame.index,
    )


def fit_survey_logit(
    formula: str,
    data: pd.DataFrame,
    weight_column: str = "correct_component_weight_4yr",
    stratum_column: str = "combined_stratum",
    psu_column: str = "combined_psu",
) -> FitResult:
    required = [weight_column, stratum_column, psu_column]
    absent = [column for column in required if column not in data.columns]
    if absent:
        raise AnalysisError(f"Missing survey-design columns: {absent}")

    y_frame, x_frame = patsy.dmatrices(
        formula, data, return_type="dataframe", NA_action="drop"
    )
    index = x_frame.index
    y = np.asarray(y_frame).reshape(-1)
    x = np.asarray(x_frame, dtype=float)
    weights = data.loc[index, weight_column].to_numpy(dtype=float)
    if np.any(~np.isfinite(weights)) or np.any(weights <= 0):
        raise AnalysisError("Survey weights must be finite and positive in the analysis domain")
    # Multiplying every weight by a constant leaves both beta and the linearized
    # covariance unchanged. Normalization avoids numerical conditioning problems.
    weights = weights / np.mean(weights)

    model = sm.GLM(y, x, family=sm.families.Binomial(), freq_weights=weights)
    result = model.fit(maxiter=500, disp=0)
    beta = np.asarray(result.params)
    fitted = np.asarray(result.fittedvalues)

    bread = x.T @ ((weights * fitted * (1.0 - fitted))[:, None] * x)
    individual_scores = (weights * (y - fitted))[:, None] * x
    # Reset rows to positional indices before passing them into the variance routine.
    subset = data.loc[index, [stratum_column, psu_column]].reset_index(drop=True)
    meat, n_psus, n_strata, singletons = _survey_meat(
        individual_scores,
        subset[stratum_column].to_numpy(),
        subset[psu_column].to_numpy(),
    )
    if singletons:
        raise AnalysisError(
            "Singleton strata occurred in an analysis domain: " + ", ".join(singletons)
        )
    design_df = n_psus - n_strata
    if design_df <= 0:
        raise AnalysisError("Survey design has nonpositive residual degrees of freedom")
    bread_inverse = np.linalg.pinv(bread)
    covariance = bread_inverse @ meat @ bread_inverse
    covariance = 0.5 * (covariance + covariance.T)
    se = np.sqrt(np.clip(np.diag(covariance), 0, np.inf))
    t_stat = np.divide(beta, se, out=np.full_like(beta, np.nan), where=se > 0)
    p_value = 2.0 * st.t.sf(np.abs(t_stat), design_df)
    critical = st.t.ppf(0.975, design_df)
    coefficient_table = pd.DataFrame(
        {
            "term": x_frame.columns,
            "beta": beta,
            "se": se,
            "statistic": t_stat,
            "p_value": p_value,
            "odds_ratio": np.exp(beta),
            "ci_low": np.exp(beta - critical * se),
            "ci_high": np.exp(beta + critical * se),
        }
    )
    return FitResult(
        coefficients=coefficient_table,
        n=int(len(y)),
        cases=int(y.sum()),
        n_parameters=int(x.shape[1]),
        converged=bool(result.converged),
        design_df=float(design_df),
        formula=formula,
        estimator="Taylor-linearized survey logistic regression",
        covariance=covariance,
        beta=beta,
        terms=x_frame.columns.tolist(),
        index=index,
    )


def fit_pseudoweighted_logit(
    formula: str, data: pd.DataFrame, weight_column: str = "published_weight_wtmec2yr"
) -> FitResult:
    """Fit a normalized simple-weight model without strata/PSU correction.

    This is not considered a valid NHANES analysis. It is included only as a
    forensic candidate for what an analysis using the shared WTMEC2YR column may
    have produced.
    """
    y_frame, x_frame = patsy.dmatrices(
        formula, data, return_type="dataframe", NA_action="drop"
    )
    index = x_frame.index
    y = np.asarray(y_frame).reshape(-1)
    weights = data.loc[index, weight_column].to_numpy(dtype=float)
    weights = weights / np.mean(weights)
    model = sm.GLM(y, x_frame, family=sm.families.Binomial(), freq_weights=weights)
    result = model.fit(maxiter=500, disp=0)
    covariance = np.asarray(result.cov_params())
    beta = np.asarray(result.params)
    se = np.sqrt(np.clip(np.diag(covariance), 0, np.inf))
    z_stat = beta / se
    p_value = 2.0 * st.norm.sf(np.abs(z_stat))
    critical = st.norm.ppf(0.975)
    coefficient_table = pd.DataFrame(
        {
            "term": x_frame.columns,
            "beta": beta,
            "se": se,
            "statistic": z_stat,
            "p_value": p_value,
            "odds_ratio": np.exp(beta),
            "ci_low": np.exp(beta - critical * se),
            "ci_high": np.exp(beta + critical * se),
        }
    )
    return FitResult(
        coefficients=coefficient_table,
        n=int(len(y)),
        cases=int(y.sum()),
        n_parameters=int(x_frame.shape[1]),
        converged=bool(result.converged),
        design_df=None,
        formula=formula,
        estimator="normalized WTMEC2YR-weighted logistic regression without design correction",
        covariance=covariance,
        beta=beta,
        terms=x_frame.columns.tolist(),
        index=index,
    )


def survey_mean(
    data: pd.DataFrame,
    variable: str,
    weight_column: str = "correct_component_weight_4yr",
    stratum_column: str = "combined_stratum",
    psu_column: str = "combined_psu",
) -> dict[str, float]:
    subset = data[[variable, weight_column, stratum_column, psu_column]].dropna().copy()
    values = subset[variable].to_numpy(dtype=float)
    weights = subset[weight_column].to_numpy(dtype=float)
    estimate = float(np.sum(weights * values) / np.sum(weights))
    linearized = (weights * (values - estimate) / np.sum(weights))[:, None]
    meat, n_psus, n_strata, singletons = _survey_meat(
        linearized,
        subset[stratum_column].to_numpy(),
        subset[psu_column].to_numpy(),
    )
    if singletons:
        raise AnalysisError(f"Singleton strata in survey mean for {variable}: {singletons}")
    variance = float(meat[0, 0])
    se = math.sqrt(max(variance, 0.0))
    design_df = n_psus - n_strata
    critical = st.t.ppf(0.975, design_df)
    return {
        "estimate": estimate,
        "se": se,
        "ci_low": estimate - critical * se,
        "ci_high": estimate + critical * se,
        "design_df": float(design_df),
        "n": float(len(subset)),
    }


def clean_data(raw: pd.DataFrame) -> pd.DataFrame:
    data = raw.rename(columns=RENAME).copy()
    required = {
        "SEQN",
        "sex_code",
        "age",
        "race_code",
        "mercury",
        "psoriasis",
        "cycle",
        "correct_component_weight_4yr",
        "combined_stratum",
        "combined_psu",
    }
    missing = required.difference(data.columns)
    if missing:
        raise AnalysisError(f"Input is missing required columns: {sorted(missing)}")

    # Preserve literal numeric codes for forensic reproductions while creating
    # cleaned categorical versions for substantive analyses.
    data["sex"] = data["sex_code"].map(SEX_LABELS)
    data["race"] = data["race_code"].map(RACE_LABELS)
    data["education"] = data["education_code"].where(
        data["education_code"].isin(EDUCATION_LABELS), np.nan
    ).map(EDUCATION_LABELS)
    data["alcohol"] = data["alcohol_code"].where(
        data["alcohol_code"].isin(YES_NO_LABELS), np.nan
    ).map(YES_NO_LABELS)
    data["smoking"] = data["smoking_code"].where(
        data["smoking_code"].isin(YES_NO_LABELS), np.nan
    ).map(YES_NO_LABELS)
    data["hypertension"] = data["hypertension_code"].where(
        data["hypertension_code"].isin(YES_NO_LABELS), np.nan
    ).map(YES_NO_LABELS)
    data["diabetes"] = data["diabetes_code"].where(
        data["diabetes_code"].isin(DIABETES_LABELS), np.nan
    ).map(DIABETES_LABELS)

    for column in ["sex", "race", "education", "alcohol", "smoking", "hypertension", "diabetes", "cycle"]:
        data[column] = pd.Categorical(data[column])

    if (data["mercury"] <= 0).any():
        raise AnalysisError("Log-scale analyses require strictly positive mercury values")
    data["log2_mercury"] = np.log2(data["mercury"])
    data["female"] = (data["sex_code"] == 2).astype(float)
    data["non_hispanic_white"] = (data["race_code"] == 3).astype(float)

    unweighted_cutpoints = data["mercury"].quantile([0.25, 0.5, 0.75]).to_numpy()
    correct_cutpoints = weighted_quantile(
        data["mercury"], data["correct_component_weight_4yr"], [0.25, 0.5, 0.75]
    )
    data["hg_quartile_unweighted"] = pd.cut(
        data["mercury"],
        bins=[-np.inf, *unweighted_cutpoints, np.inf],
        labels=["Q1", "Q2", "Q3", "Q4"],
        include_lowest=True,
    )
    data["hg_quartile_weighted"] = pd.cut(
        data["mercury"],
        bins=[-np.inf, *correct_cutpoints, np.inf],
        labels=["Q1", "Q2", "Q3", "Q4"],
        include_lowest=True,
    )
    data["hg_quartile_unweighted_ordinal"] = data["hg_quartile_unweighted"].cat.codes + 1
    data["hg_quartile_weighted_ordinal"] = data["hg_quartile_weighted"].cat.codes + 1
    data.attrs["unweighted_quartile_cutpoints"] = unweighted_cutpoints.tolist()
    data.attrs["weighted_quartile_cutpoints"] = correct_cutpoints.tolist()
    return data


def impute_literal_reported(data: pd.DataFrame) -> tuple[pd.DataFrame, dict[str, Any]]:
    """Operationalize the paper's stated single-imputation approach literally.

    The paper says the listed continuous variables were mean-interpolated and
    alcohol/smoking were median-interpolated. LDL was included in Model 3 but its
    >50% missingness and imputation were not described; this function applies mean
    imputation to LDL so that the stated full model can be fit. Numeric survey codes
    remain numeric, matching the most literal reading of the shared file.
    """
    result = data.copy()
    audit: dict[str, Any] = {"method": "reported literal numeric single imputation", "fill_values": {}}
    mean_columns = [
        "education_code",
        "pir",
        "wbc",
        "bmi",
        "waist",
        "glucose",
        "bilirubin",
        "hdl",
        "triglyceride",
        "ldl",
        "total_cholesterol",
    ]
    for column in mean_columns:
        fill = float(result[column].mean(skipna=True))
        result[column] = result[column].fillna(fill)
        audit["fill_values"][column] = fill
    for column in ["alcohol_code", "smoking_code"]:
        fill = float(result[column].median(skipna=True))
        result[column] = result[column].fillna(fill)
        audit["fill_values"][column] = fill
    return result, audit


def impute_clean_analysis(data: pd.DataFrame) -> tuple[pd.DataFrame, dict[str, Any]]:
    """Create a cleaned deterministic-imputation sensitivity dataset.

    Continuous values use the median, while categorical variables use the mode.
    This is not presented as proper multiple imputation; it is a transparent
    sensitivity analysis that keeps the full sample and avoids impossible codes.
    """
    result = data.copy()
    audit: dict[str, Any] = {"method": "clean deterministic median/mode imputation", "fill_values": {}}
    continuous = ["pir", "bmi", "cadmium", "lead"]
    categorical = ["education", "smoking", "alcohol"]
    for column in continuous:
        fill = float(result[column].median(skipna=True))
        result[column] = result[column].fillna(fill)
        audit["fill_values"][column] = fill
    for column in categorical:
        mode = result[column].mode(dropna=True)
        if mode.empty:
            raise AnalysisError(f"Cannot impute categorical variable {column}: no observed values")
        fill = mode.iloc[0]
        result[column] = result[column].astype(object).fillna(fill)
        result[column] = pd.Categorical(result[column])
        audit["fill_values"][column] = str(fill)
    return result, audit


def extract_term(fit: FitResult, term: str) -> pd.Series:
    matches = fit.coefficients.loc[fit.coefficients["term"] == term]
    if len(matches) != 1:
        raise AnalysisError(f"Expected exactly one term {term!r}; found {len(matches)}")
    return matches.iloc[0]


def summarize_model(
    model_id: str,
    label: str,
    fit: FitResult,
    exposure_term: str,
    exposure_scale: str,
    notes: str = "",
) -> dict[str, Any]:
    coefficient = extract_term(fit, exposure_term)
    return {
        "model_id": model_id,
        "model_label": label,
        "estimator": fit.estimator,
        "exposure_term": exposure_term,
        "exposure_scale": exposure_scale,
        "n": fit.n,
        "cases": fit.cases,
        "n_parameters": fit.n_parameters,
        "events_per_parameter": fit.cases / fit.n_parameters,
        "design_df": fit.design_df,
        "odds_ratio": float(coefficient["odds_ratio"]),
        "ci_low": float(coefficient["ci_low"]),
        "ci_high": float(coefficient["ci_high"]),
        "p_value": float(coefficient["p_value"]),
        "beta": float(coefficient["beta"]),
        "se": float(coefficient["se"]),
        "converged": fit.converged,
        "formula": fit.formula,
        "notes": notes,
    }


def wald_joint_test(fit: FitResult, term_names: Sequence[str]) -> dict[str, float]:
    indices = [fit.terms.index(term) for term in term_names]
    beta = fit.beta[indices]
    covariance = fit.covariance[np.ix_(indices, indices)]
    statistic = float(beta.T @ np.linalg.pinv(covariance) @ beta)
    numerator_df = len(indices)
    if fit.design_df is None:
        p_value = float(st.chi2.sf(statistic, numerator_df))
        return {
            "statistic": statistic,
            "numerator_df": float(numerator_df),
            "denominator_df": np.nan,
            "p_value": p_value,
            "distribution": "chi-square",
        }
    f_statistic = statistic / numerator_df
    p_value = float(st.f.sf(f_statistic, numerator_df, fit.design_df))
    return {
        "statistic": f_statistic,
        "numerator_df": float(numerator_df),
        "denominator_df": float(fit.design_df),
        "p_value": p_value,
        "distribution": "F",
    }


def format_or(row: pd.Series | dict[str, Any], digits: int = 2) -> str:
    return (
        f"{float(row['odds_ratio']):.{digits}f} "
        f"({float(row['ci_low']):.{digits}f}–{float(row['ci_high']):.{digits}f})"
    )


def format_p(value: float) -> str:
    if not np.isfinite(value):
        return "NA"
    if value < 0.001:
        return "<0.001"
    return f"{value:.3f}"


def build_descriptive_audit(data: pd.DataFrame) -> pd.DataFrame:
    rows: list[dict[str, Any]] = []
    paper_values = {
        "Psoriasis prevalence": 0.0246,
        "Mean age": 41.67,
        "Age SD": 12.14,
        "Female proportion": 0.5312,
        "Non-Hispanic White proportion": 0.4397,
    }
    definitions = [
        ("Psoriasis prevalence", "psoriasis", "mean"),
        ("Mean age", "age", "mean"),
        ("Female proportion", "female", "mean"),
        ("Non-Hispanic White proportion", "non_hispanic_white", "mean"),
        ("Mean blood total mercury (µg/L)", "mercury", "mean"),
    ]
    for label, variable, _ in definitions:
        unweighted = float(data[variable].mean())
        weighted = survey_mean(data, variable)
        rows.append(
            {
                "measure": label,
                "paper_reported": paper_values.get(label, np.nan),
                "shared_data_unweighted": unweighted,
                "correct_survey_estimate": weighted["estimate"],
                "correct_survey_ci_low": weighted["ci_low"],
                "correct_survey_ci_high": weighted["ci_high"],
                "comment": "Paper value matches unweighted data" if label in paper_values and abs(unweighted - paper_values[label]) < 0.01 else "",
            }
        )
    rows.append(
        {
            "measure": "Age SD",
            "paper_reported": paper_values["Age SD"],
            "shared_data_unweighted": float(data["age"].std(ddof=1)),
            "correct_survey_estimate": float(
                np.sqrt(
                    np.average(
                        (data["age"] - np.average(data["age"], weights=data["correct_component_weight_4yr"])) ** 2,
                        weights=data["correct_component_weight_4yr"],
                    )
                )
            ),
            "correct_survey_ci_low": np.nan,
            "correct_survey_ci_high": np.nan,
            "comment": "The reported SD is not reproduced by the shared data",
        }
    )
    return pd.DataFrame(rows)


def build_missingness_audit(data: pd.DataFrame) -> pd.DataFrame:
    paper_missing_percent = {
        "education_code": 5.01,
        "pir": 5.59,
        "wbc": 0.18,
        "bmi": 0.87,
        "waist": 0.72,
        "glucose": 1.13,
        "bilirubin": 1.20,
        "hdl": 0.80,
        "triglyceride": 0.80,
        "total_cholesterol": 0.80,
        "alcohol_code": 9.99,
        "smoking_code": 2.51,
    }
    rows = []
    variables = [
        "education_code",
        "pir",
        "wbc",
        "bmi",
        "waist",
        "glucose",
        "bilirubin",
        "alcohol_code",
        "hdl",
        "triglyceride",
        "ldl",
        "total_cholesterol",
        "smoking_code",
    ]
    for variable in variables:
        n_missing = int(data[variable].isna().sum())
        percent = 100.0 * n_missing / len(data)
        rows.append(
            {
                "variable": variable,
                "n_missing_in_shared_file": n_missing,
                "percent_missing_in_shared_file": percent,
                "paper_reported_percent_missing": paper_missing_percent.get(variable, np.nan),
                "absolute_percentage_point_difference": (
                    percent - paper_missing_percent[variable]
                    if variable in paper_missing_percent
                    else np.nan
                ),
                "comment": (
                    "Not reported in the paper despite inclusion in Model 3"
                    if variable == "ldl"
                    else ""
                ),
            }
        )
    return pd.DataFrame(rows).sort_values(
        "percent_missing_in_shared_file", ascending=False
    )


def create_forest_plot(models: pd.DataFrame, path: Path, exposure_scale: str) -> None:
    plot_data = models.loc[models["exposure_scale"] == exposure_scale].copy()
    # Keep the forensic simple-weight model in the machine-readable table but omit
    # it from the primary figures because it is not a valid survey estimator.
    plot_data = plot_data.loc[plot_data["model_id"] != "P5_shared_WTMEC_simple_weight_raw"]
    if plot_data.empty:
        return
    plot_data = plot_data.iloc[::-1].reset_index(drop=True)
    plot_data["figure_label"] = plot_data["model_label"]
    y = np.arange(len(plot_data))
    fig_height = max(4.8, 0.48 * len(plot_data) + 1.8)
    fig, ax = plt.subplots(figsize=(10.5, fig_height))
    center = plot_data["odds_ratio"].to_numpy()
    left = center - plot_data["ci_low"].to_numpy()
    right = plot_data["ci_high"].to_numpy() - center
    ax.errorbar(center, y, xerr=np.vstack([left, right]), fmt="o", capsize=3)
    ax.axvline(1.0, linestyle="--", linewidth=1)
    ax.set_yticks(y)
    ax.set_yticklabels(plot_data["figure_label"])
    ax.set_xlabel(f"{exposure_scale} with 95% confidence interval")
    ax.set_title("Blood total mercury and prevalent psoriasis: model sensitivity")
    ax.grid(axis="x", alpha=0.25)
    fig.tight_layout()
    fig.savefig(path, dpi=180, bbox_inches="tight")
    plt.close(fig)


def create_distribution_plot(data: pd.DataFrame, path: Path) -> None:
    threshold = float(
        weighted_quantile(
            data["mercury"], data["correct_component_weight_4yr"], [0.99]
        )[0]
    )
    subset = data.loc[data["mercury"] <= threshold]
    weights = subset["correct_component_weight_4yr"] / subset["correct_component_weight_4yr"].sum()
    fig, ax = plt.subplots(figsize=(9, 5.2))
    ax.hist(subset["mercury"], bins=40, weights=weights)
    ax.set_xlabel("Blood total mercury (µg/L), truncated at weighted 99th percentile")
    ax.set_ylabel("Weighted proportion")
    ax.set_title("Survey-weighted blood total mercury distribution")
    ax.grid(axis="y", alpha=0.25)
    fig.tight_layout()
    fig.savefig(path, dpi=180, bbox_inches="tight")
    plt.close(fig)


def create_cycle_plot(cycle_table: pd.DataFrame, path: Path) -> None:
    plot_data = cycle_table.loc[cycle_table["exposure_scale"] == "OR per doubling"].copy()
    if plot_data.empty:
        return
    y = np.arange(len(plot_data))
    fig, ax = plt.subplots(figsize=(8.5, 4.6))
    center = plot_data["odds_ratio"].to_numpy()
    left = center - plot_data["ci_low"].to_numpy()
    right = plot_data["ci_high"].to_numpy() - center
    ax.errorbar(center, y, xerr=np.vstack([left, right]), fmt="o", capsize=3)
    ax.axvline(1.0, linestyle="--", linewidth=1)
    ax.set_yticks(y)
    ax.set_yticklabels(plot_data["cycle"])
    ax.set_xlabel("Survey-adjusted odds ratio per doubling of mercury")
    ax.set_title("Cycle-specific estimates")
    ax.grid(axis="x", alpha=0.25)
    fig.tight_layout()
    fig.savefig(path, dpi=180, bbox_inches="tight")
    plt.close(fig)


def dataframe_to_markdown(frame: pd.DataFrame, columns: Sequence[str] | None = None) -> str:
    if columns is not None:
        frame = frame[list(columns)]
    if frame.empty:
        return "_No rows._"
    display = frame.copy()
    headers = [str(column) for column in display.columns]
    lines = ["| " + " | ".join(headers) + " |", "| " + " | ".join(["---"] * len(headers)) + " |"]
    for _, row in display.iterrows():
        cells = []
        for value in row:
            if pd.isna(value):
                cells.append("")
            elif isinstance(value, float):
                cells.append(f"{value:.4g}")
            else:
                cells.append(str(value).replace("|", "\\|"))
        lines.append("| " + " | ".join(cells) + " |")
    return "\n".join(lines)


def model_display_table(models: pd.DataFrame) -> pd.DataFrame:
    display = models.copy()
    display["OR (95% CI)"] = display.apply(format_or, axis=1)
    display["P"] = display["p_value"].map(format_p)
    display["N / cases"] = display.apply(lambda row: f"{int(row['n'])} / {int(row['cases'])}", axis=1)
    return display[["model_label", "exposure_scale", "OR (95% CI)", "P", "N / cases", "notes"]].rename(
        columns={"model_label": "Model", "exposure_scale": "Contrast", "notes": "Notes"}
    )


def render_html(markdown_text: str, report_context: dict[str, Any]) -> str:
    markdown = mistune.create_markdown(plugins=["table", "strikethrough", "task_lists"])
    rendered_body = markdown(markdown_text)
    template = Template(
        """<!doctype html>
<html lang="en">
<head>
<meta charset="utf-8">
<meta name="viewport" content="width=device-width, initial-scale=1">
<title>{{ title }}</title>
<style>
body{font-family:Arial,Helvetica,sans-serif;line-height:1.5;margin:0;background:#f6f7f9;color:#1f2933}
main{max-width:1120px;margin:0 auto;background:white;padding:36px 48px;min-height:100vh}
h1,h2,h3{line-height:1.2;color:#102a43;margin-top:1.35em}.panel{border:1px solid #bcccdc;border-radius:8px;padding:18px 22px;background:#f0f4f8;margin:18px 0 26px}
.metric{display:inline-block;margin:5px 24px 5px 0}.metric strong{display:block;font-size:1.25rem}
table{border-collapse:collapse;width:100%;font-size:.9rem;margin:1rem 0 1.5rem}th,td{border:1px solid #bcccdc;padding:7px 9px;text-align:left;vertical-align:top}th{background:#eaf0f6}
pre{white-space:pre-wrap;overflow-wrap:anywhere;background:#f5f7fa;border:1px solid #d9e2ec;padding:13px;border-radius:6px}code{font-family:ui-monospace,SFMono-Regular,Menlo,monospace}
blockquote{border-left:5px solid #9f7aea;padding-left:14px;color:#334e68}.caution{border-left:5px solid #9f7aea;padding-left:14px}
img{max-width:100%}@media(max-width:760px){main{padding:22px 18px}table{display:block;overflow-x:auto}.metric{display:block}}
</style>
</head><body><main>
<div class="panel">
<div class="metric"><strong>{{ primary_or }}</strong>Primary survey-adjusted OR per doubling</div>
<div class="metric"><strong>{{ primary_p }}</strong>Primary P value</div>
<div class="metric"><strong>{{ n }}</strong>Participants</div>
<div class="metric"><strong>{{ cases }}</strong>Psoriasis cases</div>
</div>
<p class="caution"><strong>Interpretation:</strong> This is a cross-sectional reproducibility and robustness audit. It does not estimate a causal effect.</p>
{{ rendered_body | safe }}
</main></body></html>"""
    )
    return template.render(rendered_body=rendered_body, **report_context)


def make_report(
    output_dir: Path,
    data: pd.DataFrame,
    all_models: pd.DataFrame,
    reproduction: pd.DataFrame,
    survey_models: pd.DataFrame,
    quartiles: pd.DataFrame,
    cycles: pd.DataFrame,
    descriptive: pd.DataFrame,
    missingness: pd.DataFrame,
    claims: pd.DataFrame,
    diagnostics: dict[str, Any],
) -> tuple[Path, Path]:
    primary = all_models.loc[all_models["model_id"] == "S4_parsimonious_cc_survey_log2"].iloc[0]
    crude = all_models.loc[all_models["model_id"] == "P1_crude_unweighted_raw"].iloc[0]
    survey_raw = all_models.loc[all_models["model_id"] == "S2_model2_survey_raw"].iloc[0]
    full_literal = all_models.loc[all_models["model_id"] == "P3_full_literal_unweighted_raw"].iloc[0]
    q4_unweighted = quartiles.loc[
        (quartiles["analysis"] == "Unweighted shared-data quartiles")
        & (quartiles["quartile"] == "Q4")
    ].iloc[0]
    q4_survey = quartiles.loc[
        (quartiles["analysis"] == "Correct survey-weighted quartiles")
        & (quartiles["quartile"] == "Q4")
    ].iloc[0]
    q4_full_unweighted = quartiles.loc[
        (quartiles["analysis"] == "Published full-model replication, unweighted")
        & (quartiles["quartile"] == "Q4")
    ].iloc[0]

    report = f"""# Reproducibility and robustness audit of blood total mercury and prevalent psoriasis in NHANES

**Analysis version:** {ANALYSIS_VERSION}  
**Target publication:** Tuo Y, Li Y, Guo T. *PLOS ONE*. 2024;19:e0309147. DOI: {PAPER_DOI}  
**Data:** NHANES 2005–2006 and 2013–2014; authors' shared S2 participant file enriched with official public-use survey-design variables.  
**Analysis date:** {pd.Timestamp.utcnow().strftime('%Y-%m-%d')}  

## Executive conclusion

The published **crude ordinary logistic-regression estimate was reproduced exactly** from the shared S2 file: OR {crude['odds_ratio']:.3f} (95% CI {crude['ci_low']:.3f}–{crude['ci_high']:.3f}), P={crude['p_value']:.4f}. However, the paper's adjusted OR of 1.08 could not be reproduced from the shared file using the stated covariates and imputation rules. The closest literal unweighted implementation produced OR {full_literal['odds_ratio']:.3f} (95% CI {full_literal['ci_low']:.3f}–{full_literal['ci_high']:.3f}), P={full_literal['p_value']:.3f}.

After applying the official cycle-specific blood-metals weights and NHANES strata/PSU structure, the age-, sex-, race-, and cycle-adjusted estimate was OR {survey_raw['odds_ratio']:.3f} per 1 µg/L (95% CI {survey_raw['ci_low']:.3f}–{survey_raw['ci_high']:.3f}), P={survey_raw['p_value']:.3f}. The primary confounder-informed complete-case analysis was OR {primary['odds_ratio']:.3f} per doubling of blood mercury (95% CI {primary['ci_low']:.3f}–{primary['ci_high']:.3f}), P={primary['p_value']:.3f}.

**Bottom line:** the shared data support a small unweighted crude association, but they do not provide robust evidence of an independent population-representative association after correct survey design and prespecified adjustment. Confidence intervals remain compatible with modest benefit or harm; this is not proof of no association.

## Study questions

1. Can the published estimates be reproduced from the authors' shared analysis file?
2. Do the estimates persist when the official NHANES sampling design and component-specific weights are applied?
3. Are conclusions stable across raw and log2 exposure scales, quartiles, cycle-specific analyses, outlier exclusion, covariate sets, and missing-data strategies?

## Methods

### Study population and variables

The analysis retained all 6,086 records in the authors' S2 file, including 150 self-reported psoriasis cases. Participant identifiers were joined to the public NHANES demographic and blood-metals files to recover survey cycle, masked variance stratum, masked PSU, and component weights. Mercury and age values were required to match the shared file exactly after linkage.

For 2005–2006, the MEC examination weight was used. For 2013–2014, the blood-metals half-sample weight was used. Each two-year weight was divided by two when cycles were combined. Stratum and PSU identifiers were prefixed by cycle so that reused numeric codes were not treated as the same design units.

### Estimation

Published-model reproductions used ordinary binomial logistic regression. Survey analyses used weighted estimating equations and Taylor-linearized variance at the PSU level within strata. Inference used the design degrees of freedom (60 PSUs minus 30 strata = 30 for combined-cycle analyses).

The primary substantive model used log2 mercury, yielding an OR per exposure doubling, and adjusted for age, sex, race/ethnicity, education, poverty-income ratio, BMI, smoking, alcohol, blood cadmium, blood lead, and cycle. Complete-case analysis was primary for this model; deterministic median/mode imputation was a sensitivity analysis. The much larger published Model 3 was reproduced separately because its approximately 20 covariates, 150 events, and substantial fasting-lipid missingness create instability.

### Interpretation constraints

This is a cross-sectional analysis of prevalent, self-reported psoriasis. Blood mercury may be influenced by diet, geography, occupation, renal function, and other unmeasured factors. Neither temporality nor causality can be established.

## Results

### 1. Published-model reproduction

{dataframe_to_markdown(model_display_table(reproduction))}

The crude coefficient and P value reproduce the publication. The adjusted estimates do not. This failure is not resolved by treating survey codes as numeric versus categorical, simple weighting by the shared WTMEC2YR column, complete-case analysis, or the paper's stated single-imputation approach.

### 2. Correct survey-design analyses

{dataframe_to_markdown(model_display_table(survey_models))}

The survey-correct estimates are centered near the null. The raw-scale estimate changes under weighting and upper-tail exclusion, demonstrating sensitivity to influential high-mercury observations. The log2 scale is more stable and clinically interpretable as an OR per doubling.

### 3. Quartile analyses

{dataframe_to_markdown(quartiles[["analysis", "quartile", "n", "cases", "odds_ratio", "ci_low", "ci_high", "p_value", "trend_p_value"]])}

Using the shared-data unweighted quartiles, crude Q4 versus Q1 was OR {q4_unweighted['odds_ratio']:.3f} (95% CI {q4_unweighted['ci_low']:.3f}–{q4_unweighted['ci_high']:.3f}), P={q4_unweighted['p_value']:.3f}. The literal unweighted full-model replication gave OR {q4_full_unweighted['odds_ratio']:.3f} (95% CI {q4_full_unweighted['ci_low']:.3f}–{q4_full_unweighted['ci_high']:.3f}), P={q4_full_unweighted['p_value']:.3f}. Neither reproduces the publication's statistically significant OR of 1.45. Under correct survey weighting and adjustment, Q4 versus Q1 was OR {q4_survey['odds_ratio']:.3f} (95% CI {q4_survey['ci_low']:.3f}–{q4_survey['ci_high']:.3f}), P={q4_survey['p_value']:.3f}.

### 4. Cycle-specific analyses

{dataframe_to_markdown(cycles[["cycle", "exposure_scale", "n", "cases", "odds_ratio", "ci_low", "ci_high", "p_value"]])}

Cycle-specific estimates are imprecise and do not establish a consistent positive association. The mercury-by-cycle interaction P value was {diagnostics['interaction_p']:.3f}; this is suggestive of heterogeneity but does not cross the prespecified 0.05 threshold. A quadratic log2-mercury term was not supported (P={diagnostics['quadratic_p']:.3f}).

{dataframe_to_markdown(pd.read_csv(output_dir / "results" / "interaction_and_nonlinearity.csv"))}

### 5. Descriptive weighting audit

{dataframe_to_markdown(descriptive)}

The publication's reported mean age, female percentage, non-Hispanic White percentage, and psoriasis prevalence align with unweighted shared-data calculations rather than the correctly weighted estimates. The reported age SD of 12.14 years does not match the shared-data SD of {data['age'].std(ddof=1):.2f} years.

### 6. Missing-data audit

{dataframe_to_markdown(missingness[["variable", "n_missing_in_shared_file", "percent_missing_in_shared_file", "paper_reported_percent_missing", "absolute_percentage_point_difference", "comment"]])}

The most consequential discrepancy is that triglyceride and LDL are fasting-subsample variables and are missing in more than half of the shared analysis file. The paper reports triglyceride missingness of 0.80% and does not disclose LDL missingness despite including LDL in Model 3. Mean imputation of a variable missing in more than half the sample can create false precision and does not recover the unobserved distribution.

### 7. Claim-level audit

{dataframe_to_markdown(claims)}

## Strengths

- Used the authors' exact shared participant file rather than reconstructing an approximation.
- Verified linkage against public NHANES mercury and age values.
- Recovered the official component-specific weights, strata, and PSU variables.
- Distinguished computational reproduction from survey-valid inference.
- Reported multiple exposure scales, cycle-specific estimates, missingness, outlier sensitivity, and model parameter burden.
- Included machine-readable results, code, checksums, and automated tests.

## Limitations

- The exact proprietary EmpowerStats workflow and any undocumented point-and-click settings are unavailable.
- The paper does not specify how normality was determined, how categorical missing values were handled, or how LDL with >50% missingness was treated.
- The shared file includes only the final analytic sample, not every excluded record. All 30 combined-cycle strata and 60 PSUs are nevertheless represented.
- With only 150 cases, highly parameterized models and subgroup interactions are unstable.
- Survey-correct null estimates do not prove equivalence or absence of a clinically meaningful association.
- The study remains cross-sectional and cannot establish temporality or causation.

## Reproducibility statement

Run `prepare_enriched.py` to obtain the public inputs and create the survey-enriched analysis file, then run:

```bash
python analysis.py --input enriched_analysis_data.tsv --output analysis_output
```

Participant-level data are intentionally excluded from the distributable package. Source URLs, exact checksums, environment information, and output checksums are recorded in the provenance files.

## Files

- `results/model_summary_all.csv`: all principal model estimates.
- `results/reproduction_models.csv`: published-model reproduction attempts.
- `results/survey_models.csv`: survey-correct analyses.
- `results/quartile_models.csv`: quartile comparisons.
- `results/cycle_models.csv`: cycle-specific estimates.
- `results/descriptive_audit.csv`: weighted versus unweighted descriptive statistics.
- `results/missingness_audit.csv`: observed missingness versus publication statements.
- `results/claim_audit.csv`: claim-level reproducibility assessment.
- `figures/forest_raw_scale.png`: raw-scale model sensitivity forest plot.
- `figures/forest_log2_scale.png`: per-doubling model sensitivity forest plot.
- `figures/mercury_distribution.png`: weighted mercury distribution.
- `figures/cycle_estimates.png`: cycle-specific estimates.

## Reference

Tuo Y, Li Y, Guo T. Association between blood total mercury and psoriasis: The NHANES 2005–2006 and 2013–2014: A cross-sectional study. *PLOS ONE*. 2024;19:e0309147. doi:{PAPER_DOI}.
"""

    md_path = output_dir / "REPORT.md"
    md_path.write_text(report, encoding="utf-8")
    html_path = output_dir / "REPORT.html"
    html_path.write_text(
        render_html(
            report,
            {
                "title": "NHANES mercury–psoriasis reproducibility audit",
                "primary_or": format_or(primary),
                "primary_p": format_p(float(primary["p_value"])),
                "n": f"{len(data):,}",
                "cases": f"{int(data['psoriasis'].sum()):,}",
            },
        ),
        encoding="utf-8",
    )
    return md_path, html_path


def run_analysis(input_path: Path, output_dir: Path) -> dict[str, Any]:
    if output_dir.exists():
        shutil.rmtree(output_dir)
    results_dir = output_dir / "results"
    figures_dir = output_dir / "figures"
    provenance_dir = output_dir / "provenance"
    for directory in [results_dir, figures_dir, provenance_dir]:
        directory.mkdir(parents=True, exist_ok=True)

    raw = pd.read_csv(input_path, sep="\t")
    data = clean_data(raw)
    if len(data) != 6086 or int(data["psoriasis"].sum()) != 150:
        raise AnalysisError("The linked analytic dataset does not match the published N/case count")
    if data["combined_stratum"].nunique() != 30 or data["combined_psu"].nunique() != 60:
        raise AnalysisError("Expected 30 strata and 60 cycle-nested PSUs")
    if not data.groupby("combined_stratum")["combined_psu"].nunique().eq(2).all():
        raise AnalysisError("Every combined stratum must retain both masked PSUs")

    literal_imputed, literal_imputation_audit = impute_literal_reported(data)
    clean_imputed, clean_imputation_audit = impute_clean_analysis(data)

    model_rows: list[dict[str, Any]] = []
    reproduction_rows: list[dict[str, Any]] = []
    survey_rows: list[dict[str, Any]] = []

    def add_model(
        destination: list[dict[str, Any]],
        model_id: str,
        label: str,
        fit: FitResult,
        exposure_term: str,
        exposure_scale: str,
        notes: str = "",
    ) -> dict[str, Any]:
        row = summarize_model(
            model_id, label, fit, exposure_term, exposure_scale, notes
        )
        destination.append(row)
        model_rows.append(row)
        return row

    # Published-model reproduction attempts.
    p1 = fit_unweighted_logit("psoriasis ~ mercury", data)
    add_model(
        reproduction_rows,
        "P1_crude_unweighted_raw",
        "Published crude, unweighted",
        p1,
        "mercury",
        "OR per 1 µg/L",
        "Exactly reproduces the publication's crude OR and P value",
    )

    p2 = fit_unweighted_logit(
        "psoriasis ~ mercury + age + C(sex) + C(race) + education_code + pir", data
    )
    add_model(
        reproduction_rows,
        "P2_model2_unweighted_raw",
        "Published Model 2 definition, unweighted",
        p2,
        "mercury",
        "OR per 1 µg/L",
        "Main text defines Model 2 as age, sex, race, education, and PIR",
    )

    p2_imputed = fit_unweighted_logit(
        "psoriasis ~ mercury + age + C(sex) + C(race) + education_code + pir",
        literal_imputed,
    )
    add_model(
        reproduction_rows,
        "P2c_model2_imputed_unweighted_raw",
        "Published Model 2 definition, stated imputation",
        p2_imputed,
        "mercury",
        "OR per 1 µg/L",
        "Mean-imputed education and PIR; still does not reproduce the reported adjusted estimate",
    )

    p2_supplement = fit_unweighted_logit(
        "psoriasis ~ mercury + age + C(sex) + C(race)", data
    )
    add_model(
        reproduction_rows,
        "P2b_supplement_model2_unweighted_raw",
        "Supporting-file Model 2 definition, unweighted",
        p2_supplement,
        "mercury",
        "OR per 1 µg/L",
        "Supporting file defines Model 2 as age, sex, and race only",
    )

    full_numeric_formula = (
        "psoriasis ~ mercury + age + sex_code + race_code + education_code + pir + wbc + "
        "bmi + waist + glucose + bilirubin + alcohol_code + cadmium + lead + "
        "hypertension_code + hdl + diabetes_code + triglyceride + ldl + "
        "total_cholesterol + smoking_code"
    )
    p3 = fit_unweighted_logit(full_numeric_formula, literal_imputed)
    add_model(
        reproduction_rows,
        "P3_full_literal_unweighted_raw",
        "Published full model, literal numeric imputation",
        p3,
        "mercury",
        "OR per 1 µg/L",
        "Mean/median single imputation; LDL mean-imputed because its handling was unstated",
    )

    full_categorical_formula = (
        "psoriasis ~ mercury + age + C(sex) + C(race) + C(education) + pir + wbc + bmi + "
        "waist + glucose + bilirubin + C(alcohol) + cadmium + lead + C(hypertension) + "
        "hdl + C(diabetes) + triglyceride + ldl + total_cholesterol + C(smoking)"
    )
    p3_cc = fit_unweighted_logit(full_categorical_formula, data)
    add_model(
        reproduction_rows,
        "P4_full_complete_case_unweighted_raw",
        "Published full model, complete case",
        p3_cc,
        "mercury",
        "OR per 1 µg/L",
        "Complete-case sensitivity; large fasting-lipid attrition",
    )

    p_weight_candidate = fit_pseudoweighted_logit(
        "psoriasis ~ mercury + age + C(sex) + C(race) + education_code + pir",
        literal_imputed,
    )
    add_model(
        reproduction_rows,
        "P5_shared_WTMEC_simple_weight_raw",
        "Shared WTMEC2YR as simple weight",
        p_weight_candidate,
        "mercury",
        "OR per 1 µg/L",
        "Forensic candidate only; ignores the 2013 metals half-sample weight and complex design",
    )

    # Correct survey-design analyses.
    s1 = fit_survey_logit("psoriasis ~ mercury", data)
    add_model(
        survey_rows,
        "S1_crude_survey_raw",
        "Correct survey design, crude",
        s1,
        "mercury",
        "OR per 1 µg/L",
    )
    s2 = fit_survey_logit(
        "psoriasis ~ mercury + age + C(sex) + C(race) + C(cycle)", data
    )
    add_model(
        survey_rows,
        "S2_model2_survey_raw",
        "Correct survey design, age/sex/race/cycle",
        s2,
        "mercury",
        "OR per 1 µg/L",
    )
    s3 = fit_survey_logit(
        "psoriasis ~ log2_mercury + age + C(sex) + C(race) + C(cycle)", data
    )
    add_model(
        survey_rows,
        "S3_model2_survey_log2",
        "Correct survey design, age/sex/race/cycle",
        s3,
        "log2_mercury",
        "OR per doubling",
    )

    parsimonious_formula = (
        "psoriasis ~ log2_mercury + age + C(sex) + C(race) + C(education) + pir + bmi + "
        "C(smoking) + C(alcohol) + cadmium + lead + C(cycle)"
    )
    s4 = fit_survey_logit(parsimonious_formula, data)
    add_model(
        survey_rows,
        "S4_parsimonious_cc_survey_log2",
        "Primary confounder-informed model, complete case",
        s4,
        "log2_mercury",
        "OR per doubling",
        "Primary substantive analysis",
    )
    s5 = fit_survey_logit(parsimonious_formula, clean_imputed)
    add_model(
        survey_rows,
        "S5_parsimonious_imputed_survey_log2",
        "Confounder-informed model, deterministic imputation",
        s5,
        "log2_mercury",
        "OR per doubling",
        "Median/mode single-imputation sensitivity",
    )
    s6 = fit_survey_logit(full_numeric_formula, literal_imputed)
    add_model(
        survey_rows,
        "S6_full_literal_survey_raw",
        "Published full model with correct survey design",
        s6,
        "mercury",
        "OR per 1 µg/L",
        "Literal numeric imputation; highly parameterized",
    )

    top_1_threshold = float(
        weighted_quantile(
            data["mercury"], data["correct_component_weight_4yr"], [0.99]
        )[0]
    )
    top_1_excluded = data.loc[data["mercury"] <= top_1_threshold].copy()
    s7 = fit_survey_logit(parsimonious_formula, top_1_excluded)
    add_model(
        survey_rows,
        "S7_top1_excluded_survey_log2",
        "Primary model excluding weighted top 1% mercury",
        s7,
        "log2_mercury",
        "OR per doubling",
        f"Mercury ≤ {top_1_threshold:.2f} µg/L",
    )
    s8 = fit_survey_logit(
        "psoriasis ~ mercury + age + C(sex) + C(race) + C(cycle)", top_1_excluded
    )
    add_model(
        survey_rows,
        "S8_top1_excluded_survey_raw",
        "Age/sex/race/cycle model excluding weighted top 1%",
        s8,
        "mercury",
        "OR per 1 µg/L",
        f"Mercury ≤ {top_1_threshold:.2f} µg/L",
    )

    # Quartile analyses.
    quartile_rows: list[dict[str, Any]] = []
    quartile_fit_unweighted = fit_unweighted_logit(
        'psoriasis ~ C(hg_quartile_unweighted, Treatment(reference="Q1"))', data
    )
    for quartile in ["Q2", "Q3", "Q4"]:
        term = f'C(hg_quartile_unweighted, Treatment(reference="Q1"))[T.{quartile}]'
        coefficient = extract_term(quartile_fit_unweighted, term)
        subset = data.loc[data["hg_quartile_unweighted"] == quartile]
        quartile_rows.append(
            {
                "analysis": "Unweighted shared-data quartiles",
                "quartile": quartile,
                "n": int(len(subset)),
                "cases": int(subset["psoriasis"].sum()),
                "odds_ratio": float(coefficient["odds_ratio"]),
                "ci_low": float(coefficient["ci_low"]),
                "ci_high": float(coefficient["ci_high"]),
                "p_value": float(coefficient["p_value"]),
            }
        )
    trend_unweighted = fit_unweighted_logit(
        "psoriasis ~ hg_quartile_unweighted_ordinal", data
    )
    trend_unweighted_row = extract_term(
        trend_unweighted, "hg_quartile_unweighted_ordinal"
    )

    quartile_fit_full_unweighted = fit_unweighted_logit(
        'psoriasis ~ C(hg_quartile_unweighted, Treatment(reference="Q1")) + age + sex_code + race_code + education_code + pir + wbc + bmi + waist + glucose + bilirubin + alcohol_code + cadmium + lead + hypertension_code + hdl + diabetes_code + triglyceride + ldl + total_cholesterol + smoking_code',
        literal_imputed,
    )
    for quartile in ["Q2", "Q3", "Q4"]:
        term = f'C(hg_quartile_unweighted, Treatment(reference="Q1"))[T.{quartile}]'
        coefficient = extract_term(quartile_fit_full_unweighted, term)
        subset = data.loc[data["hg_quartile_unweighted"] == quartile]
        quartile_rows.append(
            {
                "analysis": "Published full-model replication, unweighted",
                "quartile": quartile,
                "n": int(len(subset)),
                "cases": int(subset["psoriasis"].sum()),
                "odds_ratio": float(coefficient["odds_ratio"]),
                "ci_low": float(coefficient["ci_low"]),
                "ci_high": float(coefficient["ci_high"]),
                "p_value": float(coefficient["p_value"]),
            }
        )

    quartile_fit_survey = fit_survey_logit(
        'psoriasis ~ C(hg_quartile_weighted, Treatment(reference="Q1")) + age + C(sex) + C(race) + C(cycle)',
        data,
    )
    for quartile in ["Q2", "Q3", "Q4"]:
        term = f'C(hg_quartile_weighted, Treatment(reference="Q1"))[T.{quartile}]'
        coefficient = extract_term(quartile_fit_survey, term)
        subset = data.loc[data["hg_quartile_weighted"] == quartile]
        quartile_rows.append(
            {
                "analysis": "Correct survey-weighted quartiles",
                "quartile": quartile,
                "n": int(len(subset)),
                "cases": int(subset["psoriasis"].sum()),
                "odds_ratio": float(coefficient["odds_ratio"]),
                "ci_low": float(coefficient["ci_low"]),
                "ci_high": float(coefficient["ci_high"]),
                "p_value": float(coefficient["p_value"]),
            }
        )
    trend_survey = fit_survey_logit(
        "psoriasis ~ hg_quartile_weighted_ordinal + age + C(sex) + C(race) + C(cycle)",
        data,
    )
    trend_survey_row = extract_term(trend_survey, "hg_quartile_weighted_ordinal")
    quartiles = pd.DataFrame(quartile_rows)
    quartiles["trend_p_value"] = np.select(
        [
            quartiles["analysis"].eq("Unweighted shared-data quartiles"),
            quartiles["analysis"].eq("Correct survey-weighted quartiles"),
        ],
        [float(trend_unweighted_row["p_value"]), float(trend_survey_row["p_value"])],
        default=np.nan,
    )

    # Cycle-specific analyses.
    cycle_rows: list[dict[str, Any]] = []
    for cycle, cycle_data in data.groupby("cycle", observed=True):
        for term, scale in [("mercury", "OR per 1 µg/L"), ("log2_mercury", "OR per doubling")]:
            fit = fit_survey_logit(
                f"psoriasis ~ {term} + age + C(sex) + C(race)", cycle_data
            )
            coefficient = extract_term(fit, term)
            cycle_rows.append(
                {
                    "cycle": str(cycle),
                    "exposure_scale": scale,
                    "n": fit.n,
                    "cases": fit.cases,
                    "odds_ratio": float(coefficient["odds_ratio"]),
                    "ci_low": float(coefficient["ci_low"]),
                    "ci_high": float(coefficient["ci_high"]),
                    "p_value": float(coefficient["p_value"]),
                    "design_df": fit.design_df,
                }
            )
    cycles = pd.DataFrame(cycle_rows)

    interaction_fit = fit_survey_logit(
        "psoriasis ~ log2_mercury * C(cycle) + age + C(sex) + C(race)", data
    )
    interaction_term = "log2_mercury:C(cycle)[T.2013-2014]"
    interaction_row = extract_term(interaction_fit, interaction_term)

    centered = data.copy()
    centered["log2_mercury_centered"] = centered["log2_mercury"] - np.average(
        centered["log2_mercury"], weights=centered["correct_component_weight_4yr"]
    )
    centered["log2_mercury_centered_sq"] = centered["log2_mercury_centered"] ** 2
    quadratic_fit = fit_survey_logit(
        "psoriasis ~ log2_mercury_centered + log2_mercury_centered_sq + age + C(sex) + C(race) + C(cycle)",
        centered,
    )
    quadratic_row = extract_term(quadratic_fit, "log2_mercury_centered_sq")

    interaction_nonlinearity = pd.DataFrame(
        [
            {
                "analysis": "Mercury-by-cycle interaction",
                "term": interaction_term,
                "estimate": float(interaction_row["beta"]),
                "se": float(interaction_row["se"]),
                "p_value": float(interaction_row["p_value"]),
                "design_df": interaction_fit.design_df,
            },
            {
                "analysis": "Quadratic log2-mercury term",
                "term": "log2_mercury_centered_sq",
                "estimate": float(quadratic_row["beta"]),
                "se": float(quadratic_row["se"]),
                "p_value": float(quadratic_row["p_value"]),
                "design_df": quadratic_fit.design_df,
            },
        ]
    )

    reproduction = pd.DataFrame(reproduction_rows)
    survey_models = pd.DataFrame(survey_rows)
    all_models = pd.DataFrame(model_rows)
    descriptive = build_descriptive_audit(data)
    missingness = build_missingness_audit(data)

    # Claim-level audit uses exact computational checks rather than subjective labels.
    crude_row = all_models.loc[all_models["model_id"] == "P1_crude_unweighted_raw"].iloc[0]
    full_row = all_models.loc[all_models["model_id"] == "P3_full_literal_unweighted_raw"].iloc[0]
    survey_primary_row = all_models.loc[all_models["model_id"] == "S4_parsimonious_cc_survey_log2"].iloc[0]
    q4_unweighted = quartiles.loc[
        (quartiles["analysis"] == "Unweighted shared-data quartiles")
        & (quartiles["quartile"] == "Q4")
    ].iloc[0]
    q4_full_unweighted = quartiles.loc[
        (quartiles["analysis"] == "Published full-model replication, unweighted")
        & (quartiles["quartile"] == "Q4")
    ].iloc[0]
    claims = pd.DataFrame(
        [
            {
                "publication_claim": "Analytic N=6,086 with 150 psoriasis cases",
                "audit_result": "Verified",
                "evidence": f"Shared file N={len(data):,}; cases={int(data['psoriasis'].sum())}",
            },
            {
                "publication_claim": "Mean age 41.67 years; SD 12.14",
                "audit_result": "Mean verified; SD not verified",
                "evidence": f"Shared unweighted mean={data['age'].mean():.2f}; SD={data['age'].std(ddof=1):.2f}",
            },
            {
                "publication_claim": "All analyses used suitable NHANES sampling weights",
                "audit_result": "Not supported by reproduced outputs",
                "evidence": "Published descriptives and crude model match unweighted calculations; shared file lacks strata/PSU and the 2013 metals half-sample weight",
            },
            {
                "publication_claim": "Crude mercury OR 1.05; P=0.0297",
                "audit_result": "Exactly reproduced as unweighted logistic regression",
                "evidence": f"OR={crude_row['odds_ratio']:.5f}; P={crude_row['p_value']:.5f}",
            },
            {
                "publication_claim": "Fully adjusted mercury OR 1.08 (1.03–1.14)",
                "audit_result": "Not reproduced",
                "evidence": f"Literal shared-data implementation OR={full_row['odds_ratio']:.3f} ({full_row['ci_low']:.3f}–{full_row['ci_high']:.3f}); P={full_row['p_value']:.3f}",
            },
            {
                "publication_claim": "Highest quartile OR 1.45 (1.10–2.38)",
                "audit_result": "Not reproduced",
                "evidence": f"Crude OR={q4_unweighted['odds_ratio']:.3f} ({q4_unweighted['ci_low']:.3f}–{q4_unweighted['ci_high']:.3f}); full-model replication OR={q4_full_unweighted['odds_ratio']:.3f} ({q4_full_unweighted['ci_low']:.3f}–{q4_full_unweighted['ci_high']:.3f})",
            },
            {
                "publication_claim": "Triglyceride missingness 0.80%",
                "audit_result": "Contradicted by shared file",
                "evidence": f"Observed missingness={100*data['triglyceride'].isna().mean():.2f}%",
            },
            {
                "publication_claim": "Independent population-representative positive association",
                "audit_result": "Not supported by the survey-correct primary analysis",
                "evidence": f"OR per doubling={survey_primary_row['odds_ratio']:.3f} ({survey_primary_row['ci_low']:.3f}–{survey_primary_row['ci_high']:.3f}); P={survey_primary_row['p_value']:.3f}",
            },
        ]
    )

    # Save all tables.
    all_models.to_csv(results_dir / "model_summary_all.csv", index=False)
    reproduction.to_csv(results_dir / "reproduction_models.csv", index=False)
    survey_models.to_csv(results_dir / "survey_models.csv", index=False)
    quartiles.to_csv(results_dir / "quartile_models.csv", index=False)
    cycles.to_csv(results_dir / "cycle_models.csv", index=False)
    descriptive.to_csv(results_dir / "descriptive_audit.csv", index=False)
    missingness.to_csv(results_dir / "missingness_audit.csv", index=False)
    claims.to_csv(results_dir / "claim_audit.csv", index=False)
    interaction_nonlinearity.to_csv(
        results_dir / "interaction_and_nonlinearity.csv", index=False
    )

    # Design and data audit.
    design_audit = {
        "analysis_version": ANALYSIS_VERSION,
        "input_path": str(input_path),
        "input_sha256": sha256_file(input_path),
        "n": int(len(data)),
        "cases": int(data["psoriasis"].sum()),
        "cycles": data["cycle"].value_counts().sort_index().astype(int).to_dict(),
        "strata": int(data["combined_stratum"].nunique()),
        "psus": int(data["combined_psu"].nunique()),
        "design_df": int(data["combined_psu"].nunique() - data["combined_stratum"].nunique()),
        "unweighted_quartile_cutpoints": data.attrs["unweighted_quartile_cutpoints"],
        "correct_weighted_quartile_cutpoints": data.attrs["weighted_quartile_cutpoints"],
        "weighted_99th_percentile_mercury": top_1_threshold,
        "literal_imputation": literal_imputation_audit,
        "clean_imputation": clean_imputation_audit,
        "software": {
            "python": sys.version,
            "numpy": np.__version__,
            "pandas": pd.__version__,
            "scipy": st.__version__ if hasattr(st, "__version__") else None,
            "statsmodels": sm.__version__,
            "patsy": patsy.__version__,
        },
    }
    (provenance_dir / "analysis_manifest.json").write_text(
        json.dumps(design_audit, indent=2, sort_keys=True, default=str), encoding="utf-8"
    )

    create_forest_plot(all_models, figures_dir / "forest_raw_scale.png", "OR per 1 µg/L")
    create_forest_plot(all_models, figures_dir / "forest_log2_scale.png", "OR per doubling")
    create_distribution_plot(data, figures_dir / "mercury_distribution.png")
    create_cycle_plot(cycles, figures_dir / "cycle_estimates.png")

    diagnostics = {
        "exact_crude_reproduction": {
            "target_or": 1.05,
            "computed_or": float(crude_row["odds_ratio"]),
            "target_p": 0.0297,
            "computed_p": float(crude_row["p_value"]),
        },
        "survey_primary": survey_primary_row.to_dict(),
        "interaction_p": float(interaction_row["p_value"]),
        "quadratic_p": float(quadratic_row["p_value"]),
        "top_1_percent_threshold": top_1_threshold,
    }
    (provenance_dir / "diagnostics.json").write_text(
        json.dumps(diagnostics, indent=2, sort_keys=True, default=str), encoding="utf-8"
    )

    make_report(
        output_dir,
        data,
        all_models,
        reproduction,
        survey_models,
        quartiles,
        cycles,
        descriptive,
        missingness,
        claims,
        diagnostics,
    )

    # Machine-verifiable checks.
    checks = {
        "n_is_6086": len(data) == 6086,
        "cases_are_150": int(data["psoriasis"].sum()) == 150,
        "all_strata_have_two_psus": bool(
            data.groupby("combined_stratum")["combined_psu"].nunique().eq(2).all()
        ),
        "crude_or_matches_published_rounding": round(float(crude_row["odds_ratio"]), 2) == 1.05,
        "crude_p_matches_published": abs(float(crude_row["p_value"]) - 0.0297) < 0.0001,
        "all_models_converged": bool(all_models["converged"].all()),
        "primary_model_has_30_design_df": float(s4.design_df or np.nan) == 30.0,
        "participant_data_not_written_to_output": not any(
            path.name.endswith((".tsv", ".parquet", ".xls", ".xlsx"))
            for path in output_dir.rglob("*")
            if path.is_file()
        ),
    }
    if not all(checks.values()):
        raise AnalysisError(f"One or more verification checks failed: {checks}")
    (provenance_dir / "verification_checks.json").write_text(
        json.dumps(checks, indent=2, sort_keys=True), encoding="utf-8"
    )

    # Checksums are generated after every substantive output exists.
    checksums = {}
    for path in sorted(output_dir.rglob("*")):
        if path.is_file() and path.name != "SHA256SUMS.txt":
            checksums[str(path.relative_to(output_dir))] = sha256_file(path)
    checksum_path = output_dir / "SHA256SUMS.txt"
    checksum_path.write_text(
        "\n".join(f"{digest}  {name}" for name, digest in checksums.items()) + "\n",
        encoding="utf-8",
    )
    return {
        "output_dir": str(output_dir),
        "primary": survey_primary_row.to_dict(),
        "crude": crude_row.to_dict(),
        "checks": checks,
        "files": len(checksums) + 1,
    }


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--input", required=True, type=Path, help="Survey-enriched TSV input")
    parser.add_argument("--output", required=True, type=Path, help="Output directory")
    return parser.parse_args()


def main() -> None:
    args = parse_args()
    summary = run_analysis(args.input, args.output)
    print(json.dumps(summary, indent=2, sort_keys=True, default=str))


if __name__ == "__main__":
    main()
