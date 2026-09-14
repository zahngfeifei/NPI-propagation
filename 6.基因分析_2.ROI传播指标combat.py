# -*- coding: utf-8 -*-
"""
Gene analysis — Step 2
Apply ComBat to parcel-wise early propagation slopes.

Input
-----
roi_early_slope_1_10_raw_wide.csv

Expected structure:
    sub_id,
    ROI_000_early_slope_1_10,
    ...
    ROI_399_early_slope_1_10

Pipeline
--------
1. Load the raw ROI early-slope matrix generated in Step 1.
2. Match the same covariates and, when available, the exact same subject set
   used by the original system-level ComBat analysis.
3. Run neuroCombat with:
       batch_col       = Site
       categorical     = Group, Sex
       continuous      = Age, FIQ (FIQ only when available)
4. Save raw-scale ComBat-adjusted ROI slopes.
5. Multiply the adjusted values by 1000 only after ComBat, matching the
   existing system-level pipeline.
6. Aggregate ROI values within H1--H4 for cross-scale QC against the existing
   raw and system-level ComBat metrics.

This script does not modify or rerun the existing behavior analysis.
"""

from __future__ import annotations

import pickle
import re
from datetime import datetime
from pathlib import Path
from typing import Dict, List, Optional, Sequence, Tuple

import numpy as np
import pandas as pd

try:
    from neuroCombat import neuroCombat
except ImportError as exc:
    raise ImportError(
        "neuroCombat is not installed. Install it first with:\n"
        "    pip install neuroCombat"
    ) from exc


PROJECT_ROOT = Path(__file__).resolve().parent


# ============================================================
# 1. PATH SETTINGS — ALL ABSOLUTE PATHS
# ============================================================

# Step 1 输出：未经缩放的 400 个 ROI 早期传播斜率。
# 必须使用 raw-wide 文件，不使用 scaled-wide 文件。
ROI_RAW_CSV = Path(
    PROJECT_ROOT
    / "ABIDE2_主流程结果1_ROI传播指标"
    / "roi-early-slope-1-10-raw-wide.csv"
)

# ComBat 协变量文件。
# 预期列：
# sub_id, Group, Age, Sex, Site, FIQ, MeanFD
DEMO_CSV = Path(
    PROJECT_ROOT / "subject_info_for_stats.csv"
)

# 不使用另外的参考被试文件。
# 当前脚本直接使用 DEMO_CSV，并与 ROI_RAW_CSV 按 sub_id 取交集。
REFERENCE_SUBJECTS_CSV: Optional[Path] = None

# ROI 与 H1--H4 层级对应表。
# 预期列：
# ROI_index_0based, Hierarchy
ROI_DEFINITIONS_CSV = Path(
    PROJECT_ROOT / "ABIDE2_主流程必要输入" / "roi-definitions.csv"
)

# 上游未经 ComBat 的系统级传播指标，仅用于原始尺度重建 QC。
RAW_SYSTEM_METRICS_CSV: Optional[Path] = Path(
    PROJECT_ROOT / "ABIDE2_新结果1" / "EC_SEC_metrics_all_subjects.csv"
)

# 如已有“系统级指标直接运行 ComBat”的结果，可在此填写绝对路径。
# 当前未提供，因此跳过与系统级 ComBat 结果的数值比较。
SYSTEM_COMBAT_METRICS_CSV: Optional[Path] = None

# Step 2 输出目录。
OUT_DIR = Path(
    PROJECT_ROOT / "ABIDE2_主流程结果2_ROI传播指标ComBat"
)
OUT_DIR.mkdir(
    parents=True,
    exist_ok=True,
)


# ============================================================
# 2. COMBAT SETTINGS
# ============================================================

BATCH_COL = "Site"
CATEGORICAL_COLS = ["Group", "Sex"]
CONTINUOUS_CANDIDATE_COLS = ["Age", "FIQ"]

EPS_VARIANCE = 1e-12
OUTCOME_SCALE = 1000.0
EXPECTED_N_ROI: Optional[int] = 400

# Exact matching to the original system-level ComBat sample is recommended.
REQUIRE_EXACT_REFERENCE_SUBJECT_SET = True

# The existing system-level script copied invalid/constant features unchanged.
# This script follows the same convention and records them in the feature report.
COPY_INVALID_FEATURES_UNCHANGED = True

# Numerical tolerance for the pre-ComBat reconstruction check.
RAW_RECONSTRUCTION_TOLERANCE = 1e-8

SYSTEMS = {
    "H1_sensory": "H1_sensory.csv",
    "H2_attention": "H2_attention.csv",
    "H3_control": "H3_control.csv",
    "H4_DMN": "H4_DMN.csv",
}

SYSTEM_METRIC_SUFFIX = "early_slope_1_10"

ROI_FEATURE_PATTERN = re.compile(
    r"^ROI_(\d+)_early_slope_1_10$",
    flags=re.IGNORECASE,
)


# ============================================================
# 3. OUTPUT FILES
# ============================================================

OUT_RAW_MATCHED = OUT_DIR / "roi-early-slope-raw-matched-for-combat.csv"
OUT_COMBAT_RAW = OUT_DIR / "roi-early-slope-combat-raw-wide.csv"
OUT_COMBAT_SCALED = OUT_DIR / "roi-early-slope-combat-scaled-wide.csv"
OUT_COMBAT_LONG = OUT_DIR / "roi-early-slope-combat-long.csv"

OUT_RAW_NPY = OUT_DIR / "roi-early-slope-raw-matched.npy"
OUT_COMBAT_RAW_NPY = OUT_DIR / "roi-early-slope-combat-raw.npy"
OUT_COMBAT_SCALED_NPY = OUT_DIR / "roi-early-slope-combat-scaled.npy"
OUT_SUBJECT_ORDER = OUT_DIR / "roi-combat-subject-order.csv"

OUT_FEATURE_REPORT = OUT_DIR / "roi-combat-feature-report.csv"
OUT_VALID_MASK = OUT_DIR / "roi-combat-valid-feature-mask.npy"
OUT_COMBAT_MODEL = OUT_DIR / "roi-combat-model-estimates.pkl"

OUT_MATCH_AUDIT = OUT_DIR / "roi-combat-subject-match-audit.csv"
OUT_SITE_COUNTS = OUT_DIR / "roi-combat-site-counts.csv"
OUT_SITE_GROUP = OUT_DIR / "roi-combat-site-by-group.csv"

OUT_SYSTEM_QC_DETAIL = OUT_DIR / "roi-combat-system-aggregation-qc-detail.csv"
OUT_SYSTEM_QC_SUMMARY = OUT_DIR / "roi-combat-system-aggregation-qc-summary.csv"
OUT_ROI_SET_AUDIT = OUT_DIR / "roi-combat-roi-set-audit.csv"

OUT_REPORT = OUT_DIR / "roi-early-slope-combat-report.txt"


# ============================================================
# 4. GENERAL UTILITIES
# ============================================================

def normalize_sub_id(value) -> Optional[str]:
    """
    Normalize IDs to:
        sub-Sub00000
    """
    if pd.isna(value):
        return None

    text = str(value).strip()

    if re.fullmatch(r"sub-Sub\d{5}", text):
        return text

    match = re.search(
        r"Sub0*(\d+)",
        text,
        flags=re.IGNORECASE,
    )
    if match:
        return f"sub-Sub{int(match.group(1)):05d}"

    if re.fullmatch(r"\d+(\.0)?", text):
        return f"sub-Sub{int(float(text)):05d}"

    return None


def clean_columns(table: pd.DataFrame) -> pd.DataFrame:
    table = table.copy()
    table.columns = [str(column).strip() for column in table.columns]
    return table


def write_text(path: Path, text: str) -> None:
    with open(path, "w", encoding="utf-8") as file:
        file.write(text)


def first_existing_id_series(table: pd.DataFrame) -> pd.Series:
    for column in ["sub_id", "SUB_ID", "FILE_ID", "SUB_ID_norm"]:
        if column in table.columns:
            return table[column]

    raise ValueError(
        "No subject-ID column found. Expected one of: "
        "sub_id, SUB_ID, FILE_ID, SUB_ID_norm."
    )


def safe_pearson(x: Sequence[float], y: Sequence[float]) -> float:
    x_array = np.asarray(x, dtype=float)
    y_array = np.asarray(y, dtype=float)

    valid = np.isfinite(x_array) & np.isfinite(y_array)
    x_array = x_array[valid]
    y_array = y_array[valid]

    if len(x_array) < 3:
        return np.nan

    if np.std(x_array) == 0 or np.std(y_array) == 0:
        return np.nan

    return float(np.corrcoef(x_array, y_array)[0, 1])


# ============================================================
# 5. LOAD ROI EARLY-SLOPE TABLE
# ============================================================

def discover_roi_feature_columns(
    table: pd.DataFrame,
) -> Tuple[List[str], List[int]]:
    """
    Find and sort ROI columns by their 0-based ROI index.
    """
    parsed = []

    for column in table.columns:
        match = ROI_FEATURE_PATTERN.fullmatch(str(column))
        if match:
            parsed.append((int(match.group(1)), str(column)))

    if len(parsed) == 0:
        raise ValueError(
            "No ROI early-slope columns were found. Expected names such as:\n"
            "    ROI_000_early_slope_1_10"
        )

    parsed = sorted(parsed, key=lambda item: item[0])

    roi_indices = [item[0] for item in parsed]
    feature_columns = [item[1] for item in parsed]

    if len(set(roi_indices)) != len(roi_indices):
        raise ValueError("Duplicate ROI indices were found in the ROI table.")

    if EXPECTED_N_ROI is not None and len(feature_columns) != EXPECTED_N_ROI:
        raise ValueError(
            f"Expected {EXPECTED_N_ROI} ROI columns, "
            f"but found {len(feature_columns)}."
        )

    if roi_indices != list(range(len(roi_indices))):
        raise ValueError(
            "ROI columns are not a complete consecutive 0-based sequence. "
            f"First indices: {roi_indices[:10]}; "
            f"last indices: {roi_indices[-10:]}"
        )

    return feature_columns, roi_indices


def load_roi_table(
    path: Path,
) -> Tuple[pd.DataFrame, List[str], List[int]]:
    if not path.exists():
        raise FileNotFoundError(f"ROI raw CSV not found: {path}")

    table = pd.read_csv(path, low_memory=False, encoding="utf-8-sig")
    table = clean_columns(table)

    if "sub_id" not in table.columns:
        raise ValueError("ROI raw CSV must contain a sub_id column.")

    table["sub_id"] = table["sub_id"].apply(normalize_sub_id)
    table = table.dropna(subset=["sub_id"]).copy()

    duplicated = table[table.duplicated(subset=["sub_id"], keep=False)]
    if not duplicated.empty:
        examples = duplicated["sub_id"].drop_duplicates().head(10).tolist()
        raise ValueError(
            "Duplicate subjects were found in the ROI raw CSV. "
            f"Examples: {examples}"
        )

    feature_columns, roi_indices = discover_roi_feature_columns(table)

    for column in feature_columns:
        table[column] = pd.to_numeric(table[column], errors="coerce")

    if not np.isfinite(
        table[feature_columns].to_numpy(dtype=float)
    ).all():
        bad_count = int(
            np.sum(
                ~np.isfinite(
                    table[feature_columns].to_numpy(dtype=float)
                )
            )
        )
        raise ValueError(
            f"ROI feature matrix contains {bad_count} NaN/Inf values."
        )

    return table[["sub_id"] + feature_columns].copy(), feature_columns, roi_indices


# ============================================================
# 6. LOAD COVARIATES AND ALIGN SUBJECTS
# ============================================================

def load_covariate_source() -> Tuple[pd.DataFrame, str, bool]:
    """
    读取 subject_info_for_stats.csv，并准备 ComBat 协变量。

    必需列
    ------
    sub_id, Group, Age, Sex, Site

    可选列
    ------
    FIQ:
        存在且完整时作为连续协变量进入 ComBat。

    MeanFD:
        保留在输出中作为被试信息，但不进入 ComBat，
        以匹配上游分析设定。
    """
    if DEMO_CSV is None or not DEMO_CSV.exists():
        raise FileNotFoundError(
            f"Covariate CSV not found: {DEMO_CSV}"
        )

    source_path = DEMO_CSV
    source_name = "subject_info_for_stats_csv"
    using_reference = False

    table = pd.read_csv(
        source_path,
        low_memory=False,
        encoding="utf-8-sig",
    )
    table = clean_columns(table)

    id_series = first_existing_id_series(table)

    table = table.copy()
    table["sub_id"] = id_series.apply(
        normalize_sub_id
    )
    table = table.dropna(
        subset=["sub_id"]
    ).copy()

    required_columns = [
        "Group",
        "Age",
        "Sex",
        "Site",
    ]

    missing = [
        column
        for column in required_columns
        if column not in table.columns
    ]

    if missing:
        raise ValueError(
            "Covariate source is missing required columns: "
            f"{missing}. "
            f"Available columns: {table.columns.tolist()}"
        )

    keep_columns = [
        "sub_id",
        "Group",
        "Age",
        "Sex",
        "Site",
    ]

    if "FIQ" in table.columns:
        keep_columns.append("FIQ")

    if "MeanFD" in table.columns:
        keep_columns.append("MeanFD")

    table = table[keep_columns].copy()

    table["Group"] = (
        table["Group"]
        .astype(str)
        .str.strip()
    )
    table["Sex"] = (
        table["Sex"]
        .astype(str)
        .str.strip()
    )
    table["Site"] = (
        table["Site"]
        .astype(str)
        .str.strip()
    )

    table["Age"] = pd.to_numeric(
        table["Age"],
        errors="coerce",
    )

    if "FIQ" in table.columns:
        table["FIQ"] = pd.to_numeric(
            table["FIQ"],
            errors="coerce",
        )

    if "MeanFD" in table.columns:
        table["MeanFD"] = pd.to_numeric(
            table["MeanFD"],
            errors="coerce",
        )

    # Group / Age / Sex / Site 必须完整。
    required_complete = [
        "Group",
        "Age",
        "Sex",
        "Site",
    ]

    # 若 FIQ 列存在，则为了保持上游 ComBat 设定，
    # 仅保留 FIQ 完整的被试。
    if "FIQ" in table.columns:
        required_complete.append("FIQ")

    table = table.dropna(
        subset=required_complete
    ).copy()

    duplicated = table[
        table.duplicated(
            subset=["sub_id"],
            keep=False,
        )
    ]

    if not duplicated.empty:
        examples = (
            duplicated["sub_id"]
            .drop_duplicates()
            .head(10)
            .tolist()
        )
        raise ValueError(
            "Duplicate subjects were found in "
            "subject_info_for_stats.csv. "
            f"Examples: {examples}"
        )

    table = table.reset_index(drop=True)
    table["_reference_order"] = np.arange(
        len(table),
        dtype=int,
    )

    return (
        table,
        source_name,
        using_reference,
    )


def align_roi_and_covariates(
    roi_table: pd.DataFrame,
    covariates: pd.DataFrame,
    using_reference: bool,
) -> Tuple[pd.DataFrame, pd.DataFrame]:
    roi_subjects = set(roi_table["sub_id"])
    cov_subjects = set(covariates["sub_id"])

    audit_rows = []

    for subject in sorted(roi_subjects | cov_subjects):
        audit_rows.append(
            {
                "sub_id": subject,
                "in_roi_table": subject in roi_subjects,
                "in_covariate_source": subject in cov_subjects,
            }
        )

    audit = pd.DataFrame(audit_rows)
    audit["in_both"] = (
        audit["in_roi_table"]
        & audit["in_covariate_source"]
    )
    audit.to_csv(
        OUT_MATCH_AUDIT,
        index=False,
        encoding="utf-8-sig",
    )

    missing_roi = sorted(cov_subjects - roi_subjects)
    extra_roi = sorted(roi_subjects - cov_subjects)

    if (
        using_reference
        and REQUIRE_EXACT_REFERENCE_SUBJECT_SET
        and (len(missing_roi) > 0 or len(extra_roi) > 0)
    ):
        raise ValueError(
            "ROI subjects do not exactly match the original system-level "
            "ComBat subject set.\n"
            f"Reference subjects missing ROI data: {len(missing_roi)}\n"
            f"ROI subjects absent from reference set: {len(extra_roi)}\n"
            f"See audit: {OUT_MATCH_AUDIT}"
        )

    merged = covariates.merge(
        roi_table,
        on="sub_id",
        how="inner",
        validate="one_to_one",
    )

    merged = merged.sort_values("_reference_order").reset_index(drop=True)
    merged = merged.drop(columns=["_reference_order"])

    if merged.empty:
        raise RuntimeError("No subjects remained after ROI/covariate matching.")

    return merged, audit


# ============================================================
# 7. BUILD COMBAT COVARIATE TABLE
# ============================================================

def build_combat_covariates(
    matched: pd.DataFrame,
) -> Tuple[pd.DataFrame, List[str]]:
    continuous_columns = [
        column
        for column in CONTINUOUS_CANDIDATE_COLS
        if column in matched.columns
    ]

    covars = pd.DataFrame(
        {
            "Site": matched["Site"].astype(str).values,
            "Group": matched["Group"].astype(str).values,
            "Age": pd.to_numeric(
                matched["Age"],
                errors="raise",
            ).values,
            "Sex": matched["Sex"].astype(str).values,
        }
    )

    if "FIQ" in continuous_columns:
        covars["FIQ"] = pd.to_numeric(
            matched["FIQ"],
            errors="raise",
        ).values

    return covars, continuous_columns


def save_batch_audits(
    covars: pd.DataFrame,
) -> Tuple[pd.Series, pd.DataFrame]:
    site_counts = covars["Site"].value_counts().sort_index()
    site_counts.rename("N").reset_index().rename(
        columns={"index": "Site"}
    ).to_csv(
        OUT_SITE_COUNTS,
        index=False,
        encoding="utf-8-sig",
    )

    site_group = pd.crosstab(
        covars["Site"],
        covars["Group"],
    )
    site_group.to_csv(
        OUT_SITE_GROUP,
        encoding="utf-8-sig",
    )

    if covars["Site"].nunique() < 2:
        raise RuntimeError(
            "ComBat requires at least two Site/batch levels."
        )

    singleton_sites = site_counts[site_counts < 2]
    if not singleton_sites.empty:
        print("Warning: singleton sites were found:")
        print(singleton_sites)

    return site_counts, site_group


# ============================================================
# 8. RUN ROI-LEVEL COMBAT
# ============================================================

def run_roi_combat(
    matched: pd.DataFrame,
    feature_columns: List[str],
    covars: pd.DataFrame,
    continuous_columns: List[str],
) -> Tuple[
    np.ndarray,
    np.ndarray,
    np.ndarray,
    pd.DataFrame,
    dict,
]:
    """
    Returns
    -------
    raw_feature_by_subject:
        features × subjects

    combat_feature_by_subject:
        features × subjects

    valid_feature_mask:
        Boolean vector over features

    feature_report:
        Per-feature QC table

    combat_result:
        Full neuroCombat return object for valid features
    """
    raw_subject_by_feature = matched[
        feature_columns
    ].to_numpy(dtype=np.float64)

    raw_feature_by_subject = raw_subject_by_feature.T

    finite_mask = np.isfinite(
        raw_feature_by_subject
    ).all(axis=1)

    variance = np.nanvar(
        raw_feature_by_subject,
        axis=1,
    )

    valid_feature_mask = (
        finite_mask
        & (variance > EPS_VARIANCE)
    )

    n_valid = int(np.sum(valid_feature_mask))
    n_total = int(len(feature_columns))

    if n_valid == 0:
        raise RuntimeError("No valid ROI features are available for ComBat.")

    if not COPY_INVALID_FEATURES_UNCHANGED and n_valid != n_total:
        raise RuntimeError(
            f"{n_total - n_valid} invalid ROI features were found."
        )

    print("=" * 80)
    print("Running ROI-level ComBat")
    print("=" * 80)
    print(f"Subjects: {raw_feature_by_subject.shape[1]}")
    print(f"ROI features: {n_total}")
    print(f"Valid features used by ComBat: {n_valid}")
    print(f"Features copied unchanged: {n_total - n_valid}")

    combat_feature_by_subject = raw_feature_by_subject.copy()

    combat_result = neuroCombat(
        dat=raw_feature_by_subject[
            valid_feature_mask,
            :
        ],
        covars=covars,
        batch_col=BATCH_COL,
        categorical_cols=CATEGORICAL_COLS,
        continuous_cols=continuous_columns,
    )

    combat_data = np.asarray(
        combat_result["data"],
        dtype=np.float64,
    )

    expected_shape = (
        n_valid,
        raw_feature_by_subject.shape[1],
    )
    if combat_data.shape != expected_shape:
        raise RuntimeError(
            "Unexpected neuroCombat output shape. "
            f"Expected {expected_shape}, got {combat_data.shape}."
        )

    combat_feature_by_subject[
        valid_feature_mask,
        :
    ] = combat_data

    report_rows = []

    for feature_index, feature_name in enumerate(feature_columns):
        raw_values = raw_feature_by_subject[feature_index, :]
        combat_values = combat_feature_by_subject[feature_index, :]

        report_rows.append(
            {
                "roi_index_0based": feature_index,
                "feature": feature_name,
                "used_for_combat": bool(
                    valid_feature_mask[feature_index]
                ),
                "raw_mean": float(np.mean(raw_values)),
                "raw_sd": float(np.std(raw_values)),
                "raw_min": float(np.min(raw_values)),
                "raw_max": float(np.max(raw_values)),
                "combat_mean": float(np.mean(combat_values)),
                "combat_sd": float(np.std(combat_values)),
                "combat_min": float(np.min(combat_values)),
                "combat_max": float(np.max(combat_values)),
                "mean_change": float(
                    np.mean(combat_values - raw_values)
                ),
                "raw_vs_combat_r": safe_pearson(
                    raw_values,
                    combat_values,
                ),
            }
        )

    feature_report = pd.DataFrame(report_rows)

    return (
        raw_feature_by_subject,
        combat_feature_by_subject,
        valid_feature_mask,
        feature_report,
        combat_result,
    )


# ============================================================
# 9. ROI-SET AND SYSTEM-AGGREGATION QC
# ============================================================

def read_roi_index_0based(
    path: Path,
) -> np.ndarray:
    if not path.exists():
        raise FileNotFoundError(f"ROI definition file not found: {path}")

    table = pd.read_csv(path)
    table = clean_columns(table)

    if table.empty:
        return np.array([], dtype=int)

    column = (
        "ROI_index_0based"
        if "ROI_index_0based" in table.columns
        else table.columns[0]
    )

    indices = (
        pd.to_numeric(
            table[column],
            errors="coerce",
        )
        .dropna()
        .astype(int)
        .to_numpy()
    )

    return np.sort(np.unique(indices))


def load_system_rois(
    n_roi: int,
) -> Dict[str, np.ndarray]:
    """
    从 roi_to_hierarchy_mapping.csv 读取 H1--H4 ROI 集合。

    必需列
    ------
    ROI_index_0based:
        ROI 的 0-based 编号，预期范围为 0..399。

    Hierarchy:
        ROI 所属层级：
        H1_sensory
        H2_attention
        H3_control
        H4_DMN
    """
    if not ROI_DEFINITIONS_CSV.exists():
        raise FileNotFoundError(
            f"ROI definition file not found: "
            f"{ROI_DEFINITIONS_CSV}"
        )

    roi_definition_table = clean_columns(
        pd.read_csv(
            ROI_DEFINITIONS_CSV,
            low_memory=False,
            encoding="utf-8-sig",
        )
    )
    roi_definition_table = roi_definition_table.rename(
        columns={"roi_index_0based": "ROI_index_0based", "system": "Hierarchy"}
    )

    required_columns = {
        "ROI_index_0based",
        "Hierarchy",
    }

    missing_columns = required_columns.difference(
        roi_definition_table.columns
    )

    if missing_columns:
        raise ValueError(
            "ROI definition file is missing required columns: "
            f"{sorted(missing_columns)}. "
            f"Available columns: "
            f"{roi_definition_table.columns.tolist()}"
        )

    roi_definition_table = (
        roi_definition_table.copy()
    )

    roi_definition_table[
        "Hierarchy"
    ] = (
        roi_definition_table[
            "Hierarchy"
        ]
        .astype(str)
        .str.strip()
    )

    system_rois: Dict[str, np.ndarray] = {}

    membership_count = np.zeros(
        n_roi,
        dtype=int,
    )

    audit_rows = []

    for system_name in SYSTEMS:

        indices = (
            pd.to_numeric(
                roi_definition_table.loc[
                    roi_definition_table[
                        "Hierarchy"
                    ].eq(system_name),
                    "ROI_index_0based",
                ],
                errors="coerce",
            )
            .dropna()
            .astype(int)
            .to_numpy()
        )

        indices = np.sort(
            np.unique(indices)
        )

        if len(indices) == 0:
            raise ValueError(
                f"No ROI was found for hierarchy: "
                f"{system_name}"
            )

        bad = indices[
            (indices < 0)
            | (indices >= n_roi)
        ]

        if len(bad) > 0:
            raise ValueError(
                f"{system_name} contains invalid ROI indices. "
                f"Expected 0..{n_roi - 1}; "
                f"examples={bad[:10].tolist()}"
            )

        system_rois[system_name] = indices
        membership_count[indices] += 1

        audit_rows.append(
            {
                "system": system_name,
                "n_roi": int(len(indices)),
                "min_roi": int(indices.min()),
                "max_roi": int(indices.max()),
            }
        )

    audit_rows.extend(
        [
            {
                "system": "UNION",
                "n_roi": int(
                    np.sum(
                        membership_count > 0
                    )
                ),
                "min_roi": np.nan,
                "max_roi": np.nan,
            },
            {
                "system": "UNCOVERED",
                "n_roi": int(
                    np.sum(
                        membership_count == 0
                    )
                ),
                "min_roi": np.nan,
                "max_roi": np.nan,
            },
            {
                "system": "OVERLAPPING",
                "n_roi": int(
                    np.sum(
                        membership_count > 1
                    )
                ),
                "min_roi": np.nan,
                "max_roi": np.nan,
            },
        ]
    )

    pd.DataFrame(
        audit_rows
    ).to_csv(
        OUT_ROI_SET_AUDIT,
        index=False,
        encoding="utf-8-sig",
    )

    return system_rois


def aggregate_roi_matrix_to_systems(
    subject_by_roi: np.ndarray,
    system_rois: Dict[str, np.ndarray],
) -> pd.DataFrame:
    output = {}

    for system_name, indices in system_rois.items():
        if len(indices) == 0:
            output[system_name] = np.full(
                subject_by_roi.shape[0],
                np.nan,
            )
        else:
            output[system_name] = np.mean(
                subject_by_roi[:, indices],
                axis=1,
            )

    return pd.DataFrame(output)


def load_optional_system_metrics(
    path: Optional[Path],
    value_prefix: str,
) -> Optional[pd.DataFrame]:
    if path is None or not path.exists():
        return None

    table = pd.read_csv(
        path,
        low_memory=False,
        encoding="utf-8-sig",
    )
    table = clean_columns(table)

    id_series = first_existing_id_series(table)
    table = table.copy()
    table["sub_id"] = id_series.apply(normalize_sub_id)
    table = table.dropna(subset=["sub_id"])
    table = table.drop_duplicates(subset=["sub_id"], keep="first")

    keep = ["sub_id"]
    rename = {}

    for system_name in SYSTEMS:
        original_column = (
            f"{system_name}_{SYSTEM_METRIC_SUFFIX}"
        )
        if original_column not in table.columns:
            raise ValueError(
                f"Missing system metric column in {path}: "
                f"{original_column}"
            )

        new_column = f"{value_prefix}_{system_name}"
        keep.append(original_column)
        rename[original_column] = new_column

    return table[keep].rename(columns=rename)


def build_system_aggregation_qc(
    matched: pd.DataFrame,
    raw_subject_by_roi: np.ndarray,
    combat_subject_by_roi: np.ndarray,
) -> Tuple[pd.DataFrame, pd.DataFrame]:
    n_roi = raw_subject_by_roi.shape[1]
    system_rois = load_system_rois(n_roi)

    raw_aggregated = aggregate_roi_matrix_to_systems(
        raw_subject_by_roi,
        system_rois,
    )
    combat_aggregated = aggregate_roi_matrix_to_systems(
        combat_subject_by_roi,
        system_rois,
    )

    base = pd.DataFrame(
        {
            "sub_id": matched["sub_id"].values,
        }
    )

    for system_name in SYSTEMS:
        base[f"roi_raw_mean_{system_name}"] = (
            raw_aggregated[system_name].values
        )
        base[f"roi_combat_mean_{system_name}"] = (
            combat_aggregated[system_name].values
        )

    raw_system = load_optional_system_metrics(
        RAW_SYSTEM_METRICS_CSV,
        value_prefix="saved_raw",
    )
    combat_system = load_optional_system_metrics(
        SYSTEM_COMBAT_METRICS_CSV,
        value_prefix="saved_combat",
    )

    if raw_system is not None:
        base = base.merge(
            raw_system,
            on="sub_id",
            how="left",
            validate="one_to_one",
        )

    if combat_system is not None:
        base = base.merge(
            combat_system,
            on="sub_id",
            how="left",
            validate="one_to_one",
        )

    detail_rows = []

    for _, row in base.iterrows():
        subject = row["sub_id"]

        for system_name in SYSTEMS:
            roi_raw = float(
                row[f"roi_raw_mean_{system_name}"]
            )
            roi_combat = float(
                row[f"roi_combat_mean_{system_name}"]
            )

            saved_raw_column = f"saved_raw_{system_name}"
            saved_combat_column = f"saved_combat_{system_name}"

            saved_raw = (
                float(row[saved_raw_column])
                if (
                    saved_raw_column in base.columns
                    and pd.notna(row[saved_raw_column])
                )
                else np.nan
            )

            saved_combat = (
                float(row[saved_combat_column])
                if (
                    saved_combat_column in base.columns
                    and pd.notna(row[saved_combat_column])
                )
                else np.nan
            )

            detail_rows.append(
                {
                    "sub_id": subject,
                    "system": system_name,
                    "n_roi_in_system": int(
                        len(system_rois[system_name])
                    ),
                    "roi_raw_mean": roi_raw,
                    "saved_raw_system_slope": saved_raw,
                    "abs_diff_roi_raw_vs_saved_raw": (
                        abs(roi_raw - saved_raw)
                        if np.isfinite(saved_raw)
                        else np.nan
                    ),
                    "roi_combat_mean": roi_combat,
                    "saved_system_combat_slope": saved_combat,
                    "difference_roi_combat_vs_system_combat": (
                        roi_combat - saved_combat
                        if np.isfinite(saved_combat)
                        else np.nan
                    ),
                    "abs_diff_roi_combat_vs_system_combat": (
                        abs(roi_combat - saved_combat)
                        if np.isfinite(saved_combat)
                        else np.nan
                    ),
                }
            )

    detail = pd.DataFrame(detail_rows)

    summary_rows = []

    for system_name, system_table in detail.groupby(
        "system",
        sort=False,
    ):
        raw_diff = pd.to_numeric(
            system_table["abs_diff_roi_raw_vs_saved_raw"],
            errors="coerce",
        ).dropna()

        combat_diff = pd.to_numeric(
            system_table[
                "abs_diff_roi_combat_vs_system_combat"
            ],
            errors="coerce",
        ).dropna()

        summary_rows.append(
            {
                "system": system_name,
                "comparison": "roi_raw_mean_vs_saved_raw_system",
                "N": int(len(raw_diff)),
                "mean_abs_diff": (
                    float(raw_diff.mean())
                    if len(raw_diff) > 0
                    else np.nan
                ),
                "median_abs_diff": (
                    float(raw_diff.median())
                    if len(raw_diff) > 0
                    else np.nan
                ),
                "max_abs_diff": (
                    float(raw_diff.max())
                    if len(raw_diff) > 0
                    else np.nan
                ),
                "pearson_r": safe_pearson(
                    system_table["roi_raw_mean"],
                    system_table["saved_raw_system_slope"],
                ),
                "n_above_raw_tolerance": (
                    int(
                        np.sum(
                            raw_diff
                            > RAW_RECONSTRUCTION_TOLERANCE
                        )
                    )
                    if len(raw_diff) > 0
                    else 0
                ),
            }
        )

        summary_rows.append(
            {
                "system": system_name,
                "comparison": (
                    "roi_combat_mean_vs_saved_system_level_combat"
                ),
                "N": int(len(combat_diff)),
                "mean_abs_diff": (
                    float(combat_diff.mean())
                    if len(combat_diff) > 0
                    else np.nan
                ),
                "median_abs_diff": (
                    float(combat_diff.median())
                    if len(combat_diff) > 0
                    else np.nan
                ),
                "max_abs_diff": (
                    float(combat_diff.max())
                    if len(combat_diff) > 0
                    else np.nan
                ),
                "pearson_r": safe_pearson(
                    system_table["roi_combat_mean"],
                    system_table["saved_system_combat_slope"],
                ),
                "n_above_raw_tolerance": np.nan,
            }
        )

    summary = pd.DataFrame(summary_rows)

    detail.to_csv(
        OUT_SYSTEM_QC_DETAIL,
        index=False,
        encoding="utf-8-sig",
    )
    summary.to_csv(
        OUT_SYSTEM_QC_SUMMARY,
        index=False,
        encoding="utf-8-sig",
    )

    return detail, summary


# ============================================================
# 10. SAVE MAIN OUTPUTS
# ============================================================

def save_main_outputs(
    matched: pd.DataFrame,
    feature_columns: List[str],
    raw_feature_by_subject: np.ndarray,
    combat_feature_by_subject: np.ndarray,
    valid_feature_mask: np.ndarray,
    feature_report: pd.DataFrame,
    combat_result: dict,
    continuous_columns: List[str],
) -> Tuple[pd.DataFrame, pd.DataFrame]:
    raw_subject_by_feature = raw_feature_by_subject.T
    combat_subject_by_feature = combat_feature_by_subject.T
    combat_scaled_subject_by_feature = (
        combat_subject_by_feature * OUTCOME_SCALE
    )

    subject_columns = [
        column
        for column in [
            "sub_id",
            "Group",
            "Age",
            "Sex",
            "Site",
            "FIQ",
            "MeanFD",
        ]
        if column in matched.columns
    ]

    subject_info = matched[subject_columns].copy()

    raw_output = pd.concat(
        [
            subject_info.reset_index(drop=True),
            pd.DataFrame(
                raw_subject_by_feature,
                columns=feature_columns,
            ),
        ],
        axis=1,
    )

    combat_output = pd.concat(
        [
            subject_info.reset_index(drop=True),
            pd.DataFrame(
                combat_subject_by_feature,
                columns=feature_columns,
            ),
        ],
        axis=1,
    )

    scaled_feature_columns = [
        f"{column}_scaled"
        for column in feature_columns
    ]

    combat_scaled_output = pd.concat(
        [
            subject_info.reset_index(drop=True),
            pd.DataFrame(
                combat_scaled_subject_by_feature,
                columns=scaled_feature_columns,
            ),
        ],
        axis=1,
    )

    raw_output.to_csv(
        OUT_RAW_MATCHED,
        index=False,
        encoding="utf-8-sig",
    )

    combat_output.to_csv(
        OUT_COMBAT_RAW,
        index=False,
        encoding="utf-8-sig",
    )

    combat_scaled_output.to_csv(
        OUT_COMBAT_SCALED,
        index=False,
        encoding="utf-8-sig",
    )

    np.save(
        OUT_RAW_NPY,
        raw_subject_by_feature,
    )
    np.save(
        OUT_COMBAT_RAW_NPY,
        combat_subject_by_feature,
    )
    np.save(
        OUT_COMBAT_SCALED_NPY,
        combat_scaled_subject_by_feature,
    )
    np.save(
        OUT_VALID_MASK,
        valid_feature_mask,
    )

    subject_info.to_csv(
        OUT_SUBJECT_ORDER,
        index=False,
        encoding="utf-8-sig",
    )

    feature_report.to_csv(
        OUT_FEATURE_REPORT,
        index=False,
        encoding="utf-8-sig",
    )

    long_rows = []
    for subject_index, subject in enumerate(
        matched["sub_id"].tolist()
    ):
        for roi_index, feature_name in enumerate(feature_columns):
            long_rows.append(
                {
                    "sub_id": subject,
                    "roi_index_0based": roi_index,
                    "feature": feature_name,
                    "early_slope_raw": float(
                        raw_subject_by_feature[
                            subject_index,
                            roi_index,
                        ]
                    ),
                    "early_slope_combat_raw": float(
                        combat_subject_by_feature[
                            subject_index,
                            roi_index,
                        ]
                    ),
                    "early_slope_combat_scaled": float(
                        combat_scaled_subject_by_feature[
                            subject_index,
                            roi_index,
                        ]
                    ),
                }
            )

    pd.DataFrame(long_rows).to_csv(
        OUT_COMBAT_LONG,
        index=False,
        encoding="utf-8-sig",
    )

    model_payload = {
        "estimates": combat_result.get("estimates"),
        "info": combat_result.get("info"),
        "feature_columns": feature_columns,
        "valid_feature_mask": valid_feature_mask,
        "batch_col": BATCH_COL,
        "categorical_cols": CATEGORICAL_COLS,
        "continuous_cols": continuous_columns,
        "subject_order": matched["sub_id"].tolist(),
    }

    with open(OUT_COMBAT_MODEL, "wb") as file:
        pickle.dump(
            model_payload,
            file,
            protocol=pickle.HIGHEST_PROTOCOL,
        )

    return combat_output, combat_scaled_output


# ============================================================
# 11. MAIN
# ============================================================

def main() -> None:
    print("=" * 80)
    print("ROI-level early-slope ComBat")
    print("=" * 80)

    required_input_paths = {
        "ROI_RAW_CSV": ROI_RAW_CSV,
        "DEMO_CSV": DEMO_CSV,
        "ROI_DEFINITIONS_CSV": ROI_DEFINITIONS_CSV,
    }

    if RAW_SYSTEM_METRICS_CSV is not None:
        required_input_paths[
            "RAW_SYSTEM_METRICS_CSV"
        ] = RAW_SYSTEM_METRICS_CSV

    missing_paths = {
        name: path
        for name, path in required_input_paths.items()
        if path is None or not Path(path).exists()
    }

    if missing_paths:
        message_lines = [
            "The following required input paths do not exist:"
        ]
        for name, path in missing_paths.items():
            message_lines.append(
                f"  {name}: {path}"
            )
        raise FileNotFoundError(
            "\n".join(message_lines)
        )

    print(f"ROI raw input: {ROI_RAW_CSV}")
    print(f"Covariates: {DEMO_CSV}")
    print(f"ROI hierarchy mapping: {ROI_DEFINITIONS_CSV}")
    print(f"Raw system metrics: {RAW_SYSTEM_METRICS_CSV}")
    print(f"Output directory: {OUT_DIR}")

    roi_table, feature_columns, roi_indices = load_roi_table(
        ROI_RAW_CSV
    )

    covariates, covariate_source_name, using_reference = (
        load_covariate_source()
    )

    matched, match_audit = align_roi_and_covariates(
        roi_table=roi_table,
        covariates=covariates,
        using_reference=using_reference,
    )

    covars, continuous_columns = build_combat_covariates(
        matched
    )

    site_counts, site_group = save_batch_audits(covars)

    (
        raw_feature_by_subject,
        combat_feature_by_subject,
        valid_feature_mask,
        feature_report,
        combat_result,
    ) = run_roi_combat(
        matched=matched,
        feature_columns=feature_columns,
        covars=covars,
        continuous_columns=continuous_columns,
    )

    combat_output, combat_scaled_output = save_main_outputs(
        matched=matched,
        feature_columns=feature_columns,
        raw_feature_by_subject=raw_feature_by_subject,
        combat_feature_by_subject=combat_feature_by_subject,
        valid_feature_mask=valid_feature_mask,
        feature_report=feature_report,
        combat_result=combat_result,
        continuous_columns=continuous_columns,
    )

    raw_subject_by_roi = raw_feature_by_subject.T
    combat_subject_by_roi = combat_feature_by_subject.T

    system_qc_detail, system_qc_summary = build_system_aggregation_qc(
        matched=matched,
        raw_subject_by_roi=raw_subject_by_roi,
        combat_subject_by_roi=combat_subject_by_roi,
    )

    n_total_features = len(feature_columns)
    n_valid_features = int(np.sum(valid_feature_mask))
    n_invalid_features = n_total_features - n_valid_features

    reference_missing_roi = int(
        (
            match_audit["in_covariate_source"]
            & ~match_audit["in_roi_table"]
        ).sum()
    )
    roi_absent_reference = int(
        (
            match_audit["in_roi_table"]
            & ~match_audit["in_covariate_source"]
        ).sum()
    )

    raw_qc = system_qc_summary[
        system_qc_summary["comparison"]
        == "roi_raw_mean_vs_saved_raw_system"
    ]
    combat_qc = system_qc_summary[
        system_qc_summary["comparison"]
        == "roi_combat_mean_vs_saved_system_level_combat"
    ]

    max_raw_reconstruction_error = (
        pd.to_numeric(
            raw_qc["max_abs_diff"],
            errors="coerce",
        ).max()
        if not raw_qc.empty
        else np.nan
    )

    minimum_combat_system_r = (
        pd.to_numeric(
            combat_qc["pearson_r"],
            errors="coerce",
        ).min()
        if not combat_qc.empty
        else np.nan
    )

    report_lines = [
        "ROI-level early propagation slope ComBat report",
        f"Generated at: {datetime.now()}",
        "",
        "Purpose",
        "  Apply ComBat to 400 ROI early propagation slopes.",
        "  Preserve the original behavior analysis unchanged.",
        "",
        "Input",
        f"  ROI_RAW_CSV: {ROI_RAW_CSV}",
        f"  Covariate source: {covariate_source_name}",
        f"  REFERENCE_SUBJECTS_CSV: {REFERENCE_SUBJECTS_CSV}",
        f"  DEMO_CSV: {DEMO_CSV}",
        f"  RAW_SYSTEM_METRICS_CSV: {RAW_SYSTEM_METRICS_CSV}",
        f"  SYSTEM_COMBAT_METRICS_CSV: {SYSTEM_COMBAT_METRICS_CSV}",
        "",
        "ComBat specification",
        f"  batch_col: {BATCH_COL}",
        f"  categorical_cols: {', '.join(CATEGORICAL_COLS)}",
        f"  continuous_cols: {', '.join(continuous_columns)}",
        f"  outcome scaling after ComBat: x{OUTCOME_SCALE}",
        f"  EPS_VARIANCE: {EPS_VARIANCE}",
        "",
        "Sample",
        f"  ROI-table subjects: {roi_table.shape[0]}",
        f"  Covariate-source subjects: {covariates.shape[0]}",
        f"  Matched subjects used: {matched.shape[0]}",
        f"  Reference subjects missing ROI data: {reference_missing_roi}",
        f"  ROI subjects absent from reference: {roi_absent_reference}",
        "",
        "Features",
        f"  Total ROI features: {n_total_features}",
        f"  Features used for ComBat: {n_valid_features}",
        f"  Features copied unchanged: {n_invalid_features}",
        "",
        "Batch overview",
        f"  Number of sites: {covars['Site'].nunique()}",
        f"  Minimum site N: {int(site_counts.min())}",
        f"  Maximum site N: {int(site_counts.max())}",
        "",
        "Cross-scale QC",
        "  Pre-ComBat ROI means should reproduce the original system slopes.",
        f"  Max raw reconstruction error: {max_raw_reconstruction_error}",
        "  Post-ComBat ROI aggregation is not expected to equal the",
        "  separately fitted system-level ComBat output exactly.",
        f"  Minimum system-wise post-ComBat Pearson r: {minimum_combat_system_r}",
        "",
        "Primary output for the next analysis",
        f"  {OUT_COMBAT_SCALED}",
        "",
        "Other outputs",
        f"  Raw matched table: {OUT_RAW_MATCHED}",
        f"  ComBat raw table: {OUT_COMBAT_RAW}",
        f"  ComBat scaled table: {OUT_COMBAT_SCALED}",
        f"  Long table: {OUT_COMBAT_LONG}",
        f"  Feature report: {OUT_FEATURE_REPORT}",
        f"  ComBat model estimates: {OUT_COMBAT_MODEL}",
        f"  Subject order: {OUT_SUBJECT_ORDER}",
        f"  Subject-match audit: {OUT_MATCH_AUDIT}",
        f"  System aggregation detail: {OUT_SYSTEM_QC_DETAIL}",
        f"  System aggregation summary: {OUT_SYSTEM_QC_SUMMARY}",
        f"  ROI-set audit: {OUT_ROI_SET_AUDIT}",
    ]

    write_text(
        OUT_REPORT,
        "\n".join(report_lines),
    )

    print("")
    print("=" * 80)
    print("ROI-level ComBat completed")
    print("=" * 80)
    print(f"Subjects: {matched.shape[0]}")
    print(f"ROI features: {n_total_features}")
    print(f"ComBat-adjusted raw output: {OUT_COMBAT_RAW}")
    print(f"ComBat-adjusted scaled output: {OUT_COMBAT_SCALED}")
    print(f"System aggregation QC: {OUT_SYSTEM_QC_SUMMARY}")
    print(f"Report: {OUT_REPORT}")


if __name__ == "__main__":
    main()
