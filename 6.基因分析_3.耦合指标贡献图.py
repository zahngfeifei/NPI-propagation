# -*- coding: utf-8 -*-
"""
Gene analysis — revised Step 3
Construct exact parcel-wise contributions to the ORIGINAL four-system
behavior coupling beta, using only ROIs covered by H1--H4.

Original behavior model
-----------------------
For participant s and hierarchy system h:

    y_sh = alpha_s + beta_s * G_star_sh + error_sh

where h = H1, H2, H3, H4 and:

    y_sh = original system-level early_slope_scaled
    G_star_sh = original participant-specific system gradient position

The original behavior coupling index is:

    beta_s =
        sum_h[(G_star_sh - mean_h G_star_sh)
              * (y_sh - mean_h y_sh)]
        / sum_h[(G_star_sh - mean_h G_star_sh)^2]

Exact ROI decomposition
-----------------------
ROI-level ComBat slopes do not exactly equal the separately ComBat-adjusted
system slopes after aggregation. Therefore, within each system, the ROI slopes
are shifted by a single participant-specific constant so that their system mean
exactly equals the ORIGINAL system-level early_slope_scaled:

    u_calibrated_sp =
        u_ROI_ComBat_sp
        + [y_sh - mean_{p in H_h}(u_ROI_ComBat_sp)]

This preserves all within-system ROI differences while enforcing:

    mean_{p in H_h}(u_calibrated_sp) = y_sh

For ROI p belonging to system h, define:

    contribution_sp =
        [(G_star_sh - mean_h G_star_sh)
         / sum_k(G_star_sk - mean_h G_star_s)^2]
        * [(u_calibrated_sp - mean_h y_sh) / n_h]

Then:

    sum_{p in H1 union H2 union H3 union H4}
        contribution_sp
    = original behavior beta_s

Only ROIs covered by H1--H4 are retained. Uncovered Schaefer400 ROIs are not
included and are not filled with zero.

Important
---------
- This script does not modify or rerun the behavior/ADOS analyses.
- It exactly decomposes the original four-system behavior coupling beta.
- Parcel values are additive algebraic contributions, not independently
  estimated parcel-specific slopes.
"""

from __future__ import annotations

import re
from datetime import datetime
from pathlib import Path
from typing import Dict, List, Optional, Sequence, Tuple

import numpy as np
import pandas as pd


PROJECT_ROOT = Path(__file__).resolve().parent


# ============================================================
# 1. PATH SETTINGS — ALL ABSOLUTE PATHS
# ============================================================

# Step 2 输出：
# ROI 级 ComBat 校正后、再乘以 1000 的早期传播斜率。
ROI_COMBAT_SCALED_CSV = Path(
    PROJECT_ROOT
    / "ABIDE2_主流程结果2_ROI传播指标ComBat"
    / "roi-early-slope-combat-scaled-wide.csv"
)

# 原始四系统长表。
# 该文件必须是计算原始 Coupling_index_scaled 时使用的同一张表。
# 必需列：
#   sub_id, system, G_star, early_slope_scaled
ORIGINAL_LONG_CSV = Path(
    PROJECT_ROOT / "ABIDE2_结果1" / "result1_input_long_table_strict.csv"
)

# 保存有原始 Coupling_index_scaled 的被试级行为指标表。
# 仅用于外部一致性验证；若文件不存在，程序仍可继续运行。
ORIGINAL_BEHAVIOR_SUBJECT_CSV: Optional[Path] = None

# ROI 与 H1--H4 层级对应表。
# 必需列：
#   ROI_index_0based, Hierarchy
ROI_DEFINITIONS_CSV = Path(
    PROJECT_ROOT / "ABIDE2_主流程必要输入" / "roi-definitions.csv"
)

# Step 3 输出目录。
OUT_DIR = Path(
    PROJECT_ROOT / "ABIDE2_主流程结果3_耦合指标贡献"
)
OUT_DIR.mkdir(
    parents=True,
    exist_ok=True,
)


# ============================================================
# 2. ANALYSIS SETTINGS
# ============================================================

EXPECTED_TOTAL_N_ROI: Optional[int] = 400

SYSTEM_ORDER = [
    "H1_sensory",
    "H2_attention",
    "H3_control",
    "H4_DMN",
]

SYSTEM_FILES = {
    "H1_sensory": "H1_sensory.csv",
    "H2_attention": "H2_attention.csv",
    "H3_control": "H3_control.csv",
    "H4_DMN": "H4_DMN.csv",
}

# Explicitly set the index convention used in the ROI-definition CSV files.
# Keep 0 when the files contain ROI_index_0based.
ROI_INDEX_BASE = 0

# Exact closure tolerances.
DECOMPOSITION_ABS_TOL = 1e-10
ORIGINAL_OLS_ABS_TOL = 1e-10
SAVED_BEHAVIOR_ABS_TOL = 1e-10
SYSTEM_MEAN_CALIBRATION_ABS_TOL = 1e-10

MIN_GSTAR_SUM_SQUARES = 1e-12

CONTRIBUTION_COLUMN_TEMPLATE = (
    "ROI_{roi:03d}_exact_behavior_beta_contribution"
)

ROI_SLOPE_PATTERN = re.compile(
    r"^ROI_(\d+)_early_slope_1_10_scaled$",
    flags=re.IGNORECASE,
)


# ============================================================
# 3. OUTPUT FILES
# ============================================================

OUT_CONTRIBUTION_WIDE = (
    OUT_DIR / "h1-h4-exact-behavior-beta-contribution-wide.csv"
)

OUT_CONTRIBUTION_LONG = (
    OUT_DIR / "h1-h4-exact-behavior-beta-contribution-long.csv"
)

OUT_SUBJECT_BETA = (
    OUT_DIR / "h1-h4-exact-behavior-beta-subject-qc.csv"
)

OUT_CONTRIBUTION_NPY = (
    OUT_DIR / "h1-h4-exact-behavior-beta-contribution.npy"
)

OUT_SELECTED_ROI_ORDER = (
    OUT_DIR / "h1-h4-selected-roi-order.csv"
)

OUT_SUBJECT_ORDER = (
    OUT_DIR / "h1-h4-exact-behavior-beta-subject-order.csv"
)

OUT_ROI_DICTIONARY = (
    OUT_DIR / "h1-h4-exact-behavior-beta-roi-dictionary.csv"
)

OUT_ROI_SET_AUDIT = (
    OUT_DIR / "h1-h4-roi-set-audit.csv"
)

OUT_SUBJECT_MATCH_AUDIT = (
    OUT_DIR / "h1-h4-exact-behavior-beta-subject-match-audit.csv"
)

OUT_FAILED = (
    OUT_DIR / "h1-h4-exact-behavior-beta-failed-subjects.csv"
)

OUT_QC = (
    OUT_DIR / "h1-h4-exact-behavior-beta-contribution-qc-report.txt"
)


# ============================================================
# 4. GENERAL UTILITIES
# ============================================================

def normalize_sub_id(value) -> Optional[str]:
    """
    Normalize subject IDs to:
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
    table.columns = [
        str(column).strip()
        for column in table.columns
    ]
    return table


def write_text(path: Path, text: str) -> None:
    with open(path, "w", encoding="utf-8") as file:
        file.write(text)


def first_existing_subject_id_column(
    table: pd.DataFrame,
) -> str:
    for column in [
        "sub_id",
        "SUB_ID",
        "FILE_ID",
        "SUB_ID_norm",
    ]:
        if column in table.columns:
            return column

    raise ValueError(
        "No subject-ID column found. Expected one of: "
        "sub_id, SUB_ID, FILE_ID, SUB_ID_norm."
    )


# ============================================================
# 5. LOAD ROI-LEVEL COMBAT SLOPES
# ============================================================

def discover_roi_slope_columns(
    table: pd.DataFrame,
) -> Tuple[Dict[int, str], List[int]]:
    parsed: List[Tuple[int, str]] = []

    for column in table.columns:
        match = ROI_SLOPE_PATTERN.fullmatch(
            str(column)
        )

        if match:
            parsed.append(
                (
                    int(match.group(1)),
                    str(column),
                )
            )

    if len(parsed) == 0:
        raise ValueError(
            "No ROI early-slope columns found. Expected names such as "
            "ROI_000_early_slope_1_10_scaled."
        )

    parsed = sorted(
        parsed,
        key=lambda item: item[0],
    )

    roi_indices = [
        item[0]
        for item in parsed
    ]

    if len(set(roi_indices)) != len(roi_indices):
        raise ValueError(
            "Duplicate ROI indices found in ROI slope columns."
        )

    if (
        EXPECTED_TOTAL_N_ROI is not None
        and len(roi_indices) != EXPECTED_TOTAL_N_ROI
    ):
        raise ValueError(
            f"Expected {EXPECTED_TOTAL_N_ROI} ROI slope columns, "
            f"but found {len(roi_indices)}."
        )

    if roi_indices != list(range(len(roi_indices))):
        raise ValueError(
            "ROI slope columns are not a complete consecutive "
            "0-based sequence."
        )

    column_map = {
        roi: column
        for roi, column in parsed
    }

    return column_map, roi_indices


def load_roi_combat_table(
    path: Path,
) -> Tuple[pd.DataFrame, Dict[int, str], List[str]]:
    if not path.exists():
        raise FileNotFoundError(
            f"ROI ComBat scaled CSV not found: {path}"
        )

    table = pd.read_csv(
        path,
        low_memory=False,
        encoding="utf-8-sig",
    )

    table = clean_columns(table)

    if "sub_id" not in table.columns:
        raise ValueError(
            "ROI ComBat table must contain sub_id."
        )

    table["sub_id"] = (
        table["sub_id"]
        .apply(normalize_sub_id)
    )

    table = table.dropna(
        subset=["sub_id"]
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
            "Duplicate subjects found in ROI ComBat table. "
            f"Examples: {examples}"
        )

    roi_column_map, _ = (
        discover_roi_slope_columns(
            table
        )
    )

    roi_slope_columns = list(
        roi_column_map.values()
    )

    for column in roi_slope_columns:
        table[column] = pd.to_numeric(
            table[column],
            errors="coerce",
        )

    matrix = table[
        roi_slope_columns
    ].to_numpy(
        dtype=np.float64
    )

    if not np.isfinite(matrix).all():
        n_bad = int(
            np.sum(
                ~np.isfinite(matrix)
            )
        )

        raise ValueError(
            f"ROI ComBat matrix contains {n_bad} NaN/Inf values."
        )

    metadata_columns = [
        column
        for column in table.columns
        if column not in roi_slope_columns
    ]

    return table, roi_column_map, metadata_columns


# ============================================================
# 6. LOAD H1--H4 ROI DEFINITIONS
# ============================================================

def read_roi_indices(
    path: Path,
) -> np.ndarray:
    if not path.exists():
        raise FileNotFoundError(
            f"ROI definition file not found: {path}"
        )

    table = pd.read_csv(
        path,
        encoding="utf-8-sig",
    )

    table = clean_columns(table)

    if table.empty:
        raise ValueError(
            f"ROI definition file is empty: {path}"
        )

    preferred_columns = [
        "ROI_index_0based",
        "roi_index_0based",
        "ROI_0based",
        "roi",
        "ROI_ID",
    ]

    column = None

    for candidate in preferred_columns:
        if candidate in table.columns:
            column = candidate
            break

    if column is None:
        column = table.columns[0]

    indices = (
        pd.to_numeric(
            table[column],
            errors="coerce",
        )
        .dropna()
        .astype(int)
        .to_numpy()
    )

    indices = indices - ROI_INDEX_BASE

    indices = np.sort(
        np.unique(indices)
    )

    return indices


def load_system_roi_sets(
    n_total_roi: int,
) -> Tuple[
    Dict[str, np.ndarray],
    np.ndarray,
    pd.DataFrame,
]:
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

    精确加和分解要求：
    - 每个选中 ROI 只能属于一个系统；
    - 未被 H1--H4 覆盖的 ROI 不进入贡献矩阵；
    - 不会给未覆盖 ROI 填 0。
    """
    if not ROI_DEFINITIONS_CSV.exists():
        raise FileNotFoundError(
            f"ROI definition file not found: "
            f"{ROI_DEFINITIONS_CSV}"
        )

    roi_definition_table = pd.read_csv(
        ROI_DEFINITIONS_CSV,
        low_memory=False,
        encoding="utf-8-sig",
    )
    roi_definition_table = clean_columns(
        roi_definition_table
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

    roi_definition_table[
        "ROI_index_0based"
    ] = pd.to_numeric(
        roi_definition_table[
            "ROI_index_0based"
        ],
        errors="coerce",
    )

    system_rois: Dict[str, np.ndarray] = {}

    membership_count = np.zeros(
        n_total_roi,
        dtype=int,
    )

    audit_rows = []

    for system in SYSTEM_ORDER:
        indices = (
            roi_definition_table.loc[
                roi_definition_table[
                    "Hierarchy"
                ].eq(system),
                "ROI_index_0based",
            ]
            .dropna()
            .astype(int)
            .to_numpy()
        )

        indices = np.sort(
            np.unique(indices)
        )

        invalid = indices[
            (indices < 0)
            | (indices >= n_total_roi)
        ]

        if len(invalid) > 0:
            raise ValueError(
                f"{system} contains invalid ROI indices. "
                f"Expected 0..{n_total_roi - 1}; "
                f"examples={invalid[:10].tolist()}"
            )

        if len(indices) == 0:
            raise ValueError(
                f"{system} contains no ROIs."
            )

        system_rois[system] = indices
        membership_count[indices] += 1

        audit_rows.append(
            {
                "record_type": "system_summary",
                "system": system,
                "roi_index_0based": np.nan,
                "n_roi": int(len(indices)),
                "membership_count": np.nan,
            }
        )

    overlapping = np.where(
        membership_count > 1
    )[0]

    if len(overlapping) > 0:
        raise ValueError(
            "H1--H4 ROI sets overlap. Exact additive "
            "decomposition requires every selected ROI "
            "to belong to exactly one system. "
            f"Overlapping examples: "
            f"{overlapping[:10].tolist()}"
        )

    selected_rois = np.where(
        membership_count == 1
    )[0]

    uncovered = np.where(
        membership_count == 0
    )[0]

    roi_to_system = {}

    for system, indices in system_rois.items():
        for roi in indices:
            roi_to_system[int(roi)] = system

    for roi in range(n_total_roi):
        audit_rows.append(
            {
                "record_type": "roi_membership",
                "system": roi_to_system.get(
                    roi,
                    "",
                ),
                "roi_index_0based": roi,
                "n_roi": np.nan,
                "membership_count": int(
                    membership_count[roi]
                ),
            }
        )

    audit_rows.extend(
        [
            {
                "record_type": "overall_summary",
                "system": "SELECTED_UNION",
                "roi_index_0based": np.nan,
                "n_roi": int(len(selected_rois)),
                "membership_count": np.nan,
            },
            {
                "record_type": "overall_summary",
                "system": "UNCOVERED",
                "roi_index_0based": np.nan,
                "n_roi": int(len(uncovered)),
                "membership_count": np.nan,
            },
            {
                "record_type": "overall_summary",
                "system": "OVERLAPPING",
                "roi_index_0based": np.nan,
                "n_roi": int(len(overlapping)),
                "membership_count": np.nan,
            },
        ]
    )

    audit = pd.DataFrame(
        audit_rows
    )

    audit.to_csv(
        OUT_ROI_SET_AUDIT,
        index=False,
        encoding="utf-8-sig",
    )

    return (
        system_rois,
        selected_rois,
        audit,
    )


# ============================================================
# 7. LOAD ORIGINAL FOUR-SYSTEM LONG TABLE
# ============================================================

def load_original_long(
    path: Path,
) -> pd.DataFrame:
    if not path.exists():
        raise FileNotFoundError(
            f"Original long table not found: {path}"
        )

    table = pd.read_csv(
        path,
        low_memory=False,
        encoding="utf-8-sig",
    )

    table = clean_columns(table)

    required_columns = [
        "sub_id",
        "system",
        "G_star",
        "early_slope_scaled",
    ]

    missing = [
        column
        for column in required_columns
        if column not in table.columns
    ]

    if missing:
        raise ValueError(
            f"Original long table is missing columns: {missing}"
        )

    table = table.copy()

    table["sub_id"] = (
        table["sub_id"]
        .apply(normalize_sub_id)
    )

    table["system"] = (
        table["system"]
        .astype(str)
        .str.strip()
    )

    table = table[
        table["system"].isin(
            SYSTEM_ORDER
        )
    ].copy()

    table["G_star"] = pd.to_numeric(
        table["G_star"],
        errors="coerce",
    )

    table["early_slope_scaled"] = (
        pd.to_numeric(
            table["early_slope_scaled"],
            errors="coerce",
        )
    )

    table = table.dropna(
        subset=[
            "sub_id",
            "system",
            "G_star",
            "early_slope_scaled",
        ]
    ).copy()

    duplicated = table[
        table.duplicated(
            subset=[
                "sub_id",
                "system",
            ],
            keep=False,
        )
    ]

    if not duplicated.empty:
        examples = duplicated[
            [
                "sub_id",
                "system",
            ]
        ].head(10)

        raise ValueError(
            "Duplicate subject-system rows found in original long table:\n"
            f"{examples}"
        )

    system_counts = table.groupby(
        "sub_id"
    )["system"].nunique()

    incomplete_subjects = system_counts[
        system_counts != len(
            SYSTEM_ORDER
        )
    ]

    if len(incomplete_subjects) > 0:
        raise ValueError(
            f"{len(incomplete_subjects)} subjects do not have all four "
            "systems in the original long table."
        )

    return table


# ============================================================
# 8. OPTIONAL SAVED BEHAVIOR INDEX
# ============================================================

def load_saved_behavior_index() -> Optional[pd.DataFrame]:
    path = ORIGINAL_BEHAVIOR_SUBJECT_CSV

    if path is None or not path.exists():
        return None

    table = pd.read_csv(
        path,
        low_memory=False,
        encoding="utf-8-sig",
    )

    table = clean_columns(table)

    if "Coupling_index_scaled" not in table.columns:
        raise ValueError(
            f"{path} does not contain Coupling_index_scaled."
        )

    subject_id_column = (
        first_existing_subject_id_column(
            table
        )
    )

    output = pd.DataFrame(
        {
            "sub_id":
                table[
                    subject_id_column
                ].apply(
                    normalize_sub_id
                ),
            "saved_Coupling_index_scaled":
                pd.to_numeric(
                    table[
                        "Coupling_index_scaled"
                    ],
                    errors="coerce",
                ),
        }
    )

    output = (
        output.dropna(
            subset=[
                "sub_id",
                "saved_Coupling_index_scaled",
            ]
        )
        .drop_duplicates(
            subset=["sub_id"],
            keep="first",
        )
    )

    return output


# ============================================================
# 9. SUBJECT MATCH AUDIT
# ============================================================

def build_subject_match_audit(
    roi_subjects: Sequence[str],
    long_subjects: Sequence[str],
    saved_subjects: Optional[Sequence[str]],
) -> pd.DataFrame:
    roi_set = set(roi_subjects)
    long_set = set(long_subjects)

    saved_set = (
        set(saved_subjects)
        if saved_subjects is not None
        else set()
    )

    all_subjects = sorted(
        roi_set
        | long_set
        | saved_set
    )

    rows = []

    for subject in all_subjects:
        rows.append(
            {
                "sub_id": subject,
                "in_ROI_ComBat_table":
                    subject in roi_set,
                "in_original_long_table":
                    subject in long_set,
                "in_saved_behavior_table":
                    (
                        subject in saved_set
                        if saved_subjects is not None
                        else np.nan
                    ),
                "in_primary_intersection":
                    (
                        subject in roi_set
                        and subject in long_set
                    ),
            }
        )

    audit = pd.DataFrame(
        rows
    )

    audit.to_csv(
        OUT_SUBJECT_MATCH_AUDIT,
        index=False,
        encoding="utf-8-sig",
    )

    return audit


# ============================================================
# 10. EXACT SUBJECT-LEVEL DECOMPOSITION
# ============================================================

def calculate_exact_behavior_contribution(
    original_subject_system: pd.DataFrame,
    roi_slope_row: pd.Series,
    roi_column_map: Dict[int, str],
    system_rois: Dict[str, np.ndarray],
) -> dict:
    """
    Calculate exact H1--H4 ROI contributions to the original behavior beta.
    """
    system_table = (
        original_subject_system
        .set_index("system")
        .reindex(SYSTEM_ORDER)
    )

    if system_table[
        [
            "G_star",
            "early_slope_scaled",
        ]
    ].isna().any().any():
        raise ValueError(
            "Missing original G_star or early_slope_scaled."
        )

    g_star = system_table[
        "G_star"
    ].to_numpy(
        dtype=np.float64
    )

    original_system_slope = system_table[
        "early_slope_scaled"
    ].to_numpy(
        dtype=np.float64
    )

    g_star_mean = float(
        np.mean(g_star)
    )

    g_star_centered = (
        g_star - g_star_mean
    )

    system_slope_mean = float(
        np.mean(original_system_slope)
    )

    system_slope_centered = (
        original_system_slope
        - system_slope_mean
    )

    denominator = float(
        np.sum(
            g_star_centered ** 2
        )
    )

    if (
        not np.isfinite(denominator)
        or denominator
        <= MIN_GSTAR_SUM_SQUARES
    ):
        raise ValueError(
            "G_star centered sum of squares is zero or near zero."
        )

    original_beta_closed_form = float(
        np.sum(
            g_star_centered
            * system_slope_centered
        )
        / denominator
    )

    design = np.column_stack(
        [
            np.ones(
                len(SYSTEM_ORDER),
                dtype=float,
            ),
            g_star,
        ]
    )

    coefficients = np.linalg.lstsq(
        design,
        original_system_slope,
        rcond=None,
    )[0]

    original_intercept_lstsq = float(
        coefficients[0]
    )

    original_beta_lstsq = float(
        coefficients[1]
    )

    original_ols_error = (
        original_beta_closed_form
        - original_beta_lstsq
    )

    contribution_by_roi: Dict[int, float] = {}
    roi_detail_rows: List[dict] = []
    system_qc_rows: List[dict] = []

    bridge_beta_uncalibrated = 0.0

    for system_index, system in enumerate(
        SYSTEM_ORDER
    ):
        rois = system_rois[
            system
        ]

        n_system_roi = int(
            len(rois)
        )

        raw_roi_values = np.array(
            [
                float(
                    roi_slope_row[
                        roi_column_map[int(roi)]
                    ]
                )
                for roi in rois
            ],
            dtype=np.float64,
        )

        if not np.isfinite(
            raw_roi_values
        ).all():
            raise ValueError(
                f"Non-finite ROI slope in {system}."
            )

        raw_roi_mean = float(
            np.mean(
                raw_roi_values
            )
        )

        target_system_mean = float(
            original_system_slope[
                system_index
            ]
        )

        calibration_shift = (
            target_system_mean
            - raw_roi_mean
        )

        calibrated_roi_values = (
            raw_roi_values
            + calibration_shift
        )

        calibrated_system_mean = float(
            np.mean(
                calibrated_roi_values
            )
        )

        calibration_error = (
            calibrated_system_mean
            - target_system_mean
        )

        system_weight = float(
            g_star_centered[
                system_index
            ]
            / denominator
        )

        parcel_weight = (
            system_weight
            / n_system_roi
        )

        centered_calibrated_values = (
            calibrated_roi_values
            - system_slope_mean
        )

        system_contributions = (
            parcel_weight
            * centered_calibrated_values
        )

        bridge_beta_uncalibrated += (
            system_weight
            * (
                raw_roi_mean
                - np.mean(
                    [
                        np.mean(
                            [
                                float(
                                    roi_slope_row[
                                        roi_column_map[
                                            int(other_roi)
                                        ]
                                    ]
                                )
                                for other_roi
                                in system_rois[
                                    other_system
                                ]
                            ]
                        )
                        for other_system
                        in SYSTEM_ORDER
                    ]
                )
            )
        )

        for local_index, roi in enumerate(
            rois
        ):
            roi_int = int(roi)

            contribution_value = float(
                system_contributions[
                    local_index
                ]
            )

            contribution_by_roi[
                roi_int
            ] = contribution_value

            roi_detail_rows.append(
                {
                    "roi_index_0based":
                        roi_int,
                    "system":
                        system,
                    "n_roi_in_system":
                        n_system_roi,
                    "G_star":
                        float(
                            g_star[
                                system_index
                            ]
                        ),
                    "G_star_centered":
                        float(
                            g_star_centered[
                                system_index
                            ]
                        ),
                    "G_star_centered_sum_squares":
                        denominator,
                    "system_OLS_weight":
                        system_weight,
                    "parcel_OLS_weight":
                        parcel_weight,
                    "original_system_early_slope_scaled":
                        target_system_mean,
                    "mean_original_system_early_slope_scaled":
                        system_slope_mean,
                    "ROI_early_slope_ComBat_scaled_raw":
                        float(
                            raw_roi_values[
                                local_index
                            ]
                        ),
                    "ROI_system_mean_before_calibration":
                        raw_roi_mean,
                    "ROI_to_original_system_calibration_shift":
                        calibration_shift,
                    "ROI_early_slope_calibrated":
                        float(
                            calibrated_roi_values[
                                local_index
                            ]
                        ),
                    "ROI_early_slope_calibrated_centered":
                        float(
                            centered_calibrated_values[
                                local_index
                            ]
                        ),
                    "exact_behavior_beta_contribution":
                        contribution_value,
                }
            )

        system_qc_rows.append(
            {
                "system": system,
                "n_roi": n_system_roi,
                "original_system_early_slope_scaled":
                    target_system_mean,
                "ROI_system_mean_before_calibration":
                    raw_roi_mean,
                "calibration_shift":
                    calibration_shift,
                "ROI_system_mean_after_calibration":
                    calibrated_system_mean,
                "calibration_error":
                    calibration_error,
                "G_star":
                    float(
                        g_star[
                            system_index
                        ]
                    ),
                "G_star_centered":
                    float(
                        g_star_centered[
                            system_index
                        ]
                    ),
                "system_OLS_weight":
                    system_weight,
                "sum_system_ROI_contributions":
                    float(
                        np.sum(
                            system_contributions
                        )
                    ),
                "expected_system_numerator_contribution":
                    float(
                        system_weight
                        * system_slope_centered[
                            system_index
                        ]
                    ),
            }
        )

    contribution_sum = float(
        np.sum(
            list(
                contribution_by_roi.values()
            )
        )
    )

    decomposition_error = (
        contribution_sum
        - original_beta_closed_form
    )

    max_system_calibration_error = float(
        np.max(
            np.abs(
                [
                    row[
                        "calibration_error"
                    ]
                    for row in system_qc_rows
                ]
            )
        )
    )

    max_system_contribution_error = float(
        np.max(
            np.abs(
                [
                    row[
                        "sum_system_ROI_contributions"
                    ]
                    - row[
                        "expected_system_numerator_contribution"
                    ]
                    for row in system_qc_rows
                ]
            )
        )
    )

    return {
        "contribution_by_roi":
            contribution_by_roi,
        "roi_detail_rows":
            roi_detail_rows,
        "system_qc_rows":
            system_qc_rows,
        "original_beta_closed_form":
            original_beta_closed_form,
        "original_intercept_lstsq":
            original_intercept_lstsq,
        "original_beta_lstsq":
            original_beta_lstsq,
        "original_ols_error":
            original_ols_error,
        "contribution_sum":
            contribution_sum,
        "decomposition_error":
            decomposition_error,
        "max_system_calibration_error":
            max_system_calibration_error,
        "max_system_contribution_error":
            max_system_contribution_error,
    }


# ============================================================
# 11. MAIN
# ============================================================

def main() -> None:
    print("=" * 80)
    print(
        "Constructing exact H1--H4 ROI contributions "
        "to the original behavior coupling beta"
    )
    print("=" * 80)

    required_input_paths = {
        "ROI_COMBAT_SCALED_CSV":
            ROI_COMBAT_SCALED_CSV,
        "ORIGINAL_LONG_CSV":
            ORIGINAL_LONG_CSV,
        "ROI_DEFINITIONS_CSV":
            ROI_DEFINITIONS_CSV,
    }

    missing_required_paths = {
        name: path
        for name, path in required_input_paths.items()
        if not Path(path).exists()
    }

    if missing_required_paths:
        message_lines = [
            "The following required input paths "
            "do not exist:"
        ]

        for name, path in (
            missing_required_paths.items()
        ):
            message_lines.append(
                f"  {name}: {path}"
            )

        raise FileNotFoundError(
            "\n".join(message_lines)
        )

    if (
        ORIGINAL_BEHAVIOR_SUBJECT_CSV
        is not None
        and not ORIGINAL_BEHAVIOR_SUBJECT_CSV.exists()
    ):
        print(
            "Warning: optional saved behavior file "
            "was not found. External verification "
            "of Coupling_index_scaled will be skipped:"
        )
        print(
            f"  {ORIGINAL_BEHAVIOR_SUBJECT_CSV}"
        )

    print(
        f"ROI ComBat scaled input: "
        f"{ROI_COMBAT_SCALED_CSV}"
    )
    print(
        f"Original four-system long table: "
        f"{ORIGINAL_LONG_CSV}"
    )
    print(
        f"ROI hierarchy mapping: "
        f"{ROI_DEFINITIONS_CSV}"
    )
    print(
        f"Optional saved behavior table: "
        f"{ORIGINAL_BEHAVIOR_SUBJECT_CSV}"
    )
    print(
        f"Output directory: {OUT_DIR}"
    )

    (
        roi_table,
        roi_column_map,
        metadata_columns,
    ) = load_roi_combat_table(
        ROI_COMBAT_SCALED_CSV
    )

    original_long = load_original_long(
        ORIGINAL_LONG_CSV
    )

    saved_behavior = (
        load_saved_behavior_index()
    )

    (
        system_rois,
        selected_rois,
        roi_set_audit,
    ) = load_system_roi_sets(
        n_total_roi=len(
            roi_column_map
        )
    )

    selected_roi_records = []

    roi_to_system = {}

    for system in SYSTEM_ORDER:
        for roi in system_rois[
            system
        ]:
            roi_int = int(roi)
            roi_to_system[
                roi_int
            ] = system

    for output_position, roi in enumerate(
        selected_rois
    ):
        roi_int = int(roi)

        selected_roi_records.append(
            {
                "selected_output_position_0based":
                    output_position,
                "roi_index_0based":
                    roi_int,
                "ROI_ID_1based":
                    roi_int + 1,
                "system":
                    roi_to_system[
                        roi_int
                    ],
                "input_slope_column":
                    roi_column_map[
                        roi_int
                    ],
                "contribution_column":
                    CONTRIBUTION_COLUMN_TEMPLATE.format(
                        roi=roi_int
                    ),
            }
        )

    selected_roi_order = pd.DataFrame(
        selected_roi_records
    )

    selected_roi_order.to_csv(
        OUT_SELECTED_ROI_ORDER,
        index=False,
        encoding="utf-8-sig",
    )

    saved_subjects = (
        saved_behavior["sub_id"].tolist()
        if saved_behavior is not None
        else None
    )

    match_audit = build_subject_match_audit(
        roi_subjects=
            roi_table["sub_id"].tolist(),
        long_subjects=
            original_long["sub_id"]
            .drop_duplicates()
            .tolist(),
        saved_subjects=
            saved_subjects,
    )

    primary_subjects = sorted(
        set(
            roi_table["sub_id"]
        )
        & set(
            original_long["sub_id"]
        )
    )

    roi_indexed = roi_table.set_index(
        "sub_id",
        drop=False,
    )

    original_groups = {
        subject: group.copy()
        for subject, group in original_long.groupby(
            "sub_id",
            sort=False,
        )
    }

    saved_map = {}

    if saved_behavior is not None:
        saved_map = dict(
            zip(
                saved_behavior[
                    "sub_id"
                ],
                saved_behavior[
                    "saved_Coupling_index_scaled"
                ],
            )
        )

    contribution_rows: List[dict] = []
    long_rows: List[dict] = []
    beta_rows: List[dict] = []
    failed_rows: List[dict] = []

    contribution_columns = [
        CONTRIBUTION_COLUMN_TEMPLATE.format(
            roi=int(roi)
        )
        for roi in selected_rois
    ]

    for subject_number, sub_id in enumerate(
        primary_subjects,
        start=1,
    ):
        try:
            roi_row = roi_indexed.loc[
                sub_id
            ]

            original_subject_system = (
                original_groups[
                    sub_id
                ]
            )

            result = (
                calculate_exact_behavior_contribution(
                    original_subject_system=
                        original_subject_system,
                    roi_slope_row=
                        roi_row,
                    roi_column_map=
                        roi_column_map,
                    system_rois=
                        system_rois,
                )
            )

            metadata = {
                column: roi_row[column]
                for column in metadata_columns
                if column in roi_row.index
            }

            contribution_row = dict(
                metadata
            )

            for roi in selected_rois:
                roi_int = int(roi)

                contribution_row[
                    CONTRIBUTION_COLUMN_TEMPLATE.format(
                        roi=roi_int
                    )
                ] = result[
                    "contribution_by_roi"
                ][roi_int]

            contribution_rows.append(
                contribution_row
            )

            saved_beta = saved_map.get(
                sub_id,
                np.nan,
            )

            saved_error = (
                result[
                    "original_beta_closed_form"
                ]
                - saved_beta
                if np.isfinite(
                    saved_beta
                )
                else np.nan
            )

            beta_rows.append(
                {
                    **metadata,
                    "original_behavior_beta_closed_form":
                        result[
                            "original_beta_closed_form"
                        ],
                    "original_behavior_intercept_lstsq":
                        result[
                            "original_intercept_lstsq"
                        ],
                    "original_behavior_beta_lstsq":
                        result[
                            "original_beta_lstsq"
                        ],
                    "closed_form_minus_lstsq":
                        result[
                            "original_ols_error"
                        ],
                    "sum_H1H4_ROI_exact_contributions":
                        result[
                            "contribution_sum"
                        ],
                    "contribution_sum_minus_original_beta":
                        result[
                            "decomposition_error"
                        ],
                    "saved_Coupling_index_scaled":
                        saved_beta,
                    "recomputed_minus_saved_behavior_beta":
                        saved_error,
                    "max_system_mean_calibration_error":
                        result[
                            "max_system_calibration_error"
                        ],
                    "max_system_contribution_closure_error":
                        result[
                            "max_system_contribution_error"
                        ],
                    "exact_decomposition_pass":
                        bool(
                            abs(
                                result[
                                    "decomposition_error"
                                ]
                            )
                            <= DECOMPOSITION_ABS_TOL
                        ),
                    "original_OLS_verification_pass":
                        bool(
                            abs(
                                result[
                                    "original_ols_error"
                                ]
                            )
                            <= ORIGINAL_OLS_ABS_TOL
                        ),
                    "saved_behavior_verification_pass":
                        (
                            bool(
                                abs(
                                    saved_error
                                )
                                <= SAVED_BEHAVIOR_ABS_TOL
                            )
                            if np.isfinite(
                                saved_error
                            )
                            else np.nan
                        ),
                    "system_mean_calibration_pass":
                        bool(
                            result[
                                "max_system_calibration_error"
                            ]
                            <= SYSTEM_MEAN_CALIBRATION_ABS_TOL
                        ),
                }
            )

            system_qc_map = {
                row["system"]: row
                for row in result[
                    "system_qc_rows"
                ]
            }

            for detail in result[
                "roi_detail_rows"
            ]:
                long_rows.append(
                    {
                        **metadata,
                        **detail,
                    }
                )

            if (
                subject_number % 25 == 0
                or subject_number
                == len(primary_subjects)
            ):
                print(
                    f"Processed "
                    f"{subject_number}/"
                    f"{len(primary_subjects)}"
                )

        except Exception as error:
            failed_rows.append(
                {
                    "sub_id": sub_id,
                    "reason": repr(error),
                }
            )

    if len(contribution_rows) == 0:
        raise RuntimeError(
            "No subjects were successfully processed."
        )

    contribution_wide = pd.DataFrame(
        contribution_rows
    )

    contribution_wide.to_csv(
        OUT_CONTRIBUTION_WIDE,
        index=False,
        encoding="utf-8-sig",
    )

    long_table = pd.DataFrame(
        long_rows
    )

    long_table.to_csv(
        OUT_CONTRIBUTION_LONG,
        index=False,
        encoding="utf-8-sig",
    )

    beta_table = pd.DataFrame(
        beta_rows
    )

    beta_table.to_csv(
        OUT_SUBJECT_BETA,
        index=False,
        encoding="utf-8-sig",
    )

    failed_table = pd.DataFrame(
        failed_rows,
        columns=[
            "sub_id",
            "reason",
        ],
    )

    failed_table.to_csv(
        OUT_FAILED,
        index=False,
        encoding="utf-8-sig",
    )

    contribution_matrix = (
        contribution_wide[
            contribution_columns
        ]
        .to_numpy(
            dtype=np.float64
        )
    )

    np.save(
        OUT_CONTRIBUTION_NPY,
        contribution_matrix,
    )

    subject_order_columns = [
        column
        for column in metadata_columns
        if column in contribution_wide.columns
    ]

    contribution_wide[
        subject_order_columns
    ].to_csv(
        OUT_SUBJECT_ORDER,
        index=False,
        encoding="utf-8-sig",
    )

    dictionary_rows = []

    for record in selected_roi_records:
        dictionary_rows.append(
            {
                **record,
                "definition":
                    (
                        "[G_star_centered(system) / "
                        "sum_h(G_star_centered^2)] * "
                        "[(ROI_slope_calibrated - "
                        "mean_h(original_system_slope)) / "
                        "n_ROI_in_system]"
                    ),
                "calibration":
                    (
                        "Within each subject and system, add one "
                        "constant to all ROI slopes so their mean "
                        "equals original system early_slope_scaled."
                    ),
            }
        )

    pd.DataFrame(
        dictionary_rows
    ).to_csv(
        OUT_ROI_DICTIONARY,
        index=False,
        encoding="utf-8-sig",
    )

    # ========================================================
    # 12. QC REPORT
    # ========================================================

    decomposition_errors = pd.to_numeric(
        beta_table[
            "contribution_sum_minus_original_beta"
        ],
        errors="coerce",
    )

    ols_errors = pd.to_numeric(
        beta_table[
            "closed_form_minus_lstsq"
        ],
        errors="coerce",
    )

    calibration_errors = pd.to_numeric(
        beta_table[
            "max_system_mean_calibration_error"
        ],
        errors="coerce",
    )

    system_contribution_errors = pd.to_numeric(
        beta_table[
            "max_system_contribution_closure_error"
        ],
        errors="coerce",
    )

    saved_errors = pd.to_numeric(
        beta_table[
            "recomputed_minus_saved_behavior_beta"
        ],
        errors="coerce",
    )

    valid_saved = np.isfinite(
        saved_errors
    )

    n_decomposition_fail = int(
        np.sum(
            np.abs(
                decomposition_errors
            )
            > DECOMPOSITION_ABS_TOL
        )
    )

    n_ols_fail = int(
        np.sum(
            np.abs(
                ols_errors
            )
            > ORIGINAL_OLS_ABS_TOL
        )
    )

    n_calibration_fail = int(
        np.sum(
            calibration_errors
            > SYSTEM_MEAN_CALIBRATION_ABS_TOL
        )
    )

    if np.any(valid_saved):
        max_saved_error = float(
            np.max(
                np.abs(
                    saved_errors[
                        valid_saved
                    ]
                )
            )
        )

        n_saved_fail = int(
            np.sum(
                np.abs(
                    saved_errors[
                        valid_saved
                    ]
                )
                > SAVED_BEHAVIOR_ABS_TOL
            )
        )
    else:
        max_saved_error = np.nan
        n_saved_fail = 0

    membership_rows = roi_set_audit[
        roi_set_audit[
            "record_type"
        ] == "overall_summary"
    ]

    selected_count = int(
        membership_rows.loc[
            membership_rows["system"]
            == "SELECTED_UNION",
            "n_roi",
        ].iloc[0]
    )

    uncovered_count = int(
        membership_rows.loc[
            membership_rows["system"]
            == "UNCOVERED",
            "n_roi",
        ].iloc[0]
    )

    overlap_count = int(
        membership_rows.loc[
            membership_rows["system"]
            == "OVERLAPPING",
            "n_roi",
        ].iloc[0]
    )

    beta_values = pd.to_numeric(
        beta_table[
            "original_behavior_beta_closed_form"
        ],
        errors="coerce",
    )

    contribution_values = (
        contribution_matrix.ravel()
    )

    qc_lines = [
        "Exact H1--H4 ROI contribution to original behavior beta QC report",
        f"Generated at: {datetime.now()}",
        "",
        "Purpose",
        "  Construct parcel-wise additive contributions that sum exactly",
        "  to the original four-system Coupling_index_scaled.",
        "  Only ROIs covered by H1--H4 are included.",
        "",
        "Input",
        f"  ROI_COMBAT_SCALED_CSV: "
        f"{ROI_COMBAT_SCALED_CSV}",
        f"  ORIGINAL_LONG_CSV: "
        f"{ORIGINAL_LONG_CSV}",
        f"  ORIGINAL_BEHAVIOR_SUBJECT_CSV: "
        f"{ORIGINAL_BEHAVIOR_SUBJECT_CSV}",
        f"  ROI_DEFINITIONS_CSV: {ROI_DEFINITIONS_CSV}",
        f"  ROI_INDEX_BASE: {ROI_INDEX_BASE}",
        "",
        "ROI coverage",
        f"  Total Schaefer ROIs available: "
        f"{len(roi_column_map)}",
        f"  H1--H4 selected union ROIs: "
        f"{selected_count}",
        f"  Uncovered ROIs excluded: "
        f"{uncovered_count}",
        f"  Overlapping ROIs: "
        f"{overlap_count}",
        f"  Contribution matrix shape: "
        f"{contribution_matrix.shape}",
        "",
        "Sample",
        f"  ROI table subjects: "
        f"{roi_table.shape[0]}",
        f"  Original long-table subjects: "
        f"{original_long['sub_id'].nunique()}",
        f"  Primary matched subjects: "
        f"{len(primary_subjects)}",
        f"  Successful subjects: "
        f"{contribution_wide.shape[0]}",
        f"  Failed subjects: "
        f"{len(failed_rows)}",
        "",
        "Exact definition",
        "  For each subject/system, ROI ComBat slopes are shifted by",
        "  one constant so their system mean equals the original",
        "  system-level early_slope_scaled.",
        "  contribution_sp =",
        "    [G_star_centered_sh / sum_h(G_star_centered_sh^2)]",
        "    * [(ROI_slope_calibrated_sp - mean_h(y_sh)) / n_h]",
        "  Exact identity:",
        "    sum_{p in H1 union H2 union H3 union H4}",
        "      contribution_sp = original behavior beta_s",
        "",
        "Exact decomposition QC",
        f"  Tolerance: {DECOMPOSITION_ABS_TOL}",
        f"  Maximum absolute contribution-sum error: "
        f"{np.max(np.abs(decomposition_errors)):.16g}",
        f"  Subjects exceeding tolerance: "
        f"{n_decomposition_fail}",
        "",
        "Original four-system OLS verification",
        f"  Tolerance: {ORIGINAL_OLS_ABS_TOL}",
        f"  Maximum |closed-form beta - lstsq beta|: "
        f"{np.max(np.abs(ols_errors)):.16g}",
        f"  Subjects exceeding tolerance: "
        f"{n_ols_fail}",
        "",
        "System-mean calibration QC",
        f"  Tolerance: "
        f"{SYSTEM_MEAN_CALIBRATION_ABS_TOL}",
        f"  Maximum absolute calibrated-system-mean error: "
        f"{np.max(calibration_errors):.16g}",
        f"  Maximum within-system contribution closure error: "
        f"{np.max(system_contribution_errors):.16g}",
        f"  Subjects exceeding calibration tolerance: "
        f"{n_calibration_fail}",
        "",
        "Saved behavior-index verification",
        f"  Saved behavior file available: "
        f"{saved_behavior is not None}",
        f"  Subjects with saved behavior beta: "
        f"{int(np.sum(valid_saved))}",
        f"  Tolerance: {SAVED_BEHAVIOR_ABS_TOL}",
        f"  Maximum |recomputed - saved|: "
        f"{max_saved_error}",
        f"  Subjects exceeding tolerance: "
        f"{n_saved_fail}",
        "",
        "Original behavior beta distribution",
        f"  mean: {beta_values.mean():.12g}",
        f"  sd: {beta_values.std(ddof=0):.12g}",
        f"  min: {beta_values.min():.12g}",
        f"  max: {beta_values.max():.12g}",
        "",
        "Exact parcel contribution distribution",
        f"  mean: {np.mean(contribution_values):.12g}",
        f"  sd: {np.std(contribution_values):.12g}",
        f"  min: {np.min(contribution_values):.12g}",
        f"  max: {np.max(contribution_values):.12g}",
        "",
        "Interpretation",
        "  Positive parcel contribution supports a more positive",
        "  original four-system behavior coupling beta.",
        "  Negative parcel contribution reduces or reverses that beta.",
        "  Values are exact additive contributions, not independent",
        "  parcel-specific regression coefficients.",
        "  Uncovered Schaefer400 ROIs are absent, not assigned zero.",
        "",
        "Primary outputs",
        f"  Contribution wide CSV: "
        f"{OUT_CONTRIBUTION_WIDE}",
        f"  Contribution long CSV: "
        f"{OUT_CONTRIBUTION_LONG}",
        f"  Subject-level beta/QC: "
        f"{OUT_SUBJECT_BETA}",
        f"  Contribution NPY: "
        f"{OUT_CONTRIBUTION_NPY}",
        f"  Selected ROI order: "
        f"{OUT_SELECTED_ROI_ORDER}",
        f"  Subject order: "
        f"{OUT_SUBJECT_ORDER}",
        f"  ROI dictionary: "
        f"{OUT_ROI_DICTIONARY}",
        f"  ROI-set audit: "
        f"{OUT_ROI_SET_AUDIT}",
        f"  Subject-match audit: "
        f"{OUT_SUBJECT_MATCH_AUDIT}",
        f"  Failed subjects: "
        f"{OUT_FAILED}",
    ]

    write_text(
        OUT_QC,
        "\n".join(qc_lines),
    )

    print("")
    print("=" * 80)
    print(
        "Exact H1--H4 behavior-beta contribution "
        "construction completed"
    )
    print("=" * 80)

    print(
        f"Selected H1--H4 ROIs: "
        f"{selected_count}"
    )

    print(
        f"Uncovered ROIs excluded: "
        f"{uncovered_count}"
    )

    print(
        f"Successful subjects: "
        f"{contribution_wide.shape[0]}"
    )

    print(
        f"Failed subjects: "
        f"{len(failed_rows)}"
    )

    print(
        f"Contribution wide output: "
        f"{OUT_CONTRIBUTION_WIDE}"
    )

    print(
        f"Subject-level exact beta QC: "
        f"{OUT_SUBJECT_BETA}"
    )

    print(
        f"QC report: {OUT_QC}"
    )


if __name__ == "__main__":
    main()
