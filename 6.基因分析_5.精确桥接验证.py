# -*- coding: utf-8 -*-
"""
Gene analysis — exact bridge validation
Validate that the H1--H4 parcel-wise exact behavior-beta contributions
reconstruct the unchanged original participant-level behavior coupling index.

Original behavior index
-----------------------
For participant s:

    Coupling_index_scaled_s
        = slope across H1--H4 of
          early_slope_scaled_sh ~ G_star_sh

Exact parcel contribution identity
----------------------------------
The revised Step-3 contribution table contains only ROIs covered by H1--H4.
For every participant:

    sum_{p in H1 union H2 union H3 union H4}
        exact_behavior_beta_contribution_sp
    = Coupling_index_scaled_s

This script independently verifies the identity against:

1. The original four-system long table.
2. The saved behavior-analysis Coupling_index_scaled.
3. The revised Step-3 subject-level QC output.

Uncovered Schaefer400 ROIs are absent and are not assigned zero.

Outputs
-------
1. Participant-level exact-bridge comparison table.
2. Overall and group-specific agreement statistics.
3. Original-index reproduction QC.
4. Exact parcel-sum closure QC.
5. H1--H4 ROI coverage audit.
6. Optional system-level contribution closure QC.
7. Scatter and Bland--Altman plots.
8. Text QC report.

Important
---------
- This script does not alter or rerun the behavior/ADOS analyses.
- The parcel sum is an exact algebraic reconstruction, not an approximate
  cross-scale correspondence.
"""

from __future__ import annotations

import re
from datetime import datetime
from pathlib import Path
from typing import Dict, List, Optional, Sequence, Tuple

import numpy as np
import pandas as pd
import matplotlib

matplotlib.use("Agg")
import matplotlib.pyplot as plt
from scipy import stats
from scipy.stats import linregress


PROJECT_ROOT = Path(__file__).resolve().parent


# ============================================================
# 1. PATH SETTINGS — ALL ABSOLUTE PATHS
# ============================================================

# Step 3 输出：
# 每个被试在 H1--H4 所覆盖 ROI 上的精确耦合贡献。
EXACT_CONTRIBUTION_WIDE_CSV = Path(
    PROJECT_ROOT
    / "ABIDE2_主流程结果3_耦合指标贡献"
    / "h1-h4-exact-behavior-beta-contribution-wide.csv"
)

# Step 3 输出：
# 选中 ROI 的固定顺序、系统归属和贡献列名。
SELECTED_ROI_ORDER_CSV = Path(
    PROJECT_ROOT
    / "ABIDE2_主流程结果3_耦合指标贡献"
    / "h1-h4-selected-roi-order.csv"
)

# Step 3 输出：
# ROI 级长表，用于可选的系统贡献闭合验证。
EXACT_CONTRIBUTION_LONG_CSV: Optional[Path] = Path(
    PROJECT_ROOT
    / "ABIDE2_主流程结果3_耦合指标贡献"
    / "h1-h4-exact-behavior-beta-contribution-long.csv"
)

# Step 3 输出：
# 被试级精确分解 QC，用于独立复核。
STEP3_SUBJECT_QC_CSV: Optional[Path] = Path(
    PROJECT_ROOT
    / "ABIDE2_主流程结果3_耦合指标贡献"
    / "h1-h4-exact-behavior-beta-subject-qc.csv"
)

# 原始四系统长表。
# 必须是计算原始 Coupling_index_scaled 时使用的同一张表。
# 必需列：
#   sub_id, system, G_star, early_slope_scaled
ORIGINAL_LONG_CSV = Path(
    PROJECT_ROOT / "ABIDE2_结果1" / "result1_input_long_table_strict.csv"
)

# 原始行为分析保存的被试级 Coupling_index_scaled。
# 用于验证从原始四系统长表重算的 beta 是否与行为分析输入一致。
ORIGINAL_BEHAVIOR_SUBJECT_CSV: Optional[Path] = None

# 如另有合并后的行为指标文件，可在此填写绝对路径。
# 当前不使用替代文件。
ORIGINAL_BEHAVIOR_MERGED_CSV: Optional[Path] = None

# Step 5 输出目录。
OUT_DIR = Path(
    PROJECT_ROOT / "ABIDE2_主流程结果5_精确桥接验证"
)
OUT_DIR.mkdir(
    parents=True,
    exist_ok=True,
)


# ============================================================
# 2. ANALYSIS SETTINGS
# ============================================================

SYSTEM_ORDER = [
    "H1_sensory",
    "H2_attention",
    "H3_control",
    "H4_DMN",
]

REQUIRE_ALL_FOUR_SYSTEMS = True

ORIGINAL_REPRODUCTION_TOLERANCE = 1e-10
EXACT_CONTRIBUTION_SUM_TOLERANCE = 1e-10
STEP3_QC_TOLERANCE = 1e-10
SYSTEM_CONTRIBUTION_TOLERANCE = 1e-10

SAVE_PLOTS = True
PLOT_DPI = 300

CONTRIBUTION_COLUMN_PATTERN = re.compile(
    r"^ROI_(\d+)_exact_behavior_beta_contribution$",
    flags=re.IGNORECASE,
)


# ============================================================
# 3. OUTPUT FILES
# ============================================================

OUT_SUBJECT_COMPARISON = (
    OUT_DIR / "h1-h4-exact-bridge-subject-level-comparison.csv"
)

OUT_OVERALL_METRICS = (
    OUT_DIR / "h1-h4-exact-bridge-agreement-metrics-overall.csv"
)

OUT_GROUP_METRICS = (
    OUT_DIR / "h1-h4-exact-bridge-agreement-metrics-by-group.csv"
)

OUT_ORIGINAL_REPRODUCTION = (
    OUT_DIR / "h1-h4-original-index-reproduction-qc.csv"
)

OUT_EXACT_SUM_CLOSURE = (
    OUT_DIR / "h1-h4-exact-contribution-sum-qc.csv"
)

OUT_STEP3_QC_COMPARISON = (
    OUT_DIR / "h1-h4-step3-subject-qc-comparison.csv"
)

OUT_SYSTEM_CLOSURE = (
    OUT_DIR / "h1-h4-system-contribution-closure-qc.csv"
)

OUT_SUBJECT_MATCH_AUDIT = (
    OUT_DIR / "h1-h4-exact-bridge-subject-match-audit.csv"
)

OUT_ROI_COVERAGE_AUDIT = (
    OUT_DIR / "h1-h4-exact-bridge-roi-coverage-audit.csv"
)

OUT_SCATTER = (
    OUT_DIR / "h1-h4-exact-contribution-sum-vs-original-scatter.png"
)

OUT_BLAND_ALTMAN = (
    OUT_DIR / "h1-h4-exact-contribution-sum-vs-original-bland-altman.png"
)

OUT_REPORT = (
    OUT_DIR / "h1-h4-exact-behavior-bridge-qc-report.txt"
)


# ============================================================
# 4. GENERAL UTILITIES
# ============================================================

def normalize_sub_id(value) -> Optional[str]:
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


def safe_group_label(value) -> str:
    if pd.isna(value):
        return ""

    text = str(value).strip().lower()

    if re.fullmatch(r"-?\d+\.0", text):
        text = str(int(float(text)))

    if text in {
        "1",
        "asd",
        "autism",
        "autistic",
        "patient",
    }:
        return "ASD"

    if text in {
        "0",
        "2",
        "hc",
        "control",
        "healthy control",
        "healthy_control",
        "td",
        "typical",
        "typically developing",
    }:
        return "HC"

    return str(value).strip()


def first_existing_id_series(table: pd.DataFrame) -> pd.Series:
    for column in [
        "sub_id",
        "SUB_ID",
        "FILE_ID",
        "SUB_ID_norm",
    ]:
        if column in table.columns:
            return table[column]

    raise ValueError(
        "No subject-ID column found. Expected one of: "
        "sub_id, SUB_ID, FILE_ID, SUB_ID_norm."
    )


def safe_pearson(
    x: Sequence[float],
    y: Sequence[float],
) -> Tuple[float, float]:
    x_array = np.asarray(x, dtype=float)
    y_array = np.asarray(y, dtype=float)

    valid = (
        np.isfinite(x_array)
        & np.isfinite(y_array)
    )

    x_array = x_array[valid]
    y_array = y_array[valid]

    if (
        len(x_array) < 3
        or np.std(x_array) == 0
        or np.std(y_array) == 0
    ):
        return np.nan, np.nan

    result = stats.pearsonr(
        x_array,
        y_array,
    )

    return (
        float(result.statistic),
        float(result.pvalue),
    )


def safe_spearman(
    x: Sequence[float],
    y: Sequence[float],
) -> Tuple[float, float]:
    x_array = np.asarray(x, dtype=float)
    y_array = np.asarray(y, dtype=float)

    valid = (
        np.isfinite(x_array)
        & np.isfinite(y_array)
    )

    x_array = x_array[valid]
    y_array = y_array[valid]

    if len(x_array) < 3:
        return np.nan, np.nan

    result = stats.spearmanr(
        x_array,
        y_array,
    )

    return (
        float(result.statistic),
        float(result.pvalue),
    )


# ============================================================
# 5. AGREEMENT STATISTICS
# ============================================================

def concordance_correlation_coefficient(
    reference: Sequence[float],
    candidate: Sequence[float],
) -> float:
    x = np.asarray(reference, dtype=float)
    y = np.asarray(candidate, dtype=float)

    valid = np.isfinite(x) & np.isfinite(y)
    x = x[valid]
    y = y[valid]

    if len(x) < 2:
        return np.nan

    variance_x = np.var(
        x,
        ddof=1,
    )

    variance_y = np.var(
        y,
        ddof=1,
    )

    covariance = np.cov(
        x,
        y,
        ddof=1,
    )[0, 1]

    denominator = (
        variance_x
        + variance_y
        + (
            np.mean(x)
            - np.mean(y)
        ) ** 2
    )

    if denominator == 0:
        return np.nan

    return float(
        2.0 * covariance
        / denominator
    )


def icc_absolute_agreement_single(
    reference: Sequence[float],
    candidate: Sequence[float],
) -> float:
    x = np.asarray(reference, dtype=float)
    y = np.asarray(candidate, dtype=float)

    valid = np.isfinite(x) & np.isfinite(y)

    data = np.column_stack(
        [
            x[valid],
            y[valid],
        ]
    )

    n_subject, n_method = data.shape

    if n_subject < 2 or n_method < 2:
        return np.nan

    grand_mean = np.mean(data)

    subject_means = np.mean(
        data,
        axis=1,
    )

    method_means = np.mean(
        data,
        axis=0,
    )

    ss_subject = (
        n_method
        * np.sum(
            (
                subject_means
                - grand_mean
            ) ** 2
        )
    )

    ss_method = (
        n_subject
        * np.sum(
            (
                method_means
                - grand_mean
            ) ** 2
        )
    )

    residual = (
        data
        - subject_means[:, None]
        - method_means[None, :]
        + grand_mean
    )

    ss_error = np.sum(
        residual ** 2
    )

    ms_subject = (
        ss_subject
        / (n_subject - 1)
    )

    ms_method = (
        ss_method
        / (n_method - 1)
    )

    ms_error = (
        ss_error
        / (
            (n_subject - 1)
            * (n_method - 1)
        )
    )

    denominator = (
        ms_subject
        + (
            n_method - 1
        ) * ms_error
        + (
            n_method
            * (
                ms_method
                - ms_error
            )
            / n_subject
        )
    )

    if denominator == 0:
        return np.nan

    return float(
        (
            ms_subject
            - ms_error
        )
        / denominator
    )


def calculate_agreement_metrics(
    table: pd.DataFrame,
    reference_column: str,
    candidate_column: str,
    comparison_name: str,
    group_label: str = "ALL",
) -> dict:
    reference = pd.to_numeric(
        table[reference_column],
        errors="coerce",
    ).to_numpy(dtype=float)

    candidate = pd.to_numeric(
        table[candidate_column],
        errors="coerce",
    ).to_numpy(dtype=float)

    valid = (
        np.isfinite(reference)
        & np.isfinite(candidate)
    )

    reference = reference[valid]
    candidate = candidate[valid]

    if len(reference) == 0:
        return {
            "comparison": comparison_name,
            "group": group_label,
            "N": 0,
        }

    difference = (
        candidate - reference
    )

    pearson_r, pearson_p = safe_pearson(
        reference,
        candidate,
    )

    spearman_rho, spearman_p = safe_spearman(
        reference,
        candidate,
    )

    if (
        len(reference) >= 2
        and np.std(reference) > 0
    ):
        regression = linregress(
            reference,
            candidate,
        )

        regression_slope = float(
            regression.slope
        )

        regression_intercept = float(
            regression.intercept
        )

        regression_r_squared = float(
            regression.rvalue ** 2
        )
    else:
        regression_slope = np.nan
        regression_intercept = np.nan
        regression_r_squared = np.nan

    mean_difference = float(
        np.mean(difference)
    )

    sd_difference = (
        float(
            np.std(
                difference,
                ddof=1,
            )
        )
        if len(difference) > 1
        else np.nan
    )

    if np.isfinite(sd_difference):
        loa_lower = float(
            mean_difference
            - 1.96 * sd_difference
        )

        loa_upper = float(
            mean_difference
            + 1.96 * sd_difference
        )
    else:
        loa_lower = np.nan
        loa_upper = np.nan

    return {
        "comparison": comparison_name,
        "group": group_label,
        "reference_variable": reference_column,
        "candidate_variable": candidate_column,
        "difference_definition": "candidate_minus_reference",
        "N": int(len(reference)),
        "reference_mean": float(
            np.mean(reference)
        ),
        "reference_sd": (
            float(
                np.std(
                    reference,
                    ddof=1,
                )
            )
            if len(reference) > 1
            else np.nan
        ),
        "candidate_mean": float(
            np.mean(candidate)
        ),
        "candidate_sd": (
            float(
                np.std(
                    candidate,
                    ddof=1,
                )
            )
            if len(candidate) > 1
            else np.nan
        ),
        "pearson_r": pearson_r,
        "pearson_p": pearson_p,
        "spearman_rho": spearman_rho,
        "spearman_p": spearman_p,
        "Lin_CCC":
            concordance_correlation_coefficient(
                reference,
                candidate,
            ),
        "ICC_absolute_agreement_single":
            icc_absolute_agreement_single(
                reference,
                candidate,
            ),
        "regression_candidate_on_reference_slope":
            regression_slope,
        "regression_candidate_on_reference_intercept":
            regression_intercept,
        "regression_R2":
            regression_r_squared,
        "mean_difference":
            mean_difference,
        "sd_difference":
            sd_difference,
        "median_difference":
            float(
                np.median(
                    difference
                )
            ),
        "MAE":
            float(
                np.mean(
                    np.abs(
                        difference
                    )
                )
            ),
        "RMSE":
            float(
                np.sqrt(
                    np.mean(
                        difference ** 2
                    )
                )
            ),
        "median_absolute_difference":
            float(
                np.median(
                    np.abs(
                        difference
                    )
                )
            ),
        "max_absolute_difference":
            float(
                np.max(
                    np.abs(
                        difference
                    )
                )
            ),
        "Bland_Altman_lower_95_limit":
            loa_lower,
        "Bland_Altman_upper_95_limit":
            loa_upper,
    }


# ============================================================
# 6. LOAD SELECTED ROI ORDER AND EXACT CONTRIBUTIONS
# ============================================================

def load_selected_roi_order(
    path: Path,
) -> pd.DataFrame:
    if not path.exists():
        raise FileNotFoundError(
            f"Selected ROI order file not found: {path}"
        )

    table = pd.read_csv(
        path,
        low_memory=False,
        encoding="utf-8-sig",
    )

    table = clean_columns(
        table
    )

    required = [
        "selected_output_position_0based",
        "roi_index_0based",
        "ROI_ID_1based",
        "system",
        "contribution_column",
    ]

    missing = [
        column
        for column in required
        if column not in table.columns
    ]

    if missing:
        raise ValueError(
            f"Selected ROI order is missing columns: {missing}"
        )

    table = table.copy()

    for column in [
        "selected_output_position_0based",
        "roi_index_0based",
        "ROI_ID_1based",
    ]:
        table[column] = pd.to_numeric(
            table[column],
            errors="raise",
        ).astype(int)

    table["system"] = (
        table["system"]
        .astype(str)
        .str.strip()
    )

    table["contribution_column"] = (
        table["contribution_column"]
        .astype(str)
        .str.strip()
    )

    table = table.sort_values(
        "selected_output_position_0based"
    ).reset_index(drop=True)

    if (
        table[
            "selected_output_position_0based"
        ].tolist()
        != list(range(len(table)))
    ):
        raise ValueError(
            "Selected output positions are not a complete "
            "0-based consecutive sequence."
        )

    if table[
        "roi_index_0based"
    ].duplicated().any():
        raise ValueError(
            "Selected ROI order contains duplicated ROI indices."
        )

    if table[
        "contribution_column"
    ].duplicated().any():
        raise ValueError(
            "Selected ROI order contains duplicated contribution columns."
        )

    if not np.array_equal(
        (
            table[
                "roi_index_0based"
            ]
            + 1
        ).to_numpy(),
        table[
            "ROI_ID_1based"
        ].to_numpy(),
    ):
        raise ValueError(
            "ROI_ID_1based is inconsistent with roi_index_0based + 1."
        )

    allowed_systems = set(
        SYSTEM_ORDER
    )

    unexpected = sorted(
        set(
            table["system"]
        )
        - allowed_systems
    )

    if unexpected:
        raise ValueError(
            f"Unexpected system labels: {unexpected}"
        )

    return table


def load_exact_contribution_table(
    path: Path,
    selected_roi_order: pd.DataFrame,
) -> Tuple[
    pd.DataFrame,
    List[str],
]:
    if not path.exists():
        raise FileNotFoundError(
            f"Exact contribution table not found: {path}"
        )

    table = pd.read_csv(
        path,
        low_memory=False,
        encoding="utf-8-sig",
    )

    table = clean_columns(
        table
    )

    if "sub_id" not in table.columns:
        raise ValueError(
            "Exact contribution table must contain sub_id."
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
        raise ValueError(
            "Duplicate subjects found in exact contribution table."
        )

    expected_columns = (
        selected_roi_order[
            "contribution_column"
        ].tolist()
    )

    missing_columns = [
        column
        for column in expected_columns
        if column not in table.columns
    ]

    if missing_columns:
        raise ValueError(
            "Exact contribution table is missing selected ROI columns. "
            f"Examples: {missing_columns[:10]}"
        )

    discovered = []

    for column in table.columns:
        match = CONTRIBUTION_COLUMN_PATTERN.fullmatch(
            str(column)
        )

        if match:
            discovered.append(
                str(column)
            )

    unexpected_columns = [
        column
        for column in discovered
        if column not in set(
            expected_columns
        )
    ]

    if unexpected_columns:
        raise ValueError(
            "Exact contribution table contains contribution columns "
            "not listed in selected ROI order. "
            f"Examples: {unexpected_columns[:10]}"
        )

    for column in expected_columns:
        table[column] = pd.to_numeric(
            table[column],
            errors="coerce",
        )

    matrix = table[
        expected_columns
    ].to_numpy(
        dtype=float
    )

    if not np.isfinite(matrix).all():
        raise ValueError(
            "Exact contribution matrix contains NaN/Inf."
        )

    return table, expected_columns


def save_roi_coverage_audit(
    selected_roi_order: pd.DataFrame,
) -> None:
    rows = []

    for system in SYSTEM_ORDER:
        system_table = selected_roi_order[
            selected_roi_order[
                "system"
            ] == system
        ]

        rows.append(
            {
                "record_type": "system_summary",
                "system": system,
                "n_selected_ROI":
                    int(
                        len(
                            system_table
                        )
                    ),
                "min_roi_index_0based":
                    (
                        int(
                            system_table[
                                "roi_index_0based"
                            ].min()
                        )
                        if len(system_table) > 0
                        else np.nan
                    ),
                "max_roi_index_0based":
                    (
                        int(
                            system_table[
                                "roi_index_0based"
                            ].max()
                        )
                        if len(system_table) > 0
                        else np.nan
                    ),
            }
        )

    rows.append(
        {
            "record_type": "overall_summary",
            "system": "H1_H4_UNION",
            "n_selected_ROI":
                int(
                    len(
                        selected_roi_order
                    )
                ),
            "min_roi_index_0based":
                int(
                    selected_roi_order[
                        "roi_index_0based"
                    ].min()
                ),
            "max_roi_index_0based":
                int(
                    selected_roi_order[
                        "roi_index_0based"
                    ].max()
                ),
        }
    )

    rows.append(
        {
            "record_type": "overall_summary",
            "system": "UNCOVERED_SCHAEFER400",
            "n_selected_ROI":
                int(
                    400
                    - len(
                        selected_roi_order
                    )
                ),
            "min_roi_index_0based":
                np.nan,
            "max_roi_index_0based":
                np.nan,
        }
    )

    pd.DataFrame(
        rows
    ).to_csv(
        OUT_ROI_COVERAGE_AUDIT,
        index=False,
        encoding="utf-8-sig",
    )


# ============================================================
# 7. LOAD ORIGINAL FOUR-SYSTEM DATA
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

    table = clean_columns(
        table
    )

    required = [
        "sub_id",
        "system",
        "G_star",
        "early_slope_scaled",
    ]

    missing = [
        column
        for column in required
        if column not in table.columns
    ]

    if missing:
        raise ValueError(
            f"Original long table is missing: {missing}"
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

    table[
        "early_slope_scaled"
    ] = pd.to_numeric(
        table[
            "early_slope_scaled"
        ],
        errors="coerce",
    )

    table = table.dropna(
        subset=[
            "sub_id",
            "system",
            "G_star",
            "early_slope_scaled",
        ]
    ).copy()

    if "Group" in table.columns:
        table["Group"] = (
            table["Group"]
            .apply(
                safe_group_label
            )
        )

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
        raise ValueError(
            "Duplicate subject-system rows found in original long table."
        )

    counts = table.groupby(
        "sub_id"
    )["system"].nunique()

    incomplete = counts[
        counts != len(
            SYSTEM_ORDER
        )
    ]

    if (
        REQUIRE_ALL_FOUR_SYSTEMS
        and len(incomplete) > 0
    ):
        raise ValueError(
            f"{len(incomplete)} subjects do not have all four systems."
        )

    return table


def choose_behavior_subject_file() -> Optional[Path]:
    for path in [
        ORIGINAL_BEHAVIOR_SUBJECT_CSV,
        ORIGINAL_BEHAVIOR_MERGED_CSV,
    ]:
        if (
            path is not None
            and path.exists()
        ):
            return path

    return None


def load_original_behavior_index() -> Optional[pd.DataFrame]:
    path = choose_behavior_subject_file()

    if path is None:
        print(
            "[WARN] No saved behavior table found. "
            "Only the long-table recomputed index will be used."
        )
        return None

    table = pd.read_csv(
        path,
        low_memory=False,
        encoding="utf-8-sig",
    )

    table = clean_columns(
        table
    )

    if "Coupling_index_scaled" not in table.columns:
        raise ValueError(
            f"{path} does not contain Coupling_index_scaled."
        )

    id_series = first_existing_id_series(
        table
    )

    output = pd.DataFrame(
        {
            "sub_id":
                id_series.apply(
                    normalize_sub_id
                ),
            "Original_saved_Coupling_index_scaled":
                pd.to_numeric(
                    table[
                        "Coupling_index_scaled"
                    ],
                    errors="coerce",
                ),
        }
    )

    for column in [
        "Group",
        "Age",
        "Sex",
        "Sex_bin",
        "Site",
        "FIQ",
        "MeanFD",
    ]:
        if column in table.columns:
            output[column] = table[
                column
            ].values

    output = (
        output.dropna(
            subset=["sub_id"]
        )
        .drop_duplicates(
            subset=["sub_id"],
            keep="first",
        )
    )

    if "Group" in output.columns:
        output["Group"] = (
            output["Group"]
            .apply(
                safe_group_label
            )
        )

    return output


# ============================================================
# 8. RECOMPUTE ORIGINAL BEHAVIOR INDEX
# ============================================================

def estimate_original_behavior_index(
    group: pd.DataFrame,
) -> pd.Series:
    temporary = (
        group[
            [
                "system",
                "G_star",
                "early_slope_scaled",
            ]
        ]
        .set_index("system")
        .reindex(
            SYSTEM_ORDER
        )
        .reset_index()
    )

    temporary = temporary.dropna(
        subset=[
            "G_star",
            "early_slope_scaled",
        ]
    )

    output = {
        "Original_recomputed_Coupling_index_scaled":
            np.nan,
        "Original_recomputed_Coupling_intercept_scaled":
            np.nan,
        "Original_recomputed_Coupling_r_scaled":
            np.nan,
        "Original_recomputed_Coupling_p_scaled":
            np.nan,
        "Original_recomputed_Coupling_SE_scaled":
            np.nan,
        "Original_recomputed_n_systems":
            int(
                temporary.shape[0]
            ),
    }

    if temporary.shape[0] != len(
        SYSTEM_ORDER
    ):
        return pd.Series(
            output
        )

    if (
        temporary[
            "G_star"
        ].std(
            ddof=0
        ) == 0
        or temporary[
            "early_slope_scaled"
        ].std(
            ddof=0
        ) == 0
    ):
        return pd.Series(
            output
        )

    result = linregress(
        temporary[
            "G_star"
        ],
        temporary[
            "early_slope_scaled"
        ],
    )

    output[
        "Original_recomputed_Coupling_index_scaled"
    ] = float(
        result.slope
    )

    output[
        "Original_recomputed_Coupling_intercept_scaled"
    ] = float(
        result.intercept
    )

    output[
        "Original_recomputed_Coupling_r_scaled"
    ] = float(
        result.rvalue
    )

    output[
        "Original_recomputed_Coupling_p_scaled"
    ] = float(
        result.pvalue
    )

    output[
        "Original_recomputed_Coupling_SE_scaled"
    ] = float(
        result.stderr
    )

    return pd.Series(
        output
    )


def build_original_subject_table(
    original_long: pd.DataFrame,
) -> pd.DataFrame:
    recomputed = (
        original_long.groupby(
            "sub_id",
            sort=False,
        )
        .apply(
            estimate_original_behavior_index,
        )
        .reset_index()
    )

    metadata_candidates = [
        "sub_id",
        "Group",
        "Age",
        "Sex",
        "Sex_bin",
        "Site",
        "FIQ",
        "MeanFD",
    ]

    metadata_columns = [
        column
        for column in metadata_candidates
        if column in original_long.columns
    ]

    metadata = (
        original_long[
            metadata_columns
        ]
        .drop_duplicates(
            subset=["sub_id"],
            keep="first",
        )
    )

    return metadata.merge(
        recomputed,
        on="sub_id",
        how="right",
        validate="one_to_one",
    )


# ============================================================
# 9. STEP-3 QC AND SYSTEM CLOSURE
# ============================================================

def load_step3_subject_qc() -> Optional[pd.DataFrame]:
    path = STEP3_SUBJECT_QC_CSV

    if (
        path is None
        or not path.exists()
    ):
        return None

    table = pd.read_csv(
        path,
        low_memory=False,
        encoding="utf-8-sig",
    )

    table = clean_columns(
        table
    )

    required = [
        "sub_id",
        "original_behavior_beta_closed_form",
        "sum_H1H4_ROI_exact_contributions",
    ]

    missing = [
        column
        for column in required
        if column not in table.columns
    ]

    if missing:
        raise ValueError(
            f"Step-3 subject QC is missing: {missing}"
        )

    table["sub_id"] = (
        table["sub_id"]
        .apply(
            normalize_sub_id
        )
    )

    keep = [
        "sub_id",
        "original_behavior_beta_closed_form",
        "sum_H1H4_ROI_exact_contributions",
    ]

    for column in [
        "saved_Coupling_index_scaled",
        "contribution_sum_minus_original_beta",
        "recomputed_minus_saved_behavior_beta",
        "max_system_mean_calibration_error",
        "max_system_contribution_closure_error",
    ]:
        if column in table.columns:
            keep.append(
                column
            )

    return (
        table[
            keep
        ]
        .dropna(
            subset=["sub_id"]
        )
        .drop_duplicates(
            subset=["sub_id"],
            keep="first",
        )
    )


def build_system_contribution_closure_qc() -> Optional[pd.DataFrame]:
    path = EXACT_CONTRIBUTION_LONG_CSV

    if (
        path is None
        or not path.exists()
    ):
        return None

    table = pd.read_csv(
        path,
        low_memory=False,
        encoding="utf-8-sig",
    )

    table = clean_columns(
        table
    )

    required = [
        "sub_id",
        "system",
        "G_star_centered",
        "G_star_centered_sum_squares",
        "original_system_early_slope_scaled",
        "mean_original_system_early_slope_scaled",
        "exact_behavior_beta_contribution",
    ]

    missing = [
        column
        for column in required
        if column not in table.columns
    ]

    if missing:
        raise ValueError(
            f"Exact contribution long table is missing: {missing}"
        )

    table["sub_id"] = (
        table["sub_id"]
        .apply(
            normalize_sub_id
        )
    )

    for column in required[2:]:
        table[column] = pd.to_numeric(
            table[column],
            errors="coerce",
        )

    grouped = (
        table.groupby(
            [
                "sub_id",
                "system",
            ],
            sort=False,
        )
        .agg(
            n_selected_ROI=(
                "exact_behavior_beta_contribution",
                "size",
            ),
            sum_exact_ROI_contributions=(
                "exact_behavior_beta_contribution",
                "sum",
            ),
            G_star_centered=(
                "G_star_centered",
                "first",
            ),
            G_star_centered_sum_squares=(
                "G_star_centered_sum_squares",
                "first",
            ),
            original_system_early_slope_scaled=(
                "original_system_early_slope_scaled",
                "first",
            ),
            mean_original_system_early_slope_scaled=(
                "mean_original_system_early_slope_scaled",
                "first",
            ),
        )
        .reset_index()
    )

    grouped[
        "expected_system_contribution"
    ] = (
        grouped[
            "G_star_centered"
        ]
        / grouped[
            "G_star_centered_sum_squares"
        ]
        * (
            grouped[
                "original_system_early_slope_scaled"
            ]
            - grouped[
                "mean_original_system_early_slope_scaled"
            ]
        )
    )

    grouped[
        "system_contribution_error"
    ] = (
        grouped[
            "sum_exact_ROI_contributions"
        ]
        - grouped[
            "expected_system_contribution"
        ]
    )

    grouped[
        "absolute_system_contribution_error"
    ] = np.abs(
        grouped[
            "system_contribution_error"
        ]
    )

    grouped[
        "closure_pass"
    ] = (
        grouped[
            "absolute_system_contribution_error"
        ]
        <= SYSTEM_CONTRIBUTION_TOLERANCE
    )

    grouped.to_csv(
        OUT_SYSTEM_CLOSURE,
        index=False,
        encoding="utf-8-sig",
    )

    return grouped


# ============================================================
# 10. PLOTS
# ============================================================

def save_scatter_plot(
    table: pd.DataFrame,
    reference_column: str,
    candidate_column: str,
    output_path: Path,
) -> None:
    plot_data = table[
        [
            reference_column,
            candidate_column,
        ]
    ].apply(
        pd.to_numeric,
        errors="coerce",
    ).dropna()

    if plot_data.empty:
        return

    x = plot_data[
        reference_column
    ].to_numpy(
        dtype=float
    )

    y = plot_data[
        candidate_column
    ].to_numpy(
        dtype=float
    )

    figure, axis = plt.subplots(
        figsize=(6.5, 5.5)
    )

    axis.scatter(
        x,
        y,
        s=16,
        alpha=0.65,
    )

    lower = float(
        min(
            np.min(x),
            np.min(y),
        )
    )

    upper = float(
        max(
            np.max(x),
            np.max(y),
        )
    )

    axis.plot(
        [
            lower,
            upper,
        ],
        [
            lower,
            upper,
        ],
        linestyle="--",
        linewidth=1.0,
    )

    axis.set_xlabel(
        "Original four-system behavior coupling index"
    )

    axis.set_ylabel(
        "Sum of exact H1--H4 ROI contributions"
    )

    axis.set_title(
        "Exact H1--H4 reconstruction of behavior coupling"
    )

    figure.tight_layout()

    figure.savefig(
        output_path,
        dpi=PLOT_DPI,
        bbox_inches="tight",
    )

    plt.close(
        figure
    )


def save_bland_altman_plot(
    table: pd.DataFrame,
    reference_column: str,
    candidate_column: str,
    output_path: Path,
) -> None:
    plot_data = table[
        [
            reference_column,
            candidate_column,
        ]
    ].apply(
        pd.to_numeric,
        errors="coerce",
    ).dropna()

    if plot_data.empty:
        return

    reference = plot_data[
        reference_column
    ].to_numpy(
        dtype=float
    )

    candidate = plot_data[
        candidate_column
    ].to_numpy(
        dtype=float
    )

    mean_value = (
        reference
        + candidate
    ) / 2.0

    difference = (
        candidate
        - reference
    )

    mean_difference = float(
        np.mean(
            difference
        )
    )

    sd_difference = float(
        np.std(
            difference,
            ddof=1,
        )
    )

    lower_limit = (
        mean_difference
        - 1.96 * sd_difference
    )

    upper_limit = (
        mean_difference
        + 1.96 * sd_difference
    )

    figure, axis = plt.subplots(
        figsize=(6.5, 5.5)
    )

    axis.scatter(
        mean_value,
        difference,
        s=16,
        alpha=0.65,
    )

    axis.axhline(
        mean_difference,
        linestyle="-",
        linewidth=1.1,
    )

    axis.axhline(
        lower_limit,
        linestyle="--",
        linewidth=1.0,
    )

    axis.axhline(
        upper_limit,
        linestyle="--",
        linewidth=1.0,
    )

    axis.set_xlabel(
        "Mean of original index and exact parcel sum"
    )

    axis.set_ylabel(
        "Exact parcel sum minus original index"
    )

    axis.set_title(
        "Bland–Altman exact-closure check"
    )

    figure.tight_layout()

    figure.savefig(
        output_path,
        dpi=PLOT_DPI,
        bbox_inches="tight",
    )

    plt.close(
        figure
    )


# ============================================================
# 11. MAIN
# ============================================================

def main() -> None:
    print("=" * 80)
    print(
        "Validating exact H1--H4 ROI reconstruction "
        "of the original behavior coupling index"
    )
    print("=" * 80)

    required_input_paths = {
        "EXACT_CONTRIBUTION_WIDE_CSV":
            EXACT_CONTRIBUTION_WIDE_CSV,
        "SELECTED_ROI_ORDER_CSV":
            SELECTED_ROI_ORDER_CSV,
        "ORIGINAL_LONG_CSV":
            ORIGINAL_LONG_CSV,
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

    optional_input_paths = {
        "EXACT_CONTRIBUTION_LONG_CSV":
            EXACT_CONTRIBUTION_LONG_CSV,
        "STEP3_SUBJECT_QC_CSV":
            STEP3_SUBJECT_QC_CSV,
        "ORIGINAL_BEHAVIOR_SUBJECT_CSV":
            ORIGINAL_BEHAVIOR_SUBJECT_CSV,
        "ORIGINAL_BEHAVIOR_MERGED_CSV":
            ORIGINAL_BEHAVIOR_MERGED_CSV,
    }

    for name, path in optional_input_paths.items():
        if path is not None and not Path(path).exists():
            print(
                f"Warning: optional input not found; "
                f"the related validation will be skipped:"
            )
            print(
                f"  {name}: {path}"
            )

    print(
        f"Exact contribution wide input: "
        f"{EXACT_CONTRIBUTION_WIDE_CSV}"
    )
    print(
        f"Selected ROI order: "
        f"{SELECTED_ROI_ORDER_CSV}"
    )
    print(
        f"Exact contribution long input: "
        f"{EXACT_CONTRIBUTION_LONG_CSV}"
    )
    print(
        f"Step-3 subject QC: "
        f"{STEP3_SUBJECT_QC_CSV}"
    )
    print(
        f"Original four-system long table: "
        f"{ORIGINAL_LONG_CSV}"
    )
    print(
        f"Original saved behavior table: "
        f"{ORIGINAL_BEHAVIOR_SUBJECT_CSV}"
    )
    print(
        f"Output directory: "
        f"{OUT_DIR}"
    )

    selected_roi_order = load_selected_roi_order(
        SELECTED_ROI_ORDER_CSV
    )

    save_roi_coverage_audit(
        selected_roi_order
    )

    (
        contribution_table,
        contribution_columns,
    ) = load_exact_contribution_table(
        EXACT_CONTRIBUTION_WIDE_CSV,
        selected_roi_order,
    )

    original_long = load_original_long(
        ORIGINAL_LONG_CSV
    )

    original_subject = build_original_subject_table(
        original_long
    )

    saved_behavior = load_original_behavior_index()

    contribution_subject = pd.DataFrame(
        {
            "sub_id":
                contribution_table[
                    "sub_id"
                ].values,
            "Exact_H1H4_contribution_sum":
                contribution_table[
                    contribution_columns
                ].sum(axis=1).to_numpy(
                    dtype=float
                ),
        }
    )

    # Preserve available metadata from the exact contribution table.
    for column in [
        "Group",
        "Age",
        "Sex",
        "Sex_bin",
        "Site",
        "FIQ",
        "MeanFD",
    ]:
        if column in contribution_table.columns:
            contribution_subject[
                column
            ] = contribution_table[
                column
            ].values

    subject_table = original_subject.merge(
        contribution_subject,
        on="sub_id",
        how="inner",
        suffixes=(
            "_original",
            "_contribution",
        ),
        validate="one_to_one",
    )

    # Resolve duplicated metadata columns by preferring the exact
    # contribution table, then the original long table.
    for column in [
        "Group",
        "Age",
        "Sex",
        "Sex_bin",
        "Site",
        "FIQ",
        "MeanFD",
    ]:
        original_column = (
            f"{column}_original"
        )

        contribution_column = (
            f"{column}_contribution"
        )

        if (
            contribution_column
            in subject_table.columns
        ):
            subject_table[
                column
            ] = subject_table[
                contribution_column
            ]

        elif (
            original_column
            in subject_table.columns
        ):
            subject_table[
                column
            ] = subject_table[
                original_column
            ]

    if saved_behavior is not None:
        saved_columns = [
            "sub_id",
            "Original_saved_Coupling_index_scaled",
        ]

        subject_table = subject_table.merge(
            saved_behavior[
                saved_columns
            ],
            on="sub_id",
            how="left",
            validate="one_to_one",
        )
    else:
        subject_table[
            "Original_saved_Coupling_index_scaled"
        ] = np.nan

    if "Group" in subject_table.columns:
        subject_table["Group"] = (
            subject_table["Group"]
            .apply(
                safe_group_label
            )
        )

    step3_qc = load_step3_subject_qc()

    if step3_qc is not None:
        step3_renamed = step3_qc.rename(
            columns={
                "original_behavior_beta_closed_form":
                    "Step3_original_behavior_beta_closed_form",
                "sum_H1H4_ROI_exact_contributions":
                    "Step3_exact_contribution_sum",
                "saved_Coupling_index_scaled":
                    "Step3_saved_Coupling_index_scaled",
                "contribution_sum_minus_original_beta":
                    "Step3_contribution_sum_minus_original_beta",
                "recomputed_minus_saved_behavior_beta":
                    "Step3_recomputed_minus_saved_behavior_beta",
                "max_system_mean_calibration_error":
                    "Step3_max_system_mean_calibration_error",
                "max_system_contribution_closure_error":
                    "Step3_max_system_contribution_closure_error",
            }
        )

        subject_table = subject_table.merge(
            step3_renamed,
            on="sub_id",
            how="left",
            validate="one_to_one",
        )

    # --------------------------------------------------------
    # Subject-level differences and exact closure
    # --------------------------------------------------------
    subject_table[
        "Exact_sum_minus_original_recomputed"
    ] = (
        subject_table[
            "Exact_H1H4_contribution_sum"
        ]
        - subject_table[
            "Original_recomputed_Coupling_index_scaled"
        ]
    )

    subject_table[
        "absolute_Exact_sum_minus_original_recomputed"
    ] = np.abs(
        subject_table[
            "Exact_sum_minus_original_recomputed"
        ]
    )

    subject_table[
        "Exact_sum_vs_recomputed_pass"
    ] = (
        subject_table[
            "absolute_Exact_sum_minus_original_recomputed"
        ]
        <= EXACT_CONTRIBUTION_SUM_TOLERANCE
    )

    subject_table[
        "Original_recomputed_minus_saved"
    ] = (
        subject_table[
            "Original_recomputed_Coupling_index_scaled"
        ]
        - subject_table[
            "Original_saved_Coupling_index_scaled"
        ]
    )

    subject_table[
        "absolute_Original_recomputed_minus_saved"
    ] = np.abs(
        subject_table[
            "Original_recomputed_minus_saved"
        ]
    )

    subject_table[
        "Original_reproduction_pass"
    ] = (
        subject_table[
            "absolute_Original_recomputed_minus_saved"
        ]
        <= ORIGINAL_REPRODUCTION_TOLERANCE
    )

    subject_table[
        "Exact_sum_minus_saved"
    ] = (
        subject_table[
            "Exact_H1H4_contribution_sum"
        ]
        - subject_table[
            "Original_saved_Coupling_index_scaled"
        ]
    )

    subject_table[
        "absolute_Exact_sum_minus_saved"
    ] = np.abs(
        subject_table[
            "Exact_sum_minus_saved"
        ]
    )

    subject_table[
        "Exact_sum_vs_saved_pass"
    ] = (
        subject_table[
            "absolute_Exact_sum_minus_saved"
        ]
        <= EXACT_CONTRIBUTION_SUM_TOLERANCE
    )

    if (
        "Step3_exact_contribution_sum"
        in subject_table.columns
    ):
        subject_table[
            "Current_exact_sum_minus_Step3_exact_sum"
        ] = (
            subject_table[
                "Exact_H1H4_contribution_sum"
            ]
            - subject_table[
                "Step3_exact_contribution_sum"
            ]
        )

        subject_table[
            "absolute_Current_exact_sum_minus_Step3_exact_sum"
        ] = np.abs(
            subject_table[
                "Current_exact_sum_minus_Step3_exact_sum"
            ]
        )

        subject_table[
            "Current_vs_Step3_exact_sum_pass"
        ] = (
            subject_table[
                "absolute_Current_exact_sum_minus_Step3_exact_sum"
            ]
            <= STEP3_QC_TOLERANCE
        )

    subject_table.to_csv(
        OUT_SUBJECT_COMPARISON,
        index=False,
        encoding="utf-8-sig",
    )

    original_reproduction = subject_table[
        [
            "sub_id",
            "Original_recomputed_Coupling_index_scaled",
            "Original_saved_Coupling_index_scaled",
            "Original_recomputed_minus_saved",
            "absolute_Original_recomputed_minus_saved",
            "Original_reproduction_pass",
        ]
    ].copy()

    original_reproduction.to_csv(
        OUT_ORIGINAL_REPRODUCTION,
        index=False,
        encoding="utf-8-sig",
    )

    exact_sum_closure = subject_table[
        [
            "sub_id",
            "Original_recomputed_Coupling_index_scaled",
            "Original_saved_Coupling_index_scaled",
            "Exact_H1H4_contribution_sum",
            "Exact_sum_minus_original_recomputed",
            "absolute_Exact_sum_minus_original_recomputed",
            "Exact_sum_vs_recomputed_pass",
            "Exact_sum_minus_saved",
            "absolute_Exact_sum_minus_saved",
            "Exact_sum_vs_saved_pass",
        ]
    ].copy()

    exact_sum_closure.to_csv(
        OUT_EXACT_SUM_CLOSURE,
        index=False,
        encoding="utf-8-sig",
    )

    if (
        "Step3_exact_contribution_sum"
        in subject_table.columns
    ):
        step3_columns = [
            column
            for column in [
                "sub_id",
                "Exact_H1H4_contribution_sum",
                "Step3_original_behavior_beta_closed_form",
                "Step3_exact_contribution_sum",
                "Current_exact_sum_minus_Step3_exact_sum",
                "absolute_Current_exact_sum_minus_Step3_exact_sum",
                "Current_vs_Step3_exact_sum_pass",
                "Step3_contribution_sum_minus_original_beta",
                "Step3_recomputed_minus_saved_behavior_beta",
                "Step3_max_system_mean_calibration_error",
                "Step3_max_system_contribution_closure_error",
            ]
            if column in subject_table.columns
        ]

        subject_table[
            step3_columns
        ].to_csv(
            OUT_STEP3_QC_COMPARISON,
            index=False,
            encoding="utf-8-sig",
        )

    # --------------------------------------------------------
    # Subject match audit
    # --------------------------------------------------------
    contribution_subjects = set(
        contribution_table[
            "sub_id"
        ]
    )

    original_subjects = set(
        original_long[
            "sub_id"
        ]
    )

    saved_subjects = (
        set(
            saved_behavior[
                "sub_id"
            ]
        )
        if saved_behavior is not None
        else set()
    )

    step3_subjects = (
        set(
            step3_qc[
                "sub_id"
            ]
        )
        if step3_qc is not None
        else set()
    )

    final_subjects = set(
        subject_table[
            "sub_id"
        ]
    )

    audit_rows = []

    for subject in sorted(
        contribution_subjects
        | original_subjects
        | saved_subjects
        | step3_subjects
    ):
        audit_rows.append(
            {
                "sub_id": subject,
                "in_exact_contribution_table":
                    subject in contribution_subjects,
                "in_original_long_table":
                    subject in original_subjects,
                "in_saved_behavior_table":
                    (
                        subject in saved_subjects
                        if saved_behavior is not None
                        else np.nan
                    ),
                "in_step3_subject_QC":
                    (
                        subject in step3_subjects
                        if step3_qc is not None
                        else np.nan
                    ),
                "in_final_comparison":
                    subject in final_subjects,
            }
        )

    pd.DataFrame(
        audit_rows
    ).to_csv(
        OUT_SUBJECT_MATCH_AUDIT,
        index=False,
        encoding="utf-8-sig",
    )

    # --------------------------------------------------------
    # Agreement metrics
    # --------------------------------------------------------
    overall_rows = [
        calculate_agreement_metrics(
            table=subject_table,
            reference_column=(
                "Original_recomputed_Coupling_index_scaled"
            ),
            candidate_column=(
                "Exact_H1H4_contribution_sum"
            ),
            comparison_name=(
                "Exact_H1H4_contribution_sum_vs_original_recomputed"
            ),
        )
    ]

    if subject_table[
        "Original_saved_Coupling_index_scaled"
    ].notna().any():
        overall_rows.append(
            calculate_agreement_metrics(
                table=subject_table,
                reference_column=(
                    "Original_saved_Coupling_index_scaled"
                ),
                candidate_column=(
                    "Original_recomputed_Coupling_index_scaled"
                ),
                comparison_name=(
                    "Original_recomputed_vs_original_saved"
                ),
            )
        )

        overall_rows.append(
            calculate_agreement_metrics(
                table=subject_table,
                reference_column=(
                    "Original_saved_Coupling_index_scaled"
                ),
                candidate_column=(
                    "Exact_H1H4_contribution_sum"
                ),
                comparison_name=(
                    "Exact_H1H4_contribution_sum_vs_original_saved"
                ),
            )
        )

    if (
        "Step3_exact_contribution_sum"
        in subject_table.columns
    ):
        overall_rows.append(
            calculate_agreement_metrics(
                table=subject_table,
                reference_column=(
                    "Step3_exact_contribution_sum"
                ),
                candidate_column=(
                    "Exact_H1H4_contribution_sum"
                ),
                comparison_name=(
                    "Current_exact_sum_vs_Step3_exact_sum"
                ),
            )
        )

    overall_metrics = pd.DataFrame(
        overall_rows
    )

    overall_metrics.to_csv(
        OUT_OVERALL_METRICS,
        index=False,
        encoding="utf-8-sig",
    )

    group_rows = []

    if "Group" in subject_table.columns:
        for group_name, group_table in subject_table.groupby(
            "Group",
            dropna=False,
            sort=False,
        ):
            group_label = (
                str(group_name)
                if str(group_name) != ""
                else "MISSING"
            )

            group_rows.append(
                calculate_agreement_metrics(
                    table=group_table,
                    reference_column=(
                        "Original_recomputed_Coupling_index_scaled"
                    ),
                    candidate_column=(
                        "Exact_H1H4_contribution_sum"
                    ),
                    comparison_name=(
                        "Exact_H1H4_contribution_sum_vs_original_recomputed"
                    ),
                    group_label=group_label,
                )
            )

            if group_table[
                "Original_saved_Coupling_index_scaled"
            ].notna().any():
                group_rows.append(
                    calculate_agreement_metrics(
                        table=group_table,
                        reference_column=(
                            "Original_saved_Coupling_index_scaled"
                        ),
                        candidate_column=(
                            "Exact_H1H4_contribution_sum"
                        ),
                        comparison_name=(
                            "Exact_H1H4_contribution_sum_vs_original_saved"
                        ),
                        group_label=group_label,
                    )
                )

    pd.DataFrame(
        group_rows
    ).to_csv(
        OUT_GROUP_METRICS,
        index=False,
        encoding="utf-8-sig",
    )

    # --------------------------------------------------------
    # Optional system-level contribution closure
    # --------------------------------------------------------
    system_closure = (
        build_system_contribution_closure_qc()
    )

    # --------------------------------------------------------
    # Plots
    # --------------------------------------------------------
    reference_for_plot = (
        "Original_saved_Coupling_index_scaled"
        if subject_table[
            "Original_saved_Coupling_index_scaled"
        ].notna().any()
        else "Original_recomputed_Coupling_index_scaled"
    )

    if SAVE_PLOTS:
        save_scatter_plot(
            table=subject_table,
            reference_column=reference_for_plot,
            candidate_column=(
                "Exact_H1H4_contribution_sum"
            ),
            output_path=OUT_SCATTER,
        )

        save_bland_altman_plot(
            table=subject_table,
            reference_column=reference_for_plot,
            candidate_column=(
                "Exact_H1H4_contribution_sum"
            ),
            output_path=OUT_BLAND_ALTMAN,
        )

    # --------------------------------------------------------
    # QC report
    # --------------------------------------------------------
    valid_original_saved = subject_table[
        [
            "Original_recomputed_Coupling_index_scaled",
            "Original_saved_Coupling_index_scaled",
        ]
    ].notna().all(axis=1)

    if valid_original_saved.any():
        max_original_reproduction_error = float(
            subject_table.loc[
                valid_original_saved,
                "absolute_Original_recomputed_minus_saved",
            ].max()
        )

        n_original_reproduction_fail = int(
            np.sum(
                ~subject_table.loc[
                    valid_original_saved,
                    "Original_reproduction_pass",
                ].astype(bool)
            )
        )
    else:
        max_original_reproduction_error = np.nan
        n_original_reproduction_fail = 0

    max_exact_recomputed_error = float(
        subject_table[
            "absolute_Exact_sum_minus_original_recomputed"
        ].max()
    )

    n_exact_recomputed_fail = int(
        np.sum(
            ~subject_table[
                "Exact_sum_vs_recomputed_pass"
            ].astype(bool)
        )
    )

    valid_exact_saved = subject_table[
        [
            "Exact_H1H4_contribution_sum",
            "Original_saved_Coupling_index_scaled",
        ]
    ].notna().all(axis=1)

    if valid_exact_saved.any():
        max_exact_saved_error = float(
            subject_table.loc[
                valid_exact_saved,
                "absolute_Exact_sum_minus_saved",
            ].max()
        )

        n_exact_saved_fail = int(
            np.sum(
                ~subject_table.loc[
                    valid_exact_saved,
                    "Exact_sum_vs_saved_pass",
                ].astype(bool)
            )
        )
    else:
        max_exact_saved_error = np.nan
        n_exact_saved_fail = 0

    if (
        "absolute_Current_exact_sum_minus_Step3_exact_sum"
        in subject_table.columns
    ):
        valid_step3 = subject_table[
            "absolute_Current_exact_sum_minus_Step3_exact_sum"
        ].notna()

        max_step3_error = float(
            subject_table.loc[
                valid_step3,
                "absolute_Current_exact_sum_minus_Step3_exact_sum",
            ].max()
        )

        n_step3_fail = int(
            np.sum(
                ~subject_table.loc[
                    valid_step3,
                    "Current_vs_Step3_exact_sum_pass",
                ].astype(bool)
            )
        )
    else:
        max_step3_error = np.nan
        n_step3_fail = 0

    if system_closure is not None:
        max_system_error = float(
            system_closure[
                "absolute_system_contribution_error"
            ].max()
        )

        n_system_fail = int(
            np.sum(
                ~system_closure[
                    "closure_pass"
                ].astype(bool)
            )
        )
    else:
        max_system_error = np.nan
        n_system_fail = 0

    selected_counts = (
        selected_roi_order[
            "system"
        ]
        .value_counts()
        .reindex(
            SYSTEM_ORDER,
            fill_value=0,
        )
    )

    exact_metric = overall_metrics[
        overall_metrics[
            "comparison"
        ] == (
            "Exact_H1H4_contribution_sum_vs_original_recomputed"
        )
    ].iloc[0]

    report_lines = [
        "Exact H1--H4 bridge to the original behavior coupling index",
        f"Generated at: {datetime.now()}",
        "",
        "Purpose",
        "  Verify that the H1--H4 parcel-wise exact contributions",
        "  reconstruct the unchanged original Coupling_index_scaled.",
        "",
        "Spatial scope",
        f"  Selected H1--H4 ROIs: "
        f"{len(selected_roi_order)}",
        f"  H1_sensory: "
        f"{int(selected_counts['H1_sensory'])}",
        f"  H2_attention: "
        f"{int(selected_counts['H2_attention'])}",
        f"  H3_control: "
        f"{int(selected_counts['H3_control'])}",
        f"  H4_DMN: "
        f"{int(selected_counts['H4_DMN'])}",
        f"  Uncovered Schaefer400 ROIs excluded: "
        f"{400 - len(selected_roi_order)}",
        "  Uncovered ROIs are not assigned zero.",
        "",
        "Input",
        f"  EXACT_CONTRIBUTION_WIDE_CSV: "
        f"{EXACT_CONTRIBUTION_WIDE_CSV}",
        f"  SELECTED_ROI_ORDER_CSV: "
        f"{SELECTED_ROI_ORDER_CSV}",
        f"  ORIGINAL_LONG_CSV: "
        f"{ORIGINAL_LONG_CSV}",
        f"  Saved behavior table available: "
        f"{saved_behavior is not None}",
        f"  Step-3 subject QC available: "
        f"{step3_qc is not None}",
        f"  Exact contribution long table available: "
        f"{system_closure is not None}",
        "",
        "Sample",
        f"  Exact contribution subjects: "
        f"{contribution_table.shape[0]}",
        f"  Original long-table subjects: "
        f"{original_long['sub_id'].nunique()}",
        f"  Final matched subjects: "
        f"{subject_table.shape[0]}",
        "",
        "Original-index reproduction",
        f"  Tolerance: "
        f"{ORIGINAL_REPRODUCTION_TOLERANCE}",
        f"  Maximum |recomputed - saved|: "
        f"{max_original_reproduction_error}",
        f"  Subjects exceeding tolerance: "
        f"{n_original_reproduction_fail}",
        "",
        "Exact H1--H4 parcel-sum closure",
        f"  Tolerance: "
        f"{EXACT_CONTRIBUTION_SUM_TOLERANCE}",
        f"  Maximum |parcel sum - recomputed original beta|: "
        f"{max_exact_recomputed_error}",
        f"  Subjects exceeding tolerance: "
        f"{n_exact_recomputed_fail}",
        f"  Maximum |parcel sum - saved original beta|: "
        f"{max_exact_saved_error}",
        f"  Subjects exceeding tolerance versus saved beta: "
        f"{n_exact_saved_fail}",
        "",
        "Agreement statistics: exact sum versus recomputed original",
        f"  Pearson r: "
        f"{float(exact_metric['pearson_r'])}",
        f"  Lin CCC: "
        f"{float(exact_metric['Lin_CCC'])}",
        f"  ICC(A,1): "
        f"{float(exact_metric['ICC_absolute_agreement_single'])}",
        f"  Mean difference: "
        f"{float(exact_metric['mean_difference'])}",
        f"  MAE: "
        f"{float(exact_metric['MAE'])}",
        f"  RMSE: "
        f"{float(exact_metric['RMSE'])}",
        "",
        "Step-3 independent output verification",
        f"  Tolerance: "
        f"{STEP3_QC_TOLERANCE}",
        f"  Maximum |current parcel sum - Step-3 saved parcel sum|: "
        f"{max_step3_error}",
        f"  Subjects exceeding tolerance: "
        f"{n_step3_fail}",
        "",
        "System-level contribution closure",
        f"  Tolerance: "
        f"{SYSTEM_CONTRIBUTION_TOLERANCE}",
        f"  Maximum absolute system contribution error: "
        f"{max_system_error}",
        f"  Subject-system rows exceeding tolerance: "
        f"{n_system_fail}",
        "",
        "Interpretation",
        "  The selected parcel contributions are an exact algebraic",
        "  decomposition of the original four-system behavior coupling",
        "  index, not an approximate cross-scale bridge.",
        "  The original behavior and ADOS analyses remain unchanged.",
        "",
        "Outputs",
        f"  Subject comparison: "
        f"{OUT_SUBJECT_COMPARISON}",
        f"  Overall metrics: "
        f"{OUT_OVERALL_METRICS}",
        f"  Group metrics: "
        f"{OUT_GROUP_METRICS}",
        f"  Original reproduction QC: "
        f"{OUT_ORIGINAL_REPRODUCTION}",
        f"  Exact parcel-sum QC: "
        f"{OUT_EXACT_SUM_CLOSURE}",
        f"  Step-3 QC comparison: "
        f"{OUT_STEP3_QC_COMPARISON}",
        f"  System closure QC: "
        f"{OUT_SYSTEM_CLOSURE}",
        f"  Subject match audit: "
        f"{OUT_SUBJECT_MATCH_AUDIT}",
        f"  ROI coverage audit: "
        f"{OUT_ROI_COVERAGE_AUDIT}",
        f"  Scatter plot: "
        f"{OUT_SCATTER}",
        f"  Bland--Altman plot: "
        f"{OUT_BLAND_ALTMAN}",
    ]

    write_text(
        OUT_REPORT,
        "\n".join(
            report_lines
        ),
    )

    print("")
    print("=" * 80)
    print(
        "Exact H1--H4 behavior bridge validation completed"
    )
    print("=" * 80)

    print(
        f"Selected H1--H4 ROIs: "
        f"{len(selected_roi_order)}"
    )

    print(
        f"Subjects compared: "
        f"{subject_table.shape[0]}"
    )

    print(
        f"Maximum exact parcel-sum error: "
        f"{max_exact_recomputed_error:.16g}"
    )

    print(
        f"Subjects exceeding tolerance: "
        f"{n_exact_recomputed_fail}"
    )

    print(
        f"Subject comparison: "
        f"{OUT_SUBJECT_COMPARISON}"
    )

    print(
        f"QC report: "
        f"{OUT_REPORT}"
    )


if __name__ == "__main__":
    main()
