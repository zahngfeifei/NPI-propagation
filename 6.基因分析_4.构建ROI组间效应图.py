# -*- coding: utf-8 -*-
"""
Gene analysis — revised Step 4
Construct the parcel-wise ASD–HC group-beta map from the exact H1--H4
ROI contributions to the ORIGINAL behavior coupling beta.

Input outcome
-------------
For participant s and selected ROI p:

    exact_behavior_beta_contribution_sp

These parcel values were constructed in revised Step 3 and satisfy:

    sum_{p in H1 union H2 union H3 union H4}
        exact_behavior_beta_contribution_sp
    = original Coupling_index_scaled_s

Parcel-wise model
-----------------
For every H1--H4-covered ROI p:

    contribution_sp =
        intercept_p
        + beta_Group,p * Group_ASD_s
        + beta_Age,p * Age_s
        + beta_FIQ,p * FIQ_s
        + beta_Sex,p * Sex_s
        + error_sp

where:
    Group_ASD = 1 for ASD
    Group_ASD = 0 for HC

Primary imaging-transcriptomic phenotype
----------------------------------------
    beta_HC_minus_ASD,p = -beta_Group,p

Interpretation
--------------
    beta_HC_minus_ASD > 0:
        the ROI's exact contribution to the original behavior coupling beta
        is lower in ASD.

    beta_HC_minus_ASD < 0:
        the ROI's exact contribution is higher in ASD.

Only H1--H4-covered ROIs are included. Uncovered Schaefer400 ROIs are not
inserted and are not assigned zero.

Exact group-level closure
-------------------------
Because all selected parcel models use the same subjects and design matrix:

    sum_p(beta_Group,p)
        = Group coefficient from the model fitted to
          sum_p(contribution_sp)

Since the participant-level parcel sum equals the original behavior beta,
the global model is the group model for the original Coupling_index_scaled
on the same complete-case sample.

HC3 robust standard errors, statistics, p values, confidence intervals,
and FDR q values are saved. The continuous unthresholded HC-minus-ASD beta
map should be used for AHBA/PLS analysis.
"""

from __future__ import annotations

import re
from datetime import datetime
from pathlib import Path
from typing import List, Optional, Tuple

import numpy as np
import pandas as pd
from scipy import stats
from statsmodels.stats.multitest import multipletests


PROJECT_ROOT = Path(__file__).resolve().parent


# ============================================================
# 1. PATH SETTINGS — ALL ABSOLUTE PATHS
# ============================================================

# Step 3 输出：
# 每个被试在 H1--H4 所覆盖 ROI 上的精确耦合贡献。
CONTRIBUTION_WIDE_CSV = Path(
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
# 用于再次验证所有 ROI 贡献之和是否等于原始行为耦合 beta。
# 该文件是可选文件；不存在时跳过该项外部验证。
EXACT_SUBJECT_QC_CSV: Optional[Path] = Path(
    PROJECT_ROOT
    / "ABIDE2_主流程结果3_耦合指标贡献"
    / "h1-h4-exact-behavior-beta-subject-qc.csv"
)

# Step 4 输出目录。
OUT_DIR = Path(
    PROJECT_ROOT / "ABIDE2_主流程结果4_ROI组间效应"
)
OUT_DIR.mkdir(
    parents=True,
    exist_ok=True,
)


# ============================================================
# 2. MODEL SETTINGS
# ============================================================

GROUP_COLUMN = "Group"
AGE_COLUMN = "Age"
SEX_COLUMN = "Sex"
FIQ_COLUMN = "FIQ"
SITE_COLUMN = "Site"

ASD_GROUP_VALUES = {
    "1",
    "asd",
    "autism",
    "autistic",
    "patient",
}

HC_GROUP_VALUES = {
    "0",
    "2",
    "hc",
    "control",
    "healthy control",
    "healthy_control",
    "td",
    "typical",
    "typically developing",
}

CONTINUOUS_COVARIATES = [
    AGE_COLUMN,
    FIQ_COLUMN,
]

CATEGORICAL_COVARIATES = [
    SEX_COLUMN,
]

# Site is not included because ROI early slopes were harmonized by ComBat
# while preserving Group, Age, Sex, and FIQ.
INCLUDE_SITE_IN_GROUP_MODEL = False

USE_T_DISTRIBUTION_FOR_HC3 = True
ALPHA = 0.05
FDR_METHOD = "fdr_bh"

MIN_HC3_DENOMINATOR = 1e-12
COEFFICIENT_CLOSURE_TOLERANCE = 1e-12
SUBJECT_SUM_CLOSURE_TOLERANCE = 1e-10

CONTRIBUTION_COLUMN_PATTERN = re.compile(
    r"^ROI_(\d+)_exact_behavior_beta_contribution$",
    flags=re.IGNORECASE,
)


# ============================================================
# 3. OUTPUT FILES
# ============================================================

OUT_FULL_RESULTS = (
    OUT_DIR / "h1-h4-group-effect-full-results.csv"
)

OUT_PRIMARY_MAP_CSV = (
    OUT_DIR / "h1-h4-group-beta-hc-minus-asd-map.csv"
)

# Array order is exactly the order in OUT_SELECTED_ROI_ORDER_USED.
OUT_PRIMARY_MAP_NPY = (
    OUT_DIR / "h1-h4-group-beta-hc-minus-asd.npy"
)

OUT_ASD_MINUS_HC_MAP_NPY = (
    OUT_DIR / "h1-h4-group-beta-asd-minus-hc.npy"
)

OUT_T_MAP_HC_MINUS_ASD_NPY = (
    OUT_DIR / "h1-h4-group-t-hc-minus-asd.npy"
)

OUT_P_MAP_NPY = (
    OUT_DIR / "h1-h4-group-p-value.npy"
)

OUT_Q_MAP_NPY = (
    OUT_DIR / "h1-h4-group-fdr-q-value.npy"
)

OUT_SELECTED_ROI_ORDER_USED = (
    OUT_DIR / "h1-h4-selected-roi-order-used.csv"
)

OUT_DESIGN_MATRIX = (
    OUT_DIR / "h1-h4-group-model-design-matrix.csv"
)

OUT_SUBJECTS_USED = (
    OUT_DIR / "h1-h4-group-model-subjects-used.csv"
)

OUT_DROPPED_SUBJECTS = (
    OUT_DIR / "h1-h4-group-model-dropped-subjects.csv"
)

OUT_SUBJECT_SUM_QC = (
    OUT_DIR / "h1-h4-subject-contribution-sum-qc.csv"
)

OUT_GLOBAL_MODEL = (
    OUT_DIR / "h1-h4-original-behavior-beta-group-model.csv"
)

OUT_CLOSURE = (
    OUT_DIR / "h1-h4-group-coefficient-closure.csv"
)

OUT_MODEL_SPECIFICATION = (
    OUT_DIR / "h1-h4-group-model-specification.txt"
)

OUT_QC = (
    OUT_DIR / "h1-h4-group-beta-qc-report.txt"
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


# ============================================================
# 5. LOAD SELECTED H1--H4 ROI ORDER
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

    table = clean_columns(table)

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
            f"Selected ROI order file is missing columns: {missing}"
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

    expected_positions = list(
        range(len(table))
    )

    if (
        table[
            "selected_output_position_0based"
        ].tolist()
        != expected_positions
    ):
        raise ValueError(
            "selected_output_position_0based must be a complete "
            "consecutive sequence beginning at 0."
        )

    if table["roi_index_0based"].duplicated().any():
        duplicated = (
            table.loc[
                table[
                    "roi_index_0based"
                ].duplicated(),
                "roi_index_0based",
            ]
            .tolist()
        )

        raise ValueError(
            f"Duplicated selected ROI indices: {duplicated[:10]}"
        )

    if table["contribution_column"].duplicated().any():
        raise ValueError(
            "Duplicated contribution-column names in selected ROI order."
        )

    expected_roi_id = (
        table["roi_index_0based"]
        + 1
    )

    if not np.array_equal(
        expected_roi_id.to_numpy(),
        table["ROI_ID_1based"].to_numpy(),
    ):
        raise ValueError(
            "ROI_ID_1based is inconsistent with roi_index_0based + 1."
        )

    allowed_systems = {
        "H1_sensory",
        "H2_attention",
        "H3_control",
        "H4_DMN",
    }

    unexpected_systems = sorted(
        set(table["system"])
        - allowed_systems
    )

    if unexpected_systems:
        raise ValueError(
            f"Unexpected hierarchy-system labels: {unexpected_systems}"
        )

    return table


# ============================================================
# 6. LOAD CONTRIBUTION DATA
# ============================================================

def discover_contribution_columns(
    table: pd.DataFrame,
) -> Tuple[List[str], List[int]]:
    parsed = []

    for column in table.columns:
        match = CONTRIBUTION_COLUMN_PATTERN.fullmatch(
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
            "No exact behavior-beta contribution columns were found. "
            "Expected names such as "
            "ROI_000_exact_behavior_beta_contribution."
        )

    parsed = sorted(
        parsed,
        key=lambda item: item[0],
    )

    roi_indices = [
        item[0]
        for item in parsed
    ]

    contribution_columns = [
        item[1]
        for item in parsed
    ]

    if len(set(roi_indices)) != len(roi_indices):
        raise ValueError(
            "Duplicate ROI indices found in contribution columns."
        )

    # The selected ROI indices are not required to be consecutive,
    # because uncovered Schaefer400 ROIs are intentionally absent.
    return contribution_columns, roi_indices


def load_contribution_table(
    path: Path,
    selected_roi_order: pd.DataFrame,
) -> Tuple[pd.DataFrame, List[str], List[int]]:
    if not path.exists():
        raise FileNotFoundError(
            f"Contribution CSV not found: {path}"
        )

    table = pd.read_csv(
        path,
        low_memory=False,
        encoding="utf-8-sig",
    )

    table = clean_columns(table)

    required = [
        "sub_id",
        GROUP_COLUMN,
        AGE_COLUMN,
        SEX_COLUMN,
        FIQ_COLUMN,
    ]

    missing = [
        column
        for column in required
        if column not in table.columns
    ]

    if missing:
        raise ValueError(
            f"Contribution table is missing required columns: {missing}"
        )

    table = table.copy()

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
            "Duplicate subjects found in contribution table. "
            f"Examples: {examples}"
        )

    discovered_columns, discovered_rois = (
        discover_contribution_columns(
            table
        )
    )

    expected_columns = (
        selected_roi_order[
            "contribution_column"
        ].tolist()
    )

    expected_rois = (
        selected_roi_order[
            "roi_index_0based"
        ].tolist()
    )

    missing_expected_columns = [
        column
        for column in expected_columns
        if column not in table.columns
    ]

    unexpected_discovered_columns = [
        column
        for column in discovered_columns
        if column not in set(expected_columns)
    ]

    if missing_expected_columns:
        raise ValueError(
            "Contribution table is missing selected H1--H4 columns. "
            f"Examples: {missing_expected_columns[:10]}"
        )

    if unexpected_discovered_columns:
        raise ValueError(
            "Contribution table contains contribution columns not listed "
            "in the selected ROI order. "
            f"Examples: {unexpected_discovered_columns[:10]}"
        )

    if set(discovered_rois) != set(expected_rois):
        raise ValueError(
            "ROI indices discovered from contribution columns do not "
            "match the selected H1--H4 ROI order."
        )

    for column in expected_columns:
        table[column] = pd.to_numeric(
            table[column],
            errors="coerce",
        )

    return (
        table,
        expected_columns,
        expected_rois,
    )


# ============================================================
# 7. GROUP CODING
# ============================================================

def canonical_group_value(value) -> str:
    if pd.isna(value):
        return ""

    text = str(value).strip().lower()

    if re.fullmatch(r"-?\d+\.0", text):
        text = str(int(float(text)))

    return text


def encode_group_asd(
    series: pd.Series,
) -> pd.Series:
    encoded = []
    unknown_values = set()

    for value in series:
        canonical = canonical_group_value(
            value
        )

        if canonical in ASD_GROUP_VALUES:
            encoded.append(1.0)

        elif canonical in HC_GROUP_VALUES:
            encoded.append(0.0)

        else:
            encoded.append(np.nan)

            if canonical != "":
                unknown_values.add(
                    str(value)
                )

    if unknown_values:
        raise ValueError(
            "Unrecognized Group values: "
            f"{sorted(unknown_values)}\n"
            "Edit ASD_GROUP_VALUES and HC_GROUP_VALUES."
        )

    return pd.Series(
        encoded,
        index=series.index,
        dtype=float,
    )


# ============================================================
# 8. BUILD ONE SHARED COMPLETE-CASE DESIGN
# ============================================================

def prepare_analysis_sample(
    table: pd.DataFrame,
    contribution_columns: List[str],
) -> Tuple[
    pd.DataFrame,
    pd.DataFrame,
    pd.DataFrame,
]:
    working = table.copy()

    working["Group_ASD"] = encode_group_asd(
        working[GROUP_COLUMN]
    )

    for column in CONTINUOUS_COVARIATES:
        if column not in working.columns:
            raise ValueError(
                f"Missing continuous covariate: {column}"
            )

        working[column] = pd.to_numeric(
            working[column],
            errors="coerce",
        )

    categorical_columns = list(
        CATEGORICAL_COVARIATES
    )

    if INCLUDE_SITE_IN_GROUP_MODEL:
        if SITE_COLUMN not in working.columns:
            raise ValueError(
                f"Site was requested but is missing: {SITE_COLUMN}"
            )

        categorical_columns.append(
            SITE_COLUMN
        )

    for column in categorical_columns:
        working[column] = (
            working[column]
            .astype("string")
            .str.strip()
        )

        working.loc[
            working[column].isin(
                [
                    "",
                    "nan",
                    "None",
                    "<NA>",
                ]
            ),
            column,
        ] = pd.NA

    outcome_matrix = working[
        contribution_columns
    ].to_numpy(
        dtype=np.float64
    )

    finite_outcomes = np.isfinite(
        outcome_matrix
    ).all(axis=1)

    required_columns = [
        "sub_id",
        "Group_ASD",
    ] + CONTINUOUS_COVARIATES + categorical_columns

    complete_covariates = (
        working[
            required_columns
        ]
        .notna()
        .all(axis=1)
        .to_numpy()
    )

    keep_mask = (
        finite_outcomes
        & complete_covariates
    )

    dropped = working.loc[
        ~keep_mask,
        [
            column
            for column in [
                "sub_id",
                GROUP_COLUMN,
                AGE_COLUMN,
                SEX_COLUMN,
                FIQ_COLUMN,
                SITE_COLUMN,
            ]
            if column in working.columns
        ],
    ].copy()

    dropped["reason"] = (
        "missing/non-finite model variable or selected ROI outcome"
    )

    analysis = (
        working.loc[
            keep_mask
        ]
        .copy()
        .reset_index(drop=True)
    )

    if analysis.empty:
        raise RuntimeError(
            "No complete-case subjects remained."
        )

    if analysis["Group_ASD"].nunique() != 2:
        raise ValueError(
            "Both ASD and HC groups must be present."
        )

    design_parts = [
        pd.DataFrame(
            {
                "Intercept": np.ones(
                    len(analysis),
                    dtype=float,
                ),
                "Group_ASD": analysis[
                    "Group_ASD"
                ].to_numpy(
                    dtype=float
                ),
            }
        )
    ]

    for column in CONTINUOUS_COVARIATES:
        design_parts.append(
            pd.DataFrame(
                {
                    column: analysis[
                        column
                    ].to_numpy(
                        dtype=float
                    )
                }
            )
        )

    for column in categorical_columns:
        category_values = (
            analysis[column]
            .astype(str)
            .str.strip()
        )

        dummies = pd.get_dummies(
            category_values,
            prefix=column,
            drop_first=True,
            dtype=float,
        )

        if dummies.shape[1] == 0:
            raise ValueError(
                f"Categorical covariate {column} has fewer than two levels."
            )

        design_parts.append(
            dummies.reset_index(drop=True)
        )

    design = pd.concat(
        design_parts,
        axis=1,
    )

    design.insert(
        0,
        "sub_id",
        analysis["sub_id"].values,
    )

    numeric_design = design.drop(
        columns=["sub_id"]
    )

    matrix = numeric_design.to_numpy(
        dtype=np.float64
    )

    rank = int(
        np.linalg.matrix_rank(
            matrix
        )
    )

    if rank != matrix.shape[1]:
        raise ValueError(
            "Design matrix is rank deficient.\n"
            f"Rank={rank}; columns={matrix.shape[1]}; "
            f"names={numeric_design.columns.tolist()}"
        )

    return analysis, design, dropped


# ============================================================
# 9. MULTIVARIATE OLS + HC3
# ============================================================

def fit_shared_design_ols_hc3(
    design: pd.DataFrame,
    outcome_matrix: np.ndarray,
) -> dict:
    design_columns = [
        column
        for column in design.columns
        if column != "sub_id"
    ]

    X = design[
        design_columns
    ].to_numpy(
        dtype=np.float64
    )

    Y = np.asarray(
        outcome_matrix,
        dtype=np.float64,
    )

    if Y.ndim == 1:
        Y = Y[:, None]

    if X.shape[0] != Y.shape[0]:
        raise ValueError(
            "X and Y have different numbers of subjects."
        )

    n_subject, n_parameter = X.shape
    n_outcome = Y.shape[1]

    degrees_of_freedom = (
        n_subject - n_parameter
    )

    if degrees_of_freedom <= 0:
        raise ValueError(
            "Residual degrees of freedom must be positive."
        )

    xtx = X.T @ X
    xtx_inverse = np.linalg.inv(
        xtx
    )

    coefficients = (
        xtx_inverse
        @ X.T
        @ Y
    )

    fitted = X @ coefficients
    residuals = Y - fitted

    hat_diagonal = np.sum(
        (X @ xtx_inverse) * X,
        axis=1,
    )

    one_minus_hat = (
        1.0 - hat_diagonal
    )

    if np.any(
        one_minus_hat
        <= MIN_HC3_DENOMINATOR
    ):
        raise RuntimeError(
            "At least one observation has 1-h_ii too close to zero."
        )

    adjusted_residuals = (
        residuals
        / one_minus_hat[:, None]
    )

    standard_errors = np.full(
        (
            n_parameter,
            n_outcome,
        ),
        np.nan,
        dtype=np.float64,
    )

    for parameter_index in range(
        n_parameter
    ):
        coefficient_weights = (
            X
            @ xtx_inverse[
                :,
                parameter_index,
            ]
        )

        variances = np.sum(
            (
                coefficient_weights[:, None]
                ** 2
            )
            * (
                adjusted_residuals ** 2
            ),
            axis=0,
        )

        standard_errors[
            parameter_index,
            :,
        ] = np.sqrt(
            np.maximum(
                variances,
                0.0,
            )
        )

    with np.errstate(
        divide="ignore",
        invalid="ignore",
    ):
        statistics = (
            coefficients
            / standard_errors
        )

    if USE_T_DISTRIBUTION_FOR_HC3:
        p_values = 2.0 * stats.t.sf(
            np.abs(statistics),
            df=degrees_of_freedom,
        )

        critical_value = float(
            stats.t.ppf(
                1.0 - ALPHA / 2.0,
                df=degrees_of_freedom,
            )
        )
    else:
        p_values = 2.0 * stats.norm.sf(
            np.abs(statistics)
        )

        critical_value = float(
            stats.norm.ppf(
                1.0 - ALPHA / 2.0
            )
        )

    confidence_lower = (
        coefficients
        - critical_value
        * standard_errors
    )

    confidence_upper = (
        coefficients
        + critical_value
        * standard_errors
    )

    residual_sum_squares = np.sum(
        residuals ** 2,
        axis=0,
    )

    outcome_centered = (
        Y
        - np.mean(
            Y,
            axis=0,
            keepdims=True,
        )
    )

    total_sum_squares = np.sum(
        outcome_centered ** 2,
        axis=0,
    )

    with np.errstate(
        divide="ignore",
        invalid="ignore",
    ):
        r_squared = (
            1.0
            - residual_sum_squares
            / total_sum_squares
        )

        adjusted_r_squared = (
            1.0
            - (
                1.0 - r_squared
            )
            * (
                n_subject - 1
            )
            / degrees_of_freedom
        )

    return {
        "design_columns": design_columns,
        "coefficients": coefficients,
        "standard_errors": standard_errors,
        "statistics": statistics,
        "p_values": p_values,
        "confidence_lower": confidence_lower,
        "confidence_upper": confidence_upper,
        "r_squared": r_squared,
        "adjusted_r_squared": adjusted_r_squared,
        "hat_diagonal": hat_diagonal,
        "degrees_of_freedom": degrees_of_freedom,
    }


# ============================================================
# 10. OPTIONAL STEP-3 SUBJECT CLOSURE VERIFICATION
# ============================================================

def build_subject_sum_qc(
    analysis_table: pd.DataFrame,
    contribution_columns: List[str],
) -> pd.DataFrame:
    output = pd.DataFrame(
        {
            "sub_id":
                analysis_table["sub_id"].values,
            "sum_selected_ROI_contributions":
                analysis_table[
                    contribution_columns
                ].sum(axis=1).to_numpy(
                    dtype=float
                ),
        }
    )

    if (
        EXACT_SUBJECT_QC_CSV is not None
        and EXACT_SUBJECT_QC_CSV.exists()
    ):
        step3 = pd.read_csv(
            EXACT_SUBJECT_QC_CSV,
            low_memory=False,
            encoding="utf-8-sig",
        )

        step3 = clean_columns(
            step3
        )

        if (
            "sub_id" not in step3.columns
            or "original_behavior_beta_closed_form"
            not in step3.columns
        ):
            raise ValueError(
                "Step-3 subject QC file must contain sub_id and "
                "original_behavior_beta_closed_form."
            )

        step3["sub_id"] = (
            step3["sub_id"]
            .apply(normalize_sub_id)
        )

        step3[
            "original_behavior_beta_closed_form"
        ] = pd.to_numeric(
            step3[
                "original_behavior_beta_closed_form"
            ],
            errors="coerce",
        )

        step3 = (
            step3[
                [
                    "sub_id",
                    "original_behavior_beta_closed_form",
                ]
            ]
            .drop_duplicates(
                subset=["sub_id"],
                keep="first",
            )
        )

        output = output.merge(
            step3,
            on="sub_id",
            how="left",
            validate="one_to_one",
        )

        output[
            "sum_minus_original_behavior_beta"
        ] = (
            output[
                "sum_selected_ROI_contributions"
            ]
            - output[
                "original_behavior_beta_closed_form"
            ]
        )

        output[
            "absolute_sum_error"
        ] = np.abs(
            output[
                "sum_minus_original_behavior_beta"
            ]
        )

        output[
            "closure_pass"
        ] = (
            output[
                "absolute_sum_error"
            ]
            <= SUBJECT_SUM_CLOSURE_TOLERANCE
        )

    return output


# ============================================================
# 11. MAIN
# ============================================================

def main() -> None:
    print("=" * 80)
    print(
        "Constructing H1--H4 parcel-wise ASD–HC group-beta map "
        "for exact behavior-beta contributions"
    )
    print("=" * 80)

    required_input_paths = {
        "CONTRIBUTION_WIDE_CSV":
            CONTRIBUTION_WIDE_CSV,
        "SELECTED_ROI_ORDER_CSV":
            SELECTED_ROI_ORDER_CSV,
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
        EXACT_SUBJECT_QC_CSV is not None
        and not EXACT_SUBJECT_QC_CSV.exists()
    ):
        print(
            "Warning: optional Step-3 subject QC file "
            "was not found. Subject-level closure "
            "verification will be skipped:"
        )
        print(
            f"  {EXACT_SUBJECT_QC_CSV}"
        )

    print(
        f"Contribution input: "
        f"{CONTRIBUTION_WIDE_CSV}"
    )
    print(
        f"Selected ROI order: "
        f"{SELECTED_ROI_ORDER_CSV}"
    )
    print(
        f"Optional Step-3 subject QC: "
        f"{EXACT_SUBJECT_QC_CSV}"
    )
    print(
        f"Output directory: {OUT_DIR}"
    )

    selected_roi_order = load_selected_roi_order(
        SELECTED_ROI_ORDER_CSV
    )

    (
        full_table,
        contribution_columns,
        roi_indices,
    ) = load_contribution_table(
        CONTRIBUTION_WIDE_CSV,
        selected_roi_order,
    )

    (
        analysis_table,
        design_table,
        dropped_table,
    ) = prepare_analysis_sample(
        table=full_table,
        contribution_columns=contribution_columns,
    )

    selected_roi_order.to_csv(
        OUT_SELECTED_ROI_ORDER_USED,
        index=False,
        encoding="utf-8-sig",
    )

    design_table.to_csv(
        OUT_DESIGN_MATRIX,
        index=False,
        encoding="utf-8-sig",
    )

    subject_output_columns = [
        column
        for column in [
            "sub_id",
            GROUP_COLUMN,
            "Group_ASD",
            AGE_COLUMN,
            SEX_COLUMN,
            FIQ_COLUMN,
            SITE_COLUMN,
            "MeanFD",
        ]
        if column in analysis_table.columns
    ]

    analysis_table[
        subject_output_columns
    ].to_csv(
        OUT_SUBJECTS_USED,
        index=False,
        encoding="utf-8-sig",
    )

    dropped_table.to_csv(
        OUT_DROPPED_SUBJECTS,
        index=False,
        encoding="utf-8-sig",
    )

    contribution_matrix = (
        analysis_table[
            contribution_columns
        ]
        .to_numpy(
            dtype=np.float64
        )
    )

    subject_sum_qc = build_subject_sum_qc(
        analysis_table=analysis_table,
        contribution_columns=contribution_columns,
    )

    subject_sum_qc.to_csv(
        OUT_SUBJECT_SUM_QC,
        index=False,
        encoding="utf-8-sig",
    )

    fit = fit_shared_design_ols_hc3(
        design=design_table,
        outcome_matrix=contribution_matrix,
    )

    design_columns = fit[
        "design_columns"
    ]

    if "Group_ASD" not in design_columns:
        raise RuntimeError(
            "Group_ASD coefficient is missing."
        )

    group_index = design_columns.index(
        "Group_ASD"
    )

    beta_asd_minus_hc = (
        fit["coefficients"][
            group_index,
            :
        ]
    )

    se_group = (
        fit["standard_errors"][
            group_index,
            :
        ]
    )

    t_asd_minus_hc = (
        fit["statistics"][
            group_index,
            :
        ]
    )

    p_group = (
        fit["p_values"][
            group_index,
            :
        ]
    )

    ci_lower_asd_minus_hc = (
        fit["confidence_lower"][
            group_index,
            :
        ]
    )

    ci_upper_asd_minus_hc = (
        fit["confidence_upper"][
            group_index,
            :
        ]
    )

    beta_hc_minus_asd = (
        -beta_asd_minus_hc
    )

    t_hc_minus_asd = (
        -t_asd_minus_hc
    )

    ci_lower_hc_minus_asd = (
        -ci_upper_asd_minus_hc
    )

    ci_upper_hc_minus_asd = (
        -ci_lower_asd_minus_hc
    )

    reject_fdr, q_values, _, _ = (
        multipletests(
            p_group,
            alpha=ALPHA,
            method=FDR_METHOD,
        )
    )

    group_asd = (
        analysis_table[
            "Group_ASD"
        ].to_numpy(
            dtype=float
        )
    )

    hc_mask = group_asd == 0.0
    asd_mask = group_asd == 1.0

    hc_mean = np.mean(
        contribution_matrix[
            hc_mask,
            :
        ],
        axis=0,
    )

    asd_mean = np.mean(
        contribution_matrix[
            asd_mask,
            :
        ],
        axis=0,
    )

    raw_hc_minus_asd = (
        hc_mean - asd_mean
    )

    roi_metadata = (
        selected_roi_order[
            [
                "selected_output_position_0based",
                "roi_index_0based",
                "ROI_ID_1based",
                "system",
                "contribution_column",
            ]
        ]
        .copy()
        .reset_index(drop=True)
    )

    if contribution_columns != (
        roi_metadata[
            "contribution_column"
        ].tolist()
    ):
        raise RuntimeError(
            "Contribution matrix order does not match selected ROI order."
        )

    results = roi_metadata.copy()

    results["N_total"] = (
        contribution_matrix.shape[0]
    )

    results["N_HC"] = int(
        np.sum(hc_mask)
    )

    results["N_ASD"] = int(
        np.sum(asd_mask)
    )

    results["mean_contribution_HC"] = (
        hc_mean
    )

    results["mean_contribution_ASD"] = (
        asd_mean
    )

    results["raw_mean_HC_minus_ASD"] = (
        raw_hc_minus_asd
    )

    results["beta_ASD_minus_HC"] = (
        beta_asd_minus_hc
    )

    results["beta_HC_minus_ASD"] = (
        beta_hc_minus_asd
    )

    results["HC3_SE"] = (
        se_group
    )

    results["t_ASD_minus_HC"] = (
        t_asd_minus_hc
    )

    results["t_HC_minus_ASD"] = (
        t_hc_minus_asd
    )

    results["p_HC3"] = (
        p_group
    )

    results["FDR_q"] = (
        q_values
    )

    results["FDR_significant"] = (
        reject_fdr.astype(bool)
    )

    results["CI95_lower_ASD_minus_HC"] = (
        ci_lower_asd_minus_hc
    )

    results["CI95_upper_ASD_minus_HC"] = (
        ci_upper_asd_minus_hc
    )

    results["CI95_lower_HC_minus_ASD"] = (
        ci_lower_hc_minus_asd
    )

    results["CI95_upper_HC_minus_ASD"] = (
        ci_upper_hc_minus_asd
    )

    results["model_R2"] = (
        fit["r_squared"]
    )

    results["model_adjusted_R2"] = (
        fit["adjusted_r_squared"]
    )

    results.to_csv(
        OUT_FULL_RESULTS,
        index=False,
        encoding="utf-8-sig",
    )

    primary_map = results[
        [
            "selected_output_position_0based",
            "roi_index_0based",
            "ROI_ID_1based",
            "system",
            "beta_HC_minus_ASD",
        ]
    ].copy()

    primary_map.to_csv(
        OUT_PRIMARY_MAP_CSV,
        index=False,
        encoding="utf-8-sig",
    )

    np.save(
        OUT_PRIMARY_MAP_NPY,
        beta_hc_minus_asd.astype(
            np.float64
        ),
    )

    np.save(
        OUT_ASD_MINUS_HC_MAP_NPY,
        beta_asd_minus_hc.astype(
            np.float64
        ),
    )

    np.save(
        OUT_T_MAP_HC_MINUS_ASD_NPY,
        t_hc_minus_asd.astype(
            np.float64
        ),
    )

    np.save(
        OUT_P_MAP_NPY,
        p_group.astype(
            np.float64
        ),
    )

    np.save(
        OUT_Q_MAP_NPY,
        q_values.astype(
            np.float64
        ),
    )

    # ========================================================
    # 12. GLOBAL ORIGINAL-BEHAVIOR-BETA MODEL
    # ========================================================

    global_outcome = np.sum(
        contribution_matrix,
        axis=1,
    )

    global_fit = fit_shared_design_ols_hc3(
        design=design_table,
        outcome_matrix=global_outcome,
    )

    global_rows = []

    for parameter_index, parameter_name in enumerate(
        global_fit["design_columns"]
    ):
        global_rows.append(
            {
                "parameter": parameter_name,
                "coefficient":
                    float(
                        global_fit[
                            "coefficients"
                        ][
                            parameter_index,
                            0,
                        ]
                    ),
                "HC3_SE":
                    float(
                        global_fit[
                            "standard_errors"
                        ][
                            parameter_index,
                            0,
                        ]
                    ),
                "statistic":
                    float(
                        global_fit[
                            "statistics"
                        ][
                            parameter_index,
                            0,
                        ]
                    ),
                "p_value":
                    float(
                        global_fit[
                            "p_values"
                        ][
                            parameter_index,
                            0,
                        ]
                    ),
                "CI95_lower":
                    float(
                        global_fit[
                            "confidence_lower"
                        ][
                            parameter_index,
                            0,
                        ]
                    ),
                "CI95_upper":
                    float(
                        global_fit[
                            "confidence_upper"
                        ][
                            parameter_index,
                            0,
                        ]
                    ),
                "N":
                    contribution_matrix.shape[0],
                "df_resid":
                    global_fit[
                        "degrees_of_freedom"
                    ],
                "model_R2":
                    float(
                        global_fit[
                            "r_squared"
                        ][0]
                    ),
                "model_adjusted_R2":
                    float(
                        global_fit[
                            "adjusted_r_squared"
                        ][0]
                    ),
            }
        )

    global_model = pd.DataFrame(
        global_rows
    )

    global_model.to_csv(
        OUT_GLOBAL_MODEL,
        index=False,
        encoding="utf-8-sig",
    )

    closure_rows = []

    for parameter_index, parameter_name in enumerate(
        fit["design_columns"]
    ):
        sum_roi_coefficient = float(
            np.sum(
                fit["coefficients"][
                    parameter_index,
                    :
                ]
            )
        )

        global_coefficient = float(
            global_fit["coefficients"][
                parameter_index,
                0,
            ]
        )

        difference = (
            sum_roi_coefficient
            - global_coefficient
        )

        closure_rows.append(
            {
                "parameter": parameter_name,
                "n_selected_ROIs":
                    len(roi_indices),
                "sum_of_selected_ROI_coefficients":
                    sum_roi_coefficient,
                "original_behavior_beta_model_coefficient":
                    global_coefficient,
                "difference":
                    difference,
                "absolute_difference":
                    abs(difference),
                "tolerance":
                    COEFFICIENT_CLOSURE_TOLERANCE,
                "pass":
                    bool(
                        abs(difference)
                        <= COEFFICIENT_CLOSURE_TOLERANCE
                    ),
            }
        )

    closure = pd.DataFrame(
        closure_rows
    )

    closure.to_csv(
        OUT_CLOSURE,
        index=False,
        encoding="utf-8-sig",
    )

    # ========================================================
    # 13. MODEL SPECIFICATION
    # ========================================================

    model_lines = [
        "H1--H4 exact behavior-beta parcel group model",
        f"Generated at: {datetime.now()}",
        "",
        "Outcome",
        "  Exact parcel-wise additive contribution to the original",
        "  four-system Coupling_index_scaled.",
        "",
        "Spatial scope",
        "  Only ROIs covered by H1--H4.",
        "  Uncovered Schaefer400 ROIs are absent, not assigned zero.",
        "",
        "Model",
        "  exact_contribution_sp =",
        "      intercept_p",
        "      + beta_Group,p * Group_ASD_s",
        "      + beta_Age,p * Age_s",
        "      + beta_FIQ,p * FIQ_s",
        "      + beta_Sex,p * Sex_s",
        "      + error_sp",
        "",
        "Group coding",
        "  Group_ASD = 1: ASD",
        "  Group_ASD = 0: HC",
        "",
        "Primary map",
        "  beta_HC_minus_ASD = -beta_Group",
        "",
        "Inference",
        "  HC3 heteroscedasticity-robust standard errors",
        f"  Use t distribution: {USE_T_DISTRIBUTION_FOR_HC3}",
        f"  alpha: {ALPHA}",
        f"  FDR method: {FDR_METHOD}",
        "",
        "Shared design columns",
    ]

    model_lines.extend(
        [
            f"  {column}"
            for column in fit[
                "design_columns"
            ]
        ]
    )

    write_text(
        OUT_MODEL_SPECIFICATION,
        "\n".join(model_lines),
    )

    # ========================================================
    # 14. QC REPORT
    # ========================================================

    group_closure_row = closure[
        closure["parameter"]
        == "Group_ASD"
    ]

    if group_closure_row.empty:
        raise RuntimeError(
            "Group_ASD closure result is missing."
        )

    group_closure_abs_diff = float(
        group_closure_row[
            "absolute_difference"
        ].iloc[0]
    )

    global_group_row = global_model[
        global_model["parameter"]
        == "Group_ASD"
    ].iloc[0]

    max_hat = float(
        np.max(
            fit["hat_diagonal"]
        )
    )

    min_one_minus_hat = float(
        np.min(
            1.0
            - fit["hat_diagonal"]
        )
    )

    if "absolute_sum_error" in subject_sum_qc.columns:
        valid_subject_sum = pd.to_numeric(
            subject_sum_qc[
                "absolute_sum_error"
            ],
            errors="coerce",
        ).notna()

        max_subject_sum_error = float(
            subject_sum_qc.loc[
                valid_subject_sum,
                "absolute_sum_error",
            ].max()
        )

        n_subject_sum_fail = int(
            np.sum(
                ~subject_sum_qc.loc[
                    valid_subject_sum,
                    "closure_pass",
                ].astype(bool)
            )
        )
    else:
        max_subject_sum_error = np.nan
        n_subject_sum_fail = 0

    system_counts = (
        selected_roi_order[
            "system"
        ]
        .value_counts()
        .reindex(
            [
                "H1_sensory",
                "H2_attention",
                "H3_control",
                "H4_DMN",
            ],
            fill_value=0,
        )
    )

    qc_lines = [
        "H1--H4 exact behavior-beta group-map QC report",
        f"Generated at: {datetime.now()}",
        "",
        "Input",
        f"  CONTRIBUTION_WIDE_CSV: {CONTRIBUTION_WIDE_CSV}",
        f"  SELECTED_ROI_ORDER_CSV: {SELECTED_ROI_ORDER_CSV}",
        f"  EXACT_SUBJECT_QC_CSV: {EXACT_SUBJECT_QC_CSV}",
        "",
        "Spatial scope",
        f"  Selected H1--H4 ROIs: {len(roi_indices)}",
        f"  H1_sensory ROIs: {int(system_counts['H1_sensory'])}",
        f"  H2_attention ROIs: {int(system_counts['H2_attention'])}",
        f"  H3_control ROIs: {int(system_counts['H3_control'])}",
        f"  H4_DMN ROIs: {int(system_counts['H4_DMN'])}",
        "  Uncovered Schaefer400 ROIs are excluded and not assigned zero.",
        "",
        "Sample",
        f"  Input subjects: {full_table.shape[0]}",
        f"  Complete-case subjects: {analysis_table.shape[0]}",
        f"  Dropped subjects: {dropped_table.shape[0]}",
        f"  HC subjects: {int(np.sum(hc_mask))}",
        f"  ASD subjects: {int(np.sum(asd_mask))}",
        "",
        "Design",
        f"  Columns: {', '.join(fit['design_columns'])}",
        f"  Number of parameters: {len(fit['design_columns'])}",
        f"  Residual degrees of freedom: {fit['degrees_of_freedom']}",
        f"  Include Site after ComBat: {INCLUDE_SITE_IN_GROUP_MODEL}",
        "",
        "Step-3 participant-level exact closure",
        f"  Tolerance: {SUBJECT_SUM_CLOSURE_TOLERANCE}",
        f"  Maximum |parcel sum - original behavior beta|: "
        f"{max_subject_sum_error}",
        f"  Subjects exceeding tolerance: {n_subject_sum_fail}",
        "",
        "Group effect orientation",
        "  Fitted coefficient: ASD minus HC",
        "  Primary gene-analysis map: HC minus ASD",
        "",
        "HC3 diagnostics",
        f"  Maximum hat diagonal: {max_hat:.12g}",
        f"  Minimum 1-h_ii: {min_one_minus_hat:.12g}",
        "",
        "Primary beta-map distribution: HC minus ASD",
        f"  mean: {np.mean(beta_hc_minus_asd):.12g}",
        f"  sd: {np.std(beta_hc_minus_asd):.12g}",
        f"  min: {np.min(beta_hc_minus_asd):.12g}",
        f"  max: {np.max(beta_hc_minus_asd):.12g}",
        "",
        "ROI-wise inference",
        f"  Nominal p < {ALPHA}: {int(np.sum(p_group < ALPHA))}",
        f"  FDR q < {ALPHA}: {int(np.sum(reject_fdr))}",
        f"  Minimum p: {np.min(p_group):.12g}",
        f"  Minimum FDR q: {np.min(q_values):.12g}",
        "",
        "Original behavior-beta global model",
        f"  Global ASD-minus-HC coefficient: "
        f"{float(global_group_row['coefficient']):.12g}",
        f"  Global HC3 SE: {float(global_group_row['HC3_SE']):.12g}",
        f"  Global statistic: {float(global_group_row['statistic']):.12g}",
        f"  Global p: {float(global_group_row['p_value']):.12g}",
        "",
        "Exact coefficient closure",
        "  The sum of selected parcel coefficients should equal",
        "  the coefficient from the original behavior-beta model.",
        f"  Group absolute closure error: "
        f"{group_closure_abs_diff:.16g}",
        f"  Tolerance: {COEFFICIENT_CLOSURE_TOLERANCE}",
        f"  All coefficient closure tests passed: "
        f"{bool(closure['pass'].all())}",
        "",
        "Primary output for AHBA/PLS",
        f"  {OUT_PRIMARY_MAP_CSV}",
        f"  {OUT_PRIMARY_MAP_NPY}",
        "  Array order is defined by:",
        f"  {OUT_SELECTED_ROI_ORDER_USED}",
        "",
        "Interpretation",
        "  Positive beta_HC_minus_ASD:",
        "    the ROI contribution to the original behavior beta",
        "    is lower in ASD.",
        "  Negative beta_HC_minus_ASD:",
        "    the ROI contribution is higher in ASD.",
        "  Use the continuous unthresholded selected-ROI beta map.",
        "",
        "Outputs",
        f"  Full selected-ROI results: {OUT_FULL_RESULTS}",
        f"  Primary beta map CSV: {OUT_PRIMARY_MAP_CSV}",
        f"  Primary beta map NPY: {OUT_PRIMARY_MAP_NPY}",
        f"  Selected ROI order used: {OUT_SELECTED_ROI_ORDER_USED}",
        f"  ASD-minus-HC beta NPY: {OUT_ASD_MINUS_HC_MAP_NPY}",
        f"  HC-minus-ASD t NPY: {OUT_T_MAP_HC_MINUS_ASD_NPY}",
        f"  p-value NPY: {OUT_P_MAP_NPY}",
        f"  FDR q-value NPY: {OUT_Q_MAP_NPY}",
        f"  Design matrix: {OUT_DESIGN_MATRIX}",
        f"  Subjects used: {OUT_SUBJECTS_USED}",
        f"  Dropped subjects: {OUT_DROPPED_SUBJECTS}",
        f"  Subject sum QC: {OUT_SUBJECT_SUM_QC}",
        f"  Original behavior-beta global model: {OUT_GLOBAL_MODEL}",
        f"  Coefficient closure: {OUT_CLOSURE}",
        f"  Model specification: {OUT_MODEL_SPECIFICATION}",
    ]

    write_text(
        OUT_QC,
        "\n".join(qc_lines),
    )

    print("")
    print("=" * 80)
    print(
        "H1--H4 exact behavior-beta group map completed"
    )
    print("=" * 80)

    print(
        f"Subjects used: {analysis_table.shape[0]}"
    )

    print(
        f"HC: {int(np.sum(hc_mask))}; "
        f"ASD: {int(np.sum(asd_mask))}"
    )

    print(
        f"Selected H1--H4 ROIs: {len(roi_indices)}"
    )

    print(
        f"Primary HC-minus-ASD beta map: {OUT_PRIMARY_MAP_CSV}"
    )

    print(
        f"Selected ROI order: {OUT_SELECTED_ROI_ORDER_USED}"
    )

    print(
        f"Full results: {OUT_FULL_RESULTS}"
    )

    print(
        f"Coefficient closure: {OUT_CLOSURE}"
    )

    print(
        f"QC report: {OUT_QC}"
    )


if __name__ == "__main__":
    main()
