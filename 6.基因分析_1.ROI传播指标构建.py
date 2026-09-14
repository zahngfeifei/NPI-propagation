# -*- coding: utf-8 -*-
"""
Gene analysis — Step 1
Construct ROI-level early propagation slopes from saved V_all.npy files.

Input
-----
For each participant:
    sub-Subxxxxx_V_all.npy

V_all shape:
    steps × ROIs
    normally 40 × 400

Each value:
    V_all[step, roi]
    = propagation intensity reaching that ROI at the corresponding step.

Main output
-----------
1. roi_early_slope_1_10_raw_wide.csv
   Rows = subjects
   Columns = 400 ROI early slopes
   This is the recommended input for ROI-level ComBat.

2. roi_early_slope_1_10_scaled_wide.csv
   Same values multiplied by 1000.
   For convenience only; do not use this for ComBat if preserving the
   original pipeline order.

3. roi_early_slope_1_10_long.csv
   Long-format table.

4. system_reconstruction_subject_system.csv
   Checks whether:
       mean(ROI early slopes within Hk)
   reconstructs:
       Hk system early slope.

5. ROI-set audits and QC report.

This script:
- does not run ComBat;
- does not run behavior analysis;
- does not alter Coupling_index_scaled;
- does not run gene-expression analysis.
"""

from __future__ import annotations

import re
from pathlib import Path
from datetime import datetime
from typing import Dict, List, Optional, Tuple

import numpy as np
import pandas as pd


PROJECT_ROOT = Path(__file__).resolve().parent


# ============================================================
# 1. PATH SETTINGS
# ============================================================

# 每个被试的 sub-Subxxxxx_V_all.npy 文件目录。
V_ALL_DIR = Path(
    PROJECT_ROOT / "ABIDE2_新结果1" / "curves_per_subject"
)

# 上游传播分析生成的、未经 ComBat 的 H1--H4 系统级指标。
# 仅用于系统斜率重建 QC。
SYSTEM_METRICS_CSV = Path(
    PROJECT_ROOT / "ABIDE2_新结果1" / "EC_SEC_metrics_all_subjects.csv"
)

# ROI 与层级系统的对应表。
# 需要包含列：
#   ROI_index_0based
#   Hierarchy
ROI_DEFINITIONS_CSV = Path(
    PROJECT_ROOT / "ABIDE2_主流程必要输入" / "roi-definitions.csv"
)

# 输出目录。
OUT_DIR = Path(
    PROJECT_ROOT / "ABIDE2_主流程结果1_ROI传播指标"
)
OUT_DIR.mkdir(
    parents=True,
    exist_ok=True,
)


# ============================================================
# 2. ANALYSIS SETTINGS
# ============================================================

SYSTEMS = {
    "H1_sensory": "H1_sensory.csv",
    "H2_attention": "H2_attention.csv",
    "H3_control": "H3_control.csv",
    "H4_DMN": "H4_DMN.csv",
}

SYSTEM_METRIC_SUFFIX = "early_slope_1_10"

# Must match the upstream propagation analysis.
EARLY_L_START = 1
EARLY_L_END = 10
EPS = 1e-8

# Same later scaling as the system-level integration script.
OUTCOME_SCALE = 1000.0

# Expected Schaefer parcellation size.
# Set to None if it should only be inferred from V_all.
EXPECTED_N_ROI: Optional[int] = 400

# Numerical tolerance for reconstruction QC.
RECONSTRUCTION_ABS_TOL = 1e-8

# Require every value in steps 1--10 to be finite.
REQUIRE_FINITE_EARLY_WINDOW = True

ROI_COLUMN_TEMPLATE = "ROI_{roi:03d}_early_slope_1_10"


# ============================================================
# 3. OUTPUT FILES
# ============================================================

OUT_RAW_WIDE = (
    OUT_DIR / "roi-early-slope-1-10-raw-wide.csv"
)

OUT_SCALED_WIDE = (
    OUT_DIR / "roi-early-slope-1-10-scaled-wide.csv"
)

OUT_LONG = (
    OUT_DIR / "roi-early-slope-1-10-long.csv"
)

OUT_RAW_NPY = (
    OUT_DIR / "roi-early-slope-1-10-raw.npy"
)

OUT_SCALED_NPY = (
    OUT_DIR / "roi-early-slope-1-10-scaled.npy"
)

OUT_SUBJECT_ORDER = (
    OUT_DIR / "roi-early-slope-subject-order.csv"
)

OUT_RECON_DETAIL = (
    OUT_DIR / "system-reconstruction-subject-system.csv"
)

OUT_RECON_SUMMARY = (
    OUT_DIR / "system-reconstruction-summary.csv"
)

OUT_ROI_DICTIONARY = (
    OUT_DIR / "roi-dictionary-and-system-membership.csv"
)

OUT_ROI_SET_AUDIT = (
    OUT_DIR / "roi-set-audit.csv"
)

OUT_FAILED = (
    OUT_DIR / "roi-early-slope-failed-subjects.csv"
)

OUT_QC = (
    OUT_DIR / "roi-early-slope-qc-report.txt"
)


# ============================================================
# 4. GENERAL UTILITIES
# ============================================================

def normalize_sub_id(x) -> Optional[str]:
    """
    Convert different subject ID styles to:
        sub-Sub00000
    """
    if pd.isna(x):
        return None

    s = str(x).strip()

    if re.fullmatch(r"sub-Sub\d{5}", s):
        return s

    match = re.search(
        r"Sub0*(\d+)",
        s,
        flags=re.IGNORECASE,
    )

    if match:
        return f"sub-Sub{int(match.group(1)):05d}"

    if re.fullmatch(r"\d+(\.0)?", s):
        return f"sub-Sub{int(float(s)):05d}"

    return None


def infer_sub_id_from_vall_path(
    path: Path,
) -> Optional[str]:
    """
    Expected filename:
        sub-Subxxxxx_V_all.npy
    """
    name = re.sub(
        r"_V_all\.npy$",
        "",
        path.name,
        flags=re.IGNORECASE,
    )

    return normalize_sub_id(name)


def write_text(
    path: Path,
    text: str,
) -> None:
    with open(
        path,
        "w",
        encoding="utf-8",
    ) as file:
        file.write(text)


def safe_float(x) -> float:
    try:
        return float(x)
    except Exception:
        return np.nan


# ============================================================
# 5. ROI-DEFINITION FUNCTIONS
# ============================================================

def read_roi_index_0based(
    csv_path: Path,
) -> np.ndarray:
    """
    Read a system ROI list.

    Preferred column:
        ROI_index_0based

    Otherwise:
        use the first CSV column.
    """
    if not csv_path.exists():
        raise FileNotFoundError(
            f"ROI definition file not found: {csv_path}"
        )

    table = pd.read_csv(csv_path)

    if table.empty:
        return np.array([], dtype=int)

    if "ROI_index_0based" in table.columns:
        column = "ROI_index_0based"
    else:
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

    return np.sort(
        np.unique(indices)
    )


def load_system_rois(
    n_roi: int,
) -> Dict[str, np.ndarray]:
    """
    从 roi_to_hierarchy_mapping.csv 中读取并验证 H1--H4 ROI 集合。

    必需列
    ------
    ROI_index_0based:
        ROI 的 0-based 编号，预期范围为 0..n_roi-1。

    Hierarchy:
        ROI 所属层级，预期包括：
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

    roi_definition_table = pd.read_csv(
        ROI_DEFINITIONS_CSV,
        low_memory=False,
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
                f"{system_name} contains out-of-range "
                f"ROI indices. Expected 0..{n_roi - 1}; "
                f"examples={bad[:10].tolist()}"
            )

        system_rois[system_name] = indices

    return system_rois


def build_roi_membership_audit(
    n_roi: int,
    system_rois: Dict[str, np.ndarray],
) -> Tuple[pd.DataFrame, pd.DataFrame]:
    """
    Audit:
    - number of ROIs in each system;
    - uncovered ROIs;
    - ROIs belonging to multiple systems.
    """
    membership_count = np.zeros(
        n_roi,
        dtype=int,
    )

    memberships: List[List[str]] = [
        [] for _ in range(n_roi)
    ]

    system_rows = []

    for system_name, indices in system_rois.items():

        membership_count[indices] += 1

        for roi in indices:
            memberships[int(roi)].append(
                system_name
            )

        system_rows.append(
            {
                "system": system_name,
                "n_roi": int(len(indices)),
                "min_roi": (
                    int(indices.min())
                    if len(indices) > 0
                    else np.nan
                ),
                "max_roi": (
                    int(indices.max())
                    if len(indices) > 0
                    else np.nan
                ),
            }
        )

    roi_rows = []

    for roi in range(n_roi):
        roi_rows.append(
            {
                "roi_index_0based": roi,
                "raw_feature_column":
                    ROI_COLUMN_TEMPLATE.format(
                        roi=roi
                    ),
                "scaled_feature_column":
                    ROI_COLUMN_TEMPLATE.format(
                        roi=roi
                    ) + "_scaled",
                "n_system_memberships":
                    int(membership_count[roi]),
                "system_memberships":
                    ";".join(memberships[roi]),
                "is_uncovered":
                    bool(membership_count[roi] == 0),
                "is_overlapping":
                    bool(membership_count[roi] > 1),
            }
        )

    system_audit = pd.DataFrame(system_rows)

    extra_rows = pd.DataFrame(
        [
            {
                "system": "UNION",
                "n_roi": int(
                    np.sum(membership_count > 0)
                ),
                "min_roi": np.nan,
                "max_roi": np.nan,
            },
            {
                "system": "UNCOVERED",
                "n_roi": int(
                    np.sum(membership_count == 0)
                ),
                "min_roi": np.nan,
                "max_roi": np.nan,
            },
            {
                "system": "OVERLAPPING",
                "n_roi": int(
                    np.sum(membership_count > 1)
                ),
                "min_roi": np.nan,
                "max_roi": np.nan,
            },
        ]
    )

    system_audit = pd.concat(
        [
            system_audit,
            extra_rows,
        ],
        ignore_index=True,
    )

    return (
        pd.DataFrame(roi_rows),
        system_audit,
    )


# ============================================================
# 6. EARLY-SLOPE FUNCTIONS
# ============================================================

def linear_slope_compatible(
    y: np.ndarray,
) -> float:
    """
    Exact reproduction of the upstream linear_slope() function.

    Steps:
    1. retain finite observations;
    2. require at least two observations;
    3. return 0 if the trajectory is numerically constant;
    4. x = 0, 1, ..., n-1;
    5. center x and y;
    6. calculate OLS slope;
    7. add EPS to the denominator, matching upstream code.
    """
    y = np.asarray(
        y,
        dtype=float,
    )

    if len(y) < 2:
        return np.nan

    valid = np.isfinite(y)

    if np.sum(valid) < 2:
        return np.nan

    y = y[valid]

    if np.allclose(
        y,
        y[0],
    ):
        return 0.0

    x = np.arange(
        len(y),
        dtype=float,
    )

    x = x - x.mean()
    y_centered = y - y.mean()

    denominator = (
        np.sum(x * x)
        + EPS
    )

    numerator = np.sum(
        x * y_centered
    )

    return float(
        numerator / denominator
    )


def calculate_roi_early_slopes(
    v_all: np.ndarray,
) -> np.ndarray:
    """
    Compute one early slope for every ROI.

    Input
    -----
    v_all:
        shape = steps × ROIs

    Output
    ------
    slopes:
        shape = ROIs
    """
    if v_all.ndim != 2:
        raise ValueError(
            f"V_all must be 2D. "
            f"Got shape={v_all.shape}"
        )

    n_steps, n_roi = v_all.shape

    if EARLY_L_START < 1:
        raise ValueError(
            "EARLY_L_START must be >= 1."
        )

    if EARLY_L_END > n_steps:
        raise ValueError(
            f"Early window ends at step "
            f"{EARLY_L_END}, but V_all only "
            f"contains {n_steps} steps."
        )

    early = np.asarray(
        v_all[
            EARLY_L_START - 1:
            EARLY_L_END,
            :
        ],
        dtype=float,
    )

    if (
        REQUIRE_FINITE_EARLY_WINDOW
        and not np.isfinite(early).all()
    ):
        n_bad = int(
            np.sum(
                ~np.isfinite(early)
            )
        )

        raise ValueError(
            f"Early V_all window contains "
            f"{n_bad} NaN/Inf values."
        )

    slopes = np.full(
        n_roi,
        np.nan,
        dtype=float,
    )

    for roi in range(n_roi):
        slopes[roi] = linear_slope_compatible(
            early[:, roi]
        )

    return slopes


def reconstruct_system_slopes_from_roi_slopes(
    roi_slopes: np.ndarray,
    system_rois: Dict[str, np.ndarray],
) -> Dict[str, float]:
    """
    Reconstruct each system slope by taking the mean
    of its ROI slopes.
    """
    reconstructed = {}

    for system_name, indices in system_rois.items():

        if len(indices) == 0:
            reconstructed[system_name] = np.nan
        else:
            reconstructed[system_name] = float(
                np.mean(
                    roi_slopes[indices]
                )
            )

    return reconstructed


def recompute_system_slopes_directly(
    v_all: np.ndarray,
    system_rois: Dict[str, np.ndarray],
) -> Dict[str, float]:
    """
    Direct reproduction of the upstream system procedure:

        system SEC curve
        = mean of V_all over system ROIs

        system early slope
        = slope of system curve over steps 1--10
    """
    results = {}

    for system_name, indices in system_rois.items():

        if len(indices) == 0:
            results[system_name] = np.nan
            continue

        system_curve = np.mean(
            v_all[:, indices],
            axis=1,
        )

        early_curve = system_curve[
            EARLY_L_START - 1:
            EARLY_L_END
        ]

        results[system_name] = (
            linear_slope_compatible(
                early_curve
            )
        )

    return results


# ============================================================
# 7. LOAD SAVED SYSTEM METRICS
# ============================================================

def load_system_metrics(
    path: Path,
) -> Optional[pd.DataFrame]:
    """
    Load raw, pre-ComBat H1--H4 early-slope values
    for reconstruction QC.
    """
    if not path.exists():
        print(
            "Warning: system metrics CSV not found. "
            "Saved-metric reconstruction QC will be skipped."
        )
        return None

    table = pd.read_csv(
        path,
        low_memory=False,
    )

    if "sub_id" not in table.columns:
        raise ValueError(
            f"System metrics CSV is missing sub_id: "
            f"{path}"
        )

    table = table.copy()

    table["sub_id_norm"] = (
        table["sub_id"]
        .apply(normalize_sub_id)
    )

    table = table.dropna(
        subset=["sub_id_norm"]
    )

    table = table.drop_duplicates(
        subset=["sub_id_norm"],
        keep="first",
    )

    keep_columns = ["sub_id_norm"]

    for system_name in SYSTEMS:

        column = (
            f"{system_name}_"
            f"{SYSTEM_METRIC_SUFFIX}"
        )

        if column not in table.columns:
            raise ValueError(
                f"Missing system metric column: "
                f"{column}"
            )

        keep_columns.append(column)

    return table[keep_columns].copy()


# ============================================================
# 8. MAIN PROCESSING
# ============================================================

def main() -> None:

    print("=" * 80)
    print("Constructing ROI-level early propagation slopes")
    print("=" * 80)

    # --------------------------------------------------------
    # Locate V_all files
    # --------------------------------------------------------
    v_paths = sorted(
        V_ALL_DIR.glob(
            "sub-*_V_all.npy"
        )
    )

    if len(v_paths) == 0:
        v_paths = sorted(
            V_ALL_DIR.glob(
                "*_V_all.npy"
            )
        )

    if len(v_paths) == 0:
        raise FileNotFoundError(
            f"No *_V_all.npy files found in: "
            f"{V_ALL_DIR}"
        )

    # --------------------------------------------------------
    # Infer dimensions from first file
    # --------------------------------------------------------
    sample_path = v_paths[0]

    sample_v = np.load(
        sample_path,
        mmap_mode="r",
    )

    if sample_v.ndim != 2:
        raise ValueError(
            f"Sample V_all is not 2D. "
            f"File={sample_path}; "
            f"shape={sample_v.shape}"
        )

    n_steps = int(
        sample_v.shape[0]
    )

    n_roi = int(
        sample_v.shape[1]
    )

    if (
        EXPECTED_N_ROI is not None
        and n_roi != EXPECTED_N_ROI
    ):
        raise ValueError(
            f"Expected {EXPECTED_N_ROI} ROIs, "
            f"but V_all contains {n_roi}."
        )

    if EARLY_L_END > n_steps:
        raise ValueError(
            f"EARLY_L_END={EARLY_L_END} "
            f"exceeds V_all steps={n_steps}."
        )

    print(
        f"V_all files found: {len(v_paths)}"
    )
    print(
        f"V_all shape: {n_steps} × {n_roi}"
    )

    # --------------------------------------------------------
    # Load H1--H4 ROI definitions
    # --------------------------------------------------------
    system_rois = load_system_rois(
        n_roi=n_roi
    )

    (
        roi_dictionary,
        roi_set_audit,
    ) = build_roi_membership_audit(
        n_roi=n_roi,
        system_rois=system_rois,
    )

    roi_dictionary.to_csv(
        OUT_ROI_DICTIONARY,
        index=False,
        encoding="utf-8-sig",
    )

    roi_set_audit.to_csv(
        OUT_ROI_SET_AUDIT,
        index=False,
        encoding="utf-8-sig",
    )

    # --------------------------------------------------------
    # Load existing raw system slopes
    # --------------------------------------------------------
    system_metrics = load_system_metrics(
        SYSTEM_METRICS_CSV
    )

    # --------------------------------------------------------
    # Output containers
    # --------------------------------------------------------
    raw_rows: List[dict] = []
    long_rows: List[dict] = []
    reconstruction_rows: List[dict] = []
    failed_rows: List[dict] = []

    roi_feature_columns = [
        ROI_COLUMN_TEMPLATE.format(
            roi=roi
        )
        for roi in range(n_roi)
    ]

    # --------------------------------------------------------
    # Process every subject
    # --------------------------------------------------------
    for file_index, path in enumerate(
        v_paths,
        start=1,
    ):

        sub_id = infer_sub_id_from_vall_path(
            path
        )

        if sub_id is None:

            failed_rows.append(
                {
                    "file": str(path),
                    "sub_id": "",
                    "reason":
                        "subject_id_normalization_failed",
                }
            )

            continue

        try:
            v_all = np.load(path)

            if v_all.ndim != 2:
                raise ValueError(
                    f"V_all is not 2D: "
                    f"shape={v_all.shape}"
                )

            if v_all.shape != (
                n_steps,
                n_roi,
            ):
                raise ValueError(
                    f"Inconsistent V_all shape. "
                    f"Expected {(n_steps, n_roi)}, "
                    f"got {v_all.shape}"
                )

            if not np.isfinite(v_all).all():
                raise ValueError(
                    "V_all contains NaN or Inf."
                )

            # ----------------------------------------------
            # ROI early slopes
            # ----------------------------------------------
            roi_slopes = (
                calculate_roi_early_slopes(
                    v_all
                )
            )

            if not np.isfinite(
                roi_slopes
            ).all():

                n_bad = int(
                    np.sum(
                        ~np.isfinite(roi_slopes)
                    )
                )

                raise ValueError(
                    f"ROI slopes contain "
                    f"{n_bad} non-finite values."
                )

            # ----------------------------------------------
            # Wide raw row
            # ----------------------------------------------
            raw_row = {
                "sub_id": sub_id
            }

            raw_row.update(
                {
                    roi_feature_columns[roi]:
                        float(roi_slopes[roi])
                    for roi in range(n_roi)
                }
            )

            raw_rows.append(raw_row)

            # ----------------------------------------------
            # Long rows
            # ----------------------------------------------
            for roi in range(n_roi):

                long_rows.append(
                    {
                        "sub_id": sub_id,
                        "roi_index_0based": roi,
                        "early_slope_1_10_raw":
                            float(roi_slopes[roi]),
                        "early_slope_1_10_scaled":
                            float(
                                roi_slopes[roi]
                                * OUTCOME_SCALE
                            ),
                    }
                )

            # ----------------------------------------------
            # Reconstruction from ROI slopes
            # ----------------------------------------------
            reconstructed_from_roi = (
                reconstruct_system_slopes_from_roi_slopes(
                    roi_slopes=roi_slopes,
                    system_rois=system_rois,
                )
            )

            # ----------------------------------------------
            # Direct system reconstruction from V_all
            # ----------------------------------------------
            direct_from_vall = (
                recompute_system_slopes_directly(
                    v_all=v_all,
                    system_rois=system_rois,
                )
            )

            # ----------------------------------------------
            # Existing system metric row
            # ----------------------------------------------
            saved_row = None

            if system_metrics is not None:

                matched = system_metrics[
                    system_metrics[
                        "sub_id_norm"
                    ] == sub_id
                ]

                if not matched.empty:
                    saved_row = matched.iloc[0]

            # ----------------------------------------------
            # One QC row per subject × system
            # ----------------------------------------------
            for system_name in SYSTEMS:

                saved_column = (
                    f"{system_name}_"
                    f"{SYSTEM_METRIC_SUFFIX}"
                )

                if saved_row is None:
                    saved_value = np.nan
                else:
                    saved_value = safe_float(
                        saved_row[saved_column]
                    )

                roi_mean_value = (
                    reconstructed_from_roi[
                        system_name
                    ]
                )

                direct_value = (
                    direct_from_vall[
                        system_name
                    ]
                )

                reconstruction_rows.append(
                    {
                        "sub_id": sub_id,
                        "system": system_name,
                        "n_roi_in_system":
                            int(
                                len(
                                    system_rois[
                                        system_name
                                    ]
                                )
                            ),
                        "system_slope_from_mean_roi_slopes":
                            roi_mean_value,
                        "system_slope_direct_from_V_all":
                            direct_value,
                        "system_slope_saved_metric_csv":
                            saved_value,
                        "abs_diff_roi_mean_vs_direct":
                            (
                                abs(
                                    roi_mean_value
                                    - direct_value
                                )
                                if (
                                    np.isfinite(
                                        roi_mean_value
                                    )
                                    and np.isfinite(
                                        direct_value
                                    )
                                )
                                else np.nan
                            ),
                        "abs_diff_direct_vs_saved":
                            (
                                abs(
                                    direct_value
                                    - saved_value
                                )
                                if (
                                    np.isfinite(
                                        direct_value
                                    )
                                    and np.isfinite(
                                        saved_value
                                    )
                                )
                                else np.nan
                            ),
                        "abs_diff_roi_mean_vs_saved":
                            (
                                abs(
                                    roi_mean_value
                                    - saved_value
                                )
                                if (
                                    np.isfinite(
                                        roi_mean_value
                                    )
                                    and np.isfinite(
                                        saved_value
                                    )
                                )
                                else np.nan
                            ),
                    }
                )

            if (
                file_index % 25 == 0
                or file_index == len(v_paths)
            ):
                print(
                    f"Processed "
                    f"{file_index}/{len(v_paths)}"
                )

        except Exception as error:

            failed_rows.append(
                {
                    "file": str(path),
                    "sub_id": sub_id,
                    "reason": repr(error),
                }
            )

    # ========================================================
    # 9. CREATE OUTPUT TABLES
    # ========================================================

    if len(raw_rows) == 0:
        raise RuntimeError(
            "No subject was successfully processed."
        )

    raw_wide = pd.DataFrame(raw_rows)

    raw_wide = (
        raw_wide
        .sort_values("sub_id")
        .reset_index(drop=True)
    )

    # --------------------------------------------------------
    # Scaled version
    # --------------------------------------------------------
    scaled_wide = raw_wide.copy()

    scaled_wide[
        roi_feature_columns
    ] = (
        scaled_wide[
            roi_feature_columns
        ]
        * OUTCOME_SCALE
    )

    scaled_column_mapping = {
        column: column + "_scaled"
        for column in roi_feature_columns
    }

    scaled_wide = scaled_wide.rename(
        columns=scaled_column_mapping
    )

    # --------------------------------------------------------
    # Save wide CSV files
    # --------------------------------------------------------
    raw_wide.to_csv(
        OUT_RAW_WIDE,
        index=False,
        encoding="utf-8-sig",
    )

    scaled_wide.to_csv(
        OUT_SCALED_WIDE,
        index=False,
        encoding="utf-8-sig",
    )

    # --------------------------------------------------------
    # Save NPY matrices
    # --------------------------------------------------------
    raw_matrix = raw_wide[
        roi_feature_columns
    ].to_numpy(
        dtype=np.float64
    )

    scaled_feature_columns = [
        column + "_scaled"
        for column in roi_feature_columns
    ]

    scaled_matrix = scaled_wide[
        scaled_feature_columns
    ].to_numpy(
        dtype=np.float64
    )

    np.save(
        OUT_RAW_NPY,
        raw_matrix,
    )

    np.save(
        OUT_SCALED_NPY,
        scaled_matrix,
    )

    # Save subject order corresponding to NPY rows.
    raw_wide[
        ["sub_id"]
    ].to_csv(
        OUT_SUBJECT_ORDER,
        index=False,
        encoding="utf-8-sig",
    )

    # --------------------------------------------------------
    # Save long-format table
    # --------------------------------------------------------
    long_table = pd.DataFrame(
        long_rows
    )

    long_table = (
        long_table
        .sort_values(
            [
                "sub_id",
                "roi_index_0based",
            ]
        )
        .reset_index(drop=True)
    )

    long_table.to_csv(
        OUT_LONG,
        index=False,
        encoding="utf-8-sig",
    )

    # ========================================================
    # 10. RECONSTRUCTION QC
    # ========================================================

    reconstruction = pd.DataFrame(
        reconstruction_rows
    )

    reconstruction = (
        reconstruction
        .sort_values(
            [
                "sub_id",
                "system",
            ]
        )
        .reset_index(drop=True)
    )

    reconstruction.to_csv(
        OUT_RECON_DETAIL,
        index=False,
        encoding="utf-8-sig",
    )

    reconstruction_summary_rows = []

    comparison_columns = [
        "abs_diff_roi_mean_vs_direct",
        "abs_diff_direct_vs_saved",
        "abs_diff_roi_mean_vs_saved",
    ]

    for system_name, system_table in (
        reconstruction.groupby(
            "system",
            sort=False,
        )
    ):

        for comparison in comparison_columns:

            values = pd.to_numeric(
                system_table[comparison],
                errors="coerce",
            ).dropna()

            reconstruction_summary_rows.append(
                {
                    "system": system_name,
                    "comparison": comparison,
                    "N": int(len(values)),
                    "mean_abs_diff":
                        (
                            float(values.mean())
                            if len(values) > 0
                            else np.nan
                        ),
                    "median_abs_diff":
                        (
                            float(values.median())
                            if len(values) > 0
                            else np.nan
                        ),
                    "max_abs_diff":
                        (
                            float(values.max())
                            if len(values) > 0
                            else np.nan
                        ),
                    "n_above_tolerance":
                        (
                            int(
                                np.sum(
                                    values
                                    > RECONSTRUCTION_ABS_TOL
                                )
                            )
                            if len(values) > 0
                            else 0
                        ),
                    "tolerance":
                        RECONSTRUCTION_ABS_TOL,
                }
            )

    reconstruction_summary = pd.DataFrame(
        reconstruction_summary_rows
    )

    reconstruction_summary.to_csv(
        OUT_RECON_SUMMARY,
        index=False,
        encoding="utf-8-sig",
    )

    # ========================================================
    # 11. FAILED SUBJECTS
    # ========================================================

    failed_table = pd.DataFrame(
        failed_rows,
        columns=[
            "file",
            "sub_id",
            "reason",
        ],
    )

    failed_table.to_csv(
        OUT_FAILED,
        index=False,
        encoding="utf-8-sig",
    )

    # ========================================================
    # 12. QC REPORT
    # ========================================================

    raw_values = raw_matrix.ravel()
    scaled_values = scaled_matrix.ravel()

    n_uncovered = int(
        roi_dictionary[
            "is_uncovered"
        ].sum()
    )

    n_overlapping = int(
        roi_dictionary[
            "is_overlapping"
        ].sum()
    )

    max_roi_mean_vs_direct = (
        pd.to_numeric(
            reconstruction[
                "abs_diff_roi_mean_vs_direct"
            ],
            errors="coerce",
        ).max()
    )

    max_direct_vs_saved = (
        pd.to_numeric(
            reconstruction[
                "abs_diff_direct_vs_saved"
            ],
            errors="coerce",
        ).max()
    )

    qc_lines = [
        "ROI-level early propagation slope QC report",
        f"Generated at: {datetime.now()}",
        "",
        "Input",
        f"  V_ALL_DIR: {V_ALL_DIR}",
        f"  SYSTEM_METRICS_CSV: "
        f"{SYSTEM_METRICS_CSV}",
        f"  ROI_DEFINITIONS_CSV: {ROI_DEFINITIONS_CSV}",
        "",
        "Analysis definition",
        f"  Early window: "
        f"steps {EARLY_L_START}..{EARLY_L_END}",
        f"  EPS: {EPS}",
        f"  Convenience scale: {OUTCOME_SCALE}",
        "  Slope implementation: exact copy of "
        "upstream linear_slope()",
        "",
        "Data dimensions",
        f"  V_all files found: {len(v_paths)}",
        f"  Successful subjects: "
        f"{raw_wide.shape[0]}",
        f"  Failed subjects: "
        f"{len(failed_rows)}",
        f"  Steps per V_all: {n_steps}",
        f"  ROIs per V_all: {n_roi}",
        f"  Raw matrix shape: "
        f"{raw_matrix.shape}",
        "",
        "ROI-set audit",
        f"  Uncovered ROIs: {n_uncovered}",
        f"  Overlapping ROIs: {n_overlapping}",
        "",
        "Raw ROI early-slope distribution",
        f"  mean: {np.mean(raw_values):.12g}",
        f"  sd: {np.std(raw_values):.12g}",
        f"  min: {np.min(raw_values):.12g}",
        f"  max: {np.max(raw_values):.12g}",
        "",
        "Scaled ROI early-slope distribution",
        f"  mean: {np.mean(scaled_values):.12g}",
        f"  sd: {np.std(scaled_values):.12g}",
        f"  min: {np.min(scaled_values):.12g}",
        f"  max: {np.max(scaled_values):.12g}",
        "",
        "System reconstruction",
        f"  Tolerance: "
        f"{RECONSTRUCTION_ABS_TOL}",
        "  Max |mean ROI slopes "
        "- direct V_all system slope|:",
        f"    {max_roi_mean_vs_direct}",
        "  Max |direct V_all system slope "
        "- saved raw system metric|:",
        f"    {max_direct_vs_saved}",
        "",
        "Pipeline note",
        "  Use the RAW wide CSV as the input to "
        "ROI-level ComBat.",
        "  Multiply ComBat-adjusted ROI slopes by "
        "1000 only after ComBat, matching the "
        "existing system-level pipeline.",
        "",
        "Outputs",
        f"  Raw wide CSV: {OUT_RAW_WIDE}",
        f"  Scaled wide CSV: {OUT_SCALED_WIDE}",
        f"  Long CSV: {OUT_LONG}",
        f"  Raw NPY: {OUT_RAW_NPY}",
        f"  Scaled NPY: {OUT_SCALED_NPY}",
        f"  Subject order: {OUT_SUBJECT_ORDER}",
        f"  Reconstruction detail: "
        f"{OUT_RECON_DETAIL}",
        f"  Reconstruction summary: "
        f"{OUT_RECON_SUMMARY}",
        f"  ROI dictionary: "
        f"{OUT_ROI_DICTIONARY}",
        f"  ROI-set audit: "
        f"{OUT_ROI_SET_AUDIT}",
        f"  Failed subjects: {OUT_FAILED}",
    ]

    write_text(
        OUT_QC,
        "\n".join(qc_lines),
    )

    # ========================================================
    # 13. FINAL CONSOLE OUTPUT
    # ========================================================

    print("")
    print("=" * 80)
    print(
        "ROI-level early-slope construction completed"
    )
    print("=" * 80)

    print(
        f"Successful subjects: "
        f"{raw_wide.shape[0]}"
    )

    print(
        f"Failed subjects: "
        f"{len(failed_rows)}"
    )

    print(
        f"ROIs: {n_roi}"
    )

    print(
        f"Raw ComBat input: "
        f"{OUT_RAW_WIDE}"
    )

    print(
        f"Scaled convenience output: "
        f"{OUT_SCALED_WIDE}"
    )

    print(
        f"Reconstruction QC: "
        f"{OUT_RECON_DETAIL}"
    )

    print(
        f"QC report: {OUT_QC}"
    )


if __name__ == "__main__":
    main()
