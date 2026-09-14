# -*- coding: utf-8 -*-
"""
Gene analysis — Step 6
Build the AHBA/PLS phenotype table from the Step-4 HC-minus-ASD ROI beta map.

Input
-----
Step-4 group map:
    h1-h4-group-beta-hc-minus-asd-map.csv

Required columns
----------------
selected_output_position_0based
roi_index_0based
ROI_ID_1based
system
beta_HC_minus_ASD

Outputs
-------
1. phenotypes.parquet
2. phenotype-bridge-qc.json

Phenotype columns
-----------------
raw_beta:
    Continuous unthresholded HC-minus-ASD ROI beta from Step 4.

system_residual:
    raw_beta after removing the four H1--H4 system means.

raw_beta_z:
    Z-scored raw_beta across selected ROIs.

system_residual_z:
    Z-scored system_residual across selected ROIs.

Important
---------
- Only H1--H4-covered ROIs are included.
- Uncovered Schaefer400 ROIs are not inserted and are not assigned zero.
- The ROI order is inherited from selected_output_position_0based.
"""

from __future__ import annotations

import json
from pathlib import Path

import numpy as np
import pandas as pd


PROJECT_ROOT = Path(__file__).resolve().parent


# ============================================================
# 1. PATH SETTINGS — ALL ABSOLUTE PATHS
# ============================================================

# Step 4 primary continuous HC-minus-ASD ROI beta map.
GROUP_MAP_PATH = Path(
    PROJECT_ROOT
    / "ABIDE2_主流程结果4_ROI组间效应"
    / "h1-h4-group-beta-hc-minus-asd-map.csv"
)

# AHBA / PLS analysis root.
GENE_ANALYSIS_ROOT = Path(
    PROJECT_ROOT / "ABIDE2_主流程结果6_AHBA表型"
)

# Main phenotype input for downstream AHBA / PLS analysis.
PHENOTYPE_OUTPUT_PATH = (
    GENE_ANALYSIS_ROOT
    / "ABIDE2_H1H4_phenotypes.parquet"
)

# Bridge QC output.
BRIDGE_QC_PATH = (
    GENE_ANALYSIS_ROOT
    / "phenotype-bridge-qc.json"
)


# ============================================================
# 2. ANALYSIS SETTINGS
# ============================================================

COHORT = "ABIDE2"

# Expected number of H1--H4-covered ROIs.
# Keep this fixed to detect accidental ROI loss or inclusion.
EXPECTED_ROI_COUNT = 374

SYSTEM_ORDER = [
    "H1_sensory",
    "H2_attention",
    "H3_control",
    "H4_DMN",
]

SYSTEM_RESIDUAL_MEAN_TOLERANCE = 1e-10
MIN_STANDARD_DEVIATION = 1e-12


# ============================================================
# 3. UTILITIES
# ============================================================

def cleanColumns(table: pd.DataFrame) -> pd.DataFrame:
    """Trim whitespace from column names."""
    table = table.copy()
    table.columns = [
        str(column).strip()
        for column in table.columns
    ]
    return table


def validateInputPath() -> None:
    """Check that the Step-4 group map exists."""
    if not GROUP_MAP_PATH.exists():
        raise FileNotFoundError(
            "Step-4 group map was not found:\n"
            f"  {GROUP_MAP_PATH}"
        )


def buildPhenotypeFrame(
    groupMapFrame: pd.DataFrame,
) -> pd.DataFrame:
    """
    Construct the downstream AHBA/PLS phenotype table.

    The raw HC-minus-ASD beta map is retained, and a second phenotype is
    created after removing the mean effect of the four hierarchy systems.
    """
    groupMapFrame = cleanColumns(
        groupMapFrame
    )

    requiredColumns = (
        "selected_output_position_0based",
        "roi_index_0based",
        "ROI_ID_1based",
        "system",
        "beta_HC_minus_ASD",
    )

    missingColumns = set(
        requiredColumns
    ).difference(
        groupMapFrame.columns
    )

    if missingColumns:
        raise ValueError(
            "Group map is missing required columns: "
            f"{sorted(missingColumns)}. "
            f"Available columns: "
            f"{groupMapFrame.columns.tolist()}"
        )

    phenotypeFrame = (
        groupMapFrame
        .loc[:, requiredColumns]
        .copy()
    )

    for column in [
        "selected_output_position_0based",
        "roi_index_0based",
        "ROI_ID_1based",
    ]:
        phenotypeFrame[column] = pd.to_numeric(
            phenotypeFrame[column],
            errors="raise",
        ).astype(int)

    phenotypeFrame["system"] = (
        phenotypeFrame["system"]
        .astype(str)
        .str.strip()
    )

    phenotypeFrame[
        "beta_HC_minus_ASD"
    ] = pd.to_numeric(
        phenotypeFrame[
            "beta_HC_minus_ASD"
        ],
        errors="coerce",
    )

    if phenotypeFrame[
        "beta_HC_minus_ASD"
    ].isna().any():
        badCount = int(
            phenotypeFrame[
                "beta_HC_minus_ASD"
            ].isna().sum()
        )
        raise ValueError(
            "beta_HC_minus_ASD contains "
            f"{badCount} missing/non-numeric values."
        )

    if not np.isfinite(
        phenotypeFrame[
            "beta_HC_minus_ASD"
        ].to_numpy(dtype=float)
    ).all():
        raise ValueError(
            "beta_HC_minus_ASD contains NaN or Inf."
        )

    phenotypeFrame = (
        phenotypeFrame
        .sort_values(
            "selected_output_position_0based"
        )
        .reset_index(drop=True)
    )

    if len(phenotypeFrame) != EXPECTED_ROI_COUNT:
        raise ValueError(
            f"Expected {EXPECTED_ROI_COUNT} selected ROIs, "
            f"found {len(phenotypeFrame)}."
        )

    expectedPositions = list(
        range(
            len(phenotypeFrame)
        )
    )

    actualPositions = phenotypeFrame[
        "selected_output_position_0based"
    ].tolist()

    if actualPositions != expectedPositions:
        raise ValueError(
            "selected_output_position_0based must be "
            "a complete consecutive sequence beginning at 0."
        )

    if phenotypeFrame[
        "roi_index_0based"
    ].duplicated().any():
        duplicatedRois = (
            phenotypeFrame.loc[
                phenotypeFrame[
                    "roi_index_0based"
                ].duplicated(
                    keep=False
                ),
                "roi_index_0based",
            ]
            .drop_duplicates()
            .head(10)
            .tolist()
        )
        raise ValueError(
            "Duplicate ROI indices were found. "
            f"Examples: {duplicatedRois}"
        )

    expectedRoiId = (
        phenotypeFrame[
            "roi_index_0based"
        ]
        + 1
    )

    if not np.array_equal(
        expectedRoiId.to_numpy(),
        phenotypeFrame[
            "ROI_ID_1based"
        ].to_numpy(),
    ):
        raise ValueError(
            "ROI_ID_1based is inconsistent with "
            "roi_index_0based + 1."
        )

    unexpectedSystems = sorted(
        set(
            phenotypeFrame[
                "system"
            ]
        )
        - set(
            SYSTEM_ORDER
        )
    )

    if unexpectedSystems:
        raise ValueError(
            "Unexpected hierarchy labels were found: "
            f"{unexpectedSystems}"
        )

    missingSystems = [
        system
        for system in SYSTEM_ORDER
        if system not in set(
            phenotypeFrame[
                "system"
            ]
        )
    ]

    if missingSystems:
        raise ValueError(
            "The following expected hierarchy systems "
            f"are missing: {missingSystems}"
        )

    phenotypeFrame[
        "hemisphere"
    ] = np.where(
        phenotypeFrame[
            "ROI_ID_1based"
        ].astype(int)
        <= 200,
        "LH",
        "RH",
    )

    phenotypeFrame[
        "raw_beta"
    ] = phenotypeFrame[
        "beta_HC_minus_ASD"
    ].astype(float)

    # Remove the mean of each hierarchy system.
    systemDesignFrame = pd.get_dummies(
        phenotypeFrame[
            "system"
        ].astype(str),
        prefix="system",
        drop_first=False,
        dtype=float,
    )

    # Force a stable and explicit column order.
    expectedSystemDummyColumns = [
        f"system_{system}"
        for system in SYSTEM_ORDER
    ]

    systemDesignFrame = (
        systemDesignFrame
        .reindex(
            columns=expectedSystemDummyColumns,
            fill_value=0.0,
        )
    )

    systemDesign = systemDesignFrame.to_numpy(
        dtype=float
    )

    rawBeta = phenotypeFrame[
        "raw_beta"
    ].to_numpy(
        dtype=float
    )

    systemEffects = (
        np.linalg.pinv(
            systemDesign
        )
        @ rawBeta
    )

    fittedSystemMeans = (
        systemDesign
        @ systemEffects
    )

    phenotypeFrame[
        "system_residual"
    ] = (
        rawBeta
        - fittedSystemMeans
    )

    rawBetaSd = float(
        phenotypeFrame[
            "raw_beta"
        ].std(
            ddof=1
        )
    )

    residualSd = float(
        phenotypeFrame[
            "system_residual"
        ].std(
            ddof=1
        )
    )

    if (
        not np.isfinite(rawBetaSd)
        or rawBetaSd
        <= MIN_STANDARD_DEVIATION
    ):
        raise ValueError(
            "raw_beta has zero or near-zero standard deviation."
        )

    if (
        not np.isfinite(residualSd)
        or residualSd
        <= MIN_STANDARD_DEVIATION
    ):
        raise ValueError(
            "system_residual has zero or near-zero "
            "standard deviation."
        )

    phenotypeFrame[
        "raw_beta_z"
    ] = (
        phenotypeFrame[
            "raw_beta"
        ]
        - phenotypeFrame[
            "raw_beta"
        ].mean()
    ) / rawBetaSd

    phenotypeFrame[
        "system_residual_z"
    ] = (
        phenotypeFrame[
            "system_residual"
        ]
        - phenotypeFrame[
            "system_residual"
        ].mean()
    ) / residualSd

    return phenotypeFrame


def savePhenotypeParquet(
    phenotypeFrame: pd.DataFrame,
) -> None:
    """
    Save the phenotype table as Parquet.

    pandas requires pyarrow or fastparquet.
    """
    PHENOTYPE_OUTPUT_PATH.parent.mkdir(
        parents=True,
        exist_ok=True,
    )

    try:
        phenotypeFrame.to_parquet(
            PHENOTYPE_OUTPUT_PATH,
            index=False,
        )
    except ImportError as error:
        raise ImportError(
            "Saving Parquet requires pyarrow or fastparquet. "
            "Install pyarrow with:\n"
            "    pip install pyarrow"
        ) from error


def buildBridgeQc(
    phenotypeFrame: pd.DataFrame,
) -> dict:
    """
    Build QC information for the Step-4 to AHBA phenotype bridge.
    """
    systemResidualMeans = (
        phenotypeFrame
        .groupby(
            "system",
            sort=False,
        )[
            "system_residual"
        ]
        .mean()
    )

    maximumAbsoluteSystemMean = float(
        systemResidualMeans
        .abs()
        .max()
    )

    systemCounts = (
        phenotypeFrame[
            "system"
        ]
        .value_counts()
        .reindex(
            SYSTEM_ORDER,
            fill_value=0,
        )
    )

    bridgeQc = {
        "cohort": COHORT,
        "sourceGroupMap": str(
            GROUP_MAP_PATH
        ),
        "phenotypeOutput": str(
            PHENOTYPE_OUTPUT_PATH
        ),
        "bridgeQcOutput": str(
            BRIDGE_QC_PATH
        ),
        "roiCount": int(
            len(
                phenotypeFrame
            )
        ),
        "expectedRoiCount": int(
            EXPECTED_ROI_COUNT
        ),
        "systemCounts": {
            system: int(
                systemCounts[
                    system
                ]
            )
            for system in SYSTEM_ORDER
        },
        "rawBetaMean": float(
            phenotypeFrame[
                "raw_beta"
            ].mean()
        ),
        "rawBetaStandardDeviation": float(
            phenotypeFrame[
                "raw_beta"
            ].std(
                ddof=1
            )
        ),
        "systemResidualMean": float(
            phenotypeFrame[
                "system_residual"
            ].mean()
        ),
        "systemResidualStandardDeviation": float(
            phenotypeFrame[
                "system_residual"
            ].std(
                ddof=1
            )
        ),
        "maximumAbsoluteSystemResidualMean":
            maximumAbsoluteSystemMean,
        "systemResidualMeanTolerance":
            SYSTEM_RESIDUAL_MEAN_TOLERANCE,
        "allValuesFinite": bool(
            np.isfinite(
                phenotypeFrame[
                    [
                        "raw_beta",
                        "system_residual",
                        "raw_beta_z",
                        "system_residual_z",
                    ]
                ].to_numpy(
                    dtype=float
                )
            ).all()
        ),
        "passed": bool(
            maximumAbsoluteSystemMean
            < SYSTEM_RESIDUAL_MEAN_TOLERANCE
        ),
    }

    return bridgeQc


def saveBridgeQc(
    bridgeQc: dict,
) -> None:
    """Save QC JSON."""
    BRIDGE_QC_PATH.parent.mkdir(
        parents=True,
        exist_ok=True,
    )

    BRIDGE_QC_PATH.write_text(
        json.dumps(
            bridgeQc,
            ensure_ascii=False,
            indent=2,
        ),
        encoding="utf-8",
    )


# ============================================================
# 4. MAIN
# ============================================================

def main() -> None:
    print("=" * 80)
    print(
        "Building AHBA/PLS phenotype table "
        "from the Step-4 HC-minus-ASD ROI beta map"
    )
    print("=" * 80)

    validateInputPath()

    print(
        f"Step-4 group map: "
        f"{GROUP_MAP_PATH}"
    )
    print(
        f"Phenotype output: "
        f"{PHENOTYPE_OUTPUT_PATH}"
    )
    print(
        f"Bridge QC output: "
        f"{BRIDGE_QC_PATH}"
    )

    groupMapFrame = pd.read_csv(
        GROUP_MAP_PATH,
        low_memory=False,
        encoding="utf-8-sig",
    )

    phenotypeFrame = buildPhenotypeFrame(
        groupMapFrame
    )

    savePhenotypeParquet(
        phenotypeFrame
    )

    bridgeQc = buildBridgeQc(
        phenotypeFrame
    )

    saveBridgeQc(
        bridgeQc
    )

    if not bridgeQc[
        "allValuesFinite"
    ]:
        raise RuntimeError(
            "Phenotype table contains non-finite values."
        )

    if not bridgeQc[
        "passed"
    ]:
        raise RuntimeError(
            "System residualization QC failed. "
            "The system-specific residual means exceed "
            f"{SYSTEM_RESIDUAL_MEAN_TOLERANCE}."
        )

    print("")
    print("=" * 80)
    print(
        "AHBA/PLS phenotype construction completed"
    )
    print("=" * 80)
    print(
        f"Selected ROIs: "
        f"{len(phenotypeFrame)}"
    )
    print(
        "Maximum absolute system residual mean: "
        f"{bridgeQc['maximumAbsoluteSystemResidualMean']:.16g}"
    )
    print(
        f"Phenotype table: "
        f"{PHENOTYPE_OUTPUT_PATH}"
    )
    print(
        f"QC JSON: "
        f"{BRIDGE_QC_PATH}"
    )


if __name__ == "__main__":
    main()
