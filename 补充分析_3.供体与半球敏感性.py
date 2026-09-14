from __future__ import annotations

import importlib.util
from pathlib import Path
import sys

import numpy as np
import pandas as pd
from scipy import stats

COMMON_SCRIPT_PATH = Path(__file__).resolve().parent / "补充分析_1.公共配置与函数.py"
commonModuleSpec = importlib.util.spec_from_file_location("analysis_common", COMMON_SCRIPT_PATH)
if commonModuleSpec is None or commonModuleSpec.loader is None:
    raise RuntimeError(f"无法加载公共配置：{COMMON_SCRIPT_PATH}")
geneAnalysisCommon = importlib.util.module_from_spec(commonModuleSpec)
sys.modules[commonModuleSpec.name] = geneAnalysisCommon
commonModuleSpec.loader.exec_module(geneAnalysisCommon)

from analysis_common import (
    AGGREGATE_EXPRESSION_PATH,
    DONOR_EXPRESSION_MANIFEST_PATH,
    GENE_SET_OUTPUT_DIRECTORY,
    PHENOTYPE_SPECS,
    PHENOTYPE_PATH,
    PRIMARY_HYPOTHESIS,
    SENSITIVITY_OUTPUT_DIRECTORY,
    normalizeExpressionColumns,
    residualizeValues,
    zscoreColumns,
)


def residualizeWithAvailable(
    scoreValues: np.ndarray, phenotypeFrame: pd.DataFrame, adjustment: str
) -> np.ndarray:
    scoreArray = np.asarray(scoreValues, dtype=float)
    validMask = np.isfinite(scoreArray)
    adjustedValues = np.full(len(scoreArray), np.nan, dtype=float)
    if validMask.sum() < 10:
        return adjustedValues
    adjustedValues[validMask] = residualizeValues(
        scoreArray[validMask],
        phenotypeFrame.loc[validMask].reset_index(drop=True),
        adjustment,
    )
    return adjustedValues


def spearmanWithAvailable(scoreValues: np.ndarray, phenotypeValues: np.ndarray) -> float:
    validMask = np.isfinite(scoreValues) & np.isfinite(phenotypeValues)
    if validMask.sum() < 10:
        return np.nan
    return float(
        stats.spearmanr(
            scoreValues[validMask], phenotypeValues[validMask]
        ).statistic
    )


def summarizeCorrelationRows(
    correlationFrame: pd.DataFrame,
    correlationColumn: str,
    aggregateCorrelation: float,
    prefix: str,
) -> dict[str, float | int]:
    correlationValues = correlationFrame[correlationColumn].to_numpy(dtype=float)
    signAgreementCount = int(
        np.sum(np.sign(correlationValues) == np.sign(aggregateCorrelation))
    )
    return {
        f"{prefix}CorrelationCount": len(correlationValues),
        f"{prefix}SignAgreementCount": signAgreementCount,
        f"{prefix}SignAgreementFraction": signAgreementCount
        / len(correlationValues),
        f"{prefix}MedianRho": float(np.median(correlationValues)),
        f"{prefix}MinimumRho": float(np.min(correlationValues)),
        f"{prefix}MaximumRho": float(np.max(correlationValues)),
        f"{prefix}MaximumAbsoluteDeltaFromAggregate": float(
            np.max(np.abs(correlationValues - aggregateCorrelation))
        ),
    }


def main() -> None:
    phenotypeName = PRIMARY_HYPOTHESIS["phenotypeName"]
    geneSetName = PRIMARY_HYPOTHESIS["geneSetName"]
    phenotypeSpecification = next(
        specification
        for specification in PHENOTYPE_SPECS
        if specification["phenotypeName"] == phenotypeName
    )
    adjustment = phenotypeSpecification["scoreAdjustment"]
    phenotypeFrame = pd.read_parquet(PHENOTYPE_PATH)
    selectedRoiIds = phenotypeFrame["ROI_ID_1based"].astype(int).tolist()
    phenotypeValues = phenotypeFrame[phenotypeName].to_numpy(dtype=float)
    geneSetFrame = pd.read_csv(
        GENE_SET_OUTPUT_DIRECTORY / "normalized-gene-sets.csv"
    )
    geneSymbols = sorted(
        geneSetFrame.loc[
            geneSetFrame["geneSetName"].eq(geneSetName)
            & geneSetFrame["presentInAhbaExpression"],
            "geneSymbol",
        ]
        .astype(str)
        .str.upper()
        .tolist()
    )

    aggregateExpressionFrame = pd.read_parquet(AGGREGATE_EXPRESSION_PATH).set_index(
        "ROI_ID_1based"
    )
    aggregateExpressionFrame = normalizeExpressionColumns(aggregateExpressionFrame)[
        geneSymbols
    ]
    aggregateStandardizedFrame = zscoreColumns(aggregateExpressionFrame).loc[
        selectedRoiIds
    ]
    aggregateScoreValues = aggregateStandardizedFrame.mean(axis=1).to_numpy(dtype=float)
    adjustedAggregateScores = residualizeWithAvailable(
        aggregateScoreValues, phenotypeFrame, adjustment
    )
    aggregateCorrelation = spearmanWithAvailable(
        adjustedAggregateScores, phenotypeValues
    )

    donorManifestFrame = pd.read_csv(DONOR_EXPRESSION_MANIFEST_PATH)
    donorExpressionArrays = []
    donorFrames = []
    donorKeys = []
    for donorRow in donorManifestFrame.itertuples(index=False):
        donorFrame = pd.read_parquet(Path(str(donorRow.outputPath))).set_index(
            "ROI_ID_1based"
        )
        donorFrame = normalizeExpressionColumns(donorFrame).reindex(
            index=range(1, 401), columns=geneSymbols
        )
        donorFrames.append(donorFrame)
        donorExpressionArrays.append(donorFrame.to_numpy(dtype=np.float32))
        donorKeys.append(str(donorRow.donorKey))

    singleDonorRows = []
    for donorKey, donorFrame in zip(donorKeys, donorFrames):
        donorStandardizedFrame = zscoreColumns(donorFrame).loc[selectedRoiIds]
        donorScores = donorStandardizedFrame.mean(axis=1).to_numpy(dtype=float)
        adjustedDonorScores = residualizeWithAvailable(
            donorScores, phenotypeFrame, adjustment
        )
        donorCorrelation = spearmanWithAvailable(
            adjustedDonorScores, phenotypeValues
        )
        singleDonorRows.append(
            {
                "donorKey": str(donorKey),
                "phenotypeName": phenotypeName,
                "geneSetName": geneSetName,
                "scoreMethod": "mean_z",
                "spearmanR": donorCorrelation,
                "aggregateSpearmanR": aggregateCorrelation,
                "directionMatchesAggregate": bool(
                    np.sign(donorCorrelation) == np.sign(aggregateCorrelation)
                ),
            }
        )
    singleDonorFrame = pd.DataFrame(singleDonorRows)

    donorExpressionValues = np.stack(donorExpressionArrays, axis=0)

    leaveOneDonorOutRows = []
    for excludedDonorIndex, excludedDonorKey in enumerate(donorKeys):
        retainedDonorValues = np.delete(
            donorExpressionValues, excludedDonorIndex, axis=0
        )
        availableDonorCounts = np.sum(np.isfinite(retainedDonorValues), axis=0)
        leaveOneOutValues = np.divide(
            np.nansum(retainedDonorValues, axis=0),
            availableDonorCounts,
            out=np.full(retainedDonorValues.shape[1:], np.nan, dtype=np.float32),
            where=availableDonorCounts > 0,
        )
        leaveOneOutFrame = pd.DataFrame(
            leaveOneOutValues, index=range(1, 401), columns=geneSymbols
        )
        leaveOneOutStandardizedFrame = zscoreColumns(leaveOneOutFrame).loc[
            selectedRoiIds
        ]
        leaveOneOutScores = leaveOneOutStandardizedFrame.mean(axis=1).to_numpy(
            dtype=float
        )
        adjustedLeaveOneOutScores = residualizeWithAvailable(
            leaveOneOutScores, phenotypeFrame, adjustment
        )
        leaveOneOutCorrelation = spearmanWithAvailable(
            adjustedLeaveOneOutScores, phenotypeValues
        )
        leaveOneDonorOutRows.append(
            {
                "excludedDonorKey": excludedDonorKey,
                "phenotypeName": phenotypeName,
                "geneSetName": geneSetName,
                "scoreMethod": "mean_z",
                "spearmanR": leaveOneOutCorrelation,
                "aggregateSpearmanR": aggregateCorrelation,
                "directionMatchesAggregate": bool(
                    np.sign(leaveOneOutCorrelation) == np.sign(aggregateCorrelation)
                ),
            }
        )
    leaveOneDonorOutFrame = pd.DataFrame(leaveOneDonorOutRows)

    hemisphereRows = []
    for hemisphereName in ["LH", "RH"]:
        hemisphereMask = phenotypeFrame["hemisphere"].eq(hemisphereName).to_numpy()
        hemisphereCorrelation = spearmanWithAvailable(
            adjustedAggregateScores[hemisphereMask], phenotypeValues[hemisphereMask]
        )
        hemisphereRows.append(
            {
                "hemisphere": hemisphereName,
                "phenotypeName": phenotypeName,
                "geneSetName": geneSetName,
                "scoreMethod": "mean_z",
                "roiCount": int(hemisphereMask.sum()),
                "spearmanR": hemisphereCorrelation,
                "aggregateSpearmanR": aggregateCorrelation,
                "directionMatchesAggregate": bool(
                    np.sign(hemisphereCorrelation) == np.sign(aggregateCorrelation)
                ),
                "inferenceStatus": "direction_sensitivity_no_hemisphere_spatial_null",
            }
        )
    hemisphereFrame = pd.DataFrame(hemisphereRows)

    summaryPayload = {
        "phenotypeName": phenotypeName,
        "geneSetName": geneSetName,
        "scoreMethod": "mean_z",
        "aggregateSpearmanR": aggregateCorrelation,
        **summarizeCorrelationRows(
            singleDonorFrame,
            "spearmanR",
            aggregateCorrelation,
            "singleDonor",
        ),
        **summarizeCorrelationRows(
            leaveOneDonorOutFrame,
            "spearmanR",
            aggregateCorrelation,
            "lodo",
        ),
        "leftHemisphereRho": float(
            hemisphereFrame.loc[
                hemisphereFrame["hemisphere"].eq("LH"), "spearmanR"
            ].iloc[0]
        ),
        "rightHemisphereRho": float(
            hemisphereFrame.loc[
                hemisphereFrame["hemisphere"].eq("RH"), "spearmanR"
            ].iloc[0]
        ),
        "hemisphereSignAgreementCount": int(
            hemisphereFrame["directionMatchesAggregate"].sum()
        ),
    }
    outputDirectory = SENSITIVITY_OUTPUT_DIRECTORY
    outputDirectory.mkdir(parents=True, exist_ok=True)
    singleDonorFrame.to_csv(
        outputDirectory / "single-donor-correlations.csv",
        index=False,
        encoding="utf-8-sig",
    )
    leaveOneDonorOutFrame.to_csv(
        outputDirectory / "leave-one-donor-out.csv",
        index=False,
        encoding="utf-8-sig",
    )
    hemisphereFrame.to_csv(
        outputDirectory / "hemisphere-sensitivity.csv",
        index=False,
        encoding="utf-8-sig",
    )
    pd.DataFrame([summaryPayload]).to_csv(
        outputDirectory / "primary-stability-summary.csv",
        index=False,
        encoding="utf-8-sig",
    )
    # Compatibility summary with explicitly named LODO fields.
    pd.DataFrame([summaryPayload]).to_csv(
        outputDirectory / "leave-one-donor-out-summary.csv",
        index=False,
        encoding="utf-8-sig",
    )


if __name__ == "__main__":
    main()
