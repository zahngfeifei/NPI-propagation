from __future__ import annotations

import importlib.util
from pathlib import Path
import sys

import numpy as np
import pandas as pd

COMMON_SCRIPT_PATH = Path(__file__).resolve().parent / "补充分析_1.公共配置与函数.py"
commonModuleSpec = importlib.util.spec_from_file_location("analysis_common", COMMON_SCRIPT_PATH)
if commonModuleSpec is None or commonModuleSpec.loader is None:
    raise RuntimeError(f"无法加载公共配置：{COMMON_SCRIPT_PATH}")
geneAnalysisCommon = importlib.util.module_from_spec(commonModuleSpec)
sys.modules[commonModuleSpec.name] = geneAnalysisCommon
commonModuleSpec.loader.exec_module(geneAnalysisCommon)

from analysis_common import (
    ALPHA,
    MATCHED_NULL_OUTPUT_DIRECTORY,
    PRIMARY_HYPOTHESIS,
    SENSITIVITY_OUTPUT_DIRECTORY,
    SPATIAL_CORRELATION_OUTPUT_DIRECTORY,
    SUMMARY_OUTPUT_DIRECTORY,
    adjustBenjaminiHochberg,
)


def assignAnalysisRole(resultRow: pd.Series) -> str:
    if all(
        resultRow[fieldName] == expectedValue
        for fieldName, expectedValue in PRIMARY_HYPOTHESIS.items()
    ):
        return "primary"
    if (
        resultRow["phenotypeName"] == "raw_beta"
        and resultRow["geneSetName"]
        == "SFARI_high_confidence_nonsyndromic"
        and resultRow["scoreMethod"] == "mean_z"
    ):
        return "baseline"
    if (
        resultRow["phenotypeName"] == "system_residual"
        and resultRow["geneSetName"] == "SFARI_syndromic"
        and resultRow["scoreMethod"] == "mean_z"
    ):
        return "gene_set_validation"
    return "secondary"


def assignMultiplicityMetadata(analysisRole: str) -> tuple[str, str]:
    metadata = {
        "primary": (
            "primary_single_prespecified",
            "not applicable; single prespecified primary hypothesis",
        ),
        "gene_set_validation": (
            "validation_single_prespecified",
            "not applicable; single prespecified validation gene set",
        ),
        "baseline": ("baseline_descriptive", "descriptive; no multiplicity claim"),
        "secondary": ("secondary_exploratory", "descriptive secondary analysis"),
    }
    return metadata[analysisRole]


def assignResultStatus(resultRow: pd.Series) -> str:
    spatialPassed = pd.notna(resultRow["pSpatial"]) and resultRow["pSpatial"] < ALPHA
    matchedPassed = pd.notna(resultRow["pGeneSet"]) and resultRow["pGeneSet"] < ALPHA
    spatialQcPassed = bool(resultRow["spatialQcPassed"])
    matchingQcPassed = bool(resultRow.get("matchingQcPassed", False))
    if resultRow["analysisRole"] == "primary":
        return (
            "primary_supported"
            if spatialPassed
            and matchedPassed
            and spatialQcPassed
            and matchingQcPassed
            else "primary_not_supported"
        )
    if resultRow["analysisRole"] == "gene_set_validation":
        if spatialPassed and matchedPassed and spatialQcPassed and matchingQcPassed:
            return "validation_supported"
        if spatialPassed or matchedPassed:
            return "partial_validation"
        return "not_validated"
    if resultRow["analysisRole"] == "baseline":
        return "descriptive_baseline"
    return "secondary_descriptive"


def main() -> None:
    spatialResultFrame = pd.read_csv(
        SPATIAL_CORRELATION_OUTPUT_DIRECTORY / "spatial-correlation-results.csv"
    )
    matchedResultFrame = pd.read_csv(
        MATCHED_NULL_OUTPUT_DIRECTORY / "matched-gene-set-results.csv"
    ).rename(
        columns={
            "testTail": "matchedTestTail",
            "pFormula": "matchedPFormula",
            "alpha": "matchedAlpha",
        }
    )
    combinedFrame = spatialResultFrame.merge(
        matchedResultFrame,
        on=["phenotypeName", "geneSetName", "scoreMethod"],
        how="left",
    )
    combinedFrame["analysisRole"] = combinedFrame.apply(assignAnalysisRole, axis=1)
    multiplicityMetadata = combinedFrame["analysisRole"].map(
        assignMultiplicityMetadata
    )
    combinedFrame["multiplicityFamily"] = multiplicityMetadata.map(
        lambda value: value[0]
    )
    combinedFrame["multiplicityAdjustment"] = multiplicityMetadata.map(
        lambda value: value[1]
    )
    combinedFrame["primaryHypothesis"] = combinedFrame.apply(
        lambda resultRow: all(
            resultRow[fieldName] == expectedValue
            for fieldName, expectedValue in PRIMARY_HYPOTHESIS.items()
        ),
        axis=1,
    )
    combinedFrame["pConjunction"] = np.where(
        combinedFrame[["pSpatial", "pGeneSet"]].notna().all(axis=1),
        combinedFrame[["pSpatial", "pGeneSet"]].max(axis=1),
        np.nan,
    )
    combinedFrame["qSpatialAcrossTwoSfariSets"] = np.nan
    combinedFrame["qGeneSetAcrossTwoSfariSets"] = np.nan
    meanZMask = combinedFrame["scoreMethod"].eq("mean_z")
    for phenotypeName, groupIndices in combinedFrame.loc[meanZMask].groupby(
        "phenotypeName", sort=True
    ).groups.items():
        selectedIndices = list(groupIndices)
        combinedFrame.loc[
            selectedIndices, "qSpatialAcrossTwoSfariSets"
        ] = adjustBenjaminiHochberg(
            combinedFrame.loc[selectedIndices, "pSpatial"]
        )
        combinedFrame.loc[
            selectedIndices, "qGeneSetAcrossTwoSfariSets"
        ] = adjustBenjaminiHochberg(
            combinedFrame.loc[selectedIndices, "pGeneSet"]
        )

    combinedFrame["resultStatus"] = combinedFrame.apply(assignResultStatus, axis=1)
    combinedFrame["primaryDecision"] = np.where(
        combinedFrame["primaryHypothesis"], combinedFrame["resultStatus"], np.nan
    )

    stabilityFrame = pd.read_csv(
        SENSITIVITY_OUTPUT_DIRECTORY / "primary-stability-summary.csv"
    )
    stabilityColumns = [
        columnName
        for columnName in stabilityFrame.columns
        if columnName not in {"phenotypeName", "geneSetName", "scoreMethod"}
    ]
    for columnName in stabilityColumns:
        combinedFrame[columnName] = np.nan
    primaryMask = combinedFrame["primaryHypothesis"]
    for columnName in stabilityColumns:
        combinedFrame.loc[primaryMask, columnName] = stabilityFrame[columnName].iloc[0]

    outputDirectory = SUMMARY_OUTPUT_DIRECTORY
    outputDirectory.mkdir(parents=True, exist_ok=True)
    combinedFrame.to_csv(
        outputDirectory / "all-correlation-results.csv",
        index=False,
        encoding="utf-8-sig",
    )
    combinedFrame.loc[combinedFrame["analysisRole"].eq("primary")].to_csv(
        outputDirectory / "primary-result.csv", index=False, encoding="utf-8-sig"
    )
    combinedFrame.loc[combinedFrame["analysisRole"].eq("baseline")].to_csv(
        outputDirectory / "baseline-result.csv", index=False, encoding="utf-8-sig"
    )
    combinedFrame.loc[
        combinedFrame["analysisRole"].eq("gene_set_validation")
    ].to_csv(
        outputDirectory / "gene-set-validation-result.csv",
        index=False,
        encoding="utf-8-sig",
    )
    combinedFrame["resultStatus"].value_counts().rename_axis(
        "resultStatus"
    ).reset_index(name="resultCount").to_csv(
        outputDirectory / "analysis-status-counts.csv",
        index=False,
        encoding="utf-8-sig",
    )


if __name__ == "__main__":
    main()
