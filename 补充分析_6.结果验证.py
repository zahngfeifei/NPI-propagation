from __future__ import annotations

import importlib.util
import json
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
    AGGREGATE_EXPRESSION_PATH,
    EXPRESSION_OUTPUT_DIRECTORY,
    GENE_SET_OUTPUT_DIRECTORY,
    GENE_SET_NAMES,
    MATCHED_NULL_OUTPUT_DIRECTORY,
    PHENOTYPE_PATH,
    PRIMARY_HYPOTHESIS,
    RANKED_GSEA_OUTPUT_DIRECTORY,
    SCORE_OUTPUT_DIRECTORY,
    SENSITIVITY_OUTPUT_DIRECTORY,
    SPATIAL_CORRELATION_OUTPUT_DIRECTORY,
    SPATIAL_NULL_OUTPUT_DIRECTORY,
    SUMMARY_OUTPUT_DIRECTORY,
    VALIDATION_OUTPUT_DIRECTORY,
    empiricalTwoSidedP,
    loadConfig,
    residualizeValues,
    validateSystemResidualDefinition,
    writeJson,
)


def main() -> None:
    config = loadConfig()
    phenotypeFrame = pd.read_parquet(PHENOTYPE_PATH)
    aggregateExpressionFrame = pd.read_parquet(AGGREGATE_EXPRESSION_PATH)
    expressionQcPayload = json.loads(
        (EXPRESSION_OUTPUT_DIRECTORY / "expression-qc.json").read_text(
            encoding="utf-8"
        )
    )
    resultFrame = pd.read_csv(
        SUMMARY_OUTPUT_DIRECTORY / "all-correlation-results.csv"
    )
    primaryFrame = resultFrame.loc[resultFrame["primaryHypothesis"]].copy()
    geneSetSummaryFrame = pd.read_csv(
        GENE_SET_OUTPUT_DIRECTORY / "gene-set-summary.csv"
    ).set_index("geneSetName")
    matchingFrame = pd.read_csv(
        MATCHED_NULL_OUTPUT_DIRECTORY / "matching-qc-summary.csv"
    )
    fullScoreFrame = pd.read_parquet(
        SCORE_OUTPUT_DIRECTORY / "gene-set-scores-400.parquet"
    )
    selectedScoreFrame = pd.read_parquet(
        SCORE_OUTPUT_DIRECTORY / "gene-set-scores-selected-374.parquet"
    )
    spatialQcFrame = pd.read_csv(
        SPATIAL_NULL_OUTPUT_DIRECTORY / "spatial-null-qc.csv"
    )
    spatialMethodPayload = json.loads(
        (SPATIAL_NULL_OUTPUT_DIRECTORY / "spatial-null-method.json").read_text(
            encoding="utf-8"
        )
    )
    surrogateArchive = np.load(
        SPATIAL_NULL_OUTPUT_DIRECTORY / "phenotype-spatial-surrogates.npz"
    )
    matchedSetArchive = np.load(
        MATCHED_NULL_OUTPUT_DIRECTORY / "matched-gene-sets.npz"
    )
    spatialNullFrame = pd.read_parquet(
        SPATIAL_CORRELATION_OUTPUT_DIRECTORY / "spatial-null-correlations.parquet"
    )
    matchedNullFrame = pd.read_parquet(
        MATCHED_NULL_OUTPUT_DIRECTORY / "matched-gene-set-null-correlations.parquet"
    )
    sensitivityFrame = pd.read_csv(
        SENSITIVITY_OUTPUT_DIRECTORY / "leave-one-donor-out.csv"
    )
    singleDonorFrame = pd.read_csv(
        SENSITIVITY_OUTPUT_DIRECTORY / "single-donor-correlations.csv"
    )
    hemisphereFrame = pd.read_csv(
        SENSITIVITY_OUTPUT_DIRECTORY / "hemisphere-sensitivity.csv"
    )
    gseaFrame = pd.read_csv(
        RANKED_GSEA_OUTPUT_DIRECTORY / "ranked-gene-gsea-results.csv"
    )

    expectedRawCounts = {
        geneSetName: int(config["geneSets"][geneSetName]["expectedRawGeneCount"])
        for geneSetName in GENE_SET_NAMES
    }
    primaryIdentityExact = bool(
        len(primaryFrame) == 1
        and all(
            primaryFrame[fieldName].iloc[0] == expectedValue
            for fieldName, expectedValue in PRIMARY_HYPOTHESIS.items()
        )
    )
    residualDefinitionMatches = True
    try:
        validateSystemResidualDefinition(phenotypeFrame)
    except RuntimeError:
        residualDefinitionMatches = False

    primarySpatialNulls = spatialNullFrame.loc[
        spatialNullFrame["phenotypeName"].eq(PRIMARY_HYPOTHESIS["phenotypeName"])
        & spatialNullFrame["geneSetName"].eq(PRIMARY_HYPOTHESIS["geneSetName"])
        & spatialNullFrame["scoreMethod"].eq(PRIMARY_HYPOTHESIS["scoreMethod"]),
        "nullSpearmanR",
    ].to_numpy(dtype=float)
    primaryMatchedNulls = matchedNullFrame.loc[
        matchedNullFrame["phenotypeName"].eq(PRIMARY_HYPOTHESIS["phenotypeName"])
        & matchedNullFrame["geneSetName"].eq(PRIMARY_HYPOTHESIS["geneSetName"]),
        "nullSpearmanR",
    ].to_numpy(dtype=float)
    primaryPValueRecomputedExactly = False
    if len(primaryFrame) == 1:
        primaryObservedRho = float(primaryFrame["spearmanR"].iloc[0])
        recomputedSpatialP = empiricalTwoSidedP(
            primaryObservedRho, primarySpatialNulls
        )
        recomputedMatchedP = empiricalTwoSidedP(
            primaryObservedRho, primaryMatchedNulls
        )
        primaryPValueRecomputedExactly = bool(
            np.isclose(
                recomputedSpatialP,
                float(primaryFrame["pSpatial"].iloc[0]),
                atol=1e-15,
                rtol=0,
            )
            and np.isclose(
                recomputedMatchedP,
                float(primaryFrame["pGeneSet"].iloc[0]),
                atol=1e-15,
                rtol=0,
            )
        )
    else:
        recomputedSpatialP = np.nan
        recomputedMatchedP = np.nan

    noDuplicateGenesWithinMatchedSets = all(
        all(len(set(matchedSet.tolist())) == len(matchedSet) for matchedSet in archiveValues)
        for archiveValues in [
            matchedSetArchive[geneSetName] for geneSetName in GENE_SET_NAMES
        ]
    )
    primaryFiniteColumns = [
        "spearmanR",
        "pSpatial",
        "pGeneSet",
        "pConjunction",
        "spatialNullZ",
        "matchedNullZ",
    ]
    directSurrogatesAvailable = bool(
        set(surrogateArchive.files) == {"raw_beta", "system_residual"}
        and all(
            surrogateArchive[phenotypeName].shape
            == (int(config["statistics"]["spatialNullCount"]), len(phenotypeFrame))
            for phenotypeName in surrogateArchive.files
        )
        and spatialMethodPayload["systemResidualSurrogatesGeneratedDirectly"]
    )
    fixedSeedRecorded = bool(
        spatialMethodPayload["baseSeed"] == int(config["project"]["seed"])
        and matchingFrame["geneSetSeed"].notna().all()
        and matchingFrame["referenceGeneSeed"].nunique() == 1
    )

    checks = {
        "schaeferAtlasIs400Parcels7Networks2mm": bool(
            expressionQcPayload["parameters"]["atlasParcels"] == 400
            and expressionQcPayload["parameters"]["atlasNetworks"] == 7
            and expressionQcPayload["parameters"]["atlasResolutionMm"] == 2
        ),
        "aggregateExpressionRoiCountEquals400": bool(
            aggregateExpressionFrame["ROI_ID_1based"].nunique() == 400
            and len(aggregateExpressionFrame) == 400
        ),
        "exactGeneSetNames": set(resultFrame["geneSetName"]) == set(GENE_SET_NAMES),
        "rawGeneCounts": geneSetSummaryFrame["rawGeneCount"].astype(int).to_dict()
        == expectedRawCounts,
        "effectiveGeneCounts": geneSetSummaryFrame["effectiveGeneCount"].ge(10).all(),
        "totalResultCountEquals4": len(resultFrame) == 4,
        "onlyMeanZScoring": resultFrame["scoreMethod"].eq("mean_z").all(),
        "primaryHypothesisIdentityExact": primaryIdentityExact,
        "primaryHypothesisCountEqualsOne": len(primaryFrame) == 1,
        "selectedRoiCountEquals374": phenotypeFrame["ROI_ID_1based"].nunique()
        == int(config["project"]["expectedSelectedRois"])
        and selectedScoreFrame["ROI_ID_1based"].nunique()
        == int(config["project"]["expectedSelectedRois"]),
        "fullRoiCountEquals400": fullScoreFrame["ROI_ID_1based"].nunique()
        == int(config["project"]["expectedFullRois"]),
        "roiIdsUnique": phenotypeFrame["ROI_ID_1based"].is_unique,
        "allPrimaryValuesFinite": len(primaryFrame) == 1
        and np.isfinite(primaryFrame[primaryFiniteColumns].to_numpy(dtype=float)).all(),
        "systemResidualDefinitionMatches": residualDefinitionMatches,
        "directSystemResidualSurrogatesAvailable": directSurrogatesAvailable,
        "spatialNullCountEquals10000": resultFrame["spatialNullCount"]
        .eq(int(config["statistics"]["spatialNullCount"]))
        .all(),
        "matchedNullCountEquals10000": matchingFrame["acceptedMatchedSetCount"]
        .eq(int(config["statistics"]["matchedGeneSetNullCount"]))
        .all(),
        "variogramQcPassed": spatialQcFrame["variogramQcPassed"].all(),
        "moranQcPassed": spatialQcFrame["moranQcPassed"].all(),
        "matchingBalancePassed": matchingFrame["matchingQcPassed"].all()
        and matchingFrame["worstAcceptedAbsoluteSmd"]
        .le(
            float(
                config["statistics"][
                    "matchedMaxAbsoluteStandardizedMeanDifference"
                ]
            )
            + 1e-12
        )
        .all(),
        "primaryPValueRecomputedExactly": primaryPValueRecomputedExactly,
        "noDuplicateGenesWithinMatchedSets": noDuplicateGenesWithinMatchedSets,
        "fixedSeedRecorded": fixedSeedRecorded,
        "singleDonorAnalysisComplete": len(singleDonorFrame) == 6
        and singleDonorFrame["spearmanR"].notna().all(),
        "leaveOneDonorOutAnalysisComplete": len(sensitivityFrame) == 6
        and sensitivityFrame["spearmanR"].notna().all(),
        "hemisphereSensitivityComplete": len(hemisphereFrame) == 2
        and hemisphereFrame["spearmanR"].notna().all(),
        "rankedGeneGseaExploratoryOnly": len(gseaFrame) == 4
        and gseaFrame["calibrationStatus"]
        .eq("exploratory_not_spatially_calibrated")
        .all(),
    }
    observedResultMetrics = {
        "primaryResultStatus": primaryFrame["resultStatus"].iloc[0]
        if len(primaryFrame) == 1
        else None,
        "primarySpearmanR": float(primaryFrame["spearmanR"].iloc[0])
        if len(primaryFrame) == 1
        else np.nan,
        "primarySpatialP": float(primaryFrame["pSpatial"].iloc[0])
        if len(primaryFrame) == 1
        else np.nan,
        "primaryMatchedP": float(primaryFrame["pGeneSet"].iloc[0])
        if len(primaryFrame) == 1
        else np.nan,
        "singleDonorSignAgreementCount": int(
            singleDonorFrame["directionMatchesAggregate"].sum()
        ),
        "lodoSignAgreementCount": int(
            sensitivityFrame["directionMatchesAggregate"].sum()
        ),
        "hemisphereSignAgreementCount": int(
            hemisphereFrame["directionMatchesAggregate"].sum()
        ),
        "analysisStatusCounts": resultFrame["resultStatus"].value_counts().to_dict(),
    }
    validationPayload = {
        "overallPassed": all(bool(checkPassed) for checkPassed in checks.values()),
        "checks": {checkName: bool(checkPassed) for checkName, checkPassed in checks.items()},
        "metrics": {
            "totalResultCount": len(resultFrame),
            "primaryHypothesisCount": len(primaryFrame),
            "maximumWorstAcceptedAbsoluteSmd": float(
                matchingFrame["worstAcceptedAbsoluteSmd"].max()
            ),
            "recomputedPrimarySpatialP": recomputedSpatialP,
            "recomputedPrimaryMatchedP": recomputedMatchedP,
        },
        "observedResultMetrics": observedResultMetrics,
    }
    writeJson(VALIDATION_OUTPUT_DIRECTORY / "validation.json", validationPayload)
    if not validationPayload["overallPassed"]:
        failedChecks = [name for name, passed in checks.items() if not passed]
        raise RuntimeError(f"Validation failed: {failedChecks}")


if __name__ == "__main__":
    main()
