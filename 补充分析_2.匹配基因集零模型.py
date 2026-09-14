from __future__ import annotations

import importlib.util
from pathlib import Path
import sys

import numpy as np
import pandas as pd
from scipy import stats
from sklearn.neighbors import NearestNeighbors
from sklearn.preprocessing import StandardScaler

COMMON_SCRIPT_PATH = Path(__file__).resolve().parent / "补充分析_1.公共配置与函数.py"
commonModuleSpec = importlib.util.spec_from_file_location("analysis_common", COMMON_SCRIPT_PATH)
if commonModuleSpec is None or commonModuleSpec.loader is None:
    raise RuntimeError(f"无法加载公共配置：{COMMON_SCRIPT_PATH}")
geneAnalysisCommon = importlib.util.module_from_spec(commonModuleSpec)
sys.modules[commonModuleSpec.name] = geneAnalysisCommon
commonModuleSpec.loader.exec_module(geneAnalysisCommon)

from analysis_common import (
    AGGREGATE_EXPRESSION_PATH,
    ALPHA,
    EMPIRICAL_P_FORMULA,
    GENE_DIFFERENTIAL_STABILITY_PATH,
    GENE_SET_NAMES,
    GENE_SET_OUTPUT_DIRECTORY,
    MATCHED_NULL_OUTPUT_DIRECTORY,
    PHENOTYPE_SPECS,
    PHENOTYPE_PATH,
    ROI_CENTROID_PATH,
    SPATIAL_CORRELATION_OUTPUT_DIRECTORY,
    TEST_TAIL,
    buildHemisphereSpatialWeights,
    empiricalTwoSidedP,
    loadConfig,
    normalizeExpressionColumns,
    residualizeRows,
    stableSeed,
    standardizedMeanDifference,
    summarizeNullDistribution,
    zscoreColumns,
)


MATCHING_FEATURE_COLUMNS = [
    "expressionMean400",
    "expressionSd400",
    "expressionMad400",
    "moranI374",
    "meanAbsoluteCorrelation374",
    "differentialStability",
]


def calculateMoranForGenes(
    expressionValues: np.ndarray, spatialWeights
) -> np.ndarray:
    centeredValues = expressionValues - np.nanmean(
        expressionValues, axis=0, keepdims=True
    )
    numerator = np.nansum(
        centeredValues * spatialWeights.dot(centeredValues), axis=0
    )
    denominator = np.nansum(centeredValues**2, axis=0)
    return (
        expressionValues.shape[0] / float(spatialWeights.sum())
    ) * numerator / denominator


def selectFixedReferenceGenes(
    geneSymbols: list[str], referenceGeneCount: int, seed: int
) -> list[str]:
    sortedGeneSymbols = sorted(geneSymbols)
    randomGenerator = np.random.default_rng(seed)
    selectedCount = min(referenceGeneCount, len(sortedGeneSymbols))
    selectedIndices = np.sort(
        randomGenerator.choice(len(sortedGeneSymbols), selectedCount, replace=False)
    )
    return [sortedGeneSymbols[index] for index in selectedIndices]


def calculateMeanAbsoluteCorrelation(
    expressionFrame: pd.DataFrame, referenceGenes: list[str]
) -> np.ndarray:
    expressionValues = expressionFrame.to_numpy(dtype=float)
    centeredValues = expressionValues - np.mean(
        expressionValues, axis=0, keepdims=True
    )
    standardDeviations = np.std(centeredValues, axis=0, ddof=1, keepdims=True)
    standardizedValues = centeredValues / standardDeviations
    referenceIndices = [expressionFrame.columns.get_loc(gene) for gene in referenceGenes]
    referenceValues = standardizedValues[:, referenceIndices]
    correlationMatrix = (
        standardizedValues.T @ referenceValues / (standardizedValues.shape[0] - 1)
    )
    return np.mean(np.abs(correlationMatrix), axis=1)


def buildGeneFeatureFrame(
    fullExpressionFrame: pd.DataFrame,
    selectedExpressionFrame: pd.DataFrame,
    coordinateFrame: pd.DataFrame,
    referenceGenes: list[str],
) -> pd.DataFrame:
    stabilityFrame = pd.read_csv(GENE_DIFFERENTIAL_STABILITY_PATH)
    stabilityFrame["geneSymbol"] = stabilityFrame["geneSymbol"].astype(str).str.upper()
    stabilitySeries = stabilityFrame.groupby("geneSymbol")[
        "differentialStability"
    ].mean()
    coordinateValues = coordinateFrame[["sphereX", "sphereY", "sphereZ"]].to_numpy(
        dtype=float
    )
    hemisphereValues = coordinateFrame["hemisphere"].astype(str).to_numpy()
    spatialWeights = buildHemisphereSpatialWeights(
        coordinateValues, hemisphereValues
    )
    selectedValues = selectedExpressionFrame.to_numpy(dtype=float)
    featureFrame = pd.DataFrame(
        {
            "geneSymbol": fullExpressionFrame.columns,
            "expressionMean400": fullExpressionFrame.mean(axis=0).to_numpy(
                dtype=float
            ),
            "expressionSd400": fullExpressionFrame.std(axis=0, ddof=1).to_numpy(
                dtype=float
            ),
            "expressionMad400": stats.median_abs_deviation(
                fullExpressionFrame.to_numpy(dtype=float), axis=0, nan_policy="omit"
            ),
            "moranI374": calculateMoranForGenes(selectedValues, spatialWeights),
            "meanAbsoluteCorrelation374": calculateMeanAbsoluteCorrelation(
                selectedExpressionFrame, referenceGenes
            ),
        }
    )
    featureFrame["differentialStability"] = featureFrame["geneSymbol"].map(
        stabilitySeries
    )
    return featureFrame


def generateMatchedSets(
    featureFrame: pd.DataFrame,
    trueGenes: list[str],
    excludedGenes: set[str],
    nullCount: int,
    neighborPoolSize: int,
    maximumAttemptsMultiplier: int,
    maximumAbsoluteSmd: float,
    randomGenerator: np.random.Generator,
) -> tuple[np.ndarray, np.ndarray, pd.DataFrame, int, int]:
    validFeatureFrame = featureFrame.dropna(
        subset=MATCHING_FEATURE_COLUMNS
    ).reset_index(drop=True)
    trueFeatureFrame = validFeatureFrame.set_index("geneSymbol").loc[trueGenes]
    candidateFeatureFrame = validFeatureFrame.loc[
        ~validFeatureFrame["geneSymbol"].isin(excludedGenes)
    ].reset_index(drop=True)
    scaler = StandardScaler().fit(validFeatureFrame[MATCHING_FEATURE_COLUMNS])
    candidateScaledFeatures = scaler.transform(
        candidateFeatureFrame[MATCHING_FEATURE_COLUMNS]
    )
    trueScaledFeatures = scaler.transform(trueFeatureFrame[MATCHING_FEATURE_COLUMNS])
    neighborModel = NearestNeighbors(
        n_neighbors=min(neighborPoolSize, len(candidateFeatureFrame)), algorithm="auto"
    ).fit(candidateScaledFeatures)
    neighborDistances, neighborIndices = neighborModel.kneighbors(
        trueScaledFeatures, return_distance=True
    )

    acceptedSets = []
    acceptedNeighborDistances = []
    qcRows = []
    maximumAttempts = nullCount * maximumAttemptsMultiplier
    attemptCount = 0
    while len(acceptedSets) < nullCount and attemptCount < maximumAttempts:
        attemptCount += 1
        usedCandidateIndices: set[int] = set()
        selectedCandidateIndices = np.full(len(trueGenes), -1, dtype=int)
        selectedDistances = np.full(len(trueGenes), np.nan, dtype=float)
        selectionFailed = False
        for trueGeneIndex in randomGenerator.permutation(len(trueGenes)):
            candidateRanks = randomGenerator.permutation(
                neighborIndices.shape[1]
            )
            selectedRank = next(
                (
                    int(candidateRank)
                    for candidateRank in candidateRanks
                    if int(neighborIndices[trueGeneIndex, candidateRank])
                    not in usedCandidateIndices
                ),
                None,
            )
            if selectedRank is None:
                selectionFailed = True
                break
            selectedCandidateIndex = int(
                neighborIndices[trueGeneIndex, selectedRank]
            )
            usedCandidateIndices.add(selectedCandidateIndex)
            selectedCandidateIndices[trueGeneIndex] = selectedCandidateIndex
            selectedDistances[trueGeneIndex] = float(
                neighborDistances[trueGeneIndex, selectedRank]
            )
        if selectionFailed:
            continue
        selectedFeatureValues = candidateFeatureFrame.loc[
            selectedCandidateIndices, MATCHING_FEATURE_COLUMNS
        ]
        featureSmds = {
            featureName: abs(
                standardizedMeanDifference(
                    trueFeatureFrame[featureName].to_numpy(dtype=float),
                    selectedFeatureValues[featureName].to_numpy(dtype=float),
                )
            )
            for featureName in MATCHING_FEATURE_COLUMNS
        }
        maximumSetSmd = max(featureSmds.values())
        if maximumSetSmd > maximumAbsoluteSmd:
            continue
        acceptedSets.append(
            candidateFeatureFrame.loc[
                selectedCandidateIndices, "geneSymbol"
            ].tolist()
        )
        acceptedNeighborDistances.append(selectedDistances)
        qcRows.append(
            {
                "matchedSetIndex": len(acceptedSets) - 1,
                "maximumAbsoluteSmd": maximumSetSmd,
                "meanPairwiseNeighborDistance": float(selectedDistances.mean()),
                "maximumPairwiseNeighborDistance": float(selectedDistances.max()),
                **{
                    f"absoluteSmd_{featureName}": featureSmd
                    for featureName, featureSmd in featureSmds.items()
                },
            }
        )
    if len(acceptedSets) < nullCount:
        raise RuntimeError(
            f"Only {len(acceptedSets)} matched sets passed SMD <= {maximumAbsoluteSmd} "
            f"after {attemptCount} attempts."
        )
    return (
        np.asarray(acceptedSets, dtype=object),
        np.asarray(acceptedNeighborDistances, dtype=float),
        pd.DataFrame(qcRows),
        attemptCount,
        len(candidateFeatureFrame),
    )


def calculateNullCorrelations(
    selectedStandardizedExpression: pd.DataFrame,
    matchedGeneSets: np.ndarray,
    phenotypeValues: np.ndarray,
    phenotypeFrame: pd.DataFrame,
    adjustment: str,
    batchSize: int = 100,
) -> np.ndarray:
    geneIndex = {
        geneSymbol: index
        for index, geneSymbol in enumerate(selectedStandardizedExpression.columns)
    }
    matchedIndices = np.vectorize(geneIndex.get)(matchedGeneSets).astype(np.int32)
    expressionValues = selectedStandardizedExpression.to_numpy(dtype=np.float32)
    phenotypeRanks = stats.rankdata(phenotypeValues, method="average").astype(float)
    phenotypeCentered = phenotypeRanks - phenotypeRanks.mean()
    phenotypeDenominator = np.sqrt(np.sum(phenotypeCentered**2))
    nullCorrelations = np.empty(len(matchedIndices), dtype=float)
    for batchStart in range(0, len(matchedIndices), batchSize):
        batchEnd = min(batchStart + batchSize, len(matchedIndices))
        batchIndices = matchedIndices[batchStart:batchEnd]
        batchScores = expressionValues[:, batchIndices].mean(axis=2).T
        batchScores = residualizeRows(batchScores, phenotypeFrame, adjustment)
        batchRanks = stats.rankdata(batchScores, axis=1, method="average")
        batchCentered = batchRanks - batchRanks.mean(axis=1, keepdims=True)
        nullCorrelations[batchStart:batchEnd] = (
            batchCentered @ phenotypeCentered
        ) / (np.sqrt(np.sum(batchCentered**2, axis=1)) * phenotypeDenominator)
    return nullCorrelations


def main() -> None:
    config = loadConfig()
    statisticsConfig = config["statistics"]
    baseSeed = int(config["project"]["seed"])
    nullCount = int(statisticsConfig["matchedGeneSetNullCount"])
    phenotypeFrame = pd.read_parquet(PHENOTYPE_PATH)
    selectedRoiIds = phenotypeFrame["ROI_ID_1based"].astype(int).tolist()
    fullExpressionFrame = pd.read_parquet(AGGREGATE_EXPRESSION_PATH).set_index(
        "ROI_ID_1based"
    )
    fullExpressionFrame = normalizeExpressionColumns(fullExpressionFrame)
    fullExpressionFrame = fullExpressionFrame.reindex(
        columns=sorted(fullExpressionFrame.columns)
    )
    selectedExpressionFrame = fullExpressionFrame.loc[selectedRoiIds]
    selectedStandardizedExpression = zscoreColumns(fullExpressionFrame).loc[
        selectedRoiIds
    ]
    coordinateFrame = (
        pd.read_csv(ROI_CENTROID_PATH)
        .set_index("ROI_ID_1based")
        .loc[selectedRoiIds]
        .reset_index()
    )
    referenceGeneSeed = stableSeed(baseSeed, "matched-reference-genes")
    referenceGenes = selectFixedReferenceGenes(
        fullExpressionFrame.columns.tolist(),
        int(statisticsConfig["matchedReferenceGeneCount"]),
        referenceGeneSeed,
    )
    featureFrame = buildGeneFeatureFrame(
        fullExpressionFrame,
        selectedExpressionFrame,
        coordinateFrame,
        referenceGenes,
    )
    geneSetFrame = pd.read_csv(
        GENE_SET_OUTPUT_DIRECTORY / "normalized-gene-sets.csv"
    )
    effectiveGeneSets = {
        geneSetName: sorted(
            groupFrame.loc[
                groupFrame["presentInAhbaExpression"], "geneSymbol"
            ].tolist()
        )
        for geneSetName, groupFrame in geneSetFrame.groupby("geneSetName", sort=True)
    }
    allTestedSfariGenes = set().union(*effectiveGeneSets.values())
    spatialResultFrame = pd.read_csv(
        SPATIAL_CORRELATION_OUTPUT_DIRECTORY / "spatial-correlation-results.csv"
    )
    meanZObservedFrame = spatialResultFrame.loc[
        spatialResultFrame["scoreMethod"].eq("mean_z")
    ]
    outputDirectory = MATCHED_NULL_OUTPUT_DIRECTORY
    outputDirectory.mkdir(parents=True, exist_ok=True)
    featureFrame.to_parquet(
        outputDirectory / "gene-matching-features.parquet", index=False
    )
    pd.DataFrame(
        {
            "referenceGeneIndex": np.arange(len(referenceGenes), dtype=int),
            "geneSymbol": referenceGenes,
            "selectionSeed": referenceGeneSeed,
        }
    ).to_csv(
        outputDirectory / "matched-reference-genes.csv",
        index=False,
        encoding="utf-8-sig",
    )

    resultRows = []
    nullRows = []
    matchingSummaryRows = []
    archivePayload = {}
    for geneSetName in GENE_SET_NAMES:
        trueGenes = effectiveGeneSets[geneSetName]
        geneSetSeed = stableSeed(baseSeed, f"matched-null::{geneSetName}")
        randomGenerator = np.random.default_rng(geneSetSeed)
        (
            matchedGeneSets,
            matchedNeighborDistances,
            matchingQcFrame,
            attemptCount,
            candidatePoolSize,
        ) = generateMatchedSets(
            featureFrame,
            trueGenes,
            allTestedSfariGenes,
            nullCount,
            int(statisticsConfig["matchedNeighborPoolSize"]),
            int(statisticsConfig["matchedMaxAttemptsMultiplier"]),
            float(
                statisticsConfig[
                    "matchedMaxAbsoluteStandardizedMeanDifference"
                ]
            ),
            randomGenerator,
        )
        if any(len(set(matchedSet)) != len(trueGenes) for matchedSet in matchedGeneSets):
            raise RuntimeError("A matched set contains duplicate genes.")
        matchingQcFrame.insert(0, "geneSetName", geneSetName)
        matchingQcFrame.to_parquet(
            outputDirectory / f"matching-qc-{geneSetName}.parquet", index=False
        )
        matchedPairFrame = pd.DataFrame(
            {
                "matchedSetIndex": np.repeat(np.arange(nullCount), len(trueGenes)),
                "trueGeneSymbol": np.tile(trueGenes, nullCount),
                "matchedGeneSymbol": matchedGeneSets.reshape(-1),
                "neighborDistance": matchedNeighborDistances.reshape(-1),
            }
        )
        matchedPairFrame.insert(0, "geneSetName", geneSetName)
        matchedPairFrame.to_parquet(
            outputDirectory / f"matched-gene-pairs-{geneSetName}.parquet",
            index=False,
        )
        canonicalMatchedSets = [
            "|".join(sorted(matchedSet)) for matchedSet in matchedGeneSets
        ]
        duplicateMatchedSetFraction = float(
            pd.Series(canonicalMatchedSets).duplicated().mean()
        )
        matchingQcPassed = bool(
            matchingQcFrame["maximumAbsoluteSmd"].le(
                float(
                    statisticsConfig[
                        "matchedMaxAbsoluteStandardizedMeanDifference"
                    ]
                )
                + 1e-12
            ).all()
        )
        matchingSummaryRows.append(
            {
                "geneSetName": geneSetName,
                "trueGeneCount": len(trueGenes),
                "acceptedMatchedSetCount": len(matchedGeneSets),
                "attemptCount": attemptCount,
                "acceptanceFraction": len(matchedGeneSets) / attemptCount,
                "worstAcceptedAbsoluteSmd": float(
                    matchingQcFrame["maximumAbsoluteSmd"].max()
                ),
                "meanAcceptedAbsoluteSmd": float(
                    matchingQcFrame["maximumAbsoluteSmd"].mean()
                ),
                "medianMaximumSmd": float(
                    matchingQcFrame["maximumAbsoluteSmd"].median()
                ),
                "q95MaximumSmd": float(
                    matchingQcFrame["maximumAbsoluteSmd"].quantile(0.95)
                ),
                "meanPairwiseNeighborDistance": float(
                    matchedNeighborDistances.mean()
                ),
                "maximumPairwiseNeighborDistance": float(
                    matchedNeighborDistances.max()
                ),
                "duplicateMatchedSetFraction": duplicateMatchedSetFraction,
                "candidatePoolSize": candidatePoolSize,
                "excludedTestedSfariGeneCount": len(allTestedSfariGenes),
                "candidateBackgroundRule": "exclude_union_of_all_tested_sfari_genes",
                "matchingSpatialFeature": "spherical_distance_inverse_weight_moran_i_374",
                "matchingQcPassed": matchingQcPassed,
                "geneSetSeed": geneSetSeed,
                "referenceGeneSeed": referenceGeneSeed,
            }
        )
        archivePayload[geneSetName] = matchedGeneSets.astype(str)

        for phenotypeSpecification in PHENOTYPE_SPECS:
            phenotypeName = phenotypeSpecification["phenotypeName"]
            nullCorrelations = calculateNullCorrelations(
                selectedStandardizedExpression,
                matchedGeneSets,
                phenotypeFrame[phenotypeName].to_numpy(dtype=float),
                phenotypeFrame,
                phenotypeSpecification["scoreAdjustment"],
            )
            observedCorrelation = float(
                meanZObservedFrame.loc[
                    meanZObservedFrame["geneSetName"].eq(geneSetName)
                    & meanZObservedFrame["phenotypeName"].eq(phenotypeName),
                    "spearmanR",
                ].iloc[0]
            )
            geneSetP = empiricalTwoSidedP(observedCorrelation, nullCorrelations)
            commonResult = {
                "phenotypeName": phenotypeName,
                "geneSetName": geneSetName,
                "pGeneSet": geneSetP,
                "matchedNullCount": len(nullCorrelations),
                "testTail": TEST_TAIL,
                "pFormula": EMPIRICAL_P_FORMULA,
                "alpha": ALPHA,
                "matchingQcPassed": matchingQcPassed,
                **summarizeNullDistribution(
                    observedCorrelation, nullCorrelations, "matchedNull"
                ),
            }
            resultRows.append(
                {
                    **commonResult,
                    "scoreMethod": "mean_z",
                    "matchedNullApplicability": "direct",
                }
            )
            nullRows.extend(
                {
                    "phenotypeName": phenotypeName,
                    "geneSetName": geneSetName,
                    "nullIndex": nullIndex,
                    "nullSpearmanR": float(nullCorrelation),
                }
                for nullIndex, nullCorrelation in enumerate(nullCorrelations)
            )
    np.savez_compressed(
        outputDirectory / "matched-gene-sets.npz", **archivePayload
    )
    pd.DataFrame(matchingSummaryRows).to_csv(
        outputDirectory / "matching-qc-summary.csv",
        index=False,
        encoding="utf-8-sig",
    )
    pd.DataFrame(resultRows).to_csv(
        outputDirectory / "matched-gene-set-results.csv",
        index=False,
        encoding="utf-8-sig",
    )
    pd.DataFrame(nullRows).to_parquet(
        outputDirectory / "matched-gene-set-null-correlations.parquet", index=False
    )


if __name__ == "__main__":
    main()
