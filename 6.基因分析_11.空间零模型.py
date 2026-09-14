from __future__ import annotations

import importlib.util
from pathlib import Path
import sys

import numpy as np
import pandas as pd
from brainsmash.mapgen.base import Base
from scipy import stats

COMMON_SCRIPT_PATH = Path(__file__).with_name("6.基因分析_14.公共配置与函数.py")
commonModuleSpec = importlib.util.spec_from_file_location("analysis_common", COMMON_SCRIPT_PATH)
if commonModuleSpec is None or commonModuleSpec.loader is None:
    raise RuntimeError(f"无法加载公共配置：{COMMON_SCRIPT_PATH}")
geneAnalysisCommon = importlib.util.module_from_spec(commonModuleSpec)
sys.modules[commonModuleSpec.name] = geneAnalysisCommon
commonModuleSpec.loader.exec_module(geneAnalysisCommon)

from analysis_common import (
    PHENOTYPE_SPECS,
    PHENOTYPE_PATH,
    ROI_CENTROID_PATH,
    SPATIAL_NULL_OUTPUT_DIRECTORY,
    buildHemisphereSpatialWeights,
    calculateMoranI,
    loadConfig,
    sphericalDistanceMatrix,
    stableSeed,
    validateSystemResidualDefinition,
    vectorizedRowCorrelations,
    writeJson,
)


def generateHemisphereSurrogates(
    phenotypeValues: np.ndarray,
    distanceMatrix: np.ndarray,
    nullCount: int,
    seed: int,
    nJobs: int,
    variogramPercentile: int,
) -> tuple[np.ndarray, Base]:
    surrogateGenerator = Base(
        phenotypeValues,
        distanceMatrix,
        seed=seed,
        n_jobs=nJobs,
        resample=False,
        pv=variogramPercentile,
    )
    surrogateValues = np.asarray(
        surrogateGenerator(n=nullCount, batch_size=100), dtype=np.float32
    )
    return surrogateValues, surrogateGenerator


def evaluateVariogramFit(
    phenotypeName: str,
    hemisphereName: str,
    phenotypeValues: np.ndarray,
    surrogateValues: np.ndarray,
    surrogateGenerator: Base,
    config: dict,
) -> tuple[dict[str, float | int | str | bool], pd.DataFrame]:
    statisticsConfig = config["statistics"]
    sampledCount = min(
        int(statisticsConfig["spatialNullQcSampleCount"]), len(surrogateValues)
    )
    sampledIndices = np.linspace(0, len(surrogateValues) - 1, sampledCount, dtype=int)
    empiricalVariogram, distanceBins = surrogateGenerator.compute_smooth_variogram(
        phenotypeValues, return_h=True
    )
    surrogateVariograms = np.vstack(
        [
            surrogateGenerator.compute_smooth_variogram(surrogateValues[index])
            for index in sampledIndices
        ]
    )
    surrogateMean = surrogateVariograms.mean(axis=0)
    surrogateLower = np.quantile(surrogateVariograms, 0.025, axis=0)
    surrogateUpper = np.quantile(surrogateVariograms, 0.975, axis=0)
    variogramCorrelation = float(
        stats.pearsonr(empiricalVariogram, surrogateMean).statistic
    )
    empiricalRange = float(np.ptp(empiricalVariogram))
    normalizedRootMeanSquareError = float(
        np.sqrt(np.mean((empiricalVariogram - surrogateMean) ** 2))
        / (empiricalRange if empiricalRange > 0 else 1.0)
    )
    envelopeCoverage = float(
        np.mean(
            (empiricalVariogram >= surrogateLower)
            & (empiricalVariogram <= surrogateUpper)
        )
    )
    variogramQcPassed = bool(
        variogramCorrelation
        >= float(statisticsConfig["spatialVariogramMinimumCorrelation"])
        and normalizedRootMeanSquareError
        <= float(statisticsConfig["spatialVariogramMaximumNormalizedRmse"])
        and envelopeCoverage
        >= float(statisticsConfig["spatialVariogramMinimumEnvelopeCoverage"])
    )
    summaryRow = {
        "phenotypeName": phenotypeName,
        "hemisphere": hemisphereName,
        "sampledSurrogateCount": sampledCount,
        "variogramCorrelation": variogramCorrelation,
        "normalizedRootMeanSquareError": normalizedRootMeanSquareError,
        "empiricalEnvelopeCoverage": envelopeCoverage,
        "variogramQcPassed": variogramQcPassed,
    }
    detailFrame = pd.DataFrame(
        {
            "phenotypeName": phenotypeName,
            "hemisphere": hemisphereName,
            "distanceBin": distanceBins,
            "empiricalVariogram": empiricalVariogram,
            "surrogateVariogramMean": surrogateMean,
            "surrogateVariogramLower95": surrogateLower,
            "surrogateVariogramUpper95": surrogateUpper,
        }
    )
    return summaryRow, detailFrame


def main() -> None:
    config = loadConfig()
    statisticsConfig = config["statistics"]
    nullCount = int(statisticsConfig["spatialNullCount"])
    variogramPercentileCandidates = [
        int(candidate)
        for candidate in statisticsConfig["spatialVariogramPercentileCandidates"]
    ]
    if not variogramPercentileCandidates:
        raise ValueError("At least one variogram percentile candidate is required.")
    baseSeed = int(config["project"]["seed"])
    nJobs = int(statisticsConfig["nJobs"])
    phenotypeFrame = pd.read_parquet(PHENOTYPE_PATH)
    validateSystemResidualDefinition(phenotypeFrame)
    coordinateFrame = pd.read_csv(ROI_CENTROID_PATH).set_index("ROI_ID_1based")
    selectedRoiIds = phenotypeFrame["ROI_ID_1based"].astype(int).to_numpy()
    coordinateFrame = coordinateFrame.loc[selectedRoiIds].reset_index()
    coordinateValues = coordinateFrame[["sphereX", "sphereY", "sphereZ"]].to_numpy(
        dtype=float
    )
    hemisphereValues = phenotypeFrame["hemisphere"].astype(str).to_numpy()
    if not np.array_equal(
        coordinateFrame["hemisphere"].astype(str).to_numpy(), hemisphereValues
    ):
        raise RuntimeError("Centroid and phenotype hemisphere labels are not aligned.")
    spatialWeights = buildHemisphereSpatialWeights(coordinateValues, hemisphereValues)

    outputDirectory = SPATIAL_NULL_OUTPUT_DIRECTORY
    outputDirectory.mkdir(parents=True, exist_ok=True)
    hemisphereQcRows = []
    phenotypeQcRows = []
    variogramDetailFrames = []
    fitSelectionAuditRows = []
    surrogateArchive = {}
    for phenotypeSpecification in PHENOTYPE_SPECS:
        phenotypeName = phenotypeSpecification["phenotypeName"]
        phenotypeValues = phenotypeFrame[phenotypeName].to_numpy(dtype=float)
        combinedSurrogates = np.empty(
            (nullCount, len(phenotypeFrame)), dtype=np.float32
        )
        currentHemisphereRows = []
        for hemisphereName in ["LH", "RH"]:
            hemisphereMask = hemisphereValues == hemisphereName
            hemisphereDistances = sphericalDistanceMatrix(
                coordinateValues[hemisphereMask]
            )
            hemisphereSeed = stableSeed(
                baseSeed, f"spatial-null::{phenotypeName}::{hemisphereName}"
            )
            hemisphereSurrogates = None
            hemisphereQcRow = None
            variogramDetailFrame = None
            for variogramPercentile in variogramPercentileCandidates:
                candidateSurrogates, candidateGenerator = generateHemisphereSurrogates(
                    phenotypeValues[hemisphereMask],
                    hemisphereDistances,
                    nullCount,
                    hemisphereSeed,
                    nJobs,
                    variogramPercentile,
                )
                candidateQcRow, candidateDetailFrame = evaluateVariogramFit(
                    phenotypeName,
                    hemisphereName,
                    phenotypeValues[hemisphereMask],
                    candidateSurrogates,
                    candidateGenerator,
                    config,
                )
                candidateAuditRow = dict(candidateQcRow)
                candidateAuditRow["variogramPercentile"] = variogramPercentile
                candidateAuditRow["selected"] = bool(
                    candidateQcRow["variogramQcPassed"]
                )
                fitSelectionAuditRows.append(candidateAuditRow)
                if candidateQcRow["variogramQcPassed"]:
                    hemisphereSurrogates = candidateSurrogates
                    hemisphereQcRow = candidateQcRow
                    variogramDetailFrame = candidateDetailFrame
                    hemisphereQcRow["variogramPercentile"] = variogramPercentile
                    variogramDetailFrame["variogramPercentile"] = variogramPercentile
                    break
            if (
                hemisphereSurrogates is None
                or hemisphereQcRow is None
                or variogramDetailFrame is None
            ):
                pd.DataFrame(fitSelectionAuditRows).to_csv(
                    outputDirectory / "spatial-null-fit-selection-audit.csv",
                    index=False,
                    encoding="utf-8-sig",
                )
                raise RuntimeError(
                    "No BrainSMASH variogram range passed QC for "
                    f"{phenotypeName}/{hemisphereName}."
                )
            combinedSurrogates[:, hemisphereMask] = hemisphereSurrogates
            hemisphereQcRow["surrogateCount"] = nullCount
            hemisphereQcRow["seed"] = hemisphereSeed
            currentHemisphereRows.append(hemisphereQcRow)
            hemisphereQcRows.append(hemisphereQcRow)
            variogramDetailFrames.append(variogramDetailFrame)

        surrogateArchive[phenotypeSpecification["surrogateKey"]] = combinedSurrogates
        sampledCount = min(
            int(statisticsConfig["spatialNullQcSampleCount"]), nullCount
        )
        sampledIndices = np.linspace(0, nullCount - 1, sampledCount, dtype=int)
        sampledMoranValues = np.asarray(
            [
                calculateMoranI(combinedSurrogates[index], spatialWeights)
                for index in sampledIndices
            ],
            dtype=float,
        )
        observedMoran = calculateMoranI(phenotypeValues, spatialWeights)
        moranLower = float(np.quantile(sampledMoranValues, 0.025))
        moranUpper = float(np.quantile(sampledMoranValues, 0.975))
        moranQcPassed = bool(moranLower <= observedMoran <= moranUpper)
        observedNullCorrelations = vectorizedRowCorrelations(
            phenotypeValues, combinedSurrogates
        )
        variogramQcPassed = bool(
            all(row["variogramQcPassed"] for row in currentHemisphereRows)
        )
        spatialQcPassed = bool(variogramQcPassed and moranQcPassed)
        phenotypeQcRows.append(
            {
                "phenotypeName": phenotypeName,
                "surrogateKey": phenotypeSpecification["surrogateKey"],
                "surrogateCount": nullCount,
                "observedMoranI": observedMoran,
                "nullMoranIMean": float(sampledMoranValues.mean()),
                "nullMoranISd": float(sampledMoranValues.std(ddof=1)),
                "nullMoranIQ025": moranLower,
                "nullMoranIQ975": moranUpper,
                "moranQcPassed": moranQcPassed,
                "variogramFitError": float(
                    max(
                        row["normalizedRootMeanSquareError"]
                        for row in currentHemisphereRows
                    )
                ),
                "variogramQcPassed": variogramQcPassed,
                "observedNullCorrelationMedian": float(
                    np.median(observedNullCorrelations)
                ),
                "observedNullCorrelationSd": float(
                    observedNullCorrelations.std(ddof=1)
                ),
                "observedNullCorrelationQ025": float(
                    np.quantile(observedNullCorrelations, 0.025)
                ),
                "observedNullCorrelationQ975": float(
                    np.quantile(observedNullCorrelations, 0.975)
                ),
                "distanceType": "spherical_great_circle_angular_distance",
                "hemisphereSeparated": True,
                "crossHemisphereMatching": False,
                "leftHemisphereSurrogateCount": nullCount,
                "rightHemisphereSurrogateCount": nullCount,
                "spatialQcPassed": spatialQcPassed,
            }
        )

    hemisphereQcFrame = pd.DataFrame(hemisphereQcRows)
    phenotypeQcFrame = pd.DataFrame(phenotypeQcRows)
    hemisphereQcFrame.to_csv(
        outputDirectory / "spatial-null-hemisphere-qc.csv",
        index=False,
        encoding="utf-8-sig",
    )
    phenotypeQcFrame.to_csv(
        outputDirectory / "spatial-null-qc.csv",
        index=False,
        encoding="utf-8-sig",
    )
    pd.concat(variogramDetailFrames, ignore_index=True).to_csv(
        outputDirectory / "spatial-null-variogram-qc.csv",
        index=False,
        encoding="utf-8-sig",
    )
    pd.DataFrame(fitSelectionAuditRows).to_csv(
        outputDirectory / "spatial-null-fit-selection-audit.csv",
        index=False,
        encoding="utf-8-sig",
    )
    np.savez_compressed(
        outputDirectory / "phenotype-spatial-surrogates.npz", **surrogateArchive
    )
    writeJson(
        outputDirectory / "spatial-null-method.json",
        {
            "software": "BrainSMASH",
            "nullCountPerPhenotype": nullCount,
            "directPhenotypes": [
                phenotypeSpecification["phenotypeName"]
                for phenotypeSpecification in PHENOTYPE_SPECS
            ],
            "systemResidualSurrogatesGeneratedDirectly": True,
            "hemisphereSeparated": True,
            "crossHemisphereMatching": False,
            "distanceType": "spherical great-circle angular distance between frozen Schaefer-400 parcel centroids",
            "variogramPercentileCandidates": variogramPercentileCandidates,
            "variogramPercentileSelection": "first candidate passing all predefined variogram QC thresholds per phenotype and hemisphere",
            "variogramQcPassed": bool(phenotypeQcFrame["variogramQcPassed"].all()),
            "moranQcPassed": bool(phenotypeQcFrame["moranQcPassed"].all()),
            "spatialQcPassed": bool(phenotypeQcFrame["spatialQcPassed"].all()),
            "baseSeed": baseSeed,
        },
    )
    if not phenotypeQcFrame["spatialQcPassed"].all():
        failedPhenotypes = phenotypeQcFrame.loc[
            ~phenotypeQcFrame["spatialQcPassed"], "phenotypeName"
        ].tolist()
        raise RuntimeError(
            f"Direct BrainSMASH QC failed for phenotypes: {failedPhenotypes}"
        )


if __name__ == "__main__":
    main()
