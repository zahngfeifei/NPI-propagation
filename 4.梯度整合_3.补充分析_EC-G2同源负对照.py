# -*- coding: utf-8 -*-
"""
补充分析：EC-G2 同源负对照。

仅对主要分析数据 ABIDE I 执行：
1. 从 participant-specific positive EC matrix 所产生、经 Procrustes 对齐和
   ComBat 处理的 EC-G2 向量中，计算 H1-H4 系统均值；
2. 在每名参与者的四个系统均值之间进行标准化，得到 EC_G2_star；
3. 使用与主分析相同的 participant-clustered robust OLS，分别拟合 EC-G1
   和 EC-G2 模型；
4. 通过按 Group 分层的参与者层级 bootstrap，比较
   beta(EC-G1 x Group) - beta(EC-G2 x Group) 的差异。

注意：
- EC-G1 直接使用现有主分析长表中的 G_star，不重新估计；
- EC-G2 使用既有 EC-G2 模板方向和 Procrustes 对齐结果，不在 bootstrap
  中重新计算或重新定向梯度；
- 输出文件仅保存稳定哈希 participantKey，不输出原始 sub_id。
"""

from __future__ import annotations

import hashlib
from pathlib import Path

import numpy as np
import pandas as pd
import statsmodels.formula.api as smf
from patsy import dmatrices


# =========================================================
# 1) 路径与分析设置
# =========================================================

OUTPUT_ROOT = Path(
    r"I:\DYF\NPI-4-code\4.梯度整合\补充_EC-G2 同源负对照"
)

HIERARCHY_DIRECTORY = Path(
    r"I:\DYF\NPI-4-code\3.步进分析\种子集合"
)

DATASET_CONFIGS = {
    "ABIDE1": {
        "mainTablePath": Path(
            r"I:\DYF\NPI-4-code\4.梯度整合\ABIDE1_结果1"
            r"\result1_input_long_table_strict.csv"
        ),
        "ecGradientRoot": Path(
            r"I:\DYF\NPI-4-code\2.梯度分析\正向连接-独立模板"
            r"\ABIDE1_结果2_combat"
        ),
    },
}

SYSTEM_FILES = {
    "H1_sensory": "H1_sensory.csv",
    "H2_attention": "H2_attention.csv",
    "H3_control": "H3_control.csv",
    "H4_DMN": "H4_DMN.csv",
}

SYSTEM_ORDER = list(SYSTEM_FILES.keys())
GROUP_ORDER = ["HC", "ASD"]

EC_G2_FILE_NAME = "out_G2_procrustes.npy"
EXPECTED_ROI_COUNT = 400
OUTCOME_SCALE = 1000.0
EPSILON_STANDARD_DEVIATION = 1e-12

BOOTSTRAP_ITERATIONS = 5000
BOOTSTRAP_RANDOM_SEED = 20260720
BOOTSTRAP_CI_LOWER_QUANTILE = 0.025
BOOTSTRAP_CI_UPPER_QUANTILE = 0.975
MINIMUM_BOOTSTRAP_SUCCESS_RATE = 0.95

EC_G1_FORMULA = (
    "early_slope_scaled ~ EC_G1_star * Group + C(system) "
    "+ Age + Sex_bin + FIQ"
)
EC_G2_FORMULA = (
    "early_slope_scaled ~ EC_G2_star * Group + C(system) "
    "+ Age + Sex_bin + FIQ"
)


# =========================================================
# 2) 通用工具
# =========================================================

def makeParticipantKey(rawSubjectId: object) -> str:
    """生成稳定、不可逆的参与者键，避免在输出中保留原始 ID。"""
    normalizedSubjectId = str(rawSubjectId).strip().encode("utf-8")
    return hashlib.sha256(normalizedSubjectId).hexdigest()[:16]


def readSystemRoiIndices(expectedRoiCount: int) -> dict[str, np.ndarray]:
    """读取冻结的 H1-H4 ROI 定义，并返回 0-based ROI 索引。"""
    systemRoiIndices: dict[str, np.ndarray] = {}

    for systemName, fileName in SYSTEM_FILES.items():
        roiFilePath = HIERARCHY_DIRECTORY / fileName
        if not roiFilePath.exists():
            raise FileNotFoundError(f"H1-H4 ROI definition is missing: {roiFilePath}")

        roiTable = pd.read_csv(roiFilePath)
        if "ROI_index_0based" in roiTable.columns:
            roiIndices = pd.to_numeric(
                roiTable["ROI_index_0based"], errors="raise"
            ).astype(int).to_numpy()
        elif "ROI_ID" in roiTable.columns:
            roiIndices = (
                pd.to_numeric(roiTable["ROI_ID"], errors="raise")
                .astype(int)
                .to_numpy()
                - 1
            )
        else:
            raise ValueError(
                f"{roiFilePath} must contain ROI_index_0based or ROI_ID."
            )

        roiIndices = np.unique(roiIndices)
        if roiIndices.size == 0:
            raise ValueError(f"No ROI is defined for {systemName}.")
        if roiIndices.min() < 0 or roiIndices.max() >= expectedRoiCount:
            raise ValueError(
                f"ROI index for {systemName} is outside 0..{expectedRoiCount - 1}."
            )

        systemRoiIndices[systemName] = roiIndices

    return systemRoiIndices


def prepareMainTable(mainTablePath: Path) -> pd.DataFrame:
    """读取主分析长表，保留原始 EC-G1 定义及相同协变量。"""
    if not mainTablePath.exists():
        raise FileNotFoundError(f"Main analysis table is missing: {mainTablePath}")

    mainTable = pd.read_csv(mainTablePath)
    requiredColumns = {
        "sub_id",
        "system",
        "G_star",
        "early_slope",
        "Group",
        "Age",
        "Sex_bin",
        "FIQ",
    }
    missingColumns = sorted(requiredColumns - set(mainTable.columns))
    if missingColumns:
        raise ValueError(f"Main table is missing columns: {missingColumns}")

    preparedTable = mainTable.copy()
    preparedTable["sub_id"] = preparedTable["sub_id"].astype(str).str.strip()
    preparedTable["system"] = preparedTable["system"].astype(str).str.strip()
    preparedTable["Group"] = (
        preparedTable["Group"].astype(str).str.strip().str.upper()
    )

    for columnName in ["G_star", "early_slope", "Age", "Sex_bin", "FIQ"]:
        preparedTable[columnName] = pd.to_numeric(
            preparedTable[columnName], errors="coerce"
        )

    preparedTable = preparedTable.loc[
        preparedTable["system"].isin(SYSTEM_ORDER)
        & preparedTable["Group"].isin(GROUP_ORDER)
    ].copy()
    preparedTable = preparedTable.dropna(
        subset=[
            "sub_id",
            "system",
            "G_star",
            "early_slope",
            "Group",
            "Age",
            "Sex_bin",
            "FIQ",
        ]
    ).copy()

    if preparedTable.duplicated(["sub_id", "system"]).any():
        raise ValueError("Main table contains duplicate sub_id x system rows.")

    systemCounts = preparedTable.groupby("sub_id")["system"].nunique()
    completeSubjectIds = systemCounts.index[
        systemCounts.eq(len(SYSTEM_ORDER))
    ]
    preparedTable = preparedTable.loc[
        preparedTable["sub_id"].isin(completeSubjectIds)
    ].copy()

    preparedTable["EC_G1_star"] = preparedTable["G_star"]
    preparedTable["early_slope_scaled"] = (
        preparedTable["early_slope"] * OUTCOME_SCALE
    )
    preparedTable["participantKey"] = preparedTable["sub_id"].map(
        makeParticipantKey
    )

    participantKeyCounts = preparedTable.groupby("participantKey")[
        "sub_id"
    ].nunique()
    if participantKeyCounts.gt(1).any():
        raise RuntimeError("A participantKey collision was detected.")

    if preparedTable.empty:
        raise RuntimeError("No complete participant remains in the main table.")

    return preparedTable


def buildSystemGradientTable(
    subjectIds: list[str],
    gradientRoot: Path,
    gradientFileName: str,
    gradientColumnName: str,
    systemRoiIndices: dict[str, np.ndarray],
) -> tuple[pd.DataFrame, pd.DataFrame]:
    """由逐被试 400-ROI 梯度向量计算四系统均值和被试内标准化位置。"""
    if not gradientRoot.exists():
        raise FileNotFoundError(f"Gradient root is missing: {gradientRoot}")

    gradientRows: list[dict[str, object]] = []
    qualityControlRows: list[dict[str, object]] = []

    for rawSubjectId in subjectIds:
        participantKey = makeParticipantKey(rawSubjectId)
        gradientPath = gradientRoot / rawSubjectId / gradientFileName

        if not gradientPath.exists():
            qualityControlRows.append(
                {
                    "participantKey": participantKey,
                    "status": "missing_gradient_file",
                    "gradientLength": np.nan,
                    "fourSystemStandardDeviation": np.nan,
                }
            )
            continue

        try:
            gradientVector = np.load(gradientPath).reshape(-1).astype(float)
        except Exception as error:
            qualityControlRows.append(
                {
                    "participantKey": participantKey,
                    "status": f"gradient_load_failed_{type(error).__name__}",
                    "gradientLength": np.nan,
                    "fourSystemStandardDeviation": np.nan,
                }
            )
            continue

        if gradientVector.size != EXPECTED_ROI_COUNT:
            qualityControlRows.append(
                {
                    "participantKey": participantKey,
                    "status": "unexpected_gradient_length",
                    "gradientLength": int(gradientVector.size),
                    "fourSystemStandardDeviation": np.nan,
                }
            )
            continue
        if not np.isfinite(gradientVector).all():
            qualityControlRows.append(
                {
                    "participantKey": participantKey,
                    "status": "gradient_contains_non_finite_values",
                    "gradientLength": int(gradientVector.size),
                    "fourSystemStandardDeviation": np.nan,
                }
            )
            continue

        systemMeans = {
            systemName: float(np.mean(gradientVector[roiIndices]))
            for systemName, roiIndices in systemRoiIndices.items()
        }
        orderedSystemMeans = np.asarray(
            [systemMeans[systemName] for systemName in SYSTEM_ORDER],
            dtype=float,
        )
        fourSystemMean = float(np.mean(orderedSystemMeans))
        fourSystemStandardDeviation = float(np.std(orderedSystemMeans, ddof=0))

        if (
            not np.isfinite(fourSystemStandardDeviation)
            or fourSystemStandardDeviation <= EPSILON_STANDARD_DEVIATION
        ):
            qualityControlRows.append(
                {
                    "participantKey": participantKey,
                    "status": "four_system_standard_deviation_too_small",
                    "gradientLength": int(gradientVector.size),
                    "fourSystemStandardDeviation": fourSystemStandardDeviation,
                }
            )
            continue

        for systemName in SYSTEM_ORDER:
            systemMean = systemMeans[systemName]
            gradientRows.append(
                {
                    "sub_id": rawSubjectId,
                    "participantKey": participantKey,
                    "system": systemName,
                    f"{gradientColumnName}_system_mean": systemMean,
                    gradientColumnName: (
                        systemMean - fourSystemMean
                    ) / fourSystemStandardDeviation,
                }
            )

        qualityControlRows.append(
            {
                "participantKey": participantKey,
                "status": "included",
                "gradientLength": int(gradientVector.size),
                "fourSystemStandardDeviation": fourSystemStandardDeviation,
            }
        )

    return pd.DataFrame(gradientRows), pd.DataFrame(qualityControlRows)


def prepareAnalysisTable(
    mainTable: pd.DataFrame,
    ecG2Table: pd.DataFrame,
) -> pd.DataFrame:
    """合并 EC-G1、EC-G2、SEC slope、Group 和协变量，并固定完整样本。"""
    if ecG2Table.empty:
        raise RuntimeError("No valid EC-G2 gradient was available.")

    analysisTable = pd.merge(
        mainTable,
        ecG2Table.drop(columns=["participantKey"]),
        on=["sub_id", "system"],
        how="inner",
        validate="one_to_one",
    )

    requiredCompleteColumns = [
        "EC_G1_star",
        "EC_G2_star",
        "early_slope_scaled",
        "Age",
        "Sex_bin",
        "FIQ",
    ]
    analysisTable = analysisTable.dropna(
        subset=requiredCompleteColumns + ["Group", "system", "participantKey"]
    ).copy()

    systemCounts = analysisTable.groupby("participantKey")["system"].nunique()
    completeParticipantKeys = systemCounts.index[
        systemCounts.eq(len(SYSTEM_ORDER))
    ]
    analysisTable = analysisTable.loc[
        analysisTable["participantKey"].isin(completeParticipantKeys)
    ].copy()

    analysisTable["system"] = pd.Categorical(
        analysisTable["system"], categories=SYSTEM_ORDER, ordered=True
    )
    analysisTable["Group"] = pd.Categorical(
        analysisTable["Group"], categories=GROUP_ORDER, ordered=True
    )
    analysisTable = analysisTable.sort_values(
        ["participantKey", "system"]
    ).reset_index(drop=True)

    if analysisTable.empty:
        raise RuntimeError("The EC-G1/EC-G2 complete-case table is empty.")

    return analysisTable


def fitParticipantClusteredModel(
    formula: str,
    analysisTable: pd.DataFrame,
):
    """拟合与主分析一致的 participant-clustered robust OLS。"""
    return smf.ols(formula, data=analysisTable).fit(
        cov_type="cluster",
        cov_kwds={"groups": analysisTable["participantKey"]},
    )


def findInteractionTerm(model, predictorName: str) -> str:
    candidateTerms = [
        f"{predictorName}:Group[T.ASD]",
        f"Group[T.ASD]:{predictorName}",
    ]
    for candidateTerm in candidateTerms:
        if candidateTerm in model.params.index:
            return candidateTerm
    raise RuntimeError(f"Interaction term was not found for {predictorName}.")


def extractTermStatistics(model, termName: str) -> dict[str, float]:
    confidenceInterval = model.conf_int().loc[termName]
    return {
        "beta": float(model.params.loc[termName]),
        "standardError": float(model.bse.loc[termName]),
        "zOrT": float(model.tvalues.loc[termName]),
        "pValue": float(model.pvalues.loc[termName]),
        "ci95Low": float(confidenceInterval.iloc[0]),
        "ci95High": float(confidenceInterval.iloc[1]),
    }


def modelParametersToTable(model) -> pd.DataFrame:
    confidenceIntervals = model.conf_int()
    return pd.DataFrame(
        {
            "term": model.params.index,
            "beta": model.params.to_numpy(dtype=float),
            "standardError": model.bse.to_numpy(dtype=float),
            "zOrT": model.tvalues.to_numpy(dtype=float),
            "pValue": model.pvalues.to_numpy(dtype=float),
            "ci95Low": confidenceIntervals.iloc[:, 0].to_numpy(dtype=float),
            "ci95High": confidenceIntervals.iloc[:, 1].to_numpy(dtype=float),
        }
    )


# =========================================================
# 3) 参与者层级 bootstrap
# =========================================================

def prepareSubjectCrossProducts(
    formula: str,
    analysisTable: pd.DataFrame,
    participantIndexByKey: dict[str, int],
    predictorName: str,
) -> tuple[np.ndarray, np.ndarray, int]:
    """预计算每名参与者的 X'X 和 X'y，加速重复 bootstrap 拟合。"""
    outcomeDesign, predictorDesign = dmatrices(
        formula,
        analysisTable,
        return_type="dataframe",
    )
    predictorColumns = list(predictorDesign.columns)
    interactionCandidates = [
        f"{predictorName}:Group[T.ASD]",
        f"Group[T.ASD]:{predictorName}",
    ]
    interactionTerm = next(
        (
            candidateTerm
            for candidateTerm in interactionCandidates
            if candidateTerm in predictorColumns
        ),
        None,
    )
    if interactionTerm is None:
        raise RuntimeError(
            f"Bootstrap design has no interaction term for {predictorName}."
        )

    predictorMatrix = predictorDesign.to_numpy(dtype=float)
    outcomeVector = outcomeDesign.to_numpy(dtype=float).reshape(-1)
    participantIndices = analysisTable["participantKey"].map(
        participantIndexByKey
    ).to_numpy(dtype=int)

    participantCount = len(participantIndexByKey)
    parameterCount = predictorMatrix.shape[1]
    subjectCrossProductMatrices = np.zeros(
        (participantCount, parameterCount, parameterCount), dtype=float
    )
    subjectCrossProductVectors = np.zeros(
        (participantCount, parameterCount), dtype=float
    )

    for rowIndex, participantIndex in enumerate(participantIndices):
        predictorRow = predictorMatrix[rowIndex]
        outcomeValue = outcomeVector[rowIndex]
        subjectCrossProductMatrices[participantIndex] += np.outer(
            predictorRow, predictorRow
        )
        subjectCrossProductVectors[participantIndex] += (
            predictorRow * outcomeValue
        )

    interactionColumnIndex = predictorColumns.index(interactionTerm)
    return (
        subjectCrossProductMatrices,
        subjectCrossProductVectors,
        interactionColumnIndex,
    )


def solveBootstrapInteraction(
    participantCounts: np.ndarray,
    subjectCrossProductMatrices: np.ndarray,
    subjectCrossProductVectors: np.ndarray,
    interactionColumnIndex: int,
) -> float:
    weightedCrossProductMatrix = np.tensordot(
        participantCounts,
        subjectCrossProductMatrices,
        axes=(0, 0),
    )
    weightedCrossProductVector = np.tensordot(
        participantCounts,
        subjectCrossProductVectors,
        axes=(0, 0),
    )
    coefficientVector = np.linalg.solve(
        weightedCrossProductMatrix,
        weightedCrossProductVector,
    )
    return float(coefficientVector[interactionColumnIndex])


def runParticipantBootstrap(
    analysisTable: pd.DataFrame,
    observedEcG1Interaction: float,
    observedEcG2Interaction: float,
) -> tuple[pd.DataFrame, pd.DataFrame]:
    """按 Group 分层重抽参与者，并在相同重抽样本中拟合 G1/G2 模型。"""
    participantTable = (
        analysisTable[["participantKey", "Group"]]
        .drop_duplicates()
        .sort_values("participantKey")
        .reset_index(drop=True)
    )
    if participantTable["participantKey"].duplicated().any():
        raise RuntimeError("A participant belongs to more than one Group.")

    participantIndexByKey = {
        participantKey: participantIndex
        for participantIndex, participantKey in enumerate(
            participantTable["participantKey"].tolist()
        )
    }
    groupParticipantIndices = {
        groupName: participantTable.index[
            participantTable["Group"].astype(str).eq(groupName)
        ].to_numpy(dtype=int)
        for groupName in GROUP_ORDER
    }
    for groupName, participantIndices in groupParticipantIndices.items():
        if participantIndices.size == 0:
            raise RuntimeError(f"No participant is available in Group={groupName}.")

    (
        ecG1CrossProductMatrices,
        ecG1CrossProductVectors,
        ecG1InteractionColumnIndex,
    ) = prepareSubjectCrossProducts(
        EC_G1_FORMULA,
        analysisTable,
        participantIndexByKey,
        "EC_G1_star",
    )
    (
        ecG2CrossProductMatrices,
        ecG2CrossProductVectors,
        ecG2InteractionColumnIndex,
    ) = prepareSubjectCrossProducts(
        EC_G2_FORMULA,
        analysisTable,
        participantIndexByKey,
        "EC_G2_star",
    )

    randomGenerator = np.random.default_rng(BOOTSTRAP_RANDOM_SEED)
    bootstrapRows: list[dict[str, float | int]] = []
    failedReplicates = 0

    for bootstrapIteration in range(1, BOOTSTRAP_ITERATIONS + 1):
        participantCounts = np.zeros(len(participantTable), dtype=float)

        for participantIndices in groupParticipantIndices.values():
            sampledPositions = randomGenerator.integers(
                low=0,
                high=participantIndices.size,
                size=participantIndices.size,
            )
            sampledCounts = np.bincount(
                sampledPositions,
                minlength=participantIndices.size,
            )
            participantCounts[participantIndices] = sampledCounts

        try:
            ecG1Interaction = solveBootstrapInteraction(
                participantCounts,
                ecG1CrossProductMatrices,
                ecG1CrossProductVectors,
                ecG1InteractionColumnIndex,
            )
            ecG2Interaction = solveBootstrapInteraction(
                participantCounts,
                ecG2CrossProductMatrices,
                ecG2CrossProductVectors,
                ecG2InteractionColumnIndex,
            )
        except np.linalg.LinAlgError:
            failedReplicates += 1
            continue

        bootstrapRows.append(
            {
                "bootstrapIteration": bootstrapIteration,
                "ecG1InteractionBeta": ecG1Interaction,
                "ecG2InteractionBeta": ecG2Interaction,
                "deltaBetaG1MinusG2": ecG1Interaction - ecG2Interaction,
            }
        )

        if bootstrapIteration % 250 == 0:
            print(
                f"  bootstrap progress: {bootstrapIteration}/{BOOTSTRAP_ITERATIONS}"
            )

    bootstrapTable = pd.DataFrame(bootstrapRows)
    successfulReplicates = len(bootstrapTable)
    minimumSuccessfulReplicates = int(
        np.ceil(BOOTSTRAP_ITERATIONS * MINIMUM_BOOTSTRAP_SUCCESS_RATE)
    )
    if successfulReplicates < minimumSuccessfulReplicates:
        raise RuntimeError(
            "Too many bootstrap fits failed: "
            f"successful={successfulReplicates}, failed={failedReplicates}."
        )

    deltaValues = bootstrapTable["deltaBetaG1MinusG2"].to_numpy(dtype=float)
    ci95Low, ci95High = np.quantile(
        deltaValues,
        [BOOTSTRAP_CI_LOWER_QUANTILE, BOOTSTRAP_CI_UPPER_QUANTILE],
    )
    lowerTailProbability = (
        np.sum(deltaValues <= 0.0) + 1
    ) / (successfulReplicates + 1)
    upperTailProbability = (
        np.sum(deltaValues >= 0.0) + 1
    ) / (successfulReplicates + 1)
    bootstrapTwoSidedPValue = min(
        1.0,
        2.0 * min(lowerTailProbability, upperTailProbability),
    )

    observedDelta = observedEcG1Interaction - observedEcG2Interaction
    bootstrapSummary = pd.DataFrame(
        [
            {
                "contrast": "EC_G1_x_Group_minus_EC_G2_x_Group",
                "observedEcG1InteractionBeta": observedEcG1Interaction,
                "observedEcG2InteractionBeta": observedEcG2Interaction,
                "observedDeltaBeta": observedDelta,
                "bootstrapCi95Low": float(ci95Low),
                "bootstrapCi95High": float(ci95High),
                "bootstrapTwoSidedPValue": float(bootstrapTwoSidedPValue),
                "ciExcludesZero": bool(ci95Low > 0.0 or ci95High < 0.0),
                "requestedReplicates": BOOTSTRAP_ITERATIONS,
                "successfulReplicates": successfulReplicates,
                "failedReplicates": failedReplicates,
                "randomSeed": BOOTSTRAP_RANDOM_SEED,
                "resamplingUnit": "participant_stratified_by_Group",
            }
        ]
    )
    return bootstrapTable, bootstrapSummary


# =========================================================
# 4) 单数据集执行与输出
# =========================================================

def saveSanitizedAnalysisTable(
    analysisTable: pd.DataFrame,
    outputPath: Path,
) -> None:
    outputColumns = [
        "participantKey",
        "system",
        "Group",
        "Age",
        "Sex_bin",
        "FIQ",
        "early_slope",
        "early_slope_scaled",
        "EC_G1_star",
        "EC_G2_star_system_mean",
        "EC_G2_star",
    ]
    analysisTable[outputColumns].to_csv(
        outputPath,
        index=False,
        encoding="utf-8-sig",
    )


def runDatasetAnalysis(
    datasetName: str,
    datasetConfig: dict[str, Path],
    systemRoiIndices: dict[str, np.ndarray],
) -> None:
    print(f"\n========== {datasetName}: EC-G2 negative control ==========")
    datasetOutputDirectory = OUTPUT_ROOT / datasetName
    datasetOutputDirectory.mkdir(parents=True, exist_ok=True)

    mainTable = prepareMainTable(datasetConfig["mainTablePath"])
    subjectIds = mainTable["sub_id"].drop_duplicates().tolist()

    ecG2Table, gradientQualityControl = buildSystemGradientTable(
        subjectIds=subjectIds,
        gradientRoot=datasetConfig["ecGradientRoot"],
        gradientFileName=EC_G2_FILE_NAME,
        gradientColumnName="EC_G2_star",
        systemRoiIndices=systemRoiIndices,
    )
    gradientQualityControl.to_csv(
        datasetOutputDirectory / "ec-g2-gradient-qc.csv",
        index=False,
        encoding="utf-8-sig",
    )

    analysisTable = prepareAnalysisTable(mainTable, ecG2Table)
    saveSanitizedAnalysisTable(
        analysisTable,
        datasetOutputDirectory / "analysis-long-table.csv",
    )

    ecG1Model = fitParticipantClusteredModel(
        EC_G1_FORMULA,
        analysisTable,
    )
    ecG2Model = fitParticipantClusteredModel(
        EC_G2_FORMULA,
        analysisTable,
    )
    ecG1InteractionTerm = findInteractionTerm(ecG1Model, "EC_G1_star")
    ecG2InteractionTerm = findInteractionTerm(ecG2Model, "EC_G2_star")
    ecG1InteractionStatistics = extractTermStatistics(
        ecG1Model, ecG1InteractionTerm
    )
    ecG2InteractionStatistics = extractTermStatistics(
        ecG2Model, ecG2InteractionTerm
    )

    modelParametersToTable(ecG1Model).to_csv(
        datasetOutputDirectory / "ec-g1-model-parameters.csv",
        index=False,
        encoding="utf-8-sig",
    )
    modelParametersToTable(ecG2Model).to_csv(
        datasetOutputDirectory / "ec-g2-model-parameters.csv",
        index=False,
        encoding="utf-8-sig",
    )

    interactionComparison = pd.DataFrame(
        [
            {
                "dataset": datasetName,
                "gradient": "EC_G1",
                "term": ecG1InteractionTerm,
                **ecG1InteractionStatistics,
            },
            {
                "dataset": datasetName,
                "gradient": "EC_G2",
                "term": ecG2InteractionTerm,
                **ecG2InteractionStatistics,
            },
        ]
    )
    interactionComparison["participantCount"] = analysisTable[
        "participantKey"
    ].nunique()
    interactionComparison["observationCount"] = len(analysisTable)
    interactionComparison.to_csv(
        datasetOutputDirectory / "g1-g2-interaction-comparison.csv",
        index=False,
        encoding="utf-8-sig",
    )

    bootstrapTable, bootstrapSummary = runParticipantBootstrap(
        analysisTable=analysisTable,
        observedEcG1Interaction=ecG1InteractionStatistics["beta"],
        observedEcG2Interaction=ecG2InteractionStatistics["beta"],
    )
    bootstrapTable.to_csv(
        datasetOutputDirectory / "bootstrap-g1-minus-g2-distribution.csv",
        index=False,
        encoding="utf-8-sig",
    )
    bootstrapSummary.insert(0, "dataset", datasetName)
    bootstrapSummary.to_csv(
        datasetOutputDirectory / "bootstrap-g1-minus-g2-summary.csv",
        index=False,
        encoding="utf-8-sig",
    )

    includedGradientCount = int(
        gradientQualityControl["status"].eq("included").sum()
    )
    reportLines = [
        "EC-G2 homologous negative-control analysis",
        f"Dataset: {datasetName}",
        f"Participant count: {analysisTable['participantKey'].nunique()}",
        f"Observation count: {len(analysisTable)}",
        f"Valid EC-G2 gradients: {includedGradientCount}",
        "Model covariance: participant-clustered robust",
        "Cluster variable: participantKey (SHA-256-derived stable key)",
        f"Outcome: early_slope_scaled = early_slope x {OUTCOME_SCALE}",
        f"EC-G1 formula: {EC_G1_FORMULA}",
        f"EC-G2 formula: {EC_G2_FORMULA}",
        "EC-G2 orientation: inherited from the fixed EC-G2 template; no per-subject flipping",
        (
            "Bootstrap contrast: beta(EC_G1_star x Group) - "
            "beta(EC_G2_star x Group)"
        ),
        "Bootstrap resampling: participants, stratified by Group",
        f"Bootstrap requested replicates: {BOOTSTRAP_ITERATIONS}",
        f"Bootstrap random seed: {BOOTSTRAP_RANDOM_SEED}",
        "",
        "EC-G1 model summary",
        ecG1Model.summary().as_text(),
        "",
        "EC-G2 model summary",
        ecG2Model.summary().as_text(),
    ]
    (datasetOutputDirectory / "analysis-report.txt").write_text(
        "\n".join(reportLines), encoding="utf-8"
    )

    print(
        f"Completed {datasetName}: "
        f"participants={analysisTable['participantKey'].nunique()}, "
        f"observations={len(analysisTable)}"
    )


def main() -> None:
    OUTPUT_ROOT.mkdir(parents=True, exist_ok=True)
    systemRoiIndices = readSystemRoiIndices(EXPECTED_ROI_COUNT)

    for datasetName, datasetConfig in DATASET_CONFIGS.items():
        runDatasetAnalysis(
            datasetName=datasetName,
            datasetConfig=datasetConfig,
            systemRoiIndices=systemRoiIndices,
        )

    print(f"\nAll EC-G2 analyses were saved to: {OUTPUT_ROOT}")


if __name__ == "__main__":
    main()
