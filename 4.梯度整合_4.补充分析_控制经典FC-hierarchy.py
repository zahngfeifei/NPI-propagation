# -*- coding: utf-8 -*-
"""
补充分析：控制经典 FC hierarchy。

仅对主要分析数据 ABIDE I 执行：
1. 从经群体模板 Procrustes 对齐和 ComBat 处理的 participant-specific FC-G1
   向量中，计算 H1-H4 系统均值；
2. 在每名参与者的四个系统均值之间进行标准化，得到 FC_G1_star；
3. 在完全相同的 FC 完整样本中拟合：
   a) 未调整模型：EC_G1_star x Group；
   b) FC 调整模型：EC_G1_star x Group + FC_G1_star x Group；
4. 使用 participant-clustered robust OLS，报告调整前后 EC-G1 x Group 的
   beta、95% CI 和精确 P 值，并同时报告 FC-G1 x Group。

注意：
- EC_G1_star 直接使用现有主分析长表中的 G_star；
- FC-G1 使用固定 FC 群体模板方向，不进行逐被试翻转；
- 输出文件仅保存稳定哈希 participantKey，不输出原始 sub_id。
"""

from __future__ import annotations

import hashlib
from pathlib import Path

import numpy as np
import pandas as pd
import statsmodels.formula.api as smf


# =========================================================
# 1) 路径与分析设置
# =========================================================

OUTPUT_ROOT = Path(
    r"I:\DYF\NPI-4-code\4.梯度整合\补充_控制经典 FC hierarchy"
)

HIERARCHY_DIRECTORY = Path(
    r"I:\DYF\NPI-4-code\3.步进分析\种子集合"
)

DATASET_CONFIGS = {
    "ABIDE1": {
        "mainTablePath": Path(
            r"I:\DYF\NPI-3\4.梯度整合\ABIDE1_结果1"
            r"\result1_input_long_table_strict.csv"
        ),
        "fcGradientRoot": Path(
            r"I:\DYF\NPI-4-code\2.梯度分析\功能梯度"
            r"\ABIDE1_功能梯度_群体模板_comabt"
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

FC_G1_FILE_NAME = "FC_G1_procrustes_combat.npy"
EXPECTED_ROI_COUNT = 400
OUTCOME_SCALE = 1000.0
EPSILON_STANDARD_DEVIATION = 1e-12

UNADJUSTED_FORMULA = (
    "early_slope_scaled ~ EC_G1_star * Group + C(system) "
    "+ Age + Sex_bin + FIQ"
)
FC_ADJUSTED_FORMULA = (
    "early_slope_scaled ~ EC_G1_star * Group + FC_G1_star * Group "
    "+ C(system) + Age + Sex_bin + FIQ"
)


# =========================================================
# 2) 数据读取与 H1-H4 系统位置
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
    """读取主分析长表，保留 EC-G1、SEC slope、Group 和相同协变量。"""
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
    systemRoiIndices: dict[str, np.ndarray],
) -> tuple[pd.DataFrame, pd.DataFrame]:
    """由逐被试 FC-G1 计算四系统均值和 FC_G1_star。"""
    if not gradientRoot.exists():
        raise FileNotFoundError(f"FC gradient root is missing: {gradientRoot}")

    gradientRows: list[dict[str, object]] = []
    qualityControlRows: list[dict[str, object]] = []

    for rawSubjectId in subjectIds:
        participantKey = makeParticipantKey(rawSubjectId)
        gradientPath = gradientRoot / rawSubjectId / FC_G1_FILE_NAME

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
                    "FC_G1_system_mean": systemMean,
                    "FC_G1_star": (
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
    fcG1Table: pd.DataFrame,
) -> pd.DataFrame:
    """合并 FC-G1，并锁定调整前后模型共同的完整样本。"""
    if fcG1Table.empty:
        raise RuntimeError("No valid FC-G1 gradient was available.")

    analysisTable = pd.merge(
        mainTable,
        fcG1Table.drop(columns=["participantKey"]),
        on=["sub_id", "system"],
        how="inner",
        validate="one_to_one",
    )
    completeColumns = [
        "EC_G1_star",
        "FC_G1_star",
        "early_slope_scaled",
        "Age",
        "Sex_bin",
        "FIQ",
        "Group",
        "system",
        "participantKey",
    ]
    analysisTable = analysisTable.dropna(subset=completeColumns).copy()

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
        raise RuntimeError("The EC-G1/FC-G1 complete-case table is empty.")

    return analysisTable


# =========================================================
# 3) 聚类稳健模型和结果提取
# =========================================================

def fitParticipantClusteredModel(
    formula: str,
    analysisTable: pd.DataFrame,
):
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
        "FC_G1_system_mean",
        "FC_G1_star",
    ]
    analysisTable[outputColumns].to_csv(
        outputPath,
        index=False,
        encoding="utf-8-sig",
    )


def buildAdjustmentComparison(
    datasetName: str,
    analysisTable: pd.DataFrame,
    unadjustedModel,
    adjustedModel,
) -> tuple[pd.DataFrame, pd.DataFrame]:
    unadjustedEcTerm = findInteractionTerm(unadjustedModel, "EC_G1_star")
    adjustedEcTerm = findInteractionTerm(adjustedModel, "EC_G1_star")
    adjustedFcTerm = findInteractionTerm(adjustedModel, "FC_G1_star")

    unadjustedEcStatistics = extractTermStatistics(
        unadjustedModel, unadjustedEcTerm
    )
    adjustedEcStatistics = extractTermStatistics(adjustedModel, adjustedEcTerm)
    adjustedFcStatistics = extractTermStatistics(adjustedModel, adjustedFcTerm)

    participantCount = analysisTable["participantKey"].nunique()
    observationCount = len(analysisTable)
    interactionComparison = pd.DataFrame(
        [
            {
                "dataset": datasetName,
                "model": "unadjusted_same_fc_complete_case_sample",
                "interaction": "EC_G1_star_x_Group",
                "term": unadjustedEcTerm,
                **unadjustedEcStatistics,
                "participantCount": participantCount,
                "observationCount": observationCount,
            },
            {
                "dataset": datasetName,
                "model": "adjusted_for_FC_G1_star_x_Group",
                "interaction": "EC_G1_star_x_Group",
                "term": adjustedEcTerm,
                **adjustedEcStatistics,
                "participantCount": participantCount,
                "observationCount": observationCount,
            },
            {
                "dataset": datasetName,
                "model": "adjusted_for_FC_G1_star_x_Group",
                "interaction": "FC_G1_star_x_Group",
                "term": adjustedFcTerm,
                **adjustedFcStatistics,
                "participantCount": participantCount,
                "observationCount": observationCount,
            },
        ]
    )

    unadjustedBeta = unadjustedEcStatistics["beta"]
    adjustedBeta = adjustedEcStatistics["beta"]
    betaChange = adjustedBeta - unadjustedBeta
    if abs(unadjustedBeta) > EPSILON_STANDARD_DEVIATION:
        signedPercentChange = 100.0 * betaChange / unadjustedBeta
        magnitudeAttenuationPercent = 100.0 * (
            abs(unadjustedBeta) - abs(adjustedBeta)
        ) / abs(unadjustedBeta)
    else:
        signedPercentChange = np.nan
        magnitudeAttenuationPercent = np.nan

    adjustmentSummary = pd.DataFrame(
        [
            {
                "dataset": datasetName,
                "contrast": "EC_G1_x_Group_after_vs_before_FC_adjustment",
                "unadjustedBeta": unadjustedBeta,
                "adjustedBeta": adjustedBeta,
                "adjustedMinusUnadjustedBeta": betaChange,
                "absoluteBetaChange": abs(betaChange),
                "signedPercentChange": signedPercentChange,
                "magnitudeAttenuationPercent": magnitudeAttenuationPercent,
                "sameCompleteCaseSample": True,
                "participantCount": participantCount,
                "observationCount": observationCount,
            }
        ]
    )
    return interactionComparison, adjustmentSummary


def buildGradientDiagnostics(analysisTable: pd.DataFrame) -> pd.DataFrame:
    """记录 EC-G1 与 FC-G1 系统位置的重叠程度及 FC-G1 方向 QC。"""
    ecG1Values = analysisTable["EC_G1_star"].to_numpy(dtype=float)
    fcG1Values = analysisTable["FC_G1_star"].to_numpy(dtype=float)
    pooledPearsonCorrelation = float(np.corrcoef(ecG1Values, fcG1Values)[0, 1])

    systemMeanPivot = analysisTable.pivot(
        index="participantKey",
        columns="system",
        values="FC_G1_system_mean",
    )
    fcH4MinusH1 = (
        systemMeanPivot["H4_DMN"] - systemMeanPivot["H1_sensory"]
    )

    return pd.DataFrame(
        [
            {
                "diagnostic": "pooled_EC_G1_star_FC_G1_star_Pearson_r",
                "value": pooledPearsonCorrelation,
                "definition": "correlation across all participant x system observations",
            },
            {
                "diagnostic": "mean_FC_G1_H4_minus_H1",
                "value": float(fcH4MinusH1.mean()),
                "definition": "positive values support sensory-low to DMN-high orientation",
            },
            {
                "diagnostic": "proportion_FC_G1_H4_minus_H1_positive",
                "value": float((fcH4MinusH1 > 0.0).mean()),
                "definition": "participant proportion with H4_DMN above H1_sensory",
            },
        ]
    )


# =========================================================
# 4) 单数据集执行与输出
# =========================================================

def runDatasetAnalysis(
    datasetName: str,
    datasetConfig: dict[str, Path],
    systemRoiIndices: dict[str, np.ndarray],
) -> None:
    print(f"\n========== {datasetName}: control FC hierarchy ==========")
    datasetOutputDirectory = OUTPUT_ROOT / datasetName
    datasetOutputDirectory.mkdir(parents=True, exist_ok=True)

    mainTable = prepareMainTable(datasetConfig["mainTablePath"])
    subjectIds = mainTable["sub_id"].drop_duplicates().tolist()
    fcG1Table, gradientQualityControl = buildSystemGradientTable(
        subjectIds=subjectIds,
        gradientRoot=datasetConfig["fcGradientRoot"],
        systemRoiIndices=systemRoiIndices,
    )
    gradientQualityControl.to_csv(
        datasetOutputDirectory / "fc-g1-gradient-qc.csv",
        index=False,
        encoding="utf-8-sig",
    )

    analysisTable = prepareAnalysisTable(mainTable, fcG1Table)
    saveSanitizedAnalysisTable(
        analysisTable,
        datasetOutputDirectory / "analysis-long-table.csv",
    )

    # 两个模型必须使用同一个 FC 完整样本，避免样本构成造成系数变化。
    unadjustedModel = fitParticipantClusteredModel(
        UNADJUSTED_FORMULA,
        analysisTable,
    )
    adjustedModel = fitParticipantClusteredModel(
        FC_ADJUSTED_FORMULA,
        analysisTable,
    )

    modelParametersToTable(unadjustedModel).to_csv(
        datasetOutputDirectory / "unadjusted-model-parameters.csv",
        index=False,
        encoding="utf-8-sig",
    )
    modelParametersToTable(adjustedModel).to_csv(
        datasetOutputDirectory / "fc-adjusted-model-parameters.csv",
        index=False,
        encoding="utf-8-sig",
    )

    interactionComparison, adjustmentSummary = buildAdjustmentComparison(
        datasetName=datasetName,
        analysisTable=analysisTable,
        unadjustedModel=unadjustedModel,
        adjustedModel=adjustedModel,
    )
    interactionComparison.to_csv(
        datasetOutputDirectory / "interaction-adjustment-comparison.csv",
        index=False,
        encoding="utf-8-sig",
    )
    adjustmentSummary.to_csv(
        datasetOutputDirectory / "ec-g1-adjustment-change-summary.csv",
        index=False,
        encoding="utf-8-sig",
    )

    gradientDiagnostics = buildGradientDiagnostics(analysisTable)
    gradientDiagnostics.insert(0, "dataset", datasetName)
    gradientDiagnostics.to_csv(
        datasetOutputDirectory / "gradient-overlap-diagnostics.csv",
        index=False,
        encoding="utf-8-sig",
    )

    includedGradientCount = int(
        gradientQualityControl["status"].eq("included").sum()
    )
    reportLines = [
        "Classical FC hierarchy control analysis",
        f"Dataset: {datasetName}",
        f"Participant count: {analysisTable['participantKey'].nunique()}",
        f"Observation count: {len(analysisTable)}",
        f"Valid FC-G1 gradients: {includedGradientCount}",
        "Model covariance: participant-clustered robust",
        "Cluster variable: participantKey (SHA-256-derived stable key)",
        f"Outcome: early_slope_scaled = early_slope x {OUTCOME_SCALE}",
        f"Unadjusted formula: {UNADJUSTED_FORMULA}",
        f"FC-adjusted formula: {FC_ADJUSTED_FORMULA}",
        "Site and MeanFD are not included in the regression model, matching the main analysis.",
        "FC-G1 input is already ComBat-adjusted before this regression.",
        "FC-G1 orientation: inherited from the fixed FC template; no per-subject flipping.",
        "Adjusted and unadjusted models use the same FC complete-case sample.",
        "",
        "Unadjusted model summary",
        unadjustedModel.summary().as_text(),
        "",
        "FC-adjusted model summary",
        adjustedModel.summary().as_text(),
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

    print(f"\nAll FC-control analyses were saved to: {OUTPUT_ROOT}")


if __name__ == "__main__":
    main()
