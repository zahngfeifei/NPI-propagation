# -*- coding: utf-8 -*-
"""
跨队列固定EC-G1模板分析：检查梯度—传播结果是否依赖同一EC矩阵
================================================================

分析方向
--------
1. ABIDE I 的HC个体G_star -> ABIDE I固定系统模板
   -> 合并到ABIDE II每名参与者的传播指标 -> 检验模板×ASD组差异。

2. ABIDE II 的HC个体G_star -> ABIDE II固定系统模板
   -> 合并到ABIDE I每名参与者的传播指标 -> 检验模板×ASD组差异。

关键原则
--------
- 测试队列自己的G_sys/G_star不进入跨队列模型。
- 训练模板只由训练队列构建，默认只使用HC。
- 模板方向只根据训练队列内部的H1_sensory与H4_DMN确定。
- 两个队列之间只按system合并，不按sub_id匹配。
- 由于固定模板在每个system内不随参与者变化，模板主效应与C(system)
  完全共线。因此主要模型只检验：
      template_x_ASD = G_template_star * I(Group == ASD)
  其系数表示ASD-HC系统差异是否沿独立模板梯度发生线性变化。
- 次要描述模型不含C(system)，用于给出HC/ASD沿模板的简单斜率；
  该模型不能替代主要的system-adjusted检验。

输入
----
两个现有长表均应包含：
sub_id, system, G_sys, G_star, early_slope, Group,
Age, Sex_bin, FIQ, early_slope_scaled

输出
----
- 两个训练队列固定模板CSV
- 两个跨队列分析长表CSV
- 主要模型参数与关键交互结果
- 次要描述模型及HC/ASD简单斜率
- 参与者层级双重bootstrap结果（训练模板+测试参与者）
- 跨方向汇总与QC报告

不重新估计NPI EC，不重新计算传播。
"""

from __future__ import annotations

from pathlib import Path
import hashlib
import warnings

import numpy as np
import pandas as pd
import statsmodels.formula.api as smf


# =========================================================
# 1) 路径
# =========================================================

ABIDE1_FILE = Path(
    r"I:\DYF\NPI-4-code\4.梯度整合\ABIDE1_结果1"
    r"\result1_input_long_table_strict.csv"
)

ABIDE2_FILE = Path(
    r"I:\DYF\NPI-4-code\4.梯度整合\ABIDE2_结果1"
    r"\result1_input_long_table_strict.csv"
)

OUT_DIR = Path(
    r"I:\DYF\NPI-4-code\4.梯度整合\补充_跨队列固定模板同源检查"
)
OUT_DIR.mkdir(parents=True, exist_ok=True)


# =========================================================
# 2) 设置
# =========================================================

SYSTEM_ORDER = [
    "H1_sensory",
    "H2_attention",
    "H3_control",
    "H4_DMN",
]
GROUP_ORDER = ["HC", "ASD"]

# 默认使用训练队列HC建立规范模板。
# 设为 None 时使用训练队列的全部参与者。
TEMPLATE_GROUP: str | None = "HC"

# 使用现有被试内标准化梯度构建模板；每名训练参与者权重相同。
TEMPLATE_SOURCE_COLUMN = "G_star"

OUTCOME_SCALE = 1000.0
EPSILON = 1e-12

# 双重bootstrap：
# 每次同时重抽训练参与者和按组重抽测试参与者。

BOOTSTRAP_ITERATIONS = 1000
BOOTSTRAP_RANDOM_SEED = 20260724
MINIMUM_BOOTSTRAP_SUCCESS_RATE = 0.95

PRIMARY_FORMULA = (
    "early_slope_scaled ~ C(system) + Group + template_x_ASD "
    "+ Age + Sex_bin + FIQ"
)

# 仅用于描述HC/ASD沿模板的简单斜率；没有控制system固定效应。
DESCRIPTIVE_FORMULA = (
    "early_slope_scaled ~ G_template_star * Group "
    "+ Age + Sex_bin + FIQ"
)


# =========================================================
# 3) 通用工具
# =========================================================

def stable_participant_key(dataset: str, raw_id: object) -> str:
    """生成不含原始ID的稳定键，仅用于输出和bootstrap。"""
    token = f"{dataset}|{str(raw_id).strip()}".encode("utf-8")
    return hashlib.sha256(token).hexdigest()[:16]


def require_columns(df: pd.DataFrame, required: set[str], label: str) -> None:
    missing = sorted(required - set(df.columns))
    if missing:
        raise ValueError(f"{label}缺少必要列: {missing}")


def load_dataset(path: Path, dataset: str) -> tuple[pd.DataFrame, dict[str, object]]:
    """读取、清理并冻结一个队列的四系统长表。"""
    if not path.exists():
        raise FileNotFoundError(f"{dataset}输入文件不存在: {path}")

    df = pd.read_csv(path)
    require_columns(
        df,
        {
            "sub_id",
            "system",
            "G_sys",
            "G_star",
            "early_slope",
            "Group",
            "Age",
            "Sex_bin",
            "FIQ",
        },
        dataset,
    )

    df = df.copy()
    df["sub_id"] = df["sub_id"].astype(str).str.strip()
    df["system"] = df["system"].astype(str).str.strip()
    df["Group"] = df["Group"].astype(str).str.strip().str.upper()

    numeric_columns = [
        "G_sys",
        "G_star",
        "early_slope",
        "Age",
        "Sex_bin",
        "FIQ",
    ]
    for column in numeric_columns:
        df[column] = pd.to_numeric(df[column], errors="coerce")

    df = df.loc[
        df["system"].isin(SYSTEM_ORDER)
        & df["Group"].isin(GROUP_ORDER)
    ].copy()

    if df.duplicated(["sub_id", "system"]).any():
        examples = (
            df.loc[
                df.duplicated(["sub_id", "system"], keep=False),
                ["sub_id", "system"],
            ]
            .drop_duplicates()
            .head(10)
            .to_dict("records")
        )
        raise ValueError(
            f"{dataset}存在重复sub_id×system，示例: {examples}"
        )

    before_subjects = df["sub_id"].nunique()

    df = df.dropna(
        subset=[
            "sub_id",
            "system",
            "Group",
            "G_sys",
            "G_star",
            "early_slope",
            "Age",
            "Sex_bin",
            "FIQ",
        ]
    ).copy()

    complete_counts = df.groupby("sub_id")["system"].nunique()
    complete_subjects = complete_counts.index[
        complete_counts.eq(len(SYSTEM_ORDER))
    ]
    df = df.loc[df["sub_id"].isin(complete_subjects)].copy()

    df["early_slope_scaled"] = (
        pd.to_numeric(df["early_slope"], errors="coerce") * OUTCOME_SCALE
    )
    df["participantKey"] = df["sub_id"].map(
        lambda value: stable_participant_key(dataset, value)
    )
    df["dataset"] = dataset

    df["system"] = pd.Categorical(
        df["system"], categories=SYSTEM_ORDER, ordered=True
    )
    df["Group"] = pd.Categorical(
        df["Group"], categories=GROUP_ORDER, ordered=True
    )

    df = df.sort_values(["participantKey", "system"]).reset_index(drop=True)

    if df.empty:
        raise RuntimeError(f"{dataset}清理后没有完整参与者。")

    qc = {
        "dataset": dataset,
        "inputFile": str(path),
        "subjectsBeforeCompleteCase": int(before_subjects),
        "subjectsFinal": int(df["participantKey"].nunique()),
        "observationsFinal": int(len(df)),
        "hcSubjects": int(
            df.loc[df["Group"].astype(str).eq("HC"), "participantKey"].nunique()
        ),
        "asdSubjects": int(
            df.loc[df["Group"].astype(str).eq("ASD"), "participantKey"].nunique()
        ),
    }
    return df, qc


def orient_and_standardize_template(
    template: pd.DataFrame,
) -> tuple[pd.DataFrame, bool]:
    """
    仅依据训练模板本身固定方向：
    H4_DMN应高于H1_sensory，否则整体乘以-1。
    然后在四系统间标准化一次。
    """
    output = template.copy().sort_values("system").reset_index(drop=True)

    value_by_system = output.set_index("system")["G_template_raw"]
    sensory_value = float(value_by_system.loc["H1_sensory"])
    dmn_value = float(value_by_system.loc["H4_DMN"])

    sign_flipped = bool(dmn_value < sensory_value)
    if sign_flipped:
        output["G_template_raw"] = -output["G_template_raw"]

    mean_value = float(output["G_template_raw"].mean())
    sd_value = float(output["G_template_raw"].std(ddof=0))
    if not np.isfinite(sd_value) or sd_value <= EPSILON:
        raise RuntimeError("训练模板四系统标准差为0或非有限值。")

    output["G_template_star"] = (
        output["G_template_raw"] - mean_value
    ) / sd_value

    output["system"] = pd.Categorical(
        output["system"], categories=SYSTEM_ORDER, ordered=True
    )
    output = output.sort_values("system").reset_index(drop=True)
    return output, sign_flipped


def build_fixed_template(
    training_df: pd.DataFrame,
    training_dataset: str,
    sampled_participant_keys: list[str] | None = None,
) -> tuple[pd.DataFrame, dict[str, object]]:
    """
    由训练队列参与者的现有G_star计算四系统固定模板。

    sampled_participant_keys不为None时允许重复，用于参与者bootstrap。
    """
    template_source = training_df.copy()

    if TEMPLATE_GROUP is not None:
        template_source = template_source.loc[
            template_source["Group"].astype(str).eq(TEMPLATE_GROUP)
        ].copy()

    available_keys = set(template_source["participantKey"].unique())
    if not available_keys:
        raise RuntimeError(
            f"{training_dataset}中没有可用于模板的{TEMPLATE_GROUP}参与者。"
        )

    if sampled_participant_keys is not None:
        missing_keys = sorted(set(sampled_participant_keys) - available_keys)
        if missing_keys:
            raise RuntimeError(
                f"bootstrap模板包含不存在的参与者键: {missing_keys[:5]}"
            )

        sampled_frames = []
        for draw_index, participant_key in enumerate(sampled_participant_keys):
            copied = template_source.loc[
                template_source["participantKey"].eq(participant_key),
                ["system", TEMPLATE_SOURCE_COLUMN],
            ].copy()
            copied["templateDrawIndex"] = draw_index
            sampled_frames.append(copied)

        template_source = pd.concat(sampled_frames, ignore_index=True)

    participant_count = (
        len(sampled_participant_keys)
        if sampled_participant_keys is not None
        else template_source["participantKey"].nunique()
    )

    template = (
        template_source.groupby("system", observed=False)[TEMPLATE_SOURCE_COLUMN]
        .mean()
        .reindex(SYSTEM_ORDER)
        .rename("G_template_raw")
        .reset_index()
    )

    if template["G_template_raw"].isna().any():
        raise RuntimeError(
            f"{training_dataset}模板缺少某些系统: "
            f"{template.loc[template['G_template_raw'].isna(), 'system'].tolist()}"
        )

    template, sign_flipped = orient_and_standardize_template(template)
    template.insert(0, "trainingDataset", training_dataset)
    template["templateGroup"] = (
        TEMPLATE_GROUP if TEMPLATE_GROUP is not None else "ALL"
    )
    template["templateParticipantCount"] = int(participant_count)
    template["templateSourceColumn"] = TEMPLATE_SOURCE_COLUMN
    template["signFlipped"] = sign_flipped

    metadata = {
        "trainingDataset": training_dataset,
        "templateGroup": (
            TEMPLATE_GROUP if TEMPLATE_GROUP is not None else "ALL"
        ),
        "templateParticipantCount": int(participant_count),
        "templateSourceColumn": TEMPLATE_SOURCE_COLUMN,
        "signFlipped": sign_flipped,
    }
    return template, metadata


def create_cross_cohort_table(
    test_df: pd.DataFrame,
    template: pd.DataFrame,
    training_dataset: str,
    test_dataset: str,
) -> pd.DataFrame:
    """
    将训练队列四系统固定模板按system复制给测试队列所有参与者。
    测试队列自己的G_sys和G_star被明确删除，不进入模型。
    """
    test_columns = [
        "participantKey",
        "sub_id",
        "system",
        "early_slope",
        "early_slope_scaled",
        "Group",
        "Age",
        "Sex_bin",
        "FIQ",
    ]
    test_table = test_df[test_columns].copy()

    template_columns = [
        "system",
        "G_template_raw",
        "G_template_star",
    ]
    analysis_table = pd.merge(
        test_table,
        template[template_columns],
        on="system",
        how="inner",
        validate="many_to_one",
    )

    analysis_table["trainingDataset"] = training_dataset
    analysis_table["testDataset"] = test_dataset
    analysis_table["ASD_indicator"] = (
        analysis_table["Group"].astype(str).eq("ASD").astype(int)
    )
    analysis_table["template_x_ASD"] = (
        analysis_table["G_template_star"]
        * analysis_table["ASD_indicator"]
    )

    analysis_table["system"] = pd.Categorical(
        analysis_table["system"], categories=SYSTEM_ORDER, ordered=True
    )
    analysis_table["Group"] = pd.Categorical(
        analysis_table["Group"], categories=GROUP_ORDER, ordered=True
    )
    analysis_table = analysis_table.sort_values(
        ["participantKey", "system"]
    ).reset_index(drop=True)

    expected_rows = analysis_table["participantKey"].nunique() * len(SYSTEM_ORDER)
    if len(analysis_table) != expected_rows:
        raise RuntimeError(
            f"{training_dataset}->{test_dataset}合并后行数不完整: "
            f"observed={len(analysis_table)}, expected={expected_rows}"
        )

    # 核心同源性检查：跨队列表中不保留测试参与者自己的梯度列。
    forbidden_columns = {"G_sys", "G_star"}
    if forbidden_columns.intersection(analysis_table.columns):
        raise RuntimeError("测试队列个体梯度意外进入跨队列分析表。")

    return analysis_table


def fit_cluster_model(formula: str, data: pd.DataFrame):
    """参与者聚类稳健OLS。"""
    return smf.ols(formula, data=data).fit(
        cov_type="cluster",
        cov_kwds={"groups": data["participantKey"]},
    )


def extract_term(model, term: str) -> dict[str, float]:
    if term not in model.params.index:
        raise RuntimeError(f"模型中找不到参数: {term}")

    ci = model.conf_int().loc[term]
    return {
        "term": term,
        "beta": float(model.params.loc[term]),
        "standardError": float(model.bse.loc[term]),
        "zOrT": float(model.tvalues.loc[term]),
        "pValue": float(model.pvalues.loc[term]),
        "ci95Low": float(ci.iloc[0]),
        "ci95High": float(ci.iloc[1]),
    }


def model_to_table(model) -> pd.DataFrame:
    ci = model.conf_int()
    return pd.DataFrame(
        {
            "term": model.params.index,
            "beta": model.params.to_numpy(dtype=float),
            "standardError": model.bse.to_numpy(dtype=float),
            "zOrT": model.tvalues.to_numpy(dtype=float),
            "pValue": model.pvalues.to_numpy(dtype=float),
            "ci95Low": ci.iloc[:, 0].to_numpy(dtype=float),
            "ci95High": ci.iloc[:, 1].to_numpy(dtype=float),
        }
    )


def find_descriptive_interaction(model) -> str:
    candidates = [
        "G_template_star:Group[T.ASD]",
        "Group[T.ASD]:G_template_star",
    ]
    for term in candidates:
        if term in model.params.index:
            return term
    raise RuntimeError("描述模型中未找到模板×Group交互项。")


def linear_combo_test(
    model,
    weights: dict[str, float],
) -> dict[str, float]:
    names = list(model.params.index)
    contrast = np.zeros(len(names), dtype=float)

    for term, weight in weights.items():
        if term not in names:
            raise RuntimeError(f"线性组合中找不到参数: {term}")
        contrast[names.index(term)] = float(weight)

    test = model.t_test(contrast)
    ci = np.ravel(test.conf_int())
    return {
        "beta": float(np.ravel(test.effect)[0]),
        "standardError": float(np.ravel(test.sd)[0]),
        "zOrT": float(np.ravel(test.tvalue)[0]),
        "pValue": float(np.ravel(test.pvalue)[0]),
        "ci95Low": float(ci[0]),
        "ci95High": float(ci[1]),
    }


def descriptive_simple_slopes(model) -> pd.DataFrame:
    interaction_term = find_descriptive_interaction(model)
    hc = extract_term(model, "G_template_star")
    asd = linear_combo_test(
        model,
        {
            "G_template_star": 1.0,
            interaction_term: 1.0,
        },
    )
    difference = extract_term(model, interaction_term)

    rows = [
        {
            "contrast": "HC_template_slope",
            **{k: v for k, v in hc.items() if k != "term"},
            "definition": "beta(G_template_star)",
        },
        {
            "contrast": "ASD_template_slope",
            **asd,
            "definition": (
                "beta(G_template_star) + "
                f"beta({interaction_term})"
            ),
        },
        {
            "contrast": "ASD_minus_HC_template_slope",
            **{k: v for k, v in difference.items() if k != "term"},
            "definition": f"beta({interaction_term})",
        },
    ]
    return pd.DataFrame(rows)


# =========================================================
# 4) 参与者层级双重bootstrap
# =========================================================

def bootstrap_one_direction(
    training_df: pd.DataFrame,
    test_df: pd.DataFrame,
    training_dataset: str,
    test_dataset: str,
    iterations: int,
    seed: int,
) -> tuple[pd.DataFrame, pd.DataFrame]:
    """
    每次：
    1. 有放回重抽训练模板参与者；
    2. 在测试队列中按Group分层、有放回重抽参与者；
    3. 重新构建模板和跨队列表；
    4. 用普通OLS拟合主要模型。

    因为测试重抽单位是参与者，bootstrap分布本身处理了四系统重复观测。
    """
    if iterations <= 0:
        return pd.DataFrame(), pd.DataFrame()

    training_pool = training_df.copy()
    if TEMPLATE_GROUP is not None:
        training_pool = training_pool.loc[
            training_pool["Group"].astype(str).eq(TEMPLATE_GROUP)
        ].copy()

    training_keys = (
        training_pool["participantKey"].drop_duplicates().to_numpy()
    )
    test_keys_by_group = {
        group: (
            test_df.loc[
                test_df["Group"].astype(str).eq(group),
                "participantKey",
            ]
            .drop_duplicates()
            .to_numpy()
        )
        for group in GROUP_ORDER
    }

    if len(training_keys) < 2:
        raise RuntimeError(
            f"{training_dataset}模板bootstrap参与者少于2。"
        )
    for group, keys in test_keys_by_group.items():
        if len(keys) < 2:
            raise RuntimeError(
                f"{test_dataset}测试组{group}参与者少于2。"
            )

    rng = np.random.default_rng(seed)
    rows: list[dict[str, object]] = []
    failed = 0

    for iteration in range(1, iterations + 1):
        sampled_training_keys = rng.choice(
            training_keys,
            size=len(training_keys),
            replace=True,
        ).tolist()

        try:
            template, _ = build_fixed_template(
                training_df=training_df,
                training_dataset=training_dataset,
                sampled_participant_keys=sampled_training_keys,
            )

            sampled_test_frames = []
            for group in GROUP_ORDER:
                source_keys = test_keys_by_group[group]
                sampled_keys = rng.choice(
                    source_keys,
                    size=len(source_keys),
                    replace=True,
                )
                for draw_index, participant_key in enumerate(sampled_keys):
                    copied = test_df.loc[
                        test_df["participantKey"].eq(participant_key)
                    ].copy()
                    copied["participantKey"] = (
                        f"{group}_bootstrap_{draw_index:05d}"
                    )
                    sampled_test_frames.append(copied)

            bootstrap_test_df = pd.concat(
                sampled_test_frames, ignore_index=True
            )

            bootstrap_table = create_cross_cohort_table(
                test_df=bootstrap_test_df,
                template=template,
                training_dataset=training_dataset,
                test_dataset=test_dataset,
            )

            # 参与者已作为整体重抽，因此这里用普通OLS。
            bootstrap_model = smf.ols(
                PRIMARY_FORMULA,
                data=bootstrap_table,
            ).fit()

            beta = float(
                bootstrap_model.params.loc["template_x_ASD"]
            )
            rows.append(
                {
                    "iteration": iteration,
                    "trainingDataset": training_dataset,
                    "testDataset": test_dataset,
                    "templateInteractionBeta": beta,
                }
            )
        except (
            np.linalg.LinAlgError,
            RuntimeError,
            ValueError,
            KeyError,
        ):
            failed += 1
            continue

        if iteration % 250 == 0:
            print(
                f"  {training_dataset}->{test_dataset} bootstrap: "
                f"{iteration}/{iterations}"
            )

    distribution = pd.DataFrame(rows)
    successful = len(distribution)
    minimum_successful = int(
        np.ceil(iterations * MINIMUM_BOOTSTRAP_SUCCESS_RATE)
    )
    if successful < minimum_successful:
        raise RuntimeError(
            f"{training_dataset}->{test_dataset} bootstrap成功率不足: "
            f"successful={successful}, failed={failed}"
        )

    values = distribution["templateInteractionBeta"].to_numpy(dtype=float)
    ci_low, ci_high = np.quantile(values, [0.025, 0.975])

    lower_tail = (np.sum(values <= 0.0) + 1) / (successful + 1)
    upper_tail = (np.sum(values >= 0.0) + 1) / (successful + 1)
    two_sided_p = min(1.0, 2.0 * min(lower_tail, upper_tail))

    summary = pd.DataFrame(
        [
            {
                "trainingDataset": training_dataset,
                "testDataset": test_dataset,
                "bootstrapMeanBeta": float(np.mean(values)),
                "bootstrapMedianBeta": float(np.median(values)),
                "bootstrapCi95Low": float(ci_low),
                "bootstrapCi95High": float(ci_high),
                "bootstrapTwoSidedPValue": float(two_sided_p),
                "ciExcludesZero": bool(ci_low > 0.0 or ci_high < 0.0),
                "requestedIterations": int(iterations),
                "successfulIterations": int(successful),
                "failedIterations": int(failed),
                "randomSeed": int(seed),
                "trainingResamplingUnit": "participant",
                "testResamplingUnit": "participant_stratified_by_Group",
            }
        ]
    )
    return distribution, summary


# =========================================================
# 5) 单方向执行
# =========================================================

def run_direction(
    training_df: pd.DataFrame,
    test_df: pd.DataFrame,
    training_dataset: str,
    test_dataset: str,
    bootstrap_seed: int,
) -> dict[str, object]:
    direction_name = f"{training_dataset}_template_to_{test_dataset}"
    direction_dir = OUT_DIR / direction_name
    direction_dir.mkdir(parents=True, exist_ok=True)

    template, template_metadata = build_fixed_template(
        training_df=training_df,
        training_dataset=training_dataset,
    )
    template.to_csv(
        direction_dir / "fixed-gradient-template.csv",
        index=False,
        encoding="utf-8-sig",
    )

    analysis_table = create_cross_cohort_table(
        test_df=test_df,
        template=template,
        training_dataset=training_dataset,
        test_dataset=test_dataset,
    )

    # 不输出原始sub_id，仅输出稳定participantKey。
    sanitized_columns = [
        "participantKey",
        "system",
        "G_template_raw",
        "G_template_star",
        "early_slope",
        "early_slope_scaled",
        "Group",
        "Age",
        "Sex_bin",
        "FIQ",
        "ASD_indicator",
        "template_x_ASD",
        "trainingDataset",
        "testDataset",
    ]
    analysis_table[sanitized_columns].to_csv(
        direction_dir / "cross-cohort-analysis-table.csv",
        index=False,
        encoding="utf-8-sig",
    )

    primary_model = fit_cluster_model(
        PRIMARY_FORMULA,
        analysis_table,
    )
    primary_params = model_to_table(primary_model)
    primary_params.to_csv(
        direction_dir / "primary-system-adjusted-model-parameters.csv",
        index=False,
        encoding="utf-8-sig",
    )
    primary_stats = extract_term(primary_model, "template_x_ASD")

    with open(
        direction_dir / "primary-system-adjusted-model-summary.txt",
        "w",
        encoding="utf-8",
    ) as handle:
        handle.write(
            "Cross-cohort fixed-template matrix-origin check\n"
        )
        handle.write(f"Direction: {training_dataset} -> {test_dataset}\n")
        handle.write(f"Formula: {PRIMARY_FORMULA}\n")
        handle.write(
            "Primary term: template_x_ASD\n"
            "Interpretation: ASD-HC difference in test-cohort propagation "
            "along the independently defined training-cohort template, "
            "after system fixed effects and covariates.\n\n"
        )
        handle.write(primary_model.summary().as_text())

    descriptive_model = fit_cluster_model(
        DESCRIPTIVE_FORMULA,
        analysis_table,
    )
    model_to_table(descriptive_model).to_csv(
        direction_dir / "descriptive-model-parameters.csv",
        index=False,
        encoding="utf-8-sig",
    )
    simple_slopes = descriptive_simple_slopes(descriptive_model)
    simple_slopes.to_csv(
        direction_dir / "descriptive-simple-slopes.csv",
        index=False,
        encoding="utf-8-sig",
    )

    bootstrap_distribution, bootstrap_summary = bootstrap_one_direction(
        training_df=training_df,
        test_df=test_df,
        training_dataset=training_dataset,
        test_dataset=test_dataset,
        iterations=BOOTSTRAP_ITERATIONS,
        seed=bootstrap_seed,
    )
    if not bootstrap_distribution.empty:
        bootstrap_distribution.to_csv(
            direction_dir / "bootstrap-distribution.csv",
            index=False,
            encoding="utf-8-sig",
        )
        bootstrap_summary.to_csv(
            direction_dir / "bootstrap-summary.csv",
            index=False,
            encoding="utf-8-sig",
        )

    direction_summary = {
        "direction": direction_name,
        "trainingDataset": training_dataset,
        "testDataset": test_dataset,
        **template_metadata,
        "testParticipantCount": int(
            analysis_table["participantKey"].nunique()
        ),
        "testObservationCount": int(len(analysis_table)),
        "primaryTerm": "template_x_ASD",
        **{
            f"primary{k[0].upper()}{k[1:]}": v
            for k, v in primary_stats.items()
            if k != "term"
        },
    }

    if not bootstrap_summary.empty:
        for column in [
            "bootstrapMeanBeta",
            "bootstrapMedianBeta",
            "bootstrapCi95Low",
            "bootstrapCi95High",
            "bootstrapTwoSidedPValue",
            "ciExcludesZero",
            "successfulIterations",
            "failedIterations",
        ]:
            direction_summary[column] = bootstrap_summary.iloc[0][column]

    return direction_summary


# =========================================================
# 6) 主程序
# =========================================================

def main() -> None:
    warnings.filterwarnings(
        "ignore",
        message="invalid value encountered in cast",
        category=RuntimeWarning,
    )

    abide1, qc1 = load_dataset(ABIDE1_FILE, "ABIDE1")
    abide2, qc2 = load_dataset(ABIDE2_FILE, "ABIDE2")

    pd.DataFrame([qc1, qc2]).to_csv(
        OUT_DIR / "input-qc-summary.csv",
        index=False,
        encoding="utf-8-sig",
    )

    summaries = [
        run_direction(
            training_df=abide1,
            test_df=abide2,
            training_dataset="ABIDE1",
            test_dataset="ABIDE2",
            bootstrap_seed=BOOTSTRAP_RANDOM_SEED,
        ),
        run_direction(
            training_df=abide2,
            test_df=abide1,
            training_dataset="ABIDE2",
            test_dataset="ABIDE1",
            bootstrap_seed=BOOTSTRAP_RANDOM_SEED + 1,
        ),
    ]

    summary_table = pd.DataFrame(summaries)
    summary_table["samePrimaryDirection"] = (
        np.sign(summary_table["primaryBeta"].iloc[0])
        == np.sign(summary_table["primaryBeta"].iloc[1])
    )
    summary_table.to_csv(
        OUT_DIR / "cross-direction-summary.csv",
        index=False,
        encoding="utf-8-sig",
    )

    report_lines = [
        "Cross-cohort fixed-gradient-template matrix-origin check",
        "=" * 72,
        f"ABIDE1 input: {ABIDE1_FILE}",
        f"ABIDE2 input: {ABIDE2_FILE}",
        f"Output directory: {OUT_DIR}",
        f"Template group: {TEMPLATE_GROUP}",
        f"Template source column: {TEMPLATE_SOURCE_COLUMN}",
        f"Primary formula: {PRIMARY_FORMULA}",
        f"Descriptive formula: {DESCRIPTIVE_FORMULA}",
        f"Bootstrap iterations: {BOOTSTRAP_ITERATIONS}",
        "",
        "Interpretation:",
        (
            "The test cohort's participant-specific G_sys/G_star columns are "
            "not used. Each test participant receives the same four-system "
            "template from the other cohort. The primary template_x_ASD "
            "coefficient tests whether ASD-HC propagation differences follow "
            "the independent template hierarchy."
        ),
        "",
        "Direction summaries:",
        summary_table.to_string(index=False),
    ]

    (OUT_DIR / "analysis-report.txt").write_text(
        "\n".join(report_lines),
        encoding="utf-8",
    )

    print("Cross-cohort fixed-template analysis finished.")
    print("Output directory:", OUT_DIR)
    print("")
    print(summary_table.to_string(index=False))


if __name__ == "__main__":
    main()
