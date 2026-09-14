from pathlib import Path
import re

import numpy as np
import pandas as pd

try:
    from neuroCombat import neuroCombat
except ImportError as exc:
    raise ImportError(
        "neuroCombat is not installed. Please install it first: pip install neuroCombat"
    ) from exc


# ============================================================
# 1. Path settings
# ============================================================

# 参数敏感性分析生成的统计指标 CSV 路径
# 该文件应为上一段传播脚本输出的总表：
# EC_SEC_metrics_all_subjects_sensitivity.csv
METRIC_CSV = Path(
    r"I:\DYF\NPI-4-code\3.步进分析\ABIDE1_新结果1-参数敏感性分析\EC_SEC_metrics_all_subjects_sensitivity.csv"
)

# 输出目录
OUTPUT_ROOT = Path(
    r"I:\DYF\NPI-4-code\3.步进分析\ABIDE1_新结果1-敏感性分析_comnbat"
)

# 如果统计指标 CSV 本身已经包含 Group, Age, Sex, Site, FIQ，
# 则设置为 None：
# DEMO_CSV = None
DEMO_CSV = Path(
    r"I:\DYF\NPI-4-code\subject_info_for_stats.csv"
)

OUTPUT_ROOT.mkdir(parents=True, exist_ok=True)


# ============================================================
# 2. Sensitivity analysis settings
# ============================================================

# 敏感性分析中的步数窗口
L_MAX_COL = "L_MAX"
L_MAX_LIST = [30, 40, 50, 60]

# 推荐：True
# True 表示按 L_MAX 分层分别做 ComBat：
#   L_MAX=30 单独做一次
#   L_MAX=40 单独做一次
#   L_MAX=50 单独做一次
#   L_MAX=60 单独做一次
#
# 这样可以避免同一个被试在不同 L_MAX 下重复进入同一个 ComBat 模型。
PROCESS_SEPARATELY_BY_LMAX = True


# ============================================================
# 3. ComBat settings
# ============================================================

EPS_VARIANCE = 1e-12
OVERWRITE_OUTPUT_FILES = True

BATCH_COL = "Site"
CATEGORICAL_COLS = ["Group", "Sex"]

# ComBat 连续协变量仅包括 Age 和 FIQ
# 如果没有 FIQ，脚本会自动跳过
CONTINUOUS_CANDIDATE_COLS = ["Age", "FIQ"]


# ============================================================
# 4. Feature selection settings
# ============================================================

# 手动指定 ComBat 特征列时，在此填写列名列表：
# FEATURE_COLS = [
#     "H1_sensory_peak",
#     "H1_sensory_temporal_centroid",
#     "H1_sensory_early_slope_1_5",
#     "H1_sensory_early_slope_1_10",
#     "H1_sensory_early_slope_1_15",
#     "H1_sensory_auc",
#     ...
# ]
FEATURE_COLS = None

# 针对 EC-SEC 指标表结构，默认只对这些前缀的指标列做 ComBat：
# H1 / H2 / H3 / H4 四组指标，以及 W_outstrength 指标
#
# 敏感性分析列示例：
#   H1_sensory_early_slope_1_5
#   H1_sensory_early_slope_1_10
#   H1_sensory_early_slope_1_15
# 会自动被纳入，因为它们以 H1_ / H2_ / H3_ / H4_ 开头。
FEATURE_PREFIXES = (
    "H1_",
    "H2_",
    "H3_",
    "H4_",
    "W_outstrength_",
)

# 这些是 ID、模式、路径、算法诊断列，不建议做 ComBat
NON_FEATURE_COLS = {
    "sub_id",
    "SUB_ID_norm",
    "_input_row_order",

    # 敏感性分析参数列
    "L_MAX",
    "L_MAX_PROPAGATION",
    "full_auc_window",
    "early_windows",

    # 模型与路径列
    "PROP_MODE",
    "curve_out_path",
    "full_curve_out_path",
    "w_path",
    "V_all_path",
    "key_steps_dir",

    # 协变量列
    "Group",
    "Age",
    "Sex",
    "Site",
    "FIQ",
    "MeanFD",
}

# 这些列虽然是数字，但更像固定参数或访问统计，不建议 ComBat
# 需要纳入 ComBat 的列应从该排除列表中移除
EXCLUDE_NUMERIC_COLS = {
    "P",
    "W_edges",
    "visited_count_last",
    "visited_count_max",
}


# ============================================================
# 5. Output path utilities
# ============================================================

def make_output_paths(label: str):
    """
    根据分析标签生成输出路径。

    例如：
      label = "Lmax_30"
      输出：
        EC_SEC_metrics_all_subjects_sensitivity_combat_Lmax_30.csv
        EC_SEC_metrics_all_subjects_sensitivity_combat_subjects_used_Lmax_30.csv
        ...
    """
    suffix = f"_{label}" if label else ""

    return {
        "combat_csv": OUTPUT_ROOT / f"{METRIC_CSV.stem}_combat{suffix}.csv",
        "subjects_csv": OUTPUT_ROOT / f"{METRIC_CSV.stem}_combat_subjects_used{suffix}.csv",
        "features_csv": OUTPUT_ROOT / f"{METRIC_CSV.stem}_combat_features_report{suffix}.csv",
        "feature_mask_npy": OUTPUT_ROOT / f"{METRIC_CSV.stem}_valid_feature_mask{suffix}.npy",
        "report_txt": OUTPUT_ROOT / f"{METRIC_CSV.stem}_combat_report{suffix}.txt",
        "missing_covariates_csv": OUTPUT_ROOT / f"{METRIC_CSV.stem}_subjects_missing_covariates{suffix}.csv",
    }


OUTPUT_COMBINED_COMBAT_CSV = OUTPUT_ROOT / f"{METRIC_CSV.stem}_combat_combined_by_Lmax.csv"
OUTPUT_COMBINED_FEATURES_CSV = OUTPUT_ROOT / f"{METRIC_CSV.stem}_combat_features_report_combined_by_Lmax.csv"
OUTPUT_COMBINED_REPORT_TXT = OUTPUT_ROOT / f"{METRIC_CSV.stem}_combat_combined_by_Lmax_report.txt"
OUTPUT_FAILED_CONFIGS_CSV = OUTPUT_ROOT / f"{METRIC_CSV.stem}_combat_failed_Lmax_configs.csv"


# ============================================================
# 6. Subject ID utilities
# ============================================================

def normalize_sub_id(raw_sub_id):
    """
    Convert different subject ID styles to the same numeric core.

    Examples:
        sub-Sub28741 -> 28741
        Sub28741     -> 28741
        28741        -> 28741
        28741.0      -> 28741
    """
    if pd.isna(raw_sub_id):
        return ""

    sid = str(raw_sub_id).strip()

    if re.fullmatch(r"\d+\.0", sid):
        sid = sid[:-2]

    sid = re.sub(r"^sub-Sub", "", sid, flags=re.IGNORECASE)
    sid = re.sub(r"^sub-", "", sid, flags=re.IGNORECASE)
    sid = re.sub(r"^Sub", "", sid, flags=re.IGNORECASE)

    digit_match = re.findall(r"\d+", sid)
    if digit_match:
        sid = digit_match[-1].lstrip("0")

    if sid == "":
        sid = "0"

    return sid


# ============================================================
# 7. Load metric table
# ============================================================

def load_metric_table(metric_csv):
    table = pd.read_csv(metric_csv, encoding="utf-8-sig")

    if "sub_id" not in table.columns:
        raise ValueError(f"Missing required column in metric CSV: sub_id")

    table["_input_row_order"] = np.arange(len(table), dtype=int)

    table["SUB_ID_norm"] = table["sub_id"].apply(normalize_sub_id)
    table = table[table["SUB_ID_norm"] != ""].copy()

    # 敏感性分析表中，同一被试会有多个 L_MAX 条件。
    # 不能像旧脚本那样只按 SUB_ID_norm 去重，否则会只保留一个 L_MAX。
    if L_MAX_COL in table.columns:
        table[L_MAX_COL] = pd.to_numeric(table[L_MAX_COL], errors="coerce")

        missing_lmax = table[table[L_MAX_COL].isna()].copy()
        if len(missing_lmax) > 0:
            raise ValueError(
                f"Found rows with missing or non-numeric {L_MAX_COL}: {len(missing_lmax)}"
            )

        table[L_MAX_COL] = table[L_MAX_COL].astype(int)

        duplicated = table[
            table.duplicated(subset=["SUB_ID_norm", L_MAX_COL], keep=False)
        ]

        if len(duplicated) > 0:
            print(
                "Warning: duplicate subject rows found within the same L_MAX. "
                "Keeping the first row for each SUB_ID_norm x L_MAX."
            )
            print(duplicated[["sub_id", "SUB_ID_norm", L_MAX_COL]])

        table = table.drop_duplicates(
            subset=["SUB_ID_norm", L_MAX_COL],
            keep="first",
        ).reset_index(drop=True)

    else:
        duplicated = table[table.duplicated(subset="SUB_ID_norm", keep=False)]

        if len(duplicated) > 0:
            print("Warning: duplicate subjects found in metric CSV. Keeping the first row.")
            print(duplicated[["sub_id", "SUB_ID_norm"]])

        table = table.drop_duplicates(
            subset="SUB_ID_norm",
            keep="first",
        ).reset_index(drop=True)

    return table


# ============================================================
# 8. Load covariates
# ============================================================

def load_covariates(metric_table):
    """
    If DEMO_CSV is None, covariates are read from metric_table.
    Otherwise, covariates are read from DEMO_CSV.
    """

    if DEMO_CSV is None:
        covariate_table = metric_table.copy()
    else:
        header_columns = pd.read_csv(
            DEMO_CSV,
            nrows=0,
            encoding="utf-8-sig",
        ).columns

        required_columns = ["sub_id", "Group", "Age", "Sex", "Site"]
        optional_columns = ["FIQ"]

        missing_columns = [
            col for col in required_columns
            if col not in header_columns
        ]

        if missing_columns:
            raise ValueError(
                f"Missing required covariate columns in {DEMO_CSV}: {missing_columns}"
            )

        use_columns = required_columns + [
            col for col in optional_columns
            if col in header_columns
        ]

        covariate_table = pd.read_csv(
            DEMO_CSV,
            dtype=str,
            encoding="utf-8-sig",
            usecols=use_columns,
        )

    required_columns = ["sub_id", "Group", "Age", "Sex", "Site"]

    missing_columns = [
        col for col in required_columns
        if col not in covariate_table.columns
    ]

    if missing_columns:
        raise ValueError(
            f"Missing required covariate columns: {missing_columns}"
        )

    covariate_table["SUB_ID_norm"] = covariate_table["sub_id"].apply(normalize_sub_id)

    for col in ["Group", "Sex", "Site"]:
        covariate_table[col] = covariate_table[col].astype(str).str.strip()

    available_continuous_cols = []

    for col in CONTINUOUS_CANDIDATE_COLS:
        if col in covariate_table.columns:
            covariate_table[col] = pd.to_numeric(
                covariate_table[col],
                errors="coerce",
            )
            available_continuous_cols.append(col)

    required_for_combat = [
        "SUB_ID_norm",
        "Group",
        "Age",
        "Sex",
        "Site",
    ]

    if "FIQ" in covariate_table.columns:
        required_for_combat.append("FIQ")

    before_count = len(covariate_table)

    covariate_table = covariate_table.dropna(
        subset=required_for_combat
    ).copy()

    covariate_table = covariate_table[
        covariate_table["SUB_ID_norm"] != ""
    ].copy()

    after_count = len(covariate_table)

    if before_count != after_count:
        print(
            f"Dropped {before_count - after_count} rows with missing ComBat covariates."
        )

    duplicated = covariate_table[
        covariate_table.duplicated(subset="SUB_ID_norm", keep=False)
    ]

    if len(duplicated) > 0:
        print("Warning: duplicate subject rows found in covariates. Keeping the first row.")
        print(
            duplicated[
                ["sub_id", "Group", "Age", "Sex", "Site"]
                + [col for col in ["FIQ"] if col in covariate_table.columns]
            ]
        )

    covariate_table = covariate_table.drop_duplicates(
        subset="SUB_ID_norm",
        keep="first",
    ).reset_index(drop=True)

    return covariate_table, available_continuous_cols


# ============================================================
# 9. Match metric rows with covariates
# ============================================================

def match_metric_with_covariates(metric_table, covariate_table, missing_covariates_csv):
    covariate_cols = [
        "SUB_ID_norm",
        "Group",
        "Age",
        "Sex",
        "Site",
    ]

    if "FIQ" in covariate_table.columns:
        covariate_cols.append("FIQ")

    if DEMO_CSV is None:
        merged = metric_table.copy()
    else:
        merged = metric_table.merge(
            covariate_table[covariate_cols],
            on="SUB_ID_norm",
            how="left",
        )

    missing_covariates = merged[
        merged[["Group", "Age", "Sex", "Site"]].isna().any(axis=1)
    ].copy()

    if len(missing_covariates) > 0:
        missing_covariates[["sub_id", "SUB_ID_norm"]].to_csv(
            missing_covariates_csv,
            index=False,
            encoding="utf-8-sig",
        )
        print(f"Subjects missing covariates: {len(missing_covariates)}")
        print(f"Missing covariate report: {missing_covariates_csv}")

    matched = merged.dropna(
        subset=["Group", "Age", "Sex", "Site"]
    ).copy()

    if "FIQ" in matched.columns:
        matched = matched.dropna(subset=["FIQ"]).copy()

    if len(matched) == 0:
        raise RuntimeError("No subjects matched between metric table and covariates.")

    matched = matched.reset_index(drop=True)

    return matched


# ============================================================
# 10. Select feature columns
# ============================================================

def select_feature_columns(table):
    if FEATURE_COLS is not None:
        missing = [col for col in FEATURE_COLS if col not in table.columns]
        if missing:
            raise ValueError(f"FEATURE_COLS contains missing columns: {missing}")
        return list(FEATURE_COLS)

    excluded = set(NON_FEATURE_COLS) | set(EXCLUDE_NUMERIC_COLS)

    numeric_candidate_cols = []

    for col in table.columns:
        if col in excluded:
            continue

        if col.endswith("_path"):
            continue

        if col.endswith("_dir"):
            continue

        numeric_series = pd.to_numeric(table[col], errors="coerce")

        if numeric_series.notna().all():
            numeric_candidate_cols.append(col)

    prefix_feature_cols = [
        col for col in numeric_candidate_cols
        if col.startswith(FEATURE_PREFIXES)
    ]

    if len(prefix_feature_cols) > 0:
        return prefix_feature_cols

    return numeric_candidate_cols


# ============================================================
# 11. Build ComBat covariate table
# ============================================================

def build_combat_covariates(records, continuous_columns):
    covariates = pd.DataFrame(
        {
            "Site": records["Site"].astype(str).values,
            "Group": records["Group"].astype(str).values,
            "Age": pd.to_numeric(records["Age"], errors="raise").values,
            "Sex": records["Sex"].astype(str).values,
        }
    )

    if "FIQ" in continuous_columns:
        covariates["FIQ"] = pd.to_numeric(
            records["FIQ"],
            errors="raise",
        ).values

    for col in continuous_columns:
        covariates[col] = pd.to_numeric(
            covariates[col],
            errors="raise",
        )

    return covariates


def check_batch_balance(covariates):
    site_counts = covariates["Site"].value_counts()

    if covariates["Site"].nunique() < 2:
        raise RuntimeError(
            "ComBat requires at least 2 sites / batches. "
            "Your matched data contain only one Site."
        )

    singleton_sites = site_counts[site_counts < 2]

    if len(singleton_sites) > 0:
        print("\nWarning: some sites have fewer than 2 subjects.")
        print("ComBat estimates may be unstable or may fail for these sites:")
        print(singleton_sites)

    site_group_table = pd.crosstab(covariates["Site"], covariates["Group"])

    print("\nSITE x Group table:")
    print(site_group_table)

    return site_counts, site_group_table


# ============================================================
# 12. Run ComBat for table features
# ============================================================

def run_table_combat(records, feature_cols, covariates, continuous_columns):
    """
    Input table:
        rows = subjects
        columns = features

    neuroCombat requires:
        dat shape = features x subjects
    """

    feature_df = records[feature_cols].apply(
        pd.to_numeric,
        errors="coerce",
    )

    data_feature_by_subject = feature_df.to_numpy(dtype=np.float64).T

    finite_mask = np.isfinite(data_feature_by_subject).all(axis=1)
    variance_by_feature = np.nanvar(data_feature_by_subject, axis=1)
    valid_feature_mask = finite_mask & (variance_by_feature > EPS_VARIANCE)

    total_feature_count = int(data_feature_by_subject.shape[0])
    valid_feature_count = int(np.sum(valid_feature_mask))

    if valid_feature_count == 0:
        raise RuntimeError("No valid features for ComBat.")

    print("\n========== Running ComBat for metric CSV ==========")
    print(f"Subjects: {data_feature_by_subject.shape[1]}")
    print(f"Total candidate features: {total_feature_count}")
    print(f"Features used for ComBat: {valid_feature_count}")
    print(f"Features copied without ComBat: {total_feature_count - valid_feature_count}")

    harmonized_feature_by_subject = data_feature_by_subject.copy()

    combat_result = neuroCombat(
        dat=data_feature_by_subject[valid_feature_mask, :],
        covars=covariates,
        batch_col=BATCH_COL,
        categorical_cols=CATEGORICAL_COLS,
        continuous_cols=continuous_columns,
    )

    harmonized_feature_by_subject[valid_feature_mask, :] = combat_result["data"]

    harmonized_subject_by_feature = harmonized_feature_by_subject.T

    harmonized_feature_df = pd.DataFrame(
        harmonized_subject_by_feature,
        columns=feature_cols,
        index=records.index,
    )

    feature_report_rows = []

    for feature_index, feature_name in enumerate(feature_cols):
        raw_values = data_feature_by_subject[feature_index, :]
        combat_values = harmonized_feature_by_subject[feature_index, :]

        feature_report_rows.append(
            {
                "feature": feature_name,
                "used_for_combat": bool(valid_feature_mask[feature_index]),
                "raw_mean": float(np.nanmean(raw_values)),
                "raw_std": float(np.nanstd(raw_values)),
                "combat_mean": float(np.nanmean(combat_values)),
                "combat_std": float(np.nanstd(combat_values)),
                "raw_min": float(np.nanmin(raw_values)),
                "raw_max": float(np.nanmax(raw_values)),
                "combat_min": float(np.nanmin(combat_values)),
                "combat_max": float(np.nanmax(combat_values)),
            }
        )

    return harmonized_feature_df, valid_feature_mask, feature_report_rows


# ============================================================
# 13. Save outputs
# ============================================================

def save_outputs(
    matched_records,
    feature_cols,
    harmonized_feature_df,
    valid_feature_mask,
    covariates,
    continuous_columns,
    feature_report_rows,
    output_paths,
    analysis_label,
    lmax_value=None,
):
    if output_paths["combat_csv"].exists() and not OVERWRITE_OUTPUT_FILES:
        raise FileExistsError(f"Output already exists: {output_paths['combat_csv']}")

    output_table = matched_records.copy()

    # 用 ComBat 后的数值替换原始指标列
    for col in feature_cols:
        output_table[col] = harmonized_feature_df[col].values

    # 输出给用户的表中不保留内部辅助排序列
    output_table_to_save = output_table.drop(
        columns=["_input_row_order"],
        errors="ignore",
    )

    output_table_to_save.to_csv(
        output_paths["combat_csv"],
        index=False,
        encoding="utf-8-sig",
    )

    subject_cols = [
        "sub_id",
        "SUB_ID_norm",
    ]

    if L_MAX_COL in matched_records.columns:
        subject_cols.append(L_MAX_COL)

    subject_cols += [
        "Site",
        "Group",
        "Age",
        "Sex",
    ]

    if "FIQ" in matched_records.columns:
        subject_cols.append("FIQ")

    matched_records[subject_cols].to_csv(
        output_paths["subjects_csv"],
        index=False,
        encoding="utf-8-sig",
    )

    feature_report_df = pd.DataFrame(feature_report_rows)
    feature_report_df.insert(0, "analysis_label", analysis_label)

    if lmax_value is not None:
        feature_report_df.insert(1, L_MAX_COL, lmax_value)

    feature_report_df.to_csv(
        output_paths["features_csv"],
        index=False,
        encoding="utf-8-sig",
    )

    np.save(output_paths["feature_mask_npy"], valid_feature_mask)

    with open(output_paths["report_txt"], "w", encoding="utf-8") as report_file:
        report_file.write("Metric CSV ComBat report\n")
        report_file.write("========================\n\n")

        report_file.write("Analysis label\n")
        report_file.write(f"  analysis_label: {analysis_label}\n")

        if lmax_value is not None:
            report_file.write(f"  {L_MAX_COL}: {lmax_value}\n")

        report_file.write("\n")

        report_file.write("Paths\n")
        report_file.write(f"  METRIC_CSV: {METRIC_CSV}\n")
        report_file.write(f"  OUTPUT_ROOT: {OUTPUT_ROOT}\n")
        report_file.write(f"  DEMO_CSV: {DEMO_CSV}\n\n")

        report_file.write("ComBat settings\n")
        report_file.write("  dat: features x subjects\n")
        report_file.write(f"  batch_col: {BATCH_COL}\n")
        report_file.write(f"  categorical_cols: {', '.join(CATEGORICAL_COLS)}\n")
        report_file.write(f"  continuous_cols: {', '.join(continuous_columns)}\n")
        report_file.write(f"  EPS_VARIANCE: {EPS_VARIANCE}\n\n")

        report_file.write("Sample size\n")
        report_file.write(f"  matched valid subjects: {len(matched_records)}\n\n")

        report_file.write("Feature columns used\n")
        for col in feature_cols:
            report_file.write(f"  {col}\n")
        report_file.write("\n")

        report_file.write("Site counts\n")
        report_file.write(covariates["Site"].value_counts().to_string())
        report_file.write("\n\n")

        report_file.write("Group counts\n")
        report_file.write(covariates["Group"].value_counts().to_string())
        report_file.write("\n\n")

        report_file.write("Sex counts\n")
        report_file.write(covariates["Sex"].value_counts().to_string())
        report_file.write("\n\n")

        report_file.write("Site x Group\n")
        report_file.write(
            pd.crosstab(covariates["Site"], covariates["Group"]).to_string()
        )
        report_file.write("\n\n")

        report_file.write("Outputs\n")
        report_file.write(f"  combat CSV: {output_paths['combat_csv']}\n")
        report_file.write(f"  subject CSV: {output_paths['subjects_csv']}\n")
        report_file.write(f"  feature report CSV: {output_paths['features_csv']}\n")
        report_file.write(f"  valid feature mask NPY: {output_paths['feature_mask_npy']}\n")
        report_file.write(f"  text report: {output_paths['report_txt']}\n")

    return output_table_to_save, feature_report_df


# ============================================================
# 14. Run one ComBat analysis
# ============================================================

def run_one_combat_analysis(
    metric_subset,
    covariate_table,
    continuous_columns,
    analysis_label,
    lmax_value=None,
):
    output_paths = make_output_paths(analysis_label)

    print("\n============================================================")
    print(f"ComBat analysis: {analysis_label}")

    if lmax_value is not None:
        print(f"{L_MAX_COL}: {lmax_value}")

    print("============================================================")

    matched_records = match_metric_with_covariates(
        metric_table=metric_subset,
        covariate_table=covariate_table,
        missing_covariates_csv=output_paths["missing_covariates_csv"],
    )

    print(f"Matched subjects: {len(matched_records)}")

    # 如果同一个 L_MAX 内仍有同一被试重复，保留第一行
    if L_MAX_COL in matched_records.columns:
        duplicated = matched_records[
            matched_records.duplicated(subset=["SUB_ID_norm", L_MAX_COL], keep=False)
        ]

        if len(duplicated) > 0:
            print(
                "Warning: duplicate matched rows found within the same L_MAX. "
                "Keeping the first row."
            )
            print(duplicated[["sub_id", "SUB_ID_norm", L_MAX_COL]])

            matched_records = matched_records.drop_duplicates(
                subset=["SUB_ID_norm", L_MAX_COL],
                keep="first",
            ).reset_index(drop=True)

    else:
        duplicated = matched_records[
            matched_records.duplicated(subset=["SUB_ID_norm"], keep=False)
        ]

        if len(duplicated) > 0:
            print(
                "Warning: duplicate matched subject rows found. "
                "Keeping the first row."
            )
            print(duplicated[["sub_id", "SUB_ID_norm"]])

            matched_records = matched_records.drop_duplicates(
                subset=["SUB_ID_norm"],
                keep="first",
            ).reset_index(drop=True)

    feature_cols = select_feature_columns(matched_records)

    if len(feature_cols) == 0:
        raise RuntimeError("No feature columns selected for ComBat.")

    print("\n========== Selected feature columns ==========")
    for col in feature_cols:
        print(col)

    covariates = build_combat_covariates(
        records=matched_records,
        continuous_columns=continuous_columns,
    )

    print("\n========== ComBat covariate overview ==========")
    print("Site counts:")
    print(covariates["Site"].value_counts())

    print("\nGroup counts:")
    print(covariates["Group"].value_counts())

    print("\nSex counts:")
    print(covariates["Sex"].value_counts())

    check_batch_balance(covariates)

    harmonized_feature_df, valid_feature_mask, feature_report_rows = run_table_combat(
        records=matched_records,
        feature_cols=feature_cols,
        covariates=covariates,
        continuous_columns=continuous_columns,
    )

    print("\n========== Saving outputs ==========")

    output_table, feature_report_df = save_outputs(
        matched_records=matched_records,
        feature_cols=feature_cols,
        harmonized_feature_df=harmonized_feature_df,
        valid_feature_mask=valid_feature_mask,
        covariates=covariates,
        continuous_columns=continuous_columns,
        feature_report_rows=feature_report_rows,
        output_paths=output_paths,
        analysis_label=analysis_label,
        lmax_value=lmax_value,
    )

    print("\n========== Done ==========")
    print(f"ComBat CSV: {output_paths['combat_csv']}")
    print(f"Subject report: {output_paths['subjects_csv']}")
    print(f"Feature report: {output_paths['features_csv']}")
    print(f"Text report: {output_paths['report_txt']}")

    return output_table, feature_report_df


# ============================================================
# 15. Combined outputs
# ============================================================

def save_combined_outputs(
    combined_combat_tables,
    combined_feature_reports,
    failed_config_rows,
    processed_lmax_values,
):
    if len(combined_combat_tables) > 0:
        combined_combat_df = pd.concat(
            combined_combat_tables,
            axis=0,
            ignore_index=True,
        )

        sort_cols = []
        if L_MAX_COL in combined_combat_df.columns:
            sort_cols.append(L_MAX_COL)

        if "SUB_ID_norm" in combined_combat_df.columns:
            sort_cols.append("SUB_ID_norm")

        if sort_cols:
            combined_combat_df = combined_combat_df.sort_values(
                by=sort_cols,
                kind="mergesort",
            ).reset_index(drop=True)

        combined_combat_df.to_csv(
            OUTPUT_COMBINED_COMBAT_CSV,
            index=False,
            encoding="utf-8-sig",
        )

    else:
        combined_combat_df = pd.DataFrame()

    if len(combined_feature_reports) > 0:
        combined_feature_report_df = pd.concat(
            combined_feature_reports,
            axis=0,
            ignore_index=True,
        )

        combined_feature_report_df.to_csv(
            OUTPUT_COMBINED_FEATURES_CSV,
            index=False,
            encoding="utf-8-sig",
        )

    else:
        combined_feature_report_df = pd.DataFrame()

    if len(failed_config_rows) > 0:
        failed_config_df = pd.DataFrame(failed_config_rows)
        failed_config_df.to_csv(
            OUTPUT_FAILED_CONFIGS_CSV,
            index=False,
            encoding="utf-8-sig",
        )
    else:
        failed_config_df = pd.DataFrame()

    with open(OUTPUT_COMBINED_REPORT_TXT, "w", encoding="utf-8") as report_file:
        report_file.write("Metric CSV ComBat combined sensitivity report\n")
        report_file.write("============================================\n\n")

        report_file.write("Input\n")
        report_file.write(f"  METRIC_CSV: {METRIC_CSV}\n")
        report_file.write(f"  DEMO_CSV: {DEMO_CSV}\n\n")

        report_file.write("Sensitivity settings\n")
        report_file.write(f"  PROCESS_SEPARATELY_BY_LMAX: {PROCESS_SEPARATELY_BY_LMAX}\n")
        report_file.write(f"  L_MAX_COL: {L_MAX_COL}\n")
        report_file.write(f"  requested L_MAX_LIST: {L_MAX_LIST}\n")
        report_file.write(f"  processed L_MAX values: {processed_lmax_values}\n\n")

        report_file.write("ComBat settings\n")
        report_file.write(f"  batch_col: {BATCH_COL}\n")
        report_file.write(f"  categorical_cols: {CATEGORICAL_COLS}\n")
        report_file.write(f"  continuous candidate cols: {CONTINUOUS_CANDIDATE_COLS}\n")
        report_file.write(f"  EPS_VARIANCE: {EPS_VARIANCE}\n\n")

        report_file.write("Outputs\n")
        report_file.write(f"  combined combat CSV: {OUTPUT_COMBINED_COMBAT_CSV}\n")
        report_file.write(f"  combined feature report CSV: {OUTPUT_COMBINED_FEATURES_CSV}\n")
        report_file.write(f"  combined report TXT: {OUTPUT_COMBINED_REPORT_TXT}\n")

        if len(failed_config_df) > 0:
            report_file.write(f"  failed configs CSV: {OUTPUT_FAILED_CONFIGS_CSV}\n")

        report_file.write("\n")

        report_file.write("Combined result size\n")
        report_file.write(f"  combined combat rows: {len(combined_combat_df)}\n")
        report_file.write(f"  combined feature report rows: {len(combined_feature_report_df)}\n")
        report_file.write(f"  failed configs: {len(failed_config_df)}\n")

    print("\n========== Combined outputs ==========")
    print(f"Combined ComBat CSV: {OUTPUT_COMBINED_COMBAT_CSV}")
    print(f"Combined feature report: {OUTPUT_COMBINED_FEATURES_CSV}")
    print(f"Combined report: {OUTPUT_COMBINED_REPORT_TXT}")

    if len(failed_config_rows) > 0:
        print(f"Failed config report: {OUTPUT_FAILED_CONFIGS_CSV}")


# ============================================================
# 16. Main
# ============================================================

def main():
    print("========== Metric CSV ComBat for Sensitivity Analysis ==========")

    metric_table = load_metric_table(METRIC_CSV)
    print(f"Metric rows loaded: {len(metric_table)}")
    print(f"Metric columns loaded: {len(metric_table.columns)}")

    covariate_table, continuous_columns = load_covariates(metric_table)
    print(f"Covariate rows loaded: {len(covariate_table)}")
    print(f"Continuous covariates: {continuous_columns}")

    combined_combat_tables = []
    combined_feature_reports = []
    failed_config_rows = []
    processed_lmax_values = []

    if PROCESS_SEPARATELY_BY_LMAX and L_MAX_COL in metric_table.columns:
        present_lmax_values = sorted(
            pd.to_numeric(
                metric_table[L_MAX_COL],
                errors="coerce",
            ).dropna().astype(int).unique().tolist()
        )

        print(f"\nPresent {L_MAX_COL} values in metric table: {present_lmax_values}")

        target_lmax_values = [
            int(x) for x in L_MAX_LIST
            if int(x) in present_lmax_values
        ]

        missing_requested_lmax = [
            int(x) for x in L_MAX_LIST
            if int(x) not in present_lmax_values
        ]

        if len(missing_requested_lmax) > 0:
            print(
                f"Warning: requested L_MAX values not found in metric table: "
                f"{missing_requested_lmax}"
            )

        if len(target_lmax_values) == 0:
            raise RuntimeError(
                f"No requested L_MAX values found. "
                f"Requested={L_MAX_LIST}, present={present_lmax_values}"
            )

        for lmax_value in target_lmax_values:
            analysis_label = f"Lmax_{lmax_value}"

            metric_subset = metric_table[
                metric_table[L_MAX_COL] == lmax_value
            ].copy()

            try:
                combat_table, feature_report_df = run_one_combat_analysis(
                    metric_subset=metric_subset,
                    covariate_table=covariate_table,
                    continuous_columns=continuous_columns,
                    analysis_label=analysis_label,
                    lmax_value=lmax_value,
                )

                combined_combat_tables.append(combat_table)
                combined_feature_reports.append(feature_report_df)
                processed_lmax_values.append(lmax_value)

            except Exception as exc:
                print(f"\n✘ ComBat failed for {analysis_label}: {str(exc)}")

                failed_config_rows.append(
                    {
                        "analysis_label": analysis_label,
                        L_MAX_COL: lmax_value,
                        "reason": str(exc),
                    }
                )

        save_combined_outputs(
            combined_combat_tables=combined_combat_tables,
            combined_feature_reports=combined_feature_reports,
            failed_config_rows=failed_config_rows,
            processed_lmax_values=processed_lmax_values,
        )

    else:
        # 兼容旧版：如果没有 L_MAX 列，或者不想按 L_MAX 分层，
        # 则把整张表作为一次普通 ComBat 分析。
        analysis_label = "all_rows"

        try:
            combat_table, feature_report_df = run_one_combat_analysis(
                metric_subset=metric_table,
                covariate_table=covariate_table,
                continuous_columns=continuous_columns,
                analysis_label=analysis_label,
                lmax_value=None,
            )

            combined_combat_tables.append(combat_table)
            combined_feature_reports.append(feature_report_df)

        except Exception as exc:
            print(f"\n✘ ComBat failed for {analysis_label}: {str(exc)}")

            failed_config_rows.append(
                {
                    "analysis_label": analysis_label,
                    L_MAX_COL: np.nan,
                    "reason": str(exc),
                }
            )

        save_combined_outputs(
            combined_combat_tables=combined_combat_tables,
            combined_feature_reports=combined_feature_reports,
            failed_config_rows=failed_config_rows,
            processed_lmax_values=[],
        )

    print("\n========== All Done ==========")


if __name__ == "__main__":
    main()
