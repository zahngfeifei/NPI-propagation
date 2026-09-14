import os
import re
import warnings
from pathlib import Path
from datetime import datetime

import numpy as np
import pandas as pd
import matplotlib.pyplot as plt

import statsmodels.formula.api as smf
from patsy.highlevel import dmatrix
from patsy.build import build_design_matrices


# ============================================================
# 1. 路径设置
# ============================================================

BASE_DIR = r"I:\DYF\NPI-4-code\5.行为分析"

# 脑指标 + 协变量表。
# 这个文件来自前一步 Result 4 数据准备脚本，包含：
# sub_id, Group, Age, Sex_for_model, FIQ, Coupling_index_scaled 等。
# 注意：本版本不把 MeanFD 和 Site 放入模型。
BRAIN_INPUT_FILE = os.path.join(
    BASE_DIR,
    "ABIDE2_结果1行为相关输出",
    "result4_brain_subject_metrics_for_behavior.csv"
)

# 行为量表 CSV 文件夹。
# 本脚本会扫描该文件夹中的所有 .csv 文件，并自动识别每个 CSV 内可数值化的量表列。
# 如果每个 CSV 是一个单量表文件，则每个 CSV 会产生一个模型；如果某个 CSV 内有多个数值量表列，则每列分别建模。
BEHAVIOR_CSV_DIR = r"I:\DYF\NPI-4-code\5.行为分析\ABIDE2_行为量表\ABIDE2\ADOS"

# 是否递归扫描子文件夹。
SCAN_RECURSIVE = False

# 可选：跳过不想分析的文件名关键词。例如想跳过汇总表，可加入 "summary"。
EXCLUDE_FILE_KEYWORDS = [
    "summary",
    "audit",
]

# 不作为行为因变量的基础列。除这些列外，能转换为数值且至少有 1 个非缺失值的列会被作为候选量表列。
NON_BEHAVIOR_COLUMNS = {
    "Dataset", "SITE_ID", "SUB_ID", "sub_id", "FILE_ID", "DX_GROUP", "Group", "Group_behavior",
    "SEX", "Sex", "AGE_AT_SCAN", "Age", "FIQ", "VIQ", "PIQ", "MeanFD", "Site",
}



# ============================================================
# 2. 分析变量设置
# ============================================================

BRAIN_PREDICTORS = [
    "Coupling_index_scaled",
]

# ============================================================
# 2a. 多参数敏感性分析设置
# ============================================================
# 每一组 spec 会被完整运行一次，并写入独立输出文件夹。
# 该敏感性分析仅改变 spline 相关参数；所有模型均不加入 MeanFD 和 Site fixed effect，
# 因为站点/头动已在 ComBat 或前处理步骤中控制。
#
# 重要说明：
#   - 本轮敏感性分析不再纳入 df=1、df=2、df=3；
#     AIC、BIC、CV_RMSE 选择范围均限制为 spline df=4--10。
#   - 固定 df 敏感性分析也只保留 df=4--10。
#   - df>=4 时使用目标 cubic B-spline degree=3。
#   - 这样可以避免线性-only 或过低自由度模型对非线性结果解释造成干扰。

SENSITIVITY_ROOT_DIR = os.path.join(
    BASE_DIR,
    "ABIDE2_优先级4_ASD_非线性Spline_多参数敏感性分析_noFD_noSite_df4to10_HC1"
)

BASE_NOFD_NOSITE_SPEC = {
    "covariates": ["Age", "Sex_for_model", "FIQ"],
    "use_site_fixed_effect": False,
    "min_site_n": 5,
    "cv_n_folds": 5,
    "cv_random_seed": 20260615,
    "spline_degree": 3,
    "plot_x_lower_q": 0.01,
    "plot_x_upper_q": 0.99,
    "min_n_for_model": 30,
}

SENSITIVITY_ANALYSIS_SPECS = [
    {
        **BASE_NOFD_NOSITE_SPEC,
        "analysis_set_name": "S0_main_noFD_noSite_AIC_df4to10",
        "spline_df_candidates": list(range(4, 11)),
        "spline_df_selection_method": "AIC",
        "note": "主分析扩展版：Age + Sex + FIQ；不加入 MeanFD 和 Site fixed effect；AIC 在 df=4--10 中选择 spline df；df>=4 使用 cubic B-spline degree=3。",
    },
    {
        **BASE_NOFD_NOSITE_SPEC,
        "analysis_set_name": "S1_noFD_noSite_BIC_df4to10",
        "spline_df_candidates": list(range(4, 11)),
        "spline_df_selection_method": "BIC",
        "note": "敏感性分析：不加入 MeanFD 和 Site fixed effect；BIC 在 df=4--10 中选择 spline df；df>=4 使用 cubic B-spline degree=3。",
    },
    {
        **BASE_NOFD_NOSITE_SPEC,
        "analysis_set_name": "S2_noFD_noSite_CVRMSE_df4to10",
        "spline_df_candidates": list(range(4, 11)),
        "spline_df_selection_method": "CV_RMSE",
        "note": "敏感性分析：不加入 MeanFD 和 Site fixed effect；5-fold CV_RMSE 在 df=4--10 中选择 spline df；df>=4 使用 cubic B-spline degree=3。",
    },
]

# 固定 df=4--10 的敏感性分析。
# 不再生成固定 df=1、df=2、df=3 的边缘自由度模型。
SENSITIVITY_ANALYSIS_SPECS += [
    {
        **BASE_NOFD_NOSITE_SPEC,
        "analysis_set_name": f"S{fixed_df - 1}_noFD_noSite_fixed_df{fixed_df}",
        "spline_df_candidates": [fixed_df],
        "spline_df_selection_method": "AIC",
        "note": (
            f"敏感性分析：固定 spline df={fixed_df}；不加入 MeanFD 和 Site fixed effect；"
            "使用 cubic B-spline degree=3。"
        ),
    }
    for fixed_df in range(4, 11)
]

# 额外保留 quadratic B-spline 参数组，作为 spline degree 的敏感性分析。
# 该组同样只在 df=4--10 中选择，不再包含 df=1--3。
SENSITIVITY_ANALYSIS_SPECS += [
    {
        **BASE_NOFD_NOSITE_SPEC,
        "analysis_set_name": "S10_noFD_noSite_quadraticSpline_AIC_df4to10",
        "spline_df_candidates": list(range(4, 11)),
        "spline_df_selection_method": "AIC",
        "spline_degree": 2,
        "note": "敏感性分析：使用 quadratic B-spline degree=2；不加入 MeanFD 和 Site fixed effect；AIC 在 df=4--10 中选择 df。",
    }
]
# 以下变量会在每组敏感性分析运行前由 apply_sensitivity_spec() 自动更新。
CURRENT_SENSITIVITY_SPEC = {}

ANALYSIS_SET_NAME = "S0_main_noFD_noSite_AIC_df4to10"

COVARIATES_BASE = [
    "Age",
    "Sex_for_model",
    "FIQ",
]

SITE_COL = "Site"
GROUP_COL = "Group"
ID_COL = "sub_id"

MIN_N_FOR_MODEL = 30
USE_SITE_FIXED_EFFECT = False
MIN_SITE_N = 5

SPLINE_DF_CANDIDATES = list(range(4, 11))
SPLINE_DF_SELECTION_METHOD = "AIC"
CV_N_FOLDS = 5
CV_RANDOM_SEED = 20260615

SPLINE_DEGREE = 3

# 作图范围使用 1% 到 99% 分位数，避免极端值拉长曲线。
PLOT_X_LOWER_Q = 0.01
PLOT_X_UPPER_Q = 0.99

MISSING_VALUES = [-9999, -999, 9999, 999, "-9999", "-999", "9999", "999", "NA", "N/A", "", "nan", "NaN"]


# ============================================================
# 3. 基础工具函数
# ============================================================

def write_text(path, text):
    with open(path, "w", encoding="utf-8") as f:
        f.write(text)


def clean_columns(df):
    df = df.copy()
    df.columns = [str(c).strip() for c in df.columns]
    return df


def replace_missing_values(df):
    df = df.copy()
    df = df.replace(MISSING_VALUES, np.nan)
    return df


def safe_filename(x):
    x = str(x)
    x = re.sub(r"[^\w\-.]+", "_", x)
    x = x.strip("_")
    return x


def list_to_semicolon_string(x):
    if isinstance(x, (list, tuple)):
        return ";".join([str(v) for v in x])
    return str(x)


def get_run_file_prefix():
    return safe_filename(ANALYSIS_SET_NAME)


def apply_sensitivity_spec(spec):
    """
    将一组敏感性分析参数写入全局变量。
    这样后续所有函数仍可沿用原脚本结构，但每轮运行参数不同。
    """
    global CURRENT_SENSITIVITY_SPEC
    global ANALYSIS_SET_NAME, COVARIATES_BASE
    global USE_SITE_FIXED_EFFECT, MIN_SITE_N, MIN_N_FOR_MODEL
    global SPLINE_DF_CANDIDATES, SPLINE_DF_SELECTION_METHOD
    global CV_N_FOLDS, CV_RANDOM_SEED, SPLINE_DEGREE
    global PLOT_X_LOWER_Q, PLOT_X_UPPER_Q
    global OUT_DIR, PLOT_SPLINE_DIR, MERGED_MODEL_INPUT_DIR

    CURRENT_SENSITIVITY_SPEC = dict(spec)

    ANALYSIS_SET_NAME = str(spec["analysis_set_name"])
    COVARIATES_BASE = list(spec.get("covariates", ["Age", "Sex_for_model", "FIQ"]))

    USE_SITE_FIXED_EFFECT = bool(spec.get("use_site_fixed_effect", False))
    MIN_SITE_N = int(spec.get("min_site_n", 5))
    MIN_N_FOR_MODEL = int(spec.get("min_n_for_model", 30))

    SPLINE_DF_CANDIDATES = list(spec.get("spline_df_candidates", [4, 5, 6, 7, 8, 9, 10]))
    SPLINE_DF_SELECTION_METHOD = str(spec.get("spline_df_selection_method", "AIC")).upper()
    CV_N_FOLDS = int(spec.get("cv_n_folds", 5))
    CV_RANDOM_SEED = int(spec.get("cv_random_seed", 20260615))
    SPLINE_DEGREE = int(spec.get("spline_degree", 3))

    PLOT_X_LOWER_Q = float(spec.get("plot_x_lower_q", 0.01))
    PLOT_X_UPPER_Q = float(spec.get("plot_x_upper_q", 0.99))

    OUT_DIR = os.path.join(SENSITIVITY_ROOT_DIR, get_run_file_prefix())
    PLOT_SPLINE_DIR = os.path.join(OUT_DIR, "plots_adjusted_spline_curve")
    MERGED_MODEL_INPUT_DIR = os.path.join(OUT_DIR, "merged_model_inputs")

    Path(OUT_DIR).mkdir(parents=True, exist_ok=True)
    Path(PLOT_SPLINE_DIR).mkdir(parents=True, exist_ok=True)
    Path(MERGED_MODEL_INPUT_DIR).mkdir(parents=True, exist_ok=True)


def current_sensitivity_metadata():
    """返回当前参数组元信息，用于写入跨模型汇总表。"""
    return {
        "analysis_set": ANALYSIS_SET_NAME,
        "sensitivity_note": CURRENT_SENSITIVITY_SPEC.get("note", ""),
        "covariates_used": list_to_semicolon_string(COVARIATES_BASE),
        "use_site_fixed_effect": USE_SITE_FIXED_EFFECT,
        "min_site_n": MIN_SITE_N,
        "min_n_for_model": MIN_N_FOR_MODEL,
        "spline_df_candidates": list_to_semicolon_string(SPLINE_DF_CANDIDATES),
        "spline_df_selection_method": SPLINE_DF_SELECTION_METHOD,
        "cv_n_folds": CV_N_FOLDS,
        "cv_random_seed": CV_RANDOM_SEED,
        "spline_degree": SPLINE_DEGREE,
        "plot_x_lower_q": PLOT_X_LOWER_Q,
        "plot_x_upper_q": PLOT_X_UPPER_Q,
    }


def normalize_sub_id(x):
    """把各种 ID 格式统一为 sub-Sub00000。"""
    if pd.isna(x):
        return np.nan

    s = str(x).strip()

    if re.fullmatch(r"sub-Sub\d{5}", s):
        return s

    m = re.search(r"Sub0*(\d+)", s, flags=re.IGNORECASE)
    if m:
        return f"sub-Sub{int(m.group(1)):05d}"

    if re.fullmatch(r"\d+(\.0)?", s):
        return f"sub-Sub{int(float(s)):05d}"

    return np.nan


def force_numeric(df, cols):
    df = df.copy()
    for col in cols:
        if col in df.columns:
            df[col] = pd.to_numeric(df[col], errors="coerce")
    return df


def q(col):
    """Patsy 公式安全列名。"""
    return 'Q("' + str(col).replace('"', r'\"') + '")'


def safe_group_label(x):
    if pd.isna(x):
        return np.nan
    s = str(x).strip().upper()
    if s in ["1", "ASD", "AUTISM", "AUTISM SPECTRUM DISORDER"]:
        return "ASD"
    if s in ["2", "HC", "CONTROL", "TD", "TDC"]:
        return "HC"
    return np.nan


# ============================================================
# 4. 读取脑指标表和文件夹中的所有行为 CSV
# ============================================================

def load_brain_table(path):
    if not os.path.exists(path):
        raise FileNotFoundError(f"Brain input file not found: {path}")

    df = pd.read_csv(path, low_memory=False)
    df = clean_columns(df)
    df = replace_missing_values(df)

    if ID_COL not in df.columns:
        raise ValueError(f"Brain table must contain column: {ID_COL}")

    df[ID_COL] = df[ID_COL].apply(normalize_sub_id)
    df = df.dropna(subset=[ID_COL]).copy()

    if GROUP_COL in df.columns:
        df[GROUP_COL] = df[GROUP_COL].map(safe_group_label)

    # 如果没有 Sex_for_model，但有 Sex_bin 或 Sex，则自动生成。
    if "Sex_for_model" not in df.columns:
        if "Sex_bin" in df.columns:
            df["Sex_for_model"] = pd.to_numeric(df["Sex_bin"], errors="coerce")
        elif "Sex" in df.columns:
            df["Sex_for_model"] = pd.to_numeric(df["Sex"], errors="coerce")

    required_cols = [ID_COL] + BRAIN_PREDICTORS + COVARIATES_BASE
    if USE_SITE_FIXED_EFFECT:
        required_cols.append(SITE_COL)

    missing = sorted([c for c in set(required_cols) if c not in df.columns])
    if missing:
        raise ValueError(
            "Brain table is missing critical columns: " + "; ".join(missing) + "\n"
            "Please check result4_brain_subject_metrics_for_behavior.csv."
        )

    numeric_cols = BRAIN_PREDICTORS + COVARIATES_BASE
    df = force_numeric(df, [c for c in numeric_cols if c in df.columns])

    # 每个被试一行。
    df = df.drop_duplicates(subset=[ID_COL], keep="first").copy()

    return df


def find_subject_id_series(df, path):
    """从单量表 CSV 中识别被试 ID 列。"""
    if ID_COL in df.columns:
        return df[ID_COL]
    if "SUB_ID" in df.columns:
        return df["SUB_ID"]
    if "FILE_ID" in df.columns:
        return df["FILE_ID"]

    raise ValueError(
        f"No subject ID column found in {path}. "
        "Expected one of: sub_id, SUB_ID, FILE_ID."
    )


def discover_behavior_csv_specs(folder, recursive=False):
    """
    扫描文件夹中的所有 CSV，并自动生成待分析量表清单。

    规则：
    1. 跳过 EXCLUDE_FILE_KEYWORDS 中关键词匹配到的文件；
    2. 每个 CSV 必须有 sub_id / SUB_ID / FILE_ID 中至少一个 ID 列；
    3. 排除 NON_BEHAVIOR_COLUMNS 中的基础列；
    4. 剩余列只要可转换为数值且至少有 1 个非缺失值，就作为一个行为量表列；
    5. 如果不同文件出现同名量表列，会自动给 behavior_var 加上文件名前缀以避免覆盖。
    """
    folder_path = Path(folder)
    if not folder_path.exists():
        raise FileNotFoundError(f"Behavior CSV folder not found: {folder}")
    if not folder_path.is_dir():
        raise NotADirectoryError(f"Behavior CSV path is not a folder: {folder}")

    pattern = "**/*.csv" if recursive else "*.csv"
    csv_paths = sorted(folder_path.glob(pattern))

    if EXCLUDE_FILE_KEYWORDS:
        csv_paths = [
            p for p in csv_paths
            if not any(k.lower() in p.name.lower() for k in EXCLUDE_FILE_KEYWORDS)
        ]

    specs = []
    used_behavior_vars = set()
    discovery_rows = []

    for csv_path in csv_paths:
        try:
            df = pd.read_csv(csv_path, low_memory=False)
            df = clean_columns(df)
            df = replace_missing_values(df)

            # 检查 ID 列是否存在。
            _ = find_subject_id_series(df, str(csv_path))

            candidate_cols = []
            for col in df.columns:
                col_str = str(col).strip()
                if col_str in NON_BEHAVIOR_COLUMNS:
                    continue
                if col_str.startswith("Unnamed"):
                    continue

                x = pd.to_numeric(df[col_str], errors="coerce")
                n_nonmissing = int(x.notna().sum())
                sd = x.std(skipna=True, ddof=0)

                if n_nonmissing <= 0:
                    discovery_rows.append({
                        "source_csv": str(csv_path),
                        "value_col": col_str,
                        "selected": False,
                        "reason": "not_numeric_or_all_missing",
                        "N_nonmissing": n_nonmissing,
                        "sd": sd,
                    })
                    continue

                candidate_cols.append((col_str, n_nonmissing, sd))

            for value_col, n_nonmissing, sd in candidate_cols:
                base_name = safe_filename(value_col)
                behavior_var = base_name
                if behavior_var in used_behavior_vars:
                    behavior_var = f"{safe_filename(csv_path.stem)}__{base_name}"

                # 极端情况下仍重复时继续加编号。
                counter = 2
                unique_name = behavior_var
                while unique_name in used_behavior_vars:
                    unique_name = f"{behavior_var}_{counter}"
                    counter += 1
                behavior_var = unique_name
                used_behavior_vars.add(behavior_var)

                specs.append({
                    "behavior_var": behavior_var,
                    "value_col": value_col,
                    "path": str(csv_path),
                    "source_file": csv_path.name,
                    "source_stem": csv_path.stem,
                })

                discovery_rows.append({
                    "source_csv": str(csv_path),
                    "value_col": value_col,
                    "behavior_var": behavior_var,
                    "selected": True,
                    "reason": "selected_numeric_behavior_column",
                    "N_nonmissing": n_nonmissing,
                    "sd": sd,
                })

        except Exception as e:
            discovery_rows.append({
                "source_csv": str(csv_path),
                "value_col": "",
                "behavior_var": "",
                "selected": False,
                "reason": f"file_error: {repr(e)}",
                "N_nonmissing": 0,
                "sd": np.nan,
            })

    discovery_df = pd.DataFrame(discovery_rows)

    if len(specs) == 0:
        raise ValueError(
            "No analyzable numeric behavior columns were discovered in the CSV folder. "
            "Check BEHAVIOR_CSV_DIR, ID columns, and NON_BEHAVIOR_COLUMNS."
        )

    return specs, discovery_df


def load_single_behavior_scale(scale_spec):
    """读取一个 CSV 中的一个量表列，生成 sub_id + behavior_var 的两列表。"""
    path = scale_spec["path"]
    behavior_var = scale_spec["behavior_var"]
    value_col = scale_spec["value_col"]

    if not os.path.exists(path):
        raise FileNotFoundError(f"Behavior scale file not found: {path}")

    df = pd.read_csv(path, low_memory=False)
    df = clean_columns(df)
    df = replace_missing_values(df)

    if value_col not in df.columns:
        raise ValueError(
            f"Column {value_col} not found in {path}. "
            f"Available columns: {list(df.columns)}"
        )

    id_raw = find_subject_id_series(df, path)

    out = pd.DataFrame()
    out[ID_COL] = id_raw.apply(normalize_sub_id)
    out[behavior_var] = pd.to_numeric(df[value_col], errors="coerce")

    # 保留一些审计字段，不进入模型。
    if "Dataset" in df.columns:
        out[f"Dataset_{behavior_var}"] = df["Dataset"]
    if "SITE_ID" in df.columns:
        out[f"SITE_ID_{behavior_var}"] = df["SITE_ID"]
    if "SUB_ID" in df.columns:
        out[f"SUB_ID_{behavior_var}"] = df["SUB_ID"]
    if "FILE_ID" in df.columns:
        out[f"FILE_ID_{behavior_var}"] = df["FILE_ID"]

    out = out.dropna(subset=[ID_COL]).copy()
    out = out.dropna(subset=[behavior_var]).copy()

    # 如果重复，被试只保留第一条非缺失记录。
    out = out.drop_duplicates(subset=[ID_COL], keep="first").copy()

    return out


def merge_brain_with_behavior(brain_df, behavior_df, behavior_var):
    """
    单个量表与脑指标表按 sub_id 匹配。
    这里的 merge 不是合并不同量表，而是同一被试的脑指标与该行为量表匹配。
    """
    merged = brain_df.merge(behavior_df, on=ID_COL, how="inner")

    # 如果脑表没有 Site，但行为表里有 SITE_ID，可用于兜底。
    site_from_behavior = f"SITE_ID_{behavior_var}"
    if SITE_COL not in merged.columns and site_from_behavior in merged.columns:
        merged[SITE_COL] = merged[site_from_behavior]

    return merged


# ============================================================
# 5. ASD 筛选和 Site 处理
# ============================================================

def filter_asd(df):
    """只保留 ASD 被试。单量表 CSV 已是 ASD-only，但这里再做一次保险筛选。"""
    df = df.copy()

    if GROUP_COL in df.columns:
        out = df[df[GROUP_COL].astype(str).str.upper() == "ASD"].copy()
        filter_note = "Filtered by Group == ASD."
    elif "Group_behavior" in df.columns:
        out = df[df["Group_behavior"].astype(str).str.upper() == "ASD"].copy()
        filter_note = "Filtered by Group_behavior == ASD."
    else:
        out = df.copy()
        filter_note = "No Group column found. No additional ASD filtering applied; input scale files are assumed ASD-only."

    return out, filter_note


def collapse_small_sites(df, site_col=SITE_COL, min_site_n=MIN_SITE_N):
    """在当前模型样本中将人数过少的 Site 合并为 Site_Other。"""
    df = df.copy()

    if not USE_SITE_FIXED_EFFECT:
        df["Site_for_model"] = "Site_not_used"
        return df, "Site fixed effect not used."

    if site_col not in df.columns:
        df["Site_for_model"] = "Site_missing"
        return df, "Site column not found. Site_for_model set to Site_missing."

    site = df[site_col].astype(str).replace({
        "nan": "Site_missing",
        "None": "Site_missing",
        "": "Site_missing",
    })

    counts = site.value_counts(dropna=False)
    small_sites = counts[counts < min_site_n].index.tolist()

    df["Site_for_model"] = site.where(~site.isin(small_sites), "Site_Other")

    note = (
        f"Site_for_model created from {site_col}; "
        f"{len(small_sites)} site levels with N < {min_site_n} collapsed to Site_Other."
    )

    return df, note


# ============================================================
# 6. 模型数据构建
# ============================================================

def build_model_dataframe(df, behavior_var, brain_var):
    """构建单个行为变量的 complete-case 模型样本。"""
    needed_cols = [ID_COL, behavior_var, brain_var] + COVARIATES_BASE

    if USE_SITE_FIXED_EFFECT:
        needed_cols.append(SITE_COL)

    existing_cols = [c for c in needed_cols if c in df.columns]
    missing_cols = [c for c in needed_cols if c not in df.columns]

    model_df = df[existing_cols].copy()

    numeric_cols = [behavior_var, brain_var] + [
        c for c in COVARIATES_BASE if c in model_df.columns
    ]

    model_df = force_numeric(model_df, numeric_cols)

    drop_cols = [behavior_var, brain_var] + [
        c for c in COVARIATES_BASE if c in model_df.columns
    ]

    model_df = model_df.dropna(subset=drop_cols).copy()

    site_note = "Site not used."
    if USE_SITE_FIXED_EFFECT:
        model_df, site_note = collapse_small_sites(model_df, SITE_COL, MIN_SITE_N)

    return model_df, missing_cols, site_note


# ============================================================
# 7. 非线性 spline 特征构建
# ============================================================

def effective_bs_degree_for_df(spline_df):
    """
    返回当前 df 下可用的 B-spline degree。

    patsy 的 bs(..., include_intercept=False) 要求 df >= degree。
    因此：
      - df=1：作为线性-only 基线模型处理；实际 degree=1，但经正交化后没有非线性成分；
      - df=2：最高只能使用 degree=2；
      - df>=3：使用目标 SPLINE_DEGREE，通常为 cubic degree=3。
    """
    spline_df = int(spline_df)
    if spline_df <= 1:
        return 1
    return int(min(SPLINE_DEGREE, spline_df))


def add_orthogonal_nonlinear_spline_features(model_df, brain_var, spline_df):
    """
    构建非线性 spline 特征。

    做法：
    1. 对 Coupling_index_scaled 构建 B-spline basis；
    2. 将 spline basis 中可由 intercept 和 linear Coupling 解释的部分去掉；
    3. 剩余部分作为纯非线性成分 NL_spline_1, NL_spline_2, ...；
    4. 模型中同时放入 Coupling_index_scaled + NL_spline_*。

    当 spline_df=1 时，模型退化为线性-only 基线模型，不生成非线性成分。
    """
    df = model_df.copy()

    spline_df = int(spline_df)
    effective_degree = effective_bs_degree_for_df(spline_df)

    x = pd.to_numeric(df[brain_var], errors="coerce").values.astype(float)

    if np.isnan(x).any():
        raise ValueError("Brain variable contains NaN after model dataframe construction.")

    if np.std(x) == 0:
        raise ValueError("Brain variable has zero variance.")

    if spline_df <= 1:
        transform = {
            "design_info": None,
            "coef_basis_on_linear": None,
            "v_keep": np.zeros((0, 0)),
            "brain_var": brain_var,
            "spline_df": spline_df,
            "target_spline_degree": SPLINE_DEGREE,
            "effective_spline_degree": 0,
            "linear_only": True,
        }
        spline_note = (
            f"Linear-only model for df={spline_df}; "
            f"target B-spline degree={SPLINE_DEGREE}; "
            "orthogonal nonlinear components=0."
        )
        return df, [], transform, spline_note

    bs_formula = (
        f"bs(x, df={spline_df}, degree={effective_degree}, "
        f"include_intercept=False) - 1"
    )

    bs_df = dmatrix(
        bs_formula,
        {"x": x},
        return_type="dataframe"
    )

    bs_mat = np.asarray(bs_df)

    x_linear = np.column_stack([
        np.ones_like(x),
        x,
    ])

    coef_basis_on_linear = np.linalg.lstsq(
        x_linear,
        bs_mat,
        rcond=None
    )[0]

    bs_resid = bs_mat - x_linear @ coef_basis_on_linear

    u, s, vt = np.linalg.svd(bs_resid, full_matrices=False)

    if len(s) == 0 or np.all(s == 0):
        keep = np.array([], dtype=bool)
    else:
        tol = max(bs_resid.shape) * np.finfo(float).eps * max(s)
        keep = s > tol

    nl_names = []

    if keep.sum() == 0:
        # df=1 时通常进入这里：B-spline basis 与 intercept + linear x 完全共线。
        # 这不是错误，而是线性-only 基线模型。
        v_keep = np.zeros((0, bs_mat.shape[1]))
        nl_mat = np.zeros((len(x), 0))
    else:
        v_keep = vt[keep, :]
        nl_mat = bs_resid @ v_keep.T

        for j in range(nl_mat.shape[1]):
            col = f"NL_spline_{j + 1}"
            df[col] = nl_mat[:, j]
            nl_names.append(col)

    transform = {
        "design_info": bs_df.design_info,
        "coef_basis_on_linear": coef_basis_on_linear,
        "v_keep": v_keep,
        "brain_var": brain_var,
        "spline_df": spline_df,
        "target_spline_degree": SPLINE_DEGREE,
        "effective_spline_degree": effective_degree,
    }

    spline_note = (
        f"B-spline df={spline_df}, target degree={SPLINE_DEGREE}, "
        f"effective degree={effective_degree}; "
        f"raw basis columns={bs_mat.shape[1]}; "
        f"orthogonal nonlinear components={len(nl_names)}."
    )

    return df, nl_names, transform, spline_note

def add_spline_features_to_newdata(new_df, brain_var, transform, nl_names):
    """对预测用的新数据生成同样的非线性 spline 特征。"""
    df = new_df.copy()

    if transform.get("linear_only", False) or transform.get("design_info") is None or len(nl_names) == 0:
        return df

    x_new = pd.to_numeric(df[brain_var], errors="coerce").values.astype(float)

    bs_new = build_design_matrices(
        [transform["design_info"]],
        {"x": x_new}
    )[0]

    bs_new = np.asarray(bs_new)

    x_linear_new = np.column_stack([
        np.ones_like(x_new),
        x_new,
    ])

    bs_resid_new = bs_new - x_linear_new @ transform["coef_basis_on_linear"]
    nl_new = bs_resid_new @ transform["v_keep"].T

    for j, col in enumerate(nl_names):
        df[col] = nl_new[:, j]

    return df


# ============================================================
# 8. 模型公式与统计检验
# ============================================================

def formula_for_linear_model(behavior_var, brain_var, model_df):
    """线性模型公式，用于 AIC/R2 对照。"""
    terms = [q(brain_var)]

    for cov in COVARIATES_BASE:
        if cov in model_df.columns:
            terms.append(q(cov))

    if USE_SITE_FIXED_EFFECT and "Site_for_model" in model_df.columns:
        if model_df["Site_for_model"].nunique(dropna=True) > 1:
            terms.append("C(" + q("Site_for_model") + ")")

    return q(behavior_var) + " ~ " + " + ".join(terms)


def formula_for_spline_model(behavior_var, brain_var, model_df, nl_names):
    """非线性 spline 模型公式。"""
    terms = [q(brain_var)] + nl_names

    for cov in COVARIATES_BASE:
        if cov in model_df.columns:
            terms.append(q(cov))

    if USE_SITE_FIXED_EFFECT and "Site_for_model" in model_df.columns:
        if model_df["Site_for_model"].nunique(dropna=True) > 1:
            terms.append("C(" + q("Site_for_model") + ")")

    return q(behavior_var) + " ~ " + " + ".join(terms)


def get_param_index(result, var_name):
    """获取某个参数在模型中的位置。"""
    param_names = list(result.model.exog_names)
    candidates = [var_name, q(var_name), f'Q("{var_name}")']

    for cand in candidates:
        if cand in param_names:
            return param_names.index(cand)

    matches = [i for i, name in enumerate(param_names) if var_name in name]

    if len(matches) == 1:
        return matches[0]

    raise ValueError(
        f"Parameter {var_name} not found in fitted model parameters. "
        f"Available parameters: {param_names}"
    )


def wald_test_indices(robust_result, idx_list):
    """对一组参数做 robust Wald joint test。"""
    idx_list = list(idx_list)

    if len(idx_list) == 0:
        return np.nan, np.nan, 0, np.nan, "NA", "NA"

    n_params = len(robust_result.params)
    r_matrix = np.zeros((len(idx_list), n_params))

    for row_i, param_i in enumerate(idx_list):
        r_matrix[row_i, param_i] = 1.0

    wt = robust_result.wald_test(r_matrix, scalar=True)

    stat = float(np.ravel(wt.statistic)[0])
    p_value = float(wt.pvalue)
    df_constraint = len(idx_list)
    df_denom = float(getattr(wt, "df_denom", robust_result.df_resid))
    distribution = str(getattr(wt, "distribution", "F"))
    statistic_name = "F" if distribution.upper() == "F" else "chi2"

    return stat, p_value, df_constraint, df_denom, distribution, statistic_name


def format_wald_test_report(statistic_name, statistic_value, df_num, df_den, p_value):
    """Return a publication-ready Wald test string with explicit distribution and df."""
    if pd.isna(statistic_value) or pd.isna(p_value) or int(df_num) == 0:
        return "HC1 robust Wald nonlinear test not applicable: no nonlinear spline components."

    if statistic_name == "F" and pd.notna(df_den):
        return (
            f"HC1 robust Wald F({int(df_num)},{int(round(df_den))}) = "
            f"{statistic_value:.2f}, p = {p_value:.3g}"
        )

    return (
        f"HC1 robust Wald chi2({int(df_num)}) = "
        f"{statistic_value:.2f}, p = {p_value:.3g}"
    )


def fit_spline_model_hc1(model_df, behavior_var, brain_var, nl_names):
    """
    拟合线性模型和非线性 spline 模型。

    返回：
    - linear_fit
    - spline_fit
    - spline_robust
    - 关键统计量
    """
    linear_formula = formula_for_linear_model(behavior_var, brain_var, model_df)
    spline_formula = formula_for_spline_model(behavior_var, brain_var, model_df, nl_names)

    linear_fit = smf.ols(formula=linear_formula, data=model_df).fit()

    spline_model = smf.ols(formula=spline_formula, data=model_df)
    spline_fit = spline_model.fit()
    spline_robust = spline_fit.get_robustcov_results(cov_type="HC1")

    param_names = list(spline_robust.model.exog_names)

    linear_idx = get_param_index(spline_robust, brain_var)

    nl_idx = []
    for name in nl_names:
        if name in param_names:
            nl_idx.append(param_names.index(name))

    coupling_idx = [linear_idx] + nl_idx

    (
        overall_stat,
        overall_p,
        overall_df,
        overall_df_denom,
        overall_distribution,
        overall_statistic_name,
    ) = wald_test_indices(
        spline_robust,
        coupling_idx
    )

    (
        nonlinear_stat,
        nonlinear_p,
        nonlinear_df,
        nonlinear_df_denom,
        nonlinear_distribution,
        nonlinear_statistic_name,
    ) = wald_test_indices(
        spline_robust,
        nl_idx
    )

    beta_linear = float(spline_robust.params[linear_idx])
    se_linear = float(spline_robust.bse[linear_idx])
    t_linear = float(spline_robust.tvalues[linear_idx])
    p_linear = float(spline_robust.pvalues[linear_idx])
    ci_linear_low, ci_linear_high = spline_robust.conf_int()[linear_idx]

    out = {
        "linear_formula": linear_formula,
        "spline_formula": spline_formula,

        "beta_linear_component": beta_linear,
        "se_linear_component_hc1": se_linear,
        "t_linear_component_hc1": t_linear,
        "p_linear_component_hc1": p_linear,
        "ci95_linear_component_low": float(ci_linear_low),
        "ci95_linear_component_high": float(ci_linear_high),

        "wald_overall_coupling_hc1": overall_stat,
        "p_overall_coupling_hc1": overall_p,
        "df_overall_coupling": overall_df,
        "wald_test_type_overall_coupling_hc1": f"HC1 robust Wald {overall_statistic_name}",
        "wald_statistic_name_overall_coupling_hc1": overall_statistic_name,
        "wald_distribution_overall_coupling_hc1": overall_distribution,
        "wald_f_overall_coupling_hc1": overall_stat if overall_statistic_name == "F" else np.nan,
        "df_num_overall_coupling": overall_df,
        "df_den_overall_coupling": overall_df_denom,
        "p_exact_overall_coupling_hc1": overall_p,
        "wald_report_overall_coupling_hc1": format_wald_test_report(
            overall_statistic_name,
            overall_stat,
            overall_df,
            overall_df_denom,
            overall_p,
        ),

        "wald_nonlinear_hc1": nonlinear_stat,
        "p_nonlinear_hc1": nonlinear_p,
        "df_nonlinear": nonlinear_df,
        "wald_test_type_nonlinear_hc1": f"HC1 robust Wald {nonlinear_statistic_name}",
        "wald_statistic_name_nonlinear_hc1": nonlinear_statistic_name,
        "wald_distribution_nonlinear_hc1": nonlinear_distribution,
        "wald_f_nonlinear_hc1": nonlinear_stat if nonlinear_statistic_name == "F" else np.nan,
        "df_num_nonlinear": nonlinear_df,
        "df_den_nonlinear": nonlinear_df_denom,
        "p_exact_nonlinear_hc1": nonlinear_p,
        "wald_report_nonlinear_hc1": format_wald_test_report(
            nonlinear_statistic_name,
            nonlinear_stat,
            nonlinear_df,
            nonlinear_df_denom,
            nonlinear_p,
        ),

        "linear_r2": float(linear_fit.rsquared),
        "linear_adj_r2": float(linear_fit.rsquared_adj),
        "linear_aic": float(linear_fit.aic),
        "linear_bic": float(linear_fit.bic),

        "spline_r2": float(spline_fit.rsquared),
        "spline_adj_r2": float(spline_fit.rsquared_adj),
        "spline_aic": float(spline_fit.aic),
        "spline_bic": float(spline_fit.bic),

        "delta_aic_linear_minus_spline": float(linear_fit.aic - spline_fit.aic),
        "delta_bic_linear_minus_spline": float(linear_fit.bic - spline_fit.bic),

        "df_model_spline": float(spline_fit.df_model),
        "df_resid_spline": float(spline_fit.df_resid),
    }

    return linear_fit, spline_fit, spline_robust, out


# ============================================================
# 8b. spline df 选择：AIC / BIC / K 折交叉验证
# ============================================================

def make_kfold_indices(n, n_splits=5, seed=20260615):
    """生成 K 折交叉验证的测试集索引。"""
    if n_splits < 2:
        raise ValueError("CV_N_FOLDS must be at least 2.")

    n_splits = min(int(n_splits), int(n))
    rng = np.random.default_rng(seed)
    indices = np.arange(n)
    rng.shuffle(indices)

    folds = np.array_split(indices, n_splits)
    return [fold for fold in folds if len(fold) > 0]


def cv_rmse_for_formula(model_df, formula, behavior_var, n_splits=5, seed=20260615):
    """
    对已经生成 spline 特征的 model_df 做 K 折交叉验证。

    注意：spline basis 在全样本 X 上生成，然后只用训练折拟合 y。
    这避免了测试折 x 超出训练折 knot 范围时 patsy bs() 预测失败。
    该过程没有使用测试折 y 来生成 spline basis。
    """
    df = model_df.copy().reset_index(drop=True)

    # 固定 Site 类别水平，避免某个训练折缺少某个站点水平时 predict 报错。
    if USE_SITE_FIXED_EFFECT and "Site_for_model" in df.columns:
        categories = sorted(df["Site_for_model"].dropna().astype(str).unique().tolist())
        df["Site_for_model"] = pd.Categorical(
            df["Site_for_model"].astype(str),
            categories=categories
        )

    y_true_all = []
    y_pred_all = []

    folds = make_kfold_indices(
        n=df.shape[0],
        n_splits=n_splits,
        seed=seed
    )

    for test_idx in folds:
        train_idx = np.setdiff1d(np.arange(df.shape[0]), test_idx)
        train_df = df.iloc[train_idx].copy()
        test_df = df.iloc[test_idx].copy()

        fit = smf.ols(formula=formula, data=train_df).fit()
        pred = fit.predict(test_df)

        y_true_all.append(pd.to_numeric(test_df[behavior_var], errors="coerce").values.astype(float))
        y_pred_all.append(np.asarray(pred).astype(float))

    y_true = np.concatenate(y_true_all)
    y_pred = np.concatenate(y_pred_all)

    resid = y_true - y_pred
    rmse = float(np.sqrt(np.mean(resid ** 2)))
    mae = float(np.mean(np.abs(resid)))

    return rmse, mae


def select_spline_df(model_df, behavior_var, brain_var):
    """
    在 SPLINE_DF_CANDIDATES 中选择 spline df。

    选择标准由 SPLINE_DF_SELECTION_METHOD 控制：
    - AIC：选择全样本 spline 模型 AIC 最小者；
    - BIC：选择全样本 spline 模型 BIC 最小者；
    - CV_RMSE：选择 K 折交叉验证 RMSE 最小者。
    """
    rows = []

    for candidate_df in SPLINE_DF_CANDIDATES:
        row = {
            "candidate_spline_df": int(candidate_df),
            "selection_method": SPLINE_DF_SELECTION_METHOD,
            "cv_n_folds": CV_N_FOLDS,
            "status": "",
            "candidate_aic": np.nan,
            "candidate_bic": np.nan,
            "candidate_r2": np.nan,
            "candidate_adj_r2": np.nan,
            "candidate_cv_rmse": np.nan,
            "candidate_cv_mae": np.nan,
            "candidate_n_nonlinear_spline_components": np.nan,
            "candidate_spline_formula": "",
            "error": "",
        }

        try:
            model_df_spline, nl_names, _, _ = add_orthogonal_nonlinear_spline_features(
                model_df,
                brain_var=brain_var,
                spline_df=int(candidate_df)
            )

            _, spline_fit, _, stat = fit_spline_model_hc1(
                model_df=model_df_spline,
                behavior_var=behavior_var,
                brain_var=brain_var,
                nl_names=nl_names
            )

            cv_rmse, cv_mae = cv_rmse_for_formula(
                model_df=model_df_spline,
                formula=stat["spline_formula"],
                behavior_var=behavior_var,
                n_splits=CV_N_FOLDS,
                seed=CV_RANDOM_SEED
            )

            row.update({
                "status": "ok",
                "candidate_aic": float(spline_fit.aic),
                "candidate_bic": float(spline_fit.bic),
                "candidate_r2": float(spline_fit.rsquared),
                "candidate_adj_r2": float(spline_fit.rsquared_adj),
                "candidate_cv_rmse": cv_rmse,
                "candidate_cv_mae": cv_mae,
                "candidate_n_nonlinear_spline_components": int(len(nl_names)),
                "candidate_spline_formula": stat["spline_formula"],
            })

        except Exception as e:
            row["status"] = "error"
            row["error"] = repr(e)

        rows.append(row)

    selection_df = pd.DataFrame(rows)
    ok_df = selection_df[selection_df["status"] == "ok"].copy()

    if ok_df.empty:
        raise ValueError(
            "All candidate spline df values failed. "
            "Check SPLINE_DF_CANDIDATES, sample size, and model specification."
        )

    method = str(SPLINE_DF_SELECTION_METHOD).upper()

    if method == "AIC":
        sort_cols = ["candidate_aic", "candidate_spline_df"]
        ascending = [True, True]
        criterion_col = "candidate_aic"
    elif method == "BIC":
        sort_cols = ["candidate_bic", "candidate_spline_df"]
        ascending = [True, True]
        criterion_col = "candidate_bic"
    elif method == "CV_RMSE":
        sort_cols = ["candidate_cv_rmse", "candidate_spline_df"]
        ascending = [True, True]
        criterion_col = "candidate_cv_rmse"
    else:
        raise ValueError(
            "SPLINE_DF_SELECTION_METHOD must be one of: AIC, BIC, CV_RMSE. "
            f"Got: {SPLINE_DF_SELECTION_METHOD}"
        )

    best = ok_df.sort_values(sort_cols, ascending=ascending).iloc[0]
    selected_df = int(best["candidate_spline_df"])

    selection_df["selected_spline_df"] = selected_df
    selection_df["selected_by_method"] = method
    selection_df["is_selected"] = selection_df["candidate_spline_df"] == selected_df

    selection_note = (
        f"Selected spline df={selected_df} by {method}; "
        f"criterion {criterion_col}={best[criterion_col]:.6g}; "
        f"candidates={SPLINE_DF_CANDIDATES}."
    )

    selected_info = {
        "selected_spline_df": selected_df,
        "spline_df_selection_method": method,
        "selected_spline_df_aic": float(best["candidate_aic"]),
        "selected_spline_df_bic": float(best["candidate_bic"]),
        "selected_spline_df_cv_rmse": float(best["candidate_cv_rmse"]),
        "selected_spline_df_cv_mae": float(best["candidate_cv_mae"]),
        "spline_df_selection_note": selection_note,
    }

    return selected_df, selection_df, selection_note, selected_info


# ============================================================
# 9. 未校正 p 值汇总
# ============================================================

def make_raw_p_summary(results_df):
    """生成未进行多重比较校正的结果汇总表。"""
    if results_df.empty:
        return pd.DataFrame()

    cols = [
        "analysis_set",
        "behavior_var",
        "brain_predictor",
        "N",

        "N_behavior_file",
        "N_after_brain_behavior_match",
        "N_after_ASD_filter",

        "beta_linear_component",
        "se_linear_component_hc1",
        "t_linear_component_hc1",
        "p_linear_component_hc1",

        "wald_overall_coupling_hc1",
        "wald_test_type_overall_coupling_hc1",
        "wald_statistic_name_overall_coupling_hc1",
        "wald_f_overall_coupling_hc1",
        "df_num_overall_coupling",
        "df_den_overall_coupling",
        "p_overall_coupling_hc1",
        "p_exact_overall_coupling_hc1",
        "wald_report_overall_coupling_hc1",

        "wald_nonlinear_hc1",
        "wald_test_type_nonlinear_hc1",
        "wald_statistic_name_nonlinear_hc1",
        "wald_f_nonlinear_hc1",
        "df_num_nonlinear",
        "df_den_nonlinear",
        "p_nonlinear_hc1",
        "p_exact_nonlinear_hc1",
        "wald_report_nonlinear_hc1",

        "linear_r2",
        "spline_r2",
        "delta_aic_linear_minus_spline",
        "delta_bic_linear_minus_spline",

        "selected_spline_df",
        "spline_df_selection_method",
        "selected_spline_df_aic",
        "selected_spline_df_bic",
        "selected_spline_df_cv_rmse",
        "selected_spline_df_cv_mae",
        "spline_df_selection_note",

        "n_nonlinear_spline_components",
        "spline_note",
        "spline_formula",
        "adjusted_spline_plot",
    ]

    keep = [c for c in cols if c in results_df.columns]

    return results_df[keep].sort_values(
        by=["p_overall_coupling_hc1", "p_nonlinear_hc1"],
        ascending=[True, True]
    ).copy()


# ============================================================
# 10. 作图
# ============================================================

def save_adjusted_spline_plot(
    model_df,
    behavior_var,
    brain_var,
    spline_fit,
    spline_robust,
    spline_transform,
    nl_names,
    result_row,
):
    """
    保存调整协变量后的 spline 曲线图。

    图中：
    - 散点：原始行为分数 vs Coupling
    - 曲线：Age、Sex、FIQ 固定在中位数时的预测行为分数
    """
    x = pd.to_numeric(model_df[brain_var], errors="coerce")
    y = pd.to_numeric(model_df[behavior_var], errors="coerce")

    x_low = x.quantile(PLOT_X_LOWER_Q)
    x_high = x.quantile(PLOT_X_UPPER_Q)

    if pd.isna(x_low) or pd.isna(x_high) or x_low == x_high:
        x_low = x.min()
        x_high = x.max()

    x_grid = np.linspace(x_low, x_high, 200)

    grid_df = pd.DataFrame({
        brain_var: x_grid
    })

    for cov in COVARIATES_BASE:
        if cov in model_df.columns:
            grid_df[cov] = pd.to_numeric(model_df[cov], errors="coerce").median()

    if USE_SITE_FIXED_EFFECT and "Site_for_model" in model_df.columns:
        site_mode = model_df["Site_for_model"].mode(dropna=True)
        if len(site_mode) > 0:
            grid_df["Site_for_model"] = site_mode.iloc[0]
        else:
            grid_df["Site_for_model"] = "Site_missing"

    grid_df = add_spline_features_to_newdata(
        grid_df,
        brain_var=brain_var,
        transform=spline_transform,
        nl_names=nl_names
    )

    try:
        pred = spline_robust.get_prediction(grid_df).summary_frame(alpha=0.05)
    except Exception:
        pred = spline_fit.get_prediction(grid_df).summary_frame(alpha=0.05)

    fig, ax = plt.subplots(figsize=(6.5, 5.5))

    ax.scatter(x, y, alpha=0.60)
    ax.plot(x_grid, pred["mean"].values, linewidth=2)

    if "mean_ci_lower" in pred.columns and "mean_ci_upper" in pred.columns:
        ax.fill_between(
            x_grid,
            pred["mean_ci_lower"].values,
            pred["mean_ci_upper"].values,
            alpha=0.20
        )

    ax.set_xlabel(brain_var)
    ax.set_ylabel(behavior_var)

    ax.set_title(
        f"Adjusted nonlinear spline\n"
        f"{behavior_var} ~ {brain_var}\n"
        f"N={int(result_row['N'])}, "
        f"selected df={int(result_row.get('selected_spline_df', spline_transform.get('spline_df', -1)))}, "
        f"overall {result_row.get('wald_statistic_name_overall_coupling_hc1', 'Wald')}="
        f"{result_row['wald_overall_coupling_hc1']:.3g}, "
        f"p={result_row['p_overall_coupling_hc1']:.4g}"
    )

    fig.tight_layout()

    out_path = os.path.join(
        PLOT_SPLINE_DIR,
        f"spline_{safe_filename(behavior_var)}__{safe_filename(brain_var)}.png"
    )

    fig.savefig(out_path, dpi=300)
    plt.close(fig)

    return out_path


# ============================================================
# 11. QC report
# ============================================================

def write_qc_report(
    path,
    brain_input_file,
    scale_specs,
    brain_df,
    scale_audit_df,
    results_df,
    sample_df,
    spline_selection_df,
):
    lines = []

    lines.append("Priority 4 ASD-only nonlinear spline brain-behavior association QC report")
    lines.append("All CSV behavior scale-specific files in folder + Coupling_index_scaled only")
    lines.append(f"Generated at: {datetime.now()}")

    lines.append("")
    lines.append("=" * 80)
    lines.append("Input")
    lines.append("=" * 80)
    lines.append(f"Brain input file: {brain_input_file}")
    lines.append(f"Behavior CSV folder: {BEHAVIOR_CSV_DIR}")
    lines.append(f"Recursive scan: {SCAN_RECURSIVE}")
    lines.append(f"Brain rows: {brain_df.shape[0]}")
    lines.append(f"Brain unique subjects: {brain_df[ID_COL].nunique() if ID_COL in brain_df.columns else 'NA'}")

    if GROUP_COL in brain_df.columns:
        lines.append("")
        lines.append("Brain table Group counts:")
        for k, v in brain_df[GROUP_COL].value_counts(dropna=False).items():
            lines.append(f"  {k}: {int(v)}")

    lines.append("")
    lines.append("Discovered behavior CSV files/columns:")
    for spec in scale_specs:
        lines.append(f"  {spec['behavior_var']} <- {spec['path']}")

    lines.append("")
    lines.append("=" * 80)
    lines.append("Scale file audit")
    lines.append("=" * 80)
    if scale_audit_df.empty:
        lines.append("No scale audit information.")
    else:
        for _, r in scale_audit_df.iterrows():
            lines.append(
                f"{r['behavior_var']}: "
                f"N_behavior_file={r['N_behavior_file']}, "
                f"N_after_brain_behavior_match={r['N_after_brain_behavior_match']}, "
                f"N_after_ASD_filter={r['N_after_ASD_filter']}, "
                f"N_complete_case_min_across_predictors={r['N_complete_case_min_across_predictors']}, "
                f"mean={r['mean']}, sd={r['sd']}, min={r['min']}, max={r['max']}"
            )

    lines.append("")
    lines.append("=" * 80)
    lines.append("Analysis settings")
    lines.append("=" * 80)
    lines.append(f"Analysis set: {ANALYSIS_SET_NAME}")
    lines.append(f"Sensitivity note: {CURRENT_SENSITIVITY_SPEC.get('note', '')}")
    lines.append(f"Brain predictor: {', '.join(BRAIN_PREDICTORS)}")
    lines.append("Behavior variables are automatically discovered from all CSV files in BEHAVIOR_CSV_DIR.")
    lines.append("Different behavior variables are not merged into one composite score; each discovered numeric column is modeled separately.")
    lines.append("Nonlinear model uses orthogonalized B-spline components.")
    lines.append(f"Spline df candidates: {SPLINE_DF_CANDIDATES}")
    lines.append(f"Spline df selection method: {SPLINE_DF_SELECTION_METHOD}")
    lines.append(f"CV folds for df selection: {CV_N_FOLDS}")
    lines.append(f"Spline degree: {SPLINE_DEGREE}")
    lines.append(f"Covariates: {', '.join(COVARIATES_BASE)}")
    lines.append(f"Use Site fixed effect: {USE_SITE_FIXED_EFFECT}")
    lines.append(f"Minimum N for model: {MIN_N_FOR_MODEL}")
    if USE_SITE_FIXED_EFFECT:
        lines.append(f"Minimum Site N before collapsing: {MIN_SITE_N}")
    else:
        lines.append("Site fixed effect is not used in this model version.")

    lines.append("")
    lines.append("=" * 80)
    lines.append("Main statistical tests")
    lines.append("=" * 80)
    lines.append("p_overall_coupling_hc1 tests linear Coupling + nonlinear spline components jointly.")
    lines.append("p_nonlinear_hc1 tests nonlinear spline components after accounting for the linear Coupling term.")
    lines.append("HC1 robust standard errors are used for all Wald tests.")
    lines.append("The saved wald_* columns are F statistics when wald_statistic_name_* is F; report them as F(df_num, df_den), not as chi-square.")
    lines.append("No FDR or other multiple-comparison correction is applied; all reported p values are raw p values.")
    lines.append("Spline df is selected before final inference using the pre-specified criterion above, not by choosing the df with the smallest p value.")

    lines.append("")
    lines.append("=" * 80)
    lines.append("Model sample sizes")
    lines.append("=" * 80)
    if sample_df.empty:
        lines.append("No models were attempted.")
    else:
        for _, r in sample_df.iterrows():
            lines.append(
                f"{r['analysis_set']} | {r['behavior_var']} ~ {r['brain_predictor']}: "
                f"N={r['N']}, status={r['status']}, missing_columns={r['missing_columns']}"
            )

    lines.append("")
    lines.append("=" * 80)
    lines.append("Spline results sorted by raw overall p value")
    lines.append("=" * 80)
    if results_df.empty:
        lines.append("No valid spline models.")
    else:
        sorted_df = results_df.sort_values(
            ["p_overall_coupling_hc1", "p_nonlinear_hc1"],
            ascending=[True, True]
        )

        for _, r in sorted_df.iterrows():
            lines.append(
                f"{r['analysis_set']} | {r['behavior_var']} ~ {r['brain_predictor']}: "
                f"N={int(r['N'])}, "
                f"selected_df={int(r['selected_spline_df']) if 'selected_spline_df' in r.index and pd.notna(r['selected_spline_df']) else 'NA'}, "
                f"overall={r.get('wald_report_overall_coupling_hc1', 'NA')}, "
                f"nonlinear={r.get('wald_report_nonlinear_hc1', 'NA')}, "
                f"linear_component_t={r['t_linear_component_hc1']:.6g}, "
                f"linear_component_p={r['p_linear_component_hc1']:.6g}, "
                f"linear_R2={r['linear_r2']:.4f}, "
                f"spline_R2={r['spline_r2']:.4f}, "
                f"delta_AIC_linear_minus_spline={r['delta_aic_linear_minus_spline']:.4f}"
            )

    lines.append("")
    lines.append("=" * 80)
    lines.append("Spline df selection audit")
    lines.append("=" * 80)
    if spline_selection_df.empty:
        lines.append("No spline df selection information.")
    else:
        selected_only = spline_selection_df[spline_selection_df.get("is_selected", False) == True].copy()
        for _, r in selected_only.iterrows():
            lines.append(
                f"{r['analysis_set']} | {r['behavior_var']} ~ {r['brain_predictor']}: "
                f"selected_df={int(r['selected_spline_df'])}, "
                f"method={r['selected_by_method']}, "
                f"AIC={r['candidate_aic']:.6g}, BIC={r['candidate_bic']:.6g}, "
                f"CV_RMSE={r['candidate_cv_rmse']:.6g}"
            )

    lines.append("")
    lines.append("=" * 80)
    lines.append("Interpretation guide")
    lines.append("=" * 80)
    lines.append("If p_overall_coupling_hc1 is significant, Coupling_index_scaled is associated with the behavior score in either a linear or nonlinear way.")
    lines.append("If p_nonlinear_hc1 is significant, the association deviates from a simple linear relationship.")
    lines.append("If p_overall is significant but p_nonlinear is not, the association is likely mainly linear.")
    lines.append("For manuscript reporting, use the wald_report_* text or explicitly report F(df_num, df_den), exact p, and the linear-component t value separately.")
    lines.append("Positive delta_AIC_linear_minus_spline means the spline model has lower AIC than the linear model.")
    lines.append("Variables that represent reliability/QC indicators rather than symptom severity scores should be interpreted cautiously.")

    write_text(path, "\n".join(lines))


# ============================================================
# 12. 主程序
# ============================================================

def run_one_analysis():
    warnings.filterwarnings("ignore")

    print("=" * 80)
    print("Priority 4 ASD-only nonlinear spline brain-behavior association")
    print("All CSV behavior scale-specific files in folder + Coupling_index_scaled only")
    print("=" * 80)

    brain_df = load_brain_table(BRAIN_INPUT_FILE)

    scale_specs, discovery_df = discover_behavior_csv_specs(
        BEHAVIOR_CSV_DIR,
        recursive=SCAN_RECURSIVE
    )

    print("")
    print(f"Discovered analyzable behavior columns: {len(scale_specs)}")
    for spec in scale_specs:
        print(f"  - {spec['behavior_var']} <- {spec['path']} :: {spec['value_col']}")

    result_rows = []
    sample_rows = []
    scale_audit_rows = []
    spline_selection_rows = []
    model_cache_for_plots = []

    for scale_spec in scale_specs:
        behavior_var = scale_spec["behavior_var"]

        print("")
        print("-" * 80)
        print(f"Reading behavior file: {behavior_var}")
        print(scale_spec["path"])

        behavior_df = load_single_behavior_scale(scale_spec)
        merged_df = merge_brain_with_behavior(brain_df, behavior_df, behavior_var)
        asd_df, filter_note = filter_asd(merged_df)

        # 保存每个量表与脑指标匹配后的模型输入表，便于核查 N。
        merged_input_path = os.path.join(
            MERGED_MODEL_INPUT_DIR,
            f"model_input_{safe_filename(behavior_var)}.csv"
        )
        asd_df.to_csv(merged_input_path, index=False, encoding="utf-8-sig")

        # 审计：量表文件本身 N、脑行为匹配后 N、ASD 筛选后 N。
        audit_numeric = pd.to_numeric(asd_df[behavior_var], errors="coerce") if behavior_var in asd_df.columns else pd.Series(dtype=float)
        complete_case_cols = [behavior_var] + BRAIN_PREDICTORS + [c for c in COVARIATES_BASE if c in asd_df.columns]
        complete_case_cols = [c for c in complete_case_cols if c in asd_df.columns]
        n_complete_case_min = int(asd_df[complete_case_cols].dropna().shape[0]) if complete_case_cols else 0

        scale_audit_rows.append({
            "behavior_var": behavior_var,
            "value_col": scale_spec["value_col"],
            "source_csv": scale_spec["path"],
            "merged_model_input_csv": merged_input_path,
            "N_behavior_file": int(behavior_df[ID_COL].nunique()),
            "N_after_brain_behavior_match": int(merged_df[ID_COL].nunique()),
            "N_after_ASD_filter": int(asd_df[ID_COL].nunique()),
            "N_complete_case_min_across_predictors": n_complete_case_min,
            "mean": audit_numeric.mean(skipna=True),
            "sd": audit_numeric.std(skipna=True, ddof=1),
            "median": audit_numeric.median(skipna=True),
            "min": audit_numeric.min(skipna=True),
            "max": audit_numeric.max(skipna=True),
            "filter_note": filter_note,
        })

        print(
            f"Loaded {behavior_var}: "
            f"N_behavior_file={behavior_df[ID_COL].nunique()}, "
            f"N_after_match={merged_df[ID_COL].nunique()}, "
            f"N_after_ASD_filter={asd_df[ID_COL].nunique()}"
        )

        # 对当前量表建模。
        for brain_var in BRAIN_PREDICTORS:
            sample_row = {
                "analysis_set": ANALYSIS_SET_NAME,
                "behavior_var": behavior_var,
                "brain_predictor": brain_var,
                "N": 0,
                "N_behavior_file": int(behavior_df[ID_COL].nunique()),
                "N_after_brain_behavior_match": int(merged_df[ID_COL].nunique()),
                "N_after_ASD_filter": int(asd_df[ID_COL].nunique()),
                "status": "",
                "missing_columns": "",
                "site_note": "",
                "spline_note": "",
                "spline_formula": "",
                "selected_spline_df": np.nan,
                "spline_df_selection_method": SPLINE_DF_SELECTION_METHOD,
                "spline_df_selection_note": "",
                "merged_model_input_csv": merged_input_path,
            }

            if behavior_var not in asd_df.columns:
                sample_row["status"] = "skipped_behavior_column_missing"
                sample_row["missing_columns"] = behavior_var
                sample_rows.append(sample_row)
                continue

            model_df, missing_cols, site_note = build_model_dataframe(
                asd_df,
                behavior_var,
                brain_var
            )

            sample_row["N"] = int(model_df.shape[0])
            sample_row["missing_columns"] = ";".join(missing_cols)
            sample_row["site_note"] = site_note

            if missing_cols:
                sample_row["status"] = "skipped_missing_required_columns"
                sample_rows.append(sample_row)
                continue

            if model_df.shape[0] < MIN_N_FOR_MODEL:
                sample_row["status"] = f"skipped_N_less_than_{MIN_N_FOR_MODEL}"
                sample_rows.append(sample_row)
                print(
                    f"SKIP | {behavior_var} ~ spline({brain_var}) | "
                    f"N={model_df.shape[0]} < {MIN_N_FOR_MODEL}"
                )
                continue

            if model_df[behavior_var].std(ddof=0) == 0:
                sample_row["status"] = "skipped_zero_variance_behavior"
                sample_rows.append(sample_row)
                print(
                    f"SKIP | {behavior_var} ~ spline({brain_var}) | "
                    f"behavior has zero variance"
                )
                continue

            if model_df[brain_var].std(ddof=0) == 0:
                sample_row["status"] = "skipped_zero_variance_brain"
                sample_rows.append(sample_row)
                print(
                    f"SKIP | {behavior_var} ~ spline({brain_var}) | "
                    f"brain predictor has zero variance"
                )
                continue

            try:
                selected_spline_df, selection_df, selection_note, selected_info = select_spline_df(
                    model_df=model_df,
                    behavior_var=behavior_var,
                    brain_var=brain_var
                )

                selection_df.insert(0, "analysis_set", ANALYSIS_SET_NAME)
                selection_df.insert(1, "behavior_var", behavior_var)
                selection_df.insert(2, "brain_predictor", brain_var)
                selection_df.insert(3, "N", int(model_df.shape[0]))
                selection_df.insert(4, "source_csv", scale_spec["path"])
                spline_selection_rows.extend(selection_df.to_dict("records"))

                model_df_spline, nl_names, spline_transform, spline_note = (
                    add_orthogonal_nonlinear_spline_features(
                        model_df,
                        brain_var=brain_var,
                        spline_df=selected_spline_df
                    )
                )

                sample_row["spline_note"] = spline_note
                sample_row["selected_spline_df"] = selected_spline_df
                sample_row["spline_df_selection_method"] = selected_info["spline_df_selection_method"]
                sample_row["spline_df_selection_note"] = selection_note

                linear_fit, spline_fit, spline_robust, stat = fit_spline_model_hc1(
                    model_df=model_df_spline,
                    behavior_var=behavior_var,
                    brain_var=brain_var,
                    nl_names=nl_names
                )

                sample_row["spline_formula"] = stat["spline_formula"]

                result_row = {
                    "analysis_set": ANALYSIS_SET_NAME,
                    "behavior_var": behavior_var,
                    "brain_predictor": brain_var,
                    "N": int(model_df_spline.shape[0]),
                    "N_behavior_file": int(behavior_df[ID_COL].nunique()),
                    "N_after_brain_behavior_match": int(merged_df[ID_COL].nunique()),
                    "N_after_ASD_filter": int(asd_df[ID_COL].nunique()),
                    "source_csv": scale_spec["path"],
                    "merged_model_input_csv": merged_input_path,
                    "n_nonlinear_spline_components": len(nl_names),
                    "selected_spline_df": selected_spline_df,
                    "spline_df_selection_method": selected_info["spline_df_selection_method"],
                    "selected_spline_df_aic": selected_info["selected_spline_df_aic"],
                    "selected_spline_df_bic": selected_info["selected_spline_df_bic"],
                    "selected_spline_df_cv_rmse": selected_info["selected_spline_df_cv_rmse"],
                    "selected_spline_df_cv_mae": selected_info["selected_spline_df_cv_mae"],
                    "spline_df_selection_note": selected_info["spline_df_selection_note"],
                    "spline_note": spline_note,
                    "site_note": site_note,
                    "n_site_levels": int(model_df_spline["Site_for_model"].nunique()) if "Site_for_model" in model_df_spline.columns else 0,
                }

                result_row.update(stat)

                result_rows.append(result_row)

                model_cache_for_plots.append({
                    "behavior_var": behavior_var,
                    "brain_var": brain_var,
                    "model_df_spline": model_df_spline,
                    "spline_fit": spline_fit,
                    "spline_robust": spline_robust,
                    "spline_transform": spline_transform,
                    "nl_names": nl_names,
                })

                sample_row["status"] = "model_ok"
                sample_rows.append(sample_row)

                print(
                    f"OK | {ANALYSIS_SET_NAME} | {behavior_var} ~ spline({brain_var}) | "
                    f"N={model_df_spline.shape[0]} | "
                    f"overall={result_row['wald_report_overall_coupling_hc1']} | "
                    f"nonlinear={result_row['wald_report_nonlinear_hc1']} | "
                    f"linear_t={result_row['t_linear_component_hc1']:.4g} | "
                    f"linear_p={result_row['p_linear_component_hc1']:.4g} | "
                    f"selected_df={selected_spline_df} | "
                    f"delta_AIC={result_row['delta_aic_linear_minus_spline']:.3f}"
                )

            except Exception as e:
                sample_row["status"] = f"model_error: {repr(e)}"
                sample_rows.append(sample_row)

                print(
                    f"ERROR | {ANALYSIS_SET_NAME} | {behavior_var} ~ spline({brain_var}) | "
                    f"{repr(e)}"
                )

    results_df = pd.DataFrame(result_rows)
    sample_df = pd.DataFrame(sample_rows)
    scale_audit_df = pd.DataFrame(scale_audit_rows)
    spline_selection_df = pd.DataFrame(spline_selection_rows)

    # ------------------------------------------------------------
    # 未校正结果排序和作图
    # ------------------------------------------------------------
    if not results_df.empty:
        plot_paths = []

        result_lookup = {}
        for _, r in results_df.iterrows():
            result_lookup[(r["behavior_var"], r["brain_predictor"])] = r

        for cache in model_cache_for_plots:
            key = (cache["behavior_var"], cache["brain_var"])
            if key not in result_lookup:
                plot_paths.append({
                    "behavior_var": cache["behavior_var"],
                    "brain_predictor": cache["brain_var"],
                    "adjusted_spline_plot": "",
                })
                continue

            r = result_lookup[key]

            plot_path = save_adjusted_spline_plot(
                model_df=cache["model_df_spline"],
                behavior_var=cache["behavior_var"],
                brain_var=cache["brain_var"],
                spline_fit=cache["spline_fit"],
                spline_robust=cache["spline_robust"],
                spline_transform=cache["spline_transform"],
                nl_names=cache["nl_names"],
                result_row=r,
            )

            plot_paths.append({
                "behavior_var": cache["behavior_var"],
                "brain_predictor": cache["brain_var"],
                "adjusted_spline_plot": plot_path,
            })

        plot_df = pd.DataFrame(plot_paths)

        results_df = results_df.merge(
            plot_df,
            on=["behavior_var", "brain_predictor"],
            how="left"
        )

        results_df = results_df.sort_values(
            ["p_overall_coupling_hc1", "p_nonlinear_hc1"],
            ascending=[True, True]
        ).reset_index(drop=True)

    raw_p_summary = make_raw_p_summary(results_df) if not results_df.empty else pd.DataFrame()

    # ------------------------------------------------------------
    # 保存输出
    # ------------------------------------------------------------
    results_path = os.path.join(
        OUT_DIR,
        f"{get_run_file_prefix()}_all_results.csv"
    )

    raw_p_summary_path = os.path.join(
        OUT_DIR,
        f"{get_run_file_prefix()}_raw_p_summary.csv"
    )

    sample_path = os.path.join(
        OUT_DIR,
        f"{get_run_file_prefix()}_model_sample_sizes.csv"
    )

    scale_audit_path = os.path.join(
        OUT_DIR,
        f"{get_run_file_prefix()}_scale_file_audit.csv"
    )

    discovery_audit_path = os.path.join(
        OUT_DIR,
        f"{get_run_file_prefix()}_discovery_audit.csv"
    )

    selected_behavior_path = os.path.join(
        OUT_DIR,
        f"{get_run_file_prefix()}_selected_behavior_vars.csv"
    )

    spline_df_selection_path = os.path.join(
        OUT_DIR,
        f"{get_run_file_prefix()}_spline_df_selection_audit.csv"
    )

    report_path = os.path.join(
        OUT_DIR,
        f"{get_run_file_prefix()}_QC_report.txt"
    )

    results_df.to_csv(
        results_path,
        index=False,
        encoding="utf-8-sig"
    )

    raw_p_summary.to_csv(
        raw_p_summary_path,
        index=False,
        encoding="utf-8-sig"
    )

    sample_df.to_csv(
        sample_path,
        index=False,
        encoding="utf-8-sig"
    )

    scale_audit_df.to_csv(
        scale_audit_path,
        index=False,
        encoding="utf-8-sig"
    )

    discovery_df.to_csv(
        discovery_audit_path,
        index=False,
        encoding="utf-8-sig"
    )

    pd.DataFrame([
        {
            "behavior_var": spec["behavior_var"],
            "value_col": spec["value_col"],
            "source_csv": spec["path"],
        }
        for spec in scale_specs
    ]).to_csv(
        selected_behavior_path,
        index=False,
        encoding="utf-8-sig"
    )

    spline_selection_df.to_csv(
        spline_df_selection_path,
        index=False,
        encoding="utf-8-sig"
    )

    write_qc_report(
        path=report_path,
        brain_input_file=BRAIN_INPUT_FILE,
        scale_specs=scale_specs,
        brain_df=brain_df,
        scale_audit_df=scale_audit_df,
        results_df=results_df,
        sample_df=sample_df,
        spline_selection_df=spline_selection_df,
    )

    # ------------------------------------------------------------
    # 控制台输出
    # ------------------------------------------------------------
    print("")
    print("=" * 80)
    print("Priority 4 ASD-only nonlinear spline analysis completed.")
    print("All CSV behavior scale-specific files in folder + Coupling_index_scaled only.")
    print("=" * 80)

    print("")
    print("Output directory:")
    print(OUT_DIR)

    print("")
    print("Main results:")
    print(results_path)

    print("")
    print("Raw p summary:")
    print(raw_p_summary_path)

    print("")
    print("All discovered behavior variables:")
    print(selected_behavior_path)

    print("")
    print("Spline df selection audit:")
    print(spline_df_selection_path)

    print("")
    print("Scale file audit:")
    print(scale_audit_path)

    print("")
    print("Discovery audit:")
    print(discovery_audit_path)

    print("")
    print("Model sample sizes:")
    print(sample_path)

    print("")
    print("Merged model inputs:")
    print(MERGED_MODEL_INPUT_DIR)

    print("")
    print("QC report:")
    print(report_path)

    print("")
    print("Adjusted spline plots:")
    print(PLOT_SPLINE_DIR)


def make_sensitivity_concordance_summary(combined_df):
    """
    汇总每个行为变量在多组敏感性分析中的稳定性。
    重点看：
      1. raw p < 0.05 的参数组数量；
      2. 最小/最大 p 值；
      3. beta 方向是否稳定。
    """
    if combined_df.empty:
        return pd.DataFrame()

    rows = []

    group_cols = ["behavior_var", "brain_predictor"]
    for (behavior_var, brain_predictor), g in combined_df.groupby(group_cols, dropna=False):
        beta = pd.to_numeric(g.get("beta_linear_component"), errors="coerce")
        p_overall = pd.to_numeric(g.get("p_overall_coupling_hc1"), errors="coerce")
        p_nl = pd.to_numeric(g.get("p_nonlinear_hc1"), errors="coerce")

        beta_nonzero = beta.dropna()
        beta_signs = set(np.sign(beta_nonzero[beta_nonzero != 0]).astype(int).tolist())

        rows.append({
            "behavior_var": behavior_var,
            "brain_predictor": brain_predictor,
            "n_sensitivity_specs_with_results": int(g.shape[0]),
            "analysis_sets": ";".join(g["analysis_set"].astype(str).tolist()) if "analysis_set" in g.columns else "",
            "n_overall_raw_p_lt_0_05": int((p_overall < 0.05).sum()),
            "n_nonlinear_raw_p_lt_0_05": int((p_nl < 0.05).sum()),
            "min_p_overall_coupling_hc1": float(p_overall.min(skipna=True)) if p_overall.notna().any() else np.nan,
            "max_p_overall_coupling_hc1": float(p_overall.max(skipna=True)) if p_overall.notna().any() else np.nan,
            "min_p_nonlinear_hc1": float(p_nl.min(skipna=True)) if p_nl.notna().any() else np.nan,
            "max_p_nonlinear_hc1": float(p_nl.max(skipna=True)) if p_nl.notna().any() else np.nan,
            "min_beta_linear_component": float(beta.min(skipna=True)) if beta.notna().any() else np.nan,
            "max_beta_linear_component": float(beta.max(skipna=True)) if beta.notna().any() else np.nan,
            "linear_beta_direction_consistent": bool(len(beta_signs) <= 1) if len(beta_nonzero) > 0 else False,
        })

    out = pd.DataFrame(rows)

    if not out.empty:
        out = out.sort_values(
            ["n_overall_raw_p_lt_0_05", "min_p_overall_coupling_hc1"],
            ascending=[False, True]
        ).reset_index(drop=True)

    return out


def main():
    warnings.filterwarnings("ignore")

    Path(SENSITIVITY_ROOT_DIR).mkdir(parents=True, exist_ok=True)

    print("=" * 80)
    print("Multi-parameter sensitivity analysis started")
    print("=" * 80)
    print(f"Number of sensitivity specs: {len(SENSITIVITY_ANALYSIS_SPECS)}")
    print(f"Sensitivity root directory: {SENSITIVITY_ROOT_DIR}")

    combined_raw_p_rows = []
    run_status_rows = []

    for i, spec in enumerate(SENSITIVITY_ANALYSIS_SPECS, start=1):
        apply_sensitivity_spec(spec)

        print("")
        print("#" * 80)
        print(f"Sensitivity run {i}/{len(SENSITIVITY_ANALYSIS_SPECS)}")
        print(f"ANALYSIS_SET_NAME = {ANALYSIS_SET_NAME}")
        print(f"Covariates = {COVARIATES_BASE}")
        print(f"Use Site FE = {USE_SITE_FIXED_EFFECT}")
        print(f"Spline df candidates = {SPLINE_DF_CANDIDATES}")
        print(f"Spline df selection = {SPLINE_DF_SELECTION_METHOD}")
        print(f"Output = {OUT_DIR}")
        print("#" * 80)

        status_row = current_sensitivity_metadata()
        status_row.update({
            "run_index": i,
            "status": "",
            "error": "",
            "out_dir": OUT_DIR,
        })

        try:
            run_one_analysis()

            raw_p_summary_path = os.path.join(
                OUT_DIR,
                f"{get_run_file_prefix()}_raw_p_summary.csv"
            )

            if os.path.exists(raw_p_summary_path):
                tmp = pd.read_csv(raw_p_summary_path, low_memory=False)
                for k, v in current_sensitivity_metadata().items():
                    tmp[k] = v
                tmp["sensitivity_out_dir"] = OUT_DIR
                combined_raw_p_rows.append(tmp)

            status_row["status"] = "ok"

        except Exception as e:
            status_row["status"] = "error"
            status_row["error"] = repr(e)
            print("")
            print(f"ERROR in sensitivity run {ANALYSIS_SET_NAME}: {repr(e)}")
            print("Continue to next sensitivity spec.")

        run_status_rows.append(status_row)

    run_status_df = pd.DataFrame(run_status_rows)

    status_path = os.path.join(
        SENSITIVITY_ROOT_DIR,
        "sensitivity_run_status.csv"
    )
    run_status_df.to_csv(status_path, index=False, encoding="utf-8-sig")

    if combined_raw_p_rows:
        combined_raw_p_df = pd.concat(combined_raw_p_rows, ignore_index=True)
    else:
        combined_raw_p_df = pd.DataFrame()

    combined_path = os.path.join(
        SENSITIVITY_ROOT_DIR,
        "sensitivity_combined_raw_p_summary.csv"
    )
    combined_raw_p_df.to_csv(combined_path, index=False, encoding="utf-8-sig")

    concordance_df = make_sensitivity_concordance_summary(combined_raw_p_df)
    concordance_path = os.path.join(
        SENSITIVITY_ROOT_DIR,
        "sensitivity_concordance_summary.csv"
    )
    concordance_df.to_csv(concordance_path, index=False, encoding="utf-8-sig")

    print("")
    print("=" * 80)
    print("Multi-parameter sensitivity analysis completed.")
    print("=" * 80)
    print("")
    print("Run status:")
    print(status_path)
    print("")
    print("Combined raw p summary:")
    print(combined_path)
    print("")
    print("Concordance summary:")
    print(concordance_path)


if __name__ == "__main__":
    main()
