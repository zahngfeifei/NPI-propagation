from pathlib import Path
from collections import defaultdict
import re
import pandas as pd


# =========================
# 1. 路径设置
# =========================

# 原始 ABIDE 量表文件
abide1_csv = Path(r"I:\DYF\NPI-3\ABIDEⅠ.csv")
abide2_csv = Path(r"I:\DYF\NPI-3\ABIDEⅡ.csv")

# 当前统一输出目录
data_dir = Path(r"I:\DYF\NPI-3\被试信息")
data_dir.mkdir(parents=True, exist_ok=True)

# 因果梯度分析结果目录
result_root = Path(r"I:\DYF\NPI-3\2.梯度分析\正向连接-独立模板\ABIDE_结果1")

# 输出文件
merged_all_csv = data_dir / "ABIDEⅠ_Ⅱ_合并_完整量表.csv"
computed_csv = data_dir / "ABIDEⅠ_Ⅱ_合并_仅保留有因果梯度结果_完整量表.csv"
missing_computed_csv = data_dir / "ABIDEⅠ_Ⅱ_合并_缺失因果梯度结果被试.csv"
computed_subject_list_csv = data_dir / "因果梯度完成计算被试列表.csv"
count_txt = data_dir / "因果梯度完成计算被试数量.txt"
report_txt = data_dir / "ABIDE_完整量表与因果梯度计算后统计报告.txt"


# =========================
# 2. 工具函数
# =========================

def read_csv_safely(csv_path: Path) -> pd.DataFrame:
    """
    兼容 utf-8-sig / utf-8 / gbk / gb18030 编码读取 CSV。
    所有列按字符串读取，避免 SUB_ID 被自动转换。
    """
    encodings = ["utf-8-sig", "utf-8", "gbk", "gb18030"]

    last_error = None

    for enc in encodings:
        try:
            df = pd.read_csv(
                csv_path,
                dtype=str,
                encoding=enc,
                keep_default_na=False
            )
            df.columns = [c.strip() for c in df.columns]
            return df
        except Exception as e:
            last_error = e

    raise RuntimeError(f"无法读取文件：{csv_path}\n最后一次错误：{last_error}")


def normalize_sub_id(value) -> str:
    """
    标准化 SUB_ID，增强 CSV 与文件夹名之间的匹配能力。

    示例：
    51456          -> 51456
    51456.0        -> 51456
    sub-51456      -> 51456
    Sub51456       -> 51456
    sub-Sub51456   -> 51456
    0051456        -> 51456
    """
    if value is None:
        return ""

    s = str(value).strip()

    if s == "":
        return ""

    if re.fullmatch(r"\d+\.0", s):
        s = s[:-2]

    if s.lower().startswith("sub-"):
        s = s[4:]

    if s.lower().startswith("sub"):
        s = s[3:]

    s = s.strip()

    if re.fullmatch(r"\d+", s):
        s = str(int(s))

    return s


def extract_subject_from_result_folder(folder_path: Path) -> str:
    """
    从因果梯度分析结果文件夹名中提取 SUB_ID。

    支持：
    sub-Sub51456
    sub-51456
    Sub51456
    51456
    """
    name = folder_path.name.strip()
    return normalize_sub_id(name)


def build_computed_result_index(result_dir: Path, required_file=None):
    """
    建立：
    标准化 SUB_ID -> 因果梯度结果文件夹路径列表

    required_file:
    - None：只要存在 sub-Subxxxxx 文件夹，就认为该被试完成计算。
    - 指定文件名，例如 "out_G1_procrustes.npy"：只有文件夹内存在该文件，才认为完成计算。
    """
    if not result_dir.exists():
        raise FileNotFoundError(f"因果梯度结果文件夹不存在：{result_dir}")

    result_folders = [
        p for p in result_dir.iterdir()
        if p.is_dir()
    ]

    index = defaultdict(list)

    for folder in result_folders:
        sub_id = extract_subject_from_result_folder(folder)

        if not sub_id:
            continue

        if required_file is not None:
            if not (folder / required_file).exists():
                continue

        index[sub_id].append(str(folder))

    return index, result_folders


def merge_abide_scale_files(abide1_path: Path, abide2_path: Path) -> pd.DataFrame:
    """
    合并 ABIDEⅠ 和 ABIDEⅡ 量表文件。
    """
    df1 = read_csv_safely(abide1_path)
    df2 = read_csv_safely(abide2_path)

    df1["DATASET"] = "ABIDEⅠ"
    df2["DATASET"] = "ABIDEⅡ"

    df_all = pd.concat(
        [df1, df2],
        axis=0,
        ignore_index=True,
        sort=False
    )

    if "SUB_ID" not in df_all.columns:
        raise ValueError("合并后的 CSV 中没有 SUB_ID 列，请检查 ABIDEⅠ.csv 和 ABIDEⅡ.csv。")

    df_all["SUB_ID"] = df_all["SUB_ID"].map(normalize_sub_id)

    return df_all


def filter_by_computed_results(df: pd.DataFrame, computed_index: dict):
    """
    在完整量表数据基础上，只保留已有因果梯度分析结果文件夹的被试。
    """
    if "SUB_ID" not in df.columns:
        raise ValueError("输入 CSV 中没有 SUB_ID 列，请检查文件。")

    df = df.copy()

    df["_SUB_ID_NORM"] = df["SUB_ID"].map(normalize_sub_id)

    df["HAS_CAUSAL_GRADIENT_RESULT"] = df["_SUB_ID_NORM"].map(
        lambda x: x in computed_index
    )

    df["CAUSAL_GRADIENT_RESULT_COUNT"] = df["_SUB_ID_NORM"].map(
        lambda x: len(computed_index.get(x, []))
    )

    df["CAUSAL_GRADIENT_RESULT_FOLDER"] = df["_SUB_ID_NORM"].map(
        lambda x: ";".join(computed_index.get(x, []))
    )

    computed_df = df[df["HAS_CAUSAL_GRADIENT_RESULT"]].copy()
    missing_df = df[~df["HAS_CAUSAL_GRADIENT_RESULT"]].copy()

    computed_df = computed_df.drop(columns=["_SUB_ID_NORM"])
    missing_df = missing_df.drop(columns=["_SUB_ID_NORM"])

    return computed_df, missing_df


def create_computed_subject_list(computed_index: dict) -> pd.DataFrame:
    """
    生成结果文件夹中可识别的完成计算被试列表。
    兼容纯数字 SUB_ID 和非纯数字 SUB_ID，避免 int 与 str 混合排序报错。
    """
    rows = []

    def subject_sort_key(x):
        x = str(x).strip()

        # 纯数字被试排在前面，按数值大小排序
        if x.isdigit():
            return (0, int(x))

        # 非纯数字被试排在后面，按字符串排序
        return (1, x)

    for sub_id in sorted(computed_index.keys(), key=subject_sort_key):
        folders = computed_index[sub_id]

        rows.append({
            "SUB_ID": sub_id,
            "CAUSAL_GRADIENT_RESULT_COUNT": len(folders),
            "CAUSAL_GRADIENT_RESULT_FOLDER": ";".join(folders)
        })

    return pd.DataFrame(rows)


def get_age_series(df: pd.DataFrame) -> pd.Series:
    if "AGE_AT_SCAN" not in df.columns:
        return pd.Series(dtype=float)

    age = pd.to_numeric(df["AGE_AT_SCAN"], errors="coerce")
    age = age.dropna()
    return age


def count_site(df: pd.DataFrame) -> int:
    if "SITE_ID" not in df.columns:
        return 0

    site = df["SITE_ID"].astype(str).str.strip()
    site = site[site != ""]
    return int(site.nunique())


def count_sex(df: pd.DataFrame):
    """
    ABIDE 中 SEX 通常编码为：
    1 = 男
    2 = 女
    """
    if "SEX" not in df.columns:
        return 0, 0, len(df)

    sex = df["SEX"].astype(str).str.strip()

    male = int((sex == "1").sum())
    female = int((sex == "2").sum())
    unknown = int(len(df) - male - female)

    return male, female, unknown


def count_dx_group(df: pd.DataFrame):
    """
    ABIDE 中 DX_GROUP 通常编码为：
    1 = ASD
    2 = HC
    """
    if "DX_GROUP" not in df.columns:
        return 0, 0, len(df)

    dx = df["DX_GROUP"].astype(str).str.strip()

    asd = int((dx == "1").sum())
    hc = int((dx == "2").sum())
    unknown = int(len(df) - asd - hc)

    return asd, hc, unknown


def summarize_dataframe(df: pd.DataFrame, name: str) -> str:
    n_subjects = len(df)
    n_sites = count_site(df)

    age = get_age_series(df)
    if len(age) > 0:
        age_text = f"{age.min():.2f} - {age.max():.2f}"
    else:
        age_text = "无法统计，AGE_AT_SCAN 缺失或无法转换为数字"

    male, female, sex_unknown = count_sex(df)
    asd, hc, dx_unknown = count_dx_group(df)

    lines = []
    lines.append(name)
    lines.append("-" * 70)
    lines.append(f"被试总数：{n_subjects}")
    lines.append(f"站点数量：{n_sites}")
    lines.append(f"年龄范围：{age_text}")
    lines.append(f"男性人数：{male}")
    lines.append(f"女性人数：{female}")
    lines.append(f"性别未知或异常编码人数：{sex_unknown}")
    lines.append(f"ASD 人数：{asd}")
    lines.append(f"HC 人数：{hc}")
    lines.append(f"诊断未知或异常编码人数：{dx_unknown}")
    lines.append("")

    return "\n".join(lines)


def summarize_by_dataset(df: pd.DataFrame, title: str) -> str:
    """
    如果存在 DATASET 列，则分别统计 ABIDEⅠ 和 ABIDEⅡ。
    """
    lines = []
    lines.append(title)
    lines.append("=" * 70)

    lines.append(summarize_dataframe(df, f"{title}：总体"))

    if "DATASET" in df.columns:
        dataset_values = df["DATASET"].astype(str).str.strip()
        for dataset_name in sorted(dataset_values[dataset_values != ""].unique()):
            sub_df = df[dataset_values == dataset_name].copy()
            lines.append(summarize_dataframe(sub_df, f"{title}：{dataset_name}"))

    return "\n".join(lines)


def generate_count_text(
    all_df: pd.DataFrame,
    computed_df: pd.DataFrame,
    missing_df: pd.DataFrame,
    computed_index: dict,
    result_folder_count: int,
    output_path: Path
):
    """
    单独输出完成计算被试数量统计。
    """
    lines = []
    lines.append("因果梯度完成计算被试数量统计")
    lines.append("=" * 70)
    lines.append(f"完整量表合并后被试数：{len(all_df)}")
    lines.append(f"扫描到结果文件夹数：{result_folder_count}")
    lines.append(f"结果文件夹中可识别的唯一完成计算被试数：{len(computed_index)}")
    lines.append(f"完整量表中成功匹配到的完成计算被试数：{len(computed_df)}")
    lines.append(f"完整量表中缺失因果梯度结果的被试数：{len(missing_df)}")
    lines.append("")
    lines.append("最终用于后续行为/量表分析的被试数，应使用：")
    lines.append(f"{len(computed_df)}")

    with open(output_path, "w", encoding="utf-8-sig") as f:
        f.write("\n".join(lines))


def generate_report(
    all_df: pd.DataFrame,
    computed_df: pd.DataFrame,
    missing_df: pd.DataFrame,
    result_folder_count: int,
    unique_result_subject_count: int,
    output_path: Path
):
    report = []

    report.append("ABIDE 完整量表与因果梯度计算后统计报告")
    report.append("=" * 70)
    report.append("")
    report.append("说明：")
    report.append("1. 完整量表数据由 ABIDEⅠ.csv 和 ABIDEⅡ.csv 合并得到。")
    report.append("2. 因果梯度计算后数据指在完整量表基础上，进一步保留存在结果文件夹的被试。")
    report.append("3. 因果梯度结果文件夹格式按 sub-Subxxxx、sub-xxxx、Subxxxx 或 xxxx 匹配。")
    report.append("4. SEX 编码按 ABIDE 常用规则统计：1 = 男，2 = 女。")
    report.append("5. DX_GROUP 编码按 ABIDE 常用规则统计：1 = ASD，2 = HC。")
    report.append("6. 年龄来自 AGE_AT_SCAN。")
    report.append("")
    report.append("核心数量统计：")
    report.append("-" * 70)
    report.append(f"扫描到因果梯度结果文件夹数：{result_folder_count}")
    report.append(f"结果文件夹中可识别的唯一完成计算被试数：{unique_result_subject_count}")
    report.append(f"完整量表合并后被试数：{len(all_df)}")
    report.append(f"完整量表中成功匹配到的完成计算被试数：{len(computed_df)}")
    report.append(f"完整量表中缺失因果梯度结果被试数：{len(missing_df)}")
    report.append("")
    report.append(f"最终用于后续行为/量表分析的被试数：{len(computed_df)}")
    report.append("")

    report.append(summarize_by_dataset(
        all_df,
        "一、完整量表数据统计"
    ))

    report.append("")
    report.append(summarize_by_dataset(
        computed_df,
        "二、因果梯度计算后保留数据统计"
    ))

    report.append("")
    report.append(summarize_by_dataset(
        missing_df,
        "三、完整量表中缺失因果梯度结果的数据统计"
    ))

    with open(output_path, "w", encoding="utf-8-sig") as f:
        f.write("\n".join(report))


# =========================
# 3. 主程序
# =========================

def main():
    # 是否要求结果文件夹中必须存在某个核心结果文件
    # 如果只按文件夹判断完成计算，设为 None
    # 严格模式可要求 out_G1_procrustes.npy 存在：
    # required_file = "out_G1_procrustes.npy"
    required_file = None

    # 1. 合并 ABIDEⅠ 和 ABIDEⅡ 完整量表
    all_df = merge_abide_scale_files(
        abide1_path=abide1_csv,
        abide2_path=abide2_csv
    )

    # 保存完整合并量表
    all_df.to_csv(
        merged_all_csv,
        index=False,
        encoding="utf-8-sig"
    )

    # 2. 建立因果梯度结果索引
    computed_index, result_folders = build_computed_result_index(
        result_dir=result_root,
        required_file=required_file
    )

    # 3. 输出结果文件夹中可识别的完成计算被试列表
    computed_subject_list_df = create_computed_subject_list(computed_index)

    computed_subject_list_df.to_csv(
        computed_subject_list_csv,
        index=False,
        encoding="utf-8-sig"
    )

    # 4. 按因果梯度计算结果筛选完整量表
    computed_df, missing_df = filter_by_computed_results(
        df=all_df,
        computed_index=computed_index
    )

    # 5. 保存筛选后的完整量表
    computed_df.to_csv(
        computed_csv,
        index=False,
        encoding="utf-8-sig"
    )

    # 6. 保存缺失因果梯度结果的被试
    missing_df.to_csv(
        missing_computed_csv,
        index=False,
        encoding="utf-8-sig"
    )

    # 7. 单独保存完成计算数量
    generate_count_text(
        all_df=all_df,
        computed_df=computed_df,
        missing_df=missing_df,
        computed_index=computed_index,
        result_folder_count=len(result_folders),
        output_path=count_txt
    )

    # 8. 生成完整统计报告
    generate_report(
        all_df=all_df,
        computed_df=computed_df,
        missing_df=missing_df,
        result_folder_count=len(result_folders),
        unique_result_subject_count=len(computed_index),
        output_path=report_txt
    )

    # 9. 终端输出
    print("处理完成")
    print("=" * 70)
    print(f"ABIDEⅠ 输入文件：{abide1_csv}")
    print(f"ABIDEⅡ 输入文件：{abide2_csv}")
    print(f"因果梯度结果目录：{result_root}")
    print(f"统一输出目录：{data_dir}")
    print("")

    print("输出文件")
    print("-" * 70)
    print(f"完整合并量表：{merged_all_csv}")
    print(f"因果梯度计算后完整量表：{computed_csv}")
    print(f"缺失因果梯度结果被试表：{missing_computed_csv}")
    print(f"完成计算被试列表：{computed_subject_list_csv}")
    print(f"完成计算被试数量统计：{count_txt}")
    print(f"统计报告：{report_txt}")
    print("")

    print("被试数量统计")
    print("-" * 70)
    print(f"完整量表合并后被试数：{len(all_df)}")
    print(f"扫描到结果文件夹数：{len(result_folders)}")
    print(f"结果文件夹中可识别的唯一完成计算被试数：{len(computed_index)}")
    print(f"完整量表中成功匹配到的完成计算被试数：{len(computed_df)}")
    print(f"完整量表中缺失因果梯度结果的被试数：{len(missing_df)}")
    print("")

    print("=" * 70)
    print(f"最终用于后续行为/量表分析的被试总数：{len(computed_df)}")
    print("=" * 70)


if __name__ == "__main__":
    main()
