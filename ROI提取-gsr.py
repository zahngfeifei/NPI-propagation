import os
import glob
import re
import numpy as np
import pandas as pd

from nilearn import datasets
from nilearn.maskers import NiftiLabelsMasker

from concurrent.futures import ProcessPoolExecutor, as_completed
from multiprocessing import freeze_support


# ============================================================
# 用户参数设置
# ============================================================

fmri_dir = r"F:\ABIDE_功能数据_2143"
confounds_dir = r"F:\回归文件\回归文件"

output_root = r"I:\DYF\NPI-3\ROI\pipeline_full_36p_gsr_spike"

tr_csv_file = r"F:\EFC\TR分类结果.csv"
phenotype_csv = r"F:\EFC\数据预计算\ABIDE_filtered_exist_in_folder.csv"

n_jobs = 8
os.makedirs(output_root, exist_ok=True)


# ============================================================
# Atlas 和滤波参数
# ============================================================

N_ROIS = 400
YEO_NETWORKS = 7
RESOLUTION_MM = 2

LOW_PASS = 0.1
HIGH_PASS = 0.01


# ============================================================
# FD 质量控制参数
# ============================================================

USE_FD_QC = True

# 平均 FD > 0.3 mm 的 run 被排除
MEAN_FD_THRESHOLD = 0.3

# FD spike regressors 阈值：FD > 0.5 mm 的时间点加入 spike 回归量
FD_SPIKE_THRESHOLD = 0.5


# ============================================================
# 主分析预处理方案：仅保留最严格 pipeline
# ============================================================

PIPELINES = [
    {
        "name": "pipeline_full_36p_gsr_spike",
        "description": (
            "Motion 6 + WM + CSF + GSR + derivatives + squares "
            "+ FD spike regressors"
        ),
        "confound_model": "full",
        "use_gsr": True,
        "use_derivatives": True,
        "use_squares": True,
        "use_fd_spikes": True,
        "fd_spike_threshold": FD_SPIKE_THRESHOLD,
        "use_non_steady_state": False
    }
]

MAIN_PIPELINE_NAME = "pipeline_full_36p_gsr_spike"

if len(PIPELINES) != 1 or PIPELINES[0]["name"] != MAIN_PIPELINE_NAME:
    raise RuntimeError("当前脚本应仅保留主分析使用的最严格预处理方案。")

for pipeline in PIPELINES:
    os.makedirs(os.path.join(output_root, pipeline["name"]), exist_ok=True)

print(f"主分析仅使用预处理方案: {MAIN_PIPELINE_NAME}")


# ============================================================
# 文件名与被试编号匹配
# ============================================================

_sub_re = re.compile(r"Sub(\d+)", re.IGNORECASE)
_digit_re = re.compile(r"\d+")


def normalize_sub_value(value):
    """
    将不同形式的被试编号统一为 int：
    Sub0051457 -> 51457
    0051457    -> 51457
    51457      -> 51457
    51457.0    -> 51457
    """
    if value is None:
        return None

    if pd.isna(value):
        return None

    text = str(value).strip()

    if text == "":
        return None

    m = _sub_re.search(text)
    if m:
        return int(m.group(1))

    m = _digit_re.search(text)
    if m:
        return int(m.group(0))

    return None


def file_matches_subid(filename: str, subid: str) -> bool:
    target = normalize_sub_value(subid)

    if target is None:
        return False

    base = os.path.basename(filename)

    m = _sub_re.search(base)
    if m:
        return int(m.group(1)) == target

    for chunk in _digit_re.findall(base):
        if chunk and int(chunk) == target:
            return True

    return False


def parse_bids_entities(filename: str) -> dict:
    """
    简单解析 BIDS 文件名中的实体：
    sub, ses, task, acq, run, space, desc 等。
    """
    base = os.path.basename(filename)

    if base.endswith(".nii.gz"):
        base = base[:-7]
    elif base.endswith(".nii"):
        base = base[:-4]
    elif base.endswith(".tsv"):
        base = base[:-4]
    elif base.endswith(".json"):
        base = base[:-5]

    entities = {}

    for part in base.split("_"):
        if "-" in part:
            key, value = part.split("-", 1)
            entities[key] = value

    return entities


# ============================================================
# confounds 读取与 FD 质量控制
# ============================================================

def read_confounds(confounds_file: str) -> pd.DataFrame:
    confounds = pd.read_csv(confounds_file, sep="\t")
    confounds.columns = confounds.columns.str.strip()
    return confounds


def get_mean_fd(confounds: pd.DataFrame) -> float:
    """
    计算平均 FD。
    与 fMRIPrep 输出习惯一致，首帧 FD 可能为 NaN；
    这里将 NaN 和无穷值替换为 0 后计算平均 FD。
    """
    if "framewise_displacement" not in confounds.columns:
        return np.nan

    fd = (
        pd.to_numeric(confounds["framewise_displacement"], errors="coerce")
        .replace([np.inf, -np.inf], np.nan)
        .fillna(0)
    )

    return float(np.mean(fd.to_numpy()))


def pass_fd_qc(confounds: pd.DataFrame, confounds_file: str) -> tuple:
    """
    FD 质量控制：
    1. 若缺少 framewise_displacement 列，则排除该 run；
    2. 若平均 FD > 0.3 mm，则排除该 run；
    3. 否则保留。
    """
    if not USE_FD_QC:
        return True, np.nan, "fd_qc_disabled"

    if "framewise_displacement" not in confounds.columns:
        print(
            f"排除 run: {os.path.basename(confounds_file)}；"
            f"缺少 framewise_displacement，无法执行 mean FD QC 和 FD spike 回归"
        )
        return False, np.nan, "excluded_missing_fd"

    mean_fd = get_mean_fd(confounds)

    if mean_fd > MEAN_FD_THRESHOLD:
        print(
            f"排除 run: {os.path.basename(confounds_file)}；"
            f"mean FD = {mean_fd:.4f} mm > {MEAN_FD_THRESHOLD} mm"
        )
        return False, mean_fd, "excluded_mean_fd"

    return True, mean_fd, "fd_qc_passed"


# ============================================================
# 混杂回归矩阵构建
# ============================================================

def build_confounds(confounds: pd.DataFrame, confounds_file: str, pipeline: dict):
    """
    构建主分析使用的扩展混杂回归矩阵。

    当前主分析 pipeline:
        pipeline_full_36p_gsr_spike

    包括：
        1. 6 个头动参数：
           trans_x, trans_y, trans_z, rot_x, rot_y, rot_z

        2. 3 个生理/全局信号：
           white_matter, csf, global_signal

        3. 上述 9 个基础回归量的一阶导数：
           *_derivative1

        4. 上述 9 个基础回归量的平方项：
           *_power2

        5. 上述 9 个基础回归量的一阶导数平方项：
           *_derivative1_power2

        6. FD spike regressors：
           对 FD > 0.5 mm 的时间点加入一列 one-hot spike 回归量
    """

    motion_base = [
        "trans_x", "trans_y", "trans_z",
        "rot_x", "rot_y", "rot_z"
    ]

    tissue_base = [
        "white_matter",
        "csf"
    ]

    if pipeline["use_gsr"]:
        tissue_base.append("global_signal")

    base_vars = motion_base + tissue_base

    selected = []

    for name in base_vars:
        candidate_cols = [name]

        if pipeline["use_derivatives"]:
            candidate_cols.append(f"{name}_derivative1")

        if pipeline["use_squares"]:
            candidate_cols.append(f"{name}_power2")

        if pipeline["use_derivatives"] and pipeline["use_squares"]:
            candidate_cols.append(f"{name}_derivative1_power2")

        for col in candidate_cols:
            if col in confounds.columns:
                selected.append(col)
            else:
                print(
                    f"警告：{os.path.basename(confounds_file)} 缺少列 {col}；"
                    f"该列不会进入混杂回归矩阵"
                )

    selected = list(dict.fromkeys(selected))

    if len(selected) > 0:
        base_confounds = confounds[selected].copy()
    else:
        base_confounds = pd.DataFrame(index=confounds.index)
        print(
            f"警告：{os.path.basename(confounds_file)} 未找到可用基础 confounds"
        )

    # --------------------------------------------------------
    # FD spike regressors
    # --------------------------------------------------------
    spike_df = pd.DataFrame(index=confounds.index)

    if pipeline["use_fd_spikes"]:
        threshold = pipeline["fd_spike_threshold"]

        if "framewise_displacement" not in confounds.columns:
            raise ValueError(
                f"{os.path.basename(confounds_file)} 缺少 framewise_displacement，"
                f"无法生成 FD spike regressors"
            )

        fd = (
            pd.to_numeric(confounds["framewise_displacement"], errors="coerce")
            .replace([np.inf, -np.inf], np.nan)
            .fillna(0)
        )

        spike_indices = np.where(fd.to_numpy() > threshold)[0]

        if len(spike_indices) > 0:
            spike_mat = np.zeros(
                (len(confounds), len(spike_indices)),
                dtype=float
            )

            for k, idx in enumerate(spike_indices):
                spike_mat[idx, k] = 1.0

            spike_df = pd.DataFrame(
                spike_mat,
                columns=[
                    f"spike_fd_gt_{threshold}_{idx}"
                    for idx in spike_indices
                ],
                index=confounds.index
            )

    # --------------------------------------------------------
    # 可选 non-steady-state regressors
    # 当前主分析未启用
    # --------------------------------------------------------
    nonsteady_df = pd.DataFrame(index=confounds.index)

    if pipeline["use_non_steady_state"]:
        nonsteady_cols = [
            c for c in confounds.columns
            if c.startswith("non_steady_state_outlier")
        ]

        if len(nonsteady_cols) > 0:
            nonsteady_df = confounds[nonsteady_cols].copy()

    confound_mat = pd.concat(
        [base_confounds, spike_df, nonsteady_df],
        axis=1
    )

    confound_mat = (
        confound_mat
        .replace([np.inf, -np.inf], np.nan)
        .fillna(0)
    )

    if confound_mat.shape[1] == 0:
        return None

    return confound_mat


# ============================================================
# BOLD 与 confounds 文件匹配
# ============================================================

def find_matching_confounds(fmri_file: str, confounds_files: list) -> str:
    """
    按 BIDS 实体匹配 BOLD 和 confounds。
    优先匹配 sub / ses / task / acq / run。
    """

    bold_entities = parse_bids_entities(fmri_file)
    bold_sub = normalize_sub_value(bold_entities.get("sub"))

    if bold_sub is None:
        return None

    candidates = []

    for cfile in confounds_files:
        conf_entities = parse_bids_entities(cfile)
        conf_sub = normalize_sub_value(conf_entities.get("sub"))

        if conf_sub != bold_sub:
            continue

        score = 0

        for key in ["ses", "task", "acq", "run"]:
            bold_value = bold_entities.get(key)
            conf_value = conf_entities.get(key)

            if bold_value is not None and conf_value is not None:
                if bold_value == conf_value:
                    score += 2
                else:
                    score -= 10

        if conf_entities.get("task") == "rest":
            score += 1

        candidates.append((score, cfile))

    if len(candidates) == 0:
        return None

    candidates.sort(key=lambda x: x[0], reverse=True)

    return candidates[0][1]


# ============================================================
# 单个被试处理函数
# ============================================================

def process_subject(subid, TR, atlas_filename, fmri_files, confounds_files):
    subid = str(subid)

    matched_files = [
        f for f in fmri_files
        if file_matches_subid(f, subid)
    ]

    if not matched_files:
        print(f"被试 {subid} 未找到对应 BOLD 文件，跳过")
        return [{
            "subid": subid,
            "pipeline": MAIN_PIPELINE_NAME,
            "status": "no_bold",
            "bold_file": "",
            "confounds_file": "",
            "mean_fd": np.nan,
            "n_timepoints": np.nan,
            "n_rois": np.nan,
            "n_confounds": np.nan,
            "output_file": ""
        }]

    records = []

    for fmri_file in matched_files:
        base = os.path.basename(fmri_file)

        print(f"处理被试 {subid}, TR={TR}, 文件: {base}")

        confounds_file = find_matching_confounds(fmri_file, confounds_files)

        if confounds_file is None:
            print(f"被试 {subid} 文件 {base} 未找到对应 confounds，跳过")

            records.append({
                "subid": subid,
                "pipeline": MAIN_PIPELINE_NAME,
                "status": "no_confounds",
                "bold_file": base,
                "confounds_file": "",
                "mean_fd": np.nan,
                "n_timepoints": np.nan,
                "n_rois": np.nan,
                "n_confounds": np.nan,
                "output_file": ""
            })

            continue

        print(f"使用 confounds: {os.path.basename(confounds_file)}")

        try:
            confounds = read_confounds(confounds_file)
        except Exception as e:
            print(
                f"读取 confounds 失败: {os.path.basename(confounds_file)}；"
                f"错误：{repr(e)}"
            )

            records.append({
                "subid": subid,
                "pipeline": MAIN_PIPELINE_NAME,
                "status": "confounds_read_failed",
                "bold_file": base,
                "confounds_file": os.path.basename(confounds_file),
                "mean_fd": np.nan,
                "n_timepoints": np.nan,
                "n_rois": np.nan,
                "n_confounds": np.nan,
                "output_file": ""
            })

            continue

        qc_passed, mean_fd, qc_status = pass_fd_qc(
            confounds,
            confounds_file
        )

        if not qc_passed:
            records.append({
                "subid": subid,
                "pipeline": MAIN_PIPELINE_NAME,
                "status": qc_status,
                "bold_file": base,
                "confounds_file": os.path.basename(confounds_file),
                "mean_fd": mean_fd,
                "n_timepoints": np.nan,
                "n_rois": np.nan,
                "n_confounds": np.nan,
                "output_file": ""
            })

            continue

        if base.endswith(".nii.gz"):
            out_name = base[:-7]
        elif base.endswith(".nii"):
            out_name = base[:-4]
        else:
            out_name = os.path.splitext(base)[0]

        for pipeline in PIPELINES:
            pipeline_name = pipeline["name"]
            pipeline_output_dir = os.path.join(output_root, pipeline_name)
            os.makedirs(pipeline_output_dir, exist_ok=True)

            print(f"被试 {subid}: 运行 {pipeline_name}")

            try:
                confound_mat = build_confounds(
                    confounds=confounds,
                    confounds_file=confounds_file,
                    pipeline=pipeline
                )
            except Exception as e:
                print(
                    f"构建 confounds 失败：{base}；"
                    f"pipeline={pipeline_name}；错误：{repr(e)}"
                )

                records.append({
                    "subid": subid,
                    "pipeline": pipeline_name,
                    "status": "confounds_build_failed",
                    "bold_file": base,
                    "confounds_file": os.path.basename(confounds_file),
                    "mean_fd": mean_fd,
                    "n_timepoints": np.nan,
                    "n_rois": np.nan,
                    "n_confounds": np.nan,
                    "output_file": ""
                })

                continue

            n_confounds = 0 if confound_mat is None else confound_mat.shape[1]

            masker = NiftiLabelsMasker(
                labels_img=atlas_filename,
                standardize="zscore_sample",
                detrend=True,
                low_pass=LOW_PASS,
                high_pass=HIGH_PASS,
                t_r=float(TR),
                resampling_target="data"
            )

            try:
                time_series = masker.fit_transform(
                    fmri_file,
                    confounds=confound_mat
                )
            except Exception as e:
                print(
                    f"处理失败：{base}；"
                    f"pipeline={pipeline_name}；错误：{repr(e)}"
                )

                records.append({
                    "subid": subid,
                    "pipeline": pipeline_name,
                    "status": "failed",
                    "bold_file": base,
                    "confounds_file": os.path.basename(confounds_file),
                    "mean_fd": mean_fd,
                    "n_timepoints": np.nan,
                    "n_rois": np.nan,
                    "n_confounds": n_confounds,
                    "output_file": ""
                })

                continue

            if time_series.shape[1] != N_ROIS:
                print(
                    f"警告：{base} 在 {pipeline_name} 中提取到的 ROI 数量为 "
                    f"{time_series.shape[1]}，期望为 {N_ROIS}"
                )

            ts_file = os.path.join(
                pipeline_output_dir,
                f"{out_name}_schaefer{N_ROIS}_{YEO_NETWORKS}yeo_"
                f"2mm_{pipeline_name}_0p01_0p1Hz_timeseries.npy"
            )

            np.save(ts_file, time_series)

            print(
                f"保存完成: {ts_file}, "
                f"shape={time_series.shape}, "
                f"meanFD={mean_fd:.4f}, "
                f"n_confounds={n_confounds}"
            )

            records.append({
                "subid": subid,
                "pipeline": pipeline_name,
                "status": "done",
                "bold_file": base,
                "confounds_file": os.path.basename(confounds_file),
                "mean_fd": mean_fd,
                "n_timepoints": time_series.shape[0],
                "n_rois": time_series.shape[1],
                "n_confounds": n_confounds,
                "output_file": ts_file
            })

    return records


# ============================================================
# Windows 安全入口
# ============================================================

if __name__ == "__main__":
    freeze_support()

    # --------------------------------------------------------
    # 1. 下载 / 加载 Schaefer 400 ROI + 7 Yeo networks + 2 mm
    # --------------------------------------------------------
    print("正在下载/加载 Schaefer 400 ROI (7 Yeo networks, 2 mm) 模板...")

    atlas = datasets.fetch_atlas_schaefer_2018(
        n_rois=N_ROIS,
        yeo_networks=YEO_NETWORKS,
        resolution_mm=RESOLUTION_MM
    )

    atlas_filename = atlas["maps"]

    print("模板加载完成:", atlas_filename)

    # --------------------------------------------------------
    # 2. 读取 TR CSV，并统一被试编号
    # --------------------------------------------------------
    tr_df = pd.read_csv(tr_csv_file)
    tr_df.columns = tr_df.columns.str.strip()

    if "subid" not in tr_df.columns or "TR" not in tr_df.columns:
        raise ValueError("TR CSV 必须包含列：subid 和 TR")

    tr_df["subid_norm"] = tr_df["subid"].apply(normalize_sub_value)
    tr_df = tr_df.dropna(subset=["subid_norm", "TR"]).copy()
    tr_df["subid_norm"] = tr_df["subid_norm"].astype(int).astype(str)

    tr_dict = dict(
        zip(
            tr_df["subid_norm"],
            tr_df["TR"]
        )
    )

    # --------------------------------------------------------
    # 3. 读取 phenotype CSV 白名单，并统一被试编号
    # --------------------------------------------------------
    pheno_df = pd.read_csv(phenotype_csv)
    pheno_df.columns = pheno_df.columns.str.strip()

    if "SUB_ID" not in pheno_df.columns:
        raise ValueError("phenotype CSV 必须包含列：SUB_ID")

    pheno_df["SUB_ID_norm"] = pheno_df["SUB_ID"].apply(normalize_sub_value)
    pheno_df = pheno_df.dropna(subset=["SUB_ID_norm"]).copy()
    pheno_df["SUB_ID_norm"] = pheno_df["SUB_ID_norm"].astype(int).astype(str)

    allowed_subids = set(pheno_df["SUB_ID_norm"])

    # --------------------------------------------------------
    # 4. 只处理同时存在于 TR 表和表型筛选列表中的被试
    # --------------------------------------------------------
    filtered_tr_dict = {
        sid: tr
        for sid, tr in tr_dict.items()
        if sid in allowed_subids
    }

    print(
        f"TR表: {len(tr_dict)} 人；"
        f"表型筛选列表: {len(allowed_subids)} 人；"
        f"最终进入 ROI 提取流程: {len(filtered_tr_dict)} 人"
    )

    # --------------------------------------------------------
    # 5. 扫描 BOLD 文件
    # --------------------------------------------------------
    fmri_files = (
        glob.glob(
            os.path.join(fmri_dir, "**", "*desc-preproc_bold.nii"),
            recursive=True
        )
        +
        glob.glob(
            os.path.join(fmri_dir, "**", "*desc-preproc_bold.nii.gz"),
            recursive=True
        )
    )

    print(f"扫描到 fMRIPrep preprocessed BOLD 文件数: {len(fmri_files)}")

    if len(fmri_files) == 0:
        print(
            "未找到 *desc-preproc_bold.nii(.gz)，"
            "改用宽松模式扫描所有 .nii / .nii.gz"
        )

        fmri_files = (
            glob.glob(
                os.path.join(fmri_dir, "**", "*.nii"),
                recursive=True
            )
            +
            glob.glob(
                os.path.join(fmri_dir, "**", "*.nii.gz"),
                recursive=True
            )
        )

    print(f"最终 BOLD 文件数: {len(fmri_files)}")

    if len(fmri_files) == 0:
        raise RuntimeError("没有找到 BOLD 文件，请检查 fmri_dir")

    # --------------------------------------------------------
    # 6. 扫描 confounds 文件
    # --------------------------------------------------------
    confounds_files = glob.glob(
        os.path.join(
            confounds_dir,
            "**",
            "*desc-confounds_timeseries.tsv"
        ),
        recursive=True
    )

    print(f"扫描到 confounds 文件数: {len(confounds_files)}")

    if len(confounds_files) == 0:
        raise RuntimeError(
            "没有找到 *desc-confounds_timeseries.tsv。"
            "请检查 confounds_dir 是否为 F:\\回归文件\\回归文件 或其他正确目录。"
        )

    # --------------------------------------------------------
    # 7. 并行处理
    # --------------------------------------------------------
    all_records = []

    with ProcessPoolExecutor(max_workers=n_jobs) as executor:
        futures = [
            executor.submit(
                process_subject,
                subid,
                TR,
                atlas_filename,
                fmri_files,
                confounds_files
            )
            for subid, TR in filtered_tr_dict.items()
        ]

        for future in as_completed(futures):
            records = future.result()
            all_records.extend(records)

    # --------------------------------------------------------
    # 8. 保存处理日志
    # --------------------------------------------------------
    report_df = pd.DataFrame(all_records)

    report_file = os.path.join(
        output_root,
        f"roi_timeseries_processing_report_{MAIN_PIPELINE_NAME}.csv"
    )

    report_df.to_csv(
        report_file,
        index=False,
        encoding="utf-8-sig"
    )

    # --------------------------------------------------------
    # 9. 保存主 pipeline 单独日志
    # --------------------------------------------------------
    pipeline_dir = os.path.join(output_root, MAIN_PIPELINE_NAME)

    pipeline_report_file = os.path.join(
        pipeline_dir,
        f"{MAIN_PIPELINE_NAME}_processing_report.csv"
    )

    report_df.to_csv(
        pipeline_report_file,
        index=False,
        encoding="utf-8-sig"
    )

    # --------------------------------------------------------
    # 10. 输出状态汇总
    # --------------------------------------------------------
    print("\n全部处理完成")
    print(f"总处理日志已保存: {report_file}")
    print(f"主 pipeline 日志已保存: {pipeline_report_file}")

    if "status" in report_df.columns:
        print("\n处理状态汇总:")
        print(report_df["status"].value_counts(dropna=False))

    done_df = report_df[report_df["status"] == "done"].copy()
    print(f"\n成功提取 ROI 时间序列的 BOLD run 数: {len(done_df)}")

    if len(done_df) > 0:
        unique_subjects = done_df["subid"].nunique()
        print(f"成功提取 ROI 时间序列的被试数: {unique_subjects}")