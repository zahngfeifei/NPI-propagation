# ================================
# Procrustes 梯度向量 ComBat：敏感性分析适配版
# + 输入结构：
#   INPUT_ROOT/p05/k01/sub-xxx/out_G1_procrustes.npy
#   INPUT_ROOT/p05/k03/sub-xxx/out_G1_procrustes.npy
#   INPUT_ROOT/p05/k03/sub-xxx/out_G2_procrustes.npy
#   ...
# + 阈值敏感性：p10, p15, p20, p25
# + 对齐成分敏感性：k03, k05, k07, k09
# + k01 只处理 G1
# + k03/k05/k07/k09 处理 G1 和 G2
# ================================

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

INPUT_ROOT = Path(
    r"I:\DYF\NPI-4-code\2.梯度分析\正向连接-参数敏感性分析\ABIDE2_结果2_sensitivity"
)

OUTPUT_ROOT = Path(
    r"I:\DYF\NPI-4-code\2.梯度分析\正向连接-参数敏感性分析\ABIDE2_结果2_sensitivity_combat"
)

# CSV 需要包含：
# sub_id, Group, Age, Sex, Site
# 可选包含：FIQ
DEMO_CSV = Path(r"I:\DYF\NPI-3\subject_info_for_stats.csv")

OUTPUT_ROOT.mkdir(parents=True, exist_ok=True)


# ============================================================
# 2. Sensitivity settings
# ============================================================

P_THRESHOLDS = [0.10, 0.15, 0.20, 0.25]
K_ALIGN_LIST = [3, 5, 7, 9]


def p_label(p):
    return f"p{int(round(p * 100)):02d}"


def k_label(k):
    return f"k{k:02d}"


# ============================================================
# 3. ComBat settings
# ============================================================

EXPECTED_FEATURES = 400
EPS_VARIANCE = 1e-12
OVERWRITE_OUTPUT_FILES = True

BATCH_COL = "Site"
CATEGORICAL_COLS = ["Group", "Sex"]

# MeanFD 不作为 ComBat 连续协变量
CONTINUOUS_CANDIDATES = ["Age", "FIQ"]

GRADIENT_FILES_ALL = {
    "G1": "out_G1_procrustes.npy",
    "G2": "out_G2_procrustes.npy",
}


def get_gradient_files_for_k(k_align):
    """
    k01 只有 G1；
    k03/k05/k07/k09 有 G1 和 G2。
    """
    if k_align < 2:
        return {
            "G1": "out_G1_procrustes.npy"
        }

    return {
        "G1": "out_G1_procrustes.npy",
        "G2": "out_G2_procrustes.npy",
    }


# ============================================================
# 4. Subject ID utilities
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


def get_sub_id_from_folder(folder_name):
    first_entity = str(folder_name).split("_", 1)[0]
    return normalize_sub_id(first_entity)


# ============================================================
# 5. Covariates
# ============================================================

def load_covariates(csv_file):
    header_columns = pd.read_csv(
        csv_file,
        nrows=0,
        encoding="utf-8-sig"
    ).columns

    required_columns = ["sub_id", "Group", "Age", "Sex", "Site"]

    missing_columns = [
        col for col in required_columns
        if col not in header_columns
    ]

    if missing_columns:
        raise ValueError(
            f"Missing required covariate columns in {csv_file}: {missing_columns}"
        )

    available_continuous_cols = [
        col for col in CONTINUOUS_CANDIDATES
        if col in header_columns
    ]

    use_columns = required_columns + [
        col for col in available_continuous_cols
        if col not in required_columns
    ]

    covariate_table = pd.read_csv(
        csv_file,
        dtype=str,
        encoding="utf-8-sig",
        usecols=use_columns,
    )

    covariate_table["SUB_ID_norm"] = covariate_table["sub_id"].apply(normalize_sub_id)

    for col in ["Group", "Sex", "Site"]:
        covariate_table[col] = covariate_table[col].astype(str).str.strip()

    for col in available_continuous_cols:
        covariate_table[col] = pd.to_numeric(
            covariate_table[col],
            errors="coerce"
        )

    required_for_combat = [
        "SUB_ID_norm",
        "Group",
        "Age",
        "Sex",
        "Site",
    ] + [
        col for col in available_continuous_cols
        if col not in ["Age"]
    ]

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

    duplicated_rows = covariate_table[
        covariate_table.duplicated(subset="SUB_ID_norm", keep=False)
    ]

    if len(duplicated_rows) > 0:
        print("Warning: duplicate subject rows found. Keeping the first row per subject.")
        print(
            duplicated_rows[
                ["sub_id", "Group", "Age", "Sex", "Site"]
                + [col for col in ["FIQ"] if col in covariate_table.columns]
            ]
        )

    covariate_table = covariate_table.drop_duplicates(
        subset="SUB_ID_norm",
        keep="first",
    )

    continuous_columns = [
        col for col in available_continuous_cols
        if col in covariate_table.columns
    ]

    return covariate_table, continuous_columns


# ============================================================
# 6. Input matching
# ============================================================

def find_subject_folders(combo_input_root):
    subject_folders = sorted(
        [p for p in combo_input_root.glob("sub-*") if p.is_dir()]
    )
    return subject_folders


def match_subjects(
    subject_folders,
    covariate_table,
    combo_output_root,
    gradient_files,
    report_dir,
):
    covariate_by_sub_id = {
        row["SUB_ID_norm"]: row
        for _, row in covariate_table.iterrows()
    }

    matched_records = []
    missing_covariates = []
    missing_gradient_files = []

    for folder in subject_folders:
        folder_name = folder.name
        normalized_sub_id = get_sub_id_from_folder(folder_name)

        if normalized_sub_id not in covariate_by_sub_id:
            missing_covariates.append(
                {
                    "sub_id": normalized_sub_id,
                    "folder_name": folder_name,
                    "folder": str(folder),
                }
            )
            continue

        gradient_paths = {
            grad_name: folder / file_name
            for grad_name, file_name in gradient_files.items()
        }

        missing_files = [
            str(path) for path in gradient_paths.values()
            if not path.exists()
        ]

        if missing_files:
            missing_gradient_files.append(
                {
                    "sub_id": normalized_sub_id,
                    "folder_name": folder_name,
                    "missing_files": "; ".join(missing_files),
                }
            )
            continue

        covariate_row = covariate_by_sub_id[normalized_sub_id]
        output_dir = combo_output_root / folder_name

        record = {
            "sub_id": normalized_sub_id,
            "folder_name": folder_name,
            "input_dir": folder,
            "output_dir": output_dir,
            "Group": str(covariate_row["Group"]),
            "Age": float(covariate_row["Age"]),
            "Sex": str(covariate_row["Sex"]),
            "Site": str(covariate_row["Site"]),
        }

        if "FIQ" in covariate_row.index:
            record["FIQ"] = float(covariate_row["FIQ"])

        for grad_name, path in gradient_paths.items():
            record[f"{grad_name}_input_file"] = path
            record[f"{grad_name}_output_file"] = output_dir / gradient_files[grad_name]
            record[f"{grad_name}_output_alias_file"] = (
                output_dir / gradient_files[grad_name].replace(".npy", "_combat.npy")
            )

        matched_records.append(record)

    if missing_covariates:
        missing_covariates_csv = report_dir / "subjects_missing_covariates.csv"
        pd.DataFrame(missing_covariates).to_csv(
            missing_covariates_csv,
            index=False,
            encoding="utf-8-sig",
        )
        print(f"Subjects missing covariates: {len(missing_covariates)}")
        print(f"Missing covariate report: {missing_covariates_csv}")

    if missing_gradient_files:
        missing_gradient_csv = report_dir / "subjects_missing_gradient_files.csv"
        pd.DataFrame(missing_gradient_files).to_csv(
            missing_gradient_csv,
            index=False,
            encoding="utf-8-sig",
        )
        print(f"Subjects missing gradient files: {len(missing_gradient_files)}")
        print(f"Missing gradient report: {missing_gradient_csv}")

    return matched_records, missing_covariates, missing_gradient_files


# ============================================================
# 7. Gradient loading
# ============================================================

def load_gradient_vector(npy_file):
    arr = np.load(npy_file).astype(np.float64, copy=False)
    arr = np.squeeze(arr)

    if arr.ndim != 1:
        raise ValueError(
            f"Gradient file is not a 1D vector after squeeze: shape={arr.shape}"
        )

    if EXPECTED_FEATURES is not None and arr.shape[0] != EXPECTED_FEATURES:
        raise ValueError(
            f"Gradient feature count is {arr.shape[0]}, expected {EXPECTED_FEATURES}"
        )

    if not np.isfinite(arr).all():
        raise ValueError("Gradient vector contains NaN or Inf")

    return arr


def load_gradients(records, gradient_files, report_dir):
    """
    Returns:
        gradient_data_dict:
            {
                "G1": array, shape = subjects x features,
                "G2": array, shape = subjects x features,
            }
        valid_records
    """
    gradient_lists = {
        grad_name: []
        for grad_name in gradient_files.keys()
    }

    valid_records = []
    failed_records = []

    for record in records:
        try:
            loaded = {}

            for grad_name in gradient_files.keys():
                loaded[grad_name] = load_gradient_vector(
                    record[f"{grad_name}_input_file"]
                )

            for grad_name in gradient_files.keys():
                gradient_lists[grad_name].append(loaded[grad_name])

            valid_records.append(record)

        except Exception as exc:
            failed_records.append(
                {
                    "sub_id": record["sub_id"],
                    "folder_name": record["folder_name"],
                    "error": f"{type(exc).__name__}: {exc}",
                }
            )

    if failed_records:
        failed_csv = report_dir / "subjects_failed_gradient_loading.csv"
        pd.DataFrame(failed_records).to_csv(
            failed_csv,
            index=False,
            encoding="utf-8-sig",
        )
        print(f"Failed gradient loads: {len(failed_records)}")
        print(f"Failed load report: {failed_csv}")

    if not valid_records:
        raise RuntimeError("No valid gradient files were loaded.")

    gradient_data_dict = {
        grad_name: np.stack(gradient_lists[grad_name], axis=0)
        for grad_name in gradient_files.keys()
    }

    return gradient_data_dict, valid_records, failed_records


# ============================================================
# 8. ComBat covariate table
# ============================================================

def build_combat_covariates(records, continuous_columns):
    covariates = pd.DataFrame(
        {
            "Site": [record["Site"] for record in records],
            "Group": [record["Group"] for record in records],
            "Age": [record["Age"] for record in records],
            "Sex": [record["Sex"] for record in records],
        }
    )

    if "FIQ" in continuous_columns:
        covariates["FIQ"] = [record["FIQ"] for record in records]

    covariates["Site"] = covariates["Site"].astype(str)
    covariates["Group"] = covariates["Group"].astype(str)
    covariates["Sex"] = covariates["Sex"].astype(str)

    for col in continuous_columns:
        covariates[col] = pd.to_numeric(
            covariates[col],
            errors="raise"
        )

    return covariates


def check_batch_balance(covariates):
    site_counts = covariates["Site"].value_counts()

    singleton_sites = site_counts[site_counts < 2]

    if len(singleton_sites) > 0:
        print("\nWarning: some sites have fewer than 2 subjects.")
        print("ComBat estimates may be unstable or may fail for these sites:")
        print(singleton_sites)

    site_group_table = pd.crosstab(
        covariates["Site"],
        covariates["Group"]
    )

    print("\nSITE x Group table:")
    print(site_group_table)

    return site_counts, site_group_table


# ============================================================
# 9. Run ComBat
# ============================================================

def run_gradient_combat(
    gradient_matrix_subject_by_feature,
    covariates,
    continuous_columns,
    gradient_name,
):
    """
    Input:
        gradient_matrix_subject_by_feature:
            shape = subjects x features

    neuroCombat requires:
        dat:
            shape = features x subjects
    """

    data_feature_by_subject = gradient_matrix_subject_by_feature.T

    finite_mask = np.isfinite(data_feature_by_subject).all(axis=1)
    variance_by_feature = np.var(data_feature_by_subject, axis=1)

    valid_feature_mask = finite_mask & (variance_by_feature > EPS_VARIANCE)

    total_feature_count = int(data_feature_by_subject.shape[0])
    valid_feature_count = int(np.sum(valid_feature_mask))

    if valid_feature_count == 0:
        raise RuntimeError(f"No valid features for ComBat in {gradient_name}.")

    print(f"\n========== Running ComBat for {gradient_name} ==========")
    print(f"Total features: {total_feature_count}")
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

    feature_report = {
        "gradient": gradient_name,
        "total_features": total_feature_count,
        "combat_features": valid_feature_count,
        "copied_features": total_feature_count - valid_feature_count,
        "raw_mean": float(np.mean(data_feature_by_subject)),
        "raw_std": float(np.std(data_feature_by_subject)),
        "combat_mean": float(np.mean(harmonized_feature_by_subject)),
        "combat_std": float(np.std(harmonized_feature_by_subject)),
    }

    return harmonized_subject_by_feature, valid_feature_mask, feature_report


# ============================================================
# 10. Save outputs
# ============================================================

def save_combat_gradients(
    records,
    gradient_files,
    combat_gradient_dict,
    report_subjects_csv,
):
    output_rows = []

    for subject_index, record in enumerate(records):
        output_dir = record["output_dir"]
        output_dir.mkdir(parents=True, exist_ok=True)

        row = {
            "sub_id": record["sub_id"],
            "folder_name": record["folder_name"],
            "input_dir": str(record["input_dir"]),
            "output_dir": str(output_dir),
            "Site": record["Site"],
            "Group": record["Group"],
            "Age": record["Age"],
            "Sex": record["Sex"],
            "FIQ": record.get("FIQ", np.nan),
        }

        for grad_name, combat_matrix in combat_gradient_dict.items():
            combat_vector = combat_matrix[subject_index, :]

            output_file = record[f"{grad_name}_output_file"]
            output_alias_file = record[f"{grad_name}_output_alias_file"]

            if output_file.exists() and not OVERWRITE_OUTPUT_FILES:
                raise FileExistsError(f"Output already exists: {output_file}")

            np.save(output_file, combat_vector)
            np.save(output_alias_file, combat_vector)

            row[f"{grad_name}_input_file"] = str(record[f"{grad_name}_input_file"])
            row[f"{grad_name}_output_file"] = str(output_file)
            row[f"{grad_name}_output_alias_file"] = str(output_alias_file)
            row[f"{grad_name}_combat_min"] = float(np.min(combat_vector))
            row[f"{grad_name}_combat_max"] = float(np.max(combat_vector))
            row[f"{grad_name}_combat_mean"] = float(np.mean(combat_vector))
            row[f"{grad_name}_combat_std"] = float(np.std(combat_vector))

        output_rows.append(row)

    pd.DataFrame(output_rows).to_csv(
        report_subjects_csv,
        index=False,
        encoding="utf-8-sig",
    )


def save_reports(
    records,
    covariates,
    continuous_columns,
    feature_reports,
    valid_feature_masks,
    combo_input_root,
    combo_output_root,
    report_subjects_csv,
    report_features_csv,
    report_txt,
):
    pd.DataFrame(feature_reports).to_csv(
        report_features_csv,
        index=False,
        encoding="utf-8-sig",
    )

    for grad_name, mask in valid_feature_masks.items():
        mask_file = combo_output_root / f"{grad_name}_valid_feature_mask.npy"
        np.save(mask_file, mask)

    with open(report_txt, "w", encoding="utf-8") as report_file:
        report_file.write("Gradient vector ComBat report\n")
        report_file.write("=============================\n\n")

        report_file.write("Paths\n")
        report_file.write(f"  combo INPUT_ROOT: {combo_input_root}\n")
        report_file.write(f"  combo OUTPUT_ROOT: {combo_output_root}\n")
        report_file.write(f"  DEMO_CSV: {DEMO_CSV}\n\n")

        report_file.write("ComBat settings\n")
        report_file.write("  dat: gradient ROI/features, features x subjects\n")
        report_file.write(f"  batch_col: {BATCH_COL}\n")
        report_file.write(f"  categorical_cols: {', '.join(CATEGORICAL_COLS)}\n")
        report_file.write(f"  continuous_cols: {', '.join(continuous_columns)}\n")
        report_file.write(f"  EXPECTED_FEATURES: {EXPECTED_FEATURES}\n")
        report_file.write(f"  EPS_VARIANCE: {EPS_VARIANCE}\n\n")

        report_file.write("Sample size\n")
        report_file.write(f"  matched valid subjects: {len(records)}\n\n")

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
            pd.crosstab(
                covariates["Site"],
                covariates["Group"]
            ).to_string()
        )
        report_file.write("\n\n")

        report_file.write("Feature summary\n")
        for feature_report in feature_reports:
            report_file.write(f"  Gradient: {feature_report['gradient']}\n")
            for key, value in feature_report.items():
                if key != "gradient":
                    report_file.write(f"    {key}: {value}\n")
            report_file.write("\n")

        report_file.write("Outputs\n")
        report_file.write(f"  subject report: {report_subjects_csv}\n")
        report_file.write(f"  feature report: {report_features_csv}\n")
        report_file.write(f"  text report: {report_txt}\n")


# ============================================================
# 11. Run one sensitivity combination
# ============================================================

def run_one_combination(
    p,
    k_align,
    covariate_table,
    continuous_columns,
):
    ptag = p_label(p)
    ktag = k_label(k_align)

    combo_input_root = INPUT_ROOT / ptag / ktag
    combo_output_root = OUTPUT_ROOT / ptag / ktag

    combo_output_root.mkdir(parents=True, exist_ok=True)

    report_subjects_csv = combo_output_root / "gradient_combat_subjects.csv"
    report_features_csv = combo_output_root / "gradient_combat_features_report.csv"
    report_txt = combo_output_root / "gradient_combat_report.txt"

    gradient_files = get_gradient_files_for_k(k_align)

    print("\n" + "=" * 100)
    print(f"Running ComBat: {ptag}, {ktag}")
    print(f"Input:  {combo_input_root}")
    print(f"Output: {combo_output_root}")
    print(f"Gradients: {list(gradient_files.keys())}")
    print("=" * 100)

    if not combo_input_root.exists():
        raise RuntimeError(f"Combination input directory does not exist: {combo_input_root}")

    subject_folders = find_subject_folders(combo_input_root)
    print(f"Found subject folders: {len(subject_folders)}")

    if not subject_folders:
        raise RuntimeError(f"No subject folders found under {combo_input_root}")

    matched_records, missing_covariates, missing_gradient_files = match_subjects(
        subject_folders=subject_folders,
        covariate_table=covariate_table,
        combo_output_root=combo_output_root,
        gradient_files=gradient_files,
        report_dir=combo_output_root,
    )

    print(f"Matched subjects with covariates and gradient files: {len(matched_records)}")

    if not matched_records:
        raise RuntimeError("No subjects matched covariates and gradient files.")

    gradient_data_dict, valid_records, failed_gradient_loading = load_gradients(
        records=matched_records,
        gradient_files=gradient_files,
        report_dir=combo_output_root,
    )

    print(f"Valid subjects loaded: {len(valid_records)}")

    for grad_name, data in gradient_data_dict.items():
        print(f"{grad_name} data shape, subjects x features: {data.shape}")

    covariates = build_combat_covariates(
        records=valid_records,
        continuous_columns=continuous_columns,
    )

    print("\n========== ComBat covariate overview ==========")
    print("Site counts:")
    print(covariates["Site"].value_counts())

    print("\nGroup counts:")
    print(covariates["Group"].value_counts())

    print("\nSex counts:")
    print(covariates["Sex"].value_counts())

    print("\nContinuous covariates:")
    print(continuous_columns)

    check_batch_balance(covariates)

    combat_gradient_dict = {}
    valid_feature_masks = {}
    feature_reports = []

    for grad_name, gradient_matrix in gradient_data_dict.items():
        combat_gradient, valid_feature_mask, feature_report = run_gradient_combat(
            gradient_matrix_subject_by_feature=gradient_matrix,
            covariates=covariates,
            continuous_columns=continuous_columns,
            gradient_name=grad_name,
        )

        combat_gradient_dict[grad_name] = combat_gradient
        valid_feature_masks[grad_name] = valid_feature_mask
        feature_reports.append(feature_report)

    print("\n========== Saving ComBat gradients ==========")

    save_combat_gradients(
        records=valid_records,
        gradient_files=gradient_files,
        combat_gradient_dict=combat_gradient_dict,
        report_subjects_csv=report_subjects_csv,
    )

    save_reports(
        records=valid_records,
        covariates=covariates,
        continuous_columns=continuous_columns,
        feature_reports=feature_reports,
        valid_feature_masks=valid_feature_masks,
        combo_input_root=combo_input_root,
        combo_output_root=combo_output_root,
        report_subjects_csv=report_subjects_csv,
        report_features_csv=report_features_csv,
        report_txt=report_txt,
    )

    combo_summary = {
        "p": p,
        "p_label": ptag,
        "K_ALIGN": k_align,
        "k_label": ktag,
        "status": "success",
        "input_root": str(combo_input_root),
        "output_root": str(combo_output_root),
        "gradient_names": ";".join(gradient_files.keys()),
        "n_subject_folders": len(subject_folders),
        "n_missing_covariates": len(missing_covariates),
        "n_missing_gradient_files": len(missing_gradient_files),
        "n_matched_records": len(matched_records),
        "n_failed_gradient_loading": len(failed_gradient_loading),
        "n_valid_subjects": len(valid_records),
        "continuous_columns": ";".join(continuous_columns),
        "report_subjects_csv": str(report_subjects_csv),
        "report_features_csv": str(report_features_csv),
        "report_txt": str(report_txt),
        "error": "",
    }

    for feature_report in feature_reports:
        grad_name = feature_report["gradient"]
        combo_summary[f"{grad_name}_combat_features"] = feature_report["combat_features"]
        combo_summary[f"{grad_name}_copied_features"] = feature_report["copied_features"]
        combo_summary[f"{grad_name}_raw_std"] = feature_report["raw_std"]
        combo_summary[f"{grad_name}_combat_std"] = feature_report["combat_std"]

    return combo_summary


# ============================================================
# 12. Main
# ============================================================

def main():
    print("========== Sensitivity Gradient vector ComBat ==========")
    print(f"INPUT_ROOT: {INPUT_ROOT}")
    print(f"OUTPUT_ROOT: {OUTPUT_ROOT}")
    print(f"DEMO_CSV: {DEMO_CSV}")

    covariate_table, continuous_columns = load_covariates(DEMO_CSV)

    print("\n========== Loaded covariates ==========")
    print(f"Covariate rows after QC: {len(covariate_table)}")
    print(f"Continuous columns: {continuous_columns}")

    overall_rows = []

    for p in P_THRESHOLDS:
        for k_align in K_ALIGN_LIST:

            try:
                combo_summary = run_one_combination(
                    p=p,
                    k_align=k_align,
                    covariate_table=covariate_table,
                    continuous_columns=continuous_columns,
                )

                overall_rows.append(combo_summary)

                print(
                    f"✔ Done: {p_label(p)}, {k_label(k_align)}, "
                    f"valid subjects={combo_summary['n_valid_subjects']}"
                )

            except Exception as exc:
                print(f"✘ Failed: {p_label(p)}, {k_label(k_align)}: {exc}")

                overall_rows.append(
                    {
                        "p": p,
                        "p_label": p_label(p),
                        "K_ALIGN": k_align,
                        "k_label": k_label(k_align),
                        "status": "failed",
                        "input_root": str(INPUT_ROOT / p_label(p) / k_label(k_align)),
                        "output_root": str(OUTPUT_ROOT / p_label(p) / k_label(k_align)),
                        "gradient_names": ";".join(get_gradient_files_for_k(k_align).keys()),
                        "n_subject_folders": np.nan,
                        "n_missing_covariates": np.nan,
                        "n_missing_gradient_files": np.nan,
                        "n_matched_records": np.nan,
                        "n_failed_gradient_loading": np.nan,
                        "n_valid_subjects": 0,
                        "continuous_columns": ";".join(continuous_columns),
                        "report_subjects_csv": "",
                        "report_features_csv": "",
                        "report_txt": "",
                        "error": f"{type(exc).__name__}: {exc}",
                    }
                )

                continue

    overall_csv = OUTPUT_ROOT / "overall_gradient_combat_sensitivity_summary.csv"
    pd.DataFrame(overall_rows).to_csv(
        overall_csv,
        index=False,
        encoding="utf-8-sig",
    )

    overall_txt = OUTPUT_ROOT / "overall_gradient_combat_sensitivity_report.txt"

    with open(overall_txt, "w", encoding="utf-8") as f:
        f.write("Overall gradient ComBat sensitivity report\n")
        f.write("=========================================\n\n")

        f.write("Design\n")
        f.write(f"  Thresholds: {P_THRESHOLDS}\n")
        f.write(f"  K_ALIGN_LIST: {K_ALIGN_LIST}\n")
        f.write("  k01 processes G1 only.\n")
        f.write("  k03/k05/k07/k09 process G1 and G2.\n\n")

        f.write("Paths\n")
        f.write(f"  INPUT_ROOT: {INPUT_ROOT}\n")
        f.write(f"  OUTPUT_ROOT: {OUTPUT_ROOT}\n")
        f.write(f"  DEMO_CSV: {DEMO_CSV}\n\n")

        f.write("ComBat settings\n")
        f.write(f"  batch_col: {BATCH_COL}\n")
        f.write(f"  categorical_cols: {', '.join(CATEGORICAL_COLS)}\n")
        f.write(f"  continuous_columns: {', '.join(continuous_columns)}\n")
        f.write(f"  EXPECTED_FEATURES: {EXPECTED_FEATURES}\n")
        f.write(f"  EPS_VARIANCE: {EPS_VARIANCE}\n\n")

        f.write("Combination results\n")
        f.write("=" * 100 + "\n\n")

        for row in overall_rows:
            f.write(
                f"{row['p_label']} | {row['k_label']} | "
                f"status={row['status']} | "
                f"valid_subjects={row['n_valid_subjects']} | "
                f"gradients={row['gradient_names']}\n"
            )

            if row["status"] != "success":
                f.write(f"  error: {row['error']}\n")

            f.write("\n")

        f.write("Outputs\n")
        f.write(f"  overall CSV: {overall_csv}\n")
        f.write(f"  overall report: {overall_txt}\n")

    print("\n========== Done ==========")
    print(f"ComBat output directory: {OUTPUT_ROOT}")
    print(f"Overall summary CSV: {overall_csv}")
    print(f"Overall report TXT: {overall_txt}")


if __name__ == "__main__":
    main()
