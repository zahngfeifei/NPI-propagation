# =====================================================
# EC-SEC ROI 定义文件（CSV 版本，最终冻结版）
# 从 Schaefer400 + Yeo7 网络标签生成
# + 自动生成 ROI 定义总结报告 txt
#
# Yeo7 网络：
# Cont / Default / DorsAttn / Limbic / SalVentAttn / SomMot / Vis
# =====================================================

import os
import pandas as pd
from datetime import datetime

# -----------------------------------------------------
# 路径设置
# -----------------------------------------------------
CSV_PATH = r"I:\DYF\NPI-4-code\3.步进分析\Schaefer400_7Yeo_network_labels_with_ID.csv"
OUT_DIR  = r"I:\DYF\NPI-4-code\3.步进分析\种子集合"
os.makedirs(OUT_DIR, exist_ok=True)

print("读取 CSV:")
print(CSV_PATH)

# -----------------------------------------------------
# 读取 CSV
# -----------------------------------------------------
df = pd.read_csv(CSV_PATH)

required_cols = ["ROI_ID", "ROI_label", "Yeo7_network"]
for col in required_cols:
    if col not in df.columns:
        raise ValueError(f"CSV 缺少必要列: {col}")

print(f"ROI 数量: {len(df)}")

# -----------------------------------------------------
# 1. 添加 0-based ROI 索引
# -----------------------------------------------------
df["ROI_index_0based"] = df["ROI_ID"] - 1

# -----------------------------------------------------
# 2. 定义种子集合 S（感觉系统）
# -----------------------------------------------------
SEED_NETWORKS = ["Vis", "SomMot"]

df_seed = df[df["Yeo7_network"].isin(SEED_NETWORKS)].copy()

seed_csv = os.path.join(OUT_DIR, "seeds_S_sensory.csv")
df_seed[
    ["ROI_index_0based", "ROI_ID", "ROI_label", "Yeo7_network"]
].to_csv(seed_csv, index=False, encoding="utf-8-sig")

print(f"种子集合 S 保存完成: {seed_csv}")
print(f"  ROI 数量 = {len(df_seed)}")

# -----------------------------------------------------
# 3. 定义层级系统 H1–H4（基于 Yeo7）
# -----------------------------------------------------
HIERARCHY_DEFINITION = {
    "H1_sensory":   ["Vis", "SomMot"],
    "H2_attention": ["DorsAttn", "SalVentAttn"],
    "H3_control":   ["Cont"],
    "H4_DMN":       ["Default"]
}

hierarchy_counts = {}

for h_name, networks in HIERARCHY_DEFINITION.items():

    df_h = df[df["Yeo7_network"].isin(networks)].copy()
    hierarchy_counts[h_name] = len(df_h)

    out_csv = os.path.join(OUT_DIR, f"{h_name}.csv")
    df_h[
        ["ROI_index_0based", "ROI_ID", "ROI_label", "Yeo7_network"]
    ].to_csv(out_csv, index=False, encoding="utf-8-sig")

    print(f"{h_name} 保存完成: {out_csv}")
    print(f"  ROI 数量 = {len(df_h)}")

# -----------------------------------------------------
# 4. 生成 ROI → 层级映射文件
# -----------------------------------------------------
def assign_hierarchy(yeo_net):
    for h_name, nets in HIERARCHY_DEFINITION.items():
        if yeo_net in nets:
            return h_name
    return "Other"   # Limbic

df["Hierarchy"] = df["Yeo7_network"].apply(assign_hierarchy)

mapping_csv = os.path.join(OUT_DIR, "roi_to_hierarchy_mapping.csv")
df[
    ["ROI_index_0based", "ROI_ID", "ROI_label", "Yeo7_network", "Hierarchy"]
].to_csv(mapping_csv, index=False, encoding="utf-8-sig")

print("\nROI → 层级 映射文件已保存:")
print(mapping_csv)

# -----------------------------------------------------
# 5. 生成总结报告 txt
# -----------------------------------------------------
summary_path = os.path.join(OUT_DIR, "roi_definition_summary.txt")

n_total = len(df)
n_seed  = len(df_seed)
n_other = int((df["Hierarchy"] == "Other").sum())
n_covered = n_total - n_other

# 检查层级重叠（理论上不应存在）
overlap_flag = False
for i, h1 in enumerate(HIERARCHY_DEFINITION.keys()):
    for h2 in list(HIERARCHY_DEFINITION.keys())[i+1:]:
        inter = set(
            df[df["Hierarchy"] == h1]["ROI_index_0based"]
        ).intersection(
            df[df["Hierarchy"] == h2]["ROI_index_0based"]
        )
        if len(inter) > 0:
            overlap_flag = True

with open(summary_path, "w", encoding="utf-8") as f:
    f.write("EC-SEC ROI Definition Summary Report\n")
    f.write(f"Generated at: {datetime.now()}\n\n")

    f.write("Input:\n")
    f.write(f"  CSV file: {CSV_PATH}\n")
    f.write(f"  Total ROIs: {n_total}\n\n")

    f.write("Seeds (Sensory system):\n")
    f.write(f"  Networks: {SEED_NETWORKS}\n")
    f.write(f"  ROI count: {n_seed}\n\n")

    f.write("Hierarchy definition (Yeo7-based):\n")
    for h, nets in HIERARCHY_DEFINITION.items():
        f.write(f"  {h}: {nets}, ROI count = {hierarchy_counts[h]}\n")
    f.write("\n")

    f.write("Coverage:\n")
    f.write(f"  ROIs in H1–H4: {n_covered}\n")
    f.write(f"  ROIs labeled as Other (e.g., Limbic): {n_other}\n\n")

    f.write("Quality control:\n")
    f.write(f"  Hierarchy overlap detected: {overlap_flag}\n")
    f.write("  Note: Limbic network is not included in the main hierarchy chain.\n")

print("\n=== EC-SEC ROI 定义完成（含总结报告） ===")
print("输出目录:", OUT_DIR)
print("总结报告:", summary_path)

print("\n生成的文件列表:")
for fname in sorted(os.listdir(OUT_DIR)):
    print(" -", fname)
