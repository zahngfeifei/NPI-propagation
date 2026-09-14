import os
import re
import pandas as pd
from pathlib import Path

# =========================
# 1. 路径设置
# =========================

abide1_csv = r"I:\DYF\NPI-3\ABIDEⅠ.csv"
abide2_csv = r"I:\DYF\NPI-3\ABIDEⅡ.csv"

result_dir = r"I:\DYF\NPI-3\2.梯度分析\正向连接-独立模板\ABIDE_结果1"

output_csv = r"I:\DYF\NPI-3\被试信息\ABIDE_Ⅰ_Ⅱ_仅完成因果梯度被试_完整量表.csv"


# =========================
# 2. 读取 CSV
# =========================

def read_csv_safely(path):
    """
    自动尝试常见编码读取 CSV。
    """
    encodings = ["utf-8-sig", "utf-8", "gbk", "gb18030"]
    for enc in encodings:
        try:
            return pd.read_csv(path, encoding=enc)
        except UnicodeDecodeError:
            continue
    raise UnicodeDecodeError(f"无法读取文件编码: {path}")


df1 = read_csv_safely(abide1_csv)
df2 = read_csv_safely(abide2_csv)

df1["DATASET"] = "ABIDEI"
df2["DATASET"] = "ABIDEII"

# =========================
# 3. 合并两个量表文件
# =========================

df_all = pd.concat([df1, df2], axis=0, ignore_index=True, sort=False)

# 确保 SUB_ID 是字符串，便于和文件夹名匹配
df_all["SUB_ID"] = df_all["SUB_ID"].astype(str).str.strip()

# 去除可能出现的 .0
df_all["SUB_ID"] = df_all["SUB_ID"].str.replace(r"\.0$", "", regex=True)

# =========================
# 4. 扫描已经完成计算的被试文件夹
# =========================

result_path = Path(result_dir)

completed_subjects = []

for folder in result_path.iterdir():
    if folder.is_dir():
        folder_name = folder.name

        # 匹配 sub-Sub28742 这种格式
        match = re.match(r"sub-Sub(\d+)$", folder_name)
        if match:
            sub_id = match.group(1)
            completed_subjects.append(sub_id)

completed_subjects = sorted(set(completed_subjects))

print(f"结果文件夹中找到完成计算的被试数量: {len(completed_subjects)}")

# =========================
# 5. 只保留完成计算的被试
# =========================

df_completed = df_all[df_all["SUB_ID"].isin(completed_subjects)].copy()

print(f"量表文件中成功匹配到的被试数量: {df_completed.shape[0]}")

# =========================
# 6. 检查未匹配情况
# =========================

csv_subjects = set(df_all["SUB_ID"])
folder_subjects = set(completed_subjects)

folders_not_in_csv = sorted(folder_subjects - csv_subjects)
csv_not_in_folders = sorted(csv_subjects - folder_subjects)

print(f"结果文件夹中存在，但两个 CSV 中找不到的被试数量: {len(folders_not_in_csv)}")
print(f"两个 CSV 中存在，但结果文件夹中没有的被试数量: {len(csv_not_in_folders)}")

if len(folders_not_in_csv) > 0:
    print("\n结果文件夹中存在，但 CSV 中找不到的前 20 个被试:")
    print(folders_not_in_csv[:20])

# =========================
# 7. 按结果文件夹中的被试顺序排序
# =========================

subject_order = {sub: i for i, sub in enumerate(completed_subjects)}
df_completed["SUB_ORDER"] = df_completed["SUB_ID"].map(subject_order)

df_completed = df_completed.sort_values(
    by=["SUB_ORDER", "DATASET"]
).drop(columns=["SUB_ORDER"])

# =========================
# 8. 保存完整量表 CSV
# =========================

df_completed.to_csv(output_csv, index=False, encoding="utf-8-sig")

print("\n已保存完整量表文件:")
print(output_csv)