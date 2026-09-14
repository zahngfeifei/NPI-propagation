import os
import pandas as pd
import numpy as np
import matplotlib.pyplot as plt
from matplotlib.lines import Line2D

# =========================
# 1. File paths
# =========================
POS_ABIDE2_CSV = r"I:\DYF\NPI-3\2.梯度分析\正向连接-独立模板\ABIDE2_结果6_integrated_G1G2_absDistance_DMN_VisSMN\GLM_hierarchy_distances_G1_Default_dist_VisSomMot_main_effect.csv"

NEG_ABIDE1_CSV = r"I:\DYF\NPI-3\2.梯度分析\负向链接-独立模板\ABIDE1_结果6_integrated_G1G2_absDistance_DMN_VisSMN_负向\负向连接_GLM_hierarchy_distances_G1G2_main_effect_splitFDR.csv"

NEG_ABIDE2_CSV = r"I:\DYF\NPI-3\2.梯度分析\负向链接-独立模板\ABIDE2_结果6_integrated_G1G2_absDistance_DMN_VisSMN_负向\负向连接_GLM_hierarchy_distances_G1G2_main_effect_splitFDR.csv"

OUT_DIR = r"I:\DYF\NPI-3\补充内容新\S4"
os.makedirs(OUT_DIR, exist_ok=True)

OUT_FIG_PNG = os.path.join(
    OUT_DIR,
    "Supplementary_Fig_S4_Positive_ABIDE2_and_absNegative_absDefault_VisSomMot_G1_separation.png"
)

OUT_FIG_SVG = os.path.join(
    OUT_DIR,
    "Supplementary_Fig_S4_Positive_ABIDE2_and_absNegative_absDefault_VisSomMot_G1_separation.svg"
)

# =========================
# 2. Global font style
# =========================
FONT_NAME = "Arial"
FONT_SIZE = 21
STAT_FONT_SIZE = 15
LEGEND_FONT_SIZE = 15

plt.rcParams["font.family"] = FONT_NAME
plt.rcParams["font.size"] = FONT_SIZE
plt.rcParams["axes.titlesize"] = FONT_SIZE
plt.rcParams["axes.labelsize"] = FONT_SIZE
plt.rcParams["xtick.labelsize"] = FONT_SIZE
plt.rcParams["ytick.labelsize"] = FONT_SIZE
plt.rcParams["legend.fontsize"] = LEGEND_FONT_SIZE
plt.rcParams["figure.titlesize"] = FONT_SIZE
plt.rcParams["axes.linewidth"] = 1.2
plt.rcParams["svg.fonttype"] = "none"

# =========================
# 3. Helper functions
# =========================
def load_target_row(csv_path, dataset, ec_type):
    df = pd.read_csv(csv_path)
    df["dataset"] = dataset
    df["ec_type"] = ec_type

    df = df[df["term"].astype(str).str.contains("T.ASD", regex=False, na=False)].copy()

    target_items = [
        "G1_Default_dist_VisSomMot",
        "G1_Default_dist_VisSMN",
        "G1_Default_VisSomMot",
        "G1_Default_VisSMN"
    ]

    sub = df[
        df["item"].astype(str).isin(target_items) &
        df["dv"].astype(str).str.contains("G1", case=False, na=False)
    ].copy()

    if sub.empty:
        sub = df[
            df["item"].astype(str).str.contains("Default", case=False, na=False) &
            df["item"].astype(str).str.contains("Vis", case=False, na=False) &
            df["item"].astype(str).str.contains("Mot|SMN", case=False, na=False, regex=True) &
            df["dv"].astype(str).str.contains("G1", case=False, na=False)
        ].copy()

    if sub.empty:
        print(f"Warning: no target row found in {csv_path}")
        return pd.DataFrame()

    return sub.iloc[[0]].copy()


def abs_ci_interval(beta, ci_low, ci_high):
    beta_abs = abs(beta)

    if ci_low <= 0 <= ci_high:
        ci_low_abs = 0.0
        ci_high_abs = max(abs(ci_low), abs(ci_high))
    else:
        ci_low_abs = min(abs(ci_low), abs(ci_high))
        ci_high_abs = max(abs(ci_low), abs(ci_high))

    ci_low_abs = min(ci_low_abs, beta_abs)
    ci_high_abs = max(ci_high_abs, beta_abs)

    return beta_abs, ci_low_abs, ci_high_abs


# =========================
# 4. Load all results
# =========================
dfs = [
    load_target_row(POS_ABIDE2_CSV, "ABIDE II", "Positive EC"),
    load_target_row(NEG_ABIDE1_CSV, "ABIDE I", "|Negative EC|"),
    load_target_row(NEG_ABIDE2_CSV, "ABIDE II", "|Negative EC|")
]

df = pd.concat(dfs, ignore_index=True)
df["sig_fdr"] = df["p_FDR"] < 0.05

abs_values = df.apply(
    lambda r: abs_ci_interval(r["beta"], r["ci_low"], r["ci_high"]),
    axis=1,
    result_type="expand"
)

df["beta_plot"] = abs_values[0]
df["ci_low_plot"] = abs_values[1]
df["ci_high_plot"] = abs_values[2]

df["plot_label"] = df["ec_type"] + "\n" + df["dataset"]

plot_order = [
    ("Positive EC", "ABIDE II"),
    ("|Negative EC|", "ABIDE I"),
    ("|Negative EC|", "ABIDE II")
]

df["plot_order"] = df.apply(
    lambda r: plot_order.index((r["ec_type"], r["dataset"]))
    if (r["ec_type"], r["dataset"]) in plot_order else 999,
    axis=1
)

df = df.sort_values("plot_order").reset_index(drop=True)

print("Selected rows:")
print(df[[
    "ec_type", "dataset", "dv", "item",
    "beta", "ci_low", "ci_high",
    "beta_plot", "ci_low_plot", "ci_high_plot",
    "t", "p", "n", "p_FDR", "sig"
]])

# =========================
# 5. Figure
# =========================
fig, ax = plt.subplots(figsize=(15.0, 6.2))

color_map = {
    "Positive EC": "#3B6FB6",
    "|Negative EC|": "#D56B3D"
}

STAT_X = 0.132
X_MAX = 0.185

y_pos = np.arange(len(df))

for i, row in df.iterrows():
    beta = row["beta_plot"]
    ci_low = row["ci_low_plot"]
    ci_high = row["ci_high_plot"]

    xerr_low = max(0, beta - ci_low)
    xerr_high = max(0, ci_high - beta)

    color = color_map.get(row["ec_type"], "black")
    facecolor = color if row["sig_fdr"] else "white"

    ax.errorbar(
        beta,
        i,
        xerr=[[xerr_low], [xerr_high]],
        fmt="o",
        markersize=12,
        markerfacecolor=facecolor,
        markeredgecolor=color,
        markeredgewidth=2,
        ecolor=color,
        elinewidth=2.2,
        capsize=5,
        zorder=3
    )

    stat_text = (
        f"|β|={beta:.3f}, "
        f"95% |CI| [{ci_low:.3f}, {ci_high:.3f}], "
        f"q={row['p_FDR']:.3g}"
    )

    ax.text(
        STAT_X,
        i,
        stat_text,
        ha="left",
        va="center",
        fontname=FONT_NAME,
        fontsize=STAT_FONT_SIZE
    )

    if row["p_FDR"] < 0.001:
        star = "***"
    elif row["p_FDR"] < 0.01:
        star = "**"
    elif row["p_FDR"] < 0.05:
        star = "*"
    else:
        star = ""

    if star:
        ax.text(
            ci_high + 0.004,
            i,
            star,
            ha="left",
            va="center",
            fontname=FONT_NAME,
            fontsize=FONT_SIZE,
            color=color
        )

ax.axvline(
    0,
    color="black",
    linewidth=1.1,
    linestyle="--",
    zorder=1
)

ax.grid(
    axis="x",
    linestyle=":",
    linewidth=0.8,
    color="0.82",
    zorder=0
)

ax.set_yticks(y_pos)
ax.set_yticklabels(
    df["plot_label"].tolist(),
    fontname=FONT_NAME,
    fontsize=FONT_SIZE
)

ax.invert_yaxis()

ax.set_xlabel(
    "ASD–HC difference in |Default–VisSomMot G1 separation| (|β|)",
    fontname=FONT_NAME,
    fontsize=FONT_SIZE
)

ax.text(
    0.0,
    1.04,
    "|Default–VisSomMot| G1 separation in positive EC and |negative EC|",
    transform=ax.transAxes,
    ha="left",
    va="bottom",
    fontname=FONT_NAME,
    fontsize=FONT_SIZE
)

ax.set_xlim(0.000, X_MAX)

legend_elements = [
    Line2D(
        [0], [0],
        marker="o",
        color="#3B6FB6",
        markerfacecolor="#3B6FB6",
        markeredgecolor="#3B6FB6",
        markersize=10,
        linewidth=2,
        label="Positive EC"
    ),
    Line2D(
        [0], [0],
        marker="o",
        color="#D56B3D",
        markerfacecolor="#D56B3D",
        markeredgecolor="#D56B3D",
        markersize=10,
        linewidth=2,
        label="|Negative EC|"
    ),
    Line2D(
        [0], [0],
        marker="o",
        color="black",
        markerfacecolor="black",
        markeredgecolor="black",
        markersize=10,
        linewidth=0,
        label="FDR < 0.05"
    ),
    Line2D(
        [0], [0],
        marker="o",
        color="black",
        markerfacecolor="white",
        markeredgecolor="black",
        markersize=10,
        linewidth=0,
        label="FDR ≥ 0.05"
    )
]

ax.legend(
    handles=legend_elements,
    frameon=False,
    loc="center left",
    bbox_to_anchor=(1.02, 0.48),
    prop={"family": FONT_NAME, "size": LEGEND_FONT_SIZE}
)

ax.spines["top"].set_visible(False)
ax.spines["right"].set_visible(False)

for tick in ax.get_xticklabels() + ax.get_yticklabels():
    tick.set_fontname(FONT_NAME)
    tick.set_fontsize(FONT_SIZE)

plt.subplots_adjust(
    left=0.25,
    right=0.70,
    top=0.84,
    bottom=0.20
)

plt.savefig(
    OUT_FIG_PNG,
    dpi=600,
    bbox_inches="tight"
)

plt.savefig(
    OUT_FIG_SVG,
    format="svg",
    bbox_inches="tight"
)

plt.show()

print("Saved PNG:", OUT_FIG_PNG)
print("Saved SVG:", OUT_FIG_SVG)