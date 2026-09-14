import os
import pandas as pd
import numpy as np
import matplotlib.pyplot as plt

# =========================
# 1. File paths
# =========================
ABIDE1_CSV = r"I:\DYF\NPI-3\2.梯度分析\正向连接-独立模板\ABIDE1_结果6_integrated_G1G2_absDistance_DMN_VisSMN\GLM_Yeo7_network_G1G2_main_effect_splitFDR.csv"
ABIDE2_CSV = r"I:\DYF\NPI-3\2.梯度分析\正向连接-独立模板\ABIDE2_结果6_integrated_G1G2_absDistance_DMN_VisSMN\GLM_Yeo7_network_G1G2_main_effect_splitFDR.csv"

OUT_DIR = r"I:\DYF\NPI-3\补充内容新\S3"
os.makedirs(OUT_DIR, exist_ok=True)

OUT_FIG_PNG = os.path.join(
    OUT_DIR,
    "Supplementary_Fig_S3_Yeo7_network_centroid_G1_replication.png"
)

OUT_FIG_SVG = os.path.join(
    OUT_DIR,
    "Supplementary_Fig_S3_Yeo7_network_centroid_G1_replication.svg"
)

# =========================
# 2. Global font style
# =========================
FONT_NAME = "Arial"
FONT_SIZE = 21

plt.rcParams["font.family"] = FONT_NAME
plt.rcParams["font.size"] = FONT_SIZE
plt.rcParams["axes.titlesize"] = FONT_SIZE
plt.rcParams["axes.labelsize"] = FONT_SIZE
plt.rcParams["xtick.labelsize"] = FONT_SIZE
plt.rcParams["ytick.labelsize"] = FONT_SIZE
plt.rcParams["legend.fontsize"] = FONT_SIZE
plt.rcParams["figure.titlesize"] = FONT_SIZE
plt.rcParams["axes.linewidth"] = 1.2
plt.rcParams["pdf.fonttype"] = 42
plt.rcParams["ps.fonttype"] = 42
plt.rcParams["svg.fonttype"] = "none"

# =========================
# 3. Load data
# =========================
df1 = pd.read_csv(ABIDE1_CSV)
df2 = pd.read_csv(ABIDE2_CSV)

df1["dataset"] = "ABIDE I"
df2["dataset"] = "ABIDE II"

df = pd.concat([df1, df2], ignore_index=True)

df = df[df["term"].astype(str).str.contains("T.ASD", regex=False)].copy()

TARGET_DV = "G1_network"
df = df[df["dv"].astype(str).eq(TARGET_DV)].copy()

# =========================
# 4. Network order and abbreviation labels
# =========================
network_order = [
    "Vis",
    "SomMot",
    "DorsAttn",
    "SalVentAttn",
    "Limbic",
    "Cont",
    "Default"
]

network_label_map = {
    "Vis": "Vis",
    "SomMot": "SMN",
    "DorsAttn": "DAN",
    "SalVentAttn": "VAN",
    "Limbic": "Lim",
    "Cont": "Cont",
    "Default": "DMN"
}

df = df[df["item"].isin(network_order)].copy()
df["item"] = pd.Categorical(
    df["item"],
    categories=network_order,
    ordered=True
)
df = df.sort_values(["item", "dataset"]).reset_index(drop=True)

df["sig_fdr"] = df["p_FDR"] < 0.05

# =========================
# 5. Basic check
# =========================
expected_n = len(network_order) * 2
if len(df) != expected_n:
    print("Warning: expected", expected_n, "rows, but found", len(df))
    print(df[["dataset", "dv", "item", "beta", "p_FDR"]])

# =========================
# 6. Figure layout
# =========================
fig = plt.figure(figsize=(18, 8.5))

gs = fig.add_gridspec(
    1,
    2,
    width_ratios=[1.35, 1.0],
    wspace=0.42
)

ax1 = fig.add_subplot(gs[0, 0])
ax2 = fig.add_subplot(gs[0, 1])

# =========================
# 7. Panel A: Forest plot
# =========================
y_base = np.arange(len(network_order))

offset_map = {
    "ABIDE I": -0.16,
    "ABIDE II": 0.16
}

color_map = {
    "ABIDE I": "#3B6FB6",
    "ABIDE II": "#D56B3D"
}

for dataset in ["ABIDE I", "ABIDE II"]:
    sub = df[df["dataset"] == dataset].copy()

    for _, row in sub.iterrows():
        network = row["item"]
        y = network_order.index(network) + offset_map[dataset]

        beta = row["beta"]
        ci_low = row["ci_low"]
        ci_high = row["ci_high"]

        xerr_low = beta - ci_low
        xerr_high = ci_high - beta

        facecolor = color_map[dataset] if row["sig_fdr"] else "white"
        edgecolor = color_map[dataset]

        label = dataset if network == network_order[0] else None

        ax1.errorbar(
            beta,
            y,
            xerr=[[xerr_low], [xerr_high]],
            fmt="o",
            markersize=9,
            markerfacecolor=facecolor,
            markeredgecolor=edgecolor,
            markeredgewidth=2,
            ecolor=color_map[dataset],
            elinewidth=2,
            capsize=4,
            label=label,
            zorder=3
        )

ax1.axvline(
    0,
    color="black",
    linewidth=1.2,
    linestyle="--",
    zorder=1
)

ax1.set_yticks(y_base)
ax1.set_yticklabels(
    [network_label_map[n] for n in network_order],
    fontname=FONT_NAME,
    fontsize=FONT_SIZE
)
ax1.invert_yaxis()

ax1.set_xlabel(
    "ASD–HC centroid difference (β)",
    fontname=FONT_NAME,
    fontsize=FONT_SIZE
)

ax1.text(
    0.0, 1.02,
    "A  Network centroid group effects",
    transform=ax1.transAxes,
    ha="left",
    va="bottom",
    fontname=FONT_NAME,
    fontsize=FONT_SIZE
)

ax1.legend(
    frameon=False,
    loc="lower right",
    prop={"family": FONT_NAME, "size": FONT_SIZE}
)

ax1.spines["top"].set_visible(False)
ax1.spines["right"].set_visible(False)

for tick in ax1.get_xticklabels() + ax1.get_yticklabels():
    tick.set_fontname(FONT_NAME)
    tick.set_fontsize(FONT_SIZE)

# =========================
# 8. Panel B: Replication scatter
# =========================
wide = df.pivot_table(
    index="item",
    columns="dataset",
    values=["beta", "p_FDR"],
    aggfunc="first"
)

wide.columns = ["_".join(col).strip() for col in wide.columns.values]
wide = wide.reset_index()

x = wide["beta_ABIDE I"]
y = wide["beta_ABIDE II"]

all_beta = pd.concat([x, y], ignore_index=True)
lim = np.nanmax(np.abs(all_beta)) * 1.25
if not np.isfinite(lim) or lim == 0:
    lim = 0.1

ax2.axhline(
    0,
    color="black",
    linewidth=1.1,
    linestyle="--",
    zorder=1
)

ax2.axvline(
    0,
    color="black",
    linewidth=1.1,
    linestyle="--",
    zorder=1
)

ax2.plot(
    [-lim, lim],
    [-lim, lim],
    color="gray",
    linewidth=1.2,
    linestyle=":",
    zorder=1
)

for _, row in wide.iterrows():
    net = row["item"]
    bx = row["beta_ABIDE I"]
    by = row["beta_ABIDE II"]

    sig1 = row.get("p_FDR_ABIDE I", np.nan) < 0.05
    sig2 = row.get("p_FDR_ABIDE II", np.nan) < 0.05

    if sig1 and sig2:
        facecolor = "black"
    elif sig1:
        facecolor = "#3B6FB6"
    elif sig2:
        facecolor = "#D56B3D"
    else:
        facecolor = "white"

    ax2.scatter(
        bx,
        by,
        s=95,
        facecolor=facecolor,
        edgecolor="black",
        linewidth=1.5,
        zorder=3
    )

    ax2.text(
        bx,
        by,
        " " + network_label_map[str(net)],
        fontsize=FONT_SIZE,
        fontname=FONT_NAME,
        va="center",
        ha="left"
    )

ax2.set_xlim(-lim, lim)
ax2.set_ylim(-lim, lim)
ax2.set_aspect("auto")

ax2.set_xlabel(
    "ABIDE I β",
    fontname=FONT_NAME,
    fontsize=FONT_SIZE
)

ax2.set_ylabel(
    "ABIDE II β",
    fontname=FONT_NAME,
    fontsize=FONT_SIZE
)

ax2.text(
    0.0, 1.02,
    "B  Effect-size replication",
    transform=ax2.transAxes,
    ha="left",
    va="bottom",
    fontname=FONT_NAME,
    fontsize=FONT_SIZE
)

ax2.spines["top"].set_visible(False)
ax2.spines["right"].set_visible(False)

for tick in ax2.get_xticklabels() + ax2.get_yticklabels():
    tick.set_fontname(FONT_NAME)
    tick.set_fontsize(FONT_SIZE)

# =========================
# 9. Layout adjust
# =========================
plt.subplots_adjust(
    left=0.09,
    right=0.98,
    top=0.92,
    bottom=0.12,
    wspace=0.40
)

# =========================
# 10. Save figure
# =========================
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