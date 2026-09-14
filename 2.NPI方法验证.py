# -*- coding: utf-8 -*-

from pathlib import Path
import json
import shutil
import numpy as np
import pandas as pd
import torch
import torch.nn as nn


# =========================================================
# 1. 路径设置
# =========================================================

# 之前 NPI 主计算保存的结果目录：
# 该目录下应包含每个被试的子文件夹，每个子文件夹内有：
# model.pth, X.npy, Y.npy, config.json
INPUT_RESULT_ROOT = Path(
    r"I:\DYF\NPI-4-code\1.NPI\ABIDE1_NPI"
)

# 新的保存地址
OUTPUT_DIR = Path(
    r"I:\DYF\NPI-4-code\1.NPI\ABIDE1_NPI验证"
)

OUTPUT_DIR.mkdir(parents=True, exist_ok=True)

OUT_CSV = OUTPUT_DIR / "NPI_FC_reproduction_modelFC_empiricalFC.csv"
OUT_SUMMARY_CSV = OUTPUT_DIR / "NPI_FC_reproduction_summary.csv"

# 是否保存每个被试的 empirical_FC、model_FC 和 model-generated signal
SAVE_SUBJECT_FC = True

# 是否把 config.json 复制到新的输出被试文件夹
COPY_CONFIG_TO_OUTPUT = True


# =========================================================
# 2. ANN_MLP，必须与原始训练代码一致
# =========================================================
class ANN_MLP(nn.Module):
    def __init__(self, input_dim, hidden_dim, latent_dim, output_dim):
        super().__init__()
        self.net = nn.Sequential(
            nn.Linear(input_dim, hidden_dim),
            nn.ReLU(),
            nn.Linear(hidden_dim, latent_dim),
            nn.ReLU(),
            nn.Linear(latent_dim, output_dim),
        )

    def forward(self, x):
        return self.net(x)


# =========================================================
# 3. 从 X.npy 和 Y.npy 还原原始 time series
# =========================================================
def reconstruct_time_series_from_XY(X, Y, steps, N):
    """
    根据原始 multi2one() 逻辑还原 time series：

        X[0] = time_series[0:steps].flatten()
        Y[:] = time_series[steps:T]

    因此：
        time_series = vstack([X[0].reshape(steps, N), Y])
    """

    first_steps = X[0].reshape(steps, N)
    time_series = np.vstack([first_steps, Y])

    return time_series.astype(np.float32)


# =========================================================
# 4. 递归生成 ANN model BOLD signal
# =========================================================
@torch.no_grad()
def generate_model_signal_recurrent(model, time_series, device, steps):
    """
    使用训练好的 ANN surrogate model 递归生成 model BOLD signal。

    前 steps 个时间点使用 empirical BOLD 初始化；
    后续时间点使用模型输出递归生成。
    """

    model.eval()

    T, N = time_series.shape

    generated = np.zeros((T, N), dtype=np.float32)
    generated[:steps] = time_series[:steps]

    for t in range(steps, T):
        x = generated[t - steps:t].reshape(1, steps * N)

        x_tensor = torch.tensor(
            x,
            dtype=torch.float32,
            device=device
        )

        y_hat = model(x_tensor).detach().cpu().numpy()[0]
        generated[t] = y_hat.astype(np.float32)

    return generated


# =========================================================
# 5. FC 计算与上三角相关
# =========================================================
def compute_fc_matrix(time_series):
    """
    计算 ROI x ROI FC 矩阵。
    """

    fc = np.corrcoef(time_series.T)
    return fc.astype(np.float32)


def fc_upper_triangle_corr(fc_empirical, fc_model):
    """
    计算 empirical FC 与 model FC 上三角的 Pearson 相关。
    排除对角线。
    """

    iu = np.triu_indices_from(fc_empirical, k=1)

    v1 = fc_empirical[iu]
    v2 = fc_model[iu]

    mask = np.isfinite(v1) & np.isfinite(v2)

    if mask.sum() < 10:
        return np.nan

    return float(np.corrcoef(v1[mask], v2[mask])[0, 1])


# =========================================================
# 6. 读取 torch state_dict
# =========================================================
def load_state_dict_safely(model_file, device):
    """
    兼容不同 PyTorch 版本。
    """

    try:
        state_dict = torch.load(
            model_file,
            map_location=device,
            weights_only=True
        )
    except TypeError:
        state_dict = torch.load(
            model_file,
            map_location=device
        )

    return state_dict


# =========================================================
# 7. 单被试处理
# =========================================================
def process_one_subject_dir(subj_dir, output_dir, device):
    config_file = subj_dir / "config.json"
    model_file = subj_dir / "model.pth"
    x_file = subj_dir / "X.npy"
    y_file = subj_dir / "Y.npy"

    if not config_file.exists():
        return {
            "subject": subj_dir.name,
            "status": "missing_config"
        }

    if not model_file.exists():
        return {
            "subject": subj_dir.name,
            "status": "missing_model"
        }

    if not x_file.exists() or not y_file.exists():
        return {
            "subject": subj_dir.name,
            "status": "missing_X_or_Y"
        }

    with open(config_file, "r", encoding="utf-8") as f:
        config = json.load(f)

    subject = config.get("subject", subj_dir.name)
    source_file = config.get("source_file", "")

    steps = int(config["steps"])
    N = int(config["N"])

    input_dim = int(config["input_dim"])
    hidden_dim = int(config["hidden_dim"])
    latent_dim = int(config["latent_dim"])
    output_dim = int(config["output_dim"])

    X = np.load(x_file).astype(np.float32)
    Y = np.load(y_file).astype(np.float32)

    time_series = reconstruct_time_series_from_XY(
        X=X,
        Y=Y,
        steps=steps,
        N=N
    )

    model = ANN_MLP(
        input_dim=input_dim,
        hidden_dim=hidden_dim,
        latent_dim=latent_dim,
        output_dim=output_dim
    ).to(device)

    state_dict = load_state_dict_safely(
        model_file=model_file,
        device=device
    )

    model.load_state_dict(state_dict)

    model_signal = generate_model_signal_recurrent(
        model=model,
        time_series=time_series,
        device=device,
        steps=steps
    )

    empirical_fc = compute_fc_matrix(time_series)
    model_fc = compute_fc_matrix(model_signal)

    fc_r = fc_upper_triangle_corr(
        fc_empirical=empirical_fc,
        fc_model=model_fc
    )

    if np.isfinite(fc_r):
        fc_z = float(np.arctanh(np.clip(fc_r, -0.999999, 0.999999)))
    else:
        fc_z = np.nan

    subject_out_dir = output_dir / subject
    subject_out_dir.mkdir(parents=True, exist_ok=True)

    if SAVE_SUBJECT_FC:
        np.save(
            subject_out_dir / "empirical_FC.npy",
            empirical_fc.astype(np.float32)
        )

        np.save(
            subject_out_dir / "model_FC.npy",
            model_fc.astype(np.float32)
        )

        np.save(
            subject_out_dir / "model_generated_signal.npy",
            model_signal.astype(np.float32)
        )

    if COPY_CONFIG_TO_OUTPUT:
        shutil.copy2(
            config_file,
            subject_out_dir / "config.json"
        )

    del model

    if torch.cuda.is_available():
        torch.cuda.empty_cache()

    return {
        "subject": subject,
        "source_file": source_file,
        "status": "done",
        "input_subject_dir": str(subj_dir),
        "output_subject_dir": str(subject_out_dir),
        "T": int(time_series.shape[0]),
        "N": int(time_series.shape[1]),
        "steps": int(steps),
        "model_FC_empirical_FC_r": fc_r,
        "model_FC_empirical_FC_z": fc_z
    }


# =========================================================
# 8. 汇总统计
# =========================================================
def make_summary(df):
    done = df[df["status"] == "done"].copy()

    if len(done) == 0:
        return pd.DataFrame([{
            "N_done": 0,
            "r_mean": np.nan,
            "r_sd": np.nan,
            "r_median": np.nan,
            "r_min": np.nan,
            "r_max": np.nan,
            "z_mean": np.nan,
            "z_sd": np.nan,
            "z_median": np.nan,
            "z_min": np.nan,
            "z_max": np.nan
        }])

    r = done["model_FC_empirical_FC_r"].astype(float)
    z = done["model_FC_empirical_FC_z"].astype(float)

    summary = pd.DataFrame([{
        "N_done": int(len(done)),
        "r_mean": float(r.mean()),
        "r_sd": float(r.std(ddof=1)),
        "r_median": float(r.median()),
        "r_min": float(r.min()),
        "r_max": float(r.max()),
        "z_mean": float(z.mean()),
        "z_sd": float(z.std(ddof=1)),
        "z_median": float(z.median()),
        "z_min": float(z.min()),
        "z_max": float(z.max())
    }])

    return summary


# =========================================================
# 9. 主程序
# =========================================================
if __name__ == "__main__":

    device = torch.device("cuda:0" if torch.cuda.is_available() else "cpu")

    subj_dirs = [
        p for p in INPUT_RESULT_ROOT.iterdir()
        if p.is_dir()
    ]

    subj_dirs = sorted(subj_dirs)

    print(f"Input result root: {INPUT_RESULT_ROOT}")
    print(f"Output directory: {OUTPUT_DIR}")
    print(f"Subject folders: {len(subj_dirs)}")
    print(f"Device: {device}")

    if len(subj_dirs) == 0:
        raise RuntimeError(f"No subject folders found in: {INPUT_RESULT_ROOT}")

    results = []

    for i, subj_dir in enumerate(subj_dirs, 1):
        print(f"[{i}/{len(subj_dirs)}] Processing {subj_dir.name}")

        try:
            result = process_one_subject_dir(
                subj_dir=subj_dir,
                output_dir=OUTPUT_DIR,
                device=device
            )
        except Exception as e:
            result = {
                "subject": subj_dir.name,
                "status": "error",
                "message": repr(e),
                "input_subject_dir": str(subj_dir)
            }

        results.append(result)

        if result.get("status") == "done":
            print(
                "  done:",
                result["subject"],
                "r =",
                result["model_FC_empirical_FC_r"]
            )
        else:
            print("  skipped/error:", result)

    df = pd.DataFrame(results)

    df.to_csv(
        OUT_CSV,
        index=False,
        encoding="utf-8-sig"
    )

    summary_df = make_summary(df)

    summary_df.to_csv(
        OUT_SUMMARY_CSV,
        index=False,
        encoding="utf-8-sig"
    )

    print("Finished.")
    print(f"Subject-level results saved to: {OUT_CSV}")
    print(f"Summary saved to: {OUT_SUMMARY_CSV}")

    print("Summary:")
    print(summary_df.to_string(index=False))