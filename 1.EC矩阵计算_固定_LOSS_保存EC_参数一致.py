# =======================
# 必须在导入计算库前设置环境变量
# =======================
import os
os.environ["OMP_NUM_THREADS"] = "8"
os.environ["MKL_NUM_THREADS"] = "8"
os.environ["OPENBLAS_NUM_THREADS"] = "8"
os.environ["NUMEXPR_NUM_THREADS"] = "8"
os.environ["PYTORCH_CUDA_ALLOC_CONF"] = "max_split_size_mb:256"

import json
import numpy as np
import torch
import torch.nn as nn
import torch.utils.data as data
from concurrent.futures import ProcessPoolExecutor, as_completed
from multiprocessing import freeze_support
import pandas as pd


# =========================================================
# Global NPI settings
# =========================================================

# ROI 提取阶段使用 Schaefer-400，与原始 demo 设置不同
N_ROIS = 400

# 与 NPI demo 对齐
USING_STEPS = 3
BATCH_SIZE = 50
TRAIN_SET_PROPORTION = 0.8
NUM_EPOCHS = 100
LR = 1e-3

# demo 中调用 train_NN 时使用 l2=5e-5
# 设为 0 时使用 NPI.py 函数的默认正则化参数
L2 = 5e-5

# 与 demo 中 model_EC(..., pert_strength=1.0) 对齐
PERT_STRENGTH = 1.0

# MLP 维度与 demo 逻辑对齐：
# hidden_dim = 2 * ROI_num
# latent_dim = int(0.8 * ROI_num)
HIDDEN_DIM_FACTOR = 2.0
LATENT_DIM_FACTOR = 0.8

# ROI 提取阶段已经 standardize="zscore_sample"
# 为避免与原始 NPI 使用方式不一致，这里默认不再二次 z-score
DO_EXTRA_ZSCORE_IN_NPI = False


# =========================================================
# ANN_MLP model
# =========================================================
class ANN_MLP(nn.Module):
    """
    Use MLP as a surrogate brain.
    Structure is consistent with original NPI ANN_MLP:
    Linear -> ReLU -> Linear -> ReLU -> Linear
    """

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
# Data processing
# =========================================================
def multi2one(time_series, steps):
    """
    Split the data into input-output pairs.

    Input:
        time_series: T x N
        steps: number of previous time points

    Output:
        input_X:  (T - steps) x (steps * N)
        target_Y: (T - steps) x N
    """

    n_area = time_series.shape[1]
    n_step = time_series.shape[0]

    input_X = np.zeros((n_step - steps, n_area * steps), dtype=np.float32)
    target_Y = np.zeros((n_step - steps, n_area), dtype=np.float32)

    for i in range(n_step - steps):
        input_X[i] = time_series[i:steps + i].flatten()
        target_Y[i] = time_series[steps + i].flatten()

    return input_X, target_Y


# =========================================================
# Training
# =========================================================
def train_NN(
    model,
    input_X,
    target_Y,
    device,
    batch_size=BATCH_SIZE,
    train_set_proportion=TRAIN_SET_PROPORTION,
    num_epochs=NUM_EPOCHS,
    lr=LR,
    l2=L2
):
    """
    Use empirical data to tune the model.
    Parameters are aligned with NPI demo:
        batch_size = 50
        train_set_proportion = 0.8
        num_epochs = 100
        lr = 1e-3
        l2 = 5e-5
    """

    split = int(train_set_proportion * input_X.shape[0])

    train_inputs = torch.tensor(
        input_X[:split],
        dtype=torch.float32,
        device=device
    )
    train_targets = torch.tensor(
        target_Y[:split],
        dtype=torch.float32,
        device=device
    )

    test_inputs = torch.tensor(
        input_X[split:],
        dtype=torch.float32,
        device=device
    )
    test_targets = torch.tensor(
        target_Y[split:],
        dtype=torch.float32,
        device=device
    )

    train_dataset = data.TensorDataset(train_inputs, train_targets)
    test_dataset = data.TensorDataset(test_inputs, test_targets)

    train_iter = data.DataLoader(
        train_dataset,
        batch_size=batch_size,
        shuffle=True,
        drop_last=False
    )

    test_iter = data.DataLoader(
        test_dataset,
        batch_size=batch_size,
        shuffle=False,
        drop_last=False
    )

    loss_fn = nn.MSELoss()
    trainer = torch.optim.Adam(
        model.parameters(),
        lr=lr,
        weight_decay=l2
    )

    train_epoch_loss = []
    test_epoch_loss = []

    for _ in range(num_epochs):
        model.train()

        for X_batch, y_batch in train_iter:
            y_hat = model(X_batch)
            loss = loss_fn(y_hat, y_batch)

            trainer.zero_grad(set_to_none=True)
            loss.backward()
            trainer.step()

        model.eval()

        with torch.no_grad():
            total_loss = 0.0
            total_num = 0

            for X_batch, y_batch in train_iter:
                y_hat = model(X_batch)
                loss = loss_fn(y_hat, y_batch)
                total_loss += loss.item() * y_batch.shape[0]
                total_num += y_batch.shape[0]

            train_epoch_loss.append(
                total_loss / total_num if total_num > 0 else np.nan
            )

            total_loss = 0.0
            total_num = 0

            for X_batch, y_batch in test_iter:
                y_hat = model(X_batch)
                loss = loss_fn(y_hat, y_batch)
                total_loss += loss.item() * y_batch.shape[0]
                total_num += y_batch.shape[0]

            test_epoch_loss.append(
                total_loss / total_num if total_num > 0 else np.nan
            )

    return model, train_epoch_loss, test_epoch_loss


# =========================================================
# EC - NPI
# =========================================================
@torch.no_grad()
def model_EC_NPI(model, input_X, target_Y, device, pert_strength=PERT_STRENGTH):
    """
    Infer EC by perturbing the surrogate brain.

    This follows the original NPI model_EC logic:
        1. Compute unperturbed output.
        2. For each node, perturb only the last time step.
        3. Compute mean output difference across samples.
    """

    model.eval()

    node_num = target_Y.shape[1]
    steps = int(input_X.shape[1] / node_num)

    input_tensor = torch.tensor(
        input_X,
        dtype=torch.float32,
        device=device
    )

    unperturbed_output = model(input_tensor).detach().cpu().numpy()

    NPI_EC = np.zeros((node_num, node_num), dtype=np.float32)

    for node in range(node_num):
        perturbation = np.zeros((steps, node_num), dtype=np.float32)
        perturbation[-1, node] = pert_strength

        perturbed_input = input_X + perturbation.flatten()

        perturbed_output = model(
            torch.tensor(
                perturbed_input,
                dtype=torch.float32,
                device=device
            )
        ).detach().cpu().numpy()

        NPI_EC[node] = np.mean(
            perturbed_output - unperturbed_output,
            axis=0
        )

    return NPI_EC


# =========================================================
# Save reproducibility package
# =========================================================
def save_subject_package(
    save_dir,
    model,
    input_X,
    target_Y,
    NPI_EC,
    train_loss,
    test_loss,
    subj,
    T,
    N,
    steps,
    pert_strength,
    fpath
):
    os.makedirs(save_dir, exist_ok=True)

    np.save(
        os.path.join(save_dir, "EC_NPI.npy"),
        NPI_EC.astype(np.float32)
    )

    np.save(
        os.path.join(save_dir, "train_loss.npy"),
        np.array(train_loss, dtype=np.float32)
    )

    np.save(
        os.path.join(save_dir, "test_loss.npy"),
        np.array(test_loss, dtype=np.float32)
    )

    torch.save(
        model.state_dict(),
        os.path.join(save_dir, "model.pth")
    )

    np.save(
        os.path.join(save_dir, "X.npy"),
        input_X.astype(np.float32)
    )

    np.save(
        os.path.join(save_dir, "Y.npy"),
        target_Y.astype(np.float32)
    )

    config = {
        "subject": subj,
        "source_file": fpath,
        "T": int(T),
        "N": int(N),
        "roi_type": "Schaefer-400, 7 Yeo networks, 2 mm",
        "steps": int(steps),
        "input_dim": int(steps * N),
        "hidden_dim": int(HIDDEN_DIM_FACTOR * N),
        "latent_dim": int(LATENT_DIM_FACTOR * N),
        "output_dim": int(N),
        "model": "ANN_MLP",
        "model_structure": "Linear-ReLU-Linear-ReLU-Linear",
        "batch_size": int(BATCH_SIZE),
        "train_set_proportion": float(TRAIN_SET_PROPORTION),
        "num_epochs": int(NUM_EPOCHS),
        "lr": float(LR),
        "l2": float(L2),
        "pert_strength": float(pert_strength),
        "dtype": "float32",
        "extra_zscore_in_NPI": bool(DO_EXTRA_ZSCORE_IN_NPI),
        "diagonal_zeroed": True,
        "note": (
            "ROI time series were assumed to be already cleaned and "
            "standardized during ROI extraction. NPI settings follow the "
            "original demo except ROI_num = 400."
        )
    }

    with open(
        os.path.join(save_dir, "config.json"),
        "w",
        encoding="utf-8"
    ) as f:
        json.dump(config, f, indent=4, ensure_ascii=False)


# =========================================================
# Single-subject / single-run processing
# =========================================================
def process_one_subject(fpath, out_root, gpu_id):
    torch.cuda.set_device(gpu_id)
    device = torch.device(f"cuda:{gpu_id}")

    fname = os.path.basename(fpath)
    subj = fname[:-4] if fname.endswith(".npy") else os.path.splitext(fname)[0]

    print(f"▶ Processing {subj} on GPU {gpu_id}")

    try:
        time_series = np.load(fpath)
    except Exception as e:
        return {
            "subject": subj,
            "source_file": fpath,
            "status": "load_error",
            "message": repr(e)
        }

    if time_series.ndim != 2:
        return {
            "subject": subj,
            "source_file": fpath,
            "status": "skipped",
            "message": f"invalid ndim: {time_series.ndim}"
        }

    if time_series.shape[1] != N_ROIS:
        return {
            "subject": subj,
            "source_file": fpath,
            "status": "skipped",
            "message": f"invalid ROI number: {time_series.shape[1]}, expected {N_ROIS}"
        }

    if time_series.shape[0] <= USING_STEPS:
        return {
            "subject": subj,
            "source_file": fpath,
            "status": "skipped",
            "message": f"too few time points: T={time_series.shape[0]}, steps={USING_STEPS}"
        }

    if not np.isfinite(time_series).all():
        return {
            "subject": subj,
            "source_file": fpath,
            "status": "skipped",
            "message": "NaN or Inf in data"
        }

    time_series = time_series.astype(np.float32)

    if DO_EXTRA_ZSCORE_IN_NPI:
        time_series = (
            time_series - time_series.mean(axis=0, keepdims=True)
        ) / (
            time_series.std(axis=0, keepdims=True) + 1e-8
        )

    T, N = time_series.shape

    input_X, target_Y = multi2one(
        time_series,
        steps=USING_STEPS
    )

    hidden_dim = int(HIDDEN_DIM_FACTOR * N)
    latent_dim = int(LATENT_DIM_FACTOR * N)

    model = ANN_MLP(
        input_dim=USING_STEPS * N,
        hidden_dim=hidden_dim,
        latent_dim=latent_dim,
        output_dim=N
    ).to(device)

    model, train_loss, test_loss = train_NN(
        model=model,
        input_X=input_X,
        target_Y=target_Y,
        device=device,
        batch_size=BATCH_SIZE,
        train_set_proportion=TRAIN_SET_PROPORTION,
        num_epochs=NUM_EPOCHS,
        lr=LR,
        l2=L2
    )

    NPI_EC = model_EC_NPI(
        model=model,
        input_X=input_X,
        target_Y=target_Y,
        device=device,
        pert_strength=PERT_STRENGTH
    )

    # 与原始 demo 一致：去除自连接
    np.fill_diagonal(NPI_EC, 0)

    save_dir = os.path.join(out_root, subj)

    save_subject_package(
        save_dir=save_dir,
        model=model,
        input_X=input_X,
        target_Y=target_Y,
        NPI_EC=NPI_EC,
        train_loss=train_loss,
        test_loss=test_loss,
        subj=subj,
        T=T,
        N=N,
        steps=USING_STEPS,
        pert_strength=PERT_STRENGTH,
        fpath=fpath
    )

    del model
    torch.cuda.empty_cache()

    return {
        "subject": subj,
        "source_file": fpath,
        "status": "done",
        "T": int(T),
        "N": int(N),
        "steps": int(USING_STEPS),
        "hidden_dim": int(hidden_dim),
        "latent_dim": int(latent_dim),
        "epochs": int(NUM_EPOCHS),
        "batch_size": int(BATCH_SIZE),
        "lr": float(LR),
        "l2": float(L2),
        "pert_strength": float(PERT_STRENGTH),
        "save_dir": save_dir
    }


# =========================================================
# Main: multi-GPU parallel
# =========================================================
if __name__ == "__main__":
    freeze_support()

    root = r"I:\DYF\NPI-4-code\1.NPI\ABIDE1_ROI"
    out_root = r"I:\DYF\NPI-4-code\1.NPI\ABIDE1_NPI"

    os.makedirs(out_root, exist_ok=True)

    npy_files = [
        os.path.join(root, f)
        for f in os.listdir(root)
        if f.endswith(".npy")
    ]

    npy_files = sorted(npy_files)

    n_gpus = torch.cuda.device_count()

    if n_gpus < 1:
        raise RuntimeError("No CUDA GPUs found.")

    gpu_ids = list(range(n_gpus))
    max_workers = len(gpu_ids)

    print(f"Found {n_gpus} GPU(s): {gpu_ids}")
    print(f"Input ROI directory: {root}")
    print(f"Saving results to: {out_root}")
    print(f"Total NPY files: {len(npy_files)}")

    print("NPI settings:")
    print(f"  ROI number: {N_ROIS}")
    print(f"  steps: {USING_STEPS}")
    print(f"  hidden_dim: 2 * N = {int(HIDDEN_DIM_FACTOR * N_ROIS)}")
    print(f"  latent_dim: int(0.8 * N) = {int(LATENT_DIM_FACTOR * N_ROIS)}")
    print(f"  batch_size: {BATCH_SIZE}")
    print(f"  train_set_proportion: {TRAIN_SET_PROPORTION}")
    print(f"  num_epochs: {NUM_EPOCHS}")
    print(f"  lr: {LR}")
    print(f"  l2: {L2}")
    print(f"  pert_strength: {PERT_STRENGTH}")
    print(f"  extra z-score in NPI: {DO_EXTRA_ZSCORE_IN_NPI}")

    if len(npy_files) == 0:
        raise RuntimeError(f"No .npy files found in: {root}")

    results = []

    with ProcessPoolExecutor(max_workers=max_workers) as executor:
        futures = []

        for idx, fpath in enumerate(npy_files):
            gpu_id = gpu_ids[idx % len(gpu_ids)]

            futures.append(
                executor.submit(
                    process_one_subject,
                    fpath,
                    out_root,
                    gpu_id
                )
            )

        for fut in as_completed(futures):
            try:
                result = fut.result()
                results.append(result)

                if result["status"] == "done":
                    print(
                        "  DONE:",
                        result["subject"],
                        f"GPU result saved to {result['save_dir']}"
                    )
                else:
                    print(
                        "  SKIPPED/ERROR:",
                        result.get("subject", "unknown"),
                        result.get("message", "")
                    )

            except Exception as e:
                error_result = {
                    "subject": "unknown",
                    "source_file": "unknown",
                    "status": "worker_error",
                    "message": repr(e)
                }
                results.append(error_result)
                print("  ❌ error in worker:", repr(e))

    report_file = os.path.join(
        out_root,
        "NPI_processing_report.csv"
    )

    pd.DataFrame(results).to_csv(
        report_file,
        index=False,
        encoding="utf-8-sig"
    )

    print("全部处理完成")
    print(f"处理日志已保存: {report_file}")
