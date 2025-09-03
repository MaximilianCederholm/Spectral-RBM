import os
import math
import numpy as np
import pandas as pd
import matplotlib.pyplot as plt
import multiprocessing as mp
from tqdm import tqdm

# =======================
# Config
# =======================
hidden_units = 42
initial_temp = 5.0
final_temp = 0.01
anneal_steps = 10000
repeats_per_dp = 5
chunk_size = 50              # number of datapoints per worker chunk
num_workers = None           # None -> use cpu_count(); set int to cap
force_cpu = False            # set True to avoid CUDA in workers

results_dir = f"forward_full_eval/{hidden_units}_hidden"
os.makedirs(results_dir, exist_ok=True)

# =======================
# Worker
# =======================
def worker_eval_chunk(index_chunk):
    """
    Loads data + model inside the worker (spawn safe),
    runs annealed forward inference repeats on each datapoint in chunk,
    returns list of dict rows.
    """
    try:
        import torch

        # Device per worker
        if force_cpu:
            device = torch.device("cpu")
        else:
            device = torch.device("cuda:0" if torch.cuda.is_available() else "cpu")

        # Load dataset (CPU)
        data = pd.read_csv("t'-t''-J_data.csv")
        x_np = data.iloc[:, 0:3].values
        y_np = data.iloc[:, 3:].values
        full_np = np.hstack([x_np, y_np])
        full_data = torch.tensor(full_np, dtype=torch.float32)  # CPU tensor

        # Model def
        class JointGRBM(torch.nn.Module):
            def __init__(self, n_vis, n_hid, x_idx, y_idx):
                super().__init__()
                self.W = torch.nn.Parameter(torch.randn(n_hid, n_vis) * 0.01)
                self.v_bias = torch.nn.Parameter(torch.zeros(n_vis))
                self.h_bias = torch.nn.Parameter(torch.zeros(n_hid))
                self.sigma = torch.nn.Parameter(torch.ones(n_vis))
                self.x_idx = x_idx
                self.y_idx = y_idx

            def sample_h(self, v, temp):
                prob = torch.sigmoid(((v / (self.sigma**2)) @ self.W.t() + self.h_bias) / temp)
                return prob, torch.bernoulli(prob)

            def sample_v(self, h, temp, clamp_x=False, original_v=None):
                mean = self.v_bias + (h @ self.W) * (self.sigma**2)
                sample = mean + torch.randn_like(mean) * self.sigma * temp
                if clamp_x and original_v is not None:
                    sample[:, self.x_idx] = original_v[:, self.x_idx]
                return mean, sample

            def forward(self, v, steps=1, clamp_x=False, temp=1.0):
                v_sample = v.clone()
                for _ in range(steps):
                    h_prob, h_sample = self.sample_h(v_sample, temp)
                    _, v_sample = self.sample_v(h_sample, temp, clamp_x=clamp_x, original_v=v)
                return v_sample

        # Load checkpoint on CPU, then move to device
        model_file = f"joint-grbm/results/joint_grbm_{hidden_units}_hidden.pth"
        checkpoint = torch.load(model_file, map_location="cpu")
        visible_units = len(checkpoint["v_bias"])

        rbm = JointGRBM(visible_units, hidden_units, checkpoint["x_idx"], checkpoint["y_idx"]).to(device)
        rbm.W.data = checkpoint["W"].to(device)
        rbm.v_bias.data = checkpoint["v_bias"].to(device)
        rbm.h_bias.data = checkpoint["h_bias"].to(device)
        rbm.sigma.data = checkpoint["sigma"].to(device)

        x_idx, y_idx = rbm.x_idx, rbm.y_idx

        # Annealing schedule
        temp_decay = (final_temp / initial_temp) ** (1 / anneal_steps)

        rows = []
        for idx in index_chunk:
            # Pull sample to device
            sample = full_data[idx].unsqueeze(0).to(device)
            true_y = sample[:, y_idx]
            x_vals = sample[0, x_idx].detach().cpu().numpy()  # t', t'', J

            mses = []
            for _ in range(repeats_per_dp):
                v_input = sample.clone()
                v_input[:, y_idx] = torch.randn_like(v_input[:, y_idx]) * 5

                for step in range(1, anneal_steps + 1):
                    temperature = initial_temp * (temp_decay ** step)
                    v_input = rbm.forward(v_input, steps=1, clamp_x=True, temp=temperature)

                mse = ((v_input[:, y_idx] - true_y) ** 2).mean().item()
                mses.append(mse)

            rows.append({
                "idx": int(idx),
                "t_prime": float(x_vals[0]),
                "t_dblprime": float(x_vals[1]),
                "J": float(x_vals[2]),
                "mse_mean": float(np.mean(mses)),
                "mse_std": float(np.std(mses)),
            })

        return rows

    except Exception as e:
        # return structured info so parent can surface it
        return [{"error": str(e)}]

# =======================
# Parent (chunk + pool)
# =======================
def main():
    # Build 0..N-1 index list from CSV length
    data = pd.read_csv("t'-t''-J_data.csv")
    N = len(data)
    indices = list(range(N))

    # chunk
    chunks = [indices[i:i+chunk_size] for i in range(0, N, chunk_size)]

    ctx = mp.get_context("spawn")
    workers = num_workers or os.cpu_count() or 2
    results_accum = []

    with ctx.Pool(processes=workers) as pool:
        for rows in tqdm(pool.imap(worker_eval_chunk, chunks), total=len(chunks), desc="Evaluating all datapoints"):
            if len(rows) == 1 and "error" in rows[0]:
                print("❌ Worker error:", rows[0]["error"])
                continue
            results_accum.extend(rows)

    # Assemble DataFrame, sort by mse_mean
    df = pd.DataFrame(results_accum)
    if "error" in df.columns:
        df = df[df["error"].isna()]  # drop any error rows just in case

    df_sorted = df.sort_values("mse_mean", ascending=True).reset_index(drop=True)

    out_csv = os.path.join(results_dir, f"forward_full_eval_sorted_{hidden_units}h.csv")
    df_sorted.to_csv(out_csv, index=False)
    print(f"✅ Saved sorted per-datapoint results to: {out_csv}")

    # Quick summary
    print("\n=== Summary ===")
    print(f"Datapoints: {len(df_sorted)}")
    print(f"Mean of means: {df_sorted['mse_mean'].mean():.6f}")
    print(f"Median of means: {df_sorted['mse_mean'].median():.6f}")
    print(f"Min / Max: {df_sorted['mse_mean'].min():.6f} / {df_sorted['mse_mean'].max():.6f}")

    # Optional: save a histogram for sanity
    plt.figure()
    plt.hist(df_sorted["mse_mean"], bins=50)
    plt.xlabel("Per-datapoint MSE (mean over repeats)")
    plt.ylabel("Count")
    plt.title(f"MSE distribution — {hidden_units} hidden units")
    plt.tight_layout()
    plt.savefig(os.path.join(results_dir, f"mse_hist_{hidden_units}h.png"))
    plt.close()

if __name__ == "__main__":
    mp.set_start_method("spawn", force=True)
    main()
