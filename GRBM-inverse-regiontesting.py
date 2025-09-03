#!/usr/bin/env python3
import os
import numpy as np
import pandas as pd
import matplotlib.pyplot as plt
import multiprocessing as mp
from tqdm import tqdm

# =======================
# Config
# =======================
hidden_units = 46           # Single-model region test
initial_temp = 5.0
final_temp = 0.01
anneal_steps = 10000
repeats_per_dp = 5
chunk_size = 50             # number of datapoints per worker chunk
num_workers = None          # None -> use cpu_count(); set int to cap
force_cpu = False           # set True to avoid CUDA in workers
randomize_x_scale = 5.0     # stddev for initial X randomization

results_dir = f"inverse_full_eval/{hidden_units}_hidden"
os.makedirs(results_dir, exist_ok=True)

data_csv = "t'-t''-J_data.csv"
model_file = f"joint-grbm/results/joint_grbm_{hidden_units}_hidden.pth"

# =======================
# Worker
# =======================
def worker_eval_chunk(index_chunk):
    """
    Loads data + model inside the worker (spawn safe),
    runs annealed inverse inference repeats on each datapoint in chunk,
    returns list of dict rows with per-component MAE means/stds.
    """
    try:
        import torch

        # Device per worker
        if force_cpu:
            device = torch.device("cpu")
        else:
            device = torch.device("cuda:0" if torch.cuda.is_available() else "cpu")

        # Load dataset (CPU)
        data = pd.read_csv(data_csv)
        x_np = data.iloc[:, 0:3].values   # (t', t'', J)
        y_np = data.iloc[:, 3:].values    # spectral / measurement vector
        full_np = np.hstack([x_np, y_np])
        full_data = torch.tensor(full_np, dtype=torch.float32)  # CPU tensor

        # Model def (mirrors your forward script; adds clamp_y path)
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

            def sample_v(self, h, temp, clamp_x=False, clamp_y=False, original_v=None):
                mean = self.v_bias + (h @ self.W) * (self.sigma**2)
                sample = mean + torch.randn_like(mean) * self.sigma * temp
                if original_v is not None:
                    if clamp_x:
                        sample[:, self.x_idx] = original_v[:, self.x_idx]
                    if clamp_y:
                        sample[:, self.y_idx] = original_v[:, self.y_idx]
                return mean, sample

            def forward(self, v, steps=1, clamp_x=False, clamp_y=False, temp=1.0, original_v=None):
                v_sample = v.clone()
                for _ in range(steps):
                    h_prob, h_sample = self.sample_h(v_sample, temp)
                    _, v_sample = self.sample_v(
                        h_sample, temp,
                        clamp_x=clamp_x, clamp_y=clamp_y,
                        original_v=original_v if original_v is not None else v
                    )
                return v_sample

        # Load checkpoint on CPU, then move to device
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
            true_x = sample[:, x_idx]
            true_y = sample[:, y_idx]

            # Collect per-repeat absolute errors for each X component
            errs_tprime = []
            errs_tdblprime = []
            errs_J = []

            for _ in range(repeats_per_dp):
                v_input = sample.clone()

                # Randomize X group (free variables) with a fairly wide Gaussian
                v_input[:, x_idx] = torch.randn_like(v_input[:, x_idx]) * randomize_x_scale

                # Clamp Y group to the true observations during annealing
                for step in range(1, anneal_steps + 1):
                    temperature = initial_temp * (temp_decay ** step)
                    v_input = rbm.forward(
                        v_input, steps=1,
                        clamp_y=True, clamp_x=False,
                        temp=temperature, original_v=sample
                    )

                # Absolute error in each X component
                abs_err = (v_input[:, x_idx] - true_x).abs().detach().cpu().numpy().flatten()
                errs_tprime.append(float(abs_err[0]))
                errs_tdblprime.append(float(abs_err[1]))
                errs_J.append(float(abs_err[2]))

            # Aggregate across repeats
            def mean_std(v):
                arr = np.asarray(v, dtype=np.float64)
                return float(arr.mean()), float(arr.std(ddof=1) if arr.size > 1 else 0.0)

            m_tprime, s_tprime = mean_std(errs_tprime)
            m_tdblprime, s_tdblprime = mean_std(errs_tdblprime)
            m_J, s_J = mean_std(errs_J)

            rows.append({
                "idx": int(idx),
                "t_prime_true": float(true_x[0,0].detach().cpu().item()),
                "t_dprime_true": float(true_x[0,1].detach().cpu().item()),
                "J_true": float(true_x[0,2].detach().cpu().item()),
                "MAE_t_prime": m_tprime,
                "MAE_t_dprime": m_tdblprime,
                "MAE_J": m_J,
                "STD_t_prime": s_tprime,
                "STD_t_dprime": s_tdblprime,
                "STD_J": s_J,
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
    data = pd.read_csv(data_csv)
    N = len(data)
    indices = list(range(N))

    # chunk
    chunks = [indices[i:i+chunk_size] for i in range(0, N, chunk_size)]

    ctx = mp.get_context("spawn")
    workers = num_workers or os.cpu_count() or 2
    results_accum = []

    with ctx.Pool(processes=workers) as pool:
        for rows in tqdm(pool.imap(worker_eval_chunk, chunks), total=len(chunks), desc="Evaluating all datapoints (inverse)"):
            if len(rows) == 1 and "error" in rows[0]:
                print("❌ Worker error:", rows[0]["error"])
                continue
            results_accum.extend(rows)

    # Assemble DataFrame
    df = pd.DataFrame(results_accum)
    if "error" in df.columns:
        df = df[df["error"].isna()]  # drop any error rows just in case

    # Save unsorted
    out_master = os.path.join(results_dir, f"inverse_full_eval_{hidden_units}h.csv")
    df.to_csv(out_master, index=False)
    print(f"✅ Saved per-datapoint inverse results to: {out_master}")

    # Quick per-component summaries
    def summarize(col):
        s = {
            "count": int(df[col].count()),
            "mean": float(df[col].mean()),
            "median": float(df[col].median()),
            "std": float(df[col].std(ddof=1)) if df[col].count() > 1 else 0.0,
            "min": float(df[col].min()),
            "p5": float(df[col].quantile(0.05)),
            "p25": float(df[col].quantile(0.25)),
            "p75": float(df[col].quantile(0.75)),
            "p95": float(df[col].quantile(0.95)),
            "max": float(df[col].max()),
        }
        return s

    for comp in ["MAE_t_prime", "MAE_t_dprime", "MAE_J"]:
        s = summarize(comp)
        print(f"\n=== Summary: {comp} ===")
        for k, v in s.items():
            print(f"{k:>6}: {v:.6f}" if isinstance(v, float) else f"{k:>6}: {v}")

    # Also export three separately sorted CSVs (scale-agnostic — sorted per component)
    for comp in ["MAE_t_prime", "MAE_t_dprime", "MAE_J"]:
        df_sorted = df.sort_values(comp, ascending=True).reset_index(drop=True)
        out_csv = os.path.join(results_dir, f"inverse_sorted_by_{comp}_{hidden_units}h.csv")
        df_sorted.to_csv(out_csv, index=False)
        print(f"📄 Saved sorted-by-{comp}: {out_csv}")

        # Histogram per component
        plt.figure()
        plt.hist(df[comp].values, bins=50)
        plt.xlabel(f"{comp} (mean over {repeats_per_dp} repeats)")
        plt.ylabel("Count")
        plt.title(f"{comp} distribution — {hidden_units} hidden units")
        plt.tight_layout()
        plt.savefig(os.path.join(results_dir, f"hist_{comp}_{hidden_units}h.png"))
        plt.close()

if __name__ == "__main__":
    mp.set_start_method("spawn", force=True)
    main()
