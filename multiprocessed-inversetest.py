# multiprocessed-inverse-test.py
import os
import numpy as np
import pandas as pd
import matplotlib.pyplot as plt  # (not strictly needed, but handy if you add plots)
import multiprocessing as mp
from tqdm import tqdm

# =======================
# 1. Config
# =======================
hidden_unit_options = [34, 36, 38, 40, 42, 44, 46, 48, 50, 52, 54, 56, 58, 60, 62]  # add/remove as you like
initial_temp = 5.0
final_temp = 0.01
anneal_steps = 10_000
num_rows = 100          # evaluate 100 rows per RBM
repeats_per_row = 5     # each row is run 5 times and MAE is averaged
rand_seed = 1337        # ensures the SAME 100 rows across all RBMs
results_dir = "multiprocessed_inverse_results"
os.makedirs(results_dir, exist_ok=True)

def build_indices_for_all_models():
    """Sample the SAME 'num_rows' indices once, used by every worker."""
    data = pd.read_csv("t'-t''-J_data.csv")
    n = len(data)
    rng = np.random.default_rng(rand_seed)
    idx = rng.choice(n, size=num_rows, replace=False)
    return idx.tolist()

def evaluate_model(args):
    """
    Worker function (spawned). Runs inverse inference:
    clamp y, randomize x, anneal temperature, measure per-parameter |error|.
    Returns per-model summary and writes per-model CSV/TXT.
    """
    hidden_units, shared_indices = args
    try:
        import torch  # import inside worker for CUDA-safety

        # ---------- Device ----------
        device = torch.device("cuda:0" if torch.cuda.is_available() else "cpu")

        # ---------- Load dataset (CPU) ----------
        data = pd.read_csv("t'-t''-J_data.csv")
        x_data = data.iloc[:, 0:3].values
        y_data = data.iloc[:, 3:].values
        full_np = np.hstack([x_data, y_data])
        full_data = torch.tensor(full_np, dtype=torch.float32)  # keep on CPU; move per-sample

        # ---------- Model ----------
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

            def sample_v(self, h, temp, clamp_y=False, original_v=None):
                mean = self.v_bias + (h @ self.W) * (self.sigma**2)
                sample = mean + torch.randn_like(mean) * self.sigma * temp
                if clamp_y and original_v is not None:
                    sample[:, self.y_idx] = original_v[:, self.y_idx]
                return mean, sample

            def forward(self, v, steps=1, clamp_y=False, temp=1.0):
                v_sample = v.clone()
                for _ in range(steps):
                    h_prob, h_sample = self.sample_h(v_sample, temp)
                    _, v_sample = self.sample_v(h_sample, temp, clamp_y=clamp_y, original_v=v)
                return v_sample

        # ---------- Load checkpoint ----------
        model_file = f"joint-grbm/results/joint_grbm_{hidden_units}_hidden.pth"
        checkpoint = torch.load(model_file, map_location="cpu")
        visible_units = len(checkpoint["v_bias"])

        rbm = JointGRBM(visible_units, hidden_units, checkpoint["x_idx"], checkpoint["y_idx"]).to(device)
        rbm.W.data = checkpoint["W"].to(device)
        rbm.v_bias.data = checkpoint["v_bias"].to(device)
        rbm.h_bias.data = checkpoint["h_bias"].to(device)
        rbm.sigma.data = checkpoint["sigma"].to(device)

        print(f"✅ Loaded inverse RBM with {hidden_units} hidden units")

        x_idx, y_idx = rbm.x_idx, rbm.y_idx

        # ---------- Annealing schedule ----------
        temp_decay = (final_temp / initial_temp) ** (1 / anneal_steps)

        # ---------- Per-model results (one row per evaluated data index) ----------
        rows = []
        pbar = tqdm(shared_indices, desc=f"Inverse eval {hidden_units}h")
        for idx in pbar:
            sample = full_data[idx].unsqueeze(0).to(device)
            true_x = sample[:, x_idx]   # (1,3)
            true_x_cpu = true_x.detach().cpu().numpy().reshape(-1)  # for saving

            # Average MAE across repeats
            mae_accum = np.zeros(3, dtype=np.float64)

            for _ in range(repeats_per_row):
                v_input = sample.clone()
                # Randomize x block (while we'll clamp y during inference)
                v_input[:, x_idx] = torch.randn_like(v_input[:, x_idx]) * 5

                # Anneal
                for step in range(1, anneal_steps + 1):
                    temperature = initial_temp * (temp_decay ** step)
                    v_input = rbm.forward(v_input, steps=1, clamp_y=True, temp=temperature)

                # Per-parameter absolute error
                abs_err = (v_input[:, x_idx] - true_x).abs().detach().cpu().numpy().reshape(-1)  # (3,)
                mae_accum += abs_err

            mae_avg = mae_accum / repeats_per_row  # (3,)
            rows.append({
                "idx": int(idx),
                "t_prime_true": true_x_cpu[0],
                "t_dprime_true": true_x_cpu[1],
                "J_true": true_x_cpu[2],
                "MAE_t_prime": float(mae_avg[0]),
                "MAE_t_dprime": float(mae_avg[1]),
                "MAE_J": float(mae_avg[2]),
                "MAE_mean": float(mae_avg.mean())
            })

        # ---------- Save per-model CSV ----------
        df = pd.DataFrame(rows).sort_values("idx").reset_index(drop=True)
        csv_path = f"{results_dir}/inverse_eval_{hidden_units}h.csv"
        df.to_csv(csv_path, index=False)

        # ---------- Summaries ----------
        mean_mae = df[["MAE_t_prime", "MAE_t_dprime", "MAE_J"]].mean().values
        std_mae  = df[["MAE_t_prime", "MAE_t_dprime", "MAE_J"]].std(ddof=0).values
        overall_mean = df["MAE_mean"].mean()
        overall_std  = df["MAE_mean"].std(ddof=0)

        # Per-model TXT summary
        with open(f"{results_dir}/inverse_summary_{hidden_units}h.txt", "w") as f:
            f.write("=== Inverse (y→x) Annealed Evaluation ===\n")
            f.write(f"Hidden units: {hidden_units}\n")
            f.write(f"Rows: {num_rows} | Repeats/row: {repeats_per_row}\n")
            f.write(f"Initial Temp: {initial_temp} | Final Temp: {final_temp} | Steps: {anneal_steps}\n\n")
            f.write("Per-parameter MAE (mean ± std):\n")
            f.write(f"  t'     : {mean_mae[0]:.6f} ± {std_mae[0]:.6f}\n")
            f.write(f"  t''    : {mean_mae[1]:.6f} ± {std_mae[1]:.6f}\n")
            f.write(f"  J      : {mean_mae[2]:.6f} ± {std_mae[2]:.6f}\n\n")
            f.write(f"Overall mean MAE across params: {overall_mean:.6f} ± {overall_std:.6f}\n")
            f.write(f"CSV: {csv_path}\n")

        return {
            "hidden_units": hidden_units,
            "mean_t'": float(mean_mae[0]),
            "std_t'": float(std_mae[0]),
            "mean_t''": float(mean_mae[1]),
            "std_t''": float(std_mae[1]),
            "mean_J": float(mean_mae[2]),
            "std_J": float(std_mae[2]),
            "overall_mean": float(overall_mean),
            "overall_std": float(overall_std),
            "csv": csv_path
        }

    except Exception as e:
        print(f"❌ Error with {hidden_units} hidden units: {e}")
        return {
            "hidden_units": hidden_units,
            "error": str(e)
        }

def main():
    # Build a single shared list of indices for ALL models for consistency
    shared_indices = build_indices_for_all_models()

    ctx = mp.get_context("spawn")  # CUDA-safe
    with ctx.Pool(processes=min(len(hidden_unit_options), os.cpu_count())) as pool:
        results = pool.map(evaluate_model, [(h, shared_indices) for h in hidden_unit_options])

    # Sort and write a compact cross-model summary
    results_sorted = sorted(results, key=lambda r: r["hidden_units"])
    summary_txt = os.path.join(results_dir, "inverse_across_models_summary.txt")
    with open(summary_txt, "w") as f:
        f.write("=== Inverse (y→x) Annealed Evaluation: Cross-Model Summary ===\n")
        f.write(f"Rows per model: {num_rows} | Repeats/row: {repeats_per_row}\n")
        f.write(f"Initial Temp: {initial_temp} | Final Temp: {final_temp} | Steps: {anneal_steps}\n\n")
        f.write("Hidden | t' MAE (mean±std) | t'' MAE (mean±std) | J MAE (mean±std) | Overall mean±std\n")
        f.write("-------------------------------------------------------------------------------\n")
        for r in results_sorted:
            if "error" in r:
                f.write(f"{r['hidden_units']:>6} | ERROR: {r['error']}\n")
            else:
                f.write(
                    f"{r['hidden_units']:>6} | "
                    f"{r['mean_t']:.6f}±{r['std_t']:.6f} | "
                    f"{r['mean_t''']:.6f}±{r['std_t''']:.6f} | "
                    f"{r['mean_J']:.6f}±{r['std_J']:.6f} | "
                    f"{r['overall_mean']:.6f}±{r['overall_std']:.6f}\n"
                )

    print("✅ Completed parallel inverse testing for all hidden unit configurations")
    print(f"📄 Per-model CSVs + TXT summaries in: {results_dir}")
    print(f"📄 Cross-model summary: {summary_txt}")

if __name__ == "__main__":
    mp.set_start_method("spawn", force=True)
    main()
