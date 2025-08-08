import os
import numpy as np
import pandas as pd
import matplotlib.pyplot as plt
import multiprocessing as mp
from tqdm import tqdm

# =======================
# 1. Config
# =======================
hidden_unit_options = [40, 42, 44, 46, 48, 50]
initial_temp = 5.0
final_temp = 0.01
anneal_steps = 10000
mse_samples = 100   # set what you want
results_dir = "multiprocessed_forward_results"
os.makedirs(results_dir, exist_ok=True)

def evaluate_model(hidden_units: int):
    """
    Worker function. Runs entirely under 'spawn' and only touches CUDA here.
    """
    try:
        import torch  # import inside worker to be extra safe

        # ---------- Device (CUDA only inside worker) ----------
        device = torch.device("cuda:0" if torch.cuda.is_available() else "cpu")

        # ---------- Load dataset (CPU) ----------
        data = pd.read_csv("t'-t''-J_data.csv")
        x_data = data.iloc[:, 0:3].values
        y_data = data.iloc[:, 3:].values
        full_data = torch.tensor(np.hstack([x_data, y_data]), dtype=torch.float32)  # stays on CPU for now

        # ---------- Model definition (no CUDA ops here) ----------
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

        # ---------- Load checkpoint on CPU, then move to device ----------
        model_file = f"joint-grbm/results/joint_grbm_{hidden_units}_hidden.pth"
        checkpoint = torch.load(model_file, map_location="cpu")
        visible_units = len(checkpoint["v_bias"])

        rbm = JointGRBM(visible_units, hidden_units, checkpoint["x_idx"], checkpoint["y_idx"]).to(device)
        rbm.W.data = checkpoint["W"].to(device)
        rbm.v_bias.data = checkpoint["v_bias"].to(device)
        rbm.h_bias.data = checkpoint["h_bias"].to(device)
        rbm.sigma.data = checkpoint["sigma"].to(device)

        print(f"✅ Loaded model with {hidden_units} hidden units")

        x_idx, y_idx = rbm.x_idx, rbm.y_idx

        # ---------- Annealing schedule ----------
        temp_decay = (final_temp / initial_temp) ** (1 / anneal_steps)

        # Choose random indices ONCE per worker
        indices = torch.randint(0, full_data.shape[0], (mse_samples,))

        all_mse = []
        last_true_y = None
        last_pred_y = None

        for idx in tqdm(indices, desc=f"Evaluating {hidden_units} hidden"):
            # Move a single sample to device as needed
            sample = full_data[idx].unsqueeze(0).to(device)
            true_y = sample[:, y_idx]

            # Start with noisy spectrum; clamp x during forward
            v_input = sample.clone()
            v_input[:, y_idx] = torch.randn_like(v_input[:, y_idx]) * 5

            for step in range(1, anneal_steps + 1):
                temperature = initial_temp * (temp_decay ** step)
                v_input = rbm.forward(v_input, steps=1, clamp_x=True, temp=temperature)

            final_mse = ((v_input[:, y_idx] - true_y) ** 2).mean().item()
            all_mse.append(final_mse)

            # keep last for plotting
            last_true_y = true_y.detach().cpu()
            last_pred_y = v_input[:, y_idx].detach().cpu()

        mean_mse = float(np.mean(all_mse))
        std_mse = float(np.std(all_mse))

        # ---------- Save per-model plot ----------
        plt.figure()
        plt.plot(last_true_y[0], label="True")
        plt.plot(last_pred_y[0], label="Predicted")
        plt.title(f"Final Forward Prediction ({hidden_units} Hidden Units)")
        plt.xlabel("ω index")
        plt.ylabel("A(ω)")
        plt.legend()
        plt.tight_layout()
        plt.savefig(f"{results_dir}/forward_final_{hidden_units}.png")
        plt.close()

        # ---------- Save per-model txt result ----------
        with open(f"{results_dir}/mse_{hidden_units}.txt", "w") as f:
            f.write("=== Forward MSE (annealed) ===\n")
            f.write(f"Hidden units: {hidden_units}\n")
            f.write(f"Samples: {mse_samples}\n")
            f.write(f"Initial Temp: {initial_temp} | Final Temp: {final_temp} | Steps: {anneal_steps}\n")
            f.write(f"Mean MSE: {mean_mse:.6f}\nStd MSE: {std_mse:.6f}\n")

        return hidden_units, mean_mse, std_mse

    except Exception as e:
        print(f"❌ Error with {hidden_units} hidden units: {e}")
        return hidden_units, None, None

def main():
    ctx = mp.get_context("spawn")
    with ctx.Pool(processes=min(len(hidden_unit_options), os.cpu_count())) as pool:
        results = pool.map(evaluate_model, hidden_unit_options)

    results = sorted(results, key=lambda x: x[0])
    with open(f"{results_dir}/mse_summary.txt", "w") as f:
        f.write("=== Final MSE Summary ===\n")
        f.write("Hidden Units | Mean MSE | Std Dev\n")
        f.write("------------------------------\n")
        for hidden_units, mean_mse, std_mse in results:
            if mean_mse is not None:
                f.write(f"{hidden_units:<13} | {mean_mse:.6f} | {std_mse:.6f}\n")
            else:
                f.write(f"{hidden_units:<13} | ERROR\n")

    print("✅ Completed parallel forward testing for all hidden unit configurations")

if __name__ == "__main__":
    mp.set_start_method("spawn", force=True)
    main()
