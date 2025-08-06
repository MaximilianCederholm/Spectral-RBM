import torch
import matplotlib.pyplot as plt
import pandas as pd
import os
import numpy as np

# =======================
# 1. Config
# =======================
hidden_units = 64 # CHANGE THIS TO LOAD DIFFERENT MODELS
temperature = 0.25  # NEW TEMPERATURE PARAMETER
model_file = f"joint-grbm/results/joint_grbm_{hidden_units}_hidden.pth"
results_dir = f"test_results/{hidden_units}_hidden_{temperature}_temp"
os.makedirs(results_dir, exist_ok=True)

steps_list = [1, 2, 10, 50]
num_eval_samples = 100
device = torch.device("cuda" if torch.cuda.is_available() else "cpu")

# =======================
# 2. Load Dataset
# =======================
data = pd.read_csv("t'-t''-J_data.csv")
x_data = data.iloc[:, 0:3].values
y_data = data.iloc[:, 3:].values
full_data = torch.tensor(np.hstack([x_data, y_data]), dtype=torch.float32)

# =======================
# 3. Load Model
# =======================
class JointGRBM(torch.nn.Module):
    def __init__(self, n_vis, n_hid, x_idx, y_idx):
        super(JointGRBM, self).__init__()
        self.W = torch.nn.Parameter(torch.randn(n_hid, n_vis) * 0.01)
        self.v_bias = torch.nn.Parameter(torch.zeros(n_vis))
        self.h_bias = torch.nn.Parameter(torch.zeros(n_hid))
        self.sigma = torch.nn.Parameter(torch.ones(n_vis))
        self.x_idx = x_idx
        self.y_idx = y_idx

    def sample_h(self, v):
        prob = torch.sigmoid(((v / (self.sigma**2)) @ self.W.t() + self.h_bias) / temperature)
        return prob, torch.bernoulli(prob)

    def sample_v(self, h, clamp_x=False, clamp_y=False, original_v=None):
        mean = self.v_bias + (h @ self.W) * (self.sigma**2)
        sample = mean + torch.randn_like(mean) * self.sigma * temperature
        if clamp_x and original_v is not None:
            sample[:, self.x_idx] = original_v[:, self.x_idx]
        if clamp_y and original_v is not None:
            sample[:, self.y_idx] = original_v[:, self.y_idx]
        return mean, sample

    def forward(self, v, steps=1, clamp_x=False, clamp_y=False):
        v_sample = v.clone()
        for _ in range(steps):
            h_prob, h_sample = self.sample_h(v_sample)
            _, v_sample = self.sample_v(h_sample, clamp_x=clamp_x, clamp_y=clamp_y, original_v=v)
        return v_sample

# Load model checkpoint
checkpoint = torch.load(model_file, map_location=device)
visible_units = len(checkpoint["v_bias"])
rbm = JointGRBM(visible_units, hidden_units, checkpoint["x_idx"], checkpoint["y_idx"]).to(device)
rbm.W.data = checkpoint["W"]
rbm.v_bias.data = checkpoint["v_bias"]
rbm.h_bias.data = checkpoint["h_bias"]
rbm.sigma.data = checkpoint["sigma"]

print(f"✅ Loaded model: {model_file}")

x_idx, y_idx = rbm.x_idx, rbm.y_idx

# =======================
# Helper Functions
# =======================
def mse(a, b):
    return ((a - b) ** 2).mean().item()

def mae(a, b):
    return (a - b).abs().mean().item()

def summarize_metrics(values):
    return np.mean(values), np.std(values)

# =======================
# 4. Example Plots (Single Sample)
# =======================
example_idx = torch.randint(0, full_data.shape[0], (1,)).item()
example_sample = full_data[example_idx].unsqueeze(0).to(device)

colors = plt.cm.viridis(np.linspace(0, 1, len(steps_list)))

# --- Reconstruction ---
plt.figure()
true_y = example_sample[:, y_idx]
for color, steps in zip(colors, steps_list):
    recon = rbm.forward(example_sample, steps=steps).detach().cpu()
    mse_val = mse(recon[:, y_idx], true_y.cpu())
    size = 20 if steps == 50 else 10
    plt.scatter(range(len(recon[0, y_idx])), recon[0, y_idx],
                label=f"{steps}-step (MSE={mse_val:.4f})", s=size, color=color)
plt.scatter(range(len(true_y[0])), true_y[0].cpu(), label="True Spectrum", color="black", s=10)
params = example_sample[0, x_idx].cpu().numpy()
plt.title(f"Reconstruction\n(t'={params[0]:.3f}, t''={params[1]:.3f}, J={params[2]:.3f})")
plt.xlabel("ω index")
plt.ylabel("A(ω)")
plt.legend(fontsize=8)
plt.tight_layout()
plt.savefig(f"{results_dir}/reconstruction_example.png")
plt.close()

# --- Forward ---
plt.figure()
test_input = example_sample.clone()
test_input[:, y_idx] = torch.randn_like(test_input[:, y_idx])
true_y = example_sample[:, y_idx]
for color, steps in zip(colors, steps_list):
    recon = rbm.forward(test_input, steps=steps, clamp_x=True).detach().cpu()
    mse_val = mse(recon[:, y_idx], true_y.cpu())
    size = 20 if steps == 50 else 10
    plt.scatter(range(len(recon[0, y_idx])), recon[0, y_idx],
                label=f"{steps}-step (MSE={mse_val:.4f})", s=size, color=color)
plt.scatter(range(len(true_y[0])), true_y[0].cpu(), label="True Spectrum", color="black", s=10)
plt.title(f"Forward (x→y)\n(t'={params[0]:.3f}, t''={params[1]:.3f}, J={params[2]:.3f})")
plt.xlabel("ω index")
plt.ylabel("A(ω)")
plt.legend(fontsize=8)
plt.tight_layout()
plt.savefig(f"{results_dir}/forward_example.png")
plt.close()

# --- Inverse ---
plt.figure()
test_input = example_sample.clone()
test_input[:, x_idx] = torch.randn_like(test_input[:, x_idx])
true_x = example_sample[:, x_idx].cpu()

for color, steps in zip(colors, steps_list):
    recon = rbm.forward(test_input, steps=steps, clamp_y=True).detach().cpu()
    mae_vals = [mae(recon[:, x_idx[i]], true_x[:, i]) for i in range(3)]
    size = 50 if steps == 50 else 30
    plt.scatter(range(3), mae_vals, label=f"{steps}-step", s=size, color=color)

plt.title("Inverse (y→x) | Per-Parameter Error")
plt.xlabel("Parameter Index (0=t', 1=t'', 2=J)")
plt.ylabel("MAE")
plt.legend(fontsize=8)
plt.tight_layout()
plt.savefig(f"{results_dir}/inverse_example.png")
plt.close()

print(f"✅ Testing complete. Plots with 50-step highlighted saved in '{results_dir}'")

# =======================
# 5. 100-Sample Evaluation
# =======================
results = {"reconstruction": {}, "forward": {}, "inverse": {0: {}, 1: {}, 2: {}}}

indices = torch.randint(0, full_data.shape[0], (num_eval_samples,))
for steps in steps_list:  # steps_list now includes 50
    # Reconstruction
    rec_mse = []
    for idx in indices:
        sample = full_data[idx].unsqueeze(0).to(device)
        true_y = sample[:, y_idx]
        recon = rbm.forward(sample, steps=steps).detach().cpu()
        rec_mse.append(mse(recon[:, y_idx], true_y.cpu()))
    results["reconstruction"][steps] = summarize_metrics(rec_mse)

    # Forward
    fwd_mse = []
    for idx in indices:
        sample = full_data[idx].unsqueeze(0).to(device)
        true_y = sample[:, y_idx]
        test_input = sample.clone()
        test_input[:, y_idx] = torch.randn_like(test_input[:, y_idx])
        recon = rbm.forward(test_input, steps=steps, clamp_x=True).detach().cpu()
        fwd_mse.append(mse(recon[:, y_idx], true_y.cpu()))
    results["forward"][steps] = summarize_metrics(fwd_mse)

    # Inverse
    inv_mae = {0: [], 1: [], 2: []}
    for idx in indices:
        sample = full_data[idx].unsqueeze(0).to(device)
        true_x = sample[:, x_idx].cpu()
        test_input = sample.clone()
        test_input[:, x_idx] = torch.randn_like(test_input[:, x_idx])
        recon = rbm.forward(test_input, steps=steps, clamp_y=True).detach().cpu()
        for i in range(3):
            inv_mae[i].append(mae(recon[:, x_idx[i]], true_x[:, i]))
    for i in range(3):
        results["inverse"][i][steps] = summarize_metrics(inv_mae[i])

# =======================
# 6. Save Results to File
# =======================
with open(f"{results_dir}/evaluation_results_{hidden_units}h_{temperature}T.txt", "w") as f:
    f.write("=== Joint GRBM Evaluation Results ===\n")
    f.write(f"Temperature: {temperature}\n\n")  # include temperature in the header

    f.write("Reconstruction (y):\n")
    for steps in steps_list:
        mean, std = results["reconstruction"][steps]
        f.write(f"  Step {steps}: MSE = {mean:.6f} ± {std:.6f}\n")

    f.write("\nForward (x→y):\n")
    for steps in steps_list:
        mean, std = results["forward"][steps]
        f.write(f"  Step {steps}: MSE = {mean:.6f} ± {std:.6f}\n")

    f.write("\nInverse (y→x):\n")
    for i, name in enumerate(["t'", "t''", "J"]):
        f.write(f"  {name}:\n")
        for steps in steps_list:
            mean, std = results["inverse"][i][steps]
            f.write(f"    Step {steps}: MAE = {mean:.6f} ± {std:.6f}\n")

print(f"✅ Testing complete. Results (including 50-step) saved in '{results_dir}'")
