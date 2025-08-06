import torch
import torch.nn as nn
import matplotlib.pyplot as plt
import pandas as pd
import os

# =======================
# 1. Hyperparameters
# =======================
x_units = 3          # t', t'', J
y_units = 301        # Spectral function points
visible_units = x_units + y_units
hidden_units = 64
lr = 0.001
epochs = 1000
batch_size = 32
k = 1
device = torch.device("cuda" if torch.cuda.is_available() else "cpu")

# =======================
# 2. Joint Gaussian RBM
# =======================
class JointGRBM(nn.Module):
    def __init__(self, n_vis, n_hid, x_idx, y_idx):
        super(JointGRBM, self).__init__()
        self.W = nn.Parameter(torch.randn(n_hid, n_vis) * 0.01)
        self.v_bias = nn.Parameter(torch.zeros(n_vis))
        self.h_bias = nn.Parameter(torch.zeros(n_hid))
        self.sigma = nn.Parameter(torch.ones(n_vis))
        self.x_idx = x_idx
        self.y_idx = y_idx

    def sample_h(self, v):
        prob = torch.sigmoid((v / (self.sigma**2)) @ self.W.t() + self.h_bias)
        return prob, torch.bernoulli(prob)

    def sample_v(self, h, clamp_x=False, clamp_y=False, original_v=None):
        mean = self.v_bias + (h @ self.W) * (self.sigma**2)
        sample = mean + torch.randn_like(mean) * self.sigma

        if clamp_x and original_v is not None:
            sample[:, self.x_idx] = original_v[:, self.x_idx]
        if clamp_y and original_v is not None:
            sample[:, self.y_idx] = original_v[:, self.y_idx]

        return mean, sample

    def contrastive_divergence(self, v):
        h_prob, _ = self.sample_h(v)
        pos_grad = h_prob.t() @ (v / (self.sigma**2))

        v_k = v
        for _ in range(k):
            h_prob_k, h_sample_k = self.sample_h(v_k)
            _, v_k = self.sample_v(h_sample_k)

        h_prob_neg, _ = self.sample_h(v_k)
        neg_grad = h_prob_neg.t() @ (v_k / (self.sigma**2))

        self.W.data += lr * (pos_grad - neg_grad) / v.size(0)
        self.v_bias.data += lr * torch.sum(v - v_k, dim=0) / v.size(0)
        self.h_bias.data += lr * torch.sum(h_prob - h_prob_neg, dim=0) / v.size(0)

        return torch.mean((v[:, self.y_idx] - v_k[:, self.y_idx]) ** 2)  # Focus loss on spectral part

    def forward(self, v, steps=1, clamp_x=True, clamp_y=False):
        v_sample = v
        for _ in range(steps):
            h_prob, h_sample = self.sample_h(v_sample)
            _, v_sample = self.sample_v(h_sample, clamp_x=clamp_x, clamp_y=clamp_y, original_v=v)
        return v_sample

# =======================
# 3. Load Dataset
# =======================
data = pd.read_csv("t'-t''-J_data.csv")

x_data = data.iloc[:, 0:3].values    # t', t'', J
y_data = data.iloc[:, 3:].values     # Spectral function

full_data = torch.tensor(
    pd.concat([pd.DataFrame(x_data), pd.DataFrame(y_data)], axis=1).values,
    dtype=torch.float32
)

# Indices for groups
x_idx = list(range(0, 3))
y_idx = list(range(3, 3 + y_units))

train_loader = torch.utils.data.DataLoader(full_data, batch_size=batch_size, shuffle=True)

# =======================
# 4. Train Joint GRBM
# =======================

rbm = JointGRBM(visible_units, hidden_units, x_idx, y_idx).to(device)

# Optimizer & Scheduler
optimizer = torch.optim.SGD([rbm.W, rbm.v_bias, rbm.h_bias, rbm.sigma], lr=lr)
scheduler = torch.optim.lr_scheduler.ReduceLROnPlateau(optimizer, factor=0.5, patience=20)

# Training loop
import time

# Track training time
start_time = time.time()

losses = []
for epoch in range(epochs):
    epoch_start = time.time()
    total_loss = 0
    for batch in train_loader:
        batch = batch.to(device)
        optimizer.zero_grad()

        loss = rbm.contrastive_divergence(batch)
        total_loss += loss.item()

        loss.backward = lambda: None  # RBM doesn't use autograd
        optimizer.step()

    avg_loss = total_loss / len(train_loader)
    losses.append(avg_loss)

    scheduler.step(avg_loss)

    epoch_time = time.time() - epoch_start
    print(f"Epoch [{epoch+1}/{epochs}], Loss: {avg_loss:.6f}, Time: {epoch_time:.2f}s, LR: {optimizer.param_groups[0]['lr']:.6e}")

total_time = time.time() - start_time
time_per_epoch = total_time / epochs
final_loss = losses[-1]



# =======================
# 5. Save Model
# =======================
os.makedirs("joint-grbm/results", exist_ok=True)
model_filename = f"joint_grbm_{hidden_units}_hidden.pth"

torch.save({
    'W': rbm.W,
    'v_bias': rbm.v_bias,
    'h_bias': rbm.h_bias,
    'sigma': rbm.sigma,
    'x_idx': rbm.x_idx,
    'y_idx': rbm.y_idx
}, f"joint-grbm/results/{model_filename}")

print(f"✅ Saved Joint GRBM to joint-grbm/results/{model_filename}")


# =======================
# 6. Plot Training Loss (with legend)
# =======================
os.makedirs("joint-grbm/plots", exist_ok=True)
plt.plot(losses, label="Loss")
plt.xlabel("Epoch")
plt.ylabel("Reconstruction Loss (Spectral Part)")
plt.title(f"Joint GRBM Training Loss\n{hidden_units} Hidden Units")

# Add legend with metrics
plt.legend([f"Time/epoch: {time_per_epoch:.2f}s\nFinal Loss: {final_loss:.6f}"], loc="upper right")

plt.savefig(f"joint-grbm/plots/loss_curve_{hidden_units}.png")
plt.close()

# =======================
# 7. Spectral Function Evolution (Random Sample)
# =======================
sample_idx = torch.randint(0, full_data.shape[0], (1,)).item()
sample = full_data[sample_idx].unsqueeze(0).to(device)

recon_1 = rbm.forward(sample, steps=1, clamp_x=True).detach().cpu().squeeze()
recon_5 = rbm.forward(sample, steps=5, clamp_x=True).detach().cpu().squeeze()

params = sample[0, x_idx].cpu().numpy()

plt.plot(sample[0, y_idx].cpu(), label="Input", color="blue")
plt.plot(recon_1[y_idx], label="1-step", color="orange")
plt.plot(recon_5[y_idx], label="5-steps", color="green")
plt.xlabel("ω index")
plt.ylabel("A(ω)")
plt.title(f"Joint GRBM Spectral Evolution\n(t'={params[0]:.3f}, t''={params[1]:.3f}, J={params[2]:.3f})")
plt.legend()
plt.savefig("joint-grbm/plots/spectral_evolution.png")
plt.close()

print("✅ Saved spectral evolution plot with parameter legend to joint-grbm/plots/spectral_evolution.png")
