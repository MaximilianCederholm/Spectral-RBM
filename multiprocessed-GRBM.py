from multiprocessing import Process

def train_grbm(hidden_units):
    import torch
    import pandas as pd
    import os
    import matplotlib.pyplot as plt
    import time
    import torch.nn as nn

    # =======================
    # 1. Setup
    # =======================
    x_units = 3
    y_units = 301
    visible_units = x_units + y_units
    lr = 0.0005
    epochs = 1000
    batch_size = 32
    initial_k = 1
    final_k = 5
    k_schedule_epoch = 500
    initial_momentum = 0.5
    final_momentum = 0.9
    momentum_schedule_epoch = 300
    weight_decay = 1e-4
    use_pcd = True
    diagnostics = []

    class JointGRBM(nn.Module):
        def __init__(self, n_vis, n_hid, x_idx, y_idx, k=1, momentum=0.5):
            super(JointGRBM, self).__init__()
            self.W = nn.Parameter(torch.randn(n_hid, n_vis) * 0.01)
            self.v_bias = nn.Parameter(torch.zeros(n_vis))
            self.h_bias = nn.Parameter(torch.zeros(n_hid))
            self.sigma = nn.Parameter(torch.ones(n_vis))
            self.x_idx = x_idx
            self.y_idx = y_idx
            self.k = k
            self.momentum = momentum
            self.persistent_chain = None

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

            if use_pcd:
                if self.persistent_chain is None or self.persistent_chain.shape[0] != v.shape[0]:
                    self.persistent_chain = v.clone().detach()

                v_k = self.persistent_chain
                for _ in range(self.k):
                    h_prob_k, h_sample_k = self.sample_h(v_k)
                    _, v_k = self.sample_v(h_sample_k)
                self.persistent_chain = v_k.detach()
            else:
                v_k = v
                for _ in range(self.k):
                    h_prob_k, h_sample_k = self.sample_h(v_k)
                    _, v_k = self.sample_v(h_sample_k)

            h_prob_neg, _ = self.sample_h(v_k)
            neg_grad = h_prob_neg.t() @ (v_k / (self.sigma**2))
            reg_term = weight_decay * self.W
            loss = torch.mean((v[:, y_idx] - v_k[:, y_idx]) ** 2)
            return loss, pos_grad, neg_grad, h_prob, h_prob_neg, v_k, reg_term

    # =======================
    # 2. Load Dataset
    # =======================
    data = pd.read_csv("t'-t''-J_data.csv")
    x_data = data.iloc[:, 0:3].values
    y_data = data.iloc[:, 3:].values
    full_data = torch.tensor(pd.concat([pd.DataFrame(x_data), pd.DataFrame(y_data)], axis=1).values, dtype=torch.float32)
    x_idx = list(range(0, 3))
    y_idx = list(range(3, 3 + y_units))
    train_loader = torch.utils.data.DataLoader(full_data, batch_size=batch_size, shuffle=True)

    # =======================
    # 3. Training
    # =======================
    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    rbm = JointGRBM(visible_units, hidden_units, x_idx, y_idx, k=initial_k, momentum=initial_momentum).to(device)
    optimizer = torch.optim.SGD(rbm.parameters(), lr=lr)

    losses, sigma_history = [], []
    for epoch in range(epochs):
        total_loss = 0
        rbm.k = min(final_k, initial_k + (final_k - initial_k) * epoch // k_schedule_epoch)
        rbm.momentum = min(final_momentum, initial_momentum + (final_momentum - initial_momentum) * epoch // momentum_schedule_epoch)
        for batch in train_loader:
            batch = batch.to(device)
            optimizer.zero_grad()
            loss, *_ = rbm.contrastive_divergence(batch)
            loss.backward()
            optimizer.step()
            total_loss += loss.item()

        avg_loss = total_loss / len(train_loader)
        losses.append(avg_loss)
        sigma_history.append(rbm.sigma.detach().cpu().numpy())

        print(f"[{hidden_units}] Epoch {epoch+1}/{epochs} | Loss: {avg_loss:.6f}")

    # =======================
    # 4. Save Model
    # =======================
    model_filename = f"joint_grbm_{hidden_units}_hidden.pth"
    os.makedirs("joint-grbm/results", exist_ok=True)
    torch.save({
        'W': rbm.W,
        'v_bias': rbm.v_bias,
        'h_bias': rbm.h_bias,
        'sigma': rbm.sigma,
        'x_idx': rbm.x_idx,
        'y_idx': rbm.y_idx
    }, f"joint-grbm/results/{model_filename}")
    print(f"✅ Saved GRBM ({hidden_units} hidden) to {model_filename}")

# =======================
# 5. Spawn processes
# =======================
if __name__ == "__main__":
    configs = [40, 42, 44, 46, 48, 50]
    jobs = [Process(target=train_grbm, args=(h,)) for h in configs]
    for job in jobs:
        job.start()
    for job in jobs:
        job.join()
