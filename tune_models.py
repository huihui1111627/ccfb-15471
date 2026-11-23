"""快速调参:检查三个模型的准确率分层与训练稳定性。

分层手段:小模型只看 PCA 前 8 维 + 线性,中模型看前 20 维 + 小 MLP,
大模型看全部 40 维 + 大 MLP(类似真实业务里小模型用小 embedding)。
"""
import numpy as np

from cascade_sim.data import World
from cascade_sim.models import MLP


def fit_pca(X, k):
    mu = X.mean(0)
    _, _, Vt = np.linalg.svd(X - mu, full_matrices=False)
    return mu, Vt[:k].T


world = World(drift_start=10.0)
dim, K = world.dim, world.K
Xval, yval = world.sample(20_000, 0.0)

Xtr_l, ytr_l = world.sample(100_000, 0.0)
mu8, P8 = fit_pca(Xtr_l, 8)
mu20, P20 = fit_pca(Xtr_l, 20)

specs = [
    ("small ", [8, K], lambda X: (X - mu8) @ P8, 20_000, dict(epochs=20, lr=0.02)),
    ("medium", [20, 32, K], lambda X: (X - mu20) @ P20, 30_000, dict(epochs=15, lr=0.008)),
    ("large ", [dim, 256, 128, K], lambda X: X, 100_000, dict(epochs=15, lr=0.004)),
]
for name, dims, proj, n_tr, kw in specs:
    Xtr, ytr = world.sample(n_tr, 0.0)
    m = MLP(dims, seed=1).fit(proj(Xtr), ytr, seed=1, **kw)
    acc = (m.logits(proj(Xval)).argmax(1) == yval).mean()
    print(f"{name}: val_acc={acc:.4f}")
