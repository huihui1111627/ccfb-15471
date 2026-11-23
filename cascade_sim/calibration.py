"""置信度校准:温度缩放 + 分箱校准,以及 ECE / 可靠性曲线数据。"""
import numpy as np

from .models import softmax


def nll_of_T(logits, y, T):
    p = softmax(logits / T)
    return -np.log(p[np.arange(len(y)), y] + 1e-12).mean()


def fit_temperature(logits, y, grid=None):
    """在保留校准集上最小化 NLL 搜索温度 T。"""
    if grid is None:
        grid = np.concatenate([np.linspace(0.3, 2.0, 35), np.linspace(2.1, 6.0, 40)])
    nlls = np.array([nll_of_T(logits, y, T) for T in grid])
    T0 = grid[int(np.argmin(nlls))]
    # 细搜索
    fine = np.linspace(max(0.1, T0 - 0.15), T0 + 0.15, 31)
    nlls = np.array([nll_of_T(logits, y, T) for T in fine])
    return float(fine[int(np.argmin(nlls))])


def confidence(logits, T=1.0):
    """温度缩放后的 top-1 置信度。"""
    return softmax(logits / T).max(1)


class HistogramBinning:
    """等频分箱校准:把温度缩放后的置信度映射到经验准确率。"""

    def __init__(self, n_bins=15):
        self.n_bins = n_bins
        self.edges = None
        self.values = None

    def fit(self, conf, correct):
        qs = np.linspace(0, 1, self.n_bins + 1)
        self.edges = np.quantile(conf, qs)
        self.edges[0], self.edges[-1] = 0.0, 1.0 + 1e-9
        self.values = np.zeros(self.n_bins)
        for i in range(self.n_bins):
            m = (conf >= self.edges[i]) & (conf < self.edges[i + 1])
            self.values[i] = correct[m].mean() if m.any() else conf[m].mean() if m.any() else 0.5
        return self

    def predict(self, conf):
        idx = np.clip(np.searchsorted(self.edges, conf, side="right") - 1, 0, self.n_bins - 1)
        return self.values[idx]


def ece(conf, correct, n_bins=15):
    """期望校准误差(等宽分箱)。"""
    edges = np.linspace(0, 1, n_bins + 1)
    total = 0.0
    for i in range(n_bins):
        m = (conf >= edges[i]) & (conf < edges[i + 1] if i < n_bins - 1 else conf <= edges[i + 1])
        if m.any():
            total += m.mean() * abs(correct[m].mean() - conf[m].mean())
    return float(total)


def reliability_curve(conf, correct, n_bins=15):
    """返回 (bin中心置信度, bin经验准确率, bin样本数)。"""
    edges = np.linspace(0, 1, n_bins + 1)
    centers, accs, counts = [], [], []
    for i in range(n_bins):
        m = (conf >= edges[i]) & (conf < edges[i + 1] if i < n_bins - 1 else conf <= edges[i + 1])
        if m.any():
            centers.append(conf[m].mean())
            accs.append(correct[m].mean())
            counts.append(int(m.sum()))
    return np.array(centers), np.array(accs), np.array(counts)
