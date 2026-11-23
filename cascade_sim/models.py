"""NumPy 实现的分类网络(代替大中小三个模型)和一个逻辑回归判别器。

三个模型规模/训练数据量/推理成本依次递增:
- small : 线性 softmax(40->20),           成本 1
- medium: MLP 40->48->20,                 成本 8
- large : MLP 40->192->96->20,            成本 60
"""
import numpy as np


def softmax(z):
    z = z - z.max(axis=1, keepdims=True)
    e = np.exp(z)
    return e / e.sum(axis=1, keepdims=True)


class MLP:
    def __init__(self, dims, seed=0):
        rng = np.random.default_rng(seed)
        self.dims = dims
        self.W, self.b = [], []
        for i in range(len(dims) - 1):
            self.W.append(rng.normal(0, np.sqrt(2.0 / dims[i]), (dims[i], dims[i + 1])))
            self.b.append(np.zeros(dims[i + 1]))

    def forward(self, X, rng=None, drop=0.0):
        """前向;传入 rng 和 drop>0 时启用 dropout(用于多次随机推理)。"""
        A = X
        if rng is not None and drop > 0:
            A = A * (rng.random(A.shape) >= drop) / (1 - drop)
        self.cache = [A]
        for i, (W, b) in enumerate(zip(self.W, self.b)):
            Z = A @ W + b
            last = i == len(self.W) - 1
            A = Z if last else np.maximum(Z, 0)
            if not last and rng is not None and drop > 0:
                A = A * (rng.random(A.shape) >= drop) / (1 - drop)
            self.cache.append(A)
        return A  # logits

    def logits(self, X):
        return self.forward(X)

    def fit(self, X, y, epochs=10, batch_size=256, lr=0.01, seed=0, verbose=False):
        rng = np.random.default_rng(seed)
        n = len(y)
        # Adam 状态
        mW = [np.zeros_like(W) for W in self.W]
        vW = [np.zeros_like(W) for W in self.W]
        mb = [np.zeros_like(b) for b in self.b]
        vb = [np.zeros_like(b) for b in self.b]
        b1, b2, eps, step = 0.9, 0.999, 1e-8, 0
        for ep in range(epochs):
            perm = rng.permutation(n)
            for s in range(0, n, batch_size):
                idx = perm[s:s + batch_size]
                xb, yb = X[idx], y[idx]
                logits = self.forward(xb)
                G = softmax(logits)
                G[np.arange(len(yb)), yb] -= 1
                G /= len(yb)
                gW, gb = [None] * len(self.W), [None] * len(self.b)
                for i in reversed(range(len(self.W))):
                    gW[i] = self.cache[i].T @ G
                    gb[i] = G.sum(0)
                    G = G @ self.W[i].T
                    if i > 0:
                        G = G * (self.cache[i] > 0)  # ReLU'
                # 全局梯度范数裁剪,防止发散
                norm = np.sqrt(sum((g ** 2).sum() for g in gW)
                               + sum((g ** 2).sum() for g in gb))
                if not np.isfinite(norm):
                    continue
                if norm > 5.0:
                    scale = 5.0 / norm
                    gW = [g * scale for g in gW]
                    gb = [g * scale for g in gb]
                step += 1
                for i in range(len(self.W)):
                    mW[i] = b1 * mW[i] + (1 - b1) * gW[i]
                    vW[i] = b2 * vW[i] + (1 - b2) * gW[i] * gW[i]
                    mb[i] = b1 * mb[i] + (1 - b1) * gb[i]
                    vb[i] = b2 * vb[i] + (1 - b2) * gb[i] * gb[i]
                    mh = mW[i] / (1 - b1 ** step)
                    vh = vW[i] / (1 - b2 ** step)
                    self.W[i] -= lr * mh / (np.sqrt(vh) + eps)
                    self.b[i] -= lr * (mb[i] / (1 - b1 ** step)) / (np.sqrt(vb[i] / (1 - b2 ** step)) + eps)
            if verbose:
                pred = self.logits(X[:5000]).argmax(1)
                print(f"  epoch {ep}: acc(head5k)={np.mean(pred == y[:5000]):.4f}")
        return self

    def mc_predict(self, X, m=6, drop=0.15, seed=0):
        """m 次随机前向,返回 (多数派预测, 一致率, 平均置信)。"""
        rng = np.random.default_rng(seed)
        K = self.dims[-1]
        counts = np.zeros((len(X), K))
        confs = []
        for _ in range(m):
            p = softmax(self.forward(X, rng=rng, drop=drop))
            counts += np.eye(K)[p.argmax(1)]
            confs.append(p.max(1))
        maj = counts.argmax(1)
        agree = counts.max(1) / m
        return maj, agree, np.mean(confs, axis=0)


class LogisticDisc:
    """判断"小模型这条请求会不会答错"的二分类判别器。"""

    def __init__(self):
        self.mu = self.sd = None
        self.w = self.b0 = None

    def fit(self, X, y, lr=0.5, epochs=300, l2=1e-3):
        self.mu, self.sd = X.mean(0), X.std(0) + 1e-8
        Z = (X - self.mu) / self.sd
        self.w = np.zeros(Z.shape[1])
        self.b0 = 0.0
        n = len(y)
        for _ in range(epochs):
            p = 1 / (1 + np.exp(-(Z @ self.w + self.b0)))
            g = p - y
            self.w -= lr * (Z.T @ g / n + l2 * self.w)
            self.b0 -= lr * g.mean()
        return self

    def prob_correct(self, X):
        Z = (X - self.mu) / self.sd
        return 1 / (1 + np.exp(-(Z @ self.w + self.b0)))


def disc_features(prob1, mc_agree, mc_conf, X):
    """由小模型输出 + 多次随机推理一致性 + 输入特征构造判别器特征。"""
    s = np.sort(prob1, axis=1)
    margin = s[:, -1] - s[:, -2]
    ent = -(prob1 * np.log(prob1 + 1e-12)).sum(1)
    return np.column_stack([
        prob1.max(1), margin, ent, mc_agree, mc_conf,
        np.linalg.norm(X, axis=1), np.abs(X).max(1),
    ])
