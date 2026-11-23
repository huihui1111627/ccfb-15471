"""仿真请求流:客服问答 + 内容审核混合业务。

特点(对应真实业务):
- 20 个意图/审核类别,类别先验不均衡(头部类别流量大)
- 每个类别难度不同(类内噪声 sigma_c 不同,部分类别对互相混淆)
- 单请求难度不均:大多数是"简单"样本,一部分是"难"样本
  (落在两个类别的决策边界附近,本质上有歧义)
- 分布随时间漂移:类别先验漂移 + 特征空间缓慢旋转 + 难样本比例上升
"""
import numpy as np

CLASS_NAMES = [
    "账单咨询", "退款售后", "物流查询", "账号安全", "密码重置",
    "产品咨询", "使用故障", "投诉建议", "优惠活动", "发票开具",
    "审核-广告", "审核-色情", "审核-暴恐", "审核-涉政", "审核-辱骂",
    "审核-诈骗", "审核-侵权", "审核-垃圾信息", "审核-未成年人", "审核-谣言",
]


class World:
    def __init__(self, n_classes=20, dim=40, n_sub=3, seed=7,
                 drift_start=10.0, drift_ramp=8.0):
        assert n_classes <= len(CLASS_NAMES)
        self.K, self.dim, self.n_sub = n_classes, dim, n_sub
        self.drift_start, self.drift_ramp = drift_start, drift_ramp
        rng = np.random.default_rng(seed)
        self.rng = rng

        # 类别中心
        self.class_center = rng.normal(0, 2.2, (n_classes, dim))
        # 制造几对容易混淆的类别(中心拉近)
        self.confuse_pairs = []
        perm = rng.permutation(n_classes)
        for i in range(0, 12, 2):
            a, b = int(perm[i]), int(perm[i + 1])
            m = 0.5 * (self.class_center[a] + self.class_center[b])
            self.class_center[a] = m + 0.30 * (self.class_center[a] - m)
            self.class_center[b] = m + 0.30 * (self.class_center[b] - m)
            self.confuse_pairs.append((a, b))

        # 每个类别由 n_sub 个子簇组成(多模态) -> 决策边界非线性,
        # 线性小模型无法完全拟合,大模型可以 -> 模型能力拉开差距
        self.sub_offset = np.zeros((n_classes, n_sub, dim))
        for c in range(n_classes):
            dirs = rng.normal(0, 1, (n_sub, dim))
            dirs /= np.linalg.norm(dirs, axis=1, keepdims=True)
            self.sub_offset[c] = dirs * rng.uniform(2.5, 4.5, (n_sub, 1))

        # 每个类别难度不同
        self.sigma = rng.uniform(0.9, 1.8, n_classes)
        # 类别先验不均衡
        self.base_prior = rng.dirichlet(np.full(n_classes, 1.5))
        # 漂移方向:各类别流量有升有降(持续性分布偏移)
        self.drift_delta = rng.normal(0, 1, n_classes)
        # 特征漂移用的随机旋转平面
        self.rot_planes = []
        for _ in range(4):
            i, j = rng.choice(dim, size=2, replace=False)
            self.rot_planes.append((int(i), int(j)))

    # ---- 漂移强度:0 (t<=drift_start) 平滑升到 1 ----
    def drift_amount(self, t):
        s = np.clip((t - self.drift_start) / self.drift_ramp, 0.0, 1.0)
        return s * s * (3 - 2 * s)

    def prior(self, t):
        s = self.drift_amount(t)
        if s <= 0:
            return self.base_prior
        w = self.base_prior * np.exp(1.2 * s * self.drift_delta)
        return w / w.sum()

    def _rotate(self, X, theta):
        X = X.copy()
        for i, j in self.rot_planes:
            ci, cj = X[:, i].copy(), X[:, j].copy()
            X[:, i] = np.cos(theta) * ci - np.sin(theta) * cj
            X[:, j] = np.sin(theta) * ci + np.cos(theta) * cj
        return X

    def sample(self, n, t, seed=None):
        """在逻辑时间 t(天)采样 n 条请求,返回 (X, y)。"""
        rng = np.random.default_rng(seed) if seed is not None else self.rng
        s = self.drift_amount(t)
        p = self.prior(t)
        c = rng.choice(self.K, size=n, p=p)

        hard_frac = 0.25 + 0.12 * s           # 漂移后难样本变多
        hard = rng.random(n) < hard_frac
        sub = rng.integers(0, self.n_sub, n)
        noise = self.sigma[c][:, None] * (1 + 0.25 * s) * rng.normal(size=(n, self.dim))
        X = self.class_center[c] + self.sub_offset[c, sub] + noise

        # 难样本:两个类别子簇之间的插值(更靠近真实类但贴着边界)+ 更大噪声
        if hard.any():
            idx = np.where(hard)[0]
            d = rng.integers(0, self.K, len(idx))
            alpha = rng.uniform(0.50, 0.65, len(idx))[:, None]
            sub_d = rng.integers(0, self.n_sub, len(idx))
            proto_c = self.class_center[c[idx]] + self.sub_offset[c[idx], sub[idx]]
            proto_d = self.class_center[d] + self.sub_offset[d, sub_d]
            X[idx] = (alpha * proto_c + (1 - alpha) * proto_d
                      + 1.3 * self.sigma[c[idx]][:, None] * rng.normal(size=(len(idx), self.dim)))

        theta = 0.5 * s                        # 特征空间缓慢旋转
        if theta > 0:
            X = self._rotate(X, theta)
        return X, c
