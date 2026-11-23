"""在线级联服务:批处理、前级结果复用、抽查监控、自动重校准与阈值调整。

- 请求按批到达,整批走小模型;升级的请求在批内重新组批送下一级
- 前一级算过的 logits/置信度被缓存复用(判别器、校准、监控直接用,不重算)
- 按 audit_rate 抽查请求同时送大模型,得到参考标签,进入滑动缓冲
- 监控到实际准确率下滑 / 校准失效时,用缓冲数据重新拟合温度+分箱,
  并在预算约束下重新搜索阈值
"""
import numpy as np

from .calibration import fit_temperature, confidence, HistogramBinning, ece
from .cascade import search_thresholds, cascade_eval, LEVEL_COSTS


class CascadeService:
    def __init__(self, costs=LEVEL_COSTS, budget=None, acc_target=None,
                 audit_rate=0.05, auto_recal=False, recal_cooldown=2,
                 acc_drop_tol=0.03, ece_tol=0.10, buffer_max=2000, min_buffer=1500,
                 target_margin=0.02, seed=0):
        self.costs = costs
        self.budget = budget
        self.acc_target = acc_target
        self.audit_rate = audit_rate if auto_recal else 0.0
        self.auto_recal = auto_recal
        self.recal_cooldown = recal_cooldown
        self.acc_drop_tol = acc_drop_tol
        self.ece_tol = ece_tol
        self.buffer_max = buffer_max
        self.min_buffer = min_buffer
        self.target_margin = target_margin
        self.rng = np.random.default_rng(seed)
        # 校准状态
        self.T = [1.0, 1.0, 1.0]
        self.binning = [None, None, None]
        self.t1, self.t2 = 0.5, 0.5
        # 监控状态
        self.buf = {k: [] for k in ("l1", "l2", "l3", "ref")}
        self.days_since_recal = recal_cooldown
        self.recal_events = []
        self.baseline_acc = None

    # ---- 校准 ----
    def calibrate(self, calib_logits, calib_y):
        for i in range(3):
            self.T[i] = fit_temperature(calib_logits[i], calib_y)
            conf = confidence(calib_logits[i], self.T[i])
            correct = (calib_logits[i].argmax(1) == calib_y).astype(float)
            self.binning[i] = HistogramBinning(15).fit(conf, correct)

    def conf(self, level, logits):
        c = confidence(logits, self.T[level])
        if self.binning[level] is not None:
            c = self.binning[level].predict(c)
        return c

    # ---- 单天处理(批处理 + 组批升级 + 抽查) ----
    def process_day(self, day, logits1, logits2, logits3, y, batch_size=128, score1=None):
        n = len(y)
        level_of = np.zeros(n, dtype=int)
        pred = np.zeros(n, dtype=int)
        # 第一级:整批小模型(置信度由缓存 logits 直接算,复用;
        # 也可由判别器分数 score1 替代)
        conf1 = score1 if score1 is not None else self.conf(0, logits1)
        esc1 = conf1 < self.t1
        level_of[~esc1] = 1
        pred[~esc1] = logits1[~esc1].argmax(1)
        # 升级的请求重新组批送第二级
        idx1 = np.where(esc1)[0]
        conf2 = self.conf(1, logits2[idx1]) if len(idx1) else np.array([])
        esc2 = conf2 < self.t2
        lvl2 = idx1[~esc2]
        level_of[lvl2] = 2
        pred[lvl2] = logits2[lvl2].argmax(1)
        idx2 = idx1[esc2]
        level_of[idx2] = 3
        pred[idx2] = logits3[idx2].argmax(1)

        cost = (np.mean(level_of == 1) * self.costs[0]
                + np.mean(level_of == 2) * (self.costs[0] + self.costs[1])
                + np.mean(level_of == 3) * sum(self.costs))

        # 抽查:随机请求送大模型 + 人工复核,得到参考标签(仿真中用真实标签
        # 代表复核结果;大模型调用计入成本)。若只用大模型预测当参考,阈值搜索
        # 会变成"最大化与大模型的一致率",倾向于不升级(小模型和大模型同错时
        # 一致率仍为 1),所以复核标签是必要的。
        n_audit = 0
        est_acc = None
        if self.audit_rate > 0:
            amask = self.rng.random(n) < self.audit_rate
            n_audit = int(amask.sum())
            if n_audit:
                ref = y[amask]  # 复核后的参考标签
                est_acc = float((pred[amask] == ref).mean())  # 抽查估计的实际准确率
                self.buf["l1"].append(logits1[amask])
                self.buf["l2"].append(logits2[amask])
                self.buf["l3"].append(logits3[amask])
                self.buf["ref"].append(ref)
                for k in ("l1", "l2", "l3", "ref"):
                    self.buf[k] = [np.concatenate(self.buf[k])[-self.buffer_max:]]
                # 抽查成本:没走到大模型的请求补一次大模型调用
                extra = amask & (level_of < 3)
                cost += extra.mean() * sum(self.costs)

        acc = float((pred == y).mean())
        props = tuple(float((level_of == k).mean()) for k in (1, 2, 3))
        # 批处理统计:各级实际组批数 / 平均批大小
        nb1 = int(np.ceil(n / batch_size))
        nb2 = int(np.ceil(len(idx1) / batch_size)) if len(idx1) else 0
        nb3 = int(np.ceil(len(idx2) / batch_size)) if len(idx2) else 0
        stats = dict(day=day, acc=acc, cost=float(cost), props=props,
                     batches=(nb1, nb2, nb3),
                     avg_batch=(n / max(nb1, 1),
                                len(idx1) / max(nb2, 1) if nb2 else 0.0,
                                len(idx2) / max(nb3, 1) if nb3 else 0.0),
                     t1=self.t1, t2=self.t2, n_audit=n_audit, est_acc=est_acc)
        self.days_since_recal += 1
        if self.auto_recal:
            self._maybe_recalibrate(day, stats)
        return stats

    # ---- 监控与自动重校准 ----
    def _buf_arrays(self):
        if not self.buf["ref"]:
            return None
        return tuple(np.concatenate(self.buf[k]) for k in ("l1", "l2", "l3", "ref"))

    def _maybe_recalibrate(self, day, stats):
        buf = self._buf_arrays()
        if (buf is None or len(buf[3]) < self.min_buffer
                or self.days_since_recal < self.recal_cooldown):
            return
        l1, l2, l3, ref = buf
        # 在滑动抽查缓冲上(复核标签)估计当前实际表现,作为监控指标
        conf1 = self.conf(0, l1)
        corr1 = (l1.argmax(1) == ref).astype(float)
        cur_ece = ece(conf1, corr1)
        conf2 = self.conf(1, l2)
        corr2 = (l2.argmax(1) == ref).astype(float)
        corr3 = (l3.argmax(1) == ref).astype(float)
        acc_now, _, _ = cascade_eval(conf1, conf2, corr1, corr2, corr3,
                                     self.t1, self.t2, self.costs)
        if self.baseline_acc is None:
            self.baseline_acc = acc_now
        acc_drop = self.baseline_acc - acc_now
        # 周期性重校准评估(等同工程上的每日 cron):滑动窗口内数据始终较新,
        # 比"估计偏差触发"稳健——触发式估计本身也受校准失效影响。
        # 监控指标(准确率下滑、ECE)随事件记录,用于告警展示。
        triggered = acc_drop >= self.acc_drop_tol or cur_ece >= self.ece_tol
        # 三份交叉 + 滞回采纳:
        #   A 半拟合候选温度+分箱,B 搜索候选阈值,C 独立评估当前 vs 候选工作点
        # (若在同一份数据上搜索又评估,4900 个组合里选最优会产生"赢家诅咒"偏差)
        # 只有候选在 C 上明显更优才采纳,避免小样本噪声导致生产配置抖动
        n = len(ref)
        perm = self.rng.permutation(n)
        ia = perm[: n // 2]
        ib = perm[n // 2: n * 3 // 4]
        ic = perm[n * 3 // 4:]
        new_T, new_bin = [], []
        for i, lg in enumerate((l1, l2, l3)):
            new_T.append(fit_temperature(lg[ia], ref[ia]))
            conf = confidence(lg[ia], new_T[i])
            corr = (lg[ia].argmax(1) == ref[ia]).astype(float)
            new_bin.append(HistogramBinning(15).fit(conf, corr))

        def conf_with(T, bn, lg):
            return bn.predict(confidence(lg, T))

        # 候选阈值(候选校准,在 B 上搜索)
        c1 = conf_with(new_T[0], new_bin[0], l1[ib])
        c2 = conf_with(new_T[1], new_bin[1], l2[ib])
        kb = [(lg[ib].argmax(1) == ref[ib]).astype(float) for lg in (l1, l2, l3)]
        target = self.acc_target + self.target_margin if self.acc_target else None
        r = search_thresholds(c1, c2, *kb, self.costs,
                              budget=self.budget, acc_target=target)
        if r is None:  # 目标不可达:退化为预算不限时准确率最高的方案
            r = search_thresholds(c1, c2, *kb, self.costs,
                                  budget=sum(self.costs))
        # 在独立的 C 上评估当前 vs 候选工作点
        kc = [(lg[ic].argmax(1) == ref[ic]).astype(float) for lg in (l1, l2, l3)]
        acc_cur, cost_cur, _ = cascade_eval(self.conf(0, l1[ic]), self.conf(1, l2[ic]),
                                            *kc, self.t1, self.t2, self.costs)
        obj_cur = self._objective(acc_cur, cost_cur)
        if r:
            acc_new, cost_new, _ = cascade_eval(
                conf_with(new_T[0], new_bin[0], l1[ic]),
                conf_with(new_T[1], new_bin[1], l2[ic]),
                *kc, r["t1"], r["t2"], self.costs)
            obj_new = self._objective(acc_new, cost_new)
        else:
            obj_new = np.inf
        adopted = obj_new < obj_cur - 2.0
        if adopted:
            self.T, self.binning = new_T, new_bin
            self.t1, self.t2 = r["t1"], r["t2"]
        self.recal_events.append(dict(
            day=day, acc_drop=float(acc_drop), ece=float(cur_ece),
            triggered=bool(triggered), adopted=bool(adopted),
            obj_cur=round(float(obj_cur), 1), obj_new=round(float(obj_new), 1),
            new_T=[round(t, 2) for t in self.T],
            new_t1=self.t1, new_t2=self.t2))
        self.days_since_recal = 0
        self.baseline_acc = acc_now

    def _objective(self, acc, cost):
        """工作点目标函数(成本单位):目标准确率模式惩罚欠额,预算模式惩罚超支。"""
        if self.acc_target is not None:
            return max(0.0, self.acc_target + self.target_margin - acc) * 1000 + cost
        return max(0.0, cost - (self.budget or 0)) * 100 - acc * 100
