"""级联决策:阈值搜索(预算下最优准确率 / 准确率目标下最低成本)与方案评估。"""
import numpy as np

LEVEL_COSTS = (1.0, 8.0, 60.0)   # small / medium / large 单次推理成本


def cascade_eval(conf1, conf2, corr1, corr2, corr3, t1, t2, costs=LEVEL_COSTS):
    """给定两级阈值,返回 (准确率, 平均成本, 各级处理比例)。"""
    corr1, corr2, corr3 = (np.asarray(c, dtype=bool) for c in (corr1, corr2, corr3))
    at1 = conf1 >= t1
    at2 = (~at1) & (conf2 >= t2)
    at3 = ~(at1 | at2)
    correct = at1 * corr1 + at2 * corr2 + at3 * corr3
    cost = at1 * costs[0] + at2 * (costs[0] + costs[1]) + at3 * sum(costs)
    return (float(correct.mean()), float(cost.mean()),
            (float(at1.mean()), float(at2.mean()), float(at3.mean())))


def search_thresholds(conf1, conf2, corr1, corr2, corr3, costs=LEVEL_COSTS,
                      budget=None, acc_target=None, grid=None):
    """网格搜索 (t1, t2)。

    budget 模式    : 平均成本 <= budget 下准确率最高
    acc_target 模式: 准确率 >= acc_target 下成本最低
    返回 dict(t1, t2, acc, cost, props)
    """
    if grid is None:
        grid = np.round(np.arange(0.30, 0.995, 0.01), 2)
    corr1, corr2, corr3 = (np.asarray(c, dtype=bool) for c in (corr1, corr2, corr3))
    best = None
    for t1 in grid:
        at1 = conf1 >= t1
        esc = ~at1
        if not esc.any():
            continue
        c2_esc, k2, k3 = conf2[esc], corr2[esc], corr3[esc]
        n1, ok1 = at1.sum(), (at1 & corr1).sum()
        for t2 in grid:
            at2 = c2_esc >= t2
            n2, ok2 = at2.sum(), (at2 & k2).sum()
            n3, ok3 = esc.sum() - n2, ((~at2) & k3).sum()
            acc = (ok1 + ok2 + ok3) / len(conf1)
            cost = (n1 * costs[0] + n2 * (costs[0] + costs[1])
                    + n3 * sum(costs)) / len(conf1)
            if budget is not None:
                feasible = cost <= budget + 1e-9
                better = best is None or acc > best["acc"]
            else:
                feasible = acc >= acc_target - 1e-9
                better = best is None or cost < best["cost"]
            if feasible and better:
                best = dict(t1=float(t1), t2=float(t2), acc=float(acc), cost=float(cost),
                            props=(n1 / len(conf1), n2 / len(conf1), n3 / len(conf1)))
    return best


def sweep_budgets(conf1, conf2, corr1, corr2, corr3, budgets, costs=LEVEL_COSTS):
    """扫描一组预算,返回 [(budget, acc, cost, t1, t2)],用于画成本-准确率曲线。"""
    out = []
    for B in budgets:
        r = search_thresholds(conf1, conf2, corr1, corr2, corr3, costs, budget=B)
        if r:
            out.append((B, r["acc"], r["cost"], r["t1"], r["t2"]))
    return out
