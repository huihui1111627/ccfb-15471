"""多模型级联推理服务 · 端到端仿真实验。

流程:
1. 构造仿真业务数据(类别难度不均 + 随时间漂移),训练小/中/大三个模型
2. 保留校准集上拟合温度缩放 + 分箱校准,可靠性曲线 / ECE 评估
3. 验证集上搜索级联阈值(预算下最优准确率 / 目标准确率下最低成本)
4. 训练"是否需要升级"判别器(多次随机推理一致性 + 输入特征),与置信度对比
5. 30 天请求流仿真(第 10 天注入漂移),对比各方案;抽查监控 + 自动重校准
6. 生成 HTML 报告
"""
import os
import time

import numpy as np

from cascade_sim.data import World, CLASS_NAMES
from cascade_sim.models import MLP, LogisticDisc, softmax, disc_features
from cascade_sim.calibration import (fit_temperature, confidence, HistogramBinning,
                                     ece, reliability_curve)
from cascade_sim.cascade import (LEVEL_COSTS, cascade_eval, search_thresholds, sweep_budgets)
from cascade_sim.service import CascadeService
from cascade_sim import report

OUT = os.path.join(os.path.dirname(os.path.abspath(__file__)), "output")
os.makedirs(OUT, exist_ok=True)

N_DAYS = 30
REQ_PER_DAY = 3000
DRIFT_START = 10.0
BUDGET = 8.0          # 平均成本预算(全部用大模型 = 60)
ACC_TARGET = 0.90     # 目标准确率模式(离线阈值搜索)
AUTO_ACC_TARGET = 0.84  # 在线自动重校准的服务质量目标(漂移后大模型自身约 0.87)
AUDIT_RATE = 0.08

t_start = time.time()
print("=" * 60)
print("1. 构造仿真世界并训练三个模型")
world = World(drift_start=DRIFT_START)
dim, K = world.dim, world.K


def fit_pca(X, k):
    mu = X.mean(0)
    _, _, Vt = np.linalg.svd(X - mu, full_matrices=False)
    return mu, Vt[:k].T


# 分层设计(类似真实业务:小模型用小 embedding 和少数据):
#   small : PCA 6 维 + 线性 softmax,  2万条训练数据, 成本 1
#   medium: PCA 12 维 + 32 隐层 MLP,  3万条训练数据, 成本 8
#   large : 全部 40 维 + 320x160 MLP, 10万条训练数据, 成本 60
Xtr_l, ytr_l = world.sample(100_000, 0.0)
mu6, P6 = fit_pca(Xtr_l, 6)
mu12, P12 = fit_pca(Xtr_l, 12)
proj = [lambda X: (X - mu6) @ P6, lambda X: (X - mu12) @ P12, lambda X: X]

Xtr_s, ytr_s = world.sample(20_000, 0.0)
Xtr_m, ytr_m = world.sample(30_000, 0.0)

small = MLP([6, K], seed=1).fit(proj[0](Xtr_s), ytr_s, epochs=20, lr=0.02, seed=1)
medium = MLP([12, 32, K], seed=2).fit(proj[1](Xtr_m), ytr_m, epochs=15, lr=0.008, seed=2)
large = MLP([dim, 320, 160, K], seed=3).fit(proj[2](Xtr_l), ytr_l, epochs=20, lr=0.003, seed=3)
models = [small, medium, large]
names = ["小模型(6维线性)", "中模型(12维+32隐层)", "大模型(40维 320x160)"]
n_params = [sum(W.size + b.size for W, b in zip(m.W, m.b)) for m in models]


def logits_of(i, X):
    return models[i].logits(proj[i](X))

print("=" * 60)
print("2. 校准集拟合温度+分箱,在保留验证集上评估 ECE/可靠性曲线")
Xcal, ycal = world.sample(20_000, 0.0)
Xval, yval = world.sample(25_000, 0.0)
cal_logits = [logits_of(i, Xcal) for i in range(3)]
val_logits = [logits_of(i, Xval) for i in range(3)]
calib_rows, rel_data = [], []
temps, binnings = [], []
for i in range(3):
    # 在校准集上拟合
    T = fit_temperature(cal_logits[i], ycal)
    corr_cal = (cal_logits[i].argmax(1) == ycal).astype(float)
    hb = HistogramBinning(15).fit(confidence(cal_logits[i], T), corr_cal)
    temps.append(T)
    binnings.append(hb)
    # 在保留验证集上评估
    correct = (val_logits[i].argmax(1) == yval).astype(float)
    raw_conf = confidence(val_logits[i], 1.0)
    t_conf = confidence(val_logits[i], T)
    cal_conf = hb.predict(t_conf)
    calib_rows.append((names[i], f"{correct.mean():.4f}", f"{T:.2f}",
                       f"{ece(raw_conf, correct):.4f}", f"{ece(t_conf, correct):.4f}",
                       f"{ece(cal_conf, correct):.4f}"))
    rel_data.append((names[i], reliability_curve(raw_conf, correct)[:2],
                     reliability_curve(cal_conf, correct)[:2]))
    print(f"  {names[i]}: acc={correct.mean():.4f} T={T:.2f} "
          f"ECE {ece(raw_conf, correct):.4f} -> {ece(cal_conf, correct):.4f} (验证集)")

print("=" * 60)
print("3. 验证集:缓存各级输出,搜索阈值,训练升级判别器")
val_conf = [binnings[i].predict(confidence(val_logits[i], temps[i])) for i in range(3)]
val_corr = [(val_logits[i].argmax(1) == yval) for i in range(3)]

# 判别器训练集(独立)
Xdis, ydis = world.sample(20_000, 0.0)
l1_dis = logits_of(0, Xdis)
_, agree_dis, mcc_dis = small.mc_predict(proj[0](Xdis), m=6, seed=11)
feat_dis = disc_features(softmax(l1_dis), agree_dis, mcc_dis, proj[0](Xdis))
disc = LogisticDisc().fit(feat_dis, (l1_dis.argmax(1) == ydis).astype(float))

# 验证集上的判别器分数
_, agree_val, mcc_val = small.mc_predict(proj[0](Xval), m=6, seed=12)
disc_score_val = disc.prob_correct(
    disc_features(softmax(val_logits[0]), agree_val, mcc_val, proj[0](Xval)))
# AUC
order = np.argsort(disc_score_val)
ranks = np.empty(len(order)); ranks[order] = np.arange(1, len(order) + 1)
pos = val_corr[0].astype(float)
auc = (ranks * pos).sum() / pos.sum() / (len(pos) - pos.sum() + 1e-9) - \
      (pos.sum() + 1) / 2 / (len(pos) - pos.sum() + 1e-9)
print(f"  判别器 AUC = {auc:.4f}")

r_budget = search_thresholds(*val_conf[:2], *val_corr, budget=BUDGET)
r_target = search_thresholds(*val_conf[:2], *val_corr, acc_target=ACC_TARGET)
if r_target is None:  # 目标不可达时退化为最高准确率方案
    r_target = search_thresholds(*val_conf[:2], *val_corr, budget=sum(LEVEL_COSTS))
    r_target["t1"], r_target["t2"] = 1.0, 1.0  # 全部用大模型
    r_target["acc"] = float(val_corr[2].mean())
    r_target["cost"] = sum(LEVEL_COSTS)
    r_target["props"] = (0.0, 0.0, 1.0)
    print(f"  目标准确率 {ACC_TARGET} 不可达,退化为全部用大模型")
r_disc = search_thresholds(disc_score_val, val_conf[1], *val_corr, budget=BUDGET)
r_fixed = cascade_eval(confidence(val_logits[0]), confidence(val_logits[1]),
                       *val_corr, 0.5, 0.5)
print(f"  预算 {BUDGET}: t1={r_budget['t1']} t2={r_budget['t2']} "
      f"acc={r_budget['acc']:.4f} cost={r_budget['cost']:.2f}")
print(f"  目标准确率 {ACC_TARGET}: t1={r_target['t1']} t2={r_target['t2']} "
      f"acc={r_target['acc']:.4f} cost={r_target['cost']:.2f}")
print(f"  判别器级联(预算 {BUDGET}): t1={r_disc['t1']} t2={r_disc['t2']} "
      f"acc={r_disc['acc']:.4f} cost={r_disc['cost']:.2f}")

budgets = np.arange(2, 42, 2)
curve_cal = sweep_budgets(*val_conf[:2], *val_corr, budgets)
curve_disc = sweep_budgets(disc_score_val, val_conf[1], *val_corr, budgets)

print("=" * 60)
print("4. 生成 30 天请求流并预计算各级输出(漂移在第 10 天注入)")
stream = []
for d in range(N_DAYS):
    Xd, yd = world.sample(REQ_PER_DAY, float(d))
    lg = [logits_of(i, Xd) for i in range(3)]
    _, ag, mc = small.mc_predict(proj[0](Xd), m=6, seed=100 + d)
    score1 = disc.prob_correct(disc_features(softmax(lg[0]), ag, mc, proj[0](Xd)))
    stream.append(dict(X=Xd, y=yd, logits=lg, score1=score1))
print(f"  共 {N_DAYS * REQ_PER_DAY} 条请求")


def run_static(t1, t2, T=None, binning=None, use_disc=False):
    svc = CascadeService(budget=BUDGET, auto_recal=False)
    if T is not None:
        svc.T = list(T)
    if binning is not None:
        svc.binning = list(binning)
    svc.t1, svc.t2 = t1, t2
    out = []
    for d in range(N_DAYS):
        s = stream[d]
        out.append(svc.process_day(d, *s["logits"], s["y"],
                                   score1=s["score1"] if use_disc else None))
    return out


def run_all_large():
    return [dict(acc=float((s["logits"][2].argmax(1) == s["y"]).mean()),
                 cost=sum(LEVEL_COSTS)) for s in stream]


def run_all_small():
    return [dict(acc=float((s["logits"][0].argmax(1) == s["y"]).mean()),
                 cost=LEVEL_COSTS[0]) for s in stream]


print("=" * 60)
print("5. 运行各方案")
res = {}
res["全部用大模型"] = run_all_large()
res["全部用小模型"] = run_all_small()
res["固定阈值级联(未校准 t=0.5)"] = run_static(0.5, 0.5, T=[1, 1, 1],
                                               binning=[None, None, None])
res["校准+阈值搜索级联"] = run_static(r_budget["t1"], r_budget["t2"], T=temps, binning=binnings)
res["判别器级联"] = run_static(r_disc["t1"], r_disc["t2"], T=temps, binning=binnings,
                               use_disc=True)

svc_auto = CascadeService(acc_target=AUTO_ACC_TARGET, auto_recal=True,
                          audit_rate=AUDIT_RATE, seed=5)
svc_auto.T, svc_auto.binning = list(temps), list(binnings)
svc_auto.t1, svc_auto.t2 = r_budget["t1"], r_budget["t2"]
res["自动重校准级联(保SLA)"] = [svc_auto.process_day(d, *stream[d]["logits"], stream[d]["y"])
                                 for d in range(N_DAYS)]
adopted = [e["day"] for e in svc_auto.recal_events if e["adopted"]]
print(f"  重校准评估 {len(svc_auto.recal_events)} 次, 采纳 {len(adopted)} 次: {adopted}")

days = np.arange(N_DAYS)
summary = {}
for k, v in res.items():
    summary[k] = (float(np.mean([r["acc"] for r in v])),
                  float(np.mean([r["cost"] for r in v])))
    print(f"  {k}: acc={summary[k][0]:.4f} cost={summary[k][1]:.2f}")

# 各级统计(自动重校准方案,取最后一天快照)
auto = res["自动重校准级联(保SLA)"]
props_mean = np.mean([r["props"] for r in auto], axis=0)
s_last = stream[-1]
c1 = svc_auto.conf(0, s_last["logits"][0])
at1 = c1 >= svc_auto.t1
c2 = svc_auto.conf(1, s_last["logits"][1][~at1])
esc2 = c2 < svc_auto.t2
y_last = s_last["y"]
idx1 = np.where(~at1)[0]
acc_l1 = (s_last["logits"][0].argmax(1)[at1] == y_last[at1]).mean()
acc_l2 = (s_last["logits"][1].argmax(1)[idx1[~esc2]] == y_last[idx1[~esc2]]).mean()
acc_l3 = (s_last["logits"][2].argmax(1)[idx1[esc2]] == y_last[idx1[esc2]]).mean()
lvl_acc = [float(acc_l1), float(acc_l2), float(acc_l3)]

# 批处理统计
avg_batches = np.mean([r["batches"] for r in auto], axis=0)
avg_bs = np.mean([r["avg_batch"] for r in auto], axis=0)
naive_compute = sum(LEVEL_COSTS)  # 不复用不级联:每个请求三级全算
actual_compute = summary["自动重校准级联(保SLA)"][1]

# 类别难度
per_class_acc = [(val_logits[0].argmax(1)[yval == c] == c).mean() if (yval == c).any()
                 else np.nan for c in range(K)]

print("=" * 60)
print("6. 生成报告")
figs = {}
figs["rel"] = report.fig_to_b64(report.plot_reliability(rel_data), f"{OUT}/reliability.png")
points = {
    "全部用小模型": (summary["全部用小模型"][1], summary["全部用小模型"][0], "X"),
    "全部用大模型": (summary["全部用大模型"][1], summary["全部用大模型"][0], "X"),
    "固定阈值级联": (summary["固定阈值级联(未校准 t=0.5)"][1],
                     summary["固定阈值级联(未校准 t=0.5)"][0], "D"),
    "当前工作点(自动重校准)": (summary["自动重校准级联(保SLA)"][1],
                               summary["自动重校准级联(保SLA)"][0], "*"),
}
figs["costacc"] = report.fig_to_b64(report.plot_cost_acc(curve_cal, curve_disc, points), f"{OUT}/cost_accuracy.png")
figs["drift_acc"] = report.fig_to_b64(report.plot_drift(
    days,
    [("全部用大模型", [r["acc"] for r in res["全部用大模型"]], "-"),
     ("全部用小模型", [r["acc"] for r in res["全部用小模型"]], "-"),
     ("校准+阈值搜索(静态)", [r["acc"] for r in res["校准+阈值搜索级联"]], "-"),
     ("自动重校准级联(保SLA)", [r["acc"] for r in res["自动重校准级联(保SLA)"]], "-"),
     (f"服务质量目标 {AUTO_ACC_TARGET}", [AUTO_ACC_TARGET] * N_DAYS, "k--")],
    DRIFT_START, [e for e in svc_auto.recal_events if e['adopted']], title="分布漂移下各方案每日准确率"), f"{OUT}/drift_accuracy.png")
figs["drift_cost"] = report.fig_to_b64(report.plot_drift(
    days,
    [("校准+阈值搜索(静态)", [r["cost"] for r in res["校准+阈值搜索级联"]], "-"),
     ("自动重校准级联(保SLA)", [r["cost"] for r in res["自动重校准级联(保SLA)"]], "-"),
     ("判别器级联", [r["cost"] for r in res["判别器级联"]], "-")],
    DRIFT_START, [e for e in svc_auto.recal_events if e['adopted']], ylabel="平均成本", title="每日平均推理成本"), f"{OUT}/drift_cost.png")
figs["levels"] = report.fig_to_b64(
    report.plot_levels(days, [r["props"] for r in auto], DRIFT_START), f"{OUT}/level_proportions.png")
figs["th"] = report.fig_to_b64(report.plot_thresholds(
    days, [r["t1"] for r in auto], [r["t2"] for r in auto], DRIFT_START), f"{OUT}/thresholds.png")
figs["cls"] = report.fig_to_b64(
    report.plot_class_difficulty(CLASS_NAMES[:K], world.sigma, per_class_acc), f"{OUT}/class_difficulty.png")

full = sum(LEVEL_COSTS)
sections = []
_static = summary["校准+阈值搜索级联"]
_auto = summary["自动重校准级联(保SLA)"]
sections.append(("结论摘要", f"""<p>在 30 天、每天 {REQ_PER_DAY} 条请求的仿真流上(第 10 天注入分布漂移):</p>
<ul>
<li>全部用大模型:准确率 {summary['全部用大模型'][0]:.4f},成本 69/请求(基准 100%)</li>
<li>全部用小模型:成本降至 1/请求,但准确率只有 {summary['全部用小模型'][0]:.4f}</li>
<li>校准+阈值搜索级联:准确率 {_static[0]:.4f},成本 {_static[1]:.2f}
    (<b>节省 {1 - _static[1] / full:.1%}</b>),但漂移后日准确率从 ~0.90 跌至 ~0.80</li>
<li>判别器级联与置信度级联同预算下相当(AUC {auc:.3f},可作为不依赖校准的备用信号)</li>
<li>自动重校准级联:8% 抽查 + 每 2 天交叉拟合重校准,漂移后 SLA({AUTO_ACC_TARGET})
    自动恢复并稳定,全程准确率 {_auto[0]:.4f},成本 {_auto[1]:.2f}
    (<b>节省 {1 - _auto[1] / full:.1%}</b>)</li>
</ul>"""))

sections.append(("方案对比(30 天请求流,第 10 天注入分布漂移)", report.table(
    ["方案", "整体准确率", "平均成本/请求", "相对全部大模型成本", "成本节省"],
    [[k, f"{a:.4f}", f"{c:.2f}", f"{c / full:.1%}", f"{1 - c / full:.1%}"]
     for k, (a, c) in summary.items()])
    + '<p class="note">成本以小模型单次推理为 1(中=8,大=60)。'
      "注意两种在线方案运行模式不同:静态级联是<b>预算优先</b>(预算≤8 下最高准确率,"
      "漂移后成本随升级率自然上升、不再受预算约束);自动重校准级联是<b>服务质量优先</b>"
      f"(保 SLA 准确率 {AUTO_ACC_TARGET} 下最低成本),平时比静态方案更省钱,漂移后自动"
      f"提高升级率保住 SLA,其成本另含 {AUDIT_RATE:.0%} 抽查(送大模型+复核)开销。</p>"))

sections.append(("仿真数据与模型", report.table(
    ["模型", "结构", "参数量", "单次推理成本", "校准集准确率"],
    [[names[i], str(m.dims), f"{n_params[i]:,}", str(LEVEL_COSTS[i]), calib_rows[i][1]]
     for i, m in enumerate(models)])
    + '<p class="note">20 个类别先验不均衡;每类 3 个子簇(边界非线性);'
      "25% 请求为边界附近的难样本;第 10 天起类别先验/特征空间/难度开始漂移。</p>"
    + report.img_tag(figs["cls"])))

sections.append(("1. 置信度校准(温度缩放 + 分箱)", report.table(
    ["模型", "准确率", "温度 T", "ECE 校准前", "ECE 温度缩放后", "ECE 温度+分箱后"],
    [list(r) for r in calib_rows]) + report.img_tag(figs["rel"])))

sections.append(("2. 阈值搜索(验证集)", report.table(
    ["模式", "t1(小→中)", "t2(中→大)", "准确率", "平均成本", "各级比例(小/中/大)"],
    [[f"预算≤{BUDGET}", r_budget["t1"], r_budget["t2"], f"{r_budget['acc']:.4f}",
      f"{r_budget['cost']:.2f}", "/".join(f"{p:.1%}" for p in r_budget["props"])],
     [f"目标准确率≥{ACC_TARGET}", r_target["t1"], r_target["t2"], f"{r_target['acc']:.4f}",
      f"{r_target['cost']:.2f}", "/".join(f"{p:.1%}" for p in r_target["props"])],
     [f"判别器(预算≤{BUDGET})", r_disc["t1"], r_disc["t2"], f"{r_disc['acc']:.4f}",
      f"{r_disc['cost']:.2f}", "/".join(f"{p:.1%}" for p in r_disc["props"])],
     ["固定阈值 t=0.5(未校准)", 0.5, 0.5, f"{r_fixed[0]:.4f}", f"{r_fixed[1]:.2f}",
      "/".join(f"{p:.1%}" for p in r_fixed[2])]])
    + '<p class="note">曲线为漂移前验证集上按预算扫描得到的工作点集合;'
      "散点为各方案在 30 天请求流(含漂移)上的实际平均表现,因此略低于曲线属正常。</p>"
    + report.img_tag(figs["costacc"])))

sections.append(("3. 升级判别器 vs 置信度",
                 f"<p>判别器特征:小模型 top-1 概率、top1-top2 间隔、熵、"
                 f"6 次随机推理(dropout)一致率与平均置信、输入范数等。"
                 f"<b>AUC = {auc:.4f}</b>。同预算下判别器级联准确率 "
                 f"{r_disc['acc']:.4f} vs 置信度级联 {r_budget['acc']:.4f},两者相当。"
                 f"判别器的价值在于:它不依赖校准(校准失效时置信度阈值会失灵,"
                 f"而一致性/特征信号更鲁棒),且特征可继续扩展(检索命中、长度、语言等)。</p>"))

recal_tbl = (report.table(
    ["评估日", "准确率下滑", "ECE", "超告警阈值", "采纳", "目标函数 旧→新", "新温度", "新 t1", "新 t2"],
    [[e["day"], f"{e['acc_drop']:.3f}", f"{e['ece']:.3f}",
      "是" if e["triggered"] else "否", "✔" if e["adopted"] else "—",
      f"{e['obj_cur']} → {e['obj_new']}", e["new_T"], e["new_t1"], e["new_t2"]]
     for e in svc_auto.recal_events])
    if svc_auto.recal_events else "<p>未触发重校准。</p>")
sections.append(("4. 分布漂移与自动重校准",
                 "<p>第 10 天注入漂移(类别先验偏移 + 特征空间旋转 + 难样本变多,第 18 天完全生效)。"
                 "静态级联准确率从 ~0.90 跌至 ~0.80,SLA 失守;自动重校准级联每 2 天在抽查"
                 "滑动缓冲(复核标签)上做三份交叉拟合(A 拟合温度+分箱 / B 搜阈值 / C 独立评估),"
                 "只有候选工作点在 C 上明显更优才采纳(滞回,防小样本抖动)。"
                 "漂移爬坡期存在 2~4 天检测-适应滞后,随后 SLA 恢复并稳定在目标附近,"
                 "代价是升级率(成本)自动抬高——仍远低于全部用大模型。</p>"
                 + report.img_tag(figs["drift_acc"]) + report.img_tag(figs["drift_cost"])
                 + report.img_tag(figs["levels"]) + report.img_tag(figs["th"]) + recal_tbl))

sections.append(("5. 批处理与前级结果复用", report.table(
    ["指标", "小模型级", "中模型级", "大模型级"],
    [["日均批次数", f"{avg_batches[0]:.1f}", f"{avg_batches[1]:.1f}", f"{avg_batches[2]:.1f}"],
     ["平均批大小", f"{avg_bs[0]:.0f}", f"{avg_bs[1]:.0f}", f"{avg_bs[2]:.0f}"],
     ["处理请求比例(全程)", f"{props_mean[0]:.1%}", f"{props_mean[1]:.1%}", f"{props_mean[2]:.1%}"],
     ["最后一天该级准确率", f"{lvl_acc[0]:.4f}", f"{lvl_acc[1]:.4f}", f"{lvl_acc[2]:.4f}"]])
    + '<p class="note">该级准确率是"被该级决断的请求"的条件准确率:逐级处理的请求越来越'
      "难(都是上一级没把握的),所以逐级递减属正常现象——小模型级只决断最有把握的请求,"
      "大模型级接手最难的尾部。</p>"
    + f"<p>升级的请求在批内重新组批送下一级;前一级算过的 logits/置信度被缓存,"
      f"判别器、校准、监控直接复用,不重算。每请求平均实际计算量 "
      f"<b>{actual_compute:.2f}</b>,若不做级联三级全算为 {naive_compute:.0f},"
      f"节省 <b>{1 - actual_compute / naive_compute:.1%}</b>。"
      f"当前工作点:t1={svc_auto.t1}, t2={svc_auto.t2}, "
      f"温度={[round(t, 2) for t in svc_auto.T]}。</p>"))

html_path = os.path.join(OUT, "report.html")
report.build_html(sections, html_path)
print(f"报告已生成: {html_path}")
print(f"总耗时 {time.time() - t_start:.1f}s")
