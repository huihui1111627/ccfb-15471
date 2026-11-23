"""绘图 + HTML 报告生成。"""
import base64
import io

import matplotlib
matplotlib.use("Agg")
import matplotlib.pyplot as plt
import numpy as np

plt.rcParams["font.sans-serif"] = ["PingFang SC", "Hiragino Sans GB", "Arial Unicode MS", "sans-serif"]
plt.rcParams["axes.unicode_minus"] = False


def fig_to_b64(fig, path=None):
    if path:
        fig.savefig(path, dpi=110, bbox_inches="tight")
    buf = io.BytesIO()
    fig.savefig(buf, format="png", dpi=110, bbox_inches="tight")
    plt.close(fig)
    return base64.b64encode(buf.getvalue()).decode()


def plot_reliability(models_rel):
    """models_rel: [(name, (centers_raw, accs_raw), (centers_cal, accs_cal)), ...]"""
    fig, axes = plt.subplots(1, 3, figsize=(13, 4))
    for ax, (name, raw, cal) in zip(axes, models_rel):
        ax.plot([0, 1], [0, 1], "k--", lw=1, label="理想校准")
        ax.plot(raw[0], raw[1], "o-", color="#d62728", label="校准前")
        ax.plot(cal[0], cal[1], "s-", color="#1f77b4", label="温度+分箱校准后")
        ax.set_title(name)
        ax.set_xlabel("预测置信度")
        ax.set_ylabel("实际准确率")
        ax.legend(fontsize=8)
        ax.grid(alpha=0.3)
    fig.suptitle("可靠性曲线(校准集)")
    return fig


def plot_cost_acc(curve_cal, curve_disc, points):
    """curve_*: [(budget, acc, cost, ...)]; points: {name: (cost, acc, marker)}"""
    fig, ax = plt.subplots(figsize=(7.5, 5))
    ax.plot([c[2] for c in curve_cal], [c[1] for c in curve_cal], "o-",
            color="#1f77b4", label="校准+阈值搜索级联")
    ax.plot([c[2] for c in curve_disc], [c[1] for c in curve_disc], "^--",
            color="#2ca02c", label="判别器级联")
    for name, (cost, acc, marker) in points.items():
        ax.scatter([cost], [acc], marker=marker, s=110, zorder=5, label=name)
    ax.set_xlabel("平均成本 / 请求(相对小模型=1)")
    ax.set_ylabel("准确率")
    ax.set_title("成本 - 准确率曲线与当前工作点")
    ax.legend(fontsize=8)
    ax.grid(alpha=0.3)
    return fig


def plot_drift(days, series, drift_start, recal_events, ylabel="准确率", title=""):
    fig, ax = plt.subplots(figsize=(9, 4.5))
    for name, vals, style in series:
        ax.plot(days, vals, style, label=name, lw=1.6)
    ax.axvline(drift_start, color="red", ls="--", alpha=0.6, label="注入分布漂移")
    for ev in recal_events:
        ax.axvline(ev["day"], color="orange", ls=":", alpha=0.8)
    if recal_events:
        ax.plot([], [], ":", color="orange", label="自动重校准触发")
    ax.set_xlabel("天")
    ax.set_ylabel(ylabel)
    ax.set_title(title)
    ax.legend(fontsize=8)
    ax.grid(alpha=0.3)
    return fig


def plot_levels(days, props, drift_start):
    props = np.array(props)  # (days, 3)
    fig, ax = plt.subplots(figsize=(9, 4))
    ax.stackplot(days, props[:, 0], props[:, 1], props[:, 2],
                 labels=["小模型", "中模型", "大模型"],
                 colors=["#9ecae1", "#fdae6b", "#f4a3a3"])
    ax.axvline(drift_start, color="red", ls="--", alpha=0.6)
    ax.set_xlabel("天")
    ax.set_ylabel("请求比例")
    ax.set_title("各级处理的请求比例(自动重校准级联)")
    ax.legend(loc="upper right", fontsize=8)
    ax.set_ylim(0, 1)
    return fig


def plot_thresholds(days, t1s, t2s, drift_start):
    fig, ax = plt.subplots(figsize=(9, 3.6))
    ax.plot(days, t1s, "o-", ms=3, label="t1(小模型升级阈值)")
    ax.plot(days, t2s, "s-", ms=3, label="t2(中模型升级阈值)")
    ax.axvline(drift_start, color="red", ls="--", alpha=0.6)
    ax.set_xlabel("天")
    ax.set_ylabel("阈值")
    ax.set_title("自动重校准下的阈值轨迹")
    ax.legend(fontsize=8)
    ax.grid(alpha=0.3)
    return fig


def plot_class_difficulty(names, sigma, acc_small):
    order = np.argsort(sigma)
    fig, ax = plt.subplots(figsize=(9, 4.5))
    x = np.arange(len(names))
    ax.bar(x - 0.2, np.array(sigma)[order], width=0.4, label="类内噪声 σ(难度)")
    ax.bar(x + 0.2, np.array(acc_small)[order], width=0.4, label="小模型类别准确率")
    ax.set_xticks(x)
    ax.set_xticklabels([names[i] for i in order], rotation=45, ha="right", fontsize=8)
    ax.set_title("不同类别难度不同")
    ax.legend(fontsize=8)
    ax.grid(alpha=0.3, axis="y")
    return fig


def img_tag(b64):
    return f'<img src="data:image/png;base64,{b64}" style="max-width:100%;border:1px solid #ddd;border-radius:6px;margin:8px 0">'


def table(headers, rows):
    h = "".join(f"<th>{c}</th>" for c in headers)
    rs = ""
    for r in rows:
        rs += "<tr>" + "".join(f"<td>{c}</td>" for c in r) + "</tr>"
    return f'<table><thead><tr>{h}</tr></thead><tbody>{rs}</tbody></table>'


def build_html(sections, path):
    css = """
    body{font-family:-apple-system,'PingFang SC',sans-serif;max-width:1080px;margin:24px auto;padding:0 16px;color:#222}
    h1{border-bottom:3px solid #1f77b4;padding-bottom:8px}
    h2{color:#1f77b4;margin-top:36px}
    table{border-collapse:collapse;margin:10px 0;font-size:14px}
    th,td{border:1px solid #ccc;padding:5px 12px;text-align:right}
    th{background:#f0f6fb}
    td:first-child{text-align:left}
    .note{color:#666;font-size:13px}
    """
    body = "".join(f"<h2>{t}</h2>{c}" for t, c in sections)
    html = f"""<!DOCTYPE html><html lang="zh"><head><meta charset="utf-8">
<title>多模型级联推理仿真报告</title><style>{css}</style></head>
<body><h1>多模型级联推理服务 · 仿真实验报告</h1>{body}</body></html>"""
    with open(path, "w", encoding="utf-8") as f:
        f.write(html)
