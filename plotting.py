"""
plotting.py — 8 张图，每图一函数。
已修改：支持返回 fig 对象供 Streamlit 渲染。
"""

import os
import numpy as np
import pandas as pd
import matplotlib
matplotlib.use("Agg")
import matplotlib.pyplot as plt
from scipy.interpolate import make_interp_spline

plt.rcParams["font.sans-serif"] = ["Microsoft YaHei", "SimHei", "SimSun", "DejaVu Sans"]
plt.rcParams["font.family"] = "sans-serif"
plt.rcParams["axes.unicode_minus"] = False

COLORS = {"A1": "#7f7f7f", "A2": "#e41a1c", "A3": "#377eb8", "A4": "#4daf4a"}
LABELS = {"A1": "A1 纯BESS", "A2": "A2 无序充电", "A3": "A3 智能充电", "A4": "A4 V2G"}

# ---------- 图1：四情景每日成本 ----------
def plot_daily_cost(daily_df, out_dir, active_a="A4"):
    fig, ax = plt.subplots(figsize=(12, 5))
    for col in daily_df.columns:
        if col not in COLORS: continue
        ax.plot(daily_df.index + 1, daily_df[col], "-o", markersize=4,
                color=COLORS[col], lw=1.8, label=LABELS[col])
    ax.set_xlabel("天"); ax.set_ylabel("日总成本 (元)")
    ax.set_title("四情景每日成本对比")
    ax.grid(alpha=0.3); ax.legend()
    plt.tight_layout()
    if out_dir: plt.savefig(os.path.join(out_dir, "图1_每日成本对比.png"), dpi=150)
    return fig

# ---------- 图2：累计成本 ----------
def plot_cumulative_cost(daily_df, out_dir):
    fig, ax = plt.subplots(figsize=(12, 5))
    for col in daily_df.columns:
        if col not in COLORS: continue
        ax.plot(daily_df.index + 1, np.cumsum(daily_df[col]), "-",
                color=COLORS[col], lw=2, label=LABELS[col])
    ax.set_xlabel("天"); ax.set_ylabel("累计成本 (元)")
    ax.set_title("累计成本对比")
    ax.grid(alpha=0.3); ax.legend()
    plt.tight_layout()
    if out_dir: plt.savefig(os.path.join(out_dir, "图2_累计成本.png"), dpi=150)
    return fig

# ---------- 图3：典型日调度曲线 ----------
def plot_typical_day_dispatch(day_data, out_dir, title_suffix=""):
    fig, axes = plt.subplots(2, 1, figsize=(13, 8))
    t = np.arange(48) / 2.0
    ax = axes[0]
    ax.plot(t, day_data["load"], "k-", lw=1.8, label="负荷")
    ax.plot(t, day_data["pv"], "orange", lw=1.8, label="光伏")
    ax.plot(t, day_data["res"]["pg"], "b-", lw=1.6, label="购电")
    ax.plot(t, -day_data["res"]["ps"], "c-", lw=1.4, label="售电(反送)")
    ax.plot(t, day_data["res"]["pc_b"], "g--", lw=1.2, label="BESS充电")
    ax.plot(t, day_data["res"]["pd_b"], "r--", lw=1.2, label="BESS放电")
    ax.set_ylabel("功率 kW"); ax.grid(alpha=0.3); ax.legend(ncol=3, fontsize=8)
    ax.set_title(f"典型日调度曲线 {title_suffix}")

    ax = axes[1]
    ax.plot(t, day_data["res"]["eb_b"][1:], "m-", lw=1.8, label="BESS SOC")
    ax.axhline(10800, color="gray", ls=":"); ax.axhline(1200, color="gray", ls=":")
    ax.axhline(6000, color="blue", ls="--", label="初始")
    ax.set_xlabel("时间 (小时)"); ax.set_ylabel("储能 kWh")
    ax.grid(alpha=0.3); ax.legend()
    plt.tight_layout()
    if out_dir: plt.savefig(os.path.join(out_dir, f"图3_典型日调度{title_suffix}.png"), dpi=150)
    return fig

# ---------- 图4：EV 充放电对比 ----------
def plot_ev_charge_discharge(ev_curves, out_dir):
    fig, ax = plt.subplots(figsize=(12, 5))
    t = np.arange(48) / 2.0
    colors = ["#e41a1c", "#377eb8", "#4daf4a", "#984ea3"]
    for (label, (chg, dis)), c in zip(ev_curves.items(), colors):
        ax.plot(t, chg, color=c, lw=2, label=f"{label} 充电")
        if np.any(dis > 1e-3):
            ax.plot(t, -dis, color=c, lw=2, ls="--", label=f"{label} 放电")
    ax.axhline(0, color="k", lw=0.8)
    ax.set_xlabel("时间 (小时)"); ax.set_ylabel("EV 充放电功率 (kW)")
    ax.set_title("EV 充电行为对比")
    ax.grid(alpha=0.3); ax.legend(fontsize=9)
    plt.tight_layout()
    if out_dir: plt.savefig(os.path.join(out_dir, "图4_EV充电行为对比.png"), dpi=150)
    return fig

# ---------- 图5：电网净负荷冲击 ----------
def plot_grid_impact(impact_df, out_dir):
    metric_map = [
        ("peak_kw",     "峰值 (kW)"), ("valley_kw",   "谷值 (kW)"),
        ("p2v_kw",      "峰谷差 (kW)"), ("std_kw",      "标准差 (kW)"),
        ("peak_h_kw",   "晚高峰均值 (kW)"), ("valley_h_kw", "凌晨均值 (kW)"),
    ]
    fig, axes = plt.subplots(2, 3, figsize=(15, 8))
    axes = axes.flatten()
    for ax, (col, title) in zip(axes, metric_map):
        if col in impact_df.columns:
            bars = ax.bar(impact_df.index, impact_df[col], color=[COLORS.get(i, "#888") for i in impact_df.index])
            for b, v in zip(bars, impact_df[col]):
                ax.text(b.get_x() + b.get_width() / 2, v, f"{v:.0f}", ha="center", va="bottom", fontsize=9)
            ax.set_title(title); ax.grid(axis="y", alpha=0.3)
        else:
            ax.set_title(f"{title}\n(列 '{col}' 不存在)")
    plt.suptitle("电网净负荷冲击指标对比", fontsize=14)
    plt.tight_layout()
    if out_dir: plt.savefig(os.path.join(out_dir, "图5_电网冲击对比.png"), dpi=150)
    return fig

# ---------- 图6：收益分配饼图 ----------
def plot_revenue_share(share_df, total_gain, out_dir):
    fig, ax = plt.subplots(figsize=(7, 7))
    ax.pie(share_df["比例"], labels=share_df["角色"],
           autopct=lambda p: f"{p:.0f}%\n({p * total_gain / 100:.0f}元)",
           colors=["#4daf4a", "#377eb8", "#ff7f00"], startangle=90, wedgeprops={"edgecolor": "k"})
    ax.set_title(f"V2G 收益分配 (总收益 {total_gain:.0f} 元)\n电网不参与套利分红")
    plt.tight_layout()
    if out_dir: plt.savefig(os.path.join(out_dir, "图6_收益分配饼图.png"), dpi=150)
    return fig

# ---------- 图7：CVaR 与期望成本权衡 ----------
def plot_cvar_tradeoff(lam_list, mean_costs, cvar_costs, out_dir):
    fig, ax = plt.subplots(figsize=(9, 5.5))
    ax.plot(lam_list, mean_costs, "-o", color="#377eb8", lw=2, label="期望成本")
    ax.plot(lam_list, cvar_costs, "-s", color="#e41a1c", lw=2, label="CVaR 成本")
    ax.set_xlabel("CVaR 权重 λ"); ax.set_ylabel("成本 (元)")
    ax.set_title("CVaR 与期望成本权衡"); ax.grid(alpha=0.3); ax.legend()
    plt.tight_layout()
    if out_dir: plt.savefig(os.path.join(out_dir, "图7_CVaR权衡.png"), dpi=150)
    return fig

# ---------- 图8：EV 规模扫描 ----------
def plot_ev_sweep(sweep_df, out_dir):
    fig, axes = plt.subplots(1, 3, figsize=(16, 4.5))
    def smooth_plot(ax, x, y, color, marker, label):
        ax.plot(x, y, marker, color=color, markersize=6, label=f"{label}(实际点)")
        if len(x) >= 3:
            x_smooth = np.linspace(x.min(), x.max(), 300)
            spl = make_interp_spline(x, y, k=2)
            y_smooth = spl(x_smooth); y_smooth = np.clip(y_smooth, 0, None)
            ax.plot(x_smooth, y_smooth, "-", color=color, lw=2)
        else: ax.plot(x, y, "-", color=color, lw=2)

    if "total_cost" in sweep_df.columns:
        smooth_plot(axes[0], sweep_df["n_ev"].values, sweep_df["total_cost"].values, "#4daf4a", "o", "总成本")
        axes[0].set_xlabel("EV 数量"); axes[0].set_ylabel("总成本 (元)"); axes[0].set_title("EV 规模 vs 总成本"); axes[0].grid(alpha=0.3); axes[0].legend()
    if "per_ev" in sweep_df.columns:
        smooth_plot(axes[1], sweep_df["n_ev"].values, sweep_df["per_ev"].values, "#377eb8", "s", "单车日均")
        axes[1].set_xlabel("EV 数量"); axes[1].set_ylabel("单车日均贡献 (元)"); axes[1].set_title("边际效益"); axes[1].grid(alpha=0.3); axes[1].legend()
    if "v2g_gain" in sweep_df.columns:
        smooth_plot(axes[2], sweep_df["n_ev"].values, sweep_df["v2g_gain"].values, "#e41a1c", "^", "V2G增益")
        axes[2].set_xlabel("EV 数量"); axes[2].set_ylabel("V2G 增益 (元)"); axes[2].set_title("V2G 总增益"); axes[2].grid(alpha=0.3); axes[2].legend()
    plt.tight_layout()
    if out_dir: plt.savefig(os.path.join(out_dir, "图8_EV规模扫描.png"), dpi=150)
    return fig
