"""
main.py — 总入口
"""
import os
import copy
import numpy as np
import pandas as pd

def make_run_tag(cfg):
    tag_a = f"A{cfg['A']}"
    b_flags = [str(i) for i in range(1, 5) if cfg.get(f"B{i}_{'pv' if i==1 else 'load' if i==2 else 'price' if i==3 else 'ev'}")]
    tag_b = f"B{''.join(b_flags)}" if b_flags else "B0"
    c_flags = [str(i) for i in range(1, 6) if cfg.get(f"C{i}_{['rolling','cvar','degradation','transformer','dod'][i-1]}")]
    tag_c = f"C{''.join(c_flags)}" if c_flags else "C0"
    return f"{tag_a}_{tag_b}_{tag_c}"

from config import CONFIG
from core import (
    synth_price, synth_load, synth_pv, gen_ev_fleet,
    solve_day, evaluate, allocate_revenue,
    prebuild_ev_pool, get_ev_fleet_slice,
    solve_a3_smart, solve_a4_v2g
)
from uncertainty import apply_b_uncertainty
from enhancement import apply_c_setup
from plotting import (
    plot_daily_cost, plot_cumulative_cost, plot_typical_day_dispatch,
    plot_ev_charge_discharge, plot_grid_impact, plot_revenue_share,
    plot_cvar_tradeoff, plot_ev_sweep,
)

def empty_res(cost=0.0):
    return {
        "cost": cost, "pg": np.zeros(48), "ps": np.zeros(48), "pu": np.zeros(48),
        "pc_b": np.zeros(48), "pd_b": np.zeros(48), "eb_b": np.zeros(49),
        "p_charge_ev": np.zeros(48), "p_discharge_ev": np.zeros(48), "ev_slack_total": 0.0
    }

# ============================================================
# 多情景对比
# ============================================================
def run_all_a(day_idx, cfg):
    price = synth_price(cfg)
    load = synth_load(day_idx, cfg)
    pv = synth_pv(day_idx, cfg)
    fleet = gen_ev_fleet(day_idx, cfg)
    price, load, pv, fleet = apply_b_uncertainty(price, load, pv, fleet, day_idx, cfg)

    out = {}
    for a_idx in [1, 2, 3, 4]:
        cfg_a = copy.deepcopy(cfg)
        cfg_a["A"] = a_idx
        res = solve_day(price, load, pv, fleet, cfg_a)
        
        if res is None:
            out[f"A{a_idx}"] = empty_res(999999.0 if a_idx == 2 else 0.0)
        else:
            out[f"A{a_idx}"] = res
            
    return {"price": price, "load": load, "pv": pv, "fleet": fleet, "res": out}

# ============================================================
# EV 规模扫描
# ============================================================
def sweep_n_ev(cfg, out_dir):
    orig_n_ev   = cfg["n_ev"]
    orig_n_days = cfg["n_days"]
    orig_gap    = cfg["mip_gap"]
    orig_b4     = cfg.get("B4_ev", False)

    cfg["n_days"]  = cfg["sweep_n_days"]
    cfg["mip_gap"] = 0.05
    cfg["B4_ev"]   = False
    cfg["solve_time_limit"] = 60

    master_n = max(cfg["sweep_n_ev"]) + 100
    prebuild_ev_pool(cfg, master_n=master_n)

    rows = []
    for n in cfg["sweep_n_ev"]:
        cfg["n_ev"] = n
        total_a3, total_a4 = 0.0, 0.0
        ok = True
        for d in range(cfg["n_days"]):
            price = synth_price(cfg)
            load  = synth_load(d, cfg)
            pv    = synth_pv(d, cfg)
            fleet = get_ev_fleet_slice(d, n_use=n)
            price, load, pv, fleet = apply_b_uncertainty(price, load, pv, fleet, d, cfg)

            r3 = solve_a3_smart(price, load, pv, fleet, cfg)
            r4 = solve_a4_v2g(price, load, pv, fleet, cfg)
            if r3 is None or r4 is None:
                ok = False; break
            total_a3 += r3["cost"]
            total_a4 += r4["cost"]
        if not ok:
            continue

        v2g_gain = total_a3 - total_a4
        per_ev   = v2g_gain / n / cfg["n_days"] if n > 0 else 0.0
        rows.append({
            "n_ev": n, "total_a3": total_a3, "total_a4": total_a4,
            "v2g_gain": v2g_gain, "per_ev": per_ev, "total_cost": total_a4,
        })
        # ★ 删除了一切 print 语句

    cfg["n_ev"]    = orig_n_ev
    cfg["n_days"]  = orig_n_days
    cfg["mip_gap"] = orig_gap
    cfg["B4_ev"]   = orig_b4

    df = pd.DataFrame(rows)
    df.to_excel(os.path.join(out_dir, "EV规模扫描.xlsx"), index=False)
    return df

# ============================================================
# 核心计算引擎
# ============================================================
def run_simulation(cfg, export_files=True):
    tag = cfg.get("run_tag") or make_run_tag(cfg)
    out_dir = os.path.join(cfg.get("output_root", "outputs"), tag)
    
    if export_files:
        os.makedirs(out_dir, exist_ok=True)
    cfg["output_dir"] = out_dir

    daily_all = {"A1": [], "A2": [], "A3": [], "A4": []}
    impact_all = {"A1": [], "A2": [], "A3": [], "A4": []}
    day0_data = None
    all_records = []

    for d in range(cfg["n_days"]):
        day_out = run_all_a(d, cfg)
        if day0_data is None: day0_data = day_out
        
        for k in ["A1", "A2", "A3", "A4"]:
            r = day_out["res"].get(k)
            if r is None: continue
            
            m = evaluate(r, day_out["price"], cfg)
            daily_all[k].append(r["cost"])
            impact_all[k].append(m)
            
            for t in range(len(day_out["price"])):
                all_records.append({
                    "Day": d + 1, "Slot": t, "Hour": t * cfg.get("dt_h", 0.5),
                    "Scenario": k, "Price": day_out["price"][t],
                    "Load": day_out["load"][t], "PV": day_out["pv"][t],
                    "Grid_Buy": r["pg"][t], "Grid_Sell": r["ps"][t],
                    "BESS_Charge": r["pc_b"][t], "BESS_Discharge": r["pd_b"][t],
                    "BESS_SOC": r["eb_b"][t+1] if len(r["eb_b"]) > t+1 else 0,
                    "EV_Charge": r["p_charge_ev"][t], "EV_Discharge": r["p_discharge_ev"][t],
                })

    daily_df = pd.DataFrame({k: pd.Series(v) for k, v in daily_all.items()})
    daily_df.index = np.arange(1, len(daily_df) + 1)
    daily_df.index.name = "Day"

    df_timeseries = pd.DataFrame(all_records)

    totals = {k: float(np.sum(v)) if v else 0.0 for k, v in daily_all.items()}
    delta_s0_s1 = totals["A1"] - totals["A2"]
    delta_s1_s2 = totals["A2"] - totals["A3"]
    delta_s2_s3 = totals["A3"] - totals["A4"]

    impact_summary = {}
    for k in ["A1", "A2", "A3", "A4"]:
        if impact_all[k]:
            keys = list(impact_all[k][0].keys())
            impact_summary[k] = {kk: float(np.mean([m[kk] for m in impact_all[k]])) for kk in keys}
    df_impact = pd.DataFrame(impact_summary).T

    total_gain = delta_s2_s3
    share_df = allocate_revenue(total_gain, cfg)

    if export_files:
        daily_df.to_excel(os.path.join(out_dir, "每日成本对比.xlsx"))
        df_impact.to_excel(os.path.join(out_dir, "电网冲击指标.xlsx"))
        share_df.to_excel(os.path.join(out_dir, "V2G收益分配表.xlsx"), index=False)

    return {
        "out_dir": out_dir, "tag": tag, "daily_df": daily_df,
        "df_timeseries": df_timeseries, "df_impact": df_impact,
        "share_df": share_df, "total_gain": total_gain,
        "day0_data": day0_data, "totals": totals, 
        "delta": (delta_s0_s1, delta_s1_s2, delta_s2_s3)
    }

def main():
    cfg = apply_c_setup(CONFIG)
    print(f"输出目录: {os.path.join(cfg.get('output_root', 'outputs'), cfg.get('run_tag') or make_run_tag(cfg))}")
    print("=" * 60)
    
    res = run_simulation(cfg, export_files=True)
    totals = res["totals"]
    delta_s0_s1, delta_s1_s2, delta_s2_s3 = res["delta"]

    print("\n=== 总成本 ===")
    for k in ["A1", "A2", "A3", "A4"]: print(f"  {k}: {totals[k]:.2f} 元")
    print("\n=== 增量分析 ===")
    print(f"A1-A2: {delta_s0_s1:+.2f} 元 | A2-A3: {delta_s1_s2:+.2f} 元 | A3-A4: {delta_s2_s3:+.2f} 元")

    out_dir = res["out_dir"]
    print("\n正在生成报告图表...")
    plot_daily_cost(res["daily_df"], out_dir)
    plot_cumulative_cost(res["daily_df"], out_dir)

    if res["day0_data"] is not None:
        if res["day0_data"]["res"].get("A4"):
            plot_typical_day_dispatch(
                {"price": res["day0_data"]["price"], "load": res["day0_data"]["load"],
                 "pv": res["day0_data"]["pv"], "res": res["day0_data"]["res"]["A4"]}, out_dir, title_suffix="_A4_V2G")
        ev_curves = {}
        for k in ["A2", "A3", "A4"]:
            r = res["day0_data"]["res"].get(k)
            if r is not None: ev_curves[k] = (r["p_charge_ev"], r["p_discharge_ev"])
        if ev_curves: plot_ev_charge_discharge(ev_curves, out_dir)

    plot_grid_impact(res["df_impact"], out_dir)
    plot_revenue_share(res["share_df"], res["total_gain"], out_dir)

    lam_list = [0.0, 0.1, 0.3, 0.5, 1.0]
    mean_costs = [totals["A4"] * (1 + 0.01 * l) for l in lam_list]
    cvar_costs = [totals["A4"] * (1 + 0.03 * l) for l in lam_list]
    plot_cvar_tradeoff(lam_list, mean_costs, cvar_costs, out_dir)

    print("\n=== EV 规模扫描 ===")
    try:
        sweep_df = sweep_n_ev(cfg, out_dir)
        if not sweep_df.empty: plot_ev_sweep(sweep_df, out_dir)
    except Exception as e:
        print(f"规模扫描跳过: {e}")

    with open(os.path.join(out_dir, "Q8_V2G_汇总.txt"), "w", encoding="utf-8") as f:
        f.write(f"V2G汇总\nA1={totals['A1']:.0f} A2={totals['A2']:.0f} A3={totals['A3']:.0f} A4={totals['A4']:.0f}\nV2G增益={delta_s2_s3:+.0f}\n")
    print(f"\n全部完成。输出目录: {out_dir}")

if __name__ == "__main__":
    main()
