"""
core.py — 数据生成 + EV 生成 + A 组四个 MILP 求解函数 + 评估器 + 规模扫描池
"""

import os
import numpy as np
import pandas as pd
import pulp

# ============================================================
# 数据生成
# ============================================================
def synth_price(cfg):
    arr = np.zeros(48)
    for t in range(48):
        h = t // 2
        if h < 8: arr[t] = 0.30
        elif h < 17: arr[t] = 0.70
        elif h < 22: arr[t] = 1.30
        else: arr[t] = 0.70
    return arr * cfg.get("price_scale", 1.0)

def synth_load(day_idx, cfg):
    rng = np.random.default_rng(cfg["seed"] + day_idx * 31)
    t = np.arange(48)
    h = t / 2.0
    p1 = 2000.0 * np.exp(-((h - 9) ** 2) / 2.0)
    p2 = 3000.0 * np.exp(-((h - 19) ** 2) / 3.0)
    arr = 4000.0 + p1 + p2 + rng.normal(0, 100, 48)
    return np.clip(arr * cfg.get("load_scale", 1.0), 0, None)

def synth_pv(day_idx, cfg):
    rng = np.random.default_rng(cfg["seed"] + day_idx * 37 + 1)
    t = np.arange(48)
    h = t / 2.0
    season = 0.7 + 0.3 * np.sin(2 * np.pi * (day_idx % 365 - 80) / 365)
    pv = 5000.0 * season * np.sin(np.pi * (h - 6) / 12)
    pv = np.where((h >= 6) & (h <= 18), pv, 0.0)
    arr = pv + rng.normal(0, 50, 48)
    return np.clip(arr * cfg.get("pv_scale", 1.0), 0, None)

# ============================================================
# EV 车队生成
# ============================================================
def gen_ev_fleet(day_idx, cfg):
    n_ev = cfg["n_ev"]
    rng = np.random.default_rng(cfg["seed"] + day_idx * 17)
    peaks = cfg["ev_arr_peaks"]
    w = np.array([p[2] for p in peaks]); w = w / w.sum()
    which = rng.choice(len(peaks), size=n_ev, p=w)
    arrivals = np.array([rng.normal(peaks[k][0], peaks[k][1]) for k in which])
    arrivals = np.clip(arrivals, 0.0, 22.0)
    dwell = np.clip(rng.normal(cfg["ev_dwell_mean_h"], cfg["ev_dwell_std_h"], n_ev), cfg["ev_dwell_min_h"], cfg["ev_dwell_max_h"])
    departures = np.clip(arrivals + dwell, 0.0, 23.9)
    will = rng.beta(cfg["ev_will_alpha"], cfg["ev_will_beta"], n_ev)
    e_cap = rng.choice(cfg["ev_cap_choices"], size=n_ev, p=cfg["ev_cap_probs"]).astype(float)
    soc_arr = np.clip(rng.normal(cfg["ev_soc_arr_mean"], cfg["ev_soc_arr_std"], n_ev), 0.10, 0.60)
    soc_dep = np.clip(rng.normal(cfg["ev_soc_dep_mean"], cfg["ev_soc_dep_std"], n_ev), 0.70, 0.95)
    t_arr = np.clip((arrivals * 2).astype(int), 0, 47)
    t_dep = np.clip((departures * 2).astype(int), 1, 48)

    for i in range(n_ev):
        dt_h = cfg["dt_h"]
        max_kwh = (cfg["ev_p_max"] * cfg["ev_eta_c"] * (t_dep[i] - t_arr[i]) * dt_h * max(will[i], 0.1))
        max_soc = soc_arr[i] + max_kwh / e_cap[i]
        soc_dep[i] = min(soc_dep[i], max_soc)
        soc_dep[i] = max(soc_dep[i], soc_arr[i])
        if t_dep[i] <= t_arr[i]:
            t_arr[i], t_dep[i] = 0, 1
            soc_arr[i] = soc_dep[i] = 0.5

    return {
        "n_ev": n_ev, "e_cap": e_cap, "soc_arr": soc_arr, "soc_dep": soc_dep,
        "will": will, "t_arr": t_arr, "t_dep": t_dep,
        "p_chg_max": np.full(n_ev, cfg["ev_p_max"]), "p_dis_max": np.full(n_ev, cfg["ev_p_max"]),
    }

def unordered_charge_profile(fleet, dt_h):
    profile = np.zeros(48)
    if fleet is None: return profile
    for i in range(fleet["n_ev"]):
        need = fleet["e_cap"][i] * (fleet["soc_dep"][i] - fleet["soc_arr"][i])
        if need <= 1e-6: continue
        p_eff = fleet["p_chg_max"][i] * max(fleet["will"][i], 0.1)
        per_slot = p_eff * 0.9 * dt_h
        if per_slot <= 1e-6: continue
        n_slots = int(np.ceil(need / per_slot))
        end = min(fleet["t_arr"][i] + n_slots, fleet["t_dep"][i])
        for t in range(fleet["t_arr"][i], end):
            profile[t] += p_eff
    return profile

# ============================================================
# MILP 建模工具
# ============================================================
def _g(v):
    if v is None: return 0.0
    val = v.varValue
    return float(val) if val is not None else 0.0

def _build_milp(price, load, pv, fleet, cfg, mode, e_bess_init=None, e_ev_init=None, require_bess_end=True):
    n = len(price)
    dt = cfg["dt_h"]
    n_ev = fleet["n_ev"] if fleet is not None else 0

    E0 = e_bess_init if e_bess_init is not None else cfg["bess_e0"]
    E_max, E_min = cfg["bess_e_max"], cfg["bess_e_min"]
    pc_max, pd_max = cfg["bess_pc_max"], cfg["bess_pd_max"]
    eta_b, eta_bd = cfg["bess_eta_c"], cfg["bess_eta_d"]
    eta_ev, eta_evd = cfg["ev_eta_c"], cfg["ev_eta_d"]

    use_sell = (mode == "A4")
    use_ev_var = mode in ("A3", "A4")
    use_ev_dis = (mode == "A4")

    m = pulp.LpProblem(f"v2g_{mode}", pulp.LpMinimize)

    pg   = [pulp.LpVariable(f"pg_{t}",  0) for t in range(n)]
    pg_emerg = [pulp.LpVariable(f"pg_emerg_{t}", 0) for t in range(n)]
    pc_b = [pulp.LpVariable(f"pcb_{t}", 0, pc_max) for t in range(n)]
    pd_b = [pulp.LpVariable(f"pdb_{t}", 0, pd_max) for t in range(n)]
    pu   = [pulp.LpVariable(f"pu_{t}",  0) for t in range(n)]
    eb_b = [pulp.LpVariable(f"ebb_{t}", E_min, E_max) for t in range(n + 1)]
    zc_b = [pulp.LpVariable(f"zcb_{t}", cat="Binary") for t in range(n)]
    zd_b = [pulp.LpVariable(f"zdb_{t}", cat="Binary") for t in range(n)]

    ps = z_buy = z_sell = None
    if use_sell:
        ps     = [pulp.LpVariable(f"ps_{t}",  0) for t in range(n)]
        z_buy  = [pulp.LpVariable(f"zb_{t}",  cat="Binary") for t in range(n)]
        z_sell = [pulp.LpVariable(f"zs_{t}",  cat="Binary") for t in range(n)]

    pce = ebe = zce = pde = zde = None
    ev_slack = []
    if use_ev_var and n_ev > 0:
        pce = [[pulp.LpVariable(f"pce_{i}_{t}", 0, fleet["p_chg_max"][i]) for t in range(n)] for i in range(n_ev)]
        ebe = [[pulp.LpVariable(f"ebe_{i}_{t}", cfg["ev_soc_min"] * fleet["e_cap"][i], cfg["ev_soc_max"] * fleet["e_cap"][i]) for t in range(n + 1)] for i in range(n_ev)]
        zce = [[pulp.LpVariable(f"zce_{i}_{t}", cat="Binary") for t in range(n)] for i in range(n_ev)]
        if use_ev_dis:
            pde = [[pulp.LpVariable(f"pde_{i}_{t}", 0, fleet["p_dis_max"][i]) for t in range(n)] for i in range(n_ev)]
            zde = [[pulp.LpVariable(f"zde_{i}_{t}", cat="Binary") for t in range(n)] for i in range(n_ev)]
        ev_slack = [pulp.LpVariable(f"slack_{i}", 0) for i in range(n_ev)]

    ev_unord = unordered_charge_profile(fleet, dt) if mode == "A2" and fleet else None

    # 目标函数
    obj = pulp.lpSum(price[t] * pg[t] * dt for t in range(n))
    obj += pulp.lpSum(cfg["emergency_ratio"] * price[t] * pg_emerg[t] * dt for t in range(n))
    if use_sell: obj += -pulp.lpSum(price[t] * cfg["sell_ratio"] * ps[t] * dt for t in range(n))

    if cfg.get("C3_degradation", False):
        c3 = cfg.get("C3_cost_per_kwh", 0.167)
        obj += pulp.lpSum(c3 * pd_b[t] * dt for t in range(n))
        if use_ev_dis:
            for i in range(n_ev): obj += pulp.lpSum(c3 * pde[i][t] * dt for t in range(n))

    if cfg.get("C2_cvar", False) and n > 0:
        lam = cfg.get("C2_lambda", 0.30)
        alpha = cfg.get("C2_alpha", 0.95)
        thr = float(np.quantile(price, alpha))
        for t in range(n):
            if price[t] > thr: obj += lam * (price[t] - thr) * pg[t] * dt

    if use_ev_var:
        for i in range(n_ev): obj += cfg["ev_slack_penalty"] * ev_slack[i]

    m += obj

    # 约束
    for t in range(n):
        lhs = pg[t] + pv[t] + pd_b[t]
        rhs = load[t] + pu[t] + pc_b[t]
        if use_sell: rhs += ps[t]
        if use_ev_dis:
            for i in range(n_ev): lhs += pde[i][t]
        if use_ev_var:
            for i in range(n_ev): rhs += pce[i][t]
        if mode == "A2" and ev_unord is not None: rhs += ev_unord[t]

        m += lhs == rhs
        m += pu[t] <= pv[t]
        m += pc_b[t] <= pc_max * zc_b[t]
        m += pd_b[t] <= pd_max * zd_b[t]
        m += zc_b[t] + zd_b[t] <= 1
        m += eb_b[t + 1] == eb_b[t] + eta_b * pc_b[t] * dt - (1.0 / eta_bd) * pd_b[t] * dt

        if use_sell:
            m += pg[t] <= 1e8 * z_buy[t]
            m += ps[t] <= 1e8 * z_sell[t]
            m += z_buy[t] + z_sell[t] <= 1

        if cfg.get("C4_transformer", False):
            limit = cfg.get("C4_limit_kw", 4000)
            m += pg[t] - (ps[t] if use_sell else 0) <= limit + pg_emerg[t]

        if use_ev_var:
            for i in range(n_ev):
                if t < fleet["t_arr"][i] or t >= fleet["t_dep"][i]:
                    m += pce[i][t] == 0
                    if use_ev_dis: m += pde[i][t] == 0
                    m += ebe[i][t + 1] == ebe[i][t]
                else:
                    m += pce[i][t] <= fleet["p_chg_max"][i] * fleet["will"][i] * zce[i][t]
                    if use_ev_dis:
                        m += pde[i][t] <= fleet["p_dis_max"][i] * fleet["will"][i] * zde[i][t]
                        m += zce[i][t] + zde[i][t] <= 1
                        m += ebe[i][t + 1] == ebe[i][t] + eta_ev * pce[i][t] * dt - (1.0 / eta_evd) * pde[i][t] * dt
                    else:
                        m += ebe[i][t + 1] == ebe[i][t] + eta_ev * pce[i][t] * dt

    m += eb_b[0] == E0
    if require_bess_end: m += eb_b[n] == cfg["bess_e0"]

    if use_ev_var:
        for i in range(n_ev):
            init_e = (e_ev_init[i] if e_ev_init is not None else fleet["e_cap"][i] * fleet["soc_arr"][i])
            m += ebe[i][0] == init_e
            dep_t = min(fleet["t_dep"][i], n)
            m += ebe[i][dep_t] + ev_slack[i] >= fleet["e_cap"][i] * fleet["soc_dep"][i]
            if use_ev_dis:
                dod = cfg.get("ev_discharge_dod_max", 0.30)
                m += pulp.lpSum(pde[i][t] * dt for t in range(n)) <= dod * fleet["e_cap"][i]

    return m, {
        "pg": pg, "pg_emerg": pg_emerg, "ps": ps, "pc_b": pc_b, "pd_b": pd_b, "pu": pu,
        "eb_b": eb_b, "pce": pce, "pde": pde, "ebe": ebe, "ev_slack": ev_slack, "n_ev": n_ev,
        "mode": mode, "use_sell": use_sell,
    }

def _solve_and_extract(m, v, price, load, pv, cfg):
    solver = pulp.HiGHS(msg=0, timeLimit=cfg["solve_time_limit"], gapRel=cfg["mip_gap"], threads=cfg["solver_threads"])
    m.solve(solver)
    if pulp.LpStatus[m.status] != "Optimal": return None

    n = len(price)
    dt = cfg["dt_h"]
    n_ev = v["n_ev"]

    pg   = np.array([_g(x) for x in v["pg"]])
    pg_emerg = np.array([_g(x) for x in v["pg_emerg"]])
    pc_b = np.array([_g(x) for x in v["pc_b"]])
    pd_b = np.array([_g(x) for x in v["pd_b"]])
    pu   = np.array([_g(x) for x in v["pu"]])
    eb_b = np.array([_g(x) for x in v["eb_b"]])
    ps   = np.array([_g(x) for x in v["ps"]]) if v["use_sell"] else np.zeros(n)

    if v["pce"] is not None:
        pce = np.array([[_g(v["pce"][i][t]) for t in range(n)] for i in range(n_ev)])
        ebe = np.array([[_g(v["ebe"][i][t]) for t in range(n + 1)] for i in range(n_ev)])
        pde = (np.array([[_g(v["pde"][i][t]) for t in range(n)] for i in range(n_ev)]) if v["pde"] is not None else np.zeros((n_ev, n)))
    else:
        pce = np.zeros((0, n)); pde = np.zeros((0, n)); ebe = np.zeros((0, n + 1))

    cost_buy = float(np.sum(price * pg + cfg["emergency_ratio"] * price * pg_emerg) * dt)
    rev_sell = (float(np.sum(price * cfg["sell_ratio"] * ps) * dt) if v["use_sell"] else 0.0)
    slack_total = float(sum(_g(x) for x in v["ev_slack"]))
    cost = cost_buy - rev_sell + cfg["ev_slack_penalty"] * slack_total

    return {
        "pg": pg, "pg_emerg": pg_emerg, "ps": ps, "pc_b": pc_b, "pd_b": pd_b, "pu": pu, "eb_b": eb_b,
        "pce": pce, "pde": pde, "ebe": ebe,
        "p_charge_ev":    pce.sum(axis=0) if pce.size > 0 else np.zeros(n),
        "p_discharge_ev": pde.sum(axis=0) if pde.size > 0 else np.zeros(n),
        "e_ev":           ebe.sum(axis=0) if ebe.size > 0 else np.zeros(n + 1),
        "cost": cost, "cost_buy": cost_buy, "revenue_sell": rev_sell, "ev_slack_total": slack_total,
    }

# ============================================================
# A 组求解器 + 评估器
# ============================================================
def solve_a1_bess_only(price, load, pv, cfg):
    m, v = _build_milp(price, load, pv, None, cfg, "A1")
    return _solve_and_extract(m, v, price, load, pv, cfg)

def solve_a2_unordered(price, load, pv, fleet, cfg):
    m, v = _build_milp(price, load, pv, fleet, cfg, "A2")
    res = _solve_and_extract(m, v, price, load, pv, cfg)
    if res is not None:
        res["p_charge_ev"] = unordered_charge_profile(fleet, cfg["dt_h"])
    return res

def solve_a3_smart(price, load, pv, fleet, cfg):
    m, v = _build_milp(price, load, pv, fleet, cfg, "A3")
    return _solve_and_extract(m, v, price, load, pv, cfg)

def solve_a4_v2g(price, load, pv, fleet, cfg):
    m, v = _build_milp(price, load, pv, fleet, cfg, "A4")
    return _solve_and_extract(m, v, price, load, pv, cfg)

def solve_day(price, load, pv, fleet, cfg):
    if cfg.get("C1_rolling", False):
        from enhancement import c1_rolling_solve
        return c1_rolling_solve(price, load, pv, fleet, cfg)
    a = cfg["A"]
    if a == 1: return solve_a1_bess_only(price, load, pv, cfg)
    if a == 2: return solve_a2_unordered(price, load, pv, fleet, cfg)
    if a == 3: return solve_a3_smart(price, load, pv, fleet, cfg)
    if a == 4: return solve_a4_v2g(price, load, pv, fleet, cfg)
    return None

def evaluate(res, price, cfg):
    dt = cfg["dt_h"]
    net = res["pg"] - res["ps"]
    return {
        "cost":         res["cost"],
        "buy_kwh":      float(np.sum(res["pg"]) * dt),
        "sell_kwh":     float(np.sum(res["ps"]) * dt),
        "curtail_kwh":  float(np.sum(res["pu"]) * dt),
        "peak_kw":      float(np.max(net)),
        "valley_kw":    float(np.min(net)),
        "p2v_kw":       float(np.max(net) - np.min(net)),
        "std_kw":       float(np.std(net)),
        "peak_h_kw":    float(np.mean(net[34:44])),
        "valley_h_kw":  float(np.mean(net[0:16])),
        "ev_chg_kwh":   float(np.sum(res["p_charge_ev"]) * dt),
        "ev_dis_kwh":   float(np.sum(res["p_discharge_ev"]) * dt),
        "ev_slack":     float(res.get("ev_slack_total", 0.0)),
    }

def allocate_revenue(total_gain, cfg):
    rows = [{"角色": r, "比例": p, "金额(元)": total_gain * p} for r, p in cfg["share"].items()]
    return pd.DataFrame(rows)

# ============================================================
# EV 母车队池（供规模扫描使用）
# ============================================================
_GLOBAL_EV_POOL = {}

def _gen_ev_fleet_master(day_idx, cfg, master_n):
    rng = np.random.default_rng(cfg["seed"] + day_idx * 17)
    peaks = cfg["ev_arr_peaks"]
    w = np.array([p[2] for p in peaks]); w = w / w.sum()

    which = rng.choice(len(peaks), size=master_n, p=w)
    arrivals = np.array([rng.normal(peaks[k][0], peaks[k][1]) for k in which])
    arrivals = np.clip(arrivals, 0.0, 22.0)
    dwell = np.clip(
        rng.normal(cfg["ev_dwell_mean_h"], cfg["ev_dwell_std_h"], master_n),
        cfg["ev_dwell_min_h"], cfg["ev_dwell_max_h"])
    departures = np.clip(arrivals + dwell, 0.0, 23.9)
    will = rng.beta(cfg["ev_will_alpha"], cfg["ev_will_beta"], master_n)
    e_cap = rng.choice(cfg["ev_cap_choices"], size=master_n,
                       p=cfg["ev_cap_probs"]).astype(float)
    soc_arr = np.clip(rng.normal(cfg["ev_soc_arr_mean"],
                                  cfg["ev_soc_arr_std"], master_n), 0.10, 0.60)
    soc_dep = np.clip(rng.normal(cfg["ev_soc_dep_mean"],
                                  cfg["ev_soc_dep_std"], master_n), 0.70, 0.95)
    t_arr = np.clip((arrivals * 2).astype(int), 0, 47)
    t_dep = np.clip((departures * 2).astype(int), 1, 48)

    for i in range(master_n):
        max_kwh = (cfg["ev_p_max"] * cfg["ev_eta_c"]
                   * (t_dep[i] - t_arr[i]) * cfg["dt_h"]
                   * max(will[i], 0.1))
        max_soc = soc_arr[i] + max_kwh / e_cap[i]
        soc_dep[i] = min(soc_dep[i], max_soc)
        soc_dep[i] = max(soc_dep[i], soc_arr[i])
        if t_dep[i] <= t_arr[i]:
            t_arr[i], t_dep[i] = 0, 1
            soc_arr[i] = soc_dep[i] = 0.5

    return {
        "n_ev": master_n,
        "e_cap": e_cap, "soc_arr": soc_arr, "soc_dep": soc_dep,
        "will": will, "t_arr": t_arr, "t_dep": t_dep,
        "p_chg_max": np.full(master_n, cfg["ev_p_max"]),
        "p_dis_max": np.full(master_n, cfg["ev_p_max"]),
    }

def prebuild_ev_pool(cfg, master_n=1500):
    global _GLOBAL_EV_POOL
    _GLOBAL_EV_POOL.clear()
    for d in range(cfg["n_days"]):
        _GLOBAL_EV_POOL[d] = _gen_ev_fleet_master(d, cfg, master_n)
    # ★ 删除了此处的 print 语句！

def get_ev_fleet_slice(day_idx, n_use):
    pool = _GLOBAL_EV_POOL[day_idx]
    sl = slice(0, n_use)
    return {
        "n_ev": n_use,
        "e_cap": pool["e_cap"][sl], "soc_arr": pool["soc_arr"][sl],
        "soc_dep": pool["soc_dep"][sl], "will": pool["will"][sl],
        "t_arr": pool["t_arr"][sl], "t_dep": pool["t_dep"][sl],
        "p_chg_max": pool["p_chg_max"][sl], "p_dis_max": pool["p_dis_max"][sl],
    }
