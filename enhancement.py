"""
enhancement.py — C 组五个工程增强
"""
import numpy as np
import pulp

def _mode_from_a(a_val):
    return {1: "A1", 2: "A2", 3: "A3", 4: "A4"}[a_val]

def _shift_fleet(fleet, t_offset, n_seg):
    n_ev = fleet["n_ev"]
    new = {k: (v.copy() if hasattr(v, "copy") else v) for k, v in fleet.items()}
    t_arr = np.zeros(n_ev, dtype=int)
    t_dep = np.zeros(n_ev, dtype=int)
    for i in range(n_ev):
        ta = fleet["t_arr"][i] - t_offset
        td = fleet["t_dep"][i] - t_offset
        t_arr[i] = int(np.clip(ta, 0, n_seg))
        t_dep[i] = int(np.clip(td, 0, n_seg))
        if t_dep[i] <= t_arr[i]: t_dep[i] = min(t_arr[i] + 1, n_seg)
    new["t_arr"] = t_arr
    new["t_dep"] = t_dep
    return new

def c1_rolling_solve(price, load, pv, fleet, cfg):
    from core import _build_milp, _solve_and_extract
    n = len(price)
    dt = cfg["dt_h"]
    step = int(cfg["C1_roll_hours"] / dt)
    roll_starts = list(range(0, n, step))

    pg_all = np.zeros(n); ps_all = np.zeros(n); pu_all = np.zeros(n)
    pc_b_all = np.zeros(n); pd_b_all = np.zeros(n)
    eb_b_all = np.zeros(n + 1)

    n_ev = fleet["n_ev"]
    pce_all = np.zeros((n_ev, n))
    pde_all = np.zeros((n_ev, n))
    ebe_all = np.zeros((n_ev, n + 1))

    e_bess_cur = cfg["bess_e0"]
    e_ev_cur = fleet["e_cap"] * fleet["soc_arr"]
    eb_b_all[0] = e_bess_cur
    ebe_all[:, 0] = e_ev_cur

    mode = _mode_from_a(cfg["A"])
    prev_pg_plan = np.zeros(n)

    for idx, t0 in enumerate(roll_starts):
        t1 = roll_starts[idx + 1] if idx + 1 < len(roll_starts) else n
        n_seg = n - t0
        fleet_seg = _shift_fleet(fleet, t0, n_seg)

        m, v = _build_milp(
            price[t0:], load[t0:], pv[t0:], fleet_seg, cfg, mode,
            e_bess_init=e_bess_cur, e_ev_init=e_ev_cur, require_bess_end=(t1 >= n),
        )
        
        if idx > 0:
            over_buy = [pulp.LpVariable(f"over_{t}", lowBound=0) for t in range(n_seg)]
            under_buy = [pulp.LpVariable(f"under_{t}", lowBound=0) for t in range(n_seg)]
            for t in range(n_seg):
                m += over_buy[t] >= v["pg"][t] - prev_pg_plan[t0 + t]
                m += under_buy[t] >= prev_pg_plan[t0 + t] - v["pg"][t]
            m.objective += pulp.lpSum(
                (cfg["rolling_over_buy_ratio"] * price[t0+t] * over_buy[t] + 
                 cfg["rolling_under_buy_ratio"] * price[t0+t] * under_buy[t]) * dt for t in range(n_seg))

        res = _solve_and_extract(m, v, price[t0:], load[t0:], pv[t0:], cfg)
        if res is None:
            pg_all[t0:t1] = np.maximum(load[t0:t1] - pv[t0:t1], 0)
            eb_b_all[t0 + 1:t1 + 1] = e_bess_cur
            continue
            
        prev_pg_plan[t0:] = res["pg"]
        commit = t1 - t0
        pg_all[t0:t1]   = res["pg"][:commit]
        ps_all[t0:t1]   = res["ps"][:commit]
        pu_all[t0:t1]   = res["pu"][:commit]
        pc_b_all[t0:t1] = res["pc_b"][:commit]
        pd_b_all[t0:t1] = res["pd_b"][:commit]
        eb_b_all[t0 + 1:t1 + 1] = res["eb_b"][1:commit + 1]

        if res["pce"].size > 0:
            pce_all[:, t0:t1] = res["pce"][:, :commit]
            pde_all[:, t0:t1] = res["pde"][:, :commit]
            ebe_all[:, t0 + 1:t1 + 1] = res["ebe"][:, 1:commit + 1]

        e_bess_cur = eb_b_all[t1]
        e_ev_cur = ebe_all[:, t1]

    cost_buy = float(np.sum(price * pg_all) * dt)
    rev_sell = (float(np.sum(price * cfg["sell_ratio"] * ps_all) * dt) if cfg["A"] == 4 else 0.0)

    # ★ 完整的返回字典
    return {
        "pg": pg_all, "pg_emerg": np.zeros(n), "ps": ps_all, "pu": pu_all,
        "pc_b": pc_b_all, "pd_b": pd_b_all, "eb_b": eb_b_all,
        "pce": pce_all, "pde": pde_all, "ebe": ebe_all,
        "p_charge_ev": pce_all.sum(axis=0), "p_discharge_ev": pde_all.sum(axis=0), "e_ev": ebe_all.sum(axis=0),
        "cost": cost_buy - rev_sell, "cost_buy": cost_buy, "revenue_sell": rev_sell, "ev_slack_total": 0.0,
    }

def apply_c_setup(cfg):
    if cfg.get("C2_cvar", False):
        assert 0 < cfg.get("C2_alpha", 0.95) < 1
        assert cfg.get("C2_lambda", 0.30) >= 0
    if cfg.get("C3_degradation", False):
        assert cfg.get("C3_cost_per_kwh", 0.167) >= 0
    if cfg.get("C4_transformer", False):
        assert cfg.get("C4_limit_kw", 4000) > 0
    if cfg.get("C5_dod", False):
        assert 0 < cfg.get("ev_discharge_dod_max", 0.30) <= 1
    return cfg
