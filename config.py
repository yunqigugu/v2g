"""
config.py — 全局配置字典
所有参数集中在 CONFIG，Streamlit 里改字段即可。
"""
    
CONFIG = {
     # ==================== 输出目录 ====================
    "output_root": "outputs",       # 根目录
    "run_tag":     None,            # None=自动根据 A/B/C 生成；字符串=强制指定
    # ==================== 基础 ====================
    "seed": 42,
    "n_days": 30,                  # 仿真天数
    "dt_minutes": 30,              # 时间粒度
    "dt_h": 0.5,
    "n_slots_per_day": 48,
    "output_dir": "outputs",

    # ==================== BESS ====================
    "bess_e0": 6000.0,
    "bess_e_max": 10800.0,
    "bess_e_min": 1200.0,
    "bess_pc_max": 2000.0,
    "bess_pd_max": 2000.0,
    "bess_eta_c": 0.9,
    "bess_eta_d": 0.9,

    # ==================== EV ====================
    "n_ev": 1500,
    "ev_cap_choices": [40.0, 60.0, 80.0],
    "ev_cap_probs":   [0.4, 0.4, 0.2],
    "ev_p_max": 7.0,
    "ev_eta_c": 0.9,
    "ev_eta_d": 0.9,
    "ev_soc_min": 0.10,
    "ev_soc_max": 0.95,
    "ev_soc_arr_mean": 0.40,
    "ev_soc_arr_std": 0.10,
    "ev_soc_dep_mean": 0.80,
    "ev_soc_dep_std": 0.05,
    "ev_dwell_mean_h": 10.0,
    "ev_dwell_std_h": 2.0,
    "ev_dwell_min_h": 1.0,
    "ev_dwell_max_h": 14.0,
    "ev_arr_peaks": [(8.0, 1.0, 0.40), (18.0, 1.5, 0.60)],
    "ev_will_alpha": 5.0,
    "ev_will_beta": 2.0,
    "ev_slack_penalty": 0.4,
    "ev_discharge_dod_max": 0.30,

        # ==================== 市场与惩罚 ====================
    "sell_ratio": 0.60,             # 售电价 = 购电价 × 0.60
    "emergency_ratio": 5.0,         # 紧急购电惩罚倍数 (5λ)
    "rolling_over_buy_ratio": 1.5,  # 滚动多买惩罚 (1.5λ)
    "rolling_under_buy_ratio": 0.5, # 滚动少买惩罚 (0.5λ)
    
    # EV 离站 SOC 松弛惩罚 (0.4 元/kWh)
    "ev_slack_penalty": 0.4,

    # ==================== A 组：理想探索（互斥单选） ====================
    "A": 4,                          # 1=A1纯BESS 2=A2无序 3=A3智能 4=A4 V2G

    # ==================== B 组：不确定性叠加（顺序激活） ====================
    "B1_pv":    True,
    "B2_load":  True,#True or False
    "B3_price": True,
    "B4_ev":    True,
    "B_pv_n_scen":     24,
    "B_pv_quantile":   0.25,
    "B_load_n_scen":   24,
    "B_load_quantile": 0.75,
    "B_price_noise":   0.05,
    "B_price_roll_h":  6,

    # ==================== C 组：工程增强（任意组合） ====================
    "C1_rolling":     True,#True or False
    "C2_cvar":        True,
    "C3_degradation": True,
    "C4_transformer": True,
    "C5_dod":         True,
    "C1_roll_hours":  6,
    "C2_alpha":       0.95,
    "C2_lambda":      0.30,
    "C3_cost_per_kwh": 0.167,
    "C4_limit_kw":    8000.0,

    # ==================== 缩放 ====================
    "price_scale": 1.0,
    "load_scale":  1.0,
    "pv_scale":    1.0,

    # ==================== 求解器 ====================
    "mip_gap":           0.02,
    "solve_time_limit":  120,
    "solver_threads":    4,

    # ==================== 收益分配 ====================
    "share": {
        "车主": 0.50,
        "充电站/聚合商": 0.30,
        "耦合商/平台":   0.20,
    },

    # ==================== EV 规模扫描 ====================
    # 不要放 50, 100, 150... 这种，直接跑几个关键规模
    "sweep_n_ev":   [100, 400, 800, 1200, 1500], 
    "sweep_n_days": 10,            # 保持 10 天求平均，能有效消除单日波动带来的毛刺
}
