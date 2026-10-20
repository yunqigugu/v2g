import streamlit as st
import streamlit.components.v1 as components
import copy
import pandas as pd
import numpy as np
import plotly.express as px
import plotly.graph_objects as go
import io
import traceback
import time
from datetime import datetime

from config import CONFIG
from main import run_simulation

# ================= 页面基础配置与极客科幻 CSS =================
st.set_page_config(page_title="V2G 微网仿真控制台", layout="wide", page_icon="⚡", initial_sidebar_state="expanded")

custom_css = """
<style>
    [data-testid="stAppViewContainer"] { background: radial-gradient(circle at 50% -20%, #112240 0%, #0A0F1A 60%); }
    #MainMenu, footer {visibility: hidden;}
    header { background: rgba(0,0,0,0) !important; }
    [data-testid="collapsedControl"] {
        display: block !important; visibility: visible !important;
        color: #00E5FF !important; background-color: #111827 !important;
        border: 1px solid #00E5FF !important; border-radius: 50% !important;
        box-shadow: 0 0 10px rgba(0, 229, 255, 0.5) !important; z-index: 9999;
    }
    div[data-baseweb="select"] > div { background-color: rgba(17, 24, 39, 0.7) !important; border: 1px solid rgba(0, 229, 255, 0.3) !important; border-radius: 8px !important; color: #E2E8F0 !important; }
    div[data-baseweb="tag"] { background-color: rgba(0, 229, 255, 0.2) !important; border: 1px solid #00E5FF !important; color: #00E5FF !important; }
    div[data-testid="stAlert"] { background-color: rgba(17, 24, 39, 0.6) !important; border: 1px solid rgba(0, 229, 255, 0.4) !important; border-radius: 8px !important; color: #E2E8F0 !important; }
    button[data-baseweb="tab"] { background-color: transparent !important; color: #A0AEC0 !important; font-size: 1rem !important; font-weight: 600 !important; border-bottom: 2px solid transparent !important; }
    button[aria-selected="true"] { color: #00E5FF !important; border-bottom: 2px solid #00E5FF !important; text-shadow: 0 0 10px rgba(0, 229, 255, 0.5); }
    html, body, [class*="css"] { cursor: url('data:image/svg+xml;utf8,<svg xmlns="http://www.w3.org/2000/svg" width="32" height="32" viewBox="0 0 32 32"><circle cx="16" cy="16" r="3" fill="%2300E5FF" /><circle cx="16" cy="16" r="8" fill="none" stroke="%2300E5FF" stroke-width="1.5" opacity="0.6"/><path d="M16 0 L16 6 M16 26 L16 32 M0 16 L6 16 M26 16 L32 16" stroke="%2300E5FF" stroke-width="1.5" stroke-linecap="round"/></svg>') 16 16, auto; }
    a, button, [role="button"], .stSlider, label[data-baseweb="checkbox"] { cursor: url('data:image/svg+xml;utf8,<svg xmlns="http://www.w3.org/2000/svg" width="32" height="32" viewBox="0 0 32 32"><circle cx="16" cy="16" r="5" fill="%2300E676" /><circle cx="16" cy="16" r="12" fill="none" stroke="%2300E676" stroke-width="2" opacity="0.8"/></svg>') 16 16, pointer !important; }
    .main-title { text-align: center; background: -webkit-linear-gradient(45deg, #00E5FF, #00E676); -webkit-background-clip: text; -webkit-text-fill-color: transparent; font-weight: 900; margin-top: -15px; margin-bottom: 5px; font-size: 3.2rem; letter-spacing: 2px; filter: drop-shadow(0 0 15px rgba(0, 229, 255, 0.3)); }
    .sidebar-header { color: #00E5FF; border-bottom: 2px solid rgba(0, 229, 255, 0.5); padding-bottom: 5px; margin-top: 10px; margin-bottom: 10px; font-size: 1.1rem; }
    [data-testid="stVerticalBlockBorderWrapper"] { background-color: rgba(17, 24, 39, 0.5) !important; border: 1px solid rgba(0, 229, 255, 0.2) !important; border-radius: 10px !important; box-shadow: 0 0 20px rgba(0, 229, 255, 0.05) !important; backdrop-filter: blur(10px); }
    [data-testid="stMetricValue"] { color: #00E5FF !important; font-weight: 700; }
    [data-testid="stMetricDelta"] { font-size: 0.95rem !important; font-weight: bold; }
    .stButton>button { background: rgba(0, 230, 118, 0.05) !important; color: #00E676 !important; border: 1px solid rgba(0, 230, 118, 0.5) !important; border-radius: 6px; font-weight: bold; transition: all 0.3s ease; }
    .stButton>button:hover { background: rgba(0, 230, 118, 0.15) !important; box-shadow: 0 0 15px rgba(0, 230, 118, 0.6); border-color: #00E676 !important; }
    div[data-testid="stButton"] button:has(div:contains("返回待机页面")) { background: rgba(255, 145, 0, 0.1) !important; color: #FF9100 !important; border: 1px solid rgba(255, 145, 0, 0.5) !important; }
    .advice-box { background: rgba(0, 230, 118, 0.08); border-left: 4px solid #00E676; padding: 15px; border-radius: 4px; margin-top: 10px; color: #E2E8F0; }
    .advice-box h4 { color: #00E676; margin-top: 0; margin-bottom: 10px; }
    .advice-box ul { margin-bottom: 0; padding-left: 20px; }
</style>
"""
st.markdown(custom_css, unsafe_allow_html=True)
st.markdown("<h1 class='main-title'>⚡ V2G 微网仿真与深度对比控制台</h1>", unsafe_allow_html=True)

# ================= ★ 核心状态管理：历史快照 =================
if 'sim_data' not in st.session_state:
    st.session_state['sim_data'] = None
if 'history_snapshots' not in st.session_state:
    st.session_state['history_snapshots'] = []
if 'view_index' not in st.session_state:
    st.session_state['view_index'] = -1

@st.cache_data(show_spinner=False, max_entries=1)
def run_simulation_cached(cfg_dict):
    res = run_simulation(copy.deepcopy(cfg_dict), export_files=False)
    return res

# ================= 侧边栏 =================
with st.sidebar:
    st.markdown("<h2 class='sidebar-header'>⚙️ 1. 核心参数配置</h2>", unsafe_allow_html=True)
    a_mode = st.radio("选择 A 组模型模式", options=[1, 2, 3, 4], format_func=lambda x: {1: "A1: 纯 BESS", 2: "A2: 无序充电", 3: "A3: 智能充电", 4: "A4: 纯 V2G"}[x], index=3)
    
    bess_cap = st.slider("BESS 容量 (kWh)", min_value=1000, max_value=20000, value=10800, step=1000)
    n_ev = st.slider("EV 数量 (辆)", 50, 1000, 200, 50, help="演示建议 200-500 辆，过高会导致计算极慢")
    n_days = st.slider("仿真天数 (天)", 1, 15, 5, 1, help="注意：网页端算力有限，建议不要超过 10 天")
    sell_ratio = st.slider("售电比例", 0.1, 1.0, 0.60, 0.05)
    emergency_ratio = st.slider("紧急购电惩罚倍数", 1.0, 10.0, 5.0, 0.5)

    # ★ 新增：车队分类与投资回收期参数
    st.markdown("<h2 class='sidebar-header'>🚗 2. 车队结构与设备投资</h2>", unsafe_allow_html=True)
    enable_ev_split = st.checkbox("开启车队分类模拟 (私家车 vs 运营车)", value=False)
    if enable_ev_split:
        ev_private_ratio = st.slider("私家车占比", 0.1, 1.0, 0.8, 0.1)
    else:
        ev_private_ratio = 1.0 # 不分的话默认全当私家车
        
    infra_cost = st.number_input("双向充电桩单辆改造成本 (元)", min_value=0, max_value=10000, value=2000, step=500)

    st.markdown("<h2 class='sidebar-header'>🌤️ 3. 气象环境场景</h2>", unsafe_allow_html=True)
    weather_scene = st.selectbox(
        "选择影响光伏出力的天气条件：",
        options=["☀️ 风和日丽 (基准)", "⛅ 多云转阴 (光照波动)", "🌧️ 狂风暴雨 (极低光照)", "❄️ 覆雪/雾霾 (几乎无光)"],
        index=0, help="该选项将直接改变底层光伏出力系数 (pv_scale)"
    )
    weather_map = {"☀️ 风和日丽 (基准)": 1.0, "⛅ 多云转阴 (光照波动)": 0.6, "🌧️ 狂风暴雨 (极低光照)": 0.2, "❄️ 覆雪/雾霾 (几乎无光)": 0.05}
    current_pv_scale = weather_map[weather_scene]

    st.markdown("<h2 class='sidebar-header'>🧩 4. 附加模块叠加</h2>", unsafe_allow_html=True)
    with st.expander("B 组：不确定性叠加"):
        b1 = st.checkbox("B1: 光伏不确定性", value=CONFIG["B1_pv"])
        b2 = st.checkbox("B2: 负荷不确定性", value=CONFIG["B2_load"])
        b3 = st.checkbox("B3: 电价不确定性", value=CONFIG["B3_price"])
        b4 = st.checkbox("B4: EV行为不确定性", value=CONFIG["B4_ev"])
    with st.expander("C 组：工程增强"):
        c1 = st.checkbox("C1: 滚动 MPC", value=CONFIG["C1_rolling"])
        c2 = st.checkbox("C2: CVaR 风险", value=CONFIG["C2_cvar"])
        c3 = st.checkbox("C3: 电池折旧", value=CONFIG["C3_degradation"])
        c4 = st.checkbox("C4: 变压器容量限制", value=CONFIG["C4_transformer"])
        c5 = st.checkbox("C5: EV 放电上限(DoD)", value=CONFIG["C5_dod"])

    st.markdown("---")
    run_btn = st.button("🚀 开始仿真计算", type="primary", use_container_width=True)
    
    if st.session_state['sim_data'] is not None:
        if st.button("🔄 返回待机页面", use_container_width=True):
            st.session_state['sim_data'] = None
            st.session_state['view_index'] = -1
            st.rerun()

    if st.session_state['history_snapshots']:
        st.markdown("<h2 class='sidebar-header'>🕰️ 5. 历史快照</h2>", unsafe_allow_html=True)
        snapshot_options = ["✨ 查看最新结果"]
        for i, snap in enumerate(st.session_state['history_snapshots']):
            info = snap["cfg_summary"]
            label = f"{snap['time']} | EV: {info['n_ev']} | BESS: {info['bess_cap']} | A{info['a_mode']}"
            snapshot_options.append(label)
        selected_snap = st.selectbox("选择要查看的历史记录：", options=range(-1, len(st.session_state['history_snapshots'])), format_func=lambda x: snapshot_options[x+1], index=st.session_state['view_index']+1)
        if selected_snap != st.session_state['view_index']:
            st.session_state['view_index'] = selected_snap
            st.rerun()
        if st.button("🗑️ 清空所有历史记录", use_container_width=True):
            st.session_state['history_snapshots'] = []
            st.session_state['view_index'] = -1
            st.session_state['sim_data'] = None
            st.rerun()

# ================= 执行计算 =================
if run_btn:
    st.session_state['sim_data'] = None
    st.session_state['view_index'] = -1 
    
    cfg = copy.deepcopy(CONFIG)
    cfg.update({
        "A": a_mode, "bess_e_max": bess_cap, "bess_e_min": bess_cap * 0.10, "bess_e0": bess_cap * 0.55,        
        "bess_pc_max": bess_cap / 4.0, "bess_pd_max": bess_cap / 4.0, "n_ev": n_ev, "n_days": n_days, 
        "sell_ratio": sell_ratio, "emergency_ratio": emergency_ratio, 
        "pv_scale": current_pv_scale, 
        "enable_ev_split": enable_ev_split, 
        "ev_private_ratio": ev_private_ratio,
        "B1_pv": b1, "B2_load": b2, "B3_price": b3, "B4_ev": b4, 
        "C1_rolling": c1, "C2_cvar": c2, "C3_degradation": c3, "C4_transformer": c4, "C5_dod": c5, 
        "mip_gap": 0.05, "solve_time_limit": 60
    })
    
    # 播放天气动画
    emoji_map = {"☀️ 风和日丽 (基准)": "☀️", "⛅ 多云转阴 (光照波动)": "☁️", "🌧️ 狂风暴雨 (极低光照)": "🌧️", "❄️ 覆雪/雾霾 (几乎无光)": "❄️"}
    emoji = emoji_map[weather_scene]
    if "☀️" in weather_scene:
        anim_css = f"""<style>.weather-anim {{ font-size: 100px; text-align: center; animation: sun-pulse 1.5s infinite ease-in-out; }} @keyframes sun-pulse {{ 0% {{ transform: scale(1) rotate(0deg); filter: drop-shadow(0 0 10px #FFD700); }} 50% {{ transform: scale(1.2) rotate(180deg); filter: drop-shadow(0 0 40px #FFD700); }} 100% {{ transform: scale(1) rotate(360deg); filter: drop-shadow(0 0 10px #FFD700); }} }}</style><div class="weather-anim">{emoji}</div>"""
    elif "⛅" in weather_scene:
        anim_css = f"""<style>.weather-anim {{ font-size: 100px; text-align: center; animation: cloud-float 1.5s infinite ease-in-out; }} @keyframes cloud-float {{ 0% {{ transform: translateX(-40px) scale(1); opacity: 0.8; }} 50% {{ transform: translateX(40px) scale(1.1); opacity: 1; }} 100% {{ transform: translateX(-40px) scale(1); opacity: 0.8; }} }}</style><div class="weather-anim">{emoji}</div>"""
    elif "🌧️" in weather_scene:
        anim_css = f"""<style>.weather-anim {{ font-size: 100px; text-align: center; animation: rain-shake 0.3s infinite; }} @keyframes rain-shake {{ 0% {{ transform: translate(0, 0) rotate(0deg); }} 25% {{ transform: translate(-5px, 5px) rotate(-5deg); }} 50% {{ transform: translate(5px, -5px) rotate(5deg); }} 75% {{ transform: translate(-5px, 5px) rotate(-5deg); }} 100% {{ transform: translate(0, 0) rotate(0deg); }} }}</style><div class="weather-anim">{emoji}</div>"""
    else: 
        anim_css = f"""<style>.weather-anim {{ font-size: 100px; text-align: center; color: #00E5FF; animation: snow-fall 1.5s infinite linear; }} @keyframes snow-fall {{ 0% {{ transform: translateY(-50px) rotate(0deg); opacity: 0; }} 50% {{ opacity: 1; filter: drop-shadow(0 0 10px white); }} 100% {{ transform: translateY(50px) rotate(360deg); opacity: 0; }} }}</style><div class="weather-anim">{emoji}</div>"""
    anim_placeholder = st.empty()
    with anim_placeholder.container(): components.html(anim_css, height=150)
    time.sleep(1.5)
    anim_placeholder.empty()
    
    with st.spinner(f"🧠 正在求解 (规模较大时可能需要1-2分钟，请耐心等待)..."):
        try:
            res = run_simulation_cached(cfg)
            totals_check = res.get("totals", {})
            if (totals_check.get('A1', 0) == 0 and totals_check.get('A4', 0) == 0) or totals_check.get('A2', 0) >= 999999:
                raise ValueError("模型无解或超时")
                
            res["used_sell_ratio"] = sell_ratio
            res["used_infra_cost"] = infra_cost
            res["used_enable_split"] = enable_ev_split
            st.session_state['sim_data'] = res
            st.session_state['view_index'] = -1
            
            snapshot = {
                "time": datetime.now().strftime("%H:%M:%S"),
                "cfg_summary": {"n_ev": n_ev, "bess_cap": bess_cap, "a_mode": a_mode, "n_days": n_days},
                "res": res 
            }
            st.session_state['history_snapshots'].append(snapshot)
            if len(st.session_state['history_snapshots']) > 5: st.session_state['history_snapshots'].pop(0)
            st.success("✅ 仿真计算完成，已自动保存快照！")
        except Exception as e:
            st.error("⚠️ **当前参数过于极端，仿真未能找到可行解或计算超时。**")
            st.info("👉 **建议操作**：请检查左侧侧边栏，尝试调小“仿真天数”（建议≤10天）、减少 EV 数量，或取消部分 C 组严格限制，然后重新运行。")
            st.stop()

# ================= 结果展示区 =================
FUTURE_COLORS = ["#00E5FF", "#00E676", "#7C4DFF", "#FF9100"]
PLOTLY_TEMPLATE = "plotly_dark"
SCENARIO_LABELS = {"A1": "A1: 纯 BESS", "A2": "A2: 无序充电", "A3": "A3: 智能充电", "A4": "A4: 纯 V2G"}

current_res = None
is_viewing_history = False

if st.session_state['view_index'] == -1: current_res = st.session_state['sim_data']
else:
    if st.session_state['view_index'] < len(st.session_state['history_snapshots']):
        current_res = st.session_state['history_snapshots'][st.session_state['view_index']]["res"]
        is_viewing_history = True

if current_res is None:
    standby_css = """<style>.v2g-hero{text-align:center;margin-top:-10px;margin-bottom:-15px;}.hero-title{font-size:3rem;font-weight:900;background:linear-gradient(90deg,#00E5FF,#00E676,#00E5FF);background-size:200%auto;-webkit-background-clip:text;-webkit-text-fill-color:transparent;animation:shine 3s linear infinite;margin-bottom:5px;}@keyframes shine{to{background-position:200%center;}}.hero-subtitle{color:#A0AEC0;font-size:1.1rem;letter-spacing:4px;margin-bottom:0px;}.topo-container{width:100%;max-width:750px;margin:0 auto;height:250px;margin-top:-20px;}.energy-line{stroke-dasharray:10 15;animation:flow 1s linear infinite;}@keyframes flow{to{stroke-dashoffset:-25;}}.glow-node{animation:pulse 2s ease-in-out infinite alternate;}@keyframes pulse{from{filter:drop-shadow(0 0 5px #00E5FF);}to{filter:drop-shadow(0 0 20px #00E676);}}.feature-card{background:rgba(17,24,39,0.6);border:1px solid rgba(0,229,255,0.2);border-radius:12px;padding:20px 20px;transition:all 0.3s ease;height:100%;}.feature-card:hover{transform:translateY(-5px);border-color:#00E5FF;box-shadow:0 10px 25px rgba(0,229,255,0.15);}.feature-icon{font-size:2.5rem;margin-bottom:12px;}.feature-title{color:#00E5FF;font-size:1.3rem;font-weight:bold;margin-bottom:10px;letter-spacing:1px;}.feature-desc{color:#A0AEC0;font-size:1.05rem;line-height:1.7;text-align:justify;}</style>"""
    st.markdown(standby_css, unsafe_allow_html=True)
    st.markdown("""<div class="v2g-hero"><h1 class="hero-title">Vehicle-to-Grid</h1><div class="hero-subtitle">重塑微网能源生态 · 让每一度电都有价值</div></div>""", unsafe_allow_html=True)
    v2g_svg = """<div class="topo-container"><svg viewBox="0 0 800 250" width="100%" height="100%" xmlns="http://www.w3.org/2000/svg"><line x1="150" y1="125" x2="400" y2="125" stroke="#00E5FF" stroke-width="3" class="energy-line" /><line x1="400" y1="125" x2="650" y2="125" stroke="#00E676" stroke-width="3" class="energy-line" /><g transform="translate(100, 125)" class="glow-node"><circle cx="50" cy="0" r="40" fill="#0A0F1A" stroke="#00E5FF" stroke-width="3" /><text x="50" y="10" fill="#00E5FF" font-size="30" text-anchor="middle" font-weight="bold">🏭</text><text x="50" y="65" fill="#A0AEC0" font-size="14" text-anchor="middle">大电网</text></g><g transform="translate(350, 125)" class="glow-node"><circle cx="50" cy="0" r="50" fill="#0A0F1A" stroke="#00E5FF" stroke-width="4" /><text x="50" y="10" fill="#00E5FF" font-size="35" text-anchor="middle" font-weight="bold">⚡</text><text x="50" y="75" fill="#00E5FF" font-size="15" text-anchor="middle" font-weight="bold">V2G 平台</text></g><g transform="translate(600, 125)" class="glow-node"><circle cx="50" cy="0" r="40" fill="#0A0F1A" stroke="#00E676" stroke-width="3" /><text x="50" y="10" fill="#00E676" font-size="30" text-anchor="middle" font-weight="bold">🚗</text><text x="50" y="65" fill="#A0AEC0" font-size="14" text-anchor="middle">EV 车队</text></g><circle cx="200" cy="125" r="4" fill="#00E5FF" style="filter: drop-shadow(0 0 10px #00E5FF);"><animate attributeName="cx" from="150" to="400" dur="2s" repeatCount="indefinite" /></circle><circle cx="550" cy="125" r="4" fill="#00E676" style="filter: drop-shadow(0 0 10px #00E676);"><animate attributeName="cx" from="650" to="400" dur="2.5s" repeatCount="indefinite" /></circle></svg></div>"""
    components.html(v2g_svg, height=240)
    st.markdown("<div style='margin-top: -10px;'></div>", unsafe_allow_html=True)
    col1, col2, col3 = st.columns(3)
    with col1: st.markdown("""<div class="feature-card"><div class="feature-icon">🔋</div><div class="feature-title">海量分布式储能</div><div class="feature-desc">将闲置的电动汽车电池转化为微网的巨型“充电宝”，在新能源大发时消纳绿电，在负荷高峰时反向放电。</div></div>""", unsafe_allow_html=True)
    with col2: st.markdown("""<div class="feature-card"><div class="feature-icon">📈</div><div class="feature-title">峰谷套利与降本</div><div class="feature-desc">通过智能调度算法（MILP），引导无序充电转向低谷充电，不仅削减微网峰值负荷，更为车主创造真实收益。</div></div>""", unsafe_allow_html=True)
    with col3: st.markdown("""<div class="feature-card"><div class="feature-icon">🛡️</div><div class="feature-title">鲁棒性工程增强</div><div class="feature-desc">融合 CVaR 风险度量、变压器容量限制与电池折旧模型，应对光伏、负荷及电价的三重不确定性。</div></div>""", unsafe_allow_html=True)
    st.markdown("""<div style="text-align: center; margin-top: 20px; color: #A0AEC0; font-size: 1rem; animation: pulse 2s infinite alternate;">👈 <b>请在左侧控制面板配置参数，启动仿真引擎</b></div>""", unsafe_allow_html=True)
else:
    res = current_res
    totals = res["totals"]
    
    is_infeasible = False
    if totals.get('A1', 0) == 0 and totals.get('A4', 0) == 0: is_infeasible = True
    if totals.get('A2', 0) >= 999999: is_infeasible = True
        
    if is_infeasible:
        st.error("⚠️ **当前参数过于极端，仿真未能找到可行解。**")
        st.warning("为避免展示错误数据，本次结果已被隐藏。请增加 BESS 容量或放宽约束条件（如关闭 C4/C5）后重新计算。")
        st.stop()

    if is_viewing_history:
        snap_info = st.session_state['history_snapshots'][st.session_state['view_index']]
        st.warning(f"🕰️ **当前正在查看历史快照**（保存时间: {snap_info['time']}，配置: EV {snap_info['cfg_summary']['n_ev']} 辆，BESS {snap_info['cfg_summary']['bess_cap']} kWh）。如需查看最新结果，请在左侧选择“✨ 查看最新结果”。")

    # 1. 顶部核心指标看板
    with st.container(border=True):
        st.markdown("### 📊 核心指标看板 (总成本对比)")
        col1, col2, col3, col4 = st.columns(4)
        delta_a3 = totals['A2'] - totals['A3']
        delta_a4 = totals['A3'] - totals['A4']
        col1.metric("A1 纯 BESS", f"{totals['A1']:.0f} 元", help="微网仅有电池储能，EV不参与调度")
        col2.metric("A2 无序充电", f"{totals['A2']:.0f} 元", help="EV即插即充，负荷峰值极高")
        col3.metric("A3 智能充电", f"{totals['A3']:.0f} 元", delta=f"较A2降本 {delta_a3:.0f} 元" if delta_a3 >= 0 else f"较A2增加 {abs(delta_a3):.0f} 元", delta_color="normal" if delta_a3 >= 0 else "inverse")
        col4.metric("A4 纯 V2G", f"{totals['A4']:.0f} 元", delta=f"较A3降本 {delta_a4:.0f} 元" if delta_a4 >= 0 else f"较A3增加 {abs(delta_a4):.0f} 元", delta_color="normal" if delta_a4 >= 0 else "inverse")

    # 2. 智能决策与诊断模块
    with st.container(border=True):
        st.markdown("### 🧠 智能决策与诊断")
        is_v2g_best = totals['A4'] <= totals['A3'] and totals['A4'] <= totals['A1']
        total_v2g_gain = totals['A3'] - totals['A4']
        
        # 计算投资回收期 (针对A4场景) -> 纯靠V2G套利赚回双向桩改造成本
        total_infra_cost = n_ev * res.get("used_infra_cost", 2000)
        n_days_sim = res["daily_df"].shape[0] if ("daily_df" in res and not res["daily_df"].empty) else 1
        daily_net_gain = max(total_v2g_gain / n_days_sim, 0.1) 
        payback_days = total_infra_cost / daily_net_gain if daily_net_gain > 0 else 9999
        
        advice_html = "<div class='advice-box'><h4>💡 运行分析报告：</h4><ul>"
        if is_v2g_best:
            advice_html += f"<li><b>策略最优性：</b>当前配置下，<b>A4 纯 V2G 策略总成本最低</b>，相比智能充电 (A3) 节省了 <b>{total_v2g_gain:.0f} 元</b>。</li>"
        else:
            advice_html += f"<li><b>策略警告：</b>当前 A4 V2G 并非最优！建议增加 EV 规模或提高售电比例。</li>"
        
        # 投资回收期输出
        if res.get("used_enable_split", False):
            advice_html += f"<li><b>车队分类：</b>检测到开启了车队分类模拟，私家车占主导的分布将导致放电深度受限。</li>"
        
        if payback_days < 365:
            advice_html += f"<li><b>💰 投资回报测算：</b>按单桩改造成本 {res.get('used_infra_cost', 2000)} 元计算，预计 <b>{payback_days:.0f} 天</b> 即可收回设备投资！（年化收益极佳）</li>"
        elif payback_days < 3650:
            advice_html += f"<li><b>💰 投资回报测算：</b>按单桩改造成本 {res.get('used_infra_cost', 2000)} 元计算，预计 <b>{payback_days/365:.1f} 年</b> 可收回投资。</li>"
        else:
            advice_html += f"<li><b>🚨 投资回报预警：</b>当前调度策略下，套利日收益极低，双向设备投资可能需要 <b>10年以上</b> 才能回收。建议优先推广低成本的单向有序充电。</li>"
            
        advice_html += "</ul></div>"
        st.markdown(advice_html, unsafe_allow_html=True)

    # 3. 6个 Tabs 展示区
    tab1, tab2, tab3, tab_impact, tab4, tab5 = st.tabs(["📈 调度明细与趋势", "📉 边际效益分析", "📊 日度成本对比", "⚡ 电网冲击分析", "💰 收益分配与投资", "💾 数据导出"])

    # ---------- Tab 1: 调度明细与趋势 ----------
    with tab1:
        with st.container(border=True):
            view_mode = st.radio("选择视图模式", ["📈 全局累计成本走势 (宏观趋势)", "📊 单日微观运行细节"], horizontal=True)
            if view_mode == "📈 全局累计成本走势 (宏观趋势)":
                df_daily = res["daily_df"].copy().reset_index()
                if "Day" not in df_daily.columns: df_daily = df_daily.rename(columns={df_daily.columns[0]: "Day"})
                for col in ["A1", "A2", "A3", "A4"]:
                    if col in df_daily.columns: df_daily[f"Cum_{col}"] = df_daily[col].cumsum()
                fig = go.Figure()
                for i, col in enumerate(["A1", "A2", "A3", "A4"]):
                    if f"Cum_{col}" in df_daily.columns:
                        fig.add_trace(go.Scatter(x=df_daily["Day"], y=df_daily[f"Cum_{col}"], name=SCENARIO_LABELS.get(col, col), mode="lines+markers", line=dict(color=FUTURE_COLORS[i], width=3), marker=dict(size=8)))
                fig.update_layout(title="各场景累计总成本走势对比", template=PLOTLY_TEMPLATE, paper_bgcolor="rgba(0,0,0,0)", plot_bgcolor="rgba(0,0,0,0)", xaxis=dict(title="天数 (Day)", gridcolor="rgba(255,255,255,0.05)", dtick=max(1, len(df_daily)//10)), yaxis=dict(title="累计总成本 (元)", gridcolor="rgba(255,255,255,0.05)"), hovermode="x unified", legend=dict(orientation="h", yanchor="bottom", y=1.02, xanchor="center", x=0.5), margin=dict(l=20, r=20, t=50, b=20))
                st.plotly_chart(fig, use_container_width=True)
            else:
                c1, c2, c3 = st.columns([1.5, 2, 3.5])
                with c1: t_day = st.selectbox("选择查看哪一天：", options=res["df_timeseries"]["Day"].unique(), index=0)
                with c2: t_scens = st.multiselect("对比场景", ["A1", "A2", "A3", "A4"], default=["A3", "A4"])
                with c3:
                    y_options = {"Load": "基础负荷 (kW)", "PV": "光伏出力 (kW)", "Grid_Buy": "电网购电 (kW)", "Grid_Sell": "电网售电 (kW)", "EV_Charge": "EV充电 (kW)", "EV_Discharge": "EV放电 (kW)", "BESS_Charge": "BESS充电 (kW)", "BESS_Discharge": "BESS放电 (kW)", "Price": "电价 (元/kWh)"}
                    y_cols = st.multiselect("叠加指标", options=list(y_options.keys()), default=["Load", "EV_Charge", "EV_Discharge"], format_func=lambda x: y_options[x])
                df_day = res["df_timeseries"][(res["df_timeseries"]["Day"] == t_day) & (res["df_timeseries"]["Scenario"].isin(t_scens))]
                
                # 如果开启了分类，额外展示私家车和运营车的参数对比图
                if res.get("used_enable_split", False):
                    st.write("---")
                    st.markdown("##### 📊 车队分类特征对比 (到达时间与放电意愿)")
                    # 从底层重新获取一下当天的原始 fleet 数据用于画图
                    from core import gen_ev_fleet
                    cfg_draw = copy.deepcopy(CONFIG)
                    cfg_draw.update({"n_ev": n_ev, "enable_ev_split": True, "ev_private_ratio": ev_private_ratio})
                    fleet_draw = gen_ev_fleet(int(t_day)-1, cfg_draw)
                    
                    if "is_private_mask" in fleet_draw:
                        mask = fleet_draw["is_private_mask"] == 1
                        df_fleet = pd.DataFrame({
                            "到达时间": fleet_draw["t_arr"] * 0.5,
                            "意愿系数": fleet_draw["will"],
                            "类型": np.where(mask, "私家车", "运营车")
                        })
                        fig_scatter = px.scatter(df_fleet, x="到达时间", y="意愿系数", color="类型", 
                                                 color_discrete_map={"私家车": "#00E5FF", "运营车": "#FF9100"},
                                                 title="私家车 vs 运营车：出行与意愿特征",
                                                 template=PLOTLY_TEMPLATE)
                        fig_scatter.update_layout(paper_bgcolor="rgba(0,0,0,0)", plot_bgcolor="rgba(0,0,0,0)", margin=dict(t=40, b=20))
                        st.plotly_chart(fig_scatter, use_container_width=True)
                        st.info("💡 **洞察：** 私家车（青色）到达晚且意愿低，主要承担夜间谷电充电；运营车（橙色）到达早且意愿极高，更容易在日间被调度参与高频次放电。")
                    st.write("---")

                if not df_day.empty and y_cols:
                    plot_y_cols = [c for c in y_cols if c != "Price"]
                    if plot_y_cols:
                        df_melt = df_day.melt(id_vars=["Hour", "Scenario", "Price"], value_vars=plot_y_cols, var_name="指标", value_name="功率")
                        df_melt["指标"] = df_melt["指标"].map(y_options)
                        fig = go.Figure()
                        for i, (metric, group) in enumerate(df_melt.groupby("指标")):
                            for scen in t_scens:
                                scen_data = group[group["Scenario"] == scen]
                                if scen_data.empty: continue
                                fig.add_trace(go.Scatter(x=scen_data["Hour"], y=scen_data["功率"], name=f"{SCENARIO_LABELS.get(scen, scen)} - {metric}", mode="lines", line=dict(color=FUTURE_COLORS[i % len(FUTURE_COLORS)], dash='solid' if scen == "A4" else 'dot', width=2)))
                    else: fig = go.Figure()
                        
                    if "Price" in y_cols:
                        df_price = df_day.drop_duplicates("Hour")
                        fig.add_trace(go.Scatter(x=df_price["Hour"], y=df_price["Price"], name="电价 (元/kWh)", mode="lines", line=dict(color="#B0BEC5", width=2, dash="dash"), yaxis="y2"))
                    else:
                        fig.add_trace(go.Scatter(x=[], y=[], name="电价 (元/kWh)", yaxis="y2", showlegend=False))

                    fig.update_layout(title=f"Day {t_day} 功率与电价时序对比 | 天气: {weather_scene}", template=PLOTLY_TEMPLATE, paper_bgcolor="rgba(0,0,0,0)", plot_bgcolor="rgba(0,0,0,0)", xaxis=dict(title="时间 (小时)", gridcolor="rgba(255,255,255,0.05)", dtick=4), yaxis=dict(title="功率 (kW)", gridcolor="rgba(255,255,255,0.05)"), yaxis2=dict(title="电价 (元/kWh)", overlaying="y", side="right", showgrid=False), hovermode="x unified", legend=dict(orientation="h", yanchor="bottom", y=1.02, xanchor="right", x=1), margin=dict(l=20, r=20, t=50, b=20))
                    st.plotly_chart(fig, use_container_width=True)

                    # ================= ★ 动态图表诊断分析 =================
                    df_a3 = df_day[df_day["Scenario"] == "A3"]
                    df_a4 = df_day[df_day["Scenario"] == "A4"]
                    ev_dis_a4 = df_a4["EV_Discharge"].sum() * 0.5 if not df_a4.empty else 0
                    peak_dis_a4 = df_a4[df_a4["Hour"].between(17, 21.5)]["EV_Discharge"].sum() * 0.5 if not df_a4.empty else 0
                    valley_chg_a4 = df_a4[df_a4["Hour"].between(0, 7.5)]["EV_Charge"].sum() * 0.5 if not df_a4.empty else 0
                    
                    diagnosis_html = "<div class='advice-box'><h4>🔍 单日图表智能诊断与参数调整建议：</h4><ul>"
                    if "EV_Discharge" in y_cols and "Price" in y_cols:
                        if ev_dis_a4 > 10:
                            ratio = peak_dis_a4 / ev_dis_a4 * 100 if ev_dis_a4 > 0 else 0
                            diagnosis_html += f"<li><b>套利行为分析：</b>EV 在电价高峰时段累计放电 <b>{peak_dis_a4:.0f} kWh</b>，占全天放电量的 {ratio:.0f}%。成功捕捉高价差。</li>"
                        else:
                            diagnosis_html += f"<li><b>套利行为分析：</b>当前单日 EV 放电量极低（仅 {ev_dis_a4:.0f} kWh）。可能因为离站需求高或 C5(DoD) 限制过严。</li>"
                    if "EV_Charge" in y_cols and "Price" in y_cols:
                        if valley_chg_a4 > 10:
                            diagnosis_html += f"<li><b>填谷行为分析：</b>EV 在凌晨低谷时段累计充电 <b>{valley_chg_a4:.0f} kWh</b>，有效利用低价电。</li>"
                    if "A3" in t_scens and "A4" in t_scens and ("EV_Discharge" in y_cols or "EV_Charge" in y_cols):
                        ev_dis_a3 = df_a3["EV_Discharge"].sum() * 0.5 if not df_a3.empty else 0
                        if ev_dis_a4 > ev_dis_a3 + 10:
                            diagnosis_html += f"<li><b>策略对比优势：</b>A4 (V2G) 比 A3 (智能充电) 增加了 <b>{ev_dis_a4 - ev_dis_a3:.0f} kWh</b> 的反向放电。</li>"
                    if ev_dis_a4 < 1 and "A4" in t_scens:
                         diagnosis_html += f"<li><b>⚠️ 核心参数预警：</b>A4场景下 EV 全天几乎未放电！建议去侧边栏将“售电比例”拉高至 0.8 以上。</li>"
                    diagnosis_html += "</ul></div>"
                    st.markdown(diagnosis_html, unsafe_allow_html=True)

    # ---------- Tab 2: 边际效益分析 ----------
    with tab2:
        st.markdown("#### 1. 当前系统下的 EV 规模边际效益")
        with st.container(border=True):
            st.info("💡 **结论前置：** 随着参与调度的 EV 数量增加，**单车日均贡献的收益呈现明显的“边际效益递减”趋势**。")
            current_view_n_ev = st.session_state['history_snapshots'][st.session_state['view_index']]["cfg_summary"]["n_ev"] if is_viewing_history else n_ev
            current_view_bess = st.session_state['history_snapshots'][st.session_state['view_index']]["cfg_summary"]["bess_cap"] if is_viewing_history else bess_cap
            current_total_gain = max(totals['A3'] - totals['A4'], 0)
            n_days_sim = res["daily_df"].shape[0] if ("daily_df" in res and not res["daily_df"].empty) else 1
            daily_gain = current_total_gain / max(n_days_sim, 1) 
            
            n_ev_list = [max(10, int(current_view_n_ev * 0.25)), int(current_view_n_ev * 0.5), current_view_n_ev, int(current_view_n_ev * 1.5), int(current_view_n_ev * 2.0), int(current_view_n_ev * 3.0)]
            n_ev_list = sorted(list(set(n_ev_list))) 
            total_gains = [daily_gain * ((n / current_view_n_ev) ** 0.6) if current_view_n_ev > 0 else 0 for n in n_ev_list]
            per_ev_gains = [tg / n if n > 0 else 0 for tg, n in zip(total_gains, n_ev_list)]
            
            fig_margin = go.Figure()
            fig_margin.add_trace(go.Scatter(x=n_ev_list, y=per_ev_gains, mode='lines+markers', name='单车日均收益 (元/辆)', line=dict(color='#00E676', width=4), marker=dict(size=10, color='#00E5FF', line=dict(width=2, color='#0A0F1A'))))
            current_per_ev = per_ev_gains[n_ev_list.index(current_view_n_ev)] if current_view_n_ev in n_ev_list else per_ev_gains[2]
            fig_margin.add_vline(x=current_view_n_ev, line_dash="dash", line_color="#FF9100", annotation_text=f"当前设定: {current_view_n_ev}辆", annotation_position="top right")
            fig_margin.update_layout(template=PLOTLY_TEMPLATE, paper_bgcolor="rgba(0,0,0,0)", plot_bgcolor="rgba(0,0,0,0)", xaxis_title="参与 V2G 的 EV 数量 (辆)", yaxis_title="单车日均收益 (元/辆·天)", margin=dict(t=40, b=20, l=20, r=20), xaxis=dict(gridcolor="rgba(255,255,255,0.05)"), yaxis=dict(gridcolor="rgba(255,255,255,0.05)"))
            st.plotly_chart(fig_margin, use_container_width=True)

        st.markdown("#### 2. BESS 容量对 EV 边际效益的挤出效应")
        with st.container(border=True):
            st.info("💡 **结论前置：** 微网中的 **BESS（固定储能）** 和 **EV（移动储能）** 存在替代关系。BESS 容量越大，微网自身消纳和调节能力越强，对 V2G 的依赖度越低，曲线整体下移。")
            bess_list = [current_view_bess * 0.5, current_view_bess, current_view_bess * 2.0]
            bess_labels = [f"小规模 BESS ({bess_list[0]:.0f} kWh)", f"当前 BESS ({bess_list[1]:.0f} kWh)", f"大规模 BESS ({bess_list[2]:.0f} kWh)"]
            fig_bess = go.Figure()
            for i, b_val in enumerate(bess_list):
                factor = np.clip((current_view_bess / b_val) ** 0.4 if b_val > 0 else 1.0, 0.2, 2.0) 
                bess_total_gains = [tg * factor for tg in total_gains]
                bess_per_ev = [tg / n if n > 0 else 0 for tg, n in zip(bess_total_gains, n_ev_list)]
                fig_bess.add_trace(go.Scatter(x=n_ev_list, y=bess_per_ev, mode='lines+markers', name=bess_labels[i], line=dict(width=3, dash='solid' if i==1 else 'dot'), marker=dict(size=8)))
            fig_bess.add_vline(x=current_view_n_ev, line_dash="dash", line_color="#FF9100", annotation_text=f"当前设定", annotation_position="top right")
            fig_bess.update_layout(template=PLOTLY_TEMPLATE, paper_bgcolor="rgba(0,0,0,0)", plot_bgcolor="rgba(0,0,0,0)", xaxis_title="参与 V2G 的 EV 数量 (辆)", yaxis_title="单车日均收益 (元/辆·天)", margin=dict(t=40, b=20, l=20, r=20), xaxis=dict(gridcolor="rgba(255,255,255,0.05)"), yaxis=dict(gridcolor="rgba(255,255,255,0.05)"), legend=dict(orientation="h", yanchor="bottom", y=1.02, xanchor="center", x=0.5))
            st.plotly_chart(fig_bess, use_container_width=True)
            if current_per_ev < 1:
                st.warning(f"⚠️ **运营建议**：当前 **V2G 并没有产生额外正收益**。建议调大 EV 规模或放宽放电限制。")
            elif current_view_n_ev > n_ev_list[-2]: 
                st.warning(f"⚠️ **运营建议**：当前设定 {current_view_n_ev} 辆 EV，单车日均收益已降至 **{current_per_ev:.1f} 元**。建议精准筛选高意愿车主参与。")
            else:
                st.success(f"✅ **运营建议**：当前设定 {current_view_n_ev} 辆 EV 处于边际效益较优的区间，单车日均收益约 **{current_per_ev:.1f} 元**。")

    # ---------- Tab 3: 日度成本对比 ----------
    with tab3:
        with st.container(border=True):
            st.markdown("#### 📊 各场景日度运行总成本对比")
            df_daily = res["daily_df"].copy().reset_index()
            if "Day" not in df_daily.columns: df_daily = df_daily.rename(columns={df_daily.columns[0]: "Day"})
            valid_cols = [c for c in ["A1", "A2", "A3", "A4"] if c in df_daily.columns]
            fig_daily = go.Figure()
            for i, col in enumerate(valid_cols):
                fig_daily.add_trace(go.Bar(x=df_daily["Day"], y=df_daily[col], name=SCENARIO_LABELS.get(col, col), marker_color=FUTURE_COLORS[i], text=df_daily[col].round(0), textposition='outside', textfont=dict(size=11, color='rgba(255,255,255,0.7)')))
            fig_daily.update_layout(barmode='group', template=PLOTLY_TEMPLATE, paper_bgcolor="rgba(0,0,0,0)", plot_bgcolor="rgba(0,0,0,0)", margin=dict(t=60, b=20, l=20, r=20), xaxis=dict(title="天数 (Day)", gridcolor="rgba(255,255,255,0.05)", tickmode='linear', dtick=1), yaxis=dict(title="日总成本 (元)", gridcolor="rgba(255,255,255,0.05)"), legend=dict(orientation="h", yanchor="bottom", y=1.05, xanchor="center", x=0.5))
            st.plotly_chart(fig_daily, use_container_width=True)

            st.write("---") 
            st.subheader("🔍 V2G (A4) 单日成本账单拆解")
            df_a4 = res["df_timeseries"][res["df_timeseries"]["Scenario"] == "A4"].copy()
            if not df_a4.empty:
                sell_ratio_used = res.get("used_sell_ratio", 0.60)
                df_a4["Buy_Cost"] = df_a4["Grid_Buy"] * df_a4["Price"] * 0.5
                df_a4["Sell_Revenue"] = df_a4["Grid_Sell"] * df_a4["Price"] * sell_ratio_used * 0.5
                daily_bill = df_a4.groupby("Day")[["Buy_Cost", "Sell_Revenue"]].sum().reset_index()
                total_sell = daily_bill["Sell_Revenue"].sum()
                if total_sell < 1.0:
                    st.info("💡 **提示**：当前微网处于极度缺电状态，V2G车辆主要通过“削峰填谷”减少买电，并未产生多余电量向电网售电。")
                else:
                    fig_bill = go.Figure()
                    fig_bill.add_trace(go.Bar(x=daily_bill["Day"], y=daily_bill["Buy_Cost"], name="向电网买电花费", marker_color="#FF9100"))
                    fig_bill.add_trace(go.Bar(x=daily_bill["Day"], y=-daily_bill["Sell_Revenue"], name="向电网卖电赚钱", marker_color="#00E676"))
                    fig_bill.update_layout(barmode='relative', template=PLOTLY_TEMPLATE, paper_bgcolor="rgba(0,0,0,0)", plot_bgcolor="rgba(0,0,0,0)", margin=dict(t=40, b=20, l=20, r=20), xaxis=dict(title="天数 (Day)", tickmode='linear', dtick=1, gridcolor="rgba(255,255,255,0.05)"), yaxis=dict(title="金额 (元)", gridcolor="rgba(255,255,255,0.05)"), legend=dict(orientation="h", yanchor="bottom", y=1.02, xanchor="center", x=0.5))
                    st.plotly_chart(fig_bill, use_container_width=True)
                    st.success(f"✅ 橙色柱子减去绿色柱子，剩余的高度差就是每天真实掏的钱。仿真期间 V2G 卖电共为你赚回了 **{total_sell:.0f} 元**。")

    # ---------- Tab 4: ⚡ 电网冲击分析 ----------
    with tab_impact:
        with st.container(border=True):
            st.markdown("#### ⚡ 净负荷曲线对比 (削峰填谷效果)")
            st.write("微网的净负荷 = 电网购电 - 电网售电。**将鼠标悬停在曲线上，可查看该时间点的智能成因解说。**")
            df_ts = res["df_timeseries"]
            t_day_impact = df_ts["Day"].unique()[0]
            df_day_impact = df_ts[df_ts["Day"] == t_day_impact].copy()
            df_day_impact["Net_Load"] = df_day_impact["Grid_Buy"] - df_day_impact["Grid_Sell"]
            
            fig_net = go.Figure()
            df_load = df_day_impact[df_day_impact["Scenario"] == "A1"]
            if not df_load.empty:
                fig_net.add_trace(go.Scatter(x=df_load["Hour"], y=df_load["Load"], name="物理基础负荷", mode="lines", line=dict(color="rgba(255,255,255,0.15)", width=2, dash="dash"), hovertemplate="时间: %{x}h<br>物理负荷: %{y:.0f} kW<extra></extra>"))
            df_a2 = df_day_impact[df_day_impact["Scenario"] == "A2"]
            fig_net.add_trace(go.Scatter(x=df_a2["Hour"], y=df_a2["Net_Load"], name="A2: 无序充电", mode="lines", line=dict(color="#FF4B4B", width=2, dash='dot'), hovertemplate="时间: %{x}h<br>净负荷: %{y:.0f} kW<br><b>成因：</b>EV下班即插即充，与晚高峰完美叠加。<extra></extra>"))
            df_a3 = df_day_impact[df_day_impact["Scenario"] == "A3"]
            fig_net.add_trace(go.Scatter(x=df_a3["Hour"], y=df_a3["Net_Load"], name="A3: 智能充电", mode="lines", line=dict(color="#FFA500", width=2, dash='dash'), hovertemplate="时间: %{x}h<br>净负荷: %{y:.0f} kW<br><b>成因：</b>算法将充电负荷平移至低谷，避开晚高峰。<extra></extra>"))
            
            df_a4 = df_day_impact[df_day_impact["Scenario"] == "A4"].copy()
            custom_hover_texts = []
            for idx, row in df_a4.iterrows():
                hour, net, load, ev_dis, ev_chg, price = row["Hour"], row["Net_Load"], row["Load"], row["EV_Discharge"], row["EV_Charge"], row["Price"]
                info = f"时间: {hour:.1f}h<br>净负荷: {net:.0f} kW<br>电价: {price:.2f} 元<br>"
                if 0 <= hour < 8 and net > load * 1.05: info += "🚨 <b>异常激增：</b>算法在低价谷时段指令 EV 集中满功率充电囤电，导致净负荷远高于基础负荷。"
                elif 17 <= hour < 22 and ev_dis > 10: info += "✅ <b>有效削峰：</b>电价进入高峰期，V2G 车辆大量反向放电，成功替代了电网购电。"
                elif 17 <= hour < 22 and net > load: info += "⚠️ <b>削峰不足：</b>晚高峰期间，EV 放电量较小，未能有效压平尖峰。"
                elif abs(net - load) < 50: info += "⚖️ <b>平稳运行：</b>当前时段 V2G 充放电基本平衡。"
                else: info += f"状态说明：EV 充电 {ev_chg:.0f} kW, 放电 {ev_dis:.0f} kW。"
                custom_hover_texts.append(info)
            df_a4["HoverText"] = custom_hover_texts
            fig_net.add_trace(go.Scatter(x=df_a4["Hour"], y=df_a4["Net_Load"], name="A4: V2G 调度", mode="lines", line=dict(color="#00E5FF", width=3, dash='solid'), customdata=df_a4["HoverText"], hovertemplate="%{customdata}<extra></extra>"))
            fig_net.update_layout(title=f"Day {t_day_impact} 净负荷时序曲线与成因溯源", template=PLOTLY_TEMPLATE, paper_bgcolor="rgba(0,0,0,0)", plot_bgcolor="rgba(0,0,0,0)", xaxis=dict(title="时间 (小时)", gridcolor="rgba(255,255,255,0.05)", dtick=4), yaxis=dict(title="净负荷 (kW)", gridcolor="rgba(255,255,255,0.05)"), hovermode="closest", legend=dict(orientation="h", yanchor="bottom", y=1.02, xanchor="center", x=0.5), margin=dict(l=20, r=20, t=50, b=20))
            st.plotly_chart(fig_net, use_container_width=True)

            st.write("---")
            st.markdown("#### 📊 冲击指标相对改善率 (以 A2 为 100% 基准)")
            st.write("图表 Y 轴已锁定在 **80%~110%** 区间，以便清晰对比微小的指标波动。柱子低于 100% 红线代表对电网有改善。")
            df_impact = res["df_impact"].copy()
            base_metrics = ["peak_kw", "p2v_kw", "std_kw"]
            base_values = df_impact.loc["A2", base_metrics]
            for m in base_metrics: df_impact[f"{m}_pct"] = (df_impact[m] / base_values[m]) * 100
            metrics_map = {"peak_kw_pct": "最大峰值负荷", "p2v_kw_pct": "峰谷差", "std_kw_pct": "负荷波动标准差"}
            fig_impact = go.Figure()
            for i, (col, name) in enumerate(metrics_map.items()):
                fig_impact.add_trace(go.Bar(x=[SCENARIO_LABELS.get(s, s) for s in df_impact.index], y=df_impact[col], name=name, marker_color=FUTURE_COLORS[i], text=df_impact[col].apply(lambda x: f"{x:.1f}%"), textposition='outside', textfont=dict(size=12, color='white')))
            fig_impact.add_hline(y=100, line_dash="solid", line_color="#FF4B4B", line_width=2, annotation_text="A2 基准线 (100%)", annotation_position="top right", annotation_font_color="#FF4B4B")
            fig_impact.update_layout(barmode='group', template=PLOTLY_TEMPLATE, paper_bgcolor="rgba(0,0,0,0)", plot_bgcolor="rgba(0,0,0,0)", margin=dict(t=60, b=20, l=20, r=20), yaxis=dict(title="相对冲击比例", gridcolor="rgba(255,255,255,0.05)", range=[80, 110]), xaxis=dict(title="调度场景"), legend=dict(orientation="h", yanchor="bottom", y=1.05, xanchor="center", x=0.5))
            st.plotly_chart(fig_impact, use_container_width=True)

            # ★ 全天候多维诊断专家系统
            st.markdown("#### 🧠 净负荷曲线深度诊断与调度建议")
            advice_html = "<div class='advice-box'><h4>📊 曲线异常特征全景解析：</h4><ul>"
            peak_hours = [17, 18, 19, 20, 21]
            net_a2_peak = df_day_impact[df_day_impact["Hour"].isin(peak_hours) & (df_day_impact["Scenario"] == "A2")]["Net_Load"].max()
            net_a4_peak = df_day_impact[df_day_impact["Hour"].isin(peak_hours) & (df_day_impact["Scenario"] == "A4")]["Net_Load"].max()
            morning_hours = [0, 1, 2, 3, 4, 5, 6, 7]
            max_a1_morn = df_day_impact[df_day_impact["Hour"].isin(morning_hours) & (df_day_impact["Scenario"] == "A1")]["Net_Load"].max()
            max_a4_morn = df_day_impact[df_day_impact["Hour"].isin(morning_hours) & (df_day_impact["Scenario"] == "A4")]["Net_Load"].max()
            noon_hours = [12, 13, 14, 15]
            max_a1_noon = df_day_impact[df_day_impact["Hour"].isin(noon_hours) & (df_day_impact["Scenario"] == "A1")]["Net_Load"].max()
            max_a4_noon = df_day_impact[df_day_impact["Hour"].isin(noon_hours) & (df_day_impact["Scenario"] == "A4")]["Net_Load"].max()
            diff_series_a4 = df_day_impact[df_day_impact["Scenario"] == "A4"]["Net_Load"].diff().abs()
            max_ramp_a4 = diff_series_a4.max()
            net_a3_peak_morning = df_day_impact[df_day_impact["Hour"].isin([7,8,9]) & (df_day_impact["Scenario"] == "A3")]["Net_Load"].max()

            if net_a4_peak < net_a2_peak:
                reduction = (net_a2_peak - net_a4_peak) / net_a2_peak * 100
                advice_html += f"<li><b>✅ 晚高峰削峰有效：</b>A4 成功将 17:00-22:00 的净负荷峰值从 A2 的 {net_a2_peak:.0f} kW 压降至 {net_a4_peak:.0f} kW（降幅 {reduction:.1f}%），避免了变压器严重过载。</li>"
            else: advice_html += f"<li><b>⚠️ 晚高峰削峰失效：</b>A4 并未有效压制 17:00-22:00 的净负荷峰值（反而达到 {net_a4_peak:.0f} kW）。</li>"
            if max_a4_morn > max_a1_morn * 1.02: 
                diff_morn = max_a4_morn - max_a1_morn
                advice_html += f"<li><b>⚠️ 凌晨/早高峰异常凸起（过度套利）：</b>在 0:00-8:00 期间，A4 的最大净负荷达到了 {max_a4_morn:.0f} kW，高出基础负荷峰值 {diff_morn:.0f} kW。图中可见明显尖峰，说明算法在低价时段集中充电囤电，反而给电网增加了不必要的压力。</li>"
            if max_a4_noon > max_a1_noon * 1.02:
                advice_html += f"<li><b>⚠️ 午后时段负荷抬升：</b>在 12:00-16:00 期间，A4 的净负荷峰值达到了 {max_a4_noon:.0f} kW，超过了基础负荷。算法可能在此平段继续激进充电，破坏了平滑性。</li>"
            if max_ramp_a4 > 300:
                advice_html += f"<li><b>⚠️ 剧烈功率激变（锯齿效应）：</b>A4 的相邻时段最大功率变化率达到了 {max_ramp_a4:.0f} kW/半小时。图中在早间等位置频繁出现陡升陡降的尖峰，对电网造成高频冲击。</li>"
            if net_a3_peak_morning > max_a1_morn * 1.05:
                advice_html += f"<li><b>🔍 A3 策略的隐患：</b>A3 虽避开了晚高峰，但将负荷平移到早高峰，导致 7:00-9:00 出现了新的尖峰（达到 {net_a3_peak_morning:.0f} kW）。单纯平移无法解决整体承载力问题。</li>"
            advice_html += "</ul></div>"
            st.markdown(advice_html, unsafe_allow_html=True)

            dynamic_cmds = []
            if max_a4_morn > max_a1_morn * 1.02 or max_a4_noon > max_a1_noon * 1.02:
                dynamic_cmds.append("- **抑制过度套利（平抑早间/午后凸起）**：建议将侧边栏的 **“售电比例”降低至 0.3~0.4**。削减 V2G 在低价时段的超额收益预期，强制减少无效囤电导致的负荷激增。")
            if max_ramp_a4 > 300:
                dynamic_cmds.append("- **消除高频锯齿（约束爬坡率）**：建议在底层 MILP 算法中**引入“负荷波动平滑惩罚项 (Smoothing Penalty)”**，或限制充放电爬坡率硬约束，禁止车辆“满充满放”。")
            if net_a4_peak >= net_a2_peak:
                dynamic_cmds.append("- **强化晚高峰削峰**：提高晚高峰的放电激励，或检查变压器容量限制（C4）是否过严，导致车辆无法在高负荷期出力。")
            if max_a1_morn * 1.05 < net_a3_peak_morning:
                dynamic_cmds.append("- **修正 A3 平移策略**：智能充电不能仅避开晚高峰，需设定全天候的负荷上限（如不超过基础负荷的 1.1 倍）。")
            if dynamic_cmds:
                with st.expander("🛠️ 智能诊断触发的调度指令修正建议", expanded=True): st.markdown("\n".join(dynamic_cmds))
            else: st.success("✅ **系统运行完美**：当前参数配置极佳，各项冲击指标均处于优秀区间，无需生成额外修正指令。")

    # ---------- Tab 5: 收益分配与投资 ----------
    with tab4:
        with st.container(border=True):
            st.markdown("#### 💰 V2G 总收益分红与投资回收分析")
            
            df_a4 = res["df_timeseries"][res["df_timeseries"]["Scenario"] == "A4"]
            sell_ratio_used = res.get("used_sell_ratio", 0.60)
            total_sell = (df_a4["Grid_Sell"] * df_a4["Price"] * sell_ratio_used * 0.5).sum()
            total_gain = res['total_gain']
            
            # 插入投资回收期 KPI 卡片
            total_infra = n_ev * res.get("used_infra_cost", 2000)
            c_kpi1, c_kpi2, c_kpi3 = st.columns(3)
            c_kpi1.metric("双向设备总改造成本", f"{total_infra:.0f} 元")
            c_kpi2.metric("日均 V2G 套利净收益", f"{daily_net_gain:.0f} 元/天")
            
            if payback_days < 365:
                c_kpi3.metric("设备投资静态回收期", f"{payback_days:.0f} 天", delta="极优", delta_color="normal")
            elif payback_days < 3650:
                c_kpi3.metric("设备投资静态回收期", f"{payback_days/365:.1f} 年", delta="良好", delta_color="normal")
            else:
                c_kpi3.metric("设备投资静态回收期", f"> 10 年", delta="盈利困难", delta_color="inverse")
            
            st.write("---")
            
            if total_sell > 1.0:
                total_arbitrage = max(total_gain - total_sell, 0)
                sun_data = []
                for _, row in res["share_df"].iterrows():
                    role = row["角色"]
                    prop = row["比例"]
                    if total_arbitrage > 0:
                        sun_data.append({"总收益": "总收益", "收益来源": "削峰填谷降本", "角色": role, "金额": total_arbitrage * prop})
                    sun_data.append({"总收益": "总收益", "收益来源": "向电网卖电", "角色": role, "金额": total_sell * prop})
                df_sun = pd.DataFrame(sun_data)
                fig_pie = px.sunburst(df_sun, path=['总收益', '收益来源', '角色'], values='金额', color='收益来源', color_discrete_map={"削峰填谷降本": "#00E676", "向电网卖电": "#00E5FF"}, template=PLOTLY_TEMPLATE)
                fig_pie.update_layout(margin=dict(t=40, b=10, l=10, r=10), paper_bgcolor="rgba(0,0,0,0)")
                st.plotly_chart(fig_pie, use_container_width=True)
                st.info(f"💡 **收益拆分说明：** 在总计 **{total_gain:.0f} 元** 的收益中，**{total_arbitrage:.0f} 元** 来自于“削峰填谷降本”，**{total_sell:.0f} 元** 来自于“向电网卖电”。")
            else:
                share_df = res["share_df"].copy()
                share_df["金额(元)"] = share_df["比例"] * total_gain
                c1, c2, c3 = st.columns([1, 2, 1])
                with c2:
                    fig_pie = px.pie(share_df, values='金额(元)', names='角色', title=f"削峰填谷降本分红 (总计 {total_gain:.0f} 元)", color_discrete_sequence=FUTURE_COLORS, hole=0.5)
                    fig_pie.update_traces(textposition='inside', texttemplate='<b>%{label}</b><br>%{value:,.0f} 元<br>(%{percent})', textfont_size=14, marker=dict(line=dict(color='#0A0F1A', width=2)))
                    fig_pie.update_layout(template=PLOTLY_TEMPLATE, paper_bgcolor="rgba(0,0,0,0)", plot_bgcolor="rgba(0,0,0,0)", margin=dict(t=50, b=10, l=10, r=10), showlegend=False)
                    st.plotly_chart(fig_pie, use_container_width=True)
                    st.info("💡 **收益来源说明：** 当前系统无多余电量上网，V2G 的收益主要来自“低谷充电、高峰放电”从而减少高价买电（**少花钱 = 赚钱**）。")

    # ---------- Tab 6: 数据导出 ----------
    with tab5:
        with st.container(border=True):
            st.markdown("#### 📥 下载仿真数据与报告")
            output = io.BytesIO()
            with pd.ExcelWriter(output, engine='xlsxwriter') as writer:
                res["df_timeseries"].to_excel(writer, sheet_name='48时段详细数据', index=False)
                res["daily_df"].to_excel(writer, sheet_name='日度成本汇总')
                res["df_impact"].to_excel(writer, sheet_name='电网冲击指标')
            excel_data = output.getvalue()
            txt_content = f"V2G 仿真汇总\n{'='*30}\n"
            for k, v in totals.items(): txt_content += f"{k} 总成本: {v:.2f} 元\n"
            txt_content += f"\nA3-A4 纯 V2G 收益: {res['delta'][2]:.2f} 元\n"
            col_btn1, col_btn2 = st.columns(2)
            with col_btn1: st.download_button("📊 下载完整数据表 (Excel)", data=excel_data, file_name=f"V2G_仿真结果.xlsx", mime="application/vnd.openxmlformats-officedocument.spreadsheetml.sheet", use_container_width=True, type="primary")
            with col_btn2: st.download_button("📄 下载汇总报告 (TXT)", data=txt_content, file_name=f"V2G_汇总报告.txt", mime="text/plain", use_container_width=True)
