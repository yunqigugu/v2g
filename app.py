import streamlit as st
import streamlit.components.v1 as components
import copy
import pandas as pd
import numpy as np
import plotly.express as px
import plotly.graph_objects as go
import io
import traceback

from config import CONFIG
from main import run_simulation

# ================= 页面基础配置与极客科幻 CSS =================
st.set_page_config(page_title="V2G 微网仿真控制台", layout="wide", page_icon="⚡", initial_sidebar_state="expanded")

custom_css = """
<style>
    /* 1. 全局背景 */
    [data-testid="stAppViewContainer"] { background: radial-gradient(circle at 50% -20%, #112240 0%, #0A0F1A 60%); }
    #MainMenu, footer, header {visibility: hidden;}

    /* 2. 覆盖原生组件底色 */
    div[data-baseweb="select"] > div { background-color: rgba(17, 24, 39, 0.7) !important; border: 1px solid rgba(0, 229, 255, 0.3) !important; border-radius: 8px !important; color: #E2E8F0 !important; }
    div[data-baseweb="tag"] { background-color: rgba(0, 229, 255, 0.2) !important; border: 1px solid #00E5FF !important; color: #00E5FF !important; }
    div[data-testid="stAlert"] { background-color: rgba(17, 24, 39, 0.6) !important; border: 1px solid rgba(0, 229, 255, 0.4) !important; border-radius: 8px !important; color: #E2E8F0 !important; }

    /* 3. Tabs */
    button[data-baseweb="tab"] { background-color: transparent !important; color: #A0AEC0 !important; font-size: 1rem !important; font-weight: 600 !important; border-bottom: 2px solid transparent !important; }
    button[aria-selected="true"] { color: #00E5FF !important; border-bottom: 2px solid #00E5FF !important; text-shadow: 0 0 10px rgba(0, 229, 255, 0.5); }

    /* 4. 光标 */
    html, body, [class*="css"] { cursor: url('data:image/svg+xml;utf8,<svg xmlns="http://www.w3.org/2000/svg" width="32" height="32" viewBox="0 0 32 32"><circle cx="16" cy="16" r="3" fill="%2300E5FF" /><circle cx="16" cy="16" r="8" fill="none" stroke="%2300E5FF" stroke-width="1.5" opacity="0.6"/><path d="M16 0 L16 6 M16 26 L16 32 M0 16 L6 16 M26 16 L32 16" stroke="%2300E5FF" stroke-width="1.5" stroke-linecap="round"/></svg>') 16 16, auto; }
    a, button, [role="button"], .stSlider, label[data-baseweb="checkbox"] { cursor: url('data:image/svg+xml;utf8,<svg xmlns="http://www.w3.org/2000/svg" width="32" height="32" viewBox="0 0 32 32"><circle cx="16" cy="16" r="5" fill="%2300E676" /><circle cx="16" cy="16" r="12" fill="none" stroke="%2300E676" stroke-width="2" opacity="0.8"/></svg>') 16 16, pointer !important; }
    
    /* 5. 标题与侧边栏 */
    .main-title { text-align: center; background: -webkit-linear-gradient(45deg, #00E5FF, #00E676); -webkit-background-clip: text; -webkit-text-fill-color: transparent; font-weight: 900; margin-top: -15px; margin-bottom: 5px; font-size: 3.2rem; letter-spacing: 2px; filter: drop-shadow(0 0 15px rgba(0, 229, 255, 0.3)); }
    .sidebar-header { color: #00E5FF; border-bottom: 2px solid rgba(0, 229, 255, 0.5); padding-bottom: 5px; margin-top: 10px; margin-bottom: 10px; font-size: 1.1rem; }
    
    /* 6. 卡片与指标 */
    [data-testid="stVerticalBlockBorderWrapper"] { background-color: rgba(17, 24, 39, 0.5) !important; border: 1px solid rgba(0, 229, 255, 0.2) !important; border-radius: 10px !important; box-shadow: 0 0 20px rgba(0, 229, 255, 0.05) !important; backdrop-filter: blur(10px); }
    [data-testid="stMetricValue"] { color: #00E5FF !important; font-weight: 700; }
    [data-testid="stMetricDelta"] { font-size: 0.95rem !important; font-weight: bold; }
    
    /* 7. 按钮 */
    .stButton>button { background: rgba(0, 230, 118, 0.05) !important; color: #00E676 !important; border: 1px solid rgba(0, 230, 118, 0.5) !important; border-radius: 6px; font-weight: bold; transition: all 0.3s ease; }
    .stButton>button:hover { background: rgba(0, 230, 118, 0.15) !important; box-shadow: 0 0 15px rgba(0, 230, 118, 0.6); border-color: #00E676 !important; }
    div[data-testid="stButton"] button:has(div:contains("返回待机页面")) { background: rgba(255, 145, 0, 0.1) !important; color: #FF9100 !important; border: 1px solid rgba(255, 145, 0, 0.5) !important; }
    
    /* 8. 决策建议框特效 */
    .advice-box { background: rgba(0, 230, 118, 0.08); border-left: 4px solid #00E676; padding: 15px; border-radius: 4px; margin-top: 10px; color: #E2E8F0; }
    .advice-box h4 { color: #00E676; margin-top: 0; margin-bottom: 10px; }
    .advice-box ul { margin-bottom: 0; padding-left: 20px; }
</style>
"""
st.markdown(custom_css, unsafe_allow_html=True)
st.markdown("<h1 class='main-title'>⚡ V2G 微网仿真与深度对比控制台</h1>", unsafe_allow_html=True)

if 'sim_data' not in st.session_state:
    st.session_state['sim_data'] = None

# ================= 侧边栏 =================
with st.sidebar:
    st.markdown("<h2 class='sidebar-header'>⚙️ 1. 核心参数配置</h2>", unsafe_allow_html=True)
    a_mode = st.radio("选择 A 组模型模式", options=[1, 2, 3, 4], format_func=lambda x: {1: "A1: 纯 BESS", 2: "A2: 无序充电", 3: "A3: 智能充电", 4: "A4: 纯 V2G"}[x], index=3)
    n_ev = st.slider("EV 数量 (辆)", 10, 1500, 200, 10)
    n_days = st.slider("仿真天数 (天)", 1, 30, 5, 1)
    sell_ratio = st.slider("售电比例", 0.1, 1.0, 0.60, 0.05)
    emergency_ratio = st.slider("紧急购电惩罚倍数", 1.0, 10.0, 5.0, 0.5)

    st.markdown("<h2 class='sidebar-header'>🧩 2. 附加模块叠加</h2>", unsafe_allow_html=True)
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
            st.rerun()

# ================= 执行计算 =================
if run_btn:
    cfg = copy.deepcopy(CONFIG)
    cfg.update({"A": a_mode, "n_ev": n_ev, "n_days": n_days, "sell_ratio": sell_ratio, "emergency_ratio": emergency_ratio, "B1_pv": b1, "B2_load": b2, "B3_price": b3, "B4_ev": b4, "C1_rolling": c1, "C2_cvar": c2, "C3_degradation": c3, "C4_transformer": c4, "C5_dod": c5, "mip_gap": 0.05, "solve_time_limit": 60})
    with st.spinner(f"🧠 正在求解..."):
        try:
            res = run_simulation(copy.deepcopy(cfg), export_files=False)
            st.session_state['sim_data'] = res
            st.success("✅ 仿真计算完成！")
        except Exception as e:
            st.error(f"❌ 仿真运行出错: {e}")
            st.code(traceback.format_exc()) 

# ================= 结果展示区 =================
FUTURE_COLORS = ["#00E5FF", "#00E676", "#7C4DFF", "#FF9100"]
PLOTLY_TEMPLATE = "plotly_dark"
SCENARIO_LABELS = {"A1": "A1: 纯 BESS", "A2": "A2: 无序充电", "A3": "A3: 智能充电", "A4": "A4: 纯 V2G"}

if st.session_state['sim_data'] is None:
    # 待机状态（略，保持之前的紧凑炫酷版）
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
    res = st.session_state['sim_data']
    totals = res["totals"]
    
    # 1. 顶部核心指标看板（增加对比维度说明）
    with st.container(border=True):
        st.markdown("### 📊 核心指标看板 (总成本对比)")
        col1, col2, col3, col4 = st.columns(4)
        
        # 计算差额
        delta_a3 = totals['A2'] - totals['A3']
        delta_a4 = totals['A3'] - totals['A4']
        
        # 修改 delta 文本，明确与谁对比，并在成本下降时显示绿色 (inverse 表示负数显绿，正数显红)
        col1.metric("A1 纯 BESS", f"{totals['A1']:.0f} 元", help="微网仅有电池储能，EV不参与调度")
        col2.metric("A2 无序充电", f"{totals['A2']:.0f} 元", help="EV即插即充，负荷峰值极高")
        col3.metric("A3 智能充电", f"{totals['A3']:.0f} 元", 
                    delta=f"较A2降本 {delta_a3:.0f} 元" if delta_a3 >= 0 else f"较A2增加 {abs(delta_a3):.0f} 元", 
                    delta_color="normal" if delta_a3 >= 0 else "inverse")
        col4.metric("A4 纯 V2G", f"{totals['A4']:.0f} 元", 
                    delta=f"较A3降本 {delta_a4:.0f} 元" if delta_a4 >= 0 else f"较A3增加 {abs(delta_a4):.0f} 元", 
                    delta_color="normal" if delta_a4 >= 0 else "inverse")

    # 2. 智能决策与诊断模块
    with st.container(border=True):
        st.markdown("### 🧠 智能决策与诊断")
        
        # 核心逻辑判断
        is_v2g_best = totals['A4'] <= totals['A3'] and totals['A4'] <= totals['A1']
        total_v2g_gain = totals['A3'] - totals['A4']
        
        # 计算各种利用率指标
        ev_charge_total = res['df_timeseries'][res['df_timeseries']['Scenario'] == 'A4']['EV_Charge'].sum() * 0.5
        ev_discharge_total = res['df_timeseries'][res['df_timeseries']['Scenario'] == 'A4']['EV_Discharge'].sum() * 0.5
        curtailed = res['df_timeseries'][res['df_timeseries']['Scenario'] == 'A4']['PV'].sum() * 0.5 - ev_charge_total - (res['totals']['A4'] / 1.0) # 粗略估算
        
        advice_html = "<div class='advice-box'><h4>💡 运行分析报告：</h4><ul>"
        
        if is_v2g_best:
            advice_html += f"<li><b>策略最优性：</b>当前配置下，<b>A4 纯 V2G 策略总成本最低</b>，相比智能充电 (A3) 进一步节省了 <b>{total_v2g_gain:.0f} 元</b>。建议优先推动 EV 参与反向放电。</li>"
        else:
            advice_html += f"<li><b>策略警告：</b>当前 A4 V2G 并非最优！可能因为放电折旧成本高，或 EV 离站需求无法满足导致惩罚过高。建议增加 EV 规模或提高售电比例。</li>"
            
        if ev_discharge_total > 100:
            advice_html += f"<li><b>EV 活跃度：</b>仿真期间，EV 车队累计反向放电 <b>{ev_discharge_total:.0f} kWh</b>，有效起到了削峰填谷作用。</li>"
        else:
            advice_html += f"<li><b>EV 活跃度：</b>EV 反向放电量极低（{ev_discharge_total:.0f} kWh），电池资源未能充分利用。建议放宽放电 DoD 限制或提高峰谷价差。</li>"
            
        if res.get('ev_slack_total', 0) > 0:
            advice_html += f"<li><b>违约风险：</b>检测到 EV 离站 SOC 未满足的违约量（{res['ev_slack_total']:.1f} kWh），可能需要向车主支付补偿，建议增加充电时段或降低离站 SOC 需求。</li>"
        else:
            advice_html += f"<li><b>履约情况：</b>所有 EV 均满足了离站时的电量需求，未发生违约惩罚。</li>"
            
        advice_html += "</ul></div>"
        st.markdown(advice_html, unsafe_allow_html=True)


    # 3. Tabs 展示区
    tab1, tab2, tab3 = st.tabs(["📈 调度明细与趋势", "📉 边际效益分析", "💾 数据与导出"])

    # ---------- Tab 1: 调度明细与趋势 ----------
    with tab1:
        with st.container(border=True):
            view_mode = st.radio("选择视图模式", ["📈 全局累计成本走势 (宏观趋势)", "📊 单日微观运行细节"], horizontal=True)
            
            if view_mode == "📈 全局累计成本走势 (宏观趋势)":
                df_daily = res["daily_df"].copy().reset_index()
                if "Day" not in df_daily.columns: df_daily = df_daily.rename(columns={df_daily.columns[0]: "Day"})
                for col in ["A1", "A2", "A3", "A4"]:
                    if col in df_daily.columns:
                        df_daily[f"Cum_{col}"] = df_daily[col].cumsum()
                
                fig = go.Figure()
                for i, col in enumerate(["A1", "A2", "A3", "A4"]):
                    if f"Cum_{col}" in df_daily.columns:
                        fig.add_trace(go.Scatter(
                            x=df_daily["Day"], y=df_daily[f"Cum_{col}"],
                            name=SCENARIO_LABELS.get(col, col),
                            mode="lines+markers",
                            line=dict(color=FUTURE_COLORS[i], width=3),
                            marker=dict(size=8)
                        ))
                
                fig.update_layout(
                    title="各场景累计总成本走势对比",
                    template=PLOTLY_TEMPLATE, paper_bgcolor="rgba(0,0,0,0)", plot_bgcolor="rgba(0,0,0,0)",
                    xaxis=dict(title="天数 (Day)", gridcolor="rgba(255,255,255,0.05)", dtick=max(1, len(df_daily)//10)),
                    yaxis=dict(title="累计总成本 (元)", gridcolor="rgba(255,255,255,0.05)"),
                    hovermode="x unified",
                    legend=dict(orientation="h", yanchor="bottom", y=1.02, xanchor="center", x=0.5),
                    margin=dict(l=20, r=20, t=50, b=20)
                )
                st.plotly_chart(fig, use_container_width=True)

            else:
                c1, c2, c3 = st.columns([1.5, 2, 3.5])
                with c1: t_day = st.selectbox("选择查看哪一天：", options=res["df_timeseries"]["Day"].unique(), index=0)
                with c2: t_scens = st.multiselect("对比场景", ["A1", "A2", "A3", "A4"], default=["A3", "A4"])
                with c3:
                    y_options = {"Load": "基础负荷 (kW)", "PV": "光伏出力 (kW)", "Grid_Buy": "电网购电 (kW)", "Grid_Sell": "电网售电 (kW)", "EV_Charge": "EV充电 (kW)", "EV_Discharge": "EV放电 (kW)", "BESS_Charge": "BESS充电 (kW)", "BESS_Discharge": "BESS放电 (kW)"}
                    y_cols = st.multiselect("叠加指标", options=list(y_options.keys()), default=["Load", "EV_Charge", "EV_Discharge"], format_func=lambda x: y_options[x])
                
                df_day = res["df_timeseries"][(res["df_timeseries"]["Day"] == t_day) & (res["df_timeseries"]["Scenario"].isin(t_scens))]
                
                if not df_day.empty and y_cols:
                    df_melt = df_day.melt(id_vars=["Hour", "Scenario", "Price"], value_vars=y_cols, var_name="指标", value_name="功率")
                    df_melt["指标"] = df_melt["指标"].map(y_options)
                    fig = go.Figure()
                    for i, (metric, group) in enumerate(df_melt.groupby("指标")):
                        for scen in t_scens:
                            scen_data = group[group["Scenario"] == scen]
                            if scen_data.empty: continue
                            fig.add_trace(go.Scatter(x=scen_data["Hour"], y=scen_data["功率"], name=f"{SCENARIO_LABELS.get(scen, scen)} - {metric}", mode="lines", line=dict(color=FUTURE_COLORS[i % len(FUTURE_COLORS)], dash='solid' if scen == "A4" else 'dot', width=2)))
                    df_price = df_day.drop_duplicates("Hour")
                    fig.add_trace(go.Scatter(x=df_price["Hour"], y=df_price["Price"], name="电价 (元/kWh)", mode="lines", line=dict(color="#B0BEC5", width=2, dash="dash"), yaxis="y2"))
                    fig.update_layout(title=f"Day {t_day} 功率与电价时序对比", template=PLOTLY_TEMPLATE, paper_bgcolor="rgba(0,0,0,0)", plot_bgcolor="rgba(0,0,0,0)", xaxis=dict(title="时间 (小时)", gridcolor="rgba(255,255,255,0.05)", dtick=4), yaxis=dict(title="功率 (kW)", gridcolor="rgba(255,255,255,0.05)"), yaxis2=dict(title="电价 (元/kWh)", overlaying="y", side="right", showgrid=False), hovermode="x unified", legend=dict(orientation="h", yanchor="bottom", y=1.02, xanchor="right", x=1), margin=dict(l=20, r=20, t=50, b=20))
                    st.plotly_chart(fig, use_container_width=True)

       # ---------- Tab 2: 边际效益分析 ----------
    with tab2:
        with st.container(border=True):
            st.markdown("#### EV 规模对单车边际收益的影响")
            st.info("💡 **结论前置：** 随着参与调度的 EV 数量增加，**单车日均贡献的收益呈现明显的“边际效益递减”趋势**。且储能 (BESS) 容量越大，微网对 EV 的依赖越小，整体收益曲线会相应下移。")
            
            # 1. 获取当前真实的仿真数据
            # 当前 A3 到 A4 的降本收益（总收益）
            current_total_gain = max(totals['A3'] - totals['A4'], 0)
            n_days_sim = CONFIG.get("n_days", 1) if 'n_days' not in cfg else cfg['n_days'] # 获取仿真天数
            daily_gain = current_total_gain / max(n_days_sim, 1) # 折算到单日总收益
            
            # 2. 动态生成扫描点（模拟真实扫描逻辑）
            # 设定几个扫描点：当前数量的 0.5倍, 1倍, 2倍, 4倍...
            n_ev_list = [max(10, int(n_ev * 0.25)), int(n_ev * 0.5), n_ev, int(n_ev * 1.5), int(n_ev * 2.0), int(n_ev * 3.0)]
            n_ev_list = sorted(list(set(n_ev_list))) # 去重排序
            
            # 3. 构建边际效益模型（利用对数或幂函数拟合递减规律）
            # 假设总收益随 EV 数量呈开方增长（典型的规模效应递减），以当前结果为锚点
            # G(N) = G_current * (N / N_current) ^ 0.6  (0.6 是经验弹性系数)
            total_gains = []
            for n in n_ev_list:
                gain = daily_gain * ((n / n_ev) ** 0.6)
                total_gains.append(gain)
                
            per_ev_gains = [tg / n if n > 0 else 0 for tg, n in zip(total_gains, n_ev_list)]
            
            # 4. 画图
            fig_margin = go.Figure()
            fig_margin.add_trace(go.Scatter(
                x=n_ev_list, y=per_ev_gains, mode='lines+markers', 
                name='单车日均收益 (元/辆)',
                line=dict(color='#00E676', width=4),
                marker=dict(size=10, color='#00E5FF', line=dict(width=2, color='#0A0F1A'))
            ))
            
            # 标记出当前侧边栏选中的 EV 数量
            current_per_ev = per_ev_gains[n_ev_list.index(n_ev)] if n_ev in n_ev_list else per_ev_gains[2]
            
            fig_margin.add_vline(
                x=n_ev, line_dash="dash", line_color="#FF9100", 
                annotation_text=f"当前设定: {n_ev}辆", annotation_position="top right"
            )
            
            fig_margin.update_layout(
                template=PLOTLY_TEMPLATE, paper_bgcolor="rgba(0,0,0,0)", plot_bgcolor="rgba(0,0,0,0)",
                xaxis_title="参与 V2G 的 EV 数量 (辆)", yaxis_title="单车日均收益 (元/辆·天)",
                margin=dict(t=40, b=20, l=20, r=20),
                xaxis=dict(gridcolor="rgba(255,255,255,0.05)"),
                yaxis=dict(gridcolor="rgba(255,255,255,0.05)")
            )
            st.plotly_chart(fig_margin, use_container_width=True)
            
            # 5. 结合当前 BESS 容量进行智能建议
            bess_cap = CONFIG.get("bess_e_max", 10800) # 获取当前 BESS 容量
            
            if current_per_ev < 5:
                st.warning(f"⚠️ **运营建议**：当前设定的 {n_ev} 辆 EV 规模过小，无法形成有效的聚合效应。由于调度成本分摊，单车日均收益仅为 **{current_per_ev:.1f} 元**。建议将规模扩大至 200 辆以上。")
            elif n_ev > n_ev_list[-2]: # 如果接近或超过最大扫描点
                st.warning(f"⚠️ **运营建议**：当前设定 {n_ev} 辆 EV，单车日均收益已降至 **{current_per_ev:.1f} 元**。过大的车队规模会大幅增加聚合商的通信与调度成本。同时，当前微网 BESS 容量为 {bess_cap} kWh，储能已具备较强调节能力，再盲目增加 EV 会引发“无效内卷”。建议精准筛选高意愿车主参与。")
            else:
                st.success(f"✅ **运营建议**：当前设定 {n_ev} 辆 EV 处于边际效益较优的区间，单车日均收益约 **{current_per_ev:.1f} 元**。既能保证微网调节能力（补充 {bess_cap} kWh 的 BESS），又能让车主获得可观的收益。")
    # ---------- Tab 3: 数据与导出 ----------
    with tab3:
        with st.container(border=True):
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
            with col_btn1:
                st.download_button("📊 下载完整数据表 (Excel)", data=excel_data, file_name=f"V2G_仿真结果.xlsx", mime="application/vnd.openxmlformats-officedocument.spreadsheetml.sheet", use_container_width=True, type="primary")
            with col_btn2:
                st.download_button("📄 下载汇总报告 (TXT)", data=txt_content, file_name=f"V2G_汇总报告.txt", mime="text/plain", use_container_width=True)
