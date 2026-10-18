import streamlit as st
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
st.set_page_config(page_title="V2G 微网仿真控制台", layout="wide", page_icon="⚡")

custom_css = """
<style>
    /* 1. 隐藏右上角主菜单和底部水印 */
    #MainMenu {visibility: hidden;}
    footer {visibility: hidden;}
    
    /* 2. ★ 科幻十字准星光标 (替换默认鼠标) */
    html, body, [class*="css"] {
        cursor: crosshair;
    }
    
    /* 3. 主标题：青色霓虹发光效果 */
    .main-title {
        text-align: center;
        color: #00E5FF;
        font-weight: 800;
        margin-bottom: 30px;
        text-shadow: 0 0 10px rgba(0, 229, 255, 0.6), 0 0 20px rgba(0, 229, 255, 0.3);
        letter-spacing: 2px;
    }
    
    /* 4. 按钮：节能绿发光 + 悬停动效 */
    .stButton>button {
        background: transparent;
        color: #00E676 !important;
        border: 2px solid #00E676;
        border-radius: 8px;
        font-weight: bold;
        transition: all 0.3s ease;
        box-shadow: 0 0 5px rgba(0, 230, 118, 0.2);
    }
    .stButton>button:hover {
        transform: translateY(-2px);
        background: rgba(0, 230, 118, 0.1);
        box-shadow: 0 0 15px rgba(0, 230, 118, 0.8), inset 0 0 10px rgba(0, 230, 118, 0.4);
        border-color: #00E676;
    }
    
    /* 5. 数据卡片：深蓝底色 + 青色科幻边框 */
    [data-testid="stVerticalBlockBorderWrapper"] {
        background-color: rgba(17, 24, 39, 0.7) !important;
        border: 1px solid rgba(0, 229, 255, 0.3) !important;
        border-radius: 12px !important;
        box-shadow: 0 0 15px rgba(0, 229, 255, 0.05) !important;
        backdrop-filter: blur(5px);
    }
    
    /* 6. 侧边栏头部：青色下划线 */
    .sidebar-header {
        color: #00E5FF;
        border-bottom: 2px solid rgba(0, 229, 255, 0.5);
        padding-bottom: 5px;
        margin-bottom: 15px;
        text-shadow: 0 0 5px rgba(0, 229, 255, 0.3);
    }

    /* 7. 指标卡数字颜色 */
    [data-testid="stMetricValue"] {
        color: #00E5FF !important;
    }
</style>
"""
st.markdown(custom_css, unsafe_allow_html=True)
st.markdown("<h1 class='main-title'>⚡ V2G 微网仿真与深度对比控制台</h1>", unsafe_allow_html=True)

# ================= 初始化 Session State =================
if 'sim_data' not in st.session_state:
    st.session_state['sim_data'] = None

# ================= 侧边栏：参数与模块配置 =================
with st.sidebar:
    st.markdown("<h2 class='sidebar-header'>⚙️ 1. 核心参数配置</h2>", unsafe_allow_html=True)
    
    a_mode = st.radio(
        "选择 A 组模型模式", 
        options=[1, 2, 3, 4], 
        format_func=lambda x: {1: "A1: 纯 BESS", 2: "A2: 无序充电", 3: "A3: 智能充电", 4: "A4: 纯 V2G"}[x],
        index=3
    )
    
    n_ev = st.slider("EV 数量 (辆)", min_value=10, max_value=1500, value=200, step=10)
    n_days = st.slider("仿真天数 (天)", min_value=1, max_value=30, value=2, step=1)
    sell_ratio = st.slider("售电比例 (0.1~1.0)", min_value=0.1, max_value=1.0, value=0.60, step=0.05)
    emergency_ratio = st.slider("紧急购电惩罚倍数 (5λ)", min_value=1.0, max_value=10.0, value=5.0, step=0.5)

    st.markdown("<h2 class='sidebar-header'>🧩 2. 附加模块叠加 (B/C)</h2>", unsafe_allow_html=True)
    with st.expander("📦 B 组：不确定性叠加"):
        b1 = st.checkbox("B1: 光伏不确定性", value=CONFIG["B1_pv"])
        b2 = st.checkbox("B2: 负荷不确定性", value=CONFIG["B2_load"])
        b3 = st.checkbox("B3: 电价不确定性", value=CONFIG["B3_price"])
        b4 = st.checkbox("B4: EV行为不确定性", value=CONFIG["B4_ev"])
        
    with st.expander("⚙️ C 组：工程增强"):
        c1 = st.checkbox("C1: 滚动 MPC", value=CONFIG["C1_rolling"])
        c2 = st.checkbox("C2: CVaR 风险", value=CONFIG["C2_cvar"])
        c3 = st.checkbox("C3: 电池折旧", value=CONFIG["C3_degradation"])
        c4 = st.checkbox("C4: 变压器容量限制", value=CONFIG["C4_transformer"])
        c5 = st.checkbox("C5: EV 放电上限(DoD)", value=CONFIG["C5_dod"])

    st.markdown("---")
    run_btn = st.button("🚀 开始仿真计算", type="primary", use_container_width=True)

# ================= 执行计算 =================
if run_btn:
    cfg = copy.deepcopy(CONFIG)
    cfg["A"] = a_mode
    cfg["n_ev"] = n_ev
    cfg["n_days"] = n_days
    cfg["sell_ratio"] = sell_ratio
    cfg["emergency_ratio"] = emergency_ratio
    cfg["B1_pv"], cfg["B2_load"], cfg["B3_price"], cfg["B4_ev"] = b1, b2, b3, b4
    cfg["C1_rolling"], cfg["C2_cvar"], cfg["C3_degradation"] = c1, c2, c3
    cfg["C4_transformer"], cfg["C5_dod"] = c4, c5
    cfg["mip_gap"] = 0.05
    cfg["solve_time_limit"] = 60

    with st.spinner(f"🧠 正在求解 {n_days} 天、{n_ev} 辆车的 MILP 模型..."):
        try:
            res = run_simulation(copy.deepcopy(cfg), export_files=False)
            st.session_state['sim_data'] = res
            st.success("✅ 仿真计算完成！")
        except Exception as e:
            st.error(f"❌ 仿真运行出错: {e}")
            st.code(traceback.format_exc()) 

# ================= 结果展示区 =================
# 颜色定义：科幻蓝、节能绿、暗紫、深橙
FUTURE_COLORS = ["#00E5FF", "#00E676", "#7C4DFF", "#FF9100"]
PLOTLY_TEMPLATE = "plotly_dark"

if st.session_state['sim_data'] is not None:
    res = st.session_state['sim_data']
    totals = res["totals"]
    
    # 1. 顶部核心指标看板
    with st.container(border=True):
        st.markdown("### 📊 核心指标看板 (总成本对比)")
        col1, col2, col3, col4 = st.columns(4)
        
        # 去掉了刺眼的红色，全部用白色/青色展示
        col1.metric("A1 纯 BESS", f"{totals['A1']:.0f} 元")
        col2.metric("A2 无序充电", f"{totals['A2']:.0f} 元")
        col3.metric("A3 智能充电", f"{totals['A3']:.0f} 元", delta=f"{totals['A3']-totals['A2']:.0f} 元")
        col4.metric("A4 纯 V2G", f"{totals['A4']:.0f} 元", delta=f"{totals['A4']-totals['A3']:.0f} 元")
        
        st.info(f"**💡 增量分析：** A2-A3 智能充电降本: `{res['delta'][1]:+.0f}` 元 | **A3-A4 纯 V2G 收益: `{res['delta'][2]:+.0f}` 元**")

    tab1, tab2, tab3, tab4 = st.tabs(["📈 微观对比", "📉 日度成本", "💰 收益分配", "💾 数据下载"])

    # ---------- Tab 1: 交互式微观对比 ----------
    with tab1:
        with st.container(border=True):
            st.markdown("#### 48 时段运行细节自由对比")
            
            c1, c2 = st.columns([1, 2])
            with c1:
                t_day = st.selectbox("选择查看哪一天：", options=res["df_timeseries"]["Day"].unique(), index=0)
            with c2:
                t_scens = st.multiselect("选择对比场景：", ["A1", "A2", "A3", "A4"], default=["A3", "A4"])

            y_options = {
                "Load": "基础负荷 (kW)", "PV": "光伏出力 (kW)",
                "Grid_Buy": "电网购电 (kW)", "Grid_Sell": "电网售电 (kW)",
                "EV_Charge": "EV充电 (kW)", "EV_Discharge": "EV放电 (kW)",
                "BESS_Charge": "BESS充电 (kW)", "BESS_Discharge": "BESS放电 (kW)"
            }
            y_cols = st.multiselect("选择叠加的指标：", options=list(y_options.keys()), 
                                    default=["Load", "EV_Charge", "EV_Discharge"],
                                    format_func=lambda x: y_options[x])
            
            df_day = res["df_timeseries"]
            df_day = df_day[(df_day["Day"] == t_day) & (df_day["Scenario"].isin(t_scens))]
            
            if not df_day.empty and y_cols:
                df_melt = df_day.melt(id_vars=["Hour", "Scenario", "Price"], value_vars=y_cols, var_name="指标", value_name="功率")
                df_melt["指标"] = df_melt["指标"].map(y_options)
                
                fig = go.Figure()
                
                for i, (metric, group) in enumerate(df_melt.groupby("指标")):
                    for scen in t_scens:
                        scen_data = group[group["Scenario"] == scen]
                        if scen_data.empty: continue
                        fig.add_trace(go.Scatter(
                            x=scen_data["Hour"], y=scen_data["功率"],
                            name=f"{scen} - {metric}", mode="lines",
                            # 使用科幻色系
                            line=dict(color=FUTURE_COLORS[i % len(FUTURE_COLORS)], dash='solid' if scen == "A4" else 'dot', width=2)
                        ))
                
                df_price = df_day.drop_duplicates("Hour")
                fig.add_trace(go.Scatter(
                    x=df_price["Hour"], y=df_price["Price"], name="电价 (元/kWh)",
                    mode="lines", line=dict(color="#B0BEC5", width=2, dash="dash"), yaxis="y2"
                ))
                
                fig.update_layout(
                    title=f"Day {t_day} 功率与电价时序对比",
                    template=PLOTLY_TEMPLATE, # 深色图表
                    paper_bgcolor="rgba(0,0,0,0)", # 图表背景透明
                    plot_bgcolor="rgba(0,0,0,0)",
                    xaxis_title="时间 (小时)",
                    yaxis=dict(title="功率 (kW)", gridcolor="rgba(255,255,255,0.1)"),
                    yaxis2=dict(title="电价 (元/kWh)", overlaying="y", side="right", showgrid=False),
                    hovermode="x unified",
                    legend=dict(orientation="h", yanchor="bottom", y=1.02, xanchor="right", x=1),
                    margin=dict(l=20, r=20, t=50, b=20)
                )
                st.plotly_chart(fig, use_container_width=True)

    # ---------- Tab 2: 宏观日度成本 ----------
    with tab2:
        with st.container(border=True):
            st.markdown("#### 日度成本趋势对比")
            
            df_daily = res["daily_df"].copy().reset_index()
            if "Day" not in df_daily.columns:
                df_daily = df_daily.rename(columns={df_daily.columns[0]: "Day"})
            
            valid_cols = [c for c in ["A1", "A2", "A3", "A4"] if c in df_daily.columns]
            df_melt = df_daily.melt(id_vars=["Day"], value_vars=valid_cols, var_name="场景", value_name="成本 (元)")
            
            fig_daily = px.line(
                df_melt, x="Day", y="成本 (元)", color="场景", 
                markers=True, title="各场景日度运行成本对比",
                color_discrete_sequence=FUTURE_COLORS # 使用科幻色系
            )
            fig_daily.update_layout(
                template=PLOTLY_TEMPLATE, 
                paper_bgcolor="rgba(0,0,0,0)",
                plot_bgcolor="rgba(0,0,0,0)",
                margin=dict(t=50, b=20, l=20, r=20),
                xaxis_title="天数",
                yaxis_title="日总成本 (元)",
                xaxis=dict(gridcolor="rgba(255,255,255,0.1)"),
                yaxis=dict(gridcolor="rgba(255,255,255,0.1)")
            )
            st.plotly_chart(fig_daily, use_container_width=True)

    # ---------- Tab 3: 收益分配 ----------
    with tab3:
        with st.container(border=True):
            st.markdown("#### V2G 总收益分红")
            
            share_df = res["share_df"].copy()
            share_df["金额(元)"] = share_df["比例"] * res['total_gain']
            
            fig_pie = px.pie(
                share_df, 
                values='金额(元)', 
                names='角色', 
                title=f"V2G 总收益分红 (总计 {res['total_gain']:.0f} 元)",
                color_discrete_sequence=FUTURE_COLORS, # 统一使用科幻色系
                hole=0.5
            )
            fig_pie.update_traces(
                textposition='inside', 
                texttemplate='<b>%{label}</b><br>%{value:,.0f} 元<br>(%{percent})',
                hovertemplate='<b>%{label}</b><br>金额: %{value:,.2f} 元<br>占比: %{percent}<extra></extra>',
                textfont_size=14,
                marker=dict(line=dict(color='#0A0F1A', width=2)) # 饼图块之间的深色边框，增加立体感
            )
            fig_pie.update_layout(
                template=PLOTLY_TEMPLATE,
                paper_bgcolor="rgba(0,0,0,0)",
                plot_bgcolor="rgba(0,0,0,0)",
                margin=dict(t=50, b=20, l=20, r=20)
            )
            st.plotly_chart(fig_pie, use_container_width=True)

    # ---------- Tab 4: 数据与图表下载 ----------
    with tab4:
        with st.container(border=True):
            st.markdown("#### 📥 下载仿真结果")
            
            output = io.BytesIO()
            with pd.ExcelWriter(output, engine='xlsxwriter') as writer:
                res["df_timeseries"].to_excel(writer, sheet_name='48时段详细数据', index=False)
                res["daily_df"].to_excel(writer, sheet_name='日度成本汇总')
                res["df_impact"].to_excel(writer, sheet_name='电网冲击指标')
            excel_data = output.getvalue()
            
            txt_content = f"V2G 仿真汇总\n{'='*30}\n"
            for k, v in totals.items():
                txt_content += f"{k} 总成本: {v:.2f} 元\n"
            txt_content += f"\nA3-A4 纯 V2G 收益: {res['delta'][2]:.2f} 元\n"
            
            col_btn1, col_btn2 = st.columns(2)
            with col_btn1:
                st.download_button(
                    label="📊 下载完整数据表 (Excel)",
                    data=excel_data,
                    file_name=f"V2G_仿真结果_A{a_mode}.xlsx",
                    mime="application/vnd.openxmlformats-officedocument.spreadsheetml.sheet",
                    use_container_width=True,
                    type="primary"
                )
            with col_btn2:
                st.download_button(
                    label="📄 下载汇总报告 (TXT)",
                    data=txt_content,
                    file_name=f"V2G_汇总报告_A{a_mode}.txt",
                    mime="text/plain",
                    use_container_width=True
                )
