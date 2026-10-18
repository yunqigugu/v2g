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
from plotting import plot_revenue_share, plot_daily_cost

# ================= 页面基础配置与全局美化 CSS =================
st.set_page_config(page_title="V2G 微网仿真控制台", layout="wide", page_icon="⚡")

# 注入自定义 CSS
custom_css = """
<style>
    /* 隐藏右上角主菜单和底部水印 */
    #MainMenu {visibility: hidden;}
    footer {visibility: hidden;}
    
    /* 主标题居中与美化 */
    .main-title {
        text-align: center;
        color: #1f77b4;
        font-weight: 700;
        margin-bottom: 20px;
        text-shadow: 1px 1px 2px rgba(0,0,0,0.1);
    }
    
    /* 按钮圆角与悬停动画 */
    .stButton>button {
        border-radius: 8px;
        font-weight: bold;
        transition: all 0.3s ease;
    }
    .stButton>button:hover {
        transform: translateY(-2px);
        box-shadow: 0 4px 8px rgba(0,0,0,0.15);
    }
    
    /* 侧边栏头部样式 */
    .sidebar-header {
        color: #2e86c1;
        border-bottom: 2px solid #2e86c1;
        padding-bottom: 5px;
        margin-bottom: 15px;
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
    
    col_a, col_b = st.columns(2)
    with col_a:
        n_ev = st.number_input("EV 数量 (辆)", min_value=10, max_value=2000, value=200, step=10)
    with col_b:
        n_days = st.number_input("仿真天数 (天)", min_value=1, max_value=30, value=2, step=1)
        
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
    # 覆盖参数
    cfg["A"] = a_mode
    cfg["n_ev"] = n_ev
    cfg["n_days"] = n_days
    cfg["sell_ratio"] = sell_ratio
    cfg["emergency_ratio"] = emergency_ratio
    cfg["B1_pv"], cfg["B2_load"], cfg["B3_price"], cfg["B4_ev"] = b1, b2, b3, b4
    cfg["C1_rolling"], cfg["C2_cvar"], cfg["C3_degradation"] = c1, c2, c3
    cfg["C4_transformer"], cfg["C5_dod"] = c4, c5
    
    # 网页端限制求解时间与精度，防止卡死
    cfg["mip_gap"] = 0.05
    cfg["solve_time_limit"] = 60

    with st.spinner(f"🧠 正在求解 {n_days} 天、{n_ev} 辆车的 MILP 模型，请稍候..."):
        try:
            res = run_simulation(copy.deepcopy(cfg), export_files=False)
            st.session_state['sim_data'] = res
            st.success("✅ 仿真计算完成！")
        except Exception as e:
            st.error(f"❌ 仿真运行出错: {e}")
            st.code(traceback.format_exc()) 

# ================= 结果展示区 =================
if st.session_state['sim_data'] is not None:
    res = st.session_state['sim_data']
    totals = res["totals"]
    
    # 1. 顶部核心指标看板（装入卡片容器中）
    with st.container(border=True):
        st.markdown("### 📊 核心指标看板 (总成本对比)")
        col1, col2, col3, col4 = st.columns(4)
        
        # 使用 delta 属性让数据更有对比感
        col1.metric("A1 纯 BESS", f"{totals['A1']:.0f} 元")
        col2.metric("A2 无序充电", f"{totals['A2']:.0f} 元")
        col3.metric("A3 智能充电", f"{totals['A3']:.0f} 元", delta=f"{totals['A3']-totals['A2']:.0f} 元", delta_color="inverse")
        col4.metric("A4 纯 V2G", f"{totals['A4']:.0f} 元", delta=f"{totals['A4']-totals['A3']:.0f} 元", delta_color="inverse")
        
        st.info(f"**💡 增量分析：** A2-A3 智能充电降本: `{res['delta'][1]:+.0f}` 元 | **A3-A4 纯 V2G 收益: `{res['delta'][2]:+.0f}` 元**")

    # 2. 结果 Tabs
    tab1, tab2, tab3, tab4 = st.tabs(["📈 微观对比", "📉 日度成本", "💰 收益分配", "💾 数据下载"])

    # ---------- Tab 1: 交互式微观对比 ----------
    with tab1:
        with st.container(border=True):
            st.markdown("#### 48 时段运行细节自由对比")
            st.caption("您可以任意叠加：负荷、光伏、电网购售、EV充放、BESS充放。鼠标悬停可查看精准数值，支持框选缩放与图例隐藏。")
            
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
                colors = px.colors.qualitative.Plotly
                
                for i, (metric, group) in enumerate(df_melt.groupby("指标")):
                    for scen in t_scens:
                        scen_data = group[df_melt["Scenario"] == scen] if "Scenario" in group else group
                        if scen_data.empty: continue
                        fig.add_trace(go.Scatter(
                            x=scen_data["Hour"], y=scen_data["功率"],
                            name=f"{scen} - {metric}", mode="lines",
                            line=dict(color=colors[i % len(colors)], dash='solid' if scen == "A4" else 'dot')
                        ))
                
                df_price = df_day.drop_duplicates("Hour")
                fig.add_trace(go.Scatter(
                    x=df_price["Hour"], y=df_price["Price"], name="电价 (元/kWh)",
                    mode="lines", line=dict(color="gray", width=2, dash="dash"), yaxis="y2"
                ))
                
                fig.update_layout(
                    title=f"Day {t_day} 功率与电价时序对比",
                    xaxis_title="时间 (小时)",
                    yaxis=dict(title="功率 (kW)"),
                    yaxis2=dict(title="电价 (元/kWh)", overlaying="y", side="right", showgrid=False),
                    hovermode="x unified",
                    legend=dict(orientation="h", yanchor="bottom", y=1.02, xanchor="right", x=1),
                    template="plotly_white",
                    margin=dict(l=20, r=20, t=50, b=20)
                )
                st.plotly_chart(fig, use_container_width=True)

    # ---------- Tab 2: 宏观日度成本 ----------
    with tab2:
        with st.container(border=True):
            st.markdown("#### 日度成本趋势对比")
            fig_mat = plot_daily_cost(res["daily_df"], out_dir=None)
            # Matplotlib 图表居中显示
            col1, col2, col3 = st.columns([1, 8, 1])
            with col2:
                st.pyplot(fig_mat, use_container_width=True)

       # ---------- Tab 3: 收益分配 ----------
    with tab3:
        with st.container(border=True):
            st.markdown("#### V2G 总收益分红")
            
            # 使用 Plotly 画饼图（完美支持中文）
            share_df = res["share_df"]
            fig_pie = px.pie(
                share_df, 
                values='金额(元)', 
                names='角色', 
                title=f"V2G 总收益分红 (总计 {res['total_gain']:.0f} 元)",
                color_discrete_sequence=px.colors.qualitative.Pastel,
                hole=0.4 # 变成环形图更高级
            )
            fig_pie.update_traces(textposition='inside', textinfo='percent+label')
            fig_pie.update_layout(margin=dict(t=50, b=20, l=20, r=20))
            
            st.plotly_chart(fig_pie, use_container_width=True)

    # ---------- Tab 4: 数据与图表下载 ----------
    with tab4:
        with st.container(border=True):
            st.markdown("#### 📥 下载仿真结果")
            st.caption("所有仿真详细数据和汇总报告均可一键导出，方便科研绘图或撰写报告。")
            
            # 生成 Excel 的缓冲数据
            output = io.BytesIO()
            with pd.ExcelWriter(output, engine='xlsxwriter') as writer:
                res["df_timeseries"].to_excel(writer, sheet_name='48时段详细数据', index=False)
                res["daily_df"].to_excel(writer, sheet_name='日度成本汇总')
                res["df_impact"].to_excel(writer, sheet_name='电网冲击指标')
            excel_data = output.getvalue()
            
            # 生成 TXT 的文本数据
            txt_content = f"V2G 仿真汇总\n{'='*30}\n"
            for k, v in totals.items():
                txt_content += f"{k} 总成本: {v:.2f} 元\n"
            txt_content += f"\nA3-A4 纯 V2G 收益: {res['delta'][2]:.2f} 元\n"
            
            # 按钮并排显示
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
