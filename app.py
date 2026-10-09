import streamlit as st
import pandas as pd

from agent import (
    ask_agent,
    route_question,
    detect_persona,
    load_csv
)


# ============================================================
# 1. 页面配置
# ============================================================

st.set_page_config(
    page_title="恋与深空玩家研究 AI Agent",
    page_icon="🎮",
    layout="wide"
)


# ============================================================
# 2. CSS
# ============================================================

st.markdown(
    """
    <style>

    .main-title {
        font-size: 34px;
        font-weight: 700;
        margin-bottom: 4px;
    }

    .sub-title {
        color: #666666;
        font-size: 15px;
        margin-bottom: 24px;
    }

    .route-box {
        padding: 12px 16px;
        border-radius: 14px;
        background: #f4f4f2;
        margin-bottom: 10px;
        font-size: 14px;
    }

    .persona-box {
        padding: 16px;
        border-radius: 18px;
        background: #f7f7f5;
        margin-bottom: 10px;
    }

    .small-note {
        color: #777;
        font-size: 12px;
        line-height: 1.6;
    }

    </style>
    """,
    unsafe_allow_html=True
)


# ============================================================
# 3. Session State
# ============================================================

if "messages" not in st.session_state:

    st.session_state.messages = []


if "last_routes" not in st.session_state:

    st.session_state.last_routes = []


if "last_question" not in st.session_state:

    st.session_state.last_question = ""


# ============================================================
# 4. Header
# ============================================================

st.markdown(
    '<div class="main-title">《恋与深空》玩家研究 AI Agent</div>',
    unsafe_allow_html=True
)

st.markdown(
    """
    <div class="sub-title">
    基于 TapTap + iOS 玩家评论、流失驱动、Persona、
    舆情风险与竞品 Evidence 的自然语言研究 Agent
    </div>
    """,
    unsafe_allow_html=True
)


# ============================================================
# 5. 左右布局
# ============================================================

left_col, right_col = st.columns(
    [2.2, 1]
)


# ============================================================
# 6. 左侧 Chat
# ============================================================

with left_col:

    st.subheader(
        "研究对话"
    )


    # --------------------------------------------------------
    # 历史消息
    # --------------------------------------------------------

    for message in st.session_state.messages:

        with st.chat_message(
            message["role"]
        ):

            st.markdown(
                message["content"]
            )


    # --------------------------------------------------------
    # 快捷问题
    # --------------------------------------------------------

    st.caption(
        "快捷问题"
    )


    q1, q2, q3, q4 = st.columns(
        4
    )


    quick_question = None


    with q1:

        if st.button(
            "玩家为什么流失？",
            use_container_width=True
        ):

            quick_question = (
                "玩家为什么流失？"
            )


    with q2:

        if st.button(
            "平台有什么区别？",
            use_container_width=True
        ):

            quick_question = (
                "TapTap 和 iOS 的舆情有什么区别？"
            )


    with q3:

        if st.button(
            "怎么挽回价值观玩家？",
            use_container_width=True
        ):

            quick_question = (
                "价值观底线型玩家应该怎么挽回？"
            )


    with q4:

        if st.button(
            "未定有什么可借鉴？",
            use_container_width=True
        ):

            quick_question = (
                "未定事件簿有哪些值得恋与深空借鉴的地方？"
            )


    # --------------------------------------------------------
    # 输入框
    # --------------------------------------------------------

    user_input = st.chat_input(
        "输入你的研究问题，例如：哪类玩家最值得优先挽回？"
    )


    question = (
        quick_question
        or user_input
    )


    # --------------------------------------------------------
    # Agent 调用
    # --------------------------------------------------------

    if question:

        st.session_state.last_question = (
            question
        )


        st.session_state.messages.append(
            {
                "role": "user",
                "content": question
            }
        )


        with st.chat_message(
            "user"
        ):

            st.markdown(
                question
            )


        routes = route_question(
            question
        )


        route_names = [

            route[
                "route_name"
            ]

            for route in routes
        ]


        st.session_state.last_routes = (
            route_names
        )


        with st.chat_message(
            "assistant"
        ):

            with st.spinner(
                "Agent 正在选择分析工具并读取研究数据..."
            ):

                try:

                    answer = ask_agent(
                        question,
                        show_route=False
                    )


                    st.markdown(
                        answer
                    )


                    st.session_state.messages.append(
                        {
                            "role": "assistant",
                            "content": answer
                        }
                    )


                except Exception as e:

                    error_text = (
                        f"Agent 调用失败：{e}"
                    )


                    st.error(
                        error_text
                    )


                    st.session_state.messages.append(
                        {
                            "role": "assistant",
                            "content": error_text
                        }
                    )


# ============================================================
# 7. 右侧 Research Panel
# ============================================================

with right_col:

    st.subheader(
        "Research Panel"
    )


    # --------------------------------------------------------
    # 本轮 Router
    # --------------------------------------------------------

    st.markdown(
        "#### 本轮自动调用"
    )


    if (
        st.session_state.last_routes
    ):

        for route_name in (
            st.session_state.last_routes
        ):

            st.markdown(
                f"""
                <div class="route-box">
                ✓ {route_name}
                </div>
                """,
                unsafe_allow_html=True
            )


    else:

        st.caption(
            "提出问题后，这里会显示 Agent 自动选择的工具。"
        )


    st.divider()


    # ========================================================
    # 8. 数据概览
    # ========================================================

    st.markdown(
        "#### 数据概览"
    )


    try:

        platform_df = load_csv(
            "01_platform_summary.csv"
        )


        if (
            "platform"
            in platform_df.columns
        ):

            tap_row = platform_df[
                platform_df[
                    "platform"
                ] == "TapTap"
            ]


            ios_row = platform_df[
                platform_df[
                    "platform"
                ] == "iOS"
            ]


            m1, m2 = st.columns(
                2
            )


            with m1:

                if len(
                    tap_row
                ) > 0:

                    st.metric(
                        "TapTap",
                        f"{int(tap_row.iloc[0]['sample_size']):,} 条"
                    )


            with m2:

                if len(
                    ios_row
                ) > 0:

                    st.metric(
                        "iOS",
                        f"{int(ios_row.iloc[0]['sample_size']):,} 条"
                    )


    except Exception:

        st.caption(
            "平台数据暂不可用"
        )


    st.divider()


    # ========================================================
    # 9. Persona
    # ========================================================

    st.markdown(
        "#### Persona"
    )


    detected_persona = None


    if (
        st.session_state.last_question
    ):

        detected_persona = (
            detect_persona(
                st.session_state.last_question
            )
        )


    try:

        persona_df = load_csv(
            "05_persona_summary.csv"
        )


        if detected_persona:

            target = persona_df[
                persona_df[
                    "final_persona"
                ] == detected_persona
            ]


            if len(
                target
            ) > 0:

                row = target.iloc[0]


                st.markdown(
                    f"""
                    <div class="persona-box">

                    <b>{detected_persona}</b><br><br>

                    样本占比：
                    {row['sample_share_pct']}%<br>

                    流失信号：
                    {row['churn_signal_pct']}%<br>

                    明确流失：
                    {row['explicit_churn_pct']}%<br>

                    高风险：
                    {row['high_risk_pct']}%

                    </div>
                    """,
                    unsafe_allow_html=True
                )


        else:

            persona_display = persona_df[
                [
                    "final_persona",
                    "sample_share_pct",
                    "churn_signal_pct"
                ]
            ].copy()


            persona_display.columns = [
                "Persona",
                "占比 %",
                "流失信号 %"
            ]


            st.dataframe(
                persona_display,
                hide_index=True,
                use_container_width=True
            )


    except Exception:

        st.caption(
            "Persona 数据暂不可用"
        )


    st.divider()


    # ========================================================
    # 10. Agent 架构
    # ========================================================

    st.markdown(
        "#### Agent 架构"
    )


    st.markdown(
        """
        **Research Agent**

        ↓

        **Router**

        ↓

        7 个 Knowledge Tools

        ↓

        **Evidence Guard**

        ↓

        DeepSeek 综合回答
        """
    )


    st.divider()


    # ========================================================
    # 11. 数据边界
    # ========================================================

    st.markdown(
        "#### 研究边界"
    )


    st.markdown(
        """
        <div class="small-note">

        • 评论比例描述采集样本，不代表全部玩家总体。<br><br>

        • iOS 为事件窗口样本。<br><br>

        • Driver Score 用于探索性排序，不代表因果。<br><br>

        • 竞品仅使用 verified evidence。<br><br>

        • Persona 不推断年龄、职业、收入或地区。

        </div>
        """,
        unsafe_allow_html=True
    )


# ============================================================
# 12. Sidebar
# ============================================================

with st.sidebar:

    st.header(
        "AI Research Agent"
    )


    st.caption(
        "项目能力"
    )


    st.markdown(
        """
        - 平台差异
        - 流失驱动
        - 舆情风险
        - Persona
        - Persona Strategy
        - 竞品研究
        - 评论证据
        """
    )


    st.divider()


    if st.button(
        "清空对话",
        use_container_width=True
    ):

        st.session_state.messages = []

        st.session_state.last_routes = []

        st.session_state.last_question = ""

        st.rerun()


    st.divider()


    st.caption(
        "Knowledge Base"
    )


    st.code(
        """
01 Platform
02 Churn
03 Risk
04 Need
05 Persona
06 Persona Topics
07 Persona Strategy
08 Competitor Evidence
09 Competitor Matrix
10 Review Evidence
        """
    )