import os
import json
from pathlib import Path

import pandas as pd
from openai import OpenAI


# ============================================================
# 1. 基础路径
# ============================================================

BASE_DIR = Path(__file__).resolve().parent

KNOWLEDGE_DIR = (
    BASE_DIR
    / "analysis_outputs"
    / "agent_knowledge"
)

MANIFEST_FILE = (
    KNOWLEDGE_DIR
    / "agent_manifest.json"
)


# ============================================================
# 2. DeepSeek API
# ============================================================

api_key = os.getenv(
    "DEEPSEEK_API_KEY"
)

if not api_key:

    print(
        "❌ 没读取到 DEEPSEEK_API_KEY"
    )

    raise SystemExit


client = OpenAI(

    api_key=api_key,

    base_url=
        "https://api.deepseek.com",

    timeout=120.0
)


MODEL_NAME = (
    "deepseek-v4-flash"
)


# ============================================================
# 3. 检查 Knowledge Base
# ============================================================

if not MANIFEST_FILE.exists():

    print(
        "❌ 找不到 agent_manifest.json"
    )

    print(
        "请先运行 build_agent_knowledge.py"
    )

    raise SystemExit


with open(
    MANIFEST_FILE,
    "r",
    encoding="utf-8"
) as f:

    manifest = json.load(f)


ROUTES = manifest[
    "routing_rules"
]


# ============================================================
# 4. 数据缓存
# ============================================================

DATA_CACHE = {}


def load_csv(
    filename
):

    if filename in DATA_CACHE:

        return DATA_CACHE[
            filename
        ]


    path = (
        KNOWLEDGE_DIR
        / filename
    )


    if not path.exists():

        raise FileNotFoundError(
            f"找不到知识文件：{filename}"
        )


    df = pd.read_csv(
        path
    )


    DATA_CACHE[
        filename
    ] = df


    return df


# ============================================================
# 5. DataFrame 文本化
# ============================================================

def df_to_text(
    df,
    max_rows=15
):

    if df is None:

        return ""


    if len(df) == 0:

        return "无匹配数据"


    temp = df.head(
        max_rows
    ).copy()


    return temp.to_string(
        index=False
    )


# ============================================================
# 6. Persona 检测
# ============================================================

def detect_persona(
    question
):

    persona_keywords = {

        "信任敏感/长期支持型": [

            "信任",
            "官方",
            "沟通",
            "长期支持",
            "长期玩家",
            "承诺",
            "失信",
            "冷处理"
        ],

        "价值观底线型": [

            "价值观",
            "女性",
            "女性安全",
            "历史",
            "民族",
            "底线",
            "731",
            "立场",
            "辱华"
        ],

        "情感陪伴/角色投入型": [

            "角色",
            "男主",
            "陪伴",
            "剧情",
            "情感",
            "角色公平",
            "主线",
            "断更"
        ],

        "产品/付费体验型": [

            "产品",
            "付费",
            "氪金",
            "抽卡",
            "玩法",
            "性能",
            "福利",
            "卡池",
            "资源"
        ]
    }


    scores = []


    for persona, keywords in (
        persona_keywords.items()
    ):

        score = 0


        for keyword in keywords:

            if keyword in question:

                score += 1


        scores.append(
            (
                persona,
                score
            )
        )


    scores = sorted(
        scores,
        key=lambda x:
            x[1],
        reverse=True
    )


    if (
        scores
        and scores[0][1] > 0
    ):

        return scores[0][0]


    return None


# ============================================================
# 7. Route 工具
# ============================================================

def get_route_by_name(
    name
):

    for route in ROUTES:

        if (
            route[
                "route_name"
            ]
            == name
        ):

            return route


    return None


def add_route(
    selected,
    route_name
):

    existing = {

        route[
            "route_name"
        ]

        for route in selected
    }


    if route_name in existing:

        return


    route = get_route_by_name(
        route_name
    )


    if route:

        selected.append(
            route
        )


# ============================================================
# 8. V2 Router
# ============================================================

def route_question(question):
    """根据问题涉及的研究维度选择工具；显式要求优先于关键词评分。"""
    q = str(question).lower()
    selected = []

    def contains(*words):
        return any(word.lower() in q for word in words)

    # 显式研究维度：不会因为其他工具关键词更多而被排除。
    if contains("平台", "taptap", "ios", "跨平台", "平台对比", "平台差异"):
        add_route(selected, "platform_comparison")
    if contains("流失", "退游", "停玩", "停氪", "留存", "挽回", "churn"):
        add_route(selected, "churn_analysis")
    if contains("persona", "画像", "哪类玩家", "哪种玩家", "玩家群体", "用户群体", "不同类型玩家", "四类玩家", "不同玩家"):
        add_route(selected, "persona_analysis")
    if contains("竞品", "未定事件簿", "世界之外", "借鉴", "竞对"):
        add_route(selected, "competitor_analysis")
    if contains("原文", "原话", "真实评论", "评论证据", "评论案例", "评论证明", "具体评论", "引用评论"):
        add_route(selected, "review_evidence")
    if contains("风险", "舆情", "危机", "集体行动", "举报", "抵制"):
        add_route(selected, "risk_analysis")
    if contains("运营建议", "运营策略", "优化建议", "内容运营", "怎么运营", "如何挽回", "如何留住", "怎么办", "策略建议"):
        add_route(selected, "persona_strategy")

    # 没有明确研究维度时，参考 manifest 的关键词，但不使用相对分数截断。
    if not selected:
        scored = []
        for route in ROUTES:
            score = sum(bool(keyword) and str(keyword).lower() in q for keyword in route.get("keywords", []))
            if score:
                scored.append((score, route["route_name"]))
        for _, name in sorted(scored, reverse=True)[:3]:
            add_route(selected, name)

    if not selected:
        add_route(selected, "persona_analysis")
        add_route(selected, "churn_analysis")

    # 最多五个，保留明确问题中的核心分析工具，策略表可作为次级补充。
    priority = [
        "persona_analysis", "churn_analysis", "platform_comparison",
        "competitor_analysis", "review_evidence", "risk_analysis", "persona_strategy"
    ]
    selected.sort(key=lambda route: priority.index(route["route_name"]) if route["route_name"] in priority else 99)
    return selected[:5]


# ============================================================
# 9. Tool:
# Platform
# ============================================================

def tool_platform_comparison(
    question
):

    df = load_csv(
        "01_platform_summary.csv"
    )


    return (
        "【跨平台核心指标】\n"
        + df_to_text(
            df,
            10
        )
    )


# ============================================================
# 10. Tool:
# Churn
# ============================================================

def tool_churn_analysis(
    question
):

    topic_df = load_csv(
        "02_churn_drivers.csv"
    ).copy()


    need_df = load_csv(
        "04_need_churn_drivers.csv"
    ).copy()


    if (
        "topic"
        in topic_df.columns
    ):

        topic_df = topic_df[
            topic_df[
                "topic"
            ]
            != "退游/流失意向"
        ]


    if (
        "driver_score"
        in topic_df.columns
    ):

        topic_df = topic_df.sort_values(
            "driver_score",
            ascending=False
        )


    if (
        "need_driver_score"
        in need_df.columns
    ):

        need_df = need_df.sort_values(
            "need_driver_score",
            ascending=False
        )


    return (
        "【Topic 流失驱动】\n"
        + df_to_text(
            topic_df,
            18
        )
        + "\n\n"
        + "【Need 流失驱动】\n"
        + df_to_text(
            need_df,
            18
        )
    )


# ============================================================
# 11. Tool:
# Risk
# ============================================================

def tool_risk_analysis(
    question
):

    df = load_csv(
        "03_risk_drivers.csv"
    ).copy()


    if (
        "topic"
        in df.columns
    ):

        df = df[
            df[
                "topic"
            ]
            != "退游/流失意向"
        ]


    if (
        "risk_driver_score"
        in df.columns
    ):

        df = df.sort_values(
            "risk_driver_score",
            ascending=False
        )


    return (
        "【高风险舆情驱动】\n"
        + df_to_text(
            df,
            18
        )
    )


# ============================================================
# 12. Tool:
# Persona
# ============================================================

def tool_persona_analysis(
    question
):

    summary = load_csv(
        "05_persona_summary.csv"
    ).copy()


    topics = load_csv(
        "06_persona_topics.csv"
    ).copy()


    strategy = load_csv(
        "07_persona_strategy.csv"
    ).copy()


    persona = detect_persona(question)
    if any(term in question for term in ("四类", "所有", "不同类型", "各类", "分别", "全部", "不同玩家")):
        persona = None

    if persona:

        if (
            "final_persona"
            in summary.columns
        ):

            summary = summary[
                summary[
                    "final_persona"
                ] == persona
            ]


        if (
            "final_persona"
            in topics.columns
        ):

            topics = topics[
                topics[
                    "final_persona"
                ] == persona
            ]


        if (
            "persona"
            in strategy.columns
        ):

            strategy = strategy[
                strategy[
                    "persona"
                ] == persona
            ]


        return (
            f"【目标 Persona：{persona}】\n"
            + df_to_text(
                summary,
                5
            )
            + "\n\n"
            + "【Persona Topics】\n"
            + df_to_text(
                topics,
                12
            )
            + "\n\n"
            + "【Persona Strategy】\n"
            + df_to_text(
                strategy,
                5
            )
        )


    return (
        "【Persona 总览】\n"
        + df_to_text(
            summary,
            10
        )
        + "\n\n"
        + "【Persona Topics】\n"
        + df_to_text(
            topics,
            30
        )
    )


# ============================================================
# 13. Tool:
# Persona Strategy
# ============================================================

def tool_persona_strategy(
    question
):

    strategy = load_csv(
        "07_persona_strategy.csv"
    ).copy()


    persona = detect_persona(
        question
    )


    if (
        persona
        and "persona"
        in strategy.columns
    ):

        subset = strategy[
            strategy[
                "persona"
            ] == persona
        ]


        if len(subset) > 0:

            return (
                f"【目标 Persona：{persona}】\n"
                + df_to_text(
                    subset,
                    5
                )
            )


    return (
        "【Persona Strategy 全表】\n"
        + df_to_text(
            strategy,
            10
        )
    )


# ============================================================
# 14. Tool:
# Competitor
# ============================================================

def tool_competitor_analysis(
    question
):

    knowledge = load_csv(
        "08_competitor_knowledge.csv"
    ).copy()


    matrix = load_csv(
        "09_competitor_matrix.csv"
    ).copy()


    competitor = None


    if "世界之外" in question and ("未定事件簿" in question or "未定" in question):
        competitor = None

    elif (
        "未定事件簿"
        in question
        or "未定"
        in question
    ):

        competitor = (
            "未定事件簿"
        )


    elif (
        "世界之外"
        in question
    ):

        competitor = (
            "世界之外"
        )


    persona = detect_persona(question)
    if any(term in question for term in ("四类", "所有", "不同类型", "各类", "分别", "全部", "不同玩家")):
        persona = None

    subset = knowledge.copy()


    if (
        competitor
        and "competitor"
        in subset.columns
    ):

        subset = subset[
            subset[
                "competitor"
            ]
            == competitor
        ]


    if (
        persona
        and "persona"
        in subset.columns
    ):

        persona_subset = subset[
            subset[
                "persona"
            ]
            == persona
        ]


        if len(
            persona_subset
        ) > 0:

            subset = (
                persona_subset
            )


    return (
        "【Verified 竞品 Evidence】\n"
        + df_to_text(
            subset,
            15
        )
        + "\n\n"
        + "【Persona × Competitor Matrix】\n"
        + df_to_text(
            matrix,
            10
        )
    )


# ============================================================
# 15. Tool:
# Review Evidence
# ============================================================

def tool_review_evidence(
    question
):

    df = load_csv(
        "10_review_evidence.csv"
    ).copy()


    working = df.copy()


    persona = detect_persona(
        question
    )


    if (
        persona
        and "final_persona"
        in working.columns
    ):

        persona_subset = working[
            working[
                "final_persona"
            ] == persona
        ]


        if len(
            persona_subset
        ) > 0:

            working = (
                persona_subset
            )


    search_terms = [

        "731",
        "女性",
        "退游",
        "停氪",
        "主线",
        "角色",
        "抽卡",
        "氪金",
        "官方",
        "更新",
        "下架",
        "维权",
        "举报",
        "价值观",
        "历史",
        "客服",
        "运营"
    ]


    matched_terms = [

        term
        for term in search_terms
        if term in question
    ]


    if (
        matched_terms
        and "review"
        in working.columns
    ):

        mask = pd.Series(
            False,
            index=working.index
        )


        for term in matched_terms:

            mask = mask | (

                working[
                    "review"
                ]
                .fillna("")
                .astype(str)
                .str.contains(
                    term,
                    case=False,
                    na=False
                )

            )


        filtered = working[
            mask
        ]


        if len(filtered) > 0:

            working = filtered


    sort_cols = []


    if (
        "llm_risk_level"
        in working.columns
    ):

        sort_cols.append(
            "llm_risk_level"
        )


    if (
        "llm_churn_level"
        in working.columns
    ):

        sort_cols.append(
            "llm_churn_level"
        )


    if sort_cols:

        working = working.sort_values(

            sort_cols,

            ascending=[
                False
            ] * len(
                sort_cols
            )
        )


    evidence_columns = [

        "platform",

        "review",

        "llm_topics",

        "llm_churn_level",

        "llm_collective_action",

        "llm_risk_level",

        "llm_need_category",

        "llm_primary_issue",

        "final_persona"
    ]


    existing = [

        col
        for col in evidence_columns
        if col in working.columns
    ]


    return (
        "【代表性玩家评论证据】\n"
        + df_to_text(
            working[
                existing
            ],
            12
        )
    )


# ============================================================
# 16. Tool Registry
# ============================================================

TOOL_REGISTRY = {

    "platform_comparison":
        tool_platform_comparison,

    "churn_analysis":
        tool_churn_analysis,

    "risk_analysis":
        tool_risk_analysis,

    "persona_analysis":
        tool_persona_analysis,

    "persona_strategy":
        tool_persona_strategy,

    "competitor_analysis":
        tool_competitor_analysis,

    "review_evidence":
        tool_review_evidence
}


# ============================================================
# 17. 执行 Tools
# ============================================================

def run_tools(
    question,
    routes
):

    results = []


    for route in routes:

        route_name = route[
            "route_name"
        ]


        tool = TOOL_REGISTRY.get(
            route_name
        )


        if not tool:

            continue


        try:

            output = tool(
                question
            )


            results.append({

                "route":
                    route_name,

                "description":
                    route[
                        "description"
                    ],

                "output":
                    output
            })


        except Exception as e:

            results.append({

                "route":
                    route_name,

                "description":
                    route[
                        "description"
                    ],

                "output":
                    f"Tool Error: {e}"
            })


    return results


# ============================================================
# 18. 构建 Context
# ============================================================

def build_context(
    tool_results
):

    parts = []


    for result in tool_results:

        parts.append(

            "\n"
            + "=" * 70
            + "\n"

            + "TOOL: "
            + result[
                "route"
            ]
            + "\n"

            + result[
                "description"
            ]
            + "\n"

            + "-" * 70
            + "\n"

            + result[
                "output"
            ]

        )


    return "\n".join(
        parts
    )


# ============================================================
# 19. Evidence Guard
# ============================================================

def build_guard_text(
    routes
):

    route_names = [

        route[
            "route_name"
        ]

        for route in routes
    ]


    available_tools = [

        route[
            "route_name"
        ]

        for route in ROUTES
    ]


    return f"""
本轮实际调用的工具：
{", ".join(route_names)}

系统当前可用工具：
{", ".join(available_tools)}

重要规则：

1. “本轮未调用某工具”不等于“知识库没有该数据”。
2. 如果本轮没有调用 competitor_analysis，
   只能说“本轮未使用竞品证据”，
   不能说“当前没有竞品 evidence”。
3. 如果本轮没有调用 review_evidence，
   只能说“本轮未调用评论证据”，
   不能要求用户补充系统已经拥有的评论数据。
4. 禁止自行创造具体时间 KPI：
   例如 24小时、48小时、72小时、
   1-4周、1-3个月等，
   除非 Knowledge Tool 明确提供。
5. 禁止自行创造业务阈值、百分比、预算、
   SLA、人数、转化目标等数字。
6. 可以说：
   即时响应 / 中期整改 / 长期机制建设，
   但不要自行指定具体天数。
7. 不要把相关性写成：
   “导致”“决定”“直接造成”“必然引发”。
8. 可以写成：
   “与流失信号高度相关”
   “在相关评论中更集中”
   “是值得优先关注的关联因素”。
"""


# ============================================================
# 20. System Prompt V2
# ============================================================

SYSTEM_PROMPT = """
你是一名游戏用户研究、玩家舆情、内容运营与竞品研究 AI Agent。

你服务于《恋与深空》玩家舆情与内容运营策略分析项目。

你必须优先依据系统提供的 Knowledge Tools 回答，
不得为了让答案显得专业而自行补充没有证据的数字或事实。

============================================================
一、研究原则
============================================================

1. 区分：
   - 数据事实
   - 研究解释
   - 运营建议

2. Driver Score：
   只能用于探索性排序，
   不代表因果关系。

3. 玩家评论比例：
   描述的是采集评论样本，
   不代表全部玩家总体比例。

4. iOS：
   属于事件窗口样本，
   不应直接解释为长期 iOS 玩家总体。

5. 竞品：
   只能使用 Knowledge Tool 中提供的 verified evidence。

6. 如果证据不足：
   明确说明证据不足，
   不要为了完整而编造。

7. 不得推断：
   年龄、职业、收入、地区等未采集属性。


============================================================
二、禁止无依据量化
============================================================

禁止自行提出：

- 24小时响应
- 48小时
- 72小时
- 1-4周
- 1-3个月
- XX% KPI
- 转化目标
- SLA
- 预算
- 人员规模

除非这些数字来自 Knowledge Tool。

如果需要表达时间阶段，
使用：

- 即时响应
- 中期整改
- 长期机制建设

而不是自行指定具体时长。


============================================================
三、禁止伪因果
============================================================

不要写：

“某问题导致玩家流失”
“某因素直接决定退游”
“该事件必然引发流失”
“回应质量直接决定流失率”

除非存在真正因果证据。

优先写：

“与流失信号高度相关”
“在流失相关评论中更集中”
“是当前值得优先关注的关联因素”


============================================================
四、工具认知
============================================================

非常重要：

“本轮没有调用一个工具”
不等于
“系统没有这个数据”。

例如：

如果本轮未调用 competitor_analysis，
不要说：
“当前没有竞品数据”。

应该说：
“本轮回答未使用竞品 Evidence。”

如果本轮未调用 review_evidence，
不要要求用户重新提供评论，
因为系统本身拥有 review evidence tool。


============================================================
五、运营建议
============================================================

运营建议必须：

- 与当前数据和 Persona 对应
- 可执行
- 不自行制造数字 KPI
- 区分立即动作、机制整改和长期方向
- 不把竞品机制直接等同于玩家满意度


============================================================
六、回答结构
============================================================

复杂研究问题优先：

【核心结论】
【数据依据】
【研究解释】
【运营建议】
【证据边界】

简单问题可以缩短。

不要在回答结尾主动说：
“如果需要我可以做PPT”
“如果需要我可以继续……”
除非用户明确询问。
"""


# ============================================================
# 21. Agent Answer
# ============================================================

def ask_agent(question, show_route=True):
    routes = route_question(question)
    tool_results = run_tools(question, routes)
    context = build_context(tool_results)
    guard_text = build_guard_text(routes)
    route_names = [route["route_name"] for route in routes]

    if show_route:
        print("\n🧭 Agent Route：" + " + ".join(route_names))

    failed_tools = [r["route"] for r in tool_results if str(r["output"]).startswith("Tool Error:")]
    if failed_tools:
        print("⚠️ Tool errors:", ", ".join(failed_tools))

    user_prompt = f"""用户问题：
{question}

本轮实际调用的 Knowledge Tools：
{context}

Evidence Guard：
{guard_text}

请直接回答问题，按以下要求：
1. 综合本轮所有成功工具的证据；如果工具出错，明确说明而非编造。
2. 优先给出简明结论、关键数据、三项可执行建议、证据边界。
3. 数据必须注明样本语境；流失信号不等于实际流失，相关不等于因果。
4. 不自行创造业务数字、时间 KPI、竞品满意度结论。
5. 对于四类 Persona 的比较，不得只报告其中一类。
6. 避免冗长重复，确保完整结束，不主动提供无关服务。
"""

    messages = [
        {"role": "system", "content": SYSTEM_PROMPT},
        {"role": "user", "content": user_prompt},
    ]
    parts = []
    max_rounds = 3
    for attempt in range(max_rounds):
        response = client.chat.completions.create(
            model=MODEL_NAME,
            messages=messages,
            extra_body={"thinking": {"type": "disabled"}},
            max_tokens=4096,
        )
        choice = response.choices[0]
        piece = choice.message.content or ""
        reason = choice.finish_reason
        print(f"🧪 DeepSeek finish_reason (round {attempt + 1}): {reason}")
        parts.append(piece)
        if reason != "length":
            break
        if attempt == max_rounds - 1:
            parts.append("\n\n⚠️ 回答达到最大续写次数，可能仍不完整。")
            break
        messages.append({"role": "assistant", "content": piece})
        messages.append({
            "role": "user",
            "content": "请从刚才停止的位置继续完成尚未写完的内容，不要重复前文。请简洁收尾并包含证据边界。",
        })

    return "\n".join(parts).strip()


# ============================================================
# 22. 欢迎页
# ============================================================

def print_welcome():

    print()

    print("=" * 80)

    print(
        "《恋与深空》玩家研究 AI Agent V2"
    )

    print("=" * 80)

    print()

    print(
        "架构："
    )

    print(
        "1 个 Research Agent"
    )

    print(
        "+ Router"
    )

    print(
        "+ 7 个 Knowledge Tools"
    )

    print(
        "+ Evidence Guard"
    )

    print()

    print(
        "当前能力："
    )

    print(
        "• TapTap vs iOS 平台差异"
    )

    print(
        "• 玩家流失驱动分析"
    )

    print(
        "• 高风险舆情分析"
    )

    print(
        "• 4 类 Persona"
    )

    print(
        "• Persona 运营策略"
    )

    print(
        "• 未定事件簿 / 世界之外竞品研究"
    )

    print(
        "• 真实玩家评论证据检索"
    )

    print()

    print(
        "输入 exit / quit / 退出 结束。"
    )

    print()


# ============================================================
# 23. Main
# ============================================================

if __name__ == "__main__":

    print_welcome()


    while True:

        try:

            question = input(
                "你 > "
            ).strip()


        except (
            KeyboardInterrupt,
            EOFError
        ):

            print()

            print(
                "Agent 已退出。"
            )

            break


        if not question:

            continue


        if question.lower() in [

            "exit",
            "quit",
            "q",
            "退出",
            "结束"

        ]:

            print(
                "Agent 已退出。"
            )

            break


        try:

            answer = ask_agent(
                question,
                show_route=True
            )


            print()

            print(
                "Agent >"
            )

            print()

            print(
                answer
            )

            print()


        except Exception as e:

            print()

            print(
                "❌ Agent 调用失败："
                + str(e)
            )

            print()
            