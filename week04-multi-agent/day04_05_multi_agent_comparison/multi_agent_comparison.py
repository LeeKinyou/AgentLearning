"""
Day 4-5: LangGraph多Agent协作对比

本模块使用LangGraph实现多Agent协作，与CrewAI进行对比，包括：
1. 使用LangGraph实现多Agent协作工作流
2. 对比CrewAI与LangGraph的实现差异
3. 分析两种框架的优劣势
4. 总结适用场景

学习目标：
- 掌握使用LangGraph构建多Agent系统的方法
- 理解两种框架的设计哲学差异
- 能够根据场景选择合适的框架

依赖安装：
    pip install langgraph langchain langchain-openai pydantic python-dotenv
"""

import os
import sys
from typing import TypedDict, Annotated, List, Optional, Literal
from dotenv import load_dotenv
from pydantic import SecretStr

# 加载环境变量
load_dotenv()

# 添加项目根目录到路径
sys.path.insert(0, os.path.join(os.path.dirname(__file__), '..'))


# ============================================================
# 第一部分：LLM配置
# ============================================================

def create_llm(temperature: float = 0.0, model: str | None = None):
    """
    创建LLM实例
    
    参数：
        temperature: 输出随机性
        model: 模型名称
    
    返回：
        ChatOpenAI: LangChain聊天模型实例
    """
    from langchain_openai import ChatOpenAI
    
    api_key = os.getenv("OPENAI_API_KEY", "lm-studio")
    base_url = os.getenv("API_BASE_URL", "http://localhost:1234/v1")
    model_name = os.getenv("MODEL_NAME", model or "gpt-3.5-turbo")
    
    return ChatOpenAI(
        model=model_name,
        temperature=temperature,
        api_key=SecretStr(api_key),
        base_url=base_url,
    )


# ============================================================
# 第二部分：LangGraph多Agent协作实现
# ============================================================

def demonstrate_langgraph_multi_agent():
    """
    使用LangGraph实现多Agent协作
    
    功能说明：
        实现与CrewAI相同的多Agent协作场景：
        1. 信息收集Agent
        2. 数据分析Agent
        3. 内容撰写Agent
        4. 质量审核Agent
        
    与CrewAI的差异：
        - 使用StateGraph而非Crew抽象
        - 通过State共享数据而非Task上下文
        - 显式定义Node和Edge而非隐式依赖
    """
    print("=" * 60)
    print("LangGraph 多Agent协作实现")
    print("=" * 60)
    
    from langgraph.graph import StateGraph, END, START

    llm = create_llm(temperature=0.7)
    
    # --- 1. 定义State ---
    # 与CrewAI不同，LangGraph需要显式定义全局状态
    class MultiAgentState(TypedDict):
        """多Agent协作状态"""
        topic: str                          # 研究主题
        collected_info: str                 # 收集的信息
        analysis_result: str                # 分析结果
        draft_report: str                   # 报告草稿
        final_report: str                   # 最终报告
        review_feedback: str                # 审核反馈
        review_count: int                   # 审核次数
        max_reviews: int                    # 最大审核次数
        steps: List[str]                    # 执行步骤记录
    
    # --- 2. 定义Agent节点 ---
    def information_collector_node(state: MultiAgentState) -> dict:
        """
        信息收集Agent节点
        
        功能说明：
            收集指定主题的相关信息。
            对应CrewAI中的information_collector Agent。
        """
        print(f"  [信息收集Agent] 开始收集主题：{state['topic']}")
        
        # 实际实现中会调用LLM或搜索工具
        # from langchain_core.prompts import ChatPromptTemplate
        # prompt = ChatPromptTemplate.from_messages([
        #     ("system", "你是信息收集专家，请收集以下主题的最新信息：{topic}"),
        # ])
        # chain = prompt | llm
        # result = chain.invoke({"topic": state["topic"]})
        
        collected_info = f"关于'{state['topic']}'的信息摘要：\n1. 最新发展趋势\n2. 主要市场参与者\n3. 关键技术突破\n4. 典型应用案例"
        
        print(f"  [信息收集Agent] 信息收集完成")
        
        return {
            "collected_info": collected_info,
            "steps": state.get("steps", []) + ["information_collection"]
        }
    
    def data_analyst_node(state: MultiAgentState) -> dict:
        """
        数据分析Agent节点
        
        功能说明：
            分析收集到的信息，提取关键洞察。
            对应CrewAI中的data_analyst Agent。
        """
        print(f"  [数据分析Agent] 开始分析信息")
        
        # 实际实现中会调用LLM分析
        analysis_result = f"分析报告：\n1. 趋势：该领域正在快速增长\n2. 机会：企业服务市场潜力巨大\n3. 挑战：技术成熟度需要提升\n4. 建议：关注实际应用场景"
        
        print(f"  [数据分析Agent] 分析完成")
        
        return {
            "analysis_result": analysis_result,
            "steps": state.get("steps", []) + ["data_analysis"]
        }
    
    def content_writer_node(state: MultiAgentState) -> dict:
        """
        内容撰写Agent节点
        
        功能说明：
            根据分析结果撰写行业报告。
            对应CrewAI中的content_writer Agent。
        """
        print(f"  [内容撰写Agent] 开始撰写报告")
        
        # 实际实现中会调用LLM撰写
        draft_report = f"""行业报告：{state['topic']}

一、执行摘要
本报告分析了{state['topic']}的最新发展状况。

二、行业概述
该领域正在经历快速增长，技术创新不断涌现。

三、市场分析
市场规模持续扩大，企业服务成为主要增长点。

四、技术趋势
AI技术不断成熟，应用场景日益丰富。

五、发展建议
建议关注实际应用场景，提升技术成熟度。

六、结论
该领域前景广阔，但需要理性看待技术局限性。"""
        
        print(f"  [内容撰写Agent] 报告撰写完成")
        
        return {
            "draft_report": draft_report,
            "steps": state.get("steps", []) + ["content_writing"]
        }
    
    def quality_reviewer_node(state: MultiAgentState) -> dict:
        """
        质量审核Agent节点
        
        功能说明：
            审核报告质量，提供改进建议。
            对应CrewAI中的quality_reviewer Agent。
            
        特殊处理：
            使用循环机制，审核不通过时返回修改。
        """
        print(f"  [质量审核Agent] 开始审核报告（第{state.get('review_count', 0) + 1}次）")
        
        # 实际实现中会调用LLM审核
        review_count = state.get("review_count", 0) + 1
        
        # 模拟审核逻辑
        # 第一次审核：需要修改
        # 第二次审核：通过
        if review_count == 1:
            review_feedback = "审核意见：\n1. 需要补充具体数据支撑\n2. 市场分析部分需要更深入\n3. 建议增加案例分析"
            approved = False
        else:
            review_feedback = "审核通过：报告质量良好，结构清晰，论证充分。"
            approved = True
        
        print(f"  [质量审核Agent] 审核结果：{'通过' if approved else '需要修改'}")
        
        return {
            "review_feedback": review_feedback,
            "review_count": review_count,
            "final_report": state["draft_report"] if approved else "",
            "steps": state.get("steps", []) + ["quality_review"]
        }
    
    def revise_report_node(state: MultiAgentState) -> dict:
        """
        报告修改节点
        
        功能说明：
            根据审核反馈修改报告。
            这是LangGraph特有的循环处理机制。
        """
        print(f"  [报告修改Agent] 根据审核反馈修改报告")
        
        # 实际实现中会调用LLM修改
        revised_report = f"""行业报告：{state['topic']}（修订版）

一、执行摘要
本报告分析了{state['topic']}的最新发展状况。

二、行业概述
该领域正在经历快速增长，技术创新不断涌现。
【补充数据】根据最新研究，市场规模年增长率超过30%。

三、市场分析
市场规模持续扩大，企业服务成为主要增长点。
【深入分析】主要驱动因素包括数字化转型需求和AI技术成熟。

四、技术趋势
AI技术不断成熟，应用场景日益丰富。
【案例分析】某企业通过AI技术提升效率40%。

五、发展建议
建议关注实际应用场景，提升技术成熟度。

六、结论
该领域前景广阔，但需要理性看待技术局限性。"""
        
        print(f"  [报告修改Agent] 报告修改完成")
        
        return {
            "draft_report": revised_report,
            "steps": state.get("steps", []) + ["report_revision"]
        }
    
    # --- 3. 定义路由函数 ---
    def check_review_result(state: MultiAgentState) -> Literal["revise", "finalize"]:
        """
        审核结果路由函数
        
        功能说明：
            根据审核结果决定下一步：
            - 审核通过：进入finalize
            - 审核不通过且未超过最大次数：进入revise
            - 超过最大次数：强制进入finalize
        """
        max_reviews = state.get("max_reviews", 2)
        
        if state["final_report"]:
            return "finalize"
        elif state["review_count"] >= max_reviews:
            print(f"  [路由] 达到最大审核次数，强制通过")
            return "finalize"
        else:
            return "revise"
    
    # --- 4. 构建图 ---
    print("\n[构建LangGraph多Agent工作流]")
    print("-" * 40)
    
    builder = StateGraph(MultiAgentState)
    
    # 添加Agent节点
    builder.add_node("collect_info", information_collector_node)
    builder.add_node("analyze_data", data_analyst_node)
    builder.add_node("write_report", content_writer_node)
    builder.add_node("review_report", quality_reviewer_node)
    builder.add_node("revise_report", revise_report_node)
    
    # 设置入口
    builder.add_edge(START, "collect_info")
    
    # 定义顺序执行边
    builder.add_edge("collect_info", "analyze_data")
    builder.add_edge("analyze_data", "write_report")
    builder.add_edge("write_report", "review_report")
    
    # 添加条件边（审核路由）
    builder.add_conditional_edges(
        "review_report",
        check_review_result,
        {
            "revise": "revise_report",
            "finalize": END
        }
    )
    
    # 修改后重新审核
    builder.add_edge("revise_report", "review_report")
    
    # 编译图
    graph = builder.compile()
    
    print("  ✓ 节点：collect_info, analyze_data, write_report, review_report, revise_report")
    print("  ✓ 边：顺序执行 + 条件分支（审核循环）")
    print("  ✓ 特殊机制：审核不通过时循环修改")
    
    # --- 5. 执行工作流 ---
    print("\n[执行LangGraph多Agent工作流]")
    print("-" * 40)
    
    topic = "生成式AI在企业服务中的应用"
    
    result = graph.invoke({
        "topic": topic,
        "collected_info": "",
        "analysis_result": "",
        "draft_report": "",
        "final_report": "",
        "review_feedback": "",
        "review_count": 0,
        "max_reviews": 2,
        "steps": []
    })
    
    print(f"\n[执行完成]")
    print(f"  执行步骤：{' -> '.join(result['steps'])}")
    print(f"  审核次数：{result['review_count']}")
    print(f"  最终报告长度：{len(result['final_report'])} 字符")
    
    return graph


# ============================================================
# 第三部分：CrewAI与LangGraph实现对比
# ============================================================

def compare_implementation_approaches():
    """
    对比CrewAI与LangGraph的实现方式
    
    功能说明：
        从代码层面详细对比两种框架的实现差异。
    """
    print("\n" + "=" * 60)
    print("CrewAI vs LangGraph 实现对比")
    print("=" * 60)
    
    print("\n[CrewAI实现方式]")
    print("-" * 40)
    print("""
# CrewAI代码示例
from crewai import Agent, Task, Crew, Process

# 1. 定义Agent
agent = Agent(
    role="角色名称",
    goal="目标",
    backstory="背景故事",
    llm=llm,
)

# 2. 定义Task（通过context指定依赖）
task = Task(
    description="任务描述",
    expected_output="期望输出",
    agent=agent,
    context=[previous_task],  # 依赖关系
)

# 3. 创建Crew并执行
crew = Crew(
    agents=[agent1, agent2],
    tasks=[task1, task2],
    process=Process.sequential,
)
result = crew.kickoff(inputs={"topic": "主题"})

特点：
  - 高层抽象，业务导向
  - 通过Task的context管理依赖
  - Agent封装了角色、目标、背景
  - 执行模式预定义（sequential/hierarchical）
    """)
    
    print("\n[LangGraph实现方式]")
    print("-" * 40)
    print("""
# LangGraph代码示例
from langgraph.graph import StateGraph, END, START
from typing import TypedDict

# 1. 定义State
class State(TypedDict):
    topic: str
    result: str
    steps: List[str]

# 2. 定义Node（函数）
def agent_node(state: State) -> dict:
    # 处理逻辑
    return {"result": "...", "steps": [...]}

# 3. 构建图
builder = StateGraph(State)
builder.add_node("agent", agent_node)
builder.add_edge(START, "agent")
builder.add_edge("agent", END)

graph = builder.compile()
result = graph.invoke({"topic": "主题", ...})

特点：
  - 底层抽象，技术导向
  - 通过State共享数据
  - Node是纯函数
  - 工作流完全自定义（任意有向图）
    """)
    
    print("\n[核心差异总结]")
    print("-" * 40)
    print("""
1. 抽象层次
   CrewAI:   Agent/Task/Crew（业务概念）
   LangGraph: State/Node/Edge（计算概念）

2. 依赖管理
   CrewAI:   Task.context隐式传递
   LangGraph: State显式共享

3. 工作流控制
   CrewAI:   预定义模式（顺序/层级）
   LangGraph: 完全自定义（图结构）

4. 循环处理
   CrewAI:   需要自定义实现
   LangGraph: 原生支持（条件边）

5. 调试能力
   CrewAI:   中等（日志输出）
   LangGraph: 高（状态可视化）
    """)


# ============================================================
# 第四部分：适用场景分析
# ============================================================

def analyze_use_cases():
    """
    分析两种框架的适用场景
    
    功能说明：
        给出具体场景的框架选择建议。
    """
    print("\n" + "=" * 60)
    print("适用场景分析")
    print("=" * 60)
    
    scenarios = [
        {
            "name": "内容创作流水线",
            "description": "研究员收集信息 → 写手撰写 → 编辑审校",
            "recommendation": "CrewAI",
            "reason": "任务之间有明确的顺序依赖，角色分工清晰，CrewAI的高层抽象更适合快速构建。"
        },
        {
            "name": "智能客服系统",
            "description": "意图识别 → 知识检索 → 回答生成 → 人工审核",
            "recommendation": "LangGraph",
            "reason": "需要条件分支（不同意图路由）和Human-in-the-loop机制，LangGraph的图模型更适合。"
        },
        {
            "name": "数据分析报告",
            "description": "数据收集 → 分析 → 可视化 → 报告生成",
            "recommendation": "CrewAI",
            "reason": "流水线模式，每个步骤有明确的角色和输出，CrewAI的Task依赖管理更直观。"
        },
        {
            "name": "代码审查系统",
            "description": "代码分析 → 问题检测 → 修复建议 → 人工确认 → 自动修复",
            "recommendation": "LangGraph",
            "reason": "需要循环（修复后重新检查）和人工审核，LangGraph的条件边和循环支持更好。"
        },
        {
            "name": "市场调研系统",
            "description": "多源数据收集 → 交叉分析 → 趋势预测 → 报告撰写",
            "recommendation": "混合使用",
            "reason": "数据收集用CrewAI（多Agent并行），工作流控制用LangGraph（复杂分支）。"
        },
        {
            "name": "教育辅导系统",
            "description": "学生评估 → 个性化计划 → 教学内容生成 → 效果评估 → 调整计划",
            "recommendation": "LangGraph",
            "reason": "需要根据评估结果动态调整流程，循环和条件分支是核心需求。"
        }
    ]
    
    for i, scenario in enumerate(scenarios, 1):
        print(f"\n[场景{i}] {scenario['name']}")
        print(f"  描述：{scenario['description']}")
        print(f"  推荐：{scenario['recommendation']}")
        print(f"  原因：{scenario['reason']}")


# ============================================================
# 第五部分：设计模式与最佳实践
# ============================================================

def summarize_best_practices():
    """
    总结多Agent系统的设计模式与最佳实践
    
    功能说明：
        归纳多Agent系统的最佳实践和常见陷阱。
    """
    print("\n" + "=" * 60)
    print("多Agent系统设计模式与最佳实践")
    print("=" * 60)
    
    print("\n[设计模式]")
    print("-" * 40)
    print("""
1. 流水线模式（Pipeline）
   结构：Agent A → Agent B → Agent C
   适用：内容创作、报告生成
   实现：
     - CrewAI: Process.sequential
     - LangGraph: 顺序边

2. 并行模式（Parallel）
   结构：Agent A ┐
                    ├ → 聚合
         Agent B ┘
   适用：多角度分析、并行研究
   实现：
     - CrewAI: 多个Task无依赖
     - LangGraph: 并行Node

3. 评审循环模式（Review Loop）
   结构：生成 → 审核 → [不通过] → 修改 → 审核 → [通过] → 输出
   适用：质量保证、代码审查
   实现：
     - CrewAI: 需要自定义循环
     - LangGraph: 条件边 + 循环

4. 路由模式（Router）
   结构：输入 → 分类 → Agent A/B/C
   适用：智能客服、任务分发
   实现：
     - CrewAI: 需要自定义路由
     - LangGraph: add_conditional_edges

5. 层级模式（Hierarchy）
   结构：Manager → Worker A, Worker B, Worker C
   适用：复杂任务分解
   实现：
     - CrewAI: Process.hierarchical
     - LangGraph: 需要自定义Manager节点
    """)
    
    print("\n[最佳实践]")
    print("-" * 40)
    print("""
1. 从简单开始
   - 先用单Agent验证核心逻辑
   - 逐步增加Agent和复杂度
   - 避免一开始就设计复杂的多Agent系统

2. 明确Agent边界
   - 每个Agent职责单一
   - 避免功能重叠
   - 定义清晰的输入输出格式

3. 管理状态和依赖
   - CrewAI：明确Task的context依赖
   - LangGraph：设计合理的State结构
   - 避免循环依赖

4. 加入可观测性
   - 记录每个Agent的执行日志
   - 使用LangFuse追踪调用链
   - 监控Token消耗和成本

5. 设置超时和重试
   - 防止单个Agent卡死整个流程
   - 实现优雅降级
   - 设置最大重试次数

6. 缓存中间结果
   - 避免重复计算
   - 提高执行效率
   - 便于调试和复盘

7. 编写测试用例
   - 测试单个Agent的功能
   - 测试Agent间的协作
   - 测试异常处理
    """)
    
    print("\n[常见陷阱]")
    print("-" * 40)
    print("""
1. 过度设计
   问题：使用过多Agent，增加复杂度
   解决：评估是否真的需要多Agent

2. 信息丢失
   问题：Agent间传递信息不完整
   解决：使用Pydantic验证输入输出

3. 成本失控
   问题：多Agent导致LLM调用激增
   解决：设置Token预算，优化Prompt

4. 调试困难
   问题：难以定位问题所在Agent
   解决：加入详细日志和追踪

5. 死循环
   问题：Agent间形成循环等待
   解决：设计DAG，设置最大循环次数
    """)


# ============================================================
# 主函数
# ============================================================

def main():
    """
    主函数：运行所有演示
    
    执行流程：
        1. LangGraph多Agent协作实现
        2. 实现方式对比
        3. 适用场景分析
        4. 设计模式与最佳实践总结
    """
    print("=" * 60)
    print("Week04 Day04-05: LangGraph多Agent协作对比")
    print("=" * 60)
    
    # 演示1：LangGraph多Agent实现
    demonstrate_langgraph_multi_agent()
    
    # 演示2：实现对比
    compare_implementation_approaches()
    
    # 演示3：场景分析
    analyze_use_cases()
    
    # 演示4：最佳实践
    summarize_best_practices()
    
    print("\n" + "=" * 60)
    print("所有演示完成！")
    print("=" * 60)


if __name__ == "__main__":
    main()
