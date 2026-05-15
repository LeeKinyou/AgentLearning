"""
Day 1-3: CrewAI多Agent协作探索

本模块演示CrewAI框架的核心概念和实战项目，包括：
1. Agent角色定义与分工
2. Task编排与依赖管理
3. 通信机制与结果聚合
4. 完整Demo：多角色协作场景（研究员+写手+审校）
5. 与LangGraph的设计差异对比

学习目标：
- 理解CrewAI的多Agent协作范式
- 掌握Agent、Task、Crew的核心抽象
- 能够构建多角色协作的工作流
- 对比CrewAI与LangGraph的设计哲学差异

依赖安装：
    pip install crewai crewai-tools langchain-openai python-dotenv
"""

import os
import sys
from typing import Optional
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
# 第二部分：CrewAI核心概念演示
# ============================================================

def demonstrate_crewai_basics():
    """
    演示CrewAI的核心概念
    
    功能说明：
        展示CrewAI的基本用法：
        1. Agent定义：角色、目标、背景故事
        2. Task定义：描述、期望输出、工具
        3. Crew编排：Agent和Task的组合
        4. 执行流程：顺序执行与层级执行
    """
    print("=" * 60)
    print("CrewAI 核心概念演示")
    print("=" * 60)
    
    try:
        from crewai import Agent, Task, Crew, Process
    except ImportError:
        print("  [警告] 未安装crewai，请先运行: pip install crewai crewai-tools")
        print("  [跳过] 演示跳过，请安装依赖后重试")
        return None
    
    llm = create_llm(temperature=0.7)
    
    # --- 1. Agent定义 ---
    # Agent是CrewAI的核心抽象，代表一个具有特定角色的智能体
    researcher = Agent(
        role="高级研究员",
        goal="深入研究指定主题，收集全面的信息和数据",
        backstory="""你是一位经验丰富的高级研究员，擅长从多个角度分析问题。
            你能够从海量信息中筛选出关键数据，并提供结构化的研究报告。
            你的研究风格严谨、客观，注重数据的准确性和来源的可靠性。""",
        verbose=True,
        allow_delegation=False,
        llm=llm,
    )
    
    writer = Agent(
        role="内容写手",
        goal="将研究结果转化为通俗易懂、引人入胜的文章",
        backstory="""你是一位资深的内容创作者，擅长将复杂的专业知识
转化为普通读者也能理解的文章。你的写作风格生动、流畅，
善于使用比喻和案例来解释抽象概念。""",
        verbose=True,
        allow_delegation=False,
        llm=llm,
    )
    
    editor = Agent(
        role="审校编辑",
        goal="确保文章质量，检查逻辑、语法和格式",
        backstory="""你是一位严谨的审校编辑，拥有多年的编辑经验。
你擅长发现文章中的逻辑漏洞、语法错误和格式问题。
你的审校标准严格，追求文章的完美呈现。""",
        verbose=True,
        allow_delegation=False,
        llm=llm,
    )
    
    print("\n[Agent定义完成]")
    print(f"  研究员：{researcher.role}")
    print(f"  写手：{writer.role}")
    print(f"  审校：{editor.role}")
    
    # --- 2. Task定义 ---
    # Task定义了Agent需要完成的具体工作
    research_task = Task(
        description="""研究主题：{topic}
        
请完成以下研究任务：
1. 收集该主题的关键信息和最新进展
2. 分析该主题的核心技术和应用场景
3. 总结该主题的优势、挑战和未来趋势
4. 提供结构化的研究报告，包含至少3个主要部分""",
        expected_output="一份结构化的研究报告，包含主题概述、核心技术、应用场景、挑战与趋势",
        agent=researcher,
    )
    
    writing_task = Task(
        description="""根据研究报告，撰写一篇通俗易懂的文章。
        
要求：
1. 使用生动的语言和具体的案例
2. 将专业术语转化为普通读者能理解的内容
3. 文章结构清晰，包含引言、正文和结论
4. 字数在800-1000字左右""",
        expected_output="一篇通俗易懂、结构完整的文章",
        agent=writer,
        context=[research_task],  # 依赖前一个任务的输出
    )
    
    editing_task = Task(
        description="""审校文章，确保质量。
        
检查项：
1. 逻辑是否连贯，是否有矛盾之处
2. 语言是否流畅，是否有语法错误
3. 格式是否规范，段落结构是否合理
4. 提供修改建议和优化后的最终版本""",
        expected_output="审校报告和优化后的最终文章",
        agent=editor,
        context=[writing_task],  # 依赖写作任务
    )
    
    print("\n[Task定义完成]")
    print(f"  研究任务：{research_task.description[:50]}...")
    print(f"  写作任务：{writing_task.description[:50]}...")
    print(f"  审校任务：{editing_task.description[:50]}...")
    
    # --- 3. Crew编排 ---
    # Crew将Agent和Task组合成一个可执行的工作流
    crew = Crew(
        agents=[researcher, writer, editor],
        tasks=[research_task, writing_task, editing_task],
        verbose=True,
        process=Process.sequential,  # 顺序执行
    )
    
    print("\n[Crew编排完成]")
    print(f"  Agent数量：{len(crew.agents)}")
    print(f"  Task数量：{len(crew.tasks)}")
    print(f"  执行模式：顺序执行")
    
    # --- 4. 执行 ---
    print("\n[开始执行Crew...]")
    print("-" * 40)
    
    # 实际执行需要有效的LLM连接，这里展示结构
    # result = crew.kickoff(inputs={"topic": "人工智能在医疗领域的应用"})
    
    print("\n[CrewAI核心概念演示完成]")
    print("  注意：实际执行需要配置有效的LLM API")
    
    return crew


# ============================================================
# 第三部分：完整多Agent协作Demo
# ============================================================

def demonstrate_full_collaboration_demo():
    """
    演示完整的多Agent协作场景
    
    功能说明：
        实现一个联合撰写行业报告的完整流程：
        1. 信息收集Agent：搜索和整理信息
        2. 数据分析Agent：分析数据和趋势
        3. 内容撰写Agent：撰写报告内容
        4. 质量审核Agent：审核和优化报告
    """
    print("\n" + "=" * 60)
    print("多Agent协作完整Demo：联合撰写行业报告")
    print("=" * 60)
    
    try:
        from crewai import Agent, Task, Crew, Process
    except ImportError:
        print("  [警告] 未安装crewai，请先运行: pip install crewai crewai-tools")
        print("  [跳过] 演示跳过，请安装依赖后重试")
        return None
    
    llm = create_llm(temperature=0.7)
    
    # --- 1. 定义Agent角色 ---
    print("\n[步骤1] 定义Agent角色")
    print("-" * 40)
    
    information_collector = Agent(
        role="信息收集专家",
        goal="全面收集指定主题的相关信息",
        backstory="""你是一位专业的信息收集专家，擅长从多个渠道
                    收集高质量的信息。你能够快速筛选出有价值的资料，
                    并提供结构化的信息摘要。""",
        verbose=True,
        allow_delegation=False,
        llm=llm,
    )
    
    data_analyst = Agent(
        role="数据分析师",
        goal="分析收集到的信息，提取关键洞察",
        backstory="""你是一位资深的数据分析师，擅长从复杂的数据中
                    发现模式和趋势。你能够将原始数据转化为有价值的洞察，
                    并提供数据支持的建议。""",  
        verbose=True,
        allow_delegation=False,
        llm=llm,
    )
    
    content_writer = Agent(
        role="报告撰写专家",
        goal="将分析和洞察转化为专业的行业报告",
        backstory="""你是一位专业的报告撰写专家，擅长撰写结构清晰、
                    内容详实的行业报告。你的报告风格专业、客观，
                    注重数据的准确性和论证的逻辑性。""",
        verbose=True,
        allow_delegation=False,
        llm=llm,
    )
    
    quality_reviewer = Agent(
        role="质量审核专家",
        goal="确保报告质量，提供改进建议",
        backstory="""你是一位严格的质量审核专家，拥有多年的行业报告
                审核经验。你能够从内容、结构、逻辑等多个维度评估报告质量，
                并提供具体的改进建议。""",
        verbose=True,
        allow_delegation=False,
        llm=llm,
    )
    
    print("  ✓ 信息收集专家")
    print("  ✓ 数据分析师")
    print("  ✓ 报告撰写专家")
    print("  ✓ 质量审核专家")
    
    # --- 2. 定义Task和依赖 ---
    print("\n[步骤2] 定义Task和依赖关系")
    print("-" * 40)
    
    # Task 1: 信息收集
    collect_info_task = Task(
        description="""收集关于"{topic}"的最新信息。
            任务要求：
            1. 收集该领域的最新发展和趋势
            2. 收集主要参与者和市场动态
            3. 收集相关的技术进展和应用案例
            4. 整理成结构化的信息摘要""",
        expected_output="结构化的信息摘要，包含最新动态、主要参与者、技术进展和应用案例",
        agent=information_collector,
    )
    
    # Task 2: 数据分析（依赖Task 1）
    analyze_data_task = Task(
        description="""分析收集到的信息，提取关键洞察。
            任务要求：
            1. 识别主要趋势和模式
            2. 分析市场机会和挑战
            3. 评估技术成熟度和应用前景
            4. 提供数据支持的建议""",
        expected_output="数据分析报告，包含趋势分析、市场评估和建议",
        agent=data_analyst,
        context=[collect_info_task],
    )
    
    # Task 3: 内容撰写（依赖Task 2）
    write_report_task = Task(
        description="""根据数据分析结果，撰写专业的行业报告。
            报告结构：
            1. 执行摘要（200字）
            2. 行业概述（300字）
            3. 市场分析（400字）
            4. 技术趋势（300字）
            5. 发展建议（300字）
            6. 结论（100字）""",
        expected_output="完整的行业报告，包含所有要求的章节",
        agent=content_writer,
        context=[analyze_data_task],
    )
    
    # Task 4: 质量审核（依赖Task 3）
    review_report_task = Task(
        description="""审核行业报告，确保质量。
            审核维度：
            1. 内容完整性：是否覆盖所有关键方面
            2. 数据准确性：数据和事实是否准确
            3. 逻辑连贯性：论证是否严密
            4. 格式规范性：结构是否清晰
            5. 提供具体的改进建议和最终版本""",
        expected_output="审核报告和优化后的最终行业报告",
        agent=quality_reviewer,
        context=[write_report_task],
    )
    
    print("  ✓ 信息收集任务")
    print("  ✓ 数据分析任务（依赖：信息收集）")
    print("  ✓ 报告撰写任务（依赖：数据分析）")
    print("  ✓ 质量审核任务（依赖：报告撰写）")
    
    # --- 3. 创建Crew ---
    print("\n[步骤3] 创建Crew工作流")
    print("-" * 40)
    
    crew = Crew(
        agents=[information_collector, data_analyst, content_writer, quality_reviewer],
        tasks=[collect_info_task, analyze_data_task, write_report_task, review_report_task],
        verbose=True,
        process=Process.sequential,
    )
    
    print(f"  ✓ Agent数量：{len(crew.agents)}")
    print(f"  ✓ Task数量：{len(crew.tasks)}")
    print(f"  ✓ 执行模式：顺序执行")
    
    # --- 4. 执行工作流 ---
    print("\n[步骤4] 执行工作流")
    print("-" * 40)
    
    topic = "生成式AI在企业服务中的应用"
    print(f"  主题：{topic}")
    print("\n  [执行中...]")
    print("  注意：实际执行需要配置有效的LLM API")
    print("  这里展示工作流结构和执行逻辑")
    
    # 实际执行代码（需要有效的LLM配置）
    # result = crew.kickoff(inputs={"topic": topic})
    # print(f"\n[执行完成]")
    # print(f"  最终输出：{result.raw[:500]}...")
    
    print("\n[多Agent协作Demo完成]")
    print("  工作流结构：")
    print("    信息收集 → 数据分析 → 报告撰写 → 质量审核")
    
    return crew


# ============================================================
# 第四部分：CrewAI与LangGraph对比
# ============================================================

def demonstrate_framework_comparison():
    """
    对比CrewAI与LangGraph的设计差异
    
    功能说明：
        从多个维度对比两个框架：
        1. 设计哲学
        2. 抽象层次
        3. 工作流模式
        4. 适用场景
        5. 学习曲线
    """
    print("\n" + "=" * 60)
    print("CrewAI vs LangGraph 框架对比")
    print("=" * 60)
    
    comparison_table = """
┌─────────────────────┬─────────────────────────┬─────────────────────────┐
│      对比维度       │        CrewAI           │       LangGraph         │
├─────────────────────┼─────────────────────────┼─────────────────────────┤
│ 设计哲学            │ 多Agent协作优先          │ 图计算模型优先           │
│                     │ 强调角色分工             │ 强调状态流转             │
├─────────────────────┼─────────────────────────┼─────────────────────────┤
│ 核心抽象            │ Agent, Task, Crew        │ State, Node, Edge       │
│                     │ 高层业务抽象             │ 底层计算抽象             │
├─────────────────────┼─────────────────────────┼─────────────────────────┤
│ 工作流模式          │ 顺序/层级/Consensus      │ 任意有向图               │
│                     │ 预定义模式               │ 完全自定义               │
├─────────────────────┼─────────────────────────┼─────────────────────────┤
│ Agent通信           │ 通过Task上下文传递        │ 通过State共享            │
│                     │ 隐式依赖管理             │ 显式状态管理             │
├─────────────────────┼─────────────────────────┼─────────────────────────┤
│ 适用场景            │ 多角色协作任务            │ 复杂工作流控制           │
│                     │ 内容创作、研究报告        │ 条件分支、循环、审核     │
├─────────────────────┼─────────────────────────┼─────────────────────────┤
│ 学习曲线            │ 较低（业务导向）          │ 较高（技术导向）         │
├─────────────────────┼─────────────────────────┼─────────────────────────┤
│ 灵活性              │ 中等（预定义模式）        │ 高（完全自定义）         │
├─────────────────────┼─────────────────────────┼─────────────────────────┤
│ Human-in-the-loop   │ 支持但需自定义            │ 原生支持                │
├─────────────────────┼─────────────────────────┼─────────────────────────┤
│ 调试可观测性        │ 中等                     │ 高（状态可视化）         │
└─────────────────────┴─────────────────────────┴─────────────────────────┘
    """
    
    print(comparison_table)
    
    print("\n[选择建议]")
    print("-" * 40)
    print("""
        选择CrewAI的场景：
        ✓ 需要快速构建多角色协作系统
        ✓ 任务之间有明确的依赖关系
        ✓ 关注业务逻辑而非技术细节
        ✓ 团队背景偏业务而非技术

        选择LangGraph的场景：
        ✓ 需要精细控制工作流执行
        ✓ 复杂的条件分支和循环逻辑
        ✓ 需要Human-in-the-loop机制
        ✓ 需要状态可视化和调试
        ✓ 构建生产级Agent系统

        混合使用：
        ✓ 用LangGraph构建核心工作流
        ✓ 用CrewAI封装多Agent协作模块
        ✓ 根据场景灵活选择
            """)


# ============================================================
# 第五部分：设计模式总结
# ============================================================

def summarize_design_patterns():
    """
    总结多Agent系统的设计模式与常见陷阱
    
    功能说明：
        归纳多Agent系统的最佳实践：
        1. 常见设计模式
        2. 通信模式
        3. 常见陷阱
        4. 优化建议
    """
    print("\n" + "=" * 60)
    print("多Agent系统设计模式总结")
    print("=" * 60)
    
    print("\n[常见设计模式]")
    print("-" * 40)
    print("""
    1. 流水线模式（Pipeline）
    - Agent按顺序执行，前一个输出是后一个输入
    - 适用：报告撰写、内容创作
    - 示例：收集 → 分析 → 撰写 → 审核

    2. 协作模式（Collaboration）
    - 多个Agent并行工作，最后聚合结果
    - 适用：多角度分析、并行研究
    - 示例：研究员A + 研究员B → 汇总分析

    3. 评审模式（Review）
    - 一个Agent生成，另一个Agent审核
    - 适用：质量保证、错误检查
    - 示例：撰写 → 审核 → 修改 → 发布

    4. 路由模式（Routing）
    - 根据输入类型路由到不同Agent
    - 适用：意图分类、任务分发
    - 示例：用户请求 → 分类 → 专业Agent处理

    5. 层级模式（Hierarchy）
    - Manager Agent分配任务给Worker Agent
    - 适用：复杂任务分解
    - 示例：项目经理 → 分配任务 → 团队成员
        """)
    
    print("\n[通信模式]")
    print("-" * 40)
    print("""
    1. 直接传递
    - Agent A的输出直接作为Agent B的输入
    - 优点：简单直接
    - 缺点：耦合度高

    2. 共享状态
    - 所有Agent读写共享的状态空间
    - 优点：灵活
    - 缺点：需要状态管理

    3. 消息队列
    - Agent通过消息队列异步通信
    - 优点：解耦、可扩展
    - 缺点：复杂度高
        """)
    
    print("\n[常见陷阱]")
    print("-" * 40)
    print("""
    1. 过度设计
    - 问题：使用过多Agent，增加复杂度
    - 解决：从简单开始，按需增加Agent

    2. 信息丢失
    - 问题：Agent间传递信息不完整
    - 解决：明确定义输入输出格式

    3. 循环依赖
    - 问题：Agent之间形成循环等待
    - 解决：设计有向无环图（DAG）

    4. 成本失控
    - 问题：多Agent导致LLM调用次数激增
    - 解决：设置Token预算，优化Prompt

    5. 调试困难
    - 问题：多Agent系统难以定位问题
    - 解决：加入日志和可观测性
        """)
    
    print("\n[优化建议]")
    print("-" * 40)
    print("""
    1. 明确Agent边界
    - 每个Agent职责单一、明确
    - 避免Agent功能重叠

    2. 标准化接口
    - 定义清晰的输入输出格式
    - 使用Pydantic等工具验证

    3. 加入可观测性
    - 记录每个Agent的执行日志
    - 使用LangFuse等工具追踪

    4. 设置超时和重试
    - 防止单个Agent卡死整个流程
    - 实现优雅降级

    5. 缓存中间结果
    - 避免重复计算
    - 提高执行效率
        """)


# ============================================================
# 主函数
# ============================================================

def main():
    """
    主函数：运行所有演示
    
    执行流程：
        1. CrewAI核心概念演示
        2. 完整多Agent协作Demo
        3. 框架对比分析
        4. 设计模式总结
    """
    print("=" * 60)
    print("Week04 Day01-03: CrewAI多Agent协作探索")
    print("=" * 60)
    
    # 演示1：核心概念
    demonstrate_crewai_basics()
    
    # 演示2：完整协作Demo
    demonstrate_full_collaboration_demo()
    
    # 演示3：框架对比
    demonstrate_framework_comparison()
    
    # 演示4：设计模式总结
    summarize_design_patterns()
    
    print("\n" + "=" * 60)
    print("所有演示完成！")
    print("=" * 60)


if __name__ == "__main__":
    main()
