# Week04: 多Agent协作与框架对比

## 学习目标

理解多Agent协作范式，掌握至少一种多Agent框架。

## 目录结构

```
week04-multi-agent/
├── day01_03_crewai/                     # Day 1-3: CrewAI探索
│   ├── __init__.py
│   └── crewai_demo.py                   # CrewAI核心概念与完整Demo
├── day04_05_multi_agent_comparison/     # Day 4-5: LangGraph多Agent对比
│   ├── __init__.py
│   └── multi_agent_comparison.py        # LangGraph多Agent实现与框架对比
└── tests/                               # 测试用例
    ├── __init__.py
    └── tests.py                         # 测试覆盖核心功能
```

## 安装依赖

```bash
# 基础依赖（已在pyproject.toml中）
pip install langgraph langchain langchain-openai pydantic python-dotenv

# CrewAI依赖（需要额外安装）
pip install crewai crewai-tools
```

## 运行方式

### 运行CrewAI演示

```bash
cd week04-multi-agent/day01_03_crewai
python crewai_demo.py
```

### 运行LangGraph多Agent对比演示

```bash
cd week04-multi-agent/day04_05_multi_agent_comparison
python multi_agent_comparison.py
```

### 运行测试

```bash
cd week04-multi-agent/tests
python tests.py
```

## 核心知识点

### Day 1-3: CrewAI探索

1. **核心抽象**
   - `Agent`：具有角色、目标、背景故事的智能体
   - `Task`：定义具体任务，通过`context`管理依赖
   - `Crew`：将Agent和Task组合成工作流
   - `Process`：执行模式（sequential/hierarchical）

2. **Agent角色定义**
   - `role`：Agent的角色名称
   - `goal`：Agent的目标
   - `backstory`：Agent的背景故事，影响行为风格
   - `allow_delegation`：是否允许委派任务

3. **Task编排**
   - `description`：任务描述
   - `expected_output`：期望输出格式
   - `context`：依赖的前置任务
   - `agent`：执行任务的Agent

4. **通信机制**
   - 通过Task的`context`传递前置任务输出
   - 隐式依赖管理，无需显式定义数据流

### Day 4-5: LangGraph多Agent对比

1. **LangGraph多Agent实现**
   - 使用`StateGraph`定义全局状态
   - 每个Agent是一个Node（函数）
   - 通过State共享数据
   - 支持条件分支和循环

2. **框架对比**

| 对比维度 | CrewAI | LangGraph |
|---------|--------|-----------|
| 设计哲学 | 多Agent协作优先 | 图计算模型优先 |
| 核心抽象 | Agent, Task, Crew | State, Node, Edge |
| 工作流模式 | 顺序/层级/Consensus | 任意有向图 |
| Agent通信 | 通过Task上下文传递 | 通过State共享 |
| 适用场景 | 多角色协作任务 | 复杂工作流控制 |
| 学习曲线 | 较低（业务导向） | 较高（技术导向） |
| 灵活性 | 中等（预定义模式） | 高（完全自定义） |
| Human-in-the-loop | 支持但需自定义 | 原生支持 |

3. **选择建议**

选择CrewAI的场景：
- 需要快速构建多角色协作系统
- 任务之间有明确的依赖关系
- 关注业务逻辑而非技术细节
- 团队背景偏业务而非技术

选择LangGraph的场景：
- 需要精细控制工作流执行
- 复杂的条件分支和循环逻辑
- 需要Human-in-the-loop机制
- 需要状态可视化和调试
- 构建生产级Agent系统

## 测试覆盖

| 测试类 | 测试内容 | 测试数 |
|--------|----------|--------|
| TestCrewAIBasics | Agent定义、Task定义、Crew结构 | 3 |
| TestLangGraphMultiAgent | State定义、图创建、图执行、条件路由、循环机制 | 5 |
| TestFrameworkComparison | LLM创建、设计模式文档 | 2 |
| TestMultiAgentWorkflow | 工作流结构、任务依赖 | 2 |

## 关键代码示例

### CrewAI多Agent协作

```python
from crewai import Agent, Task, Crew, Process

# 定义Agent
researcher = Agent(
    role="研究员",
    goal="研究指定主题",
    backstory="你是经验丰富的研究员",
    llm=llm,
)

writer = Agent(
    role="写手",
    goal="撰写文章",
    backstory="你是资深内容创作者",
    llm=llm,
)

# 定义Task（通过context指定依赖）
research_task = Task(
    description="研究主题：{topic}",
    expected_output="研究报告",
    agent=researcher,
)

writing_task = Task(
    description="根据研究报告撰写文章",
    expected_output="文章",
    agent=writer,
    context=[research_task],  # 依赖研究任务
)

# 创建Crew并执行
crew = Crew(
    agents=[researcher, writer],
    tasks=[research_task, writing_task],
    process=Process.sequential,
)
result = crew.kickoff(inputs={"topic": "AI发展趋势"})
```

### LangGraph多Agent协作

```python
from langgraph.graph import StateGraph, END
from typing import TypedDict, List

# 定义State
class MultiAgentState(TypedDict):
    topic: str
    research_result: str
    article: str
    steps: List[str]

# 定义Agent节点
def researcher_node(state: MultiAgentState) -> dict:
    return {
        "research_result": f"关于{state['topic']}的研究",
        "steps": state.get("steps", []) + ["research"]
    }

def writer_node(state: MultiAgentState) -> dict:
    return {
        "article": f"基于研究的文章",
        "steps": state.get("steps", []) + ["writing"]
    }

# 构建图
builder = StateGraph(MultiAgentState)
builder.add_node("research", researcher_node)
builder.add_node("writing", writer_node)
builder.set_entry_point("research")
builder.add_edge("research", "writing")
builder.add_edge("writing", END)

graph = builder.compile()
result = graph.invoke({
    "topic": "AI发展趋势",
    "research_result": "",
    "article": "",
    "steps": []
})
```

### 条件分支与循环

```python
from typing import Literal

def review_node(state: ReviewState) -> dict:
    # 审核逻辑
    approved = check_quality(state["draft"])
    return {"approved": approved}

def route_after_review(state: ReviewState) -> Literal["publish", "revise"]:
    return "publish" if state["approved"] else "revise"

builder.add_conditional_edges(
    "review",
    route_after_review,
    {"publish": "publish", "revise": "revise"}
)

builder.add_edge("revise", "review")  # 循环
builder.add_edge("publish", END)
```

## 设计模式

### 1. 流水线模式（Pipeline）

```
Agent A → Agent B → Agent C
```

适用：内容创作、报告生成

### 2. 并行模式（Parallel）

```
Agent A ┐
          ├ → 聚合
Agent B ┘
```

适用：多角度分析、并行研究

### 3. 评审循环模式（Review Loop）

```
生成 → 审核 → [不通过] → 修改 → 审核 → [通过] → 输出
```

适用：质量保证、代码审查

### 4. 路由模式（Router）

```
输入 → 分类 → Agent A/B/C
```

适用：智能客服、任务分发

### 5. 层级模式（Hierarchy）

```
Manager → Worker A, Worker B, Worker C
```

适用：复杂任务分解

## 最佳实践

1. **从简单开始**
   - 先用单Agent验证核心逻辑
   - 逐步增加Agent和复杂度

2. **明确Agent边界**
   - 每个Agent职责单一
   - 避免功能重叠

3. **管理状态和依赖**
   - CrewAI：明确Task的context依赖
   - LangGraph：设计合理的State结构

4. **加入可观测性**
   - 记录每个Agent的执行日志
   - 使用LangFuse追踪调用链

5. **设置超时和重试**
   - 防止单个Agent卡死整个流程
   - 实现优雅降级

6. **缓存中间结果**
   - 避免重复计算
   - 提高执行效率

## 常见陷阱

1. **过度设计**
   - 问题：使用过多Agent，增加复杂度
   - 解决：评估是否真的需要多Agent

2. **信息丢失**
   - 问题：Agent间传递信息不完整
   - 解决：使用Pydantic验证输入输出

3. **成本失控**
   - 问题：多Agent导致LLM调用激增
   - 解决：设置Token预算，优化Prompt

4. **调试困难**
   - 问题：难以定位问题所在Agent
   - 解决：加入详细日志和追踪

5. **死循环**
   - 问题：Agent间形成循环等待
   - 解决：设计DAG，设置最大循环次数
