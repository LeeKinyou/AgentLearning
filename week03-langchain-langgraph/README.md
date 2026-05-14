# Week03: LangChain与LangGraph框架精通

## 学习目标

掌握主流Agent框架，用可控图构建复杂Agent。

## 目录结构

```
week03-langchain-langgraph/
├── day01_03_langchain_basics/       # Day 1-3: LangChain基础
│   ├── __init__.py
│   └── langchain_basics.py          # Chain、Agent、Tool、Memory演示
├── day04_07_langgraph/              # Day 4-7: LangGraph实战
│   ├── __init__.py
│   └── langgraph_practice.py        # StateGraph、条件分支、工作流演示
└── tests/                           # 测试用例
    ├── __init__.py
    └── tests.py                     # 8个测试类，覆盖所有核心功能
```

## 安装依赖

```bash
pip install langchain langchain-openai langchain-community langgraph chromadb pydantic python-dotenv
```

## 运行方式

### 运行LangChain基础演示

```bash
cd week03-langchain-langgraph/day01_03_langchain_basics
python langchain_basics.py
```

### 运行LangGraph实战演示

```bash
cd week03-langchain-langgraph/day04_07_langgraph
python langgraph_practice.py
```

### 运行测试

```bash
cd week03-langchain-langgraph/tests
python tests.py
```

## 核心知识点

### Day 1-3: LangChain基础

1. **Chain（链）**
   - LCEL（LangChain Expression Language）：使用 `|` 操作符构建处理管道
   - 顺序链：多个步骤按顺序执行
   - 结构化输出：使用Pydantic模型定义输出格式

2. **Tool（工具）**
   - 使用 `@tool` 装饰器定义工具
   - 工具描述用于Agent理解工具用途
   - 内置工具与自定义工具

3. **Memory（记忆）**
   - `ConversationBufferMemory`：保存完整对话历史
   - `ConversationSummaryMemory`：使用LLM总结对话
   - `ConversationBufferWindowMemory`：保存最近N轮对话

4. **Agent**
   - `create_tool_calling_agent`：创建支持工具调用的Agent
   - `AgentExecutor`：执行Agent循环

### Day 4-7: LangGraph实战

1. **核心概念**
   - `StateGraph`：定义图的状态结构
   - `Node`：图中的处理节点（函数）
   - `Edge`：连接节点的边
   - `END`：图结束标记

2. **条件分支**
   - `add_conditional_edges`：根据状态决定下一个节点
   - 路由函数返回 `Literal` 类型指定目标节点

3. **Human-in-the-loop**
   - 在关键节点暂停，等待人工审核
   - 根据审核结果决定后续路径

4. **工作流模式**
   - 线性工作流：A -> B -> C -> END
   - 条件分支：根据状态路由到不同节点
   - 循环：审核不通过时返回修改

## 测试覆盖

| 测试类 | 测试内容 | 测试数 |
|--------|----------|--------|
| TestLangChainBasics | LLM创建、LCEL链、结构化输出 | 3 |
| TestTools | 计算器、日期、文本长度工具 | 3 |
| TestMemory | 缓冲区记忆、多轮对话 | 2 |
| TestLangGraphBasics | StateGraph创建、简单图执行、列表累积 | 3 |
| TestConditionalRouting | 条件边路由 | 1 |
| TestWorkflow | 线性工作流、带循环的工作流 | 2 |
| TestHumanInLoop | 审核通过、审核拒绝 | 2 |
| TestGraphStructure | 节点数量、多分支图 | 2 |

## 关键代码示例

### LCEL链构建

```python
from langchain_core.prompts import ChatPromptTemplate
from langchain_core.output_parsers import StrOutputParser

prompt = ChatPromptTemplate.from_messages([
    ("system", "你是一个助手"),
    ("user", "{question}")
])

chain = prompt | llm | StrOutputParser()
result = chain.invoke({"question": "你好"})
```

### LangGraph StateGraph

```python
from langgraph.graph import StateGraph, END
from typing import TypedDict

class MyState(TypedDict):
    input: str
    result: str

def node_one(state: MyState) -> dict:
    return {"result": f"processed_{state['input']}"}

builder = StateGraph(MyState)
builder.add_node("process", node_one)
builder.set_entry_point("process")
builder.add_edge("process", END)

graph = builder.compile()
result = graph.invoke({"input": "test", "result": ""})
```

### 条件分支

```python
from typing import Literal

def route(state: MyState) -> Literal["path_a", "path_b"]:
    return "path_a" if condition else "path_b"

builder.add_conditional_edges(
    "decision_node",
    route,
    {"path_a": "node_a", "path_b": "node_b"}
)
```
