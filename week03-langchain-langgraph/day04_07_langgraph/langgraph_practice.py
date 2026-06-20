"""
Day 4-7: LangGraph实战

本模块演示LangGraph框架的核心概念和实战项目，包括：
1. 核心概念：
   - State定义与类型安全
   - Node与Edge
   - 条件分支与循环
2. 实战项目：
   - 用Graph构建带记忆的对话Agent
   - 实现条件分支（意图分类路由）
   - 加入Human-in-the-loop机制
3. 实践：用LangGraph实现一个多步骤工作流Agent

学习目标：
- 理解LangGraph的图计算模型
- 掌握StateGraph的使用方法
- 能够构建复杂的Agent工作流
- 学会使用条件分支和人工审核机制

依赖安装：
    pip install langgraph langchain langchain-openai langchain-community pydantic python-dotenv
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
# 第一部分：LLM初始化（复用）
# ============================================================

def create_llm(temperature: float = 0.0, model: str | None = None):
    """
    创建LLM实例
    
    参数：
        temperature: 输出随机性
        model: 模型名称
    
    返回：
        BaseChatModel: LangChain聊天模型实例
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
# 第二部分：LangGraph核心概念演示
# ============================================================

def demonstrate_langgraph_basics():
    """
    演示LangGraph的核心概念
    
    功能说明：
        展示StateGraph的基本用法：
        1. State定义：使用TypedDict定义图的状态
        2. Node定义：每个节点是一个函数
        3. Edge定义：连接节点的边
        4. 编译和运行图
    """
    print("=" * 60)
    print("LangGraph 核心概念演示")
    print("=" * 60)
    
    from langgraph.graph import StateGraph, END, START

    # --- 1. State定义 ---
    # State是图的全局状态，所有节点共享和修改这个状态
    # 使用TypedDict定义类型安全的状态结构
    class BasicState(TypedDict):
        """基础状态定义"""
        input: str           # 用户输入
        result: str          # 处理结果
        steps: List[str]     # 执行步骤记录
    
    # --- 2. Node定义 ---
    # 每个节点是一个函数，接收当前状态，返回状态更新
    def step_one(state: BasicState) -> dict:
        """
        第一步：分析输入
        
        参数：
            state: 当前状态
        
        返回：
            dict: 状态更新（只返回需要更新的字段）
        """
        print(f"  [节点1] 分析输入：{state['input']}")
        return {
            "result": f"已分析：{state['input']}",
            "steps": state.get("steps", []) + ["step_one"]
        }
    
    def step_two(state: BasicState) -> dict:
        """
        第二步：处理数据
        
        参数：
            state: 当前状态
        
        返回：
            dict: 状态更新
        """
        print(f"  [节点2] 处理数据：{state['result']}")
        return {
            "result": state["result"] + " -> 已处理",
            "steps": state.get("steps", []) + ["step_two"]
        }
    
    def step_three(state: BasicState) -> dict:
        """
        第三步：生成输出
        
        参数：
            state: 当前状态
        
        返回：
            dict: 状态更新
        """
        print(f"  [节点3] 生成输出：{state['result']}")
        return {
            "result": state["result"] + " -> 已完成",
            "steps": state.get("steps", []) + ["step_three"]
        }
    
    # --- 3. 构建图 ---
    # 创建StateGraph实例，指定State类型
    graph_builder = StateGraph(BasicState)
    
    # 添加节点（Node）
    # 每个节点有一个名称和对应的处理函数
    graph_builder.add_node("analyze", step_one)
    graph_builder.add_node("process", step_two)
    graph_builder.add_node("finalize", step_three)
    
    # 添加边（Edge）
    # add_edge(START, node): 设置入口节点
    # add_edge: 添加从一个节点到另一个节点的边
    graph_builder.add_edge(START, "analyze")
    graph_builder.add_edge("analyze", "process")
    graph_builder.add_edge("process", "finalize")
    graph_builder.add_edge("finalize", END)  # END表示图结束
    
    # 编译图
    # compile() 将图定义转换为可执行的图
    graph = graph_builder.compile()
    
    # --- 4. 运行图 ---
    print("\n运行基础图：")
    print("-" * 40)
    
    # invoke() 执行图，传入初始状态
    result = graph.invoke({
        "input": "用户请求：生成一份报告",
        "result": "",
        "steps": []
    })
    
    print(f"\n最终结果：{result['result']}")
    print(f"执行步骤：{result['steps']}")
    
    # 可视化图结构
    print(f"\n图结构可视化（文本表示）：")
    print(f"  [入口] -> analyze -> process -> finalize -> [END]")
    
    return graph


# ============================================================
# 第三部分：带记忆的对话Agent
# ============================================================

def demonstrate_conversation_agent():
    """
    演示使用LangGraph构建带记忆的对话Agent
    
    功能说明：
        构建一个能够记住对话历史的Agent。
        使用State来维护对话上下文。
    """
    print("\n" + "=" * 60)
    print("带记忆的对话Agent演示")
    print("=" * 60)
    
    from langgraph.graph import StateGraph, END, START
    from langchain_core.messages import HumanMessage, AIMessage
    from langchain_core.prompts import ChatPromptTemplate, MessagesPlaceholder
    from langchain_core.output_parsers import StrOutputParser
    
    llm = create_llm(temperature=0.7)
    
    # --- 1. 定义状态 ---
    class ConversationState(TypedDict):
        """对话状态"""
        messages: List    # 对话消息历史
        response: str     # 当前响应
    
    # --- 2. 定义节点 ---
    def chat_node(state: ConversationState) -> dict:
        """
        对话节点：接收用户输入，生成回复
        
        参数：
            state: 当前对话状态
        
        返回：
            dict: 状态更新
        """
        # 构建Prompt，包含对话历史
        prompt = ChatPromptTemplate.from_messages([
            ("system", "你是一个友好的对话助手。请根据对话历史回应用户。"),
            MessagesPlaceholder(variable_name="messages"),
        ])
        
        chain = prompt | llm | StrOutputParser()
        
        # 获取最新消息（用户输入）
        last_message = state["messages"][-1]
        
        # 生成回复
        response = chain.invoke({"messages": state["messages"]})
        
        return {
            "messages": [AIMessage(content=response)],
            "response": response
        }
    
    # --- 3. 构建图 ---
    builder = StateGraph(ConversationState)
    builder.add_node("chat", chat_node)
    builder.add_edge(START, "chat")
    builder.add_edge("chat", END)
    
    graph = builder.compile()
    
    # --- 4. 模拟多轮对话 ---
    print("\n模拟多轮对话：")
    print("-" * 40)
    
    # 维护对话历史
    messages = []
    
    user_inputs = [
        "你好，我叫小明",
        "我喜欢编程，特别是Python",
        "你还记得我叫什么吗？",
        "我刚才说了我喜欢什么？"
    ]
    
    for user_input in user_inputs:
        print(f"\n用户：{user_input}")
        
        # 添加用户消息
        messages.append(HumanMessage(content=user_input))
        
        # 执行图
        result = graph.invoke({"messages": messages, "response": ""})
        
        # 更新消息历史
        messages.append(AIMessage(content=result["response"]))
        
        print(f"助手：{result['response']}")
    
    print(f"\n对话历史共 {len(messages)} 条消息")


# ============================================================
# 第四部分：条件分支（意图分类路由）
# ============================================================

def demonstrate_conditional_routing():
    """
    演示使用LangGraph实现条件分支（意图分类路由）
    
    功能说明：
        根据用户输入的意图，路由到不同的处理节点。
        这是构建复杂Agent的关键技术。
    """
    print("\n" + "=" * 60)
    print("条件分支（意图分类路由）演示")
    print("=" * 60)
    
    from langgraph.graph import StateGraph, END, START

    # --- 1. 定义状态 ---
    class RouterState(TypedDict):
        """路由状态"""
        input: str          # 用户输入
        intent: str         # 识别的意图
        response: str       # 响应
        steps: List[str]    # 执行步骤
    
    # --- 2. 定义节点 ---
    def classify_intent(state: RouterState) -> dict:
        """
        意图分类节点
        
        功能说明：
            分析用户输入，识别意图类型。
            这里使用简单的关键词匹配模拟LLM分类。
        
        支持的意图：
            - math: 数学计算
            - date: 日期查询
            - greeting: 问候
            - unknown: 未知意图
        """
        user_input = state["input"].lower()
        
        # 简单的意图分类（实际项目中应使用LLM）
        if any(word in user_input for word in ["计算", "多少", "+", "-", "*", "/", "sqrt"]):
            intent = "math"
        elif any(word in user_input for word in ["日期", "今天", "星期", "时间"]):
            intent = "date"
        elif any(word in user_input for word in ["你好", "hello", "hi", "早上好", "晚上好"]):
            intent = "greeting"
        else:
            intent = "unknown"
        
        print(f"  [分类节点] 识别意图：{intent}")
        
        return {
            "intent": intent,
            "steps": state.get("steps", []) + ["classify"]
        }
    
    def handle_math(state: RouterState) -> dict:
        """数学计算处理节点"""
        import math, re
        print(f"  [数学节点] 处理计算请求")
        
        # 提取表达式并计算
        expression = state["input"]
        # 清理表达式
        expression = expression.replace("计算", "").replace("等于", "").strip()
        
        try:
            safe_expr = expression.replace('sqrt', 'math.sqrt')
            safe_expr = re.sub(r'[^0-9+\-*/().\s\w]', '', safe_expr)
            result = eval(safe_expr, {"__builtins__": {}, "math": math})
            response = f"计算结果：{expression} = {result}"
        except:
            response = "抱歉，我无法解析这个数学表达式。"
        
        return {
            "response": response,
            "steps": state.get("steps", []) + ["math_handler"]
        }
    
    def handle_date(state: RouterState) -> dict:
        """日期查询处理节点"""
        from datetime import datetime
        print(f"  [日期节点] 处理日期查询")
        
        now = datetime.now()
        response = f"当前时间：{now.strftime('%Y年%m月%d日 %H:%M:%S')}，{now.strftime('%A')}"
        
        return {
            "response": response,
            "steps": state.get("steps", []) + ["date_handler"]
        }
    
    def handle_greeting(state: RouterState) -> dict:
        """问候处理节点"""
        print(f"  [问候节点] 处理问候")
        
        response = "你好！很高兴见到你。有什么我可以帮助你的吗？"
        
        return {
            "response": response,
            "steps": state.get("steps", []) + ["greeting_handler"]
        }
    
    def handle_unknown(state: RouterState) -> dict:
        """未知意图处理节点"""
        print(f"  [默认节点] 处理未知意图")
        
        response = "抱歉，我不太理解你的意思。你可以问我数学计算、日期查询等问题。"
        
        return {
            "response": response,
            "steps": state.get("steps", []) + ["unknown_handler"]
        }
    
    # --- 3. 定义条件路由函数 ---
    def route_by_intent(state: RouterState) -> Literal["math", "date", "greeting", "unknown"]:
        """
        路由函数：根据意图决定下一个节点
        
        参数：
            state: 当前状态
        
        返回：
            str: 下一个节点的名称
        
        注意：
            返回值必须是图中已定义的节点名称
        """
        intent: Literal["math", "date", "greeting", "unknown"] = state["intent"]  # type: ignore[assignment]
        return intent
    
    # --- 4. 构建图 ---
    builder = StateGraph(RouterState)
    
    # 添加所有节点
    builder.add_node("classify", classify_intent)
    builder.add_node("math", handle_math)
    builder.add_node("date", handle_date)
    builder.add_node("greeting", handle_greeting)
    builder.add_node("unknown", handle_unknown)
    
    # 设置入口
    builder.add_edge(START, "classify")
    
    # 添加条件边
    # add_conditional_edges: 根据路由函数的返回值决定下一个节点
    builder.add_conditional_edges(
        "classify",           # 从classify节点出发
        route_by_intent,      # 路由函数
        {                     # 路由映射：意图 -> 节点
            "math": "math",
            "date": "date",
            "greeting": "greeting",
            "unknown": "unknown"
        }
    )
    
    # 所有处理节点都通向END
    builder.add_edge("math", END)
    builder.add_edge("date", END)
    builder.add_edge("greeting", END)
    builder.add_edge("unknown", END)
    
    graph = builder.compile()
    
    # --- 5. 测试不同意图 ---
    print("\n测试不同意图的路由：")
    print("-" * 40)
    
    test_inputs = [
        ("你好", "问候意图"),
        ("计算 2 + 3 * 4", "数学意图"),
        ("今天星期几", "日期意图"),
        ("量子力学是什么", "未知意图"),
    ]
    
    for user_input, expected_intent in test_inputs:
        print(f"\n用户：{user_input} （期望：{expected_intent}）")
        result = graph.invoke({
            "input": user_input,
            "intent": "",
            "response": "",
            "steps": []
        })
        print(f"响应：{result['response']}")
        print(f"执行路径：{' -> '.join(result['steps'])}")
    
    return graph


# ============================================================
# 第五部分：Human-in-the-loop机制
# ============================================================

def demonstrate_human_in_loop():
    """
    演示Human-in-the-loop（人工审核）机制
    
    功能说明：
        在Agent执行过程中加入人工审核环节。
        适用于需要人工确认的关键操作场景。
    """
    print("\n" + "=" * 60)
    print("Human-in-the-loop 演示")
    print("=" * 60)
    
    from langgraph.graph import StateGraph, END, START

    # --- 1. 定义状态 ---
    class ApprovalState(TypedDict):
        """审核状态"""
        input: str              # 用户输入
        draft: str              # 生成的草稿
        approved: bool          # 是否通过审核
        final_response: str     # 最终响应
        steps: List[str]        # 执行步骤
    
    # --- 2. 定义节点 ---
    def generate_draft(state: ApprovalState) -> dict:
        """生成草稿节点"""
        print(f"  [草稿节点] 生成回复草稿")
        
        # 模拟生成草稿
        draft = f"草稿：针对'{state['input']}'的回复..."
        
        return {
            "draft": draft,
            "steps": state.get("steps", []) + ["generate_draft"]
        }
    
    def human_review(state: ApprovalState) -> dict:
        """
        人工审核节点
        
        功能说明：
            暂停执行，等待人工审核。
            在实际应用中，这里会等待用户输入。
        """
        print(f"  [审核节点] 等待人工审核...")
        print(f"  草稿内容：{state['draft']}")
        
        # 模拟人工审核（实际应用中应等待用户输入）
        # 这里自动批准
        approved = True
        print(f"  审核结果：{'通过' if approved else '拒绝'}")
        
        return {
            "approved": approved,
            "steps": state.get("steps", []) + ["human_review"]
        }
    
    def publish_response(state: ApprovalState) -> dict:
        """发布响应节点（审核通过后）"""
        print(f"  [发布节点] 发布最终响应")
        
        return {
            "final_response": state["draft"].replace("草稿：", "最终回复："),
            "steps": state.get("steps", []) + ["publish"]
        }
    
    def reject_response(state: ApprovalState) -> dict:
        """拒绝响应节点（审核未通过）"""
        print(f"  [拒绝节点] 拒绝发布")
        
        return {
            "final_response": "回复未通过审核，需要重新生成。",
            "steps": state.get("steps", []) + ["reject"]
        }
    
    # --- 3. 定义路由函数 ---
    def check_approval(state: ApprovalState) -> Literal["publish", "reject"]:
        """根据审核结果决定下一步"""
        if state["approved"]:
            result: Literal["publish", "reject"] = "publish"  # type: ignore[assignment]
            return result
        else:
            result: Literal["publish", "reject"] = "reject"  # type: ignore[assignment]
            return result
    
    # --- 4. 构建图 ---
    builder = StateGraph(ApprovalState)
    
    builder.add_node("generate", generate_draft)
    builder.add_node("review", human_review)
    builder.add_node("publish", publish_response)
    builder.add_node("reject", reject_response)
    
    builder.add_edge(START, "generate")
    builder.add_edge("generate", "review")
    
    builder.add_conditional_edges(
        "review",
        check_approval,
        {"publish": "publish", "reject": "reject"}
    )
    
    builder.add_edge("publish", END)
    builder.add_edge("reject", END)
    
    graph = builder.compile()
    
    # --- 5. 运行 ---
    print("\n运行人工审核流程：")
    print("-" * 40)
    
    result = graph.invoke({
        "input": "请帮我写一封邮件",
        "draft": "",
        "approved": False,
        "final_response": "",
        "steps": []
    })
    
    print(f"\n最终结果：{result['final_response']}")
    print(f"执行路径：{' -> '.join(result['steps'])}")
    
    return graph


# ============================================================
# 第六部分：多步骤工作流Agent
# ============================================================

def demonstrate_workflow_agent():
    """
    演示使用LangGraph实现多步骤工作流Agent
    
    功能说明：
        构建一个完整的报告生成工作流：
        1. 需求分析
        2. 资料收集
        3. 内容撰写
        4. 质量审核
        5. 最终输出
    """
    print("\n" + "=" * 60)
    print("多步骤工作流Agent演示")
    print("=" * 60)
    
    from langgraph.graph import StateGraph, END, START

    # --- 1. 定义状态 ---
    class WorkflowState(TypedDict):
        """工作流状态"""
        topic: str                  # 报告主题
        requirements: str           # 需求分析结果
        research: str               # 资料收集结果
        draft: str                  # 草稿
        review_passed: bool         # 审核是否通过
        review_feedback: str        # 审核反馈
        final_report: str           # 最终报告
        steps: List[str]            # 执行步骤
    
    # --- 2. 定义节点 ---
    def analyze_requirements(state: WorkflowState) -> dict:
        """需求分析节点"""
        print(f"  [需求分析] 主题：{state['topic']}")
        requirements = f"需求：生成关于'{state['topic']}'的报告，要求简洁明了。"
        return {
            "requirements": requirements,
            "steps": state.get("steps", []) + ["analyze_requirements"]
        }
    
    def collect_research(state: WorkflowState) -> dict:
        """资料收集节点"""
        print(f"  [资料收集] 收集相关资料...")
        research = f"资料：关于'{state['topic']}'的关键信息点已收集。"
        return {
            "research": research,
            "steps": state.get("steps", []) + ["collect_research"]
        }
    
    def write_draft(state: WorkflowState) -> dict:
        """内容撰写节点"""
        print(f"  [内容撰写] 撰写报告草稿...")
        draft = f"草稿：\n# {state['topic']}报告\n\n## 概述\n{state['research']}\n\n## 详情\n{state['requirements']}"
        return {
            "draft": draft,
            "steps": state.get("steps", []) + ["write_draft"]
        }
    
    def review_draft(state: WorkflowState) -> dict:
        """质量审核节点"""
        print(f"  [质量审核] 审核报告草稿...")
        
        # 模拟审核（实际应用中应使用LLM或人工审核）
        passed = True
        feedback = "质量良好，可以发布。" if passed else "需要修改：内容不够详细。"
        
        return {
            "review_passed": passed,
            "review_feedback": feedback,
            "steps": state.get("steps", []) + ["review_draft"]
        }
    
    def finalize_report(state: WorkflowState) -> dict:
        """最终发布节点"""
        print(f"  [最终发布] 生成最终报告...")
        final = f"【最终报告】\n{state['draft']}\n\n审核意见：{state['review_feedback']}"
        return {
            "final_report": final,
            "steps": state.get("steps", []) + ["finalize_report"]
        }
    
    def revise_draft(state: WorkflowState) -> dict:
        """修改草稿节点（审核未通过时）"""
        print(f"  [修改草稿] 根据反馈修改...")
        revised = f"{state['draft']}\n\n修改说明：已根据'{state['review_feedback']}'进行修改。"
        return {
            "draft": revised,
            "steps": state.get("steps", []) + ["revise_draft"]
        }
    
    # --- 3. 定义路由函数 ---
    def check_review(state: WorkflowState) -> Literal["finalize", "revise"]:
        """根据审核结果决定下一步"""
        if state["review_passed"]:
            result: Literal["finalize", "revise"] = "finalize"  # type: ignore[assignment]
            return result
        else:
            result: Literal["finalize", "revise"] = "revise"  # type: ignore[assignment]
            return result
    
    # --- 4. 构建图 ---
    builder = StateGraph(WorkflowState)
    
    # 添加节点
    builder.add_node("analyze", analyze_requirements)
    builder.add_node("research", collect_research)
    builder.add_node("write", write_draft)
    builder.add_node("review", review_draft)
    builder.add_node("finalize", finalize_report)
    builder.add_node("revise", revise_draft)
    
    # 添加边
    builder.add_edge(START, "analyze")
    builder.add_edge("analyze", "research")
    builder.add_edge("research", "write")
    builder.add_edge("write", "review")
    
    # 条件分支
    builder.add_conditional_edges(
        "review",
        check_review,
        {"finalize": "finalize", "revise": "revise"}
    )
    
    # 修改后重新审核
    builder.add_edge("revise", "review")
    builder.add_edge("finalize", END)
    
    graph = builder.compile()
    
    # --- 5. 运行工作流 ---
    print("\n运行报告生成工作流：")
    print("-" * 40)
    
    result = graph.invoke({
        "topic": "Python异步编程",
        "requirements": "",
        "research": "",
        "draft": "",
        "review_passed": False,
        "review_feedback": "",
        "final_report": "",
        "steps": []
    })
    
    print(f"\n{'=' * 40}")
    print(f"最终报告：")
    print(f"{'=' * 40}")
    print(result['final_report'])
    print(f"\n执行路径：{' -> '.join(result['steps'])}")
    
    return graph


# ============================================================
# 第七部分：综合演示
# ============================================================

def main():
    """
    主程序入口
    
    运行方式：
        python langgraph_practice.py
    """
    print("=" * 60)
    print("Week03 Day 4-7: LangGraph实战")
    print("=" * 60)
    
    # 演示核心概念
    demonstrate_langgraph_basics()
    
    # 演示带记忆的对话Agent
    demonstrate_conversation_agent()
    
    # 演示条件分支
    demonstrate_conditional_routing()
    
    # 演示人工审核
    demonstrate_human_in_loop()
    
    # 演示多步骤工作流
    demonstrate_workflow_agent()
    
    print("\n" + "=" * 60)
    print("演示完成！")
    print("=" * 60)


if __name__ == "__main__":
    main()
