"""
Week03 测试用例

测试范围：
1. LangChain基础：Chain、Tool、Memory
2. LangGraph核心：StateGraph、Node、Edge
3. 条件分支：意图路由
4. 工作流：多步骤流程
5. Human-in-the-loop：人工审核

运行方式：
    cd week03-langchain-langgraph/tests
    python tests.py
"""

import os
import sys
import unittest
from typing import List, TypedDict

# 添加week03根目录到路径，支持包导入
_week03_root = os.path.abspath(os.path.join(os.path.dirname(__file__), '..'))
if _week03_root not in sys.path:
    sys.path.insert(0, _week03_root)


# ============================================================
# 测试1：LangChain Chain测试
# ============================================================

class TestLangChainBasics(unittest.TestCase):
    """LangChain基础功能测试"""
    
    def test_llm_creation(self):
        """测试LLM实例创建"""
        from day01_03_langchain_basics.langchain_basics import create_llm
        
        llm = create_llm(temperature=0.5)
        self.assertIsNotNone(llm)
        self.assertEqual(llm.temperature, 0.5)
    
    def test_lcel_chain_structure(self):
        """测试LCEL链结构"""
        from langchain_core.prompts import ChatPromptTemplate
        from langchain_core.output_parsers import StrOutputParser
        
        # 构建链结构（不调用LLM）
        prompt = ChatPromptTemplate.from_messages([
            ("system", "你是一个助手"),
            ("user", "{question}")
        ])
        
        # 验证Prompt结构
        self.assertEqual(len(prompt.messages), 2)
    
    def test_structured_output_definition(self):
        """测试结构化输出定义"""
        from pydantic import BaseModel, Field
        
        class BookRecommendation(BaseModel):
            title: str = Field(description="书名")
            author: str = Field(description="作者")
            reason: str = Field(description="推荐理由")
            difficulty: str = Field(description="难度级别")
        
        # 验证字段定义
        self.assertEqual(BookRecommendation.model_fields["title"].description, "书名")
        self.assertEqual(BookRecommendation.model_fields["author"].description, "作者")


# ============================================================
# 测试2：Tool测试
# ============================================================

class TestTools(unittest.TestCase):
    """工具功能测试"""
    
    def test_calculator_tool(self):
        """测试计算器工具"""
        from langchain_core.tools import tool
        
        @tool
        def calculator(expression: str) -> str:
            """计算数学表达式"""
            import math, re
            try:
                safe_expr = expression.replace('sqrt', 'math.sqrt')
                safe_expr = re.sub(r'[^0-9+\-*/().\s\w]', '', safe_expr)
                result = eval(safe_expr, {"__builtins__": {}, "math": math})
                return str(result)
            except Exception as e:
                return f"计算错误：{str(e)}"
        
        # 测试基本运算
        result = calculator.invoke("2 + 3")
        self.assertEqual(result, "5")
        
        # 测试乘法优先级
        result = calculator.invoke("2 + 3 * 4")
        self.assertEqual(result, "14")
    
    def test_date_tool(self):
        """测试日期工具"""
        from langchain_core.tools import tool
        from datetime import datetime
        
        @tool
        def get_current_date(format: str = "%Y-%m-%d") -> str:
            """获取当前日期"""
            return datetime.now().strftime(format)
        
        # 使用空字典调用，触发默认参数
        result = get_current_date.invoke({})
        # 验证返回的是日期格式
        self.assertTrue(len(result) >= 8)  # YYYY-MM-DD
    
    def test_text_length_tool(self):
        """测试文本长度工具"""
        from langchain_core.tools import tool
        
        @tool
        def text_length(text: str) -> str:
            """计算文本长度"""
            chars = len(text)
            words = len(text.split())
            return f"字符数：{chars}，单词数：{words}"
        
        result = text_length.invoke("Hello World")
        self.assertIn("字符数：11", result)
        self.assertIn("单词数：2", result)


# ============================================================
# 测试3：Memory测试
# ============================================================

class TestMemory(unittest.TestCase):
    """记忆系统测试"""
    
    def test_buffer_memory(self):
        """测试缓冲区记忆"""
        from langchain_core.chat_history import InMemoryChatMessageHistory
        from langchain_core.messages import HumanMessage, AIMessage
        
        # 使用LangChain Core的内存聊天历史
        history = InMemoryChatMessageHistory()
        
        # 添加对话
        history.add_message(HumanMessage(content="你好"))
        history.add_message(AIMessage(content="你好！"))
        
        # 验证记忆内容
        messages = history.messages
        self.assertEqual(len(messages), 2)
        self.assertEqual(messages[0].content, "你好")
        self.assertEqual(messages[1].content, "你好！")
    
    def test_buffer_memory_multiple_turns(self):
        """测试多轮对话记忆"""
        from langchain_core.chat_history import InMemoryChatMessageHistory
        from langchain_core.messages import HumanMessage, AIMessage
        
        history = InMemoryChatMessageHistory()
        
        # 添加多轮对话
        history.add_message(HumanMessage(content="第一轮"))
        history.add_message(AIMessage(content="回复1"))
        history.add_message(HumanMessage(content="第二轮"))
        history.add_message(AIMessage(content="回复2"))
        history.add_message(HumanMessage(content="第三轮"))
        history.add_message(AIMessage(content="回复3"))
        
        self.assertEqual(len(history.messages), 6)


# ============================================================
# 测试4：LangGraph StateGraph测试
# ============================================================

class TestLangGraphBasics(unittest.TestCase):
    """LangGraph基础测试"""
    
    def test_state_graph_creation(self):
        """测试StateGraph创建"""
        from langgraph.graph import StateGraph, END
        
        class TestState(TypedDict):
            value: str
        
        builder = StateGraph(TestState)
        self.assertIsNotNone(builder)
    
    def test_simple_graph_execution(self):
        """测试简单图执行"""
        from langgraph.graph import StateGraph, END
        
        class TestState(TypedDict):
            input: str
            result: str
        
        def step_one(state: TestState) -> dict:
            return {"result": f"processed_{state['input']}"}
        
        def step_two(state: TestState) -> dict:
            return {"result": f"final_{state['result']}"}
        
        builder = StateGraph(TestState)
        builder.add_node("one", step_one)
        builder.add_node("two", step_two)
        builder.set_entry_point("one")
        builder.add_edge("one", "two")
        builder.add_edge("two", END)
        
        graph = builder.compile()
        result = graph.invoke({"input": "test", "result": ""})
        
        self.assertEqual(result["result"], "final_processed_test")
    
    def test_graph_with_list_accumulation(self):
        """测试列表累积（Annotated）"""
        from langgraph.graph import StateGraph, END
        from typing import Annotated
        import operator
        
        class AccumState(TypedDict):
            items: Annotated[list, operator.add]
        
        def add_item(state: AccumState) -> dict:
            return {"items": ["item1"]}
        
        def add_another(state: AccumState) -> dict:
            return {"items": ["item2"]}
        
        builder = StateGraph(AccumState)
        builder.add_node("first", add_item)
        builder.add_node("second", add_another)
        builder.set_entry_point("first")
        builder.add_edge("first", "second")
        builder.add_edge("second", END)
        
        graph = builder.compile()
        result = graph.invoke({"items": []})
        
        self.assertEqual(result["items"], ["item1", "item2"])


# ============================================================
# 测试5：条件分支测试
# ============================================================

class TestConditionalRouting(unittest.TestCase):
    """条件分支测试"""
    
    def test_conditional_edges(self):
        """测试条件边"""
        from langgraph.graph import StateGraph, END
        from typing import Literal
        
        class RouteState(TypedDict):
            input: str
            route: str
            result: str
        
        def classify(state: RouteState) -> dict:
            if "math" in state["input"]:
                return {"route": "math"}
            else:
                return {"route": "text"}
        
        def math_handler(state: RouteState) -> dict:
            return {"result": "数学处理结果"}
        
        def text_handler(state: RouteState) -> dict:
            return {"result": "文本处理结果"}
        
        def route_func(state: RouteState) -> Literal["math", "text"]:
            route: Literal["math", "text"] = state["route"]  # type: ignore[assignment]
            return route
        
        builder = StateGraph(RouteState)
        builder.add_node("classify", classify)
        builder.add_node("math", math_handler)
        builder.add_node("text", text_handler)
        
        builder.set_entry_point("classify")
        builder.add_conditional_edges(
            "classify",
            route_func,
            {"math": "math", "text": "text"}
        )
        builder.add_edge("math", END)
        builder.add_edge("text", END)
        
        graph = builder.compile()
        
        # 测试数学路由
        result = graph.invoke({"input": "math problem", "route": "", "result": ""})
        self.assertEqual(result["result"], "数学处理结果")
        
        # 测试文本路由
        result = graph.invoke({"input": "text content", "route": "", "result": ""})
        self.assertEqual(result["result"], "文本处理结果")


# ============================================================
# 测试6：工作流测试
# ============================================================

class TestWorkflow(unittest.TestCase):
    """工作流测试"""
    
    def test_linear_workflow(self):
        """测试线性工作流"""
        from langgraph.graph import StateGraph, END
        
        class WorkflowState(TypedDict):
            step1: str
            step2: str
            step3: str
        
        def process_1(state: WorkflowState) -> dict:
            return {"step1": "完成步骤1"}
        
        def process_2(state: WorkflowState) -> dict:
            return {"step2": "完成步骤2"}
        
        def process_3(state: WorkflowState) -> dict:
            return {"step3": "完成步骤3"}
        
        builder = StateGraph(WorkflowState)
        builder.add_node("s1", process_1)
        builder.add_node("s2", process_2)
        builder.add_node("s3", process_3)
        
        builder.set_entry_point("s1")
        builder.add_edge("s1", "s2")
        builder.add_edge("s2", "s3")
        builder.add_edge("s3", END)
        
        graph = builder.compile()
        result = graph.invoke({"step1": "", "step2": "", "step3": ""})
        
        self.assertEqual(result["step1"], "完成步骤1")
        self.assertEqual(result["step2"], "完成步骤2")
        self.assertEqual(result["step3"], "完成步骤3")
    
    def test_workflow_with_loop(self):
        """测试带循环的工作流"""
        from langgraph.graph import StateGraph, END
        from typing import Literal
        
        class LoopState(TypedDict):
            count: int
            done: bool
        
        def increment(state: LoopState) -> dict:
            new_count = state["count"] + 1
            return {"count": new_count, "done": new_count >= 3}
        
        def finish(state: LoopState) -> dict:
            return {"done": True}
        
        def check_loop(state: LoopState) -> Literal["increment", "finish"]:
            if state["count"] >= 3:
                return "finish"
            return "increment"
        
        builder = StateGraph(LoopState)
        builder.add_node("increment", increment)
        builder.add_node("finish", finish)
        
        builder.set_entry_point("increment")
        builder.add_conditional_edges("increment", check_loop)
        builder.add_edge("finish", END)
        
        graph = builder.compile()
        result = graph.invoke({"count": 0, "done": False})
        
        self.assertEqual(result["count"], 3)
        self.assertTrue(result["done"])


# ============================================================
# 测试7：Human-in-the-loop测试
# ============================================================

class TestHumanInLoop(unittest.TestCase):
    """人工审核测试"""
    
    def test_approval_flow(self):
        """测试审核通过流程"""
        from langgraph.graph import StateGraph, END
        from typing import Literal
        
        class ApprovalState(TypedDict):
            draft: str
            approved: bool
            final: str
        
        def generate(state: ApprovalState) -> dict:
            return {"draft": "草稿内容"}
        
        def review(state: ApprovalState) -> dict:
            return {"approved": True}
        
        def publish(state: ApprovalState) -> dict:
            return {"final": f"发布：{state['draft']}"}
        
        def reject(state: ApprovalState) -> dict:
            return {"final": "拒绝发布"}
        
        def check(state: ApprovalState) -> Literal["publish", "reject"]:
            return "publish" if state["approved"] else "reject"
        
        builder = StateGraph(ApprovalState)
        builder.add_node("generate", generate)
        builder.add_node("review", review)
        builder.add_node("publish", publish)
        builder.add_node("reject", reject)
        
        builder.set_entry_point("generate")
        builder.add_edge("generate", "review")
        builder.add_conditional_edges("review", check)
        builder.add_edge("publish", END)
        builder.add_edge("reject", END)
        
        graph = builder.compile()
        result = graph.invoke({"draft": "", "approved": False, "final": ""})
        
        self.assertEqual(result["final"], "发布：草稿内容")
    
    def test_rejection_flow(self):
        """测试审核拒绝流程"""
        from langgraph.graph import StateGraph, END
        from typing import Literal
        
        class ApprovalState(TypedDict):
            draft: str
            approved: bool
            final: str
        
        def generate(state: ApprovalState) -> dict:
            return {"draft": "草稿内容"}
        
        def review(state: ApprovalState) -> dict:
            return {"approved": False}
        
        def publish(state: ApprovalState) -> dict:
            return {"final": f"发布：{state['draft']}"}
        
        def reject(state: ApprovalState) -> dict:
            return {"final": "拒绝发布"}
        
        def check(state: ApprovalState) -> Literal["publish", "reject"]:
            return "publish" if state["approved"] else "reject"
        
        builder = StateGraph(ApprovalState)
        builder.add_node("generate", generate)
        builder.add_node("review", review)
        builder.add_node("publish", publish)
        builder.add_node("reject", reject)
        
        builder.set_entry_point("generate")
        builder.add_edge("generate", "review")
        builder.add_conditional_edges("review", check)
        builder.add_edge("publish", END)
        builder.add_edge("reject", END)
        
        graph = builder.compile()
        result = graph.invoke({"draft": "", "approved": False, "final": ""})
        
        self.assertEqual(result["final"], "拒绝发布")


# ============================================================
# 测试8：LangGraph图结构测试
# ============================================================

class TestGraphStructure(unittest.TestCase):
    """图结构测试"""
    
    def test_graph_nodes_count(self):
        """测试节点数量"""
        from langgraph.graph import StateGraph, END, START
        
        class TestState(TypedDict):
            value: str
        
        def node_func_1(state: TestState) -> dict:
            return {"value": "1"}
        
        def node_func_2(state: TestState) -> dict:
            return {"value": "2"}
        
        def node_func_3(state: TestState) -> dict:
            return {"value": "3"}
        
        builder = StateGraph(TestState)
        builder.add_node("node1", node_func_1)
        builder.add_node("node2", node_func_2)
        builder.add_node("node3", node_func_3)
        
        builder.add_edge(START, "node1")
        builder.add_edge("node1", "node2")
        builder.add_edge("node2", "node3")
        builder.add_edge("node3", END)
        
        graph = builder.compile()
        # 验证图可以正常编译
        self.assertIsNotNone(graph)
    
    def test_graph_with_multiple_edges(self):
        """测试多分支图"""
        from langgraph.graph import StateGraph, END
        from typing import Literal
        
        class MultiState(TypedDict):
            input: str
            branch: str
            result_a: str
            result_b: str
        
        def split(state: MultiState) -> dict:
            return {"branch": state["input"]}
        
        def branch_a(state: MultiState) -> dict:
            return {"result_a": f"A处理：{state['branch']}"}
        
        def branch_b(state: MultiState) -> dict:
            return {"result_b": f"B处理：{state['branch']}"}
        
        def route(state: MultiState) -> Literal["a", "b"]:
            return "a" if state["branch"].startswith("A") else "b"
        
        builder = StateGraph(MultiState)
        builder.add_node("split", split)
        builder.add_node("a", branch_a)
        builder.add_node("b", branch_b)
        
        builder.set_entry_point("split")
        builder.add_conditional_edges("split", route)
        builder.add_edge("a", END)
        builder.add_edge("b", END)
        
        graph = builder.compile()
        
        # 测试A分支
        result = graph.invoke({"input": "A任务", "branch": "", "result_a": "", "result_b": ""})
        self.assertIn("A处理", result["result_a"])
        
        # 测试B分支
        result = graph.invoke({"input": "B任务", "branch": "", "result_a": "", "result_b": ""})
        self.assertIn("B处理", result["result_b"])


# ============================================================
# 主程序
# ============================================================

def run_tests():
    """运行所有测试"""
    print("=" * 60)
    print("Week03 测试套件")
    print("=" * 60)
    
    # 创建测试套件
    loader = unittest.TestLoader()
    suite = unittest.TestSuite()
    
    # 添加所有测试类
    test_classes = [
        TestLangChainBasics,
        TestTools,
        TestMemory,
        TestLangGraphBasics,
        TestConditionalRouting,
        TestWorkflow,
        TestHumanInLoop,
        TestGraphStructure,
    ]
    
    for test_class in test_classes:
        tests = loader.loadTestsFromTestCase(test_class)
        suite.addTests(tests)
    
    # 运行测试
    runner = unittest.TextTestRunner(verbosity=2)
    result = runner.run(suite)
    
    # 输出总结
    print("\n" + "=" * 60)
    print(f"测试完成！")
    print(f"总计：{result.testsRun} 个测试")
    print(f"通过：{result.testsRun - len(result.failures) - len(result.errors)}")
    print(f"失败：{len(result.failures)}")
    print(f"错误：{len(result.errors)}")
    print("=" * 60)
    
    return result.wasSuccessful()


if __name__ == "__main__":
    success = run_tests()
    sys.exit(0 if success else 1)
