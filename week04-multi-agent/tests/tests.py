"""
Week04 多Agent协作与框架对比 - 测试用例

测试覆盖：
1. CrewAI核心概念测试
2. LangGraph多Agent协作测试
3. 框架对比分析测试
4. 设计模式验证测试
"""

import os
import sys
import unittest
from typing import TypedDict, List

# 添加项目路径
sys.path.insert(0, os.path.join(os.path.dirname(__file__), '..'))


class TestCrewAIBasics(unittest.TestCase):
    """测试CrewAI核心概念"""
    
    def test_agent_definition(self):
        """测试Agent定义"""
        try:
            from crewai import Agent
            
            agent = Agent(
                role="测试研究员",
                goal="测试Agent定义",
                backstory="这是一个测试Agent",
            )
            
            self.assertEqual(agent.role, "测试研究员")
            self.assertEqual(agent.goal, "测试Agent定义")
        except ImportError:
            self.skipTest("crewai未安装")
    
    def test_task_definition(self):
        """测试Task定义"""
        try:
            from crewai import Agent, Task
            
            agent = Agent(
                role="测试研究员",
                goal="测试",
                backstory="测试",
            )
            
            task = Task(
                description="测试任务",
                expected_output="测试结果",
                agent=agent,
            )
            
            self.assertEqual(task.description, "测试任务")
            self.assertEqual(task.expected_output, "测试结果")
        except ImportError:
            self.skipTest("crewai未安装")
    
    def test_crew_structure(self):
        """测试Crew结构"""
        try:
            from crewai import Agent, Task, Crew, Process
            
            agent1 = Agent(role="研究员", goal="研究", backstory="研究")
            agent2 = Agent(role="写手", goal="写作", backstory="写作")
            
            task1 = Task(description="研究任务", expected_output="研究报告", agent=agent1)
            task2 = Task(description="写作任务", expected_output="文章", agent=agent2, context=[task1])
            
            crew = Crew(
                agents=[agent1, agent2],
                tasks=[task1, task2],
                process=Process.sequential,
            )
            
            self.assertEqual(len(crew.agents), 2)
            self.assertEqual(len(crew.tasks), 2)
        except ImportError:
            self.skipTest("crewai未安装")


class TestLangGraphMultiAgent(unittest.TestCase):
    """测试LangGraph多Agent协作"""
    
    def test_state_definition(self):
        """测试State定义"""
        import sys
        sys.path.insert(0, os.path.join(os.path.dirname(__file__), '..'))
        from day04_05_multi_agent_comparison.multi_agent_comparison import create_llm
        
        class TestState(TypedDict):
            topic: str
            result: str
            steps: List[str]
        
        state: TestState = {
            "topic": "测试主题",
            "result": "",
            "steps": []
        }
        
        self.assertEqual(state["topic"], "测试主题")
        self.assertIsInstance(state["steps"], list)
    
    def test_graph_creation(self):
        """测试图创建"""
        from langgraph.graph import StateGraph, END, START
        from typing import TypedDict, List

        class SimpleState(TypedDict):
            input: str
            output: str
            steps: List[str]

        def node_a(state: SimpleState) -> dict:
            return {
                "output": f"processed_{state['input']}",
                "steps": state.get("steps", []) + ["node_a"]
            }

        builder = StateGraph(SimpleState)
        builder.add_node("process", node_a)
        builder.add_edge(START, "process")
        builder.add_edge("process", END)
        
        graph = builder.compile()
        
        self.assertIsNotNone(graph)
    
    def test_graph_execution(self):
        """测试图执行"""
        from langgraph.graph import StateGraph, END, START
        from typing import TypedDict, List

        class TestState(TypedDict):
            input: str
            output: str
            steps: List[str]

        def step_one(state: TestState) -> dict:
            return {
                "output": f"step1_{state['input']}",
                "steps": state.get("steps", []) + ["step_one"]
            }

        def step_two(state: TestState) -> dict:
            return {
                "output": f"{state['output']}_step2",
                "steps": state.get("steps", []) + ["step_two"]
            }

        builder = StateGraph(TestState)
        builder.add_node("step1", step_one)
        builder.add_node("step2", step_two)
        builder.add_edge(START, "step1")
        builder.add_edge("step1", "step2")
        builder.add_edge("step2", END)
        
        graph = builder.compile()
        
        result = graph.invoke({
            "input": "test",
            "output": "",
            "steps": []
        })
        
        self.assertEqual(result["output"], "step1_test_step2")
        self.assertEqual(result["steps"], ["step_one", "step_two"])
    
    def test_conditional_routing(self):
        """测试条件路由"""
        from langgraph.graph import StateGraph, END, START
        from typing import TypedDict, Literal

        class RouterState(TypedDict):
            input: str
            route: str
            result: str

        def classify(state: RouterState) -> dict:
            if "数学" in state["input"]:
                route = "math"
            else:
                route = "other"
            return {"route": route}

        def handle_math(state: RouterState) -> dict:
            return {"result": "数学处理结果"}

        def handle_other(state: RouterState) -> dict:
            return {"result": "其他处理结果"}

        def route(state: RouterState) -> Literal["math", "other"]:
            route_val: str = state["route"]
            if route_val == "math":
                return "math"
            return "other"

        builder = StateGraph(RouterState)
        builder.add_node("classify", classify)
        builder.add_node("math", handle_math)
        builder.add_node("other", handle_other)

        builder.add_edge(START, "classify")
        builder.add_conditional_edges("classify", route, {"math": "math", "other": "other"})
        builder.add_edge("math", END)
        builder.add_edge("other", END)
        
        graph = builder.compile()
        
        result_math = graph.invoke({"input": "数学问题", "route": "", "result": ""})
        result_other = graph.invoke({"input": "其他问题", "route": "", "result": ""})
        
        self.assertEqual(result_math["result"], "数学处理结果")
        self.assertEqual(result_other["result"], "其他处理结果")
    
    def test_loop_mechanism(self):
        """测试循环机制"""
        from langgraph.graph import StateGraph, END, START
        from typing import TypedDict, Literal

        class LoopState(TypedDict):
            count: int
            max_count: int
            done: bool

        def increment(state: LoopState) -> dict:
            new_count = state["count"] + 1
            return {
                "count": new_count,
                "done": new_count >= state["max_count"]
            }

        def check_done(state: LoopState) -> Literal["continue", "end"]:
            return "end" if state["done"] else "continue"

        builder = StateGraph(LoopState)
        builder.add_node("increment", increment)
        builder.add_edge(START, "increment")
        builder.add_conditional_edges("increment", check_done, {"continue": "increment", "end": END})
        
        graph = builder.compile()
        
        result = graph.invoke({"count": 0, "max_count": 3, "done": False})
        
        self.assertEqual(result["count"], 3)
        self.assertTrue(result["done"])


class TestFrameworkComparison(unittest.TestCase):
    """测试框架对比分析"""
    
    def test_llm_creation(self):
        """测试LLM创建"""
        import sys
        sys.path.insert(0, os.path.join(os.path.dirname(__file__), '..'))
        from day01_03_crewai.crewai_demo import create_llm as crewai_create_llm
        from day04_05_multi_agent_comparison.multi_agent_comparison import create_llm as langgraph_create_llm
        
        llm1 = crewai_create_llm()
        llm2 = langgraph_create_llm()
        
        self.assertIsNotNone(llm1)
        self.assertIsNotNone(llm2)
    
    def test_design_patterns(self):
        """测试设计模式文档完整性"""
        import sys
        sys.path.insert(0, os.path.join(os.path.dirname(__file__), '..'))
        from day01_03_crewai.crewai_demo import summarize_design_patterns
        from day04_05_multi_agent_comparison.multi_agent_comparison import summarize_best_practices
        
        # 这些函数应该能正常执行（打印输出）
        try:
            summarize_design_patterns()
            summarize_best_practices()
        except Exception as e:
            self.fail(f"设计模式总结执行失败: {e}")


class TestMultiAgentWorkflow(unittest.TestCase):
    """测试多Agent工作流"""
    
    def test_workflow_structure(self):
        """测试工作流结构"""
        from langgraph.graph import StateGraph, END, START
        from typing import TypedDict, List

        class WorkflowState(TypedDict):
            topic: str
            collected_info: str
            analysis_result: str
            final_report: str
            steps: List[str]

        def collect(state: WorkflowState) -> dict:
            return {
                "collected_info": f"关于{state['topic']}的信息",
                "steps": state.get("steps", []) + ["collect"]
            }

        def analyze(state: WorkflowState) -> dict:
            return {
                "analysis_result": f"分析结果",
                "steps": state.get("steps", []) + ["analyze"]
            }

        def write(state: WorkflowState) -> dict:
            return {
                "final_report": f"最终报告",
                "steps": state.get("steps", []) + ["write"]
            }

        builder = StateGraph(WorkflowState)
        builder.add_node("collect", collect)
        builder.add_node("analyze", analyze)
        builder.add_node("write", write)

        builder.add_edge(START, "collect")
        builder.add_edge("collect", "analyze")
        builder.add_edge("analyze", "write")
        builder.add_edge("write", END)
        
        graph = builder.compile()
        
        nodes = list(graph.get_graph().nodes.keys())
        
        self.assertIn("collect", nodes)
        self.assertIn("analyze", nodes)
        self.assertIn("write", nodes)
    
    def test_task_dependency(self):
        """测试任务依赖"""
        try:
            from crewai import Agent, Task
            
            agent = Agent(role="测试", goal="测试", backstory="测试")
            
            task1 = Task(description="任务1", expected_output="输出1", agent=agent)
            task2 = Task(description="任务2", expected_output="输出2", agent=agent, context=[task1])
            task3 = Task(description="任务3", expected_output="输出3", agent=agent, context=[task2])
            
            self.assertEqual(len(task2.context), 1)  # type: ignore[arg-type]
            self.assertEqual(len(task3.context), 1)  # type: ignore[arg-type]
            self.assertIn(task1, task2.context)  # type: ignore[arg-type]
            self.assertIn(task2, task3.context)  # type: ignore[arg-type]
        except ImportError:
            self.skipTest("crewai未安装")


if __name__ == "__main__":
    unittest.main()
