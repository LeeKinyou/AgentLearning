"""
Week01 测试用例

本模块包含week01所有核心功能的测试用例。
测试范围：
1. LLM基础客户端
2. Pydantic模型验证
3. Prompt模板库
4. 工具和Agent核心组件

运行方式：
    python tests.py
"""

import sys
import os

# 添加项目根目录到Python路径
sys.path.insert(0, os.path.join(os.path.dirname(__file__), '..'))

from pydantic import ValidationError
from day03_04_async_prompt.prompt_templates import PromptTemplate, PromptLibrary, PromptBuilder
from day05_07_react_agent.react_agent import (
    Tool, ToolParameter, ToolRegistry,
    ReActAgent, AgentStep,
    create_search_tool, create_calculator_tool, create_date_tool
)


# ============================================================
# 测试1：Prompt模板测试
# ============================================================

def test_prompt_template():
    """测试PromptTemplate基础功能"""
    print("=" * 60)
    print("测试1：PromptTemplate")
    print("=" * 60)
    
    # 创建模板
    template = PromptTemplate(
        name="test_greeting",
        template="你好，{name}！今天是{day}。",
        description="测试用问候模板"
    )
    
    # 测试变量提取
    assert 'name' in template.variables, "应提取变量name"
    assert 'day' in template.variables, "应提取变量day"
    print(f"[PASS] 变量提取成功：{template.variables}")
    
    # 测试模板格式化
    result = template.format(name="小明", day="星期一")
    assert result == "你好，小明！今天是星期一。", f"格式化结果不正确：{result}"
    print(f"[PASS] 模板格式化成功：{result}")
    
    # 测试变量验证
    assert template.validate(name="小明", day="星期一") == True
    assert template.validate(name="小明") == False
    print("[PASS] 变量验证成功")
    
    print("[PASS] PromptTemplate测试通过\n")


def test_prompt_library():
    """测试PromptLibrary功能"""
    print("=" * 60)
    print("测试2：PromptLibrary")
    print("=" * 60)
    
    library = PromptLibrary()
    
    # 测试模板列表
    templates = library.list_templates()
    assert len(templates) > 0, "模板库应包含预定义模板"
    print(f"[PASS] 模板库包含 {len(templates)} 个预定义模板")
    
    # 测试获取模板
    template = library.get("system_assistant")
    assert template.name == "system_assistant"
    print(f"[PASS] 成功获取模板：{template.name}")
    
    # 测试模板不存在异常
    try:
        library.get("nonexistent_template")
        assert False, "应抛出KeyError"
    except KeyError:
        print("[PASS] 不存在的模板正确抛出KeyError")
    
    print("[PASS] PromptLibrary测试通过\n")


def test_prompt_builder():
    """测试PromptBuilder功能"""
    print("=" * 60)
    print("测试3：PromptBuilder")
    print("=" * 60)
    
    builder = PromptBuilder()
    prompt = (builder
        .system("你是一个助手")
        .context("这是背景信息")
        .few_shot([{"input": "Hi", "output": "你好"}])
        .constraints(["保持简洁"])
        .user("你好")
        .build()
    )
    
    assert "【系统提示】" in prompt
    assert "【上下文】" in prompt
    assert "【示例】" in prompt
    assert "【约束条件】" in prompt
    assert "【用户输入】" in prompt
    print(f"[PASS] PromptBuilder构建成功")
    print(f"生成的Prompt长度：{len(prompt)} 字符")
    
    print("[PASS] PromptBuilder测试通过\n")


# ============================================================
# 测试2：工具系统测试
# ============================================================

def test_tool_parameter():
    """测试ToolParameter功能"""
    print("=" * 60)
    print("测试4：ToolParameter")
    print("=" * 60)
    
    param = ToolParameter(
        name="query",
        param_type="str",
        description="搜索关键词",
        required=True
    )
    
    param_dict = param.to_dict()
    assert param_dict["name"] == "query"
    assert param_dict["type"] == "str"
    assert param_dict["required"] == True
    print(f"[PASS] ToolParameter转换成功：{param_dict}")
    
    print("[PASS] ToolParameter测试通过\n")


def test_tool():
    """测试Tool功能"""
    print("=" * 60)
    print("测试5：Tool")
    print("=" * 60)
    
    # 创建测试工具
    def add_numbers(a: int, b: int) -> int:
        return a + b
    
    tool = Tool(
        name="adder",
        description="两个数相加",
        parameters=[
            ToolParameter("a", "int", "第一个数"),
            ToolParameter("b", "int", "第二个数")
        ],
        func=add_numbers
    )
    
    # 测试执行
    result = tool.execute(a=3, b=5)
    assert result == "8", f"计算结果不正确：{result}"
    print(f"[PASS] 工具执行成功：{result}")
    
    # 测试LLM描述生成
    desc = tool.get_description_for_llm()
    assert "adder" in desc
    assert "第一个数" in desc
    print(f"[PASS] LLM描述生成成功")
    
    print("[PASS] Tool测试通过\n")


def test_tool_registry():
    """测试ToolRegistry功能"""
    print("=" * 60)
    print("测试6：ToolRegistry")
    print("=" * 60)
    
    registry = ToolRegistry()
    
    # 注册工具
    search_tool = create_search_tool()
    calc_tool = create_calculator_tool()
    
    registry.register(search_tool)
    registry.register(calc_tool)
    
    # 测试工具列表
    tools = registry.list_tools()
    assert "search" in tools
    assert "calculator" in tools
    print(f"[PASS] 工具注册成功：{tools}")
    
    # 测试获取工具
    tool = registry.get("search")
    assert tool is not None
    assert tool.name == "search"
    print(f"[PASS] 工具获取成功：{tool.name}")
    
    # 测试工具描述
    desc = registry.get_tools_description_for_llm()
    assert len(desc) > 0
    print(f"[PASS] 工具描述生成成功，长度：{len(desc)} 字符")
    
    # 测试工具执行
    result = registry.execute_tool("calculator", expression="2 + 3")
    assert "5" in result
    print(f"[PASS] 工具执行成功：{result}")
    
    # 测试不存在的工具
    try:
        registry.execute_tool("nonexistent_tool")
        assert False, "应抛出ValueError"
    except ValueError:
        print("[PASS] 不存在的工具正确抛出ValueError")
    
    print("[PASS] ToolRegistry测试通过\n")


# ============================================================
# 测试3：预定义工具测试
# ============================================================

def test_search_tool():
    """测试搜索工具"""
    print("=" * 60)
    print("测试7：Search Tool")
    print("=" * 60)
    
    tool = create_search_tool()
    
    # 测试已知关键词
    result = tool.execute(query="python")
    assert "Python" in result
    print(f"[PASS] 搜索'python'成功")
    
    # 测试未知关键词
    result = tool.execute(query="未知关键词")
    assert "模拟搜索结果" in result
    print(f"[PASS] 未知关键词处理成功")
    
    print("[PASS] Search Tool测试通过\n")


def test_calculator_tool():
    """测试计算器工具"""
    print("=" * 60)
    print("测试8：Calculator Tool")
    print("=" * 60)
    
    tool = create_calculator_tool()
    
    # 测试基本运算
    result = tool.execute(expression="2 + 3 * 4")
    assert "14" in result, f"计算结果不正确：{result}"
    print(f"[PASS] 基本运算成功：{result}")
    
    # 测试数学函数
    result = tool.execute(expression="sqrt(16)")
    assert "4" in result, f"数学函数结果不正确：{result}"
    print(f"[PASS] 数学函数成功：{result}")
    
    # 测试错误处理
    result = tool.execute(expression="invalid expression")
    assert "错误" in result
    print(f"[PASS] 错误处理成功：{result}")
    
    print("[PASS] Calculator Tool测试通过\n")


def test_date_tool():
    """测试日期工具"""
    print("=" * 60)
    print("测试9：Date Tool")
    print("=" * 60)
    
    tool = create_date_tool()
    
    result = tool.execute()
    assert "当前日期时间" in result
    print(f"[PASS] 日期获取成功：{result}")
    
    print("[PASS] Date Tool测试通过\n")


# ============================================================
# 测试4：Agent核心组件测试
# ============================================================

def test_agent_step():
    """测试AgentStep记录"""
    print("=" * 60)
    print("测试10：AgentStep")
    print("=" * 60)
    
    step = AgentStep(
        step_number=1,
        thought="我需要搜索信息",
        action="search",
        action_input="Python",
        observation="Python是一种编程语言"
    )
    
    step_str = str(step)
    assert "步骤 1" in step_str
    assert "search" in step_str
    print(f"[PASS] AgentStep格式化成功")
    
    print("[PASS] AgentStep测试通过\n")


def test_react_agent_initialization():
    """测试ReActAgent初始化"""
    print("=" * 60)
    print("测试11：ReActAgent初始化")
    print("=" * 60)
    
    # 注意：此测试需要OPENAI_API_KEY
    try:
        agent = ReActAgent(
            model="gpt-3.5-turbo",
            max_iterations=5,
            verbose=False
        )
        
        # 添加工具
        agent.add_tool(create_search_tool())
        agent.add_tool(create_calculator_tool())
        
        # 验证工具注册
        tools = agent.tool_registry.list_tools()
        assert "search" in tools
        assert "calculator" in tools
        print(f"[PASS] Agent初始化成功，已注册工具：{tools}")
        
        # 验证系统提示词生成
        system_prompt = agent._build_system_prompt()
        assert "Thought" in system_prompt
        assert "Action" in system_prompt
        assert "Final Answer" in system_prompt
        print(f"[PASS] 系统提示词生成成功，长度：{len(system_prompt)} 字符")
        
        print("[PASS] ReActAgent初始化测试通过\n")
        
    except ValueError as e:
        print(f"[SKIP] 跳过测试（需要API密钥）：{e}\n")


def test_response_parsing():
    """测试LLM响应解析"""
    print("=" * 60)
    print("测试12：响应解析")
    print("=" * 60)
    
    # 创建临时Agent用于测试解析功能
    try:
        agent = ReActAgent(verbose=False)
        
        # 测试Final Answer解析
        response1 = """Thought: 我有足够信息了
Final Answer: Python是一种编程语言"""
        parsed1 = agent._parse_llm_response(response1)
        assert parsed1["final_answer"] == "Python是一种编程语言"
        print(f"[PASS] Final Answer解析成功：{parsed1['final_answer']}")
        
        # 测试Action解析
        response2 = """Thought: 我需要搜索
Action: search
Action Input: Python"""
        parsed2 = agent._parse_llm_response(response2)
        assert parsed2["action"] == "search"
        assert parsed2["action_input"] == "Python"
        print(f"[PASS] Action解析成功：{parsed2['action']}({parsed2['action_input']})")
        
        print("[PASS] 响应解析测试通过\n")
        
    except ValueError as e:
        print(f"[SKIP] 跳过测试（需要API密钥）：{e}\n")


# ============================================================
# 测试5：Pydantic模型测试
# ============================================================

def test_pydantic_model():
    """测试Pydantic模型验证"""
    print("=" * 60)
    print("测试13：Pydantic模型验证")
    print("=" * 60)
    
    # 动态导入以避免循环依赖
    from day03_04_async_prompt.async_pydantic_demo import MovieReview, TaskExtraction
    
    # 测试有效实例
    review = MovieReview(
        title="测试电影",
        rating=8.5,
        genre="剧情",
        summary="这是一部好电影",
        pros=["演员出色", "剧情紧凑"],
        cons=["结局仓促"]
    )
    assert review.title == "测试电影"
    assert review.rating == 8.5
    print(f"[PASS] 有效实例创建成功：{review.title}")
    
    # 测试验证错误
    try:
        invalid_review = MovieReview(
            title="测试",
            rating=15.0,  # 超出范围
            genre="剧情",
            summary="测试"
        )
        assert False, "应抛出ValidationError"
    except ValidationError:
        print("[PASS] 评分范围验证成功（15.0被拒绝）")
    
    # 测试模型序列化
    review_dict = review.to_dict()
    assert "title" in review_dict
    assert "rating" in review_dict
    print(f"[PASS] 模型序列化成功")
    
    # 测试TaskExtraction
    task = TaskExtraction(
        task_title="紧急任务",
        description="这是一个紧急任务",
        priority="high",
        due_date="2026-12-31",
        subtasks=["子任务1", "子任务2"]
    )
    assert task.priority == "high"
    print(f"[PASS] TaskExtraction创建成功：{task.task_title}")
    
    print("[PASS] Pydantic模型测试通过\n")


# ============================================================
# 主测试运行器
# ============================================================

def run_all_tests():
    """运行所有测试"""
    print("\n" + "=" * 60)
    print("Week01 测试套件")
    print("=" * 60 + "\n")
    
    tests = [
        ("Prompt模板测试", test_prompt_template),
        ("Prompt库测试", test_prompt_library),
        ("Prompt构建器测试", test_prompt_builder),
        ("ToolParameter测试", test_tool_parameter),
        ("Tool测试", test_tool),
        ("ToolRegistry测试", test_tool_registry),
        ("搜索工具测试", test_search_tool),
        ("计算器工具测试", test_calculator_tool),
        ("日期工具测试", test_date_tool),
        ("AgentStep测试", test_agent_step),
        ("ReActAgent初始化测试", test_react_agent_initialization),
        ("响应解析测试", test_response_parsing),
        ("Pydantic模型测试", test_pydantic_model),
    ]
    
    passed = 0
    failed = 0
    skipped = 0
    
    for test_name, test_func in tests:
        try:
            test_func()
            passed += 1
        except AssertionError as e:
            print(f"[FAIL] {test_name} 失败：{e}\n")
            failed += 1
        except Exception as e:
            print(f"[ERROR] {test_name} 错误：{e}\n")
            failed += 1
    
    # 打印总结
    print("=" * 60)
    print("测试总结")
    print("=" * 60)
    print(f"总测试数：{len(tests)}")
    print(f"通过：{passed}")
    print(f"失败：{failed}")
    print(f"通过率：{passed/len(tests)*100:.1f}%")
    print("=" * 60)
    
    return failed == 0


if __name__ == "__main__":
    success = run_all_tests()
    sys.exit(0 if success else 1)
