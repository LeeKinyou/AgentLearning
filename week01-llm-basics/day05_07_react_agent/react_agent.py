"""
Day 5-7: 手写 ReAct Agent 完整实现

本模块从零实现一个完整的ReAct Agent，不依赖任何Agent框架。

ReAct 范式核心概念：
    ReAct = Reason + Act
    - Reason (推理): LLM思考当前状态，决定下一步行动
    - Act (行动): 执行工具调用获取外部信息
    - Observe (观察): 获取工具执行结果
    - 循环上述过程，直到得出最终答案

核心组件：
    1. LLM调用层：与LLM交互，获取推理和行动指令
    2. Tool定义与注册：定义可用工具及其调用方式
    3. Thought-Action-Observation循环：Agent的核心执行逻辑
    4. 终止条件判断：确定何时停止循环并输出最终答案

学习目标：
    - 理解ReAct Agent的底层工作原理
    - 掌握工具定义、注册和调用机制
    - 学会设计Agent的循环逻辑和状态管理
    - 能够从零构建一个可用的Agent系统

使用前提：
    - 已安装 openai 库：pip install openai
    - 已安装 python-dotenv：pip install python-dotenv
"""

import os
import re
import json
import math
import asyncio
from typing import Optional, List, Dict, Any, Callable
from dataclasses import dataclass, field
from enum import Enum
from datetime import datetime
from dotenv import load_dotenv

# 加载环境变量
load_dotenv()


# ============================================================
# 第一部分：工具定义与注册机制
# ============================================================

class ToolParameter:
    """
    工具参数定义类
    
    功能说明：
        定义工具的参数规范，包括名称、类型、描述和是否必需。
        用于生成工具描述和验证参数。
    
    属性说明：
        - name: 参数名称
        - param_type: 参数类型（str, int, float, bool）
        - description: 参数描述
        - required: 是否必需参数
    """
    
    def __init__(
        self,
        name: str,
        param_type: str = "str",
        description: str = "",
        required: bool = True
    ):
        """
        初始化参数定义
        
        参数：
            name: 参数名称
            param_type: 参数类型
            description: 参数描述
            required: 是否必需
        """
        self.name = name
        self.param_type = param_type
        self.description = description
        self.required = required
    
    def to_dict(self) -> dict:
        """
        转换为字典格式
        
        返回：
            dict: 参数的字典表示
        """
        return {
            "name": self.name,
            "type": self.param_type,
            "description": self.description,
            "required": self.required
        }


class Tool:
    """
    工具定义类
    
    功能说明：
        封装一个可被Agent调用的工具。
        包含工具的名称、描述、参数和执行函数。
    
    核心属性：
        - name: 工具名称（英文，用于LLM识别）
        - description: 工具描述（告诉LLM何时使用此工具）
        - parameters: 参数定义列表
        - func: 实际执行的函数
    
    使用示例：
        >>> def search(query: str) -> str:
        ...     return f"搜索结果：{query}"
        >>> tool = Tool(
        ...     name="search",
        ...     description="搜索网络信息",
        ...     parameters=[ToolParameter("query", "str", "搜索关键词")],
        ...     func=search
        ... )
    """
    
    def __init__(
        self,
        name: str,
        description: str,
        parameters: List[ToolParameter],
        func: Callable
    ):
        """
        初始化工具
        
        参数：
            name: 工具名称
            description: 工具描述
            parameters: 参数定义列表
            func: 执行函数，接受关键字参数，返回字符串
        """
        self.name = name
        self.description = description
        self.parameters = parameters
        self.func = func
    
    def execute(self, **kwargs) -> str:
        """
        执行工具
        
        参数：
            **kwargs: 工具参数（键值对）
        
        返回：
            str: 工具执行结果
        
        异常：
            ValueError: 缺少必需参数时抛出
        """
        # 验证必需参数
        for param in self.parameters:
            if param.required and param.name not in kwargs:
                raise ValueError(f"工具 '{self.name}' 缺少必需参数：{param.name}")
        
        # 执行函数
        try:
            result = self.func(**kwargs)
            return str(result)
        except Exception as e:
            return f"工具执行失败：{str(e)}"
    
    def get_description_for_llm(self) -> str:
        """
        生成给LLM看的工具描述
        
        返回：
            str: 格式化的工具描述，包含名称、描述和参数说明
        """
        desc = f"工具名称：{self.name}\n"
        desc += f"描述：{self.description}\n"
        
        if self.parameters:
            desc += "参数：\n"
            for param in self.parameters:
                required_mark = "(必需)" if param.required else "(可选)"
                desc += f"  - {param.name} ({param.param_type}): {param.description} {required_mark}\n"
        
        return desc


class ToolRegistry:
    """
    工具注册表
    
    功能说明：
        管理所有可用工具的注册、查询和描述生成。
        Agent通过注册表获取可用工具列表并执行工具调用。
    
    使用示例：
        >>> registry = ToolRegistry()
        >>> registry.register(search_tool)
        >>> registry.register(calculator_tool)
        >>> tool = registry.get("search")
        >>> result = tool.execute(query="Python")
    """
    
    def __init__(self):
        """初始化工具注册表"""
        self._tools: Dict[str, Tool] = {}
    
    def register(self, tool: Tool):
        """
        注册工具
        
        参数：
            tool: Tool实例
        """
        self._tools[tool.name] = tool
    
    def get(self, name: str) -> Optional[Tool]:
        """
        获取指定名称的工具
        
        参数：
            name: 工具名称
        
        返回：
            Tool: 工具实例，如果不存在则返回None
        """
        return self._tools.get(name)
    
    def list_tools(self) -> List[str]:
        """
        列出所有已注册的工具名称
        
        返回：
            List[str]: 工具名称列表
        """
        return list(self._tools.keys())
    
    def get_tools_description_for_llm(self) -> str:
        """
        生成所有工具的LLM可读描述
        
        返回：
            str: 所有工具的格式化描述
        """
        descriptions = []
        for tool in self._tools.values():
            descriptions.append(tool.get_description_for_llm())
        
        return "\n\n".join(descriptions)
    
    def execute_tool(self, tool_name: str, **kwargs) -> str:
        """
        执行指定工具
        
        参数：
            tool_name: 工具名称
            **kwargs: 工具参数
        
        返回：
            str: 工具执行结果
        
        异常：
            ValueError: 工具不存在时抛出
        """
        tool = self.get(tool_name)
        if not tool:
            raise ValueError(f"工具 '{tool_name}' 未注册")
        
        return tool.execute(**kwargs)


# ============================================================
# 第二部分：预定义工具实现
# ============================================================

def create_search_tool():
    """
    创建搜索工具
    
    功能说明：
        模拟网络搜索功能。在实际应用中，可以替换为真实的搜索引擎API
        （如Google Search API、SerpAPI等）。
    
    返回：
        Tool: 搜索工具实例
    """
    def search(query: str) -> str:
        """
        模拟搜索函数
        
        参数：
            query: 搜索关键词
        
        返回：
            str: 模拟的搜索结果
        """
        # 模拟搜索结果数据库
        mock_results = {
            "python": "Python是一种广泛使用的高级编程语言，由Guido van Rossum于1991年发布。Python支持多种编程范式，包括面向对象、命令式、函数式和过程式编程。",
            "人工智能": "人工智能（Artificial Intelligence, AI）是计算机科学的一个分支，致力于创建能够执行通常需要人类智能才能完成的任务的系统，如学习、推理、感知和理解语言。",
            "机器学习": "机器学习是人工智能的一个子领域，专注于开发能够从数据中学习和改进的算法和模型，而无需显式编程。",
            "2024年奥运会": "2024年夏季奥运会在法国巴黎举行，时间为2024年7月26日至8月11日。中国代表团表现出色，在多个项目上获得奖牌。",
            "天气": "今日天气晴朗，气温20-28°C，适宜户外活动。",
            "北京": "北京是中国的首都，政治、文化、教育中心。著名景点包括故宫、长城、天坛等。"
        }
        
        # 简单的关键词匹配
        query_lower = query.lower()
        for key, result in mock_results.items():
            if key in query_lower:
                return result
        
        # 默认返回
        return f"关于'{query}'的搜索结果：这是一个模拟搜索结果。在实际应用中，这里会返回真实的搜索引擎结果。"
    
    return Tool(
        name="search",
        description="搜索网络信息，获取关于特定主题的知识",
        parameters=[ToolParameter("query", "str", "搜索关键词或问题")],
        func=search
    )


def create_calculator_tool():
    """
    创建计算器工具
    
    功能说明：
        执行数学计算，支持基本运算和常见数学函数。
    
    返回：
        Tool: 计算器工具实例
    """
    def calculate(expression: str) -> str:
        """
        计算数学表达式
        
        参数：
            expression: 数学表达式字符串，如 "2 + 3 * 4"
        
        返回：
            str: 计算结果
        """
        try:
            # 支持常见数学函数（先替换，再清理）
            safe_expr = expression.replace('sqrt', 'math.sqrt')
            safe_expr = safe_expr.replace('pow', 'math.pow')
            safe_expr = safe_expr.replace('sin', 'math.sin')
            safe_expr = safe_expr.replace('cos', 'math.cos')
            safe_expr = safe_expr.replace('tan', 'math.tan')
            safe_expr = safe_expr.replace('pi', 'math.pi')
            safe_expr = safe_expr.replace('e', 'math.e')
            
            # 安全的表达式求值（仅允许数学运算字符）
            safe_expr = re.sub(r'[^0-9+\-*/().\s\w]', '', safe_expr)
            
            result = eval(safe_expr, {"__builtins__": {}, "math": math})
            return f"计算结果：{expression} = {result}"
        except Exception as e:
            return f"计算错误：{str(e)}"
    
    return Tool(
        name="calculator",
        description="执行数学计算，支持加减乘除和常见数学函数",
        parameters=[ToolParameter("expression", "str", "数学表达式，如 '2 + 3 * 4' 或 'sqrt(16)'")],
        func=calculate
    )


def create_date_tool():
    """
    创建日期工具
    
    功能说明：
        获取当前日期时间信息。
    
    返回：
        Tool: 日期工具实例
    """
    def get_date(format: str = "%Y-%m-%d") -> str:
        """
        获取当前日期
        
        参数：
            format: 日期格式字符串
        
        返回：
            str: 格式化的当前日期
        """
        now = datetime.now()
        return f"当前日期时间：{now.strftime(format)}"
    
    return Tool(
        name="get_date",
        description="获取当前的日期和时间信息",
        parameters=[ToolParameter("format", "str", "日期格式，默认为 %Y-%m-%d", required=False)],
        func=get_date
    )


def create_text_tool():
    """
    创建文本处理工具
    
    功能说明：
        提供基本文本处理功能，如字数统计、翻译模拟等。
    
    返回：
        Tool: 文本处理工具实例
    """
    def process_text(text: str, operation: str = "count") -> str:
        """
        处理文本
        
        参数：
            text: 要处理的文本
            operation: 操作类型（count/uppercase/lowercase）
        
        返回：
            str: 处理结果
        """
        if operation == "count":
            return f"字数统计：{len(text)} 个字符，{len(text.split())} 个单词"
        elif operation == "uppercase":
            return f"大写转换：{text.upper()}"
        elif operation == "lowercase":
            return f"小写转换：{text.lower()}"
        else:
            return f"不支持的操作：{operation}"
    
    return Tool(
        name="text_processor",
        description="文本处理工具，支持字数统计、大小写转换等操作",
        parameters=[
            ToolParameter("text", "str", "要处理的文本"),
            ToolParameter("operation", "str", "操作类型：count/uppercase/lowercase", required=False)
        ],
        func=process_text
    )


# ============================================================
# 第三部分：LLM 调用层
# ============================================================

class LLMClient:
    """
    LLM 调用客户端
    
    功能说明：
        封装与LLM的交互逻辑，提供简洁的调用接口。
        支持多种LLM服务：OpenAI、LM Studio、Ollama、其他兼容OpenAI API的服务。
        支持普通对话和ReAct格式的推理输出。
    
    支持的LLM服务配置：
        1. OpenAI官方API: API_BASE_URL=https://api.openai.com/v1
        2. LM Studio本地模型: API_BASE_URL=http://localhost:1234/v1
        3. Ollama本地模型: API_BASE_URL=http://localhost:11434/v1
        4. 其他兼容服务: API_BASE_URL=https://your-server.com/v1
    """
    
    def __init__(
        self,
        model: str = "gpt-3.5-turbo",
        temperature: float = 0.0,
        max_tokens: int = 1000,
        api_key: Optional[str] = None,
        base_url: Optional[str] = None
    ):
        """
        初始化LLM客户端
        
        参数：
            model: LLM模型名称
            temperature: 输出随机性（ReAct建议使用低温度）
            max_tokens: 最大生成Token数
            api_key: API密钥，优先使用此参数，否则从环境变量读取
            base_url: API基础URL，优先使用此参数，否则从环境变量读取
        """
        self.api_key = api_key or os.getenv("OPENAI_API_KEY")
        if not self.api_key:
            raise ValueError(
                "未找到 OPENAI_API_KEY 环境变量。\n"
                "请在 .env 文件中配置API密钥（LM Studio可填写任意值）"
            )
        
        # 从参数或环境变量获取base_url
        self.base_url = base_url or os.getenv("API_BASE_URL")
        if not self.base_url:
            self.base_url = "https://api.openai.com/v1"
        
        # 从环境变量获取模型名称（优先于参数）
        self.model = os.getenv("MODEL_NAME", model)
        self.temperature = temperature
        self.max_tokens = max_tokens
        self._client = None
    
    def _get_client(self):
        """获取OpenAI客户端实例"""
        if self._client is None:
            from openai import OpenAI
            self._client = OpenAI(
                api_key=self.api_key,
                base_url=self.base_url
            )
        return self._client
    
    def chat(self, messages: List[Dict[str, str]]) -> str:
        """
        发送消息并获取回复
        
        参数：
            messages: 消息列表，格式为 [{"role": "user/assistant/system", "content": "..."}]
        
        返回：
            str: LLM生成的回复内容
        """
        try:
            response = self._get_client().chat.completions.create(
                model=self.model,
                messages=messages,
                temperature=self.temperature,
                max_tokens=self.max_tokens
            )
            return response.choices[0].message.content.strip()
        except Exception as e:
            return f"API调用失败：{str(e)}"


# ============================================================
# 第四部分：ReAct Agent 核心实现
# ============================================================

class AgentStep:
    """
    Agent执行步骤记录
    
    功能说明：
        记录Agent每次循环的完整信息，包括思考、行动和观察结果。
        用于调试、日志记录和对话历史维护。
    """
    
    def __init__(
        self,
        step_number: int,
        thought: str,
        action: str,
        action_input: str,
        observation: str
    ):
        """
        初始化步骤记录
        
        参数：
            step_number: 步骤序号
            thought: LLM的思考过程
            action: 选择的工具名称
            action_input: 工具输入
            observation: 工具执行结果
        """
        self.step_number = step_number
        self.thought = thought
        self.action = action
        self.action_input = action_input
        self.observation = observation
    
    def __str__(self) -> str:
        """格式化的步骤字符串"""
        return (
            f"步骤 {self.step_number}:\n"
            f"  思考：{self.thought}\n"
            f"  行动：{self.action}({self.action_input})\n"
            f"  观察：{self.observation}"
        )


class ReActAgent:
    """
    ReAct Agent 核心实现
    
    功能说明：
        实现完整的ReAct循环：Thought -> Action -> Observation -> ... -> Final Answer
        
        工作流程：
        1. 接收用户问题
        2. LLM分析问题，生成Thought（思考）和Action（行动指令）
        3. 解析Action，执行对应工具
        4. 将Observation（观察结果）反馈给LLM
        5. 重复2-4，直到LLM生成Final Answer
        6. 返回最终答案
    
    核心组件：
        - LLM客户端：负责推理和决策
        - 工具注册表：提供可用工具
        - 循环控制器：管理执行流程
        - 状态管理器：维护对话历史
    
    使用示例：
        >>> agent = ReActAgent()
        >>> agent.add_tool(create_search_tool())
        >>> agent.add_tool(create_calculator_tool())
        >>> result = agent.run("2024年奥运会中国获得了多少金牌？")
        >>> print(result)
    """
    
    # 最大循环次数，防止无限循环
    MAX_ITERATIONS = 10
    
    def __init__(
        self,
        model: str = "gpt-3.5-turbo",
        max_iterations: int = 10,
        verbose: bool = True
    ):
        """
        初始化ReAct Agent
        
        参数：
            model: 使用的LLM模型
            max_iterations: 最大循环次数
            verbose: 是否打印详细执行过程
        """
        # 初始化LLM客户端
        self.llm = LLMClient(model=model, temperature=0.0)
        
        # 初始化工具注册表
        self.tool_registry = ToolRegistry()
        
        # 配置参数
        self.max_iterations = max_iterations
        self.verbose = verbose
        
        # 执行历史记录
        self.history: List[AgentStep] = []
    
    def add_tool(self, tool: Tool):
        """
        添加工具到Agent
        
        参数：
            tool: Tool实例
        """
        self.tool_registry.register(tool)
    
    def _build_system_prompt(self) -> str:
        """
        构建系统提示词
        
        功能说明：
            生成ReAct格式的系统提示词，告诉LLM：
            1. 可用工具列表
            2. ReAct执行格式
            3. 思考和行动的规则
        
        返回：
            str: 完整的系统提示词
        """
        # 获取工具描述
        tools_desc = self.tool_registry.get_tools_description_for_llm()
        
        # 构建系统提示词
        system_prompt = f"""你是一个智能助手，可以使用工具来回答问题。

## 可用工具

{tools_desc}

## 回答格式

你必须严格按照以下格式回答：

Thought: [你的思考过程，分析问题和决定下一步行动]
Action: [工具名称]
Action Input: [工具输入参数]
Observation: [工具返回的结果]
...（可以重复 Thought/Action/Observation 循环）...
Thought: 我已经有了足够的信息来回答问题
Final Answer: [最终答案]

## 规则

1. 每次只能执行一个Action
2. 必须使用Thought来解释你的决策
3. 充分利用Observation进行推理
4. 当你有足够信息时，必须输出Final Answer
5. Final Answer必须是完整、准确的回答

现在，请开始回答问题。"""
        
        return system_prompt
    
    def _parse_llm_response(self, response: str) -> Dict[str, str]:
        """
        解析LLM的响应
        
        功能说明：
            从LLM的输出中提取Thought、Action、Action Input或Final Answer。
            使用正则表达式进行解析。
        
        参数：
            response: LLM的原始输出
        
        返回：
            dict: 包含解析结果的字典
                - thought: 思考内容
                - action: 工具名称（如果有）
                - action_input: 工具输入（如果有）
                - final_answer: 最终答案（如果有）
        """
        result = {
            "thought": "",
            "action": None,
            "action_input": None,
            "final_answer": None
        }
        
        # 提取Thought
        thought_match = re.search(r'Thought:\s*(.*?)(?=Action:|Final Answer:|$)', response, re.DOTALL)
        if thought_match:
            result["thought"] = thought_match.group(1).strip()
        
        # 提取Final Answer（优先级最高）
        final_answer_match = re.search(r'Final Answer:\s*(.*)', response, re.DOTALL)
        if final_answer_match:
            result["final_answer"] = final_answer_match.group(1).strip()
            return result
        
        # 提取Action
        action_match = re.search(r'Action:\s*(.*?)(?:\n|$)', response)
        if action_match:
            result["action"] = action_match.group(1).strip()
        
        # 提取Action Input
        action_input_match = re.search(r'Action Input:\s*(.*)', response, re.DOTALL)
        if action_input_match:
            result["action_input"] = action_input_match.group(1).strip()
        
        return result
    
    def _build_conversation_history(self, question: str) -> List[Dict[str, str]]:
        """
        构建对话历史
        
        功能说明：
            将系统提示词、用户问题和执行历史组合成完整的对话历史。
            用于维持LLM的上下文理解。
        
        参数：
            question: 用户的原始问题
        
        返回：
            List[Dict]: 对话历史消息列表
        """
        messages = []
        
        # 添加系统提示词
        messages.append({
            "role": "system",
            "content": self._build_system_prompt()
        })
        
        # 添加用户问题
        messages.append({
            "role": "user",
            "content": f"问题：{question}"
        })
        
        # 添加执行历史
        if self.history:
            history_text = "\n\n".join([
                f"Thought: {step.thought}\nAction: {step.action}\nAction Input: {step.action_input}\nObservation: {step.observation}"
                for step in self.history
            ])
            messages.append({
                "role": "assistant",
                "content": history_text
            })
        
        return messages
    
    def run(self, question: str) -> str:
        """
        运行Agent，回答问题
        
        功能说明：
            执行完整的ReAct循环，直到得出最终答案或达到最大循环次数。
        
        参数：
            question: 用户的问题
        
        返回：
            str: 最终答案
        
        使用示例：
            >>> agent = ReActAgent()
            >>> agent.add_tool(create_search_tool())
            >>> answer = agent.run("Python是什么？")
        """
        # 清空历史记录
        self.history = []
        
        if self.verbose:
            print("=" * 60)
            print(f"问题：{question}")
            print("=" * 60)
        
        # ReAct循环
        for iteration in range(1, self.max_iterations + 1):
            if self.verbose:
                print(f"\n--- 第 {iteration} 轮循环 ---")
            
            # 1. 构建对话历史并调用LLM
            messages = self._build_conversation_history(question)
            response = self.llm.chat(messages)
            
            if self.verbose:
                print(f"LLM原始输出：\n{response}\n")
            
            # 2. 解析LLM响应
            parsed = self._parse_llm_response(response)
            
            # 3. 检查是否有最终答案
            if parsed["final_answer"]:
                if self.verbose:
                    print(f"\n最终答案：{parsed['final_answer']}")
                return parsed["final_answer"]
            
            # 4. 检查是否有行动指令
            if not parsed["action"] or not parsed["action_input"]:
                if self.verbose:
                    print("LLM未提供有效的Action，尝试继续...")
                # 添加一个提示继续循环
                self.history.append(AgentStep(
                    step_number=iteration,
                    thought=parsed["thought"],
                    action="None",
                    action_input="None",
                    observation="请继续执行，使用工具获取更多信息，或者给出Final Answer"
                ))
                continue
            
            # 5. 执行工具
            tool_name = parsed["action"]
            tool_input = parsed["action_input"]
            
            if self.verbose:
                print(f"思考：{parsed['thought']}")
                print(f"行动：{tool_name}({tool_input})")
            
            try:
                # 执行工具并获取观察结果
                observation = self.tool_registry.execute_tool(
                    tool_name,
                    query=tool_input if tool_name == "search" else tool_input
                )
            except Exception as e:
                observation = f"工具执行错误：{str(e)}"
            
            if self.verbose:
                print(f"观察：{observation}")
            
            # 6. 记录步骤
            self.history.append(AgentStep(
                step_number=iteration,
                thought=parsed["thought"],
                action=tool_name,
                action_input=tool_input,
                observation=observation
            ))
        
        # 达到最大循环次数仍未得出答案
        final_message = "达到最大循环次数，未能得出完整答案。"
        if self.verbose:
            print(f"\n{final_message}")
        return final_message
    
    def get_execution_log(self) -> List[str]:
        """
        获取执行日志
        
        返回：
            List[str]: 格式化的执行步骤列表
        """
        return [str(step) for step in self.history]


# ============================================================
# 第五部分：演示和使用示例
# ============================================================

def create_demo_agent() -> ReActAgent:
    """
    创建演示用的Agent实例
    
    功能说明：
        初始化一个包含常用工具的ReAct Agent，用于演示和测试。
    
    返回：
        ReActAgent: 配置好的Agent实例
    """
    # 创建Agent
    agent = ReActAgent(
        model="gpt-3.5-turbo",
        max_iterations=5,
        verbose=True
    )
    
    # 添加工具
    agent.add_tool(create_search_tool())
    agent.add_tool(create_calculator_tool())
    agent.add_tool(create_date_tool())
    agent.add_tool(create_text_tool())
    
    return agent


def demo_single_question():
    """
    演示单个问题的完整执行过程
    """
    print("\n" + "=" * 60)
    print("单问题演示")
    print("=" * 60)
    
    # 创建Agent
    agent = create_demo_agent()
    
    # 测试问题
    questions = [
        "Python是什么编程语言？",
        "计算 25 * 48 + 100 的结果",
        "今天的日期是什么？"
    ]
    
    for question in questions:
        print(f"\n{'=' * 60}")
        answer = agent.run(question)
        print(f"\n最终回答：{answer}")
        print(f"{'=' * 60}")


def demo_interactive_mode():
    """
    交互式模式
    
    功能说明：
        启动命令行交互界面，用户可以持续向Agent提问。
    """
    print("\n" + "=" * 60)
    print("ReAct Agent 交互式模式")
    print("=" * 60)
    print("输入问题开始对话（输入 'quit' 退出）\n")
    
    # 创建Agent
    agent = create_demo_agent()
    
    while True:
        question = input("你的问题：").strip()
        
        if question.lower() in ['quit', 'exit', 'q']:
            print("感谢使用，再见！")
            break
        
        if not question:
            continue
        
        answer = agent.run(question)
        print(f"\n回答：{answer}\n")


if __name__ == "__main__":
    """
    主程序入口
    
    运行方式：
        python react_agent.py
    """
    print("欢迎学习 ReAct Agent！\n")
    print("ReAct = Reason + Act")
    print("Agent通过推理(Thought)决定行动(Action)，观察结果(Observation)后继续推理\n")
    
    # 运行单问题演示
    demo_single_question()
    
    # 可选：启动交互模式
    print("\n是否进入交互模式？(y/n)")
    choice = input().strip().lower()
    if choice == 'y':
        demo_interactive_mode()
