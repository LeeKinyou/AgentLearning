import os
import json
from typing import Any
from dotenv import load_dotenv
from openai import OpenAI

load_dotenv(os.path.join(os.path.dirname(__file__), '..', '.env'))
client = OpenAI()

# === 1. 本地工具实现 ===

def get_project_tree() -> str:
    return """
├── repomind/
│   ├── __init__.py
│   ├── core.py
│   └── utils.py
├── tests/
│   └── test_core.py
├── pyproject.toml
└── README.md
    """.strip()

def search_code(query: str) -> str:
    if "QueryEngine" in query:
        return "在 repomind/core.py 中找到类: class QueryEngine:"
    return "未找到相关代码"

def read_file(file_path: str) -> str:
    if file_path == "repomind/core.py":
        return "class QueryEngine:\n    def execute(self):\n        pass"
    return "文件不存在或无法读取"

# 工具映射表
tools_map = {
    "get_project_tree": get_project_tree,
    "search_code": search_code,
    "read_file": read_file
}

# 给大模型的 Schema
tools_schema: list[Any] = [
    {
        "type": "function",
        "function": {
            "name": "get_project_tree",
            "description": "获取当前项目的整体目录树结构，用于了解文件分布。"
        }
    },
    {
        "type": "function",
        "function": {
            "name": "search_code",
            "description": "全局搜索代码关键字",
            "parameters": {
                "type": "object",
                "properties": {"query": {"type": "string", "description": "要搜索的关键字"}}
            }
        }
    },
    {
        "type": "function",
        "function": {
            "name": "read_file",
            "description": "读取指定路径的文件内容",
            "parameters": {
                "type": "object",
                "properties": {"file_path": {"type": "string", "description": "文件路径"}}
            }
        }
    }
]

# === 2. 简易 AgentRunner ===

class AgentRunner:
    def __init__(self, max_steps: int = 5):
        self.max_steps = max_steps
        self.model_name = os.getenv("MODEL_NAME", "gpt-4o-mini")
        
    def run(self, user_issue: str):
        print(f"=== Agent 启动，最大迭代步数：{self.max_steps} ===")
        print(f"【目标任务】：{user_issue}\n")
        
        messages: list[Any] = [
            {"role": "system", "content": "你是一个自动化的代码分析 Agent。请通过调用工具逐步排查报错，直到得出最终原因并给出修复建议。收集完足够证据后，请直接输出最终分析报告即可，无需继续调用工具。"},
            {"role": "user", "content": user_issue}
        ]
        
        step = 0
        while step < self.max_steps:
            step += 1
            print(f"--- [Step {step}] 思考中 ---")
            
            response = client.chat.completions.create(
                model=self.model_name,
                messages=messages,
                tools=tools_schema,
                tool_choice="auto"
            )
            
            message = response.choices[0].message
            
            # 如果没有工具调用，说明 Agent 认为已经完成任务并得出了结论
            if not message.tool_calls:
                print("\n✅ Agent 推理结束，最终结论：")
                print(message.content)
                return
                
            # 记录 Agent 思考的内容和希望调用的工具
            messages.append(message)
            
            # 执行所有工具调用
            for tool_call in message.tool_calls:
                if tool_call.type != "function":
                    continue
                    
                tool_name = tool_call.function.name
                
                # 处理无参数或者参数为空的场景
                try:
                    tool_args = json.loads(tool_call.function.arguments) if tool_call.function.arguments else {}
                except Exception:
                    tool_args = {}
                    
                print(f"  🔧 Agent 决定调用工具：{tool_name}，参数：{tool_args}")
                
                func = tools_map.get(tool_name)
                if func:
                    try:
                        result = func(**tool_args)
                    except Exception as e:
                        result = f"Error execution: {str(e)}"
                else:
                    result = f"Error: Tool {tool_name} not found."
                    
                # print(f"  ⬅️ 工具返回：{result[:50]}...")
                
                messages.append({
                    "tool_call_id": tool_call.id,
                    "role": "tool",
                    "name": tool_name,
                    "content": result,
                })
                
        print("\n❌ 达到最大步数限制，强制终止循环。")

if __name__ == "__main__":
    error_issue = (
        "我试图从 repomind.core 导入 QueryEngine，但提示找不到了，这是怎么回事？"
    )
    
    runner = AgentRunner(max_steps=5)
    runner.run(error_issue)
