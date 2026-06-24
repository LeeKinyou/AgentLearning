import os
import json
from typing import Any
from dotenv import load_dotenv
from openai import OpenAI

load_dotenv(os.path.join(os.path.dirname(__file__), '..', '.env'))
client = OpenAI()

# === 1. 定义后端真实工具函数 ===

def search_code(query: str, top_k: int = 5) -> str:
    """Mock 的搜索代码函数"""
    print(f"  [后端执行] 调用了 search_code, query='{query}', top_k={top_k}")
    # 为了演示，直接返回固定的 mock 数据
    if "ModuleNotFoundError" in query or "repomind" in query:
        return "找到可能的关联文件: pyproject.toml, setup.py, src/repomind/__init__.py"
    return "没有找到匹配的代码片段"

def read_file(file_path: str, start_line: int = 1, end_line: int = 100) -> str:
    """Mock 的读取文件函数"""
    print(f"  [后端执行] 调用了 read_file, file_path='{file_path}', 行范围={start_line}-{end_line}")
    if file_path == "pyproject.toml":
        return """
        [project]
        name = "repomind"
        version = "0.1.0"
        dependencies = ["requests", "pydantic"]
        # 注意：这里可能没有正确配置 packages 路径导致找不到包
        """.strip()
    return f"文件 {file_path} 读取为空"

# === 2. 将 Python 函数与其在 LLM 中的 schema 绑定 ===

# 定义给模型的工具列表
tools_schema: list[Any] = [
    {
        "type": "function",
        "function": {
            "name": "search_code",
            "description": "当用户问题需要定位仓库中的函数、类、报错或调用链时，使用该工具搜索代码片段。",
            "parameters": {
                "type": "object",
                "properties": {
                    "query": {
                        "type": "string",
                        "description": "搜索关键词或报错信息"
                    },
                    "top_k": {
                        "type": "integer",
                        "description": "返回最相关的结果数量"
                    }
                },
                "required": ["query"]
            }
        }
    },
    {
        "type": "function",
        "function": {
            "name": "read_file",
            "description": "读取指定文件的内容，用于查看代码实现或配置详情。",
            "parameters": {
                "type": "object",
                "properties": {
                    "file_path": {
                        "type": "string",
                        "description": "文件的相对路径"
                    },
                    "start_line": {
                        "type": "integer"
                    },
                    "end_line": {
                        "type": "integer"
                    }
                },
                "required": ["file_path"]
            }
        }
    }
]

# 用于在后端通过名字调用真实函数的映射表
available_tools = {
    "search_code": search_code,
    "read_file": read_file,
}

# === 3. Tool Calling 核心流 ===

def run_tool_calling_demo():
    model_name = os.getenv("MODEL_NAME", "gpt-4o-mini")
    
    # 初始的对话历史
    messages: list[Any] = [
        {"role": "system", "content": "你是一个代码仓库分析助手，你可以调用搜索和读取文件工具来帮助用户排查报错。请一步步收集证据。"},
        {"role": "user", "content": "为什么运行项目时出现 ModuleNotFoundError: No module named 'repomind'？"}
    ]
    
    print("【第一轮】：发送用户问题，带有工具列表...")
    response = client.chat.completions.create(
        model=model_name,
        messages=messages,
        tools=tools_schema,
        tool_choice="auto",
    )
    
    message = response.choices[0].message
    
    # 判断模型是否决定调用工具
    if message.tool_calls:
        print("【大模型决策】：需要调用工具！")
        # 把助手的回复加入历史中，注意需要包含 tool_calls
        messages.append(message)
        
        # 遍历模型请求调用的每一个工具
        for tool_call in message.tool_calls:
            if tool_call.type != "function":
                continue
                
            tool_name = tool_call.function.name
            tool_args = json.loads(tool_call.function.arguments)
            
            print(f"\n=> 准备执行工具: {tool_name}")
            print(f"=> 解析到的参数: {tool_args}")
            
            # 后端执行对应的 Python 函数
            function_to_call = available_tools.get(tool_name)
            if function_to_call:
                function_result = function_to_call(**tool_args)
            else:
                function_result = f"Error: Tool {tool_name} not found"
                
            print(f"<= 工具返回结果: {function_result}")
            
            # 把工具的返回结果追加到 messages 中
            messages.append(
                {
                    "tool_call_id": tool_call.id,
                    "role": "tool",
                    "name": tool_name,
                    "content": function_result,
                }
            )
            
        print("\n【第二轮】：将工具返回结果喂给模型，让其继续推理...")
        second_response = client.chat.completions.create(
            model=model_name,
            messages=messages,
            tools=tools_schema,
        )
        print("\n【大模型最终回答】：")
        print(second_response.choices[0].message.content)
        
    else:
        print("模型直接给出了回答，没有调用工具：")
        print(message.content)

if __name__ == "__main__":
    run_tool_calling_demo()
