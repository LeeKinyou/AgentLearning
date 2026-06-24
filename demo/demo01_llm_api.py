import os
import json
from dotenv import load_dotenv
from openai import OpenAI
from pydantic import BaseModel, Field
from typing import Any
from openai.types.chat import ChatCompletionMessageParam

# 加载环境变量 (读取项目根目录的 .env 文件)
load_dotenv(os.path.join(os.path.dirname(__file__), '..', '.env'))

# 初始化 OpenAI 客户端
# 默认会从环境变量读取 OPENAI_API_KEY 和 OPENAI_BASE_URL (若配置了的话)
client = OpenAI()

# 定义希望大模型输出的结构化 JSON 格式
class ErrorAnalysis(BaseModel):
    error_type: str = Field(description="错误的具体类型，例如 ModuleNotFoundError")
    possible_reason: str = Field(description="导致该报错的可能原因分析")
    suggestion: str = Field(description="给出可执行的修复建议")
    need_more_context: bool = Field(description="当前上下文是否足够定位问题，如果需要更多代码或文件请设为 true")

def analyze_error_log(error_log: str) -> str:
    """调用大模型分析报错日志并返回结构化结果"""
    messages: list[ChatCompletionMessageParam] = [
        {
            "role": "system",
            "content": (
                "你是一个资深的 Python 后端开发专家和排错助手。\n"
                "任务：根据用户提供的报错日志，进行准确的问题定位和分析。\n"
                "限制：请只基于报错内容回答，不要过度发散，必须以结构化的 JSON 格式返回结果。"
            )
        },
        {
            "role": "user",
            "content": f"请帮我分析这段报错日志：\n\n{error_log}"
        }
    ]

    try:
        # 使用模型，如果未指定 MODEL 环境变量，默认使用 gpt-4o-mini
        model_name = os.getenv("MODEL_NAME", "gpt-4o-mini")
        
        print(f"正在调用模型 {model_name} 分析报错...")
        
        # 使用 OpenAI 的 beta.pydantic_response_format 强制输出结构化 JSON
        response = client.beta.chat.completions.parse(
            model=model_name,
            messages=messages,
            temperature=0.2, # 较低的 temperature 保持输出稳定
            response_format=ErrorAnalysis,
            max_tokens=800
        )
        
        # 获取解析后的 Pydantic 对象
        analysis = response.choices[0].message.parsed
        if analysis is None:
            return json.dumps({"error": "模型未能返回预期的结构化结果"}, ensure_ascii=False)
            
        # 将 Pydantic 对象转换为漂亮打印的 JSON 字符串返回
        return analysis.model_dump_json(indent=2)
        
    except Exception as e:
        return json.dumps({"error": f"大模型调用失败: {str(e)}"}, ensure_ascii=False)

if __name__ == "__main__":
    # 模拟用户遇到的一段 Python 报错日志
    sample_error_log = """
Traceback (most recent call last):
  File "main.py", line 2, in <module>
    from repomind.core import QueryEngine
ModuleNotFoundError: No module named 'repomind'
    """
    
    print("【输入】用户报错日志：")
    print(sample_error_log.strip())
    print("-" * 40)
    
    result_json = analyze_error_log(sample_error_log)
    
    print("【输出】模型分析结果 (JSON格式)：")
    print(result_json)
