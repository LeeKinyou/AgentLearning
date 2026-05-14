"""
Day 1-3: LangChain基础 - Chain、Agent、Tool与记忆系统

本模块演示LangChain框架的核心抽象，包括：
1. Chain：顺序链、并行链、条件链
2. Agent：ReAct Agent、OpenAI Functions Agent
3. Tool：内置工具与自定义工具
4. 记忆系统（Memory）：
   - ConversationBufferMemory
   - ConversationSummaryMemory
   - VectorStore-backed Memory

学习目标：
- 掌握LangChain核心抽象的使用方法
- 理解Chain、Agent、Tool之间的关系
- 学会使用不同类型的记忆系统
- 能够用LangChain重写手写的ReAct Agent

依赖安装：
    pip install langchain langchain-openai langchain-community chromadb pydantic python-dotenv
"""

import os
import sys
from typing import List, Dict, Optional, Any
from dotenv import load_dotenv
from pydantic import SecretStr

# 加载环境变量
load_dotenv()

# 添加项目根目录到路径
sys.path.insert(0, os.path.join(os.path.dirname(__file__), '..'))


# ============================================================
# 第一部分：LLM初始化
# ============================================================

def create_llm(temperature: float = 0.0, model: str | None = None):
    """
    创建LLM实例
    
    功能说明：
        根据环境变量配置创建LangChain兼容的LLM实例。
        支持OpenAI、LM Studio、Ollama等多种服务。
    
    参数：
        temperature: 输出随机性（0.0-2.0）
        model: 模型名称，优先使用环境变量MODEL_NAME
    
    返回：
        BaseChatModel: LangChain聊天模型实例
    
    使用示例：
        >>> llm = create_llm(temperature=0.7)
        >>> response = llm.invoke("你好")
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
# 第二部分：Chain（链）
# ============================================================

def demonstrate_chains():
    """
    演示LangChain中的各种Chain
    
    功能说明：
        展示三种基本Chain类型：
        1. 顺序链（Sequential Chain）：按顺序执行多个步骤
        2. LCEL链（LangChain Expression Language）：现代链构建方式
        3. 条件链：根据输入决定执行路径
    """
    print("=" * 60)
    print("LangChain Chain 演示")
    print("=" * 60)
    
    llm = create_llm(temperature=0.7)
    
    # --- 1. LCEL链（推荐方式）---
    print("\n[1] LCEL链（LangChain Expression Language）")
    print("-" * 40)
    
    from langchain_core.prompts import ChatPromptTemplate
    from langchain_core.output_parsers import StrOutputParser
    
    # 使用LCEL构建链：Prompt -> LLM -> 输出解析
    # LCEL是LangChain推荐的现代链构建方式，替代了旧的Chain类
    prompt = ChatPromptTemplate.from_messages([
        ("system", "你是一个简洁的助手。请用一句话回答问题。"),
        ("user", "{question}")
    ])
    
    # 使用 | 操作符连接组件，形成处理管道
    # 数据流向：输入 -> prompt -> llm -> output_parser -> 输出
    simple_chain = prompt | llm | StrOutputParser()
    
    # 调用链
    result = simple_chain.invoke({"question": "什么是机器学习？"})
    print(f"问题：什么是机器学习？")
    print(f"回答：{result}")
    
    # --- 2. 多步骤顺序链 ---
    print("\n[2] 多步骤顺序链")
    print("-" * 40)
    
    # 第一步：生成大纲
    outline_prompt = ChatPromptTemplate.from_messages([
        ("system", "你是一个专业的写作助手。请为以下主题生成一个简要大纲，只列出3-5个要点。"),
        ("user", "主题：{topic}")
    ])
    
    outline_chain = outline_prompt | llm | StrOutputParser()
    
    # 第二步：根据大纲扩展内容
    expand_prompt = ChatPromptTemplate.from_messages([
        ("system", "你是一个专业的写作助手。请根据以下大纲，为每个要点写一段简短的说明。"),
        ("user", "大纲：\n{outline}")
    ])
    
    expand_chain = expand_prompt | llm | StrOutputParser()
    
    # 组合成顺序链
    # 先执行outline_chain，将其输出作为expand_chain的输入
    sequential_chain = outline_chain | expand_chain
    
    result = sequential_chain.invoke({"topic": "Python编程入门"})
    print(f"主题：Python编程入门")
    print(f"扩展内容：{result[:300]}...")
    
    # --- 3. 带输出结构的链 ---
    print("\n[3] 带输出结构的链（Pydantic输出解析）")
    print("-" * 40)
    
    from pydantic import BaseModel, Field
    
    # 定义输出结构
    class BookRecommendation(BaseModel):
        """书籍推荐输出结构"""
        title: str = Field(description="书名")
        author: str = Field(description="作者")
        reason: str = Field(description="推荐理由")
        difficulty: str = Field(description="难度级别：入门/中级/高级")
    
    # 使用with_structured_output实现结构化输出
    structured_llm = llm.with_structured_output(BookRecommendation)
    
    rec_prompt = ChatPromptTemplate.from_messages([
        ("system", "你是一个图书推荐专家。请根据用户的兴趣推荐一本书。"),
        ("user", "我对{interest}感兴趣，请推荐一本书。")
    ])
    
    recommendation_chain = rec_prompt | structured_llm
    
    recommendation: BookRecommendation = recommendation_chain.invoke({"interest": "人工智能"})
    print(f"推荐书籍：{recommendation.title}")
    print(f"作者：{recommendation.author}")
    print(f"推荐理由：{recommendation.reason}")
    print(f"难度：{recommendation.difficulty}")


# ============================================================
# 第三部分：Tool（工具）
# ============================================================

def demonstrate_tools():
    """
    演示LangChain中的Tool使用
    
    功能说明：
        展示如何定义和使用自定义工具。
        工具是Agent执行操作的基本单元。
    """
    print("\n" + "=" * 60)
    print("LangChain Tool 演示")
    print("=" * 60)
    
    from langchain_core.tools import tool
    
    # --- 1. 使用@tool装饰器定义工具 ---
    @tool
    def calculator(expression: str) -> str:
        """
        计算数学表达式
        
        参数：
            expression: 数学表达式，如 "2 + 3 * 4"
        
        返回：
            str: 计算结果
        """
        import math
        import re
        
        try:
            safe_expr = expression.replace('sqrt', 'math.sqrt')
            safe_expr = safe_expr.replace('pow', 'math.pow')
            safe_expr = re.sub(r'[^0-9+\-*/().\s\w]', '', safe_expr)
            result = eval(safe_expr, {"__builtins__": {}, "math": math})
            return str(result)
        except Exception as e:
            return f"计算错误：{str(e)}"
    
    @tool
    def get_current_date(format: str = "%Y-%m-%d") -> str:
        """
        获取当前日期
        
        参数：
            format: 日期格式，默认 %Y-%m-%d
        
        返回：
            str: 格式化的当前日期
        """
        from datetime import datetime
        return datetime.now().strftime(format)
    
    @tool
    def text_length(text: str) -> str:
        """
        计算文本长度
        
        参数：
            text: 要计算长度的文本
        
        返回：
            str: 字符数和单词数
        """
        chars = len(text)
        words = len(text.split())
        return f"字符数：{chars}，单词数：{words}"
    
    # 注册工具列表
    tools = [calculator, get_current_date, text_length]
    
    print(f"\n已注册 {len(tools)} 个工具：")
    for t in tools:
        print(f"  - {t.name}: {t.description}")
    
    # 测试工具
    print(f"\n测试工具调用：")
    print(f"  calculator('2 + 3 * 4') = {calculator.invoke('2 + 3 * 4')}")
    print(f"  get_current_date() = {get_current_date.invoke('')}")
    print(f"  text_length('Hello World') = {text_length.invoke('Hello World')}")
    
    return tools


# ============================================================
# 第四部分：Memory（记忆系统）
# ============================================================

def demonstrate_memory():
    """
    演示LangChain中的记忆系统
    
    功能说明：
        展示三种记忆系统：
        1. ConversationBufferMemory：保存完整对话历史
        2. ConversationSummaryMemory：总结对话历史
        3. ConversationBufferWindowMemory：保存最近N轮对话
    """
    print("\n" + "=" * 60)
    print("LangChain Memory 演示")
    print("=" * 60)
    
    from langchain_core.prompts import ChatPromptTemplate, MessagesPlaceholder
    from langchain_core.output_parsers import StrOutputParser
    from langchain_core.chat_history import InMemoryChatMessageHistory
    from langchain_core.messages import HumanMessage, AIMessage
    
    llm = create_llm(temperature=0.7)
    
    # --- 1. ConversationBufferMemory（对话缓冲区记忆）---
    print("\n[1] ConversationBufferMemory（完整对话历史）")
    print("-" * 40)
    
    buffer_memory = InMemoryChatMessageHistory()
    
    # 添加对话
    buffer_memory.add_message(HumanMessage(content="你好，我叫小明"))
    buffer_memory.add_message(AIMessage(content="你好小明！很高兴认识你。"))
    buffer_memory.add_message(HumanMessage(content="我喜欢编程"))
    buffer_memory.add_message(AIMessage(content="编程是一项很有用的技能！"))
    
    # 查看记忆内容
    print(f"记忆中的对话：")
    for msg in buffer_memory.messages:
        print(f"  {msg.type}: {msg.content}")
    
    # --- 2. ConversationSummaryMemory（对话摘要）---
    print("\n[2] ConversationSummaryMemory（对话摘要）")
    print("-" * 40)
    
    summary_memory = InMemoryChatMessageHistory()
    
    # 添加对话
    summary_memory.add_message(HumanMessage(content="我想学习Python"))
    summary_memory.add_message(AIMessage(content="Python是一门很好的入门语言。"))
    summary_memory.add_message(HumanMessage(content="有什么推荐资源吗？"))
    summary_memory.add_message(AIMessage(content="推荐官方文档和Codecademy。"))
    
    # 使用LLM生成摘要
    summary_prompt = ChatPromptTemplate.from_messages([
        ("system", "请用一句话总结以下对话内容："),
        MessagesPlaceholder(variable_name="messages"),
    ])
    summary_chain = summary_prompt | llm | StrOutputParser()
    summary_text = summary_chain.invoke({"messages": summary_memory.messages})
    print(f"对话摘要：{summary_text}")
    
    # --- 3. 带记忆的对话链 ---
    print("\n[3] 带记忆的对话链")
    print("-" * 40)
    
    # 创建带记忆的Prompt
    # MessagesPlaceholder用于在Prompt中插入历史消息
    memory_prompt = ChatPromptTemplate.from_messages([
        ("system", "你是一个友好的助手。请根据对话历史回答问题。"),
        MessagesPlaceholder(variable_name="chat_history"),
        ("human", "{input}")
    ])
    
    # 构建带记忆的链
    memory_chain = memory_prompt | llm | StrOutputParser()
    
    # 模拟多轮对话
    chat_history = []
    
    questions = [
        "你好，我叫小红",
        "我喜欢画画，你有什么建议吗？",
        "我刚才说了我喜欢什么？"
    ]
    
    for question in questions:
        print(f"\n用户：{question}")
        
        # 调用链，传入当前对话历史
        response = memory_chain.invoke({
            "input": question,
            "chat_history": chat_history
        })
        
        print(f"助手：{response}")
        
        # 更新对话历史
        chat_history.append(HumanMessage(content=question))
        chat_history.append(AIMessage(content=response))


# ============================================================
# 第五部分：LangChain ReAct Agent
# ============================================================

def demonstrate_langchain_agent():
    """
    演示使用LangChain构建ReAct Agent
    
    功能说明：
        使用LangChain的内置Agent类型重写Week01的手写ReAct Agent。
        展示LangChain如何简化Agent的构建过程。
    """
    print("\n" + "=" * 60)
    print("LangChain ReAct Agent 演示")
    print("=" * 60)
    
    from langchain_core.tools import tool
    from langgraph.prebuilt import create_react_agent
    from langchain_core.messages import HumanMessage
    
    llm = create_llm(temperature=0.0)
    
    # 定义工具
    @tool
    def calculator(expression: str) -> str:
        """计算数学表达式。参数：数学表达式字符串，如 '2 + 3 * 4' 或 'sqrt(16)'"""
        import math, re
        try:
            safe_expr = expression.replace('sqrt', 'math.sqrt').replace('pow', 'math.pow')
            safe_expr = re.sub(r'[^0-9+\-*/().\s\w]', '', safe_expr)
            result = eval(safe_expr, {"__builtins__": {}, "math": math})
            return str(result)
        except Exception as e:
            return f"计算错误：{str(e)}"
    
    @tool
    def get_date_info(query: str) -> str:
        """获取当前日期信息。参数：查询字符串（可忽略）"""
        from datetime import datetime
        now = datetime.now()
        return f"今天是 {now.strftime('%Y年%m月%d日')}，{now.strftime('%A')}"
    
    tools = [calculator, get_date_info]
    
    # 使用LangGraph的create_react_agent创建Agent
    # 这是新版langchain推荐的Agent构建方式
    agent = create_react_agent(llm, tools)
    
    # 测试Agent
    test_questions = [
        "2的10次方是多少？",
        "今天是星期几？",
        "sqrt(144)等于多少？",
    ]
    
    for question in test_questions:
        print(f"\n{'=' * 40}")
        print(f"问题：{question}")
        print(f"{'=' * 40}")
        
        result = agent.invoke({"messages": [HumanMessage(content=question)]})
        
        # 提取最后一条AI消息作为答案
        from langchain_core.messages import AIMessage, ToolMessage
        last_ai_msg = None
        for msg in reversed(result.get("messages", [])):
            if isinstance(msg, AIMessage) and msg.content:
                last_ai_msg = msg
                break
        
        if last_ai_msg:
            print(f"\n最终答案：{last_ai_msg.content}")
        else:
            print(f"\n最终答案：无回复")


# ============================================================
# 第六部分：综合演示
# ============================================================

def main():
    """
    主程序入口
    
    运行方式：
        python langchain_basics.py
    """
    print("=" * 60)
    print("Week03 Day 1-3: LangChain基础")
    print("=" * 60)
    
    # 演示Chain
    demonstrate_chains()
    
    # 演示Tool
    tools = demonstrate_tools()
    
    # 演示Memory
    demonstrate_memory()
    
    # 演示Agent
    demonstrate_langchain_agent()
    
    print("\n" + "=" * 60)
    print("演示完成！")
    print("=" * 60)


if __name__ == "__main__":
    main()
