"""
Day 1-2: LLM 初体验 - 基础对话脚本

本模块演示如何与LLM进行基础对话交互，包括：
1. 环境配置与API密钥管理
2. 首次API调用
3. 理解核心参数（Token、Temperature等）
4. 简单对话脚本实现

学习目标：
- 掌握LLM SDK的基本使用方法
- 理解API调用的核心参数及其影响
- 能够编写简单的对话程序

使用前提：
- 已安装 openai 库：pip install openai
- 已安装 python-dotenv：pip install python-dotenv
- 已在 .env 文件中配置以下环境变量：
  * OPENAI_API_KEY: API密钥（LM Studio可填写任意值，如 "lm-studio"）
  * API_BASE_URL: API基础URL
    - OpenAI官方: https://api.openai.com/v1
    - LM Studio: http://localhost:1234/v1
    - 其他兼容OpenAI API的服务
  * MODEL_NAME: 模型名称（可选）
"""

import os
from typing import Optional
from dotenv import load_dotenv

# 加载 .env 文件中的环境变量
load_dotenv()


class LLMChatClient:
    """
    LLM 对话客户端封装类
    
    功能说明：
        封装OpenAI兼容API的调用逻辑，提供简洁的对话接口。
        支持多种LLM服务：OpenAI、LM Studio、Ollama、其他兼容OpenAI API的服务。
        支持配置Temperature、Max Tokens等核心参数。
    
    核心参数说明：
        - model: 使用的LLM模型名称，如 "gpt-3.5-turbo", "gpt-4"
        - temperature: 控制输出的随机性（0.0-2.0）
            * 0.0: 输出最确定、最保守
            * 0.7: 平衡创造性和一致性（推荐默认值）
            * 2.0: 输出最随机、最有创造性
        - max_tokens: 限制生成的最大Token数量
            * Token是LLM处理文本的基本单位
            * 1个Token约等于0.75个英文单词或1个中文字符
        - system_prompt: 系统提示词，定义AI的角色和行为准则
    
    支持的LLM服务配置示例：
        
        1. OpenAI官方API:
           OPENAI_API_KEY=sk-xxx
           API_BASE_URL=https://api.openai.com/v1
           MODEL_NAME=gpt-3.5-turbo
        
        2. LM Studio本地模型:
           OPENAI_API_KEY=lm-studio  # 任意值即可
           API_BASE_URL=http://localhost:1234/v1
           MODEL_NAME=local-model    # LM Studio中加载的模型
        
        3. Ollama本地模型:
           OPENAI_API_KEY=ollama     # 任意值即可
           API_BASE_URL=http://localhost:11434/v1
           MODEL_NAME=llama2
        
        4. 其他兼容服务（如OneAPI、NewAPI等）:
           OPENAI_API_KEY=your-key
           API_BASE_URL=https://your-proxy.com/v1
           MODEL_NAME=gpt-3.5-turbo
    
    使用示例：
        >>> client = LLMChatClient(temperature=0.7, max_tokens=500)
        >>> response = client.chat("你好，请介绍一下自己")
        >>> print(response)
    """
    
    def __init__(
        self,
        model: str = "gpt-3.5-turbo",
        temperature: float = 0.7,
        max_tokens: int = 500,
        system_prompt: Optional[str] = None,
        api_key: Optional[str] = None,
        base_url: Optional[str] = None
    ):
        """
        初始化LLM对话客户端
        
        参数：
            model: LLM模型名称，默认使用gpt-3.5-turbo
            temperature: 输出随机性控制参数，范围0.0-2.0，默认0.7
            max_tokens: 最大生成Token数，默认500
            system_prompt: 系统提示词，用于定义AI角色，可选
            api_key: API密钥，优先使用此参数，否则从环境变量读取
            base_url: API基础URL，优先使用此参数，否则从环境变量读取
        
        异常：
            ValueError: 当API密钥未配置时抛出
        """
        # 从参数或环境变量获取API密钥
        self.api_key = api_key or os.getenv("OPENAI_API_KEY")
        if not self.api_key:
            raise ValueError(
                "未找到 OPENAI_API_KEY 环境变量。\n"
                "请在 .env 文件中配置：\n"
                "  OPENAI_API_KEY=your_api_key\n"
                "  API_BASE_URL=https://api.openai.com/v1  # 或LM Studio等本地服务URL"
            )
        
        # 从参数或环境变量获取base_url
        self.base_url = base_url or os.getenv("API_BASE_URL")
        if not self.base_url:
            # 默认使用OpenAI官方API
            self.base_url = "https://api.openai.com/v1"
        
        # 从环境变量获取模型名称（优先于参数）
        self.model = os.getenv("MODEL_NAME", model)
        self.temperature = temperature
        self.max_tokens = max_tokens
        
        # 默认系统提示词：定义AI为一个有帮助的助手
        self.system_prompt = system_prompt or (
            "你是一个有帮助的AI助手。请用简洁、准确的方式回答问题。"
        )
        
        # 初始化OpenAI客户端
        # 注意：这里使用延迟导入，避免未安装openai库时直接报错
        self._client = None
        
        # 打印当前配置信息（方便调试）
        print(f"[LLM配置]")
        print(f"  模型: {self.model}")
        print(f"  API地址: {self.base_url}")
        print(f"  Temperature: {self.temperature}")
        print(f"  Max Tokens: {self.max_tokens}")
        print()
        
    def _get_client(self):
        """
        获取OpenAI客户端实例（延迟初始化）
        
        返回：
            openai.OpenAI: OpenAI客户端实例
        """
        if self._client is None:
            from openai import OpenAI
            # 使用base_url支持多种LLM服务（OpenAI、LM Studio、Ollama等）
            self._client = OpenAI(
                api_key=self.api_key,
                base_url=self.base_url
            )
        return self._client
    
    def chat(
        self,
        user_message: str,
        conversation_history: Optional[list] = None
    ) -> str:
        """
        发送消息并获取LLM回复
        
        参数：
            user_message: 用户输入的消息内容
            conversation_history: 可选的对话历史记录
                格式：[{"role": "user/assistant", "content": "消息内容"}, ...]
                用于保持多轮对话的上下文连贯性
        
        返回：
            str: LLM生成的回复内容
        
        使用示例：
            # 单轮对话
            >>> client = LLMChatClient()
            >>> response = client.chat("什么是机器学习？")
            
            # 多轮对话
            >>> history = [{"role": "user", "content": "你好"}]
            >>> history.append({"role": "assistant", "content": client.chat("你好")})
            >>> response = client.chat("请解释下人工智能", conversation_history=history)
        """
        # 构建消息列表
        messages = []
        
        # 添加系统提示词（定义AI的角色和行为）
        messages.append({
            "role": "system",
            "content": self.system_prompt
        })
        
        # 添加对话历史记录（如果有）
        if conversation_history:
            messages.extend(conversation_history)
        
        # 添加当前用户消息
        messages.append({
            "role": "user",
            "content": user_message
        })
        
        try:
            # 调用OpenAI API
            # chat.completions.create 是核心的API调用方法
            response = self._get_client().chat.completions.create(
                model=self.model,           # 指定使用的模型
                messages=messages,          # 完整的消息列表
                temperature=self.temperature,  # 控制输出随机性
                max_tokens=self.max_tokens    # 限制输出长度
            )
            
            # 提取并返回生成的文本内容
            # response.choices[0].message.content 包含AI的回复
            return response.choices[0].message.content or ""
            
        except Exception as e:
            # 捕获并格式化API调用异常
            return f"API调用失败：{str(e)}"
    
    def chat_stream(self, user_message: str):
        """
        流式对话生成器（逐Token输出）
        
        功能说明：
            使用流式模式逐步获取AI回复，适合需要实时显示的场景。
            返回一个生成器，可以逐块获取生成的文本。
        
        参数：
            user_message: 用户输入的消息内容
        
        返回：
            generator: 逐块返回生成的文本内容
        
        使用示例：
            >>> client = LLMChatClient()
            >>> for chunk in client.chat_stream("讲一个故事"):
            ...     print(chunk, end="", flush=True)
        """
        messages = [
            {"role": "system", "content": self.system_prompt},
            {"role": "user", "content": user_message}
        ]
        
        # 启用stream=True开启流式输出
        response = self._get_client().chat.completions.create(
            model=self.model,
            messages=messages,  # type: ignore[arg-type]
            temperature=self.temperature,
            max_tokens=self.max_tokens,
            stream=True  # 关键参数：启用流式输出
        )
        
        # 遍历流式响应，逐块获取内容
        for chunk in response:
            # 检查是否有内容生成
            if chunk.choices[0].delta.content is not None:
                yield chunk.choices[0].delta.content


def demonstrate_temperature_effect():
    """
    演示Temperature参数对输出的影响
    
    功能说明：
        使用相同的提示词，不同的Temperature值，观察输出差异。
        帮助理解Temperature参数的实际效果。
    
    Temperature值说明：
        - 0.1: 非常保守，输出确定性高，适合事实性问题
        - 0.7: 平衡模式，适合大多数场景
        - 1.5: 创造性模式，输出更有想象力
    """
    print("=" * 60)
    print("Temperature 参数效果演示")
    print("=" * 60)
    
    # 测试提示词：要求创作性输出
    prompt = "请用一句话描述春天"
    
    # 不同的Temperature值
    temperatures = [0.1, 0.7, 1.5]
    
    for temp in temperatures:
        print(f"\n--- Temperature = {temp} ---")
        try:
            client = LLMChatClient(temperature=temp, max_tokens=50)
            response = client.chat(prompt)
            print(f"AI回复：{response}")
        except Exception as e:
            print(f"调用失败：{e}")
    
    print("\n" + "=" * 60)
    print("观察要点：")
    print("- Temperature越低，输出越稳定、保守")
    print("- Temperature越高，输出越有创造性、随机性")
    print("=" * 60)


def demonstrate_token_usage():
    """
    演示Token计数和API调用信息
    
    功能说明：
        展示如何获取API调用的Token使用量信息。
        帮助理解Token计数和API成本计算。
    """
    print("\n" + "=" * 60)
    print("Token 使用量演示")
    print("=" * 60)
    
    prompt = "请简要解释什么是人工智能"
    
    try:
        client = LLMChatClient(max_tokens=200)
        
        # 获取完整响应对象（包含usage信息）
        response = client._get_client().chat.completions.create(
            model=client.model,
            messages=[
                {"role": "system", "content": client.system_prompt},
                {"role": "user", "content": prompt}
            ],
            temperature=client.temperature,
            max_tokens=client.max_tokens
        )
        
        # 提取Token使用信息
        usage = response.usage
        assert usage is not None
        print(f"提示词Token数（Prompt Tokens）：{usage.prompt_tokens}")
        print(f"生成Token数（Completion Tokens）：{usage.completion_tokens}")
        print(f"总Token数（Total Tokens）：{usage.total_tokens}")
        print(f"\nAI回复：{response.choices[0].message.content}")
        
    except Exception as e:
        print(f"调用失败：{e}")
    
    print("\n" + "=" * 60)
    print("Token说明：")
    print("- Prompt Tokens：输入消息消耗的Token")
    print("- Completion Tokens：AI生成内容消耗的Token")
    print("- Total Tokens：总计消耗，API费用基于此计算")
    print("=" * 60)


def interactive_chat():
    """
    交互式对话模式
    
    功能说明：
        启动一个命令行交互界面，用户可以持续与AI对话。
        支持多轮对话，自动维护对话历史。
    
    使用方法：
        运行此函数后，在终端输入消息即可与AI对话。
        输入 'quit' 或 'exit' 退出对话。
    """
    print("=" * 60)
    print("交互式对话模式")
    print("=" * 60)
    print("输入消息开始对话（输入 'quit' 退出）\n")
    
    # 初始化客户端
    try:
        client = LLMChatClient(temperature=0.7, max_tokens=500)
    except ValueError as e:
        print(f"初始化失败：{e}")
        return
    
    # 维护对话历史
    conversation_history = []
    
    while True:
        # 获取用户输入
        user_input = input("你：").strip()
        
        # 检查退出条件
        if user_input.lower() in ['quit', 'exit', 'q']:
            print("感谢使用，再见！")
            break
        
        # 跳过空输入
        if not user_input:
            continue
        
        # 调用API获取回复
        print("\nAI：", end="", flush=True)
        
        # 使用流式输出，实时显示AI回复
        for chunk in client.chat_stream(user_input):
            print(chunk, end="", flush=True)
        
        print()  # 换行
        
        # 更新对话历史
        conversation_history.append({"role": "user", "content": user_input})
        # 注意：流式模式下需要单独获取完整回复存入历史
        # 这里简化处理，实际应用中应保存完整回复


if __name__ == "__main__":
    """
    主程序入口
    
    运行方式：
        python llm_basics.py
    
    默认执行顺序：
        1. Temperature参数效果演示
        2. Token使用量演示
        3. 交互式对话模式
    """
    print("欢迎使用 LLM 初体验教程！\n")
    
    # 演示Temperature参数效果
    demonstrate_temperature_effect()
    
    # 演示Token使用量
    demonstrate_token_usage()
    
    # 启动交互式对话
    print("\n即将进入交互式对话模式...\n")
    interactive_chat()
