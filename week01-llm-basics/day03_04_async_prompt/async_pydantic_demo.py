"""
Day 3-4: 异步编程与 Pydantic 结构化输出

本模块包含两个核心部分：
1. Python asyncio 异步编程基础与实践
2. 使用 Pydantic 实现 LLM 结构化输出

学习目标：
- 理解异步编程的概念和优势
- 掌握 asyncio、async/await 语法
- 学会使用 Pydantic 定义数据模型
- 实现 LLM 的结构化输出

使用前提：
- 已安装 openai 库：pip install openai
- 已安装 pydantic 库：pip install pydantic
- 已安装 python-dotenv：pip install python-dotenv
"""

import os
import asyncio
import time
from typing import Optional, List
from datetime import datetime
from dotenv import load_dotenv
from pydantic import BaseModel, Field, field_validator

# 加载环境变量
load_dotenv()


# ============================================================
# 第一部分：Asyncio 异步编程基础
# ============================================================

async def async_basic_demo():
    """
    异步编程基础演示
    
    核心概念：
        - async def: 定义异步函数（协程）
        - await: 等待异步操作完成
        - asyncio.run(): 运行异步程序的入口点
        - asyncio.gather(): 并发执行多个异步任务
    
    为什么需要异步？
        在Agent开发中，我们经常需要：
        1. 同时调用多个API（如多个LLM、多个工具）
        2. 等待I/O操作（网络请求、文件读写）
        3. 提高系统吞吐量和响应速度
        
        异步编程可以让程序在等待I/O时不阻塞，继续执行其他任务。
    """
    print("=" * 60)
    print("Asyncio 基础演示")
    print("=" * 60)
    
    # 示例1：基础异步函数
    print("\n【示例1】基础异步函数")
    
    async def say_hello(name: str, delay: float):
        """
        异步问候函数
        
        参数：
            name: 问候对象的名字
            delay: 模拟的延迟时间（秒）
        """
        print(f"  开始问候 {name}...")
        # asyncio.sleep() 是异步睡眠，不会阻塞其他任务
        await asyncio.sleep(delay)
        print(f"  你好，{name}！（延迟了 {delay} 秒）")
        return f"问候完成：{name}"
    
    # 顺序执行（慢）
    print("\n  顺序执行：")
    start = time.time()
    await say_hello("Alice", 1)
    await say_hello("Bob", 1)
    await say_hello("Charlie", 1)
    print(f"  顺序执行耗时：{time.time() - start:.2f} 秒")
    
    # 并发执行（快）
    print("\n  并发执行：")
    start = time.time()
    # asyncio.gather() 并发执行多个任务
    results = await asyncio.gather(
        say_hello("Alice", 1),
        say_hello("Bob", 1),
        say_hello("Charlie", 1)
    )
    print(f"  并发执行耗时：{time.time() - start:.2f} 秒")
    print(f"  返回结果：{results}")


async def async_llm_simulation():
    """
    模拟异步LLM调用场景
    
    功能说明：
        模拟同时向多个LLM发送请求的场景。
        展示异步在Agent开发中的实际应用。
    """
    print("\n" + "=" * 60)
    print("模拟异步 LLM 调用")
    print("=" * 60)
    
    async def simulate_llm_call(model_name: str, prompt: str, delay: float):
        """
        模拟LLM API调用
        
        参数：
            model_name: 模型名称
            prompt: 提示词
            delay: 模拟的网络延迟
        
        返回：
            dict: 包含模型名称和模拟回复的字典
        """
        print(f"  [{model_name}] 开始调用...")
        # 模拟网络请求延迟
        await asyncio.sleep(delay)
        
        # 模拟返回结果
        response = f"[{model_name}] 的回复：这是关于'{prompt}'的回答"
        print(f"  [{model_name}] 调用完成")
        
        return {
            "model": model_name,
            "response": response,
            "latency": delay
        }
    
    # 模拟同时调用多个模型
    prompts = ["解释量子计算", "什么是机器学习", "介绍Python"]
    models = ["GPT-4", "Claude", "Gemini"]
    
    print("\n  并发调用多个LLM模型：")
    start = time.time()
    
    # 并发执行所有调用
    tasks = [
        simulate_llm_call(model, prompt, delay)
        for model, prompt, delay in zip(models, prompts, [1.5, 1.0, 2.0])
    ]
    
    results = await asyncio.gather(*tasks)
    
    print(f"\n  总耗时：{time.time() - start:.2f} 秒")
    print(f"\n  所有结果：")
    for result in results:
        print(f"    - {result['response']}")


# ============================================================
# 第二部分：Pydantic 结构化输出
# ============================================================

class MovieReview(BaseModel):
    """
    电影评论数据模型
    
    功能说明：
        使用 Pydantic 定义结构化的数据模型。
        LLM 可以按照这个模型格式输出结构化数据。
    
    字段说明：
        - title: 电影名称（必填）
        - rating: 评分，1-10分（必填）
        - genre: 电影类型（必填）
        - summary: 简短总结，最多100字（必填）
        - pros: 优点列表
        - cons: 缺点列表
        - review_date: 评论日期（自动生成）
    """
    # 电影标题，使用 Field 添加描述信息
    title: str = Field(
        description="电影名称",
        min_length=1,
        max_length=100
    )
    
    # 评分，使用 ge/le 限制范围
    rating: float = Field(
        description="电影评分（1-10分）",
        ge=1.0,  # greater than or equal
        le=10.0  # less than or equal
    )
    
    # 电影类型
    genre: str = Field(
        description="电影类型",
        examples=["科幻", "动作", "剧情", "喜剧"]
    )
    
    # 简短总结
    summary: str = Field(
        description="电影简短总结",
        max_length=200
    )
    
    # 优点列表，默认为空列表
    pros: List[str] = Field(
        default_factory=list,
        description="电影优点列表"
    )
    
    # 缺点列表，默认为空列表
    cons: List[str] = Field(
        default_factory=list,
        description="电影缺点列表"
    )
    
    # 评论日期，自动生成为当前日期
    review_date: str = Field(
        default_factory=lambda: datetime.now().strftime("%Y-%m-%d"),
        description="评论日期"
    )
    
    @field_validator('rating')
    @classmethod
    def validate_rating(cls, v):
        """
        评分验证器
        
        功能：确保评分保留一位小数
        """
        return round(v, 1)
    
    def to_dict(self) -> dict:
        """
        将模型转换为字典
        
        返回：
            dict: 包含所有字段的字典
        """
        return self.model_dump()
    
    def display(self) -> str:
        """
        格式化的显示字符串
        
        返回：
            str: 格式化的电影评论展示文本
        """
        output = f"""
            电影评论：{self.title}
            {'=' * 40}
            评分：{self.rating}/10
            类型：{self.genre}
            日期：{self.review_date}

            总结：
            {self.summary}

            优点：
            {chr(10).join(f'  + {p}' for p in self.pros) if self.pros else '  无'}

            缺点：
            {chr(10).join(f'  - {c}' for c in self.cons) if self.cons else '  无'}
            {'=' * 40}
            """
        return output


class TaskExtraction(BaseModel):
    """
    任务提取模型
    
    功能说明：
        从自然语言文本中提取结构化任务信息。
        演示LLM如何将非结构化文本转换为结构化数据。
    """
    # 任务标题
    task_title: str = Field(description="任务的简短标题")
    
    # 任务描述
    description: str = Field(description="任务的详细描述")
    
    # 优先级：high/medium/low
    priority: str = Field(
        description="任务优先级",
        pattern="^(high|medium|low)$"  # 正则表达式验证
    )
    
    # 截止日期
    due_date: Optional[str] = Field(
        default=None,
        description="任务截止日期（YYYY-MM-DD格式）"
    )
    
    # 子任务列表
    subtasks: List[str] = Field(
        default_factory=list,
        description="子任务列表"
    )
    
    @field_validator('priority')
    @classmethod
    def validate_priority(cls, v):
        """验证优先级值"""
        valid_priorities = ['high', 'medium', 'low']
        if v.lower() not in valid_priorities:
            raise ValueError(f"优先级必须是 {valid_priorities} 之一")
        return v.lower()


class LLMStructuredClient:
    """
    支持结构化输出的LLM客户端
    
    功能说明：
        继承基础LLM客户端，添加结构化输出功能。
        使用 OpenAI 的 response_format 参数实现JSON模式输出。
    """
    
    def __init__(self, model: str = "gpt-3.5-turbo", temperature: float = 0.3):
        """
        初始化结构化输出客户端
        
        参数：
            model: LLM模型名称
            temperature: 使用较低温度以获得更稳定的结构化输出
        """
        self.api_key = os.getenv("OPENAI_API_KEY")
        if not self.api_key:
            raise ValueError("未找到 OPENAI_API_KEY 环境变量")
        
        self.model = model
        self.temperature = temperature
        self._client = None
    
    def _get_client(self):
        """获取OpenAI客户端实例"""
        if self._client is None:
            from openai import OpenAI
            self._client = OpenAI(api_key=self.api_key)
        return self._client
    
    def extract_structured_data(
        self,
        prompt: str,
        response_format: type[BaseModel]
    ) -> BaseModel | None:
        """
        提取结构化数据
        
        功能说明：
            使用LLM从自然语言中提取结构化信息。
            通过JSON模式确保输出符合指定的Pydantic模型。
        
        参数：
            prompt: 用户输入的提示词
            response_format: Pydantic模型类，定义输出格式
        
        返回：
            dict: 符合模型结构的字典数据
        
        使用示例：
            >>> client = LLMStructuredClient()
            >>> result = client.extract_structured_data(
            ...     "请评价电影《流浪地球2》",
            ...     MovieReview
            ... )
        """
        # 构建系统提示词，指导LLM输出JSON格式
        system_prompt = f"""
你是一个数据提取助手。请分析用户的输入，并按照以下JSON Schema格式返回结构化数据。

必须严格遵守JSON格式，不要添加任何额外说明。
"""
        
        try:
            # 调用OpenAI API，启用JSON模式
            response = self._get_client().chat.completions.create(
                model=self.model,
                messages=[
                    {"role": "system", "content": system_prompt},
                    {"role": "user", "content": prompt}
                ],
                temperature=self.temperature,
                response_format={
                    "type": "json_schema",
                    "json_schema": {
                        "name": response_format.__name__,
                        "schema": response_format.model_json_schema()
                    }
                }
            )
            
            # 解析JSON响应
            import json
            result = json.loads(response.choices[0].message.content or "{}")
            
            # 使用Pydantic模型验证并返回
            return response_format(**result)
            
        except Exception as e:
            print(f"结构化提取失败：{e}")
            return None


def demonstrate_pydantic_basics():
    """
    Pydantic 基础功能演示
    
    展示内容：
        1. 模型定义和实例化
        2. 字段验证
        3. 数据序列化
        4. 错误处理
    """
    print("=" * 60)
    print("Pydantic 基础演示")
    print("=" * 60)
    
    # 示例1：创建有效实例
    print("\n【示例1】创建有效的电影评论")
    review = None
    try:
        review = MovieReview(
            title="流浪地球2",
            rating=8.5,
            genre="科幻",
            summary="中国科幻电影的里程碑之作，视觉效果震撼",
            pros=["视觉效果出色", "剧情紧凑", "演员表演到位"],
            cons=["部分情节略显拖沓", "科学设定有争议"]
        )
        print(review.display())
    except Exception as e:
        print(f"创建失败：{e}")
    
    # 示例2：验证错误处理
    print("\n【示例2】验证错误处理")
    try:
        # 尝试创建评分超出范围的实例
        invalid_review = MovieReview(
            title="测试电影",
            rating=15.0,  # 超出1-10范围
            genre="剧情",
            summary="这是一个测试"
        )
    except Exception as e:
        print(f"验证错误（预期）：{e}")
    
    # 示例3：模型转字典
    print("\n【示例3】模型序列化")
    if review is None:
        return
    review_dict = review.to_dict()
    print(f"字典格式：{review_dict}")
    print(f"JSON Schema：{MovieReview.model_json_schema()['properties'].keys()}")


async def demonstrate_async_with_pydantic():
    """
    结合异步和Pydantic的演示
    
    功能说明：
        模拟从LLM异步获取结构化数据的场景。
    """
    print("\n" + "=" * 60)
    print("异步 + Pydantic 结合演示")
    print("=" * 60)
    
    async def simulate_structured_extraction(
        task_name: str,
        delay: float
    ) -> TaskExtraction:
        """
        模拟结构化任务提取
        
        参数：
            task_name: 任务名称
            delay: 模拟延迟
        
        返回：
            TaskExtraction: 提取的任务信息
        """
        print(f"  正在提取任务：{task_name}...")
        await asyncio.sleep(delay)
        
        # 模拟提取结果
        task = TaskExtraction(
            task_title=task_name,
            description=f"这是关于{task_name}的详细任务描述",
            priority="high" if "紧急" in task_name else "medium",
            due_date="2026-12-31",
            subtasks=["子任务1", "子任务2", "子任务3"]
        )
        
        print(f"  任务提取完成：{task_name}")
        return task
    
    # 并发提取多个任务
    tasks = ["紧急bug修复", "功能开发", "文档更新"]
    
    print("\n  并发提取多个任务：")
    results = await asyncio.gather(*[
        simulate_structured_extraction(task, delay)
        for task, delay in zip(tasks, [1.0, 1.5, 0.8])
    ])
    
    print(f"\n  提取结果：")
    for task in results:
        print(f"    - {task.task_title} [优先级：{task.priority}]")


if __name__ == "__main__":
    """
    主程序入口
    
    运行方式：
        python async_pydantic_demo.py
    """
    print("欢迎学习异步编程与结构化输出！\n")
    
    # 运行异步基础演示
    asyncio.run(async_basic_demo())
    
    # 运行异步LLM模拟
    asyncio.run(async_llm_simulation())
    
    # 运行Pydantic基础演示
    demonstrate_pydantic_basics()
    
    # 运行异步+Pydantic结合演示
    asyncio.run(demonstrate_async_with_pydantic())
    
    print("\n" + "=" * 60)
    print("学习要点总结：")
    print("1. async/await 让I/O密集型任务可以并发执行")
    print("2. asyncio.gather() 是并发执行多个任务的利器")
    print("3. Pydantic 提供强大的数据验证和序列化功能")
    print("4. 结构化输出让LLM的结果可以直接用于程序处理")
    print("=" * 60)
