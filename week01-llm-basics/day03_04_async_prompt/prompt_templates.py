"""
Day 3-4: Prompt 工程模板库

本模块实现了一个完整的Prompt工程模板库，包含：
1. System Prompt 设计原则与模板
2. Few-shot Prompting（少样本提示）
3. Chain-of-Thought (CoT) 思维链提示
4. ReAct Prompting（推理+行动提示）
5. Prompt模板管理与组合

学习目标：
- 掌握Prompt工程的核心技巧
- 理解不同Prompt模式的适用场景
- 能够设计和组合复杂的Prompt模板
- 建立可复用的Prompt模板库

使用前提：
- 已安装 openai 库：pip install openai
- 已安装 python-dotenv：pip install python-dotenv
"""

import os
from typing import Optional, List, Dict, Any
from dataclasses import dataclass, field
from dotenv import load_dotenv

# 加载环境变量
load_dotenv()


# ============================================================
# 数据模型定义
# ============================================================

@dataclass
class PromptTemplate:
    """
    Prompt 模板数据类
    
    功能说明：
        定义一个可复用的Prompt模板，包含模板内容、变量和元数据。
        支持变量替换和模板组合。
    
    属性说明：
        - name: 模板名称，用于标识和检索
        - template: 模板字符串，使用 {variable} 语法定义变量
        - description: 模板描述，说明用途和使用场景
        - variables: 模板中定义的变量列表
        - examples: 使用示例列表
    """
    # 模板名称
    name: str
    
    # 模板内容字符串
    template: str
    
    # 模板描述
    description: str = ""
    
    # 变量列表（自动从template中提取）
    variables: List[str] = field(default_factory=list)
    
    # 使用示例
    examples: List[Dict[str, str]] = field(default_factory=list)
    
    def __post_init__(self):
        """
        初始化后处理：自动提取模板变量
        
        从template字符串中提取所有 {variable} 格式的变量名
        """
        import re
        # 使用正则表达式匹配 {variable} 格式
        self.variables = re.findall(r'\{(\w+)\}', self.template)
    
    def format(self, **kwargs) -> str:
        """
        格式化模板，替换变量
        
        参数：
            **kwargs: 变量名=值的键值对
        
        返回：
            str: 替换变量后的完整Prompt
        
        使用示例：
            >>> template = PromptTemplate(
            ...     name="greeting",
            ...     template="你好，{name}！今天是{day}。"
            ... )
            >>> template.format(name="小明", day="星期一")
            '你好，小明！今天是星期一。'
        """
        try:
            return self.template.format(**kwargs)
        except KeyError as e:
            raise ValueError(f"缺少必需的变量：{e}")
    
    def validate(self, **kwargs) -> bool:
        """
        验证提供的变量是否满足模板需求
        
        参数：
            **kwargs: 要验证的变量
        
        返回：
            bool: 如果所有必需变量都已提供则返回True
        """
        return all(var in kwargs for var in self.variables)


# ============================================================
# Prompt 模板库
# ============================================================

class PromptLibrary:
    """
    Prompt 模板库管理类
    
    功能说明：
        管理和组织各种Prompt模板，支持：
        1. 模板注册和检索
        2. 模板组合
        3. 预定义常用模板
        4. 模板版本管理
    
    使用示例：
        >>> library = PromptLibrary()
        >>> library.add_template(my_template)
        >>> prompt = library.get("cot_reasoning").format(question="问题内容")
    """
    
    def __init__(self):
        """初始化模板库，加载预定义模板"""
        self._templates: Dict[str, PromptTemplate] = {}
        self._load_predefined_templates()
    
    def _load_predefined_templates(self):
        """
        加载预定义的Prompt模板
        
        包含以下类型的模板：
        1. System Prompt 模板
        2. Few-shot 示例模板
        3. Chain-of-Thought 模板
        4. ReAct 模板
        5. 其他常用模板
        """
        
        # ========== System Prompt 模板 ==========
        
        self.add_template(PromptTemplate(
            name="system_assistant",
            template=(
                "你是一个{role}助手。\n"
                "你的职责是：{responsibilities}\n"
                "请遵循以下规则：\n"
                "{rules}\n"
                "回答要求：\n"
                "{response_requirements}"
            ),
            description="通用系统提示词模板，用于定义AI助手的角色和行为",
            examples=[{
                "role": "技术专家",
                "responsibilities": "解答编程问题，提供代码示例",
                "rules": "1. 代码必须可运行\n2. 添加详细注释",
                "response_requirements": "简洁、准确、包含示例"
            }]
        ))
        
        self.add_template(PromptTemplate(
            name="system_code_reviewer",
            template=(
                "你是一位资深代码审查专家。\n"
                "请审查以下代码，关注：\n"
                "1. 代码质量和最佳实践\n"
                "2. 潜在bug和安全问题\n"
                "3. 性能和可维护性\n"
                "4. 命名规范和注释\n\n"
                "请按照以下格式输出审查结果：\n"
                "- 总体评价\n"
                "- 优点\n"
                "- 需要改进的地方\n"
                "- 具体建议"
            ),
            description="代码审查专用的System Prompt"
        ))
        
        # ========== Few-shot Prompting 模板 ==========
        
        self.add_template(PromptTemplate(
            name="few_shot_sentiment",
            template=(
                "请判断以下文本的情感倾向（正面/负面/中性）。\n\n"
                "示例：\n"
                "{examples}\n\n"
                "请判断：{text}\n"
                "情感："
            ),
            description="情感分析的Few-shot模板",
            examples=[{
                "examples": (
                    "文本：这部电影太棒了，我非常喜欢！\n情感：正面\n\n"
                    "文本：服务很差，不会再来了。\n情感：负面\n\n"
                    "文本：今天天气还不错。\n情感：中性"
                ),
                "text": "产品质量一般，但价格还算合理"
            }]
        ))
        
        self.add_template(PromptTemplate(
            name="few_shot_classification",
            template=(
                "请将以下文本分类到合适的类别中。\n\n"
                "可选类别：{categories}\n\n"
                "示例：\n"
                "{examples}\n\n"
                "请分类：{text}\n"
                "类别："
            ),
            description="文本分类的Few-shot模板"
        ))
        
        # ========== Chain-of-Thought (CoT) 模板 ==========
        
        self.add_template(PromptTemplate(
            name="cot_basic",
            template=(
                "问题：{question}\n\n"
                "请一步步思考，展示你的推理过程：\n"
                "思考步骤：\n"
                "1. 首先，我需要理解问题的核心\n"
                "2. 然后，分析关键信息和条件\n"
                "3. 接下来，运用相关知识进行推理\n"
                "4. 最后，得出结论\n\n"
                "详细推理过程：\n"
            ),
            description="基础思维链模板，引导LLM展示推理过程"
        ))
        
        self.add_template(PromptTemplate(
            name="cot_math",
            template=(
                "数学问题：{problem}\n\n"
                "请按照以下步骤解答：\n"
                "1. 理解题目：明确已知条件和求解目标\n"
                "2. 制定计划：选择合适的解题方法\n"
                "3. 执行计算：逐步展示计算过程\n"
                "4. 验证答案：检查结果是否合理\n\n"
                "解答过程：\n"
            ),
            description="数学问题的思维链模板"
        ))
        
        self.add_template(PromptTemplate(
            name="cot_decision",
            template=(
                "决策问题：{scenario}\n\n"
                "请从以下角度进行分析：\n"
                "1. 问题定义：核心问题是什么？\n"
                "2. 选项分析：有哪些可行的选择？\n"
                "3. 利弊评估：每个选项的优缺点\n"
                "4. 风险评估：潜在风险和应对策略\n"
                "5. 最终建议：综合考量后的推荐方案\n\n"
                "分析过程：\n"
            ),
            description="决策分析专用的思维链模板"
        ))
        
        # ========== ReAct Prompting 模板 ==========
        
        self.add_template(PromptTemplate(
            name="react_basic",
            template=(
                "你是一个智能助手，可以使用工具来回答问题。\n\n"
                "可用工具：\n"
                "{tools}\n\n"
                "回答格式：\n"
                "Thought: 我的思考过程\n"
                "Action: 工具名称\n"
                "Action Input: 工具输入\n"
                "Observation: 工具返回结果\n"
                "...（可以重复Thought/Action/Observation循环）...\n"
                "Thought: 我已经有了足够的信息\n"
                "Final Answer: 最终答案\n\n"
                "问题：{question}\n\n"
                "开始：\n"
            ),
            description="基础ReAct模板，定义工具使用格式"
        ))
        
        self.add_template(PromptTemplate(
            name="react_with_constraints",
            template=(
                "你是一个专业的任务执行助手。\n\n"
                "可用工具：{tools}\n\n"
                "约束条件：\n"
                "{constraints}\n\n"
                "请按照ReAct格式完成任务：\n"
                "Thought: [思考当前步骤]\n"
                "Action: [选择工具]\n"
                "Action Input: [工具输入]\n"
                "Observation: [观察结果]\n\n"
                "注意：\n"
                "1. 每次只执行一个Action\n"
                "2. 充分利用Observation进行推理\n"
                "3. 确保最终答案准确完整\n\n"
                "任务：{task}\n\n"
                "开始执行：\n"
            ),
            description="带约束条件的ReAct模板"
        ))
        
        # ========== 其他常用模板 ==========
        
        self.add_template(PromptTemplate(
            name="code_generation",
            template=(
                "请生成{language}代码来实现以下功能：\n\n"
                "功能描述：{description}\n\n"
                "要求：\n"
                "1. 代码必须完整且可运行\n"
                "2. 包含详细的注释说明\n"
                "3. 遵循{language}最佳实践\n"
                "4. 包含错误处理\n"
                "5. 提供使用示例\n\n"
                "代码：\n"
            ),
            description="代码生成专用模板"
        ))
        
        self.add_template(PromptTemplate(
            name="text_summarization",
            template=(
                "请对以下文本进行摘要：\n\n"
                "{text}\n\n"
                "要求：\n"
                "1. 保留核心信息和关键点\n"
                "2. 控制在{max_length}字以内\n"
                "3. 语言简洁准确\n"
                "4. 保持原文的语气和立场\n\n"
                "摘要：\n"
            ),
            description="文本摘要模板"
        ))
        
        self.add_template(PromptTemplate(
            name="translation",
            template=(
                "请将以下文本从{source_lang}翻译到{target_lang}：\n\n"
                "{text}\n\n"
                "翻译要求：\n"
                "1. 保持原意不变\n"
                "2. 符合目标语言的表达习惯\n"
                "3. 专业术语使用准确\n"
                "4. 保留原文的格式和结构\n\n"
                "翻译结果：\n"
            ),
            description="翻译任务模板"
        ))
    
    def add_template(self, template: PromptTemplate):
        """
        添加模板到模板库
        
        参数：
            template: PromptTemplate实例
        """
        self._templates[template.name] = template
    
    def get(self, name: str) -> PromptTemplate:
        """
        获取指定名称的模板
        
        参数：
            name: 模板名称
        
        返回：
            PromptTemplate: 模板实例
        
        异常：
            KeyError: 模板不存在时抛出
        """
        if name not in self._templates:
            raise KeyError(f"模板 '{name}' 不存在。可用模板：{list(self._templates.keys())}")
        return self._templates[name]
    
    def list_templates(self) -> List[str]:
        """
        列出所有可用模板
        
        返回：
            List[str]: 模板名称列表
        """
        return list(self._templates.keys())
    
    def combine_templates(self, template_names: List[str], separator: str = "\n\n") -> str:
        """
        组合多个模板
        
        参数：
            template_names: 要组合的模板名称列表
            separator: 模板之间的分隔符
        
        返回：
            str: 组合后的模板内容
        """
        templates = [self.get(name).template for name in template_names]
        return separator.join(templates)


# ============================================================
# Prompt 构建器
# ============================================================

class PromptBuilder:
    """
    Prompt 构建器
    
    功能说明：
        提供链式API来构建复杂的Prompt。
        支持添加系统提示、上下文、示例和用户输入。
    
    使用示例：
        >>> builder = PromptBuilder()
        >>> prompt = (builder
        ...     .system("你是一个翻译助手")
        ...     .context("原文：Hello World")
        ...     .few_shot([{"input": "Hi", "output": "你好"}])
        ...     .user("翻译：Good morning")
        ...     .build())
    """
    
    def __init__(self):
        """初始化Prompt构建器"""
        self._parts = []
        self._system_prompt = None
        self._context = None
        self._examples = []
        self._user_input = None
        self._constraints = []
    
    def system(self, prompt: str) -> 'PromptBuilder':
        """
        添加系统提示词
        
        参数：
            prompt: 系统提示词内容
        
        返回：
            PromptBuilder: 自身引用，支持链式调用
        """
        self._system_prompt = prompt
        return self
    
    def context(self, context: str) -> 'PromptBuilder':
        """
        添加上下文信息
        
        参数：
            context: 背景上下文信息
        
        返回：
            PromptBuilder: 自身引用
        """
        self._context = context
        return self
    
    def few_shot(self, examples: List[Dict[str, str]]) -> 'PromptBuilder':
        """
        添加Few-shot示例
        
        参数：
            examples: 示例列表，每个示例是一个字典
        
        返回：
            PromptBuilder: 自身引用
        """
        self._examples = examples
        return self
    
    def constraints(self, constraints: List[str]) -> 'PromptBuilder':
        """
        添加约束条件
        
        参数：
            constraints: 约束条件列表
        
        返回：
            PromptBuilder: 自身引用
        """
        self._constraints = constraints
        return self
    
    def user(self, input_text: str) -> 'PromptBuilder':
        """
        添加用户输入
        
        参数：
            input_text: 用户的实际输入/问题
        
        返回：
            PromptBuilder: 自身引用
        """
        self._user_input = input_text
        return self
    
    def build(self) -> str:
        """
        构建完整的Prompt
        
        返回：
            str: 组合好的完整Prompt
        """
        parts = []
        
        # 添加系统提示词
        if self._system_prompt:
            parts.append(f"【系统提示】\n{self._system_prompt}")
        
        # 添加上下文
        if self._context:
            parts.append(f"【上下文】\n{self._context}")
        
        # 添加Few-shot示例
        if self._examples:
            parts.append("【示例】")
            for i, example in enumerate(self._examples, 1):
                parts.append(f"示例 {i}:")
                for key, value in example.items():
                    parts.append(f"  {key}: {value}")
                parts.append("")
        
        # 添加约束条件
        if self._constraints:
            parts.append("【约束条件】")
            for i, constraint in enumerate(self._constraints, 1):
                parts.append(f"{i}. {constraint}")
            parts.append("")
        
        # 添加用户输入
        if self._user_input:
            parts.append(f"【用户输入】\n{self._user_input}")
        
        return "\n\n".join(parts)


# ============================================================
# 演示和测试函数
# ============================================================

def demonstrate_prompt_templates():
    """
    演示各种Prompt模板的使用
    """
    print("=" * 60)
    print("Prompt 模板库演示")
    print("=" * 60)
    
    # 初始化模板库
    library = PromptLibrary()
    
    # 示例1：System Prompt模板
    print("\n【示例1】System Prompt 模板")
    print("-" * 40)
    template = library.get("system_assistant")
    prompt = template.format(
        role="Python编程",
        responsibilities="解答Python问题，提供代码示例，解释最佳实践",
        rules="1. 代码必须PEP8规范\n2. 添加类型注解\n3. 包含错误处理",
        response_requirements="代码完整、注释详细、可运行"
    )
    print(prompt)
    
    # 示例2：Few-shot模板
    print("\n【示例2】Few-shot 情感分析模板")
    print("-" * 40)
    template = library.get("few_shot_sentiment")
    prompt = template.format(
        examples=(
            "文本：这部电影太棒了！\n情感：正面\n\n"
            "文本：体验很差，不推荐。\n情感：负面\n\n"
            "文本：产品收到了。\n情感：中性"
        ),
        text="物流速度一般，但包装很好"
    )
    print(prompt)
    
    # 示例3：CoT模板
    print("\n【示例3】Chain-of-Thought 模板")
    print("-" * 40)
    template = library.get("cot_basic")
    prompt = template.format(
        question="如果一个房间有3个开关，对应隔壁房间的3盏灯，你只能进入隔壁房间一次，如何确定每个开关对应哪盏灯？"
    )
    print(prompt)
    
    # 示例4：ReAct模板
    print("\n【示例4】ReAct 模板")
    print("-" * 40)
    template = library.get("react_basic")
    prompt = template.format(
        tools="- search: 搜索网络信息\n- calculate: 执行数学计算\n- read_file: 读取文件内容",
        question="2024年奥运会中国获得了多少金牌？比2020年多多少？"
    )
    print(prompt)


def demonstrate_prompt_builder():
    """
    演示PromptBuilder的使用
    """
    print("\n" + "=" * 60)
    print("Prompt Builder 演示")
    print("=" * 60)
    
    # 构建一个复杂的Prompt
    builder = PromptBuilder()
    prompt = (builder
        .system("你是一个专业的代码审查助手，擅长发现潜在问题和提供改进建议。")
        .context("以下代码是一个Python Web API的端点实现")
        .few_shot([
            {
                "代码": "def get_user(id): return db.query(id)",
                "审查": "缺少错误处理和类型注解"
            },
            {
                "代码": "def calculate(a, b): return a / b",
                "审查": "未处理除零异常"
            }
        ])
        .constraints([
            "关注安全性问题",
            "检查错误处理",
            "评估性能影响",
            "提供改进代码示例"
        ])
        .user("""
def get_users(request):
    users = database.get_all_users()
    return json(users)
        """)
        .build()
    )
    
    print("\n构建的完整Prompt：")
    print("-" * 40)
    print(prompt)


def demonstrate_template_combination():
    """
    演示模板组合功能
    """
    print("\n" + "=" * 60)
    print("模板组合演示")
    print("=" * 60)
    
    library = PromptLibrary()
    
    # 组合CoT和代码生成模板
    combined = library.combine_templates(
        ["cot_basic", "code_generation"],
        separator="\n\n---\n\n"
    )
    
    print("\n组合后的模板（CoT + 代码生成）：")
    print("-" * 40)
    print(combined)


if __name__ == "__main__":
    """
    主程序入口
    
    运行方式：
        python prompt_templates.py
    """
    print("欢迎学习 Prompt 工程！\n")
    
    # 演示模板库
    demonstrate_prompt_templates()
    
    # 演示Prompt构建器
    demonstrate_prompt_builder()
    
    # 演示模板组合
    demonstrate_template_combination()
    
    print("\n" + "=" * 60)
    print("Prompt 工程要点总结：")
    print("1. System Prompt 定义AI角色和行为准则")
    print("2. Few-shot 提供示例，引导输出格式")
    print("3. Chain-of-Thought 激发推理能力")
    print("4. ReAct 结合推理和行动（工具使用）")
    print("5. 模板化和组合提高Prompt复用性")
    print("=" * 60)
