"""
Day 5-7: 可观测性与评估系统

本模块实现RAG系统的可观测性和自动化评估：
1. LangFuse调用链接入（Trace/Span概念）
2. 关键指标埋点（延迟、Token消耗、成本）
3. Ragas评估框架集成
4. 核心指标：Faithfulness、Answer Relevance、Context Precision

学习目标：
- 理解可观测性在RAG系统中的重要性
- 掌握Trace/Span的概念和使用方法
- 学会使用LangFuse记录调用链
- 能够使用Ragas进行自动化评估

依赖安装：
    pip install langfuse ragas chromadb openai pydantic python-dotenv
"""

import os
import sys
import time
import json
from typing import List, Dict, Optional, Any
from dataclasses import dataclass, asdict
from datetime import datetime
from dotenv import load_dotenv

# 加载环境变量
load_dotenv()

# 添加项目根目录到路径，支持跨模块导入
sys.path.insert(0, os.path.join(os.path.dirname(__file__), '..'))


# ============================================================
# 第一部分：可观测性数据模型
# ============================================================

@dataclass
class TraceSpan:
    """
    Trace Span 数据模型
    
    功能说明：
        表示一次操作的时间跨度，记录操作的开始、结束时间和元数据。
        Span是Trace的组成部分，一个Trace可以包含多个Span。
    
    核心概念：
        - Trace: 一次完整的请求链路（如一次用户查询）
        - Span: Trace中的一个操作步骤（如检索、LLM调用）
    
    属性：
        name: Span名称（如"retrieval", "llm_call"）
        start_time: 开始时间（Unix时间戳）
        end_time: 结束时间（Unix时间戳）
        duration_ms: 持续时间（毫秒）
        metadata: 附加元数据
        status: 状态（success/error）
    
    使用示例：
        >>> span = TraceSpan(name="retrieval")
        >>> span.start()
        >>> # ... 执行操作 ...
        >>> span.end()
        >>> print(f"耗时：{span.duration_ms}ms")
    """
    name: str
    start_time: Optional[float] = None
    end_time: Optional[float] = None
    duration_ms: Optional[float] = None
    metadata: Optional[Dict] = None
    status: str = "pending"
    
    def __post_init__(self):
        """数据初始化后处理"""
        if self.metadata is None:
            self.metadata = {}
    
    def start(self):
        """开始计时"""
        self.start_time = time.time()
        self.status = "running"
    
    def end(self, status: str = "success"):
        """
        结束计时
        
        参数：
            status: 操作状态（success/error）
        """
        self.end_time = time.time()
        self.duration_ms = (self.end_time - (self.start_time or 0)) * 1000
        self.status = status
    
    def to_dict(self) -> Dict:
        """转换为字典"""
        return asdict(self)


@dataclass
class Trace:
    """
    Trace 数据模型
    
    功能说明：
        表示一次完整的请求链路，包含多个Span。
        用于追踪从用户查询到答案生成的完整过程。
    
    属性：
        trace_id: Trace唯一标识符
        name: Trace名称
        start_time: 开始时间
        end_time: 结束时间
        spans: 包含的Span列表
        metadata: 附加元数据
    
    使用示例：
        >>> trace = Trace(name="rag_query")
        >>> span = trace.add_span("retrieval")
        >>> span.start()
        >>> # ... 执行检索 ...
        >>> span.end()
        >>> trace.end()
    """
    trace_id: str
    name: str
    start_time: Optional[float] = None
    end_time: Optional[float] = None
    spans: Optional[List[TraceSpan]] = None
    metadata: Optional[Dict] = None
    
    def __post_init__(self):
        """数据初始化后处理"""
        if self.spans is None:
            self.spans = []
        if self.metadata is None:
            self.metadata = {}

    def add_span(self, name: str) -> TraceSpan:
        """
        添加新的Span

        参数：
            name: Span名称

        返回：
            TraceSpan: 新创建的Span
        """
        span = TraceSpan(name=name)
        self.spans.append(span)  # type: ignore[union-attr]
        return span

    def start(self):
        """开始Trace"""
        self.start_time = time.time()

    def end(self):
        """结束Trace"""
        self.end_time = time.time()
        if self.metadata is not None:
            self.metadata['total_duration_ms'] = (self.end_time - (self.start_time or 0)) * 1000

    def to_dict(self) -> Dict:
        """转换为字典"""
        return {
            'trace_id': self.trace_id,
            'name': self.name,
            'start_time': self.start_time,
            'end_time': self.end_time,
            'spans': [span.to_dict() for span in (self.spans or [])],
            'metadata': self.metadata
        }


# ============================================================
# 第二部分：指标收集器
# ============================================================

class MetricsCollector:
    """
    指标收集器
    
    功能说明：
        收集RAG系统运行过程中的关键指标，包括：
        - 延迟指标：检索时间、LLM调用时间、总时间
        - Token消耗：输入Token、输出Token、总Token
        - 成本估算：基于Token使用量计算费用
    
    使用示例：
        >>> collector = MetricsCollector()
        >>> collector.record_retrieval_time(150.5)
        >>> collector.record_tokens(prompt=100, completion=200)
        >>> report = collector.generate_report()
    """
    
    def __init__(self):
        """初始化指标收集器"""
        self.metrics = {
            'retrieval_times': [],      # 检索时间列表（毫秒）
            'llm_times': [],            # LLM调用时间列表（毫秒）
            'total_times': [],          # 总时间列表（毫秒）
            'prompt_tokens': [],        # 输入Token列表
            'completion_tokens': [],    # 输出Token列表
            'total_tokens': [],         # 总Token列表
            'costs': [],                # 成本列表（美元）
            'query_count': 0            # 查询总数
        }
    
    def record_retrieval_time(self, time_ms: float):
        """
        记录检索时间
        
        参数：
            time_ms: 检索耗时（毫秒）
        """
        self.metrics['retrieval_times'].append(time_ms)
    
    def record_llm_time(self, time_ms: float):
        """
        记录LLM调用时间
        
        参数：
            time_ms: LLM调用耗时（毫秒）
        """
        self.metrics['llm_times'].append(time_ms)
    
    def record_total_time(self, time_ms: float):
        """
        记录总时间
        
        参数：
            time_ms: 总耗时（毫秒）
        """
        self.metrics['total_times'].append(time_ms)
    
    def record_tokens(self, prompt: int, completion: int):
        """
        记录Token使用量
        
        参数：
            prompt: 输入Token数
            completion: 输出Token数
        """
        self.metrics['prompt_tokens'].append(prompt)
        self.metrics['completion_tokens'].append(completion)
        self.metrics['total_tokens'].append(prompt + completion)
    
    def record_cost(self, cost: float):
        """
        记录成本
        
        参数：
            cost: 成本（美元）
        """
        self.metrics['costs'].append(cost)
    
    def increment_query_count(self):
        """增加查询计数"""
        self.metrics['query_count'] += 1
    
    def get_average(self, metric_name: str) -> float:
        """
        获取指标平均值
        
        参数：
            metric_name: 指标名称
        
        返回：
            float: 平均值
        """
        values = self.metrics.get(metric_name, [])
        if not values:
            return 0.0
        return sum(values) / len(values)
    
    def generate_report(self) -> Dict:
        """
        生成指标报告
        
        返回：
            Dict: 包含所有统计指标的报告
        """
        report = {
            'query_count': self.metrics['query_count'],
            'avg_retrieval_time_ms': self.get_average('retrieval_times'),
            'avg_llm_time_ms': self.get_average('llm_times'),
            'avg_total_time_ms': self.get_average('total_times'),
            'avg_prompt_tokens': self.get_average('prompt_tokens'),
            'avg_completion_tokens': self.get_average('completion_tokens'),
            'avg_total_tokens': self.get_average('total_tokens'),
            'total_cost': sum(self.metrics['costs']),
        }
        
        return report


# ============================================================
# 第三部分：LangFuse集成（模拟实现）
# ============================================================

class LangFuseObserver:
    """
    LangFuse可观测性集成
    
    功能说明：
        集成LangFuse平台，记录RAG系统的调用链路。
        支持Trace/Span的记录和可视化。
    
    核心概念：
        - Trace: 一次完整的用户查询链路
        - Span: Trace中的操作步骤（检索、LLM调用等）
        - Event: 特定事件（如错误、警告）
    
    配置方法：
        在.env文件中配置：
        LANGFUSE_PUBLIC_KEY=pk-lf-xxx
        LANGFUSE_SECRET_KEY=sk-lf-xxx
        LANGFUSE_HOST=https://cloud.langfuse.com
    
    使用示例：
        >>> observer = LangFuseObserver()
        >>> trace = observer.start_trace("rag_query")
        >>> span = trace.add_span("retrieval")
        >>> span.start()
        >>> # ... 执行检索 ...
        >>> span.end()
        >>> observer.end_trace(trace)
    """
    
    def __init__(self):
        """
        初始化LangFuse观察者
        
        注意：
            这里使用模拟实现，实际项目中应使用langfuse SDK。
            模拟实现演示了核心概念和工作流程。
        """
        self.public_key = os.getenv("LANGFUSE_PUBLIC_KEY")
        self.secret_key = os.getenv("LANGFUSE_SECRET_KEY")
        self.host = os.getenv("LANGFUSE_HOST", "https://cloud.langfuse.com")
        
        self.traces = []
        self._client = None
        
        if not self.public_key or not self.secret_key:
            print("[LangFuse] 未配置API密钥，使用本地记录模式")
    
    def _get_client(self):
        """
        获取LangFuse客户端
        
        返回：
            langfuse.Langfuse: LangFuse客户端实例
        """
        if self._client is None:
            if self.public_key and self.secret_key:
                try:
                    from langfuse import Langfuse
                    self._client = Langfuse(
                        public_key=self.public_key,
                        secret_key=self.secret_key,
                        host=self.host
                    )
                except ImportError:
                    print("[LangFuse] 未安装langfuse库，使用本地记录模式")
                    print("[LangFuse] 安装命令：pip install langfuse")
            else:
                print("[LangFuse] 使用本地记录模式")
        
        return self._client
    
    def start_trace(self, name: str, metadata: Optional[Dict] = None) -> Trace:
        """
        开始Trace
        
        参数：
            name: Trace名称
            metadata: 附加元数据
        
        返回：
            Trace: 新创建的Trace
        """
        trace_id = f"trace_{datetime.now().strftime('%Y%m%d_%H%M%S')}_{id(self)}"
        
        trace = Trace(
            trace_id=trace_id,
            name=name,
            metadata=metadata or {}
        )
        
        trace.start()
        self.traces.append(trace)
        
        return trace
    
    def end_trace(self, trace: Trace):
        """
        结束Trace
        
        参数：
            trace: 要结束的Trace
        """
        trace.end()
        
        # 如果配置了LangFuse，发送到云端
        client = self._get_client()
        if client:
            try:
                # 实际项目中应使用LangFuse SDK的API
                pass
            except Exception as e:
                print(f"[LangFuse] 发送失败：{e}")
    
    def get_trace_summary(self, trace: Trace) -> str:
        """
        获取Trace摘要
        
        参数：
            trace: Trace对象
        
        返回：
            str: Trace摘要信息
        """
        summary = f"Trace: {trace.name}\n"
        summary += f"  ID: {trace.trace_id}\n"
        summary += f"  总耗时: {(trace.metadata or {}).get('total_duration_ms', 0):.2f}ms\n"
        summary += f"  Span数量: {len(trace.spans or [])}\n"

        for span in (trace.spans or []):
            status_icon = "✓" if span.status == "success" else "✗"
            summary += f"    {status_icon} {span.name}: {span.duration_ms:.2f}ms\n"
        
        return summary


# ============================================================
# 第四部分：Ragas评估框架集成
# ============================================================

@dataclass
class EvaluationSample:
    """
    评估样本数据模型
    
    功能说明：
        表示一个用于评估的样本，包含问题、上下文、答案和参考答案。
        用于Ragas评估框架的数据输入。
    
    属性：
        question: 用户问题
        contexts: 检索到的上下文列表
        answer: 生成的答案
        ground_truth: 参考答案（可选）
    """
    question: str
    contexts: List[str]
    answer: str
    ground_truth: Optional[str] = None


class RagasEvaluator:
    """
    Ragas评估器
    
    功能说明：
        集成Ragas评估框架，对RAG系统进行自动化评估。
        支持多种评估指标：
        - Faithfulness: 答案是否基于上下文（忠实度）
        - Answer Relevance: 答案是否与问题相关
        - Context Precision: 上下文是否精确
        - Context Recall: 上下文是否召回了必要信息
    
    核心指标说明：
        1. Faithfulness（忠实度）
           - 评估答案是否完全基于提供的上下文
           - 防止LLM产生幻觉（Hallucination）
           - 分数越高，答案越忠实于上下文
        
        2. Answer Relevance（答案相关性）
           - 评估答案是否直接回答了问题
           - 检查答案是否偏离主题
           - 分数越高，答案越相关
        
        3. Context Precision（上下文精确度）
           - 评估检索到的上下文是否精确相关
           - 检查是否检索到了不相关的文档
           - 分数越高，上下文越精确
        
        4. Context Recall（上下文召回率）
           - 评估是否检索到了所有必要的上下文
           - 检查是否遗漏了关键信息
           - 分数越高，召回越完整
    
    使用示例：
        >>> evaluator = RagasEvaluator()
        >>> results = evaluator.evaluate(samples)
        >>> print(f"Faithfulness: {results['faithfulness']:.3f}")
    """
    
    def __init__(self):
        """初始化Ragas评估器"""
        self.results = []
    
    def evaluate_faithfulness(self, sample: EvaluationSample) -> float:
        """
        评估忠实度
        
        参数：
            sample: 评估样本
        
        返回：
            float: 忠实度分数（0-1）
        
        评估逻辑：
            1. 提取答案中的所有陈述
            2. 检查每个陈述是否能在上下文中找到依据
            3. 计算有依据的陈述比例
        """
        # 模拟实现：基于关键词重叠度估算
        context_text = " ".join(sample.contexts).lower()
        answer_words = set(sample.answer.lower().split())
        
        if not answer_words:
            return 0.0
        
        # 计算答案词在上下文中的覆盖率
        matching_words = sum(1 for word in answer_words if word in context_text)
        score = matching_words / len(answer_words)
        
        return min(1.0, score)
    
    def evaluate_answer_relevance(self, sample: EvaluationSample) -> float:
        """
        评估答案相关性
        
        参数：
            sample: 评估样本
        
        返回：
            float: 相关性分数（0-1）
        
        评估逻辑：
            1. 分析问题中的关键词
            2. 检查答案是否包含相关概念
            3. 计算答案与问题的语义相关性
        """
        # 模拟实现：基于问题-答案关键词重叠
        question_words = set(sample.question.lower().split())
        answer_words = set(sample.answer.lower().split())
        
        if not question_words:
            return 0.0
        
        # 计算重叠度
        matching_words = question_words.intersection(answer_words)
        score = len(matching_words) / len(question_words)
        
        return min(1.0, score)
    
    def evaluate_context_precision(self, sample: EvaluationSample) -> float:
        """
        评估上下文精确度
        
        参数：
            sample: 评估样本
        
        返回：
            float: 精确度分数（0-1）
        
        评估逻辑：
            1. 检查每个上下文片段是否与问题相关
            2. 计算相关上下文的比例
            3. 比例越高，精确度越高
        """
        # 模拟实现：基于上下文与问题的相关性
        question_words = set(sample.question.lower().split())
        
        if not sample.contexts:
            return 0.0
        
        relevant_count = 0
        for context in sample.contexts:
            context_words = set(context.lower().split())
            # 计算重叠度
            overlap = len(question_words.intersection(context_words))
            if overlap > 0:
                relevant_count += 1
        
        return relevant_count / len(sample.contexts)
    
    def evaluate_sample(self, sample: EvaluationSample) -> Dict:
        """
        评估单个样本
        
        参数：
            sample: 评估样本
        
        返回：
            Dict: 包含各项指标分数的字典
        """
        return {
            'faithfulness': self.evaluate_faithfulness(sample),
            'answer_relevance': self.evaluate_answer_relevance(sample),
            'context_precision': self.evaluate_context_precision(sample),
        }
    
    def evaluate_batch(self, samples: List[EvaluationSample]) -> Dict:
        """
        批量评估
        
        参数：
            samples: 评估样本列表
        
        返回：
            Dict: 包含平均指标分数的字典
        """
        all_scores = {
            'faithfulness': [],
            'answer_relevance': [],
            'context_precision': [],
        }
        
        for sample in samples:
            scores = self.evaluate_sample(sample)
            for metric, score in scores.items():
                all_scores[metric].append(score)
        
        # 计算平均分
        avg_scores = {}
        for metric, scores in all_scores.items():
            if scores:
                avg_scores[metric] = sum(scores) / len(scores)
            else:
                avg_scores[metric] = 0.0
        
        return avg_scores
    
    def generate_report(self, scores: Dict) -> str:
        """
        生成评估报告
        
        参数：
            scores: 评估分数
        
        返回：
            str: 格式化的评估报告
        """
        report = "=" * 60 + "\n"
        report += "RAG系统评估报告\n"
        report += "=" * 60 + "\n\n"
        
        for metric, score in scores.items():
            # 指标名称映射
            metric_names = {
                'faithfulness': '忠实度（Faithfulness）',
                'answer_relevance': '答案相关性（Answer Relevance）',
                'context_precision': '上下文精确度（Context Precision）',
            }
            
            name = metric_names.get(metric, metric)
            report += f"{name}: {score:.3f}\n"
        
        report += "\n" + "=" * 60 + "\n"
        report += "指标说明：\n"
        report += "- 忠实度：答案是否基于上下文，防止幻觉\n"
        report += "- 答案相关性：答案是否直接回答了问题\n"
        report += "- 上下文精确度：检索到的上下文是否精确相关\n"
        report += "=" * 60
        
        return report


# ============================================================
# 第五部分：可观测性RAG系统
# ============================================================

class ObservableRAGSystem:
    """
    可观测性RAG系统
    
    功能说明：
        整合RAG核心功能与可观测性、评估体系的完整系统。
        每次查询都会记录详细的调用链路和指标。
    
    功能特性：
        1. 完整的Trace/Span记录
        2. 延迟、Token、成本指标收集
        3. 自动化评估集成
        4. 报告生成
    
    使用示例：
        >>> rag = ObservableRAGSystem()
        >>> rag.index_documents("./docs")
        >>> answer = rag.query("如何配置Python？")
        >>> rag.generate_report()
    """
    
    def __init__(self, collection_name: str = "observable_rag"):
        """
        初始化可观测性RAG系统
        
        参数：
            collection_name: 向量数据库集合名称
        """
        # 导入基础组件
        from day01_02_basic_rag.basic_rag_pipeline import (
            DocumentLoader, TextChunker, VectorStore, LLMClient, SearchResult
        )
        from day03_04_advanced_rag.advanced_rag_strategies import HybridRetriever
        
        # 初始化基础组件
        self.loader = DocumentLoader()
        self.chunker = TextChunker(chunk_size=500, chunk_overlap=50)
        self.hybrid_retriever = HybridRetriever()
        self.llm = LLMClient(temperature=0.0)
        
        # 初始化可观测性组件
        self.observer = LangFuseObserver()
        self.metrics = MetricsCollector()
        self.evaluator = RagasEvaluator()
        
        # 评估样本收集
        self.evaluation_samples = []
    
    def index_documents(self, source: str, source_type: str = "directory"):
        """
        索引文档
        
        参数：
            source: 文档来源
            source_type: 来源类型（file/directory）
        """
        print(f"[1/3] 加载文档：{source}")
        
        if source_type == "file":
            documents = self.loader.load_from_file(source)
        else:
            documents = self.loader.load_from_directory(source)
        
        print(f"  加载了 {len(documents)} 个文档")
        
        print(f"[2/3] 文本分块")
        
        all_chunks = []
        for doc in documents:
            chunks = self.chunker.split(doc, strategy="paragraph")
            all_chunks.extend(chunks)
        
        print(f"  生成了 {len(all_chunks)} 个文本块")
        
        print(f"[3/3] 建立索引")
        
        self.hybrid_retriever.index(all_chunks)
        
        print(f"  索引完成！")
    
    def query(self, question: str, ground_truth: Optional[str] = None) -> str:
        """
        查询问答（带可观测性记录）
        
        参数：
            question: 用户问题
            ground_truth: 参考答案（用于评估）
        
        返回：
            str: LLM生成的答案
        """
        # 开始Trace
        trace = self.observer.start_trace(
            name="rag_query",
            metadata={"question": question}
        )
        
        # 记录开始时间
        start_time = time.time()
        
        # 步骤1：检索
        retrieval_span = trace.add_span("retrieval")
        retrieval_span.start()
        
        retrieval_start = time.time()
        results = self.hybrid_retriever.search(question, top_k=3)
        retrieval_time = (time.time() - retrieval_start) * 1000
        
        retrieval_span.end()
        self.metrics.record_retrieval_time(retrieval_time)
        
        # 步骤2：构建上下文
        context = self._build_context(results)
        
        # 步骤3：LLM调用
        llm_span = trace.add_span("llm_call")
        llm_span.start()
        
        prompt = self._build_prompt(question, context)
        messages = [
            {"role": "system", "content": "你是一个基于文档知识的问答助手。"},
            {"role": "user", "content": prompt}
        ]
        
        llm_start = time.time()
        answer = self.llm.chat(messages)
        llm_time = (time.time() - llm_start) * 1000
        
        llm_span.end()
        self.metrics.record_llm_time(llm_time)
        
        # 记录总时间
        total_time = (time.time() - start_time) * 1000
        self.metrics.record_total_time(total_time)
        
        # 记录查询计数
        self.metrics.increment_query_count()
        
        # 结束Trace
        trace.end()
        
        # 收集评估样本
        if ground_truth:
            sample = EvaluationSample(
                question=question,
                contexts=[r.chunk.content for r in results],
                answer=answer,
                ground_truth=ground_truth
            )
            self.evaluation_samples.append(sample)
        
        # 打印Trace摘要
        print(f"\n{self.observer.get_trace_summary(trace)}")
        
        return answer
    
    def _build_context(self, results) -> str:
        """构建上下文字符串"""
        context_parts = []
        for i, result in enumerate(results):
            context_parts.append(f"[文档{i+1}]\n{result.chunk.content}")
        return "\n\n".join(context_parts)
    
    def _build_prompt(self, question: str, context: str) -> str:
        """构建RAG Prompt"""
        return f"""请根据以下上下文回答问题。

上下文：
{context}

问题：{question}

答案："""
    
    def generate_report(self) -> str:
        """
        生成综合报告
        
        返回：
            str: 包含指标和评估结果的综合报告
        """
        report = "=" * 60 + "\n"
        report += "RAG系统综合报告\n"
        report += "=" * 60 + "\n\n"
        
        # 指标报告
        metrics_report = self.metrics.generate_report()
        report += "性能指标：\n"
        report += f"  查询总数：{metrics_report['query_count']}\n"
        report += f"  平均检索时间：{metrics_report['avg_retrieval_time_ms']:.2f}ms\n"
        report += f"  平均LLM时间：{metrics_report['avg_llm_time_ms']:.2f}ms\n"
        report += f"  平均总时间：{metrics_report['avg_total_time_ms']:.2f}ms\n"
        report += f"  平均输入Token：{metrics_report['avg_prompt_tokens']:.0f}\n"
        report += f"  平均输出Token：{metrics_report['avg_completion_tokens']:.0f}\n"
        report += f"  总成本：${metrics_report['total_cost']:.4f}\n\n"
        
        # 评估报告
        if self.evaluation_samples:
            scores = self.evaluator.evaluate_batch(self.evaluation_samples)
            report += self.evaluator.generate_report(scores)
        
        return report


# ============================================================
# 第六部分：演示函数
# ============================================================

def demonstrate_observability():
    """
    演示可观测性系统
    
    功能说明：
        展示完整的可观测性工作流程。
    """
    print("=" * 60)
    print("可观测性RAG系统演示")
    print("=" * 60)
    
    # 创建示例文档
    sample_dir = os.path.join(os.path.dirname(__file__), "..", "sample_docs")
    os.makedirs(sample_dir, exist_ok=True)
    
    sample_content = """# Git版本控制指南

## 初始化仓库
```bash
git init
```

## 添加文件
```bash
git add .
git commit -m "Initial commit"
```

## 查看状态
```bash
git status
```

## 查看历史
```bash
git log
git log --oneline
```

## 分支管理
创建分支：
```bash
git branch feature-1
git checkout feature-1
```

合并分支：
```bash
git checkout main
git merge feature-1
```
"""
    
    with open(os.path.join(sample_dir, "git_guide.txt"), 'w', encoding='utf-8') as f:
        f.write(sample_content)
    
    print(f"\n示例文档已创建：{sample_dir}\n")
    
    # 初始化系统
    rag = ObservableRAGSystem(collection_name="demo_observable")
    
    # 索引文档
    rag.index_documents(sample_dir, source_type="directory")
    
    # 执行查询
    questions = [
        ("如何初始化Git仓库？", "使用 git init 命令初始化仓库"),
        ("如何创建和切换分支？", "使用 git branch 创建分支，git checkout 切换分支"),
    ]
    
    for question, ground_truth in questions:
        print(f"\n{'=' * 60}")
        print(f"问题：{question}")
        print(f"{'=' * 60}")
        answer = rag.query(question, ground_truth=ground_truth)
        print(f"\n答案：{answer}")
    
    # 生成报告
    print("\n")
    print(rag.generate_report())


if __name__ == "__main__":
    """
    主程序入口
    
    运行方式：
        python observability_evaluation.py
    """
    demonstrate_observability()
