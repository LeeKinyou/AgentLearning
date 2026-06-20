"""
Day 5-7: 鲁棒性设计与高级调试

本模块演示生产级Agent的鲁棒性设计和调试技巧，包括：
1. 常见故障模式与应对策略
2. LLM幻觉检测与缓解
3. Tool调用失败重试策略
4. 超时与降级处理
5. 调试技巧：LangFuse Trace、日志分析
6. 单元测试与集成测试

学习目标：
- 理解生产环境常见故障模式
- 掌握重试、超时、降级等鲁棒性设计
- 学会使用LangFuse进行调试
- 编写全面的测试用例

依赖安装：
    pip install langfuse tenacity pytest
"""

import os
import sys
import time
import json
import logging
import functools
from typing import Dict, List, Optional, Any, Callable
from datetime import datetime
from enum import Enum
from dotenv import load_dotenv

# 加载环境变量
load_dotenv()

# 添加项目根目录到路径
sys.path.insert(0, os.path.join(os.path.dirname(__file__), '..', '..'))


# ============================================================
# 日志配置
# ============================================================

def setup_logging():
    """配置日志系统"""
    logging.basicConfig(
        level=logging.INFO,
        format='%(asctime)s - %(name)s - %(levelname)s - %(message)s',
        handlers=[
            logging.FileHandler('agent.log', encoding='utf-8'),
            logging.StreamHandler()
        ]
    )
    return logging.getLogger("AgentService")


logger = setup_logging()

# 模块启动时间，用于计算运行时长
start_time: float = time.time()


# ============================================================
# 重试策略
# ============================================================

class RetryStrategy:
    """重试策略"""
    
    @staticmethod
    def with_exponential_backoff(
        max_retries: int = 3,
        base_delay: float = 1.0,
        max_delay: float = 60.0,
        exceptions: tuple = (Exception,)
    ):
        """
        指数退避重试装饰器
        
        参数：
            max_retries: 最大重试次数
            base_delay: 基础延迟（秒）
            max_delay: 最大延迟（秒）
            exceptions: 需要重试的异常类型
        """
        def decorator(func: Callable) -> Callable:
            @functools.wraps(func)
            def wrapper(*args, **kwargs):
                for attempt in range(max_retries + 1):
                    try:
                        return func(*args, **kwargs)
                    except exceptions as e:
                        if attempt == max_retries:
                            logger.error(f"函数 {func.__name__} 在 {max_retries} 次重试后仍然失败: {str(e)}")
                            raise
                        
                        # 计算延迟时间（指数退避）
                        delay = min(base_delay * (2 ** attempt), max_delay)
                        logger.warning(
                            f"函数 {func.__name__} 第 {attempt + 1} 次尝试失败: {str(e)}. "
                            f"{delay:.1f}秒后重试..."
                        )
                        time.sleep(delay)
            
            return wrapper
        return decorator
    
    @staticmethod
    def with_circuit_breaker(
        failure_threshold: int = 5,
        recovery_timeout: float = 60.0
    ):
        """
        断路器装饰器
        
        参数：
            failure_threshold: 失败阈值
            recovery_timeout: 恢复超时（秒）
        """
        def decorator(func: Callable) -> Callable:
            failures = 0
            last_failure_time = None
            
            @functools.wraps(func)
            def wrapper(*args, **kwargs):
                nonlocal failures, last_failure_time
                
                # 检查断路器状态
                if failures >= failure_threshold:
                    if last_failure_time and (time.time() - last_failure_time) < recovery_timeout:
                        raise Exception(f"断路器打开: 服务 {func.__name__} 暂时不可用")
                    else:
                        # 尝试恢复
                        logger.info(f"断路器半开: 尝试恢复服务 {func.__name__}")
                        failures = 0
                
                try:
                    result = func(*args, **kwargs)
                    # 成功则重置失败计数
                    if failures > 0:
                        logger.info(f"服务 {func.__name__} 恢复成功")
                        failures = 0
                    return result
                    
                except Exception as e:
                    failures += 1
                    last_failure_time = time.time()
                    logger.error(f"服务 {func.__name__} 失败 ({failures}/{failure_threshold}): {str(e)}")
                    raise
            
            return wrapper
        return decorator


# ============================================================
# 超时处理
# ============================================================

class TimeoutHandler:
    """超时处理器"""
    
    @staticmethod
    def with_timeout(timeout_seconds: float):
        """
        超时装饰器（同步版本）
        
        参数：
            timeout_seconds: 超时时间（秒）
        """
        def decorator(func: Callable) -> Callable:
            @functools.wraps(func)
            def wrapper(*args, **kwargs):
                import signal
                import sys

                if sys.platform != "win32" and hasattr(signal, 'SIGALRM'):
                    def handler(signum, frame):
                        raise TimeoutError(f"函数 {func.__name__} 执行超时（{timeout_seconds}秒）")

                    # 设置信号处理器
                    old_handler = signal.signal(signal.SIGALRM, handler)
                    signal.alarm(int(timeout_seconds))

                    try:
                        result = func(*args, **kwargs)
                        return result
                    finally:
                        signal.alarm(0)
                        signal.signal(signal.SIGALRM, old_handler)
                else:
                    return func(*args, **kwargs)

            return wrapper
        return decorator
    
    @staticmethod
    async def async_with_timeout(coroutine, timeout_seconds: float):
        """
        异步超时处理
        
        参数：
            coroutine: 协程对象
            timeout_seconds: 超时时间（秒）
        """
        import asyncio
        try:
            return await asyncio.wait_for(coroutine, timeout=timeout_seconds)
        except asyncio.TimeoutError:
            raise TimeoutError(f"协程执行超时（{timeout_seconds}秒）")


# ============================================================
# 降级策略
# ============================================================

class FallbackStrategy:
    """降级策略"""
    
    @staticmethod
    def with_fallback(fallback_func: Callable):
        """
        降级装饰器
        
        参数：
            fallback_func: 降级时调用的函数
        """
        def decorator(func: Callable) -> Callable:
            @functools.wraps(func)
            def wrapper(*args, **kwargs):
                try:
                    return func(*args, **kwargs)
                except Exception as e:
                    logger.warning(f"主函数 {func.__name__} 失败，执行降级: {str(e)}")
                    return fallback_func(*args, **kwargs)
            
            return wrapper
        return decorator
    
    @staticmethod
    def simple_fallback_response(query: str) -> str:
        """简单降级响应"""
        return (
            f"抱歉，系统暂时无法处理您的请求: '{query[:50]}...'\n"
            "请稍后重试，或联系技术支持。"
        )


# ============================================================
# 幻觉检测
# ============================================================

class HallucinationDetector:
    """幻觉检测器"""
    
    def __init__(self):
        self.confidence_threshold = 0.7
        self.fact_check_patterns = [
            "不确定", "可能", "也许", "大概", "应该",
            "我不确定", "我无法确认", "据我所知"
        ]
    
    def detect(self, response: str, context: Optional[str] = None) -> Dict[str, Any]:
        """
        检测响应中可能的幻觉
        
        参数：
            response: LLM响应
            context: 上下文信息
        
        返回：
            Dict: 检测结果
        """
        issues = []
        confidence = 1.0
        
        # 检查不确定性表达
        for pattern in self.fact_check_patterns:
            if pattern in response.lower():
                issues.append(f"发现不确定性表达: '{pattern}'")
                confidence -= 0.1
        
        # 检查具体数据（数字、日期等）
        import re
        numbers = re.findall(r'\d+\.?\d*', response)
        if len(numbers) > 5:
            issues.append("响应包含大量数字，建议核实数据来源")
            confidence -= 0.1
        
        # 检查响应长度
        if len(response) < 20:
            issues.append("响应过短，可能不完整")
            confidence -= 0.2
        
        # 检查上下文一致性
        if context and len(context) > 0:
            # 简单检查：响应是否包含上下文中的关键词
            context_words = set(context.lower().split())
            response_words = set(response.lower().split())
            overlap = len(context_words & response_words) / len(context_words)
            if overlap < 0.1:
                issues.append("响应与上下文关联度低")
                confidence -= 0.15
        
        confidence = max(0.0, confidence)
        
        return {
            "has_hallucination_risk": confidence < self.confidence_threshold,
            "confidence_score": confidence,
            "issues": issues,
            "recommendation": "建议人工审核" if confidence < self.confidence_threshold else "通过"
        }
    
    def mitigate(self, response: str, detection_result: Dict[str, Any]) -> str:
        """
        缓解幻觉
        
        参数：
            response: 原始响应
            detection_result: 检测结果
        
        返回：
            str: 缓解后的响应
        """
        if not detection_result["has_hallucination_risk"]:
            return response
        
        # 添加免责声明
        disclaimer = (
            "\n\n[系统提示] 以上内容可能包含不准确信息，"
            "建议核实关键数据和事实。"
        )
        
        return response + disclaimer


# ============================================================
# LangFuse可观测性
# ============================================================

class ObservabilityManager:
    """可观测性管理器"""
    
    def __init__(self):
        self.enabled = False
        self.langfuse = None
        self._init_langfuse()
    
    def _init_langfuse(self):
        """初始化LangFuse"""
        try:
            from langfuse import Langfuse
            
            public_key = os.getenv("LANGFUSE_PUBLIC_KEY")
            secret_key = os.getenv("LANGFUSE_SECRET_KEY")
            host = os.getenv("LANGFUSE_HOST", "http://localhost:3000")
            
            if public_key and secret_key:
                self.langfuse = Langfuse(
                    public_key=public_key,
                    secret_key=secret_key,
                    host=host
                )
                self.enabled = True
                logger.info("LangFuse初始化成功")
            else:
                logger.warning("LangFuse密钥未配置，可观测性功能已禁用")
                
        except ImportError:
            logger.warning("LangFuse未安装，可观测性功能已禁用")
    
    def trace_llm_call(
        self, 
        trace_id: str,
        model: str,
        prompt: str,
        response: str,
        latency_ms: float,
        tokens_used: int,
        metadata: Optional[Dict] = None
    ):
        """
        追踪LLM调用
        
        参数：
            trace_id: 追踪ID
            model: 模型名称
            prompt: 提示词
            response: 响应
            latency_ms: 延迟（毫秒）
            tokens_used: Token使用量
            metadata: 元数据
        """
        if not self.enabled:
            return
        
        try:
            trace = self.langfuse.trace(  # type: ignore[union-attr]
                id=trace_id,
                name="llm_call",
                metadata=metadata or {}
            )
            
            trace.generation(
                name="completion",
                model=model,
                input=prompt,
                output=response,
                usage={
                    "input": tokens_used // 2,
                    "output": tokens_used // 2,
                    "total": tokens_used
                },
                metadata={
                    "latency_ms": latency_ms
                }
            )
            
            logger.info(f"Trace已记录: {trace_id}")
            
        except Exception as e:
            logger.error(f"记录Trace失败: {str(e)}")
    
    def trace_tool_call(
        self,
        trace_id: str,
        tool_name: str,
        parameters: Dict,
        result: Any,
        latency_ms: float,
        success: bool
    ):
        """追踪工具调用"""
        if not self.enabled:
            return
        
        try:
            trace = self.langfuse.trace(id=trace_id, name="tool_call")  # type: ignore[union-attr]
            trace.span(
                name=tool_name,
                input=parameters,
                output=result,
                metadata={
                    "latency_ms": latency_ms,
                    "success": success
                }
            )
        except Exception as e:
            logger.error(f"记录Tool Trace失败: {str(e)}")


# ============================================================
# 鲁棒性Agent服务
# ============================================================

class RobustAgentService:
    """鲁棒性Agent服务"""
    
    def __init__(self):
        self.hallucination_detector = HallucinationDetector()
        self.observability = ObservabilityManager()
        self.request_count = 0
        self.error_count = 0
    
    @RetryStrategy.with_exponential_backoff(
        max_retries=3,
        base_delay=1.0,
        exceptions=(Exception,)
    )
    def call_llm_with_retry(self, prompt: str) -> str:
        """带重试的LLM调用"""
        # 模拟LLM调用
        import random
        if random.random() < 0.3:  # 30%概率失败
            raise Exception("LLM服务暂时不可用")
        return f"响应: {prompt[:50]}..."
    
    @FallbackStrategy.with_fallback(FallbackStrategy.simple_fallback_response)
    def process_with_fallback(self, query: str) -> str:
        """带降级的处理"""
        return self.call_llm_with_retry(query)
    
    def process_query(self, query: str) -> Dict[str, Any]:
        """
        处理查询（完整流程）
        
        流程：
        1. 记录请求
        2. 调用LLM（带重试）
        3. 幻觉检测
        4. 缓解处理
        5. 记录可观测性数据
        """
        self.request_count += 1
        trace_id = f"trace_{datetime.now().strftime('%Y%m%d_%H%M%S')}_{self.request_count}"
        start_time = time.time()
        
        try:
            # 调用LLM
            response = self.process_with_fallback(query)
            
            # 幻觉检测
            detection_result = self.hallucination_detector.detect(response, query)
            
            # 缓解处理
            final_response = self.hallucination_detector.mitigate(response, detection_result)
            
            # 计算延迟
            latency_ms = (time.time() - start_time) * 1000
            
            # 记录可观测性
            self.observability.trace_llm_call(
                trace_id=trace_id,
                model="gpt-3.5-turbo",
                prompt=query,
                response=final_response,
                latency_ms=latency_ms,
                tokens_used=len(query.split()) + len(final_response.split())
            )
            
            return {
                "success": True,
                "response": final_response,
                "trace_id": trace_id,
                "latency_ms": latency_ms,
                "hallucination_check": detection_result
            }
            
        except Exception as e:
            self.error_count += 1
            logger.error(f"处理查询失败: {str(e)}")
            return {
                "success": False,
                "error": str(e),
                "trace_id": trace_id,
                "error_rate": self.error_count / self.request_count
            }
    
    def get_stats(self) -> Dict[str, Any]:
        """获取服务统计"""
        return {
            "total_requests": self.request_count,
            "total_errors": self.error_count,
            "error_rate": self.error_count / self.request_count if self.request_count > 0 else 0,
            "uptime_seconds": time.time() - start_time if 'start_time' in globals() else 0
        }


# ============================================================
# 演示函数
# ============================================================

def demonstrate_retry_strategy():
    """演示重试策略"""
    print("=" * 60)
    print("重试策略演示")
    print("=" * 60)
    
    @RetryStrategy.with_exponential_backoff(max_retries=3, base_delay=0.5)
    def unstable_function():
        """不稳定的函数"""
        import random
        if random.random() < 0.7:
            raise Exception("随机失败")
        return "成功！"
    
    try:
        result = unstable_function()
        print(f"结果: {result}")
    except Exception as e:
        print(f"最终失败: {str(e)}")


def demonstrate_circuit_breaker():
    """演示断路器"""
    print("\n" + "=" * 60)
    print("断路器演示")
    print("=" * 60)
    
    @RetryStrategy.with_circuit_breaker(failure_threshold=3, recovery_timeout=2)
    def failing_function():
        """总是失败的函数"""
        raise Exception("服务故障")
    
    for i in range(5):
        try:
            result = failing_function()
            print(f"尝试 {i+1}: 成功")
        except Exception as e:
            print(f"尝试 {i+1}: {str(e)}")
        time.sleep(0.5)


def demonstrate_hallucination_detection():
    """演示幻觉检测"""
    print("\n" + "=" * 60)
    print("幻觉检测演示")
    print("=" * 60)
    
    detector = HallucinationDetector()
    
    test_responses = [
        "Python是一种编程语言，由Guido van Rossum于1991年创建。",
        "我不太确定，但Python可能是一种编程语言吧。",
        "根据我的分析，答案是42。",
        "不确定"
    ]
    
    for response in test_responses:
        result = detector.detect(response)
        print(f"\n响应: {response[:50]}...")
        print(f"  风险: {'是' if result['has_hallucination_risk'] else '否'}")
        print(f"  置信度: {result['confidence_score']:.2f}")
        print(f"  问题: {result['issues']}")


def demonstrate_robust_agent():
    """演示鲁棒性Agent"""
    print("\n" + "=" * 60)
    print("鲁棒性Agent演示")
    print("=" * 60)
    
    agent = RobustAgentService()
    
    queries = [
        "什么是Python？",
        "解释量子计算",
        "如何学习机器学习？"
    ]
    
    for query in queries:
        print(f"\n查询: {query}")
        result = agent.process_query(query)
        print(f"  成功: {result['success']}")
        if result['success']:
            print(f"  响应: {result['response'][:100]}...")
            print(f"  延迟: {result['latency_ms']:.2f}ms")
        else:
            print(f"  错误: {result['error']}")
    
    # 显示统计
    stats = agent.get_stats()
    print(f"\n服务统计:")
    print(f"  总请求: {stats['total_requests']}")
    print(f"  总错误: {stats['total_errors']}")
    print(f"  错误率: {stats['error_rate']:.2%}")


# ============================================================
# 主函数
# ============================================================

def main():
    """主函数"""
    print("=" * 60)
    print("Week05 Day 5-7: 鲁棒性设计与高级调试")
    print("=" * 60)
    
    demonstrate_retry_strategy()
    demonstrate_circuit_breaker()
    demonstrate_hallucination_detection()
    demonstrate_robust_agent()
    
    print("\n" + "=" * 60)
    print("演示完成！")
    print("=" * 60)


if __name__ == "__main__":
    main()
