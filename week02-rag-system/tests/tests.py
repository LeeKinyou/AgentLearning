"""
Week02 测试套件

本模块包含Week02所有核心功能的测试用例：
1. 基础RAG流水线测试
2. 进阶RAG策略测试
3. 可观测性与评估测试

运行方式：
    python tests.py
"""

import os
import sys
import tempfile
import shutil

# 添加项目路径
sys.path.insert(0, os.path.join(os.path.dirname(__file__), '..'))

from day01_02_basic_rag.basic_rag_pipeline import (
    Document, Chunk, SearchResult,
    DocumentLoader, TextChunker, VectorStore,
    BasicRAGSystem
)
from day03_04_advanced_rag.advanced_rag_strategies import (
    BM25Retriever, HybridRetriever,
    Reranker, QueryRewriter,
    AdvancedRAGSystem
)
from day05_07_observability.observability_evaluation import (
    TraceSpan, Trace, MetricsCollector,
    LangFuseObserver, EvaluationSample,
    RagasEvaluator, ObservableRAGSystem
)


def run_test(test_name: str, test_func):
    """
    运行单个测试
    
    参数：
        test_name: 测试名称
        test_func: 测试函数
    """
    print(f"\n{'=' * 60}")
    print(f"测试：{test_name}")
    print(f"{'=' * 60}")
    
    try:
        test_func()
        print(f"[PASS] {test_name}测试通过")
    except AssertionError as e:
        print(f"[FAIL] {test_name}测试失败：{e}")
    except Exception as e:
        print(f"[ERROR] {test_name}测试异常：{e}")


def test_document_model():
    """测试Document数据模型"""
    doc = Document(
        content="测试内容",
        metadata={"source": "test.txt"},
        doc_id="doc_001"
    )
    
    assert doc.content == "测试内容", "内容不正确"
    assert doc.metadata is not None and doc.metadata["source"] == "test.txt", "元数据不正确"
    assert doc.doc_id == "doc_001", "文档ID不正确"
    
    print(f"[PASS] Document模型创建成功：{doc.doc_id}")
    
    # 测试默认值
    doc2 = Document(content="测试内容2")
    assert doc2.metadata == {}, "默认元数据应为空字典"
    assert doc2.doc_id.startswith("doc_"), "默认文档ID格式不正确"
    
    print(f"[PASS] Document默认值正确")


def test_chunk_model():
    """测试Chunk数据模型"""
    chunk = Chunk(
        content="文本块内容",
        metadata={"chunk_index": 0},
        chunk_id="chunk_001",
        doc_id="doc_001"
    )
    
    assert chunk.content == "文本块内容", "内容不正确"
    assert chunk.chunk_id == "chunk_001", "块ID不正确"
    assert chunk.doc_id == "doc_001", "文档ID不正确"
    
    print(f"[PASS] Chunk模型创建成功：{chunk.chunk_id}")


def test_document_loader():
    """测试文档加载器"""
    # 创建临时目录和文件
    temp_dir = tempfile.mkdtemp()
    test_file = os.path.join(temp_dir, "test.txt")
    
    try:
        with open(test_file, 'w', encoding='utf-8') as f:
            f.write("测试文档内容")
        
        loader = DocumentLoader()
        
        # 测试单文件加载
        docs = loader.load_from_file(test_file)
        assert len(docs) == 1, "应加载1个文档"
        assert docs[0].content == "测试文档内容", "内容不正确"
        
        print(f"[PASS] 单文件加载成功")
        
        # 测试目录加载
        docs = loader.load_from_directory(temp_dir)
        assert len(docs) >= 1, "应至少加载1个文档"
        
        print(f"[PASS] 目录加载成功：{len(docs)}个文档")
        
    finally:
        shutil.rmtree(temp_dir)


def test_text_chunker():
    """测试文本分块器"""
    # 创建测试文档
    doc = Document(
        content="第一段内容。\n\n第二段内容。\n\n第三段内容。",
        metadata={"source": "test.txt"},
        doc_id="doc_001"
    )
    
    chunker = TextChunker(chunk_size=50, chunk_overlap=10)
    
    # 测试段落分割
    chunks = chunker.split(doc, strategy="paragraph")
    assert len(chunks) > 0, "应生成分块"
    
    print(f"[PASS] 段落分割成功：{len(chunks)}个块")
    
    # 测试固定长度分割
    chunks = chunker.split(doc, strategy="fixed")
    assert len(chunks) > 0, "应生成分块"
    
    print(f"[PASS] 固定长度分割成功：{len(chunks)}个块")


def test_vector_store():
    """测试向量数据库"""
    store = VectorStore(collection_name="test_store")
    
    # 创建测试块
    chunks = [
        Chunk(content="Python是一种编程语言", chunk_id="chunk_001"),
        Chunk(content="Java也是一种编程语言", chunk_id="chunk_002"),
    ]
    
    # 添加块
    store.add_chunks(chunks)
    print(f"[PASS] 向量存储成功")
    
    # 搜索
    results = store.search("Python", top_k=1)
    assert len(results) > 0, "应返回搜索结果"
    assert "Python" in results[0].chunk.content, "结果应包含Python"
    
    print(f"[PASS] 向量搜索成功：相似度={results[0].score:.3f}")
    
    # 清理
    store.clear()


def test_bm25_retriever():
    """测试BM25检索器"""
    retriever = BM25Retriever()
    
    # 添加文档（使用英文内容以匹配BM25的分词器）
    chunks = [
        Chunk(content="Python installation tutorial: how to install Python environment", chunk_id="chunk_001"),
        Chunk(content="Java installation guide: configure Java development environment", chunk_id="chunk_002"),
        Chunk(content="Python virtual environment management best practices", chunk_id="chunk_003"),
    ]
    
    retriever.add_documents(chunks)
    print(f"[PASS] BM25索引建立成功")
    
    # 搜索（使用英文关键词）
    results = retriever.search("Python installation", top_k=2)
    assert len(results) > 0, "应返回搜索结果"
    
    print(f"[PASS] BM25搜索成功：{len(results)}个结果")


def test_hybrid_retriever():
    """测试混合检索器"""
    retriever = HybridRetriever()
    
    # 添加文档（使用英文内容）
    chunks = [
        Chunk(content="Python installation tutorial: how to install Python environment", chunk_id="chunk_001"),
        Chunk(content="Java installation guide: configure Java development environment", chunk_id="chunk_002"),
    ]
    
    retriever.index(chunks, collection_name="test_hybrid")
    print(f"[PASS] 混合索引建立成功")
    
    # 搜索（使用英文关键词）
    results = retriever.search("Python installation", top_k=1)
    assert len(results) > 0, "应返回搜索结果"
    
    print(f"[PASS] 混合搜索成功：{len(results)}个结果")


def test_trace_span():
    """测试TraceSpan模型"""
    span = TraceSpan(name="test_span")
    
    # 测试计时
    span.start()
    import time
    time.sleep(0.1)  # 等待100ms
    span.end()
    
    assert span.status == "success", "状态应为success"
    assert span.duration_ms is not None and span.duration_ms > 0, "持续时间应大于0"
    assert span.duration_ms is not None and span.duration_ms >= 90, "持续时间应接近100ms"  # 允许误差
    
    print(f"[PASS] TraceSpan计时成功：{span.duration_ms:.2f}ms")


def test_trace():
    """测试Trace模型"""
    trace = Trace(trace_id="test_001", name="test_trace")
    
    # 添加Span
    span1 = trace.add_span("retrieval")
    span2 = trace.add_span("llm_call")
    
    assert trace.spans is not None and len(trace.spans) == 2, "应有2个Span"

    print(f"[PASS] Trace创建成功：{len(trace.spans) if trace.spans else 0}个Span")
    
    # 测试计时
    trace.start()
    import time
    time.sleep(0.05)
    trace.end()
    
    assert trace.metadata is not None and trace.metadata['total_duration_ms'] > 0, "总时间应大于0"

    print(f"[PASS] Trace计时成功：{trace.metadata['total_duration_ms']:.2f}ms")


def test_metrics_collector():
    """测试指标收集器"""
    collector = MetricsCollector()
    
    # 记录指标
    collector.record_retrieval_time(100.0)
    collector.record_retrieval_time(150.0)
    collector.record_llm_time(500.0)
    collector.record_total_time(600.0)
    collector.record_tokens(prompt=100, completion=200)
    collector.increment_query_count()
    
    # 生成报告
    report = collector.generate_report()
    
    assert report['query_count'] == 1, "查询计数应为1"
    assert report['avg_retrieval_time_ms'] == 125.0, "平均检索时间应为125ms"
    assert report['avg_prompt_tokens'] == 100.0, "平均输入Token应为100"
    
    print(f"[PASS] 指标收集成功：{report}")


def test_evaluation_sample():
    """测试评估样本"""
    sample = EvaluationSample(
        question="如何安装Python？",
        contexts=["Python安装教程..."],
        answer="访问Python官网下载安装包",
        ground_truth="从python.org下载"
    )
    
    assert sample.question == "如何安装Python？", "问题不正确"
    assert len(sample.contexts) == 1, "上下文数量不正确"
    
    print(f"[PASS] 评估样本创建成功")


def test_ragas_evaluator():
    """测试Ragas评估器"""
    evaluator = RagasEvaluator()
    
    # 创建测试样本
    sample = EvaluationSample(
        question="Python安装",
        contexts=["Python安装教程：访问官网下载"],
        answer="访问Python官网下载安装包",
        ground_truth="从python.org下载"
    )
    
    # 评估
    scores = evaluator.evaluate_sample(sample)
    
    assert 'faithfulness' in scores, "应包含faithfulness指标"
    assert 'answer_relevance' in scores, "应包含answer_relevance指标"
    assert 'context_precision' in scores, "应包含context_precision指标"
    
    print(f"[PASS] 评估成功：{scores}")


def test_langfuse_observer():
    """测试LangFuse观察者"""
    observer = LangFuseObserver()
    
    # 创建Trace
    trace = observer.start_trace("test_trace")
    span = trace.add_span("test_span")
    
    span.start()
    import time
    time.sleep(0.05)
    span.end()
    
    trace.end()
    
    # 获取摘要
    summary = observer.get_trace_summary(trace)
    assert "test_trace" in summary, "摘要应包含Trace名称"
    
    print(f"[PASS] LangFuse观察者工作正常")


def main():
    """主测试函数"""
    print("=" * 60)
    print("Week02 测试套件")
    print("=" * 60)
    
    tests = [
        ("Document模型", test_document_model),
        ("Chunk模型", test_chunk_model),
        ("文档加载器", test_document_loader),
        ("文本分块器", test_text_chunker),
        ("向量数据库", test_vector_store),
        ("BM25检索器", test_bm25_retriever),
        ("混合检索器", test_hybrid_retriever),
        ("TraceSpan", test_trace_span),
        ("Trace", test_trace),
        ("指标收集器", test_metrics_collector),
        ("评估样本", test_evaluation_sample),
        ("Ragas评估器", test_ragas_evaluator),
        ("LangFuse观察者", test_langfuse_observer),
    ]
    
    passed = 0
    failed = 0
    
    for name, func in tests:
        try:
            func()
            passed += 1
        except Exception as e:
            print(f"[FAIL] {name}测试失败：{e}")
            failed += 1
    
    # 打印总结
    print(f"\n{'=' * 60}")
    print("测试总结")
    print(f"{'=' * 60}")
    print(f"总测试数：{len(tests)}")
    print(f"通过：{passed}")
    print(f"失败：{failed}")
    print(f"通过率：{passed/len(tests)*100:.1f}%")
    print(f"{'=' * 60}")


if __name__ == "__main__":
    main()
