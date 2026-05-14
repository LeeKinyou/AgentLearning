"""
Day 3-4: 进阶RAG策略实现

本模块实现三种进阶RAG策略，提升检索质量和答案准确性：
1. 混合检索（BM25 + 向量检索）
2. 重排序（Reranking）
3. 查询重写与扩展

学习目标：
- 理解不同检索策略的优缺点
- 掌握混合检索的实现方法
- 学会使用重排序提升结果相关性
- 能够通过查询重写改善检索效果

依赖安装：
    pip install chromadb openai pydantic python-dotenv rank-bm25 jieba
"""

import os
import re
import sys
from typing import List, Dict, Optional, Tuple
from dataclasses import dataclass
from dotenv import load_dotenv

# 加载环境变量
load_dotenv()

# 添加项目根目录到路径，支持跨模块导入
sys.path.insert(0, os.path.join(os.path.dirname(__file__), '..'))

# 导入基础RAG模块的数据模型
from day01_02_basic_rag.basic_rag_pipeline import Document, Chunk, SearchResult, VectorStore, LLMClient


# ============================================================
# 第一部分：BM25检索器
# ============================================================

class BM25Retriever:
    """
    BM25检索器
    
    功能说明：
        实现基于BM25算法的关键词检索。
        BM25是一种经典的文本相关性排序算法，广泛用于搜索引擎。
    
    算法原理：
        BM25通过计算查询词与文档的词频（TF）和逆文档频率（IDF）
        来评估相关性。相比简单的词频统计，BM25考虑了：
        - 词频饱和：避免高频词过度影响
        - 文档长度归一化：避免长文档优势
        - 逆文档频率：降低常见词的权重
    
    使用示例：
        >>> retriever = BM25Retriever()
        >>> retriever.add_documents(chunks)
        >>> results = retriever.search("Python安装", top_k=5)
    """
    
    def __init__(self, k1: float = 1.5, b: float = 0.75):
        """
        初始化BM25检索器
        
        参数：
            k1: 词频饱和参数（默认1.5）
                - 值越大，词频影响越大
                - 通常范围：1.2-2.0
            b: 文档长度归一化参数（默认0.75）
                - 0表示不考虑文档长度
                - 1表示完全归一化
                - 通常范围：0.5-1.0
        """
        self.k1 = k1
        self.b = b
        self.documents = []  # 存储所有文档
        self.doc_freqs = {}  # 文档频率：词 -> 包含该词的文档数
        self.avgdl = 0  # 平均文档长度
        self.idf = {}  # 逆文档频率：词 -> IDF值
    
    def _tokenize(self, text: str) -> List[str]:
        """
        文本分词
        
        参数：
            text: 要分词的文本
        
        返回：
            List[str]: 分词后的词列表
        
        功能说明：
            简单的分词实现，按空格和标点符号分割。
            对于中文，建议使用jieba等专门的分词库。
        """
        # 转换为小写
        text = text.lower()
        # 按非字母数字字符分割
        tokens = re.findall(r'\b\w+\b', text)
        return tokens
    
    def add_documents(self, chunks: List[Chunk]):
        """
        添加文档到索引
        
        参数：
            chunks: 要添加的文本块列表
        
        功能说明：
            1. 对每个文档进行分词
            2. 计算文档频率（DF）
            3. 计算平均文档长度
            4. 计算逆文档频率（IDF）
        """
        self.documents = []
        self.doc_freqs = {}
        
        # 遍历所有文档
        for chunk in chunks:
            tokens = self._tokenize(chunk.content)
            self.documents.append({
                'chunk': chunk,
                'tokens': tokens,
                'length': len(tokens)
            })
            
            # 更新文档频率
            unique_tokens = set(tokens)
            for token in unique_tokens:
                if token not in self.doc_freqs:
                    self.doc_freqs[token] = 0
                self.doc_freqs[token] += 1
        
        # 计算平均文档长度
        if self.documents:
            total_length = sum(doc['length'] for doc in self.documents)
            self.avgdl = total_length / len(self.documents)
        
        # 计算IDF
        N = len(self.documents)
        for token, df in self.doc_freqs.items():
            # IDF公式：log((N - df + 0.5) / (df + 0.5) + 1)
            self.idf[token] = (
                (N - df + 0.5) / (df + 0.5) + 1
            )
    
    def search(self, query: str, top_k: int = 5) -> List[SearchResult]:
        """
        BM25检索
        
        参数：
            query: 查询文本
            top_k: 返回最相关的k个结果
        
        返回：
            List[SearchResult]: 搜索结果列表，按相关性降序排列
        
        算法步骤：
            1. 对查询进行分词
            2. 对每个文档计算BM25分数
            3. 按分数排序返回top_k个结果
        """
        query_tokens = self._tokenize(query)
        scores = []
        
        # 对每个文档计算BM25分数
        for doc in self.documents:
            score = 0.0
            
            # 对每个查询词计算贡献
            for token in query_tokens:
                if token not in self.idf:
                    continue
                
                # 词频（TF）
                tf = doc['tokens'].count(token)
                
                # 文档长度归一化
                dl = doc['length']
                norm = 1 - self.b + self.b * (dl / self.avgdl)
                
                # BM25分数公式
                term_score = (
                    self.idf[token] * 
                    (tf * (self.k1 + 1)) / 
                    (tf + self.k1 * norm)
                )
                
                score += term_score
            
            scores.append(score)
        
        # 按分数排序
        indexed_scores = list(enumerate(scores))
        indexed_scores.sort(key=lambda x: x[1], reverse=True)
        
        # 构建结果
        results = []
        for rank, (idx, score) in enumerate(indexed_scores[:top_k]):
            if score > 0:  # 只返回有相关性的结果
                doc = self.documents[idx]
                results.append(SearchResult(
                    chunk=doc['chunk'],
                    score=score,
                    rank=rank + 1
                ))
        
        return results


# ============================================================
# 第二部分：混合检索器
# ============================================================

class HybridRetriever:
    """
    混合检索器
    
    功能说明：
        结合BM25关键词检索和向量语义检索的优势。
        BM25擅长精确匹配关键词，向量检索擅长语义理解。
        两者结合可以显著提升检索质量。
    
    融合策略：
        使用RRF（Reciprocal Rank Fusion）算法融合两种检索结果。
        RRF通过排名的倒数来计算综合分数，避免不同检索方法
        的分数尺度差异问题。
    
    使用示例：
        >>> retriever = HybridRetriever()
        >>> retriever.index(chunks)
        >>> results = retriever.search("如何安装Python", top_k=5)
    """
    
    def __init__(self, k: float = 60.0, vector_weight: float = 0.5):
        """
        初始化混合检索器
        
        参数：
            k: RRF算法的平滑参数（默认60）
                - 通常范围：10-100
                - 值越大，排名影响越小
            vector_weight: 向量检索的权重（默认0.5）
                - 0.0: 完全使用BM25
                - 1.0: 完全使用向量检索
                - 0.5: 两者平等
        """
        self.k = k
        self.vector_weight = vector_weight
        self.bm25_retriever = BM25Retriever()
        self.vector_store: Optional[VectorStore] = None
    
    def index(self, chunks: List[Chunk], collection_name: str = "hybrid_index"):
        """
        建立索引
        
        参数：
            chunks: 要索引的文本块列表
            collection_name: 向量数据库集合名称
        
        功能说明：
            1. 为BM25建立倒排索引
            2. 为向量检索建立向量索引
        """
        # BM25索引
        self.bm25_retriever.add_documents(chunks)
        
        # 向量索引
        self.vector_store = VectorStore(collection_name=collection_name)
        
        # 确保元数据不为空
        for chunk in chunks:
            if not chunk.metadata:
                chunk.metadata = {"chunk_id": chunk.chunk_id}
        
        self.vector_store.add_chunks(chunks)
    
    def search(self, query: str, top_k: int = 5) -> List[SearchResult]:
        """
        混合检索
        
        参数：
            query: 查询文本
            top_k: 返回最相关的k个结果
        
        返回：
            List[SearchResult]: 融合后的搜索结果
        
        算法步骤（RRF融合）：
            1. 分别执行BM25和向量检索
            2. 对每个结果计算RRF分数：1 / (k + rank)
            3. 加权融合两种检索的RRF分数
            4. 按综合分数排序
        """
        # 分别检索
        bm25_results = self.bm25_retriever.search(query, top_k=top_k * 2)
        if self.vector_store is None:
            raise RuntimeError("请先调用 index() 方法建立向量索引")
        vector_results = self.vector_store.search(query, top_k=top_k * 2)
        
        # 构建结果映射
        results_map = {}
        
        # 处理BM25结果
        for result in bm25_results:
            chunk_id = result.chunk.chunk_id
            rrf_score = 1.0 / (self.k + result.rank)
            results_map[chunk_id] = {
                'chunk': result.chunk,
                'bm25_score': rrf_score * (1 - self.vector_weight),
                'vector_score': 0.0
            }
        
        # 处理向量结果
        for result in vector_results:
            chunk_id = result.chunk.chunk_id
            rrf_score = 1.0 / (self.k + result.rank)
            
            if chunk_id in results_map:
                results_map[chunk_id]['vector_score'] = rrf_score * self.vector_weight
            else:
                results_map[chunk_id] = {
                    'chunk': result.chunk,
                    'bm25_score': 0.0,
                    'vector_score': rrf_score * self.vector_weight
                }
        
        # 计算综合分数并排序
        combined_results = []
        for chunk_id, data in results_map.items():
            total_score = data['bm25_score'] + data['vector_score']
            if total_score > 0:
                combined_results.append({
                    'chunk': data['chunk'],
                    'score': total_score
                })
        
        # 按分数排序
        combined_results.sort(key=lambda x: x['score'], reverse=True)
        
        # 构建最终结果
        final_results = []
        for rank, item in enumerate(combined_results[:top_k]):
            final_results.append(SearchResult(
                chunk=item['chunk'],
                score=item['score'],
                rank=rank + 1
            ))
        
        return final_results


# ============================================================
# 第三部分：重排序器
# ============================================================

class Reranker:
    """
    重排序器
    
    功能说明：
        对初步检索结果进行二次排序，提升结果相关性。
        使用LLM评估每个结果与查询的相关程度。
    
    工作流程：
        1. 初步检索获取候选结果（如top-20）
        2. 使用重排序模型评估每个结果
        3. 按相关性分数重新排序
        4. 返回精排后的top-k结果
    
    使用示例：
        >>> reranker = Reranker()
        >>> refined_results = reranker.rerank(query, initial_results, top_k=3)
    """
    
    def __init__(self, llm_client: Optional[LLMClient] = None):
        """
        初始化重排序器
        
        参数：
            llm_client: LLM客户端实例，用于相关性评估
        """
        self.llm = llm_client or LLMClient(temperature=0.0)
    
    def rerank(
        self,
        query: str,
        results: List[SearchResult],
        top_k: int = 3
    ) -> List[SearchResult]:
        """
        对检索结果进行重排序
        
        参数：
            query: 原始查询
            results: 初步检索结果
            top_k: 返回的精排结果数量
        
        返回：
            List[SearchResult]: 重排序后的结果
        
        功能说明：
            使用LLM评估每个文档与查询的相关性，
            然后根据相关性分数重新排序。
        """
        if not results:
            return []
        
        # 对每个结果计算相关性分数
        scored_results = []
        
        for result in results:
            # 构建评估Prompt
            prompt = self._build_rerank_prompt(query, result.chunk.content)
            
            messages = [
                {"role": "system", "content": "你是一个文档相关性评估专家。请根据查询评估文档的相关性，返回0-1之间的分数。"},
                {"role": "user", "content": prompt}
            ]
            
            # 调用LLM评估
            response = self.llm.chat(messages)
            
            # 提取分数（从响应中解析数字）
            score = self._extract_score(response)
            
            scored_results.append(SearchResult(
                chunk=result.chunk,
                score=score,
                rank=result.rank
            ))
        
        # 按新分数排序
        scored_results.sort(key=lambda x: x['score'], reverse=True)
        
        # 更新排名
        final_results = []
        for rank, result in enumerate(scored_results[:top_k]):
            result.rank = rank + 1
            final_results.append(result)
        
        return final_results
    
    def _build_rerank_prompt(self, query: str, document: str) -> str:
        """
        构建重排序评估Prompt
        
        参数：
            query: 查询文本
            document: 文档内容
        
        返回：
            str: 评估Prompt
        """
        prompt = f"""请评估以下文档与查询的相关程度，返回0-1之间的分数。

评分标准：
- 1.0: 文档完全回答了查询的问题
- 0.7-0.9: 文档高度相关，包含大部分关键信息
- 0.4-0.6: 文档部分相关，包含一些有用信息
- 0.1-0.3: 文档轻微相关，只有少量有用信息
- 0.0: 文档完全不相关

查询：{query}

文档：{document[:500]}

请只返回一个数字（0-1之间），不要其他内容。"""
        
        return prompt
    
    def _extract_score(self, response: str) -> float:
        """
        从LLM响应中提取分数
        
        参数：
            response: LLM的响应文本
        
        返回：
            float: 提取的分数（0-1之间）
        """
        # 尝试提取数字
        match = re.search(r'(\d\.?\d*)', response)
        if match:
            score = float(match.group(1))
            return max(0.0, min(1.0, score))  # 限制在0-1范围
        
        return 0.0  # 默认分数


# ============================================================
# 第四部分：查询重写器
# ============================================================

class QueryRewriter:
    """
    查询重写器
    
    功能说明：
        对用户原始查询进行重写和扩展，改善检索效果。
        解决用户查询过于简短、模糊或缺少关键词的问题。
    
    重写策略：
        1. 查询扩展：添加相关关键词
        2. 查询分解：将复杂查询拆分为多个子查询
        3. 假设文档生成：生成假设性相关文档片段
    
    使用示例：
        >>> rewriter = QueryRewriter()
        >>> expanded_queries = rewriter.expand("Python安装")
        >>> print(expanded_queries)
    """
    
    def __init__(self, llm_client: Optional[LLMClient] = None):
        """
        初始化查询重写器
        
        参数：
            llm_client: LLM客户端实例
        """
        self.llm = llm_client or LLMClient(temperature=0.3)
    
    def expand_query(self, query: str, num_expansions: int = 3) -> List[str]:
        """
        查询扩展
        
        参数：
            query: 原始查询
            num_expansions: 生成的扩展查询数量
        
        返回：
            List[str]: 扩展后的查询列表
        
        功能说明：
            使用LLM生成多个语义相关但表述不同的查询，
            然后分别检索并合并结果，提高召回率。
        
        使用示例：
            >>> rewriter = QueryRewriter()
            >>> expansions = rewriter.expand("Python安装")
            >>> print(expansions)
            ['如何安装Python', 'Python环境配置步骤', 'Python下载安装教程']
        """
        prompt = f"""请为以下查询生成{num_expansions}个语义相关但表述不同的扩展查询。
每个扩展查询应该：
1. 保持原始查询的核心意图
2. 使用不同的表述方式
3. 更加具体和明确

原始查询：{query}

请只返回扩展查询，每行一个，不要编号或其他内容。"""
        
        messages = [
            {"role": "system", "content": "你是一个查询扩展专家。请生成相关但表述不同的查询。"},
            {"role": "user", "content": prompt}
        ]
        
        response = self.llm.chat(messages)
        
        # 解析响应，每行一个查询
        if response is None:
            return []
        expansions = [line.strip() for line in response.split('\n') if line.strip()]
        
        # 限制数量
        return expansions[:num_expansions]
    
    def decompose_query(self, query: str) -> List[str]:
        """
        查询分解
        
        参数：
            query: 复杂查询
        
        返回：
            List[str]: 分解后的子查询列表
        
        功能说明：
            将复杂的多部分查询分解为多个简单的子查询，
            分别检索后再综合结果，提高检索精度。
        
        使用示例：
            >>> rewriter = QueryRewriter()
            >>> sub_queries = rewriter.decompose_query("Python和Java的安装步骤有什么区别")
            >>> print(sub_queries)
            ['如何安装Python', '如何安装Java', 'Python和Java安装步骤对比']
        """
        prompt = f"""请将以下复杂查询分解为多个简单的子查询。
每个子查询应该：
1. 只包含一个明确的问题
2. 更容易被搜索引擎理解
3. 能够独立检索

复杂查询：{query}

请只返回子查询，每行一个，不要编号或其他内容。"""
        
        messages = [
            {"role": "system", "content": "你是一个查询分解专家。请将复杂查询分解为简单子查询。"},
            {"role": "user", "content": prompt}
        ]
        
        response = self.llm.chat(messages)
        
        # 解析响应
        if response is None:
            return []
        sub_queries = [line.strip() for line in response.split('\n') if line.strip()]
        
        return sub_queries
    
    def hypothetical_document(self, query: str) -> str:
        """
        假设文档生成（HyDE策略）
        
        参数：
            query: 查询文本
        
        返回：
            str: 假设的相关文档片段
        
        功能说明：
            HyDE（Hypothetical Document Embeddings）策略：
            1. 让LLM生成一个假设性的答案文档
            2. 用这个假设文档去检索真实文档
            3. 因为假设文档与真实文档在向量空间中更接近
            
        使用示例：
            >>> rewriter = QueryRewriter()
            >>> hypo_doc = rewriter.hypothetical_document("Python安装步骤")
            >>> print(hypo_doc)
        """
        prompt = f"""请为以下查询生成一个假设性的答案文档片段（约100-200字）。
这个文档应该包含可能回答该查询的信息。

查询：{query}

假设文档："""
        
        messages = [
            {"role": "system", "content": "你是一个文档生成专家。请生成假设性的相关文档。"},
            {"role": "user", "content": prompt}
        ]
        
        response = self.llm.chat(messages)
        
        return response


# ============================================================
# 第五部分：进阶RAG系统
# ============================================================

class AdvancedRAGSystem:
    """
    进阶RAG系统
    
    功能说明：
        整合混合检索、重排序和查询重写策略的完整RAG系统。
        提供多种检索模式，可根据场景选择最佳策略。
    
    支持的检索模式：
        - basic: 基础向量检索
        - hybrid: 混合检索（BM25 + 向量）
        - rerank: 混合检索 + 重排序
        - expanded: 查询扩展 + 混合检索
    
    使用示例：
        >>> rag = AdvancedRAGSystem(mode="rerank")
        >>> rag.index_documents("./docs")
        >>> answer = rag.query("如何配置Python环境？")
    """
    
    def __init__(
        self,
        mode: str = "hybrid",
        collection_name: str = "advanced_rag",
        chunk_size: int = 500,
        chunk_overlap: int = 50,
        top_k: int = 3
    ):
        """
        初始化进阶RAG系统
        
        参数：
            mode: 检索模式（basic/hybrid/rerank/expanded）
            collection_name: 向量数据库集合名称
            chunk_size: 文本块大小
            chunk_overlap: 文本块重叠大小
            top_k: 检索返回的结果数量
        """
        self.mode = mode
        self.top_k = top_k
        
        # 初始化基础组件（使用已导入的模块）
        from day01_02_basic_rag.basic_rag_pipeline import DocumentLoader, TextChunker
        
        self.loader = DocumentLoader()
        self.chunker = TextChunker(chunk_size=chunk_size, chunk_overlap=chunk_overlap)
        
        # 初始化检索组件
        self.hybrid_retriever = HybridRetriever()
        self.reranker = Reranker()
        self.query_rewriter = QueryRewriter()
        
        # LLM客户端
        self.llm = LLMClient(temperature=0.0)
    
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
    
    def query(self, question: str) -> str:
        """
        查询问答
        
        参数：
            question: 用户问题
        
        返回：
            str: LLM生成的答案
        """
        # 根据模式执行不同的检索策略
        if self.mode == "basic":
            results = self._basic_search(question)
        elif self.mode == "hybrid":
            results = self._hybrid_search(question)
        elif self.mode == "rerank":
            results = self._rerank_search(question)
        elif self.mode == "expanded":
            results = self._expanded_search(question)
        else:
            results = self._hybrid_search(question)
        
        # 构建上下文
        context = self._build_context(results)
        
        # 构建Prompt
        prompt = self._build_prompt(question, context)
        
        # 调用LLM
        messages = [
            {"role": "system", "content": "你是一个基于文档知识的问答助手。请根据提供的上下文回答问题。"},
            {"role": "user", "content": prompt}
        ]
        
        answer = self.llm.chat(messages)
        
        return answer
    
    def _basic_search(self, query: str) -> List[SearchResult]:
        """基础向量检索"""
        if self.hybrid_retriever.vector_store is None:
            raise RuntimeError("请先调用 index() 方法建立向量索引")
        return self.hybrid_retriever.vector_store.search(query, top_k=self.top_k)
    
    def _hybrid_search(self, query: str) -> List[SearchResult]:
        """混合检索"""
        return self.hybrid_retriever.search(query, top_k=self.top_k)
    
    def _rerank_search(self, query: str) -> List[SearchResult]:
        """混合检索 + 重排序"""
        # 先获取较多的候选结果
        initial_results = self.hybrid_retriever.search(query, top_k=self.top_k * 3)
        
        # 重排序
        refined_results = self.reranker.rerank(query, initial_results, top_k=self.top_k)
        
        return refined_results
    
    def _expanded_search(self, query: str) -> List[SearchResult]:
        """查询扩展 + 混合检索"""
        # 扩展查询
        expansions = self.query_rewriter.expand_query(query, num_expansions=3)
        
        # 合并所有查询
        all_queries = [query] + expansions
        
        # 对每个查询进行检索
        all_results = {}
        
        for q in all_queries:
            results = self.hybrid_retriever.search(q, top_k=self.top_k)
            
            for result in results:
                chunk_id = result.chunk.chunk_id
                if chunk_id not in all_results:
                    all_results[chunk_id] = result
        
        # 按分数排序
        sorted_results = sorted(all_results.values(), key=lambda x: x.score, reverse=True)
        
        return sorted_results[:self.top_k]
    
    def _build_context(self, results: List[SearchResult]) -> str:
        """构建上下文字符串"""
        context_parts = []
        
        for i, result in enumerate(results):
            context_parts.append(f"[文档{i+1}]\n{result.chunk.content}")
        
        return "\n\n".join(context_parts)
    
    def _build_prompt(self, question: str, context: str) -> str:
        """构建RAG Prompt"""
        prompt = f"""请根据以下上下文回答问题。如果上下文中没有相关信息，请明确说明。

上下文：
{context}

问题：{question}

答案："""
        
        return prompt


# ============================================================
# 第六部分：演示函数
# ============================================================

def demonstrate_advanced_rag():
    """
    演示进阶RAG策略
    
    功能说明：
        展示不同检索策略的效果对比。
    """
    print("=" * 60)
    print("进阶RAG策略演示")
    print("=" * 60)
    
    # 创建示例文档
    sample_dir = os.path.join(os.path.dirname(__file__), "..", "sample_docs")
    os.makedirs(sample_dir, exist_ok=True)
    
    sample_content = """# Python虚拟环境管理

## 什么是虚拟环境
虚拟环境是Python项目的独立空间，包含该项目所需的依赖包。
使用虚拟环境可以避免不同项目之间的依赖冲突。

## 创建虚拟环境
使用venv模块创建虚拟环境：
```bash
python -m venv myenv
```

## 激活虚拟环境
Windows:
```bash
myenv\\Scripts\\activate
```

Linux/Mac:
```bash
source myenv/bin/activate
```

## 安装依赖
激活虚拟环境后，使用pip安装依赖：
```bash
pip install requests numpy pandas
```

## 导出依赖
将当前环境的依赖导出到文件：
```bash
pip freeze > requirements.txt
```

## 从文件安装依赖
从requirements.txt安装所有依赖：
```bash
pip install -r requirements.txt
```

## 常见问题
Q: 虚拟环境占用多少空间？
A: 通常几十MB到几百MB，取决于安装的包数量。

Q: 如何删除虚拟环境？
A: 直接删除虚拟环境文件夹即可。
"""
    
    with open(os.path.join(sample_dir, "venv_guide.txt"), 'w', encoding='utf-8') as f:
        f.write(sample_content)
    
    print(f"\n示例文档已创建：{sample_dir}\n")
    
    # 测试不同检索模式
    modes = ["hybrid", "expanded"]
    question = "如何创建和激活Python虚拟环境？"
    
    for mode in modes:
        print(f"\n{'=' * 60}")
        print(f"检索模式：{mode}")
        print(f"{'=' * 60}")
        
        rag = AdvancedRAGSystem(
            mode=mode,
            collection_name=f"demo_{mode}",
            top_k=2
        )
        
        rag.index_documents(sample_dir, source_type="directory")
        
        print(f"\n问题：{question}")
        print("-" * 60)
        answer = rag.query(question)
        print(f"\n答案：{answer}")


if __name__ == "__main__":
    """
    主程序入口
    
    运行方式：
        python advanced_rag_strategies.py
    """
    demonstrate_advanced_rag()
