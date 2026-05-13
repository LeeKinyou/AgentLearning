"""
Day 1-2: 基础RAG流水线实现

本模块演示如何构建一个完整的RAG（Retrieval-Augmented Generation）系统，包括：
1. 文档加载与分块（Chunking策略）
2. 向量化与向量数据库（Chroma）
3. 相似度检索与答案生成

学习目标：
- 理解RAG系统的核心组件和工作流程
- 掌握文档分块策略的选择和实现
- 学会使用向量数据库进行相似度检索
- 能够构建基础的RAG问答系统

依赖安装：
    pip install chromadb openai pydantic python-dotenv
"""

import os
import re
from typing import List, Dict, Optional, Tuple
from dataclasses import dataclass
from dotenv import load_dotenv

# 加载环境变量
load_dotenv()


# ============================================================
# 第一部分：数据模型定义
# ============================================================

@dataclass
class Document:
    """
    文档数据模型
    
    功能说明：
        表示一个完整的文档，包含原始内容和元数据。
        用于文档加载阶段的统一数据表示。
    
    属性：
        content: 文档的原始文本内容
        metadata: 文档元数据，如来源、标题、创建时间等
        doc_id: 文档唯一标识符
    
    使用示例：
        >>> doc = Document(
        ...     content="这是文档内容...",
        ...     metadata={"source": "manual.pdf", "page": 1},
        ...     doc_id="doc_001"
        ... )
    """
    content: str
    metadata: Dict = None
    doc_id: str = ""
    
    def __post_init__(self):
        """
        数据初始化后处理
        
        功能说明：
            自动设置默认值，确保数据完整性。
        """
        if self.metadata is None:
            self.metadata = {}
        if not self.doc_id:
            # 使用内容哈希生成唯一ID
            self.doc_id = f"doc_{hash(self.content) % 100000:05d}"


@dataclass
class Chunk:
    """
    文本块数据模型
    
    功能说明：
        表示文档分块后的文本片段，包含原始文档的引用信息。
        用于向量化和检索阶段的数据表示。
    
    属性：
        content: 文本块的内容
        metadata: 文本块元数据，包含原始文档信息
        chunk_id: 文本块唯一标识符
        doc_id: 所属文档的ID
    
    使用示例：
        >>> chunk = Chunk(
        ...     content="这是文本块内容...",
        ...     metadata={"source": "manual.pdf", "chunk_index": 0},
        ...     chunk_id="chunk_001",
        ...     doc_id="doc_001"
        ... )
    """
    content: str
    metadata: Dict = None
    chunk_id: str = ""
    doc_id: str = ""
    
    def __post_init__(self):
        """数据初始化后处理"""
        if self.metadata is None:
            self.metadata = {}
        if not self.chunk_id:
            self.chunk_id = f"chunk_{hash(self.content) % 100000:05d}"


@dataclass
class SearchResult:
    """
    搜索结果数据模型
    
    功能说明：
        表示一次检索操作返回的结果，包含匹配的文本块和相似度分数。
        用于RAG系统的检索阶段输出。
    
    属性：
        chunk: 匹配的文本块
        score: 相似度分数（越高表示越相关）
        rank: 结果排名
    
    使用示例：
        >>> result = SearchResult(chunk=chunk, score=0.85, rank=1)
        >>> print(f"相似度：{result.score}")
    """
    chunk: Chunk
    score: float
    rank: int = 0


# ============================================================
# 第二部分：文档加载器
# ============================================================

class DocumentLoader:
    """
    文档加载器
    
    功能说明：
        从不同来源加载文档，支持文本文件、目录批量加载等。
        提供统一的文档加载接口。
    
    使用示例：
        >>> loader = DocumentLoader()
        >>> docs = loader.load_from_file("sample.txt")
        >>> docs = loader.load_from_directory("./docs")
    """
    
    def load_from_file(self, file_path: str, encoding: str = "utf-8") -> List[Document]:
        """
        从单个文件加载文档
        
        参数：
            file_path: 文件路径
            encoding: 文件编码，默认utf-8
        
        返回：
            List[Document]: 加载的文档列表
        
        使用示例：
            >>> loader = DocumentLoader()
            >>> docs = loader.load_from_file("manual.txt")
        """
        if not os.path.exists(file_path):
            raise FileNotFoundError(f"文件不存在：{file_path}")
        
        with open(file_path, 'r', encoding=encoding) as f:
            content = f.read()
        
        # 提取文件名作为元数据
        file_name = os.path.basename(file_path)
        
        return [Document(
            content=content,
            metadata={"source": file_name, "type": "file"},
            doc_id=f"doc_{file_name}"
        )]
    
    def load_from_directory(
        self,
        directory: str,
        encoding: str = "utf-8",
        extensions: List[str] = None
    ) -> List[Document]:
        """
        从目录批量加载文档
        
        参数：
            directory: 目录路径
            encoding: 文件编码，默认utf-8
            extensions: 要加载的文件扩展名列表，默认['.txt', '.md']
        
        返回：
            List[Document]: 加载的所有文档列表
        
        使用示例：
            >>> loader = DocumentLoader()
            >>> docs = loader.load_from_directory("./docs", extensions=['.txt', '.md'])
        """
        if extensions is None:
            extensions = ['.txt', '.md']
        
        if not os.path.isdir(directory):
            raise NotADirectoryError(f"目录不存在：{directory}")
        
        documents = []
        
        # 遍历目录中的所有文件
        for root, _, files in os.walk(directory):
            for file_name in files:
                # 检查文件扩展名
                if any(file_name.lower().endswith(ext) for ext in extensions):
                    file_path = os.path.join(root, file_name)
                    try:
                        docs = self.load_from_file(file_path, encoding)
                        documents.extend(docs)
                    except Exception as e:
                        print(f"加载文件失败 {file_path}: {e}")
        
        return documents


# ============================================================
# 第三部分：文本分块器
# ============================================================

class TextChunker:
    """
    文本分块器
    
    功能说明：
        将长文档切分为适合向量化的文本块。
        支持多种分块策略：固定长度、段落分割、递归分割等。
    
    核心参数说明：
        - chunk_size: 每个文本块的最大字符数
        - chunk_overlap: 相邻文本块的重叠字符数（用于保持上下文连贯性）
    
    使用示例：
        >>> chunker = TextChunker(chunk_size=500, chunk_overlap=50)
        >>> chunks = chunker.split(document)
    """
    
    def __init__(self, chunk_size: int = 500, chunk_overlap: int = 50):
        """
        初始化文本分块器
        
        参数：
            chunk_size: 每个文本块的最大字符数，默认500
            chunk_overlap: 相邻文本块的重叠字符数，默认50
        
        注意：
            chunk_overlap 应小于 chunk_size，否则会导致无限循环
        """
        if chunk_overlap >= chunk_size:
            raise ValueError("chunk_overlap 必须小于 chunk_size")
        
        self.chunk_size = chunk_size
        self.chunk_overlap = chunk_overlap
    
    def split_by_fixed_size(self, text: str) -> List[str]:
        """
        按固定长度分割文本
        
        功能说明：
            将文本按固定字符数分割，相邻块之间有重叠。
            重叠部分用于保持上下文连贯性，避免关键信息被切断。
        
        参数：
            text: 要分割的文本
        
        返回：
            List[str]: 分割后的文本块列表
        
        算法步骤：
            1. 从文本开头取chunk_size个字符作为第一个块
            2. 下一个块从当前位置 - overlap处开始
            3. 重复直到文本结束
        """
        chunks = []
        start = 0
        text_length = len(text)
        
        while start < text_length:
            # 计算当前块的结束位置
            end = start + self.chunk_size
            
            # 提取文本块
            chunk = text[start:end]
            chunks.append(chunk)
            
            # 移动起始位置（减去重叠部分）
            start = end - self.chunk_overlap
            
            # 如果剩余文本很少，直接合并到最后一个块
            if text_length - start < self.chunk_size // 2:
                if start < text_length:
                    chunks[-1] += text[start:]
                break
        
        return chunks
    
    def split_by_paragraph(self, text: str) -> List[str]:
        """
        按段落分割文本
        
        功能说明：
            先按段落分割，然后将过长的段落进一步拆分。
            这种方式能更好地保持语义完整性。
        
        参数：
            text: 要分割的文本
        
        返回：
            List[str]: 分割后的文本块列表
        
        算法步骤：
            1. 按双换行符分割段落
            2. 合并短段落，直到接近chunk_size
            3. 对超长段落使用固定长度分割
        """
        # 按段落分割（双换行符分隔）
        paragraphs = re.split(r'\n\s*\n', text)
        paragraphs = [p.strip() for p in paragraphs if p.strip()]
        
        chunks = []
        current_chunk = ""
        
        for para in paragraphs:
            # 如果单个段落就超过chunk_size，需要进一步分割
            if len(para) > self.chunk_size:
                # 先保存当前累积的块
                if current_chunk:
                    chunks.append(current_chunk)
                    current_chunk = ""
                
                # 对超长段落使用固定长度分割
                sub_chunks = self.split_by_fixed_size(para)
                chunks.extend(sub_chunks)
            
            # 如果加上当前段落会超过chunk_size，先保存当前块
            elif len(current_chunk) + len(para) + 2 > self.chunk_size:
                chunks.append(current_chunk)
                current_chunk = para
            
            # 否则继续累积
            else:
                if current_chunk:
                    current_chunk += "\n\n" + para
                else:
                    current_chunk = para
        
        # 添加最后一个块
        if current_chunk:
            chunks.append(current_chunk)
        
        return chunks
    
    def split(
        self,
        document: Document,
        strategy: str = "paragraph"
    ) -> List[Chunk]:
        """
        对文档进行分块
        
        参数：
            document: 要分块的文档
            strategy: 分块策略，可选值：
                - "fixed": 固定长度分割
                - "paragraph": 按段落分割（推荐）
        
        返回：
            List[Chunk]: 分块后的文本块列表
        
        使用示例：
            >>> chunker = TextChunker(chunk_size=500, chunk_overlap=50)
            >>> chunks = chunker.split(doc, strategy="paragraph")
        """
        # 根据策略选择分割方法
        if strategy == "fixed":
            texts = self.split_by_fixed_size(document.content)
        elif strategy == "paragraph":
            texts = self.split_by_paragraph(document.content)
        else:
            raise ValueError(f"不支持的分块策略：{strategy}")
        
        # 转换为Chunk对象
        chunks = []
        for i, text in enumerate(texts):
            chunk = Chunk(
                content=text,
                metadata={
                    **document.metadata,
                    "chunk_index": i,
                    "chunk_size": len(text),
                    "strategy": strategy
                },
                chunk_id=f"{document.doc_id}_chunk_{i:03d}",
                doc_id=document.doc_id
            )
            chunks.append(chunk)
        
        return chunks


# ============================================================
# 第四部分：向量数据库（基于Chroma）
# ============================================================

class VectorStore:
    """
    向量数据库封装类
    
    功能说明：
        封装Chroma向量数据库的操作，提供简洁的存储和检索接口。
        支持文本的向量化存储和相似度检索。
    
    核心概念：
        - Embedding: 将文本转换为高维向量表示
        - 相似度检索: 通过向量距离计算文本相似度
        - Collection: 向量集合，类似数据库中的表
    
    使用示例：
        >>> store = VectorStore(collection_name="my_docs")
        >>> store.add_chunks(chunks)
        >>> results = store.search("查询文本", top_k=3)
    """
    
    def __init__(self, collection_name: str = "default", persist_directory: str = None):
        """
        初始化向量数据库
        
        参数：
            collection_name: 集合名称，用于区分不同的文档集合
            persist_directory: 持久化存储目录，None表示仅内存存储
        
        使用示例：
            # 仅内存存储（适合测试）
            >>> store = VectorStore("test_collection")
            
            # 持久化存储（适合生产）
            >>> store = VectorStore("prod_collection", "./chroma_db")
        """
        self.collection_name = collection_name
        self.persist_directory = persist_directory
        self._client = None
        self._collection = None
    
    def _get_client(self):
        """
        获取Chroma客户端实例（延迟初始化）
        
        返回：
            chromadb.Client: Chroma客户端实例
        """
        if self._client is None:
            import chromadb
            
            if self.persist_directory:
                # 持久化客户端（数据保存到磁盘）
                self._client = chromadb.PersistentClient(path=self.persist_directory)
            else:
                # 内存客户端（速度快，重启后数据丢失）
                self._client = chromadb.Client()
        
        return self._client
    
    def _get_collection(self):
        """
        获取或创建集合
        
        返回：
            chromadb.Collection: Chroma集合实例
        """
        if self._collection is None:
            client = self._get_client()
            
            # 尝试获取现有集合，不存在则创建
            try:
                self._collection = client.get_collection(self.collection_name)
            except:
                self._collection = client.create_collection(self.collection_name)
        
        return self._collection
    
    def add_chunks(self, chunks: List[Chunk]):
        """
        添加文本块到向量数据库
        
        参数：
            chunks: 要添加的文本块列表
        
        功能说明：
            1. 提取文本内容和元数据
            2. 自动生成ID
            3. 调用Chroma进行向量化和存储
        
        使用示例：
            >>> store = VectorStore("my_docs")
            >>> store.add_chunks(chunks)
        """
        collection = self._get_collection()
        
        # 准备批量添加的数据
        ids = [chunk.chunk_id for chunk in chunks]
        documents = [chunk.content for chunk in chunks]
        
        # 确保元数据不为空（Chroma要求非空元数据）
        metadatas = []
        for chunk in chunks:
            if chunk.metadata:
                metadatas.append(chunk.metadata)
            else:
                # 如果元数据为空，添加默认元数据
                metadatas.append({"chunk_id": chunk.chunk_id})
        
        # 批量添加到数据库（Chroma会自动进行向量化）
        collection.add(
            ids=ids,
            documents=documents,
            metadatas=metadatas
        )
    
    def search(self, query: str, top_k: int = 3) -> List[SearchResult]:
        """
        相似度检索
        
        参数：
            query: 查询文本
            top_k: 返回最相关的k个结果
        
        返回：
            List[SearchResult]: 搜索结果列表，按相似度降序排列
        
        功能说明：
            1. 将查询文本向量化
            2. 计算与数据库中所有向量的相似度
            3. 返回最相似的top_k个结果
        
        使用示例：
            >>> results = store.search("如何配置API密钥", top_k=3)
            >>> for result in results:
            ...     print(f"相似度：{result.score}")
            ...     print(f"内容：{result.chunk.content[:100]}")
        """
        collection = self._get_collection()
        
        # 执行相似度查询
        results = collection.query(
            query_texts=[query],
            n_results=top_k,
            include=["documents", "metadatas", "distances"]
        )
        
        # 解析结果
        search_results = []
        for i in range(len(results['ids'][0])):
            chunk = Chunk(
                content=results['documents'][0][i],
                metadata=results['metadatas'][0][i],
                chunk_id=results['ids'][0][i]
            )
            
            # Chroma返回的是距离（越小越相似），转换为相似度分数
            distance = results['distances'][0][i]
            score = 1.0 / (1.0 + distance)  # 距离转相似度
            
            search_results.append(SearchResult(
                chunk=chunk,
                score=score,
                rank=i + 1
            ))
        
        return search_results
    
    def clear(self):
        """
        清空集合中的所有数据
        
        使用示例：
            >>> store.clear()
        """
        client = self._get_client()
        try:
            client.delete_collection(self.collection_name)
        except:
            pass
        
        self._collection = None


# ============================================================
# 第五部分：LLM调用封装（复用Week01的代码）
# ============================================================

class LLMClient:
    """
    LLM调用客户端
    
    功能说明：
        封装与LLM的交互逻辑，支持多种LLM服务。
        复用Week01的实现，保持代码一致性。
    """
    
    def __init__(
        self,
        model: str = "gpt-3.5-turbo",
        temperature: float = 0.0,
        max_tokens: int = 1000,
        api_key: Optional[str] = None,
        base_url: Optional[str] = None
    ):
        """
        初始化LLM客户端
        
        参数：
            model: LLM模型名称
            temperature: 输出随机性（RAG建议使用低温度）
            max_tokens: 最大生成Token数
            api_key: API密钥
            base_url: API基础URL
        """
        self.api_key = api_key or os.getenv("OPENAI_API_KEY")
        if not self.api_key:
            raise ValueError("未找到 OPENAI_API_KEY 环境变量")
        
        self.base_url = base_url or os.getenv("API_BASE_URL")
        if not self.base_url:
            self.base_url = "https://api.openai.com/v1"
        
        self.model = os.getenv("MODEL_NAME", model)
        self.temperature = temperature
        self.max_tokens = max_tokens
        self._client = None
    
    def _get_client(self):
        """获取OpenAI客户端实例"""
        if self._client is None:
            from openai import OpenAI
            self._client = OpenAI(
                api_key=self.api_key,
                base_url=self.base_url
            )
        return self._client
    
    def chat(self, messages: List[Dict[str, str]]) -> str:
        """
        发送消息并获取回复
        
        参数：
            messages: 消息列表
        
        返回：
            str: LLM生成的回复内容
        """
        try:
            response = self._get_client().chat.completions.create(
                model=self.model,
                messages=messages,
                temperature=self.temperature,
                max_tokens=self.max_tokens
            )
            return response.choices[0].message.content.strip()
        except Exception as e:
            return f"API调用失败：{str(e)}"


# ============================================================
# 第六部分：RAG系统核心
# ============================================================

class BasicRAGSystem:
    """
    基础RAG系统
    
    功能说明：
        整合文档加载、分块、向量化、检索和答案生成的完整RAG流水线。
        提供简洁的接口用于文档索引和问答。
    
    工作流程：
        1. 文档加载：从文件/目录加载原始文档
        2. 文本分块：将长文档切分为小块
        3. 向量化存储：将文本块转换为向量并存储
        4. 相似度检索：根据查询找到最相关的文本块
        5. 答案生成：将检索结果作为上下文，让LLM生成答案
    
    使用示例：
        >>> rag = BasicRAGSystem()
        >>> rag.index_documents("./docs")
        >>> answer = rag.query("如何配置API密钥？")
        >>> print(answer)
    """
    
    def __init__(
        self,
        collection_name: str = "rag_docs",
        chunk_size: int = 500,
        chunk_overlap: int = 50,
        top_k: int = 3
    ):
        """
        初始化RAG系统
        
        参数：
            collection_name: 向量数据库集合名称
            chunk_size: 文本块大小
            chunk_overlap: 文本块重叠大小
            top_k: 检索返回的结果数量
        """
        # 初始化各组件
        self.loader = DocumentLoader()
        self.chunker = TextChunker(chunk_size=chunk_size, chunk_overlap=chunk_overlap)
        self.vector_store = VectorStore(collection_name=collection_name)
        self.llm = LLMClient(temperature=0.0)  # RAG使用低温度
        self.top_k = top_k
    
    def index_documents(self, source: str, source_type: str = "directory"):
        """
        索引文档
        
        参数：
            source: 文档来源（文件路径或目录路径）
            source_type: 来源类型，"file" 或 "directory"
        
        功能说明：
            1. 加载文档
            2. 分块处理
            3. 存储到向量数据库
        
        使用示例：
            >>> rag = BasicRAGSystem()
            >>> rag.index_documents("./docs", source_type="directory")
            >>> rag.index_documents("manual.txt", source_type="file")
        """
        print(f"[1/3] 加载文档：{source}")
        
        # 加载文档
        if source_type == "file":
            documents = self.loader.load_from_file(source)
        else:
            documents = self.loader.load_from_directory(source)
        
        print(f"  加载了 {len(documents)} 个文档")
        
        print(f"[2/3] 文本分块")
        
        # 分块处理
        all_chunks = []
        for doc in documents:
            chunks = self.chunker.split(doc, strategy="paragraph")
            all_chunks.extend(chunks)
        
        print(f"  生成了 {len(all_chunks)} 个文本块")
        
        print(f"[3/3] 向量化存储")
        
        # 存储到向量数据库
        self.vector_store.add_chunks(all_chunks)
        
        print(f"  索引完成！")
    
    def query(self, question: str) -> str:
        """
        查询问答
        
        参数：
            question: 用户问题
        
        返回：
            str: LLM生成的答案
        
        功能说明：
            1. 检索相关文档
            2. 构建Prompt（包含检索结果作为上下文）
            3. 调用LLM生成答案
        
        使用示例：
            >>> answer = rag.query("如何配置API密钥？")
            >>> print(answer)
        """
        # 步骤1：检索相关文档
        print(f"[检索] 查询：{question}")
        results = self.vector_store.search(question, top_k=self.top_k)
        
        # 打印检索结果
        for i, result in enumerate(results):
            print(f"  [{i+1}] 相似度：{result.score:.3f}")
            print(f"      内容：{result.chunk.content[:100]}...")
        
        # 步骤2：构建上下文
        context = self._build_context(results)
        
        # 步骤3：构建Prompt
        prompt = self._build_prompt(question, context)
        
        # 步骤4：调用LLM生成答案
        messages = [
            {"role": "system", "content": "你是一个基于文档知识的问答助手。请根据提供的上下文回答问题。如果上下文中没有相关信息，请明确说明。"},
            {"role": "user", "content": prompt}
        ]
        
        answer = self.llm.chat(messages)
        
        return answer
    
    def _build_context(self, results: List[SearchResult]) -> str:
        """
        构建检索结果的上下文字符串
        
        参数：
            results: 检索结果列表
        
        返回：
            str: 格式化的上下文字符串
        """
        context_parts = []
        
        for i, result in enumerate(results):
            context_parts.append(f"[文档{i+1}]\n{result.chunk.content}")
        
        return "\n\n".join(context_parts)
    
    def _build_prompt(self, question: str, context: str) -> str:
        """
        构建RAG Prompt
        
        参数：
            question: 用户问题
            context: 检索到的上下文
        
        返回：
            str: 完整的Prompt
        """
        prompt = f"""请根据以下上下文回答问题。如果上下文中没有相关信息，请明确说明"根据提供的文档，我无法回答这个问题"。

上下文：
{context}

问题：{question}

答案："""
        
        return prompt


# ============================================================
# 第七部分：演示函数
# ============================================================

def demonstrate_basic_rag():
    """
    演示基础RAG系统
    
    功能说明：
        展示完整的RAG工作流程：
        1. 创建示例文档
        2. 索引文档
        3. 执行查询
        4. 显示结果
    """
    print("=" * 60)
    print("基础RAG系统演示")
    print("=" * 60)
    
    # 创建示例文档
    sample_dir = os.path.join(os.path.dirname(__file__), "..", "sample_docs")
    os.makedirs(sample_dir, exist_ok=True)
    
    # 写入示例文档
    sample_content = """# Python环境配置指南

## 安装Python
1. 访问Python官网：https://www.python.org/
2. 下载适合你操作系统的安装包
3. 运行安装程序，确保勾选"Add Python to PATH"
4. 完成安装后，在命令行输入 python --version 验证

## 配置虚拟环境
推荐使用venv或conda创建虚拟环境：

使用venv：
```bash
python -m venv myenv
source myenv/bin/activate  # Linux/Mac
myenv\\Scripts\\activate    # Windows
```

## 安装依赖
使用pip安装项目依赖：
```bash
pip install -r requirements.txt
```

## 常见问题
Q: 如何升级pip？
A: 运行 python -m pip install --upgrade pip

Q: 如何解决依赖冲突？
A: 使用虚拟环境隔离不同项目的依赖
"""
    
    with open(os.path.join(sample_dir, "python_guide.txt"), 'w', encoding='utf-8') as f:
        f.write(sample_content)
    
    print(f"\n示例文档已创建：{sample_dir}\n")
    
    # 初始化RAG系统
    rag = BasicRAGSystem(
        collection_name="demo_rag",
        chunk_size=300,
        chunk_overlap=30,
        top_k=2
    )
    
    # 索引文档
    print("\n开始索引文档...")
    rag.index_documents(sample_dir, source_type="directory")
    
    # 执行查询
    print("\n" + "=" * 60)
    print("执行查询")
    print("=" * 60)
    
    questions = [
        "如何安装Python？",
        "如何创建虚拟环境？",
        "如何解决依赖冲突？"
    ]
    
    for question in questions:
        print(f"\n问题：{question}")
        print("-" * 60)
        answer = rag.query(question)
        print(f"\n答案：{answer}")
        print("=" * 60)


if __name__ == "__main__":
    """
    主程序入口
    
    运行方式：
        python basic_rag_pipeline.py
    """
    demonstrate_basic_rag()
