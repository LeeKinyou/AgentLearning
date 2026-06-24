import os
from pathlib import Path
from dotenv import load_dotenv

import chromadb
from openai import OpenAI

# LangChain 工具用于快速读取和切分
from langchain_community.document_loaders import TextLoader
from langchain_text_splitters import RecursiveCharacterTextSplitter
from langchain_openai import OpenAIEmbeddings

load_dotenv(os.path.join(os.path.dirname(__file__), '..', '.env'))

# 初始化 OpenAI 客户端
client = OpenAI()

def build_index(repo_path: str, collection_name: str = "repo_index"):
    """
    离线阶段：读取代码文件、切分并构建 ChromaDB 向量索引
    """
    print(f"正在扫描目录：{repo_path}")
    
    documents = []
    # 1. Document Loading：递归读取 .py 文件，排除没用的目录
    exclude_dirs = {".venv", "__pycache__", ".git", "node_modules"}
    
    for root, dirs, files in os.walk(repo_path):
        # 过滤目录
        dirs[:] = [d for d in dirs if d not in exclude_dirs]
        
        for file in files:
            if file.endswith(".py"):
                file_path = os.path.join(root, file)
                try:
                    loader = TextLoader(file_path, encoding='utf-8')
                    documents.extend(loader.load())
                except Exception as e:
                    print(f"读取文件 {file_path} 失败: {e}")

    if not documents:
        print("未找到任何 Python 代码文件！")
        return None

    print(f"共加载了 {len(documents)} 个代码文件，开始切分...")

    # 2. Chunking：按代码特征切块
    text_splitter = RecursiveCharacterTextSplitter(
        chunk_size=800,
        chunk_overlap=100,
        separators=["\nclass ", "\ndef ", "\n\n", "\n", " ", ""]
    )
    chunks = text_splitter.split_documents(documents)
    print(f"共切分为 {len(chunks)} 个代码片段。")

    # 3. 准备写入 ChromaDB
    chroma_client = chromadb.PersistentClient(path="./demo_chroma_db")
    
    # 每次运行为了演示，先删除旧集合
    try:
        chroma_client.delete_collection(name=collection_name)
    except:
        pass
        
    collection = chroma_client.create_collection(name=collection_name)

    # 提取内容和元数据
    ids = [f"chunk_{i}" for i in range(len(chunks))]
    texts = [chunk.page_content for chunk in chunks]
    metadatas = [{"source": chunk.metadata.get("source", "unknown")} for chunk in chunks]

    # 使用 OpenAI Embeddings 模型将文本向量化并写入数据库 (Chroma 也可以默认内置 embedding 模型)
    print("正在调用 Embedding 模型写入向量数据库...")
    embeddings_model = OpenAIEmbeddings(model="text-embedding-3-small")
    embeddings = embeddings_model.embed_documents(texts)
    
    collection.add(
        ids=ids,
        documents=texts,
        embeddings=embeddings, # type: ignore
        metadatas=metadatas    # type: ignore
    )
    
    print("✅ 索引构建完成！\n")
    return collection

def ask_repo(query: str, collection):
    """
    在线阶段：用户提问 -> 检索 -> LLM 生成
    """
    print(f"【用户提问】：{query}")
    
    # 1. 把问题也转为向量
    embeddings_model = OpenAIEmbeddings(model="text-embedding-3-small")
    query_embedding = embeddings_model.embed_query(query)
    
    # 2. 检索 Top-K 相关片段
    top_k = 3
    print(f"正在检索最相关的 {top_k} 个代码片段...")
    results = collection.query(
        query_embeddings=[query_embedding],
        n_results=top_k
    )
    
    # 组装上下文证据
    context_blocks = []
    documents = results['documents'][0]
    metadatas = results['metadatas'][0]
    
    for i in range(len(documents)):
        source = metadatas[i]['source']
        content = documents[i]
        context_blocks.append(f"【文件路径】: {source}\n【代码片段】:\n{content}\n")
        
    context_str = "\n".join(context_blocks)
    
    # 3. Prompt 组装：把检索结果交给模型
    prompt = f"""你是一个专业的代码仓库分析助手。
请只基于给定的代码片段证据来回答用户的问题。
回答中必须注明代码所在的【文件路径】。

【用户提问】
{query}

【检索到的代码证据】
{context_str}

【输出要求】
1. 结论
2. 证据引用（文件路径）
"""
    
    print("正在调用 LLM 基于证据生成回答...\n")
    model_name = os.getenv("MODEL_NAME", "gpt-4o-mini")
    response = client.chat.completions.create(
        model=model_name,
        messages=[
            {"role": "user", "content": prompt}
        ],
        temperature=0.1
    )
    
    print("【系统回答】")
    print(response.choices[0].message.content)

if __name__ == "__main__":
    # 使用项目根目录作为示例分析仓库
    project_root = str(Path(__file__).parent.parent.absolute())
    
    # 构建离线索引
    collection = build_index(repo_path=project_root)
    
    if collection:
        # 在线检索问答
        ask_repo("当前项目中，FastAPI 的服务是在哪个文件启动的？端口是多少？", collection)
