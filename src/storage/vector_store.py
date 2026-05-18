from __future__ import annotations

"""
Vector Store 模块 — 存储和检索向量
====================================

核心概念：
- Vector DB 存储三样东西：向量(vector) + 原始文本(document) + 元数据(metadata)
- 你可以把它想象成一个特殊的数据库：
  普通数据库: 用 SQL 按条件查（WHERE name = 'Jerry'）
  向量数据库: 用"相似度"查（找到和这个向量最像的前 k 个）

为什么选 ChromaDB？
- 零配置，pip install 就能用，不需要启动单独的服务
- 支持内存模式（Phase 1）和持久化模式（Phase 6）
- API 简洁，适合学习
- 生产环境可以换成 Qdrant / Pinecone / Weaviate

面试考点：
- Vector DB 不只是存向量，还要存原文和 metadata（来源、页码、时间等）
- 相似度算法：cosine similarity（最常用）, dot product, L2 distance
- ChromaDB 默认用 cosine similarity
"""

import chromadb
from src.config import CHROMA_COLLECTION_NAME


# Phase 1: 使用内存模式（数据只在程序运行期间存在）
# Phase 6 会改为持久化模式: chromadb.PersistentClient(path=CHROMA_PERSIST_DIR)
_client = chromadb.Client()


def get_or_create_collection(
    collection_name: str = CHROMA_COLLECTION_NAME,
) -> chromadb.Collection:
    """
    获取或创建一个 ChromaDB collection。
    Collection 类似于传统数据库中的"表"。
    """
    # get_or_create: 如果已存在就获取，不存在就创建
    # metadata 中指定距离度量方式为 cosine（余弦相似度）
    collection = _client.get_or_create_collection(
        name=collection_name,
        metadata={"hnsw:space": "cosine"},
    )
    return collection


def add_documents(
    collection: chromadb.Collection,
    documents: list[str],
    embeddings: list[list[float]],
    metadatas: list[dict] | None = None,
    ids: list[str] | None = None,
) -> None:
    """
    向 collection 中添加文档。

    参数:
        collection: ChromaDB collection 对象
        documents: 原始文本列表（存储原文，检索后可以直接返回给用户看）
        embeddings: 对应的向量列表（用于相似度计算）
        metadatas: 元数据列表（来源、页码等额外信息，方便溯源）
        ids: 唯一 ID 列表（ChromaDB 要求每条记录有唯一 ID）

    为什么要同时存 document 和 embedding？
    - embedding 用于高效的相似度搜索（纯数学运算）
    - document 存原文，搜到之后要把原文返回给 LLM 作为 context
    - metadata 存来源信息，回答时可以引用"这个信息来自 xx 文件第 xx 页"
    """
    # 如果没提供 ID，自动生成（用序号作为 ID）
    if ids is None:
        # 查询已有文档数量，确保 ID 不重复
        existing_count = collection.count()
        ids = [f"doc_{existing_count + i}" for i in range(len(documents))]

    if metadatas is None:
        metadatas = [{}] * len(documents)

    collection.add(
        documents=documents,
        embeddings=embeddings,
        metadatas=metadatas,
        ids=ids,
    )


def query_collection(
    collection: chromadb.Collection,
    query_embedding: list[float],
    top_k: int = 3,
) -> dict:
    """
    用 query 向量在 collection 中搜索最相似的文档。

    参数:
        collection: ChromaDB collection 对象
        query_embedding: 用户问题的 embedding 向量
        top_k: 返回最相似的前 k 个结果

    返回:
        ChromaDB 的查询结果字典，包含:
        - "documents": 匹配的原始文本
        - "distances": 距离分数（cosine 距离，越小越相似）
        - "metadatas": 元数据

    注意：ChromaDB 返回的是"距离"而不是"相似度"
    - cosine distance = 1 - cosine similarity
    - 距离越小 = 越相似
    - 距离 0 = 完全相同，距离 2 = 完全相反
    """
    results = collection.query(
        query_embeddings=[query_embedding],
        n_results=top_k,
        include=["documents", "distances", "metadatas"],
    )
    return results
