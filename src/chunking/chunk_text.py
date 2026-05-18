"""
Chunking 模块 — 把长文本切分成小段落
======================================

支持两种策略，可以在 config.py 中通过 CHUNK_STRATEGY 切换：

1. recursive（默认）— RecursiveCharacterTextSplitter
   原理：按优先级尝试不同的分隔符 \\n\\n → \\n → " " → ""
   优点：速度快、零额外成本、效果已经不错
   缺点：纯基于字符位置，不理解语义

2. semantic — SemanticChunker (langchain_experimental)
   原理：先把文本拆成句子 → 计算每句话的 embedding → 相邻句子相似度低于阈值时切分
   优点：切分点在"语义转折处"，每个 chunk 内部主题一致
   缺点：需要调用 embedding API（有成本和延迟）

面试考点：
- Chunking 策略直接影响检索质量，是 RAG 调优的关键参数
- Recursive: 工业界最常用的 baseline，简单高效
- Semantic: 学术界推崇，但实际提升取决于文档类型
  - 对主题切换明显的长文档（笔记、会议记录）效果好
  - 对结构清晰的文档（有标题的 Markdown）提升不大
- 实际生产中常见做法：先用 recursive 跑通，再用 semantic 做 A/B 测试
"""

from __future__ import annotations

from langchain_text_splitters import RecursiveCharacterTextSplitter
from src.config import CHUNK_SIZE, CHUNK_OVERLAP, CHUNK_STRATEGY


def chunk_documents(
    documents: list[dict],
    strategy: str | None = None,
) -> tuple[list[str], list[dict]]:
    """
    对一组文档执行 chunking。

    参数:
        documents: load_text / load_markdown 返回的列表
                   每个元素: {"text": "...", "metadata": {...}}
        strategy: 覆盖 config 中的默认策略（"recursive" 或 "semantic"）

    返回:
        (chunks, metadatas)
    """
    strategy = strategy or CHUNK_STRATEGY

    if strategy == "semantic":
        return _chunk_semantic(documents)
    return _chunk_recursive(documents)


# ============================================================
#  Strategy 1: Recursive Character Splitting
# ============================================================

def _chunk_recursive(documents: list[dict]) -> tuple[list[str], list[dict]]:
    splitter = RecursiveCharacterTextSplitter(
        chunk_size=CHUNK_SIZE,
        chunk_overlap=CHUNK_OVERLAP,
        length_function=len,
        separators=["\n\n", "\n", " ", ""],
    )

    all_chunks = []
    all_metadatas = []

    for doc in documents:
        text = doc["text"]
        meta = doc["metadata"]
        chunks = splitter.split_text(text)

        for i, chunk in enumerate(chunks):
            all_chunks.append(chunk)
            all_metadatas.append({
                **meta,
                "chunk_index": i,
                "total_chunks": len(chunks),
                "strategy": "recursive",
            })

    return all_chunks, all_metadatas


# ============================================================
#  Strategy 2: Semantic Chunking
# ============================================================

def _chunk_semantic(documents: list[dict]) -> tuple[list[str], list[dict]]:
    """
    用 SemanticChunker 按语义边界切分。

    工作原理：
    1. 把文本拆成句子
    2. 对每个句子做 embedding
    3. 计算相邻句子的 cosine similarity
    4. 在相似度骤降的地方切分（说明话题发生了转折）

    breakpoint_threshold_type 选项：
    - "percentile": 在相似度最低的 X% 处切分（默认）
    - "standard_deviation": 低于均值 - N*标准差处切分
    - "interquartile": 用四分位距判断异常低的相似度
    """
    from langchain_experimental.text_splitter import SemanticChunker
    from langchain_openai import OpenAIEmbeddings
    from src.config import OPENAI_API_KEY

    embeddings = OpenAIEmbeddings(
        model="text-embedding-3-small",
        openai_api_key=OPENAI_API_KEY,
    )

    chunker = SemanticChunker(
        embeddings=embeddings,
        breakpoint_threshold_type="percentile",
        breakpoint_threshold_amount=70,
    )

    all_chunks = []
    all_metadatas = []

    for doc in documents:
        text = doc["text"]
        meta = doc["metadata"]

        lc_docs = chunker.create_documents([text])
        chunks = [d.page_content for d in lc_docs]

        for i, chunk in enumerate(chunks):
            all_chunks.append(chunk)
            all_metadatas.append({
                **meta,
                "chunk_index": i,
                "total_chunks": len(chunks),
                "strategy": "semantic",
            })

    return all_chunks, all_metadatas
