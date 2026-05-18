"""
Embedding 模块 — 把文本变成向量
================================

核心概念：
- Embedding 就是把一段文字变成一组数字（向量），比如 1536 维的浮点数数组
- 语义相近的文本，embedding 向量在空间中也很近
- 例如："我喜欢吃苹果" 和 "我爱吃水果" 的向量距离会很近
- 而 "我喜欢吃苹果" 和 "量子力学" 的向量距离会很远

面试考点：
- Embedding model ≠ LLM
  - Embedding model: 输入文本 → 输出固定长度的向量（只做编码，不生成文字）
  - LLM: 输入文本 → 输出文本（可以生成、对话、推理）
- 为什么需要 embedding？因为计算机不能直接计算两段文字的"相似度"，
  但可以计算两个向量的距离（cosine similarity, dot product 等）

本模块设计原则：
- 用一个统一的 get_embeddings() 函数，内部可以切换不同的 embedding 提供商
- Phase 1 先用 OpenAI，后续 Phase 可以加本地模型（sentence-transformers 等）
"""

from openai import OpenAI
from src.config import OPENAI_API_KEY, EMBEDDING_MODEL
from src.utils.cost_tracker import get_tracker, Timer


# 创建 OpenAI 客户端（全局复用，避免每次调用都创建新连接）
_client = OpenAI(api_key=OPENAI_API_KEY)


def get_embeddings(texts: list[str]) -> list[list[float]]:
    """
    把一组文本转换为 embedding 向量。

    参数:
        texts: 要转换的文本列表，例如 ["你好世界", "今天天气不错"]

    返回:
        向量列表，每个向量是一个 float 数组
        例如 [[0.01, -0.03, ...], [0.05, 0.02, ...]]

    为什么接受 list 而不是单个 string？
    - 批量请求比逐个请求更高效（减少网络往返次数）
    - OpenAI 的 embedding API 本身就支持批量输入
    """
    with Timer() as t:
        response = _client.embeddings.create(
            input=texts,
            model=EMBEDDING_MODEL,
        )

    embeddings = [item.embedding for item in response.data]

    get_tracker().record_embedding(
        model=EMBEDDING_MODEL,
        input_tokens=response.usage.total_tokens,
        latency_ms=t.elapsed_ms,
    )

    return embeddings


def get_single_embedding(text: str) -> list[float]:
    """
    把单个文本转换为 embedding 向量（便捷方法）。
    常用于把用户的 query 转成向量，然后和文档向量做相似度比较。
    """
    return get_embeddings([text])[0]