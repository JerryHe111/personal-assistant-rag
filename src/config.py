"""
全局配置模块 — 管理所有 API key 和参数
===========================================
这个文件的作用：把所有"可调节的旋钮"集中在一个地方。
在 RAG 系统中，有很多参数会影响效果（chunk_size, top_k, model 选择等），
统一管理可以方便实验和调试。
"""

import os
from dotenv import load_dotenv

# 从 .env 文件加载环境变量（API key 等敏感信息不应该硬编码在代码里）
load_dotenv()


# ============================================================
#  API Keys
# ============================================================
OPENAI_API_KEY = os.getenv("OPENAI_API_KEY", "")

# ============================================================
#  Embedding 配置
# ============================================================
# Embedding model 负责把文本变成向量（一组数字）
# text-embedding-3-small: OpenAI 最便宜的 embedding model，维度 1536
# 面试考点：embedding model ≠ LLM，embedding model 只做"文本→向量"，不生成文字
EMBEDDING_MODEL = os.getenv("EMBEDDING_MODEL", "text-embedding-3-small")

# ============================================================
#  LLM 配置
# ============================================================
# LLM 负责最终的答案生成
LLM_MODEL = os.getenv("LLM_MODEL", "gpt-4o-mini")
LLM_TEMPERATURE = 0.3  # 低 temperature = 更稳定、更少随机性，适合问答场景
LLM_MAX_TOKENS = 1024  # 限制回答长度，避免 token 浪费

# ============================================================
#  Chunking 配置
# ============================================================
CHUNK_SIZE = 500       # 每个 chunk 的最大字符数
CHUNK_OVERLAP = 50     # 相邻 chunk 之间的重叠字符数（避免在边界处切断语义）
# 策略选择: "recursive"（按字符+分隔符） 或 "semantic"（按语义相似度）
CHUNK_STRATEGY = os.getenv("CHUNK_STRATEGY", "recursive")

# ============================================================
#  Retrieval 配置
# ============================================================
TOP_K = 3              # 检索时返回最相关的前 k 个 chunk
SCORE_THRESHOLD = 0.1  # 相似度低于此值的结果会被过滤掉（0~1，越高越严格）

# ============================================================
#  ChromaDB 配置
# ============================================================
CHROMA_PERSIST_DIR = os.getenv("CHROMA_PERSIST_DIR", "./data/chroma_db")
CHROMA_COLLECTION_NAME = "personal_docs"  # ChromaDB 中的 collection 名称（类似数据库的表名）
