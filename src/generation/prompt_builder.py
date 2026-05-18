from __future__ import annotations

"""
Prompt Builder 模块 — 构建最终发给 LLM 的 prompt
===================================================

核心概念：
- RAG 的最后一步是把"用户问题"和"检索到的文档"拼接成一个 prompt
- Prompt 的质量直接决定 LLM 回答的质量
- 一个好的 RAG prompt 包含：
  1. System instruction: 告诉 LLM 它的角色和规则
  2. Retrieved context: 检索到的相关文档（这就是 RAG 的"R"的产物）
  3. User question: 用户的原始问题
  4. Answer format instruction: 回答格式要求

面试考点：
- 为什么不直接把用户问题发给 LLM？
  → 因为 LLM 不知道你的私有数据，会产生 hallucination（幻觉/编造）
- 为什么不把所有文档都塞进 prompt？
  → Token limit（上下文窗口有限）、成本高、噪声多反而影响回答质量
- RAG 的核心价值：只把最相关的几段文字放进 prompt，既省 token 又减少幻觉
"""


# System prompt 模板：定义 LLM 的角色和行为规则
SYSTEM_PROMPT = """You are a helpful personal life assistant. Your job is to answer questions based on the user's personal documents and data.

Rules:
- Answer ONLY based on the provided context when the question is about the user's documents.
- If the context contains relevant information, use it to give a clear, concise answer.
- Cite the source (filename, page, section) when available in the metadata.
- If the context does not contain enough information to answer, say: "Based on your documents, I don't have enough information to answer this question."
- Do NOT make up information that is not in the context.
- You may use your general knowledge for greetings or simple questions unrelated to user documents."""


def build_prompt(
    user_question: str,
    retrieved_chunks: list[str],
    chunk_metadatas: list[dict] | None = None,
) -> list[dict]:
    """
    构建发给 LLM 的完整 messages 列表。

    参数:
        user_question: 用户的原始问题
        retrieved_chunks: 检索到的文本 chunk 列表
        chunk_metadatas: 每个 chunk 的元数据（来源、页码等）

    返回:
        OpenAI Chat API 格式的 messages 列表:
        [{"role": "system", ...}, {"role": "user", ...}]

    Prompt 的结构设计思路：
    - System message: 角色定义 + 行为规则（全局有效）
    - User message: 上下文 + 问题 + 格式指令（每次对话变化）
    """
    # 拼接检索到的文档，作为 context
    # 每个 chunk 前面加序号和来源信息，方便 LLM 引用
    context_parts = []
    for i, chunk in enumerate(retrieved_chunks):
        source_info = ""
        if chunk_metadatas and i < len(chunk_metadatas):
            meta = chunk_metadatas[i]
            # 从 metadata 中提取来源信息
            source = meta.get("source", "unknown")
            page = meta.get("page", "")
            section = meta.get("section", "")
            source_info = f" [Source: {source}"
            if page:
                source_info += f", Page: {page}"
            if section:
                source_info += f", Section: {section}"
            source_info += "]"

        context_parts.append(f"[Document {i + 1}]{source_info}\n{chunk}")

    # 用分隔线连接所有 chunk
    context_text = "\n\n---\n\n".join(context_parts)

    # 组装 user message：context + 问题
    user_message = f"""Here is the relevant context from my personal documents:

{context_text}

---

My question: {user_question}

Please answer based on the context above. Cite sources when possible."""

    # 返回 OpenAI Chat API 格式的 messages
    messages = [
        {"role": "system", "content": SYSTEM_PROMPT},
        {"role": "user", "content": user_message},
    ]

    return messages
