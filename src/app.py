"""
Personal RAG Assistant — Phase 3
====================================

完整 pipeline：
1. 自动扫描 data/raw/ 下的 .txt, .md, .pdf 文件
2. Chunking：RecursiveCharacterTextSplitter 或 SemanticChunker
3. Embedding + 存入 ChromaDB
4. 用户提问 → 检索 → Streaming 生成答案
5. Cost & Latency 实时监控

运行方式:
    cd personal_rag_assistant
    python -m src.app
"""

import os

from src.config import TOP_K, SCORE_THRESHOLD
from src.embeddings.embedding_model import get_embeddings, get_single_embedding
from src.storage.vector_store import (
    get_or_create_collection,
    add_documents,
    query_collection,
)
from src.generation.prompt_builder import build_prompt
from src.generation.llm_client import generate_answer
from src.ingest.load_text import load_text_files
from src.ingest.load_markdown import load_markdown_files
from src.ingest.load_pdf import load_pdf_files
from src.chunking.chunk_text import chunk_documents
from src.utils.cost_tracker import get_tracker


# ============================================================
#  OFFLINE PHASE: 数据准备 + 向量化 + 存储
# ============================================================

_BASE_DIR = os.path.dirname(os.path.dirname(os.path.abspath(__file__)))
_RAW_DIR = os.path.join(_BASE_DIR, "data", "raw")


def load_all_documents() -> list[dict]:
    """
    自动扫描 data/raw/ 目录，加载所有支持的文件格式。

    返回:
        统一格式的文档列表:
        [{"text": "...", "metadata": {"source": "...", "type": "..."}}, ...]

    后续 Phase 会在这里加 PDF / CSV / Image loader。
    """
    docs = []
    docs.extend(load_text_files(_RAW_DIR))
    docs.extend(load_markdown_files(_RAW_DIR))
    docs.extend(load_pdf_files(_RAW_DIR))

    if not docs:
        print("   ⚠️  data/raw/ 中没有找到文件，使用内置示例数据")
        docs = [
            {"text": "My name is Jerry. I live in New York City. I work as a software engineer.",
             "metadata": {"source": "about_me (hardcoded)", "type": "hardcoded"}},
            {"text": "My weekly schedule: Monday standup at 9:30 AM, gym at 6 PM. Tuesday one-on-one with manager at 10 AM.",
             "metadata": {"source": "schedule (hardcoded)", "type": "hardcoded"}},
            {"text": "My 2024 goals: get promoted to senior engineer, run a half marathon, save 20% of income.",
             "metadata": {"source": "goals (hardcoded)", "type": "hardcoded"}},
        ]

    return docs


def ingest_documents() -> None:
    """
    离线阶段核心流程：
    加载文档 → Chunking → Embedding → 存入 Vector DB
    """
    print("=" * 60)
    print("OFFLINE PHASE: 文档导入 (Document Ingestion)")
    print("=" * 60)

    # Step 1: 加载所有文档
    raw_docs = load_all_documents()
    print(f"\n📄 加载了 {len(raw_docs)} 个文档:")
    for i, doc in enumerate(raw_docs):
        preview = doc["text"][:80].replace("\n", " ") + "..."
        print(f"   [{i + 1}] {doc['metadata']['source']}: {preview}")

    # Step 2: Chunking — 把长文档切成小段落
    print(f"\n✂️  正在进行 Chunking...")
    chunks, metadatas = chunk_documents(raw_docs)
    print(f"   {len(raw_docs)} 个文档 → {len(chunks)} 个 chunks")

    # Step 3: Embedding — 把每个 chunk 变成向量
    print(f"\n🔢 正在生成 embedding 向量...")
    embeddings = get_embeddings(chunks)
    print(f"   生成了 {len(embeddings)} 个向量，每个向量 {len(embeddings[0])} 维")

    # Step 4: 存入 Vector DB
    print(f"\n💾 正在存入 ChromaDB...")
    collection = get_or_create_collection()
    add_documents(
        collection=collection,
        documents=chunks,
        embeddings=embeddings,
        metadatas=metadatas,
    )
    print(f"   ChromaDB collection '{collection.name}' 现在有 {collection.count()} 条记录")
    print(f"\n✅ 离线阶段完成！文档已经可以被检索了。\n")


# ============================================================
#  ONLINE PHASE: 用户提问 → 检索 → 生成答案
# ============================================================

def ask_question(question: str) -> str:
    """
    在线阶段的核心流程：提问 → 向量化 → 检索 → 构建 prompt → LLM 生成答案

    这就是 RAG 的 "online phase"：
    用户问题 → query embedding → vector search → 取 top-k → 构建 prompt → LLM → 答案
    """
    print("=" * 60)
    print("ONLINE PHASE: 问答 (Question Answering)")
    print("=" * 60)
    print(f"\n❓ 用户问题: {question}")

    # Step 1: 把用户问题转成 embedding 向量
    # 用同一个 embedding model 确保问题和文档在同一个向量空间中
    # 面试考点：query 和 document 必须用同一个 embedding model！
    print(f"\n🔢 正在对问题生成 embedding...")
    query_embedding = get_single_embedding(question)
    print(f"   问题向量维度: {len(query_embedding)}")

    # Step 2: 在 Vector DB 中检索最相似的文档
    # ChromaDB 会计算 query 向量和所有文档向量的 cosine distance
    # 返回距离最小（最相似）的 top_k 个结果
    print(f"\n🔍 正在检索相关文档 (top_k={TOP_K})...")
    collection = get_or_create_collection()
    results = query_collection(
        collection=collection,
        query_embedding=query_embedding,
        top_k=TOP_K,
    )

    # Step 3: 解析检索结果
    retrieved_docs = results["documents"][0]       # ChromaDB 返回嵌套列表，[0] 取第一个 query 的结果
    distances = results["distances"][0]             # cosine distance: 越小越相似
    retrieved_metas = results["metadatas"][0]

    # 打印检索结果（调试时非常有用！）
    print(f"\n📋 检索到 {len(retrieved_docs)} 个相关文档:")
    for i, (doc, dist, meta) in enumerate(zip(retrieved_docs, distances, retrieved_metas)):
        # cosine similarity = 1 - cosine distance
        similarity = 1 - dist
        preview = doc[:100].replace("\n", " ") + "..."
        status = "✅" if similarity >= SCORE_THRESHOLD else "❌ (低于阈值)"
        print(f"   [{i + 1}] similarity={similarity:.4f} {status}")
        print(f"       source: {meta.get('source', 'unknown')}")
        print(f"       preview: {preview}")

    # Step 4: Score threshold 过滤
    # 过滤掉相似度太低的结果，避免不相关的内容污染 prompt
    filtered_docs = []
    filtered_metas = []
    for doc, dist, meta in zip(retrieved_docs, distances, retrieved_metas):
        similarity = 1 - dist
        if similarity >= SCORE_THRESHOLD:
            filtered_docs.append(doc)
            filtered_metas.append(meta)

    if not filtered_docs:
        print(f"\n⚠️  所有检索结果的相似度都低于阈值 ({SCORE_THRESHOLD})，没有找到相关文档。")
        return "I don't have enough information in your documents to answer this question."

    print(f"\n   经过 score threshold ({SCORE_THRESHOLD}) 过滤后，保留 {len(filtered_docs)} 个文档")

    # Step 5: 构建 prompt
    # 把检索到的文档和用户问题组合成最终的 prompt
    print(f"\n📝 正在构建 prompt...")
    messages = build_prompt(
        user_question=question,
        retrieved_chunks=filtered_docs,
        chunk_metadatas=filtered_metas,
    )
    # 打印 prompt（调试时查看 LLM 实际看到了什么）
    print(f"   System prompt: {len(messages[0]['content'])} 字符")
    print(f"   User message: {len(messages[1]['content'])} 字符")

    # Step 6: 调用 LLM 生成答案（streaming 模式，逐 token 输出）
    print(f"\n🤖 LLM 答案:")
    print("-" * 60)
    tracker = get_tracker()
    records_before = len(tracker.records)
    answer = generate_answer(messages)
    print("-" * 60)

    # 显示本次问答的 cost/latency
    new_records = tracker.records[records_before:]
    q_cost = sum(r.estimated_cost for r in new_records)
    q_latency = sum(r.latency_ms for r in new_records)
    q_tokens = sum(r.input_tokens + r.output_tokens for r in new_records)
    print(f"   💰 本次: ${q_cost:.6f} | {q_latency:.0f}ms | {q_tokens} tokens\n")

    return answer


# ============================================================
#  主程序：交互式问答循环
# ============================================================

def main():
    """
    Phase 1 主程序：
    1. 先执行离线阶段（ingest 文档）
    2. 进入交互式问答循环
    """
    print("\n🚀 Personal RAG Assistant — Phase 3: PDF + Cost Monitor")
    print("=" * 60)

    # 离线阶段：导入文档
    ingest_documents()

    tracker = get_tracker()
    print(f"   💰 Ingest 阶段花费: ${tracker.total_cost:.6f} "
          f"({tracker.total_latency_ms / 1000:.1f}s)")

    # 在线阶段：交互式问答
    print("\n💡 现在你可以开始提问了！")
    print("   /chunks [compare]  — 查看/对比 chunks")
    print("   /cost              — 查看累计花费和延迟")
    print("   quit               — 退出\n")

    while True:
        try:
            question = input("You: ").strip()
        except (EOFError, KeyboardInterrupt):
            print("\n👋 Bye!")
            break

        if not question:
            continue
        if question.lower() in ("quit", "exit", "q"):
            tracker.print_summary()
            print("👋 Bye!")
            break

        if question.startswith("/chunks"):
            _handle_chunks_command(question)
            continue
        if question == "/cost":
            tracker.print_summary()
            continue

        ask_question(question)


def _handle_chunks_command(command: str) -> None:
    """处理 /chunks 调试命令。"""
    from src.chunking.preview_chunks import print_chunks, compare_strategies

    raw_docs = load_all_documents()
    parts = command.split()

    if len(parts) >= 2 and parts[1] == "compare":
        compare_strategies(raw_docs)
    else:
        strategy = parts[1] if len(parts) >= 2 else None
        label = strategy or "当前策略"
        chunks, metas = chunk_documents(raw_docs, strategy=strategy)
        print_chunks(chunks, metas, label)


if __name__ == "__main__":
    main()
