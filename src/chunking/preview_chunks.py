"""
Chunk Preview 工具 — 查看和对比不同 chunking 策略的效果
========================================================

运行方式:
    python -m src.chunking.preview_chunks                  # 只看当前策略
    python -m src.chunking.preview_chunks --compare         # 对比 recursive vs semantic
    python -m src.chunking.preview_chunks --strategy semantic  # 只看 semantic

这个工具帮你回答：
1. 每个 chunk 的内容是什么？长度多少？
2. chunk 边界切在了哪里？有没有切断语义？
3. recursive 和 semantic 策略的切分结果有什么区别？

调试 chunking 效果是 RAG 调优的第一步。
"""

from __future__ import annotations

import argparse
import os
import sys

sys.path.insert(0, os.path.dirname(os.path.dirname(os.path.dirname(os.path.abspath(__file__)))))

from src.ingest.load_text import load_text_files
from src.ingest.load_markdown import load_markdown_files
from src.chunking.chunk_text import chunk_documents


_BASE_DIR = os.path.dirname(os.path.dirname(os.path.dirname(os.path.abspath(__file__))))
_RAW_DIR = os.path.join(_BASE_DIR, "data", "raw")


def _load_docs() -> list[dict]:
    docs = []
    docs.extend(load_text_files(_RAW_DIR))
    docs.extend(load_markdown_files(_RAW_DIR))
    return docs


def print_chunks(chunks: list[str], metadatas: list[dict], strategy_name: str) -> None:
    """格式化打印 chunk 列表。"""
    print(f"\n{'=' * 70}")
    print(f"  策略: {strategy_name}  |  共 {len(chunks)} 个 chunks")
    print(f"{'=' * 70}")

    current_source = None
    for i, (chunk, meta) in enumerate(zip(chunks, metadatas)):
        source = meta.get("source", "unknown")
        if source != current_source:
            current_source = source
            print(f"\n  📄 文件: {source}")
            print(f"  {'─' * 60}")

        char_count = len(chunk)
        ci = meta.get("chunk_index", "?")
        total = meta.get("total_chunks", "?")

        print(f"\n  ┌─ Chunk {ci + 1}/{total}  ({char_count} chars)")
        for line in chunk.split("\n"):
            print(f"  │ {line}")
        print(f"  └{'─' * 50}")

    avg_len = sum(len(c) for c in chunks) / len(chunks) if chunks else 0
    min_len = min(len(c) for c in chunks) if chunks else 0
    max_len = max(len(c) for c in chunks) if chunks else 0
    print(f"\n  📊 统计: {len(chunks)} chunks | "
          f"avg={avg_len:.0f} chars | min={min_len} | max={max_len}")
    print()


def compare_strategies(docs: list[dict]) -> None:
    """并排对比 recursive 和 semantic 两种策略。"""
    print("\n🔄 正在用 recursive 策略切分...")
    r_chunks, r_metas = chunk_documents(docs, strategy="recursive")
    print_chunks(r_chunks, r_metas, "Recursive (按字符+分隔符)")

    print("🔄 正在用 semantic 策略切分（需要调用 embedding API）...")
    s_chunks, s_metas = chunk_documents(docs, strategy="semantic")
    print_chunks(s_chunks, s_metas, "Semantic (按语义相似度)")

    # 总结对比
    print("=" * 70)
    print("  📊 对比总结")
    print("=" * 70)
    r_avg = sum(len(c) for c in r_chunks) / len(r_chunks) if r_chunks else 0
    s_avg = sum(len(c) for c in s_chunks) / len(s_chunks) if s_chunks else 0
    print(f"  {'':>20} {'Recursive':>12} {'Semantic':>12}")
    print(f"  {'Chunk 数量':>20} {len(r_chunks):>12} {len(s_chunks):>12}")
    print(f"  {'平均长度 (chars)':>20} {r_avg:>12.0f} {s_avg:>12.0f}")
    print(f"  {'最短 chunk':>20} {min(len(c) for c in r_chunks):>12} {min(len(c) for c in s_chunks):>12}")
    print(f"  {'最长 chunk':>20} {max(len(c) for c in r_chunks):>12} {max(len(c) for c in s_chunks):>12}")
    print()


def main():
    parser = argparse.ArgumentParser(description="预览 chunking 效果")
    parser.add_argument("--strategy", choices=["recursive", "semantic"],
                        default=None, help="指定策略（默认用 config 中的设置）")
    parser.add_argument("--compare", action="store_true",
                        help="对比 recursive 和 semantic 两种策略")
    args = parser.parse_args()

    docs = _load_docs()
    if not docs:
        print("❌ data/raw/ 中没有找到任何 .txt 或 .md 文件")
        return

    print(f"\n📄 加载了 {len(docs)} 个文档:")
    for d in docs:
        print(f"   - {d['metadata']['source']} ({len(d['text'])} chars)")

    if args.compare:
        compare_strategies(docs)
    else:
        strategy = args.strategy
        label = strategy or "config default"
        print(f"\n🔄 正在用 {label} 策略切分...")
        chunks, metas = chunk_documents(docs, strategy=strategy)
        print_chunks(chunks, metas, label)


if __name__ == "__main__":
    main()
