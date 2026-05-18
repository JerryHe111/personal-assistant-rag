"""
Markdown Loader — 从文件夹中加载 .md 文件
==========================================

Markdown 和 TXT 的区别：
- Markdown 有结构信息（标题 #、列表 -、代码块 ```）
- 我们可以利用标题作为 metadata，帮助 LLM 理解 chunk 所在的上下文
- 例如：一个 chunk 如果来自 "## 2024 Goals > ### Career" 这个章节，
  LLM 就知道这段话是关于"2024 年职业目标"的

面试考点：
- 为什么 metadata 很重要？
  → 相同的文字在不同上下文中含义不同。metadata 提供了上下文线索
- Section-aware loading 的好处：
  → 后续 chunking 时可以按 section 切分，避免把不同主题的内容混在一个 chunk 里
"""

from __future__ import annotations

import os
import re


def load_markdown_files(directory: str) -> list[dict]:
    """
    加载目录下所有 .md 文件。

    和 load_text 不同的是，这里会：
    1. 提取顶级标题（第一个 # 标题）作为 metadata 中的 title
    2. 保留完整原文（chunking 交给 chunk_text 模块处理）

    参数:
        directory: 文件夹路径

    返回:
        列表，每个元素:
        {
            "text": "Markdown 文件完整内容",
            "metadata": {"source": "文件名", "type": "markdown", "title": "文档标题"}
        }
    """
    results = []

    if not os.path.isdir(directory):
        return results

    for filename in sorted(os.listdir(directory)):
        if not filename.lower().endswith(".md"):
            continue

        filepath = os.path.join(directory, filename)
        with open(filepath, "r", encoding="utf-8", errors="replace") as f:
            text = f.read().strip()

        if not text:
            continue

        title = _extract_title(text, filename)

        results.append({
            "text": text,
            "metadata": {"source": filename, "type": "markdown", "title": title},
        })

    return results


def _extract_title(text: str, fallback: str) -> str:
    """从 Markdown 中提取第一个 # 标题，找不到则用文件名。"""
    match = re.search(r"^#\s+(.+)", text, re.MULTILINE)
    if match:
        return match.group(1).strip()
    return fallback
