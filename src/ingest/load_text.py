"""
Text Loader — 从文件夹中加载 .txt 文件
========================================

最简单的 loader：读取纯文本文件，返回文本内容 + metadata。

面试考点：
- 每个 loader 的职责是把"原始文件"变成"统一的 (text, metadata) 格式"
- metadata 记录来源信息（文件名、文件类型），后续回答时用于引用溯源
- 编码问题：不同系统可能产生 UTF-8 / GBK / Latin-1 编码的文件
  这里用 errors="replace" 做容错处理，避免因单个文件编码问题导致整个 pipeline 崩溃
"""

from __future__ import annotations

import os


def load_text_files(directory: str) -> list[dict]:
    """
    加载目录下所有 .txt 文件。

    参数:
        directory: 文件夹路径（例如 "data/raw"）

    返回:
        列表，每个元素是一个 dict:
        {
            "text": "文件的完整内容",
            "metadata": {"source": "文件名", "type": "txt"}
        }
    """
    results = []

    if not os.path.isdir(directory):
        return results

    for filename in sorted(os.listdir(directory)):
        if not filename.lower().endswith(".txt"):
            continue

        filepath = os.path.join(directory, filename)
        with open(filepath, "r", encoding="utf-8", errors="replace") as f:
            text = f.read().strip()

        if text:
            results.append({
                "text": text,
                "metadata": {"source": filename, "type": "txt"},
            })

    return results
