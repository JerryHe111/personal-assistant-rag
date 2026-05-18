"""
PDF Loader — 从文件夹中加载 .pdf 文件
=======================================

核心概念：
- PDF 是最常见也最难解析的文档格式
- 挑战：多栏布局、表格、图片、扫描版、加密
- 本模块用 PyMuPDF (fitz) 做基础文本提取，速度快、依赖少

工具选型对比：
- PyMuPDF (fitz): 速度最快，纯文本提取效果好，本项目的选择
- pdfplumber:     表格提取更好，但速度慢一些
- PyPDF2:        纯 Python，轻量，但复杂 PDF 效果差
- LlamaParse:    云端服务，效果最好，但收费 + 需要网络
- Unstructured:  大而全的框架，支持多种格式，但重量级

面试考点：
- 为什么 PDF 解析是 RAG 的痛点？
  → PDF 是"展示格式"而非"存储格式"，它记录的是"在哪里画什么字"
    而不是"这段文字属于哪个段落"。重建结构很难
- metadata 的重要性：记录页码，回答时可以说"根据第 X 页..."
- 扫描版 PDF 需要 OCR（Phase 5 才涉及）
"""

from __future__ import annotations

import os

import fitz  # PyMuPDF


def load_pdf_files(directory: str) -> list[dict]:
    """
    加载目录下所有 .pdf 文件，按页提取文本。

    每一页作为一个独立的文档条目（而不是整个 PDF 作为一个文档），
    因为：
    1. 页码信息是很好的 metadata（方便引用"第 X 页"）
    2. 长 PDF 整体作为一个 chunk 太大了
    3. 即使后续还会再 chunking，按页切分也是合理的初始粒度

    参数:
        directory: 文件夹路径

    返回:
        列表，每个元素:
        {
            "text": "这一页的文本内容",
            "metadata": {"source": "文件名", "type": "pdf", "page": 页码(1-indexed)}
        }
    """
    results = []

    if not os.path.isdir(directory):
        return results

    for filename in sorted(os.listdir(directory)):
        if not filename.lower().endswith(".pdf"):
            continue

        filepath = os.path.join(directory, filename)
        try:
            doc = fitz.open(filepath)
        except Exception as e:
            print(f"   ⚠️  无法打开 PDF: {filename} ({e})")
            continue

        for page_num in range(len(doc)):
            page = doc[page_num]
            text = page.get_text("text").strip()

            if not text:
                continue

            results.append({
                "text": text,
                "metadata": {
                    "source": filename,
                    "type": "pdf",
                    "page": page_num + 1,
                    "total_pages": len(doc),
                },
            })

        doc.close()

    return results
