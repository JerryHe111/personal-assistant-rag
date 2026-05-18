# RAG 学习路线图 — 从零到个人生活助手

---

## Phase 0: RAG 概念全景图 (Concept Map)

### 什么是 RAG？

RAG = **Retrieval-Augmented Generation**（检索增强生成）

传统 LLM 的问题：LLM 只知道训练数据里的内容，不了解你的个人文档、笔记、数据库。RAG 的核心思想是：**先从你的私有数据中检索（Retrieve）相关内容，再把这些内容喂给 LLM 生成（Generate）答案。**

### RAG 完整流程图

```
┌─────────────────────────────────────────────────────────────┐
│                    OFFLINE PHASE（离线阶段）                  │
│                                                             │
│  原始文档 → 解析/清洗 → Chunking → Embedding → Vector DB    │
│  (PDF/CSV/  (提取文本   (切分为     (文本变成    (存储向量     │
│   MD/TXT/    去噪声)     小段落)     数字向量)    +元数据)     │
│   Image)                                                    │
└─────────────────────────────────────────────────────────────┘

┌─────────────────────────────────────────────────────────────┐
│                    ONLINE PHASE（在线阶段）                   │
│                                                             │
│  用户提问 → Query Rewrite → Recall(检索) → Rerank(重排序)   │
│            (改写查询)     (向量搜索      (精排，选出        │
│                          +关键词搜索)    最相关的)          │
│                                                             │
│  → Top-k / Score Threshold → 构建 Prompt → LLM 生成答案    │
│    (选出最好的k个结果)      (拼接上下文)   (基于证据回答)    │
└─────────────────────────────────────────────────────────────┘
```

---

## 各阶段详细规划

### Phase 1: 最小可运行的 RAG (Minimal RAG)

| 项目 | 说明 |
|------|------|
| **构建内容** | 用几段硬编码的文本，完成"文本→embedding→存储→检索→回答"的完整流程 |
| **为什么重要** | 让你在 30 分钟内看到 RAG 的完整 pipeline 跑通，建立直觉 |
| **创建文件** | `src/config.py`, `src/app.py`, `requirements.txt`, `.env.example`, `README.md` |
| **使用工具** | `chromadb`（最简单的本地 vector DB）, `openai`（embedding + LLM）|
| **可能遇到的问题** | API key 配置错误；embedding 维度不匹配；Chroma 版本 API 变化 |
| **调试方法** | 打印每一步的输出：chunk 内容、embedding 维度、检索结果和 score |
| **面试要能解释** | RAG 的整体流程；为什么不直接把所有文档塞进 prompt；embedding 的基本概念 |

---

### Phase 2: 文档导入 — Markdown / TXT

| 项目 | 说明 |
|------|------|
| **构建内容** | 从 `data/raw/` 文件夹读取 `.md` 和 `.txt` 文件，自动解析并 ingest |
| **为什么重要** | 真实 RAG 系统的第一步是自动化 data ingestion（数据摄入） |
| **创建文件** | `src/ingest/load_text.py`, `src/ingest/load_markdown.py`, `src/chunking/chunk_text.py` |
| **使用工具** | Python 内置文件读取, `langchain.text_splitter.RecursiveCharacterTextSplitter` |
| **可能遇到的问题** | 编码问题（UTF-8 vs GBK）；chunk 太大或太小导致检索质量差 |
| **调试方法** | 打印每个 chunk 的内容和长度；手动检查 chunk 边界是否切断了语义 |
| **面试要能解释** | Chunking 的原理和参数选择；`chunk_size` 和 `chunk_overlap` 的权衡；为什么 `RecursiveCharacterTextSplitter` 比简单按字符切更好 |

---

### Phase 3: PDF 解析

| 项目 | 说明 |
|------|------|
| **构建内容** | 解析 PDF 文件，提取文本、保留页码等 metadata |
| **为什么重要** | PDF 是最常见的文档格式，但也是最难解析的（布局、表格、图片混排） |
| **创建文件** | `src/ingest/load_pdf.py` |
| **使用工具** | `PyMuPDF`（速度快）, `pdfplumber`（表格好）, 可选 `LlamaParse`（云端高质量解析） |
| **可能遇到的问题** | 扫描版 PDF 无法直接提取文本；表格提取乱序；多栏布局文本顺序错乱 |
| **调试方法** | 对比原 PDF 和提取文本；检查页码 metadata 是否正确 |
| **面试要能解释** | 不同 PDF 解析工具的优缺点；为什么 PDF 解析是 RAG 的痛点；metadata 的重要性 |

---

### Phase 4: CSV / 表格支持

| 项目 | 说明 |
|------|------|
| **构建内容** | 将 CSV 文件转换为可检索的文本 chunk |
| **为什么重要** | 结构化数据（账单、日程、联系人）是个人助手的核心数据源 |
| **创建文件** | `src/ingest/load_csv.py` |
| **使用工具** | `pandas` |
| **可能遇到的问题** | 表格行之间缺少上下文；数字数据 embedding 效果差；列名不清晰 |
| **调试方法** | 打印转换后的文本，确认是否保留了表头和行的语义关系 |
| **面试要能解释** | 结构化数据和非结构化数据在 RAG 中的区别；Table-aware chunking 的策略 |

---

### Phase 5: 图片支持 (OCR / Vision)

| 项目 | 说明 |
|------|------|
| **构建内容** | 从图片中提取文字或描述内容 |
| **为什么重要** | 手写笔记、截图、照片中有大量有价值的信息 |
| **创建文件** | `src/ingest/load_image.py` |
| **使用工具** | `pytesseract`（OCR）, OpenAI/Gemini Vision API（图片理解） |
| **可能遇到的问题** | OCR 精度差（手写体、模糊图片）；Vision API 成本高；图片描述可能不够精确 |
| **调试方法** | 对比原图和提取文本；调整 OCR 参数 |
| **面试要能解释** | OCR vs Vision model 的区别和各自适用场景 |

---

### Phase 6: Vector DB 持久化

| 项目 | 说明 |
|------|------|
| **构建内容** | 将 vector DB 从内存模式改为持久化存储，支持增量更新 |
| **为什么重要** | 每次重启不需要重新 ingest 所有文档 |
| **创建/修改文件** | `src/storage/vector_store.py` |
| **使用工具** | `chromadb`（persist 模式）, 可选迁移到 `FAISS` |
| **可能遇到的问题** | 重复 ingest 导致数据重复；更新文档后旧 chunk 残留 |
| **调试方法** | 检查 collection 中的文档数量；实现 document dedup（去重）逻辑 |
| **面试要能解释** | Vector DB 存储的三要素：vector + text + metadata；持久化 vs 内存模式的权衡；FAISS vs Chroma vs Qdrant 的区别 |

---

### Phase 7: Hybrid Search（混合检索）

| 项目 | 说明 |
|------|------|
| **构建内容** | 同时使用 Vector Search 和 Keyword Search (BM25)，合并结果 |
| **为什么重要** | Vector search 擅长语义匹配但可能漏掉精确关键词；BM25 擅长关键词匹配但不懂语义。两者互补 |
| **创建文件** | `src/retrieval/vector_retriever.py`, `src/retrieval/keyword_retriever.py`, `src/retrieval/hybrid_retriever.py` |
| **使用工具** | `rank-bm25`, `chromadb` |
| **可能遇到的问题** | 两种检索结果的 score 不在同一个量纲；合并策略的选择（RRF vs 加权） |
| **调试方法** | 分别打印 vector search 和 keyword search 的结果，对比差异 |
| **面试要能解释** | 为什么需要 hybrid search；Reciprocal Rank Fusion (RRF) 的原理；BM25 算法的基本思想 |

---

### Phase 8: Reranking（重排序）

| 项目 | 说明 |
|------|------|
| **构建内容** | 对 recall 阶段返回的候选 chunk 进行精细排序 |
| **为什么重要** | 第一阶段检索追求"召回率"（recall），rerank 追求"精确率"（precision）。类似搜索引擎的两阶段排序 |
| **创建文件** | `src/retrieval/reranker.py` |
| **使用工具** | `sentence-transformers` cross-encoder, 可选 Cohere Rerank API |
| **可能遇到的问题** | Cross-encoder 速度慢（不适合大量候选）；模型下载大 |
| **调试方法** | 打印 rerank 前后的排序变化；比较有无 rerank 的回答质量 |
| **面试要能解释** | Bi-encoder（embedding）vs Cross-encoder（reranker）的区别；为什么 rerank 比直接用 cross-encoder 检索更高效 |

---

### Phase 9: Query Rewrite（查询改写）

| 项目 | 说明 |
|------|------|
| **构建内容** | 在检索前改写用户的原始问题，提高检索质量 |
| **为什么重要** | 用户提问往往模糊、口语化、缺少上下文。改写后的 query 更适合检索 |
| **创建文件** | `src/retrieval/query_rewrite.py` |
| **使用工具** | 规则改写 + LLM 改写 |
| **可能遇到的问题** | 过度改写导致偏离原意；增加 latency 和 API 成本 |
| **调试方法** | 打印原始 query 和改写后的 query；对比改写前后的检索结果 |
| **面试要能解释** | Multi-query retrieval；Step-back prompting；什么时候 query rewrite 反而有害 |

---

### Phase 10: Prompt 构建与答案生成

| 项目 | 说明 |
|------|------|
| **构建内容** | 将检索到的 context 和用户问题组合成最终的 prompt，发给 LLM |
| **为什么重要** | Prompt 的质量直接决定最终回答的质量。好的 prompt = 明确的指令 + 相关的 context + 格式要求 |
| **创建文件** | `src/generation/prompt_builder.py`, `src/generation/llm_client.py` |
| **使用工具** | `openai` / `google-generativeai` |
| **可能遇到的问题** | Context 太长超过 token limit；prompt 指令不清晰导致幻觉；源引用不准确 |
| **调试方法** | 打印完整的 final prompt；统计 token 数量；测试 context 不足时的拒答行为 |
| **面试要能解释** | Prompt 的结构设计；如何防止 hallucination；token limit 的处理策略 |

---

### Phase 11: 评估与调试工具

| 项目 | 说明 |
|------|------|
| **构建内容** | 构建评估和调试 RAG 系统的工具集 |
| **为什么重要** | 没有评估，就不知道系统好不好；没有调试工具，出问题时无从下手 |
| **创建文件** | `src/evaluation/debug_retrieval.py` |
| **使用工具** | 自定义调试函数, 可选 `ragas` |
| **可能遇到的问题** | 评估指标的选择；缺少 ground truth 数据 |
| **调试方法** | 构建几个测试 query 和预期答案，手动评估 |
| **面试要能解释** | Retrieval quality 的衡量方式；Answer faithfulness vs relevance；Context precision vs recall |

**评估维度：**

| 指标 | 含义 |
|------|------|
| Retrieval Precision | 检索到的 chunk 中有多少是真正相关的 |
| Retrieval Recall | 所有相关 chunk 中有多少被检索到了 |
| Answer Faithfulness | 回答是否忠实于 context（不编造） |
| Answer Relevance | 回答是否回答了用户的问题 |
| Latency | 端到端延迟 |
| Cost | API 调用成本 |

---

### Phase 12: 完整的个人生活助手

| 项目 | 说明 |
|------|------|
| **构建内容** | 将所有模块整合成一个命令行个人助手 |
| **为什么重要** | 这是学习成果的体现，也是面试展示项目 |
| **修改文件** | `src/app.py` |
| **最终功能** | 自动 ingest `data/raw/` 中的所有文档 → 存入 Vector DB → 命令行交互 → 检索 + Rerank → 生成带来源引用的回答 |

---

## 项目结构

```
personal_rag_assistant/
├── README.md
├── ROADMAP.md                 ← 本文件
├── requirements.txt
├── .env.example
├── data/
│   ├── raw/                   ← 放你的原始文档（PDF, CSV, MD, TXT, 图片）
│   └── processed/             ← 存放处理后的中间结果
├── src/
│   ├── config.py              ← 配置管理（API key, 参数）
│   ├── app.py                 ← 命令行入口
│   ├── ingest/
│   │   ├── load_text.py
│   │   ├── load_markdown.py
│   │   ├── load_pdf.py
│   │   ├── load_csv.py        (Phase 4)
│   │   └── load_image.py      (Phase 5)
│   ├── chunking/
│   │   ├── chunk_text.py      ← Recursive + Semantic chunking
│   │   └── preview_chunks.py  ← 调试工具：查看/对比 chunk 效果
│   ├── embeddings/
│   │   └── embedding_model.py
│   ├── storage/
│   │   └── vector_store.py
│   ├── retrieval/             (Phase 7-9)
│   │   ├── vector_retriever.py
│   │   ├── keyword_retriever.py
│   │   ├── hybrid_retriever.py
│   │   ├── reranker.py
│   │   └── query_rewrite.py
│   ├── generation/
│   │   ├── prompt_builder.py
│   │   └── llm_client.py
│   ├── evaluation/            (Phase 11)
│   │   └── debug_retrieval.py
│   └── utils/
│       └── cost_tracker.py    ← API 花费和延迟监控
```

---

## 技术选型总结

| 组件 | 学习阶段用 | 生产环境可替换为 |
|------|-----------|---------------|
| Vector DB | ChromaDB (本地) | Qdrant / Pinecone / Weaviate |
| Embedding | OpenAI `text-embedding-3-small` | BGE / E5 / Cohere embed |
| LLM | OpenAI GPT-4o-mini | GPT-4o / Claude / Gemini |
| PDF 解析 | PyMuPDF | LlamaParse / Unstructured |
| Keyword Search | rank-bm25 | Elasticsearch |
| Reranker | Cross-encoder (local) | Cohere Rerank / BGE-reranker |
| OCR | pytesseract | Google Vision / Azure OCR |

---

## 学习节奏建议

| Phase | 预计时间 | 难度 |
|-------|---------|------|
| Phase 0 (概念) | 30 分钟 | ★☆☆☆☆ |
| Phase 1 (最小 RAG) | 1-2 小时 | ★★☆☆☆ |
| Phase 2 (MD/TXT) | 1 小时 | ★★☆☆☆ |
| Phase 3 (PDF) | 1-2 小时 | ★★★☆☆ |
| Phase 4 (CSV) | 1 小时 | ★★☆☆☆ |
| Phase 5 (图片) | 1-2 小时 | ★★★☆☆ |
| Phase 6 (持久化) | 1 小时 | ★★☆☆☆ |
| Phase 7 (Hybrid Search) | 2 小时 | ★★★★☆ |
| Phase 8 (Rerank) | 1-2 小时 | ★★★☆☆ |
| Phase 9 (Query Rewrite) | 1 小时 | ★★★☆☆ |
| Phase 10 (Prompt) | 1-2 小时 | ★★★☆☆ |
| Phase 11 (评估) | 1-2 小时 | ★★★★☆ |
| Phase 12 (整合) | 2-3 小时 | ★★★★☆ |
