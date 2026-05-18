"""
Cost & Latency Tracker — 监控 API 调用的花费和延迟
====================================================

核心概念：
- RAG 系统中有两类 API 调用需要监控：
  1. Embedding API: 把文本变成向量（按 token 计费，但很便宜）
  2. LLM API: 生成答案（按 input/output token 分别计费，是主要成本）

- 为什么要监控？
  1. 成本控制：避免不知不觉花了很多钱
  2. 性能优化：知道延迟瓶颈在哪里（embedding? retrieval? LLM?）
  3. 面试展示：展现你的工程素养（production-aware thinking）

面试考点：
- OpenAI 计费方式：按 token 而非字符。1 token ≈ 4 个英文字符 ≈ 0.75 个英文单词
- Embedding 很便宜：text-embedding-3-small $0.02 / 1M tokens
- LLM 分 input/output：input tokens（prompt）比 output tokens（completion）便宜
- gpt-4o-mini: input $0.15/1M tokens, output $0.60/1M tokens
- 实际项目中应该用 tiktoken 精确计算 token 数，这里用近似估算

定价来源（截至 2024 年末，可能更新）:
https://openai.com/pricing
"""

from __future__ import annotations

import time
from dataclasses import dataclass, field


# ============================================================
#  定价表（$ per 1M tokens）
# ============================================================
PRICING = {
    "text-embedding-3-small": {"input": 0.02},
    "text-embedding-3-large": {"input": 0.13},
    "gpt-4o-mini":            {"input": 0.15,  "output": 0.60},
    "gpt-4o":                 {"input": 2.50,  "output": 10.00},
    "gpt-4-turbo":            {"input": 10.00, "output": 30.00},
}


@dataclass
class APICallRecord:
    """一次 API 调用的记录。"""
    call_type: str          # "embedding" or "llm"
    model: str
    input_tokens: int
    output_tokens: int
    latency_ms: float
    estimated_cost: float


@dataclass
class CostTracker:
    """
    全局成本追踪器。记录所有 API 调用的 token 用量、延迟和估算成本。

    用法:
        tracker = get_tracker()
        tracker.record_embedding(model, tokens, latency_ms)
        tracker.record_llm(model, input_tokens, output_tokens, latency_ms)
        tracker.print_summary()
    """
    records: list[APICallRecord] = field(default_factory=list)

    def record_embedding(self, model: str, input_tokens: int, latency_ms: float) -> None:
        price = PRICING.get(model, {}).get("input", 0)
        cost = input_tokens * price / 1_000_000
        self.records.append(APICallRecord(
            call_type="embedding",
            model=model,
            input_tokens=input_tokens,
            output_tokens=0,
            latency_ms=latency_ms,
            estimated_cost=cost,
        ))

    def record_llm(
        self, model: str, input_tokens: int, output_tokens: int, latency_ms: float,
    ) -> None:
        prices = PRICING.get(model, {"input": 0, "output": 0})
        cost = (
            input_tokens * prices.get("input", 0) / 1_000_000
            + output_tokens * prices.get("output", 0) / 1_000_000
        )
        self.records.append(APICallRecord(
            call_type="llm",
            model=model,
            input_tokens=input_tokens,
            output_tokens=output_tokens,
            latency_ms=latency_ms,
            estimated_cost=cost,
        ))

    # ── Aggregate helpers ──────────────────────────────────

    @property
    def total_cost(self) -> float:
        return sum(r.estimated_cost for r in self.records)

    @property
    def total_input_tokens(self) -> int:
        return sum(r.input_tokens for r in self.records)

    @property
    def total_output_tokens(self) -> int:
        return sum(r.output_tokens for r in self.records)

    @property
    def total_latency_ms(self) -> float:
        return sum(r.latency_ms for r in self.records)

    @property
    def embedding_records(self) -> list[APICallRecord]:
        return [r for r in self.records if r.call_type == "embedding"]

    @property
    def llm_records(self) -> list[APICallRecord]:
        return [r for r in self.records if r.call_type == "llm"]

    def print_summary(self) -> None:
        """打印累计统计。"""
        emb = self.embedding_records
        llm = self.llm_records

        print()
        print("=" * 60)
        print("  💰 Cost & Latency Summary")
        print("=" * 60)

        if emb:
            emb_tokens = sum(r.input_tokens for r in emb)
            emb_cost = sum(r.estimated_cost for r in emb)
            emb_latency = sum(r.latency_ms for r in emb)
            print(f"\n  Embedding ({len(emb)} calls)")
            print(f"    Tokens:  {emb_tokens:,}")
            print(f"    Cost:    ${emb_cost:.6f}")
            print(f"    Latency: {emb_latency:,.0f} ms")

        if llm:
            llm_in = sum(r.input_tokens for r in llm)
            llm_out = sum(r.output_tokens for r in llm)
            llm_cost = sum(r.estimated_cost for r in llm)
            llm_latency = sum(r.latency_ms for r in llm)
            print(f"\n  LLM ({len(llm)} calls)")
            print(f"    Input tokens:  {llm_in:,}")
            print(f"    Output tokens: {llm_out:,}")
            print(f"    Cost:    ${llm_cost:.6f}")
            print(f"    Latency: {llm_latency:,.0f} ms")

        print(f"\n  {'─' * 40}")
        print(f"  Total cost:    ${self.total_cost:.6f}")
        print(f"  Total latency: {self.total_latency_ms:,.0f} ms "
              f"({self.total_latency_ms / 1000:.1f}s)")
        print(f"  Total tokens:  {self.total_input_tokens + self.total_output_tokens:,}")
        print("=" * 60)
        print()


# ============================================================
#  全局单例 + 计时工具
# ============================================================

_tracker = CostTracker()


def get_tracker() -> CostTracker:
    return _tracker


class Timer:
    """简单的上下文管理器计时器。"""
    def __enter__(self):
        self.start = time.perf_counter()
        return self

    def __exit__(self, *args):
        self.elapsed_ms = (time.perf_counter() - self.start) * 1000
