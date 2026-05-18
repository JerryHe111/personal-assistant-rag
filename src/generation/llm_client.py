"""
LLM Client 模块 — 调用大语言模型生成答案
==========================================

核心概念：
- 这是 RAG pipeline 的最后一步：把构建好的 prompt 发给 LLM，获取答案
- LLM 看到的是：系统指令 + 检索到的文档 + 用户问题
- LLM 的任务是基于提供的 context 生成答案（而不是凭空编造）

面试考点：
- Temperature 参数：
  - 0.0 = 完全确定性输出（每次回答一样）
  - 0.3 = 稍有变化但仍然稳定（推荐用于 RAG 问答）
  - 1.0 = 高随机性（适合创意写作，不适合事实问答）
- Max tokens: 限制回答长度，避免浪费 token
- 为什么 RAG 场景用低 temperature？因为我们要的是基于事实的准确回答，不需要"创意"
- Streaming: 流式输出让用户立刻看到第一个 token，体感延迟大幅降低
  - 非 streaming: 等待 LLM 生成完所有 token 后一次性返回（感觉很慢）
  - streaming: LLM 每生成一个 token 就立刻发回来（打字机效果，感觉很快）
  - 实际总时间差不多，但 Time-to-First-Token (TTFT) 从几秒降到几百毫秒
"""

import sys
from openai import OpenAI
from src.config import OPENAI_API_KEY, LLM_MODEL, LLM_TEMPERATURE, LLM_MAX_TOKENS
from src.utils.cost_tracker import get_tracker, Timer


_client = OpenAI(api_key=OPENAI_API_KEY)


def generate_answer(messages: list[dict], stream: bool = True) -> str:
    """
    调用 LLM 生成答案，默认使用 streaming 模式逐 token 输出。

    参数:
        messages: OpenAI Chat API 格式的消息列表
        stream: 是否启用流式输出（默认 True）

    返回:
        LLM 生成的完整答案文本
    """
    if not stream:
        with Timer() as t:
            response = _client.chat.completions.create(
                model=LLM_MODEL,
                messages=messages,
                temperature=LLM_TEMPERATURE,
                max_tokens=LLM_MAX_TOKENS,
            )
        usage = response.usage
        get_tracker().record_llm(
            model=LLM_MODEL,
            input_tokens=usage.prompt_tokens,
            output_tokens=usage.completion_tokens,
            latency_ms=t.elapsed_ms,
        )
        return response.choices[0].message.content

    # Streaming 模式：需要开启 stream_options 才能拿到最终 usage
    with Timer() as t:
        response_stream = _client.chat.completions.create(
            model=LLM_MODEL,
            messages=messages,
            temperature=LLM_TEMPERATURE,
            max_tokens=LLM_MAX_TOKENS,
            stream=True,
            stream_options={"include_usage": True},
        )

        collected = []
        usage_info = None
        for chunk in response_stream:
            if chunk.usage is not None:
                usage_info = chunk.usage
            if chunk.choices and chunk.choices[0].delta.content:
                sys.stdout.write(chunk.choices[0].delta.content)
                sys.stdout.flush()
                collected.append(chunk.choices[0].delta.content)

    sys.stdout.write("\n")
    sys.stdout.flush()

    if usage_info:
        get_tracker().record_llm(
            model=LLM_MODEL,
            input_tokens=usage_info.prompt_tokens,
            output_tokens=usage_info.completion_tokens,
            latency_ms=t.elapsed_ms,
        )

    return "".join(collected)
