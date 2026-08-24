# SPDX-FileCopyrightText: 2025 QinHan
# SPDX-License-Identifier: MPL-2.0

import asyncio
import codecs
import json
import logging
import re  # 新增：用于正则估算
import time
from collections import deque
from concurrent.futures import ThreadPoolExecutor
from dataclasses import dataclass
from threading import Lock
from typing import Literal, Callable, Any
from urllib.parse import urlparse

import httpx

from docutranslate.agents.provider import get_provider_by_domain
from docutranslate.agents.thinking.thinking_factory import get_thinking_mode, ProviderType
from docutranslate.config import LLM_STREAMING, NON_STREAM_TIMEOUT, STREAM_IDLE_TIMEOUT
from docutranslate.logger import global_logger
from docutranslate.utils.utils import get_httpx_proxies

MAX_REQUESTS_PER_ERROR = 15
MAX_CONTINUE_FETCHES = 2  # 响应被截断时，最多继续获取的次数

ThinkingMode = Literal["enable", "disable", "default"]


def _parse_response_json(response: httpx.Response) -> dict:
    """
    解析API响应，正确处理前缀空行（如DeepSeek API在高负载时返回的空行）

    Args:
        response: httpx.Response 对象

    Returns:
        解析后的 JSON 字典

    Raises:
        json.JSONDecodeError: 如果响应无法解析为 JSON
    """
    text = response.text
    # 跳过开头的空行和空白字符，找到第一个非空白字符
    # 这可以处理 DeepSeek API 返回的前缀空行
    stripped_text = text.lstrip()
    if not stripped_text:
        raise json.JSONDecodeError("Expecting value", text, 0)
    # 从第一个非空白字符开始解析
    return json.loads(stripped_text)


def _content_to_text(content: Any) -> str:
    """将 OpenAI 兼容接口的多种 content 形态统一为字符串。"""
    if isinstance(content, str):
        return content
    if not isinstance(content, list):
        return ""

    parts = []
    for item in content:
        if isinstance(item, str):
            parts.append(item)
        elif isinstance(item, dict):
            text = item.get("text", item.get("content", ""))
            if isinstance(text, str):
                parts.append(text)
    return "".join(parts)


class _StreamingResponseAccumulator:
    """增量解析 OpenAI 兼容的 SSE/NDJSON，并聚合为原来的响应字典。"""

    def __init__(self):
        self._decoder = codecs.getincrementaldecoder("utf-8")()
        self._line_buffer = ""
        self._event_data: list[str] = []
        self._content: list[str] = []
        self._finish_reason = None
        self._usage: dict | None = None
        self._saw_payload = False
        self.done = False

    def feed(self, chunk: bytes):
        self._feed_text(self._decoder.decode(chunk))

    def _feed_text(self, text: str):
        self._line_buffer += text
        while "\n" in self._line_buffer:
            line, self._line_buffer = self._line_buffer.split("\n", 1)
            self._handle_line(line.rstrip("\r"))

    def _handle_line(self, line: str):
        if not line:
            self._dispatch_event()
            return
        if line.startswith(":"):
            # SSE heartbeat/comment。收到原始字节时空闲计时器已经重置。
            return
        if line.startswith("data:"):
            self._event_data.append(line[5:].lstrip())
            return
        if line.startswith(("event:", "id:", "retry:")):
            return

        # 兼容部分本地 OpenAI 服务使用的 NDJSON。
        stripped = line.strip()
        if stripped.startswith("{"):
            self._consume_payload(stripped)

    def _dispatch_event(self):
        if not self._event_data:
            return
        payload = "\n".join(self._event_data)
        self._event_data.clear()
        self._consume_payload(payload)

    def _consume_payload(self, payload: str):
        if payload.strip() == "[DONE]":
            self.done = True
            return
        try:
            data = json.loads(payload)
        except json.JSONDecodeError:
            return
        if not isinstance(data, dict):
            return
        if data.get("error"):
            raise ValueError(f"流式响应返回错误: {data['error']}")

        self._saw_payload = True
        usage = data.get("usage")
        if isinstance(usage, dict):
            self._usage = usage

        choices = data.get("choices")
        if not isinstance(choices, list) or not choices:
            return
        choice = choices[0]
        if not isinstance(choice, dict):
            return
        finish_reason = choice.get("finish_reason")
        if finish_reason is not None:
            self._finish_reason = finish_reason

        delta = choice.get("delta")
        message = choice.get("message")
        if isinstance(delta, dict):
            content = delta.get("content")
        elif isinstance(message, dict):
            content = message.get("content")
        else:
            content = choice.get("text")
        text = _content_to_text(content)
        if text:
            self._content.append(text)

    def finish(self, raw_body: bytes) -> dict:
        tail = self._decoder.decode(b"", final=True)
        if tail:
            self._feed_text(tail)
        if self._line_buffer:
            self._handle_line(self._line_buffer.rstrip("\r"))
            self._line_buffer = ""
        self._dispatch_event()

        if self._saw_payload:
            result = {
                "choices": [{
                    "message": {"content": "".join(self._content)},
                    "finish_reason": self._finish_reason,
                }]
            }
            if self._usage is not None:
                result["usage"] = self._usage
            return result

        # 有些兼容服务会忽略 stream=true，直接返回普通 JSON。
        text = raw_body.decode("utf-8-sig").lstrip()
        if not text:
            raise json.JSONDecodeError("Expecting value", text, 0)
        return json.loads(text)


class AgentResultError(ValueError):
    """一个特殊的异常，用于表示结果由AI正常返回，但返回的结果有问题。该错误不计入总错误数"""

    def __init__(self, message):
        super().__init__(message)


class PartialAgentResultError(ValueError):
    """一个特殊的异常，用于表示结果不完整但包含了部分成功的数据，以便触发重试。该错误不计入总错误数"""

    def __init__(self, message, partial_result: dict, append_prompt: str = None):
        super().__init__(message)
        self.partial_result = partial_result
        self.append_prompt = append_prompt


@dataclass(kw_only=True)
class AgentConfig:
    logger: logging.Logger = global_logger
    base_url: str
    api_key: str | None = None
    model_id: str
    temperature: float = 0.7
    top_p: float = 0.9
    concurrent: int = 30
    timeout: int = 1200  # 单个分片包含限流等待、续写和全部重试的总时限（秒）
    connect_timeout: float = 5.0
    read_timeout: float | None = None  # None 时沿用 timeout，兼容旧版直接构造方式
    write_timeout: float = 300.0
    pool_timeout: float = 10.0
    thinking: ThinkingMode = "disable"
    retry: int = 2
    system_proxy_enable: bool = False
    force_json: bool = False
    rpm: int | None = None  # 每分钟请求数限制
    tpm: int | None = None  # 每分钟Token数限制
    provider: ProviderType | None = None
    progress_callback: Callable[[int,int],None]|None = None  # 进度回调 (current: int, total: int) -> None
    extra_body: str | None = None  # JSON字符串格式的额外请求体参数
    # 以下字段仅扩展内部传输策略，不改变现有公开方法和 Web/API 请求结构。
    streaming: bool = LLM_STREAMING
    stream_idle_timeout: float = STREAM_IDLE_TIMEOUT
    non_stream_timeout: float | None = NON_STREAM_TIMEOUT


class TotalErrorCounter:
    def __init__(self, logger: logging.Logger, max_errors_count=10):
        self.lock = Lock()
        self.count = 0
        self.logger = logger
        self.max_errors_count = max_errors_count

    def add(self):
        with self.lock:
            self.count += 1
            if self.count > self.max_errors_count:
                self.logger.info(f"错误响应过多")
            return self.reach_limit()

    def reach_limit(self):
        return self.count > self.max_errors_count


# --- 新增 RateLimiter 类 ---
class RateLimiter:
    """
    基于滑动窗口的速率限制器，支持 RPM 和 TPM 控制。
    同时支持 Async 和 Sync 调用。
    """

    def __init__(self, rpm: int | None, tpm: int | None):
        self.rpm = rpm
        self.tpm = tpm
        # 双端队列存储 (timestamp, value)，value对于RPM是1，对于TPM是token数量
        self.request_timestamps = deque()
        self.token_timestamps = deque()
        self.lock = Lock()  # 用于同步模式和保护共享数据

    def _cleanup_window(self, now: float):
        """清理60秒窗口之前的数据"""
        window_start = now - 60.0

        while self.request_timestamps and self.request_timestamps[0] <= window_start:
            self.request_timestamps.popleft()

        while self.token_timestamps and self.token_timestamps[0][0] <= window_start:
            self.token_timestamps.popleft()

    def _check_and_get_wait_time(self, tokens: int) -> float:
        """检查是否满足限制，返回需要等待的秒数。如果不需等待返回 0"""
        now = time.time()
        self._cleanup_window(now)

        wait_time = 0.0

        # Check RPM
        if self.rpm and len(self.request_timestamps) >= self.rpm:
            earliest = self.request_timestamps[0]
            wait_time = max(wait_time, 60 - (now - earliest))

        # Check TPM
        if self.tpm:
            current_tokens = sum(t[1] for t in self.token_timestamps)
            if current_tokens + tokens > self.tpm:
                if self.token_timestamps:
                    earliest = self.token_timestamps[0][0]
                    wait_time = max(wait_time, 60 - (now - earliest))
                else:
                    pass

        return wait_time

    def _record_usage(self, tokens: int):
        """记录使用量"""
        now = time.time()
        if self.rpm is not None:
            self.request_timestamps.append(now)
        if self.tpm is not None:
            self.token_timestamps.append((now, tokens))

    async def acquire_async(self, tokens: int = 0):
        """异步等待配额"""
        if self.rpm is None and self.tpm is None:
            return

        while True:
            # print(f"[RateLimiter-Async] 准备获取锁...")
            with self.lock:
                # print(f"[RateLimiter-Async] 已加锁 (Checking)")

                wait_time = self._check_and_get_wait_time(tokens)
                if wait_time <= 0:
                    self._record_usage(tokens)
                    # print(f"[RateLimiter-Async] 释放锁 (成功获取配额)")
                    return

                # print(f"[RateLimiter-Async] 释放锁 (需等待 {wait_time:.2f}s)")

            # 释放锁后等待
            await asyncio.sleep(wait_time + 0.1)

    def acquire_sync(self, tokens: int = 0):
        """同步等待配额（线程阻塞）"""
        if self.rpm is None and self.tpm is None:
            return

        while True:
            # print(f"[RateLimiter-Sync] 准备获取锁...")
            with self.lock:
                # print(f"[RateLimiter-Sync] 已加锁 (Checking)")

                wait_time = self._check_and_get_wait_time(tokens)
                if wait_time <= 0:
                    self._record_usage(tokens)
                    # print(f"[RateLimiter-Sync] 释放锁 (成功获取配额)")
                    return

                # print(f"[RateLimiter-Sync] 释放锁 (需等待 {wait_time:.2f}s)")

            time.sleep(wait_time + 0.1)


def extract_token_info(response_data: dict) -> tuple[int, int, int, int, int]:
    """从API响应中提取token信息
    返回: (input_tokens, cached_tokens, output_tokens, reasoning_tokens, total_tokens)
    """
    if "usage" not in response_data:
        return 0, 0, 0, 0, 0

    usage = response_data["usage"]
    # print(usage)
    input_tokens = usage.get("prompt_tokens", 0)
    output_tokens = usage.get("completion_tokens", 0)
    total_tokens = usage.get("total_tokens", 0)

    cached_tokens = 0
    reasoning_tokens = 0
    try:
        # 尝试多种可能的 cached_tokens 字段位置
        if (
                "input_tokens_details" in usage
                and "cached_tokens" in usage["input_tokens_details"]
        ):
            cached_tokens = usage["input_tokens_details"]["cached_tokens"]
        elif (
                "prompt_tokens_details" in usage
                and "cached_tokens" in usage["prompt_tokens_details"]
        ):
            cached_tokens = usage["prompt_tokens_details"]["cached_tokens"]
        elif "prompt_cache_hit_tokens" in usage:
            cached_tokens = usage["prompt_cache_hit_tokens"]

        # 尝试多种可能的 reasoning_tokens 字段位置
        if (
                "output_tokens_details" in usage
                and "reasoning_tokens" in usage["output_tokens_details"]
        ):
            reasoning_tokens = usage["output_tokens_details"]["reasoning_tokens"]
        elif (
                "completion_tokens_details" in usage
                and "reasoning_tokens" in usage["completion_tokens_details"]
        ):
            reasoning_tokens = usage["completion_tokens_details"]["reasoning_tokens"]
        else:
            # Gemini特殊处理: 如果total_tokens大于prompt+completion，差额很可能是思考token
            if total_tokens > 0 and (input_tokens + output_tokens) > 0 and total_tokens > (input_tokens + output_tokens):
                reasoning_tokens = total_tokens - (input_tokens + output_tokens)

        return input_tokens, cached_tokens, output_tokens, reasoning_tokens, total_tokens
    except (TypeError, KeyError, AttributeError):
        return -1, -1, -1, -1, -1


class TokenCounter:
    def __init__(self, logger: logging.Logger):
        self.lock = Lock()
        self.input_tokens = 0
        self.cached_tokens = 0
        self.output_tokens = 0
        self.reasoning_tokens = 0
        self.total_tokens = 0
        self.logger = logger

    def add(
            self,
            input_tokens: int,
            cached_tokens: int,
            output_tokens: int,
            reasoning_tokens: int,
            api_total_tokens: int = 0,
    ):
        with self.lock:
            self.input_tokens += input_tokens
            self.cached_tokens += cached_tokens
            self.output_tokens += output_tokens
            self.reasoning_tokens += reasoning_tokens
            # 如果API返回了total_tokens，优先使用；否则自己计算
            if api_total_tokens > 0:
                self.total_tokens += api_total_tokens
            else:
                self.total_tokens += input_tokens + output_tokens

    def get_stats(self):
        with self.lock:
            return {
                "input_tokens": self.input_tokens,
                "cached_tokens": self.cached_tokens,
                "output_tokens": self.output_tokens,
                "reasoning_tokens": self.reasoning_tokens,
                "total_tokens": self.total_tokens,
            }

    def reset(self):
        with self.lock:
            self.input_tokens = 0
            self.cached_tokens = 0
            self.output_tokens = 0
            self.reasoning_tokens = 0
            self.total_tokens = 0


PreSendHandlerType = Callable[[str, str], tuple[str, str]]
ResultHandlerType = Callable[[str, str, logging.Logger], Any]
ErrorResultHandlerType = Callable[[str, logging.Logger], Any]

# _CJK_PATTERN = re.compile(r'[\u4e00-\u9fff\u3040-\u30ff\uac00-\ud7af]')
# 扩展正则范围，包含：
# CJK (中日韩): \u2e80-\u9fff
# 西里尔 (俄语等): \u0400-\u04ff
# 阿拉伯语: \u0600-\u06ff
# 泰语: \u0e00-\u0e7f
# 梵文 (印地语等): \u0900-\u097f
# 标点和特殊符号范围较广，这里主要抓取非拉丁体系的主要语言
_COMPLEX_SCRIPT_PATTERN = re.compile(
    r'[\u2e80-\u9fff\u0400-\u04ff\u0600-\u06ff\u0e00-\u0e7f\u0900-\u097f]'
)

class Agent:

    def __init__(self, config: AgentConfig):
        self.baseurl = config.base_url.strip()
        if self.baseurl.endswith("/"):
            self.baseurl = self.baseurl[:-1]
        self.domain = urlparse(self.baseurl).netloc.strip()
        self.key = config.api_key.strip() if config.api_key else "xx"
        self.model_id = config.model_id.strip()
        self.system_prompt = ""
        self.temperature = config.temperature
        self.top_p = config.top_p
        self.max_concurrent = config.concurrent
        # 流式和非流式使用两套独立策略：流式仅限制连续静默时间；
        # 非流式保留硬总时限。旧 timeout/read_timeout 配置仍用于非流式。
        self.non_stream_total_timeout_seconds = (
            config.timeout if config.non_stream_timeout is None else config.non_stream_timeout
        )
        self.total_timeout_seconds = self.non_stream_total_timeout_seconds
        read_timeout = config.timeout if config.read_timeout is None else config.read_timeout
        self.timeout = httpx.Timeout(
            connect=config.connect_timeout,
            read=read_timeout,
            write=config.write_timeout,
            pool=config.pool_timeout,
        )
        self.stream_idle_timeout_seconds = config.stream_idle_timeout
        self.stream_timeout = httpx.Timeout(
            connect=config.connect_timeout,
            read=self.stream_idle_timeout_seconds,
            write=config.write_timeout,
            pool=config.pool_timeout,
        )
        self.streaming_enabled = config.streaming
        self._streaming_disabled = not config.streaming
        self.thinking = config.thinking
        self.logger = config.logger
        self.total_error_counter = TotalErrorCounter(logger=self.logger)
        self.unresolved_error_lock = Lock()
        self.unresolved_error_count = 0
        self.token_counter = TokenCounter(logger=self.logger)
        self._request_count = 0  # 记录请求数量
        self.retry = config.retry
        self.system_proxy_enable = config.system_proxy_enable
        self.progress_callback = config.progress_callback  # 进度回调

        # 新增：初始化速率限制器
        self.rate_limiter = RateLimiter(rpm=config.rpm, tpm=config.tpm)

        self.provider = config.provider if config.provider is not None else get_provider_by_domain(self.domain)
        self.extra_body = config.extra_body

    async def _request_completion_async(
            self,
            client: httpx.AsyncClient,
            data: dict,
            headers: dict,
    ) -> dict:
        """请求一次 completion；优先流式，明确不支持时自动兼容非流式。"""
        if self._streaming_disabled:
            return await self._request_non_stream_async(client, data, headers)

        stream_data = dict(data)
        stream_data["stream"] = True
        try:
            async with client.stream(
                    "POST",
                    f"{self.baseurl}/chat/completions",
                    json=stream_data,
                    headers=headers,
                    timeout=self.stream_timeout,
            ) as response:
                if response.is_error:
                    await response.aread()
                response.raise_for_status()
                if response.is_stream_consumed:
                    accumulator = _StreamingResponseAccumulator()
                    accumulator.feed(response.content)
                    return accumulator.finish(response.content)
                accumulator = _StreamingResponseAccumulator()
                raw_body = bytearray()
                iterator = response.aiter_raw()
                while not accumulator.done:
                    try:
                        chunk = await asyncio.wait_for(
                            anext(iterator), timeout=self.stream_idle_timeout_seconds
                        )
                    except StopAsyncIteration:
                        break
                    except TimeoutError as exc:
                        raise httpx.ReadTimeout(
                            f"流式响应连续 {self.stream_idle_timeout_seconds} 秒无活动",
                            request=response.request,
                        ) from exc
                    if not chunk:
                        continue
                    raw_body.extend(chunk)
                    accumulator.feed(chunk)
                return accumulator.finish(bytes(raw_body))
        except httpx.HTTPStatusError as stream_error:
            # 仅对常见的“不支持 stream 参数/端点”状态进行一次非流式兼容尝试。
            if stream_error.response.status_code not in {400, 404, 405, 415, 422, 501}:
                raise
            self.logger.warning(
                f"当前 LLM 接口拒绝流式请求 ({stream_error.response.status_code})，"
                "尝试兼容非流式响应。"
            )
            self._streaming_disabled = True
            result = await self._request_non_stream_async(client, data, headers)
            return result

    async def _request_non_stream_async(
            self,
            client: httpx.AsyncClient,
            data: dict,
            headers: dict,
    ) -> dict:
        """执行非流式请求，并施加与流式空闲窗口独立的硬总时限。"""
        total_timeout = asyncio.timeout(self.non_stream_total_timeout_seconds)
        try:
            async with total_timeout:
                response = await client.post(
                    f"{self.baseurl}/chat/completions",
                    json=data,
                    headers=headers,
                    timeout=self.timeout,
                )
        except TimeoutError as exc:
            if not total_timeout.expired():
                raise
            raise httpx.ReadTimeout(
                f"非流式请求超过硬总时限 {self.non_stream_total_timeout_seconds} 秒"
            ) from exc
        response.raise_for_status()
        return _parse_response_json(response)

    def _request_completion_sync(
            self,
            client: httpx.Client,
            data: dict,
            headers: dict,
    ) -> dict:
        """同步入口的流式兼容实现；实际 socket 空闲检测由 HTTPX read timeout 完成。"""
        if self._streaming_disabled:
            response = client.post(
                f"{self.baseurl}/chat/completions",
                json=data,
                headers=headers,
                timeout=self.timeout,
            )
            response.raise_for_status()
            return _parse_response_json(response)

        stream_data = dict(data)
        stream_data["stream"] = True
        try:
            with client.stream(
                    "POST",
                    f"{self.baseurl}/chat/completions",
                    json=stream_data,
                    headers=headers,
                    timeout=self.stream_timeout,
            ) as response:
                if response.is_error:
                    response.read()
                response.raise_for_status()
                if response.is_stream_consumed:
                    accumulator = _StreamingResponseAccumulator()
                    accumulator.feed(response.content)
                    return accumulator.finish(response.content)
                accumulator = _StreamingResponseAccumulator()
                raw_body = bytearray()
                for chunk in response.iter_raw():
                    if not chunk:
                        continue
                    raw_body.extend(chunk)
                    accumulator.feed(chunk)
                    if accumulator.done:
                        break
                return accumulator.finish(bytes(raw_body))
        except httpx.HTTPStatusError as stream_error:
            if stream_error.response.status_code not in {400, 404, 405, 415, 422, 501}:
                raise
            self.logger.warning(
                f"当前 LLM 接口拒绝流式请求 ({stream_error.response.status_code})，"
                "尝试兼容非流式响应。"
            )
            self._streaming_disabled = True
            response = client.post(
                f"{self.baseurl}/chat/completions",
                json=data,
                headers=headers,
                timeout=self.timeout,
            )
            response.raise_for_status()
            result = _parse_response_json(response)
            return result

    def _estimate_tokens(self, text: str) -> int:
        """
        改进的纯 Python 估算，适配更多语言。
        """
        if not text:
            return 0

        total_len = len(text)

        # 统计复杂字符数量 (CJK, 俄语, 阿拉伯语等)
        complex_char_count = len(_COMPLEX_SCRIPT_PATTERN.findall(text))

        # 简单的 ASCII 或拉丁字符
        simple_char_count = total_len - complex_char_count

        # 权重设定：
        # 复杂字符：保守估计 1.0 (GPT-4o 对中文优化很好，约为0.6-0.7，但为了限流安全，建议设高一点)
        # 简单字符：0.3 (英文平均 1个token ≈ 3.5字符)
        # 额外：加上消息的固定开销 (Message Overhead)，通常每条消息有 3-4 个 token 的系统开销

        estimated = (complex_char_count * 1.0) + (simple_char_count * 0.3)

        # 向上取整
        return int(estimated) + 1

    def _sanitize_result(self, text: str) -> str:
        """
        清理响应内容：如果内容以 <think>...</think> 开头，移除该部分。
        使用 DOTALL 模式以匹配跨行的 thinking 内容。
        """
        if not text:
            return text
        # 匹配开头的 <think> 标签块，允许标签前后有空白字符
        # .*? 非贪婪匹配，确保只匹配第一个闭合标签
        return re.sub(r'^\s*<think>.*?</think>', '', text, flags=re.DOTALL)

    def get_continue_prompt(self, accumulated_result: str, prompt: str) -> str:
        """
        获取继续获取时的提示词。
        子类可以重写此方法来自定义继续获取的行为。

        默认行为：直接拼接内容，让模型继续输出。
        """
        return f"{prompt}\n\n[系统提示：请继续完成之前的响应。之前已输出内容为：\n---\n{accumulated_result}\n---\n请从中断处继续输出剩余内容。]"

    def merge_continue_result(self, accumulated_result: str, additional_result: str) -> str:
        """
        合并继续获取的结果。
        子类可以重写此方法来处理追加模式的数组合并。

        默认行为：直接拼接字符串。
        """
        return accumulated_result + additional_result

    def _add_thinking_mode(self, data: dict):
        thinking_mode_result = get_thinking_mode(self.provider, data.get("model"))
        if thinking_mode_result is None:
            return
        field_thinking, val_enable, val_disable = thinking_mode_result

        # 获取要设置的值
        if self.thinking == "enable":
            value = val_enable
        elif self.thinking == "disable":
            value = val_disable
        else:
            return

        # 特殊处理 extra_body 类型：不是设置 data["extra_body"]，而是直接合并到 data 中
        if field_thinking == "extra_body":
            if isinstance(value, dict):
                data.update(value)
        else:
            # 普通字段直接设置
            data[field_thinking] = value

    def _prepare_request_data(
            self, prompt: str, system_prompt: str, temperature=None, top_p=None, json_format=False
    ):
        if temperature is None:
            temperature = self.temperature
        if top_p is None:
            top_p = self.top_p
        headers = {
            "Content-Type": "application/json",
            "Authorization": f"Bearer {self.key}",
        }
        data = {
            "model": self.model_id,
            "messages": [
                {"role": "system", "content": system_prompt},
                {"role": "user", "content": prompt},
            ],
            "temperature": temperature,
            "top_p": top_p,
        }

        # 先应用思考模式
        if self.thinking != "default":
            self._add_thinking_mode(data)

        # 再应用用户的 extra_body（用户配置优先，可以覆盖思考模式）
        if self.extra_body and self.extra_body.strip():
            try:
                import json
                extra = json.loads(self.extra_body)
                if isinstance(extra, dict):
                    data.update(extra)
            except (json.JSONDecodeError, ValueError):
                self.logger.warning(f"Failed to parse extra_body JSON: {self.extra_body}")

        if json_format:
            data["response_format"] = {"type": "json_object"}

        return headers, data

    async def _continue_fetch_async(
            self,
            client: httpx.AsyncClient,
            prompt: str,
            system_prompt: str,
            force_json: bool,
            pre_send_handler: PreSendHandlerType,
            result_handler: ResultHandlerType,
            error_result_handler: ErrorResultHandlerType,
            retry_count: int,
            accumulated_result: str = "",
            continue_count: int = 0,
    ) -> Any:
        """
        当 finish_reason 为 length 时，继续获取剩余内容。
        注意：很多 API 并不支持这种"继续获取"模式，可能直接返回 stop 或不返回 length。
        本方法具有退化机制：如果 API 不支持继续获取，会返回已累计的结果。
        最多继续获取 MAX_CONTINUE_FETCHES 次，防止无限循环。
        """
        if continue_count >= MAX_CONTINUE_FETCHES:
            self.logger.warning(
                f"已达到最大继续获取次数 ({MAX_CONTINUE_FETCHES})，返回已累计结果 ({len(accumulated_result)} 字符)")
            # 移除可能存在的 <think> 块
            accumulated_result = self._sanitize_result(accumulated_result)
            return (
                accumulated_result
                if result_handler is None
                else result_handler(accumulated_result, prompt, self.logger)
            )

        self.logger.info(
            f"继续获取剩余内容 (已累计 {len(accumulated_result)} 字符, 第 {continue_count + 1}/{MAX_CONTINUE_FETCHES} 次)...")

        # 构造继续请求的提示
        # 调用子类的 get_continue_prompt 方法，允许子类自定义继续获取的行为
        continue_prompt = self.get_continue_prompt(accumulated_result, prompt)

        if pre_send_handler:
            system_prompt, continue_prompt = pre_send_handler(system_prompt, continue_prompt)

        # 速率限制检查
        estimated_tokens = self._estimate_tokens(system_prompt) + self._estimate_tokens(continue_prompt)
        await self.rate_limiter.acquire_async(tokens=estimated_tokens)

        headers, data = self._prepare_request_data(continue_prompt, system_prompt, json_format=force_json)

        try:
            response_data = await self._request_completion_async(client, data, headers)

            # 安全提取 choices 和 content
            choices = response_data.get("choices", [])
            if not choices:
                self.logger.error(f"API响应中未找到 choices 字段")
                raise ValueError("API响应格式错误：缺少 choices 字段")

            choice = choices[0]
            finish_reason = choice.get("finish_reason", None)
            message = choice.get("message", {})
            additional_result = message.get("content", "")

            input_tokens, cached_tokens, output_tokens, reasoning_tokens, api_total_tokens = (
                extract_token_info(response_data)
            )
            self.token_counter.add(input_tokens, cached_tokens, output_tokens, reasoning_tokens, api_total_tokens)

            # 累加结果（使用 merge_continue_result 方法处理追加模式的合并）
            accumulated_result = self.merge_continue_result(accumulated_result, additional_result)

            # 如果仍然是 length，继续获取（限制最大轮数防止无限循环）
            if finish_reason == "length":
                return await self._continue_fetch_async(
                    client=client,
                    prompt=prompt,
                    system_prompt=system_prompt,
                    force_json=force_json,
                    pre_send_handler=pre_send_handler,
                    result_handler=result_handler,
                    error_result_handler=error_result_handler,
                    retry_count=retry_count,
                    accumulated_result=accumulated_result,
                    continue_count=continue_count + 1,
                )

            # 非 length 结束，返回累加结果
            try:
                # 最终清理结果
                accumulated_result = self._sanitize_result(accumulated_result)
                return (
                    accumulated_result
                    if result_handler is None
                    else result_handler(accumulated_result, prompt, self.logger)
                )
            except PartialAgentResultError as e:
                # 继续获取成功但结果部分不完整，返回已合并的部分结果
                self.logger.warning(f"继续获取完成但结果部分不完整: {e}")
                if e.partial_result:
                    return e.partial_result
                # 如果没有部分结果，尝试从已获取的内容中解析
                return accumulated_result
            except AgentResultError as e:
                # 继续获取成功但结果完全无效
                self.logger.warning(f"继续获取完成但结果无效: {e}")
                return accumulated_result

        except (httpx.HTTPStatusError, httpx.RequestError, KeyError, IndexError, ValueError) as e:
            self.logger.error(f"继续获取内容失败: {repr(e)}")
            # 退化：返回已获取的部分结果，而不是报错
            if accumulated_result:
                self.logger.warning(f"API不支持继续获取，返回已获取的部分结果 ({len(accumulated_result)} 字符)")
                # 即使是部分结果，也尝试清理一下
                accumulated_result = self._sanitize_result(accumulated_result)
                return (
                    accumulated_result
                    if result_handler is None
                    else result_handler(accumulated_result, prompt, self.logger)
                )
            # 如果没有部分结果，调用错误处理器
            return (
                prompt
                if error_result_handler is None
                else error_result_handler(prompt, self.logger)
            )

    async def send_async(
            self,
            client: httpx.AsyncClient,
            prompt: str,
            system_prompt: None | str = None,
            retry=True,
            retry_count=0,
            force_json=False,
            pre_send_handler: PreSendHandlerType = None,
            result_handler: ResultHandlerType = None,
            error_result_handler: ErrorResultHandlerType = None,
            best_partial_result: dict | None = None,
    ) -> Any:
        if system_prompt is None:
            system_prompt = self.system_prompt
        if pre_send_handler:
            system_prompt, prompt = pre_send_handler(system_prompt, prompt)

        # 新增：速率限制检查
        # 计算估算的 tokens (system + user)
        estimated_tokens = self._estimate_tokens(system_prompt) + self._estimate_tokens(prompt)
        # 等待配额
        await self.rate_limiter.acquire_async(tokens=estimated_tokens)

        headers, data = self._prepare_request_data(prompt, system_prompt, json_format=force_json)
        should_retry = False
        is_hard_error = False
        current_partial_result = None
        input_tokens = 0
        output_tokens = 0
        try:
            response_data = await self._request_completion_async(client, data, headers)

            # 检查 finish_reason
            choices = response_data.get("choices", [])
            if not choices:
                self.logger.error(f"API响应中未找到 choices 字段")
                raise ValueError("API响应格式错误：缺少 choices 字段")

            finish_reason = choices[0].get("finish_reason", None)
            result = choices[0].get("message", {}).get("content", "")

            # 处理不同的 finish_reason
            if finish_reason == "stop":
                # 正常结束
                pass
            elif finish_reason == "length":
                # 长度限制，尝试继续获取
                self.logger.warning(f"响应因长度限制被截断，尝试继续获取...")
                # 注意：这里传入原始result，清理工作在 _continue_fetch_async 最终返回时统一处理
                return await self._continue_fetch_async(
                    client=client,
                    prompt=prompt,
                    system_prompt=system_prompt,
                    force_json=force_json,
                    pre_send_handler=pre_send_handler,
                    result_handler=result_handler,
                    error_result_handler=error_result_handler,
                    retry_count=retry_count,
                    accumulated_result=result,
                )
            elif finish_reason in ("tool_calls", "function_call"):
                # 工具调用场景，当前代码可能不支持，直接返回已获取结果
                self.logger.warning(f"finish_reason 为 '{finish_reason}'，当前不支持工具调用，返回已获取内容")
                result = self._sanitize_result(result)
                return result if result else (
                    prompt if error_result_handler is None
                    else error_result_handler(prompt, self.logger)
                )
            elif finish_reason == "content_filter":
                # 内容被过滤
                self.logger.error(f"响应内容被过滤")
                raise ValueError("内容被过滤")
            elif finish_reason is None:
                # 某些 API 可能不返回 finish_reason，将其视为正常结束
                self.logger.warning(f"API未返回 finish_reason，视为正常结束")
            else:
                # 其他未知的 finish_reason，记录警告并返回结果
                self.logger.warning(f"未知的 finish_reason: '{finish_reason}'，返回已获取内容")

            input_tokens, cached_tokens, output_tokens, reasoning_tokens, api_total_tokens = (
                extract_token_info(response_data)
            )

            self.token_counter.add(
                input_tokens, cached_tokens, output_tokens, reasoning_tokens, api_total_tokens
            )

            if retry_count > 0:
                self.logger.info(f"重试成功 (第 {retry_count}/{self.retry} 次尝试)。")

            # 清理 <think> 标签后再处理结果
            result = self._sanitize_result(result)

            return (
                result
                if result_handler is None
                else result_handler(result, prompt, self.logger)
            )

        except AgentResultError as e:
            self.logger.error(f"AI返回结果有误: {e}")
            should_retry = True
        except PartialAgentResultError as e:
            self.logger.error(f"收到部分返回结果，将尝试重试: {e}")
            current_partial_result = e.partial_result
            should_retry = True
            if e.append_prompt:
                prompt += e.append_prompt

        except httpx.HTTPStatusError as e:
            self.logger.error(
                f"AI请求HTTP状态错误 (async): {e.response.status_code} - {e.response.text}"
            )
            should_retry = True
            is_hard_error = True
            # 如果是因为 Rate Limit (429) 错误，最好在这里多睡一会儿，虽然我们有了本地 Limiter
            if e.response.status_code == 429:
                await asyncio.sleep(5)

        except httpx.RequestError as e:
            # 根据错误类型给出更清晰的提示
            if isinstance(e, httpx.ReadError):
                self.logger.error(f"AI请求读取响应失败 (async): {type(e).__name__}: {e} (可能是服务器关闭连接或网络中断)")
            elif isinstance(e, httpx.ConnectError):
                self.logger.error(f"AI请求连接失败 (async): {type(e).__name__}: {e} (无法连接到服务器，请检查网络或base_url)")
            elif isinstance(e, httpx.WriteError):
                self.logger.error(f"AI请求发送数据失败 (async): {type(e).__name__}: {e}")
            elif isinstance(e, httpx.TimeoutException):
                self.logger.error(f"AI请求超时 (async): {type(e).__name__}: {e} (请求超过{self.timeout}秒未完成)")
            else:
                self.logger.error(f"AI请求连接错误 (async): {type(e).__name__}: {e}")
            should_retry = True
            is_hard_error = True
        except (KeyError, IndexError, ValueError, json.JSONDecodeError) as e:
            self.logger.error(f"AI响应格式或值错误 (async), 将尝试重试: {repr(e)}")
            should_retry = True
            is_hard_error = True

        if current_partial_result:
            best_partial_result = current_partial_result

        if should_retry and retry and retry_count < self.retry:
            if is_hard_error:
                if retry_count == 0:
                    if self.total_error_counter.add():
                        self.logger.error("错误次数过多，已达到上限，不再重试。")
                        with self.unresolved_error_lock:
                            self.unresolved_error_count += 1
                        return (
                            best_partial_result
                            if best_partial_result
                            else (
                                prompt
                                if error_result_handler is None
                                else error_result_handler(prompt, self.logger)
                            )
                        )
                elif self.total_error_counter.reach_limit():
                    self.logger.error("错误次数过多，已达到上限，不再为该请求重试。")
                    with self.unresolved_error_lock:
                        self.unresolved_error_count += 1
                    return (
                        best_partial_result
                        if best_partial_result
                        else (
                            prompt
                            if error_result_handler is None
                            else error_result_handler(prompt, self.logger)
                        )
                    )

            self.logger.info(f"正在重试第 {retry_count + 1}/{self.retry} 次...")
            # 指数退避
            await asyncio.sleep(0.5 * (2 ** retry_count))
            return await self.send_async(
                client,
                prompt,
                system_prompt,
                retry=True,
                retry_count=retry_count + 1,
                force_json=force_json,
                pre_send_handler=pre_send_handler,
                result_handler=result_handler,
                error_result_handler=error_result_handler,
                best_partial_result=best_partial_result,
            )
        else:
            if should_retry:
                self.logger.error(f"所有重试均失败，已达到重试次数上限。")
                with self.unresolved_error_lock:
                    self.unresolved_error_count += 1

            if best_partial_result:
                self.logger.info("所有重试失败，但存在部分翻译结果，将使用该结果。")
                return best_partial_result

            return (
                prompt
                if error_result_handler is None
                else error_result_handler(prompt, self.logger)
            )

    async def send_prompts_async(
            self,
            prompts: list[str],
            system_prompt: str | None = None,
            max_concurrent: int | None = None,
            force_json=False,
            pre_send_handler: PreSendHandlerType = None,
            result_handler: ResultHandlerType = None,
            error_result_handler: ErrorResultHandlerType = None,
    ) -> list[Any]:
        max_concurrent = (
            self.max_concurrent if max_concurrent is None else max_concurrent
        )
        total = len(prompts)
        rpm_info = f", RPM:{self.rate_limiter.rpm}" if self.rate_limiter.rpm else ""
        tpm_info = f", TPM:{self.rate_limiter.tpm}" if self.rate_limiter.tpm else ""
        transport_mode = "non-stream" if self._streaming_disabled else "stream"

        self.logger.info(
            f"provider:{self.provider},base-url:{self.baseurl},model-id:{self.model_id},concurrent:{max_concurrent}{rpm_info}{tpm_info},temperature:{self.temperature},"
            f"transport:{transport_mode},timeouts(stream-idle:{self.stream_idle_timeout_seconds}s,stream-total:none,"
            f"non-stream-total:{self.non_stream_total_timeout_seconds}s,non-stream-read:{self.timeout.read}s,"
            f"connect:{self.timeout.connect}s,write:{self.timeout.write}s,pool:{self.timeout.pool}s),"
            f"system_proxy:{self.system_proxy_enable},json_output:{force_json}"
        )
        self.logger.info(f"预计发送{total}个请求")

        self.total_error_counter.max_errors_count = (
                len(prompts) // MAX_REQUESTS_PER_ERROR
        )

        self.unresolved_error_count = 0
        self.token_counter.reset()
        self._request_count = len(prompts)  # 记录请求数量

        count = 0
        semaphore = asyncio.Semaphore(max_concurrent)
        tasks = []

        proxies = get_httpx_proxies(asyn=True) if self.system_proxy_enable else None

        limits = httpx.Limits(
            max_connections=self.max_concurrent * 2,
            max_keepalive_connections=self.max_concurrent,
        )

        async with httpx.AsyncClient(
                trust_env=False, mounts=proxies, verify=False, limits=limits
        ) as client:
            async def send_with_semaphore(p_text: str, prompt_index: int):
                async with semaphore:
                    # 注意：我们在 semaphore 内部调用 send_async
                    # send_async 内部会调用 rate_limiter.acquire_async
                    # 这样可以防止并发过高，同时 rate_limiter 防止频率过快
                    async def perform_send():
                        return await self.send_async(
                            client=client,
                            prompt=p_text,
                            system_prompt=system_prompt,
                            force_json=force_json,
                            pre_send_handler=pre_send_handler,
                            result_handler=result_handler,
                            error_result_handler=error_result_handler,
                        )

                    if self._streaming_disabled:
                        total_timeout = asyncio.timeout(self.non_stream_total_timeout_seconds)
                        try:
                            async with total_timeout:
                                result = await perform_send()
                        except TimeoutError:
                            # 非流式模式继续保留旧版“整个分片”的硬截止时间。
                            if not total_timeout.expired():
                                raise
                            self.logger.error(
                                f"非流式分片 {prompt_index}/{total} 超过总时限 "
                                f"{self.non_stream_total_timeout_seconds} 秒，"
                                "已终止该分片并使用降级结果。"
                            )
                            with self.unresolved_error_lock:
                                self.unresolved_error_count += 1
                            result = (
                                p_text
                                if error_result_handler is None
                                else error_result_handler(p_text, self.logger)
                            )
                    else:
                        # DeepSeek Harness 风格：流式请求没有总墙钟上限，
                        # 仅由每次收到原始数据都会重置的 idle watchdog 判定超时。
                        result = await perform_send()
                    nonlocal count
                    count += 1
                    self.logger.info(f"协程-已完成{count}/{total}")
                    # 调用进度回调
                    if self.progress_callback:
                        self.progress_callback(count, total)
                    return result

            for prompt_index, p_text in enumerate(prompts, start=1):
                task = asyncio.create_task(send_with_semaphore(p_text, prompt_index))
                tasks.append(task)

            results = await asyncio.gather(*tasks, return_exceptions=False)

            self.logger.info(
                f"所有请求处理完毕。未解决的错误总数: {self.unresolved_error_count}"
            )

            token_stats = self.token_counter.get_stats()
            self.logger.info(
                f"Token使用统计 - 输入: {token_stats['input_tokens'] / 1000:.2f}K(含cached: {token_stats['cached_tokens'] / 1000:.2f}K), "
                f"输出: {token_stats['output_tokens'] / 1000:.2f}K(含reasoning: {token_stats['reasoning_tokens'] / 1000:.2f}K), "
                f"总计: {token_stats['total_tokens'] / 1000:.2f}K"
            )

            return results

    def _continue_fetch(
            self,
            client: httpx.Client,
            prompt: str,
            system_prompt: str,
            force_json: bool,
            pre_send_handler,
            result_handler,
            error_result_handler,
            retry_count: int,
            accumulated_result: str = "",
            continue_count: int = 0,
    ) -> Any:
        """
        当 finish_reason 为 length 时，继续获取剩余内容（同步版本）。
        注意：很多 API 并不支持这种"继续获取"模式，可能直接返回 stop 或不返回 length。
        本方法具有退化机制：如果 API 不支持继续获取，会返回已累计的结果。
        最多继续获取 MAX_CONTINUE_FETCHES 次，防止无限循环。
        """
        if continue_count >= MAX_CONTINUE_FETCHES:
            self.logger.warning(
                f"已达到最大继续获取次数 ({MAX_CONTINUE_FETCHES})，返回已累计结果 ({len(accumulated_result)} 字符)")
            # 清理
            accumulated_result = self._sanitize_result(accumulated_result)
            return (
                accumulated_result
                if result_handler is None
                else result_handler(accumulated_result, prompt, self.logger)
            )

        self.logger.info(
            f"继续获取剩余内容 (已累计 {len(accumulated_result)} 字符, 第 {continue_count + 1}/{MAX_CONTINUE_FETCHES} 次)...")

        # 构造继续请求的提示
        # 调用子类的 get_continue_prompt 方法，允许子类自定义继续获取的行为
        continue_prompt = self.get_continue_prompt(accumulated_result, prompt)

        if pre_send_handler:
            system_prompt, continue_prompt = pre_send_handler(system_prompt, continue_prompt)

        estimated_tokens = self._estimate_tokens(system_prompt) + self._estimate_tokens(continue_prompt)
        self.rate_limiter.acquire_sync(tokens=estimated_tokens)

        headers, data = self._prepare_request_data(continue_prompt, system_prompt, json_format=force_json)

        try:
            response_data = self._request_completion_sync(client, data, headers)

            # 安全提取 choices 和 content
            choices = response_data.get("choices", [])
            if not choices:
                self.logger.error(f"API响应中未找到 choices 字段")
                raise ValueError("API响应格式错误：缺少 choices 字段")

            choice = choices[0]
            finish_reason = choice.get("finish_reason", None)
            message = choice.get("message", {})
            additional_result = message.get("content", "")

            input_tokens, cached_tokens, output_tokens, reasoning_tokens, api_total_tokens = (
                extract_token_info(response_data)
            )
            self.token_counter.add(input_tokens, cached_tokens, output_tokens, reasoning_tokens, api_total_tokens)

            # 累加结果（使用 merge_continue_result 方法处理追加模式的合并）
            accumulated_result = self.merge_continue_result(accumulated_result, additional_result)

            if finish_reason == "length":
                return self._continue_fetch(
                    client=client,
                    prompt=prompt,
                    system_prompt=system_prompt,
                    force_json=force_json,
                    pre_send_handler=pre_send_handler,
                    result_handler=result_handler,
                    error_result_handler=error_result_handler,
                    retry_count=retry_count,
                    accumulated_result=accumulated_result,
                    continue_count=continue_count + 1,
                )

            # 非 length 结束，返回累加结果
            try:
                # 最终清理结果
                accumulated_result = self._sanitize_result(accumulated_result)
                return (
                    accumulated_result
                    if result_handler is None
                    else result_handler(accumulated_result, prompt, self.logger)
                )
            except PartialAgentResultError as e:
                # 继续获取成功但结果部分不完整，返回已合并的部分结果
                self.logger.warning(f"继续获取完成但结果部分不完整: {e}")
                if e.partial_result:
                    return e.partial_result
                # 如果没有部分结果，尝试从已获取的内容中解析
                return accumulated_result
            except AgentResultError as e:
                # 继续获取成功但结果完全无效
                self.logger.warning(f"继续获取完成但结果无效: {e}")
                return accumulated_result

        except (httpx.HTTPStatusError, httpx.RequestError, KeyError, IndexError, ValueError) as e:
            self.logger.error(f"继续获取内容失败: {repr(e)}")
            # 退化：返回已获取的部分结果，而不是报错
            if accumulated_result:
                self.logger.warning(f"API不支持继续获取，返回已获取的部分结果 ({len(accumulated_result)} 字符)")
                accumulated_result = self._sanitize_result(accumulated_result)
                return (
                    accumulated_result
                    if result_handler is None
                    else result_handler(accumulated_result, prompt, self.logger)
                )
            return (
                prompt
                if error_result_handler is None
                else error_result_handler(prompt, self.logger)
            )

    def send(
            self,
            client: httpx.Client,
            prompt: str,
            system_prompt: None | str = None,
            retry=True,
            retry_count=0,
            force_json=False,
            pre_send_handler=None,
            result_handler=None,
            error_result_handler=None,
            best_partial_result: dict | None = None,
    ) -> Any:
        if system_prompt is None:
            system_prompt = self.system_prompt
        if pre_send_handler:
            system_prompt, prompt = pre_send_handler(system_prompt, prompt)

        # 新增：同步环境下的速率限制
        estimated_tokens = self._estimate_tokens(system_prompt) + self._estimate_tokens(prompt)
        self.rate_limiter.acquire_sync(tokens=estimated_tokens)

        headers, data = self._prepare_request_data(prompt, system_prompt, json_format=force_json)
        should_retry = False
        is_hard_error = False
        current_partial_result = None

        try:
            response_data = self._request_completion_sync(client, data, headers)

            # 检查 finish_reason
            choices = response_data.get("choices", [])
            if not choices:
                self.logger.error(f"API响应中未找到 choices 字段")
                raise ValueError("API响应格式错误：缺少 choices 字段")

            finish_reason = choices[0].get("finish_reason", None)
            result = choices[0].get("message", {}).get("content", "")

            # 处理不同的 finish_reason
            if finish_reason == "stop":
                # 正常结束
                pass
            elif finish_reason == "length":
                # 长度限制，尝试继续获取
                self.logger.warning(f"响应因长度限制被截断，尝试继续获取...")
                # 注意：这里传入原始result，清理工作在 _continue_fetch 最终返回时统一处理
                return self._continue_fetch(
                    client=client,
                    prompt=prompt,
                    system_prompt=system_prompt,
                    force_json=force_json,
                    pre_send_handler=pre_send_handler,
                    result_handler=result_handler,
                    error_result_handler=error_result_handler,
                    retry_count=retry_count,
                    accumulated_result=result,
                )
            elif finish_reason in ("tool_calls", "function_call"):
                # 工具调用场景，当前代码可能不支持，直接返回已获取结果
                self.logger.warning(f"finish_reason 为 '{finish_reason}'，当前不支持工具调用，返回已获取内容")
                result = self._sanitize_result(result)
                return result if result else (
                    prompt if error_result_handler is None
                    else error_result_handler(prompt, self.logger)
                )
            elif finish_reason == "content_filter":
                # 内容被过滤
                self.logger.error(f"响应内容被过滤")
                raise ValueError("内容被过滤")
            elif finish_reason is None:
                # 某些 API 可能不返回 finish_reason，将其视为正常结束
                self.logger.warning(f"API未返回 finish_reason，视为正常结束")
            else:
                # 其他未知的 finish_reason，记录警告并返回结果
                self.logger.warning(f"未知的 finish_reason: '{finish_reason}'，返回已获取内容")

            input_tokens, cached_tokens, output_tokens, reasoning_tokens, api_total_tokens = (
                extract_token_info(response_data)
            )

            self.token_counter.add(
                input_tokens, cached_tokens, output_tokens, reasoning_tokens, api_total_tokens
            )

            if retry_count > 0:
                self.logger.info(f"重试成功 (第 {retry_count}/{self.retry} 次尝试)。")

            # 清理 <think> 标签后再处理结果
            result = self._sanitize_result(result)

            return (
                result
                if result_handler is None
                else result_handler(result, prompt, self.logger)
            )
        except AgentResultError as e:
            self.logger.error(f"AI返回结果有误: {e}")
            should_retry = True
        except PartialAgentResultError as e:
            self.logger.error(f"收到部分翻译结果，将尝试重试: {e}")
            current_partial_result = e.partial_result
            should_retry = True

        except httpx.HTTPStatusError as e:
            self.logger.error(
                f"AI请求HTTP状态错误 (sync): {e.response.status_code} - {e.response.text}"
            )
            should_retry = True
            is_hard_error = True
            if e.response.status_code == 429:
                time.sleep(5)

        except httpx.RequestError as e:
            # 根据错误类型给出更清晰的提示
            if isinstance(e, httpx.ReadError):
                self.logger.error(f"AI请求读取响应失败 (sync): {type(e).__name__}: {e} (可能是服务器关闭连接或网络中断)\nprompt:{prompt}")
            elif isinstance(e, httpx.ConnectError):
                self.logger.error(f"AI请求连接失败 (sync): {type(e).__name__}: {e} (无法连接到服务器，请检查网络或base_url)\nprompt:{prompt}")
            elif isinstance(e, httpx.WriteError):
                self.logger.error(f"AI请求发送数据失败 (sync): {type(e).__name__}: {e}\nprompt:{prompt}")
            elif isinstance(e, httpx.TimeoutException):
                self.logger.error(f"AI请求超时 (sync): {type(e).__name__}: {e} (请求超过{self.timeout}秒未完成)\nprompt:{prompt}")
            else:
                self.logger.error(f"AI请求连接错误 (sync): {type(e).__name__}: {e}\nprompt:{prompt}")
            should_retry = True
            is_hard_error = True
        except (KeyError, IndexError, ValueError, json.JSONDecodeError) as e:
            self.logger.error(f"AI响应格式或值错误 (sync), 将尝试重试: {repr(e)}")
            should_retry = True
            is_hard_error = True

        if current_partial_result:
            best_partial_result = current_partial_result

        if should_retry and retry and retry_count < self.retry:
            if is_hard_error:
                if retry_count == 0:
                    if self.total_error_counter.add():
                        self.logger.error("错误次数过多，已达到上限，不再重试。")
                        with self.unresolved_error_lock:
                            self.unresolved_error_count += 1
                        return (
                            best_partial_result
                            if best_partial_result
                            else (
                                prompt
                                if error_result_handler is None
                                else error_result_handler(prompt, self.logger)
                            )
                        )
                elif self.total_error_counter.reach_limit():
                    self.logger.error("错误次数过多，已达到上限，不再为该请求重试。")
                    with self.unresolved_error_lock:
                        self.unresolved_error_count += 1
                    return (
                        best_partial_result
                        if best_partial_result
                        else (
                            prompt
                            if error_result_handler is None
                            else error_result_handler(prompt, self.logger)
                        )
                    )

            self.logger.info(f"正在重试第 {retry_count + 1}/{self.retry} 次...")
            time.sleep(0.5 * (2 ** retry_count))
            return self.send(
                client,
                prompt,
                system_prompt,
                retry=True,
                retry_count=retry_count + 1,
                force_json=force_json,
                pre_send_handler=pre_send_handler,
                result_handler=result_handler,
                error_result_handler=error_result_handler,
                best_partial_result=best_partial_result,
            )
        else:
            if should_retry:
                self.logger.error(f"所有重试均失败，已达到重试次数上限。")
                with self.unresolved_error_lock:
                    self.unresolved_error_count += 1

            if best_partial_result:
                self.logger.info("所有重试失败，但存在部分翻译结果，将使用该结果。")
                return best_partial_result

            return (
                prompt
                if error_result_handler is None
                else error_result_handler(prompt, self.logger)
            )

    def send_prompts(
            self,
            prompts: list[str],
            system_prompt: str | None = None,
            json_format=False,
            pre_send_handler: PreSendHandlerType = None,
            result_handler: ResultHandlerType = None,
            error_result_handler: ErrorResultHandlerType = None,
    ) -> list[Any]:
        def run_async_batch() -> list[Any]:
            return asyncio.run(
                self.send_prompts_async(
                    prompts=prompts,
                    system_prompt=system_prompt,
                    force_json=json_format,
                    pre_send_handler=pre_send_handler,
                    result_handler=result_handler,
                    error_result_handler=error_result_handler,
                )
            )

        try:
            asyncio.get_running_loop()
        except RuntimeError:
            # 常规同步调用：直接创建临时事件循环。
            return run_async_batch()

        # 若同步 API 被误用于已有事件循环中，不能在同一线程调用 asyncio.run。
        # 使用一个桥接线程保持原有同步行为；实际 HTTP 工作仍由可取消的异步实现完成。
        with ThreadPoolExecutor(max_workers=1) as executor:
            return executor.submit(run_async_batch).result()

    def get_full_stats(self) -> dict:
        """
        获取完整统计信息，包括token统计、请求数、未解决错误数和错误率。

        Returns:
            dict: 包含完整统计信息的字典
        """
        token_stats = self.token_counter.get_stats()
        request_count = getattr(self, '_request_count', 0)
        unresolved = self.unresolved_error_count
        error_rate = unresolved / request_count if request_count > 0 else 0.0
        return {
            "input_tokens": token_stats["input_tokens"],
            "cached_tokens": token_stats["cached_tokens"],
            "output_tokens": token_stats["output_tokens"],
            "reasoning_tokens": token_stats["reasoning_tokens"],
            "total_tokens": token_stats["total_tokens"],
            "request_count": request_count,
            "unresolved_errors": unresolved,
            "unresolved_error_rate": error_rate
        }
