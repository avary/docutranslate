"""
Tests for logger module
"""
import asyncio
import io
import logging

import pytest

from docutranslate.agents.agent import Agent, AgentConfig
from docutranslate.logger.logger import global_logger
from docutranslate.server.core import QueueAndHistoryHandler
from docutranslate.utils.utils import mask_secrets


def test_logger_initialization():
    """Test that logger is properly initialized"""
    assert isinstance(global_logger, logging.Logger)
    assert global_logger.name == "TranslaterLogger"
    assert global_logger.level == logging.DEBUG


def test_logger_has_console_handler():
    """Test that logger has a StreamHandler configured"""
    handlers = global_logger.handlers
    assert len(handlers) >= 1
    assert any(isinstance(h, logging.StreamHandler) for h in handlers)


def test_logger_can_log_messages(caplog):
    """Test that logger can log messages at different levels"""
    # Set caplog to capture DEBUG level
    caplog.set_level(logging.DEBUG)

    test_messages = [
        (logging.DEBUG, "Debug message"),
        (logging.INFO, "Info message"),
        (logging.WARNING, "Warning message"),
        (logging.ERROR, "Error message"),
        (logging.CRITICAL, "Critical message"),
    ]

    for level, message in test_messages:
        global_logger.log(level, message)

    # Check that all messages were logged
    for level, message in test_messages:
        assert any(
            record.levelno == level and message in record.message
            for record in caplog.records
        )


def test_mask_secrets_covers_headers_and_common_credentials():
    raw_values = [
        "very-secret-bearer-token",
        "sk-this-is-a-very-secret-openai-key",
        "mineru-secret-token",
        "gateway-password",
    ]
    masked = mask_secrets(
        "Authorization: Bearer very-secret-bearer-token "
        "api_key=sk-this-is-a-very-secret-openai-key "
        "mineru_token=mineru-secret-token password=gateway-password"
    )

    assert all(value not in masked for value in raw_values)
    assert "[BEARER_TOKEN_HIDDEN]" in masked
    assert "[API_KEY_HIDDEN]" in masked
    assert "[MINERU_TOKEN_HIDDEN]" in masked
    assert "[HIDDEN]" in masked


def test_queue_handler_is_bounded_and_keeps_latest_entries():
    queue = asyncio.Queue(maxsize=2)
    history = []
    handler = QueueAndHistoryHandler(queue, history, 2, "test-task")
    handler.setFormatter(logging.Formatter("%(message)s"))
    logger = logging.getLogger("test.bounded-task-log")
    logger.handlers = [handler]
    logger.setLevel(logging.INFO)
    logger.propagate = False

    try:
        logger.info("first")
        logger.info("second")
        logger.info("third")

        assert history == ["second", "third"]
        assert queue.get_nowait() == "second"
        assert queue.get_nowait() == "third"
        assert handler.dropped_logs == 1
    finally:
        logger.handlers = []


@pytest.mark.asyncio
async def test_batch_logs_context_without_prompt_or_endpoint_credentials(monkeypatch):
    stream = io.StringIO()
    logger = logging.getLogger("test.safe-agent-log")
    logger.handlers = [logging.StreamHandler(stream)]
    logger.setLevel(logging.DEBUG)
    logger.propagate = False
    agent = Agent(
        AgentConfig(
            logger=logger,
            base_url=(
                "https://endpoint-user:endpoint-password@example.com:8443/v1/"
                "?api_key=endpoint-query-secret"
            ),
            api_key="request-api-secret",
            model_id="test-model",
            provider="openai",
            concurrent=1,
        )
    )

    async def fake_send_async(**_kwargs):
        return "translated"

    monkeypatch.setattr(agent, "send_async", fake_send_async)
    result = await agent.send_prompts_async(["private document content"])
    output = stream.getvalue()

    assert result == ["translated"]
    assert "[分片 1/1] 开始" in output
    assert "[分片 1/1] 完成" in output
    assert "id=" in output
    assert "private document content" not in output
    assert "endpoint-user" not in output
    assert "endpoint-password" not in output
    assert "endpoint-query-secret" not in output
    assert "request-api-secret" not in output
    assert "https://example.com:8443/v1" in output
