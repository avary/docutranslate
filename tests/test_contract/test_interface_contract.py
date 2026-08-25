import ast
import inspect
from pathlib import Path

import pytest
from pydantic import TypeAdapter

import docutranslate.sdk as sdk_module
from docutranslate.core.schemas import (
    DocxWorkflowParams,
    TranslatePayload,
    XlsxWorkflowParams,
)
from docutranslate.sdk import Client
from docutranslate.server.core import TranslationService
from docutranslate.translator.ai_translator.txt_translator import (
    TXTTranslator,
    TXTTranslatorConfig,
)


SDK_INHERITED_DEFAULT_FIELDS = {
    "insert_mode",
    "separator",
    "segment_mode",
    "convert_engine",
    "mineru_token",
    "md2docx_engine",
    "model_version",
    "formula_ocr",
    "code_ocr",
    "mineru_language",
    "mineru_deploy_base_url",
    "mineru_deploy_backend",
    "mineru_deploy_effort",
    "mineru_deploy_parse_method",
    "mineru_deploy_formula_enable",
    "mineru_deploy_table_enable",
    "mineru_deploy_image_analysis",
    "mineru_deploy_start_page_id",
    "mineru_deploy_end_page_id",
    "mineru_deploy_lang_list",
    "mineru_deploy_server_url",
}

STREAM_POLICY_FIELDS = {
    "streaming",
    "stream_idle_timeout",
    "non_stream_timeout",
}


def test_sdk_sync_and_async_signatures_inherit_workflow_defaults():
    sync_params = inspect.signature(Client.translate).parameters
    async_params = inspect.signature(Client.translate_async).parameters

    for field_name in SDK_INHERITED_DEFAULT_FIELDS:
        assert sync_params[field_name].default is None
        assert async_params[field_name].default is None


@pytest.mark.asyncio
async def test_sdk_constructor_values_are_not_overwritten_by_call_defaults(
    monkeypatch, tmp_path
):
    source = tmp_path / "paper.pdf"
    source.write_bytes(b"test")
    captured = {}

    class DummyWorkflow:
        def read_path(self, path):
            self.path = path

    def create_workflow(payload):
        captured["payload"] = payload
        return DummyWorkflow()

    monkeypatch.setattr(sdk_module, "create_workflow_from_payload", create_workflow)

    client = Client(convert_engine="mineru", mineru_token="constructor-token")
    await client.translate_async(str(source), skip_translate=True)

    payload = captured["payload"]
    assert payload.convert_engine == "mineru"
    assert payload.mineru_token == "constructor-token"


@pytest.mark.asyncio
@pytest.mark.parametrize(
    ("filename", "expected_type"),
    [("encrypted.docx", DocxWorkflowParams), ("encrypted.xlsx", XlsxWorkflowParams)],
)
async def test_auto_workflow_preserves_office_password(
    monkeypatch, filename, expected_type
):
    service = TranslationService()
    captured = {}

    async def perform_translation(task_id, payload, file_contents, original_filename):
        captured["payload"] = payload

    monkeypatch.setattr(service, "_perform_translation", perform_translation)
    payload = TypeAdapter(TranslatePayload).validate_python(
        {
            "workflow_type": "auto",
            "skip_translate": True,
            "office_password": "secret-password",
        }
    )

    await service.start_translation("contract-task", payload, b"test", filename)
    await service.tasks_state["contract-task"]["current_task_ref"]

    routed_payload = captured["payload"]
    assert isinstance(routed_payload, expected_type)
    assert routed_payload.office_password == "secret-password"


def test_translator_runtime_honors_stream_policy_overrides():
    config = TXTTranslatorConfig(
        base_url="https://example.invalid/v1",
        api_key="test-key",
        model_id="test-model",
        streaming=False,
        stream_idle_timeout=17,
        non_stream_timeout=29,
    )

    translator = TXTTranslator(config)

    assert translator.translate_agent.streaming_enabled is False
    assert translator.translate_agent.stream_idle_timeout_seconds == 17
    assert translator.translate_agent.non_stream_total_timeout_seconds == 29


def test_every_translator_agent_config_forwards_stream_policy_fields():
    translator_root = (
        Path(__file__).parents[2]
        / "docutranslate"
        / "translator"
        / "ai_translator"
    )
    config_types = {
        "SegmentsTranslateAgentConfig",
        "MDTranslateAgentConfig",
        "GlossaryAgentConfig",
    }
    checked_calls = []

    for source_file in translator_root.glob("*.py"):
        tree = ast.parse(source_file.read_text(encoding="utf-8"))
        for node in ast.walk(tree):
            if not isinstance(node, ast.Call) or not isinstance(node.func, ast.Name):
                continue
            if node.func.id not in config_types:
                continue

            keywords = {keyword.arg for keyword in node.keywords}
            assert STREAM_POLICY_FIELDS <= keywords, (
                f"{source_file.name}:{node.lineno} 创建 {node.func.id} 时"
                f"没有完整透传 {sorted(STREAM_POLICY_FIELDS - keywords)}"
            )
            checked_calls.append((source_file.name, node.lineno, node.func.id))

    assert len(checked_calls) == 11
