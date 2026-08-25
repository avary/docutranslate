# SPDX-FileCopyrightText: 2025 QinHan
# SPDX-License-Identifier: MPL-2.0
import re
from dataclasses import dataclass

from .agent import Agent, AgentConfig
from ..glossary.glossary import Glossary


def get_original_markdown(prompt: str):
    match = re.search(r'<input>\n(.*)\n</input>', prompt, re.DOTALL)
    if match:
        return match.group(1)
    else:
        raise ValueError("无法从prompt中提取初始文本")


def generate_prompt(markdown_text: str, to_lang: str):
    """Build the dynamic suffix; reusable translation rules live in the system prompt."""
    return f"""Translate the following markdown input into {to_lang}:
<input>
{markdown_text}
</input>
"""


def _build_system_prompt(to_lang: str) -> str:
    return f"""# Role
You are a professional machine translation engine.

# Task
Treat each input as markdown and translate it into {to_lang}.

# Requirements
- Output the translated markdown only. Do not add explanations or notes and do not wrap the result in a markdown code block.
- Preserve special tags and non-translatable elements such as code, brand names, and technical terms.
- Represent every formula as valid, parsable LaTeX enclosed by `$`, `\\(\\)`, or `$$`; correct malformed formula delimiters when necessary.
- Remove or correct obviously abnormal characters without changing the original meaning.
- Preserve citation text exactly and do not translate it. Examples include `[1] Author A, Author B. "Original Title". Journal, 2023.` and `[2] 作者C. 《中文标题》. 期刊, 2022.`
"""


@dataclass
class MDTranslateAgentConfig(AgentConfig):
    to_lang: str
    custom_prompt: str | None = None
    glossary_dict: dict[str, str] | None = None


class MDTranslateAgent(Agent):
    def __init__(self, config: MDTranslateAgentConfig):
        super().__init__(config)
        self.to_lang = config.to_lang
        self.system_prompt = _build_system_prompt(self.to_lang)
        self.custom_prompt = config.custom_prompt
        if config.custom_prompt:
            self.system_prompt += "\n# **Important rules or background** \n" + self.custom_prompt + '\nEND\n'
        self.glossary_dict = config.glossary_dict

    def _pre_send_handler(self, system_prompt, prompt):
        if self.glossary_dict:
            glossary = Glossary(glossary_dict=self.glossary_dict)
            incremental_glossary = glossary.build_incremental_prompt(prompt)
            if incremental_glossary:
                prompt = incremental_glossary + "\n" + prompt
        return system_prompt, prompt

    def send_chunks(self, prompts: list[str]):
        prompts = [generate_prompt(prompt, self.to_lang) for prompt in prompts]
        return super().send_prompts(prompts=prompts, pre_send_handler=self._pre_send_handler,
                                    error_result_handler=lambda prompt, logger: get_original_markdown(prompt))

    async def send_chunks_async(self, prompts: list[str]):
        prompts = [generate_prompt(prompt, self.to_lang) for prompt in prompts]
        return await super().send_prompts_async(prompts=prompts, pre_send_handler=self._pre_send_handler,
                                                error_result_handler=lambda prompt, logger: get_original_markdown(
                                                    prompt))

    def update_glossary_dict(self, update_dict: dict | None):
        if self.glossary_dict is None:
            self.glossary_dict = {}
        if update_dict is not None:
            self.glossary_dict = self.glossary_dict | update_dict
