# SPDX-FileCopyrightText: 2025 QinHan
# SPDX-License-Identifier: MPL-2.0
import csv
import re
from io import StringIO

from docutranslate.ir.document import Document


class Glossary:
    def __init__(self, glossary_dict: dict[str,str] = None):
        if glossary_dict:
            self.glossary_dict = glossary_dict
        else:
            self.glossary_dict={}


    def update(self, update_dict: dict[str,str]):
        for src, dst in update_dict.items():
            if src.strip().lower() not in self.glossary_dict:
                self.glossary_dict[src.strip().lower()] = dst

    def build_incremental_prompt(self, text: str) -> str:
        """Build a deterministic glossary containing only terms used by this input."""
        text_casefold = text.casefold()
        matched_entries = sorted(
            (
                (src, dst)
                for src, dst in self.glossary_dict.items()
                if src and src.casefold() in text_casefold
            ),
            key=lambda item: (item[0].casefold(), item[0]),
        )
        if not matched_entries:
            return ""

        glossary_lines = "\n".join(
            f"{src}=>{dst}" for src, dst in matched_entries
        )
        return (
            "<incremental_glossary>\n"
            "Use the following glossary entries only when the corresponding source "
            "term appears in this input:\n"
            f"{glossary_lines}\n"
            "Glossary ends\n"
            "</incremental_glossary>\n"
        )

    def append_system_prompt(self, text: str):
        """Compatibility alias for callers using the previous method name."""
        return self.build_incremental_prompt(text)

    @staticmethod
    def glossary_dict2csv(glossary_dict: dict[str, str], delimiter=",", stem="glossary_gen") -> Document:
        csv_rows = [[src, dst] for src, dst in glossary_dict.items()]
        content = StringIO()
        writer = csv.writer(content, delimiter=delimiter)
        writer.writerow(['src', 'dst'])
        writer.writerows(csv_rows)
        bom = '\ufeff'
        content_with_bom = bom + content.getvalue()
        return Document.from_bytes(content=content_with_bom.encode("utf-8"), suffix=".csv", stem=stem)
