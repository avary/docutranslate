# SPDX-FileCopyrightText: 2025 QinHan
# SPDX-License-Identifier: MPL-2.0
import logging



global_logger = logging.getLogger("TranslaterLogger")
global_logger.setLevel(logging.DEBUG)

# Avoid duplicate output when development servers reload this module.
if not any(getattr(handler, "_docutranslate_console", False) for handler in global_logger.handlers):
    console_handler = logging.StreamHandler()
    console_handler._docutranslate_console = True
    console_handler.setFormatter(
        logging.Formatter("%(asctime)s - %(levelname)s - %(message)s")
    )
    global_logger.addHandler(console_handler)
