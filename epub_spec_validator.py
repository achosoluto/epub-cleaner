#!/usr/bin/env python3
"""
Future EPUB spec validator module.
Provides a validate_epub function that can be expanded to run the official epubcheck tool
or any custom validation logic. Currently returns a successful placeholder.
"""
from typing import Dict


def validate_epub(path: str, strict: bool = False) -> Dict[str, object]:
    """Validate an EPUB file against the W3C EPUB specification.

    Args:
        path: Absolute path to the .epub file.
        strict: If True, treat any warning as a failure.

    Returns:
        A dict with keys:
            - ok (bool): True if validation passed.
            - messages (List[str]): Human‑readable messages from the validator.
    """
    # Placeholder implementation – always succeeds.
    # Future implementation could invoke `epubcheck` if installed:
    #   result = subprocess.run(["epubcheck", path], capture_output=True, text=True)
    #   ok = result.returncode == 0
    #   messages = result.stdout.splitlines() + result.stderr.splitlines()
    #   if strict and any("warning" in m.lower() for m in messages):
    #       ok = False
    #   return {"ok": ok, "messages": messages}
    return {"ok": True, "messages": ["Validator stub – always passes."]}
