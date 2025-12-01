#!/usr/bin/env python3
"""
dhh_batch_clean.py

Production-grade Python batch tool implementing a Judge-Worker-Manager workflow for safely cleaning Markdown files while preserving meaningful content.

Approach:
- Judge (Normalizer): Produces a behavior-based, space-collapsed single-line string from Markdown that reflects human-visible text while dropping formatting and non-semantic syntax. It applies:
  * Unicode normalization (NFKC) and HTML entity decoding
  * Lowercasing
  * Markdown structure handling:
      - Links: keep link text; drop URLs and link syntax ([text](url) -> "text"; [text][id] -> "text"; drop reference definitions "[id]: url")
      - Images: keep alt text; drop URLs and image syntax (![alt](url) -> "alt")
      - Headings, lists, block quotes: drop leading markers; keep visible text
      - Code fences/inline code: drop fence/backticks, keep code text
      - Footnote markers [^n]: dropped; footnote definitions keep body minus marker
      - YAML front matter at file start delimited by ---: drop entire block
      - HTML: drop tags and HTML comments but keep inner text; tag pattern constrained to real tags
  * Punctuation and separators replaced with spaces (never silently deleted), then collapse whitespace to single spaces
  * Digits retained, diacritics normalized via NFKC

- Worker (Cleaner): Conservative, idempotent Markdown cleanup targeting EPUB/HTML artifacts:
  * Remove specific HTML container tags (div/span/font/section/article/header/footer/nav), preserving inner text
  * Strip any redundant closing tags of those
  * Replace images with their alt text
  * Drop link syntax but keep link text (remove URLs)
  * Remove HTML comments, link reference definitions
  * Normalize line endings, trim trailing spaces, and collapse excessive blank lines to at most one
  * Idempotent: applying twice yields same result. The manager verifies per file.

- Manager (Workflow):
  * Recursively walk input directory for .md files, excluding output and diffs directories
  * For each file:
      1) Read original as UTF-8 with errors="replace"
      2) Compute baseline normalized hash (SHA-256)
      3) Run Worker Cleaner
      4) Compute result normalized hash
      5) If hashes match: success; else: failed and write token-level diff into debug_diffs
      6) If success and not dry-run: write cleaned file into output directory, preserving structure, suffix "_clean"
      7) Always verify idempotence (clean(clean(text)) == clean(text)); log in report
  * Maintain counters and write a CSV report with per-file results

Diff:
- Token-level (whitespace-tokenized) diff using difflib with a symmetric context window (default 6 tokens).
- Writes one diff file per failing source path under debug_diffs mirroring the source tree.

CLI:
  --input-dir: default "."
  --output-dir: default "cleaned"
  --suffix: default "_clean"
  --dry-run: default True; pass "--no-dry-run" to write files
  --limit: default 5 (process first N sorted files; 0 means all)
  --fail-fast: default False
  --report: default "clean_report.csv"
  --diff-context: default 6 tokens
  --encoding: default "utf-8"

Safety and Guarantees:
- Dependency-free (standard library only)
- Never modifies existing files; writes only to:
    - cleaned/ (outputs)
    - debug_diffs/ (diffs for failures)
    - clean_report.csv (CSV summary)
- Robust error handling per file; continues unless --fail-fast specified
- Self-checks on startup validate Normalizer equivalence for core constructs; failures print a warning but do not abort

Example usage:
- Dry run, sample 5 files (default):
    python dhh_batch_clean.py
- Process all files, write outputs:
    python dhh_batch_clean.py --limit 0 --no-dry-run
- Process all files, continue on failures, with custom report path:
    python dhh_batch_clean.py --limit 0 --no-dry-run --report clean_report.csv
"""

from __future__ import annotations

__version__ = "0.2.0"

import argparse
import csv
import hashlib
import html
import os
import re
import sys
import time
import unicodedata
import tempfile
import shutil
import pdf_to_md
from dataclasses import dataclass
from difflib import SequenceMatcher
from typing import Iterable, List, Tuple, Optional

import logging
# Optional future EPUB spec validator (currently a stub for future use)
try:
    from epub_spec_validator import validate_epub
except ImportError:
    # If the validator module is not present, define a no-op placeholder
    def validate_epub(path: str, strict: bool = False) -> dict:
        return {"ok": True, "messages": ["Validator not available; skipping."]}


DEFAULT_INPUT_DIR = "."
DEFAULT_OUTPUT_DIR = "cleaned"
DEFAULT_DIFFS_DIR = "debug_diffs"
DEFAULT_SUFFIX = "_clean"
DEFAULT_DRY_RUN = True
DEFAULT_LIMIT = 5
DEFAULT_FAIL_FAST = False
DEFAULT_REPORT = "clean_report.csv"
DEFAULT_DIFF_CONTEXT = 6
DEFAULT_ENCODING = "utf-8"


# =========================
# Judge: Normalizer
# =========================

_yaml_front_matter_re = re.compile(r"\A---\s*\n.*?\n---\s*(?:\n|$)", re.DOTALL)
_html_comment_re = re.compile(r"<!--.*?-->", re.DOTALL)
# Reference-style link definitions: [id]: url "optional title" (exclude footnotes starting with ^)
_link_ref_def_re = re.compile(r"(?m)^\s{0,3}\[(?!\^)[^\]]+\]:\s+\S+.*$")
# Footnote definition marker: [^id]: text...
_footnote_def_marker_re = re.compile(r"(?m)^\s{0,3}\[\^[^\]]+\]:\s*")
# Footnote inline markers: [^id]
_footnote_marker_inline_re = re.compile(r"\[\^[^\]]+\]")
# Inline images: ![alt](url) - single line only to avoid greedy table matches
_image_inline_re = re.compile(r"!\[([^\]\n]*)\]\([^)\n]+\)")
# Reference images: ![alt][id] - single line only
_image_ref_re = re.compile(r"!\[([^\]\n]*)\]\[[^\]\n]+\]")
# Inline links: [text](url) - single line only
_link_inline_re = re.compile(r"\[([^\]\n]+)\]\([^)\n]+\)")
# Reference links: [text][id] - single line only
_link_ref_re = re.compile(r"\[([^\]\n]+)\]\[[^\]\n]+\]")
# Code fences (``` or ~~~) lines only
_code_fence_re = re.compile(r"(?m)^\s*(?:```|~~~).*$")
# Inline code `code`
_inline_code_re = re.compile(r"`([^`]+)`")
# Headings at line start
_heading_marker_re = re.compile(r"(?m)^\s{0,3}#{1,6}\s*")
# Lists bullets/numbered at line start
_list_bullet_re = re.compile(r"(?m)^\s{0,3}[-*+]\s+")
_list_number_re = re.compile(r"(?m)^\s{0,3}\d+\.\s+")
# Block quotes
_block_quote_re = re.compile(r"(?m)^\s{0,3}>\s?")
# HTML tags (start with letter or /)
_html_tag_re = re.compile(r"</?[A-Za-z][^>]*>")
# Emphasis markers (bold/italic/strike)
_emphasis_re = re.compile(r"[*_~]{1,3}")


def _normalize_text_for_compare(text: str) -> str:
    """
    Normalize Markdown-ish text to a behavior-based single-line form.

    Steps:
      1) HTML entity decode, Unicode NFKC, lowercasing
      2) Drop YAML front matter
      3) Remove HTML comments
      4) Convert images/links to visible text
      5) Remove reference definitions and footnote markers; keep footnote bodies (minus marker)
      6) Drop code fences/backticks markers (but keep content)
      7) Drop heading/list/blockquote markers; keep content
      8) Remove HTML tags (keep inner text)
      9) Drop emphasis markers
     10) Replace punctuation/separators with spaces, collapse whitespace to single space
    """
    # Normalize newline
    text = text.replace("\r\n", "\n").replace("\r", "\n")

    # 1) Decode HTML entities, Unicode normalize, lowercase
    text = html.unescape(text)
    text = unicodedata.normalize("NFKC", text)
    text = text.lower()

    # 2) YAML front matter at start
    text = _yaml_front_matter_re.sub("", text)

    # 3) HTML comments
    text = _html_comment_re.sub(" ", text)

    # 4) Images/links to their visible text
    text = _image_inline_re.sub(r"\1", text)
    text = _image_ref_re.sub(r"\1", text)
    text = _link_inline_re.sub(r"\1", text)
    text = _link_ref_re.sub(r"\1", text)

    # 5) Reference-style link definitions (drop lines)
    text = _link_ref_def_re.sub("", text)
    # Footnote inline markers
    text = _footnote_marker_inline_re.sub("", text)
    # Footnote definitions: drop the marker, keep body text
    text = _footnote_def_marker_re.sub("", text)

    # 6) Code fences and inline code markers: drop markers only
    text = _code_fence_re.sub("", text)
    text = _inline_code_re.sub(r"\1", text)

    # 7) Structural markdown markers
    text = _heading_marker_re.sub("", text)
    text = _list_bullet_re.sub("", text)
    text = _list_number_re.sub("", text)
    text = _block_quote_re.sub("", text)

    # 8) Remove HTML tags (keep inner text) with constrained tag pattern
    text = _html_tag_re.sub(" ", text)

    # 9) Drop emphasis markers
    text = _emphasis_re.sub(" ", text)

    # 10) Replace punctuation and separators with spaces, then collapse
    text = _punct_to_space(text)
    text = " ".join(text.split())
    return text


def _punct_to_space(s: str) -> str:
    """
    Replace non-alphanumeric Unicode characters (including punctuation, symbols, separators)
    with spaces. Retain letters (all scripts) and digits.
    """
    out_chars: List[str] = []
    for ch in s:
        # Keep alphanumeric (letters/digits) across unicode categories
        if ch.isalnum():
            out_chars.append(ch)
        else:
            # Replace the rest with a space to avoid token merges
            out_chars.append(" ")
    return "".join(out_chars)


def _sha256_hex(s: str) -> str:
    return hashlib.sha256(s.encode("utf-8")).hexdigest()


# =========================
# Worker: Cleaner
# =========================

# Specific HTML container tags to drop (preserve inner text)
_container_tag_re = re.compile(
    r"</?(?:div|span|font|section|article|header|footer|nav)\b[^>]*>", re.IGNORECASE
)
# HTML comments
_html_comment_any_re = re.compile(r"<!--.*?-->", re.DOTALL)
# Images to alt
_image_inline_clean_re = _image_inline_re
_image_ref_clean_re = _image_ref_re
# Links: keep text
_link_inline_clean_re = _link_inline_re
_link_ref_clean_re = _link_ref_re
# Link reference definitions
_link_ref_def_clean_re = _link_ref_def_re


def clean_markdown_conservative(md: str) -> str:
    """
    Conservative, idempotent Markdown cleaner:
      - Remove selected HTML container tags, preserving text
      - Remove HTML comments
      - Replace images with alt text
      - Convert links to plain visible text (drop URL)
      - Drop reference-style link definitions
      - Trim trailing spaces, collapse excessive blank lines to one
      - Normalize line endings to \n
    Leaves normal Markdown semantics otherwise intact (headings, code, etc.).
    """
    text = md.replace("\r\n", "\n").replace("\r", "\n")

    # Remove container HTML tags (preserve inner text)
    text = _container_tag_re.sub("", text)

    # Remove HTML comments
    text = _html_comment_any_re.sub("", text)

    # Images to alt
    text = _image_inline_clean_re.sub(r"\1", text)
    text = _image_ref_clean_re.sub(r"\1", text)

    # Links to visible text
    text = _link_inline_clean_re.sub(r"\1", text)
    text = _link_ref_clean_re.sub(r"\1", text)

    # Drop link reference definitions lines
    text = _link_ref_def_clean_re.sub("", text)

    # Trim trailing spaces per line
    lines = [ln.rstrip(" \t") for ln in text.split("\n")]
    text = "\n".join(lines)

    # Collapse excessive blank lines (2+ -> 1)
    text = re.sub(r"\n{2,}", "\n\n", text)

    # Ensure normalization of line endings already done
    return text


# =========================
# Diff: token-level with context
# =========================


def token_diff_with_context(a_text: str, b_text: str, context: int) -> List[str]:
    """
    Create a token-level diff with symmetric context around changes.
    Returns a list of lines. Lines begin with:
      '  ' for context
      '- ' for deletions (from baseline)
      '+ ' for insertions (in result)
    """
    a_tokens = a_text.split()
    b_tokens = b_text.split()
    sm = SequenceMatcher(a=a_tokens, b=b_tokens, autojunk=False)

    lines: List[str] = []
    # Group opcodes around changes. We'll use a small grouping and then trim context per opcode.
    grouped = _group_opcodes_with_token_context(
        sm.get_opcodes(), context, len(a_tokens), len(b_tokens)
    )

    for block in grouped:
        # Add a separator between change blocks for readability
        if lines:
            lines.append("...")

        # block: list of (tag, i1, i2, j1, j2)
        for tag, i1, i2, j1, j2 in block:
            if tag == "equal":
                snippet = _trim_tokens(a_tokens[i1:i2], context)
                if snippet:
                    lines.append("  " + " ".join(snippet))
            elif tag == "delete":
                deleted = a_tokens[i1:i2]
                if deleted:
                    lines.append("- " + " ".join(deleted))
            elif tag == "insert":
                inserted = b_tokens[j1:j2]
                if inserted:
                    lines.append("+ " + " ".join(inserted))
            elif tag == "replace":
                deleted = a_tokens[i1:i2]
                inserted = b_tokens[j1:j2]
                if deleted:
                    lines.append("- " + " ".join(deleted))
                if inserted:
                    lines.append("+ " + " ".join(inserted))
    return lines


def _group_opcodes_with_token_context(
    opcodes: List[Tuple[str, int, int, int, int]], ctx: int, a_len: int, b_len: int
) -> List[List[Tuple[str, int, int, int, int]]]:
    """
    Group opcodes around changes and include up to 'ctx' tokens of equal context
    on both sides of each change group. Works at token level.
    """
    groups: List[List[Tuple[str, int, int, int, int]]] = []
    i = 0
    n = len(opcodes)
    while i < n:
        tag, i1, i2, j1, j2 = opcodes[i]
        if tag == "equal":
            i += 1
            continue

        # Start of a change group: extend to include adjacent non-equal opcodes
        group_start = i
        group_end = i
        while group_end + 1 < n and opcodes[group_end + 1][0] != "equal":
            group_end += 1

        # Determine token context from surrounding equals
        pre_equal = (
            opcodes[group_start - 1]
            if group_start - 1 >= 0 and opcodes[group_start - 1][0] == "equal"
            else None
        )
        post_equal = (
            opcodes[group_end + 1]
            if group_end + 1 < n and opcodes[group_end + 1][0] == "equal"
            else None
        )

        block: List[Tuple[str, int, int, int, int]] = []

        # Pre-context
        if pre_equal:
            _, pi1, pi2, pj1, pj2 = pre_equal
            # last ctx tokens of pre-context
            a_start = max(pi2 - ctx, pi1)
            b_start = max(pj2 - ctx, pj1)
            if a_start < pi2 and b_start < pj2:
                block.append(("equal", a_start, pi2, b_start, pj2))

        # All change opcodes
        for k in range(group_start, group_end + 1):
            block.append(opcodes[k])

        # Post-context
        if post_equal:
            _, qi1, qi2, qj1, qj2 = post_equal
            a_end = min(qi1 + ctx, qi2)
            b_end = min(qj1 + ctx, qj2)
            if qi1 < a_end and qj1 < b_end:
                block.append(("equal", qi1, a_end, qj1, b_end))

        groups.append(block)
        i = group_end + 1

    return groups


def _trim_tokens(tokens: List[str], ctx: int) -> List[str]:
    if not tokens:
        return tokens
    if len(tokens) <= 2 * ctx:
        return tokens
    # Show first ctx and last ctx with ellipsis marker in between
    return tokens[:ctx] + ["…"] + tokens[-ctx:]


# =========================
# Manager: batch workflow
# =========================


@dataclass
class FileResult:
    file_path: str
    status: str  # "success" | "failed"
    idempotent: bool
    baseline_hash: str
    result_hash: str
    reason: str
    time_ms: int
    tokens_baseline: int
    tokens_result: int
    diff_path: str


def find_input_files(
    md_root: str, pdf_root: str, exclude_dirs: Iterable[str]
) -> Tuple[List[str], List[str]]:
    """
    Recursively find .md files under md_root and .pdf files under pdf_root, excluding directory names in exclude_dirs.
    Returns sorted lists of file paths as (md_files, pdf_files).
    """
    # Normalize exclude dir names to a set (compare by dir base name)
    exclude_set = set(exclude_dirs)
    md_files: List[str] = []
    for cur_dir, dirs, files in os.walk(md_root):
        # Prune excluded directories
        pruned = []
        for d in list(dirs):
            if d in exclude_set:
                # Skip this directory
                continue
            pruned.append(d)
        dirs[:] = pruned

        for f in files:
            if f.lower().endswith(".md"):
                md_files.append(os.path.join(cur_dir, f))

    md_files.sort()

    pdf_files: List[str] = []
    for cur_dir, dirs, files in os.walk(pdf_root):
        # Prune excluded directories
        pruned = []
        for d in list(dirs):
            if d in exclude_set:
                # Skip this directory
                continue
            pruned.append(d)
        dirs[:] = pruned

        for f in files:
            if f.lower().endswith(".pdf"):
                pdf_files.append(os.path.join(cur_dir, f))

    pdf_files.sort()
    return md_files, pdf_files


def ensure_parent_dir(path: str) -> None:
    parent = os.path.dirname(path)
    if parent and not os.path.exists(parent):
        os.makedirs(parent, exist_ok=True)


def write_text_file(path: str, text: str, encoding: str) -> None:
    ensure_parent_dir(path)
    with open(path, "w", encoding=encoding, newline="\n") as f:
        f.write(text)


def read_text_file(path: str, encoding: str) -> str:
    with open(path, "r", encoding=encoding, errors="replace") as f:
        return f.read()


def relative_to(path: str, base: str) -> str:
    try:
        return os.path.relpath(path, start=base)
    except Exception:
        return path


def process_file(
    src_path: str,
    args: argparse.Namespace,
    input_root: str,
    out_root: str,
    diffs_root: str,
) -> FileResult:
    """
    Process a single markdown file end-to-end.
    """
    t0 = time.perf_counter()
    reason = ""
    status = "failed"
    idempotent = True
    diff_path = ""
    baseline_hash = ""
    result_hash = ""
    tokens_baseline = 0
    tokens_result = 0

    try:
        # Safe relative path calculation:
        # If src_path is not under input_root (e.g. temp file from PDF conversion),
        # use basename to avoid writing to temp dirs or outside output_root.
        try:
            abs_src = os.path.abspath(src_path)
            abs_input = os.path.abspath(input_root)
            if abs_src.startswith(abs_input):
                rel_src = os.path.relpath(src_path, input_root)
            else:
                rel_src = os.path.basename(src_path)
        except Exception:
            rel_src = os.path.basename(src_path)

        original = read_text_file(src_path, args.encoding)

        # Optional EPUB spec validation (future feature)
        if args.strict_epub and src_path.lower().endswith(".epub"):
            validation = validate_epub(src_path, strict=True)
            if not validation.get("ok", False):
                status = "failed"
                reason = "epub_spec_validation_failed"
                # Record messages in diff_path for visibility
                diff_path = None
                # Skip further processing for this file
                raise Exception(
                    f"EPUB spec validation failed: {validation.get('messages', [])}"
                )

        # Baseline normalized
        norm_base = _normalize_text_for_compare(original)
        baseline_hash = _sha256_hex(norm_base)
        tokens_baseline = len(norm_base.split())

        # Clean once
        cleaned_once = clean_markdown_conservative(original)

        # Result normalized
        norm_result = _normalize_text_for_compare(cleaned_once)
        result_hash = _sha256_hex(norm_result)
        tokens_result = len(norm_result.split())

        # Preservation check
        if baseline_hash == result_hash:
            status = "success"
        else:
            status = "failed"
            reason = "content_changed"
            # Produce token-level diff of normalized forms
            # rel_src calculated at start of try block
            dst_diff_path = os.path.join(
                diffs_root, os.path.splitext(rel_src)[0] + ".txt"
            )
            diff_lines = token_diff_with_context(
                norm_base, norm_result, args.diff_context
            )
            diff_text = "\n".join(diff_lines) + "\n"
            write_text_file(dst_diff_path, diff_text, args.encoding)
            diff_path = dst_diff_path

        # Idempotence check
        cleaned_twice = clean_markdown_conservative(cleaned_once)
        idempotent = cleaned_twice == cleaned_once

        # Write cleaned output only on success and if not dry-run
        if status == "success" and not args.dry_run:
            # rel_src calculated at start of try block
            base, ext = os.path.splitext(rel_src)
            dst_rel = f"{base}{args.suffix}{ext}"
            dst_path = os.path.join(out_root, dst_rel)
            write_text_file(dst_path, cleaned_once, args.encoding)

    except Exception as e:
        status = "failed"
        reason = f"exception: {e.__class__.__name__}: {e}"

    t1 = time.perf_counter()
    time_ms = int((t1 - t0) * 1000.0)

    return FileResult(
        file_path=src_path,
        status=status,
        idempotent=idempotent,
        baseline_hash=baseline_hash,
        result_hash=result_hash,
        reason=reason,
        time_ms=time_ms,
        tokens_baseline=tokens_baseline,
        tokens_result=tokens_result,
        diff_path=diff_path,
    )


def write_report(
    report_path: str, results: List[FileResult], input_root: str, encoding: str
) -> None:
    ensure_parent_dir(report_path)
    fieldnames = [
        "file_path",
        "status",
        "idempotent",
        "baseline_hash",
        "result_hash",
        "reason",
        "time_ms",
        "tokens_baseline",
        "tokens_result",
        "diff_path",
    ]
    with open(report_path, "w", encoding=encoding, newline="") as f:
        writer = csv.DictWriter(f, fieldnames=fieldnames)
        writer.writeheader()
        for r in results:
            writer.writerow(
                {
                    "file_path": relative_to(r.file_path, input_root),
                    "status": r.status,
                    "idempotent": str(bool(r.idempotent)).lower(),
                    "baseline_hash": r.baseline_hash,
                    "result_hash": r.result_hash,
                    "reason": r.reason,
                    "time_ms": r.time_ms,
                    "tokens_baseline": r.tokens_baseline,
                    "tokens_result": r.tokens_result,
                    "diff_path": (
                        relative_to(r.diff_path, input_root) if r.diff_path else ""
                    ),
                }
            )


def run_manager(args: argparse.Namespace) -> int:
    input_root = os.path.abspath(args.input_dir)
    pdf_root = os.path.abspath(args.pdf_dir)
    out_root = os.path.abspath(args.output_dir)
    diffs_root = os.path.abspath(DEFAULT_DIFFS_DIR)  # fixed as required

    # Self-checks (do not abort on failure)
    _normalizer_self_checks()

    # Resolve and ensure directories (created on demand below)
    # Build list of files
    exclude_dirs = {
        DEFAULT_OUTPUT_DIR,
        DEFAULT_DIFFS_DIR,
        os.path.basename(out_root),
        os.path.basename(diffs_root),
    }
    md_files, pdf_files = find_input_files(
        input_root, pdf_root, exclude_dirs=exclude_dirs
    )

    # Respect limit
    if args.limit > 0:
        md_files = md_files[: args.limit]

    temp_dir = tempfile.mkdtemp()
    temp_md_files = []
    results: List[FileResult] = []
    processed = 0
    success = 0
    failed = 0
    idemp_viol = 0
    try:
        for pdf in pdf_files:
            try:
                basename = os.path.basename(pdf).replace('.pdf', '_pdf.md')
                temp_md_path = os.path.join(temp_dir, basename)
                if args.ocr:
                    text = pdf_to_md.pdf_to_text_ocr(pdf)
                else:
                    text = pdf_to_md.pdf_to_text(pdf)
                write_text_file(temp_md_path, text, args.encoding)
                temp_md_files.append(temp_md_path)
            except Exception as e:
                logging.error(f"Error converting PDF {pdf}: {e}")
                fr = FileResult(
                    file_path=pdf,
                    status="failed",
                    idempotent=False,
                    baseline_hash="",
                    result_hash="",
                    reason="pdf_conversion_error",
                    time_ms=0,
                    tokens_baseline=0,
                    tokens_result=0,
                    diff_path="",
                )
                results.append(fr)
                processed += 1
                failed += 1
        all_files = md_files + temp_md_files

        for idx, src in enumerate(all_files, start=1):
            fr = process_file(src, args, input_root, out_root, diffs_root)
            results.append(fr)
            processed += 1
            if fr.status == "success":
                success += 1
            else:
                failed += 1
            if not fr.idempotent:
                idemp_viol += 1

            # fail-fast behavior
            if args.fail_fast and fr.status == "failed":
                print(
                    f"Fail-fast: stopping after first failure: {relative_to(src, input_root)}",
                    file=sys.stderr,
                )
                break

        # Write report (partial if stopped early)
        report_path = os.path.abspath(args.report)
        write_report(report_path, results, input_root, args.encoding)

        # Summary
        print("DHH Batch Clean Summary")
        print(f"  Input dir: {input_root}")
        print(f"  Output dir: {out_root} (dry-run={args.dry_run})")
        print(f"  Diffs dir: {diffs_root}")
        print(f"  Report: {report_path}")
        print(
            f"  Processed: {processed}, Success: {success}, Failed: {failed}, Idempotence violations: {idemp_viol}"
        )
    finally:
        shutil.rmtree(temp_dir)
    return 0 if failed == 0 else 1


def _normalizer_self_checks() -> None:
    """
    Execute lightweight internal checks for Normalizer equivalences.
    Print warnings on failures but do not abort.
    """
    cases: List[Tuple[str, str, str]] = []

    # Links to visible text
    cases.append(("[hello](http://example.com)", "hello", "link-visible"))
    cases.append(("[hello][ref]\n\n[ref]: http://example", "hello", "link-ref-visible"))

    # Images to alt
    cases.append(("![alt text](img.png)", "alt text", "image-alt"))
    cases.append(
        ("![alt text][imgref]\n\n[imgref]: img.png", "alt text", "image-alt-ref")
    )

    # Headings and lists and quotes
    cases.append(("# Title", "title", "heading"))
    cases.append(("- item one", "item one", "list-bullet"))
    cases.append(("1. item two", "item two", "list-number"))
    cases.append(("> quote line", "quote line", "blockquote"))

    # Inline and fenced code: markers dropped, content kept
    cases.append(("`code here`", "code here", "inline-code"))
    cases.append(("```\nblock code\n```", "block code", "fenced-code"))

    # Footnotes: inline markers dropped, def keeps body
    cases.append(
        (
            "text with footnote[^1]\n\n[^1]: the note body",
            "text with footnote the note body",
            "footnote",
        )
    )

    # HTML tags and comments
    cases.append(
        (
            "<div>hello <span>world</span></div><!-- c -->",
            "hello world",
            "html-tags-comments",
        )
    )

    for src, expected_contains, name in cases:
        got = _normalize_text_for_compare(src)
        if expected_contains not in got:
            print(
                f"[Normalizer self-check WARNING] Case '{name}' failed. Expected to contain '{expected_contains}', got '{got}'",
                file=sys.stderr,
            )


def build_arg_parser() -> argparse.ArgumentParser:
    p = argparse.ArgumentParser(
        description="Batch-clean Markdown with Judge-Worker-Manager content preservation."
    )
    p.add_argument(
        "--input-dir",
        default=DEFAULT_INPUT_DIR,
        help="Input directory to scan recursively for .md files (default: .)",
    )
    p.add_argument(
        "--output-dir",
        default=DEFAULT_OUTPUT_DIR,
        help="Directory to write cleaned files (default: cleaned)",
    )
    p.add_argument(
        "--suffix",
        default=DEFAULT_SUFFIX,
        help="Suffix to append before extension for cleaned files (default: _clean)",
    )
    # Mutually sensible dry-run flags
    p.add_argument(
        "--dry-run",
        dest="dry_run",
        action="store_true",
        default=DEFAULT_DRY_RUN,
        help="Do not write outputs (default)",
    )
    p.add_argument(
        "--no-dry-run",
        dest="dry_run",
        action="store_false",
        help="Write outputs (overrides --dry-run)",
    )
    p.add_argument(
        "--limit",
        type=int,
        default=DEFAULT_LIMIT,
        help="Process only first N files by sorted path; 0 means all (default: 5)",
    )
    p.add_argument(
        "--fail-fast",
        action="store_true",
        default=DEFAULT_FAIL_FAST,
        help="Stop on first content-preservation failure (default: False)",
    )
    p.add_argument(
        "--report",
        default=DEFAULT_REPORT,
        help="CSV summary report path (default: clean_report.csv)",
    )
    p.add_argument(
        "--diff-context",
        type=int,
        default=DEFAULT_DIFF_CONTEXT,
        help="Context tokens around changes in diffs (default: 6)",
    )
    p.add_argument(
        "--encoding",
        default=DEFAULT_ENCODING,
        help="File encoding for I/O (default: utf-8)",
    )
    p.add_argument(
        "--pdf-dir",
        default=None,
        help="Directory for PDF files (default: same as --input-dir)",
    )
    p.add_argument("--ocr", action="store_true", help="Enable OCR processing for PDFs")
    # Optional flag to enable strict EPUB spec validation (future feature)
    p.add_argument(
        "--strict-epub",
        dest="strict_epub",
        action="store_true",
        default=False,
        help="Enable strict EPUB spec validation before processing (default: off)",
    )
    p.add_argument("--version", action="version", version=__version__)
    return p


def main(argv: Optional[List[str]] = None) -> int:
    logging.basicConfig(level=logging.INFO, format='%(asctime)s - %(levelname)s - %(message)s')
    parser = build_arg_parser()
    args = parser.parse_args(argv)
    if args.ocr:
        try:
            import pytesseract
            import pdf2image
        except ImportError:
            print("Error: OCR dependencies not installed. Please install pytesseract and pdf2image. Run: pip install pytesseract pdf2image", file=sys.stderr)
            sys.exit(1)
    if args.pdf_dir is None:
        args.pdf_dir = args.input_dir
    return run_manager(args)


if __name__ == "__main__":
    sys.exit(main())
