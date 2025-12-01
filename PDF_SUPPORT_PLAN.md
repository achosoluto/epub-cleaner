# PDF Support Implementation Plan (Concise)

## Scope
- Accept existing `*.md` **and** `*.pdf` files.
- Optional `--ocr` flag for scanned PDFs.
- Output: cleaned Markdown in `--output-dir`.

## CLI Additions
- `--pdf-dir` (defaults to `--input-dir`).
- `--ocr` (store_true).

## Pipeline
1. **Discover** MD and PDF files.
2. **Convert PDFs** to temporary MD:
   - Text‑only: `pdfminer.six` → `extract_text`.
   - Scanned: `pdf2image` + `pytesseract` (when `--ocr`).
3. **Run existing cleaner** on all MD (original + temp).
4. **Write output** & CSV report.
5. **Cleanup** temporary folder.

## Files Added/Modified
- `pdf_to_md.py` (conversion helpers).
- Update `build_arg_parser()` for new flags.
- Extend file‑discovery to return PDF list.
- Temporary workspace handling in `run_manager()`.
- Update `README.md` and `requirements.txt`.

## Testing
- Unit tests for `pdf_to_text` and `pdf_to_text_ocr`.
- End‑to‑end test on mixed folder (MD + PDF) with `--dry-run`.

## Estimated Effort
~6 hours total (coding, docs, tests).

## Result
The same `epub‑cleaner` command now processes PDFs (with optional OCR) and delivers clean Markdown to the user’s Obsidian vault, preserving all safety checks.
