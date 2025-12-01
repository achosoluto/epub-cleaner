# Document Cleaner

A pragmatic, DHH-style Python tool for batch cleaning Markdown files converted from EPUBs and PDF documents.

## Philosophy: Factory vs. Showroom

- **The Factory**: Where your raw files and this script live (e.g., this repo or your downloads folder).
- **The Showroom**: Where your clean, readable notes go (e.g., your Obsidian Vault).

This tool is designed to take raw, messy Markdown from the Factory and deliver pristine files to the Showroom.

## Usage

### 1. Setup
Clone this repository:
```bash
git clone https://github.com/anthonycho/epub-cleaner.git
cd epub-cleaner
```

### 2. Run the Cleaner
Run the script pointing to your raw files and your destination:

```bash
python3 dhh_batch_clean.py \
  --input-dir "/path/to/raw/files" \
  --output-dir "/path/to/Obsidian/Vault/Cleaned" \
  --no-dry-run \
  --limit 0
```

### Options
- `--dry-run`: (Default) Don't write files, just check what would happen.
- `--limit N`: Process only the first N files (useful for testing). Set to `0` for all files.
- `--fail-fast`: Stop immediately if a file fails validation.
- `--report`: Path to the CSV report file.

## Safety Features
- **Non-Destructive**: Never overwrites source files.
- **Idempotent**: Running the tool multiple times on the same text produces the same result.
- **Content Preservation**: Uses a "Judge" (Normalizer) to ensure no meaningful content is lost during cleaning.

## PDF Support

The tool now accepts PDF files in addition to Markdown. PDFs are converted to Markdown temporarily and then processed like regular MD files.

### New CLI Flags

- `--pdf-dir`: Directory containing PDF files (default: same as `--input-dir`)
- `--ocr`: Enable OCR processing for scanned PDFs

### Dependencies

- **Mandatory**: `pdfminer.six`
- **Optional**: `pytesseract` and `pdf2image` (required for OCR functionality)

### Usage Example

```bash
python3 dhh_batch_clean.py \
  --input-dir "/path/to/raw/files" \
  --pdf-dir "/path/to/pdf/files" \
  --output-dir "/path/to/Obsidian/Vault/Cleaned" \
  --ocr \
  --no-dry-run \
  --limit 0
```
