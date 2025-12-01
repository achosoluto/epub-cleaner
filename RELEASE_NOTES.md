## Version 0.2.0

- Added PDF support: The tool now accepts PDF files in addition to Markdown files. PDFs are converted to Markdown temporarily and then processed like regular MD files.
- New CLI options:
  - `--pdf-dir`: Directory containing PDF files (defaults to `--input-dir`)
  - `--ocr`: Enable OCR processing for scanned PDFs
- Error handling improvements: Enhanced error handling for PDF processing, including custom exceptions for parsing errors, syntax errors, and OCR failures.
- Tests added: Comprehensive unit tests for PDF text extraction and OCR functionality, covering success cases, error scenarios, and edge cases like empty PDFs.
- Other enhancements: Updated dependencies (added pdfminer.six, optional pytesseract and pdf2image), updated README.md with PDF support documentation, and updated requirements.txt.

## 버전 0.2.0

- PDF 지원 추가: 도구가 이제 Markdown 파일 외에도 PDF 파일을 받아들입니다. PDF는 일시적으로 Markdown으로 변환된 후 일반 MD 파일처럼 처리됩니다.
- 새로운 CLI 옵션:
  - `--pdf-dir`: PDF 파일이 포함된 디렉토리 (기본값: `--input-dir`)
  - `--ocr`: 스캔된 PDF에 대한 OCR 처리 활성화
- 오류 처리 개선: PDF 구문 분석 오류, 구문 오류 및 OCR 실패를 포함한 PDF 처리에 대한 사용자 정의 예외를 포함한 향상된 오류 처리.
- 테스트 추가: PDF 텍스트 추출 및 OCR 기능에 대한 포괄적인 단위 테스트로 성공 사례, 오류 시나리오 및 빈 PDF와 같은 엣지 케이스를 다룹니다.
- 기타 개선 사항: 종속성 업데이트 (pdfminer.six 추가, 선택적 pytesseract 및 pdf2image), PDF 지원 문서로 README.md 업데이트, requirements.txt 업데이트.