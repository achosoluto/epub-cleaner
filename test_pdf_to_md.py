import pytest
from unittest import mock
from epub_cleaner.pdf_to_md import pdf_to_text, pdf_to_text_ocr, PDFProcessingError


def test_pdf_to_text_success():
    """Test successful text extraction from PDF."""
    with mock.patch("epub_cleaner.pdf_to_md.extract_text") as mock_extract:
        mock_extract.return_value = "Sample extracted text"
        result = pdf_to_text("test.pdf")
        assert result == "Sample extracted text"
        mock_extract.assert_called_once_with("test.pdf")


def test_pdf_to_text_file_not_found():
    """Test FileNotFoundError when PDF file does not exist."""
    with mock.patch("epub_cleaner.pdf_to_md.extract_text") as mock_extract:
        mock_extract.side_effect = FileNotFoundError()
        with pytest.raises(FileNotFoundError, match="PDF file not found: test.pdf"):
            pdf_to_text("test.pdf")


def test_pdf_to_text_parsing_error():
    """Test generic parsing error during text extraction."""
    with mock.patch("epub_cleaner.pdf_to_md.extract_text") as mock_extract:
        mock_extract.side_effect = Exception("Parsing error")
        with pytest.raises(PDFProcessingError, match=r"Unexpected error parsing PDF test\.pdf: Parsing error"):
            pdf_to_text("test.pdf")


def test_pdf_to_text_empty_pdf():
    """Test text extraction from an empty PDF (no pages)."""
    with mock.patch("epub_cleaner.pdf_to_md.extract_text") as mock_extract:
        mock_extract.return_value = ""
        result = pdf_to_text("test.pdf")
        assert result == ""
        mock_extract.assert_called_once_with("test.pdf")


def test_pdf_to_text_no_extractable_text():
    """Test text extraction from PDF with no extractable text (e.g., image-only)."""
    with mock.patch("epub_cleaner.pdf_to_md.extract_text") as mock_extract:
        mock_extract.return_value = "   \n\t   "  # whitespace only
        result = pdf_to_text("test.pdf")
        assert result == "   \n\t   "
        mock_extract.assert_called_once_with("test.pdf")


def test_pdf_to_text_ocr_success():
    """Test successful OCR text extraction from PDF with multiple pages."""
    mock_image1 = mock.Mock()
    mock_image2 = mock.Mock()
    with mock.patch(
        "epub_cleaner.pdf_to_md.convert_from_path"
    ) as mock_convert, mock.patch(
        "epub_cleaner.pdf_to_md.pytesseract.image_to_string"
    ) as mock_ocr:
        mock_convert.return_value = [mock_image1, mock_image2]
        mock_ocr.side_effect = ["Page 1 text", "Page 2 text"]
        result = pdf_to_text_ocr("test.pdf")
        assert result == "Page 1 text\nPage 2 text\n"
        mock_convert.assert_called_once_with("test.pdf")
        mock_ocr.assert_has_calls([mock.call(mock_image1), mock.call(mock_image2)])


def test_pdf_to_text_ocr_file_not_found():
    """Test FileNotFoundError when PDF file does not exist for OCR."""
    with mock.patch("epub_cleaner.pdf_to_md.convert_from_path") as mock_convert:
        mock_convert.side_effect = FileNotFoundError()
        with pytest.raises(FileNotFoundError, match="PDF file not found: test.pdf"):
            pdf_to_text_ocr("test.pdf")


def test_pdf_to_text_ocr_processing_error():
    """Test processing error during PDF to image conversion."""
    with mock.patch("epub_cleaner.pdf_to_md.convert_from_path") as mock_convert:
        mock_convert.side_effect = Exception("Conversion failed")
        with pytest.raises(
            PDFProcessingError, match=r"Unexpected error during OCR processing in file test\.pdf: Conversion failed"
        ):
            pdf_to_text_ocr("test.pdf")


def test_pdf_to_text_ocr_ocr_error():
    """Test OCR error during text extraction from images."""
    mock_image = mock.Mock()
    with mock.patch(
        "epub_cleaner.pdf_to_md.convert_from_path"
    ) as mock_convert, mock.patch(
        "epub_cleaner.pdf_to_md.pytesseract.image_to_string"
    ) as mock_ocr:
        mock_convert.return_value = [mock_image]
        mock_ocr.side_effect = Exception("OCR failed")
        with pytest.raises(
            PDFProcessingError, match=r"Unexpected error during OCR processing in file test\.pdf: OCR failed"
        ):
            pdf_to_text_ocr("test.pdf")


def test_pdf_to_text_ocr_empty_pdf():
    """Test OCR extraction from an empty PDF (no pages)."""
    with mock.patch(
        "epub_cleaner.pdf_to_md.convert_from_path"
    ) as mock_convert, mock.patch(
        "epub_cleaner.pdf_to_md.pytesseract.image_to_string"
    ) as mock_ocr:
        mock_convert.return_value = []
        result = pdf_to_text_ocr("test.pdf")
        assert result == ""
        mock_convert.assert_called_once_with("test.pdf")
        mock_ocr.assert_not_called()
