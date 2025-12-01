from pdfminer.high_level import extract_text
from pdfminer.pdfparser import PDFSyntaxError
from pdf2image import convert_from_path
from pdf2image.exceptions import PDFPageCountError, PDFSyntaxError as PDF2ImageSyntaxError
import pytesseract
from pytesseract import TesseractError


class PDFProcessingError(Exception):
    """Custom exception for PDF processing errors."""


def pdf_to_text(path):
    """
    Extract text from a PDF file using pdfminer.six.

    Args:
        path (str): Path to the PDF file.

    Returns:
        str: Extracted text from the PDF.

    Raises:
        FileNotFoundError: If the PDF file is not found.
        PDFProcessingError: For PDF parsing errors.
    """
    try:
        text = extract_text(path)
        return text
    except FileNotFoundError:
        raise FileNotFoundError(f"PDF file not found: {path}")
    except PDFSyntaxError as e:
        raise PDFProcessingError(f"Invalid PDF syntax in file {path}: {e}")
    except Exception as e:
        raise PDFProcessingError(f"Unexpected error parsing PDF {path}: {e}")


def pdf_to_text_ocr(path):
    """
    Extract text from a PDF file using OCR with pdf2image and pytesseract.
    Processes pages one by one to handle large PDFs.

    Args:
        path (str): Path to the PDF file.

    Returns:
        str: Extracted text from all pages concatenated.

    Raises:
        FileNotFoundError: If the PDF file is not found.
        PDFProcessingError: For PDF processing or OCR errors.
    """
    try:
        images = convert_from_path(path)
    except PDFPageCountError as e:
        raise PDFProcessingError(f"PDF page count error in file {path}: {e}")
    except PDF2ImageSyntaxError as e:
        raise PDFProcessingError(f"Invalid PDF syntax for OCR in file {path}: {e}")
    except FileNotFoundError:
        raise FileNotFoundError(f"PDF file not found: {path}")
    except Exception as e:
        raise PDFProcessingError(f"Unexpected error during OCR processing in file {path}: {e}")

    text = ""
    try:
        for i, image in enumerate(images, start=1):
            page_text = pytesseract.image_to_string(image)
            text += page_text + "\n"
    except TesseractError as e:
        raise PDFProcessingError(f"Tesseract OCR error on page {i} in file {path}: {e}")
    except Exception as e:
        raise PDFProcessingError(f"Unexpected error during OCR processing in file {path}: {e}")

    return text
