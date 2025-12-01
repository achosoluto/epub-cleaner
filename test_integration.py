import tempfile
import subprocess
import os
import csv
import base64


def test_integration_mixed_md_pdf():
    """
    Integration test for epub-cleaner with mixed MD and PDF files.
    Creates temporary directories with sample files, runs the cleaner,
    and verifies outputs and report.
    """
    with tempfile.TemporaryDirectory() as temp_dir:
        input_dir = os.path.join(temp_dir, "input")
        pdf_dir = os.path.join(temp_dir, "pdf")
        output_dir = os.path.join(temp_dir, "output")
        report_path = os.path.join(temp_dir, "report.csv")

        os.makedirs(input_dir)
        os.makedirs(pdf_dir)
        os.makedirs(output_dir)

        # Create sample MD file
        md_content = "# Hello\n\nThis is a test markdown file with some **bold** text."
        with open(os.path.join(input_dir, "sample.md"), "w", encoding="utf-8") as f:
            f.write(md_content)

        # Create sample PDF file (minimal PDF with "Hello World" using base64-encoded content)
        pdf_b64 = "JVBERi0xLjQKMSAwIG9iago8PAovVHlwZSAvQ2F0YWxvZwovUGFnZXMgMiAwIFIKPj4KZW5kb2JqCjIgMCBvYmoKPDwKL1R5cGUgL1BhZ2VzCi9LaWRzIFszIDAgUl0KL0NvdW50IDEKPj4KZW5kb2JqCjMgMCBvYmoKPDwKL1R5cGUgL1BhZ2UKL1BhcmVudCAyIDAgUgovTWVkaWFCb3ggWzAgMCA2MTIgNzkyXQovQ29udGVudHMgNCAwIFIKL1Jlc291cmNlcyA8PAovRm9udCA8PAovRjEgNSAwIFIKPj4KPj4KPj4KZW5kb2JqCjQgMCBvYmoKPDwKL0xlbmd0aCA0NAo+PgpzdHJlYW0KQlQKL0YxIDEyIFRmCjcyIDcyMCBUZAooSGVsbG8gV29ybGQpIFRqCkVUCmVuZHN0cmVhbQplbmRvYmoKNSAwIG9iago8PAovVHlwZSAvRm9udAovU3VidHlwZSAvVHlwZTEKL0Jhc2VGb250IC9IZWx2ZXRpY2EKPj4KZW5kb2JqCnhyZWYKMCA2CjAwMDAwMDAwMDAgNjU1MzUgZgowMDAwMDAwMDA5IDAwMDAwIG4KMDAwMDAwMDA1OCAwMDAwMCBuCjAwMDAwMDAxMTUgMDAwMDAgbgowMDAwMDAwMjc0IDAwMDAwIG4KMDAwMDAwMDM1NCAwMDAwMCBuCnRyYWlsZXIKPDwKL1NpemUgNgovUm9vdCAxIDAgUgo+PgpzdGFydHhyZWYKNDU5CiUlRU9G"
        with open(os.path.join(pdf_dir, "sample.pdf"), "wb") as f:
            f.write(base64.b64decode(pdf_b64))

        # Run the epub-cleaner command
        cmd = [
            "./venv/bin/python3",
            "dhh_batch_clean.py",
            "--input-dir",
            input_dir,
            "--pdf-dir",
            pdf_dir,
            "--output-dir",
            output_dir,
            "--no-dry-run",
            "--report",
            report_path,
            "--limit",
            "0",
        ]
        result = subprocess.run(cmd, cwd=".", capture_output=True, text=True)

        # Assert command succeeded
        assert result.returncode == 0, f"Command failed with stderr: {result.stderr}"

        # Assert cleaned files exist in output directory
        assert os.path.exists(
            os.path.join(output_dir, "sample_clean.md")
        ), "MD output file not found"
        # PDF converted to MD and cleaned
        assert os.path.exists(
            os.path.join(output_dir, "sample_pdf_clean.md")
        ), "PDF-converted output file not found"

        # Assert CSV report shows 0 failures
        with open(report_path, "r", encoding="utf-8") as f:
            reader = csv.DictReader(f)
            rows = list(reader)
            assert len(rows) == 2, f"Expected 2 rows in report, got {len(rows)}"
            for row in rows:
                assert (
                    row["status"] == "success"
                ), f"File {row['file_path']} failed: {row['reason']}"
                assert (
                    row["idempotent"] == "true"
                ), f"Idempotence violation for {row['file_path']}"

        # Validate content of cleaned MD file
        with open(os.path.join(output_dir, "sample_clean.md"), "r", encoding="utf-8") as f:
            md_content = f.read()
        assert "Hello" in md_content, "Expected 'Hello' in cleaned MD content"
        assert "bold" in md_content, "Expected 'bold' in cleaned MD content"

        # Validate content of cleaned PDF-derived MD file
        with open(os.path.join(output_dir, "sample_pdf_clean.md"), "r", encoding="utf-8") as f:
            pdf_md_content = f.read()
        assert "Hello World" in pdf_md_content, "Expected 'Hello World' in cleaned PDF-derived MD content"

        # Temporary files are cleaned up by the tool itself (temp_dir removed by context manager)
