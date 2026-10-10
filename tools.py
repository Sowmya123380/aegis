
from pathlib import Path
from langchain_core.tools import tool
from document_processor import extract_text, extract_key_information


@tool
def read_document(file_path: str) -> str:
    """Read a PDF, DOCX, or CSV file and return its text."""
    path = Path(file_path)

    if not path.is_file():
        return "Error: File not found."

    try:
        text = extract_text(path.name, path.read_bytes())
        return text[:20000]
    except Exception as e:
        return f"Error reading document: {e}"


@tool
def extract_important_parts(file_path: str) -> str:
    """Extract key lines, email addresses, dates, and word count."""
    path = Path(file_path)

    if not path.is_file():
        return "Error: File not found."

    try:
        text = extract_text(path.name, path.read_bytes())
        info = extract_key_information(text)
        return str(info)
    except Exception as e:
        return f"Error extracting information: {e}"


if __name__ == "__main__":
    print("Aegis document tools are ready.")
    print(read_document.name)
    print(extract_important_parts.name)