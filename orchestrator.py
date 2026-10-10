
from document_processor import extract_text
from agents.extraction_agent import run_extraction_agent
from agents.summary_agent import run_summary_agent
from agents.comparison_agent import run_comparison_agent


def process_document(filename, file_bytes):
    text = extract_text(filename, file_bytes)

    return {
        "filename": filename,
        "text": text,
        "extracted_info": run_extraction_agent(text),
        "summary": run_summary_agent(text),
    }


def compare_two_documents(text1, text2):
    return run_comparison_agent(text1, text2)