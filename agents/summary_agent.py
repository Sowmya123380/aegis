
from document_processor import summarize_locally

def run_summary_agent(text):
    """Agent 2: Generate a concise document summary."""
    return summarize_locally(text, max_sentences=6)