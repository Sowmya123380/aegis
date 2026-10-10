"""Unit tests for Aegis Registered Tool Implementations."""

import json
import pandas as pd
import pytest

from agents.tools import (
    analyze_csv_impl,
    compare_documents_impl,
    create_document_tools,
    detect_conflicts_impl,
    extract_key_facts_impl,
    find_dates_and_deadlines_impl,
    find_monetary_values_impl,
    get_document_metadata_impl,
    list_documents_impl,
    read_document_impl,
    safe_calculate,
    search_documents_impl,
    validate_source_references_impl,
)


@pytest.fixture
def sample_docs():
    df = pd.DataFrame({
        "id": [1, 2, 3, 4],
        "score": [10.5, 20.0, None, 40.5],
        "category": ["A", "B", "A", "B"],
    })

    return {
        "report_alpha.pdf": {
            "name": "report_alpha.pdf",
            "type": "pdf",
            "text": "Header: Quarterly Review\nRevenue is up 12% to $50,000.\nDeadline is Nov 30, 2026.\nConclusion: Good quarter",
            "page_count": 1,
            "pages": [{"page_number": 1, "text": "Header: Quarterly Review\nRevenue is up 12% to $50,000.\nDeadline is Nov 30, 2026.\nConclusion: Good quarter"}],
        },
        "report_beta.docx": {
            "name": "report_beta.docx",
            "type": "docx",
            "text": "Header: Quarterly Review\nRevenue is up 15% to $75,000.\nDeadline is Dec 15, 2026.\nConclusion: Good quarter",
            "page_count": 1,
        },
        "metrics.csv": {
            "name": "metrics.csv",
            "type": "csv",
            "text": "id,score,category\n1,10.5,A\n2,20.0,B",
            "df": df,
            "dataframe": df,
            "page_count": 1,
        },
    }


def test_list_documents(sample_docs):
    res = list_documents_impl(sample_docs)
    assert res["count"] == 3


def test_get_document_metadata(sample_docs):
    res = get_document_metadata_impl("report_alpha.pdf", sample_docs)
    assert res["name"] == "report_alpha.pdf"
    assert res["page_count"] == 1


def test_read_document(sample_docs):
    res = read_document_impl("report_alpha.pdf", sample_docs, page_number=1)
    assert "Quarterly Review" in res["excerpt"]


def test_search_documents(sample_docs):
    res = search_documents_impl("Revenue", sample_docs)
    assert res["matching_documents_count"] == 2


def test_extract_key_facts(sample_docs):
    res = extract_key_facts_impl(sample_docs)
    assert "facts" in res


def test_find_dates_and_deadlines(sample_docs):
    res = find_dates_and_deadlines_impl(sample_docs)
    assert "dates_and_deadlines" in res


def test_find_monetary_values(sample_docs):
    res = find_monetary_values_impl(sample_docs)
    assert "monetary_values" in res


def test_compare_documents(sample_docs):
    res = compare_documents_impl("report_alpha.pdf", "report_beta.docx", sample_docs)
    assert res["matching_lines_count"] >= 1


def test_detect_conflicts(sample_docs):
    res = detect_conflicts_impl("report_alpha.pdf", "report_beta.docx", sample_docs)
    assert res["has_potential_conflicts"] is True


def test_analyze_csv(sample_docs):
    res = analyze_csv_impl("metrics.csv", sample_docs)
    assert res["rows"] == 4


def test_safe_calculate():
    res_valid = safe_calculate("(12000 - 8500) / 12000 * 100")
    assert res_valid["result"] == pytest.approx(29.1666, rel=1e-3)


def test_validate_source_references(sample_docs):
    valid_res = validate_source_references_impl("report_alpha.pdf", sample_docs, page_number=1)
    assert valid_res["valid"] is True

    invalid_res = validate_source_references_impl("ghost.pdf", sample_docs)
    assert invalid_res["valid"] is False


def test_create_document_tools(sample_docs):
    tools = create_document_tools(sample_docs)
    assert len(tools) == 17
