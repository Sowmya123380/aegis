"""Unit tests for Aegis Local Analyzer Service."""

import pandas as pd
import pytest
from services.local_analyzer import (
    analyze_csv_dataframe,
    compare_texts_local,
    extract_regex_facts,
    generate_local_report,
)


def test_extract_regex_facts():
    text = (
        "The project deadline is Oct 15, 2026. "
        "Total budget allocated is $50,000 USD. "
        "Growth increased by 15.5%. Contact contact@aegis.ai. "
        "The vendor shall deliver all software components on time."
    )
    facts = extract_regex_facts(text)

    assert len(facts["dates"]) >= 1
    assert "Oct 15, 2026" in facts["dates"]
    assert any("$50,000" in m for m in facts["monetary_amounts"])
    assert "15.5%" in facts["percentages"]
    assert "contact@aegis.ai" in facts["emails"]
    assert len(facts["obligations"]) >= 1


def test_compare_texts_local():
    text_a = "Line 1\nLine 2\nLine 3"
    text_b = "Line 1\nLine 2\nLine 4"

    res = compare_texts_local(text_a, text_b, "doc_a.txt", "doc_b.txt")
    assert res["matching_lines_count"] == 2
    assert res["only_in_a_count"] == 1
    assert res["only_in_b_count"] == 1
    assert res["similarity_percentage"] == 50.0


def test_analyze_csv_dataframe():
    df = pd.DataFrame({
        "id": [1, 2, 3, 3],
        "val": [10.0, None, 30.0, 30.0],
    })
    res = analyze_csv_dataframe(df, "data.csv")

    assert res["rows"] == 4
    assert res["columns"] == 2
    assert res["total_missing_cells"] == 1
    assert res["duplicate_rows"] == 1
    assert "val" in res["numeric_summary"]


def test_generate_local_report():
    docs = {
        "doc1.txt": {"name": "doc1.txt", "type": "txt", "text": "Sample text Content", "page_count": 1}
    }
    report = generate_local_report("Analyze documents", docs, reason="No API Key")

    assert report["execution_mode"] == "local_fallback"
    assert "Aegis AI — Document Intelligence Report" in report["answer"]
    assert "doc1.txt" in report["answer"]
