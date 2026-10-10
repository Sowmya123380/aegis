"""Unit tests for Aegis Document Parser Service."""

import pandas as pd
import pytest
from services.document_parser import parse_csv, parse_document


def test_parse_csv_valid():
    csv_bytes = b"id,name,value\n1,Alpha,100\n2,Beta,200\n"
    res = parse_csv(csv_bytes, "test.csv")

    assert res["name"] == "test.csv"
    assert res["type"] == "csv"
    assert isinstance(res["dataframe"], pd.DataFrame)
    assert len(res["dataframe"]) == 2
    assert res["page_count"] == 1
    assert "Alpha" in res["text"]


def test_parse_csv_encoding_fallback():
    csv_bytes = "id,name,value\n1,Café,10.5\n".encode("latin1")
    res = parse_csv(csv_bytes, "latin.csv")

    assert res["name"] == "latin.csv"
    assert len(res["dataframe"]) == 1


def test_parse_document_invalid_extension():
    with pytest.raises(ValueError, match="Unsupported document format"):
        parse_document(b"dummy data", filename="unsupported.xyz")
