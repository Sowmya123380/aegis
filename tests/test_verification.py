"""Unit tests for Aegis Evidence Verification Engine."""

import pytest
from agents.verification import batch_verify_claims, verify_claim_against_docs


@pytest.fixture
def sample_docs():
    return {
        "contract.pdf": {
            "name": "contract.pdf",
            "type": "pdf",
            "text": "Header: Legal Agreement\nSection 1: Payment of $75,000 USD is due on November 30, 2026.\nSection 2: Termination requires 30 days notice.",
            "pages": [
                {"page_number": 1, "text": "Header: Legal Agreement\nSection 1: Payment of $75,000 USD is due on November 30, 2026."},
                {"page_number": 2, "text": "Section 2: Termination requires 30 days notice."},
            ]
        }
    }


def test_verify_claim_supported(sample_docs):
    claim = "Payment of $75,000 USD is due on November 30, 2026"
    res = verify_claim_against_docs(claim, sample_docs)

    assert res["status"] in {"supported", "partially_supported"}
    assert res["source_file"] == "contract.pdf"
    assert res["page_or_location"] == "Page 1"
    assert "$75,000 USD" in res["excerpt"]


def test_verify_claim_not_found(sample_docs):
    claim = "The penalty for late submission is $1,000,000"
    res = verify_claim_against_docs(claim, sample_docs)

    assert res["status"] == "not_found"
    assert res["source_file"] == "Not found"


def test_batch_verify_claims(sample_docs):
    claims = [
        "Payment of $75,000 USD is due",
        "Termination requires 30 days notice",
        "Invalid non-existent claim statement",
    ]
    results = batch_verify_claims(claims, sample_docs)

    assert len(results) == 3
    assert results[0]["status"] in {"supported", "partially_supported"}
    assert results[1]["status"] in {"supported", "partially_supported"}
    assert results[2]["status"] == "not_found"
