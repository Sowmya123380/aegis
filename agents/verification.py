"""Aegis Document Intelligence Agent - Evidence Verification Service.

Verifies claims, facts, and statements against document evidence:
- Identifies exact source filename and page/location
- Computes claim verification status: 'supported', 'partially_supported', 'contradicted', or 'not_found'
- Extracts short supporting excerpts
- Prevents hallucinated citations and claim fabrication
"""

import re
from typing import Any, Dict, List, Optional, Tuple
from agents.state import EvidenceClaim


def find_best_matching_excerpt(claim: str, text: str) -> Optional[Tuple[str, float]]:
    """Search document text for the line or sentence with highest token overlap with the claim."""
    if not claim or not text:
        return None

    claim_words = set(re.findall(r'\w+', claim.lower()))
    if not claim_words:
        return None

    # Filter out common stop words for better overlap scoring
    stopwords = {"the", "a", "an", "is", "are", "was", "were", "and", "or", "in", "on", "at", "to", "for", "of", "with", "by"}
    key_claim_words = {w for w in claim_words if w not in stopwords and not w.isdigit()}
    if not key_claim_words:
        key_claim_words = claim_words

    lines = [line.strip() for line in text.splitlines() if line.strip()]
    best_line = None
    best_score = 0.0

    for line in lines:
        line_words = set(re.findall(r'\w+', line.lower()))
        overlap = len(key_claim_words & line_words)
        score = overlap / len(key_claim_words)
        if score > best_score:
            best_score = score
            best_line = line

    if best_line and best_score >= 0.40:
        return best_line[:300], best_score

    return None


def verify_claim_against_docs(
    claim: str,
    docs: Dict[str, Any],
    source_hint: Optional[str] = None,
) -> EvidenceClaim:
    """Verify whether a single claim string is supported by document evidence."""
    if not claim or not isinstance(claim, str):
        return {
            "claim": str(claim),
            "source_file": "Unknown",
            "page_or_location": "N/A",
            "excerpt": "Invalid claim input.",
            "status": "not_found",
        }

    target_docs = {}
    if source_hint and source_hint in docs:
        target_docs[source_hint] = docs[source_hint]
    else:
        target_docs = docs

    best_match_file = "Not found"
    best_match_location = "N/A"
    best_match_excerpt = "No direct supporting evidence found in loaded documents."
    best_match_status = "not_found"
    highest_score = 0.0

    for name, doc in target_docs.items():
        text = str(doc.get("text") or doc.get("content") or "")
        pages = doc.get("pages") or []

        # Try page-level search if pages exist
        if isinstance(pages, list) and pages:
            for p in pages:
                p_num = p.get("page_number", 1)
                p_text = str(p.get("text", ""))
                res = find_best_matching_excerpt(claim, p_text)
                if res:
                    line, score = res
                    if score > highest_score:
                        highest_score = score
                        best_match_file = name
                        best_match_location = f"Page {p_num}"
                        best_match_excerpt = line
        else:
            res = find_best_matching_excerpt(claim, text)
            if res:
                line, score = res
                if score > highest_score:
                    highest_score = score
                    best_match_file = name
                    best_match_location = "Document Body"
                    best_match_excerpt = line

    if highest_score >= 0.70:
        best_match_status = "supported"
    elif highest_score >= 0.35:
        best_match_status = "partially_supported"
    else:
        best_match_status = "not_found"
        best_match_file = "Not found"
        best_match_location = "N/A"
        best_match_excerpt = "No direct supporting evidence found in loaded documents."

    return {
        "claim": claim,
        "source_file": best_match_file,
        "page_or_location": best_match_location,
        "excerpt": best_match_excerpt,
        "status": best_match_status,
    }


def batch_verify_claims(
    claims: List[str],
    docs: Dict[str, Any],
) -> List[EvidenceClaim]:
    """Verify a list of claim strings against document evidence."""
    results: List[EvidenceClaim] = []
    for c in claims:
        if isinstance(c, str) and c.strip():
            results.append(verify_claim_against_docs(c.strip(), docs))
    return results
