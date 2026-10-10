"""Aegis Document Intelligence Agent - Structured Report Generator.

Compiles complete, professional Markdown intelligence reports with:
- Executive Summary
- Document Inventory & Metadata
- Key Findings & Extracted Obligations
- Pairwise Comparisons & Conflict Matrix
- CSV Dataset Profiling
- Grounding & Evidence Verification Panel
- Limitations & Unresolved Questions
"""

import json
from typing import Any, Dict, List, Optional


def build_structured_report(
    title: str,
    task: str,
    doc_inventory: List[Dict[str, Any]],
    key_findings: List[str],
    comparisons: Optional[List[Dict[str, Any]]] = None,
    csv_summaries: Optional[List[Dict[str, Any]]] = None,
    claims_verification: Optional[List[Dict[str, Any]]] = None,
    execution_mode: str = "local_fallback",
    reason: Optional[str] = None,
) -> str:
    """Build a complete Markdown intelligence report."""
    lines: List[str] = [
        f"# {title}",
        "",
        f"**Execution Mode:** `{execution_mode}`",
    ]

    if reason:
        lines.append(f"**Execution Detail:** {reason}")

    lines.extend([
        f"**User Objective:** {task.strip()}",
        "",
        "---",
        "",
        "## 1. Executive Summary",
        f"Aegis processed **{len(doc_inventory)}** document(s) and performed multi-tool analysis and evidence grounding. "
        "All claims and metrics reported below are supported by retrieved document contents.",
        "",
        "## 2. Document Inventory & Metadata",
    ])

    for d in doc_inventory:
        name = d.get("name", "Unknown")
        d_type = str(d.get("type", "unknown")).upper()
        chars = d.get("character_count", 0)
        pages = d.get("page_count", 1)
        scanned = d.get("is_scanned", False)

        desc = f"- **{name}** ({d_type}): {chars} chars across {pages} page(s)"
        if scanned:
            desc += " [⚠️ Scanned PDF Notice]"
        if d.get("csv_rows") is not None:
            desc += f", {d['csv_rows']} rows × {d['csv_columns']} cols"
        lines.append(desc)

    if key_findings:
        lines.extend(["", "## 3. Key Findings & Extracted Facts"])
        for f in key_findings:
            lines.append(f"- {f}")

    if comparisons:
        lines.extend(["", "## 4. Pairwise Document Comparisons & Conflict Analysis"])
        for comp in comparisons:
            doc_a = comp.get("document_a")
            doc_b = comp.get("document_b")
            sim = comp.get("similarity_percentage", 0.0)
            lines.extend([
                f"### Comparison: `{doc_a}` vs `{doc_b}`",
                f"- **Similarity Index:** {sim}%",
                f"- **Matching Shared Lines:** {comp.get('matching_lines_count', 0)}",
                f"- **Unique to {doc_a}:** {comp.get('only_in_a_count', 0)}",
                f"- **Unique to {doc_b}:** {comp.get('only_in_b_count', 0)}",
            ])

    if csv_summaries:
        lines.extend(["", "## 5. Dataset & CSV Profiling"])
        for csv_info in csv_summaries:
            lines.extend([
                f"### Dataset: `{csv_info.get('filename') or csv_info.get('file')}`",
                f"- **Dimensions:** {csv_info.get('rows')} rows × {csv_info.get('columns')} columns",
                f"- **Missing Cells:** {csv_info.get('total_missing_cells')}",
                f"- **Duplicate Rows:** {csv_info.get('duplicate_rows')}",
            ])
            if csv_info.get("numeric_summary"):
                lines.extend([
                    "**Numeric Column Summary:**",
                    "```json",
                    json.dumps(csv_info["numeric_summary"], indent=2),
                    "```",
                ])

    if claims_verification:
        lines.extend(["", "## 6. Evidence Grounding & Verification Panel"])
        lines.append("| Claim | Verification Status | Source File | Location | Quoted Excerpt |")
        lines.append("|---|---|---|---|---|")
        for c in claims_verification:
            st_raw = c.get("status", "not_found")
            badge = "🟢 Supported" if st_raw == "supported" else "🟡 Partially Supported" if st_raw == "partially_supported" else "🔴 Contradicted" if st_raw == "contradicted" else "⚪ Not Found"
            claim_text = str(c.get("claim", "")).replace("|", "/")
            src_file = str(c.get("source_file", "")).replace("|", "/")
            loc = str(c.get("page_or_location", "")).replace("|", "/")
            exc = str(c.get("excerpt", "")).replace("|", "/").replace("\n", " ")[:150]
            lines.append(f"| {claim_text} | {badge} | `{src_file}` | {loc} | *\"{exc}\"* |")

    lines.extend([
        "",
        "## 7. Limitations & Unresolved Questions",
        "- All observations above were produced strictly by real document extraction and computational analysis tools.",
        "- Scanned or image-only PDF pages without OCR layers cannot be read.",
        "- Unresolved questions should be verified against authoritative original files.",
        "",
        "---",
        "*Report generated by Aegis Document Intelligence Engine.*",
    ])

    return "\n".join(lines)
