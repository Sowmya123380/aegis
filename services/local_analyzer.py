"""Aegis Document Intelligence Agent - Local Analyzer Service.

Provides deterministic, offline-capable document analysis when LLM API keys
are unavailable, rate-limited, or encounter network errors.
Includes regex-based key fact extraction, text line diff comparison,
CSV dataset profiling, and structured report compilation.
"""

import json
import re
from typing import Any, Dict, List, Optional, Tuple, Set
import pandas as pd


def extract_regex_facts(text: str) -> Dict[str, List[str]]:
    """Extract candidate key facts (dates, currency, percentages, emails, obligations) using deterministic regex."""
    if not text:
        return {"dates": [], "monetary_amounts": [], "percentages": [], "emails": [], "obligations": []}

    dates = re.findall(
        r'\b(?:\d{1,2}[-/.]\d{1,2}[-/.]\d{2,4}|(?:Jan|Feb|Mar|Apr|May|Jun|Jul|Aug|Sep|Oct|Nov|Dec)[a-z]* \d{1,2},? \d{4}|\d{4}-\d{2}-\d{2})\b',
        text,
        flags=re.IGNORECASE,
    )

    monetary = re.findall(
        r'(?:\$|€|£|INR|USD|EUR)\s?\d+(?:,\d{3})*(?:\.\d{2})?|\b\d+(?:,\d{3})*(?:\.\d{2})?\s?(?:USD|EUR|GBP|dollars)\b',
        text,
        flags=re.IGNORECASE,
    )

    percentages = re.findall(r'\b\d+(?:\.\d+)?\s?%', text)

    emails = re.findall(r'\b[A-Za-z0-9._%+-]+@[A-Za-z0-9.-]+\.[A-Z|a-z]{2,}\b', text)

    obligation_patterns = re.findall(
        r'[^.\n]*?\b(?:shall|must|is required to|agrees to|covenants|undertakes)\b[^.\n]*[.]',
        text,
        flags=re.IGNORECASE,
    )

    return {
        "dates": sorted(list(set(dates)))[:15],
        "monetary_amounts": sorted(list(set(monetary)))[:15],
        "percentages": sorted(list(set(percentages)))[:15],
        "emails": sorted(list(set(emails)))[:10],
        "obligations": [s.strip() for s in obligation_patterns[:10]],
    }


def compare_texts_local(text_a: str, text_b: str, name_a: str, name_b: str) -> Dict[str, Any]:
    """Compare two document texts deterministically."""
    lines_a = {line.strip() for line in text_a.splitlines() if line.strip()}
    lines_b = {line.strip() for line in text_b.splitlines() if line.strip()}

    common = sorted(list(lines_a & lines_b))
    only_a = sorted(list(lines_a - lines_b))
    only_b = sorted(list(lines_b - lines_a))

    total_union = len(lines_a | lines_b)
    similarity_pct = round((len(common) / total_union * 100), 1) if total_union > 0 else 0.0

    return {
        "document_a": name_a,
        "document_b": name_b,
        "similarity_percentage": similarity_pct,
        "matching_lines_count": len(common),
        "only_in_a_count": len(only_a),
        "only_in_b_count": len(only_b),
        "matching_lines_sample": common[:30],
        "only_in_a_sample": only_a[:30],
        "only_in_b_sample": only_b[:30],
    }


def analyze_csv_dataframe(df: pd.DataFrame, filename: str) -> Dict[str, Any]:
    """Calculate deterministic CSV dataset metrics, missing values, duplicates, and numeric stats."""
    if df is None or not isinstance(df, pd.DataFrame):
        return {"error": f"File '{filename}' is not a valid pandas DataFrame."}

    missing_dict = {str(k): int(v) for k, v in df.isna().sum().items()}
    numeric_df = df.select_dtypes(include="number")
    numeric_summary = numeric_df.describe().round(2).to_dict() if not numeric_df.empty else {}

    return {
        "filename": filename,
        "rows": int(df.shape[0]),
        "columns": int(df.shape[1]),
        "column_names": [str(c) for c in df.columns],
        "missing_values": missing_dict,
        "total_missing_cells": int(sum(missing_dict.values())),
        "duplicate_rows": int(df.duplicated().sum()),
        "numeric_summary": numeric_summary,
        "sample_rows": df.head(10).to_dict(orient="records"),
    }


def generate_local_report(
    task: str,
    docs: Dict[str, Any],
    reason: str,
    events: Optional[List[Dict[str, Any]]] = None,
) -> Dict[str, Any]:
    """Compile a complete deterministic Markdown intelligence report when offline or on LLM fallback."""
    doc_count = len(docs)
    lines: List[str] = [
        "# Aegis AI — Document Intelligence Report",
        "",
        "**Execution Mode:** Local Tools Fallback (Deterministic Offline Engine)",
        f"**Fallback Reason:** {reason}",
        f"**User Task:** {task.strip()}",
        "",
        "---",
        "",
        "## 1. Executive Summary",
        f"Aegis successfully inspected **{doc_count}** document(s) using offline deterministic analysis tools. "
        "All metrics, extracted facts, comparisons, and CSV statistics below were computed directly from uploaded files "
        "without external model hallucination.",
        "",
        "## 2. Document Inventory & Metadata",
    ]

    doc_readings: Dict[str, Dict[str, Any]] = {}
    csv_analyses: List[Dict[str, Any]] = []

    for name, doc in docs.items():
        doc_type = str(doc.get("type", "unknown")).upper()
        char_count = doc.get("character_count", len(str(doc.get("text", ""))))
        page_count = doc.get("page_count", 1)
        is_scanned = doc.get("is_scanned", False)

        desc = f"- **{name}** ({doc_type}): {char_count} chars across {page_count} page(s)"
        if is_scanned:
            desc += " [⚠️ Scanned PDF Notice]"

        df = doc.get("dataframe") if doc.get("dataframe") is not None else doc.get("df")
        if isinstance(df, pd.DataFrame):
            desc += f", {len(df)} rows × {len(df.columns)} columns"
            csv_analyses.append(analyze_csv_dataframe(df, name))

        lines.append(desc)

        text = str(doc.get("text") or doc.get("content") or "")
        facts = extract_regex_facts(text)
        doc_readings[name] = {
            "text": text,
            "facts": facts,
            "type": doc_type,
            "chars": char_count,
        }

    # Section 3: Extracted Facts & Key Entities
    lines.extend(["", "## 3. Key Fact Extraction"])
    for name, r in doc_readings.items():
        facts = r["facts"]
        lines.extend([
            f"### Document: `{name}`",
            f"- **Dates Found:** {', '.join(facts['dates']) if facts['dates'] else 'None detected'}",
            f"- **Monetary Values:** {', '.join(facts['monetary_amounts']) if facts['monetary_amounts'] else 'None detected'}",
            f"- **Percentages:** {', '.join(facts['percentages']) if facts['percentages'] else 'None detected'}",
            f"- **Emails:** {', '.join(facts['emails']) if facts['emails'] else 'None detected'}",
        ])
        if facts["obligations"]:
            lines.append("- **Candidate Obligations & Clauses:**")
            for ob in facts["obligations"][:5]:
                lines.append(f"  - *\"{ob}\"*")

    # Section 4: Comparisons
    text_docs = [k for k, v in docs.items() if v.get("type") != "csv"]
    if len(text_docs) >= 2:
        lines.extend(["", "## 4. Document Comparisons"])
        for i in range(len(text_docs)):
            for j in range(i + 1, min(len(text_docs), i + 3)):
                doc_a, doc_b = text_docs[i], text_docs[j]
                comp = compare_texts_local(
                    doc_readings[doc_a]["text"],
                    doc_readings[doc_b]["text"],
                    doc_a,
                    doc_b,
                )
                lines.extend([
                    f"### Comparison: `{doc_a}` vs `{doc_b}`",
                    f"- **Similarity Index:** {comp['similarity_percentage']}%",
                    f"- **Matching Shared Lines:** {comp['matching_lines_count']}",
                    f"- **Unique to {doc_a}:** {comp['only_in_a_count']}",
                    f"- **Unique to {doc_b}:** {comp['only_in_b_count']}",
                ])

    # Section 5: CSV Profiling
    if csv_analyses:
        lines.extend(["", "## 5. Dataset & CSV Profiling"])
        for csv_info in csv_analyses:
            lines.extend([
                f"### Dataset: `{csv_info['filename']}`",
                f"- **Dimensions:** {csv_info['rows']} rows × {csv_info['columns']} columns",
                f"- **Duplicate Rows:** {csv_info['duplicate_rows']}",
                f"- **Total Missing Cells:** {csv_info['total_missing_cells']}",
                f"- **Columns:** {', '.join(csv_info['column_names'])}",
            ])
            if csv_info.get("numeric_summary"):
                lines.extend([
                    "**Numeric Column Statistics:**",
                    "```json",
                    json.dumps(csv_info["numeric_summary"], indent=2),
                    "```",
                ])

    lines.extend([
        "",
        "## 6. Evidence Grounding & Limitations",
        "- All facts and statistics were produced deterministically by Python parsing, regex extraction, and pandas.",
        "- No paid LLM credits were consumed for this report run.",
        "- To enable full autonomous AI reasoning, provide a valid `GEMINI_API_KEY` or `OPENAI_API_KEY`.",
        "",
        "---",
        "*Report generated by Aegis Local Intelligence Engine.*",
    ])

    report_content = "\n".join(lines)
    return {
        "answer": report_content,
        "report": report_content,
        "execution_mode": "local_fallback",
        "events": events or [],
    }
