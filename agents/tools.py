"""Aegis Document Intelligence Agent - Complete 17-Tool Toolkit.

Provides 17 registered, validated tools compatible with LangChain / LangGraph:
1. list_documents - Inventories files, types, page counts, dimensions, scanned status
2. get_document_metadata - Retrieves metadata dict for a file
3. read_document - Retrieves section or bounded page excerpt
4. search_documents - Searches keywords/phrases with page numbers and line snippets
5. extract_tables - Extracts tables from DOCX or CSV datasets
6. extract_key_facts - Extracts candidate dates, amounts, emails, obligations
7. find_dates_and_deadlines - Extracts candidate dates and deadline clauses
8. find_monetary_values - Extracts currency amounts and prices
9. compare_documents - Compares two documents, computes similarity index %, line diffs
10. compare_document_sections - Compares specific page numbers or sections
11. detect_conflicts - Detects conflicting dates, monetary amounts, or clauses
12. analyze_csv - Computes dataset quality, missing values, duplicates, numeric statistics
13. calculate - Performs safe AST-based numeric calculations (eval-free)
14. retrieve_evidence - Retrieves top-k relevant text chunks matching a query
15. verify_claims - Checks claim support against retrieved document evidence
16. validate_source_references - Validates whether source file and page numbers exist
17. generate_report - Compiles structured Markdown intelligence report
"""

import ast
import json
import operator
import re
from typing import Any, Dict, List, Optional, Union
import pandas as pd

from langchain_core.tools import StructuredTool
from pydantic import BaseModel, Field

from agents.verification import verify_claim_against_docs, batch_verify_claims
from services.local_analyzer import extract_regex_facts, compare_texts_local, analyze_csv_dataframe
from services.retrieval import retrieve_relevant_chunks
from services.report_generator import build_structured_report

MAX_TOOL_OUTPUT_CHARS = 3500


# =========================================================
# 1. PYDANTIC SCHEMAS FOR INPUT VALIDATION
# =========================================================

class ListDocumentsInput(BaseModel):
    """Input schema for listing loaded documents."""
    pass


class GetDocumentMetadataInput(BaseModel):
    """Input schema for retrieving document metadata."""
    name: str = Field(description="Exact filename of the document to inspect.")


class ReadDocumentInput(BaseModel):
    """Input schema for reading a document excerpt or page."""
    name: str = Field(description="Exact filename of the document to inspect.")
    page_number: Optional[int] = Field(default=None, description="Optional 1-based page number to inspect.")
    start_char: Optional[int] = Field(default=0, description="Starting character offset.")
    length: Optional[int] = Field(default=2000, description="Maximum number of characters to read.")


class SearchDocumentsInput(BaseModel):
    """Input schema for searching text across documents."""
    query: str = Field(description="Keyword, phrase, or pattern to search for.")
    name: Optional[str] = Field(default=None, description="Optional exact filename to search within.")


class ExtractTablesInput(BaseModel):
    """Input schema for extracting tables."""
    name: str = Field(description="Exact filename of the document containing tables or CSV.")


class ExtractKeyFactsInput(BaseModel):
    """Input schema for extracting key facts."""
    name: Optional[str] = Field(default=None, description="Optional filename. If omitted, extracts from all files.")


class FindDatesAndDeadlinesInput(BaseModel):
    """Input schema for finding dates and deadlines."""
    name: Optional[str] = Field(default=None, description="Optional filename.")


class FindMonetaryValuesInput(BaseModel):
    """Input schema for finding monetary amounts."""
    name: Optional[str] = Field(default=None, description="Optional filename.")


class CompareDocumentsInput(BaseModel):
    """Input schema for comparing two documents."""
    document_a: str = Field(description="Exact filename of the first document.")
    document_b: str = Field(description="Exact filename of the second document.")


class CompareDocumentSectionsInput(BaseModel):
    """Input schema for comparing specific sections or pages."""
    document_a: str = Field(description="Filename of first document.")
    page_a: int = Field(default=1, description="Page number of first document.")
    document_b: str = Field(description="Filename of second document.")
    page_b: int = Field(default=1, description="Page number of second document.")


class DetectConflictsInput(BaseModel):
    """Input schema for detecting conflicts between documents."""
    document_a: str = Field(description="Filename of first document.")
    document_b: str = Field(description="Filename of second document.")


class AnalyzeCsvInput(BaseModel):
    """Input schema for analyzing a CSV dataset."""
    name: str = Field(description="Exact filename of the CSV file to analyze.")


class CalculateInput(BaseModel):
    """Input schema for safe numeric math calculation."""
    expression: str = Field(description="Mathematical expression to evaluate, e.g. '(12000 - 8500) / 12000 * 100'.")


class RetrieveEvidenceInput(BaseModel):
    """Input schema for evidence chunk retrieval."""
    query: str = Field(description="Search query to retrieve top evidence chunks for.")
    top_k: Optional[int] = Field(default=5, description="Number of top relevant chunks to return.")


class VerifyClaimsInput(BaseModel):
    """Input schema for verifying claims against evidence."""
    claims: List[str] = Field(description="List of claim strings to verify against document evidence.")


class ValidateSourceReferencesInput(BaseModel):
    """Input schema for validating source references."""
    source_file: str = Field(description="Filename referenced.")
    page_number: Optional[int] = Field(default=None, description="Page number referenced.")


class GenerateReportInput(BaseModel):
    """Input schema for generating a Markdown report."""
    title: Optional[str] = Field(default="Aegis Document Intelligence Report", description="Report title.")
    summary: Optional[str] = Field(default="", description="Executive summary overview.")
    findings: Optional[List[str]] = Field(default=None, description="List of key findings.")


# =========================================================
# 2. SAFE AST CALCULATOR EVALUATOR
# =========================================================

SAFE_OPERATORS = {
    ast.Add: operator.add,
    ast.Sub: operator.sub,
    ast.Mult: operator.mul,
    ast.Div: operator.truediv,
    ast.FloorDiv: operator.floordiv,
    ast.Mod: operator.mod,
    ast.Pow: operator.pow,
    ast.USub: operator.neg,
    ast.UAdd: operator.pos,
}


def _safe_eval_node(node: ast.AST) -> Union[int, float]:
    """Recursively evaluate an AST expression using only safe numeric operators."""
    if isinstance(node, ast.Expression):
        return _safe_eval_node(node.body)

    elif isinstance(node, ast.Constant):
        if isinstance(node.value, (int, float)):
            return node.value
        raise ValueError(f"Unsupported constant type: {type(node.value)}")

    elif hasattr(ast, "Num") and isinstance(node, getattr(ast, "Num")):
        return node.n

    elif isinstance(node, ast.UnaryOp):
        op_type = type(node.op)
        if op_type in SAFE_OPERATORS:
            operand = _safe_eval_node(node.operand)
            return SAFE_OPERATORS[op_type](operand)
        raise ValueError(f"Unsupported unary operator: {op_type}")

    elif isinstance(node, ast.BinOp):
        op_type = type(node.op)
        if op_type in SAFE_OPERATORS:
            left = _safe_eval_node(node.left)
            right = _safe_eval_node(node.right)
            return SAFE_OPERATORS[op_type](left, right)
        raise ValueError(f"Unsupported binary operator: {op_type}")

    else:
        raise ValueError(f"Unsafe or unsupported AST node: {type(node).__name__}")


def safe_calculate(expression: str) -> Dict[str, Any]:
    """Safely calculate numeric expressions without eval()."""
    if not expression or not isinstance(expression, str):
        return {"error": "Expression must be a non-empty string."}

    cleaned = expression.strip()
    try:
        parsed = ast.parse(cleaned, mode="eval")
        val = _safe_eval_node(parsed)
        return {
            "expression": cleaned,
            "result": val,
            "formatted": f"{val:,.4f}".rstrip("0").rstrip("."),
        }
    except Exception as exc:
        return {"error": f"Invalid math expression '{cleaned}': {type(exc).__name__} - {exc}"}


# =========================================================
# 3. HELPER FUNCTIONS & TOOL IMPLEMENTATIONS
# =========================================================

def find_matching_doc_key(name: str, docs: Dict[str, Any]) -> Optional[str]:
    """Case-insensitive and whitespace-tolerant document key lookup."""
    if not name or not docs:
        return None
    cleaned = name.strip()
    if cleaned in docs:
        return cleaned
    cleaned_lower = cleaned.lower()
    for k in docs:
        if k.lower() == cleaned_lower:
            return k
    return None


def list_documents_impl(docs: Dict[str, Any]) -> Dict[str, Any]:
    """List loaded files with format, page count, character count, CSV shape, and scanned flag."""
    if not docs:
        return {"count": 0, "documents": [], "message": "No documents are currently loaded."}

    items: List[Dict[str, Any]] = []
    for name, doc in docs.items():
        df = doc.get("dataframe") if doc.get("dataframe") is not None else doc.get("df")
        text = str(doc.get("text") or doc.get("content") or "")
        doc_type = str(doc.get("type", "unknown")).upper()
        page_count = doc.get("page_count", 1)
        is_scanned = doc.get("is_scanned", False)

        csv_rows = len(df) if isinstance(df, pd.DataFrame) else None
        csv_cols = len(df.columns) if isinstance(df, pd.DataFrame) else None

        items.append({
            "name": name,
            "type": doc_type,
            "page_count": page_count,
            "character_count": len(text),
            "is_scanned": is_scanned,
            "csv_rows": csv_rows,
            "csv_columns": csv_cols,
        })

    return {"count": len(items), "documents": items}


def get_document_metadata_impl(name: str, docs: Dict[str, Any]) -> Dict[str, Any]:
    """Retrieve metadata dictionary for a file."""
    matched_name = find_matching_doc_key(name, docs)
    if not matched_name:
        return {"error": f"Document '{name}' not found.", "available_documents": list(docs.keys())}
    doc = docs[matched_name]
    return {
        "name": matched_name,
        "metadata": doc.get("metadata", {}),
        "type": doc.get("type"),
        "page_count": doc.get("page_count", 1),
        "character_count": doc.get("character_count", len(str(doc.get("text", "")))),
        "is_scanned": doc.get("is_scanned", False),
    }


def read_document_impl(
    name: str,
    docs: Dict[str, Any],
    page_number: Optional[int] = None,
    start_char: int = 0,
    length: int = 2000,
) -> Dict[str, Any]:
    """Retrieve section or bounded page excerpt from a document."""
    matched_name = find_matching_doc_key(name, docs)
    if not matched_name:
        return {"error": f"Document '{name}' not found.", "available_documents": list(docs.keys())}

    doc = docs[matched_name]
    pages = doc.get("pages") or []

    if page_number is not None and isinstance(pages, list) and pages:
        p_found = next((p for p in pages if p.get("page_number") == page_number), None)
        if not p_found:
            return {
                "error": f"Page {page_number} not found in '{matched_name}'. Document has {len(pages)} page(s).",
                "available_pages": [p.get("page_number") for p in pages],
            }
        p_text = str(p_found.get("text", ""))
        return {
            "name": matched_name,
            "page_number": page_number,
            "total_page_characters": len(p_text),
            "excerpt": p_text[:length],
            "truncated": len(p_text) > length,
        }

    full_text = str(doc.get("text") or doc.get("content") or "")
    sliced_text = full_text[start_char : start_char + length]
    truncated = (start_char + length) < len(full_text)

    return {
        "name": matched_name,
        "start_char": start_char,
        "length_read": len(sliced_text),
        "total_characters": len(full_text),
        "excerpt": sliced_text,
        "truncated": truncated,
    }


def search_documents_impl(
    query: str,
    docs: Dict[str, Any],
    name: Optional[str] = None,
) -> Dict[str, Any]:
    """Search query keywords across documents with page and line location snippets."""
    if not query or not isinstance(query, str):
        return {"error": "Search query must be a non-empty string."}

    q_lower = query.strip().lower()
    matches: List[Dict[str, Any]] = []

    target_docs = {}
    if name:
        matched_name = find_matching_doc_key(name, docs)
        if matched_name:
            target_docs[matched_name] = docs[matched_name]
        else:
            return {"error": f"Document '{name}' not found.", "available_documents": list(docs.keys())}
    else:
        target_docs = docs

    for doc_name, doc in target_docs.items():
        pages = doc.get("pages") or []
        doc_matches = []

        if isinstance(pages, list) and pages:
            for p in pages:
                p_num = p.get("page_number", 1)
                p_text = str(p.get("text", ""))
                for line_idx, line in enumerate(p_text.splitlines(), start=1):
                    if q_lower in line.lower():
                        doc_matches.append({
                            "page": p_num,
                            "line": line_idx,
                            "snippet": line.strip()[:250],
                        })
        else:
            text = str(doc.get("text") or doc.get("content") or "")
            for line_idx, line in enumerate(text.splitlines(), start=1):
                if q_lower in line.lower():
                    doc_matches.append({
                        "page": 1,
                        "line": line_idx,
                        "snippet": line.strip()[:250],
                    })

        if doc_matches:
            matches.append({
                "document": doc_name,
                "match_count": len(doc_matches),
                "snippets": doc_matches[:20],
            })

    return {
        "query": query,
        "matching_documents_count": len(matches),
        "results": matches,
    }


def extract_tables_impl(name: str, docs: Dict[str, Any]) -> Dict[str, Any]:
    """Extract table content from DOCX or CSV files."""
    matched_name = find_matching_doc_key(name, docs)
    if not matched_name:
        return {"error": f"Document '{name}' not found."}

    doc = docs[matched_name]
    df = doc.get("dataframe") if doc.get("dataframe") is not None else doc.get("df")

    if isinstance(df, pd.DataFrame):
        return {
            "name": matched_name,
            "type": "CSV Table",
            "rows": len(df),
            "columns": list(df.columns),
            "preview_rows": df.head(15).to_dict(orient="records"),
        }

    text = str(doc.get("text") or doc.get("content") or "")
    table_lines = [line.strip() for line in text.splitlines() if "|" in line or "--- Table" in line]
    return {
        "name": matched_name,
        "type": "Extracted Text Tables",
        "table_line_count": len(table_lines),
        "table_lines_sample": table_lines[:30],
    }


def extract_key_facts_impl(docs: Dict[str, Any], name: Optional[str] = None) -> Dict[str, Any]:
    """Extract candidate key facts (dates, amounts, obligations, emails) from documents."""
    target_docs = {}
    if name:
        matched_name = find_matching_doc_key(name, docs)
        if matched_name:
            target_docs[matched_name] = docs[matched_name]
        else:
            return {"error": f"Document '{name}' not found."}
    else:
        target_docs = docs

    extracted: Dict[str, Any] = {}
    for doc_name, doc in target_docs.items():
        text = str(doc.get("text") or doc.get("content") or "")
        facts = extract_regex_facts(text)
        extracted[doc_name] = facts

    return {"facts": extracted}


def find_dates_and_deadlines_impl(docs: Dict[str, Any], name: Optional[str] = None) -> Dict[str, Any]:
    """Extract dates and candidate deadline clauses."""
    facts = extract_key_facts_impl(docs, name=name).get("facts", {})
    deadlines = {}
    for doc_name, f in facts.items():
        doc = docs.get(doc_name, {})
        text = str(doc.get("text", ""))
        deadline_clauses = re.findall(
            r'[^.\n]*?\b(?:due|deadline|before|by|expire|expiration|cutoff)\b[^.\n]*[.]',
            text,
            flags=re.IGNORECASE,
        )
        deadlines[doc_name] = {
            "dates": f.get("dates", []),
            "deadline_clauses": [c.strip() for c in deadline_clauses[:10]],
        }
    return {"dates_and_deadlines": deadlines}


def find_monetary_values_impl(docs: Dict[str, Any], name: Optional[str] = None) -> Dict[str, Any]:
    """Extract currency amounts and monetary values."""
    facts = extract_key_facts_impl(docs, name=name).get("facts", {})
    monetary = {doc_name: f.get("monetary_amounts", []) for doc_name, f in facts.items()}
    return {"monetary_values": monetary}


def compare_documents_impl(document_a: str, document_b: str, docs: Dict[str, Any]) -> Dict[str, Any]:
    """Compare two documents deterministically."""
    match_a = find_matching_doc_key(document_a, docs)
    match_b = find_matching_doc_key(document_b, docs)

    if not match_a or not match_b:
        missing = []
        if not match_a:
            missing.append(f"document_a '{document_a}'")
        if not match_b:
            missing.append(f"document_b '{document_b}'")
        return {"error": f"Comparison failed: Missing {', '.join(missing)}.", "available_documents": list(docs.keys())}

    if match_a == match_b:
        return {"error": f"Cannot compare document '{match_a}' with itself. Choose two distinct documents."}

    text_a = str(docs[match_a].get("text") or docs[match_a].get("content") or "")
    text_b = str(docs[match_b].get("text") or docs[match_b].get("content") or "")

    return compare_texts_local(text_a, text_b, match_a, match_b)


def compare_document_sections_impl(document_a: str, page_a: int, document_b: str, page_b: int, docs: Dict[str, Any]) -> Dict[str, Any]:
    """Compare specific pages or sections between two documents."""
    read_a = read_document_impl(document_a, docs, page_number=page_a)
    read_b = read_document_impl(document_b, docs, page_number=page_b)

    if "error" in read_a:
        return read_a
    if "error" in read_b:
        return read_b

    text_a = read_a.get("excerpt", "")
    text_b = read_b.get("excerpt", "")

    return compare_texts_local(text_a, text_b, f"{document_a} (Page {page_a})", f"{document_b} (Page {page_b})")


def detect_conflicts_impl(document_a: str, document_b: str, docs: Dict[str, Any]) -> Dict[str, Any]:
    """Detect conflicting dates, monetary amounts, or terms between two documents."""
    facts_a = extract_key_facts_impl(docs, name=document_a).get("facts", {}).get(document_a, {})
    facts_b = extract_key_facts_impl(docs, name=document_b).get("facts", {}).get(document_b, {})

    dates_a = set(facts_a.get("dates", []))
    dates_b = set(facts_b.get("dates", []))

    money_a = set(facts_a.get("monetary_amounts", []))
    money_b = set(facts_b.get("monetary_amounts", []))

    conflicting_dates = list(dates_a ^ dates_b)
    conflicting_money = list(money_a ^ money_b)

    return {
        "document_a": document_a,
        "document_b": document_b,
        "conflicting_dates_sample": conflicting_dates[:10],
        "conflicting_monetary_sample": conflicting_money[:10],
        "has_potential_conflicts": bool(conflicting_dates or conflicting_money),
    }


def analyze_csv_impl(name: str, docs: Dict[str, Any]) -> Dict[str, Any]:
    """Analyze a CSV dataset."""
    matched_name = find_matching_doc_key(name, docs)
    if not matched_name:
        return {"error": f"Document '{name}' not found.", "available_documents": list(docs.keys())}

    doc = docs[matched_name]
    df = doc.get("dataframe") if doc.get("dataframe") is not None else doc.get("df")

    if df is None or not isinstance(df, pd.DataFrame):
        return {"error": f"Document '{matched_name}' is not a CSV dataset."}

    return analyze_csv_dataframe(df, matched_name)


def validate_source_references_impl(source_file: str, docs: Dict[str, Any], page_number: Optional[int] = None) -> Dict[str, Any]:
    """Validate whether source file and page numbers exist in workspace."""
    matched_name = find_matching_doc_key(source_file, docs)
    if not matched_name:
        return {"valid": False, "reason": f"Source file '{source_file}' does not exist in workspace."}

    doc = docs[matched_name]
    if page_number is not None:
        p_count = doc.get("page_count", 1)
        if page_number < 1 or page_number > p_count:
            return {"valid": False, "reason": f"Page {page_number} is out of bounds for '{matched_name}' (total pages: {p_count})."}

    return {"valid": True, "source_file": matched_name, "page_number": page_number}


# =========================================================
# 4. LANGCHAIN STRUCTURED TOOL BINDING FACTORY
# =========================================================

def create_document_tools(docs: Dict[str, Any]) -> List[StructuredTool]:
    """Create 17 LangChain StructuredTool instances bound to current document collection."""

    def _list_documents() -> str:
        return json.dumps(list_documents_impl(docs), ensure_ascii=False, default=str)[:MAX_TOOL_OUTPUT_CHARS]

    def _get_document_metadata(name: str) -> str:
        return json.dumps(get_document_metadata_impl(name, docs), ensure_ascii=False, default=str)[:MAX_TOOL_OUTPUT_CHARS]

    def _read_document(name: str, page_number: Optional[int] = None, start_char: int = 0, length: int = 2000) -> str:
        return json.dumps(read_document_impl(name, docs, page_number=page_number, start_char=start_char, length=length), ensure_ascii=False, default=str)[:MAX_TOOL_OUTPUT_CHARS]

    def _search_documents(query: str, name: Optional[str] = None) -> str:
        return json.dumps(search_documents_impl(query, docs, name=name), ensure_ascii=False, default=str)[:MAX_TOOL_OUTPUT_CHARS]

    def _extract_tables(name: str) -> str:
        return json.dumps(extract_tables_impl(name, docs), ensure_ascii=False, default=str)[:MAX_TOOL_OUTPUT_CHARS]

    def _extract_key_facts(name: Optional[str] = None) -> str:
        return json.dumps(extract_key_facts_impl(docs, name=name), ensure_ascii=False, default=str)[:MAX_TOOL_OUTPUT_CHARS]

    def _find_dates_and_deadlines(name: Optional[str] = None) -> str:
        return json.dumps(find_dates_and_deadlines_impl(docs, name=name), ensure_ascii=False, default=str)[:MAX_TOOL_OUTPUT_CHARS]

    def _find_monetary_values(name: Optional[str] = None) -> str:
        return json.dumps(find_monetary_values_impl(docs, name=name), ensure_ascii=False, default=str)[:MAX_TOOL_OUTPUT_CHARS]

    def _compare_documents(document_a: str, document_b: str) -> str:
        return json.dumps(compare_documents_impl(document_a, document_b, docs), ensure_ascii=False, default=str)[:MAX_TOOL_OUTPUT_CHARS]

    def _compare_document_sections(document_a: str, page_a: int, document_b: str, page_b: int) -> str:
        return json.dumps(compare_document_sections_impl(document_a, page_a, document_b, page_b, docs), ensure_ascii=False, default=str)[:MAX_TOOL_OUTPUT_CHARS]

    def _detect_conflicts(document_a: str, document_b: str) -> str:
        return json.dumps(detect_conflicts_impl(document_a, document_b, docs), ensure_ascii=False, default=str)[:MAX_TOOL_OUTPUT_CHARS]

    def _analyze_csv(name: str) -> str:
        return json.dumps(analyze_csv_impl(name, docs), ensure_ascii=False, default=str)[:MAX_TOOL_OUTPUT_CHARS]

    def _calculate(expression: str) -> str:
        return json.dumps(safe_calculate(expression), ensure_ascii=False, default=str)[:MAX_TOOL_OUTPUT_CHARS]

    def _retrieve_evidence(query: str, top_k: int = 5) -> str:
        return json.dumps(retrieve_relevant_chunks(query, docs, top_k=top_k), ensure_ascii=False, default=str)[:MAX_TOOL_OUTPUT_CHARS]

    def _verify_claims(claims: List[str]) -> str:
        return json.dumps(batch_verify_claims(claims, docs), ensure_ascii=False, default=str)[:MAX_TOOL_OUTPUT_CHARS]

    def _validate_source_references(source_file: str, page_number: Optional[int] = None) -> str:
        return json.dumps(validate_source_references_impl(source_file, docs, page_number=page_number), ensure_ascii=False, default=str)[:MAX_TOOL_OUTPUT_CHARS]

    def _generate_report(title: Optional[str] = "Aegis Intelligence Report", summary: Optional[str] = "", findings: Optional[List[str]] = None) -> str:
        res = build_structured_report(title or "Aegis Report", "User task", list(docs.values()), findings or [])
        return res[:MAX_TOOL_OUTPUT_CHARS]

    return [
        StructuredTool.from_function(func=_list_documents, name="list_documents", description="List loaded filenames, types, page counts, CSV shape, and scanned status.", args_schema=ListDocumentsInput),
        StructuredTool.from_function(func=_get_document_metadata, name="get_document_metadata", description="Retrieve metadata dictionary for a document.", args_schema=GetDocumentMetadataInput),
        StructuredTool.from_function(func=_read_document, name="read_document", description="Retrieve section or bounded page excerpt by filename and optional page_number.", args_schema=ReadDocumentInput),
        StructuredTool.from_function(func=_search_documents, name="search_documents", description="Search query keywords across documents with page numbers and line snippets.", args_schema=SearchDocumentsInput),
        StructuredTool.from_function(func=_extract_tables, name="extract_tables", description="Extract table data from DOCX or CSV datasets.", args_schema=ExtractTablesInput),
        StructuredTool.from_function(func=_extract_key_facts, name="extract_key_facts", description="Extract candidate dates, monetary amounts, emails, and obligations.", args_schema=ExtractKeyFactsInput),
        StructuredTool.from_function(func=_find_dates_and_deadlines, name="find_dates_and_deadlines", description="Extract candidate dates and deadline clauses.", args_schema=FindDatesAndDeadlinesInput),
        StructuredTool.from_function(func=_find_monetary_values, name="find_monetary_values", description="Extract currency amounts and prices.", args_schema=FindMonetaryValuesInput),
        StructuredTool.from_function(func=_compare_documents, name="compare_documents", description="Compare two documents line-by-line, computing shared/unique lines and similarity %.", args_schema=CompareDocumentsInput),
        StructuredTool.from_function(func=_compare_document_sections, name="compare_document_sections", description="Compare specific page numbers or sections between two documents.", args_schema=CompareDocumentSectionsInput),
        StructuredTool.from_function(func=_detect_conflicts, name="detect_conflicts", description="Detect conflicting dates, monetary amounts, or clauses between document pairs.", args_schema=DetectConflictsInput),
        StructuredTool.from_function(func=_analyze_csv, name="analyze_csv", description="Analyze a CSV dataset: row/column dimensions, missing values, duplicates, and numeric stats.", args_schema=AnalyzeCsvInput),
        StructuredTool.from_function(func=_calculate, name="calculate", description="Perform deterministic numeric math calculations (e.g., '(12000 - 8500) / 12000 * 100').", args_schema=CalculateInput),
        StructuredTool.from_function(func=_retrieve_evidence, name="retrieve_evidence", description="Retrieve top-k relevant text chunks matching a query.", args_schema=RetrieveEvidenceInput),
        StructuredTool.from_function(func=_verify_claims, name="verify_claims", description="Check claim support ('supported', 'partially_supported', 'contradicted', 'not_found') against document evidence.", args_schema=VerifyClaimsInput),
        StructuredTool.from_function(func=_validate_source_references, name="validate_source_references", description="Validate whether source file and page numbers exist in workspace.", args_schema=ValidateSourceReferencesInput),
        StructuredTool.from_function(func=_generate_report, name="generate_report", description="Generate a structured Markdown intelligence report.", args_schema=GenerateReportInput),
    ]
