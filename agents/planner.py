"""Aegis Document Intelligence Agent - Task Planner.

Provides task decomposition, goal breakdown, and dependency tracking for agent execution:
- Decomposes user requests into actionable subtasks
- Maps subtasks to registered tools
- Tracks step completion and planning status
"""

from typing import Any, Dict, List, Optional


def create_agent_plan(user_task: str, docs: Dict[str, Any]) -> Dict[str, Any]:
    """Decompose user task into structured sub-goals with assigned tools and completion criteria."""
    task_clean = (user_task or "").strip() or "Analyze documents"
    doc_count = len(docs)
    has_csv = any(isinstance(d.get("dataframe") if d.get("dataframe") is not None else d.get("df"), object) and d.get("type") == "csv" for d in docs.values())
    has_multiple_text = len([d for d in docs.values() if d.get("type") != "csv"]) >= 2

    subtasks: List[Dict[str, Any]] = [
        {
            "step_id": 1,
            "title": "Inventory & Metadata Inspection",
            "description": f"Inspect {doc_count} loaded file(s) and extract character length, pages, and format properties.",
            "recommended_tools": ["list_documents", "get_document_metadata"],
            "status": "pending",
        },
        {
            "step_id": 2,
            "title": "Keyword & Fact Extraction",
            "description": "Extract key dates, monetary values, emails, candidate obligations, and search keywords.",
            "recommended_tools": ["extract_key_facts", "find_dates_and_deadlines", "find_monetary_values", "search_documents"],
            "status": "pending",
        },
    ]

    if has_multiple_text:
        subtasks.append({
            "step_id": 3,
            "title": "Pairwise Comparison & Conflict Detection",
            "description": "Compare text differences between document pairs and detect conflicting dates, amounts, or clauses.",
            "recommended_tools": ["compare_documents", "compare_document_sections", "detect_conflicts"],
            "status": "pending",
        })

    if has_csv:
        subtasks.append({
            "step_id": 4,
            "title": "CSV Dataset Profiling & Calculations",
            "description": "Calculate dataset dimensions, missing cells, duplicate counts, numeric summary stats, and perform calculations.",
            "recommended_tools": ["analyze_csv", "extract_tables", "calculate"],
            "status": "pending",
        })

    subtasks.append({
        "step_id": 5,
        "title": "Evidence Verification & Intelligence Synthesis",
        "description": "Verify findings against retrieved document evidence, validate source citations, and compile final report.",
        "recommended_tools": ["retrieve_evidence", "verify_claims", "validate_source_references", "generate_report"],
        "status": "pending",
    })

    return {
        "user_task": task_clean,
        "document_count": doc_count,
        "subtasks_count": len(subtasks),
        "subtasks": subtasks,
    }
