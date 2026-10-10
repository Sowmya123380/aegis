"""Aegis Document Intelligence Agent - State & Event Data Models.

Defines structured state types for execution events, evidence claims,
and LangGraph task execution state tracking.
"""

from typing import Any, Dict, List, Optional, Sequence, TypedDict, Union
from pydantic import BaseModel, Field


class AgentEvent(TypedDict, total=False):
    """Execution event telemetry record."""
    time: str
    stage: str
    status: str  # 'running', 'completed', 'failed', 'info'
    detail: str
    tool_name: Optional[str]
    arguments: Optional[Dict[str, Any]]
    result: Optional[Any]
    elapsed: Optional[float]
    elapsed_seconds: Optional[float]


class EvidenceClaim(TypedDict):
    """Verified document claim grounding record."""
    claim: str
    source_file: str
    page_or_location: str
    excerpt: str
    status: str  # 'supported', 'partially_supported', 'contradicted', 'not_found'


class AgentState(TypedDict, total=False):
    """LangGraph execution state."""
    messages: Sequence[Any]
    iteration: int
    events: List[Dict[str, Any]]
    evidence_claims: List[Dict[str, Any]]
    final_report: Optional[str]
    execution_mode: str
    status: str
