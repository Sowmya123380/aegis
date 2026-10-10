"""Integration and unit tests for Aegis Agent Orchestrator & Execution Loop."""

import pandas as pd
import pytest
from unittest.mock import MagicMock, patch

from agents.orchestrator import (
    AgentState,
    build_langgraph_agent,
    run_agent,
)
from agents.tools import create_document_tools
from langchain_core.messages import AIMessage, HumanMessage


@pytest.fixture
def sample_docs_dict():
    df = pd.DataFrame({"col_a": [1, 2, 3], "col_b": [10.0, 20.0, 30.0]})
    return {
        "doc_one.txt": {
            "name": "doc_one.txt",
            "type": "txt",
            "text": "Project Alpha overview.\nDeadline: Q3.\nBudget: $50k.",
            "page_count": 1,
        },
        "doc_two.txt": {
            "name": "doc_two.txt",
            "type": "txt",
            "text": "Project Alpha overview.\nDeadline: Q4.\nBudget: $70k.",
            "page_count": 1,
        },
        "data.csv": {
            "name": "data.csv",
            "type": "csv",
            "df": df,
            "dataframe": df,
            "page_count": 1,
        },
    }


def test_run_agent_empty_docs():
    with pytest.raises(ValueError, match="Upload and process at least one document first"):
        run_agent(docs=[])


def test_run_agent_local_fallback_no_key(sample_docs_dict):
    captured_events = []

    def on_event(ev):
        captured_events.append(ev)

    result = run_agent(
        api_key="",
        user_task="Analyze and compare documents.",
        docs=sample_docs_dict,
        on_event=on_event,
    )

    assert isinstance(result, dict)
    assert result["execution_mode"] == "local_fallback"
    assert "Aegis AI — Document Intelligence Report" in result["answer"]
    assert len(captured_events) > 0


def test_langgraph_graph_compilation(sample_docs_dict):
    tools = create_document_tools(sample_docs_dict)
    mock_llm = MagicMock()
    mock_llm.bind_tools.return_value = mock_llm

    events = []
    def dummy_emit(stage, status, detail, **extra):
        events.append(stage)

    graph = build_langgraph_agent(
        llm=mock_llm,
        tools=tools,
        emit=dummy_emit,
        docs=sample_docs_dict,
        max_iterations=3,
    )

    assert graph is not None
    drawable = graph.get_graph()
    nodes = set(drawable.nodes.keys())
    assert "planner" in nodes
    assert "executor" in nodes
    assert "verifier" in nodes
    assert "synthesizer" in nodes


def test_langgraph_bounded_iterations(sample_docs_dict):
    tools = create_document_tools(sample_docs_dict)
    
    mock_llm = MagicMock()
    tool_call_response = AIMessage(
        content="Requesting tool list_documents.",
        tool_calls=[{"name": "list_documents", "args": {}, "id": "call_123"}],
    )
    mock_llm.bind_tools.return_value.invoke.return_value = tool_call_response
    mock_llm.invoke.return_value = AIMessage(content="Final report after reaching max iterations.")

    events = []
    def dummy_emit(stage, status, detail, **extra):
        events.append(stage)

    max_iter = 2
    graph = build_langgraph_agent(
        llm=mock_llm,
        tools=tools,
        emit=dummy_emit,
        docs=sample_docs_dict,
        max_iterations=max_iter,
    )

    initial_state = {
        "messages": [HumanMessage(content="Analyze everything.")],
        "iteration": 0,
        "events": [],
        "evidence_claims": [],
        "final_report": None,
        "execution_mode": "mock_test",
        "status": "running",
    }

    final_state = graph.invoke(initial_state)

    assert final_state["iteration"] >= max_iter
    assert "Final report" in final_state["final_report"]
    assert "Evidence Verification" in events
    assert "Report Generation" in events
