
import json
import os
from pathlib import Path
from auth import require_login

import pandas as pd
import streamlit as st
from dotenv import load_dotenv

from document_processor import (
    extract_text,
    extract_key_information,
    summarize_locally,
    compare_documents,
)

load_dotenv(override=True)

st.set_page_config(
    page_title="Aegis | AI Document Intelligence",
    page_icon="🛡️",
    layout="wide",
)
st.set_page_config(
    page_title="Aegis | AI Document Intelligence",
    page_icon="🛡️",
    layout="wide",
)

# -------------------- AUTHENTICATION --------------------
require_login()

# -------------------- SESSION STATE --------------------
DEFAULTS = {
    "aegis_results": {},
    "aegis_live_report": None,
    "gemini_model": os.getenv("GEMINI_MODEL", "gemini-2.5-flash"),
    "gemini_api_key": os.getenv("GEMINI_API_KEY", ""),
}

# -------------------- SESSION STATE --------------------

DEFAULTS = {
    "aegis_results": {},
    "aegis_live_report": None,
    "gemini_model": os.getenv("GEMINI_MODEL", "gemini-2.5-flash"),
    "gemini_api_key": os.getenv("GEMINI_API_KEY", ""),
}

for key, value in DEFAULTS.items():
    if key not in st.session_state:
        st.session_state[key] = value


# -------------------- HELPER FUNCTIONS --------------------

def get_document_text(result):
    """Return text from a processed document result."""
    if isinstance(result, dict):
        return str(result.get("text", result.get("content", "")))
    return str(result)


def normalize_document_result(name, result):
    """Normalize a document result for planning and verification."""
    text = get_document_text(result)
    return {
        "type": "csv" if name.lower().endswith(".csv") else "document",
        "text": text,
        "content": text,
    }


def create_fallback_plan(task, docs):
    """Create a local plan if the planner module is unavailable."""
    steps = [
        {
            "step_id": 1,
            "title": "Inspect documents",
            "description": f"Inspect {len(docs)} processed document(s).",
            "status": "ready",
        },
        {
            "step_id": 2,
            "title": "Extract important information",
            "description": "Identify key facts, dates, amounts, and keywords.",
            "status": "ready",
        },
        {
            "step_id": 3,
            "title": "Generate and verify the report",
            "description": "Answer the task and check claims against document text.",
            "status": "ready",
        },
    ]
    return {
        "user_task": task,
        "document_count": len(docs),
        "subtasks_count": len(steps),
        "subtasks": steps,
    }


def make_local_report(task, docs):
    """Produce a useful report without calling the Gemini API."""
    output = []

    for name, result in docs.items():
        text = get_document_text(result)
        info = extract_key_information(text) if text.strip() else {}

        output.append(f"## {name}")
        output.append(
            summarize_locally(text, max_sentences=5)
            if text.strip()
            else "No extractable text was found."
        )

        if info:
            output.append("\n### Key information")
            output.append(json.dumps(info, indent=2, ensure_ascii=False, default=str))

    return (
        "Local document analysis completed.\n\n"
        f"Requested task: {task}\n\n"
        + "\n\n".join(output)
        + "\n\nNote: This report was generated locally because the AI model "
        "was unavailable or not configured."
    )


def run_gemini_agent(task, docs):
    """Run the agent while remaining compatible with either agent.py signature."""
    from agent import run_agent
    import inspect

    context_parts = []
    for name, result in docs.items():
        text = get_document_text(result)
        context_parts.append(
            f"DOCUMENT NAME: {name}\n"
            f"DOCUMENT CONTENT:\n{text[:10000]}"
        )

    prompt = (
        "You are Aegis, an AI Document Intelligence Agent.\n"
        "Answer the user's task using only the supplied document evidence.\n"
        "Do not invent facts. Clearly state when information is missing.\n"
        "Mention the source document when possible.\n\n"
        f"USER TASK:\n{task}\n\n"
        "DOCUMENTS:\n"
        + "\n\n---\n\n".join(context_parts)
    )

    parameters = inspect.signature(run_agent).parameters

    if "model" in parameters or "api_key" in parameters:
        kwargs = {}
        if "model" in parameters:
            kwargs["model"] = st.session_state.get("gemini_model")
        if "api_key" in parameters:
            kwargs["api_key"] = (
                st.session_state.get("gemini_api_key")
                or os.getenv("GEMINI_API_KEY")
            )
        return run_agent(prompt, **kwargs)

    return run_agent(prompt)


# -------------------- SIDEBAR --------------------

with st.sidebar:
    st.title("🛡️ Aegis")
    st.caption("AI Document Intelligence Agent")
    st.divider()

    st.subheader("Gemini Configuration")

    st.text_input(
        "Gemini API key",
        type="password",
        key="gemini_api_key",
        help="Your key stays in this app session. Do not share it.",
    )

    st.selectbox(
        "Gemini model",
        options=[
            "gemini-2.5-flash",
            "gemini-2.5-pro",
            "gemini-flash-latest",
        ],
        key="gemini_model",
    )

    st.caption(
        "If the API quota is exhausted, use Local Analysis to continue "
        "without an AI API call."
    )

    st.divider()
    st.markdown("**Supported formats**")
    st.write("PDF · DOCX · CSV")

    if st.button("Clear current session", use_container_width=True):
        st.session_state["aegis_results"] = {}
        st.session_state["aegis_live_report"] = None
        st.rerun()


# -------------------- HEADER --------------------

st.title("🛡️ Aegis — AI Document Intelligence")
st.write(
    "Upload documents, extract facts, compare files, analyze CSV data, "
    "and run an AI-assisted document workflow."
)

st.divider()


# -------------------- DOCUMENT UPLOAD --------------------

st.header("📁 1. Upload Documents")

uploaded_files = st.file_uploader(
    "Select one or more PDF, DOCX, or CSV files",
    type=["pdf", "docx", "csv"],
    accept_multiple_files=True,
)

if st.button(
    "⚙️ Process Documents",
    type="primary",
    use_container_width=True,
):
    if not uploaded_files:
        st.warning("Please upload at least one document first.")
    else:
        processed = {}
        progress = st.progress(0)

        for index, uploaded in enumerate(uploaded_files):
            try:
                file_bytes = uploaded.getvalue()
                text = extract_text(uploaded.name, file_bytes)

                processed[uploaded.name] = {
                    "text": text,
                    "content": text,
                    "type": (
                        "csv" if uploaded.name.lower().endswith(".csv")
                        else "document"
                    ),
                    "size_bytes": len(file_bytes),
                }

                st.write(f"✅ Processed: {uploaded.name}")

            except Exception as exc:
                st.error(f"Could not process {uploaded.name}: {exc}")

            progress.progress((index + 1) / len(uploaded_files))

        if processed:
            st.session_state["aegis_results"] = processed
            st.session_state["aegis_live_report"] = None
            st.success(f"Processed {len(processed)} document(s).")


docs = st.session_state.get("aegis_results", {})

if docs:
    st.subheader("Loaded documents")

    for name, result in docs.items():
        text = get_document_text(result)
        file_size = result.get("size_bytes", 0) if isinstance(result, dict) else 0

        with st.expander(f"📄 {name}"):
            col1, col2 = st.columns(2)
            col1.metric("Characters", f"{len(text):,}")
            col2.metric("File size", f"{file_size:,} bytes")
            st.text_area(
                "Extracted text preview",
                value=text[:4000],
                height=180,
                key=f"preview_{name}",
            )


# -------------------- DOCUMENT ANALYSIS --------------------

st.divider()
st.header("📊 2. Document Analysis")

tab_summary, tab_compare, tab_csv, tab_facts = st.tabs(
    [
        "Summaries",
        "Compare Documents",
        "CSV Analysis",
        "Key Information",
    ]
)

with tab_summary:
    if not docs:
        st.info("Upload and process documents to view summaries.")
    else:
        for name, result in docs.items():
            text = get_document_text(result)

            with st.expander(f"Summary: {name}", expanded=True):
                if text.strip():
                    st.write(summarize_locally(text, max_sentences=6))
                else:
                    st.warning("No extractable text is available.")


with tab_compare:
    text_documents = {
        name: get_document_text(result)
        for name, result in docs.items()
        if not name.lower().endswith(".csv")
    }

    if len(text_documents) < 2:
        st.info("Upload and process at least two PDF/DOCX documents to compare.")
    else:
        names = list(text_documents.keys())
        col1, col2 = st.columns(2)

        with col1:
            doc_a = st.selectbox(
                "First document",
                names,
                key="compare_doc_a",
            )

        with col2:
            doc_b_options = [name for name in names if name != doc_a]
            doc_b = st.selectbox(
                "Second document",
                doc_b_options,
                key="compare_doc_b",
            )

        if st.button("🔍 Compare Selected Documents"):
            comparison = compare_documents(
                text_documents[doc_a],
                text_documents[doc_b],
            )
            st.session_state["aegis_comparison"] = comparison

        comparison = st.session_state.get("aegis_comparison")

        if comparison:
            st.metric(
                "Text similarity",
                f"{comparison.get('similarity_percent', 0)}%",
            )

            col1, col2 = st.columns(2)

            with col1:
                st.subheader(f"Only in {doc_a}")
                st.write(comparison.get("only_in_document_1", []))

            with col2:
                st.subheader(f"Only in {doc_b}")
                st.write(comparison.get("only_in_document_2", []))

            with st.expander("Common lines"):
                st.write(comparison.get("common_lines", []))


with tab_csv:
    csv_docs = {
        name: result
        for name, result in docs.items()
        if name.lower().endswith(".csv")
    }

    if not csv_docs:
        st.info("Upload and process a CSV file to see dataset analysis.")
    else:
        selected_csv = st.selectbox(
            "Select CSV file",
            list(csv_docs.keys()),
            key="selected_csv",
        )

        try:
            raw_csv = next(
                (
                    uploaded.getvalue()
                    for uploaded in (uploaded_files or [])
                    if uploaded.name == selected_csv
                ),
                None,
            )

            if raw_csv is not None:
                from io import BytesIO
                dataframe = pd.read_csv(BytesIO(raw_csv))
            else:
                st.info(
                    "Re-upload the CSV in this session to display its table. "
                    "Extracted text remains available for the agent."
                )
                dataframe = None

            if dataframe is not None:
                st.dataframe(dataframe, use_container_width=True)
                c1, c2, c3 = st.columns(3)
                c1.metric("Rows", len(dataframe))
                c2.metric("Columns", len(dataframe.columns))
                c3.metric("Missing cells", int(dataframe.isna().sum().sum()))

                st.subheader("Numeric summary")
                st.dataframe(
                    dataframe.describe(include="all").transpose(),
                    use_container_width=True,
                )

        except Exception as exc:
            st.error(f"CSV analysis failed: {exc}")


with tab_facts:
    if not docs:
        st.info("Process documents to extract key information.")
    else:
        for name, result in docs.items():
            text = get_document_text(result)
            with st.expander(f"Key information: {name}"):
                if text.strip():
                    info = extract_key_information(text)
                    st.json(info)
                else:
                    st.warning("No text is available for extraction.")


# -------------------- AI AGENT WORKFLOW --------------------

# -------------------- AUTOMATED AEGIS AGENT WORKFLOW --------------------

st.divider()
st.header("🤖 Aegis Agent Workflow")

st.write(
    "Aegis automatically plans, analyzes, extracts, verifies, "
    "and generates a report from your processed documents."
)

st.markdown("""
**Workflow**

1. 📂 Document Inspection
2. 🧠 AI Task Planning
3. 🔍 Information and Key Fact Extraction
4. 📊 Document Comparison and CSV Analysis
5. ✅ Evidence Verification
6. 📝 Final Intelligence Report
""")

if st.button(
    "▶ Execute Aegis Agent Workflow",
    type="primary",
    use_container_width=True,
):
    docs = st.session_state.get("aegis_results", {})

    if not docs:
        st.warning("Please upload and process documents first.")
    else:
        with st.status("Aegis workflow running...", expanded=True) as status:
            try:
                # Step 1: Inspect documents
                st.write("📂 Step 1: Inspecting documents...")

                document_info = {}
                for name, result in docs.items():
                    text = get_document_text(result)
                    document_info[name] = {
                        "characters": len(text),
                        "words": len(text.split()),
                        "type": (
                            "CSV" if name.lower().endswith(".csv")
                            else Path(name).suffix.upper().lstrip(".")
                        ),
                    }

                # Step 2: Automatically create a plan
                st.write("🧠 Step 2: Creating the agent plan...")

                try:
                    from agents.planner import create_agent_plan

                    plan_docs = {
                        name: normalize_document_result(name, result)
                        for name, result in docs.items()
                    }

                    plan = create_agent_plan(
                        "Analyze all uploaded documents, extract key facts, "
                        "identify important dates and amounts, compare "
                        "documents where applicable, analyze CSV data, "
                        "verify findings, and generate a final report.",
                        plan_docs,
                    )
                except Exception:
                    plan = create_fallback_plan(
                        "Automatic document intelligence workflow",
                        docs,
                    )

                # Step 3: Extract key information
                st.write("🔍 Step 3: Extracting key information...")

                extracted_information = {}
                for name, result in docs.items():
                    text = get_document_text(result)
                    extracted_information[name] = (
                        extract_key_information(text)
                        if text.strip()
                        else {}
                    )

                # Step 4: Generate document summaries
                st.write("📄 Step 4: Generating document summaries...")

                summaries = {}
                for name, result in docs.items():
                    text = get_document_text(result)
                    summaries[name] = (
                        summarize_locally(text, max_sentences=6)
                        if text.strip()
                        else "No text available."
                    )

                # Step 5: Compare text documents when possible
                st.write("📊 Step 5: Comparing documents and profiling CSV files...")

                text_docs = {
                    name: get_document_text(result)
                    for name, result in docs.items()
                    if not name.lower().endswith(".csv")
                }

                comparisons = {}
                names = list(text_docs.keys())

                for i in range(len(names)):
                    for j in range(i + 1, len(names)):
                        comparisons[f"{names[i]} vs {names[j]}"] = (
                            compare_documents(
                                text_docs[names[i]],
                                text_docs[names[j]],
                            )
                        )

                csv_profiles = {}
                for name, result in docs.items():
                    if name.lower().endswith(".csv"):
                        text = get_document_text(result)
                        csv_profiles[name] = {
                            "rows_in_extracted_text": len(text.splitlines()),
                            "columns_or_fields": (
                                text.splitlines()[0].split(",")
                                if text.strip()
                                else []
                            ),
                        }

                # Step 6: Attempt AI synthesis; retain local results if unavailable
                st.write("🤖 Step 6: Generating the intelligence report...")

                automatic_task = (
                    "Analyze all supplied documents. Summarize each document, "
                    "extract key information, dates, amounts and email addresses, "
                    "identify similarities and differences between documents, "
                    "and provide evidence-based findings. Clearly identify "
                    "information that cannot be verified. Do not invent facts."
                )

                try:
                    ai_answer = run_gemini_agent(automatic_task, docs)
                    ai_status = "AI response generated"
                except Exception as ai_error:
                    ai_answer = (
                        "Gemini was unavailable, so Aegis completed the "
                        "local analysis workflow. Reason: "
                        f"{type(ai_error).__name__}: {ai_error}"
                    )
                    ai_status = "Local fallback used"

                # Step 7: Verify AI response against available document text
                st.write("✅ Step 7: Checking response against document evidence...")

                verification = []
                try:
                    from agents.verification import batch_verify_claims

                    claims = [
                        line.strip().lstrip("-•*0123456789. \t")
                        for line in str(ai_answer).splitlines()
                        if len(line.strip()) > 20
                        and not line.strip().startswith("#")
                    ][:10]

                    evidence_docs = {
                        name: {
                            "text": get_document_text(result),
                            "content": get_document_text(result),
                        }
                        for name, result in docs.items()
                    }

                    if claims:
                        verification = batch_verify_claims(
                            claims,
                            evidence_docs,
                        )
                except Exception as verify_error:
                    verification = {
                        "message": "Evidence verification could not complete.",
                        "error": str(verify_error),
                    }

                # Step 8: Save final report
                final_report = {
                    "workflow": "Aegis Automated Document Intelligence",
                    "status": "completed",
                    "ai_status": ai_status,
                    "documents": document_info,
                    "task_plan": plan,
                    "summaries": summaries,
                    "extracted_information": extracted_information,
                    "document_comparisons": comparisons,
                    "csv_profiles": csv_profiles,
                    "ai_findings": str(ai_answer),
                    "evidence_verification": verification,
                }

                st.session_state["aegis_live_report"] = final_report

                status.update(
                    label="Aegis workflow completed",
                    state="complete",
                    expanded=False,
                )

            except Exception as exc:
                status.update(
                    label="Aegis workflow failed",
                    state="error",
                    expanded=True,
                )
                st.error(f"{type(exc).__name__}: {exc}")


# -------------------- AUTOMATED REPORT DISPLAY --------------------

report = st.session_state.get("aegis_live_report")

if report:
    st.divider()
    st.header("📋 Aegis Intelligence Report")

    st.success("Automated workflow execution finished.")

    st.subheader("📂 Document Inventory")
    st.json(report.get("documents", {}))

    st.subheader("🧠 Agent Task Plan")
    st.json(report.get("task_plan", {}))

    st.subheader("📄 Document Summaries")
    for name, summary in report.get("summaries", {}).items():
        with st.expander(name):
            st.write(summary)

    st.subheader("🔍 Extracted Information")
    st.json(report.get("extracted_information", {}))

    st.subheader("📊 Document Comparisons")
    if report.get("document_comparisons"):
        st.json(report["document_comparisons"])
    else:
        st.info("Upload two or more PDF/DOCX files for comparisons.")

    st.subheader("📈 CSV Profiles")
    st.json(report.get("csv_profiles", {}))

    st.subheader("🤖 AI Findings")
    st.write(report.get("ai_findings", ""))

    st.subheader("✅ Evidence Verification")
    st.json(report.get("evidence_verification", {}))

    st.download_button(
        "⬇️ Download Complete Aegis Report",
        data=json.dumps(
            report,
            indent=2,
            ensure_ascii=False,
            default=str,
        ),
        file_name="aegis_intelligence_report.json",
        mime="application/json",
        use_container_width=True,
        key="download_automated_aegis_report",
    )

# -------------------- FOOTER --------------------

st.divider()
st.caption(
    "Aegis — AI Document Intelligence Agent | "
    "Local analysis and AI-assisted workflows"
)
