# Aegis — AI Document Intelligence Agent

Aegis is an autonomous AI document intelligence agent powered by **LangGraph**, **Google Gemini**, **OpenAI**, and **LangChain**. It parses uploaded documents (PDF, DOCX, CSV), dynamically selects and executes registered Python tools, verifies facts against document evidence, and synthesizes structured intelligence reports.

---

## 🌟 Key Upgrades & Architecture

- **Modular Architecture**: Clean separation of UI (`app.py`), agent orchestration (`agents/`), parsing services (`services/`), and automated test suites (`tests/`).
- **LangGraph Agent Execution Loop**: 4-node state graph (`planner` ➔ `executor` ➔ `verifier` ➔ `synthesizer`) with bounded iterations (`max_iterations = 6`) to prevent infinite recursion.
- **9 Registered Tools**:
  1. `list_documents`: Filenames, formats, page counts, character length, CSV shape, scanned flag.
  2. `read_document`: Section or bounded page excerpt retrieval.
  3. `search_documents`: Keyword & phrase search with page numbers and line snippets.
  4. `extract_key_facts`: Regex & heuristic extraction of dates, amounts, emails, obligations.
  5. `compare_documents`: Text line diffs, shared/unique lines, Jaccard similarity index %.
  6. `analyze_csv`: Dimensions, missing values per column, duplicate row counts, numeric describe() stats.
  7. `calculate`: AST-based safe math calculator (eval-free).
  8. `verify_claims`: Claim status checking (`supported`, `partially_supported`, `contradicted`, `not_found`).
  9. `generate_report`: Structured Markdown report generation.
- **Evidence Verification Engine**: Verifies report statements against retrieved text excerpts and exact page numbers.
- **Deterministic Local Fallback**: When API keys are missing, 401 unauthorized, or 429 quota exhausted, Aegis routes cleanly to an offline deterministic execution engine without crashing.
- **Real-Time Streamlit UI Telemetry**: Live trace monitor displaying tool calls, arguments, timing, observations, claim verification panel, and Markdown report download.

---

## 🛠️ Project Directory Structure

```
aegis/
├── app.py                      # Streamlit web UI & real-time telemetry monitor
├── agents/
│   ├── orchestrator.py         # LangGraph 4-node loop, state graph & fallback
│   ├── tools.py                # 9 registered tools with Pydantic schemas & safe calculator
│   ├── state.py                # Telemetry event & evidence claim data models
│   └── verification.py         # Evidence claim verification engine
├── services/
│   ├── document_parser.py      # PDF (with page tracking & OCR detection), DOCX, CSV parsing
│   ├── local_analyzer.py       # Deterministic offline analysis engine
│   └── llm_client.py           # Unified client for Gemini / OpenAI with 401/429 error handling
├── tests/
│   ├── test_document_parser.py # Document parsing & scanned PDF tests
│   ├── test_local_analyzer.py  # Local analyzer & regex fact tests
│   ├── test_verification.py    # Claim verification & grounding tests
│   ├── test_tools.py           # 9 registered tools & AST calculator tests
│   └── test_agent.py           # LangGraph loop, iterations limit, and fallback tests
├── pytest.ini                  # Pytest configuration
├── requirements.txt            # Python dependencies
├── README.md                   # Project documentation
└── sample_data.csv             # Sample dataset for testing
```

---

## 🚀 Quickstart Guide (Windows PowerShell)

### 1. Create & Activate Virtual Environment
```powershell
python -m venv .venv
.\.venv\Scripts\Activate.ps1
```

### 2. Install Dependencies
```powershell
pip install -r requirements.txt
```

### 3. Configure API Keys (Optional)
Create or edit `.env` in the project root:
```env
GEMINI_API_KEY="your-gemini-api-key-here"
OPENAI_API_KEY="your-openai-api-key-here"
```
*(If left empty, Aegis automatically runs in Local Fallback Mode using its deterministic offline engine.)*

### 4. Run Automated Test Suite
```powershell
.\.venv\Scripts\pytest -v
```

### 5. Launch Streamlit Web Application
```powershell
.\.venv\Scripts\streamlit run app.py
```

---

## 🧪 Testing Modes

1. **API-Enabled Mode**:
   - Provide a valid `GEMINI_API_KEY` or `OPENAI_API_KEY` in `.env` or sidebar.
   - Aegis performs autonomous LLM tool selection via Gemini / OpenAI.
2. **Local Fallback Mode**:
   - Leave API key empty or input an invalid key.
   - Aegis notifies the user and executes local deterministic document search, fact extraction, comparison, CSV profiling, and report generation without consuming API credits.

---

## ⚠️ Notes & Limitations

- **Scanned PDFs**: Aegis inspects page text length. PDFs with sparse text are flagged with an explicit OCR notice.
- **Untrusted Document Boundaries**: System prompts explicitly prevent uploaded document text from overriding agent system directives or invoking unauthorized external actions.
