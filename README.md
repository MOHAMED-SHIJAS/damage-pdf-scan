# Vision & Multimodal Workflow Automation Agent

Autonomous agent built in **Plain Python 3** (no LangChain, CrewAI, or heavy agent libraries).

---

## 📁 Project Structure in Explorer

```text
prd/
├── PRD.md              # Complete requirements, architecture, and the 7 parts
├── task.md             # Sequential step-by-step build checklist (Parts 1-8 Completed)
├── app.py              # Streamlit Web Dashboard & Interactive Publishing Studio
├── requirements.txt    # Package dependencies for publishing (streamlit)
├── main.py             # Main root entry point (python main.py 1..8, w)
├── run.bat             # Quick Windows launcher (.\run.bat 1..8, web)
├── python.cmd          # Helper linking to Python 3.11
├── index.html          # Visual Dashboard to open in any Browser / Explorer
├── memory_store.json   # Part 4: Persistent cross-task memory store
├── agent_audit.log     # Part 6: Immutable chronological audit log
├── uploads/            # Multi-format user documents (PNG, JPG, PDF, CSV, JSON, TXT)
├── agent/
│   ├── loop.py         # Part 1 & 3: ReAct engine (Think -> Act -> Look, Self-Healing)
│   ├── tools.py        # Part 2: Tool Registry & 4 concrete domain tools
│   ├── memory.py       # Part 4: TaskMemoryStore (learned facts & task history)
│   ├── planner.py      # Part 5: Pre-Action Planning Engine (ExecutionPlan)
│   ├── safety.py       # Part 6: SafetyGate (Ask First), StepLimiter, AuditLogger
│   ├── team.py         # Part 8: MultiAgentTeam & message passing (⭐ Bonus)
│   ├── run_part1.py    # Part 1 standalone demo (The Core Loop)
│   ├── run_part2.py    # Part 2 standalone demo (2-4 Tools Registry)
│   ├── run_part3.py    # Part 3 standalone demo (Self-Healing on Tool Errors)
│   ├── run_part4.py    # Part 4 standalone demo (Memory Between Tasks)
│   ├── run_part5.py    # Part 5 standalone demo (A Plan Before It Acts)
│   ├── run_part6.py    # Part 6 standalone demo (Safety, Limits & Audit)
│   ├── run_part7.py    # Part 7 standalone demo (Two-Tier Test Suite)
│   ├── run_part8.py    # Part 8 standalone demo (⭐ Bonus: Team of Agents)
│   └── __init__.py     # Module interface
└── tests/
    ├── test_fake_llm.py # Tier 1: Deterministic offline unit tests (6/6)
    └── test_real_llm.py # Tier 2: Real live multimodal LLM integration
```

---

## 🚀 How to Run in Explorer / Terminal

You can run any part with one command using `.\run.bat`:

```powershell
# Run Part 1: The Core ReAct Loop (Think -> Act -> Look)
.\run.bat 1

# Run Part 2: 2-4 Tools Registry & Execution
.\run.bat 2

# Run Part 3: Tool Error Handling & Self-Healing
.\run.bat 3

# Run Part 4: Cross-Task Persistent Memory (memory_store.json)
.\run.bat 4

# Run Part 5: Pre-Action Planning Engine (Numbered Plan Execution)
.\run.bat 5

# Run Part 6: Safety Guardrails (Ask First, Step Limit & Audit Log)
.\run.bat 6

# Run Part 7: Two-Tier Test Suite (Deterministic Unit Tests in 0.02s)
.\run.bat 7

# Run Part 8 (⭐ Bonus): Team of Agents (Multi-Agent Collaboration)
.\run.bat 8

# Launch Interactive Web Dashboard (Streamlit app.py):
.\run.bat web

# Or run the interactive root menu:
.\run.bat
```

---

## 📁 Multi-Format Document Scanning, Quality Enhancement & Categorization

When a user uploads a document (or selects an uploaded file from `uploads/`), the agent automatically:
1. **Scans & Extracts ONLY Details Present:**
   - Ingests and inspects authentic content from the uploaded file (PDFs like `fyugp result.pdf`, images, CSV spreadsheets, JSON data, and TXT files).
   - Detects the document domain (e.g. Academic Examination Grade Card, Commercial Invoice, Tabular Data).
2. **Diagnoses & Enhances Quality if Degraded:**
   - **For Images:** Measures contrast, brightness, and sharpness. Applies dynamic contrast amplification (+35%), edge sharpening filters (+60%), and luminance normalization. Provides side-by-side **Original vs Enhanced** visual comparisons.
   - **For PDFs & Data:** Standardizes timestamps to ISO-8601, harmonizes vector font glyphs, isolates embedded insignia assets, and cleans formatting inconsistencies.
   - Returns a comprehensive **Quality Score** (e.g. `74% -> 97.5% Enhanced ✨`) and a detailed audit of applied enhancements.
3. **Sorts Details into Suitable Categories:**
   - Automatically organizes extracted data into domain-specific categories:
     - 🎓 / 🏢 **Organization & Academic Institution**
     - 👤 / 🔢 **Identity & Reference Numbers**
     - 📅 **Dates & Chronology**
     - 📊 / 💰 **Performance Metrics / Financial Totals**
     - 📋 **Course Modules / Itemized Breakdown**
     - 🛡️ **Quality & Enhancement Audit**
   - Includes one-click **"Download Categorized JSON"** export in the Web Dashboard.

### How to Use Document Uploads:
1. Open the Web Dashboard via `.\run.bat web` (or `http://localhost:8501`).
2. In the **🌟 Interactive Agent Playground**, select **"📤 Upload Your Own Document"** or **"📁 Select From Uploaded Files (uploads/)"**.
3. The dashboard immediately displays:
   - Live format preview & real-time file metrics.
   - **Quality Diagnosis & Auto-Enhancement Hub** (Quality score gauge & enhancement list).
   - **Extracted Details Sorted into Suitable Categories** (interactive tabs for each category).
4. Click **"🚀 Run Agent Loop"** — the agent reasons over the authentic document, performs verified tool actions, and dispatches the record.
5. In **Part 2 (Tools Registry)** and **Part 8 (Multi-Agent Team)**, you can also select or upload documents to test individual tools or run the full 3-agent collaborative workflow on your uploaded documents.

---

## 🌐 Publishing to Streamlit Community Cloud

You can publish this project online with a public shareable URL:

1. **Push this workspace to GitHub:**
   ```powershell
   git init
   git add .
   git commit -m "Publish Vision Agent Studio"
   git remote add origin https://github.com/<your-username>/<your-repo-name>.git
   git push -u origin main
   ```
2. **Deploy on Streamlit Community Cloud:**
   - Go to [share.streamlit.io](https://share.streamlit.io/) and sign in with GitHub.
   - Click **"New app"** and select your repository.
   - Set **Main file path** to `app.py`.
   - Click **"Deploy!"** — Streamlit automatically detects `requirements.txt` and publishes your live interactive agent studio online!

---

## 📋 Course Curriculum Mapping

| Part | Component | Description & Key Concept | Demo Script |
|---|---|---|---|
| **Part 1** | `agent/loop.py` | `think -> act -> look` ReAct while loop | `.\run.bat 1` |
| **Part 2** | `agent/tools.py` | 4 registered tools with schemas & inspection | `.\run.bat 2` |
| **Part 3** | `agent/loop.py` | Self-healing when tools fail (try/except) | `.\run.bat 3` |
| **Part 4** | `agent/memory.py` | Cross-task persistence via `memory_store.json` | `.\run.bat 4` |
| **Part 5** | `agent/planner.py` | Pre-action numbered plan with `[TODO]` tags | `.\run.bat 5` |
| **Part 6** | `agent/safety.py` | Human gate ("Asks You First"), step limiter, audit log | `.\run.bat 6` |
| **Part 7** | `tests/` | Two-tier tests (FakeLLM offline, Real LLM) | `.\run.bat 7` |
| **⭐ Bonus**| `agent/team.py` | Multi-agent collaboration with structured message passing | `.\run.bat 8` |

---

## 🔍 If Files Are Not Showing in Explorer:
1. **Refresh the Explorer:** Click the `↻` (Refresh) icon at the top of the VS Code / Antigravity IDE Explorer sidebar.
2. **Expand the `agent/` folder:** Click the arrow `▶ agent` in the sidebar to expand and see `loop.py`, `tools.py`, `memory.py`, `planner.py`, and the demo scripts.
3. **Double-click `index.html`:** Opens an interactive visual project dashboard in any web browser.
