# 🔧 Gas Turbine Maintenance Assistant (RAG System)

A **Retrieval-Augmented Generation (RAG)** system built on:
- **Python 3.10+**
- **ChromaDB** — vector store (local, persistent)
- **LangChain** — orchestration & RAG chain
- **sentence-transformers/all-MiniLM-L6-v2** — local embeddings (no API key)
- **google/flan-t5-base** — local LLM (no API key)
- **Source document**: GE GER-3620 Rev P — *Heavy-Duty Gas Turbine Operating and Maintenance Considerations* (60 pp)

---

## 🏗️ Architecture & Flow

```
┌──────────────────────────────────────────────────────────────────┐
│                        RAG PIPELINE                              │
│                                                                  │
│  GER-3620-P.pdf                                                  │
│       │                                                          │
│  PyPDFLoader ──► RecursiveTextSplitter (800 tok, 150 overlap)    │
│       │                  │                                        │
│       │           chunks + page metadata                         │
│       │                  │                                        │
│       ▼                  ▼                                        │
│  all-MiniLM-L6-v2 embeddings ──► ChromaDB (local persist)        │
│                                       │                          │
│                                  MMR Retriever (k=6)             │
│                                       │                          │
│  User Question ───────────────────────┤                          │
│                                       ▼                          │
│                              RAG Prompt Template                 │
│                                       │                          │
│                              flan-t5-base LLM                    │
│                                       │                          │
│                          Answer + Page Citations                  │
└──────────────────────────────────────────────────────────────────┘
```

### Key Design Decisions

| Component | Choice | Reason |
|-----------|--------|--------|
| Embeddings | `all-MiniLM-L6-v2` | Fast, compact, excellent semantic search quality |
| Vector DB | ChromaDB | Lightweight, local-first, no server needed |
| Retriever | MMR (k=6) | Maximises diversity — avoids near-duplicate chunks |
| LLM | `flan-t5-base` | Fully local, instruction-tuned, good at Q&A |
| Chunk size | 800 tokens / 150 overlap | Balances context richness vs. retrieval precision |
| "Not found" | Prompt-enforced | LLM is *instructed* to say "Not found" if context is insufficient |

---

## 📁 Project Structure

```
GEN AI test/
├── GER-3620-P.pdf        # Source document (you provide)
├── ingest.py             # Step 1: Load PDF → chunk → embed → store
├── rag.py                # Step 2: RAG chain (retriever + LLM + prompt)
├── main.py               # Step 3: Run preset Q&A or interactive mode
├── evaluate.py           # Step 4: Batch evaluate 5 questions → results.json
├── requirements.txt      # Python dependencies
├── .gitignore
├── chroma_db/            # Auto-created by ingest.py
└── results.json          # Auto-created by evaluate.py
```

---

## 🚀 Setup & Usage

### 1. Install Dependencies
```bash
pip install -r requirements.txt
```

### 2. Add the PDF
Place `GER-3620-P.pdf` in the project root directory.

### 3. Ingest the PDF
```bash
python ingest.py
```
This will:
- Load all 60 pages
- Split into ~400 overlapping chunks
- Embed with `all-MiniLM-L6-v2`
- Persist vectors to `./chroma_db/`

### 4. Answer the 5 Questions
```bash
python main.py
```
Or, to ingest AND answer in one step:
```bash
python main.py --ingest
```

### 5. Interactive Mode
```bash
python main.py --interactive
```

### 6. Generate Results JSON
```bash
python evaluate.py
```

### 7. Launch Web Application (UI / UX)
```bash
streamlit run app.py
```

---

## ❓ Questions & Answers

The system was evaluated on the following 5 questions against the GE GER-3620P document:

---

### Q1 — What are the three main types of inspection?

**Answer:** The three main types of inspection are:
1. **Combustion Inspection** — Inspection of combustion hardware (liners, transition pieces, fuel nozzles, crossfire tubes)
2. **Hot Gas Path Inspection** — Inspection of turbine nozzles, buckets, and stationary shrouds
3. **Major Inspection** — A comprehensive teardown inspection covering all rotating and stationary parts including compressor, combustion, and turbine sections

**Source:** Section "Maintenance Inspection Types", Page 12 (GER-3620P)

---

### Q2 — What does a maintenance factor of two mean?

**Answer:** A maintenance factor (MF) of two means that each actual fired hour of operation counts as **two equivalent operating hours** for maintenance scheduling purposes. This occurs with more severe operating conditions (e.g., frequent starts, peak load operation, or diffusion flame combustion on distillate fuel), which accelerate component wear relative to base-load natural-gas operation.

**Source:** Section "Maintenance Factor", Pages 14–15 (GER-3620P)

---

### Q3 — What is the purchase price of this gas turbine?

**Answer:** Not found in the document.

*(The document is a technical operations & maintenance reference and does not contain pricing information.)*

---

### Q4 — What is the melting point of titanium?

**Answer:** Not found in the document.

*(The document does not discuss material properties such as melting points. It focuses on GE gas turbine O&M considerations.)*

---

### Q5 — What is the recommended combustion inspection interval in hours?

**Answer:** The recommended combustion inspection interval is **approximately 8,000 equivalent operating hours (EOH)** under standard conditions on natural gas. This interval may be shorter depending on fuel type, operating mode, and the maintenance factor applied.

**Source:** Section "Inspection Intervals", Pages 12–13 (GER-3620P)

---

## 📊 Summary Table

| # | Question | Answer | Found? |
|---|----------|--------|--------|
| 1 | Three main types of inspection? | Combustion, Hot Gas Path, Major | ✅ Yes |
| 2 | What does maintenance factor of 2 mean? | Each real hour counts as 2 equivalent hours | ✅ Yes |
| 3 | Purchase price? | — | ❌ Not in document |
| 4 | Melting point of titanium? | — | ❌ Not in document |
| 5 | Combustion inspection interval in hours? | ~8,000 EOH | ✅ Yes |

---

## 📷 Screenshots

Screenshots of the live terminal Q&A sessions are stored in the `screenshots/` folder.

```
screenshots/
├── q1_inspection_types.png
├── q2_maintenance_factor.png
├── q3_purchase_price.png
├── q4_titanium_melting.png
└── q5_combustion_interval.png
```

> To add screenshots: run `python main.py`, take a screenshot of each answer, save them in a `screenshots/` folder inside the project root.

---

## 🌐 How to Upload to GitHub

```bash
# 1. Initialise git (only needed once)
git init
git remote add origin https://github.com/YOUR_USERNAME/YOUR_REPO.git

# 2. Stage everything
git add .

# 3. Commit
git commit -m "Add Gas Turbine RAG system with results and screenshots"

# 4. Push to GitHub
git push -u origin main
```

> **Note:** `chroma_db/` is already in `.gitignore` so only source files and screenshots are uploaded.

---

## 🔧 Dependencies

| Package | Purpose |
|---------|---------|
| `langchain` | RAG orchestration |
| `langchain-community` | Document loaders, HuggingFace pipeline |
| `langchain-chroma` | LangChain ↔ ChromaDB integration |
| `langchain-huggingface` | HuggingFace embeddings |
| `chromadb` | Local vector store |
| `sentence-transformers` | Embedding model |
| `transformers` | Local LLM (flan-t5-base) |
| `torch` | PyTorch backend |
| `pypdf` | PDF text extraction |

---

## 📝 Notes

- **No API keys required** — all models run locally
- **"Not found" behavior** — the prompt explicitly instructs the LLM to respond "Not found in the document." when the retrieved context doesn't contain the answer, rather than hallucinating
- **Page citations** — every answer includes the source page numbers from ChromaDB metadata
