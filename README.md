# 🏥 Healthcare Vector DB – AI-Powered Patient Search & Medical QA

An intelligent, full-stack healthcare application built with **Streamlit**, **PostgreSQL + pgvector**, and **LLM integration via Groq (LLaMA 3 70B)**. It enables vector-based search on patient data and supports advanced AI-driven **natural language medical queries** and **RAG (Retrieval-Augmented Generation)** Q&A with real-time embeddings and analysis.

---

## 🚀 Features

### 🔍 Patient Data Embedding & Search
- Imports patient records from CSV with auto-embedding using Groq LLM embeddings API.
- Stores vector representations (`vector(384)`) in PostgreSQL via `pgvector`.
- Supports duplicate detection, batch embedding, and fast vector search on medical metadata.

### 💬 AI Medical Q&A (RAG)
- Asks complex medical questions and receives accurate, LLaMA 3–generated responses.
- Contextually augments responses using a custom vector store of 100k+ medical Q&A pairs (like MedQuAD).
- Optionally retrieves top-k relevant QA contexts for grounding.

### 🧠 AI-Powered SQL Agent
- Natural language queries (e.g., *"Show average hemoglobin for diabetic females"*) are converted to SQL.
- Dynamically queries patient data using AI-generated SQL and presents structured results.
- Fully schema-aware and capable of handling PostgreSQL metadata, JSONB fields, and vector similarity.

### 📊 Real-Time Data Import & Enrichment
- Patient & QA data imported with:
  - Real-time progress tracking
  - Duplicate detection
  - Batch optimized DB inserts
- Option to generate embeddings locally or via Groq API.

### 🧪 RAG Evaluation with RAGAS
- Measures `faithfulness`, `context recall`, `answer relevancy`, and `precision` using `ragas` metrics.
- Evaluates how well the AI used the retrieved context in its response.

---

## 🧱 System Architecture

```text
Streamlit UI
│
├── Patient Import (CSV) ──> Batch Embedding ──┐
│                                             │
├── Q&A Import (CSV) ──────> Batch Embedding ──┘
│                                             ↓
├── PostgreSQL (pgvector) <── Vector Search ── AI Retrieval
│                                             ↓
├── AI SQL Agent ──> SQL ↔ Query Execution ↔ Data
│
└── RAG Answering ──> Groq LLaMA 3 70B
        ↑
     Context from QA pairs
```

---

## 📁 File Structure (Key Components)

| File / Folder      | Description |
|--------------------|-------------|
| `app.py`           | Main Streamlit app and backend logic |
| `healthcareapp.env`| Environment variables (Groq API key) |
| `medquad.csv`      | Medical QA pairs for RAG (pre-imported) |
| `patients.csv`     | Patient records for embedding |
| `developer.log`    | Debug logging output (configurable path) |

---

## 🛠 Technologies Used

- **Frontend**: Streamlit + custom dark theme styling
- **Database**: PostgreSQL 14+, `pgvector` + `pg_trgm` extensions
- **Embedding**: Groq API (`llama3-70b-8192`) or local fallback
- **LLM Integration**: Groq + OpenAI Python SDK
- **RAG Evaluation**: `ragas` + `datasets`
- **Embeddings**: `vector(384)` for patients, `vector(128)` for QA pairs
- **Logging & Debugging**: Python `logging`, configurable file path
- **Security**: `.env` file for API keys (not committed)

---
## 📦 Installation

1. **Clone the repo:**
   ```bash
   git clone https://github.com/yourusername/healthcare-vector-db.git
   cd healthcare-vector-db
2. **Install PostgreSQL with pgvector extension**
   POSTGRESQL:
   https://www.postgresql.org/download/
   PGVECTOR:
   https://github.com/pgvector/pgvector
3.**Make sure your patient data CSV has the following headers:**
   name,age,gender,history,last_visit,hemoglobin,wbc,platelets,bp_sys,bp_dia,heart_rate,temp
4. **Update the healthcareapp.env file with the specifications of the database and postgresql and your LlamaAI API key**
5. **Create a venv in the directory of the installation**
   ```bash
   cd "Your directory"
   python venv venv
   venv\Scripts\activate
   pip install -r requirements.txt
6. **Update path in run_healthcare_app.bat**
   cd /d "path to folder for healthcareapp"
7. **Update path to medquad.csv in app.py**
   cd /d "path to folder for healthcareapp"
8. **Run run_healthcare_app.bat**
After opening the app u can add patient data through the add patient tab or write data through import csv function in the left widget
Note: Using import csv function will append data into database and not delete it.

## 🔐 API Key Usage

The app relies on the **Groq LLaMA 3** API key. You can provide this:
- Via `.env` file (`LLAMA_API_KEY`)
- Or dynamically through the Streamlit sidebar input

> If no valid key is set, embedding and LLM Q&A will be disabled.

---

## 🧪 Example Queries You Can Try

- **Medical Q&A**:
  - *"What are the symptoms of anemia?"*
  - *"How do you treat hypertension?"*
- **Database-Aware AI Queries**:
  - *"Show average age of patients with asthma"*
  - *"How many female patients have diabetes?"*
  - *"List common conditions among patients over 60"*

---

## ✅ TODO / Enhancements

- [ ] Role-based access control
- [ ] Streamlit login state for key management
- [ ] Fine-tuned embedding model for patient features
- [ ] Frontend visualization for vector similarity scores

---

## 🧾 Disclaimer

This application is intended **for educational and experimental use only**. It does **not provide medical advice**. Always consult with licensed professionals for any healthcare concerns.


