# Prism RAG 

**Prism** is an intelligent, multimodal Retrieval-Augmented Generation (RAG) system designed to provide deep insights across a unified knowledge base. It allows users to query documents (PDF, DOCX, Excel), images, audio, and video using natural language, leveraging state-of-the-art local LLMs and computer vision models.

---

## 🚀 Key Features

*   **Multimodal Ingestion**: 
    *   **Documents**: PDF, Word (DOCX), Excel (XLSX), PowerPoint (PPTX), Text, Markdown.
    *   **Images**: Intelligent analysis using OCR (PaddleOCR) and Visual LLMs (LLaVA/BakLLaVA).
    *   **Audio**: Speech-to-text transcription using Whisper.
    *   **Video**: Frame extraction and analysis.
*   **Advanced RAG Pipeline**:
    *   **Hybrid Search**: Combines **Semantic Vector Search** (FAISS) with **Keyword Rescue** strategies to ensure no critical data (like specific names or ID numbers) is missed.
    *   **Reranking**: Uses Cross-Encoders (`ms-marco`) to refine and prioritize the top results for maximum relevance.
    *   **Folder Scoping**: Filter searches to specific projects or folders for targeted queries.
*   **Local Privacy**: Powered entirely by local models (Ollama), ensuring data privacy and security (no data leaves your infrastructure).
*   **Intelligent UI**: A modern, responsive React frontend with real-time progress tracking, file management, and chat history.

---

## 🛠️ Technology Stack

### Backend
*   **Language**: Python 3.10+
*   **API Framework**: FastAPI (Async, High-performance)
*   **LLM Engine**: [Ollama](https://ollama.com/) (Managing LLaMA 3.2, LLaVA)
*   **Vector Store**: FAISS (Facebook AI Similarity Search) - CPU Optimized
*   **Embeddings**: `nomic-embed-text` / `sentence-transformers`
*   **Reranker**: `cross-encoder/ms-marco-Minilm-L-6-v2`
*   **Audio Processing**: `pywhispercpp` (Whisper)
*   **OCR**: PaddleOCR (High accuracy for tables and text in images)
*   **Document Parsing**: `pdfplumber`, `python-docx`, `openpyxl`, `python-pptx`

### Frontend
*   **Framework**: React (Vite)
*   **Styling**: TailwindCSS (v4)
*   **Components**: Headless UI
*   **State/Network**: Axios, React Router DOM
*   **Visuals**: Lucide React (Icons), Framer Motion (Animations)

---

## 🧠 Models Used

| Component | Model | Description |
| :--- | :--- | :--- |
| **LLM (Text)** | `llama3.2` | Robust instruction-following model for reasoning and answer generation. |
| **LLM (Vision)** | `llava` | Visual understanding for describing images and video frames. |
| **Embeddings** | `nomic-embed-text` | High-quality text embeddings for semantic search. |
| **Reranker** | `ms-marco-Minilm-L-6-v2` | Cross-encoder for re-scoring retrieved chunks. |
| **Audio** | `whisper-base/small` | OpenAI's Whisper for robust speech transcription. |
| **OCR** | `PaddleOCR` | Optical Character Recognition for structure retention (tables etc.). |

---

## 📦 Installation & Setup

### Prerequisites
1.  **Python**: Version 3.10 or higher.
2.  **Node.js**: Version 18+ (for Frontend).
3.  **Ollama**: Installed and running [Download Here](https://ollama.com/).

### 1. Setup Models (Ollama)
Ensure Ollama is running and pull the required models:
```bash
ollama pull llama3.2
ollama pull llava
ollama pull nomic-embed-text
```

### 2. Backend Setup
Navigate to the `backend` directory:
```bash
cd backend
```

Create a virtual environment:
```bash
python -m venv .venv
source .venv/bin/activate  # On Windows: .venv\Scripts\activate
```

Install dependencies:
```bash
pip install -r requirements.txt
```
*(Note: If you encounter issues with `torch` or `paddlepaddle`, ensure you have the correct version for your hardware/OS).*

Run the backend server:
```bash
python -m uvicorn app.main:app --reload --host 0.0.0.0 --port 8000
```
The API will be available at `http://localhost:8000`.

### 3. Frontend Setup
Navigate to the `frontend` directory:
```bash
cd frontend
```

Install dependencies:
```bash
npm install
```

Start the development server:
```bash
npm run dev
```
The UI will be available at `http://localhost:3000`.

---

## 🔍 Architecture Concepts

### 1. Ingestion Pipeline
When a file is uploaded:
1.  **Detection**: The file type is identified.
2.  **Parsing/Extraction**: 
    *   **PDFs**: Text is extracted page-by-page. Images within PDFs are OCR'd.
    *   **Office Docs**: Parsed natively (`docx`, `pptx`, `xlsx`).
    *   **Media**: Audio is transcribed; Video frames are sampled and described.
3.  **Chunking**: Content is split into manageable chunks (recursive character splitting).
4.  **Embedding**: Text chunks are converted to vectors using the embedding model.
5.  **Indexing**: Vectors are stored in the FAISS index for fast retrieval.

### 2. Retrieval Pipeline (RAG)
When a user asks a question:
1.  **Query Analysis**: The query is cleaned and key terms are extracted.
2.  **Vector Search**: The top `k` semantically similar chunks are retrieved from FAISS.
3.  **Hybrid Filtering**:
    *   **Folder Filter**: Results are narrowed to the selected folder (if any).
    *   **Keyword Rescue**: Chunks containing exact matches of critical terms (e.g., "EPI", "2024") are "rescued" and prioritized, even if their vector score is low.
4.  **Reranking**: A cross-encoder model scores the candidate chunks against the question to pick the absolute best context.
5.  **Generation**: The top chunks are fed to `llama3.2` as context to generate a grounded, accurate answer.

---

## 🛡️ Audit & Logs
All interactions are logged for quality assurance:
*   `data/logs/rag_audit_log.jsonl`: Detailed JSON logs of every query, including retrieval stats, chunks used, and generation time.
