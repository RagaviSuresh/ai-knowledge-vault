# Vault AI - Personal Knowledge Vault

Vault AI is a modern, premium **AI Personal Knowledge Vault** web application. It allows users to securely upload documents, automatically extract contents, generate AI summaries and keyword tags, and interact with their private documents through a localized **Retrieval-Augmented Generation (RAG) Search Assistant**.

---

## Key Features

1. **Dashboard Analytics**: Real-time trackers for document counts, total vault size, processing queues, and primary topics.
2. **Asynchronous Processing**: Uploaded documents are saved, queued, and processed in the background (extracting text, generating summaries and tags, and updating the index).
3. **Advanced Local RAG Search**: Ask questions across your vault. The search engine breaks documents into paragraphs, ranks passages using local token frequency overlap, and synthesizes a citation-linked answer.
4. **Document Explorer**: Search, filter, inspect details (AI summaries, tags, size), or delete documents from the vault.
5. **SQLite Persistence**: All vault index states, summaries, and document metadata are stored in a local SQLite file (`backend/vault.db`).

---

## Project Structure

```text
ai-knowledge-vault/
├── backend/
│   ├── uploads/            # Location of raw uploaded documents
│   ├── venv/               # Python virtual environment
│   ├── ai.py               # Extraction, Summarization, and RAG search engine
│   ├── crud.py             # SQLite CRUD helper operations
│   ├── database.py         # SQLAlchemy & SQLite setup
│   ├── main.py             # FastAPI backend server & router endpoints
│   ├── models.py           # SQLAlchemy Document schema model
│   ├── schemas.py          # Pydantic schemas for request/response validation
│   └── requirements.txt    # Python library requirements
│
├── frontend/
│   ├── src/
│   │   ├── App.tsx         # Main React UI component (Dashboard, Search, Settings)
│   │   ├── index.css       # Custom design system styling (Glassmorphism, animations)
│   │   └── main.tsx        # React mounting entry point
│   ├── index.html          # Web application page template
│   ├── package.json        # Frontend NPM script definitions & dependencies
│   └── vite.config.ts      # Vite bundler options
│
└── README.md               # Master setup instructions (this file)
```

---

## Setup & Running Instructions

Follow these steps to run both the FastAPI backend and Vite/React frontend concurrently.

### 1. Backend Server Setup

Navigate into the `backend/` directory, set up your Python environment, and start the FastAPI server:

1. Open a terminal and navigate to the project root directory.
2. Ensure you have Python 3.10+ installed.
3. Install dependencies using the pre-configured virtual environment:
   ```bash
   backend/venv/Scripts/pip.exe install -r backend/requirements.txt
   ```
4. Start the FastAPI server by running the main module through the virtual environment's python runner:
   ```bash
   backend/venv/Scripts/python.exe backend/main.py
   ```
   *The backend will boot up at `http://127.0.0.1:8000` with hot-reloading enabled.*

### 2. Frontend React Setup

Open a separate terminal window to run the Vite dev server for the React app:

1. Install Node.js packages (including `lucide-react` icons):
   ```bash
   npm.cmd install
   ```
2. Launch the Vite development server:
   ```bash
   npm.cmd run dev
   ```
   *The React development server will start at `http://localhost:5173`.*

---

## Technical Details

- **File Support**: Supports Text (`.txt`), Markdown (`.md`), PDF (`.pdf`), JSON (`.json`), and CSV (`.csv`) formats.
- **Local Indexing**:
  - **Summary**: Extractive scoring selects high-scoring paragraphs based on TF-IDF term weights.
  - **Tags**: Noun-frequency filtration identifies the top 5 characteristic tags.
  - **RAG queries**: Text contents are broken down into overlapping paragraph blocks, scored against query tokens, and sorted to compile relevant context with exact citations.
