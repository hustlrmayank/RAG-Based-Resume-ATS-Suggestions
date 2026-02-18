# 📄 Resume ATS Analyzer

A **LangChain RAG-powered** Streamlit application that analyzes resumes and provides expert-level, structured feedback — just like a senior hiring manager would.

---

## ✨ Features

- 📤 **PDF Upload** — drag-and-drop or browse to upload any resume PDF
- 🤖 **RAG Pipeline** — uses FAISS vector store + HuggingFace embeddings for semantic retrieval
- 🧠 **Gemini 2.5 Flash** — powered by Google's latest LLM for high-quality feedback
- 🎯 **5 Analysis Modes** — choose from preset queries or write your own
- 🌑 **Polished Dark UI** — professional Streamlit interface with custom styling

---

## 🖥️ Demo

| Upload Resume | Get Analysis |
|---|---|
| Upload any `.pdf` resume | Receive structured strengths, weaknesses, skill gaps & suggestions |

---

## 🚀 Getting Started

### 1. Clone the Repository

```bash
git clone https://github.com/your-username/resume-ats-analyzer.git
cd resume-ats-analyzer
```

### 2. Install Dependencies

```bash
pip install -r requirements.txt
```

### 3. Set Up Environment Variables

Create a `.env` file in the project root:

```env
GOOGLE_API_KEY=your_google_gemini_api_key_here
```

> **Get your API key:** [Google AI Studio](https://aistudio.google.com/app/apikey)

### 4. Run the App

```bash
streamlit run resume_ats_app.py
```

The app will open at `http://localhost:8501`

---

## 📦 Requirements

Create a `requirements.txt` with the following:

```txt
streamlit
langchain
langchain-google-genai
langchain-huggingface
langchain-community
langchain-text-splitters
faiss-cpu
pymupdf
sentence-transformers
python-dotenv
```

Or install manually:

```bash
pip install streamlit langchain langchain-google-genai langchain-huggingface \
            langchain-community langchain-text-splitters faiss-cpu pymupdf \
            sentence-transformers python-dotenv
```

---

## 🧰 Tech Stack

| Layer | Technology |
|---|---|
| **Frontend** | Streamlit (custom CSS) |
| **LLM** | Google Gemini 2.5 Flash via `langchain-google-genai` |
| **Embeddings** | `sentence-transformers/all-MiniLM-L6-v2` (HuggingFace) |
| **Vector Store** | FAISS (in-memory) |
| **PDF Loader** | PyMuPDF via `langchain-community` |
| **RAG Framework** | LangChain (retrieval + stuff chain) |

---

## 🔍 How It Works

```
PDF Upload
    │
    ▼
PyMuPDF Loader  ──►  Document Pages
    │
    ▼
RecursiveCharacterTextSplitter
(chunk_size=800, overlap=100)
    │
    ▼
HuggingFace Embeddings
(all-MiniLM-L6-v2)
    │
    ▼
FAISS Vector Store  ──►  Retriever (top-3 chunks)
    │
    ▼
Stuff Documents Chain  +  ChatPromptTemplate
    │
    ▼
Gemini 2.5 Flash  ──►  Structured Analysis
```

1. The uploaded PDF is parsed and split into overlapping text chunks.
2. Each chunk is embedded using a HuggingFace sentence transformer.
3. Chunks are indexed in a FAISS vector store for semantic search.
4. When a query is submitted, the top-3 most relevant chunks are retrieved.
5. These chunks + the query are passed to Gemini via a structured prompt.
6. The model returns a detailed analysis with strengths, weaknesses, skill gaps, and suggestions.

---

## 🎯 Analysis Modes

| Mode | Description |
|---|---|
| **Full Analysis** | Complete review with all four sections |
| **ATS Score & Keywords** | ATS friendliness rating + missing keywords |
| **Strengths Only** | Detailed breakdown of candidate strengths |
| **Improvement Suggestions** | Actionable, specific recommendations |
| **Tailored for Data Science** | Fit assessment for DS/ML roles |
| **Custom Question** | Ask anything about the resume |

---

## 📁 Project Structure

```
resume-ats-analyzer/
│
├── resume_ats_app.py       # Main Streamlit application
├── Resume_ats.ipynb        # Original Jupyter notebook (prototype)
├── requirements.txt        # Python dependencies
├── .env                    # API keys (not committed to git)
├── .gitignore
└── README.md
```

---

## 🔐 Environment Variables

| Variable | Description | Required |
|---|---|---|
| `GOOGLE_API_KEY` | Google Gemini API key | ✅ Yes |

> If you don't have a `.env` file, the app will prompt you to enter the API key directly in the sidebar.

---

## 🛡️ .gitignore

Make sure to add the following to your `.gitignore`:

```gitignore
.env
__pycache__/
*.pyc
*.pdf
.streamlit/
```

---

## 🤝 Contributing

Pull requests are welcome! For major changes, please open an issue first to discuss what you'd like to change.

1. Fork the repository
2. Create your feature branch: `git checkout -b feature/my-feature`
3. Commit your changes: `git commit -m 'Add some feature'`
4. Push to the branch: `git push origin feature/my-feature`
5. Open a pull request

---

## 📝 License

This project is licensed under the MIT License — see the [LICENSE](LICENSE) file for details.

---

## 🙏 Acknowledgements

- [LangChain](https://www.langchain.com/) — RAG framework
- [Google Gemini](https://deepmind.google/technologies/gemini/) — LLM backbone
- [HuggingFace](https://huggingface.co/) — Embedding models
- [FAISS](https://github.com/facebookresearch/faiss) — Vector similarity search
- [Streamlit](https://streamlit.io/) — Web app framework
