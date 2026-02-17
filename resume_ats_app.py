import os
import tempfile
import streamlit as st
from dotenv import load_dotenv

# Load environment variables
load_dotenv()

# ── Page config ──────────────────────────────────────────────────────────────
st.set_page_config(
    page_title="Resume ATS Analyzer",
    page_icon="📄",
    layout="wide",
    initial_sidebar_state="collapsed",
)

# ── Custom CSS ────────────────────────────────────────────────────────────────
st.markdown("""
<style>
@import url('https://fonts.googleapis.com/css2?family=Playfair+Display:wght@700;900&family=DM+Sans:wght@300;400;500&display=swap');

:root {
    --bg:       #0d0f14;
    --surface:  #161a23;
    --border:   #252b38;
    --accent:   #e8c84a;
    --accent2:  #4ae8b4;
    --text:     #e8eaf0;
    --muted:    #7a8296;
}

html, body, [data-testid="stAppViewContainer"] {
    background: var(--bg) !important;
    color: var(--text) !important;
    font-family: 'DM Sans', sans-serif;
}

[data-testid="stHeader"] { background: transparent !important; }

/* Hero */
.hero {
    text-align: center;
    padding: 3rem 1rem 2rem;
}
.hero h1 {
    font-family: 'Playfair Display', serif;
    font-size: clamp(2.4rem, 5vw, 4rem);
    font-weight: 900;
    color: var(--text);
    letter-spacing: -1px;
    margin: 0;
    line-height: 1.1;
}
.hero h1 span { color: var(--accent); }
.hero p {
    color: var(--muted);
    font-size: 1.05rem;
    margin-top: .8rem;
    font-weight: 300;
}

/* Divider */
.divider {
    height: 1px;
    background: linear-gradient(90deg, transparent, var(--border), transparent);
    margin: 0 auto 2rem;
    max-width: 700px;
}

/* Upload zone */
.upload-zone {
    border: 2px dashed var(--border);
    border-radius: 16px;
    padding: 2.5rem;
    text-align: center;
    background: var(--surface);
    transition: border-color .3s;
}

/* Card */
.card {
    background: var(--surface);
    border: 1px solid var(--border);
    border-radius: 14px;
    padding: 1.6rem 1.8rem;
    margin-bottom: 1rem;
}
.card-title {
    font-family: 'Playfair Display', serif;
    font-size: 1.15rem;
    font-weight: 700;
    color: var(--accent);
    margin-bottom: .6rem;
    display: flex;
    align-items: center;
    gap: .5rem;
}

/* Streamlit overrides */
[data-testid="stFileUploader"] {
    background: var(--surface) !important;
    border: 2px dashed var(--border) !important;
    border-radius: 14px !important;
    padding: 1rem !important;
}
[data-testid="stFileUploader"] label { color: var(--muted) !important; }

div[data-testid="stTextArea"] textarea {
    background: var(--surface) !important;
    color: var(--text) !important;
    border: 1px solid var(--border) !important;
    border-radius: 10px !important;
    font-family: 'DM Sans', sans-serif !important;
}

div.stButton > button {
    background: var(--accent) !important;
    color: #0d0f14 !important;
    font-weight: 700 !important;
    border: none !important;
    border-radius: 10px !important;
    padding: .75rem 2.5rem !important;
    font-size: 1rem !important;
    letter-spacing: .5px !important;
    transition: opacity .2s !important;
    width: 100%;
}
div.stButton > button:hover { opacity: .85 !important; }

.stSpinner > div { border-top-color: var(--accent) !important; }

/* Selectbox */
div[data-testid="stSelectbox"] select,
div[data-testid="stSelectbox"] > div {
    background: var(--surface) !important;
    color: var(--text) !important;
    border-color: var(--border) !important;
}

/* Status pills */
.pill {
    display: inline-block;
    padding: .2rem .75rem;
    border-radius: 20px;
    font-size: .78rem;
    font-weight: 500;
    margin-right: .4rem;
    margin-bottom: .4rem;
}
.pill-green  { background: #1a3a2a; color: var(--accent2); border: 1px solid #2a5a3a; }
.pill-yellow { background: #3a2e0a; color: var(--accent);  border: 1px solid #5a4a10; }
.pill-red    { background: #3a1a1a; color: #e86060;        border: 1px solid #5a2a2a; }

/* Score ring (CSS only) */
.score-wrap {
    display: flex;
    align-items: center;
    justify-content: center;
    gap: 2rem;
    padding: 1.5rem 0;
}
.score-ring {
    width: 110px; height: 110px;
    border-radius: 50%;
    display: flex; align-items: center; justify-content: center;
    font-family: 'Playfair Display', serif;
    font-size: 2rem; font-weight: 900;
}
.score-label { font-size: .85rem; color: var(--muted); margin-top: .3rem; text-align:center; }

/* Answer markdown */
.answer-block { color: var(--text); line-height: 1.75; }
.answer-block h3 { font-family: 'Playfair Display', serif; color: var(--accent); margin-top: 1.2rem; }
.answer-block ul  { padding-left: 1.4rem; }
.answer-block li  { margin-bottom: .4rem; }

/* Footer */
.footer {
    text-align: center;
    color: var(--muted);
    font-size: .8rem;
    padding: 2rem 0 1rem;
    border-top: 1px solid var(--border);
    margin-top: 3rem;
}
</style>
""", unsafe_allow_html=True)


# ── Helper: build RAG chain ───────────────────────────────────────────────────
@st.cache_resource(show_spinner=False)
def get_llm():
    from langchain_google_genai import ChatGoogleGenerativeAI
    api_key = os.environ.get("GOOGLE_API_KEY", "")
    return ChatGoogleGenerativeAI(
        model="gemini-2.5-flash",
        google_api_key=api_key,
        temperature=0.7,
    )

def build_chain(pdf_path: str):
    from langchain_community.document_loaders import PyMuPDFLoader
    from langchain_text_splitters import RecursiveCharacterTextSplitter
    from langchain_huggingface import HuggingFaceEmbeddings
    from langchain_community.vectorstores import FAISS
    from langchain_classic.chains.retrieval import create_retrieval_chain
    from langchain_classic.chains.combine_documents.stuff import create_stuff_documents_chain
    from langchain_core.prompts import ChatPromptTemplate

    docs = PyMuPDFLoader(pdf_path).load()
    chunks = RecursiveCharacterTextSplitter(chunk_size=800, chunk_overlap=100).split_documents(docs)
    embeddings = HuggingFaceEmbeddings(model_name="sentence-transformers/all-MiniLM-L6-v2")
    vectorstore = FAISS.from_documents(chunks, embeddings)
    retriever = vectorstore.as_retriever(search_type="similarity", search_kwargs={"k": 3})

    prompt = ChatPromptTemplate.from_template("""
You are a senior hiring manager reviewing a candidate's resume.

Context:
{context}

Question:
{input}

Provide structured output with clear sections:
1. Key Strengths
2. Weaknesses
3. Skill Gaps
4. Specific Improvement Suggestions
""")
    doc_chain = create_stuff_documents_chain(get_llm(), prompt)
    return create_retrieval_chain(retriever, doc_chain), len(docs)


QUERIES = {
    "Full Analysis": "Analyze this resume thoroughly and suggest improvements.",
    "ATS Score & Keywords": "Rate this resume's ATS friendliness out of 10 and list the top missing keywords for a software/data engineering role.",
    "Strengths Only": "List all the key strengths of this candidate in detail.",
    "Improvement Suggestions": "Give me specific, actionable improvement suggestions for this resume.",
    "Tailored for Data Science": "How well does this resume fit a Data Science / ML role? What should be added or changed?",
    "Custom Question": None,
}


# ── UI ────────────────────────────────────────────────────────────────────────
st.markdown("""
<div class="hero">
    <h1>Resume <span>ATS</span> Analyzer</h1>
    <p>Upload your résumé — get expert-level feedback powered by LangChain RAG & Gemini.</p>
</div>
<div class="divider"></div>
""", unsafe_allow_html=True)

col_left, col_right = st.columns([1, 1.15], gap="large")

# ── LEFT: Upload + options ────────────────────────────────────────────────────
with col_left:
    st.markdown('<div class="card"><div class="card-title">📎 Upload Résumé (PDF)</div>', unsafe_allow_html=True)
    uploaded = st.file_uploader("", type=["pdf"], label_visibility="collapsed")
    st.markdown('</div>', unsafe_allow_html=True)

    if uploaded:
        st.markdown(f'<span class="pill pill-green">✓ {uploaded.name}</span>', unsafe_allow_html=True)

    st.markdown('<div class="card"><div class="card-title">🔍 Analysis Mode</div>', unsafe_allow_html=True)
    mode = st.selectbox("", list(QUERIES.keys()), label_visibility="collapsed")

    custom_q = ""
    if mode == "Custom Question":
        custom_q = st.text_area("Your question:", placeholder="e.g. Is this resume suitable for a backend engineering role at a startup?", height=100)

    st.markdown('</div>', unsafe_allow_html=True)

    api_key_input = ""
    if not os.environ.get("GOOGLE_API_KEY"):
        st.markdown('<div class="card"><div class="card-title">🔑 Google API Key</div>', unsafe_allow_html=True)
        api_key_input = st.text_input("", type="password", placeholder="Paste your Gemini API key here", label_visibility="collapsed")
        st.markdown('</div>', unsafe_allow_html=True)

    analyze_btn = st.button("⚡ Analyze Resume", use_container_width=True)


# ── RIGHT: Results ────────────────────────────────────────────────────────────
with col_right:
    st.markdown('<div class="card" style="min-height:420px;">', unsafe_allow_html=True)

    if not uploaded:
        st.markdown("""
        <div style="display:flex;flex-direction:column;align-items:center;justify-content:center;height:360px;color:#7a8296;gap:1rem;">
            <div style="font-size:3.5rem;">📄</div>
            <div style="font-size:1rem;">Your analysis will appear here</div>
            <div style="font-size:.85rem;opacity:.6;">Upload a PDF and click Analyze</div>
        </div>
        """, unsafe_allow_html=True)

    elif analyze_btn:
        # Set API key if provided manually
        if api_key_input:
            os.environ["GOOGLE_API_KEY"] = api_key_input

        if not os.environ.get("GOOGLE_API_KEY"):
            st.error("⚠️ Please provide a Google API Key.")
        else:
            question = custom_q if mode == "Custom Question" else QUERIES[mode]
            if not question:
                st.warning("Please enter your custom question.")
            else:
                with st.spinner("🔍 Analyzing your résumé…"):
                    try:
                        # Save uploaded PDF to temp file
                        with tempfile.NamedTemporaryFile(delete=False, suffix=".pdf") as tmp:
                            tmp.write(uploaded.read())
                            tmp_path = tmp.name

                        chain, pages = build_chain(tmp_path)
                        response = chain.invoke({"input": question})
                        answer = response["answer"]
                        os.unlink(tmp_path)

                        # Display
                        st.markdown(f'<span class="pill pill-green">✓ Analyzed {pages} page(s)</span>', unsafe_allow_html=True)
                        st.markdown(f'<span class="pill pill-yellow">{mode}</span>', unsafe_allow_html=True)
                        st.markdown("<br>", unsafe_allow_html=True)
                        st.markdown(f'<div class="answer-block">', unsafe_allow_html=True)
                        st.markdown(answer)
                        st.markdown('</div>', unsafe_allow_html=True)

                    except Exception as e:
                        st.error(f"❌ Error: {str(e)}")
                        st.info("Make sure you have all dependencies installed:\n```\npip install langchain langchain-google-genai langchain-huggingface langchain-community faiss-cpu pymupdf sentence-transformers python-dotenv\n```")
    else:
        st.markdown("""
        <div style="display:flex;flex-direction:column;align-items:center;justify-content:center;height:360px;color:#7a8296;gap:1rem;">
            <div style="font-size:3.5rem;">✨</div>
            <div style="font-size:1rem;">Ready to analyze</div>
            <div style="font-size:.85rem;opacity:.6;">Click "Analyze Resume" to get started</div>
        </div>
        """, unsafe_allow_html=True)

    st.markdown('</div>', unsafe_allow_html=True)


# ── Footer ────────────────────────────────────────────────────────────────────
st.markdown("""
<div class="footer">
    Built with LangChain RAG · Gemini 2.5 Flash · FAISS · HuggingFace Embeddings · Streamlit
</div>
""", unsafe_allow_html=True)
