import os
import re
import json
from typing import Optional, List, Dict, Tuple

import streamlit as st
import fitz  # PyMuPDF
import faiss
import numpy as np

from huggingface_hub import InferenceClient, HfApi
from sentence_transformers import SentenceTransformer


# =============================
# Config
# =============================
PDF_FOLDER = "./pdfs"
CACHE_JSON_NAME = "pdf_data.json"
EMBED_MODEL_NAME = "BAAI/bge-small-en-v1.5"
HF_MODEL_NAME = "deepseek-ai/DeepSeek-R1-Distill-Qwen-1.5B"


# =============================
# Hugging Face Token
# =============================
def get_hf_token() -> Optional[str]:
    try:
        tok = st.secrets.get("HF_TOKEN", None)
        if tok:
            return str(tok).strip()
    except Exception:
        pass

    tok = os.getenv("HF_TOKEN") or os.getenv("HUGGINGFACEHUB_API_TOKEN")
    return tok.strip() if tok else None


def hf_token_ok(token: str) -> bool:
    try:
        HfApi().whoami(token=token)
        return True
    except Exception:
        return False


# =============================
# PDF Extraction + Caching
# =============================
def extract_text_from_pdf(pdf_path: str) -> str:
    """Extract text from a PDF with light cleanup."""
    text_parts: List[str] = []
    with fitz.open(pdf_path) as doc:
        for page in doc:
            raw = page.get_text()
            raw = re.sub(r"\n\s*\n+", "\n", raw)
            raw = re.sub(r"Page\s+\d+\s*", "", raw, flags=re.IGNORECASE)
            text_parts.append(raw.strip())
    return "\n".join([t for t in text_parts if t]).strip()


def load_all_pdfs(folder_path: str) -> List[Dict[str, str]]:
    """
    Loads all PDFs. Uses a json cache that invalidates when:
    - file list changes
    - any file mtime increases
    """
    if not os.path.exists(folder_path):
        return []

    pdf_files = sorted([f for f in os.listdir(folder_path) if f.lower().endswith(".pdf")])
    if not pdf_files:
        return []

    json_path = os.path.join(folder_path, CACHE_JSON_NAME)

    # Try cache first
    if os.path.exists(json_path):
        try:
            with open(json_path, "r", encoding="utf-8") as f:
                saved = json.load(f)

            saved_map = {x["filename"]: x for x in saved}
            if set(saved_map.keys()) == set(pdf_files):
                cache_ok = True
                for fn in pdf_files:
                    p = os.path.join(folder_path, fn)
                    if saved_map[fn]["last_modified"] < os.path.getmtime(p):
                        cache_ok = False
                        break

                if cache_ok:
                    return [{"filename": x["filename"], "text": x["text"]} for x in saved]
        except Exception:
            pass

    # Rebuild cache
    docs = []
    for fn in pdf_files:
        p = os.path.join(folder_path, fn)
        docs.append(
            {
                "filename": fn,
                "text": extract_text_from_pdf(p),
                "last_modified": os.path.getmtime(p),
            }
        )

    with open(json_path, "w", encoding="utf-8") as f:
        json.dump(docs, f)

    return [{"filename": x["filename"], "text": x["text"]} for x in docs]


# =============================
# Chunking + Embeddings + FAISS (Cosine)
# =============================
def chunk_text(text: str, chunk_size: int = 1200, overlap: int = 150) -> List[str]:
    """
    Simple char-based chunking with overlap.
    Smaller chunks + overlap improves hit-rate for specific questions.
    """
    text = text.strip()
    if not text:
        return []

    chunks = []
    start = 0
    n = len(text)

    while start < n:
        end = min(start + chunk_size, n)
        chunk = text[start:end].strip()
        if chunk:
            chunks.append(chunk)
        start = end - overlap
        if start < 0:
            start = 0
        if end == n:
            break

    return chunks


@st.cache_resource
def get_embedder() -> SentenceTransformer:
    return SentenceTransformer(EMBED_MODEL_NAME)


def _normalize(v: np.ndarray) -> np.ndarray:
    norms = np.linalg.norm(v, axis=1, keepdims=True) + 1e-12
    return v / norms


def build_vector_store(docs: List[Dict[str, str]]) -> Tuple[faiss.Index, List[str], List[Dict[str, str]]]:
    """
    Build cosine-similarity FAISS index:
    - embed chunks
    - normalize embeddings
    - IndexFlatIP (inner product) == cosine after normalization
    """
    all_chunks: List[str] = []
    metadatas: List[Dict[str, str]] = []

    for d in docs:
        chunks = chunk_text(d["text"], chunk_size=1200, overlap=150)
        all_chunks.extend(chunks)
        metadatas.extend([{"filename": d["filename"]}] * len(chunks))

    if not all_chunks:
        # Create a dummy index to avoid crashes (won't retrieve anything)
        index = faiss.IndexFlatIP(384)  # bge-small is 384 dims
        return index, [], []

    embedder = get_embedder()
    embs = embedder.encode(all_chunks, convert_to_numpy=True, show_progress_bar=False)
    embs = _normalize(embs).astype("float32")

    dim = embs.shape[1]
    index = faiss.IndexFlatIP(dim)
    index.add(embs)

    return index, all_chunks, metadatas


def retrieve_context(
    query: str,
    index: faiss.Index,
    chunks: List[str],
    metadatas: List[Dict[str, str]],
    top_k: int = 6,
    min_score: float = 0.25,
) -> str:
    """
    Retrieves top_k chunks. Uses a minimum cosine score threshold to reduce garbage context.
    """
    if not chunks:
        return ""

    embedder = get_embedder()
    q = embedder.encode([query], convert_to_numpy=True, show_progress_bar=False)
    q = _normalize(q).astype("float32")

    scores, idxs = index.search(q, top_k)

    blocks: List[str] = []
    for score, i in zip(scores[0], idxs[0]):
        if i < 0 or i >= len(chunks):
            continue
        if float(score) < min_score:
            continue
        fname = metadatas[i].get("filename", "unknown") if metadatas else "unknown"
        blocks.append(f"[Source: {fname} | score={float(score):.2f}]\n{chunks[i]}")

    return "\n\n---\n\n".join(blocks).strip()


# =============================
# LLM Helpers
# =============================
def parse_response(text: str) -> Dict[str, str]:
    """
    DeepSeek R1 models often wrap chain-of-thought in <think>.
    We remove it from the user-visible answer.
    """
    m = re.search(r"<think>(.*?)</think>(.*)", text, re.DOTALL)
    if m:
        return {"reasoning": m.group(1).strip(), "content": m.group(2).strip()}
    return {"reasoning": "", "content": text.strip()}


def build_system_prompt(retrieved_context: str) -> str:
    return f"""
You are a course assistant for MKT 333 (Beer • AI • Video Games).

Rules:
- Use ONLY the retrieved context as the factual source.
- If the context does not contain the answer, say exactly:
  "I don’t have enough information in the documents to answer that question."
- Always cite with: [Source: filename.pdf]
- Write clean Markdown with spacing.

Formatting requirements (IMPORTANT):
- Use headings (##) and bullet points
- Leave a blank line between sections
- Bold key terms using **bold**
- Keep paragraphs short (2–3 lines max)

Response template:
## Answer
- 3–6 bullets max

## Key details
- bullets (bold key terms)

## Evidence
- bullet each claim + citation(s)

Retrieved context:
{retrieved_context}
""".strip()


# =============================
# Streamlit UI
# =============================
st.set_page_config(
    page_title="MKT 333 — Beer AI & Video Games",
    page_icon="🍺",
    layout="centered",
    initial_sidebar_state="expanded",
)

HF_TOKEN = get_hf_token()
if not HF_TOKEN or not hf_token_ok(HF_TOKEN):
    st.error("Hugging Face token missing or invalid. Add HF_TOKEN to secrets or env.")
    st.stop()

hf_client = InferenceClient(model=HF_MODEL_NAME, token=HF_TOKEN)

if "ui_dark_mode" not in st.session_state:
    st.session_state.ui_dark_mode = False

# Theme colors
if st.session_state.ui_dark_mode:
    bg = "#0b0d12"
    text = "#e7eaf0"
    mut = "#a7b0c0"
    border = "rgba(231,234,240,0.12)"
    accent = "#ffcc00"
    user_bg = "rgba(30, 34, 46, 0.92)"
    input_bg = "rgba(12, 14, 22, 0.85)"
    sb_card = "rgba(15, 18, 28, 0.86)"
else:
    bg = "#fafafa"
    text = "#0b1220"
    mut = "#4b5563"
    border = "rgba(11,18,32,0.10)"
    accent = "#b38600"
    user_bg = "rgba(248,250,252,0.98)"
    input_bg = "rgba(255,255,255,0.98)"
    sb_card = "rgba(255,255,255,0.92)"

# Header
st.markdown(
    f"""
<style>
.stApp {{
  background: {bg};
  color: {text};
}}
.block-container {{
  padding-top: 1.0rem;
  max-width: 980px;
}}

.top-banner {{
  border: 1px solid {border};
  border-radius: 18px;
  padding: 18px;
  text-align: center;
  background: transparent;
}}
.hero-title {{
  font-size: 1.70rem;
  font-weight: 900;
}}
.hero-sub {{
  margin-top: 6px;
  font-size: 1.02rem;
  color: {mut};
}}

/* =========================================
   CHAT FIX: Remove assistant white box
   ========================================= */

/* Default chat message container: transparent */
.stChatMessage {{
  background: transparent !important;
  border: none !important;
  padding: 0.30rem 0 !important;
  margin: 0.65rem 0 !important;
  max-width: 100% !important;
}}

/* USER bubble: keep nice box */
[data-testid="stChatMessage"][aria-label="user"] {{
  background: {user_bg} !important;
  border: 1px solid {border} !important;
  border-radius: 18px !important;
  padding: 1.00rem 1.05rem !important;
  margin-left: auto !important;
  max-width: 88% !important;
}}

/* AI message outer container: transparent */
[data-testid="stChatMessage"][aria-label="AI"] {{
  background: transparent !important;
  border: none !important;
  padding: 0.10rem 0 !important;
  margin-right: auto !important;
  max-width: 96% !important;
}}

/* AI inner content container ALSO transparent (this is the “white box” culprit) */
[data-testid="stChatMessage"][aria-label="AI"] [data-testid="stChatMessageContent"] {{
  background: transparent !important;
  border: none !important;
  padding: 0 !important;
}}

/* AI typography: readable + structured */
[data-testid="stChatMessage"][aria-label="AI"] [data-testid="stChatMessageContent"] {{
  font-size: 0.98rem !important;
  line-height: 1.65 !important;
}}
[data-testid="stChatMessage"][aria-label="AI"] h1,
[data-testid="stChatMessage"][aria-label="AI"] h2,
[data-testid="stChatMessage"][aria-label="AI"] h3 {{
  margin: 0.55rem 0 0.35rem 0 !important;
}}
[data-testid="stChatMessage"] a {{
  color: {accent} !important;
}}
.reasoning, .reasoning * {{
  color: {mut} !important;
  font-style: italic;
}}

/* Chat input */
.stChatInput {{
  border-top: 1px solid {border};
  background: transparent;
}}
.stChatInput textarea {{
  background: {input_bg} !important;
  color: {text} !important;
  border-radius: 16px !important;
  border: 1px solid {border} !important;
  font-size: 1.05rem !important;
  line-height: 1.45 !important;
  min-height: 72px !important;
  padding: 14px 16px !important;
}}
.stChatInput textarea::placeholder {{
  color: {mut} !important;
}}

/* Sidebar "quick link cards" like your screenshot */
.sidebar-card {{
  background: {sb_card};
  border: 1px solid {border};
  border-radius: 18px;
  padding: 16px;
}}
.sidebar-title {{
  font-weight: 900;
  font-size: 1.05rem;
  margin: 0;
}}
.sidebar-sub {{
  margin-top: 8px;
  color: {mut};
  font-size: 0.95rem;
}}
.sidebar-link {{
  display: block;
  text-decoration: none;
  margin-top: 12px;
  padding: 16px 14px;
  border-radius: 14px;
  border: 1px solid {border};
  background: rgba(0,0,0,0.03);
  color: {text} !important;
  font-weight: 800;
}}
.sidebar-link:hover {{
  border-color: rgba(255,204,0,0.35);
  box-shadow: 0 0 0 2px rgba(255,204,0,0.08);
}}
</style>
""",
    unsafe_allow_html=True,
)

st.markdown(
    """
    <div class="top-banner">
      <div class="hero-title">Beer • AI • Video Games</div>
      <div class="hero-sub">Ask the course PDFs. Get clean, cited answers.</div>
    </div>
    """,
    unsafe_allow_html=True,
)

# =============================
# Session state
# =============================
if "messages" not in st.session_state:
    st.session_state.messages = [{"role": "assistant", "content": "Hi! Ask me anything from the course PDFs."}]
if "model_config" not in st.session_state:
    st.session_state.model_config = {"temperature": 0.2, "top_p": 0.9, "max_tokens": 900}
if "show_reasoning" not in st.session_state:
    st.session_state.show_reasoning = False  # default OFF (cleaner)


# =============================
# Vector store init / reload
# =============================
def initialize_vector_store() -> None:
    docs = load_all_pdfs(PDF_FOLDER)
    if not docs:
        st.session_state.vector_index = None
        st.session_state.chunks = None
        st.session_state.metadatas = None
        return

    index, chunks, metas = build_vector_store(docs)
    st.session_state.vector_index = index
    st.session_state.chunks = chunks
    st.session_state.metadatas = metas


if "vector_index" not in st.session_state:
    initialize_vector_store()


# =============================
# Sidebar
# =============================
with st.sidebar:
    st.markdown(
        """
        <div class="sidebar-card">
          <div class="sidebar-title">USC Links</div>
          <div class="sidebar-sub">Open official pages in a new tab.</div>

          <a class="sidebar-link" href="https://www.usc.edu" target="_blank">USC — University of Southern California</a>
          <a class="sidebar-link" href="https://gould.usc.edu/faculty/profile/d-daniel-sokol/" target="_blank">Professor D. Sokol</a>
          <a class="sidebar-link" href="https://www.marshall.usc.edu" target="_blank">USC Marshall School of Business</a>
          <a class="sidebar-link" href="https://www.marshall.usc.edu/departments/marketing" target="_blank">Marshall Marketing Department</a>
        </div>
        """,
        unsafe_allow_html=True,
    )

    st.divider()
    st.markdown("### Controls")
    st.toggle("Show reasoning", key="show_reasoning")

    if st.button("♻️ Reload PDFs", use_container_width=True):
        with st.spinner("Reloading PDFs..."):
            initialize_vector_store()
        st.success("Reloaded!")

    with st.expander("Model"):
        st.session_state.model_config["temperature"] = st.slider(
            "Creativity", 0.0, 1.0, float(st.session_state.model_config["temperature"]), 0.05
        )
        st.session_state.model_config["top_p"] = st.slider(
            "Diversity", 0.1, 1.0, float(st.session_state.model_config["top_p"]), 0.05
        )


# =============================
# Chat
# =============================
def sanitize_messages(msgs: List[Dict[str, str]]) -> List[Dict[str, str]]:
    cleaned = []
    for m in msgs:
        if not isinstance(m, dict):
            continue
        role = m.get("role")
        content = m.get("content", "")
        if role in {"system", "user", "assistant"}:
            cleaned.append({"role": role, "content": str(content)})
    return cleaned


def generate_response(user_prompt: str) -> None:
    index = st.session_state.get("vector_index")
    chunks = st.session_state.get("chunks")
    metas = st.session_state.get("metadatas")

    retrieved_context = ""
    if index is not None and chunks is not None and metas is not None:
        retrieved_context = retrieve_context(
            user_prompt,
            index,
            chunks,
            metas,
            top_k=6,
            min_score=0.25,
        )

    system_prompt = build_system_prompt(retrieved_context)

    with st.chat_message("assistant", avatar="🤖"):
        placeholder = st.empty()

        try:
            resp = hf_client.chat.completions.create(
                messages=sanitize_messages(
                    [{"role": "system", "content": system_prompt}, *st.session_state.messages[-6:], {"role": "user", "content": user_prompt}]
                ),
                max_tokens=int(st.session_state.model_config["max_tokens"]),
                temperature=float(st.session_state.model_config["temperature"]),
                top_p=float(st.session_state.model_config["top_p"]),
            )

            assistant_text = resp.choices[0].message.content or ""
            if not assistant_text.strip():
                assistant_text = "I don’t have enough information in the documents to answer that question."
        except Exception:
            assistant_text = "I’m having trouble reaching the model right now. Please try again."

        parsed = parse_response(assistant_text)

        # Show reasoning optionally (clean UI by default)
        if st.session_state.show_reasoning and parsed["reasoning"]:
            placeholder.markdown(f"<div class='reasoning'>🤔 {parsed['reasoning']}</div>\n\n{parsed['content']}", unsafe_allow_html=True)
        else:
            placeholder.markdown(parsed["content"])

        st.session_state.messages.append({"role": "assistant", "content": parsed["content"], "reasoning": parsed["reasoning"]})


# Render history
for m in st.session_state.messages:
    role = m["role"]
    avatar = "👤" if role == "user" else "🤖"
    with st.chat_message(role, avatar=avatar):
        if role == "assistant" and st.session_state.show_reasoning and m.get("reasoning"):
            st.markdown(f"<div class='reasoning'>🤔 {m['reasoning']}</div>\n\n{m['content']}", unsafe_allow_html=True)
        else:
            st.markdown(m["content"])

# Input
if prompt := st.chat_input("Type your message..."):
    st.session_state.messages.append({"role": "user", "content": prompt})
    with st.chat_message("user", avatar="👤"):
        st.markdown(prompt)
    generate_response(prompt)
