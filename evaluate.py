import os
import re
import json
import time
from io import StringIO
from typing import List, Dict, Tuple, Optional

import streamlit as st
import fitz
import faiss
import numpy as np
from huggingface_hub import InferenceClient, HfApi
from sentence_transformers import SentenceTransformer


# =============================
# Config (match app.py)
# =============================
PDF_FOLDER = "./pdfs"
CACHE_JSON_NAME = "pdf_data.json"
EMBED_MODEL_NAME = "BAAI/bge-small-en-v1.5"
HF_MODEL_NAME = "deepseek-ai/DeepSeek-R1-Distill-Qwen-1.5B"


# =============================
# HF token
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
# PDF + caching
# =============================
def extract_text_from_pdf(pdf_path: str) -> str:
    parts = []
    with fitz.open(pdf_path) as doc:
        for page in doc:
            raw = page.get_text()
            raw = re.sub(r"\n\s*\n+", "\n", raw)
            raw = re.sub(r"Page\s+\d+\s*", "", raw, flags=re.IGNORECASE)
            parts.append(raw.strip())
    return "\n".join([p for p in parts if p]).strip()


def load_all_pdfs(folder_path: str) -> List[Dict[str, str]]:
    if not os.path.exists(folder_path):
        return []

    pdf_files = sorted([f for f in os.listdir(folder_path) if f.lower().endswith(".pdf")])
    if not pdf_files:
        return []

    json_path = os.path.join(folder_path, CACHE_JSON_NAME)

    if os.path.exists(json_path):
        try:
            with open(json_path, "r", encoding="utf-8") as f:
                saved = json.load(f)
            saved_map = {x["filename"]: x for x in saved}
            if set(saved_map.keys()) == set(pdf_files):
                ok = True
                for fn in pdf_files:
                    p = os.path.join(folder_path, fn)
                    if saved_map[fn]["last_modified"] < os.path.getmtime(p):
                        ok = False
                        break
                if ok:
                    return [{"filename": x["filename"], "text": x["text"]} for x in saved]
        except Exception:
            pass

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
# Chunking + cosine FAISS
# =============================
def chunk_text(text: str, chunk_size: int = 1200, overlap: int = 150) -> List[str]:
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
    return v / (np.linalg.norm(v, axis=1, keepdims=True) + 1e-12)


def build_vector_store(docs: List[Dict[str, str]]) -> Tuple[faiss.Index, List[str], List[Dict[str, str]]]:
    all_chunks = []
    metas = []
    for d in docs:
        cs = chunk_text(d["text"])
        all_chunks.extend(cs)
        metas.extend([{"filename": d["filename"]}] * len(cs))

    if not all_chunks:
        index = faiss.IndexFlatIP(384)
        return index, [], []

    embedder = get_embedder()
    embs = embedder.encode(all_chunks, convert_to_numpy=True, show_progress_bar=False)
    embs = _normalize(embs).astype("float32")

    index = faiss.IndexFlatIP(embs.shape[1])
    index.add(embs)
    return index, all_chunks, metas


def retrieve_context(
    query: str,
    index: faiss.Index,
    chunks: List[str],
    metas: List[Dict[str, str]],
    top_k: int = 6,
    min_score: float = 0.25,
) -> Tuple[str, List[Dict[str, str]]]:
    if not chunks:
        return "", []

    embedder = get_embedder()
    q = embedder.encode([query], convert_to_numpy=True, show_progress_bar=False)
    q = _normalize(q).astype("float32")

    scores, idxs = index.search(q, top_k)

    blocks = []
    hits = []
    for score, i in zip(scores[0], idxs[0]):
        if i < 0 or i >= len(chunks):
            continue
        if float(score) < min_score:
            continue
        fname = metas[i].get("filename", "unknown") if metas else "unknown"
        blocks.append(f"[Source: {fname} | score={float(score):.2f}]\n{chunks[i]}")
        hits.append({"filename": fname, "score": float(score)})

    return "\n\n---\n\n".join(blocks).strip(), hits


# =============================
# LLM prompt (evaluation)
# =============================
def build_system_prompt(retrieved_context: str) -> str:
    return f"""
You are evaluating a RAG chatbot.

Rules:
- Answer ONLY using the retrieved context.
- If the context does not contain the answer, say exactly:
  "I don’t have enough information in the documents to answer that question."
- Keep the answer concise and in bullet points when possible.
- Cite evidence like: [Source: filename.pdf]

Retrieved context:
{retrieved_context}
""".strip()


# =============================
# Test set (you can replace)
# =============================
TEST_SET = [
    # Replace with your MKT333 questions + expected answers
    {"query": "Why did Carlsberg care so much about packaging?", "expected": "Packaging mattered because ... (from your PDFs)."},
]


# =============================
# Evaluation
# =============================
def cosine_sim(a: np.ndarray, b: np.ndarray) -> float:
    return float(np.dot(a, b) / ((np.linalg.norm(a) * np.linalg.norm(b)) + 1e-12))


def evaluate(test_set, hf_client, index, chunks, metas, sim_threshold=0.82):
    embedder = get_embedder()
    results = []

    for item in test_set:
        q = item["query"]
        expected = item["expected"]

        ctx, hits = retrieve_context(q, index, chunks, metas)

        system_prompt = build_system_prompt(ctx)
        messages = [{"role": "system", "content": system_prompt}, {"role": "user", "content": q}]

        try:
            resp = hf_client.chat.completions.create(
                messages=messages,
                max_tokens=600,
                temperature=0.2,
                top_p=0.9,
            )
            answer = resp.choices[0].message.content or ""
        except Exception as e:
            answer = f"[ERROR calling model] {e}"

        # Similarity (rough metric)
        emb_exp = embedder.encode([expected], convert_to_numpy=True)[0]
        emb_ans = embedder.encode([answer], convert_to_numpy=True)[0]
        sim = cosine_sim(emb_exp, emb_ans)

        results.append(
            {
                "query": q,
                "expected": expected,
                "response": answer,
                "similarity": sim,
                "pass_similarity": sim >= sim_threshold,
                "retrieval_hits": hits,
                "ctx_chars": len(ctx),
            }
        )

    return results


def generate_report(results: List[Dict]) -> str:
    out = StringIO()
    out.write("=== RAG Bot Evaluation Report ===\n")
    out.write(f"Date: {time.strftime('%Y-%m-%d %H:%M:%S')}\n")
    out.write(f"Total Queries: {len(results)}\n")

    pass_count = sum(1 for r in results if r["pass_similarity"])
    out.write(f"Pass(similarity): {pass_count}/{len(results)}\n\n")

    for i, r in enumerate(results, 1):
        out.write(f"--- Query {i} ---\n")
        out.write(f"Q: {r['query']}\n")
        out.write(f"Expected: {r['expected']}\n")
        out.write(f"Response: {r['response']}\n")
        out.write(f"Similarity: {r['similarity']:.3f} | Pass: {r['pass_similarity']}\n")
        out.write(f"Retrieved context chars: {r['ctx_chars']}\n")
        out.write("Top retrieval hits:\n")
        for h in r["retrieval_hits"][:5]:
            out.write(f"  - {h['filename']} (score={h['score']:.2f})\n")
        out.write("\n")

    return out.getvalue()


# =============================
# Streamlit UI
# =============================
st.set_page_config(page_title="RAG Bot Evaluation", initial_sidebar_state="expanded")

HF_TOKEN = get_hf_token()
if not HF_TOKEN or not hf_token_ok(HF_TOKEN):
    st.error("HF_TOKEN missing/invalid. Add it to secrets or env to run evaluation.")
    st.stop()

hf_client = InferenceClient(model=HF_MODEL_NAME, token=HF_TOKEN)

st.title("RAG Bot Evaluation")

if "vector_index" not in st.session_state:
    docs = load_all_pdfs(PDF_FOLDER)
    idx, ch, mt = build_vector_store(docs) if docs else (None, None, None)
    st.session_state.vector_index = idx
    st.session_state.chunks = ch
    st.session_state.metadatas = mt

with st.sidebar:
    st.header("Controls")
    if st.button("♻️ Reload PDFs"):
        docs = load_all_pdfs(PDF_FOLDER)
        idx, ch, mt = build_vector_store(docs) if docs else (None, None, None)
        st.session_state.vector_index = idx
        st.session_state.chunks = ch
        st.session_state.metadatas = mt
        st.success("Reloaded PDFs + rebuilt index.")

    sim_threshold = st.slider("Similarity threshold", 0.50, 0.95, 0.82, 0.01)

    if st.button("📊 Run Evaluation"):
        idx = st.session_state.vector_index
        ch = st.session_state.chunks
        mt = st.session_state.metadatas

        if idx is None or ch is None or mt is None:
            st.warning("No PDFs / index loaded. Put PDFs in ./pdfs first.")
        else:
            with st.spinner("Running evaluation..."):
                results = evaluate(TEST_SET, hf_client, idx, ch, mt, sim_threshold=sim_threshold)
                report = generate_report(results)

            st.text_area("Report", report, height=350)
            st.download_button(
                "📥 Download Report",
                data=report,
                file_name=f"rag_eval_report_{time.strftime('%Y%m%d_%H%M%S')}.txt",
                mime="text/plain",
            )

st.info(
    "Note: Similarity scoring is only a rough metric. For RAG bots, you’ll get better evaluation by checking citations and whether retrieved context contains the answer."
)
