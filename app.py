import streamlit as st
import re
import os
import fitz  # PyMuPDF
import faiss
import numpy as np
import json
from typing import Optional

from huggingface_hub import InferenceClient, HfApi
from sentence_transformers import SentenceTransformer

# -----------------------------
# Syllabus file helper (UI)
# -----------------------------
SYLLABUS_CANDIDATE_PATHS = [
    r"D:\Maitri\USC\Grader\knowledge_base\MKT 333 - Innovation Economics and Business - Beer AI and Video Games - Syllabus - 12-26-2025.pdf"
]

def load_syllabus_bytes():
    for p in SYLLABUS_CANDIDATE_PATHS:
        try:
            if os.path.exists(p):
                with open(p, "rb") as f:
                    return f.read(), os.path.basename(p)
        except Exception:
            continue
    return None, None

# -----------------------------
# Page config (UI)
# -----------------------------
st.set_page_config(
    page_title="MKT 333 — Beer AI & Video Games",
    page_icon="🍺",
    layout="centered",
    initial_sidebar_state="expanded",
)

# -----------------------------
# Hugging Face token
# -----------------------------
def get_hf_token() -> Optional[str]:
    try:
        tok = st.secrets.get("HF_TOKEN", None)
        if tok:
            return str(tok).strip()
    except Exception:
        pass
    tok = os.getenv("HF_TOKEN") or os.getenv("HUGGINGFACEHUB_API_TOKEN")
    return tok.strip() if tok else None

HF_TOKEN = get_hf_token()

def hf_token_ok(token: str) -> bool:
    try:
        HfApi().whoami(token=token)
        return True
    except Exception:
        return False

if not HF_TOKEN or not hf_token_ok(HF_TOKEN):
    st.error("Hugging Face token missing or invalid.")
    st.stop()

hf_client = InferenceClient(
    model="deepseek-ai/DeepSeek-R1-Distill-Qwen-1.5B",
    token=HF_TOKEN,
)

###########################################
# PDF Extraction and RAG Functions with Caching
###########################################

def extract_text_from_pdf(pdf_path):
    text = ""
    with fitz.open(pdf_path) as doc:
        for page in doc:
            raw_text = page.get_text()
            cleaned_text = re.sub(r"\n\s*\n+", "\n", raw_text)
            cleaned_text = re.sub(r"Page \d+", "", cleaned_text)
            text += cleaned_text + "\n"
    return text.strip()

def load_all_pdfs(folder_path):
    """Load all PDFs using cached JSON if available and up-to-date."""
    json_path = os.path.join(folder_path, "pdf_data.json")
    current_files = [f for f in os.listdir(folder_path) if f.lower().endswith(".pdf")]

    if os.path.exists(json_path):
        try:
            with open(json_path, "r") as f:
                saved_data = json.load(f)

            needs_refresh = False
            saved_files = {entry["filename"]: entry for entry in saved_data}

            current_set = set(current_files)
            saved_set = set(saved_files.keys())

            if current_set != saved_set:
                needs_refresh = True
            else:
                for filename in current_files:
                    file_path = os.path.join(folder_path, filename)
                    current_mtime = os.path.getmtime(file_path)
                    if saved_files[filename]["last_modified"] < current_mtime:
                        needs_refresh = True
                        break

            if not needs_refresh:
                return [{"filename": entry["filename"], "text": entry["text"]} for entry in saved_data]
        except (json.JSONDecodeError, KeyError):
            pass

    docs = []
    for filename in current_files:
        path = os.path.join(folder_path, filename)
        text = extract_text_from_pdf(path)
        docs.append(
            {
                "filename": filename,
                "text": text,
                "last_modified": os.path.getmtime(path),
            }
        )

    with open(json_path, "w") as f:
        json.dump(docs, f)

    return [{"filename": doc["filename"], "text": doc["text"]} for doc in docs]

def split_text(text, max_length=5000):
    sentences = text.split("\n")
    chunks = []
    current_chunk = ""
    for sentence in sentences:
        if len(current_chunk) + len(sentence) < max_length:
            current_chunk += sentence + "\n"
        else:
            chunks.append(current_chunk.strip())
            current_chunk = sentence + "\n"
    if current_chunk:
        chunks.append(current_chunk.strip())
    return chunks

@st.cache_resource
def get_embedder():
    return SentenceTransformer("BAAI/bge-small-en-v1.5")

model = get_embedder()

def build_vector_store(docs):
    all_chunks = []
    metadata = []
    for doc in docs:
        chunks = split_text(doc["text"])
        all_chunks.extend(chunks)
        metadata.extend([{"filename": doc["filename"]}] * len(chunks))
    embeddings = model.encode(all_chunks, convert_to_numpy=True)
    dimension = embeddings.shape[1]
    index = faiss.IndexFlatL2(dimension)
    index.add(embeddings)
    return index, all_chunks, metadata

def retrieve_context(query, index, chunks, metadatas, top_k=5):
    """Retrieve relevant context from the vector store, including [Source: filename] labels."""
    query_embedding = model.encode([query], convert_to_numpy=True)
    distances, indices = index.search(query_embedding, top_k)

    retrieved_blocks = []
    for i in indices[0]:
        fname = metadatas[i].get("filename", "unknown") if metadatas else "unknown"
        retrieved_blocks.append(f"[Source: {fname}]\n{chunks[i]}")
    return "\n\n---\n\n".join(retrieved_blocks)

###########################################
# UI Theme + Styling
###########################################

if "ui_dark_mode" not in st.session_state:
    st.session_state.ui_dark_mode = False

left, right = st.columns([0.97, 0.20], vertical_alignment="center")
with left:
    st.markdown(
        """
        <div class="top-banner">
          <div class="hero-title">Beer • AI • Video Games</div>
          <div class="hero-sub">Ask the course PDFs. Get clean, cited answers.</div>
        </div>
        """,
        unsafe_allow_html=True,
    )

# Theme vars
if st.session_state.ui_dark_mode:
    bg = "#0b0d12"
    text = "#e7eaf0"
    mut = "#a7b0c0"
    border = "rgba(231,234,240,0.12)"
    accent2 = "#ffcc00"
    user_bg = "rgba(30, 34, 46, 0.92)"
    input_bg = "rgba(12, 14, 22, 0.85)"

    # Sidebar cards like your screenshot
    sb_card = "rgba(15, 18, 28, 0.86)"
    sb_border = "rgba(231,234,240,0.12)"
    sb_btn_bg = "rgba(12, 14, 22, 0.75)"
    sb_btn_border = "rgba(231,234,240,0.10)"
    sb_badge_bg = "rgba(255,204,0,0.12)"
    sb_badge_border = "rgba(255,204,0,0.22)"
    sb_badge_text = "rgba(231,234,240,1)"
else:
    bg = "#fafafa"
    text = "#0b1220"
    mut = "#4b5563"
    border = "rgba(11,18,32,0.10)"
    accent2 = "#b38600"
    user_bg = "rgba(248,250,252,0.98)"
    input_bg = "rgba(255,255,255,0.98)"

    sb_card = "rgba(255,255,255,0.92)"
    sb_border = "rgba(11,18,32,0.10)"
    sb_btn_bg = "rgba(11,18,32,0.03)"
    sb_btn_border = "rgba(11,18,32,0.10)"
    sb_badge_bg = "rgba(179,134,0,0.10)"
    sb_badge_border = "rgba(179,134,0,0.18)"
    sb_badge_text = "rgba(11,18,32,0.85)"

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
  padding: 18px 18px;
  text-align: center;
  background: rgba(255,255,255,0.00);
}}
.hero-title {{
  margin-top: 8px;
  font-size: 1.70rem;
  font-weight: 900;
  letter-spacing: 0.2px;
}}
.hero-sub {{
  margin-top: 6px;
  font-size: 1.02rem;
  color: {mut};
}}

/* ===== Chat: remove "white box" look ===== */
/* Make message containers transparent by default */
.stChatMessage {{
  background: transparent !important;
  border: none !important;
  padding: 0.35rem 0 !important;
  margin: 0.65rem 0 !important;
  max-width: 100% !important;
}}
/* User bubble stays bubble */
[data-testid="stChatMessage"][aria-label="user"] {{
  background: {user_bg} !important;
  border: 1px solid {border} !important;
  border-radius: 18px !important;
  padding: 1.00rem 1.05rem !important;
  margin-left: auto !important;
  max-width: 88% !important;
}}
/* AI answer: no box, just clean text */
[data-testid="stChatMessage"][aria-label="AI"] {{
  background: transparent !important;
  border: none !important;
  padding: 0.1rem 0 !important;
  margin-right: auto !important;
  max-width: 96% !important;
}}
/* Keep text colors consistent */
[data-testid="stChatMessage"] * {{
  color: {text} !important;
}}
.reasoning, .reasoning * {{
  color: {mut} !important;
  font-style: italic;
}}
[data-testid="stChatMessage"] a {{
  color: {accent2} !important;
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
  font-size: 1.08rem !important;
  line-height: 1.45 !important;
  min-height: 72px !important;
  padding: 14px 16px !important;
}}
.stChatInput textarea::placeholder {{
  color: {mut} !important;
}}

/* Softer buttons */
.stButton button {{
  border-radius: 14px !important;
}}
hr {{
  opacity: 0.25;
}}

/* ===== Suggestions (your template vibe) ===== */
.sugg-head {{
  display:flex;
  align-items:flex-end;
  justify-content:space-between;
  margin-top: 0.8rem;
  margin-bottom: 0.55rem;
}}
.sugg-title {{
  font-weight: 900;
  font-size: 1.15rem;
}}
.sugg-sub {{
  color: {mut};
  font-size: 0.95rem;
  margin-top: 2px;
}}
.sugg-grid {{
  display:grid;
  grid-template-columns: repeat(3, 1fr);
  gap: 0.75rem;
}}
@media (max-width: 900px) {{
  .sugg-grid {{ grid-template-columns: 1fr; }}
}}
.sugg-card {{
  background: rgba(255,255,255,0.02);
  border: 1px solid {border};
  border-radius: 18px;
  padding: 14px 14px;
  min-height: 92px;
}}
.sugg-card-title {{
  font-weight: 850;
  font-size: 1.02rem;
  margin-bottom: 6px;
}}
.sugg-card-desc {{
  opacity: 0.80;
  font-size: 0.93rem;
  line-height: 1.35;
}}

/* ===== Sidebar quick links (like screenshot 2) ===== */
.sidebar-card {{
  background: {sb_card};
  border: 1px solid {sb_border};
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
.sidebar-badge {{
  display: inline-block;
  margin-left: 10px;
  padding: 3px 10px;
  border-radius: 999px;
  background: {sb_badge_bg};
  border: 1px solid {sb_badge_border};
  color: {sb_badge_text};
  font-size: 0.78rem;
  font-weight: 800;
}}
.sidebar-links a {{
  display: block;
  text-decoration: none;
  margin-top: 12px;
  padding: 16px 14px;
  border-radius: 14px;
  border: 1px solid {sb_btn_border};
  background: {sb_btn_bg};
  color: {text} !important;
  font-weight: 800;
}}
.sidebar-links a:hover {{
  border-color: rgba(255,204,0,0.35);
  box-shadow: 0 0 0 2px rgba(255,204,0,0.08);
}}
</style>
""",
    unsafe_allow_html=True,
)

###########################################
# Session State Initialization
###########################################

if "messages" not in st.session_state:
    st.session_state.messages = [{"role": "assistant", "content": "Hello! How can I assist you today? 🚀"}]
    st.session_state.model_config = {
        "temperature": 0.2,
        "top_p": 0.9,
        "max_tokens": 912,        # internal
        "repeat_penalty": 1.1,    # internal
    }
    st.session_state.show_thinking = True
    st.session_state.show_reasoning = True

# Show suggestions ONLY once per fresh session until the user interacts (click OR types)
if "show_suggestions" not in st.session_state:
    st.session_state.show_suggestions = True

# Load saved settings
if os.path.exists("settings.json"):
    with open("settings.json", "r") as f:
        saved_settings = json.load(f)
        st.session_state.show_thinking = saved_settings.get("show_thinking", st.session_state.show_thinking)
        st.session_state.show_reasoning = saved_settings.get("show_reasoning", st.session_state.show_reasoning)
        st.session_state.model_config["temperature"] = saved_settings.get(
            "temperature", st.session_state.model_config["temperature"]
        )
        st.session_state.model_config["max_tokens"] = saved_settings.get(
            "max_tokens", st.session_state.model_config["max_tokens"]
        )

# Initialize vector store
if "vector_index" not in st.session_state:
    pdf_folder = "./pdfs"
    if os.path.exists(pdf_folder):
        docs = load_all_pdfs(pdf_folder)
        vector_index, chunks, metadatas = build_vector_store(docs)
        st.session_state.vector_index = vector_index
        st.session_state.chunks = chunks
        st.session_state.metadatas = metadatas
    else:
        st.session_state.vector_index = None
        st.session_state.chunks = None
        st.session_state.metadatas = None

###########################################
# Sidebar (USC quick links back + minimal controls)
###########################################

with st.sidebar:
    st.markdown(
        """
        <div class="sidebar-card">
          <div class="sidebar-title">USC Links <span class="sidebar-badge">Quick</span></div>
          <div class="sidebar-sub">Open official pages in a new tab.</div>

          <div class="sidebar-links">
            <a href="https://www.usc.edu" target="_blank">USC — University of Southern California</a>
            <a href="https://gould.usc.edu/faculty/profile/d-daniel-sokol/" target="_blank">Professor D. Sokol</a>
            <a href="https://www.marshall.usc.edu" target="_blank">USC Marshall School of Business</a>
            <a href="https://www.marshall.usc.edu/departments/marketing" target="_blank">Marshall Marketing Department</a>
          </div>
        </div>
        """,
        unsafe_allow_html=True,
    )

    st.divider()

    st.markdown("### Controls")
    st.toggle("Thinking animation", key="show_thinking")
    st.toggle("Show reasoning", key="show_reasoning")

    with st.expander("Advanced", expanded=False):
        st.session_state.model_config["temperature"] = st.slider(
            "Creativity",
            0.0, 1.0,
            st.session_state.model_config["temperature"],
            0.1
        )
        st.session_state.model_config["top_p"] = st.slider(
            "Diversity",
            0.1, 1.0,
            st.session_state.model_config.get("top_p", 0.9),
            0.05
        )

        def save_settings():
            settings = {
                "show_thinking": st.session_state.show_thinking,
                "show_reasoning": st.session_state.show_reasoning,
                "temperature": st.session_state.model_config["temperature"],
                "max_tokens": st.session_state.model_config["max_tokens"],
            }
            with open("settings.json", "w") as f:
                json.dump(settings, f)
            st.success("Saved ✨")

        if st.button("Save", use_container_width=True):
            save_settings()

    # Keep your existing experimental PDF recalculation behavior
    def recalculate_pdf_data():
        pdf_folder = "./knowledge_base"
        pdf_files = [f for f in os.listdir(pdf_folder) if f.endswith(".pdf")]

        if not pdf_files:
            st.sidebar.error("No PDFs found in 'pdfs' folder.")
            return

        pdf_data = {"files": pdf_files}
        with open("pdf_data.json", "w") as json_file:
            json.dump(pdf_data, json_file, indent=4)

        st.sidebar.success("PDF data recalculated!")

    st.markdown("### Docs")
    if st.button("♻️ Recalculate PDF Data", use_container_width=True):
        with st.spinner("Processing PDFs..."):
            recalculate_pdf_data()

# Avatars
user_avatar = "👤"
ai_avatar = "🤖"

###########################################
# Chat Functions
###########################################

def parse_response(response):
    """Extract reasoning and content from response using <think> tags."""
    match = re.search(r"<think>(.*?)</think>(.*)", response, re.DOTALL)
    if match:
        return {"reasoning": match.group(1).strip(), "content": match.group(2).strip()}
    return {"reasoning": "", "content": response}

def display_response(parsed, placeholder):
    """Display response with optional reasoning."""
    final_display = []
    if st.session_state.show_reasoning and parsed["reasoning"]:
        final_display.append(f"<div class='reasoning'>🤔 {parsed['reasoning']}</div>")
    final_display.append(parsed["content"])
    placeholder.markdown("\n".join(final_display), unsafe_allow_html=True)

def sanitize_messages(msgs):
    """Remove extra keys (like reasoning) before sending to HF."""
    cleaned = []
    for m in msgs:
        if not isinstance(m, dict):
            continue
        role = m.get("role")
        content = m.get("content", "")
        if role in {"system", "user", "assistant"}:
            cleaned.append({"role": role, "content": str(content)})
    return cleaned

def build_system_prompt(retrieved_context: str) -> str:
    # More structured outputs (heading / subheadings / bullets)
    return f"""
You are a course assistant for MKT 333 (Beer • AI • Video Games).

Hard rules:
- Use ONLY the retrieved context as the factual source. If the context doesn’t support a claim, say: "I don’t have enough information in the documents."
- Cite evidence using the labels in the context, like: [Source: filename.pdf]
- Do NOT invent citations.
- Write in clean Markdown.

Output structure (use this exact skeleton; depth can vary):
# <Short Title>

## 1) Executive takeaway
- 2–4 bullets max.

## 2) Key points
### What it means
- bullets

### Why it matters (business lens)
- bullets

## 3) How to apply
- Step-by-step bullets or short numbered steps.

## 4) Quick examples (if helpful)
- bullets

## 5) Evidence from the PDFs
- Bullet each claim + citation like: [Source: ...]

If the user asks a simple definition, keep sections short but still use headings.

Retrieved context:
{retrieved_context}
""".strip()

def generate_response():
    """Generate and display AI response with RAG context."""
    user_prompt = st.session_state.messages[-1]["content"]

    retrieved_context = ""
    if (
        st.session_state.vector_index is not None
        and st.session_state.chunks is not None
        and st.session_state.metadatas is not None
    ):
        retrieved_context = retrieve_context(
            user_prompt,
            st.session_state.vector_index,
            st.session_state.chunks,
            st.session_state.metadatas,
        )

    system_prompt = build_system_prompt(retrieved_context)

    with st.chat_message("assistant", avatar=ai_avatar):
        response_placeholder = st.empty()

        if st.session_state.show_thinking:
            response_placeholder.markdown(
                """
                <div style="display: flex; align-items: center; gap: 0.5rem">
                    <div class="typing-animation">
                        <div class="dot"></div>
                        <div class="dot"></div>
                        <div class="dot"></div>
                    </div>
                </div>
                """,
                unsafe_allow_html=True,
            )

        try:
            response = hf_client.chat.completions.create(
                messages=sanitize_messages([
                    {"role": "system", "content": system_prompt},
                    *st.session_state.messages[-6:],
                ]),
                max_tokens=st.session_state.model_config["max_tokens"],
                temperature=st.session_state.model_config["temperature"],
                top_p=st.session_state.model_config["top_p"],
            )

            assistant_text = response.choices[0].message.content or ""
            if not assistant_text.strip():
                assistant_text = "I don’t have enough information in the documents to answer that question."

        except Exception:
            assistant_text = "I’m having trouble reaching the model right now. Please try again in a moment."

        # Simulated streaming
        full_response = ""
        for token in assistant_text.split():
            full_response += token + " "
            cursor = "▌" if not st.session_state.show_thinking else ""
            response_placeholder.markdown(full_response + cursor)

        parsed = parse_response(full_response)
        st.session_state.messages.append(
            {
                "role": "assistant",
                "content": parsed["content"],
                "reasoning": parsed["reasoning"],
            }
        )

        display_response(parsed, response_placeholder)

###########################################
# One-time Suggestions (Recommended only)
###########################################

SUGGESTIONS = [
    {
        "title": "Marketing Basics: STP",
        "desc": "Segmentation → Targeting → Positioning with citations.",
        "prompt": "Using the PDFs, explain STP (segmentation, targeting, positioning) and apply it to Beer/AI/Video Games. Include [Source: ...] citations."
    },
    {
        "title": "GTM Plan (30 days)",
        "desc": "Channels, experiments, KPIs.",
        "prompt": "Create a 30-day go-to-market plan based on the PDFs: channels, weekly experiments, messaging, KPIs, and risks. Include [Source: ...] citations."
    },
    {
        "title": "BD Partnerships",
        "desc": "Targets + deal structures.",
        "prompt": "Suggest 8 partnership targets and propose 3 deal structures (rev share/license/co-sell). Add a partner outreach email. Use PDFs + citations."
    },
    {
        "title": "Positioning & Differentiation",
        "desc": "Sharp positioning + defensible moat.",
        "prompt": "Using the PDFs, write a positioning statement, 3 differentiators, and a competitor counter-positioning plan. Include [Source: ...] citations."
    },
    {
        "title": "Pricing & Monetization",
        "desc": "Models + how to test pricing.",
        "prompt": "Using the PDFs, propose 3 pricing models, when each wins, and a pricing experiment plan (hypothesis, test design, metrics). Include [Source: ...] citations."
    },
    {
        "title": "Advanced: Growth Experiments",
        "desc": "Hypothesis-driven growth loops.",
        "prompt": "Using the PDFs, propose 6 growth experiments (hypothesis, channel, steps, success metric, risk). Include [Source: ...] citations."
    },
    {
        "title": "Segmentation & ICP",
        "desc": "Find best-fit customers + what to say.",
        "prompt": "Using the PDFs, define 3 customer segments + ICP, each segment’s pain points, and tailored messaging. Include [Source: ...] citations."
    },
    {
        "title": "Sales Pitch & Objections",
        "desc": "Pitch + discovery + objections.",
        "prompt": "Create a 30-second pitch, 2-minute pitch, 10 discovery questions, and 6 objections with responses using the PDFs. Include [Source: ...] citations."
    },
]

def run_suggestion(prompt: str):
    st.session_state.show_suggestions = False
    st.session_state.messages.append({"role": "user", "content": prompt})
    generate_response()

###########################################
# Chat History Display
###########################################

for i, message in enumerate(st.session_state.messages):
    role = "user" if message["role"] == "user" else "AI"
    with st.chat_message(role, avatar=user_avatar if role == "user" else ai_avatar):
        reasoning = message.get("reasoning", "")
        content = message.get("content", "")

        # Only the FIRST assistant message gets the hello bubble
        if role == "AI" and i == 0:
            content = f"<div class='hello-bubble'>{content}</div>"

        if st.session_state.show_reasoning and reasoning:
            st.markdown(f"<div class='reasoning'>🤔 {reasoning}</div>{content}", unsafe_allow_html=True)
        else:
            st.markdown(content, unsafe_allow_html=True)


# Suggestions: show once (after greeting), disappear forever after first click OR typing
if st.session_state.show_suggestions and len(st.session_state.messages) <= 1:
    st.markdown(
        """
        <div class="sugg-head">
          <div>
            <div class="sugg-title">Recommended</div>
            <div class="sugg-sub">Tap one to get a guided, cited answer.</div>
          </div>
        </div>
        """,
        unsafe_allow_html=True,
    )

    cols = st.columns(3)
    for idx, item in enumerate(SUGGESTIONS):
        with cols[idx % 3]:
            st.markdown(
                f"""
                <div class="sugg-card">
                  <div class="sugg-card-title">{item["title"]}</div>
                  <div class="sugg-card-desc">{item["desc"]}</div>
                </div>
                """,
                unsafe_allow_html=True,
            )
            if st.button("Open", key=f"sugg_{idx}", use_container_width=True):
                run_suggestion(item["prompt"])

###########################################
# Regenerate flow (kept)
###########################################

if hasattr(st.session_state, "regenerate") and st.session_state.regenerate:
    if st.session_state.messages and st.session_state.messages[-1]["role"] == "user":
        generate_response()
    st.session_state.regenerate = False

###########################################
# User Input Handling
###########################################

if prompt := st.chat_input("Type your message..."):
    st.session_state.show_suggestions = False

    st.session_state.messages.append({"role": "user", "content": prompt})
    with st.chat_message("user", avatar=user_avatar):
        st.markdown(prompt)
    generate_response()

# Regenerate button (kept)
row_l, row_r = st.columns([0.78, 0.22], vertical_alignment="center")
with row_l:
    st.markdown("###")
with row_r:
    last_is_ai = len(st.session_state.messages) > 0 and st.session_state.messages[-1]["role"] == "assistant"
    if st.button("Regenerate", disabled=not last_is_ai, use_container_width=True):
        if st.session_state.messages and st.session_state.messages[-1]["role"] == "assistant":
            st.session_state.messages.pop()
            st.session_state.regenerate = True
            st.rerun()

