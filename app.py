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

if st.session_state.ui_dark_mode:
    bg = "#0b0d12"
    panel = "rgba(15, 18, 28, 0.86)"
    text = "#e7eaf0"
    mut = "#a7b0c0"
    border = "rgba(231,234,240,0.12)"
    accent2 = "#ffcc00"
    user_bg = "rgba(30, 34, 46, 0.92)"
    ai_bg = "rgba(153, 0, 0, 0.22)"
    input_bg = "rgba(12, 14, 22, 0.85)"
else:
    bg = "#fafafa"
    panel = "rgba(255,255,255,0.92)"
    text = "#0b1220"
    mut = "#4b5563"
    border = "rgba(11,18,32,0.10)"
    accent2 = "#b38600"
    user_bg = "rgba(248,250,252,0.98)"
    ai_bg = "rgba(153, 0, 0, 0.10)"
    input_bg = "rgba(255,255,255,0.98)"

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
  background: {panel};
  border: 1px solid {border};
  border-radius: 18px;
  padding: 18px 18px;
  text-align: center;
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
.stChatMessage {{
  padding: 1.05rem 1.10rem;
  border-radius: 22px;
  margin: 0.80rem 0;
  max-width: 88%;
  border: 1px solid {border};
  background: {panel};
}}
[data-testid="stChatMessage"][aria-label="user"] {{
  background: {user_bg};
  margin-left: auto;
}}
[data-testid="stChatMessage"][aria-label="AI"] {{
  background: {ai_bg};
  margin-right: auto;
}}
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
/* Cleaner dividers */
hr {{
  opacity: 0.25;
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
        "max_tokens": 912,        # kept internally; removed from UI
        "repeat_penalty": 1.1,    # kept internally
    }
    st.session_state.show_thinking = True
    st.session_state.show_reasoning = True

if "mode" not in st.session_state:
    st.session_state.mode = "Marketing"

# Load saved settings (keeps existing behavior; max_tokens stays internal)
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

# Initialize vector store with cached PDF loading
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
# Sidebar (Gen Z subtle, no tokens, no navigation section)
###########################################

with st.sidebar:
    st.markdown("### ⚡ Controls")
    st.caption("Pick a mode. Tap a card. Get a strategy + citations.")

    st.session_state.mode = st.selectbox(
        "Mode",
        ["Marketing", "Sales", "Business Development", "Product Strategy"],
        index=["Marketing", "Sales", "Business Development", "Product Strategy"].index(st.session_state.mode)
        if st.session_state.mode in ["Marketing", "Sales", "Business Development", "Product Strategy"] else 0
    )

    st.divider()

    st.toggle("Thinking animation", key="show_thinking")
    st.toggle("Show reasoning", key="show_reasoning")

    st.divider()

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
                "max_tokens": st.session_state.model_config["max_tokens"],  # internal
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

def build_system_prompt(retrieved_context: str, mode: str) -> str:
    mode_focus = {
        "Marketing": "positioning, segmentation, brand strategy, channels, pricing, experiments, messaging",
        "Sales": "discovery, qualification (BANT/MEDDICC), objection handling, pitch, ROI framing, closing",
        "Business Development": "partners, alliances, deal structures, integrations, co-marketing, distribution",
        "Product Strategy": "JTBD, differentiation, roadmap, growth loops, activation, retention"
    }.get(mode, "marketing strategy")

    return f"""
You are a {mode} analyst for MKT 333 (Beer • AI • Video Games).
Focus areas: {mode_focus}

Rules:
- Use ONLY the retrieved context as the factual source. If the context doesn’t support a claim, say: "I don’t have enough information in the documents."
- Cite evidence using the labels in the context, like: [Source: filename.pdf]
- Be concrete, actionable, and business-oriented (plans, scripts, metrics).
- If the user asks for a plan, provide steps + metrics + risks + assumptions.

Retrieved context:
{retrieved_context}

Response format (always):
1) Executive takeaway (1–2 sentences)
2) Evidence (bullets with citations)
3) Strategy / Recommendation (what to do next)
4) Script (pitch/email/talking points appropriate to the selected mode)
5) Metrics to track (3–6)
6) Risks & assumptions
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

    system_prompt = build_system_prompt(retrieved_context, st.session_state.mode)

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

        # Simulated streaming (UI)
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
# Quick Cards (Recent / Frequent / Recommended)
###########################################

if "recent_prompts" not in st.session_state:
    st.session_state.recent_prompts = []  # newest first

if "prompt_counts" not in st.session_state:
    st.session_state.prompt_counts = {}  # prompt -> count

def _track_prompt(prompt: str):
    p = prompt.strip()
    if not p:
        return
    st.session_state.recent_prompts = [x for x in st.session_state.recent_prompts if x != p]
    st.session_state.recent_prompts.insert(0, p)
    st.session_state.recent_prompts = st.session_state.recent_prompts[:6]
    st.session_state.prompt_counts[p] = st.session_state.prompt_counts.get(p, 0) + 1

def _run_prompt(prompt: str):
    _track_prompt(prompt)
    st.session_state.messages.append({"role": "user", "content": prompt})
    generate_response()

def get_recent_cards():
    return st.session_state.recent_prompts[:6]

def get_frequent_cards():
    items = sorted(st.session_state.prompt_counts.items(), key=lambda x: x[1], reverse=True)
    return [p for p, c in items[:6]]

RECOMMENDED = [
    {
        "title": "Marketing Basics: STP",
        "desc": "Segmentation → Targeting → Positioning with citations.",
        "prompt": "Using the PDFs, explain STP (segmentation, targeting, positioning) and apply it to Beer/AI/Video Games. Include [Source: ...] citations."
    },
    {
        "title": "Positioning & Differentiation",
        "desc": "Sharp positioning + defensible moat.",
        "prompt": "Using the PDFs, write a positioning statement, 3 differentiators, and a competitor counter-positioning plan. Include [Source: ...] citations."
    },
    {
        "title": "Segmentation & ICP",
        "desc": "Find best-fit customers + what to say.",
        "prompt": "Using the PDFs, define 3 customer segments + ICP, each segment’s pain points, and tailored messaging. Include [Source: ...] citations."
    },
    {
        "title": "GTM Plan (30 days)",
        "desc": "Channels, experiments, KPIs.",
        "prompt": "Create a 30-day go-to-market plan based on the PDFs: channels, weekly experiments, messaging, KPIs, and risks. Include [Source: ...] citations."
    },
    {
        "title": "Pricing & Monetization",
        "desc": "Models + how to test pricing.",
        "prompt": "Using the PDFs, propose 3 pricing models, when each wins, and a pricing experiment plan (hypothesis, test design, metrics). Include [Source: ...] citations."
    },
    {
        "title": "Sales Pitch & Objections",
        "desc": "Pitch + discovery + objections.",
        "prompt": "Create a 30-second pitch, 2-minute pitch, 10 discovery questions, and 6 objections with responses using the PDFs. Include [Source: ...] citations."
    },
    {
        "title": "BD Partnerships",
        "desc": "Targets + deal structures.",
        "prompt": "Suggest 8 partnership targets and propose 3 deal structures (rev share/license/co-sell). Add a partner outreach email. Use PDFs + citations."
    },
    {
        "title": "Advanced: Growth Experiments",
        "desc": "Hypothesis-driven growth loops.",
        "prompt": "Using the PDFs, propose 6 growth experiments (hypothesis, channel, steps, success metric, risk). Include [Source: ...] citations."
    },
]

st.markdown(
    """
    <style>
    .qgrid { display:grid; grid-template-columns: repeat(3, 1fr); gap: 0.75rem; }
    @media (max-width: 900px) { .qgrid { grid-template-columns: 1fr; } }
    .qcard {
      border: 1px solid rgba(255,255,255,0.08);
      background: rgba(255,255,255,0.04);
      border-radius: 18px;
      padding: 14px 14px;
      min-height: 92px;
      margin-bottom: 0.35rem;
    }
    .qcard-top { display:flex; gap:0.5rem; align-items:center; margin-bottom: 0.35rem; }
    .qcard-ico { font-size: 1.1rem; opacity: 0.9; }
    .qcard-title { font-weight: 850; font-size: 1.02rem; }
    .qcard-desc { opacity: 0.75; font-size: 0.93rem; line-height: 1.35; }
    </style>
    """,
    unsafe_allow_html=True,
)

def _render_card(title: str, desc: str, prompt: str, icon: str, key: str):
    st.markdown(
        f"""
        <div class="qcard">
          <div class="qcard-top">
            <span class="qcard-ico">{icon}</span>
            <span class="qcard-title">{title}</span>
          </div>
          <div class="qcard-desc">{desc}</div>
        </div>
        """,
        unsafe_allow_html=True,
    )
    if st.button("Open", key=key, use_container_width=True):
        _run_prompt(prompt)

tab_recent, tab_freq, tab_rec = st.tabs(["🕒 Recent", "🔥 Frequent", "⭐ Recommended"])

with tab_recent:
    recents = get_recent_cards()
    if not recents:
        st.caption("No recent prompts yet — tap a Recommended card ⭐")
    else:
        cols = st.columns(3)
        for idx, p in enumerate(recents):
            with cols[idx % 3]:
                _render_card(
                    title="Recent prompt",
                    desc=(p[:90] + "…") if len(p) > 90 else p,
                    prompt=p,
                    icon="🕒",
                    key=f"recent_{idx}"
                )

with tab_freq:
    freq = get_frequent_cards()
    if not freq:
        st.caption("Frequent prompts show up after a few clicks.")
    else:
        cols = st.columns(3)
        for idx, p in enumerate(freq):
            with cols[idx % 3]:
                _render_card(
                    title="Frequently used",
                    desc=(p[:90] + "…") if len(p) > 90 else p,
                    prompt=p,
                    icon="🔥",
                    key=f"freq_{idx}"
                )

with tab_rec:
    st.caption("Basics → Advanced marketing themes. Tap to generate a guided answer.")
    cols = st.columns(3)
    for idx, item in enumerate(RECOMMENDED):
        with cols[idx % 3]:
            _render_card(
                title=item["title"],
                desc=item["desc"],
                prompt=item["prompt"],
                icon="⭐",
                key=f"rec_{idx}"
            )

###########################################
# Chat History Display
###########################################

for message in st.session_state.messages:
    role = "user" if message["role"] == "user" else "AI"
    with st.chat_message(role, avatar=user_avatar if role == "user" else ai_avatar):
        reasoning = message.get("reasoning", "")
        content = message.get("content", "")
        if st.session_state.show_reasoning and reasoning:
            st.markdown(f"<div class='reasoning'>🤔 {reasoning}</div>{content}", unsafe_allow_html=True)
        else:
            st.markdown(content)

if hasattr(st.session_state, "regenerate") and st.session_state.regenerate:
    if st.session_state.messages and st.session_state.messages[-1]["role"] == "user":
        generate_response()
    st.session_state.regenerate = False

###########################################
# User Input Handling
###########################################

if prompt := st.chat_input("Type your message..."):
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
