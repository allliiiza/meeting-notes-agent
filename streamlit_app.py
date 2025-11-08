import re
from io import BytesIO
from typing import List, Dict, Tuple

import streamlit as st
from transformers import pipeline
from pypdf import PdfReader

# ---------- Page config ----------
st.set_page_config(
    page_title="Meeting Notes Agent",
    layout="wide"
)

# ---------- Sidebar settings ----------
st.sidebar.title("⚙️ Settings")

MODEL_NAME = st.sidebar.selectbox(
    "Summarizer model",
    options=[
        "facebook/bart-large-cnn",
        "t5-small",
    ],
    index=0,
    help="Use BART for higher quality (slower). Use T5-small for faster tests."
)

level = st.sidebar.radio(
    "Summary length",
    ["Brief", "Standard", "Detailed"],
    index=1,
)

APPLY_REDACTION = st.sidebar.checkbox(
    "Redact emails & phone numbers",
    value=True
)


def level_presets(level: str) -> Tuple[int, int, int, int]:
    level = level.lower()
    if level == "brief":
        return 110, 40, 160, 60
    if level == "detailed":
        return 200, 80, 300, 140
    return 140, 60, 220, 80  # standard


inner_max, inner_min, final_max, final_min = level_presets(level)

# ---------- Helpers ----------


def redact_pii(text: str) -> str:
    t = re.sub(r"[\w\.-]+@[\w\.-]+", "[redacted_email]", text)
    t = re.sub(r"\(?\d{3}\)?[-\s.]?\d{3}[-\s.]?\d{4}", "[redacted_phone]", t)
    return t


def simple_word_chunk(
    text: str,
    max_words: int = 350,
    overlap: int = 50,
    max_chars: int = 1000,
) -> List[str]:
    """Split long text into overlapping chunks by words + char limit."""
    words = text.split()
    if not words:
        return []

    chunks: List[str] = []
    start = 0

    while start < len(words):
        current_chunk_words = []
        current_chunk_chars = 0

        for i in range(start, len(words)):
            word = words[i]
            extra_space = 1 if current_chunk_chars > 0 else 0

            # Stop if next word would break limits
            if (
                len(current_chunk_words) >= max_words
                or current_chunk_chars + len(word) + extra_space > max_chars
            ):
                if not current_chunk_words:
                    # Edge case: single huge word
                    current_chunk_words.append(word)
                break

            current_chunk_words.append(word)
            current_chunk_chars += len(word) + extra_space

        if not current_chunk_words:
            break

        chunks.append(" ".join(current_chunk_words))

        # Move start with overlap
        step = len(current_chunk_words) - overlap
        if step <= 0:
            step = len(current_chunk_words)
        start += step

        # Safety: avoid infinite loop on tiny tails
        if start >= len(words) - overlap and len(current_chunk_words) <= overlap:
            start = len(words)

    return chunks


def _add_task_prefix(text: str, model_name: str) -> str:
    """Add T5-style 'summarize:' prefix when needed."""
    return f"summarize: {text}" if model_name.lower().startswith("t5") else text


@st.cache_resource(show_spinner=False)
def load_summarizer(model_name: str):
    return pipeline("summarization", model=model_name)


class Summarizer:
    def __init__(self, model_name: str):
        self.model_name = model_name
        self.pipe = load_summarizer(model_name)

    def summarize(
        self,
        text: str,
        max_length: int,
        min_length: int,
        do_sample: bool = False,
    ) -> str:
        t = _add_task_prefix(text.strip(), self.model_name)
        if not t:
            return ""
        out = self.pipe(
            t,
            max_length=max_length,
            min_length=min_length,
            do_sample=do_sample,
            truncation=True,
        )
        if isinstance(out, list) and out and "summary_text" in out[0]:
            return out[0]["summary_text"].strip()
        return str(out)


def map_reduce_summary(
    text: str,
    model_name: str,
    chunk_words: int,
    overlap: int,
    inner_max: int,
    inner_min: int,
    final_max: int,
    final_min: int,
) -> Dict[str, str]:
    """Chunk → summarize each → combine → summarize again."""
    text = text.strip()
    if not text:
        return {"chunks": [], "partials": [], "combined": "", "final": ""}

    chunks = simple_word_chunk(text, max_words=chunk_words, overlap=overlap)
    sm = Summarizer(model_name=model_name)

    if not chunks:
        return {"chunks": [], "partials": [], "combined": "", "final": ""}

    partials: List[str] = []
    for c in chunks:
        w = len(c.split())
        if w <= 80:
            imax, imin = 80, 30
        elif w <= 160:
            imax, imin = 120, 40
        else:
            imax, imin = inner_max, inner_min
        partials.append(sm.summarize(c, max_length=imax, min_length=imin))

    # drop empties
    partials = [p for p in partials if p.strip()]
    if not partials:
        return {"chunks": chunks, "partials": [], "combined": "", "final": ""}

    combined = "\n".join(f"- {p}" for p in partials)
    final = sm.summarize(combined, max_length=final_max, min_length=final_min)

    return {
        "chunks": chunks,
        "partials": partials,
        "combined": combined,
        "final": (final or "").strip(),
    }


def extract_text_from_file(uploaded_file) -> str:
    """Read text from uploaded .txt or .pdf using pypdf (no PyMuPDF)."""
    if uploaded_file is None:
        return ""

    name = uploaded_file.name.lower()

    # Text file
    if name.endswith(".txt"):
        data = uploaded_file.read()
        return data.decode("utf-8", errors="ignore")

    # PDF file
    if name.endswith(".pdf"):
        try:
            data = uploaded_file.read()
            reader = PdfReader(BytesIO(data))
            text = ""
            for page in reader.pages:
                page_text = page.extract_text() or ""
                text += page_text + "\n"
            return text.strip()
        except Exception as e:
            st.error(f"Could not read PDF: {e}")
            return ""

    # Fallback
    st.warning("Unsupported file type. Please upload a .txt or .pdf file.")
    return ""


# ---------- UI ----------
st.title("📝 Meeting Notes Agent")
st.write(
    "Paste or upload a meeting transcript and generate a clean summary with key points."
)

input_mode = st.radio(
    "How do you want to provide your content?",
    ("Paste text", "Upload file (.txt / .pdf)"),
    horizontal=True,
)

raw_text = ""

if input_mode == "Paste text":
    raw_text = st.text_area(
        "Paste your meeting transcript here:",
        height=260,
        placeholder="Paste your Zoom transcript, meeting notes, or documentation...",
    )
else:
    uploaded_file = st.file_uploader(
        "Upload a .txt or .pdf file",
        type=["txt", "pdf"],
    )
    if uploaded_file is not None:
        with st.spinner("Reading file..."):
            raw_text = extract_text_from_file(uploaded_file)
        if raw_text:
            st.success(f"Loaded {len(raw_text)} characters from file.")
        else:
            st.error("Could not read text from this file. Please try another.")

if st.button("🚀 Generate Summary", type="primary"):
    if not raw_text or not raw_text.strip():
        st.warning("Please paste text or upload a file before summarizing.")
    else:
        with st.spinner("Summarizing..."):
            text_to_use = redact_pii(raw_text) if APPLY_REDACTION else raw_text

            result = map_reduce_summary(
                text=text_to_use,
                model_name=MODEL_NAME,
                chunk_words=350,
                overlap=50,
                inner_max=inner_max,
                inner_min=inner_min,
                final_max=final_max,
                final_min=final_min,
            )

        final_summary = (result.get("final") or "").strip()

        st.subheader("✅ Final Summary")
        if final_summary:
            # Bullet-ize like your Colab cell
            sentences = re.split(r"(?<=[.!?])\s+|\n", final_summary)
            bullets = [f"- {s.strip()}" for s in sentences if s.strip()]
            st.markdown("\n".join(bullets))
        else:
            st.info("No summary could be generated. Try a longer or clearer input.")

        # Optional: show chunk summaries
        partials = result.get("partials", [])
        if partials:
            with st.expander("Show intermediate chunk summaries"):
                for i, p in enumerate(partials, start=1):
                    st.markdown(f"**Chunk {i}:** {p}")
