"""
AI Video Assistant — Streamlit frontend.

This module is a thin UI layer on top of the existing backend pipeline
defined in `main.py` (`run_pipeline`) and `core/rag_engine.py`
(`ask_question`). No AI/backend logic lives here — this file only
handles layout, session state, user interaction, and error handling.
"""

import tempfile
import traceback
from pathlib import Path

import streamlit as st

from main import run_pipeline
from core.rag_engine import ask_question


# --------------------------------------------------------------------------
# Page configuration
# --------------------------------------------------------------------------

st.set_page_config(
    page_title="AI Video Assistant",
    page_icon="🎬",
    layout="wide",
    initial_sidebar_state="expanded",
)

SUPPORTED_VIDEO_TYPES = ["mp4", "mov", "avi", "mkv"]
SUPPORTED_AUDIO_TYPES = ["mp3", "wav", "m4a"]
SUPPORTED_FILE_TYPES = SUPPORTED_VIDEO_TYPES + SUPPORTED_AUDIO_TYPES

PROCESSING_STEPS = [
    "Preparing input",
    "Extracting & chunking audio",
    "Transcribing with Whisper",
    "Generating title & summary",
    "Extracting action items",
    "Extracting key decisions",
    "Extracting open questions",
    "Building knowledge base (RAG)",
]


# --------------------------------------------------------------------------
# Styling — custom CSS to move away from default Streamlit look
# --------------------------------------------------------------------------

CUSTOM_CSS = """
<style>
    #MainMenu, footer, header {visibility: hidden;}

    .block-container {
        padding-top: 2rem;
        padding-bottom: 3rem;
        max-width: 1100px;
    }

    html, body, [class*="css"] {
        font-family: -apple-system, BlinkMacSystemFont, "Segoe UI", Inter, Roboto, sans-serif;
    }

    .app-header {
        padding-bottom: 0.25rem;
    }
    .app-header h1 {
        font-size: 2.1rem;
        font-weight: 700;
        margin-bottom: 0.15rem;
        letter-spacing: -0.02em;
    }
    .app-header p {
        color: #6b7280;
        font-size: 1.02rem;
        margin-top: 0;
    }

    .av-card {
        background: #ffffff;
        border: 1px solid #e5e7eb;
        border-radius: 12px;
        padding: 1.3rem 1.5rem;
        margin-bottom: 1rem;
    }
    .av-card h4 {
        margin-top: 0;
        margin-bottom: 0.7rem;
        font-size: 1.05rem;
        font-weight: 650;
    }

    .title-banner {
        background: linear-gradient(135deg, #111827 0%, #1f2937 100%);
        color: #ffffff;
        border-radius: 12px;
        padding: 1.4rem 1.6rem;
        margin-bottom: 1.2rem;
    }
    .title-banner .eyebrow {
        text-transform: uppercase;
        letter-spacing: 0.08em;
        font-size: 0.72rem;
        color: #9ca3af;
        margin-bottom: 0.35rem;
    }
    .title-banner h2 {
        margin: 0;
        font-size: 1.4rem;
        font-weight: 650;
    }

    .action-item, .decision-item, .question-item {
        border-left: 3px solid #d1d5db;
        padding: 0.55rem 0.9rem;
        margin-bottom: 0.5rem;
        border-radius: 6px;
        background: #f9fafb;
        font-size: 0.95rem;
    }
    .action-item { border-left-color: #10b981; }
    .decision-item { border-left-color: #3b82f6; }
    .question-item { border-left-color: #f59e0b; }

    .transcript-box {
        max-height: 480px;
        overflow-y: auto;
        padding: 1rem 1.2rem;
        background: #f9fafb;
        border: 1px solid #e5e7eb;
        border-radius: 10px;
        font-size: 0.92rem;
        line-height: 1.6;
        white-space: pre-wrap;
    }

    .stButton > button {
        border-radius: 8px;
        font-weight: 600;
        padding: 0.55rem 1.1rem;
    }

    .empty-state {
        text-align: center;
        padding: 3.5rem 1rem;
        color: #6b7280;
    }
    .empty-state h3 {
        color: #374151;
        margin-bottom: 0.4rem;
    }
</style>
"""

st.markdown(CUSTOM_CSS, unsafe_allow_html=True)


# --------------------------------------------------------------------------
# Session state initialization
# --------------------------------------------------------------------------

def init_session_state() -> None:
    defaults = {
        "result": None,
        "rag_chain": None,
        "messages": [],
        "processed": False,
        "source_label": None,
        "language_used": None,
        "last_error": None,
    }
    for key, value in defaults.items():
        if key not in st.session_state:
            st.session_state[key] = value


init_session_state()


# --------------------------------------------------------------------------
# Helpers
# --------------------------------------------------------------------------

def save_uploaded_file(uploaded_file) -> str:
    """Persist an uploaded file to a temp path and return its path as a string.

    run_pipeline / process_input expect a path or URL string, so the
    uploaded file's bytes are written to a temporary file on disk.
    """
    suffix = Path(uploaded_file.name).suffix
    tmp_dir = Path(tempfile.gettempdir()) / "ai_video_assistant_uploads"
    tmp_dir.mkdir(parents=True, exist_ok=True)
    tmp_path = tmp_dir / f"{next(tempfile._get_candidate_names())}{suffix}"
    with open(tmp_path, "wb") as f:
        f.write(uploaded_file.getbuffer())
    return str(tmp_path)


def reset_video_session() -> None:
    st.session_state["result"] = None
    st.session_state["rag_chain"] = None
    st.session_state["messages"] = []
    st.session_state["processed"] = False
    st.session_state["source_label"] = None
    st.session_state["language_used"] = None
    st.session_state["last_error"] = None


def clear_chat() -> None:
    st.session_state["messages"] = []


def run_analysis(source: str, language: str, source_label: str) -> None:
    """Run the backend pipeline once and store results in session state."""
    st.session_state["last_error"] = None
    try:
        with st.status("Analyzing video…", expanded=True) as status:
            for step in PROCESSING_STEPS:
                status.write(step)
            result = run_pipeline(source, language)
            status.update(label="Analysis complete", state="complete", expanded=False)

        st.session_state["result"] = result
        st.session_state["rag_chain"] = result.get("rag_chain")
        st.session_state["messages"] = []
        st.session_state["processed"] = True
        st.session_state["source_label"] = source_label
        st.session_state["language_used"] = language

    except Exception as exc:  # noqa: BLE001 — surfaced cleanly to the user below
        st.session_state["processed"] = False
        st.session_state["last_error"] = {
            "message": str(exc),
            "trace": traceback.format_exc(),
        }


def render_list_section(items, css_class: str, empty_message: str) -> None:
    """Render a list-like result field as styled items, or plain text fallback."""
    if not items:
        st.caption(empty_message)
        return

    if isinstance(items, (list, tuple)):
        for item in items:
            st.markdown(f'<div class="{css_class}">{item}</div>', unsafe_allow_html=True)
    else:
        # Fall back gracefully if the backend returns a plain string/block
        st.markdown(f'<div class="{css_class}">{items}</div>', unsafe_allow_html=True)


# --------------------------------------------------------------------------
# Sidebar
# --------------------------------------------------------------------------

with st.sidebar:
    st.markdown("### AI Video Assistant")
    st.divider()

    st.markdown("#### Current Video")
    if st.session_state["processed"] and st.session_state["result"]:
        st.write(f"**Title:** {st.session_state['result'].get('title', 'Untitled')}")
        st.write(f"**Source:** {st.session_state['source_label']}")
        st.write(f"**Language:** {st.session_state['language_used'].capitalize()}")
    else:
        st.caption("No video processed yet.")

    st.divider()

    st.markdown("#### Actions")
    col_a, col_b = st.columns(2)
    with col_a:
        if st.button("New Video", use_container_width=True):
            reset_video_session()
            st.rerun()
    with col_b:
        if st.button("Clear Chat", use_container_width=True):
            clear_chat()
            st.rerun()

    if st.button("Clear Results", use_container_width=True):
        reset_video_session()
        st.rerun()

    st.divider()

    st.markdown("#### About")
    st.caption(
        "AI Video Assistant converts videos into searchable knowledge "
        "using transcription, summarization and retrieval-augmented "
        "generation (RAG)."
    )


# --------------------------------------------------------------------------
# Header
# --------------------------------------------------------------------------

st.markdown(
    """
    <div class="app-header">
        <h1>AI Video Assistant</h1>
        <p>Turn long videos into summaries, insights, decisions and an interactive knowledge base.</p>
    </div>
    """,
    unsafe_allow_html=True,
)


# --------------------------------------------------------------------------
# Input section
# --------------------------------------------------------------------------

if not st.session_state["processed"]:
    st.markdown('<div class="av-card">', unsafe_allow_html=True)
    st.markdown("#### Add a video")

    source_tab, file_tab = st.tabs(["YouTube URL", "Local File"])

    with source_tab:
        youtube_url = st.text_input(
            "Enter YouTube video URL",
            placeholder="https://www.youtube.com/watch?v=...",
            key="youtube_url_input",
        )

    with file_tab:
        uploaded_file = st.file_uploader(
            "Upload a video or audio file",
            type=SUPPORTED_FILE_TYPES,
            key="file_upload_input",
        )
        st.caption("Supported formats: " + ", ".join(SUPPORTED_FILE_TYPES))

    st.markdown("**Language**")
    language_choice = st.radio(
        "Language",
        options=["English", "Hinglish"],
        index=0,
        horizontal=True,
        label_visibility="collapsed",
    )

    analyze_clicked = st.button("Analyze Video", type="primary", use_container_width=True)

    st.markdown("</div>", unsafe_allow_html=True)

    if analyze_clicked:
        source_value = None
        source_label = None

        if youtube_url and youtube_url.strip():
            source_value = youtube_url.strip()
            source_label = "YouTube URL"
        elif uploaded_file is not None:
            source_value = save_uploaded_file(uploaded_file)
            source_label = f"Local file ({uploaded_file.name})"

        if not source_value:
            st.error("Please provide a YouTube URL or upload a file before analyzing.")
        else:
            run_analysis(source_value, language_choice.lower(), source_label)
            st.rerun()

    if st.session_state["last_error"]:
        st.error("Something went wrong while processing the video.\n\nPlease check your input and try again.")
        with st.expander("Show technical details"):
            st.code(st.session_state["last_error"]["trace"])

    if not st.session_state["last_error"]:
        st.markdown(
            """
            <div class="empty-state">
                <h3>No video analyzed yet</h3>
                <p>Add a YouTube URL or upload a file above, then click "Analyze Video" to get started.</p>
            </div>
            """,
            unsafe_allow_html=True,
        )


# --------------------------------------------------------------------------
# Results dashboard
# --------------------------------------------------------------------------

if st.session_state["processed"] and st.session_state["result"]:
    result = st.session_state["result"]

    st.markdown(
        f"""
        <div class="title-banner">
            <div class="eyebrow">Generated Title</div>
            <h2>{result.get('title', 'Untitled video')}</h2>
        </div>
        """,
        unsafe_allow_html=True,
    )

    overview_tab, actions_tab, decisions_tab, questions_tab, transcript_tab, chat_tab = st.tabs(
        ["Summary", "Action Items", "Key Decisions", "Open Questions", "Transcript", "Chat with Video"]
    )

    with overview_tab:
        st.markdown('<div class="av-card">', unsafe_allow_html=True)
        st.markdown("#### Summary")
        st.write(result.get("summary", "No summary available."))
        st.markdown("</div>", unsafe_allow_html=True)

    with actions_tab:
        st.markdown('<div class="av-card">', unsafe_allow_html=True)
        st.markdown("#### Action Items")
        render_list_section(
            result.get("action_items"),
            "action-item",
            "No action items were identified.",
        )
        st.markdown("</div>", unsafe_allow_html=True)

    with decisions_tab:
        st.markdown('<div class="av-card">', unsafe_allow_html=True)
        st.markdown("#### Key Decisions")
        render_list_section(
            result.get("key_decisions"),
            "decision-item",
            "No key decisions were identified.",
        )
        st.markdown("</div>", unsafe_allow_html=True)

    with questions_tab:
        st.markdown('<div class="av-card">', unsafe_allow_html=True)
        st.markdown("#### Open Questions")
        render_list_section(
            result.get("open_questions"),
            "question-item",
            "No open questions were identified.",
        )
        st.markdown("</div>", unsafe_allow_html=True)

    with transcript_tab:
        st.markdown('<div class="av-card">', unsafe_allow_html=True)
        st.markdown("#### Full Transcript")
        transcript_text = result.get("transcript", "")
        st.markdown(f'<div class="transcript-box">{transcript_text}</div>', unsafe_allow_html=True)
        st.markdown("</div>", unsafe_allow_html=True)

    with chat_tab:
        st.markdown("#### Chat with Video")
        st.caption("Ask questions about the video's content. Answers are grounded in the transcript via RAG.")

        for message in st.session_state["messages"]:
            with st.chat_message(message["role"]):
                st.write(message["content"])

        user_question = st.chat_input("Ask a question about this video…")

        if user_question:
            st.session_state["messages"].append({"role": "user", "content": user_question})
            with st.chat_message("user"):
                st.write(user_question)

            with st.chat_message("assistant"):
                try:
                    with st.spinner("Thinking…"):
                        answer = ask_question(st.session_state["rag_chain"], user_question)
                    st.write(answer)
                    st.session_state["messages"].append({"role": "assistant", "content": answer})
                except Exception as exc:  # noqa: BLE001
                    st.error("Something went wrong while answering your question.\n\nPlease try again.")
                    with st.expander("Show technical details"):
                        st.code(traceback.format_exc())