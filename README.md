# AI Video Assistant

Turns a YouTube video or a local video/audio file into a transcript, a generated title, a summary, and structured insights (action items, key decisions, open questions). You can then ask questions about the content through a RAG chat interface.

The project provides a Streamlit UI (`app.py`) and a command-line workflow (`main.py`).

## Features

- Accept a YouTube URL or a local video/audio file
- Extract audio, convert it to WAV, and split it into 10-minute chunks
- Transcribe English audio locally with OpenAI Whisper
- Transcribe Hinglish audio with Sarvam AI (speech-to-text with translation to English)
- Generate a short title and a bullet-point summary with Mistral
- Extract action items, key decisions, and unresolved questions from the transcript
- Store transcript chunks in a local Chroma vector database
- Answer questions about the video using retrieval-augmented generation (RAG)
- Streamlit dashboard for input, results, and chat
- CLI for the same pipeline plus an interactive Q&A loop

## How It Works

```text
YouTube URL or local file
→ Audio download / WAV conversion
→ 10-minute audio chunks
→ Transcription (Whisper or Sarvam)
→ Title, summary, action items, decisions, open questions (Mistral)
→ Chroma vector store (HuggingFace embeddings)
→ RAG Q&A
```

1. **Input** — A YouTube URL is downloaded as audio with `yt-dlp`. A local file is converted to 16 kHz mono WAV with pydub/FFmpeg.
2. **Audio processing** — The WAV file is split into 10-minute chunks.
3. **Transcription** — `english` uses a local Whisper model (default: `base`). `hinglish` sends 25-second pieces to Sarvam’s speech-to-text-translate API.
4. **Analysis** — Mistral (`mistral-small-latest`) generates a title, a map-reduce summary, action items (task, owner, deadline), key decisions, and open questions.
5. **Vector store** — The transcript is split into overlapping chunks, embedded with `all-MiniLM-L6-v2`, and stored in Chroma (`vector_db/`).
6. **RAG Q&A** — Questions retrieve the top 4 similar chunks and are answered by Mistral using only that context.

## Project Structure

```text
ai-video-assistant/
├── app.py                 # Streamlit UI
├── main.py                # Pipeline orchestration and CLI
├── core/
│   ├── transcriber.py     # Whisper (English) and Sarvam (Hinglish) transcription
│   ├── summarizer.py      # Title generation and map-reduce summarization
│   ├── extractor.py       # Action items, key decisions, open questions
│   ├── vector_store.py    # Chroma store and similarity retriever
│   └── rag_engine.py      # RAG chain construction and Q&A
├── utils/
│   └── audio_processor.py # YouTube download, WAV conversion, audio chunking
├── pyproject.toml         # Project metadata
├── requirements.txt       # Python dependencies
└── .python-version        # Python 3.11
```

Downloaded audio is written to `downloades/`. The vector database is written to `vector_db/`. Both directories are gitignored.

## Requirements

- Python 3.11+
- [FFmpeg](https://ffmpeg.org/) on your PATH (used by `yt-dlp` and pydub)
- A [Mistral API](https://mistral.ai/) key
- A [Sarvam AI](https://sarvam.ai/) API key if you use Hinglish transcription

## Setup

```bash
python -m venv .venv
source .venv/bin/activate   # Windows: .venv\Scripts\activate
pip install -r requirements.txt
```

Create a `.env` file in the project root:

```env
MISTRAL_API_KEY=your_mistral_api_key
SARVAM_API_KEY=your_sarvam_api_key
```

Optional:

```env
WHISPER_MODEL=base          # Whisper model name (default: base)
SARVAM_STT_MODEL=saaras:v2.5
```

## Usage

### Streamlit UI

```bash
streamlit run app.py
```

1. Paste a YouTube URL or upload a file (`mp4`, `mov`, `avi`, `mkv`, `mp3`, `wav`, `m4a`).
2. Choose **English** or **Hinglish**.
3. Click **Analyze Video**.
4. Review the title, summary, action items, key decisions, open questions, and full transcript.
5. Use **Chat with Video** to ask questions grounded in the transcript.

### Command line

```bash
python main.py
```

You will be prompted for a YouTube URL or local file path, then a language (`english` or `hinglish`). After analysis, type questions in the terminal. Enter `exit`, `quit`, or `q` to stop.

## Notes

- English transcription runs locally with Whisper and PyTorch. The first run downloads the selected Whisper model.
- Hinglish transcription requires `SARVAM_API_KEY` and sends audio to Sarvam’s API.
- Title, summary, extraction, and RAG all require `MISTRAL_API_KEY`.
- The UI always runs the full pipeline for a new video; it does not reload a previous vector store from disk.
