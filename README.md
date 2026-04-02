# NoteKing

**Video to Illustrated PDF Lecture Notes -- The Most Powerful Tool**

---

## What is NoteKing?

NoteKing turns any video URL into **beautiful, illustrated PDF lecture notes** with keyframe screenshots, structured chapters, highlighted knowledge boxes, and professional LaTeX typesetting.

**Core Feature: PDF Lecture Notes with Keyframe Illustrations**

```
Video URL  -->  Download  -->  Smart Keyframe Extraction  -->  Subtitle/ASR  -->  LLM Notes  -->  LaTeX PDF
```

### Key Capabilities

- **Smart Keyframe Extraction** -- Scene detection + info-density scoring + perceptual hash dedup. Auto-selects the best frames.
- **Professional LaTeX PDF** -- tcolorbox highlight boxes, syntax-highlighted code blocks, math formulas, cover page, TOC, headers/footers.
- **13 Output Templates** -- Detailed notes, brief summary, mind map, flashcards, quiz, timeline, exam review, tutorial, news, podcast, Xiaohongshu note, custom prompt.
- **30+ Platform Support** -- Bilibili, YouTube, Douyin, Xiaohongshu, TikTok, and 1800+ sites via yt-dlp.
- **Batch Processing** -- Process entire video collections (e.g. a 26-episode course) in one go with concurrent execution.
- **Dual PDF Engine** -- LaTeX (professional academic style) + HTML/Chrome fallback (zero-dependency).
- **Subtitle Three-Level Fallback** -- CC subtitles -> Whisper ASR -> visual mode.

---

## Demo: MiniMind Complete Course (26 Episodes)

We processed the entire [MiniMind course](https://www.bilibili.com/video/BV1T2k6BaEeC/) (26 episodes, 3.3 hours) into illustrated PDF lecture notes.

See [`demos/minimind/`](demos/minimind/) for the full output:

- `pdf/` -- 26 individual LaTeX PDFs + merged full-course PDF (33MB)
- `notes/` -- 26 Markdown source files
- `frames/` -- 300+ extracted keyframes with scene detection
- `templates_demo/` -- Episode 7 rendered in all 13 output templates

### PDF Features

| Feature | Description |
|---------|-------------|
| Cover Page | Course title, metadata, video link, NoteKing branding |
| Table of Contents | Auto-generated from section structure |
| Keyframe Screenshots | Smart-selected video frames inserted at relevant positions |
| Highlight Boxes | Key Points (yellow), Background Knowledge (blue), Caution (red) |
| Code Blocks | Dark theme with syntax highlighting and line numbers |
| Math Formulas | Native LaTeX rendering |
| Headers/Footers | NoteKing branding, page numbers, social links |

---

## Quick Start

### 1. Install Dependencies

```bash
pip install yt-dlp openai httpx pillow imagehash scenedetect opencv-python-headless
```

For LaTeX PDF output (optional but recommended):

```bash
# macOS
curl -sL "https://yihui.org/tinytex/install-bin-unix.sh" | sh
# Then install required packages:
tlmgr install ctex tcolorbox listings booktabs float fancyhdr xcolor enumitem etoolbox
```

### 2. Set Up API Key

NoteKing works with any OpenAI-compatible LLM API (DeepSeek, MiniMax, Qwen, etc.):

```bash
export NOTEKING_LLM_API_KEY="your-api-key"
export NOTEKING_LLM_BASE_URL="https://api.minimax.chat/v1"  # or any provider
export NOTEKING_LLM_MODEL="MiniMax-M2.7"
```

### 3. Generate Notes

```bash
# CLI
python -m noteking.cli run "https://www.bilibili.com/video/BV1T2k6BaEeC?p=7" --template detailed

# Python API
from noteking.core import summarize
result = summarize("https://www.bilibili.com/video/BV1T2k6BaEeC?p=7", template="detailed")
```

---

## 13 Output Templates

| Template | Name | Use Case |
|----------|------|----------|
| `brief` | Brief Summary | Quick overview in 3-5 sentences |
| `detailed` | Detailed Notes | Full chapter-structured notes for study |
| `mindmap` | Mind Map | Markmap-compatible hierarchical outline |
| `flashcard` | Flashcards | Anki-compatible Q&A cards |
| `quiz` | Quiz | Multiple choice + short answer questions |
| `timeline` | Timeline | Time-stamped knowledge points |
| `exam` | Exam Review | Formula cheat sheet + practice problems |
| `tutorial` | Tutorial Steps | Step-by-step implementation guide |
| `news` | News Brief | Journalist-style tech report |
| `podcast` | Podcast Summary | Interview/discussion format |
| `xhs_note` | Xiaohongshu Note | Social media style sharing |
| `latex_pdf` | LaTeX PDF | Professional academic-style PDF lecture notes |
| `custom` | Custom Prompt | Your own prompt template |

---

## Platform Support

| Platform | Status | Features |
|----------|--------|----------|
| Bilibili | Full | Single video, collections, multi-part, SESSDATA for HD |
| YouTube | Full | Single video, playlists, channels (proxy support for China) |
| Douyin | Supported | Short videos |
| Xiaohongshu | Supported | Auto short-link parsing |
| Kuaishou | Supported | Short videos |
| TikTok | Supported | International |
| Twitter/X | Supported | Video tweets |
| 1800+ others | Via yt-dlp | Nearly all video sites |
| Local files | Supported | MP4/MP3/WAV/FLAC |

---

## Architecture

```
noteking/
  core/
    __init__.py      # Main pipeline: summarize()
    config.py        # Configuration management
    parser.py        # URL parsing (30+ platforms)
    downloader.py    # yt-dlp wrapper
    subtitle.py      # Three-level subtitle fallback
    transcriber.py   # ASR engines (Whisper, Groq, etc.)
    frames.py        # Smart keyframe extraction
    pdf_engine.py    # PDF generation pipeline
    llm.py           # LLM interface (OpenAI-compatible)
    templates/       # 13 output templates
  cli/               # Command-line interface
  api/               # FastAPI REST API
  mcp/               # MCP server (for Cursor/Claude)
  skill/             # AI agent skill definition
  assets/            # LaTeX template
  demos/             # Example outputs
```

---

## License

MIT

---

**GitHub**: [github.com/bcefghj/noteking](https://github.com/bcefghj/noteking)
**Xiaohongshu**: bcefghj
