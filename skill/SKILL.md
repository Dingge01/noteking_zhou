---
name: noteking
description: |
  Convert any video or blog into structured learning notes. Supports Bilibili, YouTube, Douyin, Xiaohongshu, TikTok, and 1800+ platforms via yt-dlp. Offers 13 output templates including detailed notes, mind maps, flashcards, quizzes, exam notes, timeline, LaTeX PDF, and more. Use when the user provides a video URL and wants summarized content, study notes, or any form of structured output from video/audio content.
---

# NoteKing - Video/Blog to Learning Notes

Use this skill when the user wants to convert a video, podcast, or blog into structured notes.

## Supported Platforms

- **Bilibili** (B站): single videos, multi-part (分P), series, collections, favorites
- **YouTube**: single videos, playlists, channels (proxy support for China)
- **Douyin** (抖音), **Xiaohongshu** (小红书), **Kuaishou** (快手)
- **TikTok**, **Twitter/X**, **Instagram**, **Twitch**, **Vimeo**, **Facebook**
- **1800+ additional sites** via yt-dlp
- **Local files**: MP4, MP3, WAV, FLAC

## Available Templates

| Template | Name | Best For |
|----------|------|----------|
| `brief` | 简要总结 | Quick 3-5 sentence overview |
| `detailed` | 详细学习笔记 | Systematic study with chapters & key points |
| `mindmap` | 思维导图 | Visual knowledge structure (Markmap) |
| `flashcard` | 闪卡 (Anki) | Spaced repetition study |
| `quiz` | 测验题 | Self-testing with MCQ + short answer |
| `timeline` | 时间线笔记 | Timestamped notes with video links |
| `exam` | 考试复习笔记 | Exam preparation with formulas & key concepts |
| `tutorial` | 教程步骤 | Step-by-step how-to extraction |
| `news` | 新闻速览 | News-style quick summary |
| `podcast` | 播客/访谈摘要 | Interview & discussion summary |
| `xhs_note` | 小红书笔记 | Xiaohongshu-style social media note |
| `latex_pdf` | LaTeX PDF 讲义 | Professional academic lecture notes |
| `custom` | 自定义 | User-defined prompt template |

## Dependencies

The following tools must be available in the environment:

- `yt-dlp` - Video downloading (install: `pip install yt-dlp`)
- `ffmpeg` - Audio/video processing (install: `brew install ffmpeg` or `apt install ffmpeg`)

Python packages (auto-installed):
- `openai` - LLM API client
- `youtube-transcript-api` - YouTube subtitle extraction
- `httpx` - HTTP client
- `pysrt` - SRT subtitle parsing

Optional:
- `faster-whisper` - Local speech recognition (install: `pip install faster-whisper`)

## Environment Variables

| Variable | Required | Description |
|----------|----------|-------------|
| `NOTEKING_LLM_API_KEY` | Yes | API key for LLM (OpenAI/DeepSeek/etc.) |
| `NOTEKING_LLM_BASE_URL` | No | Custom API base URL |
| `NOTEKING_LLM_MODEL` | No | Model name (default: gpt-4o-mini) |
| `NOTEKING_PROXY` | No | Proxy URL for YouTube access from China |
| `BILIBILI_SESSDATA` | No | Bilibili login cookie for HD & member content |

## Usage

### Basic Usage

When the user provides a video URL, run the NoteKing core:

```bash
cd /path/to/noteking
python -m cli run "<VIDEO_URL>" -t <TEMPLATE>
```

### Examples

```bash
# Detailed notes from a Bilibili video
python -m cli run "https://www.bilibili.com/video/BV1xx411c79H" -t detailed

# Mind map from a YouTube video
python -m cli run "https://youtu.be/dQw4w9WgXcQ" -t mindmap

# Flashcards from a local lecture
python -m cli run "./lecture.mp4" -t flashcard

# Exam notes with custom API
python -m cli run "https://www.bilibili.com/video/BV1xx" -t exam --api-key sk-xxx --model deepseek-chat --base-url https://api.deepseek.com
```

### Batch Processing

For playlists or video collections (e.g., a 50-lecture math course):

```bash
python -m cli run "https://www.bilibili.com/video/BV1xx?p=1" -t detailed
python -m cli run "https://youtube.com/playlist?list=PLxxx" -t exam
```

### Available CLI Commands

```bash
python -m cli run <URL> -t <TEMPLATE>    # Process video
python -m cli transcript <URL>            # Extract transcript only
python -m cli templates                   # List all templates
python -m cli setup                       # Interactive setup
python -m cli cache-clear                 # Clear cache
```

## Workflow

1. **Parse URL** → Detect platform, check if playlist/collection
2. **Extract subtitles** → Three-level fallback: CC subtitles → ASR → Visual mode
3. **Get metadata** → Title, chapters, duration, thumbnail
4. **Generate notes** → Apply selected template via LLM
5. **Save output** → Markdown file + SRT subtitles + transcript

## Configuration

Config file location: `~/.noteking/config.json`

Run `python -m cli setup` for interactive configuration, or edit the JSON directly.

## Notes for AI Agents

- When the user says "summarize this video" or "help me take notes on this video", use this skill
- Default to `detailed` template unless the user specifies otherwise
- For exam preparation, recommend `exam` + `flashcard` templates
- For quick browsing, recommend `brief` template
- For courses with multiple videos, use batch processing
- Always check if `yt-dlp` and `ffmpeg` are installed first
