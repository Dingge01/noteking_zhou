# NoteKing 👑

**The Ultimate Video/Blog to Learning Notes Tool**

[English](#english) | [中文](#中文)

---

<a id="中文"></a>

## 中文

### 什么是 NoteKing？

NoteKing 是一个开源的视频/博客转学习笔记工具，支持 **30+ 平台**，提供 **13 种输出模板**，覆盖 **8 种使用方式**。

把任何视频链接丢给它，就能得到结构化的学习笔记、思维导图、闪卡、测验题等。

### 支持平台

| 平台 | 状态 | 说明 |
|------|------|------|
| 哔哩哔哩 | 核心支持 | 单视频/合集/分P/系列/收藏夹/大会员 |
| YouTube | 核心支持 | 单视频/播放列表/频道 (内置代理支持) |
| 抖音 | 支持 | 短视频 |
| 小红书 | 支持 | 短链自动解析 |
| 快手 | 支持 | 短视频 |
| TikTok | 支持 | 国际版 |
| Twitter/X | 支持 | 视频推文 |
| 1800+ 其他 | 通过 yt-dlp | 几乎所有视频网站 |
| 本地文件 | 支持 | MP4/MP3/WAV/FLAC |

### 13 种输出模板

| 模板 | 用途 | 适合场景 |
|------|------|----------|
| `brief` 简要总结 | 3-5 句话概括 | 快速了解 |
| `detailed` 详细笔记 | 带章节结构的完整笔记 | 系统学习 |
| `mindmap` 思维导图 | Markmap 格式 | 知识梳理 |
| `flashcard` 闪卡 | Anki 兼容的问答卡 | 记忆背诵 |
| `quiz` 测验题 | 选择题+简答题 | 自我测试 |
| `timeline` 时间线 | 带时间戳的笔记 | 回看定位 |
| `exam` 考试笔记 | 公式+概念+易错点 | 考前复习 |
| `tutorial` 教程步骤 | 步骤化操作指南 | 实操学习 |
| `news` 新闻速览 | 新闻摘要格式 | 资讯获取 |
| `podcast` 播客摘要 | 访谈/讨论总结 | 播客收听 |
| `xhs_note` 小红书笔记 | 小红书风格排版 | 内容创作 |
| `latex_pdf` LaTeX 讲义 | 学术级 PDF | 课程讲义 |
| `custom` 自定义 | 用户自定义 Prompt | 个性需求 |

### 快速开始

#### 方式一：Docker 一键部署 (推荐)

```bash
git clone https://github.com/bcefghj/noteking.git
cd noteking
cp .env.example .env
# 编辑 .env 填入你的 LLM API Key
docker compose up -d
# 访问 http://localhost:3000
```

#### 方式二：pip 安装

```bash
pip install noteking
noteking setup  # 交互式配置
noteking run "https://www.bilibili.com/video/BVxxx" -t detailed
```

#### 方式三：OpenClaw / Claude Code

```bash
npx skills add bcefghj/noteking
```

然后对 AI 说: "帮我总结这个视频 https://www.bilibili.com/video/BVxxx"

### 8 种使用方式

1. **Web 应用** - 浏览器打开，粘贴链接即可
2. **CLI 工具** - `noteking run <URL> -t <模板>`
3. **MCP Server** - 接入 Cursor / Claude Desktop
4. **OpenClaw Skill** - 对小龙虾说话即可
5. **Python API** - `from core import summarize`
6. **REST API** - POST /api/v1/summarize
7. **桌面端** - Tauri 跨平台应用 (开发中)
8. **Docker** - 一键部署

### 核心特性

- **字幕三级回退**: CC 字幕 → ASR 语音识别 → 纯视觉模式
- **7+ ASR 引擎**: faster-whisper / Groq / OpenAI / 火山引擎 / 阿里云 / Deepgram / Bcut
- **关键帧提取**: 场景检测 + 字幕时间对齐
- **批量处理**: 合集/播放列表一键处理 (支持 50+ 集课程)
- **YouTube 代理**: 内置 HTTP/SOCKS5 代理支持
- **多 LLM 支持**: OpenAI / DeepSeek / Claude / Gemini / 通义千问 / 本地 Ollama
- **缓存**: 避免重复处理相同视频

### 项目结构

```
noteking/
├── core/           # 核心引擎 (Python)
├── api/            # FastAPI 后端
├── web/            # Next.js 前端
├── mcp/            # MCP Server (TypeScript)
├── skill/          # Agent Skill (SKILL.md)
├── cli/            # CLI 工具
├── desktop/        # Tauri 桌面端
├── docs/           # 文档教程
├── docker-compose.yml
├── Dockerfile
└── install.sh      # 一键安装脚本
```

### 文档

- [Docker 部署教程](docs/deploy-docker.md)
- [阿里云部署教程](docs/deploy-aliyun.md)
- [腾讯云部署教程](docs/deploy-tencent.md)
- [Vercel 部署教程](docs/deploy-vercel.md)
- [YouTube 代理配置](docs/youtube-proxy.md)
- [OpenClaw 安装教程](docs/openclaw-install.md)
- [小白入门指南](docs/beginner-guide.md)

### 致谢

NoteKing 站在巨人的肩膀上，汲取了 30+ 开源工具的精华：

- [BiliNote](https://github.com/JefferyHcool/BiliNote) - B站笔记生成器
- [wdkns-skills](https://github.com/wdkns/wdkns-skills) - 视频转 LaTeX PDF 讲义
- [steipete/summarize](https://github.com/steipete/summarize) - Chrome 扩展 + CLI
- [transcriptor-mcp](https://github.com/samson-art/transcriptor-mcp) - 多平台 MCP
- [vidscribe](https://github.com/XFWang522/vidscribe) - 多 ASR 引擎
- [TubePilot](https://github.com/ixex/tubepilot) - YouTube MCP 27 工具
- [302_video_summary](https://github.com/302ai/302_video_summary) - 思维导图 + AI Q&A
- [Video_note_generator](https://github.com/whotto/Video_note_generator) - 小红书笔记
- [PlanOpticon](https://github.com/ConflictHQ/PlanOpticon) - 知识图谱
- [lecture-mind](https://github.com/matte1782/lecture-mind) - 闪卡 + 间隔重复

### License

MIT

---

<a id="english"></a>

## English

### What is NoteKing?

NoteKing is an open-source tool that converts videos and blogs into structured learning notes.
It supports **30+ platforms**, offers **13 output templates**, and provides **8 deployment options**.

### Quick Start

```bash
# Docker (recommended)
git clone https://github.com/bcefghj/noteking.git
cd noteking && cp .env.example .env
# Edit .env with your LLM API key
docker compose up -d
# Visit http://localhost:3000

# CLI
pip install noteking
noteking setup
noteking run "https://youtu.be/xxx" -t detailed

# MCP / Skill
npx skills add bcefghj/noteking
```

### Features

- **30+ Platforms**: Bilibili, YouTube, Douyin, Xiaohongshu, TikTok, Twitter, and 1800+ more
- **13 Templates**: Brief, Detailed, Mind Map, Flashcard, Quiz, Timeline, Exam, Tutorial, News, Podcast, Xiaohongshu Note, LaTeX PDF, Custom
- **8 Interfaces**: Web, CLI, MCP Server, Skill, Python API, REST API, Desktop, Docker
- **Smart Subtitles**: Three-level fallback (CC → ASR → Visual)
- **7+ ASR Engines**: faster-whisper, Groq, OpenAI, Volcengine, Aliyun, Deepgram, Bcut
- **Batch Processing**: Playlists and courses with 50+ videos
- **YouTube Proxy**: Built-in proxy for China users
- **Multi-LLM**: OpenAI, DeepSeek, Claude, Gemini, Qwen, Ollama

### License

MIT
