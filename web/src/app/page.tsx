"use client";

import { useState } from "react";
import ReactMarkdown from "react-markdown";

const TEMPLATES = [
  { name: "brief", label: "简要总结", icon: "⚡" },
  { name: "detailed", label: "详细学习笔记", icon: "📝" },
  { name: "mindmap", label: "思维导图", icon: "🧠" },
  { name: "flashcard", label: "闪卡 (Anki)", icon: "🃏" },
  { name: "quiz", label: "测验题", icon: "❓" },
  { name: "timeline", label: "时间线笔记", icon: "⏱️" },
  { name: "exam", label: "考试复习笔记", icon: "📚" },
  { name: "tutorial", label: "教程步骤", icon: "🛠️" },
  { name: "news", label: "新闻速览", icon: "📰" },
  { name: "podcast", label: "播客摘要", icon: "🎙️" },
  { name: "xhs_note", label: "小红书笔记", icon: "📕" },
  { name: "latex_pdf", label: "LaTeX PDF", icon: "📄" },
  { name: "custom", label: "自定义", icon: "✏️" },
];

type Result = {
  title: string;
  content: string;
  template: string;
  source: string;
  platform: string;
  duration: number;
};

export default function Home() {
  const [url, setUrl] = useState("");
  const [template, setTemplate] = useState("detailed");
  const [customPrompt, setCustomPrompt] = useState("");
  const [loading, setLoading] = useState(false);
  const [result, setResult] = useState<Result | null>(null);
  const [error, setError] = useState("");
  const [darkMode, setDarkMode] = useState(false);

  const handleSubmit = async () => {
    if (!url.trim()) return;
    setLoading(true);
    setError("");
    setResult(null);

    try {
      const resp = await fetch("/api/v1/summarize", {
        method: "POST",
        headers: { "Content-Type": "application/json" },
        body: JSON.stringify({
          url: url.trim(),
          template,
          custom_prompt: customPrompt,
        }),
      });

      if (!resp.ok) {
        const errData = await resp.json().catch(() => ({}));
        throw new Error(errData.detail || `HTTP ${resp.status}`);
      }

      const data = await resp.json();
      setResult(data);
    } catch (e: any) {
      setError(e.message || "Processing failed");
    } finally {
      setLoading(false);
    }
  };

  const handleCopy = () => {
    if (result?.content) {
      navigator.clipboard.writeText(result.content);
    }
  };

  const handleDownload = () => {
    if (!result?.content) return;
    const blob = new Blob([result.content], { type: "text/markdown" });
    const a = document.createElement("a");
    a.href = URL.createObjectURL(blob);
    a.download = `${result.title || "notes"}_${template}.md`;
    a.click();
  };

  return (
    <div className={darkMode ? "dark" : ""}>
      <div className="min-h-screen bg-[var(--bg-primary)] text-[var(--text-primary)]">
        {/* Header */}
        <header className="border-b border-[var(--border)] bg-[var(--bg-secondary)]">
          <div className="max-w-6xl mx-auto px-4 py-4 flex items-center justify-between">
            <div className="flex items-center gap-3">
              <span className="text-3xl">👑</span>
              <div>
                <h1 className="text-xl font-bold">NoteKing</h1>
                <p className="text-xs text-[var(--text-secondary)]">
                  Video &amp; Blog to Learning Notes
                </p>
              </div>
            </div>
            <button
              onClick={() => setDarkMode(!darkMode)}
              className="px-3 py-1.5 rounded-lg border border-[var(--border)] hover:bg-[var(--bg-secondary)] transition text-sm"
            >
              {darkMode ? "☀️ Light" : "🌙 Dark"}
            </button>
          </div>
        </header>

        <main className="max-w-6xl mx-auto px-4 py-8">
          {/* Input Section */}
          <div className="bg-[var(--bg-secondary)] rounded-2xl p-6 border border-[var(--border)] mb-8">
            <div className="flex gap-3 mb-4">
              <input
                type="text"
                value={url}
                onChange={(e) => setUrl(e.target.value)}
                placeholder="Paste video URL here... (Bilibili, YouTube, Douyin, etc.)"
                className="flex-1 px-4 py-3 rounded-xl bg-[var(--bg-primary)] border border-[var(--border)] focus:outline-none focus:ring-2 focus:ring-[var(--accent)] text-base"
                onKeyDown={(e) => e.key === "Enter" && handleSubmit()}
              />
              <button
                onClick={handleSubmit}
                disabled={loading || !url.trim()}
                className="px-6 py-3 bg-[var(--accent)] text-white rounded-xl font-medium hover:opacity-90 transition disabled:opacity-50 whitespace-nowrap"
              >
                {loading ? "Processing..." : "Generate Notes"}
              </button>
            </div>

            {/* Template Selector */}
            <div className="flex flex-wrap gap-2">
              {TEMPLATES.map((t) => (
                <button
                  key={t.name}
                  onClick={() => setTemplate(t.name)}
                  className={`px-3 py-1.5 rounded-lg text-sm transition ${
                    template === t.name
                      ? "bg-[var(--accent)] text-white"
                      : "bg-[var(--bg-primary)] border border-[var(--border)] hover:border-[var(--accent)]"
                  }`}
                >
                  {t.icon} {t.label}
                </button>
              ))}
            </div>

            {template === "custom" && (
              <textarea
                value={customPrompt}
                onChange={(e) => setCustomPrompt(e.target.value)}
                placeholder="Enter your custom prompt..."
                className="w-full mt-3 px-4 py-3 rounded-xl bg-[var(--bg-primary)] border border-[var(--border)] focus:outline-none focus:ring-2 focus:ring-[var(--accent)] min-h-[80px] text-sm"
              />
            )}
          </div>

          {/* Error */}
          {error && (
            <div className="bg-red-50 dark:bg-red-900/20 border border-red-200 dark:border-red-800 text-red-700 dark:text-red-300 rounded-xl p-4 mb-6">
              {error}
            </div>
          )}

          {/* Loading */}
          {loading && (
            <div className="flex flex-col items-center justify-center py-16">
              <div className="w-12 h-12 border-4 border-[var(--accent)] border-t-transparent rounded-full animate-spin mb-4" />
              <p className="text-[var(--text-secondary)]">
                Extracting subtitles and generating notes...
              </p>
              <p className="text-xs text-[var(--text-secondary)] mt-1">
                This may take 30-60 seconds depending on video length
              </p>
            </div>
          )}

          {/* Result */}
          {result && (
            <div className="bg-[var(--bg-secondary)] rounded-2xl border border-[var(--border)]">
              <div className="flex items-center justify-between px-6 py-4 border-b border-[var(--border)]">
                <div>
                  <h2 className="font-semibold text-lg">{result.title}</h2>
                  <div className="flex gap-3 text-xs text-[var(--text-secondary)] mt-1">
                    <span>Platform: {result.platform}</span>
                    <span>Source: {result.source}</span>
                    {result.duration > 0 && (
                      <span>Duration: {Math.round(result.duration / 60)} min</span>
                    )}
                  </div>
                </div>
                <div className="flex gap-2">
                  <button
                    onClick={handleCopy}
                    className="px-3 py-1.5 rounded-lg border border-[var(--border)] hover:bg-[var(--bg-primary)] transition text-sm"
                  >
                    Copy
                  </button>
                  <button
                    onClick={handleDownload}
                    className="px-3 py-1.5 rounded-lg border border-[var(--border)] hover:bg-[var(--bg-primary)] transition text-sm"
                  >
                    Download .md
                  </button>
                </div>
              </div>
              <div className="p-6 note-content prose prose-slate dark:prose-invert max-w-none">
                <ReactMarkdown>{result.content}</ReactMarkdown>
              </div>
            </div>
          )}

          {/* Features Section */}
          {!result && !loading && (
            <div className="grid grid-cols-1 md:grid-cols-3 gap-4 mt-8">
              <FeatureCard
                icon="🌐"
                title="30+ Platforms"
                desc="Bilibili, YouTube, Douyin, Xiaohongshu, TikTok, and 1800+ more"
              />
              <FeatureCard
                icon="📋"
                title="13 Templates"
                desc="Notes, mind maps, flashcards, quizzes, exam prep, and more"
              />
              <FeatureCard
                icon="🎯"
                title="Smart Extraction"
                desc="Three-level subtitle fallback: CC subs, ASR, visual mode"
              />
              <FeatureCard
                icon="📦"
                title="Batch Processing"
                desc="Process entire playlists and courses with 50+ videos"
              />
              <FeatureCard
                icon="🔌"
                title="Multiple Interfaces"
                desc="Web, CLI, MCP Server, OpenClaw Skill, Desktop, and API"
              />
              <FeatureCard
                icon="🌏"
                title="YouTube Proxy"
                desc="Built-in proxy support for accessing YouTube from China"
              />
            </div>
          )}
        </main>

        {/* Footer */}
        <footer className="border-t border-[var(--border)] mt-16">
          <div className="max-w-6xl mx-auto px-4 py-6 text-center text-sm text-[var(--text-secondary)]">
            NoteKing - Open Source Video to Learning Notes Tool |{" "}
            <a
              href="https://github.com/bcefghj/noteking"
              className="text-[var(--accent)] hover:underline"
            >
              GitHub
            </a>
          </div>
        </footer>
      </div>
    </div>
  );
}

function FeatureCard({
  icon,
  title,
  desc,
}: {
  icon: string;
  title: string;
  desc: string;
}) {
  return (
    <div className="bg-[var(--bg-secondary)] rounded-xl p-5 border border-[var(--border)]">
      <span className="text-2xl">{icon}</span>
      <h3 className="font-semibold mt-2">{title}</h3>
      <p className="text-sm text-[var(--text-secondary)] mt-1">{desc}</p>
    </div>
  );
}
