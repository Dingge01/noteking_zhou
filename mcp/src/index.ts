#!/usr/bin/env node
/**
 * NoteKing MCP Server
 *
 * Provides 12+ tools for video/blog to notes conversion.
 * Supports STDIO and HTTP/SSE transport modes.
 * Compatible with Cursor, Claude Desktop, OpenClaw, Codex.
 */

import { McpServer } from "@modelcontextprotocol/sdk/server/mcp.js";
import { StdioServerTransport } from "@modelcontextprotocol/sdk/server/stdio.js";
import { z } from "zod";
import { execSync } from "child_process";

const server = new McpServer({
  name: "noteking",
  version: "0.1.0",
});

// Helper: call the Python core engine
function callCore(args: string[]): string {
  try {
    const result = execSync(
      `python -m cli ${args.map((a) => `"${a}"`).join(" ")}`,
      {
        encoding: "utf-8",
        timeout: 600_000,
        cwd: process.env.NOTEKING_DIR || ".",
        maxBuffer: 50 * 1024 * 1024,
      }
    );
    return result;
  } catch (err: any) {
    return `Error: ${err.message || err}`;
  }
}

function callAPI(endpoint: string, body: Record<string, any>): string {
  const apiBase = process.env.NOTEKING_API || "http://127.0.0.1:8000";
  try {
    const result = execSync(
      `curl -s -X POST "${apiBase}${endpoint}" -H "Content-Type: application/json" -d '${JSON.stringify(body)}'`,
      { encoding: "utf-8", timeout: 600_000 }
    );
    return result;
  } catch (err: any) {
    return `Error: ${err.message || err}`;
  }
}

// ---------- Tools ----------

const TEMPLATES = [
  "brief",
  "detailed",
  "mindmap",
  "flashcard",
  "quiz",
  "timeline",
  "exam",
  "tutorial",
  "news",
  "podcast",
  "xhs_note",
  "latex_pdf",
  "custom",
];

server.tool(
  "summarize_video",
  "Convert a video/blog URL into structured notes using the specified template",
  {
    url: z.string().describe("Video URL or local file path"),
    template: z
      .enum(TEMPLATES as [string, ...string[]])
      .default("detailed")
      .describe("Output template"),
    custom_prompt: z
      .string()
      .optional()
      .describe("Custom prompt for 'custom' template"),
  },
  async ({ url, template, custom_prompt }) => {
    const body: Record<string, any> = { url, template };
    if (custom_prompt) body.custom_prompt = custom_prompt;
    const result = callAPI("/api/v1/summarize", body);
    try {
      const parsed = JSON.parse(result);
      return {
        content: [
          {
            type: "text" as const,
            text: `# ${parsed.title || "Video Notes"}\n\n${parsed.content || result}`,
          },
        ],
      };
    } catch {
      return { content: [{ type: "text" as const, text: result }] };
    }
  }
);

server.tool(
  "batch_summarize",
  "Process a playlist/collection/series and generate notes for all videos",
  {
    url: z.string().describe("Playlist or collection URL"),
    template: z
      .enum(TEMPLATES as [string, ...string[]])
      .default("detailed")
      .describe("Output template"),
  },
  async ({ url, template }) => {
    const result = callAPI("/api/v1/batch", { url, template });
    try {
      const parsed = JSON.parse(result);
      return {
        content: [
          {
            type: "text" as const,
            text: `# Batch Results (${parsed.completed}/${parsed.total})\n\n${parsed.content || result}`,
          },
        ],
      };
    } catch {
      return { content: [{ type: "text" as const, text: result }] };
    }
  }
);

server.tool(
  "get_transcript",
  "Extract subtitles/transcript text from a video",
  {
    url: z.string().describe("Video URL"),
  },
  async ({ url }) => {
    const result = callAPI(`/api/v1/transcript?url=${encodeURIComponent(url)}`, {});
    try {
      const parsed = JSON.parse(result);
      return {
        content: [
          { type: "text" as const, text: parsed.transcript || result },
        ],
      };
    } catch {
      return { content: [{ type: "text" as const, text: result }] };
    }
  }
);

server.tool(
  "get_video_info",
  "Get video metadata (title, duration, chapters, subtitle availability)",
  {
    url: z.string().describe("Video URL"),
  },
  async ({ url }) => {
    try {
      const result = execSync(
        `curl -s "${process.env.NOTEKING_API || "http://127.0.0.1:8000"}/api/v1/info?url=${encodeURIComponent(url)}"`,
        { encoding: "utf-8", timeout: 60_000 }
      );
      return { content: [{ type: "text" as const, text: result }] };
    } catch (err: any) {
      return {
        content: [{ type: "text" as const, text: `Error: ${err.message}` }],
      };
    }
  }
);

server.tool(
  "search_in_transcript",
  "Search for specific text within a video transcript",
  {
    url: z.string().describe("Video URL"),
    query: z.string().describe("Search query"),
  },
  async ({ url, query }) => {
    const transcriptResult = callAPI(
      `/api/v1/transcript?url=${encodeURIComponent(url)}`,
      {}
    );
    try {
      const parsed = JSON.parse(transcriptResult);
      const text = parsed.transcript || "";
      const lines = text.split("\n");
      const matches = lines.filter((l: string) =>
        l.toLowerCase().includes(query.toLowerCase())
      );
      return {
        content: [
          {
            type: "text" as const,
            text: matches.length
              ? `Found ${matches.length} matches:\n\n${matches.join("\n")}`
              : `No matches found for "${query}"`,
          },
        ],
      };
    } catch {
      return {
        content: [{ type: "text" as const, text: transcriptResult }],
      };
    }
  }
);

server.tool(
  "answer_from_video",
  "Answer a question based on video content",
  {
    url: z.string().describe("Video URL"),
    question: z.string().describe("Question to answer"),
  },
  async ({ url, question }) => {
    const result = callAPI("/api/v1/summarize", {
      url,
      template: "custom",
      custom_prompt: `请根据视频内容回答以下问题:\n\n${question}\n\n要求: 引用视频中的具体内容来支持你的回答。如果视频中没有相关信息，请明确说明。`,
    });
    try {
      const parsed = JSON.parse(result);
      return {
        content: [{ type: "text" as const, text: parsed.content || result }],
      };
    } catch {
      return { content: [{ type: "text" as const, text: result }] };
    }
  }
);

server.tool(
  "list_templates",
  "List all available output templates with descriptions",
  {},
  async () => {
    const templateList = TEMPLATES.map((t) => `- **${t}**`).join("\n");
    return {
      content: [
        {
          type: "text" as const,
          text: `# Available Templates\n\n${templateList}\n\nUse with summarize_video tool's template parameter.`,
        },
      ],
    };
  }
);

// ---------- Start ----------

async function main() {
  const transport = new StdioServerTransport();
  await server.connect(transport);
  console.error("NoteKing MCP Server running on STDIO");
}

main().catch(console.error);
