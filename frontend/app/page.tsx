"use client";

import { useState } from "react";

const API_URL = process.env.NEXT_PUBLIC_API_URL || "http://localhost:7860";

type Source = { source: string; page: number | null; type: string };

type Message =
  | { role: "user"; content: string }
  | {
      role: "assistant";
      content: string;
      steps: string[];
      sources: Source[];
      webUsed: boolean;
    };

const STEP_LABEL: Record<string, string> = {
  retrieve: "retrieve",
  grade_documents: "grade docs",
  web_search: "web search",
  generate: "generate",
};

export default function Page() {
  const [messages, setMessages] = useState<Message[]>([]);
  const [question, setQuestion] = useState("");
  const [loading, setLoading] = useState(false);
  const [error, setError] = useState<string | null>(null);

  async function send() {
    const q = question.trim();
    if (!q || loading) return;

    setError(null);
    setMessages((m) => [...m, { role: "user", content: q }]);
    setQuestion("");
    setLoading(true);

    try {
      const res = await fetch(`${API_URL}/chat`, {
        method: "POST",
        headers: { "Content-Type": "application/json" },
        body: JSON.stringify({ question: q }),
      });
      if (!res.ok) throw new Error(`Backend returned ${res.status}`);
      const data = await res.json();

      setMessages((m) => [
        ...m,
        {
          role: "assistant",
          content: data.answer,
          steps: data.steps || [],
          sources: data.sources || [],
          webUsed: data.web_used,
        },
      ]);
    } catch (e: any) {
      setError(
        `Couldn't reach the agent (${e.message}). Is the backend running at ${API_URL}?`
      );
    } finally {
      setLoading(false);
    }
  }

  return (
    <main className="shell">
      <div className="header">
        <h1>Adaptive Agentic RAG</h1>
        <p>Retrieves, grades its own evidence, falls back to the web, and self-checks before answering.</p>
      </div>

      {error && <div className="error">{error}</div>}

      <div className="thread">
        {messages.length === 0 && !error && (
          <div className="empty">Ask something about your ingested documents.</div>
        )}

        {messages.map((m, i) =>
          m.role === "user" ? (
            <div key={i} className="bubble user">{m.content}</div>
          ) : (
            <div key={i} className="bubble assistant">
              <div>{m.content}</div>

              {m.steps.length > 0 && (
                <div className="steps">
                  {m.steps.map((s, j) => (
                    <span key={j} className={`pill ${s}`}>
                      {STEP_LABEL[s] || s}
                    </span>
                  ))}
                </div>
              )}

              {m.sources.length > 0 && (
                <div className="sources">
                  Sources{m.webUsed ? " (includes web results)" : ""}:
                  <ul>
                    {m.sources.map((s, j) => (
                      <li key={j}>
                        [{s.type}] {s.source}
                        {s.page ? `, p. ${s.page}` : ""}
                      </li>
                    ))}
                  </ul>
                </div>
              )}
            </div>
          )
        )}

        {loading && <div className="bubble assistant">Thinking…</div>}
      </div>

      <div className="composer">
        <input
          value={question}
          onChange={(e) => setQuestion(e.target.value)}
          onKeyDown={(e) => e.key === "Enter" && send()}
          placeholder="Ask a question…"
          disabled={loading}
        />
        <button onClick={send} disabled={loading || !question.trim()}>
          Send
        </button>
      </div>
    </main>
  );
}
