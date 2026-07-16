import { useEffect, useState } from "react";
import { fetchChatSessionDetail, fetchChatSessions } from "./api";
import type { ChatSessionDetail, ChatSessionSummary } from "./types";

// Read-only viewer for past /chat sessions (GET /chat/sessions,
// GET /chat/sessions/{id}) -- deliberately separate from ChatPanel, which
// keeps driving the live conversation exactly as before. Opening a past
// session here never lets the user continue chatting into it: there's no
// input box and no onDetectionResult wiring, so "read-only" is enforced by
// this component's shape, not just by convention.
export default function ChatHistoryPanel() {
  const [expanded, setExpanded] = useState(false);
  const [sessions, setSessions] = useState<ChatSessionSummary[] | null>(null);
  const [selected, setSelected] = useState<ChatSessionDetail | null>(null);
  const [loading, setLoading] = useState(false);
  const [error, setError] = useState<string | null>(null);

  useEffect(() => {
    if (!expanded || sessions !== null) return;
    setLoading(true);
    setError(null);
    fetchChatSessions()
      .then(setSessions)
      .catch((err) => setError(String(err instanceof Error ? err.message : err)))
      .finally(() => setLoading(false));
  }, [expanded, sessions]);

  async function openSession(sessionId: string) {
    setLoading(true);
    setError(null);
    try {
      setSelected(await fetchChatSessionDetail(sessionId));
    } catch (err) {
      setError(String(err instanceof Error ? err.message : err));
    } finally {
      setLoading(false);
    }
  }

  return (
    <div className="chat-panel">
      <div className="chat-title chat-history-toggle" onClick={() => setExpanded((v) => !v)}>
        Past sessions {expanded ? "▾" : "▸"}
      </div>

      {expanded && !selected && (
        <>
          {loading && <div className="chat-empty">Loading...</div>}
          {error && <div className="banner error">{error}</div>}
          {sessions && sessions.length === 0 && <div className="chat-empty">No past sessions yet.</div>}
          {sessions && sessions.length > 0 && (
            <div className="chat-history-list">
              {sessions.map((s) => (
                <div key={s.id} className="chat-history-item" onClick={() => openSession(s.id)}>
                  <div className="chat-history-item-title">{s.title || "(untitled)"}</div>
                  <div className="chat-history-item-date">{new Date(s.updated_at).toLocaleString()}</div>
                </div>
              ))}
            </div>
          )}
        </>
      )}

      {expanded && selected && (
        <>
          <div className="chat-history-back" onClick={() => setSelected(null)}>
            &laquo; back to sessions
          </div>
          {error && <div className="banner error">{error}</div>}
          <div className="chat-messages">
            {selected.messages.map((m) => (
              <div key={m.id} className={`chat-bubble chat-${m.role === "tool" ? "assistant" : m.role}`}>
                {m.role === "tool" ? (
                  <div className="chat-history-tool">
                    <div className="chat-history-tool-name">tool: {m.tool_name}</div>
                    {m.tool_args && (
                      <pre className="chat-history-tool-json">{JSON.stringify(m.tool_args)}</pre>
                    )}
                    {m.tool_result && (
                      <pre className="chat-history-tool-json">{JSON.stringify(m.tool_result)}</pre>
                    )}
                  </div>
                ) : (
                  m.content
                )}
              </div>
            ))}
          </div>
        </>
      )}
    </div>
  );
}
