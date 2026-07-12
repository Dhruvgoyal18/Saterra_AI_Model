import { useState } from "react";
import { sendChatMessage } from "./api";
import type { DetectResponse, DisplayOptions } from "./types";

interface ChatBubble {
  role: "user" | "assistant";
  content: string;
}

interface Props {
  // Called when a turn's tool call actually ran the model -- the parent
  // feeds this into the same selectedId/detection state the button's click
  // handler uses, so the result renders through the exact same
  // DetectionCanvas/SummaryPanel components, not a second rendering path.
  onDetectionResult: (result: DetectResponse) => void;
  // Called when a turn's tool call was set_display_options instead -- a
  // rendering-preference change (overlay color/visibility/view mode), never
  // a detection.
  onDisplayOptions: (options: DisplayOptions) => void;
}

export default function ChatPanel({ onDetectionResult, onDisplayOptions }: Props) {
  const [messages, setMessages] = useState<ChatBubble[]>([]);
  const [input, setInput] = useState("");
  const [conversationId, setConversationId] = useState<string | null>(null);
  const [sending, setSending] = useState(false);
  const [error, setError] = useState<string | null>(null);

  async function handleSend(e: React.FormEvent) {
    e.preventDefault();
    const text = input.trim();
    if (!text || sending) return;

    setMessages((prev) => [...prev, { role: "user", content: text }]);
    setInput("");
    setSending(true);
    setError(null);

    try {
      const res = await sendChatMessage(text, conversationId);
      setConversationId(res.conversation_id);
      setMessages((prev) => [...prev, { role: "assistant", content: res.reply }]);
      if (res.detection_result) {
        onDetectionResult(res.detection_result);
      }
      if (res.display_options) {
        onDisplayOptions(res.display_options);
      }
    } catch (err) {
      setError(String(err instanceof Error ? err.message : err));
    } finally {
      setSending(false);
    }
  }

  return (
    <div className="chat-panel">
      <div className="chat-title">Ask the assistant</div>
      <p className="subtitle chat-hint">
        e.g. "Run artifact detection on Image_7" -- results render in the canvas above, same as
        clicking the button. Ask anything outside this dataset/tool and it will decline rather
        than guess.
      </p>

      <div className="chat-messages">
        {messages.length === 0 && <div className="chat-empty">No messages yet.</div>}
        {messages.map((m, i) => (
          <div key={i} className={`chat-bubble chat-${m.role}`}>
            {m.content}
          </div>
        ))}
        {sending && <div className="chat-bubble chat-assistant chat-pending">Thinking...</div>}
      </div>

      {error && <div className="banner error">{error}</div>}

      <form onSubmit={handleSend} className="chat-input-row">
        <input
          type="text"
          value={input}
          onChange={(e) => setInput(e.target.value)}
          placeholder="Ask about an image..."
          disabled={sending}
        />
        <button type="submit" disabled={sending || !input.trim()}>
          Send
        </button>
      </form>
    </div>
  );
}
