import { supabase } from "./supabaseClient";
import type {
  ChatApiResponse,
  ChatSessionDetail,
  ChatSessionSummary,
  DetectResponse,
  ImagesResponse,
} from "./types";

export const API_BASE_URL = import.meta.env.VITE_API_BASE_URL ?? "http://127.0.0.1:8010";

// Free-tier hosts (HF Spaces) sleep when idle -- the first request after that
// can take 30-60s (model load + bucket fetch + tiling ~20k tiles on CPU).
// This timeout is deliberately generous so a cold start doesn't look like a
// network failure; the UI shows a "waking up" message while it waits (App.tsx).
const DETECT_TIMEOUT_MS = 120_000;

// Read the session token fresh, right before each call, rather than caching
// it once at login -- Supabase access tokens expire (1hr default) and the
// client auto-refreshes the underlying session, so capturing the string
// once would silently start failing with 401s an hour into a session.
export async function authHeaders(): Promise<Record<string, string>> {
  const {
    data: { session },
  } = await supabase.auth.getSession();
  if (!session) {
    throw new Error("Not signed in");
  }
  return { Authorization: `Bearer ${session.access_token}` };
}

// Every image/detect endpoint is gated behind Supabase auth (see backend's
// require_user) -- but a plain <img src=...> / new Image() load can't carry
// a custom Authorization header, so the display image is fetched here with
// the header attached and handed to the caller as an object URL instead.
// Caller is responsible for URL.revokeObjectURL() when done with it.
export async function fetchDisplayImageObjectUrl(imageId: string): Promise<string> {
  const headers = await authHeaders();
  const res = await fetch(`${API_BASE_URL}/images/${imageId}/display`, { headers });
  if (!res.ok) {
    throw new Error(`Failed to load display image: ${res.status} ${res.statusText}`);
  }
  const blob = await res.blob();
  return URL.createObjectURL(blob);
}

export async function fetchImages(): Promise<ImagesResponse> {
  const headers = await authHeaders();
  const res = await fetch(`${API_BASE_URL}/images`, { headers });
  if (!res.ok) {
    throw new Error(`Failed to list images: ${res.status} ${res.statusText}`);
  }
  return res.json();
}

export async function detectArtifacts(imageId: string): Promise<DetectResponse> {
  const controller = new AbortController();
  const timeoutId = setTimeout(() => controller.abort(), DETECT_TIMEOUT_MS);

  try {
    const headers = { "Content-Type": "application/json", ...(await authHeaders()) };
    const res = await fetch(`${API_BASE_URL}/detect`, {
      method: "POST",
      headers,
      body: JSON.stringify({ image_id: imageId }),
      signal: controller.signal,
    });
    if (!res.ok) {
      const detail = await res.text();
      throw new Error(`Detection failed: ${res.status} ${detail}`);
    }
    return res.json();
  } catch (err) {
    if (err instanceof DOMException && err.name === "AbortError") {
      throw new Error(
        "Request timed out. The inference service may be cold-starting -- try again in a moment."
      );
    }
    throw err;
  } finally {
    clearTimeout(timeoutId);
  }
}

// Checkpoint 3: natural-language trigger for the same detection path
// detectArtifacts() above calls -- the backend's /chat may invoke the
// run_artifact_detection tool under the hood, which is just as slow as a
// direct /detect call (same cold-start/model-load/tiling cost), so this
// gets the same generous timeout.
export async function sendChatMessage(
  message: string,
  conversationId: string | null
): Promise<ChatApiResponse> {
  const controller = new AbortController();
  const timeoutId = setTimeout(() => controller.abort(), DETECT_TIMEOUT_MS);

  try {
    const headers = { "Content-Type": "application/json", ...(await authHeaders()) };
    const res = await fetch(`${API_BASE_URL}/chat`, {
      method: "POST",
      headers,
      body: JSON.stringify({ message, conversation_id: conversationId }),
      signal: controller.signal,
    });
    if (!res.ok) {
      const detail = await res.text();
      throw new Error(`Chat request failed: ${res.status} ${detail}`);
    }
    return res.json();
  } catch (err) {
    if (err instanceof DOMException && err.name === "AbortError") {
      throw new Error(
        "Request timed out. The inference service may be cold-starting -- try again in a moment."
      );
    }
    throw err;
  } finally {
    clearTimeout(timeoutId);
  }
}

// This user's past chat sessions, most-recently-active first -- for a
// history list UI. Cheap read, no cold-start risk, so no special timeout
// unlike detectArtifacts/sendChatMessage above.
export async function fetchChatSessions(): Promise<ChatSessionSummary[]> {
  const headers = await authHeaders();
  const res = await fetch(`${API_BASE_URL}/chat/sessions`, { headers });
  if (!res.ok) {
    throw new Error(`Failed to list chat sessions: ${res.status} ${res.statusText}`);
  }
  return res.json();
}

// One past session's full transcript, including tool-call arguments/results
// -- read-only, there is no way to continue chatting into this session.
export async function fetchChatSessionDetail(sessionId: string): Promise<ChatSessionDetail> {
  const headers = await authHeaders();
  const res = await fetch(`${API_BASE_URL}/chat/sessions/${sessionId}`, { headers });
  if (!res.ok) {
    throw new Error(`Failed to load chat session: ${res.status} ${res.statusText}`);
  }
  return res.json();
}
