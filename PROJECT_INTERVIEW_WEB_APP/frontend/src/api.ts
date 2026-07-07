import type { DetectResponse, ImagesResponse } from "./types";

const API_BASE_URL = import.meta.env.VITE_API_BASE_URL ?? "http://127.0.0.1:8010";

// Free-tier hosts (HF Spaces) sleep when idle -- the first request after that
// can take 30-60s (model load + bucket fetch + tiling ~20k tiles on CPU).
// This timeout is deliberately generous so a cold start doesn't look like a
// network failure; the UI shows a "waking up" message while it waits (App.tsx).
const DETECT_TIMEOUT_MS = 120_000;

export function displayImageUrl(imageId: string): string {
  return `${API_BASE_URL}/images/${imageId}/display`;
}

export async function fetchImages(): Promise<ImagesResponse> {
  const res = await fetch(`${API_BASE_URL}/images`);
  if (!res.ok) {
    throw new Error(`Failed to list images: ${res.status} ${res.statusText}`);
  }
  return res.json();
}

export async function detectArtifacts(imageId: string): Promise<DetectResponse> {
  const controller = new AbortController();
  const timeoutId = setTimeout(() => controller.abort(), DETECT_TIMEOUT_MS);

  try {
    const res = await fetch(`${API_BASE_URL}/detect`, {
      method: "POST",
      headers: { "Content-Type": "application/json" },
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
