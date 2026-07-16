import type { Session } from "@supabase/supabase-js";
import { useEffect, useState } from "react";
import "./App.css";
import { detectArtifacts, fetchDisplayImageObjectUrl, fetchImages } from "./api";
import ChatHistoryPanel from "./ChatHistoryPanel";
import ChatPanel from "./ChatPanel";
import { DEFAULT_OVERLAY_COLOR, OVERLAY_COLOR_PALETTE, type OverlayColorName } from "./colormap";
import DeepZoomCanvas from "./DeepZoomCanvas";
import DetectionCanvas from "./DetectionCanvas";
import LoginForm from "./LoginForm";
import SummaryPanel from "./SummaryPanel";
import { supabase } from "./supabaseClient";
import type { DetectResponse, ImageInfo } from "./types";

type LoadState = "idle" | "loading" | "error";

function App() {
  const [session, setSession] = useState<Session | null>(null);
  const [authChecked, setAuthChecked] = useState(false);

  useEffect(() => {
    supabase.auth.getSession().then(({ data: { session } }) => {
      setSession(session);
      setAuthChecked(true);
    });
    const {
      data: { subscription },
    } = supabase.auth.onAuthStateChange((_event, session) => {
      setSession(session);
    });
    return () => subscription.unsubscribe();
  }, []);

  if (!authChecked) return null;
  if (!session) return <LoginForm />;

  return <DetectorApp userEmail={session.user.email ?? ""} />;
}

function DetectorApp({ userEmail }: { userEmail: string }) {
  const [images, setImages] = useState<ImageInfo[]>([]);
  const [selectedId, setSelectedId] = useState<string | null>(null);
  const [imageObjectUrl, setImageObjectUrl] = useState<string | null>(null);
  const [detection, setDetection] = useState<DetectResponse | null>(null);
  const [detectState, setDetectState] = useState<LoadState>("idle");
  const [detectError, setDetectError] = useState<string | null>(null);
  const [showOverlay, setShowOverlay] = useState(true);
  const [showUncertainty, setShowUncertainty] = useState(false);
  const [overlayAlpha, setOverlayAlpha] = useState(0.45);
  const [imagesError, setImagesError] = useState<string | null>(null);
  // Additive: default "flat" preserves today's exact behavior. "deepzoom"
  // switches to DeepZoomCanvas (a real pan/zoom tile pyramid) instead of the
  // single flat downsampled PNG -- DetectionCanvas.tsx itself is untouched,
  // so this feature can be abandoned by simply never flipping this toggle.
  const [viewMode, setViewMode] = useState<"flat" | "deepzoom">("flat");
  // Additive, same "default preserves today's behavior" discipline as
  // viewMode above: DEFAULT_OVERLAY_COLOR matches the old hardcoded RED
  // exactly, so leaving this untouched reproduces today's colors pixel-for-
  // pixel.
  const [overlayColor, setOverlayColor] = useState<OverlayColorName>(DEFAULT_OVERLAY_COLOR);

  useEffect(() => {
    fetchImages()
      .then((res) => {
        setImages(res.images);
        if (res.images.length > 0) setSelectedId(res.images[0].image_id);
      })
      .catch((err) => setImagesError(String(err)));
  }, []);

  // The display endpoint is auth-gated, so it's fetched (with the auth
  // header) into a blob object URL here rather than passed as a plain API
  // URL -- see api.ts's fetchDisplayImageObjectUrl and DetectionCanvas.tsx.
  // Each selection's object URL is revoked once superseded, since object
  // URLs otherwise leak for the life of the page.
  useEffect(() => {
    if (!selectedId) {
      setImageObjectUrl(null);
      return;
    }
    let cancelled = false;
    let objectUrl: string | null = null;
    fetchDisplayImageObjectUrl(selectedId)
      .then((url) => {
        if (cancelled) {
          URL.revokeObjectURL(url);
          return;
        }
        objectUrl = url;
        setImageObjectUrl(url);
      })
      .catch((err) => setImagesError(String(err)));
    return () => {
      cancelled = true;
      if (objectUrl) URL.revokeObjectURL(objectUrl);
    };
  }, [selectedId]);

  const selectedImage = images.find((img) => img.image_id === selectedId) ?? null;

  function handleSelectImage(imageId: string) {
    setSelectedId(imageId);
    setDetection(null);
    setDetectState("idle");
    setDetectError(null);
  }

  async function handleDetect() {
    if (!selectedId) return;
    setDetectState("loading");
    setDetectError(null);
    try {
      const result = await detectArtifacts(selectedId);
      setDetection(result);
      setDetectState("idle");
    } catch (err) {
      setDetectError(String(err instanceof Error ? err.message : err));
      setDetectState("error");
    }
  }

  // Checkpoint 3: the chat panel's tool call ran the model for some image_id
  // (not necessarily whatever's currently selected in the dropdown) -- switch
  // the selection to match, so the existing imageObjectUrl effect (keyed on
  // selectedId) fetches the right display image, and feed the result into
  // the same `detection` state the button's handleDetect uses. Same
  // DetectionCanvas/SummaryPanel rendering path, different trigger.
  function handleChatDetectionResult(result: DetectResponse) {
    setSelectedId(result.image_id);
    setDetection(result);
    setDetectState("idle");
    setDetectError(null);
  }

  return (
    <div className="app">
      <div className="user-bar">
        <span>{userEmail}</span>
        <button onClick={() => supabase.auth.signOut()}>Sign out</button>
      </div>

      <header>
        <h1>DAPI Artifact Detector</h1>
        <p className="subtitle">
          Tile-classification model from Checkpoint 1, served from a private Hugging Face dataset
          repo + cloud inference endpoint. The full-resolution image never reaches the browser --
          only a downsampled display PNG and a small confidence grid do.
        </p>
      </header>

      {imagesError && <div className="banner error">Failed to load image list: {imagesError}</div>}

      <div className="controls">
        <label>
          Image:
          <select value={selectedId ?? ""} onChange={(e) => handleSelectImage(e.target.value)}>
            {images.map((img) => (
              <option key={img.image_id} value={img.image_id}>
                {img.image_id}
              </option>
            ))}
          </select>
        </label>

        <button onClick={handleDetect} disabled={!selectedId || detectState === "loading"}>
          {detectState === "loading" ? "Detecting..." : "Detect artifacts"}
        </button>

        <label className="checkbox">
          <input type="checkbox" checked={showOverlay} onChange={(e) => setShowOverlay(e.target.checked)} />
          Show confidence overlay
        </label>

        <label className="checkbox">
          <input
            type="checkbox"
            checked={showUncertainty}
            onChange={(e) => setShowUncertainty(e.target.checked)}
          />
          Highlight uncertain tiles (0.4-0.6)
        </label>

        <label className="slider">
          Overlay opacity
          <input
            type="range"
            min={0.1}
            max={0.9}
            step={0.05}
            value={overlayAlpha}
            onChange={(e) => setOverlayAlpha(Number(e.target.value))}
          />
        </label>

        <label className="checkbox">
          <input
            type="checkbox"
            checked={viewMode === "deepzoom"}
            onChange={(e) => setViewMode(e.target.checked ? "deepzoom" : "flat")}
          />
          Deep-zoom view (experimental)
        </label>

        <label>
          Artifact color:
          <select value={overlayColor} onChange={(e) => setOverlayColor(e.target.value as OverlayColorName)}>
            {Object.keys(OVERLAY_COLOR_PALETTE).map((name) => (
              <option key={name} value={name}>
                {name}
              </option>
            ))}
          </select>
        </label>
      </div>

      {detectState === "loading" && (
        <div className="banner info">
          Running inference -- if the service has been idle, this can take 30-60s for a cold
          start (model load + image fetch + tiling ~20k tiles on CPU). Subsequent requests are
          much faster.
        </div>
      )}
      {detectState === "error" && <div className="banner error">{detectError}</div>}

      <div className="main-content">
        <div className="canvas-wrapper">
          {selectedImage && viewMode === "flat" && (
            <DetectionCanvas
              imageObjectUrl={imageObjectUrl}
              detection={detection}
              showOverlay={showOverlay}
              showUncertainty={showUncertainty}
              overlayAlpha={overlayAlpha}
              overlayColor={overlayColor}
            />
          )}
          {selectedImage && viewMode === "deepzoom" && (
            <DeepZoomCanvas
              imageId={selectedId}
              detection={detection}
              showOverlay={showOverlay}
              showUncertainty={showUncertainty}
              overlayAlpha={overlayAlpha}
              overlayColor={overlayColor}
            />
          )}
        </div>

        <div className="side-panel">
          <div className="legend">
            <div className="legend-title">Confidence</div>
            <div className="legend-gradient" />
            <div className="legend-labels">
              <span>0.0 clean</span>
              <span>0.5 uncertain</span>
              <span>1.0 artifact</span>
            </div>
          </div>

          {detection && <SummaryPanel summary={detection.summary} />}

          <ChatPanel onDetectionResult={handleChatDetectionResult} />

          <ChatHistoryPanel />
        </div>
      </div>
    </div>
  );
}

export default App;
