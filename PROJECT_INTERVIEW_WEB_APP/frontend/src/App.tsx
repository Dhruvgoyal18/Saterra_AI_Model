import { useEffect, useState } from "react";
import "./App.css";
import { detectArtifacts, displayImageUrl, fetchImages } from "./api";
import DetectionCanvas from "./DetectionCanvas";
import SummaryPanel from "./SummaryPanel";
import type { DetectResponse, ImageInfo } from "./types";

type LoadState = "idle" | "loading" | "error";

function App() {
  const [images, setImages] = useState<ImageInfo[]>([]);
  const [selectedId, setSelectedId] = useState<string | null>(null);
  const [detection, setDetection] = useState<DetectResponse | null>(null);
  const [detectState, setDetectState] = useState<LoadState>("idle");
  const [detectError, setDetectError] = useState<string | null>(null);
  const [showOverlay, setShowOverlay] = useState(true);
  const [showUncertainty, setShowUncertainty] = useState(false);
  const [overlayAlpha, setOverlayAlpha] = useState(0.45);
  const [imagesError, setImagesError] = useState<string | null>(null);

  useEffect(() => {
    fetchImages()
      .then((res) => {
        setImages(res.images);
        if (res.images.length > 0) setSelectedId(res.images[0].image_id);
      })
      .catch((err) => setImagesError(String(err)));
  }, []);

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

  return (
    <div className="app">
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
                {img.image_id} {img.labeled ? "(labeled)" : "(unannotated test image)"}
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
          {selectedImage && (
            <DetectionCanvas
              displayImageUrl={displayImageUrl(selectedImage.image_id)}
              imageInfo={selectedImage}
              detection={detection}
              showOverlay={showOverlay}
              showUncertainty={showUncertainty}
              overlayAlpha={overlayAlpha}
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
        </div>
      </div>
    </div>
  );
}

export default App;
