import { useEffect, useRef } from "react";
import type { DetectResponse, ImageInfo } from "./types";
import { confidenceToRgba, isUncertain } from "./colormap";

interface Props {
  displayImageUrl: string;
  imageInfo: ImageInfo | null;
  detection: DetectResponse | null;
  showOverlay: boolean;
  showUncertainty: boolean;
  overlayAlpha: number;
}

// Everything -- the base display image and the confidence-grid overlay -- is
// drawn into ONE canvas at the image's native display resolution. That's
// deliberate: it means the overlay rectangles and the image pixels they sit
// on are baked into the same bitmap, so CSS responsively resizing the canvas
// (width: 100%, height: auto) scales both together and they can never drift
// apart the way two separately-positioned/resized layers (an <img> plus an
// absolutely-positioned <canvas> on top) could.
export default function DetectionCanvas({
  displayImageUrl,
  imageInfo,
  detection,
  showOverlay,
  showUncertainty,
  overlayAlpha,
}: Props) {
  const canvasRef = useRef<HTMLCanvasElement>(null);

  useEffect(() => {
    const canvas = canvasRef.current;
    if (!canvas || !imageInfo) return;

    const [displayW, displayH] = imageInfo.display_dims;
    canvas.width = displayW;
    canvas.height = displayH;

    const ctx = canvas.getContext("2d");
    if (!ctx) return;

    const img = new Image();
    img.crossOrigin = "anonymous";
    img.onload = () => {
      ctx.clearRect(0, 0, displayW, displayH);
      ctx.drawImage(img, 0, 0, displayW, displayH);

      if (showOverlay && detection) {
        drawOverlay(ctx, detection, displayW, displayH, showUncertainty, overlayAlpha);
      }
    };
    img.src = displayImageUrl;
  }, [displayImageUrl, imageInfo, detection, showOverlay, showUncertainty, overlayAlpha]);

  return (
    <canvas
      ref={canvasRef}
      style={{ width: "100%", height: "auto", display: "block", borderRadius: 8 }}
    />
  );
}

function drawOverlay(
  ctx: CanvasRenderingContext2D,
  detection: DetectResponse,
  displayW: number,
  displayH: number,
  showUncertainty: boolean,
  overlayAlpha: number
) {
  const [rows, cols] = detection.grid_shape;
  const [level0W, level0H] = detection.level0_dims;
  const tileSize = detection.tile_size;

  // Exact mapping: the display PNG is a pure downsample of level 0 (see
  // scripts/prepare_display_images.py), so a single scale factor per axis
  // maps a level-0 tile rectangle straight onto display pixels -- no
  // approximation, no drift.
  const scaleX = displayW / level0W;
  const scaleY = displayH / level0H;
  const tileDisplayW = tileSize * scaleX;
  const tileDisplayH = tileSize * scaleY;

  for (let row = 0; row < rows; row++) {
    for (let col = 0; col < cols; col++) {
      const confidence = detection.tile_confidence[row][col];
      if (confidence === 0) continue; // tile wasn't evaluated (background) -- leave it clear

      const x = col * tileDisplayW;
      const y = row * tileDisplayH;

      if (showUncertainty && isUncertain(confidence)) {
        ctx.fillStyle = "rgba(250, 204, 21, 0.55)"; // amber highlight for uncertain tiles
        ctx.fillRect(x, y, tileDisplayW, tileDisplayH);
        ctx.strokeStyle = "rgba(120, 90, 0, 0.9)";
        ctx.lineWidth = 1;
        ctx.strokeRect(x, y, tileDisplayW, tileDisplayH);
      } else {
        ctx.fillStyle = confidenceToRgba(confidence, overlayAlpha);
        ctx.fillRect(x, y, tileDisplayW, tileDisplayH);
      }
    }
  }
}
