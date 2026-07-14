import { useEffect, useRef } from "react";
import type { DetectResponse } from "./types";
import { confidenceToRgba, DEFAULT_OVERLAY_COLOR, isUncertain, type OverlayColorName } from "./colormap";

interface Props {
  // An object URL (from URL.createObjectURL), not a plain API URL -- the
  // display endpoint is auth-gated and a native Image() load can't carry an
  // Authorization header, so the caller fetches it with the header attached
  // and hands this component the resulting blob's object URL instead (see
  // api.ts's fetchDisplayImageObjectUrl and App.tsx).
  imageObjectUrl: string | null;
  detection: DetectResponse | null;
  showOverlay: boolean;
  showUncertainty: boolean;
  overlayAlpha: number;
  overlayColor?: OverlayColorName;
}

// Everything -- the base display image and the confidence-grid overlay -- is
// drawn into ONE canvas at the image's native display resolution. That's
// deliberate: it means the overlay rectangles and the image pixels they sit
// on are baked into the same bitmap, so CSS responsively resizing the canvas
// (width: 100%, height: auto) scales both together and they can never drift
// apart the way two separately-positioned/resized layers (an <img> plus an
// absolutely-positioned <canvas> on top) could.
//
// Canvas dimensions come from the loaded image's own naturalWidth/
// naturalHeight, not from a pre-known display_dims value passed in from
// /images -- that field is null until the backend has generated this
// image's display PNG at least once (it's a get-or-create cache now, not a
// pre-built manifest), so the canvas can't depend on it being populated in
// advance. The overlay's scale factor still uses /detect's level0_dims,
// which is always computed fresh from the actual array at request time.
export default function DetectionCanvas({
  imageObjectUrl,
  detection,
  showOverlay,
  showUncertainty,
  overlayAlpha,
  overlayColor = DEFAULT_OVERLAY_COLOR,
}: Props) {
  const canvasRef = useRef<HTMLCanvasElement>(null);

  useEffect(() => {
    const canvas = canvasRef.current;
    if (!canvas || !imageObjectUrl) return;

    const ctx = canvas.getContext("2d");
    if (!ctx) return;

    const img = new Image();
    img.onload = () => {
      const displayW = img.naturalWidth;
      const displayH = img.naturalHeight;
      canvas.width = displayW;
      canvas.height = displayH;

      ctx.clearRect(0, 0, displayW, displayH);
      ctx.drawImage(img, 0, 0, displayW, displayH);

      if (showOverlay && detection) {
        drawOverlay(ctx, detection, displayW, displayH, showUncertainty, overlayAlpha, overlayColor);
      }
    };
    img.src = imageObjectUrl;
  }, [imageObjectUrl, detection, showOverlay, showUncertainty, overlayAlpha, overlayColor]);

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
  overlayAlpha: number,
  overlayColor: OverlayColorName
) {
  const [rows, cols] = detection.grid_shape;
  const [level0W, level0H] = detection.level0_dims;
  const tileSize = detection.tile_size;

  // Exact mapping: the display PNG is a pure downsample of level 0 (see
  // backend/app/ml/display_prep.py), so a single scale factor per axis maps
  // a level-0 tile rectangle straight onto display pixels -- no
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
        ctx.fillStyle = confidenceToRgba(confidence, overlayAlpha, overlayColor);
        ctx.fillRect(x, y, tileDisplayW, tileDisplayH);
      }
    }
  }
}
