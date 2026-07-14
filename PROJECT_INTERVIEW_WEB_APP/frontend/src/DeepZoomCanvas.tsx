import { useEffect, useRef } from "react";
import OpenSeadragon from "openseadragon";
import type { DetectResponse } from "./types";
import { API_BASE_URL, authHeaders } from "./api";
import { confidenceToRgba, DEFAULT_OVERLAY_COLOR, isUncertain, type OverlayColorName } from "./colormap";

interface Props {
  imageId: string | null;
  detection: DetectResponse | null;
  showOverlay: boolean;
  showUncertainty: boolean;
  overlayAlpha: number;
  overlayColor?: OverlayColorName;
}

// Overlay canvas resolution: pixels per grid cell. Grid is at most ~144x144
// cells (9000px image / 64px classifier tile), so 8px/cell keeps the overlay
// canvas comfortably small (~1150px on the long side) while still looking
// crisp at typical zoom levels -- this is scaled by OSD via CSS on top, the
// same accepted v1 tradeoff as DetectionCanvas's flat-canvas rendering
// (solid-color confidence blocks don't need native-resolution redraw per
// frame; see the plan's "accepted v1 scope cut" if this ever looks soft).
const OVERLAY_PX_PER_CELL = 8;

// The deep-zoom counterpart to DetectionCanvas.tsx -- renders the same image
// via a real Deep Zoom (DZI) tile pyramid instead of one flat downsampled
// PNG, so panning/zooming works at native resolution without ever loading a
// single ~9000x9000 bitmap into the browser at once. DetectionCanvas.tsx is
// left completely untouched; this is a new sibling component, switched to
// by a viewMode toggle in App.tsx, specifically so this feature can be
// abandoned with zero risk to the existing flat-canvas path.
//
// Takes imageId (not an object URL/blob) -- OpenSeadragon builds its own DZI
// descriptor + tile request URLs internally; there's no single blob for it
// to consume the way DetectionCanvas's <img> load has.
export default function DeepZoomCanvas({
  imageId,
  detection,
  showOverlay,
  showUncertainty,
  overlayAlpha,
  overlayColor = DEFAULT_OVERLAY_COLOR,
}: Props) {
  const containerRef = useRef<HTMLDivElement>(null);
  const viewerRef = useRef<OpenSeadragon.Viewer | null>(null);
  const overlayCanvasRef = useRef<HTMLCanvasElement | null>(null);

  // Latest prop values, readable from inside the viewer-construction effect
  // below without adding them to its dependency array (which would tear
  // down and rebuild the whole OSD viewer -- an expensive operation -- just
  // because the user toggled a checkbox). Declared here, before the effect
  // that closes over them, purely for readability -- since the "open"
  // handler only runs asynchronously after this render's synchronous body
  // has already finished, the effect below could equally reference refs
  // declared after it with identical runtime behavior.
  const detectionRef = useRef(detection);
  const showOverlayRef = useRef(showOverlay);
  const showUncertaintyRef = useRef(showUncertainty);
  const overlayAlphaRef = useRef(overlayAlpha);
  const overlayColorRef = useRef(overlayColor);
  detectionRef.current = detection;
  showOverlayRef.current = showOverlay;
  showUncertaintyRef.current = showUncertainty;
  overlayAlphaRef.current = overlayAlpha;
  overlayColorRef.current = overlayColor;

  useEffect(() => {
    if (!containerRef.current || !imageId) return;

    let cancelled = false;

    // Bearer token is fetched fresh per viewer construction (same "never
    // cache the token" discipline as api.ts's authHeaders callers) and
    // handed to OpenSeadragon's own ajaxHeaders/loadTilesWithAjax option --
    // this is what lets tile requests carry the Authorization header a
    // plain <img>-based tile loader can't attach. A viewer instance holds
    // this token for its whole lifetime; since a fresh viewer is
    // constructed on every image switch (see the cleanup below), switching
    // images also re-supplies a non-expired token.
    authHeaders().then((headers) => {
      if (cancelled || !containerRef.current) return;

      const viewer = OpenSeadragon({
        element: containerRef.current,
        tileSources: `${API_BASE_URL}/images/${imageId}/dzi.dzi`,
        loadTilesWithAjax: true,
        ajaxHeaders: headers,
        crossOriginPolicy: "Anonymous",
        showNavigator: true,
        gestureSettingsMouse: { clickToZoom: false },
        // OSD's default nav-button icons (zoom in/out, home, fullpage) are
        // referenced relative to the page URL by default, which 404s in a
        // Vite app -- copied into public/ once (frontend/public/
        // openseadragon-images/, sourced from the openseadragon npm
        // package's own build/openseadragon/images/) and pointed at here.
        prefixUrl: "/openseadragon-images/",
      });
      viewerRef.current = viewer;

      // The confidence overlay is one full-image <canvas>, not one
      // DOM-positioned overlay per grid cell (a full grid can be up to
      // ~140x140 = ~19,600 cells -- that many absolutely-positioned DOM
      // nodes would be a real, avoidable perf regression vs. today's
      // single fillRect-loop-in-canvas approach). Added once the viewer has
      // actually opened the image (so tiledImage.getBounds() reflects the
      // real image's aspect ratio, not a default), positioned to cover the
      // whole image via OSD's own viewport-coordinate bounds rather than
      // hand-derived normalized coordinates -- getBounds() already accounts
      // for non-square aspect ratios correctly.
      viewer.addHandler("open", () => {
        if (cancelled) return;
        const tiledImage = viewer.world.getItemAt(0);
        if (!tiledImage) return;

        const canvas = document.createElement("canvas");
        canvas.style.width = "100%";
        canvas.style.height = "100%";
        overlayCanvasRef.current = canvas;
        viewer.addOverlay(canvas, tiledImage.getBounds());
        redrawOverlay(
          canvas,
          detectionRef.current,
          showOverlayRef.current,
          showUncertaintyRef.current,
          overlayAlphaRef.current,
          overlayColorRef.current
        );
      });
    });

    return () => {
      cancelled = true;
      overlayCanvasRef.current = null;
      viewerRef.current?.destroy();
      viewerRef.current = null;
    };
    // eslint-disable-next-line react-hooks/exhaustive-deps -- only imageId should tear down/rebuild the viewer; overlay content updates are handled by the effect below via refs, not by rebuilding the whole viewer.
  }, [imageId]);

  // Redraw the (already-added) overlay canvas whenever the data or toggles
  // change -- this does NOT touch the OSD viewer/tile layer at all, so
  // toggling "show overlay" or dragging the opacity slider never triggers a
  // re-request of any base image tile.
  useEffect(() => {
    const canvas = overlayCanvasRef.current;
    if (!canvas) return;
    redrawOverlay(canvas, detection, showOverlay, showUncertainty, overlayAlpha, overlayColor);
  }, [detection, showOverlay, showUncertainty, overlayAlpha, overlayColor]);

  return (
    <div
      ref={containerRef}
      style={{ width: "100%", aspectRatio: "1 / 1", borderRadius: 8, overflow: "hidden", background: "#111" }}
    />
  );
}

function redrawOverlay(
  canvas: HTMLCanvasElement,
  detection: DetectResponse | null,
  showOverlay: boolean,
  showUncertainty: boolean,
  overlayAlpha: number,
  overlayColor: OverlayColorName
) {
  if (!detection || !showOverlay) {
    canvas.width = 0;
    canvas.height = 0;
    return;
  }

  const [rows, cols] = detection.grid_shape;
  canvas.width = cols * OVERLAY_PX_PER_CELL;
  canvas.height = rows * OVERLAY_PX_PER_CELL;

  const ctx = canvas.getContext("2d");
  if (!ctx) return;
  ctx.clearRect(0, 0, canvas.width, canvas.height);

  const tileW = canvas.width / cols;
  const tileH = canvas.height / rows;

  for (let row = 0; row < rows; row++) {
    for (let col = 0; col < cols; col++) {
      const confidence = detection.tile_confidence[row][col];
      if (confidence === 0) continue; // tile wasn't evaluated (background) -- leave it clear

      const x = col * tileW;
      const y = row * tileH;

      if (showUncertainty && isUncertain(confidence)) {
        ctx.fillStyle = "rgba(250, 204, 21, 0.55)"; // amber highlight for uncertain tiles
        ctx.fillRect(x, y, tileW, tileH);
        ctx.strokeStyle = "rgba(120, 90, 0, 0.9)";
        ctx.lineWidth = 1;
        ctx.strokeRect(x, y, tileW, tileH);
      } else {
        ctx.fillStyle = confidenceToRgba(confidence, overlayAlpha, overlayColor);
        ctx.fillRect(x, y, tileW, tileH);
      }
    }
  }
}
