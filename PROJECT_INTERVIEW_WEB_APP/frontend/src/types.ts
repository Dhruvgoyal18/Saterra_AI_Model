export interface ImageInfo {
  image_id: string;
  // Both null until this image's display PNG has been generated at least
  // once on the backend (see backend/app/storage/hf_storage.py). Not used
  // for canvas sizing -- DetectionCanvas sizes itself from the loaded
  // image's own natural width/height instead, so these are informational
  // only. There is no "labeled" field anymore: that distinguished
  // Checkpoint-1-annotated images from unannotated ones via a manifest.json
  // that no longer exists -- the HF repo listing this now comes from has no
  // equivalent concept, and reintroducing it client-side would mean
  // hardcoding Checkpoint-1-specific knowledge back in.
  level0_dims: [number, number] | null; // [W, H]
  display_dims: [number, number] | null; // [w, h]
}

export interface ImagesResponse {
  images: ImageInfo[];
}

export interface DetectSummary {
  total_artifact_area_px: number;
  total_artifact_area_frac_of_tissue: number;
  num_connected_regions: number;
  largest_region_area_px: number;
  n_tiles_evaluated: number;
  n_tiles_total: number;
}

export interface DetectResponse {
  image_id: string;
  grid_shape: [number, number]; // [rows, cols]
  tile_confidence: number[][]; // rows x cols, in [0, 1]
  summary: DetectSummary;
  level0_dims: [number, number]; // [W, H]
  tile_size: number;
}

// Checkpoint 3 (new-tools): a rendering-preference change requested via chat
// (set_display_options), distinct from a detection -- never touches the
// model, only changes what's drawn. Fields are independently nullable since
// a request like "show me the artifacts in green" only sets overlay_color.
export interface DisplayOptions {
  overlay_color: string | null;
  show_overlay: boolean | null;
  show_uncertainty: boolean | null;
  view_mode: "flat" | "deepzoom" | null;
}

// Checkpoint 3: POST /chat's response. detection_result is non-null exactly
// when the assistant's tool call actually ran the model this turn -- same
// shape as DetectResponse, rendered through the same DetectionCanvas/
// SummaryPanel components the button's click handler uses. display_options
// is non-null exactly when the turn's tool call was set_display_options
// instead -- the two are mutually exclusive in practice (one tool call per
// turn today) but independently optional in the type.
export interface ChatApiResponse {
  reply: string;
  detection_result: DetectResponse | null;
  display_options: DisplayOptions | null;
  conversation_id: string;
}
