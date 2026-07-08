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
