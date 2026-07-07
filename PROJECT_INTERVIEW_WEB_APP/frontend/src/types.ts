export interface ImageInfo {
  image_id: string;
  labeled: boolean;
  level0_dims: [number, number]; // [W, H]
  display_dims: [number, number]; // [w, h]
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
