from pydantic import BaseModel


class ImageInfo(BaseModel):
    image_id: str
    labeled: bool
    level0_dims: tuple[int, int]  # (W, H)
    display_dims: tuple[int, int]  # (w, h)


class ImagesResponse(BaseModel):
    images: list[ImageInfo]


class DetectRequest(BaseModel):
    image_id: str


class DetectSummary(BaseModel):
    total_artifact_area_px: int
    total_artifact_area_frac_of_tissue: float
    num_connected_regions: int
    largest_region_area_px: int
    n_tiles_evaluated: int
    n_tiles_total: int


class DetectResponse(BaseModel):
    image_id: str
    grid_shape: tuple[int, int]  # (rows, cols)
    tile_confidence: list[list[float]]  # rows x cols, in [0, 1]
    summary: DetectSummary
    level0_dims: tuple[int, int]  # (W, H)
    tile_size: int
