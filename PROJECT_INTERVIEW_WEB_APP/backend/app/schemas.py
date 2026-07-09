from pydantic import BaseModel


class ImageInfo(BaseModel):
    image_id: str
    # Both None until this image's display PNG has been generated at least
    # once (see storage/hf_storage.py's get_cached_dims). GET /images lists
    # images straight from the HF repo's file listing, which is cheap and
    # requires no download -- but level0_dims/display_dims can only be known
    # once an image has actually been fetched and processed, so listing
    # doesn't force that to happen for all 8 images just to answer one
    # request. The frontend sizes its canvas from the loaded display image's
    # own natural dimensions rather than depending on this field (see
    # DetectionCanvas.tsx), so these are informational, not load-bearing.
    #
    # The `labeled` field from the previous (manifest.json-based) version of
    # this schema is dropped entirely, not just defaulted: it recorded which
    # images had Checkpoint 1 ground-truth annotations, but that information
    # was never uploaded to the HF repo (only the 8 original TIFFs were) and
    # isn't derivable from listing the repo. Keeping it would mean hardcoding
    # Checkpoint-1-specific knowledge back into a pipeline whose entire point
    # is that the HF repo is the single source of truth.
    level0_dims: tuple[int, int] | None = None  # (W, H)
    display_dims: tuple[int, int] | None = None  # (w, h)


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


class ChatRequest(BaseModel):
    message: str
    # None on a conversation's first turn -- the server generates one and
    # returns it in ChatResponse; the client passes it back on subsequent
    # turns to continue the same in-memory conversation (see chat.py).
    conversation_id: str | None = None


class ChatResponse(BaseModel):
    reply: str
    # Non-null exactly when this turn's tool call actually ran the model --
    # same shape DetectResponse already has, so the frontend renders it
    # through the exact same DetectionCanvas/SummaryPanel components the
    # button's click handler uses.
    detection_result: DetectResponse | None = None
    conversation_id: str
