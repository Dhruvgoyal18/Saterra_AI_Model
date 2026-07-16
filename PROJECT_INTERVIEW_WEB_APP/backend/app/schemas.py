from dataclasses import dataclass

from pydantic import BaseModel


@dataclass
class AuthedUser:
    """What main.py's require_user resolves a valid Supabase session down to.

    Lives here, not in main.py, so both main.py and chat.py can import it
    without a circular import (main.py imports chat as chat_module). `token`
    is the raw bearer token itself -- needed alongside the user id so
    chat.py/storage/chat_storage.py can forward it to Supabase's PostgREST
    API, where it's what lets Row Level Security resolve auth.uid() to this
    real user rather than the anon key's no-user-at-all identity.
    """

    id: str
    token: str
    raw: dict


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


class ChatSessionSummary(BaseModel):
    id: str
    title: str | None
    created_at: str
    updated_at: str


class ChatMessageRecord(BaseModel):
    id: int
    role: str
    content: str | None
    # Non-null only for a role="tool" row -- the tool actually invoked plus
    # its arguments and raw result, stored durably so a past session's
    # transcript can show exactly what ran, not just the assistant's prose
    # summary of it (see chat.py's SYSTEM_PROMPT: "never claim a detection
    # ran if the tool wasn't actually called").
    tool_name: str | None = None
    tool_args: dict | None = None
    tool_result: dict | None = None
    created_at: str


class ChatSessionDetail(ChatSessionSummary):
    messages: list[ChatMessageRecord]
