"""POST /chat's LLM orchestration (Checkpoint 3).

Function-calling over the exact same detect_artifacts path /detect uses
(detection.run_detection) -- no second detection pipeline, no second model
load, no HTTP round-trip to this app's own /detect route. Uses Groq's
OpenAI-compatible chat completions API for tool calling.

Conversation history is kept in-memory, per conversation_id, for this demo
only -- no database. Same "simple, no persistence beyond what's needed"
judgment already applied to the Supabase auth layer (steps_of_implementation_
checkpoint2.md SS11): a chat demo doesn't need durable history to prove the
tool-calling path works, and a restart clearing it is an acceptable tradeoff
for what this checkpoint is actually demonstrating.
"""

import json
import logging
import uuid

from groq import Groq

from . import storage
from .config import settings
from .detection import run_detection
from .schemas import ChatResponse, DisplayOptions

logger = logging.getLogger("artifact-detector")

# Fixed, small palette for set_display_options' overlay_color -- same
# allowlist discipline already applied to image_id (storage.validate_image_id):
# once an LLM can freely generate a value from natural language, it can't be
# trusted without a fixed set to check against. Deliberately excludes "blue"
# -- that's the colormap's fixed "confidently clean" anchor (colormap.ts),
# and allowing it as the artifact color too would make the diverging
# blue-white-red-style gradient collapse into a single hue with no visual
# distinction between the two ends.
OVERLAY_COLOR_PALETTE = ["red", "green", "orange", "purple", "yellow", "pink", "teal"]
VIEW_MODES = ["flat", "deepzoom"]

# Fail fast and clearly, same pattern as the HF_TOKEN/Supabase checks in
# main.py -- /chat can't function at all without this, so surface a plain
# actionable error at startup rather than a confusing traceback on first use.
if not settings.groq_api_key:
    raise RuntimeError(
        "GROQ_API_KEY must be set -- /chat's function-calling runs through Groq's "
        "API. Set it in backend/.env for local dev, or as a secret on your "
        "hosting platform for a deployment."
    )

_client = Groq(api_key=settings.groq_api_key)
_MODEL = "llama-3.3-70b-versatile"

SYSTEM_PROMPT = (
    "You are an assistant for a DAPI whole-slide-image artifact detection tool. "
    "You have exactly three capabilities, each via its own tool: "
    "(1) run_artifact_detection -- run the trained artifact-detection model on "
    "one of the known images, returning the percentage of tissue flagged as "
    "artifact, the number of connected artifact regions, and the largest "
    "region's size; "
    "(2) list_images -- look up which image IDs actually exist, so you never "
    "have to guess or invent one; "
    "(3) set_display_options -- change how the result is DISPLAYED (overlay "
    "color, whether the overlay/uncertainty-highlight is shown, flat vs "
    "deep-zoom view). This is a rendering preference only -- it never runs the "
    "model and never changes any numbers, it only changes what's drawn on "
    "screen. "
    "You cannot delete, modify, upload, or replace images, and you cannot "
    "answer questions unrelated to this dataset or this tool. If the user "
    "asks for something outside these bounds, or doesn't specify which image "
    "they mean for a detection, say so plainly and ask a clarifying question "
    "instead of guessing or inventing an answer -- never claim a detection "
    "ran if the tool wasn't actually called, and never make up numbers. "
    "If a detection has already been run earlier in this conversation for the "
    "image being asked about, answer directly from that earlier result "
    "instead of running the tool again."
)

TOOLS = [
    {
        "type": "function",
        "function": {
            "name": "run_artifact_detection",
            "description": (
                "Run the trained artifact-detection model on a specific DAPI "
                "image and return tile-level predictions plus a summary "
                "(artifact area %, region count, largest region size). "
                "Only call this for one of the known image IDs; if the user "
                "doesn't specify one, or asks for something this tool can't do "
                "(e.g. deleting or modifying an image), ask a clarifying "
                "question or explain the limitation instead of guessing."
            ),
            "parameters": {
                "type": "object",
                "properties": {
                    "image_id": {"type": "string", "description": "e.g. 'Image_6'"}
                },
                "required": ["image_id"],
            },
        },
    },
    {
        "type": "function",
        "function": {
            "name": "list_images",
            "description": (
                "List the image IDs that actually exist in the dataset. Call "
                "this if the user asks what images are available, or if you "
                "need to confirm an image ID before running detection on it "
                "rather than guessing."
            ),
            "parameters": {"type": "object", "properties": {}},
        },
    },
    {
        "type": "function",
        "function": {
            "name": "set_display_options",
            "description": (
                "Change how the current result is displayed -- does NOT run "
                "the model or change any numbers, only what's drawn on "
                "screen. Only set the fields the user actually asked to "
                "change; leave the rest out."
            ),
            "parameters": {
                "type": "object",
                "properties": {
                    "overlay_color": {
                        "type": "string",
                        "enum": OVERLAY_COLOR_PALETTE,
                        "description": "Color for confidently-artifact tiles in the confidence overlay.",
                    },
                    "show_overlay": {
                        "type": "boolean",
                        "description": "Whether the confidence overlay is shown at all.",
                    },
                    "show_uncertainty": {
                        "type": "boolean",
                        "description": "Whether tiles with confidence 0.4-0.6 are highlighted in amber.",
                    },
                    "view_mode": {
                        "type": "string",
                        "enum": VIEW_MODES,
                        "description": "'flat' for the downsampled static image, 'deepzoom' for the pan/zoom tile viewer.",
                    },
                },
            },
        },
    },
]

# conversation_id -> message history (OpenAI/Groq chat-completion message
# dicts). In-memory only -- see module docstring.
_conversations: dict[str, list[dict]] = {}


def _execute_tool_call(name: str, arguments: dict):
    """Returns (tool_result_dict_for_the_llm, DetectResponse_or_None, DisplayOptions_or_None).

    Exactly one of the second/third elements is ever non-None (one tool call
    produces one kind of frontend-facing payload) -- kept as two independent
    optional slots rather than a tagged union since that's simpler for
    handle_chat_message and the frontend to check ("did a detection happen"
    and "did a display-option change happen" are two separate questions).

    For run_artifact_detection, deliberately returns only the summary (not
    the full tile_confidence grid) to the LLM -- it only needs the summary to
    describe the result in words; sending the full grid would waste tokens
    for no benefit. The frontend gets the full DetectResponse separately, via
    the second element.
    """
    if name == "run_artifact_detection":
        image_id = arguments.get("image_id", "")
        try:
            detection = run_detection(image_id)
        except storage.ImageNotFoundError as exc:
            return {"error": str(exc)}, None, None
        except storage.HFAuthError as exc:
            return {"error": f"Could not fetch image data: {exc}"}, None, None
        return {"ok": True, "image_id": image_id, "summary": detection.summary.model_dump()}, detection, None

    if name == "list_images":
        try:
            images = storage.list_images()
        except storage.HFAuthError as exc:
            return {"error": f"Could not list images: {exc}"}, None, None
        return {"ok": True, "images": images}, None, None

    if name == "set_display_options":
        color = arguments.get("overlay_color")
        if color is not None and color not in OVERLAY_COLOR_PALETTE:
            return (
                {"error": f"Unknown color '{color}'. Choose one of: {', '.join(OVERLAY_COLOR_PALETTE)}"},
                None,
                None,
            )
        mode = arguments.get("view_mode")
        if mode is not None and mode not in VIEW_MODES:
            return {"error": f"Unknown view_mode '{mode}'. Choose one of: {', '.join(VIEW_MODES)}"}, None, None

        options = DisplayOptions(
            overlay_color=color,
            show_overlay=arguments.get("show_overlay"),
            show_uncertainty=arguments.get("show_uncertainty"),
            view_mode=mode,
        )
        return {"ok": True, "applied": options.model_dump(exclude_none=True)}, None, options

    return {"error": f"Unknown tool '{name}'"}, None, None


def handle_chat_message(message: str, conversation_id: str | None) -> ChatResponse:
    conv_id = conversation_id if conversation_id in _conversations else str(uuid.uuid4())
    history = _conversations.setdefault(conv_id, [{"role": "system", "content": SYSTEM_PROMPT}])

    history.append({"role": "user", "content": message})

    response = _client.chat.completions.create(
        model=_MODEL, messages=history, tools=TOOLS, tool_choice="auto"
    )
    choice = response.choices[0].message

    detection_result = None
    display_options = None

    if choice.tool_calls:
        # Echo the assistant's tool-call message back into history exactly as
        # the OpenAI-compatible protocol requires, so the follow-up call can
        # resolve each tool_call_id against a matching "tool" message below.
        history.append(
            {
                "role": "assistant",
                "content": choice.content or "",
                "tool_calls": [
                    {
                        "id": tc.id,
                        "type": "function",
                        "function": {"name": tc.function.name, "arguments": tc.function.arguments},
                    }
                    for tc in choice.tool_calls
                ],
            }
        )

        for tool_call in choice.tool_calls:
            args = json.loads(tool_call.function.arguments or "{}")
            tool_result, detection, options = _execute_tool_call(tool_call.function.name, args)
            if detection is not None:
                detection_result = detection
            if options is not None:
                display_options = options
            history.append(
                {"role": "tool", "tool_call_id": tool_call.id, "content": json.dumps(tool_result)}
            )

        followup = _client.chat.completions.create(model=_MODEL, messages=history)
        reply = followup.choices[0].message.content or ""
    else:
        reply = choice.content or ""

    history.append({"role": "assistant", "content": reply})

    return ChatResponse(
        reply=reply, detection_result=detection_result, display_options=display_options, conversation_id=conv_id
    )
