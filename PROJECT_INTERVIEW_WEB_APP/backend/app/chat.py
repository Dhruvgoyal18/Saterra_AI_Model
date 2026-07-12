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
from .schemas import ChatResponse

logger = logging.getLogger("artifact-detector")

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
    "You have exactly two capabilities, each via its own tool: "
    "(1) run_artifact_detection -- run the trained artifact-detection model on "
    "one of the known images, returning the percentage of tissue flagged as "
    "artifact, the number of connected artifact regions, and the largest "
    "region's size; "
    "(2) list_images -- look up which image IDs actually exist, so you never "
    "have to guess or invent one. "
    "You cannot delete, modify, upload, or replace images, you cannot change "
    "how results are displayed (that's controlled directly in the UI, not "
    "through chat), and you cannot answer questions unrelated to this dataset "
    "or this tool. If the user asks for something outside these bounds, or "
    "doesn't specify which image they mean for a detection, say so plainly "
    "and ask a clarifying question instead of guessing or inventing an answer "
    "-- never claim a detection ran if the tool wasn't actually called, and "
    "never make up numbers. "
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
]

# conversation_id -> message history (OpenAI/Groq chat-completion message
# dicts). In-memory only -- see module docstring.
_conversations: dict[str, list[dict]] = {}


def _execute_tool_call(name: str, arguments: dict):
    """Returns (tool_result_dict_for_the_llm, DetectResponse_or_None).

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
            return {"error": str(exc)}, None
        except storage.HFAuthError as exc:
            return {"error": f"Could not fetch image data: {exc}"}, None
        return {"ok": True, "image_id": image_id, "summary": detection.summary.model_dump()}, detection

    if name == "list_images":
        try:
            images = storage.list_images()
        except storage.HFAuthError as exc:
            return {"error": f"Could not list images: {exc}"}, None
        return {"ok": True, "images": images}, None

    return {"error": f"Unknown tool '{name}'"}, None


def _looks_like_malformed_tool_call(content: str | None) -> bool:
    """Groq's Llama-3.3-70B occasionally hallucinates a pseudo-function-call as
    plain text (e.g. '<function.run_artifact_detection {"image_id": "Image_7"}>
    </function>') instead of using the real tool-calling protocol -- observed in
    testing specifically after a prior successful tool call earlier in the same
    conversation (~1-in-4 failure rate reproduced against the deployed
    list_images -> run_artifact_detection sequence). Not fixable by a schema
    change (tried giving list_images a non-empty parameter -- made it worse,
    surfacing as an outright 400 tool_use_failed with the same malformed
    pattern instead). This is inherent model flakiness, not a bug in this
    code's use of the tool-calling API, so the mitigation is a one-time retry
    of the same request rather than a code-level fix -- reduces the effective
    failure rate, doesn't guarantee eliminating it (a retry can, in principle,
    fail the same way).
    """
    return content is not None and "<function" in content


def handle_chat_message(message: str, conversation_id: str | None) -> ChatResponse:
    conv_id = conversation_id if conversation_id in _conversations else str(uuid.uuid4())
    history = _conversations.setdefault(conv_id, [{"role": "system", "content": SYSTEM_PROMPT}])

    history.append({"role": "user", "content": message})

    response = _client.chat.completions.create(
        model=_MODEL, messages=history, tools=TOOLS, tool_choice="auto"
    )
    choice = response.choices[0].message

    if not choice.tool_calls and _looks_like_malformed_tool_call(choice.content):
        logger.warning("Malformed pseudo-tool-call detected, retrying once: %r", choice.content)
        response = _client.chat.completions.create(
            model=_MODEL, messages=history, tools=TOOLS, tool_choice="auto"
        )
        choice = response.choices[0].message

    detection_result = None

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
            tool_result, detection = _execute_tool_call(tool_call.function.name, args)
            if detection is not None:
                detection_result = detection
            history.append(
                {"role": "tool", "tool_call_id": tool_call.id, "content": json.dumps(tool_result)}
            )

        followup = _client.chat.completions.create(model=_MODEL, messages=history)
        reply = followup.choices[0].message.content or ""
    else:
        reply = choice.content or ""

    history.append({"role": "assistant", "content": reply})

    return ChatResponse(reply=reply, detection_result=detection_result, conversation_id=conv_id)
