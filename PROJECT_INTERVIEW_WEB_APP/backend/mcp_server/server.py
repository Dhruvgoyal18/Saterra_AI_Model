"""Local MCP server exposing this project's detection tools to Claude
Desktop over SSE. Additive, not a replacement for POST /chat (app/chat.py,
Groq function-calling) -- that keeps serving the web UI unchanged. Both
front doors ultimately hit the same require_user-gated FastAPI routes and
the same run_detection() path, so there's still exactly one detection
implementation (see steps_of_implementation_checkpoint3.md's note on this).

Run standalone: `python -m mcp_server.server` from backend/, with the
FastAPI backend (uvicorn app.main:app --reload --port 8010) already running
in a separate terminal.

Served over plain HTTP (SSE transport, no TLS).

Auth is lazy and tool-triggered, not connection-level OAuth: the MCP
transport itself is open (same 127.0.0.1-only trust boundary as always, see
app/config.py's mcp_server_host comment), and each tool call checks
login_session.is_authenticated itself. When signed out, a tool hands back a
plain /login link instead of prompting for a password in chat -- see
login_routes.py for the actual page, login_session.py for the single-user
in-memory session it fills in.
"""

import httpx
from mcp.server.fastmcp import FastMCP

from app.config import settings

from . import backend_client
from .login_routes import build_login_routes
from .login_session import login_session

_issuer_url = f"http://{settings.mcp_server_host}:{settings.mcp_server_port}"

mcp = FastMCP(
    "dapi-artifact-detector",
    instructions=(
        "You have two tools for a DAPI whole-slide-image artifact detection "
        "dataset: list_images (look up real image IDs, never guess one), and "
        "run_artifact_detection (run the trained model on one image, returning "
        "a text summary: artifact area %, region count, largest region size -- "
        "no image is returned). If either tool reports the user isn't signed "
        "in, relay the login link it returns verbatim and ask the user to open "
        "it and sign in there, then ask again -- never ask the user for their "
        "email or password yourself, and never invent a login link. You cannot "
        "delete, modify, upload, or replace images. If a detection has already "
        "been run earlier in this conversation for the image being asked about, "
        "answer from that result instead of re-running the tool."
    ),
    host=settings.mcp_server_host,
    port=settings.mcp_server_port,
)

_LOGIN_PROMPT = f"Not signed in. Open {_issuer_url}/login to sign in, then ask again."


@mcp.tool()
async def list_images() -> str:
    """List the image IDs that actually exist in the dataset.

    Call this if the user asks what images are available, or to confirm an
    image ID before running detection on it rather than guessing.
    """
    if not login_session.is_authenticated:
        return _LOGIN_PROMPT

    token = login_session.token
    try:
        images = await backend_client.backend_list_images(token)
    except httpx.HTTPStatusError as exc:
        if exc.response.status_code == 401:
            return "Session expired -- " + _LOGIN_PROMPT
        return f"Could not list images: {exc}"

    return "Available images: " + ", ".join(images)


@mcp.tool()
async def run_artifact_detection(image_id: str) -> str:
    """Run the trained artifact-detection model on one image.

    Returns a text summary only (artifact area %, connected region count,
    largest region size) -- no image. Only call this for a known image ID --
    call list_images first if unsure.
    """
    if not login_session.is_authenticated:
        return _LOGIN_PROMPT

    token = login_session.token
    try:
        detection = await backend_client.backend_run_detection(token, image_id)
    except httpx.HTTPStatusError as exc:
        status = exc.response.status_code
        if status == 401:
            return "Session expired -- " + _LOGIN_PROMPT
        if status == 404:
            return f"'{image_id}' is not a known image ID -- call list_images to see what's available."
        return f"Detection failed: {exc}"

    return _summary_text(image_id, detection["summary"])


def _summary_text(image_id: str, summary: dict) -> str:
    pct = summary["total_artifact_area_frac_of_tissue"] * 100
    return (
        f"Detection on {image_id}: {pct:.2f}% of tissue flagged as artifact, "
        f"{summary['num_connected_regions']} connected artifact region(s), "
        f"largest region {summary['largest_region_area_px']} px "
        f"({summary['n_tiles_evaluated']}/{summary['n_tiles_total']} tiles evaluated)."
    )


if __name__ == "__main__":
    import uvicorn

    # mcp.run(transport="sse") builds this same app internally but gives no
    # way to add routes to it -- build it ourselves so /login can be mounted.
    app = mcp.sse_app()
    for route in build_login_routes():
        app.routes.append(route)
    uvicorn.run(app, host=settings.mcp_server_host, port=settings.mcp_server_port)
