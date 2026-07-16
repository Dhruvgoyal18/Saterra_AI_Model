"""HTTP calls the MCP tools (server.py) make -- to Supabase directly for
login, and to this project's own FastAPI backend (app/main.py) for
everything else. Deliberately over HTTP, not an in-process import of
app.detection/app.storage: this MCP server reuses the exact same
require_user auth boundary the web frontend goes through, rather than
bypassing it (see app/config.py's mcp_backend_base_url comment).

One shared AsyncClient per process, same pattern as main.py's own
module-level _http_client, rather than constructing a new client per call.
"""

import httpx

from app.config import settings

_client = httpx.AsyncClient()


class AuthError(Exception):
    """Raised on a failed Supabase login (bad credentials, network issue)."""


async def supabase_authenticate(email: str, password: str) -> tuple[str, int]:
    """Logs in via Supabase's password grant, returns (access_token, expires_in).

    Same mechanism the frontend uses client-side
    (frontend/src/supabaseClient.ts's signInWithPassword), just issued from
    this server process instead of a browser. expires_in (seconds) lets a
    caller compute a real expiry for the token instead of holding it forever.
    """
    try:
        resp = await _client.post(
            f"{settings.supabase_url}/auth/v1/token?grant_type=password",
            headers={"apikey": settings.supabase_anon_key},
            json={"email": email, "password": password},
            timeout=10.0,
        )
    except httpx.TransportError as exc:
        raise AuthError(f"Could not reach Supabase: {exc}") from exc

    if resp.status_code != 200:
        detail = resp.json().get("error_description", resp.text) if resp.content else resp.text
        raise AuthError(detail)

    body = resp.json()
    return body["access_token"], body["expires_in"]


async def backend_list_images(token: str) -> list[str]:
    """GET /images -- returns just the image_id list (see app/schemas.py's
    ImagesResponse); this server has no use for level0_dims/display_dims."""
    resp = await _client.get(
        f"{settings.mcp_backend_base_url}/images",
        headers={"Authorization": f"Bearer {token}"},
        timeout=10.0,
    )
    resp.raise_for_status()
    return [img["image_id"] for img in resp.json()["images"]]


async def backend_run_detection(token: str, image_id: str) -> dict:
    """POST /detect. Timeout matches frontend/src/api.ts's DETECT_TIMEOUT_MS
    (120s) -- a cold-start + CPU-tiling run can legitimately take that long,
    a short default here would produce a spurious client-side timeout on a
    request the backend would have finished."""
    resp = await _client.post(
        f"{settings.mcp_backend_base_url}/detect",
        headers={"Authorization": f"Bearer {token}"},
        json={"image_id": image_id},
        timeout=120.0,
    )
    resp.raise_for_status()
    return resp.json()


async def backend_fetch_display_png(token: str, image_id: str) -> bytes:
    """GET /images/{id}/display -- already an 8-bit normalized, cached PNG
    (see app/ml/display_prep.py); no new image processing needed here."""
    resp = await _client.get(
        f"{settings.mcp_backend_base_url}/images/{image_id}/display",
        headers={"Authorization": f"Bearer {token}"},
        timeout=30.0,
    )
    resp.raise_for_status()
    return resp.content
