"""The single module that talks to Supabase's PostgREST API for durable chat
history (chat_sessions/chat_messages -- see ../../sql/chat_history.sql).

Same "raw HTTP, no SDK" convention main.py's require_user already uses for
Supabase Auth -- no supabase-py dependency added for what's a handful of
straightforward REST calls. Every call forwards the *caller's own* bearer
token (never the anon key alone) as the Authorization header, which is what
lets each table's Row Level Security policies resolve auth.uid() to the real
user -- the anon key by itself carries no `sub` claim, so RLS would reject
(or worse, if a policy were ever misconfigured, silently scope to nothing)
every request without it.

Kept synchronous (plain httpx.Client, not AsyncClient) to match chat.py's
handle_chat_message, which is deliberately not `async def` -- see that
module's docstring for why.
"""

import httpx

from ..config import settings

_REST_URL = f"{settings.supabase_url}/rest/v1"
_client = httpx.Client(timeout=10.0)


class ChatStorageError(Exception):
    """A PostgREST call failed (network error or unexpected status). Meant to
    be caught at the route layer and turned into a clean 502, same pattern as
    HFAuthError in hf_storage.py."""


class SessionNotFoundError(Exception):
    """No such session -- or a session that isn't this user's, which RLS
    makes indistinguishable from nonexistent. Route layer maps this to 404,
    deliberately not 403, mirroring the existing image_id 404-collapse."""


def _headers(token: str, *, prefer: str | None = None) -> dict[str, str]:
    headers = {
        "apikey": settings.supabase_anon_key,
        "Authorization": f"Bearer {token}",
        "Content-Type": "application/json",
    }
    if prefer:
        headers["Prefer"] = prefer
    return headers


def create_session(token: str, user_id: str, title: str | None) -> str:
    """Insert a new chat_sessions row and return its generated id -- this id
    becomes the conversation_id used for the rest of the conversation (see
    chat.py), so it must come back in the response, hence
    Prefer: return=representation (PostgREST's default POST response is
    empty otherwise).
    """
    try:
        resp = _client.post(
            f"{_REST_URL}/chat_sessions",
            headers=_headers(token, prefer="return=representation"),
            json={"user_id": user_id, "title": title},
        )
    except httpx.TransportError as exc:
        raise ChatStorageError(f"Could not reach Supabase to create a chat session: {exc}") from exc
    if resp.status_code != 201:
        raise ChatStorageError(f"Unexpected status creating chat session: {resp.status_code} {resp.text}")
    return resp.json()[0]["id"]


def append_message(
    token: str,
    user_id: str,
    session_id: str,
    role: str,
    content: str | None,
    tool_name: str | None = None,
    tool_args: dict | None = None,
    tool_result: dict | None = None,
) -> None:
    """Insert one chat_messages row. Prefer: return=minimal -- the caller
    doesn't need the row back, unlike create_session above.
    """
    try:
        resp = _client.post(
            f"{_REST_URL}/chat_messages",
            headers=_headers(token, prefer="return=minimal"),
            json={
                "session_id": session_id,
                "user_id": user_id,
                "role": role,
                "content": content,
                "tool_name": tool_name,
                "tool_args": tool_args,
                "tool_result": tool_result,
            },
        )
    except httpx.TransportError as exc:
        raise ChatStorageError(f"Could not reach Supabase to store a chat message: {exc}") from exc
    if resp.status_code != 201:
        raise ChatStorageError(f"Unexpected status storing chat message: {resp.status_code} {resp.text}")


def list_sessions(token: str, user_id: str, limit: int = 50) -> list[dict]:
    """This user's past sessions, most-recently-active first. RLS already
    scopes rows to this user; user_id is still passed in (unused in the
    query itself) so the signature stays self-documenting alongside the
    other functions here, which do need it.
    """
    try:
        resp = _client.get(
            f"{_REST_URL}/chat_sessions",
            headers=_headers(token),
            params={
                "select": "id,title,created_at,updated_at",
                "order": "updated_at.desc",
                "limit": str(limit),
            },
        )
    except httpx.TransportError as exc:
        raise ChatStorageError(f"Could not reach Supabase to list chat sessions: {exc}") from exc
    if resp.status_code != 200:
        raise ChatStorageError(f"Unexpected status listing chat sessions: {resp.status_code} {resp.text}")
    return resp.json()


def get_session_transcript(token: str, user_id: str, session_id: str) -> dict:
    """One session plus its ordered messages, fetched in a single round trip
    via PostgREST resource embedding (chat_messages.session_id -> the FK to
    chat_sessions.id). RLS applies independently to the parent row and the
    embedded child rows -- a session that isn't this user's comes back as an
    empty list here, not an error, which is why an empty result is treated as
    SessionNotFoundError rather than assumed to mean "no messages yet."
    """
    try:
        resp = _client.get(
            f"{_REST_URL}/chat_sessions",
            headers=_headers(token),
            params={
                "id": f"eq.{session_id}",
                "select": (
                    "id,title,created_at,updated_at,"
                    "chat_messages(id,role,content,tool_name,tool_args,tool_result,created_at)"
                ),
                "chat_messages.order": "id.asc",
            },
        )
    except httpx.TransportError as exc:
        raise ChatStorageError(f"Could not reach Supabase to load chat session: {exc}") from exc
    if resp.status_code != 200:
        raise ChatStorageError(f"Unexpected status loading chat session: {resp.status_code} {resp.text}")

    rows = resp.json()
    if not rows:
        raise SessionNotFoundError(f"No such session '{session_id}'")

    row = rows[0]
    row["messages"] = row.pop("chat_messages")
    return row
