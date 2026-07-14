"""In-memory Supabase auth-session holder for the local MCP server process.

Single-user, single-process scope on purpose -- this server is a local
Claude Desktop/Code bridge for one person, not a multi-tenant service, so a
plain module-level singleton is enough. The token is never written to disk:
it's lost on process restart or on expiry, requiring the user to click the
/login link again, the same non-persistence tradeoff app/chat.py's in-memory
conversation history already makes.

Deliberately not full OAuth (see login_routes.py's docstring for why): tool
calls check is_authenticated themselves and hand back a plain /login link
when it's False, rather than gating the whole MCP connection on an
authorization handshake before any tool is even callable.
"""

import time


class LoginSession:
    def __init__(self) -> None:
        self._token: str | None = None
        self._email: str | None = None
        self._expires_at: float | None = None

    @property
    def is_authenticated(self) -> bool:
        return self._token is not None and self._expires_at is not None and time.time() < self._expires_at

    @property
    def token(self) -> str | None:
        return self._token if self.is_authenticated else None

    def set(self, token: str, email: str, expires_in: int) -> None:
        self._token = token
        self._email = email
        self._expires_at = time.time() + expires_in


login_session = LoginSession()
