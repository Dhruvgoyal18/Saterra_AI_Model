"""The browser login page for the MCP server.

Deliberately NOT an OAuth authorization endpoint (no login_id, no
redirect-back-to-client, no code exchange) -- MCP tool calls (server.py)
check login_session.is_authenticated themselves and, when signed out, hand
the user a plain link to this page instead of prompting for a password in
chat. There's only ever one pending login (single-user, single-process, see
login_session.py), so GET /login always just renders the form and POST
/login always just fills in the one global session -- no per-request state
to track between the two requests.
"""

from starlette.requests import Request
from starlette.responses import HTMLResponse, Response
from starlette.routing import Route

from . import backend_client
from .login_session import login_session

_FORM_TEMPLATE = """\
<!doctype html>
<html>
<head><title>Sign in -- DAPI artifact detector</title></head>
<body style="font-family: sans-serif; max-width: 360px; margin: 4rem auto;">
  <h2>Sign in</h2>
  {message}
  <form method="post" action="/login">
    <label>Email<br><input type="email" name="email" required autofocus style="width:100%"></label><br><br>
    <label>Password<br><input type="password" name="password" required style="width:100%"></label><br><br>
    <button type="submit">Sign in</button>
  </form>
</body>
</html>
"""

_SIGNED_IN_TEMPLATE = """\
<!doctype html>
<html>
<head><title>Signed in -- DAPI artifact detector</title></head>
<body style="font-family: sans-serif; max-width: 360px; margin: 4rem auto;">
  <h2>You're signed in</h2>
  <p>Go back to Claude and ask your question again.</p>
</body>
</html>
"""


def build_login_routes() -> list[Route]:
    async def login_get(request: Request) -> Response:
        return HTMLResponse(_FORM_TEMPLATE.format(message=""))

    async def login_post(request: Request) -> Response:
        form = await request.form()
        email = str(form.get("email", ""))
        password = str(form.get("password", ""))

        try:
            access_token, expires_in = await backend_client.supabase_authenticate(email, password)
        except backend_client.AuthError:
            error_html = "<p style='color:red'>Invalid email or password. Please try again.</p>"
            return HTMLResponse(_FORM_TEMPLATE.format(message=error_html), status_code=401)

        login_session.set(access_token, email, expires_in)
        return HTMLResponse(_SIGNED_IN_TEMPLATE)

    return [
        Route("/login", login_get, methods=["GET"]),
        Route("/login", login_post, methods=["POST"]),
    ]
