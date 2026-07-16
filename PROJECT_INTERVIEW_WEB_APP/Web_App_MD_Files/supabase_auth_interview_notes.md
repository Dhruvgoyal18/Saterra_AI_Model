# Supabase Authentication — System Design & Interview Reference

Reference notes on how invite-only access control works in this app: the design, the real code, the exact request/response shapes, and a full interview Q&A bank. Written for interview review.

---

## 1. The problem, in one sentence

The backend has to be reachable by a real browser over the public internet (so it can't just be network-blocked), but access still needs to be invite-only, not a public demo — that single constraint drives every decision below.

---

## 2. The big picture

Two separate concerns, kept deliberately separate:

- **Identity** ("who is this user") — owned entirely by Supabase, a managed auth service (its own user store, JWT issuance, password hashing). This app never builds or stores credentials itself.
- **Enforcement** ("is this specific request allowed through") — owned by a single FastAPI dependency, `require_user`, on the backend. This is the only real security boundary. The frontend's login screen is UX only — anyone can call the API directly with curl/Postman, bypassing the React app entirely, so a check that only lives in the browser is not a gate at all.

---

## 3. End-to-end flow

1. **Login** — `LoginForm.tsx` calls `supabase.auth.signInWithPassword({ email, password })` directly against Supabase's hosted service. The backend is not involved in this step.
2. **Session held client-side** — the Supabase JS client (`supabaseClient.ts`) holds the session in the browser and silently auto-refreshes it before it expires (1hr default).
3. **Every API call attaches the token fresh** — `api.ts`'s `authHeaders()` calls `supabase.auth.getSession()` immediately before each request, returning `{ Authorization: "Bearer <token>" }`. Re-read every call, never cached — a tab left open for hours keeps working through a silent token refresh.
4. **The backend actually checks it** — every data-bearing route in `main.py` carries `dependencies=[Depends(require_user)]`. `require_user` forwards the token to Supabase's `GET /auth/v1/user`; only a 200 lets the request through.
5. **`App.tsx` gates the UI** — no session → `LoginForm`; a session → the real app. Subscribed to `onAuthStateChange` so sign-out reflects immediately.

---

## 4. The two Supabase endpoints — generate vs. validate

| Endpoint | Purpose | Called by |
|---|---|---|
| `POST /auth/v1/token?grant_type=password` | **Generates** a token from email+password | The Supabase JS SDK, inside `signInWithPassword` |
| `POST /auth/v1/token?grant_type=refresh_token` | **Regenerates** a token from the refresh token, silently | The SDK, automatically, near the 1hr expiry |
| `GET /auth/v1/user` | **Validates** an existing token, returns the user or an error | This app's `require_user`, on every gated request |
| `POST /auth/v1/signup` | Creates a new account — **disabled** in this project (`disable_signup`) | Not used in the normal flow; this is what makes access invite-only |

Split because minting requires proving identity from scratch (expensive, infrequent); validating just checks something already held (cheap, needed on every request).

---

## 5. What `signInWithPassword` actually returns

```json
{
  "access_token": "eyJhbGciOiJIUzI1NiJ9.eyJzdWIiOiJmMjFhOTk4Yi0uLi4iLCJlbWFpbCI6ImRocnV2Z295YWw5OTBAZ21haWwuY29tIiwicm9sZSI6ImF1dGhlbnRpY2F0ZWQiLCJleHAiOjE3NTI0NzM2MDB9.XXXXX",
  "refresh_token": "v1.MRl9k3...",
  "expires_at": 1752473600,
  "user": { "id": "f21a998b-...", "email": "dhruvgoyal990@gmail.com" }
}
```

The `access_token` is three base64 chunks joined by dots: `header.payload.signature`.
- **Header** — signing algorithm used.
- **Payload** — plain, readable claims (`sub`, `email`, `role`, `exp`). NOT encrypted — anyone can decode it. That's fine; it's not meant to be secret, only tamper-evident.
- **Signature** — a hash of header+payload computed with Supabase's private signing key. Proves the token is genuine and unmodified. Editing the payload changes its bytes, which no longer match the signature — a forged/edited token fails verification.

**Decoding vs. verifying:** decoding just reads the base64, no key needed, proves nothing. Verifying checks the signature against the correct secret — the step that actually proves authenticity. This app's own backend never performs the verify step itself (see §7).

---

## 6. How validation actually works — the exact code path

```python
async def require_user(authorization: str | None = Header(None)) -> dict:
    if not authorization or not authorization.startswith("Bearer "):
        raise HTTPException(status_code=401, detail="Missing bearer token")
    token = authorization.removeprefix("Bearer ")
    try:
        resp = await _http_client.get(
            f"{settings.supabase_url}/auth/v1/user",
            headers={"Authorization": f"Bearer {token}", "apikey": settings.supabase_anon_key},
        )
    except httpx.TransportError as exc:
        raise HTTPException(status_code=503, detail="Auth service temporarily unavailable -- try again") from exc
    if resp.status_code != 200:
        raise HTTPException(status_code=401, detail="Invalid or expired session")
    return resp.json()
```

1. No/malformed header → instant 401, no network call (nothing to check).
2. Header present → the token is forwarded, as-is, to Supabase's `/auth/v1/user`. **The backend performs zero cryptography itself** — it outsources "is this genuine" entirely to Supabase, which checks the signature against its own key, checks `exp`, and checks the session hasn't been revoked.
3. Supabase 200 → `require_user` returns the user dict, the real route executes, real data goes back to the frontend.
4. Supabase non-200 → uniform 401 "Invalid or expired session," regardless of the specific reason.
5. Supabase unreachable (`httpx.TransportError`) → 503 "Auth service temporarily unavailable" — a real production bug fix: originally no timeout was set, a slow response threw an uncaught `httpx.ReadTimeout` that surfaced as a raw 500. Fixed with an explicit `timeout=10.0` and this catch block.

---

## 7. Why this design — the key rationale points

- **Supabase Auth over a hand-rolled `users` table:** owning password storage/hashing is real security surface for no benefit over a service that already does it correctly and for free.
- **"Invite-only" is a project setting, not app code:** Authentication → disable "Allow new users to sign up." Verified as a real boundary, not just cosmetic, by hitting `/auth/v1/signup` directly before/after flipping it (proceeds → `422 signup_disabled`).
- **The backend dependency is the real gate, the frontend check is not:** verified directly — curl with no token → 401; curl with a real token obtained straight from Supabase's own token endpoint (bypassing the UI entirely) → 200 with correct data.
- **Round-trip validation over local verification — a known, accepted gap.** `require_user` calls Supabase on every request instead of verifying the JWT signature locally (`pyjwt` + the project's signing secret, no network call). Noted explicitly in this project's own code review as a future upgrade, never implemented.
  - *Round-trip pros:* instant revocation — a killed session is rejected on the very next request.
  - *Round-trip cons:* added latency per request, and the whole app becomes hard-dependent on Supabase's uptime.
  - *Local verification would flip both:* faster, no uptime dependency, but a revoked session would stay "valid" until its natural expiry, since nothing would re-check with Supabase in between.
- **Anon key is safe to ship client-side; `HF_TOKEN` never is.** The anon key only unlocks endpoints Supabase already exposes publicly (sign in, get session) — by design, not a master credential. Set as a plain deployment *variable*, not a secret.
- **Why the HF Space itself was later made public, not private:** once Supabase Auth was the real gate, keeping the Space private too was redundant (a private Space blocks unauthenticated requests at HF's own gateway before the app runs at all) and would've broken the frontend for real signed-in users. There's no safe way to hold an HF credential client-side anyway — embedding one in the shipped JS bundle is trivially extractable via DevTools.
- **Token refresh, handled by never caching the token:** `authHeaders()` re-reads the session fresh before every call rather than capturing `access_token` once — the SDK auto-refreshes the underlying session, so a long-lived tab keeps working without the user noticing the 1hr expiry.

---

## 8. The `<img>` auth gap — a real design catch

`DetectionCanvas`/`DeepZoomCanvas` need to load images, but a plain `<img src="...">` (or OSD's default tile loader) has **no way to attach a custom `Authorization` header**. Gating those endpoints the same way as everything else would have silently broken image display.

**Fix:** `api.ts`'s `fetchDisplayImageObjectUrl` fetches the endpoint manually via `fetch()` (which *can* carry headers), converts the response to a `Blob`, and hands the component a local `URL.createObjectURL()` reference instead of a plain API URL — `App.tsx` owns revoking it on switch/unmount. The deep-zoom viewer hits the identical problem, solved the identical way conceptually: `loadTilesWithAjax: true` + `ajaxHeaders` instead of letting OpenSeadragon default to `<img>`-based tile loading.

---

## 9. Known, explicitly accepted gaps (from this project's own code review)

- No rate limiting on `/detect` — a signed-in user could repeatedly trigger CPU-bound inference.
- `CORS_ORIGINS=*` on both deployed backends — assessed as low-risk specifically *because* this app uses Bearer-header auth, not cookies (no CSRF-style piggyback), but still flagged to tighten to the real frontend origin.
- Test account password (`1234`) is trivially weak; Supabase's password-strength minimum isn't actually enforced by default.
- `require_user`'s per-request round-trip to Supabase (see §7) — latency + availability coupling, known, not yet fixed.

---

## 10. Files this touches

| File | Role |
|---|---|
| `frontend/src/supabaseClient.ts` | Creates the Supabase client with the public anon key |
| `frontend/src/LoginForm.tsx` | Email/password form, calls `signInWithPassword` |
| `frontend/src/App.tsx` | Session check on load, `onAuthStateChange` subscription, renders `LoginForm` vs. real app |
| `frontend/src/api.ts` | `authHeaders()` (fresh token per call), `fetchDisplayImageObjectUrl` (the `<img>` workaround) |
| `backend/app/main.py` | `require_user` dependency — the actual enforcement point; applied to every data route except `/health` |
| `backend/app/config.py` | `supabase_url`/`supabase_anon_key` settings; fail-fast `RuntimeError` at startup if missing |

---

## 11. Interview Q&A bank

### Round-trip vs. local verification

**Q: Why does `require_user` round-trip to Supabase on every request instead of verifying the JWT locally?**
A: The backend never holds Supabase's signing secret, so it can't check the signature itself. It delegates the entire "is this genuine" question to `/auth/v1/user` on every request — a known, deliberately accepted gap, documented in this project's own code review as a future upgrade never implemented.

**Q: What's the actual tradeoff between local verification and the round-trip approach?**
A: Local verification (`pyjwt` + the signing secret, checking signature + `exp`) is faster and removes the Supabase-uptime dependency, but loses instant revocation — a locally-checked token stays "valid" until natural expiry even if the session was killed server-side. The round-trip gets real-time revocation at the cost of latency and an availability coupling.

**Q: What would you concretely need to switch to local verification?**
A: The project's Supabase JWT signing secret, plus a JWT library to check the signature and `exp` claim — zero network calls.

**Q: What happens if Supabase itself is unreachable?**
A: Every gated endpoint fails — no fallback exists today. Handled somewhat gracefully (clean 503, not a raw crash) but the app is functionally down until Supabase recovers.

**Q: Could a stolen valid token be replayed days later?**
A: No — only until it naturally expires, a max ~1hr window. This bounds the blast radius of a leaked token versus a permanent API key.

### JWT structure

**Q: What are the three parts of a JWT?**
A: `header.payload.signature`. Header = signing algorithm; payload = claims (user id, email, role, expiry); signature = a hash of header+payload using Supabase's private key, proving authenticity and integrity.

**Q: The payload is just base64, not encrypted — security problem?**
A: No — a JWT is tamper-evident, not secret. Nothing sensitive (like a password) goes in the payload, so plain-text readability is expected and fine.

**Q: Could someone decode the payload, edit it, and get it accepted?**
A: No — editing the payload changes its bytes, which no longer match the signature computed over the original bytes. Whoever verifies it (Supabase, here) rejects the mismatch.

**Q: Difference between decoding and verifying a JWT?**
A: Decoding just reads the base64, no key needed, proves nothing. Verifying checks the signature against the correct key — the step that actually proves authenticity. This backend never does the verify step itself; it outsources it.

### Token lifecycle

**Q: Why expire the access token after an hour instead of forever?**
A: Limits the damage window if a token ever leaks.

**Q: If it expires hourly, why doesn't the user re-login every hour?**
A: A longer-lived `refresh_token`, issued alongside it, silently mints a new access token via the SDK before expiry — invisible unless subscribed to `onAuthStateChange`.

**Q: Difference in purpose between `access_token` and `refresh_token`?**
A: Access token: sent on every API request, short-lived, what `require_user` checks. Refresh token: never sent to this backend, only used directly against Supabase to mint a new access token, longer-lived.

**Q: Two tabs open, both try refreshing near the 1hr mark — what happens?**
A: A real race-condition class of problem worth reasoning through rather than asserting certainty about — good to flag as something you'd want to verify rather than assume works cleanly.

### Response/status-code behavior

**Q: Why does a missing token short-circuit to 401 with no network call, but an invalid one requires a round trip first?**
A: A missing token is detectable locally (nothing to check). An invalid/expired one looks well-formed, and since the backend never verifies signatures itself, the only way to know is to ask Supabase — that path always pays the round-trip cost, even to say no.

**Q: Why 503, specifically, for "Supabase unreachable," not 500?**
A: 503 signals a transient, retryable condition, not a bug in the request. Ties to a real production incident: an unhandled `httpx.ReadTimeout` originally surfaced as a raw 500; fixed with an explicit client timeout + a caught `TransportError` → clean 503.

**Q: Difference between 401 and 403 in this app?**
A: Only one tier exists — authenticated or not (401). No role/permission system, so a 403 case doesn't currently apply.

### Frontend/backend boundary

**Q: What actually stops someone from calling the API directly with curl?**
A: Nothing except `require_user` — exactly why it, not the login screen, is the real boundary.

**Q: Is the anon key a secret?**
A: No — it only unlocks endpoints already meant to be public; shipped openly, set as a deployment variable, not a secret, unlike `HF_TOKEN`.

**Q: Why isn't `/health` gated?**
A: No security benefit, and it would break external uptime monitoring for nothing gained.

**Q: Why fetch the display image as a blob instead of `<img src>`?**
A: A plain `<img>` can't carry a custom `Authorization` header; the endpoint is auth-gated, so it's fetched manually and converted to an object URL instead (see §8).

**Q: Is `/detect` rate-limited?**
A: No — known, accepted gap; a signed-in user could repeatedly trigger CPU-bound inference.

**Q: Is `CORS_ORIGINS=*` a real risk here?**
A: Assessed low-risk specifically because auth is Bearer-header-based, not cookie-based (no CSRF-style forgery path), but still flagged to tighten to the real origin.

**Q: How would you harden the weak test-account password?**
A: Enforce Supabase's password-strength minimum under Authentication → Settings — currently allowed a password like `1234` despite warning about it.

### The two Supabase endpoints

**Q: Which endpoint generates a token, which validates it, and why separate?**
A: `POST /auth/v1/token?grant_type=password` generates (login); `GET /auth/v1/user` validates. Separate because minting proves identity from scratch (expensive, rare) while validating checks something already held (cheap, needed every request).

**Q: Why does `/auth/v1/user` catch revocation when a locally-decoded token's `exp` hasn't passed?**
A: It's a live lookup against Supabase's current state, not just a timestamp check — if the session was killed since the token was issued, this reflects it immediately. Exactly the property lost by switching to local-only verification.
