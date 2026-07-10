# Checkpoint 3: Chat Interface — Implementation Log

Everything below was actually built and run against the real deployed
backend (Hugging Face Space, `Dhruvgoyal18/saterra-ml-model`) and Supabase
project -- no hypothetical transcripts, no invented numbers. Checkpoints 1
and 2's own logs (`step_of_implementation_checkpoint1.md`,
`steps_of_implementation_checkpoint2.md`) are the prerequisite context: a
trained tile classifier, a FastAPI backend with Supabase-gated auth, and a
Vercel frontend already driving picker -> Detect -> overlay -> summary
through `POST /detect`.

---

## 1. What Checkpoint 3 adds, in one sentence

A chat panel in the same frontend where a signed-in user can type a request
in natural language and get the same overlay/summary the "Detect artifacts"
button already produces, plus a natural-language reply -- via LLM
function-calling over the *exact same* detection code path, not a second
pipeline.

## 2. Architecture decision: extend the existing FastAPI app

One new endpoint, `POST /chat`, added to the same `backend/app/main.py`,
gated by the same `require_user` Supabase dependency every other
data-bearing route already uses. No new auth model, no new server process,
no second exposed surface.

**Function-calling, not an MCP server.** This was an explicit, spec-level
choice, not a shortcut taken silently: the build spec named "expose it as an
MCP tool, or function-calling as an accepted fallback" and function-calling
was the one actually implemented, with MCP explicitly deferred as "a thin
wrapper choice, not a reason to rebuild the detection call again" if ever
wanted later. The distinction, for the record:
- **Function-calling** (what's built): the LLM API itself (Groq,
  OpenAI-compatible) accepts a tool schema alongside the conversation: the
  model can respond "call `run_artifact_detection(image_id=...)`" instead of
  plain text. *This backend's own code* executes that function in-process
  and feeds the result back for the model to phrase a reply -- one HTTP
  round-trip to Groq, no new protocol, no new process.
- **MCP (Model Context Protocol)**: a standardized JSON-RPC-based protocol
  for exposing tools to *any* LLM host application, not just one codebase.
  Requires a standalone **MCP server** process (answers "here are my tools" /
  "here's this tool's result") and an **MCP client** embedded in the
  consuming LLM application (Claude Desktop, Claude Code, or a custom agent),
  which itself still lowers those tool definitions into the underlying LLM
  provider's own function-calling format to actually invoke a model. MCP
  doesn't replace function-calling -- it standardizes packaging/discovery on
  top of it, so one server can serve many different clients without custom
  per-app integration.

Given this project's chat surface is a single, self-contained web UI talking
to its own backend, function-calling is the correct-weight solution; an MCP
server would be real added infrastructure (a new process, a client capable
of speaking the protocol) for no capability this specific chat panel needs.
If a real MCP server is wanted later (e.g. to make the same tool usable from
Claude Desktop or another MCP client, not just this web app), it would wrap
`detection.run_detection()` / `chat._execute_tool_call()` directly -- not
duplicate them.

## 3. Prerequisite fixed first: `image_id` allowlist validation

Checkpoint 2's own review (SS13, finding #4) had already flagged that
`/detect` and `/images/{id}/display` didn't validate `image_id` against the
real image list before using it to construct an HF filename -- a
low-priority gap when only a `<select>` dropdown could supply `image_id`,
not once an LLM can freely invent one from natural language.

`backend/app/storage/hf_storage.py` gained `validate_image_id()`, checked
against `list_images()` before any HF fetch is attempted -- called at the
top of `get_original_path()` (and therefore automatically covering
`get_display_image_path()`, which calls it internally).

**A real pre-existing bug found and fixed in the process, not just the
gap originally flagged:** `get_original_path`'s exception handling classified
*any* 404 from `hf_hub_download` -- both "repo not found/inaccessible" and
"this filename doesn't exist in an otherwise-accessible repo" -- as an
`HFAuthError` ("check your `HF_TOKEN`"), because both cases return HTTP 404
and the code's `status in (401, 403, 404)` check couldn't tell them apart.
A request for a nonexistent `image_id` was therefore producing a confusing
"check your HF_TOKEN" message instead of a clean "no such image." Verified
before the fix (via direct testing) and after:

```
# Before the allowlist check existed (this exact ambiguity), a request for
# an unknown image would have hit the misclassified branch.
# After validate_image_id() short-circuits first:
POST /detect {"image_id": "Image_99"}
-> 404 {"detail":"No such image 'Image_99'. Known images: Image_1, Image_2, ..., Image_8"}
```

This is also the mechanism that makes the "graceful decline" demo turn (SS6)
actually work correctly rather than by accident.

## 4. LLM provider: Groq

Chosen over Gemini (both offered as options in the build spec) because:
- Its chat completions API is OpenAI-compatible, so the tool schema in the
  spec (flat `name`/`description`/`parameters`, wrapped in
  `{"type": "function", "function": {...}}`) maps directly with no
  translation, unlike Gemini's `FunctionDeclaration` SDK objects.
- Fast enough that a chat turn triggering a real detection (TIFF download +
  ~20k-tile CPU inference) doesn't add noticeable LLM-side latency on top of
  that already-dominant cost.

Model: `llama-3.3-70b-versatile`. API key stored as a Space **secret**
(`GROQ_API_KEY`), same treatment as `HF_TOKEN` -- backend-only, never sent to
the frontend, same reasoning as every other real credential in this project.

## 5. Tool schema and system prompt

```json
{
  "type": "function",
  "function": {
    "name": "run_artifact_detection",
    "description": "Run the trained artifact-detection model on a specific DAPI image and return tile-level predictions plus a summary (artifact area %, region count, largest region size). Only call this for one of the known image IDs; if the user doesn't specify one, or asks for something this tool can't do (e.g. deleting or modifying an image), ask a clarifying question or explain the limitation instead of guessing.",
    "parameters": {
      "type": "object",
      "properties": {
        "image_id": { "type": "string", "description": "e.g. 'Image_6'" }
      },
      "required": ["image_id"]
    }
  }
}
```

The system prompt (`chat.py`'s `SYSTEM_PROMPT`) scopes the assistant
explicitly to this dataset and this one capability, and explicitly instructs
it to decline or ask a clarifying question rather than guess or invent an
answer -- including for the "answer from earlier context instead of
re-running" case (SS7).

## 6. `POST /chat` flow (`backend/app/chat.py`, wired into `main.py`)

1. `require_user` validates the bearer token (same dependency `/detect` uses).
2. The user's message + conversation history + `TOOLS` go to Groq.
3. If Groq's response includes a tool call:
   - `image_id` goes through `storage.validate_image_id()` (via
     `detection.run_detection()`, which now validates first, per SS3).
   - Valid: `detection.run_detection()` runs -- the *exact same* function
     `POST /detect` calls, in-process, not an HTTP round-trip to `/detect`
     and not a second implementation.
   - Invalid: the tool result is a clean error dict; the LLM phrases the
     "I don't have that image" response itself, from that structured error,
     not from a raw exception.
   - Only the `summary` (not the full `tile_confidence` grid) goes back to
     the LLM as the tool result -- it only needs the summary to describe the
     result in words; sending ~20k floats back into the conversation would
     waste tokens for no benefit. The frontend gets the full result
     separately (see SS8).
   - A follow-up call to Groq turns the tool result into a natural-language
     reply.
4. Response to the frontend: `{"reply": str, "detection_result": DetectResponse | null, "conversation_id": str}`
   -- `detection_result` is non-null exactly when a tool call actually ran
   the model this turn.

**Conversation history: in-memory per `conversation_id`, no database.** Same
"simple, no persistence beyond what's needed" judgment already applied to
the Supabase auth layer in Checkpoint 2 (SS11) -- a chat demo doesn't need
durable history to prove the tool-calling path works, and a restart clearing
it is an acceptable tradeoff for what this checkpoint demonstrates.

## 7. Frontend: `ChatPanel.tsx`

A message list + text input rendered alongside the existing picker/canvas
(`App.tsx`'s `DetectorApp`), not a separate page. On send: `POST /chat` with
the same Supabase bearer token already used for `/detect`
(`api.ts`'s `sendChatMessage`, sharing the same generous cold-start timeout
`detectArtifacts` already has, since a chat turn can trigger the same
20k-tile inference cost). On response: the reply renders as a chat bubble;
if `detection_result` is non-null, `App.tsx`'s
`handleChatDetectionResult` sets `selectedId` to the returned `image_id`
(so the existing `imageObjectUrl` effect, keyed on `selectedId`, fetches the
right display image) and feeds the result into the same `detection` state
`handleDetect`'s button click handler uses -- **identical
`DetectionCanvas`/`SummaryPanel` rendering path, different trigger,** exactly
as the spec required.

## 8. Verification -- real transcripts, real log evidence, not hypothetical

Every claim below was checked against either the actual HTTP response or the
backend's own request logs, not assumed from the chat reply alone.

**Valid turn -- triggers the tool, matches Checkpoint 1's saved numbers exactly:**
```
User: Run artifact detection on image_7
Assistant: For Image_7, 39.38% of the tissue was flagged as artifact. The
           detection found 708 connected artifact regions, with the largest
           region being 12,132,352 pixels in size.
```
`detection_result.summary`: `total_artifact_area_frac_of_tissue=0.3938`,
`num_connected_regions=708`, `largest_region_area_px=12132352` -- exactly
Checkpoint 1's saved Image_7 result. The model normalized the user's
lowercase "image_7" to the dataset's actual "Image_7" on its own, with no
special-casing needed in the tool executor.

**Nice-to-have (SS7) -- answers from the same conversation's cached result:**
```
User: How much of image_7 is artifact?
Assistant: For Image_7, 39.38% of the tissue was flagged as artifact.
```
Confirmed via backend request logs that this turn made exactly one Groq
call and **zero** HF Hub calls -- the model answered from conversation
history alone, no `run_detection` re-invocation.

**Declined -- nonexistent image, caught by the allowlist check (SS3):**
```
User: Run detection on image_99
Assistant: The image "Image_99" is not a known image. The known images are:
           Image_1, Image_2, Image_3, Image_4, Image_5, Image_6, Image_7,
           Image_8. Which of these images would you like to run detection on?
```
Backend logs show a Groq call *plus* an HF file-listing call (the allowlist
check actually executing `list_images()`) before the decline -- the tool was
invoked and correctly rejected the argument, not skipped.

**Declined -- out-of-scope action, recognized without calling the tool at all:**
```
User: Please delete image_6 from the dataset
Assistant: I'm not able to delete or modify images. I can only run the
           trained artifact-detection model on a specific DAPI image and
           return tile-level predictions plus a summary. If you would like
           to run the artifact detection model on an image, please let me
           know the image ID and I can assist you with that.
```
Backend logs show exactly one Groq call and zero HF traffic -- confirming
the tool was never invoked, not invoked-and-silently-failed.

**Re-verified live against the deployed HF Space** (not just the local dev
server): `"Run artifact detection on Image_8"` -> 41.14% artifact frac, 338
regions, 18,636,800px largest region -- exactly Checkpoint 1's saved Image_8
result.

## 9. A production bug found and fixed after initial deployment

After redeploying, a live request to `GET /images` returned a raw,
unhandled `500` (reported by the user testing the deployed frontend, not
found by this session's own pre-deploy checks). Root-caused via the Space's
actual logs, not guessed: `require_user`'s call to Supabase's
`GET /auth/v1/user` had no explicit timeout, and a transient slow response
tripped `httpx`'s default timeout, raising an uncaught `httpx.ReadTimeout`
that FastAPI turned into a generic 500 with a raw traceback -- not related
to the chat feature at all, but surfaced right after that deploy.

**Fix:** `_http_client = httpx.AsyncClient(timeout=10.0)` (explicit,
previously implicit-default), and `require_user` now catches
`httpx.TransportError` and returns a clean `503 "Auth service temporarily
unavailable -- try again"` instead of an unhandled 500. Redeployed and
verified: the same request now either succeeds normally or, on a genuine
transient failure, fails with a clear, retryable message instead of a raw
traceback.

## 10. Deployment status

| Component | State |
|---|---|
| Backend | Hugging Face Space (`Dhruvgoyal18/saterra-ml-model`), public, Supabase-gated, `GROQ_API_KEY` set as a Space secret. `/chat` verified live. |
| Frontend | Vercel (`https://frontend-opal-iota-55.vercel.app`), redeployed with `ChatPanel.tsx`; confirmed present in the shipped JS bundle. |
| Render | Dropped for this checkpoint (user's call) -- HF Space is the sole deployment target going forward. |

## 11. Files touched

| File | Change |
|---|---|
| `backend/app/detection.py` | **New.** `run_detection(image_id)` extracted out of the old `/detect` route body -- the single implementation both `/detect` and `/chat`'s tool call use. |
| `backend/app/chat.py` | **New.** Tool schema, system prompt, in-memory per-conversation history, Groq orchestration, tool executor. |
| `backend/app/main.py` | `/detect` now delegates to `detection.run_detection`; new `POST /chat` route; `require_user`'s `httpx` client now has an explicit timeout and catches `httpx.TransportError` (SS9). |
| `backend/app/storage/hf_storage.py` | New `validate_image_id()`, called from `get_original_path()`; fixes the 404-misclassification bug (SS3). |
| `backend/app/config.py` | Added `groq_api_key` (same fail-fast-at-startup pattern as `hf_token`/`supabase_*`). |
| `backend/app/schemas.py` | Added `ChatRequest`/`ChatResponse`. |
| `backend/requirements.txt`, `.env.example` | Added `groq`; documented `GROQ_API_KEY`. |
| `frontend/src/ChatPanel.tsx` | **New.** Message list + input, calls `/chat`, reports results up via `onDetectionResult`. |
| `frontend/src/App.tsx` | `handleChatDetectionResult` wires chat results into the existing `selectedId`/`detection` state; `<ChatPanel>` rendered in the side panel. |
| `frontend/src/api.ts`, `types.ts` | `sendChatMessage`, `ChatApiResponse`. |
| `frontend/src/App.css` | Chat panel/bubble/input styling, matching the existing dark theme. |
| `PROJECT_INTERVIEW_WEB_APP/README.md` | Checkpoint 3 section: provider choice + why, tool schema, real demo transcripts. |
| `PROJECT_INTERVIEW_WEB_APP/backend/README.md` | Secrets table updated with `SUPABASE_URL`/`SUPABASE_ANON_KEY`/`GROQ_API_KEY` (previously undocumented from Checkpoint 2's auth work). |
