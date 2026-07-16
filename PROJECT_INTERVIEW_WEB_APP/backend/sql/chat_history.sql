-- Durable, per-user chat history for POST /chat (see backend/app/chat.py and
-- backend/app/storage/chat_storage.py). Apply this once via the Supabase
-- dashboard's SQL editor for the same project already used for auth -- there
-- is no migration tooling in this repo.
--
-- After running this, smoke-test RLS before trusting it: create a session as
-- user A, then try `GET /rest/v1/chat_sessions?id=eq.<A's id>` using user B's
-- token + the shared anon key, and confirm an empty array comes back. A typo
-- in these policies fails OPEN (silent cross-user leakage), not closed.

create table chat_sessions (
    id          uuid primary key default gen_random_uuid(),
    user_id     uuid not null references auth.users(id) default auth.uid(),
    title       text,
    created_at  timestamptz not null default now(),
    updated_at  timestamptz not null default now()
);

create table chat_messages (
    id          bigserial primary key,
    session_id  uuid not null references chat_sessions(id) on delete cascade,
    user_id     uuid not null default auth.uid(),
    role        text not null check (role in ('user', 'assistant', 'tool')),
    content     text,
    tool_name   text,
    tool_args   jsonb,
    tool_result jsonb,
    created_at  timestamptz not null default now()
);

create index chat_sessions_user_id_updated_at_idx on chat_sessions (user_id, updated_at desc);
create index chat_messages_session_id_id_idx on chat_messages (session_id, id);

alter table chat_sessions enable row level security;
alter table chat_messages enable row level security;

create policy chat_sessions_select_own on chat_sessions
    for select using (auth.uid() = user_id);
create policy chat_sessions_insert_own on chat_sessions
    for insert with check (auth.uid() = user_id);
-- No update/delete policy: sessions are read-only after creation by design
-- (viewing history never re-enters a live, editable conversation).
-- updated_at is bumped by the trigger below, not a client PATCH, so no
-- update policy is needed on chat_sessions at all.

create policy chat_messages_select_own on chat_messages
    for select using (auth.uid() = user_id);
create policy chat_messages_insert_own on chat_messages
    for insert with check (
        auth.uid() = user_id
        and exists (select 1 from chat_sessions s where s.id = session_id and s.user_id = auth.uid())
    );
-- The exists() check matters even with the denormalized user_id column above:
-- without it, a client could still send its own user_id but someone else's
-- session_id and have the row accepted.

create function bump_session_updated_at() returns trigger as $$
begin
  update chat_sessions set updated_at = now() where id = new.session_id;
  return new;
end;
$$ language plpgsql security definer;

create trigger chat_messages_bump_session after insert on chat_messages
  for each row execute function bump_session_updated_at();
