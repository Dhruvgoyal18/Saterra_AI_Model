import { createClient } from "@supabase/supabase-js";

// Safe to expose client-side -- the anon key only grants what Supabase's
// public auth endpoints (sign in, get session) already allow; it is not a
// master credential the way the backend's HF_TOKEN is.
export const supabase = createClient(
  import.meta.env.VITE_SUPABASE_URL,
  import.meta.env.VITE_SUPABASE_ANON_KEY
);
