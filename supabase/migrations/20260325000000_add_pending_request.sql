-- Add pending_request column to slack_thread_contexts for auto re-execution
-- after repo clarification.
ALTER TABLE slack_thread_contexts
  ADD COLUMN IF NOT EXISTS pending_request TEXT;

