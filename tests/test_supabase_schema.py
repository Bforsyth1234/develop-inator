from pathlib import Path
import unittest


class SupabaseSchemaTests(unittest.TestCase):
    def test_schema_contains_required_pgvector_objects(self) -> None:
        migration = Path("supabase/migrations/20260320164500_slack_bot_persistence.sql").read_text()

        self.assertIn("create extension if not exists vector;", migration)
        self.assertIn("create table if not exists public.slack_thread_messages", migration)
        self.assertIn("create table if not exists public.documentation_chunks", migration)
        self.assertIn("using ivfflat (embedding vector_cosine_ops)", migration)
        self.assertIn("create or replace function public.match_documentation_chunks", migration)


if __name__ == "__main__":
    unittest.main()