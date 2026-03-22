import unittest

from slack_bot_backend.services.indexer import (
    AST_NODE_MAX_CHARS,
    CHUNK_MAX_CHARS,
    CHUNK_OVERLAP_CHARS,
    _ast_chunk,
    _chunk_text,
    _chunk_text_fallback,
)


class ChunkTextFallbackTests(unittest.TestCase):
    """Tests for the character-based _chunk_text_fallback helper."""

    def test_short_text_returns_single_chunk(self):
        result = _chunk_text_fallback("hello world")
        self.assertEqual(result, ["hello world"])

    def test_text_at_max_returns_single_chunk(self):
        text = "x" * CHUNK_MAX_CHARS
        result = _chunk_text_fallback(text)
        self.assertEqual(len(result), 1)
        self.assertEqual(result[0], text)

    def test_long_text_is_split_with_overlap(self):
        text = "a" * (CHUNK_MAX_CHARS + 500)
        result = _chunk_text_fallback(text)
        self.assertGreater(len(result), 1)
        # First chunk is max_chars long
        self.assertEqual(len(result[0]), CHUNK_MAX_CHARS)
        # Overlap: second chunk starts at max_chars - overlap
        expected_start = CHUNK_MAX_CHARS - CHUNK_OVERLAP_CHARS
        self.assertEqual(result[1], text[expected_start : expected_start + CHUNK_MAX_CHARS])

    def test_custom_max_and_overlap(self):
        text = "abcdefghij"  # 10 chars
        result = _chunk_text_fallback(text, max_chars=4, overlap=2)
        self.assertEqual(result[0], "abcd")
        self.assertEqual(result[1], "cdef")


class AstChunkTests(unittest.TestCase):
    """Tests for _ast_chunk using tree-sitter parsing."""

    def test_returns_none_for_unsupported_extension(self):
        self.assertIsNone(_ast_chunk("# heading", ".md"))
        self.assertIsNone(_ast_chunk("{}", ".json"))
        self.assertIsNone(_ast_chunk("key: val", ".yaml"))

    def test_python_functions_produce_separate_chunks(self):
        code = (
            "def foo():\n"
            "    pass\n"
            "\n"
            "def bar():\n"
            "    return 1\n"
        )
        chunks = _ast_chunk(code, ".py")
        self.assertIsNotNone(chunks)
        self.assertEqual(len(chunks), 2)
        self.assertIn("def foo", chunks[0])
        self.assertIn("def bar", chunks[1])

    def test_python_class_is_single_chunk(self):
        code = (
            "class Greeter:\n"
            "    def greet(self):\n"
            "        return 'hi'\n"
        )
        chunks = _ast_chunk(code, ".py")
        self.assertIsNotNone(chunks)
        self.assertEqual(len(chunks), 1)
        self.assertIn("class Greeter", chunks[0])
        self.assertIn("def greet", chunks[0])

    def test_python_imports_grouped_as_preamble(self):
        code = (
            "import os\n"
            "import sys\n"
            "\n"
            "def main():\n"
            "    pass\n"
        )
        chunks = _ast_chunk(code, ".py")
        self.assertIsNotNone(chunks)
        self.assertEqual(len(chunks), 2)
        self.assertIn("import os", chunks[0])
        self.assertIn("import sys", chunks[0])
        self.assertIn("def main", chunks[1])

    def test_typescript_function_and_interface(self):
        code = (
            "interface User { name: string; }\n"
            "\n"
            "function greet(u: User): string {\n"
            "  return u.name;\n"
            "}\n"
        )
        chunks = _ast_chunk(code, ".ts")
        self.assertIsNotNone(chunks)
        self.assertEqual(len(chunks), 2)
        self.assertIn("interface User", chunks[0])
        self.assertIn("function greet", chunks[1])

    def test_javascript_functions(self):
        code = (
            "function add(a, b) { return a + b; }\n"
            "\n"
            "function sub(a, b) { return a - b; }\n"
        )
        chunks = _ast_chunk(code, ".js")
        self.assertIsNotNone(chunks)
        self.assertEqual(len(chunks), 2)

    def test_jsx_extension_is_supported(self):
        code = "function App() { return null; }\n"
        chunks = _ast_chunk(code, ".jsx")
        self.assertIsNotNone(chunks)

    def test_tsx_extension_is_supported(self):
        code = "function App(): JSX.Element { return <div/>; }\n"
        chunks = _ast_chunk(code, ".tsx")
        self.assertIsNotNone(chunks)

    def test_syntax_error_returns_none(self):
        bad_code = "def (\n\n\n class %%%\n"
        result = _ast_chunk(bad_code, ".py")
        self.assertIsNone(result)


    def test_oversized_node_is_sub_chunked(self):
        body = "    x = 1\n" * (AST_NODE_MAX_CHARS // 10 + 1)
        code = f"def big():\n{body}"
        chunks = _ast_chunk(code, ".py")
        self.assertIsNotNone(chunks)
        self.assertGreater(len(chunks), 1, "oversized function should be sub-chunked")
        for chunk in chunks:
            self.assertLessEqual(len(chunk), CHUNK_MAX_CHARS)

    def test_only_preamble_file(self):
        code = "import os\nimport sys\n"
        chunks = _ast_chunk(code, ".py")
        self.assertIsNotNone(chunks)
        self.assertEqual(len(chunks), 1)
        self.assertIn("import os", chunks[0])


class ChunkTextDispatcherTests(unittest.TestCase):
    """Tests for the top-level _chunk_text dispatcher."""

    def test_backward_compatible_no_ext(self):
        result = _chunk_text("short text")
        self.assertEqual(result, ["short text"])

    def test_backward_compatible_no_ext_long(self):
        text = "a" * (CHUNK_MAX_CHARS + 100)
        result = _chunk_text(text)
        self.assertGreater(len(result), 1)

    def test_python_ext_uses_ast(self):
        code = "def a():\n    pass\n\ndef b():\n    pass\n"
        result = _chunk_text(code, ext=".py")
        self.assertEqual(len(result), 2)
        self.assertIn("def a", result[0])
        self.assertIn("def b", result[1])

    def test_markdown_ext_uses_fallback(self):
        text = "# Hello\nworld"
        result = _chunk_text(text, ext=".md")
        self.assertEqual(result, [text])

    def test_invalid_code_falls_back(self):
        bad = "def (\n class %%%\n"
        result = _chunk_text(bad, ext=".py")
        # Should not raise — falls back to character-based
        self.assertIsInstance(result, list)
        self.assertTrue(len(result) >= 1)

    def test_custom_max_chars_used_in_fallback(self):
        text = "abcdefghij"
        result = _chunk_text(text, max_chars=4, overlap=2)
        self.assertEqual(result[0], "abcd")


if __name__ == "__main__":
    unittest.main()

