import json
from pathlib import Path
import tempfile
import unittest

from check_links import check
from prepare import evidence_section, evidence_summary


class DocumentationTests(unittest.TestCase):
    def test_links_and_fragments(self):
        with tempfile.TemporaryDirectory() as directory:
            root = Path(directory)
            (root / "index.html").write_text('<a href="other.html#target">ok</a><img src="p.png">')
            (root / "other.html").write_text('<p id="target">Found</p>')
            (root / "p.png").write_bytes(b"fixture")
            self.assertEqual(check(root), [])
            (root / "other.html").write_text('<p id="wrong">Missing</p>')
            self.assertIn("missing fragment", check(root)[0])

    def test_missing_asset_and_escape(self):
        with tempfile.TemporaryDirectory() as directory:
            root = Path(directory)
            (root / "index.html").write_text('<a href="../outside.html">bad</a><script src="missing.js"></script>')
            self.assertEqual(len(check(root)), 2)

    def test_evidence_preserves_failure_and_escapes_html(self):
        with tempfile.TemporaryDirectory() as directory:
            path = Path(directory) / "validation.json"
            path.write_text(json.dumps({"status": "failed", "detail": "<script>bad</script>"}))
            rendered = evidence_section(path)
            self.assertIn("failed", rendered)
            self.assertNotIn("<script>", rendered)
            self.assertIn("&lt;script&gt;", rendered)

    def test_toml_and_invalid_evidence(self):
        with tempfile.TemporaryDirectory() as directory:
            path = Path(directory) / "benchmark.toml"
            path.write_text('status = "skipped"\n')
            self.assertIn("skipped", evidence_section(path))
            path = Path(directory) / "validation.json"
            path.write_text("null")
            with self.assertRaises(ValueError):
                evidence_section(path)

    def test_stale_and_missing_backend_are_explicit(self):
        data = {"backends": {
            "cpu": {"status": "PASS", "assertions_passed": 2, "source_fingerprint_sha256": "old"},
            "cuda": {"status": "NOT_RUN"},
        }}
        rendered = evidence_summary(data, "new")
        self.assertIn("STALE: source differs", rendered)
        self.assertIn("not yet measured (NOT_RUN)", rendered)
        self.assertIn("Assertions passed", rendered)

    def test_matching_fingerprint(self):
        rendered = evidence_summary({"status": "FAIL", "source_fingerprint_sha256": "same"}, "same")
        self.assertIn("matches source at render", rendered)
        self.assertIn("FAIL", rendered)


if __name__ == "__main__":
    unittest.main()
