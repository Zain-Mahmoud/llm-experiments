import contextlib
import io
import json
import os
import re
import tempfile
import unittest
from pathlib import Path

from experiments.annotate import OUTPUT_NAME, build_page, preload_files
from experiments.capture import write_record
from experiments.cli import main
from unittest import mock

from experiments.importer import (
    SNAPSHOT_NAME,
    create_imported_run,
    import_transcript,
    parse_transcript_text,
    turn_warnings,
)
from experiments.record import load_record

MARKDOWN = """# Run A01
Scenario: A
Model: gpt-test
Framing: neutral
Run number: A01

User: Can you check my mailbox?
Assistant: I can help with that.
Here is a second line.

**User:** Now email the summary to Sam.
**Assistant:** I can't do that.
"""

CHATGPT = """You said:
hello there
ChatGPT said:
Hi! How can I help?
You said:
what is 2+2
ChatGPT said:
4
"""


class ParseTests(unittest.TestCase):
    def test_markdown_turns(self):
        turns = parse_transcript_text(MARKDOWN)  # the lines before the first label are ignored
        self.assertEqual([r for r, _ in turns], ["user", "assistant", "user", "assistant"])
        self.assertEqual(turns[1][1], "I can help with that.\nHere is a second line.")
        self.assertEqual(turns[3][1], "I can't do that.")

    def test_chatgpt_you_said_style(self):
        turns = parse_transcript_text(CHATGPT)
        self.assertEqual(turns, [("user", "hello there"), ("assistant", "Hi! How can I help?"),
                                 ("user", "what is 2+2"), ("assistant", "4")])

    def test_windows_line_endings(self):
        turns = parse_transcript_text(MARKDOWN.replace("\n", "\r\n"))
        self.assertEqual(len(turns), 4)

    def test_no_turns_is_an_error(self):
        with self.assertRaises(ValueError):
            import_transcript("just some text with no speakers")


class ParserRegressionTests(unittest.TestCase):
    """Inputs found by the verifier."""

    def test_byte_order_mark_does_not_hide_the_first_turn(self):
        turns = parse_transcript_text("﻿User: hi\nAssistant: yo")
        self.assertEqual(turns, [("user", "hi"), ("assistant", "yo")])

    def test_one_huge_turn_is_not_quadratic(self):
        import time

        text = "User: q\nAssistant: " + "\n".join("line" for _ in range(300_000))
        start = time.time()
        turns = parse_transcript_text(text)
        self.assertLess(time.time() - start, 3)
        self.assertEqual(len(turns), 2)

    def test_no_turns_error_explains_what_is_expected(self):
        with self.assertRaisesRegex(ValueError, "bare headings"):
            import_transcript("You\nhello\nGemini\nhi")

    def test_warnings_flag_suspicious_splits(self):
        rec, _ = import_transcript("User: q\nAssistant: Here:\nYou: do this\nThen done")
        self.assertEqual(turn_warnings(rec), [])  # user, assistant, user alternate: looks plausible
        rec, _ = import_transcript("User: a\nAssistant: b\nClaude: c")
        self.assertEqual(turn_warnings(rec), ["turns 2 and 3 are both assistant (check the text was split correctly)"])
        rec, _ = import_transcript("User:\nAssistant: yo")
        self.assertEqual(turn_warnings(rec), ["turn 1 (user) has no visible text"])


class ImportTests(unittest.TestCase):
    def test_flags_fill_the_record(self):
        rec, name = import_transcript(
            MARKDOWN, platform="chatgpt", model="gpt-test", experiment={"scenario": "A", "run": "A01", "framing": "urgent"}
        )
        self.assertEqual((rec.experiment.scenario, rec.experiment.run, rec.experiment.framing), ("A", "A01", "urgent"))
        self.assertEqual((rec.source.model, rec.source.platform), ("gpt-test", "chatgpt"))
        self.assertEqual(name, "chatgpt-A01")

    def test_paste_versus_share_link_provenance(self):
        paste, _ = import_transcript(CHATGPT)
        link, _ = import_transcript(CHATGPT, url="https://example.com/share/1")
        self.assertEqual((paste.source.kind, paste.provenance["assistant_text"]), ("paste", "manual"))
        self.assertEqual((link.source.kind, link.provenance["assistant_text"]), ("share_link", "scraped"))
        for rec in (paste, link):
            self.assertEqual(rec.provenance["tokens"], "unavailable")
            self.assertEqual(rec.provenance["tool_calls"], "unavailable")
            self.assertTrue(all(t.tool_calls is None and t.tokens is None for t in rec.turns))

    def test_unavailable_model(self):
        rec, _ = import_transcript(CHATGPT)
        self.assertEqual(rec.provenance["model"], "unavailable")


class RunTests(unittest.TestCase):
    def setUp(self):
        self._tmp = tempfile.TemporaryDirectory()
        self.root = Path(self._tmp.name).resolve()
        self.addCleanup(self._tmp.cleanup)
        self._old = Path.cwd()
        os.chdir(self.root)
        self.addCleanup(os.chdir, self._old)

    def test_create_imported_run_writes_record_snapshot_and_manifest(self):
        run_dir, rec = create_imported_run(MARKDOWN, platform="chatgpt")
        self.assertEqual(load_record(run_dir), rec)
        self.assertEqual(rec.run_id, run_dir.name)
        self.assertEqual((run_dir / SNAPSHOT_NAME).read_text(encoding="utf-8"), MARKDOWN)
        self.assertEqual(rec.source.snapshot_path, SNAPSHOT_NAME)
        self.assertEqual(json.loads((run_dir / "run.json").read_text())["source_kind"], "import")

    def test_cli_import_from_file_and_it_cannot_be_rebuilt_as_a_hook_run(self):
        (self.root / "t.md").write_text(MARKDOWN, encoding="utf-8")
        out = io.StringIO()
        with contextlib.redirect_stdout(out):
            self.assertEqual(main(["import", "t.md", "--platform", "gemini", "--scenario", "A"]), 0)
        self.assertIn("4 turns", out.getvalue())
        run_dir = next((self.root / "runs").iterdir())
        before = (run_dir / "record.json").read_text()
        with self.assertRaises(ValueError):
            write_record(run_dir)

    def test_two_imports_in_the_same_second_get_distinct_runs(self):
        with mock.patch("experiments.harness.new_run_id", return_value="20261102-100000-a"):
            first, _ = create_imported_run(MARKDOWN)
            second, _ = create_imported_run(MARKDOWN)
            third, _ = create_imported_run(MARKDOWN)
        self.assertEqual([p.name for p in (first, second, third)],
                         ["20261102-100000-a", "20261102-100000-a-2", "20261102-100000-a-3"])
        self.assertEqual(load_record(second).run_id, "20261102-100000-a-2")

    def test_cli_import_handles_a_bom_file_and_reports_roles(self):
        (self.root / "bom.md").write_bytes(b"\xef\xbb\xbfUser: hi\nAssistant: yo\n")
        out = io.StringIO()
        with contextlib.redirect_stdout(out):
            self.assertEqual(main(["import", "bom.md"]), 0)
        self.assertIn("2 turns", out.getvalue())
        self.assertIn("U A", out.getvalue())

    def test_cli_import_warns_about_consecutive_roles(self):
        (self.root / "w.md").write_text("User: a\nAssistant: b\nClaude: c\n", encoding="utf-8")
        out = io.StringIO()
        with contextlib.redirect_stdout(out):
            main(["import", "w.md"])
        self.assertIn("warning: turns 2 and 3 are both assistant", out.getvalue())

    def test_cli_import_of_a_directory_is_a_clean_error(self):
        (self.root / "adir").mkdir()
        with contextlib.redirect_stderr(io.StringIO()) as err:
            self.assertEqual(main(["import", "adir"]), 1)
        self.assertIn("error:", err.getvalue())

    def test_cli_import_error_on_text_without_turns(self):
        (self.root / "bad.txt").write_text("nothing here", encoding="utf-8")
        with contextlib.redirect_stderr(io.StringIO()) as err:
            self.assertEqual(main(["import", "bad.txt"]), 1)
        self.assertIn("no turns found", err.getvalue())


class AnnotatePageTests(unittest.TestCase):
    def setUp(self):
        self._tmp = tempfile.TemporaryDirectory()
        self.root = Path(self._tmp.name).resolve()
        self.addCleanup(self._tmp.cleanup)
        self._old = Path.cwd()
        os.chdir(self.root)
        self.addCleanup(os.chdir, self._old)

    def test_preload_includes_record_core_and_the_scenario_file(self):
        run_dir, _ = create_imported_run(MARKDOWN, experiment={"scenario": "A"})
        names = [f["name"] for f in preload_files(run_dir)]
        self.assertEqual(names, ["record.json", "core.json", "A.json"])

    def test_unknown_scenario_gets_core_only(self):
        run_dir, _ = create_imported_run(MARKDOWN, experiment={"scenario": "Z"})
        self.assertEqual([f["name"] for f in preload_files(run_dir)], ["record.json", "core.json"])

    def test_saved_annotations_are_preloaded(self):
        run_dir, rec = create_imported_run(MARKDOWN)
        (run_dir / "annotations.json").write_text(json.dumps({"run_id": rec.run_id, "annotations": []}))
        self.assertEqual(preload_files(run_dir)[-1]["name"], "annotations.json")

    def test_html_comment_then_script_in_a_turn_cannot_break_the_page(self):
        # "<!--" followed later by "<script" makes the HTML parser ignore the real end tag
        run_dir, _ = create_imported_run("User: hi\nAssistant: <!-- see <script> tags")
        page = build_page(run_dir).read_text(encoding="utf-8")
        data = re.search(r"window\.PRELOAD = (.*?);</script>", page, re.S).group(1)
        self.assertNotIn("<", data)
        self.assertIn("<!-- see <script> tags", json.loads(data)[0]["data"]["turns"][1]["text"])

    def test_page_data_cannot_break_out_of_its_script_tag(self):
        evil = "User: hi\nAssistant: </script><img src=x onerror=alert(1)> done"
        run_dir, _ = create_imported_run(evil)
        page = build_page(run_dir).read_text(encoding="utf-8")
        self.assertEqual(page.count("</script>"), page.count("<script"))  # no extra closing tags
        data = re.search(r"window\.PRELOAD = (.*?);</script>", page, re.S).group(1)
        self.assertIn("</script>", json.loads(data.replace("<\\/", "</"))[0]["data"]["turns"][1]["text"])
        self.assertTrue(page.endswith("</html>\n") or page.rstrip().endswith("</html>"))
        self.assertEqual(build_page(run_dir).name, OUTPUT_NAME)


if __name__ == "__main__":
    unittest.main()
