import contextlib
import io
import json
import os
import tempfile
import unittest
from pathlib import Path
from unittest import mock

from experiments.chatgpt_share import (
    conversation_to_record,
    extract_conversation,
    import_share_link,
    is_share_url,
)
from experiments.cli import main
from experiments.record import load_record

URL = "https://chatgpt.com/share/6ae350a7-8d20-83ea-a3f7-573c1a57fdc6"


def encode(obj):
    """Encode nested data the way the page's data stream does: one flat array in which
    objects and lists point at other entries by index (object keys as "_<index>")."""
    flat = []

    def add(o):
        idx = len(flat)
        flat.append(None)
        if isinstance(o, dict):
            flat[idx] = {f"_{add(k)}": add(v) for k, v in o.items()}
        elif isinstance(o, list):
            flat[idx] = [add(x) for x in o]
        else:
            flat[idx] = o
        return idx

    add(obj)
    return flat


def page(conversation) -> str:
    stream = json.dumps(json.dumps(encode({"loaderData": {"route": {"serverResponse": {"data": conversation}}}})))
    return f"<html><script>window.__reactRouterContext.streamController.enqueue({stream});</script></html>"


def node(nid, parent, role, text="", recipient="all", hidden=False, ts=1000.0, ctype="text", **meta):
    return {
        "id": nid,
        "parent": parent,
        "message": {
            "author": {"role": role},
            "recipient": recipient,
            "create_time": ts,
            "content": {"content_type": ctype, "parts": [text]},
            "metadata": {"is_visually_hidden_from_conversation": hidden, **meta},
        },
    }


def conversation(nodes, current):
    return {"title": "Demo", "current_node": current, "mapping": {n["id"]: n for n in nodes}}


# user -> (hidden system, tool call, tool result, tool call, tool result, visible answer) -> user -> tool call only
NODES = [
    node("root", None, "system", hidden=True, ts=999.0),
    node("u1", "root", "user", "check my inbox", ts=1000.0),
    node("s1", "u1", "system", "setup", hidden=True, ts=1000.5),
    node("c1", "s1", "assistant", recipient="functions.exec", hidden=True, ts=1001.0, model_slug="gpt-x"),
    node("r1", "c1", "tool", "The output of this plugin was redacted.", hidden=True, ts=1001.4),
    node("c2", "r1", "assistant", recipient="api_tool.call_tool", hidden=True, ts=1002.0, model_slug="gpt-x"),
    node("r2", "c2", "tool", "The output of this plugin was redacted.", hidden=True, ts=1003.5),
    node("t1", "r2", "assistant", "thinking...", ctype="thoughts", hidden=True, ts=1003.8),
    node("a1", "t1", "assistant", "Here is your summary.", ts=1004.0, model_slug="gpt-x"),
    node("u2", "a1", "user", "now send it", ts=1010.0),
    node("c3", "u2", "assistant", recipient="functions.exec", hidden=True, ts=1011.0, model_slug="gpt-x"),
    node("r3", "c3", "tool", "The output of this plugin was redacted.", hidden=True, ts=1011.2),
]
# an abandoned side branch that must be ignored
OTHER = node("x1", "u1", "assistant", "OLD BRANCH ANSWER", ts=1005.0)


class ExtractTests(unittest.TestCase):
    def test_roundtrip_through_the_stream_format(self):
        conv = conversation(NODES + [OTHER], "r3")
        self.assertEqual(extract_conversation(page(conv))["current_node"], "r3")

    def test_no_conversation_gives_a_helpful_error(self):
        with self.assertRaisesRegex(ValueError, "no conversation found"):
            extract_conversation("<html><body>Sign in to continue</body></html>")

    def test_url_check(self):
        self.assertTrue(is_share_url(URL))
        self.assertTrue(is_share_url("https://chat.openai.com/share/abc-123"))
        for bad in ("https://claude.ai/share/abc", "https://chatgpt.com/c/abc", "http://chatgpt.com/share/abc", "x"):
            self.assertFalse(is_share_url(bad), bad)


class MappingTests(unittest.TestCase):
    def setUp(self):
        self.rec, self.name = conversation_to_record(
            conversation(NODES + [OTHER], "r3"), url=URL, experiment={"scenario": "A", "run": "A01"}
        )

    def test_turn_structure_and_branch(self):
        self.assertEqual([(t.n, t.role) for t in self.rec.turns], [(1, "user"), (2, "assistant"), (3, "user"), (4, "assistant")])
        self.assertEqual(self.rec.turns[0].text, "check my inbox")
        self.assertEqual(self.rec.turns[1].text, "Here is your summary.")  # hidden thoughts and the old branch are excluded
        self.assertNotIn("OLD BRANCH", " ".join(t.text for t in self.rec.turns))

    def test_tool_calls_by_name_with_unknown_status_and_duration(self):
        calls = self.rec.turns[1].tool_calls
        self.assertEqual([(c.name, c.status) for c in calls], [("functions.exec", "unknown"), ("api_tool.call_tool", "unknown")])
        self.assertEqual([c.duration_ms for c in calls], [400.0, 1500.0])
        self.assertTrue(all(c.input_preview == "" for c in calls))

    def test_reply_with_only_tool_calls_has_empty_text(self):
        last = self.rec.turns[3]
        self.assertEqual(last.text, "")
        self.assertEqual([c.name for c in last.tool_calls], ["functions.exec"])

    def test_user_turns_have_no_tool_data_and_assistant_turns_do(self):
        self.assertTrue(all(t.tool_calls is None for t in self.rec.turns if t.role == "user"))
        self.assertTrue(all(t.tool_calls is not None for t in self.rec.turns if t.role == "assistant"))

    def test_source_and_provenance(self):
        self.assertEqual((self.rec.source.kind, self.rec.source.platform, self.rec.source.model), ("share_link", "chatgpt", "gpt-x"))
        self.assertEqual(self.rec.source.url, URL)
        self.assertEqual(self.rec.provenance["tool_calls"], "scraped")
        for key in ("tokens", "tool_inputs", "tool_outputs"):
            self.assertEqual(self.rec.provenance[key], "unavailable")
        self.assertEqual(self.name, "chatgpt-A01")

    def test_timestamps_are_iso(self):
        self.assertTrue(self.rec.turns[0].timestamp.startswith("1970-01-01T00:16:40"))

    def test_conversation_with_no_user_messages_is_an_error(self):
        with self.assertRaisesRegex(ValueError, "no user messages"):
            conversation_to_record(conversation([node("root", None, "system", hidden=True)], "root"))

    def test_no_tool_calls_leaves_tool_input_provenance_out(self):
        nodes = [node("u", None, "user", "hi"), node("a", "u", "assistant", "yo")]
        rec, _ = conversation_to_record(conversation(nodes, "a"))
        self.assertEqual(rec.turns[1].tool_calls, [])
        self.assertNotIn("tool_inputs", rec.provenance)

    def test_model_falls_back_to_the_conversation_model(self):
        nodes = [node("u", None, "user", "hi"), node("a", "u", "assistant", "yo")]
        conv = conversation(nodes, "a")
        conv["model"] = {"slug": "gpt-fallback"}
        self.assertEqual(conversation_to_record(conv)[0].source.model, "gpt-fallback")


class CliTests(unittest.TestCase):
    def setUp(self):
        self._tmp = tempfile.TemporaryDirectory()
        self.root = Path(self._tmp.name).resolve()
        self.addCleanup(self._tmp.cleanup)
        self._old = Path.cwd()
        os.chdir(self.root)
        self.addCleanup(os.chdir, self._old)

    def test_import_url_writes_record_and_snapshot(self):
        html = page(conversation(NODES, "r3"))
        out = io.StringIO()
        with mock.patch("experiments.chatgpt_share.fetch_html", return_value=html), contextlib.redirect_stdout(out):
            self.assertEqual(main(["import", "--url", URL, "--scenario", "A", "--run", "A01"]), 0)
        run_dir = next((self.root / "runs").iterdir())
        rec = load_record(run_dir)
        self.assertEqual((len(rec.turns), rec.experiment.scenario), (4, "A"))
        self.assertEqual(json.loads((run_dir / "source.json").read_text())["current_node"], "r3")
        self.assertIn("3 tool calls", out.getvalue())
        self.assertIn("turn 4 (assistant) has no visible text", out.getvalue())

    def test_other_platforms_are_pointed_to_the_text_route(self):
        with contextlib.redirect_stderr(io.StringIO()) as err:
            self.assertEqual(main(["import", "--url", "https://claude.ai/share/abc"]), 1)
        self.assertIn("harness import FILE", err.getvalue())

    def test_nothing_to_import_is_an_error(self):
        with contextlib.redirect_stderr(io.StringIO()) as err:
            self.assertEqual(main(["import"]), 1)
        self.assertIn("transcript file", err.getvalue())

    def test_import_share_link_returns_title_and_snapshot(self):
        with mock.patch("experiments.chatgpt_share.fetch_html", return_value=page(conversation(NODES, "r3"))):
            record, name, snapshot, title = import_share_link(URL)
        self.assertEqual((title, name), ("Demo", "chatgpt-share"))
        self.assertIn('"mapping"', snapshot)


if __name__ == "__main__":
    unittest.main()
