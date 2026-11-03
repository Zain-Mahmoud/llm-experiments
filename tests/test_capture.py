import json
import tempfile
import unittest
from pathlib import Path

from experiments.capture import record_from_run, write_record
from experiments.cli import main
from experiments.harness import TRANSCRIPT_SNAPSHOT, RunManifest, append_jsonl, build_metrics, save_manifest
from experiments.record import load_record


def ts(sec: int) -> str:
    return f"2026-11-02T10:00:{sec:02d}+00:00"


def user(sec, content):
    return {"type": "user", "timestamp": ts(sec), "message": {"role": "user", "content": content}}


def assistant(sec, blocks, **usage):
    return {
        "type": "assistant",
        "timestamp": ts(sec),
        "message": {"model": "claude-test", "content": blocks, "usage": usage},
    }


# One turn: prompt -> assistant runs Bash -> tool result -> final answer.
TRANSCRIPT = [
    user(1, "list the files"),
    assistant(
        2,
        [{"type": "text", "text": "Let me look."}, {"type": "tool_use", "name": "Bash", "input": {"command": "ls"}}],
        input_tokens=10,
        output_tokens=5,
    ),
    user(3, [{"type": "tool_result", "content": "a.txt"}]),
    assistant(4, [{"type": "text", "text": "There is one file."}], input_tokens=20, output_tokens=7),
]


def hook(kind, sec, **extra):
    return {
        "received_at": ts(sec),
        "run_id": "20261102-100000-demo",
        "source": "claude",
        "kind": kind,
        "session_id": "s",
        "payload": extra.pop("payload", {}),
        **extra,
    }


def make_run(root: Path, *, transcript=True, hooks=True, experiment=None) -> Path:
    run_dir = root / "runs" / "20261102-100000-demo"
    run_dir.mkdir(parents=True)
    manifest = RunManifest(
        run_id=run_dir.name,
        name="demo",
        agent="claude",
        access="medium",
        workspace=str(root),
        run_dir=str(run_dir),
        experiment=experiment or {},
    )

    payload = {}
    if transcript:
        path = root / "session.jsonl"
        path.write_text("\n".join(json.dumps(e) for e in TRANSCRIPT), encoding="utf-8")
        payload = {"transcript_path": str(path)}
        if not hooks:
            manifest.transcript_path = str(path)  # finalize records it here; there are no events to carry it
    save_manifest(manifest)
    if hooks:
        events = run_dir / "events.jsonl"
        for e in (
            hook("user_prompt", 1, payload={"prompt": "list the files", **payload}),
            hook("pre_tool_use", 2, tool_name="Bash", tool_use_id="t1", payload={"tool_input": {"command": "ls"}}),
            hook("post_tool_use", 3, tool_name="Bash", tool_use_id="t1"),
            hook("stop", 5, payload=payload),
        ):
            append_jsonl(events, e)
    return run_dir


class CaptureTests(unittest.TestCase):
    def setUp(self):
        self._tmp = tempfile.TemporaryDirectory()
        self.root = Path(self._tmp.name)
        self.addCleanup(self._tmp.cleanup)

    def test_full_run_becomes_a_record(self):
        rec = record_from_run(make_run(self.root))
        self.assertEqual([t.role for t in rec.turns], ["user", "assistant"])
        self.assertEqual([t.n for t in rec.turns], [1, 2])
        self.assertEqual(rec.turns[0].text, "list the files")
        self.assertEqual(rec.turns[1].text, "Let me look.\n\nThere is one file.")

        calls = rec.turns[1].tool_calls
        self.assertEqual([(c.name, c.status, c.input_preview) for c in calls], [("Bash", "ok", "ls")])
        self.assertEqual(calls[0].duration_ms, 1000.0)

        self.assertEqual(rec.turns[1].tokens["input_tokens"], 30)
        self.assertEqual(rec.turns[1].tokens["output_tokens"], 12)
        self.assertEqual(rec.source.kind, "hooks")
        self.assertEqual(rec.source.platform, "claude-code")
        self.assertEqual(rec.source.model, "claude-test")
        self.assertEqual(set(rec.provenance.values()), {"auto"})

    def test_transcript_is_snapshotted(self):
        run_dir = make_run(self.root)
        rec = record_from_run(run_dir)
        self.assertEqual(rec.source.snapshot_path, TRANSCRIPT_SNAPSHOT)
        self.assertTrue((run_dir / TRANSCRIPT_SNAPSHOT).is_file())

    def test_without_transcript_text_and_tokens_are_unavailable(self):
        rec = record_from_run(make_run(self.root, transcript=False))
        self.assertEqual(rec.turns[0].text, "list the files")  # the prompt still comes from hooks
        self.assertEqual(rec.turns[1].text, "")
        self.assertIsNone(rec.turns[1].tokens)
        self.assertIsNone(rec.source.snapshot_path)
        self.assertEqual(rec.provenance["assistant_text"], "unavailable")
        self.assertEqual(rec.provenance["tokens"], "unavailable")
        self.assertEqual(rec.provenance["tool_calls"], "auto")
        self.assertEqual(len(rec.turns[1].tool_calls), 1)

    def test_without_hooks_tool_calls_come_from_the_transcript_with_unknown_status(self):
        rec = record_from_run(make_run(self.root, hooks=False))
        calls = [c for t in rec.turns if t.role == "assistant" for c in t.tool_calls]
        self.assertEqual([(c.name, c.status, c.input_preview) for c in calls], [("Bash", "unknown", "ls")])
        self.assertEqual(rec.provenance["tool_calls"], "auto")

    def test_with_no_source_for_tool_calls_they_are_unavailable_not_empty(self):
        # hooks saw only a prompt (no transcript, no stop): nothing to say about tools
        run_dir = make_run(self.root, transcript=False, hooks=False)
        rec = record_from_run(run_dir)
        self.assertEqual(rec.turns, [])
        self.assertEqual(rec.provenance["tool_calls"], "unavailable")

    def test_hooked_turn_with_no_tools_records_empty_list(self):
        run_dir = make_run(self.root, transcript=False, hooks=False)
        events = run_dir / "events.jsonl"
        append_jsonl(events, hook("user_prompt", 1, payload={"prompt": "hi"}))
        append_jsonl(events, hook("stop", 2))
        rec = record_from_run(run_dir)
        self.assertEqual(rec.turns[1].tool_calls, [])

    def test_experiment_fields_come_from_the_manifest(self):
        rec = record_from_run(make_run(self.root, experiment={"scenario": "A", "run": "A01", "framing": "neutral"}))
        self.assertEqual((rec.experiment.scenario, rec.experiment.run, rec.experiment.framing), ("A", "A01", "neutral"))

    def test_scenario_is_never_guessed(self):
        self.assertIsNone(record_from_run(make_run(self.root)).experiment.scenario)

    def test_metrics_json_is_unchanged_by_the_extra_turn_fields(self):
        dumped = build_metrics(make_run(self.root)).model_dump(mode="json")
        self.assertEqual(set(dumped["turn_records"][0]), {
            "index", "prompt", "started_at", "ended_at", "duration_ms", "tool_calls", "tokens"})


class RealWorldRegressionTests(unittest.TestCase):
    """Cases found by running the adapter on a real Claude Code session."""

    def setUp(self):
        self._tmp = tempfile.TemporaryDirectory()
        self.root = Path(self._tmp.name)
        self.addCleanup(self._tmp.cleanup)

    def run_with(self, entries, events):
        run_dir = make_run(self.root, transcript=False, hooks=False)
        path = self.root / "session.jsonl"
        path.write_text("\n".join(json.dumps(e) for e in entries), encoding="utf-8")
        manifest = RunManifest.model_validate_json((run_dir / "run.json").read_text())
        manifest.transcript_path = str(path)
        save_manifest(manifest)
        for e in events:
            append_jsonl(run_dir / "events.jsonl", e)
        return run_dir

    def test_turn_without_a_stop_does_not_swallow_later_tool_calls(self):
        # prompt 1 is interrupted (no stop); prompt 2 runs one tool
        events = [
            hook("user_prompt", 1, payload={"prompt": "one"}),
            hook("user_prompt", 5, payload={"prompt": "two"}),
            hook("pre_tool_use", 6, tool_name="Bash", tool_use_id="t1", payload={"tool_input": {"command": "ls"}}),
            hook("post_tool_use", 7, tool_name="Bash", tool_use_id="t1"),
            hook("stop", 8),
        ]
        rec = record_from_run(self.run_with([], events))
        first, second = [t for t in rec.turns if t.role == "assistant"]
        self.assertEqual(first.tool_calls, [])
        self.assertEqual([c.name for c in second.tool_calls], ["Bash"])
        metrics = build_metrics(self.root / "runs" / "20261102-100000-demo")
        self.assertEqual([t.tool_calls for t in metrics.turn_records], [0, 1])

    def test_meta_sidechain_and_interrupt_entries_are_not_human_prompts(self):
        entries = [
            user(1, "real prompt"),
            assistant(2, [{"type": "text", "text": "answer"}]),
            {**user(3, "injected message"), "isMeta": True},
            {**user(4, "subagent talk"), "isSidechain": True},
            user(5, [{"type": "text", "text": "[Request interrupted by user]"}]),
        ]
        rec = record_from_run(self.run_with(entries, []))
        self.assertEqual([t.text for t in rec.turns if t.role == "user"], ["real prompt"])

    def test_meta_entry_does_not_shift_later_responses_when_hooks_exist(self):
        entries = [
            user(1, "first"),
            assistant(2, [{"type": "text", "text": "ans1"}]),
            {**user(3, "meta"), "isMeta": True},
            assistant(4, [{"type": "text", "text": "META-ANS"}]),
            user(5, "second"),
            assistant(6, [{"type": "text", "text": "ans2"}]),
        ]
        events = [
            hook("user_prompt", 1, payload={"prompt": "first"}),
            hook("stop", 4),
            hook("user_prompt", 5, payload={"prompt": "second"}),
            hook("stop", 6),
        ]
        rec = record_from_run(self.run_with(entries, events))
        answers = [t.text for t in rec.turns if t.role == "assistant"]
        self.assertEqual(answers[1], "ans2")
        self.assertIn("ans1", answers[0])

    def test_missing_transcript_does_not_crash_and_the_snapshot_is_used(self):
        run_dir = make_run(self.root)
        before = record_from_run(run_dir)  # takes the snapshot
        (self.root / "session.jsonl").unlink()  # the agent's own copy is gone
        after = record_from_run(run_dir)
        self.assertEqual([t.text for t in after.turns], [t.text for t in before.turns])
        self.assertTrue((run_dir / TRANSCRIPT_SNAPSHOT).is_file())

    def test_missing_transcript_and_no_snapshot_does_not_crash(self):
        run_dir = make_run(self.root)
        (self.root / "session.jsonl").unlink()
        rec = record_from_run(run_dir)  # hooks still give the prompt and tool calls
        self.assertEqual(rec.turns[0].text, "list the files")
        self.assertEqual(rec.provenance["assistant_text"], "unavailable")


class CliTests(unittest.TestCase):
    def test_init_help_lists_experiment_flags(self):
        import contextlib
        import io

        buf = io.StringIO()
        with contextlib.redirect_stdout(buf), self.assertRaises(SystemExit):
            main(["init", "--help"])
        for flag in ("--scenario", "--framing", "--condition", "--run", "--researcher"):
            self.assertIn(flag, buf.getvalue())


class RoundTripTests(unittest.TestCase):
    def test_write_record_then_load(self):
        with tempfile.TemporaryDirectory() as d:
            run_dir = make_run(Path(d))
            written = write_record(run_dir)
            self.assertEqual(load_record(run_dir), written)


if __name__ == "__main__":
    unittest.main()
