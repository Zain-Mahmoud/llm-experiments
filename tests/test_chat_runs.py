import contextlib
import io
import json
import os
import tempfile
import unittest
from pathlib import Path
from unittest import mock

from experiments import agent as agent_module
from experiments.agent import launch_command
from experiments.capture import record_from_run
from experiments.cli import main
from experiments.harness import RunManifest, append_jsonl, load_manifest, save_manifest
from experiments.record import load_record
from experiments.transcript import find_codex_rollout
from test_capture import CODEX_ROLLOUT, hook
from test_chatgpt_share import NODES, URL, conversation, page

CHAT = "User: hello there\nAssistant: Hi! How can I help?\nUser: what is 2+2\nAssistant: 4\n"


class ChatRunTests(unittest.TestCase):
    def setUp(self):
        self._tmp = tempfile.TemporaryDirectory()
        self.root = Path(self._tmp.name).resolve()
        self.addCleanup(self._tmp.cleanup)
        self._old = Path.cwd()
        os.chdir(self.root)
        self.addCleanup(os.chdir, self._old)

    def init(self, agent):
        with contextlib.redirect_stdout(io.StringIO()) as out:
            self.assertEqual(main(["init", "--agent", agent, "--scenario", "A", "--run", "A01"]), 0)
        return next((self.root / "runs").iterdir()), out.getvalue()

    def finalize(self, run_dir, *flags):
        with contextlib.redirect_stdout(io.StringIO()), contextlib.redirect_stderr(io.StringIO()) as err:
            code = main(["finalize", run_dir.name, *flags])
        return code, err.getvalue()

    def test_init_installs_no_hooks_and_starts_no_watcher(self):
        run_dir, out = self.init("claude-chat")
        manifest = load_manifest(run_dir)
        self.assertEqual((manifest.source_kind, manifest.hook_settings_path), ("chat", None))
        self.assertFalse((self.root / ".claude").exists())
        self.assertFalse((run_dir / "watcher.pid").exists())
        self.assertIn(manifest.launch_command, out)
        self.assertIn("--file chat.txt", out)

    def test_chatgpt_names_the_chatgpt_app(self):
        _, out = self.init("chatgpt")
        self.assertIn("Open the ChatGPT app", out)
        self.assertIn("--url <the chat's share link>", out)

    def test_finalize_from_pasted_text_keeps_the_init_labels(self):
        run_dir, _ = self.init("claude-chat")
        (self.root / "chat.txt").write_text(CHAT, encoding="utf-8")
        self.assertEqual(self.finalize(run_dir, "--file", "chat.txt", "--url", "https://claude.ai/share/x")[0], 0)
        rec = load_record(run_dir)
        self.assertEqual(rec.run_id, run_dir.name)
        self.assertEqual((rec.experiment.scenario, rec.experiment.run), ("A", "A01"))
        self.assertEqual((rec.source.platform, rec.source.url), ("claude.ai", "https://claude.ai/share/x"))
        self.assertEqual([t.role for t in rec.turns], ["user", "assistant", "user", "assistant"])
        self.assertEqual((run_dir / "source.txt").read_text(encoding="utf-8"), CHAT)
        self.assertEqual(load_manifest(run_dir).status, "finished")

    def test_finalize_fetches_a_chatgpt_share_link(self):
        run_dir, _ = self.init("chatgpt")
        with mock.patch("experiments.chatgpt_share.fetch_html", return_value=page(conversation(NODES, "r3"))):
            self.assertEqual(self.finalize(run_dir, "--url", URL)[0], 0)
        rec = load_record(run_dir)
        self.assertEqual((rec.run_id, rec.source.platform, len(rec.turns)), (run_dir.name, "chatgpt", 4))
        self.assertEqual(json.loads((run_dir / "source.json").read_text())["current_node"], "r3")

    def test_a_claude_share_link_alone_asks_for_the_text(self):
        run_dir, _ = self.init("claude-chat")
        code, err = self.finalize(run_dir, "--url", "https://claude.ai/share/x")
        self.assertEqual(code, 1)
        self.assertIn("--file", err)
        self.assertFalse((run_dir / "record.json").exists())

    def test_a_chat_run_with_nothing_given_asks_for_the_text(self):
        run_dir, _ = self.init("chatgpt")
        code, err = self.finalize(run_dir)
        self.assertEqual(code, 1)
        self.assertIn("--file", err)

    def test_url_and_file_are_refused_for_hooked_runs(self):
        with contextlib.redirect_stdout(io.StringIO()):
            main(["init", "--agent", "claude", "--no-watch"])
        run_dir = next((self.root / "runs").iterdir())
        code, err = self.finalize(run_dir, "--file", "chat.txt")
        self.assertEqual(code, 1)
        self.assertIn("for chat runs", err)


class LaunchTests(unittest.TestCase):
    def launch(self, agent, os_name):
        manifest = RunManifest(run_id="r", name="r", agent=agent, access="medium", workspace=".", run_dir=".")
        with mock.patch.object(agent_module.os, "name", os_name):
            return launch_command(manifest, ".")

    def test_macos_opens_each_app(self):
        self.assertEqual(self.launch("codex", "posix"), "open -b com.openai.codex")
        self.assertEqual(self.launch("chatgpt", "posix"), "open -b com.openai.codex")
        self.assertEqual(self.launch("claude-chat", "posix"), "open -a Claude")

    def test_windows_uses_claude_links_and_has_no_chatgpt_command(self):
        self.assertEqual(self.launch("claude", "nt"), 'start "" "claude://claude.ai/new"')
        self.assertIsNone(self.launch("codex", "nt"))
        self.assertIsNone(self.launch("chatgpt", "nt"))

    def test_no_command_means_no_launch_script_and_start_menu_advice(self):
        with tempfile.TemporaryDirectory() as d:
            old = Path.cwd()
            os.chdir(d)
            self.addCleanup(os.chdir, old)
            with mock.patch("experiments.harness.launch_command", return_value=None), \
                    contextlib.redirect_stdout(io.StringIO()) as out:
                main(["init", "--agent", "codex", "--no-watch"])
            manifest = load_manifest(next((Path(d) / "runs").iterdir()))
        self.assertIsNone(manifest.launch_script)
        self.assertIn("Open the ChatGPT app from the Start menu, then open this folder in it", out.getvalue())


class CodexModeTests(unittest.TestCase):
    """The ChatGPT app's Work and Codex modes both run Codex; the rollout says which."""

    def make(self, root, rollout):
        run_dir = root / "runs" / "20261102-100000-demo"
        run_dir.mkdir(parents=True)
        path = root / "rollout.jsonl"
        path.write_text("\n".join(json.dumps(e) for e in rollout), encoding="utf-8")
        save_manifest(RunManifest(run_id=run_dir.name, name="demo", agent="codex", access="medium",
                                  workspace=str(root), run_dir=str(run_dir)))
        append_jsonl(run_dir / "events.jsonl", hook("stop", 14, payload={"transcript_path": str(path)}))
        return run_dir

    def test_platform_is_the_rollout_originator(self):
        rollout = [dict(CODEX_ROLLOUT[0], payload={"id": "s", "originator": "codex_work_desktop"}), *CODEX_ROLLOUT[1:]]
        with tempfile.TemporaryDirectory() as d:
            self.assertEqual(record_from_run(self.make(Path(d), rollout)).source.platform, "codex_work_desktop")

    def test_without_an_originator_the_platform_is_generic(self):
        with tempfile.TemporaryDirectory() as d:
            self.assertEqual(record_from_run(self.make(Path(d), CODEX_ROLLOUT)).source.platform, "codex-desktop")



class NoHooksTests(unittest.TestCase):
    """Codex Desktop on Windows may never run project hooks; the run is then
    found by the folder its session ran in."""

    def setUp(self):
        self._tmp = tempfile.TemporaryDirectory()
        self.addCleanup(self._tmp.cleanup)
        self.root = Path(self._tmp.name).resolve()
        self.workspace = self.root / "ws"
        self.workspace.mkdir()
        self.sessions = self.root / "codex" / "sessions" / "2026" / "11" / "02"
        self.sessions.mkdir(parents=True)

    def rollout(self, name, cwd, mtime, entries=None):
        meta = dict(CODEX_ROLLOUT[0], payload={"id": name, "cwd": str(cwd), "originator": "Codex Desktop"})
        path = self.sessions / f"rollout-{name}.jsonl"
        path.write_text("\n".join(json.dumps(e) for e in [meta, *(CODEX_ROLLOUT[1:] if entries is None else entries)]),
                        encoding="utf-8")
        os.utime(path, (mtime, mtime))
        return path

    def test_picks_the_conversation_not_the_empty_companion_or_other_folders(self):
        real = self.rollout("real", self.workspace, 2000)
        self.rollout("companion", self.workspace, 3000, entries=[])  # newer, but no prompt
        self.rollout("elsewhere", self.root, 4000)
        self.rollout("before-init", self.workspace, 500)
        self.assertEqual(find_codex_rollout(self.workspace, 1000, self.sessions.parents[2]), real)

    def test_nothing_in_the_folder_since_init_finds_nothing(self):
        self.rollout("old", self.workspace, 500)
        self.assertIsNone(find_codex_rollout(self.workspace, 1000, self.sessions.parents[2]))

    def test_a_run_with_no_hook_events_gets_a_full_record(self):
        self.rollout("real", self.workspace, 2_000_000_000)
        run_dir = self.root / "runs" / "r1"
        run_dir.mkdir(parents=True)
        save_manifest(RunManifest(run_id="r1", name="r1", agent="codex", access="medium", workspace=str(self.workspace),
                                  run_dir=str(run_dir), created_at="2026-01-01T00:00:00+00:00"))
        with mock.patch.dict(os.environ, {"CODEX_HOME": str(self.root / "codex")}):
            rec = record_from_run(run_dir)
        self.assertEqual([t.text for t in rec.turns], ["hi", "Hello!", "list the files", "One file."])
        self.assertEqual((rec.source.platform, rec.source.model), ("Codex Desktop", "gpt-test"))


if __name__ == "__main__":
    unittest.main()
