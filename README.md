# llm-experiments

Record experiments with AI assistants into one standard file, then annotate them.

1. **Record a local agent session** (Claude Code / Codex). The harness listens while you use the agent.
2. **Read a shareable link** (ChatGPT) or a pasted transcript (any platform).
3. Both give the same `record.json`. **`harness annotate`** opens a page where you comment on turns, mark where the simulation ends, and set an outcome.

## Install

Needs Python 3.14+ and [uv](https://docs.astral.sh/uv/).

```bash
uv sync
```

This gives you a `harness` command (`.venv/bin/harness`). Run it from the folder you want runs saved in: `runs/` is created there.

## 1. Local agent

```bash
harness init --scenario A --run A01 --framing neutral
```

It prints a launch command (or a `launch.sh` path). Start the agent yourself with that, from another terminal, and use it as normal. Hooks installed by `init` report what happens to the harness. When you are done:

```bash
harness finalize latest      # writes record.json (and metrics.json, summary.md)
harness annotate latest
```

`init` writes hook settings into the folder you are in (`.claude/settings.local.json`); `finalize` puts the original back. Use a scratch folder, not your real project.

## 2. Shareable link or pasted text

ChatGPT share link (fetched automatically):

```bash
harness import --url https://chatgpt.com/share/<id> --scenario A --run A01
```

Any other platform: copy the chat into a file with `User:` / `Assistant:` lines (or ChatGPT's `You said:` / `ChatGPT said:`), then:

```bash
harness import chat.txt --platform gemini --scenario A --run A01 --url <the share link, optional>
```

Check the `roles` line the import prints (`U A U A ...`). A line inside a reply that starts with `You:` is read as a new turn, and the import warns when the roles look wrong.

## 3. Annotate

```bash
harness annotate latest
```

Opens `runs/<id>/annotate.html`: the conversation, with the codebook already loaded.

- Click a turn (or select text) to add a comment with an issue type.
- **Simulation ends here** marks the phase change.
- **Outcome** records one verdict for the run: code, rationale, evidence turns, missing evidence, verified checks, setup steps.
- **Export annotations** downloads `annotations.json`. Move it into the run folder; `harness annotate` loads it next time. The page also autosaves in the browser, but the file is what makes it permanent.

## What a run folder holds

| File | From | What it is |
| --- | --- | --- |
| `record.json` | harness | The standard record: experiment labels, source, turns, tool calls, provenance |
| `source.json` / `source.txt` | import | Saved copy of what was imported |
| `transcript.jsonl`, `events.jsonl` | local agent | The agent's transcript and the raw hook log |
| `metrics.json`, `summary.md`, `diff.patch` | local agent | Totals, readable summary, files changed |
| `run.json` | harness | Small manifest |
| `annotate.html` | `harness annotate` | The page, with the data inside |
| `annotations.json` | you (Export) | Your comments, phase marker and outcome |

The record only ever holds facts. Your judgments live in `annotations.json`, so the raw data is never edited.

`provenance` in the record says how each part was obtained (`auto`, `scraped`, `manual`) or that the source cannot show it (`unavailable`), so a missing number is never mistaken for zero. A ChatGPT link gives tool-call names only; tool inputs and outputs are redacted. A web chat has no token counts.

## The codebook

`codebook/core.json` holds the outcome codes and issue types shared by all scenarios; `codebook/scenarios/A.json` adds Scenario A's examples and extra issue type. The page reads them for its dropdowns and rules. To add a scenario, add `codebook/scenarios/B.json`; it can add issue types but not redefine core ones.

## Code

```text
src/experiments/
  cli.py            commands
  harness.py        runs, hook-based metrics, git diff (the original harness)
  agent.py, hook.py install hooks and report events from Claude Code / Codex
  transcript.py     reads Claude Code transcripts
  record.py         the standard record
  capture.py        local-agent run -> record
  chatgpt_share.py  ChatGPT share link -> record
  importer.py       pasted text -> record
  annotate.py       builds the annotation page with data inside
tools/annotator.html   the page (one file, no server, no network)
codebook/              outcome codes and issue types
tests/                 python -m unittest discover -s tests
```

## Limits

- Only ChatGPT links are fetched automatically; other platforms need the text route.
- Messages with images or code-interpreter output are not captured from ChatGPT links; reasoning summaries are skipped.
- The local-agent path has been run live for chat only (no tool calls yet), and only with Claude Code in a terminal.
- Nothing yet collects many runs into one table.
