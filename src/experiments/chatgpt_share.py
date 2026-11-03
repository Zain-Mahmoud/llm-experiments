"""Adapter: a public ChatGPT share link -> the standard record.

A share page embeds the whole conversation in its HTML (React Router's
"turbo-stream" format), so a plain download is enough; no browser is needed.
The page shows message text, timestamps and the model, and the *names* of tools
the assistant called. It does not show tool inputs, and tool outputs are
redacted, so those are marked ``unavailable`` in the record's provenance.
"""

from __future__ import annotations

import json
import re
import urllib.request
from collections import Counter
from datetime import UTC, datetime
from typing import Any

from .record import Experiment, Record, Source, ToolCallRecord, TurnRecord

SNAPSHOT_NAME = "source.json"
SHARE_URL_RE = re.compile(r"^https://(chatgpt\.com|chat\.openai\.com)/share/[0-9a-fA-F-]+", re.IGNORECASE)
USER_AGENT = "Mozilla/5.0 (Macintosh; Intel Mac OS X 10_15_7) AppleWebKit/537.36 (KHTML, like Gecko) Chrome/126.0 Safari/537.36"


def is_share_url(url: str) -> bool:
    return bool(SHARE_URL_RE.match(url))


def fetch_html(url: str) -> str:
    request = urllib.request.Request(url, headers={"User-Agent": USER_AGENT})
    with urllib.request.urlopen(request, timeout=30) as response:
        return response.read().decode("utf-8", errors="replace")


# --------------------------------------------------------------------------- #
# decoding the embedded data
# --------------------------------------------------------------------------- #
def _resolve(flat: list) -> Any:
    """Expand turbo-stream's flat array, where objects and lists refer to other
    entries by index, back into ordinary nested data. Negative numbers are its
    markers for undefined/null/NaN and become None."""
    memo: dict[int, Any] = {}

    def value(i: Any) -> Any:
        if isinstance(i, bool) or not isinstance(i, int):
            return i
        if i < 0:
            return None
        if i in memo:
            return memo[i]
        v = flat[i]
        if isinstance(v, dict):
            out = memo[i] = {}
            for key, item in v.items():
                name = flat[int(key[1:])] if re.fullmatch(r"_\d+", key) else key
                out[name] = value(item)
            return out
        if isinstance(v, list):
            out = memo[i] = []
            # a leading type tag (date, promise, ...) marks a special value; keep it raw
            out.extend(v if v and v[0] in ("D", "P", "S", "Y", "B", "R", "E", "Z") else [value(x) for x in v])
            return out
        memo[i] = v
        return v

    return value(0)


def _find_conversation(node: Any, seen: set[int] | None = None) -> dict | None:
    seen = seen if seen is not None else set()
    if id(node) in seen:
        return None
    seen.add(id(node))
    if isinstance(node, dict):
        if node.get("mapping") and node.get("current_node"):
            return node
        children = node.values()
    elif isinstance(node, list):
        children = node
    else:
        return None
    for child in children:
        found = _find_conversation(child, seen)
        if found:
            return found
    return None


def extract_conversation(html: str) -> dict:
    chunks = re.findall(r'streamController\.enqueue\(("\[.*?")\);', html, re.DOTALL)
    for chunk in chunks:
        conversation = _find_conversation(_resolve(json.loads(json.loads(chunk))))
        if conversation:
            return conversation
    raise ValueError(
        "no conversation found in that page. Is the link public, and is it a ChatGPT share link? "
        "Otherwise copy the chat text into a file and use `harness import FILE`."
    )


# --------------------------------------------------------------------------- #
# conversation -> record
# --------------------------------------------------------------------------- #
def _text(msg: dict) -> str:
    parts = (msg.get("content") or {}).get("parts") or []
    return "\n".join(p for p in parts if isinstance(p, str)).strip()


def _hidden(msg: dict) -> bool:
    return bool((msg.get("metadata") or {}).get("is_visually_hidden_from_conversation"))


def _iso(ts: float | None) -> str | None:
    return datetime.fromtimestamp(ts, UTC).isoformat() if ts else None


def _branch(conversation: dict) -> list[dict]:
    """The messages on the shared branch, oldest first (current_node back to the root)."""
    mapping, path, node = conversation["mapping"], [], conversation["current_node"]
    while node:
        path.append(node)
        node = (mapping.get(node) or {}).get("parent")
    return [mapping[n]["message"] for n in reversed(path) if mapping.get(n, {}).get("message")]


def conversation_to_record(
    conversation: dict, url: str | None = None, experiment: dict[str, str] | None = None
) -> tuple[Record, str]:
    msgs = _branch(conversation)
    turns: list[TurnRecord] = []
    reply: dict | None = None  # the assistant turn being collected for the latest user prompt

    def flush() -> None:
        if reply is not None:
            turns.append(
                TurnRecord(
                    n=len(turns) + 1,
                    role="assistant",
                    text="\n\n".join(reply["text"]),
                    timestamp=_iso(reply["end"]),
                    tool_calls=reply["calls"],  # [] = the page showed that none were used
                )
            )

    for i, msg in enumerate(msgs):
        role = (msg.get("author") or {}).get("role")
        recipient = msg.get("recipient")
        ts = msg.get("create_time")
        visible_text = (msg.get("content") or {}).get("content_type") == "text" and not _hidden(msg) and _text(msg)

        if role == "user" and recipient in (None, "all") and visible_text:
            flush()
            turns.append(TurnRecord(n=len(turns) + 1, role="user", text=_text(msg), timestamp=_iso(ts)))
            reply = {"text": [], "calls": [], "end": ts}
            continue
        if reply is None:
            continue  # system setup before the first prompt
        if ts:
            reply["end"] = ts
        if role == "assistant" and recipient not in (None, "all", "assistant"):
            call = ToolCallRecord(name=recipient)  # input is not shown; the outcome is redacted
            nxt = msgs[i + 1] if i + 1 < len(msgs) else None
            if nxt and (nxt.get("author") or {}).get("role") == "tool" and nxt.get("create_time") and ts:
                call.duration_ms = round((nxt["create_time"] - ts) * 1000, 1)
            reply["calls"].append(call)
        elif role == "assistant" and visible_text:
            reply["text"].append(_text(msg))
    flush()

    if not turns:
        raise ValueError("the conversation has no user messages")

    models = Counter(
        (m.get("metadata") or {}).get("model_slug") for m in msgs if (m.get("metadata") or {}).get("model_slug")
    )
    model = (models.most_common(1)[0][0] if models else None) or (conversation.get("model") or {}).get("slug")
    has_calls = any(t.tool_calls for t in turns)

    exp = Experiment(**(experiment or {}))
    record = Record(
        run_id="",  # filled in once the run directory exists
        experiment=exp,
        source=Source(kind="share_link", platform="chatgpt", model=model, url=url, snapshot_path=SNAPSHOT_NAME),
        turns=turns,
        provenance={
            "assistant_text": "scraped",
            "tool_calls": "scraped",
            "model": "scraped" if model else "unavailable",
            "tokens": "unavailable",
            **({"tool_inputs": "unavailable", "tool_outputs": "unavailable"} if has_calls else {}),
        },
    )
    return record, f"chatgpt-{exp.run or exp.scenario or 'share'}"


def import_share_link(url: str, experiment: dict[str, str] | None = None) -> tuple[Record, str, str, str]:
    """Fetch and convert a share link. Returns (record, run name, snapshot text, page title)."""
    conversation = extract_conversation(fetch_html(url))
    record, name = conversation_to_record(conversation, url=url, experiment=experiment)
    return record, name, json.dumps(conversation, indent=1, default=str), str(conversation.get("title") or "")
