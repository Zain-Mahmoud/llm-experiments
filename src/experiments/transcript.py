"""Parsing for Claude Code session transcripts.

Claude Code writes one JSONL entry per conversation turn. Assistant entries
carry the API ``usage`` block, which is the source of token counts.
"""

from __future__ import annotations

import json
from dataclasses import dataclass, field
from pathlib import Path
from typing import Any

TOKEN_KEYS = (
    "input_tokens",
    "output_tokens",
    "cache_read_input_tokens",
    "cache_creation_input_tokens",
)


@dataclass
class TokenUsage:
    input_tokens: int = 0
    output_tokens: int = 0
    cache_read_input_tokens: int = 0
    cache_creation_input_tokens: int = 0

    @classmethod
    def from_dict(cls, usage: dict[str, Any] | None) -> "TokenUsage":
        usage = usage or {}
        return cls(**{k: int(usage.get(k) or 0) for k in TOKEN_KEYS})

    def __add__(self, other: "TokenUsage") -> "TokenUsage":
        return TokenUsage(**{k: getattr(self, k) + getattr(other, k) for k in TOKEN_KEYS})

    def as_dict(self) -> dict[str, int]:
        return {k: getattr(self, k) for k in TOKEN_KEYS}

    def total(self) -> int:
        return sum(getattr(self, k) for k in TOKEN_KEYS)


@dataclass
class ToolUse:
    name: str
    input: dict[str, Any]


@dataclass
class Entry:
    kind: str
    timestamp: str | None = None
    model: str | None = None
    text: str = ""
    usage: TokenUsage = field(default_factory=TokenUsage)
    tool_uses: list[ToolUse] = field(default_factory=list)
    is_prompt: bool = False


def _blocks(content: Any) -> list[dict[str, Any]]:
    return [b for b in content if isinstance(b, dict)] if isinstance(content, list) else []


def _stringify(content: Any) -> str:
    if isinstance(content, str):
        return content
    return "\n".join(
        str(b.get("text", ""))
        for b in _blocks(content)
        if b.get("type") == "text"
    )


def parse_entry(obj: dict[str, Any]) -> Entry:
    message = obj.get("message") or {}
    content = message.get("content")
    blocks = _blocks(content)
    kind = obj.get("type") or "other"
    has_tool_result = any(b.get("type") == "tool_result" for b in blocks)
    text = _stringify(content)
    # A "user" entry is not always a human prompt: tool results, injected/meta
    # messages, subagent traffic and interrupt markers are all written as user entries.
    human = not (
        has_tool_result
        or obj.get("isMeta")
        or obj.get("isSidechain")
        or text.startswith("[Request interrupted")
    )
    entry = Entry(
        kind=kind,
        timestamp=obj.get("timestamp"),
        model=message.get("model"),
        text=text,
        usage=TokenUsage.from_dict(message.get("usage") or obj.get("usage")),
        is_prompt=kind == "user" and human,
    )
    for block in blocks:
        if block.get("type") == "tool_use":
            entry.tool_uses.append(
                ToolUse(name=str(block.get("name", "")), input=block.get("input") or {})
            )
    return entry


def parse_transcript(path: str | Path) -> list[Entry]:
    lines = Path(path).read_text(encoding="utf-8").splitlines()
    return [parse_entry(json.loads(line)) for line in lines if line.strip()]


def summarize(entries: list[Entry]) -> dict[str, Any]:
    tokens = TokenUsage()
    per_model: dict[str, TokenUsage] = {}
    assistant_messages = 0
    for entry in entries:
        if entry.kind == "assistant":
            assistant_messages += 1
            tokens = tokens + entry.usage
            model = entry.model or "unknown"
            per_model[model] = per_model.get(model, TokenUsage()) + entry.usage
    return {
        "tokens": tokens,
        "tokens_by_model": per_model,
        "assistant_messages": assistant_messages,
    }
