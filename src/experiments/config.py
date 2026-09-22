from __future__ import annotations

from enum import Enum
from pathlib import Path
from typing import Literal

import yaml
from pydantic import BaseModel

Agent = Literal["claude", "codex"]


class Access(Enum):
    low = "low"
    medium = "medium"
    high = "high"

    def claude_permission_mode(self) -> str:
        """Map access level to claude-agent-sdk permission_mode."""
        return {
            Access.low: "default",
            Access.medium: "acceptEdits",
            Access.high: "bypassPermissions",
        }[self]

    def codex_sandbox_mode(self) -> str:
        """Map our access level to openai-codex Sandbox preset."""
        return {
            Access.low: "read_only",
            Access.medium: "workspace_write",
            Access.high: "full_access",
        }[self]


class ExperimentConfig(BaseModel):
    name: str
    task: str
    agent: Agent = "claude"
    access: Access = Access.medium
    repeats: int = 1
    max_turns: int | None = None
    model: str | None = None


def load_config(path: str | Path) -> ExperimentConfig:
    f = open(path, "r")
    data = yaml.safe_load(f)
    return ExperimentConfig.model_validate(data)