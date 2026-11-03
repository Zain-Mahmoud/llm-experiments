"""Launcher for the annotation page: nothing here knows what an annotation is.

It writes a copy of ``tools/annotator.html`` with a run's ``record.json``, the
codebook files and any saved ``annotations.json`` already loaded, so the page
opens ready to use. The page itself stays a single file with no server.
"""

from __future__ import annotations

import json
import webbrowser
from pathlib import Path

REPO_ROOT = Path(__file__).resolve().parents[2]
TEMPLATE = REPO_ROOT / "tools" / "annotator.html"
CODEBOOK_ROOT = REPO_ROOT / "codebook"
OUTPUT_NAME = "annotate.html"


def _read(path: Path) -> dict:
    return json.loads(path.read_text(encoding="utf-8"))


def preload_files(run_dir: Path) -> list[dict]:
    """The files the page should start with, as {name, data}."""
    files = [{"name": "record.json", "data": _read(run_dir / "record.json")}]
    files.append({"name": "core.json", "data": _read(CODEBOOK_ROOT / "core.json")})
    scenario = (files[0]["data"].get("experiment") or {}).get("scenario")
    scenario_file = CODEBOOK_ROOT / "scenarios" / f"{scenario}.json" if scenario else None
    if scenario_file and scenario_file.is_file():
        files.append({"name": scenario_file.name, "data": _read(scenario_file)})
    if (run_dir / "annotations.json").is_file():
        files.append({"name": "annotations.json", "data": _read(run_dir / "annotations.json")})
    return files


def build_page(run_dir: Path) -> Path:
    html = TEMPLATE.read_text(encoding="utf-8")
    # Escape every "<": "</script>" would end the block early, and "<!--" followed by
    # "<script" switches the HTML parser into a mode where the real end tag is ignored.
    data = json.dumps(preload_files(run_dir)).replace("<", "\\u003c")
    marker = "<script>\n\"use strict\";"
    if marker not in html:
        raise ValueError(f"{TEMPLATE} has no script block to preload into")
    page = html.replace(marker, f"<script>window.PRELOAD = {data};</script>\n{marker}", 1)
    out = run_dir / OUTPUT_NAME
    out.write_text(page, encoding="utf-8")
    return out


def open_page(run_dir: Path, *, show: bool = True) -> Path:
    out = build_page(run_dir)
    if show:
        webbrowser.open(out.as_uri())
    return out
