from __future__ import annotations

from pathlib import Path
from typing import Any


def project_root() -> Path:
    return Path(__file__).resolve().parents[2]


def prompt_path(name: str, prompts_dir: str | Path | None = None) -> Path:
    base = Path(prompts_dir) if prompts_dir else project_root() / "configs" / "prompts"
    filename = name if name.endswith(".md") else f"{name}.md"
    return base / filename


def load_prompt(name: str, prompts_dir: str | Path | None = None, default: str = "") -> str:
    path = prompt_path(name, prompts_dir)
    if not path.exists():
        return default
    return path.read_text(encoding="utf-8")


def render_template(template: str, **values: Any) -> str:
    safe_values = {key: "" if value is None else value for key, value in values.items()}
    return template.format(**safe_values)
