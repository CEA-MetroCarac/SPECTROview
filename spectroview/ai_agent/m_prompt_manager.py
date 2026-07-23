"""m_prompt_manager.py — PromptManager for the SPECTROview AI Agent.

Loads the Markdown prompt fragments that make up the LLM system prompt and
assembles them into one string, with an in-memory cache that can hot-reload
edited files so prompt engineering needs no application restart.

Layout
------
    prompts/     – Core identity and per-topic instructions
    rules/       – Behavioural constraints
    knowledge/   – Static domain facts
    examples/    – Few-shot conversation examples
    config/      – YAML configuration (model.yaml, settings.yaml)

Usage
-----
::

    mgr = PromptManager()
    prompt = mgr.build_prompt(
        prompts=["system", "chat", "plotting"],
        rules=["general", "plotting", "spectroview"],
        knowledge=["features"],
        examples=["plotting_examples"],
    )

The caller chooses the fragments; see ``VMChat._build_system_prompt`` for the
two tiers actually in use (full and small-model).
"""
from __future__ import annotations

import logging
from pathlib import Path
from typing import Optional, Sequence

import yaml

logger = logging.getLogger(__name__)

#: Separator between assembled sections.
_SECTION_SEP = "\n\n---\n\n"


class PromptManager:
    """Load, cache, and assemble Markdown prompt files.

    Parameters
    ----------
    base_dir:
        Absolute path to the ``ai_agent/`` directory. Defaults to the
        directory containing this module.
    """

    def __init__(self, base_dir: Optional[Path | str] = None) -> None:
        self._base = Path(base_dir) if base_dir is not None else Path(__file__).parent

        # Cache: relative_path_key → (content, mtime)
        self._cache: dict[str, tuple[str, float]] = {}

        self._model_cfg: dict = self._load_yaml("config/model.yaml")
        self._settings: dict = self._load_yaml("config/settings.yaml")

    # ------------------------------------------------------------------
    # Configuration accessors
    # ------------------------------------------------------------------

    @property
    def model_config(self) -> dict:
        """Return the parsed ``config/model.yaml`` settings."""
        return dict(self._model_cfg)

    # ------------------------------------------------------------------
    # Public loading API
    # ------------------------------------------------------------------

    def load_prompt(self, name: str) -> str:
        """Load ``prompts/<name>.md``."""
        return self._load_md(f"prompts/{name}.md")

    def load_rule(self, name: str) -> str:
        """Load ``rules/<name>.md``."""
        return self._load_md(f"rules/{name}.md")

    def load_knowledge(self, name: str) -> str:
        """Load ``knowledge/<name>.md``."""
        return self._load_md(f"knowledge/{name}.md")

    def load_example(self, name: str) -> str:
        """Load ``examples/<name>.md``."""
        return self._load_md(f"examples/{name}.md")

    # ------------------------------------------------------------------
    # Prompt assembly
    # ------------------------------------------------------------------

    def build_prompt(
        self,
        *,
        prompts: Sequence[str] = (),
        rules: Sequence[str] = (),
        knowledge: Sequence[str] = (),
        examples: Sequence[str] = (),
    ) -> str:
        """Assemble a system prompt from the named Markdown fragments.

        Each argument is a list of file stems within the matching directory.
        Missing files are skipped with a warning rather than raising, so a
        packaging slip degrades the prompt instead of breaking the chat.

        Returns
        -------
        str
            The concatenated prompt text, sections separated by ``---``.
        """
        sections: list[str] = [t for name in prompts if (t := self.load_prompt(name))]

        for names, loader in (
            (rules, self.load_rule),
            (knowledge, self.load_knowledge),
            (examples, self.load_example),
        ):
            parts = [t for name in names if (t := loader(name))]
            if parts:
                sections.append(_SECTION_SEP.join(parts))

        result = _SECTION_SEP.join(sections)

        if self._settings.get("debug_prompt", False):
            logger.info("[PromptManager] Assembled prompt (%d chars):\n%s",
                        len(result), result)

        return result

    # ------------------------------------------------------------------
    # Internal helpers
    # ------------------------------------------------------------------

    def _load_md(self, relative_path: str) -> str:
        """Load a Markdown file relative to *base_dir*, with caching.

        Returns the file contents (UTF-8), or an empty string if the file is
        missing or unreadable.
        """
        abs_path = self._base / relative_path
        cache_enabled = self._settings.get("enable_cache", True)
        auto_reload = self._settings.get("auto_reload", True)

        if cache_enabled and relative_path in self._cache:
            cached_content, cached_mtime = self._cache[relative_path]
            if not auto_reload:
                return cached_content
            try:
                if abs_path.stat().st_mtime <= cached_mtime:
                    return cached_content       # cache hit, not modified
                logger.debug("Auto-reloading modified file: %s", relative_path)
            except FileNotFoundError:
                pass                            # deleted; fall through

        try:
            content = abs_path.read_text(encoding="utf-8")
            if cache_enabled:
                self._cache[relative_path] = (content, abs_path.stat().st_mtime)
            return content
        except FileNotFoundError:
            logger.warning("Prompt file not found (skipping): %s", abs_path)
            return ""
        except OSError as exc:
            logger.error("Error reading prompt file %s: %s", abs_path, exc)
            return ""

    def _load_yaml(self, relative_path: str) -> dict:
        """Load and parse a YAML config file, returning {} on failure."""
        abs_path = self._base / relative_path
        try:
            return yaml.safe_load(abs_path.read_text(encoding="utf-8")) or {}
        except FileNotFoundError:
            logger.warning("Config file not found: %s", abs_path)
            return {}
        except Exception as exc:                # noqa: BLE001 - never break the chat
            logger.error("Error loading config %s: %s", abs_path, exc)
            return {}
