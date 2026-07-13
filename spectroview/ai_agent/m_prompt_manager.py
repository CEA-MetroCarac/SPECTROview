"""m_prompt_manager.py — PromptManager for SPECTROview AI Agent.

This module provides a reusable, cache-aware class that loads Markdown
prompt files from the modular ``AI/`` directory structure and assembles
them into the final LLM system prompt.

Architecture
------------
The ``ai_agent/`` package ships with the following subdirectories:

    prompts/     – Core identity, JSON schema, per-intent instructions
    rules/       – Behavioural constraints
    knowledge/   – Static domain facts
    examples/    – Few-shot conversation examples
    templates/   – Reusable Markdown output formats
    config/      – YAML configuration files

PromptManager loads files from these directories on demand, caches
them in memory, and optionally auto-reloads them when modified on disk
(useful during prompt engineering iterations without restarting the
application).

Usage
-----
::

    from spectroview.ai_agent.m_prompt_manager import PromptManager

    mgr = PromptManager()

    # Load individual sections
    system_text = mgr.load_prompt("system")
    plotting_rules = mgr.load_rule("plotting")
    features_kb = mgr.load_knowledge("features")

    # Build a complete system prompt from selected sections
    prompt = mgr.build_prompt(
        intent="plotting",
        prompts=["system", "chat", "plotting"],
        rules=["general", "plotting"],
        knowledge=["features"],
        examples=["plotting_examples"],
    )
"""
from __future__ import annotations

import logging
from pathlib import Path
from typing import Optional

logger = logging.getLogger(__name__)

# ---------------------------------------------------------------------------
# Inline YAML parser — avoids mandatory PyYAML dependency
# ---------------------------------------------------------------------------

def _parse_simple_yaml(text: str) -> dict:
    """Parse a minimal subset of YAML (key: value, comments with #).

    This avoids a mandatory ``PyYAML`` dependency for the simple
    config files bundled with the package.  Only top-level scalar
    mappings are supported.  For complex YAML, install PyYAML and
    replace this with ``yaml.safe_load(text)``.
    """
    result: dict = {}
    for line in text.splitlines():
        line = line.split("#", 1)[0].strip()       # strip comments
        if ":" not in line:
            continue
        key, _, val = line.partition(":")
        key = key.strip()
        val = val.strip()
        # Type coercion
        if val.lower() == "true":
            result[key] = True
        elif val.lower() == "false":
            result[key] = False
        elif val.lower() in ("null", "~", ""):
            result[key] = None
        else:
            try:
                result[key] = int(val)
            except ValueError:
                try:
                    result[key] = float(val)
                except ValueError:
                    result[key] = val
    return result


# ---------------------------------------------------------------------------
# Intent keyword mapping
# ---------------------------------------------------------------------------

#: Maps intent name → sets of trigger keywords (lowercase).
_INTENT_KEYWORDS: dict[str, frozenset[str]] = {
    "plotting": frozenset({
        "plot", "chart", "graph", "scatter", "box", "bar", "histogram",
        "wafer", "map", "visuali", "draw", "show me a", "create a",
    }),
    "fitting": frozenset({
        "fit", "peak", "fwhm", "lorentzian", "gaussian", "pseudovoigt",
        "fano", "baseline", "r_squared", "r squared", "decay", "tau",
    }),
    "coding": frozenset({
        "code", "script", "python", "pandas", "function", "generate code",
        "write a script", "snippet", "def ", "import ",
    }),
}


def _detect_intent(user_message: str) -> str:
    """Infer the user's intent from the last message text.

    Returns one of: ``"plotting"``, ``"fitting"``, ``"coding"``,
    or ``"chat"`` (default).
    """
    lower = user_message.lower()
    for intent, keywords in _INTENT_KEYWORDS.items():
        if any(kw in lower for kw in keywords):
            return intent
    return "chat"


# ---------------------------------------------------------------------------
# Default component selection per intent
# ---------------------------------------------------------------------------

#: Defines which prompt/rule/knowledge/example files are loaded by default
#: for each recognised intent.  Can be extended without code changes.
_INTENT_DEFAULTS: dict[str, dict[str, list[str]]] = {
    "chat": {
        "prompts": ["system", "chat"],
        "rules": ["general", "spectroview"],
        "knowledge": ["features"],
        "examples": [],
    },
    "plotting": {
        "prompts": ["system", "chat", "plotting"],
        "rules": ["general", "plotting", "spectroview"],
        "knowledge": ["features"],
        "examples": ["plotting_examples"],
    },
    "fitting": {
        "prompts": ["system", "fitting"],
        "rules": ["general", "fitting", "spectroview"],
        "knowledge": ["features"],
        "examples": ["filtering_examples"],
    },
    "coding": {
        "prompts": ["system", "coding"],
        "rules": ["general", "python", "spectroview"],
        "knowledge": [],
        "examples": [],
    },
}




# ---------------------------------------------------------------------------
# PromptManager
# ---------------------------------------------------------------------------

class PromptManager:
    """Load, cache, and assemble Markdown prompt files.

    Parameters
    ----------
    base_dir:
        Absolute path to the ``ai_agent/`` directory.  Defaults to the
        directory containing this module.
    settings_override:
        Optional dictionary that overrides values read from
        ``config/settings.yaml``.  Useful for testing or programmatic
        configuration.
    """

    def __init__(
        self,
        base_dir: Optional[Path | str] = None,
        settings_override: Optional[dict] = None,
    ) -> None:
        if base_dir is None:
            self._base = Path(__file__).parent
        else:
            self._base = Path(base_dir)

        # Cache: relative_path_key → (content, mtime)
        self._cache: dict[str, tuple[str, float]] = {}

        # Load config files
        self._model_cfg: dict = self._load_yaml("config/model.yaml")
        self._settings: dict = self._load_yaml("config/settings.yaml")
        if settings_override:
            self._settings.update(settings_override)

        logger.debug(
            "PromptManager initialised at %s | cache=%s auto_reload=%s",
            self._base,
            self._settings.get("enable_cache", True),
            self._settings.get("auto_reload", True),
        )

    # ------------------------------------------------------------------
    # Configuration accessors
    # ------------------------------------------------------------------

    @property
    def model_config(self) -> dict:
        """Return the parsed ``config/model.yaml`` settings."""
        return dict(self._model_cfg)

    @property
    def settings(self) -> dict:
        """Return the parsed ``config/settings.yaml`` settings."""
        return dict(self._settings)



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

    def load_template(self, name: str) -> str:
        """Load ``templates/<name>.md``."""
        return self._load_md(f"templates/{name}.md")

    # ------------------------------------------------------------------
    # Prompt assembly
    # ------------------------------------------------------------------

    def build_prompt(
        self,
        *,
        intent: str = "chat",
        user_message: str = "",
        prompts: Optional[list[str]] = None,
        rules: Optional[list[str]] = None,
        knowledge: Optional[list[str]] = None,
        examples: Optional[list[str]] = None,
    ) -> str:
        """Assemble a complete system prompt from selected Markdown files.

        When explicit lists are provided they override the intent-based
        defaults.  When a list is ``None``, the intent defaults are used.

        Parameters
        ----------
        intent:
            Logical intent category: ``"chat"``, ``"plotting"``,
            ``"fitting"``, or ``"coding"``.  Controls which files are
            loaded when explicit lists are not supplied.
        user_message:
            The last user message text.  Used for automatic intent
            detection when ``enable_intent_detection`` is True in
            settings.
        prompts:
            Explicit list of prompt file stems to include.
        rules:
            Explicit list of rule file stems to include.
        knowledge:
            Explicit list of knowledge file stems to include.
        examples:
            Explicit list of example file stems to include.

        Returns
        -------
        str
            Concatenated prompt text with section separators.
        """
        # ── Intent detection ─────────────────────────────────────────────
        if self._settings.get("enable_intent_detection", False) and user_message:
            detected = _detect_intent(user_message)
            if detected != "chat":
                intent = detected
                logger.debug("Intent detected: %s", intent)

        defaults = _INTENT_DEFAULTS.get(intent, _INTENT_DEFAULTS["chat"])

        resolved_prompts = prompts if prompts is not None else defaults["prompts"]
        resolved_rules = rules if rules is not None else defaults["rules"]
        resolved_knowledge = (
            knowledge if knowledge is not None else (
                defaults["knowledge"] if self._settings.get("enable_knowledge", True) else []
            )
        )
        resolved_examples = (
            examples if examples is not None else (
                defaults["examples"] if self._settings.get("enable_examples", True) else []
            )
        )

        # ── Assemble ─────────────────────────────────────────────────────
        sections: list[str] = []

        for name in resolved_prompts:
            text = self.load_prompt(name)
            if text:
                sections.append(text)

        if resolved_rules:
            rule_parts: list[str] = []
            for name in resolved_rules:
                text = self.load_rule(name)
                if text:
                    rule_parts.append(text)
            if rule_parts:
                sections.append("\n\n---\n\n".join(rule_parts))

        if resolved_knowledge:
            kb_parts: list[str] = []
            for name in resolved_knowledge:
                text = self.load_knowledge(name)
                if text:
                    kb_parts.append(text)
            if kb_parts:
                sections.append("\n\n---\n\n".join(kb_parts))

        if resolved_examples:
            ex_parts: list[str] = []
            for name in resolved_examples:
                text = self.load_example(name)
                if text:
                    ex_parts.append(text)
            if ex_parts:
                sections.append("\n\n---\n\n".join(ex_parts))

        result = "\n\n---\n\n".join(sections)

        if self._settings.get("debug_prompt", False):
            logger.info(
                "[PromptManager] Assembled prompt (%d chars, intent=%s):\n%s",
                len(result), intent, result,
            )

        return result

    # ------------------------------------------------------------------
    # Cache management
    # ------------------------------------------------------------------

    def clear_cache(self) -> None:
        """Invalidate the entire file content cache."""
        self._cache.clear()
        logger.debug("PromptManager cache cleared.")

    # ------------------------------------------------------------------
    # Internal helpers
    # ------------------------------------------------------------------

    def _load_md(self, relative_path: str) -> str:
        """Load a Markdown file relative to *base_dir*, with caching.

        Parameters
        ----------
        relative_path:
            Path relative to ``self._base`` (e.g. ``"prompts/system.md"``).

        Returns
        -------
        str
            File contents (UTF-8), or empty string if not found / on error.
        """
        abs_path = self._base / relative_path
        cache_enabled = self._settings.get("enable_cache", True)
        auto_reload = self._settings.get("auto_reload", True)

        if cache_enabled and relative_path in self._cache:
            cached_content, cached_mtime = self._cache[relative_path]
            if auto_reload:
                try:
                    current_mtime = abs_path.stat().st_mtime
                    if current_mtime <= cached_mtime:
                        return cached_content  # Cache hit, not modified
                    logger.debug("Auto-reloading modified file: %s", relative_path)
                except FileNotFoundError:
                    pass  # File deleted; fall through to the not-found path
            else:
                return cached_content  # Cache hit, no reload check

        # Cache miss or auto-reload triggered — read from disk
        try:
            content = abs_path.read_text(encoding="utf-8")
            mtime = abs_path.stat().st_mtime
            if cache_enabled:
                self._cache[relative_path] = (content, mtime)
            logger.debug("Loaded: %s (%d chars)", relative_path, len(content))
            return content
        except FileNotFoundError:
            logger.warning(
                "Prompt file not found (skipping): %s", abs_path
            )
            return ""
        except OSError as exc:
            logger.error("Error reading prompt file %s: %s", abs_path, exc)
            return ""

    def _load_yaml(self, relative_path: str) -> dict:
        """Load and parse a YAML config file, returning {} on failure."""
        abs_path = self._base / relative_path
        try:
            text = abs_path.read_text(encoding="utf-8")
            try:
                import yaml  # type: ignore[import]
                return yaml.safe_load(text) or {}
            except ImportError:
                return _parse_simple_yaml(text)
        except FileNotFoundError:
            logger.warning("Config file not found: %s", abs_path)
            return {}
        except Exception as exc:
            logger.error("Error loading config %s: %s", abs_path, exc)
            return {}
