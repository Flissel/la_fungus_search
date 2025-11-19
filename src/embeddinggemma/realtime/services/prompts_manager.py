"""Prompts management service - handles prompt defaults and custom overrides."""

from __future__ import annotations
import os
import json

# Import mode-specific prompt modules
try:
    from embeddinggemma.modeprompts import deep as _pm_deep  # type: ignore
    from embeddinggemma.modeprompts import structure as _pm_structure  # type: ignore
    from embeddinggemma.modeprompts import exploratory as _pm_exploratory  # type: ignore
    from embeddinggemma.modeprompts import summary as _pm_summary  # type: ignore
    from embeddinggemma.modeprompts import repair as _pm_repair  # type: ignore
    from embeddinggemma.modeprompts import steering as _pm_steering  # type: ignore
    from embeddinggemma.modeprompts import architecture as _pm_architecture  # type: ignore
    from embeddinggemma.modeprompts import bugs as _pm_bugs  # type: ignore
    from embeddinggemma.modeprompts import quality as _pm_quality  # type: ignore
    from embeddinggemma.modeprompts import documentation as _pm_documentation  # type: ignore
    from embeddinggemma.modeprompts import features as _pm_features  # type: ignore
    from embeddinggemma.modeprompts import focused as _pm_focused  # type: ignore
    from embeddinggemma.prompts import _default_instructions as _base_default_instructions  # type: ignore
except ImportError:
    _pm_deep = None
    _pm_structure = None
    _pm_exploratory = None
    _pm_summary = None
    _pm_repair = None
    _pm_steering = None
    _pm_architecture = None
    _pm_bugs = None
    _pm_quality = None
    _pm_documentation = None
    _pm_features = None
    _pm_focused = None
    _base_default_instructions = None  # type: ignore

# Import settings directory constant
from .settings_manager import SETTINGS_DIR

# Available modes - Task Modes + Judge Modes
AVAILABLE_MODES = [
    'architecture', 'bugs', 'quality', 'documentation', 'features',  # New task modes
    'deep', 'structure', 'exploratory', 'summary', 'repair',  # Existing task modes
    'steering', 'focused'  # Judge modes
]


def get_prompt_default_for_mode(mode: str) -> str:
    """
    Get default prompt instructions for a given mode.

    Args:
        mode: The prompt mode ('deep', 'structure', etc.)

    Returns:
        Default instructions string for the mode
    """
    m = (mode or "deep").lower()
    try:
        # New task modes
        if m == 'architecture' and _pm_architecture:
            return _pm_architecture.instructions()  # type: ignore
        if m == 'bugs' and _pm_bugs:
            return _pm_bugs.instructions()  # type: ignore
        if m == 'quality' and _pm_quality:
            return _pm_quality.instructions()  # type: ignore
        if m == 'documentation' and _pm_documentation:
            return _pm_documentation.instructions()  # type: ignore
        if m == 'features' and _pm_features:
            return _pm_features.instructions()  # type: ignore
        # Existing task modes
        if m == 'deep' and _pm_deep:
            return _pm_deep.instructions()  # type: ignore
        if m == 'structure' and _pm_structure:
            return _pm_structure.instructions()  # type: ignore
        if m == 'exploratory' and _pm_exploratory:
            return _pm_exploratory.instructions()  # type: ignore
        if m == 'summary' and _pm_summary:
            return _pm_summary.instructions()  # type: ignore
        if m == 'repair' and _pm_repair:
            return _pm_repair.instructions()  # type: ignore
        # Judge modes
        if m == 'steering' and _pm_steering:
            return _pm_steering.instructions()  # type: ignore
        if m == 'focused' and _pm_focused:
            return _pm_focused.instructions()  # type: ignore
    except Exception:
        pass
    try:
        if callable(_base_default_instructions):
            return _base_default_instructions(m)
    except Exception:
        return ""
    return ""


def get_prompt_overrides() -> dict[str, str]:
    """
    Load prompt overrides from disk.

    Returns:
        Dictionary mapping mode names to custom prompt instructions
    """
    overrides = {}
    try:
        path = os.path.join(SETTINGS_DIR, "prompts_overrides.json")
        if os.path.exists(path):
            with open(path, 'r', encoding='utf-8') as f:
                obj = json.load(f)
                if isinstance(obj, dict):
                    overrides = {str(k): str(v) for k, v in obj.items() if isinstance(v, str)}
    except Exception:
        pass
    return overrides


def get_all_prompt_defaults() -> dict[str, str]:
    """
    Get all default prompts for available modes.

    Returns:
        Dictionary mapping mode names to default instructions
    """
    return {mode: get_prompt_default_for_mode(mode) for mode in AVAILABLE_MODES}


def save_prompt_overrides(overrides: dict) -> None:
    """
    Save prompt overrides to disk.

    Args:
        overrides: Dictionary mapping mode names to custom instructions

    Raises:
        Exception: If save fails
    """
    if not isinstance(overrides, dict):
        raise ValueError("overrides must be a dictionary")

    prompts_dir = SETTINGS_DIR
    os.makedirs(prompts_dir, exist_ok=True)
    path = os.path.join(prompts_dir, "prompts_overrides.json")
    with open(path, 'w', encoding='utf-8') as f:
        json.dump({str(k): str(v) for k, v in overrides.items()}, f, ensure_ascii=False, indent=2)


__all__ = [
    'AVAILABLE_MODES',
    'get_prompt_default_for_mode',
    'get_prompt_overrides',
    'get_all_prompt_defaults',
    'save_prompt_overrides',
]
