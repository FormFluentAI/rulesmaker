"""Small project setup helpers exposed as console_scripts entry points.

Provides two lightweight commands:
- install(): prints recommended pip install commands (does not run them).
- doctor(): runs quick health checks (python import checks) and reports missing deps.

These are intentionally non-destructive and safe to run in CI or dev shells.
"""
from __future__ import annotations

import importlib
from typing import List


CORE_PACKAGES = [
    "pydantic",
    "requests",
    "beautifulsoup4",
    "click",
    "fake-useragent",
    "jinja2",
    "aiohttp",
    "numpy",
]

ML_PACKAGES = ["scikit-learn", "sentence-transformers", "nltk"]
LLM_PACKAGES = ["openai", "anthropic"]


def _check_imports(packages: List[str]) -> List[str]:
    missing = []
    for pkg in packages:
        try:
            importlib.import_module(pkg)
        except Exception:
            missing.append(pkg)
    return missing


def install():
    """Print recommended pip install commands for the user to copy/paste.

    This function intentionally only prints instructions so CI or accidental runs
    won't modify environments automatically.
    """
    print("Recommended setup steps for rules-maker:\n")
    print("1) Create and activate a virtual environment (recommended):")
    print("   python -m venv .venv && source .venv/bin/activate\n")

    print("2) Install core dependencies:")
    print("   pip install " + " ".join(CORE_PACKAGES) + "\n")

    print("3) Optional: ML features and LLM integrations:")
    print("   pip install scikit-learn sentence-transformers nltk")
    print("   pip install openai anthropic  # then set API keys in your environment\n")

    print("4) If you prefer editable install for development:")
    print("   pip install -e .[dev]\n")

    print("Note: Some CLI commands in docs require PYTHONPATH=src when running directly from repo.")


def doctor():
    """Run quick health checks and report missing optional packages and runtime issues."""
    print("rules-maker doctor: running quick checks...\n")
    print("Checking core imports...")
    missing_core = _check_imports(CORE_PACKAGES)
    if missing_core:
        print(f"Missing core packages: {missing_core}")
    else:
        print("Core packages OK")

    print("\nChecking ML packages (optional)...")
    missing_ml = _check_imports(ML_PACKAGES)
    if missing_ml:
        print(f"Missing ML packages: {missing_ml}")
    else:
        print("ML packages OK or not required")

    print("\nChecking LLM packages (optional)...")
    missing_llm = _check_imports(LLM_PACKAGES)
    if missing_llm:
        print(f"Missing LLM packages: {missing_llm}")
    else:
        print("LLM packages OK or not required")

    print("\nRuntime sanity checks:")
    try:
        import rules_maker  # type: ignore

        print(f"Imported package: rules_maker (version: {getattr(rules_maker.version, '__version__', 'unknown')})")
    except Exception as exc:  # pragma: no cover - diagnostic path
        print(f"Failed to import rules_maker package: {exc}")


if __name__ == "__main__":
    # simple CLI when executed directly: run doctor
    doctor()
