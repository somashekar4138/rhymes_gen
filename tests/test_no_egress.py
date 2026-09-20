"""Prohibition: the user's lyric text never leaves the runtime.

Lyrics about a named child are personal content. The only network egress this
package may perform is a checkpoint download from the model host.

This is an import-level check over the package's own source. It catches
`import requests`; it does not catch `__import__("requests")`,
`importlib.import_module`, or egress through an already-imported transitive
dependency -- see TODOS.md. It is a real guard against the "just a little
usage ping" commit six months from now, which is what it is for.
"""

from __future__ import annotations

import ast
from pathlib import Path

import pytest

from rhymes import engine

SRC = Path(__file__).resolve().parent.parent / "src" / "rhymes"

# The usual suspects. Anything here that is not huggingface_hub is a finding.
NETWORK_CAPABLE = {
    "socket",
    "ssl",
    "http",
    "https",
    "urllib",
    "urllib3",
    "requests",
    "httpx",
    "aiohttp",
    "smtplib",
    "ftplib",
    "telnetlib",
    "xmlrpc",
    "webbrowser",
    "huggingface_hub",
}

SUSPICIOUS_SUBSTRINGS = ("telemetry", "analytics", "sentry", "posthog", "mixpanel")


def imported_top_level_modules() -> set[str]:
    found: set[str] = set()
    for path in sorted(SRC.rglob("*.py")):
        tree = ast.parse(path.read_text(encoding="utf-8"), filename=str(path))
        for node in ast.walk(tree):
            if isinstance(node, ast.Import):
                found.update(alias.name.split(".")[0] for alias in node.names)
            elif isinstance(node, ast.ImportFrom) and node.module and node.level == 0:
                found.add(node.module.split(".")[0])
    return found


def test_the_package_has_source_to_check() -> None:
    """Guard against the check silently passing over an empty directory."""
    assert list(SRC.rglob("*.py"))
    assert imported_top_level_modules()


def test_huggingface_hub_is_the_only_network_capable_import() -> None:
    assert imported_top_level_modules() & NETWORK_CAPABLE == {"huggingface_hub"}


@pytest.mark.parametrize("needle", SUSPICIOUS_SUBSTRINGS)
def test_no_telemetry_shaped_imports(needle: str) -> None:
    assert not [name for name in imported_top_level_modules() if needle in name.lower()]


def test_downloads_carry_only_checkpoint_repo_ids_never_lyrics(
    download_recorder: list[dict], tmp_path: Path
) -> None:
    engine.ensure_checkpoints(tmp_path)

    allowed = {repo_id for repo_id, _ in engine.CHECKPOINTS}
    for call in download_recorder:
        assert call["repo_id"] in allowed
        assert "lyrics" not in str(call).lower()
