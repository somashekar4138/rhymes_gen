"""Interpreter guard, declared metadata, and attribution."""

from __future__ import annotations

import json
import re
import subprocess
from pathlib import Path

import pytest
import tomllib

from rhymes import cli

ROOT = Path(__file__).resolve().parent.parent


def pyproject() -> dict:
    with open(ROOT / "pyproject.toml", "rb") as fh:
        return tomllib.load(fh)


@pytest.mark.parametrize("version", [(3, 9), (3, 8), (2, 7)])
def test_interpreter_below_the_floor_is_refused_with_the_reason(
    version: tuple[int, int],
) -> None:
    with pytest.raises(RuntimeError) as exc:
        cli._check_interpreter(version)

    message = str(exc.value)
    assert "3.10" in message
    assert f"{version[0]}.{version[1]}" in message


@pytest.mark.parametrize("version", [(3, 13), (3, 14)])
def test_newer_interpreters_are_allowed(version: tuple[int, int]) -> None:
    """There is no upper bound. numpy 2.0.2 ships no cp313 wheel but does ship
    an sdist, so pip compiles it from source -- which is what Colab, now on
    3.13, actually does. An upper bound here blocked the only supported
    runtime, which is the worst way for a constraint to be wrong."""
    cli._check_interpreter(version)  # must not raise


@pytest.mark.parametrize("version", [(3, 10), (3, 11), (3, 12)])
def test_supported_interpreters_pass(version: tuple[int, int]) -> None:
    cli._check_interpreter(version)  # must not raise


def test_interpreter_failure_renders_as_exit_1_not_a_traceback(
    monkeypatch: pytest.MonkeyPatch, capsys: pytest.CaptureFixture[str]
) -> None:
    def refuse(version: tuple[int, int]) -> None:
        raise RuntimeError("needs Python 3.10-3.12 because numpy 2.0.2 has no wheel")

    monkeypatch.setattr(cli, "_check_interpreter", refuse)

    rc = cli.main(["styles"])

    err = capsys.readouterr().err
    assert rc == 1
    assert "Traceback" not in err
    assert len(err.strip().splitlines()) == 1


def test_requires_python_floor_is_declared_with_no_upper_bound() -> None:
    """3.10 is the floor our own syntax needs. There is deliberately NO upper
    bound: heartlib declares >=3.9, and Colab (the supported runtime) is on
    3.13, where numpy 2.0.2 builds from its sdist."""
    assert pyproject()["project"]["requires-python"] == ">=3.10"


def test_console_script_is_declared() -> None:
    assert pyproject()["project"]["scripts"]["rhymes"] == "rhymes.cli:main"


def test_declared_dependencies_are_pinned_exactly() -> None:
    """T-01-SC: a later substitution shows up as a test failure."""
    project = pyproject()["project"]
    assert project["dependencies"] == ["huggingface_hub>=0.26"]

    gpu = project["optional-dependencies"]["gpu"]
    assert len(gpu) == 1
    assert "HeartMuLa/heartlib" in gpu[0]


def test_notice_credits_heartlib() -> None:
    notice = (ROOT / "NOTICE").read_text(encoding="utf-8")

    assert "heartlib" in notice
    assert "Apache" in notice


# --- The Colab notebook -----------------------------------------------------

NOTEBOOK = ROOT / "notebooks" / "rhymes_colab.ipynb"
PLACEHOLDER_RE = re.compile(r"<[a-zA-Z_][a-zA-Z0-9_ -]*>")
PLACEHOLDER_TOKENS = ("YOUR_", "TODO", "CHANGEME", "FIXME")


def notebook() -> dict:
    return json.loads(NOTEBOOK.read_text(encoding="utf-8"))


def code_source() -> str:
    return "\n".join(
        "".join(cell["source"]) for cell in notebook()["cells"] if cell["cell_type"] == "code"
    )


def origin_owner_repo() -> str:
    url = subprocess.run(
        ["git", "remote", "get-url", "origin"],
        cwd=ROOT,
        capture_output=True,
        text=True,
        check=True,
    ).stdout.strip()
    return url.removesuffix(".git").split("github.com/")[-1]


def test_notebook_is_valid_nbformat_4() -> None:
    nb = notebook()

    assert nb["nbformat"] == 4
    assert nb["cells"]
    assert all(cell["cell_type"] in {"code", "markdown"} for cell in nb["cells"])


def test_notebook_carries_no_saved_output() -> None:
    """A committed notebook with stale output is a diff hazard and a
    misleading artifact."""
    for cell in notebook()["cells"]:
        if cell["cell_type"] == "code":
            assert cell.get("outputs") == []
            assert cell.get("execution_count") is None


def test_notebook_drives_install_to_playback() -> None:
    source = code_source()

    assert "HeartMuLa/heartlib" in source
    assert "[Verse]" in source
    assert "rhymes render" in source
    assert "Audio" in source


def test_notebook_installs_this_package_from_its_real_origin() -> None:
    """SPEC R8 claims Run All with no edits. A notebook that asks the reader to
    substitute their own URL fails that, which is the whole of ENV-02."""
    source = code_source()

    assert origin_owner_repo() in source
    assert "pip install" in source


@pytest.mark.parametrize("token", PLACEHOLDER_TOKENS)
def test_notebook_has_no_placeholder_tokens(token: str) -> None:
    assert token not in NOTEBOOK.read_text(encoding="utf-8")


def test_notebook_has_no_angle_bracket_placeholders() -> None:
    assert not PLACEHOLDER_RE.findall(code_source())
