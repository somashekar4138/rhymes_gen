"""Interpreter guard, declared metadata, and attribution."""

from __future__ import annotations

from pathlib import Path

import pytest
import tomllib

from rhymes import cli

ROOT = Path(__file__).resolve().parent.parent


def pyproject() -> dict:
    with open(ROOT / "pyproject.toml", "rb") as fh:
        return tomllib.load(fh)


@pytest.mark.parametrize("version", [(3, 13), (3, 14), (3, 9)])
def test_unsupported_interpreter_is_refused_with_the_reason(version: tuple[int, int]) -> None:
    with pytest.raises(RuntimeError) as exc:
        cli._check_interpreter(version)

    message = str(exc.value)
    assert "numpy" in message
    assert "2.0.2" in message
    assert f"{version[0]}.{version[1]}" in message


@pytest.mark.parametrize("version", [(3, 10), (3, 11), (3, 12)])
def test_supported_interpreters_pass(version: tuple[int, int]) -> None:
    assert cli._check_interpreter(version) is None


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


def test_requires_python_upper_bound_is_declared() -> None:
    """The upper bound is load-bearing, not tidiness: numpy 2.0.2 is a heartlib
    hard pin and publishes no cp313 wheel."""
    assert pyproject()["project"]["requires-python"] == ">=3.10,<3.13"


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
