"""torch and heartlib must not be imported at module scope.

SPEC R3's fail-fast promise, a test suite that runs on a CUDA-less Mac with no
heartlib installed, and an instant `rhymes styles` all depend on this.

Run in a SUBPROCESS deliberately: asserting on the parent pytest process's
`sys.modules` would pass for the wrong reason on a machine where something else
already imported torch.
"""

from __future__ import annotations

import subprocess
import sys
import textwrap
from pathlib import Path

PROBE = """
import sys
from rhymes import cli

assert "torch" not in sys.modules, "torch imported by `import rhymes.cli`"
assert "heartlib" not in sys.modules, "heartlib imported by `import rhymes.cli`"

cli.main(["styles"])
assert "torch" not in sys.modules, "torch imported by `rhymes styles`"

rc = cli.main(["render", {bad!r}])
assert rc == 1, f"expected exit 1 for a malformed file, got {{rc}}"
assert "torch" not in sys.modules, "torch imported while rejecting a malformed file"
assert "heartlib" not in sys.modules, "heartlib imported while rejecting a malformed file"
print("OK")
"""


def test_neither_torch_nor_heartlib_is_imported(tmp_path: Path) -> None:
    bad = tmp_path / "malformed.txt"
    bad.write_text("no section header anywhere\n", encoding="utf-8")

    result = subprocess.run(
        [sys.executable, "-c", textwrap.dedent(PROBE.format(bad=str(bad)))],
        capture_output=True,
        text=True,
        check=False,
    )

    assert result.returncode == 0, result.stdout + result.stderr
    assert "OK" in result.stdout
