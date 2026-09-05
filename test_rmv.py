"""Test suite. Plain asserts, no framework. Run: python test_rmv.py

One test function per unit of work (see PLAN.md §3).
"""

from __future__ import annotations

import re
import subprocess
import sys
from pathlib import Path

ROOT = Path(__file__).parent

# Pinned by SPEC.md §8 / PLAN.md §2.3. numba is what we actually use; scipy is the
# test oracle only; pandas is the reporting boundary only.
EXPECTED_DEPS = {"alpaca-py", "numba", "numpy", "pandas", "scipy"}

# Alpaca key IDs are PK (paper) or AK (live) + exactly 18 uppercase alphanumerics. Matching
# the shape rather than the specific leaked key keeps any secret out of this repo
# permanently and catches the next one too. The secret half is generic base64 and not
# reliably detectable, so the key id is the marker.
ALPACA_KEY_ID = re.compile(r"\b[AP]K[A-Z0-9]{18}\b")

SKIP_DIRS = {".venv", ".git", ".ruff_cache", "__pycache__", "cache"}
SKIP_SUFFIXES = {".npy", ".pdf", ".lock", ".png", ".parquet"}  # binary / large, from Unit 1 on


def _project_deps() -> set[str]:
    """Package names from [project].dependencies. tomllib is 3.11+, we're on 3.10."""
    text = (ROOT / "pyproject.toml").read_text(encoding="utf-8")
    # Scope to the [project] table first: a dependencies key in [dependency-groups] must
    # not be read instead, whichever order the tables appear in.
    table = re.search(r"^\[project\]\s*$(.*?)(?=^\[|\Z)", text, re.S | re.M)
    assert table, "no [project] table in pyproject.toml"
    body = re.sub(r"#.*", "", table.group(1))  # a commented-out dep is not a dep
    block = re.search(r"dependencies\s*=\s*\[(.*?)\]", body, re.S)
    assert block, "no [project].dependencies block in pyproject.toml"
    # Extract quoted specifiers, not lines -- two deps on one line must not hide one.
    # Split on the first specifier/extra/marker/URL character to get the bare name.
    return {
        re.split(r"[><=!~\[;@\s]", spec.strip())[0]
        for spec in re.findall(r"""['"]([^'"]+)['"]""", block.group(1))
    }


def test_unit0_hygiene() -> None:
    """Unit 0 done-when: exact dep set, no vectorbt, no credentials on disk."""
    deps = _project_deps()
    assert deps == EXPECTED_DEPS, f"dependency drift: {deps ^ EXPECTED_DEPS}"

    lock = (ROOT / "uv.lock").read_text(encoding="utf-8")
    for dead in ("vectorbt", "plotly", "pyyaml", "scikit-learn"):
        assert f'name = "{dead}"' not in lock, f"{dead} still in uv.lock -- relock needed"

    # matplotlib is dev-only -- its consumer is the Unit 9 equity-curve plot, which does
    # not exist yet. It must never become a runtime dep.
    assert "matplotlib" not in deps, "matplotlib belongs in the dev group"

    stray = [p for p in ROOT.rglob("*.y*ml") if not SKIP_DIRS & set(p.parts)]
    assert not stray, f"credential yaml is back: {stray}"
    assert (ROOT / ".env.example").exists(), "missing .env.example"
    assert ".env" in (ROOT / ".gitignore").read_text(encoding="utf-8"), ".env is not gitignored"


def test_unit0_no_secrets_tracked() -> None:
    """No leaked key material anywhere in the tree, and none in git history."""
    for path in ROOT.rglob("*"):
        if not path.is_file() or SKIP_DIRS & set(path.parts):
            continue
        if path.suffix.lower() in SKIP_SUFFIXES:
            continue
        try:
            text = path.read_text(encoding="utf-8", errors="ignore")
        except OSError:
            continue
        found = ALPACA_KEY_ID.search(text)
        assert not found, f"Alpaca key id in {path.name}: {found.group()[:6]}..."

    # The yaml was gitignored from the first commit, so history should be clean.
    git = subprocess.run(
        ["git", "log", "--all", "--pretty=format:", "--name-only"],
        cwd=ROOT, capture_output=True, text=True,
    )
    # Without these two the assert below passes vacuously whenever git fails.
    assert git.returncode == 0, f"git history check did not run: {git.stderr.strip()}"
    assert git.stdout.strip(), "git history check returned no files -- did not run"
    assert "alpaca_api.yaml" not in git.stdout, "credential file IS in git history -- scrub it"


def test_unit0_spec_exists() -> None:
    """SPEC.md is the only durable record of the papers -- the PDFs are gitignored."""
    spec = (ROOT / "SPEC.md").read_text(encoding="utf-8")
    for section in ("## 1. The indicator", "## 2. Trading rules", "## 6. Metric definitions",
                    "## 7. Numerics contract", "## 9. Known source discrepancies"):
        assert section in spec, f"SPEC.md missing {section}"
    assert "*.pdf" in (ROOT / ".gitignore").read_text(encoding="utf-8")


def main() -> int:
    tests = [v for k, v in sorted(globals().items()) if k.startswith("test_")]
    failed = 0
    for test in tests:
        try:
            test()
            print(f"  PASS  {test.__name__}")
        except Exception as exc:  # not just AssertionError -- a missing file must
            failed += 1           # report as FAIL, not kill the run before the summary
            label = "" if isinstance(exc, AssertionError) else f"{type(exc).__name__}: "
            print(f"  FAIL  {test.__name__}: {label}{exc}")
    print(f"\n{len(tests) - failed}/{len(tests)} passed")
    return 1 if failed else 0


if __name__ == "__main__":
    sys.exit(main())
