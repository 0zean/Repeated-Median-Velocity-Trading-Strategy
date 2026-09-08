"""Test suite. Plain asserts, no framework. Run: python test_rmv.py

One test function per unit of work (see PLAN.md §3).
"""

from __future__ import annotations

import functools
import math
import re
import shutil
import subprocess
import sys
import tempfile
from datetime import datetime, timezone
from pathlib import Path
from types import SimpleNamespace

import numpy as np
import pandas as pd

import data
import rmv

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
    # Files git ignores cannot leak, and .env is *designed* to hold keys locally. Scan
    # exactly what could be committed -- one git call, not a hardcoded exclusion list.
    ignored = subprocess.run(
        ["git", "ls-files", "--others", "--ignored", "--exclude-standard"],
        cwd=ROOT, capture_output=True, text=True,
    )
    assert ignored.returncode == 0, f"could not list ignored files: {ignored.stderr.strip()}"
    skip = {(ROOT / line).resolve() for line in ignored.stdout.splitlines() if line}

    for path in ROOT.rglob("*"):
        if not path.is_file() or SKIP_DIRS & set(path.parts):
            continue
        if path.suffix.lower() in SKIP_SUFFIXES or path.resolve() in skip:
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




# ============================================================ Unit 1: data layer


def _session_ts(day: str, first: str = "04:00", last: str = "19:55") -> np.ndarray:
    """Epoch-ns for one ET extended-hours session on the 5-minute grid."""
    idx = pd.date_range(f"{day} {first}", f"{day} {last}", freq="5min", tz=data.ET)
    return idx.tz_convert("UTC").to_numpy(dtype="datetime64[ns]").astype(np.int64)


def _synthetic(days=("2024-03-07", "2024-03-08", "2024-03-11"), seed: int = 0):
    """Extended-hours bars over several sessions. The default span crosses the 2024-03-10
    DST switch, so anything time-of-day dependent has to survive it."""
    ts = np.concatenate([_session_ts(d) for d in days])
    rng = np.random.default_rng(seed)
    close = (500 + np.cumsum(rng.normal(0, 0.05, ts.size))).astype(np.float32)
    return ts, close


class _FakeClient:
    """Stands in for StockHistoricalDataClient, serving a fixed series. Lets the cache,
    splice and slicing paths be tested without credentials or network."""

    def __init__(self, ts, close):
        self.ts, self.close = ts, close
        self.calls = []

    def get_stock_bars(self, request):
        self.calls.append((request.start, request.end))
        lo, hi = data._to_ns(request.start), data._to_ns(request.end)
        m = (self.ts >= lo) & (self.ts <= hi)
        index = pd.MultiIndex.from_arrays(
            [np.full(int(m.sum()), "SPY"), pd.to_datetime(self.ts[m], unit="ns", utc=True)],
            names=["symbol", "timestamp"],
        )
        return SimpleNamespace(df=pd.DataFrame({"close": self.close[m]}, index=index))


def _in_temp_cache(fn):
    """Run fn with data.CACHE_DIR pointed at a scratch directory."""
    original = data.CACHE_DIR
    data.CACHE_DIR = Path(tempfile.mkdtemp(prefix="rmv-cache-"))
    try:
        return fn()
    finally:
        shutil.rmtree(data.CACHE_DIR, ignore_errors=True)
        data.CACHE_DIR = original


def test_unit1_gate_window() -> None:
    """SPEC 2: gate is open on [10:00, 15:55) ET, closed either side."""
    ts = _session_ts("2024-06-03")
    gate = data.build_gate(ts)
    at = {t.strftime("%H:%M"): g for t, g in zip(data.to_et(ts), gate)}

    assert at["09:55"] == 0, "gate open before 10:00"
    assert at["10:00"] == 1, "gate shut at 10:00 -- the threshold is inclusive"
    assert at["15:50"] == 1, "gate shut on the last tradeable bar"
    assert at["15:55"] == 0, "gate open at 15:55 -- must be flat"
    assert at["04:00"] == 0 and at["19:55"] == 0, "gate open outside the trading window"

    # 10:00 through 15:50 inclusive = 71 five-minute bars.
    assert gate.sum() == 71, f"expected 71 gated bars, got {gate.sum()}"
    assert gate.dtype == np.int8


def test_unit1_gate_is_dst_aware() -> None:
    """The gate must follow ET wall clock, not a fixed UTC offset: 10:00 ET is 15:00 UTC in
    winter and 14:00 UTC in summer. A naive offset shifts every trade by an hour for ~8
    months of the year."""
    winter, summer = _session_ts("2024-01-16"), _session_ts("2024-07-16")

    for label, ts in (("winter", winter), ("summer", summer)):
        gate = data.build_gate(ts)
        et = data.to_et(ts)
        opens = et[int(np.flatnonzero(gate)[0])]
        closes = et[int(np.flatnonzero(gate)[-1])]
        assert opens.strftime("%H:%M") == "10:00", f"{label} opens at {opens}"
        assert closes.strftime("%H:%M") == "15:50", f"{label} closes at {closes}"

    # ...and the underlying UTC hours really do differ, so the assertions above have teeth.
    def first_gated_utc_hour(ts):
        gate = data.build_gate(ts)
        return pd.to_datetime(ts[int(np.flatnonzero(gate)[0])], unit="ns", utc=True).hour

    assert first_gated_utc_hour(winter) == 15, "winter 10:00 ET should be 15:00 UTC"
    assert first_gated_utc_hour(summer) == 14, "summer 10:00 ET should be 14:00 UTC"


def test_unit1_gate_blackout_after_gap() -> None:
    """SPEC 3.1: an RMedV window spanning a data gap measures the gap. The gate stays shut
    for max_n bars after any break -- a halt, or a missing pre-market bucket."""
    ts = _session_ts("2024-06-03")
    et = data.to_et(ts)
    halt_start = pd.Timestamp("2024-06-03 11:00", tz=data.ET)
    halt_end = pd.Timestamp("2024-06-03 11:30", tz=data.ET)
    holed = ts[~((et >= halt_start) & (et < halt_end))]

    gate = data.build_gate(holed, max_n=24)
    resume = int(np.flatnonzero(data.to_et(holed) >= halt_end)[0])

    assert gate[resume - 1] == 1, "gate should be open on the bar before the halt"
    assert gate[resume : resume + 24].sum() == 0, "gate open inside the post-gap blackout"
    assert gate[resume + 24] == 1, "blackout ran longer than max_n bars"

    # A clean session has only the overnight break and no intraday blackout.
    assert data.build_gate(ts).sum() == 71


def test_unit1_validate_rejects_bad_series() -> None:
    """validate() is the contract every downstream unit relies on. Each mutation must trip."""
    ts, close = _synthetic(days=("2024-06-03",))
    data.validate(ts, close)  # the clean case must pass

    def rejects(bad_ts, bad_close, why):
        # ValueError, not AssertionError: python -O strips asserts and this is the data
        # contract for a system that places orders.
        try:
            data.validate(bad_ts, bad_close)
        except ValueError:
            return
        raise AssertionError(f"validate accepted {why}")

    dup = ts.copy()
    dup[5] = dup[4]
    rejects(dup, close, "duplicate timestamps")

    swapped = ts.copy()
    swapped[[3, 4]] = swapped[[4, 3]]
    rejects(swapped, close, "unsorted timestamps")

    misaligned = ts.copy()
    misaligned[7] += 60 * 1_000_000_000
    rejects(misaligned, close, "a bar off the 5-minute grid")

    nan_close = close.copy()
    nan_close[2] = np.nan
    rejects(ts, nan_close, "NaN close")

    neg_close = close.copy()
    neg_close[2] = -1.0
    rejects(ts, neg_close, "negative close")

    rejects(ts, close.astype(np.float64), "float64 close")
    rejects(ts[:-1], close, "length mismatch")


def test_unit1_cache_roundtrip_is_exact() -> None:
    """Unit 1 done-when: a cached round-trip is bit-identical to a fresh fetch."""
    ts, close = _synthetic()
    fake = _FakeClient(ts, close)
    start, end = data._as_utc(ts[0]), data._as_utc(ts[-1])

    def run():
        fresh = data.load_bars("SPY", start, end, client=fake)
        assert fake.calls, "fetch was never called"
        cached = data.load_bars("SPY", start, end, refresh=False)
        assert np.array_equal(fresh.ts, cached.ts), "ts changed across the cache"
        assert np.array_equal(fresh.close, cached.close), "close changed across the cache"
        assert np.array_equal(fresh.gate, cached.gate), "gate changed across the cache"
        assert fresh.ts.dtype == np.int64 and fresh.close.dtype == np.float32
        assert len(fresh) == int(data.session_mask(ts).sum()), "session filter not applied"
        assert np.array_equal(fresh.ts, ts[data.session_mask(ts)])

    _in_temp_cache(run)


def test_unit1_refresh_appends_without_duplicating() -> None:
    """Append-only refresh: a later load extends the cache and never re-adds a bar."""
    ts, close = _synthetic()
    cut = int(np.flatnonzero(ts >= _session_ts("2024-03-08")[0])[0])

    def run():
        first = data.load_bars(
            "SPY", data._as_utc(ts[0]), data._as_utc(ts[cut - 1]),
            client=_FakeClient(ts[:cut], close[:cut]),
        )
        want_first = int(data.session_mask(ts[:cut]).sum())
        assert len(first) == want_first, f"expected {want_first} bars, got {len(first)}"

        full = data.load_bars(
            "SPY", data._as_utc(ts[0]), data._as_utc(ts[-1]),
            client=_FakeClient(ts, close),
        )
        keep = data.session_mask(ts)
        assert len(full) == int(keep.sum()), f"expected {keep.sum()} bars, got {len(full)}"
        assert np.array_equal(full.ts, ts[keep]), "refresh reordered or duplicated bars"
        assert np.array_equal(full.close, close[keep]), "refresh altered prices"
        data.validate(full.ts, full.close)

    _in_temp_cache(run)


def test_unit1_refresh_detects_rewritten_history() -> None:
    """The guard that makes adjustment=split meaningful: if the server ever returns a
    different price for an already-cached bar, refuse rather than blend two price series."""
    ts, close = _synthetic()
    cut = ts.size - 50

    def run():
        data.load_bars("SPY", data._as_utc(ts[0]), data._as_utc(ts[cut - 1]),
                       client=_FakeClient(ts[:cut], close[:cut]))
        rewritten = close.copy()
        rewritten[:cut] *= 0.997  # what a dividend adjustment does to history
        try:
            data.load_bars("SPY", data._as_utc(ts[0]), data._as_utc(ts[-1]),
                           client=_FakeClient(ts, rewritten))
        except RuntimeError as exc:
            assert "changed underneath us" in str(exc)
            return
        raise AssertionError("silently accepted rewritten history")

    _in_temp_cache(run)


def test_unit1_slicing_bounds() -> None:
    """[start, end] slicing is inclusive at both ends and never returns a partial bar."""
    ts, close = _synthetic()

    def run():
        data.load_bars("SPY", data._as_utc(ts[0]), data._as_utc(ts[-1]),
                       client=_FakeClient(ts, close), session=False)
        lo, hi = 100, 300
        got = data.load_bars("SPY", data._as_utc(ts[lo]), data._as_utc(ts[hi]),
                             refresh=False, session=False)
        assert got.ts[0] == ts[lo] and got.ts[-1] == ts[hi]
        assert len(got) == hi - lo + 1, f"expected {hi - lo + 1} bars, got {len(got)}"

    _in_temp_cache(run)


def test_unit1_session_profile_finds_holes() -> None:
    """session_profile is how SPEC 3.1's 'SPY prints in every bucket' assumption gets
    checked against reality instead of assumed."""
    days = ("2024-06-03", "2024-06-04", "2024-06-05", "2024-06-06")
    ts = np.concatenate([_session_ts(d) for d in days])
    et = data.to_et(ts)
    # Drop the 04:15 bucket on 3 of the 4 sessions, as thin pre-market really does.
    drop = (et.strftime("%H:%M") == "04:15") & (et.strftime("%Y-%m-%d") != "2024-06-06")
    profile = data.session_profile(ts[np.asarray(~drop)])

    assert abs(profile[(4 * 60 + 15) // 5] - 0.25) < 1e-9, "04:15 completeness wrong"
    assert abs(profile[(10 * 60) // 5] - 1.0) < 1e-9, "10:00 should be present every session"
    assert profile[(2 * 60) // 5] == 0.0, "02:00 is outside the session and must be empty"



def test_unit1_session_window() -> None:
    """SPEC 3.1: the series is 08:00-15:55 ET, the range measured to be 100% contiguous.

    Outside it, 2024 bucket completeness falls to 85-98%, and a missing bucket changes the
    x-spacing RMedV's slope is measured against.
    """
    ts = _session_ts("2024-06-03")  # a full 04:00-19:55 extended-hours session
    keep = data.session_mask(ts)
    et = data.to_et(ts[keep])

    assert et[0].strftime("%H:%M") == "08:00", f"session starts {et[0]}"
    assert et[-1].strftime("%H:%M") == "15:55", f"session ends {et[-1]}"
    assert keep.sum() == 96, f"expected 96 bars in the window, got {keep.sum()}"
    # Uniform grid is the precondition RMedV's slope depends on.
    assert np.all(np.diff(ts[keep]) == data.BAR_NS), "session window is not evenly spaced"


def test_unit1_warmup_covers_max_n_before_gate() -> None:
    """The reason 08:00 is the start: it is exactly MAX_N bars before the 10:00 gate, so
    the first tradeable bar's lookback ends on the session's own first bar and never
    reaches across the overnight gap.

    Both gate rules -- the 10:00 trading window and the post-gap blackout -- must agree
    on this. If they ever disagree, one of them is wrong.
    """
    ts = np.concatenate([_session_ts(d) for d in ("2024-06-03", "2024-06-04")])
    ts = ts[data.session_mask(ts)]
    gate = data.build_gate(ts, max_n=data.MAX_N)
    et = data.to_et(ts)

    day2 = np.flatnonzero(et.normalize() == pd.Timestamp("2024-06-04", tz=data.ET))
    first_open = day2[int(np.flatnonzero(gate[day2])[0])]
    assert et[first_open].strftime("%H:%M") == "10:00", f"day 2 opens at {et[first_open]}"

    # The lookback of the first tradeable bar must lie wholly inside its own session.
    window = ts[first_open - data.MAX_N + 1 : first_open + 1]
    assert window.size == data.MAX_N
    assert np.all(np.diff(window) == data.BAR_NS), "first tradeable window spans a gap"
    assert data.to_et(window[:1])[0].strftime("%H:%M") == "08:05"

    # And the blackout alone -- with the trading window disabled -- lands on the same bar.
    open_min, close_min = data.GATE_OPEN_MIN, data.GATE_CLOSE_MIN
    try:
        data.GATE_OPEN_MIN, data.GATE_CLOSE_MIN = 0, 24 * 60
        blackout_only = data.build_gate(ts, max_n=data.MAX_N)
    finally:
        data.GATE_OPEN_MIN, data.GATE_CLOSE_MIN = open_min, close_min
    first_after_blackout = day2[int(np.flatnonzero(blackout_only[day2])[0])]
    assert first_after_blackout == first_open, (
        "the 10:00 trading window and the post-gap blackout disagree on the first "
        "tradeable bar -- 08:00 is no longer MAX_N bars before the gate"
    )



def test_unit1_gate_respects_early_close() -> None:
    """NYSE closes at 13:00 on ~2 sessions a year and SPY keeps printing afterwards, so a
    fixed 15:55 exit leaves the gate open across ~3 hours of thin post-close prints.
    Measured before the fix: 589 gated bars after 13:00 across 21 early closes.
    """
    ts = _session_ts("2024-07-03")  # a real 13:00 early close
    ts = ts[data.session_mask(ts)]
    minute = data._minute_of_day(data.to_et(ts))

    early = data.build_gate(ts, close_min=np.full(ts.size, 13 * 60, np.int32))
    assert early[minute >= 13 * 60].sum() == 0, "gate open after a 13:00 close"
    assert early[minute == 12 * 60 + 55].sum() == 0, "gate open on the 12:55 exit bar"
    assert early[minute == 12 * 60 + 50].sum() == 1, "gate shut before the 12:55 exit"
    # 10:00..12:50 inclusive = 35 bars.
    assert early.sum() == 35, f"expected 35 gated bars on an early close, got {early.sum()}"

    # A regular 16:00 close is unaffected, and matches the no-calendar default.
    regular = data.build_gate(ts, close_min=np.full(ts.size, 16 * 60, np.int32))
    assert regular.sum() == 71
    assert np.array_equal(regular, data.build_gate(ts)), "16:00 calendar != no-calendar default"

    # A date the calendar does not list (close_min 0) is not tradeable at all.
    assert data.build_gate(ts, close_min=np.zeros(ts.size, np.int32)).sum() == 0


def test_unit1_no_fetch_when_cache_already_covers_request() -> None:
    """Every walk-forward slice Unit 7 requests ends long before the cache does. The old
    refresh path asked Alpaca for end < start on each one and got an opaque HTTP 400."""
    ts, close = _synthetic()

    def run():
        loader = _FakeClient(ts, close)
        data.load_bars("SPY", data._as_utc(ts[0]), data._as_utc(ts[-1]), client=loader)
        before = len(loader.calls)

        spy = _FakeClient(ts, close)
        got = data.load_bars("SPY", data._as_utc(ts[0]), data._as_utc(ts[len(ts) // 2]),
                             client=spy)
        assert not spy.calls, "hit the network for a range the cache already covers"
        assert len(got) > 0 and before > 0

    _in_temp_cache(run)


def test_unit1_splice_merges_rather_than_appends() -> None:
    """_splice must union, not append past the tail: bars before the cache's start and
    bars filling an interior hole were both being dropped silently."""
    ts, close = _synthetic(days=("2024-06-03",))
    ts, close = ts[data.session_mask(ts)], close[data.session_mask(ts)]

    # Cache holds the middle; the fetch offers earlier bars and an interior hole-filler.
    holed = np.r_[np.arange(20, 40), np.arange(41, 60)]
    merged_ts, merged_close = data._splice(ts[holed], close[holed], ts, close)

    assert np.array_equal(merged_ts, ts), "splice did not recover the full series"
    assert np.array_equal(merged_close, close), "splice altered prices while merging"
    assert merged_ts[0] == ts[0], "splice dropped bars before the cache start"


def test_unit1_cache_is_never_narrowed_or_left_invalid() -> None:
    """A narrow request must not shrink the cache, and nothing failing the contract may
    ever be written to it."""
    ts, close = _synthetic()

    def run():
        data.load_bars("SPY", data._as_utc(ts[0]), data._as_utc(ts[-1]),
                       client=_FakeClient(ts, close))
        full = data._read_cache("SPY", "sip")[0].size

        data.load_bars("SPY", data._as_utc(ts[100]), data._as_utc(ts[200]), refresh=False)
        assert data._read_cache("SPY", "sip")[0].size == full, "a narrow read shrank the cache"

        # And the writer refuses a series that would violate the contract.
        bad_ts = ts.copy()
        bad_ts[5] = bad_ts[4]
        try:
            data._write_cache("SPY", "sip", bad_ts, close)
        except ValueError:
            assert data._read_cache("SPY", "sip")[0].size == full, "cache damaged by a bad write"
            return
        raise AssertionError("_write_cache persisted a series that fails validate()")

    _in_temp_cache(run)



def test_unit1_real_cache_matches_spec_claims() -> None:
    """Guards the numbers SPEC §3.1/§3.2 assert, against the real cached series.

    Every other test here runs on synthetic bars. These figures are what the spec is
    normative about, and they had no coverage until they had already drifted once.
    Skips (rather than fails) when the cache is absent, so the suite stays hermetic.
    """
    npz, _ = data._cache_paths("SPY", "sip")
    if not npz.exists():
        print("       (skipped: no cache -- run data.load_bars to populate)")
        return

    from datetime import datetime, timezone

    bars = data.load_bars(
        "SPY",
        datetime(2016, 1, 1, tzinfo=timezone.utc),
        datetime(2026, 9, 1, tzinfo=timezone.utc),
        refresh=False,
    )
    et = data.to_et(bars.ts)
    day = et.normalize()
    per_session = pd.Series(1, index=day).groupby(day).sum()

    # SPEC §3.1: near-contiguous, 95.98 bars/session, nothing over 96, ~28 holed sessions.
    assert per_session.max() == 96, "a session has more than 96 bars -- duplicates or DST"
    assert (per_session < 96).sum() < 40, f"{(per_session < 96).sum()} holed sessions"
    assert 95.9 < len(bars) / per_session.size <= 96.0, "bars/session outside the spec range"
    assert et[0].strftime("%H:%M") >= "08:00" and et[-1].strftime("%H:%M") <= "15:55"

    # SPEC §3.2: no gated bar may sit past its own session close. Detect early closes from
    # the unfiltered feed (no prints after 17:00 ET) rather than trusting the calendar
    # twice -- this must hold even if the calendar were wrong.
    raw = data.load_bars(
        "SPY",
        datetime(2016, 1, 1, tzinfo=timezone.utc),
        datetime(2026, 9, 1, tzinfo=timezone.utc),
        refresh=False, session=False, calendar=False,
    )
    r_day = data.to_et(raw.ts).normalize()
    after_17 = pd.Series((data.to_et(raw.ts).hour >= 17).astype(int), index=r_day)
    early = after_17.groupby(r_day).sum()
    early_days = early[early == 0].index
    assert len(early_days) > 10, f"only {len(early_days)} early closes found -- detection broke"

    late = int(((day.isin(early_days)) & (et.hour >= 13) & (bars.gate == 1)).sum())
    assert late == 0, f"{late} gated bars after a 13:00 early close (was 589 before the fix)"

    # A regular session gates 10:00..15:50 = 71 bars; an early close gates 10:00..12:50 = 35.
    gated = pd.Series(bars.gate, index=day).groupby(day).sum()
    assert gated.max() == 71, f"max gated bars/session is {gated.max()}, expected 71"
    gated_early = gated[gated.index.isin(early_days)]
    assert set(gated_early.unique()) <= {35}, (
        f"early-close sessions gate {sorted(set(gated_early.unique()))} bars, expected only 35"
    )



def test_unit1_dotenv_precedence_and_scope() -> None:
    """Credentials are a trust boundary: the real environment must win over the file, and
    a stray line in .env must not be able to alter the process environment."""
    import os

    tmp = Path(tempfile.mkdtemp(prefix="rmv-env-")) / ".env"
    tmp.write_text(
        "# comment\nAPI_KEY=from_file\nSECRET_KEY='quoted_secret'\nPATH=/evil\nJUNK\n",
        encoding="utf-8",
    )
    saved = {k: os.environ.get(k) for k in ("API_KEY", "SECRET_KEY", "PATH")}
    try:
        os.environ["API_KEY"] = "from_environment"
        os.environ.pop("SECRET_KEY", None)
        real_path = os.environ.get("PATH")

        data.load_dotenv(tmp)

        assert os.environ["API_KEY"] == "from_environment", "file overrode the real environment"
        assert os.environ["SECRET_KEY"] == "quoted_secret", "quotes not stripped / not loaded"
        assert os.environ.get("PATH") == real_path, ".env was able to set PATH"
    finally:
        for k, v in saved.items():
            if v is None:
                os.environ.pop(k, None)
            else:
                os.environ[k] = v
        shutil.rmtree(tmp.parent, ignore_errors=True)



# ========================================================= Unit 2: the RMedV kernel


def _lower_median_reference(y: np.ndarray) -> float:
    """What the repeated median would give if even-length medians picked the lower middle
    value instead of averaging the two. Only used to prove the kernel does not do that."""
    def lower_median(values):
        s = sorted(values)
        return s[len(s) // 2 - 1] if len(s) % 2 == 0 else s[len(s) // 2]

    n = y.size
    inner = [lower_median([(y[j] - y[i]) / (j - i) for j in range(n) if j != i])
             for i in range(n)]
    return lower_median(inner)

def test_unit2_matches_paper_worked_examples() -> None:
    """Both papers publish a worked example whose repeated median slope is exactly 1.0,
    with outliers planted to show the estimator ignoring them (SPEC §1.1)."""
    for label, y in (
        ("M05 p.2", [1, 2, 3, 4, 5, 15, 12, 8, 9, 10]),
        ("M25 p.2", [1, 2, 10, 4, 5, 6, 7, 8, 9, 18, 11, 12, 13, 18, 15, 20]),
    ):
        series = np.asarray(y, dtype=np.float32)
        got = float(rmv.rmv_all_n(series, np.array([len(y)]))[0, -1])
        assert got == 1.0, f"{label} gave {got}, paper says exactly 1.0"

    # A clean ramp of slope 3 must come back as 3 for every n -- catches a scale error
    # that both 1.0 examples would miss, since 1.0 is a fixed point of many mistakes.
    ramp = (np.arange(40) * 3.0 + 17.0).astype(np.float32)
    out = rmv.rmv_all_n(ramp, rmv.N_VALUES)
    for a, n in enumerate(rmv.N_VALUES):
        assert np.allclose(out[a, n - 1 :], 3.0, atol=1e-5), f"n={n} lost the slope"

    # And a falling ramp must be negative -- catches an inverted (j-i) sign, which 1.0
    # and a positive ramp both survive.
    down = (100.0 - np.arange(40) * 3.0).astype(np.float32)
    out = rmv.rmv_all_n(down, rmv.N_VALUES)
    for a, n in enumerate(rmv.N_VALUES):
        assert np.allclose(out[a, n - 1 :], -3.0, atol=1e-5), f"n={n} has the sign inverted"


def test_unit2_matches_scipy_oracle() -> None:
    """SPEC §1.1: scipy.stats.siegelslopes(method='hierarchical') is the reference.

    Random-walk data at SPY's actual price and volatility, so float32 storage is exercised
    where it is weakest -- differencing ~$600 values to resolve ~$0.17 moves.
    """
    from scipy.stats import siegelslopes

    rng = np.random.default_rng(7)
    close = (600 + np.cumsum(rng.normal(0, 0.17, 3000))).astype(np.float32)
    out = rmv.rmv_all_n(close, rmv.N_VALUES)

    # Bit-exactness, not a tolerance. Output is float32 and |rmv| ~ 0.05, where float32
    # eps is 3.7e-9 -- so PLAN's original "matches to 1e-9" was below the storage
    # resolution and unachievable by construction. Requiring the kernel's float32 output
    # to equal float32(scipy's float64 answer) exactly is strictly stronger than any
    # tolerance, and it is what actually holds.
    checked = 0
    for n in (3, 4, 5, 10, 23, 24):  # 3 and 4 are where the estimator is most fragile
        a = int(np.flatnonzero(rmv.N_VALUES == n)[0])
        for t in (n - 1, 137, 1500, 2999):  # includes the very first computable bar
            window = close[t - n + 1 : t + 1].astype(np.float64)
            ref = siegelslopes(window, np.arange(n, dtype=float))[0]
            assert out[a, t] == np.float32(ref), (
                f"n={n} t={t}: kernel {out[a, t]!r} != float32(scipy) {np.float32(ref)!r}"
            )
            assert abs(float(out[a, t]) - ref) <= 8e-9, "beyond float32 resolution"
            checked += 1
    assert checked == 24, f"only {checked} oracle comparisons ran"


def test_unit2_median_tie_breaking_matches_numpy() -> None:
    """Even-length medians must average the two middle values, as numpy and scipy do.

    n is even for half the grid, and the inner medians run over n-1 points, so both
    parities occur at every n. A 'lower median' shortcut passes odd n and fails here.
    """
    from scipy.stats import siegelslopes

    # y chosen so the two middle values differ: averaging gives 2.25, picking the lower
    # middle gives 2.00. Most 4-point series give the same answer either way, so this
    # exact series is doing the work -- the guard below fails if it stops discriminating.
    y = np.array([0.0, 1.0, 5.0, 6.0], dtype=np.float32)  # n=4 -> even outer median
    got = float(rmv.rmv_all_n(y, np.array([4]))[0, -1])
    ref = float(siegelslopes(y.astype(np.float64), np.arange(4, dtype=float))[0])

    lower_middle = _lower_median_reference(y.astype(np.float64))
    assert abs(ref - lower_middle) > 1e-9, "test series no longer distinguishes the two rules"
    assert got == np.float32(ref), f"n=4 gave {got}, numpy/scipy averaging gives {ref}"
    assert abs(got - lower_middle) > 1e-9, "kernel is picking a middle value, not averaging"


def test_unit2_warmup_is_zero_never_nan() -> None:
    """SPEC §7 rules 1-2: warmup is exactly the first n-1 entries and is 0.0, not NaN.

    NaN would be the obvious sentinel and is banned, because `fastmath` compiles np.isnan
    away and every guard written against it becomes dead code.
    """
    close = (600 + np.cumsum(np.random.default_rng(1).normal(0, 0.17, 400))).astype(np.float32)
    out = rmv.rmv_all_n(close, rmv.N_VALUES)

    assert not np.isnan(out).any(), "NaN in the output -- SPEC §7 forbids NaN sentinels"
    for a, n in enumerate(rmv.N_VALUES):
        assert np.all(out[a, : n - 1] == 0.0), f"n={n}: warmup is not all zero"
        # Not "!= 0.0": a genuinely flat window computes exactly 0.0, which is the
        # accepted cost of SPEC §7 rule 2. Assert the row is populated instead.
        assert np.any(out[a, n - 1 :] != 0.0), f"n={n}: nothing computed past warmup"


def test_unit2_reproducible_and_rows_independent() -> None:
    """Checks the two properties that MAKE thread-count independence true, since a single
    process cannot change NUMBA_NUM_THREADS after import: recomputation is bit-exact, and
    each output row depends only on its own n. Cross-thread equality itself is verified
    out of process by test_unit2_thread_count_invariance."""
    close = (600 + np.cumsum(np.random.default_rng(5).normal(0, 0.17, 800))).astype(np.float32)
    first = rmv.rmv_all_n(close, rmv.N_VALUES)
    second = rmv.rmv_all_n(close, rmv.N_VALUES)
    assert np.array_equal(first, second), "kernel is not reproducible run to run"

    # A row computed alone must equal the same row computed alongside the others.
    for n in (3, 11, 24):
        a = int(np.flatnonzero(rmv.N_VALUES == n)[0])
        alone = rmv.rmv_all_n(close, np.array([n]))[0]
        assert np.array_equal(alone, first[a]), f"n={n} depends on which other n ran"


def test_unit2_contract_and_buffer_reuse() -> None:
    """The output contract Units 3-7 depend on, and the preallocated-buffer path."""
    close = (600 + np.cumsum(np.random.default_rng(2).normal(0, 0.17, 300))).astype(np.float32)

    out = rmv.rmv_all_n(close, rmv.N_VALUES)
    assert out.dtype == np.float32 and out.shape == (rmv.N_VALUES.size, close.size)
    assert out.flags["C_CONTIGUOUS"], "rows must be contiguous -- Unit 6 keeps one in L1"

    buf = np.empty_like(out)
    same = rmv.rmv_all_n(close, rmv.N_VALUES, out=buf)
    assert same is buf, "out= did not write into the caller's buffer"
    assert np.array_equal(buf, out)

    for bad, why in (
        (np.empty((2, close.size), np.float32), "wrong shape"),
        (np.empty(out.shape, np.float64), "wrong dtype"),
    ):
        try:
            rmv.rmv_all_n(close, rmv.N_VALUES, out=bad)
        except ValueError:
            continue
        raise AssertionError(f"accepted an out buffer with the {why}")

    # A repeated median is undefined below 3 points.
    for bad_ns in (np.array([2]), np.array([3, 1])):
        try:
            rmv.rmv_all_n(close, bad_ns)
        except ValueError:
            continue
        raise AssertionError(f"accepted n={bad_ns}")

    # float64 input must be accepted and give the same answer as float32 input.
    assert np.array_equal(rmv.rmv_all_n(close.astype(np.float64), rmv.N_VALUES), out)


def test_unit2_live_path_reuses_the_same_kernel() -> None:
    """Unit 12a needs backtest and live to share one code path. Live holds a ring buffer
    of the last n bars and reads [:, -1]; that must equal the full-series value at the
    same bar, or parity fails for a reason no amount of live testing would explain."""
    close = (600 + np.cumsum(np.random.default_rng(9).normal(0, 0.17, 500))).astype(np.float32)
    full = rmv.rmv_all_n(close, rmv.N_VALUES)

    for n in (3, 12, 24):
        a = int(np.flatnonzero(rmv.N_VALUES == n)[0])
        for t in (n - 1, 250, 499):
            ring = close[t - n + 1 : t + 1]
            live = rmv.rmv_all_n(ring, np.array([n]))[0, -1]
            assert live == full[a, t], f"n={n} t={t}: live {live} != backtest {full[a, t]}"


def test_unit2_budget_on_real_data() -> None:
    """SPEC §2.4 budget: < 5s for the full grid over the whole sample. Skips without cache."""
    npz, _ = data._cache_paths("SPY", "sip")
    if not npz.exists():
        print("       (skipped: no cache)")
        return

    import time
    from datetime import datetime, timezone

    bars = data.load_bars(
        "SPY", datetime(2016, 1, 1, tzinfo=timezone.utc),
        datetime(2026, 9, 1, tzinfo=timezone.utc), refresh=False,
    )
    out = np.empty((rmv.N_VALUES.size, len(bars)), np.float32)
    rmv.rmv_all_n(bars.close[:200], rmv.N_VALUES)  # warm the JIT

    start = time.perf_counter()
    rmv.rmv_all_n(bars.close, rmv.N_VALUES, out=out)
    elapsed = time.perf_counter() - start

    assert elapsed < 5.0, f"{elapsed:.2f}s for {rmv.N_VALUES.size}x{len(bars):,}, budget 5s"
    assert not np.isnan(out).any()
    # The old `nbytes < 50e6` was true by arithmetic and measured nothing. tracemalloc is
    # no better -- numba allocates through its own C runtime, which Python never sees.
    # The honest cheap check is that out= is written in place, so the 22.6 MB buffer is
    # the whole footprint; the timing assertion above is the real budget.
    assert rmv.rmv_all_n(bars.close, rmv.N_VALUES, out=out) is out



def test_unit2_rejects_bad_input() -> None:
    """The kernel is on the live order path, so a wrong-shaped call must raise, not return
    a plausible number. Each of these silently produced garbage before Unit 2's review."""
    close = (600 + np.cumsum(np.random.default_rng(4).normal(0, 0.17, 200))).astype(np.float32)

    def rejects(why, **kw):
        try:
            rmv.rmv_all_n(**kw)
        except ValueError:
            return
        raise AssertionError(f"accepted {why}")

    # A NaN comes back as a finite, plausible slope 61% of the time: numba's np.median is
    # partition-based and does not propagate NaN the way numpy's does.
    nan_close = close.copy()
    nan_close[50] = np.nan
    rejects("NaN in close", close=nan_close, ns=np.array([10]))
    inf_close = close.copy()
    inf_close[50] = np.inf
    rejects("inf in close", close=inf_close, ns=np.array([10]))

    # Too few bars silently zeroed that whole row -- the likeliest live failure, a short
    # ring buffer after a restart or a data hole.
    rejects("close shorter than max(n)", close=close[:10], ns=rmv.N_VALUES)
    rejects("2-D close", close=close.reshape(2, 100), ns=np.array([10]))
    rejects("empty ns", close=close, ns=np.array([], dtype=np.int64))
    rejects("2-D ns", close=close, ns=np.array([[3, 4]]))

    # out= aliasing its own input would have the kernel overwrite the prices it is reading.
    buf = np.empty((1, close.size), np.float32)
    rejects("out aliasing close", close=buf[0], ns=np.array([10]), out=buf)
    rejects("Fortran-ordered out", close=close, ns=np.array([3, 4]),
            out=np.asfortranarray(np.empty((2, close.size), np.float32)))

    # N_VALUES is a module global handed to every caller; an in-place edit would redefine
    # the grid process-wide.
    assert not rmv.N_VALUES.flags.writeable, "N_VALUES is mutable"


def test_unit2_live_ring_reproduces_the_crossing_rule() -> None:
    """SPEC §2's rule is a crossing, so live needs RMedV at t AND t-1.

    A ring of exactly n makes [:, -2] a warmup 0.0, so `RMedV[t-1] < vup` is always true
    and the crossing rule degrades into a level rule. Measured on a 4000-bar walk at
    n=24: 71 real buy signals become 1294. The ring must hold max(ns) + 1 bars.
    """
    close = (600 + np.cumsum(np.random.default_rng(11).normal(0, 0.17, 1200))).astype(np.float32)
    full = rmv.rmv_all_n(close, rmv.N_VALUES)

    for n in (3, 12, 24):
        a = int(np.flatnonzero(rmv.N_VALUES == n)[0])
        for t in (n, 600, 1199):  # t >= n so the ring has a real previous bar
            ring = close[t - n : t + 1]  # n + 1 bars
            live = rmv.rmv_all_n(ring, np.array([n]))
            assert live[0, -1] == full[a, t], f"n={n} t={t}: live current != backtest"
            assert live[0, -2] == full[a, t - 1], (
                f"n={n} t={t}: live previous {live[0, -2]} != backtest {full[a, t - 1]} "
                "-- the crossing rule needs a genuine RMedV[t-1], not a warmup zero"
            )
            assert live[0, -2] != 0.0 or full[a, t - 1] == 0.0

    # Demonstrate the hazard the +1 exists to avoid. A ring of exactly n is a perfectly
    # legal call -- the kernel cannot know the caller wanted a crossing -- so this is not
    # a rejection, it is the reason the docstring mandates max(ns) + 1.
    short = rmv.rmv_all_n(close[:24], np.array([24]))
    assert short[0, -1] != 0.0, "the current bar should still be computed"
    assert short[0, -2] == 0.0, (
        "a ring of exactly n must expose [:, -2] as a warmup zero -- if this ever stops "
        "being true the docstring's +1 rationale needs rewriting, not deleting"
    )


def test_unit2_inner_median_tie_breaking() -> None:
    """n=4's inner medians run over 3 points (odd), so the n=4 test above only pins the
    OUTER median. n=5 runs its inner medians over 4 points and pins the inner rule too --
    both lower- and upper-middle inner variants pass the n=4 test."""
    from scipy.stats import siegelslopes

    rng = np.random.default_rng(21)
    found = 0
    for _ in range(400):
        y = np.round(rng.uniform(0, 20, 5), 2).astype(np.float32)
        ref = float(siegelslopes(y.astype(np.float64), np.arange(5, dtype=float))[0])
        lower = _lower_median_reference(y.astype(np.float64))
        if abs(ref - lower) < 1e-9:
            continue  # this series does not discriminate; try another
        found += 1
        got = float(rmv.rmv_all_n(y, np.array([5]))[0, -1])
        assert got == np.float32(ref), f"n=5 inner tie-break: {got} vs numpy {ref} on {y}"
    assert found > 50, f"only {found} discriminating n=5 series found -- test is toothless"


def test_unit2_gate_never_exposes_an_invalid_rmv() -> None:
    """The integration invariant between Unit 1 and Unit 2, on real data.

    The kernel is deliberately gap-unaware: it computes straight across session
    boundaries and holes, and `gate` is what keeps those values from being traded. That
    contract is only worth anything if it actually holds for every n in the grid -- and
    SPEC §2's crossing rule reads RMedV[t-1], so bar t-1's window must be clean too.

    The margin here is exactly zero (the gap-to-first-gated-bar distance is MAX_N+1, the
    precise minimum), so this test is what stops a later change to MAX_N, the session
    start, or the blackout width from silently trading gap-contaminated signals.
    """
    npz, _ = data._cache_paths("SPY", "sip")
    if not npz.exists():
        print("       (skipped: no cache)")
        return

    from datetime import datetime, timezone

    # Whole sample: the check is a few cumsums, so there is no reason to subset it.
    bars = data.load_bars(
        "SPY", datetime(2016, 1, 1, tzinfo=timezone.utc),
        datetime(2026, 9, 1, tzinfo=timezone.utc), refresh=False,
    )
    gaps = np.flatnonzero(np.diff(bars.ts) > data.BAR_NS) + 1  # index of each post-gap bar
    is_break = np.zeros(len(bars), bool)
    is_break[gaps] = True
    gated = np.flatnonzero(bars.gate == 1)
    assert gated.size > 150_000, f"only {gated.size} gated bars -- fixture too small"

    for n in rmv.N_VALUES:
        # A window ending at t covers t-n+1..t; it is contaminated if any bar strictly
        # inside it starts a new session. Check bar t and bar t-1 (the crossing rule).
        for offset in (0, 1):
            ends = gated - offset
            assert ends.min() >= n - 1, (
                f"n={n}: a gated bar at index {ends.min()} has no full window "
                "(the array start is not being treated as a gap)"
            )
            starts = ends - n + 1
            # cumulative count of breaks in (start, end] must be zero
            cum = np.concatenate([[0], np.cumsum(is_break)])
            bad = int(np.count_nonzero(cum[ends + 1] - cum[starts + 1]))
            assert bad == 0, f"n={n}, t-{offset}: {bad} gated bars whose window spans a gap"


def test_unit2_intra_session_slice_is_gated_off() -> None:
    """A slice that begins mid-session has no earlier bar to diff against, so nothing
    marks its start as a gap. Before the fix an 11:00 ET start left gate[0] == 1 and 275
    gated bars across the grid reading warmup zeros."""
    npz, _ = data._cache_paths("SPY", "sip")
    if not npz.exists():
        print("       (skipped: no cache)")
        return

    from datetime import datetime, timezone

    for hour_utc, label in ((15, "11:00 ET"), (19, "15:00 ET")):
        bars = data.load_bars(
            "SPY", datetime(2019, 6, 3, hour_utc, tzinfo=timezone.utc),
            datetime(2019, 6, 10, tzinfo=timezone.utc), refresh=False,
        )
        assert bars.gate[: data.MAX_N].sum() == 0, f"{label}: gate open during warmup"
        exposed = sum(
            int(((np.arange(len(bars)) < n - 1) & (bars.gate == 1)).sum())
            for n in rmv.N_VALUES
        )
        assert exposed == 0, f"{label}: {exposed} gated bars read a warmup value"

# ------------------------------------------------------------------ unit 3: normalization

# [M25 p.28] Table B, sd(RMedV * sqrt(N)) for CL 5min, N=3..20 (SPEC §1.2). The paper's own
# `1/Std Mult Ave` for these rows is 9.693120, which is what makes them self-checking: a
# mistyped digit does not reproduce it.
CL_TABLE_B = np.array([
    0.114441, 0.112604, 0.107389, 0.106082, 0.103985, 0.103547, 0.102365, 0.102245,
    0.101559, 0.101442, 0.100930, 0.100853, 0.100549, 0.100603, 0.100329, 0.100422,
    0.100210, 0.100223,
])
CL_XMULT = 9.693120

# [M25 p.27] Table A, raw sd(RMedV) over the same N and the same stated run. Kept because it
# is the only source evidence for the 1/sqrt(N) law itself (SPEC §1.2), and because it
# disagrees with Table B -- see test_unit3_paper_tables_disagree.
CL_TABLE_A = np.array([
    0.065024, 0.055546, 0.047342, 0.042738, 0.038771, 0.036130, 0.033673, 0.031903,
    0.030213, 0.028895, 0.027620, 0.026593, 0.025612, 0.024809, 0.024003, 0.023349,
    0.022674, 0.022103,
])
CL_NS = np.arange(3, 21)

# SPY, measured over 2016-01-04..2017-12-29 on gated bars (PLAN Unit 3). This was meant to be
# the shipped constant; it is a test fixture instead, because
# test_unit3_frozen_xmult_does_not_transfer is what happened when it was measured.
XMULT_FROZEN_2016_17 = 7.183306

# PLAN §3 Unit 9: the final 6 months are written once and not opened until the project's last
# action. Unit 3 stops here even though sd(RMedV) is a property of the data rather than an OOS
# result -- the tail is cheap to respect and expensive to un-spend.
TAIL_START = datetime(2026, 3, 1, tzinfo=timezone.utc)


@functools.cache
def _real_bars():
    """Full-sample `Bars` and RMedV matrix, cache-only, truncated before the withheld tail.

    Returns `(bars, matrix float32[22, T])` or None when there is no cache. ~5 s, so Units 3
    and 4 share one copy. Truncating on load rather than masking afterwards is what keeps
    the tail structurally out of reach, and it changes nothing: RMedV at t reads only bars
    t-n+1..t, and the gate's rules are all backward-looking, so every retained value is
    identical to the same bar in a full-sample load.

    ponytail: a missing cache makes every real-data test print "(skipped)" and still count
    as PASS, so the empirical basis of Units 3 and 4 can evaporate quietly. Deferred, not
    accepted -- the fix (a skip count in `main`) belongs to Units 1-4 at once, not to one.
    """
    npz, _ = data._cache_paths("SPY", "sip")
    if not npz.exists():
        return None
    bars = data.load_bars(
        "SPY", datetime(2016, 1, 1, tzinfo=timezone.utc), TAIL_START, refresh=False,
    )
    return bars, rmv.rmv_all_n(bars.close)


@functools.cache
def _real_series():
    """Gated pre-tail RMedV as one contiguous float64 matrix, plus session bounds and ET.

    Returns `(rows float64[22, G], session_start int64[S+1], et)` or None. Gated bars are
    gathered once, so every window below is a slice rather than a mask over 257k booleans.
    """
    got = _real_bars()
    if got is None:
        return None
    bars, matrix = got
    keep = np.flatnonzero(bars.gate == 1)
    et = data.to_et(bars.ts[keep])
    day = et.normalize().asi8
    # First position of each session, plus a terminating bound, so sessions s..e are
    # rows[:, starts[s]:starts[e]].
    starts = np.concatenate([np.searchsorted(day, np.unique(day)), [keep.size]])
    return np.ascontiguousarray(matrix[:, keep], dtype=np.float64), starts, et


def _norm_sd(rows: np.ndarray, ns: np.ndarray, mult: float) -> np.ndarray:
    """sd(RMedV_N * mult * sqrt(N)) per N -- 1.0 everywhere is a perfectly normalized slice."""
    return np.std(rows, axis=1, ddof=1) * np.sqrt(ns) * mult


def test_unit3_xmult_reproduces_the_paper_appendix() -> None:
    """[M25 p.28]'s own published multiplier, 9.693120, out of `rmv.xmult`.

    The only external oracle the formula has. Table B is sd(RMedV * sqrt(N)), so the rows
    handed to `xmult` are synthesised to have sample sd exactly `table_b / sqrt(N)`; the
    function then has to put the sqrt(N) back, average 1/sd rather than sd, and stop at
    N=20. Dropping any one of those three does not land on 9.6931.
    """
    rng = np.random.default_rng(0)
    length = 4096
    rows = np.empty((CL_NS.size, length), dtype=np.float32)
    for i, target in enumerate(CL_TABLE_B / np.sqrt(CL_NS)):
        x = rng.standard_normal(length)
        rows[i] = (x - x.mean()) / x.std(ddof=1) * target
    got = rmv.xmult(rows, np.ones(length, bool), CL_NS)
    assert abs(got - CL_XMULT) < 1e-4, f"{got} != {CL_XMULT}"

    # The N=3..20 range is load-bearing, not decoration: [M25]'s table also lists N=2 with
    # sd=0, and the CL grid runs to N=24. Averaging a different range is a different number.
    wide = rmv.xmult(rows[:12], np.ones(length, bool), CL_NS[:12])
    assert abs(wide - CL_XMULT) > 0.1, "xmult is insensitive to which N it averages"

    # A gappy `ns`, because every other fixture here is 3,4,5,... and a row *index* would
    # pass all of them. `xmult` advertises an arbitrary ns and PLAN §8's cheap A/Bs would
    # hand it a sub-grid; `sqrt(ns[a])` must read the value, never the position.
    odd = np.arange(3, 21, 2)
    got_odd = rmv.xmult(rows[::2], np.ones(length, bool), odd)
    assert abs(got_odd - np.mean(1.0 / CL_TABLE_B[::2])) < 1e-4, got_odd


def test_unit3_paper_tables_disagree() -> None:
    """SPEC §9-L. [M25]'s p.27 and p.28 tables claim to be the same run, and are not.

    Table B is a uniform +1.38% above sqrt(N) * Table A at every one of the 18 N -- a scale
    offset, not a formula difference, so the two pages saw slightly different value sets. It
    matters only because re-deriving xmult from Table A gives 9.8266, not the published
    9.6931, and someone will eventually try.
    """
    ratio = CL_TABLE_B / (CL_TABLE_A * np.sqrt(CL_NS))
    assert ratio.min() > 1.013 and ratio.max() < 1.017, (ratio.min(), ratio.max())
    from_a = float(np.mean(1.0 / (CL_TABLE_A * np.sqrt(CL_NS))))
    assert abs(from_a - 9.8266) < 1e-3, from_a
    assert abs(from_a - CL_XMULT) > 0.1, "the two tables would have to agree for this to pass"


def test_unit3_xmult_rejects_bad_input() -> None:
    """Every guard, each for something a caller can actually hit."""
    rows = np.ones((3, 100), np.float32) * np.arange(100, dtype=np.float32)
    ns = np.array([3, 4, 5])
    ok = np.ones(100, bool)

    def rejects(fragment: str, *args) -> None:
        try:
            rmv.xmult(*args)
        except ValueError as exc:
            assert fragment in str(exc), f"wrong message for {fragment!r}: {exc}"
        else:
            raise AssertionError(f"accepted input that should raise {fragment!r}")

    rejects("one row per n", rows, ok, np.array([3, 4]))       # ns/matrix length mismatch
    rejects("one row per n", rows[0], ok, ns)                  # 1-D matrix
    rejects("mask must be", rows, np.ones(99, bool), ns)       # mask/series length mismatch
    # The one a caller will actually write: `Bars.gate` is int8, and numpy reads an integer
    # array as fancy indexing, not as a mask -- row[gate] returns row[0]/row[1] repeatedly and
    # the resulting sd looks perfectly reasonable. Must raise, never guess.
    rejects("mask must be", rows, np.ones(100, np.int8), ns)
    rejects("mask must be", rows, np.arange(100), ns)
    rejects("fewer than 2 bars", rows, np.zeros(100, bool), ns)
    one = np.zeros(100, bool)
    one[7] = True                                          # ddof=1 on one sample is NaN
    rejects("fewer than 2 bars", rows, one, ns)
    rejects("not finite", np.full((3, 100), np.nan, np.float32), ok, ns)
    rejects("nothing to average", rows, ok, np.array([21, 22, 23]))  # every n > CAL_N_MAX
    rejects("is 0.0 at n=3", np.zeros((3, 100), np.float32), ok, ns)  # constant slice

    # The mask is honoured, and it is not optional: leaving warmup zeros in alone moves the
    # answer by more than 1%, which is the whole reason there is no default.
    warm = rows.copy()
    warm[:, :20] = 0.0
    unmasked = rmv.xmult(warm, ok, ns)
    masked = rmv.xmult(warm, np.arange(100) >= 20, ns)
    assert abs(masked - unmasked) > 0.01 * unmasked, (masked, unmasked)


def test_unit3_sqrt_n_law_holds_for_spy() -> None:
    """SPEC §1.2's premise: sd(RMedV) falls as 1/sqrt(N). Verified on SPY, not assumed.

    SPY's log-log slope is -0.539 against the law's -0.5; [M25]'s own CL table is -0.567, so
    SPY is steeper than the law by *less* than his data is. The whole normalization rests on
    this being approximately true, and the residual is exactly why the done-when is +-0.15
    for N>=5 rather than +-0.05 everywhere.
    """
    real = _real_series()
    if real is None:
        print("       (skipped: no cache)")
        return
    rows, _, _ = real
    ns = rmv.N_VALUES
    sd = np.std(rows, axis=1, ddof=1)

    assert np.all(np.diff(sd) < 0), "sd(RMedV) is not monotonically falling in N"
    slope = float(np.polyfit(np.log(ns[ns <= 20]), np.log(sd[ns <= 20]), 1)[0])
    assert -0.60 < slope < -0.50, f"log-log slope {slope:.4f} is not near the -0.5 law"

    cl_slope = float(np.polyfit(np.log(CL_NS), np.log(CL_TABLE_A), 1)[0])
    assert slope > cl_slope, f"SPY {slope:.4f} should be shallower than CL {cl_slope:.4f}"

    # [M25 p.27] leads with sd(4)/sd(20) = 2.51 as the spread one vup range cannot cover.
    ratio = float(sd[1] / sd[17])
    assert 2.3 < ratio < 2.7, f"sd(4)/sd(20) = {ratio:.3f}, expected ~2.47"


def test_unit3_frozen_xmult_does_not_transfer() -> None:
    """The measurement that killed `norm.json`, kept re-runnable (PLAN Unit 3).

    A single frozen multiplier was the planned deliverable. Calibrated on 2016-17 it is
    ~3.2x off over 2018-25 -- RMedV is dollars per bar, and SPY went $210 -> $690 through a
    6x range of realized vol. PLAN Unit 3 pre-registered the trigger as a ~2x swing in the
    per-year diagnostic; the measured swing is 6.45x, and in the worst year 34.8% of gated
    bars sit beyond the top of the 0.25..3.50 grid, where every combo is a clone of every
    other.

    This asserts the *failure*, so it is a watch on the decision rather than on the code: if
    SPY ever stops behaving this way, this test fails and the frozen option reopens.
    """
    real = _real_series()
    if real is None:
        print("       (skipped: no cache)")
        return
    rows, _, et = real
    ns = rmv.N_VALUES
    years = et.year.to_numpy()

    calibrated = rmv.xmult(rows, years <= 2017, ns)
    assert abs(calibrated - XMULT_FROZEN_2016_17) < 1e-3, calibrated

    held_out = _norm_sd(rows[:, years >= 2018], ns, calibrated)
    off = np.abs(held_out - 1.0)[ns >= 5]
    assert off.min() > 1.0, (
        f"frozen xmult is within {off.min():.2f} of 1.0 on held-out data -- PLAN Unit 3's "
        "+-0.15 done-when may now be reachable, so revisit norm.json"
    )

    per_year = np.array([
        _norm_sd(rows[:, years == y], ns, calibrated).mean()
        for y in range(2016, 2027) if (years == y).sum() > 1000
    ])
    swing = per_year.max() / per_year.min()
    assert swing > 2.0, f"per-year swing {swing:.2f}x is inside PLAN's 2x tolerance"

    # The tail is not opened here. Asserted against a literal, not against TAIL_START:
    # comparing the data to the constant that cut it can never fail, and this unit already
    # published one table computed over the withheld period before that was noticed.
    assert str(et[-1]) == "2026-02-27 15:50:00-05:00", et[-1]


def test_unit3_per_window_xmult_spans_the_grid() -> None:
    """The replacement done-when: normalization is refitted per IS window (SPEC §1.2).

    PLAN Unit 3 asked for normalized sd within +-0.15 for N>=5. Frozen, that holds in 9% of
    windows; refitted per window it holds in 100%, worst case 0.142 -- and N=21..24 are
    genuine extrapolation, since xmult only averages N=3..20.

    Windows here are 21 gated sessions stepping 5, which is Unit 7's IS/OOS shape in bar
    space without pre-empting its calendar. Unit 7 owns the real window generator.
    """
    real = _real_series()
    if real is None:
        print("       (skipped: no cache)")
        return
    rows, starts, _ = real
    ns = rmv.N_VALUES
    big = ns >= 5
    sessions = starts.size - 1
    assert sessions > 2500, f"only {sessions} sessions -- fixture too small"

    win_ok = frozen_ok = windows = 0
    worst = 0.0
    oos_win, oos_frozen = [], []
    for s in range(21, sessions - 5, 5):
        is_slice = rows[:, starts[s - 21] : starts[s]]
        oos_slice = rows[:, starts[s] : starts[s + 5]]
        if is_slice.shape[1] < 500 or oos_slice.shape[1] < 100:
            continue
        windows += 1
        mult = rmv.xmult(is_slice, np.ones(is_slice.shape[1], bool), ns)

        dev = np.abs(_norm_sd(is_slice, ns, mult) - 1.0)[big]
        win_ok += dev.max() <= 0.15
        worst = max(worst, float(dev.max()))
        frozen_dev = np.abs(_norm_sd(is_slice, ns, XMULT_FROZEN_2016_17) - 1.0)[big]
        frozen_ok += frozen_dev.max() <= 0.15

        oos_win.append(np.median(np.abs(_norm_sd(oos_slice, ns, mult) - 1.0)[big]))
        oos_frozen.append(
            np.median(np.abs(_norm_sd(oos_slice, ns, XMULT_FROZEN_2016_17) - 1.0)[big])
        )

    assert windows > 400, f"only {windows} windows"
    assert win_ok / windows >= 0.99, (
        f"per-window normalization holds +-0.15 in only {win_ok / windows:.1%} of windows "
        f"(worst deviation {worst:.3f})"
    )
    # Without this the test would pass on any multiplier at all, frozen included.
    assert frozen_ok / windows < 0.20, (
        f"frozen xmult holds in {frozen_ok / windows:.1%} of windows -- the test no longer "
        "distinguishes the two options"
    )

    # The honest caveat, bounded so a regression shows up. Refitting fixes the *in-sample*
    # scale exactly; next week's scale is still only predicted, at a median 26% error. That
    # is strategy risk for Unit 7 to carry, not a normalization defect -- but Unit 8 should
    # know that a vup chosen on IS lands on an OOS week whose sd differs by about a quarter.
    med_win, med_frozen = float(np.median(oos_win)), float(np.median(oos_frozen))
    assert med_win < 0.40, f"OOS transfer degraded: median |sd-1| = {med_win:.3f}"
    assert med_win < med_frozen / 2, (med_win, med_frozen)
    # The tail is what Unit 9 will feel, not the median: 12 of 506 OOS weeks come in at more
    # than 2x their IS scale, which is the saturation regime the frozen constant was rejected
    # for. Bounded so a regression is visible rather than averaged away.
    p99 = float(np.percentile(oos_win, 99))
    assert p99 < 2.0, f"OOS transfer p99 |sd-1| = {p99:.3f}"


# --------------------------------------------------------------------- unit 4: simulation

# SPEC §3.2: $0.01/share round-trip slippage + ~$0.017/share SEC/TAF on the sell. One
# scalar per trade; the notional-scaling half of that is Unit 9's to verify, not Unit 4's.
COST = 0.027


def _sim_reference(rmv_row, close, gate, vup, vdn, cost) -> list[tuple]:
    """Deliberately slow, obviously-correct pure Python. PLAN Unit 4 asks for exactly this.

    Structured differently from the kernel on purpose. The kernel makes one linear pass and
    notices the gate's 1->0 edge as it goes; this splits the series into maximal runs of
    `gate == 1` first and then trades inside each, where "flatten at the end of the run" and
    "never open on the run's last bar" are single visible statements. Two implementations
    that share a shape share their bugs.
    """
    runs, lo = [], None
    for t in range(len(close)):
        if gate[t] == 1 and lo is None:
            lo = t
        elif gate[t] != 1 and lo is not None:
            runs.append((lo, t - 1))
            lo = None
    if lo is not None:
        runs.append((lo, len(close) - 1))

    trades = []
    for lo, hi in runs:
        pos, entry = 0, None
        for t in range(lo, hi + 1):
            if t == 0:
                continue  # no predecessor bar, so no crossing exists
            cur, prev = float(rmv_row[t]), float(rmv_row[t - 1])
            if cur >= vup and prev < vup:
                sig = 1
            elif cur <= -vdn and prev > -vdn:
                sig = -1
            else:
                sig = 0
            if sig == 0 or sig == pos or t == hi:
                # t == hi: an entry here would fill at the same close its own forced exit
                # fills at, so it is a guaranteed -cost trade and live would never take it.
                continue
            if pos != 0:
                trades.append((entry, t, pos, pos * (float(close[t]) - float(close[entry])) - cost))
            pos, entry = sig, t
        if pos != 0:  # SPEC §2's flat-at-15:55, filled on the run's last bar
            trades.append((entry, hi, pos, pos * (float(close[hi]) - float(close[entry])) - cost))
    return trades


def _random_case(rng) -> tuple:
    """One random (rmv_row, close, gate, vup, vdn).

    Half the cases draw the thresholds straight out of `rmv_row`, so `RMedV[t] == vup` holds
    exactly and the `>=` boundary is genuinely exercised rather than merely reachable. That
    is only constructible because `simulate` takes raw thresholds: a grid-unit boundary would
    have to survive a division, which lands exactly only ~91% of the time.
    """
    total = int(rng.integers(30, 300))
    close = (400.0 + np.cumsum(rng.normal(0, 0.15, total))).astype(np.float32)

    gate = np.zeros(total, np.int8)
    for _ in range(int(rng.integers(1, 5))):  # a few runs, sometimes touching either end
        lo = int(rng.integers(0, total))
        gate[lo : lo + int(rng.integers(1, 60))] = 1

    if rng.random() < 0.5:
        rmv_row = (rng.integers(-24, 25, total) / 8.0).astype(np.float32)
        pick = lambda: abs(float(rmv_row[rng.integers(total)])) or 0.125  # noqa: E731
        vup, vdn = pick(), pick()
    else:
        rmv_row = rng.normal(0, 0.4, total).astype(np.float32)
        n = int(rng.integers(3, 25))
        xmult = float(rng.uniform(0.679, 13.830))  # SPEC §1.2.1's measured per-window span
        vup = rmv.threshold(float(rng.uniform(0.25, 3.5)), xmult, n)
        vdn = rmv.threshold(float(rng.uniform(0.25, 3.5)), xmult, n)
    return rmv_row, close, gate, vup, vdn


def test_unit4_threshold_is_the_only_conversion() -> None:
    """`rmv.threshold` is `v / (xmult * sqrt(n))` with one rounding, and that matters.

    PLAN Unit 6 pins the convention so backtest, grid, replay and live cannot drift onto
    opposite sides of a boundary. A second call site that wrote the algebraically identical
    `v / xmult / sqrt(n)` would land on a different float64 in about a third of cases, which
    is why "one expression" is the invariant and not a stylistic preference.
    """
    rng = np.random.default_rng(3)
    n = rng.integers(3, 25, 200_000)
    xmult = rng.uniform(0.679, 13.830, 200_000)  # SPEC §1.2.1's measured per-window span
    v = rng.choice(np.arange(1, 15) * 0.25, 200_000)  # SPEC §3.3's grid

    one = v / (xmult * np.sqrt(n))
    two = v / xmult / np.sqrt(n)
    apart = int(np.count_nonzero(one != two))
    assert apart > 20_000, (
        f"the two association orders differ in only {apart}/200000 draws -- if they have "
        "become identical, `threshold`'s one-expression rationale needs restating"
    )
    assert np.max(np.abs(one - two) / one) < 1e-15, "a 1-ulp claim, not a real disagreement"

    for i in (0, 1, 12345, 199_999):
        got = rmv.threshold(float(v[i]), float(xmult[i]), int(n[i]))
        assert got == float(one[i]), f"threshold is not the pinned expression: {got} != {one[i]}"
    # And the direction of the conversion: a grid value is far larger than its raw threshold,
    # which is why passing one for the other is silent rather than loud (SPEC §1.2.1).
    assert 5.0 < 1.0 / rmv.threshold(1.0, 3.063, 3) < 6.0
    assert 14.0 < 1.0 / rmv.threshold(1.0, 3.063, 24) < 16.0


def test_unit4_matches_reference_on_random_series() -> None:
    """PLAN Unit 4's done-when: kernel == reference trade-for-trade, 1000 random cases.

    Bit-exact on the net figures, not within a tolerance. Both sides do the same float64
    arithmetic in the same order, so anything looser would hide a real divergence.
    """
    rng = np.random.default_rng(4)
    boundary = traded = 0
    for case in range(1000):
        rmv_row, close, gate, vup, vdn = _random_case(rng)
        got = rmv.simulate(rmv_row, close, gate, vup, vdn, COST)
        want = _sim_reference(rmv_row, close, gate, vup, vdn, COST)

        assert len(got) == len(want), (
            f"case {case}: kernel produced {len(got)} trades, reference {len(want)}"
        )
        for i, (e, x, d, net) in enumerate(want):
            assert (got[i, 0], got[i, 1], got[i, 2]) == (e, x, d), (
                f"case {case} trade {i}: kernel {tuple(got[i, :3])} != reference {(e, x, d)}"
            )
            assert got[i, 3] == net, f"case {case} trade {i}: net {got[i, 3]!r} != {net!r}"
        traded += len(want)
        # Count exact-threshold hits so the `>=` boundary cannot quietly stop being covered.
        boundary += int(np.count_nonzero(((rmv_row == vup) | (rmv_row == -vdn)) & (gate == 1)))

    assert traded > 5000, f"only {traded} trades over 1000 cases -- the generator went inert"
    assert boundary > 200, f"only {boundary} bars sat exactly on a threshold -- boundary untested"


def test_unit4_hand_built_trade_list() -> None:
    """The exact expected trade list, on a series small enough to check by hand."""
    #                 0    1    2    3     4     5    6    7   8   9    10    11    12  13  14   15   16  17
    gate = np.array([ 0,   0,   1,   1,    1,    1,   1,   1,  0,  0,    1,    1,    1,  0,  1,   1,   1,  0], np.int8)
    rmvr = np.array([0., 0., 1.5, 0.5, -1.5, -0.5, 1.0, 2.0, 0., 0., -2.0, -1.0, -3.0, 0., 0., 0.5, 2.0, 0.], np.float32)
    close = np.arange(100, 118, dtype=np.float32)

    got = rmv.simulate(rmvr, close, gate, 1.0, 1.0, COST)

    # The column order is the contract Unit 5 reads by index, and nothing else pins it: a
    # reorder of TRADE_COLS alone changes no behaviour, so it has to be asserted outright.
    assert rmv.TRADE_COLS == ("entry", "exit", "dir", "net")

    # Run [2..7]: buy the 1.5 crossing at 2; reverse short on -1.5 at 4; reverse long on the
    #   1.0 crossing at 6. Bar 7 is not a crossing (prev is already 1.0) and is the run's last
    #   bar anyway, so the long is flattened there by the gate.
    # Run [10..12]: short the -2.0 crossing at 10. -1.0 and -3.0 are not crossings, because
    #   prev is already at or below -1.0. Flattened at 12.
    # Run [14..16]: the 2.0 crossing lands on bar 16, the run's last bar -- suppressed.
    want = [
        (2, 4, 1, 104.0 - 102.0 - COST),
        (4, 6, -1, 104.0 - 106.0 - COST),
        (6, 7, 1, 107.0 - 106.0 - COST),
        (10, 12, -1, 110.0 - 112.0 - COST),
    ]
    assert len(got) == len(want), f"{len(got)} trades, expected {len(want)}:\n{got}"
    for i, (e, x, d, net) in enumerate(want):
        assert (got[i, 0], got[i, 1], got[i, 2]) == (e, x, d), f"trade {i}: {got[i, :3]}"
        assert abs(got[i, 3] - net) < 1e-12, f"trade {i}: net {got[i, 3]} != {net}"


def test_unit4_threshold_is_inclusive() -> None:
    """SPEC §2: `>=` and `<=`. Turning either into a strict inequality drops the trade that
    sits exactly on the threshold, which on a penny-tick instrument is not measure-zero."""
    gate = np.ones(9, np.int8)
    close = np.arange(100, 109, dtype=np.float32)

    exact = np.array([0.0, 1.0, 0.0, 0.0, 0.0, 0.0, 0.0, 0.0, 0.0], np.float32)
    assert len(rmv.simulate(exact, close, gate, 1.0, 1.0, COST)) == 1, (
        "a bar exactly equal to vup must trigger -- SPEC §2 says >=, not >"
    )
    below = exact.copy()
    below[1] = np.nextafter(np.float32(1.0), np.float32(0.0))  # one float32 ulp under
    assert len(rmv.simulate(below, close, gate, 1.0, 1.0, COST)) == 0

    down = np.array([0.0, -1.0, 0.0, 0.0, 0.0, 0.0, 0.0, 0.0, 0.0], np.float32)
    got = rmv.simulate(down, close, gate, 1.0, 1.0, COST)
    assert len(got) == 1 and got[0, 2] == -1, f"exactly -vdn must trigger a short: {got}"

    # The boundary at a threshold that is not a round number, taken straight out of the row:
    # exact by construction, which is the whole reason `vup` is raw and not a grid value.
    odd = np.array([0.0, 0.0407123, 0.0, 0.0, 0.0, 0.0, 0.0, 0.0, 0.0], np.float32)
    assert len(rmv.simulate(odd, close, gate, float(odd[1]), 1.0, COST)) == 1


def test_unit4_is_a_crossing_rule_not_a_level_rule() -> None:
    """SPEC §2's 2025 rules read RMedV[t-1]; the superseded 2005 rules do not.

    The two only diverge where the previous bar is already past the threshold and no
    position is open -- which is exactly the first gated bar of a session, whose `t-1` is
    the prior *extended-hours* bar (PLAN Unit 4 review focus, SPEC §3.1). A level rule
    would open there; a crossing rule waits for an actual crossing. This is the shape that
    catches dropping the `prev` term, and the one a same-direction repeat cannot catch,
    because holding makes a repeated signal a no-op either way.
    """
    close = np.arange(100, 106, dtype=np.float32)
    gate = np.array([0, 0, 1, 1, 1, 0], np.int8)

    up = np.array([0.0, 2.0, 2.0, 0.0, 0.0, 0.0], np.float32)
    assert len(rmv.simulate(up, close, gate, 1.0, 1.0, COST)) == 0, (
        "RMedV was already above vup on the ungated bar 1, so bar 2 is not a crossing -- "
        "a level rule opens here and this is the only place the two rules differ"
    )
    dn = np.array([0.0, -2.0, -2.0, 0.0, 0.0, 0.0], np.float32)
    assert len(rmv.simulate(dn, close, gate, 1.0, 1.0, COST)) == 0

    # ...and a genuine crossing on the same bar does open, so the assertions above are not
    # passing because the gate or the run bounds swallowed everything.
    cross = np.array([0.0, 0.0, 2.0, 0.0, 0.0, 0.0], np.float32)
    got = rmv.simulate(cross, close, gate, 1.0, 1.0, COST)
    assert len(got) == 1 and (got[0, 0], got[0, 1]) == (2, 4), got


def test_unit4_gated_first_bar_does_not_wrap_around() -> None:
    """A slice whose bar 0 is gated must not read `rmv_row[-1]` as the prior bar.

    In numba `a[-1]` is the LAST element, not an IndexError, so without the `t == 0` guard
    the first bar of such a slice would take the END of the window as its own predecessor:
    end-of-window look-ahead, silent, no exception. Unit 7's week-anchored windows never
    start on a gated bar, which is precisely why this is pinned in the kernel rather than
    left to a test that could never fire on the planned inputs.
    """
    close = np.arange(100, 106, dtype=np.float32)
    gate = np.ones(6, np.int8)
    # Bar 0 is gated and sits above vup, while the LAST bar is far below it. A wrapped
    # `prev` would read -9.0 < 1.0, call bar 0 a crossing and open a trade there.
    rmvr = np.array([2.0, 2.0, 2.0, 2.0, 2.0, -9.0], np.float32)
    got = rmv.simulate(rmvr, close, gate, 1.0, 1.0, COST)
    assert len(got) == 0, f"bar 0 read the end of the window as its predecessor: {got}"

    # Positive control: the same shape with a real crossing at bar 1 does trade, so the
    # assertion above is not passing because everything was suppressed.
    rmvr = np.array([0.0, 2.0, 2.0, 2.0, 2.0, -9.0], np.float32)
    got = rmv.simulate(rmvr, close, gate, 1.0, 1.0, COST)
    assert len(got) == 1 and (got[0, 0], got[0, 1]) == (1, 5), got


def test_unit4_reversal_is_one_bar_one_price() -> None:
    """Stop-and-reverse: the closing and opening fills are the same bar at the same price,
    and each side pays `cost` -- which is what PLAN Unit 9's cost sanity check counts."""
    gate = np.ones(7, np.int8)
    close = np.array([100.0, 101.0, 103.0, 102.0, 105.0, 104.0, 106.0], np.float32)
    rmvr = np.array([0.0, 2.0, 0.0, -2.0, 0.0, 0.0, 0.0], np.float32)

    got = rmv.simulate(rmvr, close, gate, 1.0, 1.0, COST)
    assert len(got) == 2, got
    assert got[0, 1] == got[1, 0] == 3, f"the reversal must share bar 3: {got[:, :2]}"
    assert got[0, 2] == 1 and got[1, 2] == -1
    # Long 101 -> 102, then short 102 -> 106 (flattened on the last bar). Both pay cost,
    # and both legs of the reversal fill at the same 102 -- one price, not two.
    assert abs(got[0, 3] - (102.0 - 101.0 - COST)) < 1e-12, got[0, 3]
    assert abs(got[1, 3] - (102.0 - 106.0 - COST)) < 1e-12, got[1, 3]
    assert close[int(got[0, 1])] == close[int(got[1, 0])] == 102.0


def test_unit4_no_entry_on_the_last_gated_bar() -> None:
    """A trade opened on the last gated bar of a run fills at the close its own forced exit
    fills at: zero bars, zero gross, exactly -cost.

    Live flattens at 15:55 and does not also enter, so booking these charges a cost live
    never pays and breaks the Unit 12a parity check. Measured over the pre-tail sample they
    are 1.74% of all trades (1.0-3.2% by combo) and the skip removes *exactly* the trades
    whose exit index equals their entry index -- 0 discrepancies over 16 (n, v) combos.
    """
    gate = np.array([0, 1, 1, 1, 0], np.int8)
    close = np.array([100.0, 101.0, 102.0, 103.0, 104.0], np.float32)

    # A fresh crossing exactly on bar 3, the run's last bar, with nothing open.
    rmvr = np.array([0.0, 0.0, 0.0, 2.0, 0.0], np.float32)
    assert len(rmv.simulate(rmvr, close, gate, 1.0, 1.0, COST)) == 0, (
        "an entry on the run's last bar is a guaranteed -cost trade carrying no information"
    )

    # The same crossing one bar earlier is a real trade, held to the forced exit at bar 3.
    rmvr = np.array([0.0, 0.0, 2.0, 2.0, 0.0], np.float32)
    got = rmv.simulate(rmvr, close, gate, 1.0, 1.0, COST)
    assert len(got) == 1 and (got[0, 0], got[0, 1]) == (2, 3), got

    # Suppression must not leak into an open position: the reversal signal on the last bar
    # is skipped, and the existing trade still exits there, at the price it would have.
    rmvr = np.array([0.0, 2.0, 0.0, -2.0, 0.0], np.float32)
    got = rmv.simulate(rmvr, close, gate, 1.0, 1.0, COST)
    assert len(got) == 1 and (got[0, 0], got[0, 1], got[0, 2]) == (1, 3, 1), got
    assert abs(got[0, 3] - (103.0 - 101.0 - COST)) < 1e-12, got[0, 3]

    # An array that ends while still gated force-closes on the final bar. One real session
    # in 2,680 does this in place (2019-08-12, whose data stops at 15:30), and any slice
    # that is not session-aligned can. The `t + 1 >= total` skip means the position being
    # closed was always opened strictly earlier, so this can never emit a zero-bar trade.
    open_at_end = np.array([0.0, 2.0, 0.0, 0.0, 0.0], np.float32)
    got = rmv.simulate(open_at_end, close, np.ones(5, np.int8), 1.0, 1.0, COST)
    assert len(got) == 1 and (got[0, 0], got[0, 1], got[0, 2]) == (1, 4, 1), got
    assert abs(got[0, 3] - (104.0 - 101.0 - COST)) < 1e-12, got[0, 3]


def test_unit4_gate_holds_on_real_data() -> None:
    """PLAN Unit 4's done-when against the real series: no trade opens before 10:00, none
    spans 15:55, and none is held across a bar the gate has shut.

    Checked against `gate` itself as well as against the clock. The gate carries three
    separate rules -- the 10:00 open, `session_close - 5 min`, and the max_n blackout after
    a data gap -- and on an early close the last tradeable bar opens 12:50, not 15:50.
    """
    got = _real_bars()
    if got is None:
        print("    (skipped: no SPY cache)", end="")
        return
    bars, matrix = got
    mult = rmv.xmult(matrix, bars.gate == 1)
    et = data.to_et(bars.ts)
    minute = np.asarray(et.hour * 60 + et.minute)
    day = et.normalize().asi8
    csum = np.concatenate([[0], np.cumsum(bars.gate == 1)])

    checked = 0
    for n in (3, 12, 24):
        row = matrix[int(np.flatnonzero(rmv.N_VALUES == n)[0])]
        for v in (0.25, 1.5, 3.5):
            thr = rmv.threshold(v, mult, n)
            trades = rmv.simulate(row, bars.close, bars.gate, thr, thr, COST)
            assert len(trades), f"n={n} v={v} produced no trades at all"
            entry = trades[:, 0].astype(np.int64)
            exit_ = trades[:, 1].astype(np.int64)
            tag = f"n={n} v={v}"

            assert np.all(bars.gate[entry] == 1), f"{tag}: entry on an ungated bar"
            assert np.all(bars.gate[exit_] == 1), f"{tag}: exit on an ungated bar"
            assert minute[entry].min() >= data.GATE_OPEN_MIN, (
                f"{tag}: entry at minute {minute[entry].min()}, before the 10:00 gate"
            )
            # The last gated bar OPENS 15:50 and closes 15:55; nothing opens or closes later.
            assert minute[exit_].max() < data.GATE_CLOSE_MIN, (
                f"{tag}: exit at minute {minute[exit_].max()}, at or past 15:55"
            )
            assert np.all(day[entry] == day[exit_]), f"{tag}: a trade spans a session"
            assert np.all(exit_ > entry), f"{tag}: a zero-bar trade survived"
            assert np.all(entry[1:] >= exit_[:-1]), f"{tag}: trades overlap"
            # No entry is the last gated bar of its run -- the rule itself, stated directly.
            assert np.all(bars.gate[entry + 1] == 1), f"{tag}: opened on a run's last bar"
            # Nothing is held across a shut gate. A prefix sum rather than a loop over ~10k
            # trades: a fully-gated span has as many gated bars as it has bars.
            assert np.all(csum[exit_ + 1] - csum[entry] == exit_ - entry + 1), (
                f"{tag}: a trade is held across a bar the gate has shut"
            )
            checked += len(trades)
    print(f"    ({checked} real trades checked)", end="")


def test_unit4_matches_reference_on_real_data() -> None:
    """The kernel-vs-reference equivalence again, but on the real gate.

    Early closes, the two mid-session blackouts and the 28 holed sessions are shapes
    `_random_case` does not make. One year only, because the pure-Python reference is
    ~1000x slower than the kernel -- which is the point of it.
    """
    got = _real_bars()
    if got is None:
        print("    (skipped: no SPY cache)", end="")
        return
    bars, matrix = got
    mult = rmv.xmult(matrix, bars.gate == 1)
    lo, hi = np.searchsorted(bars.ts, [
        data._to_ns(datetime(2020, 1, 1, tzinfo=timezone.utc)),
        data._to_ns(datetime(2021, 1, 1, tzinfo=timezone.utc)),
    ])
    close = np.ascontiguousarray(bars.close[lo:hi])
    gate = np.ascontiguousarray(bars.gate[lo:hi])

    for n in (3, 24):
        a = int(np.flatnonzero(rmv.N_VALUES == n)[0])
        row = np.ascontiguousarray(matrix[a, lo:hi])
        for v in (0.5, 2.0):
            thr = rmv.threshold(v, mult, n)
            k = rmv.simulate(row, close, gate, thr, thr, COST)
            r = _sim_reference(row, close, gate, thr, thr, COST)
            assert len(k) == len(r), f"n={n} v={v}: kernel {len(k)} trades, reference {len(r)}"
            for i, (e, x, d, net) in enumerate(r):
                assert (k[i, 0], k[i, 1], k[i, 2], k[i, 3]) == (e, x, d, net), (
                    f"n={n} v={v} trade {i}: {tuple(k[i])} != {(e, x, d, net)}"
                )


def test_unit4_scaling_thresholds_equals_scaling_rows() -> None:
    """PLAN Unit 6's pinned convention, checked where it actually lands -- on trades.

    Unit 6 measured 0 disagreements over 111.1M raw comparisons; that is the premise. This
    is the conclusion, and it is what a future refactor moving the division would break.
    The alternative is built the way Unit 6 rejected it: a second float32 row, scaled.
    """
    got = _real_bars()
    if got is None:
        print("    (skipped: no SPY cache)", end="")
        return
    bars, matrix = got
    mult = rmv.xmult(matrix, bars.gate == 1)
    differed = compared = 0
    for n in (3, 11, 24):
        a = int(np.flatnonzero(rmv.N_VALUES == n)[0])
        scaled = np.ascontiguousarray(matrix[a] * np.float32(mult * math.sqrt(n)))
        for v in (0.25, 1.0, 3.5):
            thr = rmv.threshold(v, mult, n)
            by_threshold = rmv.simulate(matrix[a], bars.close, bars.gate, thr, thr, COST)
            by_row = rmv.simulate(scaled, bars.close, bars.gate, v, v, COST)
            compared += len(by_threshold)
            differed += len(by_threshold) + len(by_row) - 2 * len(
                {tuple(t[:3]) for t in by_threshold} & {tuple(t[:3]) for t in by_row}
            )
    assert compared > 10000, compared
    assert differed == 0, (
        f"{differed} of {compared} trades differ between scaling thresholds and scaling "
        "rows -- PLAN Unit 6 measured 0 disagreements at the comparison level, so this is "
        "either a real regression or that measurement needs restating at the trade level"
    )


def test_unit4_costs_and_buffer_reuse() -> None:
    """`net` is gross minus exactly one `cost` per trade, and a reused buffer changes nothing.

    Unit 6 hands one buffer to all 4312 combos, so a kernel that read past `k`, or that
    depended on the buffer being clean, would pass every single-call test and fail only there.
    """
    got = _real_bars()
    if got is None:
        print("    (skipped: no SPY cache)", end="")
        return
    bars, matrix = got
    mult = rmv.xmult(matrix, bars.gate == 1)
    row = matrix[int(np.flatnonzero(rmv.N_VALUES == 12)[0])]
    thr = rmv.threshold(1.0, mult, 12)

    fresh = rmv.simulate(row, bars.close, bars.gate, thr, thr, COST).copy()
    entry = fresh[:, 0].astype(np.int64)
    exit_ = fresh[:, 1].astype(np.int64)
    gross = fresh[:, 2] * (
        bars.close[exit_].astype(np.float64) - bars.close[entry].astype(np.float64)
    )
    assert np.array_equal(fresh[:, 3], gross - COST), "net is not gross minus exactly one cost"
    # Zero cost must move every trade by exactly COST and change nothing else about it.
    free = rmv.simulate(row, bars.close, bars.gate, thr, thr, 0.0)
    assert np.array_equal(free[:, :3], fresh[:, :3]) and np.array_equal(free[:, 3], gross)

    wide = rmv.threshold(3.5, mult, 12)
    tight = rmv.threshold(0.25, mult, 12)
    buf = np.empty((len(bars), 4), np.float64)
    first = rmv.simulate(row, bars.close, bars.gate, wide, wide, COST, out=buf).copy()
    rmv.simulate(row, bars.close, bars.gate, tight, tight, COST, out=buf)  # dirties it
    again = rmv.simulate(row, bars.close, bars.gate, wide, wide, COST, out=buf)
    assert np.array_equal(first, again), "a dirty buffer changed the answer"
    # The buffer bound is len(close); the busiest combo in the grid says how much slack.
    fastest = rmv.threshold(0.25, mult, 3)
    busiest = len(rmv.simulate(matrix[0], bars.close, bars.gate, fastest, fastest, COST))
    assert busiest < len(bars), f"{busiest} trades against a {len(bars)}-row bound"
    print(f"    (busiest combo {busiest} trades / {len(bars)} rows)", end="")


def test_unit4_rejects_bad_input() -> None:
    """Every guard, each for something a caller can actually hit -- and each for a *silent*
    wrong answer rather than a crash, which is why they are all ValueError."""
    T = 40
    row = np.zeros(T, np.float32)
    close = np.full(T, 400.0, np.float32)
    gate = np.ones(T, np.int8)

    def rejects(fragment: str, *args, **kw) -> None:
        try:
            rmv.simulate(*args, **kw)
        except ValueError as exc:
            assert fragment in str(exc), f"wrong message for {fragment!r}: {exc}"
        else:
            raise AssertionError(f"accepted input that should raise {fragment!r}")

    ok = (row, close, gate, 0.1, 0.1, COST)
    rmv.simulate(*ok)  # the baseline really does pass

    rejects("close must be", row, close.astype(np.float64), gate, 0.1, 0.1, COST)
    rejects("rmv_row must be", row.astype(np.float64), close, gate, 0.1, 0.1, COST)
    # The whole 22 x T matrix instead of one row -- caught by shape, not by ndim alone.
    rejects("rmv_row must be", np.zeros((22, T), np.float32), close, gate, 0.1, 0.1, COST)
    rejects("rmv_row must be", np.zeros(T - 1, np.float32), close, gate, 0.1, 0.1, COST)
    # `Bars.gate` is int8. A bool mask would work, but pinning one dtype keeps numba to one
    # specialization and keeps every call site writing the same thing.
    rejects("gate must be", row, close, gate.astype(bool), 0.1, 0.1, COST)
    rejects("gate must be", row, close, gate.astype(np.int64), 0.1, 0.1, COST)
    # A stray value reads as "flat" under the kernel's `gate[t] != 1` and as "tradeable"
    # under the equally natural `gate[t] == 0`. Neither is wrong; guessing is.
    rejects("only 0 and 1", row, close, (gate * 2), 0.1, 0.1, COST)
    rejects("only 0 and 1", row, close, (gate * -1), 0.1, 0.1, COST)

    # The thresholds. A non-positive one makes the two crossing branches overlap; a negative
    # vdn fires the sell rule on nearly every bar; NaN makes every comparison False, which is
    # a silent flat window rather than an error. This is also where a non-positive or NaN
    # `xmult` surfaces, because `rmv.threshold` passes it straight through.
    for bad in (0.0, -1.0, np.nan, np.inf):
        rejects("vup must be", row, close, gate, bad, 0.1, COST)
        rejects("vdn must be", row, close, gate, 0.1, bad, COST)
    rejects("vup must be", row, close, gate, rmv.threshold(1.0, -3.0, 12), 0.1, COST)
    rejects("vup must be", row, close, gate, rmv.threshold(1.0, np.nan, 12), 0.1, COST)
    rejects("cost must be", row, close, gate, 0.1, 0.1, -0.01)
    rejects("cost must be", row, close, gate, 0.1, 0.1, np.nan)

    rejects("out must be float64", *ok, out=np.empty((T, 4), np.float32))
    rejects("out must be float64", *ok, out=np.empty((T, 3), np.float64))
    rejects("rows", *ok, out=np.empty((T - 1, 4), np.float64))
    rejects("C-contiguous", *ok, out=np.empty((4, T), np.float64).T)
    # A caller who sliced one scratch arena into both the price series and the buffer. The
    # views have to be contiguous to survive `ascontiguousarray`, or the alias is copied away
    # before the check can see it -- which is why the check comes after the conversion.
    arena = np.empty(T * 4, np.float64)
    arena[:] = 400.0
    alias32, alias8 = arena.view(np.float32)[:T], arena.view(np.int8)[:T]
    rejects("aliases", row, alias32, gate, 0.1, 0.1, COST, out=arena.reshape(T, 4))
    # Each input needs its own case: the guard is three `or`ed clauses and dropping any one
    # of them is invisible to a test that only aliases the first.
    rejects("aliases", alias32, close, gate, 0.1, 0.1, COST, out=arena.reshape(T, 4))
    arena[:] = 0.0  # int8 view of 0.0 is all zeros, i.e. a legal all-flat gate
    rejects("aliases", row, close, alias8, 0.1, 0.1, COST, out=arena.reshape(T, 4))


def test_unit4_budget() -> None:
    """Unit 6's < 60 ms/window has to be reachable from here, on *real* windows.

    Measured on synthetic bars this understates the cost by 3-4x: threshold density drives
    the trade count, and a synthetic row calibrated by eye produces ~5 trades per combo where
    a real window produces 16-22. So the budget runs on real IS-sized windows with each
    window's own refitted `xmult`, and asserts the whole 4312-combo sweep fits inside Unit 6's
    per-window budget on **one** thread -- `prange` is then headroom, not the thing being
    relied on.
    """
    import time

    got = _real_bars()
    if got is None:
        print("    (skipped: no SPY cache)", end="")
        return
    bars, matrix = got
    total = 1638  # PLAN §2.4's stated IS window size
    buf = np.empty((total, 4), np.float64)
    rmv._simulate(matrix[0, :total].copy(), bars.close[:total].copy(),
                  bars.gate[:total].copy(), 0.01, 0.01, COST, buf)  # JIT

    worst = worst_trades = 0
    for start in np.linspace(30_000, len(bars) - total - 1, 4).astype(int):
        sl = slice(start, start + total)
        close = np.ascontiguousarray(bars.close[sl])
        gate = np.ascontiguousarray(bars.gate[sl])
        rows = np.ascontiguousarray(matrix[:, sl])
        mult = rmv.xmult(rows, gate == 1)
        # The real grid: 22 n x 14 vup x 14 vdn = 4312, with the divide hoisted per (n, v).
        thr = np.array([[rmv.threshold(0.25 * (j + 1), mult, n) for j in range(14)]
                        for n in rmv.N_VALUES])
        trades = 0
        begin = time.perf_counter()
        for a in range(rmv.N_VALUES.size):
            row = np.ascontiguousarray(rows[a])
            for i in range(14):
                for j in range(14):
                    trades += rmv._simulate(row, close, gate, thr[a, i], thr[a, j], COST, buf)
        elapsed = time.perf_counter() - begin
        if elapsed > worst:
            worst, worst_trades = elapsed, trades
    print(f"    (worst real window {worst * 1000:.0f} ms serial for 4312 combos, "
          f"{worst_trades / 4312:.1f} trades/combo, "
          f"{worst * 1e9 / (4312 * total):.1f} ns/bar-step)", end="")
    assert worst_trades / 4312 > 10, (
        f"only {worst_trades / 4312:.1f} trades per combo -- the windows went quiet and this "
        "is measuring an empty loop, which is how the synthetic fixture understated it"
    )
    assert worst < 0.060, (
        f"{worst * 1000:.0f} ms serial for one window's 4312 combos, on one thread, against "
        "Unit 6's 60 ms budget for the whole window"
    )


def test_unit4_iex_signal_divergence_on_real_feeds() -> None:
    """PLAN Unit 4's ⚑ item, re-derivable rather than quoted. SPEC §3.2's table comes from here.

    Both feeds through the identical pipeline for June 2024. The two failures compound: IEX's
    missing pre-market bars trip the `max_n` blackout, so a quarter of SIP's tradeable bars are
    not tradeable at all on IEX; and on the bars that survive on both, the 2.50c of per-print
    noise moves the signal itself.

    Needs `cache/SPY_5min_iex.npz`, fetched once and kept beside the SIP cache precisely so
    this number stops resting on one un-repeatable network run.
    """
    npz, _ = data._cache_paths("SPY", "iex")
    if not npz.exists():
        print("    (skipped: no IEX cache)", end="")
        return
    start = datetime(2024, 6, 1, tzinfo=timezone.utc)
    end = datetime(2024, 7, 1, tzinfo=timezone.utc)
    feeds = {}
    for feed in ("sip", "iex"):
        b = data.load_bars("SPY", start, end, feed=feed, refresh=False)
        feeds[feed] = (b, rmv.rmv_all_n(b.close), rmv.xmult(rmv.rmv_all_n(b.close), b.gate == 1))
    (sip, sip_m, sip_x), (iex, iex_m, iex_x) = feeds["sip"], feeds["iex"]

    sip_gated, iex_gated = int((sip.gate == 1).sum()), int((iex.gate == 1).sum())
    assert (len(sip), sip_gated) == (1824, 1349), (len(sip), sip_gated)
    assert (len(iex), iex_gated) == (1484, 982), (len(iex), iex_gated)
    # 19 trading days in June 2024 at the full 96-bar session and 71 gated bars.
    assert sip_gated == 19 * 71 and len(sip) == 19 * 96

    shared, s_i, i_i = np.intersect1d(sip.ts, iex.ts, assume_unique=True, return_indices=True)
    both = (sip.gate[s_i] == 1) & (iex.gate[i_i] == 1)

    def signals(matrix, bars, a, thr):
        """Which way the crossing rule fires on every bar -- the decision, before position."""
        cur, prev = matrix[a][1:].astype(np.float64), matrix[a][:-1].astype(np.float64)
        sig = np.zeros(bars.ts.size, np.int8)
        sig[1:] = np.where(
            (cur >= thr) & (prev < thr), 1, np.where((cur <= -thr) & (prev > -thr), -1, 0)
        )
        return sig

    worst = 0.0
    sip_trades = iex_trades = 0
    for n in (3, 6, 12, 24):
        a = int(np.flatnonzero(rmv.N_VALUES == n)[0])
        for v in (0.25, 1.0, 2.0, 3.5):
            s_thr, i_thr = rmv.threshold(v, sip_x, n), rmv.threshold(v, iex_x, n)
            disagree = (signals(sip_m, sip, a, s_thr)[s_i][both]
                        != signals(iex_m, iex, a, i_thr)[i_i][both])
            worst = max(worst, 100 * disagree.mean())
            sip_trades += len(rmv.simulate(sip_m[a], sip.close, sip.gate, s_thr, s_thr, COST))
            iex_trades += len(rmv.simulate(iex_m[a], iex.close, iex.gate, i_thr, i_thr, COST))

    cover = 100 * iex_gated / sip_gated
    ratio = 100 * iex_trades / sip_trades
    print(f"    (IEX gates {cover:.1f}% of SIP's bars, makes {ratio:.1f}% of the trades, "
          f"worst signal disagreement {worst:.1f}%)", end="")
    # SPEC §3.2's recorded figures. Bounds, not equalities, so a cache refresh reports a
    # drift rather than a mystery -- but tight enough that drift is what it would report.
    assert 72.0 < cover < 74.0, f"IEX gated-bar coverage {cover:.1f}%, SPEC §3.2 says 72.8%"
    assert 70.0 < ratio < 77.0, f"IEX trade ratio {ratio:.1f}%, SPEC §3.2 says 73.2%"
    assert worst > 8.0, (
        f"worst signal disagreement {worst:.1f}%, SPEC §3.2 says 11.9%. If IEX has become "
        "this close to SIP, §3.2's rejection is what needs rewriting, not this bound"
    )


def test_unit4_iex_noise_would_break_the_signal() -> None:
    """PLAN Unit 4's flagged item: Unit 1's IEX *price* divergence, turned into a *signal* one.

    Unit 1 measured IEX against SIP over June 2024 inside the session window (SPEC §3.2):
    81.4% coverage, no usable pre-market warmup, and a median 2.50c difference on every
    shared print -- 15% of a median 17c bar move, 76.3% of bars off by >= 1c. Whether that
    matters is a Unit 4 question, because trades are the output.

    Fetched live once through this exact pipeline and recorded in SPEC §3.2: over June 2024
    IEX gates 982 bars against SIP's 1,349, signals disagree on up to 11.9% of the bars both
    feeds gate, and IEX yields 73.2% of SIP's trades. That measurement needs the network, so
    what runs here is the mechanism, offline: inject IEX's measured 2.50c of per-bar noise
    into the cached SIP series and count how many trades move. A repeated median resists
    outlier points, not error on every point, and this is the number that says so.
    """
    got = _real_bars()
    if got is None:
        print("    (skipped: no SPY cache)", end="")
        return
    bars, _ = got
    lo, hi = np.searchsorted(bars.ts, [
        data._to_ns(datetime(2024, 6, 1, tzinfo=timezone.utc)),
        data._to_ns(datetime(2024, 7, 1, tzinfo=timezone.utc)),
    ])
    close = np.ascontiguousarray(bars.close[lo:hi])
    gate = np.ascontiguousarray(bars.gate[lo:hi])

    # sd chosen so the median absolute perturbation is IEX's measured 2.50c: for a normal,
    # median|x| = 0.6745 sd.
    noisy = (close + np.random.default_rng(24).normal(0, 0.025 / 0.6745, close.size)).astype(
        np.float32
    )
    assert abs(float(np.median(np.abs(noisy - close))) - 0.025) < 0.004, "noise is not IEX-sized"

    clean_m = rmv.rmv_all_n(close)
    noisy_m = rmv.rmv_all_n(noisy)
    mult = rmv.xmult(clean_m, gate == 1)
    moved = total = 0
    for n in (3, 12, 24):
        a = int(np.flatnonzero(rmv.N_VALUES == n)[0])
        for v in (0.25, 1.0, 2.0):
            thr = rmv.threshold(v, mult, n)
            c = rmv.simulate(clean_m[a], close, gate, thr, thr, COST)
            d = rmv.simulate(noisy_m[a], noisy, gate, thr, thr, COST)
            same = len({tuple(t[:3]) for t in c} & {tuple(t[:3]) for t in d})
            moved += len(c) - same
            total += len(c)
    pct = 100 * moved / total
    print(f"    ({pct:.1f}% of trades move under IEX-sized noise)", end="")
    assert pct > 20.0, (
        f"only {pct:.1f}% of trades moved under 2.5c of per-bar noise. If the signal really "
        "is that robust, SPEC §3.2's IEX rejection needs re-arguing -- not this assertion "
        "loosening"
    )

# ============================================================== Unit 5: the metric set


def _metrics_reference(trades: np.ndarray) -> dict:
    """Deliberately slow numpy/scipy reference for all 24 of `rmv.METRIC_COLS`.

    Structured differently from the kernel on purpose, the same way `_sim_reference` is:
    `np.polyfit` residuals instead of hand-solved normal equations, `itertools.groupby`
    instead of running streak counters, `scipy.stats.kendalltau` instead of a pair loop,
    `np.maximum.accumulate` instead of a rolling peak. Two implementations that share a
    shape share their bugs.

    `eq2R2` comes back **unclamped**, which the kernel's cannot be -- that is what lets
    `test_unit5_matches_the_reference_on_random_trades` check that the clamp is only ever
    removing float noise and is not hiding a wrong 2nd-order solve.
    """
    from itertools import groupby

    from scipy import stats

    net = np.ascontiguousarray(trades[:, 3], dtype=np.float64)
    held = np.ascontiguousarray(trades[:, 1] - trades[:, 0], dtype=np.float64)
    k = net.size
    win, lose = net > 0.0, net < 0.0

    def med(a):
        return float(np.median(a)) if a.size else 0.0

    def longest(sign):
        runs = (len(list(g)) for key, g in groupby(np.sign(net).astype(int)) if key == sign)
        return float(max(runs, default=0))

    eq = np.cumsum(net)
    peak = np.maximum.accumulate(np.concatenate([[0.0], eq]))[1:]  # equity is 0 before trade 1
    dd = float((eq - peak).min()) if k else 0.0

    def r2(deg):
        # sstot is measured about the mean, which is what makes this R-squared and not a raw
        # residual ratio. polyfit gets the *centered* x for conditioning only -- shifting x
        # changes the coefficients but not the fitted values, so R-squared is untouched.
        #
        # The undefined value differs by column and that is deliberate, not an oversight:
        # eqR2 is screened (`< 80`, `<= 50`) so its sentinel has to FAIL, and eq2R2 is
        # argmax-picked so its sentinel has to LOSE. See `rmv.metrics`.
        undefined = 100.0 if deg == 1 else 0.0
        if k < deg + 1:
            return undefined
        y = eq - eq.mean()
        sstot = float((y * y).sum())
        if sstot <= 0.0:
            return undefined
        x = np.arange(k, dtype=np.float64)
        x = x - x.mean()
        res = y - np.polyval(np.polyfit(x, y, deg), x)
        return 100.0 * (1.0 - float((res * res).sum()) / sstot)

    tau = 0.0
    if k >= 2:
        # scipy returns nan when a variable is constant; the kernel returns 0.0 there, so
        # the mapping is made here rather than loosening the comparison downstream.
        t = 100.0 * float(stats.kendalltau(np.arange(k), eq, variant="b").statistic)
        tau = t if np.isfinite(t) else 0.0

    ownp = float(net[win].sum())
    gloss = float(-net[lose].sum())
    sd = float(np.std(net, ddof=1)) if k >= 2 else 0.0
    mean = float(net.mean()) if k else 0.0
    return {
        "tnp": float(net.sum()),
        "nT": float(k),
        "PF": np.inf if gloss == 0.0 else ownp / gloss,
        "%P": 100.0 * int(win.sum()) / k if k else 0.0,
        "mTrd": med(net),
        "mWTr": med(net[win]),
        "mLTr": med(net[lose]),
        "mLb": med(held[lose]) if lose.any() else np.inf,
        "mWb": med(held[win]) if win.any() else np.inf,
        "lr": longest(-1),
        "wr": longest(1),
        "dd": dd,
        "llt": float(net[lose].min()) if lose.any() else 0.0,
        "std": sd,
        "t": mean / (sd / math.sqrt(k)) if sd > 0.0 else 0.0,
        "eqR2": r2(1),
        "eq2R2": r2(2),
        "ktau": tau,
        "osnp": float(net.sum()),
        "ont": float(k),
        "ownp": ownp,
        "ownt": float(int(win.sum())),
        "ollt": float(net[lose].min()) if lose.any() else 0.0,
        "odd": dd,
    }


def _flat_trades(nets) -> np.ndarray:
    """A `simulate`-shaped array holding just the nets: every trade one bar long, all long.

    The metric kernel reads `net` and `exit - entry` and nothing else, so this is the whole
    input for any test about a P&L sequence rather than about the trades that produced it.
    """
    nets = np.asarray(nets, dtype=np.float64)
    out = np.zeros((nets.size, 4), dtype=np.float64)
    out[:, 0] = np.arange(nets.size, dtype=np.float64) * 2.0
    out[:, 1] = out[:, 0] + 1.0
    out[:, 2] = 1.0
    out[:, 3] = nets
    return out


def _random_trades(rng, k: int | None = None) -> np.ndarray:
    """A plausible `simulate` output: exit > entry always, dir in {-1, +1}, net float64."""
    k = int(rng.integers(0, 60)) if k is None else k
    entry = np.sort(rng.integers(0, 2000, k)).astype(np.float64)
    held = rng.integers(1, 40, k).astype(np.float64)
    out = np.empty((k, 4), np.float64)
    out[:, 0] = entry
    out[:, 1] = entry + held
    out[:, 2] = rng.choice([-1.0, 1.0], k)
    # A mixture, so some draws are all-winners, some all-losers and some near-flat -- the
    # degenerate branches have to be reachable by chance, not only by construction.
    scale = float(rng.choice([0.05, 0.5, 5.0]))
    out[:, 3] = rng.normal(float(rng.uniform(-0.4, 0.4)), scale, k)
    if rng.random() < 0.15 and k:  # exact zeros, which cost=0 can really produce
        out[rng.integers(0, k, max(1, k // 5)), 3] = 0.0
    return out


def test_unit5_matches_the_reference_on_random_trades() -> None:
    """All 24 columns against numpy/scipy over 3000 random trade arrays.

    PLAN Unit 5's done-when. The arrays are drawn to reach the degenerate branches by
    chance -- 0 trades, 1 trade, all-winners, all-losers, exact zero nets -- and the run
    asserts each was actually hit, so a later change that made them unreachable would not
    quietly turn this into a test of the easy path only.
    """
    rng = np.random.default_rng(11)
    worst = {c: 0.0 for c in rmv.METRIC_COLS}
    seen = {"k0": 0, "k1": 0, "k2": 0, "no_losers": 0, "no_winners": 0, "zero_net": 0}
    worst_neg_eq2 = 0.0
    for _ in range(3000):
        trades = _random_trades(rng)
        got = rmv.metrics(trades)
        want = _metrics_reference(trades)
        net = trades[:, 3]
        seen["k0"] += len(trades) == 0
        seen["k1"] += len(trades) == 1
        seen["k2"] += len(trades) == 2
        seen["no_losers"] += len(trades) > 0 and not (net < 0).any()
        seen["no_winners"] += len(trades) > 0 and not (net > 0).any()
        seen["zero_net"] += bool((net == 0).any())
        worst_neg_eq2 = min(worst_neg_eq2, want["eq2R2"])
        for i, col in enumerate(rmv.METRIC_COLS):
            a, b = float(got[i]), float(want[col])
            if np.isinf(b):
                assert np.isinf(a) and np.sign(a) == np.sign(b), f"{col}: {a} vs {b}"
                continue
            assert np.isfinite(a), f"{col} is {a} -- SPEC §7 rule 2 bans NaN in kernels"
            worst[col] = max(worst[col], abs(a - b) / max(1.0, abs(b)))
    for name, count in seen.items():
        assert count > 0, f"no random case exercised {name}; the branch is untested"
    # The unclamped reference is what pins the kernel's `if eq2r2 < 0.0` as noise removal.
    assert worst_neg_eq2 > -1e-6, (
        f"the reference's own 2nd-order R2 reached {worst_neg_eq2}, which a least-squares "
        "fit with an intercept cannot do -- the kernel's clamp would be hiding a real defect"
    )
    bad = {c: v for c, v in worst.items() if v > 1e-9}
    assert not bad, f"columns disagreeing with numpy/scipy by more than 1e-9: {bad}"
    print(f"    (24 cols x 3000 arrays, worst relative gap "
          f"{max(worst.values()):.2e} at {max(worst, key=worst.get)})", end="")


def test_unit5_hand_built_trade_list() -> None:
    """One five-trade list with every one of the 24 answers written out by hand.

    PLAN Unit 5's done-when asks for exactly this. The reference above and the kernel could
    in principle share a misreading of SPEC §6.1; a literal cannot.
    """
    trades = np.array([
        # entry  exit  dir    net
        [0.0,    2.0,  1.0,   1.00],
        [3.0,    5.0, -1.0,  -0.50],
        [6.0,    9.0,  1.0,   2.00],
        [10.0,  12.0,  1.0,  -0.25],
        [13.0,  20.0, -1.0,   0.75],
    ])
    got = dict(zip(rmv.METRIC_COLS, rmv.metrics(trades)))

    # equity is 0, then 1.00, 0.50, 2.50, 2.25, 3.00 -- peak 1.00 then 2.50, so the worst
    # excursion is trade 2 taking 1.00 down to 0.50.
    assert got["tnp"] == 3.0 and got["nT"] == 5.0
    assert got["ownp"] == 3.75 and got["ownt"] == 3.0          # 1.00 + 2.00 + 0.75
    assert got["PF"] == 5.0                                     # 3.75 / 0.75
    assert got["%P"] == 60.0                                    # 3 of 5, on a 0-100 scale
    assert got["mTrd"] == 0.75                                  # median of all five nets
    assert got["mWTr"] == 1.0 and got["mLTr"] == -0.375         # -0.375 = mean(-0.50, -0.25)
    assert got["mWb"] == 3.0 and got["mLb"] == 2.0              # bars held = exit - entry
    assert got["wr"] == 1.0 and got["lr"] == 1.0                # winners and losers alternate
    assert got["dd"] == -0.5 and got["llt"] == -0.5             # both stored negative
    assert abs(got["std"] - math.sqrt(4.075 / 4)) < 1e-15       # ddof=1 about a mean of 0.6
    assert abs(got["t"] - 0.6 / (got["std"] / math.sqrt(5))) < 1e-13
    assert abs(got["ktau"] - 60.0) < 1e-13                      # 8 concordant, 2 discordant
    # Both regressions as literals. `eq2R2 >= eqR2` alone would have admitted anything in
    # [74.298, 100] -- including the 100.0 an exact-fit bug produces -- which is the
    # "passes against broken logic" case PLAN §4 names, in the one test whose whole job is
    # to owe nothing to `_metrics_reference`.
    assert abs(got["eqR2"] - 74.297752808989) < 1e-11, got["eqR2"]
    assert abs(got["eq2R2"] - 74.398073836276) < 1e-11, got["eq2R2"]
    # ...and eqR2 is still the square of Pearson's r between trade index and equity, x100.
    r = float(np.corrcoef(np.arange(5), np.cumsum(trades[:, 3]))[0, 1])
    assert abs(got["eqR2"] - 100.0 * r * r) < 1e-12
    assert got["eq2R2"] > got["eqR2"], "a quadratic cannot fit worse than the line it contains"
    for is_col, oos_col in (("tnp", "osnp"), ("nT", "ont"), ("llt", "ollt"), ("dd", "odd")):
        assert got[is_col] == got[oos_col], f"{oos_col} is not {is_col} on the same trades"


def test_unit5_degenerate_combos_are_defined() -> None:
    """0, 1, 2 and 3 trades, all-winners and all-losers -- and what each does to SPEC §5.

    These are not hypothetical: over 34,496 real combos 0.24% traded not at all, 2.3% fewer
    than twice and 4.8% fewer than three times, so every filter meets them every window.
    Each assertion below is about a *selection* consequence, not just a returned number.
    """
    def row(nets, held=None):
        nets = np.asarray(nets, float)
        t = np.zeros((nets.size, 4))
        t[:, 0] = np.arange(nets.size) * 10.0
        t[:, 1] = t[:, 0] + (1.0 if held is None else np.asarray(held, float))
        t[:, 2] = 1.0
        t[:, 3] = nets
        return dict(zip(rmv.METRIC_COLS, rmv.metrics(t)))

    # Each sentinel is checked by the SELECTION it forces, not just by its value. The three
    # sentinels point in three different directions because their consumers do.
    empty = row([])
    assert np.isinf(empty["PF"]), "a no-trade row must fail every PF upper bound"
    assert empty["eqR2"] == 100.0, "a no-trade row must fail CL2's < 80 and CL4's <= 50"
    assert np.isinf(empty["mLb"]) and np.isinf(empty["mWb"]), (
        "a no-trade row must sort LAST on mLb -- CL2 and CL4 rank on the smallest"
    )
    assert empty["eq2R2"] == 0.0, "a no-trade row must never win meyers2005's max eq2R2"
    zero_valued = [c for c in rmv.METRIC_COLS if c not in ("PF", "eqR2", "mLb", "mWb")]
    assert all(empty[c] == 0.0 for c in zero_valued), empty
    # ...and the three screens it now fails, spelled out, because this is the whole point:
    assert not (1.0 <= empty["PF"] <= 2.0), "meyers2005 would admit a no-trade row"
    assert not (empty["PF"] < 4.0 and empty["lr"] < 3.0 and empty["eqR2"] < 80.0), "CL2 would"
    assert not (empty["lr"] <= 3.0 and empty["eqR2"] <= 50.0), "CL4 would"

    one = row([-1.0])
    assert one["nT"] == 1.0 and one["mLTr"] == -1.0 and one["llt"] == -1.0
    assert one["dd"] == -1.0, "equity starts at zero, so a single loser is a full drawdown"
    assert one["std"] == 0.0 and one["t"] == 0.0, "sd of one sample is undefined, not zero-div"
    assert one["eq2R2"] == 0.0 and one["ktau"] == 0.0
    assert one["eqR2"] == 100.0, "one trade is no trend, and must not pass CL4's eqR2 <= 50"
    assert one["mLb"] == 1.0 and np.isinf(one["mWb"]), "one loser has a real mLb"

    two = row([1.0, 2.0])
    assert two["eqR2"] == 100.0, "two points determine a line"
    assert two["eq2R2"] == 0.0, "a quadratic through two points is underdetermined, not exact"
    assert two["ktau"] == 100.0 and np.isinf(two["PF"])
    # ...and 100.0 is what makes a two-trade row fail CL2's eqR2 < 80 and CL4's eqR2 <= 50.

    three = row([1.0, -2.0, 4.0])
    assert three["eq2R2"] > 100.0 - 1e-9, "three points determine a parabola"
    assert three["eqR2"] < 90.0, "...but not a line"

    winners = row([1.0, 2.0, 3.0])
    assert np.isinf(winners["PF"]) and winners["mLTr"] == 0.0 and np.isinf(winners["mLb"])
    assert winners["llt"] == 0.0 and winners["dd"] == 0.0 and winners["%P"] == 100.0
    # llt and dd being 0.0 on an all-winner row is [M25 Table 1]'s published value, not a
    # choice: its 12/15/14 week reads ont 4, ownt 4, ollt 0, odd 0.
    losers = row([-1.0, -2.0])
    assert losers["PF"] == 0.0 and losers["mWTr"] == 0.0 and np.isinf(losers["mWb"])
    assert losers["%P"] == 0.0 and losers["ownp"] == 0.0 and losers["ownt"] == 0.0
    assert losers["lr"] == 2.0 and losers["wr"] == 0.0
    # And the drawdown of an all-loser row is the FULL cumulative loss, because equity is
    # zero-based and the running peak starts there. [M25 Table 1] 01/07/15: ont 2,
    # osnp -2020, odd -2020 -- a peak seeded from the first equity value would give -990.
    assert losers["dd"] == -3.0 and losers["osnp"] == -3.0 and losers["odd"] == -3.0

    # A net of exactly 0 is neither side, and breaks both streaks. cost=0 makes these real.
    zeros = row([1.0, 1.0, 0.0, 1.0, 1.0])
    assert zeros["nT"] == 5.0 and zeros["ownt"] == 4.0 and zeros["%P"] == 80.0
    assert zeros["wr"] == 2.0, "a zero-net trade must break the winning streak, not extend it"
    assert zeros["lr"] == 0.0, "...and must not count as a loser either"
    assert np.isinf(zeros["PF"]) and zeros["mTrd"] == 1.0


def test_unit5_median_matches_numpy_including_ties() -> None:
    """`rmv._median` is the whole metric set's tie-breaking and even-count convention.

    It replaces `np.median` to keep Unit 6's `prange` allocation-free, so it has to agree
    with it exactly -- including on the even counts where the answer is the mean of two
    middles, and on the heavy ties an integer bar count produces.
    """
    rng = np.random.default_rng(5)
    buf = np.empty(64, np.float64)
    for _ in range(20_000):
        m = int(rng.integers(1, 41))
        # Integers over a small range, so ties are the common case rather than a corner.
        x = rng.integers(-5, 6, m).astype(np.float64) / 2.0
        buf[:m] = x
        assert rmv._median(buf, m) == float(np.median(x)), f"m={m} {x}"
        assert np.all(np.diff(buf[:m]) >= 0), "the slice was not left sorted"
    assert rmv._median(buf, 0) == 0.0, "the empty median must be 0.0, not NaN (SPEC §7 rule 2)"
    # Reverse-sorted input is insertion sort's worst case; it must still be correct.
    buf[:9] = np.arange(9, 0, -1, dtype=np.float64)
    assert rmv._median(buf, 9) == 5.0


def test_unit5_ktau_matches_scipy() -> None:
    """`ktau` is tau-b against trade order, and scipy is the oracle.

    The trade index has no ties, which collapses tau-b to `(C - D) / sqrt((C + D) * nPairs)`.
    That collapse is only valid on the x side, so the tie cases below are on the equity
    curve, where they are genuinely reachable whenever `cost` is 0.
    """
    from scipy import stats

    rng = np.random.default_rng(19)
    worst = 0.0
    for _ in range(2000):
        k = int(rng.integers(2, 60))
        nets = rng.integers(-3, 4, k).astype(np.float64)  # integers, so equity ties happen
        t = _flat_trades(nets)
        got = float(rmv.metrics(t)[rmv.METRIC_COLS.index("ktau")])
        want = 100.0 * float(stats.kendalltau(np.arange(k), np.cumsum(nets),
                                              variant="b").statistic)
        if not np.isfinite(want):
            assert got == 0.0, "a constant equity curve must be 0.0, not scipy's nan"
            continue
        worst = max(worst, abs(got - want))
    assert worst < 1e-12, f"worst gap against scipy.stats.kendalltau: {worst}"
    # A strictly rising and a strictly falling equity curve bracket the scale, which is
    # x100 and signed -- the same 0-100 magnitude eqR2 and eq2R2 use, so a filter threshold
    # literal means one thing across all three correlation columns.
    up = _flat_trades(np.full(6, 1.0))
    dn = _flat_trades(np.full(6, -1.0))
    assert rmv.metrics(up)[rmv.METRIC_COLS.index("ktau")] == 100.0
    assert rmv.metrics(dn)[rmv.METRIC_COLS.index("ktau")] == -100.0
    print(f"    (worst gap vs scipy over 2000 curves: {worst:.2e})", end="")


def test_unit5_equity_regressions_survive_the_dollar_bases() -> None:
    """⚑ PLAN §1.7 rule 3 and SPEC §7's table, reproduced on *trade-indexed* equity.

    Two claims, and the second is the one that matters. First: the naive one-pass float32
    `n*Sxy - Sx*Sy` form really does fall apart once the equity curve sits on a dollar base,
    which is the regime [M25 Table 1]'s $233,000 `NetEq` column lives in. Second: adding
    that base to `metrics`' own input -- which a pure shift of the equity curve is, exactly
    -- moves `eqR2`, `eq2R2` and `ktau` by less than a float32 eps, because the kernel
    mean-centres in float64 before it accumulates anything.
    """
    def naive_f32_r2(y):
        """The form SPEC §7 says goes wrong: uncentred, float32, one pass."""
        y = y.astype(np.float32)
        n = np.float32(y.size)
        sx = sy = sxy = sxx = syy = np.float32(0.0)
        for i in range(y.size):
            xi = np.float32(i)
            sx += xi
            sy += y[i]
            sxy += xi * y[i]
            sxx += xi * xi
            syy += y[i] * y[i]
        den = (n * sxx - sx * sx) * (n * syy - sy * sy)
        if den <= 0:
            return float("nan")
        num = n * sxy - sx * sy
        return 100.0 * float(num * num / den)

    rng = np.random.default_rng(23)
    cols = [rmv.METRIC_COLS.index(c) for c in ("eqR2", "eq2R2", "ktau")]
    report, shifted_worst = {}, 0.0
    for base in (0.0, 100.0, 200_000.0):
        flips = nonfinite = 0
        worst_naive = 0.0
        for _ in range(600):
            k = int(rng.integers(5, 120))
            t = _flat_trades(rng.normal(0.02, 0.5, k))
            plain = rmv.metrics(t)
            # Adding `base` to the first trade's net adds it to *every* equity value and to
            # nothing else -- a pure shift of the curve, on the same k points.
            t[0, 3] += base
            shifted = rmv.metrics(t)
            shifted_worst = max(shifted_worst, max(abs(shifted[c] - plain[c]) for c in cols))

            exact = float(plain[cols[0]])
            naive = naive_f32_r2(np.cumsum(t[:, 3]))
            if not np.isfinite(naive):
                nonfinite += 1
                continue
            worst_naive = max(worst_naive, abs(naive - exact))
            flips += (exact <= 50.0) != (naive <= 50.0)
        report[base] = (worst_naive, flips, nonfinite)

    assert report[0.0][1:] == (0, 0) and report[0.0][0] < 0.01, (
        f"zero-based should be the safe row of SPEC §7's table, got {report[0.0]}"
    )
    assert report[100.0][0] > 0.1, "base $100 should already be visibly wrong in float32"
    assert report[200_000.0][2] > 100, (
        "base $200,000 should drive the naive float32 form non-finite -- if it no longer "
        "does, this test has stopped demonstrating why the kernel centres"
    )
    # float32 eps at 100 is 7.6e-6, so this is well below anything the stored column can see.
    assert shifted_worst < 1e-6, (
        f"a pure ${200_000:,} shift of the equity curve moved eqR2/eq2R2/ktau by "
        f"{shifted_worst}; the mean-centring is not doing its job"
    )
    for base, (w, f, nf) in report.items():
        print(f"    (base {base:,.0f}: naive f32 off by {w:.4g}, {f} screen flips, "
              f"{nf} non-finite)", end="\n" if base != 200_000.0 else "")
    print(f"    (kernel under a ${200_000:,} shift: {shifted_worst:.2e})", end="")


def test_unit5_scale_and_sign_conventions_are_pinned() -> None:
    """The conventions SPEC §6 does not state and SPEC §5's filters cannot survive wrong.

    Every one of these is a silent failure: a 0-1 `eqR2` passes `< 80` and `<= 50` for every
    row in the table, and an unsigned `mLTr` inverts which row `min mLTr` picks. None of
    them raises, and none is caught by comparing against a reference that shares the error.
    """
    # A perfectly straight equity line is 100, not 1.0.
    line = _flat_trades(np.full(20, 0.5))
    got = dict(zip(rmv.METRIC_COLS, rmv.metrics(line)))
    assert abs(got["eqR2"] - 100.0) < 1e-9, f"eqR2 is not on a 0-100 scale: {got['eqR2']}"
    assert abs(got["eq2R2"] - 100.0) < 1e-9, f"eq2R2 is not on a 0-100 scale: {got['eq2R2']}"
    assert got["%P"] == 100.0, "%P is not on a 0-100 scale"

    # Loss metrics come out negative, which is [M25 Figure 2]'s own convention (LLTr = -3540)
    # and SPEC §9-E's open reading. Unit 8 derives the magnitude convention with abs(); the
    # reverse is not derivable, which is why this direction is the one that is stored.
    mixed = np.array([[0., 1., 1., 3.0], [2., 4., 1., -5.0], [5., 6., 1., -1.0]])
    got = dict(zip(rmv.METRIC_COLS, rmv.metrics(mixed)))
    assert got["mLTr"] == -3.0 and got["llt"] == -5.0 and got["dd"] == -6.0
    assert got["ollt"] == got["llt"] and got["odd"] == got["dd"]
    assert abs(got["PF"] - 0.5) < 1e-15, "PF is not net gross-profit over net gross-loss"

    # SPEC §9-D's alternative `|r|` reading has to be reachable from what is stored, and it
    # is -- by transforming the THRESHOLD, which is exact, rather than the column.
    # `|r| <= 0.50` is `R2 <= 0.25` is `eqR2 <= 25`; `|r| < 0.80` is `eqR2 < 64`.
    rng = np.random.default_rng(31)
    flips = 0
    for _ in range(2000):
        t = _flat_trades(rng.normal(0.1, 1.0, int(rng.integers(4, 60))))
        eqr2 = float(rmv.metrics(t)[rmv.METRIC_COLS.index("eqR2")])
        r = abs(float(np.corrcoef(np.arange(len(t)), np.cumsum(t[:, 3]))[0, 1]))
        assert abs(math.sqrt(eqr2 / 100.0) - r) < 1e-11, "sqrt(eqR2/100) is not |r|"
        flips += (r <= 0.50) != (eqr2 <= 25.0) or (r < 0.80) != (eqr2 < 64.0)
    assert flips == 0, f"the |r| threshold transform disagreed with |r| on {flips} curves"

    # The three degenerate sentinels point in three directions, one per consumer. Asserted
    # here as well as in the degenerate test so a "tidy them all to 0.0" refactor fails.
    empty = rmv.metrics(np.zeros((0, 4)))
    assert float(empty[rmv.METRIC_COLS.index("eqR2")]) == 100.0     # screened -> must fail
    assert float(empty[rmv.METRIC_COLS.index("eq2R2")]) == 0.0      # argmax -> must lose
    assert np.isinf(empty[rmv.METRIC_COLS.index("mLb")])            # min-ranked -> must sort last
    assert np.isinf(empty[rmv.METRIC_COLS.index("PF")])             # upper-bounded -> must fail

    # `std` is the one column the net-vs-gross question provably cannot move: `cost` is a
    # constant subtracted from every trade, and variance is translation-invariant.
    t = _flat_trades(rng.normal(0.1, 1.0, 40))
    gross = t.copy()
    gross[:, 3] += COST
    i = rmv.METRIC_COLS.index("std")
    assert abs(float(rmv.metrics(t)[i]) - float(rmv.metrics(gross)[i])) < 1e-12

    # The 24 columns are PLAN §1.6's two blocks in order, and Unit 7 slices them apart.
    assert rmv.METRIC_COLS[rmv.IS_COLS] == (
        "tnp", "nT", "PF", "%P", "mTrd", "mWTr", "mLTr", "mLb", "mWb",
        "lr", "wr", "dd", "llt", "std", "t", "eqR2", "eq2R2", "ktau",
    ), "the 18 columns Unit 7 writes to pwfo_is.npy moved"
    assert rmv.METRIC_COLS[rmv.OOS_COLS] == (
        "osnp", "ont", "ownp", "ownt", "ollt", "odd",
    ), "the 6 columns Unit 7 writes to pwfo_oos.npy moved"
    assert rmv.N_METRICS == 24 and len(set(rmv.METRIC_COLS)) == 24


def test_unit5_winner_set_is_net_not_gross() -> None:
    """⚑ `ownp`/`ownt` count the NET winners, and the two sets genuinely differ.

    SPEC §6.2 reads *"Winning Trades total Net Profits"*, so a trade whose gross cleared but
    whose `cost` did not is a loser. Measured here rather than asserted in the abstract: the
    gap is 122 trades at n=6/v=0.5 and 15 at n=12/v=1.0, and a refactor that drifted onto the
    gross reading would move `ownp` without moving anything that raises.
    """
    got = _real_bars()
    if got is None:
        print("    (skipped: no SPY cache)", end="")
        return
    bars, matrix = got
    mult = rmv.xmult(matrix, bars.gate == 1)
    seen = []
    for n, v, gross_want, net_want in ((6, 0.5, 4810, 4688), (12, 1.0, 1517, 1502)):
        a = int(np.flatnonzero(rmv.N_VALUES == n)[0])
        thr = rmv.threshold(v, mult, n)
        trades = rmv.simulate(matrix[a], bars.close, bars.gate, thr, thr, COST)
        net = trades[:, 3]
        gross_winners = int((net + COST > 0).sum())
        net_winners = int((net > 0).sum())
        assert (gross_winners, net_winners) == (gross_want, net_want), (
            f"n={n} v={v}: {gross_winners}/{net_winners} against PLAN's "
            f"{gross_want}/{net_want} -- the sample or the cost convention moved"
        )
        m = dict(zip(rmv.METRIC_COLS, rmv.metrics(trades)))
        assert m["ownt"] == float(net_winners), "ownt counted the gross winner set"
        assert abs(m["ownp"] - float(net[net > 0].sum())) < 1e-9
        assert abs(m["ownp"] / -net[net < 0].sum() - m["PF"]) < 1e-12, (
            "ownp is not PF's numerator; they are the same sum and must not drift apart"
        )
        seen.append((n, v, gross_winners, net_winners))
    print("    (" + ", ".join(f"n={n} v={v}: {g} gross / {x} net winners"
                              for n, v, g, x in seen) + ")", end="")


def test_unit5_bars_held_is_never_zero_on_real_data() -> None:
    """⚑ PLAN Unit 5's inherited invariant, checked where the trades are actually produced.

    `mLb` is a rank-and-pick metric in CL2 and CL4, so a zero-bar losing trade would sort to
    the front of the bottom-k rank and hand the filter a trade that never existed. Unit 4
    suppresses the entry that would create one; this is the assertion that it stayed
    suppressed, over a real grid rather than a constructed case.
    """
    got = _real_bars()
    if got is None:
        print("    (skipped: no SPY cache)", end="")
        return
    bars, matrix = got
    total, checked, trades_seen = 1638, 0, 0
    buf = np.empty((total, 4), np.float64)
    for start in (40_000, 120_000, 200_000):
        sl = slice(start, start + total)
        close = np.ascontiguousarray(bars.close[sl])
        gate = np.ascontiguousarray(bars.gate[sl])
        rows = np.ascontiguousarray(matrix[:, sl])
        mult = rmv.xmult(rows, gate == 1)
        for a in (0, 9, 21):
            row = np.ascontiguousarray(rows[a])
            n = int(rmv.N_VALUES[a])
            for v in (0.25, 1.0, 3.5):
                thr = rmv.threshold(v, mult, n)
                k = rmv._simulate(row, close, gate, thr, thr, COST, buf)
                held = buf[:k, 1] - buf[:k, 0]
                assert k == 0 or held.min() >= 1.0, f"a {held.min()}-bar trade at n={n} v={v}"
                assert np.all(np.abs(buf[:k, 2]) == 1.0), "dir outside {-1, +1}"
                checked += 1
                trades_seen += k
    print(f"    ({trades_seen} trades over {checked} combos, none held 0 bars)", end="")


def test_unit5_reference_agrees_on_a_real_grid() -> None:
    """The kernel against numpy/scipy on trades a real window really produced.

    The random arrays above are drawn from a distribution; this is the shape the grid
    actually emits -- long runs of one sign, heavy ties in the bar counts, equity curves
    that trend. Cheap enough to run a sample of combos, which is what makes it worth having
    on top of the 3000 random ones.
    """
    got = _real_bars()
    if got is None:
        print("    (skipped: no SPY cache)", end="")
        return
    bars, matrix = got
    total, start = 1638, 150_000
    sl = slice(start, start + total)
    close = np.ascontiguousarray(bars.close[sl])
    gate = np.ascontiguousarray(bars.gate[sl])
    rows = np.ascontiguousarray(matrix[:, sl])
    mult = rmv.xmult(rows, gate == 1)
    buf = np.empty((total, 4), np.float64)
    worst, combos, trades_seen = 0.0, 0, 0
    for a in range(0, rmv.N_VALUES.size, 3):
        row = np.ascontiguousarray(rows[a])
        n = int(rmv.N_VALUES[a])
        for i in (1, 5, 11):
            for j in (2, 7, 13):
                up = rmv.threshold(0.25 * i, mult, n)
                dn = rmv.threshold(0.25 * j, mult, n)
                k = rmv._simulate(row, close, gate, up, dn, COST, buf)
                trades = buf[:k].copy()
                got_row = rmv.metrics(trades)
                want = _metrics_reference(trades)
                for idx, col in enumerate(rmv.METRIC_COLS):
                    b = want[col]
                    if np.isinf(b):
                        assert np.isinf(got_row[idx]), f"{col} at n={n}"
                        continue
                    worst = max(worst, abs(float(got_row[idx]) - b) / max(1.0, abs(b)))
                combos += 1
                trades_seen += k
    assert worst < 1e-9, f"worst relative gap on a real window: {worst}"
    print(f"    ({combos} real combos, {trades_seen} trades, worst gap {worst:.2e})", end="")


def test_unit5_float32_storage_keeps_the_filter_screens() -> None:
    """Unit 6 stores these as float32; SPEC §5's screens must not move when it does.

    SPEC §7 rule 3 permits float32 *storage* -- the danger it names is float32 accumulation,
    which the test above covers. This is the other half: a value computed exactly in float64
    and then rounded to float32 has to land on the same side of `PF < 4`, `eqR2 < 80`,
    `eqR2 <= 50`, `lr < 3`, `lr <= 3` and `nT >= 16` as the float64 value did.
    """
    got = _real_bars()
    if got is None:
        print("    (skipped: no SPY cache)", end="")
        return
    bars, matrix = got
    total, start = 1638, 90_000
    sl = slice(start, start + total)
    close = np.ascontiguousarray(bars.close[sl])
    gate = np.ascontiguousarray(bars.gate[sl])
    rows = np.ascontiguousarray(matrix[:, sl])
    mult = rmv.xmult(rows, gate == 1)
    buf = np.empty((total, 4), np.float64)
    wide = np.empty(rmv.N_METRICS, np.float64)
    narrow = np.empty(rmv.N_METRICS, np.float32)
    scratch = np.empty(total, np.float64)
    screens = (("PF", "<", 4.0), ("PF", "<=", 2.0), ("PF", ">=", 1.0), ("eqR2", "<", 80.0),
               ("eqR2", "<=", 50.0), ("lr", "<", 3.0), ("lr", "<=", 3.0), ("nT", ">=", 16.0))
    flips = {s: 0 for s in screens}
    combos = 0
    closest = np.inf
    for a in range(rmv.N_VALUES.size):
        row = np.ascontiguousarray(rows[a])
        n = int(rmv.N_VALUES[a])
        for i in range(1, 15, 2):
            for j in range(1, 15, 2):
                k = rmv._simulate(row, close, gate, rmv.threshold(0.25 * i, mult, n),
                                  rmv.threshold(0.25 * j, mult, n), COST, buf)
                # `_metrics` writes straight into Unit 6's float32 row -- the same call it
                # makes -- so this measures the real store, not a numpy round trip.
                rmv._metrics(buf[:k], wide, scratch)
                rmv._metrics(buf[:k], narrow, scratch)
                for s in screens:
                    col, op, bound = s
                    idx = rmv.METRIC_COLS.index(col)
                    hi, lo = float(wide[idx]), float(narrow[idx])
                    # Only the continuous columns say anything about margin: `lr` and `nT`
                    # are small integers, exact in float32, and land *on* their bound often.
                    if np.isfinite(hi) and col in ("PF", "eqR2") and hi != bound:
                        closest = min(closest, abs(hi - bound))
                    cmp = {"<": lambda x: x < bound, "<=": lambda x: x <= bound,
                           ">=": lambda x: x >= bound}[op]
                    flips[s] += cmp(hi) != cmp(lo)
                combos += 1
    bad = {f"{c}{o}{b}": v for (c, o, b), v in flips.items() if v}
    assert not bad, f"float32 storage flipped a filter screen: {bad} over {combos} combos"
    # Not "cannot": the flip needs the float64 value within half a float32 ulp of the bound,
    # which is 1.9e-6 at eqR2 = 50. The closest any real combo came says how much room there
    # is; if that ever falls below ~1e-5 this test is one refresh from firing.
    print(f"    ({combos} combos, 8 screens, 0 flips; closest approach to a bound "
          f"{closest:.2e})", end="")


def test_unit5_degenerate_rows_cannot_reach_the_filters() -> None:
    """⚑ Both sentinel decisions, measured against a real 4312-combo grid rather than argued.

    Two findings this pins, either of which would have been a silent wrong selection:

    1. `eqR2 = 0.0` for a row with fewer than two trades passes both `eqR2 < 80` (CL2) and
       `eqR2 <= 50` (CL4). Hundreds of rows per window have `nT < 2`, so CL4's bottom-10
       `mLb` pool would have been mostly rows that never traded. `eqR2 = 100.0` fails both.
    2. `eq2R2` is exactly 100 for every `nT == 3` row -- a quadratic through three points is
       an exact fit -- and `meyers2005` *picks* max `eq2R2`. Its `nT >= 16` screen is what
       stops that pick collapsing onto a three-trade row, so the screen is load-bearing and
       not decoration. Recorded here because a later filter written without it would look
       reasonable and silently select noise.
    """
    got = _real_bars()
    if got is None:
        print("    (skipped: no SPY cache)", end="")
        return
    bars, matrix = got
    total = 1638
    buf = np.empty((total, 4), np.float64)
    scratch = np.empty(total, np.float64)
    idx = {c: i for i, c in enumerate(rmv.METRIC_COLS)}
    report = []
    for start in (60_000, 150_000):
        sl = slice(start, start + total)
        close = np.ascontiguousarray(bars.close[sl])
        gate = np.ascontiguousarray(bars.gate[sl])
        rows = np.ascontiguousarray(matrix[:, sl])
        mult = rmv.xmult(rows, gate == 1)
        out = np.empty((4312, rmv.N_METRICS), np.float64)
        c = 0
        for a in range(rmv.N_VALUES.size):
            row = np.ascontiguousarray(rows[a])
            n = int(rmv.N_VALUES[a])
            for i in range(14):
                for j in range(14):
                    k = rmv._simulate(row, close, gate, rmv.threshold(0.25 * (i + 1), mult, n),
                                      rmv.threshold(0.25 * (j + 1), mult, n), COST, buf)
                    rmv._metrics(buf[:k], out[c], scratch)
                    c += 1
        nt, eq2, eqr2, lr, pf, mlb = (out[:, idx[x]] for x in
                                      ("nT", "eq2R2", "eqR2", "lr", "PF", "mLb"))
        thin = nt < 2
        # (1) no thin row survives any of the three filters' screens.
        assert not ((lr <= 3) & (eqr2 <= 50) & thin).any(), "CL4 admitted a row with nT < 2"
        assert not ((pf < 4) & (lr < 3) & (eqr2 < 80) & thin).any(), "CL2 did"
        assert not ((pf >= 1) & (pf <= 2) & (lr <= 3) & (nt >= 16) & thin).any(), "meyers2005 did"
        # ...and none of them can enter a bottom-k rank on mLb either.
        assert np.isinf(mlb[nt == 0]).all(), "a zero-trade row has a finite mLb"
        # (2) the eq2R2 saturation, and that nT >= 16 is what removes it.
        top = eq2 >= eq2.max() - 1e-9
        assert eq2.max() == 100.0 and top.sum() > 0
        assert (nt[top] == 3).all(), (
            f"rows at max eq2R2 have nT in {sorted(set(nt[top].astype(int)))}, not all 3 -- "
            "the exact-fit explanation for the saturation no longer holds"
        )
        surv = nt >= 16
        assert eq2[surv].max() < 100.0, "meyers2005's nT >= 16 screen no longer removes them"
        report.append((int(thin.sum()), int((nt == 0).sum()), int(top.sum()),
                       float(eq2[surv].max())))
    print("    (" + "; ".join(
        f"{t} rows nT<2 ({z} nT=0) all screened out, {s} rows at eq2R2=100 all nT=3, "
        f"best survivor {b:.1f}" for t, z, s, b in report) + ")", end="")


def test_unit5_buffer_reuse_and_zero_allocation() -> None:
    """Unit 6 calls `_metrics` inside `prange` with per-thread buffers and forbids allocation.

    Two things a single-call test cannot see: whether a dirty `scratch` or a dirty `out`
    changes the answer, and whether the kernel allocates. The allocation half needs
    `NUMBA_NRT_STATS=1` set before numba imports, so it runs in a subprocess.
    """
    import os

    rng = np.random.default_rng(41)
    busy = _random_trades(rng, k=50)
    quiet = _random_trades(rng, k=3)
    scratch = np.empty(200, np.float64)
    out = np.empty(rmv.N_METRICS, np.float64)

    fresh = rmv.metrics(quiet).copy()
    rmv.metrics(busy, out=out, scratch=scratch)          # dirties both buffers
    again = rmv.metrics(quiet, out=out, scratch=scratch)
    assert np.array_equal(fresh, again), "a dirty out/scratch buffer changed the answer"
    assert again is out, "metrics did not return the caller's buffer"

    # A float32 out row -- what Unit 6 actually passes -- must produce the float64 answer
    # rounded, not a float32-accumulated one.
    narrow = np.empty(rmv.N_METRICS, np.float32)
    rmv._metrics(busy, narrow, scratch)
    assert np.allclose(narrow, rmv.metrics(busy), rtol=1e-6, atol=1e-6), (
        "the float32 storage row is not the float64 answer rounded"
    )

    # Measured njit-to-njit through a driver, which is the call Unit 6 makes. Calling
    # `_metrics` from the interpreter instead reports 3 allocations *per call* -- one for
    # each array argument numba has to box at the boundary -- and none of them is the
    # kernel. Measuring the wrong side of that boundary is how this test lies.
    src = (
        "import numpy as np, rmv\n"
        "from numba import njit\n"
        "from numba.core.runtime import nrt\n"
        "@njit(cache=False)\n"
        "def driver(t, out, scr, reps):\n"
        "    for _ in range(reps): rmv._metrics(t, out, scr)\n"
        "rng = np.random.default_rng(0)\n"
        "t = np.zeros((40, 4)); t[:, 1] = 1.0; t[:, 3] = rng.normal(0, 1, 40)\n"
        "out = np.empty(rmv.N_METRICS); scr = np.empty(64)\n"
        "driver(t, out, scr, 1)\n"
        "a = nrt.rtsys.get_allocation_stats().alloc\n"
        "driver(t, out, scr, 500)\n"
        "boxed = nrt.rtsys.get_allocation_stats().alloc - a\n"
        "b = nrt.rtsys.get_allocation_stats().alloc\n"
        "for _ in range(500): rmv._metrics(t, out, scr)\n"
        "print(boxed, nrt.rtsys.get_allocation_stats().alloc - b)\n"
    )
    env = {**os.environ, "NUMBA_NRT_STATS": "1", "PYTHONIOENCODING": "utf-8"}
    proc = subprocess.run([sys.executable, "-c", src], capture_output=True, text=True,
                          cwd=str(Path(__file__).parent), env=env)
    assert proc.returncode == 0, proc.stderr[-2000:]
    inner, outer = (int(x) for x in proc.stdout.strip().splitlines()[-1].split())
    assert inner <= 3, (
        f"{inner} allocations for 500 calls inside njit -- Unit 6's done-when is zero "
        "allocation inside prange, and np.median / ndarray.sort are exactly what breaks it"
    )
    # The interpreter path must still show the boxing, or the driver above was inlined away
    # and `inner` is measuring nothing. A lower bound, not `== 1500`: the exact count is a
    # numba implementation detail (3 per call, one per array argument, in 0.65.1) and
    # pinning it would fail on a numba upgrade with no logic change.
    assert outer >= 500, (
        f"only {outer} allocations over 500 interpreted calls -- the boxing that makes the "
        "njit-to-njit comparison meaningful is gone, so `inner` may be measuring nothing"
    )
    print(f"    ({inner} allocations for 500 njit calls; {outer / 500:.0f}/call from the "
          f"interpreter, all argument boxing)", end="")


def test_unit5_rejects_bad_input() -> None:
    """Each guard is for a caller mistake that would otherwise return a plausible row."""
    good = _flat_trades([1.0, -1.0, 2.0, -0.5, 0.25])

    def rejects(fragment, **kw):
        try:
            rmv.metrics(**kw)
        except ValueError as exc:
            assert fragment in str(exc), f"wrong message for {fragment!r}: {exc}"
            return
        raise AssertionError(f"accepted input that should raise on {fragment!r}")

    rejects("float64[k, 4]", trades=good[:, :3])                     # dropped a column
    rejects("float64[k, 4]", trades=good.astype(np.float32))         # wrong dtype
    rejects("float64[k, 4]", trades=good[:, 3])                      # 1-D, the net column
    rejects("out must be", trades=good, out=np.empty(23))            # 23, not 24
    rejects("out must be", trades=good, out=np.empty(24, np.float32))
    rejects("scratch has", trades=good, scratch=np.empty(4))         # one slot short
    rejects("scratch must be", trades=good, scratch=np.empty(9, np.float32))
    rejects("aliases trades", trades=good, scratch=good[:, 3])       # a view of the input
    # 5 slots for 5 trades is exactly enough, and 0 trades must not demand a buffer at all.
    rmv.metrics(good, scratch=np.empty(5))
    rmv.metrics(np.zeros((0, 4)))


def test_unit5_budget() -> None:
    """Simulation plus metrics, one real window's 4312 combos, against Unit 6's 60 ms.

    Unit 4 measured its own half at ~12 ms serial and Unit 6 has to fit both inside 60 ms
    on one thread before `prange` is allowed to be the reason it passes. The metric pass is
    O(k^2) in two places -- the insertion-sort medians and the Kendall pair loop -- so it is
    the trade count, not the bar count, that decides whether this holds.
    """
    import time

    got = _real_bars()
    if got is None:
        print("    (skipped: no SPY cache)", end="")
        return
    bars, matrix = got
    total = 1638
    buf = np.empty((total, 4), np.float64)
    scratch = np.empty(total, np.float64)
    out = np.empty((4312, rmv.N_METRICS), np.float32)
    k = rmv._simulate(matrix[0, :total].copy(), bars.close[:total].copy(),
                      bars.gate[:total].copy(), 0.01, 0.01, COST, buf)
    rmv._metrics(buf[:k], out[0], scratch)  # JIT

    worst = worst_trades = 0.0
    for start in np.linspace(30_000, len(bars) - total - 1, 4).astype(int):
        sl = slice(start, start + total)
        close = np.ascontiguousarray(bars.close[sl])
        gate = np.ascontiguousarray(bars.gate[sl])
        rows = np.ascontiguousarray(matrix[:, sl])
        mult = rmv.xmult(rows, gate == 1)
        thr = np.array([[rmv.threshold(0.25 * (j + 1), mult, n) for j in range(14)]
                        for n in rmv.N_VALUES])
        trades = c = 0
        begin = time.perf_counter()
        for a in range(rmv.N_VALUES.size):
            row = np.ascontiguousarray(rows[a])
            for i in range(14):
                for j in range(14):
                    k = rmv._simulate(row, close, gate, thr[a, i], thr[a, j], COST, buf)
                    rmv._metrics(buf[:k], out[c], scratch)
                    trades += k
                    c += 1
        elapsed = time.perf_counter() - begin
        if elapsed > worst:
            worst, worst_trades = elapsed, trades
    print(f"    (worst real window {worst * 1000:.0f} ms serial for 4312 simulate+metrics, "
          f"{worst_trades / 4312:.1f} trades/combo)", end="")
    assert worst_trades / 4312 > 10, (
        f"only {worst_trades / 4312:.1f} trades per combo -- this is timing an empty loop"
    )
    assert worst < 0.060, (
        f"{worst * 1000:.0f} ms serial for one window's 4312 simulate+metrics, on one "
        "thread, against Unit 6's 60 ms budget for the whole window"
    )


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
