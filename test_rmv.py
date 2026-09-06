"""Test suite. Plain asserts, no framework. Run: python test_rmv.py

One test function per unit of work (see PLAN.md §3).
"""

from __future__ import annotations

import re
import shutil
import subprocess
import sys
import tempfile
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
