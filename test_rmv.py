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
