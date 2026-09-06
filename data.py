"""Alpaca 5-minute bar loading, caching, and the trading gate. See SPEC.md §3.

Everything below this module sees structure-of-arrays (int64/float32/int8), never a
DataFrame. pandas is used here and only here, for timezone arithmetic at the boundary.

The bar series is the measured-contiguous window 08:00-15:55 ET (SPEC §3.1) -- *not* RTH,
and not the full 04:00-20:00 extended-hours range, which has 2-15% missing buckets. RMedV
is computed on every bar in the window; only *trading* is gated (SPEC §2).
"""

from __future__ import annotations

import json
import os
from datetime import date, datetime, timezone
from pathlib import Path
from typing import NamedTuple

import numpy as np
import pandas as pd
from alpaca.data.enums import Adjustment, DataFeed
from alpaca.data.historical import StockHistoricalDataClient
from alpaca.data.requests import StockBarsRequest
from alpaca.data.timeframe import TimeFrame, TimeFrameUnit

ET = "America/New_York"
BAR_NS = 5 * 60 * 1_000_000_000  # one 5-minute bar, in epoch nanoseconds
CACHE_DIR = Path(__file__).parent / "cache"

# SPEC §3.1: the session window RMedV is computed on. Measured, not assumed -- 2024 bucket
# completeness is ~100% from 08:00 to 16:00 ET and falls to 85-98% outside it. A missing
# bucket silently rescales the slope, so the series stops where it stops being contiguous.
#
# 08:00 also sits MAX_N bars before the 10:00 gate, so the first tradeable bar's lookback
# never reaches across the overnight gap.
SESSION_START_MIN = 8 * 60  # 08:00
SESSION_END_MIN = 16 * 60  # 16:00 exclusive -> last bar opens 15:55

# SPEC §2, ET wall clock (DST-aware, never a fixed UTC offset).
GATE_OPEN_MIN = 10 * 60  # 10:00 -- first trade of day
EXIT_BEFORE_CLOSE_MIN = 5  # flat 5 minutes before the close
REGULAR_CLOSE_MIN = 16 * 60  # 16:00, when the calendar says nothing else
GATE_CLOSE_MIN = REGULAR_CLOSE_MIN - EXIT_BEFORE_CLOSE_MIN  # 15:55 on a regular day

# SPEC §3.3: largest lookback in the grid. A window spanning a data gap is not a valid
# RMedV, so the gate stays shut until max_n clean bars have accumulated after one.
MAX_N = 24

# SPEC §3.2: SIP. IEX is not usable -- it supplies almost no pre-market bars and carries
# ~2.5c of noise on every print (measured, Unit 1).
DEFAULT_FEED = "sip"

# Refetched on every refresh and compared against the cache. This is what makes "a cached
# round-trip is bit-identical to a fresh fetch" a continuous guarantee rather than a
# one-off, and it is how a retroactive price adjustment would get caught.
OVERLAP_BARS = 200


class Bars(NamedTuple):
    """Structure-of-arrays. Parallel, equal-length, index-aligned.

    `gate` is the only session signal downstream needs: a 1->0 edge means "flatten", and
    that is the correct response whether the cause is the 15:55 exit, an early close, or
    a post-gap blackout. Deliberately no separate day/session array -- see SPEC §3.2.
    """

    ts: np.ndarray  # int64[T]   epoch nanoseconds, UTC, strictly increasing
    close: np.ndarray  # float32[T]
    gate: np.ndarray  # int8[T]    1 = a position may be opened or held on this bar

    def __len__(self) -> int:
        return self.ts.size


# --------------------------------------------------------------------------- pure


def validate(ts: np.ndarray, close: np.ndarray) -> None:
    """Invariants every consumer downstream is entitled to assume.

    Raises ValueError, not assert: `python -O` strips asserts, and this is the data
    contract for a system that places orders.
    """

    def check(ok: bool, msg: str) -> None:
        if not ok:
            raise ValueError(f"bar series invalid: {msg}")

    check(ts.dtype == np.int64, f"ts must be int64, got {ts.dtype}")
    check(close.dtype == np.float32, f"close must be float32, got {close.dtype}")
    check(ts.size == close.size, f"length mismatch: {ts.size} vs {close.size}")
    if ts.size == 0:
        return
    check(bool(np.all(np.diff(ts) > 0)), "timestamps not strictly increasing")
    # 5 minutes divides the hour and the epoch starts on the hour, so every genuine bar
    # open lands on an exact multiple. Catches misaligned or resampled data.
    check(bool(np.all(ts % BAR_NS == 0)), "bars not aligned to the 5-minute grid")
    check(bool(np.all(np.isfinite(close))), "non-finite close price")
    check(bool(np.all(close > 0)), "non-positive close price")


def to_et(ts: np.ndarray) -> pd.DatetimeIndex:
    """Epoch-ns UTC -> ET wall clock. DST-aware: 10:00 ET is 15:00 UTC in winter and
    14:00 in summer, so every gate decision must be made here, never on a fixed offset."""
    return pd.to_datetime(ts, unit="ns", utc=True).tz_convert(ET)


def session_mask(
    ts: np.ndarray,
    start_min: int = SESSION_START_MIN,
    end_min: int = SESSION_END_MIN,
) -> np.ndarray:
    """Bars inside the contiguous session window (SPEC §3.1).

    Applied on load, not on fetch: the cache holds everything the API returned, so the
    window can be changed without refetching a decade of bars. Purely time-based -- early
    closes are handled by the gate, not by reshaping the series.
    """
    if ts.size == 0:
        return np.empty(0, bool)
    minute_of_day = _minute_of_day(to_et(ts))
    return np.asarray((minute_of_day >= start_min) & (minute_of_day < end_min))


def _minute_of_day(et: pd.DatetimeIndex) -> np.ndarray:
    return np.asarray(et.hour * 60 + et.minute, dtype=np.int32)


def build_gate(
    ts: np.ndarray,
    max_n: int = MAX_N,
    close_min: np.ndarray | None = None,
) -> np.ndarray:
    """1 where a position may be opened or held. Derived from ts, never stored.

    Three rules:
      * the 10:00 ET first-trade gate (SPEC §2);
      * flat EXIT_BEFORE_CLOSE_MIN before *that session's* close -- 15:55 on a regular
        day, 12:55 on a 13:00 early close. Pass `close_min` (per-bar, minute-of-day) from
        the exchange calendar; without it every day is assumed to close at 16:00, which
        is wrong on the 21 early closes in the sample and is why load_bars always
        supplies it;
      * a max_n-bar blackout after any gap in the series, because an RMedV window
        spanning a gap measures the gap rather than the trend (SPEC §3.1).
    """
    if ts.size == 0:
        return np.empty(0, np.int8)

    minute_of_day = _minute_of_day(to_et(ts))
    if close_min is None:
        cut = np.full(ts.size, GATE_CLOSE_MIN, dtype=np.int32)
    else:
        cut = np.minimum(
            np.asarray(close_min, dtype=np.int32) - EXIT_BEFORE_CLOSE_MIN, GATE_CLOSE_MIN
        )
    gate = ((minute_of_day >= GATE_OPEN_MIN) & (minute_of_day < cut)).astype(np.int8)

    # The blackout runs to i+max_n inclusive, one bar longer than "max_n clean bars".
    # That extra bar is load-bearing: SPEC §2's crossing rule reads RMedV[t-1], so the
    # *previous* bar's own N-bar window must also be gap-free before a signal is valid.
    # ponytail: loop over gaps, not bars -- a few thousand slice assignments, milliseconds.
    for i in np.flatnonzero(np.diff(ts) > BAR_NS):
        gate[i + 1 : i + 1 + max_n] = 0

    # The start of the array is a gap too -- there is simply no earlier bar to diff
    # against. Without this, a slice beginning mid-session gates bars whose RMedV is still
    # warmup: measured 275 such bars across the n grid on an 11:00 ET start. Harmless on a
    # normal 08:00 load, where these bars are outside the trading window anyway.
    gate[:max_n] = 0

    return gate


def session_profile(ts: np.ndarray) -> np.ndarray:
    """Fraction of sessions having a bar in each of the day's 288 five-minute buckets.

    A missing bucket silently changes the x-spacing that RMedV's slope is measured
    against, so this is what turns SPEC §3.1's contiguity claim into a measurement.
    """
    if ts.size == 0:
        return np.zeros(288, np.float64)
    et = to_et(ts)
    n_sessions = et.normalize().nunique()
    counts = np.bincount(_minute_of_day(et) // 5, minlength=288)
    return counts / n_sessions


def overnight_gap_stats(ts: np.ndarray, close: np.ndarray) -> dict[str, float]:
    """Size of the price jump across each break in the series, in dollars, against the
    size of an ordinary bar move."""
    empty = {"n": 0, "median_jump": 0.0, "p95_jump": 0.0,
             "median_bar_move": 0.0, "jump_over_bar": 0.0}
    if ts.size < 2:
        return empty
    breaks = np.flatnonzero(np.diff(ts) > BAR_NS)
    if breaks.size == 0:
        return empty

    prices = close.astype(np.float64)
    jumps = np.abs(prices[breaks + 1] - prices[breaks])
    moves = np.abs(np.diff(prices))
    contiguous = np.delete(moves, breaks)
    median_move = float(np.median(contiguous)) if contiguous.size else 0.0
    return {
        "n": int(breaks.size),
        "median_jump": float(np.median(jumps)),
        "p95_jump": float(np.percentile(jumps, 95)),
        "median_bar_move": median_move,
        "jump_over_bar": float(np.median(jumps) / median_move) if median_move else 0.0,
    }


# ------------------------------------------------------------------------ network


def load_dotenv(path: Path | None = None) -> None:
    """Read .env into the environment if present. Real environment variables win.

    ponytail: six lines of stdlib instead of a python-dotenv dependency. Only the keys
    this project uses are honoured, so a stray line in .env cannot alter the environment.
    """
    path = path or Path(__file__).parent / ".env"
    if not path.exists():
        return
    wanted = {"API_KEY", "SECRET_KEY", "APCA_API_BASE_URL"}
    for line in path.read_text(encoding="utf-8").splitlines():
        line = line.strip()
        if not line or line.startswith("#") or "=" not in line:
            continue
        key, _, value = line.partition("=")
        if key.strip() in wanted:
            os.environ.setdefault(key.strip(), value.strip().strip("\"'"))


def _credentials() -> tuple[str, str]:
    load_dotenv()
    key, secret = os.environ.get("API_KEY"), os.environ.get("SECRET_KEY")
    if not key or not secret:
        raise RuntimeError(
            "Alpaca credentials missing. Export API_KEY and SECRET_KEY "
            "(see .env.example and SPEC.md §8)."
        )
    return key, secret


def _client() -> StockHistoricalDataClient:
    return StockHistoricalDataClient(*_credentials())


def fetch(
    symbol: str,
    start: datetime,
    end: datetime,
    feed: str = DEFAULT_FEED,
    client: StockHistoricalDataClient | None = None,
) -> tuple[np.ndarray, np.ndarray]:
    """Network. 5-minute bars as (ts epoch-ns UTC, close float32), unfiltered.

    adjustment=SPLIT, never ALL: dividend adjustment retroactively rewrites every prior
    bar, which would make the cache mutable and stop the backtest matching what live sees
    (SPEC §3.2). SPY has not split since 2005, so over an Alpaca-era sample this is a no-op.
    """
    request = StockBarsRequest(
        symbol_or_symbols=symbol,
        timeframe=TimeFrame(5, TimeFrameUnit.Minute),
        start=start,
        end=end,
        adjustment=Adjustment.SPLIT,
        feed=DataFeed(feed),
    )
    df = (client or _client()).get_stock_bars(request).df
    if df.empty:
        return np.empty(0, np.int64), np.empty(0, np.float32)

    close_series = df["close"]
    if "symbol" in getattr(close_series.index, "names", []):
        close_series = close_series.droplevel("symbol")
    close_series = close_series.sort_index()

    index = close_series.index.tz_convert("UTC")
    ts = index.to_numpy(dtype="datetime64[ns]").astype(np.int64)
    close = close_series.to_numpy(dtype=np.float32)
    validate(ts, close)
    return ts, close


def fetch_calendar(start: date, end: date) -> dict[str, int]:
    """ET date -> that session's close, as minute-of-day. The authoritative early-close
    source: NYSE closes at 13:00 on ~2 days a year and SPY keeps printing afterwards, so
    without this the gate would trade three hours of thin post-close prints."""
    from alpaca.trading.client import TradingClient
    from alpaca.trading.requests import GetCalendarRequest

    key, secret = _credentials()
    sessions = TradingClient(key, secret, paper=True).get_calendar(
        GetCalendarRequest(start=start, end=end)
    )
    return {str(s.date): s.close.hour * 60 + s.close.minute for s in sessions}


def load_calendar(start: date, end: date, refresh: bool = True) -> dict[str, int]:
    """Cached exchange calendar. Shared across symbols -- it is the NYSE calendar."""
    path = CACHE_DIR / "nyse_calendar.json"
    cached: dict[str, int] = {}
    if path.exists():
        cached = json.loads(path.read_text(encoding="utf-8"))

    covered = cached and min(cached) <= str(start) and max(cached) >= str(end)
    if refresh and not covered:
        cached.update(fetch_calendar(start, end))
        CACHE_DIR.mkdir(exist_ok=True)
        _atomic_write(path, json.dumps(cached, indent=0, sort_keys=True))
    return cached


# -------------------------------------------------------------------------- cache


def _atomic_write(path: Path, text: str) -> None:
    """Write via a temporary file and rename. A crash mid-write must not destroy the
    cache this module exists to protect."""
    tmp = path.with_suffix(path.suffix + ".tmp")
    tmp.write_text(text, encoding="utf-8")
    os.replace(tmp, path)


def _cache_paths(symbol: str, feed: str) -> tuple[Path, Path]:
    stem = CACHE_DIR / f"{symbol}_5min_{feed}"
    return stem.with_suffix(".npz"), stem.with_suffix(".json")


def _write_cache(symbol: str, feed: str, ts: np.ndarray, close: np.ndarray) -> None:
    validate(ts, close)  # never persist a series that would fail the contract
    CACHE_DIR.mkdir(exist_ok=True)
    npz, meta = _cache_paths(symbol, feed)

    tmp = npz.with_suffix(".npz.tmp")
    with tmp.open("wb") as fh:
        np.savez(fh, ts=ts, close=close)
    os.replace(tmp, npz)

    _atomic_write(
        meta,
        json.dumps(
            {
                "symbol": symbol,
                "feed": feed,
                "adjustment": "split",
                "timeframe": "5Min",
                "bars": int(ts.size),
                "first": str(to_et(ts[:1])[0]) if ts.size else None,
                "last": str(to_et(ts[-1:])[0]) if ts.size else None,
                "written": datetime.now(timezone.utc).isoformat(timespec="seconds"),
            },
            indent=2,
        ),
    )


def _read_cache(symbol: str, feed: str) -> tuple[np.ndarray, np.ndarray] | None:
    npz, meta_path = _cache_paths(symbol, feed)
    if not (npz.exists() and meta_path.exists()):
        return None
    meta = json.loads(meta_path.read_text(encoding="utf-8"))
    for field, expected in (("symbol", symbol), ("feed", feed), ("adjustment", "split")):
        if meta.get(field) != expected:
            raise RuntimeError(
                f"cache {npz.name} was written with {field}={meta.get(field)!r}, "
                f"expected {expected!r}. Delete it and refetch."
            )
    with np.load(npz) as d:
        return d["ts"].astype(np.int64), d["close"].astype(np.float32)


def load_bars(
    symbol: str = "SPY",
    start: datetime | None = None,
    end: datetime | None = None,
    feed: str = DEFAULT_FEED,
    max_n: int = MAX_N,
    refresh: bool = True,
    session: bool = True,
    calendar: bool = True,
    client: StockHistoricalDataClient | None = None,
) -> Bars:
    """Cached 5-minute bars, sliced to [start, end] and to the session window.

    Append-only. An existing cache is extended from its last bar forward, and the last
    OVERLAP_BARS are refetched and compared to prove the history did not move underneath
    us; any mismatch raises rather than blending two price series. The cache is never
    narrowed by a narrow request.

    session=False returns everything the API gave (04:00-20:00), for measurement only --
    that range is not contiguous and is not a valid RMedV input (SPEC §3.1).
    calendar=False assumes every session closes at 16:00, which is wrong on early closes.
    """
    cached = _read_cache(symbol, feed)
    ts, close = cached if cached is not None else (np.empty(0, np.int64), np.empty(0, np.float32))

    # refresh=True with no end silently fetched nothing. "Refresh" means "bring me current".
    if refresh and end is None:
        end = datetime.now(timezone.utc)

    # Only reach for the network when the request actually extends past the cache.
    # Without this, any historical slice asks Alpaca for end < start and gets an HTTP 400.
    wants_new = end is not None and (ts.size == 0 or _to_ns(end) > int(ts[-1]))
    if refresh and wants_new:
        if ts.size:
            overlap_from = ts[max(0, ts.size - OVERLAP_BARS)]
            new_ts, new_close = fetch(symbol, _as_utc(overlap_from), end, feed=feed, client=client)
            ts, close = _splice(ts, close, new_ts, new_close)
        else:
            if start is None:
                raise ValueError("no cache for this symbol/feed, so start is required")
            ts, close = fetch(symbol, start, end, feed=feed, client=client)
        _write_cache(symbol, feed, ts, close)

    if start is not None:
        ts, close = _slice_from(ts, close, start)
    if end is not None:
        ts, close = _slice_to(ts, close, end)

    if session:
        keep = session_mask(ts)
        ts, close = ts[keep], close[keep]

    validate(ts, close)

    close_min = None
    if calendar and ts.size:
        et = to_et(ts)
        days = et.strftime("%Y-%m-%d")
        cal = load_calendar(et[0].date(), et[-1].date(), refresh=refresh)
        # A bar on a date the calendar does not list is not a tradeable session bar.
        close_min = np.array([cal.get(d, 0) for d in days], dtype=np.int32)

    # Gate after the session filter: the filter is what creates the session boundaries
    # the gap-blackout rule keys off.
    return Bars(ts=ts, close=close, gate=build_gate(ts, max_n=max_n, close_min=close_min))


def _splice(
    ts: np.ndarray, close: np.ndarray, new_ts: np.ndarray, new_close: np.ndarray
) -> tuple[np.ndarray, np.ndarray]:
    """Union-merge new bars into the cache, verifying the overlap is unchanged.

    A union rather than an append, so bars before the cache's start and bars filling an
    interior hole are both kept instead of silently dropped.
    """
    if new_ts.size == 0:
        return ts, close
    if ts.size == 0:
        return new_ts, new_close

    shared, old_i, new_i = np.intersect1d(ts, new_ts, assume_unique=True, return_indices=True)
    if shared.size and not np.array_equal(close[old_i], new_close[new_i]):
        bad = int(np.argmax(close[old_i] != new_close[new_i]))
        raise RuntimeError(
            f"cached history changed underneath us: {shared.size} overlapping bars, "
            f"first mismatch at {to_et(shared[bad : bad + 1])[0]} "
            f"cached={close[old_i][bad]} fetched={new_close[new_i][bad]}. "
            "adjustment=split should make this impossible -- investigate before refetching."
        )

    fresh = ~np.isin(new_ts, ts, assume_unique=True)
    merged_ts = np.concatenate([ts, new_ts[fresh]])
    merged_close = np.concatenate([close, new_close[fresh]])
    order = np.argsort(merged_ts, kind="stable")
    return merged_ts[order], merged_close[order]


def _as_utc(epoch_ns: int) -> datetime:
    return datetime.fromtimestamp(int(epoch_ns) / 1e9, tz=timezone.utc)


def _to_ns(when: datetime) -> int:
    if when.tzinfo is None:
        when = when.replace(tzinfo=timezone.utc)
    return int(pd.Timestamp(when).tz_convert("UTC").value)


def _slice_from(ts, close, start):
    i = int(np.searchsorted(ts, _to_ns(start), "left"))
    return ts[i:], close[i:]


def _slice_to(ts, close, end):
    i = int(np.searchsorted(ts, _to_ns(end), "right"))
    return ts[:i], close[:i]


# --------------------------------------------------------------------- feed gate


def compare_feeds(
    symbol: str,
    start: datetime,
    end: datetime,
    session_only: bool = True,
    client: StockHistoricalDataClient | None = None,
) -> dict[str, float]:
    """SPEC §3.2 / PLAN §8-D: is IEX tradeable, or does live need paid SIP?

    Restricted to the session window by default -- comparing over 04:00-20:00 counts
    buckets the strategy never uses and overstates the coverage gap. The two numbers that
    decide it are pre-market coverage (the 08:00-10:00 warmup §3.1 exists for) and
    per-bar price noise, since the repeated median resists outliers but not noise on
    every point.
    """
    sip_ts, sip_close = fetch(symbol, start, end, feed="sip", client=client)
    iex_ts, iex_close = fetch(symbol, start, end, feed="iex", client=client)
    if session_only:
        s_keep, i_keep = session_mask(sip_ts), session_mask(iex_ts)
        sip_ts, sip_close = sip_ts[s_keep], sip_close[s_keep]
        iex_ts, iex_close = iex_ts[i_keep], iex_close[i_keep]

    shared, s_i, i_i = np.intersect1d(sip_ts, iex_ts, assume_unique=True, return_indices=True)
    diff_cents = np.abs(sip_close[s_i].astype(np.float64) - iex_close[i_i].astype(np.float64)) * 100

    # Consecutive-bar move on the *full* SIP series -- differencing shared-only closes
    # would span the pre-market holes and overstate a "bar" move.
    sip_moves = np.abs(np.diff(sip_close.astype(np.float64))) * 100
    sip_moves = sip_moves[np.diff(sip_ts) == BAR_NS]

    warmup = _minute_of_day(to_et(iex_ts)) < GATE_OPEN_MIN
    sip_warmup = _minute_of_day(to_et(sip_ts)) < GATE_OPEN_MIN

    return {
        "sip_bars": int(sip_ts.size),
        "iex_bars": int(iex_ts.size),
        "shared_bars": int(shared.size),
        "iex_coverage_pct": float(shared.size / sip_ts.size * 100) if sip_ts.size else 0.0,
        "sip_warmup_bars": int(sip_warmup.sum()),
        "iex_warmup_bars": int(warmup.sum()),
        "median_diff_cents": float(np.median(diff_cents)) if diff_cents.size else 0.0,
        "p95_diff_cents": float(np.percentile(diff_cents, 95)) if diff_cents.size else 0.0,
        "max_diff_cents": float(diff_cents.max()) if diff_cents.size else 0.0,
        "pct_bars_over_1c": float((diff_cents >= 1.0).mean() * 100) if diff_cents.size else 0.0,
        "median_bar_move_cents": float(np.median(sip_moves)) if sip_moves.size else 0.0,
        "noise_over_signal": (
            float(np.median(diff_cents) / np.median(sip_moves))
            if diff_cents.size and sip_moves.size
            else 0.0
        ),
    }
