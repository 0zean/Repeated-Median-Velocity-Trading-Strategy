"""Walk-forward window generation and the PWFO driver. PLAN §3 Unit 7, SPEC §4.

One window is a 31-day-inclusive in-sample span ending on a Friday plus the following
Mon-Fri out-of-sample week, stepped 7 days. `run` sweeps Unit 6's 4312-combo grid over
both halves of every window and streams the result to three memmapped `.npy` files.

The look-ahead barrier is **which file a block lands in**, not which columns were
computed: `run_grid` fills all 24 metric columns on every call, so `pwfo_is.npy` gets
`IS_COLS` of the *IS* run and `pwfo_oos.npy` gets `OOS_COLS` of the *OOS* run. Unit 8's
evaluator opens only `pwfo_is.npy` to select and `pwfo_oos.npy` to score.
"""

from __future__ import annotations

import json
import time
from datetime import datetime, timezone
from pathlib import Path
from typing import NamedTuple

import numpy as np

import data
import rmv

OUT_DIR = Path(__file__).parent / "pwfo"

# PLAN §3 Unit 9: the final 6 months are written once and not opened again until the
# project's last action. A window is withheld on its **OOS** start, which also withholds
# its IS half -- IS strictly precedes OOS, so an OOS start inside the tail is the only
# way any of a window's bars can be.
TAIL_START = np.datetime64("2026-03-01")

# SPEC §4: IS is a 30-day *delta*, i.e. 31 days inclusive. [M25 Table 1] p.17 row 1 reads
# 11/12/14-12/12/14, and "30 calendar days ending Friday" taken the obvious way generates
# 10/19-11/17 for the p.25 row instead of 10/18-11/17 -- off by one day on every window.
IS_DAYS = np.timedelta64(30, "D")
OOS_LO = np.timedelta64(3, "D")  # Friday -> the following Monday
OOS_HI = np.timedelta64(7, "D")  # Friday -> the following Friday
WEEK = np.timedelta64(7, "D")

# SPEC §3.2: $0.01/share round-trip slippage + ~$0.017/share SEC/TAF on the sell. The
# second term is charged on notional and so is not constant across a sample where SPY ran
# $180 -> $650; it is read here as $0.017 at SPY $600, the two figures SPEC §3.2 states in
# one sentence.
#
# ⚠ The rate and its anchor are **Unit 9's** to verify against FINRA's TAF schedule and
# the SEC's Section 31 advisories -- SPEC §3.2 already records that folding a per-share
# fee into a notional-scaled one is structurally wrong, and this does not fix that. What
# this unit owns is only *which bars* the price level is read off, which is the half that
# can leak: `cost` enters every IS metric and therefore the filter's choice of row, so it
# inherits `xmult`'s discipline exactly (PLAN Unit 7, from Unit 4).
SLIP = 0.01
SEC_TAF_PER_DOLLAR = 0.017 / 600.0


class Window(NamedTuple):
    """One walk-forward window as bar-index slices. Both halves end on an ungated bar."""

    friday: np.datetime64  # the IS end date, and the window's identity
    is_lo: int
    is_hi: int
    oos_lo: int
    oos_hi: int


def _dow(d: np.ndarray) -> np.ndarray:
    """Day of week, Mon=0. The datetime64[D] epoch 1970-01-01 was a Thursday."""
    return (d.astype(np.int64) + 3) % 7


def windows(bars: data.Bars) -> list[Window]:
    """Every fully-covered week-anchored window over `bars`, in chronological order.

    A window is emitted only when its IS start date is at or after the first bar's date
    and its OOS end Friday is at or before the last bar's -- the partial week at each end
    of the sample is dropped rather than run short. Consecutive Fridays then make the OOS
    weeks tile the timeline exactly once, which is asserted here rather than assumed.

    Both halves are **date** ranges, so both end on the last bar of a session, which is
    ungated by construction (`gate` closes at 15:55 and the last bar opens 15:55). That
    matters: `_simulate` treats the last bar of any slice as the last gated bar of a run
    -- no entry there, and an open position force-closed at the edge -- which is right at
    a session edge and silently wrong at a mid-session cut (PLAN Unit 7, from Unit 4).
    """
    et = data.to_et(bars.ts)
    day = et.normalize().tz_localize(None).values.astype("datetime64[D]")
    first, last = day[0], day[-1]

    # First anchor Friday whose 31-day IS span is fully inside the sample.
    f = first + IS_DAYS
    f += np.timedelta64(int((4 - _dow(f)) % 7), "D")
    fridays = []
    while f + OOS_HI <= last:
        fridays.append(f)
        f = f + WEEK
    if not fridays:
        raise ValueError(f"{len(bars)} bars over {first}..{last} hold no complete window")
    fr = np.array(fridays, dtype="datetime64[D]")

    # `right` on the closing bound: both spans are inclusive of their end date.
    is_lo = np.searchsorted(day, fr - IS_DAYS, "left")
    is_hi = np.searchsorted(day, fr, "right")
    oos_lo = np.searchsorted(day, fr + OOS_LO, "left")
    oos_hi = np.searchsorted(day, fr + OOS_HI, "right")

    if not np.all((is_lo < is_hi) & (oos_lo < oos_hi)):
        bad = fr[(is_lo >= is_hi) | (oos_lo >= oos_hi)]
        raise ValueError(f"empty IS or OOS half at {bad[:5]}")
    # The leakage guard, SPEC §4. Stated on timestamps rather than on the index bounds it
    # is derived from, so it would still fail if the derivation were the thing that broke.
    if not np.all(bars.ts[is_hi - 1] < bars.ts[oos_lo]):
        raise ValueError("IS bars run past the start of their own OOS week")
    if not (np.all(bars.gate[is_hi - 1] == 0) and np.all(bars.gate[oos_hi - 1] == 0)):
        raise ValueError("a window half ends on a gated bar; see this function's docstring")
    # OOS weeks tile: consecutive windows' OOS halves abut exactly, no overlap, no gap.
    if not np.array_equal(oos_lo[1:], oos_hi[:-1]):
        raise ValueError("OOS weeks do not tile the timeline exactly once")

    return [Window(*w) for w in zip(fr, is_lo, is_hi, oos_lo, oos_hi)]


def window_cost(close: np.ndarray) -> float:
    """SPEC §3.2's per-trade cost at one window's own IS price level. See `SLIP` above.

    Unmasked, where `rmv.xmult` requires `gate == 1`. The asymmetry is deliberate and
    measured: an ungated bar is not an RMedV value at all, which is a 22.8% error at N=3,
    but it is a perfectly good price, and masking moves `cost` by at most 5.2e-5 over the
    525 real pre-tail windows -- 0.3% of a quantity whose whole model Unit 9 owns.
    """
    return SLIP + SEC_TAF_PER_DOLLAR * float(np.mean(close, dtype=np.float64))


def run(
    bars: data.Bars,
    matrix: np.ndarray,
    wins: list[Window] | None = None,
    out_dir: Path | str = OUT_DIR,
    progress: int = 0,
) -> list[dict]:
    """Sweep the grid over every window and stream to `pwfo_{is,oos,tail}.npy`.

    `matrix` is `rmv.rmv_all_n(bars.close)` -- all 22 rows, never sliced, which is how the
    `ns[a]` / `rmv_window[a]` mis-pairing `run_grid` cannot check is avoided rather than
    detected (PLAN Unit 7, from Unit 6): rows are only ever taken by *column*, so `ns` is
    always the whole of `rmv.N_VALUES` and there is nothing to keep in step with.

    Returns the window index -- one dict per window, carrying the dates, the bar counts,
    and this window's `xmult` and `cost`. Both are **per window, not per row**, so they
    live here and in the sibling `pwfo_index.json`, not as a 19th column (PLAN §2.1 pins
    `pwfo_is` at 18). Storing them is what lets a replay reproduce the exact thresholds.
    """
    if wins is None:
        wins = windows(bars)
    if not wins:
        raise ValueError("no windows to run")
    if matrix.shape != (rmv.N_VALUES.size, len(bars)):
        raise ValueError(
            f"matrix must be float32[{rmv.N_VALUES.size}, {len(bars)}], got {matrix.shape}"
        )
    out_dir = Path(out_dir)
    out_dir.mkdir(parents=True, exist_ok=True)

    fridays = np.array([w.friday for w in wins], dtype="datetime64[D]")
    # On the OOS **end**: a window is withheld if *any* of its OOS bars is in the tail.
    # Identical to keying on the OOS start while `TAIL_START` falls on a weekend, and
    # still correct the day someone moves it into the middle of a week.
    is_tail = (fridays + OOS_HI) >= TAIL_START
    tail_ns = TAIL_START.astype("datetime64[ns]").astype(np.int64)
    n_combos = rmv.N_VALUES.size * rmv.V_VALUES.size**2
    n_is = int(np.count_nonzero(~is_tail))
    n_tail = int(np.count_nonzero(is_tail))
    if n_is == 0:
        # Otherwise `pwfo_is.npy` is a valid, loadable, empty table and every downstream
        # aggregate over it is a silent nan. A withheld boundary that swallows the whole
        # run is a misconfiguration, not a result.
        raise ValueError(
            f"all {len(wins)} windows are withheld by TAIL_START={TAIL_START}; the "
            "pre-tail table would be empty"
        )

    def _mm(name: str, cols: int, rows: int):
        return np.lib.format.open_memmap(
            out_dir / name, mode="w+", dtype=np.float32, shape=(rows, n_combos, cols)
        )

    mm_is = _mm("pwfo_is.npy", rmv.IS_COLS.stop - rmv.IS_COLS.start, n_is)
    mm_oos = _mm("pwfo_oos.npy", rmv.OOS_COLS.stop - rmv.OOS_COLS.start, n_is)
    # ponytail: no tail file when the run has no tail windows -- a zero-row memmap is a
    # platform question for a file with nothing in it. Every real run has both.
    mm_tail = _mm("pwfo_tail.npy", rmv.N_METRICS, n_tail) if n_tail else None

    # Sized for the longest window and reused across all of them: 1.55 + 0.39 MB against
    # 938 `run_grid` calls, so this is about not churning the allocator, not peak RSS.
    # Oversized buffers are legal, undersized ones raise (PLAN Unit 7, from Unit 6).
    t_max = max(w.is_hi - w.is_lo for w in wins)
    trades = np.empty((rmv.N_VALUES.size, t_max, 4), dtype=np.float64)
    scratch = np.empty((rmv.N_VALUES.size, t_max), dtype=np.float64)
    out_is = np.empty((n_combos, rmv.N_METRICS), dtype=np.float32)
    out_oos = np.empty((n_combos, rmv.N_METRICS), dtype=np.float32)

    index: list[dict] = []
    k_is = k_tail = 0
    try:
        for w, tail in zip(wins, is_tail):
            # Stated on the bars rather than on the dates `is_tail` was derived from, so a
            # mis-derivation cannot classify a tail window into `pwfo_is.npy` and go unseen.
            if not tail and bars.ts[w.oos_hi - 1] >= tail_ns:
                raise AssertionError(f"{w.friday}: a pre-tail window's OOS reaches {TAIL_START}")
            rows = np.ascontiguousarray(matrix[:, w.is_lo : w.is_hi])
            close = bars.close[w.is_lo : w.is_hi]
            gate = bars.gate[w.is_lo : w.is_hi]
            # SPEC §1.2.1: refitted on this window's IS bars, never frozen and never refitted
            # on OOS. The same multiplier scales the thresholds for both calls below.
            mult = rmv.xmult(rows, gate == 1)
            cost = window_cost(close)
            rmv.run_grid(rows, close, gate, rmv.N_VALUES, rmv.V_VALUES, mult, cost,
                         out=out_is, trades=trades, scratch=scratch)
            rmv.run_grid(
                np.ascontiguousarray(matrix[:, w.oos_lo : w.oos_hi]),
                bars.close[w.oos_lo : w.oos_hi],
                bars.gate[w.oos_lo : w.oos_hi],
                rmv.N_VALUES, rmv.V_VALUES, mult, cost,
                out=out_oos, trades=trades, scratch=scratch,
            )

            if tail:
                mm_tail[k_tail, :, rmv.IS_COLS] = out_is[:, rmv.IS_COLS]
                mm_tail[k_tail, :, rmv.OOS_COLS] = out_oos[:, rmv.OOS_COLS]
                tnp = mm_tail[k_tail, :, 0]
                osnp = mm_tail[k_tail, :, rmv.OOS_COLS.start]
                k_tail += 1
            else:
                mm_is[k_is] = out_is[:, rmv.IS_COLS]
                mm_oos[k_is] = out_oos[:, rmv.OOS_COLS]
                tnp, osnp = mm_is[k_is, :, 0], mm_oos[k_is, :, 0]
                k_is += 1

            # ⚑ PLAN Unit 7, from Unit 5. `_metrics` fills all 24 columns on every call and
            # `osnp`/`ont`/`ollt`/`odd` are byte-identical to `tnp`/`nT`/`llt`/`dd`, so
            # writing `out_is[:, OOS_COLS]` above would produce a file of plausible OOS metrics
            # that are really IS metrics -- and the leakage guard, the byte-identical re-run
            # and the non-empty table would all still pass. PLAN §2.1's "structurally
            # impossible" is weaker than it reads, so this is an assert on the production path.
            #
            # Read back off the **files**, not off `out_is`/`out_oos`: comparing the two
            # in-memory tables would catch the two halves being the same run and miss the
            # other half of the same defect, a correct pair of runs written to the wrong file.
            if not np.any(osnp != tnp):
                raise AssertionError(
                    f"{w.friday}: every stored OOS osnp equals its IS tnp -- the OOS block is "
                    "the IS run's, or the two halves produced identical trades"
                )

            index.append({
                "friday": str(w.friday),
                "is_start": str(w.friday - IS_DAYS), "is_end": str(w.friday),
                "oos_start": str(w.friday + OOS_LO), "oos_end": str(w.friday + OOS_HI),
                "is_bars": int(w.is_hi - w.is_lo), "oos_bars": int(w.oos_hi - w.oos_lo),
                "xmult": mult, "cost": cost,
                "file": "tail" if tail else "is", "row": k_tail - 1 if tail else k_is - 1,
            })
            if progress and len(index) % progress == 0:
                print(f"    {len(index)}/{len(wins)} windows", flush=True)

    finally:
        # ponytail: `_mmap` is private and numpy offers no public close. Without it the
        # handles live until the frame is collected, and on Windows that stops the caller
        # deleting the directory -- so an exception raised in the loop above surfaces as a
        # `NotADirectoryError` from someone's `TemporaryDirectory` and hides its own cause.
        for mm in (mm_is, mm_oos, mm_tail):
            if mm is not None:
                mm.flush()
                mm._mmap.close()

    (out_dir / "pwfo_index.json").write_text(json.dumps({
        "is_cols": list(rmv.METRIC_COLS[rmv.IS_COLS]),
        "oos_cols": list(rmv.METRIC_COLS[rmv.OOS_COLS]),
        "tail_cols": list(rmv.METRIC_COLS),
        "n_combos": n_combos,
        "tail_start": str(TAIL_START),
        "windows": index,
    }, indent=1), encoding="utf-8")
    return index


if __name__ == "__main__":
    t0 = time.perf_counter()
    bars = data.load_bars(
        "SPY",
        datetime(2016, 1, 1, tzinfo=timezone.utc),
        datetime(2030, 1, 1, tzinfo=timezone.utc),
        refresh=False,
    )
    matrix = rmv.rmv_all_n(bars.close)
    wins = windows(bars)
    print(f"{len(bars)} bars, {len(wins)} windows, RMV in {time.perf_counter() - t0:.1f} s")
    t1 = time.perf_counter()
    idx = run(bars, matrix, wins, progress=100)
    # Nothing about the tail beyond its count: the file is written and not opened again
    # until Unit 9's final step (PLAN §3 Unit 9).
    pre = [w for w in idx if w["file"] == "is"]
    print(f"PWFO {time.perf_counter() - t1:.1f} s -- {len(pre)} pre-tail windows, "
          f"{len(idx) - len(pre)} withheld")
    print(f"  xmult {min(w['xmult'] for w in pre):.3f}..{max(w['xmult'] for w in pre):.3f}, "
          f"cost {min(w['cost'] for w in pre):.4f}..{max(w['cost'] for w in pre):.4f}")
