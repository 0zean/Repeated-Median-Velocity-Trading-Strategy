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
import math
import operator
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


# ------------------------------------------------------------- Filters (PLAN §3 Unit 8)

# SPEC §5's three baselines, as data. `screens` are ANDed; `bottom` narrows to the k
# *smallest* by one metric; `pick` takes the extreme of another. [M25 p.8]: "b10mLb means
# the bottom or minimum 10 mLb rows". A metric written `|x|` is `abs(col x)`
# -- SPEC §9-E's magnitude reading of a column stored signed (§6.6), which is the only
# transform any of this needs and the reason there is no expression language here.
#
# `1 <= PF <= 2` is two screens because that is what it is; SPEC §9-G records the inclusive
# reading as a choice, not a transcription.
#
# ⚑ `bottom` has no direction knob. PLAN §3 Unit 8 wrote `rank: (metric, direction, top_k)`
# and all three baselines rank one way, so the other branch was dead code -- and SPEC §6.6
# measured what it would do if used: a **top**-k on `mLb` puts every no-loser row, sentinel
# `+inf` and all, at the head of a pool it can then never be displaced from, and a filter
# picking max `eqR2` selects an `nT <= 2` row in 24 of 24 real windows. The sanctioned
# directions are the only ones spellable here. Unit 10 reopens this deliberately or not at
# all.
FILTERS: dict[str, dict] = {
    # [M05 p.6]. No rank -- the screens are the pool and the pick is over all of it.
    "meyers2005": {
        "screens": [("PF", ">=", 1.0), ("PF", "<=", 2.0), ("lr", "<=", 3.0),
                    ("nT", ">=", 16.0)],
        "pick": ("eq2R2", "max"),
    },
    # [M25 p.11], `b50mLb|pf<4|lr<3r2<80-mLTr`.
    "CL2": {
        "screens": [("PF", "<", 4.0), ("lr", "<", 3.0), ("eqR2", "<", 80.0)],
        "bottom": ("mLb", 50),
        "pick": ("mLTr", "min"),
    },
    # [M25 p.8], `b10mLb|lr<=3r2<=50-mLTr`. SPEC §9-J: this is the paper's chosen filter.
    "CL4": {
        "screens": [("lr", "<=", 3.0), ("eqR2", "<=", 50.0)],
        "bottom": ("mLb", 10),
        "pick": ("mLTr", "min"),
    },
}

OPS = {"<": operator.lt, "<=": operator.le, ">": operator.gt, ">=": operator.ge}

# The two columns stored as 100*R^2 (SPEC §6.6). Only these move under §9-D's `|r|`
# reading. ⚑ `eq2R2` is inert today and deliberately kept: no baseline *screens* it --
# `meyers2005` only picks on it, and a pick is not transformed -- so dropping it changes
# nothing this unit can measure. It is the scale that belongs here, not the usage: the
# day a filter screens `eq2R2`, its threshold has to move by the same rule or §9-D
# quietly stops applying to half the columns it names.
R2_COLS = ("eqR2", "eq2R2")

# scipy.stats.norm.ppf(0.98). PLAN §2.3 keeps scipy out of everything but the test oracle,
# and `test_unit8_aggregates_match_a_hand_computation` pins this against it.
Z98 = 2.0537489106318225


def _base(metric: str) -> str:
    """`|mLTr|` -> `mLTr`; anything else unchanged."""
    return metric[1:-1] if metric.startswith("|") else metric


def _col(cols: dict, metric: str) -> np.ndarray:
    return np.abs(cols[_base(metric)]) if metric.startswith("|") else cols[metric]


def _metrics(filt: dict):
    """Every metric name a filter reads, screens through pick."""
    return ([m for m, _, _ in filt["screens"]]
            + ([filt["bottom"][0]] if "bottom" in filt else []) + [filt["pick"][0]])


def _r_reading(filt: dict) -> dict:
    """SPEC §9-D's `|r|` reading of an `r2` screen: move the threshold, not the data.

    With `eqR2 = 100*R^2`, reading a literal `80` as `100*|r|` means `R^2 < 0.64`, i.e.
    `eqR2 < 64` -- exact in float32, no `sqrt`, no second column. Derived here rather than
    transcribed so the two readings cannot drift apart. Screens on nothing in `R2_COLS`
    come back unchanged, which is how `meyers2005` collapses to one variant.
    """
    return {**filt, "screens": [(m, o, v * v / 100.0 if m in R2_COLS else v)
                                for m, o, v in filt["screens"]]}


def _magnitude(filt: dict) -> dict:
    """SPEC §9-E's other convention: "smallest `mLTr`" as shallowest, not deepest.

    `mLTr` is stored negative (§6.6), so `min mLTr` as stored selects the *deepest* median
    loss -- the opposite of [M25 p.8]'s stated intent. `min |mLTr|` is the intent reading.
    The reverse derivation does not exist, which is why the sign is stored.
    """
    m, direction = filt["pick"]
    return filt if m != "mLTr" else {**filt, "pick": (f"|{m}|", direction)}


def variants(filters: dict | None = None) -> dict[str, dict]:
    """Expand SPEC §9-D and §9-E's two open ambiguities. PLAN §1.5 says run both, both ways.

    Nine filters out of three, not twelve: a variant is dropped when the transform is a
    no-op on that filter. `meyers2005` screens no `r2` column and picks `eq2R2`, so both
    ambiguities collapse for it -- which is §9-D's "doubly harmless", derived rather than
    asserted. ⚑ All nine are OOS-touching looks and count in PLAN §Unit 9's multiplier.
    """
    out: dict[str, dict] = {}
    for name, f in (FILTERS if filters is None else filters).items():
        seen: list[dict] = []
        for suffix, g in (("", f), (" r", _r_reading(f)), (" |mLTr|", _magnitude(f)),
                          (" r |mLTr|", _magnitude(_r_reading(f)))):
            if g not in seen:
                seen.append(g)
                out[name + suffix] = g
    return out


def decode(c: int) -> tuple[int, float, float]:
    """Combo index -> `(n, vup, vdn)`. SPEC §3.3, the only record of the mapping."""
    a, r = divmod(int(c), rmv.V_VALUES.size**2)
    i, j = divmod(r, rmv.V_VALUES.size)
    return int(rmv.N_VALUES[a]), float(rmv.V_VALUES[i]), float(rmv.V_VALUES[j])


def load_tables(names: list[str], out_dir: Path | str = OUT_DIR):
    """Hoist the named IS columns and all six OOS columns. Returns `(cols, oos, wins)`.

    `cols` maps each name to `float32[windows, combos]`, `oos` is `float32[windows, 6]`-
    indexable as `oos[window, combo]`, and `wins` is the pre-tail slice of the on-disk
    index in row order. PLAN §2.1's second option: a row-major scan reads one metric with
    a 96-byte stride, so the eight columns a filter needs are lifted in one pass -- 63 MB
    for the IS block, 52 MB for OOS.

    Column names, the window count and the row order all come from `pwfo_index.json`
    rather than from `rmv` -- the file on disk is the contract Units 8, 9 and 11 read, and
    a table written by an older column layout has to fail here rather than silently score
    the wrong column. ⚑ `pwfo_tail.npy` is not opened; the withheld set is Unit 9's last
    action, and the index's tail rows are dropped by `file == "is"`.
    """
    out_dir = Path(out_dir)
    index = json.loads((out_dir / "pwfo_index.json").read_text(encoding="utf-8"))
    wins = [w for w in index["windows"] if w["file"] == "is"]
    if [w["row"] for w in wins] != list(range(len(wins))):
        raise ValueError("pwfo_index.json's pre-tail rows are not 0..n-1 in order")
    is_names, oos_names = index["is_cols"], index["oos_cols"]
    missing = [n for n in names if n not in is_names]
    if missing:
        raise KeyError(f"{missing} are not IS columns; pwfo_is.npy holds {is_names}")

    shape = (len(wins), index["n_combos"])
    mm = np.load(out_dir / "pwfo_is.npy", mmap_mode="r")
    try:
        if mm.shape != (*shape, len(is_names)):
            raise ValueError(f"pwfo_is.npy is {mm.shape}, index says {(*shape, len(is_names))}")
        block = np.asarray(mm[:, :, [is_names.index(n) for n in names]])
    finally:
        mm._mmap.close()  # Unit 7's Windows handle leak, same cause and same fix
    # Every sentinel in SPEC §6.6 is a finite value or +inf; a nan would pass every screen's
    # negation silently and win an argmin outright, so it is rejected at the file boundary.
    if np.isnan(block).any():
        raise ValueError("pwfo_is.npy holds nan in a filter column")
    oos = np.load(out_dir / "pwfo_oos.npy")
    if oos.shape != (*shape, len(oos_names)):
        raise ValueError(f"pwfo_oos.npy is {oos.shape}, index says {(*shape, len(oos_names))}")
    return {n: block[:, :, k] for k, n in enumerate(names)}, oos, wins


def select(cols: dict, filt: dict) -> tuple[int, int, int] | None:
    """One window's chosen row: `(combo, rank_tie, pick_tie)`, or None if nothing passed.

    `cols` maps metric -> `float32[n_combos]` for a **single** window's IS row block. That
    is the whole of PLAN §2.1's "structurally cannot screen on OOS": there is no argument
    here through which an OOS column could arrive, so the separation is a signature rather
    than a review question.

    Determinism, PLAN's review focus. `flatnonzero` is ascending, `argsort` is stable, and
    the surviving pool is re-sorted into combo order before the pick -- so every tie at
    either stage breaks on the lowest combo index, i.e. SPEC §3.3's `a`-major order, and
    the pick does not depend on how the rank happened to order its own ties.

    ⚑ Both tie widths are returned because the rank is mostly a tie-break and the tie block
    is large (PLAN Unit 8, from Unit 5): `mLb` is a median of small integer bar counts, so
    "the bottom 10 by `mLb`" is in practice ten arbitrary rows out of a block that can hold
    many more. `rank_tie` counts the eligible rows sharing the k-th ranked value, `pick_tie`
    the surviving rows sharing the winning pick value. A tie-break rule cannot fix a
    resolution problem, so the width is reported rather than hidden.
    """
    ok = np.ones(next(iter(cols.values())).size, dtype=bool)
    for m, op, v in filt["screens"]:
        ok &= OPS[op](_col(cols, m), v)
    idx = np.flatnonzero(ok)
    if idx.size == 0:
        return None

    rank_tie = 0
    if "bottom" in filt:
        m, k = filt["bottom"]
        vals = _col(cols, m)[idx]
        order = np.argsort(vals, kind="stable")
        cut = vals[order[min(k, idx.size) - 1]]
        rank_tie = int(np.count_nonzero(vals == cut))
        idx = np.sort(idx[order[:k]])

    m, direction = filt["pick"]
    vals = _col(cols, m)[idx]
    best = int(np.argmin(vals) if direction == "min" else np.argmax(vals))
    return int(idx[best]), rank_tie, int(np.count_nonzero(vals == vals[best]))


def evaluate(filt: dict, cols: dict, oos: np.ndarray, wins: list[dict]) -> list[dict]:
    """One record per OOS window: what the filter picked and what that pick then scored.

    ⚑ The two zero cases are kept apart, which is this unit's to pin (PLAN Unit 8, SPEC
    §6.4). `selected` is false when **no row passed the screens** -- [M25 p.15 Col G], no
    params exist for that week. `traded` is false when a row *was* selected and fired no
    signals -- [M25 Table 1]'s 01/14/15, 01/21/15 and 01/28/15, which carry `N`/`vup`/`vdn`
    filled with every OOS metric at 0. Both contribute a 0 to `toNP` and both stay in the
    denominator (SPEC §9-K); conflating them is worth up to 13.7 points of `%P`.

    ⚑ `nT` rides along on every record even though only `meyers2005` screens it. SPEC §6.6
    measured `CL4` and `CL2` each selecting a row with `nT < 5` in 5 of 24 windows, and
    that is faithful to [M25] -- neither published filter has a trade-count floor -- so it
    is a property to surface, not a defect to patch.
    """
    oos_names = rmv.METRIC_COLS[rmv.OOS_COLS]
    out = []
    for k, w in enumerate(wins):
        got = select({m: c[k] for m, c in cols.items()}, filt)
        rec = {"friday": w["friday"], "oos_start": w["oos_start"], "oos_end": w["oos_end"],
               "xmult": w["xmult"], "cost": w["cost"]}
        if got is None:
            rec.update(row=None, n=None, vup=None, vdn=None, nT=0.0, rank_tie=0,
                       pick_tie=0, selected=False, **dict.fromkeys(oos_names, 0.0))
        else:
            c, rank_tie, pick_tie = got
            n, vup, vdn = decode(c)
            rec.update(row=c, n=n, vup=vup, vdn=vdn, nT=float(cols["nT"][k, c]),
                       rank_tie=rank_tie, pick_tie=pick_tie, selected=True,
                       **{nm: float(v) for nm, v in zip(oos_names, oos[k, c])})
        rec["traded"] = rec["ont"] > 0
        out.append(rec)
    return out


def _longest(mask) -> int:
    """Longest run of True. 525 elements; a loop is the readable one."""
    best = run = 0
    for v in mask:
        run = run + 1 if v else 0
        best = max(best, run)
    return best


def aggregate(weeks: list[dict]) -> dict:
    """SPEC §6.3's aggregates over one filter's OOS periods. PLAN Unit 8's thirteen.

    ⚠ Four of these names collide with §6.1's per-combination metrics (SPEC §6.6): `%P`,
    `std` and `t` here are **per OOS period**, where the same names in a `weeks` record are
    per trade within one window. Unit 9 prints both blocks and has to disambiguate.

    Conventions, all pinned upstream rather than chosen here:

    - **Every window is a period**, selected or not, traded or not, contributing its 0
      (SPEC §9-K). [M25] divides its filter's total by the 446 weeks it traded while
      defining the null over all 517; one denominator on both sides is the only internally
      consistent reading, and it is what decides the Unit 9 gate.
    - **`toNP` is already net.** Cost lives inside `_simulate` per trade, so there is no
      `toGP - trades*cost` step here and no gross column to report (SPEC §6.6).
    - **Dispersion is `ddof=1`**, matching §1.2's `xmult` and §6.6's per-combination `std`.
    - **Drawdown runs off a zero baseline**, so an opening losing week is already a
      drawdown -- [M25 Table 1]'s all-loser weeks settle this. `eqDD` and `LLp` come back
      negative, matching the sign convention for every other loss metric.
    - **A zero period breaks both streaks**, exactly as a zero-net trade does in §6.6's
      winner/loser partition. `wpr + lpr` therefore need not cover the sample.

    ⚑ `oW|oL` is derived, not stored. The six OOS columns carry winners (`ownp`, `ownt`)
    and the total (`osnp`, `ont`), so the losing side is the difference -- which folds the
    net-zero trades that §6.6 counts as neither into the loser count. Measured 0 of 686,565
    real trades sit on that boundary, but `cost = 0` is a legal argument and would produce
    them. Reported as a magnitude ratio, both sides positive.
    """
    g = lambda k: np.array([w[k] for w in weeks], dtype=np.float64)  # noqa: E731
    p, ont, ownt, ownp = g("osnp"), g("ont"), g("ownt"), g("ownp")
    n = p.size
    avg = float(p.mean()) if n else 0.0
    std = float(p.std(ddof=1)) if n > 1 else 0.0
    eq = np.cumsum(p)
    peak = np.maximum(np.maximum.accumulate(eq), 0.0) if n else eq
    nt, nw = float(ont.sum()), float(ownt.sum())
    nl, wsum = nt - nw, float(ownp.sum())
    lsum = float(p.sum()) - wsum
    return {
        "n": n,
        # ⚑ These are not complements. `n_trd` is §6.3 col G, the periods that traded.
        # `n_sel` excludes only §6.4 case 2 -- a selected week that fired no signals is
        # counted in `n_sel` and not in `n_trd`, which is the whole point of keeping the
        # two zeros apart. `n - n_trd` is case 1 + case 2 together.
        "n_sel": sum(w["selected"] for w in weeks),
        "n_trd": int(np.count_nonzero(ont > 0)),
        "toNP": float(p.sum()),
        "avg": avg,
        "std": std,
        "t": avg / (std / math.sqrt(n)) if std > 0.0 and n > 1 else 0.0,
        "%P": 100.0 * float(np.count_nonzero(p > 0)) / n if n else 0.0,
        "%Wtr": 100.0 * nw / nt if nt else 0.0,
        "oW|oL": (wsum / nw) / abs(lsum / nl) if nw and nl and lsum else math.inf,
        "wpr": _longest(p > 0),
        "lpr": _longest(p < 0),
        "Blw": _longest(eq < peak),
        "eqDD": float((eq - peak).min()) if n else 0.0,
        "LLp": min(0.0, float(p.min())) if n else 0.0,
        # §6.3 col X: periods needed for a 98% chance equity is above zero, assuming
        # normality. n*avg / (std*sqrt(n)) = Z98 solves to (Z98*std/avg)^2. A filter that
        # does not make money never breaks even, hence inf rather than a large number.
        "BE": max(1, math.ceil((Z98 * std / avg) ** 2)) if avg > 0.0 else math.inf,
    }


def run_filters(filters: dict | None = None, out_dir: Path | str = OUT_DIR) -> dict:
    """Every filter against the stored pre-tail tables. `{name: {weeks, agg, filt}}`.

    One hoist shared by all of them -- the columns are the union of what they read, plus
    `nT`, which every record surfaces. ⚑ Each entry is one OOS-touching comparison and
    belongs in PLAN §Unit 9's multiplier.
    """
    filters = variants() if filters is None else filters
    names = sorted({"nT", *(_base(m) for f in filters.values() for m in _metrics(f))})
    cols, oos, wins = load_tables(names, out_dir)
    out = {}
    for name, f in filters.items():
        weeks = evaluate(f, cols, oos, wins)
        out[name] = {"filt": f, "weeks": weeks, "agg": aggregate(weeks)}
    return out



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
