"""Repeated Median Velocity kernel. See SPEC.md §1.1.

The hot path of the project: every later unit runs on this output. `scipy.stats.siegelslopes`
is the correctness oracle (tests only) -- it is ~20x slower and is never on the production
path.

Numerics per SPEC §7: no `fastmath` (it compiles `np.isnan` away), no NaN sentinels, float64
accumulation with float32 storage.
"""

from __future__ import annotations

import numpy as np
from numba import njit, prange

# SPEC §3.3: the lookback grid. 22 values. Read-only -- it is a module global handed to
# every caller, and an in-place edit would silently redefine the grid process-wide.
N_VALUES = np.arange(3, 25, dtype=np.int64)
N_VALUES.flags.writeable = False


@njit(cache=True)
def _repeated_median_slope(window: np.ndarray, inner: np.ndarray, pairs: np.ndarray) -> float:
    """Siegel (1982) repeated median slope of `window` against x = 0, 1, ... n-1.

    For each anchor point take the median of its n-1 pairwise slopes; the result is the
    median of those n medians. Breakdown point 50%.

    `inner` and `pairs` are caller-owned scratch (len >= n and n-1) so the bar loop does
    not allocate. `window` is float64 -- see the promotion in `_rmv_kernel`.

    ponytail: O(n^2) per bar, 576 ops at n=24, which is why this is fast enough without
    being clever. Above n ~ 100 the Matousek O(n log n) repeated median is the upgrade path.
    """
    n = window.shape[0]
    for i in range(n):
        k = 0
        for j in range(n):
            if j != i:
                # SPEC §1.1 maps to plain rise-over-run in array-index space: with
                # w[a] = price(t-(n-1)+a), the paper's (price(t-j)-price(t-i))/(i-j)
                # becomes (w[b]-w[a])/(b-a). Getting this sign wrong inverts the signal.
                pairs[k] = (window[j] - window[i]) / (j - i)
                k += 1
        inner[i] = np.median(pairs[:k])
    return np.median(inner[:n])


@njit(parallel=True, cache=True)
def _rmv_kernel(close: np.ndarray, ns: np.ndarray, out: np.ndarray) -> None:
    """Fill out[a, t] with the n=ns[a] repeated median slope ending at bar t.

    Parallel over n, not over bars: each thread owns one whole output row, so there is no
    reduction and no sharing, and the result is bit-identical at any thread count.
    """
    n_count = ns.shape[0]
    total = close.shape[0]
    for a in prange(n_count):
        n = ns[a]
        window = np.empty(n, dtype=np.float64)
        inner = np.empty(n, dtype=np.float64)
        pairs = np.empty(n - 1, dtype=np.float64)
        for t in range(total):
            if t < n - 1:
                # SPEC §7: warmup is 0.0, never NaN. Consumers start at index n-1.
                out[a, t] = 0.0
                continue
            # Copy into a float64 buffer so every operation below is unconditionally
            # float64. Measured: for SPY-range prices this changes nothing -- Sterbenz's
            # lemma makes float32 subtraction exact whenever the window's price ratio is
            # within [0.5, 2], and 24 SPY bars span ~1.002 -- so no test can distinguish
            # it, and none pretends to. It is kept because the guarantee then holds for
            # any input range rather than for SPY's. `pairs`/`inner` being float64 is the
            # part that is load-bearing: the medians average two middles on even counts.
            for k in range(n):
                window[k] = close[t - n + 1 + k]
            out[a, t] = _repeated_median_slope(window, inner, pairs)


def rmv_all_n(
    close: np.ndarray,
    ns: np.ndarray = N_VALUES,
    out: np.ndarray | None = None,
) -> np.ndarray:
    """Rolling repeated median slope for every n in `ns`. Returns float32[len(ns), T].

    Raw slope in price units per bar -- normalization is Unit 3's job (SPEC §1.2).

    Row-major with one row per n, C-contiguous: a window of one row is a few KB and stays
    in L1 across the whole (vup, vdn) sweep in Unit 6. Pass `out` to reuse a buffer.

    The first n-1 entries of each row are 0.0 warmup, not a computed slope. There is no
    NaN sentinel to test for -- start at index n-1.

    Live (Unit 12) uses this same function on a ring buffer of **max(ns) + 1** bars and
    reads both [:, -1] and [:, -2], so backtest and live share one code path by
    construction. The +1 is not optional: SPEC §2's rule is a *crossing*
    (`RMedV[t] >= vup and RMedV[t-1] < vup`), and a ring of exactly n makes [:, -2] a
    warmup 0.0, so `RMedV[t-1] < vup` is always true and the crossing rule silently
    degrades to a level rule. Measured on a 4000-bar walk at n=24: 71 buy signals become
    1294.
    """
    # Copy, not asarray: N_VALUES is read-only, a caller's array is not, and numba
    # compiles a separate specialization for each. Always-writable keeps it to one.
    # No test distinguishes this -- it costs 176 bytes and halves JIT compile time.
    ns = np.array(ns, dtype=np.int64)
    if ns.ndim != 1 or ns.size == 0:
        raise ValueError(f"ns must be a non-empty 1-D array, got shape {ns.shape}")
    if np.any(ns < 3):
        raise ValueError("n must be >= 3; a repeated median needs at least 3 points")

    close = np.ascontiguousarray(close, dtype=np.float32)
    if close.ndim != 1:
        raise ValueError(f"close must be 1-D, got shape {close.shape}")
    # A NaN would not surface: numba's np.median is partition-based and, unlike numpy's,
    # does not propagate NaN -- 61% of NaN-containing windows come back as a finite,
    # plausible-looking slope. data.validate covers the backtest path; the live path does
    # not go through it, so the check belongs here too.
    if not np.all(np.isfinite(close)):
        raise ValueError("close contains NaN or inf; RMedV would return a finite lie")
    if ns.max() > close.size:
        raise ValueError(
            f"close has {close.size} bars but max(n) is {ns.max()}; that row would be all "
            "warmup. For the live crossing rule you need max(ns) + 1 bars."
        )

    if out is None:
        out = np.empty((ns.size, close.size), dtype=np.float32)
    elif out.shape != (ns.size, close.size) or out.dtype != np.float32:
        raise ValueError(
            f"out must be float32{(ns.size, close.size)}, got {out.dtype}{out.shape}"
        )
    elif not out.flags["C_CONTIGUOUS"]:
        raise ValueError("out must be C-contiguous; Unit 6 keeps one row in L1")
    elif np.shares_memory(out, close):
        raise ValueError("out aliases close; the kernel would overwrite its own input")

    _rmv_kernel(close, ns, out)
    return out


# --------------------------------------------------------------- normalization (Unit 3)

# SPEC §1.2: the Appendix averages 1/sd over N=3..20, *not* over the full 3..24 grid.
# Averaging the whole grid instead moves xmult by 0.85% -- small enough to go unnoticed,
# and not the published method.
CAL_N_MAX = 20


def xmult(rmv_matrix: np.ndarray, mask: np.ndarray, ns: np.ndarray = N_VALUES) -> float:
    """Normalization multiplier for one slice: `mean_N( 1 / sd(RMedV_N * sqrt(N)) )`.

    SPEC §1.2. `RMedV * xmult * sqrt(N)` then has sd ~ 1 across every N at once, so the
    single 0.25..3.50 `(vup, vdn)` grid means the same thing at N=3 and at N=24. It is a
    unit conversion, not a strategy parameter.

    Callers scale **thresholds, not rows**: `RMedV_norm >= vup` is
    `RMedV >= vup / (xmult * sqrt(n))`, which is 14 divisions per n instead of a second
    22 x T matrix. Measured over 55.5M real (bar, n, v) triples the two orderings agree
    everywhere, but pin the convention anyway so backtest and live cannot drift apart.

    `mask` is required, and is meant to be `bars.gate == 1`. Warmup zeros and bars whose
    window straddles a session gap are not RMedV values, and including them inflates sd
    by 22.8% at N=3 (PLAN Unit 3). There is no defensible default, so it has to be
    spelled -- an unmasked row is a silent 20% error, not a crash.

    ⚠ **Per window, never frozen.** A constant fitted to one slice does not transfer:
    calibrated on 2016-17 SPY it lands 3.2x off over 2018-25, and the per-year signal
    scale swings 6.45x against PLAN Unit 3's 2x tolerance. Unit 7 calls this once per
    IS window and applies the result to that window's IS *and* OOS grid runs -- the IS
    window strictly precedes its OOS, so there is no look-ahead. Evidence, including what
    refitting does *not* fix, is in SPEC §1.2.1.
    """
    ns = np.asarray(ns)
    if rmv_matrix.ndim != 2 or ns.shape != (rmv_matrix.shape[0],):
        raise ValueError(
            f"rmv_matrix must be 2-D with one row per n; got {rmv_matrix.shape} against "
            f"{ns.size} n values"
        )
    mask = np.asarray(mask)
    if mask.dtype != np.bool_ or mask.shape != (rmv_matrix.shape[1],):
        # dtype is not pedantry. `Bars.gate` is int8, and `row[int8_gate]` is *integer* fancy
        # indexing: it silently returns row[0] and row[1] over and over and the sd it produces
        # looks entirely plausible. Pass `gate == 1`.
        raise ValueError(
            f"mask must be bool[{rmv_matrix.shape[1]}], got {mask.dtype}{mask.shape}"
        )
    sel = np.flatnonzero(ns <= CAL_N_MAX)
    if sel.size == 0:
        raise ValueError(f"no n <= {CAL_N_MAX} in ns; there is nothing to average over")
    if int(np.count_nonzero(mask)) < 2:
        raise ValueError("mask selects fewer than 2 bars; sd is undefined")

    inv = np.empty(sel.size, dtype=np.float64)
    for k, a in enumerate(sel):
        # One row at a time: the mask copies the row, and doing all 22 at once in float64
        # would peak at 33 MB on the full sample to produce a single float. ddof=1 vs 0
        # moves the result by 2.8e-6 relative here, and the float64 promotion by 4.7e-6 --
        # both immaterial, both pinned so two callers cannot disagree, and the promotion
        # kept for the same reason as `_rmv_kernel`'s (SPEC §7): the guarantee holds for
        # any input range, and no test can distinguish it on this one.
        sd = np.std(rmv_matrix[a][mask].astype(np.float64), ddof=1)
        if not np.isfinite(sd) or sd == 0.0:
            # A NaN here is worse than a crash: it makes every threshold NaN, every
            # `RMedV >= NaN` False, and the window a silent flat week instead of an error.
            # `rmv_all_n` rejects a non-finite close for the same reason.
            raise ValueError(f"sd(RMedV) is {sd} at n={ns[a]}; the slice is constant or not finite")
        inv[k] = 1.0 / (sd * np.sqrt(ns[a]))
    return float(inv.mean())


# ----------------------------------------------------------------- simulation (Unit 4)

# Column meaning of the trades array. A name, so a caller reading `trades[:, 3]` does not
# have to count and a reordering breaks loudly.
TRADE_COLS = ("entry", "exit", "dir", "net")


@njit(cache=True)
def threshold(v: float, xmult: float, n: int) -> float:
    """`RMedV_norm >= v` is `RMedV >= threshold(v, xmult, n)`. SPEC §1.2, PLAN Unit 6.

    **The only place this expression exists.** Unit 6 hoists it per `(n, v)` -- 14 divisions
    per n, not one per combo -- Unit 7 stores the window's `xmult`, Unit 11 writes it to
    `params.json`, and Unit 12b calls this per bar off that file. One expression means one
    rounding, so backtest, grid, replay and live cannot land on opposite sides of a
    boundary. Written `v / (xmult * sqrt(n))` and never `v / xmult / sqrt(n)`, which rounds
    twice and is a different float64.

    `njit` so Unit 6 can call it from inside `prange`; it is callable from Python too.
    Deliberately unguarded: `xmult <= 0` would invert the inequality and NaN would make
    every comparison False, but a raise inside a parallel region is not something to rely
    on, and `simulate` rejects the non-positive or non-finite threshold that comes out.
    """
    return v / (xmult * np.sqrt(n))


@njit(cache=True)
def _simulate(
    rmv_row: np.ndarray,
    close: np.ndarray,
    gate: np.ndarray,
    vup: float,
    vdn: float,
    cost: float,
    out: np.ndarray,
) -> int:
    """Fill `out` with closed trades and return how many. SPEC §2.

    Called from inside Unit 6's `prange` with a caller-owned buffer, so it allocates
    nothing. See `simulate` for the contract -- this is the same code without the guards.

    The `np.float64()` promotions on `close` below are not load-bearing and no test can
    distinguish them, exactly as in `_rmv_kernel`: both prices in a trade come from one
    session, so Sterbenz's lemma makes the float32 subtraction exact. Measured over 500,000
    pairs, **0** same-session subtractions differ from their float64 result, against **31,606**
    drawn from the full $180-$690 range. They are kept because the guarantee then holds for
    any input range rather than for this one -- and the day PLAN §8-F's overnight erratum is
    tested, a trade *can* span sessions and the range assumption stops holding.
    """
    total = close.shape[0]
    k = 0
    pos = 0  # -1 short, 0 flat, +1 long
    entry = 0
    for t in range(total):
        if gate[t] != 1:
            if pos != 0:
                # The gate went 1->0, so t-1 was the last bar the position could be held on
                # and its close is the fill. On a regular session that bar opens 15:50 and
                # closes 15:55 -- exactly SPEC §2's "flat at 15:55". t >= 1 always holds
                # here, because pos != 0 requires an earlier gated bar.
                #
                # ponytail: one exit rule for both reasons the gate can shut. Ceiling:
                # 2 of 2,682 1->0 edges are not the scheduled clock exit -- 2016-02-02
                # (one missing 5-minute bucket) and 2020-03-18 (the LULD halt) -- and on
                # those, close[t-1] means "assume you got out before the gap". That is
                # optimistic whenever the gap runs *toward* the position; it came out
                # conservative on this sample only because the grid was net short into both
                # down-gaps, which is a fact about the sample and not a property of the
                # rule. Fixing it would need a day index in the signature. Revisit if a
                # refresh pushes non-scheduled edges above ~0.5% of the total.
                out[k, 0] = entry
                out[k, 1] = t - 1
                out[k, 2] = pos
                out[k, 3] = pos * (np.float64(close[t - 1]) - np.float64(close[entry])) - cost
                k += 1
                pos = 0
            continue
        if t == 0:
            # Not defensive padding. In numba `rmv_row[-1]` is the LAST element, not an
            # error, so without this a slice whose first bar is gated would read the end of
            # the window as its own previous bar: end-of-window look-ahead, no exception.
            # Unit 7's week-anchored windows never start gated, which is exactly why this
            # has to be pinned here rather than discovered by a test that cannot fire.
            continue

        cur = np.float64(rmv_row[t])
        prev = np.float64(rmv_row[t - 1])
        # SPEC §2 crossing rules, both bounds inclusive, with the system's only negation of
        # vdn (which is a positive number compared against a negative velocity). Dropping
        # either `prev` term turns these back into the superseded 2005 *level* rules --
        # measured on a 4000-bar walk at n=24, 71 buy signals become 1294 (`rmv_all_n`).
        # No warmup or session guard is needed on `prev`, but only while max(ns) <= MAX_N:
        # blackout puts the first gated bar MAX_N + 1 bars past a gap, and `gate[:MAX_N]`
        # is 0, so over
        # the real sample 0 of 189,373 gated bars have a t-1 that is warmup, in another
        # session, or across a gap. That margin is exactly zero (SPEC §3.1).
        # The branches are mutually exclusive for vup, vdn > 0: cur cannot be both
        # >= vup > 0 and <= -vdn < 0.
        sig = 0
        if cur >= vup and prev < vup:
            sig = 1
        elif cur <= -vdn and prev > -vdn:
            sig = -1
        if sig == 0 or sig == pos:
            continue  # no signal, or already positioned that way -- hold (SPEC §2)

        if t + 1 >= total or gate[t + 1] != 1:
            # t is the last gated bar of its run, so a position opened at its close would be
            # flattened at that same close: zero bars, zero gross, exactly -cost. Live
            # flattens at 15:55, it does not also enter, so booking these would charge a
            # cost live never pays and break the Unit 12a parity check. Measured over the
            # pre-tail sample they are 1.74% of all trades (1.0-3.2% by combo), and the skip
            # removes *exactly* the trades whose exit index equals their entry index --
            # 0 discrepancies over 16 (n, v) combos, and nT drops by precisely the zero-bar
            # count. This is also what makes Unit 7's "N trades = N round trips" identity
            # true, so the cost convention and this skip are one decision.
            #
            # Skipping outright is right even mid-position: holding to the 1->0 edge one bar
            # later exits at close[t], the same fill a reversal here would have taken, so
            # the two are provably the same output and one branch beats two.
            #
            # ponytail: reading gate[t+1] rather than recomputing the session cut. Not price
            # look-ahead -- 2,680 of 2,682 1->0 edges are pure clock, known from
            # cache/nyse_calendar.json before the session opens, and live knows them too.
            # Ceiling: the blackout term keys on the *next* bar's arrival (data.build_gate),
            # which live cannot know at t. That is 2 of 189,373 gated bars, plus the one
            # session in 2,680 whose data stops early (2019-08-12, last bar 15:30). On those
            # three the backtest skips an entry live would take -- a deleted trade, not a
            # conservative one. Fixing it needs the calendar in this signature; not worth it
            # at 0.0016% of gated bars.
            continue

        if pos != 0:
            # Stop-and-reverse: the old position closes at this bar's close and the new one
            # opens at the same price. Each side is a trade and each pays `cost`, which is
            # what PLAN Unit 9's `trades x shares x (slippage + SEC/TAF)` check counts.
            out[k, 0] = entry
            out[k, 1] = t
            out[k, 2] = pos
            out[k, 3] = pos * (np.float64(close[t]) - np.float64(close[entry])) - cost
            k += 1
        pos = sig
        entry = t

    if pos != 0:
        # The array ended while still gated. One session in 2,680 does this in place
        # (2019-08-12, whose data stops at 15:30), and any slice that is not session-aligned
        # can. Force-closing is the only option under which every entry has a matching exit,
        # which is what Unit 5's trade-indexed equity needs; and the `t + 1 >= total` skip
        # above means the position being closed here was always opened strictly earlier, so
        # this can never emit a zero-bar trade.
        out[k, 0] = entry
        out[k, 1] = total - 1
        out[k, 2] = pos
        out[k, 3] = pos * (np.float64(close[total - 1]) - np.float64(close[entry])) - cost
        k += 1
    return k


def simulate(
    rmv_row: np.ndarray,
    close: np.ndarray,
    gate: np.ndarray,
    vup: float,
    vdn: float,
    cost: float,
    out: np.ndarray | None = None,
) -> np.ndarray:
    """Trades from one RMedV row under SPEC §2. Returns a float64[k, 4] view of `out`.

    Columns are `TRADE_COLS`: entry bar index, exit bar index, direction (+1 long, -1
    short) and **net** profit per share (gross minus `cost`). Gross is `net + cost` and
    bars-held is `exit - entry`, so nothing else is stored. `cost` is one round trip per
    completed trade; a stop-and-reverse bar closes one trade and opens another and is
    charged twice, which is what PLAN Unit 9's `trades x shares x (slippage + SEC/TAF)`
    sanity check counts.

    ⚠ **`vup` and `vdn` are raw RMedV thresholds -- price per bar -- not grid units.**
    Build them with `rmv.threshold(v, xmult, n)`, which is the only producer of this unit;
    Unit 6 hoists it per `(n, v)`. Both are positive: `vdn` is compared against `-vdn`
    internally, matching the paper's positive down-threshold against a negative velocity.

    Nothing here can detect a wrong-units caller, in either direction, and two of the three
    failure modes return a complete, plausible-looking backtest rather than nothing:

    1. grid units passed as raw -- the threshold is then 5.3x (n=3) to 15.0x (n=24) too
       large at the median `xmult`, and 24x at SPEC §1.2.1's quiet extreme: few or no trades.
    2. a raw threshold built from the **wrong window's** `xmult` -- that spans 20.4x across
       the 506 windows (SPEC §1.2.1), so a stale or global multiplier is still an entirely
       plausible positive float and the run completes with plausible metrics. This is the
       one that can smuggle look-ahead past Unit 7 in silence.
    3. a threshold scaled for a different `n` than `rmv_row` -- `n` is not an argument here,
       so not checkable at this boundary even in principle.

    None of the three is closed by validation; they are closed structurally, by
    `rmv.threshold` being the single expression and by Unit 12a's replay proving the
    backtest and live paths agree.

    `out` is a caller-owned buffer of at least `len(close)` rows, reused across Unit 6's
    4312 combos. A trade opens only on a gated bar and at most one per bar, so that many
    rows can never overflow.
    """
    # Made contiguous but never re-typed: a strided window view would otherwise make numba
    # compile a second layout specialization, and this is a no-op when it already is one.
    close = np.ascontiguousarray(close)
    rmv_row = np.ascontiguousarray(rmv_row)
    gate = np.ascontiguousarray(gate)
    total = close.shape[0]

    # Dtypes are checked, not coerced. Coercing a float64 rmv_row down to float32 would
    # round it onto the other side of a threshold and silently change the trade list --
    # the same class of defect as Unit 3's int8 mask. These are the dtypes Unit 2 and
    # `data.Bars` already produce, so a caller holding anything else has a bug upstream.
    if close.ndim != 1 or close.dtype != np.float32:
        raise ValueError(f"close must be 1-D float32, got {close.dtype}{close.shape}")
    if rmv_row.shape != (total,) or rmv_row.dtype != np.float32:
        # shape, not just ndim -- this is what catches passing the whole 22 x T matrix.
        raise ValueError(f"rmv_row must be float32[{total}], got {rmv_row.dtype}{rmv_row.shape}")
    if gate.shape != (total,) or gate.dtype != np.int8:
        raise ValueError(
            f"gate must be int8[{total}] -- pass `bars.gate`, or `mask.astype(np.int8)` -- "
            f"got {gate.dtype}{gate.shape}"
        )
    if not np.all((gate == 0) | (gate == 1)):
        # The kernel asks `gate[t] != 1`, so a stray 2 would read as flat while the equally
        # natural `gate[t] == 0` would read it as tradeable. One O(T) scan buys the right to
        # not care which one the kernel happens to use.
        raise ValueError("gate must contain only 0 and 1")
    for name, value in (("vup", vup), ("vdn", vdn)):
        if not np.isfinite(value) or value <= 0.0:
            # Not pedantry: a non-positive threshold makes the two crossing branches overlap
            # and a negative vdn fires the sell rule on nearly every bar, while a NaN makes
            # every comparison False -- a silent flat window rather than an error. This is
            # also where a non-positive or NaN `xmult` surfaces, since `rmv.threshold`
            # passes it straight through.
            raise ValueError(
                f"{name} must be a finite positive raw threshold from rmv.threshold "
                f"(a non-positive one means xmult was non-positive), got {value}"
            )
    if not np.isfinite(cost) or cost < 0.0:
        # A negative cost silently inflates every Unit 5 metric.
        raise ValueError(f"cost must be finite and >= 0, got {cost}")

    if out is None:
        out = np.empty((total, 4), dtype=np.float64)
    elif out.dtype != np.float64 or out.ndim != 2 or out.shape[1] != 4:
        raise ValueError(f"out must be float64[k, 4], got {out.dtype}{out.shape}")
    elif out.shape[0] < total:
        raise ValueError(f"out has {out.shape[0]} rows; {total} bars need at least that many")
    elif not out.flags["C_CONTIGUOUS"]:
        raise ValueError("out must be C-contiguous")
    elif any(np.shares_memory(out, a) for a in (close, rmv_row, gate)):
        # `gate` belongs here as much as the other two: the kernel writes `out[k]` as it goes
        # and reads `gate[t]` afterwards, so an aliased gate is corrupted mid-loop. Checked
        # after the contiguity conversion, because a strided input is copied away first and
        # the alias would no longer exist to find.
        raise ValueError("out aliases close, rmv_row or gate; the kernel would overwrite its input")

    k = _simulate(rmv_row, close, gate, float(vup), float(vdn), float(cost), out)
    return out[:k]


# -------------------------------------------------------------------- metrics (Unit 5)

# PLAN §1.6's 18 in-sample and 6 out-of-sample keys, in that order. Unit 7 writes columns
# 0:18 of an IS run into `pwfo_is.npy` and 18:24 of an OOS run into `pwfo_oos.npy` -- which
# is why one 24-wide row carries both blocks, and why four of them are duplicates: `osnp`,
# `ont`, `ollt` and `odd` are `tnp`, `nT`, `llt` and `dd` computed on a different slice
# (SPEC §6.1 against §6.2 -- same formulas, different trade set). 16 bytes a row against
# Unit 7 doing fancy indexing at every write.
METRIC_COLS = (
    "tnp", "nT", "PF", "%P", "mTrd", "mWTr", "mLTr", "mLb", "mWb",
    "lr", "wr", "dd", "llt", "std", "t", "eqR2", "eq2R2", "ktau",
    "osnp", "ont", "ownp", "ownt", "ollt", "odd",
)
N_METRICS = len(METRIC_COLS)
IS_COLS = slice(0, 18)
OOS_COLS = slice(18, 24)


@njit(cache=True)
def _median(buf: np.ndarray, m: int) -> float:
    """Median of `buf[:m]`, sorting that slice in place. numpy's convention on even `m`.

    Not `np.median`: numba's copies its input, and Unit 6's done-when is zero allocation
    inside `prange`. Measured with `NUMBA_NRT_STATS=1`: this is 0 allocations per call,
    `np.median` is 1, and `buf[:m].sort()` is 4 (numba's quicksort allocates a work stack).

    ponytail: insertion sort, because `m` is a trade count and that is 19.9 on average over
    34,496 real combos (p99 116, max 336). Ceiling: O(m^2), ~56k operations at the observed
    max. Quickselect is the upgrade path if the trade count ever grows.
    """
    if m == 0:
        # No trades, or none on this side of zero. 0.0 rather than a NaN sentinel: SPEC §7
        # rule 2 bans NaN in kernels, and every consumer of these is a filter comparison,
        # where a NaN quietly evaluates False on both sides of a screen.
        return 0.0
    for i in range(1, m):
        v = buf[i]
        j = i - 1
        while j >= 0 and buf[j] > v:
            buf[j + 1] = buf[j]
            j -= 1
        buf[j + 1] = v
    h = m // 2
    if m % 2 == 1:
        return buf[h]
    return 0.5 * (buf[h - 1] + buf[h])


@njit(cache=True)
def _metrics(trades: np.ndarray, out: np.ndarray, scratch: np.ndarray) -> None:
    """Fill `out[:24]` with `METRIC_COLS` from a Unit 4 trade array. SPEC §6.1 and §6.2.

    Called from inside Unit 6's `prange` on a per-thread `out` row and `scratch` buffer, so
    it allocates nothing. See `metrics` for the contract -- this is the same code without
    the guards. `out` may be float32 (Unit 6's storage row) or float64 (tests); every
    accumulator below is float64 either way, which is SPEC §7 rule 3.
    """
    k = trades.shape[0]

    # ---- one pass for the sums, the counts, the streaks and the largest loser.
    # A trade with net exactly 0 is neither a winner nor a loser, and breaks both streaks.
    # Over 686,565 real trades there are none -- `cost` is 0.027 and gross moves in cents,
    # so net cannot land on zero -- but `cost = 0` is a legal argument and a flat price move
    # then produces one, so the branch is spelled out rather than folded into a `>=`.
    tnp = 0.0
    ownp = 0.0   # sum of net over net-winning trades. Also PF's numerator: SPEC §6.2's
    gloss = 0.0  # "Winning Trades total Net Profits" is that same sum (PLAN Unit 5).
    nw = 0
    nl = 0
    llt = 0.0    # largest losing trade, stored negative; 0.0 when nothing lost
    wr = 0
    lr = 0
    run_w = 0
    run_l = 0
    for i in range(k):
        net = trades[i, 3]
        tnp += net
        if net > 0.0:
            ownp += net
            nw += 1
            run_l = 0
            run_w += 1
            if run_w > wr:
                wr = run_w
        elif net < 0.0:
            gloss -= net
            nl += 1
            run_w = 0
            run_l += 1
            if run_l > lr:
                lr = run_l
            if net < llt:
                llt = net
        else:
            run_w = 0
            run_l = 0

    # ---- five medians, `scratch` refilled before each. Bars held is `exit - entry`, which
    # Unit 4 guarantees is >= 1: it suppresses the entry on a run's last gated bar, so no
    # zero-bar trade exists to sort to the front of CL2's and CL4's bottom-k `mLb` rank.
    for i in range(k):
        scratch[i] = trades[i, 3]
    mtrd = _median(scratch, k)

    m = 0
    for i in range(k):
        if trades[i, 3] > 0.0:
            scratch[m] = trades[i, 3]
            m += 1
    mwtr = _median(scratch, m)
    m = 0
    for i in range(k):
        if trades[i, 3] > 0.0:
            scratch[m] = trades[i, 1] - trades[i, 0]
            m += 1
    mwb = _median(scratch, m) if m > 0 else np.inf  # symmetric with mLb below

    m = 0
    for i in range(k):
        if trades[i, 3] < 0.0:
            scratch[m] = trades[i, 3]
            m += 1
    mltr = _median(scratch, m)
    m = 0
    for i in range(k):
        if trades[i, 3] < 0.0:
            scratch[m] = trades[i, 1] - trades[i, 0]
            m += 1
    # +inf, not 0.0, and this one is a selection decision rather than a reporting one.
    # [M25 p.8]: *"b10mLb means the bottom or minimum 10 mLb rows"* -- CL2 and CL4 rank on
    # the SMALLEST mLb, so a 0.0 sentinel puts every no-loser and every no-trade row at the
    # head of the pool, where it displaces a real candidate and can then never win the
    # min-mLTr pick (its mLTr is 0.0, and every real one is negative). Measured on two real
    # windows, that is 321 and 260 rows of 4312 competing for ten slots. +inf sorts them
    # last instead, and is exact in float32.
    mlb = _median(scratch, m) if m > 0 else np.inf

    # ---- dispersion. ddof=1, matching `xmult`'s convention (Unit 3) so two callers cannot
    # disagree. Measured, the population form runs 2.4% low at the median trade count and
    # 18.4% low in the tail -- not a rounding difference once it reaches `t`.
    mean = tnp / k if k > 0 else 0.0
    sd = 0.0
    if k >= 2:
        acc = 0.0
        for i in range(k):
            d = trades[i, 3] - mean
            acc += d * d
        sd = np.sqrt(acc / (k - 1))
    tstat = mean / (sd / np.sqrt(k)) if sd > 0.0 else 0.0

    # ---- trade-indexed equity (SPEC §6.1), zero-based: it stands at 0 before the first
    # trade, so the running peak starts at 0 and an opening loser is already a drawdown.
    # `dd` is stored negative, matching [M25 Figure 2]'s own `eqDD = -10970` (SPEC §9-E).
    eq = 0.0
    peak = 0.0
    dd = 0.0
    sum_eq = 0.0
    for i in range(k):
        eq += trades[i, 3]
        sum_eq += eq
        if eq > peak:
            peak = eq
        elif eq - peak < dd:
            dd = eq - peak
    eq_mean = sum_eq / k if k > 0 else 0.0

    # ---- the two equity regressions. SPEC §7 rule 3: mean-centered, float64 accumulators.
    # Centering changes neither R^2 mathematically -- both fits carry an intercept, which
    # absorbs any shift in y -- but the naive uncentered float32 form is what PLAN §1.7
    # measured at 98.55 absolute error and 2283/4000 non-finite on a base-$200,000 curve.
    # x is centered too, which makes sum(xc) = sum(xc^3) = 0 and collapses the quadratic's
    # normal equations to the 2x2 solve below.
    x_mean = (k - 1) * 0.5
    sxx = 0.0
    sxy = 0.0
    syy = 0.0
    sx2y = 0.0
    sx4 = 0.0
    eq = 0.0
    for i in range(k):
        eq += trades[i, 3]
        yc = eq - eq_mean
        xc = i - x_mean
        x2 = xc * xc
        sxx += x2
        sxy += xc * yc
        syy += yc * yc
        sx2y += x2 * yc
        sx4 += x2 * x2

    # 0-100, not 0-1. The self-contained evidence is [M25 p.8]'s own screen -- *"we want the
    # R2 equity trend line correction to be <50, r2<50"* -- since a threshold of 50 against a
    # quantity bounded by 1 would pass every row in the table and make CL4's design intent
    # vacuous. [M25 Figure 2 Row 4]'s eqR2 = 82 and KTau = 93 corroborate the tool's scale.
    #
    # 100.0 when the fit is undefined (k < 2, or a flat equity curve), and that direction is
    # deliberate: 100.0 FAILS CL2's `eqR2 < 80` and CL4's `eqR2 <= 50`, while 0.0 passes
    # both and hands them a row that never traded. Measured over two real windows, 321 and
    # 260 of 4312 combos have nT < 2, and under a 0.0 sentinel every one of them entered
    # CL4's rank pool. A k = 2 row reaches 100.0 through the formula anyway.
    # `syy > 0.0` is the whole guard: a non-zero spread in y needs two distinct equity values,
    # hence k >= 2, hence sxx >= 0.5. Spelling `sxx > 0.0` as well would be a second condition
    # that can fall out of step with this one, exactly as `k >= 3` would have below.
    eqr2 = 100.0
    if syy > 0.0:
        eqr2 = 100.0 * (sxy * sxy) / (sxx * syy)
        if eqr2 > 100.0:
            eqr2 = 100.0  # Cauchy-Schwarz bounds the ratio at 1; this is float noise only

    # 0.0 when undefined, which is the OPPOSITE direction from eqR2's 100.0 -- and the
    # asymmetry is the point. Nothing screens eq2R2; `meyers2005` *picks* max eq2R2, so the
    # sentinel that fails safe is the one that can never win an argmax. Measured on two real
    # windows, 119 and 92 of 4312 combos score exactly 100.0 and every single one of them
    # has nT == 3 -- a quadratic through three points is an exact fit. None survives
    # `nT >= 16`, which is what makes that screen load-bearing rather than decorative:
    # relax it and `max eq2R2` becomes "pick a three-trade row", tie-broken arbitrarily.
    eq2r2 = 0.0
    if syy > 0.0:
        det = k * sx4 - sxx * sxx
        # `det > 0.0` IS the "at least three trades" condition, so there is no separate
        # k >= 3 guard to fall out of step with it. Cauchy-Schwarz makes det >= 0 always,
        # with equality exactly when xc^2 is constant -- which on a centred integer index
        # happens only at k <= 2. Measured: det is 0.0, 0.0, 0.0, 2.0, 16.0 at k = 0..4,
        # and the k = 2 zero is bit-exact (sxx = 0.5, sx4 = 0.125, both binary-exact).
        if det > 0.0:
            # y = a + b*xc + c*xc^2 on centered data. sum(xc) = sum(xc^3) = sum(yc) = 0
            # leaves b decoupled and a, c in one 2x2 system.
            a = -sxx * sx2y / det
            b = sxy / sxx
            c = k * sx2y / det
            ssres = 0.0
            eq = 0.0
            for i in range(k):
                eq += trades[i, 3]
                xc = i - x_mean
                r = (eq - eq_mean) - (a + b * xc + c * xc * xc)
                ssres += r * r
            # Deliberately NOT clamped at 0, unlike eqR2 at 100. A least-squares fit carrying
            # an intercept cannot do worse than the mean of y, so a negative here would be a
            # wrong solve rather than float noise -- and nothing reads eq2R2 through a sqrt,
            # so letting it through is what makes that visible. The eqR2 clamp above is a
            # different case: mutating its value is killed by the test suite, so that ratio
            # really does exceed 1 by noise.
            eq2r2 = 100.0 * (1.0 - ssres / syy)

    # ---- Kendall tau of the equity curve against trade order. x is the trade index and is
    # strictly increasing, so it carries no ties and tau-b collapses to
    # (C - D) / sqrt((C + D) * nPairs) -- which is what `scipy.stats.kendalltau` computes,
    # making scipy a valid oracle.
    #
    # Stored x100, so the range is [-100, 100] and all three correlation columns share one
    # scale. The source does not pin this: [M25 Figure 2]'s KTau = 93 sits beside eqR2 = 82
    # in the same row, but that is the aggregate column SPEC §6.3 keys as `KTau^2`, which
    # may already be squared. It is a project convention, chosen so a filter threshold
    # literal cannot mean 0-1 against one column and 0-100 against its neighbour.
    #
    # ponytail: O(k^2). Measured 3.79M pair-steps for a whole 4312-combo window against Unit
    # 6's 60 ms budget; the merge-sort inversion count is the upgrade path if k ever grows.
    ktau = 0.0
    if k >= 2:
        eq = 0.0
        for i in range(k):
            eq += trades[i, 3]
            scratch[i] = eq
        conc = 0
        disc = 0
        for i in range(k - 1):
            yi = scratch[i]
            for j in range(i + 1, k):
                if scratch[j] > yi:
                    conc += 1
                elif scratch[j] < yi:
                    disc += 1
        if conc + disc > 0:
            npairs = k * (k - 1) // 2
            ktau = 100.0 * (conc - disc) / np.sqrt(float(conc + disc) * float(npairs))

    out[0] = tnp
    out[1] = k
    # No losing trade means an undefined ratio, and `inf` is the reading that fails every
    # upper bound: SPEC §5's `PF < 4` and `1 <= PF <= 2` both reject it, which is what keeps
    # a zero-trade row out of `meyers2005` and `CL2`. `CL4` has no PF screen, so excluding
    # it there is Unit 8's zero-trade convention to pin, not this kernel's.
    out[2] = np.inf if gloss == 0.0 else ownp / gloss
    out[3] = 100.0 * nw / k if k > 0 else 0.0
    out[4] = mtrd
    out[5] = mwtr
    out[6] = mltr
    out[7] = mlb
    out[8] = mwb
    out[9] = lr
    out[10] = wr
    out[11] = dd
    out[12] = llt
    out[13] = sd
    out[14] = tstat
    out[15] = eqr2
    out[16] = eq2r2
    out[17] = ktau
    out[18] = tnp     # osnp
    out[19] = k       # ont
    out[20] = ownp
    out[21] = nw      # ownt
    out[22] = llt     # ollt
    out[23] = dd      # odd


def metrics(
    trades: np.ndarray,
    out: np.ndarray | None = None,
    scratch: np.ndarray | None = None,
) -> np.ndarray:
    """The 24 `METRIC_COLS` for one parameter combination's trades. SPEC §6.1, §6.2.

    `trades` is `float64[k, 4]` as returned by `simulate` -- `TRADE_COLS`, with column 3
    already **net** of `cost`. Every profit metric here is therefore net, including `PF`
    and `ownp`; gross is recoverable as `net + cost` but nothing in this unit wants it.

    Returns `out`, a `float64[24]`. Columns `IS_COLS` are the 18 that Unit 7 writes to
    `pwfo_is.npy` from an in-sample run; `OOS_COLS` are the 6 it writes to `pwfo_oos.npy`
    from an out-of-sample run. Both blocks are filled on every call, because the two runs
    differ only in which trades go in.

    Conventions, none of which the sources state outright and all of which change which row
    a filter picks (SPEC §5):

    - **Loss metrics are stored negative** -- `mLTr`, `llt`, `dd` -- which is [M25 Figure
      2]'s own convention (`LLTr = -3540`, `eqDD = -10970`). SPEC §9-E leaves *"smallest
      `mLTr`"* open; storing the signed value is what lets Unit 8 run both readings, since
      the magnitude convention is `abs()` of this one and not the other way round.
    - **`eqR2`, `eq2R2` and `ktau` are on a 0-100 scale** (`ktau` signed, so `[-100, 100]`).
      For `eqR2` that is sourced: [M25 p.8] screens *"r2<50"*, which against a quantity
      bounded by 1 would pass every row. For `eq2R2` and `ktau` it is a project convention
      -- no published value of either exists at the per-combination level, and [M25 p.13]'s
      `R^2 = 0.9496` is an Excel trendline label on a chart, not a column (SPEC §6.1).
    - **`eqR2` is R-squared of the straight-line fit.** SPEC §9-D's alternative `|r|`
      reading needs no second column and no `sqrt` on the data: transform the *threshold*
      instead, which is exact. CL2's `eqR2 < 80` becomes `eqR2 < 64` and CL4's
      `eqR2 <= 50` becomes `eqR2 <= 25`. A *signed* `r` is not recoverable and deliberately
      is not stored -- that needs `eqTrn`, and PLAN §2.1 pins `pwfo_is` at 18 columns.
    - **`%P` is 0-100.**
    - **A trade with `net == 0` is neither a winner nor a loser.** It still counts in `nT`
      and `tnp`, and it breaks both streaks.

    Degenerate combinations are common, not theoretical: over 34,496 real combos, 0.24%
    produced no trades at all and 4.8% produced fewer than three. Every one is defined, and
    each sentinel is chosen for the direction its consumer fails in, which is why they are
    not all the same value:

    - `PF = inf` and `eqR2 = 100.0` both **fail** SPEC §5's screens, keeping a row that
      never traded out of all three filters. A `0.0` `eqR2` passes both `< 80` and `<= 50`;
      measured, that admitted 321 and 260 of 4312 combos into CL4 on two real windows.
    - `mLb = mWb = inf` sort **last**, because CL2 and CL4 rank on the *smallest* `mLb`.
    - `eq2R2 = 0.0` can never **win** an argmax, which is how `meyers2005` uses it -- the
      opposite direction from `eqR2`, on purpose.
    - `mTrd`, `mWTr`, `mLTr`, `llt`, `dd`, `std`, `t` and the streaks are `0.0`. For `dd`
      and `llt` that is [M25 Table 1]'s published value on its all-winner and zero-trade
      weeks, not a choice.

    None of these distinguishes "no trades" from a real value; `nT` is the only column that
    does, and screening it stays Unit 8's job (PLAN Unit 8's zero-trade convention).

    ⚠ `simulate` returns a **view** into a buffer Unit 6 reuses across 4312 combos. Compute
    the metric row for combo *c* before running combo *c+1*, or copy at the boundary.

    Pass `out` and `scratch` to reuse buffers; `scratch` needs `len(trades)` float64 slots.
    """
    trades = np.ascontiguousarray(trades)
    if trades.ndim != 2 or trades.shape[1] != 4 or trades.dtype != np.float64:
        raise ValueError(
            f"trades must be float64[k, 4] from `simulate`, got {trades.dtype}{trades.shape}"
        )
    k = trades.shape[0]

    if out is None:
        out = np.empty(N_METRICS, dtype=np.float64)
    elif out.shape != (N_METRICS,) or out.dtype != np.float64:
        raise ValueError(f"out must be float64[{N_METRICS}], got {out.dtype}{out.shape}")
    if scratch is None:
        scratch = np.empty(max(k, 1), dtype=np.float64)
    elif scratch.ndim != 1 or scratch.dtype != np.float64:
        raise ValueError(f"scratch must be 1-D float64, got {scratch.dtype}{scratch.shape}")
    elif scratch.shape[0] < k:
        raise ValueError(f"scratch has {scratch.shape[0]} slots; {k} trades need that many")

    # Unconditional, and deliberately not folded into either `elif` chain above: `scratch`
    # is overwritten five times while `trades` is still being read, and `out` is written
    # before the equity passes finish, so either alias corrupts the input mid-kernel.
    if np.shares_memory(scratch, trades) or np.shares_memory(out, trades):
        raise ValueError("out or scratch aliases trades; the kernel would overwrite its input")

    # ponytail: no per-row check that `exit > entry` or that `dir` is +/-1. Those are
    # `_simulate`'s construction, and Unit 6 calls `_metrics` directly, so a scan here would
    # cost an O(k) pass to guard a path that does not go through it. `test_unit5_bars_held_
    # is_never_zero_on_real_data` checks the invariant where it is actually produced.
    _metrics(trades, out, scratch)
    return out
