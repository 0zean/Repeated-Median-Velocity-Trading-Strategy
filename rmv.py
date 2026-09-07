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
