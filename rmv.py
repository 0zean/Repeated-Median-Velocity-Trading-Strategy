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
