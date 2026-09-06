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
