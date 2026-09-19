# RMV Intraday Strategy — Build Plan

Repeated Median Velocity (RMedV) on SPY 5-minute bars, walk-forward optimized,
running on Alpaca paper → live.

Source specs: `es5rmed2.pdf` (Meyers 2005, ES 5min) and `CL5RMedV-4.pdf`
(Meyers 2025, CL 5min IV). Where they disagree, **the 2025 paper wins** — it adds
the normalization multiplier and the crossing entry rule.

*Rev 2 — incorporates an adversarial review of Rev 1. Changes marked ⚑ are corrections
to Rev 1, not new scope. Two were verified empirically before adopting; see §1.3 and §1.7.*

---

## 1. Pinned specification

Everything below is a decision, not a suggestion. Deviations need a line in §8.

### 1.1 Indicator

```
RMedV(t) = median_i { median_{j!=i} [ (price(t-j) - price(t-i)) / (i - j) ] },  i,j = 0..N-1
```

Equivalent to Siegel's (1982) repeated median slope = `scipy.stats.siegelslopes(y, x,
method="hierarchical")`. scipy is the correctness oracle, never the production path.

**Normalized** (2025 Appendix, pp.27–28): `RMedV_norm = RMedV * xmult * sqrt(N)`.

`sd(RMedV)` is proportional to `1/sqrt(N)`. Multiplying by `sqrt(N)` flattens the per-N
spread; `xmult = mean_N( 1 / sd(RMedV * sqrt(N)) )` scales the result to 1 sd.

⚑ **The paper contains two different multipliers and this matters.** The Appendix derives
`9.693120` for CL 5min — the mean of `1/sd` over **N=3..20**, excluding the N=2 row
(reproduced to 6 significant figures). But p.7 specifies `Mult = 6.7*sqrt(N)`, and Figure 3's indicator
parameter string (p.16) shows `6.7`. **Every published CL result was produced with 6.7**, so
Meyers' effective search range was 0.17–2.4 true sd, not 0.25–3.5. Both numbers go in
`SPEC.md`. Calibrate SPY over N=3..20 to match the Appendix method, and expect the top of
the grid to fire rarely.

SPY's `xmult` is unknown and must be measured (Unit 3). The `xmult = 4.00512` in the current
`rmv.py` is unsourced.

Consequence: `vup`/`vdn` are in **standard deviations**, so one grid works for all N and
transfers across symbols. This is the single most important change from the current code,
which searches raw slope units 0.02–0.40 where most of the grid never triggers.

### 1.2 Trading rules (2025 paper p.4)

| Rule | Definition |
|---|---|
| Buy | `RMedV[t] >= vup` **and** `RMedV[t-1] < vup` → long at market |
| Sell | `RMedV[t] <= -vdn` **and** `RMedV[t-1] > -vdn` → short at market |
| First trade | ignore all signals before **10:00 ET** (2005 paper p.4: 30 min after open) |
| EOD exit | flat at **15:55 ET**, never overnight |
| Between signals | hold (stop-and-reverse); flat only outside the trading window |

These are **crossing** rules. The current `rmv.py` uses level rules (`value > vup` → long),
which differ at the 10:00 re-entry: if RMedV is already above `vup` at 10:00, the level rule
enters immediately and the crossing rule waits for a fresh cross. Not cosmetic.

Fills at the **close of the signal bar** in backtest; live places a market order immediately
on bar close. Parity is enforced by test, not assumption (Unit 12a).

⚑ The 2025 paper carries a dated erratum on p.4 retracting its own first-trade rule:
*"(11/10/25) Note: this is no longer true, and future strategies should include the overnight
trades, and the Exit rule could still be at 1430."* We keep the 10:00/15:55 gate for v1
because SPY's overnight liquidity is genuinely thin, but under §1.3 the gate is the *only*
thing standing in the way — testing the erratum's suggestion later is a gate change, not a
rewrite. Logged as §8-F.

### 1.3 SPY-specific adaptations

⚑ **Bar series — corrected, and the most important fix in this revision.** Rev 1 specified an
RTH-only contiguous series with the RMedV window spanning overnight gaps. That was wrong on
both fidelity and correctness:

- *Fidelity*: the Appendix run header (p.27) reads `Total Number of Bars=736241` over
  1/1/2013–5/26/23 with `Trading Times Constraint Start Time=0 EndTime=0` — ~281 bars/session,
  i.e. the full ~24h Globex series. Meyers computes RMedV continuously and gates only
  *trading* to the pit session.
- *Correctness*: the repeated median's 50% breakdown protects against outlier **points**, not
  a level **shift** between two blocks. In a window straddling a session boundary, every
  cross-block pair carries the gap. Measured on synthetic SPY-scale data (bar sd $0.20,
  overnight gap $2.40):

  | N | clean sd | gap-straddling window | in normalized sd |
  |---|---|---|---|
  | 7 | 0.0921 | 0.4283 | **4.7** |
  | 12 | 0.0685 | 0.2831 | **4.1** |
  | 24 | 0.0487 | 0.1428 | **2.9** |

  Against a grid whose maximum is 3.5. Contamination spans the first `N-1` bars of each
  session and peaks near bar `N/2` — and **10:00 ET sits inside that region for every
  N >= 8** (open-labelled bars), i.e. 17 of the 22 N-values. The backtest would have been measuring an
  overnight-gap strategy wearing RMedV's clothes.

| Item | Decision | Why |
|---|---|---|
| Bar series ⚑ | **08:00–15:55 ET, 96 bars/session** — measured in Unit 1, narrowed from Rev 2's 04:00–20:00 | Bucket completeness is 100% only from 08:00 to 16:00; outside it 85–98%, and a missing bucket rescales RMedV's slope. 08:00 is also exactly `MAX_N` bars before the 10:00 gate. Full sample: 96.0 bars/session over 2,680 sessions. See SPEC §3.1. |
| Bar timestamp ⚑ | a bar is labelled by its **open** (Alpaca convention) | Undefined, this is worth one bar: 10:00 is the 7th RTH bar and the last gated bar opens 15:50 |
| Trading gate | `gate[t] = 1` iff `10:00 <= t_open < 15:55` ET | Trading window only; RMedV is computed everywhere |
| Price ⚑ | `adjustment="split"`, **not** `"all"` | See below |
| Costs ⚑ | $0.01/share round-trip slippage **+ SEC/TAF ~$0.017/share on sells** | TAF/SEC exceeds the slippage term at SPY $600 and scales with notional — not a constant across a sample where SPY went $180→$650 |
| Share size ⚑ | fixed **100 shares**; all backtest figures reported **per share** | Sizing is then a pure multiplier, and the per-share edge vs fixed costs is directly readable |
| Grid | `N` 3..24 step 1 (22); `vup`,`vdn` 0.25..3.50 step 0.25 (14 each) = **4312 combos** | 2025 paper p.7. ⚑ The paper states 4508, which is 23×14×14 — one of its own stated ranges is off by one N. Noted in `SPEC.md`. |
| Intraday halts ⚑ | any inter-bar gap > 5 min zeroes the gate for the next `max(N)` bars | 2020-03-09/12/16/18 circuit breakers splice an intraday price gap into the window, same failure as B1 |
| PDT ⚑ | live account must hold **>= $25,000** | ~8 round trips/week trips PDT in week one, and the consequence is closing-only for 90 days. This sets minimum capital, so it is a §1.3 input, not a Unit 13 discovery. |

⚑ **On `adjustment="split"`.** Alpaca's dividend adjustment retroactively rescales every bar
before each ex-date. That breaks four of this plan's own acceptance criteria: Unit 1's
bit-identical cache round-trip becomes unsatisfiable, append-only refresh becomes invalid,
Unit 3's frozen `xmult` drifts ~10% across 9 years as its calibration prices silently change,
and Unit 11's live-vs-backtest equivalence check compares two different price series. Live
trades raw prices. SPY has not split since 2005 and Alpaca data starts 2016, so `"split"` is
a no-op that makes the cache immutable. The ex-div gap is ~0.28% — smaller than the ordinary
overnight gap the series already contains.

### 1.4 Walk-forward scheme (PWFO equivalent)

- **In-sample**: `[friday_end - 30 days, friday_end]` — a 30-day delta, **31 calendar days
  inclusive**, ending Friday. **Out-of-sample**: the following Mon–Fri trading week. Step 7 days.
- ⚑ **Anchored to weeks.** Table 1 (p.25) is unambiguous — IS `10/18/23–11/17/23` (Friday),
  OOS `11/20/23–11/24/23` (Mon–Fri). Rev 1 left the anchor unspecified, which yields Wed–Tue
  sections and silently changes the granularity that `%P`, `lpr`, `wpr`, `Blw` and `LLp` are
  all defined on. ⚑ Rev 2: the IS span is 31 days inclusive, and [M25]'s p.4 prose contradicts
  its own Table 1 on the first window — Table 1 governs. See SPEC §4 and §9-H.
- Windows are generated across the whole series; there is no `n_windows` cap.
- Every window emits **one row per parameter combo**, carrying IS metrics *and* that combo's
  OOS metrics. This is what a PWFO file is, and it is what makes filter evaluation cheap:
  OOS results are precomputed for every combo, not just the selected one.
- Hard invariant, asserted in test: `max(IS timestamp) < min(OOS timestamp)` for every window.

### 1.5 Filter (parameter selection)

A **filter** picks one row out of the ~4312 IS rows. Structure is always `screens → rank → pick`.

| Name | Screens | Rank / pick |
|---|---|---|
| `meyers2005` | `1 <= PF <= 2`, `lr <= 3`, `nT >= 16` | max `eq2R2` |
| `CL2` (`b50mlb\|pf<4\|lr<3\|r2<80-mLTr`) | `PF < 4`, `lr < 3`, `eqR2 < 80` | bottom-50 `mLb` → min `mLTr` |
| `CL4` (`b10mLb\|lr<=3 r2<=50-mLTr`) | `lr <= 3`, `eqR2 <= 50` | bottom-10 `mLb` → min `mLTr` |

Note `eqR2 <= 50`: the 2025 filter deliberately **discards** the smoothest in-sample equity
curves — same instinct as the 2005 `PF <= 2` cap. The best-looking IS fit is the most overfit.

⚑ **`p<4` resolved: it is `pf<4`, Profit Factor.** Figure 2 col A (p.14) lists sibling filters
`b20mLb|pf<4|lr<3r2<60-mLTr`, `b10mLb|pf<5|lr<3r2<60-mLTr`, `b20mLb|pf<2|lr<3r2<60-std` — `pf` occupies
that slot, always alongside a *separate* `lr<3` term, so `p` cannot mean losing-periods-in-a-row.
Rev 1's guess was wrong; `CL2` is now implementable and is in the baseline set.

**Two remaining ambiguities**, run both ways in Unit 8 and both counted in §9's multiplier:

1. ⚑ **`r2` vs `r`.** The 2005 paper (p.6) defines `R22` as *"the correlation coefficient
   between the trade Equity line and the 2nd Order Polynomial Line"* — literally `r`. The 2025
   paper (p.15, Col V) writes `eqR2 — The correlation coefficient(R^2) of a straight-line fit`.
   Meyers uses the terms interchangeably. Harmless for `meyers2005` (a rank metric — argmax is
   preserved for non-negative `r`), **decisive for CL4**, where the `r2 <= 50` screen keeps
   `|r| <= 0.707` under one reading and `r <= 0.50` under the other.
2. **`mLTr` sign.** ⚑ The paper's own tables store loss metrics **negative** (`LLTr = -3540`,
   `LLp = -6640`, `eqDD = -10970`), under which "smallest `mLTr`" selects the *deepest* median
   loss. Rev 1 assumed positive magnitude / shallowest. The evidence points the other way;
   run both, report both.

### 1.6 Metrics per combo

IS (18): `tnp, nT, PF, %P, mTrd, mWTr, mLTr, mLb, mWb, lr, wr, dd, llt, std, t, eqR2, eq2R2, ktau`

OOS (6): `osnp, ont, ownp, ownt, ollt, odd`

Definitions from `meyersanalytics.com/Walk-Forward-Optimization.html` and CL5RMedV-4 pp.14–15,
transcribed verbatim into `SPEC.md` in Unit 0. `eqR2` is computed on **trade-indexed** equity
(2005 p.6: *"the trade Equity line"*), and is the straight-line fit — confirmed by Figure 2 Row 4
showing `eqR2 = 82` against the same curve's 2nd-order `R² = 0.9496` (p.13).

⚑ **The PDFs are gitignored (`*.pdf`) and never entered git history.** `SPEC.md` is therefore
the only durable record of all of this. Rev 1 contained several transcription errors, which
raises the stakes on Unit 0 considerably.

### 1.7 Numerics contract ⚑

Three rules, pinned here because two of them were latent bugs in Rev 1.

1. **No `fastmath=True`.** Verified in this repo (numba 0.65.1): a `fastmath` kernel scanning
   an array with 3 real NaNs reports **0** — `np.isnan` is compiled away. That silently
   invalidates Unit 2's "warmup NaNs are exactly the first N−1 samples" and Unit 5's
   "sentinel, not a NaN". The budgets have 2.5× headroom (1.95 s vs <5 s; 43 ms vs <60 ms),
   so fastmath buys nothing we need. If ever re-enabled, pass an explicit flag set excluding
   `nnan`/`ninf`.
2. **No NaN sentinels in kernels at all.** Warmup fills with `0.0` and every trade loop starts
   at index `N-1`. Correctness then does not depend on rule 1 holding.
3. **float64 accumulators; float32 only for storage.** Measured cost of getting this wrong,
   over 20,000 random equity curves:

   | equity curve | max \|r2_f32 − exact\| | screen flips at `r2 <= 50` |
   |---|---|---|
   | zero-based | 0.0003 | 0 / 20000 |
   | base $100 | **2.88** | — |
   | base $200,000 | **NaN/inf** (`n·Σy² − (Σy)²` goes negative) | — |

   So: equity regressions run on **zero-based, mean-centered** trade equity with float64
   accumulators. Table 1's `EQ`/`NetEq` columns reach $233,000 — exactly the exploding regime.

---

## 2. Architecture

Five modules. No packages, no `src/` layout, no plugin registry, no config framework.

```
data.py      Alpaca fetch → .npy cache. Returns arrays, not DataFrames.
rmv.py       numba kernels: RMV, normalization, trade sim, metrics, grid runner.
pwfo.py      window generation, PWFO driver, filter evaluation, bootstrap, report.
live.py      weekly refit job + intraday loop + fake broker for replay.
test_rmv.py  one file, plain asserts, runnable with `python test_rmv.py`.
```

### 2.1 Data layout — the whole performance argument

Structure-of-arrays, float32 storage, contiguous, preallocated. **Zero pandas below the
reporting layer.**

| Object | Layout | Size (9 yr SPY, extended hours) |
|---|---|---|
| close | `float32[T]` | ~1.4 MB |
| bar timestamps | `int64[T]` epoch ns | ~2.8 MB |
| trading gate | `int8[T]` | ~350 KB |
| RMV matrix | `float32[22, T]` C-contiguous, one row per N | ~31 MB |
| PWFO tables ⚑ | three memmapped `.npy`: `pwfo_is` (18 cols), `pwfo_oos` (6 cols), `pwfo_tail` | ~194 MB total |

Why this shape:

- One RMV row for an IS window fits **L1** (a few KB). The grid kernel reuses one hot row
  across all 196 `(vup, vdn)` pairs before touching another.
- Combo index is `a`-major (`a = c // 196`), so `prange`'s contiguous static chunks keep each
  thread on one or two N-rows. This ordering is why the measured 43 ms is 43 ms.
- Window slices are **copied** rather than passed as strided views — guarantees contiguity,
  costs ~150 KB.
- Trades live in a fixed stack buffer inside the kernel. Nothing allocates per combo.
- ⚑ **Three files, not one.** Splitting IS from OOS makes look-ahead *structurally
  impossible* rather than a review question: the Unit 8 evaluator opens only `pwfo_is`.
  Costs one extra `open_memmap`. `pwfo_tail` implements §3-Unit 9's withheld period the same way.
- ⚑ Store filter-relevant columns **column-major** (`[cols, windows, combos]`), or have Unit 8
  hoist its ~8 needed columns into RAM once. A row-major scan reads each metric with a
  96-byte stride — ~16× read amplification at Unit 10's then-planned ~2000 filters.
  ⚑ Moot: that search was cancelled, and Unit 10's `run_region` hoists **zero** IS columns.
  ⚠ With an empty name list `load_tables` checks no *column name* — `missing` is vacuously
  empty and the nan guard runs on a zero-width array. The structural claim (no IS metric
  reaches the region path) is real and is pinned by nan-filling `pwfo_is.npy`; the
  bookkeeping claim is not. Unit 10 review, finding 13.

### 2.2 Persistence

`.npy` / `np.memmap`, not Parquet. `pyarrow` is not installed, and the artifact is a dense
float32 matrix — the exact thing `.npy` is for. Memmapping also lets the filter scan read
straight off disk with no deserialization. Column names go in a sibling `.json`.

### 2.3 Dependencies

**Remove `vectorbt`** (imported nowhere; drags in matplotlib, plotly, sklearn, schedule,
ipywidgets, dill). **Add `numba` directly** — already installed as a vectorbt transitive dep
and the one piece we actually use. ⚑ Also drop `plotly` and `pyyaml` (the latter exists only
for `alpaca_api.yaml`, which Unit 0 deletes).

Target `[project].dependencies` **exactly**: `alpaca-py, numpy, pandas, scipy, numba`.
Dev group: `ruff, matplotlib`. `pandas` survives only at the reporting boundary; `scipy` only
as the test oracle.

### 2.4 Measured performance budgets

From prototypes run in this repo's venv (32 threads), not estimates. A unit is not done until
it hits its budget. ⚑ Figures below are for 546 windows (10 yr); §8-A recommends 9 yr
(~469 windows, ~194 MB), so treat these as the ceiling.

| Stage | Budget | Measured |
|---|---|---|
| RMV, 22 N × 196k bars | < 5 s, < 50 MB | **1.95 s, 34 MB** (prototype) |
| ⚑ RMV, 22 N × 257k bars, shipped kernel, no fastmath | < 5 s, < 50 MB | **1.91 s, 23 MB** |
| Same via `scipy.siegelslopes` (oracle only) | — | ~27–47 s |
| Grid, one IS window (4312 × 1638) | < 60 ms | **43 ms** (prototype) |
| ⚑ Same, shipped `run_grid`, 32 threads | < 60 ms | **3.2 ms** (16.6 ms on 1 thread) |
| Full PWFO, 546 windows, IS+OOS | < 60 s, peak RSS < 1 GB | **~32 s, 226 MB** |
| ⚑ Same, shipped `pwfo.run`, 551 windows (525 + 26 withheld) | < 60 s, peak RSS < 1 GB | **2.6 s, 411 MB** |
| ⚑ One filter over the full table, shipped `pwfo.evaluate` | < 1 s | **10–23 ms** (+ 0.11 s hoist, shared by all nine) |
| ⚑ Unit 9 report: 9 filters + 5000-iteration bootstrap + both falsifications | — | **~9 s** total, 574-line report |
| Live per-bar compute | < 1 ms, zero steady-state allocation | ⚑ `Book.on_bar`, 1620 combos: **121 µs** mean on a gated bar, p99 217 µs, max 449 µs; 9 µs ungated. **Not** allocation-free — see Unit 12a |

**The entire walk-forward is a ~35-second job.** Any design adding a job queue, a database,
a cache tier, or a distributed runner is solving a problem this project does not have.

---

## 3. Units of work

Each unit: one sitting, one testable deliverable, then an adversarial review, then **stop**.
DO NOT spawn many subagents during work to avoid hitting usage limits, only spawn the one adversarial subagent for review (Use sonnet 5 with max effort for subagent).

---

### Unit 0 — Spec, hygiene, dependency cut

**Do**

- Write `SPEC.md`: §1 of this plan plus the verbatim PWFO metric definitions. This is the only
  durable record — the PDFs are gitignored.
- **Rotate the Alpaca keys in `alpaca_api.yaml`, delete the file**, move to env vars.
  (Confirmed never committed — `*.yaml` gitignored from the first commit — so no history to scrub.)
- `pyproject.toml` → the exact dep list in §2.3. `uv sync`.
- Delete `benchmark.py`'s dummy-data `__main__`.

**Done when** ⚑ `[project].dependencies` equals `{alpaca-py, numpy, pandas, scipy, numba}`
exactly; `uv sync` clean; `uv.lock` relocked with no vectorbt entry; no credentials on disk.

---

### Unit 1 — Data layer (`data.py`)

**Do**

- `load_bars(symbol, start, end, feed) -> (ts int64[T], close float32[T], gate int8[T])`
- Alpaca `StockBarsRequest`, `TimeFrame(5, Minute)`, `adjustment="split"`, extended hours
  included at fetch; the series is then cut to the **measured-contiguous 08:00–15:55 window**
  (§1.3). Not an RTH filter — a contiguity boundary, chosen from data.
- Cache to `cache/{symbol}_5min_{feed}.npz` + `.json`; append-only, union-merging, written
  atomically, and validated before write. Exchange calendar cached beside it.
- `gate[t] = 1` iff `10:00 <= t_open < session_close − 5 min` ET — 15:55 normally, **12:55 on
  the 21 early closes in the sample**, from the exchange calendar — zeroed for `max(N)` bars
  after any inter-bar gap > 5 min.
- ⚑ **Feed gate, not a measurement.** Compare IEX vs SIP **close prices** on one month.
  Historical SIP is free (the 15-min delay is irrelevant to a backtest), so backtest on SIP
  regardless; the only live question is whether IEX bars are tradeable. IEX is ~2–3% of SPY
  volume and its prints carry cent-level error on every bar — the repeated median is robust to
  outliers, not to noise on *every* point, and 1–2¢ on a ~20¢ bar move is 5–10% per point.
  If the divergence is material, live requires paid SIP; decide **now**, because Units 2–13
  are otherwise premised on data that cannot be traded.

**Done when** cached round-trip is bit-identical to a fresh fetch (⚑ now achievable, given
`adjustment="split"`); timestamps strictly increasing, no duplicates; DST transitions give
correct ET; halted sessions handled without splicing; the feed decision is written down.

**Review focus** Silent gaps, forward-fill contamination, tz at DST, half-days, whether
extended-hours bars are actually present back to 08:00 for the whole sample.

---

### Unit 2 — RMV kernel (`rmv.py`)

**Do**

- `@njit(parallel=True)` `rmv_all_n(close, ns, out)` → `float32[len(ns), T]`.
- ⚑ Warmup filled with `0.0`, **no NaN sentinel, no `fastmath`** (§1.7).
- Inner: `O(N^2)` pairwise slopes, two median passes, stack buffers, no heap allocation in the
  bar loop.
- `# ponytail: O(N^2) per bar. Fine for N<=24 (576 ops). Above ~100, Matousek O(N log N).`

**Done when**

- ⚑ **Bit-exact** against `scipy.stats.siegelslopes(..., method="hierarchical")`: the
  kernel's float32 output equals `float32(scipy_float64)` exactly, for N in {3,4,5,10,23,24}
  at several offsets. Rev 2's "to 1e-9" was **unachievable by construction** — output is
  float32 and |rmv| ≈ 0.05, where float32 eps is 3.7e-9. Bit-exactness is strictly stronger
  than any tolerance and is what actually holds (24/24).
- Reproduces both papers' worked toy examples (each has a known answer of exactly 1.0).
- Every consumer starts at index `N-1`; a test asserts warmup values are never read.
- Budget: **< 5 s** for 22 N × 196k bars.

**Review focus** Off-by-one in the window, `j-i` sign, even-length median tie-breaking vs
scipy, float32 vs float64 accumulation.

---

### Unit 3 — Normalization calibration ✅ **shipped; the frozen constant was rejected**

**Done** `rmv.xmult(rmv_matrix, mask, ns)` — `mean over N=3..20 of 1/sd(RMedV_N * sqrt(N))`,
the [M25] Appendix method, computed from whatever slice it is handed. `rmv.CAL_N_MAX = 20`
pins the averaging range (the full 3..24 grid gives 7.244108 instead of 7.183306). `mask` is
required, not defaulted: gated bars only, because warmup zeros and gap-straddling windows
inflate `sd` by 22.8% at N=3.

**Not done, deliberately** No `norm.json`, no frozen constant, no held-out step. ⚑ Rev 2
pre-registered the trigger — *"if the saturation diagnostic swings more than ~2x across years,
switch to per-IS-window normalization, which deletes this unit, `norm.json`, and the held-out
step."* **Measured swing: 6.45x.** The trigger fired, so the unit deletes itself as designed.
Unit 7 calls `rmv.xmult` once per IS window and applies the result to that window's IS *and*
OOS grid runs — the IS window strictly precedes its OOS, so there is no look-ahead.

Full numbers in SPEC §1.2.1 (Table C, the per-year diagnostic, and the frozen/per-window
comparison). The headlines:

| over 506 pre-tail windows | frozen 2016-17 | frozen, best case | refitted per IS window |
|---|---|---|---|
| windows with every N>=5 within ±0.15 | 9.9% | 15.6% | **100%** (worst 0.142) |
| P(\|z\| > 3.5) on IS bars, median / worst | 11.5% / 69.0% | 0.09% / 20.4% | **0.59% / 2.08%** |
| saturated combos, worst window | 31.8% | — | 4.9% |
| combos that cannot trade at all, worst window | 19.6% | — | 7.1% |

"Frozen, best case" is the multiplier fitted on the whole pre-tail sample — the most favourable
frozen constant that exists. It clears 15.6%. The failure is not an artifact of having
calibrated on the two calmest years.

⚑ **The done-when moved, and this is that said out loud.** Rev 2 asked for ±0.15 for N>=5 **on
a held-out slice**. That form is not met and is not achievable: applied as written (max over
N>=5) to the following OOS week it holds in **17.2%** of windows. What is met is the in-sample
form, in 100% of them. SPEC §1.2.1 carries both numbers.

**Why the frozen version could never have worked.** RMedV is dollars per bar, and SPY ran
$210 → $690 through a 6x range of realized vol. Rev 2 anticipated the March-2020 case; the
measurement is worse than that — **every year after 2017** is off, by 2.1x to 6.5x. The
per-year mean normalized `sd` runs 0.747 (2017) to 4.819 (2025). In 2022, 34.8% of gated bars
sit beyond `vup = 3.50`, at which point the 4312 combos are near-clones and the filter is
choosing noise.

**Advance measurement from Unit 2 confirmed.** Gated-bars `sd` ratio N=3:N=24 was 3.04 against
the `1/sqrt(N)` law's 2.83; the calibration slice gives 3.086. Log-log slope -0.5386 for SPY,
-0.5668 for [M25 Table A]'s CL — SPY is steeper than the law, but *less* steep than Meyers'
own data, so the residual is a property of the estimator, not of SPY.

⚑ **Carried forward to Unit 8, not fixed here.** Refitting makes the *in-sample* scale exact
by construction; next week's is still a forecast. An IS window's `xmult` applied to the
following OOS week leaves a median `|sd − 1|` of **0.259** (p90 0.575) — a `vup` chosen on IS
lands on an OOS week whose scale differs by ~26%. That is ordinary walk-forward risk, and it
is not tunable away: estimation length is flat from 10 to 42 sessions and degrades beyond, so
the IS window is already on the plateau and a second parameter would buy nothing.

⚑ **The withheld tail was touched once, here, and it is recorded rather than argued away.**
The first version of the table above was computed over all 2,680 sessions instead of the 2,553
before 2026-03-01; the adversarial review caught it and the figures are now pre-tail. What the
tail saw was `sd(RMedV)` — a second moment of the indicator, never an OOS return, trade or
filter. Unit 9's comparison counter carries it.

**Found while writing the review-focus list, fixed in the same sitting.** `Bars.gate` is
int8, and `row[int8_gate]` is *integer* fancy indexing — numpy returns `row[0]`, `row[1]`,
`row[0]`... and the `sd` that falls out is entirely plausible. `xmult` now rejects any
non-bool mask; the caller writes `gate == 1`. This is the same class of defect as Unit 2's
NaN-through-`np.median`: a wrong answer with no exception.

---

### Unit 4 — Trade simulation kernel ✅ **shipped**

**Shipped** in `rmv.py`: `simulate(rmv_row, close, gate, vup, vdn, cost, out=None) -> trades`
over an njit `_simulate` Unit 6 can call from inside `prange`, plus
`threshold(v, xmult, n)`. 55/55 tests pass; **24 of 25 mutations killed**, the survivor
provably equivalent (below). Two of those mutations — a `TRADE_COLS` reorder and dropping one
arm of the `out`-aliasing guard — survived the first pass and were found by the §4 review, not
by the author; both are now killed.

- **`trades` is `float64[k, 4]`**, columns `TRADE_COLS = (entry, exit, dir, net)` — entry and
  exit bar index, direction ±1, and net profit per share. Gross is `net + cost` and bars-held
  is `exit - entry`; walking SPEC §6.1–6.3, 16 of the 18 starred IS metrics need only the
  ordered per-trade P&L, `mLb`/`mWb` add the bar count, and all 6 OOS metrics are sums or
  extrema over the same column. The two index columns are separately forced by this unit's own
  done-when. Caller-owned buffer of `len(close)` rows; nothing allocates per combo.
- ⚑ **`vup`/`vdn` are raw RMedV thresholds, and `rmv.threshold` is their only producer.**
  Grid units in the signature were built and then reverted. Three reasons, all measured:
  (i) §1.7's `>=` boundary is only exactly testable at the public boundary under raw units —
  a grid value has to survive a division, which lands exactly in **89.9%** of draws, so 5.1%
  of the time a *correct* `>=` kernel fails the test and 5.0% of the time a *buggy* `>` kernel
  passes it; (ii) grid units would have put `n` in the signature, and `n` matching `rmv_row` is
  structurally uncheckable — a new silent channel traded for an old one; (iii) the parity
  argument for it was false: `_simulate` returns closed trades, not a desired position, so
  `live.py` cannot call it and the formula would exist twice regardless. One shared
  `threshold()` is what actually closes that, and it is what Unit 12b calls per bar. Writing
  `v / (xmult * sqrt(n))` rather than `v / xmult / sqrt(n)` is not pedantry: **31.6%** of
  draws land on a different float64.
- **Three semantics the sources leave open are now pinned in SPEC §2.1** — no entry on the
  last gated bar of a run, one exit-fill rule for both gate causes, and `gate[t+1]` as a
  calendar read rather than look-ahead. Each was decided on measurement; see §2.1 for the
  numbers and for what each rejected alternative costs.
- **The IEX-vs-SIP signal divergence** ⚑ landed here as planned and is recorded in SPEC §3.2:
  IEX gates **72.8%** of the bars SIP does, disagrees on **up to 11.9%** of the signals on
  bars both feeds gate, and yields **73.2%** of the trades. SPEC §3.2's IEX rejection now rests
  on trades, not only on prices.

**Done when — as met.** Kernel == reference trade-for-trade and bit-exact on `net` over 1000
random series (14,917 trades, 1,287 of them sitting exactly on a threshold) and over a full year of
real bars; the hand-built series produces its exact expected list; a bar exactly equal to `vup`
triggers; on 64,067 real trades none opens before 10:00, none exits at or past 15:55, none
spans a session, and none is held across a bar the gate has shut.

Budget: **12 ms serial** for the worst of four **real** 1638-bar windows swept over the whole
4312-combo grid with each window's own refitted `xmult` — 20.8 trades/combo, 1.7 ns/bar-step;
an independent sweep over twelve windows measured 17–23 ms. Either way one thread does an
entire window inside Unit 6's 60 ms budget, so `prange` is headroom rather than the thing
being relied on. ⚑ The first version of this test used synthetic bars and reported 10 ms /
1.4 ns — **3–4× optimistic**, because threshold density drives the trade count and a row
calibrated by eye made 5 trades per combo where a real window makes 16–22. The test now
asserts `trades/combo > 10`, so it cannot quietly go back to timing an empty loop.

⚑ **The done-when as originally written did not catch the unit's largest defect.** "No trade
spans 15:55" is satisfied by a zero-bar trade *at* 15:55 — 1.74% of all trades, every one a
guaranteed `-cost` loser. The criteria that catch it are `exit > entry` and
`gate[entry + 1] == 1`, and they are now in the suite.

⚑ **Mutation survivor, resolved not chased.** `gate[t] != 1` → `gate[t] == 0` survives, and is
equivalent — but the reason has to be `build_gate`, not the wrapper. `simulate` does reject a
`gate` outside `{0, 1}` (and *that* guard's own removal is killed), yet Unit 6 calls the njit
`_simulate` directly and runs none of those guards. What makes the two comparisons identical
on the production path is that `data.build_gate` is a boolean expression cast to int8 followed
by zero-assignments, so `{0, 1}` is the only thing it can emit. The wrapper guard covers
hand-built callers; `build_gate`'s construction covers the grid.

**Review focus** `>=` vs `>` boundary; the first bar of the day (`t-1` is the prior
*extended-hours* bar, the intended contiguous reference under §1.3); forced EOD exit
accounting; reversal-on-same-bar.

---

### Unit 5 — Metric set ✅ **shipped**

**Shipped** in `rmv.py`: `METRIC_COLS` (24 keys, PLAN §1.6's order), `N_METRICS`, `IS_COLS`,
`OOS_COLS`, an allocation-free `_median`, the njit `_metrics(trades, out, scratch)` Unit 6
calls inside `prange`, and the guarded `metrics(trades, out=None, scratch=None) -> float64[24]`.
One 24-wide row carries both blocks: Unit 7 writes `row[IS_COLS]` from an IS run to
`pwfo_is.npy` and `row[OOS_COLS]` from an OOS run to `pwfo_oos.npy`. Four columns are
duplicates — `osnp`/`ont`/`ollt`/`odd` are `tnp`/`nT`/`llt`/`dd` on a different trade set —
which costs 16 bytes a row against Unit 7 fancy-indexing at every write.

**Done when — all met.** Every metric matches an independent numpy/scipy computation
(`_metrics_reference`, built on `np.polyfit` residuals, `itertools.groupby`,
`scipy.stats.kendalltau` and `np.maximum.accumulate` so it does not share the kernel's
shape): worst relative gap **4.1e-14** over 3,000 random trade arrays × 24 columns, and
**2.0e-14** over 72 combos of a real window. A hand-built five-trade list pins all 24 answers
literally — ⚑ *after* the review found that two of them were not: `eq2R2` was checked only by
`eq2R2 >= eqR2`, which admitted anything in [74.298, 100] including the 100.0 an exact-fit
bug produces, in the one test whose whole job is to owe nothing to the numpy reference. `ktau` agrees with scipy to **2.8e-14** over 2,000 curves including tie cases.
Degenerate cases are defined and tested. **38 of 38 mutations killed.**

⚑ **The base-$100 / base-$200,000 cases.** Reproduced on *trade-indexed* equity, and the test
demonstrates both halves rather than asserting one. Adding a constant to the first trade's
net shifts the whole equity curve and nothing else, so `eqR2`, `eq2R2` and `ktau` must not
move: measured **4.8e-9** under a $200,000 shift, against a float32 eps of 7.6e-6 at 100. The
same curves through the naive uncentred one-pass float32 form that §1.7 warns about go
**0.0002 → 1.72 → 96.94** absolute error at bases 0 / $100 / $200,000, with **104 screen
flips and 367 of 600 non-finite** at the top. The kernel is immune because it mean-centres in
float64 before accumulating; the test fails if the naive form ever stops breaking, so it
cannot decay into asserting nothing.

**Semantics pinned in SPEC §6.6**, all of them silent failures the sources do not settle:
signs (loss metrics negative), scales (`%P`, `eqR2`, `eq2R2`, `ktau` all ×100), `ddof=1`, the
strict `net > 0` / `net < 0` winner-loser partition with `net == 0` in neither, and a
per-consumer sentinel table. Two of those were **corrected during this unit after measuring
the selection consequence**, not the returned value:

- ⚑ **`eqR2` = 100.0, not 0.0, when the fit is undefined.** `0.0` passes both `eqR2 < 80`
  (CL2) and `eqR2 <= 50` (CL4). Measured on two real windows, **321 and 260 of 4312 combos
  have `nT < 2`, and every one of them entered CL4's rank pool** under the 0.0 sentinel.
  100.0 fails both screens.
- ⚑ **`mLb`/`mWb` = +inf, not 0.0, when the set is empty.** [M25 p.8] is explicit that
  `b10mLb` means the *ten smallest* `mLb`, so 0.0 puts every no-loser and no-trade row at the
  head of the rank pool, where it displaces a real candidate and can then never win the
  min-`mLTr` pick (its `mLTr` is 0.0; every real one is negative).
- **`eq2R2` stays 0.0** when undefined, which is the opposite direction on purpose:
  `meyers2005` *picks* max `eq2R2`, so its fail-safe sentinel is the one that cannot win an
  argmax, not the one that fails a screen.

⚑ **`meyers2005`'s `nT >= 16` screen is load-bearing.** `eq2R2` is exactly 100 for any
3-trade row. Measured, **119 and 92 of 4312 combos score exactly 100 and all of them have
`nT == 3`**; none survives `nT >= 16` (best survivor 95.9 and 98.3). Recorded in SPEC §6.6
because a later filter picking max `eq2R2` without a trade-count floor would look reasonable
and silently select noise.

⚑ **A source citation was wrong and is fixed.** SPEC §6.1 cited [M25 p.13]'s `R² = 0.9496`
as an `eq2R2` value. It is an Excel chart trendline label on Figure 1's *weekly time-indexed*
equity curve (the same figure carries `R² = 0.9285` for the net curve), not a column. No
published per-combination `eq2R2` value exists in either paper, so that scale is a project
convention. The 0–100 scale now rests on [M25 p.8]'s own screen — *"r2<50"* against a
quantity bounded by 1 — which needs no cross-table inference.

**Budget.** 4312 combos of `_simulate` + `_metrics` on a real 1638-bar window: **21 ms
serial on one thread**, against Unit 6's 60 ms for the whole window, at 20.1 trades/combo.
`_metrics` is therefore ~8 ms of it. Zero allocations inside njit, verified with
`NUMBA_NRT_STATS=1` through an njit driver — the interpreted path shows 3 per call, which is
numba boxing the three array arguments and not the kernel.

⚑ **The skipped-tests-count-as-PASS deferral now costs more than it did.** Six of the 14
`test_unit5_*` tests no-op to PASS without `cache/`. The sentinel *values* survive that —
`test_unit5_degenerate_combos_are_defined` and `test_unit5_scale_and_sign_conventions_are_pinned`
need no cache — but every ⚑ measurement above (321/260, 119/92, the eight screens, the 21 ms
budget) evaporates silently, and those measurements are the entire argument for the sentinel
directions. Still deferred as a cross-unit fix for Units 1–5, not this unit's alone; noting
that it has graduated from "the empirical basis of Units 3 and 4" to covering a design
decision.

⚑ **The withheld tail was touched once, here, and it is recorded rather than argued away.**
The adversarial review's first filter-trace probe loaded the full cache instead of truncating
at `TAIL_START`, so 2 of its 26 windows overlapped bars after 2026-03-01 and ran
`_simulate` + `_metrics` + a `CL2`/`CL4` selection over them. That is a stronger look than
Unit 3's second-moment one — the tail saw per-combo trades and P&L. Every number that reached
this file was then re-measured pre-tail, twice and independently: the review's own re-runs,
and mine, which assert a truncated load before measuring anything. The repo's tests are clean
here — they all go through `_real_bars()`, which truncates on load. Unit 9's comparison
counter carries it.

**Review focus** Division by zero; `mLTr` sign (§1.5); `lr` counting across window boundaries.
The last one cannot happen here: metrics are computed per combo per window on that window's
trade list, so a streak has no way to span windows.

---

### Unit 6 — Grid runner ✅ **shipped**

**Shipped** in `rmv.py`: `V_VALUES` (SPEC §3.3's 14 grid units, built as `0.25 * k` so every
value is exact where `np.arange`'s accumulated step is not), the `parallel=True` njit
`_run_grid`, and the guarded
`run_grid(rmv_window, close, gate, ns, vs, xmult, cost, out=None, trades=None, scratch=None)`
returning `float32[len(ns) * len(vs)**2, 24]`. `xmult` and `cost` are the two arguments the
plan's stated signature was missing; both are positional and neither has a default, because
Unit 7 must supply a per-window value for each and a default would be the one mistake this
unit cannot detect.

⚑ **`prange` runs over `a`, the N index, not the flat combo index.** §2.1 assumed the latter
and asked for static contiguous chunks to keep a thread on one or two N-rows; ranging over
`a` gets that by construction and, more usefully, makes the two scratch buffers safe with no
thread id at all — iteration `a` owns `trades[a]` and `scratch[a]` and no other iteration
touches them. Bit-identical output across thread counts is then a property of the loop shape
rather than something a test has to hope for. Buffers are `float64[>= len(ns), >= T, 4]` and
`float64[>= len(ns), >= T]` — **per iteration, not per thread** — 1.15 MB and 0.29 MB at the
1638-bar window, against the 0.41 MB table itself.

**Done when — all met.**

- **Any single row equals the Unit 4+5 path**, at two altitudes: all **4312 rows
  bit-identical** to a serial replay of `_simulate` + `_metrics`, and 8 random rows
  bit-identical to the guarded public `simulate` + `metrics`, which share no buffer, no combo
  ordering and no threshold hoisting with the kernel path. The second half is what would
  catch the kernel and its replay agreeing on a *wrong* order.
- **Bit-identical on 1 thread and on 32**, measured through `numba.set_num_threads` on a real
  window rather than argued from the loop shape.
- **3.2 ms per window** worst observed against the 60 ms budget (2.1 ms in a quiet run).
- **Zero allocation in `prange`: 7 per call at 88 combos and 7 per call at 4312**, all of it
  argument boxing at the interpreter boundary. ⚑ Unit 5's njit-to-njit driver trick is *not*
  available here — numba runs an inner `prange` serially when the kernel is called from
  another njit function, so a driver would measure a different kernel than the one that
  ships. The test instead holds the `prange` iteration count fixed at 22 (both grids keep all
  22 N-values) and varies only the inner sweep, 49×; anything the loop body allocated would
  scale with it. ⚑ **That sentence was false when first written** — see the review triage.

**26 of 26 mutations killed, 3 provably equivalent.** The equivalents are `prange` →
`range` (the loop shape is the correctness argument; the budget still passes at 16.6 ms),
`V_VALUES` built by `np.arange(0.25, 3.51, 0.25)`, which is **bit-identical** to `0.25 * k`,
and `astype(np.int64)` with and without `copy=False` on an already-int64 `ns`.

⚑ **That second equivalent caught a claim of mine that was wrong.** The comment shipped in
`rmv.py` said `arange` "accumulates its step and lands 3.5000000000000004 at the top". It does
not — `np.arange` computes `start + i * step`, not a running sum, and 0.25 is a power of two,
so both spellings are exact to the last bit. Corrected in place. The real hazard is the one
the mutation harness then confirmed: the stop is **half-open**, so `np.arange(0.25, 3.50,
0.25)` returns **13** values and silently drops `vup = 3.50` — 308 of the 4312 combos — and
every other test in this unit still passes, because they all derive their expectations from
`V_VALUES` itself. `test_unit6_grid_is_spec_3_3` pins the 14 values literally and is the only
test here that owes them nothing; it kills both that mutation and a shifted grid.

⚑ **The parallel speedup is 5.2×, not 22×, and it is recorded rather than fixed.** Measured
3.2 ms against 16.6 ms on one thread over the same four windows. The 22 tasks are far from
equal: over 24 real windows `n=3` averages **49.2 trades per combo** against `n=24`'s
**11.3**, and both `_median`'s insertion sort and the Kendall pass are O(k²), so the `n=3`
row alone is most of the wall clock. Named as a `ponytail:` ceiling in `_run_grid` with the
upgrade path (chunk the flat combo index across `numba.get_num_threads()` slices, buffers
indexed by chunk — the same disjointness argument, better balance). Not worth writing at
3.2 ms against 60.

⚑ **Free diagnostic — the effective grid is ~74% of 4312, and the plan had the mechanism
backwards.** Distinct trade sets per window, measured over 24 real pre-tail windows:
**2640–3652 of 4312 (61.2–84.7%), median 3202**. The plan predicted the clones would sit at
*low* N on a penny-tick instrument. They do not — distinctness rises monotonically with the
trade count and therefore *falls* with N: `n=3` averages 181.6 distinct of its 196 pairs,
`n=24` averages 130.2, correlation with mean `nT` **+0.94**. Zero-trade combos are not the
cause either (3.8 of 196 at `n=24`); a shorter trade list simply has fewer ways to differ.
Unit 9's multiplier should use ~3200, not 4312.

⚑ **The cheap proxy for that count is validated, not assumed.** Unit 9 can read distinct
*metric rows* straight off the stored table, which can only undercount — identical trade sets
always give identical rows. Measured against hashing the actual trade lists on three windows:
gaps of **0, 7 and 0** out of ~3300, i.e. at worst 0.22%. `test_unit6_effective_grid_size`
re-checks that on every run and fails if the proxy drifts past 1%.

⚑ **The one thing that cannot be guarded is `ns[a]` not being the N that produced
`rmv_window[a]`.** The matrix carries no labels, so a caller that slices rows and forgets to
slice `ns` the same way mis-scales every threshold by `sqrt(n_true / n)` and the run completes
with entirely plausible metrics. Only the shapes are checked. This is Unit 7's obligation and
it is stated in `run_grid`'s docstring as such.

### Review findings, triaged

One adversarial subagent per §4. It re-measured all seven claims (five reproduced exactly,
including the 24-window effective-grid figures bit-for-bit; the timings reproduced in kind at
2.3–5.7 ms across runs), confirmed both claimed equivalents independently, and ran its own
mutations against the buffer-disjointness argument — forcing every `prange` iteration onto one
`trades`/`scratch` row broke 5 of the 8 Unit 6 tests at once.

- ⚑ **Fixed — a measurement that did not match its own description, and it was mine.** The
  allocation test's "small" grid was `np.arange(3, 5)` — **2** N-values, so 8 combos and a
  `prange` count of 2 — while its docstring, its assertion message and its printed output all
  said 88 combos with the count "fixed at 22". The invariance result was real; the *reason*
  given for it was not, because the outer count moved with the inner sweep, which is exactly
  the confound the paragraph claimed to have eliminated. Now `rmv.N_VALUES` in both grids: 88
  and 4312 combos, 22 iterations either way, still 7 allocations per call. ⚑ Root cause worth
  recording: two `.replace()` calls in the patch script that wrote this test silently matched
  nothing, because this shell mangles `\n` inside heredocs — the same hazard the Unit 5
  handoff warned about. Every patch script since asserts each replacement fired.
- **Fixed — `ns` was silently coerced where every other dtype is checked and rejected.**
  `np.ascontiguousarray(ns, dtype=np.int64)` truncates, so a float64 `ns` carrying
  3.9999999999999996 for `n=4` — one ulp low, what a division or a non-dyadic `arange`
  produces — became 3 while still paired with `n=4`'s RMedV row. The reviewer demonstrated it:
  accepted with no exception, 4 of 12 rows silently computed against the wrong row. That is
  the `ns[a]` / `rmv_window[a]` mis-pairing this unit calls unguardable — and *this* channel of
  it was guardable all along. Now rejected before the cast. The other channel, a caller
  slicing matrix rows without slicing `ns`, still is not, and stays Unit 7's obligation.
- **Fixed — the guard order misattributed two errors.** `n_a` was taken from `ns` before `ns`
  was validated, so a malformed `ns` surfaced as *"rmv_window must be…"*. The grid is now
  validated first. The remaining case is deliberate: a non-finite `vs` reaches the threshold
  guard and reports *"thresholds are not all finite"*, which is the correct second line of
  defence rather than a gap — an exception is always raised.
- **Fixed — 8 fixed-seed spot rows touched only 5 of 22 N-values**, never `n=3` or `n=24`, the
  rows with the most distinct combos and the most extreme thresholds. Now six pinned grid
  corners plus six random draws. Mitigating, and measured by the reviewer: a vup/vdn swap is
  already caught by the 4312-row half at 3992 rows differing, so the public-API half's unique
  contribution is narrower than its framing suggested.
- **Fixed — the alias sweep's hardcoded `range(3, 6)`** over a 6-tuple was correct today and a
  magic-number contract tomorrow. Split into `inputs`/`outputs`; same 12 pairs, no positional
  assumption. Both new mutations against it are killed.
- **Rejected, with reason — the allocation test cannot see a dead allocation.** The reviewer
  inserted a per-combo `np.empty(4)` whose value was multiplied by a literal `0.0`; the count
  stayed at 7 because LLVM deletes an allocation that never escapes. Not a defect: it is not
  an allocation in the shipped binary. Its realistic counterpart — reallocating `scratch` per
  combo instead of reusing `scr = scratch[a]` — was caught hard, 4319 against 15.
- **Recorded, not fixed** — the proxy-validation gaps (0, 7, 0 of ~3300) reproduce only under
  `_grid_starts(24)`'s first three windows, not `_grid_starts(3)`; the shipped test checks one
  window and says so. And no look-ahead was found: `run_grid` reads only the slices and the
  two scalars it is handed.

**The withheld tail was not touched.** The reviewer asserted `len(bars) == 245_025` before
every measurement, and finished with `rmv.py` and `test_rmv.py` sha256-identical to how it
found them. Unlike Unit 5's review, this one costs Unit 9's comparison counter nothing.

**Review focus** Buffer aliasing across the three output buffers (checked pairwise, including
`scratch` carved out of `trades`, which PLAN called out); the `a`-major index arithmetic;
whether the 4312-row equality test can pass against a broken kernel.

---

### Unit 7 — PWFO driver (`pwfo.py`) ✅ **shipped**

**Shipped** in `pwfo.py`: `TAIL_START`, the `Window` namedtuple, `windows(bars)`,
`window_cost(close)` and
`run(bars, matrix, wins=None, out_dir=OUT_DIR, progress=0) -> list[dict]`, writing
`pwfo_is.npy` `float32[525, 4312, 18]`, `pwfo_oos.npy` `float32[525, 4312, 6]`,
`pwfo_tail.npy` `float32[26, 4312, 24]` and the sibling `pwfo_index.json`. 89/89 tests,
`ruff check .` clean, **31 of 35 mutations killed** (4 equivalent or deferred, below).
**2.6 s and 411 MB peak RSS** against a 60 s / 1 GB budget — 4.8 ms/window over 551
windows, both halves. Re-running is byte-identical across all four files.

- **551 windows**, IS 1824–2208 bars and OOS 372–480: **525 pre-tail**, 2016-02-05 to
  2026-02-20, and **26 withheld**. A window is emitted only when its IS start is at or
  after the first bar's date and its OOS Friday at or before the last, so the partial
  week at each end of the sample is dropped rather than run short.
- ⚑ **`friday - 30 days` is always a Wednesday** (30 mod 7 = 2), which is what makes
  SPEC §4's 31-day-inclusive span checkable on real data without the paper's calendar:
  the IS half starts exactly there in **518 of 525** windows and ends exactly on its own
  Friday in **510**, the remainder being Wednesday and Friday holidays. A 30-day-inclusive
  span would score 0 on the first census. The two [M25 Table 1] rows are pinned literally
  against a synthetic 2014 and 2023 calendar, because the SPY cache starts in 2016.
- ⚑ **The withheld set is keyed on the OOS *end*, not its start.** Identical here —
  2026-03-01 is a Sunday and no Mon–Fri week straddles it — and still correct the day
  someone moves `TAIL_START` into the middle of a week. `run` also asserts on the *bars*
  that no window it files as pre-tail reaches the withheld period, and refuses a run in
  which every window is withheld rather than writing a loadable, empty `pwfo_is.npy`.
- ⚑ **`xmult` swings 20.25× across windows** — 0.692 (2025-04-25) to 14.013 (2017-10-13),
  median 3.050. SPEC §1.2.1 measured 6.45× *per year*; per week it is three times wider,
  and the extremes are the April 2025 selloff and the quietest October on record. That is
  the strongest evidence yet for refitting, and it is also why the OOS run must reuse the
  IS multiplier: the two halves are one week apart and can still differ by a factor.
- ⚑ **Normalized IS `sd` holds, with 0.007 to spare.** Worst `|sd - 1|` over N>=5 across
  all 525 windows is **0.1429** (2017-12-29) against the ±0.15 done-when — held in 100%,
  but not comfortably. N=3–4 reach 0.2100 and are outside the claim. Anything that widens
  the IS window or narrows `xmult`'s N<=20 average should re-measure this first.
- ⚑ **`cost` is now per window and notional-scaled**: `SLIP + SEC_TAF_PER_DOLLAR *
  mean(IS close)`, running **0.0153 to 0.0296** (1.93×) against the flat 0.027 Units 4–6
  used. SPEC §3.2's own two numbers are read as $0.017 at SPY $600; the rate *and* the
  anchor stay Unit 9's to verify, and this unit fixes only which bars the price level
  comes off — the half that can leak. Deriving it from OOS prices would push a price
  level backwards into a row that was already picked.
- ⚑ **The canary has to read the files, not the buffers.** PLAN's wording ("assert
  `pwfo_oos[:, 0] != pwfo_is[:, 0]`") is on the *stored* columns and it is right: the
  first implementation compared `out_oos` against `out_is` in memory, which catches the
  two halves being one run and misses the other half of the same defect — a correct pair
  of runs written to the wrong file. Reading back off the memmaps catches both, and
  mutations 15–18 (each of the four block/file swaps, IS and tail) are all killed by it.
- ⚑ **`run` leaked its memmap handles on the exception path.** numpy offers no public
  close, so on Windows a caller's `TemporaryDirectory` could not be deleted and every
  failure inside the loop surfaced as `NotADirectoryError` with its own cause hidden —
  which is exactly how six mutation kills first reported. `mm._mmap.close()` in a
  `finally`; `_mmap` is private and there is no alternative.
- **11 tests added**, plus `_synth_bars` / `_et_days` / `_win_dates` helpers, and
  `test_rmv.TAIL_START` now derives from `pwfo.TAIL_START` so the driver and the suite
  cannot withhold different bars.
- ⚑ **PLAN §2.1's column-major option was not taken.** The tables are
  `[windows, combos, cols]`, which is what streams contiguously one window at a time;
  §2.1's stated alternative — Unit 8 hoisting its ~8 needed columns into RAM once — is
  64 MB and one pass, against restructuring the writer.

**Mutation survivors, all four resolved rather than chased.** (i) The leakage guard's
`<` → `<=` and (ii) dropping the tiling guard and (iii) dropping the pre-tail reach guard
are guards that never fire on a correct generator, so removing one alone changes nothing.
Three *combined* mutations — break the thing **and** remove the guard that catches it —
were added to prove that is what is happening, and all three are killed: a Tuesday OOS
start without the tiling guard is caught by the paper-table test, a tail keyed on the OOS
start without the reach guard by the withheld-file test, and a mid-session IS cut by the
rejection test. (iv) Taking `cost` off the first IS bar instead of the IS mean survives,
and should: both are IS-only price levels, the tests pin *IS-only* and *0.027 at $600*,
and which statistic of the IS price the fee is charged on is Unit 9's question. ⚠ The
review measured what that costs — up to **0.0016** between the two (window 2025-05-09),
11% of the width of the observed 0.0153–0.0296 range — so the deferral is a scope call,
not an immateriality claim, and the number travels with it.

### Review findings, triaged

One adversarial subagent per §4. It reproduced **every** numeric claim above exactly — the
window census, both exact-start/exact-end counts, the `xmult` extremes and median, the
0.1429 worst normalized `sd`, the cost range, and the byte-identical re-run *at the full
525-window scale* rather than the 4-window sample the shipped test uses. Nothing failed to
reproduce. Timing landed at 2.4–2.5 s and 410.8 MB against the claimed 2.6 s / 411 MB.

- **Fixed — the stored `cost` was never checked against a recomputation, where `xmult`
  was.** Replacing `cost = window_cost(close)` with the flat `0.027` of Units 4–6 — which
  silently deletes this unit's per-window cost entirely — passed all 89 tests. The matched
  `xmult` mutation dies. `cost` is given exactly `xmult`'s discipline by PLAN's own text,
  so it now gets exactly `xmult`'s test: the stored value equals `window_cost` of this
  window's IS bars, and is not the flat constant. Both new mutations are killed.
- **Fixed — the replay test read the index `run` returned, not the index on disk.** Units
  8, 9 and 11 take the JSON path, and a lossy float round-trip there would move every
  threshold. Measured lossless over 50 real windows by the reviewer; now exercised.
- **Fixed — `run(bars, matrix, [])` reported an empty window list as a tail problem**
  (*"all 0 windows are withheld"*). Unreachable through `windows`, which raises first.
- **Recorded — `window_cost` does not mask to gated bars, where `rmv.xmult` must.**
  Deliberate: an ungated bar is not an RMedV value (22.8% at N=3) but is a perfectly good
  price. Measured worst effect **5.18e-5** over 525 windows, 0.3% of `cost`. In the
  docstring, not changed — Unit 9 owns the model.
- **Corrected — "provably equivalent" was too strong for survivors (ii) and (iii).** The
  equivalence argument itself holds unconditionally. Its *evidence* does not: with
  `cache/` absent, 9 of the 11 `test_unit7_*` tests skip to PASS, and both combined
  mutations that justify those two survivors go green. That is the deferred cross-unit
  skipped-tests-count-as-PASS defect, now Units 1–7, and this is the first unit where it
  demonstrably weakens a claim rather than only an intention.
- **Recorded, no change — extending the IS slice one bar into OOS is caught by the
  `trades` buffer's bound, not by a semantic check.** `t_max` is computed before the leak,
  so the enlarged window overflows and 7 tests fail. Real defense in depth today; if
  buffer sizing ever moves per-call, the digest comparison in
  `test_unit7_xmult_and_cost_see_only_is_bars` is the only remaining net.
- **The withheld tail was not touched.** The reviewer asserted `len(bars) == 245_025`
  before every measurement, read `pwfo_index.json` but never `pwfo_tail.npy`, and finished
  with `pwfo.py` and `test_rmv.py` sha256-identical. Unit 9's comparison counter owes this
  review nothing.

**Do**

- Window generator: ⚑ **week-anchored** — IS = the 30 calendar days ending Friday, OOS = the
  following Mon–Fri, step 7 days.
- ⚑ Per window: `xmult = rmv.xmult(rmv_matrix, (gate == 1) & in_IS_window)` — Unit 3's
  normalization is refitted here, not frozen (SPEC §1.2.1). The **IS** multiplier scales the
  thresholds for that window's IS *and* OOS runs; recomputing it on OOS would be look-ahead,
  and reusing a global one saturates the grid (measured 6.45x swing across years).
- Per window: `run_grid` on IS **and** on OOS (same combos).
- ⚑ Stream to **three** memmapped files: `pwfo_is.npy`, `pwfo_oos.npy`, and `pwfo_tail.npy`
  (the withheld final 6 months, both IS and OOS, written and then not opened again until §Unit 9's
  final step).

**Done when**

- Every window asserts `max(IS ts) < min(OOS ts)` — **the leakage guard**.
- ⚑ Every window's `xmult` is derived from IS bars only, and is stored so a replay can
  reproduce the exact thresholds. It is **per window, not per row** — one extra column in the
  window index, not a 19th column in `pwfo_is` (§2.1 pins that at 18). Normalized IS `sd` within ±0.15 for N>=5 — held
  in 100% of 531 windows when Unit 3 measured it.
- OOS weeks tile the timeline exactly once: no overlap, no gap.
- Re-running produces byte-identical tables.
- Budget: **< 60 s**, peak RSS **< 1 GB**.
- A randomly chosen row's OOS `osnp` reproduces when replayed standalone.
- ⚑ **From Unit 5: a canary that the two files hold different runs.** `_metrics` fills all 24
  columns on every call and `osnp`/`ont`/`ollt`/`odd` are byte-identical to `tnp`/`nT`/`llt`/
  `dd`. So writing `table[:, OOS_COLS]` of the **IS** run into `pwfo_oos.npy` produces a file
  of plausible OOS metrics that are really IS metrics, and every other done-when here still
  passes — the leakage guard, the byte-identical re-run, the non-empty table. §2.1's
  "structurally impossible" is weaker than it reads. Assert `pwfo_oos[:, 0] != pwfo_is[:, 0]`
  on some rows of every window.

⚑ **From Unit 4: window slices must end on an ungated bar — assert it.** `_simulate` treats
the last bar of *any* slice as if it were the last gated bar of a run: no entry there, and an
open position force-closed at the slice edge. That is right for week-anchored windows, which
end Friday 15:55 on an ungated bar, so the windowed and full-sample answers coincide. It stops
being right the moment a window is cut by bar count or on a mid-session timestamp — the window
then silently loses one entry and force-closes at the window edge instead of the session edge.
One `assert gate[-1] == 0` per emitted window costs nothing and pins it.

⚑ **From Unit 4: `cost` is a selection input, like `xmult`.** It enters every IS metric, so it
enters the filter's choice of row and not merely the reported P&L. It therefore inherits
`xmult`'s discipline exactly: computed from **IS bars only**, stored per window beside
`xmult`, and applied to that window's IS *and* OOS runs. Deriving it from OOS prices would
leak a price level backwards into a row that was already picked.

⚑ **From Unit 6: three things `run_grid` hands over and one it cannot check.**

- The combo index is `c = a * len(vs)**2 + i * len(vs) + j` for `n = ns[a]`, `vup = vs[i]`,
  `vdn = vs[j]` — `a`-major, `vup` before `vdn`. Pinned in SPEC §3.3 because Units 8, 9 and 11
  all have to map a winning row back to its parameters, and nothing in the stored table
  records them.
- `xmult` and `cost` are positional arguments with no defaults, and both are per window
  (above). The same `xmult` goes into the IS and the OOS call; recomputing it on OOS is
  look-ahead.
- Pass `out`, `trades` and `scratch` once and reuse them across all ~469 windows — size them
  for the **longest** window the generator will emit, since oversized buffers are legal and
  undersized ones raise. 1.15 MB + 0.29 MB + 0.41 MB total, so this is about not churning the
  allocator 938 times, not about peak RSS.
- ⚑ **`ns[a]` must be the N that produced `rmv_window[a]`, and `run_grid` cannot check it** —
  the matrix has no labels, and a mis-pairing mis-scales every threshold by
  `sqrt(n_true / n)` and returns plausible metrics. Unit 7 slices both `rmv_matrix` and `ns`;
  a done-when here should pin that they are sliced together, or simply never slice `ns`.

⚑ **From Unit 5: the 18/6 split is a slice, not a projection.** `run_grid` emits
`float32[4312, 24]` for both the IS and the OOS run. Unit 7 writes `table[:, rmv.IS_COLS]` of
the **IS** run into `pwfo_is.npy` and `table[:, rmv.OOS_COLS]` of the **OOS** run into
`pwfo_oos.npy`. Both blocks are populated on every call — the OOS block of an IS run is
simply not written anywhere — so the look-ahead barrier is which *file* a block lands in, not
which columns were computed.

**Review focus** The `<=` / `+1 day` boundary bug pattern from the current `walk_forward`;
partial windows at the ends; holidays shortening an OOS week to 4 sessions (e.g. Thanksgiving
week 11/20–11/24/23); windows where zero rows pass any filter.

---

### Unit 8 — Filter evaluation ✅ **shipped**

**Shipped** in `pwfo.py`: `FILTERS` (SPEC §5's three baselines as data), `OPS`, `R2_COLS`,
`Z98`, `variants()`, `decode()`, `load_tables()`, `select()`, `evaluate()`, `aggregate()`
and `run_filters()`. 102/102 tests, `ruff check .` clean, **67 of 70 mutations killed**
(3 equivalent or inert, below). **10–23 ms per filter** over 525 windows x 4312 combos against a
1 s budget, after a **0.11 s** hoist of the 7 columns the filters read (63 MB IS + 54 MB
OOS). 13 tests added.

⚑ **67 of 70 with `cache/` and `pwfo/` present, and 67 of 70 without.** Unit 7's review
found the deferred skipped-tests-count-as-PASS defect weakening a real claim, so this unit
was measured against it rather than inheriting it. The first pass scored 61 without the
generated tables — all four lost kills in `load_tables` — and
`test_unit8_reads_only_the_two_tables_it_is_handed` now builds its own PWFO directory from
synthetic bars and takes all four. Every mutation kill here is cache-independent. The
defect is still open for Units 1–7.

- ⚑ **Nine filters, not three.** SPEC §9-D's two `r2` readings and §9-E's two `mLTr`
  conventions expand `CL2` and `CL4` into four each; `meyers2005` screens no `r2` column
  and picks `eq2R2`, so both transforms are no-ops on it and it collapses to one. The
  dedupe is derived, not asserted — `variants()` drops a transform that changed nothing.
  **All nine are OOS-touching looks and belong in Unit 9's multiplier.**
- ⚑ **Both ambiguities flip the sign of the answer.** They are not details to note and
  move past. `CL4` returns `toNP` **−94.31** as written and **+84.46** under the `|r|`
  reading; `CL2` returns **+132.11** as written and **−26.68** under the magnitude `mLTr`.
  Every one of the four readings of each filter is defensible from the source text, and
  they disagree about whether the strategy makes money.
- ⚑ **No filter is significant, before Unit 9 does any work.** Over 525 pre-tail weeks the
  nine `toNP` run **−160.93 to +132.11** per share with `t` from **−1.205 to +1.389**, and
  that is against a null of *zero* — Unit 9's null is a random filter with a positive mean
  (SPEC §6.5). `BE` is `inf` for five of the nine and 1149 weeks (22 years) for the best.
  Six of nine lose money. This is the unit's most important measurement and it is reported
  here rather than held for the gate.
- ⚑ **SPEC §6.4 case 2 never happens: no row fails the screens in any of 4725
  filter-windows.** [M25 p.15 Col G]'s *no params exist for that week* is real in the
  paper and empty in our sample, so the whole two-zero distinction lands on case 1, and
  the code path that produces a flat week is only reachable synthetically. It is kept and
  tested anyway — a tighter filter reaches it immediately.
- ⚑ **Case 1 is far more common than [M25]'s, and PLAN's 14-point estimate is low.**
  `meyers2005`, whose `nT >= 16` screen forces an active row, goes silent in 9 of 525
  weeks; the other eight filters in **92 to 179 (17.5%–34.1%)**, against [M25]'s 71 of 517
  (13.7%). Dropping those weeks from `%P`'s denominator moves it by **+0.8 points for
  `meyers2005` and +8.3 to +17.4 for the rest**, above SPEC §6.4's stated 13.7 ceiling —
  that ceiling is that paper's zero-week rate, not a bound. `CL2` trades in only 346 of
  525 weeks; `meyers2005` in 516.
- ⚑ **The tie block is wider than the 24-window sample said.** Over all 525 windows the
  rows tied at `CL4`'s 10th-smallest `mLb` run **1 to 55** (PLAN said 1–22) and at `CL2`'s
  50th **1 to 110** (PLAN said 3–33), medians 6 and 9. `pick_tie` — rows sharing the
  winning `mLTr` inside the kept pool — reaches 26. Both widths ride on every record.
- ⚑ **`rank` lost its direction knob; the key is `bottom`.** PLAN wrote
  `rank: (metric, direction, top_k)` and all three baselines rank one way, so the other
  branch was dead code — and SPEC §6.6 measured what it would do if used: a **top**-k on
  `mLb` puts every no-loser row, sentinel `+inf` and all, at the head of a pool it can
  never be displaced from, and a filter picking max `eqR2` selects an `nT <= 2` row in 24
  of 24 windows. The sanctioned directions are now the only ones spellable. Unit 10
  does not reopen it: the filter search is cancelled, so the sentinel directions stay pinned
  and unexercised.
- ⚑ **The trade-count floor is surfaced, not patched.** `CL4` selects a row with `nT < 5`
  in **82 of 525** windows and `CL2` in **120 of 525** (median selected `nT` 12 and 7);
  `meyers2005` never does, and its median is 28. Faithful to [M25], whose two published
  filters carry no `nT` screen — so every record ships its selected row's `nT`.
- **`select` cannot see an OOS column**, because there is no argument through which one
  could arrive: it takes one window's IS block and a filter. `load_tables` refuses an OOS
  name outright, which is the only boundary where the separation could go wrong.
- **`oW|oL` is derived, not stored.** The six OOS columns carry the winners (`ownp`,
  `ownt`) and the total, so the losing side is the difference — which folds §6.6's
  net-zero trades, counted as neither, into the loser count. Measured 0 of 686,565 real
  trades sit on that boundary; `cost = 0` is a legal argument and would produce them.
  Reported as a magnitude ratio: 1.12–1.36 across the nine, against `%P` of 33.5–47.6.
  These filters win slightly bigger than they lose and lose more often than they win.
- **`Z98 = 2.0537489106318225`** is inlined rather than imported: PLAN §2.3 keeps `scipy`
  out of everything but the test oracle, and the test pins the constant against
  `scipy.stats.norm.ppf(0.98)`.

**Mutation survivors, all three resolved rather than chased.** (i) Dropping the nan guard
in `load_tables` and (ii) dropping its `pwfo_is.npy` shape check are guards that never fire
on a correct table. Two *combined* mutations — break the thing the guard exists for **and**
remove the guard — were added to show what is left, and both are killed: a nan written into
the hoisted block fails `test_unit8_budget`'s trade-count assertion, and a window list
running past the table is caught by the sibling `pwfo_oos.npy` shape check, which the
mutation leaves standing. ⚠ The nan kill is indirect — it lands on a `mean()` going
non-finite, not on a semantic nan check — so that guard earns its place by reporting the
right cause, not by being the only net. (iii) Dropping `eq2R2` from `R2_COLS` survives, and
should: no baseline *screens* `eq2R2` and a pick is never threshold-shifted, so the member
is inert today. It stays because what belongs in that tuple is the **scale** — both columns
are stored ×100 (SPEC §6.6) — and a filter that screens `eq2R2` with it missing would leave
§9-D silently applying to only half the columns it names.

### Review findings, triaged

One adversarial subagent per §4. It reproduced **every** numeric claim above exactly — the
nine variants, the `toNP` and `t` ranges, both sign flips, `BE` inf for five of nine, case
2 at 0 of 4725, the tie widths, the `nT < 5` counts, the `%P` deltas and the trade counts —
with only the wall-clock hoist figure landing as a range (99.6–118 ms against 0.11 s).
Nothing failed to reproduce. It found no logic error in the shipped behaviour and no
look-ahead, and confirmed structurally that `select` has no parameter through which an OOS
column could arrive.

- **Fixed — `BE`'s floor of one period had no test at all.** Deleting `max(1, ...)` passed
  all 102 tests. `BE = 0` is reachable and meaningless: one profitable period has no
  dispersion, so §6.3 col X's formula gives 0 for *"the number of OOS periods you would
  have to trade"*. Now pinned on a single-week aggregate.
- **Fixed — `wpr`'s zero boundary was untested where `lpr`'s was.** Changing `p > 0` to
  `p >= 0` passed all 102 tests: the hand-computed series has its zero between two losers,
  so it discriminates one streak and not the other, while the docstring claims the rule is
  symmetric. A second series with a zero between two winners now pins it.
- **Fixed — `n_sel`'s comment called it "§6.3 col G's complement".** It is not: col G is
  the periods that *traded*, which is `n_trd`. `n_sel` excludes only case 2. The values
  were right and the comment could have misled Unit 9 about §6.5's denominators.
- **Pinned in SPEC — `oW|oL` reads two ways and the code silently chose one.** §6.3 col L's
  *"average OOS winning trades"* is a dollar average, not a count ratio, because col F of
  the same table writes *"Average **number** of"* when it means a count. The reading is now
  in SPEC §5 with that argument, alongside the difference-derived loser side.
- **Corrected — the silent-week range excluded `meyers2005` while the `%P` range included
  it.** Both figures were right and the sentence read as though one range explained the
  other. `meyers2005` goes silent 9 times; the other eight, 92–179.
- **Recorded, no change — `R2_COLS`'s `eq2R2` member is inert.** Dropping it passes all 102
  tests, because no baseline *screens* `eq2R2` and a pick is not transformed. Kept with a
  comment: it is the scale that belongs there, and the day a filter screens `eq2R2` its
  threshold has to move by the same rule or §9-D quietly stops applying to half the columns
  it names.
- **The withheld tail was not touched.** The reviewer never opened, read or hashed
  `pwfo_tail.npy`; `load_tables` reads only `pwfo_is.npy`, `pwfo_oos.npy` and the index.
  Unit 9's comparison counter owes this review nothing — but ⚑ it owes Unit 8 nine.

**Do**

- Filter as data: `{screens: [(metric, op, value)], rank: (metric, direction, top_k),
  pick: (metric, direction)}`. A dict and a ~30-line evaluator; not a class hierarchy, not a DSL.
- ⚑ The evaluator opens **only `pwfo_is.npy`** for selection and `pwfo_oos.npy` for scoring.
  Structurally cannot screen on OOS.
- Implement `meyers2005`, `CL2`, `CL4` verbatim; run both `mLTr` conventions **and** both
  `r2`/`r` readings (§1.5).
- Aggregates: `toNP, avg, std, t, %P, %Wtr, oW|oL, lr, wpr, Blw, eqDD, LLp, BE`.
- ⚑ Pin the zero-trade convention: Table 1 shows rows with `N/vup/vdn` filled and `osnp = 0`
  (params selected, no signals fired) — distinct from p.15 Col G's *no row passed the filter*.
  71 of Meyers' 517 weeks had no trades; conflating the two moves `%P` by up to 14 points.

⚑ **From Unit 5.** Five things are now settled upstream, and one is explicitly still this
unit's:

- **Both `mLTr` conventions come off one column.** It is stored signed and negative (SPEC
  §6.6), so §1.5's magnitude reading is `abs(mLTr)` and the deepest-loss reading is the
  column as stored. The reverse derivation does not exist, which is why the sign is stored.
- **Both `r2` readings come off one column too, by moving the threshold** (SPEC §9-D). With
  `eqR2 = 100·R²`, the `|r|` reading is `CL2: eqR2 < 64` and `CL4: eqR2 <= 25` — exact, no
  `sqrt`, no second column. The *signed* `r` reading is not supported and has no textual
  basis; do not add one without reopening §9-D.
- **Degenerate rows can no longer reach a filter by accident — but only below `nT = 2`.**
  `PF = inf` and `eqR2 = 100` fail every screen they touch, and `mLb = inf` sorts last, so a
  no-trade row cannot enter a bottom-k rank. ⚑ **The protection stops there.** From two
  trades up, `eqR2` is a real 2- or 3-point fit that slides under `eqR2 <= 50` honestly, and
  neither `CL2` nor `CL4` has any trade-count screen. Measured over 24 real pre-tail windows
  with the as-stored `mLTr` convention, **`CL4` selects a row with `nT < 5` in 5 of 24
  windows (`nT == 3` in 2), and `CL2` in 5 of 24**. That is faithful to [M25], whose
  published filters have no `nT` floor either — so it is a property to report, not a defect
  to patch. This unit should surface the selected row's `nT` in its output rather than
  silently returning a three-trade week.
- ⚑ **The zero-trade convention is still this unit's to pin.** Unit 5 stops a zero-trade row
  being *selected*; it does not decide how a selected-but-silent week is counted. [M25
  Table 1] confirms both cases are real and distinct — 01/14/15, 01/21/15 and 01/28/15 carry
  `N`/`vup`/`vdn` filled with every OOS metric at 0 (params chosen, no signals), which is not
  §6.3 Col G's *no row passed the filter*.
- ⚑ **`nT >= 16` in `meyers2005` is load-bearing** for its `max eq2R2` pick — measured, every
  combo scoring exactly 100 has `nT == 3` (SPEC §6.6). Any new filter picking max `eq2R2`
  needs a trade-count floor or it selects noise.

**Done when** on a synthetic table with a planted answer the filter selects exactly that row;
a no-eligible-row window produces a flat week, not a crash or a fallback pick; **< 1 s** per
filter over the full table.

⚑ **From Unit 5: `bottom-k mLb` is mostly a tie-break, and the tie block is large.** `mLb`
is a median of small integer bar counts, so its resolution — not the rank — is what binds.
Measured over 24 real pre-tail windows, the number of rows tied at exactly `CL4`'s
10th-smallest `mLb` runs **1 to 22**, and at `CL2`'s 50th, **3 to 33**. The pick is
deterministic (a-major combo order, §2.1) but "the bottom 10 by `mLb`" is in practice ten
arbitrary rows drawn from that block. Report the tie width alongside the selection; a
tie-break rule cannot fix a resolution problem.

⚑ **From Unit 7: what is actually on disk.**

- `pwfo_is.npy` is `float32[525, 4312, 18]` and `pwfo_oos.npy` `float32[525, 4312, 6]`,
  same window ordering, row-major `[window, combo, col]`. Column names, `n_combos`, the
  four dates per window, its IS/OOS bar counts and its `xmult` and `cost` are in
  `pwfo_index.json`; `pwfo_is` carries no 19th column and no parameter columns.
- A row's parameters come from SPEC §3.3's index alone: `a, r = divmod(c, 196)`,
  `i, j = divmod(r, 14)`, then `n = N_VALUES[a]`, `vup = V_VALUES[i]`, `vdn = V_VALUES[j]`.
- ⚑ **Hoist the ~8 columns the filter needs into RAM once.** PLAN §2.1 offered
  column-major storage *or* this, and Unit 7 took this: a row-major scan reads each metric
  with a 96-byte stride. Eight columns over 525 windows is 64 MB and one pass.
- ⚑ `cost` is no longer the constant 0.027 Units 4–6 tested against — it runs 0.0153 to
  0.0296 across windows, so IS P&L columns are not comparable across windows on a
  per-dollar basis without it. It is in the index, per window.

**Review focus** Tie-breaking determinism; the no-eligible-rows path; whether aggregates treat
the two zero cases distinctly.

---

### Unit 9 — Significance, costs, report ✅ **shipped — the gate fails 3 of 3**

**Shipped** in `pwfo.py`: `count_look()` and `comparisons.json` (⚑ described as committed here and in `count_look`'s docstring, but only actually committed in Unit 10 — see that unit's review); `_moments()`,
`_ktau()`, `_lin_r2()` and SPEC §6.3's eleven remaining columns folded into `aggregate()`;
`bootstrap()`, `null_moments()`, `significance()`; `shuffled_oos()`, `displace()`;
`spy_weekly()`, `R63`, `gate()`, `report()`. `python pwfo.py report` writes the deliverable
off the stored tables without touching `pwfo/`; **`UNIT9_REPORT.txt`** is that run. 111/111
tests, `ruff check .` clean, **55 of 56 mutations killed**, 8 tests added. The withheld tail
was not opened — sha256 `629122d198949afd…` unchanged.

⚑ **The first mutation pass scored 35 of 49 and the fourteen survivors were real.** Every
one was a Unit 9 test reading a *label* where it should have read a *number*: `gate`'s three
comparisons were never called, `spy_weekly`'s two boundaries were never checked, the report
asserted that a `Z Prob` row existed but not that it held `significance`'s numbers, and
`v20` was tested on a 12-week series where `p[-20:]` and `p[:20]` are the same twelve
numbers. Two more branches were unreachable on the real table and had to be built
synthetically — `displace` on a §6.4 case-2 week, and `_lin_r2`'s flat-equity sentinel.
The one surviving mutation is equivalent: adding the diagonal to `_ktau`'s pair set
contributes `sign(0)`, which is neither concordant nor discordant.

⚑ **The decision gate fails all three conditions, and none of them marginally.**

| Condition | Bar | Best of the nine | Margin |
|---|---|---|---|
| 1. bootstrap `Prob × K < 0.05` | 0.05 | `CL2` **3.05** (`p` 0.235, `K` 13) | fails ~60× |
| 2. `toNP > 0` after costs | > 0 | +132.11/share over 525 weeks | 3 of 9 positive; both open ambiguities flip the sign |
| 3. risk-adjusted > long-only SPY | μ/σ **0.1147** | μ/σ **0.0606** | fails for every filter, on both total and risk-adjusted |

- ⚑ **The power check ended the unit before the bootstrap was written**, which is exactly
  what this section said it might do. The mirror-random null is a sum of independent
  per-window uniform draws, so its mean and sd have a **closed form** — `sum_k mean_k` and
  `sqrt(sum_k var_k)` over each window's stored OOS column, one pass, no sampling. Over 525
  weeks it is **+61.48/share, sd 97.60**, so `z = 2` needs `toNP > 256.67`. The best filter
  reaches 132.11. The 5000-iteration bootstrap SPEC §6.5 specifies was written anyway and
  agrees to **61.43 / 96.96** — it is what carries the distribution's shape, and
  `null_moments` is what says whether it converged.
- ⚑ **This is not an underpowered sample — it is a clean negative.** A per-window oracle
  with perfect foresight scores **+4751.67** (floor −6328.14), so the `z = 2` bar is
  **5.4% of perfect foresight**: a filter capturing a twentieth of the achievable spread
  would clear it. And the same 525 weeks detect SPY's own drift at **t = 2.63**. The
  experiment had the power; there was no *filter* edge to find. PLAN's own free check —
  `μ/σ >= 2/sqrt(525) = 0.0873` — is missed by all nine, best **0.0606**.
  ⚑ **Superseded in part by Unit 10.** Every figure in this bullet stands. The inference
  drawn from it — "there was no edge" — does not: what had no edge was the *selection step*.
  The same 525 OOS weeks, equal-weighted over `n >= 5, v in [0.75, 2.75]` with no filter at
  all, return **+124.29/share at t = 2.39**, and the per-window IS→OOS rank correlation a
  filter depends on is **−0.0062, t = −0.45**. Read §3 Unit 10 before citing this bullet.
- ⚑ **The cost model cannot change the verdict, so the FINRA/SEC lookup is closed as a
  non-blocker.** `osnp` is already net, so gross is `net + ont × cost` exactly off the
  stored columns. Total cost paid over 525 weeks runs **$24.54 to $84.00 per share**, and
  at **zero cost** the whole CL4-as-written family stays negative while the null rises
  with it (**gross null +123.49**) — gross `CL2` +157.47 sits *closer* to the null than net
  does. SPEC §3.2's structural error (a per-share TAF folded into a notional-scaled fee, a
  Section 31 rate pinned to one year) is real and still recorded; it is worth cents on a
  result that misses by a factor of sixty.
- ⚑ **Autocorrelation is negligible, so no block bootstrap.** Lag-1..4 of `osnp` across the
  nine filters runs **−0.133 to +0.080**, against a ±2σ band of ±0.087 at n = 525 — one
  value (`meyers2005` lag-1) outside it, of thirty-six. The 23-of-30-day IS overlap does
  **not** inflate `t` here. Reported as this section required; the answer is "no action".
- ⚑ **Falsification 1 passes: shuffled OOS columns come back insignificant.** Permuting the
  combo axis per window preserves each week's OOS population and destroys only the IS→OOS
  row correspondence. Worst |z| over the nine is **1.143**.
- ⚑ **Falsification 2 is the sharpest result in the unit.** Scoring each OOS week with the
  selection made 50 windows earlier — under that week's own `xmult`, which `run` already
  stored — moves `toNP` by **−187.8 to +271.9 per share** over the 475 weeks both sides
  share, *wider than the filters' entire real range*, with displaced |z| reaching 1.919
  against real |z| ≤ 2.279. The selection carries no window-specific information; the
  filter is a static parameter prior, which is precisely the failure column-shuffling could
  never have caught. ⚑ The claim this section attaches to the test ("if that does not
  degrade…") is only *decidable* when the undisplaced filter is significant, and none is —
  so the test asserts the mechanism and the measured shift is recorded rather than asserted.
- ⚑ **Deviation from this section: the counter is keyed, not incremented.** PLAN wrote "a
  file the evaluator increments on every OOS-touching run". `K` corrects for the number of
  *distinct* hypotheses examined; re-executing a deterministic evaluation of the same nine
  filters tests nothing new, and an incrementing counter would make reported significance a
  function of how many times someone ran the script. **`K = 13`**: Unit 8's nine variants,
  the two accidental tail reads this section named, and the two falsification runs above.
- ⚑ **SPEC §6.5's published chain is now a test.** `test_unit9_significance_reproduces_
  both_rows_of_the_published_chain` pins [M25 p.9]'s own numbers — 396.7/week, z 4.92,
  4.23e-7, `K·p` 0.049 at n = 446 — *and* the §9-K row that the same total against the same
  null over all 517 periods gives `K·p` **2.234**. Significant to not significant on one
  choice of divisor, pinned so a change that re-flatters the filter fails.
- **Conventions settled here** (SPEC §6.3, "Pinned in Unit 9"): `KTau^2` stored **signed**
  ×100 despite the column name; `v20` as Δequity/Δperiod over the last 20; `aoTr` **net**;
  `toGP` reconstructed from stored columns rather than re-simulated; `tkr|bl` = 0 when `BE`
  is `inf`.

**Where this leaves the project.** ⚑ **Amended 2026-09-09 — read this with §3 Unit 10.**

As shipped, this section concluded that the gate failed on every condition and that Units
11–13 should not be written. The measurements are unchanged and the halt on live code stands.
The *inference* was wrong on two of the three conditions, and Unit 10 records why:

- Condition 3 ("beat long-only SPY") is unreachable by construction — **79% of SPY's return
  is overnight and pre-10:00**, which SPEC §2's 15:55 flatten forbids the strategy from
  touching. The gated-hours pool is **+103.79/share**, not +505.31.
- Condition 1's null is the equal-weight whole-grid portfolio, which has **gross t = +3.00**
  on its own. `K·p` measures the filter's skill *above the grid average*, not whether RMedV
  works.
- Condition 2 stands and is met.

What genuinely failed is the weekly parameter selection, and it failed for a reason no filter
search can repair: IS ranking predicts OOS ranking at **t = −0.45**. Unit 10 is therefore
rewritten from "filter search" to "region portfolio", the search is cancelled, and the
withheld tail — still unopened — becomes the project's one clean test.

### Review findings, triaged

One adversarial subagent per §4. It reproduced **every** numeric claim above independently —
the null moments, the bootstrap, the oracle ceiling and floor, the cost decomposition, the
autocorrelation range *including which of the thirty-six values is the outlier*, the SPY
benchmark, both falsifications, both sign flips — and confirmed `python pwfo.py report` is
byte-identical to the committed `UNIT9_REPORT.txt`. It found no look-ahead. It also
hand-checked `R63` against SPEC §6.3 column by column, which the shipped test could not do
(that test checks the report against `R63`, not `R63` against the source).

- ⚑ **Fixed, and it changed a headline number — `displace` wrapped its tail into a different
  price regime.** Rev 1 wrapped the last 50 windows back to window 0 to hold the denominator
  at 525. SPY is $210 there and $650 at the source, per-share dollar P/L is not scale-free
  across a 3× price change, and measured, **those 50 windows (9.5% of the sample) carried
  65.3% of the total displacement effect**, every one of the nine contributions pointing the
  same way. `displace` now **truncates**: it returns one record per window from `shift` on,
  carrying the earlier window's selection, and its control is `weeks[shift:]` — the same
  weeks, the same null, the same denominator, differing only in whose selection was applied.
  The conclusion is unchanged and now attributable; the quoted range moved from
  −181.5..+263.5 to **−187.8..+271.9**. The reviewer found the confound; the decomposition
  and the redesign are ours.
- **Fixed — a second equivalent mutation the tally had not found.** `tkr|bl`'s
  `if be != math.inf else 0.0` guard could not be killed by any test, because IEEE-754
  already gives `finite / inf == 0.0` and every factor is finite by its own zero-guard.
  Unkillable code is unfalsifiable code: the branch is deleted and the reasoning is in the
  comment. The mutation set grew from 51 to 56 as a result, now **55 killed**.
- **Fixed — `displace`'s `traded` flag had no coverage.** Mutating it to a constant passed
  both falsification tests. Nothing downstream reads it, which is exactly how it would have
  gone stale. Now pinned, on a synthetic window that fires no signals.
- **Recorded, no change — `significance` raises on `n = 0`.** `null_sd / n` runs before the
  `sd_mean > 0.0` guard. A zero-period aggregate is a misconfiguration, not a result, and
  the same reasoning already governs `run`'s all-windows-withheld `ValueError`: raising is
  the intended behaviour, not a gap.
- ⚑ **Promoted to a test — the reviewer checked something none of our code did.** The normal
  tail `significance` reads is *earned*, not assumed: the 5000-draw distribution has skew
  −0.054 and excess kurtosis −0.12 despite per-window OOS columns skewed −4.20 to +1.00, and
  the empirical tail for `CL2` (0.2434) tracks the normal `Prob` (0.2346). That is the CLT
  over 525 windows doing its job. `test_unit9_bootstrap_converges_on_its_exact_moments` now
  asserts the sum's shape and three empirical-vs-normal tail quantiles.
- **Protocol note.** The reviewer disclosed running `sha256sum` on `pwfo_tail.npy`, which
  our brief had listed among forbidden operations. The brief was over-strict: a SHA-256
  digest is one-way, reveals nothing about the values, and is this project's own sanctioned
  integrity check (§Verification). No `np.load`, no parse, hash matched. **The tail is not
  spent.**

---

**What the unit was asked to do, kept for the record.**

⚑ **From Unit 7: three things this unit inherits.**

- `pwfo_tail.npy` is `float32[26, 4312, 24]` — 26 windows, IS columns 0:18 from the IS run
  and OOS columns 18:24 from the OOS run, written 2026-09-08 and **not opened since**. Its
  first OOS week starts 2026-03-02. The withheld set is keyed on each window's OOS *end*.
- The cost model is now `SLIP + SEC_TAF_PER_DOLLAR * mean(IS close)` per window, read off
  SPEC §3.2 as $0.01 slippage plus $0.017 SEC/TAF **at SPY $600**. That anchor is an
  inference from the one sentence in §3.2 that states both numbers, and it is not sourced.
  This unit's existing obligation to verify (not re-apply) the cost model now has a
  specific target: the FINRA TAF per-share schedule, the SEC Section 31 rate for each year
  in the sample, and whether folding them into one notional-scaled scalar is defensible at
  all. Measured spread as shipped: 0.0153 to 0.0296.
- The comparison counter owes nothing to Unit 7: no filter has been evaluated and
  `pwfo_oos.npy` has been read only by tests, which compare stored rows against standalone
  replays of the same bars and never rank, screen or aggregate them.

⚑ **From Unit 8: the counter now owes something, and the answer is already in.**

- **Nine filters have been run against `pwfo_oos.npy` over all 525 pre-tail windows**
  (SPEC §5, "Pinned in Unit 8"). That is the first real entry in this unit's multiplier and
  it is not the last — §8's cheap A/Bs are further looks at the same columns. **Build the
  counter file before running anything else**; nine is the number it starts from, not zero.
- **The gate looks unreachable on this evidence.** The nine `toNP` run −160.93 to +132.11
  per share with `t` from −1.205 to +1.389, six of nine negative, against a null that is
  *positive* (§6.5). Do the free power check first — it is one line and it may end the
  project before the bootstrap is written.
- **Both open ambiguities flip the sign**, so there is no "the" answer to report: `CL4` is
  −94.31 as written and +84.46 under §9-D's `|r|` reading; `CL2` is +132.11 as written and
  −26.68 under §9-E's magnitude `mLTr`. Reporting a favourable reading without the other
  three is the failure mode §5 is named after.
- **Every aggregate is already computed** by `pwfo.aggregate` on §9-K's convention (all OOS
  periods, zero-trade weeks counted as zero) and is net, not gross. What is missing from
  §6.3 is the bootstrap block (`Prob`, `skew`, `kur`, `v20`, `KTau^2`, `eqR2`, `tkr|bl`) and
  the row-1 scalars, which are this unit's.
- **A selected week carries `rank_tie` and `pick_tie`.** Report them: the rows tied at
  `CL4`'s cut run 1–55 and at `CL2`'s 1–110, so "the bottom 10 by `mLb`" is ten arbitrary
  rows out of as many as 55. A `toNP` quoted without that width overstates what was chosen.
- **`pwfo_tail.npy` was not opened.** `load_tables` reads `pwfo_is.npy`, `pwfo_oos.npy` and
  the index only; the file is byte-identical, and one test overwrites it with unparseable
  bytes to prove nothing reaches it.

**Do**

- Mirror-random-filter bootstrap (5000 iterations): per window pick a uniformly random row's
  OOS net profit, sum, repeat → chance distribution of `toNP`. This is the **decision
  statistic**, and it is the right null: Meyers' random-filter mean was **+$65.3/week**, not zero.
- ⚑ **Use one denominator on both sides — all OOS periods, zero-trade weeks counted as zero.**
  [M25]'s published significance divides the filter's total by the 446 weeks it *traded* while
  defining the null over all 517. Same data, consistent denominator, `K·p` goes **0.049 →
  2.234** — significant to not significant. Verified; see SPEC §6.5 and §9-K. This single
  choice decides the Unit 9 gate, so it is pinned here rather than discovered later.
- ⚑ **A literal comparison counter.** A file the evaluator increments on every OOS-touching
  run, multiplied in at report time. §8's cheap A/Bs (history depth, `session_reset`, feed,
  long/short, both filter ambiguities) are all looks at the same OOS columns, ⚑ as are the two
  accidental looks at the withheld tail (Unit 3's second-moment one, and Unit 5's review
  probe, which saw per-combo trades and a filter selection over 2 windows), and "it's a
  35-second run" is exactly how the count gets lost. ~10 lines, and the only thing standing
  between this project and the failure mode §5 is named after.
- ⚑ **Report lag-1..4 autocorrelation of `osnp`.** IS windows overlap by 23 of 30 days, so
  consecutive selections are autocorrelated even though OOS weeks tile cleanly. Meyers' own
  `wpr=8`/`lpr=9` show run structure. The `t` statistic assumes i.i.d. and is therefore
  inflated — treat it as descriptive. If autocorrelation is non-trivial, block-bootstrap.
- Verify (not re-apply) the §1.3 cost model. ⚑ Sanity check is
  `trades × shares × ($0.01 + SEC/TAF)`, not slippage alone.
- ⚑ **The cost constant is what needs verifying, and it is wrong in two ways** (SPEC §3.2).
  `$0.017/share on sells` folds two structurally different fees into one notional-scaled
  number: FINRA's TAF is charged **per share** and does not scale with notional at all, while
  the SEC Section 31 fee is charged on notional at a rate that is reset periodically. So one
  of the two carries the wrong price sensitivity across the whole sample, and a single pinned
  rate is wrong in every year but one. ⚑ The actual TAF figure and the Section 31 rate history
  are **external facts this repo has not measured** — that is a source lookup against FINRA's
  schedule and the SEC's Section 31 advisories, and it belongs to this unit. Split the two,
  date-key the statutory one, and check the direction of the error before trusting any
  after-cost figure: `toNP > 0` is one of the three decision-gate conditions.
⚑ **From Unit 6: the grid is ~3200 wide, not 4312.** Distinct trade sets per window
measure **2640–3652 of 4312 (median 3202)** over 24 real windows, so a multiplier built on
4312 overstates the search by ~35%. Read it per window off the stored table as the count of
distinct metric rows — validated against hashing the real trade lists to within 0.22%.

⚑ **From Unit 5: two things the report must not misread.** `%P`, `eqR2`, `eq2R2` and
`ktau` are all stored ×100 (SPEC §6.6) — `ktau` signed, so `[-100, 100]`. And our `dd`/`llt`
are **net** where [M25 Table 1]'s `odd`/`ollt` are **gross**: Meyers subtracts cost as a
post-hoc weekly aggregate (`NOnp$13 = osnp − ont*13`) while ours is inside `_simulate` per
trade. Its 01/07/15 week reads `osnp` −2020 with `odd` −2020, where the net figure is −2046.
Any numeric parity check against the published table has to account for that before calling
a difference a defect.

- Produce a Table-1-shaped report (same columns) plus the equity curve with its 2nd-order fit.

**Done when** a filter run against shuffled OOS columns comes back insignificant; ⚑ **plus a
shuffled-alignment falsification** — apply window *k*'s selected `N/vup/vdn` to window
*k+50*'s OOS ⚑ **under window *k+50*'s own `xmult`**. Carrying *k*'s multiplier across would
mis-scale the thresholds by up to 20x (SPEC §1.2.1), and the test would then "degrade" for a
reason that has nothing to do with the filter.
If that does not degrade, the filter is a static parameter prior, not a response to the IS
window, and column-shuffling would never have caught it.

⚑ **The withheld tail is opened exactly once, as the last action of the project** — after the
filter is frozen. Rev 1 had Unit 9 report it *and* Unit 10 validate against it, which spends
the tail before using it and leaks through the human decision §4 puts between every unit.
Meyers ran it the other way (p.8): WFME64 on 517 weeks, filter frozen, then one look at the
29 excluded weeks.

**Decision gate — three conditions, not one:** ⚑

1. bootstrap `Prob × comparisons_counted < 0.05`;
2. `toNP > 0` **after costs**;
3. risk-adjusted return **exceeds long-only SPY** over the same OOS weeks.

Rev 1's single significance test would have passed a strategy that beat a random filter while
losing money after costs, or that made money purely through long bias in a market that
quadrupled. If the gate fails, **stop and reconsider before writing any live code.** This is
the gate the whole plan exists to reach.

⚑ **Power check, do this first — it is free.** With ~470 OOS weeks, reaching `t = 2` needs
`μ/σ >= 2/sqrt(470) ≈ 0.092` per week. Estimate the expected per-share edge against that
threshold *before* committing to Units 10–13.

---

### Unit 10 — Region portfolio, not filter search ✅ **shipped — the tail was opened once and the hypothesis PASSED**

**Shipped** in `pwfo.py`: `REGION`, `sym_dir()`, `window_notional()`, `region_mask()`,
`region_weeks()`, `run_region()`, `_align()`, `_bps_row()`, `region_report()`,
`load_tail_oos()`, `tail_windows()`, `region_verdict()`, `count_look(unique=)`, and three CLI
arms — `python pwfo.py region` prints the deliverable off the stored pre-tail tables,
`python pwfo.py <SYM>` runs the PWFO for a second symbol into `pwfo_<sym>/`, and
`python pwfo.py tail` is the pre-registered one-shot.
**116/116 tests**, `ruff check .` clean, **34 of 34 mutations killed**, 5 tests added.
`pwfo_qqq/` holds QQQ's 525 pre-tail windows on the same 525 Fridays as SPY's, byte-for-byte
window alignment asserted. The withheld tail was not opened — sha256 `629122d198949afd…`
unchanged, and `UNIT9_REPORT.txt` still reproduces byte-identically.

**Both done-when conditions are met.** Off `pwfo/pwfo_oos.npy` through the shipped path:

| | combos | weeks | gross $ | net $ | t | Sharpe |
|---|---|---|---|---|---|---|
| `run_region()`, SPY | 1620 | 525 | 175.36 | **124.29** | **2.3865** | 0.75 |
| the diagnosis scratch script | 1620 | 525 | 175.36 | 124.29 | 2.39 | 0.75 |

QQQ, the same 1620 combos and the same 525 weeks, never refit: gross 192.43, **net 147.69,
t = 2.65**.

⚑ **The bps figures moved by ~2%, and the shipped ones are the right ones.** The diagnosis
sized each week on the mean close of *that OOS week*; `window_notional` recovers the price
level from the stored `cost`, which `window_cost` charged at the **31-day IS mean close** —
stale by construction and in the direction that cannot leak. ⚑ It is *not* the Friday close a
live position would be sized on: over the 525 pre-tail SPY windows `(IS mean − Friday close) /
close` runs **−9.99% to +26.03%**, median −1.04%, so individual weeks move by up to 44 bps and
the largest errors sit in exactly the high-vol weeks the anti-goal says carry the payoff.
Immaterial to the statistic — blended `t` is **2.7726** on the IS mean against **2.7744** on the
Friday close, sum 3834 against 3913 bps — so this is a labelling correction, not a numbers one.
Everything scale-free is unchanged.

| | sum bps | bps/wk | sd | Sharpe | t | h1 | h2 |
|---|---|---|---|---|---|---|---|
| SPY | 30.3% | 5.76 | 55.13 | 0.75 | 2.40 | 10.8% | 19.5% |
| QQQ | 46.4% | 8.84 | 71.99 | 0.89 | 2.81 | 16.7% | 29.7% |
| **50/50** | **38.3%** | 7.30 | 60.36 | **0.87** | **2.77** | 13.7% | 24.6% |

Leg correlation +0.800; max drawdown −8.3% of notional; worst week −2.10%; 49.3% of weeks
positive; **four** negative years of eleven — 2016 −1.2%, **2017 −0.2%**, 2019 −3.5%, and a
−1.9% stub 2026. ⚑ "sum bps" is the **arithmetic** sum of weekly bps, not a compounded
return: compounding the same 525 weeks gives 45.3% against 38.3%, and −8.04% drawdown against
−8.33%. Understating is the conservative direction and the whole report is internally
consistent in arithmetic space, but the number is a sum, not a return.

- ⚑ **`run_region` hoists zero IS columns.** `load_tables([])` still runs the shape check at
  the file boundary and returns an empty `cols`, so "the selection step
  is gone" is a signature rather than a claim — pinned by a test that fills `pwfo_is.npy`
  with nan and requires the aggregate not to move. Hoisting even `nT` fails it.
- ⚑ **Equal weight is the *mean* of the six OOS columns**, not the sum: holding 1/M of each
  combo makes net, trade count, winners, largest loser and drawdown all average. `LLTr` and
  `eqDD` therefore read portfolio-weighted, not worst-leg, and `aggregate` inherits that.
  `selected` is true on every window by construction — no screens, so SPEC §6.4's case 1
  cannot arise and `n_sel == n`; case 2 still can.
- ⚑ **`Prob` is absent from `region_report` and that is the point.** SPEC §6.5's `K`
  multiplier corrects for hypotheses examined; the pre-registered claim contains no
  selection to correct for. `comparisons.json` is untouched at **K = 13**.
- **The region is a cube, checked over all 4312 combos.** `n >= 5` and each of `vup`, `vdn`
  in [0.75, 2.75] *separately* — 20 N × 9 × 9 = 1620. All three bounds inclusive, pinned in
  both directions; a band-around-the-diagonal reading would drop the corners.
- **QQQ's cache stopped at 2026-02-27, one day before `TAIL_START`**, so its first run
  withheld 0 windows and wrote no `pwfo_tail.npy` — the diagnosis fetched QQQ pre-tail only.
  ⚑ **Writing a tail table is not opening one.** Unit 7 wrote SPY's and the project has
  treated the tail as shut for three units; the `__main__` arm prints the withheld *count*
  and reads `xmult` and `cost` off `pre` alone. QQQ was therefore refreshed through the tail
  and re-run full-range on the same footing as SPY. The opening is the *read*, below.
- PLAN §5 pinned "multi-asset abstraction" at *a second symbol is actually traded*. It is,
  and the whole of the abstraction is `sys.argv[1]` and `sym_dir`. `pwfo_*/` is gitignored.

**Still open, and it is the only thing left in this unit:** the withheld tail, against the
pre-registration below, exactly once. Not run. It needs QQQ bars fetched through
2026-08-31, a QQQ PWFO over the full range, and `run_region` pointed at both tail tables —
and every one of those steps reads data no measurement above has seen.


Rev 1 of this unit was a bounded filter search, unblocked by Unit 9's failed gate and flagged
as the unit most likely to manufacture a false positive. **It is cancelled.** Diagnosis
(2026-09-09) found the failure was not in the strategy and not in the data — it was in the
selection step Unit 10 proposed to search harder. Searching harder is the one thing the
measurements rule out.

#### What Unit 9's gate actually measured

⚑ **Two of the three gate conditions test something other than "does RMedV work".**

**Condition 3, "beat long-only SPY", is unreachable by construction.** Of SPY's +505.31/share
over the sample (+485.45 pre-tail), only **+103.79 accrues between 10:00 and 15:55**; the
other **+381.66, 79%, is overnight and pre-10:00**. SPEC §2 flattens at 15:55 and never holds
overnight, so the gate asked for a return the rules forbid the strategy from touching. The
condition was written to catch "made money purely through long bias" — a real risk, and the
right guard against it is the long/short decomposition below, not a benchmark the strategy
cannot access. **Amend the condition to the gated-hours pool (+103.79), or drop it.**

**Condition 1's null is not a null.** `null_moments`' random-filter draw is the equal-weight
whole-grid portfolio, and that portfolio has **gross t = +3.00** in its own right. `K·p`
therefore measures the *filter's selection skill above the grid average*, which is a
worthwhile question and not the one the project was asking.

Condition 2 (`toNP > 0` after costs) stands as written and is met.

#### Measured: the kernel is correct, the grid has edge, the filter has none

The implementation was re-verified independently before anything else, because a broken
estimator would make every number below meaningless:

| check | result |
|---|---|
| [M05 p.2] and [M25 p.2] worked examples | exactly `1.0`, both |
| 4,400 random `(n, t)` pairs on real SPY bars, all 22 n, vs `scipy.siegelslopes` | max diff **3.18e-08**, i.e. float32 storage precision |
| `rmv._simulate` vs an independent from-SPEC-§2 re-implementation, 40,000 bars | **361/361 trades identical, 0.00 P&L difference** |

Equal-weight a parameter region, apply no filter at all, same per-window `xmult` and `cost`,
same 525 pre-tail OOS weeks, read straight off `pwfo/pwfo_oos.npy`:

| region | combos | gross $ | net $ | t | Sharpe | h1 | h2 |
|---|---|---|---|---|---|---|---|
| whole grid | 4312 | 123.49 | 61.48 | 1.51 | 0.47 | 0.48 | 61.00 |
| `n>=5, v in [0.75, 2.75]` | 1620 | 175.36 | **124.29** | **2.39** | 0.75 | 31.40 | 92.89 |
| `n>=14, v in [0.75, 2.75]` | 891 | 195.63 | **157.59** | **2.47** | 0.78 | 49.19 | 108.41 |
| Unit 8's best filter, `CL2` | 1 | 157.47 | 132.11 | 1.39 | — | — | — |

Fifteen region boundaries were tried; every one lands at **t = 2.0..2.9**, so the result is
not carried by where the boundary is drawn. The surface is smooth and physically motivated:
`n=3,4` lose to microstructure noise (gross −105.67 and +2.70), edge climbs monotonically to
`n=21` (+195.02), and `v` peaks at 1.25 and falls both ways — low `v` overtrades into costs,
high `v` starves the sample.

**Not long bias, and this is the measurement condition 3 should have been.** Re-simulating the
891-combo region and splitting by direction: **long +91.55 gross on 909 trades, short +104.08
gross on 910 trades.** The short side earns *more*, on symmetric counts, across a decade in
which SPY tripled.

**The filter is what failed, and it fails for a reason that is not fixable by searching:**

- Per-window Spearman(IS `tnp`, OOS `osnp`) across all 4312 combos: **mean −0.0062, t = −0.45,
  48.2% of windows negative.** IS ranking carries no information about OOS ranking. Not weak —
  zero.
- Median **19 IS trades** per combo per window; **24.8% of combos have fewer than 10**. Ranking
  4312 hypotheses on 19 trades is not an estimation problem that a better metric solves.
- **Picking the IS-best `tnp` each week returns −228.75, t = −2.00** — materially worse than
  random. Winner's curse: the IS maximum is the combo with the largest positive noise.
- Selecting on *trailing OOS* results — which is cheating, and strictly more data than any
  filter has — stays negative through a 52-week lookback and only reaches +151.17 at 104
  weeks, still below the static region's +157.59.

⚑ **Conclusion: at this trade frequency, weekly parameter selection has negative expected
value. A bounded search over 2000 filters is a search for a better way to do the thing that
does not work.**

#### Measured: the asset question, answered and partly refuted

The SPY-derived region, unchanged, on seven other symbols over the same 525 weeks:

| sym | net bps / 10yr | t | note |
|---|---|---|---|
| QQQ | **+6502** | **+2.87** | clean |
| SPY | +2762 | +2.39 | clean |
| IWM | +231 | +0.12 | clean |
| USO | −270 | −0.11 | crude proxy — nothing |
| TLT | −1092 | −0.84 | clean |
| GLD | −1176 | −0.72 | clean |
| TSLA | +15570 | +2.24 | split-contaminated cost model |
| NVDA | +1400 | +0.18 | cost model unusable (4:1 and 10:1 splits) |

- **SPY being an ETF rather than a futures contract is not the problem.** SPY works; QQQ works
  **2.4x better per dollar of notional** (Sharpe 0.89 vs 0.75).
- **"Meyers used CL" does not explain the result either** — the crude proxy gives nothing.
  Caveat: USO is a poor CL proxy (contango roll drag, 8:1 reverse split 2020). A real test of
  [M25]'s CL claim needs CL futures bars, which Alpaca does not carry. Recorded as unresolved,
  not as refuted.
- `$0.01`/share slippage is one half-spread per fill. That is correct for SPY, QQQ, IWM, TLT
  and GLD, whose books are penny-wide across their whole price range. It is **not** correct for
  split-adjusted TSLA and NVDA, where split-adjusted prices make per-share cost wildly wrong in
  bps — NVDA's 79% cost-to-gross ratio is a model artifact. Those two rows are not evidence.
- The edge is specific to the large-cap US equity index complex. **NQ, not ES, is the futures
  analogue worth pricing** if futures are ever revisited.
- ⚑ **SPY options are ruled out.** Net edge is **1.3 bps per trade**; 0DTE SPY bid/ask runs
  1–3% of premium, 100–300x the edge, before vega and theta enter as uncontrolled exposures.

#### What this unit builds instead

**Do** ✅ Freeze the region as a *static prior*, trade it as an equal-weight portfolio, and
run the existing PWFO machinery with the selection step removed. No search, no generated filter
space, no addition to `K` beyond the pre-registration below.

⚑ **Superseded — the diagnosis figures, kept for the record.** These size each week on the
mean close of *that OOS week*; the shipped `window_notional` uses the IS mean, which is the
non-leaking direction, and the shipped table above is the one to cite. Differences are ~2% on
the level and nil on everything scale-free.

| | cum ret | bps/wk | sd | Sharpe | t | h1 | h2 |
|---|---|---|---|---|---|---|---|
| SPY | 31.0% | 5.90 | 56.45 | 0.75 | 2.39 | 10.6% | 20.3% |
| QQQ | 47.3% | 9.02 | 73.43 | 0.89 | 2.81 | 16.3% | 31.0% |
| **50/50** | **39.2%** | 7.46 | 61.71 | **0.87** | **2.77** | 13.5% | 25.7% |

Leg correlation 0.80; max drawdown −8.7% of notional; worst week −2.31%; two losing years in
eleven. (Shipped: +0.800, −8.3%, −2.10%, **four** losing years.) Intraday only, so overnight risk is zero and Reg-T day-trading leverage applies to the
whole of it.

⚑ **Pre-register this, verbatim, before running anything against the tail:**

> The RMedV edge is a broad property of the parameter region, not a selectable point.
> Equal-weight every combo with `n >= 5` and `vup, vdn` both in `[0.75, 2.75]`, refit `xmult`
> per IS window as Unit 7 already does, apply no IS filter, and trade SPY and QQQ at 50/50.
> Success is `toNP > 0` after costs and a positive `t` on the withheld tail's weekly series.

One hypothesis, one direction, one number. Add **exactly one** entry to `comparisons.json`.

**Done when** ✅ **both met.** `run_region()` returns net 124.29 at t = 2.3865 off the stored
`pwfo/` tables through the shipped code path, and `pwfo_qqq/` holds QQQ's 525 windows on
the same Fridays.

#### Adversarial review (PLAN §4) — 22 findings, 10 fixed, 8 deferred, 0 rejected

⚑ **Four findings blocked the one-shot and are fixed.** None was an arithmetic error; all
four were about the *arm* rather than the numbers, which is exactly the class a unit-level
test does not catch:

1. **The holdout was read before the look was recorded.** The arm loaded both tail tables and
   only then discovered a leg had none — demonstrated on synthetic dirs: both tables opened,
   `ValueError` raised, `count_look` never reached, `comparisons.json` still at 13. An
   unrecorded spend. Fixed: `tail_windows()` validates both legs from `pwfo_index.json`
   **only** — present, non-empty, table on disk, identical Friday lists — and `count_look`
   runs before the first `load_tail_oos`.
2. **The verdict printed last.** Any raise inside the §6.3 formatter after `count_look` left
   the holdout spent, `K` incremented, and nothing on stdout. Fixed: the pre-registered
   `toNP`/`t`/PASS-FAIL line prints first, the table second.
3. **`count_look` is idempotent by key**, so a fix-and-rerun of the tail was free — measured
   14/14/14 over three calls. Idempotence is right for nine deterministic filters and wrong
   for a holdout that can be looked at once. Fixed: `unique=True` appends `#0`, `#1`, … so a
   second read is visible in the ledger. Unit 10's tail arm is the only caller.
4. **`comparisons.json` was never committed**, while `count_look`'s docstring and §3 Unit 9
   both claimed it was. A `git clean` would have reset `K` to 1 with no error. Fixed by
   committing it; both claims corrected.

Six more fixed in the same pass: the alignment guard moved into `region_verdict` (it guarded
the printed table and not the number the pre-registration is judged by — two legs a year
apart blended silently to `passed: True`); the report header now describes each leg's own
region instead of printing the frozen `REGION` regardless; `sym_dir` gained a test (the mutant
that ignores its argument sends QQQ's PWFO into `pwfo/` and destroys the sealed table);
the regeneration arm now **refuses to overwrite an existing `pwfo_tail.npy`** without
`--force`, because a warning in a handoff note is not a guard; the CLI dispatches on exact
`argv[1]` (`"tail" in sys.argv` fired on `python pwfo.py SPY tail`, and arm precedence made
`python pwfo.py tail report` silently run the *report*); and the full §6.3 block on the
holdout now sits behind `--full`, since it is ~70 numbers where one was pre-registered and it
is precisely the seed for hypothesis #2.

⚑ **The review's mutation pass was the useful half.** My own 17 mutants all died; the
reviewer's found **nine survivors** in surface I had not probed — all three of
`load_tail_oos`'s integrity guards (nan, shape, row order) deletable with the suite green,
`sym_dir` untested entirely, and four of `_bps_row`'s seven columns free to change
(`sqrt(52)`→`sqrt(252)`, `sqrt(n)`→`n`, `ddof=1`→`0`, `//2`→`//3`) — including the printed
`t` that sits beside the verdict's `t` with nothing cross-checking them. **Now 34 of 34.**

**Deferred, recorded not fixed:**

- **`LLTr` no longer denotes a trade.** Under equal weight the weekly `ollt` is the mean over
  1620 combos of each combo's largest loser, so §6.3 col O (SPY −14.18) is not a loss any leg
  took. `odd` is the same and is inert. The rest of §6.3 survives cleanly: `toGP` is exact by
  linearity, and `aoTr`, `ao#T`, `%Wtr`, `oW|oL` stay meaningful as ratios of sums.
- **Col G (`n_trd`) is structurally 525.** `ont` is a mean over 1620 combos and is zero only
  if every one is silent; measured minimum 0.206 (SPY). SPEC §6.4's case 2 is unreachable at
  M = 1620 while remaining correct in principle.
- **Modelled cost is an upper bound, in the result's favour.** The portfolio charges
  `cost · mean(ont)`, i.e. as if all 1620 fractional legs fill independently; a real
  implementation nets them into one order and pays the spread on the net change only. Gross
  is exact. Same direction for the denominator: a 1/M-per-combo book has gross exposure
  ≤ 1 share, so `window_notional` over-states capital deployed.
- **The pre-tail path names its OOS columns from `rmv`, positionally**, while the tail door
  looks them up by name from the index — the opposite of `load_tables`' own stated contract.
  Inherited from Unit 8's `evaluate`; no live failure while one `run` writes both files.
- **`_align` accepts a common *prefix* only.** A leg whose cache *starts* later is a
  legitimately blendable pair it refuses. Intersect on Friday if that ever arises.
- **`window_notional` re-derives a price level from current module constants against a stored
  cost.** Numerically exact today (max relative error 4.4e-08 over all 525 real costs), but
  `pwfo_index.json` stores `cost` without `SLIP`/`SEC_TAF_PER_DOLLAR`, so the day SPEC §3.2's
  acknowledged-broken fee model is fixed, every stored table's bps silently rescales. Stamping
  the constants into the index would require regenerating `pwfo/` — which destroys the sealed
  tail — so it waits for the next full run.
- **The withheld period's price level has been on disk since Unit 7.** `pwfo_index.json`
  carries all 26 tail windows' `xmult` and `cost`, and `window_notional` now turns the second
  into a 2026-03..08 mean close in one line. Nothing has printed them and the hypothesis does
  not depend on price level, but "the tail is sealed" means the `.npy`, not the index.
- **`--force` and `--full` are the only flags**, parsed by membership. A third would want a
  real parser, which PLAN §5 pins at "a third caller appears".

#### ⚑ The withheld tail, opened once on 2026-09-10. **PASS.**

    ⚑ WITHHELD TAIL OPENED -- 26 weeks, 2026-02-27..2026-08-21, K = 14 after this look

    pre-registered test (PLAN §3 Unit 10): equal-weight n >= 5, vup, vdn in [0.75, 2.75],
    no IS filter, SPY+QQQ 50/50
      toNP +240.0 bps over 26 withheld weeks, t +0.799  ->  PASS

Both pre-registered conditions hold: `toNP > 0` after costs, and `t > 0`. `comparisons.json`
gained exactly one entry, `unit10_region_tail#0`, K = 13 → **14**. The full §6.3 block was
**not** printed — it stays behind `--full`, so nothing beyond these two numbers has been read
off the holdout and the deferred vol-state hypothesis is still clean.

⚑ **What this does and does not establish.** The honest reading, stated before anyone builds
on it:

| | pre-tail, 525 wk | withheld, 26 wk |
|---|---|---|
| bps/week | 7.30 | **9.23** |
| sd | 60.36 | 58.9 |
| annualised Sharpe | 0.87 | **1.13** |
| t | 2.77 | 0.799 |

- **The holdout did not degrade — it came in slightly ahead.** If the pre-tail effect had held
  *exactly*, 26 weeks would have produced `t = 0.615`; the observed 0.799 is a shade above
  that. Overfitting's signature is decay toward zero out of sample, and there is none here.
- ⚠ **This is not a significant result and could not have been one.** `t = 2.06` is the bar
  at n = 26, and an 0.87-Sharpe strategy needs roughly **150 weeks** to clear it. The
  pre-registration set sign-based criteria precisely because a 6-month holdout has no power
  to do more — so "PASS" means *consistent with the edge persisting*, not *the edge is
  confirmed*. Anyone quoting this as validation is quoting it wrong.
- **What the test genuinely bought** is that the region was frozen in writing before the data
  was read, K is recorded, and the one number that came back was the one named in advance.
  That rules out the failure mode Unit 9's diagnosis was about — a result manufactured by
  selection — which no amount of further pre-tail work could have done.
- The contamination note stands and is not retired by the PASS: the region was chosen by
  reading SPY's own pre-tail OOS marginals across at least fifteen boundary variants, and
  those looks are not in `comparisons.json`. The 26 weeks are clean; the 525 are not.

**The tail is now spent.** There is no second withheld set. Any further parameter or region
question is answered on data that has been read, and Unit 11 onward is forward-testing —
paper first, per §3 Unit 13's go/no-go.

**Anti-goal** Do not re-derive the region from the tail, and do not widen the search when the
tail disappoints. Do not add money management to rescue a result: sizing reshapes a
distribution, it cannot create edge, and naive vol-targeting would actively hurt here — the
payoff is long-volatility (skew +1.99; **2018 +13.6% and 2022 +12.5% against 2017 −0.2%,
2019 −3.5%** — shipped figures; the diagnosis had +13.7/+13.0 against +0.1/−3.3, and 2017's
sign was the wrong way round), so scaling down into vol cuts exactly the profitable weeks.

**Deferred, pre-registered, not built:** a trailing-realized-vol *state* filter. Weekly net by
trailing-vol quintile runs **−3.3, +2.5, +6.3, +12.2, +11.2 bps**, monotone; skipping the
bottom quintile lifts Sharpe 0.75 → 0.95, the bottom 40% → 1.10. This is a *state* filter, not
a *parameter* filter — realized vol is persistent and forecastable, combo rank is not, which is
exactly why one has a chance and the other does not. It is a second hypothesis and must not be
folded into the first one's tail test.

---

### Unit 11 — Live: weekly refit (`live.py`) ✅ **shipped — re-scoped to the region portfolio**

**Shipped** in a new `live.py`: `LEGS`, `today_et()`, `last_friday()`, `is_span()`, `refit()`,
`write_params()`, `load_params()`, `stored()`, `main()`. In `data.py`, `load_bars(cache=False)`
fetches exactly `[start, end]` off the network and neither reads nor writes the bar cache —
the online half of the offline/online check has to be a second read or it compares the cache
with itself. ⚠ A second read of the **bars** only: the exchange calendar, which sets early-close
gating and so moves `xmult`, is the one input both halves share — `cache/nyse_calendar.json`,
refetched only when it does not cover the span. `params.json` is gitignored.
**120/120 tests** (4 added), `ruff check .` clean, **39 of 39 mutations killed**; the withheld
tail's sha256 is unchanged and `K` is still 14.

    python live.py refit              refit for the last closed Friday, write params.json
    python live.py refit 2025-06-13   dry run: refit off the network, compare to Unit 7, never write

⚑ **Re-scoped, not built as written.** Rev 2 (kept below) refits the grid on one IS window,
applies the frozen filter and writes the chosen `N/vup/vdn`. Unit 10 cancelled selection — IS→OOS
rank correlation t = −0.45, the IS-best pick t = −2.00 — and what passed the tail is the 1620-combo
region, equal weight, SPY+QQQ 50/50, **no IS filter**. There is no row to choose. What that
strategy still refits every week is the *scale*: each leg's `xmult` (SPEC §1.2.1, a 20× range
across windows) and `cost` (SPEC §3.2), each off that leg's own 31-day IS span. The grid run is
dropped because nothing reads its 4312 rows.

| | Rev 2 | shipped |
|---|---|---|
| refit | RMV → grid → frozen filter → one row | RMV → `xmult`, `cost`, per leg |
| file | `{N, vup, vdn, filter, as_of, is_metrics, xmult, git_sha}` + `cost` | `{as_of, region, legs: {SPY, QQQ: {xmult, cost, is_bars}}, git, written}` |
| legs | one symbol | both, each off its own bars; a one-leg file is refused, not traded at 100% |
| staleness | refuse when older than 10 days | valid **only** `as_of+3 .. as_of+7`, its own OOS week |
| failure | no row passes → flat week | refit raises → no file written → flat week |

`region` replaces `N/vup/vdn/filter`; `is_metrics` described a selected row that no longer
exists. `cost` no longer selects anything, which was Unit 4's reason to carry it — it stays in
the file and in the check because Unit 12a's replay charges it per trade and Unit 13 compares
realised fills against it.

**Done when** ✅ **met, bit-for-bit, both ways.**

- **Offline, every window.** `refit(sym, friday, cache=True)` loads the IS span standalone out of
  the cache and builds RMedV and the gate on those 31 days alone. Against `pwfo_index.json`, all
  525 pre-tail windows of both legs: **1050 of 1050 identical** in `xmult`, `cost` and `is_bars`
  — `==`, not `isclose` — and none trips the missing-session guard. ~75 s, so the test samples
  every 4th window (264, ~14 s): a 31-day span puts any single event day inside 4–5 consecutive
  windows, so stride 4 cannot step over one.
- **Online.** `python live.py refit <Friday>` fetches the span off Alpaca: **MATCH on both legs** at
  2020-03-20 (all four circuit-breaker halts inside the span), 2024-11-08 (DST end), 2024-11-29
  (a 13:00 early-close Friday), 2025-03-14 (DST start) and 2025-04-18 (Good Friday — `as_of` on a
  closed market). All five are a network test (two, until the review's #3).
- **The live arm, end to end,** for the current week: `as_of` 2026-09-04, fetched past the end of
  the cache with the calendar refreshed, written, read back — accepted 09-07 through 09-11,
  refused 09-06 and 09-12.

⚑ **Why equality holds, and why it is a test rather than a comment.** `pwfo.run` computes RMedV
and the gate on the full 10-year series and slices each window out; `refit` computes both on 31
days. They agree only because no gated bar's `t` or `t−1` window reaches back across a session
gap — SPEC §3.1's blackout, whose margin is exactly zero. A later session start, a larger
`MAX_N` or an `RMedV[t−2]` term would move `xmult` by a few ulp in some windows and nowhere else,
which only `==` over many windows can see. That invariant is now pinned from the live side too.

- ⚑ **Valid for its own week only.** Rev 2's "older than 10 days refuses" trades the Monday after a
  failed weekend refit — day 10 — on last week's `xmult`, which moves up to 20× week to week while
  `pwfo.run` scores every OOS week on its own. That is a parity break, not staleness. Refused is
  flat, and flat is the backtest's own zero.
- **Review focus, answered.** *Timezone:* `is_span` runs ET midnight to ET midnight,
  `[F−30 00:00, F+1 00:00]`. A UTC midnight is 19:00/20:00 ET the evening before, which cuts
  Friday's whole session — pinned across the 2024-03-10 DST switch by the request bounds and by
  bars planted on each edge. *Sees the coming week:* no, and checked twice — the request stops at
  Saturday 00:00 ET, and `refit` then re-checks the dates of what actually came back and refuses
  a span widened by a day at either end. The default Friday is strictly before today, so a run on
  Friday afternoon refits the *previous* week rather than a half-finished session. *No row
  passes:* cannot happen with no filter. The failure that replaces it is a refit that raises —
  no bars, a missing session, an undefined `xmult` — after which nothing is written, `main` exits
  1 with `REFIT FAILED … FLAT` naming the week, and last week's file is refused on Monday.
- ⚑ **One guard the cache path never needed.** A calendar session in the span with no bars at all
  is refused. The cache re-reads its last 200 bars on every refresh and compares; a one-shot fetch
  is compared against nothing. Intraday holes stay the gate's job, identically on both paths.
- **A dry run refuses a withheld Friday before it fetches a bar.** It exists to compare against
  Unit 7, and the tail windows' `xmult` is a vol-state reading of the spent holdout that Unit 10
  keeps unprinted for the deferred vol filter. The live arm's own IS spans overlap August 2026 for
  the next few weeks; that is trading, not a look at a tail window, and cannot be avoided.
- The live arm appended 2026-09-01..04 to `cache/nyse_calendar.json` (2680 → 2684 sessions) —
  gitignored, append-only, and nothing counts it.
- ⚠ **`pwfo.py report` no longer reproduces `UNIT9_REPORT.txt` byte-for-byte, and should not.**
  The tail look took `K` from 13 to 14, and the report prints `K`: 12 lines differ — the header,
  `K*p`, `1-(1-p)^K` and three gate rows — while every `K`-free number (bootstrap, `toNP`, `z`,
  mu/sigma) is identical. The regression check is now "differs only in `K`-dependent lines".

**Carried to Unit 12, which needs the same re-scope before it starts** — Rev 2's 12a/12b text
below still assumes one row:

- Parity is per combo: 1620 combos × 2 legs, not one `N/vup/vdn`. The position live holds is the
  mean of 1620 signals in {−1, 0, +1}, i.e. fractional. Rounding it to shares is both a sizing
  decision and a parity gap, and Unit 10 already recorded that modelled cost is an upper bound
  because 1620 fractional legs net into one order.
- Sizing: the backtest's bps are on the IS-mean close (`window_notional`); a leg sized on Friday's
  close differs from it by −10%..+26% in a given week (Unit 10).
- SPEC §3.2's fee model is still "required before any live order is sent". This unit sends none.

#### Adversarial review (PLAN §4) — 5 findings + 1 of mine: 3 fixed, 2 rejected, 1 recorded

The reviewer re-ran everything that carries a number rather than trusting it: the full
unsampled sweep (**1050/1050**, 0 exceptions, 56 s), all five online Fridays (3 by hand, since
only 2 were in the suite), the suite, ruff, and the tail's sha256. It confirmed that
`params.json`'s three-number `region` plus per-leg `xmult`/`cost` rebuilds all 1620
`(n, vup, vdn)` thresholds through `decode` and `rmv.threshold`, so Unit 12 is missing nothing,
and that `load_bars(cache=False)` reduces exactly to the old logic for every existing caller. It
touched no repository file.

1. **Fixed — `load_params` never looked at `is_bars`** (filed BREAKING). "Refuses on any doubt"
   was false for a third of every leg: `-999`, `"banana"`, `None`, `nan`, or the key deleted, all
   accepted. `is_bars` trades nothing — so this was a false claim, not a wrong trade — but it is
   the file's record of what the fit saw. Now a positive `int`, and `bool` is refused explicitly
   because `True` is an `int`. 8 refusal cases added.
2. **Rejected — a mistyped dry-run date raises instead of exiting 2** (filed SEVERE). Only the
   manual dry run can reach it; the scheduled arm computes its Friday. The traceback names the
   unparseable string and exits non-zero, and nothing is fetched or written. A handler would
   change the message, not the outcome.
3. **Fixed — three of the five online Fridays were prose, not tests** (filed SEVERE). All five
   are in `test_unit11_online_refit_matches_the_offline_table` now, +2 s. DST end and the early
   close had no online coverage at all.
4. **Rejected, and pinned — "the second `load_calendar` is redundant"** (MINOR). It is the only
   read that covers a missing *last* session on a live date. `load_bars` refreshes the calendar
   just to the last bar that came back, so a fetch missing Friday is also missing Friday from
   that refresh, and the missing-session guard passes silently. The reviewer's fix would have
   removed the guard's one live-only case. A test now builds exactly that — a calendar cache
   ending Thursday and a fetch missing Friday — and the "read over the bars" mutant dies on it.
5. **Recorded — bar amendments are spot-checked online at 5 Fridays, not swept** (MINOR). An online
   sweep is 1050 fetches; the cache's 200-bar overlap re-read already catches a retroactive change
   on every refresh. No change.
6. **Fixed, mine — a Friday-evening schedule goes flat every week.** On a Friday the last *closed*
   Friday is a week old, so the file written is for the week ending that night and Monday refuses
   it. Every week, with no failing job anywhere. The no-date arm now exits 2 on a Friday and
   says to run on Saturday, so a bad schedule fails on its first run instead of on the Monday
   after.

**39 of 39 mutations killed** — my 35, which first came back 31/35 and cost four test gaps
(the no-bars message, a non-Friday refit, `stored()` handing out tail windows, a dry run fetching
before its pre-check), plus 4 on the fixes above. The reviewer found its two real gaps by
reading, not mutating.

#### Rev 2's Unit 11, superseded — kept for the record

**Do** Fetch the last 30 calendar days → RMV → grid on that one IS window → apply the frozen
filter → write `params.json`: `{N, vup, vdn, filter, as_of, is_metrics, xmult, git_sha}`.

**Done when** a dry run against a historical date reproduces the exact `N/vup/vdn` **and the
exact `xmult`** that Unit 7's PWFO chose for that same window — ⚑ the multiplier is refit
weekly and spans 20x across windows, so a gate differing by a few bars between the full-sample
slice and a fresh 30-day fetch selects a different row — **the offline/online equivalence check**, ⚑ now meaningful
because `adjustment="split"` makes the two price series identical. Params fall inside the grid.
A `params.json` older than 10 days refuses to trade.

⚑ **From Unit 4: `params.json` must carry `cost` as well as `xmult`,** and the offline/online
equivalence check must compare it. `cost` selects the row (Unit 7), so a live loop that
reconstructs `N/vup/vdn/xmult` correctly but re-derives `cost` differently is not reproducing
the same decision.

**Review focus** Timezone of "last 30 days"; whether the refit sees the coming week; what
happens when no row passes (answer: flat week, logged loudly).

---

### Unit 12a — Offline parity harness ✅ **shipped — re-scoped to the region portfolio; parity holds on every session live could know**

**Shipped** in `live.py`: `ET`, `RING`, `Book`, `FakeBroker`, `replay()`, `trades()`, `parity()`,
and a `parity` arm on `main`. No other module changed. **122/122 tests** (2 added), `ruff check .`
clean, **29 of 29 mutations killed** (28, plus 1 on the review fix); the withheld tail's sha256
is unchanged and `K` is still 14 — nothing here opens `pwfo_tail.npy`, and `load_tables` drops
the index's tail rows before the harness sees them.

    python live.py parity [SYM ...]   every pre-tail OOS week, bar by bar, against Unit 7; ~52 s a leg

**Done when** ✅ **met, all four conditions, every pre-tail week, both legs:**

| | weeks | trades identical | excluded, on the unknowable sessions | netted turnover ÷ charged |
|---|---|---|---|---|
| SPY | 525 | **3,958,490** | 5,065 on 2019-08-12, 2020-03-18 | **83.3%** |
| QQQ | 525 | **3,937,717** | 3,262 on 2016-02-22, 2020-03-18 | **82.4%** |

1. **Gate.** Live's per-bar gate equals `bars.gate` on every OOS bar of both legs — early closes,
   Good Friday weeks, both DST regimes, and the 2020 circuit breakers with their blackout
   reopens.
2. **Reference.** All six OOS metrics of every region combo, off the reference trades, equal the
   stored `pwfo_oos.npy` row float32 for float32 — **1,701,000 rows** (525 × 1620 × 2). "Identical
   to `rmv.simulate`" therefore means identical to what Unit 7 *wrote*, not to a re-run that
   could have drifted along with the replay.
3. **Trades.** `(entry, exit, dir, net)` by `array_equal`, per combo, on every session but four;
   **0.13%** of SPY's trades and **0.08%** of QQQ's excluded, all on sessions derived from the
   bars before any trade is compared. ⚑ Each of the four actually differs — the test requires it
   on the named weeks, and the review measured it over all 525 weeks of both legs — so none is
   exempted for nothing. All four are one shape, the next bar did not come on time (table below):
   - **SPY 2019-08-12** — the feed stops at 15:30 (91 bars). The clock says 15:35 is open, so live
     may enter on 15:30; the backtest reads the next bar, Tuesday 08:00, as ungated and does not.
   - **2020-03-18, both legs** — 13:00 and 13:05 never print (SPEC §2.1 B's halt). The backtest
     exits at 12:55's close; live learns of the halt only when 13:10 arrives, and exits there.
     Both reopen on the blackout at 15:10 and agree again from that bar.
   - **QQQ 2016-02-22** — 10:30..11:05 never print (88 bars). Same shape: out at 10:25 against
     11:10, and both reopen at 13:10.
4. **Broker.** Gross off the `FakeBroker`'s fills equals the book's trade gross **exactly** in all
   1050 weeks, and the position is 0 at every week's end. ⚑ Netted turnover is **83.3% / 82.4%**
   of the per-combo turnover the backtest charges `cost` on: Unit 10's "modelled cost is an upper
   bound" is measured, and by ~17%. That is fills at the close; 12b's real fills are what test it.

⚑ **What the exclusion actually removes — measured, not asserted.** Each unknowable session's
trades split at `t*`, the gated bar whose successor came late:

| session | closed before `t*` | open across `t*`: exit differs | entered on `t*` | entered after the reopen |
|---|---|---|---|---|
| SPY 2019-08-12 (`t*` 15:30) | 716, identical | 1,006 of 1,015 | 9, live only | none |
| SPY 2020-03-18 (`t*` 12:55) | 912, identical | 1,522 of 1,522 | 0 | 900, identical, from 15:10 |
| QQQ 2020-03-18 (`t*` 12:55) | 877, identical | 1,503 of 1,550 | 47, live only | 709, identical, from 15:10 |
| QQQ 2016-02-22 (`t*` 10:25) | 0 | 90 of 90 | 0 | 36, identical, from 13:40 |

That is SPEC §2.1's two effects and nothing else: **B**, a position open across `t*` exits on the
late bar instead of on `t*`; **C**, live enters on `t*` where the kernel skips — and the 9 and 47
trades across `t*` that do *not* differ are exactly those reversals, whose closing leg fills on
`t*` either way. Everything before `t*` and everything after the reopen is identical, so excluding
the whole session is conservative rather than a hiding place. ⚑ Masked by **entry** date: SPY's
2019-08-12 positions exit on Tuesday, and exit-date masking breaks (the review's probe).

**Budget** (§2.4): `Book.on_bar` **121 µs** mean on a gated bar, p99 217 µs, max 449 µs, 9 µs
ungated; `Book()` 6 ms once a week — ~5× inside 1 ms. ⚑ **Not** allocation-free: ~10 numpy
temporaries a bar, marked `ponytail:` in the code. The zero-allocation line was written for one
combo, at one bar per five minutes it buys nothing, and an njit step is the upgrade path.

⚑ **Why this is not a function compared with itself.** The book and Unit 7 share `rmv.threshold`,
`rmv.rmv_all_n` and `pwfo.decode`, which check 2 anchors to what Unit 7 wrote — and, ⚠ first
written here as "nothing else", **`cache/nyse_calendar.json` and `data`'s gate constants**, which
nothing anchors. Check 1 compares two gate computations off one calendar, so a wrong early close
would pass it: Unit 11's disclosed caveat, again. Everything below is independent:

| | Unit 7 (`pwfo.run` → `_simulate`) | `Book` |
|---|---|---|
| RMedV | one full-sample matrix, sliced per window | recomputed every bar on a 25-bar ring |
| gate | `build_gate` over the whole series | rebuilt per bar: clock, calendar close, 120-minute span |
| last gated bar | reads `gate[t + 1]` | the clock's `minute + 5` |
| state | a scalar per combo inside a kernel loop | 1620 `int8` states, vectorised, one bar at a time |
| exit on the gate | booked at `close[t − 1]` when the gate goes 1→0 | goes flat on the last gated bar |
| trades | emitted by the kernel | reconstructed from state changes |

The test, `test_unit12a_replay_matches_unit7_trade_for_trade`, runs eight named weeks — one per
event class, because OOS weeks tile and a stride steps clean over any single day — on both legs:
**134,784 trades**, ~10 s. It pins the unknowable sessions by name and requires each to differ.

**Mutation pass: 28 of 28 killed, and every parity check kills something of its own** — which is
the evidence that none of the four is decorative.

- *Gate check:* the 10:00 or the 15:55 bound moved a bar (5 bars each), the blackout a bar short
  (2 bars, 2020-03-06) or gone (43), early closes ignored (26 bars, 2024-11-22), the clock in UTC.
- *Stored-table check:* thresholds at `n + 1`, `vup`/`vdn` swapped, the RMedV row off by one —
  the three that share an input with the reference, and so exactly the three that trade parity
  alone cannot see.
- *Trade check:* a level rule on either side, `RMedV[t−1]` read as `RMedV[t]`, no hold, no
  last-bar flatten and a bar-early one, entries a bar late, cost added instead of charged.
- *Broker check:* position set instead of accumulated, the target submitted instead of the
  difference, fills at the previous close.
- *The test's own pins:* every 1→0 edge exempted, everything skipped (0 trades compared),
  `differ` inverted, turnover counted from trades instead of fills.
- Two die by crash rather than by a check: a ring of `max(n)`, and ungated bars not flattening.

⚑ **Two survived the parity test**: `>` for `>=` and `<` for `<=`. A float32 RMedV landing
exactly on a float64 threshold does not occur in 1050 real weeks, so 7.9 M identical trades are
blind to SPEC §2's inclusive bounds. `test_unit12a_book_crosses_on_equality_like_the_kernel`
closes it: a $0.25-a-bar ramp is an exact line, so RMedV at n = 16 is exactly
±`threshold(1.0, 1.0, 16)` = ±0.25 on a gated bar, and the book has to trade it as `_simulate`
does. Both mutants die on it.

**Carried to Unit 12b** — the four the re-scope listed, plus two this unit found:

- Rounding units to shares; sizing on Friday's close against the IS mean; SPEC §3.2's fee model;
  a mid-session restart, which has to replay the session from 08:00.
- ⚑ **The 15:55 flatten has to be driven by the clock, not by a bar.** `Book` can only act when a
  bar arrives. On SPY 2019-08-12 the feed stopped at 15:30, no 15:35 bar ever came, and the
  replay held to Tuesday 08:00 — overnight, which SPEC §2 forbids. Replay cannot express a timer;
  live has one. 12b's session-cut flatten and its external 15:58 job are what close this, and
  parity cannot test either.
- ⚑ **The stale-bar guard flattens; it does not halt the session.** Every other gap in 1050 weeks
  reopens on the blackout rule and matches the backtest trade for trade.

#### Adversarial review (PLAN §4) — 6 findings: 2 fixed, 1 rejected, 3 recorded; plus 3 corrections of mine

The reviewer re-ran everything that carries a number, at full scale rather than on the test's
eight weeks: both legs' `parity` sweeps (every headline figure reproduced exactly), 122/122, ruff,
`Book.on_bar` over 3,550 gated bars in ten other weeks (**106 µs** mean, p99 169, max 299), and
all 28 mutants from its own harness against a scratch copy of `live.py`, plus 4 of its own (3
killed). ⚑ It also measured `sessions == differ` over **all** 525 weeks of both legs, which the
test asserts only on the named weeks. It touched no repository file and nothing under the tail.
No BREAKING finding.

1. **Fixed — `trades()` raised a bare `IndexError` on a combo still open on the last bar** (filed
   SEVERE). Unreachable here — every week ends ungated and `Book` flattens on every ungated bar —
   but a slice cut mid-session is exactly what 12b's restart replay would produce. Now a
   `ValueError` naming the open count, pinned in the equality test; deleting the guard brings
   the `IndexError` back and the test dies on it (mutant 29).
2. **Rejected — the parity test returns early when a leg's cache or table is missing** (filed
   SEVERE). The suite-wide convention Unit 5's review already deferred. The skip prints
   `(skipped: …)`, which the handoff's verification step checks for; a partial pass prints it too.
3. **Recorded — `math.fsum` is not shown to be load-bearing** (MINOR): a left-to-right sum ties it
   in every named week. Equality for *any* summation order needs `fsum`; that one order also
   happens to tie on this data is not a reason to drop it.
4. **Fixed, in the docstring — two turnover bases conflated** (MINOR). The bound is against the
   book's own per-combo turnover, `2 · len(live)` (the triangle inequality); the reported ratio is
   against the backtest's, `2 · len(ref)`. They differ only on the unknowable sessions — the 9 and
   47 trades in the table above — and `parity`'s docstring described the one as the other.
5. **Recorded — `calendar.get(date, 0)` is untested** (MINOR; removing the default survived the
   reviewer's mutant). A date missing from the calendar reads as a closed session, the default
   `load_bars` gives `build_gate` too; without it the path raises rather than trades.
6. **Recorded — the `parity` CLI arm has no test** (MINOR): a loop and a format string over the
   function the suite exercises at every check.

Mine, found proofreading while the review ran:

- ⚑ **"Shares `rmv.threshold` and `rmv.rmv_all_n` and nothing else" was false.** Both sides also
  read the calendar cache and `data`'s gate constants, so check 1 cannot see a wrong calendar.
  Corrected above the comparison table.
- **"Both reopen and agree again" was asserted, not measured.** Measured now — the table under the
  done-when — and it holds, narrower than claimed: the divergence is B and C and nothing else.
- The test runs in ~10 s, not ~14.

#### The re-scope, as written before any code

⚑ **Re-scoped before any code, as Unit 11 carried it.** Rev 2 (kept below) replays one
`N/vup/vdn` row against one position. What passed the tail is 1620 combos × 2 legs at equal
weight, so what live holds is the **sum of 1620 crossing-rule states** in {−1, 0, +1}, and
parity is per combo. Checked before this was written, by a scratch probe over all 525 pre-tail
weeks of both legs: a per-bar book reproduces `rmv.simulate` trade for trade on every session
but four — each one live cannot see coming — at ~100 µs a bar.

**Do** in `live.py`, offline — no network, no keys, freely re-runnable:

- `Book` — one leg's live decision path, one closed bar at a time. A ring of `MAX_N + 1` bars
  and `rmv.rmv_all_n` on it, reading `[:, -1]` and `[:, -2]` (the contract that function's
  docstring already pins for live); thresholds by `rmv.threshold` off one `params.json` leg;
  1620 `int8` states. ⚑ **The gate is rebuilt from what has arrived** — the clock, the
  calendar's close, and "the last 25 bars are contiguous", which is `build_gate`'s blackout
  read causally — and never read from `bars.gate`. The target is `sum(states)` in **combo
  units**, 1 unit = 1/1620 share: the fractional book exactly, in integers.
- The 15:55 flatten runs **before** signal evaluation (Unit 4's ordering, carried in 12b): a
  gated bar whose successor the clock says is ungated goes flat and does not enter. That is
  SPEC §2.1 A in the only form live can know it — the clock, not `gate[t+1]`.
- `FakeBroker` fills any order in full at the bar's close. `replay` reconciles every bar: read
  the position, submit the difference.
- `parity(symbol)`, and `python live.py parity [SYM ...]` over every pre-tail week.

**Done when**, every pre-tail OOS week of both legs:

1. **Live's gate equals `bars.gate` on every bar.**
2. **The reference is Unit 7's, not a re-run of it.** `rmv.simulate` on Unit 7's own inputs —
   the full-sample matrix sliced, `bars.gate`, the index's `xmult` and `cost` — scores every
   region combo to its stored `pwfo_oos.npy` row, float32 for float32, all six columns.
3. **Trade for trade, per combo** — `(entry, exit, dir, net)` by `array_equal`, all 1620 — on
   every session except those live cannot know: a gated bar whose successor is gated by the
   clock and still not in `bars.gate`, i.e. the next bar did not arrive on time (SPEC §2.1 B
   and C). ⚑ That set is computed **from the bars** — `build_gate` on timestamps shifted one
   bar — not from where trades differ, and is reported by name.
4. **The broker holds the book.** Gross off the fills equals the per-combo trades' gross
   **exactly** (`math.fsum`, and every term is exact: integer units × a float32 price, and
   Sterbenz on same-week price differences); the position is 0 at every week's end; netted
   turnover never exceeds the per-combo turnover the backtest charges `cost` on — Unit 10's
   "modelled cost is an upper bound", as a measurement.

⚑ **Post-blackout reopens are not exempted** — Unit 4's either/or, answered on 12a's side.
`Book`'s gate is `build_gate`'s own rule, so it reopens exactly where the backtest does and
only the bar with a late successor diverges. If Unit 12b's stale-bar guard *halts* the session
rather than flattening, it reintroduces Unit 4's measured gap and has to say so.

**Not this unit** — carried to 12b unchanged: rounding units to shares (a sizing decision);
sizing on Friday's close against the IS mean (−10%..+26%); SPEC §3.2's fee model; a mid-session
restart, which has to replay the session from 08:00 because every state depends on each
crossing since 10:00.

**Why this unit matters most** Every backtest-to-live discrepancy this project could suffer
shows up here or nowhere. If parity fails, nothing upstream is trustworthy.

#### Rev 2's Unit 12a, superseded — kept for the record

*Split from Rev 1's Unit 12, which bundled a polling loop, ring buffer, reconciliation engine,
five guards, a fake broker and a replay harness into one "most important" unit.*

**Do** `FakeBroker` + a bar-by-bar replay driver + the parity harness. Pure offline: no
network, no API keys, freely re-runnable.

**Done when** replaying every OOS session through the bar-by-bar path against `FakeBroker`
produces trade-for-trade equality with the Unit 7 vectorized backtest.

---

### Unit 12b — Live: Alpaca loop and guards ✅ **shipped offline — re-scoped to the region portfolio; a live session waits on real-time SIP**

**Shipped** in `live.py`:

- constants: `NOTIONAL`, `MAX_DAY_LOSS`, `MAX_GROSS`, `POLL_OFFSET`, `RETRY`, `STALE`,
  `ORDER_WAIT`, `WATCHDOG`, `HTTP_TIMEOUT`, `LOCK`;
- functions and classes: `shares()`, `Alpaca`, `reconcile()`, `session()`, `_exclusive()`,
  `dead_loop()`, `flatten()`, `run()`;
- the `run` and `flatten [now]` arms on `main`, and a rounding line in `parity`;
- one refusal added to `load_params` (a cost implying an IS mean ≤ 0).

`heartbeat` and `live.lock` are gitignored, and no other module changed.
**131/131 tests** (9 added), `ruff check .` clean, **64 of 65 mutations killed** (the one survivor is unobservable, below). The withheld
tail's sha256 is unchanged and `K` is still 14.

    python live.py run              today's session on the paper account. Exit 0 once flat
                                    after the cut (or on a closed day); exit 1 when it refused
                                    to trade (a flat day), or ended not flat, or never read a bar
    python live.py flatten [now]    the out-of-process guard. Schedule it every 5 minutes through
                                    each session day (not yet registered). It acts only when no
                                    loop holds live.lock, or when the loop's heartbeat is stale;
                                    `now` acts regardless

**Done when** ✅ **met offline, all four**, with a simulated Alpaca on a simulated clock over
real cached sessions. The simulated `bars` serves the previous session and the bar still
forming, and ignores `lo`, so the loop has to drop all three itself.

1. **A day holds the book in shares.** On 2025-07-15, a busy day, both legs' positions equal
   the book's whole-share target after every one of 99 polls.
   - The day has 84 orders, which is one per target change plus one per reversal, and three
     QQQ reversals, each sent as two orders.
   - Gross off the fills equals the whole-share book's gross **exactly** (`fsum`).
   - The heartbeat holds the last poll's clock.
2. **A restart converges on its first poll.** Killed at 11:02, with both legs flattened behind
   its back, the restart replays the session from 08:00.
   - At 11:02:00 it is back on the 11:00:15 position.
   - Every later poll is identical to the uninterrupted run's.
3. **Every guard fires in a test and leaves the position asserted:**

| guard | case | afterwards |
|---|---|---|
| the clock's cut | 2025-07-14 with the 15:50 bar withheld; 2024-11-29 (13:00 close) | flat at **15:55:00** exactly, where stale alone would hold to 15:56; done at 12:55 |
| stale | SPY 2019-08-12, feed stops at 15:30 | held at 15:40:55, flat at **15:41:00** and through the cut, while QQQ trades on. **12a's overnight hold is closed.** |
| stale, then the book | 2020-03-18, 13:00 and 13:05 missing on both legs | held at 13:05:55, flat at 13:06:00; from 13:15:15, bar for bar with the book through the blackout and the 15:10 reopen |
| daily loss | $7,000.00 before the open / $7,000.01 / $8,000 at 11:00 while holding | no fire and the day trades / flat all day, after the loss has gone / flat at 11:00:15 and never back in |
| notional cap | QQQ sized on a $0.01 IS mean | halted on its first non-zero target, no QQQ order ever sent, SPY flattened |
| params | last week's file; a cost implying an IS mean ≤ 0; a closed day | exit 1 with no order and no poll (twice); exit 0 |
| a second loop | `live.lock` already held | exit 1, no order, no poll |
| SIP | "subscription does not permit…" on the startup probe | exit 1, no order; a plain 503 there proceeds and trades |
| rejects | SPY 11:05–11:20 | retried every 5 s, back on the book at 11:20:00 |
| errors | QQQ bars, SPY position, account read, 11:00:15–:30; QQQ bars failing all day | the loop survives; QQQ holds its 10:50 target, then catches up at 11:00:30. A blind day ends flat with **exit 1**, not a quiet day's 0 |
| a rejected cut | 20 s of rejects; rejects to the close | flat at 15:55:20; exit 1 at 16:00 for the external job |
| reversal | 12 scripted `reconcile` cases | close, and open only once the close has filled; a short fill stops |
| flatten arm | 16 lock × clock × heartbeat cases, normal and early close; `flatten` against a real held lock | acts when no loop holds `live.lock`, or on a running loop's stale or missing heartbeat, and nowhere else — **not** at 15:58 beside a live loop. While acting it holds the lock itself |
| heartbeat | every position read in the busy day | the heartbeat already holds that poll's clock: written as a poll starts, so a hung poll goes stale |

4. **Online, read-only.** Against the paper account:
   - the adapter reads the paper base URL, equity, and a 404 position as 0;
   - the account refuses real-time SIP, and `run` exits 1 on that before reaching the loop.

   Offline, against a scripted `TradingClient`, the adapter:
   - signs a short negative under either qty convention;
   - reads a 404 as flat and raises on a 500;
   - cancels and settles open orders before the position read;
   - sends a sell as a sell with a negative fill;
   - cancels after `ORDER_WAIT` and returns the fill so far, and raises if an order is still
     open after the cancel.

**Measured:** `parity` now prints the whole-share book every run, **−0.33% / −0.13%** of gross
at $50,000 a leg. Every Unit 12a headline is unchanged: 3,958,490 / 3,937,717 trades, 83.3% /
82.4%.

**Mutation pass.** The first pass killed **56 of 58**. Both survivors taught something:

- ⚑ **"A bar from before today is read" survived because it is harmless.** `Book` is flat on
  every ungated bar, and its 120-minute span gate stays shut across the overnight gap until
  10:00. So yesterday's session, fed first, cannot move today's target. The `ts >= start`
  filter was not doing any work, and it is deleted with that reason in the code.
- ⚑ **"The loss limit is inclusive" survived because the boundary test never reached the
  boundary.**
  - `0.07 × 2 × 50,000` is 7000.000000000001, so a $7,000.00 loss sat an ulp *under* the
    limit under either operator.
  - The limit is now `MAX_DAY_LOSS = 700` bps, and `700 × 2 × NOTIONAL / 1e4` is exactly
    7000.0. The test asserts that exact figure before it relies on it.

The second pass drops the equivalent mutant and adds two for the fix below: **59 of 59**.

⚑ **Mine, found reading alpaca-py while the pass ran: no request has a timeout.**
`RESTClient._one_request` calls `Session.request` with no timeout, so one hung socket would
stall the loop for the rest of the day. The watchdog would flatten, but the loop would never
come back.

- Both clients' sessions now carry `timeout=HTTP_TIMEOUT` (10 s), patched onto alpaca-py's
  private `_session` and marked `ponytail:`.
- A timeout raises like any other error, and the poll retries.
- The online test pins the patch on both clients.

**The third pass** runs after the review fixes below. It drops the watchdog mutants whose code
is gone and adds 13 for the lock, the heartbeat's position in the poll, the blind day and the
moved IS-mean refusal: **64 of 65**. The survivor, "the lock is never explicitly released", cannot be seen from a test: closing the file releases a Windows lock anyway. The explicit `LK_UNLCK` stays, because Microsoft documents the release on close as happening after a delay that depends on system resources, and a `flatten` should hand the lock back promptly.

- ⚑ **The harness now runs its test sets unmutated first.** A rename of the watchdog test left
  its old name in the harness, and a trial "killed" four mutants with `AttributeError`. They
  were false kills, and a baseline that has to pass stops that from recurring.

#### Adversarial review (PLAN §4) — 5 findings: 5 fixed

The reviewer (Sonnet) re-ran everything that carries a number:

- 131/131 and ruff;
- both `parity` legs, bit for bit, including the new rounding line;
- the tail's sha256, and `K` = 14;
- the per-leg split of the 84 orders: SPY 42 with no reversal, QQQ 42 with 3.

It also:

- traced the cut and retry clamps and found no poll that can skip the cut or sleep negative;
- worked `reconcile` over every sign case;
- checked alpaca-py 0.43.4's source for the qty sign, `APIError.status_code`, the timeout
  patch's keywords, and `TERMINAL`: the statuses it omits are unreachable for a plain market
  DAY order.

It touched no repository file, placed no order, and left the tail alone. **No BREAKING
finding.**

1. **Fixed — the watchdog raced a live loop in `[close − 2 min, close)`** (filed SEVERE).
   - **The failure.** The rule was "always act from close − 2". But the suite's own
     rejected-cut case shows the loop still retrying past 15:58, and two read-then-submit
     flattens overshoot: +5 becomes −5. `session` would then exit 0 on its own fill
     bookkeeping, holding the overshoot. The reviewer showed the mechanism standalone
     (`review/race_probe.py`).
   - **The root cause.** Clock windows were guessing at "is the loop running?", which the OS
     can answer.
   - **The fix.**
     - `run` holds an exclusive lock on `live.lock` (`msvcrt`) for the session's whole life.
     - `dead_loop` acts only when that lock is free, or when a running loop's heartbeat is
       more than `WATCHDOG` old or missing.
     - `flatten` holds the lock while it acts.
     - The heartbeat is now written as each poll *starts*, so a hung poll goes stale and a
       slow one does not.
   - **The consequences.**
     - A live loop is never raced at any hour, including after the loop has flattened and
       exited: the lock is then free, and the watchdog's flatten is a no-op or clears a
       residual.
     - A second `run` refuses to start, which is a guard Rev 2 never had.
     - A dead loop is flattened on the watchdog's next tick instead of 11 minutes later.
   - **The re-scope's decision 11** ("at or after the cut, unconditionally; 12:58 and 15:58")
     is superseded by this. The schedule is now simply every 5 minutes through the session
     day.
2. **Fixed — the `run` synopsis said "exit 1 is a flat day"** (MINOR). Exit 1 also means
   "ended not flat". The synopsis above and `run`'s docstring now list all three causes.
3. **Fixed — an IS mean ≤ 0 raised a traceback out of `session`** (MINOR). The check moved
   into `load_params`, where a cost at or under `SLIP` is a refused file: a clean flat day,
   with two more refusal cases in Unit 11's test.
4. **Fixed — "N positions closed" overstated `close_all_positions`** (MINOR). That call sends
   orders and does not wait for fills, so the message now reads "close orders sent".
5. **Fixed — a day on which no bar was ever read exited 0, like a quiet day** (MINOR). It now
   ends flat with exit 1 and names the blind leg; the test fails QQQ's feed all day.

**Recorded, not changed:**

- `session` still ends on its own fill reports, not a fresh position read. With the lock,
  nothing else trades the account while the loop lives, and the watchdog's first tick after
  the loop exits clears any residual.
- The lock is Windows-only (`msvcrt`), marked `ponytail:`; `fcntl.flock` is the POSIX swap.
- A synchronous short-sale reject (`submit_order` raising) is caught by `session`'s
  `reconcile` handler like any other error. It could not be exercised without sending an
  order.

**Carried to Unit 13:**

- **A paper session end to end.** It needs real-time SIP (§8-D, now a blocker) and market
  hours.
- **The poll offset and the cost of the delay.** The loop prints each bar as first seen and
  each fill with its price; diff those against a later fetch and against the signal close.
- **SPEC §3.2's fee model**, before real money.
- **The real-money `NOTIONAL`**, and the edit away from `paper=True`.
- **Registering the schedules** (Task Scheduler). Both are the user's to create; nothing here
  created them:
  - `live.py refit` on Saturday;
  - `live.py run` before 08:00 on weekdays;
  - `live.py flatten` every 5 minutes through each session day.
- **The residual risk:** a dead host runs neither the loop nor the job. Decide on a broker-side
  stop against its churn.

#### The re-scope, as written before any code

⚑ **Re-scoped before any code, from Unit 12a's "Carried to Unit 12b".** Rev 2 (kept below)
drives one row into one position. What live runs is one `Book` per leg, each a target in combo
units, and what reaches Alpaca is that target in whole shares.

**Measured first**, by a scratch probe over the pre-tail sample (disclosed, as 12a's was; the
tail is not read):

- **The worst day.** The 50/50 book marked to market every bar, net of netted cost: worst
  intraday low **−345.5 bps** of gross notional on 2025-04-07, then 2020-03-12 −261.6 and
  2020-02-28 −215.1, over 2,529 sessions (legs alone: SPY −319.5, QQQ −371.5, same day).
- **Rounding.** At $50,000 a leg (~73 SPY, ~81 QQQ shares for a full book), the whole-share
  book's gross is **−0.33% / −0.13%** off the fractional book's. The weekly gap has sd
  **0.34 / 0.38 bps**, against a weekly sd of 55 / 72, and turnover is unchanged to three
  decimals.
- **Churn.** The share target changes on **41% / 42%** of gated bars, about 145 times a week
  per leg (21 weeks sampled). It crosses zero on 1.3%.
- **The account.** Paper, $100,000, 4× multiplier, shorting enabled. ⚠ **No real-time SIP:**
  a bar request ending inside the last 15 minutes is refused ("subscription does not permit
  querying recent SIP data"). §8-D's paid subscription is now a **blocker for Unit 13**, no
  longer just a budget line.

**Decisions**, each answering a carried item or a Rev 2 line:

1. **Sizing uses the IS mean, not Friday's close.**
   - A full book is `size = NOTIONAL / window_notional(cost)` shares. The target is
     `round(size · units / 1620)`.
   - The IS mean is known on Friday night, just as the close is. It is exactly the
     denominator of the backtest's bps, so live's bps *are* the backtest's bps.
   - Unit 10 measured the choice as immaterial to `t` (2.7726 against 2.7744).
   - `NOTIONAL` is **$50,000 a leg**, 1× gross on the paper account. The real-money figure
     is Unit 13's; $25,000 at 4× (§1.3's PDT floor) is the same gross.
2. **Rounding is to the nearest share**, with Python's `round`: half-to-even, so long and
   short are symmetric. It is immaterial (above), and `parity` now reports it on every run so
   it stays measured.
3. **The fee model moves to Unit 13, before real money.** SPEC §3.2 required it "before any
   live order" because `cost` selected the row. No live decision reads `cost` now:
   - `Book` reads `xmult` only.
   - `cost` sizes (through its exact inverse, not its fee content) and scores.
   - Paper orders pay nothing, and Unit 13 compares realised fees with the model.
4. **A restart replays today from 08:00, then reconciles.** Every poll fetches today's
   session and feeds `Book` the closed bars it has not seen. In a fresh process that is all
   of them, so a restart reconciles on its first poll, not at the next bar.
5. **The cut belongs to the clock.** At `now ≥ close − 5 min` the target is 0, whatever the
   book holds, and the loop exits once flat. Normally `Book`'s own 15:50 flatten lands on the
   same poll. This closes 12a's SPY 2019-08-12 overnight hold.
6. **A stale bar flattens; it does not halt.**
   - The trigger: the bar after the last one fed is still missing `STALE` after its close.
   - The target is then 0 until a bar arrives, and the book decides from there, as the
     backtest's gate does. A real gap means blackout and flat; a feed that was only late
     returns to the book's position.
   - This is 12a's answer to Unit 4's either/or, applied on the flatten side.
7. **A daily loss flattens and halts for the day.**
   - The trigger: `last_equity − equity > 700 bps × 2·NOTIONAL` ($7,000), twice the worst
     session the backtest ever marked.
   - It never fires on the pre-tail sample, so it is a stop for bugs and catastrophes, not a
     strategy change. When it does fire, the parity break is stated.
8. **A notional cap flattens and halts.**
   - The trigger: `|target| · close > 1.5 · NOTIONAL`.
   - By construction this cannot happen (`|units| ≤ 1620`, and the IS mean drifts −10..+26%
     from Friday), so it catches a corrupted size.
   - A size that is not positive and finite refuses at the start.
9. **Refused params mean no loop** (`load_params`, unchanged): a flat week.
10. **Reconciliation, not events.** Per leg, per poll:
    - Cancel open orders, read the position, and submit the difference as a market DAY order.
    - Wait for a terminal status; after `ORDER_WAIT`, cancel.
    - A move across zero is two orders, close then open. The open goes only once the close
      has filled, so the loop never depends on whether Alpaca accepts a flip in one order.
    - A partial fill or a reject is just the next poll's difference.
11. ⚑ **Out-of-process safety changes from Rev 2: no broker-side GTC stop.**
    - Why: a stop sized to a target that changes on 41% of gated bars would be cancelled and
      re-placed ~145 times a week per leg, each time racing the market order it brackets.
    - Replacement: one `python live.py flatten` arm, scheduled by the user outside the process
      every 5 minutes, and at 12:58 and 15:58.
    - It flattens both legs at or after the cut. Inside the gated window it flattens when the
      loop's heartbeat is older than `WATCHDOG`.
    - ⚠ Residual: a dead **host** runs neither. The exposure is one session's move on at most
      1× notional. Unit 13 decides whether a broker-side stop is worth its churn.
12. **The poll offset is 15 s after each bar close — a starting value, not a measurement.**
    This account cannot read real-time SIP, so the offset cannot be measured here.
    - The loop prints every bar as first seen and every fill price.
    - Unit 13 diffs the bars against a later fetch (amendments) and the fills against the
      signal close (delay cost). That is how the offset and its cost get charged.
13. **`paper=True` is hard-coded.** Real money is Unit 13's go/no-go, made by editing the
    code rather than passing a flag.
14. **`run` checks at startup that real-time SIP is readable**, and refuses with the reason
    if it is not.

**Do** in `live.py`:

- `NOTIONAL` and `shares()`.
- `Alpaca`, the loop's only network: SIP bars in, paper orders out.
- `reconcile()`, and `session()` for one trading day, with the clock and sleep injected.
- The `run` and `flatten` arms, a `heartbeat` file (gitignored), and the rounding line in
  `parity`.

**Done when**, offline, with a simulated broker and clock over real cached sessions:

1. **A day through `session`** holds `shares(book)` after every poll, and the simulated
   broker's gross equals the whole-share book's.
2. **A kill at 11:02 and a restart:** the position equals the uninterrupted run's at the
   restart's first poll and at every poll after it.
3. **Every guard has a test that fires it and asserts the position afterwards:**
   - the clock cut, with a 15:50 bar that never comes;
   - stale, on 2020-03-18's missing 13:00 and 13:05: flat within `STALE`, then back in on
     the blackout's reopen, bar for bar with the book;
   - daily loss: flat, and still flat after the book wants in again;
   - the notional cap;
   - refused params: no order sent;
   - a reversal: two orders;
   - a rejected order: retried on the next poll;
   - a transient fetch error: the loop survives;
   - the flatten arm: acts only past the cut or on a stale heartbeat.
4. **Online, read-only:** the adapter reads position 0, no open orders, and equity off the
   paper account, and `run` refuses on the missing SIP subscription.

**Not this unit:** a paper session end to end. It needs real-time SIP and market hours, so it
belongs to Unit 13.

**Review focus:**

- the bar-close race and the restart replay;
- partial fills, rejected orders and short-sale rejects;
- the cut under a rejected order;
- 429/5xx/timeout on the poll path.

#### Rev 2's Unit 12b, superseded — kept for the record

⚑ *(Pre-re-scope pointer:)* **Needs the same re-scope as 12a before it starts.** The text below
still assumes one row and one position; start from Unit 12a's "Carried to Unit 12b", which also
answers the stale-bar guard's halt question and adds the clock-driven flatten.

**Do**

- Poll 5-min bars. Ring buffer of `max(N)+2` closes — fixed size, no pandas, no growth.
- **Reconciliation, not events**: each bar compute the desired position, read the actual
  position from Alpaca, submit the difference. Idempotent and restart-safe by construction.
- Guards, each with an explicit stated effect on an open position: 15:55 unconditional flatten;
  stale-bar (no new bar in 10 min) → flatten and halt; max daily loss → flatten and halt;
  position cap; stale `params.json` → refuse to start.
- ⚑ **Out-of-process safety, which Rev 1 omitted entirely:**
  - a broker-side **GTC stop** placed on entry as a dead-man's switch (Meyers' own CL largest
    losing trade is ~5% of contract value; this system otherwise has *no* stop);
  - a **Task Scheduler job at 15:58** that flattens everything via one REST call, independent
    of `live.py`. If the process dies at 11:00 holding a short, nothing in Rev 1 noticed. ~10
    lines, and it covers the single most expensive failure mode of an intraday system.
- ⚑ Pin the poll offset: Alpaca amends the latest bar with late trades, so decide how many
  seconds after `:00` to poll, and charge the resulting delay to the §1.3 cost model.

**Done when** a mid-session kill and restart converges to the correct position within one bar;
every guard has a test that fires it and asserts the position afterwards.

⚑ **From Unit 4, two guard details that decide parity rather than decorate it.**

- **The 15:55 flatten must run *before* signal evaluation on the 15:50 bar.** Unit 4 opens no
  position on the last gated bar of a run; if the live loop evaluates the signal first and
  enters, live and backtest differ by **1.74% of the trade count**, all of it pure cost, with
  no failing test anywhere. Nothing in SPEC §2 pinned this ordering, so it is pinned here.
- **The stale-bar guard's "flatten *and halt*" does not match the backtest gate**, which
  reopens the same session once the `max_n` blackout expires: 2016-02-02 reopens 13:25–15:50
  (**30** more gated bars) and 2020-03-18 reopens 15:10–15:50 (**9** more). Live would have
  traded none of them. That is a parity gap of dozens of trades on those two sessions, and no
  exit-price rule closes it — either the live guard resumes on the same rule the gate uses, or
  Unit 12a's harness must exempt post-blackout reopens and say so.

**Review focus** Bar-close race; partial fills; rejected orders; short-sale rejects; 15:55
flatten under a rejected order; 429/5xx/timeout handling on the poll path.

---

### Unit 13 — Paper soak and go/no-go

⚑ **Start from Unit 12b's "Carried to Unit 13".** Real-time SIP comes first, because nothing
can run without it. The same list holds the poll offset and delay cost, the fee model, the
real-money `NOTIONAL`, the `flatten` schedule, and the dead-host residual.

Run paper for at least 4 weeks. Compare realized weekly net profit against the Unit 9
distribution. Log every divergence between expected and actual fill.

**Go criteria, set now rather than after seeing results**: realized weekly mean within 1 sd of
the OOS expectation; parity violations = 0; no guard fired unexpectedly. ⚑ Capital and share
size are settled in §1.3, not discovered here.

---

## 4. Review protocol

After each unit, before starting the next:

1. Adversarial subagent (spawn one subagent for adversarial review, not many) receives `PLAN.md`, `SPEC.md`, the unit's diff, and its test file.
2. Its brief: *find what is wrong.* Does the code do what the unit says; do the tests actually
   fail when the logic breaks (mutate a constant and check); is there look-ahead; does it hit
   the budget; what did the plan not anticipate.
3. Findings triaged into fix-now / defer-with-a-`ponytail:`-comment / reject-with-reason.
4. **Stop.** The next unit begins on an explicit go.

The tests are the deliverable as much as the code. A test that passes against broken logic is
worse than no test. ⚑ Rev 1's own review found two such cases before a line was written
(§1.7 rule 1, §1.3 bar series) — this protocol pays for itself.

---

## 5. What this plan deliberately does not build

Adding any of these requires a reason written down first.

| Not building | Why | Add when |
|---|---|---|
| WFME64's 115k-filter search | Second-order overfit — and Unit 10 measured IS→OOS rank correlation at **t = −0.45**, so no filter space contains the answer | **Never.** Cancelled in Unit 10 |
| Distributed / queued execution | The full run is 35 seconds | Never, at this data scale |
| Parquet / DuckDB / any database | Dense float32 matrix; `.npy` memmap is faster and dep-free | The table exceeds RAM |
| Config framework, YAML schema, CLI parser | Five modules and a `params.json` | A third caller appears |
| Strategy base class / plugin registry | There is one strategy | A second strategy actually exists |
| Websocket streaming | Bar-close signals. ⚑ Not because "polling has no reconnect logic" — it has 429s, 5xx and timeouts — but because sub-bar latency is worth nothing here | Sub-bar latency matters, i.e. never here |
| Multi-asset abstraction | SPY first; `symbol` is already a parameter | A second symbol is actually traded |
| Incremental / online RMV | 576 ops per bar, once per 5 minutes | Never |
| Matousek O(N log N) repeated median | N <= 24 | N > ~100 |
| Vol targeting, portfolio layer | Not in the paper; each is its own research problem | After live validation |
| ⚑ ~~Position sizing~~ | **Reversed**: a fixed share count is a required constant, not research. Pinned in §1.3 | — |
| vectorbt | Imported nowhere; six transitive deps | Never |

---

## 6. Known defects in the current code (fix by replacement, not patching)

1. `run_strategy` double-lags: `positions_from_rmv` applies `execution_lag`, then
   `strategy_returns` shifts again (`rmv.py:84`, `:120`). Two-bar lag.
2. `optimize_window` calls a per-bar Python loop 5600× per window (`:254`). Orders of magnitude off.
3. `rmv_by_n[n].reindex(close.index)` inside the combo loop (`:255`) — a reindex per combo.
4. Level rules, not the 2025 crossing rules (§1.2).
5. `walk_forward_windows`: hardcoded `n_windows=16` (`:274`); `+1 day` masks overlap (`:327`).
6. `normalize=False` by default while the grid uses raw units 0.02–0.40 — most of that grid
   can never trigger on SPY 5-min slopes.
7. ⚑ ~~`xmult = 4.00512` is unsourced~~ — **closed in Unit 3.** The old constant died with
   the Unit 2 rewrite; `rmv.xmult` replaces it with the [M25] Appendix method, refitted per
   IS window rather than frozen at any value (SPEC §1.2.1).
8. ~~`benchmark.py` re-derived the OOS window as `oos_end - 7 days`~~ — **file deleted in
   Unit 2**: it imported `calculate_returns`/`rmv_trading_system` from the old `rmv.py`, so
   the rewrite made it an unconditional `ImportError`. Nothing in it is needed; its Sharpe
   and max-drawdown are superseded by SPEC §6, and Unit 9 owns the reporting.
9. `get_data` hardcodes dates and sets no adjustment policy.
10. `alpaca_api.yaml` — plaintext credentials in the working tree.

---

## 7. Risk register

| Risk | Mitigation |
|---|---|
| Filter meta-search overfits | Bounded pre-registered space, a literal comparison counter, withheld tail opened once, three-condition gate |
| Backtest/live divergence | Unit 12a parity harness is a hard gate; `adjustment="split"` keeps both sides on one price series |
| `xmult` saturates in a volatility regime shift | Per-year saturation diagnostic; switch to per-IS-window normalization if it swings >2× |
| Overlapping IS windows inflate `t` | Bootstrap is the decision statistic; autocorrelation reported; block-bootstrap if needed |
| IEX unusable for live | Decided at Unit 1 as a gate, before 12 units are built on it |
| Gap contamination of RMedV | Fixed structurally in §1.3 (full-session series); halts gated |
| Costs swamp edge | Costs and SEC/TAF in from Unit 5; power check before Unit 10 |
| Process death holding a position | ⚑ Unit 12b: an external `live.py flatten` job every 5 minutes — flattens when no loop holds `live.lock` or the loop's heartbeat is stale, and never races a live loop. No broker-side GTC stop (the target changes on 41% of bars); a dead **host** is the residual, Unit 13's call |
| Two loops trading one account | Unit 12b: `run` refuses while `live.lock` is held |
| A hung HTTP request stalls the loop | Unit 12b: `HTTP_TIMEOUT` on both alpaca-py sessions |
| PDT restriction | Minimum capital pinned in §1.3 |

---

## 8. Open decisions

| # | Question | Recommendation |
|---|---|---|
| A | History depth | ✅ **Resolved (Unit 1): 2016-01-04 → 2026-08-31 — 257,217 bars / 2,680 sessions / 10.7 yr.** Meyers notes 10 yr is his own bias and shorter may do better; each re-look increments the comparison counter, so testing a shorter span is a deliberate spend. |
| B | `mLTr` sign convention (§1.5) | Evidence points to negative storage → "smallest" = deepest. Run both in Unit 8. |
| C | ⚑ `r2` vs `r` in the CL4 screen (§1.5) | Unresolvable from the papers. Run both; decisive for CL4. |
| D | **Data feed** | ✅ **Resolved (Unit 1): SIP only; IEX is unusable.** 57% of 5-min buckets missing, and where present a median 2.50¢ error = 13.9% of an 18¢ bar move, 76.5% of bars off by ≥1¢. **Live needs a paid SIP subscription — budget for it before go-live.** ⚑ Unit 12b: the paper account is refused any SIP request ending inside the last 15 minutes, so this is now a **blocker for Unit 13's paper soak**, not only for go-live; `live.py run` refuses at startup on it. |
| E | Long-only or long/short? | Long/short per the paper. Needs a margin account; SPY is trivially shortable. |
| F | ⚑ The 2025 erratum (§1.2) — include overnight trades? | Not in v1. Under §1.3 the RMedV series already spans the full session, so this is a gate change later, not a rewrite. |
