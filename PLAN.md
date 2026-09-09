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
  96-byte stride — ~16× read amplification at Unit 10's ~2000 filters.

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
| Live per-bar compute | < 1 ms, zero steady-state allocation | — |

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
  reopens this deliberately or not at all.
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

### Unit 9 — Significance, costs, report

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

### Unit 10 — Filter search (conditional)

⚑ **From Unit 5: pre-register the comparison DIRECTION, not just the metric and threshold.**
Every degenerate sentinel in SPEC §6.6 is fail-safe in exactly one direction — the one the
three shipped filters use — and is the worst possible value in the other. Measured over 24
real pre-tail windows, a filter picking **max `eqR2`** selects a row with `nT <= 2` in **24 of
24**, and **min `eq2R2`** in **24 of 24**; a screen `PF > x` passes every no-loser row, and a
**top**-k rank on `mLb` puts every no-loser row at the head. A generated space that crosses
metrics with both directions will therefore find "filters" that are selecting the sentinel,
score them well in-sample, and look like discoveries. Either give every generated filter an
`nT >= 3` screen or restrict the space to §6.6's directions — and write which, first.

Meyers ran 115,320 filters through WFME64. Do **not** start there.

⚑ **Pre-register or don't build.** "Build only if Unit 9 fails" makes this a conditional
second look at the same data — a forking path. Either write down the bounded space *before*
reading Unit 9's result, or skip the unit.

If built: ~2000 filters (3 screen metrics × thresholds × 3 rank metrics × `top_k` in
{5,10,25,50}), scored by bootstrap probability, every one added to the comparison counter.

**Done when** reproducible from a seed. Validation is the withheld tail, opened once, after
freezing.

**Anti-goal** 115k filters over 546 windows is feasible and is precisely the second-order
overfit Meyers' own Bonferroni correction warns about. Bounded space, or none.

---

### Unit 11 — Live: weekly refit (`live.py`)

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

### Unit 12a — Offline parity harness ⚑

*Split from Rev 1's Unit 12, which bundled a polling loop, ring buffer, reconciliation engine,
five guards, a fake broker and a replay harness into one "most important" unit.*

**Do** `FakeBroker` + a bar-by-bar replay driver + the parity harness. Pure offline: no
network, no API keys, freely re-runnable.

**Done when** replaying every OOS session through the bar-by-bar path against `FakeBroker`
produces trade-for-trade equality with the Unit 7 vectorized backtest.

**Why this unit matters most** Every backtest-to-live discrepancy this project could suffer
shows up here or nowhere. If parity fails, nothing upstream is trustworthy.

---

### Unit 12b — Live: Alpaca loop and guards

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
| WFME64's 115k-filter search | Second-order overfit; the papers' three filters are the honest baseline | Unit 10, pre-registered |
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
| Process death holding a position | Broker-side GTC stop + external 15:58 flatten job (Unit 12b) |
| PDT restriction | Minimum capital pinned in §1.3 |

---

## 8. Open decisions

| # | Question | Recommendation |
|---|---|---|
| A | History depth | ✅ **Resolved (Unit 1): 2016-01-04 → 2026-08-31 — 257,217 bars / 2,680 sessions / 10.7 yr.** Meyers notes 10 yr is his own bias and shorter may do better; each re-look increments the comparison counter, so testing a shorter span is a deliberate spend. |
| B | `mLTr` sign convention (§1.5) | Evidence points to negative storage → "smallest" = deepest. Run both in Unit 8. |
| C | ⚑ `r2` vs `r` in the CL4 screen (§1.5) | Unresolvable from the papers. Run both; decisive for CL4. |
| D | **Data feed** | ✅ **Resolved (Unit 1): SIP only; IEX is unusable.** 57% of 5-min buckets missing, and where present a median 2.50¢ error = 13.9% of an 18¢ bar move, 76.5% of bars off by ≥1¢. **Live needs a paid SIP subscription — budget for it before go-live.** |
| E | Long-only or long/short? | Long/short per the paper. Needs a margin account; SPY is trivially shortable. |
| F | ⚑ The 2025 erratum (§1.2) — include overnight trades? | Not in v1. Under §1.3 the RMedV series already spans the full session, so this is a gate change later, not a rewrite. |
