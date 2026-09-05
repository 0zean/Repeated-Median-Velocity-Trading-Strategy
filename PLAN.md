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
| RMV, 22 N × 196k bars | < 5 s, < 50 MB | **1.95 s, 34 MB** |
| Same via `scipy.siegelslopes` (oracle only) | — | ~27–47 s |
| Grid, one IS window (4312 × 1638) | < 60 ms | **43 ms** |
| Full PWFO, 546 windows, IS+OOS | < 60 s, peak RSS < 1 GB | **~32 s, 226 MB** |
| One filter over the full table | < 1 s | — |
| Live per-bar compute | < 1 ms, zero steady-state allocation | — |

**The entire walk-forward is a ~35-second job.** Any design adding a job queue, a database,
a cache tier, or a distributed runner is solving a problem this project does not have.

---

## 3. Units of work

Each unit: one sitting, one testable deliverable, then an adversarial review, then **stop**.

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

- Matches `scipy.stats.siegelslopes(..., method="hierarchical")` to 1e-9 for N in {3,5,10,24}
  at several offsets on random-walk data. *(Prototype already passes.)*
- Reproduces both papers' worked toy examples (each has a known answer of exactly 1.0).
- Every consumer starts at index `N-1`; a test asserts warmup values are never read.
- Budget: **< 5 s** for 22 N × 196k bars.

**Review focus** Off-by-one in the window, `j-i` sign, even-length median tie-breaking vs
scipy, float32 vs float64 accumulation.

---

### Unit 3 — Normalization calibration

**Do**

- On a **fixed early calibration slice** (first ~2 years), compute `sd(RMedV_N)` for
  ⚑ **N = 3..20** (matching the Appendix method).
- Verify `1/sqrt(N)` proportionality; compute `xmult = mean(1 / sd(RMedV_N * sqrt(N)))`.
- Freeze into `norm.json` keyed by `{symbol, timeframe, session, calibration range}`.
- ⚑ **Saturation diagnostic, per year**: fraction of the 4312 combos producing (a) zero trades,
  (b) saturated signal (`|RMedV_norm|` routinely > 3.5).

**Done when** ⚑ normalized `sd` within **±0.15** for N >= 5 on a held-out slice, with N<5
exempted and reported separately. Rev 1's ±0.05 for every N is unachievable by the method
itself: the paper's own table gives `9.693/8.738 = 1.109` at N=3 and `1.092` at N=4 — the
`sqrt(N)` proportionality visibly breaks below N≈5.

⚑ **Freezing is provisional, not principled.** Rev 1 justified it as keeping `vup` comparable
across windows — but nothing downstream depends on that (selection is within-window,
aggregation is over P&L). The real risk runs the other way: a constant fitted to 2016–18 SPY
(VIX ~13, SPY ~$250) applied to March 2020 (VIX 80) saturates the entire grid, and the filter
then picks arbitrarily among 4312 near-clones. **If the saturation diagnostic swings more than
~2× across years, switch to per-IS-window normalization** — which is not look-ahead (the IS
window strictly precedes its OOS) and deletes this unit, `norm.json`, and the held-out step.

---

### Unit 4 — Trade simulation kernel

**Do**

- `simulate(rmv_row, close, gate, vup, vdn, cost) -> trades`, one pass, fixed buffer.
- 2025 crossing rules, 10:00 gate, 15:55 flat, stop-and-reverse.
- A deliberately slow, obviously-correct pure-Python reference in `test_rmv.py`.

**Done when** kernel == reference trade-for-trade on 1000 random series × random
`(N, vup, vdn)`; hand-built series produce the exact expected trade list; a bar exactly equal
to `vup` triggers (`>=`, per the paper); no trade spans 15:55; none opens before 10:00.
⚑ Plus: the IEX-vs-SIP **signal** divergence from Unit 1's price comparison (this needs
Units 2–4 to exist, so it lands here, not in Unit 1).

**Review focus** `>=` vs `>` boundary; the first bar of the day (`t-1` is now the prior
*extended-hours* bar, which is the intended contiguous reference under §1.3); forced EOD exit
accounting; reversal-on-same-bar.

---

### Unit 5 — Metric set

**Do** All 24 metrics from a trade array in one numba function, `cost` applied to net figures.
⚑ Equity regressions per §1.7 rule 3: zero-based, mean-centered, float64 accumulators.
`eqR2` = R² of a straight-line fit to **trade-indexed** equity; `eq2R2` = the 2nd-order fit.

**Done when** every metric matches an independent numpy computation on a hand-built trade
list; ⚑ the base-$100 and base-$200,000 equity cases from §1.7 both stay exact; degenerate
cases return defined values (0 trades; 0 losers → `PF = inf`; `mLTr` undefined → sentinel).

**Review focus** Division by zero; `mLTr` sign (§1.5); `lr` counting across window boundaries.

---

### Unit 6 — Grid runner

**Do** `run_grid(rmv_window, close, gate, ns, vs, out)` — `prange` over an `a`-major combo
index, writing a preallocated `float32[4312, 24]`.

⚑ **Free diagnostic**: count **distinct** trade sets among the 4312 combos per window. On a
penny-tick instrument at low N many `(vup, vdn)` pairs are clones; the effective grid size is
smaller than 4312 and directly informs §Unit 9's multiplier.

**Done when** any single row equals the Unit 4+5 path; output bit-identical across
`NUMBA_NUM_THREADS=1` vs 32 (genuinely satisfiable — each combo writes an independent output
row, no reduction, and no `fastmath` per §1.7); **< 60 ms/window**; zero allocation in `prange`.

---

### Unit 7 — PWFO driver (`pwfo.py`)

**Do**

- Window generator: ⚑ **week-anchored** — IS = the 30 calendar days ending Friday, OOS = the
  following Mon–Fri, step 7 days.
- Per window: `run_grid` on IS **and** on OOS (same combos).
- ⚑ Stream to **three** memmapped files: `pwfo_is.npy`, `pwfo_oos.npy`, and `pwfo_tail.npy`
  (the withheld final 6 months, both IS and OOS, written and then not opened again until §Unit 9's
  final step).

**Done when**

- Every window asserts `max(IS ts) < min(OOS ts)` — **the leakage guard**.
- OOS weeks tile the timeline exactly once: no overlap, no gap.
- Re-running produces byte-identical tables.
- Budget: **< 60 s**, peak RSS **< 1 GB**.
- A randomly chosen row's OOS `osnp` reproduces when replayed standalone.

**Review focus** The `<=` / `+1 day` boundary bug pattern from the current `walk_forward`;
partial windows at the ends; holidays shortening an OOS week to 4 sessions (e.g. Thanksgiving
week 11/20–11/24/23); windows where zero rows pass any filter.

---

### Unit 8 — Filter evaluation

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

**Done when** on a synthetic table with a planted answer the filter selects exactly that row;
a no-eligible-row window produces a flat week, not a crash or a fallback pick; **< 1 s** per
filter over the full table.

**Review focus** Tie-breaking determinism; the no-eligible-rows path; whether aggregates treat
the two zero cases distinctly.

---

### Unit 9 — Significance, costs, report

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
  long/short, both filter ambiguities) are all looks at the same OOS columns, and "it's a
  35-second run" is exactly how the count gets lost. ~10 lines, and the only thing standing
  between this project and the failure mode §5 is named after.
- ⚑ **Report lag-1..4 autocorrelation of `osnp`.** IS windows overlap by 23 of 30 days, so
  consecutive selections are autocorrelated even though OOS weeks tile cleanly. Meyers' own
  `wpr=8`/`lpr=9` show run structure. The `t` statistic assumes i.i.d. and is therefore
  inflated — treat it as descriptive. If autocorrelation is non-trivial, block-bootstrap.
- Verify (not re-apply) the §1.3 cost model. ⚑ Sanity check is
  `trades × shares × ($0.01 + SEC/TAF)`, not slippage alone.
- Produce a Table-1-shaped report (same columns) plus the equity curve with its 2nd-order fit.

**Done when** a filter run against shuffled OOS columns comes back insignificant; ⚑ **plus a
shuffled-alignment falsification** — apply window *k*'s selected params to window *k+50*'s OOS.
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

**Done when** a dry run against a historical date reproduces the exact `N/vup/vdn` that Unit 7's
PWFO chose for that same window — **the offline/online equivalence check**, ⚑ now meaningful
because `adjustment="split"` makes the two price series identical. Params fall inside the grid.
A `params.json` older than 10 days refuses to trade.

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

1. Adversarial subagent receives `PLAN.md`, `SPEC.md`, the unit's diff, and its test file.
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
7. ⚑ `xmult = 4.00512` is unsourced — but it **is** used, at `rmv.py:44` whenever
   `normalize=True`. (Rev 1 said "unused".)
8. `benchmark.py:24` re-derives the OOS window as `oos_end - 7 days` instead of using the
   recorded dates.
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
