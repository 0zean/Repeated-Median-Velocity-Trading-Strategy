# RMV Strategy Specification

The normative reference for this repo. `PLAN.md` says what we will *do*; this says what is
*true*. Code disagreeing with this document is a bug in the code or an amendment logged in §9.

## 0. Provenance

| Ref | Document | Used for |
|---|---|---|
| **[M05]** | Meyers, D. (2005) *The Robust Repeated Median Velocity System*, ES 5min. `es5rmed2.pdf`, 10 pp. | Original rules, first-trade/EOD gates, the `meyers2005` filter |
| **[M25]** | Meyers, D. (2025) *Using The Repeated Median Velocity Strategy To Trade Crude Light CL 5min Bars IV*, 12/19/2014–5/30/2025. `CL5RMedV-4.pdf`, 28 pp. | Crossing rules, normalization multiplier, walk-forward scheme, `CL2`/`CL4` filters |
| **[PWFO]** | `meyersanalytics.com/Walk-Forward-Optimization.html` | In-sample and out-of-sample metric definitions (§6) |
| **[S82]** | Siegel, A.F. (1982) "Robust Regression using Repeated Medians", *Biometrika* 69, 242–244 | The estimator |

> ⚠ **Both PDFs are gitignored (`*.pdf`) and are not in git history.** This file is the only
> durable record of their content. Transcribe carefully; §9 lists the places the sources
> contradict themselves or each other.

Page references below are to the PDFs as distributed.

---

## 1. The indicator

### 1.1 Definition

[M25 p.3], [M05 p.2] — the repeated median slope over the last `N` bars:

```
RMedV(t) = median  { median      [ (price(t-j) - price(t-i)) / (i - j) ] }
                 i         j != i

           i, j = 0 .. N-1     (0 = current bar)
```

Take the slope of every pair of points; for each anchor point take the median of its `N-1`
pairwise slopes; the indicator is the median of those `N` medians. Breakdown point 50% —
robust to outlier *points*, but see §3.1 for what it is **not** robust to.

> **Index base is a reinterpretation, not the source text.** [M25 p.3] and [M05 p.2] both
> write the limits as `i = 1 to N`, `j = 1 to N`, which read literally excludes the current
> bar (the most recent point would be `price(t-1)`). We read it as `0..N-1` **inclusive of bar
> `t`**, supported by (a) the p.2 worked example, which uses all 16 window points, and (b) the
> `RMedV[1]` term in the p.4 crossing rules, which only means "previous bar" if `RMedV` itself
> is evaluated at `t`. This is worth exactly one bar of lag — see §9-I.

**Reference implementation** (tests only, never production):
`scipy.stats.siegelslopes(y, x, method="hierarchical")[0]`. The default `method` is
`"hierarchical"`, which is the repeated median; `method="separate"` is a different estimator
and must not be used.

**Worked examples**, both from the sources, both with exact answer `1.0`:

| Source | x | y |
|---|---|---|
| [M05 p.2] | 1..10 | `1 2 3 4 5 15 12 8 9 10` |
| [M25 p.2] | 1..16 | `1 2 10 4 5 6 7 8 9 18 11 12 13 18 15 20` |

### 1.2 Normalization multiplier

[M25 pp.27–28]. `sd(RMedV)` falls with `N` roughly as `1/sqrt(N)`, so a `vup` that is
meaningful at `N=4` is unreachable at `N=20`. The fix:

```
RMedV_norm(t) = RMedV(t) * xmult * sqrt(N)

xmult = mean over N of [ 1 / sd( RMedV_N * sqrt(N) ) ]
```

`xmult` is a **unit conversion**, not a strategy parameter. After it, `vup`/`vdn` are in
standard deviations and one grid covers every `N`.

Both source tables are transcribed below. [M25] generated them with the `#iRMedVtMULTSTD`
indicator over 712,815 computed RMedV values from CL 5min bars, 1/1/2013–5/26/23, under this
header — identical on both pages except the `sqrt(N)Norm` flag, which is what distinguishes
them:

```
CL5M010113-052623 5 min bars Date Range 1130101 to 1230526
Total Number of Bars=736241  sqrt(N)Norm=0      <- p.27; p.28 is the sqrt(N)-corrected run
Trading Times Constraint Start Time=0 EndTime=0
```

**Table A — raw `sd(RMedV)`, no correction [M25 p.27].** This is the *only* source evidence
for the `1/sqrt(N)` law, which Unit 3 must verify for SPY.

| N | sd | 1/sd | | N | sd | 1/sd |
|---|---|---|---|---|---|---|
| 2 | 0.0 | 0.0 | | 12 | 0.028895 | 34.607575 |
| 3 | 0.065024 | 15.378960 | | 13 | 0.027620 | 36.205743 |
| 4 | 0.055546 | 18.003075 | | 14 | 0.026593 | 37.603751 |
| 5 | 0.047342 | 21.122911 | | 15 | 0.025612 | 39.044376 |
| 6 | 0.042738 | 23.398412 | | 16 | 0.024809 | 40.307442 |
| 7 | 0.038771 | 25.792282 | | 17 | 0.024003 | 41.661430 |
| 8 | 0.036130 | 27.677799 | | 18 | 0.023349 | 42.828140 |
| 9 | 0.033673 | 29.697591 | | 19 | 0.022674 | 44.102415 |
| 10 | 0.031903 | 31.344970 | | 20 | 0.022103 | 45.242740 |
| 11 | 0.030213 | 33.097821 | | | `1/Std Mult Ave` | **32.617635** |

[M25 p.27]: *"the RMedV Standard Deviation for N=4 is 2.5 times the SD for N=20"* —
`0.055546 / 0.022103 = 2.513`. That spread is what makes a single `vup` range impossible
without the correction.

**Table B — `sd(RMedV · sqrt(N))`, after the correction [M25 p.28].**

| N | sd | 1/sd | | N | sd | 1/sd |
|---|---|---|---|---|---|---|
| 2 | 0.0 | 0.0 | | 12 | 0.101442 | 9.857818 |
| 3 | 0.114441 | 8.738099 | | 13 | 0.100930 | 9.907816 |
| 4 | 0.112604 | 8.880718 | | 14 | 0.100853 | 9.915461 |
| 5 | 0.107389 | 9.311984 | | 15 | 0.100549 | 9.945427 |
| 6 | 0.106082 | 9.426695 | | 16 | 0.100603 | 9.940072 |
| 7 | 0.103985 | 9.616792 | | 17 | 0.100329 | 9.967254 |
| 8 | 0.103547 | 9.657454 | | 18 | 0.100422 | 9.957990 |
| 9 | 0.102365 | 9.768961 | | 19 | 0.100210 | 9.979009 |
| 10 | 0.102245 | 9.780442 | | 20 | 0.100223 | 9.977714 |
| 11 | 0.101559 | 9.846492 | | | `1/Std Mult Ave` | **9.693120** |

```
xmult_CL_5min = 9.693120
```

Recomputing `mean(1/sd)` over **N = 3..20**, excluding the `N=2` row, gives `9.693122111` —
agreement to **6 significant figures**. The formula, the range, and the exclusion are all
confirmed.

Note the `sqrt(N)` law visibly breaks below N≈5: `9.693/8.738 = 1.109` at N=3 and `1.092` at
N=4. Normalized `sd` cannot be held to ±0.05 across the full range; ±0.15 for N≥5 with N<5
reported separately is the achievable target.

> **[M25] uses two different multipliers.** p.7 specifies `Mult = 6.7*sqrt(N)` and Figure 3's
> indicator parameter string (p.16) shows `6.7`. **Every published CL result was produced with
> 6.7, not 9.69**, so Meyers' effective search range was 0.17–2.4 true sd rather than 0.25–3.5.
> See §9-A.

`xmult` is symbol- and timeframe-specific. [M25 p.28]: *"different futures and different time
bars give different multipliers."* The `4.00512` in the pre-revamp `rmv.py` was unsourced and
is gone.

### 1.2.1 SPY — measured, and refitted per window (Unit 3)

**Table C — SPY 5 min, gated bars only, 2016-01-04..2017-12-29, 35,573 values.** Same Appendix
method as Table B: `xmult = mean over N=3..20 of 1/sd(RMedV_N * sqrt(N))`. Produced by
`rmv.xmult`; `test_unit3_frozen_xmult_does_not_transfer` re-derives the headline figure.

| N | sd | 1/sd | sd*sqrt(N) | 1/(sd*sqrt(N)) | | N | sd | 1/sd | sd*sqrt(N) | 1/(sd*sqrt(N)) |
|---|---|---|---|---|---|---|---|---|---|---|
| 3 | 0.083323 | 12.001493 | 0.144320 | 6.929065 | | 14 | 0.037053 | 26.988653 | 0.138638 | 7.213021 |
| 4 | 0.074268 | 13.464691 | 0.148537 | 6.732346 | | 15 | 0.035619 | 28.074740 | 0.137953 | 7.248867 |
| 5 | 0.063295 | 15.798965 | 0.141533 | 7.065512 | | 16 | 0.034325 | 29.133457 | 0.137299 | 7.283364 |
| 6 | 0.057823 | 17.294023 | 0.141638 | 7.060255 | | 17 | 0.033089 | 30.221589 | 0.136429 | 7.329812 |
| 7 | 0.052602 | 19.010569 | 0.139173 | 7.185320 | | 18 | 0.032001 | 31.248689 | 0.135770 | 7.365387 |
| 8 | 0.049549 | 20.182184 | 0.140145 | 7.135480 | | 19 | 0.030976 | 32.283261 | 0.135020 | 7.406288 |
| 9 | 0.046390 | 21.556141 | 0.139171 | 7.185380 | | 20 | 0.030040 | 33.288672 | 0.134344 | 7.443573 |
| 10 | 0.044203 | 22.622962 | 0.139782 | 7.154009 | | 21 | 0.029198 | 34.249331 | 0.133800 | 7.473817 |
| 11 | 0.041998 | 23.810922 | 0.139290 | 7.179263 | | 22 | 0.028420 | 35.186533 | 0.133301 | 7.501794 |
| 12 | 0.040230 | 24.857298 | 0.139360 | 7.175684 | | 23 | 0.027674 | 36.134547 | 0.132722 | 7.534574 |
| 13 | 0.038484 | 25.984791 | 0.138756 | 7.206884 | | 24 | 0.026998 | 37.039663 | 0.132263 | 7.560690 |

```
xmult_SPY_5min_2016_17 = 7.183306
```

Averaging the whole 3..24 grid instead of the published 3..20 gives **7.244108** — 0.85% out,
small enough to pass unnoticed and not the published method. `rmv.CAL_N_MAX` pins the range.

**Calibrated on gated bars, not on every bar.** Warmup zeros and the bars whose window
straddles a session gap are not RMedV values; including them inflates `sd` by 22.8% at N=3 and
6.8% at N=24. This is a deliberate departure from [M25 p.27]'s `Start Time=0 EndTime=0` run,
which had no session gaps to contend with — CL trades nearly around the clock. `rmv.xmult`
therefore takes the mask as a required argument.

**The `1/sqrt(N)` law holds, with the same bias Meyers' own data shows.** The log-log slope of
`sd` against `N` over N=3..20 is **-0.5386** for SPY and **-0.5668** for [M25 Table A]'s CL —
both steeper than the law's -0.5, SPY less so. `sd(4)/sd(20)` is 2.472 for SPY against Meyers'
2.513 and the law's 2.236. On the calibration slice normalized `sd` lands in **0.950..1.017 for
N>=5**, inside the ±0.15 target; N=21..24 are genuine extrapolation, since `xmult` only averages
to 20. N=3 and N=4 come in at 1.037 and 1.067 — the same break below N≈5 that Table B shows.

#### `xmult` is refitted per IS window, never frozen

PLAN Unit 3 planned a frozen constant in `norm.json` and pre-registered the condition for
abandoning it: a saturation swing wider than ~2x across years. **Measured, the swing is 6.45x.**
The constant is not shipped, `norm.json` does not exist, and Unit 7 calls `rmv.xmult` once per
IS window, applying the result to that window's IS *and* OOS grid runs. That is not look-ahead —
the IS window strictly precedes its OOS by construction (§4).

*The statistic behind "6.45x" is the per-year mean of normalized `sd`, not PLAN's literal
"fraction of the 4312 combos", which is degenerate at annual granularity — over a whole year
essentially every combo trades at least once, so the zero-trade fraction is 0.00% in all
eleven years and no ratio exists. The combo-level form is reported per window below, where it
is not degenerate. On the same per-year data the tail statistic `P(|z| > 3.5)` swings **234.6x**,
so the choice of statistic is conservative, not favourable.*

Why the frozen version fails: RMedV is **dollars per bar**. SPY ran $210 -> $690 across the
sample through a 6x range of realized volatility, so a multiplier fitted to the 2016-17 low-vol
regime lands 3.2x off over 2018-25. Per year under the frozen 7.183306 (withheld tail excluded,
so 2026 is January-February only):

| year | median SPY | mean normalized sd | bars beyond the grid, P(abs(z) > 3.5) |
|---|---|---|---|
| 2016 | 210 | 1.187 | 1.61% |
| 2017 | 244 | **0.747** | 0.15% |
| 2018 | 274 | 2.088 | 8.65% |
| 2019 | 292 | 1.523 | 3.66% |
| 2020 | 327 | 3.514 | 19.59% |
| 2021 | 428 | 2.473 | 12.53% |
| 2022 | 404 | 4.350 | 34.80% |
| 2023 | 428 | 2.612 | 14.63% |
| 2024 | 543 | 2.964 | 17.44% |
| 2025 | 620 | **4.819** | 28.92% |
| 2026 | 689 | 4.407 | 30.60% |

At 34.8% of bars beyond `vup = 3.50`, the top of the grid is no longer a rare event and the
4312 combos collapse toward clones of each other — which is precisely the failure PLAN Unit 3
named: *"the filter then picks arbitrarily among 4312 near-clones."*

**Price level is not the main driver, which forecloses the obvious alternative fix.** Dividing
each year's normalized `sd` by that year's median SPY price leaves a **3.51x** swing against
the price level's own 3.28x span. Realized volatility dominates, so re-expressing RMedV in
log-returns or percent-per-bar would still trip the >2x trigger. Refitting is the fix; changing
the units is not.

Refitting per window removes it. Over **506** rolling 21-session windows stepping 5 sessions
(Unit 7's IS/OOS shape in bar space, withheld tail excluded):

| | frozen 2016-17 | frozen, best case | refitted per IS window |
|---|---|---|---|
| windows with every N>=5 within ±0.15 | 9.9% | 15.6% | **100%** (worst 0.142) |
| worst normalized-sd deviation | 10.369 | 2.847 | **0.142** |
| cross-N spread over N>=5, median | 0.161 | — | **0.078** |
| P(abs(z) > 3.5) on IS bars, median / worst | 11.5% / 69.0% | 0.09% / 20.4% | **0.59% / 2.08%** |
| saturated combos (threshold beaten on >50% of bars), worst window | 31.8% | — | 4.9% |
| combos that cannot trade at all, worst window | 19.6% | — | 7.1% |

"Frozen, best case" is `xmult = 2.430574` fitted on the *whole* pre-tail sample — the most
favourable frozen constant that exists, and it is quoted because it is the stronger form of the
argument: the failure is not an artifact of having calibrated on the two calmest years. No
frozen constant clears 16% of windows; refitting clears 100%.

The refitted multiplier itself ranges **0.679 to 13.830** across the 506 windows (20.4x, median
3.063). At the quiet extreme the bottom of the grid, `vup = 0.25` at `n = 3`, is a raw slope of
**$0.0104 per bar** — at or under SPY's spread, so those combos are resolving tick noise. That
is a live input to Unit 6's "distinct trade sets" diagnostic and to Unit 9's effective grid
size, and it means the stored `xmult` is *the scale*, not a nuisance parameter: Units 11 and
12b step their live thresholds discontinuously every Monday, by up to a large factor.

**What refitting does not fix, and Units 8 and 9 need to know.** It makes the *in-sample* scale
exact by construction; next week's is still a forecast.

- PLAN Unit 3's done-when was written as ±0.15 for N>=5 **on a held-out slice**. That form is
  **not met and is not achievable**: applied as written (max over N>=5) to the following OOS
  week it holds in **17.2%** of windows — median 0.320, p90 0.652, worst 3.936. What is met is
  the in-sample form, in 100% of windows. The criterion moved; this is that stated plainly.
- Median over N of `abs(sd - 1)` on the OOS week is **0.258** (p90 0.577) against frozen's
  1.031 — a `vup` chosen on IS lands on a week whose scale differs by about a quarter.
- The tail is what will show up in the P&L, not the median: p99 **1.436**, worst **3.717**, and
  **12 of 506** OOS weeks come in at more than 2x their IS scale — the same saturation regime
  the frozen constant was rejected for, now confined to ~2.4% of weeks instead of most of them.

That residual is ordinary walk-forward risk — the thing the PWFO exists to measure — not a
normalization defect, and it is not tunable away: the estimation length is flat at median 0.246
to 0.261 from 10 through 42 sessions and degrades beyond, so the IS window already sits on the
plateau and a second parameter would buy nothing.

**The withheld tail was touched once, here.** The first version of the comparison table above
was computed over all 2,680 sessions rather than the 2,553 before 2026-03-01, and the review
caught it. The figures are now pre-tail. What the tail saw was a second-moment property of the
indicator — `sd(RMedV)` — never an OOS return, a trade, or a filter. It is recorded rather than
argued away, and PLAN Unit 9's comparison counter carries it.

---

## 2. Trading rules

[M25 p.4], which supersedes [M05 p.3] — the 2005 rules are level rules, the 2025 rules are
crossing rules.

| Rule | Definition |
|---|---|
| **Buy** | `RMedV[t] >= vup` **and** `RMedV[t-1] < vup` → long at market |
| **Sell** | `RMedV[t] <= -vdn` **and** `RMedV[t-1] > -vdn` → short at market |
| **First trade of day** | ignore all signals before **10:00 ET** [M05 p.4] |
| **EOD exit** | flat at **15:55 ET**, never overnight — *[SPY §3.2], not from either paper* |
| **Otherwise** | hold the current position (stop-and-reverse). Flat only outside the window. |

The system is always long, short, or flat-by-time — there is no separate exit signal. Both
thresholds are inclusive (`>=`, `<=`) and `vdn` is stated as a positive number compared
against a negative velocity.

Fills at the **close of the signal bar**.

The two gate times are not equally sourced. **10:00 is [M05 p.4] transferred directly**
(30 minutes after the equity open, and SPY's open is the same 09:30 the ES paper assumed).
**15:55 is ours**: [M25] exits at 1430 EST (the CL pit close) and [M05] says *"5 minutes
before the E-Mini close"*, which for SPY's 16:00 close is 15:55.

Both papers say **EST**. We use **ET** — wall-clock `America/New_York`, DST-aware. Literal
EST year-round would shift every gate by an hour for ~8 months of the year; DST-aware is what
a trader means and what the exchange session actually does. Recorded because it is a
reinterpretation.

**Rationale for the gates** [M05 p.4]: *"often there are gaps in the open creating immediate
system buys and sells. Many times these gaps are closed creating a losing whipsaw trade."*
[M25 p.4] gates to the CL pit session (09:00–14:30 ET) on the finding that 60–70% of
sustainable trends occur in pit hours.

> **[M25] retracts its own first-trade rule.** A dated note on p.4 reads: *"(11/10/25) Note:
> this is no longer true, and future strategies should include the overnight trades, and the
> Exit rule could still be at 1430."* We keep the gate for v1 (SPY overnight liquidity is
> thin). Because §3 computes RMedV on the full session regardless, testing this later is a
> gate change, not a rewrite. See §9-F.

### 2.1 What the rules do not say — pinned in Unit 4

Three cases fall between the paper's rules and the gate. All three are decided here because
they are silent: each produces a complete, plausible backtest under either reading.

**A. A trade may not *open* on the last gated bar of a gated run.** Fills are at the close of
the signal bar, and the last gated bar of a regular session opens 15:50 and closes 15:55 —
which is the EOD flat time. A position opened there is flattened at that same close: zero
bars, zero gross, exactly `-cost`. Live flattens at 15:55 and does not also enter, so booking
these charges a cost live never pays and would break the Unit 12a parity check.

Measured, pre-tail sample, 16 `(n, v)` combos, 107,067 trades: **1,866 = 1.74%** are zero-bar
(1.0–3.2% depending on the combo), every one exactly `-cost`. The skip removes **exactly** the
trades whose exit index equals their entry index — 0 discrepancies, and `nT` falls by precisely
the zero-bar count. Every remaining trade therefore has `exit > entry`, which also settles the
`mLb` bar-count question before it is asked: no trade contributes 0 bars.

This is also what makes "N trades = N round trips" true, so the cost convention and this skip
are one decision and are tested as a pair.

⚑ **PLAN Unit 4's done-when does not catch this.** "No trade spans 15:55" is satisfied by a
zero-bar trade *at* 15:55. The test that catches it asserts `exit > entry` and
`gate[entry + 1] == 1`.

**B. The exit fill is the last gated bar's close, whatever shut the gate.** `gate` goes 1→0 for
the scheduled clock exit and for the `max_n` blackout after a data gap; one rule covers both,
and `close[t-1]` is the last bar the position could be held on.

Measured, full sample: **2,682** gate 1→0 edges, of which **2,680 are the scheduled clock exit**
(2,658 at 15:50, 21 at 12:50 on early closes, 1 at 15:30) and **2 are not** — 2016-02-02 11:15
(one missing 5-minute bucket in an open market) and 2020-03-18 12:55 (the LULD halt). On those
two, `close[t-1]` means "assume you got out before the gap", which is optimistic whenever the
gap runs *toward* the position; it came out conservative on this sample only because the grid
was net short into both down-gaps, which is a fact about the sample and not a property of the
rule.

The 15:30 edge is 2019-08-12, the one session whose feed stops before 15:55. It counts as
*scheduled* here because the bar after it belongs to the next session, which the calendar
knows — but it is an exception for rule C below, where the question is what **live** could
have known at the time. The same edge, two different questions.

**C. Reading `gate[t+1]` for rule A is not look-ahead — with one measured exception.** The gate
is a pure function of timestamps and the NYSE calendar (`data.build_gate`), both known before
the session opens, and live knows them too. The blackout term is the exception: it keys on the
*next* bar's arrival, which live cannot know at `t`. That is **2 of 189,373 gated bars**, plus
the one session in 2,680 whose data stops early (2019-08-12, last bar 15:30). On those three
the backtest skips an entry live would have taken — a deleted trade, not a conservative one.
Fixing it would need the calendar inside the kernel signature; at 0.0016% of gated bars it is
recorded rather than fixed.

---

## 3. SPY adaptation

The sources are about ES and CL futures. Every transfer decision, with its reason.

### 3.1 Bar series — RMedV is computed on the full extended-hours session

**RMedV is computed on the contiguous extended-hours 5-minute series. Trading is gated to
10:00–15:55 ET. These are different things and must not be conflated.**

Fidelity: [M25 p.27] run header reads `Total Number of Bars=736241` over 1/1/2013–5/26/23 with
`Trading Times Constraint Start Time=0 EndTime=0` — ~281 bars/session, i.e. the full ~24h
Globex series. Meyers computes continuously and gates only trading.

Correctness: the repeated median's 50% breakdown protects against outlier **points**, not a
level **shift** between two blocks. In a window straddling a session boundary every cross-block
pair carries the gap, so the median slope tracks the gap rather than the trend. Measured on
synthetic SPY-scale data (bar sd $0.20, overnight gap $2.40, 4000 trials per N):

| N | clean sd | median RMedV, gap-straddling window | in normalized sd |
|---|---|---|---|
| 7 | 0.0921 | 0.4283 | **4.7** |
| 12 | 0.0685 | 0.2831 | **4.1** |
| 24 | 0.0487 | 0.1428 | **2.9** |

The grid maximum is 3.5. Contamination spans the first `N-1` bars of each session and peaks
near bar `N/2`. Under open-labelled bars (§3.2) 10:00 is the 7th RTH bar, so a window ending
there straddles the session boundary for every **N ≥ 8** — **17 of the 22 N-values**. An
RTH-spliced series would produce a near-daily gap-continuation entry at 10:00 and the backtest
would be measuring an overnight-gap strategy.

#### The session window — measured in Unit 1, not assumed

Rev 1 proposed the whole 04:00–20:00 extended-hours range. **Measurement narrowed it to
08:00–15:55.** Bucket completeness over 2024 SIP (fraction of the 252 sessions having a bar
in each 5-minute bucket):

| ET range | min | median |
|---|---|---|
| 04:00–06:00 | 0.849 | 0.901 |
| 06:00–08:00 | 0.885 | 0.978 |
| **08:00–09:30** | **1.000** | **1.000** |
| **09:30–16:00** | **0.996** | **1.000** |
| 16:00–18:00 | 0.921 | 0.986 |
| 18:00–20:00 | 0.889 | 0.942 |

A missing bucket is not a cosmetic hole: RMedV's slope is measured against an implied
x-spacing of one bar, so a gap silently rescales it. **The series must stop where it stops
being contiguous**, and that is 08:00.

**The window is near-contiguous, not perfectly contiguous.** Per-year minimum completeness
inside 08:00–16:00: 0.996 (2016), 0.996, 0.992, 0.992, **0.988 (2020)**, 0.996, 0.996, 0.992,
0.996, 1.000 (2025), 1.000 (2026). Pre-market coverage improved markedly over the decade
(04:00–06:00 median 0.590 in 2017 → 1.000 in 2026), so 2024 is representative of neither end;
08:00 is chosen to be safe in the *worst* years, and a 07:00 start would take 2017's minimum
to 0.936.

Over the full sample, 2016-01-04 to 2026-08-31: **257,217 bars / 2,680 sessions = 95.98 per
session.** Not every session is complete:

| bars in session | 96 | 95 | 94 | 93 | 91 | 90 | 84 |
|---|---|---|---|---|---|---|---|
| sessions | 2,652 | 14 | 7 | 4 | 1 | 1 | 1 |

**28 sessions carry holes** (none has *more* than 96 bars — no duplicates, no DST artifacts),
including all four 2020 circuit-breaker days. The `max_n` post-gap blackout covers them: a
holed session's gated-bar count drops from 71 to as low as 45. **Downstream must not assume a
constant session length.**

**Why 08:00 specifically — and note this is *not* the reason Rev 1 gave.** Rev 1 claimed
extended hours shrinks the overnight jump. Measured, it barely does:

Full sample, 2016-01-04 to 2026-08-31 (~2,720 session boundaries):

| series | median jump | p95 | median bar move | ratio |
|---|---|---|---|---|
| RTH-only, 16:00 → 09:30 | $1.060 | $5.536 | $0.130 | 8.15× |
| this window, 16:00 → 08:00 | $0.980 | $4.840 | $0.125 | 7.86× |

The gap is essentially as large either way. The actual benefit is **distance**: 08:00 is
`MAX_N = 24` bars before the 10:00 gate, so the first tradeable bar's lookback (08:05 → 10:00,
24 bars) lies wholly inside its own session and no gated bar's window ever contains the gap.
There is exactly **one bar of margin** — 08:00 itself is never in that window — which is why
the post-gap blackout runs to `i + max_n` inclusive rather than `i + max_n`: SPEC §2's
crossing rule reads `RMedV[t-1]`, so the previous bar's window must be gap-free too. The two
gate rules — the 10:00 trading window and the blackout — therefore land on the same bar, and
a test asserts they continue to agree.

> ⚠ **The blackout margin is exactly zero.** Measured over the full sample: the distance from
> each gap to the first gated bar after it is **25 bars = `MAX_N` + 1, the precise minimum**,
> at all 2,681 session boundaries; and the first gated index of a full-sample load is 24, also
> the exact minimum for n=24's `RMedV[t-1]` read. It holds — no gated bar's `t` or `t-1`
> window contains a gap, for any n in 3..24, across all 189,373 gated bars — but with no
> slack. Raising `MAX_N`, moving the session start later, narrowing the blackout, or adding an
> `RMedV[t-2]` term each break it silently. `test_unit2_gate_never_exposes_an_invalid_rmv`
> checks the invariant directly against real data; keep it.

### 3.2 Other transfers

⚑ **Measured in Unit 9: the fee model is still structurally wrong and it cannot change the Unit 9 gate.** Total cost paid over 525 pre-tail windows runs **$24.54 to $84.00 per share** across the nine filters. At **zero cost** the CL4-as-written family stays negative, and the random-filter null rises with the filters (gross null **+123.49** against net +61.48), so gross `CL2` at +157.47 sits *closer* to the null than net `CL2` at +132.11 does. The FINRA TAF per-share schedule and the SEC Section 31 rate history are therefore **not blockers** for the gate — they are worth cents on a result that misses by a factor of sixty. They remain required before any live order is sent.

| Item | Decision | Reason |
|---|---|---|
| **Bar timestamp** | a bar is labelled by its **open** (Alpaca convention). All gate comparisons use the open timestamp. | Undefined, this is worth one bar. Under open-labelling the last gated bar opens 15:50 and closes 15:55; 10:00 is the 7th RTH bar. |
| Session | **08:00–15:55 ET, 96 bars/session** | The measured contiguous range (§3.1). RMedV is computed on all 96 bars; only trading is gated. |
| Trading gate | `gate[t] = 1` iff `10:00 <= t_open < 15:55` ET | §2. Coincides with the `max_n` post-gap blackout by construction. |
| **Data feed** | **SIP for both backtest and live. IEX is not usable — measured, Unit 1.** Live therefore requires a paid SIP subscription before go-live. | Measured over June 2024 **inside the 08:00–15:55 window** (comparing over 04:00–20:00 would overstate the gap). Two independent failures. **(1) No warmup:** IEX supplies **117 bars in 08:00–10:00 against SIP's 456** — ~6 of the 24 buckets per session — so it cannot feed the pre-gate lookback §3.1 exists for; overall session coverage is 81.4%. **(2) Noise on every print:** where both feeds print, IEX closes differ from SIP by a **median 2.50¢ = 15% of a median 17¢ bar move**, with **76.3% of bars off by ≥1¢**. A repeated median resists outliers, not error on every point. (Inside RTH alone IEX coverage is ~99.9%, so the coverage failure is specifically pre-market — but the 2.50¢ noise persists everywhere.) [PLAN §8-D, resolved] |
| **Sample range** | SPY 5-min SIP, **2016-01-04 → 2026-08-31**: 257,217 bars over 2,680 sessions, 10.7 years | Alpaca equity history starts 2016. [PLAN §8-A, resolved] |
| Price adjustment | **`adjustment="split"`** | Dividend adjustment retroactively rescales all prior bars, so the cache can never be immutable, `xmult` drifts ~10% over 9 years, and live (raw prices) stops matching backtest. SPY has not split since 2005 and Alpaca data starts 2016, so `"split"` is a no-op. The ex-div gap is ~0.28% — smaller than the ordinary overnight gap already in the series. |
| Costs | $0.01/share round-trip slippage **+ SEC/TAF ≈ $0.017/share on sells** | SPY spread is a penny. TAF/SEC exceeds the slippage term at SPY $600 and scales with notional, so it is not constant across a sample where SPY went $180→$650. |
| Share size | **100 shares**; all backtest figures reported **per share** | Sizing is then a pure multiplier and the per-share edge vs fixed costs is directly readable. |
| **Early closes** | the EOD exit is **`session_close − 5 min`**, from the exchange calendar — 15:55 normally, **12:55 on a 13:00 close** | NYSE closes at 13:00 on ~2 sessions a year (**21 in this sample**) and SPY keeps printing afterwards, so a fixed 15:55 leaves the gate open across ~3 hours of thin post-close prints. Measured before the fix: **589 gated bars after 13:00**, on volume ~3% of the morning's. Also a hard live-parity failure — Alpaca rejects market orders outside RTH. Calendar cached at `cache/nyse_calendar.json`. |
| Intraday halts | any inter-bar gap > 5 min zeroes the gate for the next `max(N)` bars | Circuit breakers (2020-03-09/12/16/18) splice an intraday price gap into the window — the same failure as §3.1. |
| Direction | long/short | [M25]. Requires a margin account; SPY is trivially shortable. |
| Minimum live capital | **$25,000** | PDT: ~8 round trips/week trips the rule in week one, and the consequence is closing-only for 90 days. |


#### IEX at the signal level — measured in Unit 4

Unit 1 rejected IEX on price. Unit 4 is the first point at which that can be restated as
trades, which is what actually matters. Both feeds fetched for June 2024 and run through the
identical pipeline (session mask → `build_gate` → `rmv_all_n` → per-feed `xmult` → `simulate`):

| | SIP | IEX |
|---|---|---|
| session bars | 1,824 | 1,484 |
| **gated bars** | **1,349** | **982** (72.8%) |
| `xmult` | 3.100 | 3.271 (+5.5%) |
| trades over 16 `(n, v)` combos | 1,061 | 777 (**73.2%**) |

On the 982 bars **both** feeds gate, the signals disagree on **3.35%** overall and up to
**11.9%** at `n=3, v=0.25`. The coverage failure and the noise failure compound: IEX's missing
pre-market bars trip the `max_n` blackout, so a quarter of SIP's tradeable bars are not
tradeable at all on IEX, and on the ones that are, one signal in eight can differ at low `N`.

The mechanism is checked offline in the test suite rather than over the network: injecting
IEX's measured 2.50¢ of per-bar noise into the cached SIP series moves **29.7%** of trades. A
repeated median resists outlier *points*, not error on *every* point.

⚑ **The table above is re-derivable, not quoted.** The June 2024 IEX bars are cached beside
the SIP bars (`cache/SPY_5min_iex.npz`, 18 KB) exactly so this does not rest on one
un-repeatable network run, and `test_unit4_iex_signal_divergence_on_real_feeds` recomputes
every figure in it from that cache. SIP's 1,824 session bars and 1,349 gated bars are also
checkable by arithmetic: June 2024 has 19 trading days, and 19 × 96 and 19 × 71 are exactly
those two numbers.

⚑ **The cost constant needs Unit 9's attention, not Unit 4's.** `$0.017/share on sells` folds
two structurally different fees into one notional-scaled number. FINRA's TAF is charged **per
share** and does not scale with notional at all; the SEC Section 31 fee is charged on notional
and its rate is reset periodically. Folding them together means one of the two is modelled with
the wrong price sensitivity for the whole sample, and a rate pinned at one point in time is
wrong at the others.

⚑ **Unverified, and deliberately left that way here.** The per-share TAF figure and the
Section 31 rate history are external facts, not repo measurements, and this project does not
put unmeasured numbers in the durable record. Unit 9 already owns *"verify (not re-apply) the
§1.3 cost model"*; the verification is a source lookup against FINRA's TAF schedule and the
SEC's published Section 31 fee-rate advisories for each year in the sample, and the split
above is what it should be looking for. Unit 4 takes a scalar `cost` per share per trade and
is agnostic to how it was built.

### 3.3 Parameter grid

[M25 p.7]:

```
N    : 3    .. 24   step 1      (22 values)
vup  : 0.25 .. 3.50 step 0.25   (14 values)   -- normalized sd
vdn  : 0.25 .. 3.50 step 0.25   (14 values)   -- normalized sd
                                = 4312 combinations
```

[M05 p.4] used raw units and a coarser grid (`N` 10..75 step 5, `vup`/`vdn` 0.02..0.40 step
0.02). Superseded — raw-unit thresholds do not transfer across `N` or across symbols.

> [M25 p.7] states these ranges produce **4508** combinations. 22×14×14 = 4312; 4508 = 23×14×14.
> One of the paper's stated ranges is off by one N-value. See §9-B.

**Combo ordering — pinned in Unit 6.** The stored table has one row per combination and does
not record the parameters, so the index *is* the mapping:

```
c = a * len(vs)**2 + i * len(vs) + j     n = ns[a], vup = vs[i], vdn = vs[j]
```

`a`-major, then `vup`, then `vdn` — 196 consecutive rows per N. Both `vup` and `vdn` are drawn
from the same 14-value grid, so the 196 include every `vup != vdn` pair in both directions.
`rmv.run_grid` emits this order and Units 7–11 read a winning row back to its parameters with
it; the `a`-major part is also why one RMedV row stays hot across a whole 196-pair sweep
(PLAN §2.1).

⚑ **The effective grid is smaller than 4312.** Measured over 24 real pre-tail windows,
distinct trade sets number **2640–3652 (median 3202)**, because neighbouring `(vup, vdn)`
pairs frequently cross the same bars. Distinctness tracks the trade count (correlation +0.94
with mean `nT`) and therefore falls as N rises — 181.6 of 196 at `n=3`, 130.2 at `n=24`. PLAN
§Unit 9's comparison multiplier uses the distinct count, not 4312.

---

## 4. Walk-forward scheme

[M25 pp.4, 6], [M05 p.4].

- **In-sample (IS)**: `[friday_end - 30 days, friday_end]` — a **30-day delta, 31 calendar
  days inclusive**, ending on a Friday.
- **Out-of-sample (OOS)**: the following **Mon–Fri** trading week.
- **Step**: 7 days. Windows are generated across the whole series.

Read off [M25] Table 1, which is the governing source (see below):

| | IS start | IS end | OOS start | OOS end |
|---|---|---|---|---|
| first row, p.17 | 11/12/14 (Wed) | 12/12/14 (**Fri**) | 12/15/14 (Mon) | 12/19/14 (Fri) |
| a later row, p.25 | 10/18/23 (Wed) | 11/17/23 (**Fri**) | 11/20/23 (Mon) | 11/24/23 (Fri) |

Both IS spans are a 30-day *delta*, i.e. 31 days inclusive — "30 calendar days ending Friday"
read the obvious way generates `10/19/23–11/17/23` and is off by one day on every window.
This is exactly the `<=`/`+1 day` bug class Unit 7 is told to hunt for.

Every OOS section is a trading week, which is the granularity `%P`, `lpr`, `wpr`, `Blw` and
`LLp` are all defined on. Holidays shorten an OOS week without changing its date span — the
11/20–11/24/23 week above contains Thanksgiving and trades four sessions.

> **[M25] contradicts itself on the first window, and Table 1 governs.** p.4's prose says
> *"the first in-sample week will be from 11/13/2014 to 12/12/2014 and the first out-of-sample
> week will be from 12/16/14 to 12/19/14"*. Table 1 p.17 row 1 says IS `11/12/14–12/12/14`,
> OOS `12/15/14–12/19/14` — a full Mon–Fri week with no holiday in it (Christmas is 12/25).
> The prose is wrong on both the IS start and the OOS start. See §9-H.

A **PWFO file** is one window's output: **one row per parameter combination**, carrying that
combination's IS metrics *and* its OOS metrics side by side [PWFO]. OOS results are therefore
precomputed for every combo, not just the selected one — which is what makes filter evaluation
cheap.

**Invariant:** `max(IS timestamp) < min(OOS timestamp)` for every window.

⚑ **Pinned in Unit 7 — the generator's boundary rules.** The scheme above leaves four
things open, and each is the `<=`/`+1 day` class this section already warns about.

- Both spans are **inclusive date ranges** in ET, not bar counts: IS is
  `[friday - 30 days, friday]` and OOS is `[friday + 3, friday + 7]`. Since 30 mod 7 = 2,
  `friday - 30 days` is always a **Wednesday** — measured on SPY 2016–2026, the IS half
  begins exactly there in 518 of 525 windows and ends exactly on its Friday in 510, the
  rest being Wednesday and Friday holidays. Under the 30-day-*inclusive* misreading the
  first figure is 0 of 525, which is what makes the census a usable check.
- A window is emitted only when its IS start is at or after the first bar's date **and**
  its OOS Friday at or before the last: the partial week at each end of the sample is
  dropped, never run short. Consecutive Fridays then make the OOS weeks partition their
  own range exactly once — no overlap, no gap — which the generator asserts.
- Because both halves are date ranges they end on the last bar of a session, which is
  ungated by construction (the gate closes at 15:55 or `session_close - 5`, and the last
  bar opens there). That is load-bearing, not incidental: `_simulate` treats the last bar
  of any slice as the last gated bar of a run, so a window cut by bar count or at a
  mid-session timestamp silently loses an entry and force-closes at the cut. The generator
  asserts `gate[-1] == 0` on both halves and rejects the series otherwise.
- The final 6 months are withheld on each window's **OOS end**, so a window contributes to
  `pwfo_tail.npy` if *any* of its OOS bars is in the tail. Identical to keying on the OOS
  start while the boundary falls on a weekend, and still correct if it moves.

[M25] ran 546 weeks of CL (12/19/2014–5/30/2025), used 517 for the filter search, and withheld
the final 29 weeks (11/15/24–5/30/25) as a genuinely untouched future period [M25 p.8].

---

## 5. Filters

A **filter** selects one row from a window's ~4312 IS rows. Structure is always
`screens → rank → pick`. Filters are applied to **in-sample columns only**; OOS columns exist
solely to score the result.

| Name | Screens | Rank | Pick | Source |
|---|---|---|---|---|
| `meyers2005` | `1 <= PF <= 2`, `lr <= 3`, `nT >= 16` | — | max `eq2R2` | [M05 p.6] |
| `CL2` | `PF < 4`, `lr < 3`, `eqR2 < 80` | bottom-50 `mLb` | min `mLTr` | [M25 p.11] |
| `CL4` | `lr <= 3`, `eqR2 <= 50` | bottom-10 `mLb` | min `mLTr` | [M25 p.8] |

Naming grammar, from [M25 Figure 2 col A p.14]: `b10mLb|pf<5|lr<3r2<50-mLTr` reads as
*bottom-10 by `mLb`, screened on `pf<5` and `lr<3` and `r2<50`, picked by minimum `mLTr`*.
`b10mLb` means the bottom 10 rows **after** the screens are applied [M25 p.8].

**Design intent, worth understanding before treating these as tuning knobs.** `CL4` screens
`eqR2 <= 50`, deliberately discarding the *smoothest* in-sample equity curves; `meyers2005`
caps `PF <= 2`. Both encode the same belief [M05 p.5]: *"if we eliminate the optimization cases
with the very best performance results we are sure to eliminate many of the data mining system
input parameters that fitted the past spurious movements."* The best-looking in-sample fit is
the most overfit.

Filter drift is expected and is a finding, not a defect. [M25 p.11]: the four CL papers found
four different filters, and *"the current paper's methodology should be run every 6+ months."*

### Pinned in Unit 8 ⚑

The source is silent on three things a filter run cannot avoid deciding. All three change
the reported result, none of them raises, and each is settled here rather than in the code.

**Nine filters, not three.** §9-D's two `r2` readings and §9-E's two `mLTr` conventions
are both open and both must be run (PLAN §1.5). They expand `CL2` and `CL4` into four
variants each. `meyers2005` screens no `r2` column and its pick is an argmax over `eq2R2`,
so both transforms are no-ops on it and it stays one filter. **Nine is the count that
enters §6.5's `K`**, and a variant that silently collapses or duplicates moves the
significance of the whole project — so the expansion drops a transform that changed
nothing, rather than naming twelve.

**The rank has no direction.** [M25 p.8]'s grammar is `b10mLb`, *"the bottom or minimum 10
mLb rows"*, and all three baselines rank that way. §6.6 measured what the other direction
does — a **top**-k on `mLb` puts every no-loser row, `+inf` sentinel and all, at the head
of a pool it can never be displaced from — so the filter grammar admits `bottom` only. Any
filter needing the other direction reopens this section first.

**`oW\|oL` is a ratio of dollars, not of counts.** §6.3 col L reads *"Average OOS
winning trades / average OOS losing trades"*, which is also readable as a count ratio. It is
not: col F of the same table says *"Average **number** of OOS trades per week"*, so [M25]
writes "number of" when it means a count, and col L's *"average ... trades"* is the average
value of one. Ours is therefore `(sum(ownp)/sum(ownt))` over `|(sum(osnp)-sum(ownp))/(sum(ont)
-sum(ownt))|`, reported positive. ⚑ The losing side is a *difference*, because the six OOS
columns store only the winners and the total — so §6.6's net-zero trades, which are neither
winner nor loser, land in the loser count. Measured 0 of 686,565 real trades sit on that
boundary; `cost = 0` is a legal argument and would produce them.

**The zero-trade convention.** §6.4's two cases are kept distinct on every record: a week
where a row was selected and fired no signals carries its `N`/`vup`/`vdn`, a week where no
row passed the screens carries none. Both contribute a 0 to `toNP` and **both stay in the
denominator of every aggregate** (§9-K). ⚑ Measured over 525 pre-tail SPY windows, case 2
does not occur at all — no row fails the screens in any of 4725 filter-windows — while
case 1 runs 92 to 179 of 525 (17.5%–34.1%), two and a half times [M25]'s rate. See §6.4.

---

## 6. Metric definitions

### 6.1 In-sample, per parameter combination — verbatim [PWFO]

All 31 are transcribed because this file is the only record. **★ marks the 18 we implement**
(PLAN §1.6) — the ones the three baseline filters reference, plus the aggregates Unit 9 reports.
The rest are recorded for a later filter search, not built now.

| Key | Definition (verbatim) |
|---|---|
| ★ `mTrd` | The Median of All Trades in the Test(In-Sample) Section |
| ★ `tnp` | Total Net Profits of All Trades in Test(In-Sample) Section |
| ★ `nT` | Number of Trades |
| ★ `%P` | Percent Profitable Trades |
| ★ `PF` | Profit Factor |
| ★ `std` | Standard Deviation of Trades |
| ★ `t` | Student t-statistic. Used to determine the probability that the Ave Trade Profit is > 0 |
| ★ `mWTr` | Median Of The Winning Trades |
| ★ `mLTr` | Median Of The Losing Trades |
| `mWT\|LT` | Ratio of Median Winning Trades /Median Losing Trades |
| ★ `mLb` | The Median of Bars in Losing Trades |
| `tLb` | Total Losing Bars |
| ★ `mWb` | The Median of Bars in Winning Trades |
| `tWb` | Total Winning Bars |
| `mWb\|mLb` | Ratio of Median Winning Bars to Median Losing Bars |
| `tWb\|tLb` | Ratio of Total Winning Bars to Total Losing Bars |
| ★ `dd` | Maximum Drawdown |
| ★ `llt` | Largest Losing Trade |
| ★ `wr` | Maximum Consecutive Winners In-A-Row |
| ★ `lr` | Maximum Consecutive Losers In-A-Row |
| `eqTrn` | Slope of Trade Equity Regression Line |
| ★ `eqR2` | Trade Equity Regression Trend Line Coefficient of Correlation R2 |
| `eqDev` | Median of The Deviation between the absolute Values of Each Trade Equity Minus The Equity Regression Trend Line |
| ★ `ktau` | Kendall Rank Coefficient - Equity Nonparametric Coefficient Of Correlation |
| `eq2V` | Velocity of Equity Curve Least Squares 2nd Order Polynomial Line |
| `eq2A` | Acceleration of Equity Curve Least Squares 2nd Order Polynomial Line |
| ★ `eq2R2` | Equity Least Squares 2nd Order Polynomial Line Coefficient of Correlation R2 |
| `eq10` | Projected Equity 10 Trades In Future Using Equity Curve Least Squares 2nd Order Polynomial Line |
| `mKr` | Modified K-ratio = Slope of Equity Regression Line/ (The Average of The Absolute Values of The Equity at Each Trade Minus The Equity Regression Trend Line) |
| `m(ru-p)` | Median of all Trades{Maximum Trade Runup minus Final Trade Profit} |
| `m(p-rd)` | Median of all Trades{Final Trade Profit minus Maximum Trade Rundown} |

**Equity curves here are trade-indexed**, not time-indexed — [M05 p.6] defines `R22` against
*"the trade Equity line"*. `eqR2` is the **straight-line** fit and `eq2R2` the 2nd-order fit.

⚑ **The 0–100 scale rests on [M25 p.8], not on Figure 1.** The self-contained evidence is the
filter's own screen — *"we want the R2 equity trend line correction to be <50, **r2<50**"* —
because a threshold of 50 against a quantity bounded by 1 would pass every row in the table
and make `CL4`'s stated design intent vacuous. [M25 Figure 2 Row 4] corroborates it with
`eqR2 = 82` beside `KTau = 93`. **Rev 1 of this file cited [M25 p.13]'s `R² = 0.9496` as an
`eq2R2` value; it is not one.** It is an Excel chart trendline label on Figure 1's *weekly,
time-indexed* OOS equity curve, and Excel always prints R² as a 0–1 fraction. The same figure
carries a second label, `R² = 0.9285`, for the net curve. **No published `eq2R2` value exists
at the per-combination level** — Figure 2 has no such column, Table 1 has no such column, and
[M05 p.6] carries `r22` only as a ranking-variable name — so `eq2R2`'s stored scale is a
project convention (§6.6), not a transcription.

### 6.2 Out-of-sample, per parameter combination — verbatim [PWFO]

| Key | Definition (verbatim) |
|---|---|
| `osnp` | Out-Of-Sample Period total Net Profit |
| `ont` | Out-Of-Sample Period Total Number Of Trades |
| `ownp` | Out-Of-Sample Period Winning Trades total Net Profits |
| `ownt` | Out-Of-Sample Period Winning Number Of Trades |
| `ollt` | Out-Of-Sample Period Largest Losing Trade |
| `odd` | Out-Of-Sample Period Drawdown |

### 6.3 Aggregate, per filter across all OOS periods — [M25 pp.14–15]

| Col | Key | Definition |
|---|---|---|
| B | `toGP` | Total out-of-sample gross profit over all OOS periods |
| C | `toNP` | Total OOS net profit = `toGP - (# trade weeks × cost)` |
| D | `aoGP` | Average OOS gross profit per OOS period |
| E | `aoTr` | Average OOS profit per trade |
| F | `ao#T` | Average number of OOS trades per week |
| G | `#` | Number of OOS periods this filter produced a profit or loss. *For some OOS periods no strategy inputs satisfy the filter's criteria and no trades are made.* |
| H | `std` | Standard deviation of the OOS period profits and losses |
| I | `skew` | Skew of the OOS period profits and losses |
| J | `kur` | Kurtosis of the OOS period profits and losses |
| K | `t` | Student t statistic for the OOS periods |
| L | `oW\|oL` | Average OOS winning trades / average OOS losing trades |
| M | `%Wtr` | Percentage of OOS winning trades |
| N | `%P` | Percent of all OOS periods that were profitable |
| O | `LLTr` | Largest losing trade |
| P | `LLp` | Largest losing OOS period |
| Q | `eqDD` | OOS equity drawdown |
| R | `wpr` | Largest number of winning OOS periods in a row |
| S | `lpr` | Largest number of losing OOS periods in a row |
| T | `v20` | Equity velocity for the latest 20 periods |
| U | `KTau^2` | Kendall rank coefficient (non-parametric dependence test) |
| V | `eqR2` | Correlation coefficient (R²) of a straight-line fit to the equity curve |
| W | `Blw` | Max number of OOS periods the OOS equity curve failed to make a new high |
| X | `BE` | Break-even in OOS periods — assuming normality, the number of OOS periods you would have to trade for a 98% probability that OOS equity is above zero |
| Y | `tkr\|bl` | `t * ktau * eqR2 / BE` — a measure of how good the filter fits |
| Z | `Prob` | Probability the filter's `toNP` was due to pure chance. **Must be multiplied by the total number of filters examined.** |

**Row 1 — run-level scalars** [M25 p.14]: `A` = PWFO stub · `B` = file start date · `C` = file
end date · `D` = number of OOS periods · `N` = **bootstrap average** · `O` = **bootstrap
standard deviation** · `U` = cost/slippage per trade · `Z` = **number of filters run**.
`N`, `O` and `Z` are the three inputs to §6.5; nothing else supplies them.

**Cols AB–AG — the withheld future period** [M25 pp.14–15]. This is the output shape of the
"opened once, at the end" tail run, so it must exist before Unit 9 can report one:

| Col | Key | Definition |
|---|---|---|
| AB (row 1) | — | Future PWFO file start date |
| AC (row 1) | — | Future PWFO file end date |
| AD (row 1) | — | Number of PWFO files **not** included in the main run |
| AG (row 1) | — | Number of total OOS + future PWFO files |
| AB | `toGPx` | Total gross profit for the future excluded periods |
| AC | `toNPx` | Total net profit for the future excluded periods |
| AD | `aoTrx` | Average profit per trade, future excluded periods |
| AE | `aoNTx` | Average trades per week, future excluded periods |
| AF | `#x` | Number of future excluded periods this filter actually traded |
| AG | `tOnpNet` | `toNP + toNPx` — total net profits, OOS + future |

### 6.4 Per-week report columns — [M25 Table 1 p.17]

The **actual column headers** [M25 Table 1 p.17], in order:

```
In-Sample Dates | Out-of-Sample Dates | osnp | NOnp$13 | ont | ownp | ownt
                | ollt | odd | EQ | NetEq | N | vup | vdn
```

The page's *legend* uses different names than its own headers — it defines `ogp`, `Equity` and
`osnp$20` (the last as `ogp - ont*13`, so even the `$20` in the name disagrees with the
formula), none of which appear as headers. `N`, `vup` and `vdn` **are** defined in the legend
(*"N = N the lookback period"*, and `vup`/`vdn` as the velocity thresholds for a buy and a
sell signal); the two it omits entirely are `ownp` and `ownt`. Mapping, header ← legend:

`osnp` ← weekly OOS **gross** profit (`ogp`) · `NOnp$13` ← weekly OOS **net** profit
= `osnp - ont*13` · `ont` trades in the OOS week · `ownp` winning-trade gross profit ·
`ownt` number of winning trades · `ollt` largest losing trade · `odd` OOS drawdown ·
`EQ` running sum of weekly gross (`Equity`) · `NetEq` running sum of weekly net ·
`N`/`vup`/`vdn` the parameters the filter selected from that IS section.

**The `$13` cost decomposes as $10 round-trip slippage + $3 round-trip commission** [M25 p.10].
Our equivalent is in §3.2 and is *not* $13 — SPY is not CL.

*"Blank rows indicate that no out-of-sample trades were made that week"* [M25 p.17].

**Two distinct zero cases, which must not be conflated** — 71 of [M25]'s 517 weeks had no
trades [M25 p.10]. For that paper's filter (`%P` = 57) the two conventions differ by
`57 × 71/517 ≈ 8` points; the theoretical maximum is 13.7:

1. **Params selected, no signals fired** — Table 1 shows `N/vup/vdn` filled with `osnp = 0`.
2. **No row passed the filter** — [M25 p.15 Col G]. No params exist for that week.

⚑ **Measured in Unit 8, and the 13.7 above is this paper's rate, not a bound.** Over 525
pre-tail SPY windows, `meyers2005` — whose `nT >= 16` screen forces an active row — goes
silent (case 1) in **9 weeks**, and the other eight baseline filters in **92 to 179 weeks,
17.5% to 34.1%**, against [M25]'s 71 of 517. Dropping those weeks from `%P`'s denominator
moves it by **+0.8 points for `meyers2005` and +8.3 to +17.4 for the rest**. Case 2 does
**not occur once** in 4725 filter-windows: every filter finds a row every week, and the
distinction therefore lands entirely on case 1 in this sample.

### 6.5 Significance test — [M25 pp.8–9]

The **mirror random filter** bootstrap. For each of the `W` windows pick a *uniformly random*
row's OOS net profit instead of the filter's choice; sum across windows for one random `toNP`.
Repeat 5000 times. The mean and sd of that distribution give the chance probability of the real
filter's `toNP`.

The null is **not zero**: [M25 p.9] reports a random-filter average of **+$65.3/week** with sd
**$67.3** on CL. A filter can beat zero and still be worthless.

**How `Prob` is obtained** [M25 p.9], worked through on that paper's own numbers — the method,
not just the endpoints, because Unit 9 implements this:

```
1. filter mean weekly net      = toNP / n     = 176932 / 446   = 396.7
2. z vs the bootstrap null     = (396.7 - 65.3) / 67.3         = 4.92 sd
3. one-sided normal tail       = P(Z > 4.92)                   = 4.23e-7
4. correct for K filters tried = 1 - (1 - p)^K  ~=  K * p
                               = 115320 * 4.23e-7              = 0.049
```

Verified: this chain reproduces the paper's 4.92 / 4.27e-7 / 0.049 exactly. Step 4 is the whole
reason `K` must be counted honestly — at `K = 115320`, a result 4.92 sd out is only *just*
significant. The linear approximation holds while `K*p << 1`.

> ⚠ **`n = 446`, not the 501 the text names, and the choice decides the result.**
> [M25 p.9] writes *"the filter toNP/ (# of OOS) periods traded or 176932/501=396.7"*, but
> `176932/501 = 353.2`. 501 is **Col D (`aoGP`)**; the divisor that actually yields 396.7 is
> **Col G = 446**, *"the number of oos periods this filter produced a profit or loss"* — i.e.
> zero-trade weeks are **excluded** from the filter's mean.
>
> The bootstrap null is not built that way. p.9 defines it as `Average Random toNP / 517` —
> **all** periods, zero-trade weeks included. Comparing a 446-denominator mean against a
> 517-denominator null flatters the filter, and by more than the margin of the result:
>
> | denominator | mean/week | z | `K·p` | verdict |
> |---|---|---|---|---|
> | 446 (traded only, as published) | 396.7 | 4.92 | **0.049** | significant |
> | 517 (all periods, consistent with the null) | 342.2 | 4.11 | **2.234** | **not significant** |
>
> **Our convention: the same denominator on both sides — all OOS periods, zero-trade weeks
> counted as zero.** It is the conservative reading and the only internally consistent one.
> See §9-K.


### Pinned in Unit 9 ⚑

Five more things the source leaves open, all of them in §6.3's columns, and one measured
fact about the null that no reading changes.

**`KTau^2` is stored signed and unsquared**, despite the column name. [M25 Figure 2] prints
`KTau = 93` beside `eqR2 = 82` in the same row, so the printed column is on the same 0–100
scale as the other two correlation columns (§6.6). A square is derivable from the signed
value and the sign is not, so the recoverable form is the one stored — and `tkr|bl` reads it
as-is. Two of the nine filters return a **negative** `KTau^2` (−72.56, −54.55), which a
squared column could not express and which is the honest description of their equity curves.

**`v20` is Δequity/Δperiod over the last 20 periods**, i.e. the mean weekly net of those 20,
which is exactly `(eq[-1] - eq[-21]) / 20`. §6.3 col T says "equity velocity" and an OLS
slope through the same 20 points is the other reading; velocity is what the word names.

**`aoTr` is net per trade, not gross.** §6.3 col E says only "profit" where col B of the
same table is gross and col C net, so the column is ambiguous. Net is the reading §9-K
already took for every other denominator in this table.

**`toGP` is reconstructed, not re-simulated.** Cost lives inside `_simulate` per trade
(§6.6), so gross is `osnp + ont × cost` on the stored columns — exact arithmetic rather than
a second simulation that could disagree with the first.

**`tkr|bl` is 0 when `BE` is `inf`.** §6.3 col Y is `t · ktau · eqR2 / BE`, and a filter that
never breaks even would otherwise propagate `inf` through a product whose other factors are
already signed. It scores zero, which is where a filter that does not make money belongs.

⚑ **The null's moments are exact, not estimated.** The mirror random filter is a sum of
independent per-window uniform draws, so `E = sum_k mean_k` and `Var = sum_k var_k` over
each window's stored OOS column. Measured over 525 pre-tail SPY windows: **+61.48/share,
sd 97.60** — against [M25]'s +$65.3/week with sd $67.3 on CL, and confirming this section's
central point that **the null is positive**. The 5000-iteration bootstrap gives 61.43 /
96.96; `pwfo.null_moments` is what says whether it converged, and `pwfo.bootstrap` is what
carries the distribution's shape.

⚑ **The normal tail this section reads off that distribution is earned, not assumed.**
Step 3 takes a one-sided Gaussian tail, and the per-window OOS columns are badly non-normal
— skew from **−4.20 to +1.00** across the 525 windows. The sum is not: the 5000-draw
distribution has skew **−0.054** and excess kurtosis **−0.12**, and its empirical tail for
`CL2` (**0.2434**) tracks the normal `Prob` (**0.2346**). That is the central limit theorem
over 525 independent draws, and it is what licenses steps 2–3. Assert the shape of the
*sum*, never of the summands.

⚑ **Measured: no baseline filter is significant against it.** `Prob` runs 0.235 to 0.989
across the nine, so `K·p` at `K = 13` runs **3.05 to 12.85** against a 0.05 bar. `z = 2`
would need `toNP > 256.67`; the best filter reaches 132.11, and a per-window oracle with
perfect foresight reaches 4751.67. See PLAN §3 Unit 9.

---

### 6.6 Storage contract — pinned in Unit 5 ⚑

The sources define *what* each metric is and are silent on sign, scale and the degenerate
case. All three change which row a filter selects, none of them raises, and a reference
implementation that shares the misreading agrees with the kernel. So they are pinned here.

**Signs.** Loss metrics are stored **negative**: `mLTr`, `llt`, `dd`, and their OOS twins
`ollt`, `odd`. This is both papers' own convention, twenty years apart — [M25 Figure 2 Row 4]
`LLTr = -3540`, `LLp = -6640`, `eqDD = -10970`, and [M05 p.6]'s `eqDD` column reads −239,
−475, −1038, −1025 with `llw` at −188, −475, −850, −1025. Storing the signed value is what
lets Unit 8 run §9-E's two selection conventions: the magnitude reading is `abs()` of this
one, and the reverse is not derivable. `mLb`/`mWb` are **not** loss metrics — they are bar
counts, always ≥ 1, and never negated; sign-flipping them would turn `b10mLb` into a top-k.

**Scales.** `%P` is 0–100. The oracle is [M25 p.10]'s **`%Wtr`** — *"The % of all oos trades
that are positive … was 45%"* — because that is the per-*trade* quantity §6.1's `%P` defines;
[M25]'s own `%P` column is a per-*period* percentage and a different metric (see the naming
collision below). Both are integers on 0–100 in Figure 2 (cols M and N). `eqR2`, `eq2R2` and
`ktau` are ×100, `ktau`
signed, so all three correlation columns share one scale and a filter threshold literal
cannot mean 0–1 against one column and 0–100 against its neighbour. Only `eqR2`'s scale is
sourced; the other two are convention. `PF` and `t` are dimensionless; everything else is
per-share dollars or a raw count.

⚠ **[M05] and [M25] do not share scale conventions.** [M05 p.6] stores `%P` as a fraction —
0.69, 0.75, 0.81, 0.62 — where [M25 Figure 2 col N] stores 57. This is a direct caution
against inferring one paper's scale from the other's.

**The winner/loser partition.** A trade is a winner at `net > 0`, a loser at `net < 0`, and
`net == 0` is **neither** — it still counts in `nT` and `tnp`, and it breaks both streaks.
Measured, 0 of the 686,565 trades produced by eight real windows sit on the boundary (`cost`
is 0.027 and gross moves in cents), but `cost = 0` is a legal argument and a flat price move
then produces one. The
strict predicate is also what keeps the sentinels unambiguous: a defined `mLTr` is strictly
negative, so `mLTr == 0.0` can only mean "no losing trades".

**`ownp`/`ownt` are the NET winner set** — §6.2 reads *"Winning Trades total **Net**
Profits"*. Measured on the full pre-tail sample the two sets genuinely differ: 4,810 gross
winners against 4,688 net at `n=6, v=0.5`, and 1,517 against 1,502 at `n=12, v=1.0`. `ownp`
is numerically PF's numerator; one accumulator serves both.

**Dispersion** uses `ddof=1`, matching §1.2's `xmult`. Measured at the real median trade
count the population form runs 2.4% low, and 18.4% low in the tail — not a rounding
difference once it reaches `t`. `t = mean / (std / sqrt(nT))`.

**Drawdown is measured from a zero baseline**, so an opening loser is already a drawdown.
[M25 Table 1] settles this rather than leaving it to taste: on its all-loser weeks `odd`
equals the *full* cumulative loss — 01/07/15 (`ont` 2, `osnp` −2020, `odd` −2020) and
03/25/15 (`ont` 4, `osnp` −1040, `odd` −1040) — where a peak seeded from the first equity
value would give −990 and −220.

**Degenerate rows are common and every sentinel points in its consumer's fail-safe
direction.** Measured per window over 24 real pre-tail windows, **0.00–5.96%** of the 4312
combos traded not at all and **2.67–12.66%** traded fewer than three times — a range, not a
constant, because the spread across windows is an order of magnitude and any single-window
figure understates the worst case. Every filter meets these rows every window. The sentinels
are deliberately *not* all the same value:

| Metric | Undefined when | Stored | Why that value |
|---|---|---|---|
| `PF` | no losing trade (incl. no trades) | `+inf` | fails `PF < 4` and `1 ≤ PF ≤ 2`, so `meyers2005` and `CL2` reject it |
| `eqR2` | < 2 trades, or flat equity | `100.0` | fails `eqR2 < 80` and `eqR2 ≤ 50`. **A `0.0` passes both** — measured, that admitted 321 and 260 of 4312 combos into `CL4` on two real windows |
| `eq2R2` | < 3 trades, or flat equity | `0.0` | `meyers2005` **picks** max `eq2R2`, so the safe sentinel is the one that cannot win an argmax — the opposite direction from `eqR2` |
| `mLb`, `mWb` | no loser / no winner | `+inf` | `CL2` and `CL4` rank on the **smallest** `mLb` ([M25 p.8]: *"b10mLb means the bottom or minimum 10 mLb rows"*), so `0.0` would put every no-loser and no-trade row at the head of the pool, displacing real candidates it can then never beat |
| `mTrd`, `mWTr`, `mLTr`, `llt`, `dd`, `std`, `t`, `lr`, `wr` | various | `0.0` | for `llt` and `dd` this is [M25 Table 1]'s published value on its all-winner week (12/15/14: `ont` 4, `ownt` 4, `ollt` 0, `odd` 0) and on its zero-trade weeks |

No sentinel distinguishes "no trades" from a real value — `nT` is the only column that does,
and screening it is Unit 8's (§Unit 8's zero-trade convention). [M25 Table 1] confirms the
zero-trade row is a real, reportable outcome: its 01/14/15, 01/21/15 and 01/28/15 weeks carry
`N`/`vup`/`vdn` filled and every OOS metric at 0.

⚠ **Each of these values is fail-safe in exactly ONE comparison direction, and the table
above is the direction the three shipped filters use.** Reverse the comparison and the same
sentinel becomes the worst possible choice. Measured over 24 real pre-tail windows: a filter
picking **max `eqR2`** selects a row with `nT <= 2` in **24 of 24** windows, and one picking
**min `eq2R2`** does so in **24 of 24** — in every case the sentinel itself, or a trivially
perfect 2-point fit, wins. A screen `PF > x` passes every no-loser row; a **top**-k rank on
`mLb` puts every no-loser row at the head. Nothing in the metric row can prevent this, so any
new filter — and in particular §Unit 10's generated space — must either carry an `nT` floor
or be restricted to the directions above.

⚠ **The sentinels stop a degenerate row being *selected*; they do not impose a trade-count
floor.** Their protection ends at `nT < 2`, because from two trades up `eqR2` is a real fit
and slides under `eqR2 <= 50` honestly. Measured over the same 24 windows with the as-stored
`mLTr` convention, `CL4` selects a row with `nT < 5` in **5 of 24** windows (`nT == 3` in 2)
and `CL2` in **5 of 24**. This is faithful to [M25] — neither published filter has a trade
count screen, and only `meyers2005` has one — so it is recorded, not corrected.

⚠ **`+inf` is safe to sort but not to average.** `PF`, `mLb` and `mWb` can hold `+inf`, so a
column mean is `inf` and a standard deviation `nan`; measured, `mean(PF)` is non-finite in 23
of 24 real windows. Sorting, `argsort`, `partition` and a float32 `.npy` round trip are all
exact. §6.3's aggregates are OOS-side and unaffected, but any IS-side diagnostic or z-scored
rank metric has to mask first.

⚑ **`meyers2005`'s `nT >= 16` screen is load-bearing, not decoration.** `eq2R2` is exactly
100 for any 3-trade row — a quadratic through three points is an exact fit — and measured on
two real windows 119 and 92 of 4312 combos score exactly 100, *every one of them* at
`nT == 3`. None survives `nT >= 16`; the best that does scores 95.9 and 98.3. Relax that
screen, or write any new filter that picks max `eq2R2` without one, and the pick collapses
onto three-trade rows tie-broken arbitrarily.

⚑ **Our P&L columns are net; [M25]'s are gross.** Meyers subtracts cost as a post-hoc weekly
aggregate (`NOnp$13 = osnp − ont*13`), so **four** of Table 1's columns are on a gross basis —
`osnp` (the legend's `ogp`), `ownp`, `ollt` and `odd`. 01/07/15 has `osnp` −2020 and `odd`
−2020 where the net figure is −2046. Ours carry `cost` inside `_simulate`, per trade, so all
four of our equivalents are net. This is a deliberate divergence, recorded because it will
look like a defect to anyone attempting a numeric parity check against the paper.

⚑ **Four metric names mean two different things in this file**, and Unit 9 prints both blocks
side by side. §6.1's `%P`, `t`, `std` and `eqR2` are **per parameter combination, over
trades**; §6.3's cols N, K, H and V of the same names are **per filter, over OOS periods**.
`llt` and `LLTr` collide the same way. Disambiguate before reporting them together.

---

## 7. Numerics contract

Three rules. Two were latent bugs caught in review before implementation.

1. **No `fastmath=True` in any numba kernel.** Verified in this repo (numba 0.65.1): a
   `fastmath` kernel scanning an array containing 3 real NaNs reports **0** — `np.isnan` is
   compiled away. Any NaN guard inside such a kernel is dead code and any test asserting NaN
   behaviour silently passes. If ever re-enabled, pass an explicit flag set excluding
   `nnan`/`ninf`.
2. **No NaN sentinels in kernels.** Warmup fills with `0.0`; every consumer starts at index
   `N-1`. Correctness then does not depend on rule 1 holding.
3. **float64 accumulators, float32 storage.** Measured over 20,000 random equity curves with a
   naive one-pass sum-of-squares in float32:

   | equity curve | max \|r2_f32 − exact\| | screen flips at `r2 <= 50` |
   |---|---|---|
   | zero-based | 0.0003 | 0 / 20000 |
   | base $100 | **2.88** | — |
   | base $200,000 | **NaN/inf** (`n·Σy² − (Σy)²` goes negative) | — |

   Equity regressions therefore run on **zero-based, mean-centered** trade equity with float64
   accumulators; only the stored result is float32. [M25 Table 1]'s `EQ`/`NetEq` reach
   $233,000 — exactly the exploding regime.

   **Corollary for tolerances (Unit 2).** RMedV is stored float32 and |RMedV| ≈ 0.05, where
   float32 eps is **3.7e-9**. Any acceptance threshold tighter than that is unachievable by
   construction, so the kernel is checked for **bit-exactness** against
   `float32(scipy_float64)` rather than against a tolerance — a stronger test that cannot be
   quietly satisfied by loosening an epsilon.

   **What is *not* load-bearing:** promoting the price window to float64 before differencing
   changes nothing for price data. Sterbenz's lemma makes float32 subtraction exact whenever
   the two values lie within a factor of 2. Over the full sample a 24-bar SPY window spans a
   price ratio of 1.0039 median and **1.1336 worst case** (2020-03-16) — far inside the bound,
   so the float32 and float64 variants are bit-identical on all 5.7M real outputs. The float64
   buffer is kept so the guarantee holds for any input range, but no test can distinguish it
   and none claims to.

   The float64 **median accumulators** (`pairs`/`inner`) *are* load-bearing: a float32 variant
   changes 19.9% of real outputs by up to 1 ulp, and the scipy-oracle test catches it.

---

## 8. Credentials

Alpaca keys come from the environment: `API_KEY`, `SECRET_KEY` (plus `APCA_API_BASE_URL` for
paper vs live). Never a file in the repo, never a literal in source.

`data.load_dotenv()` reads `.env` if present, and `_credentials()` calls it. It is **six lines
of stdlib, not `python-dotenv`** — no new dependency, and the Unit 0 guard still pins
`[project].dependencies` to exactly five packages. Real environment variables take precedence
over the file, and **only `API_KEY`, `SECRET_KEY` and `APCA_API_BASE_URL` are honoured**, so a
stray line in `.env` cannot alter the process environment.

`.env` is gitignored, and the Unit 0 secret scan deliberately skips gitignored files — `.env`
is where credentials are *supposed* to live locally, and the scan's job is to catch key
material in files that could actually be committed. In production, variables are exported by
the shell or set on the Task Scheduler job (Unit 12b); `.env` is a development convenience.

Note that `alpaca-py` will auto-read `APCA_API_KEY_ID` / `APCA_API_SECRET_KEY` if a client is
constructed with no arguments. We use the shorter `API_KEY` / `SECRET_KEY` names and pass them
explicitly at every call site, so the two schemes never silently interact.

---

## 9. Known source discrepancies

Recorded so nobody re-derives them. Each is either resolved or carried as an explicit open
question.

| # | Discrepancy | Status |
|---|---|---|
| **A** | Normalization multiplier: [M25 p.7] and Fig 3 use `6.7`; the Appendix p.28 derives `9.693120`. | **Open.** All published CL results used 6.7, so Meyers' effective range was 0.17–2.4 sd. SPY calibrates by the Appendix method; expect the top of the grid to fire rarely. |
| **B** | [M25 p.7] states 4508 combinations; the stated ranges give 22×14×14 = 4312. `4508 = 23×14×14`. | **Resolved** — use the stated ranges (4312). One of the paper's ranges is off by one N. |
| **C** | Filter term `p<4` / `p<5` [M25 p.11] is never defined. | **Resolved: it is `pf` (Profit Factor).** [M25 Fig 2 col A p.14] shows `pf<2`, `pf<4` and `pf<5` in that grammar slot, always beside a *separate* `lr<3` term — so `p` cannot mean losing-periods-in-a-row, which is `lr`. |
| **D** | `r2` vs `r`: [M05 p.6] defines `R22` as *"the correlation coefficient"* (i.e. `r`); [M25 p.15 Col V] writes *"correlation coefficient(R^2)"*. Used interchangeably. | **Open, but narrowed in Unit 5 to two readings, not three.** Both papers define the metric as the correlation between the equity curve and *its own fitted line* — [M05 p.6] *"between the trade Equity line and the 2nd Order Polynomial Line that is fitted to"* it, [M25 p.15 Col V] *"of a straight-line fit to the equity curve"*. For any least-squares fit carrying an intercept `corr(y, ŷ) = +sqrt(R²) ≥ 0`, so a **signed** `r` has no textual support and is deliberately not stored (§6.6). The two live readings are exact transforms of one column: with `eqR2 = 100·R²`, the `\|r\|` reading is obtained by moving the **threshold**, not the data — `CL2`'s `eqR2 < 80` becomes `eqR2 < 64` and `CL4`'s `eqR2 <= 50` becomes `eqR2 <= 25`. No `sqrt`, no second column, no precision loss. Run both. Doubly harmless for `meyers2005`: its pick is an argmax over `eq2R2`, and non-negativity there is a theorem rather than the assumption §1.5 recorded. ⚑ **Both are implemented and the choice flips the sign of the result** (Unit 8): `CL4` returns `toNP` −94.31 as written and **+84.46** under the `\|r\|` reading, over the same 525 windows. Not a detail. |
| **E** | `mLTr` sign: "we want the row that has the smallest value of `mLTr`" [M25 p.8]. | **Open.** [M25 Fig 2] stores loss metrics **negative** (`LLTr = -3540`, `LLp = -6640`, `eqDD = -10970`), under which "smallest" selects the *deepest* median loss — the opposite of the stated intent (*"minimize the effect of large losing trades"*). Run both conventions. ⚑ **Both are implemented and the choice flips the sign of the result** (Unit 8): `CL2` returns `toNP` +132.11 as stored and **−26.68** under the magnitude reading, over the same 525 windows. It also moves the selected row's median `nT` from 7 to 10 and the silent-week count from 179 to 127. |
| **F** | [M25 p.4] carries a dated erratum retracting its own first-trade rule: *"(11/10/25) Note: this is no longer true…"* | **Deferred.** Keep the 10:00 gate for v1. §3.1 computes RMedV on the full session anyway, so revisiting is a gate change. |
| **G** | Strict vs inclusive: [M25 p.8] writes `lr≤3 r2≤50`; p.10 writes `lr<3\|r2<50` for the same filter. [M05 p.6] writes `PF>1` but its Fig 2 caption writes `PF>=1`. | **Resolved by choice:** `CL4` uses `lr <= 3`, `eqR2 <= 50`; `meyers2005` uses `1 <= PF <= 2`. Recorded so the choice is visible. |
| **H** | First window: [M25 p.4] prose says IS `11/13/14–12/12/14`, OOS `12/16–12/19/14`. [M25 Table 1 p.17] row 1 says IS `11/12/14–12/12/14`, OOS `12/15–12/19/14`. | **Resolved: Table 1 governs** (§4). The prose is wrong on both start dates, and its 4-session OOS week has no holiday to explain it — 12/15–12/19/14 is a full Mon–Fri. |
| **I** | Formula index base: both papers write `i,j = 1..N` over `price(t-i)`, literally excluding bar `t`. | **Resolved by reinterpretation** to `0..N-1` inclusive of bar `t` (§1.1). Supported by the p.2 worked example and the `RMedV[1]` prior-bar term. Worth one bar of lag if wrong. |
| **J** | [M25 p.10] says *"Row 3 is the filter chosen, `b10mLb\|lr≤3r2≤80`"*; pp.8–9 say **Row 4**, and the `$176932 / 57 / 45` figures p.10 quotes are Row 4's (Row 3's `toNP` is 179322). The stray `r2≤80` also collides with `CL2`'s screen. | **Resolved: Row 4**, `b10mLb\|lr≤3 r2≤50-mLTr`, which is the row highlighted in Fig 2. p.8 explains the choice: *"we choose row 4 instead of row 3 because the largest losing week (LLP) was much lower."* |
| **K** | Significance denominator: [M25 p.9] writes `176932/501 = 396.7`, but `176932/501 = 353.2`. The divisor that reproduces 396.7 is Col G = **446** (periods actually traded); 501 is Col D (`aoGP`). Meanwhile the bootstrap null is defined over **517** (all periods). | **Resolved against the paper.** Mixing denominators is what makes the published result significant — same data, consistent denominator, `K·p` goes 0.049 → 2.23. **We use all OOS periods on both sides**, zero-trade weeks counted as zero (§6.5). |
| **L** | [M25]'s two Appendix tables claim to be the same run under an identical header, but p.28 is a uniform **+1.38%** above `sqrt(N)` x p.27 at every one of the 18 N (range 1.327-1.613%). | **Recorded, not resolvable.** A scale offset, not a formula difference — the two pages saw slightly different value sets. It matters because re-deriving `xmult` from Table A gives **9.8266**, not the published 9.6931. Table B governs; `test_unit3_paper_tables_disagree` pins both numbers. |
