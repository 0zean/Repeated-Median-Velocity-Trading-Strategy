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
bars give different multipliers."* SPY's value is measured in Unit 3; the `4.00512` in the
pre-revamp `rmv.py` is unsourced.

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
*"the trade Equity line"*. `eqR2` is the **straight-line** fit and `eq2R2` the 2nd-order fit;
confirmed by [M25 Figure 2 Row 4] showing `eqR2 = 82` against the same curve's 2nd-order
`R² = 0.9496` [M25 p.13] — a straight line fitting worse than a quadratic, on a 0–100 scale.

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
formula), none of which appear as headers, and it omits `ownp`/`ownt`/`N`/`vup`/`vdn`
entirely. Mapping, header ← legend:

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
| **D** | `r2` vs `r`: [M05 p.6] defines `R22` as *"the correlation coefficient"* (i.e. `r`); [M25 p.15 Col V] writes *"correlation coefficient(R^2)"*. Used interchangeably. | **Open.** Harmless for `meyers2005` (rank metric — argmax preserved for non-negative `r`). **Decisive for `CL4`**: `r2 <= 50` keeps `\|r\| <= 0.707` under one reading, `r <= 0.50` under the other. Run both. |
| **E** | `mLTr` sign: "we want the row that has the smallest value of `mLTr`" [M25 p.8]. | **Open.** [M25 Fig 2] stores loss metrics **negative** (`LLTr = -3540`, `LLp = -6640`, `eqDD = -10970`), under which "smallest" selects the *deepest* median loss — the opposite of the stated intent (*"minimize the effect of large losing trades"*). Run both conventions. |
| **F** | [M25 p.4] carries a dated erratum retracting its own first-trade rule: *"(11/10/25) Note: this is no longer true…"* | **Deferred.** Keep the 10:00 gate for v1. §3.1 computes RMedV on the full session anyway, so revisiting is a gate change. |
| **G** | Strict vs inclusive: [M25 p.8] writes `lr≤3 r2≤50`; p.10 writes `lr<3\|r2<50` for the same filter. [M05 p.6] writes `PF>1` but its Fig 2 caption writes `PF>=1`. | **Resolved by choice:** `CL4` uses `lr <= 3`, `eqR2 <= 50`; `meyers2005` uses `1 <= PF <= 2`. Recorded so the choice is visible. |
| **H** | First window: [M25 p.4] prose says IS `11/13/14–12/12/14`, OOS `12/16–12/19/14`. [M25 Table 1 p.17] row 1 says IS `11/12/14–12/12/14`, OOS `12/15–12/19/14`. | **Resolved: Table 1 governs** (§4). The prose is wrong on both start dates, and its 4-session OOS week has no holiday to explain it — 12/15–12/19/14 is a full Mon–Fri. |
| **I** | Formula index base: both papers write `i,j = 1..N` over `price(t-i)`, literally excluding bar `t`. | **Resolved by reinterpretation** to `0..N-1` inclusive of bar `t` (§1.1). Supported by the p.2 worked example and the `RMedV[1]` prior-bar term. Worth one bar of lag if wrong. |
| **J** | [M25 p.10] says *"Row 3 is the filter chosen, `b10mLb\|lr≤3r2≤80`"*; pp.8–9 say **Row 4**, and the `$176932 / 57 / 45` figures p.10 quotes are Row 4's (Row 3's `toNP` is 179322). The stray `r2≤80` also collides with `CL2`'s screen. | **Resolved: Row 4**, `b10mLb\|lr≤3 r2≤50-mLTr`, which is the row highlighted in Fig 2. p.8 explains the choice: *"we choose row 4 instead of row 3 because the largest losing week (LLP) was much lower."* |
| **K** | Significance denominator: [M25 p.9] writes `176932/501 = 396.7`, but `176932/501 = 353.2`. The divisor that reproduces 396.7 is Col G = **446** (periods actually traded); 501 is Col D (`aoGP`). Meanwhile the bootstrap null is defined over **517** (all periods). | **Resolved against the paper.** Mixing denominators is what makes the published result significant — same data, consistent denominator, `K·p` goes 0.049 → 2.23. **We use all OOS periods on both sides**, zero-trade weeks counted as zero (§6.5). |
