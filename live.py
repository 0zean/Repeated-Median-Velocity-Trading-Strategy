"""Weekly refit for the region portfolio. PLAN §3 Unit 11.

⚑ Unit 10 cancelled parameter selection, so this is not the refit PLAN Rev 2 wrote. The
strategy is every combo in `pwfo.REGION`, equal weight, on SPY and QQQ at 50/50 notional,
with no IS filter. There is no row left to choose each week -- what is refit is the
*scale*: each leg's `xmult` and `cost`, off that leg's own 31-day IS span ending Friday
(SPEC §1.2.1, §3.2). `params.json` carries exactly those. No IS grid is run: with no
filter, nothing consumes its 4312 rows.

    python live.py refit              refit for the last closed Friday, write params.json
    python live.py refit 2025-06-13   dry run: refit that Friday off the network and
                                      compare to Unit 7's stored window; never writes
    python live.py parity [SYM ...]   replay every pre-tail OOS week bar by bar and check
                                      it against Unit 7 trade for trade; offline

Unit 12b adds the intraday loop here (PLAN §2).
"""

from __future__ import annotations

import json
import math
import subprocess
import sys
from datetime import date, datetime, timezone
from pathlib import Path
from zoneinfo import ZoneInfo

import numpy as np
import pandas as pd

import data
import pwfo
import rmv

ROOT = Path(__file__).parent
PARAMS = ROOT / "params.json"

# PLAN §3 Unit 10's pre-registration, frozen beside `pwfo.REGION`: these two legs, equal
# weight. A partial refit is a different strategy, not a smaller one, so it is refused.
LEGS = ("SPY", "QQQ")
DAY = np.timedelta64(1, "D")


def today_et() -> np.datetime64:
    return np.datetime64(pd.Timestamp.now(tz=data.ET).date(), "D")


def last_friday(today: np.datetime64) -> np.datetime64:
    """The latest Friday strictly before `today` (ET). On a Friday, the previous one.

    Strictly before, because a Friday's IS span is not closed until Friday is over: a refit
    run Friday afternoon would fit on a partial last session and trade Monday on it.
    """
    return today - int((pwfo._dow(today) - 4) % 7 or 7) * DAY


def is_span(friday: np.datetime64) -> tuple[datetime, datetime]:
    """`[friday - 30 days, friday]` as UTC instants bounding the ET dates, as `pwfo.windows`.

    ⚑ Midnight **ET**. A UTC midnight lands at 19:00/20:00 ET the evening before, which
    cuts the whole of Friday's session off the end and adds the evening before `start` --
    session-masked away, so the only symptom would be an `xmult` a day short. The end bound
    is Saturday 00:00 ET; the session mask trims it to Friday's 15:55 bar.
    """
    lo = pd.Timestamp(str(friday - pwfo.IS_DAYS), tz=data.ET)
    hi = pd.Timestamp(str(friday + DAY), tz=data.ET)
    return lo.tz_convert("UTC").to_pydatetime(), hi.tz_convert("UTC").to_pydatetime()


def refit(symbol: str, friday: np.datetime64, cache: bool = False, client=None) -> dict:
    """One leg: `{xmult, cost, is_bars}` off its own IS span ending `friday`.

    `cache=False` is the live path: exactly the span, off the network, bar cache untouched.
    `cache=True` slices the same span out of the cache with no network at all.

    Computed on the span **standalone**. `pwfo.run` takes RMedV and the gate off the full
    sample and slices; the two agree only because no gated bar's `t` or `t-1` window
    reaches back across a session gap (SPEC §3.1's zero-margin blackout), so a 31-day
    fetch reproduces every gated value. That is exactly what Unit 11's check tests.
    """
    if pwfo._dow(friday) != 4:
        raise ValueError(f"{friday} is not a Friday; every IS span ends on one (SPEC §4)")
    lo, hi = is_span(friday)
    bars = data.load_bars(symbol, lo, hi, cache=cache, refresh=not cache, client=client)
    if not len(bars):
        raise ValueError(f"{symbol}: no bars in the IS span ending {friday}")
    day = data.to_et(bars.ts).normalize().tz_localize(None).values.astype("datetime64[D]")
    # The look-ahead guard, on what came back rather than on what was asked for: the refit
    # must not see the week it fits for, whatever the request bounds turned out to mean.
    if day[0] < friday - pwfo.IS_DAYS or day[-1] > friday:
        raise ValueError(f"{symbol}: bars {day[0]}..{day[-1]} are outside the IS span "
                         f"ending {friday}")
    # A whole missing session is the hole an online fetch can have and the cache cannot:
    # `load_bars` re-reads the cache's last 200 bars on every refresh and compares, a
    # one-shot fetch is compared against nothing. Intraday holes are the gate's job and
    # behave identically on both paths. Read over the **span**, not over what came back:
    # `load_bars` refreshed the calendar only to its last bar, so on a live date a fetch
    # missing Friday would be missing Friday from the calendar too (Unit 11 review, #4).
    cal = data.load_calendar(lo.date(), friday.item(), refresh=not cache)
    want = {d for d in cal if str(friday - pwfo.IS_DAYS) <= d <= str(friday)}
    missing = sorted(want - set(np.unique(day).astype(str)))
    if missing:
        raise ValueError(f"{symbol}: no bars on {len(missing)} of {len(want)} sessions in "
                         f"the IS span ending {friday}: {missing[:5]}")
    return {"xmult": rmv.xmult(rmv.rmv_all_n(bars.close), bars.gate == 1),
            "cost": pwfo.window_cost(bars.close), "is_bars": len(bars)}


def _git() -> str:
    """Which code wrote the file. `-dirty` when the tree was not a commit."""
    try:
        out = subprocess.run(["git", "describe", "--always", "--dirty"], cwd=ROOT,
                             capture_output=True, text=True, timeout=10)
    except (OSError, subprocess.TimeoutExpired):
        return "unknown"
    return out.stdout.strip() or "unknown"


def write_params(friday: np.datetime64, legs: dict[str, dict], path: Path | str = PARAMS):
    p = {"as_of": str(friday), "region": list(pwfo.REGION), "legs": legs, "git": _git(),
         "written": datetime.now(timezone.utc).isoformat(timespec="seconds")}
    load_params(p, friday + pwfo.OOS_LO)  # never persist a file the reader would refuse
    # `repr` round-trips a float64 exactly, so the thresholds live computes off this file
    # are bit-identical to the ones the backtest computed off `pwfo_index.json`.
    data._atomic_write(Path(path), json.dumps(p, indent=1))
    return p


def load_params(path: Path | str | dict = PARAMS, today: np.datetime64 | None = None) -> dict:
    """The only reader of `params.json`. Refuses on any doubt, and a refusal is a flat week.

    ⚑ Valid for **its own OOS week only**, `as_of + 3 .. as_of + 7` in ET dates -- stricter
    than PLAN Rev 2's "older than 10 days refuses", which would trade the Monday after a
    failed weekend refit on last week's `xmult`. `pwfo.run` scores every OOS week under its
    own window's multiplier and that multiplier moves up to 20x (SPEC §1.2.1), so a stale
    file is a parity break, not a staleness nuisance. Flat is the backtest's own zero.

    "Params fall inside the grid" is `region == pwfo.REGION` -- whose bounds are grid values
    -- plus a finite positive `xmult` and `cost` per leg: `rmv.threshold` inverts on a
    negative multiplier and turns every comparison False on a nan. `is_bars` trades nothing
    but is the file's record of what the fit saw, so it is held to "a positive int" too.
    """
    try:
        p = path if isinstance(path, dict) else json.loads(Path(path).read_text(encoding="utf-8"))
        as_of = np.datetime64(p["as_of"], "D")
        today = today_et() if today is None else today
        if pwfo._dow(as_of) != 4:
            raise ValueError(f"as_of {as_of} is not a Friday")
        if not as_of + pwfo.OOS_LO <= today <= as_of + pwfo.OOS_HI:
            raise ValueError(f"it is for the week {as_of + pwfo.OOS_LO}..{as_of + pwfo.OOS_HI} "
                             f"and today is {today}")
        if p["region"] != list(pwfo.REGION):
            raise ValueError(f"region {p['region']} is not the frozen {list(pwfo.REGION)}")
        if sorted(p["legs"]) != sorted(LEGS):
            raise ValueError(f"legs {sorted(p['legs'])} are not the pre-registered {list(LEGS)}")
        for sym, leg in p["legs"].items():
            for k, kind in (("xmult", float), ("cost", float), ("is_bars", int)):
                v = leg[k]
                if isinstance(v, bool) or not (isinstance(v, kind) and math.isfinite(v)
                                               and v > 0):
                    raise ValueError(f"{sym} {k} = {v!r}")
    except (OSError, KeyError, TypeError, ValueError) as e:
        raise ValueError(f"params.json refused, trade nothing this week: {e}") from None
    return p


def stored(symbol: str, friday: np.datetime64) -> dict | None:
    """Unit 7's **pre-tail** window for `friday` off the leg's own index, or None.

    Pre-tail only: the tail windows' `xmult` is a vol-state reading of the spent holdout,
    and PLAN §3 Unit 10 keeps its per-week detail unprinted for the deferred vol filter.
    """
    f = pwfo.sym_dir(symbol) / "pwfo_index.json"
    if not f.exists():
        return None
    return next((w for w in json.loads(f.read_text(encoding="utf-8"))["windows"]
                 if w["friday"] == str(friday) and w["file"] == "is"), None)


# ------------------------------------------------------------ Unit 12a: replay and parity

ET = ZoneInfo(data.ET)
# `rmv_all_n`'s live contract: max(n) + 1 bars, so `[:, -2]` is a real RMedV[t-1]. A ring of
# exactly max(n) reads warmup there and the crossing rule silently becomes a level rule.
RING = int(rmv.N_VALUES.max()) + 1


class Book:
    """One leg's region book, fed one closed bar at a time. PLAN §3 Unit 12a.

    1620 crossing-rule states in {-1, 0, +1}, one per combo in `pwfo.REGION`, and the target
    is their sum in **combo units**: one unit is 1/1620 of a share, the backtest's equal
    weight, so the integer target is the fractional book exactly. Rounding it to shares is
    Unit 12b's sizing decision, not a rounding made here.

    Sees only what live sees -- bars as they arrive, the calendar, one `params.json` leg.
    ⚑ Never `bars.gate`. The gate is rebuilt every bar from the clock, the session's close,
    and `data.build_gate`'s blackout read causally: a gap after bar i blacks out i+1..i+MAX_N,
    so bar t is clear iff bars t-MAX_N..t hold no gap, i.e. span exactly MAX_N bars of time.
    """

    def __init__(self, leg: dict, calendar: dict[str, int], region: tuple = pwfo.REGION):
        combos = [pwfo.decode(c) for c in np.flatnonzero(pwfo.region_mask(*region))]
        self.row = np.searchsorted(rmv.N_VALUES, [n for n, _, _ in combos])
        self.up = np.array([rmv.threshold(u, leg["xmult"], n) for n, u, _ in combos])
        self.dn = np.array([rmv.threshold(d, leg["xmult"], n) for n, _, d in combos])
        self.pos = np.zeros(len(combos), np.int8)
        self.gated = False
        self.calendar = calendar
        # Zeros, so the span test fails until RING real bars have arrived: no warmup counter.
        self.ts = np.zeros(RING, np.int64)
        self.close = np.zeros(RING, np.float32)
        self.rmv = np.empty((rmv.N_VALUES.size, RING), np.float32)

    def on_bar(self, ts: int, close: float) -> int:
        """One closed bar, open-labelled (SPEC §3.2). Returns the target in combo units."""
        self.ts[:-1] = self.ts[1:]
        self.ts[-1] = ts
        self.close[:-1] = self.close[1:]
        self.close[-1] = close
        et = datetime.fromtimestamp(ts // 10**9, ET)
        minute = et.hour * 60 + et.minute
        cut = min(self.calendar.get(str(et.date()), 0) - data.EXIT_BEFORE_CLOSE_MIN,
                  data.GATE_CLOSE_MIN)
        self.gated = bool(data.GATE_OPEN_MIN <= minute < cut
                          and self.ts[-1] - self.ts[-1 - data.MAX_N] == data.MAX_N * data.BAR_NS)
        if not self.gated or minute + 5 >= cut:
            # SPEC §2.1 A, in the order Unit 12b inherits from Unit 4: a bar whose successor
            # the clock shuts goes flat *before* any signal is read, so it never enters. The
            # clock (`+ 5`, one bar) is all of the kernel's `gate[t + 1]` live can know; a
            # successor that arrives late is the rest, and `parity` names those sessions.
            self.pos[:] = 0
        else:
            # ponytail: ~10 numpy temporaries a bar against PLAN §2.4's "zero steady-state
            # allocation", a budget written for one combo. Nothing at one bar per 5 minutes;
            # an njit step over preallocated buffers is the upgrade if a profile ever asks.
            rmv.rmv_all_n(self.close, out=self.rmv)
            cur, prev = self.rmv[self.row, -1], self.rmv[self.row, -2]
            # SPEC §2: both bounds inclusive, `vdn` positive against a negative velocity.
            buy = (cur >= self.up) & (prev < self.up)
            sell = (cur <= -self.dn) & (prev > -self.dn)
            sig = buy.view(np.int8) - sell.view(np.int8)
            np.copyto(self.pos, sig, where=sig != 0)  # a signal sets the side; none holds
        return int(self.pos.sum())


class FakeBroker:
    """Alpaca's stand-in for replay: any order fills in full at the price it is handed."""

    def __init__(self) -> None:
        self.position = 0
        self.fills: list[tuple[int, float]] = []

    def submit(self, qty: int, price: float) -> None:
        self.position += qty
        self.fills.append((qty, price))


def replay(book: Book, ts: np.ndarray, close: np.ndarray):
    """Bars through `book` one at a time, reconciled against a `FakeBroker` after each: read
    the position, submit the difference -- Unit 12b's loop with the broker faked, filling at
    the signal bar's close as SPEC §2 books it. Returns every bar's states and gate, and the
    broker."""
    broker = FakeBroker()
    states = np.empty((ts.size, book.pos.size), np.int8)
    gated = np.empty(ts.size, np.int8)
    for t in range(ts.size):
        want = book.on_bar(int(ts[t]), float(close[t]))
        if want != broker.position:
            broker.submit(want - broker.position, float(close[t]))
        states[t], gated[t] = book.pos, book.gated
    return states, gated, broker


def trades(states: np.ndarray, close: np.ndarray, cost: float) -> tuple[np.ndarray, np.ndarray]:
    """Per-bar combo states -> `(combo, trades)`, trades as `rmv.TRADE_COLS`, combo-major.

    A trade opens where a combo's state changes *to* non-zero and closes where it changes
    *from* non-zero, both at that bar's close; a reversal bar does both. Padding every combo
    with a flat bar at each end makes opens and closes alternate within a combo and never
    pair across two, so the i-th open belongs to the i-th close with no loop. `net` is
    spelled exactly as `rmv._simulate` spells it, so parity is `array_equal`, not closeness.
    """
    if states[-1].any():
        # The pad would book its exit at bar T, one past `close`. `Book` is flat on every ungated
        # bar and every week ends on one, so only a slice cut mid-session lands here -- which is
        # what Unit 12b's restart replay would hand it (Unit 12a review).
        raise ValueError(f"{int(np.count_nonzero(states[-1]))} combos still open on the last bar")
    width = states.shape[0] + 2
    f = np.pad(states.T, ((0, 0), (1, 1))).ravel()
    moved = f[:-1] != f[1:]
    o = np.flatnonzero(moved & (f[1:] != 0)) + 1
    x = np.flatnonzero(moved & (f[:-1] != 0)) + 1
    entry, exit_ = o % width - 1, x % width - 1
    side = f[o].astype(np.float64)
    c = close.astype(np.float64)
    return o // width, np.column_stack([entry, exit_, side, side * (c[exit_] - c[entry]) - cost])


def parity(symbol: str, fridays: list[str] | None = None) -> dict:
    """PLAN §3 Unit 12a's done-when for one leg: every pre-tail OOS week, or only `fridays`,
    replayed bar by bar. Raises `AssertionError` at the first break; returns the counts.

    1. Live's gate equals `bars.gate` on every bar.
    2. The reference is Unit 7's: `rmv.simulate` on Unit 7's own inputs -- the full-sample
       matrix sliced, `bars.gate`, the index's `xmult` and `cost` -- scores every region combo
       to its stored `pwfo_oos.npy` row, float32 for float32. The thresholds are the book's,
       so this is also what anchors them.
    3. Trade for trade, per combo, on every session except those live cannot know (below).
    4. The broker holds the book: gross off its fills is the trades' gross exactly -- `fsum`
       of exact terms, integer units times a float32 price -- flat at every week's end, and
       netted turnover is at most the book's own per-combo turnover, `2 * len(live)` (the
       triangle inequality). The *reported* ratio is against the backtest's, `2 * len(ref)`;
       the two differ only on the unknowable sessions (Unit 12a review).
    """
    bars = data.load_bars(symbol, datetime(2016, 1, 1, tzinfo=timezone.utc),
                          datetime(2030, 1, 1, tzinfo=timezone.utc), refresh=False)
    day = data.to_et(bars.ts).strftime("%Y-%m-%d").values
    cal = data.load_calendar(date.fromisoformat(day[0]), date.fromisoformat(day[-1]),
                             refresh=False)
    # ⚑ Unknowable sessions come from the bars, never from where trades differ: a gated bar
    # whose successor the clock opens and `bars.gate` does not -- the next bar arrived late
    # (SPEC §2.1 B, C). `build_gate` one bar later gives each successor's clock; its blackout
    # term is shift-invariant and already clear on any gated bar.
    nxt = data.build_gate(bars.ts + data.BAR_NS, close_min=[cal.get(d, 0) for d in day])
    late = (bars.gate[:-1] == 1) & (bars.gate[1:] == 0) & (nxt[:-1] == 1)
    unknowable = set(day[:-1][late])

    matrix = rmv.rmv_all_n(bars.close)
    by_friday = {str(w.friday): w for w in pwfo.windows(bars)}
    _, oos, wins = pwfo.load_tables([], pwfo.sym_dir(symbol))
    combos = np.flatnonzero(pwfo.region_mask())
    out = {"weeks": 0, "trades": 0, "excluded": 0, "sessions": [], "differ": [],
           "turnover": 0, "charged": 0}
    for w in wins:
        if fridays is not None and w["friday"] not in fridays:
            continue
        where = f"{symbol} {w['friday']}"
        win = by_friday[w["friday"]]
        sl = slice(win.oos_lo, win.oos_hi)
        close, gate, wday = bars.close[sl], bars.gate[sl], day[sl]
        book = Book(w, cal)
        states, gated, broker = replay(book, bars.ts[sl], close)
        if not np.array_equal(gated, gate):
            raise AssertionError(f"{where}: live's gate differs from bars.gate on "
                                 f"{int(np.count_nonzero(gated != gate))} bars")

        rows = np.ascontiguousarray(matrix[:, sl])
        buf = np.empty((close.size, 4))
        ref, ref_combo = [], []
        for i, c in enumerate(combos):
            tr = rmv.simulate(rows[book.row[i]], close, gate, book.up[i], book.dn[i], w["cost"],
                              out=buf)
            if not np.array_equal(rmv.metrics(tr)[rmv.OOS_COLS].astype(np.float32),
                                  oos[w["row"], c]):
                raise AssertionError(f"{where}: {pwfo.decode(c)} does not score to Unit 7's row")
            ref.append(tr.copy())
            ref_combo.append(np.full(len(tr), i))
        ref, ref_combo = np.concatenate(ref), np.concatenate(ref_combo)
        combo, live = trades(states, close, w["cost"])

        lday, rday = wday[live[:, 0].astype(np.int64)], wday[ref[:, 0].astype(np.int64)]
        sessions = sorted(unknowable.intersection(wday))

        def same(kl, kr):
            return (np.array_equal(live[kl], ref[kr])
                    and np.array_equal(combo[kl], ref_combo[kr]))

        skip_l, skip_r = np.isin(lday, sessions), np.isin(rday, sessions)
        if not same(~skip_l, ~skip_r):
            raise AssertionError(f"{where}: live and Unit 7 trades differ on a session live "
                                 f"could know")
        c64 = close.astype(np.float64)
        gross = math.fsum(live[:, 2] * (c64[live[:, 1].astype(np.int64)]
                                        - c64[live[:, 0].astype(np.int64)]))
        turnover = sum(abs(q) for q, _ in broker.fills)
        if (broker.position != 0 or math.fsum(-q * p for q, p in broker.fills) != gross
                or turnover > 2 * len(live)):
            raise AssertionError(f"{where}: the broker does not hold the book (position "
                                 f"{broker.position}, turnover {turnover}, {len(live)} trades)")

        out["weeks"] += 1
        out["trades"] += int(np.count_nonzero(~skip_r))
        out["excluded"] += int(np.count_nonzero(skip_r))
        out["sessions"] += sessions
        out["differ"] += [s for s in sessions if not same(lday == s, rday == s)]
        out["turnover"] += turnover
        out["charged"] += 2 * len(ref)
    return out


def main(argv: list[str], today: np.datetime64 | None = None, client=None,
         path: Path | str = PARAMS) -> int:
    if argv[:1] == ["parity"]:
        for s in argv[1:] or LEGS:
            r = parity(s)
            print(f"  {s}  {r['weeks']} weeks: {r['trades']} trades identical over "
                  f"{int(pwfo.region_mask().sum())} combos, every stored row reproduced; "
                  f"{r['excluded']} on unknowable {r['sessions']} excluded; netted turnover "
                  f"{r['turnover'] / r['charged']:.1%} of what the backtest charges")
        return 0
    if not argv or argv[0] != "refit" or len(argv) > 2:
        print("usage: python live.py refit [YYYY-MM-DD] | parity [SYM ...]", file=sys.stderr)
        return 2
    today = today_et() if today is None else today
    dry = len(argv) == 2
    friday = np.datetime64(argv[1], "D") if dry else last_friday(today)
    if friday >= today:
        print(f"{friday} has not closed yet (today {today})", file=sys.stderr)
        return 2
    # On a Friday the last *closed* Friday is a week old and its file expires tonight, so a
    # refit scheduled Friday evening writes a week that is over and every Monday goes flat.
    # Refused here, where a scheduler sees exit 2 on its first run, not on the Monday after.
    if not dry and pwfo._dow(today) == 4:
        print(f"today is Friday {today}: its span closes at midnight ET; run the refit on "
              f"Saturday or later for {today + pwfo.OOS_LO}..{today + pwfo.OOS_HI}",
              file=sys.stderr)
        return 2
    # Before any fetch: a dry run exists to be compared, and a tail Friday would print the
    # holdout's per-week `xmult` and then find nothing to compare it with. A non-Friday has
    # no window either, so this is also where a mistyped date stops.
    want = {s: stored(s, friday) for s in LEGS} if dry else {}
    if dry and None in want.values():
        print(f"{friday}: not a pre-tail window Friday in "
              f"{[s for s, w in want.items() if w is None]}'s index", file=sys.stderr)
        return 2
    try:
        legs = {s: refit(s, friday, client=client) for s in LEGS}
    except Exception as e:  # anything at all: no file is the only safe outcome
        print(f"⚑ REFIT FAILED for {friday}: {type(e).__name__}: {e}\n"
              f"  params.json not written; load_params refuses any other week, so "
              f"{friday + pwfo.OOS_LO}..{friday + pwfo.OOS_HI} is FLAT unless a refit "
              f"succeeds first", file=sys.stderr)
        return 1
    for s, leg in legs.items():
        print(f"  {s}  xmult {leg['xmult']!r}  cost {leg['cost']!r}  {leg['is_bars']} IS bars")
    if not dry:
        write_params(friday, legs, path)
        print(f"{Path(path).name} written for {friday + pwfo.OOS_LO}..{friday + pwfo.OOS_HI}")
        return 0
    # Unit 11's done-when: exact equality with what Unit 7 chose, not closeness. Both sides
    # are float64 from the same functions on the same bars, so any difference is a defect.
    bad = 0
    for s, leg in legs.items():
        off = [k for k in ("xmult", "cost", "is_bars") if leg[k] != want[s][k]]
        print(f"  {s}  {'MATCH' if not off else 'MISMATCH ' + str(off)} vs "
              f"{pwfo.sym_dir(s).name}/pwfo_index.json")
        bad += bool(off)
    return 1 if bad else 0


if __name__ == "__main__":
    sys.exit(main(sys.argv[1:]))
