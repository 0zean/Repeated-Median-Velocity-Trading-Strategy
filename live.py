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

Units 12a/12b add the replay driver and the intraday loop here (PLAN §2).
"""

from __future__ import annotations

import json
import math
import subprocess
import sys
from datetime import datetime, timezone
from pathlib import Path

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


def main(argv: list[str], today: np.datetime64 | None = None, client=None,
         path: Path | str = PARAMS) -> int:
    if not argv or argv[0] != "refit" or len(argv) > 2:
        print("usage: python live.py refit [YYYY-MM-DD]", file=sys.stderr)
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
