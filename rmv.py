from __future__ import annotations

from itertools import product

import numpy as np
import pandas as pd
from alpaca.data.historical import StockHistoricalDataClient
from alpaca.data.requests import StockBarsRequest
from alpaca.data.timeframe import TimeFrame, TimeFrameUnit
from datetime import datetime
from scipy.stats import siegelslopes

N_VALUES = range(10, 80, 5)
V_VALUES = tuple(round(i / 100, 2) for i in range(2, 41, 2))


def get_data(ticker: str, client: StockHistoricalDataClient) -> pd.Series:
    """Get historical data from stock ticker."""
    request_params = StockBarsRequest(
        symbol_or_symbols=[ticker],
        timeframe=TimeFrame(5,TimeFrameUnit.Minute),
        start=datetime(2024, 7, 11),
        end=datetime(2026, 6, 18)
    )
    bars = client.get_stock_bars(request_params)
    return bars.df["close"].droplevel("symbol")


def repeated_median_slope(values: np.ndarray) -> float:
    """Repeated median slope for one rolling price window."""
    y = np.asarray(values, dtype=float)
    x = np.arange(len(y), dtype=float)
    slope, _ = siegelslopes(y, x)
    return float(slope)


def rmv(
    close: pd.Series, n: int, *, normalize: bool = False, xmult: float = 4.00512
) -> pd.Series:
    """Repeated Median Velocity. Raw slope matches the paper; normalize is notebook-only."""
    out = close.astype(float).rolling(n).apply(repeated_median_slope, raw=True)
    if normalize:
        out = out * (np.sqrt(n) * xmult)
    return out.rename("rmv")


def _et_index(index: pd.Index, tz: str = "America/New_York") -> pd.DatetimeIndex:
    dt_index = pd.DatetimeIndex(index)
    if dt_index.tz is None:
        return dt_index.tz_localize(tz)
    return dt_index.tz_convert(tz)


def positions_from_rmv(
    rmv_values: pd.Series,
    vup: float,
    vdn: float,
    *,
    first_trade: str = "10:00",
    exit_time: str = "15:55",
    execution_lag: int = 1,
    tz: str = "America/New_York",
) -> pd.Series:
    """Paper rules: long above vup, short below -vdn, flat before first trade and at EOD."""
    et = _et_index(rmv_values.index, tz)
    first = pd.Timestamp(first_trade).time()
    exit_at = pd.Timestamp(exit_time).time()
    desired = np.zeros(len(rmv_values), dtype=float)
    pos = 0.0

    for i, (stamp, value) in enumerate(zip(et, rmv_values.to_numpy())):
        tod = stamp.time()
        if tod < first or tod >= exit_at:
            pos = 0.0
        elif value > vup:
            pos = 1.0
        elif value < -vdn:
            pos = -1.0
        desired[i] = pos

    position = pd.Series(desired, index=rmv_values.index, name="position")
    if execution_lag:
        position = position.shift(execution_lag).fillna(0.0)

    flat = [(stamp.time() < first) or (stamp.time() >= exit_at) for stamp in et]
    position.loc[flat] = 0.0
    return position


def run_strategy(
    close: pd.Series,
    n: int,
    vup: float,
    vdn: float,
    *,
    normalize: bool = False,
    first_trade: str = "10:00",
    exit_time: str = "15:55",
    execution_lag: int = 1,
    tz: str = "America/New_York",
) -> pd.DataFrame:
    rmv_values = rmv(close, n, normalize=normalize)
    position = positions_from_rmv(
        rmv_values,
        vup,
        vdn,
        first_trade=first_trade,
        exit_time=exit_time,
        execution_lag=execution_lag,
        tz=tz,
    )
    returns = close.astype(float).pct_change().fillna(0.0)
    return pd.DataFrame(
        {
            "close": close.astype(float),
            "rmv": rmv_values,
            "position": position,
            "returns": returns,
            "strategy_returns": position.shift().fillna(0.0) * returns,
        }
    )


def trades_from_position(
    close: pd.Series,
    position: pd.Series,
    *,
    point_value: float = 1.0,
    size: float = 1.0,
) -> pd.DataFrame:
    rows = []
    side = 0.0
    entry_time = None
    entry_price = np.nan

    for time, price, new_side in zip(
        close.index, close.astype(float), position.astype(float)
    ):
        if new_side == side:
            continue
        if side:
            pnl = (price - entry_price) * side * point_value * size
            rows.append(
                {
                    "entry_time": entry_time,
                    "exit_time": time,
                    "side": "long" if side > 0 else "short",
                    "entry_price": entry_price,
                    "exit_price": price,
                    "pnl": pnl,
                }
            )
        if new_side:
            entry_time = time
            entry_price = price
        side = new_side

    if side:
        price = float(close.iloc[-1])
        pnl = (price - entry_price) * side * point_value * size
        rows.append(
            {
                "entry_time": entry_time,
                "exit_time": close.index[-1],
                "side": "long" if side > 0 else "short",
                "entry_price": entry_price,
                "exit_price": price,
                "pnl": pnl,
            }
        )

    return pd.DataFrame(rows)


def trade_metrics(trades: pd.DataFrame) -> dict[str, float]:
    if trades.empty:
        return {
            "net_profit": 0.0,
            "profit_factor": np.nan,
            "max_losses": 0.0,
            "trades": 0.0,
            "r22": np.nan,
        }

    pnl = trades["pnl"].astype(float)
    gross_profit = pnl[pnl > 0].sum()
    gross_loss = -pnl[pnl < 0].sum()
    profit_factor = np.inf if gross_loss == 0 else gross_profit / gross_loss

    max_losses = run = 0
    for value in pnl:
        run = run + 1 if value < 0 else 0
        max_losses = max(max_losses, run)

    equity = pnl.cumsum().to_numpy()
    if len(equity) >= 3 and np.std(equity) > 0:
        x = np.arange(len(equity), dtype=float)
        fit = np.polyval(np.polyfit(x, equity, 2), x)
        corr = np.corrcoef(equity, fit)[0, 1]
        r22 = float(corr * corr)
    else:
        r22 = np.nan

    return {
        "net_profit": float(pnl.sum()),
        "profit_factor": float(profit_factor),
        "max_losses": float(max_losses),
        "trades": float(len(trades)),
        "r22": r22,
    }


def score_params(
    close: pd.Series,
    rmv_values: pd.Series,
    n: int,
    vup: float,
    vdn: float,
    *,
    first_trade: str = "10:00",
    exit_time: str = "15:55",
    execution_lag: int = 1,
    tz: str = "America/New_York",
    point_value: float = 1.0,
    size: float = 1.0,
) -> dict[str, float]:
    position = positions_from_rmv(
        rmv_values,
        vup,
        vdn,
        first_trade=first_trade,
        exit_time=exit_time,
        execution_lag=execution_lag,
        tz=tz,
    )
    trades = trades_from_position(close, position, point_value=point_value, size=size)
    return {"n": n, "vup": vup, "vdn": vdn, **trade_metrics(trades)}


def optimize_window(
    close: pd.Series,
    rmv_by_n: dict[int, pd.Series],
    *,
    n_values=N_VALUES,
    v_values=V_VALUES,
    min_trades: int = 16,
    pf_min: float = 1.0,
    pf_max: float = 2.0,
    max_losses: int = 3,
    **kwargs,
) -> tuple[pd.Series | None, pd.DataFrame]:
    # ponytail: brute-force grid; replace with vectorbt/numba only if runtime becomes the bottleneck.
    rows = [
        score_params(close, rmv_by_n[n].reindex(close.index), n, vup, vdn, **kwargs)
        for n in n_values
        for vup, vdn in product(v_values, repeat=2)
    ]
    scores = pd.DataFrame(rows)
    eligible = scores[
        (scores["profit_factor"] >= pf_min)
        & (scores["profit_factor"] <= pf_max)
        & (scores["max_losses"] <= max_losses)
        & (scores["trades"] >= min_trades)
    ]
    if eligible.empty:
        return None, scores
    best = eligible.sort_values(["r22", "net_profit"], ascending=False).iloc[0]
    return best, scores


def walk_forward_windows(
    close: pd.Series,
    *,
    n_windows: int = 16,
    train_days: int = 30,
    oos_days: int = 7,
    step_days: int = 7,
    tz: str = "America/New_York",
) -> list[dict[str, pd.Timestamp]]:
    et = _et_index(close.index, tz)
    first_day = et.min().normalize()
    last_day = et.max().normalize()
    windows = []

    for i in range(n_windows):
        train_start = first_day + pd.Timedelta(days=i * step_days)
        train_end = train_start + pd.Timedelta(days=train_days)
        oos_start = train_end + pd.Timedelta(days=1)
        oos_end = oos_start + pd.Timedelta(days=oos_days - 1)
        if oos_end > last_day:
            break
        windows.append(
            {
                "train_start": train_start,
                "train_end": train_end,
                "oos_start": oos_start,
                "oos_end": oos_end,
            }
        )
    return windows


def walk_forward(
    close: pd.Series,
    *,
    n_values=N_VALUES,
    v_values=V_VALUES,
    normalize: bool = False,
    **kwargs,
) -> tuple[pd.DataFrame, pd.DataFrame]:
    et = pd.Series(_et_index(close.index), index=close.index)
    rmv_by_n = {n: rmv(close, n, normalize=normalize) for n in n_values}
    window_keys = {"n_windows", "train_days", "oos_days", "step_days", "tz"}
    signal_keys = {"first_trade", "exit_time", "execution_lag", "tz"}
    trade_keys = {"point_value", "size"}
    summary = []
    all_scores = []

    for i, window in enumerate(
        walk_forward_windows(
            close, **{k: v for k, v in kwargs.items() if k in window_keys}
        ),
        1,
    ):
        train_mask = (et >= window["train_start"]) & (
            et <= window["train_end"] + pd.Timedelta(days=1)
        )
        oos_mask = (et >= window["oos_start"]) & (
            et <= window["oos_end"] + pd.Timedelta(days=1)
        )
        train_close = close.loc[train_mask]
        oos_close = close.loc[oos_mask]
        if train_close.empty or oos_close.empty:
            continue

        best, scores = optimize_window(
            train_close,
            rmv_by_n,
            n_values=n_values,
            v_values=v_values,
            **{k: v for k, v in kwargs.items() if k not in window_keys},
        )
        scores.insert(0, "window", i)
        all_scores.append(scores)
        if best is None:
            summary.append(
                {
                    "window": i,
                    **window,
                    "n": np.nan,
                    "vup": np.nan,
                    "vdn": np.nan,
                    "oos_net_profit": np.nan,
                }
            )
            continue

        oos_position = positions_from_rmv(
            rmv_by_n[int(best["n"])].reindex(oos_close.index),
            float(best["vup"]),
            float(best["vdn"]),
            **{k: v for k, v in kwargs.items() if k in signal_keys},
        )
        oos_trades = trades_from_position(
            oos_close,
            oos_position,
            **{k: v for k, v in kwargs.items() if k in trade_keys},
        )
        oos = trade_metrics(oos_trades)
        summary.append(
            {
                "window": i,
                **window,
                "n": int(best["n"]),
                "vup": float(best["vup"]),
                "vdn": float(best["vdn"]),
                "train_net_profit": float(best["net_profit"]),
                "train_profit_factor": float(best["profit_factor"]),
                "train_max_losses": float(best["max_losses"]),
                "train_trades": float(best["trades"]),
                "train_r22": float(best["r22"]),
                "oos_net_profit": oos["net_profit"],
                "oos_trades": oos["trades"],
            }
        )

    return pd.DataFrame(summary), pd.concat(
        all_scores, ignore_index=True
    ) if all_scores else pd.DataFrame()


def calculate_rmv(prices, n):
    return (
        repeated_median_slope(np.asarray(prices)[-n:]) if len(prices) >= n else np.nan
    )


def rmv_trading_system(
    data: pd.DataFrame, n: int, vup: float, vdn: float
) -> pd.DataFrame:
    out = data.copy()
    close = pd.Series(
        out["close"].to_numpy(), index=pd.DatetimeIndex(out["datetime"]), name="close"
    )
    result = run_strategy(close, n, vup, vdn)
    out["RMV"] = result["rmv"].to_numpy()
    out["position"] = result["position"].to_numpy()
    out["signal"] = np.sign(out["position"].diff().fillna(out["position"]))
    return out


def calculate_returns(data: pd.DataFrame) -> pd.DataFrame:
    out = data.copy()
    out["returns"] = out["close"].pct_change().fillna(0.0)
    out["strategy_returns"] = out["position"].shift().fillna(0.0) * out["returns"]
    return out


def backtest(data: pd.DataFrame, n: int, vup: float, vdn: float):
    results = calculate_returns(rmv_trading_system(data, n, vup, vdn))
    total_return = results["strategy_returns"].sum()
    std = results["strategy_returns"].std()
    sharpe_ratio = (
        np.nan
        if std == 0
        else results["strategy_returns"].mean() / std * np.sqrt(252 * 78)
    )
    return results, total_return, sharpe_ratio


def _self_check() -> None:
    y = pd.Series([1, 2, 3, 4, 5, 15, 12, 8, 9, 10])
    assert repeated_median_slope(y.to_numpy()) == 1.0
    assert rmv(y, 10).iloc[-1] == 1.0

    idx = pd.date_range(
        "2024-01-02 09:30", periods=80, freq="5min", tz="America/New_York"
    )
    test_rmv = pd.Series(0.0, index=idx)
    test_rmv.loc["2024-01-02 10:00"] = 1.0
    test_rmv.loc["2024-01-02 15:50"] = -1.0
    pos = positions_from_rmv(test_rmv, 0.5, 0.5, execution_lag=1)
    assert pos.loc["2024-01-02 09:55"] == 0
    assert pos.loc["2024-01-02 10:05"] == 1
    assert pos.loc["2024-01-02 15:55"] == 0


if __name__ == "__main__":
    import os

    api_key = os.environ.get("API_KEY")
    secret_key = os.environ.get("SECRET_KEY")
    alpaca_client = StockHistoricalDataClient(api_key, secret_key)
    price = get_data("SPY", alpaca_client)

    summary, all_scores = walk_forward(price)
    print(summary)
    print(all_scores)
