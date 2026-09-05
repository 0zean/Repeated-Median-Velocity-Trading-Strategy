import numpy as np
import pandas as pd

from rmv import calculate_returns, rmv_trading_system


def benchmark_strategy(data, wfo_results):
    """
    Benchmark the strategy using optimized parameters from walk-forward optimization.

    Parameters:
    data (pd.DataFrame): Full dataset with 'datetime' and 'close' columns
    wfo_results (pd.DataFrame): Results from walk-forward optimization

    Returns:
    pd.DataFrame: Combined results of applying optimized parameters to out-of-sample periods
    """
    combined_results = []

    for _, row in wfo_results.iterrows():
        # Extract parameters and date range for this period
        n, vup, vdn = row["n"], row["vup"], row["vdn"]
        start_date, end_date = row["oos_end"] - pd.Timedelta(days=7), row["oos_end"]

        # Get data for this period
        period_data = data[
            (data["datetime"] >= start_date) & (data["datetime"] < end_date)
        ]

        # Apply RMV trading system with optimized parameters
        results = rmv_trading_system(period_data, n, vup, vdn)
        results = calculate_returns(results)

        combined_results.append(results)

    # Concatenate all results
    return pd.concat(combined_results)


def calculate_performance_metrics(results):
    """Calculate various performance metrics for the strategy."""
    total_return = results["strategy_returns"].sum()
    sharpe_ratio = (
        results["strategy_returns"].mean()
        / results["strategy_returns"].std()
        * np.sqrt(252 * 78)
    )  # Assuming 5-minute bars
    max_drawdown = (
        results["strategy_returns"].cumsum()
        - results["strategy_returns"].cumsum().cummax()
    ).min()

    return {
        "Total Return": total_return,
        "Sharpe Ratio": sharpe_ratio,
        "Max Drawdown": max_drawdown,
    }


