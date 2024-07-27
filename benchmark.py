import pandas as pd
import numpy as np
import matplotlib.pyplot as plt
from rmv import rmv_trading_system, calculate_returns  # Assuming these are in a separate file

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
        n, vup, vdn = row['n'], row['vup'], row['vdn']
        start_date, end_date = row['oos_end'] - pd.Timedelta(days=7), row['oos_end']
        
        # Get data for this period
        period_data = data[(data['datetime'] >= start_date) & (data['datetime'] < end_date)]
        
        # Apply RMV trading system with optimized parameters
        results = rmv_trading_system(period_data, n, vup, vdn)
        results = calculate_returns(results)
        
        combined_results.append(results)
    
    # Concatenate all results
    return pd.concat(combined_results)

def calculate_performance_metrics(results):
    """Calculate various performance metrics for the strategy."""
    total_return = results['strategy_returns'].sum()
    sharpe_ratio = results['strategy_returns'].mean() / results['strategy_returns'].std() * np.sqrt(252 * 78)  # Assuming 5-minute bars
    max_drawdown = (results['strategy_returns'].cumsum() - results['strategy_returns'].cumsum().cummax()).min()
    
    return {
        'Total Return': total_return,
        'Sharpe Ratio': sharpe_ratio,
        'Max Drawdown': max_drawdown
    }

# Example usage
if __name__ == "__main__":
    # Load your data here (ensure it has 'datetime' and 'close' columns)
    # data = pd.read_csv('your_data.csv', parse_dates=['datetime'])
    
    # For demonstration, let's create some dummy data
    dates = pd.date_range(start='2023-05-01', end='2024-02-02', freq='5min')
    data = pd.DataFrame({
        'datetime': dates,
        'close': np.random.randn(len(dates)).cumsum() + 1000
    })
    
    # Create dummy wfo_results (replace this with your actual results)
    wfo_results = pd.DataFrame({
        'start_date': pd.date_range(start='2023-05-01', end='2023-12-27', freq='30D'),
        'test_end': pd.date_range(start='2023-05-31', end='2024-01-26', freq='30D'),
        'oos_end': pd.date_range(start='2023-06-07', end='2024-02-02', freq='30D'),
        'n': [20, 20, 20, 10, 20, 20, 55, 15, 20],
        'vup': [0.04, 0.02, 0.30, 0.24, 0.08, 0.06, 0.02, 0.02, 0.02],
        'vdn': [0.14, 0.18, 0.06, 0.08, 0.08, 0.04, 0.06, 0.10, 0.12],
    })
    
    # Benchmark the strategy
    benchmark_results = benchmark_strategy(data, wfo_results)
    
    # Calculate performance metrics
    performance_metrics = calculate_performance_metrics(benchmark_results)
    
    print("Strategy Performance Metrics:")
    for metric, value in performance_metrics.items():
        print(f"{metric}: {value:.4f}")
    
    # Plot cumulative returns
    plt.figure(figsize=(12, 6))
    plt.plot((1 + benchmark_results['strategy_returns']).cumprod().values - 1, label='Strategy')
    plt.plot((1 + benchmark_results['returns']).cumprod().values - 1, label='Buy and Hold')
    plt.legend()
    plt.title('Cumulative Returns: Strategy vs Buy and Hold')
    plt.xlabel('Date')
    plt.ylabel('Cumulative Return')
    plt.show()