import numpy as np
import pandas as pd
import vectorbt as vbt
import yaml
from scipy.stats import siegelslopes

with open("alpaca_api.yaml", 'r') as stream:
    alpaca_api = yaml.safe_load(stream)

data = vbt.AlpacaData.download('SPY', start='2024-05-06', end='2024-07-27', timeframe='5m', limit=10000)
price = data.get('Close')


def calculate_rmv(prices, n):
    """Calculate the Repeated Median Velocity (RMV) for a series of prices."""
    if len(prices) < n:
        return np.nan
    x = np.arange(n)
    y = prices[-n:]
    slope, _ = siegelslopes(y, x)
    return slope


def rmv_trading_system(data, n, vup, vdn):
    """
    Implement the Repeated Median Velocity trading system.
    
    Parameters:
    data (pd.DataFrame): DataFrame with 'datetime' and 'close' columns
    n (int): Lookback period for RMV calculation
    vup (float): Threshold for buy signals
    vdn (float): Threshold for sell signals
    
    Returns:
    pd.DataFrame: DataFrame with signals and positions
    """
    # Calculate RMV
    data['RMV'] = data['close'].rolling(window=n).apply(lambda x: calculate_rmv(x, n))
    
    # Generate signals
    data['signal'] = 0
    data.loc[data['RMV'] > vup, 'signal'] = 1
    data.loc[data['RMV'] < -vdn, 'signal'] = -1
    
    # Implement trading rules
    data['position'] = 0
    
    # First trade of day rule (no trades before 10:00 EST)
    market_open = pd.Timestamp('10:00').time()
    data.loc[data['datetime'].dt.time < market_open, 'signal'] = 0
    
    # Close positions 5 minutes before market close
    market_close = pd.Timestamp('15:55').time()
    data.loc[data['datetime'].dt.time >= market_close, 'signal'] = 0
    
    # Calculate positions
    data['position'] = data['signal'].shift(1)
    data['position'] = data['position'].fillna(0)
    
    return data


def calculate_returns(data):
    """Calculate returns based on positions."""
    data['returns'] = data['close'].pct_change()
    data['strategy_returns'] = data['position'] * data['returns']
    return data


def backtest(data, n, vup, vdn):
    """Run a backtest of the RMV trading system."""
    results = rmv_trading_system(data, n, vup, vdn)
    results = calculate_returns(results)
    
    # Calculate performance metrics
    total_return = results['strategy_returns'].sum()
    sharpe_ratio = results['strategy_returns'].mean() / results['strategy_returns'].std() * np.sqrt(252)
    
    return results, total_return, sharpe_ratio


# Example usage
if __name__ == "__main__":
    # Load your data here (ensure it has 'datetime' and 'close' columns)
    
    # For demonstration, let's create some dummy data
    data = pd.DataFrame({
        'datetime': price.index,
        'close': price
    })
    
    # Set parameters
    n, vup, vdn = 20, 0.02, 0.12
    
    # Run backtest
    results, total_return, sharpe_ratio = backtest(data, n, vup, vdn)
    
    print(f"Total Return: {total_return:.2%}")
    print(f"Sharpe Ratio: {sharpe_ratio:.2f}")
    
    # Plot results
    import matplotlib.pyplot as plt
    
    plt.figure(figsize=(12, 6))
    plt.plot(results['datetime'], results['close'], label='Price')
    plt.plot(results['datetime'], results['close'] * (1 + results['strategy_returns'].cumsum()), label='Strategy')
    plt.legend()
    plt.title('RMV Trading System Performance')
    plt.xlabel('Date')
    plt.ylabel('Price')
    plt.show()
