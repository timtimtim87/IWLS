import pandas as pd
import numpy as np
import os
import glob
from datetime import datetime, timedelta
import matplotlib.pyplot as plt
import warnings
warnings.filterwarnings('ignore')

def load_all_iwls_results():
    """
    Load all IWLS results from the ASSETS_RESULTS folder
    """
    results_files = glob.glob("/Users/tim/IWLS-OPTIONS/ASSETS_RESULTS/*_iwls_results.csv")
    
    if not results_files:
        print("No results files found in /Users/tim/IWLS-OPTIONS/ASSETS_RESULTS/")
        return None
    
    all_results = {}
    
    for file_path in results_files:
        asset_name = os.path.basename(file_path).replace('_iwls_results.csv', '')
        
        try:
            df = pd.read_csv(file_path)
            df['date'] = pd.to_datetime(df['date'])
            df = df.dropna()  # Remove rows with NaN values
            df['price_deviation'] = ((df['price'] / df['trend_line_value']) - 1) * 100
            all_results[asset_name] = df
            print(f"Loaded {asset_name}: {len(df)} data points")
        except Exception as e:
            print(f"Error loading {asset_name}: {str(e)}")
    
    return all_results

def normalize_deviations_across_assets(all_results, date):
    """
    Normalize deviations across all assets for a given date using z-scores
    """
    asset_deviations = {}
    
    # Get deviation for each asset on this date
    for asset_name, df in all_results.items():
        asset_data = df[df['date'] <= date]
        if len(asset_data) > 0:
            latest_row = asset_data.iloc[-1]
            asset_deviations[asset_name] = latest_row['price_deviation']
    
    if len(asset_deviations) < 5:
        return {}
    
    # Calculate z-scores (normalized deviations)
    deviations = list(asset_deviations.values())
    mean_dev = np.mean(deviations)
    std_dev = np.std(deviations)
    
    if std_dev == 0:
        return {}
    
    normalized_deviations = {}
    for asset, deviation in asset_deviations.items():
        z_score = (deviation - mean_dev) / std_dev
        normalized_deviations[asset] = z_score
    
    return normalized_deviations

def get_rebalance_dates(start_date, end_date, interval_days=180):
    """
    Generate rebalance dates every N days
    """
    dates = []
    current_date = start_date
    
    while current_date <= end_date:
        dates.append(current_date)
        current_date += timedelta(days=interval_days)
    
    return dates

def get_asset_price(all_results, asset, date):
    """
    Get asset price on or before given date
    """
    if asset not in all_results:
        return None
    
    df = all_results[asset]
    asset_data = df[df['date'] <= date]
    
    if len(asset_data) == 0:
        return None
    
    return asset_data.iloc[-1]['price']

def run_rebalancing_strategy(all_results, start_date, end_date, initial_capital=10000, num_assets=5, interval_days=180):
    """
    Run the rebalancing strategy with configurable interval
    """
    print(f"\nRunning {interval_days}-day rebalancing strategy...")
    print(f"Initial capital: ${initial_capital:,.2f}")
    print(f"Strategy: Buy {num_assets} most undervalued assets every {interval_days} days")
    print(f"Period: {start_date.strftime('%Y-%m-%d')} to {end_date.strftime('%Y-%m-%d')}")
    
    # Get rebalance dates
    rebalance_dates = get_rebalance_dates(start_date, end_date, interval_days)
    print(f"Number of rebalancing periods: {len(rebalance_dates)}")
    
    portfolio = {}
    cash = initial_capital
    portfolio_history = []
    
    for i, rebalance_date in enumerate(rebalance_dates):
        print(f"\n--- Rebalance {i+1}: {rebalance_date.strftime('%Y-%m-%d')} ---")
        
        # Sell existing positions if not first rebalance
        if portfolio:
            print("Selling existing positions:")
            total_sale_proceeds = 0
            for asset, shares in portfolio.items():
                current_price = get_asset_price(all_results, asset, rebalance_date)
                if current_price:
                    proceeds = shares * current_price
                    total_sale_proceeds += proceeds
                    print(f"  {asset}: {shares:.2f} shares @ ${current_price:.2f} = ${proceeds:.2f}")
            
            cash = total_sale_proceeds
            portfolio = {}
            print(f"Total cash after sales: ${cash:.2f}")
        
        # Get normalized deviations for all assets
        normalized_devs = normalize_deviations_across_assets(all_results, rebalance_date)
        
        if len(normalized_devs) < num_assets:
            print(f"Insufficient assets with data ({len(normalized_devs)} available)")
            continue
        
        # Sort by normalized deviation (most negative = most undervalued)
        sorted_assets = sorted(normalized_devs.items(), key=lambda x: x[1])
        worst_assets = sorted_assets[:num_assets]
        
        print("Selected assets (most undervalued):")
        for asset, norm_dev in worst_assets:
            print(f"  {asset}: normalized deviation = {norm_dev:.2f}")
        
        # Buy equal amounts of each selected asset
        allocation_per_asset = cash / num_assets
        
        print(f"\nBuying positions (${allocation_per_asset:.2f} each):")
        for asset, norm_dev in worst_assets:
            current_price = get_asset_price(all_results, asset, rebalance_date)
            if current_price and current_price > 0:
                shares = allocation_per_asset / current_price
                portfolio[asset] = shares
                print(f"  {asset}: {shares:.2f} shares @ ${current_price:.2f}")
            else:
                print(f"  {asset}: No price data available")
        
        # Calculate portfolio value
        portfolio_value = 0
        for asset, shares in portfolio.items():
            current_price = get_asset_price(all_results, asset, rebalance_date)
            if current_price:
                portfolio_value += shares * current_price
        
        print(f"Total portfolio value: ${portfolio_value:.2f}")
        
        # Record portfolio state
        portfolio_history.append({
            'date': rebalance_date,
            'rebalance_num': i + 1,
            'portfolio_value': portfolio_value,
            'cash': 0,  # All cash invested
            'assets': list(portfolio.keys()),
            'normalized_deviations': dict(worst_assets)
        })
    
    # Calculate final portfolio value
    final_date = end_date
    final_portfolio_value = 0
    
    print(f"\n--- Final Portfolio Value ({final_date.strftime('%Y-%m-%d')}) ---")
    for asset, shares in portfolio.items():
        final_price = get_asset_price(all_results, asset, final_date)
        if final_price:
            asset_value = shares * final_price
            final_portfolio_value += asset_value
            print(f"{asset}: {shares:.2f} shares @ ${final_price:.2f} = ${asset_value:.2f}")
    
    total_return = ((final_portfolio_value / initial_capital) - 1) * 100
    print(f"\nStrategy Results:")
    print(f"Initial capital: ${initial_capital:.2f}")
    print(f"Final portfolio value: ${final_portfolio_value:.2f}")
    print(f"Total return: {total_return:.2f}%")
    
    return portfolio_history, final_portfolio_value

def calculate_daily_portfolio_value(all_results, portfolio_history, start_date, end_date):
    """
    Calculate daily portfolio values throughout the strategy period
    """
    print("\nCalculating daily portfolio values...")
    
    # Get all business days in the range
    all_dates = pd.date_range(start=start_date, end=end_date, freq='D')
    daily_values = []
    
    current_portfolio = {}
    current_rebalance_idx = 0
    
    for date in all_dates:
        # Check if we need to update portfolio (rebalance happened)
        if (current_rebalance_idx < len(portfolio_history) and 
            date >= portfolio_history[current_rebalance_idx]['date']):
            
            # Update to new portfolio composition
            rebalance_data = portfolio_history[current_rebalance_idx]
            current_portfolio = {}
            
            # Calculate shares for each asset based on equal allocation
            portfolio_value = rebalance_data['portfolio_value']
            num_assets = len(rebalance_data['assets'])
            
            if num_assets > 0:
                allocation_per_asset = portfolio_value / num_assets
                
                for asset in rebalance_data['assets']:
                    asset_price = get_asset_price(all_results, asset, rebalance_data['date'])
                    if asset_price and asset_price > 0:
                        shares = allocation_per_asset / asset_price
                        current_portfolio[asset] = shares
            
            current_rebalance_idx += 1
        
        # Calculate portfolio value for this date
        total_value = 0
        for asset, shares in current_portfolio.items():
            price = get_asset_price(all_results, asset, date)
            if price:
                total_value += shares * price
        
        daily_values.append({
            'date': date,
            'portfolio_value': total_value
        })
    
    return pd.DataFrame(daily_values)

def calculate_benchmark(all_results, asset_name, start_date, end_date, initial_capital=10000):
    """
    Calculate buy-and-hold benchmark performance for any asset
    """
    if asset_name not in all_results:
        print(f"Warning: {asset_name} not found in data")
        return None
    
    # Get starting price
    start_price = get_asset_price(all_results, asset_name, start_date)
    if not start_price:
        print(f"Warning: No starting price for {asset_name}")
        return None
    
    # Calculate shares purchased with initial capital
    shares = initial_capital / start_price
    
    # Calculate daily values
    all_dates = pd.date_range(start=start_date, end=end_date, freq='D')
    benchmark_values = []
    
    for date in all_dates:
        price = get_asset_price(all_results, asset_name, date)
        if price:
            value = shares * price
            benchmark_values.append({
                'date': date,
                f'{asset_name.lower()}_value': value
            })
    
    return pd.DataFrame(benchmark_values)

def plot_comprehensive_comparison(daily_portfolios, benchmarks, initial_capital=10000):
    """
    Create comprehensive comparison plot of all strategies and benchmarks
    """
    plt.figure(figsize=(16, 12))
    
    # Create subplots
    fig, ((ax1, ax2), (ax3, ax4)) = plt.subplots(2, 2, figsize=(20, 16))
    
    # Plot 1: All strategies comparison
    colors = ['blue', 'green', 'orange']
    intervals = [90, 180, 360]
    
    for i, (interval, daily_portfolio) in enumerate(daily_portfolios.items()):
        if len(daily_portfolio) > 0:
            ax1.plot(daily_portfolio['date'], daily_portfolio['portfolio_value'], 
                    label=f'IWLS {interval}-day', linewidth=2, color=colors[i])
    
    # Add benchmarks to first plot
    benchmark_colors = ['red', 'purple', 'brown', 'gray']
    benchmark_names = ['SPY', 'AMZN', 'AAPL']
    
    for i, (name, benchmark) in enumerate(benchmarks.items()):
        if benchmark is not None and len(benchmark) > 0:
            col_name = f'{name.lower()}_value'
            if col_name in benchmark.columns:
                ax1.plot(benchmark['date'], benchmark[col_name], 
                        label=f'{name} Buy & Hold', linewidth=2, color=benchmark_colors[i], alpha=0.7)
    
    ax1.axhline(y=initial_capital, color='gray', linestyle='--', alpha=0.5, label='Initial Capital')
    ax1.set_title('Strategy vs Benchmarks Comparison', fontsize=14, fontweight='bold')
    ax1.set_xlabel('Date')
    ax1.set_ylabel('Portfolio Value ($)')
    ax1.legend()
    ax1.grid(True, alpha=0.3)
    ax1.yaxis.set_major_formatter(plt.FuncFormatter(lambda x, p: f'${x:,.0f}'))
    
    # Plot 2: Returns comparison
    returns_data = []
    
    for interval, daily_portfolio in daily_portfolios.items():
        if len(daily_portfolio) > 0:
            final_value = daily_portfolio['portfolio_value'].iloc[-1]
            total_return = ((final_value / initial_capital) - 1) * 100
            returns_data.append({'Strategy': f'IWLS {interval}d', 'Return': total_return})
    
    for name, benchmark in benchmarks.items():
        if benchmark is not None and len(benchmark) > 0:
            col_name = f'{name.lower()}_value'
            if col_name in benchmark.columns:
                final_value = benchmark[col_name].iloc[-1]
                total_return = ((final_value / initial_capital) - 1) * 100
                returns_data.append({'Strategy': f'{name} B&H', 'Return': total_return})
    
    returns_df = pd.DataFrame(returns_data)
    if len(returns_df) > 0:
        bars = ax2.bar(returns_df['Strategy'], returns_df['Return'], 
                      color=['blue', 'green', 'orange', 'red', 'purple', 'brown'][:len(returns_df)])
        ax2.set_title('Total Returns Comparison', fontsize=14, fontweight='bold')
        ax2.set_ylabel('Total Return (%)')
        ax2.tick_params(axis='x', rotation=45)
        ax2.grid(True, alpha=0.3)
        
        # Add value labels on bars
        for bar, value in zip(bars, returns_df['Return']):
            height = bar.get_height()
            ax2.text(bar.get_x() + bar.get_width()/2., height + height*0.01,
                    f'{value:.1f}%', ha='center', va='bottom', fontweight='bold')
    
    # Plot 3: Volatility comparison
    volatility_data = []
    
    for interval, daily_portfolio in daily_portfolios.items():
        if len(daily_portfolio) > 0:
            returns = daily_portfolio['portfolio_value'].pct_change().dropna()
            volatility = returns.std() * np.sqrt(252) * 100  # Annualized volatility
            volatility_data.append({'Strategy': f'IWLS {interval}d', 'Volatility': volatility})
    
    for name, benchmark in benchmarks.items():
        if benchmark is not None and len(benchmark) > 0:
            col_name = f'{name.lower()}_value'
            if col_name in benchmark.columns:
                returns = benchmark[col_name].pct_change().dropna()
                volatility = returns.std() * np.sqrt(252) * 100  # Annualized volatility
                volatility_data.append({'Strategy': f'{name} B&H', 'Volatility': volatility})
    
    vol_df = pd.DataFrame(volatility_data)
    if len(vol_df) > 0:
        bars = ax3.bar(vol_df['Strategy'], vol_df['Volatility'], 
                      color=['blue', 'green', 'orange', 'red', 'purple', 'brown'][:len(vol_df)])
        ax3.set_title('Annualized Volatility Comparison', fontsize=14, fontweight='bold')
        ax3.set_ylabel('Volatility (%)')
        ax3.tick_params(axis='x', rotation=45)
        ax3.grid(True, alpha=0.3)
        
        # Add value labels on bars
        for bar, value in zip(bars, vol_df['Volatility']):
            height = bar.get_height()
            ax3.text(bar.get_x() + bar.get_width()/2., height + height*0.01,
                    f'{value:.1f}%', ha='center', va='bottom', fontweight='bold')
    
    # Plot 4: Sharpe ratio comparison
    sharpe_data = []
    
    for interval, daily_portfolio in daily_portfolios.items():
        if len(daily_portfolio) > 0:
            returns = daily_portfolio['portfolio_value'].pct_change().dropna()
            if len(returns) > 0 and returns.std() > 0:
                sharpe = (returns.mean() / returns.std()) * np.sqrt(252)  # Annualized Sharpe
                sharpe_data.append({'Strategy': f'IWLS {interval}d', 'Sharpe': sharpe})
    
    for name, benchmark in benchmarks.items():
        if benchmark is not None and len(benchmark) > 0:
            col_name = f'{name.lower()}_value'
            if col_name in benchmark.columns:
                returns = benchmark[col_name].pct_change().dropna()
                if len(returns) > 0 and returns.std() > 0:
                    sharpe = (returns.mean() / returns.std()) * np.sqrt(252)  # Annualized Sharpe
                    sharpe_data.append({'Strategy': f'{name} B&H', 'Sharpe': sharpe})
    
    sharpe_df = pd.DataFrame(sharpe_data)
    if len(sharpe_df) > 0:
        bars = ax4.bar(sharpe_df['Strategy'], sharpe_df['Sharpe'], 
                      color=['blue', 'green', 'orange', 'red', 'purple', 'brown'][:len(sharpe_df)])
        ax4.set_title('Sharpe Ratio Comparison', fontsize=14, fontweight='bold')
        ax4.set_ylabel('Sharpe Ratio')
        ax4.tick_params(axis='x', rotation=45)
        ax4.grid(True, alpha=0.3)
        
        # Add value labels on bars
        for bar, value in zip(bars, sharpe_df['Sharpe']):
            height = bar.get_height()
            ax4.text(bar.get_x() + bar.get_width()/2., height + height*0.01,
                    f'{value:.2f}', ha='center', va='bottom', fontweight='bold')
    
    plt.tight_layout()
    
    # Save the plot
    plot_path = '/Users/tim/IWLS-OPTIONS/comprehensive_comparison.png'
    plt.savefig(plot_path, dpi=300, bbox_inches='tight')
    print(f"\nComprehensive comparison plot saved to: {plot_path}")
    
    # Try to show the plot
    try:
        plt.show()
    except Exception as e:
        print(f"Could not display plot: {e}")
    
    return returns_df, vol_df, sharpe_df

def print_performance_summary(daily_portfolios, benchmarks, initial_capital=10000):
    """
    Print comprehensive performance summary
    """
    print(f"\n" + "="*80)
    print("COMPREHENSIVE PERFORMANCE SUMMARY")
    print("="*80)
    
    # Strategy results
    print(f"\nIWLS REBALANCING STRATEGIES:")
    print("-" * 40)
    
    for interval, daily_portfolio in daily_portfolios.items():
        if len(daily_portfolio) > 0:
            final_value = daily_portfolio['portfolio_value'].iloc[-1]
            total_return = ((final_value / initial_capital) - 1) * 100
            returns = daily_portfolio['portfolio_value'].pct_change().dropna()
            volatility = returns.std() * np.sqrt(252) * 100
            sharpe = (returns.mean() / returns.std()) * np.sqrt(252) if returns.std() > 0 else 0
            
            print(f"{interval}-day rebalancing:")
            print(f"  Final value: ${final_value:,.2f}")
            print(f"  Total return: {total_return:.2f}%")
            print(f"  Annualized volatility: {volatility:.2f}%")
            print(f"  Sharpe ratio: {sharpe:.3f}")
            print()
    
    # Benchmark results
    print(f"BUY & HOLD BENCHMARKS:")
    print("-" * 40)
    
    for name, benchmark in benchmarks.items():
        if benchmark is not None and len(benchmark) > 0:
            col_name = f'{name.lower()}_value'
            if col_name in benchmark.columns:
                final_value = benchmark[col_name].iloc[-1]
                total_return = ((final_value / initial_capital) - 1) * 100
                returns = benchmark[col_name].pct_change().dropna()
                volatility = returns.std() * np.sqrt(252) * 100
                sharpe = (returns.mean() / returns.std()) * np.sqrt(252) if returns.std() > 0 else 0
                
                print(f"{name}:")
                print(f"  Final value: ${final_value:,.2f}")
                print(f"  Total return: {total_return:.2f}%")
                print(f"  Annualized volatility: {volatility:.2f}%")
                print(f"  Sharpe ratio: {sharpe:.3f}")
                print()

def save_comprehensive_results(daily_portfolios, benchmarks, all_portfolio_histories):
    """
    Save all results to CSV files
    """
    print(f"\nSaving comprehensive results...")
    
    # Save each strategy's daily values
    for interval, daily_portfolio in daily_portfolios.items():
        daily_portfolio.to_csv(f"/Users/tim/IWLS-OPTIONS/daily_portfolio_values_{interval}d.csv", index=False)
        print(f"  Saved daily values for {interval}-day strategy")
    
    # Save benchmarks
    for name, benchmark in benchmarks.items():
        if benchmark is not None:
            benchmark.to_csv(f"/Users/tim/IWLS-OPTIONS/{name.lower()}_benchmark.csv", index=False)
            print(f"  Saved {name} benchmark data")
    
    # Save portfolio histories
    for interval, portfolio_history in all_portfolio_histories.items():
        portfolio_df = pd.DataFrame(portfolio_history)
        portfolio_df.to_csv(f"/Users/tim/IWLS-OPTIONS/portfolio_history_{interval}d.csv", index=False)
        print(f"  Saved portfolio history for {interval}-day strategy")

def main():
    print("IWLS COMPREHENSIVE REBALANCING STRATEGY ANALYSIS")
    print("=" * 80)
    
    # Load all IWLS results
    all_results = load_all_iwls_results()
    if not all_results:
        return
    
    # Determine date range
    all_dates = []
    for df in all_results.values():
        all_dates.extend(df['date'].tolist())
    
    start_date = min(all_dates)
    end_date = max(all_dates)
    
    # Ensure we have enough data for strategy
    min_start_date = start_date + timedelta(days=365)  # Need 1 year of IWLS data
    
    print(f"\nAvailable data range: {start_date.strftime('%Y-%m-%d')} to {end_date.strftime('%Y-%m-%d')}")
    print(f"Strategy start date: {min_start_date.strftime('%Y-%m-%d')}")
    
    # Test different rebalancing intervals
    intervals = [90, 180, 360]
    daily_portfolios = {}
    all_portfolio_histories = {}
    
    print(f"\n" + "="*60)
    print("TESTING MULTIPLE REBALANCING INTERVALS")
    print("="*60)
    
    for interval in intervals:
        print(f"\n{'='*20} {interval}-DAY REBALANCING {'='*20}")
        
        # Run strategy for this interval
        portfolio_history, final_value = run_rebalancing_strategy(
            all_results, 
            min_start_date, 
            end_date,
            initial_capital=10000,
            num_assets=5,
            interval_days=interval
        )
        
        # Calculate daily portfolio values
        daily_portfolio = calculate_daily_portfolio_value(
            all_results, portfolio_history, min_start_date, end_date
        )
        
        daily_portfolios[interval] = daily_portfolio
        all_portfolio_histories[interval] = portfolio_history
    
    # Calculate benchmarks
    print(f"\n" + "="*60)
    print("CALCULATING BENCHMARKS")
    print("="*60)
    
    benchmarks = {}
    benchmark_assets = ['SPY', 'AMZN', 'AAPL']
    
    for asset in benchmark_assets:
        print(f"\nCalculating {asset} buy-and-hold benchmark...")
        benchmark = calculate_benchmark(all_results, asset, min_start_date, end_date, initial_capital=10000)
        benchmarks[asset] = benchmark
        
        if benchmark is not None:
            final_value = benchmark[f'{asset.lower()}_value'].iloc[-1]
            total_return = ((final_value / 10000) - 1) * 100
            print(f"{asset} final value: ${final_value:,.2f} ({total_return:.2f}% return)")
    
    # Create comprehensive visualization
    print(f"\n" + "="*60)
    print("GENERATING COMPREHENSIVE ANALYSIS")
    print("="*60)
    
    returns_df, vol_df, sharpe_df = plot_comprehensive_comparison(daily_portfolios, benchmarks, initial_capital=10000)
    
    # Print performance summary
    print_performance_summary(daily_portfolios, benchmarks, initial_capital=10000)
    
    # Save results
    save_comprehensive_results(daily_portfolios, benchmarks, all_portfolio_histories)
    
    print(f"\n" + "="*60)
    print("ANALYSIS COMPLETE")
    print("="*60)
    print("Files saved:")
    print("  - comprehensive_comparison.png (4-panel comparison chart)")
    print("  - daily_portfolio_values_*d.csv (daily values for each strategy)")
    print("  - portfolio_history_*d.csv (rebalancing history for each strategy)")
    print("  - spy_benchmark.csv, amzn_benchmark.csv, aapl_benchmark.csv")

if __name__ == "__main__":
    main()