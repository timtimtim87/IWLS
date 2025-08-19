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

def calculate_annualized_return(start_value, end_value, start_date, end_date):
    """
    Calculate annualized return given start/end values and dates
    """
    years = (end_date - start_date).days / 365.25
    if years <= 0 or start_value <= 0:
        return 0
    
    annualized_return = ((end_value / start_value) ** (1 / years) - 1) * 100
    return annualized_return

def run_rebalancing_strategy(all_results, start_date, end_date, initial_capital=10000, num_assets=5, interval_days=180):
    """
    Run the rebalancing strategy with configurable interval and number of assets
    """
    print(f"\nRunning {interval_days}-day rebalancing strategy with {num_assets} assets...")
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
    annualized_return = calculate_annualized_return(initial_capital, final_portfolio_value, start_date, end_date)
    
    print(f"\nStrategy Results:")
    print(f"Initial capital: ${initial_capital:.2f}")
    print(f"Final portfolio value: ${final_portfolio_value:.2f}")
    print(f"Total return: {total_return:.2f}%")
    print(f"Annualized return: {annualized_return:.2f}%")
    
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

def plot_comprehensive_comparison(daily_portfolios, benchmarks, start_date, end_date, initial_capital=10000):
    """
    Create comprehensive comparison plot of all strategies and benchmarks
    """
    plt.figure(figsize=(20, 16))
    
    # Create subplots
    fig, ((ax1, ax2), (ax3, ax4)) = plt.subplots(2, 2, figsize=(20, 16))
    
    # Plot 1: All strategies comparison
    strategy_colors = ['blue', 'green', 'orange', 'cyan', 'magenta', 'yellow', 'red', 'purple', 'brown']
    color_idx = 0
    
    for strategy_name, daily_portfolio in daily_portfolios.items():
        if len(daily_portfolio) > 0:
            ax1.plot(daily_portfolio['date'], daily_portfolio['portfolio_value'], 
                    label=strategy_name, linewidth=2, color=strategy_colors[color_idx % len(strategy_colors)])
            color_idx += 1
    
    # Add benchmarks to first plot
    benchmark_colors = ['darkred', 'darkviolet', 'saddlebrown']
    
    for i, (name, benchmark) in enumerate(benchmarks.items()):
        if benchmark is not None and len(benchmark) > 0:
            col_name = f'{name.lower()}_value'
            if col_name in benchmark.columns:
                ax1.plot(benchmark['date'], benchmark[col_name], 
                        label=f'{name} Buy & Hold', linewidth=3, color=benchmark_colors[i], alpha=0.8, linestyle='--')
    
    ax1.axhline(y=initial_capital, color='gray', linestyle=':', alpha=0.7, label='Initial Capital')
    ax1.set_title('Strategy vs Benchmarks Comparison', fontsize=14, fontweight='bold')
    ax1.set_xlabel('Date')
    ax1.set_ylabel('Portfolio Value ($)')
    ax1.legend(bbox_to_anchor=(1.05, 1), loc='upper left')
    ax1.grid(True, alpha=0.3)
    ax1.yaxis.set_major_formatter(plt.FuncFormatter(lambda x, p: f'${x:,.0f}'))
    
    # Plot 2: Annualized returns comparison
    returns_data = []
    
    for strategy_name, daily_portfolio in daily_portfolios.items():
        if len(daily_portfolio) > 0:
            final_value = daily_portfolio['portfolio_value'].iloc[-1]
            annualized_return = calculate_annualized_return(initial_capital, final_value, start_date, end_date)
            returns_data.append({'Strategy': strategy_name, 'Annualized_Return': annualized_return})
    
    for name, benchmark in benchmarks.items():
        if benchmark is not None and len(benchmark) > 0:
            col_name = f'{name.lower()}_value'
            if col_name in benchmark.columns:
                final_value = benchmark[col_name].iloc[-1]
                annualized_return = calculate_annualized_return(initial_capital, final_value, start_date, end_date)
                returns_data.append({'Strategy': f'{name} B&H', 'Annualized_Return': annualized_return})
    
    returns_df = pd.DataFrame(returns_data)
    if len(returns_df) > 0:
        bars = ax2.bar(range(len(returns_df)), returns_df['Annualized_Return'], 
                      color=strategy_colors[:len(returns_df)])
        ax2.set_title('Annualized Returns Comparison', fontsize=14, fontweight='bold')
        ax2.set_ylabel('Annualized Return (%)')
        ax2.set_xticks(range(len(returns_df)))
        ax2.set_xticklabels(returns_df['Strategy'], rotation=45, ha='right')
        ax2.grid(True, alpha=0.3)
        
        # Add value labels on bars
        for i, (bar, value) in enumerate(zip(bars, returns_df['Annualized_Return'])):
            height = bar.get_height()
            ax2.text(bar.get_x() + bar.get_width()/2., height + height*0.01,
                    f'{value:.1f}%', ha='center', va='bottom', fontweight='bold', fontsize=8)
    
    # Plot 3: Volatility comparison
    volatility_data = []
    
    for strategy_name, daily_portfolio in daily_portfolios.items():
        if len(daily_portfolio) > 0:
            returns = daily_portfolio['portfolio_value'].pct_change().dropna()
            volatility = returns.std() * np.sqrt(252) * 100  # Annualized volatility
            volatility_data.append({'Strategy': strategy_name, 'Volatility': volatility})
    
    for name, benchmark in benchmarks.items():
        if benchmark is not None and len(benchmark) > 0:
            col_name = f'{name.lower()}_value'
            if col_name in benchmark.columns:
                returns = benchmark[col_name].pct_change().dropna()
                volatility = returns.std() * np.sqrt(252) * 100  # Annualized volatility
                volatility_data.append({'Strategy': f'{name} B&H', 'Volatility': volatility})
    
    vol_df = pd.DataFrame(volatility_data)
    if len(vol_df) > 0:
        bars = ax3.bar(range(len(vol_df)), vol_df['Volatility'], 
                      color=strategy_colors[:len(vol_df)])
        ax3.set_title('Annualized Volatility Comparison', fontsize=14, fontweight='bold')
        ax3.set_ylabel('Volatility (%)')
        ax3.set_xticks(range(len(vol_df)))
        ax3.set_xticklabels(vol_df['Strategy'], rotation=45, ha='right')
        ax3.grid(True, alpha=0.3)
        
        # Add value labels on bars
        for i, (bar, value) in enumerate(zip(bars, vol_df['Volatility'])):
            height = bar.get_height()
            ax3.text(bar.get_x() + bar.get_width()/2., height + height*0.01,
                    f'{value:.1f}%', ha='center', va='bottom', fontweight='bold', fontsize=8)
    
    # Plot 4: Sharpe ratio comparison
    sharpe_data = []
    
    for strategy_name, daily_portfolio in daily_portfolios.items():
        if len(daily_portfolio) > 0:
            returns = daily_portfolio['portfolio_value'].pct_change().dropna()
            if len(returns) > 0 and returns.std() > 0:
                sharpe = (returns.mean() / returns.std()) * np.sqrt(252)  # Annualized Sharpe
                sharpe_data.append({'Strategy': strategy_name, 'Sharpe': sharpe})
    
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
        bars = ax4.bar(range(len(sharpe_df)), sharpe_df['Sharpe'], 
                      color=strategy_colors[:len(sharpe_df)])
        ax4.set_title('Sharpe Ratio Comparison', fontsize=14, fontweight='bold')
        ax4.set_ylabel('Sharpe Ratio')
        ax4.set_xticks(range(len(sharpe_df)))
        ax4.set_xticklabels(sharpe_df['Strategy'], rotation=45, ha='right')
        ax4.grid(True, alpha=0.3)
        
        # Add value labels on bars
        for i, (bar, value) in enumerate(zip(bars, sharpe_df['Sharpe'])):
            height = bar.get_height()
            ax4.text(bar.get_x() + bar.get_width()/2., height + height*0.01,
                    f'{value:.2f}', ha='center', va='bottom', fontweight='bold', fontsize=8)
    
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

def print_performance_summary(daily_portfolios, benchmarks, start_date, end_date, initial_capital=10000):
    """
    Print comprehensive performance summary
    """
    print(f"\n" + "="*80)
    print("COMPREHENSIVE PERFORMANCE SUMMARY")
    print("="*80)
    
    # Strategy results
    print(f"\nIWLS REBALANCING STRATEGIES:")
    print("-" * 50)
    
    for strategy_name, daily_portfolio in daily_portfolios.items():
        if len(daily_portfolio) > 0:
            final_value = daily_portfolio['portfolio_value'].iloc[-1]
            total_return = ((final_value / initial_capital) - 1) * 100
            annualized_return = calculate_annualized_return(initial_capital, final_value, start_date, end_date)
            returns = daily_portfolio['portfolio_value'].pct_change().dropna()
            volatility = returns.std() * np.sqrt(252) * 100
            sharpe = (returns.mean() / returns.std()) * np.sqrt(252) if returns.std() > 0 else 0
            
            print(f"{strategy_name}:")
            print(f"  Final value: ${final_value:,.2f}")
            print(f"  Total return: {total_return:.2f}%")
            print(f"  Annualized return: {annualized_return:.2f}%")
            print(f"  Annualized volatility: {volatility:.2f}%")
            print(f"  Sharpe ratio: {sharpe:.3f}")
            print()
    
    # Benchmark results
    print(f"BUY & HOLD BENCHMARKS:")
    print("-" * 50)
    
    for name, benchmark in benchmarks.items():
        if benchmark is not None and len(benchmark) > 0:
            col_name = f'{name.lower()}_value'
            if col_name in benchmark.columns:
                final_value = benchmark[col_name].iloc[-1]
                total_return = ((final_value / initial_capital) - 1) * 100
                annualized_return = calculate_annualized_return(initial_capital, final_value, start_date, end_date)
                returns = benchmark[col_name].pct_change().dropna()
                volatility = returns.std() * np.sqrt(252) * 100
                sharpe = (returns.mean() / returns.std()) * np.sqrt(252) if returns.std() > 0 else 0
                
                print(f"{name}:")
                print(f"  Final value: ${final_value:,.2f}")
                print(f"  Total return: {total_return:.2f}%")
                print(f"  Annualized return: {annualized_return:.2f}%")
                print(f"  Annualized volatility: {volatility:.2f}%")
                print(f"  Sharpe ratio: {sharpe:.3f}")
                print()

def save_comprehensive_results(daily_portfolios, benchmarks, all_portfolio_histories):
    """
    Save all results to CSV files
    """
    print(f"\nSaving comprehensive results...")
    
    # Save each strategy's daily values
    for strategy_name, daily_portfolio in daily_portfolios.items():
        safe_name = strategy_name.replace(' ', '_').replace('-', '_')
        daily_portfolio.to_csv(f"/Users/tim/IWLS-OPTIONS/daily_portfolio_values_{safe_name}.csv", index=False)
        print(f"  Saved daily values for {strategy_name}")
    
    # Save benchmarks
    for name, benchmark in benchmarks.items():
        if benchmark is not None:
            benchmark.to_csv(f"/Users/tim/IWLS-OPTIONS/{name.lower()}_benchmark.csv", index=False)
            print(f"  Saved {name} benchmark data")
    
    # Save portfolio histories
    for strategy_name, portfolio_history in all_portfolio_histories.items():
        safe_name = strategy_name.replace(' ', '_').replace('-', '_')
        portfolio_df = pd.DataFrame(portfolio_history)
        portfolio_df.to_csv(f"/Users/tim/IWLS-OPTIONS/portfolio_history_{safe_name}.csv", index=False)
        print(f"  Saved portfolio history for {strategy_name}")

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
    
    # Test different combinations of intervals and asset counts
    intervals = [90, 180, 360]
    asset_counts = [3, 5, 8]
    daily_portfolios = {}
    all_portfolio_histories = {}
    
    print(f"\n" + "="*80)
    print("TESTING MULTIPLE REBALANCING INTERVALS AND PORTFOLIO SIZES")
    print("="*80)
    
    for interval in intervals:
        for num_assets in asset_counts:
            strategy_name = f"IWLS {interval}d-{num_assets}assets"
            print(f"\n{'='*20} {strategy_name} {'='*20}")
            
            # Run strategy for this combination
            portfolio_history, final_value = run_rebalancing_strategy(
                all_results, 
                min_start_date, 
                end_date,
                initial_capital=10000,
                num_assets=num_assets,
                interval_days=interval
            )
            
            # Calculate daily portfolio values
            daily_portfolio = calculate_daily_portfolio_value(
                all_results, portfolio_history, min_start_date, end_date
            )
            
            daily_portfolios[strategy_name] = daily_portfolio
            all_portfolio_histories[strategy_name] = portfolio_history
            
            # Print quick summary
            if len(daily_portfolio) > 0:
                final_val = daily_portfolio['portfolio_value'].iloc[-1]
                total_ret = ((final_val / 10000) - 1) * 100
                ann_ret = calculate_annualized_return(10000, final_val, min_start_date, end_date)
                print(f"Quick Summary: ${final_val:,.2f} ({total_ret:.1f}% total, {ann_ret:.1f}% annualized)")
    
    # Calculate benchmarks
    print(f"\n" + "="*80)
    print("CALCULATING BENCHMARKS")
    print("="*80)
    
    benchmarks = {}
    benchmark_assets = ['SPY', 'AMZN', 'AAPL']
    
    for asset in benchmark_assets:
        print(f"\nCalculating {asset} buy-and-hold benchmark...")
        benchmark = calculate_benchmark(all_results, asset, min_start_date, end_date, initial_capital=10000)
        benchmarks[asset] = benchmark
        
        if benchmark is not None:
            final_value = benchmark[f'{asset.lower()}_value'].iloc[-1]
            total_return = ((final_value / 10000) - 1) * 100
            annualized_return = calculate_annualized_return(10000, final_value, min_start_date, end_date)
            print(f"{asset} final value: ${final_value:,.2f} ({total_return:.1f}% total, {annualized_return:.1f}% annualized)")
    
    # Create comprehensive visualization
    print(f"\n" + "="*80)
    print("GENERATING COMPREHENSIVE ANALYSIS")
    print("="*80)
    
    returns_df, vol_df, sharpe_df = plot_comprehensive_comparison(daily_portfolios, benchmarks, min_start_date, end_date, initial_capital=10000)
    
    # Print performance summary
    print_performance_summary(daily_portfolios, benchmarks, min_start_date, end_date, initial_capital=10000)
    
    # Save results
    save_comprehensive_results(daily_portfolios, benchmarks, all_portfolio_histories)
    
    # Find and highlight best performing strategies
    print(f"\n" + "="*80)
    print("BEST PERFORMING STRATEGIES")
    print("="*80)
    
    strategy_performance = []
    for strategy_name, daily_portfolio in daily_portfolios.items():
        if len(daily_portfolio) > 0:
            final_value = daily_portfolio['portfolio_value'].iloc[-1]
            annualized_return = calculate_annualized_return(10000, final_value, min_start_date, end_date)
            returns = daily_portfolio['portfolio_value'].pct_change().dropna()
            sharpe = (returns.mean() / returns.std()) * np.sqrt(252) if returns.std() > 0 else 0
            strategy_performance.append({
                'Strategy': strategy_name,
                'Annualized_Return': annualized_return,
                'Sharpe_Ratio': sharpe,
                'Final_Value': final_value
            })
    
    perf_df = pd.DataFrame(strategy_performance)
    if len(perf_df) > 0:
        # Sort by annualized return
        perf_df_sorted = perf_df.sort_values('Annualized_Return', ascending=False)
        print("\nRANKED BY ANNUALIZED RETURN:")
        print("-" * 50)
        for i, row in perf_df_sorted.iterrows():
            print(f"{row['Strategy']:25} | {row['Annualized_Return']:8.2f}% | Sharpe: {row['Sharpe_Ratio']:6.3f} | ${row['Final_Value']:,.0f}")
        
        # Sort by Sharpe ratio
        perf_df_sharpe = perf_df.sort_values('Sharpe_Ratio', ascending=False)
        print("\nRANKED BY SHARPE RATIO (Risk-Adjusted Returns):")
        print("-" * 50)
        for i, row in perf_df_sharpe.iterrows():
            print(f"{row['Strategy']:25} | Sharpe: {row['Sharpe_Ratio']:6.3f} | {row['Annualized_Return']:8.2f}% | ${row['Final_Value']:,.0f}")
    
    print(f"\n" + "="*80)
    print("ANALYSIS COMPLETE")
    print("="*80)
    print("Files saved:")
    print("  - comprehensive_comparison.png (4-panel comparison chart)")
    print("  - daily_portfolio_values_*d-*assets.csv (daily values for each strategy)")
    print("  - portfolio_history_*d-*assets.csv (rebalancing history for each strategy)")
    print("  - spy_benchmark.csv, amzn_benchmark.csv, aapl_benchmark.csv")
    
    # Summary insights
    print(f"\n" + "="*80)
    print("KEY INSIGHTS")
    print("="*80)
    
    if len(perf_df) > 0:
        best_return = perf_df_sorted.iloc[0]
        best_sharpe = perf_df_sharpe.iloc[0]
        
        print(f"HIGHEST ANNUALIZED RETURN: {best_return['Strategy']}")
        print(f"   Return: {best_return['Annualized_Return']:.2f}% annually")
        print(f"   Final Value: ${best_return['Final_Value']:,.0f}")
        print()
        
        print(f"BEST RISK-ADJUSTED RETURN: {best_sharpe['Strategy']}")
        print(f"   Sharpe Ratio: {best_sharpe['Sharpe_Ratio']:.3f}")
        print(f"   Annualized Return: {best_sharpe['Annualized_Return']:.2f}%")
        print()
        
        # Analyze patterns
        interval_performance = {}
        asset_performance = {}
        
        for _, row in perf_df.iterrows():
            strategy = row['Strategy']
            if 'IWLS' in strategy:
                parts = strategy.split('-')
                interval = parts[0].split()[1]  # e.g., "90d"
                assets = parts[1]  # e.g., "3assets"
                
                if interval not in interval_performance:
                    interval_performance[interval] = []
                interval_performance[interval].append(row['Annualized_Return'])
                
                if assets not in asset_performance:
                    asset_performance[assets] = []
                asset_performance[assets].append(row['Annualized_Return'])
        
        print("PATTERN ANALYSIS:")
        print("-" * 30)
        
        # Best rebalancing frequency
        if interval_performance:
            avg_by_interval = {k: np.mean(v) for k, v in interval_performance.items()}
            best_interval = max(avg_by_interval, key=avg_by_interval.get)
            print(f"Best rebalancing frequency: {best_interval} (avg: {avg_by_interval[best_interval]:.2f}% annually)")
        
        # Best portfolio size
        if asset_performance:
            avg_by_assets = {k: np.mean(v) for k, v in asset_performance.items()}
            best_assets = max(avg_by_assets, key=avg_by_assets.get)
            print(f"Best portfolio size: {best_assets} (avg: {avg_by_assets[best_assets]:.2f}% annually)")
        
        print()
        print("RECOMMENDATIONS:")
        print("   - Consider the strategy with highest Sharpe ratio for best risk-adjusted returns")
        print("   - Higher frequency rebalancing may capture mean reversion better")
        print("   - Diversification effects: compare 3 vs 5 vs 8 asset performance")
        print("   - Monitor volatility levels relative to benchmark performance")

if __name__ == "__main__":
    main()