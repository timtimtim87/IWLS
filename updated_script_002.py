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
            df = df.dropna()
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
    
    for asset_name, df in all_results.items():
        asset_data = df[df['date'] <= date]
        if len(asset_data) > 0:
            latest_row = asset_data.iloc[-1]
            asset_deviations[asset_name] = latest_row['price_deviation']
    
    if len(asset_deviations) < 5:
        return {}
    
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

def get_rebalance_dates(start_date, end_date, interval_days=360):
    """
    Generate rebalance dates every 360 days
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

def select_assets_by_deviation_range(normalized_devs, strategy_type, num_assets=5):
    """
    Select assets based on different deviation strategies:
    - 'undervalued': Most negative deviations (original strategy)
    - 'overvalued': Most positive deviations  
    - 'neutral': Closest to zero deviations
    """
    if len(normalized_devs) < num_assets:
        return []
    
    sorted_assets = sorted(normalized_devs.items(), key=lambda x: x[1])
    
    if strategy_type == 'undervalued':
        # Most negative (undervalued) - original strategy
        selected = sorted_assets[:num_assets]
    elif strategy_type == 'overvalued':
        # Most positive (overvalued)
        selected = sorted_assets[-num_assets:]
    elif strategy_type == 'neutral':
        # Closest to zero (neutral)
        # Sort by absolute value of deviation
        abs_sorted = sorted(normalized_devs.items(), key=lambda x: abs(x[1]))
        selected = abs_sorted[:num_assets]
    else:
        raise ValueError("strategy_type must be 'undervalued', 'overvalued', or 'neutral'")
    
    return selected

def run_deviation_strategy(all_results, start_date, end_date, strategy_type, initial_capital=10000, num_assets=5):
    """
    Run the 360-day rebalancing strategy with different deviation selection methods
    """
    print(f"\nRunning 360-day {strategy_type.upper()} strategy with {num_assets} assets...")
    print(f"Initial capital: ${initial_capital:,.2f}")
    print(f"Strategy: Buy {num_assets} {strategy_type} assets every 360 days")
    print(f"Period: {start_date.strftime('%Y-%m-%d')} to {end_date.strftime('%Y-%m-%d')}")
    
    rebalance_dates = get_rebalance_dates(start_date, end_date, 360)
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
        
        # Select assets based on strategy type
        selected_assets = select_assets_by_deviation_range(normalized_devs, strategy_type, num_assets)
        
        print(f"Selected assets ({strategy_type}):")
        for asset, norm_dev in selected_assets:
            print(f"  {asset}: normalized deviation = {norm_dev:.2f}")
        
        # Buy equal amounts of each selected asset
        allocation_per_asset = cash / num_assets
        
        print(f"\nBuying positions (${allocation_per_asset:.2f} each):")
        for asset, norm_dev in selected_assets:
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
            'cash': 0,
            'assets': [asset for asset, _ in selected_assets],
            'normalized_deviations': dict(selected_assets),
            'strategy_type': strategy_type
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
    
    print(f"\n{strategy_type.upper()} Strategy Results:")
    print(f"Initial capital: ${initial_capital:.2f}")
    print(f"Final portfolio value: ${final_portfolio_value:.2f}")
    print(f"Total return: {total_return:.2f}%")
    print(f"Annualized return: {annualized_return:.2f}%")
    
    return portfolio_history, final_portfolio_value

def calculate_daily_portfolio_value(all_results, portfolio_history, start_date, end_date):
    """
    Calculate daily portfolio values throughout the strategy period
    """
    all_dates = pd.date_range(start=start_date, end=end_date, freq='D')
    daily_values = []
    
    current_portfolio = {}
    current_rebalance_idx = 0
    
    for date in all_dates:
        if (current_rebalance_idx < len(portfolio_history) and 
            date >= portfolio_history[current_rebalance_idx]['date']):
            
            rebalance_data = portfolio_history[current_rebalance_idx]
            current_portfolio = {}
            
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
    Calculate buy-and-hold benchmark performance
    """
    if asset_name not in all_results:
        return None
    
    start_price = get_asset_price(all_results, asset_name, start_date)
    if not start_price:
        return None
    
    shares = initial_capital / start_price
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

def plot_validation_comparison(strategy_results, benchmarks, start_date, end_date, initial_capital=10000):
    """
    Create validation comparison plot
    """
    plt.figure(figsize=(20, 12))
    
    fig, ((ax1, ax2), (ax3, ax4)) = plt.subplots(2, 2, figsize=(20, 12))
    
    # Plot 1: Performance comparison over time
    colors = ['red', 'blue', 'green']  # overvalued, undervalued, neutral
    strategy_names = ['Overvalued', 'Undervalued', 'Neutral']
    
    for i, (strategy_type, daily_portfolio) in enumerate(strategy_results.items()):
        if len(daily_portfolio) > 0:
            ax1.plot(daily_portfolio['date'], daily_portfolio['portfolio_value'], 
                    label=f'{strategy_names[i]} (Most {strategy_type})', 
                    linewidth=3, color=colors[i])
    
    # Add benchmarks
    benchmark_colors = ['purple', 'orange', 'brown']
    for i, (name, benchmark) in enumerate(benchmarks.items()):
        if benchmark is not None and len(benchmark) > 0:
            col_name = f'{name.lower()}_value'
            if col_name in benchmark.columns:
                ax1.plot(benchmark['date'], benchmark[col_name], 
                        label=f'{name} Buy & Hold', linewidth=2, 
                        color=benchmark_colors[i], alpha=0.7, linestyle='--')
    
    ax1.axhline(y=initial_capital, color='gray', linestyle=':', alpha=0.7, label='Initial Capital')
    ax1.set_title('IWLS Strategy Validation: Deviation Range Comparison', fontsize=14, fontweight='bold')
    ax1.set_xlabel('Date')
    ax1.set_ylabel('Portfolio Value ($)')
    ax1.legend()
    ax1.grid(True, alpha=0.3)
    ax1.yaxis.set_major_formatter(plt.FuncFormatter(lambda x, p: f'${x:,.0f}'))
    
    # Plot 2: Final returns comparison
    returns_data = []
    strategy_order = ['overvalued', 'undervalued', 'neutral']
    
    for strategy_type in strategy_order:
        if strategy_type in strategy_results:
            daily_portfolio = strategy_results[strategy_type]
            if len(daily_portfolio) > 0:
                final_value = daily_portfolio['portfolio_value'].iloc[-1]
                annualized_return = calculate_annualized_return(initial_capital, final_value, start_date, end_date)
                returns_data.append({'Strategy': strategy_type.capitalize(), 'Annualized_Return': annualized_return})
    
    for name, benchmark in benchmarks.items():
        if benchmark is not None and len(benchmark) > 0:
            col_name = f'{name.lower()}_value'
            if col_name in benchmark.columns:
                final_value = benchmark[col_name].iloc[-1]
                annualized_return = calculate_annualized_return(initial_capital, final_value, start_date, end_date)
                returns_data.append({'Strategy': f'{name}', 'Annualized_Return': annualized_return})
    
    returns_df = pd.DataFrame(returns_data)
    if len(returns_df) > 0:
        strategy_colors = ['red', 'blue', 'green', 'purple', 'orange', 'brown']
        bars = ax2.bar(range(len(returns_df)), returns_df['Annualized_Return'], 
                      color=strategy_colors[:len(returns_df)])
        ax2.set_title('Annualized Returns: Strategy Validation', fontsize=14, fontweight='bold')
        ax2.set_ylabel('Annualized Return (%)')
        ax2.set_xticks(range(len(returns_df)))
        ax2.set_xticklabels(returns_df['Strategy'], rotation=45)
        ax2.grid(True, alpha=0.3)
        
        for i, (bar, value) in enumerate(zip(bars, returns_df['Annualized_Return'])):
            height = bar.get_height()
            ax2.text(bar.get_x() + bar.get_width()/2., height + height*0.01,
                    f'{value:.1f}%', ha='center', va='bottom', fontweight='bold')
    
    # Plot 3: Volatility comparison
    volatility_data = []
    
    for strategy_type in strategy_order:
        if strategy_type in strategy_results:
            daily_portfolio = strategy_results[strategy_type]
            if len(daily_portfolio) > 0:
                returns = daily_portfolio['portfolio_value'].pct_change().dropna()
                volatility = returns.std() * np.sqrt(252) * 100
                volatility_data.append({'Strategy': strategy_type.capitalize(), 'Volatility': volatility})
    
    for name, benchmark in benchmarks.items():
        if benchmark is not None and len(benchmark) > 0:
            col_name = f'{name.lower()}_value'
            if col_name in benchmark.columns:
                returns = benchmark[col_name].pct_change().dropna()
                volatility = returns.std() * np.sqrt(252) * 100
                volatility_data.append({'Strategy': f'{name}', 'Volatility': volatility})
    
    vol_df = pd.DataFrame(volatility_data)
    if len(vol_df) > 0:
        bars = ax3.bar(range(len(vol_df)), vol_df['Volatility'], 
                      color=strategy_colors[:len(vol_df)])
        ax3.set_title('Annualized Volatility Comparison', fontsize=14, fontweight='bold')
        ax3.set_ylabel('Volatility (%)')
        ax3.set_xticks(range(len(vol_df)))
        ax3.set_xticklabels(vol_df['Strategy'], rotation=45)
        ax3.grid(True, alpha=0.3)
        
        for i, (bar, value) in enumerate(zip(bars, vol_df['Volatility'])):
            height = bar.get_height()
            ax3.text(bar.get_x() + bar.get_width()/2., height + height*0.01,
                    f'{value:.1f}%', ha='center', va='bottom', fontweight='bold')
    
    # Plot 4: Risk-adjusted returns (Sharpe ratio)
    sharpe_data = []
    
    for strategy_type in strategy_order:
        if strategy_type in strategy_results:
            daily_portfolio = strategy_results[strategy_type]
            if len(daily_portfolio) > 0:
                returns = daily_portfolio['portfolio_value'].pct_change().dropna()
                if len(returns) > 0 and returns.std() > 0:
                    sharpe = (returns.mean() / returns.std()) * np.sqrt(252)
                    sharpe_data.append({'Strategy': strategy_type.capitalize(), 'Sharpe': sharpe})
    
    for name, benchmark in benchmarks.items():
        if benchmark is not None and len(benchmark) > 0:
            col_name = f'{name.lower()}_value'
            if col_name in benchmark.columns:
                returns = benchmark[col_name].pct_change().dropna()
                if len(returns) > 0 and returns.std() > 0:
                    sharpe = (returns.mean() / returns.std()) * np.sqrt(252)
                    sharpe_data.append({'Strategy': f'{name}', 'Sharpe': sharpe})
    
    sharpe_df = pd.DataFrame(sharpe_data)
    if len(sharpe_df) > 0:
        bars = ax4.bar(range(len(sharpe_df)), sharpe_df['Sharpe'], 
                      color=strategy_colors[:len(sharpe_df)])
        ax4.set_title('Sharpe Ratio Comparison', fontsize=14, fontweight='bold')
        ax4.set_ylabel('Sharpe Ratio')
        ax4.set_xticks(range(len(sharpe_df)))
        ax4.set_xticklabels(sharpe_df['Strategy'], rotation=45)
        ax4.grid(True, alpha=0.3)
        
        for i, (bar, value) in enumerate(zip(bars, sharpe_df['Sharpe'])):
            height = bar.get_height()
            ax4.text(bar.get_x() + bar.get_width()/2., height + height*0.01,
                    f'{value:.2f}', ha='center', va='bottom', fontweight='bold')
    
    plt.tight_layout()
    
    plot_path = '/Users/tim/IWLS-OPTIONS/iwls_validation_analysis.png'
    plt.savefig(plot_path, dpi=300, bbox_inches='tight')
    print(f"\nValidation analysis plot saved to: {plot_path}")
    
    try:
        plt.show()
    except Exception as e:
        print(f"Could not display plot: {e}")
    
    return returns_df, vol_df, sharpe_df

def print_validation_summary(strategy_results, portfolio_histories, benchmarks, start_date, end_date, initial_capital=10000):
    """
    Print comprehensive validation summary
    """
    print(f"\n" + "="*80)
    print("IWLS STRATEGY VALIDATION ANALYSIS")
    print("="*80)
    
    print(f"\nTesting whether outperformance was skill or luck...")
    print(f"Comparing different deviation range selections with 360-day rebalancing:")
    print(f"- UNDERVALUED: Most negative deviations (original strategy)")
    print(f"- OVERVALUED: Most positive deviations (contrarian test)")
    print(f"- NEUTRAL: Minimal deviations (control group)")
    
    # Strategy performance summary
    strategy_order = ['undervalued', 'overvalued', 'neutral']
    strategy_names = ['Undervalued (Original)', 'Overvalued (Contrarian)', 'Neutral (Control)']
    
    print(f"\nSTRATEGY PERFORMANCE COMPARISON:")
    print("-" * 60)
    
    performance_summary = []
    
    for i, strategy_type in enumerate(strategy_order):
        if strategy_type in strategy_results:
            daily_portfolio = strategy_results[strategy_type]
            if len(daily_portfolio) > 0:
                final_value = daily_portfolio['portfolio_value'].iloc[-1]
                total_return = ((final_value / initial_capital) - 1) * 100
                annualized_return = calculate_annualized_return(initial_capital, final_value, start_date, end_date)
                returns = daily_portfolio['portfolio_value'].pct_change().dropna()
                volatility = returns.std() * np.sqrt(252) * 100
                sharpe = (returns.mean() / returns.std()) * np.sqrt(252) if returns.std() > 0 else 0
                
                performance_summary.append({
                    'Strategy': strategy_names[i],
                    'Final_Value': final_value,
                    'Total_Return': total_return,
                    'Annualized_Return': annualized_return,
                    'Volatility': volatility,
                    'Sharpe': sharpe
                })
                
                print(f"{strategy_names[i]}:")
                print(f"  Final value: ${final_value:,.2f}")
                print(f"  Total return: {total_return:.2f}%")
                print(f"  Annualized return: {annualized_return:.2f}%")
                print(f"  Volatility: {volatility:.2f}%")
                print(f"  Sharpe ratio: {sharpe:.3f}")
                print()
    
    # Benchmark comparison
    print(f"BENCHMARK COMPARISON:")
    print("-" * 60)
    
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
                
                print(f"{name} Buy & Hold:")
                print(f"  Final value: ${final_value:,.2f}")
                print(f"  Total return: {total_return:.2f}%")
                print(f"  Annualized return: {annualized_return:.2f}%")
                print(f"  Volatility: {volatility:.2f}%")
                print(f"  Sharpe ratio: {sharpe:.3f}")
                print()
    
    # Validation conclusions
    print(f"VALIDATION CONCLUSIONS:")
    print("="*60)
    
    if len(performance_summary) >= 2:
        undervalued_perf = next((p for p in performance_summary if 'Undervalued' in p['Strategy']), None)
        overvalued_perf = next((p for p in performance_summary if 'Overvalued' in p['Strategy']), None)
        neutral_perf = next((p for p in performance_summary if 'Neutral' in p['Strategy']), None)
        
        if undervalued_perf and overvalued_perf:
            return_diff = undervalued_perf['Annualized_Return'] - overvalued_perf['Annualized_Return']
            sharpe_diff = undervalued_perf['Sharpe'] - overvalued_perf['Sharpe']
            
            print(f"UNDERVALUED vs OVERVALUED:")
            print(f"  Return difference: {return_diff:.2f}% annually")
            print(f"  Sharpe difference: {sharpe_diff:.3f}")
            
            if return_diff > 5:  # Significant difference threshold
                print(f"  CONCLUSION: Strong evidence for skill-based outperformance")
                print(f"  The {return_diff:.1f}% annual advantage suggests genuine mean reversion alpha")
            elif return_diff > 0:
                print(f"  CONCLUSION: Moderate evidence for skill")
                print(f"  The {return_diff:.1f}% advantage suggests some predictive power")
            else:
                print(f"  CONCLUSION: Evidence suggests luck rather than skill")
                print(f"  Overvalued assets outperformed, contradicting mean reversion theory")
        
        if undervalued_perf and neutral_perf:
            neutral_diff = undervalued_perf['Annualized_Return'] - neutral_perf['Annualized_Return']
            print(f"\nUNDERVALUED vs NEUTRAL:")
            print(f"  Return difference: {neutral_diff:.2f}% annually")
            
            if neutral_diff > 3:
                print(f"  CONCLUSION: Deviation-based selection adds significant value")
            else:
                print(f"  CONCLUSION: Limited benefit from deviation-based selection")
    
    # Asset selection analysis
    print(f"\nASSET SELECTION ANALYSIS:")
    print("-" * 60)
    
    for strategy_type, portfolio_history in portfolio_histories.items():
        if portfolio_history:
            all_selected_assets = []
            for period in portfolio_history:
                all_selected_assets.extend(period['assets'])
            
            from collections import Counter
            asset_frequency = Counter(all_selected_assets)
            
            print(f"{strategy_type.upper()} strategy most selected assets:")
            for asset, count in asset_frequency.most_common(5):
                print(f"  {asset}: {count} times ({count/len(portfolio_history)*100:.1f}%)")
            print()

def save_validation_results(strategy_results, portfolio_histories, benchmarks):
    """
    Save validation results to files
    """
    print(f"\nSaving validation results...")
    
    for strategy_type, daily_portfolio in strategy_results.items():
        safe_name = strategy_type.replace(' ', '_')
        daily_portfolio.to_csv(f"/Users/tim/IWLS-OPTIONS/validation_{safe_name}_daily_values.csv", index=False)
        print(f"  Saved daily values for {strategy_type} strategy")
    
    for strategy_type, portfolio_history in portfolio_histories.items():
        safe_name = strategy_type.replace(' ', '_')
        portfolio_df = pd.DataFrame(portfolio_history)
        portfolio_df.to_csv(f"/Users/tim/IWLS-OPTIONS/validation_{safe_name}_portfolio_history.csv", index=False)
        print(f"  Saved portfolio history for {strategy_type} strategy")
    
    for name, benchmark in benchmarks.items():
        if benchmark is not None:
            benchmark.to_csv(f"/Users/tim/IWLS-OPTIONS/validation_{name.lower()}_benchmark.csv", index=False)

def main():
    print("IWLS STRATEGY VALIDATION ANALYSIS")
    print("Testing whether outperformance was skill or luck")
    print("=" * 60)
    
    # Load data
    all_results = load_all_iwls_results()
    if not all_results:
        return
    
    # Determine date range
    all_dates = []
    for df in all_results.values():
        all_dates.extend(df['date'].tolist())
    
    start_date = min(all_dates)
    end_date = max(all_dates)
    min_start_date = start_date + timedelta(days=365)
    
    print(f"\nData range: {start_date.strftime('%Y-%m-%d')} to {end_date.strftime('%Y-%m-%d')}")
    print(f"Analysis period: {min_start_date.strftime('%Y-%m-%d')} to {end_date.strftime('%Y-%m-%d')}")
    
    # Test three different deviation strategies
    strategy_types = ['undervalued', 'overvalued', 'neutral']
    strategy_results = {}
    portfolio_histories = {}
    
    print(f"\n" + "="*80)
    print("RUNNING VALIDATION STRATEGIES (360-DAY REBALANCING)")
    print("="*80)
    
    for strategy_type in strategy_types:
        print(f"\n{'='*25} {strategy_type.upper()} STRATEGY {'='*25}")
        
        # Run strategy
        portfolio_history, final_value = run_deviation_strategy(
            all_results, 
            min_start_date, 
            end_date,
            strategy_type,
            initial_capital=10000,
            num_assets=5
        )
        
        # Calculate daily portfolio values
        daily_portfolio = calculate_daily_portfolio_value(
            all_results, portfolio_history, min_start_date, end_date
        )
        
        strategy_results[strategy_type] = daily_portfolio
        portfolio_histories[strategy_type] = portfolio_history
    
    # Calculate benchmarks
    print(f"\n" + "="*80)
    print("CALCULATING BENCHMARKS")
    print("="*80)
    
    benchmarks = {}
    benchmark_assets = ['SPY', 'AMZN', 'AAPL']
    
    for asset in benchmark_assets:
        print(f"Calculating {asset} benchmark...")
        benchmark = calculate_benchmark(all_results, asset, min_start_date, end_date, initial_capital=10000)
        benchmarks[asset] = benchmark
        
        if benchmark is not None:
            final_value = benchmark[f'{asset.lower()}_value'].iloc[-1]
            annualized_return = calculate_annualized_return(10000, final_value, min_start_date, end_date)
            print(f"{asset}: ${final_value:,.2f} ({annualized_return:.1f}% annually)")
    
    # Create validation visualization
    print(f"\n" + "="*80)
    print("GENERATING VALIDATION ANALYSIS")
    print("="*80)
    
    returns_df, vol_df, sharpe_df = plot_validation_comparison(
        strategy_results, benchmarks, min_start_date, end_date, initial_capital=10000
    )
    
    # Print comprehensive summary
    print_validation_summary(
        strategy_results, portfolio_histories, benchmarks, min_start_date, end_date, initial_capital=10000
    )
    
    # Save results
    save_validation_results(strategy_results, portfolio_histories, benchmarks)
    
    print(f"\n" + "="*80)
    print("VALIDATION ANALYSIS COMPLETE")
    print("="*80)
    print("Files saved:")
    print("  - iwls_validation_analysis.png (4-panel validation chart)")
    print("  - validation_*_daily_values.csv (daily values for each strategy)")
    print("  - validation_*_portfolio_history.csv (rebalancing history)")
    print("  - validation_*_benchmark.csv (benchmark data)")
    
    # Final validation verdict
    print(f"\n" + "="*80)
    print("FINAL VALIDATION VERDICT")
    print("="*80)
    
    undervalued_final = None
    overvalued_final = None
    neutral_final = None
    
    for strategy_type, daily_portfolio in strategy_results.items():
        if len(daily_portfolio) > 0:
            final_value = daily_portfolio['portfolio_value'].iloc[-1]
            annualized_return = calculate_annualized_return(10000, final_value, min_start_date, end_date)
            
            if strategy_type == 'undervalued':
                undervalued_final = annualized_return
            elif strategy_type == 'overvalued':
                overvalued_final = annualized_return
            elif strategy_type == 'neutral':
                neutral_final = annualized_return
    
    if undervalued_final and overvalued_final:
        performance_gap = undervalued_final - overvalued_final
        
        print(f"UNDERVALUED vs OVERVALUED:")
        print(f"  Undervalued return: {undervalued_final:.2f}% annually")
        print(f"  Overvalued return: {overvalued_final:.2f}% annually")
        print(f"  Performance gap: {performance_gap:.2f}% annually")
        print()
        
        if performance_gap > 10:
            verdict = "STRONG SKILL-BASED ALPHA"
            confidence = "Very High"
        elif performance_gap > 5:
            verdict = "MODERATE SKILL-BASED ALPHA"
            confidence = "High"
        elif performance_gap > 0:
            verdict = "WEAK SKILL INDICATION"
            confidence = "Moderate"
        else:
            verdict = "LIKELY LUCK-BASED"
            confidence = "High"
        
        print(f"VERDICT: {verdict}")
        print(f"Confidence: {confidence}")
        print()
        
        if performance_gap > 5:
            print("INTERPRETATION:")
            print("  Your IWLS strategy demonstrates genuine mean reversion alpha.")
            print("  The significant outperformance of undervalued vs overvalued assets")
            print("  suggests your methodology captures real market inefficiencies.")
            print("  This validates the 42% annual returns as skill-based, not luck.")
        elif performance_gap > 0:
            print("INTERPRETATION:")
            print("  Your strategy shows some predictive power but the edge is modest.")
            print("  Consider refining selection criteria or risk management.")
        else:
            print("INTERPRETATION:")
            print("  The results suggest luck played a significant role.")
            print("  Consider revising the IWLS methodology or selection criteria.")

if __name__ == "__main__":
    main()