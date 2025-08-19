import pandas as pd
import numpy as np
import matplotlib.pyplot as plt
import os
import glob
from datetime import datetime, timedelta
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

def load_corrected_ev_data():
    """
    Load the corrected expected values data
    """
    try:
        ev_df = pd.read_csv("/Users/tim/IWLS-OPTIONS/CORRECTED_EV_ANALYSIS/corrected_expected_values.csv")
        print(f"Loaded corrected EV data with {len(ev_df)} Z-score levels")
        return ev_df
    except:
        print("Could not load corrected EV data. Run true_ev.py first.")
        return None

def calculate_current_z_scores(all_results, date, lookback_days=252):
    """
    Calculate current Z-scores for all assets on a given date
    """
    z_scores = {}
    
    for asset_name, df in all_results.items():
        # Get data up to the specified date
        asset_data = df[df['date'] <= date].copy()
        
        if len(asset_data) < lookback_days + 50:  # Need enough history
            continue
        
        # Use rolling window for Z-score calculation
        recent_data = asset_data.tail(lookback_days)
        
        if len(recent_data) < 50:
            continue
        
        # Calculate Z-score for current deviation
        current_deviation = recent_data['price_deviation'].iloc[-1]
        historical_deviations = recent_data['price_deviation']
        
        mean_deviation = historical_deviations.mean()
        std_deviation = historical_deviations.std()
        
        if std_deviation > 0:
            z_score = (current_deviation - mean_deviation) / std_deviation
            z_scores[asset_name] = z_score
    
    return z_scores

def get_expected_values_for_z_scores(ev_df, z_scores):
    """
    Get expected values for current Z-scores using interpolation
    """
    ev_z_scores = ev_df['z_score'].values
    asset_evs = {}
    
    for asset, z_score in z_scores.items():
        if asset not in ev_df.columns:
            continue
        
        # Get valid EV data for this asset
        asset_ev_data = ev_df[['z_score', asset]].dropna()
        
        if len(asset_ev_data) < 3:  # Need at least 3 points for interpolation
            continue
        
        # Interpolate expected value for current Z-score
        z_values = asset_ev_data['z_score'].values
        ev_values = asset_ev_data[asset].values
        
        # Clamp Z-score to available range
        z_score_clamped = np.clip(z_score, z_values.min(), z_values.max())
        
        # Linear interpolation
        expected_value = np.interp(z_score_clamped, z_values, ev_values)
        
        asset_evs[asset] = {
            'z_score': z_score,
            'expected_value': expected_value,
            'z_score_clamped': z_score_clamped
        }
    
    return asset_evs

def select_top_assets_by_ev(asset_evs, group_size=4):
    """
    Select top assets by expected value and create groups
    """
    # Sort assets by expected value
    sorted_assets = sorted(asset_evs.items(), key=lambda x: x[1]['expected_value'], reverse=True)
    
    # Create groups
    groups = []
    for i in range(0, len(sorted_assets), group_size):
        group = sorted_assets[i:i+group_size]
        if len(group) == group_size:  # Only include complete groups
            groups.append(group)
    
    return groups

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

def run_ev_trading_strategy(all_results, ev_df, start_date, end_date, group_num, initial_capital=10000):
    """
    Run the EV-based trading strategy for a specific group
    """
    print(f"\nRunning EV Trading Strategy - Group {group_num + 1}")
    print(f"Initial capital: ${initial_capital:,.2f}")
    print(f"Period: {start_date.strftime('%Y-%m-%d')} to {end_date.strftime('%Y-%m-%d')}")
    
    # Generate annual rebalance dates
    rebalance_dates = []
    current_date = start_date
    while current_date <= end_date:
        rebalance_dates.append(current_date)
        current_date = current_date.replace(year=current_date.year + 1)
    
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
        
        # Calculate current Z-scores
        current_z_scores = calculate_current_z_scores(all_results, rebalance_date)
        
        if len(current_z_scores) < 4:
            print(f"Insufficient Z-score data ({len(current_z_scores)} assets available)")
            continue
        
        # Get expected values for current Z-scores
        asset_evs = get_expected_values_for_z_scores(ev_df, current_z_scores)
        
        if len(asset_evs) < 4:
            print(f"Insufficient EV data ({len(asset_evs)} assets available)")
            continue
        
        # Create groups and select the specified group
        groups = select_top_assets_by_ev(asset_evs, group_size=4)
        
        if group_num >= len(groups):
            print(f"Group {group_num + 1} not available (only {len(groups)} groups)")
            continue
        
        selected_group = groups[group_num]
        selected_assets = [asset_name for asset_name, _ in selected_group]
        
        print(f"Selected assets (Group {group_num + 1}):")
        for asset_name, ev_data in selected_group:
            print(f"  {asset_name}: Z={ev_data['z_score']:.2f}, EV={ev_data['expected_value']:.1f}%")
        
        # Buy equal amounts of each selected asset
        allocation_per_asset = cash / len(selected_assets)
        
        print(f"\nBuying positions (${allocation_per_asset:.2f} each):")
        for asset_name in selected_assets:
            current_price = get_asset_price(all_results, asset_name, rebalance_date)
            if current_price and current_price > 0:
                shares = allocation_per_asset / current_price
                portfolio[asset_name] = shares
                print(f"  {asset_name}: {shares:.2f} shares @ ${current_price:.2f}")
            else:
                print(f"  {asset_name}: No price data available")
        
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
            'assets': list(portfolio.keys()),
            'group_num': group_num + 1,
            'asset_evs': {asset: ev_data for asset, ev_data in selected_group}
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
    years = (end_date - start_date).days / 365.25
    annualized_return = ((final_portfolio_value / initial_capital) ** (1 / years) - 1) * 100
    
    print(f"\nGroup {group_num + 1} Strategy Results:")
    print(f"Initial capital: ${initial_capital:.2f}")
    print(f"Final portfolio value: ${final_portfolio_value:.2f}")
    print(f"Total return: {total_return:.2f}%")
    print(f"Annualized return: {annualized_return:.2f}%")
    
    return portfolio_history, final_portfolio_value, annualized_return

def calculate_daily_portfolio_value(all_results, portfolio_history, start_date, end_date):
    """
    Calculate daily portfolio values throughout the strategy period
    """
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

def create_group_comparison_visualization(group_results, benchmarks, start_date, end_date, output_dir):
    """
    Create comprehensive visualization comparing all groups
    """
    plt.figure(figsize=(20, 12))
    
    fig, ((ax1, ax2), (ax3, ax4)) = plt.subplots(2, 2, figsize=(20, 12))
    
    # Plot 1: Performance over time
    colors = ['blue', 'green', 'orange', 'red', 'purple', 'brown']
    
    for i, (group_num, data) in enumerate(group_results.items()):
        daily_values = data['daily_values']
        if len(daily_values) > 0:
            ax1.plot(daily_values['date'], daily_values['portfolio_value'], 
                    label=f'Group {group_num + 1}', linewidth=2, color=colors[i % len(colors)])
    
    # Add benchmarks
    benchmark_colors = ['black', 'gray']
    for i, (name, benchmark) in enumerate(benchmarks.items()):
        if benchmark is not None and len(benchmark) > 0:
            col_name = f'{name.lower()}_value'
            if col_name in benchmark.columns:
                ax1.plot(benchmark['date'], benchmark[col_name], 
                        label=f'{name} Buy & Hold', linewidth=2, 
                        color=benchmark_colors[i], alpha=0.7, linestyle='--')
    
    ax1.axhline(y=10000, color='gray', linestyle=':', alpha=0.7, label='Initial Capital')
    ax1.set_title('EV Trading Strategy: Group Performance Comparison', fontsize=14, fontweight='bold')
    ax1.set_xlabel('Date')
    ax1.set_ylabel('Portfolio Value ($)')
    ax1.legend()
    ax1.grid(True, alpha=0.3)
    ax1.yaxis.set_major_formatter(plt.FuncFormatter(lambda x, p: f'${x:,.0f}'))
    
    # Plot 2: Annualized returns comparison
    group_numbers = []
    annualized_returns = []
    
    for group_num, data in group_results.items():
        group_numbers.append(f'Group {group_num + 1}')
        annualized_returns.append(data['annualized_return'])
    
    bars = ax2.bar(group_numbers, annualized_returns, 
                  color=colors[:len(group_numbers)], alpha=0.7)
    ax2.set_title('Annualized Returns by Group', fontsize=14, fontweight='bold')
    ax2.set_ylabel('Annualized Return (%)')
    ax2.grid(True, alpha=0.3)
    
    # Add value labels on bars
    for bar, value in zip(bars, annualized_returns):
        height = bar.get_height()
        ax2.text(bar.get_x() + bar.get_width()/2., height + height*0.01,
                f'{value:.1f}%', ha='center', va='bottom', fontweight='bold')
    
    # Plot 3: Final values comparison
    final_values = []
    for group_num, data in group_results.items():
        final_values.append(data['final_value'])
    
    bars = ax3.bar(group_numbers, final_values, 
                  color=colors[:len(group_numbers)], alpha=0.7)
    ax3.set_title('Final Portfolio Values', fontsize=14, fontweight='bold')
    ax3.set_ylabel('Final Value ($)')
    ax3.grid(True, alpha=0.3)
    ax3.yaxis.set_major_formatter(plt.FuncFormatter(lambda x, p: f'${x:,.0f}'))
    
    # Add value labels on bars
    for bar, value in zip(bars, final_values):
        height = bar.get_height()
        ax3.text(bar.get_x() + bar.get_width()/2., height + height*0.01,
                f'${value:,.0f}', ha='center', va='bottom', fontweight='bold')
    
    # Plot 4: Volatility analysis
    volatility_data = []
    for group_num, data in group_results.items():
        daily_values = data['daily_values']
        if len(daily_values) > 0:
            returns = daily_values['portfolio_value'].pct_change().dropna()
            volatility = returns.std() * np.sqrt(252) * 100  # Annualized volatility
            volatility_data.append(volatility)
        else:
            volatility_data.append(0)
    
    bars = ax4.bar(group_numbers, volatility_data, 
                  color=colors[:len(group_numbers)], alpha=0.7)
    ax4.set_title('Annualized Volatility', fontsize=14, fontweight='bold')
    ax4.set_ylabel('Volatility (%)')
    ax4.grid(True, alpha=0.3)
    
    # Add value labels on bars
    for bar, value in zip(bars, volatility_data):
        height = bar.get_height()
        ax4.text(bar.get_x() + bar.get_width()/2., height + height*0.01,
                f'{value:.1f}%', ha='center', va='bottom', fontweight='bold')
    
    plt.tight_layout()
    plt.savefig(f"{output_dir}/ev_trading_strategy_comparison.png", dpi=300, bbox_inches='tight')
    plt.close()

def save_strategy_results(group_results, output_dir):
    """
    Save all strategy results to CSV files
    """
    # Save individual group results
    for group_num, data in group_results.items():
        # Daily values
        daily_values = data['daily_values']
        daily_values.to_csv(f"{output_dir}/group_{group_num + 1}_daily_values.csv", index=False)
        
        # Portfolio history
        portfolio_history = data['portfolio_history']
        portfolio_df = pd.DataFrame(portfolio_history)
        portfolio_df.to_csv(f"{output_dir}/group_{group_num + 1}_portfolio_history.csv", index=False)
    
    # Create summary table
    summary_data = []
    for group_num, data in group_results.items():
        summary_data.append({
            'group': group_num + 1,
            'final_value': data['final_value'],
            'annualized_return': data['annualized_return'],
            'total_return': ((data['final_value'] / 10000) - 1) * 100
        })
    
    summary_df = pd.DataFrame(summary_data)
    summary_df.to_csv(f"{output_dir}/strategy_summary.csv", index=False)
    
    return summary_df

def print_strategy_summary(group_results, summary_df):
    """
    Print comprehensive summary of strategy results
    """
    print("\n" + "="*80)
    print("EV TRADING STRATEGY RESULTS SUMMARY")
    print("="*80)
    
    print(f"\nSTRATEGY OVERVIEW:")
    print(f"  - 6 groups of 4 assets each, ranked by Expected Value")
    print(f"  - Annual rebalancing based on current Z-scores")
    print(f"  - Equal weight allocation within each group")
    print(f"  - $10,000 initial capital per group")
    
    print(f"\nGROUP PERFORMANCE RANKING:")
    print("-" * 60)
    print(f"{'Group':<8} {'Final Value':<12} {'Total Return':<14} {'Annual Return':<14}")
    print("-" * 60)
    
    sorted_summary = summary_df.sort_values('annualized_return', ascending=False)
    for _, row in sorted_summary.iterrows():
        print(f"Group {int(row['group']):<4} ${row['final_value']:>10,.0f} "
              f"{row['total_return']:>12.1f}% {row['annualized_return']:>12.1f}%")
    
    # Statistical analysis
    returns = summary_df['annualized_return'].values
    
    print(f"\nSTATISTICAL ANALYSIS:")
    print("-" * 40)
    print(f"Best performing group: Group {int(sorted_summary.iloc[0]['group'])} ({returns.max():.1f}% annually)")
    print(f"Worst performing group: Group {int(sorted_summary.iloc[-1]['group'])} ({returns.min():.1f}% annually)")
    print(f"Average return across groups: {returns.mean():.1f}%")
    print(f"Return spread: {returns.max() - returns.min():.1f}%")
    print(f"Standard deviation: {returns.std():.1f}%")
    
    # Strategy validation
    print(f"\nSTRATEGY VALIDATION:")
    print("-" * 40)
    positive_groups = len(summary_df[summary_df['annualized_return'] > 0])
    print(f"Groups with positive returns: {positive_groups}/6")
    
    if returns.max() > 10:
        print(f"STRONG SIGNAL: Best group achieved {returns.max():.1f}% annually")
        print(f"Expected Value methodology shows promise")
    elif returns.max() > 5:
        print(f"MODERATE SIGNAL: Best group achieved {returns.max():.1f}% annually")
        print(f"Expected Value methodology shows some merit")
    else:
        print(f"WEAK SIGNAL: Best group only achieved {returns.max():.1f}% annually")
        print(f"Expected Value methodology may need refinement")
    
    # Group analysis
    print(f"\nGROUP ANALYSIS INSIGHTS:")
    print("-" * 40)
    
    best_group = int(sorted_summary.iloc[0]['group'])
    worst_group = int(sorted_summary.iloc[-1]['group'])
    
    if best_group == 1:
        print(f"TOP TIER DOMINANCE: Group 1 (highest EV assets) performed best")
        print(f"This validates the Expected Value ranking methodology")
    elif best_group <= 3:
        print(f"UPPER TIER SUCCESS: Group {best_group} performed best")
        print(f"Higher EV assets generally outperformed")
    else:
        print(f"UNEXPECTED RESULT: Group {best_group} performed best")
        print(f"This suggests EV methodology may have limitations")
    
    if worst_group == 6:
        print(f"BOTTOM TIER WEAKNESS: Group 6 (lowest EV assets) performed worst")
        print(f"This supports the Expected Value ranking")
    else:
        print(f"MIXED RESULTS: Group {worst_group} performed worst")
        print(f"EV ranking doesn't perfectly predict performance")

def main():
    print("Expected Value Trading Strategy Backtesting")
    print("="*60)
    print("Testing 6 groups of 4 assets ranked by Expected Value")
    
    # Create output directory
    output_dir = "/Users/tim/IWLS-OPTIONS/EV_TRADING_STRATEGY"
    os.makedirs(output_dir, exist_ok=True)
    print(f"\nOutput directory: {output_dir}")
    
    # Load data
    all_results = load_all_iwls_results()
    if not all_results:
        return
    
    ev_df = load_corrected_ev_data()
    if ev_df is None:
        return
    
    # Determine date range
    all_dates = []
    for df in all_results.values():
        all_dates.extend(df['date'].tolist())
    
    start_date = min(all_dates)
    end_date = max(all_dates)
    
    # Ensure we have enough data for strategy (need 2+ years of history for first Z-score calc)
    min_start_date = start_date + timedelta(days=730)
    
    print(f"\nData range: {start_date.strftime('%Y-%m-%d')} to {end_date.strftime('%Y-%m-%d')}")
    print(f"Strategy period: {min_start_date.strftime('%Y-%m-%d')} to {end_date.strftime('%Y-%m-%d')}")
    
    # Run strategy for all 6 groups
    group_results = {}
    
    for group_num in range(6):  # Groups 0-5 (displayed as 1-6)
        print(f"\n{'='*20} TESTING GROUP {group_num + 1} {'='*20}")
        
        portfolio_history, final_value, annualized_return = run_ev_trading_strategy(
            all_results, ev_df, min_start_date, end_date, group_num, initial_capital=10000
        )
        
        # Calculate daily portfolio values
        daily_values = calculate_daily_portfolio_value(
            all_results, portfolio_history, min_start_date, end_date
        )
        
        group_results[group_num] = {
            'portfolio_history': portfolio_history,
            'daily_values': daily_values,
            'final_value': final_value,
            'annualized_return': annualized_return
        }
    
    # Calculate benchmarks
    print(f"\n{'='*20} CALCULATING BENCHMARKS {'='*20}")
    benchmarks = {}
    benchmark_assets = ['SPY', 'QQQ']
    
    for asset in benchmark_assets:
        print(f"Calculating {asset} benchmark...")
        benchmark = calculate_benchmark(all_results, asset, min_start_date, end_date, initial_capital=10000)
        benchmarks[asset] = benchmark
        
        if benchmark is not None:
            final_value = benchmark[f'{asset.lower()}_value'].iloc[-1]
            years = (end_date - min_start_date).days / 365.25
            annualized_return = ((final_value / 10000) ** (1 / years) - 1) * 100
            print(f"{asset}: ${final_value:,.2f} ({annualized_return:.1f}% annually)")
    
    # Create visualizations
    print(f"\nCreating strategy comparison visualizations...")
    create_group_comparison_visualization(group_results, benchmarks, min_start_date, end_date, output_dir)
    
    # Save results
    summary_df = save_strategy_results(group_results, output_dir)
    
    # Print comprehensive summary
    print_strategy_summary(group_results, summary_df)
    
    print(f"\n" + "="*80)
    print("EV TRADING STRATEGY ANALYSIS COMPLETE")
    print("="*80)
    print("Files saved:")
    print("  - ev_trading_strategy_comparison.png (comprehensive comparison)")
    print("  - group_X_daily_values.csv (daily portfolio values)")
    print("  - group_X_portfolio_history.csv (rebalancing history)")
    print("  - strategy_summary.csv (performance summary)")

if __name__ == "__main__":
    main()