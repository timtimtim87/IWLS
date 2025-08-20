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
        print("Could not load corrected EV data. Using placeholder for EV strategies.")
        return None

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

# Strategy 1: IWLS 365-day Rebalancing (from asset_rebalance.py)
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

def run_iwls_365_strategy(all_results, start_date, end_date, initial_capital=10000):
    """
    Run the IWLS 365-day rebalancing strategy (winner from asset_rebalance)
    """
    print(f"\nRunning IWLS 365-day Rebalancing Strategy")
    
    # Generate annual rebalance dates
    rebalance_dates = []
    current_date = start_date
    while current_date <= end_date:
        rebalance_dates.append(current_date)
        current_date = current_date.replace(year=current_date.year + 1)
    
    portfolio = {}
    cash = initial_capital
    portfolio_history = []
    
    for i, rebalance_date in enumerate(rebalance_dates):
        print(f"  Rebalance {i+1}: {rebalance_date.strftime('%Y-%m-%d')}")
        
        # Sell existing positions
        if portfolio:
            total_sale_proceeds = 0
            for asset, shares in portfolio.items():
                current_price = get_asset_price(all_results, asset, rebalance_date)
                if current_price:
                    proceeds = shares * current_price
                    total_sale_proceeds += proceeds
            cash = total_sale_proceeds
            portfolio = {}
        
        # Get normalized deviations and select most undervalued
        normalized_devs = normalize_deviations_across_assets(all_results, rebalance_date)
        
        if len(normalized_devs) >= 5:
            sorted_assets = sorted(normalized_devs.items(), key=lambda x: x[1])
            selected_assets = sorted_assets[:5]  # Top 5 most undervalued
            
            allocation_per_asset = cash / 5
            
            for asset, norm_dev in selected_assets:
                current_price = get_asset_price(all_results, asset, rebalance_date)
                if current_price and current_price > 0:
                    shares = allocation_per_asset / current_price
                    portfolio[asset] = shares
        
        # Calculate portfolio value
        portfolio_value = 0
        for asset, shares in portfolio.items():
            current_price = get_asset_price(all_results, asset, rebalance_date)
            if current_price:
                portfolio_value += shares * current_price
        
        portfolio_history.append({
            'date': rebalance_date,
            'portfolio_value': portfolio_value,
            'assets': list(portfolio.keys())
        })
    
    # Calculate final value
    final_value = 0
    for asset, shares in portfolio.items():
        final_price = get_asset_price(all_results, asset, end_date)
        if final_price:
            final_value += shares * final_price
    
    return portfolio_history, final_value

# Strategy 2 & 3: EV Trading Groups 1 & 2 (from EV_trading_strat.py)
def calculate_current_z_scores(all_results, date, lookback_days=252):
    """
    Calculate current Z-scores for all assets on a given date
    """
    z_scores = {}
    
    for asset_name, df in all_results.items():
        asset_data = df[df['date'] <= date].copy()
        
        if len(asset_data) < lookback_days + 50:
            continue
        
        recent_data = asset_data.tail(lookback_days)
        
        if len(recent_data) < 50:
            continue
        
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
    if ev_df is None:
        # Fallback: use Z-scores directly as proxy for EV
        asset_evs = {}
        for asset, z_score in z_scores.items():
            # Convert Z-score to expected value proxy (more negative = higher EV)
            expected_value = max(0, (-z_score * 10))  # Simple conversion
            asset_evs[asset] = {
                'z_score': z_score,
                'expected_value': expected_value,
                'z_score_clamped': z_score
            }
        return asset_evs
    
    asset_evs = {}
    
    for asset, z_score in z_scores.items():
        if asset not in ev_df.columns:
            continue
        
        asset_ev_data = ev_df[['z_score', asset]].dropna()
        
        if len(asset_ev_data) < 3:
            continue
        
        z_values = asset_ev_data['z_score'].values
        ev_values = asset_ev_data[asset].values
        
        z_score_clamped = np.clip(z_score, z_values.min(), z_values.max())
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
    sorted_assets = sorted(asset_evs.items(), key=lambda x: x[1]['expected_value'], reverse=True)
    
    groups = []
    for i in range(0, len(sorted_assets), group_size):
        group = sorted_assets[i:i+group_size]
        if len(group) == group_size:
            groups.append(group)
    
    return groups

def run_ev_group_strategy(all_results, ev_df, start_date, end_date, group_num, initial_capital=10000):
    """
    Run EV Group strategy (Groups 1 & 2 from EV_trading_strat)
    """
    print(f"\nRunning EV Group {group_num + 1} Strategy")
    
    # Generate annual rebalance dates
    rebalance_dates = []
    current_date = start_date
    while current_date <= end_date:
        rebalance_dates.append(current_date)
        current_date = current_date.replace(year=current_date.year + 1)
    
    portfolio = {}
    cash = initial_capital
    portfolio_history = []
    
    for i, rebalance_date in enumerate(rebalance_dates):
        print(f"  Rebalance {i+1}: {rebalance_date.strftime('%Y-%m-%d')}")
        
        # Sell existing positions
        if portfolio:
            total_sale_proceeds = 0
            for asset, shares in portfolio.items():
                current_price = get_asset_price(all_results, asset, rebalance_date)
                if current_price:
                    proceeds = shares * current_price
                    total_sale_proceeds += proceeds
            cash = total_sale_proceeds
            portfolio = {}
        
        # Calculate Z-scores and select by EV
        current_z_scores = calculate_current_z_scores(all_results, rebalance_date)
        
        if len(current_z_scores) >= 4:
            asset_evs = get_expected_values_for_z_scores(ev_df, current_z_scores)
            
            if len(asset_evs) >= 4:
                groups = select_top_assets_by_ev(asset_evs, group_size=4)
                
                if group_num < len(groups):
                    selected_group = groups[group_num]
                    
                    allocation_per_asset = cash / len(selected_group)
                    
                    for asset_name, ev_data in selected_group:
                        current_price = get_asset_price(all_results, asset_name, rebalance_date)
                        if current_price and current_price > 0:
                            shares = allocation_per_asset / current_price
                            portfolio[asset_name] = shares
        
        # Calculate portfolio value
        portfolio_value = 0
        for asset, shares in portfolio.items():
            current_price = get_asset_price(all_results, asset, rebalance_date)
            if current_price:
                portfolio_value += shares * current_price
        
        portfolio_history.append({
            'date': rebalance_date,
            'portfolio_value': portfolio_value,
            'assets': list(portfolio.keys())
        })
    
    # Calculate final value
    final_value = 0
    for asset, shares in portfolio.items():
        final_price = get_asset_price(all_results, asset, end_date)
        if final_price:
            final_value += shares * final_price
    
    return portfolio_history, final_value

# Strategy 4: Absolute Deviation (Absolute) from multi_strat.py
def calculate_current_absolute_deviations(all_results, date):
    """
    Calculate current absolute deviations from IWLS growth line
    """
    abs_deviations = {}
    
    for asset_name, df in all_results.items():
        asset_data = df[df['date'] <= date].copy()
        
        if len(asset_data) < 50:
            continue
        
        current_deviation = asset_data['price_deviation'].iloc[-1]
        
        abs_deviations[asset_name] = {
            'raw_deviation': current_deviation,
            'abs_deviation': abs(current_deviation)
        }
    
    return abs_deviations

def select_assets_by_absolute_deviation(abs_deviations, num_assets=4):
    """
    Select assets by largest absolute deviation from IWLS growth line
    """
    sorted_assets = sorted(abs_deviations.items(), 
                         key=lambda x: x[1]['abs_deviation'], reverse=True)
    return sorted_assets[:num_assets]

def run_absolute_deviation_strategy(all_results, start_date, end_date, initial_capital=10000):
    """
    Run Absolute Deviation strategy (winner from multi_strat)
    """
    print(f"\nRunning Absolute Deviation (Absolute) Strategy")
    
    # Generate annual rebalance dates
    rebalance_dates = []
    current_date = start_date
    while current_date <= end_date:
        rebalance_dates.append(current_date)
        current_date = current_date.replace(year=current_date.year + 1)
    
    portfolio = {}
    cash = initial_capital
    portfolio_history = []
    
    for i, rebalance_date in enumerate(rebalance_dates):
        print(f"  Rebalance {i+1}: {rebalance_date.strftime('%Y-%m-%d')}")
        
        # Sell existing positions
        if portfolio:
            total_sale_proceeds = 0
            for asset, shares in portfolio.items():
                current_price = get_asset_price(all_results, asset, rebalance_date)
                if current_price:
                    proceeds = shares * current_price
                    total_sale_proceeds += proceeds
            cash = total_sale_proceeds
            portfolio = {}
        
        # Calculate absolute deviations and select assets
        abs_deviations = calculate_current_absolute_deviations(all_results, rebalance_date)
        
        if len(abs_deviations) >= 4:
            selected_assets = select_assets_by_absolute_deviation(abs_deviations, num_assets=4)
            
            allocation_per_asset = cash / len(selected_assets)
            
            for asset_name, dev_data in selected_assets:
                current_price = get_asset_price(all_results, asset_name, rebalance_date)
                if current_price and current_price > 0:
                    shares = allocation_per_asset / current_price
                    portfolio[asset_name] = shares
        
        # Calculate portfolio value
        portfolio_value = 0
        for asset, shares in portfolio.items():
            current_price = get_asset_price(all_results, asset, rebalance_date)
            if current_price:
                portfolio_value += shares * current_price
        
        portfolio_history.append({
            'date': rebalance_date,
            'portfolio_value': portfolio_value,
            'assets': list(portfolio.keys())
        })
    
    # Calculate final value
    final_value = 0
    for asset, shares in portfolio.items():
        final_price = get_asset_price(all_results, asset, end_date)
        if final_price:
            final_value += shares * final_price
    
    return portfolio_history, final_value

def calculate_daily_portfolio_values(all_results, portfolio_history, start_date, end_date):
    """
    Calculate daily portfolio values for a strategy
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

def create_winner_comparison_visualization(results_dict, benchmarks, start_date, end_date, output_dir):
    """
    Create comprehensive visualization comparing winning strategies
    """
    fig, ((ax1, ax2), (ax3, ax4)) = plt.subplots(2, 2, figsize=(20, 12))
    
    colors = ['blue', 'green', 'orange', 'red', 'purple']
    
    # Plot 1: Performance over time
    for i, (strategy_name, data) in enumerate(results_dict.items()):
        daily_values = data['daily_values']
        if len(daily_values) > 0:
            ax1.plot(daily_values['date'], daily_values['portfolio_value'], 
                    label=strategy_name, linewidth=3, color=colors[i % len(colors)])
    
    # Add benchmarks
    for i, (name, benchmark) in enumerate(benchmarks.items()):
        if benchmark is not None and len(benchmark) > 0:
            col_name = f'{name.lower()}_value'
            if col_name in benchmark.columns:
                ax1.plot(benchmark['date'], benchmark[col_name], 
                        label=f'{name} Buy & Hold', linewidth=2, 
                        color='gray', alpha=0.7, linestyle='--')
    
    ax1.axhline(y=10000, color='black', linestyle=':', alpha=0.7, label='Initial Capital')
    ax1.set_title('Best IWLS Strategies Comparison', fontsize=14, fontweight='bold')
    ax1.set_xlabel('Date')
    ax1.set_ylabel('Portfolio Value ($)')
    ax1.legend()
    ax1.grid(True, alpha=0.3)
    ax1.yaxis.set_major_formatter(plt.FuncFormatter(lambda x, p: f'${x:,.0f}'))
    
    # Plot 2: Annualized returns comparison
    strategy_names = list(results_dict.keys())
    annualized_returns = [data['annualized_return'] for data in results_dict.values()]
    
    bars = ax2.bar(range(len(strategy_names)), annualized_returns, 
                  color=colors[:len(strategy_names)], alpha=0.8)
    ax2.set_xticks(range(len(strategy_names)))
    ax2.set_xticklabels(strategy_names, rotation=45, ha='right')
    ax2.set_title('Annualized Returns Comparison', fontsize=14, fontweight='bold')
    ax2.set_ylabel('Annualized Return (%)')
    ax2.grid(True, alpha=0.3)
    
    for bar, value in zip(bars, annualized_returns):
        height = bar.get_height()
        ax2.text(bar.get_x() + bar.get_width()/2., height + height*0.01,
                f'{value:.1f}%', ha='center', va='bottom', fontweight='bold')
    
    # Plot 3: Volatility comparison
    volatility_data = []
    for strategy_name, data in results_dict.items():
        daily_values = data['daily_values']
        if len(daily_values) > 0:
            returns = daily_values['portfolio_value'].pct_change().dropna()
            volatility = returns.std() * np.sqrt(252) * 100
            volatility_data.append(volatility)
    
    bars = ax3.bar(range(len(strategy_names)), volatility_data, 
                  color=colors[:len(strategy_names)], alpha=0.8)
    ax3.set_xticks(range(len(strategy_names)))
    ax3.set_xticklabels(strategy_names, rotation=45, ha='right')
    ax3.set_title('Annualized Volatility Comparison', fontsize=14, fontweight='bold')
    ax3.set_ylabel('Volatility (%)')
    ax3.grid(True, alpha=0.3)
    
    for bar, value in zip(bars, volatility_data):
        height = bar.get_height()
        ax3.text(bar.get_x() + bar.get_width()/2., height + height*0.01,
                f'{value:.1f}%', ha='center', va='bottom', fontweight='bold')
    
    # Plot 4: Sharpe ratio comparison
    sharpe_data = []
    for strategy_name, data in results_dict.items():
        daily_values = data['daily_values']
        if len(daily_values) > 0:
            returns = daily_values['portfolio_value'].pct_change().dropna()
            if len(returns) > 0 and returns.std() > 0:
                sharpe = (returns.mean() / returns.std()) * np.sqrt(252)
                sharpe_data.append(sharpe)
            else:
                sharpe_data.append(0)
    
    bars = ax4.bar(range(len(strategy_names)), sharpe_data, 
                  color=colors[:len(strategy_names)], alpha=0.8)
    ax4.set_xticks(range(len(strategy_names)))
    ax4.set_xticklabels(strategy_names, rotation=45, ha='right')
    ax4.set_title('Sharpe Ratio Comparison', fontsize=14, fontweight='bold')
    ax4.set_ylabel('Sharpe Ratio')
    ax4.grid(True, alpha=0.3)
    
    for bar, value in zip(bars, sharpe_data):
        height = bar.get_height()
        ax4.text(bar.get_x() + bar.get_width()/2., height + height*0.01,
                f'{value:.2f}', ha='center', va='bottom', fontweight='bold')
    
    plt.tight_layout()
    plt.savefig(f"{output_dir}/best_strategies_comparison.png", dpi=300, bbox_inches='tight')
    plt.close()

def save_comparison_results(results_dict, output_dir):
    """
    Save comparison results
    """
    # Strategy summary
    summary_data = []
    for strategy_name, data in results_dict.items():
        summary_data.append({
            'strategy': strategy_name,
            'final_value': data['final_value'],
            'total_return': data['total_return'],
            'annualized_return': data['annualized_return']
        })
    
    summary_df = pd.DataFrame(summary_data)
    summary_df = summary_df.sort_values('annualized_return', ascending=False)
    summary_df.to_csv(f"{output_dir}/best_strategies_summary.csv", index=False)
    
    # Individual strategy details
    for strategy_name, data in results_dict.items():
        safe_name = strategy_name.replace(' ', '_').replace('(', '').replace(')', '')
        
        # Daily values
        data['daily_values'].to_csv(f"{output_dir}/{safe_name}_daily_values.csv", index=False)
        
        # Portfolio history
        portfolio_df = pd.DataFrame(data['portfolio_history'])
        portfolio_df.to_csv(f"{output_dir}/{safe_name}_portfolio_history.csv", index=False)
    
    return summary_df

def print_winner_summary(results_dict, summary_df):
    """
    Print comprehensive summary of winner comparison
    """
    print("\n" + "="*80)
    print("BEST IWLS STRATEGIES HEAD-TO-HEAD COMPARISON")
    print("="*80)
    
    print(f"\nSTRATEGIES TESTED:")
    print("1. IWLS 365-day Rebalancing (winner from asset_rebalance)")
    print("2. EV Group 1 (winner from EV_trading_strat)")
    print("3. EV Group 2 (runner-up from EV_trading_strat)")
    print("4. Absolute Deviation (winner from multi_strat)")
    
    print(f"\nPERFORMANCE RANKING:")
    print("-" * 60)
    print(f"{'Strategy':<30} {'Final Value':<12} {'Total Ret':<10} {'Annual Ret':<10}")
    print("-" * 60)
    
    for _, row in summary_df.iterrows():
        print(f"{row['strategy']:<30} ${row['final_value']:>10,.0f} "
              f"{row['total_return']:>8.1f}% {row['annualized_return']:>8.1f}%")
    
    # Analysis
    print(f"\nKEY FINDINGS:")
    print("-" * 40)
    
    best_strategy = summary_df.iloc[0]
    worst_strategy = summary_df.iloc[-1]
    
    print(f"Best performer: {best_strategy['strategy']}")
    print(f"  Annual return: {best_strategy['annualized_return']:.1f}%")
    print(f"  Final value: ${best_strategy['final_value']:,.0f}")
    
    print(f"\nWorst performer: {worst_strategy['strategy']}")
    print(f"  Annual return: {worst_strategy['annualized_return']:.1f}%")
    print(f"  Final value: ${worst_strategy['final_value']:,.0f}")
    
    performance_gap = best_strategy['annualized_return'] - worst_strategy['annualized_return']
    print(f"\nPerformance gap: {performance_gap:.1f}% annually")
    
    # Strategy insights
    print(f"\nSTRATEGY INSIGHTS:")
    print("-" * 40)
    
    if "IWLS 365" in best_strategy['strategy']:
        print("WINNER: Simple normalized deviation approach dominates")
        print("  Takeaway: Complex EV calculations may not add value")
    elif "EV Group" in best_strategy['strategy']:
        print("WINNER: Expected Value methodology proves superior")
        print("  Takeaway: Sophisticated prediction pays off")
    elif "Absolute" in best_strategy['strategy']:
        print("WINNER: Pure volatility capture beats directional bets")
        print("  Takeaway: Deviation magnitude matters more than direction")
    
    print(f"\nRECOMMENDATION:")
    print(f"Focus on {best_strategy['strategy']} for live implementation")
    print(f"Expected annual return: {best_strategy['annualized_return']:.1f}%")

def main():
    print("Best IWLS Strategies Head-to-Head Comparison")
    print("="*60)
    print("Testing the winning strategies from each analysis:")
    print("- IWLS 365-day (asset_rebalance winner)")
    print("- EV Group 1 & 2 (EV_trading_strat winners)")
    print("- Absolute Deviation (multi_strat winner)")
    
    # Create output directory
    output_dir = "/Users/tim/IWLS-OPTIONS/BEST_STRATEGIES_COMPARISON"
    os.makedirs(output_dir, exist_ok=True)
    print(f"\nOutput directory: {output_dir}")
    
    # Load data
    all_results = load_all_iwls_results()
    if not all_results:
        return
    
    ev_df = load_corrected_ev_data()
    
    # Determine date range
    all_dates = []
    for df in all_results.values():
        all_dates.extend(df['date'].tolist())
    
    start_date = min(all_dates)
    end_date = max(all_dates)
    min_start_date = start_date + timedelta(days=730)  # Need 2+ years of history
    
    print(f"\nData range: {start_date.strftime('%Y-%m-%d')} to {end_date.strftime('%Y-%m-%d')}")
    print(f"Strategy period: {min_start_date.strftime('%Y-%m-%d')} to {end_date.strftime('%Y-%m-%d')}")
    
    results_dict = {}
    
    # Strategy 1: IWLS 365-day Rebalancing
    print(f"\n{'='*20} IWLS 365-DAY REBALANCING {'='*20}")
    portfolio_history, final_value = run_iwls_365_strategy(
        all_results, min_start_date, end_date, initial_capital=10000
    )
    daily_values = calculate_daily_portfolio_values(all_results, portfolio_history, min_start_date, end_date)
    total_return = ((final_value / 10000) - 1) * 100
    annualized_return = calculate_annualized_return(10000, final_value, min_start_date, end_date)
    
    results_dict['IWLS 365-day'] = {
        'portfolio_history': portfolio_history,
        'daily_values': daily_values,
        'final_value': final_value,
        'total_return': total_return,
        'annualized_return': annualized_return
    }
    
    print(f"Results: ${final_value:,.2f} ({annualized_return:.1f}% annually)")
    
    # Strategy 2: EV Group 1
    print(f"\n{'='*20} EV GROUP 1 STRATEGY {'='*20}")
    portfolio_history, final_value = run_ev_group_strategy(
        all_results, ev_df, min_start_date, end_date, group_num=0, initial_capital=10000
    )
    daily_values = calculate_daily_portfolio_values(all_results, portfolio_history, min_start_date, end_date)
    total_return = ((final_value / 10000) - 1) * 100
    annualized_return = calculate_annualized_return(10000, final_value, min_start_date, end_date)
    
    results_dict['EV Group 1'] = {
        'portfolio_history': portfolio_history,
        'daily_values': daily_values,
        'final_value': final_value,
        'total_return': total_return,
        'annualized_return': annualized_return
    }
    
    print(f"Results: ${final_value:,.2f} ({annualized_return:.1f}% annually)")
    
    # Strategy 3: EV Group 2
    print(f"\n{'='*20} EV GROUP 2 STRATEGY {'='*20}")
    portfolio_history, final_value = run_ev_group_strategy(
        all_results, ev_df, min_start_date, end_date, group_num=1, initial_capital=10000
    )
    daily_values = calculate_daily_portfolio_values(all_results, portfolio_history, min_start_date, end_date)
    total_return = ((final_value / 10000) - 1) * 100
    annualized_return = calculate_annualized_return(10000, final_value, min_start_date, end_date)
    
    results_dict['EV Group 2'] = {
        'portfolio_history': portfolio_history,
        'daily_values': daily_values,
        'final_value': final_value,
        'total_return': total_return,
        'annualized_return': annualized_return
    }
    
    print(f"Results: ${final_value:,.2f} ({annualized_return:.1f}% annually)")
    
    # Strategy 4: Absolute Deviation
    print(f"\n{'='*20} ABSOLUTE DEVIATION STRATEGY {'='*20}")
    portfolio_history, final_value = run_absolute_deviation_strategy(
        all_results, min_start_date, end_date, initial_capital=10000
    )
    daily_values = calculate_daily_portfolio_values(all_results, portfolio_history, min_start_date, end_date)
    total_return = ((final_value / 10000) - 1) * 100
    annualized_return = calculate_annualized_return(10000, final_value, min_start_date, end_date)
    
    results_dict['Absolute Deviation'] = {
        'portfolio_history': portfolio_history,
        'daily_values': daily_values,
        'final_value': final_value,
        'total_return': total_return,
        'annualized_return': annualized_return
    }
    
    print(f"Results: ${final_value:,.2f} ({annualized_return:.1f}% annually)")
    
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
            annualized_return = calculate_annualized_return(10000, final_value, min_start_date, end_date)
            print(f"{asset}: ${final_value:,.2f} ({annualized_return:.1f}% annually)")
    
    # Create visualizations
    print(f"\nCreating comparison visualizations...")
    create_winner_comparison_visualization(results_dict, benchmarks, min_start_date, end_date, output_dir)
    
    # Save results
    summary_df = save_comparison_results(results_dict, output_dir)
    
    # Print comprehensive summary
    print_winner_summary(results_dict, summary_df)
    
    print(f"\n" + "="*80)
    print("BEST STRATEGIES COMPARISON COMPLETE")
    print("="*80)
    print("Files saved:")
    print("  - best_strategies_comparison.png (4-panel comparison)")
    print("  - best_strategies_summary.csv (performance summary)")
    print("  - *_daily_values.csv (daily portfolio values)")
    print("  - *_portfolio_history.csv (rebalancing decisions)")
    
    # Final recommendation
    if len(summary_df) > 0:
        best_strategy = summary_df.iloc[0]
        second_best = summary_df.iloc[1] if len(summary_df) > 1 else None
        
        print(f"\nFINAL RECOMMENDATION:")
        print(f"Primary Strategy: {best_strategy['strategy']}")
        print(f"  Expected Annual Return: {best_strategy['annualized_return']:.1f}%")
        print(f"  Final Value: ${best_strategy['final_value']:,.0f}")
        
        if second_best is not None:
            gap = best_strategy['annualized_return'] - second_best['annualized_return']
            print(f"\nSecond Choice: {second_best['strategy']}")
            print(f"  Expected Annual Return: {second_best['annualized_return']:.1f}%")
            print(f"  Performance Gap: {gap:.1f}% annually")
            
            if gap < 2:
                print(f"\n⚠️  CLOSE RACE: Consider diversifying between top 2 strategies")
            elif gap > 5:
                print(f"\n✅ CLEAR WINNER: Focus exclusively on {best_strategy['strategy']}")
            else:
                print(f"\n📊 MODERATE LEADER: {best_strategy['strategy']} preferred but monitor both")
        
        # Implementation guidance
        print(f"\nIMPLEMENTATION GUIDANCE:")
        print("-" * 30)
        
        if "IWLS 365" in best_strategy['strategy']:
            print("• Use normalized Z-score ranking across all assets")
            print("• Rebalance annually on fixed date")
            print("• Equal weight top 5 most undervalued assets")
            print("• Simple, robust, low maintenance approach")
            
        elif "EV Group" in best_strategy['strategy']:
            print("• Calculate Z-scores using 252-day rolling window")
            print("• Interpolate expected values from historical bins")
            print("• Select top 4 assets by expected value")
            print("• Requires EV lookup table maintenance")
            
        elif "Absolute" in best_strategy['strategy']:
            print("• Select assets with largest absolute deviations")
            print("• Direction-agnostic volatility capture")
            print("• Equal weight top 4 most deviated assets")
            print("• Simple calculation, contrarian approach")
        
        print(f"\nRISK CONSIDERATIONS:")
        print("• All strategies use annual rebalancing (low turnover)")
        print("• Concentration risk: 4-5 asset portfolios")
        print("• Mean reversion assumption may fail in trending markets")
        print("• Monitor strategy performance vs benchmarks quarterly")

if __name__ == "__main__":
    main()