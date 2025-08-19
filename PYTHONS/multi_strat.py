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

def calculate_current_absolute_deviations(all_results, date):
    """
    Calculate current absolute deviations from IWLS growth line
    """
    abs_deviations = {}
    
    for asset_name, df in all_results.items():
        asset_data = df[df['date'] <= date].copy()
        
        if len(asset_data) < 50:
            continue
        
        # Get current deviation (negative means undervalued)
        current_deviation = asset_data['price_deviation'].iloc[-1]
        
        # Store absolute deviation (larger absolute value = further from trend)
        abs_deviations[asset_name] = {
            'raw_deviation': current_deviation,
            'abs_deviation': abs(current_deviation)
        }
    
    return abs_deviations

def get_expected_values_for_z_scores(ev_df, z_scores):
    """
    Get expected values for current Z-scores using interpolation
    """
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

def select_assets_by_ev(asset_evs, num_assets=4):
    """
    Select top assets by expected value
    """
    sorted_assets = sorted(asset_evs.items(), key=lambda x: x[1]['expected_value'], reverse=True)
    return sorted_assets[:num_assets]

def select_assets_by_absolute_deviation(abs_deviations, num_assets=4, prefer_undervalued=True):
    """
    Select assets by largest absolute deviation from IWLS growth line
    
    prefer_undervalued: If True, prioritize negative deviations (undervalued assets)
                       If False, just select largest absolute deviations regardless of direction
    """
    if prefer_undervalued:
        # Sort by raw deviation (most negative first), then by absolute deviation
        sorted_assets = sorted(abs_deviations.items(), 
                             key=lambda x: (x[1]['raw_deviation'], -x[1]['abs_deviation']))
    else:
        # Sort purely by absolute deviation (largest deviation from trend)
        sorted_assets = sorted(abs_deviations.items(), 
                             key=lambda x: x[1]['abs_deviation'], reverse=True)
    
    return sorted_assets[:num_assets]

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

def run_ev_strategy(all_results, ev_df, start_date, end_date, initial_capital=10000):
    """
    Run the Expected Value strategy (annual rebalancing)
    """
    print(f"\nRunning Expected Value Strategy")
    print(f"Initial capital: ${initial_capital:,.2f}")
    print(f"Period: {start_date.strftime('%Y-%m-%d')} to {end_date.strftime('%Y-%m-%d')}")
    
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
        print(f"\n--- EV Rebalance {i+1}: {rebalance_date.strftime('%Y-%m-%d')} ---")
        
        # Sell existing positions
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
        
        # Calculate current Z-scores and select by EV
        current_z_scores = calculate_current_z_scores(all_results, rebalance_date)
        
        if len(current_z_scores) >= 4:
            asset_evs = get_expected_values_for_z_scores(ev_df, current_z_scores)
            
            if len(asset_evs) >= 4:
                selected_assets = select_assets_by_ev(asset_evs, num_assets=4)
                
                print(f"Selected assets by Expected Value:")
                for asset_name, ev_data in selected_assets:
                    print(f"  {asset_name}: Z={ev_data['z_score']:.2f}, EV={ev_data['expected_value']:.1f}%")
                
                # Buy equal amounts of each selected asset
                allocation_per_asset = cash / len(selected_assets)
                
                print(f"\nBuying positions (${allocation_per_asset:.2f} each):")
                for asset_name, ev_data in selected_assets:
                    current_price = get_asset_price(all_results, asset_name, rebalance_date)
                    if current_price and current_price > 0:
                        shares = allocation_per_asset / current_price
                        portfolio[asset_name] = shares
                        print(f"  {asset_name}: {shares:.2f} shares @ ${current_price:.2f}")
        
        # Calculate portfolio value
        portfolio_value = 0
        for asset, shares in portfolio.items():
            current_price = get_asset_price(all_results, asset, rebalance_date)
            if current_price:
                portfolio_value += shares * current_price
        
        portfolio_history.append({
            'date': rebalance_date,
            'rebalance_num': i + 1,
            'portfolio_value': portfolio_value,
            'assets': list(portfolio.keys()),
            'strategy': 'Expected_Value'
        })
    
    # Calculate final value
    final_value = 0
    for asset, shares in portfolio.items():
        final_price = get_asset_price(all_results, asset, end_date)
        if final_price:
            final_value += shares * final_price
    
    total_return = ((final_value / initial_capital) - 1) * 100
    years = (end_date - start_date).days / 365.25
    annualized_return = ((final_value / initial_capital) ** (1 / years) - 1) * 100
    
    print(f"\nEV Strategy Results:")
    print(f"Final value: ${final_value:.2f}")
    print(f"Total return: {total_return:.2f}%")
    print(f"Annualized return: {annualized_return:.2f}%")
    
    return {
        'portfolio_history': portfolio_history,
        'final_value': final_value,
        'annualized_return': annualized_return,
        'total_return': total_return,
        'strategy_name': 'Expected Value'
    }

def run_absolute_deviation_strategy(all_results, start_date, end_date, prefer_undervalued=True, initial_capital=10000):
    """
    Run the Absolute Deviation strategy (annual rebalancing)
    """
    strategy_type = "Undervalued" if prefer_undervalued else "Absolute"
    print(f"\nRunning Absolute Deviation Strategy ({strategy_type})")
    print(f"Initial capital: ${initial_capital:,.2f}")
    print(f"Period: {start_date.strftime('%Y-%m-%d')} to {end_date.strftime('%Y-%m-%d')}")
    
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
        print(f"\n--- {strategy_type} Rebalance {i+1}: {rebalance_date.strftime('%Y-%m-%d')} ---")
        
        # Sell existing positions
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
        
        # Calculate current absolute deviations and select assets
        abs_deviations = calculate_current_absolute_deviations(all_results, rebalance_date)
        
        if len(abs_deviations) >= 4:
            selected_assets = select_assets_by_absolute_deviation(abs_deviations, num_assets=4, prefer_undervalued=prefer_undervalued)
            
            print(f"Selected assets by {strategy_type} Deviation:")
            for asset_name, dev_data in selected_assets:
                print(f"  {asset_name}: Raw={dev_data['raw_deviation']:.1f}%, Abs={dev_data['abs_deviation']:.1f}%")
            
            # Buy equal amounts of each selected asset
            allocation_per_asset = cash / len(selected_assets)
            
            print(f"\nBuying positions (${allocation_per_asset:.2f} each):")
            for asset_name, dev_data in selected_assets:
                current_price = get_asset_price(all_results, asset_name, rebalance_date)
                if current_price and current_price > 0:
                    shares = allocation_per_asset / current_price
                    portfolio[asset_name] = shares
                    print(f"  {asset_name}: {shares:.2f} shares @ ${current_price:.2f}")
        
        # Calculate portfolio value
        portfolio_value = 0
        for asset, shares in portfolio.items():
            current_price = get_asset_price(all_results, asset, rebalance_date)
            if current_price:
                portfolio_value += shares * current_price
        
        portfolio_history.append({
            'date': rebalance_date,
            'rebalance_num': i + 1,
            'portfolio_value': portfolio_value,
            'assets': list(portfolio.keys()),
            'strategy': f'Absolute_{strategy_type}'
        })
    
    # Calculate final value
    final_value = 0
    for asset, shares in portfolio.items():
        final_price = get_asset_price(all_results, asset, end_date)
        if final_price:
            final_value += shares * final_price
    
    total_return = ((final_value / initial_capital) - 1) * 100
    years = (end_date - start_date).days / 365.25
    annualized_return = ((final_value / initial_capital) ** (1 / years) - 1) * 100
    
    print(f"\n{strategy_type} Strategy Results:")
    print(f"Final value: ${final_value:.2f}")
    print(f"Total return: {total_return:.2f}%")
    print(f"Annualized return: {annualized_return:.2f}%")
    
    return {
        'portfolio_history': portfolio_history,
        'final_value': final_value,
        'annualized_return': annualized_return,
        'total_return': total_return,
        'strategy_name': f'Absolute Deviation ({strategy_type})'
    }

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

def analyze_asset_selection_overlap(results_dict):
    """
    Analyze how often different strategies select the same assets
    """
    print("\n" + "="*60)
    print("ASSET SELECTION OVERLAP ANALYSIS")
    print("="*60)
    
    # Collect all selected assets by strategy and period
    strategy_selections = {}
    
    for strategy_name, data in results_dict.items():
        if 'portfolio_history' in data:
            strategy_selections[strategy_name] = {}
            for period in data['portfolio_history']:
                rebalance_num = period['rebalance_num']
                strategy_selections[strategy_name][rebalance_num] = set(period['assets'])
    
    # Calculate overlap between strategies
    strategy_names = list(strategy_selections.keys())
    
    print(f"\nAsset selection by period:")
    print("-" * 40)
    
    # Get all rebalance periods
    all_periods = set()
    for selections in strategy_selections.values():
        all_periods.update(selections.keys())
    
    for period in sorted(all_periods):
        print(f"\nRebalance Period {period}:")
        for strategy_name in strategy_names:
            if period in strategy_selections[strategy_name]:
                assets = sorted(list(strategy_selections[strategy_name][period]))
                print(f"  {strategy_name}: {', '.join(assets)}")
    
    # Calculate overall overlap statistics
    print(f"\nOVERALL OVERLAP STATISTICS:")
    print("-" * 40)
    
    for i, strategy1 in enumerate(strategy_names):
        for j, strategy2 in enumerate(strategy_names):
            if i < j:  # Avoid duplicate comparisons
                overlaps = []
                for period in all_periods:
                    if period in strategy_selections[strategy1] and period in strategy_selections[strategy2]:
                        set1 = strategy_selections[strategy1][period]
                        set2 = strategy_selections[strategy2][period]
                        overlap = len(set1.intersection(set2))
                        overlaps.append(overlap)
                
                if overlaps:
                    avg_overlap = np.mean(overlaps)
                    max_overlap = max(overlaps)
                    print(f"{strategy1} vs {strategy2}:")
                    print(f"  Average overlap: {avg_overlap:.1f}/4 assets ({avg_overlap/4*100:.1f}%)")
                    print(f"  Maximum overlap: {max_overlap}/4 assets")
    
    return strategy_selections

def create_strategy_comparison_visualization(results_dict, benchmarks, start_date, end_date, output_dir):
    """
    Create comprehensive visualization comparing EV vs Absolute Deviation strategies
    """
    fig, ((ax1, ax2), (ax3, ax4)) = plt.subplots(2, 2, figsize=(20, 12))
    
    colors = ['blue', 'red', 'green', 'orange', 'purple']
    
    # Plot 1: Performance over time
    for i, (strategy_name, data) in enumerate(results_dict.items()):
        if 'daily_values' in data:
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
    ax1.set_title('EV vs Absolute Deviation Strategy Comparison', fontsize=14, fontweight='bold')
    ax1.set_xlabel('Date')
    ax1.set_ylabel('Portfolio Value ($)')
    ax1.legend()
    ax1.grid(True, alpha=0.3)
    ax1.yaxis.set_major_formatter(plt.FuncFormatter(lambda x, p: f'${x:,.0f}'))
    
    # Plot 2: Annualized returns comparison
    strategy_names = [data['strategy_name'] for data in results_dict.values()]
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
    
    # Plot 3: Final values comparison
    final_values = [data['final_value'] for data in results_dict.values()]
    
    bars = ax3.bar(range(len(strategy_names)), final_values, 
                  color=colors[:len(strategy_names)], alpha=0.8)
    ax3.set_xticks(range(len(strategy_names)))
    ax3.set_xticklabels(strategy_names, rotation=45, ha='right')
    ax3.set_title('Final Portfolio Values', fontsize=14, fontweight='bold')
    ax3.set_ylabel('Final Value ($)')
    ax3.grid(True, alpha=0.3)
    ax3.yaxis.set_major_formatter(plt.FuncFormatter(lambda x, p: f'${x:,.0f}'))
    
    for bar, value in zip(bars, final_values):
        height = bar.get_height()
        ax3.text(bar.get_x() + bar.get_width()/2., height + height*0.01,
                f'${value:,.0f}', ha='center', va='bottom', fontweight='bold')
    
    # Plot 4: Return difference from EV strategy
    ev_return = None
    for data in results_dict.values():
        if 'Expected Value' in data['strategy_name']:
            ev_return = data['annualized_return']
            break
    
    if ev_return is not None:
        return_differences = []
        strategy_labels = []
        
        for data in results_dict.values():
            if 'Expected Value' not in data['strategy_name']:
                diff = data['annualized_return'] - ev_return
                return_differences.append(diff)
                strategy_labels.append(data['strategy_name'])
        
        if return_differences:
            bar_colors = ['green' if x > 0 else 'red' for x in return_differences]
            bars = ax4.bar(range(len(strategy_labels)), return_differences, 
                          color=bar_colors, alpha=0.8)
            ax4.set_xticks(range(len(strategy_labels)))
            ax4.set_xticklabels(strategy_labels, rotation=45, ha='right')
            ax4.set_title('Return Difference vs Expected Value Strategy', fontsize=14, fontweight='bold')
            ax4.set_ylabel('Return Difference (%)')
            ax4.axhline(y=0, color='black', linestyle='-', alpha=0.5)
            ax4.grid(True, alpha=0.3)
            
            for bar, value in zip(bars, return_differences):
                height = bar.get_height()
                ax4.text(bar.get_x() + bar.get_width()/2., 
                        height + (0.1 if height >= 0 else -0.2),
                        f'{value:+.1f}%', ha='center', 
                        va='bottom' if height >= 0 else 'top', fontweight='bold')
    
    plt.tight_layout()
    plt.savefig(f"{output_dir}/ev_vs_absolute_deviation_comparison.png", dpi=300, bbox_inches='tight')
    plt.close()

def save_comparison_results(results_dict, strategy_selections, output_dir):
    """
    Save all comparison results
    """
    # Strategy summary
    summary_data = []
    for data in results_dict.values():
        summary_data.append({
            'strategy': data['strategy_name'],
            'final_value': data['final_value'],
            'total_return': data['total_return'],
            'annualized_return': data['annualized_return']
        })
    
    summary_df = pd.DataFrame(summary_data)
    summary_df.to_csv(f"{output_dir}/strategy_comparison_summary.csv", index=False)
    
    # Individual strategy details
    for strategy_key, data in results_dict.items():
        safe_name = data['strategy_name'].replace(' ', '_').replace('(', '').replace(')', '')
        
        # Daily values
        if 'daily_values' in data:
            data['daily_values'].to_csv(f"{output_dir}/{safe_name}_daily_values.csv", index=False)
        
        # Portfolio history
        portfolio_df = pd.DataFrame(data['portfolio_history'])
        portfolio_df.to_csv(f"{output_dir}/{safe_name}_portfolio_history.csv", index=False)
    
    # Asset selection analysis
    selection_analysis = []
    for strategy_name, periods in strategy_selections.items():
        for period, assets in periods.items():
            for asset in assets:
                selection_analysis.append({
                    'strategy': strategy_name,
                    'rebalance_period': period,
                    'selected_asset': asset
                })
    
    if selection_analysis:
        selection_df = pd.DataFrame(selection_analysis)
        selection_df.to_csv(f"{output_dir}/asset_selection_analysis.csv", index=False)
    
    return summary_df

def print_comparison_summary(results_dict, summary_df):
    """
    Print comprehensive summary of strategy comparison
    """
    print("\n" + "="*80)
    print("EXPECTED VALUE vs ABSOLUTE DEVIATION STRATEGY COMPARISON")
    print("="*80)
    
    print(f"\nSTRATEGY PERFORMANCE RANKING:")
    print("-" * 60)
    print(f"{'Strategy':<35} {'Final Value':<12} {'Total Ret':<10} {'Annual Ret':<10}")
    print("-" * 60)
    
    sorted_summary = summary_df.sort_values('annualized_return', ascending=False)
    for _, row in sorted_summary.iterrows():
        print(f"{row['strategy']:<35} ${row['final_value']:>10,.0f} "
              f"{row['total_return']:>8.1f}% {row['annualized_return']:>8.1f}%")
    
    # Analysis
    print(f"\nKEY FINDINGS:")
    print("-" * 40)
    
    best_strategy = sorted_summary.iloc[0]
    worst_strategy = sorted_summary.iloc[-1]
    
    print(f"Best performing: {best_strategy['strategy']} ({best_strategy['annualized_return']:.1f}% annually)")
    print(f"Worst performing: {worst_strategy['strategy']} ({worst_strategy['annualized_return']:.1f}% annually)")
    
    # Compare EV vs Absolute strategies
    ev_strategies = summary_df[summary_df['strategy'].str.contains('Expected')]
    abs_strategies = summary_df[summary_df['strategy'].str.contains('Absolute')]
    
    if len(ev_strategies) > 0 and len(abs_strategies) > 0:
        ev_return = ev_strategies['annualized_return'].iloc[0]
        abs_returns = abs_strategies['annualized_return'].values
        
        print(f"\nMETHODOLOGY COMPARISON:")
        print(f"Expected Value return: {ev_return:.1f}% annually")
        print(f"Absolute Deviation strategies:")
        
        for _, row in abs_strategies.iterrows():
            diff = row['annualized_return'] - ev_return
            print(f"  {row['strategy']}: {row['annualized_return']:.1f}% ({diff:+.1f}% vs EV)")
        
        best_abs_return = abs_returns.max()
        
        if best_abs_return > ev_return:
            improvement = best_abs_return - ev_return
            print(f"\nCONCLUSION: Simple absolute deviation outperformed EV by {improvement:.1f}%")
            print(f"This suggests the complexity of EV calculations may not be justified")
        else:
            underperformance = ev_return - best_abs_return
            print(f"\nCONCLUSION: Expected Value outperformed absolute deviation by {underperformance:.1f}%")
            print(f"This validates the sophistication of the EV methodology")
    
    # Strategy simplicity analysis
    print(f"\nSTRATEGY COMPLEXITY ANALYSIS:")
    print("-" * 40)
    print(f"Expected Value approach:")
    print(f"  - Requires Z-score calculation")
    print(f"  - Needs EV interpolation from lookup table")
    print(f"  - Uses confidence-weighted predictions")
    print(f"  - Complex but theoretically sound")
    
    print(f"\nAbsolute Deviation approach:")
    print(f"  - Simple price vs trend line calculation")
    print(f"  - No statistical processing needed")
    print(f"  - Direct selection of most deviated assets")
    print(f"  - Simple but potentially effective")
    
    if len(abs_strategies) > 0:
        best_abs_strategy = abs_strategies.loc[abs_strategies['annualized_return'].idxmax()]
        print(f"\nRECOMMENDATION:")
        if 'Undervalued' in best_abs_strategy['strategy']:
            print(f"Focus on undervalued assets (negative deviations) appears optimal")
            print(f"Mean reversion principle confirmed with simpler methodology")
        else:
            print(f"Pure absolute deviation (regardless of direction) appears optimal")
            print(f"Volatility rather than mean reversion may be the key factor")

def main():
    print("Expected Value vs Absolute Deviation Strategy Comparison")
    print("="*70)
    print("Comparing sophisticated EV approach vs simple absolute deviation")
    
    # Create output directory
    output_dir = "/Users/tim/IWLS-OPTIONS/EV_VS_ABSOLUTE_COMPARISON"
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
    min_start_date = start_date + timedelta(days=730)  # Need 2+ years of history
    
    print(f"\nData range: {start_date.strftime('%Y-%m-%d')} to {end_date.strftime('%Y-%m-%d')}")
    print(f"Strategy period: {min_start_date.strftime('%Y-%m-%d')} to {end_date.strftime('%Y-%m-%d')}")
    
    results_dict = {}
    
    # Run Expected Value strategy
    print(f"\n{'='*20} EXPECTED VALUE STRATEGY {'='*20}")
    ev_result = run_ev_strategy(all_results, ev_df, min_start_date, end_date, initial_capital=10000)
    ev_daily_values = calculate_daily_portfolio_values(all_results, ev_result['portfolio_history'], min_start_date, end_date)
    ev_result['daily_values'] = ev_daily_values
    results_dict['EV'] = ev_result
    
    # Run Absolute Deviation strategies
    print(f"\n{'='*20} ABSOLUTE DEVIATION (UNDERVALUED) {'='*20}")
    abs_undervalued_result = run_absolute_deviation_strategy(all_results, min_start_date, end_date, prefer_undervalued=True, initial_capital=10000)
    abs_undervalued_daily_values = calculate_daily_portfolio_values(all_results, abs_undervalued_result['portfolio_history'], min_start_date, end_date)
    abs_undervalued_result['daily_values'] = abs_undervalued_daily_values
    results_dict['ABS_UNDERVALUED'] = abs_undervalued_result
    
    print(f"\n{'='*20} ABSOLUTE DEVIATION (ANY DIRECTION) {'='*20}")
    abs_any_result = run_absolute_deviation_strategy(all_results, min_start_date, end_date, prefer_undervalued=False, initial_capital=10000)
    abs_any_daily_values = calculate_daily_portfolio_values(all_results, abs_any_result['portfolio_history'], min_start_date, end_date)
    abs_any_result['daily_values'] = abs_any_daily_values
    results_dict['ABS_ANY'] = abs_any_result
    
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
    
    # Analyze asset selection overlap
    strategy_selections = analyze_asset_selection_overlap(results_dict)
    
    # Create visualizations
    print(f"\nCreating strategy comparison visualizations...")
    create_strategy_comparison_visualization(results_dict, benchmarks, min_start_date, end_date, output_dir)
    
    # Save results
    summary_df = save_comparison_results(results_dict, strategy_selections, output_dir)
    
    # Print comprehensive summary
    print_comparison_summary(results_dict, summary_df)
    
    print(f"\n" + "="*80)
    print("EV vs ABSOLUTE DEVIATION COMPARISON COMPLETE")
    print("="*80)
    print("Files saved:")
    print("  - ev_vs_absolute_deviation_comparison.png (4-panel comparison)")
    print("  - strategy_comparison_summary.csv (performance summary)")
    print("  - *_daily_values.csv (daily portfolio values)")
    print("  - *_portfolio_history.csv (rebalancing decisions)")
    print("  - asset_selection_analysis.csv (asset overlap analysis)")
    
    # Final recommendation
    if len(summary_df) > 0:
        best_strategy = summary_df.loc[summary_df['annualized_return'].idxmax()]
        print(f"\nFINAL RECOMMENDATION:")
        print(f"Best performing approach: {best_strategy['strategy']}")
        print(f"Annual return: {best_strategy['annualized_return']:.1f}%")
        print(f"Final value: ${best_strategy['final_value']:,.0f}")
        
        if 'Expected Value' in best_strategy['strategy']:
            print(f"\nThe sophisticated EV methodology proved superior")
            print(f"Complex analysis justified by outperformance")
        else:
            print(f"\nSimple absolute deviation outperformed complex EV")
            print(f"Consider adopting the simpler approach for efficiency")

if __name__ == "__main__":
    main()