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
        print("Could not load corrected EV data. Using Z-score proxy.")
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
            z_scores[asset_name] = {
                'z_score': z_score,
                'price_deviation': current_deviation
            }
    
    return z_scores

def get_expected_values_for_z_scores(ev_df, z_scores):
    """
    Get expected values for current Z-scores using interpolation
    """
    if ev_df is None:
        # Fallback: use Z-scores directly as proxy for EV
        asset_evs = {}
        for asset, data in z_scores.items():
            z_score = data['z_score']
            expected_value = max(0, (-z_score * 10))  # More negative Z = higher EV
            asset_evs[asset] = {
                'z_score': z_score,
                'expected_value': expected_value,
                'price_deviation': data['price_deviation']
            }
        return asset_evs
    
    asset_evs = {}
    
    for asset, data in z_scores.items():
        z_score = data['z_score']
        
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
            'price_deviation': data['price_deviation']
        }
    
    return asset_evs

def get_top_5_ev_assets(asset_evs):
    """
    Get the 5 assets with HIGHEST expected values (best opportunities)
    """
    if len(asset_evs) < 5:
        return []
    
    # Sort by expected value - HIGHEST first (best opportunities)
    sorted_assets = sorted(asset_evs.items(), key=lambda x: x[1]['expected_value'], reverse=True)
    
    # Return top 5
    return sorted_assets[:5]

def check_rebalance_needed(current_top5, new_top5):
    """
    Check if portfolio needs rebalancing (if top 5 assets changed)
    """
    current_assets = set([asset for asset, _ in current_top5])
    new_assets = set([asset for asset, _ in new_top5])
    
    return current_assets != new_assets

def run_top5_ev_trading_strategy(all_results, ev_df, start_date, end_date, 
                                 initial_capital=10000, rebalance_frequency='weekly'):
    """
    Run trading strategy: Always hold top 5 highest EV assets
    Rebalance when top 5 composition changes (checked weekly/daily)
    """
    print(f"\nRunning Top 5 EV Trading Strategy")
    print(f"Strategy: Always hold the 5 highest EV assets")
    print(f"Rebalance when composition changes (checked {rebalance_frequency})")
    print(f"Initial capital: ${initial_capital:,}")
    print(f"Period: {start_date.strftime('%Y-%m-%d')} to {end_date.strftime('%Y-%m-%d')}")
    
    # Generate check dates based on frequency
    if rebalance_frequency == 'daily':
        check_dates = pd.date_range(start=start_date, end=end_date, freq='D')
    elif rebalance_frequency == 'weekly':
        check_dates = pd.date_range(start=start_date, end=end_date, freq='W-MON')  # Every Monday
    else:  # monthly
        check_dates = pd.date_range(start=start_date, end=end_date, freq='MS')  # Month start
    
    # Generate all dates for daily portfolio tracking
    all_dates = pd.date_range(start=start_date, end=end_date, freq='D')
    
    # Portfolio tracking
    current_portfolio = {}  # {asset: shares}
    cash = 0
    current_top5 = []
    
    # History tracking
    rebalance_history = []
    daily_portfolio_values = []
    
    print(f"\nStrategy will check for rebalancing on {len(check_dates)} dates")
    print(f"Daily portfolio tracking over {len(all_dates)} dates")
    
    # Initial setup - start with first rebalance
    first_check_date = check_dates[0]
    
    print(f"\n--- Initial Portfolio Setup: {first_check_date.strftime('%Y-%m-%d')} ---")
    
    # Get initial top 5
    current_z_scores = calculate_current_z_scores(all_results, first_check_date)
    if len(current_z_scores) >= 5:
        asset_evs = get_expected_values_for_z_scores(ev_df, current_z_scores)
        if len(asset_evs) >= 5:
            current_top5 = get_top_5_ev_assets(asset_evs)
            
            print(f"Initial Top 5 Assets:")
            for asset, data in current_top5:
                print(f"  {asset}: EV = {data['expected_value']:.1f}%")
            
            # Buy equal amounts of each
            allocation_per_asset = initial_capital / 5
            
            for asset, data in current_top5:
                price = get_asset_price(all_results, asset, first_check_date)
                if price and price > 0:
                    shares = allocation_per_asset / price
                    current_portfolio[asset] = shares
                    print(f"  Bought {shares:.2f} shares of {asset} @ ${price:.2f}")
            
            # Record initial rebalance
            portfolio_value = sum(
                current_portfolio[asset] * get_asset_price(all_results, asset, first_check_date)
                for asset in current_portfolio
                if get_asset_price(all_results, asset, first_check_date)
            )
            
            combined_ev = sum(data['expected_value'] for _, data in current_top5)
            
            rebalance_history.append({
                'date': first_check_date,
                'reason': 'Initial Setup',
                'assets_added': [asset for asset, _ in current_top5],
                'assets_removed': [],
                'portfolio_value': portfolio_value,
                'combined_ev': combined_ev,
                'num_changes': 5
            })
    
    # Process remaining check dates for rebalancing
    rebalances_performed = 1  # Count initial setup
    
    for check_date in check_dates[1:]:  # Skip first date (already processed)
        # Get current top 5 based on latest data
        current_z_scores = calculate_current_z_scores(all_results, check_date)
        
        if len(current_z_scores) >= 5:
            asset_evs = get_expected_values_for_z_scores(ev_df, current_z_scores)
            
            if len(asset_evs) >= 5:
                new_top5 = get_top_5_ev_assets(asset_evs)
                
                # Check if rebalancing is needed
                if check_rebalance_needed(current_top5, new_top5):
                    print(f"\n--- Rebalancing on {check_date.strftime('%Y-%m-%d')} ---")
                    
                    # Calculate current portfolio value before rebalancing
                    pre_rebalance_value = sum(
                        current_portfolio[asset] * get_asset_price(all_results, asset, check_date)
                        for asset in current_portfolio
                        if get_asset_price(all_results, asset, check_date)
                    )
                    
                    # Determine changes
                    current_assets = set([asset for asset, _ in current_top5])
                    new_assets = set([asset for asset, _ in new_top5])
                    
                    assets_to_remove = current_assets - new_assets
                    assets_to_add = new_assets - current_assets
                    
                    print(f"  Assets to remove: {list(assets_to_remove)}")
                    print(f"  Assets to add: {list(assets_to_add)}")
                    
                    # Sell positions for removed assets
                    cash_from_sales = 0
                    for asset in assets_to_remove:
                        if asset in current_portfolio:
                            shares = current_portfolio[asset]
                            price = get_asset_price(all_results, asset, check_date)
                            if price:
                                proceeds = shares * price
                                cash_from_sales += proceeds
                                print(f"  Sold {shares:.2f} shares of {asset} @ ${price:.2f} = ${proceeds:.2f}")
                                del current_portfolio[asset]
                    
                    # Buy new positions
                    if len(assets_to_add) > 0:
                        allocation_per_new_asset = cash_from_sales / len(assets_to_add)
                        
                        for asset in assets_to_add:
                            price = get_asset_price(all_results, asset, check_date)
                            if price and price > 0:
                                shares = allocation_per_new_asset / price
                                current_portfolio[asset] = shares
                                print(f"  Bought {shares:.2f} shares of {asset} @ ${price:.2f}")
                    
                    # Update current top 5
                    current_top5 = new_top5
                    
                    # Calculate post-rebalance portfolio value
                    post_rebalance_value = sum(
                        current_portfolio[asset] * get_asset_price(all_results, asset, check_date)
                        for asset in current_portfolio
                        if get_asset_price(all_results, asset, check_date)
                    )
                    
                    combined_ev = sum(data['expected_value'] for _, data in current_top5)
                    
                    # Record rebalance
                    rebalance_history.append({
                        'date': check_date,
                        'reason': 'Composition Change',
                        'assets_added': list(assets_to_add),
                        'assets_removed': list(assets_to_remove),
                        'portfolio_value': post_rebalance_value,
                        'combined_ev': combined_ev,
                        'num_changes': len(assets_to_add) + len(assets_to_remove)
                    })
                    
                    rebalances_performed += 1
                    print(f"  New portfolio value: ${post_rebalance_value:.2f}")
                    print(f"  Combined EV: {combined_ev:.1f}%")
    
    # Calculate daily portfolio values
    print(f"\nCalculating daily portfolio values...")
    
    rebalance_dates = [r['date'] for r in rebalance_history]
    current_rebalance_idx = 0
    
    for date in all_dates:
        # Check if we need to update portfolio composition (new rebalance)
        if (current_rebalance_idx < len(rebalance_history) and 
            date >= rebalance_history[current_rebalance_idx]['date']):
            
            # Update portfolio to match rebalance
            current_rebalance_idx += 1
        
        # Calculate portfolio value for this date
        portfolio_value = 0
        position_count = 0
        
        for asset, shares in current_portfolio.items():
            price = get_asset_price(all_results, asset, date)
            if price:
                portfolio_value += shares * price
                position_count += 1
        
        daily_portfolio_values.append({
            'date': date,
            'portfolio_value': portfolio_value,
            'num_positions': position_count,
            'is_rebalance_date': date in rebalance_dates
        })
    
    # Calculate final metrics
    if daily_portfolio_values:
        final_value = daily_portfolio_values[-1]['portfolio_value']
        total_return = ((final_value / initial_capital) - 1) * 100
        years = (end_date - start_date).days / 365.25
        annualized_return = ((final_value / initial_capital) ** (1 / years) - 1) * 100
    else:
        final_value = initial_capital
        total_return = 0
        annualized_return = 0
    
    print(f"\n" + "="*60)
    print("STRATEGY RESULTS")
    print("="*60)
    print(f"Initial capital: ${initial_capital:,}")
    print(f"Final portfolio value: ${final_value:,.2f}")
    print(f"Total return: {total_return:.2f}%")
    print(f"Annualized return: {annualized_return:.2f}%")
    print(f"Total rebalances: {rebalances_performed}")
    print(f"Average time between rebalances: {len(all_dates)/rebalances_performed:.1f} days")
    
    return {
        'daily_values': pd.DataFrame(daily_portfolio_values),
        'rebalance_history': rebalance_history,
        'final_value': final_value,
        'total_return': total_return,
        'annualized_return': annualized_return,
        'rebalances_performed': rebalances_performed
    }

def create_portfolio_visualization(results, output_dir):
    """
    Create portfolio performance visualization with rebalance markers
    """
    daily_values = results['daily_values']
    rebalance_history = results['rebalance_history']
    
    if len(daily_values) == 0:
        print("No daily values to plot")
        return
    
    fig, ((ax1, ax2), (ax3, ax4)) = plt.subplots(2, 2, figsize=(20, 12))
    
    # Plot 1: Portfolio value over time with rebalance markers
    ax1.plot(daily_values['date'], daily_values['portfolio_value'], 
             linewidth=2, color='blue', label='Portfolio Value')
    
    # Add rebalance markers
    rebalance_dates = [r['date'] for r in rebalance_history]
    rebalance_values = []
    
    for rebalance_date in rebalance_dates:
        # Find portfolio value on rebalance date
        rebalance_row = daily_values[daily_values['date'] == rebalance_date]
        if len(rebalance_row) > 0:
            rebalance_values.append(rebalance_row['portfolio_value'].iloc[0])
        else:
            # Find closest date
            closest_idx = np.argmin(np.abs(daily_values['date'] - rebalance_date))
            rebalance_values.append(daily_values['portfolio_value'].iloc[closest_idx])
    
    ax1.scatter(rebalance_dates, rebalance_values, 
               color='red', s=100, marker='^', zorder=5, 
               label=f'Rebalances ({len(rebalance_dates)})')
    
    # Add initial capital line
    ax1.axhline(y=10000, color='gray', linestyle='--', alpha=0.7, label='Initial Capital')
    
    ax1.set_title('Top 5 EV Strategy: Portfolio Value Over Time', fontweight='bold', fontsize=14)
    ax1.set_xlabel('Date')
    ax1.set_ylabel('Portfolio Value ($)')
    ax1.legend()
    ax1.grid(True, alpha=0.3)
    ax1.yaxis.set_major_formatter(plt.FuncFormatter(lambda x, p: f'${x:,.0f}'))
    
    # Plot 2: Rebalance frequency over time
    if len(rebalance_history) > 1:
        rebalance_df = pd.DataFrame(rebalance_history)
        rebalance_df['days_since_last'] = rebalance_df['date'].diff().dt.days
        
        # Remove first entry (NaN)
        rebalance_df = rebalance_df.dropna(subset=['days_since_last'])
        
        ax2.bar(range(len(rebalance_df)), rebalance_df['days_since_last'], 
               color='orange', alpha=0.7)
        ax2.set_title('Days Between Rebalances', fontweight='bold')
        ax2.set_xlabel('Rebalance Number')
        ax2.set_ylabel('Days Since Last Rebalance')
        ax2.grid(True, alpha=0.3)
        
        # Add average line
        avg_days = rebalance_df['days_since_last'].mean()
        ax2.axhline(y=avg_days, color='red', linestyle='--', 
                   label=f'Average: {avg_days:.1f} days')
        ax2.legend()
    
    # Plot 3: Number of changes per rebalance
    if len(rebalance_history) > 0:
        rebalance_df = pd.DataFrame(rebalance_history)
        
        ax3.bar(range(len(rebalance_df)), rebalance_df['num_changes'], 
               color='green', alpha=0.7)
        ax3.set_title('Portfolio Changes per Rebalance', fontweight='bold')
        ax3.set_xlabel('Rebalance Number')
        ax3.set_ylabel('Number of Asset Changes')
        ax3.grid(True, alpha=0.3)
        
        # Add labels for rebalance dates
        ax3.set_xticks(range(len(rebalance_df)))
        ax3.set_xticklabels([r['date'].strftime('%Y-%m') for r in rebalance_history], 
                           rotation=45)
    
    # Plot 4: Combined EV over time
    if len(rebalance_history) > 0:
        rebalance_df = pd.DataFrame(rebalance_history)
        
        ax4.plot(rebalance_df['date'], rebalance_df['combined_ev'], 
                marker='o', linewidth=2, color='purple', markersize=6)
        ax4.set_title('Combined EV of Top 5 Assets at Each Rebalance', fontweight='bold')
        ax4.set_xlabel('Date')
        ax4.set_ylabel('Combined Expected Value (%)')
        ax4.grid(True, alpha=0.3)
        
        # Add average line
        avg_ev = rebalance_df['combined_ev'].mean()
        ax4.axhline(y=avg_ev, color='red', linestyle='--', 
                   label=f'Average: {avg_ev:.1f}%')
        ax4.legend()
    
    plt.tight_layout()
    plt.savefig(f"{output_dir}/top5_ev_trading_performance.png", dpi=300, bbox_inches='tight')
    plt.close()

def save_trading_results(results, output_dir):
    """
    Save trading strategy results
    """
    # Daily portfolio values
    results['daily_values'].to_csv(f"{output_dir}/top5_ev_daily_portfolio_values.csv", index=False)
    
    # Rebalance history
    rebalance_df = pd.DataFrame(results['rebalance_history'])
    rebalance_df.to_csv(f"{output_dir}/top5_ev_rebalance_history.csv", index=False)
    
    # Strategy summary
    summary = {
        'strategy': 'Top 5 Highest EV Assets',
        'final_value': results['final_value'],
        'total_return': results['total_return'],
        'annualized_return': results['annualized_return'],
        'rebalances_performed': results['rebalances_performed']
    }
    
    summary_df = pd.DataFrame([summary])
    summary_df.to_csv(f"{output_dir}/top5_ev_strategy_summary.csv", index=False)

def analyze_rebalance_patterns(results):
    """
    Analyze patterns in rebalancing
    """
    rebalance_history = results['rebalance_history']
    
    if len(rebalance_history) < 2:
        return
    
    print(f"\nREBALANCE PATTERN ANALYSIS:")
    print("-" * 40)
    
    rebalance_df = pd.DataFrame(rebalance_history)
    
    # Calculate time between rebalances
    rebalance_df['days_since_last'] = rebalance_df['date'].diff().dt.days
    time_between = rebalance_df['days_since_last'].dropna()
    
    print(f"Average days between rebalances: {time_between.mean():.1f}")
    print(f"Median days between rebalances: {time_between.median():.1f}")
    print(f"Min/Max days: {time_between.min():.0f} / {time_between.max():.0f}")
    
    # Asset turnover analysis
    all_additions = []
    all_removals = []
    
    for rebalance in rebalance_history[1:]:  # Skip initial setup
        all_additions.extend(rebalance['assets_added'])
        all_removals.extend(rebalance['assets_removed'])
    
    from collections import Counter
    
    if all_additions:
        print(f"\nMOST FREQUENTLY ADDED ASSETS:")
        addition_counts = Counter(all_additions)
        for asset, count in addition_counts.most_common(5):
            print(f"  {asset}: {count} times")
    
    if all_removals:
        print(f"\nMOST FREQUENTLY REMOVED ASSETS:")
        removal_counts = Counter(all_removals)
        for asset, count in removal_counts.most_common(5):
            print(f"  {asset}: {count} times")
    
    # Combined EV analysis
    ev_values = [r['combined_ev'] for r in rebalance_history]
    print(f"\nCOMBINED EV ANALYSIS:")
    print(f"Average combined EV: {np.mean(ev_values):.1f}%")
    print(f"EV range: {np.min(ev_values):.1f}% to {np.max(ev_values):.1f}%")
    print(f"EV standard deviation: {np.std(ev_values):.1f}%")

def main():
    print("Top 5 EV Trading Strategy with Dynamic Rebalancing")
    print("="*60)
    print("Strategy: Always hold the 5 highest EV assets")
    print("Rebalance when top 5 composition changes")
    
    # Create output directory
    output_dir = "/Users/tim/IWLS-OPTIONS/TOP5_EV_TRADING"
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
    min_start_date = start_date + timedelta(days=730)  # Need 2+ years of history for Z-scores
    
    print(f"\nData range: {start_date.strftime('%Y-%m-%d')} to {end_date.strftime('%Y-%m-%d')}")
    print(f"Strategy period: {min_start_date.strftime('%Y-%m-%d')} to {end_date.strftime('%Y-%m-%d')}")
    
    # Run trading strategy
    print(f"\n{'='*20} RUNNING TOP 5 EV TRADING STRATEGY {'='*20}")
    
    results = run_top5_ev_trading_strategy(
        all_results=all_results,
        ev_df=ev_df,
        start_date=min_start_date,
        end_date=end_date,
        initial_capital=10000,
        rebalance_frequency='weekly'  # Check for changes weekly
    )
    
    # Analyze rebalance patterns
    analyze_rebalance_patterns(results)
    
    # Create visualizations
    print(f"\nCreating visualizations...")
    create_portfolio_visualization(results, output_dir)
    
    # Save results
    save_trading_results(results, output_dir)
    
    print(f"\n" + "="*80)
    print("TOP 5 EV TRADING STRATEGY COMPLETE")
    print("="*80)
    print("Files saved:")
    print("  - top5_ev_trading_performance.png (4-panel performance analysis)")
    print("  - top5_ev_daily_portfolio_values.csv (daily portfolio tracking)")
    print("  - top5_ev_rebalance_history.csv (all rebalancing events)")
    print("  - top5_ev_strategy_summary.csv (performance summary)")
    
    # Final summary
    print(f"\nFINAL PERFORMANCE SUMMARY:")
    print(f"Strategy: Top 5 Highest EV Assets (Dynamic Rebalancing)")
    print(f"Final Value: ${results['final_value']:,.2f}")
    print(f"Total Return: {results['total_return']:.2f}%")
    print(f"Annualized Return: {results['annualized_return']:.2f}%")
    print(f"Total Rebalances: {results['rebalances_performed']}")

if __name__ == "__main__":
    main()