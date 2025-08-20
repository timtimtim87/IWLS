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

def generate_annual_rebalance_dates(start_date, end_date):
    """
    Generate rebalance dates exactly 365 calendar days apart
    """
    rebalance_dates = []
    current_date = start_date
    
    while current_date <= end_date:
        rebalance_dates.append(current_date)
        current_date = current_date + timedelta(days=365)  # Exactly 365 calendar days
    
    return rebalance_dates

def run_top5_ev_annual_rebalancing_strategy(all_results, ev_df, start_date, end_date, initial_capital=10000):
    """
    Run Top 5 EV strategy with ANNUAL rebalancing (365 calendar days)
    """
    print(f"\nRunning Top 5 EV Annual Rebalancing Strategy")
    print(f"Strategy: Hold top 5 highest EV assets, rebalance every 365 calendar days")
    print(f"Initial capital: ${initial_capital:,}")
    print(f"Period: {start_date.strftime('%Y-%m-%d')} to {end_date.strftime('%Y-%m-%d')}")
    
    # Generate annual rebalance dates (365 days apart)
    rebalance_dates = generate_annual_rebalance_dates(start_date, end_date)
    print(f"Rebalance dates: {len(rebalance_dates)} periods")
    for date in rebalance_dates:
        print(f"  {date.strftime('%Y-%m-%d')}")
    
    # Portfolio tracking
    portfolio = {}  # {asset: shares}
    cash = initial_capital
    
    # History tracking
    rebalance_history = []
    
    # Process each rebalance date
    for i, rebalance_date in enumerate(rebalance_dates):
        print(f"\n--- Rebalance {i+1}: {rebalance_date.strftime('%Y-%m-%d')} ---")
        
        # Sell all existing positions (except on first rebalance)
        if portfolio:
            print("Selling all existing positions:")
            total_proceeds = 0
            
            for asset, shares in portfolio.items():
                current_price = get_asset_price(all_results, asset, rebalance_date)
                if current_price:
                    proceeds = shares * current_price
                    total_proceeds += proceeds
                    print(f"  Sold {shares:.2f} shares of {asset} @ ${current_price:.2f} = ${proceeds:.2f}")
            
            cash = total_proceeds
            portfolio = {}
            print(f"Total cash from sales: ${cash:.2f}")
        
        # Calculate current Z-scores and EV for all assets
        current_z_scores = calculate_current_z_scores(all_results, rebalance_date)
        
        if len(current_z_scores) >= 5:
            asset_evs = get_expected_values_for_z_scores(ev_df, current_z_scores)
            
            if len(asset_evs) >= 5:
                # Get top 5 highest EV assets
                top5_assets = get_top_5_ev_assets(asset_evs)
                
                print(f"Selected Top 5 Assets by Expected Value:")
                for asset, data in top5_assets:
                    print(f"  {asset}: EV = {data['expected_value']:.1f}%, Z = {data['z_score']:.2f}")
                
                # Buy equal amounts of each (20% allocation each)
                allocation_per_asset = cash / 5
                
                print(f"\nBuying positions (${allocation_per_asset:.2f} each):")
                for asset, data in top5_assets:
                    current_price = get_asset_price(all_results, asset, rebalance_date)
                    if current_price and current_price > 0:
                        shares = allocation_per_asset / current_price
                        portfolio[asset] = shares
                        print(f"  Bought {shares:.2f} shares of {asset} @ ${current_price:.2f}")
                
                # Calculate total portfolio value
                portfolio_value = sum(
                    portfolio[asset] * get_asset_price(all_results, asset, rebalance_date)
                    for asset in portfolio
                    if get_asset_price(all_results, asset, rebalance_date)
                )
                
                combined_ev = sum(data['expected_value'] for _, data in top5_assets)
                
                # Record rebalance
                rebalance_history.append({
                    'date': rebalance_date,
                    'rebalance_num': i + 1,
                    'portfolio_value': portfolio_value,
                    'combined_ev': combined_ev,
                    'assets': [asset for asset, _ in top5_assets],
                    'individual_evs': {asset: data['expected_value'] for asset, data in top5_assets}
                })
                
                cash = 0  # All cash invested
                print(f"Portfolio value: ${portfolio_value:.2f}")
                print(f"Combined EV: {combined_ev:.1f}%")
            else:
                print(f"Insufficient EV data ({len(asset_evs)} assets)")
        else:
            print(f"Insufficient Z-score data ({len(current_z_scores)} assets)")
    
    # Calculate final portfolio value
    final_date = end_date
    final_portfolio_value = 0
    
    print(f"\n--- Final Portfolio Value ({final_date.strftime('%Y-%m-%d')}) ---")
    if portfolio:
        for asset, shares in portfolio.items():
            final_price = get_asset_price(all_results, asset, final_date)
            if final_price:
                asset_value = shares * final_price
                final_portfolio_value += asset_value
                print(f"{asset}: {shares:.2f} shares @ ${final_price:.2f} = ${asset_value:.2f}")
    
    # Calculate performance metrics
    total_return = ((final_portfolio_value / initial_capital) - 1) * 100
    years = (end_date - start_date).days / 365.25
    annualized_return = ((final_portfolio_value / initial_capital) ** (1 / years) - 1) * 100
    
    print(f"\n" + "="*60)
    print("ANNUAL REBALANCING STRATEGY RESULTS")
    print("="*60)
    print(f"Initial capital: ${initial_capital:,}")
    print(f"Final portfolio value: ${final_portfolio_value:,.2f}")
    print(f"Total return: {total_return:.2f}%")
    print(f"Annualized return: {annualized_return:.2f}%")
    print(f"Number of rebalances: {len(rebalance_dates)}")
    print(f"Days between rebalances: 365 (fixed)")
    
    return {
        'rebalance_history': rebalance_history,
        'final_value': final_portfolio_value,
        'total_return': total_return,
        'annualized_return': annualized_return,
        'rebalances_performed': len(rebalance_dates)
    }

def calculate_daily_portfolio_values(all_results, rebalance_history, start_date, end_date):
    """
    Calculate daily portfolio values between rebalances
    """
    print(f"\nCalculating daily portfolio values...")
    
    all_dates = pd.date_range(start=start_date, end=end_date, freq='D')
    daily_values = []
    
    current_portfolio = {}
    current_rebalance_idx = 0
    
    for date in all_dates:
        # Check if we need to update portfolio composition (rebalance occurred)
        if (current_rebalance_idx < len(rebalance_history) and 
            date >= rebalance_history[current_rebalance_idx]['date']):
            
            # Update to new portfolio composition
            rebalance_data = rebalance_history[current_rebalance_idx]
            portfolio_value = rebalance_data['portfolio_value']
            assets = rebalance_data['assets']
            
            # Calculate shares for each asset (equal weight)
            current_portfolio = {}
            if len(assets) > 0:
                allocation_per_asset = portfolio_value / len(assets)
                
                for asset in assets:
                    asset_price = get_asset_price(all_results, asset, rebalance_data['date'])
                    if asset_price and asset_price > 0:
                        shares = allocation_per_asset / asset_price
                        current_portfolio[asset] = shares
            
            current_rebalance_idx += 1
        
        # Calculate portfolio value for this date
        total_value = 0
        num_positions = 0
        
        for asset, shares in current_portfolio.items():
            price = get_asset_price(all_results, asset, date)
            if price:
                total_value += shares * price
                num_positions += 1
        
        daily_values.append({
            'date': date,
            'portfolio_value': total_value,
            'num_positions': num_positions
        })
    
    return pd.DataFrame(daily_values)

def create_annual_rebalancing_visualization(results, daily_values, output_dir):
    """
    Create visualization for annual rebalancing strategy
    """
    if len(daily_values) == 0:
        print("No daily values to plot")
        return
    
    fig, ((ax1, ax2), (ax3, ax4)) = plt.subplots(2, 2, figsize=(20, 12))
    
    # Plot 1: Portfolio value over time with rebalance markers
    ax1.plot(daily_values['date'], daily_values['portfolio_value'], 
             linewidth=2, color='blue', label='Portfolio Value')
    
    # Add rebalance markers
    rebalance_dates = [r['date'] for r in results['rebalance_history']]
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
               color='red', s=150, marker='^', zorder=5, 
               label=f'Annual Rebalances ({len(rebalance_dates)})')
    
    # Add initial capital line
    ax1.axhline(y=10000, color='gray', linestyle='--', alpha=0.7, label='Initial Capital')
    
    ax1.set_title('Top 5 EV Strategy: Annual Rebalancing (365 Days)', fontweight='bold', fontsize=14)
    ax1.set_xlabel('Date')
    ax1.set_ylabel('Portfolio Value ($)')
    ax1.legend()
    ax1.grid(True, alpha=0.3)
    ax1.yaxis.set_major_formatter(plt.FuncFormatter(lambda x, p: f'${x:,.0f}'))
    
    # Plot 2: Combined EV over time
    rebalance_df = pd.DataFrame(results['rebalance_history'])
    
    ax2.plot(rebalance_df['date'], rebalance_df['combined_ev'], 
            marker='o', linewidth=3, color='green', markersize=8)
    ax2.set_title('Combined Expected Value at Each Rebalance', fontweight='bold')
    ax2.set_xlabel('Date')
    ax2.set_ylabel('Combined EV (%)')
    ax2.grid(True, alpha=0.3)
    
    # Add value labels
    for _, row in rebalance_df.iterrows():
        ax2.annotate(f'{row["combined_ev"]:.1f}%', 
                    (row['date'], row['combined_ev']),
                    xytext=(0, 10), textcoords='offset points',
                    ha='center', fontweight='bold')
    
    # Plot 3: Asset allocation changes
    all_assets = set()
    for rebalance in results['rebalance_history']:
        all_assets.update(rebalance['assets'])
    
    asset_timeline = {}
    for asset in all_assets:
        asset_timeline[asset] = []
        for rebalance in results['rebalance_history']:
            if asset in rebalance['assets']:
                asset_timeline[asset].append(1)
            else:
                asset_timeline[asset].append(0)
    
    # Show top 10 most selected assets
    asset_counts = {asset: sum(timeline) for asset, timeline in asset_timeline.items()}
    top_assets = sorted(asset_counts.items(), key=lambda x: x[1], reverse=True)[:10]
    
    y_positions = range(len(top_assets))
    for i, (asset, count) in enumerate(top_assets):
        periods = []
        for j, included in enumerate(asset_timeline[asset]):
            if included:
                periods.append(j + 1)
        
        if periods:
            ax3.scatter(periods, [i] * len(periods), s=100, alpha=0.7, label=asset if i < 5 else "")
    
    ax3.set_title('Asset Selection Timeline (Top 10 Most Selected)', fontweight='bold')
    ax3.set_xlabel('Rebalance Period')
    ax3.set_ylabel('Assets')
    ax3.set_yticks(y_positions)
    ax3.set_yticklabels([asset for asset, _ in top_assets])
    ax3.grid(True, alpha=0.3)
    
    # Plot 4: Portfolio concentration
    ax4.plot(daily_values['date'], daily_values['num_positions'], 
             linewidth=2, color='orange')
    ax4.set_title('Portfolio Concentration (Number of Holdings)', fontweight='bold')
    ax4.set_xlabel('Date')
    ax4.set_ylabel('Number of Positions')
    ax4.grid(True, alpha=0.3)
    ax4.set_ylim(0, 6)
    
    plt.tight_layout()
    plt.savefig(f"{output_dir}/annual_rebalancing_analysis.png", dpi=300, bbox_inches='tight')
    plt.close()

def save_annual_rebalancing_results(results, daily_values, output_dir):
    """
    Save annual rebalancing results
    """
    # Daily portfolio values
    daily_values.to_csv(f"{output_dir}/annual_rebalance_daily_values.csv", index=False)
    
    # Rebalance history
    rebalance_df = pd.DataFrame(results['rebalance_history'])
    rebalance_df.to_csv(f"{output_dir}/annual_rebalance_history.csv", index=False)
    
    # Strategy summary
    summary = {
        'strategy': 'Top 5 EV Annual Rebalancing (365 days)',
        'final_value': results['final_value'],
        'total_return': results['total_return'],
        'annualized_return': results['annualized_return'],
        'rebalances_performed': results['rebalances_performed']
    }
    
    summary_df = pd.DataFrame([summary])
    summary_df.to_csv(f"{output_dir}/annual_rebalance_summary.csv", index=False)

def analyze_rebalancing_frequency_impact(results):
    """
    Analyze the impact of annual rebalancing frequency
    """
    print(f"\nANNUAL REBALANCING FREQUENCY ANALYSIS:")
    print("-" * 50)
    
    rebalance_history = results['rebalance_history']
    
    if len(rebalance_history) < 2:
        print("Need at least 2 rebalances for analysis")
        return
    
    # Calculate actual days between rebalances
    days_between = []
    for i in range(1, len(rebalance_history)):
        days = (rebalance_history[i]['date'] - rebalance_history[i-1]['date']).days
        days_between.append(days)
    
    print(f"Target rebalancing frequency: 365 days")
    print(f"Actual days between rebalances: {days_between}")
    print(f"Average days between rebalances: {np.mean(days_between):.1f}")
    print(f"Consistency: All should be exactly 365 days")
    
    # Asset turnover analysis
    print(f"\nASSET TURNOVER ANALYSIS:")
    print("-" * 30)
    
    total_changes = 0
    for i in range(1, len(rebalance_history)):
        prev_assets = set(rebalance_history[i-1]['assets'])
        curr_assets = set(rebalance_history[i]['assets'])
        
        assets_removed = prev_assets - curr_assets
        assets_added = curr_assets - prev_assets
        
        changes = len(assets_removed) + len(assets_added)
        total_changes += changes
        
        print(f"Rebalance {i+1}: {changes} changes")
        if assets_removed:
            print(f"  Removed: {', '.join(assets_removed)}")
        if assets_added:
            print(f"  Added: {', '.join(assets_added)}")
    
    avg_turnover = total_changes / (len(rebalance_history) - 1) if len(rebalance_history) > 1 else 0
    print(f"\nAverage turnover per rebalance: {avg_turnover:.1f} assets")
    print(f"Maximum possible turnover: 5 assets (100% turnover)")
    print(f"Turnover rate: {avg_turnover/5*100:.1f}%")

def main():
    print("Top 5 EV Strategy with ANNUAL Rebalancing (365 Calendar Days)")
    print("="*70)
    print("Strategy: Hold top 5 highest EV assets, rebalance exactly every 365 days")
    
    # Create output directory
    output_dir = "/Users/tim/IWLS-OPTIONS/ANNUAL_REBALANCING_365"
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
    print(f"Total strategy duration: {(end_date - min_start_date).days} days")
    
    # Run annual rebalancing strategy
    print(f"\n{'='*20} RUNNING ANNUAL REBALANCING STRATEGY {'='*20}")
    
    results = run_top5_ev_annual_rebalancing_strategy(
        all_results=all_results,
        ev_df=ev_df,
        start_date=min_start_date,
        end_date=end_date,
        initial_capital=10000
    )
    
    # Calculate daily portfolio values
    daily_values = calculate_daily_portfolio_values(all_results, results['rebalance_history'], min_start_date, end_date)
    
    # Analyze rebalancing frequency
    analyze_rebalancing_frequency_impact(results)
    
    # Create visualizations
    print(f"\nCreating visualizations...")
    create_annual_rebalancing_visualization(results, daily_values, output_dir)
    
    # Save results
    save_annual_rebalancing_results(results, daily_values, output_dir)
    
    print(f"\n" + "="*80)
    print("ANNUAL REBALANCING STRATEGY (365 DAYS) COMPLETE")
    print("="*80)
    print("Files saved:")
    print("  - annual_rebalancing_analysis.png (4-panel analysis)")
    print("  - annual_rebalance_daily_values.csv (daily portfolio tracking)")
    print("  - annual_rebalance_history.csv (rebalancing events)")
    print("  - annual_rebalance_summary.csv (performance summary)")
    
    # Final summary
    print(f"\nFINAL PERFORMANCE SUMMARY:")
    print(f"Strategy: Top 5 EV Assets, Annual Rebalancing (365 days)")
    print(f"Final Value: ${results['final_value']:,.2f}")
    print(f"Total Return: {results['total_return']:.2f}%")
    print(f"Annualized Return: {results['annualized_return']:.2f}%")
    print(f"Number of Rebalances: {results['rebalances_performed']}")
    print(f"Rebalancing Frequency: Exactly 365 calendar days")

if __name__ == "__main__":
    main()