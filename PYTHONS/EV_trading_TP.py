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

def check_take_profit_exits(all_results, positions, current_date, tp_target):
    """
    Check which positions should exit due to take profit targets
    """
    exits = {}
    
    for asset, position_data in positions.items():
        entry_price = position_data['entry_price']
        shares = position_data['shares']
        
        current_price = get_asset_price(all_results, asset, current_date)
        
        if current_price:
            gain_pct = ((current_price / entry_price) - 1) * 100
            
            if gain_pct >= tp_target:
                proceeds = shares * current_price
                exits[asset] = {
                    'shares': shares,
                    'entry_price': entry_price,
                    'exit_price': current_price,
                    'proceeds': proceeds,
                    'gain_pct': gain_pct,
                    'exit_date': current_date,
                    'exit_reason': f'TP_{tp_target}%'
                }
    
    return exits

def run_tp_exit_strategy(all_results, ev_df, start_date, end_date, group_num, tp_target, initial_capital=10000):
    """
    Run the take profit exit strategy with correct annual rebalancing logic
    """
    print(f"\nRunning TP Exit Strategy - Group {group_num + 1}, TP: {tp_target}%")
    print(f"Initial capital: ${initial_capital:,.2f}")
    print(f"Period: {start_date.strftime('%Y-%m-%d')} to {end_date.strftime('%Y-%m-%d')}")
    
    # Generate annual rebalance dates
    rebalance_dates = []
    current_date = start_date
    while current_date <= end_date:
        rebalance_dates.append(current_date)
        current_date = current_date.replace(year=current_date.year + 1)
    
    print(f"Number of annual rebalance periods: {len(rebalance_dates)}")
    
    accumulated_cash = initial_capital
    active_positions = {}
    closed_positions = []
    portfolio_history = []
    
    # Track daily portfolio values
    all_dates = pd.date_range(start=start_date, end=end_date, freq='D')
    daily_portfolio_values = []
    
    for current_date in all_dates:
        # Check for take profit exits on any day
        if active_positions:
            exits = check_take_profit_exits(all_results, active_positions, current_date, tp_target)
            
            for asset, exit_data in exits.items():
                accumulated_cash += exit_data['proceeds']
                closed_positions.append({
                    'asset': asset,
                    'entry_date': active_positions[asset]['entry_date'],
                    'exit_date': current_date,
                    'entry_price': exit_data['entry_price'],
                    'exit_price': exit_data['exit_price'],
                    'shares': exit_data['shares'],
                    'proceeds': exit_data['proceeds'],
                    'gain_pct': exit_data['gain_pct'],
                    'exit_reason': exit_data['exit_reason']
                })
                
                print(f"  TP Exit: {asset} +{exit_data['gain_pct']:.1f}% on {current_date.strftime('%Y-%m-%d')}")
                del active_positions[asset]
        
        # Check for annual rebalance (force close all positions and enter new ones)
        if current_date in rebalance_dates:
            rebalance_year = current_date.year
            print(f"\n--- Annual Rebalance: {current_date.strftime('%Y-%m-%d')} ---")
            
            # Force close ALL remaining positions at annual rebalance
            if active_positions:
                print("Force closing remaining positions for annual rebalance:")
                for asset, position_data in list(active_positions.items()):
                    current_price = get_asset_price(all_results, asset, current_date)
                    if current_price:
                        proceeds = position_data['shares'] * current_price
                        gain_pct = ((current_price / position_data['entry_price']) - 1) * 100
                        accumulated_cash += proceeds
                        
                        closed_positions.append({
                            'asset': asset,
                            'entry_date': position_data['entry_date'],
                            'exit_date': current_date,
                            'entry_price': position_data['entry_price'],
                            'exit_price': current_price,
                            'shares': position_data['shares'],
                            'proceeds': proceeds,
                            'gain_pct': gain_pct,
                            'exit_reason': f'Annual_Rebalance_{rebalance_year}'
                        })
                        
                        print(f"  {asset}: {gain_pct:+.1f}% (${proceeds:.2f})")
                        del active_positions[asset]
                
                print(f"Total cash after closing positions: ${accumulated_cash:.2f}")
            
            # Calculate current Z-scores and select new portfolio
            current_z_scores = calculate_current_z_scores(all_results, current_date)
            
            if len(current_z_scores) >= 4:
                asset_evs = get_expected_values_for_z_scores(ev_df, current_z_scores)
                
                if len(asset_evs) >= 4:
                    groups = select_top_assets_by_ev(asset_evs, group_size=4)
                    
                    if group_num < len(groups):
                        selected_group = groups[group_num]
                        
                        print(f"Selected assets for Group {group_num + 1}:")
                        for asset_name, ev_data in selected_group:
                            print(f"  {asset_name}: Z={ev_data['z_score']:.2f}, EV={ev_data['expected_value']:.1f}%")
                        
                        # Enter new positions using ALL accumulated cash
                        if accumulated_cash > 0:
                            allocation_per_asset = accumulated_cash / len(selected_group)
                            
                            print(f"Entering new positions with ${accumulated_cash:.2f} total cash:")
                            print(f"Allocation per asset: ${allocation_per_asset:.2f}")
                            
                            for asset_name, ev_data in selected_group:
                                current_price = get_asset_price(all_results, asset_name, current_date)
                                if current_price and current_price > 0:
                                    shares = allocation_per_asset / current_price
                                    
                                    active_positions[asset_name] = {
                                        'entry_date': current_date,
                                        'entry_price': current_price,
                                        'shares': shares,
                                        'initial_investment': allocation_per_asset,
                                        'z_score': ev_data['z_score'],
                                        'expected_value': ev_data['expected_value']
                                    }
                                    
                                    print(f"  {asset_name}: {shares:.2f} shares @ ${current_price:.2f}")
                            
                            accumulated_cash = 0  # All cash now invested
                        else:
                            print("No cash available for new entries")
                    else:
                        print(f"Group {group_num + 1} not available")
                else:
                    print("Insufficient EV data for asset selection")
            else:
                print("Insufficient Z-score data for asset selection")
        
        # Calculate current portfolio value
        position_value = 0
        for asset, position_data in active_positions.items():
            current_price = get_asset_price(all_results, asset, current_date)
            if current_price:
                position_value += position_data['shares'] * current_price
        
        total_value = accumulated_cash + position_value
        
        # Record daily portfolio value
        daily_portfolio_values.append({
            'date': current_date,
            'cash': accumulated_cash,
            'position_value': position_value,
            'total_value': total_value,
            'num_positions': len(active_positions)
        })
        
        # Record major portfolio events weekly
        if current_date in rebalance_dates or (current_date.weekday() == 4 and len(active_positions) > 0):
            portfolio_history.append({
                'date': current_date,
                'cash': accumulated_cash,
                'position_value': position_value,
                'total_value': total_value,
                'active_positions': len(active_positions),
                'assets': list(active_positions.keys()) if active_positions else []
            })
    
    # Final portfolio value calculation
    final_position_value = 0
    if active_positions:
        print(f"\nFinal positions at end of period:")
        for asset, position_data in active_positions.items():
            final_price = get_asset_price(all_results, asset, end_date)
            if final_price:
                asset_value = position_data['shares'] * final_price
                final_position_value += asset_value
                gain_pct = ((final_price / position_data['entry_price']) - 1) * 100
                print(f"  {asset}: {gain_pct:+.1f}% (${asset_value:.2f})")
    
    final_value = accumulated_cash + final_position_value
    total_return = ((final_value / initial_capital) - 1) * 100
    years = (end_date - start_date).days / 365.25
    annualized_return = ((final_value / initial_capital) ** (1 / years) - 1) * 100
    
    print(f"\nGroup {group_num + 1} TP {tp_target}% Strategy Results:")
    print(f"Initial capital: ${initial_capital:.2f}")
    print(f"Final cash: ${accumulated_cash:.2f}")
    print(f"Final position value: ${final_position_value:.2f}")
    print(f"Final total value: ${final_value:.2f}")
    print(f"Total return: {total_return:.2f}%")
    print(f"Annualized return: {annualized_return:.2f}%")
    print(f"Total trades closed: {len(closed_positions)}")
    
    if closed_positions:
        tp_exits = [t for t in closed_positions if 'TP_' in t['exit_reason']]
        annual_exits = [t for t in closed_positions if 'Annual_' in t['exit_reason']]
        
        print(f"TP exits: {len(tp_exits)}")
        print(f"Annual rebalance exits: {len(annual_exits)}")
        
        if tp_exits:
            tp_gains = [t['gain_pct'] for t in tp_exits]
            print(f"Average TP exit gain: {np.mean(tp_gains):.1f}%")
            print(f"TP success rate: {len([g for g in tp_gains if g >= tp_target])}/{len(tp_gains)} ({len([g for g in tp_gains if g >= tp_target])/len(tp_gains)*100:.1f}%)")
        
        if annual_exits:
            annual_gains = [t['gain_pct'] for t in annual_exits]
            print(f"Average annual exit gain: {np.mean(annual_gains):.1f}%")
    
    return {
        'portfolio_history': portfolio_history,
        'daily_values': pd.DataFrame(daily_portfolio_values),
        'closed_positions': closed_positions,
        'final_value': final_value,
        'annualized_return': annualized_return,
        'total_return': total_return
    }

def run_annual_rebalance_strategy(all_results, ev_df, start_date, end_date, group_num, initial_capital=10000):
    """
    Run the original annual rebalance strategy for comparison
    """
    print(f"\nRunning Annual Rebalance Strategy - Group {group_num + 1}")
    print(f"Initial capital: ${initial_capital:,.2f}")
    
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
        print(f"\n--- Rebalance {i+1}: {rebalance_date.strftime('%Y-%m-%d')} ---")
        
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
        
        # Calculate current Z-scores and select new portfolio
        current_z_scores = calculate_current_z_scores(all_results, rebalance_date)
        
        if len(current_z_scores) >= 4:
            asset_evs = get_expected_values_for_z_scores(ev_df, current_z_scores)
            
            if len(asset_evs) >= 4:
                groups = select_top_assets_by_ev(asset_evs, group_size=4)
                
                if group_num < len(groups):
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
            'cash': 0,
            'assets': list(portfolio.keys()),
            'group_num': group_num + 1
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
    
    print(f"\nAnnual Rebalance Results:")
    print(f"Final value: ${final_value:.2f}")
    print(f"Total return: {total_return:.2f}%")
    print(f"Annualized return: {annualized_return:.2f}%")
    
    # Calculate daily values for comparison
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
            'cash': 0,
            'position_value': total_value,
            'total_value': total_value,
            'num_positions': len(current_portfolio)
        })
    
    return {
        'portfolio_history': portfolio_history,
        'daily_values': pd.DataFrame(daily_values),
        'final_value': final_value,
        'annualized_return': annualized_return,
        'total_return': total_return
    }

def create_strategy_comparison_visualization(results_dict, output_dir):
    """
    Create comprehensive comparison of all strategies
    """
    fig, ((ax1, ax2), (ax3, ax4)) = plt.subplots(2, 2, figsize=(20, 12))
    
    colors = ['blue', 'green', 'orange', 'red', 'purple', 'brown', 'pink', 'gray']
    
    # Plot 1: Performance over time
    for i, (strategy_name, data) in enumerate(results_dict.items()):
        daily_values = data['daily_values']
        if len(daily_values) > 0:
            ax1.plot(daily_values['date'], daily_values['total_value'], 
                    label=strategy_name, linewidth=2, color=colors[i % len(colors)])
    
    ax1.axhline(y=10000, color='gray', linestyle=':', alpha=0.7, label='Initial Capital')
    ax1.set_title('Strategy Performance Comparison', fontsize=14, fontweight='bold')
    ax1.set_xlabel('Date')
    ax1.set_ylabel('Portfolio Value ($)')
    ax1.legend()
    ax1.grid(True, alpha=0.3)
    
    # Plot 2: Final returns comparison
    strategy_names = list(results_dict.keys())
    final_returns = [data['annualized_return'] for data in results_dict.values()]
    
    bars = ax2.bar(range(len(strategy_names)), final_returns, 
                  color=colors[:len(strategy_names)], alpha=0.7)
    ax2.set_xticks(range(len(strategy_names)))
    ax2.set_xticklabels(strategy_names, rotation=45, ha='right')
    ax2.set_title('Annualized Returns Comparison', fontsize=14, fontweight='bold')
    ax2.set_ylabel('Annualized Return (%)')
    ax2.grid(True, alpha=0.3)
    
    for bar, value in zip(bars, final_returns):
        height = bar.get_height()
        ax2.text(bar.get_x() + bar.get_width()/2., height + height*0.01,
                f'{value:.1f}%', ha='center', va='bottom', fontweight='bold')
    
    # Plot 3: Portfolio concentration over time (number of positions)
    for i, (strategy_name, data) in enumerate(results_dict.items()):
        daily_values = data['daily_values']
        if len(daily_values) > 0 and 'num_positions' in daily_values.columns:
            ax3.plot(daily_values['date'], daily_values['num_positions'], 
                    label=strategy_name, linewidth=2, color=colors[i % len(colors)])
    
    ax3.set_title('Portfolio Concentration (Number of Positions)', fontsize=14, fontweight='bold')
    ax3.set_xlabel('Date')
    ax3.set_ylabel('Number of Positions')
    ax3.legend()
    ax3.grid(True, alpha=0.3)
    
    # Plot 4: Cash allocation over time
    for i, (strategy_name, data) in enumerate(results_dict.items()):
        daily_values = data['daily_values']
        if len(daily_values) > 0 and 'cash' in daily_values.columns:
            cash_pct = daily_values['cash'] / daily_values['total_value'] * 100
            ax4.plot(daily_values['date'], cash_pct, 
                    label=strategy_name, linewidth=2, color=colors[i % len(colors)])
    
    ax4.set_title('Cash Allocation Over Time (%)', fontsize=14, fontweight='bold')
    ax4.set_xlabel('Date')
    ax4.set_ylabel('Cash Percentage (%)')
    ax4.legend()
    ax4.grid(True, alpha=0.3)
    
    plt.tight_layout()
    plt.savefig(f"{output_dir}/tp_strategy_comparison.png", dpi=300, bbox_inches='tight')
    plt.close()

def save_tp_strategy_results(results_dict, output_dir):
    """
    Save all strategy results
    """
    # Summary table
    summary_data = []
    for strategy_name, data in results_dict.items():
        summary_data.append({
            'strategy': strategy_name,
            'final_value': data['final_value'],
            'total_return': data['total_return'],
            'annualized_return': data['annualized_return']
        })
    
    summary_df = pd.DataFrame(summary_data)
    summary_df.to_csv(f"{output_dir}/tp_strategy_summary.csv", index=False)
    
    # Individual strategy details
    for strategy_name, data in results_dict.items():
        safe_name = strategy_name.replace(' ', '_').replace('%', 'pct')
        
        # Daily values
        data['daily_values'].to_csv(f"{output_dir}/{safe_name}_daily_values.csv", index=False)
        
        # Portfolio history
        portfolio_df = pd.DataFrame(data['portfolio_history'])
        portfolio_df.to_csv(f"{output_dir}/{safe_name}_portfolio_history.csv", index=False)
        
        # Closed positions (if available)
        if 'closed_positions' in data and data['closed_positions']:
            closed_df = pd.DataFrame(data['closed_positions'])
            closed_df.to_csv(f"{output_dir}/{safe_name}_closed_positions.csv", index=False)
    
    return summary_df

def print_tp_strategy_summary(results_dict, summary_df):
    """
    Print comprehensive summary of TP strategy results
    """
    print("\n" + "="*80)
    print("TAKE PROFIT vs ANNUAL REBALANCE STRATEGY COMPARISON")
    print("="*80)
    
    print(f"\nSTRATEGY COMPARISON:")
    print("-" * 60)
    print(f"{'Strategy':<25} {'Final Value':<12} {'Total Ret':<10} {'Annual Ret':<10}")
    print("-" * 60)
    
    sorted_summary = summary_df.sort_values('annualized_return', ascending=False)
    for _, row in sorted_summary.iterrows():
        print(f"{row['strategy']:<25} ${row['final_value']:>10,.0f} "
              f"{row['total_return']:>8.1f}% {row['annualized_return']:>8.1f}%")
    
    # Analysis
    print(f"\nKEY INSIGHTS:")
    print("-" * 40)
    
    best_strategy = sorted_summary.iloc[0]
    worst_strategy = sorted_summary.iloc[-1]
    
    print(f"Best performing: {best_strategy['strategy']} ({best_strategy['annualized_return']:.1f}% annually)")
    print(f"Worst performing: {worst_strategy['strategy']} ({worst_strategy['annualized_return']:.1f}% annually)")
    
    # Compare TP strategies to annual rebalance
    annual_strategies = [s for s in results_dict.keys() if 'Annual' in s]
    tp_strategies = [s for s in results_dict.keys() if 'TP' in s]
    
    if annual_strategies and tp_strategies:
        annual_returns = [summary_df[summary_df['strategy'] == s]['annualized_return'].iloc[0] for s in annual_strategies]
        tp_returns = [summary_df[summary_df['strategy'] == s]['annualized_return'].iloc[0] for s in tp_strategies]
        
        avg_annual = np.mean(annual_returns)
        avg_tp = np.mean(tp_returns)
        
        print(f"\nSTRATEGY TYPE ANALYSIS:")
        print(f"Average Annual Rebalance return: {avg_annual:.1f}%")
        print(f"Average Take Profit return: {avg_tp:.1f}%")
        
        if avg_tp > avg_annual:
            print(f"CONCLUSION: Take Profit strategies outperformed by {avg_tp - avg_annual:.1f}%")
        else:
            print(f"CONCLUSION: Annual Rebalance outperformed by {avg_annual - avg_tp:.1f}%")
    
    # Group comparison
    group1_strategies = [s for s in results_dict.keys() if 'Group 1' in s]
    group2_strategies = [s for s in results_dict.keys() if 'Group 2' in s]
    
    if group1_strategies and group2_strategies:
        group1_returns = [summary_df[summary_df['strategy'] == s]['annualized_return'].iloc[0] for s in group1_strategies]
        group2_returns = [summary_df[summary_df['strategy'] == s]['annualized_return'].iloc[0] for s in group2_strategies]
        
        avg_group1 = np.mean(group1_returns)
        avg_group2 = np.mean(group2_returns)
        
        print(f"\nGROUP COMPARISON:")
        print(f"Average Group 1 (top EV) return: {avg_group1:.1f}%")
        print(f"Average Group 2 return: {avg_group2:.1f}%")
        
        if avg_group1 > avg_group2:
            print(f"CONCLUSION: Higher EV assets outperformed by {avg_group1 - avg_group2:.1f}%")
        else:
            print(f"CONCLUSION: Group 2 outperformed top EV group by {avg_group2 - avg_group1:.1f}%")

def main():
    print("Take Profit Exit Strategy vs Annual Rebalancing")
    print("="*60)
    print("Testing Groups 1 & 2 with TP targets: 35%, 40%, 50% vs Annual Rebalancing")
    
    # Create output directory
    output_dir = "/Users/tim/IWLS-OPTIONS/TP_STRATEGY_COMPARISON"
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
    
    # Test configurations
    groups_to_test = [0, 1]  # Group 1 and Group 2 (0-indexed)
    tp_targets = [35, 40, 50]
    
    results_dict = {}
    
    # Run Take Profit strategies
    for group_num in groups_to_test:
        for tp_target in tp_targets:
            strategy_name = f"Group {group_num + 1} TP {tp_target}%"
            print(f"\n{'='*20} {strategy_name} {'='*20}")
            
            result = run_tp_exit_strategy(
                all_results, ev_df, min_start_date, end_date, 
                group_num, tp_target, initial_capital=10000
            )
            
            results_dict[strategy_name] = result
    
    # Run Annual Rebalancing strategies for comparison
    for group_num in groups_to_test:
        strategy_name = f"Group {group_num + 1} Annual Rebalance"
        print(f"\n{'='*20} {strategy_name} {'='*20}")
        
        result = run_annual_rebalance_strategy(
            all_results, ev_df, min_start_date, end_date, 
            group_num, initial_capital=10000
        )
        
        results_dict[strategy_name] = result
    
    # Create visualizations
    print(f"\nCreating strategy comparison visualizations...")
    create_strategy_comparison_visualization(results_dict, output_dir)
    
    # Save results
    summary_df = save_tp_strategy_results(results_dict, output_dir)
    
    # Print comprehensive summary
    print_tp_strategy_summary(results_dict, summary_df)
    
    print(f"\n" + "="*80)
    print("TAKE PROFIT STRATEGY ANALYSIS COMPLETE")
    print("="*80)
    print("Files saved:")
    print("  - tp_strategy_comparison.png (4-panel comparison)")
    print("  - tp_strategy_summary.csv (performance summary)")
    print("  - *_daily_values.csv (daily portfolio values)")
    print("  - *_portfolio_history.csv (major portfolio events)")
    print("  - *_closed_positions.csv (individual trade results)")
    
    # Key findings
    if len(summary_df) > 0:
        best_strategy = summary_df.loc[summary_df['annualized_return'].idxmax()]
        print(f"\nBEST PERFORMING STRATEGY:")
        print(f"  {best_strategy['strategy']}: {best_strategy['annualized_return']:.1f}% annually")
        print(f"  Final value: ${best_strategy['final_value']:,.0f}")
        
        # Check if TP strategies dominated
        tp_strategies = summary_df[summary_df['strategy'].str.contains('TP')]
        annual_strategies = summary_df[summary_df['strategy'].str.contains('Annual')]
        
        if len(tp_strategies) > 0 and len(annual_strategies) > 0:
            best_tp = tp_strategies['annualized_return'].max()
            best_annual = annual_strategies['annualized_return'].max()
            
            print(f"\nSTRATEGY TYPE WINNER:")
            if best_tp > best_annual:
                print(f"  Take Profit strategies superior: {best_tp:.1f}% vs {best_annual:.1f}%")
                print(f"  Individual exits outperformed annual rebalancing")
            else:
                print(f"  Annual Rebalancing superior: {best_annual:.1f}% vs {best_tp:.1f}%")
                print(f"  Full portfolio rebalancing outperformed individual exits")

if __name__ == "__main__":
    main()