import pandas as pd
import numpy as np
import matplotlib.pyplot as plt
import seaborn as sns
from datetime import datetime, timedelta
import warnings
import os
from collections import defaultdict
warnings.filterwarnings('ignore')

def load_all_iwls_data(v2_dir):
    """
    Load IWLS data for all assets
    """
    print("Loading IWLS data for all assets...")
    
    all_data = {}
    asset_dirs = [d for d in os.listdir(v2_dir) 
                  if os.path.isdir(os.path.join(v2_dir, d)) and 
                  not d.startswith('FORWARD_RETURNS') and not d.startswith('OLD') and
                  not d.startswith('REBALANCING') and not d.startswith('GROWTH_RATE')]
    
    for asset_name in asset_dirs:
        asset_dir = os.path.join(v2_dir, asset_name)
        iwls_file = os.path.join(asset_dir, f"{asset_name}_iwls_results.csv")
        
        if os.path.exists(iwls_file):
            try:
                df = pd.read_csv(iwls_file)
                df['date'] = pd.to_datetime(df['date'])
                df = df.dropna(subset=['price_deviation', 'price']).sort_values('date').reset_index(drop=True)
                
                if len(df) > 500:  # Need sufficient data
                    all_data[asset_name] = df
                    
            except Exception as e:
                print(f"  ⚠️  Error loading {asset_name}: {e}")
    
    print(f"✅ Successfully loaded {len(all_data)} assets")
    return all_data

def get_available_assets_on_date(all_data, target_date, lookback_days=10):
    """
    Get assets that have valid data around a specific date with their current metrics
    """
    available_assets = []
    
    for asset_name, df in all_data.items():
        # Find data points around the target date
        date_mask = (df['date'] >= target_date - timedelta(days=lookback_days)) & \
                   (df['date'] <= target_date + timedelta(days=lookback_days))
        nearby_data = df[date_mask]
        
        if len(nearby_data) > 0:
            # Get the closest data point to target date
            closest_idx = (nearby_data['date'] - target_date).abs().idxmin()
            closest_row = nearby_data.loc[closest_idx]
            
            available_assets.append({
                'asset': asset_name,
                'date': closest_row['date'],
                'price': closest_row['price'],
                'price_deviation': closest_row['price_deviation'],
                'absolute_deviation': abs(closest_row['price_deviation']),
                'trend_line_value': closest_row['trend_line_value'],
                'annual_growth': closest_row['annual_growth']
            })
    
    return pd.DataFrame(available_assets)

def select_top_underperforming_portfolios(available_df, n_per_portfolio=5):
    """
    Select top 3 portfolios of most underperforming stocks
    """
    # Focus on underperforming stocks (negative deviation)
    underperforming = available_df[available_df['price_deviation'] < 0].copy()
    
    if len(underperforming) < n_per_portfolio * 3:
        print(f"    Warning: Only {len(underperforming)} underperforming stocks available")
        # If not enough underperforming, use all stocks sorted by deviation
        sorted_df = available_df.sort_values('price_deviation', ascending=True)
    else:
        # Sort underperforming stocks by most negative deviation
        sorted_df = underperforming.sort_values('price_deviation', ascending=True)
    
    portfolios = {}
    for i in range(3):  # Only top 3 portfolios
        start_idx = i * n_per_portfolio
        end_idx = (i + 1) * n_per_portfolio
        
        if start_idx < len(sorted_df):
            portfolio_stocks = sorted_df.iloc[start_idx:end_idx].copy()
            portfolio_name = f'portfolio_{i+1}'
            portfolios[portfolio_name] = portfolio_stocks
            
            # Add portfolio info
            portfolios[portfolio_name]['portfolio'] = i + 1
            
    return portfolios

def calculate_portfolio_current_performance(portfolio_df, all_data, entry_date, current_date):
    """
    Calculate current portfolio performance (unrealized gains/losses)
    """
    if len(portfolio_df) == 0:
        return 0.0, [], 0
    
    current_returns = []
    successful_prices = 0
    
    for _, stock in portfolio_df.iterrows():
        asset_name = stock['asset']
        entry_price = stock['price']
        
        if asset_name in all_data:
            asset_df = all_data[asset_name]
            
            # Find current price (closest to current_date)
            current_mask = (asset_df['date'] >= current_date - timedelta(days=15)) & \
                          (asset_df['date'] <= current_date + timedelta(days=15))
            current_data = asset_df[current_mask]
            
            if len(current_data) > 0:
                closest_current_idx = (current_data['date'] - current_date).abs().idxmin()
                current_price = current_data.loc[closest_current_idx, 'price']
                
                stock_return = ((current_price / entry_price) - 1) * 100
                current_returns.append(stock_return)
                successful_prices += 1
    
    if len(current_returns) > 0:
        portfolio_return = np.mean(current_returns)  # Equal weight
        return portfolio_return, current_returns, successful_prices
    else:
        return 0.0, [], 0

def run_dynamic_profit_taking_strategy(all_data, start_date, end_date, 
                                     profit_targets=[15, 25, 35], max_hold_days=365):
    """
    Run dynamic profit-taking strategy with multiple target levels
    """
    print(f"\nRunning dynamic profit-taking strategy from {start_date} to {end_date}")
    print(f"Profit targets: {profit_targets}%")
    print(f"Maximum hold period: {max_hold_days} days")
    
    strategy_results = {}
    
    # Initialize results for each profit target and each portfolio
    for target in profit_targets:
        strategy_results[f'target_{target}'] = {
            'portfolio_1': {'trades': [], 'cumulative_return': 1.0, 'total_return': 0, 'active_position': None},
            'portfolio_2': {'trades': [], 'cumulative_return': 1.0, 'total_return': 0, 'active_position': None},
            'portfolio_3': {'trades': [], 'cumulative_return': 1.0, 'total_return': 0, 'active_position': None}
        }
    
    current_date = pd.to_datetime(start_date)
    end_date = pd.to_datetime(end_date)
    
    # Track daily performance for visualization
    daily_tracking = []
    
    trade_id = 0
    
    # Main strategy loop - check every 7 days
    while current_date < end_date:
        
        # Get available assets
        available_assets = get_available_assets_on_date(all_data, current_date)
        
        if len(available_assets) < 15:  # Need at least 15 for 3 portfolios of 5
            current_date += timedelta(days=7)
            continue
        
        # For each profit target level
        for target in profit_targets:
            target_key = f'target_{target}'
            
            # For each portfolio (top 3)
            for portfolio_name in ['portfolio_1', 'portfolio_2', 'portfolio_3']:
                portfolio_data = strategy_results[target_key][portfolio_name]
                active_position = portfolio_data['active_position']
                
                # Check if we have an active position
                if active_position is not None:
                    # Calculate current performance
                    entry_date = active_position['entry_date']
                    entry_portfolio = active_position['portfolio_df']
                    
                    current_return, individual_returns, successful_prices = calculate_portfolio_current_performance(
                        entry_portfolio, all_data, entry_date, current_date
                    )
                    
                    days_held = (current_date - entry_date).days
                    
                    # Check profit target or max hold period
                    should_exit = False
                    exit_reason = ""
                    
                    if current_return >= target:
                        should_exit = True
                        exit_reason = f"profit_target_{target}%"
                    elif days_held >= max_hold_days:
                        should_exit = True
                        exit_reason = "max_hold_period"
                    
                    if should_exit:
                        # Close position
                        trade_id += 1
                        
                        trade_record = {
                            'trade_id': trade_id,
                            'target_level': target,
                            'portfolio_rank': int(portfolio_name.split('_')[1]),
                            'entry_date': entry_date,
                            'exit_date': current_date,
                            'days_held': days_held,
                            'return_pct': current_return,
                            'exit_reason': exit_reason,
                            'successful_prices': successful_prices,
                            'total_stocks': len(entry_portfolio),
                            'individual_returns': individual_returns,
                            'entry_avg_deviation': entry_portfolio['price_deviation'].mean(),
                            'entry_portfolio_details': entry_portfolio.to_dict('records')
                        }
                        
                        portfolio_data['trades'].append(trade_record)
                        
                        # Update cumulative return
                        period_return = current_return / 100
                        portfolio_data['cumulative_return'] *= (1 + period_return)
                        
                        # Clear active position
                        portfolio_data['active_position'] = None
                        
                        print(f"    {target_key} {portfolio_name}: EXIT - {current_return:.2f}% after {days_held} days ({exit_reason})")
                
                # If no active position, look for entry
                if portfolio_data['active_position'] is None:
                    # Select new portfolio
                    portfolios = select_top_underperforming_portfolios(available_assets)
                    
                    if portfolio_name in portfolios:
                        new_portfolio = portfolios[portfolio_name]
                        
                        # Enter new position
                        portfolio_data['active_position'] = {
                            'entry_date': current_date,
                            'portfolio_df': new_portfolio
                        }
                        
                        avg_deviation = new_portfolio['price_deviation'].mean()
                        print(f"    {target_key} {portfolio_name}: ENTER - {len(new_portfolio)} stocks, avg deviation: {avg_deviation:.2f}%")
        
        # Record daily tracking data
        daily_record = {'date': current_date}
        
        for target in profit_targets:
            target_key = f'target_{target}'
            for portfolio_name in ['portfolio_1', 'portfolio_2', 'portfolio_3']:
                portfolio_data = strategy_results[target_key][portfolio_name]
                daily_record[f'{target_key}_{portfolio_name}_cumulative'] = portfolio_data['cumulative_return']
        
        daily_tracking.append(daily_record)
        
        # Move to next check date
        current_date += timedelta(days=7)
    
    # Close any remaining positions at end date
    for target in profit_targets:
        target_key = f'target_{target}'
        for portfolio_name in ['portfolio_1', 'portfolio_2', 'portfolio_3']:
            portfolio_data = strategy_results[target_key][portfolio_name]
            active_position = portfolio_data['active_position']
            
            if active_position is not None:
                entry_date = active_position['entry_date']
                entry_portfolio = active_position['portfolio_df']
                
                final_return, individual_returns, successful_prices = calculate_portfolio_current_performance(
                    entry_portfolio, all_data, entry_date, end_date
                )
                
                days_held = (end_date - entry_date).days
                
                trade_id += 1
                trade_record = {
                    'trade_id': trade_id,
                    'target_level': target,
                    'portfolio_rank': int(portfolio_name.split('_')[1]),
                    'entry_date': entry_date,
                    'exit_date': end_date,
                    'days_held': days_held,
                    'return_pct': final_return,
                    'exit_reason': 'strategy_end',
                    'successful_prices': successful_prices,
                    'total_stocks': len(entry_portfolio),
                    'individual_returns': individual_returns,
                    'entry_avg_deviation': entry_portfolio['price_deviation'].mean(),
                    'entry_portfolio_details': entry_portfolio.to_dict('records')
                }
                
                portfolio_data['trades'].append(trade_record)
                
                # Update cumulative return
                period_return = final_return / 100
                portfolio_data['cumulative_return'] *= (1 + period_return)
                
                portfolio_data['active_position'] = None
    
    # Calculate final total returns
    for target in profit_targets:
        target_key = f'target_{target}'
        for portfolio_name in ['portfolio_1', 'portfolio_2', 'portfolio_3']:
            portfolio_data = strategy_results[target_key][portfolio_name]
            portfolio_data['total_return'] = (portfolio_data['cumulative_return'] - 1) * 100
    
    daily_tracking_df = pd.DataFrame(daily_tracking)
    
    return strategy_results, daily_tracking_df

def analyze_strategy_performance(strategy_results):
    """
    Analyze and compare performance across different profit targets and portfolios
    """
    print("\n" + "="*80)
    print("DYNAMIC PROFIT-TAKING STRATEGY PERFORMANCE ANALYSIS")
    print("="*80)
    
    performance_summary = []
    
    # Collect all trade data
    all_trades = []
    
    for target_level in [15, 25, 35]:
        target_key = f'target_{target_level}'
        
        for portfolio_rank in [1, 2, 3]:
            portfolio_name = f'portfolio_{portfolio_rank}'
            
            if target_key in strategy_results and portfolio_name in strategy_results[target_key]:
                portfolio_data = strategy_results[target_key][portfolio_name]
                trades = portfolio_data['trades']
                
                if len(trades) > 0:
                    # Collect individual trade data
                    for trade in trades:
                        trade_record = trade.copy()
                        trade_record['target_level'] = target_level
                        trade_record['portfolio_rank'] = portfolio_rank
                        all_trades.append(trade_record)
                    
                    # Calculate performance metrics
                    trade_returns = [t['return_pct'] for t in trades]
                    trade_durations = [t['days_held'] for t in trades]
                    
                    total_return = portfolio_data['total_return']
                    avg_trade_return = np.mean(trade_returns)
                    volatility = np.std(trade_returns)
                    
                    # Win rate and other metrics
                    profitable_trades = sum(1 for r in trade_returns if r > 0)
                    win_rate = profitable_trades / len(trades) * 100
                    
                    # Target hit rate
                    target_hits = sum(1 for t in trades if t['exit_reason'] == f'profit_target_{target_level}%')
                    target_hit_rate = target_hits / len(trades) * 100
                    
                    # Average hold time
                    avg_hold_time = np.mean(trade_durations)
                    
                    # Sharpe-like ratio (return per unit of volatility)
                    sharpe_ratio = avg_trade_return / volatility if volatility > 0 else 0
                    
                    performance_summary.append({
                        'target_level': target_level,
                        'portfolio_rank': portfolio_rank,
                        'total_return': total_return,
                        'num_trades': len(trades),
                        'avg_trade_return': avg_trade_return,
                        'volatility': volatility,
                        'sharpe_ratio': sharpe_ratio,
                        'win_rate': win_rate,
                        'target_hit_rate': target_hit_rate,
                        'avg_hold_time': avg_hold_time,
                        'best_trade': max(trade_returns),
                        'worst_trade': min(trade_returns),
                        'total_target_hits': target_hits,
                        'max_hold_exits': sum(1 for t in trades if t['exit_reason'] == 'max_hold_period')
                    })
    
    performance_df = pd.DataFrame(performance_summary)
    all_trades_df = pd.DataFrame(all_trades)
    
    # Print summary table
    if len(performance_df) > 0:
        print(f"\nPERFORMANCE SUMMARY:")
        print("-" * 100)
        print(f"{'Target':<8} {'Portfolio':<10} {'Total Ret':<10} {'# Trades':<9} {'Avg Trade':<10} {'Win Rate':<9} {'Hit Rate':<9} {'Avg Days':<9}")
        print("-" * 100)
        
        for _, row in performance_df.iterrows():
            print(f"{row['target_level']:>6}% {row['portfolio_rank']:>9} "
                  f"{row['total_return']:>9.2f}% {row['num_trades']:>8} "
                  f"{row['avg_trade_return']:>9.2f}% {row['win_rate']:>8.1f}% "
                  f"{row['target_hit_rate']:>8.1f}% {row['avg_hold_time']:>8.1f}")
        
        # Best combinations
        print(f"\n🏆 TOP PERFORMERS:")
        top_performers = performance_df.nlargest(5, 'total_return')
        
        for i, (_, row) in enumerate(top_performers.iterrows()):
            print(f"  #{i+1}: {row['target_level']}% target, Portfolio {row['portfolio_rank']}")
            print(f"       Total return: {row['total_return']:.2f}% ({row['num_trades']} trades)")
            print(f"       Hit rate: {row['target_hit_rate']:.1f}%, Avg hold: {row['avg_hold_time']:.1f} days")
        
        # Compare by target level
        print(f"\n📊 PERFORMANCE BY TARGET LEVEL:")
        for target in [15, 25, 35]:
            target_data = performance_df[performance_df['target_level'] == target]
            if len(target_data) > 0:
                avg_return = target_data['total_return'].mean()
                avg_hit_rate = target_data['target_hit_rate'].mean()
                avg_hold_time = target_data['avg_hold_time'].mean()
                total_trades = target_data['num_trades'].sum()
                
                print(f"  {target}% Target: {avg_return:.2f}% avg return, "
                      f"{avg_hit_rate:.1f}% hit rate, {avg_hold_time:.1f} avg days, "
                      f"{total_trades} total trades")
        
        # Compare by portfolio rank
        print(f"\n🎯 PERFORMANCE BY PORTFOLIO RANK:")
        for rank in [1, 2, 3]:
            rank_data = performance_df[performance_df['portfolio_rank'] == rank]
            if len(rank_data) > 0:
                avg_return = rank_data['total_return'].mean()
                avg_win_rate = rank_data['win_rate'].mean()
                avg_hold_time = rank_data['avg_hold_time'].mean()
                
                print(f"  Portfolio {rank} (rank {rank} underperforming): "
                      f"{avg_return:.2f}% avg return, {avg_win_rate:.1f}% win rate, "
                      f"{avg_hold_time:.1f} avg days")
    
    return performance_df, all_trades_df

def create_comprehensive_visualizations(performance_df, all_trades_df, daily_tracking_df, output_dir):
    """
    Create comprehensive visualizations
    """
    print("\nCreating comprehensive visualizations...")
    
    # Figure 1: Performance comparison
    fig, ((ax1, ax2), (ax3, ax4)) = plt.subplots(2, 2, figsize=(20, 12))
    
    # Total returns by target and portfolio
    if len(performance_df) > 0:
        pivot_returns = performance_df.pivot(index='portfolio_rank', columns='target_level', values='total_return')
        
        im1 = ax1.imshow(pivot_returns.values, cmap='RdYlGn', aspect='auto')
        ax1.set_xticks(range(len(pivot_returns.columns)))
        ax1.set_xticklabels([f'{col}%' for col in pivot_returns.columns])
        ax1.set_yticks(range(len(pivot_returns.index)))
        ax1.set_yticklabels([f'Portfolio {idx}' for idx in pivot_returns.index])
        ax1.set_title('Total Returns by Target Level & Portfolio Rank', fontweight='bold')
        
        # Add text annotations
        for i in range(len(pivot_returns.index)):
            for j in range(len(pivot_returns.columns)):
                value = pivot_returns.iloc[i, j]
                if not np.isnan(value):
                    ax1.text(j, i, f'{value:.1f}%', ha='center', va='center', 
                            fontweight='bold', fontsize=12)
        
        plt.colorbar(im1, ax=ax1, label='Total Return (%)')
    
    # Target hit rates
    if len(performance_df) > 0:
        pivot_hits = performance_df.pivot(index='portfolio_rank', columns='target_level', values='target_hit_rate')
        
        im2 = ax2.imshow(pivot_hits.values, cmap='Blues', aspect='auto', vmin=0, vmax=100)
        ax2.set_xticks(range(len(pivot_hits.columns)))
        ax2.set_xticklabels([f'{col}%' for col in pivot_hits.columns])
        ax2.set_yticks(range(len(pivot_hits.index)))
        ax2.set_yticklabels([f'Portfolio {idx}' for idx in pivot_hits.index])
        ax2.set_title('Target Hit Rate by Target Level & Portfolio Rank', fontweight='bold')
        
        # Add text annotations
        for i in range(len(pivot_hits.index)):
            for j in range(len(pivot_hits.columns)):
                value = pivot_hits.iloc[i, j]
                if not np.isnan(value):
                    ax2.text(j, i, f'{value:.1f}%', ha='center', va='center', 
                            fontweight='bold', fontsize=12)
        
        plt.colorbar(im2, ax=ax2, label='Target Hit Rate (%)')
    
    # Average hold times
    if len(performance_df) > 0:
        pivot_hold = performance_df.pivot(index='portfolio_rank', columns='target_level', values='avg_hold_time')
        
        im3 = ax3.imshow(pivot_hold.values, cmap='plasma', aspect='auto')
        ax3.set_xticks(range(len(pivot_hold.columns)))
        ax3.set_xticklabels([f'{col}%' for col in pivot_hold.columns])
        ax3.set_yticks(range(len(pivot_hold.index)))
        ax3.set_yticklabels([f'Portfolio {idx}' for idx in pivot_hold.index])
        ax3.set_title('Average Hold Time by Target Level & Portfolio Rank', fontweight='bold')
        
        # Add text annotations
        for i in range(len(pivot_hold.index)):
            for j in range(len(pivot_hold.columns)):
                value = pivot_hold.iloc[i, j]
                if not np.isnan(value):
                    ax3.text(j, i, f'{value:.0f}d', ha='center', va='center', 
                            fontweight='bold', fontsize=12)
        
        plt.colorbar(im3, ax=ax3, label='Average Hold Time (days)')
    
    # Trade returns distribution
    if len(all_trades_df) > 0:
        for target in [15, 25, 35]:
            target_trades = all_trades_df[all_trades_df['target_level'] == target]
            if len(target_trades) > 0:
                ax4.hist(target_trades['return_pct'], bins=20, alpha=0.6, 
                        label=f'{target}% Target', density=True)
        
        ax4.set_xlabel('Trade Return (%)')
        ax4.set_ylabel('Density')
        ax4.set_title('Distribution of Trade Returns by Target Level', fontweight='bold')
        ax4.legend()
        ax4.grid(True, alpha=0.3)
        ax4.axvline(x=0, color='red', linestyle='--', alpha=0.5)
    
    plt.tight_layout()
    plt.savefig(f"{output_dir}/dynamic_strategy_performance_analysis.png", dpi=300, bbox_inches='tight')
    plt.close()
    
    # Figure 2: Cumulative performance over time
    if len(daily_tracking_df) > 0:
        fig, axes = plt.subplots(3, 1, figsize=(20, 15))
        
        colors = ['red', 'blue', 'green']
        
        for i, target in enumerate([15, 25, 35]):
            ax = axes[i]
            target_key = f'target_{target}'
            
            for j, portfolio in enumerate(['portfolio_1', 'portfolio_2', 'portfolio_3']):
                col_name = f'{target_key}_{portfolio}_cumulative'
                if col_name in daily_tracking_df.columns:
                    ax.plot(daily_tracking_df['date'], daily_tracking_df[col_name], 
                           linewidth=2, label=f'Portfolio {j+1}', color=colors[j], alpha=0.8)
            
            ax.set_ylabel('Cumulative Value', fontsize=12)
            ax.set_title(f'{target}% Profit Target - Cumulative Performance', fontweight='bold', fontsize=14)
            ax.legend()
            ax.grid(True, alpha=0.3)
            ax.axhline(y=1, color='black', linestyle='-', alpha=0.5)
        
        plt.tight_layout()
        plt.savefig(f"{output_dir}/cumulative_performance_by_target.png", dpi=300, bbox_inches='tight')
        plt.close()
    
    print("  ✅ All visualizations created")

def save_detailed_results(strategy_results, performance_df, all_trades_df, daily_tracking_df, output_dir):
    """
    Save all detailed results
    """
    print("\nSaving detailed results...")
    
    # Save performance summary
    performance_df.to_csv(f"{output_dir}/dynamic_strategy_performance_summary.csv", index=False)
    print(f"  ✅ Performance summary: {len(performance_df)} strategy combinations")
    
    # Save all trades
    all_trades_df.to_csv(f"{output_dir}/all_dynamic_trades.csv", index=False)
    print(f"  ✅ All trades: {len(all_trades_df)} trades")
    
    # Save daily tracking
    daily_tracking_df.to_csv(f"{output_dir}/daily_performance_tracking.csv", index=False)
    print(f"  ✅ Daily tracking: {len(daily_tracking_df)} daily records")
    
    # Save detailed trade breakdown by strategy
    strategy_breakdown = []
    
    for target_level in [15, 25, 35]:
        target_key = f'target_{target_level}'
        
        for portfolio_rank in [1, 2, 3]:
            portfolio_name = f'portfolio_{portfolio_rank}'
            
            if target_key in strategy_results and portfolio_name in strategy_results[target_key]:
                portfolio_data = strategy_results[target_key][portfolio_name]
                
                for trade in portfolio_data['trades']:
                    # Add individual stock returns for this trade
                    trade_detail = trade.copy()
                    trade_detail['target_level'] = target_level
                    trade_detail['portfolio_rank'] = portfolio_rank
                    
                    # Add individual stock details
                    for stock_detail in trade['entry_portfolio_details']:
                        stock_trade = trade_detail.copy()
                        stock_trade.update(stock_detail)
                        strategy_breakdown.append(stock_trade)
    
    if strategy_breakdown:
        strategy_breakdown_df = pd.DataFrame(strategy_breakdown)
        strategy_breakdown_df.to_csv(f"{output_dir}/detailed_trade_breakdown.csv", index=False)
        print(f"  ✅ Detailed breakdown: {len(strategy_breakdown_df)} stock-level records")
    
    # Create summary statistics by target level
    target_summary = []
    
    for target in [15, 25, 35]:
        target_trades = all_trades_df[all_trades_df['target_level'] == target]
        
        if len(target_trades) > 0:
            target_summary.append({
                'target_level': target,
                'total_trades': len(target_trades),
                'avg_return': target_trades['return_pct'].mean(),
                'median_return': target_trades['return_pct'].median(),
                'std_return': target_trades['return_pct'].std(),
                'win_rate': (target_trades['return_pct'] > 0).mean() * 100,
                'target_hit_rate': (target_trades['exit_reason'] == f'profit_target_{target}%').mean() * 100,
                'avg_hold_time': target_trades['days_held'].mean(),
                'median_hold_time': target_trades['days_held'].median(),
                'best_trade': target_trades['return_pct'].max(),
                'worst_trade': target_trades['return_pct'].min(),
                'max_hold_exits': (target_trades['exit_reason'] == 'max_hold_period').sum(),
                'strategy_end_exits': (target_trades['exit_reason'] == 'strategy_end').sum()
            })
    
    if target_summary:
        target_summary_df = pd.DataFrame(target_summary)
        target_summary_df.to_csv(f"{output_dir}/target_level_summary.csv", index=False)
        print(f"  ✅ Target level summary: {len(target_summary_df)} target levels")

def print_comprehensive_insights(performance_df, all_trades_df):
    """
    Print comprehensive insights and recommendations
    """
    print("\n" + "="*80)
    print("DYNAMIC PROFIT-TAKING STRATEGY - COMPREHENSIVE INSIGHTS")
    print("="*80)
    
    if len(performance_df) == 0:
        print("No performance data available for analysis")
        return
    
    # Best overall strategy
    best_strategy = performance_df.loc[performance_df['total_return'].idxmax()]
    print(f"\n🏆 BEST OVERALL STRATEGY:")
    print(f"  Target: {best_strategy['target_level']}% profit target")
    print(f"  Portfolio: {best_strategy['portfolio_rank']} (rank {best_strategy['portfolio_rank']} most underperforming)")
    print(f"  Total Return: {best_strategy['total_return']:.2f}%")
    print(f"  Number of Trades: {best_strategy['num_trades']}")
    print(f"  Average Trade Return: {best_strategy['avg_trade_return']:.2f}%")
    print(f"  Target Hit Rate: {best_strategy['target_hit_rate']:.1f}%")
    print(f"  Average Hold Time: {best_strategy['avg_hold_time']:.1f} days")
    print(f"  Win Rate: {best_strategy['win_rate']:.1f}%")
    
    # Compare vs annual rebalancing baseline (assuming ~15-25% annual return)
    annual_baseline = 20  # Rough estimate from previous analysis
    if best_strategy['total_return'] > annual_baseline:
        outperformance = best_strategy['total_return'] - annual_baseline
        print(f"\n✅ OUTPERFORMANCE vs Annual Rebalancing:")
        print(f"  Dynamic strategy: {best_strategy['total_return']:.2f}%")
        print(f"  Annual baseline: ~{annual_baseline}%")
        print(f"  Outperformance: +{outperformance:.2f}%")
        print(f"  🚀 Dynamic profit-taking appears SUPERIOR!")
    else:
        underperformance = annual_baseline - best_strategy['total_return']
        print(f"\n⚠️ UNDERPERFORMANCE vs Annual Rebalancing:")
        print(f"  Dynamic strategy: {best_strategy['total_return']:.2f}%")
        print(f"  Annual baseline: ~{annual_baseline}%")
        print(f"  Underperformance: -{underperformance:.2f}%")
    
    # Target level analysis
    print(f"\n📊 TARGET LEVEL ANALYSIS:")
    print("-" * 50)
    
    for target in [15, 25, 35]:
        target_data = performance_df[performance_df['target_level'] == target]
        
        if len(target_data) > 0:
            avg_return = target_data['total_return'].mean()
            avg_hit_rate = target_data['target_hit_rate'].mean()
            avg_hold_time = target_data['avg_hold_time'].mean()
            avg_trades = target_data['num_trades'].mean()
            
            print(f"  {target}% Target:")
            print(f"    Average Total Return: {avg_return:.2f}%")
            print(f"    Average Hit Rate: {avg_hit_rate:.1f}%")
            print(f"    Average Hold Time: {avg_hold_time:.1f} days")
            print(f"    Average # Trades: {avg_trades:.1f}")
            
            # Efficiency metric (return per day)
            if avg_hold_time > 0:
                return_per_day = avg_return / (avg_hold_time * avg_trades / 365)
                print(f"    Return Efficiency: {return_per_day:.3f}% per day equivalent")
    
    # Portfolio rank analysis
    print(f"\n🎯 PORTFOLIO RANK ANALYSIS:")
    print("-" * 40)
    
    for rank in [1, 2, 3]:
        rank_data = performance_df[performance_df['portfolio_rank'] == rank]
        
        if len(rank_data) > 0:
            avg_return = rank_data['total_return'].mean()
            avg_win_rate = rank_data['win_rate'].mean()
            avg_volatility = rank_data['volatility'].mean()
            
            rank_description = {
                1: "Most underperforming",
                2: "2nd most underperforming", 
                3: "3rd most underperforming"
            }
            
            print(f"  Portfolio {rank} ({rank_description[rank]}):")
            print(f"    Average Total Return: {avg_return:.2f}%")
            print(f"    Average Win Rate: {avg_win_rate:.1f}%")
            print(f"    Average Volatility: {avg_volatility:.2f}%")
    
    # Trade timing analysis
    if len(all_trades_df) > 0:
        print(f"\n⏱️ TRADE TIMING ANALYSIS:")
        print("-" * 30)
        
        # Analyze exit reasons
        exit_reasons = all_trades_df['exit_reason'].value_counts()
        total_trades = len(all_trades_df)
        
        print(f"  Exit Reason Breakdown ({total_trades} total trades):")
        for reason, count in exit_reasons.items():
            percentage = count / total_trades * 100
            print(f"    {reason}: {count} trades ({percentage:.1f}%)")
        
        # Quick vs slow trades
        quick_trades = all_trades_df[all_trades_df['days_held'] <= 30]
        slow_trades = all_trades_df[all_trades_df['days_held'] > 180]
        
        if len(quick_trades) > 0:
            quick_avg_return = quick_trades['return_pct'].mean()
            print(f"\n  Quick trades (≤30 days): {len(quick_trades)} trades, {quick_avg_return:.2f}% avg return")
        
        if len(slow_trades) > 0:
            slow_avg_return = slow_trades['return_pct'].mean()
            print(f"  Slow trades (>180 days): {len(slow_trades)} trades, {slow_avg_return:.2f}% avg return")
        
        # Seasonal or timing patterns
        all_trades_df['entry_month'] = pd.to_datetime(all_trades_df['entry_date']).dt.month
        monthly_performance = all_trades_df.groupby('entry_month')['return_pct'].agg(['mean', 'count'])
        
        print(f"\n  Entry Month Performance (where count ≥ 3):")
        for month, data in monthly_performance.iterrows():
            if data['count'] >= 3:
                month_name = pd.to_datetime(f'2023-{month:02d}-01').strftime('%B')
                print(f"    {month_name}: {data['mean']:.2f}% avg return ({data['count']} trades)")
    
    # Risk analysis
    print(f"\n⚠️ RISK ANALYSIS:")
    print("-" * 20)
    
    if len(all_trades_df) > 0:
        worst_trade = all_trades_df.loc[all_trades_df['return_pct'].idxmin()]
        best_trade = all_trades_df.loc[all_trades_df['return_pct'].idxmax()]
        
        print(f"  Worst Trade: {worst_trade['return_pct']:.2f}% ({worst_trade['target_level']}% target, Portfolio {worst_trade['portfolio_rank']})")
        print(f"  Best Trade: {best_trade['return_pct']:.2f}% ({best_trade['target_level']}% target, Portfolio {best_trade['portfolio_rank']})")
        
        # Drawdown analysis
        negative_trades = all_trades_df[all_trades_df['return_pct'] < 0]
        if len(negative_trades) > 0:
            avg_loss = negative_trades['return_pct'].mean()
            worst_loss = negative_trades['return_pct'].min()
            loss_rate = len(negative_trades) / len(all_trades_df) * 100
            
            print(f"  Loss Rate: {loss_rate:.1f}% of trades")
            print(f"  Average Loss: {avg_loss:.2f}%")
            print(f"  Maximum Loss: {worst_loss:.2f}%")
    
    # Strategic recommendations
    print(f"\n" + "="*80)
    print("🎯 STRATEGIC RECOMMENDATIONS")
    print("="*80)
    
    # Top 3 strategies
    top_3_strategies = performance_df.nlargest(3, 'total_return')
    
    print(f"\n🏆 TOP 3 RECOMMENDED STRATEGIES:")
    for i, (_, strategy) in enumerate(top_3_strategies.iterrows()):
        print(f"  #{i+1}: {strategy['target_level']}% target + Portfolio {strategy['portfolio_rank']}")
        print(f"       Total Return: {strategy['total_return']:.2f}%")
        print(f"       Hit Rate: {strategy['target_hit_rate']:.1f}%")
        print(f"       Avg Hold: {strategy['avg_hold_time']:.1f} days")
        print(f"       # Trades: {strategy['num_trades']}")
        
        # Efficiency score
        if strategy['avg_hold_time'] > 0 and strategy['num_trades'] > 0:
            annual_equivalent = strategy['total_return'] * (365 / (strategy['avg_hold_time'] * strategy['num_trades']))
            print(f"       Annualized Equivalent: ~{annual_equivalent:.1f}%")
        print()
    
    # Optimal target analysis
    best_15 = performance_df[performance_df['target_level'] == 15]['total_return'].max() if len(performance_df[performance_df['target_level'] == 15]) > 0 else 0
    best_25 = performance_df[performance_df['target_level'] == 25]['total_return'].max() if len(performance_df[performance_df['target_level'] == 25]) > 0 else 0
    best_35 = performance_df[performance_df['target_level'] == 35]['total_return'].max() if len(performance_df[performance_df['target_level'] == 35]) > 0 else 0
    
    print(f"💡 TARGET LEVEL INSIGHTS:")
    print(f"   15% target best performance: {best_15:.2f}%")
    print(f"   25% target best performance: {best_25:.2f}%") 
    print(f"   35% target best performance: {best_35:.2f}%")
    
    optimal_target = 15 if best_15 >= max(best_25, best_35) else (25 if best_25 >= best_35 else 35)
    print(f"   ✅ Optimal target level appears to be: {optimal_target}%")
    
    if optimal_target == 15:
        print(f"   💡 Lower targets capture profits more frequently")
    elif optimal_target == 35:
        print(f"   💡 Higher targets allow for bigger wins despite lower frequency")
    else:
        print(f"   💡 Medium targets provide balanced risk/reward")
    
    # Portfolio rank insights
    portfolio_1_avg = performance_df[performance_df['portfolio_rank'] == 1]['total_return'].mean()
    portfolio_2_avg = performance_df[performance_df['portfolio_rank'] == 2]['total_return'].mean()
    portfolio_3_avg = performance_df[performance_df['portfolio_rank'] == 3]['total_return'].mean()
    
    print(f"\n🎯 PORTFOLIO SELECTION INSIGHTS:")
    print(f"   Most underperforming (Portfolio 1): {portfolio_1_avg:.2f}% avg return")
    print(f"   2nd most underperforming (Portfolio 2): {portfolio_2_avg:.2f}% avg return")
    print(f"   3rd most underperforming (Portfolio 3): {portfolio_3_avg:.2f}% avg return")
    
    if portfolio_1_avg >= max(portfolio_2_avg, portfolio_3_avg):
        print(f"   ✅ Most underperforming stocks deliver best results!")
        print(f"   💡 Stick with the most deviated stocks strategy")
    else:
        best_portfolio = 1 if portfolio_1_avg >= max(portfolio_2_avg, portfolio_3_avg) else (2 if portfolio_2_avg >= portfolio_3_avg else 3)
        print(f"   ⚠️  Portfolio {best_portfolio} performs best on average")
        print(f"   💡 Consider focusing on rank {best_portfolio} underperforming stocks")
    
    print(f"\n🚀 IMPLEMENTATION RECOMMENDATIONS:")
    print(f"   1. Use {optimal_target}% profit target as primary strategy")
    print(f"   2. Focus on Portfolio 1 (most underperforming) stocks")
    print(f"   3. Monitor positions weekly for target achievement")
    print(f"   4. Set maximum hold period of 365 days as safety")
    print(f"   5. Maintain 5 stocks per portfolio for diversification")
    
    if best_strategy['target_hit_rate'] > 50:
        print(f"   ✅ High hit rate ({best_strategy['target_hit_rate']:.1f}%) suggests strategy is reliable")
    else:
        print(f"   ⚠️  Lower hit rate ({best_strategy['target_hit_rate']:.1f}%) - consider risk management")

def main():
    print("DYNAMIC PROFIT-TAKING REBALANCING STRATEGY")
    print("=" * 70)
    print("Strategy: Monitor portfolio performance, take profits at target levels")
    print("Targets: 15%, 25%, 35% profit levels")
    print("Portfolios: Top 3 groups of 5 most underperforming stocks")
    print("Goal: Beat annual rebalancing through dynamic profit capture")
    
    # Setup directories
    v2_dir = "/Users/tim/IWLS-OPTIONS/IWLS_ANALYSIS_V2"
    
    if not os.path.exists(v2_dir):
        print("❌ IWLS_ANALYSIS_V2 directory not found.")
        return
    
    # Create output directory
    output_dir = os.path.join(v2_dir, "DYNAMIC_PROFIT_TAKING_ANALYSIS")
    os.makedirs(output_dir, exist_ok=True)
    
    print(f"\nOutput directory: {output_dir}")
    
    # Load all IWLS data
    all_data = load_all_iwls_data(v2_dir)
    
    if len(all_data) < 15:
        print(f"❌ Insufficient assets loaded ({len(all_data)}). Need at least 15 for analysis.")
        return
    
    # Determine date range
    min_dates = []
    max_dates = []
    
    for asset_name, df in all_data.items():
        min_dates.append(df['date'].min())
        max_dates.append(df['date'].max())
    
    strategy_start = max(min_dates)
    strategy_end = min(max_dates)
    
    # Use reasonable timeframe (5-7 years)
    strategy_duration = min(timedelta(days=365 * 6), strategy_end - strategy_start)
    strategy_end = strategy_start + strategy_duration
    
    print(f"\nStrategy parameters:")
    print(f"  Date range: {strategy_start.strftime('%Y-%m-%d')} to {strategy_end.strftime('%Y-%m-%d')}")
    print(f"  Duration: {(strategy_end - strategy_start).days} days ({(strategy_end - strategy_start).days/365.25:.1f} years)")
    print(f"  Profit targets: 15%, 25%, 35%")
    print(f"  Maximum hold period: 365 days")
    print(f"  Portfolio size: 5 stocks each")
    print(f"  Check frequency: Weekly")
    
    # Run the dynamic strategy
    strategy_results, daily_tracking_df = run_dynamic_profit_taking_strategy(
        all_data, strategy_start, strategy_end, 
        profit_targets=[15, 25, 35], max_hold_days=365
    )
    
    # Analyze results
    performance_df, all_trades_df = analyze_strategy_performance(strategy_results)
    
    # Create visualizations
    create_comprehensive_visualizations(performance_df, all_trades_df, daily_tracking_df, output_dir)
    
    # Save detailed results
    save_detailed_results(strategy_results, performance_df, all_trades_df, daily_tracking_df, output_dir)
    
    # Print insights and recommendations
    print_comprehensive_insights(performance_df, all_trades_df)
    
    print(f"\n" + "="*70)
    print("DYNAMIC PROFIT-TAKING ANALYSIS COMPLETE")
    print("="*70)
    print(f"Results saved to: {output_dir}")
    print("\nFiles created:")
    print("  📄 dynamic_strategy_performance_summary.csv (strategy comparison)")
    print("  📄 all_dynamic_trades.csv (every trade with details)")
    print("  📄 daily_performance_tracking.csv (daily portfolio values)")
    print("  📄 detailed_trade_breakdown.csv (stock-level breakdown)")
    print("  📄 target_level_summary.csv (performance by target level)")
    print("  📊 dynamic_strategy_performance_analysis.png (main analysis)")
    print("  📊 cumulative_performance_by_target.png (time series performance)")
    
    print(f"\n🎯 Key questions answered:")
    print(f"   • Does dynamic profit-taking beat annual rebalancing?")
    print(f"   • Which profit target level works best (15%, 25%, 35%)?")
    print(f"   • Do most underperforming stocks still perform best?")
    print(f"   • What's the optimal hold time vs profit target trade-off?")
    print(f"   • How often do we actually hit profit targets?")
    
    if len(performance_df) > 0:
        best_return = performance_df['total_return'].max()
        best_strategy = performance_df.loc[performance_df['total_return'].idxmax()]
        
        print(f"\n🏆 BEST STRATEGY SUMMARY:")
        print(f"   Target: {best_strategy['target_level']}% profit target")
        print(f"   Portfolio: {best_strategy['portfolio_rank']} (most underperforming)")
        print(f"   Total Return: {best_return:.2f}%")
        print(f"   Hit Rate: {best_strategy['target_hit_rate']:.1f}%")
        print(f"   Avg Hold: {best_strategy['avg_hold_time']:.1f} days")

if __name__ == "__main__":
    main()