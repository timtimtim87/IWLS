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
                  not d.startswith('REBALANCING') and not d.startswith('GROWTH_RATE') and
                  not d.startswith('DYNAMIC')]
    
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

def select_portfolios_by_size(available_df, portfolio_sizes=[3, 6, 9]):
    """
    Select portfolios of different sizes (3, 6, 9 stocks) of most underperforming stocks
    """
    # Focus on underperforming stocks (negative deviation)
    underperforming = available_df[available_df['price_deviation'] < 0].copy()
    
    max_size = max(portfolio_sizes)
    if len(underperforming) < max_size:
        print(f"    Warning: Only {len(underperforming)} underperforming stocks available")
        # If not enough underperforming, use all stocks sorted by deviation
        sorted_df = available_df.sort_values('price_deviation', ascending=True)
    else:
        # Sort underperforming stocks by most negative deviation
        sorted_df = underperforming.sort_values('price_deviation', ascending=True)
    
    portfolios = {}
    for size in portfolio_sizes:
        if len(sorted_df) >= size:
            portfolio_stocks = sorted_df.iloc[:size].copy()
            portfolio_name = f'portfolio_{size}_stocks'
            portfolios[portfolio_name] = portfolio_stocks
            
            # Add portfolio info
            portfolios[portfolio_name]['portfolio_size'] = size
            
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

def run_portfolio_size_strategy(all_data, start_date, end_date, 
                               profit_target=20, portfolio_sizes=[3, 6, 9], max_hold_days=365):
    """
    Run portfolio size comparison strategy with 20% profit target
    """
    print(f"\nRunning portfolio size comparison strategy from {start_date} to {end_date}")
    print(f"Profit target: {profit_target}%")
    print(f"Portfolio sizes: {portfolio_sizes} stocks")
    print(f"Maximum hold period: {max_hold_days} days")
    
    strategy_results = {}
    
    # Initialize results for each portfolio size
    for size in portfolio_sizes:
        strategy_results[f'portfolio_{size}_stocks'] = {
            'trades': [], 
            'cumulative_return': 1.0, 
            'total_return': 0, 
            'active_position': None,
            'portfolio_size': size
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
        
        max_required = max(portfolio_sizes)
        if len(available_assets) < max_required:
            current_date += timedelta(days=7)
            continue
        
        # For each portfolio size
        for size in portfolio_sizes:
            portfolio_name = f'portfolio_{size}_stocks'
            portfolio_data = strategy_results[portfolio_name]
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
                
                if current_return >= profit_target:
                    should_exit = True
                    exit_reason = f"profit_target_{profit_target}%"
                elif days_held >= max_hold_days:
                    should_exit = True
                    exit_reason = "max_hold_period"
                
                if should_exit:
                    # Close position
                    trade_id += 1
                    
                    trade_record = {
                        'trade_id': trade_id,
                        'portfolio_size': size,
                        'entry_date': entry_date,
                        'exit_date': current_date,
                        'days_held': days_held,
                        'return_pct': current_return,
                        'exit_reason': exit_reason,
                        'successful_prices': successful_prices,
                        'total_stocks': len(entry_portfolio),
                        'individual_returns': individual_returns,
                        'entry_avg_deviation': entry_portfolio['price_deviation'].mean(),
                        'entry_min_deviation': entry_portfolio['price_deviation'].min(),
                        'entry_max_deviation': entry_portfolio['price_deviation'].max(),
                        'entry_std_deviation': entry_portfolio['price_deviation'].std(),
                        'entry_portfolio_details': entry_portfolio.to_dict('records')
                    }
                    
                    portfolio_data['trades'].append(trade_record)
                    
                    # Update cumulative return
                    period_return = current_return / 100
                    portfolio_data['cumulative_return'] *= (1 + period_return)
                    
                    # Clear active position
                    portfolio_data['active_position'] = None
                    
                    print(f"    {portfolio_name}: EXIT - {current_return:.2f}% after {days_held} days ({exit_reason})")
            
            # If no active position, look for entry
            if portfolio_data['active_position'] is None:
                # Select new portfolio
                portfolios = select_portfolios_by_size(available_assets, [size])
                
                if portfolio_name in portfolios:
                    new_portfolio = portfolios[portfolio_name]
                    
                    # Enter new position
                    portfolio_data['active_position'] = {
                        'entry_date': current_date,
                        'portfolio_df': new_portfolio
                    }
                    
                    avg_deviation = new_portfolio['price_deviation'].mean()
                    min_deviation = new_portfolio['price_deviation'].min()
                    print(f"    {portfolio_name}: ENTER - {len(new_portfolio)} stocks, avg deviation: {avg_deviation:.2f}%, min: {min_deviation:.2f}%")
        
        # Record daily tracking data
        daily_record = {'date': current_date}
        
        for size in portfolio_sizes:
            portfolio_name = f'portfolio_{size}_stocks'
            portfolio_data = strategy_results[portfolio_name]
            daily_record[f'{portfolio_name}_cumulative'] = portfolio_data['cumulative_return']
        
        daily_tracking.append(daily_record)
        
        # Move to next check date
        current_date += timedelta(days=7)
    
    # Close any remaining positions at end date
    for size in portfolio_sizes:
        portfolio_name = f'portfolio_{size}_stocks'
        portfolio_data = strategy_results[portfolio_name]
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
                'portfolio_size': size,
                'entry_date': entry_date,
                'exit_date': end_date,
                'days_held': days_held,
                'return_pct': final_return,
                'exit_reason': 'strategy_end',
                'successful_prices': successful_prices,
                'total_stocks': len(entry_portfolio),
                'individual_returns': individual_returns,
                'entry_avg_deviation': entry_portfolio['price_deviation'].mean(),
                'entry_min_deviation': entry_portfolio['price_deviation'].min(),
                'entry_max_deviation': entry_portfolio['price_deviation'].max(),
                'entry_std_deviation': entry_portfolio['price_deviation'].std(),
                'entry_portfolio_details': entry_portfolio.to_dict('records')
            }
            
            portfolio_data['trades'].append(trade_record)
            
            # Update cumulative return
            period_return = final_return / 100
            portfolio_data['cumulative_return'] *= (1 + period_return)
            
            portfolio_data['active_position'] = None
    
    # Calculate final total returns
    for size in portfolio_sizes:
        portfolio_name = f'portfolio_{size}_stocks'
        portfolio_data = strategy_results[portfolio_name]
        portfolio_data['total_return'] = (portfolio_data['cumulative_return'] - 1) * 100
    
    daily_tracking_df = pd.DataFrame(daily_tracking)
    
    return strategy_results, daily_tracking_df

def analyze_portfolio_size_performance(strategy_results, portfolio_sizes):
    """
    Analyze and compare performance across different portfolio sizes
    """
    print("\n" + "="*80)
    print("PORTFOLIO SIZE COMPARISON - 20% PROFIT TARGET ANALYSIS")
    print("="*80)
    
    performance_summary = []
    all_trades = []
    
    for size in portfolio_sizes:
        portfolio_name = f'portfolio_{size}_stocks'
        
        if portfolio_name in strategy_results:
            portfolio_data = strategy_results[portfolio_name]
            trades = portfolio_data['trades']
            
            if len(trades) > 0:
                # Collect individual trade data
                for trade in trades:
                    trade_record = trade.copy()
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
                target_hits = sum(1 for t in trades if t['exit_reason'] == f'profit_target_20%')
                target_hit_rate = target_hits / len(trades) * 100
                
                # Average hold time
                avg_hold_time = np.mean(trade_durations)
                
                # Sharpe-like ratio
                sharpe_ratio = avg_trade_return / volatility if volatility > 0 else 0
                
                # Calculate average diversification metrics
                avg_deviation_range = np.mean([t['entry_max_deviation'] - t['entry_min_deviation'] for t in trades])
                avg_std_deviation = np.mean([t['entry_std_deviation'] for t in trades if not np.isnan(t['entry_std_deviation'])])
                
                performance_summary.append({
                    'portfolio_size': size,
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
                    'max_hold_exits': sum(1 for t in trades if t['exit_reason'] == 'max_hold_period'),
                    'avg_deviation_range': avg_deviation_range,
                    'avg_std_deviation': avg_std_deviation
                })
    
    performance_df = pd.DataFrame(performance_summary)
    all_trades_df = pd.DataFrame(all_trades)
    
    # Print summary table
    if len(performance_df) > 0:
        print(f"\nPERFORMANCE SUMMARY (20% Profit Target):")
        print("-" * 110)
        print(f"{'Size':<6} {'Total Return':<12} {'# Trades':<9} {'Avg Trade':<10} {'Win Rate':<9} {'Hit Rate':<9} {'Avg Days':<9} {'Volatility':<10}")
        print("-" * 110)
        
        for _, row in performance_df.iterrows():
            print(f"{row['portfolio_size']:>4} {row['total_return']:>11.2f}% {row['num_trades']:>8} "
                  f"{row['avg_trade_return']:>9.2f}% {row['win_rate']:>8.1f}% "
                  f"{row['target_hit_rate']:>8.1f}% {row['avg_hold_time']:>8.1f} {row['volatility']:>9.2f}%")
        
        # Rank by performance
        print(f"\n🏆 RANKING BY TOTAL RETURN:")
        ranked_performance = performance_df.sort_values('total_return', ascending=False)
        
        for i, (_, row) in enumerate(ranked_performance.iterrows()):
            print(f"  #{i+1}: {row['portfolio_size']} stocks - {row['total_return']:.2f}% total return")
            print(f"       {row['num_trades']} trades, {row['target_hit_rate']:.1f}% hit rate, {row['avg_hold_time']:.1f} avg days")
        
        # Key insights
        print(f"\n📊 KEY INSIGHTS:")
        
        best_size = ranked_performance.iloc[0]['portfolio_size']
        best_return = ranked_performance.iloc[0]['total_return']
        
        print(f"  🥇 Best performing size: {best_size} stocks ({best_return:.2f}% return)")
        
        # Compare concentration vs diversification
        size_3_data = performance_df[performance_df['portfolio_size'] == 3]
        size_9_data = performance_df[performance_df['portfolio_size'] == 9]
        
        if len(size_3_data) > 0 and len(size_9_data) > 0:
            concentration_return = size_3_data.iloc[0]['total_return']
            diversified_return = size_9_data.iloc[0]['total_return']
            concentration_vol = size_3_data.iloc[0]['volatility']
            diversified_vol = size_9_data.iloc[0]['volatility']
            
            print(f"\n  🎯 CONCENTRATION vs DIVERSIFICATION:")
            print(f"    3 stocks (concentrated): {concentration_return:.2f}% return, {concentration_vol:.2f}% volatility")
            print(f"    9 stocks (diversified): {diversified_return:.2f}% return, {diversified_vol:.2f}% volatility")
            
            if concentration_return > diversified_return:
                advantage = concentration_return - diversified_return
                print(f"    ✅ Concentration advantage: +{advantage:.2f}%")
                if concentration_vol > diversified_vol:
                    print(f"    ⚠️  But with higher volatility (+{concentration_vol - diversified_vol:.2f}%)")
                else:
                    print(f"    🎯 With lower volatility (-{diversified_vol - concentration_vol:.2f}%)")
            else:
                advantage = diversified_return - concentration_return
                print(f"    ✅ Diversification advantage: +{advantage:.2f}%")
                print(f"    📉 With lower volatility (-{concentration_vol - diversified_vol:.2f}%)")
        
        # Hit rate analysis
        print(f"\n  🎯 TARGET HIT RATE ANALYSIS:")
        for _, row in performance_df.iterrows():
            efficiency = row['target_hit_rate'] / row['avg_hold_time'] * 30  # Hits per month equivalent
            print(f"    {row['portfolio_size']} stocks: {row['target_hit_rate']:.1f}% hit rate ({efficiency:.2f} hits/month equivalent)")
        
        # Risk-adjusted returns
        print(f"\n  📊 RISK-ADJUSTED PERFORMANCE (Sharpe Ratios):")
        sharpe_ranked = performance_df.sort_values('sharpe_ratio', ascending=False)
        for i, (_, row) in enumerate(sharpe_ranked.iterrows()):
            print(f"    #{i+1}: {row['portfolio_size']} stocks - Sharpe: {row['sharpe_ratio']:.3f}")
    
    return performance_df, all_trades_df

def create_portfolio_size_visualizations(performance_df, all_trades_df, daily_tracking_df, portfolio_sizes, output_dir):
    """
    Create visualizations for portfolio size comparison
    """
    print("\nCreating portfolio size comparison visualizations...")
    
    # Figure 1: Performance comparison across portfolio sizes
    fig, ((ax1, ax2), (ax3, ax4)) = plt.subplots(2, 2, figsize=(20, 12))
    
    if len(performance_df) > 0:
        # Total returns by portfolio size
        colors = ['darkred', 'steelblue', 'darkgreen'][:len(performance_df)]
        
        bars1 = ax1.bar(range(len(performance_df)), performance_df['total_return'], 
                       color=colors, alpha=0.8)
        ax1.set_xticks(range(len(performance_df)))
        ax1.set_xticklabels([f'{size} stocks' for size in performance_df['portfolio_size']])
        ax1.set_ylabel('Total Return (%)')
        ax1.set_title('Total Returns by Portfolio Size (20% Target)', fontweight='bold', fontsize=14)
        ax1.grid(True, alpha=0.3)
        ax1.axhline(y=0, color='black', linestyle='-', alpha=0.5)
        
        # Add value labels
        for i, (bar, value, trades) in enumerate(zip(bars1, performance_df['total_return'], 
                                                    performance_df['num_trades'])):
            ax1.text(bar.get_x() + bar.get_width()/2., 
                    bar.get_height() + (max(performance_df['total_return']) * 0.02 if value >= 0 else max(performance_df['total_return']) * -0.02),
                    f'{value:.1f}%\n({trades} trades)', ha='center', 
                    va='bottom' if value >= 0 else 'top', fontweight='bold', fontsize=11)
        
        # Target hit rates
        bars2 = ax2.bar(range(len(performance_df)), performance_df['target_hit_rate'], 
                       color=colors, alpha=0.8)
        ax2.set_xticks(range(len(performance_df)))
        ax2.set_xticklabels([f'{size} stocks' for size in performance_df['portfolio_size']])
        ax2.set_ylabel('Target Hit Rate (%)')
        ax2.set_title('20% Target Hit Rate by Portfolio Size', fontweight='bold', fontsize=14)
        ax2.grid(True, alpha=0.3)
        ax2.axhline(y=50, color='red', linestyle='--', alpha=0.5, label='50% baseline')
        ax2.legend()
        
        # Add value labels
        for bar, value in zip(bars2, performance_df['target_hit_rate']):
            ax2.text(bar.get_x() + bar.get_width()/2., bar.get_height() + 1,
                    f'{value:.1f}%', ha='center', va='bottom', fontweight='bold', fontsize=11)
        
        # Risk vs Return scatter
        ax3.scatter(performance_df['volatility'], performance_df['total_return'], 
                   c=performance_df['portfolio_size'], cmap='viridis', s=200, alpha=0.8)
        
        for i, row in performance_df.iterrows():
            ax3.annotate(f"{row['portfolio_size']} stocks", 
                        (row['volatility'], row['total_return']),
                        xytext=(5, 5), textcoords='offset points', 
                        fontweight='bold', fontsize=12)
        
        ax3.set_xlabel('Volatility (%)')
        ax3.set_ylabel('Total Return (%)')
        ax3.set_title('Risk vs Return by Portfolio Size', fontweight='bold', fontsize=14)
        ax3.grid(True, alpha=0.3)
        
        # Average hold times
        bars4 = ax4.bar(range(len(performance_df)), performance_df['avg_hold_time'], 
                       color=colors, alpha=0.8)
        ax4.set_xticks(range(len(performance_df)))
        ax4.set_xticklabels([f'{size} stocks' for size in performance_df['portfolio_size']])
        ax4.set_ylabel('Average Hold Time (days)')
        ax4.set_title('Average Hold Time by Portfolio Size', fontweight='bold', fontsize=14)
        ax4.grid(True, alpha=0.3)
        
        # Add value labels
        for bar, value in zip(bars4, performance_df['avg_hold_time']):
            ax4.text(bar.get_x() + bar.get_width()/2., bar.get_height() + max(performance_df['avg_hold_time'])*0.01,
                    f'{value:.1f}d', ha='center', va='bottom', fontweight='bold', fontsize=11)
    
    plt.tight_layout()
    plt.savefig(f"{output_dir}/portfolio_size_performance_comparison.png", dpi=300, bbox_inches='tight')
    plt.close()
    
    # Figure 2: Cumulative performance over time
    if len(daily_tracking_df) > 0:
        fig, ax = plt.subplots(1, 1, figsize=(16, 8))
        
        colors = ['darkred', 'steelblue', 'darkgreen']
        
        for i, size in enumerate(portfolio_sizes):
            col_name = f'portfolio_{size}_stocks_cumulative'
            if col_name in daily_tracking_df.columns:
                ax.plot(daily_tracking_df['date'], daily_tracking_df[col_name], 
                       linewidth=3, label=f'{size} stocks', color=colors[i], alpha=0.8)
        
        ax.set_ylabel('Cumulative Value (Starting at 1.0)', fontsize=12)
        ax.set_xlabel('Date', fontsize=12)
        ax.set_title('Cumulative Performance Over Time by Portfolio Size (20% Target)', 
                    fontweight='bold', fontsize=14)
        ax.legend(fontsize=12)
        ax.grid(True, alpha=0.3)
        ax.axhline(y=1, color='black', linestyle='-', alpha=0.5)
        
        plt.tight_layout()
        plt.savefig(f"{output_dir}/cumulative_performance_by_size.png", dpi=300, bbox_inches='tight')
        plt.close()
    
    # Figure 3: Trade returns distribution
    if len(all_trades_df) > 0:
        fig, (ax1, ax2) = plt.subplots(1, 2, figsize=(20, 6))
        
        # Distribution of returns by portfolio size
        for i, size in enumerate(portfolio_sizes):
            size_trades = all_trades_df[all_trades_df['portfolio_size'] == size]
            if len(size_trades) > 0:
                ax1.hist(size_trades['return_pct'], bins=15, alpha=0.6, 
                        label=f'{size} stocks', color=colors[i], density=True)
        
        ax1.set_xlabel('Trade Return (%)')
        ax1.set_ylabel('Density')
        ax1.set_title('Distribution of Trade Returns by Portfolio Size', fontweight='bold', fontsize=14)
        ax1.legend()
        ax1.grid(True, alpha=0.3)
        ax1.axvline(x=0, color='red', linestyle='--', alpha=0.5)
        ax1.axvline(x=20, color='green', linestyle='--', alpha=0.5, label='20% Target')
        
        # Box plot of returns by size
        box_data = []
        box_labels = []
        
        for size in portfolio_sizes:
            size_trades = all_trades_df[all_trades_df['portfolio_size'] == size]
            if len(size_trades) > 0:
                box_data.append(size_trades['return_pct'].values)
                box_labels.append(f'{size} stocks')
        
        if box_data:
            bp = ax2.boxplot(box_data, labels=box_labels, patch_artist=True)
            for i, patch in enumerate(bp['boxes']):
                patch.set_facecolor(colors[i])
                patch.set_alpha(0.7)
            
            ax2.set_ylabel('Trade Return (%)')
            ax2.set_title('Trade Return Distribution by Portfolio Size', fontweight='bold', fontsize=14)
            ax2.grid(True, alpha=0.3)
            ax2.axhline(y=0, color='red', linestyle='--', alpha=0.5)
            ax2.axhline(y=20, color='green', linestyle='--', alpha=0.5, label='20% Target')
            ax2.legend()
        
        plt.tight_layout()
        plt.savefig(f"{output_dir}/trade_returns_distribution_by_size.png", dpi=300, bbox_inches='tight')
        plt.close()
    
    print("  ✅ All visualizations created")

def save_portfolio_size_results(strategy_results, performance_df, all_trades_df, daily_tracking_df, output_dir):
    """
    Save all results for portfolio size comparison
    """
    print("\nSaving portfolio size comparison results...")
    
    # Save performance summary
    performance_df.to_csv(f"{output_dir}/portfolio_size_performance_summary.csv", index=False)
    print(f"  ✅ Performance summary: {len(performance_df)} portfolio sizes")
    
    # Save all trades
    all_trades_df.to_csv(f"{output_dir}/all_portfolio_size_trades.csv", index=False)
    print(f"  ✅ All trades: {len(all_trades_df)} trades")
    
    # Save daily tracking
    daily_tracking_df.to_csv(f"{output_dir}/daily_portfolio_size_tracking.csv", index=False)
    print(f"  ✅ Daily tracking: {len(daily_tracking_df)} daily records")
    
    # Save detailed breakdown by portfolio size
    size_breakdown = []
    
    for size in [3, 6, 9]:
        portfolio_name = f'portfolio_{size}_stocks'
        
        if portfolio_name in strategy_results:
            portfolio_data = strategy_results[portfolio_name]
            
            for trade in portfolio_data['trades']:
                # Add individual stock details for this trade
                trade_detail = trade.copy()
                trade_detail['portfolio_size'] = size
                
                # Add individual stock details
                for stock_detail in trade['entry_portfolio_details']:
                    stock_trade = trade_detail.copy()
                    stock_trade.update(stock_detail)
                    size_breakdown.append(stock_trade)
    
    if size_breakdown:
        size_breakdown_df = pd.DataFrame(size_breakdown)
        size_breakdown_df.to_csv(f"{output_dir}/detailed_size_breakdown.csv", index=False)
        print(f"  ✅ Detailed breakdown: {len(size_breakdown_df)} stock-level records")
    
    # Create diversification analysis
    diversification_analysis = []
    
    for size in [3, 6, 9]:
        size_trades = all_trades_df[all_trades_df['portfolio_size'] == size]
        
        if len(size_trades) > 0:
            # Calculate concentration metrics
            avg_deviation_range = size_trades['entry_max_deviation'] - size_trades['entry_min_deviation']
            
            diversification_analysis.append({
                'portfolio_size': size,
                'total_trades': len(size_trades),
                'avg_return': size_trades['return_pct'].mean(),
                'median_return': size_trades['return_pct'].median(),
                'std_return': size_trades['return_pct'].std(),
                'win_rate': (size_trades['return_pct'] > 0).mean() * 100,
                'target_hit_rate': (size_trades['exit_reason'] == 'profit_target_20%').mean() * 100,
                'avg_hold_time': size_trades['days_held'].mean(),
                'median_hold_time': size_trades['days_held'].median(),
                'best_trade': size_trades['return_pct'].max(),
                'worst_trade': size_trades['return_pct'].min(),
                'avg_entry_deviation': size_trades['entry_avg_deviation'].mean(),
                'avg_deviation_range': avg_deviation_range.mean(),
                'avg_deviation_std': size_trades['entry_std_deviation'].mean(),
                'concentration_score': 1 / size  # Simple concentration metric
            })
    
    if diversification_analysis:
        diversification_df = pd.DataFrame(diversification_analysis)
        diversification_df.to_csv(f"{output_dir}/diversification_analysis.csv", index=False)
        print(f"  ✅ Diversification analysis saved")

def print_portfolio_size_insights(performance_df, all_trades_df):
    """
    Print comprehensive insights for portfolio size comparison
    """
    print("\n" + "="*80)
    print("PORTFOLIO SIZE COMPARISON - COMPREHENSIVE INSIGHTS (20% Target)")
    print("="*80)
    
    if len(performance_df) == 0:
        print("No performance data available for analysis")
        return
    
    # Best portfolio size
    best_size_row = performance_df.loc[performance_df['total_return'].idxmax()]
    print(f"\n🏆 BEST PORTFOLIO SIZE:")
    print(f"  Size: {best_size_row['portfolio_size']} stocks")
    print(f"  Total Return: {best_size_row['total_return']:.2f}%")
    print(f"  Number of Trades: {best_size_row['num_trades']}")
    print(f"  Average Trade Return: {best_size_row['avg_trade_return']:.2f}%")
    print(f"  Target Hit Rate: {best_size_row['target_hit_rate']:.1f}%")
    print(f"  Average Hold Time: {best_size_row['avg_hold_time']:.1f} days")
    print(f"  Win Rate: {best_size_row['win_rate']:.1f}%")
    print(f"  Volatility: {best_size_row['volatility']:.2f}%")
    
    # Detailed comparison
    print(f"\n📊 DETAILED SIZE COMPARISON:")
    print("-" * 80)
    print(f"{'Size':<6} {'Return':<8} {'Trades':<7} {'Hit Rate':<9} {'Avg Days':<9} {'Volatility':<10} {'Sharpe':<8}")
    print("-" * 80)
    
    for _, row in performance_df.iterrows():
        print(f"{row['portfolio_size']:>4} {row['total_return']:>7.2f}% {row['num_trades']:>6} "
              f"{row['target_hit_rate']:>8.1f}% {row['avg_hold_time']:>8.1f} "
              f"{row['volatility']:>9.2f}% {row['sharpe_ratio']:>7.3f}")
    
    # Concentration vs Diversification analysis
    print(f"\n🎯 CONCENTRATION vs DIVERSIFICATION ANALYSIS:")
    
    # Compare 3 vs 9 stocks (most concentrated vs most diversified)
    size_3_data = performance_df[performance_df['portfolio_size'] == 3]
    size_9_data = performance_df[performance_df['portfolio_size'] == 9]
    
    if len(size_3_data) > 0 and len(size_9_data) > 0:
        conc_return = size_3_data.iloc[0]['total_return']
        div_return = size_9_data.iloc[0]['total_return']
        conc_vol = size_3_data.iloc[0]['volatility']
        div_vol = size_9_data.iloc[0]['volatility']
        conc_hit_rate = size_3_data.iloc[0]['target_hit_rate']
        div_hit_rate = size_9_data.iloc[0]['target_hit_rate']
        
        print(f"  CONCENTRATION (3 stocks):")
        print(f"    • Total Return: {conc_return:.2f}%")
        print(f"    • Volatility: {conc_vol:.2f}%")
        print(f"    • Hit Rate: {conc_hit_rate:.1f}%")
        print(f"    • Risk-Adj Return: {conc_return/conc_vol:.3f}")
        
        print(f"\n  DIVERSIFICATION (9 stocks):")
        print(f"    • Total Return: {div_return:.2f}%")
        print(f"    • Volatility: {div_vol:.2f}%")
        print(f"    • Hit Rate: {div_hit_rate:.1f}%")
        print(f"    • Risk-Adj Return: {div_return/div_vol:.3f}")
        
        return_diff = conc_return - div_return
        vol_diff = conc_vol - div_vol
        
        print(f"\n  CONCENTRATION vs DIVERSIFICATION:")
        if return_diff > 0:
            print(f"    ✅ Concentration wins by {return_diff:.2f}% return")
        else:
            print(f"    ✅ Diversification wins by {abs(return_diff):.2f}% return")
        
        if vol_diff > 0:
            print(f"    ⚠️  Concentration has {vol_diff:.2f}% higher volatility")
        else:
            print(f"    📈 Concentration has {abs(vol_diff):.2f}% lower volatility")
    
    # Middle ground analysis (6 stocks)
    size_6_data = performance_df[performance_df['portfolio_size'] == 6]
    if len(size_6_data) > 0:
        mid_return = size_6_data.iloc[0]['total_return']
        mid_vol = size_6_data.iloc[0]['volatility']
        mid_hit_rate = size_6_data.iloc[0]['target_hit_rate']
        
        print(f"\n  BALANCED APPROACH (6 stocks):")
        print(f"    • Total Return: {mid_return:.2f}%")
        print(f"    • Volatility: {mid_vol:.2f}%")
        print(f"    • Hit Rate: {mid_hit_rate:.1f}%")
        
        # Compare to extremes
        if len(size_3_data) > 0 and len(size_9_data) > 0:
            if mid_return >= max(conc_return, div_return):
                print(f"    🏆 BALANCED APPROACH WINS!")
            elif mid_return >= min(conc_return, div_return):
                print(f"    📊 Balanced approach shows middle performance")
            else:
                print(f"    📉 Balanced approach underperforms both extremes")
    
    # Efficiency analysis
    print(f"\n⚡ EFFICIENCY ANALYSIS:")
    
    for _, row in performance_df.iterrows():
        # Calculate return per day efficiency
        total_days_invested = row['num_trades'] * row['avg_hold_time']
        if total_days_invested > 0:
            return_per_day = row['total_return'] / total_days_invested * 365  # Annualized
            hit_efficiency = row['target_hit_rate'] / row['avg_hold_time'] * 30  # Hits per month
            
            print(f"  {row['portfolio_size']} stocks:")
            print(f"    • Return efficiency: {return_per_day:.2f}% annualized")
            print(f"    • Hit efficiency: {hit_efficiency:.2f} hits/month equivalent")
            print(f"    • Capital efficiency: {row['total_return']/row['num_trades']:.2f}% per trade")
    
    # Risk analysis by portfolio size
    if len(all_trades_df) > 0:
        print(f"\n⚠️ RISK ANALYSIS BY PORTFOLIO SIZE:")
        
        for size in [3, 6, 9]:
            size_trades = all_trades_df[all_trades_df['portfolio_size'] == size]
            
            if len(size_trades) > 0:
                negative_trades = size_trades[size_trades['return_pct'] < 0]
                loss_rate = len(negative_trades) / len(size_trades) * 100
                avg_loss = negative_trades['return_pct'].mean() if len(negative_trades) > 0 else 0
                worst_loss = negative_trades['return_pct'].min() if len(negative_trades) > 0 else 0
                
                print(f"  {size} stocks:")
                print(f"    • Loss rate: {loss_rate:.1f}% of trades")
                print(f"    • Average loss: {avg_loss:.2f}%")
                print(f"    • Worst loss: {worst_loss:.2f}%")
                
                # Drawdown analysis
                positive_trades = size_trades[size_trades['return_pct'] > 0]
                if len(positive_trades) > 0:
                    avg_gain = positive_trades['return_pct'].mean()
                    gain_loss_ratio = abs(avg_gain / avg_loss) if avg_loss < 0 else float('inf')
                    print(f"    • Gain/Loss ratio: {gain_loss_ratio:.2f}")
    
    # Strategic recommendations
    print(f"\n" + "="*80)
    print("🎯 STRATEGIC RECOMMENDATIONS")
    print("="*80)
    
    # Determine optimal strategy
    best_sharpe = performance_df.loc[performance_df['sharpe_ratio'].idxmax()]
    best_return = performance_df.loc[performance_df['total_return'].idxmax()]
    
    print(f"\n🏆 OPTIMAL STRATEGIES:")
    print(f"  Best Total Return: {best_return['portfolio_size']} stocks ({best_return['total_return']:.2f}%)")
    print(f"  Best Risk-Adjusted: {best_sharpe['portfolio_size']} stocks (Sharpe: {best_sharpe['sharpe_ratio']:.3f})")
    
    if best_return['portfolio_size'] == best_sharpe['portfolio_size']:
        optimal_size = best_return['portfolio_size']
        print(f"  ✅ CLEAR WINNER: {optimal_size} stocks dominates both metrics!")
    else:
        print(f"  ⚖️  Trade-off: Choose based on risk preference")
    
    # Implementation recommendations
    print(f"\n💡 IMPLEMENTATION RECOMMENDATIONS:")
    
    # High hit rate strategy
    best_hit_rate = performance_df.loc[performance_df['target_hit_rate'].idxmax()]
    if best_hit_rate['target_hit_rate'] > 60:
        print(f"  • High Success Rate: {best_hit_rate['portfolio_size']} stocks has {best_hit_rate['target_hit_rate']:.1f}% hit rate")
    
    # Fast turnover strategy
    fastest_turnover = performance_df.loc[performance_df['avg_hold_time'].idxmin()]
    print(f"  • Fastest Turnover: {fastest_turnover['portfolio_size']} stocks with {fastest_turnover['avg_hold_time']:.1f} day avg hold")
    
    # Risk considerations
    lowest_vol = performance_df.loc[performance_df['volatility'].idxmin()]
    print(f"  • Lowest Volatility: {lowest_vol['portfolio_size']} stocks with {lowest_vol['volatility']:.2f}% volatility")
    
    print(f"\n📋 FINAL RECOMMENDATIONS:")
    
    if best_return['portfolio_size'] == 3:
        print(f"  🎯 AGGRESSIVE: Use 3 stocks for maximum concentration and returns")
        print(f"     • Higher returns but increased volatility")
        print(f"     • Suitable for higher risk tolerance")
    elif best_return['portfolio_size'] == 9:
        print(f"  🛡️  CONSERVATIVE: Use 9 stocks for diversified approach")
        print(f"     • Lower volatility with solid returns")
        print(f"     • Suitable for risk-averse investors")
    else:
        print(f"  ⚖️  BALANCED: Use 6 stocks for optimal risk/return balance")
        print(f"     • Middle ground between concentration and diversification")
        print(f"     • Good for moderate risk tolerance")
    
    print(f"\n  📊 Consider your priorities:")
    print(f"     • Maximum returns → {best_return['portfolio_size']} stocks")
    print(f"     • Best risk-adjusted → {best_sharpe['portfolio_size']} stocks")
    print(f"     • Highest hit rate → {best_hit_rate['portfolio_size']} stocks")
    print(f"     • Lowest volatility → {lowest_vol['portfolio_size']} stocks")

def main():
    print("PORTFOLIO SIZE COMPARISON - 20% PROFIT TARGET")
    print("=" * 70)
    print("Strategy: Compare 3, 6, and 9 stock portfolios with 20% profit target")
    print("Analysis: Concentration vs Diversification in underperforming stock selection")
    print("Goal: Find optimal portfolio size for risk/return balance")
    
    # Setup directories
    v2_dir = "/Users/tim/IWLS-OPTIONS/IWLS_ANALYSIS_V2"
    
    if not os.path.exists(v2_dir):
        print("❌ IWLS_ANALYSIS_V2 directory not found.")
        return
    
    # Create output directory
    output_dir = os.path.join(v2_dir, "PORTFOLIO_SIZE_ANALYSIS_20PCT")
    os.makedirs(output_dir, exist_ok=True)
    
    print(f"\nOutput directory: {output_dir}")
    
    # Load all IWLS data
    all_data = load_all_iwls_data(v2_dir)
    
    if len(all_data) < 9:
        print(f"❌ Insufficient assets loaded ({len(all_data)}). Need at least 9 for analysis.")
        return
    
    # Determine date range
    min_dates = []
    max_dates = []
    
    for asset_name, df in all_data.items():
        min_dates.append(df['date'].min())
        max_dates.append(df['date'].max())
    
    strategy_start = max(min_dates)
    strategy_end = min(max_dates)
    
    # Use reasonable timeframe
    strategy_duration = min(timedelta(days=365 * 6), strategy_end - strategy_start)
    strategy_end = strategy_start + strategy_duration
    
    print(f"\nStrategy parameters:")
    print(f"  Date range: {strategy_start.strftime('%Y-%m-%d')} to {strategy_end.strftime('%Y-%m-%d')}")
    print(f"  Duration: {(strategy_end - strategy_start).days} days ({(strategy_end - strategy_start).days/365.25:.1f} years)")
    print(f"  Profit target: 20%")
    print(f"  Portfolio sizes: 3, 6, 9 stocks")
    print(f"  Maximum hold period: 365 days")
    print(f"  Check frequency: Weekly")
    
    # Run the portfolio size strategy
    portfolio_sizes = [3, 6, 9]
    strategy_results, daily_tracking_df = run_portfolio_size_strategy(
        all_data, strategy_start, strategy_end, 
        profit_target=20, portfolio_sizes=portfolio_sizes, max_hold_days=365
    )
    
    # Analyze results
    performance_df, all_trades_df = analyze_portfolio_size_performance(strategy_results, portfolio_sizes)
    
    # Create visualizations
    create_portfolio_size_visualizations(performance_df, all_trades_df, daily_tracking_df, portfolio_sizes, output_dir)
    
    # Save detailed results
    save_portfolio_size_results(strategy_results, performance_df, all_trades_df, daily_tracking_df, output_dir)
    
    # Print insights and recommendations
    print_portfolio_size_insights(performance_df, all_trades_df)
    
    print(f"\n" + "="*70)
    print("PORTFOLIO SIZE ANALYSIS COMPLETE")
    print("="*70)
    print(f"Results saved to: {output_dir}")
    print("\nFiles created:")
    print("  📄 portfolio_size_performance_summary.csv (size comparison)")
    print("  📄 all_portfolio_size_trades.csv (every trade with details)")
    print("  📄 daily_portfolio_size_tracking.csv (daily values)")
    print("  📄 detailed_size_breakdown.csv (stock-level breakdown)")
    print("  📄 diversification_analysis.csv (concentration metrics)")
    print("  📊 portfolio_size_performance_comparison.png (main analysis)")
    print("  📊 cumulative_performance_by_size.png (time series)")
    print("  📊 trade_returns_distribution_by_size.png (return distributions)")
    
    print(f"\n🎯 Key questions answered:")
    print(f"   • Does concentration (3 stocks) beat diversification (9 stocks)?")
    print(f"   • Is 6 stocks the optimal balance point?")
    print(f"   • How does portfolio size affect hit rates and volatility?")
    print(f"   • What's the risk/return trade-off across sizes?")
    print(f"   • Which size provides best capital efficiency?")
    
    if len(performance_df) > 0:
        best_return = performance_df['total_return'].max()
        best_size = performance_df.loc[performance_df['total_return'].idxmax(), 'portfolio_size']
        
        print(f"\n🏆 OPTIMAL PORTFOLIO SIZE:")
        print(f"   Size: {best_size} stocks")
        print(f"   Total Return: {best_return:.2f}%")
        print(f"   Strategy: Focus on {best_size} most underperforming stocks with 20% profit target")

if __name__ == "__main__":
    main()