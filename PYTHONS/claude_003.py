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
                  not d.startswith('FORWARD_RETURNS') and not d.startswith('OLD')]
    
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
                    print(f"  ✅ {asset_name}: {len(df)} data points")
                else:
                    print(f"  ⚠️  {asset_name}: insufficient data ({len(df)} points)")
                    
            except Exception as e:
                print(f"  ❌ {asset_name}: error loading - {e}")
    
    print(f"Successfully loaded {len(all_data)} assets")
    return all_data

def get_available_assets_on_date(all_data, target_date, lookback_days=30):
    """
    Get assets that have valid data around a specific date
    """
    available_assets = []
    
    for asset_name, df in all_data.items():
        # Find data points around the target date (within lookback_days)
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
                'trend_line_value': closest_row['trend_line_value']
            })
    
    return pd.DataFrame(available_assets)

def select_quintile_portfolios(available_df, n_per_quintile=5):
    """
    Select quintile portfolios based on absolute deviation (most underperforming first)
    """
    # Sort by absolute deviation (largest deviations first for underperforming stocks)
    # But we want the MOST NEGATIVE deviations (furthest below trend line)
    underperforming = available_df[available_df['price_deviation'] < 0].copy()
    
    if len(underperforming) < n_per_quintile * 5:
        print(f"    Warning: Only {len(underperforming)} underperforming stocks available")
        # If not enough underperforming, use all stocks sorted by deviation
        sorted_df = available_df.sort_values('price_deviation', ascending=True)
    else:
        # Sort underperforming stocks by most negative deviation (furthest below trend)
        sorted_df = underperforming.sort_values('price_deviation', ascending=True)
    
    quintiles = {}
    for i in range(5):
        start_idx = i * n_per_quintile
        end_idx = (i + 1) * n_per_quintile
        
        if start_idx < len(sorted_df):
            quintile_stocks = sorted_df.iloc[start_idx:end_idx].copy()
            quintiles[f'quintile_{i+1}'] = quintile_stocks
            
            # Add quintile info
            quintiles[f'quintile_{i+1}']['quintile'] = i + 1
            
    return quintiles

def calculate_portfolio_return(portfolio_df, all_data, hold_start_date, hold_end_date):
    """
    Calculate portfolio return for a specific holding period
    """
    if len(portfolio_df) == 0:
        return {
            'return_1year': 0,
            'individual_returns': [],
            'successful_exits': 0,
            'failed_exits': 0,
            'avg_return': 0,
            'median_return': 0,
            'best_return': 0,
            'worst_return': 0,
            'volatility': 0
        }
    
    individual_returns = []
    successful_exits = 0
    failed_exits = 0
    
    for _, stock in portfolio_df.iterrows():
        asset_name = stock['asset']
        entry_price = stock['price']
        
        if asset_name in all_data:
            asset_df = all_data[asset_name]
            
            # Find exit price (closest to hold_end_date)
            exit_mask = (asset_df['date'] >= hold_end_date - timedelta(days=30)) & \
                       (asset_df['date'] <= hold_end_date + timedelta(days=30))
            exit_data = asset_df[exit_mask]
            
            if len(exit_data) > 0:
                # Get closest exit price
                closest_exit_idx = (exit_data['date'] - hold_end_date).abs().idxmin()
                exit_price = exit_data.loc[closest_exit_idx, 'price']
                
                stock_return = ((exit_price / entry_price) - 1) * 100
                individual_returns.append({
                    'asset': asset_name,
                    'entry_price': entry_price,
                    'exit_price': exit_price,
                    'return_pct': stock_return,
                    'entry_deviation': stock['price_deviation']
                })
                successful_exits += 1
            else:
                failed_exits += 1
    
    if len(individual_returns) > 0:
        returns_only = [r['return_pct'] for r in individual_returns]
        portfolio_return = np.mean(returns_only)  # Equal weight portfolio
        
        return {
            'return_1year': portfolio_return,
            'individual_returns': individual_returns,
            'successful_exits': successful_exits,
            'failed_exits': failed_exits,
            'avg_return': np.mean(returns_only),
            'median_return': np.median(returns_only),
            'best_return': np.max(returns_only),
            'worst_return': np.min(returns_only),
            'volatility': np.std(returns_only),
            'stocks_count': len(returns_only)
        }
    else:
        return {
            'return_1year': 0,
            'individual_returns': [],
            'successful_exits': 0,
            'failed_exits': failed_exits,
            'avg_return': 0,
            'median_return': 0,
            'best_return': 0,
            'worst_return': 0,
            'volatility': 0,
            'stocks_count': 0
        }

def run_rebalancing_strategy(all_data, start_date, end_date, rebalance_frequency_days=365):
    """
    Run the rebalancing strategy across all quintiles
    """
    print(f"\nRunning rebalancing strategy from {start_date} to {end_date}")
    print(f"Rebalancing every {rebalance_frequency_days} days")
    
    # Generate rebalancing dates
    current_date = pd.to_datetime(start_date)
    end_date = pd.to_datetime(end_date)
    
    strategy_results = {
        'quintile_1': {'periods': [], 'cumulative_return': 1.0, 'total_return': 0},
        'quintile_2': {'periods': [], 'cumulative_return': 1.0, 'total_return': 0},
        'quintile_3': {'periods': [], 'cumulative_return': 1.0, 'total_return': 0},
        'quintile_4': {'periods': [], 'cumulative_return': 1.0, 'total_return': 0},
        'quintile_5': {'periods': [], 'cumulative_return': 1.0, 'total_return': 0}
    }
    
    all_period_results = []
    period_number = 0
    
    while current_date < end_date:
        period_number += 1
        hold_start_date = current_date
        hold_end_date = min(current_date + timedelta(days=rebalance_frequency_days), end_date)
        
        print(f"\n  Period {period_number}: {hold_start_date.strftime('%Y-%m-%d')} to {hold_end_date.strftime('%Y-%m-%d')}")
        
        # Get available assets on this date
        available_assets = get_available_assets_on_date(all_data, hold_start_date)
        print(f"    Available assets: {len(available_assets)}")
        
        if len(available_assets) < 25:  # Need at least 25 stocks for 5 quintiles of 5 stocks each
            print(f"    Insufficient assets ({len(available_assets)}), skipping period")
            current_date = hold_end_date
            continue
        
        # Select quintile portfolios
        quintiles = select_quintile_portfolios(available_assets)
        
        period_results = {
            'period': period_number,
            'start_date': hold_start_date,
            'end_date': hold_end_date,
            'available_assets': len(available_assets)
        }
        
        # Calculate returns for each quintile
        for quintile_name, portfolio_df in quintiles.items():
            print(f"    {quintile_name}: {len(portfolio_df)} stocks")
            
            portfolio_result = calculate_portfolio_return(
                portfolio_df, all_data, hold_start_date, hold_end_date
            )
            
            period_results[quintile_name] = portfolio_result
            
            # Update cumulative returns
            if quintile_name in strategy_results:
                period_return = portfolio_result['return_1year'] / 100  # Convert to decimal
                strategy_results[quintile_name]['cumulative_return'] *= (1 + period_return)
                strategy_results[quintile_name]['periods'].append({
                    'period': period_number,
                    'start_date': hold_start_date,
                    'end_date': hold_end_date,
                    'return_pct': portfolio_result['return_1year'],
                    'stocks_count': portfolio_result['stocks_count'],
                    'successful_exits': portfolio_result['successful_exits'],
                    'portfolio_details': portfolio_df.to_dict('records'),
                    'individual_returns': portfolio_result['individual_returns']
                })
            
            print(f"      Return: {portfolio_result['return_1year']:.2f}% "
                  f"(successful exits: {portfolio_result['successful_exits']}/{portfolio_result['successful_exits'] + portfolio_result['failed_exits']})")
        
        all_period_results.append(period_results)
        current_date = hold_end_date
    
    # Calculate final total returns
    for quintile_name in strategy_results:
        strategy_results[quintile_name]['total_return'] = (strategy_results[quintile_name]['cumulative_return'] - 1) * 100
    
    return strategy_results, all_period_results

def analyze_strategy_performance(strategy_results, all_period_results):
    """
    Analyze and compare strategy performance across quintiles
    """
    print("\n" + "="*80)
    print("REBALANCING STRATEGY PERFORMANCE ANALYSIS")
    print("="*80)
    
    performance_summary = []
    
    for quintile_name in ['quintile_1', 'quintile_2', 'quintile_3', 'quintile_4', 'quintile_5']:
        if quintile_name in strategy_results:
            quintile_data = strategy_results[quintile_name]
            periods = quintile_data['periods']
            
            if len(periods) > 0:
                period_returns = [p['return_pct'] for p in periods]
                
                # Calculate performance metrics
                total_return = quintile_data['total_return']
                avg_period_return = np.mean(period_returns)
                volatility = np.std(period_returns)
                sharpe_ratio = avg_period_return / volatility if volatility > 0 else 0
                
                # Win rate
                positive_periods = sum(1 for r in period_returns if r > 0)
                win_rate = positive_periods / len(period_returns) * 100
                
                # Max drawdown calculation
                cumulative_values = [1.0]
                for ret in period_returns:
                    cumulative_values.append(cumulative_values[-1] * (1 + ret/100))
                
                peak = cumulative_values[0]
                max_drawdown = 0
                for value in cumulative_values:
                    if value > peak:
                        peak = value
                    drawdown = (peak - value) / peak * 100
                    max_drawdown = max(max_drawdown, drawdown)
                
                performance_summary.append({
                    'quintile': quintile_name,
                    'quintile_rank': int(quintile_name.split('_')[1]),
                    'total_return': total_return,
                    'avg_period_return': avg_period_return,
                    'volatility': volatility,
                    'sharpe_ratio': sharpe_ratio,
                    'win_rate': win_rate,
                    'max_drawdown': max_drawdown,
                    'num_periods': len(periods),
                    'best_period': max(period_returns),
                    'worst_period': min(period_returns)
                })
    
    performance_df = pd.DataFrame(performance_summary)
    
    # Print summary table
    print(f"\nPERFORMANCE SUMMARY:")
    print("-" * 100)
    print(f"{'Quintile':<10} {'Total Return':<12} {'Avg Period':<11} {'Volatility':<10} {'Sharpe':<8} {'Win Rate':<9} {'Max DD':<8}")
    print("-" * 100)
    
    for _, row in performance_df.iterrows():
        print(f"{row['quintile']:<10} {row['total_return']:>11.2f}% {row['avg_period_return']:>10.2f}% "
              f"{row['volatility']:>9.2f}% {row['sharpe_ratio']:>7.2f} {row['win_rate']:>8.1f}% {row['max_drawdown']:>7.2f}%")
    
    # Key insights
    print(f"\nKEY INSIGHTS:")
    print("-" * 40)
    
    best_quintile = performance_df.loc[performance_df['total_return'].idxmax()]
    worst_quintile = performance_df.loc[performance_df['total_return'].idxmin()]
    
    print(f"  Best performing quintile: {best_quintile['quintile']} ({best_quintile['total_return']:.2f}% total return)")
    print(f"  Worst performing quintile: {worst_quintile['quintile']} ({worst_quintile['total_return']:.2f}% total return)")
    print(f"  Performance spread: {best_quintile['total_return'] - worst_quintile['total_return']:.2f}%")
    
    # Check if top quintile (most underperforming) is best
    top_quintile = performance_df[performance_df['quintile'] == 'quintile_1'].iloc[0]
    print(f"\n  Top quintile (most underperforming) performance:")
    print(f"    Total return: {top_quintile['total_return']:.2f}%")
    print(f"    Average period return: {top_quintile['avg_period_return']:.2f}%")
    print(f"    Win rate: {top_quintile['win_rate']:.1f}%")
    print(f"    Sharpe ratio: {top_quintile['sharpe_ratio']:.2f}")
    
    # Rank by total return
    ranked_performance = performance_df.sort_values('total_return', ascending=False)
    top_quintile_rank = ranked_performance[ranked_performance['quintile'] == 'quintile_1'].index[0] + 1
    
    if top_quintile_rank == 1:
        print(f"  ✅ Strategy CONFIRMED: Most underperforming stocks had the best returns!")
    else:
        print(f"  ⚠️  Strategy MIXED: Most underperforming stocks ranked #{top_quintile_rank} out of 5")
    
    return performance_df

def create_comprehensive_visualizations(strategy_results, performance_df, output_dir):
    """
    Create comprehensive visualizations of the strategy results
    """
    print("\nCreating comprehensive visualizations...")
    
    # Figure 1: Performance comparison across quintiles
    fig, ((ax1, ax2), (ax3, ax4)) = plt.subplots(2, 2, figsize=(20, 12))
    
    # Total returns comparison
    quintile_names = [f"Q{i}" for i in range(1, 6)]
    total_returns = performance_df['total_return'].values
    colors = ['darkgreen', 'green', 'orange', 'red', 'darkred']
    
    bars1 = ax1.bar(quintile_names, total_returns, color=colors, alpha=0.8)
    ax1.set_ylabel('Total Return (%)')
    ax1.set_title('Total Returns by Quintile (Q1 = Most Underperforming)', fontweight='bold', fontsize=14)
    ax1.grid(True, alpha=0.3)
    ax1.axhline(y=0, color='black', linestyle='-', alpha=0.5)
    
    # Add value labels
    for bar, value in zip(bars1, total_returns):
        ax1.text(bar.get_x() + bar.get_width()/2., 
                bar.get_height() + (max(total_returns) * 0.02 if value >= 0 else max(total_returns) * -0.02),
                f'{value:.1f}%', ha='center', 
                va='bottom' if value >= 0 else 'top', fontweight='bold', fontsize=12)
    
    # Average period returns
    avg_returns = performance_df['avg_period_return'].values
    bars2 = ax2.bar(quintile_names, avg_returns, color=colors, alpha=0.8)
    ax2.set_ylabel('Average Period Return (%)')
    ax2.set_title('Average Period Returns by Quintile', fontweight='bold', fontsize=14)
    ax2.grid(True, alpha=0.3)
    ax2.axhline(y=0, color='black', linestyle='-', alpha=0.5)
    
    # Risk-return scatter
    ax3.scatter(performance_df['volatility'], performance_df['total_return'], 
               c=range(len(performance_df)), cmap='RdYlGn_r', s=200, alpha=0.8)
    
    for i, row in performance_df.iterrows():
        ax3.annotate(f"Q{row['quintile_rank']}", 
                    (row['volatility'], row['total_return']),
                    xytext=(5, 5), textcoords='offset points', 
                    fontweight='bold', fontsize=12)
    
    ax3.set_xlabel('Volatility (%)')
    ax3.set_ylabel('Total Return (%)')
    ax3.set_title('Risk vs Return by Quintile', fontweight='bold', fontsize=14)
    ax3.grid(True, alpha=0.3)
    
    # Win rates
    win_rates = performance_df['win_rate'].values
    bars4 = ax4.bar(quintile_names, win_rates, color=colors, alpha=0.8)
    ax4.set_ylabel('Win Rate (%)')
    ax4.set_title('Win Rate by Quintile', fontweight='bold', fontsize=14)
    ax4.grid(True, alpha=0.3)
    ax4.axhline(y=50, color='black', linestyle='--', alpha=0.5, label='50% baseline')
    ax4.legend()
    
    # Add value labels
    for bar, value in zip(bars4, win_rates):
        ax4.text(bar.get_x() + bar.get_width()/2., bar.get_height() + 1,
                f'{value:.1f}%', ha='center', va='bottom', fontweight='bold', fontsize=10)
    
    plt.tight_layout()
    plt.savefig(f"{output_dir}/quintile_performance_comparison.png", dpi=300, bbox_inches='tight')
    plt.close()
    
    # Figure 2: Cumulative performance over time
    fig, ax = plt.subplots(1, 1, figsize=(16, 8))
    
    for quintile_name in ['quintile_1', 'quintile_2', 'quintile_3', 'quintile_4', 'quintile_5']:
        if quintile_name in strategy_results:
            periods = strategy_results[quintile_name]['periods']
            
            cumulative_values = [1.0]
            dates = [periods[0]['start_date']] if periods else []
            
            for period in periods:
                period_return = period['return_pct'] / 100
                cumulative_values.append(cumulative_values[-1] * (1 + period_return))
                dates.append(period['end_date'])
            
            if len(dates) > 1:
                quintile_label = f"Q{quintile_name.split('_')[1]}"
                ax.plot(dates, cumulative_values, linewidth=3, label=quintile_label, alpha=0.8)
    
    ax.set_ylabel('Cumulative Value (Starting at 1.0)', fontsize=12)
    ax.set_xlabel('Date', fontsize=12)
    ax.set_title('Cumulative Performance Over Time by Quintile', fontweight='bold', fontsize=14)
    ax.legend(fontsize=12)
    ax.grid(True, alpha=0.3)
    ax.axhline(y=1, color='black', linestyle='-', alpha=0.5)
    
    plt.tight_layout()
    plt.savefig(f"{output_dir}/cumulative_performance_over_time.png", dpi=300, bbox_inches='tight')
    plt.close()
    
    # Figure 3: Period-by-period returns heatmap
    period_returns_data = []
    
    max_periods = max(len(strategy_results[q]['periods']) for q in strategy_results if strategy_results[q]['periods'])
    
    for quintile_name in ['quintile_1', 'quintile_2', 'quintile_3', 'quintile_4', 'quintile_5']:
        if quintile_name in strategy_results:
            periods = strategy_results[quintile_name]['periods']
            quintile_num = int(quintile_name.split('_')[1])
            
            period_returns = [p['return_pct'] for p in periods]
            # Pad with NaN if fewer periods
            while len(period_returns) < max_periods:
                period_returns.append(np.nan)
            
            period_returns_data.append(period_returns)
    
    if period_returns_data:
        fig, ax = plt.subplots(1, 1, figsize=(16, 8))
        
        period_returns_array = np.array(period_returns_data)
        
        im = ax.imshow(period_returns_array, cmap='RdYlGn', aspect='auto', vmin=-50, vmax=50)
        
        # Set labels
        ax.set_yticks(range(5))
        ax.set_yticklabels([f'Q{i+1}' for i in range(5)])
        ax.set_xticks(range(max_periods))
        ax.set_xticklabels([f'P{i+1}' for i in range(max_periods)])
        ax.set_ylabel('Quintile')
        ax.set_xlabel('Period')
        ax.set_title('Period Returns Heatmap by Quintile (%)', fontweight='bold', fontsize=14)
        
        # Add colorbar
        cbar = plt.colorbar(im, ax=ax)
        cbar.set_label('Return (%)', fontsize=12)
        
        # Add text annotations
        for i in range(5):
            for j in range(max_periods):
                if not np.isnan(period_returns_array[i, j]):
                    text = f'{period_returns_array[i, j]:.1f}%'
                    ax.text(j, i, text, ha='center', va='center', fontweight='bold', fontsize=8)
        
        plt.tight_layout()
        plt.savefig(f"{output_dir}/period_returns_heatmap.png", dpi=300, bbox_inches='tight')
        plt.close()
    
    print("  ✅ All visualizations created")

def save_detailed_results(strategy_results, all_period_results, performance_df, output_dir):
    """
    Save detailed results for further analysis
    """
    print("\nSaving detailed results...")
    
    # Save performance summary
    performance_df.to_csv(f"{output_dir}/quintile_performance_summary.csv", index=False)
    
    # Save detailed period results
    detailed_periods = []
    
    for period_result in all_period_results:
        base_info = {
            'period': period_result['period'],
            'start_date': period_result['start_date'],
            'end_date': period_result['end_date'],
            'available_assets': period_result['available_assets']
        }
        
        for quintile in ['quintile_1', 'quintile_2', 'quintile_3', 'quintile_4', 'quintile_5']:
            if quintile in period_result:
                quintile_result = period_result[quintile].copy()
                quintile_result.update(base_info)
                quintile_result['quintile'] = quintile
                detailed_periods.append(quintile_result)
    
    detailed_periods_df = pd.DataFrame(detailed_periods)
    detailed_periods_df.to_csv(f"{output_dir}/detailed_period_results.csv", index=False)
    
    # Save individual stock transactions
    all_transactions = []
    
    for quintile_name, quintile_data in strategy_results.items():
        for period in quintile_data['periods']:
            for individual_return in period['individual_returns']:
                transaction = individual_return.copy()
                transaction.update({
                    'quintile': quintile_name,
                    'period': period['period'],
                    'start_date': period['start_date'],
                    'end_date': period['end_date']
                })
                all_transactions.append(transaction)
    
    if all_transactions:
        transactions_df = pd.DataFrame(all_transactions)
        transactions_df.to_csv(f"{output_dir}/individual_stock_transactions.csv", index=False)
        print(f"  ✅ Saved {len(all_transactions)} individual stock transactions")
    
    # Save portfolio compositions for each period
    portfolio_compositions = []
    
    for quintile_name, quintile_data in strategy_results.items():
        for period in quintile_data['periods']:
            for stock in period['portfolio_details']:
                composition_record = stock.copy()
                composition_record.update({
                    'quintile': quintile_name,
                    'period': period['period'],
                    'start_date': period['start_date'],
                    'end_date': period['end_date']
                })
                portfolio_compositions.append(composition_record)
    
    if portfolio_compositions:
        compositions_df = pd.DataFrame(portfolio_compositions)
        compositions_df.to_csv(f"{output_dir}/portfolio_compositions.csv", index=False)
        print(f"  ✅ Saved portfolio compositions for all periods")
    
    # Create summary statistics by quintile
    quintile_stats = []
    
    for quintile_name, quintile_data in strategy_results.items():
        if quintile_data['periods']:
            all_individual_returns = []
            for period in quintile_data['periods']:
                all_individual_returns.extend([r['return_pct'] for r in period['individual_returns']])
            
            if all_individual_returns:
                quintile_stats.append({
                    'quintile': quintile_name,
                    'total_stocks_traded': len(all_individual_returns),
                    'avg_stock_return': np.mean(all_individual_returns),
                    'median_stock_return': np.median(all_individual_returns),
                    'std_stock_return': np.std(all_individual_returns),
                    'best_stock_return': np.max(all_individual_returns),
                    'worst_stock_return': np.min(all_individual_returns),
                    'positive_stock_rate': np.mean([r > 0 for r in all_individual_returns]) * 100,
                    'num_periods': len(quintile_data['periods'])
                })
    
    if quintile_stats:
        quintile_stats_df = pd.DataFrame(quintile_stats)
        quintile_stats_df.to_csv(f"{output_dir}/quintile_stock_level_statistics.csv", index=False)
        print(f"  ✅ Saved quintile-level stock statistics")
    
    print(f"  ✅ All detailed results saved to {output_dir}")

def main():
    print("REBALANCING STRATEGY ANALYSIS")
    print("=" * 70)
    print("Strategy: Select top 5 most underperforming stocks, hold 1 year, rebalance")
    print("Analysis: Compare quintiles 1-5 (most to least underperforming)")
    print("Output: Comprehensive analysis with detailed transaction data")
    
    # Setup directories
    v2_dir = "/Users/tim/IWLS-OPTIONS/IWLS_ANALYSIS_V2"
    
    if not os.path.exists(v2_dir):
        print("❌ IWLS_ANALYSIS_V2 directory not found. Run the IWLS V2 analysis first.")
        return
    
    # Create output directory for rebalancing strategy
    output_dir = os.path.join(v2_dir, "REBALANCING_STRATEGY_ANALYSIS")
    os.makedirs(output_dir, exist_ok=True)
    
    print(f"\nBase directory: {v2_dir}")
    print(f"Output directory: {output_dir}")
    
    # Load all IWLS data
    all_data = load_all_iwls_data(v2_dir)
    
    if len(all_data) < 25:
        print(f"❌ Insufficient assets loaded ({len(all_data)}). Need at least 25 for quintile analysis.")
        return
    
    # Determine date range for strategy
    # Find common date range across all assets
    min_dates = []
    max_dates = []
    
    for asset_name, df in all_data.items():
        min_dates.append(df['date'].min())
        max_dates.append(df['date'].max())
    
    strategy_start = max(min_dates)  # Latest start date to ensure all assets have data
    strategy_end = min(max_dates)    # Earliest end date to ensure all assets have data
    
    # Ensure we have at least 3 years of data for meaningful analysis
    min_duration = timedelta(days=365 * 3)
    if strategy_end - strategy_start < min_duration:
        print(f"❌ Insufficient date range. Available: {strategy_start} to {strategy_end}")
        print(f"Need at least {min_duration.days} days, have {(strategy_end - strategy_start).days} days")
        return
    
    # Use a reasonable subset for testing (e.g., 5 years)
    strategy_duration = min(timedelta(days=365 * 7), strategy_end - strategy_start)
    strategy_end = strategy_start + strategy_duration
    
    print(f"\nStrategy date range:")
    print(f"  Start: {strategy_start.strftime('%Y-%m-%d')}")
    print(f"  End: {strategy_end.strftime('%Y-%m-%d')}")
    print(f"  Duration: {(strategy_end - strategy_start).days} days ({(strategy_end - strategy_start).days/365.25:.1f} years)")
    
    # Run the rebalancing strategy
    strategy_results, all_period_results = run_rebalancing_strategy(
        all_data, strategy_start, strategy_end, rebalance_frequency_days=365
    )
    
    # Check if we got meaningful results
    total_periods = len(all_period_results)
    if total_periods == 0:
        print("❌ No periods processed. Check date ranges and data availability.")
        return
    
    print(f"\n✅ Strategy executed successfully:")
    print(f"   • Total periods: {total_periods}")
    print(f"   • Assets per quintile: 5")
    print(f"   • Rebalancing frequency: 1 year")
    
    # Analyze performance
    performance_df = analyze_strategy_performance(strategy_results, all_period_results)
    
    # Create visualizations
    create_comprehensive_visualizations(strategy_results, performance_df, output_dir)
    
    # Save detailed results
    save_detailed_results(strategy_results, all_period_results, performance_df, output_dir)
    
    # Additional detailed analysis
    print("\n" + "="*80)
    print("DETAILED ANALYSIS BY QUINTILE")
    print("="*80)
    
    for quintile_name in ['quintile_1', 'quintile_2', 'quintile_3', 'quintile_4', 'quintile_5']:
        if quintile_name in strategy_results and strategy_results[quintile_name]['periods']:
            quintile_data = strategy_results[quintile_name]
            quintile_num = int(quintile_name.split('_')[1])
            
            print(f"\nQUINTILE {quintile_num} DETAILED ANALYSIS:")
            print("-" * 50)
            
            periods = quintile_data['periods']
            all_stock_returns = []
            
            for period in periods:
                all_stock_returns.extend([r['return_pct'] for r in period['individual_returns']])
            
            if all_stock_returns:
                print(f"  Total stocks traded: {len(all_stock_returns)}")
                print(f"  Average stock return: {np.mean(all_stock_returns):.2f}%")
                print(f"  Median stock return: {np.median(all_stock_returns):.2f}%")
                print(f"  Best stock return: {np.max(all_stock_returns):.2f}%")
                print(f"  Worst stock return: {np.min(all_stock_returns):.2f}%")
                print(f"  Positive return rate: {np.mean([r > 0 for r in all_stock_returns])*100:.1f}%")
                print(f"  Volatility: {np.std(all_stock_returns):.2f}%")
                
                # Show some example trades
                print(f"\n  Sample trades from recent periods:")
                recent_period = periods[-1] if periods else None
                if recent_period and recent_period['individual_returns']:
                    for i, trade in enumerate(recent_period['individual_returns'][:3]):
                        print(f"    {trade['asset']}: {trade['return_pct']:.2f}% "
                              f"(entry dev: {trade['entry_deviation']:.1f}%)")
    
    # Final summary and recommendations
    print("\n" + "="*80)
    print("STRATEGY EFFECTIVENESS SUMMARY")
    print("="*80)
    
    if len(performance_df) >= 5:
        # Sort by total return to see ranking
        ranked_quintiles = performance_df.sort_values('total_return', ascending=False)
        
        print(f"\nRankings by Total Return:")
        for i, (_, row) in enumerate(ranked_quintiles.iterrows()):
            quintile_num = row['quintile_rank']
            print(f"  #{i+1}: Quintile {quintile_num} ({row['total_return']:.2f}% total return)")
        
        # Check strategy effectiveness
        top_quintile_rank = ranked_quintiles[ranked_quintiles['quintile_rank'] == 1].index[0] + 1
        
        print(f"\n🎯 STRATEGY ASSESSMENT:")
        if top_quintile_rank == 1:
            print("  ✅ HIGHLY EFFECTIVE: Most underperforming stocks delivered best returns!")
            effectiveness = "CONFIRMED"
        elif top_quintile_rank <= 2:
            print("  ✅ EFFECTIVE: Most underperforming stocks ranked in top 2!")
            effectiveness = "STRONG"
        elif top_quintile_rank <= 3:
            print("  ⚠️  MODERATE: Most underperforming stocks showed middle performance")
            effectiveness = "MODERATE"
        else:
            print("  ❌ POOR: Most underperforming stocks underperformed other quintiles")
            effectiveness = "POOR"
        
        # Risk-adjusted analysis
        top_quintile_data = performance_df[performance_df['quintile_rank'] == 1].iloc[0]
        print(f"\n📊 TOP QUINTILE (MOST UNDERPERFORMING) METRICS:")
        print(f"  • Total Return: {top_quintile_data['total_return']:.2f}%")
        print(f"  • Average Period Return: {top_quintile_data['avg_period_return']:.2f}%")
        print(f"  • Volatility: {top_quintile_data['volatility']:.2f}%")
        print(f"  • Sharpe Ratio: {top_quintile_data['sharpe_ratio']:.2f}")
        print(f"  • Win Rate: {top_quintile_data['win_rate']:.1f}%")
        print(f"  • Max Drawdown: {top_quintile_data['max_drawdown']:.2f}%")
        
        # Compare to other quintiles
        avg_other_return = performance_df[performance_df['quintile_rank'] != 1]['total_return'].mean()
        outperformance = top_quintile_data['total_return'] - avg_other_return
        
        print(f"\n📈 RELATIVE PERFORMANCE:")
        print(f"  • Top quintile return: {top_quintile_data['total_return']:.2f}%")
        print(f"  • Average other quintiles: {avg_other_return:.2f}%")
        print(f"  • Outperformance: {outperformance:+.2f}%")
        
        if outperformance > 10:
            print("  🚀 SIGNIFICANT outperformance!")
        elif outperformance > 0:
            print("  📈 Positive outperformance")
        else:
            print("  📉 Underperformance vs other quintiles")
    
    print(f"\n" + "="*70)
    print("ANALYSIS COMPLETE")
    print("="*70)
    print(f"Results saved to: {output_dir}")
    print("\nFiles created:")
    print("  📄 quintile_performance_summary.csv (main results)")
    print("  📄 detailed_period_results.csv (period-by-period analysis)")
    print("  📄 individual_stock_transactions.csv (every trade)")
    print("  📄 portfolio_compositions.csv (portfolio details each period)")
    print("  📄 quintile_stock_level_statistics.csv (quintile-level stats)")
    print("  📊 quintile_performance_comparison.png (main charts)")
    print("  📊 cumulative_performance_over_time.png (time series)")
    print("  📊 period_returns_heatmap.png (period performance grid)")
    
    print(f"\n🎯 Next steps:")
    print(f"   • Review quintile_performance_summary.csv for key metrics")
    print(f"   • Analyze individual_stock_transactions.csv for trade patterns")
    print(f"   • Use portfolio_compositions.csv to understand selection criteria")
    print(f"   • Strategy effectiveness: {effectiveness}")

if __name__ == "__main__":
    main()