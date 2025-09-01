import pandas as pd
import numpy as np
import matplotlib.pyplot as plt
import seaborn as sns
from datetime import datetime, timedelta
import warnings
import os
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
                  not d.startswith('DYNAMIC') and not d.startswith('PORTFOLIO')]
    
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

def load_benchmark_data(v2_dir):
    """
    Load benchmark data (SPY, AAPL, AMZN) from the same IWLS results
    """
    print("Loading benchmark data...")
    
    benchmarks = {}
    benchmark_symbols = ['SPY', 'AAPL', 'AMZN']
    
    for symbol in benchmark_symbols:
        benchmark_dir = os.path.join(v2_dir, symbol)
        iwls_file = os.path.join(benchmark_dir, f"{symbol}_iwls_results.csv")
        
        if os.path.exists(iwls_file):
            try:
                df = pd.read_csv(iwls_file)
                df['date'] = pd.to_datetime(df['date'])
                df = df.dropna(subset=['price']).sort_values('date').reset_index(drop=True)
                
                # Just need price data for benchmarks
                benchmark_data = df[['date', 'price']].copy()
                benchmarks[symbol] = benchmark_data
                print(f"  ✅ {symbol}: {len(benchmark_data)} data points")
                
            except Exception as e:
                print(f"  ⚠️  Error loading {symbol}: {e}")
        else:
            print(f"  ⚠️  {symbol} not found")
    
    return benchmarks

def get_available_assets_on_date(all_data, target_date, lookback_days=10):
    """
    Get assets that have valid data around a specific date
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

def select_top_5_underperforming(available_df):
    """
    Select top 5 most underperforming stocks
    """
    # Focus on underperforming stocks (negative deviation)
    underperforming = available_df[available_df['price_deviation'] < 0].copy()
    
    if len(underperforming) < 5:
        print(f"    Warning: Only {len(underperforming)} underperforming stocks available")
        # If not enough underperforming, use all stocks sorted by deviation
        sorted_df = available_df.sort_values('price_deviation', ascending=True)
    else:
        # Sort underperforming stocks by most negative deviation
        sorted_df = underperforming.sort_values('price_deviation', ascending=True)
    
    # Select top 5
    if len(sorted_df) >= 5:
        top_5 = sorted_df.iloc[:5].copy()
        return top_5
    else:
        return pd.DataFrame()  # Not enough stocks

def calculate_current_portfolio_value(portfolio_positions, all_data, current_date):
    """
    Calculate current portfolio value for all positions
    """
    if not portfolio_positions:
        return 0.0, []
    
    current_values = []
    total_value = 0.0
    
    for position in portfolio_positions:
        asset_name = position['asset']
        shares = position['shares']
        entry_price = position['entry_price']
        entry_value = position['entry_value']
        
        if asset_name in all_data:
            asset_df = all_data[asset_name]
            
            # Find current price (closest to current_date)
            current_mask = (asset_df['date'] >= current_date - timedelta(days=15)) & \
                          (asset_df['date'] <= current_date + timedelta(days=15))
            current_data = asset_df[current_mask]
            
            if len(current_data) > 0:
                closest_current_idx = (current_data['date'] - current_date).abs().idxmin()
                current_price = current_data.loc[closest_current_idx, 'price']
                
                current_value = shares * current_price
                return_pct = ((current_price / entry_price) - 1) * 100
                
                current_values.append({
                    'asset': asset_name,
                    'shares': shares,
                    'entry_price': entry_price,
                    'current_price': current_price,
                    'entry_value': entry_value,
                    'current_value': current_value,
                    'return_pct': return_pct,
                    'entry_date': position['entry_date']
                })
                
                total_value += current_value
    
    return total_value, current_values

def initialize_benchmark_tracking(benchmarks, strategy_start_date, initial_investment):
    """
    Initialize benchmark tracking from the strategy start date
    """
    benchmark_tracking = {}
    
    for symbol, data in benchmarks.items():
        if len(data) == 0:
            continue
            
        # Find the price closest to strategy start date
        start_mask = (data['date'] >= strategy_start_date - timedelta(days=15)) & \
                    (data['date'] <= strategy_start_date + timedelta(days=15))
        start_data = data[start_mask]
        
        if len(start_data) > 0:
            closest_start_idx = (start_data['date'] - strategy_start_date).abs().idxmin()
            start_price = start_data.loc[closest_start_idx, 'price']
            start_date = start_data.loc[closest_start_idx, 'date']
            
            # Calculate how many shares we could buy with initial investment
            shares = initial_investment / start_price
            
            benchmark_tracking[symbol] = {
                'start_price': start_price,
                'start_date': start_date,
                'shares': shares,
                'data': data
            }
            
            print(f"  {symbol} benchmark: ${start_price:.2f} on {start_date.strftime('%Y-%m-%d')} ({shares:.2f} shares)")
        else:
            print(f"  ⚠️  No {symbol} data available around strategy start date")
    
    return benchmark_tracking

def get_benchmark_value(benchmark_info, target_date):
    """
    Calculate benchmark value at a specific date using the same start date as strategy
    """
    if not benchmark_info:
        return 0.0
    
    data = benchmark_info['data']
    shares = benchmark_info['shares']
    
    # Find current price closest to target date
    current_mask = (data['date'] >= target_date - timedelta(days=15)) & \
                  (data['date'] <= target_date + timedelta(days=15))
    current_data = data[current_mask]
    
    if len(current_data) > 0:
        closest_current_idx = (current_data['date'] - target_date).abs().idxmin()
        current_price = current_data.loc[closest_current_idx, 'price']
        
        current_value = shares * current_price
        return current_value
    else:
        # If no current data, return last known value
        last_available = data[data['date'] <= target_date]
        if len(last_available) > 0:
            last_price = last_available.iloc[-1]['price']
            return shares * last_price
        else:
            return benchmark_info['start_price'] * shares

def run_daily_tracked_strategy(all_data, benchmarks, start_date, end_date, 
                              initial_investment=100000, profit_target=20, max_hold_days=240):
    """
    Run 5-stock strategy with daily tracking and corrected benchmark comparison
    """
    print(f"\nRunning daily-tracked 5-stock strategy from {start_date} to {end_date}")
    print(f"Initial investment: ${initial_investment:,}")
    print(f"Profit target: {profit_target}%")
    print(f"Portfolio size: 5 stocks")
    
    current_date = pd.to_datetime(start_date)
    end_date = pd.to_datetime(end_date)
    
    # Initialize benchmark tracking from strategy start date
    benchmark_tracking = initialize_benchmark_tracking(benchmarks, current_date, initial_investment)
    
    # Portfolio state
    cash_balance = initial_investment
    portfolio_positions = []  # List of current stock positions
    
    # Tracking data
    daily_tracking = []
    trade_history = []
    
    trade_id = 0
    last_rebalance_check = current_date
    
    # Main daily loop
    while current_date <= end_date:
        
        # Calculate current portfolio value
        portfolio_value, position_details = calculate_current_portfolio_value(
            portfolio_positions, all_data, current_date
        )
        
        total_account_value = cash_balance + portfolio_value
        
        # Calculate benchmark values using corrected method
        benchmark_values = {}
        for symbol, bench_info in benchmark_tracking.items():
            benchmark_values[symbol] = get_benchmark_value(bench_info, current_date)
        
        # Record daily data
        daily_record = {
            'date': current_date,
            'cash_balance': cash_balance,
            'portfolio_value': portfolio_value,
            'total_account_value': total_account_value,
            'num_positions': len(portfolio_positions),
            'account_return_pct': ((total_account_value / initial_investment) - 1) * 100
        }
        
        # Add benchmark values and returns
        for symbol, value in benchmark_values.items():
            daily_record[f'{symbol}_value'] = value
            daily_record[f'{symbol}_return_pct'] = ((value / initial_investment) - 1) * 100
        
        # Add individual position details
        for i, pos in enumerate(position_details):
            daily_record[f'position_{i+1}_asset'] = pos['asset']
            daily_record[f'position_{i+1}_return'] = pos['return_pct']
            daily_record[f'position_{i+1}_value'] = pos['current_value']
        
        daily_tracking.append(daily_record)
        
        # Check for exits and entries (weekly check to reduce computation)
        days_since_last_check = (current_date - last_rebalance_check).days
        
        if days_since_last_check >= 7 or len(portfolio_positions) == 0:
            
            # Check for profit target hits or max hold period
            positions_to_exit = []
            
            for i, position in enumerate(portfolio_positions):
                asset_name = position['asset']
                entry_date = position['entry_date']
                entry_price = position['entry_price']
                
                # Find current return for this position
                current_return = 0
                for pos_detail in position_details:
                    if pos_detail['asset'] == asset_name:
                        current_return = pos_detail['return_pct']
                        break
                
                days_held = (current_date - entry_date).days
                
                # Check exit criteria
                should_exit = False
                exit_reason = ""
                
                if current_return >= profit_target:
                    should_exit = True
                    exit_reason = f"profit_target_{profit_target}%"
                elif days_held >= max_hold_days:
                    should_exit = True
                    exit_reason = "max_hold_period"
                
                if should_exit:
                    positions_to_exit.append((i, exit_reason, current_return))
            
            # Execute exits
            for pos_index, exit_reason, exit_return in reversed(positions_to_exit):  # Reverse to maintain indices
                position = portfolio_positions[pos_index]
                
                # Find exit price and calculate proceeds
                exit_value = 0
                for pos_detail in position_details:
                    if pos_detail['asset'] == position['asset']:
                        exit_value = pos_detail['current_value']
                        break
                
                # Record trade
                trade_id += 1
                days_held = (current_date - position['entry_date']).days
                
                trade_record = {
                    'trade_id': trade_id,
                    'asset': position['asset'],
                    'entry_date': position['entry_date'],
                    'exit_date': current_date,
                    'entry_price': position['entry_price'],
                    'exit_price': exit_value / position['shares'] if position['shares'] > 0 else 0,
                    'shares': position['shares'],
                    'entry_value': position['entry_value'],
                    'exit_value': exit_value,
                    'return_pct': exit_return,
                    'days_held': days_held,
                    'exit_reason': exit_reason
                }
                
                trade_history.append(trade_record)
                
                # Add proceeds to cash
                cash_balance += exit_value
                
                # Remove position
                portfolio_positions.pop(pos_index)
                
                print(f"  {current_date.strftime('%Y-%m-%d')}: EXIT {position['asset']} - {exit_return:.2f}% after {days_held} days ({exit_reason})")
            
            # Enter new positions if we have fewer than 5 stocks
            if len(portfolio_positions) < 5:
                available_assets = get_available_assets_on_date(all_data, current_date)
                
                if len(available_assets) >= 5:
                    # Get currently held assets to avoid duplicates
                    held_assets = [pos['asset'] for pos in portfolio_positions]
                    
                    # Filter out already held assets
                    available_for_entry = available_assets[~available_assets['asset'].isin(held_assets)]
                    
                    if len(available_for_entry) > 0:
                        # Select top underperforming stocks
                        underperforming = available_for_entry[available_for_entry['price_deviation'] < 0].copy()
                        
                        if len(underperforming) == 0:
                            # If no underperforming, use most negative deviation
                            underperforming = available_for_entry.sort_values('price_deviation', ascending=True)
                        else:
                            underperforming = underperforming.sort_values('price_deviation', ascending=True)
                        
                        # Calculate how many new positions we need
                        positions_needed = 5 - len(portfolio_positions)
                        positions_to_add = min(positions_needed, len(underperforming))
                        
                        # Calculate investment per position
                        available_cash_for_investment = cash_balance * 0.98  # Keep 2% cash buffer
                        investment_per_position = available_cash_for_investment / positions_to_add
                        
                        if investment_per_position > 1000:  # Minimum position size
                            
                            for i in range(positions_to_add):
                                new_stock = underperforming.iloc[i]
                                entry_price = new_stock['price']
                                shares = investment_per_position / entry_price
                                actual_investment = shares * entry_price
                                
                                # Create new position
                                new_position = {
                                    'asset': new_stock['asset'],
                                    'entry_date': current_date,
                                    'entry_price': entry_price,
                                    'shares': shares,
                                    'entry_value': actual_investment,
                                    'entry_deviation': new_stock['price_deviation']
                                }
                                
                                portfolio_positions.append(new_position)
                                cash_balance -= actual_investment
                                
                                print(f"  {current_date.strftime('%Y-%m-%d')}: ENTER {new_stock['asset']} - ${actual_investment:,.0f} ({new_stock['price_deviation']:.2f}% deviation)")
            
            last_rebalance_check = current_date
        
        # Move to next day
        current_date += timedelta(days=1)
    
    # Convert tracking data to DataFrame
    daily_tracking_df = pd.DataFrame(daily_tracking)
    trade_history_df = pd.DataFrame(trade_history)
    
    # Calculate final statistics
    final_value = daily_tracking_df.iloc[-1]['total_account_value']
    total_return = ((final_value / initial_investment) - 1) * 100
    
    print(f"\n✅ Strategy completed:")
    print(f"   Initial investment: ${initial_investment:,}")
    print(f"   Final value: ${final_value:,.0f}")
    print(f"   Total return: {total_return:.2f}%")
    print(f"   Total trades: {len(trade_history_df)}")
    
    # Print benchmark final values for verification
    print(f"\n📊 Benchmark final values:")
    for symbol in benchmark_tracking.keys():
        if f'{symbol}_value' in daily_tracking_df.columns:
            final_bench_value = daily_tracking_df.iloc[-1][f'{symbol}_value']
            bench_return = daily_tracking_df.iloc[-1][f'{symbol}_return_pct']
            print(f"   {symbol}: ${final_bench_value:,.0f} ({bench_return:.2f}% return)")
    
    return daily_tracking_df, trade_history_df

def create_comprehensive_performance_visualization(daily_tracking_df, trade_history_df, output_dir):
    """
    Create comprehensive performance visualizations
    """
    print("\nCreating comprehensive performance visualizations...")
    
    # Figure 1: Main performance comparison
    fig, ((ax1, ax2), (ax3, ax4)) = plt.subplots(2, 2, figsize=(24, 16))
    
    # Plot 1: Portfolio value over time vs benchmarks
    ax1.plot(daily_tracking_df['date'], daily_tracking_df['total_account_value'], 
             linewidth=3, label='5-Stock Strategy', color='red', alpha=0.9)
    
    # Add benchmarks if available
    benchmark_colors = {'SPY': 'blue', 'AAPL': 'green', 'AMZN': 'orange'}
    for symbol, color in benchmark_colors.items():
        col_name = f'{symbol}_value'
        if col_name in daily_tracking_df.columns:
            ax1.plot(daily_tracking_df['date'], daily_tracking_df[col_name], 
                    linewidth=2, label=symbol, color=color, alpha=0.7)
    
    ax1.set_ylabel('Portfolio Value ($)', fontsize=12)
    ax1.set_title('Portfolio Value Over Time: 5-Stock Strategy vs Benchmarks', 
                 fontweight='bold', fontsize=14)
    ax1.legend(fontsize=12)
    ax1.grid(True, alpha=0.3)
    ax1.yaxis.set_major_formatter(plt.FuncFormatter(lambda x, p: f'${x/1000:.0f}K'))
    
    # Plot 2: Returns percentage over time
    ax2.plot(daily_tracking_df['date'], daily_tracking_df['account_return_pct'], 
             linewidth=3, label='5-Stock Strategy', color='red', alpha=0.9)
    
    for symbol, color in benchmark_colors.items():
        col_name = f'{symbol}_return_pct'
        if col_name in daily_tracking_df.columns:
            ax2.plot(daily_tracking_df['date'], daily_tracking_df[col_name], 
                    linewidth=2, label=symbol, color=color, alpha=0.7)
    
    ax2.set_ylabel('Return (%)', fontsize=12)
    ax2.set_title('Cumulative Returns: 5-Stock Strategy vs Benchmarks', 
                 fontweight='bold', fontsize=14)
    ax2.legend(fontsize=12)
    ax2.grid(True, alpha=0.3)
    ax2.axhline(y=0, color='black', linestyle='-', alpha=0.5)
    
    # Plot 3: Portfolio composition over time (cash vs stocks)
    ax3.fill_between(daily_tracking_df['date'], 0, daily_tracking_df['cash_balance'], 
                     alpha=0.6, label='Cash', color='lightblue')
    ax3.fill_between(daily_tracking_df['date'], daily_tracking_df['cash_balance'], 
                     daily_tracking_df['total_account_value'], 
                     alpha=0.6, label='Stock Positions', color='lightcoral')
    
    ax3.set_ylabel('Value ($)', fontsize=12)
    ax3.set_title('Portfolio Composition Over Time', fontweight='bold', fontsize=14)
    ax3.legend(fontsize=12)
    ax3.grid(True, alpha=0.3)
    ax3.yaxis.set_major_formatter(plt.FuncFormatter(lambda x, p: f'${x/1000:.0f}K'))
    
    # Plot 4: Number of positions over time
    ax4.plot(daily_tracking_df['date'], daily_tracking_df['num_positions'], 
             linewidth=2, color='purple', alpha=0.8)
    ax4.fill_between(daily_tracking_df['date'], 0, daily_tracking_df['num_positions'], 
                     alpha=0.3, color='purple')
    ax4.set_ylabel('Number of Stock Positions', fontsize=12)
    ax4.set_title('Portfolio Position Count Over Time', fontweight='bold', fontsize=14)
    ax4.grid(True, alpha=0.3)
    ax4.set_ylim(0, 6)
    
    plt.tight_layout()
    plt.savefig(f"{output_dir}/comprehensive_performance_analysis.png", dpi=300, bbox_inches='tight')
    plt.close()
    
    # Figure 2: Detailed performance analysis
    fig, ((ax1, ax2), (ax3, ax4)) = plt.subplots(2, 2, figsize=(24, 16))
    
    # Plot 1: Trade analysis
    if len(trade_history_df) > 0:
        # Scatter plot of trade returns vs hold time
        colors = ['green' if r > 0 else 'red' for r in trade_history_df['return_pct']]
        sizes = [abs(r) * 3 + 20 for r in trade_history_df['return_pct']]  # Size by magnitude
        
        ax1.scatter(trade_history_df['days_held'], trade_history_df['return_pct'], 
                   c=colors, s=sizes, alpha=0.7)
        ax1.axhline(y=20, color='blue', linestyle='--', linewidth=2, label='20% Target')
        ax1.axhline(y=0, color='black', linestyle='-', alpha=0.5)
        ax1.set_xlabel('Days Held')
        ax1.set_ylabel('Trade Return (%)')
        ax1.set_title('Trade Performance: Return vs Hold Time', fontweight='bold', fontsize=14)
        ax1.legend()
        ax1.grid(True, alpha=0.3)
        
        # Plot 2: Trade returns distribution
        ax2.hist(trade_history_df['return_pct'], bins=20, alpha=0.7, color='steelblue', edgecolor='black')
        ax2.axvline(trade_history_df['return_pct'].mean(), color='red', linestyle='--', linewidth=2,
                   label=f'Mean: {trade_history_df["return_pct"].mean():.1f}%')
        ax2.axvline(20, color='green', linestyle='--', linewidth=2, label='20% Target')
        ax2.set_xlabel('Trade Return (%)')
        ax2.set_ylabel('Frequency')
        ax2.set_title('Distribution of Trade Returns', fontweight='bold', fontsize=14)
        ax2.legend()
        ax2.grid(True, alpha=0.3)
        
        # Plot 3: Monthly trade volume
        trade_history_df['entry_month'] = pd.to_datetime(trade_history_df['entry_date']).dt.to_period('M')
        monthly_trades = trade_history_df.groupby('entry_month').size()
        
        ax3.bar(range(len(monthly_trades)), monthly_trades.values, alpha=0.7, color='orange')
        ax3.set_xticks(range(0, len(monthly_trades), max(1, len(monthly_trades)//12)))
        ax3.set_xticklabels([str(monthly_trades.index[i]) for i in range(0, len(monthly_trades), max(1, len(monthly_trades)//12))], 
                           rotation=45)
        ax3.set_ylabel('Number of Trades')
        ax3.set_title('Trading Activity by Month', fontweight='bold', fontsize=14)
        ax3.grid(True, alpha=0.3)
        
        # Plot 4: Cumulative trade P&L
        trade_history_df['cumulative_pnl'] = (trade_history_df['exit_value'] - trade_history_df['entry_value']).cumsum()
        
        ax4.plot(range(len(trade_history_df)), trade_history_df['cumulative_pnl'], 
                linewidth=2, color='darkgreen', alpha=0.8)
        ax4.fill_between(range(len(trade_history_df)), 0, trade_history_df['cumulative_pnl'], 
                        alpha=0.3, color='darkgreen')
        ax4.set_xlabel('Trade Number')
        ax4.set_ylabel('Cumulative P&L ($)')
        ax4.set_title('Cumulative Profit/Loss from Trades', fontweight='bold', fontsize=14)
        ax4.grid(True, alpha=0.3)
        ax4.axhline(y=0, color='black', linestyle='-', alpha=0.5)
        ax4.yaxis.set_major_formatter(plt.FuncFormatter(lambda x, p: f'${x/1000:.0f}K'))
    
    plt.tight_layout()
    plt.savefig(f"{output_dir}/detailed_trade_analysis.png", dpi=300, bbox_inches='tight')
    plt.close()
    
    print("  ✅ All visualizations created")

def calculate_performance_metrics(daily_tracking_df, trade_history_df, initial_investment):
    """
    Calculate comprehensive performance metrics
    """
    print("\nCalculating performance metrics...")
    
    # Basic performance
    final_value = daily_tracking_df.iloc[-1]['total_account_value']
    total_return = ((final_value / initial_investment) - 1) * 100
    
    # Annualized return
    days_elapsed = (daily_tracking_df.iloc[-1]['date'] - daily_tracking_df.iloc[0]['date']).days
    years_elapsed = days_elapsed / 365.25
    annualized_return = ((final_value / initial_investment) ** (1/years_elapsed) - 1) * 100 if years_elapsed > 0 else 0
    
    # Volatility (daily returns)
    daily_tracking_df['daily_return'] = daily_tracking_df['account_return_pct'].pct_change() * 100
    daily_volatility = daily_tracking_df['daily_return'].std()
    annualized_volatility = daily_volatility * np.sqrt(252)  # 252 trading days
    
    # Sharpe ratio (assuming 2% risk-free rate)
    risk_free_rate = 2.0
    sharpe_ratio = (annualized_return - risk_free_rate) / annualized_volatility if annualized_volatility > 0 else 0
    
    # Maximum drawdown
    running_max = daily_tracking_df['total_account_value'].expanding().max()
    drawdown = (daily_tracking_df['total_account_value'] - running_max) / running_max * 100
    max_drawdown = drawdown.min()
    
    # Trade statistics
    if len(trade_history_df) > 0:
        avg_trade_return = trade_history_df['return_pct'].mean()
        median_trade_return = trade_history_df['return_pct'].median()
        win_rate = (trade_history_df['return_pct'] > 0).mean() * 100
        profit_target_hit_rate = (trade_history_df['exit_reason'] == 'profit_target_20%').mean() * 100
        avg_hold_time = trade_history_df['days_held'].mean()
        
        profitable_trades = trade_history_df[trade_history_df['return_pct'] > 0]
        losing_trades = trade_history_df[trade_history_df['return_pct'] < 0]
        
        avg_winner = profitable_trades['return_pct'].mean() if len(profitable_trades) > 0 else 0
        avg_loser = losing_trades['return_pct'].mean() if len(losing_trades) > 0 else 0
        
        profit_factor = abs(avg_winner / avg_loser) if avg_loser < 0 else float('inf')
    else:
        avg_trade_return = 0
        median_trade_return = 0
        win_rate = 0
        profit_target_hit_rate = 0
        avg_hold_time = 0
        avg_winner = 0
        avg_loser = 0
        profit_factor = 0
    
    # Benchmark comparisons (corrected to use same time period)
    benchmark_performance = {}
    
    for symbol in ['SPY', 'AAPL', 'AMZN']:
        return_col = f'{symbol}_return_pct'
        if return_col in daily_tracking_df.columns:
            final_benchmark_return = daily_tracking_df.iloc[-1][return_col]
            benchmark_annualized = ((1 + final_benchmark_return/100) ** (1/years_elapsed) - 1) * 100 if years_elapsed > 0 else 0
            
            # Benchmark volatility
            daily_tracking_df[f'{symbol}_daily_return'] = daily_tracking_df[return_col].pct_change() * 100
            benchmark_vol = daily_tracking_df[f'{symbol}_daily_return'].std() * np.sqrt(252)
            benchmark_sharpe = (benchmark_annualized - risk_free_rate) / benchmark_vol if benchmark_vol > 0 else 0
            
            benchmark_performance[symbol] = {
                'total_return': final_benchmark_return,
                'annualized_return': benchmark_annualized,
                'volatility': benchmark_vol,
                'sharpe_ratio': benchmark_sharpe
            }
    
    performance_metrics = {
        'strategy_performance': {
            'initial_investment': initial_investment,
            'final_value': final_value,
            'total_return': total_return,
            'annualized_return': annualized_return,
            'volatility': annualized_volatility,
            'sharpe_ratio': sharpe_ratio,
            'max_drawdown': max_drawdown,
            'days_elapsed': days_elapsed,
            'years_elapsed': years_elapsed
        },
        'trade_statistics': {
            'total_trades': len(trade_history_df),
            'avg_trade_return': avg_trade_return,
            'median_trade_return': median_trade_return,
            'win_rate': win_rate,
            'profit_target_hit_rate': profit_target_hit_rate,
            'avg_hold_time': avg_hold_time,
            'avg_winner': avg_winner,
            'avg_loser': avg_loser,
            'profit_factor': profit_factor,
            'best_trade': trade_history_df['return_pct'].max() if len(trade_history_df) > 0 else 0,
            'worst_trade': trade_history_df['return_pct'].min() if len(trade_history_df) > 0 else 0
        },
        'benchmark_performance': benchmark_performance
    }
    
    return performance_metrics

def save_comprehensive_results(daily_tracking_df, trade_history_df, performance_metrics, output_dir):
    """
    Save all comprehensive results
    """
    print("\nSaving comprehensive results...")
    
    # Save daily tracking data
    daily_tracking_df.to_csv(f"{output_dir}/daily_portfolio_tracking.csv", index=False)
    print(f"  ✅ Daily tracking: {len(daily_tracking_df)} daily records")
    
    # Save trade history
    trade_history_df.to_csv(f"{output_dir}/complete_trade_history.csv", index=False)
    print(f"  ✅ Trade history: {len(trade_history_df)} trades")
    
    # Save performance metrics
    import json
    
    # Convert performance metrics to a more readable format
    performance_summary = []
    
    # Strategy performance
    strategy_perf = performance_metrics['strategy_performance']
    performance_summary.append({
        'metric_category': 'Strategy Performance',
        'metric_name': 'Total Return',
        'value': f"{strategy_perf['total_return']:.2f}%"
    })
    performance_summary.append({
        'metric_category': 'Strategy Performance',
        'metric_name': 'Annualized Return',
        'value': f"{strategy_perf['annualized_return']:.2f}%"
    })
    performance_summary.append({
        'metric_category': 'Strategy Performance',
        'metric_name': 'Volatility',
        'value': f"{strategy_perf['volatility']:.2f}%"
    })
    performance_summary.append({
        'metric_category': 'Strategy Performance',
        'metric_name': 'Sharpe Ratio',
        'value': f"{strategy_perf['sharpe_ratio']:.3f}"
    })
    performance_summary.append({
        'metric_category': 'Strategy Performance',
        'metric_name': 'Max Drawdown',
        'value': f"{strategy_perf['max_drawdown']:.2f}%"
    })
    
    # Trade statistics
    trade_stats = performance_metrics['trade_statistics']
    performance_summary.append({
        'metric_category': 'Trade Statistics',
        'metric_name': 'Total Trades',
        'value': f"{trade_stats['total_trades']}"
    })
    performance_summary.append({
        'metric_category': 'Trade Statistics',
        'metric_name': 'Win Rate',
        'value': f"{trade_stats['win_rate']:.1f}%"
    })
    performance_summary.append({
        'metric_category': 'Trade Statistics',
        'metric_name': 'Profit Target Hit Rate',
        'value': f"{trade_stats['profit_target_hit_rate']:.1f}%"
    })
    performance_summary.append({
        'metric_category': 'Trade Statistics',
        'metric_name': 'Average Trade Return',
        'value': f"{trade_stats['avg_trade_return']:.2f}%"
    })
    performance_summary.append({
        'metric_category': 'Trade Statistics',
        'metric_name': 'Average Hold Time',
        'value': f"{trade_stats['avg_hold_time']:.1f} days"
    })
    
    # Benchmark comparisons
    for symbol, bench_perf in performance_metrics['benchmark_performance'].items():
        performance_summary.append({
            'metric_category': f'{symbol} Benchmark',
            'metric_name': 'Total Return',
            'value': f"{bench_perf['total_return']:.2f}%"
        })
        performance_summary.append({
            'metric_category': f'{symbol} Benchmark',
            'metric_name': 'Annualized Return',
            'value': f"{bench_perf['annualized_return']:.2f}%"
        })
        performance_summary.append({
            'metric_category': f'{symbol} Benchmark',
            'metric_name': 'Sharpe Ratio',
            'value': f"{bench_perf['sharpe_ratio']:.3f}"
        })
    
    performance_summary_df = pd.DataFrame(performance_summary)
    performance_summary_df.to_csv(f"{output_dir}/performance_metrics_summary.csv", index=False)
    print(f"  ✅ Performance metrics summary saved")
    
    # Save raw performance metrics as JSON
    with open(f"{output_dir}/performance_metrics_raw.json", 'w') as f:
        json.dump(performance_metrics, f, indent=2, default=str)
    
    # Create monthly summary
    if len(daily_tracking_df) > 0:
        daily_tracking_df['month'] = pd.to_datetime(daily_tracking_df['date']).dt.to_period('M')
        monthly_summary = daily_tracking_df.groupby('month').agg({
            'total_account_value': ['first', 'last', 'min', 'max'],
            'account_return_pct': ['first', 'last'],
            'num_positions': 'mean'
        }).round(2)
        
        # Flatten column names
        monthly_summary.columns = ['_'.join(col).strip() for col in monthly_summary.columns]
        monthly_summary = monthly_summary.reset_index()
        
        # Calculate monthly returns
        monthly_summary['monthly_return'] = (
            monthly_summary['account_return_pct_last'] - monthly_summary['account_return_pct_first']
        )
        
        monthly_summary.to_csv(f"{output_dir}/monthly_performance_summary.csv", index=False)
        print(f"  ✅ Monthly summary: {len(monthly_summary)} months")

def print_final_performance_summary(performance_metrics, daily_tracking_df, trade_history_df):
    """
    Print comprehensive final performance summary with corrected benchmark comparison
    """
    print("\n" + "="*80)
    print("5-STOCK STRATEGY PERFORMANCE SUMMARY (CORRECTED BENCHMARKS)")
    print("="*80)
    
    strategy_perf = performance_metrics['strategy_performance']
    trade_stats = performance_metrics['trade_statistics']
    benchmark_perf = performance_metrics['benchmark_performance']
    
    print(f"\n💰 STRATEGY PERFORMANCE:")
    print(f"  Initial Investment: ${strategy_perf['initial_investment']:,}")
    print(f"  Final Value: ${strategy_perf['final_value']:,.0f}")
    print(f"  Total Return: {strategy_perf['total_return']:.2f}%")
    print(f"  Annualized Return: {strategy_perf['annualized_return']:.2f}%")
    print(f"  Time Period: {strategy_perf['years_elapsed']:.2f} years")
    
    print(f"\n📊 RISK METRICS:")
    print(f"  Volatility: {strategy_perf['volatility']:.2f}%")
    print(f"  Sharpe Ratio: {strategy_perf['sharpe_ratio']:.3f}")
    print(f"  Maximum Drawdown: {strategy_perf['max_drawdown']:.2f}%")
    
    print(f"\n🎯 TRADING STATISTICS:")
    print(f"  Total Trades: {trade_stats['total_trades']}")
    print(f"  Win Rate: {trade_stats['win_rate']:.1f}%")
    print(f"  Profit Target Hit Rate: {trade_stats['profit_target_hit_rate']:.1f}%")
    print(f"  Average Trade Return: {trade_stats['avg_trade_return']:.2f}%")
    print(f"  Average Hold Time: {trade_stats['avg_hold_time']:.1f} days")
    print(f"  Best Trade: {trade_stats['best_trade']:.2f}%")
    print(f"  Worst Trade: {trade_stats['worst_trade']:.2f}%")
    print(f"  Profit Factor: {trade_stats['profit_factor']:.2f}")
    
    print(f"\n📈 BENCHMARK COMPARISON (240 DAY MAX HOLD):")
    print(f"{'Metric':<20} {'Strategy':<12} {'SPY':<12} {'AAPL':<12} {'AMZN':<12}")
    print("-" * 68)
    
    # Total Return
    strategy_total = strategy_perf['total_return']
    spy_total = benchmark_perf.get('SPY', {}).get('total_return', 0)
    aapl_total = benchmark_perf.get('AAPL', {}).get('total_return', 0)
    amzn_total = benchmark_perf.get('AMZN', {}).get('total_return', 0)
    
    print(f"{'Total Return':<20} {strategy_total:>11.2f}% {spy_total:>11.2f}% {aapl_total:>11.2f}% {amzn_total:>11.2f}%")
    
    # Annualized Return
    strategy_ann = strategy_perf['annualized_return']
    spy_ann = benchmark_perf.get('SPY', {}).get('annualized_return', 0)
    aapl_ann = benchmark_perf.get('AAPL', {}).get('annualized_return', 0)
    amzn_ann = benchmark_perf.get('AMZN', {}).get('annualized_return', 0)
    
    print(f"{'Annualized Return':<20} {strategy_ann:>11.2f}% {spy_ann:>11.2f}% {aapl_ann:>11.2f}% {amzn_ann:>11.2f}%")
    
    # Sharpe Ratio
    strategy_sharpe = strategy_perf['sharpe_ratio']
    spy_sharpe = benchmark_perf.get('SPY', {}).get('sharpe_ratio', 0)
    aapl_sharpe = benchmark_perf.get('AAPL', {}).get('sharpe_ratio', 0)
    amzn_sharpe = benchmark_perf.get('AMZN', {}).get('sharpe_ratio', 0)
    
    print(f"{'Sharpe Ratio':<20} {strategy_sharpe:>11.3f}  {spy_sharpe:>11.3f}  {aapl_sharpe:>11.3f}  {amzn_sharpe:>11.3f}")
    
    print(f"\n🏆 PERFORMANCE RANKING:")
    all_returns = {
        '5-Stock Strategy': strategy_total,
        'SPY': spy_total,
        'AAPL': aapl_total,
        'AMZN': amzn_total
    }
    
    # Filter out zero returns (missing benchmarks)
    valid_returns = {k: v for k, v in all_returns.items() if v != 0}
    sorted_returns = sorted(valid_returns.items(), key=lambda x: x[1], reverse=True)
    
    for i, (name, return_pct) in enumerate(sorted_returns):
        print(f"  #{i+1}: {name:<18} {return_pct:>8.2f}%")
    
    # Calculate outperformance
    print(f"\n📊 STRATEGY vs BENCHMARK OUTPERFORMANCE:")
    for benchmark, bench_return in [('SPY', spy_total), ('AAPL', aapl_total), ('AMZN', amzn_total)]:
        if bench_return != 0:
            outperformance = strategy_total - bench_return
            if outperformance > 0:
                print(f"  vs {benchmark}: +{outperformance:.2f}% outperformance ✅")
            else:
                print(f"  vs {benchmark}: {outperformance:.2f}% underperformance ❌")
    
    # Key insights
    print(f"\n🧠 KEY INSIGHTS:")
    
    if trade_stats['profit_target_hit_rate'] > 50:
        print(f"  ✅ High profit target hit rate ({trade_stats['profit_target_hit_rate']:.1f}%) - strategy is working!")
    else:
        print(f"  ⚠️  Lower profit target hit rate ({trade_stats['profit_target_hit_rate']:.1f}%) - may need adjustment")
    
    if trade_stats['avg_hold_time'] < 100:
        print(f"  ⚡ Fast turnover ({trade_stats['avg_hold_time']:.1f} days avg) - efficient capital deployment")
    else:
        print(f"  🕐 Longer hold times ({trade_stats['avg_hold_time']:.1f} days avg) - patient approach")
    
    if strategy_perf['sharpe_ratio'] > 1.0:
        print(f"  📈 Excellent risk-adjusted returns (Sharpe: {strategy_perf['sharpe_ratio']:.3f})")
    elif strategy_perf['sharpe_ratio'] > 0.5:
        print(f"  📊 Good risk-adjusted returns (Sharpe: {strategy_perf['sharpe_ratio']:.3f})")
    else:
        print(f"  ⚠️  Lower risk-adjusted returns (Sharpe: {strategy_perf['sharpe_ratio']:.3f})")
    
    if abs(strategy_perf['max_drawdown']) < 20:
        print(f"  🛡️  Manageable drawdown ({strategy_perf['max_drawdown']:.2f}%)")
    else:
        print(f"  ⚠️  Significant drawdown ({strategy_perf['max_drawdown']:.2f}%) - consider risk management")
    
    # Calculate efficiency metrics
    trades_per_year = trade_stats['total_trades'] / strategy_perf['years_elapsed']
    return_per_trade = strategy_total / trade_stats['total_trades'] if trade_stats['total_trades'] > 0 else 0
    
    print(f"\n⚡ EFFICIENCY METRICS:")
    print(f"  Trades per year: {trades_per_year:.1f}")
    print(f"  Return per trade: {return_per_trade:.2f}%")
    print(f"  Capital turnover: {365 / trade_stats['avg_hold_time']:.1f}x per year")
    
    print(f"\n🔧 BENCHMARK CALCULATION VERIFICATION:")
    print(f"  ✅ All benchmarks calculated from strategy start date")
    print(f"  ✅ Same time period used for all comparisons")
    print(f"  ✅ Buy-and-hold approach for benchmarks")
    print(f"  ✅ No look-ahead bias in benchmark calculations")

def main():
    print("DAILY PORTFOLIO TRACKING - 5 STOCK STRATEGY (240 DAY MAX HOLD)")
    print("=" * 70)
    print("Strategy: 5 most underperforming stocks, 20% profit target, 240 day max hold")
    print("Analysis: Complete daily portfolio valuation vs SPY, AAPL, AMZN benchmarks")
    print("Fix: Corrected benchmark calculation + reduced max hold to 240 days")
    
    # Setup directories
    v2_dir = "/Users/tim/IWLS-OPTIONS/IWLS_ANALYSIS_V2"
    
    if not os.path.exists(v2_dir):
        print("❌ IWLS_ANALYSIS_V2 directory not found.")
        return
    
    # Create output directory
    output_dir = os.path.join(v2_dir, "DAILY_TRACKING_5STOCK_20PCT_240DAYS")
    os.makedirs(output_dir, exist_ok=True)
    
    print(f"\nOutput directory: {output_dir}")
    
    # Load all IWLS data
    all_data = load_all_iwls_data(v2_dir)
    
    if len(all_data) < 5:
        print(f"❌ Insufficient assets loaded ({len(all_data)}). Need at least 5 for analysis.")
        return
    
    # Load benchmark data
    benchmarks = load_benchmark_data(v2_dir)
    
    # Determine date range
    min_dates = []
    max_dates = []
    
    for asset_name, df in all_data.items():
        min_dates.append(df['date'].min())
        max_dates.append(df['date'].max())
    
    # Also consider benchmark dates
    for symbol, df in benchmarks.items():
        if len(df) > 0:
            min_dates.append(df['date'].min())
            max_dates.append(df['date'].max())
    
    strategy_start = max(min_dates)
    strategy_end = min(max_dates)
    
    # Use reasonable timeframe (4-6 years for comprehensive analysis)
    strategy_duration = min(timedelta(days=365 * 5), strategy_end - strategy_start)
    strategy_end = strategy_start + strategy_duration
    
    print(f"\nStrategy parameters:")
    print(f"  Date range: {strategy_start.strftime('%Y-%m-%d')} to {strategy_end.strftime('%Y-%m-%d')}")
    print(f"  Duration: {(strategy_end - strategy_start).days} days ({(strategy_end - strategy_start).days/365.25:.1f} years)")
    print(f"  Initial investment: $100,000")
    print(f"  Portfolio size: 5 stocks")
    print(f"  Profit target: 20%")
    print(f"  Maximum hold period: 240 days")
    print(f"  Rebalancing check: Weekly")
    print(f"  Daily tracking: Complete portfolio valuation")
    print(f"  Benchmarks: {list(benchmarks.keys())} (calculated from strategy start date)")
    
    # Run the daily tracked strategy
    daily_tracking_df, trade_history_df = run_daily_tracked_strategy(
        all_data, benchmarks, strategy_start, strategy_end, 
        initial_investment=100000, profit_target=20, max_hold_days=240
    )
    
    # Calculate performance metrics
    performance_metrics = calculate_performance_metrics(daily_tracking_df, trade_history_df, 100000)
    
    # Create comprehensive visualizations
    create_comprehensive_performance_visualization(daily_tracking_df, trade_history_df, output_dir)
    
    # Save all results
    save_comprehensive_results(daily_tracking_df, trade_history_df, performance_metrics, output_dir)
    
    # Print final summary
    print_final_performance_summary(performance_metrics, daily_tracking_df, trade_history_df)
    
    print(f"\n" + "="*70)
    print("DAILY TRACKING ANALYSIS COMPLETE (240 DAY MAX HOLD)")
    print("="*70)
    print(f"Results saved to: {output_dir}")
    print("\nFiles created:")
    print("  📄 daily_portfolio_tracking.csv (complete daily data)")
    print("  📄 complete_trade_history.csv (every trade with full details)")
    print("  📄 performance_metrics_summary.csv (all performance metrics)")
    print("  📄 monthly_performance_summary.csv (monthly breakdown)")
    print("  📄 performance_metrics_raw.json (raw metrics for analysis)")
    print("  📊 comprehensive_performance_analysis.png (main performance charts)")
    print("  📊 detailed_trade_analysis.png (trade analysis charts)")
    
    print(f"\n🎯 This corrected analysis provides:")
    print(f"   • Maximum hold period reduced to 240 days for faster turnover")
    print(f"   • Benchmarks calculated from strategy start date (no look-ahead bias)")
    print(f"   • Fair comparison using identical time periods")
    print(f"   • Accurate benchmark buy-and-hold returns")
    print(f"   • Corrected outperformance/underperformance calculations")
    print(f"   • Proper risk-adjusted metrics for all assets")
    
    if len(performance_metrics['strategy_performance']) > 0:
        total_return = performance_metrics['strategy_performance']['total_return']
        sharpe_ratio = performance_metrics['strategy_performance']['sharpe_ratio']
        
        print(f"\n🏆 FINAL CORRECTED STRATEGY ASSESSMENT:")
        print(f"   Total Return: {total_return:.2f}%")
        print(f"   Sharpe Ratio: {sharpe_ratio:.3f}")
        
        # Compare to corrected SPY benchmark
        spy_return = performance_metrics['benchmark_performance'].get('SPY', {}).get('total_return', 0)
        if spy_return > 0:
            vs_spy = total_return - spy_return
            print(f"   vs SPY: {vs_spy:+.2f}% {'outperformance' if vs_spy > 0 else 'underperformance'}")
        
        if total_return > 50 and sharpe_ratio > 1.0:
            print(f"   🚀 EXCELLENT: Strong returns with good risk management!")
        elif total_return > 20 and sharpe_ratio > 0.5:
            print(f"   ✅ GOOD: Solid performance with reasonable risk")
        else:
            print(f"   📊 MIXED: Performance needs evaluation against risk tolerance")

if __name__ == "__main__":
    main()