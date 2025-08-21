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
                  not d.startswith('DYNAMIC') and not d.startswith('PORTFOLIO') and
                  not d.startswith('DAILY_TRACKING')]
    
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

def calculate_portfolio_value(portfolio_positions, all_data, current_date):
    """
    Calculate current total portfolio value
    """
    if not portfolio_positions:
        return 0.0, []
    
    position_values = []
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
                
                position_values.append({
                    'asset': asset_name,
                    'shares': shares,
                    'entry_price': entry_price,
                    'current_price': current_price,
                    'entry_value': entry_value,
                    'current_value': current_value,
                    'return_pct': return_pct,
                    'entry_date': position['entry_date'],
                    'entry_deviation': position['entry_deviation']
                })
                
                total_value += current_value
    
    return total_value, position_values

def get_benchmark_value(benchmark_data, target_date, initial_investment, strategy_start_date):
    """
    Calculate benchmark value at a specific date (buy and hold from strategy start)
    """
    if len(benchmark_data) == 0:
        return initial_investment
    
    # Find the benchmark price at strategy start date
    start_mask = (benchmark_data['date'] <= strategy_start_date)
    start_data = benchmark_data[start_mask]
    
    if len(start_data) == 0:
        return initial_investment
    
    start_price = start_data.iloc[-1]['price']
    
    # Find current benchmark price
    current_mask = (benchmark_data['date'] <= target_date)
    current_data = benchmark_data[current_mask]
    
    if len(current_data) == 0:
        return initial_investment
    
    current_price = current_data.iloc[-1]['price']
    
    # Calculate return from strategy start date
    return_mult = current_price / start_price
    return initial_investment * return_mult

def run_portfolio_level_strategy(all_data, benchmarks, start_date, end_date, 
                                initial_investment=100000, portfolio_profit_target=20, max_hold_days=365):
    """
    Run portfolio-level profit taking strategy
    """
    print(f"\nRunning PORTFOLIO-LEVEL 20% profit taking strategy")
    print(f"Strategy: Enter 5 stocks, monitor GROUP performance, exit ALL when portfolio hits 20%")
    print(f"Initial investment: ${initial_investment:,}")
    print(f"Portfolio profit target: {portfolio_profit_target}%")
    print(f"Portfolio size: 5 stocks")
    
    current_date = pd.to_datetime(start_date)
    end_date = pd.to_datetime(end_date)
    strategy_start_date = current_date
    
    # Strategy state
    cash_balance = initial_investment
    portfolio_positions = []  # Current 5-stock portfolio
    
    # Tracking data
    daily_tracking = []
    portfolio_cycles = []  # Complete portfolio cycles (entry to exit)
    
    cycle_id = 0
    portfolio_entry_date = None
    portfolio_entry_value = 0
    
    # Main daily loop
    while current_date <= end_date:
        
        # Calculate current portfolio value
        portfolio_value, position_details = calculate_portfolio_value(
            portfolio_positions, all_data, current_date
        )
        
        total_account_value = cash_balance + portfolio_value
        
        # Calculate benchmark values
        benchmark_values = {}
        for symbol, data in benchmarks.items():
            benchmark_values[symbol] = get_benchmark_value(data, current_date, initial_investment, strategy_start_date)
        
        # Calculate portfolio return if we have positions
        portfolio_return_pct = 0
        if portfolio_positions and portfolio_entry_value > 0:
            portfolio_return_pct = ((portfolio_value / portfolio_entry_value) - 1) * 100
        
        # Record daily data
        daily_record = {
            'date': current_date,
            'cash_balance': cash_balance,
            'portfolio_value': portfolio_value,
            'total_account_value': total_account_value,
            'portfolio_return_pct': portfolio_return_pct,
            'num_positions': len(portfolio_positions),
            'account_return_pct': ((total_account_value / initial_investment) - 1) * 100,
            'days_in_portfolio': (current_date - portfolio_entry_date).days if portfolio_entry_date else 0
        }
        
        # Add benchmark values
        for symbol, value in benchmark_values.items():
            daily_record[f'{symbol}_value'] = value
            daily_record[f'{symbol}_return_pct'] = ((value / initial_investment) - 1) * 100
        
        # Add individual position details
        for i, pos in enumerate(position_details):
            daily_record[f'position_{i+1}_asset'] = pos['asset']
            daily_record[f'position_{i+1}_return'] = pos['return_pct']
            daily_record[f'position_{i+1}_value'] = pos['current_value']
        
        daily_tracking.append(daily_record)
        
        # Check for portfolio exit conditions
        should_exit_portfolio = False
        exit_reason = ""
        
        if portfolio_positions:
            days_held = (current_date - portfolio_entry_date).days
            
            if portfolio_return_pct >= portfolio_profit_target:
                should_exit_portfolio = True
                exit_reason = f"portfolio_target_{portfolio_profit_target}%"
            elif days_held >= max_hold_days:
                should_exit_portfolio = True
                exit_reason = "max_hold_period"
        
        # Exit entire portfolio if conditions met
        if should_exit_portfolio:
            cycle_id += 1
            
            # Calculate final portfolio metrics
            portfolio_exit_value = portfolio_value
            days_held = (current_date - portfolio_entry_date).days
            
            # Record portfolio cycle
            cycle_record = {
                'cycle_id': cycle_id,
                'entry_date': portfolio_entry_date,
                'exit_date': current_date,
                'days_held': days_held,
                'entry_value': portfolio_entry_value,
                'exit_value': portfolio_exit_value,
                'portfolio_return_pct': portfolio_return_pct,
                'exit_reason': exit_reason,
                'num_stocks': len(portfolio_positions),
                'individual_positions': position_details.copy()
            }
            
            portfolio_cycles.append(cycle_record)
            
            # Add proceeds to cash
            cash_balance += portfolio_exit_value
            
            # Clear portfolio
            portfolio_positions = []
            portfolio_entry_date = None
            portfolio_entry_value = 0
            
            print(f"  {current_date.strftime('%Y-%m-%d')}: EXIT PORTFOLIO - {portfolio_return_pct:.2f}% after {days_held} days ({exit_reason})")
            print(f"    Portfolio value: ${portfolio_exit_value:,.0f} (from ${cycle_record['entry_value']:,.0f})")
        
        # Enter new portfolio if we don't have positions
        if not portfolio_positions:
            available_assets = get_available_assets_on_date(all_data, current_date)
            
            if len(available_assets) >= 5:
                top_5 = select_top_5_underperforming(available_assets)
                
                if len(top_5) == 5:
                    # Calculate equal investment per stock
                    available_cash = cash_balance * 0.98  # Keep 2% cash buffer
                    investment_per_stock = available_cash / 5
                    
                    if investment_per_stock > 1000:  # Minimum position size
                        total_invested = 0
                        
                        for _, stock in top_5.iterrows():
                            entry_price = stock['price']
                            shares = investment_per_stock / entry_price
                            actual_investment = shares * entry_price
                            
                            # Create position
                            position = {
                                'asset': stock['asset'],
                                'entry_date': current_date,
                                'entry_price': entry_price,
                                'shares': shares,
                                'entry_value': actual_investment,
                                'entry_deviation': stock['price_deviation']
                            }
                            
                            portfolio_positions.append(position)
                            total_invested += actual_investment
                        
                        # Update cash and portfolio tracking
                        cash_balance -= total_invested
                        portfolio_entry_date = current_date
                        portfolio_entry_value = total_invested
                        
                        avg_deviation = top_5['price_deviation'].mean()
                        min_deviation = top_5['price_deviation'].min()
                        
                        print(f"  {current_date.strftime('%Y-%m-%d')}: ENTER NEW PORTFOLIO")
                        print(f"    Stocks: {', '.join(top_5['asset'].tolist())}")
                        print(f"    Total invested: ${total_invested:,.0f}")
                        print(f"    Average deviation: {avg_deviation:.2f}% (min: {min_deviation:.2f}%)")
        
        # Move to next day
        current_date += timedelta(days=1)
    
    # Handle any remaining portfolio at end
    if portfolio_positions:
        portfolio_value, position_details = calculate_portfolio_value(
            portfolio_positions, all_data, end_date
        )
        
        if portfolio_entry_value > 0:
            final_return = ((portfolio_value / portfolio_entry_value) - 1) * 100
            days_held = (end_date - portfolio_entry_date).days
            
            cycle_id += 1
            cycle_record = {
                'cycle_id': cycle_id,
                'entry_date': portfolio_entry_date,
                'exit_date': end_date,
                'days_held': days_held,
                'entry_value': portfolio_entry_value,
                'exit_value': portfolio_value,
                'portfolio_return_pct': final_return,
                'exit_reason': 'strategy_end',
                'num_stocks': len(portfolio_positions),
                'individual_positions': position_details.copy()
            }
            
            portfolio_cycles.append(cycle_record)
            cash_balance += portfolio_value
    
    # Convert to DataFrames
    daily_tracking_df = pd.DataFrame(daily_tracking)
    portfolio_cycles_df = pd.DataFrame(portfolio_cycles)
    
    # Calculate final statistics
    final_value = daily_tracking_df.iloc[-1]['total_account_value']
    total_return = ((final_value / initial_investment) - 1) * 100
    
    print(f"\n✅ Portfolio-level strategy completed:")
    print(f"   Initial investment: ${initial_investment:,}")
    print(f"   Final value: ${final_value:,.0f}")
    print(f"   Total return: {total_return:.2f}%")
    print(f"   Total portfolio cycles: {len(portfolio_cycles_df)}")
    
    return daily_tracking_df, portfolio_cycles_df

def create_portfolio_level_visualizations(daily_tracking_df, portfolio_cycles_df, output_dir):
    """
    Create visualizations for portfolio-level strategy
    """
    print("\nCreating portfolio-level strategy visualizations...")
    
    # Figure 1: Main performance comparison
    fig, ((ax1, ax2), (ax3, ax4)) = plt.subplots(2, 2, figsize=(24, 16))
    
    # Plot 1: Portfolio value over time vs benchmarks
    ax1.plot(daily_tracking_df['date'], daily_tracking_df['total_account_value'], 
             linewidth=3, label='5-Stock Portfolio Strategy', color='red', alpha=0.9)
    
    # Add benchmarks if available
    benchmark_colors = {'SPY': 'blue', 'AAPL': 'green', 'AMZN': 'orange'}
    for symbol, color in benchmark_colors.items():
        col_name = f'{symbol}_value'
        if col_name in daily_tracking_df.columns:
            ax1.plot(daily_tracking_df['date'], daily_tracking_df[col_name], 
                    linewidth=2, label=symbol, color=color, alpha=0.7)
    
    ax1.set_ylabel('Portfolio Value ($)', fontsize=12)
    ax1.set_title('Portfolio Value Over Time: Portfolio-Level Strategy vs Benchmarks', 
                 fontweight='bold', fontsize=14)
    ax1.legend(fontsize=12)
    ax1.grid(True, alpha=0.3)
    ax1.yaxis.set_major_formatter(plt.FuncFormatter(lambda x, p: f'${x/1000:.0f}K'))
    
    # Plot 2: Returns percentage over time
    ax2.plot(daily_tracking_df['date'], daily_tracking_df['account_return_pct'], 
             linewidth=3, label='5-Stock Portfolio Strategy', color='red', alpha=0.9)
    
    for symbol, color in benchmark_colors.items():
        col_name = f'{symbol}_return_pct'
        if col_name in daily_tracking_df.columns:
            ax2.plot(daily_tracking_df['date'], daily_tracking_df[col_name], 
                    linewidth=2, label=symbol, color=color, alpha=0.7)
    
    ax2.set_ylabel('Return (%)', fontsize=12)
    ax2.set_title('Cumulative Returns: Portfolio-Level Strategy vs Benchmarks', 
                 fontweight='bold', fontsize=14)
    ax2.legend(fontsize=12)
    ax2.grid(True, alpha=0.3)
    ax2.axhline(y=0, color='black', linestyle='-', alpha=0.5)
    
    # Plot 3: Portfolio cycles performance
    if len(portfolio_cycles_df) > 0:
        colors = ['green' if r > 0 else 'red' for r in portfolio_cycles_df['portfolio_return_pct']]
        sizes = [abs(r) * 5 + 50 for r in portfolio_cycles_df['portfolio_return_pct']]
        
        ax3.scatter(portfolio_cycles_df['days_held'], portfolio_cycles_df['portfolio_return_pct'], 
                   c=colors, s=sizes, alpha=0.7)
        ax3.axhline(y=20, color='blue', linestyle='--', linewidth=2, label='20% Target')
        ax3.axhline(y=0, color='black', linestyle='-', alpha=0.5)
        ax3.set_xlabel('Days Held')
        ax3.set_ylabel('Portfolio Return (%)')
        ax3.set_title('Portfolio Cycle Performance: Return vs Hold Time', fontweight='bold', fontsize=14)
        ax3.legend()
        ax3.grid(True, alpha=0.3)
        
        # Plot 4: Portfolio returns distribution
        ax4.hist(portfolio_cycles_df['portfolio_return_pct'], bins=15, alpha=0.7, color='steelblue', edgecolor='black')
        ax4.axvline(portfolio_cycles_df['portfolio_return_pct'].mean(), color='red', linestyle='--', linewidth=2,
                   label=f'Mean: {portfolio_cycles_df["portfolio_return_pct"].mean():.1f}%')
        ax4.axvline(20, color='green', linestyle='--', linewidth=2, label='20% Target')
        ax4.set_xlabel('Portfolio Return (%)')
        ax4.set_ylabel('Frequency')
        ax4.set_title('Distribution of Portfolio Cycle Returns', fontweight='bold', fontsize=14)
        ax4.legend()
        ax4.grid(True, alpha=0.3)
    
    plt.tight_layout()
    plt.savefig(f"{output_dir}/portfolio_level_performance_analysis.png", dpi=300, bbox_inches='tight')
    plt.close()
    
    # Figure 2: Portfolio composition and timing
    fig, ((ax1, ax2), (ax3, ax4)) = plt.subplots(2, 2, figsize=(24, 16))
    
    # Plot 1: Cash vs invested over time
    ax1.fill_between(daily_tracking_df['date'], 0, daily_tracking_df['cash_balance'], 
                     alpha=0.6, label='Cash', color='lightblue')
    ax1.fill_between(daily_tracking_df['date'], daily_tracking_df['cash_balance'], 
                     daily_tracking_df['total_account_value'], 
                     alpha=0.6, label='Portfolio Positions', color='lightcoral')
    
    ax1.set_ylabel('Value ($)', fontsize=12)
    ax1.set_title('Portfolio Composition Over Time', fontweight='bold', fontsize=14)
    ax1.legend(fontsize=12)
    ax1.grid(True, alpha=0.3)
    ax1.yaxis.set_major_formatter(plt.FuncFormatter(lambda x, p: f'${x/1000:.0f}K'))
    
    # Plot 2: Number of positions over time
    ax2.plot(daily_tracking_df['date'], daily_tracking_df['num_positions'], 
             linewidth=2, color='purple', alpha=0.8)
    ax2.fill_between(daily_tracking_df['date'], 0, daily_tracking_df['num_positions'], 
                     alpha=0.3, color='purple')
    ax2.set_ylabel('Number of Stock Positions', fontsize=12)
    ax2.set_title('Portfolio Position Count Over Time', fontweight='bold', fontsize=14)
    ax2.grid(True, alpha=0.3)
    ax2.set_ylim(0, 6)
    
    # Plot 3: Days in portfolio over time
    ax3.plot(daily_tracking_df['date'], daily_tracking_df['days_in_portfolio'], 
             linewidth=2, color='orange', alpha=0.8)
    ax3.set_ylabel('Days in Current Portfolio', fontsize=12)
    ax3.set_title('Days Held in Current Portfolio Over Time', fontweight='bold', fontsize=14)
    ax3.grid(True, alpha=0.3)
    
    # Plot 4: Portfolio return tracking
    ax4.plot(daily_tracking_df['date'], daily_tracking_df['portfolio_return_pct'], 
             linewidth=2, color='darkgreen', alpha=0.8)
    ax4.axhline(y=20, color='red', linestyle='--', linewidth=2, label='20% Target')
    ax4.axhline(y=0, color='black', linestyle='-', alpha=0.5)
    ax4.set_ylabel('Current Portfolio Return (%)')
    ax4.set_title('Portfolio Return Progress Over Time', fontweight='bold', fontsize=14)
    ax4.legend()
    ax4.grid(True, alpha=0.3)
    
    plt.tight_layout()
    plt.savefig(f"{output_dir}/portfolio_composition_and_timing.png", dpi=300, bbox_inches='tight')
    plt.close()
    
    print("  ✅ All visualizations created")

def calculate_portfolio_performance_metrics(daily_tracking_df, portfolio_cycles_df, initial_investment):
    """
    Calculate performance metrics for portfolio-level strategy
    """
    print("\nCalculating portfolio-level performance metrics...")
    
    # Basic performance
    final_value = daily_tracking_df.iloc[-1]['total_account_value']
    total_return = ((final_value / initial_investment) - 1) * 100
    
    # Time metrics
    days_elapsed = (daily_tracking_df.iloc[-1]['date'] - daily_tracking_df.iloc[0]['date']).days
    years_elapsed = days_elapsed / 365.25
    annualized_return = ((final_value / initial_investment) ** (1/years_elapsed) - 1) * 100 if years_elapsed > 0 else 0
    
    # Volatility
    daily_tracking_df['daily_return'] = daily_tracking_df['account_return_pct'].pct_change() * 100
    daily_volatility = daily_tracking_df['daily_return'].std()
    annualized_volatility = daily_volatility * np.sqrt(252)
    
    # Risk metrics
    risk_free_rate = 2.0
    sharpe_ratio = (annualized_return - risk_free_rate) / annualized_volatility if annualized_volatility > 0 else 0
    
    # Drawdown
    running_max = daily_tracking_df['total_account_value'].expanding().max()
    drawdown = (daily_tracking_df['total_account_value'] - running_max) / running_max * 100
    max_drawdown = drawdown.min()
    
    # Portfolio cycle statistics
    if len(portfolio_cycles_df) > 0:
        avg_cycle_return = portfolio_cycles_df['portfolio_return_pct'].mean()
        median_cycle_return = portfolio_cycles_df['portfolio_return_pct'].median()
        cycle_win_rate = (portfolio_cycles_df['portfolio_return_pct'] > 0).mean() * 100
        target_hit_rate = (portfolio_cycles_df['exit_reason'] == 'portfolio_target_20%').mean() * 100
        avg_hold_time = portfolio_cycles_df['days_held'].mean()
        
        best_cycle = portfolio_cycles_df['portfolio_return_pct'].max()
        worst_cycle = portfolio_cycles_df['portfolio_return_pct'].min()
        
        profitable_cycles = portfolio_cycles_df[portfolio_cycles_df['portfolio_return_pct'] > 0]
        losing_cycles = portfolio_cycles_df[portfolio_cycles_df['portfolio_return_pct'] < 0]
        
        avg_winner = profitable_cycles['portfolio_return_pct'].mean() if len(profitable_cycles) > 0 else 0
        avg_loser = losing_cycles['portfolio_return_pct'].mean() if len(losing_cycles) > 0 else 0
        
        profit_factor = abs(avg_winner / avg_loser) if avg_loser < 0 else float('inf')
    else:
        avg_cycle_return = 0
        median_cycle_return = 0
        cycle_win_rate = 0
        target_hit_rate = 0
        avg_hold_time = 0
        best_cycle = 0
        worst_cycle = 0
        avg_winner = 0
        avg_loser = 0
        profit_factor = 0
    
    # Benchmark comparisons
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
        'portfolio_cycle_statistics': {
            'total_cycles': len(portfolio_cycles_df),
            'avg_cycle_return': avg_cycle_return,
            'median_cycle_return': median_cycle_return,
            'cycle_win_rate': cycle_win_rate,
            'target_hit_rate': target_hit_rate,
            'avg_hold_time': avg_hold_time,
            'best_cycle': best_cycle,
            'worst_cycle': worst_cycle,
            'avg_winner': avg_winner,
            'avg_loser': avg_loser,
            'profit_factor': profit_factor
        },
        'benchmark_performance': benchmark_performance
    }
    
    return performance_metrics

def save_portfolio_level_results(daily_tracking_df, portfolio_cycles_df, performance_metrics, output_dir):
    """
    Save all portfolio-level results
    """
    print("\nSaving portfolio-level results...")
    
    # Save daily tracking data
    daily_tracking_df.to_csv(f"{output_dir}/daily_portfolio_level_tracking.csv", index=False)
    print(f"  ✅ Daily tracking: {len(daily_tracking_df)} daily records")
    
    # Save portfolio cycles
    portfolio_cycles_df.to_csv(f"{output_dir}/portfolio_cycles_history.csv", index=False)
    print(f"  ✅ Portfolio cycles: {len(portfolio_cycles_df)} complete cycles")
    
    # Create detailed cycle breakdown
    if len(portfolio_cycles_df) > 0:
        cycle_details = []
        
        for _, cycle in portfolio_cycles_df.iterrows():
            cycle_base = {
                'cycle_id': cycle['cycle_id'],
                'entry_date': cycle['entry_date'],
                'exit_date': cycle['exit_date'],
                'days_held': cycle['days_held'],
                'portfolio_return_pct': cycle['portfolio_return_pct'],
                'exit_reason': cycle['exit_reason']
            }
            
            # Add individual stock details
            for position in cycle['individual_positions']:
                stock_detail = cycle_base.copy()
                stock_detail.update(position)
                cycle_details.append(stock_detail)
        
        cycle_details_df = pd.DataFrame(cycle_details)
        cycle_details_df.to_csv(f"{output_dir}/detailed_cycle_breakdown.csv", index=False)
        print(f"  ✅ Detailed cycle breakdown: {len(cycle_details_df)} stock-level records")
    
    # Save performance metrics summary
    performance_summary = []
    
    # Strategy performance
    strategy_perf = performance_metrics['strategy_performance']
    performance_summary.extend([
        {'metric_category': 'Strategy Performance', 'metric_name': 'Total Return', 'value': f"{strategy_perf['total_return']:.2f}%"},
        {'metric_category': 'Strategy Performance', 'metric_name': 'Annualized Return', 'value': f"{strategy_perf['annualized_return']:.2f}%"},
        {'metric_category': 'Strategy Performance', 'metric_name': 'Volatility', 'value': f"{strategy_perf['volatility']:.2f}%"},
        {'metric_category': 'Strategy Performance', 'metric_name': 'Sharpe Ratio', 'value': f"{strategy_perf['sharpe_ratio']:.3f}"},
        {'metric_category': 'Strategy Performance', 'metric_name': 'Max Drawdown', 'value': f"{strategy_perf['max_drawdown']:.2f}%"}
    ])
    
    # Portfolio cycle statistics
    cycle_stats = performance_metrics['portfolio_cycle_statistics']
    performance_summary.extend([
        {'metric_category': 'Portfolio Cycles', 'metric_name': 'Total Cycles', 'value': f"{cycle_stats['total_cycles']}"},
        {'metric_category': 'Portfolio Cycles', 'metric_name': 'Cycle Win Rate', 'value': f"{cycle_stats['cycle_win_rate']:.1f}%"},
        {'metric_category': 'Portfolio Cycles', 'metric_name': 'Target Hit Rate', 'value': f"{cycle_stats['target_hit_rate']:.1f}%"},
        {'metric_category': 'Portfolio Cycles', 'metric_name': 'Average Cycle Return', 'value': f"{cycle_stats['avg_cycle_return']:.2f}%"},
        {'metric_category': 'Portfolio Cycles', 'metric_name': 'Average Hold Time', 'value': f"{cycle_stats['avg_hold_time']:.1f} days"}
    ])
    
    # Benchmark comparisons
    for symbol, bench_perf in performance_metrics['benchmark_performance'].items():
        performance_summary.extend([
            {'metric_category': f'{symbol} Benchmark', 'metric_name': 'Total Return', 'value': f"{bench_perf['total_return']:.2f}%"},
            {'metric_category': f'{symbol} Benchmark', 'metric_name': 'Annualized Return', 'value': f"{bench_perf['annualized_return']:.2f}%"},
            {'metric_category': f'{symbol} Benchmark', 'metric_name': 'Sharpe Ratio', 'value': f"{bench_perf['sharpe_ratio']:.3f}"}
        ])
    
    performance_summary_df = pd.DataFrame(performance_summary)
    performance_summary_df.to_csv(f"{output_dir}/portfolio_level_performance_metrics.csv", index=False)
    print(f"  ✅ Performance metrics summary saved")
    
    # Save raw performance metrics as JSON
    import json
    with open(f"{output_dir}/portfolio_level_metrics_raw.json", 'w') as f:
        json.dump(performance_metrics, f, indent=2, default=str)

def print_portfolio_level_summary(performance_metrics, daily_tracking_df, portfolio_cycles_df):
    """
    Print comprehensive summary of portfolio-level strategy
    """
    print("\n" + "="*80)
    print("PORTFOLIO-LEVEL STRATEGY PERFORMANCE SUMMARY")
    print("="*80)
    
    strategy_perf = performance_metrics['strategy_performance']
    cycle_stats = performance_metrics['portfolio_cycle_statistics']
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
    
    print(f"\n🔄 PORTFOLIO CYCLE STATISTICS:")
    print(f"  Total Portfolio Cycles: {cycle_stats['total_cycles']}")
    print(f"  Cycle Win Rate: {cycle_stats['cycle_win_rate']:.1f}%")
    print(f"  Target Hit Rate (20%): {cycle_stats['target_hit_rate']:.1f}%")
    print(f"  Average Cycle Return: {cycle_stats['avg_cycle_return']:.2f}%")
    print(f"  Average Hold Time: {cycle_stats['avg_hold_time']:.1f} days")
    print(f"  Best Cycle: {cycle_stats['best_cycle']:.2f}%")
    print(f"  Worst Cycle: {cycle_stats['worst_cycle']:.2f}%")
    print(f"  Profit Factor: {cycle_stats['profit_factor']:.2f}")
    
    print(f"\n📈 BENCHMARK COMPARISON:")
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
        'Portfolio Strategy': strategy_total,
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
    
    if cycle_stats['target_hit_rate'] > 50:
        print(f"  ✅ High target hit rate ({cycle_stats['target_hit_rate']:.1f}%) - 20% portfolio target is achievable!")
    else:
        print(f"  ⚠️  Lower target hit rate ({cycle_stats['target_hit_rate']:.1f}%) - may need to adjust target or hold time")
    
    if cycle_stats['avg_hold_time'] < 150:
        print(f"  ⚡ Fast portfolio cycles ({cycle_stats['avg_hold_time']:.1f} days avg) - efficient capital deployment")
    else:
        print(f"  🕐 Longer portfolio cycles ({cycle_stats['avg_hold_time']:.1f} days avg) - patient approach")
    
    if strategy_perf['sharpe_ratio'] > 1.0:
        print(f"  📈 Excellent risk-adjusted returns (Sharpe: {strategy_perf['sharpe_ratio']:.3f})")
    elif strategy_perf['sharpe_ratio'] > 0.5:
        print(f"  📊 Good risk-adjusted returns (Sharpe: {strategy_perf['sharpe_ratio']:.3f})")
    else:
        print(f"  ⚠️  Lower risk-adjusted returns (Sharpe: {strategy_perf['sharpe_ratio']:.3f})")
    
    if abs(strategy_perf['max_drawdown']) < 25:
        print(f"  🛡️  Manageable drawdown ({strategy_perf['max_drawdown']:.2f}%)")
    else:
        print(f"  ⚠️  Significant drawdown ({strategy_perf['max_drawdown']:.2f}%) - consider risk management")
    
    # Calculate efficiency metrics
    cycles_per_year = cycle_stats['total_cycles'] / strategy_perf['years_elapsed']
    return_per_cycle = strategy_total / cycle_stats['total_cycles'] if cycle_stats['total_cycles'] > 0 else 0
    
    print(f"\n⚡ EFFICIENCY METRICS:")
    print(f"  Portfolio cycles per year: {cycles_per_year:.1f}")
    print(f"  Return per cycle: {return_per_cycle:.2f}%")
    print(f"  Capital deployment efficiency: {365 / cycle_stats['avg_hold_time']:.1f}x per year")
    
    # Strategy validation
    print(f"\n🎯 STRATEGY VALIDATION:")
    if cycle_stats['target_hit_rate'] > 40 and strategy_total > spy_total:
        print(f"  ✅ STRATEGY VALIDATED: High hit rate + outperforms market!")
        print(f"  💡 Portfolio-level 20% profit taking appears effective")
    elif cycle_stats['target_hit_rate'] > 60:
        print(f"  ✅ TARGET EFFECTIVE: High hit rate suggests 20% is achievable")
        print(f"  📊 Consider market timing or stock selection improvements")
    else:
        print(f"  ⚠️  MIXED RESULTS: Strategy needs refinement")
        print(f"  💡 Consider adjusting target % or selection criteria")

def main():
    print("PORTFOLIO-LEVEL PROFIT TAKING STRATEGY")
    print("=" * 70)
    print("Strategy: Enter 5 most underperforming stocks, monitor GROUP performance")
    print("Exit: Close ALL positions when PORTFOLIO reaches 20% gain, then repeat")
    print("Analysis: Daily tracking with benchmark comparison")
    
    # Setup directories
    v2_dir = "/Users/tim/IWLS-OPTIONS/IWLS_ANALYSIS_V2"
    
    if not os.path.exists(v2_dir):
        print("❌ IWLS_ANALYSIS_V2 directory not found.")
        return
    
    # Create output directory
    output_dir = os.path.join(v2_dir, "PORTFOLIO_LEVEL_STRATEGY")
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
    print(f"  Portfolio size: 5 stocks (most underperforming)")
    print(f"  Portfolio profit target: 20%")
    print(f"  Maximum portfolio hold period: 365 days")
    print(f"  Daily tracking: Complete portfolio valuation")
    print(f"  Benchmarks: {list(benchmarks.keys())}")
    
    print(f"\n🎯 KEY DIFFERENCE FROM PREVIOUS SCRIPT:")
    print(f"  • Monitor PORTFOLIO performance (not individual stocks)")
    print(f"  • Exit ALL 5 stocks when GROUP reaches 20% gain")
    print(f"  • Complete rebalancing into new 5 most underperforming stocks")
    print(f"  • Fewer total trades, longer hold periods per cycle")
    
    # Run the portfolio-level strategy
    daily_tracking_df, portfolio_cycles_df = run_portfolio_level_strategy(
        all_data, benchmarks, strategy_start, strategy_end, 
        initial_investment=100000, portfolio_profit_target=20, max_hold_days=365
    )
    
    # Calculate performance metrics
    performance_metrics = calculate_portfolio_performance_metrics(daily_tracking_df, portfolio_cycles_df, 100000)
    
    # Create visualizations
    create_portfolio_level_visualizations(daily_tracking_df, portfolio_cycles_df, output_dir)
    
    # Save all results
    save_portfolio_level_results(daily_tracking_df, portfolio_cycles_df, performance_metrics, output_dir)
    
    # Print final summary
    print_portfolio_level_summary(performance_metrics, daily_tracking_df, portfolio_cycles_df)
    
    print(f"\n" + "="*70)
    print("PORTFOLIO-LEVEL STRATEGY ANALYSIS COMPLETE")
    print("="*70)
    print(f"Results saved to: {output_dir}")
    print("\nFiles created:")
    print("  📄 daily_portfolio_level_tracking.csv (daily portfolio data)")
    print("  📄 portfolio_cycles_history.csv (complete portfolio cycles)")
    print("  📄 detailed_cycle_breakdown.csv (stock-level cycle details)")
    print("  📄 portfolio_level_performance_metrics.csv (all metrics)")
    print("  📄 portfolio_level_metrics_raw.json (raw metrics)")
    print("  📊 portfolio_level_performance_analysis.png (main charts)")
    print("  📊 portfolio_composition_and_timing.png (detailed analysis)")
    
    print(f"\n🎯 This CORRECTED analysis shows:")
    print(f"   • True portfolio-level profit taking (20% on combined holdings)")
    print(f"   • Complete rebalancing cycles (exit all, enter new 5)")
    print(f"   • Proper comparison to buy-and-hold benchmarks")
    print(f"   • Actual implementation of your intended strategy")
    print(f"   • Much fewer trades (portfolio cycles vs individual exits)")
    
    if len(performance_metrics['strategy_performance']) > 0:
        total_return = performance_metrics['strategy_performance']['total_return']
        cycles = performance_metrics['portfolio_cycle_statistics']['total_cycles']
        hit_rate = performance_metrics['portfolio_cycle_statistics']['target_hit_rate']
        
        print(f"\n🏆 PORTFOLIO STRATEGY RESULTS:")
        print(f"   Total Return: {total_return:.2f}%")
        print(f"   Portfolio Cycles: {cycles}")
        print(f"   Target Hit Rate: {hit_rate:.1f}%")
        
        if hit_rate > 50 and total_return > 20:
            print(f"   🚀 EXCELLENT: High hit rate + strong total returns!")
        elif hit_rate > 40:
            print(f"   ✅ GOOD: Portfolio-level targeting appears effective")
        else:
            print(f"   📊 MIXED: May need strategy refinement")

if __name__ == "__main__":
    main()