import pandas as pd
import numpy as np
import matplotlib.pyplot as plt
import matplotlib.dates as mdates
import os
import glob
from datetime import datetime, timedelta
from collections import defaultdict
import warnings
warnings.filterwarnings('ignore')

class TradingStrategy:
    def __init__(self, initial_capital=10000, min_hold_days=200, rebalance_threshold_percentile=80):
        self.initial_capital = initial_capital
        self.min_hold_days = min_hold_days
        self.rebalance_threshold_percentile = rebalance_threshold_percentile
        
        # Strategy state
        self.current_portfolio = None
        self.current_entry_date = None
        self.current_capital = initial_capital
        self.trade_history = []
        self.portfolio_history = []
        
        print(f"Initialized Trading Strategy:")
        print(f"  Initial capital: ${initial_capital:,}")
        print(f"  Minimum hold period: {min_hold_days} days")
        print(f"  Rebalance threshold: {rebalance_threshold_percentile}th percentile")

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
            df['absolute_deviation'] = df['price_deviation'].abs()
            all_results[asset_name] = df
            print(f"Loaded {asset_name}: {len(df)} data points")
        except Exception as e:
            print(f"Error loading {asset_name}: {str(e)}")
    
    return all_results

def calculate_daily_signals_and_threshold(all_results):
    """
    Calculate daily deviation signals and determine the threshold for highest quintile
    """
    print("\nCalculating daily signals and threshold...")
    
    # Get all unique dates
    all_dates = set()
    for df in all_results.values():
        all_dates.update(df['date'].tolist())
    
    all_dates = sorted(list(all_dates))
    
    daily_signals = []
    
    for current_date in all_dates:
        # Get absolute deviations for all assets on this date
        daily_deviations = {}
        
        for asset_name, df in all_results.items():
            asset_data = df[df['date'] == current_date]
            
            if len(asset_data) > 0:
                abs_deviation = asset_data['absolute_deviation'].iloc[0]
                raw_deviation = asset_data['price_deviation'].iloc[0]
                price = asset_data['price'].iloc[0]
                
                daily_deviations[asset_name] = {
                    'absolute_deviation': abs_deviation,
                    'raw_deviation': raw_deviation,
                    'price': price
                }
        
        # Skip days with insufficient data
        if len(daily_deviations) < 5:
            continue
        
        # Sort by absolute deviation (highest first)
        sorted_assets = sorted(daily_deviations.items(), 
                             key=lambda x: x[1]['absolute_deviation'], 
                             reverse=True)
        
        # Get top 5
        top5_assets = sorted_assets[:5]
        
        # Sum the absolute deviations of top 5
        sum_top5_deviations = sum(data['absolute_deviation'] for _, data in top5_assets)
        
        daily_signals.append({
            'date': current_date,
            'sum_top5_absolute_deviations': sum_top5_deviations,
            'top5_assets': [asset for asset, _ in top5_assets],
            'top5_deviations': {asset: data for asset, data in top5_assets}
        })
    
    # Calculate threshold for highest quintile
    deviations = [signal['sum_top5_absolute_deviations'] for signal in daily_signals]
    threshold = np.percentile(deviations, 80)  # 80th percentile = highest quintile
    
    print(f"Calculated threshold (80th percentile): {threshold:.2f}%")
    print(f"Total trading days: {len(daily_signals)}")
    
    return daily_signals, threshold

def get_asset_price_on_date(all_results, asset, date):
    """
    Get asset price on specific date
    """
    if asset not in all_results:
        return None
    
    df = all_results[asset]
    asset_data = df[df['date'] == date]
    
    if len(asset_data) == 0:
        return None
    
    return asset_data['price'].iloc[0]

def execute_trading_strategy(all_results, daily_signals, threshold, strategy):
    """
    Execute the rebalancing trading strategy with continuous portfolio valuation
    """
    print(f"\nExecuting trading strategy...")
    print(f"Strategy parameters:")
    print(f"  Initial capital: ${strategy.initial_capital:,}")
    print(f"  Minimum hold period: {strategy.min_hold_days} days")
    print(f"  Signal threshold: {threshold:.2f}%")
    
    # Get all unique dates and sort them
    all_dates = set()
    for df in all_results.values():
        all_dates.update(df['date'].tolist())
    all_dates = sorted(list(all_dates))
    
    # Filter signals that meet threshold
    valid_signals = [signal for signal in daily_signals 
                    if signal['sum_top5_absolute_deviations'] >= threshold]
    
    # Create a lookup for signals by date
    signal_lookup = {signal['date']: signal for signal in valid_signals}
    
    print(f"Found {len(valid_signals)} valid signals out of {len(daily_signals)} trading days")
    print(f"Processing {len(all_dates)} total trading days for continuous valuation...")
    
    processed_count = 0
    
    for current_date in all_dates:
        processed_count += 1
        
        # Progress update
        if processed_count % 500 == 0:
            print(f"  Processed {processed_count}/{len(all_dates)} dates...")
        
        # Check if this date has a valid signal
        if current_date in signal_lookup:
            signal = signal_lookup[current_date]
            signal_strength = signal['sum_top5_absolute_deviations']
            top5_assets = signal['top5_assets']
            
            # Check if we currently have a portfolio
            if strategy.current_portfolio is None:
                # First entry
                enter_new_position(strategy, all_results, current_date, top5_assets, signal_strength)
                
            else:
                # Check if we can rebalance (held for more than min_hold_days)
                days_held = (current_date - strategy.current_entry_date).days
                
                if days_held >= strategy.min_hold_days:
                    # Exit current position and enter new one
                    exit_current_position(strategy, all_results, current_date)
                    enter_new_position(strategy, all_results, current_date, top5_assets, signal_strength)
                else:
                    # Track portfolio value but don't rebalance
                    track_portfolio_value(strategy, all_results, current_date)
        else:
            # No signal on this date, just track portfolio value if we have positions
            if strategy.current_portfolio is not None:
                track_portfolio_value(strategy, all_results, current_date)
    
    # Final portfolio valuation if we still have positions
    if strategy.current_portfolio is not None:
        final_date = all_dates[-1]
        exit_current_position(strategy, all_results, final_date, is_final=True)
    
    print(f"Strategy execution complete!")
    print(f"Total trades executed: {len(strategy.trade_history)}")
    print(f"Total portfolio history records: {len(strategy.portfolio_history)}")
    
    return strategy

def enter_new_position(strategy, all_results, entry_date, assets, signal_strength):
    """
    Enter new position with equal allocation across assets - enhanced with better tracking
    """
    # Calculate equal allocation
    allocation_per_asset = strategy.current_capital / len(assets)
    
    portfolio = {}
    total_invested = 0
    failed_assets = []
    
    for asset in assets:
        price = get_asset_price_on_date(all_results, asset, entry_date)
        
        if price is not None and price > 0:
            shares = allocation_per_asset / price
            portfolio[asset] = {
                'shares': shares,
                'entry_price': price,
                'entry_value': allocation_per_asset
            }
            total_invested += allocation_per_asset
        else:
            failed_assets.append(asset)
    
    # If some assets failed, redistribute capital among successful assets
    if failed_assets and portfolio:
        successful_assets = len(portfolio)
        new_allocation_per_asset = strategy.current_capital / successful_assets
        
        # Recalculate allocations
        portfolio = {}
        total_invested = 0
        
        for asset in assets:
            if asset not in failed_assets:
                price = get_asset_price_on_date(all_results, asset, entry_date)
                shares = new_allocation_per_asset / price
                portfolio[asset] = {
                    'shares': shares,
                    'entry_price': price,
                    'entry_value': new_allocation_per_asset
                }
                total_invested += new_allocation_per_asset
    
    if not portfolio:
        print(f"WARNING: Could not enter any positions on {entry_date.strftime('%Y-%m-%d')}")
        return
    
    strategy.current_portfolio = portfolio
    strategy.current_entry_date = entry_date
    
    # Record the entry with current portfolio value (should equal current_capital)
    strategy.portfolio_history.append({
        'date': entry_date,
        'action': 'ENTER',
        'portfolio_value': total_invested,
        'signal_strength': signal_strength,
        'assets': list(portfolio.keys()),
        'num_assets': len(portfolio),
        'allocation_per_asset': total_invested / len(portfolio) if portfolio else 0,
        'failed_assets': failed_assets,
        'total_capital_deployed': total_invested
    })
    
    print(f"ENTERED position on {entry_date.strftime('%Y-%m-%d')}")
    print(f"  Signal strength: {signal_strength:.2f}%")
    print(f"  Assets: {list(portfolio.keys())}")
    print(f"  Capital allocated: ${total_invested:,.2f}")
    if failed_assets:
        print(f"  Failed assets: {failed_assets}")


def exit_current_position(strategy, all_results, exit_date, is_final=False):
    """
    Exit current position and calculate P&L
    """
    if strategy.current_portfolio is None:
        return
    
    total_exit_value = 0
    asset_performance = {}
    
    for asset, position in strategy.current_portfolio.items():
        exit_price = get_asset_price_on_date(all_results, asset, exit_date)
        
        if exit_price is not None:
            exit_value = position['shares'] * exit_price
            pnl = exit_value - position['entry_value']
            pnl_pct = (exit_price / position['entry_price'] - 1) * 100
            
            asset_performance[asset] = {
                'entry_price': position['entry_price'],
                'exit_price': exit_price,
                'entry_value': position['entry_value'],
                'exit_value': exit_value,
                'pnl': pnl,
                'pnl_pct': pnl_pct
            }
            
            total_exit_value += exit_value
        else:
            # If we can't get exit price, assume no change
            total_exit_value += position['entry_value']
            asset_performance[asset] = {
                'entry_price': position['entry_price'],
                'exit_price': position['entry_price'],
                'entry_value': position['entry_value'],
                'exit_value': position['entry_value'],
                'pnl': 0,
                'pnl_pct': 0
            }
    
    # Calculate trade performance
    days_held = (exit_date - strategy.current_entry_date).days
    trade_pnl = total_exit_value - strategy.current_capital
    trade_pnl_pct = (total_exit_value / strategy.current_capital - 1) * 100
    
    # Record the trade
    strategy.trade_history.append({
        'entry_date': strategy.current_entry_date,
        'exit_date': exit_date,
        'days_held': days_held,
        'entry_capital': strategy.current_capital,
        'exit_value': total_exit_value,
        'trade_pnl': trade_pnl,
        'trade_pnl_pct': trade_pnl_pct,
        'assets': list(strategy.current_portfolio.keys()),
        'asset_performance': asset_performance
    })
    
    # Update strategy capital
    strategy.current_capital = total_exit_value
    
    action = "FINAL EXIT" if is_final else "REBALANCE"
    print(f"{action} on {exit_date.strftime('%Y-%m-%d')}")
    print(f"  Days held: {days_held}")
    print(f"  Trade P&L: ${trade_pnl:,.2f} ({trade_pnl_pct:.2f}%)")
    print(f"  New capital: ${strategy.current_capital:,.2f}")
    
    # Record the exit
    strategy.portfolio_history.append({
        'date': exit_date,
        'action': action,
        'portfolio_value': total_exit_value,
        'trade_pnl': trade_pnl,
        'trade_pnl_pct': trade_pnl_pct,
        'days_held': days_held,
        'assets': list(strategy.current_portfolio.keys())
    })
    
    # Clear current portfolio
    strategy.current_portfolio = None
    strategy.current_entry_date = None

def track_portfolio_value(strategy, all_results, current_date):
    """
    Track portfolio value without rebalancing - enhanced with better error handling
    """
    if strategy.current_portfolio is None:
        return
    
    total_value = 0
    assets_valued = 0
    missing_assets = []
    
    for asset, position in strategy.current_portfolio.items():
        current_price = get_asset_price_on_date(all_results, asset, current_date)
        
        if current_price is not None:
            current_value = position['shares'] * current_price
            total_value += current_value
            assets_valued += 1
        else:
            # If we can't get current price, use last known value or entry value
            # First try to get the most recent price before this date
            if asset in all_results:
                asset_df = all_results[asset]
                recent_data = asset_df[asset_df['date'] <= current_date].tail(1)
                
                if len(recent_data) > 0:
                    last_price = recent_data['price'].iloc[0]
                    current_value = position['shares'] * last_price
                    total_value += current_value
                    assets_valued += 1
                else:
                    # Use entry value as fallback
                    total_value += position['entry_value']
                    missing_assets.append(asset)
            else:
                # Use entry value as fallback
                total_value += position['entry_value']
                missing_assets.append(asset)
    
    # Calculate unrealized P&L based on the capital when this position was entered
    entry_capital = sum(pos['entry_value'] for pos in strategy.current_portfolio.values())
    unrealized_pnl = total_value - entry_capital
    unrealized_pnl_pct = (total_value / entry_capital - 1) * 100 if entry_capital > 0 else 0
    
    # Record portfolio value
    strategy.portfolio_history.append({
        'date': current_date,
        'action': 'HOLD',
        'portfolio_value': total_value,
        'unrealized_pnl': unrealized_pnl,
        'unrealized_pnl_pct': unrealized_pnl_pct,
        'days_held': (current_date - strategy.current_entry_date).days,
        'assets': list(strategy.current_portfolio.keys()),
        'assets_valued': assets_valued,
        'total_assets': len(strategy.current_portfolio),
        'missing_price_assets': missing_assets
    })

def analyze_strategy_performance(strategy):
    """
    Analyze the overall strategy performance
    """
    if not strategy.trade_history:
        print("No completed trades to analyze!")
        return {}
    
    trades_df = pd.DataFrame(strategy.trade_history)
    
    # Calculate performance metrics
    total_return = (strategy.current_capital / strategy.initial_capital - 1) * 100
    num_trades = len(trades_df)
    avg_trade_return = trades_df['trade_pnl_pct'].mean()
    win_rate = (trades_df['trade_pnl_pct'] > 0).mean() * 100
    avg_hold_period = trades_df['days_held'].mean()
    
    # Best and worst trades
    best_trade = trades_df.loc[trades_df['trade_pnl_pct'].idxmax()]
    worst_trade = trades_df.loc[trades_df['trade_pnl_pct'].idxmin()]
    
    # Annual returns (approximate)
    total_days = (trades_df['exit_date'].max() - trades_df['entry_date'].min()).days
    annualized_return = ((strategy.current_capital / strategy.initial_capital) ** (365.25 / total_days) - 1) * 100
    
    analysis = {
        'initial_capital': strategy.initial_capital,
        'final_capital': strategy.current_capital,
        'total_return_pct': total_return,
        'total_return_dollar': strategy.current_capital - strategy.initial_capital,
        'annualized_return_pct': annualized_return,
        'num_trades': num_trades,
        'avg_trade_return_pct': avg_trade_return,
        'win_rate_pct': win_rate,
        'avg_hold_period_days': avg_hold_period,
        'best_trade': best_trade,
        'worst_trade': worst_trade,
        'total_days': total_days,
        'volatility': trades_df['trade_pnl_pct'].std()
    }
    
    return analysis

def create_strategy_visualizations(strategy, analysis, output_dir):
    """
    Create comprehensive strategy performance visualizations with improved portfolio tracking
    """
    if not strategy.portfolio_history:
        print("No portfolio history to visualize!")
        return
    
    portfolio_df = pd.DataFrame(strategy.portfolio_history)
    portfolio_df['date'] = pd.to_datetime(portfolio_df['date'])
    portfolio_df = portfolio_df.sort_values('date').reset_index(drop=True)
    
    # Create comprehensive figure
    fig = plt.figure(figsize=(24, 16))
    
    # Plot 1: Portfolio value over time (main chart - spans full width)
    ax1 = plt.subplot(3, 3, (1, 3))  # Span first row
    
    # Plot portfolio value with different colors for different actions
    hold_data = portfolio_df[portfolio_df['action'] == 'HOLD']
    enter_data = portfolio_df[portfolio_df['action'] == 'ENTER']
    exit_data = portfolio_df[portfolio_df['action'].isin(['REBALANCE', 'FINAL EXIT'])]
    
    # Main portfolio line
    ax1.plot(portfolio_df['date'], portfolio_df['portfolio_value'], 
             linewidth=2, color='darkblue', label='Portfolio Value', alpha=0.8)
    
    # Highlight different actions
    if len(enter_data) > 0:
        ax1.scatter(enter_data['date'], enter_data['portfolio_value'], 
                   color='green', s=100, marker='^', label='Entry Points', zorder=5, alpha=0.8)
    
    if len(exit_data) > 0:
        ax1.scatter(exit_data['date'], exit_data['portfolio_value'], 
                   color='red', s=100, marker='v', label='Exit/Rebalance Points', zorder=5, alpha=0.8)
    
    # Add initial capital line
    ax1.axhline(y=strategy.initial_capital, color='gray', linestyle='--', 
                alpha=0.7, label=f'Initial Capital (${strategy.initial_capital:,})')
    
    # Add performance statistics to the plot
    if len(portfolio_df) > 0:
        final_value = portfolio_df['portfolio_value'].iloc[-1]
        total_return_pct = (final_value / strategy.initial_capital - 1) * 100
        
        ax1.text(0.02, 0.95, f'Total Return: {total_return_pct:.1f}%\nFinal Value: ${final_value:,.0f}', 
                transform=ax1.transAxes, fontsize=12, verticalalignment='top',
                bbox=dict(boxstyle='round', facecolor='white', alpha=0.8))
    
    ax1.set_title('Portfolio Value Over Time (Continuous Tracking)', fontweight='bold', fontsize=16)
    ax1.set_ylabel('Portfolio Value ($)')
    ax1.legend(loc='upper left')
    ax1.grid(True, alpha=0.3)
    ax1.xaxis.set_major_formatter(mdates.DateFormatter('%Y-%m'))
    ax1.xaxis.set_major_locator(mdates.YearLocator())
    plt.setp(ax1.xaxis.get_majorticklabels(), rotation=45)
    
    # Format y-axis as currency
    ax1.yaxis.set_major_formatter(plt.FuncFormatter(lambda x, p: f'${x:,.0f}'))
    
    # Plot 2: Daily returns (when available)
    ax2 = plt.subplot(3, 3, 4)
    if len(portfolio_df) > 1:
        # Calculate daily returns
        portfolio_df['daily_return'] = portfolio_df['portfolio_value'].pct_change() * 100
        portfolio_df['daily_return'] = portfolio_df['daily_return'].fillna(0)
        
        # Plot daily returns
        colors = ['green' if x >= 0 else 'red' for x in portfolio_df['daily_return']]
        ax2.bar(portfolio_df['date'], portfolio_df['daily_return'], 
               color=colors, alpha=0.6, width=1)
        
        ax2.axhline(0, color='black', linestyle='-', alpha=0.5)
        ax2.set_title('Daily Returns (%)', fontweight='bold')
        ax2.set_ylabel('Daily Return (%)')
        ax2.grid(True, alpha=0.3)
        ax2.xaxis.set_major_formatter(mdates.DateFormatter('%Y'))
        plt.setp(ax2.xaxis.get_majorticklabels(), rotation=45)
    
    # Plot 3: Trade returns distribution (if we have completed trades)
    ax3 = plt.subplot(3, 3, 5)
    if strategy.trade_history:
        trades_df = pd.DataFrame(strategy.trade_history)
        
        ax3.hist(trades_df['trade_pnl_pct'], bins=15, alpha=0.7, 
                color='steelblue', edgecolor='black')
        ax3.axvline(trades_df['trade_pnl_pct'].mean(), color='red', 
                   linestyle='--', linewidth=2, label=f'Mean: {trades_df["trade_pnl_pct"].mean():.1f}%')
        ax3.axvline(0, color='black', linestyle='-', alpha=0.5)
        
        ax3.set_title('Distribution of Trade Returns', fontweight='bold')
        ax3.set_xlabel('Trade Return (%)')
        ax3.set_ylabel('Frequency')
        ax3.legend()
        ax3.grid(True, alpha=0.3)
        ax3.xaxis.set_major_formatter(mdates.DateFormatter('%Y-%m'))
        plt.setp(ax3.xaxis.get_majorticklabels(), rotation=45)
    
    # Plot 4: Holding periods
    if strategy.trade_history:
        ax4 = plt.subplot(3, 3, 6)
        
        ax4.hist(trades_df['days_held'], bins=10, alpha=0.7, 
                color='orange', edgecolor='black')
        ax4.axvline(trades_df['days_held'].mean(), color='red', 
                   linestyle='--', linewidth=2, label=f'Mean: {trades_df["days_held"].mean():.0f} days')
        ax4.axvline(strategy.min_hold_days, color='green', 
                   linestyle='--', linewidth=2, label=f'Min Hold: {strategy.min_hold_days} days')
        
        ax4.set_title('Distribution of Holding Periods', fontweight='bold')
        ax4.set_xlabel('Days Held')
        ax4.set_ylabel('Frequency')
        ax4.legend()
        ax4.grid(True, alpha=0.3)
    
    # Plot 5: Cumulative returns
    ax5 = plt.subplot(3, 3, 7)
    
    # Calculate cumulative returns
    cumulative_returns = []
    cumulative_value = strategy.initial_capital
    
    for _, row in portfolio_df.iterrows():
        if row['action'] in ['REBALANCE', 'FINAL EXIT']:
            cumulative_value = row['portfolio_value']
        cumulative_return = (cumulative_value / strategy.initial_capital - 1) * 100
        cumulative_returns.append(cumulative_return)
    
    ax5.plot(portfolio_df['date'], cumulative_returns, 
             linewidth=2, color='darkgreen')
    ax5.axhline(0, color='black', linestyle='-', alpha=0.5)
    
    ax5.set_title('Cumulative Returns (%)', fontweight='bold')
    ax5.set_xlabel('Date')
    ax5.set_ylabel('Cumulative Return (%)')
    ax5.grid(True, alpha=0.3)
    ax5.xaxis.set_major_formatter(mdates.DateFormatter('%Y-%m'))
    plt.setp(ax5.xaxis.get_majorticklabels(), rotation=45)
    
    # Plot 6: Asset allocation frequency
    if strategy.trade_history:
        ax6 = plt.subplot(3, 3, 8)
        
        # Count asset appearances
        asset_counts = defaultdict(int)
        for trade in strategy.trade_history:
            for asset in trade['assets']:
                asset_counts[asset] += 1
        
        # Plot top 10 most frequent assets
        sorted_assets = sorted(asset_counts.items(), key=lambda x: x[1], reverse=True)[:10]
        assets, counts = zip(*sorted_assets) if sorted_assets else ([], [])
        
        bars = ax6.bar(range(len(assets)), counts, alpha=0.7, color='brown')
        ax6.set_title('Top 10 Most Traded Assets', fontweight='bold')
        ax6.set_xlabel('Asset')
        ax6.set_ylabel('Number of Trades')
        ax6.set_xticks(range(len(assets)))
        ax6.set_xticklabels(assets, rotation=45)
        ax6.grid(True, alpha=0.3)
        
        # Add value labels
        for bar, count in zip(bars, counts):
            ax6.text(bar.get_x() + bar.get_width()/2., bar.get_height() + 0.1,
                    f'{count}', ha='center', va='bottom', fontweight='bold')
    
    # Plot 7: Performance statistics summary
    ax7 = plt.subplot(3, 3, 9)
    ax7.axis('off')  # Turn off axis for text display
    
    # Create performance summary text
    perf_text = f"""Strategy Performance Summary:

Initial Capital: ${analysis['initial_capital']:,}
Final Capital: ${analysis['final_capital']:,.2f}
Total Return: {analysis['total_return_pct']:.2f}%
Annualized Return: {analysis['annualized_return_pct']:.2f}%

Trading Statistics:
• Number of Trades: {analysis['num_trades']}
• Average Trade Return: {analysis['avg_trade_return_pct']:.2f}%
• Win Rate: {analysis['win_rate_pct']:.1f}%
• Average Hold Period: {analysis['avg_hold_period_days']:.0f} days
• Return Volatility: {analysis['volatility']:.2f}%

Best Trade: {analysis['best_trade']['trade_pnl_pct']:.2f}%
Worst Trade: {analysis['worst_trade']['trade_pnl_pct']:.2f}%

Strategy Period: {analysis['total_days']} days
({analysis['total_days']/365.25:.1f} years)"""
    
    ax7.text(0.05, 0.95, perf_text, transform=ax7.transAxes, 
             fontsize=12, verticalalignment='top', fontfamily='monospace',
             bbox=dict(boxstyle='round,pad=0.5', facecolor='lightblue', alpha=0.8))
    
    plt.tight_layout()
    plt.savefig(f"{output_dir}/strategy_performance_analysis.png", dpi=300, bbox_inches='tight')
    plt.close()=0.3)
    else:
        ax3.text(0.5, 0.5, 'No Completed Trades Yet', 
                transform=ax3.transAxes, ha='center', va='center', fontsize=14)
        ax3.set_title('Trade Returns Distribution', fontweight='bold')
    
    # Plot 4: Drawdown analysis
    ax4 = plt.subplot(3, 3, 6)
    if len(portfolio_df) > 1:
        # Calculate running maximum and drawdown
        portfolio_df['running_max'] = portfolio_df['portfolio_value'].expanding().max()
        portfolio_df['drawdown'] = (portfolio_df['portfolio_value'] / portfolio_df['running_max'] - 1) * 100
        
        ax4.fill_between(portfolio_df['date'], portfolio_df['drawdown'], 0, 
                        color='red', alpha=0.3)
        ax4.plot(portfolio_df['date'], portfolio_df['drawdown'], 
                color='red', linewidth=1)
        
        max_drawdown = portfolio_df['drawdown'].min()
        ax4.axhline(max_drawdown, color='darkred', linestyle='--', 
                   label=f'Max Drawdown: {max_drawdown:.1f}%')
        
        ax4.set_title('Portfolio Drawdown (%)', fontweight='bold')
        ax4.set_ylabel('Drawdown (%)')
        ax4.legend()
        ax4.grid(True, alpha=0.3)
        ax4.xaxis.set_major_formatter(mdates.DateFormatter('%Y'))
        plt.setp(ax4.xaxis.get_majorticklabels(), rotation=45)
    
    # Plot 5: Portfolio composition tracking
    ax5 = plt.subplot(3, 3, 7)
    if 'num_assets' in portfolio_df.columns:
        # Track number of assets over time
        composition_data = portfolio_df[portfolio_df['action'].isin(['ENTER', 'HOLD'])].copy()
        if len(composition_data) > 0:
            ax5.plot(composition_data['date'], composition_data.get('num_assets', 0), 
                    linewidth=2, color='purple', marker='o', markersize=3)
            ax5.set_title('Number of Assets in Portfolio', fontweight='bold')
            ax5.set_ylabel('Number of Assets')
            ax5.grid(True, alpha=0.3)
            ax5.xaxis.set_major_formatter(mdates.DateFormatter('%Y'))
            plt.setp(ax5.xaxis.get_majorticklabels(), rotation=45)
        else:
            ax5.text(0.5, 0.5, 'No Asset Composition Data', 
                    transform=ax5.transAxes, ha='center', va='center')
    
    # Plot 6: Rolling performance metrics
    ax6 = plt.subplot(3, 3, 8)
    if len(portfolio_df) > 30:  # Need sufficient data for rolling metrics
        # Calculate 30-day rolling return
        portfolio_df['rolling_30d_return'] = portfolio_df['portfolio_value'].pct_change(periods=30) * 100
        
        # Plot rolling returns
        valid_rolling = portfolio_df.dropna(subset=['rolling_30d_return'])
        if len(valid_rolling) > 0:
            ax6.plot(valid_rolling['date'], valid_rolling['rolling_30d_return'], 
                    linewidth=2, color='orange')
            ax6.axhline(0, color='black', linestyle='-', alpha=0.5)
            ax6.set_title('30-Day Rolling Returns (%)', fontweight='bold')
            ax6.set_ylabel('30-Day Return (%)')
            ax6.grid(True, alpha=0.3)
            ax6.xaxis.set_major_formatter(mdates.DateFormatter('%Y'))
            plt.setp(ax6.xaxis.get_majorticklabels(), rotation=45)
    
    # Plot 7: Performance statistics summary
    ax7 = plt.subplot(3, 3, 9)
    ax7.axis('off')  # Turn off axis for text display
    
    # Calculate additional metrics from continuous tracking
    if len(portfolio_df) > 1:
        final_value = portfolio_df['portfolio_value'].iloc[-1]
        total_return = (final_value / strategy.initial_capital - 1) * 100
        
        # Calculate volatility from daily returns if available
        if 'daily_return' in portfolio_df.columns:
            volatility = portfolio_df['daily_return'].std() * np.sqrt(252)  # Annualized
        else:
            volatility = 0
        
        # Days in strategy
        total_days = (portfolio_df['date'].max() - portfolio_df['date'].min()).days
        
        # Create performance summary text
        perf_text = f"""Continuous Portfolio Tracking:

Portfolio Performance:
• Initial Value: ${strategy.initial_capital:,}
• Final Value: ${final_value:,.2f}
• Total Return: {total_return:.2f}%
• Strategy Period: {total_days} days

Risk Metrics:
• Daily Volatility: {volatility:.2f}% (annualized)
• Max Drawdown: {portfolio_df.get('drawdown', pd.Series([0])).min():.2f}%

Portfolio Activity:
• Total Positions: {len(portfolio_df[portfolio_df['action'] == 'ENTER'])}
• Rebalances: {len(portfolio_df[portfolio_df['action'] == 'REBALANCE'])}
• Tracking Days: {len(portfolio_df):,}

Data Quality:
• Continuous Tracking: {'✓' if len(portfolio_df) > total_days * 0.8 else '✗'}
• Price Coverage: {'✓' if portfolio_df.get('assets_valued', pd.Series([0])).mean() > 4 else '⚠️'}"""
    else:
        perf_text = "Insufficient data for analysis"
    
    ax7.text(0.05, 0.95, perf_text, transform=ax7.transAxes, 
             fontsize=11, verticalalignment='top', fontfamily='monospace',
             bbox=dict(boxstyle='round,pad=0.5', facecolor='lightblue', alpha=0.8))
    
    plt.tight_layout()
    plt.savefig(f"{output_dir}/strategy_performance_analysis.png", dpi=300, bbox_inches='tight')
    plt.close()
    
    print(f"Created enhanced visualization with {len(portfolio_df)} data points")=0.3)
        ax3.xaxis.set_major_formatter(mdates.DateFormatter('%Y-%m'))
        plt.setp(ax3.xaxis.get_majorticklabels(), rotation=45)
    
    # Plot 4: Holding periods
    if strategy.trade_history:
        ax4 = plt.subplot(3, 3, 6)
        
        ax4.hist(trades_df['days_held'], bins=10, alpha=0.7, 
                color='orange', edgecolor='black')
        ax4.axvline(trades_df['days_held'].mean(), color='red', 
                   linestyle='--', linewidth=2, label=f'Mean: {trades_df["days_held"].mean():.0f} days')
        ax4.axvline(strategy.min_hold_days, color='green', 
                   linestyle='--', linewidth=2, label=f'Min Hold: {strategy.min_hold_days} days')
        
        ax4.set_title('Distribution of Holding Periods', fontweight='bold')
        ax4.set_xlabel('Days Held')
        ax4.set_ylabel('Frequency')
        ax4.legend()
        ax4.grid(True, alpha=0.3)
    
    # Plot 5: Cumulative returns
    ax5 = plt.subplot(3, 3, 7)
    
    # Calculate cumulative returns
    cumulative_returns = []
    cumulative_value = strategy.initial_capital
    
    for _, row in portfolio_df.iterrows():
        if row['action'] in ['REBALANCE', 'FINAL EXIT']:
            cumulative_value = row['portfolio_value']
        cumulative_return = (cumulative_value / strategy.initial_capital - 1) * 100
        cumulative_returns.append(cumulative_return)
    
    ax5.plot(portfolio_df['date'], cumulative_returns, 
             linewidth=2, color='darkgreen')
    ax5.axhline(0, color='black', linestyle='-', alpha=0.5)
    
    ax5.set_title('Cumulative Returns (%)', fontweight='bold')
    ax5.set_xlabel('Date')
    ax5.set_ylabel('Cumulative Return (%)')
    ax5.grid(True, alpha=0.3)
    ax5.xaxis.set_major_formatter(mdates.DateFormatter('%Y-%m'))
    plt.setp(ax5.xaxis.get_majorticklabels(), rotation=45)
    
    # Plot 6: Asset allocation frequency
    if strategy.trade_history:
        ax6 = plt.subplot(3, 3, 8)
        
        # Count asset appearances
        asset_counts = defaultdict(int)
        for trade in strategy.trade_history:
            for asset in trade['assets']:
                asset_counts[asset] += 1
        
        # Plot top 10 most frequent assets
        sorted_assets = sorted(asset_counts.items(), key=lambda x: x[1], reverse=True)[:10]
        assets, counts = zip(*sorted_assets) if sorted_assets else ([], [])
        
        bars = ax6.bar(range(len(assets)), counts, alpha=0.7, color='brown')
        ax6.set_title('Top 10 Most Traded Assets', fontweight='bold')
        ax6.set_xlabel('Asset')
        ax6.set_ylabel('Number of Trades')
        ax6.set_xticks(range(len(assets)))
        ax6.set_xticklabels(assets, rotation=45)
        ax6.grid(True, alpha=0.3)
        
        # Add value labels
        for bar, count in zip(bars, counts):
            ax6.text(bar.get_x() + bar.get_width()/2., bar.get_height() + 0.1,
                    f'{count}', ha='center', va='bottom', fontweight='bold')
    
    # Plot 7: Performance statistics summary
    ax7 = plt.subplot(3, 3, 9)
    ax7.axis('off')  # Turn off axis for text display
    
    # Create performance summary text
    perf_text = f"""Strategy Performance Summary:

Initial Capital: ${analysis['initial_capital']:,}
Final Capital: ${analysis['final_capital']:,.2f}
Total Return: {analysis['total_return_pct']:.2f}%
Annualized Return: {analysis['annualized_return_pct']:.2f}%

Trading Statistics:
• Number of Trades: {analysis['num_trades']}
• Average Trade Return: {analysis['avg_trade_return_pct']:.2f}%
• Win Rate: {analysis['win_rate_pct']:.1f}%
• Average Hold Period: {analysis['avg_hold_period_days']:.0f} days
• Return Volatility: {analysis['volatility']:.2f}%

Best Trade: {analysis['best_trade']['trade_pnl_pct']:.2f}%
Worst Trade: {analysis['worst_trade']['trade_pnl_pct']:.2f}%

Strategy Period: {analysis['total_days']} days
({analysis['total_days']/365.25:.1f} years)"""
    
    ax7.text(0.05, 0.95, perf_text, transform=ax7.transAxes, 
             fontsize=12, verticalalignment='top', fontfamily='monospace',
             bbox=dict(boxstyle='round,pad=0.5', facecolor='lightblue', alpha=0.8))
    
    plt.tight_layout()
    plt.savefig(f"{output_dir}/strategy_performance_analysis.png", dpi=300, bbox_inches='tight')
    plt.close()

def save_strategy_results(strategy, analysis, daily_signals, threshold, output_dir):
    """
    Save detailed strategy results and analysis
    """
    # Save trade history
    if strategy.trade_history:
        trades_df = pd.DataFrame(strategy.trade_history)
        trades_df['entry_date'] = trades_df['entry_date'].dt.strftime('%Y-%m-%d')
        trades_df['exit_date'] = trades_df['exit_date'].dt.strftime('%Y-%m-%d')
        
        # Flatten asset performance for CSV
        trades_summary = []
        for _, trade in trades_df.iterrows():
            base_info = {
                'entry_date': trade['entry_date'],
                'exit_date': trade['exit_date'],
                'days_held': trade['days_held'],
                'entry_capital': trade['entry_capital'],
                'exit_value': trade['exit_value'],
                'trade_pnl': trade['trade_pnl'],
                'trade_pnl_pct': trade['trade_pnl_pct'],
                'num_assets': len(trade['assets']),
                'assets': ', '.join(trade['assets'])
            }
            trades_summary.append(base_info)
        
        trades_summary_df = pd.DataFrame(trades_summary)
        trades_summary_df.to_csv(f"{output_dir}/trade_history.csv", index=False)
    
    # Save portfolio history
    if strategy.portfolio_history:
        portfolio_df = pd.DataFrame(strategy.portfolio_history)
        portfolio_df['date'] = portfolio_df['date'].dt.strftime('%Y-%m-%d')
        portfolio_df.to_csv(f"{output_dir}/portfolio_history.csv", index=False)
    
    # Save performance analysis
    perf_summary = {
        'metric': ['initial_capital', 'final_capital', 'total_return_pct', 'total_return_dollar',
                  'annualized_return_pct', 'num_trades', 'avg_trade_return_pct', 'win_rate_pct',
                  'avg_hold_period_days', 'volatility', 'total_days', 'threshold_used'],
        'value': [analysis['initial_capital'], analysis['final_capital'], analysis['total_return_pct'],
                 analysis['total_return_dollar'], analysis['annualized_return_pct'], analysis['num_trades'],
                 analysis['avg_trade_return_pct'], analysis['win_rate_pct'], analysis['avg_hold_period_days'],
                 analysis['volatility'], analysis['total_days'], threshold]
    }
    
    perf_df = pd.DataFrame(perf_summary)
    perf_df.to_csv(f"{output_dir}/performance_summary.csv", index=False)
    
    # Save signal summary
    signals_summary = []
    valid_signals = [signal for signal in daily_signals 
                    if signal['sum_top5_absolute_deviations'] >= threshold]
    
    for signal in valid_signals:
        signals_summary.append({
            'date': signal['date'].strftime('%Y-%m-%d'),
            'signal_strength': signal['sum_top5_absolute_deviations'],
            'assets': ', '.join(signal['top5_assets']),
            'num_assets': len(signal['top5_assets'])
        })
    
    signals_df = pd.DataFrame(signals_summary)
    signals_df.to_csv(f"{output_dir}/valid_signals.csv", index=False)
    
    print(f"\nSaved strategy results:")
    print(f"  - trade_history.csv ({len(strategy.trade_history)} trades)")
    print(f"  - portfolio_history.csv ({len(strategy.portfolio_history)} records)")
    print(f"  - performance_summary.csv")
    print(f"  - valid_signals.csv ({len(valid_signals)} signals)")

def print_strategy_summary(strategy, analysis, threshold):
    """
    Print comprehensive strategy performance summary
    """
    print("\n" + "="*80)
    print("REBALANCING TRADING STRATEGY RESULTS")
    print("="*80)
    
    print(f"\nSTRATEGY PARAMETERS:")
    print(f"  Initial capital: ${strategy.initial_capital:,}")
    print(f"  Minimum hold period: {strategy.min_hold_days} days")
    print(f"  Signal threshold: {threshold:.2f}% (80th percentile)")
    
    print(f"\nOVERALL PERFORMANCE:")
    print(f"  Final capital: ${analysis['final_capital']:,.2f}")
    print(f"  Total return: ${analysis['total_return_dollar']:,.2f} ({analysis['total_return_pct']:.2f}%)")
    print(f"  Annualized return: {analysis['annualized_return_pct']:.2f}%")
    print(f"  Strategy period: {analysis['total_days']} days ({analysis['total_days']/365.25:.1f} years)")
    
    print(f"\nTRADING STATISTICS:")
    print(f"  Number of completed trades: {analysis['num_trades']}")
    print(f"  Average trade return: {analysis['avg_trade_return_pct']:.2f}%")
    print(f"  Win rate: {analysis['win_rate_pct']:.1f}%")
    print(f"  Average holding period: {analysis['avg_hold_period_days']:.0f} days")
    print(f"  Return volatility: {analysis['volatility']:.2f}%")
    
    if analysis['num_trades'] > 0:
        print(f"\nBEST & WORST TRADES:")
        best = analysis['best_trade']
        worst = analysis['worst_trade']
        print(f"  Best trade: {best['exit_date']} ({best['trade_pnl_pct']:.2f}%, {best['days_held']} days)")
        print(f"  Worst trade: {worst['exit_date']} ({worst['trade_pnl_pct']:.2f}%, {worst['days_held']} days)")
    
    print(f"\nSTRATEGY INSIGHTS:")
    if analysis['annualized_return_pct'] > 10:
        print(f"  - Strong performance: {analysis['annualized_return_pct']:.1f}% annualized return")
    elif analysis['annualized_return_pct'] > 5:
        print(f"  - Moderate performance: {analysis['annualized_return_pct']:.1f}% annualized return")
    else:
        print(f"  - Conservative performance: {analysis['annualized_return_pct']:.1f}% annualized return")
    
    if analysis['win_rate_pct'] > 60:
        print(f"  - High win rate: {analysis['win_rate_pct']:.0f}% of trades profitable")
    elif analysis['win_rate_pct'] > 50:
        print(f"  - Balanced win rate: {analysis['win_rate_pct']:.0f}% of trades profitable")
    else:
        print(f"  - Lower win rate: {analysis['win_rate_pct']:.0f}% of trades profitable")
    
    if analysis['avg_hold_period_days'] > strategy.min_hold_days * 1.5:
        print(f"  - Long average holds: {analysis['avg_hold_period_days']:.0f} days (patient strategy)")
    else:
        print(f"  - Efficient rebalancing: {analysis['avg_hold_period_days']:.0f} days average hold")
    
    print(f"\nRISK ASSESSMENT:")
    sharpe_approx = analysis['annualized_return_pct'] / analysis['volatility'] if analysis['volatility'] > 0 else 0
    print(f"  Risk-adjusted return (approx Sharpe): {sharpe_approx:.2f}")
    
    if analysis['volatility'] < 15:
        print(f"  - Low volatility strategy ({analysis['volatility']:.1f}%)")
    elif analysis['volatility'] < 25:
        print(f"  - Moderate volatility strategy ({analysis['volatility']:.1f}%)")
    else:
        print(f"  - High volatility strategy ({analysis['volatility']:.1f}%)")
    
    print(f"\nRECOMMendations:")
    if analysis['num_trades'] < 5:
        print(f"  - Limited sample size ({analysis['num_trades']} trades) - extend backtest period")
    
    if analysis['win_rate_pct'] < 50 and analysis['avg_trade_return_pct'] > 0:
        print(f"  - Strategy relies on large winners to offset frequent small losses")
        print(f"  - Consider tighter risk management or position sizing")
    
    if analysis['avg_hold_period_days'] > strategy.min_hold_days * 2:
        print(f"  - Consider reducing minimum hold period for more frequent rebalancing")
    
    if sharpe_approx > 1.0:
        print(f"  - Strong risk-adjusted returns - consider increasing position sizes")
    elif sharpe_approx < 0.5:
        print(f"  - Poor risk-adjusted returns - review strategy parameters")

def main():
    print("Highest Quintile Rebalancing Trading Strategy")
    print("="*60)
    print("Strategy: Enter top 5 deviation assets with $10k, rebalance after 200+ days on new signals")
    
    # Strategy parameters
    initial_capital = 10000
    min_hold_days = 200
    threshold_percentile = 80
    
    # Create output directory
    output_dir = "/Users/tim/IWLS-OPTIONS/REBALANCING_STRATEGY"
    os.makedirs(output_dir, exist_ok=True)
    print(f"\nOutput directory: {output_dir}")
    
    # Initialize strategy
    strategy = TradingStrategy(
        initial_capital=initial_capital,
        min_hold_days=min_hold_days,
        rebalance_threshold_percentile=threshold_percentile
    )
    
    # Load all IWLS results
    print(f"\nLoading asset data...")
    all_results = load_all_iwls_results()
    if not all_results:
        return
    
    print(f"\nLoaded {len(all_results)} assets")
    
    # Calculate daily signals and threshold
    daily_signals, threshold = calculate_daily_signals_and_threshold(all_results)
    
    if not daily_signals:
        print("No signals found for analysis!")
        return
    
    # Execute trading strategy
    strategy = execute_trading_strategy(all_results, daily_signals, threshold, strategy)
    
    # Analyze performance
    analysis = analyze_strategy_performance(strategy)
    
    if not analysis:
        print("No completed trades to analyze!")
        return
    
    # Create visualizations
    print(f"\nCreating performance visualizations...")
    create_strategy_visualizations(strategy, analysis, output_dir)
    
    # Save results
    save_strategy_results(strategy, analysis, daily_signals, threshold, output_dir)
    
    # Print comprehensive summary
    print_strategy_summary(strategy, analysis, threshold)
    
    print(f"\n" + "="*80)
    print("REBALANCING STRATEGY ANALYSIS COMPLETE")
    print("="*80)
    print("Files created:")
    print("  - strategy_performance_analysis.png (9-panel performance dashboard)")
    print("  - trade_history.csv (detailed trade records)")
    print("  - portfolio_history.csv (daily portfolio tracking)")
    print("  - performance_summary.csv (key metrics)")
    print("  - valid_signals.csv (all trading signals used)")
    
    print(f"\nKEY TAKEAWAYS:")
    if analysis['total_return_pct'] > 0:
        print(f"  ✓ Strategy was profitable: {analysis['total_return_pct']:.1f}% total return")
    else:
        print(f"  ✗ Strategy lost money: {analysis['total_return_pct']:.1f}% total return")
    
    print(f"  📊 Completed {analysis['num_trades']} trades over {analysis['total_days']/365.25:.1f} years")
    print(f"  📈 Average trade: {analysis['avg_trade_return_pct']:.1f}% return")
    print(f"  🎯 Win rate: {analysis['win_rate_pct']:.0f}%")
    print(f"  ⏱️  Average hold: {analysis['avg_hold_period_days']:.0f} days")
    
    # Strategy vs buy-and-hold comparison hint
    print(f"\nNEXT STEPS:")
    print(f"  - Compare against buy-and-hold benchmark")
    print(f"  - Test different minimum hold periods (100, 150, 250 days)")
    print(f"  - Analyze performance during different market conditions")
    print(f"  - Consider risk management (stop losses, position sizing)")

if __name__ == "__main__":
    main()