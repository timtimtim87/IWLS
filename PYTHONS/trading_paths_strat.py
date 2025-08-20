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

def calculate_annualized_return(start_value, end_value, start_date, end_date):
    """
    Calculate annualized return for individual trades
    """
    days_held = (end_date - start_date).days
    if days_held <= 0:
        return 0
    
    years = days_held / 365.25
    total_return = (end_value / start_value) - 1
    
    # For very short holding periods, use simple annualization
    if years < 0.01:  # Less than ~4 days
        return (total_return * 365.25 / days_held) * 100
    
    # Standard compound annualization
    annualized_return = ((1 + total_return) ** (1 / years) - 1) * 100
    return annualized_return

class DeviationReversionPortfolio:
    """
    Portfolio that holds 5 most underperforming assets and exits when they return to zero deviation
    """
    def __init__(self, initial_capital=10000, num_positions=5):
        self.initial_capital = initial_capital
        self.num_positions = num_positions
        self.cash = initial_capital
        self.positions = {}  # {asset: position_data}
        self.completed_trades = []
        self.portfolio_history = []
        self.rebalance_history = []
        
    def get_current_deviations(self, all_results, current_date):
        """
        Get current price deviations for all assets
        """
        deviations = {}
        
        for asset_name, df in all_results.items():
            asset_data = df[df['date'] <= current_date]
            if len(asset_data) == 0:
                continue
            
            latest = asset_data.iloc[-1]
            if latest['date'] != current_date:
                continue  # No data for this exact date
            
            deviations[asset_name] = {
                'deviation': latest['price_deviation'],
                'price': latest['price'],
                'date': latest['date']
            }
        
        return deviations
    
    def find_most_underperforming_assets(self, deviations, exclude_assets=None):
        """
        Find the N most underperforming assets (most negative deviations)
        """
        if exclude_assets is None:
            exclude_assets = set()
        
        # Filter out excluded assets and sort by deviation (most negative first)
        available_assets = {k: v for k, v in deviations.items() if k not in exclude_assets}
        
        # Sort by deviation (most negative = most underperforming)
        sorted_assets = sorted(available_assets.items(), key=lambda x: x[1]['deviation'])
        
        return sorted_assets[:self.num_positions]
    
    def check_exit_conditions(self, current_deviations):
        """
        Check which positions should exit (deviation >= 0)
        """
        exits = []
        
        for asset, position in self.positions.items():
            if asset in current_deviations:
                current_deviation = current_deviations[asset]['deviation']
                current_price = current_deviations[asset]['price']
                
                # Exit when deviation returns to zero or positive
                if current_deviation >= 0:
                    exits.append({
                        'asset': asset,
                        'current_price': current_price,
                        'current_deviation': current_deviation
                    })
        
        return exits
    
    def execute_exits(self, exits, current_date):
        """
        Execute all exits and add cash back to portfolio
        """
        for exit_data in exits:
            asset = exit_data['asset']
            current_price = exit_data['current_price']
            current_deviation = exit_data['current_deviation']
            
            position = self.positions[asset]
            
            # Calculate trade results
            shares = position['shares']
            proceeds = shares * current_price
            profit = proceeds - position['investment']
            profit_pct = (profit / position['investment']) * 100
            days_held = (current_date - position['entry_date']).days
            
            # Calculate annualized return
            annualized_return = calculate_annualized_return(
                position['entry_price'], current_price, position['entry_date'], current_date
            )
            
            # Add proceeds back to cash
            self.cash += proceeds
            
            # Record completed trade
            completed_trade = {
                'asset': asset,
                'entry_date': position['entry_date'],
                'exit_date': current_date,
                'entry_price': position['entry_price'],
                'exit_price': current_price,
                'entry_deviation': position['entry_deviation'],
                'exit_deviation': current_deviation,
                'shares': shares,
                'investment': position['investment'],
                'proceeds': proceeds,
                'profit': profit,
                'profit_pct': profit_pct,
                'days_held': days_held,
                'annualized_return': annualized_return
            }
            
            self.completed_trades.append(completed_trade)
            
            print(f"  EXIT: {asset} at {current_date.strftime('%Y-%m-%d')} - "
                  f"deviation {position['entry_deviation']:.1f}% → {current_deviation:.1f}%, "
                  f"{profit_pct:+.1f}% ({annualized_return:+.1f}% annual) in {days_held} days")
            
            # Remove from positions
            del self.positions[asset]
    
    def execute_entries(self, target_assets, current_date):
        """
        Execute entries into new positions
        """
        if len(target_assets) == 0:
            return
        
        # Calculate how much to invest per position
        available_positions = self.num_positions - len(self.positions)
        if available_positions <= 0:
            return
        
        # Only enter positions we don't already have
        new_assets = []
        for asset_name, asset_data in target_assets:
            if asset_name not in self.positions:
                new_assets.append((asset_name, asset_data))
        
        new_assets = new_assets[:available_positions]
        
        if len(new_assets) == 0:
            return
        
        investment_per_asset = self.cash / len(new_assets)
        
        print(f"\n  REBALANCING at {current_date.strftime('%Y-%m-%d')} - Available cash: ${self.cash:.2f}")
        print(f"  Entering {len(new_assets)} new positions (${investment_per_asset:.2f} each):")
        
        for asset_name, asset_data in new_assets:
            if self.cash < investment_per_asset:
                break
            
            price = asset_data['price']
            deviation = asset_data['deviation']
            shares = investment_per_asset / price
            
            # Create position
            position = {
                'asset': asset_name,
                'entry_date': current_date,
                'entry_price': price,
                'entry_deviation': deviation,
                'shares': shares,
                'investment': investment_per_asset
            }
            
            self.positions[asset_name] = position
            self.cash -= investment_per_asset
            
            print(f"    ENTER: {asset_name} - deviation {deviation:.1f}%, "
                  f"${price:.2f}/share, {shares:.2f} shares, ${investment_per_asset:.2f} invested")
    
    def get_portfolio_value(self, current_deviations):
        """
        Calculate current portfolio value
        """
        position_value = 0
        
        for asset, position in self.positions.items():
            if asset in current_deviations:
                current_price = current_deviations[asset]['price']
                position_value += position['shares'] * current_price
        
        return self.cash + position_value
    
    def record_portfolio_state(self, current_date, current_deviations):
        """
        Record current portfolio state
        """
        portfolio_value = self.get_portfolio_value(current_deviations)
        
        # Get position details
        position_details = []
        for asset, position in self.positions.items():
            if asset in current_deviations:
                current_price = current_deviations[asset]['price']
                current_deviation = current_deviations[asset]['deviation']
                current_value = position['shares'] * current_price
                unrealized_pnl = current_value - position['investment']
                unrealized_pct = (unrealized_pnl / position['investment']) * 100
                
                position_details.append({
                    'asset': asset,
                    'entry_deviation': position['entry_deviation'],
                    'current_deviation': current_deviation,
                    'current_value': current_value,
                    'investment': position['investment'],
                    'unrealized_pnl': unrealized_pnl,
                    'unrealized_pct': unrealized_pct
                })
        
        self.portfolio_history.append({
            'date': current_date,
            'portfolio_value': portfolio_value,
            'cash': self.cash,
            'position_value': portfolio_value - self.cash,
            'num_positions': len(self.positions),
            'positions': position_details.copy()
        })

def run_deviation_reversion_strategy(all_results, initial_capital=10000, num_positions=5, 
                                   min_deviation_threshold=-10, rebalance_frequency_days=7):
    """
    Run the deviation reversion strategy
    """
    print(f"\nRunning Deviation Reversion Strategy:")
    print(f"Initial capital: ${initial_capital:,}")
    print(f"Number of positions: {num_positions}")
    print(f"Entry: {num_positions} most underperforming assets (below {min_deviation_threshold}%)")
    print(f"Exit: When deviation returns to 0% or positive")
    print(f"Rebalance frequency: Every {rebalance_frequency_days} days")
    
    portfolio = DeviationReversionPortfolio(initial_capital, num_positions)
    
    # Create timeline
    all_data_points = []
    for asset_name, df in all_results.items():
        for _, row in df.iterrows():
            all_data_points.append({
                'date': row['date'],
                'asset': asset_name,
                'price': row['price'],
                'price_deviation': row['price_deviation']
            })
    
    # Group by date
    dates_data = {}
    for point in all_data_points:
        date = point['date']
        if date not in dates_data:
            dates_data[date] = []
        dates_data[date].append(point)
    
    print(f"\nProcessing {len(dates_data)} unique dates...")
    
    processed_dates = 0
    last_rebalance_date = None
    
    for current_date in sorted(dates_data.keys()):
        processed_dates += 1
        
        if processed_dates % 250 == 0:  # Progress every ~1 year
            print(f"  Processed {processed_dates}/{len(dates_data)} dates ({current_date.strftime('%Y-%m-%d')})...")
        
        # Get current deviations for all assets
        current_deviations = portfolio.get_current_deviations(all_results, current_date)
        
        if len(current_deviations) == 0:
            continue
        
        # Check for exits first (any day)
        exits = portfolio.check_exit_conditions(current_deviations)
        if exits:
            portfolio.execute_exits(exits, current_date)
        
        # Check if we need to rebalance (find new entries)
        should_rebalance = False
        
        # Rebalance conditions:
        if last_rebalance_date is None:
            should_rebalance = True  # First rebalance
        elif len(portfolio.positions) < num_positions:
            # If we have empty slots due to exits
            days_since_rebalance = (current_date - last_rebalance_date).days
            if days_since_rebalance >= rebalance_frequency_days:
                should_rebalance = True
        
        if should_rebalance:
            # Find most underperforming assets that meet minimum threshold
            eligible_deviations = {k: v for k, v in current_deviations.items() 
                                 if v['deviation'] <= min_deviation_threshold}
            
            if len(eligible_deviations) > 0:
                # Get current position assets to exclude
                current_assets = set(portfolio.positions.keys())
                
                # Find most underperforming assets
                target_assets = portfolio.find_most_underperforming_assets(
                    eligible_deviations, exclude_assets=current_assets
                )
                
                if target_assets:
                    portfolio.execute_entries(target_assets, current_date)
                    last_rebalance_date = current_date
        
        # Record portfolio state weekly
        if processed_dates % 5 == 0:  # Every 5 days
            portfolio.record_portfolio_state(current_date, current_deviations)
    
    # Final portfolio state
    final_date = max(dates_data.keys())
    final_deviations = portfolio.get_current_deviations(all_results, final_date)
    
    # Force exit any remaining positions
    if portfolio.positions and final_deviations:
        print(f"\nForce exiting remaining positions at end of data...")
        remaining_exits = []
        
        for asset, position in portfolio.positions.items():
            if asset in final_deviations:
                remaining_exits.append({
                    'asset': asset,
                    'current_price': final_deviations[asset]['price'],
                    'current_deviation': final_deviations[asset]['deviation']
                })
        
        portfolio.execute_exits(remaining_exits, final_date)
    
# Final portfolio recording
    portfolio.record_portfolio_state(final_date, final_deviations)
    
    return portfolio

def analyze_deviation_strategy_results(portfolio):
    """
    Analyze the deviation reversion strategy results
    """
    print("\n" + "="*80)
    print("DEVIATION REVERSION STRATEGY RESULTS")
    print("="*80)
    
    final_value = portfolio.cash  # All positions should be closed
    total_return = ((final_value / portfolio.initial_capital) - 1) * 100
    
    print(f"\nOVERALL PERFORMANCE:")
    print(f"Initial capital: ${portfolio.initial_capital:,.2f}")
    print(f"Final value: ${final_value:,.2f}")
    print(f"Total profit: ${final_value - portfolio.initial_capital:,.2f}")
    print(f"Total return: {total_return:.2f}%")
    
    # Calculate annualized return
    if portfolio.completed_trades:
        first_trade = min(portfolio.completed_trades, key=lambda x: x['entry_date'])
        last_trade = max(portfolio.completed_trades, key=lambda x: x['exit_date'])
        strategy_years = (last_trade['exit_date'] - first_trade['entry_date']).days / 365.25
        
        if strategy_years > 0:
            annualized_return = ((final_value / portfolio.initial_capital) ** (1 / strategy_years) - 1) * 100
            print(f"Annualized return: {annualized_return:.2f}%")
            print(f"Strategy duration: {strategy_years:.1f} years")
    
    # Trading statistics
    if portfolio.completed_trades:
        trades_df = pd.DataFrame(portfolio.completed_trades)
        
        print(f"\nTRADING STATISTICS:")
        print(f"Total trades completed: {len(trades_df)}")
        print(f"Winning trades: {len(trades_df[trades_df['profit'] > 0])} "
              f"({len(trades_df[trades_df['profit'] > 0])/len(trades_df)*100:.1f}%)")
        print(f"Average trade return: {trades_df['profit_pct'].mean():.1f}%")
        print(f"Median trade return: {trades_df['profit_pct'].median():.1f}%")
        print(f"Average annualized return per trade: {trades_df['annualized_return'].mean():.1f}%")
        print(f"Average holding period: {trades_df['days_held'].mean():.1f} days")
        print(f"Median holding period: {trades_df['days_held'].median():.1f} days")
        
        # Best and worst trades
        best_trade = trades_df.loc[trades_df['profit_pct'].idxmax()]
        worst_trade = trades_df.loc[trades_df['profit_pct'].idxmin()]
        
        print(f"\nBEST TRADE:")
        print(f"  {best_trade['asset']}: {best_trade['entry_deviation']:.1f}% → {best_trade['exit_deviation']:.1f}% "
              f"deviation, {best_trade['profit_pct']:+.1f}% return in {best_trade['days_held']} days")
        
        print(f"\nWORST TRADE:")
        print(f"  {worst_trade['asset']}: {worst_trade['entry_deviation']:.1f}% → {worst_trade['exit_deviation']:.1f}% "
              f"deviation, {worst_trade['profit_pct']:+.1f}% return in {worst_trade['days_held']} days")
        
        # Asset performance analysis
        asset_performance = trades_df.groupby('asset').agg({
            'profit_pct': ['count', 'mean', 'sum'],
            'days_held': 'mean',
            'entry_deviation': 'mean'
        }).round(2)
        
        asset_performance.columns = ['Trade_Count', 'Avg_Return', 'Total_Return', 'Avg_Days', 'Avg_Entry_Dev']
        asset_performance = asset_performance.sort_values('Avg_Return', ascending=False)
        
        print(f"\nTOP 10 ASSETS BY AVERAGE RETURN:")
        print("-" * 75)
        print(f"{'Asset':<6} {'Trades':<7} {'Avg Ret':<8} {'Total Ret':<9} {'Avg Days':<8} {'Avg Entry Dev':<12}")
        print("-" * 75)
        
        for asset, row in asset_performance.head(10).iterrows():
            print(f"{asset:<6} {row['Trade_Count']:<7.0f} {row['Avg_Return']:<8.1f}% "
                  f"{row['Total_Return']:<8.1f}% {row['Avg_Days']:<8.1f} {row['Avg_Entry_Dev']:<12.1f}%")
        
        # Deviation analysis
        print(f"\nDEVIATION REVERSION ANALYSIS:")
        print(f"Average entry deviation: {trades_df['entry_deviation'].mean():.1f}%")
        print(f"Average exit deviation: {trades_df['exit_deviation'].mean():.1f}%")
        print(f"Average deviation improvement: {trades_df['exit_deviation'].mean() - trades_df['entry_deviation'].mean():.1f}%")
        
        # Holding period analysis
        print(f"\nHOLDING PERIOD BREAKDOWN:")
        bins = [0, 30, 90, 180, 365, 730, 10000]
        labels = ['<30d', '30-90d', '90-180d', '180-365d', '1-2y', '>2y']
        
        trades_df['holding_period_bin'] = pd.cut(trades_df['days_held'], bins=bins, labels=labels)
        
        for label in labels:
            bin_trades = trades_df[trades_df['holding_period_bin'] == label]
            if len(bin_trades) > 0:
                avg_return = bin_trades['profit_pct'].mean()
                count = len(bin_trades)
                print(f"  {label:<8}: {count:>3} trades, avg return {avg_return:>6.1f}%")
        
        return trades_df
    else:
        print("\nNo completed trades found!")
        return None

def create_deviation_strategy_visualizations(portfolio, trades_df, output_dir):
    """
    Create visualizations for deviation reversion strategy
    """
    fig, ((ax1, ax2), (ax3, ax4)) = plt.subplots(2, 2, figsize=(20, 12))
    
    # Plot 1: Portfolio value over time
    if portfolio.portfolio_history:
        history_df = pd.DataFrame(portfolio.portfolio_history)
        ax1.plot(history_df['date'], history_df['portfolio_value'], 
                linewidth=3, color='darkblue', label='Portfolio Value')
        ax1.axhline(y=portfolio.initial_capital, color='red', linestyle='--', 
                   alpha=0.7, label=f'Initial Capital (${portfolio.initial_capital:,})')
        
        ax1.set_title('Portfolio Value Over Time', fontweight='bold')
        ax1.set_xlabel('Date')
        ax1.set_ylabel('Portfolio Value ($)')
        ax1.legend()
        ax1.grid(True, alpha=0.3)
        ax1.yaxis.set_major_formatter(plt.FuncFormatter(lambda x, p: f'${x:,.0f}'))
    
    # Plot 2: Entry vs Exit Deviation Analysis
    if trades_df is not None and len(trades_df) > 0:
        ax2.scatter(trades_df['entry_deviation'], trades_df['exit_deviation'], 
                   c=trades_df['profit_pct'], cmap='RdYlGn', alpha=0.7, s=50)
        
        # Add diagonal line (no change)
        min_dev = min(trades_df['entry_deviation'].min(), trades_df['exit_deviation'].min())
        max_dev = max(trades_df['entry_deviation'].max(), trades_df['exit_deviation'].max())
        ax2.plot([min_dev, max_dev], [min_dev, max_dev], 'k--', alpha=0.5, label='No Change')
        
        # Add target line (exit at 0%)
        ax2.axhline(y=0, color='red', linestyle='-', alpha=0.7, label='Target Exit (0%)')
        
        ax2.set_title('Entry vs Exit Deviation (Color = Return %)', fontweight='bold')
        ax2.set_xlabel('Entry Deviation (%)')
        ax2.set_ylabel('Exit Deviation (%)')
        ax2.legend()
        ax2.grid(True, alpha=0.3)
        
        # Add colorbar
        scatter = ax2.collections[0]
        plt.colorbar(scatter, ax=ax2, label='Trade Return (%)')
    
    # Plot 3: Trade returns over time
    if trades_df is not None and len(trades_df) > 0:
        trades_sorted = trades_df.sort_values('entry_date')
        ax3.scatter(trades_sorted['entry_date'], trades_sorted['profit_pct'], 
                   alpha=0.7, s=50, c='blue')
        ax3.axhline(y=0, color='red', linestyle='--', alpha=0.5)
        
        # Add rolling average
        window_size = min(10, len(trades_sorted))
        if len(trades_sorted) >= window_size:
            rolling_avg = trades_sorted['profit_pct'].rolling(window=window_size, min_periods=1).mean()
            ax3.plot(trades_sorted['entry_date'], rolling_avg, 
                    color='red', linewidth=2, label=f'Rolling {window_size}-Trade Average')
        
        ax3.set_title('Trade Returns Over Time', fontweight='bold')
        ax3.set_xlabel('Entry Date')
        ax3.set_ylabel('Trade Return (%)')
        ax3.grid(True, alpha=0.3)
        if len(trades_sorted) >= window_size:
            ax3.legend()
    
    # Plot 4: Holding period vs Returns
    if trades_df is not None and len(trades_df) > 0:
        colors = ['green' if x > 0 else 'red' for x in trades_df['profit_pct']]
        ax4.scatter(trades_df['days_held'], trades_df['profit_pct'], 
                   c=colors, alpha=0.7, s=50)
        ax4.axhline(y=0, color='black', linestyle='--', alpha=0.5)
        
        ax4.set_title('Holding Period vs Trade Returns', fontweight='bold')
        ax4.set_xlabel('Days Held')
        ax4.set_ylabel('Trade Return (%)')
        ax4.grid(True, alpha=0.3)
    
    plt.tight_layout()
    plt.savefig(f"{output_dir}/deviation_reversion_strategy_analysis.png", dpi=300, bbox_inches='tight')
    plt.close()

def save_deviation_strategy_results(portfolio, trades_df, output_dir):
    """
    Save deviation reversion strategy results
    """
    # Save all trades
    if trades_df is not None:
        trades_file = f"{output_dir}/deviation_reversion_all_trades.csv"
        trades_df.to_csv(trades_file, index=False)
        print(f"All trades saved to: {trades_file}")
    
    # Save portfolio history
    if portfolio.portfolio_history:
        history_file = f"{output_dir}/portfolio_value_history.csv"
        history_df = pd.DataFrame(portfolio.portfolio_history)
        history_df.to_csv(history_file, index=False)
        print(f"Portfolio history saved to: {history_file}")
    
    # Save summary statistics
    final_value = portfolio.cash
    total_return = ((final_value / portfolio.initial_capital) - 1) * 100
    
    summary_stats = {
        'initial_capital': portfolio.initial_capital,
        'final_value': final_value,
        'total_profit': final_value - portfolio.initial_capital,
        'total_return_pct': total_return,
        'num_trades': len(portfolio.completed_trades),
        'num_positions_target': portfolio.num_positions
    }
    
    if portfolio.completed_trades:
        summary_stats.update({
            'avg_trade_return': np.mean([t['profit_pct'] for t in portfolio.completed_trades]),
            'win_rate': len([t for t in portfolio.completed_trades if t['profit'] > 0]) / len(portfolio.completed_trades) * 100,
            'avg_holding_days': np.mean([t['days_held'] for t in portfolio.completed_trades]),
            'avg_annualized_return': np.mean([t['annualized_return'] for t in portfolio.completed_trades])
        })
    
    summary_df = pd.DataFrame([summary_stats])
    summary_file = f"{output_dir}/strategy_summary.csv"
    summary_df.to_csv(summary_file, index=False)
    print(f"Strategy summary saved to: {summary_file}")

def main():
    print("Deviation Reversion Trading Strategy")
    print("="*50)
    print("Strategy: Enter 5 most underperforming assets, exit when deviation returns to 0%")
    print("Capital allocation: Equal weight across 5 positions, reinvest proceeds")
    
    # Create output directory
    output_dir = "/Users/tim/IWLS-OPTIONS/DEVIATION_REVERSION_STRATEGY"
    os.makedirs(output_dir, exist_ok=True)
    print(f"\nOutput directory: {output_dir}")
    
    # Load IWLS results
    all_results = load_all_iwls_results()
    if not all_results:
        return
    
    print(f"\nLoaded data for {len(all_results)} assets")
    
    # Run deviation reversion strategy
    portfolio = run_deviation_reversion_strategy(
        all_results,
        initial_capital=10000,
        num_positions=5,
        min_deviation_threshold=-10,  # Only consider assets below -10%
        rebalance_frequency_days=7    # Check for new entries weekly
    )
    
    # Analyze results
    trades_df = analyze_deviation_strategy_results(portfolio)
    
    # Create visualizations
    print(f"\nCreating visualizations...")
    create_deviation_strategy_visualizations(portfolio, trades_df, output_dir)
    
    # Save results
    save_deviation_strategy_results(portfolio, trades_df, output_dir)
    
    print(f"\n" + "="*80)
    print("DEVIATION REVERSION STRATEGY COMPLETE")
    print("="*80)
    print("Files saved:")
    print("  - deviation_reversion_all_trades.csv (all completed trades)")
    print("  - portfolio_value_history.csv (portfolio value over time)")
    print("  - strategy_summary.csv (performance summary)")
    print("  - deviation_reversion_strategy_analysis.png (4-panel visualization)")
    
    # Final summary
    final_value = portfolio.cash
    total_return = ((final_value / portfolio.initial_capital) - 1) * 100
    
    print(f"\nFINAL STRATEGY SUMMARY:")
    print(f"Initial capital: ${portfolio.initial_capital:,}")
    print(f"Final value: ${final_value:,.2f}")
    print(f"Total return: {total_return:.2f}%")
    print(f"Total profit: ${final_value - portfolio.initial_capital:,.2f}")
    print(f"Number of completed trades: {len(portfolio.completed_trades)}")
    
    if portfolio.completed_trades:
        winning_trades = len([t for t in portfolio.completed_trades if t['profit'] > 0])
        win_rate = (winning_trades / len(portfolio.completed_trades)) * 100
        avg_return = np.mean([t['profit_pct'] for t in portfolio.completed_trades])
        avg_days = np.mean([t['days_held'] for t in portfolio.completed_trades])
        
        print(f"Win rate: {win_rate:.1f}% ({winning_trades}/{len(portfolio.completed_trades)})")
        print(f"Average return per trade: {avg_return:.2f}%")
        print(f"Average holding period: {avg_days:.1f} days")
        
        # Show best performing asset
        if trades_df is not None:
            asset_performance = trades_df.groupby('asset')['profit_pct'].agg(['count', 'mean']).round(2)
            asset_performance = asset_performance[asset_performance['count'] >= 2]  # At least 2 trades
            if len(asset_performance) > 0:
                best_asset = asset_performance['mean'].idxmax()
                best_return = asset_performance.loc[best_asset, 'mean']
                trade_count = asset_performance.loc[best_asset, 'count']
                print(f"\nBest performing asset: {best_asset} ({best_return:.1f}% avg return, {trade_count} trades)")

if __name__ == "__main__":
    main()    