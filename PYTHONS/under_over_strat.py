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

def execute_individual_trades_strategy_with_max_hold(all_results, entry_threshold=-20, exit_threshold=20, max_hold_days=365, trade_amount=10000):
    """
    Execute individual trades strategy with STRICT 365-day maximum hold period
    Enter at -20%, exit at +20% OR exactly 365 days, whichever comes first
    """
    print(f"\nExecuting Individual Trades Strategy with STRICT Max Hold:")
    print(f"Entry threshold: {entry_threshold}% below trend")
    print(f"Exit threshold: {exit_threshold}% above trend")
    print(f"MAXIMUM hold period: {max_hold_days} days (STRICTLY ENFORCED)")
    print(f"Trade amount: ${trade_amount:,} per trade")
    
    all_trades = []
    active_trades = {}  # Track active trades by asset
    
    # Create combined timeline of all price movements
    all_data_points = []
    
    for asset_name, df in all_results.items():
        for _, row in df.iterrows():
            all_data_points.append({
                'date': row['date'],
                'asset': asset_name,
                'price': row['price'],
                'price_deviation': row['price_deviation']
            })
    
    # Sort all data points by date
    all_data_points.sort(key=lambda x: x['date'])
    
    print(f"\nProcessing {len(all_data_points):,} data points across all assets...")
    
    processed_count = 0
    
    for data_point in all_data_points:
        processed_count += 1
        
        if processed_count % 10000 == 0:
            print(f"  Processed {processed_count:,}/{len(all_data_points):,} data points...")
        
        current_date = data_point['date']
        asset = data_point['asset']
        price = data_point['price']
        deviation = data_point['price_deviation']
        
        # FIRST: Check for exit conditions on all active trades
        assets_to_exit = []
        for active_asset, active_trade in active_trades.items():
            entry_date = active_trade['entry_date']
            days_held = (current_date - entry_date).days
            
            # Check exit conditions
            should_exit = False
            exit_reason = ""
            
            if active_asset == asset and deviation >= exit_threshold:
                should_exit = True
                exit_reason = 'Target_Reached'
            elif days_held >= max_hold_days:
                should_exit = True
                exit_reason = 'Max_Hold_Reached'
            
            if should_exit:
                assets_to_exit.append((active_asset, exit_reason))
        
        # Process all exits
        for exit_asset, exit_reason in assets_to_exit:
            active_trade = active_trades[exit_asset]
            entry_date = active_trade['entry_date']
            entry_price = active_trade['entry_price']
            shares = active_trade['shares']
            
            # Get current price for this asset
            if exit_asset == asset:
                exit_price = price
            else:
                # Find current price for other asset
                exit_price = get_current_price_for_asset(all_results, exit_asset, current_date)
                if exit_price is None:
                    continue
            
            proceeds = shares * exit_price
            profit = proceeds - trade_amount
            profit_pct = (profit / trade_amount) * 100
            days_held = (current_date - entry_date).days
            
            # Calculate annualized return
            annualized_return = calculate_annualized_return(entry_price, exit_price, entry_date, current_date)
            
            # Record completed trade
            completed_trade = {
                'asset': exit_asset,
                'entry_date': entry_date,
                'exit_date': current_date,
                'entry_price': entry_price,
                'exit_price': exit_price,
                'entry_deviation': active_trade['entry_deviation'],
                'exit_deviation': deviation if exit_asset == asset else get_current_deviation_for_asset(all_results, exit_asset, current_date),
                'shares': shares,
                'investment': trade_amount,
                'proceeds': proceeds,
                'profit': profit,
                'profit_pct': profit_pct,
                'days_held': days_held,
                'annualized_return': annualized_return,
                'exit_reason': exit_reason
            }
            
            all_trades.append(completed_trade)
            
            # Remove from active trades
            del active_trades[exit_asset]
            
            print(f"  EXIT: {exit_asset} at {current_date.strftime('%Y-%m-%d')} - "
                  f"{profit_pct:+.1f}% ({annualized_return:+.1f}% annualized) in {days_held} days ({exit_reason})")
        
        # SECOND: Check for entry conditions (only if no active trade for this asset and meets threshold)
        if asset not in active_trades and deviation <= entry_threshold:
            # Enter new trade
            shares = trade_amount / price
            
            active_trades[asset] = {
                'entry_date': current_date,
                'entry_price': price,
                'entry_deviation': deviation,
                'shares': shares
            }
            
            print(f"  ENTER: {asset} at {current_date.strftime('%Y-%m-%d')} - "
                  f"deviation {deviation:.1f}%, price ${price:.2f}")
    
    # Handle any remaining active trades at end of data
    print(f"\nHandling {len(active_trades)} remaining active trades...")
    
    for asset, active_trade in active_trades.items():
        # Get final price for this asset
        final_data = all_results[asset].iloc[-1]
        final_price = final_data['price']
        final_date = final_data['date']
        final_deviation = final_data['price_deviation']
        
        # Calculate final trade results
        entry_price = active_trade['entry_price']
        entry_date = active_trade['entry_date']
        shares = active_trade['shares']
        proceeds = shares * final_price
        profit = proceeds - trade_amount
        profit_pct = (profit / trade_amount) * 100
        days_held = (final_date - entry_date).days
        
        # Calculate annualized return for final trades
        annualized_return = calculate_annualized_return(entry_price, final_price, entry_date, final_date)
        
        # Record completed trade
        completed_trade = {
            'asset': asset,
            'entry_date': entry_date,
            'exit_date': final_date,
            'entry_price': entry_price,
            'exit_price': final_price,
            'entry_deviation': active_trade['entry_deviation'],
            'exit_deviation': final_deviation,
            'shares': shares,
            'investment': trade_amount,
            'proceeds': proceeds,
            'profit': profit,
            'profit_pct': profit_pct,
            'days_held': days_held,
            'annualized_return': annualized_return,
            'exit_reason': 'End_of_Data'
        }
        
        all_trades.append(completed_trade)
        
        print(f"  FINAL: {asset} - {profit_pct:+.1f}% ({annualized_return:+.1f}% annualized) in {days_held} days")
    
    return all_trades

def get_current_price_for_asset(all_results, asset, date):
    """
    Get the current price for an asset on or before a given date
    """
    if asset not in all_results:
        return None
    
    df = all_results[asset]
    asset_data = df[df['date'] <= date]
    
    if len(asset_data) == 0:
        return None
    
    return asset_data.iloc[-1]['price']

def get_current_deviation_for_asset(all_results, asset, date):
    """
    Get the current deviation for an asset on or before a given date
    """
    if asset not in all_results:
        return None
    
    df = all_results[asset]
    asset_data = df[df['date'] <= date]
    
    if len(asset_data) == 0:
        return None
    
    return asset_data.iloc[-1]['price_deviation']

def analyze_trade_results(all_trades):
    """
    Analyze the results of all individual trades with enhanced annualized return analysis
    """
    if not all_trades:
        print("No trades found!")
        return None
    
    trades_df = pd.DataFrame(all_trades)
    
    print("\n" + "="*80)
    print("INDIVIDUAL TRADES STRATEGY ANALYSIS (WITH MAX HOLD)")
    print("="*80)
    
    # Overall statistics
    total_trades = len(trades_df)
    winning_trades = len(trades_df[trades_df['profit'] > 0])
    losing_trades = len(trades_df[trades_df['profit'] < 0])
    breakeven_trades = len(trades_df[trades_df['profit'] == 0])
    
    win_rate = (winning_trades / total_trades) * 100
    
    total_invested = trades_df['investment'].sum()
    total_proceeds = trades_df['proceeds'].sum()
    total_profit = trades_df['profit'].sum()
    total_return_pct = (total_profit / total_invested) * 100
    
    print(f"\nOVERALL PERFORMANCE:")
    print(f"Total trades executed: {total_trades:,}")
    print(f"Winning trades: {winning_trades:,} ({win_rate:.1f}%)")
    print(f"Losing trades: {losing_trades:,}")
    print(f"Breakeven trades: {breakeven_trades:,}")
    print(f"")
    print(f"Total invested: ${total_invested:,.2f}")
    print(f"Total proceeds: ${total_proceeds:,.2f}")
    print(f"Total profit: ${total_profit:,.2f}")
    print(f"Overall return: {total_return_pct:.2f}%")
    
    # Trade statistics (both regular and annualized)
    avg_profit = trades_df['profit'].mean()
    avg_profit_pct = trades_df['profit_pct'].mean()
    median_profit_pct = trades_df['profit_pct'].median()
    
    avg_annualized = trades_df['annualized_return'].mean()
    median_annualized = trades_df['annualized_return'].median()
    
    avg_days = trades_df['days_held'].mean()
    median_days = trades_df['days_held'].median()
    max_days = trades_df['days_held'].max()
    
    print(f"\nTRADE STATISTICS:")
    print(f"Average profit per trade: ${avg_profit:.2f} ({avg_profit_pct:.2f}%)")
    print(f"Median profit per trade: {median_profit_pct:.2f}%")
    print(f"Average annualized return: {avg_annualized:.1f}%")
    print(f"Median annualized return: {median_annualized:.1f}%")
    print(f"Average holding period: {avg_days:.1f} days")
    print(f"Median holding period: {median_days:.1f} days")
    print(f"Maximum holding period: {max_days:.0f} days")
    
    # Verify max hold enforcement
    over_365_trades = trades_df[trades_df['days_held'] > 365]
    print(f"Trades held > 365 days: {len(over_365_trades)} (should be 0 unless End_of_Data)")
    
    # Best and worst trades (by annualized return)
    best_trade = trades_df.loc[trades_df['annualized_return'].idxmax()]
    worst_trade = trades_df.loc[trades_df['annualized_return'].idxmin()]
    
    print(f"\nBEST TRADE (Annualized Return):")
    print(f"  {best_trade['asset']}: {best_trade['profit_pct']:+.1f}% total "
          f"({best_trade['annualized_return']:+.1f}% annualized) "
          f"in {best_trade['days_held']} days")
    print(f"  Period: {best_trade['entry_date'].strftime('%Y-%m-%d')} to {best_trade['exit_date'].strftime('%Y-%m-%d')}")
    
    print(f"\nWORST TRADE (Annualized Return):")
    print(f"  {worst_trade['asset']}: {worst_trade['profit_pct']:+.1f}% total "
          f"({worst_trade['annualized_return']:+.1f}% annualized) "
          f"in {worst_trade['days_held']} days")
    print(f"  Period: {worst_trade['entry_date'].strftime('%Y-%m-%d')} to {worst_trade['exit_date'].strftime('%Y-%m-%d')}")
    
    # Exit reason analysis with max hold
    print(f"\nEXIT REASON ANALYSIS:")
    exit_reasons = trades_df['exit_reason'].value_counts()
    for reason, count in exit_reasons.items():
        pct = (count / total_trades) * 100
        avg_return = trades_df[trades_df['exit_reason'] == reason]['profit_pct'].mean()
        avg_annualized = trades_df[trades_df['exit_reason'] == reason]['annualized_return'].mean()
        avg_days = trades_df[trades_df['exit_reason'] == reason]['days_held'].mean()
        print(f"  {reason}: {count} trades ({pct:.1f}%)")
        print(f"    Avg total return: {avg_return:.1f}%")
        print(f"    Avg annualized return: {avg_annualized:.1f}%")
        print(f"    Avg holding period: {avg_days:.1f} days")
    
    # Asset performance
    print(f"\nTOP 10 ASSETS BY AVERAGE ANNUALIZED RETURN:")
    print("-" * 70)
    print(f"{'Asset':<6} {'Trades':<6} {'Avg Total':<10} {'Avg Annual':<11} {'Avg Days':<8}")
    print("-" * 70)
    
    asset_performance = trades_df.groupby('asset').agg({
        'profit_pct': ['count', 'mean'],
        'annualized_return': 'mean',
        'days_held': 'mean'
    }).round(2)
    
    asset_performance.columns = ['Trades', 'Avg_Total_Return', 'Avg_Annual_Return', 'Avg_Days']
    asset_performance = asset_performance[asset_performance['Trades'] >= 1]  # At least 1 trade
    top_assets = asset_performance.sort_values('Avg_Annual_Return', ascending=False).head(10)
    
    for asset, row in top_assets.iterrows():
        print(f"{asset:<6} {row['Trades']:<6.0f} {row['Avg_Total_Return']:<10.1f}% "
              f"{row['Avg_Annual_Return']:<10.1f}% {row['Avg_Days']:<8.1f}")
    
    return trades_df

def create_enhanced_trade_visualizations(trades_df, output_dir):
    """
    Create enhanced visualizations including annualized return plots
    """
    fig, ((ax1, ax2), (ax3, ax4)) = plt.subplots(2, 2, figsize=(20, 12))
    
    # Plot 1: Annualized returns over time
    trades_df_sorted = trades_df.sort_values('entry_date')
    scatter = ax1.scatter(trades_df_sorted['entry_date'], 
                         trades_df_sorted['annualized_return'], 
                         c=trades_df_sorted['days_held'], 
                         cmap='viridis', alpha=0.7, s=50)
    ax1.axhline(y=0, color='black', linestyle='--', alpha=0.5)
    ax1.set_title('Annualized Returns Over Time (Color = Days Held)', fontweight='bold')
    ax1.set_xlabel('Entry Date')
    ax1.set_ylabel('Annualized Return (%)')
    ax1.grid(True, alpha=0.3)
    plt.colorbar(scatter, ax=ax1, label='Days Held')
    
    # Plot 2: Exit reason distribution
    exit_counts = trades_df['exit_reason'].value_counts()
    colors = ['green', 'orange', 'red']
    ax2.pie(exit_counts.values, labels=exit_counts.index, autopct='%1.1f%%', 
            colors=colors[:len(exit_counts)], startangle=90)
    ax2.set_title('Distribution of Exit Reasons', fontweight='bold')
    
    # Plot 3: Holding period vs annualized returns (colored by exit reason)
    exit_reason_colors = {'Target_Reached': 'green', 'Max_Hold_Reached': 'orange', 'End_of_Data': 'red'}
    for reason in trades_df['exit_reason'].unique():
        mask = trades_df['exit_reason'] == reason
        ax3.scatter(trades_df[mask]['days_held'], trades_df[mask]['annualized_return'], 
                   label=reason, alpha=0.7, c=exit_reason_colors.get(reason, 'blue'))
    
    ax3.axhline(y=0, color='black', linestyle='--', alpha=0.5)
    ax3.axvline(x=365, color='red', linestyle='--', alpha=0.7, label='365 Day Limit')
    ax3.set_title('Holding Period vs Annualized Returns by Exit Reason', fontweight='bold')
    ax3.set_xlabel('Days Held')
    ax3.set_ylabel('Annualized Return (%)')
    ax3.legend()
    ax3.grid(True, alpha=0.3)
    
    # Plot 4: Rolling average annualized return
    trades_df_sorted['cumulative_profit'] = trades_df_sorted['profit'].cumsum()
    
    window_size = min(5, len(trades_df_sorted))
    trades_df_sorted['rolling_avg_annual'] = trades_df_sorted['annualized_return'].rolling(
        window=window_size, min_periods=1).mean()
    
    ax4.plot(trades_df_sorted['entry_date'], 
             trades_df_sorted['rolling_avg_annual'], 
             linewidth=2, color='blue', label=f'Rolling {window_size}-Trade Avg')
    ax4.axhline(y=0, color='black', linestyle='--', alpha=0.5)
    ax4.axhline(y=trades_df['annualized_return'].mean(), color='red', linestyle='-', 
                alpha=0.7, label=f'Overall Average: {trades_df["annualized_return"].mean():.1f}%')
    ax4.set_title('Rolling Average Annualized Returns', fontweight='bold')
    ax4.set_xlabel('Entry Date')
    ax4.set_ylabel('Rolling Avg Annualized Return (%)')
    ax4.legend()
    ax4.grid(True, alpha=0.3)
    
    plt.tight_layout()
    plt.savefig(f"{output_dir}/max_hold_individual_trades_analysis.png", dpi=300, bbox_inches='tight')
    plt.close()

def save_enhanced_trade_results(trades_df, output_dir):
    """
    Save enhanced trade results with max hold analysis
    """
    # Main trades file
    trades_file = f"{output_dir}/individual_trades_max_hold_365.csv"
    trades_df.to_csv(trades_file, index=False)
    print(f"\nDetailed trades (365-day max hold) saved to: {trades_file}")
    
    # Enhanced summary statistics
    summary_stats = {
        'Total_Trades': len(trades_df),
        'Winning_Trades': len(trades_df[trades_df['profit'] > 0]),
        'Win_Rate_Pct': (len(trades_df[trades_df['profit'] > 0]) / len(trades_df)) * 100,
        'Target_Reached_Count': len(trades_df[trades_df['exit_reason'] == 'Target_Reached']),
        'Max_Hold_Reached_Count': len(trades_df[trades_df['exit_reason'] == 'Max_Hold_Reached']),
        'End_of_Data_Count': len(trades_df[trades_df['exit_reason'] == 'End_of_Data']),
        'Total_Invested': trades_df['investment'].sum(),
        'Total_Proceeds': trades_df['proceeds'].sum(), 
        'Total_Profit': trades_df['profit'].sum(),
        'Overall_Return_Pct': (trades_df['profit'].sum() / trades_df['investment'].sum()) * 100,
        'Average_Return_Pct': trades_df['profit_pct'].mean(),
        'Median_Return_Pct': trades_df['profit_pct'].median(),
        'Average_Annualized_Return': trades_df['annualized_return'].mean(),
        'Median_Annualized_Return': trades_df['annualized_return'].median(),
        'Best_Annualized_Return': trades_df['annualized_return'].max(),
        'Worst_Annualized_Return': trades_df['annualized_return'].min(),
        'Average_Days_Held': trades_df['days_held'].mean(),
        'Median_Days_Held': trades_df['days_held'].median(),
        'Max_Days_Held': trades_df['days_held'].max()
    }
    
    summary_df = pd.DataFrame([summary_stats])
    summary_file = f"{output_dir}/max_hold_trades_summary.csv"
    summary_df.to_csv(summary_file, index=False)
    print(f"Max hold summary statistics saved to: {summary_file}")
    
    return trades_file, summary_file

def main():
    print("Enhanced Individual Trades Strategy: Enter at -20%, Exit at +20% or Max 365 Days")
    print("="*80)
    print("Strategy: $10,000 per trade, one active trade per asset, STRICT 365-day max hold")
    print("Enhanced with detailed annualized return calculations")
    
    # Create output directory
    output_dir = "/Users/tim/IWLS-OPTIONS/MAX_HOLD_INDIVIDUAL_TRADES"
    os.makedirs(output_dir, exist_ok=True)
    print(f"\nOutput directory: {output_dir}")
    
    # Load IWLS results
    all_results = load_all_iwls_results()
    if not all_results:
        return
    
    print(f"\nLoaded data for {len(all_results)} assets")
    
    # Execute trading strategy with strict max hold
    print(f"\nExecuting trading strategy with STRICT 365-day maximum hold...")
    all_trades = execute_individual_trades_strategy_with_max_hold(
        all_results, 
        entry_threshold=-20, 
        exit_threshold=20,
        max_hold_days=365,
        trade_amount=10000
    )
    
    if not all_trades:
        print("No trades were executed!")
        return
    
    # Analyze results
    trades_df = analyze_trade_results(all_trades)
    
    if trades_df is None:
        return
    
    # Create enhanced visualizations
    print(f"\nCreating enhanced visualizations...")
    create_enhanced_trade_visualizations(trades_df, output_dir)
    
    # Save enhanced results
    trades_file, summary_file = save_enhanced_trade_results(trades_df, output_dir)
    
    print(f"\n" + "="*80)
    print("ENHANCED INDIVIDUAL TRADES STRATEGY (365-DAY MAX HOLD) COMPLETE")
    print("="*80)
    print("Files saved:")
    print(f"  - individual_trades_max_hold_365.csv ({len(trades_df)} trades)")
    print(f"  - max_hold_trades_summary.csv (comprehensive performance summary)")
    print(f"  - max_hold_individual_trades_analysis.png (4-panel visualization)")
    
    # Enhanced quick summary with max hold verification
    if len(trades_df) > 0:
        total_profit = trades_df['profit'].sum()
        total_invested = trades_df['investment'].sum()
        overall_return = (total_profit / total_invested) * 100
        win_rate = (len(trades_df[trades_df['profit'] > 0]) / len(trades_df)) * 100
        avg_annualized = trades_df['annualized_return'].mean()
        median_annualized = trades_df['annualized_return'].median()
        best_annualized = trades_df['annualized_return'].max()
        max_days_held = trades_df['days_held'].max()
        
        print(f"\nENHANCED QUICK SUMMARY (with STRICT 365-Day Max Hold):")
        print(f"Total trades: {len(trades_df):,}")
        print(f"Win rate: {win_rate:.1f}%")
        print(f"Total return: {overall_return:.2f}%")
        print(f"Total profit: ${total_profit:,.2f}")
        print(f"Average annualized return: {avg_annualized:.1f}%")
        print(f"Median annualized return: {median_annualized:.1f}%")
        print(f"Best annualized return: {best_annualized:.1f}%")
        print(f"Maximum days held: {max_days_held:.0f} (should be ≤365 unless End_of_Data)")
        
        # Show exit reason breakdown
        exit_breakdown = trades_df['exit_reason'].value_counts()
        print(f"\nEXIT REASON BREAKDOWN:")
        for reason, count in exit_breakdown.items():
            pct = (count / len(trades_df)) * 100
            avg_return = trades_df[trades_df['exit_reason'] == reason]['annualized_return'].mean()
            print(f"  {reason}: {count} trades ({pct:.1f}%) - Avg Annual Return: {avg_return:.1f}%")
        
        # Verify max hold enforcement
        over_365_non_end = trades_df[(trades_df['days_held'] > 365) & (trades_df['exit_reason'] != 'End_of_Data')]
        if len(over_365_non_end) > 0:
            print(f"\n⚠️  WARNING: {len(over_365_non_end)} trades exceeded 365 days without 'End_of_Data' reason!")
        else:
            print(f"\n✅ MAX HOLD VERIFICATION: All trades properly enforced 365-day limit")
        
        # Show top 5 trades by annualized return
        print(f"\nTOP 5 TRADES BY ANNUALIZED RETURN:")
        print("-" * 95)
        top_5 = trades_df.nlargest(5, 'annualized_return')
        for _, trade in top_5.iterrows():
            print(f"{trade['asset']:<6}: {trade['profit_pct']:>6.1f}% total, "
                  f"{trade['annualized_return']:>8.1f}% annual, {trade['days_held']:>3.0f} days, "
                  f"{trade['entry_date'].strftime('%Y-%m-%d')} to {trade['exit_date'].strftime('%Y-%m-%d')} ({trade['exit_reason']})")

if __name__ == "__main__":
    main()