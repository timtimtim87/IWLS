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

def execute_individual_trades_strategy(all_results, entry_threshold=-20, exit_threshold=20, trade_amount=10000):
    """
    Execute individual trades strategy: enter at -20%, exit at +20%
    Only one active trade per asset at a time
    """
    print(f"\nExecuting Individual Trades Strategy:")
    print(f"Entry threshold: {entry_threshold}% below trend")
    print(f"Exit threshold: {exit_threshold}% above trend") 
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
        
        date = data_point['date']
        asset = data_point['asset']
        price = data_point['price']
        deviation = data_point['price_deviation']
        
        # Check for exit conditions first
        if asset in active_trades:
            active_trade = active_trades[asset]
            
            # Check if we should exit (reached +20% or end of data)
            if deviation >= exit_threshold:
                # Calculate trade results
                entry_price = active_trade['entry_price']
                shares = active_trade['shares']
                proceeds = shares * price
                profit = proceeds - trade_amount
                profit_pct = (profit / trade_amount) * 100
                days_held = (date - active_trade['entry_date']).days
                
                # Record completed trade
                completed_trade = {
                    'asset': asset,
                    'entry_date': active_trade['entry_date'],
                    'exit_date': date,
                    'entry_price': entry_price,
                    'exit_price': price,
                    'entry_deviation': active_trade['entry_deviation'],
                    'exit_deviation': deviation,
                    'shares': shares,
                    'investment': trade_amount,
                    'proceeds': proceeds,
                    'profit': profit,
                    'profit_pct': profit_pct,
                    'days_held': days_held,
                    'exit_reason': 'Target_Reached'
                }
                
                all_trades.append(completed_trade)
                
                # Remove from active trades
                del active_trades[asset]
                
                print(f"  EXIT: {asset} at {date.strftime('%Y-%m-%d')} - "
                      f"{profit_pct:+.1f}% in {days_held} days")
        
        # Check for entry conditions (only if no active trade for this asset)
        if asset not in active_trades and deviation <= entry_threshold:
            # Enter new trade
            shares = trade_amount / price
            
            active_trades[asset] = {
                'entry_date': date,
                'entry_price': price,
                'entry_deviation': deviation,
                'shares': shares
            }
            
            print(f"  ENTER: {asset} at {date.strftime('%Y-%m-%d')} - "
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
        shares = active_trade['shares']
        proceeds = shares * final_price
        profit = proceeds - trade_amount
        profit_pct = (profit / trade_amount) * 100
        days_held = (final_date - active_trade['entry_date']).days
        
        # Record completed trade
        completed_trade = {
            'asset': asset,
            'entry_date': active_trade['entry_date'],
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
            'exit_reason': 'End_of_Data'
        }
        
        all_trades.append(completed_trade)
        
        print(f"  FINAL: {asset} - {profit_pct:+.1f}% in {days_held} days (end of data)")
    
    return all_trades

def analyze_trade_results(all_trades):
    """
    Analyze the results of all individual trades
    """
    if not all_trades:
        print("No trades found!")
        return
    
    trades_df = pd.DataFrame(all_trades)
    
    print("\n" + "="*80)
    print("INDIVIDUAL TRADES STRATEGY ANALYSIS")
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
    
    # Trade statistics
    avg_profit = trades_df['profit'].mean()
    avg_profit_pct = trades_df['profit_pct'].mean()
    median_profit_pct = trades_df['profit_pct'].median()
    
    avg_days = trades_df['days_held'].mean()
    median_days = trades_df['days_held'].median()
    
    print(f"\nTRADE STATISTICS:")
    print(f"Average profit per trade: ${avg_profit:.2f} ({avg_profit_pct:.2f}%)")
    print(f"Median profit per trade: {median_profit_pct:.2f}%")
    print(f"Average holding period: {avg_days:.1f} days")
    print(f"Median holding period: {median_days:.1f} days")
    
    # Best and worst trades
    best_trade = trades_df.loc[trades_df['profit_pct'].idxmax()]
    worst_trade = trades_df.loc[trades_df['profit_pct'].idxmin()]
    
    print(f"\nBEST TRADE:")
    print(f"  {best_trade['asset']}: {best_trade['profit_pct']:+.1f}% "
          f"({best_trade['days_held']} days) "
          f"{best_trade['entry_date'].strftime('%Y-%m-%d')} to {best_trade['exit_date'].strftime('%Y-%m-%d')}")
    
    print(f"\nWORST TRADE:")
    print(f"  {worst_trade['asset']}: {worst_trade['profit_pct']:+.1f}% "
          f"({worst_trade['days_held']} days) "
          f"{worst_trade['entry_date'].strftime('%Y-%m-%d')} to {worst_trade['exit_date'].strftime('%Y-%m-%d')}")
    
    # Exit reason analysis
    print(f"\nEXIT REASON ANALYSIS:")
    exit_reasons = trades_df['exit_reason'].value_counts()
    for reason, count in exit_reasons.items():
        pct = (count / total_trades) * 100
        avg_return = trades_df[trades_df['exit_reason'] == reason]['profit_pct'].mean()
        print(f"  {reason}: {count} trades ({pct:.1f}%) - avg return: {avg_return:.1f}%")
    
    # Asset performance
    print(f"\nTOP 10 ASSETS BY AVERAGE RETURN:")
    asset_performance = trades_df.groupby('asset').agg({
        'profit_pct': ['count', 'mean', 'sum'],
        'days_held': 'mean'
    }).round(2)
    
    asset_performance.columns = ['Trades', 'Avg_Return_Pct', 'Total_Return_Pct', 'Avg_Days']
    asset_performance = asset_performance[asset_performance['Trades'] >= 2]  # At least 2 trades
    top_assets = asset_performance.sort_values('Avg_Return_Pct', ascending=False).head(10)
    
    print(top_assets.to_string())
    
    # Distribution analysis
    print(f"\nRETURN DISTRIBUTION:")
    bins = [-100, -20, -10, 0, 10, 20, 30, 40, 50, 100]
    labels = ['<-20%', '-20 to -10%', '-10 to 0%', '0 to 10%', '10 to 20%', 
              '20 to 30%', '30 to 40%', '40 to 50%', '>50%']
    
    distribution = pd.cut(trades_df['profit_pct'], bins=bins, labels=labels).value_counts().sort_index()
    
    for label, count in distribution.items():
        pct = (count / total_trades) * 100
        print(f"  {label}: {count} trades ({pct:.1f}%)")
    
    return trades_df

def create_trade_visualizations(trades_df, output_dir):
    """
    Create visualizations of trade results
    """
    fig, ((ax1, ax2), (ax3, ax4)) = plt.subplots(2, 2, figsize=(20, 12))
    
    # Plot 1: Trade returns over time
    trades_df_sorted = trades_df.sort_values('entry_date')
    ax1.scatter(trades_df_sorted['entry_date'], trades_df_sorted['profit_pct'], 
               c=trades_df_sorted['profit_pct'], cmap='RdYlGn', alpha=0.7)
    ax1.axhline(y=0, color='black', linestyle='--', alpha=0.5)
    ax1.set_title('Trade Returns Over Time', fontweight='bold')
    ax1.set_xlabel('Entry Date')
    ax1.set_ylabel('Return (%)')
    ax1.grid(True, alpha=0.3)
    
    # Plot 2: Return distribution
    ax2.hist(trades_df['profit_pct'], bins=50, alpha=0.7, color='blue', edgecolor='black')
    ax2.axvline(trades_df['profit_pct'].mean(), color='red', linestyle='--', 
                label=f'Mean: {trades_df["profit_pct"].mean():.1f}%')
    ax2.axvline(0, color='black', linestyle='-', alpha=0.5)
    ax2.set_title('Distribution of Trade Returns', fontweight='bold')
    ax2.set_xlabel('Return (%)')
    ax2.set_ylabel('Number of Trades')
    ax2.legend()
    ax2.grid(True, alpha=0.3)
    
    # Plot 3: Holding period vs returns
    colors = ['red' if x < 0 else 'green' for x in trades_df['profit_pct']]
    ax3.scatter(trades_df['days_held'], trades_df['profit_pct'], c=colors, alpha=0.6)
    ax3.axhline(y=0, color='black', linestyle='--', alpha=0.5)
    ax3.set_title('Holding Period vs Returns', fontweight='bold')
    ax3.set_xlabel('Days Held')
    ax3.set_ylabel('Return (%)')
    ax3.grid(True, alpha=0.3)
    
    # Plot 4: Cumulative returns
    trades_df_sorted['cumulative_profit'] = trades_df_sorted['profit'].cumsum()
    ax4.plot(trades_df_sorted['entry_date'], trades_df_sorted['cumulative_profit'], 
             linewidth=2, color='blue')
    ax4.axhline(y=0, color='black', linestyle='--', alpha=0.5)
    ax4.set_title('Cumulative Profit Over Time', fontweight='bold')
    ax4.set_xlabel('Entry Date')
    ax4.set_ylabel('Cumulative Profit ($)')
    ax4.grid(True, alpha=0.3)
    ax4.yaxis.set_major_formatter(plt.FuncFormatter(lambda x, p: f'${x:,.0f}'))
    
    plt.tight_layout()
    plt.savefig(f"{output_dir}/individual_trades_analysis.png", dpi=300, bbox_inches='tight')
    plt.close()

def save_trade_results(trades_df, output_dir):
    """
    Save detailed trade results to CSV
    """
    # Main trades file
    trades_file = f"{output_dir}/individual_trades_detailed.csv"
    trades_df.to_csv(trades_file, index=False)
    print(f"\nDetailed trades saved to: {trades_file}")
    
    # Summary statistics
    summary_stats = {
        'Total_Trades': len(trades_df),
        'Winning_Trades': len(trades_df[trades_df['profit'] > 0]),
        'Win_Rate_Pct': (len(trades_df[trades_df['profit'] > 0]) / len(trades_df)) * 100,
        'Total_Invested': trades_df['investment'].sum(),
        'Total_Proceeds': trades_df['proceeds'].sum(), 
        'Total_Profit': trades_df['profit'].sum(),
        'Overall_Return_Pct': (trades_df['profit'].sum() / trades_df['investment'].sum()) * 100,
        'Average_Return_Pct': trades_df['profit_pct'].mean(),
        'Median_Return_Pct': trades_df['profit_pct'].median(),
        'Average_Days_Held': trades_df['days_held'].mean(),
        'Median_Days_Held': trades_df['days_held'].median()
    }
    
    summary_df = pd.DataFrame([summary_stats])
    summary_file = f"{output_dir}/trades_summary_statistics.csv"
    summary_df.to_csv(summary_file, index=False)
    print(f"Summary statistics saved to: {summary_file}")
    
    return trades_file, summary_file

def main():
    print("Individual Trades Strategy: Enter at -20%, Exit at +20%")
    print("="*60)
    print("Strategy: $10,000 per trade, one active trade per asset")
    
    # Create output directory
    output_dir = "/Users/tim/IWLS-OPTIONS/INDIVIDUAL_TRADES_STRATEGY"
    os.makedirs(output_dir, exist_ok=True)
    print(f"\nOutput directory: {output_dir}")
    
    # Load IWLS results
    all_results = load_all_iwls_results()
    if not all_results:
        return
    
    print(f"\nLoaded data for {len(all_results)} assets")
    
    # Execute trading strategy
    print(f"\nExecuting individual trades strategy...")
    all_trades = execute_individual_trades_strategy(
        all_results, 
        entry_threshold=-20, 
        exit_threshold=20, 
        trade_amount=10000
    )
    
    if not all_trades:
        print("No trades were executed!")
        return
    
    # Analyze results
    trades_df = analyze_trade_results(all_trades)
    
    # Create visualizations
    print(f"\nCreating visualizations...")
    create_trade_visualizations(trades_df, output_dir)
    
    # Save results
    trades_file, summary_file = save_trade_results(trades_df, output_dir)
    
    print(f"\n" + "="*80)
    print("INDIVIDUAL TRADES STRATEGY COMPLETE")
    print("="*80)
    print("Files saved:")
    print(f"  - individual_trades_detailed.csv ({len(trades_df)} trades)")
    print(f"  - trades_summary_statistics.csv (performance summary)")
    print(f"  - individual_trades_analysis.png (4-panel visualization)")
    
    # Quick summary
    if len(trades_df) > 0:
        total_profit = trades_df['profit'].sum()
        total_invested = trades_df['investment'].sum()
        overall_return = (total_profit / total_invested) * 100
        win_rate = (len(trades_df[trades_df['profit'] > 0]) / len(trades_df)) * 100
        
        print(f"\nQUICK SUMMARY:")
        print(f"Total trades: {len(trades_df):,}")
        print(f"Win rate: {win_rate:.1f}%")
        print(f"Total return: {overall_return:.2f}%")
        print(f"Total profit: ${total_profit:,.2f}")

if __name__ == "__main__":
    main()