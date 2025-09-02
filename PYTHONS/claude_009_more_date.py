import pandas as pd
import numpy as np
import matplotlib.pyplot as plt
from datetime import datetime, timedelta
import warnings
import os
import glob
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
                  not d.startswith('DAILY_TRACKING') and not d.startswith('MULTI')]
    
    for asset_name in asset_dirs:
        asset_dir = os.path.join(v2_dir, asset_name)
        iwls_file = os.path.join(asset_dir, f"{asset_name}_iwls_results.csv")
        
        if os.path.exists(iwls_file):
            try:
                df = pd.read_csv(iwls_file)
                df['date'] = pd.to_datetime(df['date'])
                df = df.dropna(subset=['price_deviation', 'price']).sort_values('date').reset_index(drop=True)
                
                if len(df) > 500:
                    all_data[asset_name] = df
                    
            except Exception as e:
                print(f"  Warning: Error loading {asset_name}: {e}")
    
    print(f"Successfully loaded {len(all_data)} assets")
    return all_data

def get_available_assets_on_date(all_data, target_date, lookback_days=5):
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
                'absolute_deviation': abs(closest_row['price_deviation'])
            })
    
    return pd.DataFrame(available_assets)

def select_underperforming_stocks(available_df, min_deviation=-15, num_stocks=10):
    """
    Select underperforming stocks (negative deviation below min_deviation)
    """
    # Filter for underperforming stocks below threshold
    underperforming = available_df[
        (available_df['price_deviation'] < min_deviation)
    ].copy()
    
    if len(underperforming) == 0:
        print(f"    No stocks found with deviation below {min_deviation}%")
        # Fallback: take most underperforming stocks available
        underperforming = available_df[available_df['price_deviation'] < 0].copy()
    
    if len(underperforming) == 0:
        print(f"    No underperforming stocks found at all")
        return pd.DataFrame()
    
    # Sort by most negative deviation and take top stocks
    sorted_stocks = underperforming.sort_values('price_deviation', ascending=True)
    selected_stocks = sorted_stocks.head(num_stocks)
    
    print(f"    Selected {len(selected_stocks)} underperforming stocks")
    for _, stock in selected_stocks.iterrows():
        print(f"      {stock['asset']}: {stock['price_deviation']:.2f}% deviation, ${stock['price']:.2f}")
    
    return selected_stocks

def find_suitable_options(selected_stocks, entry_date, options_base_dir, min_dte=300, max_dte=600):
    """
    Find suitable call options for selected stocks from actual options data
    """
    print(f"\nFinding suitable call options for {len(selected_stocks)} stocks...")
    print(f"Entry date: {entry_date.strftime('%Y-%m-%d')}")
    print(f"Looking for 5-15% OTM calls with {min_dte}-{max_dte} DTE")
    
    suitable_options = []
    
    for _, stock in selected_stocks.iterrows():
        asset = stock['asset']
        stock_price = stock['price']
        
        # Look for options directories for this asset
        asset_options_dir = os.path.join(options_base_dir, asset)
        
        if not os.path.exists(asset_options_dir):
            print(f"    No options data found for {asset}")
            continue
        
        print(f"\n  Analyzing {asset} (price: ${stock_price:.2f})...")
        
        # Find all expiration directories
        exp_dirs = [d for d in os.listdir(asset_options_dir) if os.path.isdir(os.path.join(asset_options_dir, d))]
        
        best_option = None
        
        for exp_dir in exp_dirs:
            try:
                # Parse expiration date from directory name (format: YYYYMMDD)
                exp_date = datetime.strptime(exp_dir, '%Y%m%d')
                dte = (exp_date - entry_date).days
                
                # Check if DTE is in range
                if not (min_dte <= dte <= max_dte):
                    continue
                
                # Look for call options files in this expiration directory
                exp_path = os.path.join(asset_options_dir, exp_dir)
                option_files = glob.glob(os.path.join(exp_path, f"{asset}_*_C_*.csv"))
                
                for option_file in option_files:
                    # Extract strike price from filename (format: ASSET_DATE_C_STRIKE.csv)
                    filename = os.path.basename(option_file)
                    try:
                        strike_str = filename.split('_C_')[1].replace('.csv', '')
                        strike_price = float(strike_str)
                    except:
                        continue
                    
                    # Calculate moneyness (how far OTM)
                    moneyness = (strike_price - stock_price) / stock_price * 100
                    
                    # Check if it's 5-15% OTM
                    if 5 <= moneyness <= 15:
                        # Load the options data to check if we have data around entry date
                        try:
                            option_df = pd.read_csv(option_file)
                            option_df['date'] = pd.to_datetime(option_df['date'])
                            
                            # Find data around entry date
                            entry_mask = (option_df['date'] >= entry_date - timedelta(days=5)) & \
                                       (option_df['date'] <= entry_date + timedelta(days=5))
                            entry_data = option_df[entry_mask]
                            
                            if len(entry_data) > 0:
                                # Get closest data to entry date
                                closest_idx = (entry_data['date'] - entry_date).abs().idxmin()
                                entry_row = entry_data.loc[closest_idx]
                                
                                option_info = {
                                    'asset': asset,
                                    'stock_price': stock_price,
                                    'price_deviation': stock['price_deviation'],
                                    'strike_price': strike_price,
                                    'moneyness': moneyness,
                                    'dte': dte,
                                    'expiration_date': exp_date,
                                    'entry_date': entry_row['date'],
                                    'entry_option_price': entry_row['close'],
                                    'option_file': option_file,
                                    'options_symbol': entry_row['options_symbol']
                                }
                                
                                # Keep the option closest to 10% OTM
                                if best_option is None or abs(moneyness - 10) < abs(best_option['moneyness'] - 10):
                                    best_option = option_info
                        
                        except Exception as e:
                            print(f"      Error loading {option_file}: {e}")
                            continue
            
            except Exception as e:
                continue
        
        if best_option:
            suitable_options.append(best_option)
            print(f"    Selected: Strike ${best_option['strike_price']:.0f} "
                  f"({best_option['moneyness']:.1f}% OTM), "
                  f"{best_option['dte']} DTE, "
                  f"Entry Price: ${best_option['entry_option_price']:.2f}")
        else:
            print(f"    No suitable options found for {asset}")
    
    return pd.DataFrame(suitable_options)

def track_option_basket_daily(option_contracts, tracking_days=300):
    """
    Track the daily value of the option basket AND individual options for 300 days
    """
    print(f"\nTracking option basket and individual options for {tracking_days} days...")
    
    daily_basket_values = []
    individual_tracking = {contract['asset']: [] for _, contract in option_contracts.iterrows()}
    entry_date = option_contracts.iloc[0]['entry_date']
    total_initial_value = option_contracts['entry_option_price'].sum()
    
    print(f"Initial basket value: ${total_initial_value:.2f}")
    print(f"Number of contracts: {len(option_contracts)}")
    
    for day in range(tracking_days + 1):  # Include day 0
        current_date = entry_date + timedelta(days=day)
        daily_total = 0
        valid_prices = 0
        individual_values = {}
        
        for _, contract in option_contracts.iterrows():
            asset = contract['asset']
            option_file = contract['option_file']
            initial_price = contract['entry_option_price']
            
            current_price = 0  # Default to 0 if no data
            
            try:
                # Load the option data
                option_df = pd.read_csv(option_file)
                option_df['date'] = pd.to_datetime(option_df['date'])
                
                # Find price data around current date
                price_mask = (option_df['date'] >= current_date - timedelta(days=5)) & \
                           (option_df['date'] <= current_date + timedelta(days=5))
                price_data = option_df[price_mask]
                
                if len(price_data) > 0:
                    # Get closest price to current date
                    closest_idx = (price_data['date'] - current_date).abs().idxmin()
                    current_price = price_data.loc[closest_idx, 'close']
                    valid_prices += 1
                else:
                    # No data available, use last known price or 0
                    past_data = option_df[option_df['date'] <= current_date]
                    if len(past_data) > 0:
                        current_price = past_data.iloc[-1]['close']
                        valid_prices += 1
                    # If no past data either, current_price remains 0
            
            except Exception as e:
                # If we can't load data, current_price remains 0
                pass
            
            daily_total += current_price
            individual_values[asset] = current_price
            
            # Track individual option performance
            individual_tracking[asset].append({
                'day': day,
                'date': current_date,
                'price': current_price,
                'initial_price': initial_price,
                'return_pct': ((current_price / initial_price) - 1) * 100 if initial_price > 0 else 0
            })
        
        # Track basket performance
        daily_basket_values.append({
            'day': day,
            'date': current_date,
            'basket_value': daily_total,
            'valid_contracts': valid_prices,
            'return_pct': ((daily_total / total_initial_value) - 1) * 100 if total_initial_value > 0 else 0,
            **{f'{asset}_price': individual_values.get(asset, 0) for asset in individual_values}
        })
        
        if day % 30 == 0:  # Print progress every 30 days
            print(f"  Day {day}: ${daily_total:.2f} ({((daily_total / total_initial_value) - 1) * 100:+.1f}%)")
    
    # Convert individual tracking to DataFrames
    individual_dfs = {}
    for asset, tracking_data in individual_tracking.items():
        individual_dfs[asset] = pd.DataFrame(tracking_data)
    
    return pd.DataFrame(daily_basket_values), individual_dfs

def calculate_individual_performance_stats(individual_dfs, option_contracts):
    """
    Calculate performance statistics for each individual option
    """
    print("\nCalculating individual option performance statistics...")
    
    performance_stats = []
    
    for asset, df in individual_dfs.items():
        if len(df) == 0:
            continue
            
        # Get contract details
        contract_info = option_contracts[option_contracts['asset'] == asset].iloc[0]
        
        # Calculate performance metrics
        initial_price = df.iloc[0]['price']
        final_price = df.iloc[-1]['price']
        max_price = df['price'].max()
        min_price = df['price'].min()
        
        total_return = ((final_price / initial_price) - 1) * 100 if initial_price > 0 else 0
        max_gain = ((max_price / initial_price) - 1) * 100 if initial_price > 0 else 0
        max_loss = ((min_price / initial_price) - 1) * 100 if initial_price > 0 else 0
        
        # Find day of max gain
        max_day = df.loc[df['price'].idxmax(), 'day'] if max_price > 0 else 0
        
        # Calculate some additional metrics
        positive_days = len(df[df['return_pct'] > 0])
        total_days = len(df)
        positive_rate = (positive_days / total_days) * 100 if total_days > 0 else 0
        
        performance_stats.append({
            'asset': asset,
            'strike_price': contract_info['strike_price'],
            'moneyness': contract_info['moneyness'],
            'dte': contract_info['dte'],
            'price_deviation': contract_info['price_deviation'],
            'initial_price': initial_price,
            'final_price': final_price,
            'max_price': max_price,
            'min_price': min_price,
            'total_return': total_return,
            'max_gain': max_gain,
            'max_loss': max_loss,
            'day_of_max_gain': max_day,
            'positive_days': positive_days,
            'positive_rate': positive_rate
        })
    
    return pd.DataFrame(performance_stats)

def create_comprehensive_plots(daily_values, individual_dfs, performance_stats, option_contracts, output_dir):
    """
    Create comprehensive plots including individual option performance
    """
    print("\nCreating comprehensive performance plots...")
    
    # Figure 1: Original basket plots
    fig, (ax1, ax2) = plt.subplots(2, 1, figsize=(12, 10))
    
    # Plot 1: Absolute basket value
    ax1.plot(daily_values['day'], daily_values['basket_value'], linewidth=2, color='blue')
    ax1.axhline(y=daily_values.iloc[0]['basket_value'], color='red', linestyle='--', 
                label=f'Initial Value: ${daily_values.iloc[0]["basket_value"]:.2f}')
    ax1.set_xlabel('Days')
    ax1.set_ylabel('Basket Value ($)')
    ax1.set_title('Options Basket Value Over Time', fontweight='bold')
    ax1.grid(True, alpha=0.3)
    ax1.legend()
    
    # Plot 2: Return percentage
    ax2.plot(daily_values['day'], daily_values['return_pct'], linewidth=2, color='green')
    ax2.axhline(y=0, color='red', linestyle='--', label='Break Even')
    ax2.set_xlabel('Days')
    ax2.set_ylabel('Return (%)')
    ax2.set_title('Options Basket Return Percentage', fontweight='bold')
    ax2.grid(True, alpha=0.3)
    ax2.legend()
    
    plt.tight_layout()
    plt.savefig(f"{output_dir}/options_basket_performance.png", dpi=300, bbox_inches='tight')
    plt.close()
    
    # Figure 2: Individual option returns over time
    n_options = len(individual_dfs)
    if n_options > 0:
        fig, axes = plt.subplots(2, 1, figsize=(14, 12))
        
        # Plot individual option values
        for asset, df in individual_dfs.items():
            axes[0].plot(df['day'], df['price'], label=asset, linewidth=1.5, alpha=0.8)
        
        axes[0].set_xlabel('Days')
        axes[0].set_ylabel('Option Price ($)')
        axes[0].set_title('Individual Option Prices Over Time', fontweight='bold')
        axes[0].grid(True, alpha=0.3)
        axes[0].legend(bbox_to_anchor=(1.05, 1), loc='upper left')
        
        # Plot individual option returns
        for asset, df in individual_dfs.items():
            axes[1].plot(df['day'], df['return_pct'], label=asset, linewidth=1.5, alpha=0.8)
        
        axes[1].axhline(y=0, color='red', linestyle='--', label='Break Even', alpha=0.5)
        axes[1].set_xlabel('Days')
        axes[1].set_ylabel('Return (%)')
        axes[1].set_title('Individual Option Returns Over Time', fontweight='bold')
        axes[1].grid(True, alpha=0.3)
        axes[1].legend(bbox_to_anchor=(1.05, 1), loc='upper left')
        
        plt.tight_layout()
        plt.savefig(f"{output_dir}/individual_option_performance.png", dpi=300, bbox_inches='tight')
        plt.close()
    
    # Figure 3: Performance comparison bar charts
    if len(performance_stats) > 0:
        fig, ((ax1, ax2), (ax3, ax4)) = plt.subplots(2, 2, figsize=(16, 12))
        
        assets = performance_stats['asset'].tolist()
        
        # Total returns
        colors = ['green' if x >= 0 else 'red' for x in performance_stats['total_return']]
        bars1 = ax1.bar(assets, performance_stats['total_return'], color=colors, alpha=0.7)
        ax1.set_ylabel('Total Return (%)')
        ax1.set_title('Final Returns by Asset', fontweight='bold')
        ax1.tick_params(axis='x', rotation=45)
        ax1.grid(True, alpha=0.3)
        ax1.axhline(y=0, color='black', linestyle='-', alpha=0.5)
        
        # Add value labels on bars
        for bar, value in zip(bars1, performance_stats['total_return']):
            height = bar.get_height()
            ax1.text(bar.get_x() + bar.get_width()/2., height + (abs(height)*0.05 if height >= 0 else -abs(height)*0.05),
                    f'{value:.1f}%', ha='center', va='bottom' if height >= 0 else 'top', fontweight='bold')
        
        # Max gains
        bars2 = ax2.bar(assets, performance_stats['max_gain'], color='lightgreen', alpha=0.7)
        ax2.set_ylabel('Max Gain (%)')
        ax2.set_title('Maximum Gains by Asset', fontweight='bold')
        ax2.tick_params(axis='x', rotation=45)
        ax2.grid(True, alpha=0.3)
        
        # Add value labels
        for bar, value in zip(bars2, performance_stats['max_gain']):
            ax2.text(bar.get_x() + bar.get_width()/2., bar.get_height() + bar.get_height()*0.02,
                    f'{value:.1f}%', ha='center', va='bottom', fontweight='bold')
        
        # Days to max gain
        bars3 = ax3.bar(assets, performance_stats['day_of_max_gain'], color='orange', alpha=0.7)
        ax3.set_ylabel('Days to Max Gain')
        ax3.set_title('Days to Maximum Gain by Asset', fontweight='bold')
        ax3.tick_params(axis='x', rotation=45)
        ax3.grid(True, alpha=0.3)
        
        # Moneyness vs Performance scatter
        ax4.scatter(performance_stats['moneyness'], performance_stats['total_return'], 
                   s=100, alpha=0.7, c=performance_stats['total_return'], 
                   cmap='RdYlGn', vmin=-100, vmax=100)
        ax4.set_xlabel('Moneyness (% OTM)')
        ax4.set_ylabel('Total Return (%)')
        ax4.set_title('Moneyness vs Total Return', fontweight='bold')
        ax4.grid(True, alpha=0.3)
        ax4.axhline(y=0, color='red', linestyle='--', alpha=0.5)
        
        # Add asset labels to scatter plot
        for _, row in performance_stats.iterrows():
            ax4.annotate(row['asset'], (row['moneyness'], row['total_return']),
                        xytext=(5, 5), textcoords='offset points', fontsize=8)
        
        plt.tight_layout()
        plt.savefig(f"{output_dir}/performance_comparison.png", dpi=300, bbox_inches='tight')
        plt.close()
    
    print("  ✅ All plots created")

def save_comprehensive_results(option_contracts, daily_values, individual_dfs, performance_stats, output_dir):
    """
    Save all results including individual tracking data
    """
    print("\nSaving comprehensive results...")
    
    # Save option contracts details
    option_contracts.to_csv(f"{output_dir}/selected_option_contracts.csv", index=False)
    print(f"  ✅ Option contracts: selected_option_contracts.csv")
    
    # Save daily tracking data
    daily_values.to_csv(f"{output_dir}/daily_basket_values.csv", index=False)
    print(f"  ✅ Daily basket values: daily_basket_values.csv")
    
    # Save individual option tracking
    for asset, df in individual_dfs.items():
        df.to_csv(f"{output_dir}/individual_{asset}_daily.csv", index=False)
    print(f"  ✅ Individual daily tracking: individual_[ASSET]_daily.csv")
    
    # Save performance statistics
    performance_stats.to_csv(f"{output_dir}/individual_performance_stats.csv", index=False)
    print(f"  ✅ Performance statistics: individual_performance_stats.csv")

def main():
    print("OPTIONS STRATEGY BASED ON IWLS DEVIATION SIGNALS")
    print("=" * 60)
    print("Strategy: Use IWLS underperforming signals to select stocks")
    print("Options: Find 5-15% OTM calls with 300-600 DTE")
    print("Hold: Track basket value for 300 days")
    
    # Set target date in February 2023
    target_date = datetime(2023, 2, 15)  # February 15, 2023
    print(f"\nTarget entry date: {target_date.strftime('%Y-%m-%d')}")
    
    # Setup directories
    v2_dir = "/Users/tim/CODE_PROJECTS/IWLS-OPTIONS/IWLS_ANALYSIS_V2"
    options_base_dir = "/Users/tim/CODE_PROJECTS/IWLS-OPTIONS/OPTIONS_DATASET"
    output_dir = "/Users/tim/CODE_PROJECTS/IWLS-OPTIONS/IWLS_ANALYSIS_V2/OPTIONS_STRATEGY_FEB2023"
    
    os.makedirs(output_dir, exist_ok=True)
    
    if not os.path.exists(v2_dir):
        print("❌ IWLS_ANALYSIS_V2 directory not found.")
        return
    
    if not os.path.exists(options_base_dir):
        print("❌ OPTIONS_DATASET directory not found.")
        return
    
    print(f"IWLS data directory: {v2_dir}")
    print(f"Options data directory: {options_base_dir}")
    print(f"Output directory: {output_dir}")
    
    # Load IWLS data
    all_iwls_data = load_all_iwls_data(v2_dir)
    
    if len(all_iwls_data) == 0:
        print("❌ No IWLS data loaded.")
        return
    
    # Get available assets on target date
    available_assets = get_available_assets_on_date(all_iwls_data, target_date)
    print(f"\nAvailable assets on {target_date.strftime('%Y-%m-%d')}: {len(available_assets)}")
    
    if len(available_assets) == 0:
        print("❌ No assets available on target date.")
        return
    
    # Select underperforming stocks
    selected_stocks = select_underperforming_stocks(available_assets, min_deviation=-15, num_stocks=10)
    
    if len(selected_stocks) == 0:
        print("❌ No underperforming stocks found.")
        return
    
    # Find suitable options
    option_contracts = find_suitable_options(
        selected_stocks, target_date, options_base_dir, min_dte=300, max_dte=600
    )
    
    if len(option_contracts) == 0:
        print("❌ No suitable options found.")
        return
    
    print(f"\n✅ Found {len(option_contracts)} suitable option contracts")
    
    # Track basket performance for 300 days
    daily_values, individual_dfs = track_option_basket_daily(option_contracts, tracking_days=300)
    
    # Calculate individual performance statistics
    performance_stats = calculate_individual_performance_stats(individual_dfs, option_contracts)
    
    # Calculate final performance
    initial_value = daily_values.iloc[0]['basket_value']
    final_value = daily_values.iloc[-1]['basket_value']
    total_return = ((final_value / initial_value) - 1) * 100
    
    # Calculate basket max gain
    basket_max_value = daily_values['basket_value'].max()
    basket_max_gain = ((basket_max_value / initial_value) - 1) * 100
    basket_max_day = daily_values.loc[daily_values['basket_value'].idxmax(), 'day']
    
    print(f"\n" + "="*60)
    print("FINAL RESULTS")
    print("="*60)
    print(f"BASKET PERFORMANCE:")
    print(f"  Initial value: ${initial_value:.2f}")
    print(f"  Final value: ${final_value:.2f}")
    print(f"  Total return: {total_return:+.2f}%")
    print(f"  Max basket value: ${basket_max_value:.2f}")
    print(f"  Max basket gain: {basket_max_gain:+.2f}%")
    print(f"  Day of max gain: {basket_max_day}")
    print(f"  Tracking period: {len(daily_values)} days")
    print(f"  Number of contracts: {len(option_contracts)}")
    
    if len(performance_stats) > 0:
        print(f"\nINDIVIDUAL OPTION PERFORMANCE:")
        print(f"{'Asset':<8} {'Final Ret':<10} {'Max Gain':<10} {'Max Day':<8} {'Strike':<8} {'OTM%':<6}")
        print("-" * 60)
        
        # Sort by total return for display
        display_stats = performance_stats.sort_values('total_return', ascending=False)
        
        for _, row in display_stats.iterrows():
            print(f"{row['asset']:<8} {row['total_return']:>9.1f}% {row['max_gain']:>9.1f}% "
                  f"{row['day_of_max_gain']:>7.0f} ${row['strike_price']:>7.0f} {row['moneyness']:>5.1f}%")
        
        # Calculate some summary stats
        avg_return = performance_stats['total_return'].mean()
        best_performer = performance_stats.loc[performance_stats['total_return'].idxmax()]
        worst_performer = performance_stats.loc[performance_stats['total_return'].idxmin()]
        winners = len(performance_stats[performance_stats['total_return'] > 0])
        
        print(f"\nSUMMARY STATISTICS:")
        print(f"  Average individual return: {avg_return:.2f}%")
        print(f"  Winners: {winners}/{len(performance_stats)} contracts")
        print(f"  Best performer: {best_performer['asset']} ({best_performer['total_return']:+.1f}%)")
        print(f"  Worst performer: {worst_performer['asset']} ({worst_performer['total_return']:+.1f}%)")
        
        # Check if basket performance was driven by outliers
        median_return = performance_stats['total_return'].median()
        print(f"  Median individual return: {median_return:.2f}%")
        
        if total_return > median_return + 20:
            print(f"  📈 Basket outperformed median by {total_return - median_return:.1f}% - likely driven by top performers")
        elif total_return < median_return - 20:
            print(f"  📉 Basket underperformed median by {median_return - total_return:.1f}% - dragged down by poor performers")
        else:
            print(f"  📊 Basket performance close to median - well balanced")
    
    # Show individual contract details
    print(f"\nCONTRACT DETAILS:")
    for _, contract in option_contracts.iterrows():
        print(f"  {contract['asset']}: Strike ${contract['strike_price']:.0f} "
              f"({contract['moneyness']:.1f}% OTM), {contract['dte']} DTE, "
              f"Entry: ${contract['entry_option_price']:.2f}")
    
    # Create comprehensive plots
    create_comprehensive_plots(daily_values, individual_dfs, performance_stats, option_contracts, output_dir)
    
    # Save comprehensive results
    save_comprehensive_results(option_contracts, daily_values, individual_dfs, performance_stats, output_dir)
    
    print(f"\n✅ Analysis complete! Results saved to {output_dir}")

if __name__ == "__main__":
    main()