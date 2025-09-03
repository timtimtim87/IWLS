import pandas as pd
import numpy as np
import matplotlib.pyplot as plt
from datetime import datetime, timedelta
import warnings
import os
import glob
warnings.filterwarnings('ignore')

# =============================================================================
# CONFIGURABLE PARAMETERS - MODIFY THESE TO CHANGE STRATEGY
# =============================================================================

# Options Selection Parameters
MIN_MONEYNESS = 15        # Minimum % OTM for option selection
MAX_MONEYNESS = 25       # Maximum % OTM for option selection
MIN_DTE = 250             # Minimum days to expiration
MAX_DTE = 350            # Maximum days to expiration

# Strategy Parameters
MIN_DEVIATION = -20      # Minimum price deviation threshold for stock selection
NUM_STOCKS = 10          # Number of underperforming stocks to select
HOLD_DURATION = 200       # Number of days to hold the option basket

# Date Range Parameters
START_DATE = datetime(2022, 1, 3)   # Strategy start date
END_DATE = datetime(2024, 12, 31)   # Strategy end date (will be adjusted for hold duration)
TEST_FREQUENCY = 10       # Test every N days (7 = weekly)

# Data cutoff consideration
DATA_CUTOFF = datetime(2025, 5, 12)  # Approximate end of available data
EFFECTIVE_END_DATE = DATA_CUTOFF - timedelta(days=HOLD_DURATION)  # Don't enter positions too close to data end

# =============================================================================

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
                  not d.startswith('DAILY_TRACKING') and not d.startswith('MULTI') and
                  not d.startswith('OPTIONS_STRATEGY')]
    
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

def select_underperforming_stocks(available_df, min_deviation=MIN_DEVIATION, num_stocks=NUM_STOCKS):
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

def find_suitable_options(selected_stocks, entry_date, options_base_dir, 
                         min_dte=MIN_DTE, max_dte=MAX_DTE, 
                         min_moneyness=MIN_MONEYNESS, max_moneyness=MAX_MONEYNESS):
    """
    Find suitable call options for selected stocks from actual options data
    """
    print(f"\nFinding suitable call options for {len(selected_stocks)} stocks...")
    print(f"Entry date: {entry_date.strftime('%Y-%m-%d')}")
    print(f"Looking for {min_moneyness}-{max_moneyness}% OTM calls with {min_dte}-{max_dte} DTE")
    
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
        target_moneyness = (min_moneyness + max_moneyness) / 2  # Target middle of range
        
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
                    
                    # Check if it's within our moneyness range
                    if min_moneyness <= moneyness <= max_moneyness:
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
                                
                                # Keep the option closest to target moneyness
                                if best_option is None or abs(moneyness - target_moneyness) < abs(best_option['moneyness'] - target_moneyness):
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

def track_option_basket_daily(option_contracts, tracking_days=HOLD_DURATION):
    """
    Track the daily value of the option basket AND individual options for specified duration
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
        
        if day % max(1, tracking_days // 10) == 0:  # Print progress proportionally
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
    
    print("  All plots created")

def save_comprehensive_results(option_contracts, daily_values, individual_dfs, performance_stats, output_dir):
    """
    Save all results including individual tracking data
    """
    print("\nSaving comprehensive results...")
    
    # Save option contracts details
    option_contracts.to_csv(f"{output_dir}/selected_option_contracts.csv", index=False)
    print(f"  Option contracts: selected_option_contracts.csv")
    
    # Save daily tracking data
    daily_values.to_csv(f"{output_dir}/daily_basket_values.csv", index=False)
    print(f"  Daily basket values: daily_basket_values.csv")
    
    # Save individual option tracking
    for asset, df in individual_dfs.items():
        df.to_csv(f"{output_dir}/individual_{asset}_daily.csv", index=False)
    print(f"  Individual daily tracking: individual_[ASSET]_daily.csv")
    
    # Save performance statistics
    performance_stats.to_csv(f"{output_dir}/individual_performance_stats.csv", index=False)
    print(f"  Performance statistics: individual_performance_stats.csv")

def process_single_date(target_date, all_iwls_data, options_base_dir, base_output_dir):
    """
    Process a single date for the options strategy
    """
    date_str = target_date.strftime('%Y%m%d')
    output_dir = os.path.join(base_output_dir, date_str)
    os.makedirs(output_dir, exist_ok=True)
    
    try:
        # Get available assets on target date
        available_assets = get_available_assets_on_date(all_iwls_data, target_date)
        
        if len(available_assets) == 0:
            print(f"    No assets available on {date_str}")
            return False
        
        # Select underperforming stocks
        selected_stocks = select_underperforming_stocks(available_assets)
        
        if len(selected_stocks) == 0:
            print(f"    No underperforming stocks found on {date_str}")
            return False
        
        # Find suitable options
        option_contracts = find_suitable_options(selected_stocks, target_date, options_base_dir)
        
        if len(option_contracts) == 0:
            print(f"    No suitable options found on {date_str}")
            return False
        
        print(f"    Found {len(option_contracts)} contracts on {date_str}")
        
        # Track basket performance for specified duration
        daily_values, individual_dfs = track_option_basket_daily(option_contracts)
        
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
        
        print(f"    Final return: {total_return:+.1f}%, Max gain: {basket_max_gain:+.1f}%")
        
        # Create comprehensive plots
        create_comprehensive_plots(daily_values, individual_dfs, performance_stats, option_contracts, output_dir)
        
        # Save comprehensive results
        save_comprehensive_results(option_contracts, daily_values, individual_dfs, performance_stats, output_dir)
        
        # Return summary data for aggregation
        return {
            'date': target_date,
            'date_str': date_str,
            'num_contracts': len(option_contracts),
            'initial_value': initial_value,
            'final_value': final_value,
            'total_return': total_return,
            'max_gain': basket_max_gain,
            'day_of_max_gain': basket_max_day,
            'winners': len(performance_stats[performance_stats['total_return'] > 0]) if len(performance_stats) > 0 else 0,
            'avg_individual_return': performance_stats['total_return'].mean() if len(performance_stats) > 0 else 0,
            'best_individual_return': performance_stats['total_return'].max() if len(performance_stats) > 0 else 0,
            'worst_individual_return': performance_stats['total_return'].min() if len(performance_stats) > 0 else 0
        }
    
    except Exception as e:
        print(f"    Error processing {date_str}: {e}")
        return False

def create_overall_summary_analysis(all_results, base_output_dir):
    """
    Create overall summary analysis across all processed dates with enhanced plots
    """
    print("\nCreating overall summary analysis...")
    
    if len(all_results) == 0:
        print("No successful results to analyze")
        return
    
    # Convert to DataFrame
    results_df = pd.DataFrame(all_results)
    
    # Create summary statistics
    summary_stats = {
        'total_strategies_tested': len(results_df),
        'date_range_start': results_df['date'].min().strftime('%Y-%m-%d'),
        'date_range_end': results_df['date'].max().strftime('%Y-%m-%d'),
        'avg_basket_return': results_df['total_return'].mean(),
        'median_basket_return': results_df['total_return'].median(),
        'best_basket_return': results_df['total_return'].max(),
        'worst_basket_return': results_df['total_return'].min(),
        'avg_max_gain': results_df['max_gain'].mean(),
        'median_max_gain': results_df['max_gain'].median(),
        'best_max_gain': results_df['max_gain'].max(),
        'profitable_strategies': len(results_df[results_df['total_return'] > 0]),
        'win_rate': len(results_df[results_df['total_return'] > 0]) / len(results_df) * 100,
        'avg_contracts_per_strategy': results_df['num_contracts'].mean(),
        'total_contracts_analyzed': results_df['num_contracts'].sum(),
        'hold_duration_days': HOLD_DURATION
    }
    
    # Save detailed results
    results_df.to_csv(f"{base_output_dir}/ALL_STRATEGIES_SUMMARY.csv", index=False)
    
    # Save summary statistics
    summary_df = pd.DataFrame([summary_stats])
    summary_df.to_csv(f"{base_output_dir}/OVERALL_SUMMARY_STATS.csv", index=False)
    
    # ENHANCED PLOTTING - Create overall performance plots with new charts
    fig, axes = plt.subplots(2, 3, figsize=(20, 12))
    
    # Plot 1: Distribution of final returns
    axes[0,0].hist(results_df['total_return'], bins=30, alpha=0.7, edgecolor='black', color='steelblue')
    axes[0,0].axvline(results_df['total_return'].mean(), color='red', linestyle='--', 
                      label=f'Mean: {results_df["total_return"].mean():.1f}%')
    axes[0,0].axvline(results_df['total_return'].median(), color='orange', linestyle='--', 
                      label=f'Median: {results_df["total_return"].median():.1f}%')
    axes[0,0].axvline(0, color='black', linestyle='-', alpha=0.5, label='Break Even')
    axes[0,0].set_xlabel('Final Return (%)')
    axes[0,0].set_ylabel('Frequency')
    axes[0,0].set_title('Distribution of Final Returns', fontweight='bold')
    axes[0,0].legend()
    axes[0,0].grid(True, alpha=0.3)
    
    # Plot 2: Distribution of max gains
    axes[0,1].hist(results_df['max_gain'], bins=30, alpha=0.7, color='lightgreen', edgecolor='black')
    axes[0,1].axvline(results_df['max_gain'].mean(), color='red', linestyle='--', 
                      label=f'Mean: {results_df["max_gain"].mean():.1f}%')
    axes[0,1].axvline(results_df['max_gain'].median(), color='orange', linestyle='--', 
                      label=f'Median: {results_df["max_gain"].median():.1f}%')
    axes[0,1].set_xlabel('Max Gain (%)')
    axes[0,1].set_ylabel('Frequency')
    axes[0,1].set_title('Distribution of Maximum Gains', fontweight='bold')
    axes[0,1].legend()
    axes[0,1].grid(True, alpha=0.3)
    
    # Plot 3: Final returns vs max gains scatter
    colors = ['green' if x >= 0 else 'red' for x in results_df['total_return']]
    axes[0,2].scatter(results_df['max_gain'], results_df['total_return'], 
                      c=colors, alpha=0.6, s=60)
    axes[0,2].plot([0, results_df['max_gain'].max()], [0, results_df['max_gain'].max()], 
                   'k--', alpha=0.5, label='Perfect Hold (Max=Final)')
    axes[0,2].set_xlabel('Max Gain (%)')
    axes[0,2].set_ylabel('Final Return (%)')
    axes[0,2].set_title('Max Gain vs Final Return', fontweight='bold')
    axes[0,2].legend()
    axes[0,2].grid(True, alpha=0.3)
    axes[0,2].axhline(y=0, color='black', linestyle='-', alpha=0.3)
    axes[0,2].axvline(x=0, color='black', linestyle='-', alpha=0.3)
    
    # Plot 4: Final returns over time
    axes[1,0].plot(results_df['date'], results_df['total_return'], alpha=0.7, marker='o', markersize=3, color='blue')
    axes[1,0].axhline(0, color='red', linestyle='--', alpha=0.5, label='Break Even')
    axes[1,0].axhline(results_df['total_return'].mean(), color='green', linestyle='--', alpha=0.7, 
                      label=f'Average: {results_df["total_return"].mean():.1f}%')
    axes[1,0].set_xlabel('Entry Date')
    axes[1,0].set_ylabel('Final Return (%)')
    axes[1,0].set_title('Final Returns Over Time', fontweight='bold')
    axes[1,0].legend()
    axes[1,0].grid(True, alpha=0.3)
    axes[1,0].tick_params(axis='x', rotation=45)
    
    # Plot 5: Max gains over time
    axes[1,1].plot(results_df['date'], results_df['max_gain'], alpha=0.7, marker='o', markersize=3, color='darkgreen')
    axes[1,1].axhline(results_df['max_gain'].mean(), color='red', linestyle='--', alpha=0.7, 
                      label=f'Average: {results_df["max_gain"].mean():.1f}%')
    axes[1,1].set_xlabel('Entry Date')
    axes[1,1].set_ylabel('Max Gain (%)')
    axes[1,1].set_title('Maximum Gains Over Time', fontweight='bold')
    axes[1,1].legend()
    axes[1,1].grid(True, alpha=0.3)
    axes[1,1].tick_params(axis='x', rotation=45)
    
    # Plot 6: Win rate by quarter
    results_df['year_quarter'] = results_df['date'].dt.to_period('Q')
    quarterly_stats = results_df.groupby('year_quarter').agg({
        'total_return': lambda x: (x > 0).mean() * 100,
        'date': 'count',
        'max_gain': 'mean'
    }).rename(columns={'total_return': 'win_rate', 'date': 'count', 'max_gain': 'avg_max_gain'})
    
    # Only show quarters with at least 3 strategies
    quarterly_stats = quarterly_stats[quarterly_stats['count'] >= 3]
    
    if len(quarterly_stats) > 0:
        x_pos = range(len(quarterly_stats))
        bars = axes[1,2].bar(x_pos, quarterly_stats['win_rate'], alpha=0.7, color='purple')
        axes[1,2].set_xticks(x_pos)
        axes[1,2].set_xticklabels([str(q) for q in quarterly_stats.index], rotation=45)
        axes[1,2].axhline(50, color='red', linestyle='--', alpha=0.5, label='50% baseline')
        axes[1,2].set_ylabel('Win Rate (%)')
        axes[1,2].set_title('Quarterly Win Rate (min 3 strategies)', fontweight='bold')
        axes[1,2].legend()
        axes[1,2].grid(True, alpha=0.3)
        
        # Add value labels on bars
        for bar, value in zip(bars, quarterly_stats['win_rate']):
            axes[1,2].text(bar.get_x() + bar.get_width()/2., bar.get_height() + 1,
                          f'{value:.0f}%', ha='center', va='bottom', fontweight='bold')
    
    plt.tight_layout()
    plt.savefig(f"{base_output_dir}/OVERALL_PERFORMANCE_SUMMARY.png", dpi=300, bbox_inches='tight')
    plt.close()
    
    # NEW: Create separate detailed charts for max gains and final returns
    fig, ((ax1, ax2), (ax3, ax4)) = plt.subplots(2, 2, figsize=(16, 12))
    
    # Chart 1: Final returns by trading session (chronological)
    colors1 = ['green' if x >= 0 else 'red' for x in results_df['total_return']]
    bars1 = ax1.bar(range(len(results_df)), results_df['total_return'], color=colors1, alpha=0.7)
    ax1.set_xlabel('Trading Session (Chronological)')
    ax1.set_ylabel('Final Return (%)')
    ax1.set_title('Final Returns by Trading Session', fontweight='bold')
    ax1.axhline(y=0, color='black', linestyle='-', alpha=0.5)
    ax1.axhline(y=results_df['total_return'].mean(), color='blue', linestyle='--', alpha=0.7,
                label=f'Mean: {results_df["total_return"].mean():.1f}%')
    ax1.grid(True, alpha=0.3)
    ax1.legend()
    
    # Chart 2: Max gains by trading session (chronological)
    bars2 = ax2.bar(range(len(results_df)), results_df['max_gain'], color='lightgreen', alpha=0.7)
    ax2.set_xlabel('Trading Session (Chronological)')
    ax2.set_ylabel('Max Gain (%)')
    ax2.set_title('Maximum Gains by Trading Session', fontweight='bold')
    ax2.axhline(y=results_df['max_gain'].mean(), color='blue', linestyle='--', alpha=0.7,
                label=f'Mean: {results_df["max_gain"].mean():.1f}%')
    ax2.grid(True, alpha=0.3)
    ax2.legend()
    
    # Chart 3: Final returns sorted (best to worst)
    sorted_returns = results_df['total_return'].sort_values(ascending=False).reset_index(drop=True)
    colors3 = ['green' if x >= 0 else 'red' for x in sorted_returns]
    bars3 = ax3.bar(range(len(sorted_returns)), sorted_returns, color=colors3, alpha=0.7)
    ax3.set_xlabel('Strategy Rank (Best to Worst)')
    ax3.set_ylabel('Final Return (%)')
    ax3.set_title('Final Returns Ranked', fontweight='bold')
    ax3.axhline(y=0, color='black', linestyle='-', alpha=0.5)
    ax3.axhline(y=sorted_returns.mean(), color='blue', linestyle='--', alpha=0.7,
                label=f'Mean: {sorted_returns.mean():.1f}%')
    ax3.grid(True, alpha=0.3)
    ax3.legend()
    
    # Chart 4: Max gains sorted (best to worst)
    sorted_max_gains = results_df['max_gain'].sort_values(ascending=False).reset_index(drop=True)
    bars4 = ax4.bar(range(len(sorted_max_gains)), sorted_max_gains, color='lightgreen', alpha=0.7)
    ax4.set_xlabel('Strategy Rank (Best to Worst Max Gain)')
    ax4.set_ylabel('Max Gain (%)')
    ax4.set_title('Maximum Gains Ranked', fontweight='bold')
    ax4.axhline(y=sorted_max_gains.mean(), color='blue', linestyle='--', alpha=0.7,
                label=f'Mean: {sorted_max_gains.mean():.1f}%')
    ax4.grid(True, alpha=0.3)
    ax4.legend()
    
    plt.tight_layout()
    plt.savefig(f"{base_output_dir}/TRADING_SESSIONS_DETAILED.png", dpi=300, bbox_inches='tight')
    plt.close()
    
    # Print summary
    print(f"\n" + "="*80)
    print("OVERALL STRATEGY PERFORMANCE SUMMARY")
    print("="*80)
    print(f"STRATEGY PARAMETERS:")
    print(f"  Moneyness range: {MIN_MONEYNESS}% to {MAX_MONEYNESS}% OTM")
    print(f"  DTE range: {MIN_DTE} to {MAX_DTE} days")
    print(f"  Hold duration: {HOLD_DURATION} days")
    print(f"  Min deviation threshold: {MIN_DEVIATION}%")
    print(f"  Number of stocks per basket: {NUM_STOCKS}")
    print(f"  Test frequency: Every {TEST_FREQUENCY} days")
    print(f"\nBACKTEST RESULTS:")
    print(f"  Total strategies tested: {summary_stats['total_strategies_tested']}")
    print(f"  Date range: {summary_stats['date_range_start']} to {summary_stats['date_range_end']}")
    print(f"  Win rate: {summary_stats['win_rate']:.1f}%")
    print(f"\nFINAL RETURNS:")
    print(f"  Average: {summary_stats['avg_basket_return']:.2f}%")
    print(f"  Median: {summary_stats['median_basket_return']:.2f}%")
    print(f"  Best: {summary_stats['best_basket_return']:.2f}%")
    print(f"  Worst: {summary_stats['worst_basket_return']:.2f}%")
    print(f"\nMAXIMUM GAINS:")
    print(f"  Average: {summary_stats['avg_max_gain']:.2f}%")
    print(f"  Median: {summary_stats['median_max_gain']:.2f}%")
    print(f"  Best: {summary_stats['best_max_gain']:.2f}%")
    print(f"\nPORTFOLIO STATS:")
    print(f"  Total option contracts analyzed: {summary_stats['total_contracts_analyzed']:,}")
    print(f"  Average contracts per strategy: {summary_stats['avg_contracts_per_strategy']:.1f}")

def main():
    print("CONFIGURABLE OPTIONS STRATEGY BASED ON IWLS DEVIATION SIGNALS")
    print("=" * 80)
    print("CURRENT STRATEGY PARAMETERS:")
    print(f"  Stock Selection: Deviation < {MIN_DEVIATION}% (most underperforming {NUM_STOCKS} stocks)")
    print(f"  Option Selection: {MIN_MONEYNESS}%-{MAX_MONEYNESS}% OTM calls, {MIN_DTE}-{MAX_DTE} DTE")
    print(f"  Hold Duration: {HOLD_DURATION} days")
    print(f"  Test Period: {START_DATE.strftime('%Y-%m-%d')} to {min(END_DATE, EFFECTIVE_END_DATE).strftime('%Y-%m-%d')}")
    print(f"  Test Frequency: Every {TEST_FREQUENCY} days")
    print(f"  Data Cutoff: {DATA_CUTOFF.strftime('%Y-%m-%d')} (effective end: {EFFECTIVE_END_DATE.strftime('%Y-%m-%d')})")
    
    # Setup directories with correct paths
    v2_dir = "/Users/tim/CODE_PROJECTS/IWLS-OPTIONS/IWLS_ANALYSIS_V2"
    options_base_dir = "/Users/tim/CODE_PROJECTS/IWLS-OPTIONS/OPTIONS_DATASET"
    
    # Create output directory name based on parameters
    params_str = f"DTE{MIN_DTE}-{MAX_DTE}_OTM{MIN_MONEYNESS}-{MAX_MONEYNESS}_HOLD{HOLD_DURATION}"
    base_output_dir = f"/Users/tim/CODE_PROJECTS/IWLS-OPTIONS/IWLS_ANALYSIS_V2/OPTIONS_STRATEGY_{params_str}"
    
    os.makedirs(base_output_dir, exist_ok=True)
    
    if not os.path.exists(v2_dir):
        print(f"ERROR: IWLS_ANALYSIS_V2 directory not found: {v2_dir}")
        return
    
    if not os.path.exists(options_base_dir):
        print(f"ERROR: OPTIONS_DATASET directory not found: {options_base_dir}")
        return
    
    print(f"\nDIRECTORIES:")
    print(f"  IWLS data: {v2_dir}")
    print(f"  Options data: {options_base_dir}")
    print(f"  Output: {base_output_dir}")
    
    # Load IWLS data once
    print("\nLoading IWLS data...")
    all_iwls_data = load_all_iwls_data(v2_dir)
    
    if len(all_iwls_data) == 0:
        print("ERROR: No IWLS data loaded.")
        return
    
    print(f"Loaded IWLS data for {len(all_iwls_data)} assets")
    
    # Calculate effective end date (don't enter positions too close to data cutoff)
    effective_end = min(END_DATE, EFFECTIVE_END_DATE)
    
    # Generate date range
    current_date = START_DATE
    test_dates = []
    
    while current_date <= effective_end:
        # Only add weekdays (Monday=0, Friday=4)
        if current_date.weekday() < 5:
            test_dates.append(current_date)
        current_date += timedelta(days=TEST_FREQUENCY)
    
    print(f"\nGenerating {len(test_dates)} test dates")
    print(f"Frequency: Every {TEST_FREQUENCY} days from {START_DATE.strftime('%Y-%m-%d')} to {effective_end.strftime('%Y-%m-%d')}")
    
    if len(test_dates) == 0:
        print("ERROR: No test dates generated. Check date range logic.")
        return
    
    # Process each date
    all_results = []
    successful_strategies = 0
    
    for i, target_date in enumerate(test_dates):
        print(f"\n[{i+1}/{len(test_dates)}] Processing {target_date.strftime('%Y-%m-%d')}")
        
        result = process_single_date(target_date, all_iwls_data, options_base_dir, base_output_dir)
        
        if result:
            all_results.append(result)
            successful_strategies += 1
        
        # Print progress every 20 strategies
        if (i + 1) % 20 == 0:
            print(f"    Progress: {i+1}/{len(test_dates)} dates processed, {successful_strategies} successful strategies")
    
    print(f"\n" + "="*80)
    print("BACKTEST COMPLETE")
    print("="*80)
    print(f"Total dates tested: {len(test_dates)}")
    print(f"Successful strategies: {successful_strategies}")
    
    if len(test_dates) > 0:
        print(f"Success rate: {successful_strategies/len(test_dates)*100:.1f}%")
    else:
        print("Success rate: N/A (no dates to test)")
    
    if len(all_results) > 0:
        # Create overall summary analysis
        create_overall_summary_analysis(all_results, base_output_dir)
        
        print(f"\nFull backtest complete!")
        print(f"Individual strategy results: {base_output_dir}/[YYYYMMDD]/")
        print(f"Overall summary: {base_output_dir}/ALL_STRATEGIES_SUMMARY.csv")
        print(f"Performance charts: {base_output_dir}/OVERALL_PERFORMANCE_SUMMARY.png")
        print(f"Detailed trading sessions: {base_output_dir}/TRADING_SESSIONS_DETAILED.png")
    else:
        print("ERROR: No successful strategies found in the entire backtest period")
        print("Try adjusting parameters:")
        print(f"  - Increase DTE range (currently {MIN_DTE}-{MAX_DTE})")
        print(f"  - Widen moneyness range (currently {MIN_MONEYNESS}%-{MAX_MONEYNESS}%)")
        print(f"  - Reduce deviation threshold (currently {MIN_DEVIATION}%)")
        print(f"  - Reduce hold duration (currently {HOLD_DURATION} days)")
    
    print(f"\nResults saved to: {base_output_dir}")

if __name__ == "__main__":
    main()