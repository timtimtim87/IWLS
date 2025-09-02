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
    Track the daily value of the option basket for 300 days
    """
    print(f"\nTracking option basket for {tracking_days} days...")
    
    daily_basket_values = []
    entry_date = option_contracts.iloc[0]['entry_date']
    total_initial_value = option_contracts['entry_option_price'].sum()
    
    print(f"Initial basket value: ${total_initial_value:.2f}")
    print(f"Number of contracts: {len(option_contracts)}")
    
    for day in range(tracking_days + 1):  # Include day 0
        current_date = entry_date + timedelta(days=day)
        daily_total = 0
        valid_prices = 0
        
        for _, contract in option_contracts.iterrows():
            option_file = contract['option_file']
            
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
                    daily_total += current_price
                    valid_prices += 1
                else:
                    # No data available, use last known price or 0
                    past_data = option_df[option_df['date'] <= current_date]
                    if len(past_data) > 0:
                        last_price = past_data.iloc[-1]['close']
                        daily_total += last_price
                        valid_prices += 1
                    # If no past data either, contribute 0 (option may have expired worthless)
            
            except Exception as e:
                # If we can't load data, skip this contract for today
                continue
        
        daily_basket_values.append({
            'day': day,
            'date': current_date,
            'basket_value': daily_total,
            'valid_contracts': valid_prices,
            'return_pct': ((daily_total / total_initial_value) - 1) * 100 if total_initial_value > 0 else 0
        })
        
        if day % 30 == 0:  # Print progress every 30 days
            print(f"  Day {day}: ${daily_total:.2f} ({((daily_total / total_initial_value) - 1) * 100:+.1f}%)")
    
    return pd.DataFrame(daily_basket_values)

def create_performance_plot(daily_values, option_contracts, output_dir):
    """
    Create a simple plot showing the basket value over time
    """
    print("\nCreating performance plot...")
    
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
    
    print(f"Plot saved to: {output_dir}/options_basket_performance.png")

def save_results(option_contracts, daily_values, output_dir):
    """
    Save results to CSV files
    """
    print("\nSaving results...")
    
    # Save option contracts details
    option_contracts.to_csv(f"{output_dir}/selected_option_contracts.csv", index=False)
    print(f"Option contracts saved to: {output_dir}/selected_option_contracts.csv")
    
    # Save daily tracking data
    daily_values.to_csv(f"{output_dir}/daily_basket_values.csv", index=False)
    print(f"Daily values saved to: {output_dir}/daily_basket_values.csv")

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
    daily_values = track_option_basket_daily(option_contracts, tracking_days=300)
    
    # Calculate final performance
    initial_value = daily_values.iloc[0]['basket_value']
    final_value = daily_values.iloc[-1]['basket_value']
    total_return = ((final_value / initial_value) - 1) * 100
    
    print(f"\n" + "="*60)
    print("FINAL RESULTS")
    print("="*60)
    print(f"Initial basket value: ${initial_value:.2f}")
    print(f"Final basket value: ${final_value:.2f}")
    print(f"Total return: {total_return:+.2f}%")
    print(f"Tracking period: {len(daily_values)} days")
    print(f"Number of contracts: {len(option_contracts)}")
    
    # Show individual contract details
    print(f"\nCONTRACT DETAILS:")
    for _, contract in option_contracts.iterrows():
        print(f"  {contract['asset']}: Strike ${contract['strike_price']:.0f} "
              f"({contract['moneyness']:.1f}% OTM), {contract['dte']} DTE, "
              f"Entry: ${contract['entry_option_price']:.2f}")
    
    # Create performance plot
    create_performance_plot(daily_values, option_contracts, output_dir)
    
    # Save results
    save_results(option_contracts, daily_values, output_dir)
    
    print(f"\n✅ Analysis complete! Results saved to {output_dir}")

if __name__ == "__main__":
    main()