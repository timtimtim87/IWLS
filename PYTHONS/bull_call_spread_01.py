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

# Bull Call Spread Parameters
BTO_MIN_MONEYNESS = -5       # BTO (Buy To Open) minimum % from ATM (can be ITM)
BTO_MAX_MONEYNESS = 5       # BTO (Buy To Open) maximum % OTM
STO_TARGET_MONEYNESS = 15    # STO (Sell To Open) target % OTM
STO_TOLERANCE = 3            # STO tolerance around target (12-18% range)

# Options Selection Parameters
MIN_DTE = 300             # Minimum days to expiration
MAX_DTE = 350            # Maximum days to expiration

# Strategy Parameters
MIN_DEVIATION = -15      # Minimum price deviation threshold for stock selection
NUM_STOCKS = 10          # Number of underperforming stocks to select
HOLD_DURATION = 300       # Number of days to hold the spread basket

# Date Range Parameters
START_DATE = datetime(2022, 1, 3)   # Strategy start date
END_DATE = datetime(2024, 12, 31)   # Strategy end date (will be adjusted for hold duration)
TEST_FREQUENCY = 20       # Test every N days (7 = weekly)

# Data cutoff consideration
DATA_CUTOFF = datetime(2025, 5, 31)  # Approximate end of available data
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

def find_suitable_bull_call_spreads(selected_stocks, entry_date, options_base_dir, 
                                   min_dte=MIN_DTE, max_dte=MAX_DTE):
    """
    Find suitable bull call spreads for selected stocks
    BTO: Buy call closer to ATM (BTO_MIN_MONEYNESS to BTO_MAX_MONEYNESS)
    STO: Sell call near STO_TARGET_MONEYNESS% OTM
    """
    print(f"\nFinding suitable bull call spreads for {len(selected_stocks)} stocks...")
    print(f"Entry date: {entry_date.strftime('%Y-%m-%d')}")
    print(f"BTO range: {BTO_MIN_MONEYNESS}% to {BTO_MAX_MONEYNESS}% from ATM")
    print(f"STO target: {STO_TARGET_MONEYNESS}% OTM (±{STO_TOLERANCE}%)")
    print(f"DTE range: {min_dte}-{max_dte} days")
    
    suitable_spreads = []
    
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
        
        best_spread = None
        
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
                
                # Parse all available strikes and their option data
                available_options = []
                
                for option_file in option_files:
                    # Extract strike price from filename (format: ASSET_DATE_C_STRIKE.csv)
                    filename = os.path.basename(option_file)
                    try:
                        strike_str = filename.split('_C_')[1].replace('.csv', '')
                        strike_price = float(strike_str)
                    except:
                        continue
                    
                    # Calculate moneyness
                    moneyness = (strike_price - stock_price) / stock_price * 100
                    
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
                            
                            available_options.append({
                                'strike_price': strike_price,
                                'moneyness': moneyness,
                                'option_price': entry_row['close'],
                                'option_file': option_file,
                                'entry_date': entry_row['date'],
                                'options_symbol': entry_row['options_symbol']
                            })
                    
                    except Exception as e:
                        continue
                
                # Now find the best BTO/STO combination
                if len(available_options) >= 2:  # Need at least 2 strikes for a spread
                    # Find BTO candidates (closer to ATM)
                    bto_candidates = [opt for opt in available_options 
                                    if BTO_MIN_MONEYNESS <= opt['moneyness'] <= BTO_MAX_MONEYNESS]
                    
                    # Find STO candidates (near target OTM)
                    sto_min = STO_TARGET_MONEYNESS - STO_TOLERANCE
                    sto_max = STO_TARGET_MONEYNESS + STO_TOLERANCE
                    sto_candidates = [opt for opt in available_options 
                                    if sto_min <= opt['moneyness'] <= sto_max]
                    
                    if bto_candidates and sto_candidates:
                        # Find best combination
                        # BTO: closest to middle of range
                        bto_target = (BTO_MIN_MONEYNESS + BTO_MAX_MONEYNESS) / 2
                        best_bto = min(bto_candidates, key=lambda x: abs(x['moneyness'] - bto_target))
                        
                        # STO: closest to target moneyness
                        best_sto = min(sto_candidates, key=lambda x: abs(x['moneyness'] - STO_TARGET_MONEYNESS))
                        
                        # Ensure BTO strike < STO strike (proper bull spread)
                        if best_bto['strike_price'] < best_sto['strike_price']:
                            # Calculate spread metrics
                            net_debit = best_bto['option_price'] - best_sto['option_price']
                            max_profit = (best_sto['strike_price'] - best_bto['strike_price']) - net_debit
                            max_loss = net_debit
                            
                            # Only consider spreads with positive max profit potential
                            if max_profit > 0 and net_debit > 0:
                                spread_info = {
                                    'asset': asset,
                                    'stock_price': stock_price,
                                    'price_deviation': stock['price_deviation'],
                                    'dte': dte,
                                    'expiration_date': exp_date,
                                    'entry_date': best_bto['entry_date'],
                                    'bto_strike': best_bto['strike_price'],
                                    'bto_moneyness': best_bto['moneyness'],
                                    'bto_price': best_bto['option_price'],
                                    'bto_file': best_bto['option_file'],
                                    'bto_symbol': best_bto['options_symbol'],
                                    'sto_strike': best_sto['strike_price'],
                                    'sto_moneyness': best_sto['moneyness'],
                                    'sto_price': best_sto['option_price'],
                                    'sto_file': best_sto['option_file'],
                                    'sto_symbol': best_sto['options_symbol'],
                                    'net_debit': net_debit,
                                    'max_profit': max_profit,
                                    'max_loss': max_loss,
                                    'profit_ratio': max_profit / max_loss if max_loss > 0 else 0
                                }
                                
                                # Keep the spread with the best profit ratio for this expiration
                                if best_spread is None or spread_info['profit_ratio'] > best_spread['profit_ratio']:
                                    best_spread = spread_info
            
            except Exception as e:
                continue
        
        if best_spread:
            suitable_spreads.append(best_spread)
            print(f"    Selected spread:")
            print(f"      BTO: ${best_spread['bto_strike']:.0f} strike ({best_spread['bto_moneyness']:.1f}% OTM) @ ${best_spread['bto_price']:.2f}")
            print(f"      STO: ${best_spread['sto_strike']:.0f} strike ({best_spread['sto_moneyness']:.1f}% OTM) @ ${best_spread['sto_price']:.2f}")
            print(f"      Net Debit: ${best_spread['net_debit']:.2f}, Max Profit: ${best_spread['max_profit']:.2f}")
            print(f"      Profit Ratio: {best_spread['profit_ratio']:.2f}, DTE: {best_spread['dte']}")
        else:
            print(f"    No suitable spread found for {asset}")
    
    return pd.DataFrame(suitable_spreads)

def track_spread_basket_daily(spread_contracts, tracking_days=HOLD_DURATION):
    """
    Track the daily value of the bull call spread basket for specified duration
    """
    print(f"\nTracking bull call spread basket for {tracking_days} days...")
    
    daily_basket_values = []
    individual_tracking = {contract['asset']: [] for _, contract in spread_contracts.iterrows()}
    entry_date = spread_contracts.iloc[0]['entry_date']
    total_initial_debit = spread_contracts['net_debit'].sum()
    
    print(f"Initial basket debit: ${total_initial_debit:.2f}")
    print(f"Number of spreads: {len(spread_contracts)}")
    
    for day in range(tracking_days + 1):  # Include day 0
        current_date = entry_date + timedelta(days=day)
        daily_total_value = 0
        valid_spreads = 0
        individual_values = {}
        
        for _, contract in spread_contracts.iterrows():
            asset = contract['asset']
            bto_file = contract['bto_file']
            sto_file = contract['sto_file']
            initial_debit = contract['net_debit']
            
            bto_price = 0
            sto_price = 0
            spread_value = 0
            
            try:
                # Load BTO option data
                bto_df = pd.read_csv(bto_file)
                bto_df['date'] = pd.to_datetime(bto_df['date'])
                
                # Find BTO price around current date
                price_mask = (bto_df['date'] >= current_date - timedelta(days=5)) & \
                           (bto_df['date'] <= current_date + timedelta(days=5))
                price_data = bto_df[price_mask]
                
                if len(price_data) > 0:
                    closest_idx = (price_data['date'] - current_date).abs().idxmin()
                    bto_price = price_data.loc[closest_idx, 'close']
                else:
                    # Use last known price
                    past_data = bto_df[bto_df['date'] <= current_date]
                    if len(past_data) > 0:
                        bto_price = past_data.iloc[-1]['close']
                
                # Load STO option data
                sto_df = pd.read_csv(sto_file)
                sto_df['date'] = pd.to_datetime(sto_df['date'])
                
                # Find STO price around current date
                price_mask = (sto_df['date'] >= current_date - timedelta(days=5)) & \
                           (sto_df['date'] <= current_date + timedelta(days=5))
                price_data = sto_df[price_mask]
                
                if len(price_data) > 0:
                    closest_idx = (price_data['date'] - current_date).abs().idxmin()
                    sto_price = price_data.loc[closest_idx, 'close']
                else:
                    # Use last known price
                    past_data = sto_df[sto_df['date'] <= current_date]
                    if len(past_data) > 0:
                        sto_price = past_data.iloc[-1]['close']
                
                # Calculate spread value (BTO value - STO value, since we sold the STO)
                spread_value = bto_price - sto_price
                valid_spreads += 1
                
            except Exception as e:
                # If we can't load data, spread_value remains 0
                pass
            
            daily_total_value += spread_value
            individual_values[asset] = spread_value
            
            # Calculate P&L (current value - initial debit paid)
            pnl = spread_value - initial_debit
            pnl_pct = (pnl / initial_debit) * 100 if initial_debit > 0 else 0
            
            # Track individual spread performance
            individual_tracking[asset].append({
                'day': day,
                'date': current_date,
                'spread_value': spread_value,
                'initial_debit': initial_debit,
                'bto_price': bto_price,
                'sto_price': sto_price,
                'pnl': pnl,
                'pnl_pct': pnl_pct
            })
        
        # Calculate basket P&L
        basket_pnl = daily_total_value - total_initial_debit
        basket_pnl_pct = (basket_pnl / total_initial_debit) * 100 if total_initial_debit > 0 else 0
        
        # Track basket performance
        daily_basket_values.append({
            'day': day,
            'date': current_date,
            'basket_value': daily_total_value,
            'initial_debit': total_initial_debit,
            'basket_pnl': basket_pnl,
            'basket_pnl_pct': basket_pnl_pct,
            'valid_spreads': valid_spreads,
            **{f'{asset}_value': individual_values.get(asset, 0) for asset in individual_values}
        })
        
        if day % max(1, tracking_days // 10) == 0:  # Print progress proportionally
            print(f"  Day {day}: P&L ${basket_pnl:.2f} ({basket_pnl_pct:+.1f}%)")
    
    # Convert individual tracking to DataFrames
    individual_dfs = {}
    for asset, tracking_data in individual_tracking.items():
        individual_dfs[asset] = pd.DataFrame(tracking_data)
    
    return pd.DataFrame(daily_basket_values), individual_dfs

def calculate_individual_spread_stats(individual_dfs, spread_contracts):
    """
    Calculate performance statistics for each individual bull call spread
    """
    print("\nCalculating individual spread performance statistics...")
    
    performance_stats = []
    
    for asset, df in individual_dfs.items():
        if len(df) == 0:
            continue
            
        # Get contract details
        contract_info = spread_contracts[spread_contracts['asset'] == asset].iloc[0]
        
        # Calculate performance metrics
        initial_debit = df.iloc[0]['initial_debit']
        final_pnl = df.iloc[-1]['pnl']
        max_pnl = df['pnl'].max()
        min_pnl = df['pnl'].min()
        
        final_return = df.iloc[-1]['pnl_pct']
        max_gain = (max_pnl / initial_debit) * 100 if initial_debit > 0 else 0
        max_loss = (min_pnl / initial_debit) * 100 if initial_debit > 0 else 0
        
        # Find day of max gain
        max_day = df.loc[df['pnl'].idxmax(), 'day'] if max_pnl > min_pnl else 0
        
        # Calculate additional metrics
        profitable_days = len(df[df['pnl'] > 0])
        total_days = len(df)
        profitable_rate = (profitable_days / total_days) * 100 if total_days > 0 else 0
        
        performance_stats.append({
            'asset': asset,
            'bto_strike': contract_info['bto_strike'],
            'sto_strike': contract_info['sto_strike'],
            'bto_moneyness': contract_info['bto_moneyness'],
            'sto_moneyness': contract_info['sto_moneyness'],
            'dte': contract_info['dte'],
            'price_deviation': contract_info['price_deviation'],
            'initial_debit': initial_debit,
            'max_profit_potential': contract_info['max_profit'],
            'final_pnl': final_pnl,
            'max_pnl': max_pnl,
            'min_pnl': min_pnl,
            'final_return': final_return,
            'max_gain': max_gain,
            'max_loss': max_loss,
            'day_of_max_gain': max_day,
            'profitable_days': profitable_days,
            'profitable_rate': profitable_rate,
            'profit_ratio': contract_info['profit_ratio']
        })
    
    return pd.DataFrame(performance_stats)

def create_spread_plots(daily_values, individual_dfs, performance_stats, spread_contracts, output_dir):
    """
    Create comprehensive plots for bull call spread performance
    """
    print("\nCreating comprehensive spread performance plots...")
    
    # Figure 1: Basket performance plots
    fig, (ax1, ax2) = plt.subplots(2, 1, figsize=(12, 10))
    
    # Plot 1: Basket P&L over time
    ax1.plot(daily_values['day'], daily_values['basket_pnl'], linewidth=2, color='blue')
    ax1.axhline(y=0, color='red', linestyle='--', label='Break Even')
    ax1.set_xlabel('Days')
    ax1.set_ylabel('Basket P&L ($)')
    ax1.set_title('Bull Call Spread Basket P&L Over Time', fontweight='bold')
    ax1.grid(True, alpha=0.3)
    ax1.legend()
    
    # Plot 2: Basket return percentage
    ax2.plot(daily_values['day'], daily_values['basket_pnl_pct'], linewidth=2, color='green')
    ax2.axhline(y=0, color='red', linestyle='--', label='Break Even')
    ax2.set_xlabel('Days')
    ax2.set_ylabel('Return (%)')
    ax2.set_title('Bull Call Spread Basket Return Percentage', fontweight='bold')
    ax2.grid(True, alpha=0.3)
    ax2.legend()
    
    plt.tight_layout()
    plt.savefig(f"{output_dir}/spread_basket_performance.png", dpi=300, bbox_inches='tight')
    plt.close()
    
    # Figure 2: Individual spread performance
    n_spreads = len(individual_dfs)
    if n_spreads > 0:
        fig, axes = plt.subplots(2, 1, figsize=(14, 12))
        
        # Plot individual spread P&L
        for asset, df in individual_dfs.items():
            axes[0].plot(df['day'], df['pnl'], label=asset, linewidth=1.5, alpha=0.8)
        
        axes[0].axhline(y=0, color='red', linestyle='--', label='Break Even', alpha=0.5)
        axes[0].set_xlabel('Days')
        axes[0].set_ylabel('P&L ($)')
        axes[0].set_title('Individual Spread P&L Over Time', fontweight='bold')
        axes[0].grid(True, alpha=0.3)
        axes[0].legend(bbox_to_anchor=(1.05, 1), loc='upper left')
        
        # Plot individual spread returns
        for asset, df in individual_dfs.items():
            axes[1].plot(df['day'], df['pnl_pct'], label=asset, linewidth=1.5, alpha=0.8)
        
        axes[1].axhline(y=0, color='red', linestyle='--', label='Break Even', alpha=0.5)
        axes[1].set_xlabel('Days')
        axes[1].set_ylabel('Return (%)')
        axes[1].set_title('Individual Spread Returns Over Time', fontweight='bold')
        axes[1].grid(True, alpha=0.3)
        axes[1].legend(bbox_to_anchor=(1.05, 1), loc='upper left')
        
        plt.tight_layout()
        plt.savefig(f"{output_dir}/individual_spread_performance.png", dpi=300, bbox_inches='tight')
        plt.close()
    
    # Figure 3: Performance comparison charts
    if len(performance_stats) > 0:
        fig, ((ax1, ax2), (ax3, ax4)) = plt.subplots(2, 2, figsize=(16, 12))
        
        assets = performance_stats['asset'].tolist()
        
        # Final returns
        colors = ['green' if x >= 0 else 'red' for x in performance_stats['final_return']]
        bars1 = ax1.bar(assets, performance_stats['final_return'], color=colors, alpha=0.7)
        ax1.set_ylabel('Final Return (%)')
        ax1.set_title('Final Returns by Asset', fontweight='bold')
        ax1.tick_params(axis='x', rotation=45)
        ax1.grid(True, alpha=0.3)
        ax1.axhline(y=0, color='black', linestyle='-', alpha=0.5)
        
        # Add value labels
        for bar, value in zip(bars1, performance_stats['final_return']):
            height = bar.get_height()
            ax1.text(bar.get_x() + bar.get_width()/2., height + (abs(height)*0.05 if height >= 0 else -abs(height)*0.05),
                    f'{value:.1f}%', ha='center', va='bottom' if height >= 0 else 'top', fontweight='bold')
        
        # Max gains
        bars2 = ax2.bar(assets, performance_stats['max_gain'], color='lightgreen', alpha=0.7)
        ax2.set_ylabel('Max Gain (%)')
        ax2.set_title('Maximum Gains by Asset', fontweight='bold')
        ax2.tick_params(axis='x', rotation=45)
        ax2.grid(True, alpha=0.3)
        
        # Profit ratios
        bars3 = ax3.bar(assets, performance_stats['profit_ratio'], color='orange', alpha=0.7)
        ax3.set_ylabel('Profit Ratio (Max Profit / Max Loss)')
        ax3.set_title('Spread Profit Ratios by Asset', fontweight='bold')
        ax3.tick_params(axis='x', rotation=45)
        ax3.grid(True, alpha=0.3)
        
        # BTO Moneyness vs Performance scatter
        ax4.scatter(performance_stats['bto_moneyness'], performance_stats['final_return'], 
                   s=100, alpha=0.7, c=performance_stats['final_return'], 
                   cmap='RdYlGn', vmin=-100, vmax=100)
        ax4.set_xlabel('BTO Moneyness (% from ATM)')
        ax4.set_ylabel('Final Return (%)')
        ax4.set_title('BTO Moneyness vs Final Return', fontweight='bold')
        ax4.grid(True, alpha=0.3)
        ax4.axhline(y=0, color='red', linestyle='--', alpha=0.5)
        
        # Add asset labels to scatter plot
        for _, row in performance_stats.iterrows():
            ax4.annotate(row['asset'], (row['bto_moneyness'], row['final_return']),
                        xytext=(5, 5), textcoords='offset points', fontsize=8)
        
        plt.tight_layout()
        plt.savefig(f"{output_dir}/spread_performance_comparison.png", dpi=300, bbox_inches='tight')
        plt.close()
    
    print("  All spread plots created")

def save_spread_results(spread_contracts, daily_values, individual_dfs, performance_stats, output_dir):
    """
    Save all bull call spread results
    """
    print("\nSaving comprehensive spread results...")
    
    # Save spread contracts details
    spread_contracts.to_csv(f"{output_dir}/selected_spread_contracts.csv", index=False)
    print(f"  Spread contracts: selected_spread_contracts.csv")
    
    # Save daily tracking data
    daily_values.to_csv(f"{output_dir}/daily_basket_values.csv", index=False)
    print(f"  Daily basket values: daily_basket_values.csv")
    
    # Save individual spread tracking
    for asset, df in individual_dfs.items():
        df.to_csv(f"{output_dir}/individual_{asset}_daily.csv", index=False)
    print(f"  Individual daily tracking: individual_[ASSET]_daily.csv")
    
    # Save performance statistics
    performance_stats.to_csv(f"{output_dir}/individual_performance_stats.csv", index=False)
    print(f"  Performance statistics: individual_performance_stats.csv")

def process_single_date(target_date, all_iwls_data, options_base_dir, base_output_dir):
    """
    Process a single date for the bull call spread strategy
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
        
        # Find suitable bull call spreads
        spread_contracts = find_suitable_bull_call_spreads(selected_stocks, target_date, options_base_dir)
        
        if len(spread_contracts) == 0:
            print(f"    No suitable spreads found on {date_str}")
            return False
        
        print(f"    Found {len(spread_contracts)} spreads on {date_str}")
        
        # Track spread basket performance
        daily_values, individual_dfs = track_spread_basket_daily(spread_contracts)
        
        # Calculate individual performance statistics
        performance_stats = calculate_individual_spread_stats(individual_dfs, spread_contracts)
        
        # Calculate final performance
        initial_debit = daily_values.iloc[0]['initial_debit']
        final_pnl = daily_values.iloc[-1]['basket_pnl']
        final_return = daily_values.iloc[-1]['basket_pnl_pct']
        
        # Calculate basket max gain
        max_pnl = daily_values['basket_pnl'].max()
        max_gain = (max_pnl / initial_debit) * 100 if initial_debit > 0 else 0
        max_day = daily_values.loc[daily_values['basket_pnl'].idxmax(), 'day']
        
        print(f"    Final return: {final_return:+.1f}%, Max gain: {max_gain:+.1f}%")
        
        # Create comprehensive plots
        create_spread_plots(daily_values, individual_dfs, performance_stats, spread_contracts, output_dir)
        
        # Save comprehensive results
        save_spread_results(spread_contracts, daily_values, individual_dfs, performance_stats, output_dir)
        
        # Return summary data for aggregation
        return {
            'date': target_date,
            'date_str': date_str,
            'num_spreads': len(spread_contracts),
            'initial_debit': initial_debit,
            'final_pnl': final_pnl,
            'final_return': final_return,
            'max_gain': max_gain,
            'day_of_max_gain': max_day,
            'winners': len(performance_stats[performance_stats['final_return'] > 0]) if len(performance_stats) > 0 else 0,
            'avg_individual_return': performance_stats['final_return'].mean() if len(performance_stats) > 0 else 0,
            'best_individual_return': performance_stats['final_return'].max() if len(performance_stats) > 0 else 0,
            'worst_individual_return': performance_stats['final_return'].min() if len(performance_stats) > 0 else 0,
            'avg_profit_ratio': performance_stats['profit_ratio'].mean() if len(performance_stats) > 0 else 0
        }
    
    except Exception as e:
        print(f"    Error processing {date_str}: {e}")
        return False

def create_overall_summary_analysis(all_results, base_output_dir):
    """
    Create overall summary analysis across all processed dates for bull call spreads
    """
    print("\nCreating overall summary analysis for bull call spreads...")
    
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
        'avg_final_return': results_df['final_return'].mean(),
        'median_final_return': results_df['final_return'].median(),
        'best_final_return': results_df['final_return'].max(),
        'worst_final_return': results_df['final_return'].min(),
        'avg_max_gain': results_df['max_gain'].mean(),
        'median_max_gain': results_df['max_gain'].median(),
        'best_max_gain': results_df['max_gain'].max(),
        'profitable_strategies': len(results_df[results_df['final_return'] > 0]),
        'win_rate': len(results_df[results_df['final_return'] > 0]) / len(results_df) * 100,
        'avg_spreads_per_strategy': results_df['num_spreads'].mean(),
        'total_spreads_analyzed': results_df['num_spreads'].sum(),
        'avg_profit_ratio': results_df['avg_profit_ratio'].mean(),
        'hold_duration_days': HOLD_DURATION
    }
    
    # Save detailed results
    results_df.to_csv(f"{base_output_dir}/ALL_SPREAD_STRATEGIES_SUMMARY.csv", index=False)
    
    # Save summary statistics
    summary_df = pd.DataFrame([summary_stats])
    summary_df.to_csv(f"{base_output_dir}/OVERALL_SPREAD_SUMMARY_STATS.csv", index=False)
    
    # ENHANCED PLOTTING for bull call spreads
    fig, axes = plt.subplots(2, 3, figsize=(20, 12))
    
    # Plot 1: Distribution of final returns
    axes[0,0].hist(results_df['final_return'], bins=30, alpha=0.7, edgecolor='black', color='steelblue')
    axes[0,0].axvline(results_df['final_return'].mean(), color='red', linestyle='--', 
                      label=f'Mean: {results_df["final_return"].mean():.1f}%')
    axes[0,0].axvline(results_df['final_return'].median(), color='orange', linestyle='--', 
                      label=f'Median: {results_df["final_return"].median():.1f}%')
    axes[0,0].axvline(0, color='black', linestyle='-', alpha=0.5, label='Break Even')
    axes[0,0].set_xlabel('Final Return (%)')
    axes[0,0].set_ylabel('Frequency')
    axes[0,0].set_title('Distribution of Final Returns (Bull Call Spreads)', fontweight='bold')
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
    colors = ['green' if x >= 0 else 'red' for x in results_df['final_return']]
    axes[0,2].scatter(results_df['max_gain'], results_df['final_return'], 
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
    axes[1,0].plot(results_df['date'], results_df['final_return'], alpha=0.7, marker='o', markersize=3, color='blue')
    axes[1,0].axhline(0, color='red', linestyle='--', alpha=0.5, label='Break Even')
    axes[1,0].axhline(results_df['final_return'].mean(), color='green', linestyle='--', alpha=0.7, 
                      label=f'Average: {results_df["final_return"].mean():.1f}%')
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
        'final_return': lambda x: (x > 0).mean() * 100,
        'date': 'count',
        'max_gain': 'mean'
    }).rename(columns={'final_return': 'win_rate', 'date': 'count', 'max_gain': 'avg_max_gain'})
    
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
    plt.savefig(f"{base_output_dir}/OVERALL_SPREAD_PERFORMANCE_SUMMARY.png", dpi=300, bbox_inches='tight')
    plt.close()
    
    # NEW: Create separate detailed charts for bull call spreads
    fig, ((ax1, ax2), (ax3, ax4)) = plt.subplots(2, 2, figsize=(16, 12))
    
    # Chart 1: Final returns by trading session (chronological)
    colors1 = ['green' if x >= 0 else 'red' for x in results_df['final_return']]
    bars1 = ax1.bar(range(len(results_df)), results_df['final_return'], color=colors1, alpha=0.7)
    ax1.set_xlabel('Trading Session (Chronological)')
    ax1.set_ylabel('Final Return (%)')
    ax1.set_title('Final Returns by Trading Session (Bull Call Spreads)', fontweight='bold')
    ax1.axhline(y=0, color='black', linestyle='-', alpha=0.5)
    ax1.axhline(y=results_df['final_return'].mean(), color='blue', linestyle='--', alpha=0.7,
                label=f'Mean: {results_df["final_return"].mean():.1f}%')
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
    sorted_returns = results_df['final_return'].sort_values(ascending=False).reset_index(drop=True)
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
    plt.savefig(f"{base_output_dir}/SPREAD_TRADING_SESSIONS_DETAILED.png", dpi=300, bbox_inches='tight')
    plt.close()
    
    # Print summary
    print(f"\n" + "="*80)
    print("OVERALL BULL CALL SPREAD STRATEGY PERFORMANCE SUMMARY")
    print("="*80)
    print(f"STRATEGY PARAMETERS:")
    print(f"  BTO range: {BTO_MIN_MONEYNESS}% to {BTO_MAX_MONEYNESS}% from ATM")
    print(f"  STO target: {STO_TARGET_MONEYNESS}% OTM (±{STO_TOLERANCE}%)")
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
    print(f"  Average: {summary_stats['avg_final_return']:.2f}%")
    print(f"  Median: {summary_stats['median_final_return']:.2f}%")
    print(f"  Best: {summary_stats['best_final_return']:.2f}%")
    print(f"  Worst: {summary_stats['worst_final_return']:.2f}%")
    print(f"\nMAXIMUM GAINS:")
    print(f"  Average: {summary_stats['avg_max_gain']:.2f}%")
    print(f"  Median: {summary_stats['median_max_gain']:.2f}%")
    print(f"  Best: {summary_stats['best_max_gain']:.2f}%")
    print(f"\nSPREAD STATS:")
    print(f"  Total spreads analyzed: {summary_stats['total_spreads_analyzed']:,}")
    print(f"  Average spreads per strategy: {summary_stats['avg_spreads_per_strategy']:.1f}")
    print(f"  Average profit ratio: {summary_stats['avg_profit_ratio']:.2f}")

def main():
    print("BULL CALL SPREAD STRATEGY BASED ON IWLS DEVIATION SIGNALS")
    print("=" * 80)
    print("CURRENT STRATEGY PARAMETERS:")
    print(f"  Stock Selection: Deviation < {MIN_DEVIATION}% (most underperforming {NUM_STOCKS} stocks)")
    print(f"  Bull Call Spreads:")
    print(f"    BTO (Buy): {BTO_MIN_MONEYNESS}% to {BTO_MAX_MONEYNESS}% from ATM")
    print(f"    STO (Sell): {STO_TARGET_MONEYNESS}% OTM (±{STO_TOLERANCE}%)")
    print(f"  DTE Range: {MIN_DTE}-{MAX_DTE} days")
    print(f"  Hold Duration: {HOLD_DURATION} days")
    print(f"  Test Period: {START_DATE.strftime('%Y-%m-%d')} to {min(END_DATE, EFFECTIVE_END_DATE).strftime('%Y-%m-%d')}")
    print(f"  Test Frequency: Every {TEST_FREQUENCY} days")
    print(f"  Data Cutoff: {DATA_CUTOFF.strftime('%Y-%m-%d')} (effective end: {EFFECTIVE_END_DATE.strftime('%Y-%m-%d')})")
    
    # Setup directories with correct paths
    v2_dir = "/Users/tim/CODE_PROJECTS/IWLS-OPTIONS/IWLS_ANALYSIS_V2"
    options_base_dir = "/Users/tim/CODE_PROJECTS/IWLS-OPTIONS/OPTIONS_DATASET"
    
    # Create output directory name based on parameters
    params_str = f"SPREADS_BTO{BTO_MIN_MONEYNESS}to{BTO_MAX_MONEYNESS}_STO{STO_TARGET_MONEYNESS}_DTE{MIN_DTE}-{MAX_DTE}_HOLD{HOLD_DURATION}"
    base_output_dir = f"/Users/tim/CODE_PROJECTS/IWLS-OPTIONS/IWLS_ANALYSIS_V2/BULL_CALL_SPREAD_{params_str}"
    
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
        
        print(f"\nFull bull call spread backtest complete!")
        print(f"Individual strategy results: {base_output_dir}/[YYYYMMDD]/")
        print(f"Overall summary: {base_output_dir}/ALL_SPREAD_STRATEGIES_SUMMARY.csv")
        print(f"Performance charts: {base_output_dir}/OVERALL_SPREAD_PERFORMANCE_SUMMARY.png")
        print(f"Detailed trading sessions: {base_output_dir}/SPREAD_TRADING_SESSIONS_DETAILED.png")
    else:
        print("ERROR: No successful strategies found in the entire backtest period")
        print("Try adjusting parameters:")
        print(f"  - Widen BTO range (currently {BTO_MIN_MONEYNESS}% to {BTO_MAX_MONEYNESS}%)")
        print(f"  - Adjust STO target/tolerance (currently {STO_TARGET_MONEYNESS}% ±{STO_TOLERANCE}%)")
        print(f"  - Increase DTE range (currently {MIN_DTE}-{MAX_DTE})")
        print(f"  - Reduce deviation threshold (currently {MIN_DEVIATION}%)")
        print(f"  - Reduce hold duration (currently {HOLD_DURATION} days)")
    
    print(f"\nResults saved to: {base_output_dir}")

if __name__ == "__main__":
    main()