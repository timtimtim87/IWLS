import pandas as pd
import numpy as np
import matplotlib.pyplot as plt
from datetime import datetime, timedelta
import warnings
import os
import glob

warnings.filterwarnings('ignore')

def find_next_valid_entry_date(all_iwls_data, start_date, max_days_forward=7):
    """
    Find the next date with valid IWLS data for entry, skipping weekends and data gaps
    """
    current_date = start_date
    
    for _ in range(max_days_forward):
        # Skip weekends
        if current_date.weekday() < 5:  # Monday=0, Friday=4
            # Check if we have IWLS data on this date
            available_assets = get_available_assets_on_date(all_iwls_data, current_date)
            if len(available_assets) > 0:
                return current_date
        
        current_date += timedelta(days=1)
    
    return None  # No valid date found within max_days_forward

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
        # Fallback: take most underperforming stocks available
        underperforming = available_df[available_df['price_deviation'] < 0].copy()
    
    if len(underperforming) == 0:
        return pd.DataFrame()
    
    # Sort by most negative deviation and take top stocks
    sorted_stocks = underperforming.sort_values('price_deviation', ascending=True)
    selected_stocks = sorted_stocks.head(num_stocks)
    
    return selected_stocks

def find_suitable_options(selected_stocks, entry_date, options_base_dir, min_dte=300, max_dte=400):
    """
    Find suitable call options for selected stocks from actual options data
    """
    suitable_options = []
    
    for _, stock in selected_stocks.iterrows():
        asset = stock['asset']
        stock_price = stock['price']
        
        # Look for options directories for this asset
        asset_options_dir = os.path.join(options_base_dir, asset)
        
        if not os.path.exists(asset_options_dir):
            continue
        
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
                            continue
            
            except Exception as e:
                continue
        
        if best_option:
            suitable_options.append(best_option)
    
    return pd.DataFrame(suitable_options)

def calculate_basket_value_on_date(option_contracts, check_date):
    """
    Calculate the total basket value on a specific date using last available price if needed
    """
    total_value = 0
    individual_values = []
    
    for _, contract in option_contracts.iterrows():
        option_file = contract['option_file']
        contracts_held = contract['contracts_held']
        expiration_date = contract['expiration_date']
        
        # Check if option has expired
        if check_date > expiration_date:
            # Option expired, value is 0
            individual_values.append({
                'asset': contract['asset'],
                'current_price': 0,
                'contracts_held': contracts_held,
                'contract_value': 0,
                'status': 'expired'
            })
            continue
        
        try:
            # Load the option data
            option_df = pd.read_csv(option_file)
            option_df['date'] = pd.to_datetime(option_df['date'])
            
            # Find price data around check date (within 10 days)
            price_mask = (option_df['date'] >= check_date - timedelta(days=10)) & \
                        (option_df['date'] <= check_date + timedelta(days=10))
            price_data = option_df[price_mask]
            
            current_price = 0
            status = 'current'
            
            if len(price_data) > 0:
                # Get closest price to check date
                closest_idx = (price_data['date'] - check_date).abs().idxmin()
                closest_row = price_data.loc[closest_idx]
                current_price = closest_row['close']
                status = 'current'
            else:
                # No recent data, use the last available price from any date
                past_data = option_df[option_df['date'] <= check_date]
                if len(past_data) > 0:
                    current_price = past_data.iloc[-1]['close']
                    status = 'last_available'
                else:
                    # No data at all, assume worthless
                    current_price = 0
                    status = 'no_data'
            
            contract_value = current_price * contracts_held * 100  # Options are per 100 shares
            total_value += contract_value
            
            individual_values.append({
                'asset': contract['asset'],
                'current_price': current_price,
                'contracts_held': contracts_held,
                'contract_value': contract_value,
                'status': status
            })
            
        except Exception as e:
            # If we can't get current price, assume worthless
            individual_values.append({
                'asset': contract['asset'],
                'current_price': 0,
                'contracts_held': contracts_held,
                'contract_value': 0,
                'status': 'error'
            })
    
    return total_value, individual_values

def run_sequential_trading_strategy(all_iwls_data, options_base_dir, start_date, end_date, 
                                  initial_capital=100000, position_size=0.5, 
                                  profit_target=None, stop_loss=None, max_hold_days=None):
    """
    Run sequential trading strategy with configurable exit criteria
    
    Parameters:
    - profit_target: Float (e.g., 50.0 for 50%) or None to disable
    - stop_loss: Float (e.g., -25.0 for -25%) or None to disable  
    - max_hold_days: Int (e.g., 180) or None to disable
    """
    print(f"\nRunning sequential trading strategy:")
    print(f"  Start date: {start_date.strftime('%Y-%m-%d')}")
    print(f"  End date: {end_date.strftime('%Y-%m-%d')}")
    print(f"  Initial capital: ${initial_capital:,}")
    print(f"  Position size: {position_size*100}% of account")
    print(f"  Profit target: {'DISABLED' if profit_target is None else f'{profit_target}%'}")
    print(f"  Stop loss: {'DISABLED' if stop_loss is None else f'{stop_loss}%'}")
    print(f"  Max hold days: {'DISABLED' if max_hold_days is None else f'{max_hold_days} days'}")
    
    # Initialize account
    current_capital = initial_capital
    current_date = start_date
    
    # Tracking data
    daily_account_values = []
    all_trades = []
    active_positions = None
    trade_id = 0
    
    while current_date <= end_date:
        # Skip weekends
        if current_date.weekday() >= 5:
            current_date += timedelta(days=1)
            continue
        
        account_value = current_capital
        position_value = 0
        
        # Check if we have active positions
        if active_positions is not None:
            # Calculate current position value
            position_value, individual_values = calculate_basket_value_on_date(active_positions, current_date)
            account_value = current_capital + position_value
            
            # Calculate return on position
            initial_position_value = active_positions['initial_position_value'].iloc[0]
            position_return = ((position_value / initial_position_value) - 1) * 100
            
            # Check days held
            entry_date = active_positions['entry_date'].iloc[0]
            days_held = (current_date - entry_date).days
            
            # Check exit conditions
            should_exit = False
            exit_reason = ""
            
            # Check profit target
            if profit_target is not None and position_return >= profit_target:
                should_exit = True
                exit_reason = f'profit_target_{profit_target}%'
            
            # Check stop loss
            elif stop_loss is not None and position_return <= stop_loss:
                should_exit = True
                exit_reason = f'stop_loss_{stop_loss}%'
            
            # Check max hold time
            elif max_hold_days is not None and days_held >= max_hold_days:
                should_exit = True
                exit_reason = f'max_hold_{max_hold_days}_days'
            
            # If no exit criteria are set, hold until expiration (emergency exit after 400 days)
            elif profit_target is None and stop_loss is None and max_hold_days is None:
                if days_held >= 400:  # Emergency exit to prevent infinite holding
                    should_exit = True
                    exit_reason = "emergency_exit_400_days"
            
            if should_exit:
                # Close position and realize profits/losses
                profit = position_value - initial_position_value
                current_capital += position_value
                
                # Record the trade
                trade_record = {
                    'trade_id': trade_id,
                    'entry_date': entry_date,
                    'exit_date': current_date,
                    'days_held': days_held,
                    'exit_reason': exit_reason,
                    'initial_investment': initial_position_value,
                    'exit_value': position_value,
                    'profit_loss': profit,
                    'return_pct': position_return,
                    'num_contracts': len(active_positions),
                    'contract_details': active_positions.to_dict('records')
                }
                
                all_trades.append(trade_record)
                print(f"  {current_date.strftime('%Y-%m-%d')}: CLOSED position #{trade_id} - "
                      f"{position_return:.1f}% return in {days_held} days, "
                      f"Profit: ${profit:,.0f}, Reason: {exit_reason}, New capital: ${current_capital:,.0f}")
                
                active_positions = None
                trade_id += 1
                account_value = current_capital  # Update account value after closing
        
        # Record daily account value
        daily_account_values.append({
            'date': current_date,
            'account_value': account_value,
            'cash': current_capital,
            'position_value': position_value,
            'total_return_pct': ((account_value / initial_capital) - 1) * 100,
            'has_position': active_positions is not None,
            'days_in_position': (current_date - active_positions['entry_date'].iloc[0]).days if active_positions is not None else 0
        })
        
        # Try to enter new position if we don't have one and it's not the last few days
        if active_positions is None and current_date <= end_date - timedelta(days=30):
            # Find next valid entry date (skip weekends and missing data)
            valid_entry_date = find_next_valid_entry_date(all_iwls_data, current_date)
            
            if valid_entry_date and valid_entry_date <= end_date - timedelta(days=30):
                # Get available assets on valid entry date
                available_assets = get_available_assets_on_date(all_iwls_data, valid_entry_date)
                
                if len(available_assets) > 0:
                    # Select underperforming stocks
                    selected_stocks = select_underperforming_stocks(available_assets, min_deviation=-15, num_stocks=10)
                    
                    if len(selected_stocks) > 0:
                        # Find suitable options
                        option_contracts = find_suitable_options(
                            selected_stocks, valid_entry_date, options_base_dir, min_dte=90, max_dte=600
                        )
                        
                        if len(option_contracts) > 0:
                            # Calculate position size
                            position_capital = current_capital * position_size
                            
                            # Calculate how many contracts to buy for each option
                            cost_per_contract_set = (option_contracts['entry_option_price'] * 100).sum()  # Total cost for one of each
                            
                            if cost_per_contract_set > 0:
                                num_contract_sets = position_capital / cost_per_contract_set
                                
                                # Add contract quantities to the dataframe
                                option_contracts['contracts_held'] = num_contract_sets
                                option_contracts['initial_position_value'] = position_capital
                                option_contracts['entry_date'] = valid_entry_date
                                
                                # Subtract position capital from cash
                                current_capital -= position_capital
                                active_positions = option_contracts
                                
                                print(f"  {valid_entry_date.strftime('%Y-%m-%d')}: OPENED position #{trade_id} - "
                                      f"${position_capital:,.0f} invested in {len(option_contracts)} options, "
                                      f"{num_contract_sets:.2f} contracts each")
                                
                                # Skip forward to the entry date if it's different from current_date
                                if valid_entry_date > current_date:
                                    current_date = valid_entry_date - timedelta(days=1)  # Will be incremented at end of loop
        
        current_date += timedelta(days=1)
    
    # Close any remaining positions at the end
    if active_positions is not None:
        final_position_value, _ = calculate_basket_value_on_date(active_positions, end_date)
        initial_position_value = active_positions['initial_position_value'].iloc[0]
        current_capital += final_position_value
        
        # Record the final trade
        entry_date = active_positions['entry_date'].iloc[0]
        days_held = (end_date - entry_date).days
        profit = final_position_value - initial_position_value
        position_return = ((final_position_value / initial_position_value) - 1) * 100
        
        trade_record = {
            'trade_id': trade_id,
            'entry_date': entry_date,
            'exit_date': end_date,
            'days_held': days_held,
            'exit_reason': 'strategy_end',
            'initial_investment': initial_position_value,
            'exit_value': final_position_value,
            'profit_loss': profit,
            'return_pct': position_return,
            'num_contracts': len(active_positions),
            'contract_details': active_positions.to_dict('records')
        }
        
        all_trades.append(trade_record)
        print(f"  {end_date.strftime('%Y-%m-%d')}: FINAL CLOSE - "
              f"{position_return:.1f}% return in {days_held} days")
    
    return pd.DataFrame(daily_account_values), pd.DataFrame(all_trades), current_capital

def create_sequential_strategy_plots(daily_values, trades_df, initial_capital, final_capital, 
                                   strategy_params, output_dir):
    """
    Create comprehensive plots for the sequential strategy with parameter info
    """
    print("\nCreating sequential strategy performance plots...")
    
    # Create parameter string for titles
    param_str = f"PT: {strategy_params.get('profit_target', 'OFF')}%, " \
                f"SL: {strategy_params.get('stop_loss', 'OFF')}%, " \
                f"Max: {strategy_params.get('max_hold_days', 'OFF')} days"
    
    # Figure 1: Account performance over time
    fig, ((ax1, ax2), (ax3, ax4)) = plt.subplots(2, 2, figsize=(18, 12))
    
    # Plot 1: Account value over time
    ax1.plot(daily_values['date'], daily_values['account_value'], linewidth=2, color='blue', label='Total Account')
    ax1.plot(daily_values['date'], daily_values['cash'], linewidth=1, color='green', alpha=0.7, label='Cash')
    ax1.plot(daily_values['date'], daily_values['position_value'], linewidth=1, color='red', alpha=0.7, label='Position Value')
    
    ax1.axhline(y=initial_capital, color='black', linestyle='--', alpha=0.5, label=f'Initial: ${initial_capital:,}')
    ax1.set_xlabel('Date')
    ax1.set_ylabel('Value ($)')
    ax1.set_title(f'Account Value Over Time ({param_str})', fontweight='bold')
    ax1.legend()
    ax1.grid(True, alpha=0.3)
    ax1.tick_params(axis='x', rotation=45)
    
    # Plot 2: Return percentage over time
    ax2.plot(daily_values['date'], daily_values['total_return_pct'], linewidth=2, color='green')
    ax2.axhline(y=0, color='red', linestyle='--', alpha=0.5, label='Break Even')
    ax2.set_xlabel('Date')
    ax2.set_ylabel('Total Return (%)')
    ax2.set_title(f'Cumulative Return Percentage ({param_str})', fontweight='bold')
    ax2.legend()
    ax2.grid(True, alpha=0.3)
    ax2.tick_params(axis='x', rotation=45)
    
    # Plot 3: Individual trade returns
    if len(trades_df) > 0:
        colors = ['green' if x >= 0 else 'red' for x in trades_df['return_pct']]
        bars = ax3.bar(range(len(trades_df)), trades_df['return_pct'], color=colors, alpha=0.7)
        ax3.axhline(y=0, color='black', linestyle='-', alpha=0.5)
        
        # Add target and stop loss lines if they exist
        if strategy_params.get('profit_target') is not None:
            ax3.axhline(y=strategy_params['profit_target'], color='blue', linestyle='--', alpha=0.7, 
                       label=f"PT: {strategy_params['profit_target']}%")
        if strategy_params.get('stop_loss') is not None:
            ax3.axhline(y=strategy_params['stop_loss'], color='orange', linestyle='--', alpha=0.7, 
                       label=f"SL: {strategy_params['stop_loss']}%")
        
        ax3.set_xlabel('Trade Number')
        ax3.set_ylabel('Trade Return (%)')
        ax3.set_title('Individual Trade Returns', fontweight='bold')
        if strategy_params.get('profit_target') is not None or strategy_params.get('stop_loss') is not None:
            ax3.legend()
        ax3.grid(True, alpha=0.3)
        
        # Add value labels on bars
        for i, (bar, value) in enumerate(zip(bars, trades_df['return_pct'])):
            height = bar.get_height()
            ax3.text(bar.get_x() + bar.get_width()/2., height + (abs(height)*0.05 if height >= 0 else -abs(height)*0.05),
                    f'{value:.1f}%', ha='center', va='bottom' if height >= 0 else 'top', fontsize=8, fontweight='bold')
    
    # Plot 4: Trade duration vs returns
    if len(trades_df) > 0:
        colors = ['green' if x >= 0 else 'red' for x in trades_df['return_pct']]
        scatter = ax4.scatter(trades_df['days_held'], trades_df['return_pct'], c=colors, alpha=0.7, s=60)
        ax4.axhline(y=0, color='black', linestyle='-', alpha=0.5)
        
        # Add target, stop loss, and max hold lines
        if strategy_params.get('profit_target') is not None:
            ax4.axhline(y=strategy_params['profit_target'], color='blue', linestyle='--', alpha=0.7, 
                       label=f"PT: {strategy_params['profit_target']}%")
        if strategy_params.get('stop_loss') is not None:
            ax4.axhline(y=strategy_params['stop_loss'], color='orange', linestyle='--', alpha=0.7, 
                       label=f"SL: {strategy_params['stop_loss']}%")
        if strategy_params.get('max_hold_days') is not None:
            ax4.axvline(x=strategy_params['max_hold_days'], color='purple', linestyle='--', alpha=0.7, 
                       label=f"Max: {strategy_params['max_hold_days']}d")
        
        ax4.set_xlabel('Days Held')
        ax4.set_ylabel('Trade Return (%)')
        ax4.set_title('Trade Duration vs Return', fontweight='bold')
        if any(strategy_params.get(k) is not None for k in ['profit_target', 'stop_loss', 'max_hold_days']):
            ax4.legend()
        ax4.grid(True, alpha=0.3)
        
        # Add trade numbers as labels
        for i, (_, trade) in enumerate(trades_df.iterrows()):
            ax4.annotate(f'T{i+1}', (trade['days_held'], trade['return_pct']),
                        xytext=(3, 3), textcoords='offset points', fontsize=8)
    
    plt.tight_layout()
    plt.savefig(f"{output_dir}/sequential_strategy_performance.png", dpi=300, bbox_inches='tight')
    plt.close()
    
    print("  ✅ Performance plots created")

def save_sequential_results(daily_values, trades_df, initial_capital, final_capital, 
                          strategy_params, output_dir):
    """
    Save all sequential strategy results including strategy parameters
    """
    print("\nSaving sequential strategy results...")
    
    # Save daily account values
    daily_values.to_csv(f"{output_dir}/daily_account_values.csv", index=False)
    print(f"  ✅ Daily values: daily_account_values.csv")
    
    # Save all trades
    trades_df.to_csv(f"{output_dir}/all_trades.csv", index=False)
    print(f"  ✅ All trades: all_trades.csv")
    
    # Create and save summary statistics
    if len(trades_df) > 0:
        total_return = ((final_capital / initial_capital) - 1) * 100
        winning_trades = len(trades_df[trades_df['return_pct'] > 0])
        losing_trades = len(trades_df[trades_df['return_pct'] <= 0])
        win_rate = (winning_trades / len(trades_df)) * 100
        avg_trade_return = trades_df['return_pct'].mean()
        avg_winning_trade = trades_df[trades_df['return_pct'] > 0]['return_pct'].mean() if winning_trades > 0 else 0
        avg_losing_trade = trades_df[trades_df['return_pct'] <= 0]['return_pct'].mean() if losing_trades > 0 else 0
        avg_hold_time = trades_df['days_held'].mean()
        
        # Calculate exit reason statistics
        exit_reasons = trades_df['exit_reason'].value_counts()
        
        summary_stats = {
            'initial_capital': initial_capital,
            'final_capital': final_capital,
            'total_return_pct': total_return,
            'profit_target': strategy_params.get('profit_target'),
            'stop_loss': strategy_params.get('stop_loss'),
            'max_hold_days': strategy_params.get('max_hold_days'),
            'position_size': strategy_params.get('position_size'),
            'total_trades': len(trades_df),
            'winning_trades': winning_trades,
            'losing_trades': losing_trades,
            'win_rate_pct': win_rate,
            'avg_trade_return_pct': avg_trade_return,
            'avg_winning_trade_pct': avg_winning_trade,
            'avg_losing_trade_pct': avg_losing_trade,
            'avg_hold_time_days': avg_hold_time,
            'best_trade_pct': trades_df['return_pct'].max(),
            'worst_trade_pct': trades_df['return_pct'].min(),
            'total_profit_loss': final_capital - initial_capital
        }
        
        # Add exit reason counts to summary
        for reason, count in exit_reasons.items():
            summary_stats[f'exits_{reason}'] = count
            summary_stats[f'exits_{reason}_pct'] = (count / len(trades_df)) * 100
        
        summary_df = pd.DataFrame([summary_stats])
        summary_df.to_csv(f"{output_dir}/strategy_summary.csv", index=False)
        print(f"  ✅ Summary stats: strategy_summary.csv")
        
        # Save exit reasons breakdown
        exit_reasons_df = exit_reasons.reset_index()
        exit_reasons_df.columns = ['exit_reason', 'count']
        exit_reasons_df['percentage'] = (exit_reasons_df['count'] / len(trades_df)) * 100
        exit_reasons_df.to_csv(f"{output_dir}/exit_reasons_breakdown.csv", index=False)
        print(f"  ✅ Exit reasons: exit_reasons_breakdown.csv")

def run_strategy_comparison(all_iwls_data, options_base_dir, start_date, end_date, 
                           initial_capital, position_size, strategy_configs, base_output_dir):
    """
    Run multiple strategy configurations and compare results
    """
    print(f"\n" + "="*80)
    print("RUNNING STRATEGY COMPARISON")
    print("="*80)
    
    comparison_results = []
    
    for i, config in enumerate(strategy_configs):
        print(f"\n[Configuration {i+1}/{len(strategy_configs)}]")
        print(f"Profit Target: {config.get('profit_target', 'DISABLED')}")
        print(f"Stop Loss: {config.get('stop_loss', 'DISABLED')}")
        print(f"Max Hold Days: {config.get('max_hold_days', 'DISABLED')}")
        
        # Create output directory for this configuration
        config_name = f"PT{config.get('profit_target', 'OFF')}_SL{config.get('stop_loss', 'OFF')}_MH{config.get('max_hold_days', 'OFF')}"
        config_output_dir = os.path.join(base_output_dir, config_name)
        os.makedirs(config_output_dir, exist_ok=True)
        
        try:
            # Run strategy with this configuration
            daily_values, trades_df, final_capital = run_sequential_trading_strategy(
                all_iwls_data, options_base_dir, start_date, end_date,
                initial_capital=initial_capital, position_size=position_size,
                profit_target=config.get('profit_target'),
                stop_loss=config.get('stop_loss'),
                max_hold_days=config.get('max_hold_days')
            )
            
            # Calculate performance metrics
            total_return = ((final_capital / initial_capital) - 1) * 100
            
            # Create plots
            create_sequential_strategy_plots(daily_values, trades_df, initial_capital, 
                                           final_capital, config, config_output_dir)
            
            # Save results
            save_sequential_results(daily_values, trades_df, initial_capital, 
                                  final_capital, config, config_output_dir)
            
            # Collect summary for comparison
            if len(trades_df) > 0:
                winning_trades = len(trades_df[trades_df['return_pct'] > 0])
                win_rate = (winning_trades / len(trades_df)) * 100
                avg_trade_return = trades_df['return_pct'].mean()
                avg_hold_time = trades_df['days_held'].mean()
                
                # Count exit reasons
                exit_reasons = trades_df['exit_reason'].value_counts()
                profit_exits = exit_reasons.get(f"profit_target_{config.get('profit_target')}%", 0) if config.get('profit_target') else 0
                stop_exits = exit_reasons.get(f"stop_loss_{config.get('stop_loss')}%", 0) if config.get('stop_loss') else 0
                max_hold_exits = exit_reasons.get(f"max_hold_{config.get('max_hold_days')}_days", 0) if config.get('max_hold_days') else 0
                
                comparison_results.append({
                    'config_name': config_name,
                    'profit_target': config.get('profit_target'),
                    'stop_loss': config.get('stop_loss'),
                    'max_hold_days': config.get('max_hold_days'),
                    'total_return_pct': total_return,
                    'final_capital': final_capital,
                    'total_trades': len(trades_df),
                    'win_rate_pct': win_rate,
                    'avg_trade_return_pct': avg_trade_return,
                    'avg_hold_time_days': avg_hold_time,
                    'best_trade_pct': trades_df['return_pct'].max(),
                    'worst_trade_pct': trades_df['return_pct'].min(),
                    'profit_target_exits': profit_exits,
                    'stop_loss_exits': stop_exits,
                    'max_hold_exits': max_hold_exits,
                    'profit_target_hit_rate': (profit_exits / len(trades_df)) * 100 if len(trades_df) > 0 else 0,
                    'stop_loss_hit_rate': (stop_exits / len(trades_df)) * 100 if len(trades_df) > 0 else 0
                })
            
            print(f"✅ Configuration completed: {total_return:+.2f}% return, {len(trades_df)} trades")
            
        except Exception as e:
            print(f"❌ Configuration failed: {e}")
            comparison_results.append({
                'config_name': config_name,
                'profit_target': config.get('profit_target'),
                'stop_loss': config.get('stop_loss'),
                'max_hold_days': config.get('max_hold_days'),
                'total_return_pct': np.nan,
                'error': str(e)
            })
    
    # Save comparison results
    if comparison_results:
        comparison_df = pd.DataFrame(comparison_results)
        comparison_df.to_csv(f"{base_output_dir}/strategy_comparison.csv", index=False)
        print(f"\n✅ Comparison results saved to: {base_output_dir}/strategy_comparison.csv")
        
        # Print comparison summary
        print(f"\n" + "="*80)
        print("STRATEGY COMPARISON SUMMARY")
        print("="*80)
        
        valid_results = comparison_df.dropna(subset=['total_return_pct'])
        if len(valid_results) > 0:
            # Sort by total return
            sorted_results = valid_results.sort_values('total_return_pct', ascending=False)
            
            print(f"\nRANKING BY TOTAL RETURN:")
            print("-" * 120)
            print(f"{'Rank':<4} {'Config':<20} {'PT':<6} {'SL':<6} {'MaxH':<6} {'Return':<8} {'Trades':<7} {'WinRate':<8} {'AvgTrade':<9} {'AvgHold':<8}")
            print("-" * 120)
            
            for i, (_, row) in enumerate(sorted_results.iterrows()):
                pt = f"{row['profit_target']}%" if pd.notna(row['profit_target']) else "OFF"
                sl = f"{row['stop_loss']}%" if pd.notna(row['stop_loss']) else "OFF"
                mh = f"{row['max_hold_days']}d" if pd.notna(row['max_hold_days']) else "OFF"
                
                print(f"{i+1:<4} {row['config_name']:<20} {pt:<6} {sl:<6} {mh:<6} "
                      f"{row['total_return_pct']:>7.2f}% {row['total_trades']:<7.0f} "
                      f"{row['win_rate_pct']:>7.1f}% {row['avg_trade_return_pct']:>8.2f}% {row['avg_hold_time_days']:>7.1f}")
            
            # Best strategy details
            best_strategy = sorted_results.iloc[0]
            print(f"\n🏆 BEST STRATEGY: {best_strategy['config_name']}")
            print(f"   Total Return: {best_strategy['total_return_pct']:+.2f}%")
            print(f"   Final Capital: ${best_strategy['final_capital']:,.0f}")
            print(f"   Win Rate: {best_strategy['win_rate_pct']:.1f}%")
            print(f"   Average Trade: {best_strategy['avg_trade_return_pct']:+.2f}%")
            print(f"   Average Hold Time: {best_strategy['avg_hold_time_days']:.1f} days")
            
            if pd.notna(best_strategy['profit_target']):
                print(f"   Profit Target Hit Rate: {best_strategy['profit_target_hit_rate']:.1f}%")
            if pd.notna(best_strategy['stop_loss']):
                print(f"   Stop Loss Hit Rate: {best_strategy['stop_loss_hit_rate']:.1f}%")

def main():
    print("CONFIGURABLE SEQUENTIAL OPTIONS TRADING STRATEGY")
    print("=" * 80)
    print("Strategy: IWLS deviation signals with configurable exit criteria")
    print("Features: Adjustable profit targets, stop losses, and max hold periods")
    
    # Setup directories
    v2_dir = "/Users/tim/CODE_PROJECTS/IWLS-OPTIONS/IWLS_ANALYSIS_V2"
    options_base_dir = "/Users/tim/CODE_PROJECTS/IWLS-OPTIONS/OPTIONS_DATASET"
    base_output_dir = "/Users/tim/CODE_PROJECTS/IWLS-OPTIONS/IWLS_ANALYSIS_V2/CONFIGURABLE_SEQUENTIAL_STRATEGY"
    
    os.makedirs(base_output_dir, exist_ok=True)
    
    if not os.path.exists(v2_dir):
        print(f"❌ IWLS_ANALYSIS_V2 directory not found: {v2_dir}")
        return
    
    if not os.path.exists(options_base_dir):
        print(f"❌ OPTIONS_DATASET directory not found: {options_base_dir}")
        return
    
    # Load IWLS data once
    all_iwls_data = load_all_iwls_data(v2_dir)
    
    if len(all_iwls_data) == 0:
        print("❌ No IWLS data loaded.")
        return
    
    # Strategy parameters
    start_date = datetime(2022, 5, 1)
    end_date = datetime(2024, 5, 9)
    initial_capital = 100000
    position_size = 0.5
    
    # Define strategy configurations to test
    strategy_configs = [
        # Original 50% profit target only
        {'profit_target': 50.0, 'stop_loss': None, 'max_hold_days': None},
        
        # Profit target with stop loss
        {'profit_target': 50.0, 'stop_loss': -50.0, 'max_hold_days': None},
        {'profit_target': 60.0, 'stop_loss': -60.0, 'max_hold_days': None},
        {'profit_target': 70.0, 'stop_loss': -70.0, 'max_hold_days': None},
        
        # Max hold only (no targets)
        {'profit_target': None, 'stop_loss': None, 'max_hold_days': 100},
        {'profit_target': None, 'stop_loss': None, 'max_hold_days': 300},
        {'profit_target': None, 'stop_loss': None, 'max_hold_days': 500},
        
        # Combination strategies
        #{'profit_target': 40.0, 'stop_loss': -25.0, 'max_hold_days': 240},
        #{'profit_target': 30.0, 'stop_loss': -20.0, 'max_hold_days': 180},
        #{'profit_target': 25.0, 'stop_loss': -15.0, 'max_hold_days': 150},
        
        # Conservative approach
        #{'profit_target': 20.0, 'stop_loss': -10.0, 'max_hold_days': 120},
        
        # Aggressive approach  
        #{'profit_target': 75.0, 'stop_loss': -40.0, 'max_hold_days': 360},
    ]
    
    print(f"\nTesting {len(strategy_configs)} different configurations:")
    for i, config in enumerate(strategy_configs):
        pt = f"{config.get('profit_target')}%" if config.get('profit_target') else "OFF"
        sl = f"{config.get('stop_loss')}%" if config.get('stop_loss') else "OFF"  
        mh = f"{config.get('max_hold_days')} days" if config.get('max_hold_days') else "OFF"
        print(f"  {i+1}. PT: {pt}, SL: {sl}, Max Hold: {mh}")
    
    # Run strategy comparison
    run_strategy_comparison(
        all_iwls_data, options_base_dir, start_date, end_date,
        initial_capital, position_size, strategy_configs, base_output_dir
    )
    
    print(f"\n✅ All strategy configurations completed!")
    print(f"📁 Results saved to: {base_output_dir}")
    print(f"📊 Individual config folders: [CONFIG_NAME]/")
    print(f"📄 Comparison summary: strategy_comparison.csv")
    
    print(f"\n💡 To run a single custom configuration, modify the strategy_configs list")
    print(f"   Example single config:")
    print(f"   strategy_configs = [")
    print(f"       {{'profit_target': 35.0, 'stop_loss': -20.0, 'max_hold_days': 200}}")
    print(f"   ]")

if __name__ == "__main__":
    main()