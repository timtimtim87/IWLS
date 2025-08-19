import pandas as pd
import numpy as np
import os
import glob
from datetime import datetime, timedelta
import warnings
warnings.filterwarnings('ignore')

def load_entry_signals():
    """
    Load all entry signal CSV files
    """
    signal_files = glob.glob("/Users/tim/IWLS-OPTIONS/ENTRY_SIGNALS/entry_signals_*.csv")
    
    if not signal_files:
        print("No entry signal files found!")
        return {}
    
    entry_signals = {}
    
    for file_path in signal_files:
        bin_name = os.path.basename(file_path).replace('entry_signals_', '').replace('.csv', '')
        df = pd.read_csv(file_path)
        df['date'] = pd.to_datetime(df['date'])
        entry_signals[bin_name] = df
        print(f"Loaded {bin_name}: {len(df)} signals")
    
    return entry_signals

def find_expiry_chains(asset):
    """
    Find available expiry chains for an asset
    """
    asset_path = f"/Users/tim/IWLS-OPTIONS/OPTIONS_DATASET/{asset}"
    
    if not os.path.exists(asset_path):
        return []
    
    expiry_folders = [d for d in os.listdir(asset_path) if os.path.isdir(os.path.join(asset_path, d))]
    return expiry_folders

def parse_expiry_date(expiry_string):
    """
    Parse expiry string (YYYYMMDD) to datetime
    """
    try:
        return datetime.strptime(expiry_string, '%Y%m%d')
    except:
        return None

def is_quarterly_expiry(expiry_date):
    """
    Check if expiry is a quarterly LEAPS (3rd Friday of MAR, JUN, SEP, DEC)
    """
    if expiry_date.month not in [3, 6, 9, 12]:
        return False
    
    # Find 3rd Friday
    first_day = expiry_date.replace(day=1)
    first_friday = first_day + timedelta(days=(4 - first_day.weekday()) % 7)
    third_friday = first_friday + timedelta(days=14)
    
    return expiry_date.day == third_friday.day

def is_january_leaps(expiry_date):
    """
    Check if expiry is January LEAPS (3rd Friday of January)
    """
    if expiry_date.month != 1:
        return False
    
    first_day = expiry_date.replace(day=1)
    first_friday = first_day + timedelta(days=(4 - first_day.weekday()) % 7)
    third_friday = first_friday + timedelta(days=14)
    
    return expiry_date.day == third_friday.day

def find_best_expiry(asset, signal_date):
    """
    Find best expiry chain (365-465 DTE, preference for LEAPS/quarterly)
    """
    expiry_chains = find_expiry_chains(asset)
    
    if not expiry_chains:
        return None, None
    
    valid_expiries = []
    
    for expiry_string in expiry_chains:
        expiry_date = parse_expiry_date(expiry_string)
        if not expiry_date:
            continue
        
        # Calculate DTE
        dte = (expiry_date - signal_date).days
        
        # Check if within 365-465 DTE range
        if 365 <= dte <= 465:
            priority = 0
            
            # Assign priority (lower = better)
            if is_january_leaps(expiry_date):
                priority = 1  # Highest priority
            elif is_quarterly_expiry(expiry_date):
                priority = 2  # Second priority
            else:
                priority = 3  # Regular expiry
            
            valid_expiries.append({
                'expiry_string': expiry_string,
                'expiry_date': expiry_date,
                'dte': dte,
                'priority': priority
            })
    
    if not valid_expiries:
        return None, None
    
    # Sort by priority, then by DTE closest to 400 days
    valid_expiries.sort(key=lambda x: (x['priority'], abs(x['dte'] - 400)))
    
    best_expiry = valid_expiries[0]
    return best_expiry['expiry_string'], best_expiry['dte']

def calculate_target_strike(price):
    """
    Calculate 10% OTM strike, rounded to nearest $5
    """
    target_price = price * 1.10  # 10% OTM
    # Round to nearest $5
    rounded_strike = round(target_price / 5) * 5
    return int(rounded_strike)

def find_options_files(asset, expiry):
    """
    Find all options files for given asset and expiry
    """
    options_path = f"/Users/tim/IWLS-OPTIONS/OPTIONS_DATASET/{asset}/{expiry}"
    
    if not os.path.exists(options_path):
        return []
    
    # Look for call options files
    call_files = glob.glob(f"{options_path}/{asset}_{expiry}_C_*.csv")
    return call_files

def extract_strike_from_filename(filename):
    """
    Extract strike price from options filename
    """
    try:
        # Format: ASSET_EXPIRY_C_STRIKE.csv
        parts = os.path.basename(filename).split('_')
        strike_str = parts[-1].replace('.csv', '')
        return int(strike_str)
    except:
        return None

def find_best_strike_file(asset, expiry, target_strike):
    """
    Find options file with strike closest to target
    """
    call_files = find_options_files(asset, expiry)
    
    if not call_files:
        return None, None
    
    best_file = None
    best_strike = None
    min_diff = float('inf')
    
    for file_path in call_files:
        strike = extract_strike_from_filename(file_path)
        if strike:
            diff = abs(strike - target_strike)
            if diff < min_diff:
                min_diff = diff
                best_file = file_path
                best_strike = strike
    
    return best_file, best_strike

def analyze_option_performance(options_file, signal_date):
    """
    Analyze option performance from signal date over next 365 days
    """
    try:
        df = pd.read_csv(options_file)
        df['date'] = pd.to_datetime(df['date'])
        
        # Find data on or after signal date
        signal_data = df[df['date'] >= signal_date].copy()
        
        if len(signal_data) == 0:
            return None
        
        # Get entry price (first available price on/after signal date)
        entry_row = signal_data.iloc[0]
        entry_price = entry_row['close']
        entry_date = entry_row['date']
        
        # Find data within 365 days of entry
        end_date = entry_date + timedelta(days=365)
        analysis_period = signal_data[signal_data['date'] <= end_date]
        
        if len(analysis_period) == 0:
            return None
        
        # Calculate max price and max gain
        max_price = analysis_period['close'].max()
        max_gain_pct = ((max_price / entry_price) - 1) * 100
        
        # Find when max occurred
        max_price_row = analysis_period[analysis_period['close'] == max_price].iloc[0]
        max_date = max_price_row['date']
        days_to_max = (max_date - entry_date).days
        
        return {
            'entry_date': entry_date,
            'entry_price': entry_price,
            'max_price': max_price,
            'max_gain_pct': max_gain_pct,
            'max_date': max_date,
            'days_to_max': days_to_max,
            'analysis_days': len(analysis_period)
        }
        
    except Exception as e:
        print(f"Error analyzing {options_file}: {str(e)}")
        return None

def process_entry_signals(entry_signals):
    """
    Process all entry signals and find matching options
    """
    all_results = []
    
    total_signals = sum(len(df) for df in entry_signals.values())
    processed = 0
    
    print(f"\nProcessing {total_signals} entry signals...")
    
    for bin_name, signals_df in entry_signals.items():
        print(f"\nProcessing {bin_name}: {len(signals_df)} signals")
        
        bin_results = []
        
        for idx, signal in signals_df.iterrows():
            processed += 1
            
            if processed % 100 == 0:
                print(f"  Processed {processed}/{total_signals} signals...")
            
            asset = signal['asset']
            signal_date = signal['date']
            signal_price = signal['price']
            
            # Find best expiry
            expiry, dte = find_best_expiry(asset, signal_date)
            
            if not expiry:
                continue
            
            # Calculate target strike (10% OTM)
            target_strike = calculate_target_strike(signal_price)
            
            # Find best strike file
            options_file, actual_strike = find_best_strike_file(asset, expiry, target_strike)
            
            if not options_file:
                continue
            
            # Analyze option performance
            performance = analyze_option_performance(options_file, signal_date)
            
            if performance:
                result = {
                    'bin_name': bin_name,
                    'signal_date': signal_date.strftime('%Y-%m-%d'),
                    'asset': asset,
                    'signal_price': signal_price,
                    'price_deviation_pct': signal['price_deviation_pct'],
                    'expiry': expiry,
                    'dte': dte,
                    'target_strike': target_strike,
                    'actual_strike': actual_strike,
                    'strike_diff': abs(actual_strike - target_strike),
                    'entry_date': performance['entry_date'].strftime('%Y-%m-%d'),
                    'entry_price': performance['entry_price'],
                    'max_price': performance['max_price'],
                    'max_gain_pct': performance['max_gain_pct'],
                    'max_date': performance['max_date'].strftime('%Y-%m-%d'),
                    'days_to_max': performance['days_to_max'],
                    'analysis_days': performance['analysis_days']
                }
                
                bin_results.append(result)
                all_results.append(result)
        
        print(f"  {bin_name}: {len(bin_results)} successful matches")
    
    return all_results

def analyze_results(results_df):
    """
    Analyze and summarize the options performance results
    """
    print("\n" + "="*80)
    print("OPTIONS PERFORMANCE ANALYSIS")
    print("="*80)
    
    if len(results_df) == 0:
        print("No results to analyze!")
        return
    
    print(f"Total successful option matches: {len(results_df):,}")
    print(f"Assets covered: {results_df['asset'].nunique()}")
    print(f"Date range: {results_df['signal_date'].min()} to {results_df['signal_date'].max()}")
    
    # Overall performance statistics
    print(f"\nOVERALL PERFORMANCE:")
    print(f"Average max gain: {results_df['max_gain_pct'].mean():.1f}%")
    print(f"Median max gain: {results_df['max_gain_pct'].median():.1f}%")
    print(f"Best performing trade: {results_df['max_gain_pct'].max():.1f}%")
    print(f"Worst performing trade: {results_df['max_gain_pct'].min():.1f}%")
    print(f"Standard deviation: {results_df['max_gain_pct'].std():.1f}%")
    
    # Win rate analysis
    positive_trades = len(results_df[results_df['max_gain_pct'] > 0])
    win_rate = (positive_trades / len(results_df)) * 100
    print(f"Win rate (positive gains): {win_rate:.1f}%")
    
    # Performance by deviation bin
    print(f"\nPERFORMANCE BY DEVIATION BIN:")
    print("-" * 60)
    bin_analysis = results_df.groupby('bin_name').agg({
        'max_gain_pct': ['count', 'mean', 'median', 'std'],
        'days_to_max': 'mean'
    }).round(1)
    
    bin_analysis.columns = ['Count', 'Avg_Gain', 'Med_Gain', 'Std_Gain', 'Avg_Days_to_Max']
    print(bin_analysis.to_string())
    
    # Performance by asset
    print(f"\nTOP 10 ASSETS BY AVERAGE GAIN:")
    print("-" * 40)
    asset_performance = results_df.groupby('asset').agg({
        'max_gain_pct': ['count', 'mean']
    }).round(1)
    asset_performance.columns = ['Count', 'Avg_Gain']
    asset_performance = asset_performance[asset_performance['Count'] >= 5]  # At least 5 trades
    top_assets = asset_performance.sort_values('Avg_Gain', ascending=False).head(10)
    print(top_assets.to_string())
    
    # Timing analysis
    print(f"\nTIMING ANALYSIS:")
    print(f"Average days to maximum gain: {results_df['days_to_max'].mean():.0f}")
    print(f"Median days to maximum gain: {results_df['days_to_max'].median():.0f}")
    
    # Strike accuracy
    print(f"\nSTRIKE SELECTION ACCURACY:")
    print(f"Average difference from target strike: ${results_df['strike_diff'].mean():.2f}")
    print(f"Perfect strike matches: {len(results_df[results_df['strike_diff'] == 0]):,} ({len(results_df[results_df['strike_diff'] == 0])/len(results_df)*100:.1f}%)")

def save_results(results):
    """
    Save results to CSV files
    """
    if not results:
        print("No results to save!")
        return
    
    results_df = pd.DataFrame(results)
    
    # Save comprehensive results
    output_file = "/Users/tim/IWLS-OPTIONS/options_analysis_results.csv"
    results_df.to_csv(output_file, index=False)
    print(f"\nResults saved to: {output_file}")
    
    # Save summary by bin
    summary_file = "/Users/tim/IWLS-OPTIONS/options_analysis_summary.csv"
    summary = results_df.groupby('bin_name').agg({
        'max_gain_pct': ['count', 'mean', 'median', 'std', 'min', 'max'],
        'days_to_max': ['mean', 'median'],
        'strike_diff': 'mean'
    }).round(2)
    summary.to_csv(summary_file)
    print(f"Summary saved to: {summary_file}")
    
    return results_df

def main():
    print("Options Performance Analysis")
    print("="*50)
    print("Analyzing entry signals with options contracts (10% OTM, 365-465 DTE)")
    
    # Load entry signals
    entry_signals = load_entry_signals()
    if not entry_signals:
        return
    
    # Process all entry signals
    results = process_entry_signals(entry_signals)
    
    if not results:
        print("No successful option matches found!")
        return
    
    # Save results and get DataFrame
    results_df = save_results(results)
    
    # Analyze results
    analyze_results(results_df)
    
    print(f"\nAnalysis complete!")

if __name__ == "__main__":
    main()