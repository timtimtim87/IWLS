import pandas as pd
import numpy as np
import os
import glob
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
            all_results[asset_name] = df
            print(f"Loaded {asset_name}: {len(df)} data points")
        except Exception as e:
            print(f"Error loading {asset_name}: {str(e)}")
    
    return all_results

def get_deviation_bin(deviation):
    """
    Categorize deviation into 5% bins (negative only, 0% to -30%)
    """
    if deviation >= -5:
        return "0% to -5%"
    elif deviation >= -10:
        return "-5% to -10%"
    elif deviation >= -15:
        return "-10% to -15%"
    elif deviation >= -20:
        return "-15% to -20%"
    elif deviation >= -25:
        return "-20% to -25%"
    elif deviation >= -30:
        return "-25% to -30%"
    else:
        return None  # Below -30%, ignore

def generate_entry_signals(all_results):
    """
    Generate entry signals for each negative deviation bin
    """
    # Initialize containers for each bin
    entry_signals = {
        "0% to -5%": [],
        "-5% to -10%": [],
        "-10% to -15%": [],
        "-15% to -20%": [],
        "-20% to -25%": [],
        "-25% to -30%": []
    }
    
    print("\nGenerating entry signals...")
    
    for asset_name, df in all_results.items():
        print(f"  Processing {asset_name}...")
        
        # Calculate price deviation from IWLS trend
        valid_data = df.dropna().copy()
        valid_data['price_deviation'] = ((valid_data['price'] / valid_data['trend_line_value']) - 1) * 100
        
        # Get deviation bin for each day
        valid_data['deviation_bin'] = valid_data['price_deviation'].apply(get_deviation_bin)
        
        # Process each day for entry signals
        for idx, row in valid_data.iterrows():
            deviation_bin = row['deviation_bin']
            
            # Only process negative deviation bins within our range
            if deviation_bin and deviation_bin in entry_signals:
                
                # Calculate additional metrics for options analysis
                signal_data = {
                    'date': row['date'].strftime('%Y-%m-%d'),
                    'asset': asset_name,
                    'price': row['price'],
                    'trend_line_value': row['trend_line_value'],
                    'price_deviation_pct': row['price_deviation'],
                    'annual_growth_rate': row['annual_growth'],
                    'deviation_bin': deviation_bin,
                    'days_to_expiry': None,  # Will be calculated when matching with options
                    'moneyness_target': None  # Will be calculated when matching with options
                }
                
                # Add signal to appropriate bin
                entry_signals[deviation_bin].append(signal_data)
    
    return entry_signals

def save_entry_signals(entry_signals):
    """
    Save entry signals to separate CSV files for each deviation bin
    """
    output_dir = "/Users/tim/IWLS-OPTIONS/ENTRY_SIGNALS"
    os.makedirs(output_dir, exist_ok=True)
    
    print(f"\nSaving entry signals to {output_dir}/")
    
    summary_stats = {}
    
    for bin_name, signals in entry_signals.items():
        if len(signals) > 0:
            # Convert to DataFrame
            df = pd.DataFrame(signals)
            
            # Sort by date
            df = df.sort_values(['date', 'asset'])
            
            # Create filename
            bin_filename = bin_name.replace('%', 'pct').replace(' ', '_').replace('to_', 'to_')
            filename = f"{output_dir}/entry_signals_{bin_filename}.csv"
            
            # Save CSV
            df.to_csv(filename, index=False)
            
            # Calculate summary stats
            summary_stats[bin_name] = {
                'total_signals': len(signals),
                'unique_assets': df['asset'].nunique(),
                'date_range': f"{df['date'].min()} to {df['date'].max()}",
                'avg_deviation': df['price_deviation_pct'].mean(),
                'avg_price': df['price'].mean()
            }
            
            print(f"  {bin_name}: {len(signals)} signals saved to entry_signals_{bin_filename}.csv")
        else:
            print(f"  {bin_name}: No signals found")
    
    return summary_stats

def print_summary_statistics(summary_stats):
    """
    Print detailed summary of entry signals
    """
    print("\n" + "="*80)
    print("ENTRY SIGNALS SUMMARY")
    print("="*80)
    
    total_signals = sum(stats['total_signals'] for stats in summary_stats.values())
    print(f"Total entry signals generated: {total_signals:,}")
    
    print(f"\nBreakdown by deviation bin:")
    print("-" * 80)
    print(f"{'Deviation Bin':<15} {'Signals':<8} {'Assets':<7} {'Avg Dev':<8} {'Avg Price':<10}")
    print("-" * 80)
    
    for bin_name, stats in summary_stats.items():
        print(f"{bin_name:<15} {stats['total_signals']:>7,} {stats['unique_assets']:>6} "
              f"{stats['avg_deviation']:>7.1f}% ${stats['avg_price']:>8.2f}")
    
    print("-" * 80)
    
    # Additional insights
    print(f"\nKey insights:")
    
    # Find most active bin
    if summary_stats:
        most_active_bin = max(summary_stats.items(), key=lambda x: x[1]['total_signals'])
        print(f"• Most active deviation bin: {most_active_bin[0]} ({most_active_bin[1]['total_signals']:,} signals)")
        
        # Find deepest average deviation
        deepest_bin = min(summary_stats.items(), key=lambda x: x[1]['avg_deviation'])
        print(f"• Deepest average deviation: {deepest_bin[0]} ({deepest_bin[1]['avg_deviation']:.1f}%)")
        
        # Calculate signal frequency
        total_days_range = len(summary_stats)
        if total_days_range > 0:
            avg_signals_per_bin = total_signals / len(summary_stats)
            print(f"• Average signals per bin: {avg_signals_per_bin:,.0f}")

def validate_entry_signals():
    """
    Validate that entry signal files exist and can be loaded
    """
    print("\n" + "="*50)
    print("VALIDATION CHECK")
    print("="*50)
    
    signal_files = glob.glob("/Users/tim/IWLS-OPTIONS/ENTRY_SIGNALS/entry_signals_*.csv")
    
    if not signal_files:
        print("❌ No entry signal files found!")
        return False
    
    print(f"✅ Found {len(signal_files)} entry signal files:")
    
    total_rows = 0
    for file_path in sorted(signal_files):
        filename = os.path.basename(file_path)
        try:
            df = pd.read_csv(file_path)
            print(f"  {filename}: {len(df):,} signals")
            total_rows += len(df)
        except Exception as e:
            print(f"  {filename}: ❌ Error loading - {str(e)}")
    
    print(f"\nTotal signals across all files: {total_rows:,}")
    
    # Sample a few signals for verification
    if signal_files:
        sample_file = signal_files[0]
        sample_df = pd.read_csv(sample_file)
        if len(sample_df) > 0:
            print(f"\nSample signals from {os.path.basename(sample_file)}:")
            print(sample_df.head(3)[['date', 'asset', 'price', 'price_deviation_pct']].to_string(index=False))
    
    return True

def main():
    print("IWLS Entry Signals Generator")
    print("="*50)
    print("Generating entry signals for negative deviation bins (0% to -30%)")
    
    # Load all IWLS results
    all_results = load_all_iwls_results()
    if not all_results:
        return
    
    print(f"\nLoaded {len(all_results)} assets")
    
    # Generate entry signals
    entry_signals = generate_entry_signals(all_results)
    
    # Save entry signals to CSV files
    summary_stats = save_entry_signals(entry_signals)
    
    # Print summary statistics
    print_summary_statistics(summary_stats)
    
    # Validate the generated files
    validate_entry_signals()
    
    print(f"\n✅ Entry signal generation complete!")
    print(f"Files saved to: /Users/tim/IWLS-OPTIONS/ENTRY_SIGNALS/")
    print(f"Ready for options strategy backtesting.")

if __name__ == "__main__":
    main()