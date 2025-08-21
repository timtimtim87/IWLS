import pandas as pd
import numpy as np
import matplotlib.pyplot as plt
from datetime import datetime
import warnings
import os
import glob
from collections import defaultdict
from scipy import stats
warnings.filterwarnings('ignore')

def iwls_regression(x_vals, y_vals, iterations=5):
    """
    Perform Iteratively Weighted Least Squares regression
    """
    n = len(x_vals)
    if n < 2:
        return np.nan, np.nan
    
    # Initialize weights to 1.0
    weights = np.ones(n)
    slope, intercept = np.nan, np.nan
    
    for iter in range(iterations):
        # Calculate weighted sums
        sum_w = np.sum(weights)
        sum_wx = np.sum(weights * x_vals)
        sum_wy = np.sum(weights * y_vals)
        sum_wxx = np.sum(weights * x_vals * x_vals)
        sum_wxy = np.sum(weights * x_vals * y_vals)
        
        # Calculate slope and intercept
        denominator = sum_w * sum_wxx - sum_wx * sum_wx
        if abs(denominator) > 1e-10:
            slope = (sum_w * sum_wxy - sum_wx * sum_wy) / denominator
            intercept = (sum_wy - slope * sum_wx) / sum_w
            
            # Calculate residuals and update weights (except on last iteration)
            if iter < iterations - 1:
                predicted = intercept + slope * x_vals
                residuals = np.abs(y_vals - predicted)
                mean_residual = np.mean(residuals)
                
                # Update weights (inverse of residual distance)
                new_weights = 1.0 / (residuals + mean_residual * 0.1)
                weights = new_weights
        else:
            break
    
    return slope, intercept

def calculate_iwls_growth_rate(data, lookback_period=1500, use_sampling=False, sample_ratio=2):
    """
    Calculate IWLS growth rate for each day in the dataset
    """
    results = []
    
    print(f"    Calculating IWLS for {len(data)} data points...")
    
    for i in range(len(data)):
        if i < lookback_period - 1:
            # Not enough data for calculation
            results.append({
                'date': data.iloc[i]['date'],
                'price': data.iloc[i]['close'],
                'annual_growth': np.nan,
                'slope': np.nan,
                'intercept': np.nan,
                'trend_line_value': np.nan,
                'price_deviation': np.nan,
                'absolute_deviation': np.nan
            })
            continue
        
        # Get lookback data
        start_idx = i - lookback_period + 1
        end_idx = i + 1
        lookback_data = data.iloc[start_idx:end_idx]
        
        # Determine step size for sampling
        step_size = sample_ratio if use_sampling and lookback_period > 500 else 1
        
        # Sample data if needed
        if step_size > 1:
            indices = range(0, len(lookback_data), step_size)
            sampled_data = lookback_data.iloc[indices]
        else:
            sampled_data = lookback_data
        
        # Prepare data for regression (x = time index, y = log price)
        x_vals = np.arange(len(sampled_data), dtype=float)
        y_vals = np.log(sampled_data['close'].values)
        
        # Perform IWLS regression
        slope, intercept = iwls_regression(x_vals, y_vals)
        
        # Convert slope to annual growth rate and calculate deviations
        if not np.isnan(slope):
            annual_growth = (np.exp(slope * 252) - 1) * 100
            # Calculate trend line value at current point
            trend_line_log = intercept + slope * (len(sampled_data) - 1)
            trend_line_value = np.exp(trend_line_log)
            
            # Calculate price deviation from trend
            current_price = data.iloc[i]['close']
            price_deviation = ((current_price / trend_line_value) - 1) * 100
            absolute_deviation = abs(price_deviation)
        else:
            annual_growth = np.nan
            trend_line_value = np.nan
            price_deviation = np.nan
            absolute_deviation = np.nan
        
        results.append({
            'date': data.iloc[i]['date'],
            'price': data.iloc[i]['close'],
            'annual_growth': annual_growth,
            'slope': slope,
            'intercept': intercept,
            'trend_line_value': trend_line_value,
            'price_deviation': price_deviation,
            'absolute_deviation': absolute_deviation,
            'data_points': len(sampled_data)
        })
        
        # Progress indicator
        if (i + 1) % 1000 == 0:
            print(f"      Processed {i + 1}/{len(data)} days...")
    
    return pd.DataFrame(results)

def get_deviation_bin(deviation):
    """
    Categorize deviation into 5% bins
    """
    if deviation >= 50:
        return ">+50%"
    elif deviation >= 45:
        return "+45% to +50%"
    elif deviation >= 40:
        return "+40% to +45%"
    elif deviation >= 35:
        return "+35% to +40%"
    elif deviation >= 30:
        return "+30% to +35%"
    elif deviation >= 25:
        return "+25% to +30%"
    elif deviation >= 20:
        return "+20% to +25%"
    elif deviation >= 15:
        return "+15% to +20%"
    elif deviation >= 10:
        return "+10% to +15%"
    elif deviation >= 5:
        return "+5% to +10%"
    elif deviation >= -5:
        return "-5% to +5%"
    elif deviation >= -10:
        return "-10% to -5%"
    elif deviation >= -15:
        return "-15% to -10%"
    elif deviation >= -20:
        return "-20% to -15%"
    elif deviation >= -25:
        return "-25% to -20%"
    elif deviation >= -30:
        return "-30% to -25%"
    elif deviation >= -35:
        return "-35% to -30%"
    elif deviation >= -40:
        return "-40% to -35%"
    elif deviation >= -45:
        return "-45% to -40%"
    elif deviation >= -50:
        return "-50% to -45%"
    else:
        return "<-50%"

def calculate_bin_statistics(results_df):
    """
    Calculate time spent in each deviation bin
    """
    valid_data = results_df.dropna().copy()
    
    if len(valid_data) == 0:
        return pd.DataFrame()
    
    # Add bin classification
    valid_data['deviation_bin'] = valid_data['price_deviation'].apply(get_deviation_bin)
    
    # Calculate bin statistics
    bin_stats = valid_data.groupby('deviation_bin').agg({
        'price_deviation': ['count', 'mean', 'std'],
        'absolute_deviation': 'mean',
        'price': 'mean'
    }).round(3)
    
    # Flatten column names
    bin_stats.columns = ['days_count', 'mean_deviation', 'std_deviation', 'mean_abs_deviation', 'mean_price']
    
    # Calculate percentages
    total_days = len(valid_data)
    bin_stats['percentage_time'] = (bin_stats['days_count'] / total_days * 100).round(2)
    
    # Reset index to make deviation_bin a column
    bin_stats = bin_stats.reset_index()
    
    return bin_stats

def calculate_zscore_analysis(results_df, lookback_window=252):
    """
    Calculate Z-scores for price deviations to normalize across assets
    """
    valid_data = results_df.dropna().copy()
    
    if len(valid_data) < lookback_window * 2:
        return pd.DataFrame()
    
    zscore_data = []
    
    for i in range(lookback_window, len(valid_data)):
        # Get historical deviations for Z-score calculation
        historical_deviations = valid_data['price_deviation'].iloc[i-lookback_window:i]
        current_deviation = valid_data['price_deviation'].iloc[i]
        
        # Calculate Z-score
        mean_dev = historical_deviations.mean()
        std_dev = historical_deviations.std()
        
        if std_dev > 0:
            z_score = (current_deviation - mean_dev) / std_dev
        else:
            z_score = 0
        
        zscore_data.append({
            'date': valid_data['date'].iloc[i],
            'price_deviation': current_deviation,
            'z_score': z_score,
            'historical_mean': mean_dev,
            'historical_std': std_dev
        })
    
    return pd.DataFrame(zscore_data)

def analyze_forward_returns(results_df, forward_days=252):
    """
    Analyze forward returns by deviation bin for pattern strength analysis
    """
    valid_data = results_df.dropna().copy()
    
    if len(valid_data) < forward_days + 100:
        return pd.DataFrame()
    
    valid_data['deviation_bin'] = valid_data['price_deviation'].apply(get_deviation_bin)
    
    forward_returns = []
    
    for i in range(len(valid_data) - forward_days):
        current_price = valid_data['price'].iloc[i]
        current_bin = valid_data['deviation_bin'].iloc[i]
        current_deviation = valid_data['price_deviation'].iloc[i]
        
        # Calculate forward returns
        future_data = valid_data.iloc[i+1:i+forward_days+1]
        
        if len(future_data) >= forward_days * 0.8:  # Need at least 80% of forward data
            # Maximum gain
            max_price = future_data['price'].max()
            max_gain = ((max_price / current_price) - 1) * 100
            
            # Final return
            final_price = future_data['price'].iloc[-1]
            final_return = ((final_price / current_price) - 1) * 100
            
            forward_returns.append({
                'deviation_bin': current_bin,
                'price_deviation': current_deviation,
                'forward_max_gain': max_gain,
                'forward_final_return': final_return,
                'entry_price': current_price
            })
    
    forward_df = pd.DataFrame(forward_returns)
    
    if len(forward_df) == 0:
        return pd.DataFrame()
    
    # Calculate statistics by bin
    bin_order = [">+50%", "+45% to +50%", "+40% to +45%", "+35% to +40%", "+30% to +35%", 
                 "+25% to +30%", "+20% to +25%", "+15% to +20%", "+10% to +15%", "+5% to +10%", 
                 "-5% to +5%", "-10% to -5%", "-15% to -10%", "-20% to -15%", "-25% to -20%", 
                 "-30% to -25%", "-35% to -30%", "-40% to -35%", "-45% to -40%", "-50% to -45%", "<-50%"]
    
    pattern_strength = []
    
    for bin_name in bin_order:
        bin_data = forward_df[forward_df['deviation_bin'] == bin_name]
        
        if len(bin_data) >= 5:  # Need minimum samples
            # Calculate confidence score based on sample size and consistency
            mean_gain = bin_data['forward_max_gain'].mean()
            std_gain = bin_data['forward_max_gain'].std()
            
            # Confidence score: higher sample size and lower variance = higher confidence
            if std_gain > 0:
                confidence_score = min(100, (len(bin_data) / 10) * (abs(mean_gain) / std_gain))
            else:
                confidence_score = len(bin_data) / 10 * 10  # If no variance, base on sample size
            
            pattern_strength.append({
                'deviation_bin': bin_name,
                'sample_count': len(bin_data),
                'avg_max_gain': round(mean_gain, 2),
                'avg_final_return': round(bin_data['forward_final_return'].mean(), 2),
                'median_max_gain': round(bin_data['forward_max_gain'].median(), 2),
                'std_max_gain': round(std_gain, 2),
                'success_rate': round((bin_data['forward_final_return'] > 0).mean() * 100, 1),
                'confidence_score': round(confidence_score, 1)
            })
    
    return pd.DataFrame(pattern_strength)

def create_asset_visualizations(results_df, asset_name, output_dir):
    """
    Create comprehensive visualizations for an asset
    """
    valid_data = results_df.dropna()
    
    if len(valid_data) < 100:
        print(f"    Insufficient data for {asset_name} visualization")
        return
    
    fig, ((ax1, ax2), (ax3, ax4)) = plt.subplots(2, 2, figsize=(20, 12))
    
    # Plot 1: Price and Trend Line
    ax1.plot(valid_data['date'], valid_data['price'], label=f'{asset_name} Price', alpha=0.7, linewidth=1, color='blue')
    ax1.plot(valid_data['date'], valid_data['trend_line_value'], label='IWLS Trend Line', 
             color='red', linewidth=2)
    ax1.set_ylabel('Price ($)')
    ax1.set_title(f'{asset_name} Price vs IWLS Trend Line', fontweight='bold')
    ax1.legend()
    ax1.grid(True, alpha=0.3)
    ax1.set_yscale('log')
    
    # Plot 2: Price Deviation from Trend
    ax2.plot(valid_data['date'], valid_data['price_deviation'], color='green', linewidth=1)
    ax2.axhline(y=0, color='black', linestyle='--', alpha=0.5)
    
    # Add deviation threshold lines
    for threshold in [10, 20, 30, -10, -20, -30]:
        ax2.axhline(y=threshold, color='red', linestyle=':', alpha=0.5)
    
    ax2.set_ylabel('Deviation from Trend (%)')
    ax2.set_title('Price Deviation from IWLS Trend Line', fontweight='bold')
    ax2.grid(True, alpha=0.3)
    
    # Plot 3: Annual Growth Rate Over Time
    ax3.plot(valid_data['date'], valid_data['annual_growth'], color='purple', linewidth=1)
    ax3.axhline(y=0, color='black', linestyle='--', alpha=0.5)
    ax3.set_ylabel('Annual Growth Rate (%)')
    ax3.set_title('IWLS Annual Growth Rate Over Time', fontweight='bold')
    ax3.grid(True, alpha=0.3)
    
    # Plot 4: Distribution of Deviations
    ax4.hist(valid_data['price_deviation'], bins=50, alpha=0.7, color='orange', edgecolor='black')
    ax4.axvline(valid_data['price_deviation'].mean(), color='red', linestyle='--', linewidth=2, 
                label=f'Mean: {valid_data["price_deviation"].mean():.1f}%')
    ax4.axvline(valid_data['price_deviation'].median(), color='blue', linestyle='--', linewidth=2, 
                label=f'Median: {valid_data["price_deviation"].median():.1f}%')
    ax4.set_xlabel('Price Deviation (%)')
    ax4.set_ylabel('Frequency')
    ax4.set_title('Distribution of Price Deviations', fontweight='bold')
    ax4.legend()
    ax4.grid(True, alpha=0.3)
    
    plt.tight_layout()
    plt.savefig(f"{output_dir}/{asset_name}_iwls_analysis.png", dpi=300, bbox_inches='tight')
    plt.close()

def process_single_asset(data, asset_name, output_dir):
    """
    Process a single asset with complete analysis
    """
    print(f"\n  Processing {asset_name}...")
    
    try:
        # Need at least 2000 days for meaningful analysis
        if len(data) < 2000:
            print(f"    Skipping {asset_name}: insufficient data ({len(data)} days)")
            return False
        
        # Create asset-specific output directory
        asset_output_dir = os.path.join(output_dir, asset_name)
        os.makedirs(asset_output_dir, exist_ok=True)
        
        # Calculate IWLS growth rate and deviations
        print(f"    Calculating IWLS analysis...")
        results = calculate_iwls_growth_rate(data, lookback_period=1500)
        
        # Save main results file
        results_file = os.path.join(asset_output_dir, f"{asset_name}_iwls_results.csv")
        results.to_csv(results_file, index=False)
        
        # Calculate bin statistics
        print(f"    Calculating bin statistics...")
        bin_stats = calculate_bin_statistics(results)
        if len(bin_stats) > 0:
            bin_stats_file = os.path.join(asset_output_dir, f"{asset_name}_bin_statistics.csv")
            bin_stats.to_csv(bin_stats_file, index=False)
        
        # Calculate Z-score analysis
        print(f"    Calculating Z-score analysis...")
        zscore_analysis = calculate_zscore_analysis(results)
        if len(zscore_analysis) > 0:
            zscore_file = os.path.join(asset_output_dir, f"{asset_name}_zscore_analysis.csv")
            zscore_analysis.to_csv(zscore_file, index=False)
        
        # Calculate forward returns pattern strength
        print(f"    Calculating pattern strength...")
        pattern_strength = analyze_forward_returns(results)
        if len(pattern_strength) > 0:
            pattern_file = os.path.join(asset_output_dir, f"{asset_name}_pattern_strength.csv")
            pattern_strength.to_csv(pattern_file, index=False)
        
        # Create visualizations
        print(f"    Creating visualizations...")
        create_asset_visualizations(results, asset_name, asset_output_dir)
        
        # Print summary
        valid_data = results.dropna()
        if len(valid_data) > 0:
            latest = valid_data.iloc[-1]
            print(f"    Summary for {asset_name}:")
            print(f"      Current price: ${latest['price']:.2f}")
            print(f"      Current deviation: {latest['price_deviation']:.2f}%")
            print(f"      Current growth rate: {latest['annual_growth']:.2f}%")
            print(f"      Valid data points: {len(valid_data):,}")
        
        print(f"    ✅ {asset_name} analysis complete")
        return True
        
    except Exception as e:
        print(f"    ❌ Error processing {asset_name}: {str(e)}")
        return False

def create_summary_analysis(output_dir):
    """
    Create cross-asset summary analysis
    """
    print(f"\nCreating cross-asset summary analysis...")
    
    # Collect all asset results
    asset_summaries = []
    all_pattern_strength = []
    
    for asset_dir in os.listdir(output_dir):
        asset_path = os.path.join(output_dir, asset_dir)
        if os.path.isdir(asset_path):
            
            # Load main results
            results_file = os.path.join(asset_path, f"{asset_dir}_iwls_results.csv")
            if os.path.exists(results_file):
                try:
                    df = pd.read_csv(results_file)
                    valid_data = df.dropna()
                    
                    if len(valid_data) > 0:
                        latest = valid_data.iloc[-1]
                        
                        asset_summaries.append({
                            'asset': asset_dir,
                            'total_days': len(df),
                            'valid_days': len(valid_data),
                            'current_price': latest['price'],
                            'current_deviation': latest['price_deviation'],
                            'current_growth_rate': latest['annual_growth'],
                            'avg_deviation': valid_data['price_deviation'].mean(),
                            'std_deviation': valid_data['price_deviation'].std(),
                            'max_deviation': valid_data['price_deviation'].max(),
                            'min_deviation': valid_data['price_deviation'].min()
                        })
                    
                    # Load pattern strength data
                    pattern_file = os.path.join(asset_path, f"{asset_dir}_pattern_strength.csv")
                    if os.path.exists(pattern_file):
                        pattern_df = pd.read_csv(pattern_file)
                        pattern_df['asset'] = asset_dir
                        all_pattern_strength.append(pattern_df)
                        
                except Exception as e:
                    print(f"    Error loading {asset_dir}: {e}")
    
    # Save summary files
    if asset_summaries:
        summary_df = pd.DataFrame(asset_summaries)
        summary_df.to_csv(os.path.join(output_dir, "SUMMARY_all_assets.csv"), index=False)
        print(f"    Saved summary for {len(asset_summaries)} assets")
    
    if all_pattern_strength:
        combined_patterns = pd.concat(all_pattern_strength, ignore_index=True)
        combined_patterns.to_csv(os.path.join(output_dir, "SUMMARY_pattern_strength.csv"), index=False)
        print(f"    Saved combined pattern strength analysis")

def setup_directories():
    """
    Setup clean directory structure
    """
    base_dir = "/Users/tim/IWLS-OPTIONS"
    
    # Create OLD directory and move existing files
    old_dir = os.path.join(base_dir, "OLD")
    os.makedirs(old_dir, exist_ok=True)
    
    # List of directories/files to move to OLD
    items_to_move = [
        "ASSETS_RESULTS", "TRADING_ANALYTICS", "CORRELATION_ANALYSIS", 
        "ZSCORE_ANALYSIS", "CORRECTED_EV_ANALYSIS", "ABSOLUTE_DEVIATION_ANALYSIS",
        "SIGNAL_TIMING_ANALYSIS", "REBALANCING_STRATEGY", "EV_VS_ABSOLUTE_COMPARISON",
        "ENTRY_SIGNALS"
    ]
    
    moved_count = 0
    for item in items_to_move:
        item_path = os.path.join(base_dir, item)
        if os.path.exists(item_path):
            import shutil
            old_item_path = os.path.join(old_dir, item)
            if not os.path.exists(old_item_path):
                try:
                    shutil.move(item_path, old_item_path)
                    moved_count += 1
                except Exception as e:
                    print(f"Warning: Could not move {item}: {e}")
    
    if moved_count > 0:
        print(f"Moved {moved_count} existing analysis folders to OLD directory")
    
    # Create new V2 directory
    v2_dir = os.path.join(base_dir, "IWLS_ANALYSIS_V2")
    os.makedirs(v2_dir, exist_ok=True)
    
    return v2_dir

def main():
    print("IWLS Analysis V2 - Comprehensive Multi-Asset Analysis")
    print("=" * 70)
    print("Processing large dataset with detailed output files")
    print("Includes: IWLS growth lines, deviation analysis, Z-scores, pattern strength")
    
    # Setup directories
    output_dir = setup_directories()
    print(f"\nOutput directory: {output_dir}")
    
    # Load the new dataset
    data_file = "/Users/tim/IWLS-OPTIONS/MERGED_STOCK_DATA.csv"
    
    if not os.path.exists(data_file):
        print(f"❌ Data file not found: {data_file}")
        return
    
    print(f"\nLoading dataset: {data_file}")
    
    try:
        df = pd.read_csv(data_file)
        print(f"✅ Loaded dataset with {len(df)} rows and {len(df.columns)} columns")
        
        # Convert timestamp to datetime
        if 'time' in df.columns:
            df['date'] = pd.to_datetime(df['time'], unit='s')
        elif 'Date' in df.columns:
            df['date'] = pd.to_datetime(df['Date'])
        else:
            print("❌ No time/date column found")
            return
        
        # Sort by date
        df = df.sort_values('date').reset_index(drop=True)
        
        print(f"Date range: {df['date'].min().strftime('%Y-%m-%d')} to {df['date'].max().strftime('%Y-%m-%d')}")
        
    except Exception as e:
        print(f"❌ Error loading data: {e}")
        return
    
    # Find all asset columns (exclude time/date columns)
    exclude_columns = ['time', 'date', 'Date', 'timestamp', 'Time']
    asset_columns = [col for col in df.columns if col not in exclude_columns]
    
    print(f"Found {len(asset_columns)} assets to analyze")
    print(f"Assets: {asset_columns[:10]}{'...' if len(asset_columns) > 10 else ''}")
    
    # Process each asset
    successful_assets = 0
    total_assets = len(asset_columns)
    
    for i, col in enumerate(asset_columns):
        print(f"\n[{i+1}/{total_assets}] Processing asset: {col}")
        
        # Clean asset name for file system (remove ** prefix if present)
        asset_name = col.replace('**', '').replace('*', '')
        asset_name = "".join(c for c in asset_name if c.isalnum() or c in (' ', '-', '_')).strip()
        
        # Create asset-specific dataframe
        asset_data = df[['date', col]].copy()
        asset_data = asset_data.rename(columns={col: 'close'})
        asset_data = asset_data.dropna()
        
        if len(asset_data) > 0:
            success = process_single_asset(asset_data, asset_name, output_dir)
            if success:
                successful_assets += 1
    
    # Create cross-asset summary
    create_summary_analysis(output_dir)
    
    print(f"\n" + "="*70)
    print("IWLS ANALYSIS V2 COMPLETE")
    print("="*70)
    print(f"Successfully processed: {successful_assets}/{total_assets} assets")
    print(f"Results saved to: {output_dir}")
    print("\nOutput structure:")
    print("  📁 IWLS_ANALYSIS_V2/")
    print("    📁 [ASSET_NAME]/")
    print("      📄 [ASSET]_iwls_results.csv (main daily data)")
    print("      📄 [ASSET]_bin_statistics.csv (time in each deviation bin)")
    print("      📄 [ASSET]_zscore_analysis.csv (normalized deviations)")
    print("      📄 [ASSET]_pattern_strength.csv (bin correlation with future returns)")
    print("      📄 [ASSET]_iwls_analysis.png (comprehensive charts)")
    print("    📄 SUMMARY_all_assets.csv (cross-asset overview)")
    print("    📄 SUMMARY_pattern_strength.csv (combined pattern analysis)")
    
    print(f"\n🎯 Ready for further analysis:")
    print(f"   • Individual asset files for detailed examination")
    print(f"   • Summary files for cross-asset comparison")
    print(f"   • Pattern strength files for strategy development")
    print(f"   • Z-score files for normalized comparisons")

if __name__ == "__main__":
    main()