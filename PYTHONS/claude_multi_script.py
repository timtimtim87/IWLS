import pandas as pd
import numpy as np
import matplotlib.pyplot as plt
from datetime import datetime
import warnings
import os
import glob
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
    
    for i in range(len(data)):
        if i < lookback_period - 1:
            # Not enough data for calculation
            results.append({
                'date': data.iloc[i]['date'],
                'price': data.iloc[i]['close'],
                'annual_growth': np.nan,
                'slope': np.nan,
                'intercept': np.nan,
                'trend_line_value': np.nan
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
        
        # Convert slope to annual growth rate
        if not np.isnan(slope):
            annual_growth = (np.exp(slope * 252) - 1) * 100
            # Calculate trend line value at current point
            trend_line_log = intercept + slope * (len(sampled_data) - 1)
            trend_line_value = np.exp(trend_line_log)
        else:
            annual_growth = np.nan
            trend_line_value = np.nan
        
        results.append({
            'date': data.iloc[i]['date'],
            'price': data.iloc[i]['close'],
            'annual_growth': annual_growth,
            'slope': slope,
            'intercept': intercept,
            'trend_line_value': trend_line_value,
            'data_points': len(sampled_data)
        })
        
        # Progress indicator
        if (i + 1) % 500 == 0:
            print(f"    Processed {i + 1}/{len(data)} days...")
    
    return pd.DataFrame(results)

def analyze_forward_returns_by_deviation(results_df):
    """
    Analyze 1-year forward maximum gains by deviation bin
    """
    # Calculate price deviation from trend
    valid_data = results_df.dropna().copy()
    valid_data['price_deviation'] = ((valid_data['price'] / valid_data['trend_line_value']) - 1) * 100
    
    # Define deviation bins in 5% increments from -50% to +50%
    def get_deviation_bin(deviation):
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
    
    valid_data['deviation_bin'] = valid_data['price_deviation'].apply(get_deviation_bin)
    
    # Calculate 1-year forward max gains
    forward_returns = []
    
    for i in range(len(valid_data)):
        current_price = valid_data.iloc[i]['price']
        current_bin = valid_data.iloc[i]['deviation_bin']
        
        # Look forward 252 trading days (1 year)
        future_data = valid_data.iloc[i+1:i+253] if i+252 < len(valid_data) else valid_data.iloc[i+1:]
        
        if len(future_data) >= 200:  # Need at least ~8 months of data
            max_future_price = future_data['price'].max()
            max_gain = ((max_future_price / current_price) - 1) * 100
            
            forward_returns.append({
                'deviation_bin': current_bin,
                'forward_max_gain': max_gain
            })
    
    forward_df = pd.DataFrame(forward_returns)
    
    # Calculate statistics by bin (full range -50% to +50% in 5% increments)
    bin_order = [">+50%", "+45% to +50%", "+40% to +45%", "+35% to +40%", "+30% to +35%", 
                 "+25% to +30%", "+20% to +25%", "+15% to +20%", "+10% to +15%", "+5% to +10%", 
                 "-5% to +5%", "-10% to -5%", "-15% to -10%", "-20% to -15%", "-25% to -20%", 
                 "-30% to -25%", "-35% to -30%", "-40% to -35%", "-45% to -40%", "-50% to -45%", "<-50%"]
    
    bin_stats = {}
    for bin_name in bin_order:
        bin_data = forward_df[forward_df['deviation_bin'] == bin_name]
        if len(bin_data) > 0:
            bin_stats[bin_name] = {
                'count': len(bin_data),
                'avg_forward_max_gain': bin_data['forward_max_gain'].mean(),
                'median_forward_max_gain': bin_data['forward_max_gain'].median()
            }
        else:
            bin_stats[bin_name] = {
                'count': 0,
                'avg_forward_max_gain': np.nan,
                'median_forward_max_gain': np.nan
            }
    
    return bin_stats, valid_data

def plot_basic_results(results_df, asset_name):
    """
    Create basic IWLS analysis plots
    """
    valid_data = results_df.dropna()
    
    fig, ((ax1, ax2), (ax3, ax4)) = plt.subplots(2, 2, figsize=(16, 12))
    
    # Plot 1: Price and Trend Line
    ax1.plot(valid_data['date'], valid_data['price'], label=f'{asset_name} Price', alpha=0.7, linewidth=1)
    ax1.plot(valid_data['date'], valid_data['trend_line_value'], label='IWLS Trend Line', 
             color='red', linewidth=2)
    ax1.set_ylabel('Price ($)')
    ax1.set_title(f'{asset_name} Price vs IWLS Trend Line')
    ax1.legend()
    ax1.grid(True, alpha=0.3)
    ax1.set_yscale('log')
    
    # Plot 2: Annual Growth Rate Over Time
    ax2.plot(valid_data['date'], valid_data['annual_growth'], color='blue', linewidth=1)
    ax2.axhline(y=0, color='black', linestyle='--', alpha=0.5)
    ax2.set_ylabel('Annual Growth Rate (%)')
    ax2.set_title('IWLS Annual Growth Rate Over Time')
    ax2.grid(True, alpha=0.3)
    
    # Plot 3: Price Deviation from Trend
    valid_data_copy = valid_data.copy()
    valid_data_copy['price_deviation'] = ((valid_data_copy['price'] / valid_data_copy['trend_line_value']) - 1) * 100
    ax3.plot(valid_data_copy['date'], valid_data_copy['price_deviation'], color='green', linewidth=1)
    ax3.axhline(y=0, color='black', linestyle='--', alpha=0.5)
    # Add threshold lines every 10%
    for threshold in [10, 20, 30, 40, 50, -10, -20, -30, -40, -50]:
        ax3.axhline(y=threshold, color='red', linestyle=':', alpha=0.5)
    ax3.set_ylabel('Deviation from Trend (%)')
    ax3.set_title('Price Deviation from IWLS Trend Line')
    ax3.grid(True, alpha=0.3)
    
    # Plot 4: Distribution of Annual Growth Rates
    ax4.hist(valid_data['annual_growth'], bins=50, alpha=0.7, color='purple', edgecolor='black')
    ax4.axvline(valid_data['annual_growth'].mean(), color='red', linestyle='--', 
                label=f'Mean: {valid_data["annual_growth"].mean():.2f}%')
    ax4.set_xlabel('Annual Growth Rate (%)')
    ax4.set_ylabel('Frequency')
    ax4.set_title('Distribution of IWLS Annual Growth Rates')
    ax4.legend()
    ax4.grid(True, alpha=0.3)
    
    plt.tight_layout()
    plt.savefig(f"/Users/tim/IWLS-OPTIONS/{asset_name}_iwls_analysis.png", dpi=300, bbox_inches='tight')
    plt.close()

def plot_forward_gains(bin_stats, asset_name):
    """
    Plot forward max gains by deviation bin
    """
    bin_order = [">+50%", "+45% to +50%", "+40% to +45%", "+35% to +40%", "+30% to +35%", 
                 "+25% to +30%", "+20% to +25%", "+15% to +20%", "+10% to +15%", "+5% to +10%", 
                 "-5% to +5%", "-10% to -5%", "-15% to -10%", "-20% to -15%", "-25% to -20%", 
                 "-30% to -25%", "-35% to -30%", "-40% to -35%", "-45% to -40%", "-50% to -45%", "<-50%"]
    
    # Get data for plotting (only bins with data)
    avg_gains = []
    bin_labels = []
    sample_counts = []
    
    for bin_name in bin_order:
        if bin_stats[bin_name]['count'] > 0:
            avg_gains.append(bin_stats[bin_name]['avg_forward_max_gain'])
            bin_labels.append(bin_name)
            sample_counts.append(bin_stats[bin_name]['count'])
    
    if not avg_gains:
        print(f"    No forward return data available for {asset_name}")
        return
    
    # Create plot
    fig, ax = plt.subplots(figsize=(16, 8))
    
    # Create color gradient from red (negative) to green (positive)
    colors = []
    for label in bin_labels:
        if "+" in label and label != "-5% to +5%":
            # Positive deviations - shades of red (overvalued)
            if ">+50%" in label or "+45%" in label or "+40%" in label:
                colors.append('darkred')
            elif "+35%" in label or "+30%" in label:
                colors.append('red')
            elif "+25%" in label or "+20%" in label:
                colors.append('lightcoral')
            else:
                colors.append('pink')
        elif "-" in label and label != "-5% to +5%":
            # Negative deviations - shades of green (undervalued)
            if "<-50%" in label or "-45%" in label or "-40%" in label:
                colors.append('darkgreen')
            elif "-35%" in label or "-30%" in label:
                colors.append('green')
            elif "-25%" in label or "-20%" in label:
                colors.append('lightgreen')
            else:
                colors.append('lightgray')
        else:
            # Normal range
            colors.append('gray')
    
    bars = ax.bar(range(len(bin_labels)), avg_gains, color=colors, alpha=0.7)
    ax.set_xticks(range(len(bin_labels)))
    ax.set_xticklabels(bin_labels, rotation=45, ha='right')
    ax.set_ylabel('Average 1-Year Forward Max Gain (%)')
    ax.set_title(f'{asset_name}: Average Forward Max Gains by IWLS Deviation Bin (5% increments)')
    ax.grid(True, alpha=0.3)
    
    # Add value labels on bars
    for i, (bar, gain, count) in enumerate(zip(bars, avg_gains, sample_counts)):
        ax.text(bar.get_x() + bar.get_width()/2., bar.get_height() + max(avg_gains)*0.01,
                f'{gain:.1f}%\n(n={count})', ha='center', va='bottom', fontsize=9, fontweight='bold')
    
    plt.tight_layout()
    plt.savefig(f"/Users/tim/IWLS-OPTIONS/{asset_name}_forward_gains.png", dpi=300, bbox_inches='tight')
    plt.close()

def extract_asset_name(column_name):
    """
    Extract clean asset name from column header
    """
    # Remove the close part and exchange info
    parts = column_name.split(' · ')
    if len(parts) > 1:
        return parts[0].strip()
    else:
        # Handle simple column names
        return column_name.replace(': close', '').strip()

def process_asset(data, asset_name):
    """
    Process a single asset
    """
    print(f"\n  Processing {asset_name}...")
    
    try:
        # Need at least 2000 days for meaningful analysis
        if len(data) < 2000:
            print(f"    Skipping {asset_name}: insufficient data ({len(data)} days)")
            return
        
        # Calculate IWLS growth rate
        print(f"    Calculating IWLS growth rates...")
        results = calculate_iwls_growth_rate(data, lookback_period=1500)
        
        # Save results
        output_file = f"/Users/tim/IWLS-OPTIONS/{asset_name}_iwls_results.csv"
        results.to_csv(output_file, index=False)
        
        # Create basic plots
        plot_basic_results(results, asset_name)
        
        # Analyze forward returns
        print(f"    Analyzing forward returns...")
        bin_stats, valid_data = analyze_forward_returns_by_deviation(results)
        
        # Print forward return results
        print(f"\n    === {asset_name}: 1-Year Forward Maximum Gain by Deviation Bin ===")
        print("    " + "-" * 60)
        
        bin_order = [">+50%", "+45% to +50%", "+40% to +45%", "+35% to +40%", "+30% to +35%", 
                     "+25% to +30%", "+20% to +25%", "+15% to +20%", "+10% to +15%", "+5% to +10%", 
                     "-5% to +5%", "-10% to -5%", "-15% to -10%", "-20% to -15%", "-25% to -20%", 
                     "-30% to -25%", "-35% to -30%", "-40% to -35%", "-45% to -40%", "-50% to -45%", "<-50%"]
        
        has_data = False
        for bin_name in bin_order:
            stats = bin_stats[bin_name]
            if stats['count'] > 0:
                has_data = True
                print(f"    {bin_name:>12}: {stats['avg_forward_max_gain']:6.1f}% average "
                      f"({stats['median_forward_max_gain']:5.1f}% median) "
                      f"[{stats['count']:3d} samples]")
        
        if not has_data:
            print(f"    No forward return data available for {asset_name}")
        
        # Plot forward gains
        plot_forward_gains(bin_stats, asset_name)
        
        # Print summary
        if len(valid_data) > 0:
            latest = valid_data.iloc[-1]
            current_deviation = ((latest['price'] / latest['trend_line_value']) - 1) * 100
            print(f"\n    Latest values for {asset_name}:")
            print(f"      Current price: ${latest['price']:.2f}")
            print(f"      Current deviation from trend: {current_deviation:.2f}%")
            print(f"      Current annual growth rate: {latest['annual_growth']:.2f}%")
        
        print(f"    Files saved: {asset_name}_iwls_results.csv, {asset_name}_iwls_analysis.png, {asset_name}_forward_gains.png")
        
    except Exception as e:
        print(f"    Error processing {asset_name}: {str(e)}")

def main():
    print("IWLS Growth Rate Analysis - Multiple Assets from Combined Dataset")
    print("=" * 70)
    
    # Look for CSV files in the directory, but exclude generated results files
    csv_files = glob.glob("/Users/tim/IWLS-OPTIONS/*.csv")
    csv_files = [f for f in csv_files if not f.endswith('_iwls_results.csv')]
    
    if not csv_files:
        print("No original CSV files found in /Users/tim/IWLS-OPTIONS/")
        return
    
    # Process each CSV file
    for file_path in csv_files:
        print(f"\nLoading {os.path.basename(file_path)}...")
        
        try:
            # Read the CSV file
            df = pd.read_csv(file_path)
            
            # Check if this is an original dataset (has 'time' column)
            if 'time' not in df.columns:
                print(f"  Skipping {os.path.basename(file_path)}: No 'time' column (appears to be a results file)")
                continue
            
            # Convert timestamp to datetime
            df['date'] = pd.to_datetime(df['time'], unit='s')
            
            # Sort by date to ensure proper order
            df = df.sort_values('date').reset_index(drop=True)
            
            print(f"Loaded {len(df)} data points from {df['date'].min().strftime('%Y-%m-%d')} to {df['date'].max().strftime('%Y-%m-%d')}")
            
            # Get all close price columns (excluding time and date)
            close_columns = [col for col in df.columns if 'close' in col.lower() and col != 'close']
            
            # If there's a 'close' column without asset name, that's probably the main asset
            if 'close' in df.columns:
                main_asset_name = os.path.basename(file_path).replace('.csv', '').replace('BATS_', '').replace(' 1D', '')
                asset_data = df[['date', 'close']].dropna()
                process_asset(asset_data, main_asset_name)
            
            # Process each asset column
            for col in close_columns:
                asset_name = extract_asset_name(col)
                
                # Create asset-specific dataframe
                asset_data = df[['date', col]].copy()
                asset_data = asset_data.rename(columns={col: 'close'})
                asset_data = asset_data.dropna()
                
                if len(asset_data) > 0:
                    process_asset(asset_data, asset_name)
            
        except Exception as e:
            print(f"Error loading {file_path}: {str(e)}")
    
    print(f"\nCompleted processing all assets. Results saved to /Users/tim/IWLS-OPTIONS/")

if __name__ == "__main__":
    main()