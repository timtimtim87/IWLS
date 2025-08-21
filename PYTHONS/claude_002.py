import pandas as pd
import numpy as np
import matplotlib.pyplot as plt
import seaborn as sns
from datetime import datetime, timedelta
import warnings
import os
import glob
from collections import defaultdict
from scipy import stats
warnings.filterwarnings('ignore')

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

def get_absolute_deviation_bin(abs_deviation):
    """
    Categorize absolute deviation magnitude into 5% bins
    """
    if abs_deviation >= 50:
        return ">50%"
    elif abs_deviation >= 45:
        return "45% to 50%"
    elif abs_deviation >= 40:
        return "40% to 45%"
    elif abs_deviation >= 35:
        return "35% to 40%"
    elif abs_deviation >= 30:
        return "30% to 35%"
    elif abs_deviation >= 25:
        return "25% to 30%"
    elif abs_deviation >= 20:
        return "20% to 25%"
    elif abs_deviation >= 15:
        return "15% to 20%"
    elif abs_deviation >= 10:
        return "10% to 15%"
    elif abs_deviation >= 5:
        return "5% to 10%"
    else:
        return "0% to 5%"

def get_zscore_bin(z_score):
    """
    Categorize Z-score into bins
    """
    if pd.isna(z_score):
        return "No Z-Score"
    elif z_score >= 3:
        return ">+3σ"
    elif z_score >= 2.5:
        return "+2.5σ to +3σ"
    elif z_score >= 2:
        return "+2σ to +2.5σ"
    elif z_score >= 1.5:
        return "+1.5σ to +2σ"
    elif z_score >= 1:
        return "+1σ to +1.5σ"
    elif z_score >= 0.5:
        return "+0.5σ to +1σ"
    elif z_score >= -0.5:
        return "-0.5σ to +0.5σ"
    elif z_score >= -1:
        return "-1σ to -0.5σ"
    elif z_score >= -1.5:
        return "-1.5σ to -1σ"
    elif z_score >= -2:
        return "-2σ to -1.5σ"
    elif z_score >= -2.5:
        return "-2.5σ to -2σ"
    elif z_score >= -3:
        return "-3σ to -2.5σ"
    else:
        return "<-3σ"

def load_asset_data(asset_name, v2_dir):
    """
    Load IWLS and Z-score data for a single asset
    """
    asset_dir = os.path.join(v2_dir, asset_name)
    
    # Load main IWLS results
    iwls_file = os.path.join(asset_dir, f"{asset_name}_iwls_results.csv")
    if not os.path.exists(iwls_file):
        return None, None
    
    iwls_df = pd.read_csv(iwls_file)
    iwls_df['date'] = pd.to_datetime(iwls_df['date'])
    iwls_df = iwls_df.sort_values('date').reset_index(drop=True)
    
    # Load Z-score data if available
    zscore_file = os.path.join(asset_dir, f"{asset_name}_zscore_analysis.csv")
    zscore_df = None
    if os.path.exists(zscore_file):
        zscore_df = pd.read_csv(zscore_file)
        zscore_df['date'] = pd.to_datetime(zscore_df['date'])
    
    return iwls_df, zscore_df

def calculate_forward_returns_single_asset(asset_name, iwls_df, zscore_df=None, forward_days=365):
    """
    Calculate 365-day forward maximum gains for a single asset
    """
    print(f"  Analyzing {asset_name}...")
    
    # Filter to only valid IWLS data (non-NaN deviations and prices)
    valid_data = iwls_df.dropna(subset=['price_deviation', 'price']).copy()
    
    if len(valid_data) < forward_days + 100:
        print(f"    Insufficient data for {asset_name} ({len(valid_data)} points)")
        return pd.DataFrame()
    
    # Merge with Z-score data if available
    if zscore_df is not None:
        valid_data = valid_data.merge(zscore_df[['date', 'z_score']], on='date', how='left')
    else:
        valid_data['z_score'] = np.nan
    
    forward_returns = []
    
    # Calculate forward returns for each valid entry point
    for i in range(len(valid_data) - forward_days):
        current_row = valid_data.iloc[i]
        entry_date = current_row['date']
        entry_price = current_row['price']
        price_deviation = current_row['price_deviation']
        z_score = current_row.get('z_score', np.nan)
        
        # Get future data for the next 365 calendar days
        end_date = entry_date + timedelta(days=forward_days)
        future_mask = (valid_data['date'] > entry_date) & (valid_data['date'] <= end_date)
        future_data = valid_data[future_mask]
        
        # Need at least 70% of expected trading days (roughly 255 trading days in 365 calendar days)
        min_required_days = int(forward_days * 0.5)  # More lenient: 50%
        
        if len(future_data) >= min_required_days:
            # Calculate forward return metrics
            future_prices = future_data['price']
            max_price = future_prices.max()
            min_price = future_prices.min()
            final_price = future_prices.iloc[-1]
            
            # Calculate returns
            max_gain = ((max_price / entry_price) - 1) * 100
            max_loss = ((min_price / entry_price) - 1) * 100
            final_return = ((final_price / entry_price) - 1) * 100
            
            # Find time to max gain
            max_idx = future_prices.idxmax()
            max_gain_date = future_data.loc[max_idx, 'date']
            days_to_max = (max_gain_date - entry_date).days
            
            # Categorize by bins
            deviation_bin = get_deviation_bin(price_deviation)
            abs_deviation_bin = get_absolute_deviation_bin(abs(price_deviation))
            zscore_bin = get_zscore_bin(z_score)
            
            forward_returns.append({
                'asset': asset_name,
                'entry_date': entry_date,
                'entry_price': entry_price,
                'price_deviation': price_deviation,
                'absolute_deviation': abs(price_deviation),
                'z_score': z_score,
                'deviation_bin': deviation_bin,
                'abs_deviation_bin': abs_deviation_bin,
                'zscore_bin': zscore_bin,
                'max_gain_365d': max_gain,
                'max_loss_365d': max_loss,
                'final_return_365d': final_return,
                'days_to_max_gain': days_to_max,
                'future_data_points': len(future_data)
            })
    
    result_df = pd.DataFrame(forward_returns)
    print(f"    Generated {len(result_df)} forward return samples")
    return result_df

def analyze_bins_performance(df, bin_column, bin_name):
    """
    Analyze performance by bins (deviation, absolute deviation, or Z-score)
    """
    if len(df) == 0:
        return pd.DataFrame()
    
    # Define bin orders for proper sorting
    if bin_column == 'deviation_bin':
        bin_order = [">+50%", "+45% to +50%", "+40% to +45%", "+35% to +40%", "+30% to +35%", 
                     "+25% to +30%", "+20% to +25%", "+15% to +20%", "+10% to +15%", "+5% to +10%", 
                     "-5% to +5%", "-10% to -5%", "-15% to -10%", "-20% to -15%", "-25% to -20%", 
                     "-30% to -25%", "-35% to -30%", "-40% to -35%", "-45% to -40%", "-50% to -45%", "<-50%"]
    elif bin_column == 'abs_deviation_bin':
        bin_order = ["0% to 5%", "5% to 10%", "10% to 15%", "15% to 20%", "20% to 25%", 
                     "25% to 30%", "30% to 35%", "35% to 40%", "40% to 45%", "45% to 50%", ">50%"]
    else:  # zscore_bin
        bin_order = [">+3σ", "+2.5σ to +3σ", "+2σ to +2.5σ", "+1.5σ to +2σ", "+1σ to +1.5σ", 
                     "+0.5σ to +1σ", "-0.5σ to +0.5σ", "-1σ to -0.5σ", "-1.5σ to -1σ", 
                     "-2σ to -1.5σ", "-2.5σ to -2σ", "-3σ to -2.5σ", "<-3σ", "No Z-Score"]
    
    analysis_results = []
    
    for bin_value in bin_order:
        bin_data = df[df[bin_column] == bin_value]
        
        if len(bin_data) >= 10:  # Minimum sample size
            max_gains = bin_data['max_gain_365d']
            final_returns = bin_data['final_return_365d']
            
            analysis_results.append({
                f'{bin_name}_bin': bin_value,
                'sample_count': len(bin_data),
                'avg_max_gain': max_gains.mean(),
                'median_max_gain': max_gains.median(),
                'std_max_gain': max_gains.std(),
                'min_max_gain': max_gains.min(),
                'max_max_gain': max_gains.max(),
                'q25_max_gain': max_gains.quantile(0.25),
                'q75_max_gain': max_gains.quantile(0.75),
                'avg_final_return': final_returns.mean(),
                'median_final_return': final_returns.median(),
                'success_rate_positive': (final_returns > 0).mean() * 100,
                'success_rate_10pct': (max_gains > 10).mean() * 100,
                'success_rate_25pct': (max_gains > 25).mean() * 100,
                'success_rate_50pct': (max_gains > 50).mean() * 100,
                'success_rate_100pct': (max_gains > 100).mean() * 100,
                'avg_max_loss': bin_data['max_loss_365d'].mean(),
                'worst_max_loss': bin_data['max_loss_365d'].min(),
                'avg_days_to_max': bin_data['days_to_max_gain'].mean(),
                'sharpe_ratio': final_returns.mean() / final_returns.std() if final_returns.std() > 0 else 0
            })
    
    return pd.DataFrame(analysis_results)

def create_asset_visualizations(df, asset_name, asset_dir):
    """
    Create comprehensive visualizations for a single asset
    """
    if len(df) < 50:
        print(f"    Insufficient data for {asset_name} visualization")
        return
    
    fig, ((ax1, ax2), (ax3, ax4)) = plt.subplots(2, 2, figsize=(20, 12))
    
    # Plot 1: Deviation vs Max Gain scatter
    scatter = ax1.scatter(df['price_deviation'], df['max_gain_365d'], 
                         c=df['days_to_max_gain'], cmap='viridis', alpha=0.6, s=30)
    ax1.set_xlabel('Price Deviation from Trend (%)')
    ax1.set_ylabel('365-Day Max Gain (%)')
    ax1.set_title(f'{asset_name}: Deviation vs Max Gain (colored by days to max)', fontweight='bold')
    ax1.grid(True, alpha=0.3)
    plt.colorbar(scatter, ax=ax1, label='Days to Max Gain')
    
    # Add trend line
    if len(df) > 10:
        z = np.polyfit(df['price_deviation'], df['max_gain_365d'], 1)
        p = np.poly1d(z)
        x_trend = np.linspace(df['price_deviation'].min(), df['price_deviation'].max(), 100)
        ax1.plot(x_trend, p(x_trend), "r--", alpha=0.8, linewidth=2, 
                label=f'Trend: {z[0]:.2f}x + {z[1]:.1f}')
        ax1.legend()
    
    # Plot 2: Max gain distribution
    ax2.hist(df['max_gain_365d'], bins=40, alpha=0.7, color='green', edgecolor='black')
    ax2.axvline(df['max_gain_365d'].mean(), color='red', linestyle='--', linewidth=2,
               label=f'Mean: {df["max_gain_365d"].mean():.1f}%')
    ax2.axvline(df['max_gain_365d'].median(), color='blue', linestyle='--', linewidth=2,
               label=f'Median: {df["max_gain_365d"].median():.1f}%')
    ax2.set_xlabel('365-Day Max Gain (%)')
    ax2.set_ylabel('Frequency')
    ax2.set_title(f'{asset_name}: Distribution of Max Gains', fontweight='bold')
    ax2.legend()
    ax2.grid(True, alpha=0.3)
    
    # Plot 3: Time series of entry points
    ax3.scatter(df['entry_date'], df['price_deviation'], 
               c=df['max_gain_365d'], cmap='RdYlGn', s=25, alpha=0.7)
    ax3.set_xlabel('Entry Date')
    ax3.set_ylabel('Price Deviation (%)')
    ax3.set_title(f'{asset_name}: Entry Points Over Time (colored by future max gain)', fontweight='bold')
    ax3.grid(True, alpha=0.3)
    ax3.axhline(y=0, color='black', linestyle='-', alpha=0.5)
    
    # Plot 4: Box plot of gains by deviation ranges
    # Create deviation ranges for box plot
    deviation_ranges = pd.cut(df['price_deviation'], bins=8, precision=1)
    box_data = []
    box_labels = []
    
    for range_val in deviation_ranges.unique():
        if pd.notna(range_val):
            range_data = df[deviation_ranges == range_val]['max_gain_365d']
            if len(range_data) >= 5:  # Minimum for meaningful box plot
                box_data.append(range_data.values)
                box_labels.append(f'{range_val.left:.1f} to {range_val.right:.1f}%')
    
    if box_data:
        bp = ax4.boxplot(box_data, labels=box_labels, patch_artist=True)
        for patch in bp['boxes']:
            patch.set_facecolor('lightblue')
            patch.set_alpha(0.7)
        
        ax4.set_xticklabels(box_labels, rotation=45, ha='right')
        ax4.set_ylabel('Max Gain Distribution (%)')
        ax4.set_xlabel('Deviation Range (%)')
        ax4.set_title(f'{asset_name}: Max Gain Distribution by Deviation Range', fontweight='bold')
        ax4.grid(True, alpha=0.3)
    
    plt.tight_layout()
    plt.savefig(f"{asset_dir}/{asset_name}_forward_returns_analysis.png", dpi=300, bbox_inches='tight')
    plt.close()

def process_single_asset(asset_name, v2_dir):
    """
    Process forward returns analysis for a single asset
    """
    print(f"\nProcessing {asset_name}...")
    
    # Load asset data
    iwls_df, zscore_df = load_asset_data(asset_name, v2_dir)
    if iwls_df is None:
        print(f"  ❌ Could not load data for {asset_name}")
        return False
    
    # Calculate forward returns
    forward_returns_df = calculate_forward_returns_single_asset(asset_name, iwls_df, zscore_df)
    
    if len(forward_returns_df) == 0:
        print(f"  ❌ No forward returns data generated for {asset_name}")
        return False
    
    # Asset directory
    asset_dir = os.path.join(v2_dir, asset_name)
    
    # Analyze by different bin types
    deviation_analysis = analyze_bins_performance(forward_returns_df, 'deviation_bin', 'deviation')
    abs_deviation_analysis = analyze_bins_performance(forward_returns_df, 'abs_deviation_bin', 'abs_deviation')
    zscore_analysis = analyze_bins_performance(
        forward_returns_df[forward_returns_df['zscore_bin'] != 'No Z-Score'], 
        'zscore_bin', 'zscore'
    )
    
    # Save results to asset folder
    forward_returns_df.to_csv(f"{asset_dir}/{asset_name}_forward_returns_365d.csv", index=False)
    
    if len(deviation_analysis) > 0:
        deviation_analysis.to_csv(f"{asset_dir}/{asset_name}_deviation_bins_forward_analysis.csv", index=False)
    
    if len(abs_deviation_analysis) > 0:
        abs_deviation_analysis.to_csv(f"{asset_dir}/{asset_name}_abs_deviation_bins_forward_analysis.csv", index=False)
    
    if len(zscore_analysis) > 0:
        zscore_analysis.to_csv(f"{asset_dir}/{asset_name}_zscore_bins_forward_analysis.csv", index=False)
    
    # Create visualizations
    create_asset_visualizations(forward_returns_df, asset_name, asset_dir)
    
    # Print summary
    print(f"  ✅ {asset_name} completed:")
    print(f"     • {len(forward_returns_df)} forward return samples")
    print(f"     • {len(deviation_analysis)} deviation bins with data")
    print(f"     • {len(abs_deviation_analysis)} absolute deviation bins with data")
    print(f"     • {len(zscore_analysis)} Z-score bins with data")
    print(f"     • Average max gain: {forward_returns_df['max_gain_365d'].mean():.2f}%")
    print(f"     • Success rate (25%+ gain): {(forward_returns_df['max_gain_365d'] > 25).mean()*100:.1f}%")
    
    return True

def create_summary_analysis(v2_dir):
    """
    Create cross-asset summary analysis from all individual results
    """
    print("\nCreating cross-asset summary analysis...")
    
    summary_dir = os.path.join(v2_dir, "FORWARD_RETURNS_SUMMARY")
    os.makedirs(summary_dir, exist_ok=True)
    
    all_forward_returns = []
    all_deviation_analysis = []
    all_abs_deviation_analysis = []
    all_zscore_analysis = []
    
    # Collect data from all asset folders
    for asset_name in os.listdir(v2_dir):
        asset_dir = os.path.join(v2_dir, asset_name)
        if not os.path.isdir(asset_dir) or asset_name == "FORWARD_RETURNS_SUMMARY":
            continue
        
        # Load forward returns data
        forward_file = os.path.join(asset_dir, f"{asset_name}_forward_returns_365d.csv")
        if os.path.exists(forward_file):
            try:
                df = pd.read_csv(forward_file)
                all_forward_returns.append(df)
            except Exception as e:
                print(f"  Warning: Could not load {forward_file}: {e}")
        
        # Load bin analyses
        for analysis_type, analysis_list in [
            ('deviation_bins_forward_analysis', all_deviation_analysis),
            ('abs_deviation_bins_forward_analysis', all_abs_deviation_analysis),
            ('zscore_bins_forward_analysis', all_zscore_analysis)
        ]:
            analysis_file = os.path.join(asset_dir, f"{asset_name}_{analysis_type}.csv")
            if os.path.exists(analysis_file):
                try:
                    df = pd.read_csv(analysis_file)
                    df['asset'] = asset_name
                    analysis_list.append(df)
                except Exception as e:
                    print(f"  Warning: Could not load {analysis_file}: {e}")
    
    # Combine and save summary data
    if all_forward_returns:
        combined_forward_returns = pd.concat(all_forward_returns, ignore_index=True)
        combined_forward_returns.to_csv(f"{summary_dir}/ALL_ASSETS_forward_returns_365d.csv", index=False)
        print(f"  ✅ Combined forward returns: {len(combined_forward_returns):,} samples")
        
        # Overall group analysis by bin types
        group_deviation_analysis = analyze_bins_performance(combined_forward_returns, 'deviation_bin', 'deviation')
        group_abs_deviation_analysis = analyze_bins_performance(combined_forward_returns, 'abs_deviation_bin', 'abs_deviation')
        group_zscore_analysis = analyze_bins_performance(
            combined_forward_returns[combined_forward_returns['zscore_bin'] != 'No Z-Score'], 
            'zscore_bin', 'zscore'
        )
        
        # Save group analyses
        if len(group_deviation_analysis) > 0:
            group_deviation_analysis.to_csv(f"{summary_dir}/GROUP_deviation_bins_analysis.csv", index=False)
        if len(group_abs_deviation_analysis) > 0:
            group_abs_deviation_analysis.to_csv(f"{summary_dir}/GROUP_abs_deviation_bins_analysis.csv", index=False)
        if len(group_zscore_analysis) > 0:
            group_zscore_analysis.to_csv(f"{summary_dir}/GROUP_zscore_bins_analysis.csv", index=False)
        
        # Asset performance summary
        asset_summary = combined_forward_returns.groupby('asset').agg({
            'max_gain_365d': ['count', 'mean', 'median', 'std'],
            'final_return_365d': ['mean', 'median'],
            'days_to_max_gain': 'mean',
            'price_deviation': ['mean', 'std'],
            'absolute_deviation': 'mean'
        }).round(3)
        
        asset_summary.columns = ['_'.join(col).strip() for col in asset_summary.columns]
        asset_summary = asset_summary.reset_index()
        
        # Add success rates
        success_rates = combined_forward_returns.groupby('asset').apply(
            lambda x: pd.Series({
                'success_rate_positive': (x['final_return_365d'] > 0).mean() * 100,
                'success_rate_25pct': (x['max_gain_365d'] > 25).mean() * 100,
                'success_rate_50pct': (x['max_gain_365d'] > 50).mean() * 100,
                'success_rate_100pct': (x['max_gain_365d'] > 100).mean() * 100
            })
        ).round(2)
        
        asset_summary = asset_summary.merge(success_rates, left_on='asset', right_index=True)
        asset_summary.to_csv(f"{summary_dir}/ASSET_PERFORMANCE_COMPARISON.csv", index=False)
        
        print(f"  ✅ Asset performance summary: {len(asset_summary)} assets")
        
        # Create summary visualizations
        create_summary_visualizations(group_deviation_analysis, group_abs_deviation_analysis, 
                                    group_zscore_analysis, asset_summary, summary_dir)
    
    return summary_dir

def create_summary_visualizations(deviation_analysis, abs_deviation_analysis, zscore_analysis, asset_summary, output_dir):
    """
    Create summary visualizations across all assets
    """
    print("  Creating summary visualizations...")
    
    # Figure 1: Deviation bins analysis
    if len(deviation_analysis) > 0:
        fig, ((ax1, ax2), (ax3, ax4)) = plt.subplots(2, 2, figsize=(20, 12))
        
        # Average max gains by deviation bin
        colors = ['darkred' if '+' in bin_name and bin_name != '-5% to +5%' 
                 else 'darkgreen' if '-' in bin_name and bin_name != '-5% to +5%' 
                 else 'gray' for bin_name in deviation_analysis['deviation_bin']]
        
        bars1 = ax1.bar(range(len(deviation_analysis)), deviation_analysis['avg_max_gain'], 
                       color=colors, alpha=0.8)
        ax1.set_xticks(range(len(deviation_analysis)))
        ax1.set_xticklabels(deviation_analysis['deviation_bin'], rotation=45, ha='right')
        ax1.set_ylabel('Average Max Gain (%)')
        ax1.set_title('365-Day Average Maximum Gains by Deviation Bin (All Assets)', fontweight='bold')
        ax1.grid(True, alpha=0.3)
        
        # Add value labels
        for i, (bar, value, count) in enumerate(zip(bars1, deviation_analysis['avg_max_gain'], 
                                                   deviation_analysis['sample_count'])):
            ax1.text(bar.get_x() + bar.get_width()/2., bar.get_height() + max(deviation_analysis['avg_max_gain'])*0.01,
                    f'{value:.1f}%\n(n={count:,})', ha='center', va='bottom', fontsize=8, fontweight='bold')
        
        # Success rates
        width = 0.25
        x = np.arange(len(deviation_analysis))
        ax2.bar(x - width, deviation_analysis['success_rate_25pct'], width, label='25%+ Gains', alpha=0.8, color='lightgreen')
        ax2.bar(x, deviation_analysis['success_rate_50pct'], width, label='50%+ Gains', alpha=0.8, color='green')
        ax2.bar(x + width, deviation_analysis['success_rate_100pct'], width, label='100%+ Gains', alpha=0.8, color='darkgreen')
        
        ax2.set_xticks(x)
        ax2.set_xticklabels(deviation_analysis['deviation_bin'], rotation=45, ha='right')
        ax2.set_ylabel('Success Rate (%)')
        ax2.set_title('Success Rates by Deviation Bin', fontweight='bold')
        ax2.legend()
        ax2.grid(True, alpha=0.3)
        
        # Risk vs Return
        ax3.scatter(deviation_analysis['avg_max_loss'], deviation_analysis['avg_max_gain'],
                   c=deviation_analysis['sample_count'], cmap='viridis', s=100, alpha=0.7)
        ax3.set_xlabel('Average Max Loss (%)')
        ax3.set_ylabel('Average Max Gain (%)')
        ax3.set_title('Risk vs Return by Deviation Bin', fontweight='bold')
        ax3.grid(True, alpha=0.3)
        
        # Sharpe ratios
        colors_sharpe = ['green' if sr > 0.5 else 'orange' if sr > 0 else 'red' for sr in deviation_analysis['sharpe_ratio']]
        bars4 = ax4.bar(range(len(deviation_analysis)), deviation_analysis['sharpe_ratio'], 
                       color=colors_sharpe, alpha=0.8)
        ax4.set_xticks(range(len(deviation_analysis)))
        ax4.set_xticklabels(deviation_analysis['deviation_bin'], rotation=45, ha='right')
        ax4.set_ylabel('Sharpe Ratio')
        ax4.set_title('Risk-Adjusted Returns by Deviation Bin', fontweight='bold')
        ax4.axhline(y=0, color='black', linestyle='-', alpha=0.5)
        ax4.grid(True, alpha=0.3)
        
        plt.tight_layout()
        plt.savefig(f"{output_dir}/GROUP_deviation_bins_analysis.png", dpi=300, bbox_inches='tight')
        plt.close()
    
    # Figure 2: Absolute deviation bins analysis
    if len(abs_deviation_analysis) > 0:
        fig, ((ax1, ax2), (ax3, ax4)) = plt.subplots(2, 2, figsize=(20, 12))
        
        # Average max gains by absolute deviation bin
        bars1 = ax1.bar(range(len(abs_deviation_analysis)), abs_deviation_analysis['avg_max_gain'], 
                       color='steelblue', alpha=0.8)
        ax1.set_xticks(range(len(abs_deviation_analysis)))
        ax1.set_xticklabels(abs_deviation_analysis['abs_deviation_bin'], rotation=45, ha='right')
        ax1.set_ylabel('Average Max Gain (%)')
        ax1.set_title('365-Day Average Maximum Gains by Absolute Deviation Magnitude', fontweight='bold')
        ax1.grid(True, alpha=0.3)
        
        # Add value labels
        for i, (bar, value, count) in enumerate(zip(bars1, abs_deviation_analysis['avg_max_gain'], 
                                                   abs_deviation_analysis['sample_count'])):
            ax1.text(bar.get_x() + bar.get_width()/2., bar.get_height() + max(abs_deviation_analysis['avg_max_gain'])*0.01,
                    f'{value:.1f}%\n(n={count:,})', ha='center', va='bottom', fontsize=8, fontweight='bold')
        
        # Success rates for absolute deviation
        width = 0.25
        x = np.arange(len(abs_deviation_analysis))
        ax2.bar(x - width, abs_deviation_analysis['success_rate_25pct'], width, label='25%+ Gains', alpha=0.8, color='lightcoral')
        ax2.bar(x, abs_deviation_analysis['success_rate_50pct'], width, label='50%+ Gains', alpha=0.8, color='red')
        ax2.bar(x + width, abs_deviation_analysis['success_rate_100pct'], width, label='100%+ Gains', alpha=0.8, color='darkred')
        
        ax2.set_xticks(x)
        ax2.set_xticklabels(abs_deviation_analysis['abs_deviation_bin'], rotation=45, ha='right')
        ax2.set_ylabel('Success Rate (%)')
        ax2.set_title('Success Rates by Absolute Deviation Magnitude', fontweight='bold')
        ax2.legend()
        ax2.grid(True, alpha=0.3)
        
        # Sample count distribution
        bars3 = ax3.bar(range(len(abs_deviation_analysis)), abs_deviation_analysis['sample_count'], 
                       color='orange', alpha=0.8)
        ax3.set_xticks(range(len(abs_deviation_analysis)))
        ax3.set_xticklabels(abs_deviation_analysis['abs_deviation_bin'], rotation=45, ha='right')
        ax3.set_ylabel('Sample Count')
        ax3.set_title('Sample Distribution by Absolute Deviation Magnitude', fontweight='bold')
        ax3.grid(True, alpha=0.3)
        
        # Time to max gain
        bars4 = ax4.bar(range(len(abs_deviation_analysis)), abs_deviation_analysis['avg_days_to_max'], 
                       color='purple', alpha=0.8)
        ax4.set_xticks(range(len(abs_deviation_analysis)))
        ax4.set_xticklabels(abs_deviation_analysis['abs_deviation_bin'], rotation=45, ha='right')
        ax4.set_ylabel('Average Days to Max Gain')
        ax4.set_title('Time to Maximum Gain by Absolute Deviation Magnitude', fontweight='bold')
        ax4.grid(True, alpha=0.3)
        
        plt.tight_layout()
        plt.savefig(f"{output_dir}/GROUP_abs_deviation_bins_analysis.png", dpi=300, bbox_inches='tight')
        plt.close()
    
    # Figure 3: Z-score bins analysis (if data available)
    if len(zscore_analysis) > 0:
        fig, ((ax1, ax2), (ax3, ax4)) = plt.subplots(2, 2, figsize=(20, 12))
        
        # Average max gains by Z-score bin
        colors = ['darkred' if '+' in bin_name else 'darkgreen' if '-' in bin_name else 'gray' 
                 for bin_name in zscore_analysis['zscore_bin']]
        bars1 = ax1.bar(range(len(zscore_analysis)), zscore_analysis['avg_max_gain'], 
                       color=colors, alpha=0.8)
        ax1.set_xticks(range(len(zscore_analysis)))
        ax1.set_xticklabels(zscore_analysis['zscore_bin'], rotation=45, ha='right')
        ax1.set_ylabel('Average Max Gain (%)')
        ax1.set_title('365-Day Average Maximum Gains by Z-Score Bin', fontweight='bold')
        ax1.grid(True, alpha=0.3)
        
        # Add value labels
        for i, (bar, value, count) in enumerate(zip(bars1, zscore_analysis['avg_max_gain'], 
                                                   zscore_analysis['sample_count'])):
            ax1.text(bar.get_x() + bar.get_width()/2., bar.get_height() + max(zscore_analysis['avg_max_gain'])*0.01,
                    f'{value:.1f}%\n(n={count:,})', ha='center', va='bottom', fontsize=8, fontweight='bold')
        
        # Success rates for Z-score
        width = 0.25
        x = np.arange(len(zscore_analysis))
        ax2.bar(x - width, zscore_analysis['success_rate_25pct'], width, label='25%+ Gains', alpha=0.8, color='lightblue')
        ax2.bar(x, zscore_analysis['success_rate_50pct'], width, label='50%+ Gains', alpha=0.8, color='blue')
        ax2.bar(x + width, zscore_analysis['success_rate_100pct'], width, label='100%+ Gains', alpha=0.8, color='darkblue')
        
        ax2.set_xticks(x)
        ax2.set_xticklabels(zscore_analysis['zscore_bin'], rotation=45, ha='right')
        ax2.set_ylabel('Success Rate (%)')
        ax2.set_title('Success Rates by Z-Score Bin', fontweight='bold')
        ax2.legend()
        ax2.grid(True, alpha=0.3)
        
        # Sharpe ratios for Z-score
        colors_sharpe = ['green' if sr > 0.5 else 'orange' if sr > 0 else 'red' for sr in zscore_analysis['sharpe_ratio']]
        bars3 = ax3.bar(range(len(zscore_analysis)), zscore_analysis['sharpe_ratio'], 
                       color=colors_sharpe, alpha=0.8)
        ax3.set_xticks(range(len(zscore_analysis)))
        ax3.set_xticklabels(zscore_analysis['zscore_bin'], rotation=45, ha='right')
        ax3.set_ylabel('Sharpe Ratio')
        ax3.set_title('Risk-Adjusted Returns by Z-Score Bin', fontweight='bold')
        ax3.axhline(y=0, color='black', linestyle='-', alpha=0.5)
        ax3.grid(True, alpha=0.3)
        
        # Sample count for Z-score
        bars4 = ax4.bar(range(len(zscore_analysis)), zscore_analysis['sample_count'], 
                       color='teal', alpha=0.8)
        ax4.set_xticks(range(len(zscore_analysis)))
        ax4.set_xticklabels(zscore_analysis['zscore_bin'], rotation=45, ha='right')
        ax4.set_ylabel('Sample Count')
        ax4.set_title('Sample Distribution by Z-Score Bin', fontweight='bold')
        ax4.grid(True, alpha=0.3)
        
        plt.tight_layout()
        plt.savefig(f"{output_dir}/GROUP_zscore_bins_analysis.png", dpi=300, bbox_inches='tight')
        plt.close()
    
    # Figure 4: Asset performance comparison
    if len(asset_summary) > 0:
        fig, ((ax1, ax2), (ax3, ax4)) = plt.subplots(2, 2, figsize=(20, 12))
        
        # Filter for assets with meaningful sample sizes
        significant_assets = asset_summary[asset_summary['max_gain_365d_count'] >= 100].copy()
        if len(significant_assets) > 20:
            significant_assets = significant_assets.nlargest(20, 'max_gain_365d_mean')
        
        if len(significant_assets) > 0:
            # Average max gain by asset
            bars1 = ax1.bar(range(len(significant_assets)), significant_assets['max_gain_365d_mean'], 
                           color='forestgreen', alpha=0.8)
            ax1.set_xticks(range(len(significant_assets)))
            ax1.set_xticklabels(significant_assets['asset'], rotation=45, ha='right')
            ax1.set_ylabel('Average Max Gain (%)')
            ax1.set_title('Top Assets by Average 365-Day Max Gain', fontweight='bold')
            ax1.grid(True, alpha=0.3)
            
            # Success rate comparison
            ax2.scatter(significant_assets['max_gain_365d_mean'], significant_assets['success_rate_25pct'],
                       s=significant_assets['max_gain_365d_count']/10, alpha=0.7, color='blue')
            ax2.set_xlabel('Average Max Gain (%)')
            ax2.set_ylabel('Success Rate for 25%+ Gains (%)')
            ax2.set_title('Average Gain vs Success Rate (bubble size = sample count)', fontweight='bold')
            ax2.grid(True, alpha=0.3)
            
            # Sample count distribution
            bars3 = ax3.bar(range(len(significant_assets)), significant_assets['max_gain_365d_count'], 
                           color='coral', alpha=0.8)
            ax3.set_xticks(range(len(significant_assets)))
            ax3.set_xticklabels(significant_assets['asset'], rotation=45, ha='right')
            ax3.set_ylabel('Sample Count')
            ax3.set_title('Sample Count by Asset', fontweight='bold')
            ax3.grid(True, alpha=0.3)
            
            # Risk vs Return by asset
            ax4.scatter(significant_assets['max_gain_365d_std'], significant_assets['max_gain_365d_mean'],
                       s=significant_assets['max_gain_365d_count']/10, alpha=0.7, color='purple')
            ax4.set_xlabel('Standard Deviation of Max Gains (%)')
            ax4.set_ylabel('Average Max Gain (%)')
            ax4.set_title('Risk vs Return by Asset (bubble size = sample count)', fontweight='bold')
            ax4.grid(True, alpha=0.3)
        
        plt.tight_layout()
        plt.savefig(f"{output_dir}/ASSET_PERFORMANCE_COMPARISON.png", dpi=300, bbox_inches='tight')
        plt.close()
    
    print("  ✅ Summary visualizations created")

def print_comprehensive_summary(summary_dir):
    """
    Print comprehensive summary of the analysis results
    """
    print("\n" + "="*80)
    print("365-DAY FORWARD RETURNS ANALYSIS - COMPREHENSIVE SUMMARY")
    print("="*80)
    
    # Load summary data
    combined_file = os.path.join(summary_dir, "ALL_ASSETS_forward_returns_365d.csv")
    if not os.path.exists(combined_file):
        print("No combined data found for summary")
        return
    
    combined_df = pd.read_csv(combined_file)
    
    print(f"\nDATASET OVERVIEW:")
    print(f"  Total forward return samples: {len(combined_df):,}")
    print(f"  Unique assets analyzed: {combined_df['asset'].nunique()}")
    print(f"  Average samples per asset: {len(combined_df) / combined_df['asset'].nunique():.0f}")
    
    print(f"\nOVERALL PERFORMANCE METRICS:")
    print(f"  Average max gain: {combined_df['max_gain_365d'].mean():.2f}%")
    print(f"  Median max gain: {combined_df['max_gain_365d'].median():.2f}%")
    print(f"  Standard deviation: {combined_df['max_gain_365d'].std():.2f}%")
    print(f"  Average final return: {combined_df['final_return_365d'].mean():.2f}%")
    
    print(f"\nSUCCESS RATES:")
    print(f"  Positive final return: {(combined_df['final_return_365d'] > 0).mean()*100:.1f}%")
    print(f"  10%+ max gain: {(combined_df['max_gain_365d'] > 10).mean()*100:.1f}%")
    print(f"  25%+ max gain: {(combined_df['max_gain_365d'] > 25).mean()*100:.1f}%")
    print(f"  50%+ max gain: {(combined_df['max_gain_365d'] > 50).mean()*100:.1f}%")
    print(f"  100%+ max gain: {(combined_df['max_gain_365d'] > 100).mean()*100:.1f}%")
    
    # Deviation analysis summary
    deviation_file = os.path.join(summary_dir, "GROUP_deviation_bins_analysis.csv")
    if os.path.exists(deviation_file):
        deviation_df = pd.read_csv(deviation_file)
        print(f"\nTOP DEVIATION BINS BY AVERAGE MAX GAIN:")
        print("-" * 60)
        top_deviation_bins = deviation_df.nlargest(5, 'avg_max_gain')
        for _, row in top_deviation_bins.iterrows():
            print(f"  {row['deviation_bin']:<15}: {row['avg_max_gain']:>7.1f}% avg gain ({row['sample_count']:,} samples)")
    
    # Absolute deviation analysis summary
    abs_deviation_file = os.path.join(summary_dir, "GROUP_abs_deviation_bins_analysis.csv")
    if os.path.exists(abs_deviation_file):
        abs_deviation_df = pd.read_csv(abs_deviation_file)
        print(f"\nTOP ABSOLUTE DEVIATION BINS BY AVERAGE MAX GAIN:")
        print("-" * 60)
        top_abs_bins = abs_deviation_df.nlargest(5, 'avg_max_gain')
        for _, row in top_abs_bins.iterrows():
            print(f"  {row['abs_deviation_bin']:<15}: {row['avg_max_gain']:>7.1f}% avg gain ({row['sample_count']:,} samples)")
    
    # Z-score analysis summary
    zscore_file = os.path.join(summary_dir, "GROUP_zscore_bins_analysis.csv")
    if os.path.exists(zscore_file):
        zscore_df = pd.read_csv(zscore_file)
        print(f"\nTOP Z-SCORE BINS BY AVERAGE MAX GAIN:")
        print("-" * 60)
        top_zscore_bins = zscore_df.nlargest(5, 'avg_max_gain')
        for _, row in top_zscore_bins.iterrows():
            print(f"  {row['zscore_bin']:<15}: {row['avg_max_gain']:>7.1f}% avg gain ({row['sample_count']:,} samples)")
    
    # Asset performance summary
    asset_file = os.path.join(summary_dir, "ASSET_PERFORMANCE_COMPARISON.csv")
    if os.path.exists(asset_file):
        asset_df = pd.read_csv(asset_file)
        significant_assets = asset_df[asset_df['max_gain_365d_count'] >= 100]
        print(f"\nTOP 10 ASSETS BY AVERAGE MAX GAIN (min 100 samples):")
        print("-" * 70)
        print(f"{'Asset':<10} {'Avg Gain':<10} {'Success Rate':<12} {'Samples':<8}")
        print("-" * 70)
        top_assets = significant_assets.nlargest(10, 'max_gain_365d_mean')
        for _, row in top_assets.iterrows():
            print(f"{row['asset']:<10} {row['max_gain_365d_mean']:>9.1f}% {row['success_rate_25pct']:>11.1f}% {row['max_gain_365d_count']:>7.0f}")
    
    print(f"\nKEY INSIGHTS:")
    print("-" * 40)
    
    # Mean reversion vs momentum analysis
    undervalued = combined_df[combined_df['price_deviation'] < -10]
    overvalued = combined_df[combined_df['price_deviation'] > 10]
    
    if len(undervalued) > 0 and len(overvalued) > 0:
        undervalued_gain = undervalued['max_gain_365d'].mean()
        overvalued_gain = overvalued['max_gain_365d'].mean()
        
        print(f"  Undervalued assets (deviation < -10%): {undervalued_gain:.1f}% avg max gain")
        print(f"  Overvalued assets (deviation > +10%): {overvalued_gain:.1f}% avg max gain")
        print(f"  Undervalued advantage: {undervalued_gain - overvalued_gain:+.1f}%")
        
        if undervalued_gain > overvalued_gain:
            print(f"  ✅ Mean reversion pattern confirmed!")
        else:
            print(f"  ⚠️  Momentum pattern detected!")
    
    # Time analysis
    avg_days_to_max = combined_df['days_to_max_gain'].mean()
    print(f"  Average time to maximum gain: {avg_days_to_max:.0f} days")
    
    if avg_days_to_max < 180:
        print(f"  ⚡ Quick gains pattern - consider shorter holding periods")
    else:
        print(f"  🕐 Patient gains pattern - longer holding periods optimal")

def main():
    print("IWLS Forward Returns Analysis - 365 Day Maximum Gains")
    print("=" * 70)
    print("Analyzing future returns based on IWLS deviation and Z-score bins")
    print("Features:")
    print("  • 5% deviation bins (signed)")
    print("  • 5% absolute deviation magnitude bins")  
    print("  • Z-score standard deviation bins")
    print("  • Individual asset files saved to existing folders")
    print("  • Group summary analysis in FORWARD_RETURNS_SUMMARY folder")
    
    # Check for V2 directory
    v2_dir = "/Users/tim/IWLS-OPTIONS/IWLS_ANALYSIS_V2"
    
    if not os.path.exists(v2_dir):
        print("❌ IWLS_ANALYSIS_V2 directory not found. Run the IWLS V2 analysis first.")
        return
    
    print(f"\nUsing base directory: {v2_dir}")
    
    # Get list of asset directories
    asset_dirs = [d for d in os.listdir(v2_dir) 
                  if os.path.isdir(os.path.join(v2_dir, d)) and d != "FORWARD_RETURNS_SUMMARY"]
    
    print(f"Found {len(asset_dirs)} asset directories to process")
    
    # Process each asset
    successful_assets = 0
    failed_assets = 0
    
    for i, asset_name in enumerate(asset_dirs):
        print(f"\n[{i+1}/{len(asset_dirs)}] Processing {asset_name}...")
        
        success = process_single_asset(asset_name, v2_dir)
        if success:
            successful_assets += 1
        else:
            failed_assets += 1
    
    print(f"\n" + "="*50)
    print(f"INDIVIDUAL ASSET PROCESSING COMPLETE")
    print(f"Successful: {successful_assets}")
    print(f"Failed: {failed_assets}")
    print(f"Success rate: {successful_assets/(successful_assets+failed_assets)*100:.1f}%")
    
    # Create cross-asset summary analysis
    summary_dir = create_summary_analysis(v2_dir)
    
    # Print comprehensive summary
    print_comprehensive_summary(summary_dir)
    
    print(f"\n" + "="*70)
    print("FORWARD RETURNS ANALYSIS COMPLETE")
    print("="*70)
    print(f"Individual results saved to: [ASSET_NAME]/ folders")
    print(f"Group summary saved to: {summary_dir}")
    print("\nFiles created per asset:")
    print("  📄 [ASSET]_forward_returns_365d.csv (raw forward return data)")
    print("  📄 [ASSET]_deviation_bins_forward_analysis.csv (signed deviation bins)")
    print("  📄 [ASSET]_abs_deviation_bins_forward_analysis.csv (absolute magnitude bins)")
    print("  📄 [ASSET]_zscore_bins_forward_analysis.csv (Z-score bins)")
    print("  📊 [ASSET]_forward_returns_analysis.png (visualizations)")
    
    print("\nGroup summary files:")
    print("  📄 ALL_ASSETS_forward_returns_365d.csv (combined raw data)")
    print("  📄 GROUP_deviation_bins_analysis.csv (group deviation analysis)")
    print("  📄 GROUP_abs_deviation_bins_analysis.csv (group absolute deviation analysis)")
    print("  📄 GROUP_zscore_bins_analysis.csv (group Z-score analysis)")
    print("  📄 ASSET_PERFORMANCE_COMPARISON.csv (cross-asset comparison)")
    print("  📊 GROUP_deviation_bins_analysis.png (group deviation visualizations)")
    print("  📊 GROUP_abs_deviation_bins_analysis.png (group absolute deviation visualizations)")
    print("  📊 GROUP_zscore_bins_analysis.png (group Z-score visualizations)")
    print("  📊 ASSET_PERFORMANCE_COMPARISON.png (asset comparison visualizations)")
    
    print(f"\n🎯 Analysis complete! Use the bin analysis files to:")
    print(f"   • Identify optimal entry signals based on deviation levels")
    print(f"   • Compare signed vs absolute deviation patterns")
    print(f"   • Analyze normalized Z-score performance")
    print(f"   • Select best performing assets and deviation ranges")

if __name__ == "__main__":
    main()