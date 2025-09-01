import pandas as pd
import numpy as np
import matplotlib.pyplot as plt
import seaborn as sns
from datetime import datetime, timedelta
import warnings
import os
from collections import defaultdict
warnings.filterwarnings('ignore')

def load_all_iwls_data(v2_dir):
    """
    Load IWLS data for all assets with growth rates
    """
    print("Loading IWLS data for all assets...")
    
    all_data = {}
    asset_dirs = [d for d in os.listdir(v2_dir) 
                  if os.path.isdir(os.path.join(v2_dir, d)) and 
                  not d.startswith('FORWARD_RETURNS') and not d.startswith('OLD') and
                  not d.startswith('REBALANCING') and not d.startswith('GROWTH_RATE') and
                  not d.startswith('DYNAMIC') and not d.startswith('PORTFOLIO') and
                  not d.startswith('DAILY_TRACKING')]
    
    for asset_name in asset_dirs:
        asset_dir = os.path.join(v2_dir, asset_name)
        iwls_file = os.path.join(asset_dir, f"{asset_name}_iwls_results.csv")
        
        if os.path.exists(iwls_file):
            try:
                df = pd.read_csv(iwls_file)
                df['date'] = pd.to_datetime(df['date'])
                # Keep all rows with annual_growth data, not just non-NaN price_deviation
                df = df.dropna(subset=['annual_growth', 'price']).sort_values('date').reset_index(drop=True)
                
                if len(df) > 500:  # Need sufficient data
                    all_data[asset_name] = df
                    print(f"  ✅ {asset_name}: {len(df)} data points")
                    
            except Exception as e:
                print(f"  ⚠️  Error loading {asset_name}: {e}")
    
    print(f"✅ Successfully loaded {len(all_data)} assets")
    return all_data

def categorize_growth_rate(growth_rate):
    """
    Categorize growth rate into 5% bins
    """
    if pd.isna(growth_rate):
        return "Unknown"
    
    # Handle extreme values
    if growth_rate < -50:
        return "<-50%"
    elif growth_rate > 100:
        return ">100%"
    
    # Create 5% bins
    bin_start = int(growth_rate // 5) * 5
    bin_end = bin_start + 5
    
    return f"{bin_start}% to {bin_end}%"

def determine_growth_trend(current_growth, previous_growth, lookback_periods=5):
    """
    Determine if growth rate is trending UP or DOWN
    Uses comparison with previous growth rates
    """
    if pd.isna(current_growth) or pd.isna(previous_growth):
        return "UNKNOWN"
    
    if current_growth > previous_growth:
        return "UP"
    elif current_growth < previous_growth:
        return "DOWN"
    else:
        return "FLAT"

def calculate_forward_max_gains(all_data, forward_days=365):
    """
    Calculate forward maximum gains for each data point in each asset
    """
    print(f"\nCalculating forward {forward_days}-day maximum gains for all assets...")
    
    all_forward_gains = []
    
    for asset_name, df in all_data.items():
        print(f"  Processing {asset_name}...")
        
        # Calculate growth rate trend (comparing to previous 5-day average)
        df['previous_growth_5d'] = df['annual_growth'].rolling(window=5, min_periods=1).mean().shift(1)
        
        for i in range(len(df) - forward_days):
            current_row = df.iloc[i]
            entry_date = current_row['date']
            entry_price = current_row['price']
            current_growth = current_row['annual_growth']
            previous_growth = current_row['previous_growth_5d']
            
            # Get future price data
            future_data = df.iloc[i+1:i+forward_days+1]
            
            if len(future_data) >= int(forward_days * 0.8):  # Need at least 80% of forward data
                # Calculate maximum gain in the forward period
                max_price = future_data['price'].max()
                max_gain = ((max_price / entry_price) - 1) * 100
                
                # Calculate final return
                final_price = future_data['price'].iloc[-1]
                final_return = ((final_price / entry_price) - 1) * 100
                
                # Categorize growth rate and trend
                growth_bin = categorize_growth_rate(current_growth)
                growth_trend = determine_growth_trend(current_growth, previous_growth)
                
                # Create combined bin (growth rate + trend)
                if growth_trend in ['UP', 'DOWN']:
                    combined_bin = f"{growth_bin} {growth_trend}"
                else:
                    combined_bin = f"{growth_bin} FLAT"
                
                # Find time to maximum gain
                max_idx = future_data['price'].idxmax()
                max_gain_date = future_data.loc[max_idx, 'date']
                days_to_max = (max_gain_date - entry_date).days
                
                all_forward_gains.append({
                    'asset': asset_name,
                    'entry_date': entry_date,
                    'entry_price': entry_price,
                    'annual_growth_rate': current_growth,
                    'previous_growth_5d': previous_growth,
                    'growth_rate_bin': growth_bin,
                    'growth_trend': growth_trend,
                    'combined_bin': combined_bin,
                    'forward_max_gain': max_gain,
                    'forward_final_return': final_return,
                    'days_to_max_gain': days_to_max,
                    'future_data_points': len(future_data)
                })
        
        print(f"    Completed {asset_name}: {len(df) - forward_days} forward return samples")
    
    forward_gains_df = pd.DataFrame(all_forward_gains)
    print(f"\n✅ Generated {len(forward_gains_df):,} total forward gain samples across all assets")
    
    return forward_gains_df

def analyze_growth_rate_bins(forward_gains_df):
    """
    Analyze forward gains by growth rate bins (without trend)
    """
    print("\nAnalyzing performance by growth rate bins...")
    
    # Define proper bin order
    bin_order = ["<-50%"]
    
    # Add bins from -50% to 100% in 5% increments
    for start in range(-50, 100, 5):
        end = start + 5
        bin_order.append(f"{start}% to {end}%")
    
    bin_order.append(">100%")
    
    bin_analysis = []
    
    for bin_name in bin_order:
        bin_data = forward_gains_df[forward_gains_df['growth_rate_bin'] == bin_name]
        
        if len(bin_data) >= 10:  # Minimum sample size
            max_gains = bin_data['forward_max_gain']
            final_returns = bin_data['forward_final_return']
            
            bin_analysis.append({
                'growth_rate_bin': bin_name,
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
                'avg_days_to_max': bin_data['days_to_max_gain'].mean(),
                'median_days_to_max': bin_data['days_to_max_gain'].median(),
                'avg_growth_rate': bin_data['annual_growth_rate'].mean(),
                'volatility': max_gains.std()
            })
    
    bin_analysis_df = pd.DataFrame(bin_analysis)
    print(f"✅ Analyzed {len(bin_analysis_df)} growth rate bins with sufficient data")
    
    return bin_analysis_df

def analyze_growth_trend_bins(forward_gains_df):
    """
    Analyze forward gains by combined growth rate + trend bins
    """
    print("\nAnalyzing performance by growth rate + trend bins...")
    
    # Get all unique combined bins and sort them
    unique_bins = forward_gains_df['combined_bin'].unique()
    
    # Sort bins by growth rate and then by trend
    def sort_key(bin_name):
        parts = bin_name.split(' ')
        if len(parts) >= 3:  # e.g., "10% to 15% UP"
            growth_part = ' '.join(parts[:-1])  # "10% to 15%"
            trend = parts[-1]  # "UP"
            
            # Extract start value for sorting
            if growth_part.startswith('<'):
                start_val = -999
            elif growth_part.startswith('>'):
                start_val = 999
            else:
                try:
                    start_val = int(growth_part.split('%')[0])
                except:
                    start_val = 0
            
            # Sort by growth rate first, then UP before DOWN
            trend_order = {'UP': 0, 'FLAT': 1, 'DOWN': 2}
            return (start_val, trend_order.get(trend, 3))
        return (0, 3)
    
    sorted_bins = sorted(unique_bins, key=sort_key)
    
    trend_analysis = []
    
    for bin_name in sorted_bins:
        bin_data = forward_gains_df[forward_gains_df['combined_bin'] == bin_name]
        
        if len(bin_data) >= 5:  # Lower minimum for trend analysis
            max_gains = bin_data['forward_max_gain']
            final_returns = bin_data['forward_final_return']
            
            # Parse bin components
            parts = bin_name.split(' ')
            if len(parts) >= 3:
                growth_bin = ' '.join(parts[:-1])
                trend = parts[-1]
            else:
                growth_bin = bin_name
                trend = 'UNKNOWN'
            
            trend_analysis.append({
                'combined_bin': bin_name,
                'growth_rate_bin': growth_bin,
                'trend_direction': trend,
                'sample_count': len(bin_data),
                'avg_max_gain': max_gains.mean(),
                'median_max_gain': max_gains.median(),
                'std_max_gain': max_gains.std(),
                'avg_final_return': final_returns.mean(),
                'median_final_return': final_returns.median(),
                'success_rate_positive': (final_returns > 0).mean() * 100,
                'success_rate_25pct': (max_gains > 25).mean() * 100,
                'success_rate_50pct': (max_gains > 50).mean() * 100,
                'avg_days_to_max': bin_data['days_to_max_gain'].mean(),
                'avg_growth_rate': bin_data['annual_growth_rate'].mean(),
                'avg_previous_growth': bin_data['previous_growth_5d'].mean(),
                'trend_strength': bin_data['annual_growth_rate'].mean() - bin_data['previous_growth_5d'].mean()
            })
    
    trend_analysis_df = pd.DataFrame(trend_analysis)
    print(f"✅ Analyzed {len(trend_analysis_df)} growth rate + trend bins with sufficient data")
    
    return trend_analysis_df

def create_comprehensive_visualizations(forward_gains_df, bin_analysis_df, trend_analysis_df, output_dir):
    """
    Create comprehensive visualizations of growth rate analysis
    """
    print("\nCreating comprehensive visualizations...")
    
    # Figure 1: Growth Rate Bins Analysis
    fig, ((ax1, ax2), (ax3, ax4)) = plt.subplots(2, 2, figsize=(24, 16))
    
    if len(bin_analysis_df) > 0:
        # Average max gains by growth rate bin
        n_bins = len(bin_analysis_df)
        colors = plt.cm.RdYlGn(np.linspace(0.2, 0.8, n_bins))
        
        bars1 = ax1.bar(range(n_bins), bin_analysis_df['avg_max_gain'], 
                       color=colors, alpha=0.8)
        ax1.set_xticks(range(n_bins))
        ax1.set_xticklabels(bin_analysis_df['growth_rate_bin'], rotation=45, ha='right')
        ax1.set_ylabel('Average Max Gain (%)')
        ax1.set_title('Average 1-Year Max Gains by Growth Rate Bin', fontweight='bold', fontsize=14)
        ax1.grid(True, alpha=0.3)
        
        # Add sample count labels
        for i, (bar, count) in enumerate(zip(bars1, bin_analysis_df['sample_count'])):
            ax1.text(bar.get_x() + bar.get_width()/2., 
                    bar.get_height() + max(bin_analysis_df['avg_max_gain'])*0.01,
                    f'n={count}', ha='center', va='bottom', fontsize=8, fontweight='bold')
        
        # Success rates
        width = 0.25
        x = np.arange(n_bins)
        ax2.bar(x - width, bin_analysis_df['success_rate_25pct'], width, 
               label='25%+ Gains', alpha=0.8, color='lightgreen')
        ax2.bar(x, bin_analysis_df['success_rate_50pct'], width, 
               label='50%+ Gains', alpha=0.8, color='green')
        ax2.bar(x + width, bin_analysis_df['success_rate_100pct'], width, 
               label='100%+ Gains', alpha=0.8, color='darkgreen')
        
        ax2.set_xticks(x)
        ax2.set_xticklabels(bin_analysis_df['growth_rate_bin'], rotation=45, ha='right')
        ax2.set_ylabel('Success Rate (%)')
        ax2.set_title('Success Rates by Growth Rate Bin', fontweight='bold', fontsize=14)
        ax2.legend()
        ax2.grid(True, alpha=0.3)
        
        # Sample count distribution
        bars3 = ax3.bar(range(n_bins), bin_analysis_df['sample_count'], 
                       color='steelblue', alpha=0.8)
        ax3.set_xticks(range(n_bins))
        ax3.set_xticklabels(bin_analysis_df['growth_rate_bin'], rotation=45, ha='right')
        ax3.set_ylabel('Sample Count')
        ax3.set_title('Sample Distribution by Growth Rate Bin', fontweight='bold', fontsize=14)
        ax3.grid(True, alpha=0.3)
        
        # Volatility vs Return
        ax4.scatter(bin_analysis_df['volatility'], bin_analysis_df['avg_max_gain'], 
                   c=bin_analysis_df['sample_count'], cmap='viridis', s=100, alpha=0.7)
        ax4.set_xlabel('Volatility (Std Dev of Max Gains)')
        ax4.set_ylabel('Average Max Gain (%)')
        ax4.set_title('Risk vs Return by Growth Rate Bin', fontweight='bold', fontsize=14)
        ax4.grid(True, alpha=0.3)
        
        # Add labels for outliers
        for i, row in bin_analysis_df.iterrows():
            if row['avg_max_gain'] > bin_analysis_df['avg_max_gain'].quantile(0.8) or \
               row['volatility'] > bin_analysis_df['volatility'].quantile(0.8):
                ax4.annotate(row['growth_rate_bin'], 
                           (row['volatility'], row['avg_max_gain']),
                           xytext=(5, 5), textcoords='offset points', 
                           fontsize=8, alpha=0.8)
    
    plt.tight_layout()
    plt.savefig(f"{output_dir}/growth_rate_bins_analysis.png", dpi=300, bbox_inches='tight')
    plt.close()
    
    # Figure 2: Trend Analysis (UP vs DOWN)
    if len(trend_analysis_df) > 0:
        fig, ((ax1, ax2), (ax3, ax4)) = plt.subplots(2, 2, figsize=(24, 16))
        
        # Separate UP and DOWN trends for comparison
        up_trends = trend_analysis_df[trend_analysis_df['trend_direction'] == 'UP']
        down_trends = trend_analysis_df[trend_analysis_df['trend_direction'] == 'DOWN']
        
        # Average max gains: UP vs DOWN
        if len(up_trends) > 0 and len(down_trends) > 0:
            # Match bins between UP and DOWN
            common_growth_bins = set(up_trends['growth_rate_bin']) & set(down_trends['growth_rate_bin'])
            
            up_gains = []
            down_gains = []
            bin_labels = []
            
            for growth_bin in sorted(common_growth_bins):
                up_data = up_trends[up_trends['growth_rate_bin'] == growth_bin]
                down_data = down_trends[down_trends['growth_rate_bin'] == growth_bin]
                
                if len(up_data) > 0 and len(down_data) > 0:
                    up_gains.append(up_data['avg_max_gain'].iloc[0])
                    down_gains.append(down_data['avg_max_gain'].iloc[0])
                    bin_labels.append(growth_bin)
            
            if bin_labels:
                x = np.arange(len(bin_labels))
                width = 0.35
                
                bars1 = ax1.bar(x - width/2, up_gains, width, label='Growth Rate UP', 
                               color='green', alpha=0.8)
                bars2 = ax1.bar(x + width/2, down_gains, width, label='Growth Rate DOWN', 
                               color='red', alpha=0.8)
                
                ax1.set_xticks(x)
                ax1.set_xticklabels(bin_labels, rotation=45, ha='right')
                ax1.set_ylabel('Average Max Gain (%)')
                ax1.set_title('Growth Rate UP vs DOWN: Average Max Gains', fontweight='bold', fontsize=14)
                ax1.legend()
                ax1.grid(True, alpha=0.3)
        
        # Trend strength vs performance
        if len(trend_analysis_df) > 0:
            colors = ['green' if t == 'UP' else 'red' if t == 'DOWN' else 'gray' 
                     for t in trend_analysis_df['trend_direction']]
            
            ax2.scatter(trend_analysis_df['trend_strength'], trend_analysis_df['avg_max_gain'], 
                       c=colors, s=trend_analysis_df['sample_count']*2, alpha=0.7)
            ax2.set_xlabel('Trend Strength (Current - Previous Growth Rate)')
            ax2.set_ylabel('Average Max Gain (%)')
            ax2.set_title('Trend Strength vs Performance (Green=UP, Red=DOWN)', fontweight='bold', fontsize=14)
            ax2.grid(True, alpha=0.3)
            ax2.axvline(x=0, color='black', linestyle='--', alpha=0.5)
        
        # Sample count by trend direction
        trend_counts = trend_analysis_df.groupby('trend_direction').agg({
            'sample_count': 'sum',
            'avg_max_gain': 'mean'
        }).reset_index()
        
        if len(trend_counts) > 0:
            bars3 = ax3.bar(trend_counts['trend_direction'], trend_counts['sample_count'], 
                           color=['green', 'red', 'gray'][:len(trend_counts)], alpha=0.8)
            ax3.set_ylabel('Total Sample Count')
            ax3.set_title('Sample Distribution by Trend Direction', fontweight='bold', fontsize=14)
            ax3.grid(True, alpha=0.3)
            
            # Add value labels
            for bar, value in zip(bars3, trend_counts['sample_count']):
                ax3.text(bar.get_x() + bar.get_width()/2., bar.get_height() + max(trend_counts['sample_count'])*0.01,
                        f'{value:,}', ha='center', va='bottom', fontweight='bold')
        
        # Performance by trend direction
        if len(trend_counts) > 0:
            bars4 = ax4.bar(trend_counts['trend_direction'], trend_counts['avg_max_gain'], 
                           color=['green', 'red', 'gray'][:len(trend_counts)], alpha=0.8)
            ax4.set_ylabel('Average Max Gain (%)')
            ax4.set_title('Average Performance by Trend Direction', fontweight='bold', fontsize=14)
            ax4.grid(True, alpha=0.3)
            
            # Add value labels
            for bar, value in zip(bars4, trend_counts['avg_max_gain']):
                ax4.text(bar.get_x() + bar.get_width()/2., bar.get_height() + max(trend_counts['avg_max_gain'])*0.01,
                        f'{value:.1f}%', ha='center', va='bottom', fontweight='bold')
    
    plt.tight_layout()
    plt.savefig(f"{output_dir}/growth_rate_trend_analysis.png", dpi=300, bbox_inches='tight')
    plt.close()
    
    # Figure 3: Heat map of performance by growth rate and trend
    if len(trend_analysis_df) > 0:
        fig, (ax1, ax2) = plt.subplots(1, 2, figsize=(20, 8))
        
        # Create pivot table for heatmap
        pivot_data = trend_analysis_df.pivot_table(
            index='growth_rate_bin', 
            columns='trend_direction', 
            values='avg_max_gain', 
            fill_value=np.nan
        )
        
        # Heatmap of average max gains
        if not pivot_data.empty:
            im1 = ax1.imshow(pivot_data.values, cmap='RdYlGn', aspect='auto', vmin=0, vmax=100)
            ax1.set_xticks(range(len(pivot_data.columns)))
            ax1.set_xticklabels(pivot_data.columns)
            ax1.set_yticks(range(len(pivot_data.index)))
            ax1.set_yticklabels(pivot_data.index, fontsize=8)
            ax1.set_title('Average Max Gain (%) by Growth Rate & Trend', fontweight='bold', fontsize=14)
            
            # Add text annotations
            for i in range(len(pivot_data.index)):
                for j in range(len(pivot_data.columns)):
                    value = pivot_data.iloc[i, j]
                    if not np.isnan(value):
                        ax1.text(j, i, f'{value:.1f}%', ha='center', va='center', 
                               fontweight='bold', fontsize=8)
            
            plt.colorbar(im1, ax=ax1, label='Average Max Gain (%)')
        
        # Heatmap of sample counts
        pivot_counts = trend_analysis_df.pivot_table(
            index='growth_rate_bin', 
            columns='trend_direction', 
            values='sample_count', 
            fill_value=0
        )
        
        if not pivot_counts.empty:
            im2 = ax2.imshow(pivot_counts.values, cmap='Blues', aspect='auto')
            ax2.set_xticks(range(len(pivot_counts.columns)))
            ax2.set_xticklabels(pivot_counts.columns)
            ax2.set_yticks(range(len(pivot_counts.index)))
            ax2.set_yticklabels(pivot_counts.index, fontsize=8)
            ax2.set_title('Sample Count by Growth Rate & Trend', fontweight='bold', fontsize=14)
            
            # Add text annotations
            for i in range(len(pivot_counts.index)):
                for j in range(len(pivot_counts.columns)):
                    value = pivot_counts.iloc[i, j]
                    ax2.text(j, i, f'{int(value)}', ha='center', va='center', 
                           fontweight='bold', fontsize=8)
            
            plt.colorbar(im2, ax=ax2, label='Sample Count')
        
        plt.tight_layout()
        plt.savefig(f"{output_dir}/growth_rate_trend_heatmap.png", dpi=300, bbox_inches='tight')
        plt.close()
    
    print("  ✅ All visualizations created")

def save_comprehensive_results(forward_gains_df, bin_analysis_df, trend_analysis_df, output_dir):
    """
    Save all comprehensive results
    """
    print("\nSaving comprehensive results...")
    
    # Save raw forward gains data
    forward_gains_df.to_csv(f"{output_dir}/forward_gains_by_growth_rate.csv", index=False)
    print(f"  ✅ Forward gains data: {len(forward_gains_df):,} samples")
    
    # Save growth rate bin analysis
    bin_analysis_df.to_csv(f"{output_dir}/growth_rate_bins_analysis.csv", index=False)
    print(f"  ✅ Growth rate bins analysis: {len(bin_analysis_df)} bins")
    
    # Save trend analysis
    trend_analysis_df.to_csv(f"{output_dir}/growth_rate_trend_analysis.csv", index=False)
    print(f"  ✅ Growth rate trend analysis: {len(trend_analysis_df)} trend bins")
    
    # Create summary statistics
    summary_stats = {
        'total_samples': len(forward_gains_df),
        'unique_assets': forward_gains_df['asset'].nunique(),
        'date_range_start': forward_gains_df['entry_date'].min(),
        'date_range_end': forward_gains_df['entry_date'].max(),
        'avg_forward_max_gain': forward_gains_df['forward_max_gain'].mean(),
        'median_forward_max_gain': forward_gains_df['forward_max_gain'].median(),
        'avg_growth_rate': forward_gains_df['annual_growth_rate'].mean(),
        'growth_rate_range_min': forward_gains_df['annual_growth_rate'].min(),
        'growth_rate_range_max': forward_gains_df['annual_growth_rate'].max(),
        'bins_analyzed': len(bin_analysis_df),
        'trend_bins_analyzed': len(trend_analysis_df)
    }
    
    summary_df = pd.DataFrame([summary_stats])
    summary_df.to_csv(f"{output_dir}/analysis_summary.csv", index=False)
    print(f"  ✅ Analysis summary saved")
    
    # Asset-level summary
    asset_summary = forward_gains_df.groupby('asset').agg({
        'forward_max_gain': ['count', 'mean', 'median', 'std'],
        'annual_growth_rate': ['mean', 'std'],
        'days_to_max_gain': 'mean'
    }).round(3)
    
    asset_summary.columns = ['_'.join(col).strip() for col in asset_summary.columns]
    asset_summary = asset_summary.reset_index()
    asset_summary.to_csv(f"{output_dir}/asset_level_summary.csv", index=False)
    print(f"  ✅ Asset-level summary: {len(asset_summary)} assets")

def print_comprehensive_insights(forward_gains_df, bin_analysis_df, trend_analysis_df):
    """
    Print comprehensive insights from the analysis
    """
    print("\n" + "="*80)
    print("GROWTH RATE vs FORWARD GAINS ANALYSIS - KEY INSIGHTS")
    print("="*80)
    
    # Overall statistics
    print(f"\n📊 DATASET OVERVIEW:")
    print(f"  Total samples: {len(forward_gains_df):,}")
    print(f"  Unique assets: {forward_gains_df['asset'].nunique()}")
    print(f"  Date range: {forward_gains_df['entry_date'].min().strftime('%Y-%m-%d')} to {forward_gains_df['entry_date'].max().strftime('%Y-%m-%d')}")
    print(f"  Average forward max gain: {forward_gains_df['forward_max_gain'].mean():.2f}%")
    print(f"  Average growth rate: {forward_gains_df['annual_growth_rate'].mean():.2f}%")
    
    # Best performing growth rate bins
    if len(bin_analysis_df) > 0:
        print(f"\n🏆 TOP GROWTH RATE BINS BY PERFORMANCE:")
        top_bins = bin_analysis_df.nlargest(5, 'avg_max_gain')
        
        for i, (_, row) in enumerate(top_bins.iterrows()):
            print(f"  #{i+1}: {row['growth_rate_bin']}")
            print(f"       Average Max Gain: {row['avg_max_gain']:.2f}%")
            print(f"       Sample Count: {row['sample_count']:,}")
            print(f"       Success Rate (25%+): {row['success_rate_25pct']:.1f}%")
    
    # Trend direction analysis
    if len(trend_analysis_df) > 0:
        print(f"\n📈 TREND DIRECTION ANALYSIS:")
        
        # Overall trend performance
        trend_summary = trend_analysis_df.groupby('trend_direction').agg({
            'avg_max_gain': 'mean',
            'sample_count': 'sum',
            'success_rate_25pct': 'mean'
        }).round(2)
        
        for trend, data in trend_summary.iterrows():
            print(f"  {trend} trends:")
            print(f"    Average Max Gain: {data['avg_max_gain']:.2f}%")
            print(f"    Total Samples: {data['sample_count']:,}")
            print(f"    Avg Success Rate (25%+): {data['success_rate_25pct']:.1f}%")
        
        # Best trend combinations
        print(f"\n🚀 TOP TREND COMBINATIONS:")
        top_trends = trend_analysis_df.nlargest(5, 'avg_max_gain')
        
        for i, (_, row) in enumerate(top_trends.iterrows()):
            print(f"  #{i+1}: {row['combined_bin']}")
            print(f"       Average Max Gain: {row['avg_max_gain']:.2f}%")
            print(f"       Sample Count: {row['sample_count']}")
            print(f"       Trend Strength: {row['trend_strength']:.2f}%")
    
    # Growth rate effectiveness analysis
    if len(bin_analysis_df) > 0:
        print(f"\n🎯 GROWTH RATE EFFECTIVENESS:")
        
        # Find optimal growth rate ranges
        high_performers = bin_analysis_df[bin_analysis_df['avg_max_gain'] > bin_analysis_df['avg_max_gain'].quantile(0.7)]
        
        if len(high_performers) > 0:
            print(f"  High-performing growth rate ranges:")
            for _, row in high_performers.iterrows():
                print(f"    {row['growth_rate_bin']}: {row['avg_max_gain']:.2f}% avg gain ({row['sample_count']:,} samples)")
        
        # Risk-adjusted analysis
        bin_analysis_df['risk_adjusted_return'] = bin_analysis_df['avg_max_gain'] / bin_analysis_df['volatility']
        best_risk_adjusted = bin_analysis_df.nlargest(3, 'risk_adjusted_return')
        
        print(f"\n  Best risk-adjusted growth rate bins:")
        for _, row in best_risk_adjusted.iterrows():
            print(f"    {row['growth_rate_bin']}: {row['risk_adjusted_return']:.3f} return/risk ratio")
    
    # UP vs DOWN trend comparison
    if len(trend_analysis_df) > 0:
        up_trends = trend_analysis_df[trend_analysis_df['trend_direction'] == 'UP']
        down_trends = trend_analysis_df[trend_analysis_df['trend_direction'] == 'DOWN']
        
        if len(up_trends) > 0 and len(down_trends) > 0:
            print(f"\n⬆️⬇️ UP vs DOWN TREND COMPARISON:")
            
            up_avg = up_trends['avg_max_gain'].mean()
            down_avg = down_trends['avg_max_gain'].mean()
            
            print(f"  UP trends average gain: {up_avg:.2f}%")
            print(f"  DOWN trends average gain: {down_avg:.2f}%")
            print(f"  Advantage: {up_avg - down_avg:+.2f}% {'for UP trends' if up_avg > down_avg else 'for DOWN trends'}")
            
            # Find best combinations within each trend
            if len(up_trends) > 0:
                best_up = up_trends.loc[up_trends['avg_max_gain'].idxmax()]
                print(f"\n  Best UP trend: {best_up['combined_bin']} ({best_up['avg_max_gain']:.2f}% gain)")
            
            if len(down_trends) > 0:
                best_down = down_trends.loc[down_trends['avg_max_gain'].idxmax()]
                print(f"  Best DOWN trend: {best_down['combined_bin']} ({best_down['avg_max_gain']:.2f}% gain)")
    
    # Sample distribution insights
    print(f"\n📈 SAMPLE DISTRIBUTION INSIGHTS:")
    
    # Growth rate distribution
    growth_stats = forward_gains_df['annual_growth_rate'].describe()
    print(f"  Growth rate distribution:")
    print(f"    Mean: {growth_stats['mean']:.2f}%")
    print(f"    Median: {growth_stats['50%']:.2f}%")
    print(f"    Range: {growth_stats['min']:.2f}% to {growth_stats['max']:.2f}%")
    
    # Trend distribution
    trend_dist = forward_gains_df['growth_trend'].value_counts()
    print(f"\n  Trend direction distribution:")
    for trend, count in trend_dist.items():
        pct = count / len(forward_gains_df) * 100
        print(f"    {trend}: {count:,} samples ({pct:.1f}%)")
    
    # Time-based patterns
    forward_gains_df['entry_year'] = pd.to_datetime(forward_gains_df['entry_date']).dt.year
    yearly_performance = forward_gains_df.groupby('entry_year')['forward_max_gain'].mean()
    
    if len(yearly_performance) > 1:
        print(f"\n📅 YEARLY PERFORMANCE PATTERNS:")
        for year, avg_gain in yearly_performance.items():
            print(f"    {year}: {avg_gain:.2f}% average max gain")
    
    # Strategic recommendations
    print(f"\n" + "="*80)
    print("🎯 STRATEGIC RECOMMENDATIONS")
    print("="*80)
    
    if len(bin_analysis_df) > 0:
        # Find sweet spot growth rates
        sweet_spots = bin_analysis_df[
            (bin_analysis_df['avg_max_gain'] > bin_analysis_df['avg_max_gain'].quantile(0.6)) &
            (bin_analysis_df['sample_count'] >= 100)  # Sufficient sample size
        ].sort_values('avg_max_gain', ascending=False)
        
        if len(sweet_spots) > 0:
            print(f"\n🏆 OPTIMAL GROWTH RATE RANGES (high performance + sufficient samples):")
            for i, (_, row) in enumerate(sweet_spots.head(5).iterrows()):
                print(f"  #{i+1}: {row['growth_rate_bin']}")
                print(f"       • Average Max Gain: {row['avg_max_gain']:.2f}%")
                print(f"       • Success Rate (25%+): {row['success_rate_25pct']:.1f}%")
                print(f"       • Sample Count: {row['sample_count']:,}")
                print(f"       • Risk-Adjusted: {row['risk_adjusted_return']:.3f}")
    
    if len(trend_analysis_df) > 0:
        print(f"\n📊 TREND-BASED RECOMMENDATIONS:")
        
        # Best trend strategies
        reliable_trends = trend_analysis_df[trend_analysis_df['sample_count'] >= 20]
        if len(reliable_trends) > 0:
            best_trends = reliable_trends.nlargest(3, 'avg_max_gain')
            
            for i, (_, row) in enumerate(best_trends.iterrows()):
                print(f"  Strategy #{i+1}: Target {row['combined_bin']}")
                print(f"    • Expected Max Gain: {row['avg_max_gain']:.2f}%")
                print(f"    • Success Rate (25%+): {row['success_rate_25pct']:.1f}%")
                print(f"    • Sample Count: {row['sample_count']}")
        
        # Trend momentum insights
        strong_up_trends = trend_analysis_df[
            (trend_analysis_df['trend_direction'] == 'UP') & 
            (trend_analysis_df['trend_strength'] > 5)  # Strong positive momentum
        ]
        
        strong_down_trends = trend_analysis_df[
            (trend_analysis_df['trend_direction'] == 'DOWN') & 
            (trend_analysis_df['trend_strength'] < -5)  # Strong negative momentum
        ]
        
        if len(strong_up_trends) > 0:
            avg_strong_up = strong_up_trends['avg_max_gain'].mean()
            print(f"\n  🚀 Strong UP momentum (>5% acceleration): {avg_strong_up:.2f}% avg max gain")
        
        if len(strong_down_trends) > 0:
            avg_strong_down = strong_down_trends['avg_max_gain'].mean()
            print(f"  📉 Strong DOWN momentum (<-5% deceleration): {avg_strong_down:.2f}% avg max gain")
    
    print(f"\n💡 KEY INSIGHTS FOR TRADING:")
    
    # Overall trend advantage
    if len(trend_analysis_df) > 0:
        up_avg = trend_analysis_df[trend_analysis_df['trend_direction'] == 'UP']['avg_max_gain'].mean()
        down_avg = trend_analysis_df[trend_analysis_df['trend_direction'] == 'DOWN']['avg_max_gain'].mean()
        
        if up_avg > down_avg:
            print(f"  1. ✅ Growth rate acceleration (UP trends) outperforms by {up_avg - down_avg:.2f}%")
            print(f"     → Focus on stocks with accelerating growth rates")
        else:
            print(f"  1. ⚠️  Growth rate deceleration (DOWN trends) outperforms by {down_avg - up_avg:.2f}%")
            print(f"     → Consider contrarian approach - slowing growth may signal value")
    
    # Growth rate insights
    if len(bin_analysis_df) > 0:
        high_growth = bin_analysis_df[bin_analysis_df['avg_growth_rate'] > 30]['avg_max_gain'].mean()
        low_growth = bin_analysis_df[bin_analysis_df['avg_growth_rate'] < 10]['avg_max_gain'].mean()
        
        if not pd.isna(high_growth) and not pd.isna(low_growth):
            if high_growth > low_growth:
                print(f"  2. 📈 High growth rates (>30%) outperform low growth (<10%) by {high_growth - low_growth:.2f}%")
                print(f"     → Target high-growth stocks for maximum gains")
            else:
                print(f"  2. 📊 Low growth rates (<10%) outperform high growth (>30%) by {low_growth - high_growth:.2f}%")
                print(f"     → Value in slower-growth, potentially undervalued stocks")
    
    # Sample reliability
    reliable_bins = bin_analysis_df[bin_analysis_df['sample_count'] >= 100] if len(bin_analysis_df) > 0 else pd.DataFrame()
    if len(reliable_bins) > 0:
        print(f"  3. 🎯 {len(reliable_bins)} growth rate bins have 100+ samples for reliable analysis")
        print(f"     → Focus on these ranges for statistically significant strategies")
    
    print(f"\n📋 IMPLEMENTATION CHECKLIST:")
    print(f"  □ Screen for stocks in optimal growth rate ranges")
    print(f"  □ Monitor growth rate trends (acceleration vs deceleration)")
    print(f"  □ Consider sample size reliability when making decisions")
    print(f"  □ Factor in risk-adjusted returns, not just average gains")
    print(f"  □ Review yearly patterns for market cycle considerations")

def main():
    print("GROWTH RATE vs FORWARD GAINS ANALYSIS")
    print("=" * 70)
    print("Analysis: Growth rate bins (5% increments) with UP/DOWN trend analysis")
    print("Objective: Find optimal growth rate ranges and trend patterns for forward gains")
    print("Output: Comprehensive binned analysis with trend direction impact")
    
    # Setup directories
    v2_dir = "/Users/tim/IWLS-OPTIONS/IWLS_ANALYSIS_V2"
    
    if not os.path.exists(v2_dir):
        print("❌ IWLS_ANALYSIS_V2 directory not found.")
        return
    
    # Create output directory
    output_dir = os.path.join(v2_dir, "GROWTH_RATE_BINS_TREND_ANALYSIS")
    os.makedirs(output_dir, exist_ok=True)
    
    print(f"\nOutput directory: {output_dir}")
    
    # Load all IWLS data
    all_data = load_all_iwls_data(v2_dir)
    
    if len(all_data) < 5:
        print(f"❌ Insufficient assets loaded ({len(all_data)}). Need at least 5 for analysis.")
        return
    
    print(f"\nAnalysis parameters:")
    print(f"  Assets loaded: {len(all_data)}")
    print(f"  Forward period: 365 days (1 year)")
    print(f"  Growth rate bins: 5% increments")
    print(f"  Trend detection: 5-day moving average comparison")
    print(f"  Trend categories: UP, DOWN, FLAT")
    
    # Calculate forward gains for all assets
    forward_gains_df = calculate_forward_max_gains(all_data, forward_days=365)
    
    if len(forward_gains_df) == 0:
        print("❌ No forward gains data generated.")
        return
    
    # Analyze by growth rate bins (without trend)
    bin_analysis_df = analyze_growth_rate_bins(forward_gains_df)
    
    # Analyze by growth rate + trend bins
    trend_analysis_df = analyze_growth_trend_bins(forward_gains_df)
    
    # Create comprehensive visualizations
    create_comprehensive_visualizations(forward_gains_df, bin_analysis_df, trend_analysis_df, output_dir)
    
    # Save all results
    save_comprehensive_results(forward_gains_df, bin_analysis_df, trend_analysis_df, output_dir)
    
    # Print comprehensive insights
    print_comprehensive_insights(forward_gains_df, bin_analysis_df, trend_analysis_df)
    
    print(f"\n" + "="*70)
    print("GROWTH RATE BINS & TREND ANALYSIS COMPLETE")
    print("="*70)
    print(f"Results saved to: {output_dir}")
    print("\nFiles created:")
    print("  📄 forward_gains_by_growth_rate.csv (raw data with all samples)")
    print("  📄 growth_rate_bins_analysis.csv (5% bin performance analysis)")
    print("  📄 growth_rate_trend_analysis.csv (growth rate + trend combinations)")
    print("  📄 asset_level_summary.csv (per-asset statistics)")
    print("  📄 analysis_summary.csv (overall summary metrics)")
    print("  📊 growth_rate_bins_analysis.png (main bin performance charts)")
    print("  📊 growth_rate_trend_analysis.png (trend UP vs DOWN comparisons)")
    print("  📊 growth_rate_trend_heatmap.png (performance matrix)")
    
    print(f"\n🎯 Key Analysis Features:")
    print(f"   • Growth rates binned in 5% increments (0-5%, 5-10%, etc.)")
    print(f"   • Each bin split by trend direction (UP/DOWN/FLAT)")
    print(f"   • 1-year forward maximum gains calculated for each entry")
    print(f"   • Trend strength measured vs 5-day moving average")
    print(f"   • Statistical significance via sample counts")
    print(f"   • Risk-adjusted performance metrics")
    
    if len(forward_gains_df) > 0:
        print(f"\n📊 DATASET SUMMARY:")
        print(f"   Total samples: {len(forward_gains_df):,}")
        print(f"   Assets analyzed: {forward_gains_df['asset'].nunique()}")
        print(f"   Growth rate bins: {len(bin_analysis_df)}")
        print(f"   Trend combinations: {len(trend_analysis_df)}")
        print(f"   Average forward gain: {forward_gains_df['forward_max_gain'].mean():.2f}%")
        
        # Quick insight
        up_samples = len(forward_gains_df[forward_gains_df['growth_trend'] == 'UP'])
        down_samples = len(forward_gains_df[forward_gains_df['growth_trend'] == 'DOWN'])
        
        if up_samples > 0 and down_samples > 0:
            up_avg = forward_gains_df[forward_gains_df['growth_trend'] == 'UP']['forward_max_gain'].mean()
            down_avg = forward_gains_df[forward_gains_df['growth_trend'] == 'DOWN']['forward_max_gain'].mean()
            
            print(f"\n🚀 QUICK INSIGHT:")
            if up_avg > down_avg:
                print(f"   Growth rate acceleration (UP) outperforms by {up_avg - down_avg:.2f}%!")
                print(f"   → Momentum strategy may be optimal")
            else:
                print(f"   Growth rate deceleration (DOWN) outperforms by {down_avg - up_avg:.2f}%!")
                print(f"   → Contrarian strategy may be optimal")

if __name__ == "__main__":
    main()