import pandas as pd
import numpy as np
import matplotlib.pyplot as plt
import seaborn as sns
from datetime import datetime, timedelta
import warnings
import os
from collections import defaultdict
import itertools
warnings.filterwarnings('ignore')

def load_all_iwls_data(v2_dir):
    """
    Load IWLS data for all assets with growth rates and price deviation
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
                # Need both annual_growth and price_deviation columns
                required_cols = ['annual_growth', 'price', 'price_deviation']
                if all(col in df.columns for col in required_cols):
                    df = df.dropna(subset=required_cols).sort_values('date').reset_index(drop=True)
                    
                    if len(df) > 500:  # Need sufficient data
                        all_data[asset_name] = df
                        print(f"  * {asset_name}: {len(df)} data points")
                else:
                    print(f"  - Missing required columns in {asset_name}")
                    
            except Exception as e:
                print(f"  ! Error loading {asset_name}: {e}")
    
    print(f"* Successfully loaded {len(all_data)} assets")
    return all_data

def categorize_growth_rate_quintiles(growth_rate, all_growth_rates):
    """
    Categorize growth rate into quintiles (5 equal-sized bins)
    """
    if pd.isna(growth_rate):
        return "Unknown"
    
    # Calculate quintiles from all growth rates
    quintiles = np.percentile(all_growth_rates.dropna(), [20, 40, 60, 80])
    
    if growth_rate <= quintiles[0]:
        return "Q1_Bottom_20pct"
    elif growth_rate <= quintiles[1]:
        return "Q2_20_40pct"
    elif growth_rate <= quintiles[2]:
        return "Q3_40_60pct"
    elif growth_rate <= quintiles[3]:
        return "Q4_60_80pct"
    else:
        return "Q5_Top_20pct"

def categorize_price_deviation_quintiles(price_deviation, all_deviations):
    """
    Categorize price deviation into quintiles (5 equal-sized bins)
    """
    if pd.isna(price_deviation):
        return "Unknown"
    
    # Calculate quintiles from all price deviations
    quintiles = np.percentile(all_deviations.dropna(), [20, 40, 60, 80])
    
    if price_deviation <= quintiles[0]:
        return "D1_Bottom_20pct"
    elif price_deviation <= quintiles[1]:
        return "D2_20_40pct"
    elif price_deviation <= quintiles[2]:
        return "D3_40_60pct"
    elif price_deviation <= quintiles[3]:
        return "D4_60_80pct"
    else:
        return "D5_Top_20pct"

def determine_growth_trend(current_growth, previous_growth):
    """
    Determine if growth rate is trending UP or DOWN
    """
    if pd.isna(current_growth) or pd.isna(previous_growth):
        return "UNKNOWN"
    
    # Use a threshold to determine trend direction
    threshold = 0.5  # 0.5% threshold for trend determination
    
    if current_growth > previous_growth + threshold:
        return "UP"
    elif current_growth < previous_growth - threshold:
        return "DOWN"
    else:
        return "UP" if current_growth >= previous_growth else "DOWN"  # No FLAT category

def calculate_forward_max_gains_multi_attribute(all_data, forward_days=365):
    """
    Calculate forward maximum gains with multiple attribute categorization
    """
    print(f"\nCalculating forward {forward_days}-day maximum gains with multi-attribute analysis...")
    
    all_forward_gains = []
    
    # First pass: collect all values to calculate quintiles
    print("  Collecting all values for quintile calculation...")
    all_growth_rates = []
    all_price_deviations = []
    
    for asset_name, df in all_data.items():
        all_growth_rates.extend(df['annual_growth'].dropna().tolist())
        all_price_deviations.extend(df['price_deviation'].dropna().tolist())
    
    all_growth_rates_series = pd.Series(all_growth_rates)
    all_price_deviations_series = pd.Series(all_price_deviations)
    
    print(f"  Total growth rate samples: {len(all_growth_rates_series):,}")
    print(f"  Growth rate range: {all_growth_rates_series.min():.2f}% to {all_growth_rates_series.max():.2f}%")
    print(f"  Total price deviation samples: {len(all_price_deviations_series):,}")
    print(f"  Price deviation range: {all_price_deviations_series.min():.2f}% to {all_price_deviations_series.max():.2f}%")
    
    for asset_name, df in all_data.items():
        print(f"  Processing {asset_name}...")
        
        # Calculate growth rate trend (comparing to previous 5-day average)
        df['previous_growth_5d'] = df['annual_growth'].rolling(window=5, min_periods=1).mean().shift(1)
        
        for i in range(len(df) - forward_days):
            current_row = df.iloc[i]
            entry_date = current_row['date']
            entry_price = current_row['price']
            current_growth = current_row['annual_growth']
            current_deviation = current_row['price_deviation']
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
                
                # Categorize all attributes
                growth_quintile = categorize_growth_rate_quintiles(current_growth, all_growth_rates_series)
                deviation_quintile = categorize_price_deviation_quintiles(current_deviation, all_price_deviations_series)
                growth_trend = determine_growth_trend(current_growth, previous_growth)
                
                # Create combined bin from all three attributes
                if growth_trend not in ['UNKNOWN'] and growth_quintile != "Unknown" and deviation_quintile != "Unknown":
                    combined_bin = f"{growth_quintile}_{deviation_quintile}_{growth_trend}"
                else:
                    combined_bin = None  # Skip incomplete data
                
                # Find time to maximum gain
                max_idx = future_data['price'].idxmax()
                max_gain_date = future_data.loc[max_idx, 'date']
                days_to_max = (max_gain_date - entry_date).days
                
                # Only include valid combinations
                if combined_bin is not None:
                    all_forward_gains.append({
                        'asset': asset_name,
                        'entry_date': entry_date,
                        'entry_price': entry_price,
                        'annual_growth_rate': current_growth,
                        'price_deviation': current_deviation,
                        'previous_growth_5d': previous_growth,
                        'growth_quintile': growth_quintile,
                        'deviation_quintile': deviation_quintile,
                        'growth_trend': growth_trend,
                        'combined_bin': combined_bin,
                        'forward_max_gain': max_gain,
                        'forward_final_return': final_return,
                        'days_to_max_gain': days_to_max,
                        'future_data_points': len(future_data)
                    })
        
        print(f"    Completed {asset_name}: {len(df) - forward_days} forward return samples")
    
    forward_gains_df = pd.DataFrame(all_forward_gains)
    print(f"\n* Generated {len(forward_gains_df):,} total forward gain samples across all assets")
    
    return forward_gains_df

def remove_outliers_from_bin(bin_data, outlier_count=3):
    """
    Remove the top 3 outlier values (highest max gains) from each bin to reduce skew
    Reduced from 5 to 3 since we'll have more bins with smaller sample sizes
    """
    if len(bin_data) <= outlier_count:
        return bin_data  # Don't remove if bin too small
    
    # Sort by forward_max_gain and remove top 3
    sorted_data = bin_data.sort_values('forward_max_gain', ascending=False)
    cleaned_data = sorted_data.iloc[outlier_count:]  # Remove top 3
    
    return cleaned_data

def analyze_multi_attribute_bins(forward_gains_df):
    """
    Analyze forward gains by multi-attribute bins (growth quintile + deviation quintile + trend)
    """
    print("\nAnalyzing performance by multi-attribute bins...")
    
    # Get all unique combined bins and sort them
    unique_bins = forward_gains_df['combined_bin'].unique()
    
    # Sort bins systematically: Growth Quintile -> Deviation Quintile -> Trend
    def sort_key(bin_name):
        parts = bin_name.split('_')
        
        # Extract growth quintile (Q1-Q5)
        growth_q = 0
        if 'Q1' in bin_name:
            growth_q = 1
        elif 'Q2' in bin_name:
            growth_q = 2
        elif 'Q3' in bin_name:
            growth_q = 3
        elif 'Q4' in bin_name:
            growth_q = 4
        elif 'Q5' in bin_name:
            growth_q = 5
        
        # Extract deviation quintile (D1-D5)
        dev_q = 0
        if 'D1' in bin_name:
            dev_q = 1
        elif 'D2' in bin_name:
            dev_q = 2
        elif 'D3' in bin_name:
            dev_q = 3
        elif 'D4' in bin_name:
            dev_q = 4
        elif 'D5' in bin_name:
            dev_q = 5
        
        # Extract trend
        trend_order = {'UP': 0, 'DOWN': 1}
        trend = 2
        if 'UP' in bin_name:
            trend = 0
        elif 'DOWN' in bin_name:
            trend = 1
        
        return (growth_q, dev_q, trend)
    
    sorted_bins = sorted(unique_bins, key=sort_key)
    
    multi_analysis = []
    
    for bin_name in sorted_bins:
        bin_data = forward_gains_df[forward_gains_df['combined_bin'] == bin_name]
        
        if len(bin_data) >= 8:  # Minimum for multi-attribute analysis (smaller threshold due to more bins)
            # Remove top 3 outliers (reduced from 5)
            cleaned_data = remove_outliers_from_bin(bin_data, outlier_count=3)
            outliers_removed = len(bin_data) - len(cleaned_data)
            
            max_gains = cleaned_data['forward_max_gain']
            final_returns = cleaned_data['forward_final_return']
            
            # Parse bin components
            parts = bin_name.split('_')
            growth_quintile = '_'.join(parts[:2])  # Q1_Bottom_20pct
            deviation_quintile = '_'.join(parts[2:4])  # D1_Bottom_20pct
            trend = parts[-1]  # UP or DOWN
            
            multi_analysis.append({
                'combined_bin': bin_name,
                'growth_quintile': growth_quintile,
                'deviation_quintile': deviation_quintile,
                'trend_direction': trend,
                'original_sample_count': len(bin_data),
                'cleaned_sample_count': len(cleaned_data),
                'outliers_removed': outliers_removed,
                'avg_max_gain': max_gains.mean(),
                'median_max_gain': max_gains.median(),
                'std_max_gain': max_gains.std(),
                'avg_final_return': final_returns.mean(),
                'median_final_return': final_returns.median(),
                'success_rate_positive': (final_returns > 0).mean() * 100,
                'success_rate_25pct': (max_gains > 25).mean() * 100,
                'success_rate_50pct': (max_gains > 50).mean() * 100,
                'success_rate_100pct': (max_gains > 100).mean() * 100,
                'avg_days_to_max': cleaned_data['days_to_max_gain'].mean(),
                'avg_growth_rate': cleaned_data['annual_growth_rate'].mean(),
                'avg_price_deviation': cleaned_data['price_deviation'].mean(),
                'avg_previous_growth': cleaned_data['previous_growth_5d'].mean(),
                'trend_strength': cleaned_data['annual_growth_rate'].mean() - cleaned_data['previous_growth_5d'].mean()
            })
    
    multi_analysis_df = pd.DataFrame(multi_analysis)
    print(f"* Analyzed {len(multi_analysis_df)} multi-attribute bins with sufficient data")
    
    if len(multi_analysis_df) > 0:
        total_outliers = multi_analysis_df['outliers_removed'].sum()
        print(f"  Removed {total_outliers:,} outlier samples across all bins")
        
        # Show distribution of bins
        print(f"\nBin distribution:")
        growth_dist = multi_analysis_df['growth_quintile'].value_counts()
        print(f"  Growth quintiles: {dict(growth_dist)}")
        
        dev_dist = multi_analysis_df['deviation_quintile'].value_counts()
        print(f"  Deviation quintiles: {dict(dev_dist)}")
        
        trend_dist = multi_analysis_df['trend_direction'].value_counts()
        print(f"  Trend directions: {dict(trend_dist)}")
    
    return multi_analysis_df

def analyze_individual_attributes(forward_gains_df):
    """
    Analyze performance by individual attributes for comparison
    """
    print("\nAnalyzing individual attribute performance...")
    
    individual_analysis = {}
    
    # Growth quintile analysis
    growth_analysis = []
    growth_quintiles = ['Q1_Bottom_20pct', 'Q2_20_40pct', 'Q3_40_60pct', 'Q4_60_80pct', 'Q5_Top_20pct']
    
    for quintile in growth_quintiles:
        data = forward_gains_df[forward_gains_df['growth_quintile'] == quintile]
        if len(data) >= 10:
            cleaned_data = remove_outliers_from_bin(data, outlier_count=5)
            max_gains = cleaned_data['forward_max_gain']
            
            growth_analysis.append({
                'quintile': quintile,
                'sample_count': len(cleaned_data),
                'avg_max_gain': max_gains.mean(),
                'median_max_gain': max_gains.median(),
                'success_rate_25pct': (max_gains > 25).mean() * 100,
                'avg_growth_rate': cleaned_data['annual_growth_rate'].mean()
            })
    
    individual_analysis['growth'] = pd.DataFrame(growth_analysis)
    
    # Deviation quintile analysis
    deviation_analysis = []
    deviation_quintiles = ['D1_Bottom_20pct', 'D2_20_40pct', 'D3_40_60pct', 'D4_60_80pct', 'D5_Top_20pct']
    
    for quintile in deviation_quintiles:
        data = forward_gains_df[forward_gains_df['deviation_quintile'] == quintile]
        if len(data) >= 10:
            cleaned_data = remove_outliers_from_bin(data, outlier_count=5)
            max_gains = cleaned_data['forward_max_gain']
            
            deviation_analysis.append({
                'quintile': quintile,
                'sample_count': len(cleaned_data),
                'avg_max_gain': max_gains.mean(),
                'median_max_gain': max_gains.median(),
                'success_rate_25pct': (max_gains > 25).mean() * 100,
                'avg_price_deviation': cleaned_data['price_deviation'].mean()
            })
    
    individual_analysis['deviation'] = pd.DataFrame(deviation_analysis)
    
    # Trend analysis
    trend_analysis = []
    for trend in ['UP', 'DOWN']:
        data = forward_gains_df[forward_gains_df['growth_trend'] == trend]
        if len(data) >= 10:
            cleaned_data = remove_outliers_from_bin(data, outlier_count=5)
            max_gains = cleaned_data['forward_max_gain']
            
            trend_analysis.append({
                'trend': trend,
                'sample_count': len(cleaned_data),
                'avg_max_gain': max_gains.mean(),
                'median_max_gain': max_gains.median(),
                'success_rate_25pct': (max_gains > 25).mean() * 100,
                'avg_trend_strength': cleaned_data['annual_growth_rate'].mean() - cleaned_data['previous_growth_5d'].mean()
            })
    
    individual_analysis['trend'] = pd.DataFrame(trend_analysis)
    
    print(f"* Individual attribute analysis complete")
    return individual_analysis

def create_multi_attribute_visualizations(forward_gains_df, multi_analysis_df, individual_analysis, output_dir):
    """
    Create comprehensive visualizations for multi-attribute analysis
    """
    print("\nCreating multi-attribute visualizations...")
    
    # Figure 1: Individual Attributes Performance
    fig, ((ax1, ax2), (ax3, ax4)) = plt.subplots(2, 2, figsize=(20, 16))
    
    # Growth quintiles performance
    if len(individual_analysis['growth']) > 0:
        growth_df = individual_analysis['growth']
        bars1 = ax1.bar(range(len(growth_df)), growth_df['avg_max_gain'], 
                       color='green', alpha=0.7)
        ax1.set_xticks(range(len(growth_df)))
        ax1.set_xticklabels(growth_df['quintile'], rotation=45, ha='right')
        ax1.set_ylabel('Average Max Gain (%)')
        ax1.set_title('Performance by Growth Rate Quintiles', fontweight='bold')
        ax1.grid(True, alpha=0.3)
        
        for i, (bar, count) in enumerate(zip(bars1, growth_df['sample_count'])):
            ax1.text(bar.get_x() + bar.get_width()/2., 
                    bar.get_height() + max(growth_df['avg_max_gain'])*0.01,
                    f'n={count}', ha='center', va='bottom', fontsize=8)
    
    # Deviation quintiles performance
    if len(individual_analysis['deviation']) > 0:
        dev_df = individual_analysis['deviation']
        bars2 = ax2.bar(range(len(dev_df)), dev_df['avg_max_gain'], 
                       color='blue', alpha=0.7)
        ax2.set_xticks(range(len(dev_df)))
        ax2.set_xticklabels(dev_df['quintile'], rotation=45, ha='right')
        ax2.set_ylabel('Average Max Gain (%)')
        ax2.set_title('Performance by Price Deviation Quintiles', fontweight='bold')
        ax2.grid(True, alpha=0.3)
        
        for i, (bar, count) in enumerate(zip(bars2, dev_df['sample_count'])):
            ax2.text(bar.get_x() + bar.get_width()/2., 
                    bar.get_height() + max(dev_df['avg_max_gain'])*0.01,
                    f'n={count}', ha='center', va='bottom', fontsize=8)
    
    # Trend performance
    if len(individual_analysis['trend']) > 0:
        trend_df = individual_analysis['trend']
        colors = ['green', 'red']
        bars3 = ax3.bar(range(len(trend_df)), trend_df['avg_max_gain'], 
                       color=colors[:len(trend_df)], alpha=0.7)
        ax3.set_xticks(range(len(trend_df)))
        ax3.set_xticklabels(trend_df['trend'])
        ax3.set_ylabel('Average Max Gain (%)')
        ax3.set_title('Performance by Growth Trend Direction', fontweight='bold')
        ax3.grid(True, alpha=0.3)
        
        for i, (bar, count) in enumerate(zip(bars3, trend_df['sample_count'])):
            ax3.text(bar.get_x() + bar.get_width()/2., 
                    bar.get_height() + max(trend_df['avg_max_gain'])*0.01,
                    f'n={count}', ha='center', va='bottom', fontsize=8)
    
    # Top multi-attribute combinations
    if len(multi_analysis_df) > 0:
        top_combos = multi_analysis_df.nlargest(10, 'avg_max_gain')
        bars4 = ax4.barh(range(len(top_combos)), top_combos['avg_max_gain'], 
                        color='purple', alpha=0.7)
        ax4.set_yticks(range(len(top_combos)))
        ax4.set_yticklabels([bin_name.replace('_', ' ') for bin_name in top_combos['combined_bin']], 
                           fontsize=8)
        ax4.set_xlabel('Average Max Gain (%)')
        ax4.set_title('Top 10 Multi-Attribute Combinations', fontweight='bold')
        ax4.grid(True, alpha=0.3)
        
        for i, (bar, count) in enumerate(zip(bars4, top_combos['cleaned_sample_count'])):
            ax4.text(bar.get_width() + max(top_combos['avg_max_gain'])*0.01, 
                    bar.get_y() + bar.get_height()/2.,
                    f'n={count}', ha='left', va='center', fontsize=7)
    
    plt.tight_layout()
    plt.savefig(f"{output_dir}/multi_attribute_performance.png", dpi=300, bbox_inches='tight')
    plt.close()
    
    # Figure 2: 3D Heatmap-style visualization
    if len(multi_analysis_df) > 0:
        fig, ((ax1, ax2), (ax3, ax4)) = plt.subplots(2, 2, figsize=(20, 16))
        
        # Create pivot tables for heatmaps
        # Growth vs Deviation (averaged across trends)
        growth_dev_pivot = multi_analysis_df.groupby(['growth_quintile', 'deviation_quintile'])['avg_max_gain'].mean().unstack(fill_value=np.nan)
        
        if not growth_dev_pivot.empty:
            im1 = ax1.imshow(growth_dev_pivot.values, cmap='RdYlGn', aspect='auto', vmin=0, vmax=100)
            ax1.set_xticks(range(len(growth_dev_pivot.columns)))
            ax1.set_xticklabels([col.replace('_', ' ') for col in growth_dev_pivot.columns], rotation=45, ha='right')
            ax1.set_yticks(range(len(growth_dev_pivot.index)))
            ax1.set_yticklabels([idx.replace('_', ' ') for idx in growth_dev_pivot.index])
            ax1.set_title('Growth Rate vs Price Deviation\n(Average Max Gain %)', fontweight='bold')
            
            # Add text annotations
            for i in range(len(growth_dev_pivot.index)):
                for j in range(len(growth_dev_pivot.columns)):
                    value = growth_dev_pivot.iloc[i, j]
                    if not np.isnan(value):
                        ax1.text(j, i, f'{value:.1f}', ha='center', va='center', 
                               fontweight='bold', fontsize=8)
            
            plt.colorbar(im1, ax=ax1, label='Average Max Gain (%)')
        
        # Growth vs Trend
        growth_trend_pivot = multi_analysis_df.groupby(['growth_quintile', 'trend_direction'])['avg_max_gain'].mean().unstack(fill_value=np.nan)
        
        if not growth_trend_pivot.empty:
            im2 = ax2.imshow(growth_trend_pivot.values, cmap='RdYlGn', aspect='auto', vmin=0, vmax=100)
            ax2.set_xticks(range(len(growth_trend_pivot.columns)))
            ax2.set_xticklabels(growth_trend_pivot.columns)
            ax2.set_yticks(range(len(growth_trend_pivot.index)))
            ax2.set_yticklabels([idx.replace('_', ' ') for idx in growth_trend_pivot.index])
            ax2.set_title('Growth Rate vs Trend Direction\n(Average Max Gain %)', fontweight='bold')
            
            for i in range(len(growth_trend_pivot.index)):
                for j in range(len(growth_trend_pivot.columns)):
                    value = growth_trend_pivot.iloc[i, j]
                    if not np.isnan(value):
                        ax2.text(j, i, f'{value:.1f}', ha='center', va='center', 
                               fontweight='bold', fontsize=8)
            
            plt.colorbar(im2, ax=ax2, label='Average Max Gain (%)')
        
        # Deviation vs Trend
        dev_trend_pivot = multi_analysis_df.groupby(['deviation_quintile', 'trend_direction'])['avg_max_gain'].mean().unstack(fill_value=np.nan)
        
        if not dev_trend_pivot.empty:
            im3 = ax3.imshow(dev_trend_pivot.values, cmap='RdYlGn', aspect='auto', vmin=0, vmax=100)
            ax3.set_xticks(range(len(dev_trend_pivot.columns)))
            ax3.set_xticklabels(dev_trend_pivot.columns)
            ax3.set_yticks(range(len(dev_trend_pivot.index)))
            ax3.set_yticklabels([idx.replace('_', ' ') for idx in dev_trend_pivot.index])
            ax3.set_title('Price Deviation vs Trend Direction\n(Average Max Gain %)', fontweight='bold')
            
            for i in range(len(dev_trend_pivot.index)):
                for j in range(len(dev_trend_pivot.columns)):
                    value = dev_trend_pivot.iloc[i, j]
                    if not np.isnan(value):
                        ax3.text(j, i, f'{value:.1f}', ha='center', va='center', 
                               fontweight='bold', fontsize=8)
            
            plt.colorbar(im3, ax=ax3, label='Average Max Gain (%)')
        
        # Sample count distribution
        sample_counts = multi_analysis_df.groupby(['growth_quintile', 'trend_direction'])['cleaned_sample_count'].sum().unstack(fill_value=0)
        
        if not sample_counts.empty:
            im4 = ax4.imshow(sample_counts.values, cmap='Blues', aspect='auto')
            ax4.set_xticks(range(len(sample_counts.columns)))
            ax4.set_xticklabels(sample_counts.columns)
            ax4.set_yticks(range(len(sample_counts.index)))
            ax4.set_yticklabels([idx.replace('_', ' ') for idx in sample_counts.index])
            ax4.set_title('Sample Count Distribution\n(Growth vs Trend)', fontweight='bold')
            
            for i in range(len(sample_counts.index)):
                for j in range(len(sample_counts.columns)):
                    value = sample_counts.iloc[i, j]
                    ax4.text(j, i, f'{int(value)}', ha='center', va='center', 
                           fontweight='bold', fontsize=8)
            
            plt.colorbar(im4, ax=ax4, label='Sample Count')
        
        plt.tight_layout()
        plt.savefig(f"{output_dir}/multi_attribute_heatmaps.png", dpi=300, bbox_inches='tight')
        plt.close()
    
    print("  * All multi-attribute visualizations created")

def save_multi_attribute_results(forward_gains_df, multi_analysis_df, individual_analysis, output_dir):
    """
    Save all multi-attribute analysis results
    """
    print("\nSaving multi-attribute results...")
    
    # Save raw forward gains data
    forward_gains_df.to_csv(f"{output_dir}/forward_gains_multi_attribute.csv", index=False)
    print(f"  * Forward gains data: {len(forward_gains_df):,} samples")
    
    # Save multi-attribute analysis
    multi_analysis_df.to_csv(f"{output_dir}/multi_attribute_analysis.csv", index=False)
    print(f"  * Multi-attribute analysis: {len(multi_analysis_df)} combinations")
    
    # Save individual attribute analyses
    for attr_name, attr_df in individual_analysis.items():
        attr_df.to_csv(f"{output_dir}/individual_{attr_name}_analysis.csv", index=False)
        print(f"  * Individual {attr_name} analysis: {len(attr_df)} bins")
    
    # Create comprehensive summary
    summary_stats = {
        'total_samples': len(forward_gains_df),
        'unique_assets': forward_gains_df['asset'].nunique(),
        'date_range_start': forward_gains_df['entry_date'].min(),
        'date_range_end': forward_gains_df['entry_date'].max(),
        'avg_forward_max_gain': forward_gains_df['forward_max_gain'].mean(),
        'median_forward_max_gain': forward_gains_df['forward_max_gain'].median(),
        'multi_attribute_combinations': len(multi_analysis_df),
        'growth_quintiles_analyzed': forward_gains_df['growth_quintile'].nunique(),
        'deviation_quintiles_analyzed': forward_gains_df['deviation_quintile'].nunique(),
        'trend_directions': forward_gains_df['growth_trend'].nunique(),
        'avg_growth_rate': forward_gains_df['annual_growth_rate'].mean(),
        'avg_price_deviation': forward_gains_df['price_deviation'].mean()
    }
    
    summary_df = pd.DataFrame([summary_stats])
    summary_df.to_csv(f"{output_dir}/multi_attribute_summary.csv", index=False)
    print(f"  * Multi-attribute summary saved")
    
    # Asset-level breakdown
    asset_summary = forward_gains_df.groupby('asset').agg({
        'forward_max_gain': ['count', 'mean', 'median', 'std'],
        'annual_growth_rate': ['mean', 'std'],
        'price_deviation': ['mean', 'std'],
        'days_to_max_gain': 'mean'
    }).round(3)
    
    asset_summary.columns = ['_'.join(col).strip() for col in asset_summary.columns]
    asset_summary = asset_summary.reset_index()
    asset_summary.to_csv(f"{output_dir}/asset_level_multi_attribute_summary.csv", index=False)
    print(f"  * Asset-level summary: {len(asset_summary)} assets")

def print_multi_attribute_insights(forward_gains_df, multi_analysis_df, individual_analysis):
    """
    Print comprehensive insights from multi-attribute analysis
    """
    print("\n" + "="*80)
    print("MULTI-ATTRIBUTE GROWTH RATE ANALYSIS - KEY INSIGHTS")
    print("="*80)
    
    # Overall statistics
    print(f"\nDATASET OVERVIEW:")
    print(f"  Total samples: {len(forward_gains_df):,}")
    print(f"  Unique assets: {forward_gains_df['asset'].nunique()}")
    print(f"  Date range: {forward_gains_df['entry_date'].min().strftime('%Y-%m-%d')} to {forward_gains_df['entry_date'].max().strftime('%Y-%m-%d')}")
    print(f"  Average forward max gain: {forward_gains_df['forward_max_gain'].mean():.2f}%")
    print(f"  Multi-attribute combinations analyzed: {len(multi_analysis_df)}")
    
    # Individual attribute performance
    print(f"\nINDIVIDUAL ATTRIBUTE PERFORMANCE:")
    
    if len(individual_analysis['growth']) > 0:
        growth_df = individual_analysis['growth']
        best_growth = growth_df.loc[growth_df['avg_max_gain'].idxmax()]
        print(f"  Best Growth Quintile: {best_growth['quintile']} ({best_growth['avg_max_gain']:.2f}% avg gain)")
    
    if len(individual_analysis['deviation']) > 0:
        dev_df = individual_analysis['deviation']
        best_dev = dev_df.loc[dev_df['avg_max_gain'].idxmax()]
        print(f"  Best Deviation Quintile: {best_dev['quintile']} ({best_dev['avg_max_gain']:.2f}% avg gain)")
    
    if len(individual_analysis['trend']) > 0:
        trend_df = individual_analysis['trend']
        best_trend = trend_df.loc[trend_df['avg_max_gain'].idxmax()]
        print(f"  Best Trend Direction: {best_trend['trend']} ({best_trend['avg_max_gain']:.2f}% avg gain)")
    
    # Top multi-attribute combinations
    if len(multi_analysis_df) > 0:
        print(f"\nTOP MULTI-ATTRIBUTE COMBINATIONS:")
        top_combos = multi_analysis_df.nlargest(10, 'avg_max_gain')
        
        for i, (_, row) in enumerate(top_combos.iterrows()):
            print(f"  #{i+1}: {row['combined_bin'].replace('_', ' ')}")
            print(f"       Average Max Gain: {row['avg_max_gain']:.2f}%")
            print(f"       Sample Count: {row['cleaned_sample_count']}")
            print(f"       Success Rate (25%+): {row['success_rate_25pct']:.1f}%")
            print(f"       Growth: {row['avg_growth_rate']:.2f}% | Deviation: {row['avg_price_deviation']:.2f}%")
    
    # Attribute interaction analysis
    print(f"\nATTRIBUTE INTERACTION INSIGHTS:")
    
    if len(multi_analysis_df) > 0:
        # Growth rate vs deviation interaction
        high_growth_high_dev = multi_analysis_df[
            (multi_analysis_df['growth_quintile'] == 'Q5_Top_20pct') &
            (multi_analysis_df['deviation_quintile'] == 'D5_Top_20pct')
        ]
        
        high_growth_low_dev = multi_analysis_df[
            (multi_analysis_df['growth_quintile'] == 'Q5_Top_20pct') &
            (multi_analysis_df['deviation_quintile'] == 'D1_Bottom_20pct')
        ]
        
        if len(high_growth_high_dev) > 0 and len(high_growth_low_dev) > 0:
            high_high_avg = high_growth_high_dev['avg_max_gain'].mean()
            high_low_avg = high_growth_low_dev['avg_max_gain'].mean()
            
            print(f"  High Growth + High Deviation: {high_high_avg:.2f}% avg gain")
            print(f"  High Growth + Low Deviation: {high_low_avg:.2f}% avg gain")
            if high_high_avg > high_low_avg:
                print(f"  -> High deviation enhances high growth by {high_high_avg - high_low_avg:.2f}%")
            else:
                print(f"  -> Low deviation is better with high growth by {high_low_avg - high_high_avg:.2f}%")
        
        # Trend direction impact across different combinations
        up_trend_avg = multi_analysis_df[multi_analysis_df['trend_direction'] == 'UP']['avg_max_gain'].mean()
        down_trend_avg = multi_analysis_df[multi_analysis_df['trend_direction'] == 'DOWN']['avg_max_gain'].mean()
        
        print(f"\n  UP Trend Average: {up_trend_avg:.2f}%")
        print(f"  DOWN Trend Average: {down_trend_avg:.2f}%")
        print(f"  Trend Advantage: {abs(up_trend_avg - down_trend_avg):.2f}% for {'UP' if up_trend_avg > down_trend_avg else 'DOWN'}")
    
    # Sample distribution analysis
    print(f"\nSAMPLE DISTRIBUTION:")
    
    growth_dist = forward_gains_df['growth_quintile'].value_counts()
    print(f"  Growth quintiles: {dict(growth_dist)}")
    
    dev_dist = forward_gains_df['deviation_quintile'].value_counts()
    print(f"  Deviation quintiles: {dict(dev_dist)}")
    
    trend_dist = forward_gains_df['growth_trend'].value_counts()
    print(f"  Trend directions: {dict(trend_dist)}")
    
    # Reliability analysis
    if len(multi_analysis_df) > 0:
        reliable_combos = multi_analysis_df[multi_analysis_df['cleaned_sample_count'] >= 20]
        print(f"\n  Reliable combinations (20+ samples): {len(reliable_combos)}/{len(multi_analysis_df)}")
        
        if len(reliable_combos) > 0:
            print(f"  Average performance of reliable combinations: {reliable_combos['avg_max_gain'].mean():.2f}%")
    
    # Strategic recommendations
    print(f"\n" + "="*80)
    print("STRATEGIC RECOMMENDATIONS")
    print("="*80)
    
    if len(multi_analysis_df) > 0:
        # Find optimal combinations with sufficient sample size
        optimal_combos = multi_analysis_df[
            (multi_analysis_df['avg_max_gain'] > multi_analysis_df['avg_max_gain'].quantile(0.8)) &
            (multi_analysis_df['cleaned_sample_count'] >= 15)
        ].sort_values('avg_max_gain', ascending=False)
        
        if len(optimal_combos) > 0:
            print(f"\nOPTIMAL COMBINATIONS (high performance + sufficient samples):")
            for i, (_, row) in enumerate(optimal_combos.head(5).iterrows()):
                print(f"  #{i+1}: {row['combined_bin'].replace('_', ' ')}")
                print(f"       Expected Max Gain: {row['avg_max_gain']:.2f}%")
                print(f"       Success Rate (25%+): {row['success_rate_25pct']:.1f}%")
                print(f"       Sample Count: {row['cleaned_sample_count']}")
                print(f"       Risk Level: {'High' if row['std_max_gain'] > 50 else 'Moderate' if row['std_max_gain'] > 30 else 'Low'}")
        
        # Best single attribute strategies
        print(f"\nBEST SINGLE ATTRIBUTE STRATEGIES:")
        
        if len(individual_analysis['growth']) > 0:
            best_growth = individual_analysis['growth'].nlargest(1, 'avg_max_gain').iloc[0]
            print(f"  Growth Rate Focus: Target {best_growth['quintile'].replace('_', ' ')} ({best_growth['avg_max_gain']:.2f}% avg)")
        
        if len(individual_analysis['deviation']) > 0:
            best_dev = individual_analysis['deviation'].nlargest(1, 'avg_max_gain').iloc[0]
            print(f"  Price Deviation Focus: Target {best_dev['quintile'].replace('_', ' ')} ({best_dev['avg_max_gain']:.2f}% avg)")
        
        if len(individual_analysis['trend']) > 0:
            best_trend = individual_analysis['trend'].nlargest(1, 'avg_max_gain').iloc[0]
            print(f"  Trend Focus: Target {best_trend['trend']} trends ({best_trend['avg_max_gain']:.2f}% avg)")
    
    print(f"\nKEY INSIGHTS FOR TRADING:")
    
    # Multi-attribute advantage
    if len(multi_analysis_df) > 0:
        multi_best = multi_analysis_df['avg_max_gain'].max()
        single_best = max([
            individual_analysis['growth']['avg_max_gain'].max() if len(individual_analysis['growth']) > 0 else 0,
            individual_analysis['deviation']['avg_max_gain'].max() if len(individual_analysis['deviation']) > 0 else 0,
            individual_analysis['trend']['avg_max_gain'].max() if len(individual_analysis['trend']) > 0 else 0
        ])
        
        if multi_best > single_best:
            print(f"  1. Multi-attribute combinations outperform single attributes by {multi_best - single_best:.2f}%")
            print(f"     -> Combine growth rate, deviation, and trend for optimal results")
        else:
            print(f"  1. Single attribute focus may be sufficient (best single: {single_best:.2f}% vs multi: {multi_best:.2f}%)")
    
    # Growth rate insights
    if len(individual_analysis['growth']) > 0:
        growth_df = individual_analysis['growth']
        q1_performance = growth_df[growth_df['quintile'] == 'Q1_Bottom_20pct']['avg_max_gain'].iloc[0] if len(growth_df[growth_df['quintile'] == 'Q1_Bottom_20pct']) > 0 else None
        q5_performance = growth_df[growth_df['quintile'] == 'Q5_Top_20pct']['avg_max_gain'].iloc[0] if len(growth_df[growth_df['quintile'] == 'Q5_Top_20pct']) > 0 else None
        
        if q1_performance is not None and q5_performance is not None:
            if q5_performance > q1_performance:
                print(f"  2. High growth rates (Q5) outperform low growth (Q1) by {q5_performance - q1_performance:.2f}%")
                print(f"     -> Focus on high-growth stocks for maximum potential")
            else:
                print(f"  2. Low growth rates (Q1) outperform high growth (Q5) by {q1_performance - q5_performance:.2f}%")
                print(f"     -> Value strategy: target slower-growth, potentially undervalued stocks")
    
    # Price deviation insights
    if len(individual_analysis['deviation']) > 0:
        dev_df = individual_analysis['deviation']
        d1_performance = dev_df[dev_df['quintile'] == 'D1_Bottom_20pct']['avg_max_gain'].iloc[0] if len(dev_df[dev_df['quintile'] == 'D1_Bottom_20pct']) > 0 else None
        d5_performance = dev_df[dev_df['quintile'] == 'D5_Top_20pct']['avg_max_gain'].iloc[0] if len(dev_df[dev_df['quintile'] == 'D5_Top_20pct']) > 0 else None
        
        if d1_performance is not None and d5_performance is not None:
            if d5_performance > d1_performance:
                print(f"  3. High price deviation (D5) outperforms low deviation (D1) by {d5_performance - d1_performance:.2f}%")
                print(f"     -> Target stocks trading significantly above/below growth trend line")
            else:
                print(f"  3. Low price deviation (D1) outperforms high deviation (D5) by {d1_performance - d5_performance:.2f}%")
                print(f"     -> Target stocks trading close to their growth trend line")
    
    print(f"\nIMPLEMENTATION CHECKLIST:")
    print(f"  □ Screen for optimal growth rate quintiles")
    print(f"  □ Analyze price deviation from growth trend line")
    print(f"  □ Monitor growth rate trend direction (acceleration/deceleration)")
    print(f"  □ Combine all three attributes for maximum edge")
    print(f"  □ Ensure sufficient sample sizes for reliable strategies")
    print(f"  □ Consider risk-adjusted returns, not just average gains")

def main():
    print("MULTI-ATTRIBUTE GROWTH RATE ANALYSIS")
    print("=" * 60)
    print("Analysis: Growth quintiles + Price deviation quintiles + Trend direction")
    print("Objective: Find optimal combinations of multiple attributes for forward gains")
    print("Attributes: 5x5x2 = 50 possible combinations")
    
    # Setup directories
    v2_dir = "/Users/tim/IWLS-OPTIONS/IWLS_ANALYSIS_V2"
    
    if not os.path.exists(v2_dir):
        print("ERROR: IWLS_ANALYSIS_V2 directory not found.")
        return
    
    # Create output directory
    output_dir = os.path.join(v2_dir, "MULTI_ATTRIBUTE_GROWTH_ANALYSIS")
    os.makedirs(output_dir, exist_ok=True)
    
    print(f"\nOutput directory: {output_dir}")
    
    # Load all IWLS data
    all_data = load_all_iwls_data(v2_dir)
    
    if len(all_data) < 5:
        print(f"ERROR: Insufficient assets loaded ({len(all_data)}). Need at least 5 for analysis.")
        return
    
    print(f"\nAnalysis parameters:")
    print(f"  Assets loaded: {len(all_data)}")
    print(f"  Forward period: 365 days (1 year)")
    print(f"  Growth rate bins: Quintiles (5 equal-sized bins)")
    print(f"  Price deviation bins: Quintiles (5 equal-sized bins)")
    print(f"  Trend directions: UP, DOWN")
    print(f"  Total combinations: 5 x 5 x 2 = 50")
    print(f"  Outlier removal: Top 3 max gains removed from each bin")
    print(f"  Minimum samples per bin: 8")
    
    # Calculate forward gains for all assets
    forward_gains_df = calculate_forward_max_gains_multi_attribute(all_data, forward_days=365)
    
    if len(forward_gains_df) == 0:
        print("ERROR: No forward gains data generated.")
        return
    
    # Analyze multi-attribute combinations
    multi_analysis_df = analyze_multi_attribute_bins(forward_gains_df)
    
    # Analyze individual attributes for comparison
    individual_analysis = analyze_individual_attributes(forward_gains_df)
    
    # Create comprehensive visualizations
    create_multi_attribute_visualizations(forward_gains_df, multi_analysis_df, individual_analysis, output_dir)
    
    # Save all results
    save_multi_attribute_results(forward_gains_df, multi_analysis_df, individual_analysis, output_dir)
    
    # Print comprehensive insights
    print_multi_attribute_insights(forward_gains_df, multi_analysis_df, individual_analysis)
    
    print(f"\n" + "="*60)
    print("MULTI-ATTRIBUTE ANALYSIS COMPLETE")
    print("="*60)
    print(f"Results saved to: {output_dir}")
    print("\nFiles created:")
    print("  forward_gains_multi_attribute.csv (raw data)")
    print("  multi_attribute_analysis.csv (combination performance)")
    print("  individual_growth_analysis.csv (growth quintile performance)")
    print("  individual_deviation_analysis.csv (deviation quintile performance)")
    print("  individual_trend_analysis.csv (trend direction performance)")
    print("  multi_attribute_summary.csv (overall summary)")
    print("  asset_level_multi_attribute_summary.csv (per-asset breakdown)")
    print("  multi_attribute_performance.png (main performance charts)")
    print("  multi_attribute_heatmaps.png (interaction heatmaps)")
    
    print(f"\nKey Analysis Features:")
    print(f"   - Growth rates in quintiles (20% buckets)")
    print(f"   - Price deviation from growth line in quintiles (20% buckets)")
    print(f"   - Growth trend direction (UP/DOWN)")
    print(f"   - All combinations analyzed for maximum gains")
    print(f"   - Individual attribute performance for comparison")
    print(f"   - Statistical significance via sample counts")
    
    if len(forward_gains_df) > 0:
        print(f"\nDATASET SUMMARY:")
        print(f"   Total samples: {len(forward_gains_df):,}")
        print(f"   Assets analyzed: {forward_gains_df['asset'].nunique()}")
        print(f"   Multi-attribute combinations: {len(multi_analysis_df)}")
        print(f"   Average forward gain: {forward_gains_df['forward_max_gain'].mean():.2f}%")
        
        if len(multi_analysis_df) > 0:
            best_combo = multi_analysis_df.loc[multi_analysis_df['avg_max_gain'].idxmax()]
            print(f"\nBEST COMBINATION:")
            print(f"   {best_combo['combined_bin'].replace('_', ' ')}")
            print(f"   Average Max Gain: {best_combo['avg_max_gain']:.2f}%")
            print(f"   Sample Count: {best_combo['cleaned_sample_count']}")

if __name__ == "__main__":
    main()