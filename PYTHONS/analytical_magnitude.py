import pandas as pd
import numpy as np
import matplotlib.pyplot as plt
import os
import glob
from datetime import datetime, timedelta
from collections import defaultdict
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
            df = df.dropna()
            df['price_deviation'] = ((df['price'] / df['trend_line_value']) - 1) * 100
            df['absolute_deviation'] = df['price_deviation'].abs()
            all_results[asset_name] = df
            print(f"Loaded {asset_name}: {len(df)} data points")
        except Exception as e:
            print(f"Error loading {asset_name}: {str(e)}")
    
    return all_results

def calculate_daily_top5_absolute_deviations(all_results):
    """
    For each day, find the 5 assets with highest absolute deviation from IWLS growth line
    """
    print("\nCalculating daily top 5 absolute deviations...")
    
    # Get all unique dates across all assets
    all_dates = set()
    for df in all_results.values():
        all_dates.update(df['date'].tolist())
    
    all_dates = sorted(list(all_dates))
    print(f"Processing {len(all_dates)} unique dates...")
    
    daily_top5_data = []
    asset_membership_count = defaultdict(int)
    
    processed_count = 0
    
    for current_date in all_dates:
        processed_count += 1
        
        if processed_count % 500 == 0:
            print(f"  Processed {processed_count}/{len(all_dates)} dates...")
        
        # Get absolute deviations for all assets on this date
        daily_deviations = {}
        
        for asset_name, df in all_results.items():
            # Find data for this exact date
            asset_data = df[df['date'] == current_date]
            
            if len(asset_data) > 0:
                abs_deviation = asset_data['absolute_deviation'].iloc[0]
                raw_deviation = asset_data['price_deviation'].iloc[0]
                price = asset_data['price'].iloc[0]
                
                daily_deviations[asset_name] = {
                    'absolute_deviation': abs_deviation,
                    'raw_deviation': raw_deviation,
                    'price': price
                }
        
        # Skip days with insufficient data
        if len(daily_deviations) < 5:
            continue
        
        # Sort by absolute deviation (highest first)
        sorted_assets = sorted(daily_deviations.items(), 
                             key=lambda x: x[1]['absolute_deviation'], 
                             reverse=True)
        
        # Get top 5
        top5_assets = sorted_assets[:5]
        
        # Sum the absolute deviations of top 5
        sum_top5_deviations = sum(data['absolute_deviation'] for _, data in top5_assets)
        
        # Count membership for each asset
        for asset_name, _ in top5_assets:
            asset_membership_count[asset_name] += 1
        
        # Record daily data
        daily_data = {
            'date': current_date,
            'sum_top5_absolute_deviations': sum_top5_deviations,
            'num_assets_available': len(daily_deviations),
            'top5_assets': [asset for asset, _ in top5_assets],
            'top5_absolute_deviations': [data['absolute_deviation'] for _, data in top5_assets],
            'top5_raw_deviations': [data['raw_deviation'] for _, data in top5_assets],
            'individual_deviations': {asset: data['absolute_deviation'] for asset, data in top5_assets}
        }
        
        daily_top5_data.append(daily_data)
    
    print(f"Completed processing {len(daily_top5_data)} days with sufficient data")
    
    return daily_top5_data, asset_membership_count

def create_membership_analysis(asset_membership_count, total_days):
    """
    Analyze how often each asset appears in the top 5
    """
    membership_data = []
    
    for asset, days_in_top5 in asset_membership_count.items():
        percentage = (days_in_top5 / total_days) * 100
        
        membership_data.append({
            'asset': asset,
            'days_in_top5': days_in_top5,
            'total_days': total_days,
            'percentage_in_top5': percentage
        })
    
    # Sort by days in top 5 (descending)
    membership_df = pd.DataFrame(membership_data)
    membership_df = membership_df.sort_values('days_in_top5', ascending=False)
    
    return membership_df

def analyze_deviation_patterns(daily_top5_data):
    """
    Analyze patterns in the daily top 5 absolute deviations
    """
    if not daily_top5_data:
        return {}
    
    daily_df = pd.DataFrame(daily_top5_data)
    
    analysis = {
        'total_days': len(daily_df),
        'mean_sum': daily_df['sum_top5_absolute_deviations'].mean(),
        'median_sum': daily_df['sum_top5_absolute_deviations'].median(),
        'std_sum': daily_df['sum_top5_absolute_deviations'].std(),
        'min_sum': daily_df['sum_top5_absolute_deviations'].min(),
        'max_sum': daily_df['sum_top5_absolute_deviations'].max(),
        'date_range': (daily_df['date'].min(), daily_df['date'].max())
    }
    
    # Find periods of high deviation
    high_deviation_threshold = daily_df['sum_top5_absolute_deviations'].quantile(0.9)
    high_deviation_days = daily_df[daily_df['sum_top5_absolute_deviations'] >= high_deviation_threshold]
    
    analysis['high_deviation_threshold'] = high_deviation_threshold
    analysis['high_deviation_days'] = len(high_deviation_days)
    analysis['high_deviation_percentage'] = (len(high_deviation_days) / len(daily_df)) * 100
    
    return analysis

def create_visualizations(daily_top5_data, membership_df, analysis, output_dir):
    """
    Create comprehensive visualizations
    """
    if not daily_top5_data:
        print("No data to visualize")
        return
    
    daily_df = pd.DataFrame(daily_top5_data)
    
    fig, ((ax1, ax2), (ax3, ax4)) = plt.subplots(2, 2, figsize=(20, 12))
    
    # Plot 1: Sum of top 5 absolute deviations over time
    ax1.plot(daily_df['date'], daily_df['sum_top5_absolute_deviations'], 
             linewidth=1.5, color='darkblue', alpha=0.8)
    
    # Add mean and threshold lines
    mean_sum = analysis['mean_sum']
    high_threshold = analysis['high_deviation_threshold']
    
    ax1.axhline(y=mean_sum, color='red', linestyle='--', alpha=0.7, 
                label=f'Mean: {mean_sum:.1f}%')
    ax1.axhline(y=high_threshold, color='orange', linestyle='--', alpha=0.7, 
                label=f'90th Percentile: {high_threshold:.1f}%')
    
    # Highlight high deviation periods
    high_deviation_mask = daily_df['sum_top5_absolute_deviations'] >= high_threshold
    if high_deviation_mask.any():
        ax1.scatter(daily_df[high_deviation_mask]['date'], 
                   daily_df[high_deviation_mask]['sum_top5_absolute_deviations'],
                   color='red', s=20, alpha=0.7, zorder=5, label='High Deviation Days')
    
    ax1.set_title('Sum of Top 5 Absolute Deviations Over Time', fontweight='bold', fontsize=14)
    ax1.set_xlabel('Date')
    ax1.set_ylabel('Sum of Absolute Deviations (%)')
    ax1.legend()
    ax1.grid(True, alpha=0.3)
    
    # Plot 2: Top 20 assets by membership frequency
    top_20_members = membership_df.head(20)
    
    colors = plt.cm.viridis(np.linspace(0, 1, len(top_20_members)))
    bars = ax2.barh(range(len(top_20_members)), top_20_members['days_in_top5'], 
                    color=colors, alpha=0.8)
    
    ax2.set_yticks(range(len(top_20_members)))
    ax2.set_yticklabels(top_20_members['asset'])
    ax2.set_xlabel('Days in Top 5')
    ax2.set_title('Top 20 Assets by Days in Top 5 Absolute Deviation', fontweight='bold')
    ax2.grid(True, alpha=0.3)
    
    # Add percentage labels
    for i, (bar, pct) in enumerate(zip(bars, top_20_members['percentage_in_top5'])):
        ax2.text(bar.get_width() + analysis['total_days'] * 0.01, 
                bar.get_y() + bar.get_height()/2,
                f'{pct:.1f}%', va='center', fontsize=9)
    
    # Plot 3: Distribution of daily sums
    ax3.hist(daily_df['sum_top5_absolute_deviations'], bins=50, 
             alpha=0.7, color='steelblue', edgecolor='black')
    
    ax3.axvline(mean_sum, color='red', linestyle='--', linewidth=2, 
                label=f'Mean: {mean_sum:.1f}%')
    ax3.axvline(analysis['median_sum'], color='green', linestyle='--', linewidth=2, 
                label=f'Median: {analysis["median_sum"]:.1f}%')
    
    ax3.set_title('Distribution of Daily Sum of Top 5 Absolute Deviations', fontweight='bold')
    ax3.set_xlabel('Sum of Absolute Deviations (%)')
    ax3.set_ylabel('Frequency (Days)')
    ax3.legend()
    ax3.grid(True, alpha=0.3)
    
    # Plot 4: Membership percentage distribution
    ax4.hist(membership_df['percentage_in_top5'], bins=30, 
             alpha=0.7, color='orange', edgecolor='black')
    
    ax4.set_title('Distribution of Top 5 Membership Percentages', fontweight='bold')
    ax4.set_xlabel('Percentage of Days in Top 5 (%)')
    ax4.set_ylabel('Number of Assets')
    ax4.grid(True, alpha=0.3)
    
    # Add statistics text
    stats_text = f"""Summary Statistics:
    Total Days: {analysis['total_days']:,}
    Mean Sum: {mean_sum:.1f}%
    Std Dev: {analysis['std_sum']:.1f}%
    Max Sum: {analysis['max_sum']:.1f}%
    High Dev Days: {analysis['high_deviation_days']} ({analysis['high_deviation_percentage']:.1f}%)"""
    
    ax4.text(0.02, 0.98, stats_text, transform=ax4.transAxes, 
             verticalalignment='top', fontsize=10,
             bbox=dict(boxstyle='round', facecolor='wheat', alpha=0.8))
    
    plt.tight_layout()
    plt.savefig(f"{output_dir}/absolute_deviation_analysis.png", dpi=300, bbox_inches='tight')
    plt.close()

def create_time_series_detail_plot(daily_top5_data, analysis, output_dir):
    """
    Create detailed time series plot with rolling averages
    """
    if not daily_top5_data:
        return
    
    daily_df = pd.DataFrame(daily_top5_data)
    
    fig, (ax1, ax2) = plt.subplots(2, 1, figsize=(20, 12))
    
    # Plot 1: Main time series with rolling averages
    ax1.plot(daily_df['date'], daily_df['sum_top5_absolute_deviations'], 
             linewidth=1, color='lightblue', alpha=0.7, label='Daily Sum')
    
    # Add rolling averages
    window_30 = daily_df['sum_top5_absolute_deviations'].rolling(window=30, center=True).mean()
    window_90 = daily_df['sum_top5_absolute_deviations'].rolling(window=90, center=True).mean()
    
    ax1.plot(daily_df['date'], window_30, linewidth=2, color='blue', 
             label='30-Day Moving Average')
    ax1.plot(daily_df['date'], window_90, linewidth=3, color='darkblue', 
             label='90-Day Moving Average')
    
    # Add mean line
    ax1.axhline(y=analysis['mean_sum'], color='red', linestyle='--', alpha=0.8, 
                label=f'Overall Mean: {analysis["mean_sum"]:.1f}%')
    
    ax1.set_title('Sum of Top 5 Absolute Deviations: Detailed Time Series', 
                  fontweight='bold', fontsize=14)
    ax1.set_ylabel('Sum of Absolute Deviations (%)')
    ax1.legend()
    ax1.grid(True, alpha=0.3)
    
    # Plot 2: Number of available assets over time
    ax2.plot(daily_df['date'], daily_df['num_assets_available'], 
             linewidth=2, color='green', alpha=0.8)
    
    ax2.set_title('Number of Assets Available for Analysis Over Time', fontweight='bold')
    ax2.set_xlabel('Date')
    ax2.set_ylabel('Number of Assets')
    ax2.grid(True, alpha=0.3)
    
    plt.tight_layout()
    plt.savefig(f"{output_dir}/detailed_time_series.png", dpi=300, bbox_inches='tight')
    plt.close()

def save_results(daily_top5_data, membership_df, analysis, output_dir):
    """
    Save all analysis results to CSV files
    """
    # Save daily top 5 data
    daily_summary = []
    for day_data in daily_top5_data:
        daily_summary.append({
            'date': day_data['date'].strftime('%Y-%m-%d'),
            'sum_top5_absolute_deviations': day_data['sum_top5_absolute_deviations'],
            'num_assets_available': day_data['num_assets_available'],
            'top5_asset_1': day_data['top5_assets'][0] if len(day_data['top5_assets']) > 0 else '',
            'top5_asset_2': day_data['top5_assets'][1] if len(day_data['top5_assets']) > 1 else '',
            'top5_asset_3': day_data['top5_assets'][2] if len(day_data['top5_assets']) > 2 else '',
            'top5_asset_4': day_data['top5_assets'][3] if len(day_data['top5_assets']) > 3 else '',
            'top5_asset_5': day_data['top5_assets'][4] if len(day_data['top5_assets']) > 4 else '',
            'deviation_1': day_data['top5_absolute_deviations'][0] if len(day_data['top5_absolute_deviations']) > 0 else 0,
            'deviation_2': day_data['top5_absolute_deviations'][1] if len(day_data['top5_absolute_deviations']) > 1 else 0,
            'deviation_3': day_data['top5_absolute_deviations'][2] if len(day_data['top5_absolute_deviations']) > 2 else 0,
            'deviation_4': day_data['top5_absolute_deviations'][3] if len(day_data['top5_absolute_deviations']) > 3 else 0,
            'deviation_5': day_data['top5_absolute_deviations'][4] if len(day_data['top5_absolute_deviations']) > 4 else 0
        })
    
    daily_summary_df = pd.DataFrame(daily_summary)
    daily_summary_df.to_csv(f"{output_dir}/daily_top5_absolute_deviations.csv", index=False)
    
    # Save membership analysis
    membership_df.to_csv(f"{output_dir}/asset_membership_analysis.csv", index=False)
    
    # Save analysis summary
    analysis_summary = {
        'metric': ['total_days', 'mean_sum', 'median_sum', 'std_sum', 'min_sum', 'max_sum',
                  'high_deviation_threshold', 'high_deviation_days', 'high_deviation_percentage'],
        'value': [analysis['total_days'], analysis['mean_sum'], analysis['median_sum'], 
                 analysis['std_sum'], analysis['min_sum'], analysis['max_sum'],
                 analysis['high_deviation_threshold'], analysis['high_deviation_days'], 
                 analysis['high_deviation_percentage']]
    }
    
    analysis_summary_df = pd.DataFrame(analysis_summary)
    analysis_summary_df.to_csv(f"{output_dir}/analysis_summary.csv", index=False)
    
    return daily_summary_df

def print_comprehensive_summary(daily_top5_data, membership_df, analysis):
    """
    Print comprehensive summary of the analysis
    """
    print("\n" + "="*80)
    print("ABSOLUTE DEVIATION ANALYSIS SUMMARY")
    print("="*80)
    
    print(f"\nDATA OVERVIEW:")
    print(f"  Total days analyzed: {analysis['total_days']:,}")
    print(f"  Date range: {analysis['date_range'][0].strftime('%Y-%m-%d')} to {analysis['date_range'][1].strftime('%Y-%m-%d')}")
    print(f"  Assets covered: {len(membership_df)}")
    
    print(f"\nABSOLUTE DEVIATION STATISTICS:")
    print(f"  Mean daily sum of top 5: {analysis['mean_sum']:.2f}%")
    print(f"  Median daily sum of top 5: {analysis['median_sum']:.2f}%")
    print(f"  Standard deviation: {analysis['std_sum']:.2f}%")
    print(f"  Minimum daily sum: {analysis['min_sum']:.2f}%")
    print(f"  Maximum daily sum: {analysis['max_sum']:.2f}%")
    
    print(f"\nHIGH DEVIATION PERIODS (>90th percentile):")
    print(f"  Threshold: {analysis['high_deviation_threshold']:.2f}%")
    print(f"  High deviation days: {analysis['high_deviation_days']} ({analysis['high_deviation_percentage']:.1f}%)")
    
    print(f"\nTOP 10 MOST FREQUENT TOP 5 MEMBERS:")
    print("-" * 60)
    print(f"{'Asset':<8} {'Days in Top 5':<12} {'Percentage':<12} {'Frequency':<10}")
    print("-" * 60)
    
    for _, row in membership_df.head(10).iterrows():
        frequency = "Very High" if row['percentage_in_top5'] > 20 else \
                   "High" if row['percentage_in_top5'] > 10 else \
                   "Medium" if row['percentage_in_top5'] > 5 else "Low"
        
        print(f"{row['asset']:<8} {row['days_in_top5']:>11,} {row['percentage_in_top5']:>11.1f}% {frequency:<10}")
    
    print(f"\nMEMBERSHIP DISTRIBUTION:")
    high_freq = len(membership_df[membership_df['percentage_in_top5'] > 10])
    med_freq = len(membership_df[(membership_df['percentage_in_top5'] > 5) & 
                                (membership_df['percentage_in_top5'] <= 10)])
    low_freq = len(membership_df[membership_df['percentage_in_top5'] <= 5])
    
    print(f"  High frequency assets (>10%): {high_freq}")
    print(f"  Medium frequency assets (5-10%): {med_freq}")
    print(f"  Low frequency assets (≤5%): {low_freq}")
    
    # Find periods of maximum deviation
    if daily_top5_data:
        daily_df = pd.DataFrame(daily_top5_data)
        max_day = daily_df.loc[daily_df['sum_top5_absolute_deviations'].idxmax()]
        
        print(f"\nMAXIMUM DEVIATION DAY:")
        print(f"  Date: {max_day['date'].strftime('%Y-%m-%d')}")
        print(f"  Sum of deviations: {max_day['sum_top5_absolute_deviations']:.2f}%")
        print(f"  Top 5 assets: {', '.join(max_day['top5_assets'])}")
        print(f"  Individual deviations: {[f'{dev:.1f}%' for dev in max_day['top5_absolute_deviations']]}")

def main():
    print("Absolute Deviation from IWLS Growth Line Analysis")
    print("="*60)
    print("Finding the 5 assets furthest from their IWLS growth line each day")
    print("Analyzing membership frequency and sum magnitude over time")
    
    # Create output directory
    output_dir = "/Users/tim/IWLS-OPTIONS/ABSOLUTE_DEVIATION_ANALYSIS"
    os.makedirs(output_dir, exist_ok=True)
    print(f"\nOutput directory: {output_dir}")
    
    # Load all IWLS results
    all_results = load_all_iwls_results()
    if not all_results:
        return
    
    print(f"\nLoaded {len(all_results)} assets")
    
    # Calculate daily top 5 absolute deviations
    daily_top5_data, asset_membership_count = calculate_daily_top5_absolute_deviations(all_results)
    
    if not daily_top5_data:
        print("No data found for analysis!")
        return
    
    # Create membership analysis
    total_days = len(daily_top5_data)
    membership_df = create_membership_analysis(asset_membership_count, total_days)
    
    # Analyze deviation patterns
    analysis = analyze_deviation_patterns(daily_top5_data)
    
    # Create visualizations
    print(f"\nCreating visualizations...")
    create_visualizations(daily_top5_data, membership_df, analysis, output_dir)
    create_time_series_detail_plot(daily_top5_data, analysis, output_dir)
    
    # Save results
    daily_summary_df = save_results(daily_top5_data, membership_df, analysis, output_dir)
    
    # Print comprehensive summary
    print_comprehensive_summary(daily_top5_data, membership_df, analysis)
    
    print(f"\n" + "="*80)
    print("ABSOLUTE DEVIATION ANALYSIS COMPLETE")
    print("="*80)
    print("Files saved:")
    print("  - absolute_deviation_analysis.png (4-panel overview)")
    print("  - detailed_time_series.png (time series with moving averages)")
    print("  - daily_top5_absolute_deviations.csv (daily top 5 data)")
    print("  - asset_membership_analysis.csv (membership frequency)")
    print("  - analysis_summary.csv (summary statistics)")
    
    # Key insights
    if len(membership_df) > 0:
        most_frequent = membership_df.iloc[0]
        print(f"\nKEY INSIGHTS:")
        print(f"  Most frequent top 5 member: {most_frequent['asset']}")
        print(f"  Appeared {most_frequent['days_in_top5']} days ({most_frequent['percentage_in_top5']:.1f}%)")
        print(f"  Average daily sum: {analysis['mean_sum']:.1f}%")
        print(f"  Volatility in deviations: {analysis['std_sum']:.1f}% standard deviation")
        
        print(f"\nIMPLICATIONS:")
        print(f"  Assets with high membership frequency may have:")
        print(f"    - More volatile price movements")
        print(f"    - Less stable trend relationships")
        print(f"    - Higher potential for mean reversion opportunities")

if __name__ == "__main__":
    main()