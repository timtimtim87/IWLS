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
            all_results[asset_name] = df
            print(f"Loaded {asset_name}: {len(df)} data points")
        except Exception as e:
            print(f"Error loading {asset_name}: {str(e)}")
    
    return all_results

def load_corrected_ev_data():
    """
    Load the corrected expected values data
    """
    try:
        ev_df = pd.read_csv("/Users/tim/IWLS-OPTIONS/CORRECTED_EV_ANALYSIS/corrected_expected_values.csv")
        print(f"Loaded corrected EV data with {len(ev_df)} Z-score levels")
        return ev_df
    except:
        print("Could not load corrected EV data. Using Z-score proxy.")
        return None

def calculate_current_z_scores(all_results, date, lookback_days=252):
    """
    Calculate current Z-scores for all assets on a given date
    """
    z_scores = {}
    
    for asset_name, df in all_results.items():
        asset_data = df[df['date'] <= date].copy()
        
        if len(asset_data) < lookback_days + 50:
            continue
        
        recent_data = asset_data.tail(lookback_days)
        
        if len(recent_data) < 50:
            continue
        
        current_deviation = recent_data['price_deviation'].iloc[-1]
        historical_deviations = recent_data['price_deviation']
        
        mean_deviation = historical_deviations.mean()
        std_deviation = historical_deviations.std()
        
        if std_deviation > 0:
            z_score = (current_deviation - mean_deviation) / std_deviation
            z_scores[asset_name] = {
                'z_score': z_score,
                'price_deviation': current_deviation
            }
    
    return z_scores

def get_expected_values_for_z_scores(ev_df, z_scores):
    """
    Get expected values for current Z-scores using interpolation
    """
    if ev_df is None:
        # Fallback: use Z-scores directly as proxy for EV (higher is better)
        asset_evs = {}
        for asset, data in z_scores.items():
            z_score = data['z_score']
            expected_value = max(0, (-z_score * 10))  # More negative Z = higher EV
            asset_evs[asset] = {
                'z_score': z_score,
                'expected_value': expected_value,
                'price_deviation': data['price_deviation']
            }
        return asset_evs
    
    asset_evs = {}
    
    for asset, data in z_scores.items():
        z_score = data['z_score']
        
        if asset not in ev_df.columns:
            continue
        
        asset_ev_data = ev_df[['z_score', asset]].dropna()
        
        if len(asset_ev_data) < 3:
            continue
        
        z_values = asset_ev_data['z_score'].values
        ev_values = asset_ev_data[asset].values
        
        z_score_clamped = np.clip(z_score, z_values.min(), z_values.max())
        expected_value = np.interp(z_score_clamped, z_values, ev_values)
        
        asset_evs[asset] = {
            'z_score': z_score,
            'expected_value': expected_value,
            'price_deviation': data['price_deviation']
        }
    
    return asset_evs

def get_top_5_ev_assets(asset_evs):
    """
    Get the 5 assets with HIGHEST expected values (best opportunities)
    """
    if len(asset_evs) < 5:
        return []
    
    # Sort by expected value - HIGHEST first (best opportunities)
    sorted_assets = sorted(asset_evs.items(), key=lambda x: x[1]['expected_value'], reverse=True)
    
    # Return top 5
    return sorted_assets[:5]

def track_top5_membership(all_results, ev_df, start_date, end_date):
    """
    Track which assets are in the top 5 EV group each day and calculate combined EV
    """
    print(f"\nTracking Top 5 EV Assets Daily Membership")
    print(f"Period: {start_date.strftime('%Y-%m-%d')} to {end_date.strftime('%Y-%m-%d')}")
    
    # Generate all business days
    all_dates = pd.date_range(start=start_date, end=end_date, freq='D')
    
    # Track membership
    daily_data = []
    membership_counter = defaultdict(int)  # Count days each asset is in top 5
    
    processed_days = 0
    
    for current_date in all_dates:
        processed_days += 1
        
        # Progress update
        if processed_days % 100 == 0:
            print(f"  Processed {processed_days:,}/{len(all_dates):,} days...")
        
        # Calculate Z-scores for all assets on this date
        current_z_scores = calculate_current_z_scores(all_results, current_date)
        
        if len(current_z_scores) < 5:
            # Not enough data for this date
            daily_data.append({
                'date': current_date,
                'num_assets_available': len(current_z_scores),
                'top5_assets': [],
                'top5_evs': [],
                'combined_top5_ev': np.nan,
                'avg_top5_ev': np.nan
            })
            continue
        
        # Get expected values for all assets
        asset_evs = get_expected_values_for_z_scores(ev_df, current_z_scores)
        
        if len(asset_evs) < 5:
            # Not enough EV data for this date
            daily_data.append({
                'date': current_date,
                'num_assets_available': len(asset_evs),
                'top5_assets': [],
                'top5_evs': [],
                'combined_top5_ev': np.nan,
                'avg_top5_ev': np.nan
            })
            continue
        
        # Get top 5 highest EV assets
        top5_assets = get_top_5_ev_assets(asset_evs)
        
        # Extract data
        top5_asset_names = [asset for asset, _ in top5_assets]
        top5_evs = [data['expected_value'] for _, data in top5_assets]
        
        # Update membership counter
        for asset in top5_asset_names:
            membership_counter[asset] += 1
        
        # Calculate combined and average EV
        combined_top5_ev = sum(top5_evs)
        avg_top5_ev = np.mean(top5_evs)
        
        daily_data.append({
            'date': current_date,
            'num_assets_available': len(asset_evs),
            'top5_assets': top5_asset_names,
            'top5_evs': top5_evs,
            'combined_top5_ev': combined_top5_ev,
            'avg_top5_ev': avg_top5_ev
        })
    
    daily_df = pd.DataFrame(daily_data)
    
    return daily_df, membership_counter

def create_membership_analysis(membership_counter, daily_df, output_dir):
    """
    Create comprehensive analysis of top 5 membership
    """
    # Calculate total valid days
    valid_data = daily_df.dropna(subset=['combined_top5_ev'])
    total_valid_days = len(valid_data)
    
    # Create membership summary
    membership_data = []
    
    for asset, days_in_top5 in membership_counter.items():
        percentage = (days_in_top5 / total_valid_days) * 100 if total_valid_days > 0 else 0
        
        membership_data.append({
            'asset': asset,
            'days_in_top5': days_in_top5,
            'total_valid_days': total_valid_days,
            'percentage_in_top5': percentage
        })
    
    # Sort by days in top 5 (descending)
    membership_df = pd.DataFrame(membership_data)
    membership_df = membership_df.sort_values('days_in_top5', ascending=False)
    
    # Save to CSV
    membership_df.to_csv(f"{output_dir}/top5_ev_membership_summary.csv", index=False)
    
    return membership_df

def analyze_combined_ev_patterns(daily_df):
    """
    Analyze patterns in the combined EV of top 5 assets
    """
    print("\n" + "="*80)
    print("TOP 5 EV ASSETS ANALYSIS")
    print("="*80)
    
    # Basic statistics
    valid_data = daily_df.dropna(subset=['combined_top5_ev'])
    
    if len(valid_data) == 0:
        print("No valid data found!")
        return
    
    print(f"\nDATA COVERAGE:")
    print(f"Total days analyzed: {len(daily_df):,}")
    print(f"Days with valid data: {len(valid_data):,} ({len(valid_data)/len(daily_df)*100:.1f}%)")
    print(f"Average assets available per day: {valid_data['num_assets_available'].mean():.1f}")
    
    # Combined EV statistics
    print(f"\nCOMBINED EV STATISTICS (Top 5 Highest EV Assets):")
    print(f"  Mean: {valid_data['combined_top5_ev'].mean():.2f}")
    print(f"  Median: {valid_data['combined_top5_ev'].median():.2f}")
    print(f"  Std Dev: {valid_data['combined_top5_ev'].std():.2f}")
    print(f"  Min: {valid_data['combined_top5_ev'].min():.2f}")
    print(f"  Max: {valid_data['combined_top5_ev'].max():.2f}")
    
    # Entry signal analysis (when combined EV >= 70)
    entry_threshold = 70
    entry_days = valid_data[valid_data['combined_top5_ev'] >= entry_threshold]
    
    print(f"\nENTRY SIGNAL ANALYSIS (Combined EV ≥ {entry_threshold}):")
    print(f"  Days meeting criteria: {len(entry_days):,} ({len(entry_days)/len(valid_data)*100:.1f}%)")
    
    if len(entry_days) > 0:
        print(f"  Average combined EV on entry days: {entry_days['combined_top5_ev'].mean():.2f}")
        print(f"  Highest combined EV: {entry_days['combined_top5_ev'].max():.2f}")
        print(f"  Date range of entry signals: {entry_days['date'].min().strftime('%Y-%m-%d')} to {entry_days['date'].max().strftime('%Y-%m-%d')}")
        
        # Show some example entry periods
        print(f"\nEXAMPLE ENTRY SIGNAL PERIODS:")
        example_entries = entry_days.nlargest(5, 'combined_top5_ev')
        for _, row in example_entries.iterrows():
            print(f"  {row['date'].strftime('%Y-%m-%d')}: Combined EV = {row['combined_top5_ev']:.2f}")
            print(f"    Top 5 assets: {', '.join(row['top5_assets'])}")
    else:
        print(f"  WARNING: No days found with combined EV ≥ {entry_threshold}")
        print(f"  Consider lowering threshold. Max combined EV was: {valid_data['combined_top5_ev'].max():.2f}")
    
    return valid_data

def create_top5_visualization(daily_df, membership_df, output_dir):
    """
    Create visualizations for top 5 EV analysis
    """
    valid_data = daily_df.dropna(subset=['combined_top5_ev'])
    
    if len(valid_data) == 0:
        print("No valid data for visualization")
        return
    
    fig, ((ax1, ax2), (ax3, ax4)) = plt.subplots(2, 2, figsize=(20, 12))
    
    # Plot 1: Combined EV over time with entry threshold
    ax1.plot(valid_data['date'], valid_data['combined_top5_ev'], 
             linewidth=1, color='blue', alpha=0.7, label='Combined EV (Top 5)')
    
    # Add entry threshold line
    entry_threshold = 70
    ax1.axhline(y=entry_threshold, color='red', linestyle='--', linewidth=2, 
                label=f'Entry Threshold ({entry_threshold})')
    
    # Highlight entry periods
    entry_days = valid_data[valid_data['combined_top5_ev'] >= entry_threshold]
    if len(entry_days) > 0:
        ax1.scatter(entry_days['date'], entry_days['combined_top5_ev'], 
                   color='red', s=30, alpha=0.8, label='Entry Signals')
    
    # Add rolling average
    rolling_mean = valid_data['combined_top5_ev'].rolling(window=30, center=True).mean()
    ax1.plot(valid_data['date'], rolling_mean, 
             linewidth=2, color='green', label='30-day Moving Average')
    
    ax1.set_title('Daily Combined EV of Top 5 Highest EV Assets', fontweight='bold')
    ax1.set_xlabel('Date')
    ax1.set_ylabel('Combined Expected Value')
    ax1.legend()
    ax1.grid(True, alpha=0.3)
    
    # Plot 2: Membership frequency (top 20 assets)
    top_members = membership_df.head(20)
    bars = ax2.barh(range(len(top_members)), top_members['days_in_top5'], 
                    color='steelblue', alpha=0.7)
    ax2.set_yticks(range(len(top_members)))
    ax2.set_yticklabels(top_members['asset'])
    ax2.set_xlabel('Days in Top 5')
    ax2.set_title('Top 20 Assets by Days in Top 5 EV Group', fontweight='bold')
    ax2.grid(True, alpha=0.3)
    
    # Add percentage labels
    for i, (bar, pct) in enumerate(zip(bars, top_members['percentage_in_top5'])):
        ax2.text(bar.get_width() + 5, bar.get_y() + bar.get_height()/2,
                f'{pct:.1f}%', va='center', fontsize=9)
    
    # Plot 3: Distribution of combined EV
    ax3.hist(valid_data['combined_top5_ev'], bins=50, alpha=0.7, color='blue', edgecolor='black')
    ax3.axvline(valid_data['combined_top5_ev'].mean(), color='red', linestyle='--', 
                label=f'Mean: {valid_data["combined_top5_ev"].mean():.2f}')
    ax3.axvline(entry_threshold, color='green', linestyle='--', 
                label=f'Entry Threshold: {entry_threshold}')
    
    ax3.set_title('Distribution of Daily Combined EV Values', fontweight='bold')
    ax3.set_xlabel('Combined Expected Value')
    ax3.set_ylabel('Frequency (Days)')
    ax3.legend()
    ax3.grid(True, alpha=0.3)
    
    # Plot 4: Membership percentage distribution
    ax4.hist(membership_df['percentage_in_top5'], bins=30, alpha=0.7, color='orange', edgecolor='black')
    ax4.set_title('Distribution of Top 5 Membership Percentages', fontweight='bold')
    ax4.set_xlabel('Percentage of Days in Top 5 (%)')
    ax4.set_ylabel('Number of Assets')
    ax4.grid(True, alpha=0.3)
    
    plt.tight_layout()
    plt.savefig(f"{output_dir}/top5_ev_analysis.png", dpi=300, bbox_inches='tight')
    plt.close()

def save_daily_data(daily_df, output_dir):
    """
    Save daily top 5 data
    """
    # Create a simplified daily summary
    daily_summary = []
    
    for _, row in daily_df.iterrows():
        if not pd.isna(row['combined_top5_ev']):
            daily_summary.append({
                'date': row['date'].strftime('%Y-%m-%d'),
                'combined_top5_ev': row['combined_top5_ev'],
                'avg_top5_ev': row['avg_top5_ev'],
                'num_assets_available': row['num_assets_available'],
                'top5_asset_1': row['top5_assets'][0] if len(row['top5_assets']) > 0 else '',
                'top5_asset_2': row['top5_assets'][1] if len(row['top5_assets']) > 1 else '',
                'top5_asset_3': row['top5_assets'][2] if len(row['top5_assets']) > 2 else '',
                'top5_asset_4': row['top5_assets'][3] if len(row['top5_assets']) > 3 else '',
                'top5_asset_5': row['top5_assets'][4] if len(row['top5_assets']) > 4 else '',
                'entry_signal': 'YES' if row['combined_top5_ev'] >= 70 else 'NO'
            })
    
    daily_summary_df = pd.DataFrame(daily_summary)
    daily_summary_df.to_csv(f"{output_dir}/daily_top5_ev_data.csv", index=False)

def main():
    print("Top 5 Highest EV Assets Tracking Analysis")
    print("="*60)
    print("Tracking which assets are in the top 5 highest EV group each day")
    print("Goal: Identify entry signals when combined EV ≥ 70")
    
    # Create output directory
    output_dir = "/Users/tim/IWLS-OPTIONS/TOP5_EV_TRACKING"
    os.makedirs(output_dir, exist_ok=True)
    print(f"\nOutput directory: {output_dir}")
    
    # Load data
    all_results = load_all_iwls_results()
    if not all_results:
        return
    
    ev_df = load_corrected_ev_data()
    
    # Determine date range
    all_dates = []
    for df in all_results.values():
        all_dates.extend(df['date'].tolist())
    
    start_date = min(all_dates)
    end_date = max(all_dates)
    min_start_date = start_date + timedelta(days=730)  # Need 2+ years of history for Z-scores
    
    print(f"\nData range: {start_date.strftime('%Y-%m-%d')} to {end_date.strftime('%Y-%m-%d')}")
    print(f"Analysis period: {min_start_date.strftime('%Y-%m-%d')} to {end_date.strftime('%Y-%m-%d')}")
    
    # Track top 5 membership
    print(f"\n{'='*20} TRACKING TOP 5 EV MEMBERSHIP {'='*20}")
    daily_df, membership_counter = track_top5_membership(
        all_results, ev_df, min_start_date, end_date
    )
    
    # Create membership analysis
    membership_df = create_membership_analysis(membership_counter, daily_df, output_dir)
    
    # Analyze combined EV patterns
    valid_data = analyze_combined_ev_patterns(daily_df)
    
    # Create visualizations
    print(f"\nCreating visualizations...")
    create_top5_visualization(daily_df, membership_df, output_dir)
    
    # Save daily data
    save_daily_data(daily_df, output_dir)
    
    print(f"\n" + "="*80)
    print("TOP 5 EV TRACKING ANALYSIS COMPLETE")
    print("="*80)
    print("Files saved:")
    print("  - top5_ev_membership_summary.csv (how many days each asset was in top 5)")
    print("  - daily_top5_ev_data.csv (daily top 5 assets and combined EV)")
    print("  - top5_ev_analysis.png (4-panel visualization)")
    
    # Print top performers
    if len(membership_df) > 0:
        print(f"\nTOP 10 MOST FREQUENT TOP 5 MEMBERS:")
        print("-" * 50)
        print(f"{'Asset':<8} {'Days in Top 5':<12} {'Percentage':<10}")
        print("-" * 50)
        
        for _, row in membership_df.head(10).iterrows():
            print(f"{row['asset']:<8} {row['days_in_top5']:>11,} {row['percentage_in_top5']:>9.1f}%")
        
        # Entry signal summary
        if valid_data is not None and len(valid_data) > 0:
            entry_days = len(valid_data[valid_data['combined_top5_ev'] >= 70])
            print(f"\nENTRY SIGNAL SUMMARY:")
            print(f"Days with combined EV ≥ 70: {entry_days:,} ({entry_days/len(valid_data)*100:.1f}%)")
            print(f"Max combined EV reached: {valid_data['combined_top5_ev'].max():.2f}")

if __name__ == "__main__":
    main()