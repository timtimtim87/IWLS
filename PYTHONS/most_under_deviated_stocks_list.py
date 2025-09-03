import pandas as pd
import numpy as np
import matplotlib.pyplot as plt
from datetime import datetime, timedelta
import warnings
import os
warnings.filterwarnings('ignore')

# =============================================================================
# CONFIGURABLE PARAMETERS
# =============================================================================

TOP_N_STOCKS = 10           # Number of most underperforming stocks to track daily
MIN_DEVIATION_THRESHOLD = -5  # Only consider stocks with deviation worse than this
DATE_STEP = 1               # Generate data every N days (1 = daily)

# Date range for generating the dataset
DATASET_START_DATE = datetime(2021, 12, 1)   # Start from end of 2021
DATASET_END_DATE = datetime(2025, 5, 31)     # Through May 2025

# =============================================================================

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
                  not d.startswith('DAILY_TRACKING') and not d.startswith('MULTI') and
                  not d.startswith('OPTIONS_STRATEGY') and not d.startswith('BULL_CALL')]
    
    for asset_name in asset_dirs:
        asset_dir = os.path.join(v2_dir, asset_name)
        iwls_file = os.path.join(asset_dir, f"{asset_name}_iwls_results.csv")
        
        if os.path.exists(iwls_file):
            try:
                df = pd.read_csv(iwls_file)
                df['date'] = pd.to_datetime(df['date'])
                df = df.dropna(subset=['price_deviation', 'price']).sort_values('date').reset_index(drop=True)
                
                if len(df) > 100:  # Lower threshold for dataset generation
                    all_data[asset_name] = df
                    
            except Exception as e:
                print(f"  Warning: Error loading {asset_name}: {e}")
    
    print(f"Successfully loaded {len(all_data)} assets")
    return all_data

def get_available_assets_on_date(all_data, target_date, lookback_days=3):
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
            
            # Only include if the data is reasonably fresh (within lookback window)
            days_diff = abs((closest_row['date'] - target_date).days)
            if days_diff <= lookback_days:
                available_assets.append({
                    'date': target_date,
                    'data_date': closest_row['date'],
                    'asset': asset_name,
                    'price': closest_row['price'],
                    'price_deviation': closest_row['price_deviation'],
                    'absolute_deviation': abs(closest_row['price_deviation']),
                    'days_from_target': days_diff
                })
    
    return pd.DataFrame(available_assets)

def generate_daily_top_underperformers(all_iwls_data, output_file):
    """
    Generate daily rankings of top underperforming stocks
    """
    print(f"\nGenerating daily top {TOP_N_STOCKS} underperformers dataset...")
    print(f"Date range: {DATASET_START_DATE.strftime('%Y-%m-%d')} to {DATASET_END_DATE.strftime('%Y-%m-%d')}")
    print(f"Minimum deviation threshold: {MIN_DEVIATION_THRESHOLD}%")
    
    # Generate date range
    current_date = DATASET_START_DATE
    all_daily_data = []
    processed_dates = 0
    
    while current_date <= DATASET_END_DATE:
        # Only process weekdays
        if current_date.weekday() < 5:  # Monday=0, Friday=4
            
            # Get available assets for this date
            available_assets = get_available_assets_on_date(all_iwls_data, current_date)
            
            if len(available_assets) > 0:
                # Filter for underperforming stocks (negative deviation below threshold)
                underperforming = available_assets[
                    available_assets['price_deviation'] < MIN_DEVIATION_THRESHOLD
                ].copy()
                
                if len(underperforming) >= TOP_N_STOCKS:
                    # Sort by most negative deviation and take top N
                    top_underperformers = underperforming.nsmallest(TOP_N_STOCKS, 'price_deviation')
                    
                    # Add ranking
                    top_underperformers['rank'] = range(1, len(top_underperformers) + 1)
                    
                    all_daily_data.append(top_underperformers)
                    processed_dates += 1
                    
                    if processed_dates % 100 == 0:
                        print(f"    Processed {processed_dates} dates... Latest: {current_date.strftime('%Y-%m-%d')}")
                        avg_assets = len(available_assets)
                        avg_under = len(underperforming)
                        print(f"      Avg assets available: {avg_assets}, Avg underperforming: {avg_under}")
        
        current_date += timedelta(days=DATE_STEP)
    
    if len(all_daily_data) > 0:
        # Combine all data
        combined_df = pd.concat(all_daily_data, ignore_index=True)
        
        # Sort by date and rank
        combined_df = combined_df.sort_values(['date', 'rank']).reset_index(drop=True)
        
        # Save to CSV
        combined_df.to_csv(output_file, index=False)
        
        print(f"\nDaily underperformers dataset created!")
        print(f"  Total records: {len(combined_df):,}")
        print(f"  Date range: {combined_df['date'].min().strftime('%Y-%m-%d')} to {combined_df['date'].max().strftime('%Y-%m-%d')}")
        print(f"  Trading days: {combined_df['date'].nunique():,}")
        print(f"  Unique assets: {combined_df['asset'].nunique()}")
        print(f"  File saved: {output_file}")
        
        return combined_df
    else:
        print("No data generated. Check date ranges and thresholds.")
        return pd.DataFrame()

def analyze_underperformers_dataset(df, output_file):
    """
    Create analysis and visualization of the underperformers dataset
    Parameters:
        df (DataFrame): The underperformers dataset
        output_file (str): Path to the output CSV file
    """
    if len(df) == 0:
        return
        
    print("\nAnalyzing underperformers dataset...")
    
    # Basic statistics
    print(f"\nDATASET STATISTICS:")
    print(f"  Total records: {len(df):,}")
    print(f"  Trading days covered: {df['date'].nunique():,}")
    print(f"  Unique assets that appeared: {df['asset'].nunique()}")
    print(f"  Average deviation: {df['price_deviation'].mean():.2f}%")
    print(f"  Median deviation: {df['price_deviation'].median():.2f}%")
    print(f"  Worst deviation: {df['price_deviation'].min():.2f}%")
    
    # Most frequently underperforming stocks
    asset_counts = df['asset'].value_counts().head(20)
    print(f"\nMOST FREQUENTLY UNDERPERFORMING STOCKS:")
    print(f"{'Rank':<5} {'Asset':<8} {'Times':<6} {'Avg Dev':<10} {'Worst Dev':<12}")
    print("-" * 50)
    
    for i, (asset, count) in enumerate(asset_counts.items(), 1):
        asset_data = df[df['asset'] == asset]
        avg_dev = asset_data['price_deviation'].mean()
        worst_dev = asset_data['price_deviation'].min()
        pct_of_days = (count / df['date'].nunique()) * 100
        print(f"{i:<5} {asset:<8} {count:<6} {avg_dev:<10.2f} {worst_dev:<12.2f}")
        if i <= 10:  # Show percentage for top 10
            print(f"      ({pct_of_days:.1f}% of trading days)")
    
    # Create visualizations
    output_dir = os.path.dirname(output_file)
    
    fig, ((ax1, ax2), (ax3, ax4)) = plt.subplots(2, 2, figsize=(16, 12))
    
    # Plot 1: Distribution of price deviations
    ax1.hist(df['price_deviation'], bins=50, alpha=0.7, edgecolor='black', color='red')
    ax1.axvline(df['price_deviation'].mean(), color='blue', linestyle='--', 
                label=f'Mean: {df["price_deviation"].mean():.1f}%')
    ax1.axvline(df['price_deviation'].median(), color='orange', linestyle='--', 
                label=f'Median: {df["price_deviation"].median():.1f}%')
    ax1.set_xlabel('Price Deviation (%)')
    ax1.set_ylabel('Frequency')
    ax1.set_title('Distribution of Price Deviations (Top Underperformers)', fontweight='bold')
    ax1.legend()
    ax1.grid(True, alpha=0.3)
    
    # Plot 2: Most frequently underperforming assets
    top_assets = asset_counts.head(15)
    ax2.bar(range(len(top_assets)), top_assets.values, alpha=0.7, color='darkred')
    ax2.set_xticks(range(len(top_assets)))
    ax2.set_xticklabels(top_assets.index, rotation=45, ha='right')
    ax2.set_ylabel('Number of Times in Top 10')
    ax2.set_title('Most Frequently Underperforming Assets', fontweight='bold')
    ax2.grid(True, alpha=0.3)
    
    # Plot 3: Average deviation by rank
    rank_analysis = df.groupby('rank')['price_deviation'].agg(['mean', 'median', 'std']).reset_index()
    ax3.plot(rank_analysis['rank'], rank_analysis['mean'], marker='o', label='Mean', linewidth=2)
    ax3.plot(rank_analysis['rank'], rank_analysis['median'], marker='s', label='Median', linewidth=2)
    ax3.fill_between(rank_analysis['rank'], 
                     rank_analysis['mean'] - rank_analysis['std'],
                     rank_analysis['mean'] + rank_analysis['std'],
                     alpha=0.3, label='±1 StdDev')
    ax3.set_xlabel('Underperformer Rank')
    ax3.set_ylabel('Price Deviation (%)')
    ax3.set_title('Price Deviation by Underperformer Rank', fontweight='bold')
    ax3.legend()
    ax3.grid(True, alpha=0.3)
    ax3.set_xticks(range(1, TOP_N_STOCKS + 1))
    
    # Plot 4: Underperformance over time (monthly aggregation)
    monthly_data = df.copy()
    monthly_data['year_month'] = monthly_data['date'].dt.to_period('M')
    monthly_stats = monthly_data.groupby('year_month')['price_deviation'].agg(['mean', 'count']).reset_index()
    monthly_stats = monthly_stats[monthly_stats['count'] >= 50]  # Only months with sufficient data
    
    if len(monthly_stats) > 0:
        ax4.plot(range(len(monthly_stats)), monthly_stats['mean'], marker='o', linewidth=2, color='red')
        ax4.set_xticks(range(0, len(monthly_stats), max(1, len(monthly_stats)//12)))
        ax4.set_xticklabels([str(monthly_stats.iloc[i]['year_month']) for i in range(0, len(monthly_stats), max(1, len(monthly_stats)//12))],
                           rotation=45)
        ax4.set_ylabel('Average Price Deviation (%)')
        ax4.set_title('Average Underperformance Over Time', fontweight='bold')
        ax4.grid(True, alpha=0.3)
        
        # Add trend line
        x = np.arange(len(monthly_stats))
        z = np.polyfit(x, monthly_stats['mean'], 1)
        p = np.poly1d(z)
        ax4.plot(x, p(x), "r--", alpha=0.8, label=f'Trend: {z[0]:.3f}% per month')
        ax4.legend()
    
    plt.tight_layout()
    plt.savefig(f"{output_dir}/underperformers_analysis.png", dpi=300, bbox_inches='tight')
    plt.close()
    
    print(f"  Analysis plots saved: {output_dir}/underperformers_analysis.png")

def load_or_generate_underperformers_dataset(v2_dir, force_regenerate=False):
    """
    Load existing underperformers dataset or generate it if it doesn't exist
    """
    output_file = os.path.join(v2_dir, f"DAILY_TOP_{TOP_N_STOCKS}_UNDERPERFORMERS.csv")
    
    # Check if dataset already exists
    if os.path.exists(output_file) and not force_regenerate:
        print(f"Found existing underperformers dataset: {output_file}")
        try:
            df = pd.read_csv(output_file)
            df['date'] = pd.to_datetime(df['date'])
            df['data_date'] = pd.to_datetime(df['data_date'])
            
            print(f"Loaded existing dataset:")
            print(f"  Records: {len(df):,}")
            print(f"  Date range: {df['date'].min().strftime('%Y-%m-%d')} to {df['date'].max().strftime('%Y-%m-%d')}")
            print(f"  Trading days: {df['date'].nunique():,}")
            print(f"  Unique assets: {df['asset'].nunique()}")
            
            return df, output_file
            
        except Exception as e:
            print(f"Error loading existing dataset: {e}")
            print("Will regenerate...")
    
    # Generate new dataset
    print("Generating new underperformers dataset...")
    all_iwls_data = load_all_iwls_data(v2_dir)
    
    if len(all_iwls_data) == 0:
        print("ERROR: No IWLS data loaded.")
        return pd.DataFrame(), output_file
    
    df = generate_daily_top_underperformers(all_iwls_data, output_file)
    return df, output_file

def main():
    print("DAILY TOP UNDERPERFORMERS DATASET GENERATOR")
    print("=" * 60)
    print(f"Generating daily rankings of top {TOP_N_STOCKS} underperforming stocks")
    print(f"Minimum deviation threshold: {MIN_DEVIATION_THRESHOLD}%")
    print(f"Date range: {DATASET_START_DATE.strftime('%Y-%m-%d')} to {DATASET_END_DATE.strftime('%Y-%m-%d')}")
    
    # Setup directories
    v2_dir = "/Users/tim/CODE_PROJECTS/IWLS-OPTIONS/IWLS_ANALYSIS_V2"
    
    if not os.path.exists(v2_dir):
        print(f"ERROR: IWLS_ANALYSIS_V2 directory not found: {v2_dir}")
        return
    
    print(f"\nWorking directory: {v2_dir}")
    
    # Ask user if they want to force regeneration
    force_regenerate = False
    output_file = os.path.join(v2_dir, f"DAILY_TOP_{TOP_N_STOCKS}_UNDERPERFORMERS.csv")
    
    if os.path.exists(output_file):
        print(f"\nExisting dataset found: {output_file}")
        response = input("Regenerate dataset? (y/N): ").strip().lower()
        force_regenerate = response in ['y', 'yes']
    
    # Load or generate dataset
    df, output_file = load_or_generate_underperformers_dataset(v2_dir, force_regenerate)
    
    if len(df) == 0:
        print("ERROR: No dataset generated or loaded.")
        return
    
    # Analyze the dataset
    analyze_underperformers_dataset(df, output_file)
    
    print(f"\nDataset ready for use in strategies!")
    print(f"File location: {output_file}")
    print(f"\nTo use this in your bull call spread strategy:")
    print(f"  1. Load the CSV file instead of calculating underperformers on-the-fly")
    print(f"  2. Filter by date to get the top {TOP_N_STOCKS} underperformers for any given day")
    print(f"  3. Use these pre-ranked assets for options strategy selection")

if __name__ == "__main__":
    main()