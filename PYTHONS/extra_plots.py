import pandas as pd
import numpy as np
import matplotlib.pyplot as plt
import os
import glob
import warnings
warnings.filterwarnings('ignore')

def load_all_results():
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

def analyze_forward_returns_by_deviation(results_df):
    """
    Analyze 1-year forward maximum gains by deviation bin (5% increments)
    """
    valid_data = results_df.dropna().copy()
    valid_data['price_deviation'] = ((valid_data['price'] / valid_data['trend_line_value']) - 1) * 100
    
    # Define deviation bins in 5% increments
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
    
    # Calculate statistics by bin
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
    
    return bin_stats

def normalize_asset_results(asset_bin_stats):
    """
    Apply range normalization (0-100) and z-score normalization to asset results
    """
    normalized_results = {}
    
    for asset_name, bin_stats in asset_bin_stats.items():
        # Get all valid returns for this asset
        returns = [stats['avg_forward_max_gain'] for stats in bin_stats.values() 
                  if not np.isnan(stats['avg_forward_max_gain'])]
        
        if len(returns) < 2:
            print(f"Skipping {asset_name}: insufficient data for normalization")
            continue
        
        asset_min = min(returns)
        asset_max = max(returns)
        asset_mean = np.mean(returns)
        asset_std = np.std(returns)
        
        # Skip if no variation
        if asset_max == asset_min or asset_std == 0:
            print(f"Skipping {asset_name}: no variation in returns")
            continue
        
        normalized_bins = {}
        
        for bin_name, stats in bin_stats.items():
            if not np.isnan(stats['avg_forward_max_gain']) and stats['count'] > 0:
                raw_return = stats['avg_forward_max_gain']
                
                # Range normalization (0-100)
                range_normalized = ((raw_return - asset_min) / (asset_max - asset_min)) * 100
                
                # Z-score normalization
                z_score = (raw_return - asset_mean) / asset_std
                
                normalized_bins[bin_name] = {
                    'raw_return': raw_return,
                    'range_normalized': range_normalized,
                    'z_score': z_score,
                    'count': stats['count']
                }
        
        normalized_results[asset_name] = {
            'bins': normalized_bins,
            'stats': {
                'min': asset_min,
                'max': asset_max,
                'mean': asset_mean,
                'std': asset_std,
                'range': asset_max - asset_min
            }
        }
    
    return normalized_results

def create_summary_table(normalized_results):
    """
    Create summary tables for analysis
    """
    # Collect data for summary
    summary_data = []
    
    bin_order = [">+50%", "+45% to +50%", "+40% to +45%", "+35% to +40%", "+30% to +35%", 
                 "+25% to +30%", "+20% to +25%", "+15% to +20%", "+10% to +15%", "+5% to +10%", 
                 "-5% to +5%", "-10% to -5%", "-15% to -10%", "-20% to -15%", "-25% to -20%", 
                 "-30% to -25%", "-35% to -30%", "-40% to -35%", "-45% to -40%", "-50% to -45%", "<-50%"]
    
    for asset_name, data in normalized_results.items():
        for bin_name in bin_order:
            if bin_name in data['bins']:
                bin_data = data['bins'][bin_name]
                summary_data.append({
                    'asset': asset_name,
                    'deviation_bin': bin_name,
                    'raw_return': bin_data['raw_return'],
                    'range_normalized': bin_data['range_normalized'],
                    'z_score': bin_data['z_score'],
                    'count': bin_data['count']
                })
    
    return pd.DataFrame(summary_data)

def plot_normalized_heatmaps(summary_df):
    """
    Create heatmaps for normalized results
    """
    bin_order = [">+50%", "+45% to +50%", "+40% to +45%", "+35% to +40%", "+30% to +35%", 
                 "+25% to +30%", "+20% to +25%", "+15% to +20%", "+10% to +15%", "+5% to +10%", 
                 "-5% to +5%", "-10% to -5%", "-15% to -10%", "-20% to -15%", "-25% to -20%", 
                 "-30% to -25%", "-35% to -30%", "-40% to -35%", "-45% to -40%", "-50% to -45%", "<-50%"]
    
    # Create pivot tables
    range_pivot = summary_df.pivot(index='asset', columns='deviation_bin', values='range_normalized')
    zscore_pivot = summary_df.pivot(index='asset', columns='deviation_bin', values='z_score')
    
    # Reorder columns to match bin order
    available_bins = [bin_name for bin_name in bin_order if bin_name in range_pivot.columns]
    range_pivot = range_pivot[available_bins]
    zscore_pivot = zscore_pivot[available_bins]
    
    # Create plots
    fig, (ax1, ax2) = plt.subplots(2, 1, figsize=(20, 16))
    
    # Range normalized heatmap
    im1 = ax1.imshow(range_pivot.values, cmap='RdYlGn', aspect='auto', vmin=0, vmax=100)
    ax1.set_xticks(range(len(available_bins)))
    ax1.set_xticklabels(available_bins, rotation=45, ha='right')
    ax1.set_yticks(range(len(range_pivot.index)))
    ax1.set_yticklabels(range_pivot.index)
    ax1.set_title('Range Normalized Returns (0-100 scale)', fontsize=14)
    
    # Add text annotations
    for i in range(len(range_pivot.index)):
        for j in range(len(available_bins)):
            value = range_pivot.iloc[i, j]
            if not np.isnan(value):
                ax1.text(j, i, f'{value:.0f}', ha='center', va='center', 
                        color='white' if value < 50 else 'black', fontsize=8)
    
    plt.colorbar(im1, ax=ax1, label='Normalized Score (0-100)')
    
    # Z-score heatmap
    vmax = max(abs(zscore_pivot.min().min()), abs(zscore_pivot.max().max()))
    im2 = ax2.imshow(zscore_pivot.values, cmap='RdBu_r', aspect='auto', vmin=-vmax, vmax=vmax)
    ax2.set_xticks(range(len(available_bins)))
    ax2.set_xticklabels(available_bins, rotation=45, ha='right')
    ax2.set_yticks(range(len(zscore_pivot.index)))
    ax2.set_yticklabels(zscore_pivot.index)
    ax2.set_title('Z-Score Normalized Returns (standard deviations from mean)', fontsize=14)
    
    # Add text annotations
    for i in range(len(zscore_pivot.index)):
        for j in range(len(available_bins)):
            value = zscore_pivot.iloc[i, j]
            if not np.isnan(value):
                ax2.text(j, i, f'{value:.1f}', ha='center', va='center', 
                        color='white' if abs(value) > 1 else 'black', fontsize=8)
    
    plt.colorbar(im2, ax=ax2, label='Z-Score (std devs from mean)')
    
    plt.tight_layout()
    plt.savefig("/Users/tim/IWLS-OPTIONS/normalized_analysis_heatmaps.png", dpi=300, bbox_inches='tight')
    plt.show()

def calculate_cross_asset_averages(summary_df):
    """
    Calculate average normalized scores across all assets for each deviation bin
    """
    bin_order = [">+50%", "+45% to +50%", "+40% to +45%", "+35% to +40%", "+30% to +35%", 
                 "+25% to +30%", "+20% to +25%", "+15% to +20%", "+10% to +15%", "+5% to +10%", 
                 "-5% to +5%", "-10% to -5%", "-15% to -10%", "-20% to -15%", "-25% to -20%", 
                 "-30% to -25%", "-35% to -30%", "-40% to -35%", "-45% to -40%", "-50% to -45%", "<-50%"]
    
    cross_asset_averages = {}
    
    for bin_name in bin_order:
        bin_data = summary_df[summary_df['deviation_bin'] == bin_name]
        
        if len(bin_data) > 0:
            cross_asset_averages[bin_name] = {
                'avg_raw_return': bin_data['raw_return'].mean(),
                'avg_range_normalized': bin_data['range_normalized'].mean(),
                'avg_z_score': bin_data['z_score'].mean(),
                'asset_count': len(bin_data),
                'total_samples': bin_data['count'].sum()
            }
        else:
            cross_asset_averages[bin_name] = {
                'avg_raw_return': np.nan,
                'avg_range_normalized': np.nan,
                'avg_z_score': np.nan,
                'asset_count': 0,
                'total_samples': 0
            }
    
    return cross_asset_averages

def plot_cross_asset_averages(cross_asset_averages):
    """
    Plot cross-asset average results
    """
    bin_order = [">+50%", "+45% to +50%", "+40% to +45%", "+35% to +40%", "+30% to +35%", 
                 "+25% to +30%", "+20% to +25%", "+15% to +20%", "+10% to +15%", "+5% to +10%", 
                 "-5% to +5%", "-10% to -5%", "-15% to -10%", "-20% to -15%", "-25% to -20%", 
                 "-30% to -25%", "-35% to -30%", "-40% to -35%", "-45% to -40%", "-50% to -45%", "<-50%"]
    
    # Filter to bins with data
    valid_bins = []
    raw_returns = []
    range_normalized = []
    z_scores = []
    asset_counts = []
    
    for bin_name in bin_order:
        if cross_asset_averages[bin_name]['asset_count'] > 0:
            valid_bins.append(bin_name)
            raw_returns.append(cross_asset_averages[bin_name]['avg_raw_return'])
            range_normalized.append(cross_asset_averages[bin_name]['avg_range_normalized'])
            z_scores.append(cross_asset_averages[bin_name]['avg_z_score'])
            asset_counts.append(cross_asset_averages[bin_name]['asset_count'])
    
    if not valid_bins:
        print("No valid bins found for cross-asset analysis")
        return
    
    # Create plots
    fig, ((ax1, ax2), (ax3, ax4)) = plt.subplots(2, 2, figsize=(20, 12))
    
    # Plot 1: Raw Returns
    colors1 = ['darkred' if '+' in bin_name and bin_name != '-5% to +5%' else 
               'darkgreen' if '-' in bin_name and bin_name != '-5% to +5%' else 'gray' 
               for bin_name in valid_bins]
    
    bars1 = ax1.bar(range(len(valid_bins)), raw_returns, color=colors1, alpha=0.7)
    ax1.set_xticks(range(len(valid_bins)))
    ax1.set_xticklabels(valid_bins, rotation=45, ha='right')
    ax1.set_ylabel('Average Raw Return (%)')
    ax1.set_title('Cross-Asset Average: Raw Forward Returns by Deviation Bin')
    ax1.grid(True, alpha=0.3)
    
    # Add value labels
    for i, (bar, value, count) in enumerate(zip(bars1, raw_returns, asset_counts)):
        ax1.text(bar.get_x() + bar.get_width()/2., bar.get_height() + max(raw_returns)*0.01,
                f'{value:.1f}%\n({count} assets)', ha='center', va='bottom', fontsize=9)
    
    # Plot 2: Range Normalized
    bars2 = ax2.bar(range(len(valid_bins)), range_normalized, color=colors1, alpha=0.7)
    ax2.set_xticks(range(len(valid_bins)))
    ax2.set_xticklabels(valid_bins, rotation=45, ha='right')
    ax2.set_ylabel('Average Range Normalized Score (0-100)')
    ax2.set_title('Cross-Asset Average: Range Normalized Scores')
    ax2.grid(True, alpha=0.3)
    
    for i, (bar, value) in enumerate(zip(bars2, range_normalized)):
        ax2.text(bar.get_x() + bar.get_width()/2., bar.get_height() + 2,
                f'{value:.1f}', ha='center', va='bottom', fontsize=10, fontweight='bold')
    
    # Plot 3: Z-Scores
    colors3 = ['green' if z > 0 else 'red' for z in z_scores]
    bars3 = ax3.bar(range(len(valid_bins)), z_scores, color=colors3, alpha=0.7)
    ax3.set_xticks(range(len(valid_bins)))
    ax3.set_xticklabels(valid_bins, rotation=45, ha='right')
    ax3.set_ylabel('Average Z-Score')
    ax3.set_title('Cross-Asset Average: Z-Scores (Std Devs from Mean)')
    ax3.axhline(y=0, color='black', linestyle='--', alpha=0.5)
    ax3.grid(True, alpha=0.3)
    
    for i, (bar, value) in enumerate(zip(bars3, z_scores)):
        ax3.text(bar.get_x() + bar.get_width()/2., 
                bar.get_height() + (0.1 if value >= 0 else -0.2),
                f'{value:.2f}', ha='center', va='bottom' if value >= 0 else 'top', 
                fontsize=10, fontweight='bold')
    
    # Plot 4: Asset Coverage
    bars4 = ax4.bar(range(len(valid_bins)), asset_counts, color='steelblue', alpha=0.7)
    ax4.set_xticks(range(len(valid_bins)))
    ax4.set_xticklabels(valid_bins, rotation=45, ha='right')
    ax4.set_ylabel('Number of Assets with Data')
    ax4.set_title('Asset Coverage by Deviation Bin')
    ax4.grid(True, alpha=0.3)
    
    for i, (bar, count) in enumerate(zip(bars4, asset_counts)):
        ax4.text(bar.get_x() + bar.get_width()/2., bar.get_height() + 0.1,
                f'{count}', ha='center', va='bottom', fontsize=10, fontweight='bold')
    
    plt.tight_layout()
    plt.savefig("/Users/tim/IWLS-OPTIONS/cross_asset_averages.png", dpi=300, bbox_inches='tight')
    plt.show()

def print_cross_asset_summary(cross_asset_averages):
    """
    Print detailed cross-asset summary
    """
    print("\n=== CROSS-ASSET AVERAGE ANALYSIS ===")
    print("=" * 60)
    print(f"{'Deviation Bin':<15} {'Raw Ret':<8} {'Range':<6} {'Z-Score':<7} {'Assets':<7} {'Samples':<8}")
    print("-" * 60)
    
    bin_order = [">+50%", "+45% to +50%", "+40% to +45%", "+35% to +40%", "+30% to +35%", 
                 "+25% to +30%", "+20% to +25%", "+15% to +20%", "+10% to +15%", "+5% to +10%", 
                 "-5% to +5%", "-10% to -5%", "-15% to -10%", "-20% to -15%", "-25% to -20%", 
                 "-30% to -25%", "-35% to -30%", "-40% to -35%", "-45% to -40%", "-50% to -45%", "<-50%"]
    
    for bin_name in bin_order:
        stats = cross_asset_averages[bin_name]
        if stats['asset_count'] > 0:
            print(f"{bin_name:<15} {stats['avg_raw_return']:7.1f}% {stats['avg_range_normalized']:5.1f} "
                  f"{stats['avg_z_score']:7.2f} {stats['asset_count']:6d} {stats['total_samples']:7d}")
    
    # Key insights
    print("\n=== KEY INSIGHTS ===")
    
    # Find best and worst performing bins
    valid_stats = {k: v for k, v in cross_asset_averages.items() if v['asset_count'] > 0}
    
    if valid_stats:
        best_raw = max(valid_stats.items(), key=lambda x: x[1]['avg_raw_return'])
        worst_raw = min(valid_stats.items(), key=lambda x: x[1]['avg_raw_return'])
        best_zscore = max(valid_stats.items(), key=lambda x: x[1]['avg_z_score'])
        worst_zscore = min(valid_stats.items(), key=lambda x: x[1]['avg_z_score'])
        
        print(f"Best raw performance: {best_raw[0]} ({best_raw[1]['avg_raw_return']:.1f}%)")
        print(f"Worst raw performance: {worst_raw[0]} ({worst_raw[1]['avg_raw_return']:.1f}%)")
        print(f"Highest relative performance: {best_zscore[0]} (z-score: {best_zscore[1]['avg_z_score']:.2f})")
        print(f"Lowest relative performance: {worst_zscore[0]} (z-score: {worst_zscore[1]['avg_z_score']:.2f})")
        
        performance_spread = best_raw[1]['avg_raw_return'] - worst_raw[1]['avg_raw_return']
        print(f"Performance spread: {performance_spread:.1f}%")
    """
    Identify assets with the strongest deviation signals
    """
    print("\n=== BEST OPPORTUNITIES ANALYSIS ===")
    print("=" * 50)
    
    opportunities = []
    
    for asset in summary_df['asset'].unique():
        asset_data = summary_df[summary_df['asset'] == asset]
        
        # Find undervalued bins (negative deviations)
        undervalued = asset_data[asset_data['deviation_bin'].str.contains('-') & 
                                ~asset_data['deviation_bin'].str.contains('\+')]
        # Find overvalued bins (positive deviations)
        overvalued = asset_data[asset_data['deviation_bin'].str.contains('\+') & 
                               ~asset_data['deviation_bin'].str.contains('-5% to \+5%')]
        
        if len(undervalued) > 0 and len(overvalued) > 0:
            # Get highest z-scores for undervalued and lowest for overvalued
            best_undervalued = undervalued.loc[undervalued['z_score'].idxmax()]
            worst_overvalued = overvalued.loc[overvalued['z_score'].idxmin()]
            
            z_score_spread = best_undervalued['z_score'] - worst_overvalued['z_score']
            range_spread = best_undervalued['range_normalized'] - worst_overvalued['range_normalized']
            
            opportunities.append({
                'asset': asset,
                'z_score_spread': z_score_spread,
                'range_spread': range_spread,
                'best_undervalued_bin': best_undervalued['deviation_bin'],
                'best_undervalued_zscore': best_undervalued['z_score'],
                'worst_overvalued_bin': worst_overvalued['deviation_bin'],
                'worst_overvalued_zscore': worst_overvalued['z_score']
            })
    
    opportunities_df = pd.DataFrame(opportunities)
    opportunities_df = opportunities_df.sort_values('z_score_spread', ascending=False)
    
    print("Top assets by Z-Score spread (undervalued vs overvalued):")
    print("-" * 50)
    for _, row in opportunities_df.head(10).iterrows():
        print(f"{row['asset']:>6}: {row['z_score_spread']:5.2f} spread "
              f"(Best: {row['best_undervalued_bin']} = {row['best_undervalued_zscore']:4.1f}, "
              f"Worst: {row['worst_overvalued_bin']} = {row['worst_overvalued_zscore']:4.1f})")
    
def find_best_opportunities(summary_df):
    """
    Identify assets with the strongest deviation signals
    """
    print("\n=== BEST OPPORTUNITIES ANALYSIS ===")
    print("=" * 50)
    
    opportunities = []
    
    for asset in summary_df['asset'].unique():
        asset_data = summary_df[summary_df['asset'] == asset]
        
        # Find undervalued bins (negative deviations)
        undervalued = asset_data[asset_data['deviation_bin'].str.contains('-') & 
                                ~asset_data['deviation_bin'].str.contains('\+')]
        # Find overvalued bins (positive deviations)
        overvalued = asset_data[asset_data['deviation_bin'].str.contains('\+') & 
                               ~asset_data['deviation_bin'].str.contains('-5% to \+5%')]
        
        if len(undervalued) > 0 and len(overvalued) > 0:
            # Get highest z-scores for undervalued and lowest for overvalued
            best_undervalued = undervalued.loc[undervalued['z_score'].idxmax()]
            worst_overvalued = overvalued.loc[overvalued['z_score'].idxmin()]
            
            z_score_spread = best_undervalued['z_score'] - worst_overvalued['z_score']
            range_spread = best_undervalued['range_normalized'] - worst_overvalued['range_normalized']
            
            opportunities.append({
                'asset': asset,
                'z_score_spread': z_score_spread,
                'range_spread': range_spread,
                'best_undervalued_bin': best_undervalued['deviation_bin'],
                'best_undervalued_zscore': best_undervalued['z_score'],
                'worst_overvalued_bin': worst_overvalued['deviation_bin'],
                'worst_overvalued_zscore': worst_overvalued['z_score']
            })
    
    opportunities_df = pd.DataFrame(opportunities)
    opportunities_df = opportunities_df.sort_values('z_score_spread', ascending=False)
    
    print("Top assets by Z-Score spread (undervalued vs overvalued):")
    print("-" * 50)
    for _, row in opportunities_df.head(10).iterrows():
        print(f"{row['asset']:>6}: {row['z_score_spread']:5.2f} spread "
              f"(Best: {row['best_undervalued_bin']} = {row['best_undervalued_zscore']:4.1f}, "
              f"Worst: {row['worst_overvalued_bin']} = {row['worst_overvalued_zscore']:4.1f})")
    
    return opportunities_df

def main():
    print("IWLS Normalized Analysis - Loading Results from ASSETS_RESULTS")
    print("=" * 70)
    
    # Load all results
    all_results = load_all_results()
    if not all_results:
        return
    
    print(f"\nLoaded {len(all_results)} assets")
    
    # Analyze forward returns for each asset
    print("\nAnalyzing forward returns by deviation bins...")
    asset_bin_stats = {}
    
    for asset_name, results_df in all_results.items():
        print(f"  Processing {asset_name}...")
        bin_stats = analyze_forward_returns_by_deviation(results_df)
        asset_bin_stats[asset_name] = bin_stats
    
    # Normalize results
    print("\nNormalizing results...")
    normalized_results = normalize_asset_results(asset_bin_stats)
    
    print(f"Successfully normalized {len(normalized_results)} assets")
    
    # Create summary table
    summary_df = create_summary_table(normalized_results)
    
    # Calculate cross-asset averages
    print("\nCalculating cross-asset averages...")
    cross_asset_averages = calculate_cross_asset_averages(summary_df)
    
    # Save summary
    summary_df.to_csv("/Users/tim/IWLS-OPTIONS/normalized_summary.csv", index=False)
    print(f"\nSummary saved to: normalized_summary.csv")
    
    # Save cross-asset averages
    cross_asset_df = pd.DataFrame([
        {
            'deviation_bin': bin_name,
            'avg_raw_return': stats['avg_raw_return'],
            'avg_range_normalized': stats['avg_range_normalized'],
            'avg_z_score': stats['avg_z_score'],
            'asset_count': stats['asset_count'],
            'total_samples': stats['total_samples']
        }
        for bin_name, stats in cross_asset_averages.items()
        if stats['asset_count'] > 0
    ])
    cross_asset_df.to_csv("/Users/tim/IWLS-OPTIONS/cross_asset_averages.csv", index=False)
    
    # Create visualizations
    print("\nCreating heatmap visualizations...")
    plot_normalized_heatmaps(summary_df)
    
    print("\nCreating cross-asset average plots...")
    plot_cross_asset_averages(cross_asset_averages)
    
    # Print cross-asset summary
    print_cross_asset_summary(cross_asset_averages)
    
    # Find best opportunities
    opportunities_df = find_best_opportunities(summary_df)
    opportunities_df.to_csv("/Users/tim/IWLS-OPTIONS/best_opportunities.csv", index=False)
    print(f"\nOpportunities analysis saved to: best_opportunities.csv")
    print(f"\nOpportunities analysis saved to: best_opportunities.csv")
    
    print(f"\nAnalysis complete. Files saved to /Users/tim/IWLS-OPTIONS/")

if __name__ == "__main__":
    main()