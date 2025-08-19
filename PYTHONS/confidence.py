import pandas as pd
import numpy as np
import matplotlib.pyplot as plt
import os
import glob
from scipy import stats
from sklearn.linear_model import LinearRegression
from sklearn.metrics import r2_score
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
                'std_forward_max_gain': bin_data['forward_max_gain'].std(),
                'median_forward_max_gain': bin_data['forward_max_gain'].median(),
                'min_forward_max_gain': bin_data['forward_max_gain'].min(),
                'max_forward_max_gain': bin_data['forward_max_gain'].max()
            }
        else:
            bin_stats[bin_name] = {
                'count': 0,
                'avg_forward_max_gain': np.nan,
                'std_forward_max_gain': np.nan,
                'median_forward_max_gain': np.nan,
                'min_forward_max_gain': np.nan,
                'max_forward_max_gain': np.nan
            }
    
    return bin_stats

def assign_bin_scores(bin_order):
    """
    Assign numerical scores to bins for correlation analysis
    Most negative deviation = highest score (expect highest gains)
    """
    scores = {}
    for i, bin_name in enumerate(reversed(bin_order)):  # Reverse so most negative gets highest score
        scores[bin_name] = i + 1
    return scores

def calculate_correlation_metrics(bin_stats, asset_name):
    """
    Calculate various correlation metrics between deviation bins and forward gains
    """
    # Prepare data
    bin_order = [">+50%", "+45% to +50%", "+40% to +45%", "+35% to +40%", "+30% to +35%", 
                 "+25% to +30%", "+20% to +25%", "+15% to +20%", "+10% to +15%", "+5% to +10%", 
                 "-5% to +5%", "-10% to -5%", "-15% to -10%", "-20% to -15%", "-25% to -20%", 
                 "-30% to -25%", "-35% to -30%", "-40% to -35%", "-45% to -40%", "-50% to -45%", "<-50%"]
    
    bin_scores = assign_bin_scores(bin_order)
    
    # Extract valid bins (with data)
    valid_bins = []
    gains = []
    scores = []
    counts = []
    stds = []
    
    for bin_name in bin_order:
        if bin_stats[bin_name]['count'] > 0 and not np.isnan(bin_stats[bin_name]['avg_forward_max_gain']):
            valid_bins.append(bin_name)
            gains.append(bin_stats[bin_name]['avg_forward_max_gain'])
            scores.append(bin_scores[bin_name])
            counts.append(bin_stats[bin_name]['count'])
            stds.append(bin_stats[bin_name]['std_forward_max_gain'] if not np.isnan(bin_stats[bin_name]['std_forward_max_gain']) else 0)
    
    if len(gains) < 3:  # Need at least 3 bins for meaningful correlation
        return {
            'asset': asset_name,
            'spearman_correlation': np.nan,
            'spearman_pvalue': np.nan,
            'pearson_correlation': np.nan,
            'pearson_pvalue': np.nan,
            'r_squared': np.nan,
            'monotonicity_score': np.nan,
            'signal_to_noise_ratio': np.nan,
            'gain_range': np.nan,
            'num_bins': len(valid_bins),
            'total_samples': sum(counts),
            'reliability_score': np.nan,
            'valid_bins': valid_bins,
            'gains': gains,
            'scores': scores,
            'counts': counts,
            'stds': stds
        }
    
    # 1. Spearman Rank Correlation (best for ordinal data)
    spearman_corr, spearman_p = stats.spearmanr(scores, gains)
    
    # 2. Pearson Correlation
    pearson_corr, pearson_p = stats.pearsonr(scores, gains)
    
    # 3. R-squared from linear regression
    X = np.array(scores).reshape(-1, 1)
    y = np.array(gains)
    reg = LinearRegression().fit(X, y)
    y_pred = reg.predict(X)
    r_squared = r2_score(y, y_pred)
    
    # 4. Monotonicity Score (how often does the expected pattern hold?)
    monotonic_count = 0
    total_transitions = 0
    for i in range(len(gains) - 1):
        # Expected: higher bin score (more negative deviation) should have higher gain
        if scores[i] < scores[i + 1]:  # Moving to more negative deviation
            if gains[i] < gains[i + 1]:  # Gain increased as expected
                monotonic_count += 1
        elif scores[i] > scores[i + 1]:  # Moving to more positive deviation
            if gains[i] > gains[i + 1]:  # Gain decreased as expected
                monotonic_count += 1
        total_transitions += 1
    
    monotonicity_score = monotonic_count / total_transitions if total_transitions > 0 else 0
    
    # 5. Signal-to-Noise Ratio
    gain_range = max(gains) - min(gains)
    avg_std = np.mean(stds) if stds else 0
    signal_to_noise = gain_range / avg_std if avg_std > 0 else 0
    
    # 6. Composite Reliability Score
    # Combine metrics (0-100 scale)
    reliability_components = []
    
    # Spearman correlation component (0-50 points)
    if not np.isnan(spearman_corr):
        spearman_component = abs(spearman_corr) * 50
        reliability_components.append(spearman_component)
    
    # Monotonicity component (0-25 points)
    monotonicity_component = monotonicity_score * 25
    reliability_components.append(monotonicity_component)
    
    # Signal-to-noise component (0-25 points, capped at reasonable level)
    snr_component = min(signal_to_noise / 5, 1) * 25  # Normalize assuming SNR of 5 is excellent
    reliability_components.append(snr_component)
    
    reliability_score = sum(reliability_components)
    
    return {
        'asset': asset_name,
        'spearman_correlation': spearman_corr,
        'spearman_pvalue': spearman_p,
        'pearson_correlation': pearson_corr,
        'pearson_pvalue': pearson_p,
        'r_squared': r_squared,
        'monotonicity_score': monotonicity_score,
        'signal_to_noise_ratio': signal_to_noise,
        'gain_range': gain_range,
        'num_bins': len(valid_bins),
        'total_samples': sum(counts),
        'reliability_score': reliability_score,
        'valid_bins': valid_bins,
        'gains': gains,
        'scores': scores,
        'counts': counts,
        'stds': stds
    }

def create_correlation_visualization(correlation_results, output_dir):
    """
    Create comprehensive visualization of correlation analysis
    """
    # Sort by reliability score
    sorted_results = sorted([r for r in correlation_results if not np.isnan(r['reliability_score'])], 
                           key=lambda x: x['reliability_score'], reverse=True)
    
    if len(sorted_results) < 2:
        print("Insufficient data for visualization")
        return
    
    # Create figure with multiple subplots
    fig, ((ax1, ax2), (ax3, ax4)) = plt.subplots(2, 2, figsize=(20, 16))
    
    # Plot 1: Reliability Score Ranking
    assets = [r['asset'] for r in sorted_results]
    reliability_scores = [r['reliability_score'] for r in sorted_results]
    
    colors = ['darkgreen' if score >= 70 else 'green' if score >= 50 else 'orange' if score >= 30 else 'red' 
              for score in reliability_scores]
    
    bars1 = ax1.barh(range(len(assets)), reliability_scores, color=colors, alpha=0.7)
    ax1.set_yticks(range(len(assets)))
    ax1.set_yticklabels(assets)
    ax1.set_xlabel('Reliability Score (0-100)')
    ax1.set_title('Asset Reliability Ranking for IWLS Deviation Strategy', fontweight='bold')
    ax1.grid(True, alpha=0.3)
    
    # Add score labels
    for i, (bar, score) in enumerate(zip(bars1, reliability_scores)):
        ax1.text(bar.get_width() + 1, bar.get_y() + bar.get_height()/2,
                f'{score:.1f}', va='center', fontweight='bold')
    
    # Plot 2: Spearman Correlation vs Monotonicity
    spearman_corrs = [abs(r['spearman_correlation']) if not np.isnan(r['spearman_correlation']) else 0 
                      for r in sorted_results]
    monotonicity_scores = [r['monotonicity_score'] for r in sorted_results]
    
    scatter = ax2.scatter(spearman_corrs, monotonicity_scores, 
                         c=reliability_scores, cmap='RdYlGn', 
                         s=100, alpha=0.7, edgecolors='black')
    
    for i, asset in enumerate(assets):
        ax2.annotate(asset, (spearman_corrs[i], monotonicity_scores[i]), 
                    xytext=(5, 5), textcoords='offset points', fontsize=9)
    
    ax2.set_xlabel('|Spearman Correlation|')
    ax2.set_ylabel('Monotonicity Score')
    ax2.set_title('Correlation vs Monotonicity Analysis', fontweight='bold')
    ax2.grid(True, alpha=0.3)
    plt.colorbar(scatter, ax=ax2, label='Reliability Score')
    
    # Plot 3: Signal-to-Noise Ratio
    snr_ratios = [r['signal_to_noise_ratio'] for r in sorted_results]
    
    bars3 = ax3.bar(range(len(assets)), snr_ratios, color=colors, alpha=0.7)
    ax3.set_xticks(range(len(assets)))
    ax3.set_xticklabels(assets, rotation=45, ha='right')
    ax3.set_ylabel('Signal-to-Noise Ratio')
    ax3.set_title('Signal-to-Noise Ratio by Asset', fontweight='bold')
    ax3.grid(True, alpha=0.3)
    
    # Plot 4: Sample Size vs Correlation Quality
    total_samples = [r['total_samples'] for r in sorted_results]
    
    scatter2 = ax4.scatter(total_samples, spearman_corrs, 
                          c=reliability_scores, cmap='RdYlGn',
                          s=100, alpha=0.7, edgecolors='black')
    
    for i, asset in enumerate(assets):
        ax4.annotate(asset, (total_samples[i], spearman_corrs[i]), 
                    xytext=(5, 5), textcoords='offset points', fontsize=9)
    
    ax4.set_xlabel('Total Sample Size')
    ax4.set_ylabel('|Spearman Correlation|')
    ax4.set_title('Sample Size vs Correlation Strength', fontweight='bold')
    ax4.grid(True, alpha=0.3)
    plt.colorbar(scatter2, ax=ax4, label='Reliability Score')
    
    plt.tight_layout()
    plt.savefig(f"{output_dir}/correlation_analysis_overview.png", dpi=300, bbox_inches='tight')
    plt.close()

def create_individual_asset_plots(correlation_results, output_dir):
    """
    Create individual plots for top and bottom performers
    """
    # Sort by reliability score
    sorted_results = sorted([r for r in correlation_results if not np.isnan(r['reliability_score'])], 
                           key=lambda x: x['reliability_score'], reverse=True)
    
    # Plot top 5 and bottom 5 performers
    top_5 = sorted_results[:5]
    bottom_5 = sorted_results[-5:] if len(sorted_results) >= 5 else []
    
    def plot_asset_bins(assets_to_plot, title_suffix, filename):
        if not assets_to_plot:
            return
            
        fig, axes = plt.subplots(len(assets_to_plot), 1, figsize=(16, 4 * len(assets_to_plot)))
        if len(assets_to_plot) == 1:
            axes = [axes]
        
        for i, result in enumerate(assets_to_plot):
            asset = result['asset']
            gains = result['gains']
            valid_bins = result['valid_bins']
            counts = result['counts']
            
            # Create color gradient (red for positive deviations, green for negative)
            colors = []
            for bin_name in valid_bins:
                if "+" in bin_name and bin_name != "-5% to +5%":
                    colors.append('lightcoral')
                elif "-" in bin_name and bin_name != "-5% to +5%":
                    colors.append('lightgreen')
                else:
                    colors.append('gray')
            
            bars = axes[i].bar(range(len(valid_bins)), gains, color=colors, alpha=0.7, edgecolor='black')
            axes[i].set_xticks(range(len(valid_bins)))
            axes[i].set_xticklabels(valid_bins, rotation=45, ha='right')
            axes[i].set_ylabel('Avg Forward Max Gain (%)')
            axes[i].set_title(f'{asset}: Reliability Score = {result["reliability_score"]:.1f}, '
                             f'Spearman = {result["spearman_correlation"]:.3f}', fontweight='bold')
            axes[i].grid(True, alpha=0.3)
            
            # Add value and count labels
            for j, (bar, gain, count) in enumerate(zip(bars, gains, counts)):
                axes[i].text(bar.get_x() + bar.get_width()/2., bar.get_height() + max(gains)*0.01,
                            f'{gain:.1f}%\n(n={count})', ha='center', va='bottom', fontsize=9)
        
        plt.tight_layout()
        plt.savefig(f"{output_dir}/{filename}", dpi=300, bbox_inches='tight')
        plt.close()
    
    plot_asset_bins(top_5, "Most Reliable", "top_5_reliable_assets.png")
    plot_asset_bins(bottom_5, "Least Reliable", "bottom_5_reliable_assets.png")

def save_correlation_results(correlation_results, output_dir):
    """
    Save detailed correlation results to CSV files
    """
    # Main summary file
    summary_data = []
    for result in correlation_results:
        summary_data.append({
            'asset': result['asset'],
            'reliability_score': result['reliability_score'],
            'spearman_correlation': result['spearman_correlation'],
            'spearman_pvalue': result['spearman_pvalue'],
            'pearson_correlation': result['pearson_correlation'],
            'pearson_pvalue': result['pearson_pvalue'],
            'r_squared': result['r_squared'],
            'monotonicity_score': result['monotonicity_score'],
            'signal_to_noise_ratio': result['signal_to_noise_ratio'],
            'gain_range': result['gain_range'],
            'num_bins': result['num_bins'],
            'total_samples': result['total_samples']
        })
    
    summary_df = pd.DataFrame(summary_data)
    summary_df = summary_df.sort_values('reliability_score', ascending=False)
    summary_df.to_csv(f"{output_dir}/correlation_analysis_summary.csv", index=False)
    
    # Detailed bin data for each asset
    detailed_data = []
    for result in correlation_results:
        asset = result['asset']
        for i, bin_name in enumerate(result['valid_bins']):
            detailed_data.append({
                'asset': asset,
                'deviation_bin': bin_name,
                'bin_score': result['scores'][i],
                'avg_forward_gain': result['gains'][i],
                'sample_count': result['counts'][i],
                'std_forward_gain': result['stds'][i],
                'reliability_score': result['reliability_score']
            })
    
    detailed_df = pd.DataFrame(detailed_data)
    detailed_df.to_csv(f"{output_dir}/detailed_bin_analysis.csv", index=False)
    
    return summary_df

def print_correlation_analysis_summary(correlation_results):
    """
    Print comprehensive summary of correlation analysis
    """
    print("\n" + "="*80)
    print("IWLS DEVIATION CORRELATION ANALYSIS SUMMARY")
    print("="*80)
    
    # Sort by reliability score
    sorted_results = sorted([r for r in correlation_results if not np.isnan(r['reliability_score'])], 
                           key=lambda x: x['reliability_score'], reverse=True)
    
    print(f"\nAnalyzed {len(correlation_results)} assets")
    print(f"Assets with sufficient data: {len(sorted_results)}")
    
    if len(sorted_results) == 0:
        print("No assets had sufficient data for correlation analysis!")
        return
    
    print(f"\nTOP 10 MOST RELIABLE ASSETS (Best Deviation-Performance Correlation):")
    print("-" * 80)
    print(f"{'Asset':<8} {'Reliability':<11} {'Spearman':<9} {'Monotonic':<10} {'S/N Ratio':<9} {'Samples':<8}")
    print("-" * 80)
    
    for i, result in enumerate(sorted_results[:10]):
        print(f"{result['asset']:<8} {result['reliability_score']:>10.1f} "
              f"{result['spearman_correlation']:>8.3f} {result['monotonicity_score']:>9.3f} "
              f"{result['signal_to_noise_ratio']:>8.1f} {result['total_samples']:>7d}")
    
    print(f"\nBOTTOM 5 LEAST RELIABLE ASSETS:")
    print("-" * 80)
    for result in sorted_results[-5:]:
        print(f"{result['asset']:<8} {result['reliability_score']:>10.1f} "
              f"{result['spearman_correlation']:>8.3f} {result['monotonicity_score']:>9.3f} "
              f"{result['signal_to_noise_ratio']:>8.1f} {result['total_samples']:>7d}")
    
    # Statistical insights
    reliability_scores = [r['reliability_score'] for r in sorted_results]
    spearman_corrs = [abs(r['spearman_correlation']) for r in sorted_results if not np.isnan(r['spearman_correlation'])]
    
    print(f"\nSTATISTICAL INSIGHTS:")
    print("-" * 40)
    print(f"Average reliability score: {np.mean(reliability_scores):.1f}")
    print(f"Median reliability score: {np.median(reliability_scores):.1f}")
    print(f"High reliability assets (>70): {sum(1 for s in reliability_scores if s >= 70)}")
    print(f"Medium reliability assets (50-70): {sum(1 for s in reliability_scores if 50 <= s < 70)}")
    print(f"Low reliability assets (<50): {sum(1 for s in reliability_scores if s < 50)}")
    
    if spearman_corrs:
        print(f"Average |Spearman correlation|: {np.mean(spearman_corrs):.3f}")
        print(f"Strong correlations (>0.7): {sum(1 for c in spearman_corrs if c >= 0.7)}")
    
    print(f"\nTRADING STRATEGY RECOMMENDATIONS:")
    print("-" * 40)
    high_reliability = [r for r in sorted_results if r['reliability_score'] >= 70]
    medium_reliability = [r for r in sorted_results if 50 <= r['reliability_score'] < 70]
    
    if high_reliability:
        print(f"HIGH CONFIDENCE ASSETS ({len(high_reliability)} assets):")
        print("  - Strong deviation-performance correlation")
        print("  - Recommended for primary IWLS strategy")
        print(f"  - Assets: {', '.join([r['asset'] for r in high_reliability[:10]])}")
    
    if medium_reliability:
        print(f"\nMEDIUM CONFIDENCE ASSETS ({len(medium_reliability)} assets):")
        print("  - Moderate correlation, use with caution")
        print("  - Consider smaller position sizes")
        print(f"  - Assets: {', '.join([r['asset'] for r in medium_reliability[:10]])}")
    
    low_reliability = [r for r in sorted_results if r['reliability_score'] < 50]
    if low_reliability:
        print(f"\nLOW CONFIDENCE ASSETS ({len(low_reliability)} assets):")
        print("  - Weak/unreliable correlation")
        print("  - NOT recommended for IWLS deviation strategy")
        print(f"  - Assets: {', '.join([r['asset'] for r in low_reliability[:10]])}")

def main():
    print("IWLS Deviation Bin Correlation Analysis")
    print("="*60)
    print("Analyzing correlation between deviation bins and future max gains")
    print("Identifying which assets have reliable deviation-performance relationships")
    
    # Create output directory
    output_dir = "/Users/tim/IWLS-OPTIONS/CORRELATION_ANALYSIS"
    os.makedirs(output_dir, exist_ok=True)
    print(f"\nOutput directory: {output_dir}")
    
    # Load all IWLS results
    all_results = load_all_iwls_results()
    if not all_results:
        return
    
    print(f"\nLoaded {len(all_results)} assets")
    
    # Analyze correlation for each asset
    print("\nAnalyzing deviation-performance correlations...")
    correlation_results = []
    
    for asset_name, results_df in all_results.items():
        print(f"  Processing {asset_name}...")
        
        # Get bin statistics
        bin_stats = analyze_forward_returns_by_deviation(results_df)
        
        # Calculate correlation metrics
        correlation_metrics = calculate_correlation_metrics(bin_stats, asset_name)
        correlation_results.append(correlation_metrics)
    
    # Save results
    print(f"\nSaving correlation analysis results...")
    summary_df = save_correlation_results(correlation_results, output_dir)
    print(f"  Saved correlation_analysis_summary.csv")
    print(f"  Saved detailed_bin_analysis.csv")
    
    # Create visualizations
    print(f"\nCreating visualizations...")
    create_correlation_visualization(correlation_results, output_dir)
    print(f"  Saved correlation_analysis_overview.png")
    
    create_individual_asset_plots(correlation_results, output_dir)
    print(f"  Saved top_5_reliable_assets.png")
    print(f"  Saved bottom_5_reliable_assets.png")
    
    # Print summary
    print_correlation_analysis_summary(correlation_results)
    
    print(f"\n" + "="*80)
    print("CORRELATION ANALYSIS COMPLETE")
    print("="*80)
    print("Files saved to /Users/tim/IWLS-OPTIONS/CORRELATION_ANALYSIS/:")
    print("  - correlation_analysis_summary.csv (main results)")
    print("  - detailed_bin_analysis.csv (bin-level data)")
    print("  - correlation_analysis_overview.png (4-panel overview)")
    print("  - top_5_reliable_assets.png (best performers)")
    print("  - bottom_5_reliable_assets.png (worst performers)")
    
    # Final recommendations
    sorted_results = sorted([r for r in correlation_results if not np.isnan(r['reliability_score'])], 
                           key=lambda x: x['reliability_score'], reverse=True)
    
    if sorted_results:
        best_asset = sorted_results[0]
        worst_asset = sorted_results[-1]
        
        print(f"\n" + "="*80)
        print("KEY FINDINGS")
        print("="*80)
        print(f"MOST RELIABLE ASSET: {best_asset['asset']}")
        print(f"  Reliability Score: {best_asset['reliability_score']:.1f}/100")
        print(f"  Spearman Correlation: {best_asset['spearman_correlation']:.3f}")
        print(f"  Interpretation: Strong deviation-performance relationship")
        
        print(f"\nLEAST RELIABLE ASSET: {worst_asset['asset']}")
        print(f"  Reliability Score: {worst_asset['reliability_score']:.1f}/100")
        print(f"  Spearman Correlation: {worst_asset['spearman_correlation']:.3f}")
        print(f"  Interpretation: Weak/unreliable deviation-performance relationship")
        
        high_confidence_count = sum(1 for r in sorted_results if r['reliability_score'] >= 70)
        medium_confidence_count = sum(1 for r in sorted_results if 50 <= r['reliability_score'] < 70)
        low_confidence_count = sum(1 for r in sorted_results if r['reliability_score'] < 50)
        
        print(f"\nSTRATEGY IMPACT:")
        print(f"  High Confidence Assets: {high_confidence_count}/{len(sorted_results)} ({high_confidence_count/len(sorted_results)*100:.1f}%)")
        print(f"  Medium Confidence Assets: {medium_confidence_count}/{len(sorted_results)} ({medium_confidence_count/len(sorted_results)*100:.1f}%)")
        print(f"  Low Confidence Assets: {low_confidence_count}/{len(sorted_results)} ({low_confidence_count/len(sorted_results)*100:.1f}%)")
        
        if high_confidence_count > 0:
            print(f"\n  RECOMMENDATION: Focus IWLS strategy on {high_confidence_count} high-confidence assets")
            print(f"  These assets show strong, reliable deviation-performance relationships")
        else:
            print(f"\n  WARNING: No assets show high confidence correlation")
            print(f"  Consider revising IWLS methodology or parameters")

if __name__ == "__main__":
    main()