import pandas as pd
import numpy as np
from scipy import stats

def load_correlation_results():
    """Load the correlation analysis results"""
    df = pd.read_csv("/Users/tim/IWLS-OPTIONS/CORRELATION_ANALYSIS/correlation_analysis_summary.csv")
    return df

def calculate_multi_method_rankings(df):
    """
    Calculate rankings using multiple methods and create composite scores
    """
    # Create a copy for ranking calculations
    ranking_df = df.copy()
    
    # 1. Rank by each individual metric (lower rank = better)
    # Note: For correlation, we want strong negative correlations, so use absolute values
    ranking_df['spearman_rank'] = (-ranking_df['spearman_correlation'].abs()).rank()
    ranking_df['pearson_rank'] = (-ranking_df['pearson_correlation'].abs()).rank()
    ranking_df['r_squared_rank'] = (-ranking_df['r_squared']).rank()
    ranking_df['monotonicity_rank'] = (-ranking_df['monotonicity_score']).rank()
    ranking_df['snr_rank'] = (-ranking_df['signal_to_noise_ratio']).rank()
    ranking_df['reliability_rank'] = (-ranking_df['reliability_score']).rank()
    
    # 2. Calculate percentile scores (0-100, higher = better)
    ranking_df['spearman_percentile'] = stats.rankdata(ranking_df['spearman_correlation'].abs()) / len(ranking_df) * 100
    ranking_df['pearson_percentile'] = stats.rankdata(ranking_df['pearson_correlation'].abs()) / len(ranking_df) * 100
    ranking_df['r_squared_percentile'] = stats.rankdata(ranking_df['r_squared']) / len(ranking_df) * 100
    ranking_df['monotonicity_percentile'] = stats.rankdata(ranking_df['monotonicity_score']) / len(ranking_df) * 100
    ranking_df['snr_percentile'] = stats.rankdata(ranking_df['signal_to_noise_ratio']) / len(ranking_df) * 100
    ranking_df['reliability_percentile'] = stats.rankdata(ranking_df['reliability_score']) / len(ranking_df) * 100
    
    # 3. Create weighted composite scores
    
    # Equal Weight Composite (all metrics equal)
    ranking_df['equal_weight_score'] = (
        ranking_df['spearman_percentile'] * 0.2 +
        ranking_df['r_squared_percentile'] * 0.2 +
        ranking_df['monotonicity_percentile'] * 0.2 +
        ranking_df['snr_percentile'] * 0.2 +
        ranking_df['reliability_percentile'] * 0.2
    )
    
    # Correlation Focused (emphasize correlation strength)
    ranking_df['correlation_focused_score'] = (
        ranking_df['spearman_percentile'] * 0.35 +
        ranking_df['pearson_percentile'] * 0.25 +
        ranking_df['r_squared_percentile'] * 0.25 +
        ranking_df['monotonicity_percentile'] * 0.1 +
        ranking_df['snr_percentile'] * 0.05
    )
    
    # Practical Trading Focused (emphasize reliability and monotonicity)
    ranking_df['trading_focused_score'] = (
        ranking_df['reliability_percentile'] * 0.3 +
        ranking_df['monotonicity_percentile'] * 0.3 +
        ranking_df['spearman_percentile'] * 0.25 +
        ranking_df['snr_percentile'] * 0.15
    )
    
    # Statistical Significance Focused (emphasize p-values and R²)
    # First convert p-values to significance scores (lower p-value = higher score)
    ranking_df['spearman_significance'] = stats.rankdata(-ranking_df['spearman_pvalue']) / len(ranking_df) * 100
    ranking_df['pearson_significance'] = stats.rankdata(-ranking_df['pearson_pvalue']) / len(ranking_df) * 100
    
    ranking_df['significance_focused_score'] = (
        ranking_df['spearman_significance'] * 0.3 +
        ranking_df['pearson_significance'] * 0.2 +
        ranking_df['r_squared_percentile'] * 0.3 +
        ranking_df['spearman_percentile'] * 0.2
    )
    
    # 4. Calculate final rankings for each composite method
    ranking_df['equal_weight_rank'] = (-ranking_df['equal_weight_score']).rank()
    ranking_df['correlation_focused_rank'] = (-ranking_df['correlation_focused_score']).rank()
    ranking_df['trading_focused_rank'] = (-ranking_df['trading_focused_score']).rank()
    ranking_df['significance_focused_rank'] = (-ranking_df['significance_focused_score']).rank()
    
    # 5. Meta-composite: Average of all ranking methods
    ranking_df['average_rank'] = (
        ranking_df['equal_weight_rank'] +
        ranking_df['correlation_focused_rank'] +
        ranking_df['trading_focused_rank'] +
        ranking_df['significance_focused_rank']
    ) / 4
    
    ranking_df['final_meta_rank'] = ranking_df['average_rank'].rank()
    
    return ranking_df

def create_ranking_consistency_analysis(ranking_df):
    """
    Analyze how consistent rankings are across different methods
    """
    ranking_methods = [
        'equal_weight_rank', 'correlation_focused_rank', 
        'trading_focused_rank', 'significance_focused_rank'
    ]
    
    consistency_analysis = []
    
    for _, row in ranking_df.iterrows():
        asset = row['asset']
        ranks = [row[method] for method in ranking_methods]
        
        consistency_analysis.append({
            'asset': asset,
            'mean_rank': np.mean(ranks),
            'std_rank': np.std(ranks),
            'min_rank': np.min(ranks),
            'max_rank': np.max(ranks),
            'rank_range': np.max(ranks) - np.min(ranks),
            'consistency_score': 100 - (np.std(ranks) / np.mean(ranks) * 100)  # Lower std relative to mean = more consistent
        })
    
    consistency_df = pd.DataFrame(consistency_analysis)
    return consistency_df

def identify_clear_winners_losers(ranking_df, consistency_df):
    """
    Identify assets that consistently rank high or low across all methods
    """
    # Merge dataframes
    analysis_df = ranking_df.merge(consistency_df[['asset', 'consistency_score', 'rank_range']], on='asset')
    
    # Define thresholds
    n_assets = len(analysis_df)
    top_threshold = n_assets * 0.2  # Top 20%
    bottom_threshold = n_assets * 0.8  # Bottom 20%
    
    # Clear winners: consistently in top 20% with low rank variance
    clear_winners = analysis_df[
        (analysis_df['final_meta_rank'] <= top_threshold) & 
        (analysis_df['rank_range'] <= 5)  # Consistent across methods
    ].sort_values('final_meta_rank')
    
    # Clear losers: consistently in bottom 20% with low rank variance
    clear_losers = analysis_df[
        (analysis_df['final_meta_rank'] >= bottom_threshold) & 
        (analysis_df['rank_range'] <= 5)  # Consistent across methods
    ].sort_values('final_meta_rank', ascending=False)
    
    # Controversial assets: high rank variance (different methods disagree)
    controversial = analysis_df[
        analysis_df['rank_range'] >= 10  # High disagreement between methods
    ].sort_values('rank_range', ascending=False)
    
    return clear_winners, clear_losers, controversial, analysis_df

def save_enhanced_rankings(ranking_df, consistency_df, clear_winners, clear_losers, controversial, analysis_df):
    """
    Save all enhanced ranking analyses
    """
    output_dir = "/Users/tim/IWLS-OPTIONS/CORRELATION_ANALYSIS"
    
    # Main enhanced rankings
    enhanced_rankings = ranking_df[[
        'asset', 'reliability_score', 'spearman_correlation', 'spearman_pvalue',
        'r_squared', 'monotonicity_score', 'signal_to_noise_ratio',
        'equal_weight_score', 'correlation_focused_score', 'trading_focused_score', 'significance_focused_score',
        'equal_weight_rank', 'correlation_focused_rank', 'trading_focused_rank', 'significance_focused_rank',
        'final_meta_rank'
    ]].sort_values('final_meta_rank')
    
    enhanced_rankings.to_csv(f"{output_dir}/enhanced_multi_method_rankings.csv", index=False)
    
    # Consistency analysis
    consistency_df.to_csv(f"{output_dir}/ranking_consistency_analysis.csv", index=False)
    
    # Clear categories
    clear_winners.to_csv(f"{output_dir}/clear_winners.csv", index=False)
    clear_losers.to_csv(f"{output_dir}/clear_losers.csv", index=False)
    controversial.to_csv(f"{output_dir}/controversial_assets.csv", index=False)
    
    # Complete analysis
    analysis_df.to_csv(f"{output_dir}/complete_ranking_analysis.csv", index=False)
    
    return enhanced_rankings

def print_enhanced_analysis_summary(enhanced_rankings, clear_winners, clear_losers, controversial):
    """
    Print comprehensive summary of enhanced ranking analysis
    """
    print("\n" + "="*80)
    print("ENHANCED MULTI-METHOD RANKING ANALYSIS")
    print("="*80)
    
    print(f"\nMETHODOLOGY:")
    print(f"  - Equal Weight: All metrics weighted equally")
    print(f"  - Correlation Focused: Emphasizes correlation strength")
    print(f"  - Trading Focused: Emphasizes reliability and monotonicity")
    print(f"  - Significance Focused: Emphasizes statistical significance")
    print(f"  - Meta-Composite: Average of all ranking methods")
    
    print(f"\nTOP 10 ASSETS (Final Meta-Ranking):")
    print("-" * 80)
    print(f"{'Rank':<5} {'Asset':<6} {'Meta':<5} {'Rel.':<5} {'Spear':<8} {'R²':<6} {'Mono':<6} {'S/N':<6}")
    print("-" * 80)
    
    for i, (_, row) in enumerate(enhanced_rankings.head(10).iterrows()):
        print(f"{int(row['final_meta_rank']):>4} {row['asset']:<6} "
              f"{row['final_meta_rank']:>4.1f} {row['reliability_score']:>4.0f} "
              f"{row['spearman_correlation']:>7.3f} {row['r_squared']:>5.3f} "
              f"{row['monotonicity_score']:>5.3f} {row['signal_to_noise_ratio']:>5.1f}")
    
    print(f"\nCLEAR WINNERS (Consistent top performers):")
    print("-" * 60)
    if len(clear_winners) > 0:
        for _, row in clear_winners.iterrows():
            print(f"{row['asset']:<6} - Rank {row['final_meta_rank']:>4.1f} "
                  f"(Range: {row['rank_range']:>3.0f}, Reliability: {row['reliability_score']:>4.0f})")
        
        print(f"\nRECOMMENDATION: Focus your IWLS strategy on these {len(clear_winners)} assets")
        print(f"These show consistent strong performance across all ranking methods")
    else:
        print("No assets show consistent top performance across all methods")
    
    print(f"\nCLEAR LOSERS (Consistent poor performers):")
    print("-" * 60)
    if len(clear_losers) > 0:
        for _, row in clear_losers.iterrows():
            print(f"{row['asset']:<6} - Rank {row['final_meta_rank']:>4.1f} "
                  f"(Range: {row['rank_range']:>3.0f}, Reliability: {row['reliability_score']:>4.0f})")
        
        print(f"\nRECOMMENDATION: Avoid these {len(clear_losers)} assets for IWLS strategy")
        print(f"These consistently show poor deviation-performance relationships")
    else:
        print("No assets show consistent poor performance across all methods")
    
    print(f"\nCONTROVERSIAL ASSETS (Methods disagree):")
    print("-" * 60)
    if len(controversial) > 0:
        for _, row in controversial.head(5).iterrows():
            print(f"{row['asset']:<6} - Rank {row['final_meta_rank']:>4.1f} "
                  f"(Range: {row['rank_range']:>3.0f}, Consistency: {row['consistency_score']:>4.1f})")
        
        print(f"\nRECOMMENDATION: Use caution with these assets")
        print(f"Different ranking methods give conflicting assessments")
    else:
        print("All assets show consistent rankings across methods")
    
    # Strategy recommendations
    print(f"\nSTRATEGY IMPLEMENTATION RECOMMENDATIONS:")
    print("="*60)
    
    if len(clear_winners) >= 5:
        print(f"TIER 1 (Core Holdings): {len(clear_winners)} clear winners")
        print(f"  - Allocate 60-70% of strategy capital")
        print(f"  - High confidence in deviation signals")
        
        tier2_candidates = enhanced_rankings[
            (~enhanced_rankings['asset'].isin(clear_winners['asset'])) &
            (enhanced_rankings['final_meta_rank'] <= len(enhanced_rankings) * 0.4)
        ]
        
        if len(tier2_candidates) > 0:
            print(f"\nTIER 2 (Secondary Holdings): {len(tier2_candidates)} good performers")
            print(f"  - Allocate 20-30% of strategy capital")
            print(f"  - Moderate confidence in deviation signals")
        
        avoid_assets = enhanced_rankings[
            enhanced_rankings['final_meta_rank'] > len(enhanced_rankings) * 0.7
        ]
        
        if len(avoid_assets) > 0:
            print(f"\nAVOID: {len(avoid_assets)} poor performers")
            print(f"  - Do not include in IWLS strategy")
            print(f"  - Weak/unreliable deviation signals")
    
    else:
        print(f"WARNING: Only {len(clear_winners)} clear winners identified")
        print(f"Consider expanding criteria or revising IWLS methodology")

def main():
    print("Enhanced Multi-Method Asset Ranking Analysis")
    print("="*60)
    print("Creating composite rankings using multiple correlation methods")
    
    # Load data
    df = load_correlation_results()
    print(f"\nLoaded correlation data for {len(df)} assets")
    
    # Calculate multi-method rankings
    print("\nCalculating multi-method rankings...")
    ranking_df = calculate_multi_method_rankings(df)
    
    # Analyze ranking consistency
    print("Analyzing ranking consistency...")
    consistency_df = create_ranking_consistency_analysis(ranking_df)
    
    # Identify clear winners/losers
    print("Identifying clear winners and losers...")
    clear_winners, clear_losers, controversial, analysis_df = identify_clear_winners_losers(ranking_df, consistency_df)
    
    # Save results
    print("Saving enhanced ranking results...")
    enhanced_rankings = save_enhanced_rankings(ranking_df, consistency_df, clear_winners, clear_losers, controversial, analysis_df)
    
    # Print summary
    print_enhanced_analysis_summary(enhanced_rankings, clear_winners, clear_losers, controversial)
    
    print(f"\n" + "="*80)
    print("ENHANCED RANKING ANALYSIS COMPLETE")
    print("="*80)
    print("New files saved:")
    print("  - enhanced_multi_method_rankings.csv")
    print("  - ranking_consistency_analysis.csv")
    print("  - clear_winners.csv")
    print("  - clear_losers.csv") 
    print("  - controversial_assets.csv")
    print("  - complete_ranking_analysis.csv")

if __name__ == "__main__":
    main()