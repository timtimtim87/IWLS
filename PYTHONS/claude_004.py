import pandas as pd
import numpy as np
import matplotlib.pyplot as plt
import seaborn as sns
from datetime import datetime, timedelta
import warnings
import os
from scipy import stats
from scipy.stats import pearsonr, spearmanr
warnings.filterwarnings('ignore')

def load_rebalancing_results(v2_dir):
    """
    Load the results from the rebalancing strategy analysis
    """
    rebalancing_dir = os.path.join(v2_dir, "REBALANCING_STRATEGY_ANALYSIS")
    
    if not os.path.exists(rebalancing_dir):
        print("❌ REBALANCING_STRATEGY_ANALYSIS directory not found. Run the rebalancing analysis first.")
        return None, None, None
    
    # Load individual stock transactions
    transactions_file = os.path.join(rebalancing_dir, "individual_stock_transactions.csv")
    if not os.path.exists(transactions_file):
        print("❌ individual_stock_transactions.csv not found.")
        return None, None, None
    
    transactions_df = pd.read_csv(transactions_file)
    transactions_df['start_date'] = pd.to_datetime(transactions_df['start_date'])
    transactions_df['end_date'] = pd.to_datetime(transactions_df['end_date'])
    
    # Load portfolio compositions to get entry deviations
    compositions_file = os.path.join(rebalancing_dir, "portfolio_compositions.csv")
    portfolio_df = None
    if os.path.exists(compositions_file):
        portfolio_df = pd.read_csv(compositions_file)
        portfolio_df['start_date'] = pd.to_datetime(portfolio_df['start_date'])
    
    # Load quintile performance summary
    performance_file = os.path.join(rebalancing_dir, "quintile_performance_summary.csv")
    performance_df = None
    if os.path.exists(performance_file):
        performance_df = pd.read_csv(performance_file)
    
    print(f"✅ Loaded {len(transactions_df)} transactions from rebalancing analysis")
    
    return transactions_df, portfolio_df, performance_df

def load_iwls_data_for_assets(v2_dir, asset_list):
    """
    Load IWLS data for specific assets to get growth rates
    """
    print(f"Loading IWLS data for {len(asset_list)} assets...")
    
    iwls_data = {}
    
    for asset in asset_list:
        asset_dir = os.path.join(v2_dir, asset)
        iwls_file = os.path.join(asset_dir, f"{asset}_iwls_results.csv")
        
        if os.path.exists(iwls_file):
            try:
                df = pd.read_csv(iwls_file)
                df['date'] = pd.to_datetime(df['date'])
                df = df.dropna(subset=['annual_growth', 'price_deviation']).sort_values('date')
                iwls_data[asset] = df
                
            except Exception as e:
                print(f"  ⚠️  Error loading {asset}: {e}")
        else:
            print(f"  ⚠️  IWLS file not found for {asset}")
    
    print(f"✅ Loaded IWLS data for {len(iwls_data)} assets")
    return iwls_data

def get_growth_rate_at_entry(asset, entry_date, iwls_data, lookback_days=7):
    """
    Get the IWLS growth rate for an asset at the time of trade entry
    """
    if asset not in iwls_data:
        return np.nan, np.nan, np.nan
    
    asset_df = iwls_data[asset]
    
    # Find the closest data point to entry date (within lookback_days)
    date_mask = (asset_df['date'] >= entry_date - timedelta(days=lookback_days)) & \
               (asset_df['date'] <= entry_date + timedelta(days=lookback_days))
    nearby_data = asset_df[date_mask]
    
    if len(nearby_data) == 0:
        return np.nan, np.nan, np.nan
    
    # Get closest data point
    closest_idx = (nearby_data['date'] - entry_date).abs().idxmin()
    closest_row = nearby_data.loc[closest_idx]
    
    return closest_row['annual_growth'], closest_row['price_deviation'], closest_row['absolute_deviation']

def enrich_transactions_with_growth_rates(transactions_df, iwls_data):
    """
    Add growth rate information to each transaction
    """
    print("Enriching transactions with IWLS growth rates...")
    
    enriched_transactions = []
    
    for _, trade in transactions_df.iterrows():
        growth_rate, entry_deviation, entry_abs_deviation = get_growth_rate_at_entry(
            trade['asset'], trade['start_date'], iwls_data
        )
        
        enriched_trade = trade.to_dict()
        enriched_trade.update({
            'iwls_growth_rate': growth_rate,
            'entry_price_deviation': entry_deviation,
            'entry_abs_deviation': entry_abs_deviation
        })
        
        enriched_transactions.append(enriched_trade)
    
    enriched_df = pd.DataFrame(enriched_transactions)
    
    # Remove trades without growth rate data
    valid_trades = enriched_df.dropna(subset=['iwls_growth_rate'])
    
    print(f"✅ Successfully enriched {len(valid_trades)}/{len(enriched_df)} transactions with growth rates")
    
    return valid_trades

def categorize_growth_rates(growth_rate):
    """
    Categorize growth rates into meaningful bins
    """
    if pd.isna(growth_rate):
        return "Unknown"
    elif growth_rate < 0:
        return "Declining (<0%)"
    elif growth_rate < 10:
        return "Low Growth (0-10%)"
    elif growth_rate < 20:
        return "Moderate Growth (10-20%)"
    elif growth_rate < 30:
        return "High Growth (20-30%)"
    elif growth_rate < 50:
        return "Very High Growth (30-50%)"
    else:
        return "Extreme Growth (>50%)"

def categorize_deviation_magnitude(abs_deviation):
    """
    Categorize absolute deviation magnitude
    """
    if pd.isna(abs_deviation):
        return "Unknown"
    elif abs_deviation < 10:
        return "Small Deviation (<10%)"
    elif abs_deviation < 20:
        return "Moderate Deviation (10-20%)"
    elif abs_deviation < 30:
        return "Large Deviation (20-30%)"
    elif abs_deviation < 50:
        return "Very Large Deviation (30-50%)"
    else:
        return "Extreme Deviation (>50%)"

def analyze_growth_rate_vs_performance(enriched_df):
    """
    Analyze the relationship between growth rates and trade performance
    """
    print("\nAnalyzing growth rate vs trade performance...")
    
    # Add categorical variables
    enriched_df['growth_rate_category'] = enriched_df['iwls_growth_rate'].apply(categorize_growth_rates)
    enriched_df['deviation_category'] = enriched_df['entry_abs_deviation'].apply(categorize_deviation_magnitude)
    
    # Overall correlation analysis
    valid_data = enriched_df.dropna(subset=['iwls_growth_rate', 'return_pct'])
    
    if len(valid_data) > 10:
        pearson_corr, pearson_p = pearsonr(valid_data['iwls_growth_rate'], valid_data['return_pct'])
        spearman_corr, spearman_p = spearmanr(valid_data['iwls_growth_rate'], valid_data['return_pct'])
        
        print(f"\nOverall Correlations (Growth Rate vs Trade Return):")
        print(f"  Pearson correlation: {pearson_corr:.3f} (p-value: {pearson_p:.3f})")
        print(f"  Spearman correlation: {spearman_corr:.3f} (p-value: {spearman_p:.3f})")
        
        if pearson_p < 0.05:
            direction = "positive" if pearson_corr > 0 else "negative"
            print(f"  ✅ Statistically significant {direction} correlation!")
        else:
            print(f"  ⚠️  No statistically significant correlation")
    
    # Analysis by growth rate categories
    growth_rate_analysis = []
    
    growth_rate_order = ["Declining (<0%)", "Low Growth (0-10%)", "Moderate Growth (10-20%)", 
                        "High Growth (20-30%)", "Very High Growth (30-50%)", "Extreme Growth (>50%)"]
    
    for category in growth_rate_order:
        category_data = enriched_df[enriched_df['growth_rate_category'] == category]
        
        if len(category_data) >= 5:  # Minimum sample size
            returns = category_data['return_pct']
            
            growth_rate_analysis.append({
                'growth_rate_category': category,
                'sample_count': len(category_data),
                'avg_return': returns.mean(),
                'median_return': returns.median(),
                'std_return': returns.std(),
                'min_return': returns.min(),
                'max_return': returns.max(),
                'success_rate': (returns > 0).mean() * 100,
                'avg_growth_rate': category_data['iwls_growth_rate'].mean(),
                'avg_deviation': category_data['entry_abs_deviation'].mean(),
                'quintile_1_count': (category_data['quintile'] == 'quintile_1').sum(),
                'quintile_2_count': (category_data['quintile'] == 'quintile_2').sum(),
                'quintile_3_count': (category_data['quintile'] == 'quintile_3').sum(),
                'quintile_4_count': (category_data['quintile'] == 'quintile_4').sum(),
                'quintile_5_count': (category_data['quintile'] == 'quintile_5').sum()
            })
    
    growth_analysis_df = pd.DataFrame(growth_rate_analysis)
    
    # Analysis by quintile and growth rate
    quintile_growth_analysis = []
    
    for quintile in ['quintile_1', 'quintile_2', 'quintile_3', 'quintile_4', 'quintile_5']:
        quintile_data = enriched_df[enriched_df['quintile'] == quintile]
        
        for category in growth_rate_order:
            subset_data = quintile_data[quintile_data['growth_rate_category'] == category]
            
            if len(subset_data) >= 3:  # Minimum for quintile analysis
                returns = subset_data['return_pct']
                
                quintile_growth_analysis.append({
                    'quintile': quintile,
                    'quintile_rank': int(quintile.split('_')[1]),
                    'growth_rate_category': category,
                    'sample_count': len(subset_data),
                    'avg_return': returns.mean(),
                    'median_return': returns.median(),
                    'success_rate': (returns > 0).mean() * 100,
                    'avg_growth_rate': subset_data['iwls_growth_rate'].mean(),
                    'avg_deviation': subset_data['entry_abs_deviation'].mean()
                })
    
    quintile_growth_df = pd.DataFrame(quintile_growth_analysis)
    
    return growth_analysis_df, quintile_growth_df, enriched_df

def analyze_deviation_growth_interaction(enriched_df):
    """
    Analyze the interaction between deviation magnitude and growth rate
    """
    print("\nAnalyzing deviation magnitude vs growth rate interaction...")
    
    # Create 2D analysis: deviation category vs growth rate category
    interaction_analysis = []
    
    deviation_order = ["Small Deviation (<10%)", "Moderate Deviation (10-20%)", 
                      "Large Deviation (20-30%)", "Very Large Deviation (30-50%)", 
                      "Extreme Deviation (>50%)"]
    
    growth_rate_order = ["Declining (<0%)", "Low Growth (0-10%)", "Moderate Growth (10-20%)", 
                        "High Growth (20-30%)", "Very High Growth (30-50%)", "Extreme Growth (>50%)"]
    
    for dev_cat in deviation_order:
        for growth_cat in growth_rate_order:
            subset = enriched_df[
                (enriched_df['deviation_category'] == dev_cat) & 
                (enriched_df['growth_rate_category'] == growth_cat)
            ]
            
            if len(subset) >= 3:  # Minimum sample
                returns = subset['return_pct']
                
                interaction_analysis.append({
                    'deviation_category': dev_cat,
                    'growth_rate_category': growth_cat,
                    'sample_count': len(subset),
                    'avg_return': returns.mean(),
                    'median_return': returns.median(),
                    'success_rate': (returns > 0).mean() * 100,
                    'avg_deviation': subset['entry_abs_deviation'].mean(),
                    'avg_growth_rate': subset['iwls_growth_rate'].mean(),
                    'best_return': returns.max(),
                    'worst_return': returns.min()
                })
    
    interaction_df = pd.DataFrame(interaction_analysis)
    
    return interaction_df

def create_comprehensive_visualizations(enriched_df, growth_analysis_df, quintile_growth_df, 
                                      interaction_df, output_dir):
    """
    Create comprehensive visualizations of growth rate analysis
    """
    print("\nCreating comprehensive visualizations...")
    
    # Figure 1: Growth Rate vs Returns Analysis
    fig, ((ax1, ax2), (ax3, ax4)) = plt.subplots(2, 2, figsize=(20, 16))
    
    # Scatter plot: Growth Rate vs Return
    valid_data = enriched_df.dropna(subset=['iwls_growth_rate', 'return_pct'])
    
    if len(valid_data) > 0:
        scatter = ax1.scatter(valid_data['iwls_growth_rate'], valid_data['return_pct'], 
                             c=valid_data['entry_abs_deviation'], cmap='viridis', 
                             alpha=0.6, s=30)
        ax1.set_xlabel('IWLS Growth Rate (%)')
        ax1.set_ylabel('Trade Return (%)')
        ax1.set_title('Growth Rate vs Trade Return (colored by deviation magnitude)', fontweight='bold')
        ax1.grid(True, alpha=0.3)
        ax1.axhline(y=0, color='red', linestyle='--', alpha=0.5)
        ax1.axvline(x=0, color='red', linestyle='--', alpha=0.5)
        
        # Add trend line
        if len(valid_data) > 10:
            z = np.polyfit(valid_data['iwls_growth_rate'], valid_data['return_pct'], 1)
            p = np.poly1d(z)
            x_trend = np.linspace(valid_data['iwls_growth_rate'].min(), 
                                 valid_data['iwls_growth_rate'].max(), 100)
            ax1.plot(x_trend, p(x_trend), "r-", alpha=0.8, linewidth=2)
        
        plt.colorbar(scatter, ax=ax1, label='Entry Deviation (%)')
    
    # Average returns by growth rate category
    if len(growth_analysis_df) > 0:
        colors = plt.cm.RdYlGn(np.linspace(0.2, 0.8, len(growth_analysis_df)))
        bars = ax2.bar(range(len(growth_analysis_df)), growth_analysis_df['avg_return'], 
                      color=colors, alpha=0.8)
        ax2.set_xticks(range(len(growth_analysis_df)))
        ax2.set_xticklabels(growth_analysis_df['growth_rate_category'], rotation=45, ha='right')
        ax2.set_ylabel('Average Trade Return (%)')
        ax2.set_title('Average Returns by Growth Rate Category', fontweight='bold')
        ax2.grid(True, alpha=0.3)
        ax2.axhline(y=0, color='black', linestyle='-', alpha=0.5)
        
        # Add sample count labels
        for i, (bar, count) in enumerate(zip(bars, growth_analysis_df['sample_count'])):
            ax2.text(bar.get_x() + bar.get_width()/2., 
                    bar.get_height() + (abs(bar.get_height()) * 0.05),
                    f'n={count}', ha='center', va='bottom', fontsize=9, fontweight='bold')
    
    # Success rates by growth rate category
    if len(growth_analysis_df) > 0:
        bars = ax3.bar(range(len(growth_analysis_df)), growth_analysis_df['success_rate'], 
                      color='steelblue', alpha=0.8)
        ax3.set_xticks(range(len(growth_analysis_df)))
        ax3.set_xticklabels(growth_analysis_df['growth_rate_category'], rotation=45, ha='right')
        ax3.set_ylabel('Success Rate (%)')
        ax3.set_title('Success Rate by Growth Rate Category', fontweight='bold')
        ax3.grid(True, alpha=0.3)
        ax3.axhline(y=50, color='red', linestyle='--', alpha=0.5, label='50% baseline')
        ax3.legend()
        
        # Add value labels
        for bar, value in zip(bars, growth_analysis_df['success_rate']):
            ax3.text(bar.get_x() + bar.get_width()/2., bar.get_height() + 1,
                    f'{value:.1f}%', ha='center', va='bottom', fontweight='bold')
    
    # Box plot: Returns distribution by growth rate category
    if len(enriched_df) > 0:
        growth_categories = enriched_df['growth_rate_category'].unique()
        box_data = []
        box_labels = []
        
        for category in growth_analysis_df['growth_rate_category']:
            category_returns = enriched_df[enriched_df['growth_rate_category'] == category]['return_pct']
            if len(category_returns) >= 5:
                box_data.append(category_returns.values)
                box_labels.append(category.replace(' ', '\n'))
        
        if box_data:
            bp = ax4.boxplot(box_data, labels=box_labels, patch_artist=True)
            for patch in bp['boxes']:
                patch.set_facecolor('lightcoral')
                patch.set_alpha(0.7)
            
            ax4.set_ylabel('Trade Return (%)')
            ax4.set_title('Return Distribution by Growth Rate Category', fontweight='bold')
            ax4.grid(True, alpha=0.3)
            ax4.axhline(y=0, color='red', linestyle='--', alpha=0.5)
    
    plt.tight_layout()
    plt.savefig(f"{output_dir}/growth_rate_vs_returns_analysis.png", dpi=300, bbox_inches='tight')
    plt.close()
    
    # Figure 2: Quintile vs Growth Rate Heatmap
    if len(quintile_growth_df) > 0:
        fig, (ax1, ax2) = plt.subplots(1, 2, figsize=(20, 8))
        
        # Create pivot table for heatmap
        heatmap_data = quintile_growth_df.pivot_table(
            index='quintile_rank', 
            columns='growth_rate_category', 
            values='avg_return', 
            fill_value=np.nan
        )
        
        # Reorder columns
        growth_order = ["Declining (<0%)", "Low Growth (0-10%)", "Moderate Growth (10-20%)", 
                       "High Growth (20-30%)", "Very High Growth (30-50%)", "Extreme Growth (>50%)"]
        available_cols = [col for col in growth_order if col in heatmap_data.columns]
        heatmap_data = heatmap_data[available_cols]
        
        # Average returns heatmap
        im1 = ax1.imshow(heatmap_data.values, cmap='RdYlGn', aspect='auto', vmin=-50, vmax=50)
        ax1.set_xticks(range(len(heatmap_data.columns)))
        ax1.set_xticklabels([col.replace(' ', '\n') for col in heatmap_data.columns], fontsize=9)
        ax1.set_yticks(range(len(heatmap_data.index)))
        ax1.set_yticklabels([f'Q{i}' for i in heatmap_data.index])
        ax1.set_title('Average Returns by Quintile & Growth Rate', fontweight='bold')
        
        # Add text annotations
        for i in range(len(heatmap_data.index)):
            for j in range(len(heatmap_data.columns)):
                value = heatmap_data.iloc[i, j]
                if not np.isnan(value):
                    ax1.text(j, i, f'{value:.1f}%', ha='center', va='center', 
                            fontweight='bold', fontsize=10)
        
        plt.colorbar(im1, ax=ax1, label='Average Return (%)')
        
        # Sample count heatmap
        sample_heatmap = quintile_growth_df.pivot_table(
            index='quintile_rank', 
            columns='growth_rate_category', 
            values='sample_count', 
            fill_value=0
        )
        sample_heatmap = sample_heatmap[available_cols]
        
        im2 = ax2.imshow(sample_heatmap.values, cmap='Blues', aspect='auto')
        ax2.set_xticks(range(len(sample_heatmap.columns)))
        ax2.set_xticklabels([col.replace(' ', '\n') for col in sample_heatmap.columns], fontsize=9)
        ax2.set_yticks(range(len(sample_heatmap.index)))
        ax2.set_yticklabels([f'Q{i}' for i in sample_heatmap.index])
        ax2.set_title('Sample Count by Quintile & Growth Rate', fontweight='bold')
        
        # Add text annotations
        for i in range(len(sample_heatmap.index)):
            for j in range(len(sample_heatmap.columns)):
                value = sample_heatmap.iloc[i, j]
                ax2.text(j, i, f'{int(value)}', ha='center', va='center', 
                        fontweight='bold', fontsize=10)
        
        plt.colorbar(im2, ax=ax2, label='Sample Count')
        plt.tight_layout()
        plt.savefig(f"{output_dir}/quintile_growth_rate_heatmaps.png", dpi=300, bbox_inches='tight')
        plt.close()
    
    # Figure 3: Deviation vs Growth Rate Interaction
    if len(interaction_df) > 0:
        fig, ax = plt.subplots(1, 1, figsize=(16, 10))
        
        # Create pivot for interaction heatmap
        interaction_pivot = interaction_df.pivot_table(
            index='deviation_category',
            columns='growth_rate_category',
            values='avg_return',
            fill_value=np.nan
        )
        
        # Reorder
        dev_order = ["Small Deviation (<10%)", "Moderate Deviation (10-20%)", 
                    "Large Deviation (20-30%)", "Very Large Deviation (30-50%)", 
                    "Extreme Deviation (>50%)"]
        growth_order = ["Declining (<0%)", "Low Growth (0-10%)", "Moderate Growth (10-20%)", 
                       "High Growth (20-30%)", "Very High Growth (30-50%)", "Extreme Growth (>50%)"]
        
        available_dev = [d for d in dev_order if d in interaction_pivot.index]
        available_growth = [g for g in growth_order if g in interaction_pivot.columns]
        
        interaction_pivot = interaction_pivot.loc[available_dev, available_growth]
        
        im = ax.imshow(interaction_pivot.values, cmap='RdYlGn', aspect='auto', vmin=-50, vmax=50)
        ax.set_xticks(range(len(interaction_pivot.columns)))
        ax.set_xticklabels([col.replace(' ', '\n') for col in interaction_pivot.columns], fontsize=9)
        ax.set_yticks(range(len(interaction_pivot.index)))
        ax.set_yticklabels([idx.replace(' ', '\n') for idx in interaction_pivot.index], fontsize=9)
        ax.set_title('Average Returns by Deviation Magnitude & Growth Rate', fontweight='bold', fontsize=14)
        
        # Add text annotations
        for i in range(len(interaction_pivot.index)):
            for j in range(len(interaction_pivot.columns)):
                value = interaction_pivot.iloc[i, j]
                if not np.isnan(value):
                    ax.text(j, i, f'{value:.1f}%', ha='center', va='center', 
                           fontweight='bold', fontsize=8)
        
        plt.colorbar(im, ax=ax, label='Average Return (%)')
        plt.tight_layout()
        plt.savefig(f"{output_dir}/deviation_growth_rate_interaction.png", dpi=300, bbox_inches='tight')
        plt.close()
    
    print("  ✅ All visualizations created")

def save_analysis_results(enriched_df, growth_analysis_df, quintile_growth_df, 
                         interaction_df, output_dir):
    """
    Save all analysis results to CSV files
    """
    print("\nSaving detailed analysis results...")
    
    # Save enriched transactions with growth rates
    enriched_df.to_csv(f"{output_dir}/transactions_with_growth_rates.csv", index=False)
    print(f"  ✅ Saved enriched transactions: {len(enriched_df)} records")
    
    # Save growth rate analysis
    growth_analysis_df.to_csv(f"{output_dir}/growth_rate_category_analysis.csv", index=False)
    print(f"  ✅ Saved growth rate analysis: {len(growth_analysis_df)} categories")
    
    # Save quintile vs growth rate analysis
    quintile_growth_df.to_csv(f"{output_dir}/quintile_growth_rate_analysis.csv", index=False)
    print(f"  ✅ Saved quintile-growth analysis: {len(quintile_growth_df)} combinations")
    
    # Save interaction analysis
    interaction_df.to_csv(f"{output_dir}/deviation_growth_interaction_analysis.csv", index=False)
    print(f"  ✅ Saved interaction analysis: {len(interaction_df)} combinations")
    
    # Create summary statistics
    summary_stats = {
        'total_trades_analyzed': len(enriched_df),
        'avg_trade_return': enriched_df['return_pct'].mean(),
        'avg_growth_rate': enriched_df['iwls_growth_rate'].mean(),
        'avg_entry_deviation': enriched_df['entry_abs_deviation'].mean(),
        'correlation_growth_return': enriched_df[['iwls_growth_rate', 'return_pct']].corr().iloc[0,1],
        'high_growth_avg_return': enriched_df[enriched_df['iwls_growth_rate'] > 30]['return_pct'].mean(),
        'low_growth_avg_return': enriched_df[enriched_df['iwls_growth_rate'] <= 10]['return_pct'].mean(),
        'large_deviation_avg_return': enriched_df[enriched_df['entry_abs_deviation'] > 30]['return_pct'].mean(),
        'small_deviation_avg_return': enriched_df[enriched_df['entry_abs_deviation'] <= 10]['return_pct'].mean()
    }
    
    summary_df = pd.DataFrame([summary_stats])
    summary_df.to_csv(f"{output_dir}/analysis_summary_statistics.csv", index=False)
    print(f"  ✅ Saved summary statistics")

def print_comprehensive_insights(enriched_df, growth_analysis_df, quintile_growth_df, interaction_df):
    """
    Print comprehensive insights from the analysis
    """
    print("\n" + "="*80)
    print("GROWTH RATE vs TRADE PERFORMANCE ANALYSIS - KEY INSIGHTS")
    print("="*80)
    
    # Overall correlation
    valid_data = enriched_df.dropna(subset=['iwls_growth_rate', 'return_pct'])
    if len(valid_data) > 10:
        correlation = valid_data[['iwls_growth_rate', 'return_pct']].corr().iloc[0,1]
        print(f"\n📊 OVERALL RELATIONSHIP:")
        print(f"  Correlation (Growth Rate vs Return): {correlation:.3f}")
        
        if abs(correlation) > 0.3:
            direction = "positive" if correlation > 0 else "negative"
            strength = "strong" if abs(correlation) > 0.5 else "moderate"
            print(f"  ✅ {strength.title()} {direction} correlation detected!")
        else:
            print(f"  ⚠️  Weak correlation - other factors may be more important")
    
    # Growth rate category insights
    if len(growth_analysis_df) > 0:
        print(f"\n🚀 GROWTH RATE CATEGORY PERFORMANCE:")
        print("-" * 60)
        
        best_growth_cat = growth_analysis_df.loc[growth_analysis_df['avg_return'].idxmax()]
        worst_growth_cat = growth_analysis_df.loc[growth_analysis_df['avg_return'].idxmin()]
        
        print(f"  Best performing: {best_growth_cat['growth_rate_category']}")
        print(f"    • Average return: {best_growth_cat['avg_return']:.2f}%")
        print(f"    • Sample count: {best_growth_cat['sample_count']}")
        print(f"    • Success rate: {best_growth_cat['success_rate']:.1f}%")
        
        print(f"\n  Worst performing: {worst_growth_cat['growth_rate_category']}")
        print(f"    • Average return: {worst_growth_cat['avg_return']:.2f}%")
        print(f"    • Sample count: {worst_growth_cat['sample_count']}")
        print(f"    • Success rate: {worst_growth_cat['success_rate']:.1f}%")
        
        performance_spread = best_growth_cat['avg_return'] - worst_growth_cat['avg_return']
        print(f"\n  Performance spread: {performance_spread:.2f}%")
        
        # Analyze high vs low growth
        high_growth = growth_analysis_df[growth_analysis_df['growth_rate_category'].isin([
            'Very High Growth (30-50%)', 'Extreme Growth (>50%)'
        ])]
        low_growth = growth_analysis_df[growth_analysis_df['growth_rate_category'].isin([
            'Low Growth (0-10%)', 'Moderate Growth (10-20%)'
        ])]
        
        if len(high_growth) > 0 and len(low_growth) > 0:
            high_growth_avg = high_growth['avg_return'].mean()
            low_growth_avg = low_growth['avg_return'].mean()
            
            print(f"\n  High growth stocks (>30%): {high_growth_avg:.2f}% avg return")
            print(f"  Low growth stocks (<20%): {low_growth_avg:.2f}% avg return")
            
            if high_growth_avg > low_growth_avg:
                print(f"  ✅ High growth stocks outperformed by {high_growth_avg - low_growth_avg:.2f}%")
                print("  💡 Insight: Fast-growing stocks with large deviations may offer better rebounds")
            else:
                print(f"  ⚠️  Low growth stocks outperformed by {low_growth_avg - high_growth_avg:.2f}%")
                print("  💡 Insight: Slower growth may indicate more stable, reliable rebounds")
    
    # Quintile-specific insights
    if len(quintile_growth_df) > 0:
        print(f"\n🎯 QUINTILE-SPECIFIC INSIGHTS:")
        print("-" * 50)
        
        # Focus on Quintile 1 (most underperforming)
        q1_data = quintile_growth_df[quintile_growth_df['quintile_rank'] == 1]
        
        if len(q1_data) > 0:
            print(f"  QUINTILE 1 (Most Underperforming) by Growth Rate:")
            
            for _, row in q1_data.iterrows():
                print(f"    {row['growth_rate_category']:<25}: "
                      f"{row['avg_return']:>7.2f}% (n={row['sample_count']})")
            
            best_q1_growth = q1_data.loc[q1_data['avg_return'].idxmax()]
            print(f"\n  🏆 Best Q1 combination: {best_q1_growth['growth_rate_category']}")
            print(f"      Return: {best_q1_growth['avg_return']:.2f}%")
            print(f"      Sample size: {best_q1_growth['sample_count']}")
        
        # Compare quintiles within same growth categories
        print(f"\n  Growth Rate Impact Across Quintiles:")
        
        # Find growth categories with data across multiple quintiles
        growth_cats_with_data = quintile_growth_df['growth_rate_category'].value_counts()
        common_growth_cats = growth_cats_with_data[growth_cats_with_data >= 3].index
        
        for growth_cat in common_growth_cats[:3]:  # Show top 3
            cat_data = quintile_growth_df[quintile_growth_df['growth_rate_category'] == growth_cat]
            cat_data_sorted = cat_data.sort_values('quintile_rank')
            
            print(f"\n    {growth_cat}:")
            for _, row in cat_data_sorted.iterrows():
                print(f"      Q{row['quintile_rank']}: {row['avg_return']:>7.2f}% (n={row['sample_count']})")
    
    # Interaction insights (deviation + growth rate)
    if len(interaction_df) > 0:
        print(f"\n🔍 DEVIATION + GROWTH RATE INTERACTION INSIGHTS:")
        print("-" * 60)
        
        # Find best combinations
        best_combination = interaction_df.loc[interaction_df['avg_return'].idxmax()]
        worst_combination = interaction_df.loc[interaction_df['avg_return'].idxmin()]
        
        print(f"  🏆 BEST COMBINATION:")
        print(f"    Deviation: {best_combination['deviation_category']}")
        print(f"    Growth Rate: {best_combination['growth_rate_category']}")
        print(f"    Average Return: {best_combination['avg_return']:.2f}%")
        print(f"    Success Rate: {best_combination['success_rate']:.1f}%")
        print(f"    Sample Count: {best_combination['sample_count']}")
        
        print(f"\n  💀 WORST COMBINATION:")
        print(f"    Deviation: {worst_combination['deviation_category']}")
        print(f"    Growth Rate: {worst_combination['growth_rate_category']}")
        print(f"    Average Return: {worst_combination['avg_return']:.2f}%")
        print(f"    Success Rate: {worst_combination['success_rate']:.1f}%")
        print(f"    Sample Count: {worst_combination['sample_count']}")
        
        # Analyze large deviations with different growth rates
        large_dev_data = interaction_df[interaction_df['deviation_category'].isin([
            'Very Large Deviation (30-50%)', 'Extreme Deviation (>50%)'
        ])]
        
        if len(large_dev_data) > 0:
            print(f"\n  📊 LARGE DEVIATIONS (>30%) BY GROWTH RATE:")
            large_dev_sorted = large_dev_data.sort_values('avg_return', ascending=False)
            
            for _, row in large_dev_sorted.head(5).iterrows():
                print(f"    {row['growth_rate_category']:<25}: "
                      f"{row['avg_return']:>7.2f}% (n={row['sample_count']})")
        
        # Key patterns
        print(f"\n  🧠 KEY PATTERNS IDENTIFIED:")
        
        # Pattern 1: High growth + large deviation
        high_growth_large_dev = interaction_df[
            (interaction_df['growth_rate_category'].isin(['Very High Growth (30-50%)', 'Extreme Growth (>50%)'])) &
            (interaction_df['deviation_category'].isin(['Very Large Deviation (30-50%)', 'Extreme Deviation (>50%)']))
        ]
        
        if len(high_growth_large_dev) > 0:
            hg_ld_return = high_growth_large_dev['avg_return'].mean()
            print(f"    • High Growth + Large Deviation: {hg_ld_return:.2f}% avg return")
        
        # Pattern 2: Low growth + large deviation
        low_growth_large_dev = interaction_df[
            (interaction_df['growth_rate_category'].isin(['Low Growth (0-10%)', 'Moderate Growth (10-20%)'])) &
            (interaction_df['deviation_category'].isin(['Very Large Deviation (30-50%)', 'Extreme Deviation (>50%)']))
        ]
        
        if len(low_growth_large_dev) > 0:
            lg_ld_return = low_growth_large_dev['avg_return'].mean()
            print(f"    • Low Growth + Large Deviation: {lg_ld_return:.2f}% avg return")
        
        # Compare the patterns
        if len(high_growth_large_dev) > 0 and len(low_growth_large_dev) > 0:
            if hg_ld_return > lg_ld_return:
                print(f"    ✅ HIGH GROWTH + LARGE DEVIATION outperforms by {hg_ld_return - lg_ld_return:.2f}%")
                print("    💡 Your hypothesis CONFIRMED: Fast growth + large selloff = better rebound!")
            else:
                print(f"    ⚠️  LOW GROWTH + LARGE DEVIATION outperforms by {lg_ld_return - hg_ld_return:.2f}%")
                print("    💡 Counter-intuitive: Slower growth may indicate more sustainable rebounds")
    
    # Final summary and actionable insights
    print(f"\n" + "="*80)
    print("🎯 ACTIONABLE TRADING INSIGHTS")
    print("="*80)
    
    # Find the optimal strategy based on all analysis
    if len(interaction_df) > 0:
        # Get combinations with meaningful sample sizes and good returns
        viable_combinations = interaction_df[
            (interaction_df['sample_count'] >= 5) & 
            (interaction_df['avg_return'] > 10)
        ].sort_values('avg_return', ascending=False)
        
        if len(viable_combinations) > 0:
            print(f"\n🏆 TOP TRADING OPPORTUNITIES (min 5 samples, >10% return):")
            print("-" * 70)
            
            for i, (_, row) in enumerate(viable_combinations.head(5).iterrows()):
                print(f"  #{i+1}: {row['avg_return']:.1f}% return")
                print(f"      Strategy: Target {row['deviation_category']} + {row['growth_rate_category']}")
                print(f"      Success Rate: {row['success_rate']:.1f}%")
                print(f"      Sample Size: {row['sample_count']} trades")
                print()
        
        # Risk warning for extreme cases
        extreme_cases = interaction_df[
            (interaction_df['deviation_category'] == 'Extreme Deviation (>50%)') |
            (interaction_df['growth_rate_category'] == 'Extreme Growth (>50%)')
        ]
        
        if len(extreme_cases) > 0:
            avg_extreme_return = extreme_cases['avg_return'].mean()
            print(f"⚠️  EXTREME CASES WARNING:")
            print(f"   Extreme deviations (>50%) or growth (>50%): {avg_extreme_return:.1f}% avg return")
            print(f"   Sample size: {extreme_cases['sample_count'].sum()} trades")
            
            if avg_extreme_return < 0:
                print("   🚨 AVOID: Extreme cases show negative returns on average")
            else:
                print("   💰 OPPORTUNITY: But proceed with caution due to high volatility")
    
    print(f"\n💡 STRATEGIC RECOMMENDATIONS:")
    print(f"   1. Focus on combinations with 15+ samples for reliability")
    print(f"   2. Prioritize moderate-high growth (20-50%) over extreme growth (>50%)")
    print(f"   3. Large deviations (20-50%) may be sweet spot vs extreme (>50%)")
    print(f"   4. Monitor success rates alongside average returns")
    print(f"   5. Consider position sizing based on historical volatility patterns")

def main():
    print("GROWTH RATE vs TRADE PERFORMANCE ANALYSIS")
    print("=" * 70)
    print("Analyzing whether IWLS growth rates affect rebalancing strategy performance")
    print("Hypothesis: High growth + large deviation = better rebound potential")
    
    # Setup directories
    v2_dir = "/Users/tim/IWLS-OPTIONS/IWLS_ANALYSIS_V2"
    
    if not os.path.exists(v2_dir):
        print("❌ IWLS_ANALYSIS_V2 directory not found.")
        return
    
    # Create output directory
    output_dir = os.path.join(v2_dir, "GROWTH_RATE_TRADE_ANALYSIS")
    os.makedirs(output_dir, exist_ok=True)
    
    print(f"\nOutput directory: {output_dir}")
    
    # Load rebalancing strategy results
    transactions_df, portfolio_df, performance_df = load_rebalancing_results(v2_dir)
    
    if transactions_df is None:
        return
    
    # Get unique assets from transactions
    unique_assets = transactions_df['asset'].unique()
    print(f"\nFound transactions for {len(unique_assets)} unique assets")
    
    # Load IWLS data for these assets
    iwls_data = load_iwls_data_for_assets(v2_dir, unique_assets)
    
    # Enrich transactions with growth rate data
    enriched_df = enrich_transactions_with_growth_rates(transactions_df, iwls_data)
    
    if len(enriched_df) == 0:
        print("❌ No transactions could be enriched with growth rate data")
        return
    
    print(f"\n✅ Analysis dataset ready: {len(enriched_df)} trades with growth rate data")
    
    # Perform comprehensive analysis
    growth_analysis_df, quintile_growth_df, enriched_df = analyze_growth_rate_vs_performance(enriched_df)
    
    # Analyze interaction between deviation and growth rate
    interaction_df = analyze_deviation_growth_interaction(enriched_df)
    
    # Create visualizations
    create_comprehensive_visualizations(
        enriched_df, growth_analysis_df, quintile_growth_df, interaction_df, output_dir
    )
    
    # Save all results
    save_analysis_results(
        enriched_df, growth_analysis_df, quintile_growth_df, interaction_df, output_dir
    )
    
    # Print comprehensive insights
    print_comprehensive_insights(enriched_df, growth_analysis_df, quintile_growth_df, interaction_df)
    
    print(f"\n" + "="*70)
    print("GROWTH RATE ANALYSIS COMPLETE")
    print("="*70)
    print(f"Results saved to: {output_dir}")
    print("\nFiles created:")
    print("  📄 transactions_with_growth_rates.csv (enriched transaction data)")
    print("  📄 growth_rate_category_analysis.csv (performance by growth rate)")
    print("  📄 quintile_growth_rate_analysis.csv (quintile vs growth rate)")
    print("  📄 deviation_growth_interaction_analysis.csv (2D interaction analysis)")
    print("  📄 analysis_summary_statistics.csv (key summary metrics)")
    print("  📊 growth_rate_vs_returns_analysis.png (main analysis charts)")
    print("  📊 quintile_growth_rate_heatmaps.png (quintile vs growth heatmaps)")
    print("  📊 deviation_growth_rate_interaction.png (2D interaction heatmap)")
    
    print(f"\n🎯 Key questions answered:")
    print(f"   • Does IWLS growth rate predict trade success?")
    print(f"   • Do high-growth stocks rebound better from large deviations?")
    print(f"   • Which growth rate + deviation combinations work best?")
    print(f"   • Are there different patterns across quintiles?")
    print(f"   • What are the optimal entry criteria?")

if __name__ == "__main__":
    main()