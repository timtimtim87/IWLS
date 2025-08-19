import pandas as pd
import numpy as np
import matplotlib.pyplot as plt
from scipy import stats
from scipy.interpolate import interp1d
from sklearn.linear_model import LinearRegression, Ridge
from sklearn.preprocessing import PolynomialFeatures
from sklearn.pipeline import make_pipeline
from sklearn.metrics import r2_score
import os
import glob
import warnings
warnings.filterwarnings('ignore')

def load_previous_analysis_results():
    """
    Load results from previous Z-score and confidence analyses
    """
    try:
        # Load Z-score expected gains
        zscore_gains = pd.read_csv("/Users/tim/IWLS-OPTIONS/ZSCORE_ANALYSIS/zscore_expected_gains_comparison.csv")
        print("✅ Loaded Z-score expected gains data")
    except:
        print("❌ Could not load Z-score expected gains. Run Z-score analysis first.")
        return None, None, None
    
    try:
        # Load confidence scores
        confidence_df = pd.read_csv("/Users/tim/IWLS-OPTIONS/CORRELATION_ANALYSIS/enhanced_multi_method_rankings.csv")
        print("✅ Loaded confidence scores")
    except:
        print("❌ Could not load confidence scores. Run correlation analysis first.")
        return None, None, None
    
    try:
        # Load raw Z-score data for calculating asset averages
        raw_zscore_data = pd.read_csv("/Users/tim/IWLS-OPTIONS/ZSCORE_ANALYSIS/zscore_forward_gains_data.csv")
        print("✅ Loaded raw Z-score data")
    except:
        print("❌ Could not load raw Z-score data. Run Z-score analysis first.")
        return None, None, None
    
    return zscore_gains, confidence_df, raw_zscore_data

def calculate_asset_historical_averages(raw_zscore_data):
    """
    Calculate historical average forward gains for each asset
    """
    asset_averages = raw_zscore_data.groupby('asset')['forward_max_gain'].agg([
        'mean', 'median', 'std', 'count'
    ]).round(2)
    
    asset_averages.columns = ['mean_gain', 'median_gain', 'std_gain', 'sample_count']
    asset_averages = asset_averages.reset_index()
    
    print(f"\nHistorical averages calculated for {len(asset_averages)} assets")
    
    return asset_averages

def calculate_corrected_expected_values(zscore_gains, confidence_df, asset_averages):
    """
    Calculate corrected expected values using confidence-weighted approach
    """
    print("\nCalculating corrected expected values...")
    
    # Create confidence lookup
    confidence_lookup = dict(zip(confidence_df['asset'], confidence_df['reliability_score']))
    
    # Create asset average lookup
    avg_lookup = dict(zip(asset_averages['asset'], asset_averages['mean_gain']))
    
    corrected_ev_data = []
    
    for _, row in zscore_gains.iterrows():
        z_score = row['z_score']
        ev_row = {'z_score': z_score}
        
        for asset in zscore_gains.columns:
            if asset == 'z_score':
                continue
                
            bin_prediction = row[asset]
            
            if pd.isna(bin_prediction):
                ev_row[asset] = np.nan
                continue
            
            # Get confidence and historical average
            confidence = confidence_lookup.get(asset, 50.0) / 100.0  # Convert to 0-1 scale
            historical_avg = avg_lookup.get(asset, bin_prediction)  # Fallback to bin prediction if no avg
            
            # Method 1: Confidence-weighted blend
            # EV = (Bin_Prediction × Confidence) + (Historical_Average × (1 - Confidence))
            corrected_ev = (bin_prediction * confidence) + (historical_avg * (1 - confidence))
            
            ev_row[asset] = corrected_ev
        
        corrected_ev_data.append(ev_row)
    
    corrected_ev_df = pd.DataFrame(corrected_ev_data)
    
    return corrected_ev_df

def calculate_prediction_confidence_analysis(zscore_gains, confidence_df, asset_averages):
    """
    Analyze how much to trust bin predictions vs historical averages
    """
    analysis_data = []
    
    # Create lookups
    confidence_lookup = dict(zip(confidence_df['asset'], confidence_df['reliability_score']))
    avg_lookup = dict(zip(asset_averages['asset'], asset_averages['mean_gain']))
    
    for asset in zscore_gains.columns:
        if asset == 'z_score':
            continue
            
        confidence_score = confidence_lookup.get(asset, 50.0)
        historical_avg = avg_lookup.get(asset, np.nan)
        
        # Get all valid bin predictions for this asset
        asset_predictions = zscore_gains[asset].dropna()
        
        if len(asset_predictions) == 0:
            continue
        
        # Calculate bin prediction statistics
        bin_mean = asset_predictions.mean()
        bin_std = asset_predictions.std()
        bin_range = asset_predictions.max() - asset_predictions.min()
        
        # Calculate how much bin predictions vary from historical average
        if not np.isnan(historical_avg):
            avg_deviation = abs(bin_mean - historical_avg)
            relative_deviation = avg_deviation / historical_avg * 100 if historical_avg != 0 else 0
        else:
            avg_deviation = np.nan
            relative_deviation = np.nan
        
        analysis_data.append({
            'asset': asset,
            'confidence_score': confidence_score,
            'historical_avg_gain': historical_avg,
            'bin_predictions_mean': bin_mean,
            'bin_predictions_std': bin_std,
            'bin_predictions_range': bin_range,
            'deviation_from_historical': avg_deviation,
            'relative_deviation_pct': relative_deviation,
            'prediction_weight': confidence_score / 100,
            'historical_weight': 1 - (confidence_score / 100)
        })
    
    return pd.DataFrame(analysis_data)

def calculate_alternative_ev_methods(zscore_gains, confidence_df, asset_averages):
    """
    Calculate expected values using different methodologies for comparison
    """
    print("\nCalculating alternative EV methods...")
    
    # Create lookups
    confidence_lookup = dict(zip(confidence_df['asset'], confidence_df['reliability_score']))
    avg_lookup = dict(zip(asset_averages['asset'], asset_averages['mean_gain']))
    median_lookup = dict(zip(asset_averages['asset'], asset_averages['median_gain']))
    
    methods_data = []
    
    for _, row in zscore_gains.iterrows():
        z_score = row['z_score']
        
        methods_row = {
            'z_score': z_score,
            # Method 1: Simple bin prediction (original)
            # Method 2: Confidence-weighted with mean (corrected)
            # Method 3: Confidence-weighted with median
            # Method 4: Conservative approach (min of bin vs historical)
            # Method 5: Optimistic approach (max of bin vs historical)
        }
        
        for asset in zscore_gains.columns:
            if asset == 'z_score':
                continue
                
            bin_prediction = row[asset]
            
            if pd.isna(bin_prediction):
                for method in ['simple', 'corrected_mean', 'corrected_median', 'conservative', 'optimistic']:
                    methods_row[f'{asset}_{method}'] = np.nan
                continue
            
            confidence = confidence_lookup.get(asset, 50.0) / 100.0
            historical_mean = avg_lookup.get(asset, bin_prediction)
            historical_median = median_lookup.get(asset, bin_prediction)
            
            # Method 1: Simple bin prediction
            methods_row[f'{asset}_simple'] = bin_prediction
            
            # Method 2: Confidence-weighted with mean
            methods_row[f'{asset}_corrected_mean'] = (
                bin_prediction * confidence + historical_mean * (1 - confidence)
            )
            
            # Method 3: Confidence-weighted with median
            methods_row[f'{asset}_corrected_median'] = (
                bin_prediction * confidence + historical_median * (1 - confidence)
            )
            
            # Method 4: Conservative (minimum)
            methods_row[f'{asset}_conservative'] = min(bin_prediction, historical_mean)
            
            # Method 5: Optimistic (maximum)
            methods_row[f'{asset}_optimistic'] = max(bin_prediction, historical_mean)
        
        methods_data.append(methods_row)
    
    return pd.DataFrame(methods_data)

def create_ev_comparison_visualization(corrected_ev_df, prediction_analysis, output_dir):
    """
    Create visualizations comparing different EV approaches
    """
    print("Creating EV comparison visualizations...")
    
    # Plot 1: Confidence vs Historical Average Comparison
    fig, ((ax1, ax2), (ax3, ax4)) = plt.subplots(2, 2, figsize=(20, 16))
    
    # Confidence score vs deviation from historical
    valid_analysis = prediction_analysis.dropna(subset=['relative_deviation_pct'])
    
    scatter = ax1.scatter(valid_analysis['confidence_score'], 
                         valid_analysis['relative_deviation_pct'],
                         c=valid_analysis['bin_predictions_range'], 
                         cmap='viridis', s=100, alpha=0.7)
    
    for _, row in valid_analysis.iterrows():
        ax1.annotate(row['asset'], 
                    (row['confidence_score'], row['relative_deviation_pct']),
                    xytext=(5, 5), textcoords='offset points', fontsize=9)
    
    ax1.set_xlabel('Confidence Score (%)')
    ax1.set_ylabel('Bin Deviation from Historical Avg (%)')
    ax1.set_title('Confidence vs Prediction Deviation', fontweight='bold')
    ax1.grid(True, alpha=0.3)
    plt.colorbar(scatter, ax=ax1, label='Bin Prediction Range')
    
    # Historical average vs bin prediction mean
    ax2.scatter(valid_analysis['historical_avg_gain'], 
               valid_analysis['bin_predictions_mean'],
               c=valid_analysis['confidence_score'], 
               cmap='RdYlGn', s=100, alpha=0.7)
    
    # Add diagonal line (perfect agreement)
    min_val = min(valid_analysis['historical_avg_gain'].min(), 
                  valid_analysis['bin_predictions_mean'].min())
    max_val = max(valid_analysis['historical_avg_gain'].max(), 
                  valid_analysis['bin_predictions_mean'].max())
    ax2.plot([min_val, max_val], [min_val, max_val], 'r--', alpha=0.5, label='Perfect Agreement')
    
    for _, row in valid_analysis.iterrows():
        ax2.annotate(row['asset'], 
                    (row['historical_avg_gain'], row['bin_predictions_mean']),
                    xytext=(5, 5), textcoords='offset points', fontsize=9)
    
    ax2.set_xlabel('Historical Average Gain (%)')
    ax2.set_ylabel('Bin Predictions Average (%)')
    ax2.set_title('Historical vs Bin Predictions', fontweight='bold')
    ax2.grid(True, alpha=0.3)
    ax2.legend()
    
    # Expected value comparison at key Z-scores
    key_z_scores = [-2.0, -1.5, -1.0, 0.0, 1.0]
    
    assets_to_plot = corrected_ev_df.columns[1:6]  # Plot first 5 assets
    
    x_pos = np.arange(len(assets_to_plot))
    width = 0.15
    
    for i, z_score in enumerate(key_z_scores):
        if z_score in corrected_ev_df['z_score'].values:
            row = corrected_ev_df[corrected_ev_df['z_score'] == z_score].iloc[0]
            values = [row[asset] for asset in assets_to_plot]
            values = [v if not pd.isna(v) else 0 for v in values]  # Replace NaN with 0 for plotting
            
            ax3.bar(x_pos + i*width, values, width, 
                   label=f'Z={z_score}', alpha=0.8)
    
    ax3.set_xlabel('Assets')
    ax3.set_ylabel('Corrected Expected Value (%)')
    ax3.set_title('Corrected Expected Values by Z-Score', fontweight='bold')
    ax3.set_xticks(x_pos + width * 2)
    ax3.set_xticklabels(assets_to_plot, rotation=45)
    ax3.legend()
    ax3.grid(True, alpha=0.3)
    
    # Confidence weighting impact
    confidence_scores = [prediction_analysis[prediction_analysis['asset'] == asset]['confidence_score'].iloc[0] 
                        if asset in prediction_analysis['asset'].values else 50 
                        for asset in assets_to_plot]
    
    bars = ax4.bar(assets_to_plot, confidence_scores, alpha=0.7, 
                  color=['green' if c >= 70 else 'orange' if c >= 50 else 'red' for c in confidence_scores])
    
    ax4.set_xlabel('Assets')
    ax4.set_ylabel('Confidence Score (%)')
    ax4.set_title('Confidence Scores (Weight Given to Bin Predictions)', fontweight='bold')
    ax4.grid(True, alpha=0.3)
    
    # Add value labels on bars
    for bar, score in zip(bars, confidence_scores):
        ax4.text(bar.get_x() + bar.get_width()/2., bar.get_height() + 1,
                f'{score:.0f}%', ha='center', va='bottom', fontweight='bold')
    
    plt.tight_layout()
    plt.savefig(f"{output_dir}/corrected_ev_analysis.png", dpi=300, bbox_inches='tight')
    plt.close()

def create_method_comparison_chart(alternative_methods_df, output_dir):
    """
    Create chart comparing different EV calculation methods
    """
    print("Creating method comparison chart...")
    
    # Focus on Z-score = -1.5 (good undervalued signal)
    target_z = -1.5
    if target_z not in alternative_methods_df['z_score'].values:
        print(f"Z-score {target_z} not found in data")
        return
    
    row = alternative_methods_df[alternative_methods_df['z_score'] == target_z].iloc[0]
    
    # Extract assets and methods
    methods = ['simple', 'corrected_mean', 'corrected_median', 'conservative', 'optimistic']
    method_labels = ['Simple Bin', 'Corrected (Mean)', 'Corrected (Median)', 'Conservative', 'Optimistic']
    
    # Get first 8 assets for clear visualization
    all_assets = [col.split('_')[0] for col in alternative_methods_df.columns if '_simple' in col][:8]
    
    fig, ax = plt.subplots(figsize=(16, 10))
    
    x = np.arange(len(all_assets))
    width = 0.15
    
    colors = ['blue', 'green', 'orange', 'red', 'purple']
    
    for i, (method, label, color) in enumerate(zip(methods, method_labels, colors)):
        values = []
        for asset in all_assets:
            col_name = f'{asset}_{method}'
            if col_name in row and not pd.isna(row[col_name]):
                values.append(row[col_name])
            else:
                values.append(0)
        
        ax.bar(x + i*width, values, width, label=label, alpha=0.8, color=color)
    
    ax.set_xlabel('Assets')
    ax.set_ylabel('Expected Value (%)')
    ax.set_title(f'EV Method Comparison at Z-Score = {target_z}', fontweight='bold', fontsize=14)
    ax.set_xticks(x + width * 2)
    ax.set_xticklabels(all_assets)
    ax.legend()
    ax.grid(True, alpha=0.3)
    
    plt.tight_layout()
    plt.savefig(f"{output_dir}/method_comparison.png", dpi=300, bbox_inches='tight')
    plt.close()

def save_corrected_ev_results(corrected_ev_df, prediction_analysis, alternative_methods_df, asset_averages, output_dir):
    """
    Save all corrected EV analysis results
    """
    print("Saving corrected EV analysis results...")
    
    # Main corrected expected values
    corrected_ev_df.to_csv(f"{output_dir}/corrected_expected_values.csv", index=False)
    
    # Prediction confidence analysis
    prediction_analysis.to_csv(f"{output_dir}/prediction_confidence_analysis.csv", index=False)
    
    # Alternative methods comparison
    alternative_methods_df.to_csv(f"{output_dir}/alternative_ev_methods.csv", index=False)
    
    # Asset historical averages
    asset_averages.to_csv(f"{output_dir}/asset_historical_averages.csv", index=False)
    
    # Create summary comparison at key Z-scores
    key_z_scores = [-2.0, -1.5, -1.0, -0.5, 0.0, 0.5, 1.0, 1.5, 2.0]
    
    summary_data = []
    for z_score in key_z_scores:
        if z_score in corrected_ev_df['z_score'].values:
            row = corrected_ev_df[corrected_ev_df['z_score'] == z_score].iloc[0]
            
            # Get top 5 assets by corrected EV at this Z-score
            asset_evs = []
            for col in corrected_ev_df.columns:
                if col != 'z_score' and not pd.isna(row[col]):
                    asset_evs.append({'asset': col, 'corrected_ev': row[col]})
            
            if asset_evs:
                asset_evs.sort(key=lambda x: x['corrected_ev'], reverse=True)
                
                summary_data.append({
                    'z_score': z_score,
                    'best_asset': asset_evs[0]['asset'],
                    'best_ev': asset_evs[0]['corrected_ev'],
                    'second_best_asset': asset_evs[1]['asset'] if len(asset_evs) > 1 else '',
                    'second_best_ev': asset_evs[1]['corrected_ev'] if len(asset_evs) > 1 else np.nan,
                    'worst_asset': asset_evs[-1]['asset'],
                    'worst_ev': asset_evs[-1]['corrected_ev'],
                    'ev_range': asset_evs[0]['corrected_ev'] - asset_evs[-1]['corrected_ev'],
                    'num_assets': len(asset_evs)
                })
    
    summary_df = pd.DataFrame(summary_data)
    summary_df.to_csv(f"{output_dir}/zscore_ev_summary.csv", index=False)
    
    return summary_df

def print_corrected_ev_summary(corrected_ev_df, prediction_analysis, summary_df):
    """
    Print comprehensive summary of corrected EV analysis
    """
    print("\n" + "="*80)
    print("CORRECTED EXPECTED VALUE ANALYSIS SUMMARY")
    print("="*80)
    
    print(f"\nMETHODOLOGY CORRECTION:")
    print(f"  OLD (Wrong): EV = Expected_Gain × (Confidence / 100)")
    print(f"  NEW (Correct): EV = (Bin_Prediction × Confidence) + (Historical_Avg × (1 - Confidence))")
    print(f"  Logic: Blend bin prediction with historical average based on confidence")
    
    print(f"\nASSET CONFIDENCE ANALYSIS:")
    print("-" * 60)
    high_conf = prediction_analysis[prediction_analysis['confidence_score'] >= 70]
    med_conf = prediction_analysis[(prediction_analysis['confidence_score'] >= 50) & 
                                  (prediction_analysis['confidence_score'] < 70)]
    low_conf = prediction_analysis[prediction_analysis['confidence_score'] < 50]
    
    print(f"High Confidence Assets (≥70%): {len(high_conf)}")
    if len(high_conf) > 0:
        print(f"  Assets: {', '.join(high_conf['asset'].tolist())}")
        print(f"  Average historical gain: {high_conf['historical_avg_gain'].mean():.1f}%")
        print(f"  Bin predictions heavily weighted")
    
    print(f"\nMedium Confidence Assets (50-70%): {len(med_conf)}")
    if len(med_conf) > 0:
        print(f"  Assets: {', '.join(med_conf['asset'].tolist())}")
        print(f"  Average historical gain: {med_conf['historical_avg_gain'].mean():.1f}%")
        print(f"  Moderate blend of bin predictions and historical")
    
    print(f"\nLow Confidence Assets (<50%): {len(low_conf)}")
    if len(low_conf) > 0:
        print(f"  Assets: {', '.join(low_conf['asset'].tolist())}")
        print(f"  Average historical gain: {low_conf['historical_avg_gain'].mean():.1f}%")
        print(f"  Historical averages heavily weighted")
    
    print(f"\nCORRECTED EV AT KEY Z-SCORES:")
    print("="*60)
    
    key_z_scores = [-2.0, -1.5, -1.0, 0.0, 1.0]
    for z_score in key_z_scores:
        if z_score in summary_df['z_score'].values:
            row = summary_df[summary_df['z_score'] == z_score].iloc[0]
            print(f"\nZ-Score = {z_score:.1f}:")
            print(f"  Best asset: {row['best_asset']:<6} ({row['best_ev']:>6.1f}% EV)")
            if not pd.isna(row['second_best_ev']):
                print(f"  Second best: {row['second_best_asset']:<6} ({row['second_best_ev']:>6.1f}% EV)")
            print(f"  Worst asset: {row['worst_asset']:<6} ({row['worst_ev']:>6.1f}% EV)")
            print(f"  EV Range: {row['ev_range']:>6.1f}%")
    
    print(f"\nKEY INSIGHTS:")
    print("-" * 40)
    
    # Compare high vs low confidence assets
    if len(high_conf) > 0 and len(low_conf) > 0:
        print(f"High confidence assets:")
        print(f"  - Bin predictions more reliable")
        print(f"  - EV closer to bin-specific predictions")
        print(f"  - Better for targeted Z-score strategies")
        
        print(f"\nLow confidence assets:")
        print(f"  - Bin predictions less reliable") 
        print(f"  - EV closer to historical averages")
        print(f"  - Still valuable if high historical returns")
    
    # Find assets where correction made biggest difference
    print(f"\nCORRECTION IMPACT EXAMPLES:")
    print("-" * 40)
    
    target_z = -1.5
    if target_z in corrected_ev_df['z_score'].values:
        row = corrected_ev_df[corrected_ev_df['z_score'] == target_z].iloc[0]
        
        # Compare a few assets to show the correction effect
        example_assets = corrected_ev_df.columns[1:4]  # First 3 assets
        
        for asset in example_assets:
            if not pd.isna(row[asset]):
                asset_info = prediction_analysis[prediction_analysis['asset'] == asset]
                if len(asset_info) > 0:
                    info = asset_info.iloc[0]
                    confidence = info['confidence_score'] / 100
                    historical = info['historical_avg_gain']
                    
                    # Calculate what old method would have given
                    # We need to get the original bin prediction from zscore_gains
                    # For now, estimate it
                    corrected_ev = row[asset]
                    
                    print(f"{asset}: {corrected_ev:.1f}% EV (Confidence: {info['confidence_score']:.0f}%)")

def main():
    print("Corrected Expected Value Analysis")
    print("="*50)
    print("Fixing the EV calculation methodology")
    print("NEW: EV = (Bin_Prediction × Confidence) + (Historical_Avg × (1 - Confidence))")
    
    # Create output directory
    output_dir = "/Users/tim/IWLS-OPTIONS/CORRECTED_EV_ANALYSIS"
    os.makedirs(output_dir, exist_ok=True)
    print(f"\nOutput directory: {output_dir}")
    
    # Load previous analysis results
    zscore_gains, confidence_df, raw_zscore_data = load_previous_analysis_results()
    if zscore_gains is None:
        return
    
    # Calculate asset historical averages
    asset_averages = calculate_asset_historical_averages(raw_zscore_data)
    
    # Calculate corrected expected values
    corrected_ev_df = calculate_corrected_expected_values(zscore_gains, confidence_df, asset_averages)
    
    # Analyze prediction confidence 
    prediction_analysis = calculate_prediction_confidence_analysis(zscore_gains, confidence_df, asset_averages)
    
    # Calculate alternative EV methods for comparison
    alternative_methods_df = calculate_alternative_ev_methods(zscore_gains, confidence_df, asset_averages)
    
    # Save results
    summary_df = save_corrected_ev_results(corrected_ev_df, prediction_analysis, 
                                          alternative_methods_df, asset_averages, output_dir)
    
    # Create visualizations
    create_ev_comparison_visualization(corrected_ev_df, prediction_analysis, output_dir)
    create_method_comparison_chart(alternative_methods_df, output_dir)
    
    # Print summary
    print_corrected_ev_summary(corrected_ev_df, prediction_analysis, summary_df)
    
    print(f"\n" + "="*80)
    print("CORRECTED EXPECTED VALUE ANALYSIS COMPLETE")
    print("="*80)
    print("Files saved to /Users/tim/IWLS-OPTIONS/CORRECTED_EV_ANALYSIS/:")
    print("  - corrected_expected_values.csv (main corrected EV table)")
    print("  - prediction_confidence_analysis.csv (confidence analysis)")
    print("  - alternative_ev_methods.csv (method comparison)")
    print("  - asset_historical_averages.csv (historical gain data)")
    print("  - zscore_ev_summary.csv (best assets by Z-score)")
    print("  - corrected_ev_analysis.png (comprehensive charts)")
    print("  - method_comparison.png (method comparison chart)")
    
    print(f"\nNEXT STEPS:")
    print("  1. Use 'corrected_expected_values.csv' for trading decisions")
    print("  2. Focus on high-confidence assets for Z-score targeting")
    print("  3. Use low-confidence assets for general historical performance")
    print("  4. Combine with entry signals for complete strategy")

if __name__ == "__main__":
    main()