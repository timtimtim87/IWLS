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

def calculate_zscore_forward_gains(results_df, asset_name):
    """
    Calculate Z-scores and forward gains for each data point
    """
    valid_data = results_df.dropna().copy()
    valid_data['price_deviation'] = ((valid_data['price'] / valid_data['trend_line_value']) - 1) * 100
    
    # Calculate rolling Z-scores using expanding window (use all historical data up to each point)
    zscore_data = []
    
    for i in range(len(valid_data)):
        if i < 252:  # Need at least 1 year of data for Z-score calculation
            continue
            
        # Get historical deviations up to this point
        historical_deviations = valid_data['price_deviation'].iloc[:i+1]
        
        # Calculate Z-score for current deviation
        current_deviation = valid_data['price_deviation'].iloc[i]
        mean_deviation = historical_deviations.mean()
        std_deviation = historical_deviations.std()
        
        if std_deviation > 0:
            z_score = (current_deviation - mean_deviation) / std_deviation
        else:
            z_score = 0
        
        # Calculate 1-year forward max gain
        current_price = valid_data['price'].iloc[i]
        future_data = valid_data.iloc[i+1:i+253] if i+252 < len(valid_data) else valid_data.iloc[i+1:]
        
        if len(future_data) >= 200:  # Need at least 8 months of forward data
            max_future_price = future_data['price'].max()
            forward_max_gain = ((max_future_price / current_price) - 1) * 100
            
            zscore_data.append({
                'date': valid_data['date'].iloc[i],
                'asset': asset_name,
                'price_deviation': current_deviation,
                'z_score': z_score,
                'forward_max_gain': forward_max_gain,
                'sample_size': len(historical_deviations)
            })
    
    return pd.DataFrame(zscore_data)

def create_zscore_gain_model(zscore_data, asset_name):
    """
    Create a model to predict expected gains from Z-scores
    """
    if len(zscore_data) < 50:  # Need sufficient data
        return None
    
    # Remove extreme outliers (beyond 3 standard deviations)
    z_threshold = 3.5
    clean_data = zscore_data[
        (zscore_data['z_score'].abs() <= z_threshold) & 
        (zscore_data['forward_max_gain'] <= zscore_data['forward_max_gain'].quantile(0.99))
    ].copy()
    
    if len(clean_data) < 30:
        return None
    
    X = clean_data['z_score'].values.reshape(-1, 1)
    y = clean_data['forward_max_gain'].values
    
    # Try different regression models
    models = {}
    
    # 1. Linear regression
    linear_model = LinearRegression()
    linear_model.fit(X, y)
    y_pred_linear = linear_model.predict(X)
    models['linear'] = {
        'model': linear_model,
        'r2': r2_score(y, y_pred_linear),
        'type': 'linear'
    }
    
    # 2. Polynomial regression (degree 2)
    poly_model = make_pipeline(PolynomialFeatures(2), Ridge(alpha=1.0))
    poly_model.fit(X, y)
    y_pred_poly = poly_model.predict(X)
    models['polynomial'] = {
        'model': poly_model,
        'r2': r2_score(y, y_pred_poly),
        'type': 'polynomial'
    }
    
    # 3. Binned averages (non-parametric)
    z_bins = np.linspace(-3, 3, 13)  # Create 12 bins from -3 to +3
    bin_centers = []
    bin_means = []
    bin_stds = []
    bin_counts = []
    
    for i in range(len(z_bins) - 1):
        bin_mask = (clean_data['z_score'] >= z_bins[i]) & (clean_data['z_score'] < z_bins[i+1])
        bin_data = clean_data[bin_mask]['forward_max_gain']
        
        if len(bin_data) >= 3:  # Need at least 3 points per bin
            bin_centers.append((z_bins[i] + z_bins[i+1]) / 2)
            bin_means.append(bin_data.mean())
            bin_stds.append(bin_data.std())
            bin_counts.append(len(bin_data))
    
    if len(bin_centers) >= 5:  # Need at least 5 bins for interpolation
        # Create interpolation function
        interp_func = interp1d(bin_centers, bin_means, kind='linear', 
                              bounds_error=False, fill_value='extrapolate')
        
        # Test interpolation
        y_pred_interp = interp_func(clean_data['z_score'])
        models['binned'] = {
            'model': interp_func,
            'r2': r2_score(y, y_pred_interp),
            'type': 'binned',
            'bin_data': {
                'centers': bin_centers,
                'means': bin_means,
                'stds': bin_stds,
                'counts': bin_counts
            }
        }
    
    # Choose best model
    best_model_name = max(models.keys(), key=lambda k: models[k]['r2'])
    best_model = models[best_model_name]
    
    return {
        'asset': asset_name,
        'best_model': best_model,
        'all_models': models,
        'data_summary': {
            'n_points': len(clean_data),
            'z_range': (clean_data['z_score'].min(), clean_data['z_score'].max()),
            'gain_range': (clean_data['forward_max_gain'].min(), clean_data['forward_max_gain'].max()),
            'mean_gain': clean_data['forward_max_gain'].mean(),
            'std_gain': clean_data['forward_max_gain'].std()
        },
        'clean_data': clean_data
    }

def predict_expected_gain(model_info, z_score):
    """
    Predict expected gain for a given Z-score using the best model
    """
    if model_info is None:
        return np.nan
    
    best_model = model_info['best_model']
    
    if best_model['type'] == 'linear':
        return best_model['model'].predict([[z_score]])[0]
    elif best_model['type'] == 'polynomial':
        return best_model['model'].predict([[z_score]])[0]
    elif best_model['type'] == 'binned':
        return best_model['model'](z_score)
    else:
        return np.nan

def create_zscore_comparison_table(all_model_info, z_score_range=(-3.0, 3.0, 0.5)):
    """
    Create a comparison table showing expected gains for different Z-scores across all assets
    """
    z_scores = np.arange(z_score_range[0], z_score_range[1] + z_score_range[2], z_score_range[2])
    
    comparison_data = []
    
    for z_score in z_scores:
        row = {'z_score': z_score}
        
        for asset, model_info in all_model_info.items():
            if model_info is not None:
                expected_gain = predict_expected_gain(model_info, z_score)
                row[asset] = expected_gain
            else:
                row[asset] = np.nan
        
        comparison_data.append(row)
    
    return pd.DataFrame(comparison_data)

def load_confidence_scores():
    """
    Load confidence scores from previous analysis
    """
    try:
        confidence_df = pd.read_csv("/Users/tim/IWLS-OPTIONS/CORRELATION_ANALYSIS/enhanced_multi_method_rankings.csv")
        confidence_dict = dict(zip(confidence_df['asset'], confidence_df['reliability_score']))
        return confidence_dict
    except:
        print("Warning: Could not load confidence scores. Using equal weights.")
        return {}

def calculate_expected_values(comparison_df, confidence_scores):
    """
    Calculate expected values combining gains with confidence scores
    """
    ev_data = []
    
    for _, row in comparison_df.iterrows():
        z_score = row['z_score']
        ev_row = {'z_score': z_score}
        
        for asset in comparison_df.columns:
            if asset != 'z_score' and not pd.isna(row[asset]):
                expected_gain = row[asset]
                confidence = confidence_scores.get(asset, 50) / 100  # Default 50% confidence
                expected_value = expected_gain * confidence
                ev_row[asset] = expected_value
            elif asset != 'z_score':
                ev_row[asset] = np.nan
        
        ev_data.append(ev_row)
    
    return pd.DataFrame(ev_data)

def create_visualization(all_model_info, comparison_df, expected_values_df, output_dir):
    """
    Create comprehensive visualizations
    """
    # Plot 1: Individual asset Z-score vs Expected Gain curves
    n_assets = len([m for m in all_model_info.values() if m is not None])
    if n_assets == 0:
        return
    
    cols = 4
    rows = (n_assets + cols - 1) // cols
    
    fig, axes = plt.subplots(rows, cols, figsize=(20, 5*rows))
    if rows == 1:
        axes = axes.reshape(1, -1)
    axes = axes.flatten()
    
    plot_idx = 0
    for asset, model_info in all_model_info.items():
        if model_info is None:
            continue
            
        ax = axes[plot_idx]
        data = model_info['clean_data']
        
        # Scatter plot of actual data
        ax.scatter(data['z_score'], data['forward_max_gain'], alpha=0.5, s=20, color='lightblue')
        
        # Plot model prediction
        z_range = np.linspace(-3, 3, 100)
        predicted_gains = [predict_expected_gain(model_info, z) for z in z_range]
        ax.plot(z_range, predicted_gains, 'r-', linewidth=2, label=f"Model (R²={model_info['best_model']['r2']:.3f})")
        
        ax.set_title(f'{asset}', fontweight='bold')
        ax.set_xlabel('Z-Score')
        ax.set_ylabel('Forward Max Gain (%)')
        ax.grid(True, alpha=0.3)
        ax.legend()
        
        plot_idx += 1
    
    # Hide empty subplots
    for i in range(plot_idx, len(axes)):
        axes[i].set_visible(False)
    
    plt.tight_layout()
    plt.savefig(f"{output_dir}/individual_zscore_models.png", dpi=300, bbox_inches='tight')
    plt.close()
    
    # Plot 2: Cross-asset comparison at key Z-scores
    key_z_scores = [-2.0, -1.5, -1.0, -0.5, 0.0, 0.5, 1.0, 1.5, 2.0]
    
    fig, (ax1, ax2) = plt.subplots(1, 2, figsize=(20, 8))
    
    # Expected gains comparison
    for z_score in key_z_scores:
        if z_score in comparison_df['z_score'].values:
            row = comparison_df[comparison_df['z_score'] == z_score].iloc[0]
            assets = []
            gains = []
            
            for asset in comparison_df.columns:
                if asset != 'z_score' and not pd.isna(row[asset]):
                    assets.append(asset)
                    gains.append(row[asset])
            
            if len(gains) > 0:
                y_pos = np.arange(len(assets))
                bars = ax1.barh(y_pos, gains, alpha=0.7, label=f'Z={z_score}')
        
        ax1.set_xlabel('Expected Gain (%)')
        ax1.set_title('Expected Gains by Z-Score', fontweight='bold')
        ax1.legend(bbox_to_anchor=(1.05, 1), loc='upper left')
    
    # Expected values comparison (gain × confidence)
    for z_score in key_z_scores:
        if z_score in expected_values_df['z_score'].values:
            row = expected_values_df[expected_values_df['z_score'] == z_score].iloc[0]
            assets = []
            ev_values = []
            
            for asset in expected_values_df.columns:
                if asset != 'z_score' and not pd.isna(row[asset]):
                    assets.append(asset)
                    ev_values.append(row[asset])
            
            if len(ev_values) > 0:
                y_pos = np.arange(len(assets))
                bars = ax2.barh(y_pos, ev_values, alpha=0.7, label=f'Z={z_score}')
        
        ax2.set_xlabel('Expected Value (Gain × Confidence)')
        ax2.set_title('Expected Values by Z-Score', fontweight='bold')
        ax2.legend(bbox_to_anchor=(1.05, 1), loc='upper left')
    
    plt.tight_layout()
    plt.savefig(f"{output_dir}/zscore_comparison.png", dpi=300, bbox_inches='tight')
    plt.close()

def save_zscore_analysis_results(all_zscore_data, all_model_info, comparison_df, expected_values_df, output_dir):
    """
    Save all analysis results
    """
    # 1. Save individual asset Z-score data
    combined_zscore_data = pd.concat(all_zscore_data.values(), ignore_index=True)
    combined_zscore_data.to_csv(f"{output_dir}/zscore_forward_gains_data.csv", index=False)
    
    # 2. Save model summary
    model_summary = []
    for asset, model_info in all_model_info.items():
        if model_info is not None:
            summary = {
                'asset': asset,
                'best_model_type': model_info['best_model']['type'],
                'best_model_r2': model_info['best_model']['r2'],
                'n_data_points': model_info['data_summary']['n_points'],
                'z_score_min': model_info['data_summary']['z_range'][0],
                'z_score_max': model_info['data_summary']['z_range'][1],
                'mean_forward_gain': model_info['data_summary']['mean_gain'],
                'std_forward_gain': model_info['data_summary']['std_gain']
            }
        else:
            summary = {
                'asset': asset,
                'best_model_type': 'No Model',
                'best_model_r2': np.nan,
                'n_data_points': 0,
                'z_score_min': np.nan,
                'z_score_max': np.nan,
                'mean_forward_gain': np.nan,
                'std_forward_gain': np.nan
            }
        model_summary.append(summary)
    
    model_summary_df = pd.DataFrame(model_summary)
    model_summary_df.to_csv(f"{output_dir}/zscore_model_summary.csv", index=False)
    
    # 3. Save Z-score comparison table
    comparison_df.to_csv(f"{output_dir}/zscore_expected_gains_comparison.csv", index=False)
    
    # 4. Save expected values table
    expected_values_df.to_csv(f"{output_dir}/zscore_expected_values.csv", index=False)
    
    return model_summary_df

def print_zscore_analysis_summary(model_summary_df, comparison_df, expected_values_df):
    """
    Print comprehensive summary
    """
    print("\n" + "="*80)
    print("Z-SCORE EXPECTED GAINS ANALYSIS SUMMARY")
    print("="*80)
    
    valid_models = model_summary_df[model_summary_df['best_model_type'] != 'No Model']
    
    print(f"\nMODEL QUALITY ASSESSMENT:")
    print(f"  Assets with valid models: {len(valid_models)}/{len(model_summary_df)}")
    print(f"  Average R²: {valid_models['best_model_r2'].mean():.3f}")
    print(f"  Models with R² > 0.1: {len(valid_models[valid_models['best_model_r2'] > 0.1])}")
    print(f"  Models with R² > 0.2: {len(valid_models[valid_models['best_model_r2'] > 0.2])}")
    
    print(f"\nTOP PREDICTIVE MODELS (by R²):")
    print("-" * 60)
    top_models = valid_models.sort_values('best_model_r2', ascending=False).head(10)
    for _, row in top_models.iterrows():
        print(f"{row['asset']:<6}: R²={row['best_model_r2']:.3f}, "
              f"Type={row['best_model_type']:<10}, N={row['n_data_points']:>4}")
    
    # Analyze key Z-score levels
    key_z_scores = [-2.0, -1.0, 0.0, 1.0, 2.0]
    
    print(f"\nEXPECTED GAINS AT KEY Z-SCORE LEVELS:")
    print("="*60)
    
    for z_score in key_z_scores:
        if z_score in comparison_df['z_score'].values:
            row = comparison_df[comparison_df['z_score'] == z_score].iloc[0]
            
            # Get valid gains (non-NaN)
            gains = []
            assets = []
            for asset in comparison_df.columns:
                if asset != 'z_score' and not pd.isna(row[asset]):
                    gains.append(row[asset])
                    assets.append(asset)
            
            if len(gains) > 0:
                print(f"\nZ-Score = {z_score:.1f}:")
                print(f"  Average expected gain: {np.mean(gains):>6.1f}%")
                print(f"  Range: {np.min(gains):>6.1f}% to {np.max(gains):>6.1f}%")
                print(f"  Best asset: {assets[np.argmax(gains)]:<6} ({np.max(gains):>6.1f}%)")
                print(f"  Worst asset: {assets[np.argmin(gains)]:<6} ({np.min(gains):>6.1f}%)")
    
    # Expected values analysis
    print(f"\nEXPECTED VALUES (Gains × Confidence) ANALYSIS:")
    print("="*60)
    
    # Find best expected values at Z = -1.5 (good undervalued level)
    target_z = -1.5
    if target_z in expected_values_df['z_score'].values:
        row = expected_values_df[expected_values_df['z_score'] == target_z].iloc[0]
        
        ev_data = []
        for asset in expected_values_df.columns:
            if asset != 'z_score' and not pd.isna(row[asset]):
                ev_data.append({'asset': asset, 'expected_value': row[asset]})
        
        if ev_data:
            ev_df = pd.DataFrame(ev_data).sort_values('expected_value', ascending=False)
            
            print(f"\nBest Expected Values at Z-Score = {target_z}:")
            print("-" * 40)
            for _, row in ev_df.head(10).iterrows():
                print(f"{row['asset']:<6}: {row['expected_value']:>6.1f}")
    
    print(f"\nSTRATEGY RECOMMENDATIONS:")
    print("="*40)
    
    high_quality_models = valid_models[valid_models['best_model_r2'] > 0.15]
    if len(high_quality_models) > 0:
        print(f"HIGH CONFIDENCE PREDICTIONS:")
        print(f"  Focus on {len(high_quality_models)} assets with R² > 0.15")
        print(f"  Assets: {', '.join(high_quality_models['asset'].head(8).tolist())}")
        
        print(f"\nTRADING SIGNALS:")
        print(f"  Z-Score < -1.5: Strong undervalued signal")
        print(f"  Z-Score < -1.0: Moderate undervalued signal") 
        print(f"  Z-Score > +1.0: Consider avoiding/shorting")
        
        print(f"\nRISK MANAGEMENT:")
        print(f"  Use Expected Values to size positions")
        print(f"  Higher EV = larger position size")
        print(f"  Combine with correlation confidence from previous analysis")
    
    else:
        print(f"WARNING: Few assets show strong Z-score predictive power")
        print(f"Consider revising methodology or combining with other signals")

def main():
    print("Z-Score Expected Gains Analysis")
    print("="*50)
    print("Analyzing expected gains as a function of Z-scores")
    print("Goal: Quantify absolute gain expectations across assets")
    
    # Create output directory
    output_dir = "/Users/tim/IWLS-OPTIONS/ZSCORE_ANALYSIS"
    os.makedirs(output_dir, exist_ok=True)
    print(f"\nOutput directory: {output_dir}")
    
    # Load IWLS results
    all_results = load_all_iwls_results()
    if not all_results:
        return
    
    print(f"\nLoaded {len(all_results)} assets")
    
    # Calculate Z-scores and forward gains for each asset
    print("\nCalculating Z-scores and forward gains...")
    all_zscore_data = {}
    all_model_info = {}
    
    for asset_name, results_df in all_results.items():
        print(f"  Processing {asset_name}...")
        
        # Calculate Z-score data
        zscore_data = calculate_zscore_forward_gains(results_df, asset_name)
        all_zscore_data[asset_name] = zscore_data
        
        # Create prediction model
        model_info = create_zscore_gain_model(zscore_data, asset_name)
        all_model_info[asset_name] = model_info
        
        if model_info:
            print(f"    Model R²: {model_info['best_model']['r2']:.3f}")
        else:
            print(f"    Insufficient data for modeling")
    
    # Create comparison table
    print("\nCreating Z-score comparison table...")
    comparison_df = create_zscore_comparison_table(all_model_info)
    
    # Load confidence scores and calculate expected values
    print("Loading confidence scores and calculating expected values...")
    confidence_scores = load_confidence_scores()
    expected_values_df = calculate_expected_values(comparison_df, confidence_scores)
    
    # Save results
    print("Saving analysis results...")
    model_summary_df = save_zscore_analysis_results(
        all_zscore_data, all_model_info, comparison_df, expected_values_df, output_dir
    )
    
    # Create visualizations
    print("Creating visualizations...")
    create_visualization(all_model_info, comparison_df, expected_values_df, output_dir)
    
    # Print summary
    print_zscore_analysis_summary(model_summary_df, comparison_df, expected_values_df)
    
    print(f"\n" + "="*80)
    print("Z-SCORE EXPECTED GAINS ANALYSIS COMPLETE")
    print("="*80)
    print("Files saved:")
    print("  - zscore_forward_gains_data.csv (raw Z-score and gain data)")
    print("  - zscore_model_summary.csv (model quality assessment)")
    print("  - zscore_expected_gains_comparison.csv (expected gains by Z-score)")
    print("  - zscore_expected_values.csv (gains × confidence)")
    print("  - individual_zscore_models.png (model visualizations)")
    print("  - zscore_comparison.png (cross-asset comparison)")

if __name__ == "__main__":
    main()