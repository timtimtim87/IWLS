def calculate_forward_returns_single_asset(asset_name, iwls_data, zscore_data, forward_days=365):
    """
    Calculate 365-day forward maximum gains for a single asset
    """
    print(f"  Analyzing {asset_name}...")
    
    forward_returns = []
    
    # Use IWLS data as base, merge with Z-score data
    data = iwls_data.copy()
    
    # Merge with Z-score data if available
    if zscore_data is not None:
        data = data.merge(zscore_data[['date', 'z_score']], on='date', how='left')
    else:
        data['z_score'] = np.nan
    
    # Only use data where we have valid IWLS calculations (after the 1500-day lookback)
    valid_data = data.dropna(subset=['price_deviation', 'trend_line_value']).copy()
    
    print(f"    Valid IWLS data points: {len(valid_data)}")
    print(f"    Date range: {valid_data['date'].min()} to {valid_data['date'].max()}")
    
    # Calculate forward returns for each entry point
    for i in range(len(valid_data) - forward_days):
        current_row = valid_data.iloc[i]
        current_price = current_row['price']
        current_deviation = current_row['price_deviation']
        current_zscore = current_row.get('z_score', np.nan)
        
        # Get future data for next 365 days
        entry_date = current_row['date']
        end_date = entry_date + timedelta(days=forward_days)
        
        # Get future prices within the forward window
        future_mask = (valid_data['date'] > entry_date) & (valid_data['date'] <= end_date)
        future_data = valid_data[future_mask]
        
        if len(future_data) >= forward_days * 0.5:  # Need at least 50% of trading days
            # Calculate various forward return metrics
            max_price = future_data['price'].max()
            min_price = future_data['price'].min()
            final_price = future_data['price'].iloc[-1]
            
            # Calculate returns
            max_gain = ((max_price / current_price) - 1) * 100
            max_loss = ((min_price / current_price) - 1) * 100
            final_return = ((final_price / current_price) - 1) * 100
            
            # Calculate time to max gain
            max_idx = future_data['price'].idxmax()
            max_gain_date = future_data.loc[max_idx, 'date']
            days_to_max = (max_gain_date - entry_date).days
            
            # Categorize into bins
            deviation_bin = get_deviation_bin(current_deviation)
            zscore_bin = get_zscore_bin(current_zscore) if not pd.isna(current_zscore) else 'No Z-Score'
            
            forward_returns.append({
                'asset': asset_name,
                'entry_date': entry_date,
                'entry_price': current_price,
                'price_deviation': current_deviation,
                'z_score': current_zscore,
                'deviation_bin': deviation_bin,
                'zscore_bin': zscore_bin,
                'max_gain_365d': max_gain,
                'max_loss_365d': max_loss,
                'final_return_365d': final_return,
                'days_to_max_gain': days_to_max,
                'max_price': max_price,
                'min_price': min_price,
                'final_price': final_price,
                'volatility_365d': future_data['price'].pct_change().std() * np.sqrt(252) * 100
            })
    
    print(f"    Generated {len(forward_returns)} forward return samples")
    return pd.DataFrame(forward_returns)import pandas as pd
import numpy as np
import matplotlib.pyplot as plt
import seaborn as sns
from datetime import datetime, timedelta
import warnings
import os
import glob
from collections import defaultdict
from scipy import stats
import matplotlib.patches as mpatches
warnings.filterwarnings('ignore')

def load_iwls_results():
    """
    Load all IWLS results from the V2 analysis
    """
    base_dir = "/Users/tim/IWLS-OPTIONS/IWLS_ANALYSIS_V2"
    
    if not os.path.exists(base_dir):
        print("❌ IWLS_ANALYSIS_V2 directory not found. Run the IWLS analysis first.")
        return {}
    
    all_results = {}
    asset_dirs = [d for d in os.listdir(base_dir) if os.path.isdir(os.path.join(base_dir, d))]
    
    print(f"Loading IWLS results for {len(asset_dirs)} assets...")
    
    for asset_dir in asset_dirs:
        asset_path = os.path.join(base_dir, asset_dir)
        iwls_file = os.path.join(asset_path, f"{asset_dir}_iwls_results.csv")
        zscore_file = os.path.join(asset_path, f"{asset_dir}_zscore_analysis.csv")
        
        if os.path.exists(iwls_file):
            try:
                # Load main IWLS results
                df = pd.read_csv(iwls_file)
                df['date'] = pd.to_datetime(df['date'])
                df = df.dropna()
                
                # Load Z-score analysis if available
                zscore_df = None
                if os.path.exists(zscore_file):
                    zscore_df = pd.read_csv(zscore_file)
                    zscore_df['date'] = pd.to_datetime(zscore_df['date'])
                
                all_results[asset_dir] = {
                    'iwls_data': df,
                    'zscore_data': zscore_df
                }
                
                print(f"  ✅ {asset_dir}: {len(df)} data points")
                
            except Exception as e:
                print(f"  ❌ Error loading {asset_dir}: {e}")
    
    print(f"Successfully loaded {len(all_results)} assets")
    return all_results

def get_deviation_bin(deviation):
    """
    Categorize deviation into 5% bins
    """
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

def get_zscore_bin(z_score):
    """
    Categorize Z-score into bins
    """
    if z_score >= 3:
        return ">+3σ"
    elif z_score >= 2.5:
        return "+2.5σ to +3σ"
    elif z_score >= 2:
        return "+2σ to +2.5σ"
    elif z_score >= 1.5:
        return "+1.5σ to +2σ"
    elif z_score >= 1:
        return "+1σ to +1.5σ"
    elif z_score >= 0.5:
        return "+0.5σ to +1σ"
    elif z_score >= -0.5:
        return "-0.5σ to +0.5σ"
    elif z_score >= -1:
        return "-1σ to -0.5σ"
    elif z_score >= -1.5:
        return "-1.5σ to -1σ"
    elif z_score >= -2:
        return "-2σ to -1.5σ"
    elif z_score >= -2.5:
        return "-2.5σ to -2σ"
    elif z_score >= -3:
        return "-3σ to -2.5σ"
    else:
        return "<-3σ"

def calculate_forward_returns_single_asset(asset_name, iwls_data, zscore_data, forward_days=365):
    """
    Calculate 365-day forward maximum gains for a single asset
    """
    print(f"  Analyzing {asset_name}...")
    
    forward_returns = []
    
    # Use IWLS data as base, merge with Z-score data
    data = iwls_data.copy()
    
    # Merge with Z-score data if available
    if zscore_data is not None:
        data = data.merge(zscore_data[['date', 'z_score']], on='date', how='left')
    else:
        data['z_score'] = np.nan
    
    # Calculate forward returns for each entry point
    for i in range(len(data) - forward_days):
        current_row = data.iloc[i]
        current_price = current_row['price']
        current_deviation = current_row['price_deviation']
        current_zscore = current_row.get('z_score', np.nan)
        
        # Skip if essential data is missing
        if pd.isna(current_price) or pd.isna(current_deviation):
            continue
        
        # Get future prices for next 365 calendar days
        entry_date = current_row['date']
        end_date = entry_date + timedelta(days=forward_days)
        
        future_data = data[(data['date'] > entry_date) & (data['date'] <= end_date)]
        
        if len(future_data) >= forward_days * 0.7:  # Need at least 70% of days
            # Calculate various forward return metrics
            max_price = future_data['price'].max()
            min_price = future_data['price'].min()
            final_price = future_data['price'].iloc[-1]
            
            # Calculate returns
            max_gain = ((max_price / current_price) - 1) * 100
            max_loss = ((min_price / current_price) - 1) * 100
            final_return = ((final_price / current_price) - 1) * 100
            
            # Calculate time to max gain
            max_idx = future_data['price'].idxmax()
            max_gain_date = future_data.loc[max_idx, 'date']
            days_to_max = (max_gain_date - entry_date).days
            
            # Categorize into bins
            deviation_bin = get_deviation_bin(current_deviation)
            zscore_bin = get_zscore_bin(current_zscore) if not pd.isna(current_zscore) else 'No Z-Score'
            
            forward_returns.append({
                'asset': asset_name,
                'entry_date': entry_date,
                'entry_price': current_price,
                'price_deviation': current_deviation,
                'z_score': current_zscore,
                'deviation_bin': deviation_bin,
                'zscore_bin': zscore_bin,
                'max_gain_365d': max_gain,
                'max_loss_365d': max_loss,
                'final_return_365d': final_return,
                'days_to_max_gain': days_to_max,
                'max_price': max_price,
                'min_price': min_price,
                'final_price': final_price,
                'volatility_365d': future_data['price'].pct_change().std() * np.sqrt(252) * 100
            })
    
    return pd.DataFrame(forward_returns)

def analyze_all_assets_forward_returns(all_results):
    """
    Analyze forward returns for all assets
    """
    print("\nCalculating 365-day forward returns for all assets...")
    
    all_forward_returns = []
    
    for asset_name, asset_data in all_results.items():
        iwls_data = asset_data['iwls_data']
        zscore_data = asset_data['zscore_data']
        
        asset_forward_returns = calculate_forward_returns_single_asset(
            asset_name, iwls_data, zscore_data
        )
        
        if len(asset_forward_returns) > 0:
            all_forward_returns.append(asset_forward_returns)
            print(f"    {asset_name}: {len(asset_forward_returns)} forward return samples")
    
    if all_forward_returns:
        combined_df = pd.concat(all_forward_returns, ignore_index=True)
        print(f"\nTotal forward return samples: {len(combined_df):,}")
        return combined_df
    else:
        print("No forward return data generated!")
        return pd.DataFrame()

def analyze_returns_by_deviation_bins(forward_returns_df):
    """
    Analyze returns by 5% deviation bins
    """
    print("\nAnalyzing returns by deviation bins...")
    
    # Define bin order for proper sorting
    bin_order = [">+50%", "+45% to +50%", "+40% to +45%", "+35% to +40%", "+30% to +35%", 
                 "+25% to +30%", "+20% to +25%", "+15% to +20%", "+10% to +15%", "+5% to +10%", 
                 "-5% to +5%", "-10% to -5%", "-15% to -10%", "-20% to -15%", "-25% to -20%", 
                 "-30% to -25%", "-35% to -30%", "-40% to -35%", "-45% to -40%", "-50% to -45%", "<-50%"]
    
    deviation_analysis = []
    
    for bin_name in bin_order:
        bin_data = forward_returns_df[forward_returns_df['deviation_bin'] == bin_name]
        
        if len(bin_data) >= 10:  # Need minimum samples
            # Calculate comprehensive statistics
            max_gains = bin_data['max_gain_365d']
            final_returns = bin_data['final_return_365d']
            
            analysis = {
                'deviation_bin': bin_name,
                'sample_count': len(bin_data),
                'assets_count': bin_data['asset'].nunique(),
                
                # Max gain statistics
                'avg_max_gain': max_gains.mean(),
                'median_max_gain': max_gains.median(),
                'std_max_gain': max_gains.std(),
                'min_max_gain': max_gains.min(),
                'max_max_gain': max_gains.max(),
                'q25_max_gain': max_gains.quantile(0.25),
                'q75_max_gain': max_gains.quantile(0.75),
                
                # Final return statistics
                'avg_final_return': final_returns.mean(),
                'median_final_return': final_returns.median(),
                'std_final_return': final_returns.std(),
                
                # Success rates
                'success_rate_positive': (final_returns > 0).mean() * 100,
                'success_rate_10pct': (max_gains > 10).mean() * 100,
                'success_rate_25pct': (max_gains > 25).mean() * 100,
                'success_rate_50pct': (max_gains > 50).mean() * 100,
                'success_rate_100pct': (max_gains > 100).mean() * 100,
                
                # Risk metrics
                'avg_max_loss': bin_data['max_loss_365d'].mean(),
                'worst_max_loss': bin_data['max_loss_365d'].min(),
                'avg_days_to_max': bin_data['days_to_max_gain'].mean(),
                'avg_volatility': bin_data['volatility_365d'].mean(),
                
                # Risk-adjusted returns
                'sharpe_ratio': final_returns.mean() / final_returns.std() if final_returns.std() > 0 else 0,
                'gain_to_loss_ratio': max_gains.mean() / abs(bin_data['max_loss_365d'].mean()) if bin_data['max_loss_365d'].mean() < 0 else np.inf
            }
            
            deviation_analysis.append(analysis)
    
    return pd.DataFrame(deviation_analysis)

def analyze_returns_by_zscore_bins(forward_returns_df):
    """
    Analyze returns by Z-score bins
    """
    print("\nAnalyzing returns by Z-score bins...")
    
    # Filter out rows without Z-scores
    zscore_data = forward_returns_df[forward_returns_df['zscore_bin'] != 'No Z-Score'].copy()
    
    if len(zscore_data) == 0:
        print("No Z-score data available")
        return pd.DataFrame()
    
    # Define Z-score bin order
    zscore_bin_order = [">+3σ", "+2.5σ to +3σ", "+2σ to +2.5σ", "+1.5σ to +2σ", "+1σ to +1.5σ", 
                        "+0.5σ to +1σ", "-0.5σ to +0.5σ", "-1σ to -0.5σ", "-1.5σ to -1σ", 
                        "-2σ to -1.5σ", "-2.5σ to -2σ", "-3σ to -2.5σ", "<-3σ"]
    
    zscore_analysis = []
    
    for bin_name in zscore_bin_order:
        bin_data = zscore_data[zscore_data['zscore_bin'] == bin_name]
        
        if len(bin_data) >= 10:  # Need minimum samples
            max_gains = bin_data['max_gain_365d']
            final_returns = bin_data['final_return_365d']
            
            analysis = {
                'zscore_bin': bin_name,
                'sample_count': len(bin_data),
                'assets_count': bin_data['asset'].nunique(),
                'avg_zscore': bin_data['z_score'].mean(),
                
                # Max gain statistics
                'avg_max_gain': max_gains.mean(),
                'median_max_gain': max_gains.median(),
                'std_max_gain': max_gains.std(),
                'q25_max_gain': max_gains.quantile(0.25),
                'q75_max_gain': max_gains.quantile(0.75),
                
                # Final return statistics
                'avg_final_return': final_returns.mean(),
                'median_final_return': final_returns.median(),
                
                # Success rates
                'success_rate_positive': (final_returns > 0).mean() * 100,
                'success_rate_25pct': (max_gains > 25).mean() * 100,
                'success_rate_50pct': (max_gains > 50).mean() * 100,
                'success_rate_100pct': (max_gains > 100).mean() * 100,
                
                # Risk metrics
                'avg_max_loss': bin_data['max_loss_365d'].mean(),
                'avg_days_to_max': bin_data['days_to_max_gain'].mean(),
                'sharpe_ratio': final_returns.mean() / final_returns.std() if final_returns.std() > 0 else 0
            }
            
            zscore_analysis.append(analysis)
    
    return pd.DataFrame(zscore_analysis)

def create_comprehensive_visualizations(forward_returns_df, deviation_analysis, zscore_analysis, output_dir):
    """
    Create comprehensive visualizations of forward returns analysis
    """
    print("\nCreating comprehensive visualizations...")
    
    # Set up the plotting style
    plt.style.use('default')
    sns.set_palette("husl")
    
    # Figure 1: Deviation Bins Analysis (6 subplots)
    fig1 = plt.figure(figsize=(24, 20))
    
    # Plot 1: Average Max Gains by Deviation Bin
    ax1 = plt.subplot(3, 2, 1)
    if len(deviation_analysis) > 0:
        bars = ax1.bar(range(len(deviation_analysis)), deviation_analysis['avg_max_gain'], 
                      color=['darkred' if '+' in bin_name and bin_name != '-5% to +5%' 
                            else 'darkgreen' if '-' in bin_name and bin_name != '-5% to +5%' 
                            else 'gray' for bin_name in deviation_analysis['deviation_bin']], alpha=0.8)
        
        ax1.set_xticks(range(len(deviation_analysis)))
        ax1.set_xticklabels(deviation_analysis['deviation_bin'], rotation=45, ha='right')
        ax1.set_ylabel('Average Max Gain (%)')
        ax1.set_title('365-Day Average Maximum Gains by Deviation Bin', fontweight='bold')
        ax1.grid(True, alpha=0.3)
        
        # Add value labels
        for i, (bar, value, count) in enumerate(zip(bars, deviation_analysis['avg_max_gain'], 
                                                   deviation_analysis['sample_count'])):
            ax1.text(bar.get_x() + bar.get_width()/2., bar.get_height() + max(deviation_analysis['avg_max_gain'])*0.01,
                    f'{value:.1f}%\n(n={count})', ha='center', va='bottom', fontsize=9, fontweight='bold')
    
    # Plot 2: Success Rate Distribution
    ax2 = plt.subplot(3, 2, 2)
    if len(deviation_analysis) > 0:
        width = 0.2
        x = np.arange(len(deviation_analysis))
        
        ax2.bar(x - width*1.5, deviation_analysis['success_rate_25pct'], width, label='25%+ Gains', alpha=0.8, color='lightgreen')
        ax2.bar(x - width*0.5, deviation_analysis['success_rate_50pct'], width, label='50%+ Gains', alpha=0.8, color='green')
        ax2.bar(x + width*0.5, deviation_analysis['success_rate_100pct'], width, label='100%+ Gains', alpha=0.8, color='darkgreen')
        
        ax2.set_xticks(x)
        ax2.set_xticklabels(deviation_analysis['deviation_bin'], rotation=45, ha='right')
        ax2.set_ylabel('Success Rate (%)')
        ax2.set_title('Success Rates by Deviation Bin', fontweight='bold')
        ax2.legend()
        ax2.grid(True, alpha=0.3)
    
    # Plot 3: Box plot of max gains distribution
    ax3 = plt.subplot(3, 2, 3)
    if len(forward_returns_df) > 0:
        # Sample bins with enough data for box plot
        sample_bins = deviation_analysis[deviation_analysis['sample_count'] >= 50]['deviation_bin'].head(8)
        box_data = []
        box_labels = []
        
        for bin_name in sample_bins:
            bin_data = forward_returns_df[forward_returns_df['deviation_bin'] == bin_name]['max_gain_365d']
            if len(bin_data) > 0:
                box_data.append(bin_data.values)
                box_labels.append(bin_name)
        
        if box_data:
            bp = ax3.boxplot(box_data, labels=box_labels, patch_artist=True)
            for patch in bp['boxes']:
                patch.set_facecolor('lightblue')
                patch.set_alpha(0.7)
            
            ax3.set_xticklabels(box_labels, rotation=45, ha='right')
            ax3.set_ylabel('Max Gain Distribution (%)')
            ax3.set_title('Distribution of Max Gains by Deviation Bin', fontweight='bold')
            ax3.grid(True, alpha=0.3)
    
    # Plot 4: Risk vs Return Scatter
    ax4 = plt.subplot(3, 2, 4)
    if len(deviation_analysis) > 0:
        scatter = ax4.scatter(deviation_analysis['avg_max_loss'], deviation_analysis['avg_max_gain'],
                             c=deviation_analysis['sample_count'], cmap='viridis', s=100, alpha=0.7)
        
        for i, bin_name in enumerate(deviation_analysis['deviation_bin']):
            ax4.annotate(bin_name, (deviation_analysis['avg_max_loss'].iloc[i], 
                                   deviation_analysis['avg_max_gain'].iloc[i]),
                        xytext=(5, 5), textcoords='offset points', fontsize=8)
        
        ax4.set_xlabel('Average Max Loss (%)')
        ax4.set_ylabel('Average Max Gain (%)')
        ax4.set_title('Risk vs Return by Deviation Bin', fontweight='bold')
        plt.colorbar(scatter, ax=ax4, label='Sample Count')
        ax4.grid(True, alpha=0.3)
    
    # Plot 5: Time to Maximum Gain
    ax5 = plt.subplot(3, 2, 5)
    if len(deviation_analysis) > 0:
        bars = ax5.bar(range(len(deviation_analysis)), deviation_analysis['avg_days_to_max'], 
                      alpha=0.8, color='orange')
        ax5.set_xticks(range(len(deviation_analysis)))
        ax5.set_xticklabels(deviation_analysis['deviation_bin'], rotation=45, ha='right')
        ax5.set_ylabel('Average Days to Max Gain')
        ax5.set_title('Time to Maximum Gain by Deviation Bin', fontweight='bold')
        ax5.grid(True, alpha=0.3)
        
        # Add value labels
        for bar, value in zip(bars, deviation_analysis['avg_days_to_max']):
            ax5.text(bar.get_x() + bar.get_width()/2., bar.get_height() + max(deviation_analysis['avg_days_to_max'])*0.01,
                    f'{value:.0f}', ha='center', va='bottom', fontweight='bold')
    
    # Plot 6: Sharpe Ratio by Deviation Bin
    ax6 = plt.subplot(3, 2, 6)
    if len(deviation_analysis) > 0:
        colors = ['green' if sr > 0.5 else 'orange' if sr > 0 else 'red' for sr in deviation_analysis['sharpe_ratio']]
        bars = ax6.bar(range(len(deviation_analysis)), deviation_analysis['sharpe_ratio'], 
                      color=colors, alpha=0.8)
        ax6.set_xticks(range(len(deviation_analysis)))
        ax6.set_xticklabels(deviation_analysis['deviation_bin'], rotation=45, ha='right')
        ax6.set_ylabel('Sharpe Ratio')
        ax6.set_title('Risk-Adjusted Returns (Sharpe Ratio) by Deviation Bin', fontweight='bold')
        ax6.axhline(y=0, color='black', linestyle='-', alpha=0.5)
        ax6.grid(True, alpha=0.3)
        
        # Add value labels
        for bar, value in zip(bars, deviation_analysis['sharpe_ratio']):
            ax6.text(bar.get_x() + bar.get_width()/2., 
                    bar.get_height() + (0.1 if value >= 0 else -0.15),
                    f'{value:.2f}', ha='center', 
                    va='bottom' if value >= 0 else 'top', fontweight='bold')
    
    plt.tight_layout()
    plt.savefig(f"{output_dir}/deviation_bins_comprehensive_analysis.png", dpi=300, bbox_inches='tight')
    plt.close()
    
    # Figure 2: Z-Score Analysis (if data available)
    if len(zscore_analysis) > 0:
        fig2 = plt.figure(figsize=(20, 12))
        
        # Z-Score Average Max Gains
        ax1 = plt.subplot(2, 3, 1)
        colors = ['darkred' if '+' in bin_name else 'darkgreen' if '-' in bin_name else 'gray' 
                 for bin_name in zscore_analysis['zscore_bin']]
        bars = ax1.bar(range(len(zscore_analysis)), zscore_analysis['avg_max_gain'], 
                      color=colors, alpha=0.8)
        ax1.set_xticks(range(len(zscore_analysis)))
        ax1.set_xticklabels(zscore_analysis['zscore_bin'], rotation=45, ha='right')
        ax1.set_ylabel('Average Max Gain (%)')
        ax1.set_title('365-Day Average Maximum Gains by Z-Score Bin', fontweight='bold')
        ax1.grid(True, alpha=0.3)
        
        # Add value labels
        for i, (bar, value, count) in enumerate(zip(bars, zscore_analysis['avg_max_gain'], 
                                                   zscore_analysis['sample_count'])):
            ax1.text(bar.get_x() + bar.get_width()/2., bar.get_height() + max(zscore_analysis['avg_max_gain'])*0.01,
                    f'{value:.1f}%\n(n={count})', ha='center', va='bottom', fontsize=9, fontweight='bold')
        
        # Z-Score Success Rates
        ax2 = plt.subplot(2, 3, 2)
        width = 0.25
        x = np.arange(len(zscore_analysis))
        
        ax2.bar(x - width, zscore_analysis['success_rate_25pct'], width, label='25%+ Gains', alpha=0.8, color='lightgreen')
        ax2.bar(x, zscore_analysis['success_rate_50pct'], width, label='50%+ Gains', alpha=0.8, color='green')
        ax2.bar(x + width, zscore_analysis['success_rate_100pct'], width, label='100%+ Gains', alpha=0.8, color='darkgreen')
        
        ax2.set_xticks(x)
        ax2.set_xticklabels(zscore_analysis['zscore_bin'], rotation=45, ha='right')
        ax2.set_ylabel('Success Rate (%)')
        ax2.set_title('Success Rates by Z-Score Bin', fontweight='bold')
        ax2.legend()
        ax2.grid(True, alpha=0.3)
        
        # Z-Score vs Average Deviation
        ax3 = plt.subplot(2, 3, 3)
        ax3.scatter(zscore_analysis['avg_zscore'], zscore_analysis['avg_max_gain'], 
                   s=zscore_analysis['sample_count']/10, alpha=0.7, color='blue')
        
        for i, bin_name in enumerate(zscore_analysis['zscore_bin']):
            ax3.annotate(bin_name, (zscore_analysis['avg_zscore'].iloc[i], 
                                   zscore_analysis['avg_max_gain'].iloc[i]),
                        xytext=(5, 5), textcoords='offset points', fontsize=8)
        
        ax3.set_xlabel('Average Z-Score')
        ax3.set_ylabel('Average Max Gain (%)')
        ax3.set_title('Z-Score vs Average Max Gain', fontweight='bold')
        ax3.grid(True, alpha=0.3)
        
        # Z-Score Sharpe Ratios
        ax4 = plt.subplot(2, 3, 4)
        colors = ['green' if sr > 0.5 else 'orange' if sr > 0 else 'red' for sr in zscore_analysis['sharpe_ratio']]
        bars = ax4.bar(range(len(zscore_analysis)), zscore_analysis['sharpe_ratio'], 
                      color=colors, alpha=0.8)
        ax4.set_xticks(range(len(zscore_analysis)))
        ax4.set_xticklabels(zscore_analysis['zscore_bin'], rotation=45, ha='right')
        ax4.set_ylabel('Sharpe Ratio')
        ax4.set_title('Risk-Adjusted Returns by Z-Score Bin', fontweight='bold')
        ax4.axhline(y=0, color='black', linestyle='-', alpha=0.5)
        ax4.grid(True, alpha=0.3)
        
        # Days to Max Gain
        ax5 = plt.subplot(2, 3, 5)
        bars = ax5.bar(range(len(zscore_analysis)), zscore_analysis['avg_days_to_max'], 
                      alpha=0.8, color='orange')
        ax5.set_xticks(range(len(zscore_analysis)))
        ax5.set_xticklabels(zscore_analysis['zscore_bin'], rotation=45, ha='right')
        ax5.set_ylabel('Average Days to Max Gain')
        ax5.set_title('Time to Maximum Gain by Z-Score Bin', fontweight='bold')
        ax5.grid(True, alpha=0.3)
        
        # Sample size distribution
        ax6 = plt.subplot(2, 3, 6)
        bars = ax6.bar(range(len(zscore_analysis)), zscore_analysis['sample_count'], 
                      alpha=0.8, color='steelblue')
        ax6.set_xticks(range(len(zscore_analysis)))
        ax6.set_xticklabels(zscore_analysis['zscore_bin'], rotation=45, ha='right')
        ax6.set_ylabel('Sample Count')
        ax6.set_title('Sample Size by Z-Score Bin', fontweight='bold')
        ax6.grid(True, alpha=0.3)
        
        # Add value labels
        for bar, value in zip(bars, zscore_analysis['sample_count']):
            ax6.text(bar.get_x() + bar.get_width()/2., bar.get_height() + max(zscore_analysis['sample_count'])*0.01,
                    f'{value}', ha='center', va='bottom', fontweight='bold')
        
        plt.tight_layout()
        plt.savefig(f"{output_dir}/zscore_bins_comprehensive_analysis.png", dpi=300, bbox_inches='tight')
        plt.close()
    
    # Figure 3: Cross-Asset Heatmaps
    create_asset_heatmaps(forward_returns_df, output_dir)
    
    print("  ✅ All visualizations created")

def create_asset_heatmaps(forward_returns_df, output_dir):
    """
    Create heatmaps showing performance across assets and bins
    """
    print("  Creating asset performance heatmaps...")
    
    # Get top assets by sample count
    asset_counts = forward_returns_df['asset'].value_counts()
    top_assets = asset_counts.head(20).index.tolist()
    
    # Create deviation bin heatmap
    fig, (ax1, ax2) = plt.subplots(2, 1, figsize=(20, 16))
    
    # Deviation bins heatmap
    deviation_pivot = forward_returns_df[forward_returns_df['asset'].isin(top_assets)].pivot_table(
        index='asset', 
        columns='deviation_bin', 
        values='max_gain_365d', 
        aggfunc='mean'
    )
    
    # Reorder columns by deviation level
    bin_order = [">+50%", "+45% to +50%", "+40% to +45%", "+35% to +40%", "+30% to +35%", 
                 "+25% to +30%", "+20% to +25%", "+15% to +20%", "+10% to +15%", "+5% to +10%", 
                 "-5% to +5%", "-10% to -5%", "-15% to -10%", "-20% to -15%", "-25% to -20%", 
                 "-30% to -25%", "-35% to -30%", "-40% to -35%", "-45% to -40%", "-50% to -45%", "<-50%"]
    
    available_bins = [bin_name for bin_name in bin_order if bin_name in deviation_pivot.columns]
    deviation_pivot = deviation_pivot[available_bins]
    
    # Create heatmap
    sns.heatmap(deviation_pivot, annot=True, fmt='.1f', cmap='RdYlGn', center=0, 
                ax=ax1, cbar_kws={'label': 'Average Max Gain (%)'})
    ax1.set_title('Average 365-Day Max Gains by Asset and Deviation Bin', fontweight='bold', fontsize=14)
    ax1.set_xlabel('Deviation Bin')
    ax1.set_ylabel('Asset')
    
    # Z-score bins heatmap (if data available)
    zscore_data = forward_returns_df[
        (forward_returns_df['asset'].isin(top_assets)) & 
        (forward_returns_df['zscore_bin'] != 'No Z-Score')
    ]
    
    if len(zscore_data) > 0:
        zscore_pivot = zscore_data.pivot_table(
            index='asset', 
            columns='zscore_bin', 
            values='max_gain_365d', 
            aggfunc='mean'
        )
        
        # Reorder Z-score columns
        zscore_order = [">+3σ", "+2.5σ to +3σ", "+2σ to +2.5σ", "+1.5σ to +2σ", "+1σ to +1.5σ", 
                       "+0.5σ to +1σ", "-0.5σ to +0.5σ", "-1σ to -0.5σ", "-1.5σ to -1σ", 
                       "-2σ to -1.5σ", "-2.5σ to -2σ", "-3σ to -2.5σ", "<-3σ"]
        
        available_zscore_bins = [bin_name for bin_name in zscore_order if bin_name in zscore_pivot.columns]
        if available_zscore_bins:
            zscore_pivot = zscore_pivot[available_zscore_bins]
            
            sns.heatmap(zscore_pivot, annot=True, fmt='.1f', cmap='RdYlGn', center=0, 
                       ax=ax2, cbar_kws={'label': 'Average Max Gain (%)'})
            ax2.set_title('Average 365-Day Max Gains by Asset and Z-Score Bin', fontweight='bold', fontsize=14)
            ax2.set_xlabel('Z-Score Bin')
            ax2.set_ylabel('Asset')
        else:
            ax2.text(0.5, 0.5, 'Insufficient Z-Score Data', ha='center', va='center', 
                    transform=ax2.transAxes, fontsize=16)
            ax2.set_title('Z-Score Analysis - Insufficient Data', fontweight='bold')
    else:
        ax2.text(0.5, 0.5, 'No Z-Score Data Available', ha='center', va='center', 
                transform=ax2.transAxes, fontsize=16)
        ax2.set_title('Z-Score Analysis - No Data Available', fontweight='bold')
    
    plt.tight_layout()
    plt.savefig(f"{output_dir}/asset_performance_heatmaps.png", dpi=300, bbox_inches='tight')
    plt.close()

def create_individual_asset_reports(forward_returns_df, output_dir):
    """
    Create individual performance reports for top assets
    """
    print("  Creating individual asset reports...")
    
    # Get top 10 assets by sample count
    asset_counts = forward_returns_df['asset'].value_counts()
    top_assets = asset_counts.head(10).index.tolist()
    
    for asset in top_assets:
        asset_data = forward_returns_df[forward_returns_df['asset'] == asset]
        
        fig, ((ax1, ax2), (ax3, ax4)) = plt.subplots(2, 2, figsize=(16, 12))
        
        # Deviation vs Max Gain scatter
        ax1.scatter(asset_data['price_deviation'], asset_data['max_gain_365d'], 
                   alpha=0.6, s=20, color='blue')
        ax1.set_xlabel('Price Deviation (%)')
        ax1.set_ylabel('365-Day Max Gain (%)')
        ax1.set_title(f'{asset}: Deviation vs Max Gain', fontweight='bold')
        ax1.grid(True, alpha=0.3)
        
        # Add trend line
        if len(asset_data) > 10:
            z = np.polyfit(asset_data['price_deviation'], asset_data['max_gain_365d'], 1)
            p = np.poly1d(z)
            ax1.plot(asset_data['price_deviation'], p(asset_data['price_deviation']), 
                    "r--", alpha=0.8, linewidth=2)
        
        # Max gain distribution
        ax2.hist(asset_data['max_gain_365d'], bins=30, alpha=0.7, color='green', edgecolor='black')
        ax2.axvline(asset_data['max_gain_365d'].mean(), color='red', linestyle='--', linewidth=2,
                   label=f'Mean: {asset_data["max_gain_365d"].mean():.1f}%')
        ax2.set_xlabel('365-Day Max Gain (%)')
        ax2.set_ylabel('Frequency')
        ax2.set_title(f'{asset}: Max Gain Distribution', fontweight='bold')
        ax2.legend()
        ax2.grid(True, alpha=0.3)
        
        # Time series of entry points colored by future performance
        ax3.scatter(asset_data['entry_date'], asset_data['price_deviation'], 
                   c=asset_data['max_gain_365d'], cmap='RdYlGn', s=30, alpha=0.7)
        ax3.set_xlabel('Entry Date')
        ax3.set_ylabel('Price Deviation (%)')
        ax3.set_title(f'{asset}: Entry Points Colored by Future Max Gain', fontweight='bold')
        ax3.grid(True, alpha=0.3)
        
        # Success rate by deviation ranges
        deviation_ranges = pd.cut(asset_data['price_deviation'], bins=10)
        success_by_range = asset_data.groupby(deviation_ranges)['max_gain_365d'].agg([
            'mean', 'count', lambda x: (x > 25).mean() * 100
        ])
        success_by_range.columns = ['avg_gain', 'count', 'success_rate_25pct']
        success_by_range = success_by_range[success_by_range['count'] >= 5]  # Filter for significance
        
        if len(success_by_range) > 0:
            x_pos = range(len(success_by_range))
            bars = ax4.bar(x_pos, success_by_range['success_rate_25pct'], alpha=0.7, color='orange')
            ax4.set_xticks(x_pos)
            ax4.set_xticklabels([f'{interval.left:.1f} to {interval.right:.1f}' 
                               for interval in success_by_range.index], rotation=45)
            ax4.set_ylabel('Success Rate for 25%+ Gains (%)')
            ax4.set_xlabel('Deviation Range (%)')
            ax4.set_title(f'{asset}: Success Rate by Deviation Range', fontweight='bold')
            ax4.grid(True, alpha=0.3)
            
            # Add count labels
            for bar, count in zip(bars, success_by_range['count']):
                ax4.text(bar.get_x() + bar.get_width()/2., bar.get_height() + 1,
                        f'n={int(count)}', ha='center', va='bottom', fontsize=8)
        
        plt.tight_layout()
        plt.savefig(f"{output_dir}/individual_reports/{asset}_performance_report.png", 
                   dpi=300, bbox_inches='tight')
        plt.close()

def save_analysis_results(forward_returns_df, deviation_analysis, zscore_analysis, output_dir):
    """
    Save comprehensive analysis results to CSV files
    """
    print("\nSaving analysis results...")
    
    # Save main forward returns data
    forward_returns_df.to_csv(f"{output_dir}/forward_returns_365d_all_data.csv", index=False)
    print(f"  ✅ Saved main dataset: {len(forward_returns_df):,} records")
    
    # Save deviation bin analysis
    deviation_analysis.to_csv(f"{output_dir}/deviation_bins_analysis.csv", index=False)
    print(f"  ✅ Saved deviation analysis: {len(deviation_analysis)} bins")
    
    # Save Z-score bin analysis
    if len(zscore_analysis) > 0:
        zscore_analysis.to_csv(f"{output_dir}/zscore_bins_analysis.csv", index=False)
        print(f"  ✅ Saved Z-score analysis: {len(zscore_analysis)} bins")
    
    # Save asset-specific summaries
    asset_summary = forward_returns_df.groupby('asset').agg({
        'max_gain_365d': ['count', 'mean', 'median', 'std'],
        'final_return_365d': ['mean', 'median'],
        'days_to_max_gain': 'mean',
        'volatility_365d': 'mean',
        'price_deviation': ['mean', 'std'],
        'z_score': ['mean', 'std']
    }).round(3)
    
    # Flatten column names
    asset_summary.columns = ['_'.join(col).strip() for col in asset_summary.columns]
    asset_summary = asset_summary.reset_index()
    
    # Add success rate calculations
    success_rates = forward_returns_df.groupby('asset').apply(
        lambda x: pd.Series({
            'success_rate_positive': (x['final_return_365d'] > 0).mean() * 100,
            'success_rate_25pct': (x['max_gain_365d'] > 25).mean() * 100,
            'success_rate_50pct': (x['max_gain_365d'] > 50).mean() * 100,
            'success_rate_100pct': (x['max_gain_365d'] > 100).mean() * 100
        })
    ).round(2)
    
    asset_summary = asset_summary.merge(success_rates, left_on='asset', right_index=True)
    asset_summary.to_csv(f"{output_dir}/asset_performance_summary.csv", index=False)
    print(f"  ✅ Saved asset summary: {len(asset_summary)} assets")
    
    # Save detailed bin breakdowns by asset
    asset_bin_details = []
    
    for asset in forward_returns_df['asset'].unique():
        asset_data = forward_returns_df[forward_returns_df['asset'] == asset]
        
        # Deviation bins for this asset
        for bin_name in asset_data['deviation_bin'].unique():
            bin_data = asset_data[asset_data['deviation_bin'] == bin_name]
            if len(bin_data) >= 5:  # Minimum sample size
                asset_bin_details.append({
                    'asset': asset,
                    'bin_type': 'deviation',
                    'bin_name': bin_name,
                    'sample_count': len(bin_data),
                    'avg_max_gain': bin_data['max_gain_365d'].mean(),
                    'median_max_gain': bin_data['max_gain_365d'].median(),
                    'std_max_gain': bin_data['max_gain_365d'].std(),
                    'success_rate_25pct': (bin_data['max_gain_365d'] > 25).mean() * 100,
                    'avg_days_to_max': bin_data['days_to_max_gain'].mean()
                })
        
        # Z-score bins for this asset (if available)
        zscore_data = asset_data[asset_data['zscore_bin'] != 'No Z-Score']
        for bin_name in zscore_data['zscore_bin'].unique():
            bin_data = zscore_data[zscore_data['zscore_bin'] == bin_name]
            if len(bin_data) >= 5:
                asset_bin_details.append({
                    'asset': asset,
                    'bin_type': 'zscore',
                    'bin_name': bin_name,
                    'sample_count': len(bin_data),
                    'avg_max_gain': bin_data['max_gain_365d'].mean(),
                    'median_max_gain': bin_data['max_gain_365d'].median(),
                    'std_max_gain': bin_data['max_gain_365d'].std(),
                    'success_rate_25pct': (bin_data['max_gain_365d'] > 25).mean() * 100,
                    'avg_days_to_max': bin_data['days_to_max_gain'].mean()
                })
    
    if asset_bin_details:
        asset_bin_df = pd.DataFrame(asset_bin_details)
        asset_bin_df.to_csv(f"{output_dir}/asset_bin_detailed_analysis.csv", index=False)
        print(f"  ✅ Saved detailed bin analysis: {len(asset_bin_df)} asset-bin combinations")

def print_comprehensive_summary(forward_returns_df, deviation_analysis, zscore_analysis):
    """
    Print comprehensive summary of the analysis
    """
    print("\n" + "="*80)
    print("365-DAY FORWARD RETURNS ANALYSIS SUMMARY")
    print("="*80)
    
    print(f"\nDATASET OVERVIEW:")
    print(f"  Total forward return samples: {len(forward_returns_df):,}")
    print(f"  Unique assets analyzed: {forward_returns_df['asset'].nunique()}")
    print(f"  Date range: {forward_returns_df['entry_date'].min().strftime('%Y-%m-%d')} to {forward_returns_df['entry_date'].max().strftime('%Y-%m-%d')}")
    print(f"  Average samples per asset: {len(forward_returns_df) / forward_returns_df['asset'].nunique():.0f}")
    
    print(f"\nOVERALL PERFORMANCE METRICS:")
    print(f"  Average max gain: {forward_returns_df['max_gain_365d'].mean():.2f}%")
    print(f"  Median max gain: {forward_returns_df['max_gain_365d'].median():.2f}%")
    print(f"  Success rate (positive final return): {(forward_returns_df['final_return_365d'] > 0).mean()*100:.1f}%")
    print(f"  Success rate (25%+ max gain): {(forward_returns_df['max_gain_365d'] > 25).mean()*100:.1f}%")
    print(f"  Success rate (50%+ max gain): {(forward_returns_df['max_gain_365d'] > 50).mean()*100:.1f}%")
    print(f"  Success rate (100%+ max gain): {(forward_returns_df['max_gain_365d'] > 100).mean()*100:.1f}%")
    
    # Deviation bins analysis
    if len(deviation_analysis) > 0:
        print(f"\nDEVIATION BINS ANALYSIS:")
        print("-" * 60)
        print(f"{'Bin':<15} {'Samples':<8} {'Avg Gain':<10} {'25%+ Rate':<10} {'Sharpe':<8}")
        print("-" * 60)
        
        for _, row in deviation_analysis.iterrows():
            print(f"{row['deviation_bin']:<15} {row['sample_count']:>7,} "
                  f"{row['avg_max_gain']:>9.1f}% {row['success_rate_25pct']:>9.1f}% "
                  f"{row['sharpe_ratio']:>7.2f}")
        
        # Find best performing bins
        best_gain_bin = deviation_analysis.loc[deviation_analysis['avg_max_gain'].idxmax()]
        best_sharpe_bin = deviation_analysis.loc[deviation_analysis['sharpe_ratio'].idxmax()]
        best_success_bin = deviation_analysis.loc[deviation_analysis['success_rate_25pct'].idxmax()]
        
        print(f"\nTOP PERFORMING DEVIATION BINS:")
        print(f"  Highest avg gain: {best_gain_bin['deviation_bin']} ({best_gain_bin['avg_max_gain']:.1f}%)")
        print(f"  Best Sharpe ratio: {best_sharpe_bin['deviation_bin']} ({best_sharpe_bin['sharpe_ratio']:.2f})")
        print(f"  Best success rate: {best_success_bin['deviation_bin']} ({best_success_bin['success_rate_25pct']:.1f}%)")
    
    # Z-score bins analysis
    if len(zscore_analysis) > 0:
        print(f"\nZ-SCORE BINS ANALYSIS:")
        print("-" * 60)
        print(f"{'Bin':<15} {'Samples':<8} {'Avg Gain':<10} {'25%+ Rate':<10} {'Sharpe':<8}")
        print("-" * 60)
        
        for _, row in zscore_analysis.iterrows():
            print(f"{row['zscore_bin']:<15} {row['sample_count']:>7,} "
                  f"{row['avg_max_gain']:>9.1f}% {row['success_rate_25pct']:>9.1f}% "
                  f"{row['sharpe_ratio']:>7.2f}")
        
        # Find best performing Z-score bins
        best_zscore_gain = zscore_analysis.loc[zscore_analysis['avg_max_gain'].idxmax()]
        best_zscore_sharpe = zscore_analysis.loc[zscore_analysis['sharpe_ratio'].idxmax()]
        
        print(f"\nTOP PERFORMING Z-SCORE BINS:")
        print(f"  Highest avg gain: {best_zscore_gain['zscore_bin']} ({best_zscore_gain['avg_max_gain']:.1f}%)")
        print(f"  Best Sharpe ratio: {best_zscore_sharpe['zscore_bin']} ({best_zscore_sharpe['sharpe_ratio']:.2f})")
    
    # Asset performance ranking
    asset_performance = forward_returns_df.groupby('asset').agg({
        'max_gain_365d': ['mean', 'count'],
        'final_return_365d': 'mean'
    }).round(2)
    asset_performance.columns = ['avg_max_gain', 'sample_count', 'avg_final_return']
    asset_performance = asset_performance[asset_performance['sample_count'] >= 50]  # Filter for significance
    asset_performance = asset_performance.sort_values('avg_max_gain', ascending=False)
    
    print(f"\nTOP 10 ASSETS BY AVERAGE MAX GAIN (min 50 samples):")
    print("-" * 50)
    print(f"{'Asset':<8} {'Avg Max Gain':<12} {'Final Return':<12} {'Samples':<8}")
    print("-" * 50)
    
    for asset, row in asset_performance.head(10).iterrows():
        print(f"{asset:<8} {row['avg_max_gain']:>11.1f}% {row['avg_final_return']:>11.1f}% {row['sample_count']:>7.0f}")
    
    print(f"\nKEY INSIGHTS:")
    print("-" * 40)
    
    # Analyze undervalued vs overvalued performance
    undervalued = forward_returns_df[forward_returns_df['price_deviation'] < -10]
    overvalued = forward_returns_df[forward_returns_df['price_deviation'] > 10]
    
    if len(undervalued) > 0 and len(overvalued) > 0:
        undervalued_gain = undervalued['max_gain_365d'].mean()
        overvalued_gain = overvalued['max_gain_365d'].mean()
        
        print(f"  Undervalued assets (>-10%): {undervalued_gain:.1f}% avg max gain")
        print(f"  Overvalued assets (>+10%): {overvalued_gain:.1f}% avg max gain")
        print(f"  Undervalued advantage: {undervalued_gain - overvalued_gain:+.1f}%")
        
        if undervalued_gain > overvalued_gain:
            print(f"  ✅ Mean reversion signal confirmed!")
        else:
            print(f"  ⚠️  Momentum signal detected!")
    
    # Time analysis
    avg_days_to_max = forward_returns_df['days_to_max_gain'].mean()
    print(f"  Average time to maximum gain: {avg_days_to_max:.0f} days")
    
    if avg_days_to_max < 180:
        print(f"  ⚡ Quick gains pattern - consider shorter holding periods")
    else:
        print(f"  🕐 Patient gains pattern - longer holding periods may be optimal")

def save_individual_asset_results(forward_returns_df, base_v2_dir):
    """
    Save individual asset forward returns data into their existing folders
    """
    print("\nSaving individual asset results to their folders...")
    
    for asset in forward_returns_df['asset'].unique():
        asset_data = forward_returns_df[forward_returns_df['asset'] == asset]
        asset_folder = os.path.join(base_v2_dir, asset)
        
        if os.path.exists(asset_folder):
            # Save asset-specific forward returns data
            asset_forward_file = os.path.join(asset_folder, f"{asset}_forward_returns_365d.csv")
            asset_data.to_csv(asset_forward_file, index=False)
            
            # Create asset-specific bin analysis
            asset_deviation_analysis = []
            
            # Analyze deviation bins for this asset
            bin_order = [">+50%", "+45% to +50%", "+40% to +45%", "+35% to +40%", "+30% to +35%", 
                         "+25% to +30%", "+20% to +25%", "+15% to +20%", "+10% to +15%", "+5% to +10%", 
                         "-5% to +5%", "-10% to -5%", "-15% to -10%", "-20% to -15%", "-25% to -20%", 
                         "-30% to -25%", "-35% to -30%", "-40% to -35%", "-45% to -40%", "-50% to -45%", "<-50%"]
            
            for bin_name in bin_order:
                bin_data = asset_data[asset_data['deviation_bin'] == bin_name]
                
                if len(bin_data) >= 3:  # Minimum samples for asset-specific analysis
                    max_gains = bin_data['max_gain_365d']
                    final_returns = bin_data['final_return_365d']
                    
                    asset_deviation_analysis.append({
                        'deviation_bin': bin_name,
                        'sample_count': len(bin_data),
                        'avg_max_gain': max_gains.mean(),
                        'median_max_gain': max_gains.median(),
                        'std_max_gain': max_gains.std(),
                        'avg_final_return': final_returns.mean(),
                        'success_rate_positive': (final_returns > 0).mean() * 100,
                        'success_rate_25pct': (max_gains > 25).mean() * 100,
                        'success_rate_50pct': (max_gains > 50).mean() * 100,
                        'avg_days_to_max': bin_data['days_to_max_gain'].mean(),
                        'avg_max_loss': bin_data['max_loss_365d'].mean()
                    })
            
            if asset_deviation_analysis:
                asset_bin_file = os.path.join(asset_folder, f"{asset}_forward_returns_by_bins.csv")
                pd.DataFrame(asset_deviation_analysis).to_csv(asset_bin_file, index=False)
            
            # Create asset-specific summary stats
            asset_summary = {
                'asset': asset,
                'total_samples': len(asset_data),
                'date_range_start': asset_data['entry_date'].min(),
                'date_range_end': asset_data['entry_date'].max(),
                'overall_avg_max_gain': asset_data['max_gain_365d'].mean(),
                'overall_median_max_gain': asset_data['max_gain_365d'].median(),
                'overall_std_max_gain': asset_data['max_gain_365d'].std(),
                'overall_avg_final_return': asset_data['final_return_365d'].mean(),
                'overall_success_rate_positive': (asset_data['final_return_365d'] > 0).mean() * 100,
                'overall_success_rate_25pct': (asset_data['max_gain_365d'] > 25).mean() * 100,
                'overall_success_rate_50pct': (asset_data['max_gain_365d'] > 50).mean() * 100,
                'overall_success_rate_100pct': (asset_data['max_gain_365d'] > 100).mean() * 100,
                'best_deviation_bin': '',
                'best_bin_avg_gain': 0
            }
            
            # Find best performing bin for this asset
            if asset_deviation_analysis:
                best_bin = max(asset_deviation_analysis, key=lambda x: x['avg_max_gain'])
                asset_summary['best_deviation_bin'] = best_bin['deviation_bin']
                asset_summary['best_bin_avg_gain'] = best_bin['avg_max_gain']
            
            asset_summary_file = os.path.join(asset_folder, f"{asset}_forward_returns_summary.csv")
            pd.DataFrame([asset_summary]).to_csv(asset_summary_file, index=False)
            
            print(f"  ✅ {asset}: {len(asset_data)} samples saved to individual folder")
        else:
            print(f"  ⚠️  {asset}: folder not found, skipping individual save")

def save_analysis_results(forward_returns_df, deviation_analysis, zscore_analysis, output_dir, base_v2_dir):
    """
    Save comprehensive analysis results - group data in summary folder, individual data in asset folders
    """
    print("\nSaving analysis results...")
    
    # Save individual asset results to their own folders
    save_individual_asset_results(forward_returns_df, base_v2_dir)
    
    # Save group/summary analysis in the summary folder
    print(f"\nSaving group analysis to summary folder...")
    
    # Save main forward returns data (all combined)
    forward_returns_df.to_csv(f"{output_dir}/ALL_ASSETS_forward_returns_365d.csv", index=False)
    print(f"  ✅ Saved combined dataset: {len(forward_returns_df):,} records")
    
    # Save deviation bin analysis (group summary)
    deviation_analysis.to_csv(f"{output_dir}/GROUP_deviation_bins_analysis.csv", index=False)
    print(f"  ✅ Saved group deviation analysis: {len(deviation_analysis)} bins")
    
    # Save Z-score bin analysis (group summary)
    if len(zscore_analysis) > 0:
        zscore_analysis.to_csv(f"{output_dir}/GROUP_zscore_bins_analysis.csv", index=False)
        print(f"  ✅ Saved group Z-score analysis: {len(zscore_analysis)} bins")
    
    # Save cross-asset performance comparison
    asset_summary = forward_returns_df.groupby('asset').agg({
        'max_gain_365d': ['count', 'mean', 'median', 'std'],
        'final_return_365d': ['mean', 'median'],
        'days_to_max_gain': 'mean',
        'volatility_365d': 'mean',
        'price_deviation': ['mean', 'std'],
        'z_score': ['mean', 'std']
    }).round(3)
    
    # Flatten column names
    asset_summary.columns = ['_'.join(col).strip() for col in asset_summary.columns]
    asset_summary = asset_summary.reset_index()
    
    # Add success rate calculations
    success_rates = forward_returns_df.groupby('asset').apply(
        lambda x: pd.Series({
            'success_rate_positive': (x['final_return_365d'] > 0).mean() * 100,
            'success_rate_25pct': (x['max_gain_365d'] > 25).mean() * 100,
            'success_rate_50pct': (x['max_gain_365d'] > 50).mean() * 100,
            'success_rate_100pct': (x['max_gain_365d'] > 100).mean() * 100
        })
    ).round(2)
    
    asset_summary = asset_summary.merge(success_rates, left_on='asset', right_index=True)
    asset_summary.to_csv(f"{output_dir}/CROSS_ASSET_performance_comparison.csv", index=False)
    print(f"  ✅ Saved cross-asset comparison: {len(asset_summary)} assets")

def create_individual_asset_reports(forward_returns_df, base_v2_dir):
    """
    Create individual performance reports and save them to each asset's folder
    """
    print("  Creating individual asset reports in their folders...")
    
    # Get top 10 assets by sample count for detailed reports
    asset_counts = forward_returns_df['asset'].value_counts()
    top_assets = asset_counts.head(10).index.tolist()
    
    for asset in top_assets:
        asset_data = forward_returns_df[forward_returns_df['asset'] == asset]
        asset_folder = os.path.join(base_v2_dir, asset)
        
        if not os.path.exists(asset_folder):
            continue
            
        fig, ((ax1, ax2), (ax3, ax4)) = plt.subplots(2, 2, figsize=(16, 12))
        
        # Deviation vs Max Gain scatter
        ax1.scatter(asset_data['price_deviation'], asset_data['max_gain_365d'], 
                   alpha=0.6, s=20, color='blue')
        ax1.set_xlabel('Price Deviation (%)')
        ax1.set_ylabel('365-Day Max Gain (%)')
        ax1.set_title(f'{asset}: Deviation vs Max Gain', fontweight='bold')
        ax1.grid(True, alpha=0.3)
        
        # Add trend line
        if len(asset_data) > 10:
            z = np.polyfit(asset_data['price_deviation'], asset_data['max_gain_365d'], 1)
            p = np.poly1d(z)
            ax1.plot(asset_data['price_deviation'], p(asset_data['price_deviation']), 
                    "r--", alpha=0.8, linewidth=2)
        
        # Max gain distribution
        ax2.hist(asset_data['max_gain_365d'], bins=30, alpha=0.7, color='green', edgecolor='black')
        ax2.axvline(asset_data['max_gain_365d'].mean(), color='red', linestyle='--', linewidth=2,
                   label=f'Mean: {asset_data["max_gain_365d"].mean():.1f}%')
        ax2.set_xlabel('365-Day Max Gain (%)')
        ax2.set_ylabel('Frequency')
        ax2.set_title(f'{asset}: Max Gain Distribution', fontweight='bold')
        ax2.legend()
        ax2.grid(True, alpha=0.3)
        
        # Time series of entry points colored by future performance
        ax3.scatter(asset_data['entry_date'], asset_data['price_deviation'], 
                   c=asset_data['max_gain_365d'], cmap='RdYlGn', s=30, alpha=0.7)
        ax3.set_xlabel('Entry Date')
        ax3.set_ylabel('Price Deviation (%)')
        ax3.set_title(f'{asset}: Entry Points Colored by Future Max Gain', fontweight='bold')
        ax3.grid(True, alpha=0.3)
        
        # Success rate by deviation ranges
        deviation_ranges = pd.cut(asset_data['price_deviation'], bins=10)
        success_by_range = asset_data.groupby(deviation_ranges)['max_gain_365d'].agg([
            'mean', 'count', lambda x: (x > 25).mean() * 100
        ])
        success_by_range.columns = ['avg_gain', 'count', 'success_rate_25pct']
        success_by_range = success_by_range[success_by_range['count'] >= 5]  # Filter for significance
        
        if len(success_by_range) > 0:
            x_pos = range(len(success_by_range))
            bars = ax4.bar(x_pos, success_by_range['success_rate_25pct'], alpha=0.7, color='orange')
            ax4.set_xticks(x_pos)
            ax4.set_xticklabels([f'{interval.left:.1f} to {interval.right:.1f}' 
                               for interval in success_by_range.index], rotation=45)
            ax4.set_ylabel('Success Rate for 25%+ Gains (%)')
            ax4.set_xlabel('Deviation Range (%)')
            ax4.set_title(f'{asset}: Success Rate by Deviation Range', fontweight='bold')
            ax4.grid(True, alpha=0.3)
            
            # Add count labels
            for bar, count in zip(bars, success_by_range['count']):
                ax4.text(bar.get_x() + bar.get_width()/2., bar.get_height() + 1,
                        f'n={int(count)}', ha='center', va='bottom', fontsize=8)
        
        plt.tight_layout()
        plt.savefig(f"{asset_folder}/{asset}_forward_returns_analysis.png", 
                   dpi=300, bbox_inches='tight')
        plt.close()
        
        print(f"    ✅ {asset}: Individual report saved to {asset_folder}/")

def main():
    print("IWLS Forward Returns Analysis - 365 Day Maximum Gains")
    print("=" * 70)
    print("Analyzing future returns based on IWLS deviation and Z-score bins")
    print("Includes: Individual asset files + group summary analysis")
    
    # Use existing V2 structure to keep things clean
    base_v2_dir = "/Users/tim/IWLS-OPTIONS/IWLS_ANALYSIS_V2"
    
    if not os.path.exists(base_v2_dir):
        print("❌ IWLS_ANALYSIS_V2 directory not found. Run the IWLS V2 analysis first.")
        return
    
    # Create forward returns summary folder for group analysis
    output_dir = os.path.join(base_v2_dir, "FORWARD_RETURNS_SUMMARY")
    os.makedirs(output_dir, exist_ok=True)
    
    print(f"\nBase directory: {base_v2_dir}")
    print(f"Group summary directory: {output_dir}")
    print("✅ Individual files will go into each asset's existing folder")
    print("✅ Group analysis will go into FORWARD_RETURNS_SUMMARY folder")
    
    # Load IWLS results
    all_results = load_iwls_results()
    if not all_results:
        return
    
    # Calculate forward returns for all assets
    forward_returns_df = analyze_all_assets_forward_returns(all_results)
    
    if len(forward_returns_df) == 0:
        print("❌ No forward returns data generated!")
        return
    
    # Analyze by deviation bins
    deviation_analysis = analyze_returns_by_deviation_bins(forward_returns_df)
    
    # Analyze by Z-score bins
    zscore_analysis = analyze_returns_by_zscore_bins(forward_returns_df)
    
    # Create comprehensive visualizations
    create_comprehensive_visualizations(forward_returns_df, deviation_analysis, 
                                      zscore_analysis, output_dir)
    
    # Create individual asset reports (saved to their own folders)
    create_individual_asset_reports(forward_returns_df, base_v2_dir)
    
    # Save all results (individual to asset folders, group to summary folder)
    save_analysis_results(forward_returns_df, deviation_analysis, zscore_analysis, output_dir, base_v2_dir)
    
    # Print comprehensive summary
    print_comprehensive_summary(forward_returns_df, deviation_analysis, zscore_analysis)
    
    print(f"\n" + "="*70)
    print("FORWARD RETURNS ANALYSIS COMPLETE")
    print("="*70)
    print(f"Results saved to: {output_dir}")
    print("\nFiles created:")
    print("  📄 forward_returns_365d_all_data.csv (complete dataset)")
    print("  📄 deviation_bins_analysis.csv (5% deviation bins summary)")
    print("  📄 zscore_bins_analysis.csv (Z-score bins summary)")
    print("  📄 asset_performance_summary.csv (per-asset metrics)")
    print("  📄 asset_bin_detailed_analysis.csv (asset-bin combinations)")
    print("  📊 deviation_bins_comprehensive_analysis.png (6-panel deviation analysis)")
    print("  📊 zscore_bins_comprehensive_analysis.png (6-panel Z-score analysis)")
    print("  📊 asset_performance_heatmaps.png (cross-asset performance)")
    print("  📁 individual_reports/ (detailed reports for top 10 assets)")
    
    print(f"\n🎯 Ready for strategy development:")
    print(f"   • Use deviation_bins_analysis.csv for entry signal optimization")
    print(f"   • Use zscore_bins_analysis.csv for normalized comparisons")
    print(f"   • Use asset_performance_summary.csv for asset selection")
    print(f"   • Review individual reports for asset-specific insights")

if __name__ == "__main__":
    main()