import pandas as pd
import numpy as np
import matplotlib.pyplot as plt
import os
import glob
from datetime import datetime, timedelta
from collections import defaultdict
import warnings
from scipy import stats
import seaborn as sns
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

def calculate_forward_performance(all_results, asset, entry_date, days_forward=365):
    """
    Calculate forward performance for an asset from entry date
    """
    if asset not in all_results:
        return None, None
    
    df = all_results[asset]
    
    # Get entry price
    entry_data = df[df['date'] == entry_date]
    if len(entry_data) == 0:
        return None, None
    
    entry_price = entry_data['price'].iloc[0]
    
    # Get future data (365 days forward)
    end_date = entry_date + timedelta(days=days_forward)
    future_data = df[(df['date'] > entry_date) & (df['date'] <= end_date)]
    
    if len(future_data) == 0:
        return None, None
    
    # Calculate 365-day forward change (end price vs entry price)
    final_data = future_data.iloc[-1]  # Last available data within 365 days
    final_price = final_data['price']
    days_held = (final_data['date'] - entry_date).days
    
    forward_365_change = ((final_price / entry_price) - 1) * 100
    
    # Calculate maximum gain during the 365-day period
    max_price = future_data['price'].max()
    forward_365_max_gain = ((max_price / entry_price) - 1) * 100
    
    return {
        'entry_date': entry_date,
        'entry_price': entry_price,
        'final_date': final_data['date'],
        'final_price': final_price,
        'days_held': days_held,
        'forward_365_change': forward_365_change,
        'forward_365_max_gain': forward_365_max_gain,
        'max_price': max_price
    }, True

def calculate_daily_trading_analytics(all_results):
    """
    For each day, find top 5 absolute deviation assets and calculate forward performance
    """
    print("\nCalculating daily trading analytics...")
    
    # Get all unique dates
    all_dates = set()
    for df in all_results.values():
        all_dates.update(df['date'].tolist())
    
    all_dates = sorted(list(all_dates))
    
    # Only process dates that have at least 365 days of forward data
    cutoff_date = max(all_dates) - timedelta(days=365)
    trading_dates = [date for date in all_dates if date <= cutoff_date]
    
    print(f"Processing {len(trading_dates)} trading dates with sufficient forward data...")
    
    daily_trading_results = []
    processed_count = 0
    
    for current_date in trading_dates:
        processed_count += 1
        
        if processed_count % 100 == 0:
            print(f"  Processed {processed_count}/{len(trading_dates)} dates...")
        
        # Get absolute deviations for all assets on this date
        daily_deviations = {}
        
        for asset_name, df in all_results.items():
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
        
        # Calculate forward performance for each asset in top 5
        group_forward_results = []
        
        for asset_name, deviation_data in top5_assets:
            forward_perf, success = calculate_forward_performance(all_results, asset_name, current_date)
            
            if success and forward_perf:
                forward_perf['asset'] = asset_name
                forward_perf['absolute_deviation'] = deviation_data['absolute_deviation']
                forward_perf['raw_deviation'] = deviation_data['raw_deviation']
                group_forward_results.append(forward_perf)
        
        # Skip if we don't have forward data for enough assets
        if len(group_forward_results) < 3:  # Need at least 3 assets with forward data
            continue
        
        # Calculate group averages
        group_avg_forward_365_change = np.mean([r['forward_365_change'] for r in group_forward_results])
        group_avg_forward_365_max_gain = np.mean([r['forward_365_max_gain'] for r in group_forward_results])
        group_avg_days_held = np.mean([r['days_held'] for r in group_forward_results])
        
        # Calculate group statistics
        group_median_forward_365_change = np.median([r['forward_365_change'] for r in group_forward_results])
        group_median_forward_365_max_gain = np.median([r['forward_365_max_gain'] for r in group_forward_results])
        
        group_std_forward_365_change = np.std([r['forward_365_change'] for r in group_forward_results])
        group_std_forward_365_max_gain = np.std([r['forward_365_max_gain'] for r in group_forward_results])
        
        # Record daily trading result
        daily_result = {
            'entry_date': current_date,
            'sum_top5_absolute_deviations': sum_top5_deviations,
            'num_assets_in_group': len(group_forward_results),
            'group_avg_forward_365_change': group_avg_forward_365_change,
            'group_avg_forward_365_max_gain': group_avg_forward_365_max_gain,
            'group_median_forward_365_change': group_median_forward_365_change,
            'group_median_forward_365_max_gain': group_median_forward_365_max_gain,
            'group_std_forward_365_change': group_std_forward_365_change,
            'group_std_forward_365_max_gain': group_std_forward_365_max_gain,
            'group_avg_days_held': group_avg_days_held,
            'top5_assets': [r['asset'] for r in group_forward_results],
            'individual_results': group_forward_results
        }
        
        daily_trading_results.append(daily_result)
    
    print(f"Completed {len(daily_trading_results)} trading entries with forward data")
    return daily_trading_results

def analyze_predictive_power(daily_trading_results):
    """
    Analyze whether absolute deviation magnitude has predictive power
    """
    if not daily_trading_results:
        return {}
    
    df = pd.DataFrame(daily_trading_results)
    
    # Create deviation magnitude bins
    df['deviation_magnitude_bin'] = pd.qcut(df['sum_top5_absolute_deviations'], 
                                          q=5, 
                                          labels=['Lowest', 'Low', 'Medium', 'High', 'Highest'])
    
    # Analyze performance by deviation magnitude
    bin_analysis = df.groupby('deviation_magnitude_bin').agg({
        'group_avg_forward_365_change': ['count', 'mean', 'median', 'std'],
        'group_avg_forward_365_max_gain': ['mean', 'median', 'std'],
        'sum_top5_absolute_deviations': ['mean', 'min', 'max']
    }).round(2)
    
    # Flatten column names
    bin_analysis.columns = ['_'.join(col).strip() for col in bin_analysis.columns]
    
    # Overall statistics
    analysis = {
        'total_trades': len(df),
        'date_range': (df['entry_date'].min(), df['entry_date'].max()),
        'overall_avg_forward_change': df['group_avg_forward_365_change'].mean(),
        'overall_avg_max_gain': df['group_avg_forward_365_max_gain'].mean(),
        'overall_std_forward_change': df['group_avg_forward_365_change'].std(),
        'overall_std_max_gain': df['group_avg_forward_365_max_gain'].std(),
        'bin_analysis': bin_analysis
    }
    
    # Calculate correlation between deviation magnitude and forward performance
    correlation_change = df['sum_top5_absolute_deviations'].corr(df['group_avg_forward_365_change'])
    correlation_max_gain = df['sum_top5_absolute_deviations'].corr(df['group_avg_forward_365_max_gain'])
    
    analysis['correlation_deviation_vs_forward_change'] = correlation_change
    analysis['correlation_deviation_vs_max_gain'] = correlation_max_gain
    
    # Statistical significance test (if we have enough data)
    if len(df) > 50:
        # Test if higher deviation magnitude leads to different forward returns
        high_deviation = df[df['deviation_magnitude_bin'] == 'Highest']['group_avg_forward_365_change']
        low_deviation = df[df['deviation_magnitude_bin'] == 'Lowest']['group_avg_forward_365_change']
        
        if len(high_deviation) > 5 and len(low_deviation) > 5:
            t_stat, p_value = stats.ttest_ind(high_deviation, low_deviation)
            analysis['t_test_high_vs_low'] = {
                't_statistic': t_stat,
                'p_value': p_value,
                'significant': p_value < 0.05
            }
    
    return analysis

def create_predictive_power_visualizations(daily_trading_results, analysis, output_dir):
    """
    Create comprehensive visualizations to analyze predictive power
    """
    if not daily_trading_results:
        print("No data to visualize")
        return
    
    df = pd.DataFrame(daily_trading_results)
    df['deviation_magnitude_bin'] = pd.qcut(df['sum_top5_absolute_deviations'], 
                                          q=5, 
                                          labels=['Lowest', 'Low', 'Medium', 'High', 'Highest'])
    
    # Create comprehensive figure with 6 subplots
    fig = plt.figure(figsize=(24, 16))
    
    # Plot 1: Scatter plot of deviation magnitude vs forward returns
    ax1 = plt.subplot(2, 3, 1)
    scatter = ax1.scatter(df['sum_top5_absolute_deviations'], 
                         df['group_avg_forward_365_change'],
                         c=df['group_avg_forward_365_max_gain'], 
                         cmap='viridis', alpha=0.6, s=50)
    
    # Add trend line
    z = np.polyfit(df['sum_top5_absolute_deviations'], df['group_avg_forward_365_change'], 1)
    p = np.poly1d(z)
    ax1.plot(df['sum_top5_absolute_deviations'], p(df['sum_top5_absolute_deviations']), 
             "r--", alpha=0.8, linewidth=2)
    
    ax1.set_xlabel('Sum of Top 5 Absolute Deviations (%)')
    ax1.set_ylabel('Group Average 365-Day Forward Change (%)')
    ax1.set_title(f'Predictive Power: Deviation vs Forward Returns\n(Correlation: {analysis["correlation_deviation_vs_forward_change"]:.3f})', 
                  fontweight='bold')
    ax1.grid(True, alpha=0.3)
    plt.colorbar(scatter, ax=ax1, label='Avg Max Gain (%)')
    
    # Plot 2: Performance by deviation magnitude bins
    ax2 = plt.subplot(2, 3, 2)
    bin_data = df.groupby('deviation_magnitude_bin').agg({
        'group_avg_forward_365_change': 'mean',
        'group_avg_forward_365_max_gain': 'mean'
    })
    
    x_pos = range(len(bin_data))
    width = 0.35
    
    bars1 = ax2.bar([x - width/2 for x in x_pos], bin_data['group_avg_forward_365_change'], 
                   width, label='Avg Forward Change', alpha=0.8, color='steelblue')
    bars2 = ax2.bar([x + width/2 for x in x_pos], bin_data['group_avg_forward_365_max_gain'], 
                   width, label='Avg Max Gain', alpha=0.8, color='orange')
    
    ax2.set_xlabel('Deviation Magnitude Quintile')
    ax2.set_ylabel('Forward Performance (%)')
    ax2.set_title('Performance by Deviation Magnitude Quintiles', fontweight='bold')
    ax2.set_xticks(x_pos)
    ax2.set_xticklabels(bin_data.index)
    ax2.legend()
    ax2.grid(True, alpha=0.3)
    
    # Add value labels on bars
    for bar in bars1:
        height = bar.get_height()
        ax2.text(bar.get_x() + bar.get_width()/2., height + 1,
                f'{height:.1f}%', ha='center', va='bottom', fontsize=9)
    
    for bar in bars2:
        height = bar.get_height()
        ax2.text(bar.get_x() + bar.get_width()/2., height + 1,
                f'{height:.1f}%', ha='center', va='bottom', fontsize=9)
    
    # Plot 3: Box plots of forward performance by quintile
    ax3 = plt.subplot(2, 3, 3)
    box_data = [df[df['deviation_magnitude_bin'] == quintile]['group_avg_forward_365_change'] 
                for quintile in ['Lowest', 'Low', 'Medium', 'High', 'Highest']]
    bp = ax3.boxplot(box_data, labels=['Lowest', 'Low', 'Medium', 'High', 'Highest'])
    ax3.set_xlabel('Deviation Magnitude Quintile')
    ax3.set_ylabel('365-Day Forward Change (%)')
    ax3.set_title('Forward Change Distribution by Quintile', fontweight='bold')
    ax3.grid(True, alpha=0.3)
    
    # Plot 4: Success rate (positive returns) by quintile
    ax4 = plt.subplot(2, 3, 4)
    success_by_bin = df.groupby('deviation_magnitude_bin').apply(
        lambda x: (x['group_avg_forward_365_change'] > 0).mean() * 100
    )
    
    bars = ax4.bar(range(len(success_by_bin)), success_by_bin.values, 
                   alpha=0.8, color='green')
    ax4.set_xlabel('Deviation Magnitude Quintile')
    ax4.set_ylabel('Success Rate (%)')
    ax4.set_title('Success Rate (Positive Returns) by Quintile', fontweight='bold')
    ax4.set_xticks(range(len(success_by_bin)))
    ax4.set_xticklabels(success_by_bin.index)
    ax4.grid(True, alpha=0.3)
    
    # Add value labels
    for bar, value in zip(bars, success_by_bin.values):
        ax4.text(bar.get_x() + bar.get_width()/2., bar.get_height() + 1,
                f'{value:.1f}%', ha='center', va='bottom', fontweight='bold')
    
    # Plot 5: Time series of deviation magnitude and forward performance
    ax5 = plt.subplot(2, 3, 5)
    window = 30
    df_sorted = df.sort_values('entry_date')
    
    rolling_deviation = df_sorted['sum_top5_absolute_deviations'].rolling(window=window, center=True).mean()
    rolling_forward_change = df_sorted['group_avg_forward_365_change'].rolling(window=window, center=True).mean()
    
    ax5_twin = ax5.twinx()
    
    line1 = ax5.plot(df_sorted['entry_date'], rolling_deviation, 
                     color='red', linewidth=2, label='Deviation Magnitude (30d avg)')
    line2 = ax5_twin.plot(df_sorted['entry_date'], rolling_forward_change, 
                         color='blue', linewidth=2, label='Forward Change (30d avg)')
    
    ax5.set_xlabel('Date')
    ax5.set_ylabel('Sum of Absolute Deviations (%)', color='red')
    ax5_twin.set_ylabel('Forward 365-Day Change (%)', color='blue')
    ax5.set_title('Time Series: Deviation Magnitude vs Forward Performance', fontweight='bold')
    
    # Combine legends
    lines = line1 + line2
    labels = [l.get_label() for l in lines]
    ax5.legend(lines, labels, loc='upper left')
    ax5.grid(True, alpha=0.3)
    
    # Plot 6: Risk-adjusted returns (Return/Volatility by quintile)
    ax6 = plt.subplot(2, 3, 6)
    risk_adj_returns = df.groupby('deviation_magnitude_bin').apply(
        lambda x: x['group_avg_forward_365_change'].mean() / x['group_avg_forward_365_change'].std()
        if x['group_avg_forward_365_change'].std() > 0 else 0
    )
    
    colors = ['red' if x < 0 else 'green' for x in risk_adj_returns.values]
    bars = ax6.bar(range(len(risk_adj_returns)), risk_adj_returns.values, 
                   alpha=0.8, color=colors)
    ax6.set_xlabel('Deviation Magnitude Quintile')
    ax6.set_ylabel('Risk-Adjusted Return (Return/Volatility)')
    ax6.set_title('Risk-Adjusted Returns by Quintile', fontweight='bold')
    ax6.set_xticks(range(len(risk_adj_returns)))
    ax6.set_xticklabels(risk_adj_returns.index)
    ax6.axhline(y=0, color='black', linestyle='-', alpha=0.5)
    ax6.grid(True, alpha=0.3)
    
    # Add value labels
    for bar, value in zip(bars, risk_adj_returns.values):
        ax6.text(bar.get_x() + bar.get_width()/2., 
                bar.get_height() + (0.01 if value >= 0 else -0.03),
                f'{value:.2f}', ha='center', 
                va='bottom' if value >= 0 else 'top', fontweight='bold')
    
    plt.tight_layout()
    plt.savefig(f"{output_dir}/predictive_power_analysis.png", dpi=300, bbox_inches='tight')
    plt.close()

def save_trading_analytics_results(daily_trading_results, analysis, output_dir):
    """
    Save all trading analytics results
    """
    # Save daily trading results
    daily_summary = []
    for result in daily_trading_results:
        daily_summary.append({
            'entry_date': result['entry_date'].strftime('%Y-%m-%d'),
            'sum_top5_absolute_deviations': result['sum_top5_absolute_deviations'],
            'num_assets_in_group': result['num_assets_in_group'],
            'group_avg_forward_365_change': result['group_avg_forward_365_change'],
            'group_avg_forward_365_max_gain': result['group_avg_forward_365_max_gain'],
            'group_median_forward_365_change': result['group_median_forward_365_change'],
            'group_median_forward_365_max_gain': result['group_median_forward_365_max_gain'],
            'group_std_forward_365_change': result['group_std_forward_365_change'],
            'group_std_forward_365_max_gain': result['group_std_forward_365_max_gain'],
            'group_avg_days_held': result['group_avg_days_held'],
            'top5_assets': ', '.join(result['top5_assets'])
        })
    
    daily_summary_df = pd.DataFrame(daily_summary)
    daily_summary_df.to_csv(f"{output_dir}/daily_trading_analytics.csv", index=False)
    
    # Save quintile analysis
    if 'bin_analysis' in analysis:
        analysis['bin_analysis'].to_csv(f"{output_dir}/quintile_analysis.csv")
    
    # Save individual trade results
    individual_trades = []
    for result in daily_trading_results:
        entry_date = result['entry_date']
        sum_deviation = result['sum_top5_absolute_deviations']
        
        for trade in result['individual_results']:
            individual_trades.append({
                'entry_date': entry_date.strftime('%Y-%m-%d'),
                'sum_group_deviation': sum_deviation,
                'asset': trade['asset'],
                'entry_price': trade['entry_price'],
                'final_price': trade['final_price'],
                'max_price': trade['max_price'],
                'days_held': trade['days_held'],
                'forward_365_change': trade['forward_365_change'],
                'forward_365_max_gain': trade['forward_365_max_gain'],
                'absolute_deviation': trade['absolute_deviation'],
                'raw_deviation': trade['raw_deviation']
            })
    
    individual_trades_df = pd.DataFrame(individual_trades)
    individual_trades_df.to_csv(f"{output_dir}/individual_trade_results.csv", index=False)
    
    return daily_summary_df, individual_trades_df

def print_trading_analytics_summary(daily_trading_results, analysis):
    """
    Print comprehensive summary of trading analytics
    """
    print("\n" + "="*80)
    print("ABSOLUTE DEVIATION TRADING ANALYTICS SUMMARY")
    print("="*80)
    
    print(f"\nTRADING STRATEGY:")
    print(f"  - Enter trade on each day with top 5 highest absolute deviation assets")
    print(f"  - Hold for 365 days (or until end of data)")
    print(f"  - Track group average forward performance")
    
    print(f"\nDATA OVERVIEW:")
    print(f"  Total trading entries: {analysis['total_trades']:,}")
    print(f"  Date range: {analysis['date_range'][0].strftime('%Y-%m-%d')} to {analysis['date_range'][1].strftime('%Y-%m-%d')}")
    
    print(f"\nOVERALL PERFORMANCE:")
    print(f"  Average 365-day forward change: {analysis['overall_avg_forward_change']:.2f}%")
    print(f"  Average 365-day max gain: {analysis['overall_avg_max_gain']:.2f}%")
    print(f"  Volatility (forward change): {analysis['overall_std_forward_change']:.2f}%")
    print(f"  Volatility (max gain): {analysis['overall_std_max_gain']:.2f}%")
    
    print(f"\nPREDICTIVE POWER ANALYSIS:")
    print(f"  Correlation (deviation magnitude vs forward change): {analysis['correlation_deviation_vs_forward_change']:.3f}")
    print(f"  Correlation (deviation magnitude vs max gain): {analysis['correlation_deviation_vs_max_gain']:.3f}")
    
    # Interpret correlations
    corr_change = analysis['correlation_deviation_vs_forward_change']
    corr_gain = analysis['correlation_deviation_vs_max_gain']
    
    print(f"\nCORRELATION INTERPRETATION:")
    if abs(corr_change) > 0.3:
        direction = "positive" if corr_change > 0 else "negative"
        strength = "strong" if abs(corr_change) > 0.5 else "moderate"
        print(f"  Forward Change: {strength} {direction} correlation")
    else:
        print(f"  Forward Change: weak correlation")
    
    if abs(corr_gain) > 0.3:
        direction = "positive" if corr_gain > 0 else "negative"
        strength = "strong" if abs(corr_gain) > 0.5 else "moderate"
        print(f"  Max Gain: {strength} {direction} correlation")
    else:
        print(f"  Max Gain: weak correlation")
    
    # Quintile analysis
    if 'bin_analysis' in analysis:
        print(f"\nQUINTILE ANALYSIS (by deviation magnitude):")
        print("-" * 60)
        
        bin_df = analysis['bin_analysis']
        
        # Extract quintile performance
        quintiles = bin_df.index.tolist()
        forward_changes = bin_df['group_avg_forward_365_change_mean'].tolist()
        max_gains = bin_df['group_avg_forward_365_max_gain_mean'].tolist()
        
        print(f"{'Quintile':<12} {'Avg Forward':<12} {'Avg Max Gain':<12} {'Trade Count':<12}")
        print("-" * 60)
        
        for i, quintile in enumerate(quintiles):
            forward_change = forward_changes[i]
            max_gain = max_gains[i]
            count = int(bin_df['group_avg_forward_365_change_count'].iloc[i])
            
            print(f"{quintile:<12} {forward_change:>10.1f}% {max_gain:>11.1f}% {count:>11,}")
        
        # Performance ranking
        best_forward_idx = np.argmax(forward_changes)
        best_gain_idx = np.argmax(max_gains)
        
        print(f"\nPERFORMANCE RANKING:")
        print(f"  Best forward returns: {quintiles[best_forward_idx]} quintile ({forward_changes[best_forward_idx]:.1f}%)")
        print(f"  Best max gains: {quintiles[best_gain_idx]} quintile ({max_gains[best_gain_idx]:.1f}%)")
    
    # Statistical significance
    if 't_test_high_vs_low' in analysis:
        t_test = analysis['t_test_high_vs_low']
        print(f"\nSTATISTICAL SIGNIFICANCE TEST:")
        print(f"  Highest vs Lowest quintile comparison:")
        print(f"  t-statistic: {t_test['t_statistic']:.3f}")
        print(f"  p-value: {t_test['p_value']:.6f}")
        print(f"  Statistically significant: {'Yes' if t_test['significant'] else 'No'}")
        
        if t_test['significant']:
            print(f"  -> There IS a statistically significant difference between high and low deviation periods")
        else:
            print(f"  -> There is NO statistically significant difference between high and low deviation periods")

def main():
    print("Absolute Deviation Trading Analytics")
    print("="*60)
    print("Testing predictive power of absolute deviation magnitude")
    print("for forward-looking asset performance")
    
    # Create output directory
    output_dir = "/Users/tim/IWLS-OPTIONS/TRADING_ANALYTICS"
    os.makedirs(output_dir, exist_ok=True)
    print(f"\nOutput directory: {output_dir}")
    
    # Load all IWLS results
    all_results = load_all_iwls_results()
    if not all_results:
        return
    
    print(f"\nLoaded {len(all_results)} assets")
    
    # Calculate daily trading analytics
    daily_trading_results = calculate_daily_trading_analytics(all_results)
    
    if not daily_trading_results:
        print("No trading data found for analysis!")
        return
    
    # Analyze predictive power
    analysis = analyze_predictive_power(daily_trading_results)
    
    # Create visualizations
    print(f"\nCreating visualizations...")
    create_predictive_power_visualizations(daily_trading_results, analysis, output_dir)
    
    # Save results
    daily_summary_df, individual_trades_df = save_trading_analytics_results(
        daily_trading_results, analysis, output_dir)
    
    # Print comprehensive summary
    print_trading_analytics_summary(daily_trading_results, analysis)
    
    print(f"\n" + "="*80)
    print("TRADING ANALYTICS COMPLETE")
    print("="*80)
    print("Files saved:")
    print("  - predictive_power_analysis.png (6-panel comprehensive analysis)")
    print("  - daily_trading_analytics.csv (daily group performance)")
    print("  - quintile_analysis.csv (performance by deviation magnitude)")
    print("  - individual_trade_results.csv (individual asset performance)")
    
    # Key insights
    print(f"\nKEY INSIGHTS:")
    if 'correlation_deviation_vs_forward_change' in analysis:
        corr = analysis['correlation_deviation_vs_forward_change']
        print(f"  Correlation between deviation magnitude and forward returns: {corr:.3f}")
        
        if abs(corr) > 0.2:
            direction = "higher" if corr > 0 else "lower"
            print(f"  -> Higher deviation magnitude tends to predict {direction} forward returns")
        else:
            print(f"  -> No strong relationship between deviation magnitude and forward returns")
    
    if 'bin_analysis' in analysis:
        bin_df = analysis['bin_analysis']
        best_quintile_idx = bin_df['group_avg_forward_365_change_mean'].idxmax()
        best_performance = bin_df['group_avg_forward_365_change_mean'].max()
        worst_performance = bin_df['group_avg_forward_365_change_mean'].min()
        
        print(f"  Best performing quintile: {best_quintile_idx} ({best_performance:.1f}% avg return)")
        print(f"  Performance spread: {best_performance - worst_performance:.1f}% between best and worst quintiles")
    
    print(f"\nTRADING IMPLICATIONS:")
    if analysis.get('correlation_deviation_vs_forward_change', 0) > 0.1:
        print(f"  - Higher deviation periods may offer better opportunities")
        print(f"  - Consider increasing position sizes during high deviation periods")
    elif analysis.get('correlation_deviation_vs_forward_change', 0) < -0.1:
        print(f"  - Higher deviation periods may be riskier")
        print(f"  - Consider reducing exposure during extreme deviation periods")
    else:
        print(f"  - Deviation magnitude appears to have limited predictive power")
        print(f"  - May need to explore other factors for timing decisions")
    
    print(f"\nRECOMMENDATIONS:")
    print(f"  1. Review individual asset performance patterns in detail")
    print(f"  2. Consider sector/market conditions during high deviation periods")
    print(f"  3. Test different holding periods (30, 90, 180 days)")
    print(f"  4. Analyze performance during different market regimes")
    print(f"  5. Consider risk-adjusted metrics for position sizing")

if __name__ == "__main__":
    main()