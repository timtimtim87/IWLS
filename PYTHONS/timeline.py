import pandas as pd
import numpy as np
import matplotlib.pyplot as plt
import matplotlib.dates as mdates
from datetime import datetime, timedelta
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
            df = df.dropna()
            df['price_deviation'] = ((df['price'] / df['trend_line_value']) - 1) * 100
            all_results[asset_name] = df
            print(f"Loaded {asset_name}: {len(df)} data points")
        except Exception as e:
            print(f"Error loading {asset_name}: {str(e)}")
    
    return all_results

def identify_trading_opportunities(all_results, deviation_threshold=-40):
    """
    Identify periods when assets are below the deviation threshold
    """
    opportunities = {}
    
    for asset_name, df in all_results.items():
        # Find periods where asset is below threshold
        below_threshold = df[df['price_deviation'] <= deviation_threshold].copy()
        
        if len(below_threshold) == 0:
            opportunities[asset_name] = pd.DataFrame()
            continue
        
        # Group consecutive periods
        below_threshold['group'] = (below_threshold['date'].diff() > pd.Timedelta(days=7)).cumsum()
        
        # Create opportunity periods
        opportunity_periods = []
        for group_id in below_threshold['group'].unique():
            group_data = below_threshold[below_threshold['group'] == group_id]
            
            opportunity_periods.append({
                'start_date': group_data['date'].min(),
                'end_date': group_data['date'].max(),
                'duration_days': (group_data['date'].max() - group_data['date'].min()).days,
                'min_deviation': group_data['price_deviation'].min(),
                'avg_deviation': group_data['price_deviation'].mean(),
                'start_price': group_data.iloc[0]['price'],
                'min_price': group_data['price'].min(),
                'data_points': len(group_data)
            })
        
        opportunities[asset_name] = pd.DataFrame(opportunity_periods)
        
        if len(opportunity_periods) > 0:
            print(f"{asset_name}: {len(opportunity_periods)} opportunity periods")
    
    return opportunities

def create_timeline_visualization(all_results, opportunities, deviation_threshold=-40, output_dir=None):
    """
    Create a comprehensive timeline visualization showing trading opportunities
    """
    # Get date range
    all_dates = []
    for df in all_results.values():
        all_dates.extend(df['date'].tolist())
    
    start_date = min(all_dates)
    end_date = max(all_dates)
    
    # Filter assets that have opportunities
    assets_with_opportunities = [asset for asset, opps in opportunities.items() if len(opps) > 0]
    
    if not assets_with_opportunities:
        print("No assets found with opportunities below threshold")
        return
    
    # Sort assets by total opportunity duration
    asset_durations = {}
    for asset in assets_with_opportunities:
        total_duration = opportunities[asset]['duration_days'].sum()
        asset_durations[asset] = total_duration
    
    sorted_assets = sorted(asset_durations.items(), key=lambda x: x[1], reverse=True)
    assets_with_opportunities = [asset for asset, _ in sorted_assets]
    
    print(f"\nFound {len(assets_with_opportunities)} assets with opportunities below {deviation_threshold}%")
    
    # Create figure
    fig, axes = plt.subplots(len(assets_with_opportunities), 1, 
                            figsize=(20, 2.5 * len(assets_with_opportunities)), 
                            sharex=True)
    
    if len(assets_with_opportunities) == 1:
        axes = [axes]
    
    for idx, asset in enumerate(assets_with_opportunities):
        ax = axes[idx]
        df = all_results[asset]
        
        # Plot price deviation over time
        ax.plot(df['date'], df['price_deviation'], color='lightblue', alpha=0.7, linewidth=1)
        
        # Highlight threshold line
        ax.axhline(y=deviation_threshold, color='red', linestyle='--', alpha=0.8, linewidth=2)
        ax.axhline(y=0, color='black', linestyle='-', alpha=0.5, linewidth=1)
        
        # Highlight opportunity periods
        for _, opp in opportunities[asset].iterrows():
            # Create a mask for the opportunity period
            mask = (df['date'] >= opp['start_date']) & (df['date'] <= opp['end_date'])
            opportunity_data = df[mask]
            
            if len(opportunity_data) > 0:
                # Fill the opportunity area
                ax.fill_between(opportunity_data['date'], 
                              opportunity_data['price_deviation'], 
                              deviation_threshold,
                              where=(opportunity_data['price_deviation'] <= deviation_threshold),
                              color='red', alpha=0.3)
                
                # Add opportunity markers
                ax.scatter([opp['start_date']], [opp['min_deviation']], 
                          color='red', s=100, marker='v', zorder=5,
                          label='Opportunity Start' if idx == 0 else "")
                
                # Add text annotation for significant opportunities
                if opp['duration_days'] > 30 or opp['min_deviation'] < -50:
                    ax.annotate(f"{opp['min_deviation']:.1f}%\n({opp['duration_days']}d)", 
                              xy=(opp['start_date'], opp['min_deviation']),
                              xytext=(10, 10), textcoords='offset points',
                              fontsize=8, ha='left',
                              bbox=dict(boxstyle='round,pad=0.3', facecolor='yellow', alpha=0.7))
        
        # Customize axis
        ax.set_ylabel(f'{asset}\nDeviation (%)', fontweight='bold')
        ax.grid(True, alpha=0.3)
        ax.set_ylim(min(df['price_deviation'].min() - 5, deviation_threshold - 10), 
                   max(df['price_deviation'].max() + 5, 20))
        
        # Color coding for different deviation ranges
        above_40 = df[df['price_deviation'] > 40]
        if len(above_40) > 0:
            ax.scatter(above_40['date'], above_40['price_deviation'], 
                      color='darkred', s=10, alpha=0.6)
        
        # Add summary statistics as text
        total_opp_days = opportunities[asset]['duration_days'].sum()
        num_opportunities = len(opportunities[asset])
        deepest_deviation = opportunities[asset]['min_deviation'].min() if num_opportunities > 0 else 0
        
        ax.text(0.02, 0.95, f'Opportunities: {num_opportunities}\nTotal Days: {total_opp_days}\nDeepest: {deepest_deviation:.1f}%', 
                transform=ax.transAxes, fontsize=9, verticalalignment='top',
                bbox=dict(boxstyle='round', facecolor='white', alpha=0.8))
    
    # Format x-axis
    axes[-1].xaxis.set_major_formatter(mdates.DateFormatter('%Y-%m'))
    axes[-1].xaxis.set_major_locator(mdates.YearLocator())
    axes[-1].xaxis.set_minor_locator(mdates.MonthLocator((1, 7)))
    
    # Add overall title and legend
    fig.suptitle(f'Trading Opportunities Timeline: Assets Below {deviation_threshold}% IWLS Deviation', 
                 fontsize=16, fontweight='bold', y=0.995)
    
    # Add legend to first subplot
    if len(assets_with_opportunities) > 0:
        axes[0].legend(loc='upper right')
    
    plt.xlabel('Date', fontweight='bold')
    plt.tight_layout()
    
    # Save plot
    if output_dir:
        plot_path = f"{output_dir}/trading_opportunities_timeline.png"
        plt.savefig(plot_path, dpi=300, bbox_inches='tight')
        print(f"\nTimeline plot saved to: {plot_path}")
    
    plt.show()

def create_opportunity_summary_table(opportunities, deviation_threshold=-40):
    """
    Create a summary table of all trading opportunities
    """
    summary_data = []
    
    for asset_name, asset_opportunities in opportunities.items():
        if len(asset_opportunities) == 0:
            continue
        
        for _, opp in asset_opportunities.iterrows():
            summary_data.append({
                'Asset': asset_name,
                'Start_Date': opp['start_date'].strftime('%Y-%m-%d'),
                'End_Date': opp['end_date'].strftime('%Y-%m-%d'),
                'Duration_Days': opp['duration_days'],
                'Min_Deviation': opp['min_deviation'],
                'Avg_Deviation': opp['avg_deviation'],
                'Start_Price': opp['start_price'],
                'Min_Price': opp['min_price'],
                'Price_Drop': ((opp['min_price'] / opp['start_price']) - 1) * 100
            })
    
    if not summary_data:
        print(f"No opportunities found below {deviation_threshold}%")
        return pd.DataFrame()
    
    summary_df = pd.DataFrame(summary_data)
    
    # Sort by minimum deviation (most extreme first)
    summary_df = summary_df.sort_values(['Min_Deviation', 'Duration_Days'], ascending=[True, False])
    
    return summary_df

def analyze_opportunity_statistics(opportunities, all_results):
    """
    Analyze statistics about the trading opportunities
    """
    print("\n" + "="*60)
    print("TRADING OPPORTUNITY ANALYSIS")
    print("="*60)
    
    # Overall statistics
    total_opportunities = sum(len(opps) for opps in opportunities.values())
    assets_with_opps = len([asset for asset, opps in opportunities.items() if len(opps) > 0])
    
    print(f"\nOVERALL STATISTICS:")
    print(f"Total opportunities found: {total_opportunities}")
    print(f"Assets with opportunities: {assets_with_opps}/{len(opportunities)}")
    
    if total_opportunities == 0:
        return
    
    # Duration analysis
    all_durations = []
    all_min_deviations = []
    
    for asset_opportunities in opportunities.values():
        if len(asset_opportunities) > 0:
            all_durations.extend(asset_opportunities['duration_days'].tolist())
            all_min_deviations.extend(asset_opportunities['min_deviation'].tolist())
    
    print(f"\nDURATION ANALYSIS:")
    print(f"Average opportunity duration: {np.mean(all_durations):.1f} days")
    print(f"Median opportunity duration: {np.median(all_durations):.1f} days")
    print(f"Longest opportunity: {max(all_durations)} days")
    print(f"Shortest opportunity: {min(all_durations)} days")
    
    print(f"\nDEVIATION ANALYSIS:")
    print(f"Average minimum deviation: {np.mean(all_min_deviations):.1f}%")
    print(f"Deepest deviation found: {min(all_min_deviations):.1f}%")
    print(f"Shallowest opportunity: {max(all_min_deviations):.1f}%")
    
    # Asset ranking
    print(f"\nTOP ASSETS BY OPPORTUNITY FREQUENCY:")
    asset_counts = {}
    asset_total_days = {}
    
    for asset_name, asset_opportunities in opportunities.items():
        if len(asset_opportunities) > 0:
            asset_counts[asset_name] = len(asset_opportunities)
            asset_total_days[asset_name] = asset_opportunities['duration_days'].sum()
    
    # Sort by frequency
    sorted_by_count = sorted(asset_counts.items(), key=lambda x: x[1], reverse=True)
    
    print(f"{'Asset':<8} {'Count':<6} {'Total Days':<12} {'Avg Duration':<12}")
    print("-" * 40)
    
    for asset, count in sorted_by_count[:10]:  # Top 10
        total_days = asset_total_days[asset]
        avg_duration = total_days / count
        print(f"{asset:<8} {count:<6} {total_days:<12} {avg_duration:<12.1f}")

def save_opportunity_data(opportunities, summary_df, output_dir):
    """
    Save opportunity data to CSV files
    """
    if output_dir and len(summary_df) > 0:
        # Save summary table
        summary_path = f"{output_dir}/trading_opportunities_summary.csv"
        summary_df.to_csv(summary_path, index=False)
        print(f"Opportunity summary saved to: {summary_path}")
        
        # Save detailed data for each asset
        detailed_path = f"{output_dir}/trading_opportunities_detailed.csv"
        detailed_data = []
        
        for asset_name, asset_opportunities in opportunities.items():
            for _, opp in asset_opportunities.iterrows():
                detailed_data.append({
                    'asset': asset_name,
                    'start_date': opp['start_date'],
                    'end_date': opp['end_date'],
                    'duration_days': opp['duration_days'],
                    'min_deviation': opp['min_deviation'],
                    'avg_deviation': opp['avg_deviation'],
                    'start_price': opp['start_price'],
                    'min_price': opp['min_price'],
                    'data_points': opp['data_points']
                })
        
        if detailed_data:
            detailed_df = pd.DataFrame(detailed_data)
            detailed_df.to_csv(detailed_path, index=False)
            print(f"Detailed opportunity data saved to: {detailed_path}")

def main():
    print("Trading Opportunities Timeline Analysis")
    print("="*50)
    print("Analyzing periods when assets are >40% below IWLS trend line")
    
    # Create output directory
    output_dir = "/Users/tim/IWLS-OPTIONS/TRADING_OPPORTUNITIES"
    os.makedirs(output_dir, exist_ok=True)
    print(f"\nOutput directory: {output_dir}")
    
    # Load IWLS results
    all_results = load_all_iwls_results()
    if not all_results:
        return
    
    print(f"\nLoaded data for {len(all_results)} assets")
    
    # Set deviation threshold (can be adjusted)
    deviation_threshold = -40  # -40% below trend line
    
    # Identify trading opportunities
    print(f"\nIdentifying opportunities below {deviation_threshold}% deviation...")
    opportunities = identify_trading_opportunities(all_results, deviation_threshold)
    
    # Create summary table
    summary_df = create_opportunity_summary_table(opportunities, deviation_threshold)
    
    # Analyze opportunity statistics
    analyze_opportunity_statistics(opportunities, all_results)
    
    # Create timeline visualization
    print(f"\nCreating timeline visualization...")
    create_timeline_visualization(all_results, opportunities, deviation_threshold, output_dir)
    
    # Save data
    save_opportunity_data(opportunities, summary_df, output_dir)
    
    # Print top opportunities
    if len(summary_df) > 0:
        print(f"\nTOP 10 MOST EXTREME OPPORTUNITIES:")
        print("-" * 80)
        print(summary_df[['Asset', 'Start_Date', 'Duration_Days', 'Min_Deviation', 'Price_Drop']].head(10).to_string(index=False))
    
    print(f"\n" + "="*60)
    print("TRADING OPPORTUNITIES ANALYSIS COMPLETE")
    print("="*60)
    print("Files saved:")
    print("  - trading_opportunities_timeline.png (timeline visualization)")
    print("  - trading_opportunities_summary.csv (opportunity summary)")
    print("  - trading_opportunities_detailed.csv (detailed data)")

if __name__ == "__main__":
    main()