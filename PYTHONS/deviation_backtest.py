import pandas as pd
import numpy as np
import matplotlib.pyplot as plt
import matplotlib.dates as mdates
import os
import seaborn as sns
from datetime import datetime, timedelta
from collections import defaultdict
import warnings
warnings.filterwarnings('ignore')

def load_trading_analytics_results(analytics_dir):
    """
    Load the trading analytics results from previous analysis
    """
    daily_file = f"{analytics_dir}/daily_trading_analytics.csv"
    
    if not os.path.exists(daily_file):
        print(f"Trading analytics file not found: {daily_file}")
        print("Please run the trading analytics script first!")
        return None
    
    df = pd.read_csv(daily_file)
    df['entry_date'] = pd.to_datetime(df['entry_date'])
    
    print(f"Loaded {len(df)} trading entries from {df['entry_date'].min().strftime('%Y-%m-%d')} to {df['entry_date'].max().strftime('%Y-%m-%d')}")
    
    return df

def identify_highest_quintile_signals(df):
    """
    Identify which trading days fall into the highest quintile of deviation magnitude
    """
    # Calculate quintiles
    df['deviation_quintile'] = pd.qcut(df['sum_top5_absolute_deviations'], 
                                     q=5, 
                                     labels=['Lowest', 'Low', 'Medium', 'High', 'Highest'])
    
    # Get highest quintile signals
    highest_quintile = df[df['deviation_quintile'] == 'Highest'].copy()
    highest_quintile = highest_quintile.sort_values('entry_date')
    
    # Calculate threshold for highest quintile
    threshold = df['sum_top5_absolute_deviations'].quantile(0.8)
    
    print(f"\nHighest Quintile Analysis:")
    print(f"  Threshold (80th percentile): {threshold:.2f}%")
    print(f"  Number of highest quintile signals: {len(highest_quintile)}")
    print(f"  Percentage of all trading days: {len(highest_quintile)/len(df)*100:.1f}%")
    print(f"  Average performance: {highest_quintile['group_avg_forward_365_change'].mean():.2f}%")
    
    return highest_quintile, threshold

def analyze_signal_frequency(highest_quintile):
    """
    Analyze the frequency and timing patterns of highest quintile signals
    """
    # Time between signals
    time_diffs = highest_quintile['entry_date'].diff().dropna()
    time_diffs_days = time_diffs.dt.days
    
    # Monthly frequency
    highest_quintile['year_month'] = highest_quintile['entry_date'].dt.to_period('M')
    monthly_counts = highest_quintile.groupby('year_month').size()
    
    # Yearly frequency
    highest_quintile['year'] = highest_quintile['entry_date'].dt.year
    yearly_counts = highest_quintile.groupby('year').size()
    
    # Day of week patterns
    highest_quintile['day_of_week'] = highest_quintile['entry_date'].dt.day_name()
    dow_counts = highest_quintile.groupby('day_of_week').size()
    
    # Month of year patterns
    highest_quintile['month'] = highest_quintile['entry_date'].dt.month_name()
    month_counts = highest_quintile.groupby('month').size()
    
    frequency_analysis = {
        'total_signals': len(highest_quintile),
        'avg_days_between_signals': time_diffs_days.mean() if len(time_diffs_days) > 0 else 0,
        'median_days_between_signals': time_diffs_days.median() if len(time_diffs_days) > 0 else 0,
        'min_days_between_signals': time_diffs_days.min() if len(time_diffs_days) > 0 else 0,
        'max_days_between_signals': time_diffs_days.max() if len(time_diffs_days) > 0 else 0,
        'monthly_counts': monthly_counts,
        'yearly_counts': yearly_counts,
        'dow_counts': dow_counts,
        'month_counts': month_counts,
        'signals_per_year': len(highest_quintile) / len(yearly_counts) if len(yearly_counts) > 0 else 0
    }
    
    return frequency_analysis

def create_signal_timing_visualizations(df, highest_quintile, threshold, frequency_analysis, output_dir):
    """
    Create comprehensive visualizations showing signal timing and frequency
    """
    # Create large figure with multiple subplots
    fig = plt.figure(figsize=(24, 20))
    
    # Plot 1: Main time series with signal highlights (top panel, full width)
    ax1 = plt.subplot(4, 3, (1, 3))  # Span first row
    
    # Plot all deviation magnitudes
    ax1.plot(df['entry_date'], df['sum_top5_absolute_deviations'], 
             color='lightblue', alpha=0.6, linewidth=1, label='All Days')
    
    # Highlight highest quintile signals
    ax1.scatter(highest_quintile['entry_date'], 
               highest_quintile['sum_top5_absolute_deviations'],
               color='red', s=50, alpha=0.8, zorder=5, label='Highest Quintile Signals')
    
    # Add threshold line
    ax1.axhline(y=threshold, color='orange', linestyle='--', linewidth=2, 
                alpha=0.8, label=f'Highest Quintile Threshold ({threshold:.1f}%)')
    
    # Add rolling average
    window = 30
    rolling_avg = df['sum_top5_absolute_deviations'].rolling(window=window, center=True).mean()
    ax1.plot(df['entry_date'], rolling_avg, color='darkblue', linewidth=2, 
             alpha=0.8, label=f'{window}-Day Moving Average')
    
    ax1.set_title('Highest Quintile Trading Signals Over Time', fontweight='bold', fontsize=16)
    ax1.set_ylabel('Sum of Top 5 Absolute Deviations (%)')
    ax1.legend(loc='upper left')
    ax1.grid(True, alpha=0.3)
    ax1.xaxis.set_major_formatter(mdates.DateFormatter('%Y-%m'))
    ax1.xaxis.set_major_locator(mdates.YearLocator())
    plt.setp(ax1.xaxis.get_majorticklabels(), rotation=45)
    
    # Plot 2: Signal frequency by year
    ax2 = plt.subplot(4, 3, 4)
    years = frequency_analysis['yearly_counts'].index
    counts = frequency_analysis['yearly_counts'].values
    
    bars = ax2.bar(years, counts, alpha=0.7, color='steelblue', edgecolor='black')
    ax2.set_title('Signals per Year', fontweight='bold')
    ax2.set_ylabel('Number of Signals')
    ax2.set_xlabel('Year')
    ax2.grid(True, alpha=0.3)
    
    # Add value labels on bars
    for bar, count in zip(bars, counts):
        ax2.text(bar.get_x() + bar.get_width()/2., bar.get_height() + 0.1,
                f'{count}', ha='center', va='bottom', fontweight='bold')
    
    # Plot 3: Monthly frequency distribution
    ax3 = plt.subplot(4, 3, 5)
    monthly_data = frequency_analysis['monthly_counts']
    monthly_avg = monthly_data.groupby(monthly_data.index.month).mean()
    
    month_names = ['Jan', 'Feb', 'Mar', 'Apr', 'May', 'Jun',
                   'Jul', 'Aug', 'Sep', 'Oct', 'Nov', 'Dec']
    
    bars = ax3.bar(range(1, 13), [monthly_avg.get(i, 0) for i in range(1, 13)], 
                   alpha=0.7, color='green', edgecolor='black')
    ax3.set_title('Average Signals per Month', fontweight='bold')
    ax3.set_ylabel('Average Number of Signals')
    ax3.set_xlabel('Month')
    ax3.set_xticks(range(1, 13))
    ax3.set_xticklabels(month_names)
    ax3.grid(True, alpha=0.3)
    
    # Add value labels
    for i, bar in enumerate(bars):
        height = bar.get_height()
        if height > 0:
            ax3.text(bar.get_x() + bar.get_width()/2., height + 0.01,
                    f'{height:.1f}', ha='center', va='bottom', fontsize=9)
    
    # Plot 4: Day of week distribution
    ax4 = plt.subplot(4, 3, 6)
    dow_order = ['Monday', 'Tuesday', 'Wednesday', 'Thursday', 'Friday', 'Saturday', 'Sunday']
    dow_counts = [frequency_analysis['dow_counts'].get(day, 0) for day in dow_order]
    
    bars = ax4.bar(range(7), dow_counts, alpha=0.7, color='orange', edgecolor='black')
    ax4.set_title('Signals by Day of Week', fontweight='bold')
    ax4.set_ylabel('Number of Signals')
    ax4.set_xlabel('Day of Week')
    ax4.set_xticks(range(7))
    ax4.set_xticklabels([day[:3] for day in dow_order])
    ax4.grid(True, alpha=0.3)
    
    # Add value labels
    for bar, count in zip(bars, dow_counts):
        if count > 0:
            ax4.text(bar.get_x() + bar.get_width()/2., bar.get_height() + 0.1,
                    f'{count}', ha='center', va='bottom', fontweight='bold')
    
    # Plot 5: Time between signals histogram
    ax5 = plt.subplot(4, 3, 7)
    if len(highest_quintile) > 1:
        time_diffs = highest_quintile['entry_date'].diff().dropna()
        time_diffs_days = time_diffs.dt.days
        
        ax5.hist(time_diffs_days, bins=20, alpha=0.7, color='purple', edgecolor='black')
        ax5.axvline(time_diffs_days.mean(), color='red', linestyle='--', 
                   linewidth=2, label=f'Mean: {time_diffs_days.mean():.1f} days')
        ax5.axvline(time_diffs_days.median(), color='orange', linestyle='--', 
                   linewidth=2, label=f'Median: {time_diffs_days.median():.1f} days')
    
    ax5.set_title('Distribution of Days Between Signals', fontweight='bold')
    ax5.set_ylabel('Frequency')
    ax5.set_xlabel('Days Between Signals')
    ax5.legend()
    ax5.grid(True, alpha=0.3)
    
    # Plot 6: Performance vs signal strength
    ax6 = plt.subplot(4, 3, 8)
    scatter = ax6.scatter(highest_quintile['sum_top5_absolute_deviations'], 
                         highest_quintile['group_avg_forward_365_change'],
                         c=highest_quintile['group_avg_forward_365_max_gain'], 
                         cmap='viridis', alpha=0.7, s=60)
    
    # Add trend line
    if len(highest_quintile) > 1:
        z = np.polyfit(highest_quintile['sum_top5_absolute_deviations'], 
                      highest_quintile['group_avg_forward_365_change'], 1)
        p = np.poly1d(z)
        ax6.plot(highest_quintile['sum_top5_absolute_deviations'], 
                p(highest_quintile['sum_top5_absolute_deviations']), 
                "r--", alpha=0.8, linewidth=2)
    
    ax6.set_title('Signal Strength vs Performance', fontweight='bold')
    ax6.set_xlabel('Deviation Magnitude (%)')
    ax6.set_ylabel('365-Day Forward Change (%)')
    ax6.grid(True, alpha=0.3)
    plt.colorbar(scatter, ax=ax6, label='Max Gain (%)')
    
    # Plot 7: Cumulative signal count over time
    ax7 = plt.subplot(4, 3, 9)
    highest_quintile_sorted = highest_quintile.sort_values('entry_date')
    cumulative_count = range(1, len(highest_quintile_sorted) + 1)
    
    ax7.plot(highest_quintile_sorted['entry_date'], cumulative_count, 
             color='darkgreen', linewidth=3, marker='o', markersize=4)
    
    ax7.set_title('Cumulative Signal Count Over Time', fontweight='bold')
    ax7.set_ylabel('Cumulative Number of Signals')
    ax7.set_xlabel('Date')
    ax7.grid(True, alpha=0.3)
    ax7.xaxis.set_major_formatter(mdates.DateFormatter('%Y'))
    ax7.xaxis.set_major_locator(mdates.YearLocator())
    plt.setp(ax7.xaxis.get_majorticklabels(), rotation=45)
    
    # Plot 8: Signal intensity heatmap by year and month
    ax8 = plt.subplot(4, 3, 10)
    
    # Create year-month matrix
    pivot_data = highest_quintile.copy()
    pivot_data['year'] = pivot_data['entry_date'].dt.year
    pivot_data['month'] = pivot_data['entry_date'].dt.month
    
    monthly_matrix = pivot_data.groupby(['year', 'month']).size().unstack(fill_value=0)
    
    if len(monthly_matrix) > 0:
        sns.heatmap(monthly_matrix, annot=True, fmt='d', cmap='YlOrRd', 
                   ax=ax8, cbar_kws={'label': 'Number of Signals'})
        ax8.set_title('Signal Intensity Heatmap (Year vs Month)', fontweight='bold')
        ax8.set_ylabel('Year')
        ax8.set_xlabel('Month')
    
    # Plot 9: Rolling frequency (signals per quarter)
    ax9 = plt.subplot(4, 3, 11)
    
    # Resample to quarterly frequency
    quarterly_counts = highest_quintile.set_index('entry_date').resample('Q').size()
    
    ax9.plot(quarterly_counts.index, quarterly_counts.values, 
             color='brown', linewidth=2, marker='s', markersize=6)
    ax9.set_title('Quarterly Signal Frequency', fontweight='bold')
    ax9.set_ylabel('Signals per Quarter')
    ax9.set_xlabel('Date')
    ax9.grid(True, alpha=0.3)
    ax9.xaxis.set_major_formatter(mdates.DateFormatter('%Y'))
    ax9.xaxis.set_major_locator(mdates.YearLocator())
    plt.setp(ax9.xaxis.get_majorticklabels(), rotation=45)
    
    # Plot 10: Signal statistics summary
    ax10 = plt.subplot(4, 3, 12)
    ax10.axis('off')  # Turn off axis for text display
    
    # Create statistics text
    stats_text = f"""Signal Frequency Statistics:
    
Total Signals: {frequency_analysis['total_signals']:,}
Signals per Year: {frequency_analysis['signals_per_year']:.1f}
    
Time Between Signals:
• Average: {frequency_analysis['avg_days_between_signals']:.1f} days
• Median: {frequency_analysis['median_days_between_signals']:.1f} days
• Minimum: {frequency_analysis['min_days_between_signals']} days
• Maximum: {frequency_analysis['max_days_between_signals']} days

Performance:
• Avg Return: {highest_quintile['group_avg_forward_365_change'].mean():.1f}%
• Avg Max Gain: {highest_quintile['group_avg_forward_365_max_gain'].mean():.1f}%
• Success Rate: {(highest_quintile['group_avg_forward_365_change'] > 0).mean()*100:.1f}%

Most Active:
• Year: {frequency_analysis['yearly_counts'].idxmax()} ({frequency_analysis['yearly_counts'].max()} signals)
• Month: {frequency_analysis['month_counts'].idxmax()} ({frequency_analysis['month_counts'].max()} signals)
• Day: {frequency_analysis['dow_counts'].idxmax()} ({frequency_analysis['dow_counts'].max()} signals)"""
    
    ax10.text(0.05, 0.95, stats_text, transform=ax10.transAxes, 
              fontsize=12, verticalalignment='top', fontfamily='monospace',
              bbox=dict(boxstyle='round,pad=0.5', facecolor='lightgray', alpha=0.8))
    
    plt.tight_layout()
    plt.savefig(f"{output_dir}/signal_timing_analysis.png", dpi=300, bbox_inches='tight')
    plt.close()

def create_signal_calendar_heatmap(highest_quintile, output_dir):
    """
    Create a calendar heatmap showing signal occurrence
    """
    fig, ax = plt.subplots(figsize=(20, 12))
    
    # Create daily signal indicator
    date_range = pd.date_range(start=highest_quintile['entry_date'].min(), 
                              end=highest_quintile['entry_date'].max(), 
                              freq='D')
    
    signal_series = pd.Series(0, index=date_range)
    signal_dates = highest_quintile['entry_date'].dt.date
    
    for date in signal_dates:
        if date in signal_series.index.date:
            signal_series[signal_series.index.date == date] = 1
    
    # Create year-day matrix for heatmap
    signal_df = signal_series.reset_index()
    signal_df.columns = ['date', 'signal']
    signal_df['year'] = signal_df['date'].dt.year
    signal_df['day_of_year'] = signal_df['date'].dt.dayofyear
    
    # Create pivot table
    pivot_matrix = signal_df.pivot_table(values='signal', index='year', 
                                        columns='day_of_year', fill_value=0)
    
    # Create heatmap
    sns.heatmap(pivot_matrix, cmap='Reds', cbar_kws={'label': 'Signal (1=Yes, 0=No)'}, 
                ax=ax, xticklabels=False, yticklabels=True)
    
    ax.set_title('Signal Occurrence Calendar Heatmap', fontweight='bold', fontsize=16)
    ax.set_xlabel('Day of Year')
    ax.set_ylabel('Year')
    
    plt.tight_layout()
    plt.savefig(f"{output_dir}/signal_calendar_heatmap.png", dpi=300, bbox_inches='tight')
    plt.close()

def save_signal_analysis_results(highest_quintile, frequency_analysis, threshold, output_dir):
    """
    Save detailed signal analysis results
    """
    # Save highest quintile signals
    signal_summary = highest_quintile.copy()
    signal_summary['entry_date'] = signal_summary['entry_date'].dt.strftime('%Y-%m-%d')
    signal_summary.to_csv(f"{output_dir}/highest_quintile_signals.csv", index=False)
    
    # Save frequency analysis
    freq_summary = {
        'metric': ['total_signals', 'signals_per_year', 'avg_days_between_signals', 
                  'median_days_between_signals', 'min_days_between_signals', 
                  'max_days_between_signals', 'threshold_value'],
        'value': [frequency_analysis['total_signals'], 
                 frequency_analysis['signals_per_year'],
                 frequency_analysis['avg_days_between_signals'],
                 frequency_analysis['median_days_between_signals'],
                 frequency_analysis['min_days_between_signals'],
                 frequency_analysis['max_days_between_signals'],
                 threshold]
    }
    
    freq_df = pd.DataFrame(freq_summary)
    freq_df.to_csv(f"{output_dir}/signal_frequency_summary.csv", index=False)
    
    # Save monthly and yearly counts
    frequency_analysis['yearly_counts'].to_csv(f"{output_dir}/yearly_signal_counts.csv")
    frequency_analysis['monthly_counts'].to_csv(f"{output_dir}/monthly_signal_counts.csv")
    
    print(f"\nSaved signal analysis files:")
    print(f"  - highest_quintile_signals.csv ({len(highest_quintile)} signals)")
    print(f"  - signal_frequency_summary.csv")
    print(f"  - yearly_signal_counts.csv")
    print(f"  - monthly_signal_counts.csv")

def print_signal_timing_summary(highest_quintile, frequency_analysis, threshold):
    """
    Print comprehensive summary of signal timing analysis
    """
    print("\n" + "="*80)
    print("HIGHEST QUINTILE SIGNAL TIMING ANALYSIS")
    print("="*80)
    
    print(f"\nSIGNAL OVERVIEW:")
    print(f"  Threshold (80th percentile): {threshold:.2f}%")
    print(f"  Total signals identified: {frequency_analysis['total_signals']:,}")
    print(f"  Average signals per year: {frequency_analysis['signals_per_year']:.1f}")
    print(f"  Date range: {highest_quintile['entry_date'].min().strftime('%Y-%m-%d')} to {highest_quintile['entry_date'].max().strftime('%Y-%m-%d')}")
    
    print(f"\nSIGNAL FREQUENCY:")
    print(f"  Average days between signals: {frequency_analysis['avg_days_between_signals']:.1f}")
    print(f"  Median days between signals: {frequency_analysis['median_days_between_signals']:.1f}")
    print(f"  Shortest gap: {frequency_analysis['min_days_between_signals']} days")
    print(f"  Longest gap: {frequency_analysis['max_days_between_signals']} days")
    
    print(f"\nSEASONAL PATTERNS:")
    most_active_year = frequency_analysis['yearly_counts'].idxmax()
    most_active_month = frequency_analysis['month_counts'].idxmax()
    most_active_dow = frequency_analysis['dow_counts'].idxmax()
    
    print(f"  Most active year: {most_active_year} ({frequency_analysis['yearly_counts'].max()} signals)")
    print(f"  Most active month: {most_active_month} ({frequency_analysis['month_counts'].max()} signals)")
    print(f"  Most active day of week: {most_active_dow} ({frequency_analysis['dow_counts'].max()} signals)")
    
    print(f"\nPERFORMANCE OF SIGNALS:")
    avg_return = highest_quintile['group_avg_forward_365_change'].mean()
    avg_max_gain = highest_quintile['group_avg_forward_365_max_gain'].mean()
    success_rate = (highest_quintile['group_avg_forward_365_change'] > 0).mean() * 100
    
    print(f"  Average 365-day return: {avg_return:.2f}%")
    print(f"  Average max gain: {avg_max_gain:.2f}%")
    print(f"  Success rate (positive returns): {success_rate:.1f}%")
    
    # Best and worst signals
    best_signal = highest_quintile.loc[highest_quintile['group_avg_forward_365_change'].idxmax()]
    worst_signal = highest_quintile.loc[highest_quintile['group_avg_forward_365_change'].idxmin()]
    
    print(f"\nBEST & WORST SIGNALS:")
    print(f"  Best signal: {best_signal['entry_date'].strftime('%Y-%m-%d')} ({best_signal['group_avg_forward_365_change']:.1f}% return)")
    print(f"  Worst signal: {worst_signal['entry_date'].strftime('%Y-%m-%d')} ({worst_signal['group_avg_forward_365_change']:.1f}% return)")
    
    print(f"\nTRADING IMPLICATIONS:")
    if frequency_analysis['signals_per_year'] >= 12:
        print(f"  - Frequent signals: ~{frequency_analysis['signals_per_year']:.0f} opportunities per year")
        print(f"  - Consider systematic approach with position sizing")
    elif frequency_analysis['signals_per_year'] >= 6:
        print(f"  - Moderate frequency: ~{frequency_analysis['signals_per_year']:.0f} opportunities per year")
        print(f"  - Can be selective with best setups")
    else:
        print(f"  - Rare signals: only ~{frequency_analysis['signals_per_year']:.0f} opportunities per year")
        print(f"  - Each signal is potentially very valuable")
    
    if frequency_analysis['avg_days_between_signals'] < 30:
        print(f"  - Signals cluster frequently (avg {frequency_analysis['avg_days_between_signals']:.0f} days apart)")
        print(f"  - May need position management for overlapping trades")
    else:
        print(f"  - Signals well-spaced (avg {frequency_analysis['avg_days_between_signals']:.0f} days apart)")
        print(f"  - Allows for focused single-position approach")

def main():
    print("Highest Quintile Signal Timing Analysis")
    print("="*60)
    print("Analyzing when the best trading signals occur and their frequency")
    
    # Define directories
    analytics_dir = "/Users/tim/IWLS-OPTIONS/TRADING_ANALYTICS"
    output_dir = "/Users/tim/IWLS-OPTIONS/SIGNAL_TIMING_ANALYSIS"
    os.makedirs(output_dir, exist_ok=True)
    
    print(f"\nInput directory: {analytics_dir}")
    print(f"Output directory: {output_dir}")
    
    # Load trading analytics results
    df = load_trading_analytics_results(analytics_dir)
    if df is None:
        return
    
    # Identify highest quintile signals
    highest_quintile, threshold = identify_highest_quintile_signals(df)
    
    if len(highest_quintile) == 0:
        print("No highest quintile signals found!")
        return
    
    # Analyze signal frequency and timing
    frequency_analysis = analyze_signal_frequency(highest_quintile)
    
    # Create visualizations
    print(f"\nCreating visualizations...")
    create_signal_timing_visualizations(df, highest_quintile, threshold, frequency_analysis, output_dir)
    create_signal_calendar_heatmap(highest_quintile, output_dir)
    
    # Save results
    save_signal_analysis_results(highest_quintile, frequency_analysis, threshold, output_dir)
    
    # Print comprehensive summary
    print_signal_timing_summary(highest_quintile, frequency_analysis, threshold)
    
    print(f"\n" + "="*80)
    print("SIGNAL TIMING ANALYSIS COMPLETE")
    print("="*80)
    print("Files created:")
    print("  - signal_timing_analysis.png (12-panel comprehensive analysis)")
    print("  - signal_calendar_heatmap.png (yearly calendar view)")
    print("  - highest_quintile_signals.csv (detailed signal data)")
    print("  - signal_frequency_summary.csv (frequency statistics)")
    print("  - yearly_signal_counts.csv & monthly_signal_counts.csv")

if __name__ == "__main__":
    main()