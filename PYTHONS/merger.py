import pandas as pd
import os

def clean_column_names(columns):
    """Clean column names by removing exchange info and keeping just stock symbols"""
    cleaned = []
    for col in columns:
        if col == 'time':
            cleaned.append('time')
        elif 'close' in col:
            # Extract stock symbol (everything before ' ·')
            if ' ·' in col:
                symbol = col.split(' ·')[0].replace('close', '').strip()
                # Handle special cases with ** markers
                if symbol.startswith('**'):
                    symbol = symbol[2:]  # Remove ** prefix
                if symbol.endswith('**'):
                    symbol = symbol[:-2]  # Remove ** suffix
                cleaned.append(symbol)
            else:
                cleaned.append(col)
        else:
            cleaned.append(col)
    return cleaned

def merge_stock_data(spy_file, axp_file, output_file):
    """Merge SPY and AXP stock data files with overlapping time periods"""
    
    # Read the CSV files
    print(f"Reading {spy_file}...")
    spy_df = pd.read_csv(spy_file)
    
    print(f"Reading {axp_file}...")
    axp_df = pd.read_csv(axp_file)
    
    # Clean column names for both dataframes
    spy_df.columns = clean_column_names(spy_df.columns)
    axp_df.columns = clean_column_names(axp_df.columns)
    
    # Rename the first 'close' column to 'SPY' and 'AXP' respectively
    if 'close' in spy_df.columns:
        spy_df = spy_df.rename(columns={'close': 'SPY'})
    
    if 'close' in axp_df.columns:
        axp_df = axp_df.rename(columns={'close': 'AXP'})
    
    # Find overlapping time period
    spy_start, spy_end = spy_df['time'].min(), spy_df['time'].max()
    axp_start, axp_end = axp_df['time'].min(), axp_df['time'].max()
    
    # Calculate the overlapping period
    overlap_start = max(spy_start, axp_start)
    overlap_end = min(spy_end, axp_end)
    
    print(f"SPY data range: {spy_start} to {spy_end}")
    print(f"AXP data range: {axp_start} to {axp_end}")
    print(f"Overlapping period: {overlap_start} to {overlap_end}")
    
    # Convert timestamps to readable dates for display
    from datetime import datetime
    print(f"SPY dates: {datetime.fromtimestamp(spy_start)} to {datetime.fromtimestamp(spy_end)}")
    print(f"AXP dates: {datetime.fromtimestamp(axp_start)} to {datetime.fromtimestamp(axp_end)}")
    print(f"Overlap dates: {datetime.fromtimestamp(overlap_start)} to {datetime.fromtimestamp(overlap_end)}")
    
    # Trim both dataframes to overlapping period
    spy_trimmed = spy_df[(spy_df['time'] >= overlap_start) & (spy_df['time'] <= overlap_end)]
    axp_trimmed = axp_df[(axp_df['time'] >= overlap_start) & (axp_df['time'] <= overlap_end)]
    
    print(f"After trimming - SPY: {len(spy_trimmed)} records, AXP: {len(axp_trimmed)} records")
    
    # Merge dataframes on 'time' column using inner join (only matching timestamps)
    print("Merging dataframes...")
    merged_df = pd.merge(spy_trimmed, axp_trimmed, on='time', how='inner', suffixes=('', '_duplicate'))
    
    # Remove any duplicate columns that might have been created
    duplicate_cols = [col for col in merged_df.columns if col.endswith('_duplicate')]
    if duplicate_cols:
        print(f"Removing duplicate columns: {duplicate_cols}")
        merged_df = merged_df.drop(columns=duplicate_cols)
    
    # Sort by time to ensure chronological order
    merged_df = merged_df.sort_values('time')
    
    # Save the merged dataframe
    print(f"Saving merged data to {output_file}...")
    merged_df.to_csv(output_file, index=False)
    
    print(f"Successfully merged data!")
    print(f"Output file contains {len(merged_df)} records")
    print(f"Columns in merged file: {list(merged_df.columns)}")
    
    return merged_df

if __name__ == "__main__":
    # File paths
    spy_file = "/Users/tim/IWLS-OPTIONS/ASSETS_SPY.csv"
    axp_file = "/Users/tim/IWLS-OPTIONS/ASSETS_AXP.csv"
    output_file = "/Users/tim/IWLS-OPTIONS/MERGED_STOCK_DATA.csv"
    
    # Check if input files exist
    if not os.path.exists(spy_file):
        print(f"Error: SPY file not found at {spy_file}")
        exit(1)
    
    if not os.path.exists(axp_file):
        print(f"Error: AXP file not found at {axp_file}")
        exit(1)
    
    # Merge the data
    try:
        merged_data = merge_stock_data(spy_file, axp_file, output_file)
        print(f"\nMerge completed successfully!")
        print(f"Output saved to: {output_file}")
        
        # Display first few rows as preview
        print("\nPreview of merged data:")
        print(merged_data.head())
        
    except Exception as e:
        print(f"Error occurred during merge: {str(e)}")
        exit(1)