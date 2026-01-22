import os
import pandas as pd
from pathlib import Path
import re

# Define paths
base_path = Path(__file__).parent.parent
processed_data_path = base_path / "ProcessedData"
labelled_path = base_path / "Labelled"
labelled_csv_path = base_path / "LabelledCSV"

# Create output directory if it doesn't exist
labelled_csv_path.mkdir(exist_ok=True)

def parse_label_file(txt_file_path):
    """
    Parse a label file and return a dictionary with punchType numbers for each frame range.
    Format: PUNCHTYPE:n followed by frame ranges fi-ff
    """
    punch_ranges = {}
    
    try:
        with open(txt_file_path, 'r') as f:
            lines = f.readlines()
    except Exception as e:
        print(f"Error reading {txt_file_path}: {e}")
        return punch_ranges
    
    current_punch_type = None
    
    for line in lines:
        line = line.strip()
        
        if not line:
            continue
        
        # Check if line contains punch type (e.g., "LEFT:1", "RIGHT:2")
        if ':' in line and not '-' in line:
            # Extract punch type number
            parts = line.split(':')
            if len(parts) == 2 and parts[1].isdigit():
                current_punch_type = int(parts[1])
        # Check if line contains frame range (e.g., "15965-16185")
        elif '-' in line and current_punch_type is not None:
            try:
                frame_parts = line.split('-')
                if len(frame_parts) == 2:
                    start_frame = int(frame_parts[0])
                    end_frame = int(frame_parts[1])
                    # Store the range with its punch type
                    punch_ranges[(start_frame, end_frame)] = current_punch_type
            except ValueError:
                continue
    
    return punch_ranges

def get_punch_type_for_frame(frame_num, punch_ranges):
    """
    Get the punch type for a given frame number.
    Returns the punch type if frame is within a range, otherwise returns 0.
    """
    for (start, end), punch_type in punch_ranges.items():
        if start <= frame_num <= end:
            return punch_type
    return 0

def find_matching_label_file(csv_filename):
    """
    Find the matching label file for a CSV file.
    Tries to match based on similar naming patterns.
    """
    # Try exact match first (without _accel.csv extension)
    csv_base = csv_filename.replace('_accel.csv', '')
    csv_base = csv_base.replace('_accel_LeftHand_plot.html', '')
    csv_base = csv_base.replace('_accel_RightHand_plot.html', '')
    
    for label_file in labelled_path.glob('*.txt'):
        if label_file.name == 'template':
            continue
        
        label_base = label_file.stem  # filename without extension
        
        # Check for exact match
        if csv_base == label_base:
            return label_file
        
        # Check for similar patterns (handle different naming conventions)
        # Remove timestamps and file format indicators
        csv_normalized = re.sub(r'_accel.*', '', csv_filename)
        label_normalized = label_file.stem
        
        if csv_normalized == label_normalized:
            return label_file
    
    return None

def process_csv_files():
    """
    Process all CSV files in ProcessedData and create new files in LabelledCSV with punchType column.
    """
    csv_files = list(processed_data_path.glob('*_accel.csv'))
    
    print(f"Found {len(csv_files)} CSV files in ProcessedData")
    
    for csv_file in csv_files:
        csv_filename = csv_file.name
        print(f"\nProcessing: {csv_filename}")
        
        # Find matching label file
        label_file = find_matching_label_file(csv_filename)
        
        if label_file is None:
            print(f"  WARNING: No matching label file found for {csv_filename}")
            # Still process it, but with all zeros for punchType
            punch_ranges = {}
        else:
            print(f"  Matched with: {label_file.name}")
            punch_ranges = parse_label_file(label_file)
            print(f"  Found {len(punch_ranges)} punch ranges")
        
        # Read CSV file
        try:
            df = pd.read_csv(csv_file)
        except Exception as e:
            print(f"  ERROR reading CSV: {e}")
            continue
        
        # Add punchType column
        df['punchType'] = df['Frame'].apply(lambda frame: get_punch_type_for_frame(frame, punch_ranges))
        
        # Create output filename (same as input)
        output_file = labelled_csv_path / csv_filename
        
        # Save to LabelledCSV
        try:
            df.to_csv(output_file, index=False)
            print(f"  Successfully saved to: {output_file}")
            print(f"  Shape: {df.shape[0]} rows x {df.shape[1]} columns")
        except Exception as e:
            print(f"  ERROR saving CSV: {e}")

if __name__ == "__main__":
    print("Starting CSV processing with punchType column addition...")
    print(f"Source: {processed_data_path}")
    print(f"Labels: {labelled_path}")
    print(f"Output: {labelled_csv_path}")
    print("=" * 80)
    
    process_csv_files()
    
    print("\n" + "=" * 80)
    print("Processing complete!")
