import pandas as pd
import numpy as np
from pathlib import Path


def downsample_data(input_file, output_file=None, original_freq=320, target_freq=60):
    """
    Downsample a CSV file from original_freq Hz to target_freq Hz.
    
    Parameters:
    -----------
    input_file : str
        Path to the input CSV file
    output_file : str, optional
        Path to the output CSV file. If None, adds '_downsampled' to the input filename
    original_freq : int
        Original sampling frequency in Hz (default: 320)
    target_freq : int
        Target sampling frequency in Hz (default: 60)
    """
    
    # Read the CSV file
    print(f"Reading file: {input_file}")
    df = pd.read_csv(input_file)
    
    # Calculate the downsample factor
    downsample_factor = original_freq / target_freq
    print(f"Downsample factor: {downsample_factor:.2f} (from {original_freq}Hz to {target_freq}Hz)")
    
    # Method 1: Simple decimation (taking every nth sample)
    # This is the simplest approach
    step = int(np.round(downsample_factor))
    df_downsampled = df.iloc[::step].reset_index(drop=True)
    
    print(f"Original samples: {len(df)}")
    print(f"Downsampled samples: {len(df_downsampled)}")
    print(f"Actual frequency: {original_freq / step:.2f}Hz")
    
    # Generate output filename if not provided
    if output_file is None:
        input_path = Path(input_file)
        output_file = input_path.parent / f"{input_path.stem}_downsampled{input_path.suffix}"
    
    # Save the downsampled data
    df_downsampled.to_csv(output_file, index=False)
    print(f"Downsampled data saved to: {output_file}")
    
    return df_downsampled


def downsample_directory(input_dir, output_dir=None, original_freq=320, target_freq=60, pattern="*.csv"):
    """
    Downsample all CSV files in a directory.
    
    Parameters:
    -----------
    input_dir : str
        Path to the input directory
    output_dir : str, optional
        Path to the output directory. If None, creates a 'downsampled' subdirectory
    original_freq : int
        Original sampling frequency in Hz (default: 320)
    target_freq : int
        Target sampling frequency in Hz (default: 60)
    pattern : str
        File pattern to match (default: "*.csv")
    """
    
    input_path = Path(input_dir)
    
    if output_dir is None:
        output_path = input_path / "downsampled"
    else:
        output_path = Path(output_dir)
    
    output_path.mkdir(exist_ok=True, parents=True)
    
    # Find all CSV files
    csv_files = list(input_path.glob(pattern))
    print(f"Found {len(csv_files)} files matching pattern '{pattern}'")
    
    for csv_file in csv_files:
        output_file = output_path / csv_file.name
        print(f"\n--- Processing: {csv_file.name} ---")
        try:
            downsample_data(csv_file, output_file, original_freq, target_freq)
        except Exception as e:
            print(f"Error processing {csv_file.name}: {e}")
    
    print(f"\nAll files processed. Output saved to: {output_path}")


if __name__ == "__main__":
    # ============= CONFIGURAZIONE =============
    # Modalità: 'file' per un singolo file, 'directory' per una cartella intera
    MODE = 'directory'
    
    # Frequenze
    ORIGINAL_FREQ = 320  # Hz
    TARGET_FREQ = 60     # Hz

    # Per modalità 'directory'
    INPUT_DIR = "../Labelled/1st"
    OUTPUT_DIR = None  # None per creare sottocartella 'downsampled'
    FILE_PATTERN = "*.csv"
    # ==========================================
    
    downsample_directory(INPUT_DIR, OUTPUT_DIR, ORIGINAL_FREQ, TARGET_FREQ, FILE_PATTERN)
