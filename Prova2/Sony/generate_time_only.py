import pandas as pd
import os

csv_files = ['Davide Take 1/QM-SS1_156F9_20260114-171158.csv',
             'Davide Take 1/QM-SS1_1C2B5_20260114-171158.csv',
             'Davide Take 1/QM-SS1_1875A_20260114-171158.csv',
             'Davide Take 1/QM-SS1_1C79A_20260114-171158.csv',
             'Davide Take 1/QM-SS1_1A122_20260114-171158.csv',
             'Davide Take 1/QM-SS1_1D19C_20260114-171158.csv']

CUTOFF = pd.to_datetime("2026-01-14 17:05:56.441")

for csv_file in csv_files:
    out_file = os.path.splitext(csv_file)[0] + "_time.csv"

    # Read CSV robustly, when encountering bad lines, just read the number of columns expected
    df = pd.read_csv(csv_file, on_bad_lines='skip')

    # Parse timestamps
    df['Timestamp'] = pd.to_datetime(df['Timestamp'], errors='coerce')
    # Drop rows with invalid timestamps
    df = df.dropna(subset=['Timestamp']).reset_index(drop=True)

    # Drop rows with a timestamp earlier than CUTOFF
    df = df[df['Timestamp'] >= CUTOFF].reset_index(drop=True)

    # Compute TimeOnly as elapsed seconds from first timestamp
    start = df['Timestamp'].iloc[0]
    # elapsed seconds
    df['Timestamp'] = (df['Timestamp'] - start).dt.total_seconds()
    # Format to 3 decimal places (milliseconds)
    df['Timestamp'] = df['Timestamp'].map(lambda x: f"{x:.3f}")

    # Optionally, replace the Timestamp column with only time component (no date)
    # But user requested "only the time (date is not necessary)"; keep both but write TimeOnly as new column

    # Save new CSV
    df.to_csv(out_file, index=False)
    print(f"Wrote {out_file} with {len(df)} rows. First TimeOnly={df['Timestamp'].iloc[0]}")
