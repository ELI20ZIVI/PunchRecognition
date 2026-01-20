import csv
import os
from datetime import datetime
import pandas as pd

csv_file = "QM-SS1_1A122_20260114-171951.csv"
out_file = os.path.splitext(csv_file)[0] + "_time.csv"

# Process file line-by-line to handle broken rows. Keep first column (timestamp)
# and the last three columns (assumed accelerations).
rows = []
with open(csv_file, 'r', encoding='utf-8', errors='replace') as fh:
	reader = csv.reader(fh)
	# read header (if present)
	header = next(reader, None)
	# If header exists and the first cell is 'Timestamp', skip it
	if header and len(header) > 0 and header[0].strip().lower().startswith('timestamp'):
		pass
	else:
		# header was actually a data row; process it
		if header:
			row = header
			if len(row) >= 4:
				ts = row[0].strip()
				acc = row[-3:]
				rows.append((ts, acc[0].strip(), acc[1].strip(), acc[2].strip()))

	for row in reader:
		if not row:
			continue
		if len(row) < 4:
			continue
		ts = row[0].strip()
		acc = row[-3:]
		# ensure we have three acceleration values
		if len(acc) < 3:
			continue
		rows.append((ts, acc[0].strip(), acc[1].strip(), acc[2].strip()))

# Build DataFrame
df = pd.DataFrame(rows, columns=['Timestamp_raw', 'Acceleration X', 'Acceleration Y', 'Acceleration Z'])

# Parse timestamps, drop invalid
df['Timestamp'] = pd.to_datetime(df['Timestamp_raw'], errors='coerce')
df = df.dropna(subset=['Timestamp']).reset_index(drop=True)

if df.empty:
	print('No valid rows found; aborting.')
else:
	# Compute TimeOnly as elapsed seconds from first timestamp, plus 0.181 so first value is 0.181
	start = df['Timestamp'].iloc[0]
	df['Timestamp'] = (df['Timestamp'] - start).dt.total_seconds() + 0.181
	df['Timestamp'] = df['Timestamp'].map(lambda x: f"{x:.3f}")

	# Write out only TimeOnly and the three accelerations
	out_df = df[['Timestamp', 'Acceleration X', 'Acceleration Y', 'Acceleration Z']]
	out_df.to_csv(out_file, index=False)
	print(f"Wrote {out_file} with {len(out_df)} rows. First TimeOnly={out_df['Timestamp'].iloc[0]}")
