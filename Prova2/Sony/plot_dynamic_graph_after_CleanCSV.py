import pandas as pd
import plotly.graph_objects as go
import os

# Read the CSV file
csv_file = "Davide Take 1/CleanCSV/QM-SS1_1D19C_20260114-171158_time.csv"
df = pd.read_csv(csv_file, on_bad_lines='skip')

# Determine time column (support 'TimeOnly' or 'Timestamp')
if 'TimeOnly' in df.columns:
    time_col = 'TimeOnly'
elif 'Timestamp' in df.columns:
    time_col = 'Timestamp'
else:
    # fallback to first column
    time_col = df.columns[0]

# Convert time column to numeric seconds
df['Time (s)'] = pd.to_numeric(df[time_col], errors='coerce')
if df['Time (s)'].isna().all():
    # try parsing as timedelta strings
    df['Time (s)'] = pd.to_timedelta(df[time_col], errors='coerce').dt.total_seconds()

# Find acceleration columns (look for 'accel' or 'acceleration'), otherwise take last 3 cols
acc_cols = [c for c in df.columns if 'accel' in c.lower() or 'acceleration' in c.lower()]
if len(acc_cols) < 3:
    acc_cols = list(df.columns[-3:])

# Ensure numeric
for c in acc_cols:
    df[c] = pd.to_numeric(df[c], errors='coerce')

# Drop rows with missing required values
df = df.dropna(subset=['Time (s)'] + acc_cols).reset_index(drop=True)

# Create figure
fig = go.Figure()

colors = ['red', 'green', 'blue']
names = ['Acc X', 'Acc Y', 'Acc Z']
for i, col in enumerate(acc_cols[:3]):
    fig.add_trace(
        go.Scatter(x=df['Time (s)'], y=df[col], name=names[i] if i < len(names) else col,
                   line=dict(color=colors[i] if i < len(colors) else None))
    )

# Update layout
fig.update_layout(
    height=600,
    title_text="Acceleration Data",
    hovermode='x unified',
    showlegend=True,
    xaxis_title="Time (s)",
    yaxis_title="Acceleration"
)

# Save to HTML file in same folder as CSV
output_file = os.path.splitext(csv_file)[0] + "_dynamic.html"
fig.write_html(output_file)
print(f"Interactive graph saved to {output_file}")

# Show the plot
fig.show()
