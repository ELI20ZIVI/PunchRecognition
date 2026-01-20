import pandas as pd
import plotly.graph_objects as go
import os

# Read the CSV file (skip malformed lines)
csv_file = "Davide Take 1/CleanCSV/QM-SS1_1A122_20260114-171158_time.csv"
df = pd.read_csv(csv_file, on_bad_lines='skip')

# Parse timestamp and build time axis in seconds from start
df['Timestamp'] = pd.to_datetime(df['Timestamp'], errors='coerce')
df = df.dropna(subset=['Timestamp']).reset_index(drop=True)
df['Time (s)'] = (df['Timestamp'] - df['Timestamp'].iloc[0]).dt.total_seconds()

# Ensure acceleration columns exist
acc_cols = [c for c in df.columns if c.lower().startswith('acc') or 'acceleration' in c.lower()]
if len(acc_cols) < 3:
    # fallback to last three columns
    acc_cols = list(df.columns[-3:])

# Convert to numeric and drop NaNs
for c in acc_cols[:3]:
    df[c] = pd.to_numeric(df[c], errors='coerce')
df = df.dropna(subset=['Time (s)'] + acc_cols[:3]).reset_index(drop=True)

# Plot only accelerations
fig = go.Figure()
colors = ['red', 'green', 'blue']
labels = ['Acc X', 'Acc Y', 'Acc Z']
for i, c in enumerate(acc_cols[:3]):
    fig.add_trace(go.Scatter(x=df['Time (s)'], y=df[c], name=labels[i] if i < len(labels) else c,
                             line=dict(color=colors[i] if i < len(colors) else None)))

fig.update_layout(
    height=600,
    title_text="Acceleration Data",
    hovermode='x unified',
    xaxis_title="Time (s)",
    yaxis_title="Acceleration",
    showlegend=True
)

# Save and show
output_file = os.path.splitext(csv_file)[0] + "_acceleration.html"
fig.write_html(output_file)
print(f"Interactive acceleration graph saved to {output_file}")
fig.show()
