import pandas as pd
import plotly.graph_objects as go
from plotly.subplots import make_subplots
import numpy as np
import os
import glob

# Trova tutti i file CSV nella cartella Lead_uppercut
csv_files = sorted(glob.glob('Lead_uppercut_split/punch_*.csv'), 
                   key=lambda x: int(os.path.basename(x).replace('punch_', '').replace('.csv', '')))

print(f"Trovati {len(csv_files)} file nella cartella Lead_uppercut")

# Definisci i parametri comuni
colori = ['red', 'green', 'blue', 'orange', 'purple', 'brown']
nomi = ['gyrX', 'gyrY', 'gyrZ', 'accX', 'accY', 'accZ']
colonne = ['gyrx(deg/s)', 'gyry(deg/s)', 'gyrz(deg/s)', 'accx(g)', 'accy(g)', 'accz(g)']

# Calcola il numero di righe e colonne per i subplot
num_files = len(csv_files)
cols = 5  # Numero di colonne nella griglia (aumentato per ridurre righe)
rows = (num_files + cols - 1) // cols  # Arrotonda per eccesso

# Calcola spacing dinamico basato sul numero di righe
vertical_spacing = min(0.02, 1.0 / (rows - 1) * 0.9) if rows > 1 else 0.02
horizontal_spacing = 0.02

# Crea la figura con subplot
fig = make_subplots(
    rows=rows, cols=cols,
    subplot_titles=[os.path.basename(f).replace('.csv', '') for f in csv_files],
    vertical_spacing=vertical_spacing,
    horizontal_spacing=horizontal_spacing
)

# Processa ogni file
for idx, csv_file in enumerate(csv_files):
    # Carica il file CSV (solo le colonne IMU, senza 'diff')
    df = pd.read_csv(csv_file, usecols=colonne)
    
    # Calcola il tempo in secondi
    df['time'] = np.arange(len(df)) * 0.005
    
    # Calcola la posizione nella griglia (1-indexed per plotly)
    row = (idx // cols) + 1
    col = (idx % cols) + 1
    
    # Aggiungi tutte e 6 le tracce per questo file
    for i, (colonna, nome, colore) in enumerate(zip(colonne, nomi, colori)):
        fig.add_trace(
            go.Scatter(
                x=df['time'], 
                y=df[colonna],
                name=nome,
                line=dict(color=colore),
                showlegend=(idx == 0),  # Mostra la legenda solo per il primo subplot
                legendgroup=nome,  # Raggruppa le tracce con lo stesso nome
                customdata=np.arange(1, len(df) + 1),  # Numero di riga (1-indexed)
                hovertemplate='<b>%{fullData.name}</b><br>' +
                              'Tempo: %{x:.3f} s<br>' +
                              'Valore: %{y:.4f}<br>' +
                              'Riga: %{customdata}<br>' +
                              '<extra></extra>'
            ),
            row=row, col=col
        )

# Calcola il numero totale di righe (esempio dal primo file)
df_first = pd.read_csv(csv_files[0], usecols=colonne)
total_rows = len(df_first)

# Configura il layout
fig.update_layout(
    title=f"Tutti i canali IMU - LeadHook ({len(csv_files)} misurazioni)",
    height=300 * rows,  # Altezza ridotta per gestire molti subplot
    showlegend=True,
    legend=dict(
        orientation="h",
        yanchor="bottom",
        y=1.01,
        xanchor="right",
        x=1
    )
)

# Aggiorna gli assi
fig.update_xaxes(title_text="Tempo (s)")
fig.update_yaxes(title_text="Valori")

# Salva come HTML
config = {'displayModeBar': True, 'responsive': True}
fig.write_html('grafico_singolo.html', config=config)

print(f"Grafico salvato in 'grafico_singolo.html'")
print(f"Visualizzati {len(csv_files)} grafici dalla cartella Lead_hook_split")
print(f"Apri il file nel browser per vedere tutti i grafici interattivi")