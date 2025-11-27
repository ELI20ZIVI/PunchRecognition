import pandas as pd
import plotly.graph_objects as go
import numpy as np

# Carica il file CSV
df = pd.read_csv('Lead_jab_filtered.csv')

# Calcola il tempo in secondi
df['time'] = np.arange(len(df)) * 0.005

# Aggiungi una colonna per il numero di riga (1-indexed)
df['row_number'] = np.arange(1, len(df) + 1)

# Crea il grafico
fig = go.Figure()

# Aggiungi tutte e 6 le tracce
colori = ['red', 'green', 'blue', 'orange', 'purple', 'brown']
nomi = ['gyrX', 'gyrY', 'gyrZ', 'accX', 'accY', 'accZ']
colonne = ['gyrx(deg/s)', 'gyry(deg/s)', 'gyrz(deg/s)', 'accx(g)', 'accy(g)', 'accz(g)']

for i, (colonna, nome, colore) in enumerate(zip(colonne, nomi, colori)):
    fig.add_trace(
        go.Scatter(
            x=df['time'], 
            y=df[colonna], 
            name=nome, 
            line=dict(color=colore),
            customdata=df['row_number'],
            hovertemplate='<b>%{fullData.name}</b><br>' +
                          'Tempo: %{x:.3f} s<br>' +
                          'Valore: %{y:.4f}<br>' +
                          'Riga: %{customdata}<br>' +
                          '<extra></extra>'
        )
    )

# Calcola il numero totale di righe
total_rows = len(df)

# Configura il layout con slider
fig.update_layout(
    title="Tutti i canali IMU",
    xaxis=dict(
        title="Tempo (s)",
        rangeslider=dict(visible=True, thickness=0.05),
        type="linear"
    ),
    yaxis=dict(title="Valori"),
    height=600,
    annotations=[
        dict(
            text=f"Righe visualizzate: {total_rows}",
            xref="paper", yref="paper",
            x=0.02, y=0.98,
            showarrow=False,
            bgcolor="rgba(255, 255, 255, 0.8)",
            bordercolor="black",
            borderwidth=1,
            font=dict(size=12)
        )
    ]
)

# Aggiungi un callback JavaScript per aggiornare la label dinamicamente
fig.update_xaxes(
    rangeselector=dict(buttons=list([]))
)

# Aggiungi script per aggiornare il conteggio delle righe durante lo zoom
fig.update_layout(
    updatemenus=[],
    xaxis=dict(
        title="Tempo (s)",
        rangeslider=dict(visible=True, thickness=0.05),
        type="linear"
    )
)

# Salva come HTML con JavaScript personalizzato per l'aggiornamento dinamico
config = {'displayModeBar': True, 'responsive': True}
html_string = fig.to_html(include_plotlyjs='cdn', config=config)

# Aggiungi script JavaScript per aggiornare il conteggio
js_code = f"""
<script>
var myPlot = document.querySelector('.plotly-graph-div');
var totalRows = {total_rows};
var timeStep = 0.005;

myPlot.on('plotly_relayout', function(eventdata) {{
    if (eventdata['xaxis.range[0]'] !== undefined && eventdata['xaxis.range[1]'] !== undefined) {{
        var xMin = eventdata['xaxis.range[0]'];
        var xMax = eventdata['xaxis.range[1]'];
        var visibleRows = Math.round((xMax - xMin) / timeStep);
        
        Plotly.relayout(myPlot, {{
            'annotations[0].text': 'Righe visualizzate: ' + visibleRows
        }});
    }} else if (eventdata['xaxis.range'] !== undefined) {{
        var xMin = eventdata['xaxis.range'][0];
        var xMax = eventdata['xaxis.range'][1];
        var visibleRows = Math.round((xMax - xMin) / timeStep);
        
        Plotly.relayout(myPlot, {{
            'annotations[0].text': 'Righe visualizzate: ' + visibleRows
        }});
    }} else if (eventdata['xaxis.autorange'] === true) {{
        Plotly.relayout(myPlot, {{
            'annotations[0].text': 'Righe visualizzate: ' + totalRows
        }});
    }}
}});
</script>
"""

# Inserisci lo script prima del tag di chiusura body
html_string = html_string.replace('</body>', js_code + '</body>')

# Salva il file HTML
with open('grafico_interattivo.html', 'w', encoding='utf-8') as f:
    f.write(html_string)

print(f"Grafico salvato in 'grafico_interattivo.html'")
print(f"Apri il file nel browser per vedere il grafico interattivo con conteggio dinamico delle righe")