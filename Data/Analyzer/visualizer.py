import pandas as pd
import plotly.graph_objects as go
import numpy as np
import os
import glob

def process_file_optimized(filename):
    print(f"Elaborazione ottimizzata di: {filename}...")
    
    # --- CONFIGURAZIONE OTTIMIZZAZIONE ---
    # Se metti 1, prendi tutti i frame. 
    # Se metti 5, prendi 1 frame ogni 5 (riduce il peso del 80%).
    # Con Scattergl (GPU), di solito 1 o 2 va già benissimo.
    DOWNSAMPLE_FACTOR = 2 
    # -------------------------------------

    try:
        df = pd.read_csv(filename, on_bad_lines='skip')
    except Exception as e:
        print(f"  -> Errore lettura: {e}")
        return

    # Pulizia nomi colonne
    df.columns = [c.strip() for c in df.columns]
    
    required_cols = ['AccX', 'AccY', 'AccZ', 'Time_sec']
    if not all(col in df.columns for col in required_cols):
        print(f"  -> Colonne mancanti. Cerco colonne alternative...")
        # Fallback se i nomi sono diversi
        return

    # Ordinamento
    df = df.sort_values(by='Time_sec').reset_index(drop=True)

    # 1. APPLICAZIONE DOWNSAMPLING
    # Prendiamo una riga ogni N
    df_plot = df.iloc[::DOWNSAMPLE_FACTOR].copy()
    
    print(f"  -> Dati ridotti da {len(df)} a {len(df_plot)} punti (Fattore: {DOWNSAMPLE_FACTOR})")

    # Calcolo timestep medio sui dati ridotti per lo zoom JS
    mean_timestep = df_plot['Time_sec'].diff().mean()
    if pd.isna(mean_timestep) or mean_timestep == 0: mean_timestep = 0.016 * DOWNSAMPLE_FACTOR

    df_plot['row_number'] = df_plot.index # Manteniamo l'indice originale per riferimento

    # 2. CONFIGURAZIONE GRAFICO VELOCE (WEBGL)
    fig = go.Figure()

    target_columns = ['AccX', 'AccY', 'AccZ', 'Acc_Total']
    names = ['Acc X', 'Acc Y', 'Acc Z', 'Modulo Totale']
    colors = ['#EF553B', '#00CC96', '#AB63FA', '#19D3F3']

    for col, name, color in zip(target_columns, names, colors):
        if col in df_plot.columns:
            # NOTA: Qui uso Scattergl invece di Scatter!
            fig.add_trace(
                go.Scattergl( 
                    x=df_plot['Time_sec'],
                    y=df_plot[col],
                    name=name,
                    mode='lines', # Rimuove i pallini dei marker per velocità
                    line=dict(color=color, width=1),
                    opacity=0.8 if col != 'Acc_Total' else 1.0,
                    customdata=df_plot['row_number'],
                    hovertemplate='<b>%{fullData.name}</b><br>' +
                                  'Tempo: %{x:.2f}s<br>' +
                                  'Acc: %{y:.2f}<br>' +
                                  'Frame Orig: %{customdata}<extra></extra>'
                )
            )

    # Layout ottimizzato
    fig.update_layout(
        title=f"Analisi Accelerazioni (Ottimizzato): {filename}",
        xaxis=dict(
            title="Tempo (secondi)",
            rangeslider=dict(visible=True), # Lo slider è pesante, se è ancora lento impostalo a False
            type="linear"
        ),
        yaxis=dict(title="Accelerazione"),
        height=600,
        hovermode="x unified",
        margin=dict(l=20, r=20, t=60, b=20)
    )

    # Genera HTML
    html_string = fig.to_html(include_plotlyjs='cdn', config={'displayModeBar': True, 'responsive': True})

    # Aggiungi lo script JS per il conteggio frame dinamico
    total_rows = len(df_plot)
    js_code = f"""
    <script>
    var timeStep = {mean_timestep};
    var plotElement = document.getElementsByClassName('plotly-graph-div')[0];
    
    plotElement.on('plotly_relayout', function(eventdata) {{
        var xMin = eventdata['xaxis.range[0]'];
        var xMax = eventdata['xaxis.range[1]'];
        
        if (!xMin && eventdata['xaxis.range']) {{
            xMin = eventdata['xaxis.range'][0];
            xMax = eventdata['xaxis.range'][1];
        }}

        if (xMin !== undefined && xMax !== undefined) {{
            var visibleFrames = Math.round((xMax - xMin) / timeStep);
            console.log("Frame visibili approssimati: " + visibleFrames);
        }}
    }});
    </script>
    """
    html_string = html_string.replace('</body>', js_code + '</body>')

    output_filename = os.path.splitext(filename)[0] + '_fast_plot.html'
    with open(output_filename, 'w', encoding='utf-8') as f:
        f.write(html_string)
    
    print(f"  -> Salvato: {output_filename}")

def main():
    csv_files = glob.glob("Data/Session2/Optitrack - 60Hz/*.csv")
    for csv_file in csv_files:
        process_file_optimized(csv_file)

if __name__ == "__main__":
    main()