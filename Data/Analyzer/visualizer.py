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
    required_cols = ['Frame', 'RightHand_Accel_X', 'RightHand_Accel_Y', 'RightHand_Accel_Z', 
                     'LeftHand_Accel_X', 'LeftHand_Accel_Y', 'LeftHand_Accel_Z']
    if not all(col in df.columns for col in required_cols):
        raise ValueError(f'Colonne richieste non trovate. Colonne disponibili: {df.columns.tolist()}')

    # Ordinamento
    df = df.sort_values(by='Frame').reset_index(drop=True)

    # 1. APPLICAZIONE DOWNSAMPLING
    # Prendiamo una riga ogni N
    df_plot = df.iloc[::DOWNSAMPLE_FACTOR].copy()
    
    print(f"  -> Dati ridotti da {len(df)} a {len(df_plot)} punti (Fattore: {DOWNSAMPLE_FACTOR})")

    # Calcolo timestep medio sui dati ridotti per lo zoom JS
    mean_timestep = df_plot['Frame'].diff().mean()
    if pd.isna(mean_timestep) or mean_timestep == 0: mean_timestep = 0.016 * DOWNSAMPLE_FACTOR

    df_plot['row_number'] = df_plot.index # Manteniamo l'indice originale per riferimento

    # 2. CONFIGURAZIONE GRAFICI SEPARATI PER RIGHT HAND E LEFT HAND
    hands_data = [
        {
            'hand': 'RightHand',
            'columns': ['RightHand_Accel_X', 'RightHand_Accel_Y', 'RightHand_Accel_Z'],
            'names': ['Acc X', 'Acc Y', 'Acc Z']
        },
        {
            'hand': 'LeftHand',
            'columns': ['LeftHand_Accel_X', 'LeftHand_Accel_Y', 'LeftHand_Accel_Z'],
            'names': ['Acc X', 'Acc Y', 'Acc Z']
        }
    ]
    
    colors = ['#EF553B', '#00CC96', '#AB63FA']
    
    for hand_info in hands_data:
        fig = go.Figure()
        
        for col, name, color in zip(hand_info['columns'], hand_info['names'], colors):
            if col in df_plot.columns:
                fig.add_trace(
                    go.Scattergl( 
                        x=df_plot['Frame'],
                        y=df_plot[col],
                        name=name,
                        mode='lines',
                        line=dict(color=color, width=1),
                        opacity=0.8,
                        customdata=df_plot['row_number'],
                        hovertemplate='<b>%{fullData.name}</b><br>' +
                                      'Tempo: %{x:.2f}s<br>' +
                                      'Acc: %{y:.2f}<br>' +
                                      'Frame Orig: %{customdata}<extra></extra>'
                    )
                )
        
        # Layout ottimizzato
        fig.update_layout(
            title=f"Analisi Accelerazioni - {hand_info['hand']}: {filename}",
            xaxis=dict(
                title="Tempo (secondi)",
                rangeslider=dict(visible=True),
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
        
        output_filename = os.path.splitext(filename)[0] + f'_{hand_info["hand"]}_plot.html'
        with open(output_filename, 'w', encoding='utf-8') as f:
            f.write(html_string)
        
        print(f"  -> Salvato: {output_filename}")

def main():
    csv_files = glob.glob("../ProcessedData/*.csv")
    for csv_file in csv_files:
        process_file_optimized(csv_file)

if __name__ == "__main__":
    main()