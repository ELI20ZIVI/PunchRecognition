import os
import numpy as np
import pandas as pd
from scipy.signal import find_peaks, savgol_filter
import matplotlib.pyplot as plt

# ====== PARAMETRI ======
FILE_PATH = "data/Lead_hook.csv"       # Assicurati che il file sia nella stessa cartella o metti il path completo
OUTPUT_FOLDER = "LeadHook_split"
WINDOW_SIZE = 180                 # Lunghezza fissa del colpo (righe)
MIN_PEAK_HEIGHT_ZS = 1.5          # Z-score minimo per considerare un picco come "colpo"
MIN_PEAK_DISTANCE = 150           # Distanza minima tra colpi forti (evita doppi conteggi dello stesso colpo)
OSCILLATION_LOOKBACK = 150        # Quanti campioni guardare indietro dal picco per cercare le onde
SMOOTH_WINDOW = 9                 # Smoothing per pulire le "micro" vibrazioni e vedere le onde reali
POLY_ORDER = 3                    # Ordine del filtro di smoothing

def load_and_process(filepath, col_name="accz(g)"):
    df = pd.read_csv(filepath)
    if col_name not in df.columns:
        # Fallback se la colonna ha nomi leggermente diversi (spazi, ecc)
        col_name = [c for c in df.columns if "accz" in c.lower()][0]
    
    sig = df[col_name].values
    # Applico un filtro Savitzky-Golay per vedere meglio le "onde" senza il rumore ad alta frequenza
    sig_smooth = savgol_filter(sig, window_length=SMOOTH_WINDOW, polyorder=POLY_ORDER)
    return df, sig, sig_smooth

def find_start_from_waves(signal_segment, peak_idx_rel):
    """
    Cerca la 3a discesa (Max -> Min) guardando indietro dal picco.
    Ritorna l'indice relativo all'inizio del segmento dove deve iniziare il colpo.
    """
    # Segmento fino al picco escluso
    # Lavoriamo al contrario (dal picco verso sinistra)
    segment_pre_peak = signal_segment[:peak_idx_rel]
    
    # Troviamo TUTTI i minimi e massimi locali nel segmento precedente
    # Usiamo una prominenza bassa per catturare anche le piccole oscillazioni
    peaks_loc, _ = find_peaks(segment_pre_peak, prominence=0.01)
    valleys_loc, _ = find_peaks(-segment_pre_peak, prominence=0.01)
    
    # Creiamo una lista di tuple (indice, tipo) ordinata per indice crescente
    extrema = [(p, 'max') for p in peaks_loc] + [(v, 'min') for v in valleys_loc]
    extrema.sort(key=lambda x: x[0])
    
    if len(extrema) < 6:
        # Non ci sono abbastanza oscillazioni (meno di 3 cicli completi)
        # Fallback: prendiamo un punto arbitrario prima del picco (es. 60 campioni prima)
        return max(0, peak_idx_rel - 60)

    # Analisi a ritroso (dall'ultimo estremo vicino al picco verso l'inizio)
    # Pattern discesa: Max -> Min.
    # Cerchiamo 3 pattern "Max -> Min" andando indietro.
    
    descents_found = 0
    target_start_idx = None
    
    # Iteriamo al contrario sugli estremi trovati
    for i in range(len(extrema) - 1, 0, -1):
        curr_type = extrema[i][1]      # Es: 'min'
        prev_type = extrema[i-1][1]    # Es: 'max'
        
        # Cerchiamo una DISCESA: un Max seguito da un Min (guardando in avanti),
        # cioè (Max, Min) nella lista ordinata.
        if prev_type == 'max' and curr_type == 'min':
            descents_found += 1
            
            if descents_found == 3:
                # Trovata la 3a discesa a ritroso!
                max_idx = extrema[i-1][0]
                min_idx = extrema[i][0]
                
                # Calcolo punto medio della discesa
                target_start_idx = (max_idx + min_idx) // 2
                break
    
    if target_start_idx is not None:
        return target_start_idx
    else:
        # Se non troviamo 3 discese chiare, torniamo all'inizio della 2a o quello che c'è
        return max(0, peak_idx_rel - 50)

def main():
    # 1. Caricamento
    if not os.path.exists(FILE_PATH):
        print(f"File {FILE_PATH} non trovato.")
        return

    df, raw_sig, smooth_sig = load_and_process(FILE_PATH)
    os.makedirs(OUTPUT_FOLDER, exist_ok=True)
    
    punch_infos = []
    
    # 2. Primo colpo: Regola Fissa (0 -> WINDOW_SIZE)
    punch_1_end = min(WINDOW_SIZE, len(df))
    
    # Salvataggio Punch 1
    fname_1 = os.path.join(OUTPUT_FOLDER, "punch_001.csv")
    df.iloc[0:punch_1_end].to_csv(fname_1, index=False)
    punch_infos.append({'file': fname_1, 'start': 0, 'peak': None, 'type': 'Fixed First'})
    print(f"Creato {fname_1} (0 - {punch_1_end})")

    # 3. Ricerca Colpi Successivi
    # Cerchiamo i picchi solo DOPO il primo colpo per evitare confusione iniziale,
    # ma permettiamo al "taglio" di tornare indietro se necessario.
    
    search_start_index = 180 
    
    # Definiamo una soglia dinamica per trovare i colpi "veri"
    # Usiamo la deviazione standard del segnale per capire cosa è un picco rilevante
    threshold = np.mean(smooth_sig) + MIN_PEAK_HEIGHT_ZS * np.std(smooth_sig)
    
    # Troviamo i picchi nel segnale completo
    peaks, _ = find_peaks(smooth_sig, height=threshold, distance=MIN_PEAK_DISTANCE)
    
    # Filtriamo i picchi che sono troppo vicini all'inizio (già coperti dal colpo 1)
    valid_peaks = [p for p in peaks if p > search_start_index]

    punch_counter = 2
    
    for p_idx in valid_peaks:
        # Definizione finestra di analisi PRE-picco
        lookback_start = max(0, p_idx - OSCILLATION_LOOKBACK)
        segment = smooth_sig[lookback_start : p_idx]
        
        # Trova start relativo all'inizio del segmento
        rel_start = find_start_from_waves(segment, len(segment))
        
        # Start assoluto
        abs_start = lookback_start + rel_start
        abs_end = abs_start + WINDOW_SIZE
        
        if abs_end > len(df):
            continue # Finestra fuori dal file
            
        # Salvataggio
        fname = os.path.join(OUTPUT_FOLDER, f"punch_{punch_counter:03d}.csv")
        df.iloc[abs_start : abs_end].to_csv(fname, index=False)
        
        punch_infos.append({
            'file': fname, 
            'start': abs_start, 
            'peak': p_idx, 
            'type': 'Detected'
        })
        print(f"Creato {fname} (Start: {abs_start}, Peak: {p_idx})")
        
        punch_counter += 1

    # 4. Visualizzazione di Debug (Opzionale ma consigliata)
    plt.figure(figsize=(12, 6))
    plt.plot(df.index, smooth_sig, label='Smoothed Acc Z', color='gray', alpha=0.5)
    
    # Disegna le finestre trovate
    colors = ['red', 'green', 'blue', 'orange', 'purple']
    for i, info in enumerate(punch_infos):
        s = info['start']
        e = s + WINDOW_SIZE
        c = colors[i % len(colors)]
        plt.axvspan(s, e, color=c, alpha=0.2)
        plt.axvline(s, color=c, linestyle='--', alpha=0.8)
        if info['peak'] is not None:
             plt.plot(info['peak'], smooth_sig[info['peak']], "x", color=c)

    plt.title("Segmentazione Colpi: Start basato su 3a oscillazione")
    plt.legend()
    plt.show()
    
    # Salva riepilogo
    pd.DataFrame(punch_infos).to_csv(os.path.join(OUTPUT_FOLDER, "summary.csv"), index=False)

if __name__ == "__main__":
    main()