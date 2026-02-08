#!/usr/bin/env python3
"""
Boxing Punch Analyzer - Python Backend per Web App
Analizza i dati dei sensori e rileva i colpi usando il modello ML.
"""

import sys
import os
import json
import numpy as np
import pandas as pd
from collections import Counter
import time
import warnings

# Disabilita warning
warnings.filterwarnings('ignore')

# Prova a importare tensorflow/keras
try:
    import tensorflow as tf
    from keras.models import load_model
    KERAS_AVAILABLE = True
except ImportError:
    KERAS_AVAILABLE = False
    print("Keras non disponibile, modalità simulazione", file=sys.stderr)


def print_progress(progress):
    """Stampa il progresso per il server Node.js."""
    print(f"PROGRESS:{progress}", flush=True)


# CONSTANTS - devono corrispondere al training
FEATURE_COLUMNS = ['RightHand_Accel_X', 'RightHand_Accel_Y', 'RightHand_Accel_Z', 
                   'LeftHand_Accel_X', 'LeftHand_Accel_Y', 'LeftHand_Accel_Z']


class WebPunchAnalyzer:
    """
    Analizzatore di colpi per web app.
    Processa file CSV e restituisce risultati JSON per il frontend.
    """
    
    # Mapping delle classi
    PUNCH_NAMES = {
        0: 'guard_noPunches',
        1: 'jab_left',
        2: 'jab_right', 
        3: 'uppercut_left',
        4: 'uppercut_right',
        5: 'hook_left',
        6: 'hook_right'
    }
    
    def __init__(self, model_path='.'):
        """
        Args:
            model_path: Percorso alla cartella con i file del modello
        """
        self.model_path = model_path
        # Parametri identici al training
        self.window_size = 120  # 60 frames @ 60Hz = ~1 second
        self.step_size = 12     # Step per sliding window (come nel training)
        self.cooldown_frames = 60  # ~0.5 secondi tra colpi
        self.confidence_threshold = 0.85  # Soglia di confidenza
        self.sample_rate = 60
        
        # Carica modello
        self.model = None
        self._load_model()
        
        # Stato e risultati
        self.detected_punches = []
        self.punch_counts = Counter()
        self.time_data = None  # Colonna temporale dal dataset
    
    def _load_model(self):
        """Carica il modello dal percorso fisso."""
        # Usa il percorso assoluto basato sulla posizione di questo script
        script_dir = os.path.dirname(os.path.abspath(__file__))
        model_path = os.path.join(script_dir, 'model.keras')
        
        if not KERAS_AVAILABLE:
            print(f"Keras non disponibile", file=sys.stderr)
            return
        
        if not os.path.exists(model_path):
            print(f"ERRORE: Modello non trovato in: {model_path}", file=sys.stderr)
            self.model = None
            return
        
        try:
            print(f"Caricamento modello: {model_path}", file=sys.stderr)
            self.model = load_model(model_path)
            print(f"Modello caricato con successo", file=sys.stderr)
        except Exception as e:
            print(f"Errore caricamento modello: {e}", file=sys.stderr)
            import traceback
            traceback.print_exc(file=sys.stderr)
            self.model = None
    
    def analyze_csv(self, csv_path):
        """Analizza un file CSV con i dati dei sensori."""
        print_progress(10)
        
        # Carica i dati
        try:
            df = pd.read_csv(csv_path)
        except Exception as e:
            raise Exception(f"Errore lettura CSV: {e}")
        
        print_progress(20)
        
        # Identifica le colonne dei sensori
        sensor_columns = self._identify_sensor_columns(df)
        if not sensor_columns:
            raise Exception("Colonne sensori non trovate nel CSV")
        
        # Estrai colonna Time se disponibile
        time_column = None
        for col_name in ['Time', 'time', 'Timestamp', 'timestamp']:
            if col_name in df.columns:
                time_column = df[col_name].values
                print(f"Trovata colonna temporale: {col_name}", file=sys.stderr)
                break
        
        # Usa DataFrame per mantenere struttura durante sliding window
        data_df = df[sensor_columns]
        data = data_df.values
        total_frames = len(data)
        
        print_progress(30)
        
        # Reset stato
        self.detected_punches = []
        self.punch_counts = Counter()
        self.time_data = time_column  # Salva per usarlo nei timestamp
        
        # Debug: tracking all predictions
        self.all_predictions = Counter()
        self.filtered_by_confidence = Counter()
        
        # Analizza con sliding window
        i = 0
        max_start = total_frames - self.window_size + 1
        
        while i < max_start:
            # Estrai finestra corrente
            window = data[i:i + self.window_size]
            
            # Rileva pattern grezzo nella finestra
            detection = self._detect_raw_pattern(window)
            
            if not detection:
                # Nessun colpo rilevato, avanza
                i += self.step_size
                continue
                
            start_offset = detection['start_offset']
            
            # CASO 1: Riallineamento (Punch parziale)
            # Se il colpo inizia dopo l'inizio della finestra, sposta la finestra
            # esattamente all'inizio del colpo
            if start_offset > 0:
                i += start_offset
                continue

            # Controlla che type sia presente (colpo allineato)
            if 'type' not in detection:
                # Non è un colpo valido, avanza standard
                i += self.step_size
                continue
            
            # CASO 2: Colpo allineato (start_offset == 0)
            # Il colpo inizia all'indice 0 della finestra corrente
            punch_type = detection['type']
            end_offset = detection['end_offset'] # Indice ultimo frame del colpo (es. 59)
            confidence = detection['confidence']
            
            # Filtra per confidenza
            if confidence < self.confidence_threshold:
                self.filtered_by_confidence[punch_type] += 1
                print(f"DEBUG: Scartato {punch_type} per bassa confidenza ({confidence:.3f})", file=sys.stderr)
                i += self.step_size # Avanza standard se scartato
                continue
                
            # Log valido
            self.all_predictions[punch_type] += 1
            if punch_type in ['jab_left', 'jab_right']:
                 print(f"DEBUG: Rilevato {punch_type} CONF:{confidence:.3f} Frames:{end_offset+1}", file=sys.stderr)
            
            # Raccogli i frame del colpo
            # Inizia con il segmento trovato nella finestra corrente
            frames_accumulator = [window[:end_offset+1]]
            
            # ESTENSIONE (While interno)
            # Se il colpo arriva fino alla fine della finestra (es. indice 59 su 60),
            # controlla le finestre successive per vedere se continua
            current_end_in_window = end_offset
            extended_frames_count = 0
            
            # Loop per estendere se necessario
            while current_end_in_window == self.window_size - 1:
                # Calcola dove inizierebbe la prossima finestra
                # Attenzione: dobbiamo guardare subito dopo l'ultimo frame processato
                # frames_accumulator contiene segmenti consecutivi
                total_frames_collected = sum(len(x) for x in frames_accumulator)
                next_window_start = i + total_frames_collected
                
                if next_window_start >= max_start:
                    break
                    
                next_window = data[next_window_start : next_window_start + self.window_size]
                next_det = self._detect_raw_pattern(next_window)
                
                # Continua solo se trova LO STESSO tipo di colpo e inizia SUBITO (aligned)
                if next_det and next_det.get('start_offset') == 0 and next_det.get('type') == punch_type:
                    next_end = next_det['end_offset']
                    frames_accumulator.append(next_window[:next_end+1])
                    current_end_in_window = next_end
                    extended_frames_count += 1
                    # Sicurezza per evitare loop infiniti su dati corrotti
                    if extended_frames_count > 10: 
                        break
                else:
                    # Il colpo è finito o cambiato
                    break
            
            # Assembla tutti i dati del colpo
            full_punch_data = np.concatenate(frames_accumulator)
            
            # Calcola metriche complete
            metrics = self._calculate_punch_metrics(full_punch_data)
            
            # Timestamp
            if self.time_data is not None:
                # Prendi il timestamp del centro del colpo
                center_idx = i + len(full_punch_data) // 2
                if center_idx < len(self.time_data):
                    timestamp = float(self.time_data[center_idx])
                else:
                    timestamp = float(self.time_data[-1])
            else:
                timestamp = (i + len(full_punch_data) // 2) / self.sample_rate
                
            punch_record = {
                'type': punch_type,
                'timestamp': timestamp,
                'confidence': confidence,
                'peakVelocity': metrics['peak_velocity'],
                'peakAcceleration': metrics['peak_acceleration'],
                'meanAcceleration': metrics['mean_acceleration'],
                'meanVelocity': metrics['mean_velocity'],
                'duration': metrics['duration'],
                'sensorData': full_punch_data.tolist()
            }
            
            self.detected_punches.append(punch_record)
            self.punch_counts[punch_type] += 1
            
            # Skip post-punch
            # Sposta la window alla prima guardia disponibile (fine del colpo)
            total_len = sum(len(x) for x in frames_accumulator)
            i += total_len
            
            # Aggiorna progresso visuale
            if i % max(1, total_frames // 10) == 0:
                prog = 30 + int((i / total_frames) * 50)
                print_progress(min(prog, 85))
        
        print_progress(85)
        
        # Debug stats
        print(f"\n=== DEBUG: Statistiche ===", file=sys.stderr)
        print(f"Totale predizioni: {dict(self.all_predictions)}", file=sys.stderr)
        print(f"Scartati per confidence: {dict(self.filtered_by_confidence)}", file=sys.stderr)
        
        # Finalizza risultati
        if time_column is not None and len(time_column) > 0:
            duration = float(time_column[-1] - time_column[0])
        else:
            duration = total_frames / self.sample_rate
            
        total_punches = sum(self.punch_counts.values())
        
        results = {
            'totalPunches': total_punches,
            'duration': duration,
            'punchesPerMinute': (total_punches / duration * 60) if duration > 0 else 0,
            'punchCounts': dict(self.punch_counts),
            'punches': self.detected_punches
        }
        
        print_progress(100)
        return results

    def _detect_raw_pattern(self, window):
        """
        Analizza una finestra e restituisce i dati grezzi di rilevamento.
        Non applica filtri di confidenza qui per permettere logiche superiori.
        Returns:
            dict: {type, start_offset, end_offset, confidence} o None
        """
        if self.model is None:
             # Simulation fallback
             t, c = self._simulate_detection(window)
             if t: return {'type': t, 'start_offset': 0, 'end_offset': len(window)-1, 'confidence': c}
             return None

        # Reshape (1, 60, 6)
        model_input = window.reshape(1, window.shape[0], window.shape[1])
        
        try:
            predictions = self.model.predict(model_input, verbose=0)
            
            # Gestione TCN (Sequenza)
            if len(predictions.shape) == 3:
                # Shape (1, 60, 7)
                probs = predictions[0] # (60, 7)
                classes = np.argmax(probs, axis=1) # (60,)
                
                # Trova indici non-guardia
                non_guard = np.where(classes != 0)[0]
                
                if len(non_guard) == 0:
                    return None
                
                start_offset = int(non_guard[0])
                
                # Se non è allineato, ritorna solo l'offset per riallineare
                if start_offset > 0:
                    return {'start_offset': start_offset}
                
                # Se è allineato (start_offset == 0)
                punch_class = classes[0]
                punch_type = self._get_label_name(punch_class)
                
                # Trova la fine della sequenza di QUESTO punch
                # Cerca il primo indice dove la classe cambia (diventa 0 o altro punch)
                changes = np.where((classes != punch_class))[0] # and (classes != 0)
                
                if len(changes) > 0:
                    end_offset = int(changes[0]) - 1 # L'ultimo indice valido è uno prima del cambio
                else:
                    end_offset = self.window_size - 1 # Fino alla fine
                
                # Calcola confidenza SOLO sui frame del colpo
                # (Fix per Jab corti che venivano mediati con la guardia)
                punch_probs = probs[:end_offset+1, punch_class]
                confidence = float(np.mean(punch_probs))
                
                return {
                    'type': punch_type,
                    'start_offset': 0,
                    'end_offset': end_offset,
                    'confidence': confidence
                }
                
            else:
                # Fallback per modelli non-sequenziali (CNN standard)
                # Assume che la predizione valga per tutta la finestra
                p_class = np.argmax(predictions[0])
                conf = float(predictions[0][p_class])
                p_type = self._get_label_name(p_class)
                
                if p_class == 0: return None
                
                return {
                    'type': p_type,
                    'start_offset': 0,
                    'end_offset': self.window_size - 1,
                    'confidence': conf
                }
                
        except Exception as e:
            # print(f"Error in prediction: {e}", file=sys.stderr)
            return None

    def _identify_sensor_columns(self, df):
        """Identifica le colonne dei sensori nel dataframe."""
        if all(col in df.columns for col in FEATURE_COLUMNS):
            return FEATURE_COLUMNS
        
        columns_lower = {c.lower(): c for c in df.columns}
        feature_cols_lower = [c.lower() for c in FEATURE_COLUMNS]
        
        if all(c in columns_lower for c in feature_cols_lower):
            return [columns_lower[c] for c in feature_cols_lower]
        
        possible_patterns = [
            ['RightHand_Accel_X', 'RightHand_Accel_Y', 'RightHand_Accel_Z', 
             'LeftHand_Accel_X', 'LeftHand_Accel_Y', 'LeftHand_Accel_Z'],
        ]
        
        for pattern in possible_patterns:
            pattern_lower = [p.lower() for p in pattern]
            if all(p in columns_lower for p in pattern_lower):
                return [columns_lower[p] for p in pattern_lower]
        
        cols_to_use = df.columns.tolist()
        for time_col in ['Time', 'time', 'Timestamp', 'timestamp', 'Label', 'label']:
            if time_col in cols_to_use:
                cols_to_use.remove(time_col)
        
        numeric_cols = df[cols_to_use].select_dtypes(include=[np.number]).columns.tolist()
        if len(numeric_cols) >= 6:
            return numeric_cols[:6]
        return None
    
    def _get_label_name(self, label_id):
        return self.PUNCH_NAMES.get(label_id, f"Unknown_{label_id}")
    
    def _simulate_detection(self, window):
        acc_magnitude = np.sqrt(np.sum(window[:, :3]**2, axis=1))
        peak_acc = np.max(acc_magnitude)
        if peak_acc < 2.0: return None, 0
        
        if window.shape[1] >= 6:
            peak_gyro = np.max(np.sqrt(np.sum(window[:, 3:6]**2, axis=1)))
        else: peak_gyro = 0
        
        hand = 'left' if np.mean(window[:, 0]) > 0 else 'right'
        if peak_gyro > 150: punch_type = f'hook_{hand}'
        elif np.mean(window[:, 2]) > 0.5: punch_type = f'uppercut_{hand}'
        else: punch_type = f'jab_{hand}'
        
        confidence = min(0.95, 0.6 + (peak_acc - 2.0) * 0.1)
        return punch_type, confidence
    
    def _calculate_punch_metrics(self, window):
        acc_1 = window[:, :3] / 100.0
        acc_2 = window[:, 3:6] / 100.0
        acc_magnitude = np.maximum(
            np.sqrt(np.sum(acc_1**2, axis=1)),
            np.sqrt(np.sum(acc_2**2, axis=1))
        )
        
        peak_acc = np.max(acc_magnitude)
        mean_acc = np.mean(acc_magnitude)
        
        dt = 1.0 / self.sample_rate
        velocity = np.cumsum(acc_magnitude) * dt
        peak_velocity = np.max(velocity)
        mean_velocity = np.mean(velocity)
        
        duration_ms = (window.shape[0] / self.sample_rate) * 1000
        
        return {
            'peak_acceleration': float(peak_acc),
            'mean_acceleration': float(mean_acc),
            'peak_velocity': float(peak_velocity),
            'mean_velocity': float(mean_velocity),
            'duration': float(duration_ms)
        }

def main():
    if len(sys.argv) < 2:
        print("Usage: python analizer.py <csv_file> [model_path]", file=sys.stderr)
        sys.exit(1)
    
    csv_path = sys.argv[1]
    model_path = sys.argv[2] if len(sys.argv) > 2 else '.'
    
    if not os.path.exists(csv_path):
        print(f"File non trovato: {csv_path}", file=sys.stderr)
        sys.exit(1)
    
    try:
        analyzer = WebPunchAnalyzer(model_path)
        results = analyzer.analyze_csv(csv_path)
        print("RESULTS_JSON:")
        print(json.dumps(results, indent=2))
        print("END_RESULTS_JSON")
    except Exception as e:
        print(f"Errore analisi: {e}", file=sys.stderr)
        sys.exit(1)

if __name__ == '__main__':
    main()