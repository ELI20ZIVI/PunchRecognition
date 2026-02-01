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
        self.window_size = 60  # 60 frames @ 60Hz = ~1 second
        self.step_size = 6     # Step per sliding window (come nel training)
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
        self.last_punch_frame = -self.cooldown_frames  # Permette detection dal primo frame
        self.time_data = time_column  # Salva per usarlo nei timestamp
        
        # Analizza con sliding window efficiente (come nel training)
        # Usa step_size per muoversi tra le finestre
        for i in range(0, total_frames - self.window_size + 1, self.step_size):
            # Verifica cooldown
            if (i - self.last_punch_frame) < self.cooldown_frames:
                continue
            
            # Estrai finestra corrente (come numpy array)
            window = data[i:i + self.window_size]
            
            # Rileva colpo
            punch = self._detect_punch_from_window(window, i)
            if punch:
                self.detected_punches.append(punch)
                self.punch_counts[punch['type']] += 1
                self.last_punch_frame = i
            
            # Aggiorna progresso
            if i % max(1, (total_frames - self.window_size) // 10) == 0:
                progress = 30 + int((i / (total_frames - self.window_size)) * 50)
                print_progress(progress)
        
        print_progress(85)
        
        # Calcola statistiche finali
        # Usa la colonna Time se disponibile, altrimenti calcola dai frame
        if time_column is not None and len(time_column) > 0:
            duration = float(time_column[-1] - time_column[0])
            print(f"Durata calcolata dalla colonna Time: {duration:.2f}s", file=sys.stderr)
        else:
            duration = total_frames / self.sample_rate
            print(f"Durata calcolata dai frame: {duration:.2f}s", file=sys.stderr)
        
        total_punches = sum(self.punch_counts.values())
        
        # Debug: stampa i tipi di punch rilevati
        if self.detected_punches:
            unique_types = set(p['type'] for p in self.detected_punches)
            print(f"Tipi di punch rilevati: {unique_types}", file=sys.stderr)
            print(f"Distribuzione: {dict(self.punch_counts)}", file=sys.stderr)
        
        results = {
            'totalPunches': total_punches,
            'duration': duration,
            'punchesPerMinute': (total_punches / duration * 60) if duration > 0 else 0,
            'punchCounts': dict(self.punch_counts),
            'punches': self.detected_punches
        }
        
        print_progress(100)
        
        return results
    
    def _identify_sensor_columns(self, df):
        """Identifica le colonne dei sensori nel dataframe."""
        # Prima cerca le colonne esatte usate nel training
        if all(col in df.columns for col in FEATURE_COLUMNS):
            print(f"Trovate colonne esatte del training: {FEATURE_COLUMNS}", file=sys.stderr)
            return FEATURE_COLUMNS
        
        # Prova varianti case-insensitive
        columns_lower = {c.lower(): c for c in df.columns}
        feature_cols_lower = [c.lower() for c in FEATURE_COLUMNS]
        
        if all(c in columns_lower for c in feature_cols_lower):
            matched = [columns_lower[c] for c in feature_cols_lower]
            print(f"Trovate colonne (case-insensitive): {matched}", file=sys.stderr)
            return matched
        
        # Pattern generici come fallback
        possible_patterns = [
            ['RightHand_Accel_X', 'RightHand_Accel_Y', 'RightHand_Accel_Z', 
             'LeftHand_Accel_X', 'LeftHand_Accel_Y', 'LeftHand_Accel_Z'],
            ['righthand_accel_x', 'righthand_accel_y', 'righthand_accel_z',
             'lefthand_accel_x', 'lefthand_accel_y', 'lefthand_accel_z'],
        ]
        
        for pattern in possible_patterns:
            pattern_lower = [p.lower() for p in pattern]
            if all(p in columns_lower for p in pattern_lower):
                return [columns_lower[p] for p in pattern_lower]
        
        print(f"ATTENZIONE: Colonne non corrispondenti al training. Disponibili: {df.columns.tolist()}", file=sys.stderr)
        
        # Fallback: usa le prime 6 colonne numeriche
        cols_to_use = df.columns.tolist()
        for time_col in ['Time', 'time', 'Timestamp', 'timestamp', 'Label', 'label']:
            if time_col in cols_to_use:
                cols_to_use.remove(time_col)
        
        numeric_cols = df[cols_to_use].select_dtypes(include=[np.number]).columns.tolist()
        if len(numeric_cols) >= 6:
            result = numeric_cols[:6]
            print(f"Usando fallback (prime 6 colonne numeriche): {result}", file=sys.stderr)
            return result
        
        return numeric_cols if numeric_cols else None
    
    def _detect_punch_from_window(self, window, frame_index):
        """Rileva un colpo da una finestra di dati.
        
        Args:
            window: numpy array di shape (window_size, num_features)
            frame_index: indice del frame iniziale della finestra
        """
        
        if self.model is not None:
            try:
                # IMPORTANTE: Il modello CNN-LSTM usa dati RAW (non normalizzati)
                # Reshape per Keras: (batch_size=1, window_size, num_features)
                model_input = window.reshape(1, window.shape[0], window.shape[1])
                
                # Predizione con modello Keras
                if KERAS_AVAILABLE and hasattr(self.model, 'predict'):
                    probs = self.model.predict(model_input, verbose=0)[0]
                    pred_class = np.argmax(probs)
                    confidence = float(probs[pred_class])
                # Fallback sklearn (se mai usato)
                elif hasattr(self.model, 'predict_proba'):
                    features = window.reshape(1, -1)
                    probs = self.model.predict_proba(features)[0]
                    pred_class = np.argmax(probs)
                    confidence = float(probs[pred_class])
                else:
                    pred_class = self.model.predict(model_input)[0]
                    confidence = 0.8
                
                # Converti label numerica in nome punch
                punch_type = self._get_label_name(pred_class)
                
                # Filtra: ignora classe Idle/guard e bassa confidence
                if punch_type == 'guard_noPunches' or confidence < self.confidence_threshold:
                    return None
                
            except Exception as e:
                print(f"Errore predizione: {e}", file=sys.stderr)
                import traceback
                traceback.print_exc(file=sys.stderr)
                return None
        else:
            # Simulazione per testing
            punch_type, confidence = self._simulate_detection(window)
            if punch_type is None:
                return None
        
        # Calcola metriche del colpo
        # frame_index rappresenta l'inizio della finestra
        # Il colpo è tipicamente al centro/fine della finestra
        if self.time_data is not None:
            # Usa il timestamp reale dalla colonna Time
            center_frame = frame_index + self.window_size // 2
            if center_frame < len(self.time_data):
                punch_timestamp = float(self.time_data[center_frame])
            else:
                punch_timestamp = float(self.time_data[-1])
        else:
            # Fallback: calcola dal frame index
            punch_timestamp = (frame_index + self.window_size // 2) / self.sample_rate
        
        metrics = self._calculate_punch_metrics(window)
        
        return {
            'type': punch_type,
            'timestamp': punch_timestamp,
            'confidence': float(confidence),
            'peakVelocity': metrics['peak_velocity'],
            'peakAcceleration': metrics['peak_acceleration'],
            'peakRotation': metrics['peak_rotation'],
            'duration': metrics['duration'],
            'sensorData': window[-30:].tolist()
        }
    
    def _get_label_name(self, label_id):
        """Converte l'ID della label nel nome del punch."""
        return self.PUNCH_NAMES.get(label_id, f"Unknown_{label_id}")
    
    def _simulate_detection(self, window):
        """Simula il rilevamento per testing senza modello."""
        # Calcola la magnitudo dell'accelerazione
        acc_magnitude = np.sqrt(np.sum(window[:, :3]**2, axis=1))
        peak_acc = np.max(acc_magnitude)
        
        # Soglia per rilevare un colpo
        if peak_acc < 2.0:
            return None, 0
        
        # Simula tipo di colpo basato sui pattern
        if window.shape[1] >= 6:
            gyro_magnitude = np.sqrt(np.sum(window[:, 3:6]**2, axis=1))
            peak_gyro = np.max(gyro_magnitude)
        else:
            peak_gyro = 0
        
        # Determina mano (sinistra/destra) basandosi su pattern
        hand = 'left' if np.mean(window[:, 0]) > 0 else 'right'
        
        # Determina tipo di colpo - usa formato lowercase con underscore
        if peak_gyro > 150:
            punch_type = f'hook_{hand}'
        elif np.mean(window[:, 2]) > 0.5:
            punch_type = f'uppercut_{hand}'
        else:
            punch_type = f'jab_{hand}'
        
        confidence = min(0.95, 0.6 + (peak_acc - 2.0) * 0.1)
        
        return punch_type, confidence
    
    def _calculate_punch_metrics(self, window):
        """Calcola le metriche dettagliate di un colpo."""
        # Accelerazione
        acc = window[:, :3]
        acc_magnitude = np.sqrt(np.sum(acc**2, axis=1))
        peak_acc = np.max(acc_magnitude)
        
        # Velocità (integrazione accelerazione)
        dt = 1.0 / self.sample_rate
        velocity = np.cumsum(acc_magnitude) * dt
        peak_velocity = np.max(velocity)
        
        # Rotazione (se disponibile)
        if window.shape[1] >= 6:
            gyro = window[:, 3:6]
            gyro_magnitude = np.sqrt(np.sum(gyro**2, axis=1))
            peak_rotation = np.max(gyro_magnitude)
        else:
            peak_rotation = 0
        
        # Durata del colpo (tempo sopra soglia)
        threshold = peak_acc * 0.5
        above_threshold = acc_magnitude > threshold
        duration_frames = np.sum(above_threshold)
        duration_ms = (duration_frames / self.sample_rate) * 1000
        
        return {
            'peak_acceleration': float(peak_acc),
            'peak_velocity': float(peak_velocity),
            'peak_rotation': float(peak_rotation),
            'duration': float(duration_ms)
        }


# ============================================================
# MAIN - Entry point per la web app
# ============================================================

def main():
    """Entry point principale per la web app."""
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
        
        # Output risultati in formato JSON
        print("RESULTS_JSON:")
        print(json.dumps(results, indent=2))
        print("END_RESULTS_JSON")
        
    except Exception as e:
        print(f"Errore analisi: {e}", file=sys.stderr)
        sys.exit(1)


if __name__ == '__main__':
    main()
