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
import joblib
from collections import deque, Counter
import time
import warnings
from scipy.stats import entropy, skew, kurtosis, iqr
from scipy.signal import welch
from sklearn.naive_bayes import GaussianNB
from sklearn.tree import DecisionTreeClassifier
from sklearn.neighbors import KNeighborsClassifier

# Disabilita warning
warnings.filterwarnings('ignore')


def print_progress(progress):
    """Stampa il progresso per il server Node.js."""
    print(f"PROGRESS:{progress}", flush=True)


class Committee:
    """Ensemble di classificatori per voting."""
    def __init__(self):
        self.models = [
            GaussianNB(),
            DecisionTreeClassifier(max_depth=10),
            KNeighborsClassifier(n_neighbors=5)
        ]

    def fit(self, X, y):
        for m in self.models:
            m.fit(X, y)

    def predict_proba(self, X):
        probs = np.array([m.predict_proba(X) for m in self.models])
        return np.mean(probs, axis=0)


def extract_features(window):
    """
    Estrae le stesse feature usate durante il training.
    IMPORTANTE: deve essere identica alla funzione nel file di training.
    """
    feats = []

    # Time-domain features
    for i in range(window.shape[1]):
        axis = window[:, i]
        feats += [
            np.mean(axis),
            np.std(axis),
            np.min(axis),
            np.max(axis),
            iqr(axis),
            entropy(np.abs(axis) + 1e-10),
            skew(axis),
            kurtosis(axis),
            np.mean(np.abs(axis - np.mean(axis)))
        ]

    # Frequency-domain (Welch PSD)
    for i in range(window.shape[1]):
        f, Pxx = welch(window[:, i], fs=200, nperseg=min(len(window), 256))
        feats.append(np.sum(Pxx))

    return np.array(feats)


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
        3: 'hook_left',
        4: 'hook_right',
        5: 'uppercut_left',
        6: 'uppercut_right'
    }
    
    def __init__(self, model_path='.'):
        """
        Args:
            model_path: Percorso alla cartella con i file del modello
        """
        self.model_path = model_path
        self.window_size = 180
        self.cooldown_frames = 150
        self.confidence_threshold = 0.7
        self.sample_rate = 200  # Hz
        
        # Carica modello
        self.model = None
        self.scaler = None
        self.label_info = None
        self._load_model()
        
        # Buffer e stato
        self.data_buffer = deque(maxlen=self.window_size)
        self.is_in_cooldown = False
        self.frames_since_last_punch = 0
        
        # Risultati
        self.detected_punches = []
        self.punch_counts = Counter()
    
    def _load_model(self):
        """Carica il modello e gli scaler."""
        # Cerca i file del modello in varie posizioni
        possible_paths = [
            self.model_path,
            os.path.dirname(os.path.abspath(__file__)),
            os.path.dirname(os.path.dirname(os.path.abspath(__file__))),
            os.path.dirname(os.path.dirname(os.path.dirname(os.path.abspath(__file__)))),
        ]
        
        for path in possible_paths:
            # Prova direttamente nella cartella
            model_file = os.path.join(path, 'punch_recognition_model.pkl')
            if os.path.exists(model_file):
                self._load_from_path(path)
                return
            
            # Prova nella sottocartella Model
            model_file = os.path.join(path, 'Model', 'punch_recognition_model.pkl')
            if os.path.exists(model_file):
                self._load_from_path(os.path.join(path, 'Model'))
                return
        
        print("Modello non trovato, uso simulazione", file=sys.stderr)
    
    def _load_from_path(self, path):
        """Carica il modello da un percorso specifico."""
        try:
            model_file = os.path.join(path, 'punch_recognition_model.pkl')
            scaler_file = os.path.join(path, 'punch_recognition_scaler.pkl')
            labels_file = os.path.join(path, 'punch_recognition_labels.pkl')
            
            self.model = joblib.load(model_file)
            print(f"Modello caricato da: {model_file}", file=sys.stderr)
            
            if os.path.exists(scaler_file):
                self.scaler = joblib.load(scaler_file)
            
            if os.path.exists(labels_file):
                self.label_info = joblib.load(labels_file)
                
        except Exception as e:
            print(f"Errore caricamento modello: {e}", file=sys.stderr)
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
        
        data = df[sensor_columns].values
        total_frames = len(data)
        
        print_progress(30)
        
        # Reset stato
        self.data_buffer.clear()
        self.detected_punches = []
        self.punch_counts = Counter()
        self.is_in_cooldown = False
        self.frames_since_last_punch = 0
        
        # Analizza frame per frame
        for i, row in enumerate(data):
            self.data_buffer.append(row)
            
            if self.is_in_cooldown:
                self.frames_since_last_punch += 1
                if self.frames_since_last_punch >= self.cooldown_frames:
                    self.is_in_cooldown = False
            
            if len(self.data_buffer) == self.window_size and not self.is_in_cooldown:
                punch = self._detect_punch(i)
                if punch:
                    self.detected_punches.append(punch)
                    self.punch_counts[punch['type']] += 1
                    self.is_in_cooldown = True
                    self.frames_since_last_punch = 0
            
            # Aggiorna progresso
            if i % max(1, total_frames // 10) == 0:
                progress = 30 + int((i / total_frames) * 50)
                print_progress(progress)
        
        print_progress(85)
        
        # Calcola statistiche finali
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
    
    def _identify_sensor_columns(self, df):
        """Identifica le colonne dei sensori nel dataframe."""
        # Pattern comuni per colonne accelerometro e giroscopio
        possible_patterns = [
            ['acc_x', 'acc_y', 'acc_z', 'gyro_x', 'gyro_y', 'gyro_z'],
            ['accel_x', 'accel_y', 'accel_z', 'gyro_x', 'gyro_y', 'gyro_z'],
            ['ax', 'ay', 'az', 'gx', 'gy', 'gz'],
            ['AccX', 'AccY', 'AccZ', 'GyroX', 'GyroY', 'GyroZ'],
            ['Acc_X', 'Acc_Y', 'Acc_Z', 'Gyro_X', 'Gyro_Y', 'Gyro_Z'],
        ]
        
        columns_lower = [c.lower() for c in df.columns]
        
        for pattern in possible_patterns:
            if all(p.lower() in columns_lower for p in pattern):
                return [df.columns[columns_lower.index(p.lower())] for p in pattern]
        
        # Fallback: rimuovi colonna Timestamp se presente e usa le prime 6 colonne numeriche
        cols_to_use = df.columns.tolist()
        if 'Timestamp' in cols_to_use:
            cols_to_use.remove('Timestamp')
        if 'timestamp' in cols_to_use:
            cols_to_use.remove('timestamp')
        
        numeric_cols = df[cols_to_use].select_dtypes(include=[np.number]).columns.tolist()
        if len(numeric_cols) >= 6:
            return numeric_cols[:6]
        
        return numeric_cols if numeric_cols else None
    
    def _detect_punch(self, frame_index):
        """Rileva un colpo dalla finestra corrente."""
        window = np.array(self.data_buffer)
        
        if self.model is not None:
            try:
                # Estrai features
                features = extract_features(window).reshape(1, -1)
                
                # Scala features se disponibile
                if self.scaler is not None:
                    features = self.scaler.transform(features)
                
                # Predizione
                if hasattr(self.model, 'predict_proba'):
                    probs = self.model.predict_proba(features)[0]
                    pred_class = np.argmax(probs)
                    confidence = probs[pred_class]
                else:
                    pred_class = self.model.predict(features)[0]
                    confidence = 0.8
                
                # Converti label in nome punch
                punch_type = self._get_label_name(pred_class)
                
                # Ignora classe Idle/guard
                if punch_type == 'guard_noPunches' or confidence < self.confidence_threshold:
                    return None
                
            except Exception as e:
                print(f"Errore predizione: {e}", file=sys.stderr)
                return None
        else:
            # Simulazione per testing
            punch_type, confidence = self._simulate_detection(window)
            if punch_type is None:
                return None
        
        # Calcola metriche del colpo
        timestamp = frame_index / self.sample_rate
        metrics = self._calculate_punch_metrics(window)
        
        return {
            'type': punch_type,
            'timestamp': timestamp,
            'confidence': float(confidence),
            'peakVelocity': metrics['peak_velocity'],
            'peakAcceleration': metrics['peak_acceleration'],
            'peakRotation': metrics['peak_rotation'],
            'duration': metrics['duration'],
            'sensorData': window[-30:].tolist()
        }
    
    def _get_label_name(self, label_id):
        """Converte l'ID della label nel nome del punch."""
        if self.label_info is not None:
            if label_id == self.label_info.get('guard_label'):
                return "guard_noPunches"
            
            for filename, label in self.label_info.get('labels', {}).items():
                if label == label_id:
                    return filename.replace('.csv', '')
        
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
        hand = 'Left' if np.mean(window[:, 0]) > 0 else 'Right'
        
        # Determina tipo di colpo
        if peak_gyro > 150:
            punch_type = f'Hook{hand}'
        elif np.mean(window[:, 2]) > 0.5:
            punch_type = f'Uppercut{hand}'
        else:
            punch_type = f'Jab{hand}'
        
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
