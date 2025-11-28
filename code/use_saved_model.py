"""
Esempio di utilizzo del modello di riconoscimento punch salvato.
Questo script mostra come caricare il modello e fare predizioni su nuovi dati.
"""

import numpy as np
import joblib
from scipy.stats import entropy, skew, kurtosis, iqr
from scipy.signal import welch


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
        f, Pxx = welch(window[:, i], fs=200)
        feats.append(np.sum(Pxx))  # spectral power

    return np.array(feats)


def load_model():
    """
    Carica il modello, lo scaler e le informazioni sulle label.
    """
    model = joblib.load('punch_recognition_model.pkl')
    scaler = joblib.load('punch_recognition_scaler.pkl')
    label_info = joblib.load('punch_recognition_labels.pkl')
    return model, scaler, label_info


def predict_punch(window_data, model, scaler):
    """
    Predice il tipo di punch da una finestra di dati.
    
    Args:
        window_data: numpy array di shape (180, n_features) contenente i dati della finestra
        model: modello caricato
        scaler: scaler caricato
    
    Returns:
        predicted_label: int, l'etichetta predetta
        probabilities: array delle probabilità per ogni classe
    """
    # 1. Estrai features
    features = extract_features(window_data)
    
    # 2. Normalizza con lo scaler
    features_scaled = scaler.transform(features.reshape(1, -1))
    
    # 3. Fai la predizione
    probabilities = model.predict_proba(features_scaled)[0]
    predicted_label = np.argmax(probabilities)
    
    return predicted_label, probabilities


def get_label_name(label_id, label_info):
    """
    Converte l'ID della label nel nome del punch.
    """
    if label_id == label_info['guard_label']:
        return "guard_noPunches"
    
    for filename, label in label_info['labels'].items():
        if label == label_id:
            return filename.replace('.csv', '')
    
    return f"Unknown ({label_id})"


def main():
    """
    Esempio di utilizzo del modello.
    """
    print("=== Caricamento modello ===")
    model, scaler, label_info = load_model()
    print("Modello caricato con successo!")
    
    print("\n=== Informazioni modello ===")
    print(f"Classi disponibili:")
    for filename, label in label_info['labels'].items():
        print(f"  {label}: {filename.replace('.csv', '')}")
    print(f"  {label_info['guard_label']}: guard_noPunches")
    
    print("\n=== Esempio di predizione ===")
    # ESEMPIO: genera dati casuali per test
    # In un caso reale, questi sarebbero i tuoi dati effettivi di accelerometro/giroscopio
    # Assumendo 6 canali (3 accelerometro + 3 giroscopio) e 180 campioni
    window_size = 180
    n_channels = 6
    example_window = np.random.randn(window_size, n_channels)
    
    print(f"Predizione su finestra di shape: {example_window.shape}")
    predicted_label, probabilities = predict_punch(example_window, model, scaler)
    predicted_name = get_label_name(predicted_label, label_info)
    
    print(f"\nPredizione: {predicted_name} (label {predicted_label})")
    print(f"Confidence: {probabilities[predicted_label]*100:.2f}%")
    
    print("\nProbabilità per tutte le classi:")
    for i, prob in enumerate(probabilities):
        label_name = get_label_name(i, label_info)
        print(f"  {label_name}: {prob*100:.2f}%")
    
    print("\n=== Come usare questo modello ===")
    print("1. Carica i tuoi dati di sensori in una finestra di 180 campioni")
    print("2. Assicurati che i dati siano in formato numpy array (180, n_channels)")
    print("3. Chiama predict_punch(window_data, model, scaler)")
    print("4. Ottieni la predizione e le probabilità")
    

if __name__ == "__main__":
    main()
