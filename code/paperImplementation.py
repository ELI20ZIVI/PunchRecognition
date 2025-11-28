import os
import numpy as np
import pandas as pd
from scipy.stats import entropy, skew, kurtosis, iqr
from scipy.signal import welch
from sklearn.model_selection import train_test_split
from sklearn.naive_bayes import GaussianNB
from sklearn.tree import DecisionTreeClassifier
from sklearn.neighbors import KNeighborsClassifier
from sklearn.metrics import accuracy_score
from sklearn.preprocessing import StandardScaler
import joblib

# -------------------------------------------------------------
# 1. CARICAMENTO DATI DA 6 FILE
# -------------------------------------------------------------

LABELS = {
    "Lead_jab.csv": 0,
    "Lead_hook.csv": 1,
    "Lead_uppercut.csv": 2,
    "Rear_jab.csv": 3,
    "Rear_hook.csv": 4,
    "Rear_uppercut.csv": 5,
}

LABEL_TO_SPLIT_DIR = {
    "Lead_jab.csv": "LeadJab/Lead_jab_split",
    "Lead_hook.csv": "LeadHook/Lead_hook_split",
    "Lead_uppercut.csv": "LeadUppercut/Lead_uppercut_split",
    "Rear_jab.csv": "RearJab/Rear_jab_split",
    "Rear_hook.csv": "RearHook/Rear_hook_split",
    "Rear_uppercut.csv": "RearUppercut/Rear_uppercut_split",
}

GUARD_LABEL = 6  # label per guard_noPunches

def load_dataset(root="../original_files/", split_root="../"):
    data_list = []
    for filename, label in LABELS.items():
        filepath = os.path.join(root, filename)
        split_dir = os.path.join(split_root, LABEL_TO_SPLIT_DIR[filename])
        
        if os.path.exists(filepath):
            df = pd.read_csv(filepath)
            
            # Carica tutti i file split per questo tipo di colpo
            # Ordina i file per numero (punch_1, punch_2, ... sono in ordine temporale)
            punch_segments = []
            if os.path.exists(split_dir):
                split_files = sorted([f for f in os.listdir(split_dir) if f.endswith('.csv')],
                                   key=lambda x: int(x.split('_')[1].split('.')[0]))
                for split_file in split_files:
                    split_path = os.path.join(split_dir, split_file)
                    split_df = pd.read_csv(split_path)
                    punch_segments.append(split_df.values)
                print(f"  Caricato {filename}: {df.shape[0]} campioni, {len(punch_segments)} segmenti punch")
            else:
                print(f"  ATTENZIONE: directory split {split_dir} non trovata!")
            
            data_list.append({
                'data': df.values,
                'label': label,
                'filename': filename,
                'punch_segments': punch_segments
            })
        else:
            print(f"  ATTENZIONE: {filename} non trovato!")
    return data_list


# -------------------------------------------------------------
# 2. FEATURE EXTRACTION (come nel paper)
# -------------------------------------------------------------

def window_contains_punch(window, punch_segment, tolerance=1e-6):
    """
    Verifica se la finestra contiene il segmento del colpo.
    Il segmento del colpo può essere più corto di 180 campioni.
    Cerca se il punch_segment appare all'interno della window.
    """
    window_len = len(window)
    segment_len = len(punch_segment)
    
    if segment_len > window_len:
        return False
    
    # Cerca il segmento punch all'interno della finestra
    for i in range(window_len - segment_len + 1):
        window_slice = window[i:i+segment_len]
        # Confronta con una certa tolleranza per errori numerici
        if np.allclose(window_slice, punch_segment, atol=tolerance):
            return True
    return False


def extract_features(window):
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


def create_windows(data_list, window_size=180, step=1):
    Xf, yf = [], []
    
    for idx, data_item in enumerate(data_list):
        sample = data_item['data']
        label = data_item['label']
        punch_segments = data_item['punch_segments']
        filename = data_item['filename']
        
        print(f"  Processando file {idx+1}/{len(data_list)} ({filename}): {len(sample)} campioni...")
        num_windows = 0
        num_punch_windows = 0
        num_guard_windows = 0
        
        # OTTIMIZZAZIONE: inizia dal primo segmento punch
        current_segment_idx = 0
        
        for i in range(0, len(sample)-window_size, step):
            window = sample[i:i+window_size]
            
            # OTTIMIZZAZIONE: Verifica se la finestra contiene uno dei segmenti punch
            # partendo dall'ultimo segmento trovato (ordine temporale)
            # Una finestra può contenere al massimo un punch, quindi ci fermiamo al primo match
            is_punch = False
            for seg_idx in range(current_segment_idx, len(punch_segments)):
                if window_contains_punch(window, punch_segments[seg_idx]):
                    is_punch = True
                    current_segment_idx = seg_idx  # Prossima finestra parte da qui
                    break
            
            # Assegna il label appropriato
            if is_punch:
                window_label = label
                num_punch_windows += 1
            else:
                window_label = GUARD_LABEL
                num_guard_windows += 1
            
            feats = extract_features(window)
            Xf.append(feats)
            yf.append(window_label)
            num_windows += 1
            
            if num_windows % 1000 == 0:
                print(f"    -> {num_windows} finestre create (punch: {num_punch_windows}, guard: {num_guard_windows})...")
        
        print(f"  File {idx+1} completato: {num_windows} finestre totali (punch: {num_punch_windows}, guard: {num_guard_windows})")
    
    return np.array(Xf), np.array(yf)


# -------------------------------------------------------------
# 3. QUERY-BY-COMMITTEE ACTIVE LEARNING (come nel paper)
# -------------------------------------------------------------

class Committee:
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
        return np.mean(probs, axis=0)  # average committee output

def entropy_of_prediction(proba):
    return -np.sum(proba * np.log(proba + 1e-10), axis=1)


def active_learning(X, y, target_ratio=0.15, batch_ratio=0.05):
    n = len(X)
    target_size = int(target_ratio * n)
    batch_size = int(batch_ratio * n)

    print(f"  Dataset totale: {n} finestre")
    print(f"  Target finale: {target_size} finestre ({target_ratio*100}%)")
    print(f"  Batch size: {batch_size} finestre ({batch_ratio*100}%)")

    # inizializzazione 5%
    idx = np.random.choice(n, batch_size, replace=False)
    train_idx = set(idx)
    remain_idx = set(range(n)) - train_idx
    print(f"  Inizializzazione: {len(train_idx)} campioni")

    iteration = 0
    while len(train_idx) < target_size:
        iteration += 1
        print(f"\n  Iterazione {iteration}:")
        iteration += 1
        print(f"\n  Iterazione {iteration}:")
        committee = Committee()
        print(f"    Training committee su {len(train_idx)} campioni...")
        committee.fit(X[list(train_idx)], y[list(train_idx)])

        # compute uncertainty
        remain_list = list(remain_idx)
        print(f"    Calcolo incertezza su {len(remain_list)} campioni rimanenti...")
        proba = committee.predict_proba(X[remain_list])
        ent = entropy_of_prediction(proba)

        # seleziona più incerti
        uncertain_idx = np.argsort(-ent)[:batch_size]
        new_samples = [remain_list[i] for i in uncertain_idx]

        train_idx |= set(new_samples)
        remain_idx -= set(new_samples)

        print(f"    Training size aggiornato: {len(train_idx)}/{target_size}")

    final_model = Committee()
    final_model.fit(X[list(train_idx)], y[list(train_idx)])
    return final_model


# -------------------------------------------------------------
# 4. MAIN PIPELINE
# -------------------------------------------------------------

def main():
    print("\n=== FASE 1: Caricamento dataset ===")
    data_list = load_dataset()
    print(f"\nCaricati {len(data_list)} file")

    print("\n=== FASE 2: Generazione finestre ===")
    X, y = create_windows(data_list)
    print(f"\nTotale finestre generate: {len(X)}")
    print(f"Shape features: {X.shape}")
    print(f"Distribuzione labels:")
    unique, counts = np.unique(y, return_counts=True)
    for label_id, count in zip(unique, counts):
        if label_id == GUARD_LABEL:
            print(f"  guard_noPunches: {count} finestre")
        else:
            label_name = [k for k, v in LABELS.items() if v == label_id][0]
            print(f"  {label_name}: {count} finestre")

    print("\n=== FASE 3: Normalizzazione ===")
    scaler = StandardScaler()
    X = scaler.fit_transform(X)
    print("Normalizzazione completata")

    print("\n=== FASE 4: Active Learning Query-by-Committee ===")
    model = active_learning(X, y)

    print("\n=== FASE 5: Valutazione su test split ===")
    X_train, X_test, y_train, y_test = train_test_split(X, y, test_size=0.2)
    print(f"Train set: {len(X_train)} campioni")
    print(f"Test set: {len(X_test)} campioni")
    print("Training finale del modello...")
    model.fit(X_train, y_train)

    pred = np.argmax(model.predict_proba(X_test), axis=1)
    acc = accuracy_score(y_test, pred)

    print(f"Accuracy finale: {acc*100:.2f}%")

    print("\n=== FASE 6: Salvataggio modello ===")
    # Salva il modello committee
    joblib.dump(model, 'punch_recognition_model.pkl')
    print("Modello salvato in: punch_recognition_model.pkl")
    
    # Salva lo scaler (necessario per normalizzare nuovi dati)
    joblib.dump(scaler, 'punch_recognition_scaler.pkl')
    print("Scaler salvato in: punch_recognition_scaler.pkl")
    
    # Salva anche le informazioni sulle label per riferimento
    label_info = {
        'labels': LABELS,
        'guard_label': GUARD_LABEL
    }
    joblib.dump(label_info, 'punch_recognition_labels.pkl')
    print("Informazioni label salvate in: punch_recognition_labels.pkl")
    print("\nModello pronto per l'uso in altri programmi!")

if __name__ == "__main__":
    main()