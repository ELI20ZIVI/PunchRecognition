import os
import numpy as np
import pandas as pd
from scipy.signal import welch, butter, filtfilt
from sklearn.preprocessing import LabelEncoder
from sklearn.model_selection import train_test_split
from sklearn.ensemble import RandomForestClassifier
from sklearn.metrics import accuracy_score, classification_report, confusion_matrix
import joblib

# -----------------------------------------
# PARAMETRI
# -----------------------------------------
DATASET_DIR = "dataset"
FEATURES = ["gyrx(deg/s)", "gyry(deg/s)", "gyrz(deg/s)", "accx(g)", "accy(g)", "accz(g)"]

FS = 200

# Welch PSD parameters (usati nei paper)
NFFT = 256
NOVERLAP = 128

# -----------------------------------------
# FILTRO PASSA-BASSO (Butterworth)
# -----------------------------------------
def butter_lowpass_filter(data, cutoff=20, fs=100, order=4):
    nyq = 0.5 * fs
    normal_cutoff = cutoff / nyq
    b, a = butter(order, normal_cutoff, btype="low", analog=False)
    return filtfilt(b, a, data)

# -----------------------------------------
# FEATURE EXTRACTION – PSD PER OGNI ASSE
# -----------------------------------------
def extract_psd_features(signal, fs):
    """
    signal: vettore 1D (es. accx)
    ritorna: PSD normalizzata (array 1D)
    """
    freqs, psd = welch(signal, fs=fs, nfft=NFFT, noverlap=NOVERLAP)
    return psd


def extract_features_from_file(path):
    df = pd.read_csv(path)
    df.columns = [c.strip().lower() for c in df.columns]

    # Array dove accumuleremo tutte le feature PSD
    features = []

    for feat in FEATURES:
        raw = df[feat].values

        # Filtro passa-basso
        filtered = butter_lowpass_filter(raw, cutoff=20, fs=FS)

        # PSD
        psd = extract_psd_features(filtered, FS)

        # Aggiungi le PSD dell'asse al vettore di feature globali
        features.extend(psd)

    return np.array(features)  # shape circa: 258*6 ≈ 1548 features

# -----------------------------------------
# CARICAMENTO COMPLETO DEL DATASET
# -----------------------------------------
X = []
y = []

for class_name in os.listdir(DATASET_DIR):
    class_path = os.path.join(DATASET_DIR, class_name)

    if not os.path.isdir(class_path):
        continue

    print(f"[+] Carico classe: {class_name}")

    for filename in os.listdir(class_path):
        if filename.endswith(".csv"):
            file_path = os.path.join(class_path, filename)

            features = extract_features_from_file(file_path)

            X.append(features)
            y.append(class_name)

X = np.array(X)
y = np.array(y)

print("Shape feature matrix:", X.shape)
print("Classi presenti:", np.unique(y))

# -----------------------------------------
# LABEL ENCODING
# -----------------------------------------
encoder = LabelEncoder()
y_encoded = encoder.fit_transform(y)

# -----------------------------------------
# TRAIN/TEST SPLIT
# -----------------------------------------
X_train, X_test, y_train, y_test = train_test_split(
    X, y_encoded,
    test_size=0.2,
    random_state=42,
    stratify=y_encoded
)

# -----------------------------------------
# RANDOM FOREST
# -----------------------------------------
clf = RandomForestClassifier(
    n_estimators=300,
    max_depth=None,
    min_samples_split=2,
    random_state=42
)

clf.fit(X_train, y_train)

# -----------------------------------------
# VALUTAZIONE
# -----------------------------------------
y_pred = clf.predict(X_test)

acc = accuracy_score(y_test, y_pred)
cm = confusion_matrix(y_test, y_pred)

print("\n======================")
print("   RISULTATI MODELLO")
print("======================")
print("Accuracy Test: {:.2f}%".format(acc * 100))
print("\nClassification Report:")
print(classification_report(y_test, y_pred, target_names=encoder.classes_))
print("\nConfusion Matrix:")
print(cm)

# -----------------------------------------
# SALVATAGGIO MODELLO E ENCODER
# -----------------------------------------
joblib.dump(clf, "random_forest_punch_classifier.pkl")
joblib.dump(encoder, "label_encoder.pkl")

print("\nModello salvato come random_forest_punch_classifier.pkl")
