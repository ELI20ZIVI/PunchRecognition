# =============================
# LIBRARIES
# =============================
import os
import numpy as np
import pandas as pd
import tensorflow as tf

from sklearn.model_selection import StratifiedKFold
from sklearn.metrics import classification_report

from keras import layers, models, regularizers
from keras.callbacks import EarlyStopping
from keras.utils import to_categorical

# =============================
# CONSTANTS
# =============================
FEATURE_COLUMNS = [
    'RightHand_Accel_X', 'RightHand_Accel_Y', 'RightHand_Accel_Z',
    'LeftHand_Accel_X', 'LeftHand_Accel_Y', 'LeftHand_Accel_Z'
]

# =============================
# LOAD DATA
# =============================
def load_data(file_path: str) -> pd.DataFrame:
    frames = []
    for root, _, files in os.walk(file_path):
        for fname in files:
            if fname.endswith(".csv"):
                path = os.path.join(root, fname)
                df = pd.read_csv(path)

                if not all(c in df.columns for c in FEATURE_COLUMNS + ['Label']):
                    continue

                frames.append(df[FEATURE_COLUMNS + ['Label']])

    data = pd.concat(frames, ignore_index=True)
    data = data.dropna()
    print(f"Loaded {len(data)} samples")
    return data

# =============================
# WINDOWING
# =============================
def create_windowed_data(data, window_size, step):
    X = data[FEATURE_COLUMNS].values
    y = data['Label'].values

    windows, labels = [], []

    for i in range(0, len(data) - window_size, step):
        w = X[i:i + window_size]
        lab = pd.Series(y[i:i + window_size]).mode()[0]
        windows.append(w)
        labels.append(lab)

    return np.array(windows), np.array(labels)

# =============================
# BASE CNN
# =============================
def build_cnn(window_size, n_features, n_classes):
    model = models.Sequential([
        layers.Input(shape=(window_size, n_features)),
        layers.Conv1D(64, 9, activation='relu'),
        layers.MaxPooling1D(2),
        layers.Dropout(0.3),

        layers.Conv1D(256, 3, activation='relu'),
        layers.MaxPooling1D(2),
        layers.Dropout(0.3),

        layers.GlobalAveragePooling1D(),
        layers.Dense(192, activation='relu'),
        layers.Dense(n_classes, activation='softmax')
    ])

    model.compile(
        optimizer='adam',
        loss='categorical_crossentropy',
        metrics=['accuracy']
    )
    return model

# =============================
# CNN + LSTM
# =============================
def build_cnn_lstm(base_model, n_classes):
    cnn = models.Sequential(base_model.layers)
    for l in cnn.layers:
        l.trainable = False

    for _ in range(5):
        cnn.pop()

    model = models.Sequential([
        cnn,
        layers.Bidirectional(layers.LSTM(256, return_sequences=True)),
        layers.Dropout(0.3),
        layers.BatchNormalization(),

        layers.Bidirectional(layers.LSTM(192)),
        layers.Dropout(0.3),
        layers.BatchNormalization(),

        layers.Dense(64, activation='relu', kernel_regularizer=regularizers.l2(0.001)),
        layers.Dropout(0.3),
        layers.Dense(n_classes, activation='softmax')
    ])

    model.compile(optimizer='adam', loss='categorical_crossentropy', metrics=['accuracy'])
    return model

# =============================
# K-FOLD TRAINING
# =============================
def kfold_train(X, y, build_fn, window_size, n_splits=5, base=None):
    skf = StratifiedKFold(n_splits=n_splits, shuffle=True, random_state=42)
    scores = []

    for fold, (tr, te) in enumerate(skf.split(X, y), 1):
        print(f"\n====== FOLD {fold} ======")

        X_tr, X_te = X[tr], X[te]
        y_tr, y_te = y[tr], y[te]

        n_classes = len(np.unique(y))
        y_tr = to_categorical(y_tr, n_classes)
        y_te = to_categorical(y_te, n_classes)

        model = build_fn(window_size, X.shape[2], n_classes) if base is None else build_fn(base, n_classes)

        es = EarlyStopping(monitor='val_loss', patience=5, restore_best_weights=True)

        model.fit(X_tr, y_tr, epochs=50, batch_size=32,
                  validation_split=0.2, callbacks=[es], verbose=0)

        loss, acc = model.evaluate(X_te, y_te, verbose=0)
        scores.append(acc)

        preds = np.argmax(model.predict(X_te), axis=1)
        print(classification_report(np.argmax(y_te, axis=1), preds))
        print(f"Fold accuracy: {acc:.4f}")

    print("\n======================")
    print(f"MEAN ACC: {np.mean(scores):.4f}")
    print(f"STD: {np.std(scores):.4f}")
    return model

# =============================
# MAIN
# =============================
if __name__ == "__main__":
    print("GPU:", tf.config.list_physical_devices('GPU'))

    data = load_data("../Labelled")

    Xw, yw = create_windowed_data(data, window_size=60, step=6)
    print("Total windows:", len(Xw))

    print("\n--- CNN K-FOLD ---")
    base = kfold_train(Xw, yw, build_cnn, window_size=60, n_splits=5)

    print("\n--- CNN + LSTM K-FOLD ---")
    model = kfold_train(Xw, yw, build_cnn_lstm, window_size=60, n_splits=5, base=base)

    model.save("model.keras")
    print("Model saved.")
