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
def create_windowed_data(data: pd.DataFrame, window_size: int, step_size: int, batch_size: int = 10000, per_timestep: bool = False):
    """
    Create overlapping windows of data for time series models in memory-efficient batches.
    Each window will have shape (window_size, num_features).
    
    Args:
        data: Input DataFrame
        window_size: Size of each window
        step_size: Step between windows
        batch_size: Number of windows to accumulate before returning (default 10000)
        per_timestep: If True, return all labels per timestep; if False, return mode label
    
    Returns:
        Generator yielding tuples of (windows_batch, labels_batch)
        - If per_timestep=False: labels_batch shape (batch_size,)
        - If per_timestep=True: labels_batch shape (batch_size, window_size)
    """
    windows_batch = []
    labels_batch = []
    batch_count = 0

    X_data = data[FEATURE_COLUMNS].values
    y_data = data['Label'].values

    for start in range(0, len(data) - window_size + 1, step_size):
        end = start + window_size
        window = X_data[start:end]  # Direct numpy indexing is faster
        
        if per_timestep:
            labels = y_data[start:end]  # All labels in the window
        else:
            labels = pd.Series(y_data[start:end]).mode()[0]  # Most common label
        
        windows_batch.append(window)
        labels_batch.append(labels)
        batch_count += 1

        # Yield when batch is full
        if batch_count >= batch_size:
            yield np.array(windows_batch), np.array(labels_batch)
            windows_batch = []
            labels_batch = []
            batch_count = 0

    # Yield remaining data
    if windows_batch:
        yield np.array(windows_batch), np.array(labels_batch)

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
# TCN MODEL
# =============================
def train_model_tcn(window_size: int, n_features: int, n_classes: int) -> models.Sequential:
    model = models.Sequential([
        layers.Input(shape=(window_size, n_features)),
        
        # --- Blocco 1: Convoluzione Dilatata (dilation=1) ---
        # Cattura pattern a breve termine, guarda avanti e indietro
        layers.Conv1D(filters=64, kernel_size=5, dilation_rate=1, padding='same', activation='relu'),
        layers.Dropout(0.3),
        layers.BatchNormalization(),

        # --- Blocco 2: Convoluzione Dilatata (dilation=2) ---
        # Aumenta il receptive field su scala media (receptive field = 2^2 = 4)
        layers.Conv1D(filters=128, kernel_size=5, dilation_rate=2, padding='same', activation='relu'),
        layers.Dropout(0.3),
        layers.BatchNormalization(),

        # --- Blocco 3: Convoluzione Dilatata (dilation=4) ---
        # Cattura pattern a lungo termine (receptive field = 2^4 = 16 step temporali)
        layers.Conv1D(filters=256, kernel_size=5, dilation_rate=4, padding='same', activation='relu'),
        layers.Dropout(0.3),
        layers.BatchNormalization(),

        # --- Blocco 4: Convoluzione Dilatata (dilation=8) ---
        # Massima copertura temporale per dipendenze a molto lungo termine
        layers.Conv1D(filters=128, kernel_size=5, dilation_rate=8, padding='same', activation='relu'),
        layers.Dropout(0.3),
        layers.BatchNormalization(),

        # --- Output layer: predice la classe per ogni timestep ---
        # Output shape: (batch_size, window_size, num_classes)
        layers.Conv1D(filters=n_classes, kernel_size=1, padding='same', activation='softmax')
    ])

    # Compile and train the model
    model.compile(optimizer='adam', loss='categorical_crossentropy', metrics=['accuracy'])

    return model

# =============================
# K-FOLD TRAINING
# =============================
def kfold_train(X, y, build_fn, window_size, n_splits=5, base=None, per_timestep=False):
    """
    K-fold cross-validation training.
    
    Args:
        X: Input features (n_samples, window_size, n_features)
        y: Labels - shape (n_samples,) for single label or (n_samples, window_size) for per-timestep
        build_fn: Model building function
        window_size: Size of time window
        n_splits: Number of folds
        base: Base model for transfer learning (optional)
        per_timestep: True if y has per-timestep labels, False if single label per window
    """
    # For stratification, use appropriate label representation
    if per_timestep:
        # Use mode (most common label) of each window for stratification
        y_stratify = np.array([pd.Series(labels).mode()[0] for labels in y])
    else:
        y_stratify = y
    
    skf = StratifiedKFold(n_splits=n_splits, shuffle=True, random_state=42)
    scores = []

    for fold, (tr, te) in enumerate(skf.split(X, y_stratify), 1):
        print(f"\n====== FOLD {fold} ======")

        X_tr, X_te = X[tr], X[te]
        y_tr, y_te = y[tr], y[te]

        n_classes = len(np.unique(y_stratify))
        
        # Convert to categorical based on label shape
        if per_timestep:
            # y has shape (n_samples, window_size) -> convert to (n_samples, window_size, n_classes)
            y_tr_cat = np.zeros((y_tr.shape[0], y_tr.shape[1], n_classes))
            y_te_cat = np.zeros((y_te.shape[0], y_te.shape[1], n_classes))
            
            for i in range(y_tr.shape[0]):
                y_tr_cat[i] = to_categorical(y_tr[i], n_classes)
            for i in range(y_te.shape[0]):
                y_te_cat[i] = to_categorical(y_te[i], n_classes)
        else:
            # y has shape (n_samples,) -> convert to (n_samples, n_classes)
            y_tr_cat = to_categorical(y_tr, n_classes)
            y_te_cat = to_categorical(y_te, n_classes)

        model = build_fn(window_size, X.shape[2], n_classes) if base is None else build_fn(base, n_classes)

        es = EarlyStopping(monitor='val_loss', patience=5, restore_best_weights=True)

        model.fit(X_tr, y_tr_cat, epochs=100, batch_size=32,
                  validation_split=0.2, callbacks=[es], verbose=0)

        loss, acc = model.evaluate(X_te, y_te_cat, verbose=0)
        scores.append(acc)

        # Predictions based on output shape
        y_pred_probs = model.predict(X_te)
        
        if per_timestep:
            # Output shape: (n_test, window_size, n_classes)
            y_pred = np.argmax(y_pred_probs, axis=2).flatten()
            y_true = np.argmax(y_te_cat, axis=2).flatten()
        else:
            # Output shape: (n_test, n_classes)
            y_pred = np.argmax(y_pred_probs, axis=1)
            y_true = np.argmax(y_te_cat, axis=1)
        
        print(classification_report(y_true, y_pred))
        print(f"Fold accuracy: {acc:.4f}")

    print("\n======================")
    print(f"MEAN ACC: {np.mean(scores):.4f}")
    print(f"STD: {np.std(scores):.4f}")

    print("\n--- Final Model ---")
    print(model.summary())

    return model

# =============================
# MAIN
# =============================
if __name__ == "__main__":
    print("GPU:", tf.config.list_physical_devices('GPU'))

    data = load_data("../Labelled/Downsampled")

    window_size = 120    # number of frames per window
    step_size = 12       # 10% overlap (60 - 6 = 54 frames overlap)

    all_windows = []
    all_labels = []
    
    # Use per_timestep=True for TCN to get all labels per window
    # NOTE: If you are not using TCN, set per_timestep=False to get single label per window
    for X_batch, y_batch in create_windowed_data(data, window_size=window_size, step_size=step_size, batch_size=5000, per_timestep=True):
        all_windows.append(X_batch)
        all_labels.append(y_batch)
        print(f"  Processed batch: {len(all_windows)} batches so far...")
    
    # Concatenate all batches
    Xw = np.concatenate(all_windows)
    yw = np.concatenate(all_labels)  # Shape: (n_samples, window_size)    
    
    print("Total windows:", len(Xw))

    # print("\n--- CNN K-FOLD ---")
    # base = kfold_train(Xw, yw, build_cnn, window_size=window_size, n_splits=5, per_timestep=False)

    # print("\n--- CNN + LSTM K-FOLD ---")
    # model = kfold_train(Xw, yw, build_cnn_lstm, window_size=window_size, n_splits=5, base=base, per_timestep=False)

    print("\n--- TCN MODEL TRAINING K-FOLD ---")
    model = kfold_train(Xw, yw, train_model_tcn, window_size=window_size, n_splits=5, per_timestep=True)

    model.save("model_stratified.keras")
    print("Model saved.")
