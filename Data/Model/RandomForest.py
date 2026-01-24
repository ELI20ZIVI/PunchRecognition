# RandomForest libraries
from sklearn.ensemble import RandomForestClassifier
from sklearn.model_selection import train_test_split
from sklearn.metrics import classification_report

# CNN libraries
import tensorflow as tf
from keras import layers, models
from keras.callbacks import EarlyStopping
from keras.utils import to_categorical

# Data libraries
import pandas as pd
import numpy as np
import os

##########
# README #
##########
# Note: If you use sklearn models, run python with -m cuml.accel to use GPU

# CONSTANTS
# Changing FEATURE_COLUMNS will affect features used for all models. 
# FEATURE_COLUMNS = ['Magnitude_RightHand', 'Magnitude_LeftHand']
FEATURE_COLUMNS = ['RightHand_Accel_X', 'RightHand_Accel_Y', 'RightHand_Accel_Z', 'LeftHand_Accel_X', 'LeftHand_Accel_Y', 'LeftHand_Accel_Z']

# Load dataset
def load_data(file_path: str) -> pd.DataFrame:
    """
    Walk `file_path` for CSVs, compute magnitude features and return a single DataFrame
    with columns: ['Time'] + FEATURE_COLUMNS + ['Label'].
    """
    frames = []

    if not os.path.isdir(file_path):
        print(f"Data folder not found: {file_path}")
        return pd.DataFrame(columns=['Time'] + FEATURE_COLUMNS + ['Label'])

    for root, _dirs, files in os.walk(file_path):
        for fname in files:
            if not fname.endswith('.csv'):
                continue

            csv_path = os.path.join(root, fname)
            try:
                csv_data = pd.read_csv(csv_path)
            except Exception as e:
                print(f"Skipping {csv_path}: {e}")
                continue

            required = [
                'RightHand_Accel_X', 'RightHand_Accel_Y', 'RightHand_Accel_Z',
                'LeftHand_Accel_X', 'LeftHand_Accel_Y', 'LeftHand_Accel_Z',
                'Label'
            ]
            missing = [c for c in required if c not in csv_data.columns]
            if missing:
                print(f"Skipping {csv_path}: missing columns {missing}")
                continue

            # Build per-file frame
            df = pd.DataFrame()
            df['Time'] = csv_data['Time'] if 'Time' in csv_data.columns else pd.RangeIndex(start=0, stop=len(csv_data))

            # Compute magnitudes
            df['Magnitude_RightHand'] = (
                (csv_data['RightHand_Accel_X'] ** 2 +
                 csv_data['RightHand_Accel_Y'] ** 2 +
                 csv_data['RightHand_Accel_Z'] ** 2) ** 0.5
            )
            df['Magnitude_LeftHand'] = (
                (csv_data['LeftHand_Accel_X'] ** 2 +
                 csv_data['LeftHand_Accel_Y'] ** 2 +
                 csv_data['LeftHand_Accel_Z'] ** 2) ** 0.5
            )
            df[FEATURE_COLUMNS] = csv_data[FEATURE_COLUMNS]
            df['Label'] = csv_data['Label']

            frames.append(df[['Time'] + FEATURE_COLUMNS + ['Label']])

    if frames:
        data = pd.concat(frames, ignore_index=True)
        data = data.dropna(subset=FEATURE_COLUMNS + ['Label'])
        print(f"Loaded {len(data)} samples from {file_path}.")
        return data
    else:
        print(f"No CSV files found in {file_path}.")
        return pd.DataFrame(columns=['Time'] + FEATURE_COLUMNS + ['Label'])


# Train Random Forest model
def train_model_RandomForest(data: pd.DataFrame) -> RandomForestClassifier:
    # Basic validation to avoid empty splits
    n = len(data)
    if n < 2:
        raise ValueError(
            f"Not enough samples to split the dataset (n={n}). "
            f"Ensure there are labelled CSVs under the data folder."
        )

    X = data[FEATURE_COLUMNS]
    y = data['Label']

    # Use a safer split when the dataset is small
    test_size = 0.2 if n >= 5 else 0.5
    X_train, X_test, y_train, y_test = train_test_split(
        X, y, test_size=test_size, random_state=42
    )

    model = RandomForestClassifier(n_estimators=100, random_state=42)
    model.fit(X_train, y_train)

    y_pred = model.predict(X_test)
    print(classification_report(y_test, y_pred))

    return model


# Train CNN model
def train_model_CNN(X: np.ndarray, y: np.ndarray, window_size: int) -> models.Sequential:
    # Basic validation to avoid empty splits
    n = len(X)
    if n < 2:
        raise ValueError(
            f"Not enough samples to split the dataset (n={n}). "
            f"Ensure there are labelled CSVs under the data folder."
        )

    # Use a safer split when the dataset is small
    test_size = 0.2 if n >= 5 else 0.5
    X_train, X_test, y_train, y_test = train_test_split(
        X, y, test_size=test_size, random_state=42
    )

    # Convert labels to one-hot encoding for categorical crossentropy
    num_classes = len(np.unique(y))
    y_train = to_categorical(y_train, num_classes=num_classes)
    y_test = to_categorical(y_test, num_classes=num_classes)

    model = models.Sequential([
        
        # --- 1. Blocco Convoluzionale "Fine" ---
        # Cattura dettagli veloci e picchi improvvisi (es. l'impatto)
        # kernel_size=3 guarda finestre temporali molto piccole
        layers.Input(shape=(window_size, X.shape[2])),
        layers.Conv1D(filters=32, kernel_size=3, activation='relu'),
        layers.MaxPooling1D(pool_size=2),
        layers.Dropout(0.3), # Previene l'overfitting (molto comune sui dati dei sensori)

        # --- 2. Blocco Convoluzionale "Grezzo" ---
        # Cattura pattern più lunghi (es. il movimento di caricamento del braccio)
        # filters=64 aumenta la capacità di astrazione
        layers.Conv1D(filters=64, kernel_size=5, activation='relu'),
        layers.MaxPooling1D(pool_size=2),
        layers.Dropout(0.3),

        # --- 3. Feature Aggregation ---
        # GlobalAveragePooling è meglio di Flatten per le time series:
        # riduce drasticamente i parametri e rende il modello più robusto.
        layers.GlobalAveragePooling1D(),

        # --- 4. Classificazione ---
        layers.Dense(64, activation='relu'),
        layers.Dense(len(np.unique(y)), activation='softmax') # Output: probabilità per ogni colpo
    ])

    model.compile(optimizer='adam', loss='categorical_crossentropy', metrics=['accuracy'])

    early_stopping = EarlyStopping(monitor='val_loss', patience=5, restore_best_weights=True)

    model.fit(X_train, y_train, epochs=50, batch_size=32, validation_split=0.2, callbacks=[early_stopping], verbose=1)

    loss, accuracy = model.evaluate(X_test, y_test, verbose=0)
    print(f'Test Accuracy: {accuracy:.4f}')

    # Do more detailed classification report
    y_pred_probs = model.predict(X_test)
    y_pred = np.argmax(y_pred_probs, axis=1)
    y_true = np.argmax(y_test, axis=1)
    print(classification_report(y_true, y_pred))

    return model


# Create windowed data for time series models
def create_windowed_data(data: pd.DataFrame, window_size: int, step_size: int, batch_size: int = 10000):
    """
    Create overlapping windows of data for time series models in memory-efficient batches.
    Each window will have shape (window_size, num_features).
    
    Args:
        data: Input DataFrame
        window_size: Size of each window
        step_size: Step between windows
        batch_size: Number of windows to accumulate before returning (default 10000)
    
    Returns:
        Generator yielding tuples of (windows_batch, labels_batch)
    """
    windows_batch = []
    labels_batch = []
    batch_count = 0

    X_data = data[FEATURE_COLUMNS].values
    y_data = data['Label'].values

    for start in range(0, len(data) - window_size + 1, step_size):
        end = start + window_size
        window = X_data[start:end]  # Direct numpy indexing is faster
        label = pd.Series(y_data[start:end]).mode()[0]  # Most common label in the window
        
        windows_batch.append(window)
        labels_batch.append(label)
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


if __name__ == "__main__":
    print(tf.config.list_physical_devices('GPU'))

    #########################
    # Load and prepare data #
    #########################
    
    data = load_data('Data/Labelled')

    # Create windowed data for time series models (memory-efficient generator)
    window_size = 50  # e.g., 50 samples per window
    step_size = 25    # e.g., 50% overlap
    
    # Process in batches instead of loading all at once
    print("Processing windowed data in batches...")
    all_windows = []
    all_labels = []
    
    for X_batch, y_batch in create_windowed_data(data, window_size, step_size, batch_size=5000):
        all_windows.append(X_batch)
        all_labels.append(y_batch)
        print(f"  Processed batch: {len(all_windows)} batches so far...")
    
    # Concatenate all batches
    X_windows = np.concatenate(all_windows)
    y_windows = np.concatenate(all_labels)
    
    print(f"Total windows created: {len(X_windows)}")

    #############################
    # Train and evaluate models #
    #############################

    model = train_model_RandomForest(data)
    # model = train_model_CNN(X_windows, y_windows, window_size)

    # Save the trained model
    # import joblib
    # model_path = 'Data/Model/random_forest_model.joblib'
    # joblib.dump(model, model_path)
    # print(f"Trained Random Forest model saved to: {model_path}")