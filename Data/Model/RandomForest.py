# RandomForest libraries
from sklearn.ensemble import RandomForestClassifier
from sklearn.model_selection import train_test_split
from sklearn.metrics import classification_report

# MLP libraries
from sklearn.neural_network import MLPClassifier

# CNN libraries
import tensorflow as tf
from keras import layers, models, regularizers
from keras.callbacks import EarlyStopping
from keras.utils import to_categorical
import keras_tuner as kt

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
def train_model_CNN_hyperparameters(X: np.ndarray, y: np.ndarray, window_size: int) -> models.Sequential:
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

    # Set up Hyperparameters
    def build_model(hp):
        model = models.Sequential([
            layers.Input(shape=(window_size, X.shape[2])),
            layers.Conv1D(
                filters=hp.Int('filters_1', min_value=16, max_value=128, step=16),
                kernel_size=hp.Int('kernel_size_1', min_value=5, max_value=15, step=2),
                activation='relu'
            ),
            layers.MaxPooling1D(pool_size=4),
            layers.Dropout(0.3),

            layers.Conv1D(
                filters=hp.Int('filters_2', min_value=32, max_value=256, step=32),
                kernel_size=hp.Int('kernel_size_2', min_value=3, max_value=7, step=2),
                activation='relu'
            ),
            layers.MaxPooling1D(pool_size=2),
            layers.Dropout(0.3),

            layers.GlobalAveragePooling1D(),
            layers.Dense(
                hp.Int('dense_units', min_value=32, max_value=256, step=32),
                activation='relu'
            ),
            layers.Dense(len(np.unique(y)), activation='softmax')
        ])
        model.compile(optimizer='adam', loss='categorical_crossentropy', metrics=['accuracy'])
        return model
    
    tuner = kt.Hyperband(
        build_model,
        objective='val_accuracy',
        max_epochs=50,
        factor=3,
        directory='cnn_tuning',
        project_name='punch_recognition',
        overwrite=True
    )

    early_stopping = EarlyStopping(monitor='val_loss', patience=5, restore_best_weights=True)

    # Hyperparameter tuning
    tuner.search(X_train, y_train, epochs=50, batch_size=32, validation_split=0.2, callbacks=[early_stopping], verbose=1)

    # Get the best hyperparameters and train the final model
    best_hps = tuner.get_best_hyperparameters(num_trials=1)[0]
    print(f"Best hyperparameters: {best_hps.values}")

    model = tuner.hypermodel.build(best_hps)
    model.compile(optimizer='adam', loss='categorical_crossentropy', metrics=['accuracy'])

    model.fit(X_train, y_train, epochs=50, batch_size=32, validation_split=0.2, callbacks=[early_stopping], verbose=1)

    # Evaluate the model
    loss, accuracy = model.evaluate(X_test, y_test, verbose=0)
    print(f'Test Accuracy: {accuracy:.4f}')

    # Do more detailed classification report
    y_pred_probs = model.predict(X_test)
    y_pred = np.argmax(y_pred_probs, axis=1)
    y_true = np.argmax(y_test, axis=1)
    print(classification_report(y_true, y_pred))

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
        layers.Conv1D(filters=64, kernel_size=9, activation='relu'),
        layers.MaxPooling1D(pool_size=2),
        layers.Dropout(0.3), # Previene l'overfitting (molto comune sui dati dei sensori)

        # --- 2. Blocco Convoluzionale "Grezzo" ---
        # Cattura pattern più lunghi (es. il movimento di caricamento del braccio)
        # filters=64 aumenta la capacità di astrazione
        layers.Conv1D(filters=256, kernel_size=3, activation='relu'),
        layers.MaxPooling1D(pool_size=2),
        layers.Dropout(0.3),

        # --- 3. Feature Aggregation ---
        # GlobalAveragePooling è meglio di Flatten per le time series:
        # riduce drasticamente i parametri e rende il modello più robusto.
        layers.GlobalAveragePooling1D(),

        # --- 4. Classificazione ---
        layers.Dense(192, activation='relu'),
        layers.Dense(len(np.unique(y)), activation='softmax') # Output: probabilità per ogni colpo
    ])

    model.compile(optimizer='adam', loss='categorical_crossentropy', metrics=['accuracy'])

    early_stopping = EarlyStopping(monitor='val_loss', patience=5, restore_best_weights=True)

    model.fit(X_train, y_train, epochs=50, batch_size=32, validation_split=0.2, callbacks=[early_stopping], verbose=1)

    # Evaluate the model
    loss, accuracy = model.evaluate(X_test, y_test, verbose=0)
    print(f'Test Accuracy: {accuracy:.4f}')

    # Do more detailed classification report
    y_pred_probs = model.predict(X_test)
    y_pred = np.argmax(y_pred_probs, axis=1)
    y_true = np.argmax(y_test, axis=1)
    print(classification_report(y_true, y_pred))

    return model


# Train MLP model
def train_model_MLP(data: pd.DataFrame) -> MLPClassifier:
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

    model = MLPClassifier(solver='adam', hidden_layer_sizes=(100, 50), max_iter=300, random_state=42, early_stopping=True, n_iter_no_change=10, verbose=True)
    model.fit(X_train, y_train)

    y_pred = model.predict(X_test)
    print(classification_report(y_test, y_pred))

    return model


# Train LSTM model
def train_model_lstm(X: np.ndarray, y: np.ndarray, window_size: int) -> models.Sequential:
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
        layers.Input(shape=(window_size, X.shape[2])),
        
        # --- Miglioramento 1: Bidirezionalità ---
        # Legge la sequenza in avanti e indietro. Raddoppia i parametri ma aumenta l'accuratezza.
        layers.Bidirectional(layers.LSTM(64, return_sequences=True)),
        
        # --- Miglioramento 2: Regolarizzazione ---
        layers.Dropout(0.3),      # Spegne il 30% dei neuroni per evitare overfitting
        layers.BatchNormalization(), # Normalizza i dati internamente tra i layer

        layers.Bidirectional(layers.LSTM(32, return_sequences=False)), # return_sequences=False perché è l'ultima LSTM
        layers.Dropout(0.3),
        layers.BatchNormalization(),

        layers.Dense(64, activation='relu', kernel_regularizer=regularizers.l2(0.001)), # L2 aiuta se hai pochi dati
        layers.Dropout(0.3),

        layers.Dense(len(np.unique(y)), activation='softmax')
    ])

    # Compile and train the model
    model.compile(optimizer='adam', loss='categorical_crossentropy', metrics=['accuracy'])

    # Early stopping to prevent overfitting
    early_stopping = EarlyStopping(monitor='val_loss', patience=5, restore_best_weights=True)

    model.fit(X_train, y_train, epochs=50, batch_size=32, validation_split=0.2, callbacks=[early_stopping], verbose=1)

    # Evaluate the model
    loss, accuracy = model.evaluate(X_test, y_test, verbose=0)
    print(f'Test Accuracy: {accuracy:.4f}')

    # Do more detailed classification report
    y_pred_probs = model.predict(X_test)
    y_pred = np.argmax(y_pred_probs, axis=1)
    y_true = np.argmax(y_test, axis=1)
    print(classification_report(y_true, y_pred))

    return model


def train_model_cnn_lstm(X: np.ndarray, y: np.ndarray, window_size: int, base_model: models.Sequential) -> models.Sequential:
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

    # Reuse base CNN model layers
    if not base_model.layers:
        raise ValueError("Base model has no layers to reuse.")
    
    input_model = models.Sequential(base_model.layers)

    # Set layers to non-trainable
    for layer in input_model.layers:
        layer.trainable = False

    # Remove final dense layer
    input_model.pop()   # Remove final dense layer
    input_model.pop()   # Remove previous dense layer
    input_model.pop()   # Remove GlobalAveragePooling1D
    input_model.pop()   # Remove Dropout
    input_model.pop()   # Remove MaxPooling1D

    model = models.Sequential([
        input_model,

        # --- Miglioramento 1: Bidirezionalità ---
        # Legge la sequenza in avanti e indietro. Raddoppia i parametri ma aumenta l'accuratezza.
        layers.Bidirectional(layers.LSTM(256, return_sequences=True)),
        
        # --- Miglioramento 2: Regolarizzazione ---
        layers.Dropout(0.3),      # Spegne il 30% dei neuroni per evitare overfitting
        layers.BatchNormalization(), # Normalizza i dati internamente tra i layer

        layers.Bidirectional(layers.LSTM(192, return_sequences=False)), # return_sequences=False perché è l'ultima LSTM
        layers.Dropout(0.3),
        layers.BatchNormalization(),

        layers.Dense(64, activation='relu', kernel_regularizer=regularizers.l2(0.001)), # L2 aiuta se hai pochi dati
        layers.Dropout(0.3),

        layers.Dense(len(np.unique(y)), activation='softmax')
    ])

    # Compile and train the model
    model.compile(optimizer='adam', loss='categorical_crossentropy', metrics=['accuracy'])

    # Early stopping to prevent overfitting
    early_stopping = EarlyStopping(monitor='val_loss', patience=5, restore_best_weights=True)
    model.fit(X_train, y_train, epochs=50, batch_size=32, validation_split=0.2, callbacks=[early_stopping], verbose=1)

    # Evaluate the model
    loss, accuracy = model.evaluate(X_test, y_test, verbose=0)
    print(f'Test Accuracy: {accuracy:.4f}')

    # Do more detailed classification report
    y_pred_probs = model.predict(X_test)
    y_pred = np.argmax(y_pred_probs, axis=1)
    y_true = np.argmax(y_test, axis=1)
    print(classification_report(y_true, y_pred))

    return model

def train_model_tcn(X: np.ndarray, y: np.ndarray, window_size: int) -> models.Sequential:
    """
    Train a Temporal Convolutional Network (TCN) model for time series classification.
    
    TCN uses dilated convolutions to capture temporal dependencies at different scales.
    Can look both forward (future) and backward (past) to predict each time step.
    Output shape matches input shape: (batch_size, window_size, num_classes).
    
    Args:
        X: Input features with shape (n_samples, window_size, n_features)
        y: Labels with shape (n_samples, window_size) for per-timestep prediction
        window_size: Size of each time window
    """
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

    # Convert labels to one-hot encoding for each timestep
    num_classes = len(np.unique(y))
    
    # y has shape (n_samples, window_size), convert to (n_samples, window_size, num_classes)
    y_train_cat = np.zeros((y_train.shape[0], y_train.shape[1], num_classes))
    y_test_cat = np.zeros((y_test.shape[0], y_test.shape[1], num_classes))
    
    for i in range(y_train.shape[0]):
        y_train_cat[i] = to_categorical(y_train[i], num_classes)
    for i in range(y_test.shape[0]):
        y_test_cat[i] = to_categorical(y_test[i], num_classes)

    model = models.Sequential([
        layers.Input(shape=(window_size, X.shape[2])),
        
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
        layers.Conv1D(filters=num_classes, kernel_size=1, padding='same', activation='softmax')
    ])

    # Compile and train the model
    model.compile(optimizer='adam', loss='categorical_crossentropy', metrics=['accuracy'])

    # Early stopping to prevent overfitting
    early_stopping = EarlyStopping(monitor='val_loss', patience=5, restore_best_weights=True)
    model.fit(X_train, y_train_cat, epochs=100, batch_size=32, validation_split=0.2, callbacks=[early_stopping], verbose=1)

    # Evaluate the model
    loss, accuracy = model.evaluate(X_test, y_test_cat, verbose=0)
    print(f'Test Accuracy: {accuracy:.4f}')

    # Do more detailed classification report
    y_pred_probs = model.predict(X_test)
    # y_pred_probs shape: (n_test_samples, window_size, num_classes)
    y_pred = np.argmax(y_pred_probs, axis=2).flatten()  # Flatten across all timesteps
    y_true = np.argmax(y_test_cat, axis=2).flatten()
    print(classification_report(y_true, y_pred))

    print("\n--- TCN Model Trained ---")
    print(model.summary())

    return model


# Create windowed data for time series models
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


if __name__ == "__main__":
    print(tf.config.list_physical_devices('GPU'))

    #########################
    # Load and prepare data #
    #########################
    
    data = load_data('../Labelled/Downsampled')

    # Create windowed data for time series models (memory-efficient generator)
    window_size = 60   # e.g., 120 samples per window
    step_size = 6      # e.g., 10% overlap
    
    # Process in batches instead of loading all at once
    print("Processing windowed data in batches...")
    all_windows = []
    all_labels = []
    
    # Use per_timestep=True for TCN to get all labels per window
    # NOTE: If you are not using TCN, set per_timestep=False to get single label per window
    for X_batch, y_batch in create_windowed_data(data, window_size, step_size, batch_size=5000, per_timestep=True):
        all_windows.append(X_batch)
        all_labels.append(y_batch)
        print(f"  Processed batch: {len(all_windows)} batches so far...")
    
    # Concatenate all batches
    X_windows = np.concatenate(all_windows)
    y_windows = np.concatenate(all_labels)  # Shape: (n_samples, window_size)
    
    print(f"Total windows created: {len(X_windows)}")
    print(f"X_windows shape: {X_windows.shape}")
    print(f"y_windows shape: {y_windows.shape}")

    #############################
    # Train and evaluate models #
    #############################

    # model = train_model_RandomForest(data)
    # model = train_model_MLP(data)
    # model = train_model_CNN(X_windows, y_windows, window_size)
    # model = train_model_CNN_hyperparameters(X_windows, y_windows, window_size)
    # model = train_model_lstm(X_windows, y_windows, window_size)
    
    ### Best hyperparameters: {'filters_1': 64, 'kernel_size_1': 9, 'filters_2': 256, 'kernel_size_2': 3, 'dense_units': 192, 'tuner/epochs': 50, 'tuner/initial_epoch': 17, 'tuner/bracket': 1, 'tuner/round': 1, 'tuner/trial_id': '0081'}
    ### Old parameters: 'filters_1': 32, 'kernel_size_1': 3, 'filters_2': 64, 'kernel_size_2': 5, 'dense_units': 64

    # model = train_model_cnn_lstm(X_windows, y_windows, window_size, base_model=train_model_CNN(X_windows, y_windows, window_size))

    model = train_model_tcn(X_windows, y_windows, window_size)

    # Save the trained model
    model.save('model_tcn.keras')
    print(f"Trained Random Forest model saved to: model_tcn.keras")