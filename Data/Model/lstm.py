import os
import numpy as np
import pandas as pd
import tensorflow as tf
import matplotlib.pyplot as plt
import seaborn as sns
from sklearn.model_selection import train_test_split
from sklearn.metrics import classification_report, confusion_matrix
from sklearn.utils import class_weight
from sklearn.preprocessing import StandardScaler
from tensorflow.keras import layers, models, callbacks, regularizers
from tensorflow.keras.utils import to_categorical

# COSTANTI DI CONFIGURAZIONE
DATASET_PATH = 'Data/Labelled'
MODEL_SAVE_PATH = 'Data/Model/boxing_bilstm.keras'

# --- PARAMETRI DI TRAINING ---
# WINDOW_SIZE: Numero di frame consecutivi che formano una finestra temporale
WINDOW_SIZE = 50

# STEP_SIZE: Numero di frame di cui far scorrere la finestra per creare la successiva
# A 5 crea finestre sovrapposte catturando meglio le transizioni tra movimenti
STEP_SIZE = 5

# BATCH_SIZE: Numero di campioni processati insieme durante il training
BATCH_SIZE = 64

# EPOCHS: Numero di volte che l'intero dataset viene processato durante il training
EPOCHS = 70

# --- COLONNE DEI DATI ---
FEATURE_COLUMNS = [
    'RightHand_Accel_X', 'RightHand_Accel_Y', 'RightHand_Accel_Z',
    'LeftHand_Accel_X', 'LeftHand_Accel_Y', 'LeftHand_Accel_Z'
]

# --- MAPPATURA DELLE CLASSI ---
LABELS_MAP = {
    0: 'Guardia',
    1: 'Diretto SX',
    2: 'Diretto DX',
    3: 'Montante SX',
    4: 'Montante DX',
    5: 'Gancio SX',
    6: 'Gancio DX'
}

# FUNZIONE DI CARICAMENTO E PREPROCESSING DEI DATI
def load_data(file_path: str) -> pd.DataFrame:
    """
    Carica e preprocessa i dati da file CSV contenuti in una directory.
    
    Questa funzione:
    1. Cerca ricorsivamente tutti i file .csv nella directory specificata
    2. Carica ogni file CSV in un DataFrame pandas
    3. Verifica che ogni file contenga tutte le colonne necessarie (feature + Label)
    4. Concatena tutti i dati validi in un unico DataFrame
    5. Rimuove eventuali righe con valori mancanti (NaN)
    
    Args:
        file_path: Percorso della directory contenente i file CSV
    
    Returns:
        DataFrame contenente tutti i dati concatenati e puliti, oppure
        DataFrame vuoto se non ci sono dati validi
    """
    # Lista che conterrà i DataFrame caricati da ogni file
    frames = []
    
    # Verifica che il percorso esista e sia una directory
    if not os.path.isdir(file_path):
        print(f"Error: Directory {file_path} not found")
        return pd.DataFrame()
    
    # Scansiona ricorsivamente la directory e tutte le sottodirectory
    for root, _, files in os.walk(file_path):
        for fname in files:
            # Considera solo i file CSV
            if not fname.endswith('.csv'): continue
            
            try:
                # Legge il file CSV
                df = pd.read_csv(os.path.join(root, fname))
                
                # Verifica che il file contenga tutte le colonne necessarie
                if all(col in df.columns for col in FEATURE_COLUMNS + ['Label']):
                    # Mantiene solo le colonne necessarie e aggiunge il DataFrame alla lista
                    frames.append(df[FEATURE_COLUMNS + ['Label']])
            except Exception as e:
                print(f"Skipping {fname}: {e}")

    # Se sono stati caricati dei dati validi
    if frames:
        # Concatena tutti i DataFrame in uno solo, resettando gli indici
        full_data = pd.concat(frames, ignore_index=True)
        
        # Rimuove tutte le righe che contengono almeno un valore NaN
        full_data.dropna(inplace=True)
        
        return full_data
    
    return pd.DataFrame()

# FUNZIONE DI CREAZIONE DELLE FINESTRE TEMPORALI
def create_windows(data, window_size, step_size):
    """
    Crea finestre temporali dai dati grezzi per il training della rete neurale.
    
    Questa funzione implementa una strategia di sliding window con:
    - Normalizzazione StandardScaler delle feature
    - Soglia del 40% per l'assegnazione delle etichette
    - Logica di etichettatura che favorisce i pugni rispetto alla guardia
    
    Il processo di etichettatura funziona così:
    1. Per ogni finestra, conta le occorrenze di ogni etichetta
    2. Se c'è un pugno che occupa almeno il 40% della finestra, assegna quel pugno
    3. Altrimenti, assegna 'Guardia' (etichetta 0)
    
    Questa soglia del 40% è un compromesso tra:
    - Sensibilità: catturare anche pugni brevi
    - Pulizia: evitare falsi positivi da transizioni rumorose
    
    Args:
        data: DataFrame con le colonne delle feature e la colonna 'Label'
        window_size: Dimensione della finestra in numero di frame
        step_size: Passo di scorrimento della finestra
    
    Returns:
        Tuple (X, y) dove:
        - X: array numpy di forma (n_windows, window_size, n_features) con i dati normalizzati
        - y: array numpy di forma (n_windows,) con le etichette assegnate
    """
    # Liste per accumulare le finestre e le relative etichette
    X = []  # Feature (finestre temporali)
    y = []  # Labels (etichetta per ogni finestra)
    
    # --- NORMALIZZAZIONE DEI DATI ---
    # Trasformazione dati per avere media 0 e deviazione standard 1
    scaler = StandardScaler()
    data_x_scaled = scaler.fit_transform(data[FEATURE_COLUMNS].values)
    
    # Estrazione etichette come array numpy
    labels_np = data['Label'].values
    
    print(f"Creating windows (Step: {step_size}, Threshold: 40%)")
    
    # --- CREAZIONE FINESTRE SCORREVOLI ---
    for i in range(0, len(data) - window_size, step_size):
        # Estrae una finestra di dati normalizzati
        window_x = data_x_scaled[i : i + window_size]
        
        # Estrae etichette corrispondenti alla finestra
        window_y = labels_np[i : i + window_size]
        
        # --- STRATEGIA DI ETICHETTATURA DELLA FINESTRA ---
        # Conta quante volte appare ogni etichetta nella finestra
        vals, counts = np.unique(window_y, return_counts=True)
        counts_dict = dict(zip(vals, counts))  # Crea un dizionario {etichetta: conteggio}

        # Di default 'Guardia' (0) se non si trova un pugno dominante
        selected_label = 0
        
        # Identifica tutti i pugni presenti nella finestra (esclude la guardia)
        punch_candidates = [l for l in counts_dict.keys() if l != 0]
        
        if punch_candidates:
            # Trova il pugno che appare più frequentemente nella finestra
            best_punch = max(punch_candidates, key=lambda key: counts_dict[key])
            
            # SOGLIA DEL 40%: Assegna il pugno solo se occupa almeno il 40% della finestra
            # Evita di etichettare come pugno finestre con brevi transizioni
            if counts_dict[best_punch] >= (window_size * 0.40):
                selected_label = best_punch
            else:
                # Se il pugno non raggiunge la soglia mantieni 'Guardia'
                selected_label = 0
        
        # Aggiunge la finestra e la sua etichetta alle liste
        X.append(window_x)
        y.append(selected_label)
    
    return np.array(X), np.array(y)

# FUNZIONE DI COSTRUZIONE DEL MODELLO BI-LSTM
def build_bilstm_model(input_shape, num_classes):
    """
    Costruisce un modello ibrido CNN-BiLSTM per la classificazione di sequenze temporali.
    
    Architettura del modello:
    1. BLOCCO CNN: Estrae feature spaziali locali dalle sequenze
       - Due layer Conv1D per catturare pattern a diverse scale temporali
       - BatchNormalization per stabilizzare il training
       - MaxPooling per ridurre la dimensionalità
       - Dropout per prevenire overfitting
    
    2. BLOCCO BIDIRECTIONAL LSTM: Analizza la sequenza in entrambe le direzioni
       - Cattura dipendenze temporali a lungo termine
       - Bidirezionale per vedere sia "passato" che "futuro" nella sequenza
       - Utile per riconoscere l'inizio e la fine di un pugno
    
    3. BLOCCO DI CLASSIFICAZIONE: Produce la predizione finale
       - Layer Dense per combinare le feature estratte
       - Softmax per ottenere probabilità per ogni classe
    
    Args:
        input_shape: Tupla (window_size, num_features) che definisce la forma dell'input
        num_classes: Numero di classi da predire (7 nel nostro caso)
    
    Returns:
        Modello Keras compilato e pronto per il training
    """
    model = models.Sequential([
        # Layer di input che definisce la forma dei dati in ingresso
        layers.Input(shape=input_shape),
        
        # ========================================================================
        # PRIMO BLOCCO CNN - Estrazione di feature locali
        # ========================================================================
        # Conv1D con kernel size 7: cattura pattern temporali di media lunghezza
        # 64 filtri apprendono 64 diversi pattern dalla sequenza
        # padding='same' mantiene la lunghezza della sequenza
        # L2 regularization (0.001) previene l'overfitting penalizzando pesi troppo grandi
        layers.Conv1D(filters=64, kernel_size=7, activation='relu', padding='same', 
                      kernel_regularizer=regularizers.l2(0.001)),
        
        # BatchNormalization: normalizza l'output per accelerare il training e renderlo più stabile
        layers.BatchNormalization(),
        
        # MaxPooling riduce la dimensionalità temporale di un fattore 2
        # Mantiene solo i valori massimi riducendo il costo computazionale
        layers.MaxPooling1D(pool_size=2),
        
        # Dropout: disattiva casualmente il 30% dei neuroni durante il training
        # Previene l'overfitting forzando il modello a non dipendere troppo da specifici neuroni
        layers.Dropout(0.3),

        # ========================================================================
        # SECONDO BLOCCO CNN - Feature più astratte
        # ========================================================================
        # Secondo layer Conv1D con kernel size 5 e 128 filtri
        # Cattura pattern più complessi combinando le feature del layer precedente
        layers.Conv1D(filters=128, kernel_size=5, activation='relu', padding='same',
                      kernel_regularizer=regularizers.l2(0.001)),
        layers.BatchNormalization(),
        layers.MaxPooling1D(pool_size=2),
        layers.Dropout(0.3),

        # ========================================================================
        # BLOCCO BIDIRECTIONAL LSTM - Analisi temporale bidirezionale
        # ========================================================================
        # Bidirectional LSTM: processa la sequenza in entrambe le direzioni
        # - Forward: vede il passato e il presente
        # - Backward: vede il futuro e il presente
        # Importante per riconoscere i pugni, perché:
        # - La direzione forward vede la preparazione del pugno
        # - La direzione backward vede il ritorno alla guardia
        # 128 unità LSTM per catturare dipendenze complesse
        # return_sequences=False: ritorna solo l'output finale (non tutta la sequenza)
        layers.Bidirectional(layers.LSTM(128, return_sequences=False)),
        
        # Dropout più alto (0.4) perché LSTM tendono a overfittare
        layers.Dropout(0.4),

        # ========================================================================
        # BLOCCO DI CLASSIFICAZIONE - Output finale
        # ========================================================================
        # Layer denso (fully connected) con 64 neuroni
        # Combina tutte le feature estratte per la decisione finale
        layers.Dense(64, activation='relu', kernel_regularizer=regularizers.l2(0.001)),
        layers.BatchNormalization(),
        
        # Layer di output con softmax:
        # - num_classes neuroni (uno per ogni tipo di movimento)
        # - softmax produce probabilità che sommano a 1
        # - La classe con probabilità maggiore è la predizione
        layers.Dense(num_classes, activation='softmax')
    ])
    
    # --- COMPILAZIONE DEL MODELLO ---
    # Adam optimizer: algoritmo di ottimizzazione adattivo
    # learning_rate=0.001: tasso di apprendimento iniziale (sarà modificato dai callbacks)
    opt = tf.keras.optimizers.Adam(learning_rate=0.001)
    
    # Compilazione con:
    # - categorical_crossentropy: loss function per classificazione multi-classe
    # - accuracy: metrica per monitorare le prestazioni durante il training
    model.compile(optimizer=opt, loss='categorical_crossentropy', metrics=['accuracy'])
    
    return model

# ============================================================================
# MAIN
# ============================================================================
if __name__ == "__main__":
    print(tf.config.list_physical_devices('GPU'))

    # --- CARICAMENTO DEI DATI ---
    print("=" * 70)
    print("FASE 1: Caricamento dei dati")
    print("=" * 70)
    df = load_data(DATASET_PATH)
    
    # Se non ci sono dati, termina il programma
    if df.empty:
        print("Errore: Nessun dato caricato. Verificare il percorso del dataset.")
        exit()
    
    print(f"Dati caricati: {len(df)} frame totali")

    # --- CREAZIONE DELLE FINESTRE ---
    print("\n" + "=" * 70)
    print("FASE 2: Creazione delle finestre temporali")
    print("=" * 70)
    X, y = create_windows(df, WINDOW_SIZE, STEP_SIZE)
    print(f"Dataset shape -> X: {X.shape}, y: {y.shape}")
    print(f"Ogni finestra: {WINDOW_SIZE} frame × {len(FEATURE_COLUMNS)} feature")

    # --- DIVISIONE DEI DATI IN TRAINING E TEST ---
    print("\n" + "=" * 70)
    print("FASE 3: Divisione Train/Test")
    print("=" * 70)
    # Stratified split: mantiene la stessa proporzione di classi in train e test
    # - test_size=0.2: 20% dei dati per il test, 80% per il training
    # - random_state=42: rende la divisione riproducibile
    # - stratify=y: assicura che ogni classe sia rappresentata proporzionalmente
    X_train, X_test, y_train, y_test = train_test_split(
        X, y, test_size=0.2, random_state=42, stratify=y
    )
    print(f"Training set: {len(X_train)} finestre")
    print(f"Test set: {len(X_test)} finestre")

    # --- CONVERSIONE DELLE ETICHETTE IN ONE-HOT ENCODING ---
    num_classes = len(np.unique(y))
    y_train_cat = to_categorical(y_train, num_classes)
    y_test_cat = to_categorical(y_test, num_classes)
    print(f"Numero di classi: {num_classes}")

    # --- CALCOLO DEI PESI DELLE CLASSI ---
    print("\n" + "=" * 70)
    print("FASE 4: Calcolo dei pesi delle classi")
    print("=" * 70)
    # I pesi delle classi servono a bilanciare dataset sbilanciati    
    # compute_class_weight con 'balanced' calcola automaticamente i pesi come peso_classe = n_campioni_totali / (n_classi * n_campioni_classe)
    # Più importanza alle classi minoritarie
    class_weights_vals = class_weight.compute_class_weight(
        class_weight='balanced',
        classes=np.unique(y_train),
        y=y_train
    )
    class_weights_dict = dict(zip(np.unique(y_train), class_weights_vals))
    
    # --- FINE-TUNING DEI PESI ---
    # Riduce leggermente il peso dei pugni per evitare troppi falsi positivi
    for k in class_weights_dict:
        if k != 0:  # Pugni (classi 1-6)
            # Riduce il peso del 15% per bilanciare meglio precisione e recall
            class_weights_dict[k] = class_weights_dict[k] * 0.85
        else:  # Guardia (classe 0)
            # Mantiene il peso originale
            class_weights_dict[k] = class_weights_dict[k] * 1.0

    print("\nPesi ottimizzati per classe:")
    for k, v in class_weights_dict.items():
        print(f"  {LABELS_MAP.get(k, k)}: {v:.2f}")

    # --- COSTRUZIONE DEL MODELLO ---
    print("\n" + "=" * 70)
    print("FASE 5: Costruzione del modello")
    print("=" * 70)
    # Crea il modello con input shape (50 frame, 6 feature) e 7 classi di output
    model = build_bilstm_model((WINDOW_SIZE, 6), num_classes)
    print("Modello creato con successo")
    print(f"Input shape: ({WINDOW_SIZE}, 6)")
    print(f"Output: {num_classes} classi")
    
    # --- CONFIGURAZIONE DEI CALLBACKS ---
    print("\nConfigurazione callbacks:")
    
    # EARLY STOPPING: Interrompe il training se il modello non migliora
    # - monitor='val_loss': osserva la loss sul validation set
    # - patience=10: aspetta 10 epoche senza miglioramento prima di fermarsi
    # - restore_best_weights=True: alla fine, ripristina i pesi dell'epoca migliore
    # - verbose=1: stampa un messaggio quando si attiva
    # Questo previene l'overfitting e risparmia tempo evitando training inutili
    early_stopping = callbacks.EarlyStopping(
        monitor='val_loss', patience=10, restore_best_weights=True, verbose=1
    )
    print("  - Early Stopping (patience=10)")
    
    # REDUCE LEARNING RATE ON PLATEAU: Riduce il learning rate quando il training si stabilizza
    # - monitor='val_loss': osserva la validation loss
    # - factor=0.2: moltiplica il learning rate per 0.2 (riduzione dell'80%)
    # - patience=4: aspetta 4 epoche senza miglioramento prima di ridurre
    # - min_lr=0.00001: learning rate minimo
    # Ridurre il learning rate aiuta il modello a fare aggiustamenti più fini
    # quando si avvicina all'ottimo
    reduce_lr = callbacks.ReduceLROnPlateau(
        monitor='val_loss', factor=0.2, patience=4, min_lr=0.00001, verbose=1
    )
    print("  - Reduce LR on Plateau (factor=0.2, patience=4)")

    # --- TRAINING DEL MODELLO ---
    print("\n" + "=" * 70)
    print("FASE 6: Training del modello")
    print("=" * 70)
    print(f"Epoche massime: {EPOCHS}")
    print(f"Batch size: {BATCH_SIZE}")
    print("Inizio training...\n")
    
    # Il metodo fit() addestra il modello
    # - X_train, y_train_cat: dati e etichette di training
    # - epochs: numero massimo di epoche (può terminare prima con early stopping)
    # - batch_size: numero di campioni processati prima di aggiornare i pesi
    # - validation_split=0.2: usa il 20% del training set per la validazione
    #   (quindi 80% training effettivo, 20% validation, 20% test)
    # - class_weight: dizionario dei pesi per bilanciare le classi
    # - callbacks: lista di callback (early stopping e reduce LR)
    # - verbose=1: mostra la progress bar durante il training
    history = model.fit(
        X_train, y_train_cat,
        epochs=EPOCHS,
        batch_size=BATCH_SIZE,
        validation_split=0.2,
        class_weight=class_weights_dict,
        callbacks=[early_stopping, reduce_lr],
        verbose=1
    )

    # --- VALUTAZIONE DEL MODELLO ---
    print("\n" + "=" * 70)
    print("FASE 7: Valutazione sul test set")
    print("=" * 70)
    
    # Calcola loss e accuracy sul test set
    # Questo ci dice quanto bene il modello generalizza su dati mai visti
    loss, acc = model.evaluate(X_test, y_test_cat, verbose=0)
    print(f"\nTest Loss: {loss:.4f}")
    print(f"Test Accuracy: {acc:.4f} ({acc*100:.2f}%)")

    # --- GENERAZIONE DELLE PREDIZIONI ---
    # model.predict() ritorna le probabilità per ogni classe
    # np.argmax() trova l'indice della classe con probabilità maggiore
    # axis=1 significa che opera lungo le righe (per ogni predizione)
    y_pred = np.argmax(model.predict(X_test), axis=1)

    # --- CLASSIFICATION REPORT ---
    # Report dettagliato con metriche per ogni classe:
    # - Precision: tra i campioni predetti come classe X, quanti sono davvero X?
    # - Recall: tra tutti i campioni di classe X, quanti sono stati correttamente identificati?
    # - F1-Score: media armonica di precision e recall
    # - Support: numero di campioni di quella classe nel test set
    print("\n" + "=" * 70)
    print("Classification Report")
    print("=" * 70)
    print(classification_report(y_test, y_pred, target_names=[LABELS_MAP[i] for i in range(num_classes)]))

    # --- CONFUSION MATRIX ---
    # Matrice che mostra:
    # - Righe: classi reali
    # - Colonne: classi predette
    # - Diagonale: predizioni corrette
    # - Off-diagonal: errori (es. "Diretto SX" classificato come "Gancio SX")
    print("\n" + "=" * 70)
    print("FASE 8: Visualizzazione Confusion Matrix")
    print("=" * 70)
    cm = confusion_matrix(y_test, y_pred)
    
    # Crea una heatmap della confusion matrix
    plt.figure(figsize=(10, 8))
    sns.heatmap(cm, annot=True, fmt='d', cmap='Purples', 
                xticklabels=[LABELS_MAP[i] for i in range(num_classes)],
                yticklabels=[LABELS_MAP[i] for i in range(num_classes)])
    plt.title('Confusion Matrix (Bi-LSTM)')
    plt.xlabel('Classe Predetta')
    plt.ylabel('Classe Reale')
    plt.tight_layout()
    plt.show()
    
    # --- SALVATAGGIO DEL MODELLO ---
    print("\n" + "=" * 70)
    print("FASE 9: Salvataggio del modello")
    print("=" * 70)
    # Salva l'intero modello (architettura + pesi + ottimizzatore) in formato Keras nativo (.keras)
    model.save(MODEL_SAVE_PATH)
    print(f"Modello salvato in: {MODEL_SAVE_PATH}")
    print("\nTraining completato con successo!")
    print("=" * 70)