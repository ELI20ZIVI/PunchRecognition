import csv
import os

START = 0
END = 0
ORIGINAL_PATH = "Lead_uppercut.csv"
FILTERED_PATH = "Lead_uppercut_filtered.csv"
SPLIT_PATH = "Lead_uppercut_split"

# Crea la directory se non esiste
os.makedirs(SPLIT_PATH, exist_ok=True)

# Contatore per numerare i file
punch_counter = 1

with open(FILTERED_PATH, 'r', newline='', encoding='utf-8') as file:
    reader = csv.reader(file)
    
    header = next(reader)
    indice_accZ = header.index("accz(g)")
    numero_riga = 1
    
    for riga in reader:
        valore_accZ = riga[indice_accZ]
        
        if valore_accZ != '0.0':
            START = numero_riga - 60
            END = START + 180

            # Conta gli zeri finali leggendo le righe da END a START
            zeroCounter = 0
            with open(FILTERED_PATH, 'r', newline='', encoding='utf-8') as temp_file:
                temp_reader = csv.reader(temp_file)
                next(temp_reader)  # Skip header
                righe = list(temp_reader)
                
                for i in range(min(END, len(righe) - 1), START, -1):
                    if i < len(righe) and righe[i][indice_accZ] == '0.0':
                        zeroCounter += 1
                    else:
                        break

            END = END - zeroCounter + 101
            print(f"Nuovo intervallo da riga {START} a riga {END} dopo la rimozione degli zeri finali")

            with open(ORIGINAL_PATH, 'r', newline='', encoding='utf-8') as file:
                originalReader = csv.reader(file)
                header = next(originalReader)

                # write the split file
                with open(f"{SPLIT_PATH}/punch_{punch_counter}.csv", 'w', newline='', encoding='utf-8') as split_file:
                    writer = csv.writer(split_file)
                    writer.writerow(header)

                    numero_riga_split = 0
                    for riga_split in originalReader:
                        if numero_riga_split >= START and numero_riga_split <= END:
                            writer.writerow(riga_split)
                        numero_riga_split += 1

            punch_counter += 1

            # Salta le righe fino a END per continuare la ricerca
            for _ in range(END - numero_riga):
                try:
                    next(reader)
                    numero_riga += 1
                except StopIteration:
                    break

        numero_riga += 1