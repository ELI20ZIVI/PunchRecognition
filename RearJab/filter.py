import pandas as pd
import numpy as np
from pathlib import Path

INPUT_PATH = "Rear_jab.csv"
df = pd.read_csv(INPUT_PATH)

# Imposta la soglia per distinguere picchi da variazioni piccole
threshold = 0.45   # da adattare

z = df["accx(g)"].values
z_filtered = np.zeros_like(z)

# Mantieni solo i punti dove c’è una variazione significativa
for i in range(1, len(z)):
    if abs(z[i] - z[i-1]) >= threshold:
        z_filtered[i] = z[i]
    else:
        z_filtered[i] = 0

# Sovrascrivi la colonna originale invece di crearne una nuova
df["accx(g)"] = z_filtered

# Salva il risultato in un nuovo file con suffisso _filtered
in_path = Path(INPUT_PATH)
out_path = in_path.with_name(in_path.stem + "_filtered" + in_path.suffix)
df.to_csv(out_path, index=False)
print(f"File filtrato salvato in: {out_path}")
