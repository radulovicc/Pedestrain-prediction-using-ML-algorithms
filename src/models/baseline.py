"""
Baseline za multistep predikciju (iterativni pristup).
Strategija: Ponavlja poslednji pomeraj za svih PRED_STEPS koraka.
Poslednji pomeraj = -rel_x_7 (delta od t-1 do t)
"""

import numpy as np
import pandas as pd
import os 
import sys

sys.path.insert(0, os.path.dirname(os.path.dirname(os.path.dirname(os.path.abspath(__file__)))))

from src.config import WINDOW_SIZE, PRED_STEPS
from src.evaluation import izracunaj_metrike, plot_prediction, plot_trajektorija

# =============================================================================
# KONFIGURACIJA
# =============================================================================

DATA_DIR = 'data/processed'
OUTPUT_DIR = os.path.join('results', 'models', 'baseline', f'window_{WINDOW_SIZE}')
os.makedirs(OUTPUT_DIR, exist_ok=True)

X_train = np.load(os.path.join(DATA_DIR, 'X_train.npy'))
y_train = np.load(os.path.join(DATA_DIR, 'y_train.npy'))
X_test = np.load(os.path.join(DATA_DIR, 'X_test.npy'))
y_test = np.load(os.path.join(DATA_DIR, 'y_test.npy'))
df_test = pd.read_csv(os.path.join(DATA_DIR, 'test.csv'))

print("\nPodaci učitani:")
print(f"  X_train: {X_train.shape}")
print(f"  y_train: {y_train.shape}")
print(f"  X_test:  {X_test.shape}")
print(f"  y_test:  {y_test.shape}")

# =============================================================================
# 2. BASELINE PREDIKCIJA
# =============================================================================
# Poslednji pomeraj (od T-1 do T) = -rel_x_{N-1} i -rel_y_{N-1}
# rel_x kolone: indeksi 0 .. (WINDOW_SIZE-2)  -> poslednja je WINDOW_SIZE-2
# rel_y kolone: indeksi (WINDOW_SIZE-1) .. (2*(WINDOW_SIZE-1)-1) -> poslednja je 2*(WINDOW_SIZE-1)-1

idx_rel_x_last = WINDOW_SIZE - 2
idx_rel_y_last = 2*(WINDOW_SIZE - 1) - 1

#Za svaku putanju uzimamo pomeraj u poslednjem trenutku
delta_x_last = -X_train[:, idx_rel_x_last]
delta_y_last = -X_train[:, idx_rel_y_last]
y_train_pred = np.column_stack([delta_x_last] * PRED_STEPS + [delta_y_last] * PRED_STEPS)

delta_x_last_test = -X_test[:, idx_rel_x_last]
delta_y_last_test = -X_test[:, idx_rel_y_last]
y_test_pred = np.column_stack([delta_x_last_test] * PRED_STEPS + [delta_y_last_test] * PRED_STEPS)

print(f"\nPredikcije napravljene (shape: {y_test_pred.shape})")
print(f"KOrisceni indeksi: rel_x_last={idx_rel_x_last}, rel_y_last={idx_rel_y_last}")

# =============================================================================
# 3. METRIKE
# =============================================================================
print(f"\n{'='*60}")
print(" EVALUACIJA")
print(f"{'='*60}")

train_metrics = izracunaj_metrike(y_train, y_train_pred, 'TRAIN', PRED_STEPS)
test_metrics = izracunaj_metrike(y_test, y_test_pred, 'TEST', PRED_STEPS)

# =============================================================================
# 4. VIZUELIZACIJA
# =============================================================================
print(f"\n{'='*60}")
print(" VIZUELIZACIJA")
print(f"{'='*60}")

plot_prediction(
    X=X_test, y_true=y_test, y_pred=y_test_pred,
    df_info=df_test, naziv_modela='Baseline (iterativni)',
    output_path=OUTPUT_DIR, n_primera=6, pred_steps=PRED_STEPS
)

plot_trajektorija(
    X=X_test, y_true=y_test, y_pred=y_test_pred,
    df_info=df_test, naziv_modela='Baseline (iterativni)',
    output_path=OUTPUT_DIR, n_uzastopnih=4, pred_steps=PRED_STEPS
)

# =============================================================================
# 5. ČUVANJE REZULTATA
# =============================================================================
np.save(os.path.join(OUTPUT_DIR, 'y_train_pred.npy'), y_train_pred)
np.save(os.path.join(OUTPUT_DIR, 'y_test_pred.npy'), y_test_pred)
print(f"\nRezultati sacuvani u '{OUTPUT_DIR}/'")

# =============================================================================
# 6. ZAKLJUČAK
# =============================================================================
print(f"\n{'='*60}")
print(" ZAKLJUČAK")
print(f"{'='*60}")
print(f"Baseline (iterativni, {PRED_STEPS} koraka) na TEST skupu:")
print(f"    ADE: {test_metrics['ade']:.6f}")
print(f"    FDE: {test_metrics['fde']:.6f}")
print(f"    ADE po koraku: {[f'{x:.6f}' for x in test_metrics['ade_po_koraku']]}")

