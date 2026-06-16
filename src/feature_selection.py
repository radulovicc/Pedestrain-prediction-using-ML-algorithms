"""
Odabir najznačajnijih atributa.
Poredi XGBoost treniran na SVIM featurima (rel + KNN) vs SAMO rel koordinatama.
Cilj: pokazati koliko KNN socijalni featurei zaista doprinose.
"""

import numpy as np
import pandas as pd
import os
import sys
sys.path.insert(0, os.path.dirname(os.path.dirname(os.path.abspath(__file__))))

from xgboost import XGBRegressor
from sklearn.metrics import r2_score
from src.config import WINDOW_SIZE, PRED_STEPS
from src.evaluation import izracunaj_metrike

DATA_DIR = 'data/processed'
OUTPUT_DIR = os.path.join('results', 'models', 'feature_selection', f'window_{WINDOW_SIZE}')
os.makedirs(OUTPUT_DIR, exist_ok=True)

ps = PRED_STEPS
N_REL = WINDOW_SIZE - 1
N_REL_TOTAL = 2 * N_REL          # broj rel kolona (rel_x + rel_y)
IDX_VX = WINDOW_SIZE - 2
IDX_VY = 2 * (WINDOW_SIZE - 1) - 1

# Najbolji XGBoost parametri (iz tuninga)
XGB_PARAMS = dict(
    n_estimators=2000, learning_rate=0.005, max_depth=3,
    subsample=0.6, colsample_bytree=0.8, reg_alpha=2, reg_lambda=2,
    random_state=42, n_jobs=-1, verbosity=0
)

# =============================================================================
# 1. UČITAVANJE
# =============================================================================
print("=" * 70)
print(f" ODABIR ATRIBUTA — Svi featurei vs samo rel koordinate (N={WINDOW_SIZE})")
print("=" * 70)

X_train = np.load(os.path.join(DATA_DIR, 'X_train.npy'))
y_train = np.load(os.path.join(DATA_DIR, 'y_train.npy'))
X_val   = np.load(os.path.join(DATA_DIR, 'X_val.npy'))
y_val   = np.load(os.path.join(DATA_DIR, 'y_val.npy'))
X_test  = np.load(os.path.join(DATA_DIR, 'X_test.npy'))
y_test  = np.load(os.path.join(DATA_DIR, 'y_test.npy'))

X_full = np.vstack([X_train, X_val])
y_full = np.vstack([y_train, y_val])

print(f"\nUkupno featura: {X_full.shape[1]} ({N_REL_TOTAL} rel + {X_full.shape[1]-N_REL_TOTAL} KNN)")

# =============================================================================
# Pomoćne funkcije
# =============================================================================
def izracunaj_ade(y_true, y_pred):
    return np.mean(np.sqrt(
        (y_true[:,:ps]-y_pred[:,:ps])**2 + (y_true[:,ps:]-y_pred[:,ps:])**2
    ))

def izracunaj_fde(y_true, y_pred):
    return np.mean(np.sqrt(
        (y_true[:,ps-1]-y_pred[:,ps-1])**2 + (y_true[:,2*ps-1]-y_pred[:,2*ps-1])**2
    ))

def per_sample_ade(y_true, y_pred):
    err = np.sqrt((y_true[:,:ps]-y_pred[:,:ps])**2 + (y_true[:,ps:]-y_pred[:,ps:])**2)
    return np.mean(err, axis=1)

# Sample weights (isto kao u glavnom XGBoost modelu)
self_vx = -X_full[:, IDX_VX]
self_vy = -X_full[:, IDX_VY]
expected_dx = self_vx * ps
expected_dy = self_vy * ps
actual_dx = y_full[:, ps-1]
actual_dy = y_full[:, 2*ps-1]
deviation_train = np.sqrt((actual_dx-expected_dx)**2 + (actual_dy-expected_dy)**2)
weights = np.where(deviation_train > 0.5, 3.0, 1.0)

# =============================================================================
# 2. TRENING — dva modela
# =============================================================================
# Model A: svi featurei
print("\n[A] Treniram XGBoost na SVIM featurima...")
model_svi = XGBRegressor(**XGB_PARAMS)
model_svi.fit(X_full, y_full, sample_weight=weights)
pred_svi = model_svi.predict(X_test)

# Model B: samo rel koordinate (prvih N_REL_TOTAL kolona)
print("[B] Treniram XGBoost SAMO na rel koordinatama...")
X_full_rel = X_full[:, :N_REL_TOTAL]
X_test_rel = X_test[:, :N_REL_TOTAL]
model_rel = XGBRegressor(**XGB_PARAMS)
model_rel.fit(X_full_rel, y_full, sample_weight=weights)
pred_rel = model_rel.predict(X_test_rel)

# =============================================================================
# 3. POREĐENJE — ukupno
# =============================================================================
print(f"\n{'='*70}")
print(" POREĐENJE — ukupne metrike (test skup)")
print(f"{'='*70}")

ade_svi, fde_svi = izracunaj_ade(y_test, pred_svi), izracunaj_fde(y_test, pred_svi)
ade_rel, fde_rel = izracunaj_ade(y_test, pred_rel), izracunaj_fde(y_test, pred_rel)
r2_svi, r2_rel = r2_score(y_test, pred_svi), r2_score(y_test, pred_rel)

print(f"\n{'Feature set':<28} {'#feat':>6} {'ADE':>10} {'FDE':>10} {'R²':>10}")
print("-" * 66)
print(f"{'Svi (rel + KNN)':<28} {X_full.shape[1]:>6} {ade_svi:>10.6f} {fde_svi:>10.6f} {r2_svi:>10.4f}")
print(f"{'Samo rel koordinate':<28} {N_REL_TOTAL:>6} {ade_rel:>10.6f} {fde_rel:>10.6f} {r2_rel:>10.4f}")

promena_ade = (ade_rel - ade_svi) / ade_svi * 100
print(f"\n Promena ADE (rel vs svi): {promena_ade:+.2f}%")
print(f" (pozitivno = 'samo rel' je lošije, negativno = 'samo rel' je bolje)")

# =============================================================================
# 4. POREĐENJE — po kategorijama putanje
# =============================================================================
print(f"\n{'='*70}")
print(" POREĐENJE — po tipu putanje (tu bi KNN trebalo da pomaže)")
print(f"{'='*70}")

# Klasifikacija test skupa
self_vx_t = -X_test[:, IDX_VX]
self_vy_t = -X_test[:, IDX_VY]
exp_dx = self_vx_t * ps
exp_dy = self_vy_t * ps
act_dx = y_test[:, ps-1]
act_dy = y_test[:, 2*ps-1]
deviation = np.sqrt((act_dx-exp_dx)**2 + (act_dy-exp_dy)**2)

pravo_mask = deviation <= 0.3
blago_mask = (deviation > 0.3) & (deviation <= 1.0)
ostro_mask = deviation > 1.0

ade_svi_ps = per_sample_ade(y_test, pred_svi)
ade_rel_ps = per_sample_ade(y_test, pred_rel)

print(f"\n{'Kategorija':<22} {'Broj':>6} {'ADE svi':>10} {'ADE rel':>10} {'Razlika':>10}")
print("-" * 60)
for naziv, mask in [('Prava linija', pravo_mask),
                    ('Blago skretanje', blago_mask),
                    ('Ostro skretanje', ostro_mask)]:
    if mask.sum() > 0:
        a_svi = np.mean(ade_svi_ps[mask])
        a_rel = np.mean(ade_rel_ps[mask])
        print(f"{naziv:<22} {mask.sum():>6} {a_svi:>10.6f} {a_rel:>10.6f} {a_rel-a_svi:>+10.6f}")

# =============================================================================
# 5. ČUVANJE
# =============================================================================
metrike = izracunaj_metrike(y_test, pred_svi, 'XGB — svi featurei', ps)
metrike_rel = izracunaj_metrike(y_test, pred_rel, 'XGB — samo rel', ps)

rezultat = pd.DataFrame([
    {'feature_set': 'svi (rel+KNN)', 'n_features': X_full.shape[1],
     'ade': ade_svi, 'fde': fde_svi, 'r2': r2_svi},
    {'feature_set': 'samo rel', 'n_features': N_REL_TOTAL,
     'ade': ade_rel, 'fde': fde_rel, 'r2': r2_rel},
])
rezultat.to_csv(os.path.join(OUTPUT_DIR, 'feature_selection.csv'), index=False)
print(f"\n Rezultati: {OUTPUT_DIR}/feature_selection.csv")

"""
Izbacivanje 15 KNN socijalnih featura ne menja performanse 
(ADE razlika ~1%, čak i na oštrim skretanjima gde bi socijalni kontekst trebalo da pomaže). 
Ovo potvrđuje feature importance nalaz da KNN nosi ~1% važnosti. 
Razlog je retka populacija ETH dataseta — pešaci su retko dovoljno blizu da utiču jedni na druge. 
Najbitniji atributi su relativne koordinate istorije kretanja.
"""