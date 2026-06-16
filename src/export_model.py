"""
Trening i eksport finalnog modela za deployment.
Model: XGBoost treniran SAMO na rel koordinatama (14 featura za window=8).
Čuva se kao .joblib za korišćenje u API-ju.
"""

import numpy as np
import os
import sys
import joblib
sys.path.insert(0, os.path.dirname(os.path.dirname(os.path.abspath(__file__))))

from xgboost import XGBRegressor
from src.config import WINDOW_SIZE, PRED_STEPS

DATA_DIR = 'data/processed'
MODEL_DIR = 'deployment'
os.makedirs(MODEL_DIR, exist_ok=True)

ps = PRED_STEPS
N_REL_TOTAL = 2 * (WINDOW_SIZE - 1)   # broj rel kolona
IDX_VX = WINDOW_SIZE - 2
IDX_VY = 2 * (WINDOW_SIZE - 1) - 1

print(f"Treniram finalni model (window={WINDOW_SIZE}, samo rel koordinate)...")

# Podaci
X_train = np.load(os.path.join(DATA_DIR, 'X_train.npy'))
y_train = np.load(os.path.join(DATA_DIR, 'y_train.npy'))
X_val   = np.load(os.path.join(DATA_DIR, 'X_val.npy'))
y_val   = np.load(os.path.join(DATA_DIR, 'y_val.npy'))
X_full  = np.vstack([X_train, X_val])
y_full  = np.vstack([y_train, y_val])

# Sample weights (kao u glavnom modelu)
self_vx = -X_full[:, IDX_VX]
self_vy = -X_full[:, IDX_VY]
deviation = np.sqrt((y_full[:, ps-1] - self_vx*ps)**2 + (y_full[:, 2*ps-1] - self_vy*ps)**2)
weights = np.where(deviation > 0.5, 3.0, 1.0)

# Samo rel koordinate
X_full_rel = X_full[:, :N_REL_TOTAL]

model = XGBRegressor(
    n_estimators=2000, learning_rate=0.005, max_depth=3,
    subsample=0.6, colsample_bytree=0.8, reg_alpha=2, reg_lambda=2,
    random_state=42, n_jobs=-1, verbosity=0
)
model.fit(X_full_rel, y_full, sample_weight=weights)

# Eksport modela + metapodataka
putanja = os.path.join(MODEL_DIR, 'model.joblib')
joblib.dump({
    'model': model,
    'window_size': WINDOW_SIZE,
    'pred_steps': PRED_STEPS,
    'n_features': N_REL_TOTAL,
    'feature_names': [f'rel_x_{i+1}' for i in range(WINDOW_SIZE-1)] +
                     [f'rel_y_{i+1}' for i in range(WINDOW_SIZE-1)],
}, putanja)

print(f"Model sačuvan: {putanja}")
print(f"Očekuje ulaz dimenzije: {N_REL_TOTAL} (rel_x_1..{WINDOW_SIZE-1}, rel_y_1..{WINDOW_SIZE-1})")
print(f"Vraća izlaz dimenzije: {2*PRED_STEPS} (delta_x_1..{PRED_STEPS}, delta_y_1..{PRED_STEPS})")