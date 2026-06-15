"""
Vizuelizacije za odbranu — svaki grafik dokazuje jednu tvrdnju.
Pokreni: uv run python src/vizuelizacija_za_odbranu.py
"""

import numpy as np
import matplotlib.pyplot as plt
import os
import sys

sys.path.insert(0, os.path.dirname(os.path.dirname(os.path.abspath(__file__))))
from src.config import WINDOW_SIZE, PRED_STEPS

# =============================================================================
# UČITAVANJE PODATAKA
# =============================================================================
print("Učitavanje podataka...")
y_test = np.load('data/processed/y_test.npy')
X_test = np.load('data/processed/X_test.npy')

ps = PRED_STEPS
N_REL = WINDOW_SIZE - 1
IDX_VX = WINDOW_SIZE - 2
IDX_VY = 2 * (WINDOW_SIZE - 1) - 1
W = f'window_{WINDOW_SIZE}'

# --- OLS se računa uživo (ne čuva se kao fajl) ---
from sklearn.linear_model import LinearRegression
X_train = np.load('data/processed/X_train.npy')
y_train = np.load('data/processed/y_train.npy')
X_val   = np.load('data/processed/X_val.npy')
y_val   = np.load('data/processed/y_val.npy')
X_full  = np.vstack([X_train, X_val])
y_full  = np.vstack([y_train, y_val])

_ols = LinearRegression()
_ols.fit(X_full, y_full)

modeli = {
    'OLS': _ols.predict(X_test),
    'RF':  np.load(f'results/models/random_forest/{W}/y_test_pred.npy'),
    'XGBoost': np.load(f'results/models/xgboost/{W}/y_test_pred.npy'),
}

save_dir = os.path.join('results', 'models', 'za_odbranu', W)
os.makedirs(save_dir, exist_ok=True)

def ade_po_uzorku(y_true, y_pred):
    err = np.sqrt((y_true[:,:ps]-y_pred[:,:ps])**2 + (y_true[:,ps:]-y_pred[:,ps:])**2)
    return np.mean(err, axis=1)

def fde_po_uzorku(y_true, y_pred):
    err = np.sqrt((y_true[:,:ps]-y_pred[:,:ps])**2 + (y_true[:,ps:]-y_pred[:,ps:])**2)
    return err[:, -1]

# =============================================================================
# GRAFIK 1: XGBoost vs OLS na oštrim skretanjima
# =============================================================================
print("1/6: XGBoost vs OLS po tipu kretanja...")

fde_ols = fde_po_uzorku(y_test, modeli['OLS'])

prava = fde_ols <= 0.3
blago = (fde_ols > 0.3) & (fde_ols <= 1.0)
ostro = fde_ols > 1.0

fig, axes = plt.subplots(1, 3, figsize=(15, 5))

for ax, (mask, naziv, boja) in zip(axes, [
    (prava, 'Prava linija\n(devijacija < 0.3m)', 'green'),
    (blago, 'Blago skretanje\n(0.3m < devijacija < 1m)', 'orange'),
    (ostro, 'Oštro skretanje\n(devijacija > 1m)', 'red'),
]):
    if mask.sum() == 0:
        ax.text(0.5, 0.5, 'Nema primera', ha='center', va='center', transform=ax.transAxes)
        ax.set_title(naziv)
        continue

    ols_ade = ade_po_uzorku(y_test[mask], modeli['OLS'][mask])
    xgb_ade = ade_po_uzorku(y_test[mask], modeli['XGBoost'][mask])

    ax.scatter(ols_ade, xgb_ade, alpha=0.5, c=boja, s=30)

    max_val = max(ols_ade.max(), xgb_ade.max())
    ax.plot([0, max_val], [0, max_val], 'k--', alpha=0.3, label='y = x (jednaki)')
    ax.fill_between([0, max_val], [0, max_val], 0, alpha=0.1, color='blue', label='XGBoost bolji')
    ax.fill_between([0, max_val], [0, max_val], max_val, alpha=0.1, color='gray', label='OLS bolji')

    xgb_bolji = (xgb_ade < ols_ade).mean() * 100
    ols_bolji = (ols_ade < xgb_ade).mean() * 100

    ax.set_xlabel('OLS ADE [m]')
    ax.set_ylabel('XGBoost ADE [m]')
    ax.set_title(f'{naziv}\nXGBoost bolji: {xgb_bolji:.0f}% | OLS bolji: {ols_bolji:.0f}%')
    ax.legend(fontsize=8)
    ax.grid(True, alpha=0.3)
    ax.axis('equal')

plt.suptitle('Poređenje XGBoost vs OLS po tipu kretanja', fontsize=14, fontweight='bold')
plt.tight_layout()
plt.savefig(f'{save_dir}/xgboost_vs_ols_po_tipu.png', dpi=120, bbox_inches='tight')
plt.close()

# =============================================================================
# GRAFIK 2: ADE po koraku za sve modele (sa standardnom devijacijom)
# =============================================================================
print("2/6: ADE po koraku...")

koraci = np.arange(1, ps + 1)
fig, ax = plt.subplots(figsize=(10, 6))

boje = {'OLS': '#2196F3', 'RF': '#4CAF50', 'XGBoost': '#FF5722'}
stilovi = {'OLS': 'o--', 'RF': 's--', 'XGBoost': '^--'}

for name, y_pred in modeli.items():
    ade_per_step = []
    ade_std_per_step = []
    for step in range(ps):
        err = np.sqrt((y_test[:,step]-y_pred[:,step])**2 + (y_test[:,ps+step]-y_pred[:,ps+step])**2)
        ade_per_step.append(np.mean(err))
        ade_std_per_step.append(np.std(err))

    ax.errorbar(koraci, ade_per_step, yerr=ade_std_per_step,
                fmt=stilovi[name], color=boje[name], capsize=5, capthick=2,
                markersize=8, linewidth=2, label=name)

ax.set_xlabel('Korak predikcije', fontsize=12)
ax.set_ylabel('ADE [m]', fontsize=12)
ax.set_title('Greška po koraku — svi modeli (sa standardnom devijacijom)', fontsize=14, fontweight='bold')
ax.set_xticks(koraci)
ax.legend(fontsize=11)
ax.grid(True, alpha=0.3)

plt.tight_layout()
plt.savefig(f'{save_dir}/ade_po_koraku.png', dpi=120, bbox_inches='tight')
plt.close()

# =============================================================================
# GRAFIK 3: Feature importance sa jasnom podelom (rel vs KNN)
# =============================================================================
print("3/6: Feature importance...")

from xgboost import XGBRegressor

# Sample weights: ista DEVIJACIJA kao u modelima (ekstrapolacija prave linije vs stvarno)
self_vx = -X_full[:, IDX_VX]
self_vy = -X_full[:, IDX_VY]
expected_dx = self_vx * ps
expected_dy = self_vy * ps
actual_dx = y_full[:, ps - 1]
actual_dy = y_full[:, 2*ps - 1]
deviation = np.sqrt((actual_dx - expected_dx)**2 + (actual_dy - expected_dy)**2)
sample_weights = np.where(deviation > 0.5, 3.0, 1.0)

model = XGBRegressor(
    n_estimators=2000, learning_rate=0.005,
    max_depth=3, subsample=0.6, colsample_bytree=0.8,
    reg_alpha=2, reg_lambda=2, random_state=42, verbosity=0
)
model.fit(X_full, y_full, sample_weight=sample_weights)

feature_names = (
    [f'rel_x_{i+1}' for i in range(N_REL)] +
    [f'rel_y_{i+1}' for i in range(N_REL)] +
    [f'nn_{k+1}_{feat}' for k in range(3) for feat in ['dx','dy','dvx','dvy','approach']]
)

importances = model.feature_importances_
n_rel_total = 2 * N_REL
rel_imp = sum(importances[:n_rel_total])
knn_imp = sum(importances[n_rel_total:])

sorted_idx = np.argsort(importances)[::-1]
sorted_names = [feature_names[i] for i in sorted_idx]
sorted_imps = [importances[i] for i in sorted_idx]

colors = ['#F44336' if 'nn_' in name else '#2196F3' for name in sorted_names]

fig, ax = plt.subplots(figsize=(12, 8))
bars = ax.barh(range(len(sorted_names)), sorted_imps, color=colors, edgecolor='white')
ax.set_yticks(range(len(sorted_names)))
ax.set_yticklabels(sorted_names, fontsize=9)
ax.invert_yaxis()
ax.set_xlabel('Važnost', fontsize=12)
ax.set_title('Feature Importance — XGBoost', fontsize=14, fontweight='bold')

for bar, imp in zip(bars, sorted_imps):
    ax.text(bar.get_width() + 0.002, bar.get_y() + bar.get_height()/2,
            f'{imp*100:.1f}%', va='center', fontsize=8)

from matplotlib.patches import Patch
legend_elements = [
    Patch(facecolor='#2196F3', label=f'Relativne koordinate ({rel_imp*100:.1f}%)'),
    Patch(facecolor='#F44336', label=f'KNN socijalni ({knn_imp*100:.1f}%)'),
]
ax.legend(handles=legend_elements, loc='lower right', fontsize=10)

first_knn_idx = next(i for i, name in enumerate(sorted_names) if 'nn_' in name)
max_imp = max(sorted_imps)
ax.annotate(f'Prvi KNN feature\n(#{first_knn_idx+1} od {len(sorted_names)})',
            xy=(sorted_imps[first_knn_idx], first_knn_idx),
            xytext=(max_imp * 0.75, first_knn_idx + 4),
            arrowprops=dict(arrowstyle='->', color='red', lw=2),
            fontsize=10, color='red', fontweight='bold',
            bbox=dict(boxstyle='round,pad=0.3', facecolor='white', edgecolor='red', alpha=0.8))

plt.tight_layout()
plt.savefig(f'{save_dir}/feature_importance.png', dpi=120, bbox_inches='tight')
plt.close()
print(f"   Relativne koordinate: {rel_imp*100:.1f}%")
print(f"   KNN socijalni: {knn_imp*100:.1f}%")

# =============================================================================
# GRAFIK 4: 5 najgorih slučajeva — svi modeli uporedo
# =============================================================================
print("4/6: 5 najgorih slučajeva...")

xgb_fde = fde_po_uzorku(y_test, modeli['XGBoost'])
najgori_idx = np.argsort(xgb_fde)[-5:][::-1]

stilovi_linija = {
    'Stvarno': {'color': 'black',   'linestyle': '-',  'linewidth': 3.0, 'marker': 'o', 'markersize': 6},
    'OLS':     {'color': '#2196F3', 'linestyle': '--', 'linewidth': 1.5, 'marker': 's', 'markersize': 4},
    'RF':      {'color': '#4CAF50', 'linestyle': '-.', 'linewidth': 1.5, 'marker': '^', 'markersize': 4},
    'XGBoost': {'color': '#FF5722', 'linestyle': ':',  'linewidth': 2.0, 'marker': 'D', 'markersize': 4},
}

fig, axes = plt.subplots(2, 3, figsize=(15, 10))
axes = axes.flatten()
axes[-1].remove()

for i, idx in enumerate(najgori_idx):
    ax = axes[i]

    past_x = np.append(X_test[idx, :N_REL], 0)
    past_y = np.append(X_test[idx, N_REL:2*N_REL], 0)

    true_fut_x = np.cumsum(np.insert(y_test[idx, :ps], 0, 0))
    true_fut_y = np.cumsum(np.insert(y_test[idx, ps:], 0, 0))

    ax.plot(past_x, past_y, color='green', linestyle='-', linewidth=2.5, alpha=0.6)
    ax.plot(0, 0, 'ks', markersize=10, label='Trenutna poz.')
    ax.plot(true_fut_x, true_fut_y, **stilovi_linija['Stvarno'], label='Stvarno')

    for name in ['OLS', 'RF', 'XGBoost']:
        pred = modeli[name][idx]
        pred_fut_x = np.cumsum(np.insert(pred[:ps], 0, 0))
        pred_fut_y = np.cumsum(np.insert(pred[ps:], 0, 0))
        ax.plot(pred_fut_x, pred_fut_y, **stilovi_linija[name], label=name)

    ax.set_title(f'Primer #{idx}', fontsize=11, fontweight='bold')
    ax.set_xlabel('x [m]', fontsize=9)
    ax.set_ylabel('y [m]', fontsize=9)
    ax.grid(True, alpha=0.3)
    ax.axis('equal')

    if i == 2:
        ax.legend(fontsize=8, loc='upper left', framealpha=0.8)

plt.suptitle('5 najgorih slučajeva — poređenje modela', fontsize=14, fontweight='bold')
plt.tight_layout()
plt.savefig(f'{save_dir}/5_najgorih_slucajeva.png', dpi=150, bbox_inches='tight')
plt.close()

# =============================================================================
# GRAFIK 5: ADE i FDE uporedo — bar chart
# =============================================================================
print("5/6: ADE/FDE bar chart...")

modeli_svi = {
    'Baseline': np.load(f'results/models/baseline/{W}/y_test_pred.npy'),
    'OLS': modeli['OLS'],
    'RF': modeli['RF'],
    'XGBoost': modeli['XGBoost'],
}

def izracunaj_metrike(y_true, y_pred):
    err = np.sqrt((y_true[:,:ps]-y_pred[:,:ps])**2 + (y_true[:,ps:]-y_pred[:,ps:])**2)
    return np.mean(err), np.mean(err[:,-1]), np.std(err), np.std(err[:,-1])

nazivi, ade_v, fde_v, ade_s, fde_s = [], [], [], [], []
for name, y_pred in modeli_svi.items():
    a, f, as_, fs = izracunaj_metrike(y_test, y_pred)
    nazivi.append(name); ade_v.append(a); fde_v.append(f); ade_s.append(as_); fde_s.append(fs)

x = np.arange(len(nazivi))
width = 0.35
fig, axes = plt.subplots(1, 2, figsize=(14, 6))
boje_bar = ['#B0BEC5', '#2196F3', '#4CAF50', '#FF5722']

ax = axes[0]
bars = ax.bar(x, ade_v, width, color=boje_bar[:len(nazivi)], edgecolor='white', linewidth=1.5, alpha=0.85)
for bar, val in zip(bars, ade_v):
    ax.text(bar.get_x() + bar.get_width()/2, bar.get_height() + 0.01,
            f'{val:.3f}m', ha='center', va='bottom', fontsize=10, fontweight='bold')
best = np.argmin(ade_v)
bars[best].set_edgecolor('#FF5722'); bars[best].set_linewidth(3)
ax.set_xticks(x); ax.set_xticklabels(nazivi, fontsize=11)
ax.set_ylabel('ADE [m]', fontsize=12)
ax.set_title('Average Displacement Error', fontsize=14, fontweight='bold')
ax.grid(axis='y', alpha=0.3)

ax = axes[1]
bars = ax.bar(x, fde_v, width, color=boje_bar[:len(nazivi)], edgecolor='white', linewidth=1.5, alpha=0.85)
for bar, val in zip(bars, fde_v):
    ax.text(bar.get_x() + bar.get_width()/2, bar.get_height() + 0.01,
            f'{val:.3f}m', ha='center', va='bottom', fontsize=10, fontweight='bold')
best = np.argmin(fde_v)
bars[best].set_edgecolor('#4CAF50'); bars[best].set_linewidth(3)
ax.set_xticks(x); ax.set_xticklabels(nazivi, fontsize=11)
ax.set_ylabel('FDE [m]', fontsize=12)
ax.set_title('Final Displacement Error', fontsize=14, fontweight='bold')
ax.grid(axis='y', alpha=0.3)

plt.suptitle('Poređenje modela — ADE i FDE (manje = bolje)', fontsize=15, fontweight='bold')
plt.tight_layout()
plt.savefig(f'{save_dir}/ade_fde_poredjenje.png', dpi=150, bbox_inches='tight')
plt.close()

# =============================================================================
# GRAFIK 6: Box plot distribucije grešaka
# =============================================================================
print("6/6: Distribucija grešaka (box plot)...")

fig, ax = plt.subplots(figsize=(12, 7))

podaci_ade, podaci_fde, labeli = [], [], []
for name, y_pred in modeli_svi.items():
    err = np.sqrt((y_test[:,:ps]-y_pred[:,:ps])**2 + (y_test[:,ps:]-y_pred[:,ps:])**2)
    podaci_ade.append(np.mean(err, axis=1))
    podaci_fde.append(err[:, -1])
    labeli.append(name)

pos_ade = np.arange(len(labeli)) * 3 - 0.4
pos_fde = np.arange(len(labeli)) * 3 + 0.4

bp1 = ax.boxplot(podaci_ade, positions=pos_ade, widths=0.6, patch_artist=True, showfliers=True,
                 flierprops=dict(marker='o', markersize=3, alpha=0.3),
                 boxprops=dict(facecolor='#2196F3', alpha=0.6), medianprops=dict(color='darkblue', linewidth=2))
bp2 = ax.boxplot(podaci_fde, positions=pos_fde, widths=0.6, patch_artist=True, showfliers=True,
                 flierprops=dict(marker='o', markersize=3, alpha=0.3),
                 boxprops=dict(facecolor='#FF5722', alpha=0.6), medianprops=dict(color='darkred', linewidth=2))

ax.set_xticks(np.arange(len(labeli)) * 3)
ax.set_xticklabels(labeli, fontsize=11)
ax.set_ylabel('Greška [m]', fontsize=12)
ax.set_title('Distribucija grešaka po modelima (box = 50% uzoraka)', fontsize=14, fontweight='bold')
ax.grid(axis='y', alpha=0.3)
ax.legend([bp1['boxes'][0], bp2['boxes'][0]], ['ADE', 'FDE'], loc='upper left', fontsize=11)
ax.axhline(0.3, color='gray', linestyle=':', alpha=0.5)

plt.tight_layout()
plt.savefig(f'{save_dir}/distribucija_gresaka_boxplot.png', dpi=150, bbox_inches='tight')
plt.close()

print(f"\n✅ Svi grafici sačuvani u '{save_dir}/'")