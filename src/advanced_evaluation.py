"""
Napredna evaluacija modela za predikciju putanja pešaka.

Analizira:
  1. Ekstremne greške (Max, 95th, 99th percentil)
  2. Greške po koracima (per-step percentili)
  3. Skretanja vs prava linija (podela test skupa)
  4. Vizuelizacija 5 najgorih slučajeva
  5. Greške po kategorijama putanje (prava/blago/oštro) — KLJUČNA analiza
"""

import numpy as np
import pandas as pd
import os
import sys
import matplotlib.pyplot as plt

sys.path.insert(0, os.path.dirname(os.path.dirname(os.path.dirname(os.path.abspath(__file__)))))

from config import WINDOW_SIZE, PRED_STEPS

DATA_DIR = 'data/processed'
OUTPUT_DIR = os.path.join('results', 'models', 'evaluation', f'window_{WINDOW_SIZE}')
os.makedirs(OUTPUT_DIR, exist_ok=True)

# Model na kome se rade detaljne analize (per-step, najgori, sažeti izveštaj)
MODEL_ZA_DETALJE = 'XGBoost'

# Indeksi brzine pešaka u T: -rel_x_{N-1} i -rel_y_{N-1}
IDX_VX = WINDOW_SIZE - 2
IDX_VY = 2 * (WINDOW_SIZE - 1) - 1
N_REL = WINDOW_SIZE - 1

# =========================================================================
# 1. Učitavanje podataka
# =========================================================================
print("=" * 70)
print(f" NAPREDNA EVALUACIJA — Analiza gresaka (N={WINDOW_SIZE})")
print("=" * 70)

X_test = np.load(os.path.join(DATA_DIR, 'X_test.npy'))
y_test = np.load(os.path.join(DATA_DIR, 'y_test.npy'))
df_test = pd.read_csv(os.path.join(DATA_DIR, 'test.csv'))

# Feature imena za referencu (rade za bilo koji window)
FEATURE_NAMES = (
    [f'rel_x_{i+1}' for i in range(N_REL)] +
    [f'rel_y_{i+1}' for i in range(N_REL)] +
    [f'nn_{k+1}_{feat}'
     for k in range(3)
     for feat in ['dx', 'dy', 'dvx', 'dvy', 'approach']]
)

# =========================================================================
# 2. Učitavanje predikcija svih modela
# =========================================================================
print("\n[1] Ucitavanje predikcija modela...")

modeli = {}

# --- OLS (računamo odmah, na train+val) ---
from sklearn.linear_model import LinearRegression
X_train = np.load(os.path.join(DATA_DIR, 'X_train.npy'))
y_train = np.load(os.path.join(DATA_DIR, 'y_train.npy'))
X_val   = np.load(os.path.join(DATA_DIR, 'X_val.npy'))
y_val   = np.load(os.path.join(DATA_DIR, 'y_val.npy'))
X_train_full = np.vstack([X_train, X_val])
y_train_full = np.vstack([y_train, y_val])

ols = LinearRegression()
ols.fit(X_train_full, y_train_full)
modeli['OLS'] = ols.predict(X_test)
print("  + OLS - izracunato")

# --- Učitavanje sačuvanih predikcija (iz window_{N} foldera) ---
def ucitaj_model(ime, putanja):
    try:
        modeli[ime] = np.load(putanja)
        print(f"  + {ime} - ucitano")
    except FileNotFoundError:
        print(f"  - {ime} - nije dostupan ({putanja})")

W = f'window_{WINDOW_SIZE}'
ucitaj_model('Baseline',      os.path.join('results/models/baseline',          W, 'y_test_pred.npy'))
ucitaj_model('Random Forest', os.path.join('results/models/random_forest',     W, 'y_test_pred.npy'))
ucitaj_model('XGBoost',       os.path.join('results/models/xgboost',           W, 'y_test_pred.npy'))

if not modeli:
    print("\n Nema nijednog modela za evaluaciju. Pokreni prvo modele.")
    sys.exit(1)

# Detaljne analize rade se na MODEL_ZA_DETALJE ako je dostupan, inače prvi u listi
if MODEL_ZA_DETALJE in modeli:
    NAJBOLJI = MODEL_ZA_DETALJE
else:
    NAJBOLJI = list(modeli.keys())[0]
    print(f"\n  ({MODEL_ZA_DETALJE} nije dostupan, koristim {NAJBOLJI})")

print(f"\n  >> Detaljna analiza radi se na modelu: {NAJBOLJI}")

# =========================================================================
# Pomoćne funkcije
# =========================================================================
def izracunaj_per_sample_errors(y_true, y_pred):
    """Vraća step_errors, ADE i FDE za svaki uzorak posebno."""
    ps = y_true.shape[1] // 2
    # Euklidska rastojanja po koracima: (N, PRED_STEPS)
    step_errors = np.sqrt(
        (y_true[:, :ps] - y_pred[:, :ps])**2 +
        (y_true[:, ps:] - y_pred[:, ps:])**2
    )
    ade_per_sample = np.mean(step_errors, axis=1)  # (N,)
    fde_per_sample = step_errors[:, -1]            # (N,)
    return step_errors, ade_per_sample, fde_per_sample

# =========================================================================
# 3. Ekstremne greške (Max i Percentili)
# =========================================================================
print("\n" + "=" * 70)
print(" 1. EKSTREMNE GRESKE (Max i Percentili)")
print("=" * 70)

for ime, y_pred in modeli.items():
    step_errors, ade_per_sample, fde_per_sample = izracunaj_per_sample_errors(y_test, y_pred)

    print(f"\n  [{ime}]")
    print(f"  {'-'*50}")
    print(f"  {'Metrika':<25} {'ADE':>10} {'FDE':>10}")
    print(f"  {'-'*50}")
    print(f"  {'Mean (prosek)':<25} {np.mean(ade_per_sample):>10.4f} {np.mean(fde_per_sample):>10.4f}")
    print(f"  {'Std':<25} {np.std(ade_per_sample):>10.4f} {np.std(fde_per_sample):>10.4f}")
    print(f"  {'Median':<25} {np.median(ade_per_sample):>10.4f} {np.median(fde_per_sample):>10.4f}")
    print(f"  {'Max':<25} {np.max(ade_per_sample):>10.4f} {np.max(fde_per_sample):>10.4f}")
    print(f"  {'95th Percentile':<25} {np.percentile(ade_per_sample, 95):>10.4f} {np.percentile(fde_per_sample, 95):>10.4f}")
    print(f"  {'99th Percentile':<25} {np.percentile(ade_per_sample, 99):>10.4f} {np.percentile(fde_per_sample, 99):>10.4f}")


# =========================================================================
# 4. Greške po koracima (per-step percentili)
# =========================================================================
print("\n" + "=" * 70)
print(" 2. GRESKE PO KORACIMA (Per-step percentili)")
print("=" * 70)

# Samo za model izabran za detalje
y_pred_best = modeli[NAJBOLJI]
step_errors, _, _ = izracunaj_per_sample_errors(y_test, y_pred_best)

print(f"\n  Model: {NAJBOLJI}")
print(f"  {'Korak':>8} {'Mean':>8} {'Median':>8} {'P90':>8} {'P95':>8} {'P99':>8} {'Max':>8}")
print(f"  {'-'*56}")

for s in range(PRED_STEPS):
    step = step_errors[:, s]
    print(f"  {s+1:>8} {np.mean(step):>8.4f} {np.median(step):>8.4f} "
          f"{np.percentile(step, 90):>8.4f} {np.percentile(step, 95):>8.4f} "
          f"{np.percentile(step, 99):>8.4f} {np.max(step):>8.4f}")
    

# =========================================================================
# 5. Analiza: Skretanja vs Prava linija
# =========================================================================
print("\n" + "=" * 70)
print(" 3. ANALIZA: Skretanja vs Prava linija")
print("=" * 70)

# Klasifikacija uzoraka po odstupanju od prave linije.
# Ideja: gde bi pešak ZAVRŠIO da je nastavio pravo (konstantnom brzinom iz T)
# vs gde je STVARNO završio na 5. koraku. Velika razlika = skretanje.
self_vx_test = -X_test[:, IDX_VX]   # brzina u T (x)
self_vy_test = -X_test[:, IDX_VY]   # brzina u T (y)

expected_dx = self_vx_test * PRED_STEPS
expected_dy = self_vy_test * PRED_STEPS

actual_dx = y_test[:, PRED_STEPS - 1]      # delta_x_5 (stvarni pomeraj na 5. koraku)
actual_dy = y_test[:, 2 * PRED_STEPS - 1]  # delta_y_5

deviation = np.sqrt((actual_dx - expected_dx)**2 + (actual_dy - expected_dy)**2)

# Tri kategorije po jačini odstupanja
pravo_mask = deviation <= 0.3
blago_mask = (deviation > 0.3) & (deviation <= 1.0)
ostro_mask = deviation > 1.0

print(f"\n  Klasifikacija test skupa ({len(deviation)} uzoraka):")
print(f"  {'Kategorija':<22} {'Broj':>8} {'Procenat':>10}")
print(f"  {'-'*42}")
print(f"  {'Prava linija (<30cm)':<22} {np.sum(pravo_mask):>8} {np.sum(pravo_mask)/len(deviation)*100:>9.1f}%")
print(f"  {'Blago skretanje':<22} {np.sum(blago_mask):>8} {np.sum(blago_mask)/len(deviation)*100:>9.1f}%")
print(f"  {'Ostro skretanje (>1m)':<22} {np.sum(ostro_mask):>8} {np.sum(ostro_mask)/len(deviation)*100:>9.1f}%")

# --- Greške po kategorijama za sve modele ---
print(f"\n  Greske po kategorijama (ADE / FDE):")
print(f"  {'Model':<18} {'Prava linija':>18} {'Blago skret.':>18} {'Ostro skret.':>18}")
print(f"  {'-'*74}")

for ime, y_pred in modeli.items():
    _, ade, fde = izracunaj_per_sample_errors(y_test, y_pred)

    def kat_greska(mask):
        if np.sum(mask) > 0:
            return np.mean(ade[mask]), np.mean(fde[mask])
        return 0.0, 0.0

    ade_pravo, fde_pravo = kat_greska(pravo_mask)
    ade_blago, fde_blago = kat_greska(blago_mask)
    ade_ostro, fde_ostro = kat_greska(ostro_mask)

    print(f"  {ime:<18} {ade_pravo:>7.4f}/{fde_pravo:<8.4f} "
          f"{ade_blago:>7.4f}/{fde_blago:<8.4f} "
          f"{ade_ostro:>7.4f}/{fde_ostro:<8.4f}")
    
# =========================================================================
# 6. Vizuelizacija: 5 najgorih slučajeva
# =========================================================================
print("\n" + "=" * 70)
print(" 4. VIZUELIZACIJA - 5 najgorih slucajeva")
print("=" * 70)

# 5 uzoraka sa najvećom FDE greškom (za model izabran za detalje)
_, _, fde_per_sample = izracunaj_per_sample_errors(y_test, y_pred_best)
najgori_indeksi = np.argsort(fde_per_sample)[-5:][::-1]

print(f"\n  Top 5 najgorih slucajeva (model: {NAJBOLJI}):")
print(f"  {'Rang':>5} {'Uzorak':>8} {'ADE':>8} {'FDE':>8}")
print(f"  {'-'*35}")
for rank, idx in enumerate(najgori_indeksi, 1):
    _, ade_i, fde_i = izracunaj_per_sample_errors(y_test[idx:idx+1], y_pred_best[idx:idx+1])
    print(f"  {rank:>5} {idx:>8} {ade_i[0]:>8.4f} {fde_i[0]:>8.4f}")

# Crtanje 5 najgorih
fig, axes = plt.subplots(1, 5, figsize=(22, 4.5))
fig.suptitle(f'5 najgorih slucajeva — {NAJBOLJI}', fontsize=14, y=1.02)

for i, (ax, idx) in enumerate(zip(axes, najgori_indeksi)):
    # Istorija: rel_x_1..(N-1) su pozicije relativne u odnosu na T(0,0)
    hist_x = X_test[idx, :N_REL]
    hist_y = X_test[idx, N_REL:2*N_REL]
    # Dodajemo T = (0,0) na kraj da se istorija spoji sa budućnošću
    full_hist_x = np.append(hist_x, 0)
    full_hist_y = np.append(hist_y, 0)

    # Stvarna budućnost: kumulativ delta (počinje od 0 = T)
    true_x = np.cumsum(np.insert(y_test[idx, :PRED_STEPS], 0, 0))
    true_y = np.cumsum(np.insert(y_test[idx, PRED_STEPS:], 0, 0))

    # Predikcija: isto
    pred_x = np.cumsum(np.insert(y_pred_best[idx, :PRED_STEPS], 0, 0))
    pred_y = np.cumsum(np.insert(y_pred_best[idx, PRED_STEPS:], 0, 0))

    ax.plot(full_hist_x, full_hist_y, 'b-', linewidth=2, alpha=0.7, label='Istorija')
    ax.plot(true_x, true_y, 'g-', linewidth=2.5, marker='o', label='Stvarno')
    ax.plot(pred_x, pred_y, 'r--', linewidth=2.5, marker='x', label='Predikcija')
    ax.plot(0, 0, 'ko', markersize=6)  # trenutna pozicija T

    _, ade_val, fde_val = izracunaj_per_sample_errors(
        y_test[idx:idx+1], y_pred_best[idx:idx+1]
    )
    ax.set_title(f'#{i+1} (ADE={ade_val[0]:.3f}, FDE={fde_val[0]:.3f})', fontsize=10)
    ax.set_xlabel('x [m]')
    ax.set_ylabel('y [m]')
    ax.grid(True, alpha=0.3)
    ax.set_aspect('equal')

    # Automatski zoom
    sve_x = np.concatenate([full_hist_x, true_x, pred_x])
    sve_y = np.concatenate([full_hist_y, true_y, pred_y])
    margin = 0.5
    x_min, x_max = np.min(sve_x), np.max(sve_x)
    y_min, y_max = np.min(sve_y), np.max(sve_y)
    ax.set_xlim(x_min - margin, x_max + margin)
    ax.set_ylim(y_min - margin, y_max + margin)

    if i == 0:
        ax.legend(fontsize=8)

plt.tight_layout()
plt.savefig(os.path.join(OUTPUT_DIR, 'najgorih_5_slucajeva.png'), dpi=150, bbox_inches='tight')
plt.close()
print(f"  -> Grafikon sacuvan: {OUTPUT_DIR}/najgorih_5_slucajeva.png")

# =========================================================================
# 7. Histogram distribucije grešaka
# =========================================================================
print("\n" + "=" * 70)
print(" 5. HISTOGRAM - Distribucija ADE i FDE gresaka")
print("=" * 70)

fig, axes = plt.subplots(1, 2, figsize=(12, 5))

for ime, y_pred in modeli.items():
    _, ade, fde = izracunaj_per_sample_errors(y_test, y_pred)
    axes[0].hist(ade, bins=50, alpha=0.5, label=ime)
    axes[1].hist(fde, bins=50, alpha=0.5, label=ime)

axes[0].set_xlabel('ADE [m]')
axes[0].set_ylabel('Broj uzoraka')
axes[0].set_title('Distribucija ADE gresaka')
axes[0].legend(fontsize=8)
axes[0].grid(True, alpha=0.3)

axes[1].set_xlabel('FDE [m]')
axes[1].set_ylabel('Broj uzoraka')
axes[1].set_title('Distribucija FDE gresaka')
axes[1].legend(fontsize=8)
axes[1].grid(True, alpha=0.3)

plt.tight_layout()
plt.savefig(os.path.join(OUTPUT_DIR, 'distribucija_gresaka.png'), dpi=150, bbox_inches='tight')
plt.close()
print(f"  -> Grafikon sacuvan: {OUTPUT_DIR}/distribucija_gresaka.png")

# =========================================================================
# 8. Sažeti izveštaj
# =========================================================================
print("\n" + "=" * 70)
print(" 6. SAZETI IZVESTAJ")
print("=" * 70)

# Metrike za model izabran za detalje
_, ade_best, fde_best = izracunaj_per_sample_errors(y_test, y_pred_best)

print(f"""
  Test skup: {len(y_test)} uzoraka
  Model za detalje: {NAJBOLJI}

  ADE (prosek svih koraka):  {np.mean(ade_best):.4f} m
  FDE (samo 5. korak):       {np.mean(fde_best):.4f} m
  95% uzoraka ima FDE <      {np.percentile(fde_best, 95):.4f} m
  99% uzoraka ima FDE <      {np.percentile(fde_best, 99):.4f} m
  Najgori slucaj (max FDE):  {np.max(fde_best):.4f} m

  Klasifikacija putanja:
    Prava linija:  {np.sum(pravo_mask):>4} ({np.sum(pravo_mask)/len(deviation)*100:>4.1f}%) — ADE: {np.mean(ade_best[pravo_mask]):.4f} m
    Blago skret.:  {np.sum(blago_mask):>4} ({np.sum(blago_mask)/len(deviation)*100:>4.1f}%) — ADE: {np.mean(ade_best[blago_mask]):.4f} m
    Ostro skret.:  {np.sum(ostro_mask):>4} ({np.sum(ostro_mask)/len(deviation)*100:>4.1f}%) — ADE: {np.mean(ade_best[ostro_mask]):.4f} m
""")

# =========================================================================
# 9. Čuvanje rezultata
# =========================================================================
results_rows = []
for ime, y_pred in modeli.items():
    step_errors, ade, fde = izracunaj_per_sample_errors(y_test, y_pred)
    row = {
        'model': ime,
        'ade_mean': np.mean(ade),
        'ade_std': np.std(ade),
        'ade_median': np.median(ade),
        'ade_p95': np.percentile(ade, 95),
        'ade_p99': np.percentile(ade, 99),
        'ade_max': np.max(ade),
        'fde_mean': np.mean(fde),
        'fde_std': np.std(fde),
        'fde_median': np.median(fde),
        'fde_p95': np.percentile(fde, 95),
        'fde_p99': np.percentile(fde, 99),
        'fde_max': np.max(fde),
    }
    # Per-step metrike
    for s in range(PRED_STEPS):
        row[f'step_{s+1}_mean'] = np.mean(step_errors[:, s])
        row[f'step_{s+1}_p95'] = np.percentile(step_errors[:, s], 95)
    # Greške po kategorijama (za ovaj model)
    row['ade_pravo'] = np.mean(ade[pravo_mask]) if np.sum(pravo_mask) > 0 else 0.0
    row['ade_blago'] = np.mean(ade[blago_mask]) if np.sum(blago_mask) > 0 else 0.0
    row['ade_ostro'] = np.mean(ade[ostro_mask]) if np.sum(ostro_mask) > 0 else 0.0
    results_rows.append(row)

results_df = pd.DataFrame(results_rows)
results_df.to_csv(os.path.join(OUTPUT_DIR, 'napredne_metrike.csv'), index=False)
print(f"  -> Metrike sacuvane: {OUTPUT_DIR}/napredne_metrike.csv")
print(f"\n{'='*70}")
print(" KRAJ NAPREDNE EVALUACIJE")
print("=" * 70)

"""
ZAKLJUČAK — Napredna evaluacija (window=8, 622 test uzorka)

FINALNO POREĐENJE (ADE / FDE):
  Baseline:       0.8922 / 1.7374   <- naivni (ponovi poslednji pomeraj)
  OLS:            0.1876 / 0.3304   <- najbolji ADE
  XGBoost:        0.1890 / 0.3275   <- najbolji FDE
  Random Forest:  0.1939 / 0.3297

GLAVNI NALAZI:

1. Svi učeni modeli ubedljivo imaju bolji rezultat od baseline-a (~4.7x bolji ADE).
   Baseline ADE 0.89 vs OLS/XGB/RF ~0.19. Dakle modeli zaista uče korisnu
   strukturu kretanja, a ne samo produžavaju poslednji pomeraj. Ovo je važno
   pokazati — dokazuje da problem nije trivijalan i da modeli imaju vrednost.

2. OLS, XGBoost i RF se prakticno ni ne razlikuju mnogo.
   Sva tri u opsegu ADE 0.188-0.194 (razlika ispod 3%, par milimetara).
   Potvrda glavne teze: problem je suštinski linearan, pa složeniji modeli
   ne donose prednost u proseku.

3. Cela greška dolazi iz oštrih skretanja.
   Podela test skupa po tipu putanje:
     Prava linija (55.5%):  ADE ~0.10-0.11   <- svi modeli odlični
     Blago skretanje (38.7%): ADE ~0.25-0.26
     Oštro skretanje (5.8%):  ADE ~0.50-0.60  <- 5x veća greška!
   Samo 5.8% primera (oštra skretanja) nosi najveći deo ukupne greške.
   Na pravoj liniji su svi modeli skoro identični jer je tamo problem
   savršeno linearan.

4. Gde se modeli razlikuju — na skretanjima.
   Prava linija:  OLS najbolji (0.0949) — problem je tu linearan, linearni model pobeđuje.
   Oštro skretanje:  XGB/RF bolji (0.505/0.509) vs OLS (0.599) — ~16% bolji.
   Dakle tree modeli opravdavaju postojanje samo na nelinearnim slučajevima,
   ali pošto ih je malo (5.8%), prednost se ponistava u ukupnom proseku.

5. Distribucija grešaka:
   XGBoost: median FDE 0.23m, ali max 2.19m. 95% uzoraka ima FDE < 0.97m,
   99% < 1.35m. Većina predikcija je odlična; mali rep teških slučajeva
   (nagla, nepredvidiva skretanja) diže prosek.

6. Greška raste sa horizontom (akumulacija).
   XGBoost mean po koraku: 0.064 → 0.119 → 0.181 → 0.253 → 0.328.
   P99 raste još brže (0.27 → 2.19 max), tj. rep se širi sa svakim korakom.
   Što dalje predviđamo, to nesigurnije — i to neravnomerno (oštra skretanja
   postaju sve nepredvidivija).

ZAKLJUČAK:
Za predikciju kretanja pešaka na ~2s horizonta, linearni i nelinearni modeli
daju gotovo identičan rezultat (~0.19m ADE) jer 94% kretanja je pravolinijsko
ili blago zakrivljeno — dakle linearno. Tree modeli (XGB, RF) pokazuju malu
prednost (~16%) samo na oštrim skretanjima (5.8% primera), ali ta prednost se
gubi u proseku. Glavni izvor greške svih modela je ona mala grupa naglih,
nepredvidivih skretanja koja se ne mogu predvideti iz same istorije kretanja —
za njih bi bili potrebni bogatiji modeli (Social LSTM/Transformer) i više
podataka o retkim manevrima.
"""