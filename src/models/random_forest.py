"""
Random Forest - detaljan tuning za multistep predikciju.
Koristi VALIDATION set za podešavanje parametara (kao XGBoost).
Test set se koristi SAMO JEDNOM na kraju.
Feature set: 29 kolona (14 rel + 15 KNN socijalnih).
"""

import numpy as np
import pandas as pd
import os
import sys
import time

sys.path.insert(0, os.path.dirname(os.path.dirname(os.path.dirname(os.path.abspath(__file__)))))

from sklearn.ensemble import RandomForestRegressor
from sklearn.linear_model import LinearRegression
from sklearn.metrics import r2_score
from src.config import WINDOW_SIZE, PRED_STEPS
from src.evaluation import izracunaj_metrike, plot_prediction, plot_trajektorija

DATA_DIR = 'data/processed'
OUTPUT_DIR = os.path.join('results', 'models', 'random_forest', f'window_{WINDOW_SIZE}')
os.makedirs(OUTPUT_DIR, exist_ok=True)

# ============================================================
# 1. UCITAVANJE
# ============================================================
print("=" * 70)
print(f" RANDOM FOREST — Multistep tuning (N={WINDOW_SIZE}, PRED_STEPS={PRED_STEPS})")
print(" (validation set za odabir parametara)")
print("=" * 70)

X_train = np.load(os.path.join(DATA_DIR, 'X_train.npy'))
y_train = np.load(os.path.join(DATA_DIR, 'y_train.npy'))
X_val   = np.load(os.path.join(DATA_DIR, 'X_val.npy'))
y_val   = np.load(os.path.join(DATA_DIR, 'y_val.npy'))
X_test  = np.load(os.path.join(DATA_DIR, 'X_test.npy'))
y_test  = np.load(os.path.join(DATA_DIR, 'y_test.npy'))
df_test = pd.read_csv(os.path.join(DATA_DIR, 'test.csv'))

print(f"\nX_train: {X_train.shape}, y_train: {y_train.shape}")
print(f"X_val:   {X_val.shape},   y_val:   {y_val.shape}")
print(f"X_test:  {X_test.shape},  y_test:  {y_test.shape}")
print(f"\nFeature set: {X_train.shape[1]} kolona ({2*(WINDOW_SIZE-1)} rel + 15 KNN)")

# ============================================================
# POMOCNA FUNKCIJA ZA ADE
# ============================================================
def izracunaj_ade(y_true, y_pred):
    ps = y_true.shape[1] // 2
    return np.mean(np.sqrt(
        (y_true[:,:ps] - y_pred[:,:ps])**2 +
        (y_true[:,ps:] - y_pred[:,ps:])**2
    ))

# ============================================================
# 2. TUNING: max_depth
# ============================================================
print(f"\n{'='*70}")
print(" 2. TUNING: max_depth (na VALIDATION setu)")
print(f"{'='*70}")

max_depths = [3,5,7,10, None]
print(f"\n{'max_depth':>12} {'Train ADE':>10} {'Val ADE':>10} {'R² val':>10} {'Gap':>10} {'Vreme':>8}")
print("-" * 65)

best_ade_md = float('inf')
best_md = None
"""
MAX_DEPTH - maksimalna dubina stabla - odredjuje koliko puta jednos tablo moze da se grana od korena do lista.
Svako grananje je jedno pitanje npr "da li je rel_x_6 > 0.8?". VEca dubina znaci da stablo moze da postavi vise 
uzastopnih pitanja i tako uhvati slozenije obrasce. Premala dubina daje stablo koje je pregrubo - ne moze da uhvati 
dovoljno detalja i dolazi do underfittinga. Prevelika dubina pravi stablo koje pamti svaki podatak na trening setu
pa dolazi do overfittinga - savrseno na train-u lose na testu. Trazimo kompromis izmedju ta dva

MIN_SAMPLES_LEAF - list je kranja tacka stabla hde se donosi predikcija. Ovaj parametar kaze "Ne pravi list ako u njemu
ima manje od N primera". Ako je 1, stablo sme da napravi list za svaki pojedinacni primer - ekstremni overfitting.
AKo je 30, svaki list mora da sadrzi bar 30 primera, pa je predikcija prosek tih 30 - glatkija, stabilnija, otpornija na sum
Kompromis: Veci broj = jednostavniji model, manje overfittinga, ali rizik od underfittinga.

N_ESTIMATORS - broj stabala u sumi - RandomForest nije jedno stablo, vec mnogo stabala cije se predikcije usrednjavaju. 
Ovaj parametar govori koliko stabala ima u toj sumi. Svako stablo se trenira na malo drugacijem uzorku podataka, pa svako gresi
na svoj nacin. - usrednjavanjem se te greske ponistavaju i dobijamo stabilniju predikciju
Kompromis: vise stabala = stabilniji i obicno bolji model, ali dobitak se zasiti. Razlika izmedju 50 i 300 stabala je velika,
izmedju 500 i 1000 obicno zanemarljiva. Vise stabala nikada ne skodi tacnoisti (ne overfitujemo dodavanjem stabala), samo trening
postaje skuplji. Zato biramo tacku gde val ADE prestane znacajno da pada.

MAX_FEATURES - broj feature-a koji se razmatra pri svakom grananju - Ovo je to random u RandomFoest. Pri svakom grananju,
stablo ne gleda svih mojih 29 feature-a, vec nasumican podskup. Time se stabla medjusobno razlikuju. AKo bi sva stabla videla
sve feature, sva bi se granala po istom najjacem feature-u i bila bi skoro identicna. Ovim ogranicenjem suma postaje raznovrsnija.
Kompromis: manji max_features = raznovrsnija, ali pojedinacno slabija stabla (vise nasumicnosti, manje korelacije medju stablima).
Veci = jaca pojedinacna stabla, ali medjusobno slicnija. 1.0 znaci da svako grananje vidi 1.0 (100%) feature-a - tada se gubi efekat 
raznovrsnosti i RF se ponasa blize obicnom Bagging-u

Prva tri hiperparametra znaci kontrolisu overfitting, a poslednji parametar kontrolise raznovrsnost stabala.

Radimo sekvencijalni tjuning parametara, a ne grid tjuning, jer je dosta jeftiniji.
"""
for md in max_depths:
    t0 = time.time()
    rf = RandomForestRegressor(
        n_estimators=300,
        max_depth=md,
        min_samples_leaf=5,
        random_state=42,                #koristimo kako bi nasi modeli mogli da se porede, za reproduktivnost, random_state radi bootstraping nad primerima, a ne na atributima kao max_faetures
        n_jobs=-1                       #govorimo programu da koristi sva dostupna jezgra za treniranje, kako bi se sto vise stabala treniralo paraleelno
    )
    rf.fit(X_train, y_train)
    y_pred_train = rf.predict(X_train)
    y_pred_val = rf.predict(X_val)
    t = time.time() - t0

    train_ade = izracunaj_ade(y_train, y_pred_train)
    val_ade = izracunaj_ade(y_val, y_pred_val)
    r2 = r2_score(y_val, y_pred_val)
    gap = train_ade - val_ade                           #Ovaj gap nam govori koliko model overfituje. Sto je gap negativniji, to model vise overfituje

    oznaka = ' <<<' if val_ade < best_ade_md else ''
    if val_ade < best_ade_md:
        best_ade_md = val_ade
        best_md = md
    
    md_str = str(md) if md is not None else "∞"
    print(f"{md_str:>12} {train_ade:>10.6f} {val_ade:>10.6f} {r2:>10.4f} {gap:>+10.6f} {t:>7.1f}s{oznaka}")

print(f"\n >> Najbolji max_depth: {best_md} (Val ADE: {best_ade_md:.6f})")

# ============================================================
# 3. TUNING: min_samples_leaf
# ============================================================
print(f"\n{'='*70}")
print(" 3. TUNING: min_samples_leaf (na VALIDATION setu)")
print(f"{'='*70}")

min_samples = [1, 3, 5, 10, 15, 20, 30]
print(f"\n{'min_leaf':>12} {'Train ADE':>10} {'Val ADE':>10} {'R² val':>10} {'Gap':>10}")
print("-" * 55)

best_ade_leaf = float('inf')
best_leaf = None

for ms in min_samples:
    rf = RandomForestRegressor(
        n_estimators=300,
        max_depth=best_md,
        min_samples_leaf=ms,
        random_state=42,
        n_jobs=-1
    )

    rf.fit(X_train, y_train)
    y_pred_train = rf.predict(X_train)
    y_pred_val = rf.predict(X_val)

    train_ade = izracunaj_ade(y_train, y_pred_train)
    val_ade = izracunaj_ade(y_val, y_pred_val)
    r2 = r2_score(y_val, y_pred_val)
    gap = train_ade - val_ade

    oznaka = ' <<<' if val_ade < best_ade_leaf else ''
    if val_ade < best_ade_leaf:
        best_ade_leaf = val_ade
        best_leaf = ms
    
    print(f"{ms:>12} {train_ade:>10.6f} {val_ade:>10.6f} {r2:>10.4f} {gap:>+10.6f}{oznaka}")

print(f"\n >> Najbolji min_samples_leaf: {best_leaf} (Val ADE: {best_ade_leaf:.6f})")

# ============================================================
# 4. TUNING: n_estimators
# ============================================================
print(f"\n{'='*70}")
print(" 4. TUNING: n_estimators (na VALIDATION setu)")
print(f"{'='*70}")

n_estimators_list = [50, 100, 200, 300, 500, 1000]
print(f"\n{'n_estim':>10} {'Train ADE':>10} {'Val ADE':>10} {'R² val':>10} {'Vreme':>8}")
print("-" * 50)

best_ade_n = float('inf')
best_n = None

for n in n_estimators_list:
    t0 = time.time()
    rf = RandomForestRegressor(
        n_estimators=n,
        max_depth=best_md,
        min_samples_leaf=best_leaf,
        random_state=42,
        n_jobs=-1
    )
    rf.fit(X_train, y_train)
    y_pred_val = rf.predict(X_val)
    y_pred_train = rf.predict(X_train)
    t = time.time() - t0

    train_ade = izracunaj_ade(y_train, y_pred_train)
    val_ade = izracunaj_ade(y_val, y_pred_val)
    r2 = r2_score(y_val, y_pred_val)

    oznaka = ' <<<' if val_ade < best_ade_n else ''
    if val_ade < best_ade_n:
        best_ade_n = val_ade
        best_n = n

    print(f"{n:>10} {train_ade:>10.6f} {val_ade:>10.6f} {r2:>10.4f} {t:>7.1f}s{oznaka}")

print(f"\n >> Najbolji n_estimators: {best_n} (Val ADE: {best_ade_n:.6f})")

# ============================================================
# 5. TUNING: max_features
# ============================================================
print(f"\n{'='*70}")
print(" 5. TUNING: max_features (na VALIDATION setu)")
print(f"{'='*70}")

max_feat_options = [0.3, 0.5, 0.7, 1.0]
print(f"\n{'max_feat':>12} {'Train ADE':>10} {'Val ADE':>10} {'R² val':>10}")
print("-" * 45)

best_ade_mf = float('inf')
best_mf = None

for mf in max_feat_options:
    rf = RandomForestRegressor(
        n_estimators=best_n,
        max_depth=best_md,
        min_samples_leaf=best_leaf,
        max_features=mf,
        random_state=42,
        n_jobs=-1
    )
    rf.fit(X_train, y_train)
    y_pred_val = rf.predict(X_val)
    y_pred_train = rf.predict(X_train)

    train_ade = izracunaj_ade(y_train, y_pred_train)
    val_ade = izracunaj_ade(y_val, y_pred_val)
    r2 = r2_score(y_val, y_pred_val)

    oznaka = ' <<<' if val_ade < best_ade_mf else ''
    if val_ade < best_ade_mf:
        best_ade_mf = val_ade
        best_mf = mf

    print(f"{mf:>12} {train_ade:>10.6f} {val_ade:>10.6f} {r2:>10.4f}{oznaka}")

print(f"\n >> Najbolji max_features: {best_mf} (Val ADE: {best_ade_mf:.6f})")

print(f"\n Najbolji parametri (izabrani na validation setu):")
print(f"   max_depth        = {best_md}")
print(f"   min_samples_leaf = {best_leaf}")
print(f"   n_estimators     = {best_n}")
print(f"   max_features     = {best_mf}")

# ============================================================
# 6. FINALNI MODEL — treniramo na train+val, evaluiramo na test
# ============================================================
print(f"\n{'='*70}")
print(" 6. FINALNI MODEL — RF sa Sample Weights (nelinearne putanje)")
print(f"{'='*70}")

#Spajamo train+validation za finalni trening
X_train_full = np.vstack([X_train, X_val])
y_train_full = np.vstack([y_train, y_val])

print(f"\nX_train_full: {X_train_full.shape}, y_train_full: {y_train_full.shape}")
print(f"X_test:       {X_test.shape},       y_test:       {y_test.shape}")

# --- SAMPLE WEIGHTS (fokus na skretanja) ---
# Brzina pešaka u T = pomeraj od T-1 do T = -(rel na poslednjem koraku).
# rel_x_{N-1} je na indeksu WINDOW_SIZE-2; rel_y_{N-1} na 2*(WINDOW_SIZE-1)-1.

idx_rel_x_last = WINDOW_SIZE - 2
idx_rel_y_last = 2 * (WINDOW_SIZE-1) - 1

self_vx = -X_train_full[:, idx_rel_x_last]
self_vy = -X_train_full[:, idx_rel_y_last]

# Ekstrapolacija: da je nastavio pravo, prešao bi vx*PRED_STEPS u X i vy*PRED_STEPS u Y
expected_dx = self_vx * PRED_STEPS
expected_dy = self_vy * PRED_STEPS

#Stvarni pomeraj na poslednjem koraku (delta_x_5 = indeks 4, delta_y_5 = indeks 9)
actual_dx = y_train_full[:, PRED_STEPS-1]
actual_dy = y_train_full[:, 2*PRED_STEPS -1]

# Odstupanje od prave linije (nelinearnost)
deviation = np.sqrt((actual_dx - expected_dx)**2 + (actual_dy - expected_dy)**2)

# Težine: 3x za nelinearne situacije, 1x za ostale
weights = np.where(deviation > 0.5, 3.0, 1.0)
n_nonlinear = np.sum(weights == 3.0)
print(f"Broj nelinearnih situacija (skretanja/kočenja): {n_nonlinear} ({n_nonlinear/len(weights)*100:.1f}%)")

"""
Sample weights govori modelu da su mu nelinearni primeri vazniji. Ova tehnika uvecava gresku za primere sa nelinearnom
kretnjom, sto znaci da se model vise primorava da nauci njih dobro. Ovo je nacin na koji sam pokusao da odradim balansiranje
ovih "klasa" pesaka, medjutim, uzeci u obzir da ih svakako nema dovoljno, model verovatno nece uspeti da ih nauci kako bi trebalo
"""

rf_final = RandomForestRegressor(
    n_estimators=best_n,
    max_depth=best_md,
    min_samples_leaf=best_leaf,
    max_features=best_mf,
    random_state=42,
    n_jobs=-1
)

rf_final.fit(X_train_full, y_train_full, sample_weight=weights)

y_train_pred = rf_final.predict(X_train_full)
y_test_pred = rf_final.predict(X_test)

print(f"\n Metrike na TEST setu (samo jednom, finalno):")
train_metrics = izracunaj_metrike(y_train_full, y_train_pred, 'TRAIN (RF final)', PRED_STEPS)
test_metrics  = izracunaj_metrike(y_test, y_test_pred, 'TEST  (RF final)', PRED_STEPS)

r2_test = r2_score(y_test, y_test_pred)
print(f"\n  R² test: {r2_test:.4f}")

# ============================================================
# 7. FEATURE IMPORTANCE
# ============================================================
print(f"\n{'='*70}")
print(f" 7. FEATURE IMPORTANCE ({X_train.shape[1]} featurea)")
print(f"{'='*70}")

feature_names = (
    [f'rel_x_{i+1}' for i in range(WINDOW_SIZE-1)] + 
    [f'rel_y_{i+1}' for i in range(WINDOW_SIZE-1)] +
    [f'nn_{k+1}_{feat}' for k in range(3)
                        for feat in ['dx', 'dy', 'dvx', 'dvy', 'approach']]
    
)

importances = rf_final.feature_importances_
indices = np.argsort(importances)[::-1] #sortiramo ih, prikazemo im indekse preko argsort, ali sortiranje je radilo u rastucem redosledu, sada ga sa [::-1] sortiramo opadajuce

print(f"\n Top 10 Najvažnijih Obeležja:")
print(f"  {'Rang':<6} {'Feature':<15} {'Vaznost':<10} {'Kumulativno':<12}")
print(f"  {'-'*6} {'-'*15} {'-'*10} {'-'*12}")

cumulative = 0
for rank, idx in enumerate(indices[:10], 1): #1, u enumerate je samo broj od kog pocinje brojanje, default=0
    cumulative += importances[idx]
    print(f"  {rank:<6} {feature_names[idx]:<15} {importances[idx]:<10.4f} {cumulative:<12.4f}")

# Gde su KNN socijalni featurei u rangu?
print(f"\n  Pozicije KNN socijalnih featurea u rangu:")

knn_imena = [f'nn_{k+1}_{feat}' for k in range(3)
                                for feat in ['dx', 'dy', 'dvx', 'dvy', 'approach']]

for feat_name in knn_imena:
    pos = feature_names.index(feat_name)
    rank = np.where(indices == pos)[0][0] + 1
    print(f"    {feat_name:<15} → rang #{rank} (vaznost: {importances[pos]:.4f})")
    
# Ukupan doprinos KNN featurea
knn_pozicije = [feature_names.index(n) for n in knn_imena]
knn_total = np.sum(importances[knn_pozicije])
print(f"\n  Ukupna važnost KNN featurea: {knn_total:.4f} ({knn_total*100:.1f}%)")
print(f"  Ukupna važnost rel koordinata: {1-knn_total:.4f} ({(1-knn_total)*100:.1f}%)")

# ============================================================
# 8. POREDJENJE SA OLS (baseline u istom feature prostoru)
# ============================================================
print(f"\n{'='*70}")
print(" 8. POREDJENJE — RF vs OLS (na test setu)")
print(f"{'='*70}")

ols = LinearRegression()
ols.fit(X_train_full, y_train_full)
y_pred_ols = ols.predict(X_test)
ade_ols = izracunaj_ade(y_test, y_pred_ols)
fde_ols = np.mean(np.sqrt((y_test[:, PRED_STEPS-1] - y_pred_ols[:, PRED_STEPS-1])**2 +
                          (y_test[:, 2*PRED_STEPS-1] - y_pred_ols[:, 2*PRED_STEPS-1])**2))
r2_ols = r2_score(y_test, y_pred_ols)

ade_rf = izracunaj_ade(y_test, y_test_pred)
fde_rf = np.mean(np.sqrt((y_test[:, PRED_STEPS-1] - y_test_pred[:, PRED_STEPS-1])**2 +
                         (y_test[:, 2*PRED_STEPS-1] - y_test_pred[:, 2*PRED_STEPS-1])**2))
r2_rf = r2_score(y_test, y_test_pred)

print(f"\n{'Model':<35} {'ADE':>10} {'FDE':>10} {'R²':>10}")
print("-" * 67)
print(f"{'OLS':<35} {ade_ols:>10.6f} {fde_ols:>10.6f} {r2_ols:>10.4f}")
print(f"{'RF (tjunirani + weights)':<35} {ade_rf:>10.6f} {fde_rf:>10.6f} {r2_rf:>10.4f}")

imp_rf_ols = (ade_ols - ade_rf) / ade_ols * 100
print(f"\n RF vs OLS: {imp_rf_ols:+.2f}%")

# ADE po koraku
print(f"\n ADE po koraku:")
print(f"{'Korak':>8} {'RF':>10} {'OLS':>10} {'Razlika':>12}")
print("-" * 42)
for s in range(PRED_STEPS):
    rf_step = np.mean(np.sqrt(
        (y_test[:, s] - y_test_pred[:, s])**2 +
        (y_test[:, PRED_STEPS+s] - y_test_pred[:, PRED_STEPS+s])**2
    ))
    ols_step = np.mean(np.sqrt(
        (y_test[:, s] - y_pred_ols[:, s])**2 +
        (y_test[:, PRED_STEPS+s] - y_pred_ols[:, PRED_STEPS+s])**2
    ))
    print(f"{s+1:>8} {rf_step:>10.6f} {ols_step:>10.6f} {rf_step-ols_step:>+12.6f}")

# ============================================================
# 9. VIZUELIZACIJA
# ============================================================
print(f"\n{'='*70}")
print(" 9. VIZUELIZACIJA")
print(f"{'='*70}")

plot_prediction(X_test, y_test, y_test_pred, df_test,
                 'RF (tjunirani + sample weights)', OUTPUT_DIR, pred_steps=PRED_STEPS)
plot_trajektorija(X_test, y_test, y_test_pred, df_test,
                  'RF (tjunirani + sample weights)', OUTPUT_DIR, pred_steps=PRED_STEPS)

# ============================================================
# 10. CUVANJE
# ============================================================
np.save(os.path.join(OUTPUT_DIR, 'y_train_pred.npy'), y_train_pred)
np.save(os.path.join(OUTPUT_DIR, 'y_test_pred.npy'), y_test_pred)
metrics_df = pd.DataFrame([test_metrics])
metrics_df.to_csv(os.path.join(OUTPUT_DIR, 'metrics.csv'), index=False)
print(f"\n Rezultati: {OUTPUT_DIR}/")

"""
RF je bolji za WINDOW_SIZE=5 za razliku od linearne regresije koja je sa tim parametrom gora od WINDOW_SIZE=8
"""