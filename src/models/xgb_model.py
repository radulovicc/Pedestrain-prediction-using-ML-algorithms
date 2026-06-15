"""
XGBoost - detaljan tuning za multistep predikciju.
Koristi VALIDATION set za podešavanje parametara.
Test set se koristi SAMO JEDNOM na kraju.
"""

import numpy as np
import pandas as pd
import os
import sys
import time
sys.path.insert(0, os.path.dirname(os.path.dirname(os.path.dirname(os.path.abspath(__file__)))))

from xgboost import XGBRegressor
from sklearn.linear_model import LinearRegression
from sklearn.metrics import r2_score
from src.config import WINDOW_SIZE, PRED_STEPS
from src.evaluation import izracunaj_metrike, plot_prediction, plot_trajektorija

DATA_DIR = 'data/processed'
OUTPUT_DIR = os.path.join('results', 'models', 'xgboost', f'window_{WINDOW_SIZE}')
os.makedirs(OUTPUT_DIR, exist_ok=True)

# ============================================================
# 1. UCITAVANJE
# ============================================================
print("=" * 70)
print(f" XGBOOST — Multistep tuning (N={WINDOW_SIZE}, PRED_STEPS={PRED_STEPS})")
print(" (validation set se koristi za odabir parametara)")
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
# 2. TUNING: learning_rate + n_estimators
# ============================================================
print(f"\n{'='*70}")
print(" 2. TUNING: learning_rate + n_estimators (na VALIDATION setu)")
print(f"{'='*70}")

#lr i n_estimators su inverzno povezani, mali learning rate znaci da svako stablo doprinosi malo, pa nam treba vise stabala da nadoknadimo taj nedostatak
#Zato se inverzno menjaju, XGBoost radi sekvencijalno i svako stablo ispravlja gresku prethodnog, za razliku od RF-a gde sva stabla rade zasebno paralelno
configs = [
    (0.3, 100),     # default
    (0.1, 300),
    (0.05, 600),
    (0.03, 1000),
    (0.01, 1500),
    (0.005, 2000),
]

print(f"\n{'LR':>8} {'n_est':>8} {'Train ADE':>10} {'Val ADE':>10} {'R² val':>10} {'Vreme':>8}")
print("-" * 58)

best_ade_lr = float('inf')
best_lr = None
best_n = None

for lr, nest in configs:
    t0 = time.time()
    model = XGBRegressor(
        n_estimators=nest,
        learning_rate=lr,
        max_depth=6,
        random_state=42,
        n_jobs=-1,
        verbosity=0     #Ovo kontrolise koliko ce informacija da mi ispisuje XGBoost. 0 = tiho
    )
    model.fit(X_train, y_train)
    y_pred_train = model.predict(X_train)
    y_pred_val = model.predict(X_val)
    t = time.time() - t0

    train_ade = izracunaj_ade(y_train, y_pred_train)
    val_ade = izracunaj_ade(y_val, y_pred_val)
    r2 = r2_score(y_val, y_pred_val)

    oznaka = ' <<<' if val_ade < best_ade_lr else ''
    if val_ade < best_ade_lr:
        best_ade_lr = val_ade
        best_lr = lr
        best_n = nest

    print(f"{lr:>8} {nest:>8} {train_ade:>10.6f} {val_ade:>10.6f} {r2:>10.4f} {t:>7.1f}s{oznaka}")

print(f"\n >> Najbolji: LR={best_lr}, n_estimators={best_n} (Val ADE: {best_ade_lr:.6f})")

# ============================================================
# 3. TUNING: max_depth
# ============================================================
print(f"\n{'='*70}")
print(" 3. TUNING: max_depth (na VALIDATION setu)")
print(f"{'='*70}")

max_depths = [3, 5, 7]
print(f"\n{'max_depth':>12} {'Train ADE':>10} {'Val ADE':>10} {'R² val':>10} {'Vreme':>8}")
print("-" * 55)

best_ade_md = float('inf')
best_md = None

for md in max_depths:
    t0 = time.time()
    model = XGBRegressor(
        n_estimators=best_n,
        learning_rate=best_lr,
        max_depth=md,
        random_state=42,
        n_jobs=-1,
        verbosity=0
    )
    model.fit(X_train, y_train)
    y_pred_val = model.predict(X_val)
    y_pred_train = model.predict(X_train)
    t = time.time() - t0

    train_ade = izracunaj_ade(y_train, y_pred_train)
    val_ade = izracunaj_ade(y_val, y_pred_val)
    r2 = r2_score(y_val, y_pred_val)

    oznaka = ' <<<' if val_ade < best_ade_md else ''
    if val_ade < best_ade_md:
        best_ade_md = val_ade
        best_md = md

    print(f"{md:>12} {train_ade:>10.6f} {val_ade:>10.6f} {r2:>10.4f} {t:>7.1f}s{oznaka}")

print(f"\n >> Najbolji max_depth: {best_md} (Val ADE: {best_ade_md:.6f})")

# ============================================================
# 4. TUNING: subsample + colsample_bytree - grid search
# ============================================================
print(f"\n{'='*70}")
print(" 4. TUNING: subsample + colsample_bytree (na VALIDATION setu)")
print(f"{'='*70}")

#Ove tehnike sprecavaju overfitting
subsamples = [0.6, 0.8, 1.0]    #EKVIVALENT bootstraping-u u RF. Znaci uzima nasumicno neki procenat od svih primera, znaci da ne vidi svako stablo iste primere
col_samples = [0.6, 0.8, 1.0]   #EKVIVALENT max_features-u u RF. Znaci uzima nasumicno neki procenat od sbih atributa. Ne vide sva stabla iste atribute. 
print(f"\n{'subsample':>12} {'colsample':>12} {'Train ADE':>10} {'Val ADE':>10} {'R² val':>10}")
print("-" * 58)

best_ade_sc = float('inf')
best_ss = None
best_cs = None

for ss in subsamples:
    for cs in col_samples:
        model = XGBRegressor(
            n_estimators=best_n,
            learning_rate=best_lr,
            max_depth=best_md,
            subsample=ss,
            colsample_bytree=cs,
            random_state=42,
            n_jobs=-1,
            verbosity=0
        )
        model.fit(X_train, y_train)
        y_pred_val = model.predict(X_val)
        y_pred_train = model.predict(X_train)

        train_ade = izracunaj_ade(y_train, y_pred_train)
        val_ade = izracunaj_ade(y_val, y_pred_val)
        r2 = r2_score(y_val, y_pred_val)

        oznaka = ' <<<' if val_ade < best_ade_sc else ''
        if val_ade < best_ade_sc:
            best_ade_sc = val_ade
            best_ss = ss
            best_cs = cs

        print(f"{ss:>12} {cs:>12} {train_ade:>10.6f} {val_ade:>10.6f} {r2:>10.4f}{oznaka}")

print(f"\n >> Najbolji: subsample={best_ss}, colsample={best_cs} (Val ADE: {best_ade_sc:.6f})")

# ============================================================
# 5. TUNING: L1/L2 regularizacija
# ============================================================
print(f"\n{'='*70}")
print(" 5. TUNING: L1/L2 regularizacija (na VALIDATION setu)")
print(f"{'='*70}")

#L1 kaznjava apsolutnu vrednost tezina U LISTOVIMA STABALA pa pojednostavljuje stabla izbacujuci slabe listove
#L2 kaznjava kvadrat tezina u listovima, sprecava da pojedini listovi imaju ekstremne vrednosti
reg_configs = [
    (0, 1),      # samo L2 default
    (0.1, 1),    # blaga L1
    (1, 1),      # jaka L1 + L2
    (0, 2),      # jaca L2
    (0.5, 1.5),  # balans
    (2, 2),      # oba jaka
]

print(f"\n{'reg_alpha':>12} {'reg_lambda':>12} {'Train ADE':>10} {'Val ADE':>10} {'R² val':>10}")
print("-" * 58)

best_ade_reg = float('inf')
best_a = None
best_l = None

for ra, rl in reg_configs:
    model = XGBRegressor(
        n_estimators=best_n,
        learning_rate=best_lr,
        max_depth=best_md,
        subsample=best_ss,
        colsample_bytree=best_cs,
        reg_alpha=ra,
        reg_lambda=rl,
        random_state=42,
        n_jobs=-1,
        verbosity=0
    )
    model.fit(X_train, y_train)
    y_pred_val = model.predict(X_val)
    y_pred_train = model.predict(X_train)

    train_ade = izracunaj_ade(y_train, y_pred_train)
    val_ade = izracunaj_ade(y_val, y_pred_val)
    r2 = r2_score(y_val, y_pred_val)

    oznaka = ' <<<' if val_ade < best_ade_reg else ''
    if val_ade < best_ade_reg:
        best_ade_reg = val_ade
        best_a = ra
        best_l = rl

    print(f"{ra:>12} {rl:>12} {train_ade:>10.6f} {val_ade:>10.6f} {r2:>10.4f}{oznaka}")

print(f"\n >> Najbolji: reg_alpha={best_a}, reg_lambda={best_l} (Val ADE: {best_ade_reg:.6f})")

# ============================================================
# 6. FINALNI MODEL — treniramo na train+val, evaluiramo na test
# ============================================================
print(f"\n{'='*70}")
print(" 6. FINALNI MODEL — XGBoost sa 'Sample Weights' (za nelinearne putanje)")
print(f"{'='*70}")

# Spajamo train + validation za finalni trening
X_train_full = np.vstack([X_train, X_val])
y_train_full = np.vstack([y_train, y_val])

print(f"\nX_train_full: {X_train_full.shape}, y_train_full: {y_train_full.shape}")
print(f"X_test:       {X_test.shape},       y_test:       {y_test.shape}")

# --- SAMPLE WEIGHTS (fokus na skretanja) ---
# Brzina pešaka u T = pomeraj od T-1 do T = -(rel na poslednjem koraku).
# rel_x_{N-1} je na indeksu WINDOW_SIZE-2; rel_y_{N-1} na 2*(WINDOW_SIZE-1)-1.
idx_rel_x_last = WINDOW_SIZE - 2
idx_rel_y_last = 2 * (WINDOW_SIZE - 1) - 1

self_vx = -X_train_full[:, idx_rel_x_last]
self_vy = -X_train_full[:, idx_rel_y_last]

# Ekstrapolacija: da je nastavio pravo, prešao bi vx*PRED_STEPS u X i vy*PRED_STEPS u Y
expected_dx = self_vx * PRED_STEPS
expected_dy = self_vy * PRED_STEPS

# Stvarni pomeraj na poslednjem koraku (delta_x_5 = indeks PRED_STEPS-1, delta_y_5 = 2*PRED_STEPS-1)
actual_dx = y_train_full[:, PRED_STEPS - 1]
actual_dy = y_train_full[:, 2 * PRED_STEPS - 1]

# Odstupanje od prave linije (nelinearnost)
deviation = np.sqrt((actual_dx - expected_dx)**2 + (actual_dy - expected_dy)**2)

# Težine: 3x za nelinearne situacije, 1x za ostale
weights = np.where(deviation > 0.5, 3.0, 1.0)
print(f"Broj nelinearnih situacija (skretanja/kočenja): {np.sum(weights == 3.0)} ({np.sum(weights==3.0)/len(weights)*100:.1f}%)")

print(f"\n Najbolji parametri (izabrani na validation setu):")
print(f"   learning_rate     = {best_lr}")
print(f"   n_estimators      = {best_n}")
print(f"   max_depth         = {best_md}")
print(f"   subsample         = {best_ss}")
print(f"   colsample_bytree  = {best_cs}")
print(f"   reg_alpha         = {best_a}")
print(f"   reg_lambda        = {best_l}")

xgb_final = XGBRegressor(
    n_estimators=best_n,
    learning_rate=best_lr,
    max_depth=best_md,
    subsample=best_ss,
    colsample_bytree=best_cs,
    reg_alpha=best_a,
    reg_lambda=best_l,
    random_state=42,
    n_jobs=-1,
    verbosity=0
)

# Fitujemo model SA TEŽINAMA
xgb_final.fit(X_train_full, y_train_full, sample_weight=weights)

y_train_pred = xgb_final.predict(X_train_full)
y_test_pred = xgb_final.predict(X_test)

print(f"\n Metrike na TEST setu (samo jednom, finalno):")
train_metrics = izracunaj_metrike(y_train_full, y_train_pred, 'TRAIN (XGB final)', PRED_STEPS)
test_metrics  = izracunaj_metrike(y_test, y_test_pred, 'TEST  (XGB final)', PRED_STEPS)

r2_test = r2_score(y_test, y_test_pred)
print(f"\n  R² test: {r2_test:.4f}")

# ============================================================
# 7. FEATURE IMPORTANCE
# ============================================================
print(f"\n{'='*70}")
print(" 7. FEATURE IMPORTANCE")
print(f"{'='*70}")

# Feature imena odgovaraju izlazu iz prepare_dataset.py:
#   (WINDOW_SIZE-1) rel_x + (WINDOW_SIZE-1) rel_y + 15 KNN (3 suseda x 5)
feature_names = (
    [f'rel_x_{i+1}' for i in range(WINDOW_SIZE - 1)] +
    [f'rel_y_{i+1}' for i in range(WINDOW_SIZE - 1)] +
    [f'nn_{k+1}_{feat}'
     for k in range(3)
     for feat in ['dx', 'dy', 'dvx', 'dvy', 'approach']]
)

importances = xgb_final.feature_importances_
indices = np.argsort(importances)[::-1]

print(f"\n Top 10 Najvažnijih Obeležja:")
print(f"  {'Rang':<6} {'Feature':<15} {'Vaznost':<10} {'Kumulativno':<12}")
print(f"  {'-'*6} {'-'*15} {'-'*10} {'-'*12}")
cumulative = 0
for rank, idx in enumerate(indices[:10], 1):
    cumulative += importances[idx]
    print(f"  {rank:<6} {feature_names[idx]:<15} {importances[idx]:<10.4f} {cumulative:<12.4f}")

# Ukupan doprinos KNN socijalnih featurea
knn_imena = [f'nn_{k+1}_{feat}'
             for k in range(3)
             for feat in ['dx', 'dy', 'dvx', 'dvy', 'approach']]
knn_pozicije = [feature_names.index(n) for n in knn_imena]
knn_total = np.sum(importances[knn_pozicije])
print(f"\n  Ukupna važnost KNN featurea: {knn_total:.4f} ({knn_total*100:.1f}%)")
print(f"  Ukupna važnost rel koordinata: {1-knn_total:.4f} ({(1-knn_total)*100:.1f}%)")

# ============================================================
# 8. UPOREDJIVANJE MODELA — XGB vs OLS (na test setu)
# ============================================================
print(f"\n{'='*70}")
print(" 8. UPOREDJIVANJE MODELA — XGBoost vs OLS (na test setu)")
print(f"{'='*70}")

# OLS u istom feature prostoru
ols = LinearRegression()
ols.fit(X_train_full, y_train_full)
y_pred_ols = ols.predict(X_test)
ade_ols = izracunaj_ade(y_test, y_pred_ols)
fde_ols = np.mean(np.sqrt((y_test[:, PRED_STEPS-1] - y_pred_ols[:, PRED_STEPS-1])**2 +
                          (y_test[:, 2*PRED_STEPS-1] - y_pred_ols[:, 2*PRED_STEPS-1])**2))
r2_ols = r2_score(y_test, y_pred_ols)

# XGB (naš finalni)
ade_xgb = izracunaj_ade(y_test, y_test_pred)
fde_xgb = np.mean(np.sqrt((y_test[:, PRED_STEPS-1] - y_test_pred[:, PRED_STEPS-1])**2 +
                          (y_test[:, 2*PRED_STEPS-1] - y_test_pred[:, 2*PRED_STEPS-1])**2))
r2_xgb = r2_score(y_test, y_test_pred)

print(f"\n{'Model':<35} {'ADE':>10} {'FDE':>10} {'R²':>10}")
print("-" * 67)
print(f"{'OLS':<35} {ade_ols:>10.6f} {fde_ols:>10.6f} {r2_ols:>10.4f}")
print(f"{'XGBoost (Weighted + KNN)':<35} {ade_xgb:>10.6f} {fde_xgb:>10.6f} {r2_xgb:>10.4f}")

imp_xgb_ols = (ade_ols - ade_xgb) / ade_ols * 100
print(f"\n XGB vs OLS: {imp_xgb_ols:+.2f}%")

# ADE po koraku
print(f"\n ADE po koraku:")
print(f"{'Korak':>8} {'XGBoost':>10} {'OLS':>10} {'Razlika':>12}")
print("-" * 42)
for s in range(PRED_STEPS):
    xgb_step = np.mean(np.sqrt(
        (y_test[:, s] - y_test_pred[:, s])**2 +
        (y_test[:, PRED_STEPS+s] - y_test_pred[:, PRED_STEPS+s])**2
    ))
    ols_step = np.mean(np.sqrt(
        (y_test[:, s] - y_pred_ols[:, s])**2 +
        (y_test[:, PRED_STEPS+s] - y_pred_ols[:, PRED_STEPS+s])**2
    ))
    print(f"{s+1:>8} {xgb_step:>10.6f} {ols_step:>10.6f} {xgb_step-ols_step:>+12.6f}")


# ============================================================
# 9. VIZUELIZACIJA
# ============================================================
print(f"\n{'='*70}")
print(" 9. VIZUELIZACIJA")
print(f"{'='*70}")

plot_prediction(X_test, y_test, y_test_pred, df_test,
                 'XGBoost (KNN)', OUTPUT_DIR, pred_steps=PRED_STEPS)
plot_trajektorija(X_test, y_test, y_test_pred, df_test,
                  'XGBoost (KNN)', OUTPUT_DIR, pred_steps=PRED_STEPS)

# ============================================================
# 10. CUVANJE
# ============================================================
np.save(os.path.join(OUTPUT_DIR, 'y_train_pred.npy'), y_train_pred)
np.save(os.path.join(OUTPUT_DIR, 'y_test_pred.npy'), y_test_pred)
metrics_df = pd.DataFrame([test_metrics])
metrics_df.to_csv(os.path.join(OUTPUT_DIR, 'metrics.csv'), index=False)
print(f"\n Rezultati: {OUTPUT_DIR}/")

"""
ZAKLJUCAK - XGBOost (window=8)

REZULTATI (test):
  OLS:                      ADE 0.1876   FDE 0.3304   R² 0.9767
  XGBoost (Weighted + KNN): ADE 0.1893   FDE 0.3295   R² 0.9756
  XGB vs OLS: -0.89% (XGBoost je NEZNATNO LOŠIJI)

IZABRANI PARAMETRI (tuning na val skupu):
  learning_rate=0.005, n_estimators=2000, max_depth=3,
  subsample=0.6, colsample_bytree=0.8, reg_alpha=2, reg_lambda=2

GLAVNI NALAZI:

1. XGBoost ne nadmašuje OLS.
    Najbolje tjunirani XGBoost, sa socijalnim featurima i sample weights,
    sedi tik ISPOD obične linearne regresije. Razlika (~0.002m, ispod 1%)
    je u granicama šuma. Zaključak: za predikciju kretanja na ~2s horizonta
    u prostoru relativnih koordinata, problem je suštinski linearan, pa
    nelinearni model nema prostora da donese prednost.

2. Tuning sam bira JEDNOSTAVNOST.
    Svaki korak tuninga gura model ka prostijem: izabran je najmanji
    max_depth (3), najjača regularizacija (alpha=2, lambda=2) i subsample
    0.6. To je signal da u podacima nema složenog nelinearnog obrasca koji
    bi dublja stabla mogla da iskoriste — model se sam "brani" od kompleksnosti.  

3. Socijalni (KNN) featurei: 1.1% ukupne važnosti.
    Svih top-10 featura su rel_x/rel_y (istorija kretanja samog pešaka).
    KNN socijalni featurei skoro ne doprinose — razlog je retka populacija
    ETH dataseta (malo pešaka po frejmu, retko dovoljno blizu da utiču).   

4. Sample weights nemaju merljiv efekat.
   Testirani su pragovi deviation > 0.5 (35% primera) i > 0.8 (17.7%).
   Promena praga ne menja ADE (0.1890 vs 0.1893). Model nema gde da nauči
   bolje na nelinearnim slučajevima — signala jednostavno nema dovoljno,
   nezavisno od toga koliko ih jako težinski naglasimo.
    
ZAKLJUCAK - XGBOost (window=5)

REZULTATI (test):
  OLS:                      ADE 0.2083   FDE 0.3632   R² 0.9746
  XGBoost (Weighted + KNN): ADE 0.2023   FDE 0.3493   R² 0.9763
  XGB vs OLS: +2.90% (XGBoost je BOLJI)

POREĐENJE SA WINDOW=8:
  Window 8: XGB 0.1893 vs OLS 0.1876  -> XGB lošiji (-0.89%)
  Window 5: XGB 0.2023 vs OLS 0.2083  -> XGB bolji  (+2.90%)

1. Kraća istorija OTKRIVA vrednost tree modela.
   Sa window=8 XGBoost ne nadmašuje OLS jer je linearni trend iz 7 prošlih
   pozicija toliko jasan da OLS savršeno radi. Sa window=5 (samo 4 prošle
   pozicije) linearna ekstrapolacija je manje pouzdana, pa XGBoost dobija
   prostor da uhvati nelinearni signal koji OLS ne može. Stepen linearnosti
   problema zavisi od dužine istorije.

2. Apsolutna tačnost je LOŠIJA nego window=8.
   Window 8: ADE 0.188 | Window 5: ADE 0.202. Više istorije = bolja predikcija.
   Dakle window 5 je bolji za demonstraciju razlike među modelima, ali window 8
   daje bolje apsolutne rezultate.

3. Tuning ponovo bira jednostavnost.
   Isti izbor kao window=8: max_depth=3, reg_alpha=2, reg_lambda=2, subsample=0.6.
   Konzistentno kroz oba window-a — model se brani od kompleksnosti.

ZAKLJUČAK ZA OVAJ PROBLEM:
    Problem je pretežno linearan, ali stepen linearnosti zavisi od dužine istorije.
    Sa dovoljno konteksta (window 8) linearni model je nenadmašiv; sa kraćim
    kontekstom (window 5) ostaje nelinearnog signala koji tree modeli iskoriste,
    posebno na dužim horizontima predikcije. Oba scenarija potvrđuju da je signal
    dominantno linearan — razlike među modelima ostaju ispod ~3% (par milimetara).
   """

