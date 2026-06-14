
"""
Linearna regresija za multistep predikciju.
Poredi dva modela:
  1. OLS (obična linearna regresija)
  2. Ridge + Polynomial Features - poboljšani model
"""

import os
import sys
import numpy as np
import pandas as pd

sys.path.insert(0, os.path.dirname(os.path.dirname(os.path.dirname(os.path.abspath(__file__)))))

from src.config import WINDOW_SIZE, PRED_STEPS
from src.evaluation import izracunaj_metrike, plot_prediction, plot_trajektorija
from sklearn.linear_model import LinearRegression, Ridge, Lasso
from sklearn.preprocessing import StandardScaler, PolynomialFeatures


DATA_DIR = 'data/processed'
OUTPUT_DIR = os.path.join('results', 'models', 'linear_regression', f'window_{WINDOW_SIZE}')
os.makedirs(OUTPUT_DIR, exist_ok=True)

# =============================================================================
# 1. UČITAVANJE
# =============================================================================
print("=" * 60)
print(f" LINEARNA REGRESIJA — Multistep (N={WINDOW_SIZE}, PRED_STEPS={PRED_STEPS})")
print("=" * 60)

X_train = np.load(os.path.join(DATA_DIR, 'X_train.npy'))
y_train = np.load(os.path.join(DATA_DIR, 'y_train.npy'))
X_test  = np.load(os.path.join(DATA_DIR, 'X_test.npy'))
y_test  = np.load(os.path.join(DATA_DIR, 'y_test.npy'))
df_test = pd.read_csv(os.path.join(DATA_DIR, 'test.csv'))

print(f"X_train: {X_train.shape}, y_train: {y_train.shape}")
print(f"X_test:  {X_test.shape},  y_test:  {y_test.shape}")

# =============================================================================
# MODEL 1: OLS (obična linearna regresija)
# =============================================================================
print(f"\n{'='*60}")
print(" MODEL 1: OLS")
print(f"{'='*60}")

model_ols = LinearRegression()
model_ols.fit(X_train, y_train)
y_pred_ols = model_ols.predict(X_test)
y_pred_ols_train = model_ols.predict(X_train)

train_metrics_ols = izracunaj_metrike(y_train, y_pred_ols_train, 'TRAIN (OLS)', PRED_STEPS)
test_metrics_ols = izracunaj_metrike(y_test, y_pred_ols, 'TEST (OLS)', PRED_STEPS)

# =============================================================================
# MODEL 2: Ridge + Polynomial Features
# =============================================================================
print(f"\n{'='*60}")
print(" MODEL 2: Ridge + Polynomial Features")
print("=" * 60)

#StandardScaler je obavezan za Ridge jer je osetljiv na skalu
#Ako bismo imali dva atributa sa razlicitim skalama, da bi dva 
#atributa imala isti uticaj, koeficijent uz manji atribut mora biti mnogo veci da bi imao isti uticaj kao i atribut cija je skala znatno veca.
#Zato prvo moramo skalirati atribute kako bi njihoovi koeficijenti bili uporedivi
scaler = StandardScaler()                   #Radi skaliranje preko normalizacije, srednja vrednost i standardna devijacija 
X_train_sc = scaler.fit_transform(X_train)
X_test_sc = scaler.transform(X_test)        #Ne radimo fit zato sto bi to bilo curenje podataka. Ne smemo trenirati model na test podacima.

#Polynomial Features (degree=2, samo interakcije)
#Dodaje nelinearne kombinacije poput rel_x_7 * rel_y_7

poly = PolynomialFeatures(degree=2, interaction_only=True, include_bias=False)
X_train_poly = poly.fit_transform(X_train_sc)
X_test_poly = poly.transform(X_test_sc)

print(f"Features: {X_train.shape[1]} -> {X_train_poly.shape[1]} (+{X_train_poly.shape[1]-X_train.shape[1]} interakcija)")

#Objasnjenje ove tehnike: Ova tehnika nam omogucava da dobijamo i atribute poput rel_x_5 * rel_y_3, a ne samo linearne atribute. Ovo omogucava da model moze da posmatra vise atributa istovremeno
#U smislu da moze da poveze da su dva atributa velika i da za to dodeli neki weight. Linearni model jednostavno ne moze da uhvati interakciju izmedju feature-a
#Pokusavamo nekako da napravimo pametniji linearni model koji ce moci da uhvati neke interakcije izmedju atributa.
#Ovo nam ne treba kod tree modela jer oni mogu po svojoj prirodi da uoce neke interakcije zbog prirode grananja
#Mozda sam mogao sam da konstruisem neke atribute koji bi mi davali neki smisao, ali o tome da li je potrebno cu kasnije razmisliti. Mozda uvedem atribut ugla putanje

# Ridge sa alfa=10 (Kaznjava velike tezine i to njihove kvadrate. ZNaci mnogo vise kaznjava vece tezine nego manje, a i ne moze da dovede tezinu do nule kao Lasso)
model_ridge = Ridge(alpha=10.0, random_state=42)
model_ridge.fit(X_train_poly, y_train)
y_pred_ridge = model_ridge.predict(X_test_poly)
y_pred_ridge_train = model_ridge.predict(X_train_poly)

train_metrics_ridge = izracunaj_metrike(y_train, y_pred_ridge_train, 'TRAIN (Ridge+Poly)', PRED_STEPS)
test_metrics_ridge  = izracunaj_metrike(y_test, y_pred_ridge, 'TEST (Ridge+Poly)', PRED_STEPS)

# =============================================================================
# MODEL 3: Lasso + Polynomial Features (feature selection)
# =============================================================================
from sklearn.linear_model import Lasso

print(f"\n{'='*60}")
print(" MODEL 3: Lasso + Polynomial Features")
print("=" * 60)

# Lasso (L1) gura nebitne težine na TAČNO nulu -> automatski feature selection
model_lasso = Lasso(alpha=0.001, max_iter=10000, random_state=42)
model_lasso.fit(X_train_poly, y_train)
y_pred_lasso = model_lasso.predict(X_test_poly)
y_pred_lasso_train = model_lasso.predict(X_train_poly)

train_metrics_lasso = izracunaj_metrike(y_train, y_pred_lasso_train, 'TRAIN (Lasso+Poly)', PRED_STEPS)
test_metrics_lasso  = izracunaj_metrike(y_test, y_pred_lasso, 'TEST (Lasso+Poly)', PRED_STEPS)

# Koliko featura je Lasso izbacio na nulu?
coef = model_lasso.coef_                    # oblik: (10 izlaza, 435 featura)
nenulti_po_izlazu = np.sum(coef != 0, axis=1)
ukupno_featura = coef.shape[1]
# Feature je "aktivan" ako je nenult za bar jedan izlaz
aktivni_featurei = np.sum(np.any(coef != 0, axis=0))

print(f"\n  Ukupno featura (posle Poly): {ukupno_featura}")
print(f"  Aktivnih featura (nenult bar za 1 izlaz): {aktivni_featurei}")
print(f"  Izbačeno na nulu: {ukupno_featura - aktivni_featurei} ({(ukupno_featura-aktivni_featurei)/ukupno_featura*100:.1f}%)")
print(f"  Prosečno aktivnih po izlazu: {np.mean(nenulti_po_izlazu):.1f} / {ukupno_featura}")

# =============================================================================
# UPOREDJIVANJE MODELA
# =============================================================================
print(f"\n{'='*60}")
print(" UPOREDJIVANJE MODELA")
print(f"{'='*60}")
print(f"\n{'Model':<30} {'ADE':>10} {'FDE':>10} {'Promena':>10}")
print("-" * 62)
ade_o = test_metrics_ols['ade']
ade_r = test_metrics_ridge['ade']
ade_l = test_metrics_lasso['ade']
print(f"{'OLS (originalni)':<30} {ade_o:>10.6f} {test_metrics_ols['fde']:>10.6f} {'0.00%':>10}")
print(f"{'Ridge + Polynomial':<30} {ade_r:>10.6f} {test_metrics_ridge['fde']:>10.6f} {(ade_o-ade_r)/ade_o*100:>+9.2f}%")
print(f"{'Lasso + Polynomial':<30} {ade_l:>10.6f} {test_metrics_lasso['fde']:>10.6f} {(ade_o-ade_l)/ade_o*100:>+9.2f}%")

# ADE po koraku
print(f"\n ADE po koraku:")
print(f"{'Korak':>8} {'OLS':>12} {'Ridge+Poly':>12} {'Lasso+Poly':>12}")
print("-" * 50)
y_test_x = y_test[:, :PRED_STEPS]
y_test_y = y_test[:, PRED_STEPS:]
for s in range(PRED_STEPS):
    o_step = np.mean(np.sqrt((y_test_x[:,s]-y_pred_ols[:,s])**2   + (y_test_y[:,s]-y_pred_ols[:,PRED_STEPS+s])**2))
    r_step = np.mean(np.sqrt((y_test_x[:,s]-y_pred_ridge[:,s])**2 + (y_test_y[:,s]-y_pred_ridge[:,PRED_STEPS+s])**2))
    l_step = np.mean(np.sqrt((y_test_x[:,s]-y_pred_lasso[:,s])**2 + (y_test_y[:,s]-y_pred_lasso[:,PRED_STEPS+s])**2))
    print(f"{s+1:>8} {o_step:>12.6f} {r_step:>12.6f} {l_step:>12.6f}")

# =============================================================================
# VIZUELIZACIJA (najbolji od linearnih: biramo po ADE)
# =============================================================================
print(f"\n{'='*60}")
print(" VIZUELIZACIJA")
print(f"{'='*60}")

# Biramo model sa najmanjim ADE za vizualizaciju
kandidati = {
    'OLS': (ade_o, y_pred_ols),
    'Ridge+Poly': (ade_r, y_pred_ridge),
    'Lasso+Poly': (ade_l, y_pred_lasso),
}
najbolji_naziv = min(kandidati, key=lambda k: kandidati[k][0])
najbolji_pred = kandidati[najbolji_naziv][1]
print(f"  Vizualizujem najbolji linearni model: {najbolji_naziv} (ADE={kandidati[najbolji_naziv][0]:.6f})")

plot_prediction(X_test, y_test, najbolji_pred, df_test, f'{najbolji_naziv} (linearni)', OUTPUT_DIR, pred_steps=PRED_STEPS)
plot_trajektorija(X_test, y_test, najbolji_pred, df_test, f'{najbolji_naziv} (linearni)', OUTPUT_DIR, pred_steps=PRED_STEPS)

# =============================================================================
# ČUVANJE
# =============================================================================
# Čuvamo predikcije najboljeg linearnog modela (za kasnije poređenje sa drugim modelima)
np.save(os.path.join(OUTPUT_DIR, 'y_train_pred.npy'), y_pred_ols_train)
np.save(os.path.join(OUTPUT_DIR, 'y_test_pred.npy'), najbolji_pred)

metrics_df = pd.DataFrame([test_metrics_ols, test_metrics_ridge, test_metrics_lasso])
metrics_df.to_csv(os.path.join(OUTPUT_DIR, 'metrics.csv'), index=False)
print(f"\n Rezultati: {OUTPUT_DIR}/")


# =============================================================================
# ZAKLJUČAK
# =============================================================================

"""
ZAKLJUCAK - Linearni modeli

Rezultati (test ADE):
  OLS (sirovi feature):     0.1854   <- najbolji
  Ridge + Polynomial:       0.1896   (-2.27% u odnosu na OLS)
  Lasso + Polynomial:       0.1863   (-0.48% u odnosu na OLS)

Rezultati (test FDE):
  OLS (sirovi feature):     0.325479  <- najbolji
  Ridge + Polynomial:       0.332734  
  Lasso + Polynomial:       0.328347 

GLAVNI NALAZ:
Dodavanje polinomijalnih interakcija NIJE poboljšalo predikciju — naprotiv,
oba regularizovana modela su LOŠIJA od običnog OLS-a. Razlika je dosledna
kroz svih 5 koraka predikcije. 

1. Problem je za vecinu pesaka sustinski linearan u prostoru relativnih koordinata.
    
2.  POlinomijalne interakcije unose uglanom sum. Proizvodi poput rel_x_1 * rel_y_3 
    nemaju fizicko znacenje za kretanje, pa model trosi kapacitet na beskorisne 
    kombinacije. Zato Ridge, koji ne izbacuje faeture nego ih samo prigusi, ispada
    najlosiji - i dalje vuce sve interakcije, samo sa malim tezinama,

3.  Lasso (L1) potvrdjuje istu pricu iz drugog ugla, Feature selection ga vraca skoro 
    na nivo OLS-a - jer izbacujuci beskorisne interakcije rekonstruise jednostavniji,
    gotovo linearni model. Da su interakcije bile korisne, Lasso bi ih zadrzao i nadmasio 
    OLS, ali to se ne desava.

Eksperiment sa Ridge/Lasso + Polynomial pokazuje da povecavanje slozenosti linearnog
modela ne pomaze. Ako nelinearnost postoji (a postoji na ostrim skretanjima), one se 
ne mozee uhvatiti rucnim proizvodima koordinata. Pretpostavljam da je za ovo potrebno
koristiti modele koji bolje hvataju interakcije izmedju atributa kao sto su tree-based
modeli. Zbog toga cemo u nastavku i preci na Random-Forest tree i XGBoost.

DODATAK: (WINDOW_SIZE=5)

Duzi prozor (8) je bolji jer daje stabilniju procenu kretanja, a dodatna slozenost modela
ne pomaze jer je problem sustiniski linearan u prostoru relativnih koordinata.

Pokusao sam ovaj eksperiment iz razloga sto manji window povecava broj primera u sliding window datasetu,
ali izgleda da cak i duplo veci broj primera nije uspeo da poboljsa model jer gubi na kontekstu.

"""
