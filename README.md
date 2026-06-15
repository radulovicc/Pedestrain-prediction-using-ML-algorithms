# Predviđanje kretanja pešaka

Predikcija kretanja pešaka na osnovu istorije njihovih pozicija. Cilj je da se na osnovu prethodnih pozicija pešaka predvidi njegovih narednih 5 pozicija (~2 sekunde unapred), korišćenjem klasičnih ML modela (linearna regresija, Random Forest, XGBoost).

## Dataset

Korišćen je **BIWI Walking Pedestrians** dataset (ETH Zurich Computer Vision Lab), procesiran u apsolutne koordinate. Sastoji se od dve scene:

- **Hotel** i **Univ** — ukupno 749 pešaka, ~12.000 detekcija.

Svaka detekcija sadrži `frame_id`, `ped_id`, `pos_x`, `pos_y`. Scene se tokom obrade tretiraju odvojeno kako se pešaci iz različitih scena ne bi pogrešno smatrali susedima.

- **Originalni dataset**: BIWI Walking Pedestrians — ETH Zurich Computer Vision Lab
  - [Zvanična stranica](https://www.vision.ee.ethz.ch/datasets/bh-pedestrians/)
- **Preprocessing (video → CSV koordinate)**: [ETH-UCY-Preprocessing](https://github.com/Habiba-Amroune/ETH-UCY-Preprocessing/tree/main) — [@Habiba-Amroune]

## Pristup

### Priprema podataka (`prepare_dataset.py`)
- Spajanje dve scene uz jedinstvene ID-jeve pešaka (`hotel_`, `univ_` prefiksi).
- Podela na train/val/test **po pešacima** (70/15/15), pre sliding-window transformacije, kako bi se izbeglo curenje podataka (data leakage).
- Sliding window: za svakog pešaka u trenutku T uzima se istorija prethodnih pozicija (relativne koordinate) i predviđa narednih 5 pomeraja.

### Featurei
- **Relativne koordinate** istorije kretanja (`rel_x`, `rel_y`) — pozicije izražene relativno u odnosu na trenutnu poziciju T (translaciona invarijantnost).
- **KNN socijalni featurei** — za 3 najbliža suseda u trenutku T: relativna pozicija (`dx`, `dy`), relativna brzina (`dvx`, `dvy`) i `approach` (brzina prilaska, naglašena samo kad je sused ispred u pravcu kretanja).

### Modeli
- **Baseline** — naivni: ponavlja poslednji pomeraj 5 puta.
- **Linearna regresija** — OLS, plus Ridge i Lasso sa polinomijalnim featurima.
- **Random Forest** — sa tuningom hiperparametara na validacionom skupu.
- **XGBoost** — sa tuningom i sample weights za naglašavanje nelinearnih kretanja (skretanja).

Tuning hiperparametara se radi na **validacionom** skupu; test skup se koristi samo jednom, na kraju. Sve metrike (ADE, FDE) računaju se za više dužina istorije (window_size = 8 i 5).

## Struktura projekta

```
.
├── data/
│   ├── raw/                  # Sirovi ETH podaci (Hotel, Univ)
│   └── processed/            # Generisani train/val/test (.npy, .csv)
├── src/
│   ├── config.py             # WINDOW_SIZE, PRED_STEPS i ostala podešavanja
│   ├── prepare_dataset.py    # Priprema podataka + sliding window
│   ├── evaluation.py         # Metrike (ADE/FDE) i vizualizacija
│   ├── explore_dataset.py    # EDA
│   ├── models/
│   │   ├── baseline.py
│   │   ├── linear_regression.py
│   │   ├── random_forest.py
│   │   └── xgb_model.py
│   ├── advanced_evaluation.py        # Analiza grešaka po tipu putanje
│   └── vizuelizacija.py   # Grafici za odbranu
└── results/
    └── models/
        ├── baseline/window_{N}/
        ├── linear_regression/window_{N}/
        ├── random_forest/window_{N}/
        ├── xgboost/window_{N}/
        ├── evaluation/window_{N}/
        └── za_odbranu/window_{N}/
```

Rezultati se čuvaju odvojeno po modelu i po dužini istorije (`window_8`, `window_5`), radi lakšeg poređenja.

## Pokretanje

Projekat koristi [`uv`](https://github.com/astral-sh/uv) za upravljanje zavisnostima.

```bash
# Instalacija zavisnosti
uv sync
```

Pipeline se pokreće redom (za jedan window). Dužina istorije se podešava u `src/config.py` (`WINDOW_SIZE`):

```bash
# 1. Priprema podataka (obavezno nakon svake izmene WINDOW_SIZE)
uv run src/prepare_dataset.py

# 2. Modeli
uv run src/models/baseline.py
uv run src/models/linear_regression.py
uv run src/models/random_forest.py
uv run src/models/xgb_model.py

# 3. Evaluacija i vizualizacija
uv run src/advanced_evaluation.py
uv run src/vizuelizacija_za_odbranu.py
```

Da bi se dobili rezultati za drugu dužinu istorije, promeni `WINDOW_SIZE` u `src/config.py` i ponovi ceo pipeline od koraka 1.

## Metrike

- **ADE** (Average Displacement Error) — prosečna greška kroz svih 5 koraka.
- **FDE** (Final Displacement Error) — greška na poslednjem (5.) koraku.

## Glavni nalazi

- Problem je u kratkom horizontu (~2s) pretežno **linearan** — OLS je gotovo jednako dobar kao tree modeli (razlike ispod nekoliko procenata).
- Najveći deo ukupne greške potiče od male grupe **oštrih skretanja**; na pravolinijskom kretanju su svi modeli skoro identični.
- Tree modeli (RF, XGBoost) pokazuju prednost upravo na nelinearnim kretanjima, ali se ta prednost gubi u proseku jer su takvi primeri retki.
- KNN socijalni featurei imaju malu ukupnu važnost, što se pripisuje retkoj populaciji ETH dataseta (malo pešaka po frejmu).