# Dokumentacija — Predviđanje kretanja pešaka

## 1. Opis problema

Cilj projekta je predviđanje budućeg kretanja pešaka na osnovu istorije njihovih pozicija. Konkretno, na osnovu prethodnih 8 pozicija pešaka (oko 3 sekunde istorije) predviđa se njegovih narednih 5 pozicija (~2 sekunde unapred). Problem ima primenu u autonomnim vozilima, robotici i sistemima za praćenje gužve, gde je ključno proceniti gde će se osoba naći u bliskoj budućnosti.

Sa stanovišta mašinskog učenja, ovo je problem **regresije sa više izlaza** (multi-output regression): za svaki primer model predviđa 10 vrednosti — pomeraje po x i y osi za svaki od 5 budućih koraka.

### Skup podataka

Korišćen je BIWI Walking Pedestrians dataset (ETH Zurich), dve scene — Hotel i Univ — sa ukupno 749 pešaka i oko 12.000 detekcija. Svaka detekcija sadrži broj frejma, ID pešaka i koordinate (x, y) u metrima.

## 2. Postupak rešavanja

### 2.1 Početno preprocesiranje

- Dve scene su spojene u jedinstven skup, uz dodavanje prefiksa ID-jevima (`hotel_`, `univ_`) kako bi se razdvojili pešaci iz različitih scena.
- Provereno je postojanje nedostajućih vrednosti i anomalija.
- Izračunate su brzine pešaka kao razlika uzastopnih pozicija.
- **Ključna odluka:** scene se tretiraju odvojeno pri računanju socijalnih featura. Pošto se brojevi frejmova preklapaju između scena, pešaci iz različitih scena bi se inače pogrešno smatrali susedima.

### 2.2 Sprečavanje curenja podataka (data leakage)

Podela na train/val/test (70/15/15) urađena je **po pešacima, pre** sliding-window transformacije. Da je podela urađena posle, isti pešak bi mogao da ima primere i u trening i u test skupu, što bi dovelo do precenjenih rezultata.

### 2.3 Eksplorativna analiza

Analiziran je broj pešaka po frejmu i dužine trajektorija. Uočeno je da scene imaju relativno retku populaciju (5–10 pešaka po frejmu), što je kasnije objasnilo slab uticaj socijalnih featura.

### 2.4 Feature engineering

Za svaki primer formira se sledeći skup obeležja:

- **Relativne koordinate** (`rel_x`, `rel_y`): svaka prošla pozicija izražena je relativno u odnosu na trenutnu poziciju T. Time model ne vidi apsolutne koordinate, pa je invarijantan na lokaciju u prostoru (translaciona invarijantnost).
- **KNN socijalni featurei**: za 3 najbliža suseda u trenutku T računaju se relativna pozicija (`dx`, `dy`), relativna brzina (`dvx`, `dvy`) i `approach` (brzina prilaska, naglašena samo kada je sused u pravcu kretanja pešaka).

Izlaz (target) su relativni pomeraji za narednih 5 koraka.

### 2.5 Odabir i treniranje modela

Isprobano je više modela različite složenosti:

- **Baseline** — naivni model koji ponavlja poslednji pomeraj 5 puta.
- **Linearna regresija** — OLS, te Ridge i Lasso sa polinomijalnim featurima.
- **Random Forest** — sa tuningom hiperparametara.
- **XGBoost** — sa tuningom i sample weights koji naglašavaju nelinearna kretanja (skretanja).

Hiperparametri su podešavani na **validacionom** skupu; test skup je korišćen samo jednom, na kraju, radi nepristrasne procene.

### 2.6 Podešavanje hiperparametara

Za tree modele korišćen je sekvencijalni tuning na validacionom skupu. Za XGBoost su podešeni learning_rate, n_estimators, max_depth, subsample, colsample_bytree i L1/L2 regularizacija. Zanimljivo, tuning je dosledno birao **jednostavnije** modele (mali max_depth, jaka regularizacija) — signal da u podacima nema složenog nelinearnog obrasca.

## 3. Rezultati

Metrike: **ADE** (Average Displacement Error) — prosečna greška kroz svih 5 koraka, i **FDE** (Final Displacement Error) — greška na poslednjem koraku.

### 3.1 Poređenje modela (window=8, test skup)

| Model | ADE [m] | FDE [m] | R² |
|-------|:-------:|:-------:|:--:|
| Baseline | 0.892 | 1.737 | — |
| OLS | 0.188 | 0.330 | 0.977 |
| Random Forest | 0.194 | 0.330 | 0.975 |
| XGBoost | 0.189 | 0.327 | 0.976 |

### 3.2 Greška po tipu putanje (XGBoost)

Test skup je podeljen po odstupanju stvarne putanje od pravolinijske ekstrapolacije:

| Kategorija | Udeo | ADE [m] |
|-----------|:----:|:-------:|
| Prava linija (<30cm) | 55.5% | 0.11 |
| Blago skretanje | 38.7% | 0.25 |
| Oštro skretanje (>1m) | 5.8% | 0.50 |

### 3.3 Značaj atributa

Feature importance pokazuje da relativne koordinate nose **~99%** ukupne važnosti, a KNN socijalni featurei samo **~1%**. Poređenje modela sa svim featurima (29) vs samo relativnim koordinatama (14) potvrdilo je da izbacivanje socijalnih featura praktično ne menja tačnost.

### 3.4 Distribucija grešaka i najgori slučajevi (XGBoost)

Distribucija grešaka je izrazito desno-iskrivljena: većina predikcija je vrlo tačna, uz mali rep teških slučajeva. Median FDE je 0.23m, dok prosek diže upravo taj rep ekstremnih grešaka.

| Mera (FDE) | Vrednost [m] |
|-----------|:------------:|
| Median | 0.23 |
| Prosek (mean) | 0.33 |
| 95. percentil | 0.97 |
| 99. percentil | 1.35 |
| Maksimum | 2.19 |

Drugim rečima, 95% predikcija ima FDE ispod ~1m, a 99% ispod ~1.35m. Samo mali broj ekstremnih slučajeva prelazi te granice.

**5 najgorih slučajeva** (po FDE, XGBoost):

| Rang | Uzorak | ADE [m] | FDE [m] |
|:----:|:------:|:-------:|:-------:|
| 1 | 587 | 0.95 | 2.19 |
| 2 | 554 | 1.03 | 1.83 |
| 3 | 615 | 0.76 | 1.51 |
| 4 | 551 | 0.66 | 1.45 |
| 5 | 552 | 0.67 | 1.44 |

Vizuelna analiza najgorih slučajeva pokazuje da su to redom **nagla, oštra skretanja**: stvarna putanja naglo menja pravac, dok model — ne videvši nikakav nagoveštaj u istoriji — nastavlja približno pravolinijski. Ovo potvrđuje da glavni izvor greške nisu sistematske slabosti modela, već retki, iz istorije nepredvidivi manevri.

### 3.5 Uticaj dužine istorije

| Window | XGBoost ADE | OLS ADE | XGB vs OLS |
|:------:|:-----------:|:-------:|:----------:|
| 8 | 0.189 | 0.188 | −0.9% |
| 5 | 0.202 | 0.208 | +2.9% |

## 4. Diskusija

**Problem je u kratkom horizontu pretežno linearan.** OLS, jednostavna linearna regresija, postiže gotovo identičan rezultat kao XGBoost i Random Forest (razlike ispod par procenata). Razlog je što se kretanje pešaka na ~2 sekunde dobro aproksimira pravom linijom — preko 94% primera je pravolinijsko ili blago zakrivljeno.

**Najveći deo greške potiče od retkih oštrih skretanja.** Iako čine samo 5.8% test skupa, oštra skretanja imaju petostruko veću grešku od pravolinijskog kretanja. Na tim primerima tree modeli (XGBoost, RF) pokazuju prednost (~15–20% bolji FDE od OLS), ali se ta prednost gubi u ukupnom proseku jer su takvi primeri malobrojni.

**Socijalni featurei imaju ograničen uticaj.** Zbog retke populacije ETH dataseta (malo pešaka po frejmu), susedi su retko dovoljno blizu da značajno utiču na kretanje, pa KNN featurei doprinose svega ~1%. Sa kraćom istorijom (window=5) njihov uticaj blago raste, jer model ima manje sopstvenih podataka pa se relativno više oslanja na okolinu.

**Uticaj dužine istorije.** Duži prozor (8) daje bolju apsolutnu tačnost jer pruža jasniji linearni trend. Sa kraćim prozorom (5) ostaje nešto nelinearnog signala koji tree modeli iskoriste, pa XGBoost tu nadmašuje OLS — posebno na dužim horizontima predikcije.

## 5. Zaključak

Za predikciju kretanja pešaka na horizontu od ~2 sekunde, problem je u prostoru relativnih koordinata pretežno linearan: jednostavni linearni model je gotovo jednako tačan kao složeniji tree modeli. Glavno ograničenje su retka, nagla skretanja koja se ne mogu predvideti iz same istorije kretanja. Dalji napredak zahtevao bi modele sa temporalnom memorijom (Social LSTM, Transformer) i više podataka o retkim manevrima.

Model je eksportovan i izložen kroz FastAPI servis sa vizuelnim interfejsom, koji omogućava interaktivno crtanje istorije kretanja i prikaz predikcije.