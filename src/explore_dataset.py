import pandas as pd
import numpy as np
import matplotlib.pyplot as plt
import os


from config import *

# =============================================================================
# 1. UČITAVANJE PODATAKA
# =============================================================================

columns = ['frame_id', 'ped_id', 'pos_x', 'pos_y']

df_hotel = pd.read_csv(HOTEL_PATH, header=None, names=columns)
df_univ = pd.read_csv(UNIV_PATH, header=None, names=columns)

df_hotel['unique_id'] = 'hotel_' + df_hotel['ped_id'].astype(int).astype(str)
df_univ['unique_id'] = 'univ_' + df_univ['ped_id'].astype(int).astype(str)

df_all = pd.concat([df_hotel, df_univ], ignore_index=True)
df_all = df_all.sort_values(by=['unique_id', 'frame_id']).reset_index(drop=True)

# =============================================================================
# 2. OSNOVNI PREGLED (EDA - Exploratory Data Analysis)
# =============================================================================
print("=" * 60)
print("ISTRAŽIVAČKA ANALIZA DATASETA (EDA)")
print("=" * 60)

#Prvo sto radimo je probera koliko imamo podataka uopste, koliko je tu zapravo pesaka uoceno u kojoj sceni
print(f"\nUkupno redova: {len(df_all)}")
print(f"Ukupno jedinstvenihj pesaka: {df_all['unique_id'].nunique()}")

print(f"Iz scene Hotel: {df_hotel['unique_id'].nunique()} pesaka")
print(f"Iz scene Univ: {df_univ['unique_id'].nunique()} pesaka")

#=============================================================================
# 3. ANALIZA DUŽINE TRAJEKTORIJA (Koliko dugo vidimo svakog pešaka?)
# =============================================================================

traj_lengths = df_all.groupby('unique_id').size()

print(f"\n DUZINE TRAJEKTORIJA (Broj frejmova po pesaku):")
print(f"    Min:     {traj_lengths.min()}")
print(f"    Max:     {traj_lengths.max()}")
print(f"    Prosek:  {traj_lengths.mean():.1f}")
print(f"    Medijan: {traj_lengths.median():.0f}")
print(f"    Std:     {traj_lengths.std():.1f}")

# Prikazujemo Kvantil - Govori nam koliko procenat pesaka ima koju vrednost

for p in [10,25, 50, 75, 90, 95 , 99]:
    print(f"    {100-p}% pesaka ima >= {traj_lengths.quantile(p/100):.0f} frejmova")


# =============================================================================
# 4. KAKO IZABRATI "WINDOW_SIZE" (N)?
# =============================================================================

print(f"\n ANALIZA ZA IZBOR 'WINDOW_SIZE' (Za naš PRED_STEPS={PRED_STEPS}):")

for n_candidate in [4,6,8,10,12,15,20]:

    min_needed = n_candidate + PRED_STEPS

    #Brojimo koliko pesaka ispunjava uslov
    enough = (traj_lengths >= min_needed).sum()

    #Racunamo procenat pesaka koji nam ostaje
    pct = 100*enough/len(traj_lengths)

    print(f"   Ako je N={n_candidate:2d} -> {enough:3d} pešaka ({pct:.1f}%) je pogodno za treniranje (≥{min_needed} tačaka)")

#Iz ove analize mozemo videti da nam ostaje svega 68% pesaka, medjutim, ja i dalje ne bih smanjivao window_size iz razloga sto su pesaci koji imaju manje frejmova od ovoga mozda nedovoljni za ucenje
#Rekao bih da mogu bolje da naucim model sa manje pesaka koji imaju vise zabelezenih frejmova, nego sa vise pesaka koji imaju manje zabelezenih frejmova 
#Ali svakako, ako se ispostavi da su te daleke koordinate nebitne za trening, mozemo smanjiti WINDOW_SIZE

# =============================================================================
# 5. ANALIZA VREMENA (FREJMOVA)
# =============================================================================

print(f"\nANALIZA FREJMOVA (Da li ima preskakanja vremena):")
print(f"    Opseg frejmova u datasetu: {df_all['frame_id'].min()} do {df_all['frame_id'].max()}")

# Računamo razliku između uzastopnih frejmova za istog pešaka
# Očekujemo da ovo uvek bude isti broj (npr. +10, ako kamera snima svakih 10 frejmova)
df_all['frame_diff'] = df_all.groupby('unique_id')['frame_id'].diff()

print(f"    Prosecan razmak izmedju frejmova istog pesaka: {df_all['frame_diff'].mean():.2f}")
print(f"    Medijan razmaka (najcesci skok): {df_all['frame_diff'].median():.0f}")
print(f"    Max skok (Ako je znatno veći od medijana, imamo 'rupe' u snimku!): {df_all['frame_diff'].max()}")

#Sve je okej, nema preskakanja frejmova.

# =============================================================================
# 6. ANALIZA KOORDINATA I RELATIVNIH KRETANJA
# =============================================================================

print(f"\nANALIZA APSOLUTNIH KOORDINATA (Pozicija u prostoru):")
print(f"    X opseg kretanja: {df_all['pos_x'].min():.2f}m do{df_all['pos_x'].max():.2f}m")
print(f"    Y opseg kretanja: {df_all['pos_y'].min():.2f}m do{df_all['pos_y'].max():.2f}m")

#Sada radiomo analizu brzina, koliko se pesaci pomeraju izmedju dva frejma

df_sorted = df_all.sort_values(['unique_id', 'frame_id']).copy()

df_sorted['delta_x'] = df_sorted.groupby('unique_id')['pos_x'].diff()
df_sorted['delta_y'] = df_sorted.groupby('unique_id')['pos_y'].diff()

print(f"\nRELATIVNA KRETANJA (Pomeraj između dva uzastopna frejma):")
print(f"   Δx - prosek: {df_sorted['delta_x'].mean():.4f}m, std: {df_sorted['delta_x'].std():.4f}")
print(f"   Δy - prosek: {df_sorted['delta_y'].mean():.4f}m, std: {df_sorted['delta_y'].std():.4f}")

df_sorted['speed'] = np.sqrt(df_sorted['delta_x']**2 + df_sorted['delta_y']**2)
print(f"   Prosečna brzina pešaka (pomeraj po frejmu): {df_sorted['speed'].mean():.4f}m")
print(f"   Maksimalna zabeležena brzina: {df_sorted['speed'].max():.4f}m")
print(df_sorted[df_sorted['speed'] == df_sorted['speed'].max()][['ped_id', 'speed']])

# =============================================================================
# 7. SOCIJALNI KONTEKST: Broj pešaka po frejmu (Gustina gužve)
# =============================================================================

print(f"GUSTINA PESAKA (broj pesaka po frejmu):")

os.makedirs('results/eda', exist_ok=True)

for scena, df_scene in [('Hotel', df_hotel), ('Univ', df_univ)]:
    #Brojimo koliko jedinstvenih pesaka ima u svakom frejmu
    peds_per_frame = df_scene.groupby('frame_id')['unique_id'].nunique()

    print(f"\n   {scena} SCENA:")
    print(f"      Ukupno posmatranih frejmova: {len(peds_per_frame)}")
    print(f"      Prosek pešaka po frejmu:     {peds_per_frame.mean():.1f}")
    print(f"      Medijan pešaka po frejmu:    {peds_per_frame.median():.0f}")
    print(f"      Max pešaka u jednom frejmu:  {peds_per_frame.max()}")

    fig, axes = plt.subplots(1, 2, figsize=(14, 5))
    # Levo: Histogram (Raspodela gužve)
    axes[0].hist(peds_per_frame, bins=15, edgecolor='black', alpha=0.7)
    axes[0].axvline(peds_per_frame.mean(), color='red', linestyle='--', label=f'Prosek={peds_per_frame.mean():.1f}')
    axes[0].set_xlabel('Broj prisutnih pešaka')
    axes[0].set_ylabel('Broj frejmova sa tom gužvom')
    axes[0].set_title(f'{scena} — Distribucija gužve')
    axes[0].legend()
    
    # Desno: Vremenska serija (Kako se gužva menja kroz vreme)
    axes[1].plot(peds_per_frame.index, peds_per_frame.values, alpha=0.7, linewidth=0.5)
    axes[1].axhline(peds_per_frame.mean(), color='red', linestyle='--', label=f'Prosek={peds_per_frame.mean():.1f}')
    axes[1].set_xlabel('Frame ID (Vreme)')
    axes[1].set_ylabel('Broj prisutnih pešaka')
    axes[1].set_title(f'{scena} — Gužva kroz vreme')
    axes[1].legend()
    
    plt.tight_layout()
    # Čuvamo sliku u folder
    putanja_slike = f'results/eda/peds_per_frame_{scena.lower()}.png'
    plt.savefig(putanja_slike, dpi=100)
    print(f"      [!] Grafik sačuvan u: {putanja_slike}")
    
    # Prikazujemo sliku na ekranu (Moras da ugasiš prozorčić da bi se kod nastavio!)
    plt.show()

print(f"\n🔍 PRVIH 10 REDOVA U DATASETU (Kako podaci zapravo izgledaju):")
print(df_all[['unique_id', 'frame_id', 'pos_x', 'pos_y']].head(10))
