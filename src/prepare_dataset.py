import pandas as pd
import numpy as np
import os
from sklearn.model_selection import train_test_split

from config import *

# ============================
# UCITAVANJE I PREDPROCESIRANJE
# ============================

columns = ['frame_id', 'ped_id', 'pos_x', 'pos_y']

df_hotel = pd.read_csv(HOTEL_PATH, header=None, names=columns)
df_univ = pd.read_csv(UNIV_PATH, header=None, names=columns)



# =========================================
# PREDPROCESIRANJE (Nedostajuce vrednosti i duplikati)
# =========================================

print("---- PREDPROCESIRANJE PODATAKA ----")
print(f"Nedostajuce vrednosti (Hotel): {df_hotel.isnull().sum().sum()}")
print(f"Nedostajuce vrednosti (Univ): {df_univ.isnull().sum().sum()}")

df_hotel = df_hotel.dropna()
df_univ = df_univ.dropna()

# Proveravanje duplikata

dup_hotel = df_hotel.duplicated(subset=['frame_id', 'ped_id']).sum()
if dup_hotel > 0:
    print("HOTEL IMA DUPLIKATE")
    df_hotel = df_hotel.drop_duplicates(subset=['frame_id', 'ped_id'])


dup_univ = df_univ.duplicated(subset=['frame_id', 'ped_id']).sum()
if dup_univ > 0:
    print("UNIV IMA DUPLIKATE")
    df_univ = df_univ.drop_duplicates(subset=['frame_id', 'ped_id'])

#Duplikate posmatramo na skupu podataka frame_id i ped_id jer mi i treba da imamo vise redova sa istim ped_id-jem, ali sa razlicitim frejmovima
#Jedino sto se moze svrstati u duplikata je ped_id koji je u istom frejmu.

#Pravimo jedinsvene ID-jeve zato sto koristimo dva dataseta iz dve razlicite scene
#Da ne radimo ovako nesto, imali bismo dva razlicita pesaka koji imaju isti ped_id i tretirali bismo ga kao jednog pesaka

print(df_univ.head())
df_hotel['unique_id'] = 'hotel_' + df_hotel['ped_id'].astype(int).astype(str)
df_univ['unique_id'] = 'univ_' + df_univ['ped_id'].astype(int).astype(str)
print(df_univ.head())

#Belezimo scenu kako bismo lakse pronasli susede naseg pesaka
df_hotel['scene'] = 'hotel'
df_univ['scene']  = 'univ'

#Spajamo ove dve tabele u jednu, nakon sto smo uspesno prepravili id-jeve da budu unique
df_all = pd.concat([df_hotel, df_univ], ignore_index=True)
df_all = df_all.sort_values(by=['unique_id', 'frame_id']).reset_index(drop=True)

#Racunanje brzine
df_all['vx'] = df_all.groupby('unique_id')['pos_x'].diff().fillna(0)
df_all['vy'] = df_all.groupby('unique_id')['pos_y'].diff().fillna(0) #diff samo racuna razliku izmedju dva atributa u uzastopnim redovima, tako dobijamo brzinu

# iter = 0
# for uid, group in df_all.groupby('unique_id'):
#     print(f"Pesak: {uid}")
#     print(group.head())
#     print("------")
#     iter += 1
#     if iter == 7:
#         break

print(f"Ukupno redova nakon spajanja: {len(df_all)}")
print(f"Ukupno jedinstvenih pesaka: {df_all['unique_id'].nunique()}")

# =========================================
# PODELA PODATAKA
# =========================================

pedestrians = df_all['unique_id'].unique()
rest_peds, test_peds = train_test_split(pedestrians, test_size=TEST_SIZE, random_state=RANDOM_SEED)

val_ratio = VAL_SIZE/(1-TEST_SIZE) #ovo radimo zato sto ne zelimo 15% od preostalog seta, vec 15% od celokupnog dataseta. Bez ovoga bismo dobili 0.15 od 0.85 a ovako dobijemo 0.15/0.85 * 0.85

train_peds, val_peds = train_test_split(rest_peds, test_size=val_ratio, random_state=RANDOM_SEED)

print(f"Train: {len(train_peds)} pesaka")
print(f"Val: {len(val_peds)} pesaka")
print(f"Test: {len(test_peds)} pesaka")

#Posto smo do sada sve podatke delili samo preko unique_id-ja, nismo eksplicitno delili redove
#na ovaj nacin pravimo boolean masku koja ce vracati True ako se odredjeni unique_id nalazi u
#splitu ili False u suprotnom. Onda mi na taj nacin mozemo da napravimo raw podatke bas u vidu tabela
#sa svaki split SAMO preko unique_id-ja.
df_train_raw = df_all[df_all['unique_id'].isin(train_peds)].copy()
df_val_raw = df_all[df_all['unique_id'].isin(val_peds)].copy()
df_test_raw = df_all[df_all['unique_id'].isin(test_peds)].copy()

def create_sliding_windows(df, window_size=8, pred_steps=1):
    frame_lookup = {}
    for _, row in df.iterrows():
        key = (row['scene'], row['frame_id'])
        if key not in frame_lookup:
            frame_lookup[key].append((
                row['unique_id'],
                row['pos_x'],
                row['pos_y'],
                row['vx'],
                row['vy']
            ))
#Ovo pravi frame_lookup koji izgleda otprilike ovako, brojevi su naravno izmisljeni
# {
#     ('hotel', 100): [('hotel_1', 2.3, 4.5, 0.1, 0.2),
#                      ('hotel_3', 3.1, 5.2, 0.0, 0.3)],

#     ('hotel', 101): [('hotel_1', 2.4, 4.7, 0.1, 0.2)],

#     ('univ', 100):  [('univ_5', 1.2, 3.4, 0.2, 0.1),
#                      ('univ_9', 4.0, 2.1, 0.1, 0.0)],
#     ...
# }
#Sadrzi sve pesake koji se nalaze u tom frejmu

    rows = []

    for ped_id, group in df.groupby('unique_id'):
        group = group.sort_values('frame_id').reset_index(drop=True) # Znaci u grupi sortiramo pozicije pesaka po frejmu i resetujemo indekse da ponovo idu rastuce od 0
        positions = group[['pos_x', 'pos_y']].values()               #.values() radi konverziju iz pandas DataFrame-a u numpy niz (ndarray), jer su operacija sa njime brze posebno indeksiranje koje koristimo u nadolazecim for petljama.
        velocities = group[['vx', 'vy']].values()
        frame_ids = group['frame_id'].values()
        ped_scene = group['scene'].iloc[0]                          # Mozemo primetiti da je scena za svaki red ista jer se radi o istom pesaku, zato koristimo iloc[0] koji nam vraca vrednost prvog reda

        min_required = window_size + pred_steps
        if len(positions) < min_required:                           # Ako nemamo dovoljno frejmova zabelezenih o ovom pesaku, preskacemo ga
            continue

        
        for i in range(len(positions) - window_size - pred_steps + 1): # Ovo je broj primera koji mozemo da izvucemo od jednog pesaka. 
                                                                       #Pesak sa 20 frejmova, nama treba 8+5 frejmova, mozemo napraviti 20-8-5+1 = 8 primera
            window = positions[i:i+window_size]
            current_pos = window[-1]
            current_vel = velocities[i+window_size-1]
            current_frame = frame_ids[i+window_size-1]

            row_data = {'unique_id': ped_id}
            
