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
            frame_lookup[key] = []

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
        positions = group[['pos_x', 'pos_y']].values               #.values() radi konverziju iz pandas DataFrame-a u numpy niz (ndarray), jer su operacija sa njime brze posebno indeksiranje koje koristimo u nadolazecim for petljama.
        velocities = group[['vx', 'vy']].values
        frame_ids = group['frame_id'].values
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

            row_data = {'unique_id': ped_id}                            # Recnik koji gradimo feature po feature i na kraju dodamo u rows listu

            # === Features: relativne koordinate (T-7 do T-1) ===
            # [FIX] Petlja ide do window_size - 1 (dakle j = 0..6),
            # čime preskačemo poslednji element prozora (T samu)
            # jer je window[-1] - window[-1] = (0, 0) uvek.
            # To su bile mrtve kolone rel_x_8 i rel_y_8.
            # Sad imamo 14 relativnih koordinata umesto 16.

            for j in range(window_size - 1):
                rel_x = window[j][0] - current_pos[0]
                rel_y = window[j][1] - current_pos[1]
                row_data[f'rel_x_{j+1}'] = rel_x        #Levo-desno
                row_data[f'rel_y_{j+1}'] = rel_y        #Napred-nazad
            # === Socijalne feature: K=3 najbliža suseda ===
            K = 3
            # [FIX] Koristimo (ped_scene, current_frame) kao key,
            # pa se nikad ne mešaju pešaci iz različitih scena.
            lookup_key = (ped_scene, current_frame)
            others = [p for p in frame_lookup.get(lookup_key, []) if p[0] != ped_id] #Others su svi ostali pesaci u tom frejmu

            nn_features = np.zeros((K, 5))

            if others:
                other_data = np.array([ [p[1], p[2], p[3], p[4]] for p in others])  #Ovim pravimo np matricu sa kolonama pos_x, pos_y, vx, vy. Na njoj vektorski racunamo sve distance umesto u petlji
                dx_all = other_data[:, 0] - current_pos[0]
                dy_all = other_data[:, 1] - current_pos[1]
                dvx_all = other_data[:, 2] - current_vel[0]
                dvy_all = other_data[:, 3] - current_vel[1]
                dists = np.sqrt(dx_all**2 + dy_all**2)

                sorted_idx = np.argsort(dists)[:K]  #Na ovaj nacin pronalazimo K najbliza suseda
                speed = np.sqrt(current_vel[0]**2 + current_vel[1]**2) #current_vel sadrzi vx,vy atribute naseg pesaka u sadasnjiem trenutku
                
                for k, idx in enumerate(sorted_idx):
                    dx, dy = dx_all[idx], dy_all[idx]
                    dvx, dvy = dvx_all[idx], dvy_all[idx]
                    dist = dists[idx]
                    approach = -(dx*dvx + dy*dvy) / dist if dist > 1e-6 else 0      #Ako su dva pesaka na istom mestu onda je approach nula, ali to bi bila samo neka greska najverovatnije
                    
                    if speed > 1e-6:
                        dot = dx*current_vel[0] + dy*current_vel[1] #Ovaj proizvod govori nam da li je sused u pravcu kretanja
                        in_front = dot / (speed * dist + 1e-6) #Posto dot nije normalizovan, deljenjem sa speed*dist dobijamo kosinus koji daje vrednosti u [-1,1]

                    else:
                        in_front = 0.0

                    #Sada samo dodajemo ove atribute u promenljivu nn_features                   
                    nn_features[k,0] = dx
                    nn_features[k,1] = dy
                    nn_features[k,2] = dvx
                    nn_features[k,3] = dvy
                    nn_features[k,4] = approach * max(0.0, in_front)
                    
            
            #Atribute dodajemo u row_data, rekli smo da je to recnik koji gradimo feature po feature
            for k in range(K):
                row_data[f'nn_{k+1}_dx']        = nn_features[k,0]
                row_data[f'nn_{k+1}_dy']        = nn_features[k,1]
                row_data[f'nn_{k+1}_dvx']       = nn_features[k,2]
                row_data[f'nn_{k+1}_dvy']       = nn_features[k,3]
                row_data[f'nn_{k+1}_approach']  = nn_features[k,4]
           
            # === Targeti: relativna pomeranja u narednih PRED_STEPS koraka ===
            for step in range(pred_steps):
                target_pos = positions[i + window_size + step]
                row_data[f'delta_x_{step+1}'] = target_pos[0] - current_pos[0]
                row_data[f'delta_y_{step+1}'] = target_pos[1] - current_pos[1]
            
            rows.append(row_data) #Na ovaj nacin smo dodali jedan red za pesaka sa svim novim atributima dobijenim kroz Sliding WIndow tehniku
        
    return pd.DataFrame(rows)
    
print("Primenjujem sliding window na train skup...")
df_train = create_sliding_windows(df_train_raw, WINDOW_SIZE, PRED_STEPS)
print("Primenjujem sliding window na validation skup...")
df_val = create_sliding_windows(df_val_raw, WINDOW_SIZE, PRED_STEPS)
print("Primenjujem sliding window na train skup...")
df_test = create_sliding_windows(df_test_raw, WINDOW_SIZE, PRED_STEPS)

print(f"\n{'='*60}")
print(f"REZULTATI NAKON SLIDING WINDOW (N={WINDOW_SIZE})")
print(f"{'='*60}")
print(f"Train primeri: {len(df_train)}")
print(f"Val primeri:   {len(df_val)}")
print(f"Test primeri:  {len(df_test)}")
print(f"Ukupno primera: {len(df_train) + len(df_val) + len(df_test)}")

# =========================================
# CUVANJE
# =========================================

os.makedirs(OUTPUT_DIR, exist_ok=True)

EFFECTIVE_WINDOW = WINDOW_SIZE - 1
K = 3

feature_cols = (
    [f'rel_x_{j+1}' for j in range(EFFECTIVE_WINDOW)] + #7
    [f'rel_y_{j+1}' for j in range(EFFECTIVE_WINDOW)] + #7
    [f'nn_{k+1}_{feat}' for k in range(K)
    for feat in ['dx', 'dy', 'dvx', 'dvy', 'approach']] # 15
)   #ukupno: 14 + 15 = 29

target_cols = (
    [f'delta_x_{s+1}' for s in range(PRED_STEPS)] +
    [f'delta_y_{s+1}' for s in range(PRED_STEPS)]
)

print(f"Broj features: {len(feature_cols)} ({K*5} = {K}*5 KNN + 14 rel = {14 + K*5})")

df_train.to_csv(os.path.join(OUTPUT_DIR, 'train.csv'), index=False)
df_val.to_csv(os.path.join(OUTPUT_DIR, 'val.csv'), index=False)
df_test.to_csv(os.path.join(OUTPUT_DIR, 'test.csv'), index=False)

#Sada cuvamo .npy fajlove zato sto su mnogo brzi za ucitavanje i spremni su za ML pipeline, dok su .csv fajlovi namenjeni laksem iscitavanju dataseta od strane coveka
np.save(os.path.join(OUTPUT_DIR, 'X_train.npy'), df_train[feature_cols].values)
np.save(os.path.join(OUTPUT_DIR, 'y_train.npy'), df_train[target_cols].values)
np.save(os.path.join(OUTPUT_DIR, 'X_val.npy'),   df_val[feature_cols].values)
np.save(os.path.join(OUTPUT_DIR, 'y_val.npy'),   df_val[target_cols].values)
np.save(os.path.join(OUTPUT_DIR, 'X_test.npy'),  df_test[feature_cols].values)
np.save(os.path.join(OUTPUT_DIR, 'y_test.npy'),  df_test[target_cols].values)

print(f"\n{'='*60}")
print(f"✅ Dataset sačuvan u '{OUTPUT_DIR}'")
print(f"{'='*60}")
print(f"X_train shape: {df_train[feature_cols].values.shape}")
print(f"y_train shape: {df_train[target_cols].values.shape}")