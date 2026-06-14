"""
Funkcije za evaluaciju i vizuelizaciju predikcija kretanja pešaka.
Podržava multistep predikciju (PRED_STEPS > 1) sa ADE i FDE metrikama.
"""

import numpy as np
import pandas as pd
import matplotlib.pyplot as plt
import matplotlib.patches as mpatches
import os
from src.config import WINDOW_SIZE

# =============================================================================
# METRIKE
# =============================================================================

def izracunaj_metrike(y_true, y_pred, naziv_skupa, pred_steps=5):
    pred_steps = y_true.shape[1] // 2 #Dobijamo pred_steps tako sto podelimo broj kolona sa 2

    y_true_x = y_true[:, :pred_steps]   #Prvih pred_steps kolona su x koordinate
    y_true_y = y_true[:, pred_steps:]   #Drugih pred_steps kolona su y koordinate
    y_pred_x = y_pred[:, :pred_steps]
    y_pred_y = y_pred[:, pred_steps:]

    euclidian_per_step = np.sqrt((y_true_x - y_pred_x)**2 + (y_true_y - y_pred_y)**2)
    print(f"euclidian_per_step.shape = {euclidian_per_step.shape}") #Trebalo bi da bude (n, pred_steps) gde je n broj primera

    #ADE: prosecno rastojanje po svim koracima i svim primerima
    ade = np.mean(euclidian_per_step)
    std_ade = np.std(euclidian_per_step)
    median_ade = np.median(euclidian_per_step)

    #FDE: Rastojanje samo poslednjeg koraka izmedju predvidjene i zeljene pozicije
    fde = np.mean(euclidian_per_step[:, -1])
    std_fde = np.std(euclidian_per_step[:, -1])
    median_fde = np.median(euclidian_per_step[:, -1])

    #ADE po svakom koraku posebno
    ade_po_koraku = np.mean(euclidian_per_step, axis=0) #axis=0, znaci da radi mean po kolonama.

    mse = np.mean((y_true-y_pred)**2)
    mae = np.mean(np.abs(y_true-y_pred))
    rmse = np.sqrt(mse)

    print(f"\n{'='*60}")
    print(f" METRIKE — {naziv_skupa}")
    print(f"{'='*60}")

    print(f"  ADE: {ade:.6f} ± {std_ade:.6f} (medijan: {median_ade:.6f})")
    print(f"  FDE: {fde:.6f} ± {std_fde:.6f} (medijan: {median_fde:.6f})")
    print(f"  MSE: {mse:.6f}")
    print(f"  MAE: {mae:.6f}")
    print(f"  RMSE: {rmse:.6f}")
    print(f"\n  ADE po koraku:")
    for s in range(pred_steps):
        print(f"    Korak {s+1}: {ade_po_koraku[s]:.6f}")

    #Razlika izmedju ADE/FDE i MSE je taj sto MSE gleda x i y komponentu koordinate odvojeno dok ih ADE/FDE spaja u jednu brojku
    #Postavio sam MSE, MAE i RMSE kao standard, ali intuitivnije i bolje prikazuje gresku bas ADE/FDE

    return {
        'skup' : naziv_skupa,
        'mse' : mse,
        'mae' : mae,
        'rmse' : rmse,
        'ade' : ade,
        'std_ade' : std_ade,
        'median_ade' : median_ade,
        'fde' : fde,
        'std_fde' : std_fde,
        'median_fde' : median_fde,
        'ade_po_koraku' : ade_po_koraku.tolist(),   #konvertujemo np niz u listu kako bismo ga lepo prebacili u dataframe/csv
        'pred_steps' : pred_steps
    }

# =============================================================================
# VIZUELIZACIJA
# =============================================================================

def plot_prediction(X, y_true, y_pred, df_info, naziv_modela, output_path,
                     n_primera=6, pred_steps=5, seed=42):
    n_rel = WINDOW_SIZE - 1

    # Biramo prvih N unikatnih pešaka (sortiranih po unique_id) da bi
    # grafici bili uporedivi između različitih WINDOW_SIZE konfiguracija.
    # Za svakog izabranog pešaka uzimamo njegov PRVI primer u datasetu.
    jedinstveni_peds = sorted(df_info['unique_id'].unique())
    izabrani_peds = jedinstveni_peds[:min(n_primera, len(jedinstveni_peds))]

    indicies = []
    for ped in izabrani_peds:
        prvi_idx = df_info[df_info['unique_id'] == ped].index[0]
        indicies.append(prvi_idx)

    fig, axes = plt.subplots(2, 3, figsize=(15, 10))
    axes = axes.flatten()

    for ax_idx, sample_idx in enumerate(indicies):
        if ax_idx >= len(axes):
            break

        ax = axes[ax_idx]

        # Istorijski pomeraji + nula na kraju (trenutna pozicija)
        window_x = np.append(X[sample_idx, :n_rel], 0)
        window_y = np.append(X[sample_idx, n_rel:2*n_rel], 0)

        # Stvarni i predviđeni budući pomeraji
        true_dx = y_true[sample_idx, :pred_steps]
        true_dy = y_true[sample_idx, pred_steps:]
        pred_dx = y_pred[sample_idx, :pred_steps]
        pred_dy = y_pred[sample_idx, pred_steps:]

        # Kumulativne pozicije
        true_pos_x = np.cumsum(np.insert(true_dx, 0, 0))
        true_pos_y = np.cumsum(np.insert(true_dy, 0, 0))
        pred_pos_x = np.cumsum(np.insert(pred_dx, 0, 0))
        pred_pos_y = np.cumsum(np.insert(pred_dy, 0, 0))

        ped_id = df_info.iloc[sample_idx]['unique_id'] if 'unique_id' in df_info.columns else 'N/A'

        # --- CRTANJE ---
        # Istorijske pozicije
        ax.plot(window_x, window_y, 'o-', color='#2196F3', linewidth=2,
                markersize=6, label='Istorija (N=8)', zorder=2)
        ax.scatter(window_x[-1], window_y[-1], color='#0D47A1', s=120,
                    marker='o', label='Trenutna pozicija', zorder=3)

        # Stvarna putanja
        ax.plot(true_pos_x, true_pos_y, 'X-', color='#4CAF50', linewidth=2,
                markersize=8, label='Stvarna putanja', zorder=4)
        for s in range(pred_steps):
            ax.annotate(f'{s+1}', (true_pos_x[s+1], true_pos_y[s+1]),
                        fontsize=8, color='#4CAF50', fontweight='bold',
                        ha='center', va='bottom')

        # Predviđena putanja
        ax.plot(pred_pos_x, pred_pos_y, 'X--', color='#F44336', linewidth=2,
                markersize=8, label='Predviđena putanja', zorder=4)
        for s in range(pred_steps):
            ax.annotate(f'{s+1}', (pred_pos_x[s+1], pred_pos_y[s+1]),
                        fontsize=8, color='#F44336', fontweight='bold',
                        ha='center', va='bottom')

        # Greška po koraku
        for s in range(pred_steps):
            ax.plot([true_pos_x[s+1], pred_pos_x[s+1]],
                   [true_pos_y[s+1], pred_pos_y[s+1]],
                   color='#FF9800', linewidth=1, linestyle=':', alpha=0.7,
                   zorder=1)

        # Koordinatni početak
        ax.axhline(0, color='gray', linewidth=0.5, linestyle='-', alpha=0.3)
        ax.axvline(0, color='gray', linewidth=0.5, linestyle='-', alpha=0.3)

        # Podešavanja
        ax.set_title(f'Pešak: {ped_id} (primer {sample_idx})', fontsize=11, fontweight='bold')
        ax.set_xlabel('Δx (m) — relativno')
        ax.set_ylabel('Δy (m) — relativno')
        ax.grid(True, alpha=0.3)

        # Jednake skale
        sve_x = list(window_x) + list(true_pos_x) + list(pred_pos_x)
        sve_y = list(window_y) + list(true_pos_y) + list(pred_pos_y)
        max_range = max(np.max(np.abs(sve_x)), np.max(np.abs(sve_y))) * 1.4
        if max_range < 0.3:
            max_range = 1.0
        ax.set_xlim(-max_range, max_range)
        ax.set_ylim(-max_range, max_range)
        ax.set_aspect('equal')

        ax.legend(fontsize=7, loc='upper left', framealpha=0.9)

    # Sakrij preostale prazne subplotove
    for i in range(n_primera, len(axes)):
        axes[i].axis('off')

    plt.suptitle(f'{naziv_modela} — Stvarna vs Predviđena putanja (test skup, {pred_steps} koraka)',
                fontsize=14, fontweight='bold', y=1.02)
    plt.tight_layout()
    plt.savefig(os.path.join(output_path, 'predikcije_test.png'), dpi=150, bbox_inches='tight')
    plt.close()
    print(f"  Grafik sačuvan: {output_path}/predikcije_test.png")

def plot_trajektorija(X, y_true, y_pred, df_info, naziv_modela, output_path,
                      n_uzastopnih=4, pred_steps=5, seed=42):
    """
    Za JEDNOG pešaka prikazuje niz uzastopnih predikcija.
    Bira prvog pogodnog pešaka (sortirano po unique_id) da bi
    trajektorija bila ista kroz različite window_size konfiguracije.
    """
    n_rel = WINDOW_SIZE - 1

    # Tražimo pešake koji imaju bar n_uzastopnih primera.
    # Uzimamo PRVOG takvog pešaka (sortirano) da bi trajektorija
    # bila ista nezavisno od window_size.
    unique_peds = df_info['unique_id'].value_counts()
    pogodni = sorted(unique_peds[unique_peds >= n_uzastopnih].index.tolist())

    if not pogodni:
        print("  Nema pešaka sa dovoljno uzastopnih primera.")
        return

    izabrani_ped = pogodni[0]
    ped_indices = df_info[df_info['unique_id'] == izabrani_ped].index.values
    izabrani_indices = ped_indices[:n_uzastopnih]

    fig, ax = plt.subplots(1, 1, figsize=(14, 10))

    boje_istorija = plt.cm.Blues(np.linspace(0.3, 0.7, n_uzastopnih))
    boje_stvarno = plt.cm.Greens(np.linspace(0.4, 0.8, n_uzastopnih))
    boje_predikcija = plt.cm.Reds(np.linspace(0.4, 0.8, n_uzastopnih))

    for i, idx in enumerate(izabrani_indices):
        window_x = np.append(X[idx, :n_rel], 0)            # rel_x_1..(N-1) + T(0)
        window_y = np.append(X[idx, n_rel:2*n_rel], 0)    # rel_y_1..(N-1) + T(0)

        true_dx = y_true[idx, :pred_steps]
        true_dy = y_true[idx, pred_steps:]
        pred_dx = y_pred[idx, :pred_steps]
        pred_dy = y_pred[idx, pred_steps:]

        # Stvarne pozicije (kumulativno)
        true_pos_x = np.cumsum(np.insert(true_dx, 0, 0))
        true_pos_y = np.cumsum(np.insert(true_dy, 0, 0))
        pred_pos_x = np.cumsum(np.insert(pred_dx, 0, 0))
        pred_pos_y = np.cumsum(np.insert(pred_dy, 0, 0))

        # Istorija
        if i == 0:
            ax.plot(window_x, window_y, 'o-', color='#2196F3', linewidth=1.5,
                    markersize=4, alpha=0.5, label=f'Istorija (N={WINDOW_SIZE})', zorder=2)
        else:
            ax.plot(window_x, window_y, 'o-', color=boje_istorija[i], linewidth=1.5,
                    markersize=4, alpha=0.4, zorder=1)

        # Stvarna putanja
        ax.plot(true_pos_x, true_pos_y, 'X-', color=boje_stvarno[i], linewidth=2,
                markersize=6, zorder=4)
        ax.annotate(f't+{i+1}', xy=(true_pos_x[-1], true_pos_y[-1]),
                   fontsize=9, color=boje_stvarno[i], fontweight='bold')

        # Predviđena putanja
        ax.plot(pred_pos_x, pred_pos_y, 'X--', color=boje_predikcija[i], linewidth=2,
                markersize=6, zorder=3)

        # Greške
        for s in range(pred_steps):
            ax.plot([true_pos_x[s+1], pred_pos_x[s+1]],
                   [true_pos_y[s+1], pred_pos_y[s+1]],
                   color='#FF9800', linewidth=0.8, linestyle=':', alpha=0.5, zorder=1)

    # Koordinatni početak
    ax.scatter(0, 0, color='#0D47A1', s=150, marker='o', label='Trenutna pozicija', zorder=5)
    ax.axhline(0, color='gray', linewidth=0.5, alpha=0.3)
    ax.axvline(0, color='gray', linewidth=0.5, alpha=0.3)

    # Legenda
    legend_elements = [
        mpatches.Patch(color='#2196F3', label=f'Istorija (prozor od {WINDOW_SIZE})'),
        plt.Line2D([0], [0], marker='o', color='w', markerfacecolor='#0D47A1', markersize=10, label='Trenutna pozicija'),
        plt.Line2D([0], [0], marker='X', color='w', markerfacecolor='#4CAF50', markersize=10, label='Stvarna putanja'),
        plt.Line2D([0], [0], marker='X', color='w', markerfacecolor='#F44336', markersize=10, label='Predviđena putanja'),
        plt.Line2D([0], [0], color='#FF9800', linestyle=':', linewidth=2, label='Greška'),
    ]
    ax.legend(handles=legend_elements, loc='upper left', fontsize=10, framealpha=0.9)

    ax.set_title(f'Pešak: {izabrani_ped} — {n_uzastopnih} uzastopnih predikcija\n'
                 f'{naziv_modela} ({pred_steps} koraka)', fontsize=13, fontweight='bold')
    ax.set_xlabel('Δx (m) — relativno')
    ax.set_ylabel('Δy (m) — relativno')
    ax.grid(True, alpha=0.3)
    ax.set_aspect('equal')

    # Dinamički range
    sve_vrednosti = []
    for idx in izabrani_indices:
        sve_vrednosti.extend(X[idx, :n_rel])
        sve_vrednosti.extend(X[idx, n_rel:2*n_rel])
        sve_vrednosti.extend(y_true[idx])
        sve_vrednosti.extend(y_pred[idx])
    max_range = max(np.abs(sve_vrednosti)) * 1.6
    if max_range < 0.5:
        max_range = 1.5
    ax.set_xlim(-max_range, max_range)
    ax.set_ylim(-max_range, max_range)

    plt.tight_layout()
    plt.savefig(os.path.join(output_path, 'trajektorija_pesaka.png'), dpi=150, bbox_inches='tight')
    plt.close()
    print(f"  Trajektorija pešaka sačuvana: {output_path}/trajektorija_pesaka.png")