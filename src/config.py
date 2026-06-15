# ============================
# HIPERPARAMETRI
# ============================

WINDOW_SIZE = 8           # Broj prethodnih pozicija koje gledamo
PRED_STEPS = 5             # koliko frejmova unapred zelimo da predvidjamo

# ============================
# PODELA PODATAKA
# ============================

TEST_SIZE = 0.15
VAL_SIZE = 0.15
RANDOM_SEED = 42            # Fiksiramo nasumicnost da bi rezultati bili isti

# ============================
# PUTANJE DO FAJLOVA
# ============================

HOTEL_PATH = 'data/raw/Final_data_eth_hotel.csv'
UNIV_PATH = 'data/raw/Final_data_eth_univ.csv'
OUTPUT_DIR = 'data/processed'