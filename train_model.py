import pandas as pd
import numpy as np
import pickle

from sklearn.tree import DecisionTreeRegressor
from sklearn.preprocessing import LabelEncoder
from sklearn.model_selection import train_test_split
from sklearn.metrics import (
    mean_absolute_error,
    mean_squared_error,
    r2_score
)

# ==========================================
# LOAD DATASET
# ==========================================

print("Loading dataset...")

df = pd.read_csv(
    "dataset/combined_kendaraan_bermotor.csv"
)

print("Jumlah data awal :", len(df))

# ==========================================
# CLEANING NAMA WILAYAH
# ==========================================

df["nama_kabupaten_kota"] = (
    df["nama_kabupaten_kota"]
    .astype(str)
    .str.strip()
    .str.upper()
)

df["nama_kabupaten_kota"] = (
    df["nama_kabupaten_kota"]
    .str.replace(r"\s+", " ", regex=True)
)

df["nama_kabupaten_kota"] = (
    df["nama_kabupaten_kota"]
    .str.replace("KAB.", "KABUPATEN", regex=False)
)

# ==========================================
# HAPUS BARIS AGREGAT JAWA BARAT
# ==========================================

df = df[
    df["nama_kabupaten_kota"] != "JAWA BARAT"
]

# ==========================================
# DAFTAR RESMI 27 KAB/KOTA JAWA BARAT
# ==========================================

wilayah_jabar = [
    "KABUPATEN BANDUNG",
    "KABUPATEN BANDUNG BARAT",
    "KABUPATEN BEKASI",
    "KABUPATEN BOGOR",
    "KABUPATEN CIAMIS",
    "KABUPATEN CIANJUR",
    "KABUPATEN CIREBON",
    "KABUPATEN GARUT",
    "KABUPATEN INDRAMAYU",
    "KABUPATEN KARAWANG",
    "KABUPATEN KUNINGAN",
    "KABUPATEN MAJALENGKA",
    "KABUPATEN PANGANDARAN",
    "KABUPATEN PURWAKARTA",
    "KABUPATEN SUBANG",
    "KABUPATEN SUKABUMI",
    "KABUPATEN SUMEDANG",
    "KABUPATEN TASIKMALAYA",
    "KOTA BANDUNG",
    "KOTA BANJAR",
    "KOTA BEKASI",
    "KOTA BOGOR",
    "KOTA CIMAHI",
    "KOTA CIREBON",
    "KOTA DEPOK",
    "KOTA SUKABUMI",
    "KOTA TASIKMALAYA"
]

df = df[
    df["nama_kabupaten_kota"].isin(
        wilayah_jabar
    )
]

print(
    "Jumlah Kabupaten/Kota:",
    df["nama_kabupaten_kota"].nunique()
)

print(
    "Jumlah data setelah cleaning:",
    len(df)
)

# ==========================================
# CEK TAHUN
# ==========================================

print("\nTahun tersedia:")
print(sorted(df["tahun"].unique()))

# ==========================================
# UBAH KE FORMAT LONG
# ==========================================

data_long = []

for _, row in df.iterrows():

    tahun   = row["tahun"]
    wilayah = row["nama_kabupaten_kota"]

    data_long.append([tahun, wilayah, 0, row["mobil_penumpang"]])
    data_long.append([tahun, wilayah, 1, row["sepeda_motor"]])
    data_long.append([tahun, wilayah, 2, row["mobil_bus"]])
    data_long.append([tahun, wilayah, 3, row["mobil_barang"]])

new_df = pd.DataFrame(
    data_long,
    columns=[
        "tahun",
        "wilayah",
        "kategori_kendaraan",
        "jumlah_kendaraan"
    ]
)

# ==========================================
# HANDLE MISSING VALUE
# ==========================================

new_df["jumlah_kendaraan"] = (
    pd.to_numeric(
        new_df["jumlah_kendaraan"],
        errors="coerce"
    ).fillna(0)
)

# ==========================================
# ENCODING WILAYAH
# ==========================================

encoder = LabelEncoder()

new_df["kode_wilayah_encoded"] = (
    encoder.fit_transform(new_df["wilayah"])
)

# ==========================================
# TIME SERIES FEATURE ENGINEERING
# ==========================================

new_df = new_df.sort_values(
    by=["wilayah", "kategori_kendaraan", "tahun"]
)

# Nilai tahun sebelumnya
new_df["lag_1"] = (
    new_df.groupby(
        ["wilayah", "kategori_kendaraan"]
    )["jumlah_kendaraan"]
    .shift(1)
)

# Nilai 2 tahun sebelumnya
new_df["lag_2"] = (
    new_df.groupby(
        ["wilayah", "kategori_kendaraan"]
    )["jumlah_kendaraan"]
    .shift(2)
)

# Pertumbuhan — hindari division by zero & infinity
new_df["growth"] = (
    (new_df["lag_1"] - new_df["lag_2"])
    / new_df["lag_2"].replace(0, np.nan)
)

new_df["lag_1"]  = new_df["lag_1"].fillna(0)
new_df["lag_2"]  = new_df["lag_2"].fillna(0)
new_df["growth"] = (
    new_df["growth"]
    .replace([np.inf, -np.inf], np.nan)
    .fillna(0)
)

# Simpan data training
new_df.to_csv("data_training_final.csv", index=False)

# ==========================================
# FEATURE & TARGET
# ==========================================

X = new_df[[
    "tahun",
    "kode_wilayah_encoded",
    "kategori_kendaraan",
    "lag_1",
    "lag_2",
    "growth"
]]

y = new_df["jumlah_kendaraan"]

# ==========================================
# SPLIT DATA
# ==========================================

X_train, X_test, y_train, y_test = train_test_split(
    X, y,
    test_size=0.2,
    random_state=42
)

# ==========================================
# TRAIN MODEL
# ==========================================

print("\nTraining model...")

model = DecisionTreeRegressor(
    random_state=42,
    max_depth=10
)

model.fit(X_train, y_train)

# ==========================================
# EVALUASI
# ==========================================

pred = model.predict(X_test)

mae  = mean_absolute_error(y_test, pred)
rmse = mean_squared_error(y_test, pred) ** 0.5
r2   = r2_score(y_test, pred)

print("\n===== HASIL EVALUASI =====")
print("MAE  :", round(mae,  2))
print("RMSE :", round(rmse, 2))
print("R²   :", round(r2,   4))

# ==========================================
# SIMPAN MODEL & ENCODER
# ==========================================

with open("decision_tree_model.pkl", "wb") as f:
    pickle.dump(model, f)

with open("label_encoder_wilayah.pkl", "wb") as f:
    pickle.dump(encoder, f)

print("\nModel berhasil disimpan!")
print("  decision_tree_model.pkl")
print("  label_encoder_wilayah.pkl")

# ==========================================
# CEK HASIL ENCODING
# ==========================================

print("\n===== DAFTAR WILAYAH =====")
for i, wilayah in enumerate(encoder.classes_):
    print(i, "-", wilayah)

# ==========================================
# PREDIKSI BEBERAPA TAHUN KE DEPAN
# ==========================================

def predict_future_years(df, model, n_years=5, tahun_mulai=None):
    """
    Memprediksi jumlah kendaraan untuk beberapa tahun ke depan
    menggunakan rolling forecast (lag diperbarui tiap iterasi).

    Parameters:
        df          : DataFrame training (sudah ada lag_1, lag_2, growth)
        model       : Model yang sudah dilatih
        n_years     : Jumlah tahun yang ingin diprediksi
        tahun_mulai : Tahun awal prediksi (default: tahun terakhir + 1)

    Returns:
        DataFrame berisi hasil prediksi
    """
    if tahun_mulai is None:
        tahun_mulai = int(df["tahun"].max()) + 1

    # Ambil data terakhir tiap grup sebagai titik awal
    last_known = (
        df.sort_values("tahun")
        .groupby(["wilayah", "kategori_kendaraan"])
        .last()
        .reset_index()
    )

    all_predictions = []

    for _, row in last_known.iterrows():
        wilayah      = row["wilayah"]
        kategori     = row["kategori_kendaraan"]
        kode_wilayah = row["kode_wilayah_encoded"]

        lag_1 = row["jumlah_kendaraan"]  # nilai tahun terakhir
        lag_2 = row["lag_1"]             # nilai 2 tahun terakhir

        for i in range(n_years):
            tahun_pred = tahun_mulai + i

            # Hitung growth — hindari division by zero
            growth = (
                (lag_1 - lag_2) / lag_2
                if lag_2 != 0 else 0.0
            )

            fitur = pd.DataFrame([{
                "tahun"               : tahun_pred,
                "kode_wilayah_encoded": kode_wilayah,
                "kategori_kendaraan"  : kategori,
                "lag_1"               : lag_1,
                "lag_2"               : lag_2,
                "growth"              : growth,
            }])

            hasil_pred = max(0, model.predict(fitur)[0])

            all_predictions.append({
                "wilayah"           : wilayah,
                "kategori_kendaraan": kategori,
                "tahun"             : tahun_pred,
                "jumlah_kendaraan"  : round(hasil_pred),
                "lag_1"             : round(lag_1),
                "lag_2"             : round(lag_2),
                "growth"            : round(growth, 4),
                "tipe"              : "prediksi"
            })

            # Rolling forecast: geser lag untuk tahun berikutnya
            lag_2 = lag_1
            lag_1 = hasil_pred

    return pd.DataFrame(all_predictions)


# ==========================================
# JALANKAN PREDIKSI
# ==========================================

N_YEARS = 5  # ubah sesuai kebutuhan

print(f"\nMemprediksi {N_YEARS} tahun ke depan...")

df_prediksi = predict_future_years(
    df=new_df,
    model=model,
    n_years=N_YEARS,
    tahun_mulai=None  # otomatis: tahun terakhir + 1
)

print(df_prediksi.head(20))

# ==========================================
# GABUNGKAN DATA HISTORIS + PREDIKSI
# ==========================================

new_df["tipe"] = "historis"

df_combined = pd.concat(
    [
        new_df[[
            "wilayah",
            "kategori_kendaraan",
            "tahun",
            "jumlah_kendaraan",
            "lag_1",
            "lag_2",
            "growth",
            "tipe"
        ]],
        df_prediksi
    ],
    ignore_index=True
)

df_combined = df_combined.sort_values(
    by=["wilayah", "kategori_kendaraan", "tahun"]
)

df_combined.to_csv("data_prediksi_masa_depan.csv", index=False)

tahun_akhir = int(new_df["tahun"].max())

print("\n✅ Prediksi selesai!")
print(f"   Tahun prediksi : {tahun_akhir + 1} - {tahun_akhir + N_YEARS}")
print("   File disimpan  : data_prediksi_masa_depan.csv")