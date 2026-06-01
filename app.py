import streamlit as st
import pandas as pd
import numpy as np
import plotly.express as px
import pickle

# ==================================================
# CONFIG
# ==================================================

st.set_page_config(
    page_title="Smart Vehicle Dashboard",
    page_icon="🚗",
    layout="wide"
)

# ==================================================
# LOGIN
# ==================================================

USERNAME = "sandy"
PASSWORD = "sandy123"

if "login" not in st.session_state:
    st.session_state.login = False

if not st.session_state.login:

    st.markdown("""
    <div style='text-align:center;padding:50px'>
        <h1>🚗 Smart Vehicle Dashboard</h1>
        <h3>Prediksi Kendaraan Bermotor Jawa Barat</h3>
        <p>Decision Tree Regressor</p>
    </div>
    """, unsafe_allow_html=True)

    user = st.text_input("Username")
    pw   = st.text_input("Password", type="password")

    if st.button("Login"):
        if user == USERNAME and pw == PASSWORD:
            st.session_state.login = True
            st.rerun()
        else:
            st.error("Username atau Password Salah")

    st.stop()

# ==================================================
# LOAD MODEL
# ==================================================

model   = None
encoder = None

try:
    with open("decision_tree_model.pkl", "rb") as f:
        model = pickle.load(f)

    with open("label_encoder_wilayah.pkl", "rb") as f:
        encoder = pickle.load(f)

except Exception as e:
    st.error("❌ Gagal memuat model")
    st.error(str(e))

# ==================================================
# LOAD DATA TRAINING
# ==================================================

df_training = None

try:
    df_training = pd.read_csv("data_training_final.csv")
except Exception:
    pass

# ==================================================
# FUNCTION CLEAN DATA
# ==================================================

def clean_dataset(df):

    df = df.copy()

    df["nama_kabupaten_kota"] = (
        df["nama_kabupaten_kota"]
        .astype(str)
        .str.strip()
    )

    mapping = {
        "Bandung"          : "KABUPATEN BANDUNG",
        "Bandung Barat"    : "KABUPATEN BANDUNG BARAT",
        "Bekasi"           : "KABUPATEN BEKASI",
        "Bogor"            : "KABUPATEN BOGOR",
        "Ciamis"           : "KABUPATEN CIAMIS",
        "Cianjur"          : "KABUPATEN CIANJUR",
        "Cirebon"          : "KABUPATEN CIREBON",
        "Garut"            : "KABUPATEN GARUT",
        "Indramayu"        : "KABUPATEN INDRAMAYU",
        "Karawang"         : "KABUPATEN KARAWANG",
        "Kuningan"         : "KABUPATEN KUNINGAN",
        "Majalengka"       : "KABUPATEN MAJALENGKA",
        "Pangandaran"      : "KABUPATEN PANGANDARAN",
        "Purwakarta"       : "KABUPATEN PURWAKARTA",
        "Subang"           : "KABUPATEN SUBANG",
        "Sukabumi"         : "KABUPATEN SUKABUMI",
        "Sumedang"         : "KABUPATEN SUMEDANG",
        "Tasikmalaya"      : "KABUPATEN TASIKMALAYA",
        "Kota Bandung"     : "KOTA BANDUNG",
        "Kota Banjar"      : "KOTA BANJAR",
        "Kota Bekasi"      : "KOTA BEKASI",
        "Kota Bogor"       : "KOTA BOGOR",
        "Kota Cimahi"      : "KOTA CIMAHI",
        "Kota Cirebon"     : "KOTA CIREBON",
        "Kota Depok"       : "KOTA DEPOK",
        "Kota Sukabumi"    : "KOTA SUKABUMI",
        "Kota Tasikmalaya" : "KOTA TASIKMALAYA"
    }

    df["nama_kabupaten_kota"] = (
        df["nama_kabupaten_kota"].replace(mapping)
    )

    df = df[df["nama_kabupaten_kota"] != "Jawa Barat"]

    return df

# ==================================================
# FUNCTION GET LAG
# ==================================================

def get_lag_values(df_train, wilayah, kategori_kode):

    subset = df_train[
        (df_train["wilayah"] == wilayah) &
        (df_train["kategori_kendaraan"] == kategori_kode)
    ].sort_values("tahun")

    if len(subset) == 0:
        return 0, 0

    last_row = subset.iloc[-1]
    lag_1    = last_row["jumlah_kendaraan"]
    lag_2    = last_row["lag_1"] if "lag_1" in last_row.index else 0

    return lag_1, lag_2

# ==================================================
# FUNCTION ROLLING FORECAST — FIXED
# ==================================================

def rolling_forecast(
    df_train, model, encoder,
    wilayah, kategori_kode, tahun_target
):
    """
    Rolling forecast dengan koreksi growth rate historis
    agar prediksi berubah tiap tahun.
    """
    wilayah_encoded = encoder.transform([wilayah])[0]

    # Ambil semua data historis wilayah + kategori
    subset = df_train[
        (df_train["wilayah"] == wilayah) &
        (df_train["kategori_kendaraan"] == kategori_kode)
    ].sort_values("tahun")

    if len(subset) == 0:
        return []

    # Hitung rata-rata growth rate dari data historis
    nilai_historis = subset["jumlah_kendaraan"].values

    if len(nilai_historis) >= 2:
        growth_rates = []
        for i in range(1, len(nilai_historis)):
            if nilai_historis[i - 1] > 0:
                g = (
                    (nilai_historis[i] - nilai_historis[i - 1])
                    / nilai_historis[i - 1]
                )
                growth_rates.append(g)
        avg_growth = np.mean(growth_rates) if growth_rates else 0.0
    else:
        avg_growth = 0.0

    # Ambil lag awal dari baris terakhir historis
    last_row      = subset.iloc[-1]
    current_lag_1 = last_row["jumlah_kendaraan"]
    current_lag_2 = (
        last_row["lag_1"]
        if "lag_1" in last_row.index else 0
    )

    tahun_mulai = int(df_train["tahun"].max()) + 1
    hasil_list  = []

    for t in range(tahun_mulai, tahun_target + 1):

        # Growth dari lag rolling
        growth_lag = (
            (current_lag_1 - current_lag_2) / current_lag_2
            if current_lag_2 != 0 else avg_growth
        )

        input_df = pd.DataFrame([{
            "tahun"               : t,
            "kode_wilayah_encoded": wilayah_encoded,
            "kategori_kendaraan"  : kategori_kode,
            "lag_1"               : current_lag_1,
            "lag_2"               : current_lag_2,
            "growth"              : growth_lag
        }])

        pred_model = max(0, model.predict(input_df)[0])

        # Proyeksi berbasis growth historis
        pred_growth = current_lag_1 * (1 + avg_growth)

        # Jika DT stuck (pred == lag_1), pakai growth historis
        if abs(pred_model - current_lag_1) < 1:
            pred_final = pred_growth
        else:
            # Blended: 70% model + 30% growth historis
            pred_final = 0.7 * pred_model + 0.3 * pred_growth

        pred_final = max(0, pred_final)

        hasil_list.append({
            "tahun"           : t,
            "jumlah_kendaraan": int(pred_final)
        })

        # Geser lag untuk iterasi berikutnya
        current_lag_2 = current_lag_1
        current_lag_1 = pred_final

    return hasil_list

# ==================================================
# FUNCTION REKOMENDASI KEBIJAKAN
# ==================================================

def get_rekomendasi(jumlah, wilayah, kategori_label, tahun):

    if df_training is not None:
        avg = df_training[
            df_training["wilayah"] == wilayah
        ]["jumlah_kendaraan"].mean()
    else:
        avg = 500_000

    pct_vs_avg = (
        (jumlah - avg) / avg * 100
    ) if avg > 0 else 0

    if jumlah > 1_500_000:
        level   = "🔴 KRITIS"
        warna   = "error"
        ringkas = "Kepadatan kendaraan sangat tinggi, diperlukan tindakan segera."
    elif jumlah > 800_000:
        level   = "🟠 PRIORITAS TINGGI"
        warna   = "warning"
        ringkas = "Pertumbuhan di atas rata-rata, perlu antisipasi dini."
    elif jumlah > 400_000:
        level   = "🟡 BEBAN SEDANG"
        warna   = "warning"
        ringkas = "Pertumbuhan terkendali, pantau secara berkala."
    else:
        level   = "🟢 BEBAN NORMAL"
        warna   = "success"
        ringkas = "Kondisi ideal, fokus pada pemeliharaan."

    if kategori_label == "Sepeda Motor":
        rekomendasi = [
            "🛣️  **Infrastruktur**: Perluas jalur motor dan parkir terstruktur.",
            "🌿  **Lingkungan**: Dorong peralihan ke motor listrik (subsidi KBLBB).",
            "💰  **Fiskal**: Optimalkan penerimaan PKB sepeda motor.",
            "🚦  **Lalu Lintas**: Tambah rambu & rekayasa lalu lintas di titik padat.",
        ]
    elif kategori_label == "Mobil Penumpang":
        rekomendasi = [
            "🛣️  **Infrastruktur**: Perlebar ruas jalan utama & flyover.",
            "🚌  **Transportasi Publik**: Kurangi ketergantungan dengan perkuat angkot/BRT.",
            "🅿️  **Parkir**: Bangun gedung parkir terpadu di pusat kota.",
            "💰  **Fiskal**: Proyeksikan pendapatan BBN-KB & PKB untuk APBD.",
        ]
    elif kategori_label == "Mobil Bus":
        rekomendasi = [
            "🛣️  **Infrastruktur**: Sediakan jalur khusus bus (busway) & terminal.",
            "🚌  **Integrasi**: Integrasikan rute antar kab/kota.",
            "🌿  **Lingkungan**: Percepat adopsi bus listrik untuk trayek utama.",
            "📋  **Regulasi**: Perbarui izin trayek & standar emisi armada.",
        ]
    else:
        rekomendasi = [
            "🛣️  **Infrastruktur**: Perkuat jalan kelas III untuk distribusi logistik.",
            "🏭  **Logistik**: Bangun dry port / pergudangan terpadu.",
            "⚖️  **Regulasi**: Perketat pengawasan batas tonase kendaraan barang.",
            "🌿  **Lingkungan**: Batasi operasional kendaraan tua berpolusi tinggi.",
        ]

    return level, warna, ringkas, rekomendasi, pct_vs_avg

# ==================================================
# SIDEBAR
# ==================================================

st.sidebar.title("🚗 Smart Vehicle")

menu = st.sidebar.radio(
    "Menu",
    ["Dashboard", "Upload CSV", "Prediksi"]
)

if st.sidebar.button("Logout"):
    st.session_state.login = False
    st.rerun()

# ==================================================
# DASHBOARD
# ==================================================

if menu == "Dashboard":

    st.title("🚗 Dashboard Kendaraan Jawa Barat")

    k1, k2, k3, k4 = st.columns(4)
    k1.metric("Provinsi",  "Jawa Barat")
    k2.metric("Periode",   "2016-2025")
    k3.metric("Model",     "Decision Tree")
    k4.metric("R² Score",  "97.6%")

    st.markdown("---")

    if "df" not in st.session_state:
        st.info("Upload dataset terlebih dahulu.")

    else:

        df = st.session_state["df"]

        st.subheader("📊 Ringkasan Dataset")

        c1, c2, c3, c4 = st.columns(4)
        c1.metric("Jumlah Data",    len(df))
        c2.metric("Jumlah Fitur",   len(df.columns))
        c3.metric(
            "Kabupaten/Kota",
            df["nama_kabupaten_kota"].nunique()
        )
        c4.metric(
            "Periode",
            f"{df['tahun'].min()} - {df['tahun'].max()}"
        )

        df["total_kendaraan"] = (
            df["mobil_penumpang"].fillna(0)
            + df["sepeda_motor"].fillna(0)
            + df["mobil_bus"].fillna(0)
            + df["mobil_barang"].fillna(0)
        )

        st.markdown("---")
        st.subheader("📈 Tren Kendaraan per Tahun")

        tren = (
            df.groupby("tahun")[[
                "mobil_penumpang", "sepeda_motor",
                "mobil_bus", "mobil_barang"
            ]]
            .sum()
            .reset_index()
        )

        fig1 = px.line(
            tren,
            x="tahun",
            y=[
                "mobil_penumpang", "sepeda_motor",
                "mobil_bus", "mobil_barang"
            ],
            markers=True
        )
        st.plotly_chart(fig1, use_container_width=True)

        col1, col2 = st.columns(2)

        with col1:
            st.subheader("🥧 Komposisi Kendaraan")
            pie_df = pd.DataFrame({
                "Jenis"  : [
                    "Mobil Penumpang", "Sepeda Motor",
                    "Mobil Bus", "Mobil Barang"
                ],
                "Jumlah" : [
                    df["mobil_penumpang"].sum(),
                    df["sepeda_motor"].sum(),
                    df["mobil_bus"].sum(),
                    df["mobil_barang"].sum()
                ]
            })
            fig2 = px.pie(
                pie_df,
                names="Jenis",
                values="Jumlah",
                hole=0.4
            )
            st.plotly_chart(fig2, use_container_width=True)

        with col2:
            st.subheader("🏆 Top 10 Wilayah (Sepeda Motor)")
            top10 = (
                df.groupby("nama_kabupaten_kota")["sepeda_motor"]
                .sum().reset_index()
                .sort_values(by="sepeda_motor", ascending=False)
                .head(10)
            )
            fig3 = px.bar(
                top10,
                x="nama_kabupaten_kota",
                y="sepeda_motor",
                color="sepeda_motor"
            )
            st.plotly_chart(fig3, use_container_width=True)

        st.markdown("---")
        st.subheader("🚗 Top 10 Total Kendaraan")

        total10 = (
            df.groupby("nama_kabupaten_kota")["total_kendaraan"]
            .sum().reset_index()
            .sort_values(by="total_kendaraan", ascending=False)
            .head(10)
        )
        fig4 = px.bar(
            total10,
            x="nama_kabupaten_kota",
            y="total_kendaraan",
            color="total_kendaraan"
        )
        st.plotly_chart(fig4, use_container_width=True)

        st.markdown("---")
        st.subheader("📈 Pertumbuhan Kendaraan Tahunan (%)")

        growth_hist = (
            tren.set_index("tahun").sum(axis=1).pct_change() * 100
        )
        growth_df = pd.DataFrame({
            "tahun"       : growth_hist.index,
            "pertumbuhan" : growth_hist.values
        })
        fig5 = px.line(
            growth_df,
            x="tahun",
            y="pertumbuhan",
            markers=True
        )
        st.plotly_chart(fig5, use_container_width=True)

# ==================================================
# UPLOAD CSV
# ==================================================

elif menu == "Upload CSV":

    st.title("📂 Upload Dataset")

    uploaded_file = st.file_uploader("Upload CSV", type=["csv"])

    if uploaded_file:

        df = pd.read_csv(uploaded_file)
        df = clean_dataset(df)

        st.session_state["df"] = df

        st.success("Dataset berhasil diupload")
        st.subheader("Preview Dataset")
        st.dataframe(df.head())

        c1, c2, c3 = st.columns(3)
        c1.metric("Jumlah Data",    len(df))
        c2.metric("Jumlah Fitur",   len(df.columns))
        c3.metric(
            "Kabupaten/Kota",
            df["nama_kabupaten_kota"].nunique()
        )

        st.subheader("Informasi Dataset")
        info = pd.DataFrame({
            "Kolom"     : df.columns,
            "Tipe Data" : df.dtypes.astype(str)
        })
        st.dataframe(info)

# ==================================================
# PREDIKSI
# ==================================================

elif menu == "Prediksi":

    st.title("🔮 Prediksi Jumlah Kendaraan")

    if model is None or encoder is None:
        st.warning(
            "Pastikan file **decision_tree_model.pkl** "
            "dan **label_encoder_wilayah.pkl** tersedia."
        )
        st.stop()

    if df_training is None:
        st.warning(
            "File **data_training_final.csv** tidak ditemukan. "
            "Jalankan **train_model.py** terlebih dahulu."
        )
        st.stop()

    kategori_map = {
        "Mobil Penumpang" : 0,
        "Sepeda Motor"    : 1,
        "Mobil Bus"       : 2,
        "Mobil Barang"    : 3
    }

    tab1, tab2 = st.tabs([
        "📅 Prediksi Satu Tahun",
        "📈 Prediksi Multi Tahun"
    ])

    # ------------------------------------------------
    # TAB 1 — PREDIKSI SATU TAHUN + REKOMENDASI
    # ------------------------------------------------

    with tab1:

        st.subheader("Prediksi & Analisis Kebijakan")

        col_a, col_b, col_c = st.columns(3)

        with col_a:
            wilayah = st.selectbox(
                "Kabupaten/Kota",
                encoder.classes_,
                key="s_wilayah"
            )

        with col_b:
            kategori = st.selectbox(
                "Kategori Kendaraan",
                list(kategori_map.keys()),
                key="s_kategori"
            )

        with col_c:
            tahun_min = int(df_training["tahun"].max()) + 1
            tahun_target = st.slider(
                "Tahun Prediksi",
                tahun_min,
                2035,
                tahun_min,
                key="s_tahun"
            )

        if st.button("🔍 Analisis Sekarang", key="btn_single"):

            kategori_kode = kategori_map[kategori]

            with st.spinner("Menghitung prediksi..."):
                hasil_list = rolling_forecast(
                    df_training, model, encoder,
                    wilayah, kategori_kode, tahun_target
                )

            hasil_tahun = next(
                (h for h in hasil_list if h["tahun"] == tahun_target),
                None
            )

            if hasil_tahun is None:
                st.error("Gagal menghitung prediksi.")
            else:
                jumlah = hasil_tahun["jumlah_kendaraan"]

                level, warna, ringkas, rekomendasi, pct = get_rekomendasi(
                    jumlah, wilayah, kategori, tahun_target
                )

                st.markdown("---")

                # ── Metrik utama ──
                m1, m2, m3 = st.columns(3)

                m1.metric(
                    label=f"Prediksi {kategori} ({tahun_target})",
                    value=f"{jumlah:,} unit"
                )

                m2.metric(
                    label="vs Rata-rata Historis Wilayah",
                    value=f"{pct:+.1f}%"
                )

                # Pertumbuhan YoY
                if len(hasil_list) >= 2:
                    prev       = hasil_list[-2]["jumlah_kendaraan"]
                    growth_yoy = (
                        (jumlah - prev) / prev * 100
                    ) if prev > 0 else 0
                else:
                    lag_1, _   = get_lag_values(
                        df_training, wilayah, kategori_kode
                    )
                    growth_yoy = (
                        (jumlah - lag_1) / lag_1 * 100
                    ) if lag_1 > 0 else 0

                m3.metric(
                    label=f"Pertumbuhan vs {tahun_target - 1}",
                    value=f"{growth_yoy:+.1f}%"
                )

                # ── Status ──
                st.markdown("---")

                if warna == "error":
                    st.error(f"**Status: {level}** — {ringkas}")
                elif warna == "warning":
                    st.warning(f"**Status: {level}** — {ringkas}")
                else:
                    st.success(f"**Status: {level}** — {ringkas}")

                # ── Rekomendasi ──
                st.markdown("#### 📋 Rekomendasi Kebijakan")
                for r in rekomendasi:
                    st.markdown(f"- {r}")

                # ── Grafik historis + semua titik prediksi ──
                st.markdown("---")
                st.markdown("#### 📊 Tren Historis + Prediksi")

                hist = df_training[
                    (df_training["wilayah"] == wilayah) &
                    (df_training["kategori_kendaraan"] == kategori_kode)
                ][["tahun", "jumlah_kendaraan"]].copy()

                hist["Tipe"]   = "Historis"
                hist.columns   = ["Tahun", "Jumlah", "Tipe"]

                pred_df        = pd.DataFrame(hasil_list)
                pred_df.columns = ["Tahun", "Jumlah"]
                pred_df["Tipe"] = "Prediksi"

                df_plot = pd.concat(
                    [hist, pred_df],
                    ignore_index=True
                )

                fig = px.line(
                    df_plot,
                    x="Tahun",
                    y="Jumlah",
                    color="Tipe",
                    markers=True,
                    title=f"{kategori} — {wilayah}",
                    color_discrete_map={
                        "Historis" : "#1f77b4",
                        "Prediksi" : "#ff7f0e"
                    }
                )

                # Highlight titik tahun yang dipilih
                fig.add_scatter(
                    x=[tahun_target],
                    y=[jumlah],
                    mode="markers",
                    marker=dict(
                        size=14,
                        color="red",
                        symbol="star"
                    ),
                    name=f"Tahun {tahun_target}"
                )

                st.plotly_chart(fig, use_container_width=True)

                # ── Tabel semua tahun prediksi ──
                st.markdown("#### 📋 Tabel Prediksi Per Tahun")

                rows = []
                prev_val = (
                    df_training[
                        (df_training["wilayah"] == wilayah) &
                        (df_training["kategori_kendaraan"] == kategori_kode)
                    ].sort_values("tahun")
                    .iloc[-1]["jumlah_kendaraan"]
                )

                for h in hasil_list:
                    g = (
                        (h["jumlah_kendaraan"] - prev_val)
                        / prev_val * 100
                    ) if prev_val > 0 else 0

                    lvl, _, ringkas_t, _, _ = get_rekomendasi(
                        h["jumlah_kendaraan"],
                        wilayah, kategori,
                        h["tahun"]
                    )

                    rows.append({
                        "Tahun"           : h["tahun"],
                        "Prediksi (unit)" : f"{h['jumlah_kendaraan']:,}",
                        "Pertumbuhan YoY" : f"{g:+.1f}%",
                        "Status"          : lvl,
                    })
                    prev_val = h["jumlah_kendaraan"]

                st.dataframe(
                    pd.DataFrame(rows).set_index("Tahun"),
                    use_container_width=True
                )

    # ------------------------------------------------
    # TAB 2 — PREDIKSI MULTI TAHUN
    # ------------------------------------------------

    with tab2:

        st.subheader("Tren Prediksi Beberapa Tahun ke Depan")

        col_a, col_b, col_c = st.columns(3)

        with col_a:
            wilayah_m = st.selectbox(
                "Kabupaten/Kota",
                encoder.classes_,
                key="m_wilayah"
            )

        with col_b:
            kategori_m = st.selectbox(
                "Kategori Kendaraan",
                list(kategori_map.keys()),
                key="m_kategori"
            )

        with col_c:
            n_years = st.slider(
                "Jumlah Tahun",
                1, 10, 5,
                key="m_nyears"
            )

        if st.button("📈 Lihat Tren", key="btn_multi"):

            kategori_kode_m = kategori_map[kategori_m]
            tahun_akhir_m   = (
                int(df_training["tahun"].max()) + n_years
            )

            with st.spinner("Menghitung prediksi..."):
                hasil_list_m = rolling_forecast(
                    df_training, model, encoder,
                    wilayah_m, kategori_kode_m, tahun_akhir_m
                )

            # Gabung historis + prediksi
            hist_m = df_training[
                (df_training["wilayah"] == wilayah_m) &
                (df_training["kategori_kendaraan"] == kategori_kode_m)
            ][["tahun", "jumlah_kendaraan"]].copy()

            hist_m["Tipe"] = "Historis"
            hist_m.columns = ["Tahun", "Jumlah", "Tipe"]

            pred_m          = pd.DataFrame(hasil_list_m)
            pred_m.columns  = ["Tahun", "Jumlah"]
            pred_m["Tipe"]  = "Prediksi"

            df_plot_m = pd.concat(
                [hist_m, pred_m],
                ignore_index=True
            )

            fig_m = px.line(
                df_plot_m,
                x="Tahun",
                y="Jumlah",
                color="Tipe",
                markers=True,
                title=f"Tren {kategori_m} — {wilayah_m}",
                color_discrete_map={
                    "Historis" : "#1f77b4",
                    "Prediksi" : "#ff7f0e"
                }
            )
            st.plotly_chart(fig_m, use_container_width=True)

            # ── Tabel detail per tahun ──
            st.subheader("📋 Detail Prediksi Per Tahun")

            rows_m   = []
            prev_val = (
                df_training[
                    (df_training["wilayah"] == wilayah_m) &
                    (df_training["kategori_kendaraan"] == kategori_kode_m)
                ].sort_values("tahun")
                .iloc[-1]["jumlah_kendaraan"]
            )

            for h in hasil_list_m:
                g = (
                    (h["jumlah_kendaraan"] - prev_val)
                    / prev_val * 100
                ) if prev_val > 0 else 0

                lvl, _, ringkas_m, _, _ = get_rekomendasi(
                    h["jumlah_kendaraan"],
                    wilayah_m, kategori_m,
                    h["tahun"]
                )

                rows_m.append({
                    "Tahun"           : h["tahun"],
                    "Prediksi (unit)" : f"{h['jumlah_kendaraan']:,}",
                    "Pertumbuhan YoY" : f"{g:+.1f}%",
                    "Status"          : lvl,
                    "Keterangan"      : ringkas_m
                })
                prev_val = h["jumlah_kendaraan"]

            st.dataframe(
                pd.DataFrame(rows_m).set_index("Tahun"),
                use_container_width=True
            )