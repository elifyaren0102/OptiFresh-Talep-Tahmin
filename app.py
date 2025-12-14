import streamlit as st
import pandas as pd
from prophet import Prophet
from prophet.plot import plot_plotly
import plotly.graph_objs as go

# --- 1. Sayfa Ayarları (En Üstte Olmalı) ---
st.set_page_config(page_title="OptiFresh - Keskinoğlu", layout="wide")

st.title("🐔 OptiFresh: Akıllı Talep Tahmin Sistemi")
st.markdown("**Keskinoğlu Operasyon Paneli** | *Yapay Zeka Destekli Satış Planlama*")

# --- 2. Veriyi Yükleme ---
@st.cache_data
def load_data():
    # CSV dosyasını okuyoruz
    df = pd.read_csv('keskinoglu_satis_verisi.csv')
    df['Tarih'] = pd.to_datetime(df['Tarih'])
    return df

try:
    df = load_data()
except FileNotFoundError:
    st.error("HATA: 'keskinoglu_satis_verisi.csv' dosyası bulunamadı. Lütfen dosyanın aynı klasörde olduğundan emin olun.")
    st.stop()

# --- 3. Sidebar (Kontrol Paneli) ---
st.sidebar.header("⚙️ Senaryo Parametreleri")

period = st.sidebar.slider("Tahmin Periyodu (Gün)", 7, 90, 30)
sicaklik_farki = st.sidebar.slider("Sıcaklık Senaryosu (°C Değişimi)", -5, +10, 0)

if sicaklik_farki != 0:
    st.sidebar.warning(f"⚠️ DİKKAT: Mevsim normallerinden {sicaklik_farki}°C farklı bir senaryo simüle ediliyor.")
else:
    st.sidebar.success("✅ Şu an mevsim normalleri (Gerçek Beklenti) kullanılıyor.")

# --- 4. Model Eğitimi ---
# Sütun isimlerini Prophet'in istediği formata çeviriyoruz
df_prophet = df.rename(columns={'Tarih': 'ds', 'Satis_Miktari_Adet': 'y'})

model = Prophet(daily_seasonality=True, yearly_seasonality=True)
model.add_regressor('Ort_Sicaklik')
model.add_regressor('Birim_Fiyat_TL')
model.add_regressor('Hafta_Sonu')

with st.spinner('Yapay Zeka modeli eğitiliyor... Lütfen bekleyin.'):
    model.fit(df_prophet)

# --- 5. Gelecek Tahmini (GARANTİLİ YÖNTEM) ---
future = model.make_future_dataframe(periods=period)

# TARİH AYIKLAMA
future['Ay'] = future['ds'].dt.month
future['Haftanin_Gunu'] = future['ds'].dt.dayofweek
future['Hafta_Sonu'] = future['Haftanin_Gunu'].apply(lambda x: 1 if x >= 5 else 0)

# MANİSA / AKHİSAR İÇİN ORTALAMA SICAKLIK SÖZLÜĞÜ (GARANTİ YÖNTEM)
# Buraya hatalı formül yerine gerçek değerleri elle giriyoruz. Asla şaşmaz.
manisa_iklimi = {
    1: 6,   2: 8,   3: 11,  4: 16,
    5: 21,  6: 26,  7: 29,  8: 29,
    9: 25,  10: 19, 11: 13, 12: 8
}

# 1. Adım: O ayın normal sıcaklığını sözlükten çek
future['Ort_Sicaklik'] = future['Ay'].map(manisa_iklimi)

# 2. Adım: Senin Slider ile seçtiğin farkı ekle
future['Ort_Sicaklik'] = future['Ort_Sicaklik'] + sicaklik_farki

# Fiyatı son güncel fiyat olarak alıyoruz
future['Birim_Fiyat_TL'] = df['Birim_Fiyat_TL'].iloc[-1]

# Tahmin Yap
forecast = model.predict(future)

# --- 6. KPI Kartları ---
col1, col2, col3 = st.columns(3)

# Gelecek dönemin toplam satışı
gelecek_satis_toplam = int(forecast.tail(period)['yhat'].sum())

# Gösterilen dönemin ortalama sıcaklığı
gosterilen_sicaklik = round(future.tail(period)['Ort_Sicaklik'].mean(), 1)

col1.metric("Tahmini Toplam Satış", f"{gelecek_satis_toplam:,} Adet")
col2.metric("Tahmin Periyodu", f"{period} Gün")
col3.metric("Ortalama Hava Sıcaklığı", f"{gosterilen_sicaklik} °C", delta=f"{sicaklik_farki}°C Fark")

# --- 7. Ana Grafik (GÖRSELLEŞTİRİLMİŞ PROFESYONEL VERSİYON) ---
st.subheader("📈 Gelecek Dönem Satış Trendi")

# 1. Temel Grafiği Oluştur
fig_main = plot_plotly(model, forecast)

# Son gerçek verinin olduğu tarihi bulalım (Bugün kabul ediyoruz)
son_gercek_tarih = df['Tarih'].max()

# 2. Profesyonel Ayarlar
fig_main.update_layout(
    # Başlık
    title=dict(
        text="<b>Günlük Satış Tahmini ve Gerçekleşen Veriler</b>",
        font=dict(size=20, color='#2c3e50'),
        x=0.01,
        y=0.95
    ),
    # Renkler
    plot_bgcolor='white',
    paper_bgcolor='white',
    hovermode="x unified",

    # Eksen Ayarları (Gri Çubuk Gitti)
    xaxis=dict(
        title="Tarih",
        title_font=dict(size=14, color='gray'),
        showgrid=True,
        gridcolor='#f0f2f6',
        rangeslider=dict(visible=False),  # Alt çubuğu kapat
        type="date"
    ),

    yaxis=dict(
        title="Satış Miktarı (Adet)",
        title_font=dict(size=14, color='gray'),
        showgrid=True,
        gridcolor='#f0f2f6',
    ),

    # Zoom Butonları (GÜNCELLENDİ: ARTIK GERÇEKLERİ DE GÖSTERİYOR)
    updatemenus=[
        dict(
            type="buttons",
            direction="right",
            x=0.85, y=1.15,
            showactive=True,
            buttons=list([
                # 1 Hafta: Son 1 hafta gerçek + 1 hafta tahmin
                dict(label="1 Hafta", method="relayout", args=[{"xaxis.range": [
                    son_gercek_tarih - pd.Timedelta(weeks=1),
                    son_gercek_tarih + pd.Timedelta(weeks=1)
                ]}]),

                # 1 Ay: Son 1 ay gerçek + 1 ay tahmin (En ideal görünüm)
                dict(label="1 Ay", method="relayout", args=[{"xaxis.range": [
                    son_gercek_tarih - pd.Timedelta(days=30),
                    future['ds'].max()
                ]}]),

                # 3 Ay: Daha geniş geçmiş
                dict(label="3 Ay", method="relayout", args=[{"xaxis.range": [
                    son_gercek_tarih - pd.Timedelta(days=90),
                    future['ds'].max()
                ]}]),

                # Tümü: Her şeyi göster
                dict(label="Tümü", method="relayout", args=[{"xaxis.autorange": True}]),
            ]),
        )
    ]
)

# 3. Nokta ve Çizgi Renklerini Düzenle
# Gerçek verileri (Noktalar) belirgin kırmızı yapıyoruz
fig_main.update_traces(marker=dict(color='#D90429', size=5, opacity=0.8), selector=dict(mode='markers'))

# Tahmin çizgisini (Mavi) düzenliyoruz
fig_main.update_traces(line=dict(color='#007bff', width=3), selector=dict(mode='lines'))

# 4. Grafiği Ekrana Bas
st.plotly_chart(fig_main, use_container_width=True)

# --- 8. Detaylı Tablo (TÜRKÇELEŞTİRİLMİŞ VE DÜZELTİLMİŞ) ---
st.subheader("📋 Günlük Tahmin Detayları")

# Sadece son günleri al
tablo_verisi = forecast[['ds', 'yhat', 'yhat_lower', 'yhat_upper']].tail(period).copy()

# Sıcaklığı da future tablosundan çekip yanına ekleyelim ki tabloda görünsün
tablo_verisi['Ort_Sicaklik'] = future['Ort_Sicaklik'].tail(period).values

# İsimleri Türkçeleştir
tablo_verisi = tablo_verisi.rename(columns={
    'ds': 'Tarih',
    'yhat': 'Tahmin (Adet)',
    'yhat_lower': 'Min. Beklenti',
    'yhat_upper': 'Maks. Beklenti',
    'Ort_Sicaklik': 'Hava (°C)'
})

# Tarih formatını sadece YIL-AY-GÜN yap
tablo_verisi['Tarih'] = tablo_verisi['Tarih'].dt.date

# Sayıları yuvarla (Virgülden kurtul)
tablo_verisi['Tahmin (Adet)'] = tablo_verisi['Tahmin (Adet)'].round(0).astype(int)
tablo_verisi['Min. Beklenti'] = tablo_verisi['Min. Beklenti'].round(0).astype(int)
tablo_verisi['Maks. Beklenti'] = tablo_verisi['Maks. Beklenti'].round(0).astype(int)
tablo_verisi['Hava (°C)'] = tablo_verisi['Hava (°C)'].round(1)

# Tabloyu göster
st.dataframe(tablo_verisi, use_container_width=True, hide_index=True)