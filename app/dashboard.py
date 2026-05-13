#!/usr/bin/env python3
"""
Phase 5: Streamlit Dashboard
==============================
Interactive dashboard for Istanbul traffic density predictions.

Features:
    - Road selection, date/time pickers
    - Map-first design, professional look (white background, dark text)
    - 7-day prediction horizon constraint with visual warnings
"""

import streamlit as st
import requests
import folium
from streamlit_folium import folium_static
from datetime import date, datetime, timedelta

# ---------------------------------------------------------------------------
# Page Config
# ---------------------------------------------------------------------------
st.set_page_config(
    page_title="İstanbul Trafik Yoğunluğu Tahmini",
    layout="wide",
    initial_sidebar_state="expanded",
)

# ---------------------------------------------------------------------------
# Custom CSS for professional look
# ---------------------------------------------------------------------------
st.markdown("""
<style>
    /* Clean white background */
    .stApp {
        background-color: #ffffff;
        color: #333333;
        font-family: 'Inter', sans-serif;
    }

    /* Sidebar styling */
    [data-testid="stSidebar"] {
        background-color: #f8f9fa;
        border-right: 1px solid #e0e0e0;
    }

    [data-testid="stSidebar"] .stMarkdown h1,
    [data-testid="stSidebar"] .stMarkdown h2,
    [data-testid="stSidebar"] .stMarkdown h3 {
        color: #333333;
    }

    /* Headers */
    h1, h2, h3 {
        color: #2c3e50 !important;
    }

    /* Buttons */
    .stButton > button {
        background-color: #3498db;
        color: white;
        border: none;
        border-radius: 4px;
        padding: 8px 16px;
        font-weight: 500;
        transition: background-color 0.2s;
        width: 100%;
    }

    .stButton > button:hover {
        background-color: #2980b9;
        color: white;
    }

    /* Metric cards below map */
    .metric-card {
        background-color: #ffffff;
        border: 1px solid #e0e0e0;
        border-radius: 8px;
        padding: 16px;
        text-align: center;
        box-shadow: 0 2px 4px rgba(0,0,0,0.05);
    }
    .metric-card h3 {
        margin: 0;
        font-size: 14px;
        color: #7f8c8d !important;
        font-weight: 500;
    }
    .metric-card p {
        margin: 8px 0 0 0;
        font-size: 24px;
        font-weight: 700;
        color: #2c3e50;
    }

    /* Status badges */
    .status-badge {
        padding: 4px 12px;
        border-radius: 12px;
        font-weight: 600;
        font-size: 12px;
        color: white;
    }
    .status-akici { background-color: #2ecc71; }
    .status-yogun { background-color: #f39c12; color: #333; }
    .status-kilit { background-color: #e74c3c; }

    /* Warnings */
    .warning-box {
        background-color: #fdeaea;
        border: 1px solid #e74c3c;
        border-radius: 4px;
        padding: 12px;
        color: #c0392b;
        font-weight: 500;
        margin-bottom: 16px;
    }

    .info-box {
        background-color: #e8f4f8;
        border: 1px solid #3498db;
        border-radius: 4px;
        padding: 12px;
        color: #2980b9;
        font-size: 14px;
        margin-top: 24px;
    }
</style>
""", unsafe_allow_html=True)

# ---------------------------------------------------------------------------
# Constants
# ---------------------------------------------------------------------------
API_URL = "http://localhost:8000"

ROAD_DISPLAY_NAMES = {
    "TEM": "TEM / O-2 Otoyolu",
    "Buyukdere_Cad": "Büyükdere Caddesi",
    "Sahil_Yolu_Avrupa": "Sahil Yolu (Avrupa)",
}

ROAD_API_NAMES = {v: k for k, v in ROAD_DISPLAY_NAMES.items()}

STATUS_COLORS = {
    "Akici": "#2ecc71",
    "Yogun": "#f39c12",
    "Kilit": "#e74c3c",
}

STATUS_LABELS = {
    "Akici": "Akıcı",
    "Yogun": "Yoğun",
    "Kilit": "Kilit",
}


# ---------------------------------------------------------------------------
# Sidebar
# ---------------------------------------------------------------------------
with st.sidebar:
    st.markdown("## İstanbul Trafik Tahmini")
    st.markdown("---")

    road_display = st.selectbox(
        "Yol Seçimi",
        options=list(ROAD_DISPLAY_NAMES.values()),
    )
    selected_road = ROAD_API_NAMES[road_display]

    today = date.today()
    max_future_date = today + timedelta(days=7)

    selected_date = st.date_input(
        "Tarih",
        value=today,
        min_value=date(2023, 1, 1),
    )

    hours = [f"{h:02d}:00" for h in range(24)]
    selected_time = st.selectbox("Saat", options=hours, index=8)

    st.markdown("---")
    predict_button = st.button("Tahmin Et")

    st.markdown("""
    <div class="info-box">
        <b>Bilgi:</b><br>
        Tahmin ufku: &plusmn;7 gün<br>
        Model maksimum 7 gün ileriye dönük tahmin yapabilir.
    </div>
    """, unsafe_allow_html=True)

# ---------------------------------------------------------------------------
# Main Content
# ---------------------------------------------------------------------------
st.markdown("## Trafik Yoğunluğu Tahmin Haritası")

is_date_invalid = selected_date > max_future_date

if is_date_invalid:
    st.markdown("""
    <div class="warning-box">
        Model maksimum 7 gün ileri tahmin yapabilir. Lütfen daha erken bir tarih seçiniz.
    </div>
    """, unsafe_allow_html=True)

elif predict_button:
    with st.spinner("Tahmin hesaplanıyor..."):
        try:
            response = requests.post(
                f"{API_URL}/predict",
                json={
                    "road_name": selected_road,
                    "date": selected_date.strftime("%Y-%m-%d"),
                    "time": selected_time,
                },
                timeout=30,
            )
            response.raise_for_status()
            data = response.json()

        except requests.exceptions.HTTPError as e:
            error_msg = e.response.json().get("detail", e.response.text) if hasattr(e.response, 'json') else e.response.text
            st.error(f"API Hatası: {error_msg}")
            st.stop()
        except Exception as e:
            st.error(f"Bağlantı Hatası: Sunucuya ulaşılamadı. ({e})")
            st.stop()

    predictions = data["predictions"]
    total = len(predictions)
    
    if total > 0:
        # Full screen map calculation
        avg_lat = sum(p["latitude"] for p in predictions) / total
        avg_lon = sum(p["longitude"] for p in predictions) / total

        m = folium.Map(
            location=[avg_lat, avg_lon],
            zoom_start=12,
            tiles="CartoDB positron",  # Light theme map
        )

        for pred in predictions:
            color = STATUS_COLORS.get(pred["traffic_status"], "gray")
            status_label = STATUS_LABELS.get(pred["traffic_status"], pred["traffic_status"])

            tooltip_html = f"""
            <div style="font-family: Inter, sans-serif; padding: 4px;">
                <b style="color: {color};">{status_label}</b><br>
                <b>Geohash:</b> {pred['geohash']}<br>
                <b>Yoğunluk Skoru:</b> %{pred['congestion_score']*100:.0f}
            </div>
            """

            folium.CircleMarker(
                location=[pred["latitude"], pred["longitude"]],
                radius=8,
                color=color,
                fill=True,
                fill_color=color,
                fill_opacity=0.8,
                weight=1,
                tooltip=folium.Tooltip(tooltip_html),
            ).add_to(m)

        # Render full-width map
        folium_static(m, width=1200, height=500)

        # Metric cards below map
        st.markdown(f"### {road_display} — {selected_date} {selected_time}")
        
        akici_count = sum(1 for p in predictions if p["traffic_status"] == "Akici")
        yogun_count = sum(1 for p in predictions if p["traffic_status"] == "Yogun")
        kilit_count = sum(1 for p in predictions if p["traffic_status"] == "Kilit")
        avg_congestion = sum(p["congestion_score"] for p in predictions) / total

        col1, col2, col3, col4, col5 = st.columns(5)

        with col1:
            st.markdown(f"""
            <div class="metric-card">
                <h3>Toplam Bölge</h3>
                <p>{total}</p>
            </div>
            """, unsafe_allow_html=True)
            
        with col2:
            st.markdown(f"""
            <div class="metric-card">
                <h3>Ortalama Yoğunluk</h3>
                <p>%{avg_congestion*100:.0f}</p>
            </div>
            """, unsafe_allow_html=True)

        with col3:
            st.markdown(f"""
            <div class="metric-card" style="border-bottom: 4px solid {STATUS_COLORS['Akici']};">
                <h3>Akıcı</h3>
                <p>{akici_count}</p>
            </div>
            """, unsafe_allow_html=True)

        with col4:
            st.markdown(f"""
            <div class="metric-card" style="border-bottom: 4px solid {STATUS_COLORS['Yogun']};">
                <h3>Yoğun</h3>
                <p>{yogun_count}</p>
            </div>
            """, unsafe_allow_html=True)

        with col5:
            st.markdown(f"""
            <div class="metric-card" style="border-bottom: 4px solid {STATUS_COLORS['Kilit']};">
                <h3>Kilit</h3>
                <p>{kilit_count}</p>
            </div>
            """, unsafe_allow_html=True)

else:
    st.info("Tahmin sonuçlarını görmek için sol menüden parametreleri belirleyip 'Tahmin Et' butonuna tıklayınız.")
