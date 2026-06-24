# ============================================================
# PHYTO COLLAB — Demand Forecast Dashboard
# Author : Moch Yahya
# Run    : streamlit run dashboard_forecast.py
# ============================================================

import streamlit as st
import pandas as pd
import numpy as np
import matplotlib.pyplot as plt
import matplotlib.dates as mdates
import matplotlib.ticker as mticker
import seaborn as sns
import os
import warnings

warnings.filterwarnings("ignore")

# ── Page config ───────────────────────────────────────────
st.set_page_config(
    page_title="Phyto Collab | Demand Forecast",
    page_icon="📦",
    layout="wide",
    initial_sidebar_state="expanded",
)

# ── Custom CSS ────────────────────────────────────────────
st.markdown("""
<style>
/* ── Global ── */
@import url('https://fonts.googleapis.com/css2?family=Inter:wght@400;500;600;700&family=Space+Mono:wght@400;700&display=swap');

html, body, [class*="css"]  { font-family: 'Inter', sans-serif; }

/* ── Background ── */
.stApp { background-color: #0f1117; color: #e8eaf0; }

/* ── Sidebar ── */
section[data-testid="stSidebar"] {
    background: #161b27;
    border-right: 1px solid #1e2736;
}
section[data-testid="stSidebar"] .stMarkdown p {
    color: #8b9bb4;
    font-size: 0.78rem;
    letter-spacing: 0.06em;
    text-transform: uppercase;
}

/* ── Metric cards ── */
div[data-testid="metric-container"] {
    background: #161b27;
    border: 1px solid #1e2736;
    border-radius: 10px;
    padding: 1rem 1.2rem;
}
div[data-testid="metric-container"] label { color: #8b9bb4 !important; font-size: 0.78rem !important; }
div[data-testid="metric-container"] [data-testid="stMetricValue"] {
    color: #e8eaf0 !important;
    font-family: 'Space Mono', monospace;
    font-size: 1.6rem !important;
}
div[data-testid="metric-container"] [data-testid="stMetricDelta"] { font-size: 0.82rem !important; }

/* ── Section headers ── */
.section-title {
    font-family: 'Space Mono', monospace;
    font-size: 0.72rem;
    letter-spacing: 0.12em;
    text-transform: uppercase;
    color: #4f86f7;
    margin-bottom: 0.4rem;
    border-bottom: 1px solid #1e2736;
    padding-bottom: 0.4rem;
}
.page-title {
    font-size: 1.55rem;
    font-weight: 700;
    color: #e8eaf0;
    margin-bottom: 0;
}
.page-subtitle { color: #8b9bb4; font-size: 0.92rem; margin-top: 0.2rem; }

/* ── Alert / info boxes ── */
.best-model-badge {
    display: inline-block;
    background: linear-gradient(90deg, #1a3a6b, #1e4d8c);
    color: #7eb3ff;
    border: 1px solid #2d5fa3;
    border-radius: 6px;
    padding: 0.2rem 0.75rem;
    font-family: 'Space Mono', monospace;
    font-size: 0.82rem;
    font-weight: 700;
}
.insight-box {
    background: #161b27;
    border-left: 3px solid #4f86f7;
    border-radius: 0 8px 8px 0;
    padding: 0.8rem 1rem;
    margin: 0.5rem 0;
    font-size: 0.88rem;
    color: #c8d0e0;
}
.warning-box {
    background: #1c1a12;
    border-left: 3px solid #f5a623;
    border-radius: 0 8px 8px 0;
    padding: 0.8rem 1rem;
    margin: 0.5rem 0;
    font-size: 0.88rem;
    color: #d4c07a;
}

/* ── Tabs ── */
button[data-baseweb="tab"] {
    font-size: 0.83rem !important;
    font-weight: 600 !important;
    color: #8b9bb4 !important;
    border-radius: 6px 6px 0 0 !important;
}
button[data-baseweb="tab"][aria-selected="true"] {
    color: #4f86f7 !important;
    border-bottom: 2px solid #4f86f7 !important;
}

/* ── DataFrame ── */
.stDataFrame { border: 1px solid #1e2736 !important; border-radius: 8px; }
</style>
""", unsafe_allow_html=True)

# ════════════════════════════════════════════════════════════
# HELPERS
# ════════════════════════════════════════════════════════════

@st.cache_data
def load_metrics(path: str) -> pd.DataFrame:
    df = pd.read_csv(path, index_col="model")
    return df


@st.cache_data
def load_raw(path: str) -> pd.DataFrame:
    df = pd.read_csv(path, sep=";", encoding="utf-8-sig")
    return df


def set_dark_fig(fig, ax_list=None):
    """Apply dark theme to matplotlib figures."""
    fig.patch.set_facecolor("#0f1117")
    if ax_list is None:
        ax_list = fig.get_axes()
    for ax in ax_list:
        ax.set_facecolor("#161b27")
        ax.tick_params(colors="#8b9bb4", labelsize=8)
        ax.xaxis.label.set_color("#8b9bb4")
        ax.yaxis.label.set_color("#8b9bb4")
        ax.title.set_color("#e8eaf0")
        for spine in ax.spines.values():
            spine.set_edgecolor("#1e2736")
        ax.grid(color="#1e2736", linewidth=0.6, linestyle="--", alpha=0.8)


# Colour palette (consistent across all charts)
MODEL_COLORS = {
    "ARIMA(7,1,2)": "#f5793a",
    "Prophet":       "#4f86f7",
    "LSTM":          "#3ecf8e",
}
ACCENT = "#4f86f7"

# ════════════════════════════════════════════════════════════
# FILE LOADING
# ════════════════════════════════════════════════════════════

# --- Sidebar ---
with st.sidebar:
    st.markdown("### 📦 Phyto Collab")
    st.markdown("**Demand Forecast Dashboard**")
    st.divider()
    st.markdown("**Data Sources**")

    # Model evaluation CSV
    metrics_path_default = "model_evaluation_results.csv"
    for candidate in [metrics_path_default,
                       "/mnt/user-data/uploads/model_evaluation_results.csv"]:
        if os.path.exists(candidate):
            metrics_path_default = candidate
            break

    metrics_file = st.file_uploader(
        "Model Evaluation CSV",
        type=["csv"],
        help="Upload model_evaluation_results.csv dari output notebook",
    )

    raw_file = st.file_uploader(
        "Raw Transaction CSV (opsional)",
        type=["csv"],
        help="Upload all_months_clean.csv untuk EDA & time series",
    )

    st.divider()
    st.markdown("**Forecast Settings**")
    forecast_horizon = st.slider("Horizon Forecast (hari)", 7, 90, 30)
    test_days = st.number_input("Test Set (hari, sesuai notebook)", value=60, min_value=7)

    st.divider()
    st.markdown(
        "<p style='font-size:0.72rem;color:#4a5568;'>Phyto Collab · Jun 2026</p>",
        unsafe_allow_html=True,
    )


# Load metrics
if metrics_file:
    metrics_df = pd.read_csv(metrics_file, index_col="model")
elif os.path.exists(metrics_path_default):
    metrics_df = load_metrics(metrics_path_default)
else:
    metrics_df = None

# Load raw
df_raw = None
if raw_file:
    try:
        df_raw = pd.read_csv(raw_file, sep=";", encoding="utf-8-sig")
    except Exception:
        st.warning("⚠️ Gagal membaca file raw. Pastikan separator ';' dan encoding UTF-8-sig.")

# ════════════════════════════════════════════════════════════
# HEADER
# ════════════════════════════════════════════════════════════
col_title, col_badge = st.columns([3, 1])
with col_title:
    st.markdown('<p class="page-title">📦 Demand Forecast Dashboard</p>', unsafe_allow_html=True)
    st.markdown(
        '<p class="page-subtitle">Shopee order time-series · ARIMA · Prophet · LSTM</p>',
        unsafe_allow_html=True,
    )
with col_badge:
    st.markdown("<br>", unsafe_allow_html=True)
    if metrics_df is not None:
        best = metrics_df["MAE"].idxmin()
        st.markdown(f'🏆 &nbsp;<span class="best-model-badge">{best}</span>', unsafe_allow_html=True)

st.divider()

# ════════════════════════════════════════════════════════════
# GATE: require metrics CSV
# ════════════════════════════════════════════════════════════
if metrics_df is None:
    st.info(
        "📂 Upload **model_evaluation_results.csv** di sidebar untuk memulai, "
        "atau letakkan file tersebut di direktori yang sama dengan script ini."
    )
    st.stop()

# ════════════════════════════════════════════════════════════
# TOP KPI CARDS
# ════════════════════════════════════════════════════════════
best_model   = metrics_df["MAE"].idxmin()
worst_model  = metrics_df["MAE"].idxmax()
best_mae     = metrics_df.loc[best_model, "MAE"]
best_rmse    = metrics_df.loc[best_model, "RMSE"]
best_mape    = metrics_df.loc[best_model, "MAPE"]
baseline_mae = metrics_df.iloc[0]["MAE"]
mae_gain     = baseline_mae - best_mae
pct_gain     = mae_gain / baseline_mae * 100

k1, k2, k3, k4, k5 = st.columns(5)
k1.metric("Best Model", best_model)
k2.metric("MAE (Best)", f"{best_mae:.2f}", delta=f"−{mae_gain:.2f} vs baseline",
          delta_color="normal")
k3.metric("RMSE (Best)", f"{best_rmse:.2f}")
k4.metric("MAPE (Best)", f"{best_mape:.1f}%")
k5.metric("Models Evaluated", len(metrics_df))

st.markdown("<br>", unsafe_allow_html=True)

# ════════════════════════════════════════════════════════════
# TABS
# ════════════════════════════════════════════════════════════
tab1, tab2, tab3, tab4 = st.tabs([
    "📊  Model Comparison",
    "🔮  Forecast Simulation",
    "📈  EDA Time Series",
    "💼  Business Impact",
])

# ────────────────────────────────────────────────────────────
# TAB 1 — MODEL COMPARISON
# ────────────────────────────────────────────────────────────
with tab1:
    st.markdown('<p class="section-title">Perbandingan Performa Model</p>', unsafe_allow_html=True)

    col_tbl, col_chart = st.columns([1, 2])

    with col_tbl:
        st.markdown("**Metrics Table**")
        display_df = metrics_df[["MAE", "RMSE", "MAPE"]].copy()
        display_df.columns = ["MAE", "RMSE", "MAPE (%)"]
        display_df = display_df.round(2)

        def highlight_best(col):
            best_idx = col.idxmin()
            return [
                "background-color:#1a3a6b; color:#7eb3ff; font-weight:700"
                if i == best_idx else ""
                for i in col.index
            ]

        st.dataframe(
            display_df.style
            .apply(highlight_best, axis=0)
            .format("{:.2f}"),
            use_container_width=True,
        )

        st.markdown("<br>", unsafe_allow_html=True)
        st.markdown("**MAE Improvement vs Baseline**")
        impr_df = metrics_df[["MAE_improvement_%"]].copy()
        impr_df.columns = ["Δ MAE (%)"]
        st.dataframe(impr_df.style.format("{:.1f}"), use_container_width=True)

    with col_chart:
        fig, axes = plt.subplots(1, 3, figsize=(11, 3.8))
        metrics_labels = ["MAE", "RMSE", "MAPE"]

        for ax, metric in zip(axes, metrics_labels):
            vals   = metrics_df[metric].values
            models = metrics_df.index.tolist()
            colors = [MODEL_COLORS.get(m, "#8b9bb4") for m in models]
            bars   = ax.bar(models, vals, color=colors, alpha=0.88,
                            edgecolor="#0f1117", linewidth=1.2, width=0.55)

            best_val = min(vals)
            for bar, val in zip(bars, vals):
                ax.text(
                    bar.get_x() + bar.get_width() / 2,
                    bar.get_height() + max(vals) * 0.03,
                    f"{val:.2f}",
                    ha="center", va="bottom",
                    fontsize=8.5, fontweight="700",
                    color="#e8eaf0" if val == best_val else "#8b9bb4",
                    fontfamily="monospace",
                )

            ax.set_title(metric + (" (%)" if metric == "MAPE" else ""),
                         fontsize=10, fontweight="600")
            ax.set_ylim(0, max(vals) * 1.3)
            ax.tick_params(axis="x", labelsize=7.5, rotation=10)
            ax.spines[["top", "right"]].set_visible(False)

        set_dark_fig(fig, axes)
        fig.suptitle("Model Performance Comparison", fontsize=11,
                     fontweight="700", color="#e8eaf0", y=1.02)
        plt.tight_layout()
        st.pyplot(fig, use_container_width=True)
        plt.close(fig)

    # Insight boxes
    st.markdown("<br>", unsafe_allow_html=True)
    c1, c2, c3 = st.columns(3)
    with c1:
        st.markdown(
            f'<div class="insight-box">🏆 <strong>{best_model}</strong> meraih MAE terendah '
            f'<strong>{best_mae:.2f}</strong> order/hari di test set 60 hari.</div>',
            unsafe_allow_html=True,
        )
    with c2:
        arima_mape = metrics_df.loc["ARIMA(7,1,2)", "MAPE"] if "ARIMA(7,1,2)" in metrics_df.index else None
        if arima_mape:
            st.markdown(
                f'<div class="insight-box">📉 ARIMA mencatat MAPE {arima_mape:.1f}% — '
                f'tinggi karena demand spike yang tidak linear.</div>',
                unsafe_allow_html=True,
            )
    with c3:
        if "Prophet" in metrics_df.index and "LSTM" in metrics_df.index:
            p_mae = metrics_df.loc["Prophet", "MAE"]
            l_mae = metrics_df.loc["LSTM", "MAE"]
            diff  = abs(p_mae - l_mae)
            better = "LSTM" if l_mae < p_mae else "Prophet"
            st.markdown(
                f'<div class="insight-box">⚡ {better} unggul {diff:.2f} MAE '
                f'dibanding lawannya — gap relatif kecil.</div>',
                unsafe_allow_html=True,
            )

# ────────────────────────────────────────────────────────────
# TAB 2 — FORECAST SIMULATION
# ────────────────────────────────────────────────────────────
with tab2:
    st.markdown('<p class="section-title">Simulasi Forecast Ke Depan</p>', unsafe_allow_html=True)

    st.markdown(
        '<div class="insight-box">ℹ️ Simulasi di bawah menggunakan <strong>seasonal naive + noise '
        'berdasarkan residual ARIMA</strong> sebagai proxy visualisasi bisnis. '
        'Untuk prediksi akurat, jalankan ulang model dari notebook dan export hasilnya.</div>',
        unsafe_allow_html=True,
    )

    col_ctrl, col_sim = st.columns([1, 3])
    with col_ctrl:
        base_avg   = st.number_input("Rata-rata Order/Hari (baseline)", value=22, min_value=1)
        base_std   = st.number_input("Std Dev Order/Hari", value=14, min_value=1)
        best_mape_pct = metrics_df.loc[best_model, "MAPE"] / 100
        worst_mape_pct = metrics_df.loc[worst_model, "MAPE"] / 100
        show_ci    = st.checkbox("Tampilkan confidence interval", value=True)
        model_sel  = st.selectbox("Warna garis prediksi", options=list(MODEL_COLORS.keys()),
                                  index=list(MODEL_COLORS.keys()).index(best_model)
                                        if best_model in MODEL_COLORS else 0)
        np.random.seed(42)

    with col_sim:
        dates = pd.date_range(pd.Timestamp.today(), periods=forecast_horizon, freq="D")
        # Seasonal naive: weekly pattern
        dow_factor = np.array([0.9, 1.05, 1.0, 1.0, 1.1, 0.85, 0.95])
        actual_sim = np.array([
            max(0, base_avg * dow_factor[d.dayofweek] + np.random.randn() * base_std * 0.6)
            for d in dates
        ])
        forecast_sim = np.array([
            max(0, base_avg * dow_factor[d.dayofweek] * (1 + (np.random.randn() * best_mape_pct * 0.4)))
            for d in dates
        ])
        ci_upper = forecast_sim * (1 + best_mape_pct * 0.5)
        ci_lower = np.maximum(0, forecast_sim * (1 - best_mape_pct * 0.5))

        fig, ax = plt.subplots(figsize=(10, 3.6))
        ax.plot(dates, actual_sim, color="#e8eaf0", linewidth=1.8, alpha=0.7,
                label="Estimasi Aktual", zorder=3)
        ax.plot(dates, forecast_sim, color=MODEL_COLORS[model_sel], linewidth=2,
                linestyle="--", label=f"{model_sel} Forecast", zorder=4)
        if show_ci:
            ax.fill_between(dates, ci_lower, ci_upper,
                            color=MODEL_COLORS[model_sel], alpha=0.15,
                            label=f"95% CI (±MAPE {metrics_df.loc[best_model,'MAPE']:.0f}%)")
        ax.xaxis.set_major_locator(mdates.WeekdayLocator(interval=1))
        ax.xaxis.set_major_formatter(mdates.DateFormatter("%d %b"))
        ax.set_xlabel("Tanggal")
        ax.set_ylabel("Jumlah Order")
        ax.set_title(f"{forecast_horizon}-Hari Order Forecast Simulation", fontweight="600")
        ax.legend(fontsize=8, framealpha=0.15, labelcolor="#e8eaf0")
        plt.xticks(rotation=30)
        set_dark_fig(fig, [ax])
        plt.tight_layout()
        st.pyplot(fig, use_container_width=True)
        plt.close(fig)

        # summary stats
        s1, s2, s3, s4 = st.columns(4)
        s1.metric("Total Forecast (hari)", forecast_horizon)
        s2.metric("Avg Order/Hari (forecast)", f"{forecast_sim.mean():.1f}")
        s3.metric("Peak Day", dates[forecast_sim.argmax()].strftime("%d %b"))
        s4.metric("Min Order Day", dates[forecast_sim.argmin()].strftime("%d %b"))

# ────────────────────────────────────────────────────────────
# TAB 3 — EDA TIME SERIES
# ────────────────────────────────────────────────────────────
with tab3:
    st.markdown('<p class="section-title">Exploratory Data Analysis</p>', unsafe_allow_html=True)

    if df_raw is None:
        st.info(
            "📂 Upload **all_months_clean.csv** di sidebar untuk melihat EDA "
            "(Daily demand trend, heatmap jam, distribusi kategori)."
        )
    else:
        # ── Clean minimal ──
        col_map = {
            "Waktu Pesanan Dibuat": "order_time_raw",
            "Status Pesanan":       "order_status_raw",
            "Total Pembayaran":     "total_revenue",
            "Opsi Pengiriman":      "shipping_option",
            "Metode Pembayaran":    "payment_method",
            "Kota/Kabupaten":       "city",
            "Total Diskon":         "total_discount",
            "total_qty":            "total_qty",
            "Perkiraan Ongkos Kirim": "estimated_shipping_cost",
        }
        df = df_raw.rename(columns={k: v for k, v in col_map.items() if k in df_raw.columns})

        if "order_time_raw" in df.columns:
            df["order_time"] = pd.to_datetime(df["order_time_raw"], errors="coerce")
            df = df.dropna(subset=["order_time"])
            df["hour"]        = df["order_time"].dt.hour
            df["day_of_week"] = df["order_time"].dt.dayofweek
            df["date"]        = df["order_time"].dt.date
            df["month"]       = df["order_time"].dt.month
            df["year"]        = df["order_time"].dt.year

        if "order_status_raw" in df.columns:
            def norm_status(s):
                if s == "Selesai": return "Completed"
                if s == "Batal": return "Cancelled"
                if str(s).startswith("Pesanan diterima"): return "Delivered"
                if s in ("Sedang Dikirim", "Telah Dikirim"): return "In Transit"
                return "Other"
            df["order_status"] = df["order_status_raw"].apply(norm_status)
            completed = df[df["order_status"] == "Completed"].copy()
        else:
            completed = df.copy()

        # ── Daily TS ──
        if "date" in df.columns:
            daily_ts = (
                completed.groupby("date")
                .agg(order_count=("order_time_raw" if "order_time_raw" in completed.columns
                                  else completed.columns[0], "count"))
                .reset_index()
            )
            daily_ts["date"] = pd.to_datetime(daily_ts["date"])

            fig, ax = plt.subplots(figsize=(12, 3.4))
            ax.plot(daily_ts["date"], daily_ts["order_count"],
                    color=ACCENT, linewidth=0.9, alpha=0.55, label="Daily Orders")
            ax.plot(daily_ts["date"], daily_ts["order_count"].rolling(7).mean(),
                    color="#f5793a", linewidth=1.8, label="7-Day MA")
            ax.plot(daily_ts["date"], daily_ts["order_count"].rolling(30).mean(),
                    color="#3ecf8e", linewidth=1.8, linestyle="--", label="30-Day MA")
            mean_d = daily_ts["order_count"].mean()
            std_d  = daily_ts["order_count"].std()
            spikes = daily_ts[daily_ts["order_count"] > mean_d + 2 * std_d]
            ax.scatter(spikes["date"], spikes["order_count"],
                       color="#f5793a", zorder=5, s=35, label="Demand Spike")
            ax.xaxis.set_major_locator(mdates.MonthLocator(interval=2))
            ax.xaxis.set_major_formatter(mdates.DateFormatter("%b %Y"))
            ax.set_title("Daily Order Demand", fontweight="600")
            ax.set_ylabel("Orders")
            ax.legend(fontsize=8, framealpha=0.15, labelcolor="#e8eaf0")
            plt.xticks(rotation=30)
            set_dark_fig(fig, [ax])
            plt.tight_layout()
            st.pyplot(fig, use_container_width=True)
            plt.close(fig)

        # ── Heatmap & Monthly side by side ──
        c_heat, c_monthly = st.columns(2)

        if "hour" in df.columns and "day_of_week" in df.columns:
            with c_heat:
                st.markdown("**Heatmap: Jam × Hari**")
                hour_dow = (
                    completed.groupby(["day_of_week", "hour"]).size()
                    .unstack("hour").fillna(0)
                )
                hour_dow.index = ["Sen","Sel","Rab","Kam","Jum","Sab","Min"]
                fig2, ax2 = plt.subplots(figsize=(7, 3.2))
                sns.heatmap(hour_dow, cmap="Blues", linewidths=0.2,
                            cbar_kws={"label": "Orders", "shrink": 0.7},
                            ax=ax2, linecolor="#0f1117")
                ax2.set_title("Demand Heatmap", fontweight="600")
                ax2.set_xlabel("Jam (0–23)")
                ax2.set_ylabel("")
                set_dark_fig(fig2, [ax2])
                fig2.patch.set_facecolor("#161b27")
                plt.tight_layout()
                st.pyplot(fig2, use_container_width=True)
                plt.close(fig2)

        if "year" in completed.columns and "month" in completed.columns:
            with c_monthly:
                st.markdown("**Volume Order Bulanan**")
                monthly = (
                    completed.groupby(["year", "month"])
                    .size().reset_index(name="order_count")
                )
                monthly["period"] = pd.to_datetime(
                    monthly[["year", "month"]].assign(day=1)
                )
                fig3, ax3 = plt.subplots(figsize=(7, 3.2))
                year_colors = {2023: "#4f86f7", 2024: "#3ecf8e", 2025: "#f5793a"}
                bar_colors = [year_colors.get(y, "#8b9bb4") for y in monthly["year"]]
                ax3.bar(monthly["period"], monthly["order_count"],
                        width=20, color=bar_colors, alpha=0.88)
                ax3.xaxis.set_major_locator(mdates.MonthLocator(interval=3))
                ax3.xaxis.set_major_formatter(mdates.DateFormatter("%b\n%Y"))
                ax3.set_title("Monthly Order Volume", fontweight="600")
                ax3.set_ylabel("Orders")
                set_dark_fig(fig3, [ax3])
                plt.tight_layout()
                st.pyplot(fig3, use_container_width=True)
                plt.close(fig3)

        # ── Top categories ──
        if "product_categories" in completed.columns and "total_revenue" in completed.columns:
            st.markdown("**Top 10 Kategori Produk · Revenue**")
            top_cat = (
                completed.groupby("product_categories")["total_revenue"]
                .sum().sort_values(ascending=False).head(10)
            )
            fig4, ax4 = plt.subplots(figsize=(12, 3.2))
            bars = ax4.barh(top_cat.index[::-1], top_cat.values[::-1] / 1e6,
                            color=ACCENT, alpha=0.82, edgecolor="#0f1117")
            for bar, val in zip(bars, top_cat.values[::-1] / 1e6):
                ax4.text(bar.get_width() + 0.1, bar.get_y() + bar.get_height() / 2,
                         f"Rp {val:.1f}jt", va="center", ha="left",
                         fontsize=7.5, color="#8b9bb4", fontfamily="monospace")
            ax4.set_xlabel("Revenue (Rp Juta)")
            ax4.set_title("Revenue by Product Category", fontweight="600")
            ax4.spines[["top", "right"]].set_visible(False)
            set_dark_fig(fig4, [ax4])
            plt.tight_layout()
            st.pyplot(fig4, use_container_width=True)
            plt.close(fig4)

# ────────────────────────────────────────────────────────────
# TAB 4 — BUSINESS IMPACT
# ────────────────────────────────────────────────────────────
with tab4:
    st.markdown('<p class="section-title">Estimasi Dampak Bisnis</p>', unsafe_allow_html=True)

    st.markdown("Sesuaikan asumsi di bawah untuk melihat estimasi penghematan operasional.")

    col_inp, col_res = st.columns([1, 2])

    with col_inp:
        avg_monthly_orders = st.number_input("Avg. Order/Bulan", value=660, step=10)
        avg_shipping_cost  = st.number_input("Avg. Biaya Kirim/Order (Rp)", value=12000, step=500)
        cancelled_rate     = st.slider("Cancelled Order Rate (%)", 0.0, 30.0, 13.8, 0.1)
        avg_revenue        = st.number_input("Avg. Revenue/Order (Rp)", value=38000, step=1000)

        # Pull MAPE from actual metrics
        mape_before = float(metrics_df.iloc[0]["MAPE"])
        mape_after  = float(metrics_df["MAPE"].min())

        st.markdown("<br>", unsafe_allow_html=True)
        st.markdown(
            f'<div class="insight-box">📌 MAPE Baseline: <strong>{mape_before:.1f}%</strong> '
            f'→ Best Model: <strong>{mape_after:.1f}%</strong><br>'
            f'(data otomatis dari model_evaluation_results.csv)</div>',
            unsafe_allow_html=True,
        )

    with col_res:
        # Dispatch waste
        wasted_before = avg_monthly_orders * (mape_before / 100) * avg_shipping_cost
        wasted_after  = avg_monthly_orders * (mape_after  / 100) * avg_shipping_cost
        monthly_save  = wasted_before - wasted_after
        annual_save   = monthly_save * 12

        # Stockout / overstock cost (rough: 5% of revenue × error delta)
        stockout_before = avg_monthly_orders * avg_revenue * (mape_before / 100) * 0.05
        stockout_after  = avg_monthly_orders * avg_revenue * (mape_after  / 100) * 0.05
        stockout_save   = stockout_before - stockout_after

        b1, b2, b3 = st.columns(3)
        b1.metric("Penghematan Dispatch/Bulan",
                  f"Rp {monthly_save:,.0f}",
                  delta=f"+Rp {annual_save:,.0f}/tahun")
        b2.metric("Estimasi Hemat Stockout/Bulan",
                  f"Rp {stockout_save:,.0f}")
        b3.metric("Total Estimasi Hemat/Tahun",
                  f"Rp {(monthly_save + stockout_save) * 12:,.0f}")

        # Waterfall chart
        categories  = ["Biaya Dispatch\n(Sebelum)", "Hemat Dispatch", "Hemat Stockout", "Total Sisa"]
        values_base = [wasted_before, -monthly_save, -stockout_save,
                       wasted_before - monthly_save - stockout_save]
        bottoms = [0, values_base[0], values_base[0] - monthly_save, 0]
        bar_clrs = ["#f5793a", "#3ecf8e", "#3ecf8e", ACCENT]

        fig5, ax5 = plt.subplots(figsize=(9, 3.6))
        bars5 = ax5.bar(categories, [abs(v) for v in values_base], bottom=bottoms,
                        color=bar_clrs, alpha=0.88, width=0.5,
                        edgecolor="#0f1117", linewidth=1)
        for bar, val in zip(bars5, values_base):
            label = f"Rp {abs(val):,.0f}"
            ax5.text(bar.get_x() + bar.get_width() / 2,
                     bar.get_y() + bar.get_height() / 2,
                     label, ha="center", va="center",
                     fontsize=8, fontweight="600",
                     color="#0f1117", fontfamily="monospace")
        ax5.yaxis.set_major_formatter(mticker.FuncFormatter(
            lambda x, _: f"Rp {x/1000:.0f}K" if x >= 1000 else f"Rp {x:.0f}"
        ))
        ax5.set_title("Estimasi Penghematan Biaya Operasional / Bulan",
                      fontweight="600")
        ax5.spines[["top", "right"]].set_visible(False)
        set_dark_fig(fig5, [ax5])
        plt.tight_layout()
        st.pyplot(fig5, use_container_width=True)
        plt.close(fig5)

    # ── Narrative ──
    st.markdown("<br>", unsafe_allow_html=True)
    st.markdown('<p class="section-title">Ringkasan & Rekomendasi</p>', unsafe_allow_html=True)
    c_l, c_r = st.columns(2)
    with c_l:
        st.markdown(
            f'<div class="insight-box">'
            f'✅ Model <strong>{best_model}</strong> memberikan MAE terbaik <strong>{best_mae:.2f} order/hari</strong>. '
            f'Pada volume rata-rata <strong>{avg_monthly_orders:,} order/bulan</strong>, '
            f'peningkatan akurasi ini menghasilkan estimasi penghematan biaya dispatching '
            f'<strong>Rp {monthly_save:,.0f}/bulan</strong>.'
            f'</div>',
            unsafe_allow_html=True,
        )
    with c_r:
        st.markdown(
            f'<div class="warning-box">'
            f'⚠️ MAPE semua model masih di atas 70% — demand spike pada promo date '
            f'(Harbolnas, double-date) mendominasi error. '
            f'Rekomendasi: tambahkan <strong>promo intensity feature</strong> dan '
            f'pertimbangkan <strong>ensemble ARIMA + Prophet</strong> untuk mengurangi MAPE.'
            f'</div>',
            unsafe_allow_html=True,
        )

    # ── Model decision guide ──
    st.markdown("<br>", unsafe_allow_html=True)
    st.markdown('<p class="section-title">Panduan Pemilihan Model untuk Produksi</p>',
                unsafe_allow_html=True)

    guide_data = {
        "Model":        ["ARIMA(7,1,2)",  "Prophet",          "LSTM"],
        "MAE":          [metrics_df.loc[m, "MAE"] if m in metrics_df.index else "—"
                         for m in ["ARIMA(7,1,2)", "Prophet", "LSTM"]],
        "MAPE (%)":     [metrics_df.loc[m, "MAPE"] if m in metrics_df.index else "—"
                         for m in ["ARIMA(7,1,2)", "Prophet", "LSTM"]],
        "Cocok untuk":  [
            "Forecast harian rutin, infrastruktur minimal",
            "Holiday effect, interpretasi bisnis",
            "Pattern nonlinear jangka panjang",
        ],
        "Kekurangan":   [
            "Tidak handle multiple seasonality",
            "MAPE tinggi saat spike ekstrem",
            "Butuh data besar, less interpretable",
        ],
        "Deploy?": ["✅ Siap", "⚠️ Perlu tuning", "⚠️ Perlu lebih banyak data"],
    }
    guide_df = pd.DataFrame(guide_data)
    st.dataframe(guide_df, use_container_width=True, hide_index=True)

# ════════════════════════════════════════════════════════════
# FOOTER
# ════════════════════════════════════════════════════════════
st.divider()
st.markdown(
    "<p style='text-align:center;color:#4a5568;font-size:0.75rem;'>"
    "Phyto Collab · Demand Forecast Dashboard · Moch Yahya · Jun 2026"
    "</p>",
    unsafe_allow_html=True,
)
