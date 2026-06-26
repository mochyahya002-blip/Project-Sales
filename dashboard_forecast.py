# ============================================================
# PHYTO COLLAB — Demand Forecast Dashboard
# Author : Moch Yahya
# Run    : streamlit run dashboard_forecast.py
# ============================================================
"""
Demand forecasting dashboard for Shopee order time-series data.

Structure
---------
1. Page / style configuration
2. Reusable helpers (data loading, chart theming, KPI cards)
3. Sidebar (data sources + forecast settings)
4. Header + top-level KPIs
5. Tabs: Model Comparison / Forecast Simulation / EDA / Business Impact
6. Footer

The whole render flow is wrapped in main() and only executed at the
bottom of the file, so the module can be imported (e.g. for tests)
without immediately triggering a Streamlit run.
"""

import os
import warnings
from pathlib import Path
from typing import Optional

import matplotlib.dates as mdates
import matplotlib.pyplot as plt
import matplotlib.ticker as mticker
import numpy as np
import pandas as pd
import seaborn as sns
import streamlit as st

warnings.filterwarnings("ignore")

BASE_PATH = Path(__file__).parent

# Colour palette (kept consistent across every chart and badge)
MODEL_COLORS = {
    "ARIMA(7,1,2)": "#f5793a",
    "Prophet": "#4f86f7",
    "LSTM": "#3ecf8e",
}
ACCENT = "#4f86f7"
BG_DARK = "#0f1117"
BG_PANEL = "#161b27"
BORDER = "#1e2736"
TEXT_PRIMARY = "#e8eaf0"
TEXT_MUTED = "#8b9bb4"


# ════════════════════════════════════════════════════════════
# 1. PAGE / STYLE CONFIGURATION
# ════════════════════════════════════════════════════════════

def configure_page() -> None:
    st.set_page_config(
        page_title="Phyto Collab | Demand Forecast",
        layout="wide",
        initial_sidebar_state="expanded",
    )


def inject_css() -> None:
    st.markdown(
        """
        <style>
        @import url('https://fonts.googleapis.com/css2?family=Inter:wght@400;500;600;700&family=Space+Mono:wght@400;700&display=swap');

        html, body, [class*="css"] { font-family: 'Inter', sans-serif; }

        .stApp { background-color: #0f1117; color: #e8eaf0; }

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

        /* ── KPI cards (custom, replaces st.metric where values can be long) ── */
        .kpi-card {
            background: #161b27;
            border: 1px solid #1e2736;
            border-radius: 10px;
            padding: 1rem 1.2rem;
            height: 100%;
        }
        .kpi-label {
            color: #8b9bb4;
            font-size: 0.78rem;
            margin-bottom: 0.45rem;
        }
        .kpi-value {
            color: #e8eaf0;
            font-family: 'Space Mono', monospace;
            font-size: 1.6rem;
            line-height: 1.25;
            word-break: break-word;
            overflow-wrap: break-word;
        }
        .kpi-value-sm {
            color: #e8eaf0;
            font-family: 'Space Mono', monospace;
            font-size: 1.05rem;
            line-height: 1.35;
            word-break: break-word;
            overflow-wrap: break-word;
        }
        .kpi-delta {
            display: inline-block;
            margin-top: 0.55rem;
            font-size: 0.8rem;
            padding: 0.15rem 0.55rem;
            border-radius: 5px;
        }
        .kpi-delta-positive { color: #3ecf8e; background: rgba(62, 207, 142, 0.12); }
        .kpi-delta-negative { color: #f5793a; background: rgba(245, 121, 58, 0.12); }
        .kpi-delta-neutral  { color: #8b9bb4; background: rgba(139, 155, 180, 0.12); }

        /* ── Native st.metric (kept for short numeric values, e.g. tab 2 summary) ── */
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
            font-size: 1.4rem !important;
            white-space: normal;
            word-break: break-word;
        }
        div[data-testid="metric-container"] [data-testid="stMetricDelta"] { font-size: 0.82rem !important; }

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
        .page-title { font-size: 1.55rem; font-weight: 700; color: #e8eaf0; margin-bottom: 0; }
        .page-subtitle { color: #8b9bb4; font-size: 0.92rem; margin-top: 0.2rem; }

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
            word-break: break-word;
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

        .stDataFrame { border: 1px solid #1e2736 !important; border-radius: 8px; }
        </style>
        """,
        unsafe_allow_html=True,
    )


# ════════════════════════════════════════════════════════════
# 2. REUSABLE HELPERS
# ════════════════════════════════════════════════════════════

def find_default_path(filename: str) -> Optional[str]:
    """Look for `filename` in the usual project locations, in priority order."""
    candidates = [
        BASE_PATH / filename,
        BASE_PATH / "data" / filename,
        Path("/mnt/user-data/uploads") / filename,
    ]
    for candidate in candidates:
        if candidate.exists():
            return str(candidate)
    return None


@st.cache_data
def load_metrics(path: str) -> pd.DataFrame:
    return pd.read_csv(path, index_col="model")


@st.cache_data
def load_raw(path: str) -> pd.DataFrame:
    return pd.read_csv(path, sep=";", encoding="utf-8-sig")


def set_dark_fig(fig, ax_list=None) -> None:
    """Apply the dashboard's dark theme to a matplotlib figure."""
    fig.patch.set_facecolor(BG_DARK)
    if ax_list is None:
        ax_list = fig.get_axes()
    for ax in ax_list:
        ax.set_facecolor(BG_PANEL)
        ax.tick_params(colors=TEXT_MUTED, labelsize=8)
        ax.xaxis.label.set_color(TEXT_MUTED)
        ax.yaxis.label.set_color(TEXT_MUTED)
        ax.title.set_color(TEXT_PRIMARY)
        for spine in ax.spines.values():
            spine.set_edgecolor(BORDER)
        ax.grid(color=BORDER, linewidth=0.6, linestyle="--", alpha=0.8)


def kpi_card(label: str, value: str, delta: Optional[str] = None, tone: str = "neutral") -> str:
    """
    Build an overflow-safe KPI card as HTML.

    Long text values (e.g. a model name like "ARIMA(7,1,2)") automatically
    drop to a smaller font and wrap instead of being clipped with "...",
    which was the cause of the truncated "Best Model" card.
    """
    value_class = "kpi-value-sm" if len(value) > 10 else "kpi-value"
    delta_html = f'<div class="kpi-delta kpi-delta-{tone}">{delta}</div>' if delta else ""
    return (
        f'<div class="kpi-card">'
        f'<div class="kpi-label">{label}</div>'
        f'<div class="{value_class}">{value}</div>'
        f"{delta_html}"
        f"</div>"
    )


def insight_box(text: str) -> str:
    return f'<div class="insight-box">{text}</div>'


def warning_box(text: str) -> str:
    return f'<div class="warning-box">{text}</div>'


# ════════════════════════════════════════════════════════════
# 3. SIDEBAR
# ════════════════════════════════════════════════════════════

def render_sidebar():
    """Render the sidebar and return (metrics_df, df_raw, forecast_horizon, test_days)."""
    metrics_path_default = find_default_path("model_evaluation_results.csv")
    raw_path_default = find_default_path("all_months_clean.csv")

    with st.sidebar:
        st.markdown("### Phyto Collab")
        st.markdown("**Demand Forecast Dashboard**")
        st.divider()
        st.markdown("**Data Sources**")

        if metrics_path_default:
            st.success(f"Data loaded: `{os.path.basename(metrics_path_default)}`")
        if raw_path_default:
            st.success(f"Raw data loaded: `{os.path.basename(raw_path_default)}`")

        st.markdown("**Override Data (Opsional)**")
        metrics_file = st.file_uploader(
            "Update Model Evaluation CSV",
            type=["csv"],
            help="Upload untuk mengganti model_evaluation_results.csv",
        )
        raw_file = st.file_uploader(
            "Update Raw Transaction CSV",
            type=["csv"],
            help="Upload untuk mengganti all_months_clean.csv",
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

    # Resolve metrics_df (uploaded file takes priority over the default path)
    metrics_df = None
    if metrics_file:
        try:
            metrics_df = pd.read_csv(metrics_file, index_col="model")
            st.toast("File metrics berhasil diperbarui.")
        except Exception as exc:
            st.error(f"Gagal membaca file metrics: {exc}")
    elif metrics_path_default:
        metrics_df = load_metrics(metrics_path_default)

    # Resolve df_raw the same way
    df_raw = None
    if raw_file:
        try:
            df_raw = pd.read_csv(raw_file, sep=";", encoding="utf-8-sig")
            st.toast("File raw data berhasil diperbarui.")
        except Exception as exc:
            st.error(f"Gagal membaca file raw: {exc}")
    elif raw_path_default:
        try:
            df_raw = load_raw(raw_path_default)
        except Exception:
            df_raw = None

    return metrics_df, df_raw, forecast_horizon, test_days


# ════════════════════════════════════════════════════════════
# 4. HEADER + TOP-LEVEL KPIs
# ════════════════════════════════════════════════════════════

def render_header(metrics_df: pd.DataFrame):
    """
    Render the page header and top KPI row.

    Returns (best_model, worst_model, best_mae, best_rmse, best_mape) for
    reuse further down the page.
    """
    col_title, col_badge = st.columns([3, 1])
    with col_title:
        st.markdown('<p class="page-title">Demand Forecast Dashboard</p>', unsafe_allow_html=True)
        st.markdown(
            '<p class="page-subtitle">Shopee order time-series · ARIMA · Prophet · LSTM</p>',
            unsafe_allow_html=True,
        )
    with col_badge:
        st.markdown("<br>", unsafe_allow_html=True)
        best_preview = metrics_df["MAE"].idxmin()
        st.markdown(
            f'<span class="kpi-label">Model Terbaik</span><br>'
            f'<span class="best-model-badge">{best_preview}</span>',
            unsafe_allow_html=True,
        )

    st.divider()

    best_model = metrics_df["MAE"].idxmin()
    worst_model = metrics_df["MAE"].idxmax()
    best_mae = metrics_df.loc[best_model, "MAE"]
    best_rmse = metrics_df.loc[best_model, "RMSE"]
    best_mape = metrics_df.loc[best_model, "MAPE"]

    # Comparison baseline = the worst-performing evaluated model, not just
    # "row 0 of the CSV" (the original logic), which previously happened to
    # *be* the best model itself and made every "vs baseline" delta read 0.
    baseline_mae = metrics_df.loc[worst_model, "MAE"]
    mae_gain = baseline_mae - best_mae
    delta_tone = "positive" if mae_gain > 0 else ("negative" if mae_gain < 0 else "neutral")
    delta_text = f"{mae_gain:+.2f} vs {worst_model}" if best_model != worst_model else None

    k1, k2, k3, k4, k5 = st.columns(5)
    with k1:
        st.markdown(kpi_card("Best Model", str(best_model)), unsafe_allow_html=True)
    with k2:
        st.markdown(
            kpi_card("MAE (Best)", f"{best_mae:.2f}", delta_text, delta_tone),
            unsafe_allow_html=True,
        )
    with k3:
        st.markdown(kpi_card("RMSE (Best)", f"{best_rmse:.2f}"), unsafe_allow_html=True)
    with k4:
        st.markdown(kpi_card("MAPE (Best)", f"{best_mape:.1f}%"), unsafe_allow_html=True)
    with k5:
        st.markdown(kpi_card("Models Evaluated", str(len(metrics_df))), unsafe_allow_html=True)

    st.markdown("<br>", unsafe_allow_html=True)
    return best_model, worst_model, best_mae, best_rmse, best_mape


# ════════════════════════════════════════════════════════════
# 5. TAB 1 — MODEL COMPARISON
# ════════════════════════════════════════════════════════════

def render_model_comparison_tab(metrics_df: pd.DataFrame, best_model: str) -> None:
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
                "background-color:#1a3a6b; color:#7eb3ff; font-weight:700" if i == best_idx else ""
                for i in col.index
            ]

        st.dataframe(
            display_df.style.apply(highlight_best, axis=0).format("{:.2f}"),
            use_container_width=True,
        )

        if "MAE_improvement_%" in metrics_df.columns:
            st.markdown("<br>", unsafe_allow_html=True)
            st.markdown("**MAE Improvement vs Baseline**")
            impr_df = metrics_df[["MAE_improvement_%"]].copy()
            impr_df.columns = ["Δ MAE (%)"]
            st.dataframe(impr_df.style.format("{:.1f}"), use_container_width=True)

    with col_chart:
        fig, axes = plt.subplots(1, 3, figsize=(11, 3.8))
        metrics_labels = ["MAE", "RMSE", "MAPE"]

        for ax, metric in zip(axes, metrics_labels):
            vals = metrics_df[metric].values
            models = metrics_df.index.tolist()
            colors = [MODEL_COLORS.get(m, TEXT_MUTED) for m in models]
            bars = ax.bar(models, vals, color=colors, alpha=0.88,
                           edgecolor=BG_DARK, linewidth=1.2, width=0.55)

            best_val = min(vals)
            for bar, val in zip(bars, vals):
                ax.text(
                    bar.get_x() + bar.get_width() / 2,
                    bar.get_height() + max(vals) * 0.03,
                    f"{val:.2f}",
                    ha="center", va="bottom",
                    fontsize=8.5, fontweight="700",
                    color=TEXT_PRIMARY if val == best_val else TEXT_MUTED,
                    fontfamily="monospace",
                )

            ax.set_title(metric + (" (%)" if metric == "MAPE" else ""), fontsize=10, fontweight="600")
            ax.set_ylim(0, max(vals) * 1.3)
            ax.tick_params(axis="x", labelsize=7.5, rotation=10)
            ax.spines[["top", "right"]].set_visible(False)

        set_dark_fig(fig, axes)
        fig.suptitle("Model Performance Comparison", fontsize=11, fontweight="700", color=TEXT_PRIMARY, y=1.02)
        plt.tight_layout()
        st.pyplot(fig, use_container_width=True)
        plt.close(fig)

    st.markdown("<br>", unsafe_allow_html=True)
    c1, c2, c3 = st.columns(3)
    with c1:
        best_mae = metrics_df.loc[best_model, "MAE"]
        st.markdown(
            insight_box(
                f"<strong>{best_model}</strong> meraih MAE terendah "
                f"<strong>{best_mae:.2f}</strong> order/hari di test set 60 hari."
            ),
            unsafe_allow_html=True,
        )
    with c2:
        if "ARIMA(7,1,2)" in metrics_df.index:
            arima_mape = metrics_df.loc["ARIMA(7,1,2)", "MAPE"]
            st.markdown(
                insight_box(
                    f"ARIMA mencatat MAPE {arima_mape:.1f}% — "
                    f"tinggi karena demand spike yang tidak linear."
                ),
                unsafe_allow_html=True,
            )
    with c3:
        if "Prophet" in metrics_df.index and "LSTM" in metrics_df.index:
            p_mae = metrics_df.loc["Prophet", "MAE"]
            l_mae = metrics_df.loc["LSTM", "MAE"]
            diff = abs(p_mae - l_mae)
            better = "LSTM" if l_mae < p_mae else "Prophet"
            st.markdown(
                insight_box(f"{better} unggul {diff:.2f} MAE dibanding lawannya — gap relatif kecil."),
                unsafe_allow_html=True,
            )


# ════════════════════════════════════════════════════════════
# 6. TAB 2 — FORECAST SIMULATION
# ════════════════════════════════════════════════════════════

def render_forecast_simulation_tab(metrics_df: pd.DataFrame, best_model: str, worst_model: str,
                                    forecast_horizon: int) -> None:
    st.markdown('<p class="section-title">Simulasi Forecast Ke Depan</p>', unsafe_allow_html=True)

    st.markdown(
        insight_box(
            "Simulasi di bawah menggunakan <strong>seasonal naive + noise "
            "berdasarkan residual ARIMA</strong> sebagai proxy visualisasi bisnis. "
            "Untuk prediksi akurat, jalankan ulang model dari notebook dan export hasilnya."
        ),
        unsafe_allow_html=True,
    )

    col_ctrl, col_sim = st.columns([1, 3])
    with col_ctrl:
        base_avg = st.number_input("Rata-rata Order/Hari (baseline)", value=22, min_value=1)
        base_std = st.number_input("Std Dev Order/Hari", value=14, min_value=1)
        best_mape_pct = metrics_df.loc[best_model, "MAPE"] / 100
        show_ci = st.checkbox("Tampilkan confidence interval", value=True)
        model_sel = st.selectbox(
            "Warna garis prediksi",
            options=list(MODEL_COLORS.keys()),
            index=list(MODEL_COLORS.keys()).index(best_model) if best_model in MODEL_COLORS else 0,
        )
        np.random.seed(42)

    with col_sim:
        dates = pd.date_range(pd.Timestamp.today(), periods=forecast_horizon, freq="D")
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
        ax.plot(dates, actual_sim, color=TEXT_PRIMARY, linewidth=1.8, alpha=0.7,
                label="Estimasi Aktual", zorder=3)
        ax.plot(dates, forecast_sim, color=MODEL_COLORS[model_sel], linewidth=2,
                linestyle="--", label=f"{model_sel} Forecast", zorder=4)
        if show_ci:
            ax.fill_between(dates, ci_lower, ci_upper, color=MODEL_COLORS[model_sel], alpha=0.15,
                             label=f"95% CI (±MAPE {metrics_df.loc[best_model, 'MAPE']:.0f}%)")
        ax.xaxis.set_major_locator(mdates.WeekdayLocator(interval=1))
        ax.xaxis.set_major_formatter(mdates.DateFormatter("%d %b"))
        ax.set_xlabel("Tanggal")
        ax.set_ylabel("Jumlah Order")
        ax.set_title(f"{forecast_horizon}-Hari Order Forecast Simulation", fontweight="600")
        ax.legend(fontsize=8, framealpha=0.15, labelcolor=TEXT_PRIMARY)
        plt.xticks(rotation=30)
        set_dark_fig(fig, [ax])
        plt.tight_layout()
        st.pyplot(fig, use_container_width=True)
        plt.close(fig)

        s1, s2, s3, s4 = st.columns(4)
        s1.metric("Total Forecast (hari)", forecast_horizon)
        s2.metric("Avg Order/Hari (forecast)", f"{forecast_sim.mean():.1f}")
        s3.metric("Peak Day", dates[forecast_sim.argmax()].strftime("%d %b"))
        s4.metric("Min Order Day", dates[forecast_sim.argmin()].strftime("%d %b"))


# ════════════════════════════════════════════════════════════
# 7. TAB 3 — EDA TIME SERIES
# ════════════════════════════════════════════════════════════

def _normalize_status(status: str) -> str:
    if status == "Selesai":
        return "Completed"
    if status == "Batal":
        return "Cancelled"
    if str(status).startswith("Pesanan diterima"):
        return "Delivered"
    if status in ("Sedang Dikirim", "Telah Dikirim"):
        return "In Transit"
    return "Other"


def render_eda_tab(df_raw: Optional[pd.DataFrame]) -> None:
    st.markdown('<p class="section-title">Exploratory Data Analysis</p>', unsafe_allow_html=True)

    if df_raw is None:
        st.info(
            "Upload **all_months_clean.csv** di sidebar untuk melihat EDA "
            "(daily demand trend, heatmap jam, distribusi kategori)."
        )
        return

    col_map = {
        "Waktu Pesanan Dibuat": "order_time_raw",
        "Status Pesanan": "order_status_raw",
        "Total Pembayaran": "total_revenue",
        "Opsi Pengiriman": "shipping_option",
        "Metode Pembayaran": "payment_method",
        "Kota/Kabupaten": "city",
        "Total Diskon": "total_discount",
        "total_qty": "total_qty",
        "Perkiraan Ongkos Kirim": "estimated_shipping_cost",
    }
    df = df_raw.rename(columns={k: v for k, v in col_map.items() if k in df_raw.columns})

    if "order_time_raw" in df.columns:
        df["order_time"] = pd.to_datetime(df["order_time_raw"], errors="coerce")
        df = df.dropna(subset=["order_time"])
        df["hour"] = df["order_time"].dt.hour
        df["day_of_week"] = df["order_time"].dt.dayofweek
        df["date"] = df["order_time"].dt.date
        df["month"] = df["order_time"].dt.month
        df["year"] = df["order_time"].dt.year

    if "order_status_raw" in df.columns:
        df["order_status"] = df["order_status_raw"].apply(_normalize_status)
        completed = df[df["order_status"] == "Completed"].copy()
    else:
        completed = df.copy()

    if "date" in df.columns:
        count_col = "order_time_raw" if "order_time_raw" in completed.columns else completed.columns[0]
        daily_ts = (
            completed.groupby("date").agg(order_count=(count_col, "count")).reset_index()
        )
        daily_ts["date"] = pd.to_datetime(daily_ts["date"])

        fig, ax = plt.subplots(figsize=(12, 3.4))
        ax.plot(daily_ts["date"], daily_ts["order_count"], color=ACCENT, linewidth=0.9,
                alpha=0.55, label="Daily Orders")
        ax.plot(daily_ts["date"], daily_ts["order_count"].rolling(7).mean(),
                color="#f5793a", linewidth=1.8, label="7-Day MA")
        ax.plot(daily_ts["date"], daily_ts["order_count"].rolling(30).mean(),
                color="#3ecf8e", linewidth=1.8, linestyle="--", label="30-Day MA")
        mean_d = daily_ts["order_count"].mean()
        std_d = daily_ts["order_count"].std()
        spikes = daily_ts[daily_ts["order_count"] > mean_d + 2 * std_d]
        ax.scatter(spikes["date"], spikes["order_count"], color="#f5793a", zorder=5, s=35,
                   label="Demand Spike")
        ax.xaxis.set_major_locator(mdates.MonthLocator(interval=2))
        ax.xaxis.set_major_formatter(mdates.DateFormatter("%b %Y"))
        ax.set_title("Daily Order Demand", fontweight="600")
        ax.set_ylabel("Orders")
        ax.legend(fontsize=8, framealpha=0.15, labelcolor=TEXT_PRIMARY)
        plt.xticks(rotation=30)
        set_dark_fig(fig, [ax])
        plt.tight_layout()
        st.pyplot(fig, use_container_width=True)
        plt.close(fig)

    c_heat, c_monthly = st.columns(2)

    if "hour" in df.columns and "day_of_week" in df.columns:
        with c_heat:
            st.markdown("**Heatmap: Jam × Hari**")
            hour_dow = (
                completed.groupby(["day_of_week", "hour"]).size().unstack("hour").fillna(0)
            )
            hour_dow.index = ["Sen", "Sel", "Rab", "Kam", "Jum", "Sab", "Min"]
            fig2, ax2 = plt.subplots(figsize=(7, 3.2))
            sns.heatmap(hour_dow, cmap="Blues", linewidths=0.2,
                        cbar_kws={"label": "Orders", "shrink": 0.7},
                        ax=ax2, linecolor=BG_DARK)
            ax2.set_title("Demand Heatmap", fontweight="600")
            ax2.set_xlabel("Jam (0–23)")
            ax2.set_ylabel("")
            set_dark_fig(fig2, [ax2])
            fig2.patch.set_facecolor(BG_PANEL)
            plt.tight_layout()
            st.pyplot(fig2, use_container_width=True)
            plt.close(fig2)

    if "year" in completed.columns and "month" in completed.columns:
        with c_monthly:
            st.markdown("**Volume Order Bulanan**")
            monthly = completed.groupby(["year", "month"]).size().reset_index(name="order_count")
            monthly["period"] = pd.to_datetime(monthly[["year", "month"]].assign(day=1))
            fig3, ax3 = plt.subplots(figsize=(7, 3.2))
            year_colors = {2023: "#4f86f7", 2024: "#3ecf8e", 2025: "#f5793a"}
            bar_colors = [year_colors.get(y, TEXT_MUTED) for y in monthly["year"]]
            ax3.bar(monthly["period"], monthly["order_count"], width=20, color=bar_colors, alpha=0.88)
            ax3.xaxis.set_major_locator(mdates.MonthLocator(interval=3))
            ax3.xaxis.set_major_formatter(mdates.DateFormatter("%b\n%Y"))
            ax3.set_title("Monthly Order Volume", fontweight="600")
            ax3.set_ylabel("Orders")
            set_dark_fig(fig3, [ax3])
            plt.tight_layout()
            st.pyplot(fig3, use_container_width=True)
            plt.close(fig3)

    if "product_categories" in completed.columns and "total_revenue" in completed.columns:
        st.markdown("**Top 10 Kategori Produk · Revenue**")
        top_cat = (
            completed.groupby("product_categories")["total_revenue"]
            .sum().sort_values(ascending=False).head(10)
        )
        fig4, ax4 = plt.subplots(figsize=(12, 3.2))
        bars = ax4.barh(top_cat.index[::-1], top_cat.values[::-1] / 1e6,
                         color=ACCENT, alpha=0.82, edgecolor=BG_DARK)
        for bar, val in zip(bars, top_cat.values[::-1] / 1e6):
            ax4.text(bar.get_width() + 0.1, bar.get_y() + bar.get_height() / 2,
                      f"Rp {val:.1f}jt", va="center", ha="left",
                      fontsize=7.5, color=TEXT_MUTED, fontfamily="monospace")
        ax4.set_xlabel("Revenue (Rp Juta)")
        ax4.set_title("Revenue by Product Category", fontweight="600")
        ax4.spines[["top", "right"]].set_visible(False)
        set_dark_fig(fig4, [ax4])
        plt.tight_layout()
        st.pyplot(fig4, use_container_width=True)
        plt.close(fig4)


# ════════════════════════════════════════════════════════════
# 8. TAB 4 — BUSINESS IMPACT
# ════════════════════════════════════════════════════════════

def render_business_impact_tab(metrics_df: pd.DataFrame, best_model: str, worst_model: str) -> None:
    """
    Estimate operational savings from switching to the best model.

    Fix vs. the previous version: the "before" state used to be hard-coded
    to the first row of the metrics CSV, which in this dataset happens to
    *be* the best model — so every saving figure collapsed to Rp 0 no
    matter what the inputs were. The baseline is now an explicit, user-
    selectable model (defaulting to the worst-performing one), so the
    comparison is always meaningful and the cards respond to your inputs.
    """
    st.markdown('<p class="section-title">Estimasi Dampak Bisnis</p>', unsafe_allow_html=True)
    st.markdown("Sesuaikan asumsi di bawah untuk melihat estimasi penghematan operasional.")

    col_inp, col_res = st.columns([1, 2])

    with col_inp:
        avg_monthly_orders = st.number_input("Avg. Order/Bulan", value=660, step=10)
        avg_shipping_cost = st.number_input("Avg. Biaya Kirim/Order (Rp)", value=12000, step=500)
        cancelled_rate = st.slider("Cancelled Order Rate (%)", 0.0, 30.0, 13.8, 0.1)
        avg_revenue = st.number_input("Avg. Revenue/Order (Rp)", value=38000, step=1000)

        model_options = metrics_df.index.tolist()
        default_idx = model_options.index(worst_model) if worst_model in model_options else 0
        baseline_choice = st.selectbox(
            "Model Pembanding (Baseline)",
            options=model_options,
            index=default_idx,
            help="Model yang dianggap mewakili kondisi operasional saat ini, untuk "
                 "dibandingkan dengan model terbaik di atas.",
        )

        mape_before = float(metrics_df.loc[baseline_choice, "MAPE"])
        mape_after = float(metrics_df.loc[best_model, "MAPE"])

        st.markdown("<br>", unsafe_allow_html=True)
        st.markdown(
            insight_box(
                f"MAPE {baseline_choice} (baseline): <strong>{mape_before:.1f}%</strong> "
                f"→ MAPE {best_model} (terbaik): <strong>{mape_after:.1f}%</strong><br>"
                f"(diambil otomatis dari model_evaluation_results.csv)"
            ),
            unsafe_allow_html=True,
        )

    with col_res:
        if baseline_choice == best_model:
            st.markdown(
                warning_box(
                    "Model pembanding yang dipilih sama dengan model terbaik, sehingga "
                    "tidak ada selisih penghematan untuk dihitung. Pilih model lain "
                    "sebagai pembanding di panel sebelah kiri."
                ),
                unsafe_allow_html=True,
            )
            return

        # Dispatch waste: extra shipping cost incurred from forecast error
        wasted_before = avg_monthly_orders * (mape_before / 100) * avg_shipping_cost
        wasted_after = avg_monthly_orders * (mape_after / 100) * avg_shipping_cost
        monthly_save = wasted_before - wasted_after
        annual_save = monthly_save * 12

        # Stockout / overstock cost: rough heuristic of revenue at risk from
        # forecast error, scaled up by how often orders get cancelled
        # (a higher cancellation rate means demand mismatches bite harder).
        stockout_factor = 0.05 + (cancelled_rate / 100) * 0.10
        stockout_before = avg_monthly_orders * avg_revenue * (mape_before / 100) * stockout_factor
        stockout_after = avg_monthly_orders * avg_revenue * (mape_after / 100) * stockout_factor
        stockout_save = stockout_before - stockout_after

        delta_tone = "positive" if monthly_save >= 0 else "negative"

        b1, b2, b3 = st.columns(3)
        with b1:
            st.markdown(
                kpi_card("Penghematan Dispatch/Bulan", f"Rp {monthly_save:,.0f}",
                         f"{'+' if annual_save >= 0 else ''}Rp {annual_save:,.0f}/tahun", delta_tone),
                unsafe_allow_html=True,
            )
        with b2:
            st.markdown(
                kpi_card("Estimasi Hemat Stockout/Bulan", f"Rp {stockout_save:,.0f}"),
                unsafe_allow_html=True,
            )
        with b3:
            st.markdown(
                kpi_card("Total Estimasi Hemat/Tahun", f"Rp {(monthly_save + stockout_save) * 12:,.0f}"),
                unsafe_allow_html=True,
            )

        st.markdown("<br>", unsafe_allow_html=True)

        categories = ["Biaya Dispatch\n(Sebelum)", "Hemat Dispatch", "Hemat Stockout", "Total Sisa"]
        values_base = [wasted_before, -monthly_save, -stockout_save,
                        wasted_before - monthly_save - stockout_save]
        bottoms = [0, values_base[0], values_base[0] - monthly_save, 0]
        bar_clrs = ["#f5793a", "#3ecf8e", "#3ecf8e", ACCENT]

        fig5, ax5 = plt.subplots(figsize=(9, 3.6))
        bars5 = ax5.bar(categories, [abs(v) for v in values_base], bottom=bottoms,
                         color=bar_clrs, alpha=0.88, width=0.5, edgecolor=BG_DARK, linewidth=1)
        for bar, val in zip(bars5, values_base):
            ax5.text(bar.get_x() + bar.get_width() / 2, bar.get_y() + bar.get_height() / 2,
                      f"Rp {abs(val):,.0f}", ha="center", va="center",
                      fontsize=8, fontweight="600", color=BG_DARK, fontfamily="monospace")
        ax5.yaxis.set_major_formatter(
            mticker.FuncFormatter(lambda x, _: f"Rp {x/1000:.0f}K" if x >= 1000 else f"Rp {x:.0f}")
        )
        ax5.set_title("Estimasi Penghematan Biaya Operasional / Bulan", fontweight="600")
        ax5.spines[["top", "right"]].set_visible(False)
        set_dark_fig(fig5, [ax5])
        plt.tight_layout()
        st.pyplot(fig5, use_container_width=True)
        plt.close(fig5)

    render_business_impact_narrative(metrics_df, best_model, avg_monthly_orders, monthly_save)


def render_business_impact_narrative(metrics_df: pd.DataFrame, best_model: str,
                                      avg_monthly_orders: float, monthly_save: float) -> None:
    best_mae = metrics_df.loc[best_model, "MAE"]

    st.markdown("<br>", unsafe_allow_html=True)
    st.markdown('<p class="section-title">Ringkasan & Rekomendasi</p>', unsafe_allow_html=True)
    c_l, c_r = st.columns(2)
    with c_l:
        st.markdown(
            insight_box(
                f"Model <strong>{best_model}</strong> memberikan MAE terbaik "
                f"<strong>{best_mae:.2f} order/hari</strong>. Pada volume rata-rata "
                f"<strong>{avg_monthly_orders:,} order/bulan</strong>, peningkatan akurasi ini "
                f"menghasilkan estimasi penghematan biaya dispatching "
                f"<strong>Rp {monthly_save:,.0f}/bulan</strong>."
            ),
            unsafe_allow_html=True,
        )
    with c_r:
        st.markdown(
            warning_box(
                "MAPE semua model masih di atas 70% — demand spike pada promo date "
                "(Harbolnas, double-date) mendominasi error. Rekomendasi: tambahkan "
                "<strong>promo intensity feature</strong> dan pertimbangkan "
                "<strong>ensemble ARIMA + Prophet</strong> untuk mengurangi MAPE."
            ),
            unsafe_allow_html=True,
        )

    st.markdown("<br>", unsafe_allow_html=True)
    st.markdown('<p class="section-title">Panduan Pemilihan Model untuk Produksi</p>', unsafe_allow_html=True)

    guide_meta = {
        "ARIMA(7,1,2)": {
            "cocok": "Forecast harian rutin, infrastruktur minimal",
            "kekurangan": "Tidak menangani multiple seasonality",
            "status": "Siap deploy",
        },
        "Prophet": {
            "cocok": "Holiday effect, interpretasi bisnis",
            "kekurangan": "MAPE tinggi saat terjadi lonjakan ekstrem",
            "status": "Perlu tuning",
        },
        "LSTM": {
            "cocok": "Pola non-linear jangka panjang",
            "kekurangan": "Butuh data besar, kurang interpretable",
            "status": "Butuh lebih banyak data",
        },
    }

    rows = []
    for model in metrics_df.index:
        meta = guide_meta.get(model, {"cocok": "—", "kekurangan": "—", "status": "—"})
        rows.append({
            "Model": model,
            "MAE": metrics_df.loc[model, "MAE"],
            "MAPE (%)": metrics_df.loc[model, "MAPE"],
            "Cocok untuk": meta["cocok"],
            "Kekurangan": meta["kekurangan"],
            "Status Deploy": meta["status"],
        })
    st.dataframe(pd.DataFrame(rows), use_container_width=True, hide_index=True)


# ════════════════════════════════════════════════════════════
# 9. FOOTER
# ════════════════════════════════════════════════════════════

def render_footer() -> None:
    st.divider()
    st.markdown(
        "<p style='text-align:center;color:#4a5568;font-size:0.75rem;'>"
        "Phyto Collab · Demand Forecast Dashboard · Moch Yahya · Jun 2026"
        "</p>",
        unsafe_allow_html=True,
    )


# ════════════════════════════════════════════════════════════
# MAIN
# ════════════════════════════════════════════════════════════

def main() -> None:
    configure_page()
    inject_css()

    metrics_df, df_raw, forecast_horizon, _test_days = render_sidebar()

    if metrics_df is None:
        st.error(
            "**Data tidak ditemukan.**\n\n"
            "Letakkan file `model_evaluation_results.csv` di salah satu lokasi:\n"
            "- Direktori yang sama dengan script ini\n"
            "- `./data/model_evaluation_results.csv`\n"
            "- `/mnt/user-data/uploads/model_evaluation_results.csv`\n\n"
            "Atau upload file via sidebar untuk override."
        )
        st.stop()

    best_model, worst_model, *_ = render_header(metrics_df)

    tab1, tab2, tab3, tab4 = st.tabs([
        "Model Comparison",
        "Forecast Simulation",
        "EDA Time Series",
        "Business Impact",
    ])

    with tab1:
        render_model_comparison_tab(metrics_df, best_model)
    with tab2:
        render_forecast_simulation_tab(metrics_df, best_model, worst_model, forecast_horizon)
    with tab3:
        render_eda_tab(df_raw)
    with tab4:
        render_business_impact_tab(metrics_df, best_model, worst_model)

    render_footer()


if __name__ == "__main__":
    main()
