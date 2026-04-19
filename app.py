import streamlit as st
import pandas as pd
import numpy as np
import plotly.graph_objects as go
import plotly.express as px
from plotly.subplots import make_subplots
from sklearn.cluster import KMeans
from sklearn.linear_model import LinearRegression
from sklearn.preprocessing import StandardScaler
from sklearn.metrics import mean_squared_error, r2_score
from scipy import stats as scipy_stats

# ── Page config ────────────────────────────────────────────────────────────────
st.set_page_config(
    page_title="Spotify Hit Prediction",
    page_icon="🎵",
    layout="wide",
    initial_sidebar_state="expanded",
)

# ── Theme / CSS ────────────────────────────────────────────────────────────────
st.markdown("""
<style>
/* ---------- global ---------- */
html, body, [class*="css"] {
    font-family: 'Inter', 'Segoe UI', sans-serif;
}
.stApp { background: #0a0a0a; }

/* ---------- sidebar ---------- */
[data-testid="stSidebar"] {
    background: #111111 !important;
    border-right: 1px solid #1a1a1a;
}
[data-testid="stSidebar"] * { color: #e0e0e0 !important; }
[data-testid="stSidebar"] .stSelectbox label,
[data-testid="stSidebar"] .stMultiSelect label,
[data-testid="stSidebar"] .stSlider label { color: #a0a0a0 !important; font-size: 12px !important; }

/* ---------- main content ---------- */
[data-testid="stMainBlockContainer"] { padding-top: 1.5rem; padding-bottom: 2rem; }

/* ---------- KPI cards ---------- */
.kpi-card {
    background: #141414;
    border: 1px solid #1f1f1f;
    border-radius: 14px;
    padding: 18px 20px 14px;
    position: relative;
    overflow: hidden;
}
.kpi-card::before {
    content: '';
    position: absolute;
    top: 0; left: 0; right: 0;
    height: 3px;
    background: #1DB954;
    border-radius: 14px 14px 0 0;
}
.kpi-label { color: #7a7a7a; font-size: 12px; text-transform: uppercase; letter-spacing: 0.8px; margin-bottom: 6px; }
.kpi-value { color: #1DB954; font-size: 28px; font-weight: 700; line-height: 1; margin-bottom: 4px; }
.kpi-sub   { color: #4a4a4a; font-size: 11px; }

/* ---------- section headers ---------- */
.section-header {
    color: #ffffff;
    font-size: 15px;
    font-weight: 600;
    letter-spacing: 0.3px;
    margin: 0 0 4px 0;
}
.section-sub {
    color: #6a6a6a;
    font-size: 12px;
    margin-bottom: 16px;
}

/* ---------- chart cards ---------- */
.chart-card {
    background: #141414;
    border: 1px solid #1f1f1f;
    border-radius: 14px;
    padding: 20px;
    margin-bottom: 16px;
}

/* ---------- tab style ---------- */
[data-testid="stTabs"] [role="tablist"] {
    background: #111111;
    border-radius: 10px;
    padding: 4px;
    gap: 2px;
    border: 1px solid #1f1f1f;
}
[data-testid="stTabs"] button[role="tab"] {
    border-radius: 8px !important;
    color: #7a7a7a !important;
    font-size: 13px !important;
    font-weight: 500 !important;
    padding: 8px 16px !important;
    border: none !important;
}
[data-testid="stTabs"] button[role="tab"][aria-selected="true"] {
    background: #1DB954 !important;
    color: #000000 !important;
    font-weight: 600 !important;
}
[data-testid="stTabs"] button[role="tab"]:hover:not([aria-selected="true"]) {
    background: #1a1a1a !important;
    color: #ffffff !important;
}

/* ---------- selectbox / multiselect ---------- */
[data-testid="stSelectbox"] > div > div,
[data-testid="stMultiSelect"] > div > div {
    background: #1a1a1a !important;
    border: 1px solid #2a2a2a !important;
    border-radius: 8px !important;
    color: #e0e0e0 !important;
}

/* ---------- slider ---------- */
[data-testid="stSlider"] .stSlider > div > div { background: #1DB954 !important; }

/* ---------- metric delta hide ---------- */
[data-testid="stMetricDelta"] { display: none !important; }

/* ---------- dataframe ---------- */
[data-testid="stDataFrame"] { border-radius: 10px; overflow: hidden; }

/* ---------- stat table ---------- */
.stat-table {
    width: 100%;
    border-collapse: collapse;
    font-size: 13px;
}
.stat-table th {
    background: #1a1a1a;
    color: #7a7a7a;
    font-weight: 600;
    font-size: 11px;
    text-transform: uppercase;
    letter-spacing: 0.6px;
    padding: 10px 14px;
    text-align: left;
    border-bottom: 1px solid #2a2a2a;
}
.stat-table td {
    padding: 9px 14px;
    border-bottom: 1px solid #1a1a1a;
    color: #d0d0d0;
}
.stat-table tr:last-child td { border-bottom: none; }
.stat-table .hit-val  { color: #1DB954; font-weight: 600; }
.stat-table .nhit-val { color: #4a9eff; font-weight: 600; }

/* ---------- top header ---------- */
.dash-header {
    display: flex;
    align-items: center;
    gap: 16px;
    padding: 0 0 20px 0;
    border-bottom: 1px solid #1a1a1a;
    margin-bottom: 24px;
}
.spotify-badge {
    background: #1DB954;
    border-radius: 50%;
    width: 46px; height: 46px;
    display: flex; align-items: center; justify-content: center;
    font-size: 22px;
    flex-shrink: 0;
}
.dash-title { color: #ffffff; font-size: 22px; font-weight: 700; margin: 0; }
.dash-subtitle { color: #6a6a6a; font-size: 13px; margin: 2px 0 0 0; }

/* ---------- experiment badge ---------- */
.exp-badge {
    display: inline-block;
    background: rgba(29,185,84,0.15);
    color: #1DB954;
    border: 1px solid rgba(29,185,84,0.3);
    border-radius: 20px;
    font-size: 11px;
    font-weight: 600;
    padding: 3px 10px;
    margin-bottom: 12px;
    letter-spacing: 0.4px;
}

/* ---------- insight box ---------- */
.insight-box {
    background: rgba(29,185,84,0.07);
    border: 1px solid rgba(29,185,84,0.2);
    border-radius: 10px;
    padding: 14px 16px;
    color: #a0d4b0;
    font-size: 13px;
    line-height: 1.6;
    margin-top: 12px;
}
.insight-box strong { color: #1DB954; }
</style>
""", unsafe_allow_html=True)

# ── Plot defaults ──────────────────────────────────────────────────────────────
PLOT_BG    = "#141414"
PAPER_BG   = "#141414"
FONT_COLOR = "#c0c0c0"
GRID_COLOR = "#1e1e1e"
GREEN      = "#1DB954"
BLUE       = "#4a9eff"
RED        = "#ff4d4d"
PURPLE     = "#b48aff"
ORANGE     = "#ff944d"

CHART_COLORS = [GREEN, BLUE, PURPLE, ORANGE, "#ff6b9d", "#50e3c2"]

def base_layout(title="", height=340, **kwargs):
    return dict(
        title=dict(text=title, font=dict(color="#c0c0c0", size=13), x=0, pad=dict(l=0, t=4)),
        paper_bgcolor=PAPER_BG, plot_bgcolor=PLOT_BG,
        font=dict(color=FONT_COLOR, size=11),
        height=height, margin=dict(l=12, r=12, t=36 if title else 16, b=12),
        legend=dict(bgcolor="rgba(0,0,0,0)", borderwidth=0, font=dict(size=11)),
        **kwargs,
    )

def axis_style(title="", **kwargs):
    return dict(
        title=title, gridcolor=GRID_COLOR, zerolinecolor=GRID_COLOR,
        linecolor="#2a2a2a", tickcolor="#2a2a2a",
        title_font=dict(color="#7a7a7a", size=11),
        tickfont=dict(color="#7a7a7a", size=10),
        **kwargs,
    )

# ── Data loading ───────────────────────────────────────────────────────────────
@st.cache_data
def load_data():
    try:
        df = pd.read_csv("spotify_cleaned.csv")
    except FileNotFoundError:
        st.error("⚠️  `spotify_cleaned.csv` not found. Please place it in the same directory.")
        st.stop()
    except Exception as e:
        st.error(f"Error loading data: {e}")
        st.stop()

    df["streams"]  = pd.to_numeric(df["streams"],  errors="coerce")
    df["bpm"]      = pd.to_numeric(df["bpm"],      errors="coerce")
    df["released_year"] = pd.to_numeric(df["released_year"], errors="coerce")

    combined_col = "danceability_%_valence_%_energy_%_acousticness_%_instrumentalness_%_liveness_%_speechiness_%_"
    if combined_col in df.columns:
        sp = df[combined_col].str.split("_", expand=True)
        for i, col in enumerate(["danceability_%","valence_%","energy_%","acousticness_%",
                                  "instrumentalness_%","liveness_%","speechiness_%"]):
            df[col] = pd.to_numeric(sp[i], errors="coerce")

    pct_cols = ["danceability_%","valence_%","energy_%","acousticness_%",
                "instrumentalness_%","liveness_%","speechiness_%"]
    for col in pct_cols:
        if col in df.columns:
            df[col] = pd.to_numeric(df[col], errors="coerce")

    return df

df_raw = load_data()

# ── Sidebar ────────────────────────────────────────────────────────────────────
with st.sidebar:
    st.markdown("""
    <div style='text-align:center;padding:16px 0 20px'>
        <div style='background:#1DB954;border-radius:50%;width:52px;height:52px;
                    display:flex;align-items:center;justify-content:center;
                    font-size:26px;margin:0 auto 10px'>🎵</div>
        <div style='color:#ffffff;font-size:16px;font-weight:700'>Spotify Analytics</div>
        <div style='color:#555;font-size:11px;margin-top:3px'>Hit Song Prediction</div>
    </div>
    <hr style='border:none;border-top:1px solid #1f1f1f;margin:0 0 20px'>
    """, unsafe_allow_html=True)

    st.markdown("**Filters**")

    all_years  = sorted(df_raw["released_year"].dropna().astype(int).unique())
    year_range = st.slider("Release year", int(min(all_years)), int(max(all_years)),
                           (2010, int(max(all_years))), key="yr")

    all_modes  = sorted(df_raw["mode"].dropna().unique()) if "mode" in df_raw.columns else []
    sel_modes  = st.multiselect("Mode", all_modes, default=all_modes)

    hit_pct    = st.select_slider("Hit threshold (top %)", options=[10,15,20,25,30], value=20)

    bpm_range  = st.slider("BPM range", 50, 220, (60, 200), key="bpm")

    all_artists = sorted(df_raw["artist(s)_name"].dropna().unique()) \
                  if "artist(s)_name" in df_raw.columns else []
    sel_artists = st.multiselect("Artist (optional)", all_artists, default=[])

    st.markdown("<hr style='border:none;border-top:1px solid #1f1f1f;margin:20px 0'>",
                unsafe_allow_html=True)
    st.markdown("<div style='color:#444;font-size:11px;text-align:center'>Data Analytics Course<br>Experiments 1–4</div>",
                unsafe_allow_html=True)

# ── Filter data ────────────────────────────────────────────────────────────────
df = df_raw.copy()
df = df[(df["released_year"] >= year_range[0]) & (df["released_year"] <= year_range[1])]
if sel_modes:
    df = df[df["mode"].isin(sel_modes)]
if sel_artists:
    df = df[df["artist(s)_name"].isin(sel_artists)]
if "bpm" in df.columns:
    df = df[(df["bpm"] >= bpm_range[0]) & (df["bpm"] <= bpm_range[1])]

threshold = df["streams"].quantile(1 - hit_pct / 100)
df["is_hit"] = df["streams"] >= threshold

PCT_COLS = [c for c in ["danceability_%","energy_%","valence_%","acousticness_%",
                         "liveness_%","speechiness_%","instrumentalness_%"] if c in df.columns]

# ── Header ─────────────────────────────────────────────────────────────────────
st.markdown("""
<div class='dash-header'>
    <div class='spotify-badge'>🎵</div>
    <div>
        <p class='dash-title'>Spotify Music Data Analysis & Hit Song Prediction</p>
        <p class='dash-subtitle'>Exploring audio features · linear regression · clustering · box plots</p>
    </div>
</div>
""", unsafe_allow_html=True)

# ── KPI row ────────────────────────────────────────────────────────────────────
total_streams = df["streams"].sum()
avg_dance     = df["danceability_%"].mean() if "danceability_%" in df.columns else 0
avg_energy    = df["energy_%"].mean()       if "energy_%"       in df.columns else 0
avg_valence   = df["valence_%"].mean()      if "valence_%"      in df.columns else 0
hit_count     = df["is_hit"].sum()
total_tracks  = len(df)
hit_pct_actual = (hit_count / total_tracks * 100) if total_tracks else 0

k1, k2, k3, k4, k5 = st.columns(5)
for col, label, val, sub in [
    (k1, "Total Streams",    f"{total_streams/1e9:.1f} bn",  f"{total_tracks:,} tracks"),
    (k2, "Danceability",     f"{avg_dance:.1f}%",             "avg across tracks"),
    (k3, "Energy",           f"{avg_energy:.1f}%",            "avg across tracks"),
    (k4, "Valence",          f"{avg_valence:.1f}%",           "avg across tracks"),
    (k5, "Hit Songs",        f"{hit_count:,}",                f"{hit_pct_actual:.1f}% of total"),
]:
    with col:
        st.markdown(f"""
        <div class='kpi-card'>
            <div class='kpi-label'>{label}</div>
            <div class='kpi-value'>{val}</div>
            <div class='kpi-sub'>{sub}</div>
        </div>
        """, unsafe_allow_html=True)

st.markdown("<br>", unsafe_allow_html=True)

# ── Tabs ───────────────────────────────────────────────────────────────────────
tab_ov, tab_box, tab_reg, tab_clus, tab_detail = st.tabs([
    "📊  Overview",
    "📦  · Box Plot",
    "📈  · Regression",
    "🎯  · Clustering",
    "🔍  Analytics",
])

# ═══════════════════════════════════════════════════════════════════════════════
# TAB 1 – OVERVIEW
# ═══════════════════════════════════════════════════════════════════════════════
with tab_ov:
    st.markdown("<br>", unsafe_allow_html=True)
    col_left, col_right = st.columns([1, 1], gap="large")

    # Donut ────────────────────────────────────────────────────────────────────
    with col_left:
        st.markdown("<div class='chart-card'>", unsafe_allow_html=True)
        st.markdown("<p class='section-header'>Hit vs Non-Hit Distribution</p>", unsafe_allow_html=True)
        hit_c   = int(df["is_hit"].sum())
        nhit_c  = int((~df["is_hit"]).sum())
        fig_pie = go.Figure(go.Pie(
            labels=["Not Hit", "Hit"],
            values=[nhit_c, hit_c],
            hole=0.62,
            marker=dict(colors=["#1e1e1e", GREEN], line=dict(color=PLOT_BG, width=3)),
            textfont=dict(size=12, color="#c0c0c0"),
            textinfo="label+percent",
        ))
        fig_pie.add_annotation(
            text=f"<b>{hit_pct_actual:.1f}%</b><br><span style='font-size:10px'>Hit rate</span>",
            x=0.5, y=0.5, showarrow=False,
            font=dict(size=14, color=GREEN), align="center",
        )
        fig_pie.update_layout(**base_layout(height=290))
        st.plotly_chart(fig_pie, use_container_width=True, config={"displayModeBar": False})
        st.markdown("</div>", unsafe_allow_html=True)

    # Feature comparison grouped bar ──────────────────────────────────────────
    with col_right:
        st.markdown("<div class='chart-card'>", unsafe_allow_html=True)
        st.markdown("<p class='section-header'>Audio Features · Hit vs Non-Hit</p>", unsafe_allow_html=True)
        feat_cols  = [c for c in ["danceability_%","energy_%","valence_%","acousticness_%"] if c in df.columns]
        feat_names = [c.replace("_%","").replace("_"," ").title() for c in feat_cols]
        hit_avg    = df[df["is_hit"]][feat_cols].mean()
        nhit_avg   = df[~df["is_hit"]][feat_cols].mean()

        fig_feat = go.Figure()
        fig_feat.add_trace(go.Bar(
            name="Not Hit", x=feat_names, y=nhit_avg.values,
            marker_color=BLUE, marker_line_width=0,
        ))
        fig_feat.add_trace(go.Bar(
            name="Hit", x=feat_names, y=hit_avg.values,
            marker_color=GREEN, marker_line_width=0,
        ))
        fig_feat.update_layout(
            **base_layout(height=290),
            barmode="group", bargap=0.3, bargroupgap=0.08,
            xaxis=axis_style(), yaxis=axis_style("Value (%)", range=[0, 110]),
        )
        st.plotly_chart(fig_feat, use_container_width=True, config={"displayModeBar": False})
        st.markdown("</div>", unsafe_allow_html=True)

    # Streams by year ──────────────────────────────────────────────────────────
    st.markdown("<div class='chart-card'>", unsafe_allow_html=True)
    st.markdown("<p class='section-header'>Track Streams Over Time</p>", unsafe_allow_html=True)
    yr_data = df.groupby("released_year")["streams"].sum().reset_index().sort_values("released_year")
    fig_yr  = go.Figure()
    fig_yr.add_trace(go.Scatter(
        x=yr_data["released_year"], y=yr_data["streams"],
        mode="lines+markers",
        line=dict(color=GREEN, width=2.5, shape="spline"),
        marker=dict(size=5, color=GREEN),
        fill="tozeroy", fillcolor="rgba(29,185,84,0.08)",
    ))
    fig_yr.update_layout(
        **base_layout(height=240),
        xaxis=axis_style("Release year"), yaxis=axis_style("Total streams"),
    )
    st.plotly_chart(fig_yr, use_container_width=True, config={"displayModeBar": False})
    st.markdown("</div>", unsafe_allow_html=True)

    # Top artists ──────────────────────────────────────────────────────────────
    if "artist(s)_name" in df.columns:
        st.markdown("<div class='chart-card'>", unsafe_allow_html=True)
        st.markdown("<p class='section-header'>Popular Artists by Total Streams</p>", unsafe_allow_html=True)
        top_art = df.groupby("artist(s)_name")["streams"].sum().nlargest(12).reset_index()
        fig_art = go.Figure(go.Bar(
            x=top_art["streams"], y=top_art["artist(s)_name"],
            orientation="h",
            marker=dict(
                color=top_art["streams"],
                colorscale=[[0,"#0d3320"],[0.5,"#0f7a38"],[1,"#1DB954"]],
                line_width=0,
            ),
            text=[f"{v/1e9:.2f}bn" for v in top_art["streams"]],
            textposition="outside", textfont=dict(size=10, color="#7a7a7a"),
        ))
        fig_art.update_layout(
            **base_layout(height=max(320, len(top_art)*34)),
            xaxis=axis_style("Total streams"),
            yaxis=dict(autorange="reversed", tickfont=dict(color="#c0c0c0", size=11), linecolor="#2a2a2a"),
        )
        st.plotly_chart(fig_art, use_container_width=True, config={"displayModeBar": False})
        st.markdown("</div>", unsafe_allow_html=True)

# ═══════════════════════════════════════════════════════════════════════════════
# TAB 2 – BOX PLOT  (Experiment 1)
# ═══════════════════════════════════════════════════════════════════════════════
with tab_box:
    st.markdown("<br>", unsafe_allow_html=True)
    st.markdown("<div class='exp-badge'>1 · CO1</div>", unsafe_allow_html=True)
    st.markdown("""
    <p class='section-header'>Box Plot Analysis — Audio Feature Distributions</p>
    <p class='section-sub'>Compare the spread and central tendency of audio features between hit and non-hit songs.</p>
    """, unsafe_allow_html=True)

    if PCT_COLS:
        feat_sel = st.selectbox(
            "Select feature",
            PCT_COLS,
            format_func=lambda x: x.replace("_%","").replace("_"," ").title()
        )

        hit_vals  = df[df["is_hit"]][feat_sel].dropna()
        nhit_vals = df[~df["is_hit"]][feat_sel].dropna()

        col_box, col_stat = st.columns([3, 2], gap="large")

        with col_box:
            st.markdown("<div class='chart-card'>", unsafe_allow_html=True)
            fig_box = go.Figure()
            for vals, label, color in [
                (nhit_vals, "Not Hit", BLUE),
                (hit_vals,  "Hit",     GREEN),
            ]:
                fig_box.add_trace(go.Box(
                    y=vals, name=label,
                    marker=dict(color=color, size=4, outliercolor=color, opacity=0.6),
                    line=dict(color=color, width=2),
                    fillcolor=color.replace("ff","33") if "#" in color else f"rgba(29,185,84,0.12)",
                    boxpoints="outliers",
                    whiskerwidth=0.7,
                ))
            fig_box.update_layout(
                **base_layout(f"{feat_sel.replace('_%','').replace('_',' ').title()} Distribution", height=400),
                xaxis=dict(tickfont=dict(color="#c0c0c0", size=12), linecolor="#2a2a2a"),
                yaxis=axis_style("Value (%)"),
            )
            st.plotly_chart(fig_box, use_container_width=True, config={"displayModeBar": False})
            st.markdown("</div>", unsafe_allow_html=True)

        with col_stat:
            st.markdown("<div class='chart-card'>", unsafe_allow_html=True)
            st.markdown("<p class='section-header'>Statistical Summary</p>", unsafe_allow_html=True)

            def stat_row(label, h, n):
                return f"""<tr>
                    <td>{label}</td>
                    <td class='hit-val'>{h:.2f}</td>
                    <td class='nhit-val'>{n:.2f}</td>
                </tr>"""

            t_stat, p_val = scipy_stats.ttest_ind(hit_vals, nhit_vals, equal_var=False)
            table_html = f"""
            <table class='stat-table'>
                <thead><tr><th>Statistic</th><th style='color:#1DB954'>Hit</th><th style='color:#4a9eff'>Not Hit</th></tr></thead>
                <tbody>
                    {stat_row("Min",   hit_vals.min(),    nhit_vals.min())}
                    {stat_row("Q1",    hit_vals.quantile(0.25), nhit_vals.quantile(0.25))}
                    {stat_row("Median",hit_vals.median(), nhit_vals.median())}
                    {stat_row("Mean",  hit_vals.mean(),   nhit_vals.mean())}
                    {stat_row("Q3",    hit_vals.quantile(0.75), nhit_vals.quantile(0.75))}
                    {stat_row("Max",   hit_vals.max(),    nhit_vals.max())}
                    {stat_row("Std",   hit_vals.std(),    nhit_vals.std())}
                </tbody>
            </table>
            """
            st.markdown(table_html, unsafe_allow_html=True)

            sig = "✅ Statistically significant" if p_val < 0.05 else "⚠️ Not significant"
            st.markdown(f"""
            <div class='insight-box' style='margin-top:14px'>
                <strong>Welch's t-test</strong><br>
                t = {t_stat:.3f} &nbsp;|&nbsp; p = {p_val:.4f}<br>
                {sig} (α = 0.05)
            </div>
            """, unsafe_allow_html=True)
            st.markdown("</div>", unsafe_allow_html=True)

        # Violin supplement
        st.markdown("<div class='chart-card'>", unsafe_allow_html=True)
        st.markdown("<p class='section-header'>All Features · Violin Overview</p>", unsafe_allow_html=True)
        cols_v = PCT_COLS[:6]
        fig_v  = make_subplots(rows=1, cols=len(cols_v),
                               subplot_titles=[c.replace("_%","").replace("_"," ").title() for c in cols_v])
        for i, col_v in enumerate(cols_v, 1):
            for hit_flag, label, color in [(True,"Hit",GREEN),(False,"Not Hit",BLUE)]:
                fig_v.add_trace(
                    go.Violin(y=df[df["is_hit"]==hit_flag][col_v].dropna(),
                              name=label, legendgroup=label, showlegend=(i==1),
                              fillcolor=f"rgba(29,185,84,0.2)" if hit_flag else f"rgba(74,158,255,0.2)",
                              line_color=color, box_visible=True, meanline_visible=True,
                              points=False),
                    row=1, col=i,
                )
        fig_v.update_layout(**base_layout(height=300))
        fig_v.update_annotations(font_size=10, font_color="#7a7a7a")
        for i in range(1, len(cols_v)+1):
            fig_v.update_yaxes(gridcolor=GRID_COLOR, linecolor="#2a2a2a",
                               tickfont=dict(color="#7a7a7a", size=9), row=1, col=i)
            fig_v.update_xaxes(linecolor="#2a2a2a", tickfont=dict(color="#7a7a7a", size=9), row=1, col=i)
        st.plotly_chart(fig_v, use_container_width=True, config={"displayModeBar": False})
        st.markdown("</div>", unsafe_allow_html=True)

# ═══════════════════════════════════════════════════════════════════════════════
# TAB 3 – LINEAR REGRESSION  (Experiment 2)
# ═══════════════════════════════════════════════════════════════════════════════
with tab_reg:
    st.markdown("<br>", unsafe_allow_html=True)
    st.markdown("<div class='exp-badge'>2 · CO2</div>", unsafe_allow_html=True)
    st.markdown("""
    <p class='section-header'>Linear Regression — Predicting Stream Count</p>
    <p class='section-sub'>Predict song popularity using audio features via Ordinary Least Squares regression.</p>
    """, unsafe_allow_html=True)

    if len(PCT_COLS) >= 2:
        sel_feats = st.multiselect(
            "Features for regression",
            PCT_COLS,
            default=PCT_COLS[:6],
            format_func=lambda x: x.replace("_%","").replace("_"," ").title(),
        )

        if sel_feats:
            reg_df = df[sel_feats + ["streams"]].dropna()
            X = reg_df[sel_feats].values
            y = reg_df["streams"].values

            split = int(0.8 * len(X))
            X_tr, X_te = X[:split], X[split:]
            y_tr, y_te = y[:split], y[split:]

            model = LinearRegression()
            model.fit(X_tr, y_tr)
            y_pred = model.predict(X_te)

            mse = mean_squared_error(y_te, y_pred)
            r2  = r2_score(y_te, y_pred)
            rmse = np.sqrt(mse)

            # Metric cards
            m1, m2, m3, m4 = st.columns(4)
            for mc, lbl, val, sub in [
                (m1,"R² Score",        f"{r2:.4f}",                  "explained variance"),
                (m2,"RMSE",            f"{rmse/1e6:.2f}M",           "root mean sq. error"),
                (m3,"Training samples",f"{split:,}",                  "80% split"),
                (m4,"Test samples",    f"{len(y_te):,}",              "20% split"),
            ]:
                with mc:
                    st.markdown(f"""
                    <div class='kpi-card'>
                        <div class='kpi-label'>{lbl}</div>
                        <div class='kpi-value' style='font-size:22px'>{val}</div>
                        <div class='kpi-sub'>{sub}</div>
                    </div>
                    """, unsafe_allow_html=True)

            st.markdown("<br>", unsafe_allow_html=True)
            col_coef, col_pred = st.columns([1,1], gap="large")

            # Coefficients
            with col_coef:
                st.markdown("<div class='chart-card'>", unsafe_allow_html=True)
                st.markdown("<p class='section-header'>Feature Coefficients</p>", unsafe_allow_html=True)
                coef_df = pd.DataFrame({
                    "Feature": [c.replace("_%","").replace("_"," ").title() for c in sel_feats],
                    "Coefficient": model.coef_,
                }).sort_values("Coefficient")
                colors = [GREEN if v >= 0 else RED for v in coef_df["Coefficient"]]
                fig_coef = go.Figure(go.Bar(
                    x=coef_df["Coefficient"], y=coef_df["Feature"],
                    orientation="h",
                    marker=dict(color=colors, line_width=0),
                    text=[f"{v:+.2e}" for v in coef_df["Coefficient"]],
                    textposition="outside",
                    textfont=dict(size=10, color="#7a7a7a"),
                ))
                fig_coef.update_layout(
                    **base_layout(height=300),
                    xaxis=axis_style("Coefficient value"),
                    yaxis=dict(tickfont=dict(color="#c0c0c0", size=11), linecolor="#2a2a2a"),
                )
                fig_coef.add_vline(x=0, line_color="#2a2a2a", line_width=1)
                st.plotly_chart(fig_coef, use_container_width=True, config={"displayModeBar": False})
                st.markdown("</div>", unsafe_allow_html=True)

            # Predicted vs actual
            with col_pred:
                st.markdown("<div class='chart-card'>", unsafe_allow_html=True)
                st.markdown("<p class='section-header'>Predicted vs Actual Streams</p>", unsafe_allow_html=True)
                min_v, max_v = min(y_te.min(), y_pred.min()), max(y_te.max(), y_pred.max())
                fig_sc = go.Figure()
                fig_sc.add_trace(go.Scatter(
                    x=y_te, y=y_pred, mode="markers",
                    marker=dict(color=GREEN, size=5, opacity=0.55),
                    name="Predictions",
                ))
                fig_sc.add_trace(go.Scatter(
                    x=[min_v, max_v], y=[min_v, max_v], mode="lines",
                    line=dict(color=RED, dash="dot", width=1.5),
                    name="Perfect fit",
                ))
                fig_sc.update_layout(
                    **base_layout(height=300),
                    xaxis=axis_style("Actual streams"),
                    yaxis=axis_style("Predicted streams"),
                )
                st.plotly_chart(fig_sc, use_container_width=True, config={"displayModeBar": False})
                st.markdown("</div>", unsafe_allow_html=True)

            # Residuals
            st.markdown("<div class='chart-card'>", unsafe_allow_html=True)
            st.markdown("<p class='section-header'>Residual Distribution</p>", unsafe_allow_html=True)
            residuals = y_te - y_pred
            fig_res   = go.Figure()
            fig_res.add_trace(go.Histogram(
                x=residuals, nbinsx=40,
                marker=dict(color=GREEN, opacity=0.7, line=dict(color=PLOT_BG, width=0.5)),
                name="Residuals",
            ))
            fig_res.add_vline(x=0, line_color=RED, line_dash="dot", line_width=1.5)
            fig_res.update_layout(
                **base_layout(height=220),
                xaxis=axis_style("Residual value"),
                yaxis=axis_style("Count"),
            )
            st.plotly_chart(fig_res, use_container_width=True, config={"displayModeBar": False})
            st.markdown("</div>", unsafe_allow_html=True)

            st.markdown(f"""
            <div class='insight-box'>
                <strong>Interpretation:</strong> An R² of {r2:.4f} means audio features explain
                ~{r2*100:.1f}% of stream variance. Linear regression alone is limited here —
                hit prediction benefits significantly from artist popularity, playlist placement,
                and release timing signals beyond audio characteristics.
            </div>
            """, unsafe_allow_html=True)

# ═══════════════════════════════════════════════════════════════════════════════
# TAB 4 – CLUSTERING  (Experiment 4)
# ═══════════════════════════════════════════════════════════════════════════════
with tab_clus:
    st.markdown("<br>", unsafe_allow_html=True)
    st.markdown("<div class='exp-badge'>3 · CO3</div>", unsafe_allow_html=True)
    st.markdown("""
    <p class='section-header'>K-Means Clustering — Song Grouping by Audio Profile</p>
    <p class='section-sub'>Unsupervised grouping of songs using standardised audio features.</p>
    """, unsafe_allow_html=True)

    if len(PCT_COLS) >= 2:
        ctrl1, ctrl2, ctrl3 = st.columns([1, 1, 2])
        with ctrl1:
            n_k = st.slider("Number of clusters (k)", 2, 8, 3)
        with ctrl2:
            cx = st.selectbox("X axis", PCT_COLS, index=0,
                              format_func=lambda x: x.replace("_%","").replace("_"," ").title())
        with ctrl3:
            cy = st.selectbox("Y axis", PCT_COLS, index=1,
                              format_func=lambda x: x.replace("_%","").replace("_"," ").title())

        clus_df   = df[PCT_COLS + ["streams","is_hit"]].dropna().copy()
        scaler    = StandardScaler()
        X_sc      = scaler.fit_transform(clus_df[PCT_COLS])
        kmeans    = KMeans(n_clusters=n_k, random_state=42, n_init="auto")
        clus_df["cluster"] = kmeans.fit_predict(X_sc)

        # Cluster summary cards
        cols_k = st.columns(n_k)
        for i, c in enumerate(cols_k):
            sub = clus_df[clus_df["cluster"] == i]
            hr  = sub["is_hit"].mean() * 100
            with c:
                st.markdown(f"""
                <div class='kpi-card' style='border-top-color:{CHART_COLORS[i%len(CHART_COLORS)]}'>
                    <div class='kpi-label'>Cluster {i}</div>
                    <div class='kpi-value' style='color:{CHART_COLORS[i%len(CHART_COLORS)]};font-size:22px'>{len(sub):,}</div>
                    <div class='kpi-sub'>songs · {hr:.1f}% hit rate</div>
                </div>
                """, unsafe_allow_html=True)

        st.markdown("<br>", unsafe_allow_html=True)
        col_sc, col_hr = st.columns([3, 2], gap="large")

        with col_sc:
            st.markdown("<div class='chart-card'>", unsafe_allow_html=True)
            st.markdown("<p class='section-header'>Cluster Scatter</p>", unsafe_allow_html=True)
            fig_cl = go.Figure()
            for i in range(n_k):
                sub = clus_df[clus_df["cluster"] == i]
                color = CHART_COLORS[i % len(CHART_COLORS)]
                fig_cl.add_trace(go.Scatter(
                    x=sub[cx], y=sub[cy],
                    mode="markers",
                    name=f"Cluster {i}",
                    marker=dict(color=color, size=6, opacity=0.65, line=dict(width=0)),
                ))
            centroids = scaler.inverse_transform(kmeans.cluster_centers_)
            cx_idx = PCT_COLS.index(cx)
            cy_idx = PCT_COLS.index(cy)
            for i in range(n_k):
                color = CHART_COLORS[i % len(CHART_COLORS)]
                fig_cl.add_trace(go.Scatter(
                    x=[centroids[i][cx_idx]], y=[centroids[i][cy_idx]],
                    mode="markers", showlegend=False,
                    marker=dict(color=color, size=14, symbol="diamond",
                                line=dict(color="#ffffff", width=2)),
                ))
            fig_cl.update_layout(
                **base_layout(height=360),
                xaxis=axis_style(cx.replace("_%","").replace("_"," ").title()),
                yaxis=axis_style(cy.replace("_%","").replace("_"," ").title()),
            )
            st.plotly_chart(fig_cl, use_container_width=True, config={"displayModeBar": False})
            st.markdown("</div>", unsafe_allow_html=True)

        with col_hr:
            st.markdown("<div class='chart-card'>", unsafe_allow_html=True)
            st.markdown("<p class='section-header'>Hit Rate per Cluster</p>", unsafe_allow_html=True)
            hr_vals   = [clus_df[clus_df["cluster"]==i]["is_hit"].mean()*100 for i in range(n_k)]
            fig_hr    = go.Figure(go.Bar(
                x=[f"Cluster {i}" for i in range(n_k)],
                y=hr_vals,
                marker=dict(color=CHART_COLORS[:n_k], line_width=0),
                text=[f"{v:.1f}%" for v in hr_vals],
                textposition="outside",
                textfont=dict(size=11, color="#c0c0c0"),
            ))
            fig_hr.update_layout(
                **base_layout(height=260),
                xaxis=axis_style(), yaxis=axis_style("Hit rate (%)", range=[0, max(hr_vals)*1.3+1]),
            )
            st.plotly_chart(fig_hr, use_container_width=True, config={"displayModeBar": False})

            st.markdown("<p class='section-header' style='margin-top:14px'>Cluster Characteristics</p>",
                        unsafe_allow_html=True)
            radar_cols = [c for c in ["danceability_%","energy_%","valence_%","acousticness_%"] if c in PCT_COLS]
            stats_tbl  = clus_df.groupby("cluster")[radar_cols].mean().round(1)
            stats_tbl.index = [f"Cluster {i}" for i in stats_tbl.index]
            stats_tbl.columns = [c.replace("_%","").replace("_"," ").title() for c in stats_tbl.columns]
            st.dataframe(stats_tbl.style
                         .background_gradient(cmap="Greens", axis=None)
                         .format("{:.1f}"),
                         use_container_width=True)
            st.markdown("</div>", unsafe_allow_html=True)



# ═══════════════════════════════════════════════════════════════════════════════
# TAB 5 – ANALYTICS
# ═══════════════════════════════════════════════════════════════════════════════
with tab_detail:
    st.markdown("<br>", unsafe_allow_html=True)
    col_a, col_b = st.columns([3, 2], gap="large")

    with col_a:
        st.markdown("<div class='chart-card'>", unsafe_allow_html=True)
        st.markdown("<p class='section-header'>Top Tracks by Streams</p>", unsafe_allow_html=True)
        if "track_name" in df.columns:
            top_tr = df.nlargest(10, "streams")[["track_name","artist(s)_name","streams","released_year"]].copy()
            top_tr["streams_bn"] = (top_tr["streams"] / 1e9).round(2)
            fig_top = go.Figure(go.Bar(
                x=top_tr["streams_bn"],
                y=top_tr["track_name"],
                orientation="h",
                marker=dict(
                    color=top_tr["streams_bn"],
                    colorscale=[[0,"#0d3320"],[0.5,"#0f7a38"],[1,"#1DB954"]],
                    line_width=0,
                ),
                text=[f"{v}bn" for v in top_tr["streams_bn"]],
                textposition="outside",
                textfont=dict(size=10, color="#7a7a7a"),
            ))
            fig_top.update_layout(
                **base_layout(height=340),
                xaxis=axis_style("Streams (bn)"),
                yaxis=dict(autorange="reversed", tickfont=dict(color="#c0c0c0", size=11), linecolor="#2a2a2a"),
            )
            st.plotly_chart(fig_top, use_container_width=True, config={"displayModeBar": False})
        st.markdown("</div>", unsafe_allow_html=True)

    with col_b:
        st.markdown("<div class='chart-card'>", unsafe_allow_html=True)
        st.markdown("<p class='section-header'>BPM Distribution</p>", unsafe_allow_html=True)
        if "bpm" in df.columns:
            fig_bpm = go.Figure()
            fig_bpm.add_trace(go.Histogram(
                x=df["bpm"].dropna(), nbinsx=30,
                marker=dict(color=GREEN, opacity=0.75, line=dict(color=PLOT_BG, width=0.5)),
            ))
            fig_bpm.update_layout(
                **base_layout(height=200),
                xaxis=axis_style("BPM"), yaxis=axis_style("Count"),
            )
            st.plotly_chart(fig_bpm, use_container_width=True, config={"displayModeBar": False})
        st.markdown("</div>", unsafe_allow_html=True)

        st.markdown("<div class='chart-card'>", unsafe_allow_html=True)
        st.markdown("<p class='section-header'>Avg BPM · Hit vs Non-Hit</p>", unsafe_allow_html=True)
        if "bpm" in df.columns:
            bpm_grp = df.groupby("is_hit")["bpm"].mean()
            fig_bpm2 = go.Figure(go.Bar(
                x=["Not Hit", "Hit"],
                y=[bpm_grp.get(False, 0), bpm_grp.get(True, 0)],
                marker=dict(color=[BLUE, GREEN], line_width=0),
                text=[f"{bpm_grp.get(False,0):.1f}", f"{bpm_grp.get(True,0):.1f}"],
                textposition="outside",
                textfont=dict(size=11, color="#c0c0c0"),
            ))
            fig_bpm2.update_layout(
                **base_layout(height=180),
                xaxis=dict(tickfont=dict(color="#c0c0c0", size=12), linecolor="#2a2a2a"),
                yaxis=axis_style("Avg BPM"),
            )
            st.plotly_chart(fig_bpm2, use_container_width=True, config={"displayModeBar": False})
        st.markdown("</div>", unsafe_allow_html=True)

    # Correlation heatmap
    st.markdown("<div class='chart-card'>", unsafe_allow_html=True)
    st.markdown("<p class='section-header'>Feature Correlation Matrix</p>", unsafe_allow_html=True)
    num_cols = [c for c in PCT_COLS + ["streams","bpm"] if c in df.columns]
    if len(num_cols) > 1:
        corr = df[num_cols].corr()
        labels = [c.replace("_%","").replace("_"," ").title() for c in num_cols]
        fig_hm = go.Figure(go.Heatmap(
            z=corr.values, x=labels, y=labels,
            colorscale=[
                [0.0,"#4a9eff"],[0.4,"#1a1a2e"],[0.5,"#141414"],
                [0.6,"#0d3320"],[1.0,"#1DB954"],
            ],
            zmid=0, zmin=-1, zmax=1,
            text=corr.round(2).values,
            texttemplate="%{text}",
            textfont=dict(size=10, color="#c0c0c0"),
            colorbar=dict(tickfont=dict(color="#7a7a7a"), thickness=12, len=0.9),
        ))
        fig_hm.update_layout(
            **base_layout(height=380),
            xaxis=dict(tickfont=dict(color="#a0a0a0", size=10), linecolor="#2a2a2a", tickangle=-30),
            yaxis=dict(tickfont=dict(color="#a0a0a0", size=10), linecolor="#2a2a2a"),
        )
        st.plotly_chart(fig_hm, use_container_width=True, config={"displayModeBar": False})
    st.markdown("</div>", unsafe_allow_html=True)

    # Summary stats heatmap
    st.markdown("<div class='chart-card'>", unsafe_allow_html=True)
    st.markdown("<p class='section-header'>Summary Statistics Heatmap</p>", unsafe_allow_html=True)
    stat_cols = [c for c in PCT_COLS if c in df.columns]
    if stat_cols:
        summ = df[stat_cols].describe().round(2)
        summ.columns = [c.replace("_%","").replace("_"," ").title() for c in summ.columns]
        fig_sm = px.imshow(
            summ, text_auto=True,
            color_continuous_scale=["#0d3320","#1DB954"],
            aspect="auto",
        )
        fig_sm.update_layout(
            **base_layout(height=280),
            coloraxis_colorbar=dict(tickfont=dict(color="#7a7a7a"), thickness=10),
            xaxis=dict(tickfont=dict(color="#a0a0a0",size=10)),
            yaxis=dict(tickfont=dict(color="#a0a0a0",size=10)),
        )
        fig_sm.update_traces(textfont_size=10, textfont_color="#e0e0e0")
        st.plotly_chart(fig_sm, use_container_width=True, config={"displayModeBar": False})
    st.markdown("</div>", unsafe_allow_html=True)

# ── Footer ─────────────────────────────────────────────────────────────────────
st.markdown("""
<hr style='border:none;border-top:1px solid #1a1a1a;margin:32px 0 16px'>
<div style='text-align:center;color:#333;font-size:12px;padding-bottom:12px'>
    🎵 &nbsp; Spotify Music Data Analysis &amp; Hit Song Prediction &nbsp;·&nbsp;
</div>
""", unsafe_allow_html=True)
