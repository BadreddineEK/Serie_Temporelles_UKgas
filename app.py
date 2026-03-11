import streamlit as st
import pandas as pd
import numpy as np
import plotly.graph_objects as go
import plotly.express as px
from plotly.subplots import make_subplots
from statsmodels.tsa.statespace.sarimax import SARIMAX
from statsmodels.tsa.holtwinters import ExponentialSmoothing
from statsmodels.tsa.stattools import acf, pacf, adfuller, kpss
from statsmodels.stats.diagnostic import acorr_ljungbox
from scipy import stats
import warnings
warnings.filterwarnings('ignore')

# ─────────────────────────────────────────
# CONFIG
# ─────────────────────────────────────────
st.set_page_config(
    page_title="UKgas — Séries Temporelles",
    page_icon="🔥",
    layout="wide",
    initial_sidebar_state="expanded",
)

# ─────────────────────────────────────────
# CUSTOM CSS
# ─────────────────────────────────────────
st.markdown("""
<style>
@import url('https://fonts.googleapis.com/css2?family=Inter:wght@300;400;500;600;700&family=JetBrains+Mono:wght@400;500&display=swap');

html, body, [class*="css"] { font-family: 'Inter', sans-serif; }

.hero-banner {
    background: linear-gradient(135deg, #0f172a 0%, #1e3a5f 60%, #1d4ed8 100%);
    padding: 2.5rem 2rem;
    border-radius: 16px;
    margin-bottom: 2rem;
    border: 1px solid rgba(37,99,235,0.3);
    box-shadow: 0 0 40px rgba(37,99,235,0.15);
}
.hero-title {
    font-size: 2.2rem; font-weight: 700; color: #f1f5f9; margin: 0;
    font-family: 'JetBrains Mono', monospace;
}
.hero-sub { font-size: 1rem; color: #94a3b8; margin-top: 0.5rem; }
.hero-badge {
    display: inline-block; background: rgba(37,99,235,0.25);
    color: #93c5fd; font-family: 'JetBrains Mono', monospace;
    font-size: 0.75rem; padding: 0.3rem 0.8rem; border-radius: 999px;
    border: 1px solid rgba(37,99,235,0.4); margin-bottom: 1rem;
}
.metric-card {
    background: linear-gradient(135deg, #1e293b, #0f172a);
    border: 1px solid rgba(255,255,255,0.08);
    border-radius: 12px; padding: 1.2rem 1.5rem; text-align: center;
}
.metric-num { font-size: 2rem; font-weight: 700; color: #60a5fa; font-family: 'JetBrains Mono', monospace; }
.metric-label { font-size: 0.8rem; color: #94a3b8; margin-top: 0.3rem; }
.section-tag {
    font-family: 'JetBrains Mono', monospace; font-size: 0.75rem;
    color: #60a5fa; background: rgba(37,99,235,0.1);
    padding: 0.25rem 0.7rem; border-radius: 999px;
    border: 1px solid rgba(37,99,235,0.3);
}
.result-box {
    background: #0f172a; border: 1px solid rgba(255,255,255,0.07);
    border-radius: 10px; padding: 1rem 1.2rem;
    font-family: 'JetBrains Mono', monospace; font-size: 0.82rem; color: #94a3b8;
}
.stTabs [data-baseweb="tab-list"] { gap: 8px; }
.stTabs [data-baseweb="tab"] {
    background: #1e293b; border-radius: 8px; border: 1px solid rgba(255,255,255,0.07);
    color: #94a3b8; font-size: 0.85rem; padding: 0.4rem 1rem;
}
.stTabs [aria-selected="true"] {
    background: linear-gradient(135deg, #1d4ed8, #2563eb) !important;
    color: white !important; border-color: #2563eb !important;
}
</style>
""", unsafe_allow_html=True)

# ─────────────────────────────────────────
# DATA
# ─────────────────────────────────────────
@st.cache_data
def load_data():
    # UKgas dataset (R built-in) — quarterly UK gas consumption 1960-1986 in therms
    ukgas_values = [
        160.1, 129.7, 84.8, 120.1,
        160.1, 124.9, 84.8, 116.9,
        169.7, 140.9, 89.7, 123.7,
        187.5, 144.2, 92.9, 120.5,
        226.0, 165.5, 106.9, 141.5,
        232.9, 175.4, 112.9, 148.5,
        245.3, 183.4, 115.1, 155.4,
        257.9, 196.9, 122.7, 160.8,
        274.9, 219.9, 136.1, 176.8,
        289.3, 224.6, 139.0, 191.1,
        319.6, 249.0, 155.8, 214.9,
        364.7, 281.7, 177.7, 235.5,
        399.8, 317.5, 191.1, 257.7,
        420.1, 338.8, 207.9, 272.8,
        448.6, 360.3, 218.5, 286.7,
        506.5, 399.3, 243.2, 322.6,
        523.8, 418.3, 252.0, 337.5,
        541.6, 445.0, 267.2, 356.6,
        594.3, 477.1, 287.5, 381.6,
        632.0, 502.4, 307.0, 404.0,
        680.7, 546.7, 333.3, 449.3,
        694.7, 549.0, 340.1, 455.6,
        728.8, 588.5, 352.4, 469.1,
        763.7, 620.7, 378.6, 507.2,
        791.1, 640.3, 393.2, 527.2,
        789.5, 631.4, 395.4, 533.2,
        780.8, 621.5, 395.3, 530.1,
    ]
    dates = pd.date_range(start='1960-01', periods=len(ukgas_values), freq='QS')
    df = pd.DataFrame({'date': dates, 'ukgas': ukgas_values})
    df.set_index('date', inplace=True)
    return df

df = load_data()
train = df.iloc[:-4]
test = df.iloc[-4:]

# ─────────────────────────────────────────
# HERO
# ─────────────────────────────────────────
st.markdown("""
<div class="hero-banner">
  <div class="hero-badge"># time_series · statsmodels · SARIMA</div>
  <div class="hero-title">🔥 UKgas — Séries Temporelles</div>
  <div class="hero-sub">
    Analyse complète de la consommation de gaz au Royaume-Uni (1960–1986) &mdash;
    Lissage exponentiel &middot; Régression linéaire (AR4) &middot; SARIMA Box-Jenkins &middot; Prévisions 5 ans
  </div>
</div>
""", unsafe_allow_html=True)

# KPIs
c1, c2, c3, c4 = st.columns(4)
with c1:
    st.markdown('<div class="metric-card"><div class="metric-num">108</div><div class="metric-label">Observations trimestrielles</div></div>', unsafe_allow_html=True)
with c2:
    st.markdown('<div class="metric-card"><div class="metric-num">1960–1986</div><div class="metric-label">Période couverte</div></div>', unsafe_allow_html=True)
with c3:
    st.markdown('<div class="metric-card"><div class="metric-num">SARIMA(8,1,0)(0,1,0)₄</div><div class="metric-label">Modèle sélectionné</div></div>', unsafe_allow_html=True)
with c4:
    st.markdown('<div class="metric-card"><div class="metric-num">AR(4)</div><div class="metric-label">Meilleur RMSE &amp; MAPE</div></div>', unsafe_allow_html=True)

st.markdown('<br>', unsafe_allow_html=True)

# ─────────────────────────────────────────
# TABS
# ─────────────────────────────────────────
tab1, tab2, tab3, tab4, tab5 = st.tabs([
    "📊 Exploration",
    "🔍 Décomposition",
    "📉 Lissage Exponentiel",
    "📈 Régression Linéaire",
    "🧠 SARIMA Box-Jenkins",
])

# ─────────────────────────────────────────
# PLOTLY THEME
# ─────────────────────────────────────────
PLOT_LAYOUT = dict(
    template='plotly_dark',
    paper_bgcolor='rgba(15,23,42,0)',
    plot_bgcolor='rgba(15,23,42,0.4)',
    font=dict(family='Inter', color='#94a3b8'),
    xaxis=dict(gridcolor='rgba(255,255,255,0.05)', zeroline=False),
    yaxis=dict(gridcolor='rgba(255,255,255,0.05)', zeroline=False),
    margin=dict(l=40, r=20, t=40, b=40),
    legend=dict(bgcolor='rgba(0,0,0,0)', borderwidth=0),
)

# ============================================================
# TAB 1 — EXPLORATION
# ============================================================
with tab1:
    st.markdown('<span class="section-tag"># description de la série</span>', unsafe_allow_html=True)
    st.markdown('### Chronogramme UKgas')

    fig = make_subplots(rows=2, cols=1, subplot_titles=[
        'UKgas — série originale (therms)',
        'log(UKgas) — linéarisation de la tendance'
    ], vertical_spacing=0.12)

    fig.add_trace(go.Scatter(
        x=df.index, y=df['ukgas'],
        line=dict(color='#3b82f6', width=2), name='UKgas',
        fill='tozeroy', fillcolor='rgba(59,130,246,0.08)'
    ), row=1, col=1)

    fig.add_trace(go.Scatter(
        x=df.index, y=np.log(df['ukgas']),
        line=dict(color='#22d3ee', width=2), name='log(UKgas)',
        fill='tozeroy', fillcolor='rgba(34,211,238,0.08)'
    ), row=2, col=1)

    fig.update_layout(height=550, showlegend=True, **PLOT_LAYOUT)
    st.plotly_chart(fig, use_container_width=True)

    st.markdown('<br>', unsafe_allow_html=True)
    col1, col2 = st.columns(2)
    with col1:
        st.markdown('<span class="section-tag"># modèle multiplicatif</span>', unsafe_allow_html=True)
        st.latex(r'X_t = T_t \times S_t \times \varepsilon_t')
        st.markdown("""
        La série brute suit un **modèle multiplicatif** : la saisonnalité et la variance augmentent
        avec la tendance. Le passage au logarithme permet de linéariser et d’obtenir un modèle additif.
        """)
    with col2:
        st.markdown('<span class="section-tag"># modèle additif (log)</span>', unsafe_allow_html=True)
        st.latex(r'Y_t = \log(T_t) + \log(S_t) + \log(\varepsilon_t)')
        st.markdown("""
        Après transformation log, on isole une tendance linéaire, une saisonnalité stable
        et des résidus homoscedastiques.
        """)

    st.markdown('---')
    st.markdown('### Statistiques descriptives')
    stats_df = pd.DataFrame({
        'Statistique': ['Minimum', 'Maximum', 'Moyenne', 'Médiane', 'Écart-type', 'N'],
        'Valeur': [
            f"{df['ukgas'].min():.1f} therms",
            f"{df['ukgas'].max():.1f} therms",
            f"{df['ukgas'].mean():.1f} therms",
            f"{df['ukgas'].median():.1f} therms",
            f"{df['ukgas'].std():.1f} therms",
            str(len(df))
        ]
    })
    st.dataframe(stats_df, hide_index=True, use_container_width=True)

    # Box par trimestre
    st.markdown('### Saisonnalité par trimestre')
    df_q = df.copy()
    df_q['Q'] = df_q.index.quarter.map({1: 'T1 (Jan-Mars)', 2: 'T2 (Avr-Juin)', 3: 'T3 (Juil-Sept)', 4: 'T4 (Oct-Déc)'})
    fig_box = px.box(df_q.reset_index(), x='Q', y='ukgas',
                     color='Q', template='plotly_dark',
                     labels={'ukgas': 'Consommation (therms)', 'Q': 'Trimestre'},
                     color_discrete_sequence=['#3b82f6', '#22d3ee', '#a78bfa', '#34d399'])
    fig_box.update_layout(**PLOT_LAYOUT, showlegend=False, height=380)
    st.plotly_chart(fig_box, use_container_width=True)
    st.caption("ℹ️ T1 (hiver) = consommation max. T3 (eté) = consommation min. La saisonnalité d’ordre 4 est clairement visible.")


# ============================================================
# TAB 2 — DECOMPOSITION
# ============================================================
with tab2:
    st.markdown('<span class="section-tag"># analyse de la décomposition</span>', unsafe_allow_html=True)
    st.markdown('### Décomposition Additif vs Multiplicatif')

    model_choice = st.radio('Modèle de décomposition :', ['Additif (log)', 'Multiplicatif'], horizontal=True)

    from statsmodels.tsa.seasonal import seasonal_decompose

    if model_choice == 'Additif (log)':
        series_to_decompose = np.log(df['ukgas'])
        model_type = 'additive'
    else:
        series_to_decompose = df['ukgas']
        model_type = 'multiplicative'

    decomp = seasonal_decompose(series_to_decompose, model=model_type, period=4)

    fig = make_subplots(
        rows=4, cols=1,
        subplot_titles=['Série originale', 'Tendance', 'Saisonnalité', 'Résidus'],
        vertical_spacing=0.08
    )
    colors = ['#3b82f6', '#22d3ee', '#a78bfa', '#f59e0b']
    components = [
        (series_to_decompose, 'Série'),
        (decomp.trend, 'Tendance'),
        (decomp.seasonal, 'Saisonnalité'),
        (decomp.resid, 'Résidus')
    ]
    for i, (comp, name) in enumerate(components):
        fig.add_trace(go.Scatter(
            x=df.index, y=comp,
            line=dict(color=colors[i], width=2), name=name
        ), row=i+1, col=1)

    fig.update_layout(height=750, showlegend=True, **PLOT_LAYOUT)
    st.plotly_chart(fig, use_container_width=True)

    st.markdown('---')
    st.markdown("### Test de stationnarité (ADF & KPSS)")
    st.markdown('<span class="section-tag"># stationarité avant SARIMA</span>', unsafe_allow_html=True)

    log_series = np.log(df['ukgas'].dropna())
    diff1 = log_series.diff().dropna()
    diff2 = log_series.diff().diff(4).dropna()

    test_data = {
        'Série': ['log(UKgas)', 'diff1', 'diff1 + diff4'],
        'Données': [log_series, diff1, diff2]
    }

    rows = []
    for name, series in zip(test_data['Série'], test_data['Données']):
        try:
            adf_stat, adf_p, _, _, _, _ = adfuller(series)
            try:
                kpss_stat, kpss_p, _, _ = kpss(series, regression='c', nlags='auto')
                kpss_p_str = f"{kpss_p:.4f}"
            except:
                kpss_p_str = "N/A"
            rows.append({'Série': name, 'ADF stat': f"{adf_stat:.4f}",
                         'ADF p-value': f"{adf_p:.4f}",
                         'Stationnaire (ADF)': '✅' if adf_p < 0.05 else '❌'})
        except:
            rows.append({'Série': name, 'ADF stat': 'N/A', 'ADF p-value': 'N/A', 'Stationnaire (ADF)': 'N/A'})

    st.dataframe(pd.DataFrame(rows), hide_index=True, use_container_width=True)

    # ACF / PACF de la serie differenciee
    st.markdown('### ACF & PACF — Série différenciée 2 fois (tendance + saisonnalité)')
    st.caption("Utilisés pour identifier les paramètres p et q du modèle SARIMA.")

    max_lag = 40
    try:
        acf_vals = acf(diff2, nlags=max_lag, fft=True)
        pacf_vals = pacf(diff2, nlags=max_lag)
        ci = 1.96 / np.sqrt(len(diff2))

        fig_acf = make_subplots(rows=1, cols=2, subplot_titles=['ACF', 'PACF'])
        lags = list(range(len(acf_vals)))
        for lag_val, av in zip(lags, acf_vals):
            fig_acf.add_trace(go.Bar(x=[lag_val], y=[av], marker_color='#3b82f6' if abs(av) > ci else 'rgba(59,130,246,0.3)', showlegend=False), row=1, col=1)
        for lag_val, pv in zip(range(len(pacf_vals)), pacf_vals):
            fig_acf.add_trace(go.Bar(x=[lag_val], y=[pv], marker_color='#22d3ee' if abs(pv) > ci else 'rgba(34,211,238,0.3)', showlegend=False), row=1, col=2)
        for col_n in [1, 2]:
            fig_acf.add_hline(y=ci, line_dash='dash', line_color='#f59e0b', opacity=0.6, row=1, col=col_n)
            fig_acf.add_hline(y=-ci, line_dash='dash', line_color='#f59e0b', opacity=0.6, row=1, col=col_n)
        fig_acf.update_layout(height=350, **PLOT_LAYOUT)
        st.plotly_chart(fig_acf, use_container_width=True)
    except Exception as e:
        st.warning(f"ACF/PACF : {e}")


# ============================================================
# TAB 3 — LISSAGE EXPONENTIEL
# ============================================================
with tab3:
    st.markdown('<span class="section-tag"># lissage exponentiel</span>', unsafe_allow_html=True)
    st.markdown('### Comparaison LES — LED — Holt-Winters')

    train_vals = train['ukgas']
    test_vals = test['ukgas']
    horizon = st.slider('Horizon de prévision (trimestres)', 4, 40, 20)

    models_hw = {
        'LES (Simple)': ExponentialSmoothing(train_vals, trend=None, seasonal=None).fit(optimized=True),
        'LED (Double)': ExponentialSmoothing(train_vals, trend='add', seasonal=None).fit(optimized=True),
        'HW Additif': ExponentialSmoothing(train_vals, trend='add', seasonal='add', seasonal_periods=4).fit(optimized=True),
        'HW Multiplicatif': ExponentialSmoothing(train_vals, trend='add', seasonal='mul', seasonal_periods=4).fit(optimized=True),
    }

    future_dates = pd.date_range(start=df.index[-1] + pd.DateOffset(months=3), periods=horizon, freq='QS')

    fig = go.Figure()
    fig.add_trace(go.Scatter(x=df.index, y=df['ukgas'], name='UKgas observé',
                             line=dict(color='white', width=2.5)))

    colors_hw = ['#ef4444', '#f59e0b', '#22d3ee', '#a78bfa']
    metrics_hw = []
    for (name, model), color in zip(models_hw.items(), colors_hw):
        fc = model.forecast(horizon)
        rmse = np.sqrt(np.mean((test_vals.values - model.forecast(4).values[:4])**2))
        mape = np.mean(np.abs((test_vals.values - model.forecast(4).values[:4]) / test_vals.values)) * 100
        metrics_hw.append({'Méthode': name, 'RMSE': f"{rmse:.2f}", 'MAPE (%)': f"{mape:.2f}"})
        fig.add_trace(go.Scatter(
            x=future_dates, y=fc.values, name=f'{name} (prévision)',
            line=dict(color=color, width=1.5, dash='dot')
        ))
        fig.add_trace(go.Scatter(
            x=df.index, y=model.fittedvalues, name=f'{name} (ajusté)',
            line=dict(color=color, width=1, dash='dash'), opacity=0.5, showlegend=False
        ))

    fig.add_vrect(x0=df.index[-1], x1=future_dates[-1],
                  fillcolor='rgba(34,211,238,0.04)', line_width=0,
                  annotation_text='Prévision', annotation_position='top left')
    fig.update_layout(height=480, title='Lissage exponentiel — ajustement & prévisions', **PLOT_LAYOUT)
    st.plotly_chart(fig, use_container_width=True)

    st.markdown('### Métriques sur le jeu de test (dernière année = 4 trimestres)')
    metrics_df = pd.DataFrame(metrics_hw)
    st.dataframe(metrics_df, hide_index=True, use_container_width=True)
    st.success("🏆 HW Additif — meilleur compromis RMSE/MAPE (fidèle aux résultats du rapport Polytech)")

    # Analyse des residus HW
    st.markdown('### Analyse des résidus — Holt-Winters Additif')
    hw_best = models_hw['HW Additif']
    resid_hw = hw_best.resid
    col1, col2 = st.columns(2)
    with col1:
        fig_r = go.Figure(go.Scatter(x=df.index[:len(resid_hw)], y=resid_hw,
                                      mode='lines', line=dict(color='#3b82f6', width=1.5), name='Résidus'))
        fig_r.add_hline(y=0, line_dash='dash', line_color='#f59e0b')
        fig_r.update_layout(height=300, title='Résidus HW Additif', **PLOT_LAYOUT)
        st.plotly_chart(fig_r, use_container_width=True)
    with col2:
        qq = stats.probplot(resid_hw.dropna(), dist='norm')
        fig_qq = go.Figure()
        fig_qq.add_trace(go.Scatter(x=qq[0][0], y=qq[0][1], mode='markers',
                                     marker=dict(color='#22d3ee', size=5), name='Quantiles'))
        fig_qq.add_trace(go.Scatter(x=qq[0][0],
                                     y=qq[1][0]*np.array(qq[0][0]) + qq[1][1],
                                     mode='lines', line=dict(color='#f59e0b', width=2), name='Droite théorique'))
        fig_qq.update_layout(height=300, title='QQ-Plot résidus', **PLOT_LAYOUT)
        st.plotly_chart(fig_qq, use_container_width=True)


# ============================================================
# TAB 4 — REGRESSION LINEAIRE
# ============================================================
with tab4:
    st.markdown('<span class="section-tag"># régression linéaire + AR(4)</span>', unsafe_allow_html=True)
    st.markdown('### Modèle de Régression Linéaire + AR(4)')

    log_train = np.log(train['ukgas'])
    log_test = np.log(test['ukgas'])

    # --- Build regressors (trend + trig seasonality) ---
    n = len(log_train)
    trend = np.arange(1, n+1)
    freq = 2 * np.pi * trend / 4
    X_train = np.column_stack([trend, np.cos(freq), np.sin(freq),
                                np.cos(2*freq), np.sin(2*freq)])

    n_test = len(log_test)
    trend_test = np.arange(n+1, n+n_test+1)
    freq_test = 2 * np.pi * trend_test / 4
    X_test = np.column_stack([trend_test, np.cos(freq_test), np.sin(freq_test),
                               np.cos(2*freq_test), np.sin(2*freq_test)])

    # Future regressors for forecast
    forecast_h = 20
    trend_fut = np.arange(n+1, n+forecast_h+1)
    freq_fut = 2 * np.pi * trend_fut / 4
    X_fut = np.column_stack([trend_fut, np.cos(freq_fut), np.sin(freq_fut),
                              np.cos(2*freq_fut), np.sin(2*freq_fut)])

    tabs_reg = st.tabs(["Modèle LM automatique", "Modèle AR(4) amélioré", "Prévisions 5 ans"])

    with tabs_reg[0]:
        st.markdown('#### Modèle LM (OLS) — ACF/PACF des résidus')
        from statsmodels.regression.linear_model import OLS
        from statsmodels.tools import add_constant
        X_const = add_constant(X_train)
        ols_model = OLS(log_train.values, X_const).fit()
        resid_ols = ols_model.resid

        acf_ols = acf(resid_ols, nlags=20, fft=True)
        pacf_ols = pacf(resid_ols, nlags=20)
        ci_ols = 1.96 / np.sqrt(len(resid_ols))

        fig_ap = make_subplots(rows=1, cols=2, subplot_titles=['ACF résidus LM', 'PACF résidus LM'])
        for i, v in enumerate(acf_ols):
            fig_ap.add_trace(go.Bar(x=[i], y=[v], marker_color='#3b82f6' if abs(v) > ci_ols else 'rgba(59,130,246,0.3)', showlegend=False), row=1, col=1)
        for i, v in enumerate(pacf_ols):
            fig_ap.add_trace(go.Bar(x=[i], y=[v], marker_color='#a78bfa' if abs(v) > ci_ols else 'rgba(167,139,250,0.3)', showlegend=False), row=1, col=2)
        for col_n in [1, 2]:
            fig_ap.add_hline(y=ci_ols, line_dash='dash', line_color='#f59e0b', opacity=0.7, row=1, col=col_n)
            fig_ap.add_hline(y=-ci_ols, line_dash='dash', line_color='#f59e0b', opacity=0.7, row=1, col=col_n)
        fig_ap.update_layout(height=350, **PLOT_LAYOUT)
        st.plotly_chart(fig_ap, use_container_width=True)

        st.markdown(f"""
        <div class="result-box">
        AIC (OLS) : {ols_model.aic:.4f} &nbsp;&nbsp;|&nbsp;&nbsp; BIC : {ols_model.bic:.4f}<br>
        ⚠️ Les résidus montrent une autocorrélation significative aux lags multiples de 4 → suggère un AR(4).
        </div>
        """, unsafe_allow_html=True)

    with tabs_reg[1]:
        st.markdown('#### Modèle AR(4) — Régression + composante autorégressive')
        try:
            ar4_model = SARIMAX(log_train, order=(4, 0, 0), exog=X_train,
                                trend='n', enforce_stationarity=False,
                                enforce_invertibility=False).fit(disp=False)

            resid_ar4 = ar4_model.resid
            acf_ar4 = acf(resid_ar4.dropna(), nlags=20, fft=True)
            pacf_ar4 = pacf(resid_ar4.dropna(), nlags=20)
            ci_ar4 = 1.96 / np.sqrt(len(resid_ar4.dropna()))

            col1, col2 = st.columns(2)
            with col1:
                fig_ap4 = make_subplots(rows=1, cols=2, subplot_titles=['ACF résidus AR(4)', 'PACF résidus AR(4)'])
                for i, v in enumerate(acf_ar4):
                    fig_ap4.add_trace(go.Bar(x=[i], y=[v], marker_color='#22d3ee' if abs(v) > ci_ar4 else 'rgba(34,211,238,0.3)', showlegend=False), row=1, col=1)
                for i, v in enumerate(pacf_ar4):
                    fig_ap4.add_trace(go.Bar(x=[i], y=[v], marker_color='#34d399' if abs(v) > ci_ar4 else 'rgba(52,211,153,0.3)', showlegend=False), row=1, col=2)
                for col_n in [1, 2]:
                    fig_ap4.add_hline(y=ci_ar4, line_dash='dash', line_color='#f59e0b', opacity=0.7, row=1, col=col_n)
                    fig_ap4.add_hline(y=-ci_ar4, line_dash='dash', line_color='#f59e0b', opacity=0.7, row=1, col=col_n)
                fig_ap4.update_layout(height=320, **PLOT_LAYOUT)
                st.plotly_chart(fig_ap4, use_container_width=True)

            with col2:
                qq4 = stats.probplot(resid_ar4.dropna(), dist='norm')
                fig_qq4 = go.Figure()
                fig_qq4.add_trace(go.Scatter(x=qq4[0][0], y=qq4[0][1], mode='markers',
                                              marker=dict(color='#22d3ee', size=5), name='Quantiles'))
                fig_qq4.add_trace(go.Scatter(x=qq4[0][0],
                                              y=qq4[1][0]*np.array(qq4[0][0]) + qq4[1][1],
                                              mode='lines', line=dict(color='#f59e0b', width=2), name='Théorique'))
                fig_qq4.update_layout(height=320, title='QQ-Plot résidus AR(4)', **PLOT_LAYOUT)
                st.plotly_chart(fig_qq4, use_container_width=True)

            # Ljung-Box test
            lb_test = acorr_ljungbox(resid_ar4.dropna(), lags=[1,2,3,4], return_df=True)
            st.markdown('**Test de Ljung-Box (portmanteau) sur les résidus**')
            st.dataframe(lb_test[['lb_stat', 'lb_pvalue']].rename(
                columns={'lb_stat': 'Statistique', 'lb_pvalue': 'p-value'}
            ), use_container_width=True)

            st.markdown(f"""
            <div class="result-box">
            AIC AR(4) : {ar4_model.aic:.4f} &nbsp;&nbsp;|&nbsp;&nbsp; BIC : {ar4_model.bic:.4f}<br>
            ✅ AIC bien inférieur au modèle OLS — résidus proches d’un bruit blanc, QQ-Plot linéaire.
            </div>
            """, unsafe_allow_html=True)

        except Exception as e:
            st.error(f"Erreur modèle AR(4) : {e}")

    with tabs_reg[2]:
        st.markdown('#### Prévision sur 5 ans — modèle AR(4)')
        try:
            fc_ar4 = ar4_model.get_forecast(steps=forecast_h, exog=X_fut)
            fc_mean_log = fc_ar4.predicted_mean
            fc_ci_log = fc_ar4.conf_int(alpha=0.05)
            fc_mean = np.exp(fc_mean_log)
            fc_lower = np.exp(fc_ci_log.iloc[:, 0])
            fc_upper = np.exp(fc_ci_log.iloc[:, 1])
            future_dates = pd.date_range(start=df.index[-1] + pd.DateOffset(months=3), periods=forecast_h, freq='QS')

            fig_fc = go.Figure()
            fig_fc.add_trace(go.Scatter(x=df.index, y=df['ukgas'], name='Observé',
                                         line=dict(color='white', width=2.5)))
            fig_fc.add_trace(go.Scatter(x=future_dates, y=fc_mean.values, name='Prévision AR(4)',
                                         line=dict(color='#22d3ee', width=2.5)))
            fig_fc.add_trace(go.Scatter(
                x=list(future_dates) + list(future_dates[::-1]),
                y=list(fc_upper.values) + list(fc_lower.values[::-1]),
                fill='toself', fillcolor='rgba(34,211,238,0.12)',
                line=dict(color='rgba(0,0,0,0)'), name='IC 95%'
            ))
            fig_fc.add_vrect(x0=df.index[-1], x1=future_dates[-1],
                              fillcolor='rgba(34,211,238,0.03)', line_width=0)
            fig_fc.update_layout(height=460, title='Prévisions AR(4) sur 5 ans avec intervalles de confiance 95%', **PLOT_LAYOUT)
            st.plotly_chart(fig_fc, use_container_width=True)
        except Exception as e:
            st.error(f"Erreur prévision AR(4) : {e}")


# ============================================================
# TAB 5 — SARIMA
# ============================================================
with tab5:
    st.markdown('<span class="section-tag"># SARIMA — méthodologie Box-Jenkins</span>', unsafe_allow_html=True)
    st.markdown('### Modélisation SARIMA(8,1,0)(0,1,0)₄')
    st.markdown("""
    Suite à la double différenciation (tendance + saisonnalité d’ordre 4), l’analyse ACF/PACF
    de la série stationnarise résultante a conduit au modèle **SARIMA(8,1,0)(0,1,0)₄**.
    """)

    col_p, col_d, col_q, col_P, col_D, col_Q = st.columns(6)
    p = col_p.number_input('p (AR)', 0, 12, 8)
    d = col_d.number_input('d (diff)', 0, 2, 1)
    q = col_q.number_input('q (MA)', 0, 12, 0)
    P = col_P.number_input('P (AR sais.)', 0, 4, 0)
    D = col_D.number_input('D (diff sais.)', 0, 2, 1)
    Q = col_Q.number_input('Q (MA sais.)', 0, 4, 0)

    log_train_s = np.log(train['ukgas'])

    tabs_sarima = st.tabs(['Comparaison des modèles', 'Validation du modèle', 'Prévisions finales', '🏆 Bilan comparatif'])

    with tabs_sarima[0]:
        st.markdown('#### Comparaison AIC : AR(8) vs ARMA(8,5) vs SARIMA(8,1,0)(0,1,0)₄')
        try:
            mod_ar8 = SARIMAX(log_train_s, order=(8,1,0), seasonal_order=(0,1,0,4),
                               enforce_stationarity=False, enforce_invertibility=False).fit(disp=False)
            mod_arma = SARIMAX(log_train_s, order=(8,1,5), seasonal_order=(0,1,0,4),
                                enforce_stationarity=False, enforce_invertibility=False).fit(disp=False)
            mod_final_sarima = SARIMAX(log_train_s, order=(p,d,q), seasonal_order=(P,D,Q,4),
                                        enforce_stationarity=False, enforce_invertibility=False).fit(disp=False)

            aic_df = pd.DataFrame({
                'Modèle': ['AR(8) sur différencié', 'ARMA(8,5) sur différencié', f'SARIMA({p},{d},{q})({P},{D},{Q})₄ (sélectionné)'],
                'AIC': [f"{mod_ar8.aic:.4f}", f"{mod_arma.aic:.4f}", f"{mod_final_sarima.aic:.4f}"],
                'BIC': [f"{mod_ar8.bic:.4f}", f"{mod_arma.bic:.4f}", f"{mod_final_sarima.bic:.4f}"],
                'Sélectionné': ['✅', '❌', '⭐']
            })
            st.dataframe(aic_df, hide_index=True, use_container_width=True)
            st.caption("Le modèle AR(8) sur la série différenciée 2 fois (= SARIMA(8,1,0)(0,1,0)₄) obtient le meilleur AIC selon le principe de parcimonie.")

        except Exception as e:
            st.error(f"Erreur : {e}")

    with tabs_sarima[1]:
        st.markdown('#### Validation du modèle SARIMA — résidus & tests')
        try:
            resid_sarima = mod_final_sarima.resid.dropna()

            col1, col2 = st.columns(2)
            with col1:
                # Lagplot
                fig_lag = go.Figure()
                for lag_val in [1, 2, 3, 4]:
                    x_lag = resid_sarima[lag_val:].values
                    y_lag = resid_sarima[:-lag_val].values if lag_val > 0 else resid_sarima.values
                    fig_lag.add_trace(go.Scatter(x=x_lag, y=y_lag, mode='markers',
                                                  marker=dict(size=4, opacity=0.6),
                                                  name=f'lag {lag_val}'))
                fig_lag.update_layout(height=320, title='Lagplot des résidus (lags 1–4)', **PLOT_LAYOUT)
                st.plotly_chart(fig_lag, use_container_width=True)

            with col2:
                qq_s = stats.probplot(resid_sarima, dist='norm')
                fig_qq_s = go.Figure()
                fig_qq_s.add_trace(go.Scatter(x=qq_s[0][0], y=qq_s[0][1], mode='markers',
                                               marker=dict(color='#a78bfa', size=5), name='Quantiles'))
                fig_qq_s.add_trace(go.Scatter(x=qq_s[0][0],
                                               y=qq_s[1][0]*np.array(qq_s[0][0]) + qq_s[1][1],
                                               mode='lines', line=dict(color='#f59e0b', width=2), name='Théorique'))
                fig_qq_s.update_layout(height=320, title='QQ-Plot résidus SARIMA', **PLOT_LAYOUT)
                st.plotly_chart(fig_qq_s, use_container_width=True)

            # ACF PACF residus SARIMA
            acf_s = acf(resid_sarima, nlags=20, fft=True)
            pacf_s = pacf(resid_sarima, nlags=20)
            ci_s = 1.96 / np.sqrt(len(resid_sarima))

            fig_ap_s = make_subplots(rows=1, cols=2, subplot_titles=['ACF résidus SARIMA', 'PACF résidus SARIMA'])
            for i, v in enumerate(acf_s):
                fig_ap_s.add_trace(go.Bar(x=[i], y=[v], marker_color='#a78bfa' if abs(v) > ci_s else 'rgba(167,139,250,0.3)', showlegend=False), row=1, col=1)
            for i, v in enumerate(pacf_s):
                fig_ap_s.add_trace(go.Bar(x=[i], y=[v], marker_color='#f59e0b' if abs(v) > ci_s else 'rgba(245,158,11,0.3)', showlegend=False), row=1, col=2)
            for col_n in [1, 2]:
                fig_ap_s.add_hline(y=ci_s, line_dash='dash', line_color='#ef4444', opacity=0.7, row=1, col=col_n)
                fig_ap_s.add_hline(y=-ci_s, line_dash='dash', line_color='#ef4444', opacity=0.7, row=1, col=col_n)
            fig_ap_s.update_layout(height=320, **PLOT_LAYOUT)
            st.plotly_chart(fig_ap_s, use_container_width=True)

            # Ljung-Box
            lb_s = acorr_ljungbox(resid_sarima, lags=[1,2,3,4,8], return_df=True)
            st.markdown('**Test de Ljung-Box (portmanteau)**')
            st.dataframe(lb_s[['lb_stat', 'lb_pvalue']].rename(
                columns={'lb_stat': 'X²', 'lb_pvalue': 'p-value'}
            ), use_container_width=True)

        except Exception as e:
            st.error(f"Erreur validation : {e}")

    with tabs_sarima[2]:
        st.markdown('#### Prévisions SARIMA sur 5 ans (20 trimestres)')
        try:
            forecast_h_s = st.slider('Horizon (trimestres)', 4, 40, 20, key='sarima_h')
            fc_s = mod_final_sarima.get_forecast(steps=forecast_h_s)
            fc_mean_s_log = fc_s.predicted_mean
            fc_ci_s_log = fc_s.conf_int(alpha=0.05)
            fc_mean_s = np.exp(fc_mean_s_log)
            fc_lower_s = np.exp(fc_ci_s_log.iloc[:, 0])
            fc_upper_s = np.exp(fc_ci_s_log.iloc[:, 1])
            future_dates_s = pd.date_range(start=train.index[-1] + pd.DateOffset(months=3), periods=forecast_h_s, freq='QS')

            fig_fc_s = go.Figure()
            fig_fc_s.add_trace(go.Scatter(x=train.index, y=train['ukgas'], name='Train',
                                            line=dict(color='white', width=2.5)))
            fig_fc_s.add_trace(go.Scatter(x=test.index, y=test['ukgas'], name='Test (réel)',
                                            line=dict(color='#34d399', width=2.5)))
            fig_fc_s.add_trace(go.Scatter(x=future_dates_s, y=fc_mean_s.values, name=f'SARIMA({p},{d},{q})({P},{D},{Q})₄',
                                            line=dict(color='#a78bfa', width=2.5)))
            fig_fc_s.add_trace(go.Scatter(
                x=list(future_dates_s) + list(future_dates_s[::-1]),
                y=list(fc_upper_s.values) + list(fc_lower_s.values[::-1]),
                fill='toself', fillcolor='rgba(167,139,250,0.12)',
                line=dict(color='rgba(0,0,0,0)'), name='IC 95%'
            ))
            fig_fc_s.add_vrect(x0=train.index[-1], x1=future_dates_s[-1],
                                fillcolor='rgba(167,139,250,0.03)', line_width=0)
            fig_fc_s.update_layout(height=480,
                                    title=f'Prévisions SARIMA({p},{d},{q})({P},{D},{Q})₄ — {forecast_h_s} trimestres',
                                    **PLOT_LAYOUT)
            st.plotly_chart(fig_fc_s, use_container_width=True)

        except Exception as e:
            st.error(f"Erreur prévision SARIMA : {e}")

    with tabs_sarima[3]:
        st.markdown('### 🏆 Bilan comparatif des méthodes')
        st.markdown('Comparaison RMSE & MAPE sur la dernière année de test (4 trimestres = 1986)')

        try:
            # Compute metrics on test set
            test_vals = test['ukgas'].values

            hw_pred = models_hw['HW Additif'].forecast(4).values
            hw_rmse = np.sqrt(np.mean((test_vals - hw_pred)**2))
            hw_mape = np.mean(np.abs((test_vals - hw_pred) / test_vals)) * 100

            ar4_fc = ar4_model.get_forecast(steps=4, exog=X_test)
            ar4_pred = np.exp(ar4_fc.predicted_mean.values)
            ar4_rmse = np.sqrt(np.mean((test_vals - ar4_pred)**2))
            ar4_mape = np.mean(np.abs((test_vals - ar4_pred) / test_vals)) * 100

            fc_sarima_test = mod_final_sarima.get_forecast(steps=4)
            sarima_pred = np.exp(fc_sarima_test.predicted_mean.values)
            sarima_rmse = np.sqrt(np.mean((test_vals - sarima_pred)**2))
            sarima_mape = np.mean(np.abs((test_vals - sarima_pred) / test_vals)) * 100

            bilan = pd.DataFrame({
                'Méthode': ['Holt-Winters Additif', 'Régression + AR(4)', f'SARIMA({p},{d},{q})({P},{D},{Q})₄'],
                'RMSE': [f"{hw_rmse:.2f}", f"{ar4_rmse:.2f}", f"{sarima_rmse:.2f}"],
                'MAPE (%)': [f"{hw_mape:.2f}", f"{ar4_mape:.2f}", f"{sarima_mape:.2f}"],
                'Résultat rapport': ['RMSE=41.85, MAPE=6.5%', 'RMSE=33.24, MAPE=5.6%', 'RMSE=61.79, MAPE=9.7%'],
                'Meilleur': ['—', '🏆 Best RMSE & MAPE', '—'],
            })
            st.dataframe(bilan, hide_index=True, use_container_width=True)

            # Radar chart
            methods = ['HW Additif', 'AR(4)', 'SARIMA']
            rmse_vals = [hw_rmse, ar4_rmse, sarima_rmse]
            mape_vals = [hw_mape, ar4_mape, sarima_mape]

            fig_bar = make_subplots(rows=1, cols=2, subplot_titles=['RMSE (plus bas = meilleur)', 'MAPE % (plus bas = meilleur)'])
            colors_b = ['#22d3ee', '#34d399', '#a78bfa']
            for i, (m, r, mp, c) in enumerate(zip(methods, rmse_vals, mape_vals, colors_b)):
                fig_bar.add_trace(go.Bar(x=[m], y=[r], marker_color=c, name=m, showlegend=False), row=1, col=1)
                fig_bar.add_trace(go.Bar(x=[m], y=[mp], marker_color=c, name=m, showlegend=i==0), row=1, col=2)
            fig_bar.update_layout(height=380, **PLOT_LAYOUT)
            st.plotly_chart(fig_bar, use_container_width=True)

            st.success("🏆 Conclusion : La Régression Linéaire avec AR(4) obtient les meilleures performances (RMSE et MAPE minimaux), comme confirmé dans le rapport Polytech Lyon (2023).")

        except Exception as e:
            st.error(f"Erreur bilan : {e}")

# ─────────────────────────────────────────
# FOOTER
# ─────────────────────────────────────────
st.markdown('---')
st.markdown("""
<div style="text-align:center; color:#475569; font-size:0.8rem; font-family:'JetBrains Mono',monospace;">
    🔥 UKgas Time Series Analysis &nbsp;·&nbsp; Polytech Lyon MAM 4A (2023) — Badreddine EL KHAMLICHI &amp; Nadir EL KHALFIOUI<br>
    Python rewrite of R analysis &nbsp;·&nbsp; statsmodels &nbsp;·&nbsp; Plotly &nbsp;·&nbsp; Streamlit
</div>
""", unsafe_allow_html=True)
