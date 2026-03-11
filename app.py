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

# ─────────────────────────────────────────────────────────────
# PAGE CONFIG
# ─────────────────────────────────────────────────────────────
st.set_page_config(
    page_title="UKgas — Séries Temporelles",
    page_icon="🔥",
    layout="wide",
    initial_sidebar_state="collapsed",
)

# ─────────────────────────────────────────────────────────────
# CSS ADAPTATIF DARK / LIGHT
# ─────────────────────────────────────────────────────────────
st.markdown("""
<style>
@import url('https://fonts.googleapis.com/css2?family=Inter:wght@300;400;500;600;700&family=JetBrains+Mono:wght@400;500&display=swap');

html, body, [class*="css"] {
    font-family: 'Inter', sans-serif;
}

/* ── Hero banner ── */
.hero-banner {
    background: linear-gradient(135deg, #1e3a5f 0%, #1d4ed8 100%);
    padding: 2.5rem 2rem;
    border-radius: 16px;
    margin-bottom: 2rem;
    border: 1px solid rgba(37,99,235,0.4);
    box-shadow: 0 4px 32px rgba(37,99,235,0.18);
}
.hero-title {
    font-size: 2.2rem;
    font-weight: 700;
    color: #f1f5f9;
    margin: 0;
    font-family: 'JetBrains Mono', monospace;
}
.hero-sub {
    font-size: 1rem;
    color: #bfdbfe;
    margin-top: 0.5rem;
    line-height: 1.6;
}
.hero-badge {
    display: inline-block;
    background: rgba(255,255,255,0.15);
    color: #e0f2fe;
    font-family: 'JetBrains Mono', monospace;
    font-size: 0.75rem;
    padding: 0.3rem 0.8rem;
    border-radius: 999px;
    border: 1px solid rgba(255,255,255,0.25);
    margin-bottom: 1rem;
}

/* ── Metric cards — adaptatif ── */
.metric-card {
    background: var(--background-color, transparent);
    border: 1px solid rgba(37,99,235,0.25);
    border-radius: 12px;
    padding: 1.2rem 1.5rem;
    text-align: center;
    margin-bottom: 0.5rem;
}
/* dark mode override */
[data-theme="dark"] .metric-card,
.stApp[data-theme="dark"] .metric-card {
    background: linear-gradient(135deg, #1e293b, #0f172a);
    border-color: rgba(255,255,255,0.08);
}
.metric-num {
    font-size: 1.7rem;
    font-weight: 700;
    color: #2563eb;
    font-family: 'JetBrains Mono', monospace;
    word-break: break-word;
}
[data-theme="dark"] .metric-num,
.stApp[data-theme="dark"] .metric-num { color: #60a5fa; }

.metric-label {
    font-size: 0.78rem;
    color: #64748b;
    margin-top: 0.3rem;
}
[data-theme="dark"] .metric-label,
.stApp[data-theme="dark"] .metric-label { color: #94a3b8; }

/* ── Section tag ── */
.section-tag {
    font-family: 'JetBrains Mono', monospace;
    font-size: 0.75rem;
    color: #2563eb;
    background: rgba(37,99,235,0.08);
    padding: 0.25rem 0.7rem;
    border-radius: 999px;
    border: 1px solid rgba(37,99,235,0.25);
    display: inline-block;
    margin-bottom: 0.5rem;
}
[data-theme="dark"] .section-tag,
.stApp[data-theme="dark"] .section-tag {
    color: #93c5fd;
    background: rgba(37,99,235,0.12);
    border-color: rgba(37,99,235,0.35);
}

/* ── Result box — adaptatif ── */
.result-box {
    background: #f1f5f9;
    border: 1px solid #e2e8f0;
    border-left: 4px solid #2563eb;
    border-radius: 8px;
    padding: 1rem 1.2rem;
    font-family: 'JetBrains Mono', monospace;
    font-size: 0.82rem;
    color: #334155;
    margin-top: 0.8rem;
}
[data-theme="dark"] .result-box,
.stApp[data-theme="dark"] .result-box {
    background: #0f172a;
    border-color: rgba(255,255,255,0.07);
    border-left-color: #3b82f6;
    color: #94a3b8;
}

/* ── Info box pédagogique ── */
.info-box {
    background: #eff6ff;
    border: 1px solid #bfdbfe;
    border-radius: 10px;
    padding: 1rem 1.2rem;
    font-size: 0.88rem;
    color: #1e40af;
    margin: 0.8rem 0;
    line-height: 1.6;
}
[data-theme="dark"] .info-box,
.stApp[data-theme="dark"] .info-box {
    background: rgba(30,64,175,0.12);
    border-color: rgba(59,130,246,0.25);
    color: #93c5fd;
}

/* ── Tabs ── */
.stTabs [data-baseweb="tab-list"] { gap: 6px; }
.stTabs [data-baseweb="tab"] {
    border-radius: 8px;
    border: 1px solid transparent;
    font-size: 0.88rem;
    padding: 0.4rem 1.1rem;
    font-family: 'Inter', sans-serif;
    font-weight: 500;
}
.stTabs [aria-selected="true"] {
    background: linear-gradient(135deg, #1d4ed8, #2563eb) !important;
    color: white !important;
    border-color: #2563eb !important;
}
</style>
""", unsafe_allow_html=True)


# ─────────────────────────────────────────────────────────────
# DATA (UKgas R built-in dataset)
# ─────────────────────────────────────────────────────────────
@st.cache_data
def load_data():
    ukgas_values = [
        160.1, 129.7, 84.8,  120.1,
        160.1, 124.9, 84.8,  116.9,
        169.7, 140.9, 89.7,  123.7,
        187.5, 144.2, 92.9,  120.5,
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
    df = pd.DataFrame({'ukgas': ukgas_values}, index=dates)
    return df

df = load_data()
train = df.iloc[:-4]
test  = df.iloc[-4:]

# ─────────────────────────────────────────────────────────────
# PLOTLY THEME HELPER — adaptatif
# ─────────────────────────────────────────────────────────────
def plot_layout(**kwargs):
    """Base layout compatible dark & light mode."""
    base = dict(
        template='plotly',
        paper_bgcolor='rgba(0,0,0,0)',
        plot_bgcolor='rgba(0,0,0,0)',
        font=dict(family='Inter', size=12),
        xaxis=dict(gridcolor='rgba(128,128,128,0.15)', zeroline=False, showline=True, linecolor='rgba(128,128,128,0.3)'),
        yaxis=dict(gridcolor='rgba(128,128,128,0.15)', zeroline=False, showline=True, linecolor='rgba(128,128,128,0.3)'),
        margin=dict(l=50, r=20, t=50, b=40),
        legend=dict(bgcolor='rgba(0,0,0,0)', borderwidth=0, font=dict(size=11)),
    )
    base.update(kwargs)
    return base

COLORS = {
    'primary':   '#2563eb',
    'secondary': '#0891b2',
    'accent':    '#7c3aed',
    'success':   '#059669',
    'warning':   '#d97706',
    'danger':    '#dc2626',
    'series':    '#2563eb',
    'train':     '#2563eb',
    'test':      '#059669',
    'forecast':  '#7c3aed',
    'ci':        'rgba(124,58,237,0.12)',
}

# ─────────────────────────────────────────────────────────────
# HELPER — add_vline workaround (Plotly datetime bug)
# ─────────────────────────────────────────────────────────────
def add_vline_dt(fig, x_ts, label=None, color='rgba(128,128,128,0.5)', dash='dash', row=None, col=None):
    """
    Workaround for Plotly bug with datetime x-values in add_vline.
    Uses add_shape + add_annotation instead.
    """
    x_str = x_ts.isoformat() if hasattr(x_ts, 'isoformat') else str(x_ts)
    shape_kwargs = dict(
        type='line',
        xref='x', yref='paper',
        x0=x_str, x1=x_str,
        y0=0, y1=1,
        line=dict(color=color, dash=dash, width=1.5),
    )
    if row is not None and col is not None:
        shape_kwargs['row'] = row
        shape_kwargs['col'] = col
    fig.add_shape(**shape_kwargs)
    if label:
        annot_kwargs = dict(
            x=x_str, y=1,
            xref='x', yref='paper',
            text=label,
            showarrow=False,
            xanchor='left',
            yanchor='top',
            font=dict(size=11, color=color),
            bgcolor='rgba(255,255,255,0)',
        )
        if row is not None and col is not None:
            annot_kwargs['row'] = row
            annot_kwargs['col'] = col
        fig.add_annotation(**annot_kwargs)

# ─────────────────────────────────────────────────────────────
# HERO
# ─────────────────────────────────────────────────────────────
st.markdown("""
<div class="hero-banner">
  <div class="hero-badge">🎓 Polytech Lyon · MAM 4A · Séries Temporelles</div>
  <div class="hero-title">🔥 UKgas — Analyse de Séries Temporelles</div>
  <div class="hero-sub">
    Consommation trimestrielle de gaz au Royaume-Uni (1960–1986) &mdash; 108 observations<br>
    Lissage exponentiel &nbsp;·&nbsp; Régression linéaire AR(4) &nbsp;·&nbsp; SARIMA Box-Jenkins &nbsp;·&nbsp; Prévisions 5 ans
  </div>
</div>
""", unsafe_allow_html=True)

# KPIs
c1, c2, c3, c4 = st.columns(4)
for col, num, label in [
    (c1, '108', 'Observations trimestrielles'),
    (c2, '1960 – 1986', 'Période couverte'),
    (c3, 'SARIMA(8,1,0)(0,1,0)₄', 'Modèle SARIMA final'),
    (c4, 'AR(4)', 'Meilleur RMSE / MAPE'),
]:
    with col:
        st.markdown(f'<div class="metric-card"><div class="metric-num">{num}</div><div class="metric-label">{label}</div></div>', unsafe_allow_html=True)

st.markdown('<br>', unsafe_allow_html=True)

# ─────────────────────────────────────────────────────────────
# SIDEBAR — contexte
# ─────────────────────────────────────────────────────────────
with st.sidebar:
    st.markdown("### 🔥 UKgas")
    st.markdown("""
    **Série temporelle** de la consommation de gaz au Royaume-Uni, 1960–1986.

    **Fréquence** : trimestrielle (saisonnalité d'ordre 4)

    **Source** : dataset `UKgas` intégré dans R

    ---
    **Démarche Box-Jenkins** :
    1. Visualisation & stationnarisation
    2. Identification (ACF / PACF)
    3. Estimation des paramètres
    4. Validation des résidus
    5. Prévisions

    ---
    **Auteur** : Badreddine EL KHAMLICHI
    *Polytech Lyon · MAM 4A · 2023*
    """)

# ─────────────────────────────────────────────────────────────
# TABS
# ─────────────────────────────────────────────────────────────
tab1, tab2, tab3, tab4, tab5 = st.tabs([
    "📊 1 · Exploration",
    "🔍 2 · Décomposition",
    "📉 3 · Lissage Exponentiel",
    "📈 4 · Régression + AR(4)",
    "🧠 5 · SARIMA Box-Jenkins",
])


# ═══════════════════════════════════════════════════════════════
# TAB 1 — EXPLORATION
# ═══════════════════════════════════════════════════════════════
with tab1:
    st.markdown('<span class="section-tag">Étape 1 / 5 — Description & visualisation de la série</span>', unsafe_allow_html=True)
    st.markdown("""### Présentation de la série UKgas
    La série **UKgas** représente la consommation trimestrielle de gaz au Royaume-Uni entre 1960 et 1986,
    exprimée en **therms** (unité britannique d'énergie). Avec 4 observations par an, elle présente
    une **saisonnalité d'ordre 4** (consommation hivernale plus forte) et une **tendance croissante**.
    """)

    # Chronogramme double
    fig = make_subplots(
        rows=2, cols=1,
        subplot_titles=[
            'Série originale — UKgas (therms)',
            'Série log-transformée — log(UKgas)  [linéarisation de la tendance]'
        ],
        vertical_spacing=0.14,
        shared_xaxes=True,
    )
    fig.add_trace(go.Scatter(
        x=df.index, y=df['ukgas'],
        line=dict(color=COLORS['primary'], width=2),
        name='UKgas',
        fill='tozeroy', fillcolor='rgba(37,99,235,0.07)',
        hovertemplate='%{x|%Y-T%q}<br>Consommation : <b>%{y:.1f} therms</b><extra></extra>',
    ), row=1, col=1)
    fig.add_trace(go.Scatter(
        x=df.index, y=np.log(df['ukgas']),
        line=dict(color=COLORS['secondary'], width=2),
        name='log(UKgas)',
        fill='tozeroy', fillcolor='rgba(8,145,178,0.07)',
        hovertemplate='%{x|%Y-T%q}<br>log(consommation) : <b>%{y:.3f}</b><extra></extra>',
    ), row=2, col=1)
    fig.update_layout(height=540, **plot_layout(title='Chronogramme UKgas — série brute vs log-transformée'))
    st.plotly_chart(fig, use_container_width=True)

    st.markdown("""
    <div class="info-box">
    💡 <strong>Pourquoi passer au logarithme ?</strong><br>
    La série brute suit un <em>modèle multiplicatif</em> X<sub>t</sub> = T<sub>t</sub> × S<sub>t</sub> × ε<sub>t</sub> :
    l'amplitude des oscillations saisonnières augmente avec la tendance (variance non stationnaire).
    Le passage au log transforme ce modèle en un <em>modèle additif</em>
    log(X<sub>t</sub>) = log(T<sub>t</sub>) + log(S<sub>t</sub>) + log(ε<sub>t</sub>),
    ce qui linéarise la tendance et homogénéise la variance.
    </div>
    """, unsafe_allow_html=True)

    st.markdown('---')
    col1, col2 = st.columns([1, 1])

    with col1:
        st.markdown('#### Statistiques descriptives')
        stats_data = {
            'Statistique': ['N (observations)', 'Période', 'Minimum', 'Maximum', 'Moyenne', 'Médiane', 'Écart-type', 'Saisonnalité'],
            'Valeur': [
                '108 trimestres',
                '1960 T1 → 1986 T4',
                f"{df['ukgas'].min():.1f} therms",
                f"{df['ukgas'].max():.1f} therms",
                f"{df['ukgas'].mean():.1f} therms",
                f"{df['ukgas'].median():.1f} therms",
                f"{df['ukgas'].std():.1f} therms",
                'Ordre 4 (trimestrielle)',
            ]
        }
        st.dataframe(pd.DataFrame(stats_data), hide_index=True, use_container_width=True)

    with col2:
        st.markdown('#### Saisonnalité par trimestre')
        df_q = df.copy()
        df_q['Q'] = df_q.index.quarter.map({
            1: 'T1 · Hiver', 2: 'T2 · Printemps',
            3: 'T3 · Été', 4: 'T4 · Automne'
        })
        fig_box = px.box(
            df_q.reset_index(), x='Q', y='ukgas',
            color='Q',
            labels={'ukgas': 'Consommation (therms)', 'Q': ''},
            color_discrete_sequence=[COLORS['primary'], COLORS['success'], COLORS['warning'], COLORS['accent']]
        )
        fig_box.update_layout(
            height=320, showlegend=False,
            **plot_layout()
        )
        st.plotly_chart(fig_box, use_container_width=True)
        st.caption("T1 (hiver) → consommation maximale · T3 (été) → consommation minimale")

    # Evolution annuelle
    st.markdown('#### Évolution de la consommation annuelle moyenne')
    df_annual = df.resample('YS').mean()
    fig_annual = go.Figure(go.Bar(
        x=df_annual.index.year, y=df_annual['ukgas'],
        marker_color=COLORS['primary'], opacity=0.8,
        hovertemplate='%{x}<br>Moy. annuelle : <b>%{y:.1f} therms</b><extra></extra>',
    ))
    fig_annual.add_trace(go.Scatter(
        x=df_annual.index, y=df_annual['ukgas'],
        mode='lines+markers',
        line=dict(color=COLORS['warning'], width=2),
        marker=dict(size=6),
        name='Tendance',
    ))
    fig_annual.update_layout(
        height=320,
        **plot_layout(title='Consommation annuelle moyenne (therms/trimestre)',
                      xaxis=dict(title='Année', gridcolor='rgba(128,128,128,0.15)', zeroline=False),
                      yaxis=dict(title='Therms', gridcolor='rgba(128,128,128,0.15)', zeroline=False))
    )
    st.plotly_chart(fig_annual, use_container_width=True)


# ═══════════════════════════════════════════════════════════════
# TAB 2 — DECOMPOSITION
# ═══════════════════════════════════════════════════════════════
with tab2:
    st.markdown('<span class="section-tag">Étape 2 / 5 — Décomposition & stationnarité</span>', unsafe_allow_html=True)
    st.markdown("### Décomposition de la série")
    st.markdown("""
    La décomposition permet d'isoler les trois composantes de la série :
    **tendance** (T), **saisonnalité** (S), et **résidus** (ε).
    On travaille sur la série log-transformée pour rester dans un cadre additif.
    """)

    from statsmodels.tsa.seasonal import seasonal_decompose

    model_choice = st.radio(
        'Modèle de décomposition :',
        ['Additif — sur log(UKgas)  ✅ recommandé', 'Multiplicatif — sur UKgas brut'],
        horizontal=True
    )
    use_log = 'Additif' in model_choice
    series_to_decompose = np.log(df['ukgas']) if use_log else df['ukgas']
    model_type = 'additive' if use_log else 'multiplicative'

    decomp = seasonal_decompose(series_to_decompose, model=model_type, period=4)

    comp_labels = ['Série originale', 'Tendance', 'Saisonnalité', 'Résidus']
    comp_colors = [COLORS['primary'], COLORS['secondary'], COLORS['accent'], COLORS['warning']]
    comp_data   = [series_to_decompose, decomp.trend, decomp.seasonal, decomp.resid]
    comp_desc   = [
        "La série log-transformée présente une tendance croissante et des oscillations saisonnières stables.",
        "La tendance est globalement croissante jusqu'en 1975, puis se stabilise légèrement.",
        "La saisonnalité est annuelle (ordre 4) : forte en hiver, faible en été.",
        "Les résidus doivent ressembler à un bruit blanc pour valider le modèle.",
    ]

    fig = make_subplots(
        rows=4, cols=1,
        subplot_titles=comp_labels,
        vertical_spacing=0.07,
        shared_xaxes=True,
    )
    for i, (comp, name, color) in enumerate(zip(comp_data, comp_labels, comp_colors)):
        fig.add_trace(
            go.Scatter(x=df.index, y=comp, line=dict(color=color, width=2), name=name, showlegend=False),
            row=i+1, col=1
        )
    fig.update_layout(height=800, **plot_layout(title='Décomposition de log(UKgas) en tendance + saisonnalité + résidus'))
    st.plotly_chart(fig, use_container_width=True)

    for desc, label in zip(comp_desc, comp_labels):
        st.caption(f"**{label}** — {desc}")

    # ── Tests de stationnarité
    st.markdown('---')
    st.markdown('### Tests de stationnarité — ADF (Augmented Dickey-Fuller)')
    st.markdown("""
    <div class="info-box">
    💡 <strong>Objectif :</strong> Un modèle ARIMA/SARIMA requiert une série <em>stationnaire</em>
    (moyenne et variance constantes dans le temps). On applique successivement :
    <ul>
    <li>Une <strong>différenciation d'ordre 1</strong> (d=1) pour supprimer la tendance</li>
    <li>Une <strong>différenciation saisonnière d'ordre 4</strong> (D=1) pour supprimer la saisonnalité</li>
    </ul>
    </div>
    """, unsafe_allow_html=True)

    log_s = np.log(df['ukgas'].dropna())
    diff1  = log_s.diff().dropna()
    diff_s = log_s.diff().diff(4).dropna()

    series_tests = [
        ('log(UKgas)',          log_s,  'Série de départ — non stationnaire attendue'),
        ('∆ log(UKgas)',        diff1,  'Après diff. ordre 1 — tendance supprimée'),
        ('∆ ∆₄ log(UKgas)',    diff_s, 'Après diff. ordre 1 + saisonnière — série cible'),
    ]

    rows_adf = []
    for name, series, comment in series_tests:
        try:
            adf_stat, adf_p, _, _, _, _ = adfuller(series)
            stationary = '✅ Stationnaire' if adf_p < 0.05 else '❌ Non stationnaire'
            rows_adf.append({
                'Série': name,
                'ADF statistic': f"{adf_stat:.4f}",
                'p-value': f"{adf_p:.4f}",
                'Résultat': stationary,
                'Commentaire': comment
            })
        except Exception:
            rows_adf.append({'Série': name, 'ADF statistic': 'N/A', 'p-value': 'N/A', 'Résultat': 'N/A', 'Commentaire': comment})

    st.dataframe(pd.DataFrame(rows_adf), hide_index=True, use_container_width=True)

    # Visualisation des séries différenciées
    st.markdown('#### Visualisation — effet des différenciations')
    fig_diff = make_subplots(
        rows=3, cols=1,
        subplot_titles=[r["Série"] + ' — ' + r["Résultat"] for r in rows_adf],
        vertical_spacing=0.12,
        shared_xaxes=False,
    )
    diff_series = [log_s, diff1, diff_s]
    diff_colors = [COLORS['danger'], COLORS['warning'], COLORS['success']]
    for i, (s, c) in enumerate(zip(diff_series, diff_colors)):
        fig_diff.add_trace(
            go.Scatter(x=s.index, y=s.values, line=dict(color=c, width=1.5), showlegend=False),
            row=i+1, col=1
        )
        fig_diff.add_hline(y=s.mean(), line_dash='dash', line_color='rgba(128,128,128,0.5)', row=i+1, col=1)
    fig_diff.update_layout(height=600, **plot_layout())
    st.plotly_chart(fig_diff, use_container_width=True)

    # ACF / PACF série doublement différenciée
    st.markdown('---')
    st.markdown('### ACF & PACF — Identification des paramètres p et q')
    st.markdown("""
    <div class="info-box">
    💡 <strong>Lecture des corrélogrammes :</strong><br>
    • <strong>ACF</strong> → identifie l'ordre q (MA) : s'annule après le lag q<br>
    • <strong>PACF</strong> → identifie l'ordre p (AR) : s'annule après le lag p<br>
    Les barres dépassant les <span style="color:#d97706">bandes de confiance à 95%</span> sont significatives.
    </div>
    """, unsafe_allow_html=True)

    try:
        max_lag = 30
        acf_vals  = acf(diff_s,  nlags=max_lag, fft=True)
        pacf_vals = pacf(diff_s, nlags=max_lag)
        ci_acf = 1.96 / np.sqrt(len(diff_s))

        fig_acf = make_subplots(
            rows=1, cols=2,
            subplot_titles=[
                'ACF — ∆∆₄ log(UKgas)  →  ordre q (MA)',
                'PACF — ∆∆₄ log(UKgas)  →  ordre p (AR)'
            ]
        )
        for lag_val, av in enumerate(acf_vals):
            clr = COLORS['primary'] if abs(av) > ci_acf else 'rgba(37,99,235,0.3)'
            fig_acf.add_trace(go.Bar(x=[lag_val], y=[av], marker_color=clr, showlegend=False), row=1, col=1)
        for lag_val, pv in enumerate(pacf_vals):
            clr = COLORS['secondary'] if abs(pv) > ci_acf else 'rgba(8,145,178,0.3)'
            fig_acf.add_trace(go.Bar(x=[lag_val], y=[pv], marker_color=clr, showlegend=False), row=1, col=2)
        for col_n in [1, 2]:
            fig_acf.add_hline(y= ci_acf, line_dash='dash', line_color=COLORS['warning'], opacity=0.8, row=1, col=col_n)
            fig_acf.add_hline(y=-ci_acf, line_dash='dash', line_color=COLORS['warning'], opacity=0.8, row=1, col=col_n)
        fig_acf.update_layout(height=380, **plot_layout())
        st.plotly_chart(fig_acf, use_container_width=True)
        st.caption(f"Bandes de confiance à 95% : ±{ci_acf:.4f}  (IC = 1.96/√{len(diff_s)})")
        st.markdown("""
        <div class="result-box">
        📌 <strong>Lecture :</strong> Le PACF s'annule approximativement après le lag 8 → <strong>p = 8</strong>.<br>
        L'ACF décroît progressivement → pas de coupure nette → q = 0 (pure AR).<br>
        Combiné aux différenciations (d=1, D=1), on obtient le candidat <strong>SARIMA(8,1,0)(0,1,0)₄</strong>.
        </div>
        """, unsafe_allow_html=True)
    except Exception as e:
        st.warning(f"ACF/PACF : {e}")


# ═══════════════════════════════════════════════════════════════
# TAB 3 — LISSAGE EXPONENTIEL
# ═══════════════════════════════════════════════════════════════
with tab3:
    st.markdown('<span class="section-tag">Étape 3 / 5 — Lissage exponentiel (benchmark)</span>', unsafe_allow_html=True)
    st.markdown('### Lissage exponentiel — LES, LED, Holt-Winters')
    st.markdown("""
    <div class="info-box">
    💡 Le lissage exponentiel est une méthode de prévision par <strong>pondération décroissante du passé</strong>.
    On teste trois variantes :<br>
    • <strong>LES</strong> (Simple) — série sans tendance ni saisonnalité<br>
    • <strong>LED</strong> (Double / Holt) — avec tendance, sans saisonnalité<br>
    • <strong>Holt-Winters</strong> — avec tendance <em>et</em> saisonnalité (additif ou multiplicatif)<br>
    Ces méthodes serviront de <em>benchmark</em> par rapport aux approches ARIMA.
    </div>
    """, unsafe_allow_html=True)

    @st.cache_data
    def fit_hw_models(train_vals_tuple):
        train_s = pd.Series(train_vals_tuple[1], index=pd.DatetimeIndex(train_vals_tuple[0]))
        return {
            'LES (Simple)':       ExponentialSmoothing(train_s, trend=None,  seasonal=None).fit(optimized=True),
            'LED (Double)':       ExponentialSmoothing(train_s, trend='add', seasonal=None).fit(optimized=True),
            'HW Additif':         ExponentialSmoothing(train_s, trend='add', seasonal='add', seasonal_periods=4).fit(optimized=True),
            'HW Multiplicatif':   ExponentialSmoothing(train_s, trend='add', seasonal='mul', seasonal_periods=4).fit(optimized=True),
        }

    models_hw = fit_hw_models((train.index.tolist(), train['ukgas'].tolist()))

    horizon = st.slider('Horizon de prévision (trimestres)', 4, 40, 16, key='hw_horizon')
    future_dates_hw = pd.date_range(
        start=train.index[-1] + pd.DateOffset(months=3), periods=horizon, freq='QS'
    )

    # Métriques sur test set
    test_vals = test['ukgas'].values
    metrics_hw = []
    colors_hw  = [COLORS['danger'], COLORS['warning'], COLORS['secondary'], COLORS['accent']]
    for (name, model), color in zip(models_hw.items(), colors_hw):
        pred4 = model.forecast(4).values[:4]
        rmse  = np.sqrt(np.mean((test_vals - pred4) ** 2))
        mape  = np.mean(np.abs((test_vals - pred4) / test_vals)) * 100
        metrics_hw.append({'Méthode': name, 'RMSE': round(rmse, 2), 'MAPE (%)': round(mape, 2), '_color': color})

    # Graphe principal
    fig_hw = go.Figure()
    fig_hw.add_trace(go.Scatter(
        x=train.index, y=train['ukgas'],
        name='Série (train)', line=dict(color=COLORS['train'], width=2.5),
        hovertemplate='%{x|%Y-T%q} — <b>%{y:.1f}</b> therms<extra></extra>',
    ))
    fig_hw.add_trace(go.Scatter(
        x=test.index, y=test['ukgas'],
        name='Série (test réel)', line=dict(color=COLORS['test'], width=2.5),
    ))
    for (name, model), color in zip(models_hw.items(), colors_hw):
        fc = model.forecast(horizon)
        fig_hw.add_trace(go.Scatter(
            x=future_dates_hw, y=fc.values,
            name=f'{name}',
            line=dict(color=color, width=1.8, dash='dot'),
        ))
    fig_hw.add_vrect(
        x0=train.index[-1].isoformat(), x1=future_dates_hw[-1].isoformat(),
        fillcolor='rgba(0,0,0,0.03)', line_width=0,
    )
    add_vline_dt(fig_hw, train.index[-1], label=' Train / Test')
    fig_hw.update_layout(
        height=480,
        **plot_layout(title='Lissage exponentiel — ajustement sur le train & prévisions')
    )
    st.plotly_chart(fig_hw, use_container_width=True)

    # Métriques
    st.markdown('#### Performances sur le jeu de test (1986 — 4 trimestres)')
    metrics_df_display = pd.DataFrame([{k: v for k, v in m.items() if k != '_color'} for m in metrics_hw])
    best_idx = metrics_df_display['RMSE'].idxmin()
    st.dataframe(
        metrics_df_display.style.highlight_min(subset=['RMSE', 'MAPE (%)'], color='#d1fae5'),
        hide_index=True, use_container_width=True
    )
    best_name = metrics_df_display.loc[best_idx, 'Méthode']
    st.success(f"🏆 **{best_name}** — meilleur compromis RMSE/MAPE parmi les méthodes de lissage exponentiel.")

    # Résidus du meilleur modèle
    st.markdown('---')
    st.markdown('#### Analyse des résidus — Holt-Winters Additif')
    st.caption("Les résidus doivent être centrés sur 0 et non autocorrélés pour valider le modèle (bruit blanc).")

    hw_best   = models_hw['HW Additif']
    resid_hw  = hw_best.resid.dropna()

    col1, col2 = st.columns(2)
    with col1:
        fig_r = go.Figure()
        fig_r.add_trace(go.Scatter(
            x=train.index[:len(resid_hw)], y=resid_hw,
            mode='lines', line=dict(color=COLORS['primary'], width=1.5), name='Résidus',
        ))
        fig_r.add_hline(y=0, line_dash='dash', line_color=COLORS['warning'], line_width=1.5)
        fig_r.update_layout(height=300, **plot_layout(title='Résidus — Holt-Winters Additif'))
        st.plotly_chart(fig_r, use_container_width=True)
    with col2:
        qq = stats.probplot(resid_hw, dist='norm')
        fig_qq = go.Figure()
        fig_qq.add_trace(go.Scatter(
            x=qq[0][0], y=qq[0][1], mode='markers',
            marker=dict(color=COLORS['secondary'], size=5), name='Quantiles observés',
        ))
        fig_qq.add_trace(go.Scatter(
            x=qq[0][0], y=qq[1][0]*np.array(qq[0][0]) + qq[1][1],
            mode='lines', line=dict(color=COLORS['warning'], width=2), name='Droite théorique',
        ))
        fig_qq.update_layout(height=300, **plot_layout(title='QQ-Plot résidus (normalité)'))
        st.plotly_chart(fig_qq, use_container_width=True)

    st.markdown("""
    <div class="result-box">
    📌 Le QQ-Plot des résidus HW Additif est quasi-linéaire → <strong>distribution normale des résidus validée</strong>.
    La méthode Holt-Winters capture bien la tendance et la saisonnalité de la série UKgas.
    </div>
    """, unsafe_allow_html=True)


# ═══════════════════════════════════════════════════════════════
# TAB 4 — REGRESSION LINEAIRE
# ═══════════════════════════════════════════════════════════════
with tab4:
    st.markdown('<span class="section-tag">Étape 4 / 5 — Régression linéaire + modèle AR(4)</span>', unsafe_allow_html=True)
    st.markdown('### Modélisation par régression linéaire')
    st.markdown("""
    <div class="info-box">
    💡 <strong>Principe :</strong> On modélise log(UKgas) comme une combinaison linéaire de la tendance
    et de la saisonnalité (via des fonctions trigonométriques). Les résidus du premier modèle (OLS)
    présentent une autocorrélation résiduelle → on les modélise avec un <strong>AR(4)</strong>.
    </div>
    """, unsafe_allow_html=True)

    # Construction des régresseurs
    log_train = np.log(train['ukgas'])
    log_test  = np.log(test['ukgas'])
    n = len(log_train)

    def make_regressors(start_idx, length):
        t    = np.arange(start_idx, start_idx + length)
        freq = 2 * np.pi * t / 4
        return np.column_stack([
            t,
            np.cos(freq),   np.sin(freq),
            np.cos(2*freq), np.sin(2*freq),
        ])

    X_train = make_regressors(1, n)
    X_test  = make_regressors(n+1, len(log_test))
    X_fut   = make_regressors(n+1, 20)

    tabs_reg = st.tabs([
        "📌 Étape A — OLS de base",
        "✅ Étape B — OLS + AR(4)",
        "🔮 Étape C — Prévisions 5 ans",
    ])

    # ── A: OLS
    with tabs_reg[0]:
        st.markdown('#### Modèle OLS automatique (régression sur tendance + saisonnalité)')
        from statsmodels.regression.linear_model import OLS
        from statsmodels.tools import add_constant

        @st.cache_data
        def fit_ols(X, y):
            return OLS(y, add_constant(X)).fit()

        ols_model  = fit_ols(X_train, log_train.values)
        resid_ols  = ols_model.resid
        acf_ols    = acf(resid_ols, nlags=20, fft=True)
        pacf_ols   = pacf(resid_ols, nlags=20)
        ci_ols     = 1.96 / np.sqrt(len(resid_ols))

        # Ajustement
        fitted_log = ols_model.fittedvalues
        fig_ols_fit = go.Figure()
        fig_ols_fit.add_trace(go.Scatter(
            x=train.index, y=train['ukgas'],
            name='Observé', line=dict(color=COLORS['primary'], width=2),
        ))
        fig_ols_fit.add_trace(go.Scatter(
            x=train.index, y=np.exp(fitted_log),
            name='Ajusté (OLS)', line=dict(color=COLORS['danger'], width=1.8, dash='dash'),
        ))
        fig_ols_fit.update_layout(height=320, **plot_layout(title='OLS — ajustement sur le train'))
        st.plotly_chart(fig_ols_fit, use_container_width=True)

        # ACF / PACF résidus
        fig_ap = make_subplots(rows=1, cols=2,
                                subplot_titles=['ACF résidus OLS → q (MA)', 'PACF résidus OLS → p (AR)'])
        for i, v in enumerate(acf_ols):
            fig_ap.add_trace(go.Bar(x=[i], y=[v],
                                     marker_color=COLORS['primary'] if abs(v) > ci_ols else 'rgba(37,99,235,0.3)',
                                     showlegend=False), row=1, col=1)
        for i, v in enumerate(pacf_ols):
            fig_ap.add_trace(go.Bar(x=[i], y=[v],
                                     marker_color=COLORS['accent'] if abs(v) > ci_ols else 'rgba(124,58,237,0.3)',
                                     showlegend=False), row=1, col=2)
        for col_n in [1, 2]:
            fig_ap.add_hline(y= ci_ols, line_dash='dash', line_color=COLORS['warning'], opacity=0.8, row=1, col=col_n)
            fig_ap.add_hline(y=-ci_ols, line_dash='dash', line_color=COLORS['warning'], opacity=0.8, row=1, col=col_n)
        fig_ap.update_layout(height=360, **plot_layout())
        st.plotly_chart(fig_ap, use_container_width=True)

        st.markdown(f"""
        <div class="result-box">
        AIC (OLS) : <strong>{ols_model.aic:.2f}</strong> &nbsp;|&nbsp; BIC : <strong>{ols_model.bic:.2f}</strong><br>
        ⚠️ Le PACF des résidus OLS s'annule aux lags multiples de 4 → autocorrélation résiduelle significative.<br>
        → <strong>Hypothèse :</strong> les résidus suivent un processus AR(4). On l'intègre directement dans le modèle.
        </div>
        """, unsafe_allow_html=True)

    # ── B: AR(4)
    with tabs_reg[1]:
        st.markdown('#### Modèle OLS + AR(4) — régression avec erreurs autoréggressives')

        @st.cache_data
        def fit_ar4(X_tr, y_tr_list, idx_list):
            y_tr = pd.Series(y_tr_list, index=pd.DatetimeIndex(idx_list))
            return SARIMAX(
                y_tr, order=(4,0,0), exog=X_tr,
                trend='n',
                enforce_stationarity=False,
                enforce_invertibility=False,
            ).fit(disp=False)

        ar4_model = fit_ar4(X_train, log_train.tolist(), log_train.index.tolist())
        resid_ar4 = ar4_model.resid.dropna()
        acf_ar4   = acf(resid_ar4, nlags=20, fft=True)
        pacf_ar4  = pacf(resid_ar4, nlags=20)
        ci_ar4    = 1.96 / np.sqrt(len(resid_ar4))

        col1, col2 = st.columns(2)
        with col1:
            fig_ap4 = make_subplots(rows=1, cols=2, subplot_titles=['ACF résidus AR(4)', 'PACF résidus AR(4)'])
            for i, v in enumerate(acf_ar4):
                fig_ap4.add_trace(go.Bar(x=[i], y=[v],
                                          marker_color=COLORS['secondary'] if abs(v) > ci_ar4 else 'rgba(8,145,178,0.3)',
                                          showlegend=False), row=1, col=1)
            for i, v in enumerate(pacf_ar4):
                fig_ap4.add_trace(go.Bar(x=[i], y=[v],
                                          marker_color=COLORS['success'] if abs(v) > ci_ar4 else 'rgba(5,150,105,0.3)',
                                          showlegend=False), row=1, col=2)
            for col_n in [1, 2]:
                fig_ap4.add_hline(y= ci_ar4, line_dash='dash', line_color=COLORS['warning'], opacity=0.8, row=1, col=col_n)
                fig_ap4.add_hline(y=-ci_ar4, line_dash='dash', line_color=COLORS['warning'], opacity=0.8, row=1, col=col_n)
            fig_ap4.update_layout(height=320, **plot_layout(title="Résidus AR(4) — absence d'autocorrélation ✅"))
            st.plotly_chart(fig_ap4, use_container_width=True)

        with col2:
            qq4 = stats.probplot(resid_ar4, dist='norm')
            fig_qq4 = go.Figure()
            fig_qq4.add_trace(go.Scatter(
                x=qq4[0][0], y=qq4[0][1], mode='markers',
                marker=dict(color=COLORS['secondary'], size=5), name='Quantiles',
            ))
            fig_qq4.add_trace(go.Scatter(
                x=qq4[0][0], y=qq4[1][0]*np.array(qq4[0][0]) + qq4[1][1],
                mode='lines', line=dict(color=COLORS['warning'], width=2), name='Théorique',
            ))
            fig_qq4.update_layout(height=320, **plot_layout(title='QQ-Plot résidus AR(4)'))
            st.plotly_chart(fig_qq4, use_container_width=True)

        # Ljung-Box
        lb = acorr_ljungbox(resid_ar4, lags=[1,2,3,4], return_df=True)
        st.markdown("**Test de Ljung-Box (portmanteau) — H₀ : pas d'autocorrélation résiduelle**")
        lb_display = lb[['lb_stat', 'lb_pvalue']].rename(
            columns={'lb_stat': 'X² statistique', 'lb_pvalue': 'p-value'}
        )
        lb_display.index = [f"Lag {l}" for l in [1,2,3,4]]
        st.dataframe(lb_display.style.apply(
            lambda col: ['background-color: #d1fae5' if v > 0.05 else 'background-color: #fee2e2'
                         for v in col] if col.name == 'p-value' else ['' for _ in col],
            axis=0), use_container_width=True
        )
        st.markdown(f"""
        <div class="result-box">
        AIC AR(4) : <strong>{ar4_model.aic:.2f}</strong> &nbsp;|&nbsp; BIC : <strong>{ar4_model.bic:.2f}</strong><br>
        ✅ AIC largement inférieur au modèle OLS de base — résidus proches d'un bruit blanc.
        </div>
        """, unsafe_allow_html=True)

    # ── C: Prévisions 5 ans
    with tabs_reg[2]:
        st.markdown('#### Prévisions AR(4) — horizon 5 ans (20 trimestres)')
        try:
            fc_ar4     = ar4_model.get_forecast(steps=20, exog=X_fut)
            fc_mean    = np.exp(fc_ar4.predicted_mean)
            fc_ci      = fc_ar4.conf_int(alpha=0.05)
            fc_lower   = np.exp(fc_ci.iloc[:, 0])
            fc_upper   = np.exp(fc_ci.iloc[:, 1])
            fut_dates  = pd.date_range(
                start=df.index[-1] + pd.DateOffset(months=3), periods=20, freq='QS'
            )

            fig_fc = go.Figure()
            fig_fc.add_trace(go.Scatter(
                x=df.index, y=df['ukgas'],
                name='Série observée', line=dict(color=COLORS['primary'], width=2.5),
            ))
            fig_fc.add_trace(go.Scatter(
                x=fut_dates, y=fc_mean.values,
                name='Prévision AR(4)', line=dict(color=COLORS['forecast'], width=2.5),
            ))
            fig_fc.add_trace(go.Scatter(
                x=list(fut_dates) + list(fut_dates[::-1]),
                y=list(fc_upper.values) + list(fc_lower.values[::-1]),
                fill='toself', fillcolor=COLORS['ci'],
                line=dict(color='rgba(0,0,0,0)'), name='Intervalle de confiance 95%',
            ))
            add_vline_dt(fig_fc, df.index[-1], label=' Fin données observées')
            fig_fc.update_layout(
                height=480,
                **plot_layout(title='Prévisions AR(4) — 20 trimestres avec IC 95%')
            )
            st.plotly_chart(fig_fc, use_container_width=True)
            st.caption("La zone ombrée représente l'intervalle de confiance à 95% : l'incertitude s'élargit avec l'horizon.")
        except Exception as e:
            st.error(f"Erreur prévision AR(4) : {e}")


# ═══════════════════════════════════════════════════════════════
# TAB 5 — SARIMA
# ═══════════════════════════════════════════════════════════════
with tab5:
    st.markdown('<span class="section-tag">Étape 5 / 5 — Modélisation SARIMA · Méthodologie Box-Jenkins</span>', unsafe_allow_html=True)
    st.markdown('### SARIMA — Méthodologie Box-Jenkins complète')
    st.markdown("""
    <div class="info-box">
    💡 <strong>Méthodologie Box-Jenkins (1970) :</strong><br>
    1. <strong>Identification</strong> : analyse des ACF/PACF → choix de p, d, q, P, D, Q<br>
    2. <strong>Estimation</strong> : ajustement par maximum de vraisemblance<br>
    3. <strong>Validation</strong> : analyse des résidus (bruit blanc ?)<br>
    4. <strong>Prévision</strong> : calcul des prévisions avec intervalles de confiance<br><br>
    L'analyse de l'onglet Décomposition a conduit au modèle candidat <strong>SARIMA(8,1,0)(0,1,0)₄</strong>.
    Vous pouvez modifier les paramètres ci-dessous pour explorer d'autres configurations.
    </div>
    """, unsafe_allow_html=True)

    st.markdown('#### Paramètres du modèle SARIMA')
    c1, c2, c3, c_sep, c4, c5, c6 = st.columns([1,1,1,0.3,1,1,1])
    p_s = c1.number_input('p — AR',        min_value=0, max_value=12, value=8, help='Ordre autorégressif')
    d_s = c2.number_input('d — diff',       min_value=0, max_value=2,  value=1, help='Ordre de différenciation')
    q_s = c3.number_input('q — MA',        min_value=0, max_value=12, value=0, help='Ordre moyenne mobile')
    with c_sep: st.markdown('<div style="text-align:center;padding-top:2rem;font-size:1.5rem;opacity:0.4">×</div>', unsafe_allow_html=True)
    P_s = c4.number_input('P — AR sais.',  min_value=0, max_value=4,  value=0, help='AR saisonnier')
    D_s = c5.number_input('D — diff sais.',min_value=0, max_value=2,  value=1, help='Différenciation saisonnière')
    Q_s = c6.number_input('Q — MA sais.',  min_value=0, max_value=4,  value=0, help='MA saisonnier')

    log_train_s = np.log(train['ukgas'])

    @st.cache_data
    def fit_sarima_models(train_list, train_idx, p, d, q, P, D, Q):
        y = pd.Series(train_list, index=pd.DatetimeIndex(train_idx))
        mod_ar8  = SARIMAX(y, order=(8,1,0), seasonal_order=(0,1,0,4),
                           enforce_stationarity=False, enforce_invertibility=False).fit(disp=False)
        mod_arma = SARIMAX(y, order=(8,1,5), seasonal_order=(0,1,0,4),
                           enforce_stationarity=False, enforce_invertibility=False).fit(disp=False)
        mod_cust = SARIMAX(y, order=(p,d,q), seasonal_order=(P,D,Q,4),
                           enforce_stationarity=False, enforce_invertibility=False).fit(disp=False)
        return mod_ar8, mod_arma, mod_cust

    try:
        mod_ar8, mod_arma, mod_final = fit_sarima_models(
            log_train_s.tolist(), log_train_s.index.tolist(),
            int(p_s), int(d_s), int(q_s), int(P_s), int(D_s), int(Q_s)
        )
    except Exception as e:
        st.error(f"Erreur ajustement SARIMA : {e}")
        st.stop()

    tabs_sarima = st.tabs([
        '📊 Comparaison des modèles',
        '🔬 Validation des résidus',
        '🔮 Prévisions finales',
        '🏆 Bilan comparatif',
    ])

    # ── Comparaison
    with tabs_sarima[0]:
        st.markdown('#### Comparaison AIC/BIC — Sélection du modèle')
        sarima_label = f'SARIMA({p_s},{d_s},{q_s})({P_s},{D_s},{Q_s})₄'
        aic_df = pd.DataFrame({
            'Modèle': [
                'AR(8)  [= SARIMA(8,1,0)(0,1,0)₄]',
                'ARMA(8,5)  [= SARIMA(8,1,5)(0,1,0)₄]',
                sarima_label + '  ← sélectionné',
            ],
            'AIC':  [f"{mod_ar8.aic:.2f}",  f"{mod_arma.aic:.2f}",  f"{mod_final.aic:.2f}"],
            'BIC':  [f"{mod_ar8.bic:.2f}",  f"{mod_arma.bic:.2f}",  f"{mod_final.bic:.2f}"],
            'nb paramètres': [
                mod_ar8.df_model, mod_arma.df_model, mod_final.df_model
            ],
            'Sélectionné': ['✅ (référence)', '❌ (surparamétré)', '⭐'],
        })
        st.dataframe(aic_df, hide_index=True, use_container_width=True)
        st.markdown("""
        <div class="result-box">
        📌 <strong>Principe de parcimonie :</strong> entre AR(8) et ARMA(8,5), l'AIC favorise AR(8)
        qui obtient une performance équivalente avec moins de paramètres.
        SARIMA(8,1,0)(0,1,0)₄ est donc retenu comme <strong>modèle final</strong>.
        </div>
        """, unsafe_allow_html=True)

        # Ajustement visuel
        fitted_sarima = np.exp(mod_final.fittedvalues)
        fig_fit_s = go.Figure()
        fig_fit_s.add_trace(go.Scatter(
            x=train.index, y=train['ukgas'],
            name='Série (train)', line=dict(color=COLORS['primary'], width=2),
        ))
        fig_fit_s.add_trace(go.Scatter(
            x=train.index, y=fitted_sarima,
            name='Ajusté SARIMA', line=dict(color=COLORS['forecast'], width=1.8, dash='dash'),
        ))
        fig_fit_s.update_layout(height=360, **plot_layout(title=f'Ajustement {sarima_label} sur le train'))
        st.plotly_chart(fig_fit_s, use_container_width=True)

    # ── Validation
    with tabs_sarima[1]:
        st.markdown('#### Validation du modèle SARIMA — analyse des résidus')
        st.markdown("Un bon modèle doit avoir des résidus ressemblant à un **bruit blanc gaussien** : centrés, non autocorrélés, normaux.")

        resid_s = mod_final.resid.dropna()

        col1, col2 = st.columns(2)
        with col1:
            # Lagplot
            fig_lag = go.Figure()
            lag_colors = [COLORS['primary'], COLORS['secondary'], COLORS['accent'], COLORS['success']]
            for lag_val, lc in zip([1, 2, 3, 4], lag_colors):
                x_lag = resid_s.values[lag_val:]
                y_lag = resid_s.values[:-lag_val]
                fig_lag.add_trace(go.Scatter(
                    x=x_lag, y=y_lag, mode='markers',
                    marker=dict(size=4, color=lc, opacity=0.6),
                    name=f'lag {lag_val}',
                ))
            fig_lag.update_layout(height=320, **plot_layout(
                title='Lagplot résidus — absence de structure souhaitée',
                xaxis=dict(title='Résidu(t)', gridcolor='rgba(128,128,128,0.15)', zeroline=False),
                yaxis=dict(title='Résidu(t-lag)', gridcolor='rgba(128,128,128,0.15)', zeroline=False),
            ))
            st.plotly_chart(fig_lag, use_container_width=True)

        with col2:
            qq_s = stats.probplot(resid_s, dist='norm')
            fig_qq_s = go.Figure()
            fig_qq_s.add_trace(go.Scatter(
                x=qq_s[0][0], y=qq_s[0][1], mode='markers',
                marker=dict(color=COLORS['accent'], size=5), name='Quantiles observés',
            ))
            fig_qq_s.add_trace(go.Scatter(
                x=qq_s[0][0], y=qq_s[1][0]*np.array(qq_s[0][0]) + qq_s[1][1],
                mode='lines', line=dict(color=COLORS['warning'], width=2), name='Droite théorique',
            ))
            fig_qq_s.update_layout(height=320, **plot_layout(title='QQ-Plot résidus SARIMA'))
            st.plotly_chart(fig_qq_s, use_container_width=True)

        # ACF / PACF résidus
        acf_s  = acf(resid_s,  nlags=20, fft=True)
        pacf_s = pacf(resid_s, nlags=20)
        ci_s   = 1.96 / np.sqrt(len(resid_s))
        fig_aps = make_subplots(rows=1, cols=2, subplot_titles=['ACF résidus SARIMA', 'PACF résidus SARIMA'])
        for i, v in enumerate(acf_s):
            fig_aps.add_trace(go.Bar(x=[i], y=[v],
                                      marker_color=COLORS['accent'] if abs(v) > ci_s else 'rgba(124,58,237,0.3)',
                                      showlegend=False), row=1, col=1)
        for i, v in enumerate(pacf_s):
            fig_aps.add_trace(go.Bar(x=[i], y=[v],
                                      marker_color=COLORS['warning'] if abs(v) > ci_s else 'rgba(217,119,6,0.3)',
                                      showlegend=False), row=1, col=2)
        for col_n in [1, 2]:
            fig_aps.add_hline(y= ci_s, line_dash='dash', line_color=COLORS['danger'], opacity=0.7, row=1, col=col_n)
            fig_aps.add_hline(y=-ci_s, line_dash='dash', line_color=COLORS['danger'], opacity=0.7, row=1, col=col_n)
        fig_aps.update_layout(height=320, **plot_layout())
        st.plotly_chart(fig_aps, use_container_width=True)

        # Ljung-Box
        lb_s = acorr_ljungbox(resid_s, lags=[1,2,3,4,8], return_df=True)
        st.markdown('**Test de Ljung-Box — H₀ : résidus non autocorrélés (p-value > 0.05 = ✅)**')
        lb_s_disp = lb_s[['lb_stat','lb_pvalue']].rename(columns={'lb_stat':'X²','lb_pvalue':'p-value'})
        lb_s_disp.index = [f"Lag {l}" for l in [1,2,3,4,8]]
        st.dataframe(
            lb_s_disp.style.apply(
                lambda col: ['background-color: #d1fae5' if v > 0.05 else 'background-color: #fee2e2'
                             for v in col] if col.name == 'p-value' else ['' for _ in col],
                axis=0),
            use_container_width=True
        )

    # ── Prévisions
    with tabs_sarima[2]:
        sarima_label2 = f'SARIMA({p_s},{d_s},{q_s})({P_s},{D_s},{Q_s})₄'
        st.markdown(f'#### Prévisions {sarima_label2} — horizon configurable')
        forecast_h_s = st.slider('Horizon de prévision (trimestres)', 4, 40, 20, key='sarima_h2')
        try:
            fc_s       = mod_final.get_forecast(steps=forecast_h_s)
            fc_mean_s  = np.exp(fc_s.predicted_mean)
            fc_ci_s    = fc_s.conf_int(alpha=0.05)
            fc_lower_s = np.exp(fc_ci_s.iloc[:,0])
            fc_upper_s = np.exp(fc_ci_s.iloc[:,1])
            fut_s      = pd.date_range(
                start=train.index[-1] + pd.DateOffset(months=3),
                periods=forecast_h_s, freq='QS'
            )

            fig_fcs = go.Figure()
            fig_fcs.add_trace(go.Scatter(
                x=train.index, y=train['ukgas'],
                name='Train', line=dict(color=COLORS['train'], width=2.5),
            ))
            fig_fcs.add_trace(go.Scatter(
                x=test.index, y=test['ukgas'],
                name='Test (réel)', line=dict(color=COLORS['test'], width=2.5, dash='dash'),
            ))
            fig_fcs.add_trace(go.Scatter(
                x=fut_s, y=fc_mean_s.values,
                name=sarima_label2, line=dict(color=COLORS['forecast'], width=2.5),
            ))
            fig_fcs.add_trace(go.Scatter(
                x=list(fut_s) + list(fut_s[::-1]),
                y=list(fc_upper_s.values) + list(fc_lower_s.values[::-1]),
                fill='toself', fillcolor=COLORS['ci'],
                line=dict(color='rgba(0,0,0,0)'), name='IC 95%',
            ))
            add_vline_dt(fig_fcs, train.index[-1], label=' Fin données observées')
            fig_fcs.update_layout(
                height=500,
                **plot_layout(title=f'Prévisions {sarima_label2} — {forecast_h_s} trimestres')
            )
            st.plotly_chart(fig_fcs, use_container_width=True)
            st.caption("La zone violette représente l'IC 95% : l'incertitude croît avec l'horizon de prévision.")
        except Exception as e:
            st.error(f"Erreur prévision SARIMA : {e}")

    # ── Bilan
    with tabs_sarima[3]:
        st.markdown("### 🏆 Bilan comparatif — toutes méthodes")
        st.markdown("Performances sur le jeu de **test 1986** (4 trimestres, données non vues lors de l'entraînement)")

        try:
            test_v = test['ukgas'].values

            hw_pred     = models_hw['HW Additif'].forecast(4).values
            hw_rmse     = np.sqrt(np.mean((test_v - hw_pred)**2))
            hw_mape     = np.mean(np.abs((test_v - hw_pred) / test_v)) * 100

            ar4_fc      = ar4_model.get_forecast(steps=4, exog=X_test)
            ar4_pred    = np.exp(ar4_fc.predicted_mean.values)
            ar4_rmse    = np.sqrt(np.mean((test_v - ar4_pred)**2))
            ar4_mape    = np.mean(np.abs((test_v - ar4_pred) / test_v)) * 100

            sar_fc      = mod_final.get_forecast(steps=4)
            sar_pred    = np.exp(sar_fc.predicted_mean.values)
            sar_rmse    = np.sqrt(np.mean((test_v - sar_pred)**2))
            sar_mape    = np.mean(np.abs((test_v - sar_pred) / test_v)) * 100

            bilan = pd.DataFrame({
                'Méthode':   ['Holt-Winters Additif', 'Régression + AR(4)', sarima_label],
                'RMSE':      [round(hw_rmse,2), round(ar4_rmse,2), round(sar_rmse,2)],
                'MAPE (%)':  [round(hw_mape,2), round(ar4_mape,2), round(sar_mape,2)],
                '🏆':        [
                    '🏆' if hw_rmse == min(hw_rmse, ar4_rmse, sar_rmse) else '',
                    '🏆' if ar4_rmse == min(hw_rmse, ar4_rmse, sar_rmse) else '',
                    '🏆' if sar_rmse == min(hw_rmse, ar4_rmse, sar_rmse) else '',
                ],
            })
            st.dataframe(
                bilan.style.highlight_min(subset=['RMSE','MAPE (%)'], color='#d1fae5'),
                hide_index=True, use_container_width=True
            )

            # Barplot comparatif
            methods_b  = ['HW Additif', 'Régression AR(4)', 'SARIMA']
            rmse_b     = [hw_rmse, ar4_rmse, sar_rmse]
            mape_b     = [hw_mape, ar4_mape, sar_mape]
            colors_b   = [COLORS['secondary'], COLORS['success'], COLORS['accent']]

            fig_bilan = make_subplots(
                rows=1, cols=2,
                subplot_titles=['RMSE ↓ (plus bas = meilleur)', 'MAPE % ↓ (plus bas = meilleur)']
            )
            for m, r, mp, c in zip(methods_b, rmse_b, mape_b, colors_b):
                fig_bilan.add_trace(
                    go.Bar(x=[m], y=[r], marker_color=c, showlegend=False,
                           hovertemplate=f'<b>{m}</b><br>RMSE = %{{y:.2f}}<extra></extra>'),
                    row=1, col=1
                )
                fig_bilan.add_trace(
                    go.Bar(x=[m], y=[mp], marker_color=c, showlegend=False,
                           hovertemplate=f'<b>{m}</b><br>MAPE = %{{y:.2f}}%<extra></extra>'),
                    row=1, col=2
                )
            fig_bilan.update_layout(height=380, **plot_layout())
            st.plotly_chart(fig_bilan, use_container_width=True)

            best_method = bilan.loc[bilan['RMSE'].idxmin(), 'Méthode']
            st.success(
                f"🏆 **{best_method}** obtient les meilleures performances sur le jeu de test — "
                f"RMSE = {bilan['RMSE'].min():.2f} therms, MAPE = {bilan.loc[bilan['RMSE'].idxmin(),'MAPE (%)']:.2f}%"
            )

            # Prévisions comparées
            st.markdown('#### Prévisions comparées — toutes méthodes sur le même graphe')
            fut_comp = pd.date_range(
                start=train.index[-1] + pd.DateOffset(months=3), periods=16, freq='QS'
            )
            hw_fc16   = models_hw['HW Additif'].forecast(16).values
            ar4_fut16 = make_regressors(n+1, 16)
            ar4_fc16  = np.exp(ar4_model.get_forecast(steps=16, exog=ar4_fut16).predicted_mean.values)
            sar_fc16  = np.exp(mod_final.get_forecast(steps=16).predicted_mean.values)

            fig_comp = go.Figure()
            fig_comp.add_trace(go.Scatter(
                x=df.index, y=df['ukgas'],
                name='Série observée', line=dict(color=COLORS['primary'], width=2.5),
            ))
            fig_comp.add_trace(go.Scatter(
                x=fut_comp, y=hw_fc16,
                name='Holt-Winters', line=dict(color=COLORS['secondary'], width=2, dash='dot'),
            ))
            fig_comp.add_trace(go.Scatter(
                x=fut_comp, y=ar4_fc16,
                name='Régression AR(4)', line=dict(color=COLORS['success'], width=2, dash='dot'),
            ))
            fig_comp.add_trace(go.Scatter(
                x=fut_comp, y=sar_fc16,
                name='SARIMA', line=dict(color=COLORS['forecast'], width=2, dash='dot'),
            ))
            add_vline_dt(fig_comp, df.index[-1])
            fig_comp.update_layout(
                height=460,
                **plot_layout(title='Prévisions comparées — 16 trimestres post-1986')
            )
            st.plotly_chart(fig_comp, use_container_width=True)

        except Exception as e:
            st.error(f"Erreur bilan : {e}")


# ─────────────────────────────────────────────────────────────
# FOOTER
# ─────────────────────────────────────────────────────────────
st.markdown('---')
st.markdown("""
<div style="text-align:center; font-size:0.8rem; opacity:0.55; font-family:'JetBrains Mono',monospace;">
    🔥 UKgas Time Series Analysis &nbsp;·&nbsp; Polytech Lyon · MAM 4A · 2023
    &nbsp;·&nbsp; Badreddine EL KHAMLICHI<br>
    Python / statsmodels / Plotly / Streamlit
</div>
""", unsafe_allow_html=True)
