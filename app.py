# -*- coding: utf-8 -*-
"""
app.py - Interface Optimisée (Design "Salle de Contrôle")
Application Streamlit — Détection de transactions bancaires suspectes / frauduleuses.

Note : le thème sombre global (couleurs de base, widgets natifs) est défini dans
.streamlit/config.toml — placez ce dossier à côté de ce fichier pour que le thème
s'applique. Le CSS ci-dessous ajoute les polices, cartes, badges et animations
que le thème natif ne peut pas gérer seul.
"""
import streamlit as st
import pandas as pd
import numpy as np
import joblib
import os
import time
import plotly.express as px
import plotly.graph_objects as go

# Configuration de base du répertoire
BASE_DIR = os.path.dirname(os.path.abspath(__file__))
MODEL_PATH = os.path.join(BASE_DIR, 'model', 'fraud_model.pkl')
DATA_PATH = os.path.join(BASE_DIR, 'data', 'Bank_transaction_scenario1.csv')

# 1. Configuration de la page
st.set_page_config(
    page_title="RiskGuard | Détection de Fraude",
    page_icon="🛡️",
    layout="wide",
    initial_sidebar_state="expanded"
)

# =========================================================
# DESIGN TOKENS
# =========================================================
COLOR_MAP = {"Normal": "#106747", "Suspect": "#FBBF24", "Fraude": "#F45555"}
ACCENT = "#22D3EE"
BG_DEEP = "#0A0F1E"
BG_SURFACE = "#121A2E"
BG_SURFACE_2 = "#1B2540"
TEXT_MUTED = "#8B96AD"

VERDICT_CONFIG = {
    "Fraude": {
        "icon": "🚨", "color": COLOR_MAP["Fraude"],
        "title": "ALERTE CRITIQUE — TRANSACTION REJETÉE",
        "desc": "Cette opération présente toutes les caractéristiques d'une fraude active. Blocage immédiat recommandé.",
        "pulse": True,
    },
    "Suspect": {
        "icon": "⚠️", "color": COLOR_MAP["Suspect"],
        "title": "VIGILANCE REQUISE — TRANSACTION SUSPECTE",
        "desc": "Comportement inhabituel détecté. Une double authentification ou une validation manuelle est nécessaire.",
        "pulse": False,
    },
    "Normal": {
        "icon": "✅", "color": COLOR_MAP["Normal"],
        "title": "TRANSACTION AUTORISÉE",
        "desc": "Aucun indicateur de risque anormal détecté. L'opération suit le protocole standard.",
        "pulse": False,
    },
}

# =========================================================
# CSS PERSONNALISÉ
# =========================================================
def inject_custom_css():
    st.markdown(f"""
    <style>
    @import url('https://fonts.googleapis.com/css2?family=Space+Grotesk:wght@500;600;700&family=Inter:wght@400;500;600&family=JetBrains+Mono:wght@500;600&display=swap');

    html, body, [class*="css"] {{
        font-family: 'Inter', sans-serif;
    }}
    h1, h2, h3, .hero-title {{
        font-family: 'Space Grotesk', sans-serif !important;
        letter-spacing: -0.01em;
    }}
    .mono {{ font-family: 'JetBrains Mono', monospace; }}

    /* ---- Hero header ---- */
    .hero-wrap {{
        display: flex; align-items: center; justify-content: space-between;
        padding: 1.4rem 1.6rem; border-radius: 18px; margin-bottom: 1.2rem;
        background: linear-gradient(120deg, {BG_SURFACE} 0%, {BG_SURFACE_2} 100%);
        border: 1px solid rgba(255,255,255,0.06);
    }}
    .hero-title {{
        font-size: 1.9rem; font-weight: 700; margin: 0;
        background: linear-gradient(90deg, #FFFFFF 0%, {ACCENT} 100%);
        -webkit-background-clip: text; background-clip: text; color: transparent;
    }}
    .hero-sub {{ color: {TEXT_MUTED}; margin-top: 0.2rem; font-size: 0.95rem; }}

    /* ---- Status pill (sidebar + hero) ---- */
    .status-pill {{
        display: inline-flex; align-items: center; gap: 0.5rem;
        padding: 0.4rem 0.9rem; border-radius: 999px; font-size: 0.82rem;
        font-family: 'JetBrains Mono', monospace; font-weight: 500;
    }}
    .status-on {{ background: rgba(52,211,153,0.12); color: {COLOR_MAP['Normal']}; border: 1px solid rgba(52,211,153,0.3); }}
    .status-off {{ background: rgba(248,113,113,0.12); color: {COLOR_MAP['Fraude']}; border: 1px solid rgba(248,113,113,0.3); }}
    .pulse-dot {{
        width: 8px; height: 8px; border-radius: 50%; background: currentColor;
        box-shadow: 0 0 0 0 currentColor; animation: dotpulse 1.8s infinite;
    }}
    @keyframes dotpulse {{
        0% {{ box-shadow: 0 0 0 0 rgba(52,211,153,0.5); }}
        70% {{ box-shadow: 0 0 0 8px rgba(52,211,153,0); }}
        100% {{ box-shadow: 0 0 0 0 rgba(52,211,153,0); }}
    }}

    /* ---- Sidebar logo ---- */
    .logo-badge {{
        width: 46px; height: 46px; border-radius: 12px; display: flex;
        align-items: center; justify-content: center; font-size: 1.4rem;
        background: linear-gradient(135deg, {ACCENT}22, {ACCENT}05);
        border: 1px solid {ACCENT}44; margin-bottom: 0.6rem;
    }}

    /* ---- Metric cards (dashboard KPI) ---- */
    .metric-card {{
        background: {BG_SURFACE}; border: 1px solid rgba(255,255,255,0.07);
        border-radius: 16px; padding: 1.1rem 1.2rem; transition: all 0.2s ease;
    }}
    .metric-card:hover {{ border-color: {ACCENT}55; transform: translateY(-2px); }}
    .metric-icon {{ font-size: 1.3rem; margin-bottom: 0.3rem; }}
    .metric-value {{ font-family: 'JetBrains Mono', monospace; font-size: 1.7rem; font-weight: 600; line-height: 1.1; }}
    .metric-label {{ color: {TEXT_MUTED}; font-size: 0.82rem; margin-top: 0.25rem; }}
    .metric-sub {{ font-size: 0.75rem; color: {TEXT_MUTED}; margin-top: 0.3rem; }}

    /* ---- Verdict card ---- */
    .verdict-card {{
        display: flex; gap: 1rem; align-items: flex-start;
        padding: 1.1rem 1.3rem; border-radius: 16px; margin: 0.6rem 0 1rem 0;
        background: color-mix(in srgb, var(--verdict-color) 10%, {BG_SURFACE});
        border-left: 4px solid var(--verdict-color);
        animation: fadein 0.4s ease;
    }}
    .verdict-icon {{ font-size: 1.6rem; }}
    .verdict-title {{ font-weight: 600; font-size: 1.02rem; color: var(--verdict-color); }}
    .verdict-desc {{ color: {TEXT_MUTED}; font-size: 0.9rem; margin-top: 0.2rem; }}
    .pulse-ring {{ animation: fadein 0.4s ease, ringpulse 1.6s infinite; }}
    @keyframes ringpulse {{
        0% {{ box-shadow: 0 0 0 0 color-mix(in srgb, var(--verdict-color) 45%, transparent); }}
        70% {{ box-shadow: 0 0 0 14px color-mix(in srgb, var(--verdict-color) 0%, transparent); }}
        100% {{ box-shadow: 0 0 0 0 color-mix(in srgb, var(--verdict-color) 0%, transparent); }}
    }}
    @keyframes fadein {{
        from {{ opacity: 0; transform: translateY(6px); }}
        to {{ opacity: 1; transform: translateY(0); }}
    }}

    /* ---- Buttons ---- */
    .stButton > button {{
        border-radius: 12px !important; font-weight: 600 !important;
        transition: all 0.15s ease !important; border: none !important;
    }}
    .stButton > button[kind="primary"] {{
        background: linear-gradient(90deg, {ACCENT}, #38BDF8) !important; color: #06121C !important;
    }}
    .stButton > button[kind="primary"]:hover {{
        transform: translateY(-1px); box-shadow: 0 6px 18px {ACCENT}44 !important;
    }}
    div[data-testid="stDownloadButton"] > button {{
        border-radius: 12px !important; border: 1px solid {ACCENT}55 !important;
    }}

    /* ---- Section divider ---- */
    .section-eyebrow {{
        font-family: 'JetBrains Mono', monospace; font-size: 0.75rem; letter-spacing: 0.08em;
        color: {ACCENT}; text-transform: uppercase; margin-bottom: 0.2rem;
    }}
    </style>
    """, unsafe_allow_html=True)


def metric_card(icon, label, value, sub=None, color=ACCENT):
    st.markdown(f"""
    <div class="metric-card">
        <div class="metric-icon" style="color:{color};">{icon}</div>
        <div class="metric-value" style="color:{color};">{value}</div>
        <div class="metric-label">{label}</div>
        {f'<div class="metric-sub">{sub}</div>' if sub else ''}
    </div>
    """, unsafe_allow_html=True)


def render_verdict_badge(label):
    cfg = VERDICT_CONFIG.get(label, VERDICT_CONFIG["Normal"])
    pulse_class = "pulse-ring" if cfg["pulse"] else ""
    st.markdown(f"""
    <div class="verdict-card {pulse_class}" style="--verdict-color:{cfg['color']};">
        <div class="verdict-icon">{cfg['icon']}</div>
        <div>
            <div class="verdict-title">{cfg['title']}</div>
            <div class="verdict-desc">{cfg['desc']}</div>
        </div>
    </div>
    """, unsafe_allow_html=True)


def hex_to_rgba(hex_color, alpha=1.0):
    """Convertit une couleur hex '#RRGGBB' en chaîne 'rgba(r,g,b,a)' compatible Plotly."""
    hex_color = hex_color.lstrip('#')
    r, g, b = (int(hex_color[i:i + 2], 16) for i in (0, 2, 4))
    return f"rgba({r},{g},{b},{alpha})"


def render_gauge(label, value):
    color = COLOR_MAP.get(label, ACCENT)
    fig = go.Figure(go.Indicator(
        mode="gauge+number",
        value=value * 100,
        number={'suffix': "%", 'font': {'family': 'JetBrains Mono', 'size': 34, 'color': color}},
        title={'text': f"Confiance — {label}", 'font': {'family': 'Inter', 'size': 14, 'color': TEXT_MUTED}},
        gauge={
            'axis': {'range': [0, 100], 'tickcolor': TEXT_MUTED, 'tickfont': {'size': 9}},
            'bar': {'color': color, 'thickness': 0.28},
            'bgcolor': BG_SURFACE_2,
            'borderwidth': 0,
            'steps': [
                {'range': [0, 50], 'color': BG_SURFACE},
                {'range': [50, 80], 'color': BG_SURFACE_2},
                {'range': [80, 100], 'color': hex_to_rgba(color, 0.13)},
            ],
        }
    ))
    fig.update_layout(
        height=230, margin=dict(l=20, r=20, t=45, b=10),
        paper_bgcolor='rgba(0,0,0,0)', font={'color': '#E8EDF7'}
    )
    return fig


inject_custom_css()

# Chargement du modèle et des données avec mise en cache
@st.cache_resource
def load_model():
    return joblib.load(MODEL_PATH)

@st.cache_data
def load_data():
    df = pd.read_csv(DATA_PATH, sep=';')
    df['Localisation'] = df['Localisation'].replace({
        'Saint-Louis': 'Saint Louis',
        'RunÃ©': 'Rufisque'
    })
    return df

def predict_transaction(artifact, montant, type_transac, status, localisation, date, heure):
    pipe = artifact['pipeline']
    le = artifact['label_encoder']

    jour_semaine = date.weekday()
    weekend = int(jour_semaine in [5, 6])
    X_new = pd.DataFrame([{
        'Montant': montant,
        'heure': heure,
        'jour_semaine': jour_semaine,
        'weekend': weekend,
        'Type de transaction': type_transac,
        'Status operation': status,
        'Localisation': localisation,
    }])

    pred = pipe.predict(X_new)[0]
    proba = pipe.predict_proba(X_new)[0]
    label = le.inverse_transform([pred])[0]
    proba_dict = dict(zip(le.classes_, proba))
    return label, proba_dict

# Vérification de la présence du modèle
try:
    artifact = load_model()
    model_ready = True
except FileNotFoundError:
    model_ready = False

# =========================================================
# BARRE LATÉRALE (SIDEBAR)
# =========================================================
with st.sidebar:
    st.markdown('<div class="logo-badge">🛡️</div>', unsafe_allow_html=True)
    st.markdown("### RiskGuard ML")
    st.caption("Système de surveillance des transactions en temps réel.")
    st.write("")

    if model_ready:
        st.markdown(
            '<div class="status-pill status-on"><span class="pulse-dot"></span>MODÈLE ACTIF · RANDOM FOREST</div>',
            unsafe_allow_html=True
        )
    else:
        st.markdown(
            '<div class="status-pill status-off"><span class="pulse-dot"></span>MODÈLE INTROUVABLE</div>',
            unsafe_allow_html=True
        )

    if model_ready:
        try:
            _df_sidebar = load_data()
            st.write("")
            st.markdown('<div class="section-eyebrow">Aperçu du jeu de données</div>', unsafe_allow_html=True)
            st.markdown(f'<span class="mono" style="color:{ACCENT}; font-size:1.3rem;">{len(_df_sidebar):,}</span> transactions référencées', unsafe_allow_html=True)
        except Exception:
            pass

    st.divider()
    st.caption("Version 2.1 (Interface Interactive) • © 2026")

# =========================================================
# CORPS PRINCIPAL — HERO
# =========================================================
status_html = (
    '<div class="status-pill status-on"><span class="pulse-dot"></span>OPÉRATIONNEL</div>'
    if model_ready else
    '<div class="status-pill status-off"><span class="pulse-dot"></span>HORS LIGNE</div>'
)
st.markdown(f"""
<div class="hero-wrap">
    <div>
        <p class="hero-title">🛡️ Plateforme Intégrée de Détection de Fraude</p>
        <p class="hero-sub">Analyse prédictive des risques bancaires et exploration analytique des flux.</p>
    </div>
    {status_html}
</div>
""", unsafe_allow_html=True)

# Navigation par Onglets
tab1, tab2 = st.tabs(["🔍 Analyse en Temps Réel", "📊 Tableau de Bord Décisionnel"])

# ---------------------------------------------------------
# ONGLET 1 : Prédiction sur une nouvelle transaction
# ---------------------------------------------------------
with tab1:
    if not model_ready:
        st.error("⚠️ Impossible de charger l'interface d'analyse. Le fichier `model/fraud_model.pkl` est manquant.")
    else:
        df = load_data()
        localisations = sorted(df['Localisation'].unique())
        types_transac = sorted(df['Type de transaction'].unique())
        statuts = sorted(df['Status operation'].unique())

        st.markdown('<div class="section-eyebrow">Étape 1</div>', unsafe_allow_html=True)
        st.subheader("📋 Formulaire d'évaluation du risque")

        with st.container(border=True):
            col1, col2 = st.columns(2)
            with col1:
                montant = st.number_input("Montant de l'opération (FCFA)", min_value=0.0, value=50000.0, step=1000.0)
                type_transac = st.selectbox("Canal / Type de transaction", types_transac)
                status = st.selectbox("Statut technique de l'opération", statuts)

            with col2:
                localisation = st.selectbox("Zone géographique / Localisation", localisations)
                date = st.date_input("Date d'exécution")
                heure = st.slider("Heure de la transaction (Format 24h)", 0, 23, 12)

            st.write("")
            submit_button = st.button("🚀 Lancer l'audit de sécurité", type="primary", use_container_width=True)

        if submit_button:
            with st.spinner("🔎 Analyse du profil de risque en cours..."):
                label, proba = predict_transaction(artifact, montant, type_transac, status, localisation, date, heure)
                time.sleep(0.35)  # laisse le temps de percevoir le scan

            toast_icons = {"Fraude": "🚨", "Suspect": "⚠️", "Normal": "✅"}
            st.toast(f"Analyse terminée — verdict : {label}", icon=toast_icons.get(label, "ℹ️"))

            st.write("")
            st.markdown('<div class="section-eyebrow">Étape 2 · Résultat</div>', unsafe_allow_html=True)
            st.subheader("🎯 Diagnostic du Système")

            render_verdict_badge(label)

            col_gauge, col_bar = st.columns([1, 1.4])
            with col_gauge:
                clean_label = label[0] if isinstance(label, (list, np.ndarray)) else str(label)
                st.plotly_chart(render_gauge(clean_label, proba.get(clean_label, 0.0)), width='stretch')

            with col_bar:
                proba_df = pd.DataFrame({
                    'Niveau de Risque': list(proba.keys()),
                    'Indice de Confiance': list(proba.values())
                }).sort_values('Indice de Confiance', ascending=False)

                fig = px.bar(
                    proba_df, x='Indice de Confiance', y='Niveau de Risque', orientation='h',
                    color='Niveau de Risque', color_discrete_map=COLOR_MAP, text_auto='.1%',
                    title="Décomposition du score de probabilité par le modèle"
                )
                fig.update_layout(
                    xaxis_tickformat='.0%', showlegend=False, height=230,
                    margin=dict(l=20, r=20, t=40, b=20),
                    paper_bgcolor='rgba(0,0,0,0)', plot_bgcolor='rgba(0,0,0,0)',
                    font={'color': '#E8EDF7'}
                )
                fig.update_xaxes(gridcolor='rgba(255,255,255,0.06)')
                st.plotly_chart(fig, width='stretch')

            # =========================================================
            # EXPORTATION CSV
            # =========================================================
            st.write("")
            st.markdown('<div class="section-eyebrow">Étape 3</div>', unsafe_allow_html=True)
            st.markdown("**💾 Exportation des résultats**")

            export_df = pd.DataFrame([{
                'Date': date.strftime('%Y-%m-%d'),
                'Heure': f"{heure}:00",
                'Montant_FCFA': montant,
                'Type_Transaction': type_transac,
                'Statut_Operation': status,
                'Localisation': localisation,
                'Statut_Predi_IA': clean_label,
                'Probabilite_Normal': f"{proba.get('Normal', 0.0) * 100:.2f}%" if 'Normal' in proba else "0.00%",
                'Probabilite_Suspect': f"{proba.get('Suspect', 0.0) * 100:.2f}%" if 'Suspect' in proba else "0.00%",
                'Probabilite_Fraude': f"{proba.get('Fraude', 0.0) * 100:.2f}%" if 'Fraude' in proba else "0.00%"
            }])

            csv_data = export_df.to_csv(index=False, sep=';').encode('utf-8')

            st.download_button(
                label="📥 Télécharger le rapport de cette prédiction (CSV)",
                data=csv_data,
                file_name=f"prediction_fraude_{date.strftime('%Y%m%d')}_{heure}h.csv",
                mime="text/csv",
                width='stretch'
            )

# ---------------------------------------------------------
# ONGLET 2 : Tableau de bord / exploration des données
# ---------------------------------------------------------
with tab2:
    df = load_data()
    target_col = 'Target'

    # ---- Filtres interactifs ----
    st.markdown('<div class="section-eyebrow">Filtres</div>', unsafe_allow_html=True)
    col_f1, col_f2, col_f3 = st.columns([2, 2, 1])
    with col_f1:
        sel_loc = st.multiselect("📍 Localisation", options=sorted(df['Localisation'].unique()), default=[])
    with col_f2:
        sel_type = st.multiselect("💳 Type de transaction", options=sorted(df['Type de transaction'].unique()), default=[])
    with col_f3:
        st.write("")
        st.write("")
        reset = st.button("↺ Réinitialiser", use_container_width=True)
        if reset:
            sel_loc, sel_type = [], []

    df_view = df.copy()
    if sel_loc:
        df_view = df_view[df_view['Localisation'].isin(sel_loc)]
    if sel_type:
        df_view = df_view[df_view['Type de transaction'].isin(sel_type)]

    if df_view.empty:
        st.warning("Aucune transaction ne correspond aux filtres sélectionnés.")
        df_view = df.copy()

    # ---- Téléchargement des données ----
    col_dl_info, col_dl_btn = st.columns([3, 1])
    with col_dl_info:
        st.caption(f"📄 {len(df_view):,} transaction(s) dans la vue actuelle sur {len(df):,} au total.")
    with col_dl_btn:
        st.download_button(
            label="⬇️ Exporter cette vue (CSV)",
            data=df_view.to_csv(index=False, sep=';').encode('utf-8'),
            file_name="transactions_filtrees.csv",
            mime="text/csv",
            use_container_width=True
        )

    st.write("")
    st.markdown('<div class="section-eyebrow">Indicateurs clés</div>', unsafe_allow_html=True)
    st.subheader("📈 Indicateurs Clés de Performance (KPI)")

    col1, col2, col3 = st.columns(3)
    with col1:
        metric_card("📦", "Flux Analysé (filtré)", f"{len(df_view):,}", sub=f"sur {len(df):,} au total", color=ACCENT)
    with col2:
        taux_fraude = (df_view[target_col] == 'Fraude').mean() * 100 if len(df_view) else 0
        metric_card("🚨", "Taux d'Incidents (Fraude)", f"{taux_fraude:.2f}%", sub="part des transactions filtrées", color=COLOR_MAP["Fraude"])
    with col3:
        taux_suspect = (df_view[target_col] == 'Suspect').mean() * 100 if len(df_view) else 0
        metric_card("⚠️", "Alertes en Attente (Suspect)", f"{taux_suspect:.2f}%", sub="part des transactions filtrées", color=COLOR_MAP["Suspect"])

    st.write("")

    col_chart1, col_chart2 = st.columns(2)

    with col_chart1:
        with st.container(border=True):
            st.markdown("**Structure globale du portefeuille de risques**")
            fig1 = px.pie(
                df_view, names=target_col, color=target_col, color_discrete_map=COLOR_MAP,
                hole=0.55, height=300
            )
            fig1.update_traces(textfont_color='#E8EDF7', marker=dict(line=dict(color=BG_DEEP, width=2)))
            fig1.update_layout(
                margin=dict(l=10, r=10, t=10, b=10),
                paper_bgcolor='rgba(0,0,0,0)', legend={'font': {'color': '#E8EDF7'}}
            )
            st.plotly_chart(fig1, width='stretch')

    with col_chart2:
        with st.container(border=True):
            st.markdown("**Distribution de la sévérité des montants (log)**")
            df_plot = df_view.copy()
            df_plot['Log-Montant (FCFA)'] = np.log1p(df_plot['Montant'])
            fig4 = px.box(
                df_plot, x=target_col, y='Log-Montant (FCFA)', color=target_col,
                color_discrete_map=COLOR_MAP, height=300
            )
            fig4.update_layout(
                showlegend=False, margin=dict(l=10, r=10, t=10, b=10),
                paper_bgcolor='rgba(0,0,0,0)', plot_bgcolor='rgba(0,0,0,0)',
                font={'color': '#E8EDF7'}
            )
            fig4.update_xaxes(gridcolor='rgba(255,255,255,0.06)')
            fig4.update_yaxes(gridcolor='rgba(255,255,255,0.06)')
            st.plotly_chart(fig4, width='stretch')

    with st.container(border=True):
        st.markdown("**Vulnérabilité par typologie et statut technique des opérations**")
        col_hist1, col_hist2 = st.columns(2)
        with col_hist1:
            fig2 = px.histogram(
                df_view, x='Type de transaction', color=target_col, barnorm='percent',
                color_discrete_map=COLOR_MAP, title="Impact par Canal", height=280
            )
            fig2.update_layout(
                paper_bgcolor='rgba(0,0,0,0)', plot_bgcolor='rgba(0,0,0,0)',
                font={'color': '#E8EDF7'}, legend={'font': {'color': '#E8EDF7'}}
            )
            fig2.update_xaxes(gridcolor='rgba(255,255,255,0.06)')
            fig2.update_yaxes(gridcolor='rgba(255,255,255,0.06)')
            st.plotly_chart(fig2, width='stretch')
        with col_hist2:
            fig3 = px.histogram(
                df_view, x='Status operation', color=target_col, barnorm='percent',
                color_discrete_map=COLOR_MAP, title="Impact par Statut Réseau", height=280
            )
            fig3.update_layout(
                paper_bgcolor='rgba(0,0,0,0)', plot_bgcolor='rgba(0,0,0,0)',
                font={'color': '#E8EDF7'}, legend={'font': {'color': '#E8EDF7'}}
            )
            fig3.update_xaxes(gridcolor='rgba(255,255,255,0.06)')
            fig3.update_yaxes(gridcolor='rgba(255,255,255,0.06)')
            st.plotly_chart(fig3, width='stretch')

    with st.expander("🗂️ Accéder au registre complet des transactions auditées"):
        st.dataframe(df_view.head(100), width='stretch')
        st.caption("Aperçu limité aux 100 premières lignes de la vue filtrée. Utilisez les boutons ci-dessous pour l'export complet.")

        col_full1, col_full2 = st.columns(2)
        with col_full1:
            st.download_button(
                label="⬇️ Télécharger la vue filtrée complète (CSV)",
                data=df_view.to_csv(index=False, sep=';').encode('utf-8'),
                file_name="transactions_filtrees_complet.csv",
                mime="text/csv",
                use_container_width=True
            )
        with col_full2:
            st.download_button(
                label="⬇️ Télécharger le jeu de données brut complet (CSV)",
                data=df.to_csv(index=False, sep=';').encode('utf-8'),
                file_name="transactions_toutes.csv",
                mime="text/csv",
                use_container_width=True
            )
